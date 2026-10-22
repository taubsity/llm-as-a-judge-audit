import argparse
import csv
import json
from pathlib import Path

from .common import load_annotations, write_jsonl

REPORTS_QUESTION_MAPPING = {
    "Q0": "accurate",
    "Q1": "thorough",
    "Q2": "useful",
    "Q3": "organized",
    "Q4": "comprehensible",
    "Q5": "succinct",
    "Q6": "clinical_utility",
}

EPIKRISE_RESPONSE_VALUES = {
    "linke Epikrise": "left",
    "rechte Epikrise": "right",
    "beide gleich": "equal",
}

TASK_MAPPING = {
    "op": "surgery",
    "radio": "radiology",
    "patho": "pathology",
}


def map_epikrise_response_to_preference(response: str, randomized_left: str) -> str:
    response_side = EPIKRISE_RESPONSE_VALUES.get(response, None)

    if response_side == "equal":
        return "TIE"
    elif response_side == "left":
        if randomized_left == "epikrise_gt":
            return "A"
        else:
            return "B"
    elif response_side == "right":
        if randomized_left == "epikrise_gt":
            return "B"
        else:
            return "A"
    else:
        return "TIE"


def load_epikrise_annotations(
    results_dir: Path, randomization_csv: Path
) -> dict[str, list[dict]]:
    import pandas as pd

    annotations = {}

    randomization_df = pd.read_csv(randomization_csv)

    pattern = "responses_epikrise_*.json"
    annotation_files = list(results_dir.glob(pattern))

    if not annotation_files:
        print(f"Warning: No epikrise annotation files found in {results_dir}")
        return annotations

    print(f"Found {len(annotation_files)} epikrise annotation files")

    output_label_mapping = {
        "Q0": "human_written",
        "Q1": "completeness",
        "Q2": "efficiency",
        "Q3": "organized",
        "Q4": "clinical_utility",
    }

    for file_path in annotation_files:
        pseudonym = file_path.stem.replace("responses_epikrise_", "")
        print(f"  Loading annotations from rater: {pseudonym}")

        with open(file_path, "r", encoding="utf-8") as f:
            rater_data = json.load(f)

        for item_idx, item_data in rater_data.items():
            try:
                sample_idx = int(item_idx)
                if sample_idx >= len(randomization_df):
                    continue

                randomized_left = randomization_df.iloc[sample_idx]["randomized_left"]
            except (ValueError, KeyError):
                continue

            labels = {}
            for q_key, label_name in output_label_mapping.items():
                matching_keys = [
                    k for k in item_data.keys() if k.startswith(f"{q_key}_")
                ]
                if matching_keys:
                    response = item_data[matching_keys[0]]
                    labels[label_name] = map_epikrise_response_to_preference(
                        response, randomized_left
                    )

            if labels:
                if "clinical_utility" in labels:
                    labels["overall"] = labels["clinical_utility"]

                if item_idx not in annotations:
                    annotations[item_idx] = []
                annotations[item_idx].append(labels)

    return annotations


def process_reports(
    csv_path: Path,
    annotations: dict[str, list[dict]],
    category: str,
    id_prefix: str = "reports",
) -> list[dict]:
    records = []
    task = TASK_MAPPING.get(category, category)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            idx = str(row_idx)

            human_labels = annotations.get(idx, [])

            if not human_labels:
                continue

            record = {
                "id": f"{id_prefix}_{row_idx + 1:03d}",
                "dataset": "reports",
                "task": task,
                "notes": row.get(
                    "text",
                    row.get("input", row.get("notes", row.get("source_text", ""))),
                ),
                "summary": row.get(
                    "summary_few_shot",
                    row.get(
                        "output", row.get("summary", row.get("generated_summary", ""))
                    ),
                ),
                "specialty": row.get("specialty", task),
                "human_labels": human_labels,
            }

            records.append(record)

    return records


def process_epikrise(
    csv_path: Path, annotations: dict[str, list[dict]], id_prefix: str = "epikrise"
) -> list[dict]:
    records = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            idx = str(row_idx)

            human_votes = annotations.get(idx, [])

            if not human_votes:
                continue

            randomized_left = row.get("randomized_left", "epikrise_gt")

            if randomized_left == "epikrise_gt":
                text_a = row.get("epikrise_gt", row.get("text_a", ""))
                text_b = row.get("epikrise_generated", row.get("text_b", ""))
            else:
                text_a = row.get("epikrise_generated", row.get("text_a", ""))
                text_b = row.get("epikrise_gt", row.get("text_b", ""))

            record = {
                "id": f"{id_prefix}_{row_idx + 1:03d}",
                "dataset": "epikrise",
                "source_text": row.get(
                    "progress_reports", row.get("source_text", "")
                ).strip(),
                "text_a": text_a,
                "text_b": text_b,
                "human_votes": human_votes,
            }

            records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Format reports and epikrise study results into JSONL format."
    )

    parser.add_argument(
        "--reports-csv",
        type=Path,
        action="append",
        default=[],
        help="Path to a CSV file for reports (can be specified multiple times)",
    )
    parser.add_argument(
        "--reports-category",
        type=str,
        action="append",
        default=[],
        help="Category for corresponding reports CSV (op, radio, patho)",
    )
    parser.add_argument(
        "--reports-results-dir",
        type=Path,
        help="Path to the directory containing report annotation JSON files",
    )
    parser.add_argument(
        "--reports-output",
        type=Path,
        default=Path("reports.jsonl"),
        help="Output JSONL file path for reports (default: reports.jsonl)",
    )

    parser.add_argument(
        "--epikrise-csv",
        type=Path,
        help="Path to the CSV file for epikrise comparison",
    )
    parser.add_argument(
        "--epikrise-results-dir",
        type=Path,
        help="Path to the directory containing epikrise annotation JSON files",
    )
    parser.add_argument(
        "--epikrise-output",
        type=Path,
        default=Path("epikrise.jsonl"),
        help="Output JSONL file path for epikrise (default: epikrise.jsonl)",
    )

    args = parser.parse_args()

    if args.reports_csv and args.reports_results_dir:
        all_report_records = []

        if len(args.reports_csv) != len(args.reports_category):
            print("Error: Number of --reports-csv must match --reports-category")
            return 1

        for csv_path, category in zip(args.reports_csv, args.reports_category):
            if not csv_path.exists():
                print(f"Error: Reports CSV not found: {csv_path}")
                return 1

            print(f"\nProcessing reports category '{category}': {csv_path}")
            annotations = load_annotations(
                args.reports_results_dir,
                f"responses_{category}_",
                REPORTS_QUESTION_MAPPING,
            )
            records = process_reports(csv_path, annotations, category)
            print(f"  Processed {len(records)} records")
            all_report_records.extend(records)

        if all_report_records:
            print(
                f"\nWriting {len(all_report_records)} report records to {args.reports_output}"
            )
            write_jsonl(all_report_records, args.reports_output)
            print("Reports done!")

    if args.epikrise_csv and args.epikrise_results_dir:
        if not args.epikrise_csv.exists():
            print(f"Error: Epikrise CSV not found: {args.epikrise_csv}")
            return 1

        print(f"\nProcessing epikrise: {args.epikrise_csv}")
        annotations = load_epikrise_annotations(
            args.epikrise_results_dir, args.epikrise_csv
        )
        records = process_epikrise(args.epikrise_csv, annotations)
        print(f"  Processed {len(records)} records")

        if records:
            print(
                f"\nWriting {len(records)} epikrise records to {args.epikrise_output}"
            )
            write_jsonl(records, args.epikrise_output)
            print("Epikrise done!")

    if not args.reports_csv and not args.epikrise_csv:
        print("Error: No input files provided. Use --reports-csv or --epikrise-csv")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
