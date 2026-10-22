import argparse
import csv
from pathlib import Path

from .common import load_annotations, write_jsonl

QUESTION_MAPPING = {
    "Q0": "citation",
    "Q1": "accurate",
    "Q2": "thorough",
    "Q3": "useful",
    "Q4": "organized",
    "Q5": "comprehensible",
    "Q6": "succinct",
    "Q7": "abstraction",
    "Q8": "stig_original",
    "Q9": "stig_summary",
}


def process_use_case_1(
    csv_path: Path, annotations: dict[str, list[dict]]
) -> list[dict]:
    records = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader):
            idx = row_number

            human_labels = annotations.get(str(idx), [])

            if not human_labels:
                print(f"  Skipping idx {idx} due to missing human labels")
                continue

            record = {
                "id": f"nursing_{int(idx) + 1:03d}",
                "dataset": "nursing",
                "task": "summary",
                "notes": row["input"],
                "summary": row["output"],
                "specialty": "general nursing",
                "metadata": {
                    "age": row.get("AGE", ""),
                    "gender": row.get("gender", ""),
                    "duration": row.get("Dauer", ""),
                    "subject_reference": row.get("subject_reference", ""),
                    "issued": row.get("issued", ""),
                },
                "human_labels": human_labels,
            }

            records.append(record)

    return records


def process_use_case_4(
    csv_path: Path, annotations: dict[str, list[dict]]
) -> list[dict]:
    records = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row["idx"]

            human_labels = annotations.get(idx, [])

            if not human_labels:
                continue

            record = {
                "id": f"nursing_op_{int(idx) + 1:03d}",
                "dataset": "nursing",
                "task": "endpoints",
                "notes": row["input"],
                "summary": row["output"],
                "specialty": "surgical nursing",
                "metadata": {
                    "date_op": row.get("date_op", ""),
                    "age": row.get("AGE", ""),
                    "gender": row.get("gender", ""),
                },
                "human_labels": human_labels,
            }

            records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Format nursing study results into JSONL format."
    )
    parser.add_argument(
        "--use-case-1-csv",
        type=Path,
        help="Path to the CSV file for use case 1 (nursing documentation summaries)",
    )
    parser.add_argument(
        "--use-case-4-csv",
        type=Path,
        help="Path to the CSV file for use case 4 (OP report summaries)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Path to the directory containing annotation JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nursing.jsonl"),
        help="Output JSONL file path (default: nursing.jsonl)",
    )

    args = parser.parse_args()

    all_records = []

    if args.use_case_1_csv:
        if not args.use_case_1_csv.exists():
            print(f"Error: Use case 1 CSV not found: {args.use_case_1_csv}")
            return 1

        print(f"\nProcessing use case 1: {args.use_case_1_csv}")
        annotations_1 = load_annotations(
            args.results_dir, "responses_use_case_1_", QUESTION_MAPPING
        )
        records_1 = process_use_case_1(args.use_case_1_csv, annotations_1)
        print(f"  Processed {len(records_1)} records")
        all_records.extend(records_1)

    if args.use_case_4_csv:
        if not args.use_case_4_csv.exists():
            print(f"Error: Use case 4 CSV not found: {args.use_case_4_csv}")
            return 1

        print(f"\nProcessing use case 4: {args.use_case_4_csv}")
        annotations_4 = load_annotations(
            args.results_dir, "responses_use_case_4_", QUESTION_MAPPING
        )
        records_4 = process_use_case_4(args.use_case_4_csv, annotations_4)
        print(f"  Processed {len(records_4)} records")
        all_records.extend(records_4)

    if not all_records:
        print("Error: No records to process. Provide at least one CSV file.")
        return 1

    print(f"\nWriting {len(all_records)} records to {args.output}")
    write_jsonl(all_records, args.output)

    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
