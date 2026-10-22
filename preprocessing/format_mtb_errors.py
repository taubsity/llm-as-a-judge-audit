import argparse
import json
from pathlib import Path

from .common import get_severity, write_jsonl

MAJOR_SEVERITIES = {"ERHEB", "KRIT"}


def load_summary_lines(summary_path: Path, doc_id: str) -> list[str]:
    try:
        if not summary_path.exists():
            print(f"Warning: Summary file not found for {doc_id}: {summary_path}")
            return []
        text = summary_path.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.split("\n\n") if line.strip()]
        return lines
    except Exception as e:
        print(f"Error reading summary for {doc_id}: {e}")
        return []


def load_source_text(source_dir: Path, doc_id: str) -> str:
    base_id = doc_id.split("_")[0]
    source_file = source_dir / f"{base_id}.txt"
    try:
        if not source_file.exists():
            print(f"Warning: Source file not found for {doc_id}: {source_file}")
            return ""
        text = source_file.read_text(encoding="utf-8").strip()
        if not text:
            print(f"Warning: Empty source text for {doc_id}: {source_file}")
        return text
    except Exception as e:
        print(f"Error reading source for {doc_id}: {e}")
        return ""


def is_major_or_critical(error_code: str) -> bool:
    return get_severity(error_code) in MAJOR_SEVERITIES


def merge_annotations(annotations_by_user: list[list[dict]]) -> dict[int, dict]:
    merged: dict[int, dict] = {}
    for annotations in annotations_by_user:
        for ann in annotations:
            line_idx = ann["line_index"]
            if line_idx not in merged:
                merged[line_idx] = {"errors": set(), "comments": []}
            for error in ann.get("errors", []):
                merged[line_idx]["errors"].add(error["error_code"])
                comment = error.get("comment", "").strip()
                if comment:
                    merged[line_idx]["comments"].append(comment)
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Format MTB error annotations into JSONL."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input folder with user subfolders",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing source txt files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file with PASS/FAIL labels",
    )
    parser.add_argument(
        "--summary-path-rewrite",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help=(
            "Rewrite a prefix in summary_path entries from progress.json "
            "(repeatable, case-insensitive). Useful when annotations were "
            "produced on a different machine."
        ),
    )
    args = parser.parse_args()

    rewrites: list[tuple[str, str]] = []
    for spec in args.summary_path_rewrite:
        if "=" not in spec:
            parser.error(f"--summary-path-rewrite expects OLD=NEW, got {spec!r}")
        old, new = spec.split("=", 1)
        rewrites.append((old, new))

    doc_annotations: dict[str, list[list[dict]]] = {}
    doc_metadata: dict[str, dict] = {}

    for user_dir in sorted(args.input_dir.iterdir()):
        if not user_dir.is_dir():
            continue
        progress_file = user_dir / "progress.json"
        if not progress_file.exists():
            continue

        print(f"Processing {user_dir.name}...")

        with open(progress_file, "r", encoding="utf-8") as f:
            progress = json.load(f)

        for doc_id, doc in progress.get("document_sets", {}).items():
            line_annotations = doc.get("line_annotations")
            if not line_annotations:
                continue

            if doc_id not in doc_annotations:
                doc_annotations[doc_id] = []
                doc_metadata[doc_id] = doc.get("metadata", {})

            doc_annotations[doc_id].append(line_annotations)

    passfail_records = {}

    for doc_id, annotations_list in sorted(doc_annotations.items()):
        metadata = doc_metadata[doc_id]

        summary_path = metadata.get("summary_path", "")
        for old, new in rewrites:
            if summary_path.lower().startswith(old.lower()):
                summary_path = new + summary_path[len(old) :]
                break

        lines = load_summary_lines(Path(summary_path), doc_id)
        source_text = load_source_text(args.source_dir, doc_id)

        if not lines:
            continue

        merged_errors = merge_annotations(annotations_list)

        section_texts = []
        for i, line in enumerate(lines):
            section_texts.append(f"[Section {i}]\n{line}")
        text = "\n\n".join(section_texts)

        record_id = f"mtb_{doc_id.lower()}"

        passfail_labels = {}
        for i, section_line in enumerate(lines):
            section_data = merged_errors.get(i, {"errors": set(), "comments": []})
            errors = section_data["errors"]
            major_errors = {e for e in errors if is_major_or_critical(e)}
            comment = " ".join(section_data["comments"])
            passfail_labels[f"section_{i}"] = {
                "text": section_line,
                "comment": comment,
                "errors": "FAIL" if major_errors else "PASS",
            }

        passfail_records[record_id] = {
            "id": record_id,
            "dataset": "mtb_errors",
            "task": "error_detection",
            "paradigm": "passfail",
            "source": source_text,
            "text": text,
            "human_labels": [passfail_labels],
        }

    write_jsonl(list(passfail_records.values()), args.output)
    print(f"Wrote {len(passfail_records)} records to {args.output}")


if __name__ == "__main__":
    main()
