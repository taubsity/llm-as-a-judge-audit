import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: list[dict], path: Path, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_annotation_value(value: str) -> int:
    if isinstance(value, int):
        return value
    value = str(value).strip().lower()
    if value in ("yes", "ja", "1", "true"):
        return 1
    elif value in ("no", "nein", "0", "false"):
        return 0
    else:
        try:
            return int(value)
        except ValueError:
            return 0


def extract_labels(item_data: dict, question_mapping: dict) -> dict:
    labels = {}
    for q_key, label_name in question_mapping.items():
        matching_keys = [k for k in item_data.keys() if k.startswith(f"{q_key}_")]
        if matching_keys:
            labels[label_name] = parse_annotation_value(item_data[matching_keys[0]])

    for key in question_mapping.values():
        if key not in labels:
            labels[key] = 0

    return labels


def load_annotations(
    results_dir: Path, prefix: str, question_mapping: dict
) -> dict[str, list[dict]]:
    annotations = {}

    annotation_files = list(results_dir.glob(f"{prefix}*.json"))
    if not annotation_files:
        print(f"Warning: No annotation files matching {prefix}*.json in {results_dir}")
        return annotations

    print(f"Found {len(annotation_files)} annotation files matching {prefix}*.json")

    for file_path in annotation_files:
        print(f"  Loading annotations from rater: {file_path.stem.replace(prefix, '')}")

        with open(file_path, "r", encoding="utf-8") as f:
            rater_data = json.load(f)

        for item_idx, item_data in rater_data.items():
            annotations.setdefault(item_idx, []).append(
                extract_labels(item_data, question_mapping)
            )

    return annotations


def get_severity(error_code: str) -> str:
    parts = error_code.split(".")
    return parts[-1] if parts else ""
