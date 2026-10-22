import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .common import load_jsonl

DATASETS = {
    "reports": {
        "file": "reports.jsonl",
        "text_fields": ["notes", "summary"],
        "group_field": "specialty",
    },
    "nursing": {
        "file": "nursing.jsonl",
        "text_fields": ["notes", "summary"],
        "group_field": None,
    },
    "epikrise": {
        "file": "epikrise.jsonl",
        "text_fields": ["source_text", "text_a", "text_b"],
        "group_field": None,
    },
    "mtb_errors": {
        "file": "mtb_errors.jsonl",
        "text_fields": ["source", "text"],
        "group_field": None,
    },
}


def describe(values: list[float], label: str) -> dict:
    a = np.array(values, dtype=float)
    return {
        "label": label,
        "n": len(a),
        "mean": round(float(a.mean()), 1),
        "median": round(float(np.median(a)), 1),
        "std": round(float(a.std()), 1),
        "min": int(a.min()),
        "max": int(a.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute word and character length statistics for dataset text fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("AUDIT_DATA_DIR", "../data_audit")),
        help="Directory containing the dataset JSONL files",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/metrics"),
        help="Output directory for the resulting CSV",
    )
    args = parser.parse_args()

    field_rows: list[dict] = []

    for dataset_name, cfg in DATASETS.items():
        path = args.data_dir / cfg["file"]
        items = load_jsonl(path)
        group_field = cfg["group_field"]

        for field in cfg["text_fields"]:
            values_chars: list[float] = []
            values_words: list[float] = []
            for item in items:
                text = item.get(field, "") or ""
                values_chars.append(len(text))
                values_words.append(len(text.split()))

            for unit, vals in [("chars", values_chars), ("words", values_words)]:
                row = describe(vals, f"{dataset_name} / {field} / {unit}")
                row.update({"dataset": dataset_name, "field": field, "unit": unit})
                field_rows.append(row)

            if group_field:
                groups: dict[str, list] = {}
                for item in items:
                    grp = item.get(group_field, "unknown")
                    text = item.get(field, "") or ""
                    groups.setdefault(grp, []).append(text)
                for grp, texts in sorted(groups.items()):
                    for unit, fn in [
                        ("chars", len),
                        ("words", lambda t: len(t.split())),
                    ]:
                        vals = [fn(t) for t in texts]
                        row = describe(
                            vals, f"{dataset_name} / {field} / {grp} / {unit}"
                        )
                        row.update(
                            {
                                "dataset": dataset_name,
                                "field": f"{field}[{grp}]",
                                "unit": unit,
                            }
                        )
                        field_rows.append(row)

    args.outdir.mkdir(parents=True, exist_ok=True)
    df_fields = pd.DataFrame(field_rows)[
        ["dataset", "field", "unit", "n", "mean", "median", "std", "min", "max"]
    ]
    df_fields.to_csv(args.outdir / "text_lengths_fields.csv", index=False)

    print(df_fields.to_string(index=False))
    print(f"\nWritten to {args.outdir / 'text_lengths_fields.csv'}")


if __name__ == "__main__":
    main()
