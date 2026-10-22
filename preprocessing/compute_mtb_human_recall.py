from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from irrCAC.raw import CAC

from .common import get_severity

SEVERITY_CONFIGS: dict[str, set[str]] = {
    "erheb": {"ERHEB"},
    "krit": {"KRIT"},
    "erheb_krit": {"ERHEB", "KRIT"},
}


def annotator_section_labels(
    line_annotations: list[dict], severities: set[str]
) -> dict[int, bool]:
    labels: dict[int, bool] = {}
    for ann in line_annotations:
        line_idx = ann["line_index"]
        errors = ann.get("errors", [])
        is_fail = any(get_severity(e["error_code"]) in severities for e in errors)
        labels[line_idx] = labels.get(line_idx, False) or is_fail
    return labels


def load_doc_annotator_raw(input_dir: Path) -> dict[str, list[list[dict]]]:
    doc_annotator_raw: dict[str, list[list[dict]]] = {}

    for user_dir in sorted(input_dir.iterdir()):
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
            doc_annotator_raw.setdefault(doc_id, []).append(line_annotations)

    return doc_annotator_raw


def labels_for_severities(
    doc_annotator_raw: dict[str, list[list[dict]]], severities: set[str]
) -> dict[str, list[dict[int, bool]]]:
    return {
        doc_id: [annotator_section_labels(la, severities) for la in raw_list]
        for doc_id, raw_list in doc_annotator_raw.items()
    }


def compute_pairwise_recall(
    doc_annotator_labels: dict[str, list[dict[int, bool]]],
) -> tuple[float, int]:
    hits = 0
    total = 0
    for annotator_labels in doc_annotator_labels.values():
        n = len(annotator_labels)
        if n < 2:
            continue
        all_sections = set()
        for labels in annotator_labels:
            all_sections.update(labels.keys())

        for sec in all_sections:
            for i in range(n):
                if not annotator_labels[i].get(sec, False):
                    continue
                for j in range(n):
                    if j == i:
                        continue
                    total += 1
                    hits += int(annotator_labels[j].get(sec, False))

    recall = hits / total if total else float("nan")
    return recall, total


def compute_human_ac1(
    doc_annotator_labels: dict[str, list[dict[int, bool]]],
) -> tuple[float, float, int]:
    multi_docs = {
        doc_id: labels
        for doc_id, labels in doc_annotator_labels.items()
        if len(labels) >= 2
    }
    if not multi_docs:
        return float("nan"), float("nan"), 0

    max_raters = max(len(labels) for labels in multi_docs.values())

    rows = []
    for annotator_labels in multi_docs.values():
        all_sections = set()
        for labels in annotator_labels:
            all_sections.update(labels.keys())

        n = len(annotator_labels)
        for sec in all_sections:
            row = [int(labels.get(sec, False)) for labels in annotator_labels]
            row += [np.nan] * (max_raters - n)
            rows.append(row)

    df_ratings = pd.DataFrame(
        rows, columns=[f"rater_{i + 1}" for i in range(max_raters)]
    ).astype(float)

    cac = CAC(df_ratings, weights="identity", categories=[0, 1])
    est = cac.gwet()["est"]
    return float(est["coefficient_value"]), float(est["se"]), len(df_ratings)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute human-human (inter-annotator) recall for MTB PASS/FAIL labels.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder with one subfolder per annotator, each with progress.json "
        "(same --input-dir as format_mtb_errors.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/metrics/human_recall_mtb.csv"),
        help="Output CSV path (dataset, severity, mean, n_items columns, "
        "matching the other human_reliability sheets, one row per severity "
        "config). A sibling '<stem>_ac1.csv' with Gwet's AC1 is also written.",
    )
    args = parser.parse_args()

    doc_annotator_raw = load_doc_annotator_raw(args.input_dir)
    n_docs = len(doc_annotator_raw)

    recall_rows = []
    ac1_rows = []
    for severity_name, severities in SEVERITY_CONFIGS.items():
        doc_annotator_labels = labels_for_severities(doc_annotator_raw, severities)
        n_multi = sum(1 for v in doc_annotator_labels.values() if len(v) >= 2)
        recall, n_pairs = compute_pairwise_recall(doc_annotator_labels)
        ac1, ac1_se, n_items_ac1 = compute_human_ac1(doc_annotator_labels)

        recall_rows.append(["mtb_errors", severity_name, recall, n_pairs])
        ac1_rows.append(["mtb_errors", severity_name, ac1, ac1_se, n_items_ac1])

        print(f"--- Severity: {severity_name} ---")
        print(f"Documents with >=2 annotators: {n_multi}")
        print(f"Pairwise (rater, section) comparisons: {n_pairs}")
        print(f"Human-human recall: {recall:.4f}")
        print(f"Human-human Gwet's AC1: {ac1:.4f} (se={ac1_se:.4f}, n={n_items_ac1})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "severity", "mean", "n_items"])
        writer.writerows(recall_rows)

    ac1_output = args.output.with_name(args.output.stem + "_ac1" + args.output.suffix)
    with open(ac1_output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "severity", "gwet_ac1", "gwet_ac1_se", "n_items"])
        writer.writerows(ac1_rows)

    print(f"Documents total: {n_docs}")
    print(f"Wrote {args.output}")
    print(f"Wrote {ac1_output}")


if __name__ == "__main__":
    main()
