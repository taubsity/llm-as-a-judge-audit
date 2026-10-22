from __future__ import annotations

from collections import Counter
from statistics import mean, median, stdev
from typing import Any

import numpy as np
import pandas as pd

from .config import MAJOR_SEVERITIES, GroupKey, MetricsConfig

NURSING_NUMERIC_KEY_MAP: dict[str, str] = {
    "1": "citation",
    "2": "accurate",
    "3": "thorough",
    "4": "useful",
    "5": "organized",
    "6": "comprehensible",
    "7": "succinct",
    "8": "abstraction",
    "9.1": "stig_original",
    "9.2": "stig_summary",
}

VARIANT_KEY_MAP: dict[str, str] = {
    "citations": "citation",
    "stigmatizing_original": "stig_original",
    "stigmatizing_summary": "stig_summary",
}


def _remap_dict_keys(d: dict, key_map: dict[str, str]) -> dict:
    return {key_map.get(k, k): v for k, v in d.items()}


def normalize_llm_keys(record: dict[str, Any]) -> dict[str, Any]:
    human_key_set: set[str] = set()
    for label_field in ("human_labels", "human_votes"):
        for d in record.get(label_field, []):
            if isinstance(d, dict):
                human_key_set.update(d.keys())

    for field_name in ("llm_labels", "llm_votes", "llm_votes_swapped"):
        entries = record.get(field_name)
        if not entries or not isinstance(entries, list):
            continue
        new_entries = []
        for d in entries:
            if not isinstance(d, dict):
                new_entries.append(d)
                continue
            if any(k in NURSING_NUMERIC_KEY_MAP for k in d):
                d = _remap_dict_keys(d, NURSING_NUMERIC_KEY_MAP)
            d = _remap_dict_keys(d, VARIANT_KEY_MAP)
            if human_key_set:
                d = {k: v for k, v in d.items() if k in human_key_set}
            new_entries.append(d)
        record[field_name] = new_entries

    for sub in record.get("individual_results", []):
        normalize_llm_keys(sub)

    return record


def filter_placeholder_raters(record: dict[str, Any]) -> dict[str, Any]:
    def _is_placeholder_numeric_dict(d: Any) -> bool:
        if not isinstance(d, dict) or not d:
            return False
        numeric_vals = [v for v in d.values() if isinstance(v, (int, float))]
        non_numeric_vals = [v for v in d.values() if not isinstance(v, (int, float))]

        if non_numeric_vals:
            return False
        if not numeric_vals:
            return False
        return all(v == 0 for v in numeric_vals)

    hl = record.get("human_labels")
    if hl and isinstance(hl, list):
        record["human_labels"] = [d for d in hl if not _is_placeholder_numeric_dict(d)]
    hv = record.get("human_votes")
    if hv and isinstance(hv, list):
        record["human_votes"] = [
            d
            for d in hv
            if not isinstance(d, dict)
            or any(v not in (0, "", None) for v in d.values())
        ]
    return record


def extract_group_key(record: dict[str, Any]) -> GroupKey:
    dataset = record.get("dataset", "unknown")
    paradigm = record.get("paradigm", "unknown")
    config = record.get("config", "unknown")

    task = record.get("task", "")
    if not task and "individual_results" in record:
        indiv = record["individual_results"]
        if indiv and isinstance(indiv, list) and len(indiv) > 0:
            task = indiv[0].get("task", "")

    model = record.get("model", "")
    if not model:
        if config == "multiagent" or "models_aggregated" in record:
            model = "MULTIAGENT"
        else:
            model = "unknown"

    return GroupKey(
        dataset=dataset, task=task, paradigm=paradigm, model=model, config=config
    )


def normalize_vote(vote: str) -> str:
    if not isinstance(vote, str):
        return "INVALID"
    v = vote.upper().strip()
    if v in {"SAME", "UNCLEAR", "TIE", "EQUAL", "NEITHER", "BOTH"}:
        return "TIE"
    if v in {"A", "OPTION_A", "OPTION A", "FIRST", "1"}:
        return "A"
    if v in {"B", "OPTION_B", "OPTION B", "SECOND", "2"}:
        return "B"
    return v


def _assign_consensus_bin(
    consensus: float,
    bins: list[tuple[str, float, float]],
) -> str:
    for name, lo, hi in bins:
        if lo <= consensus < hi:
            return name
    return "unknown"


def normalize_pointwise_to_long(
    records: list[dict[str, Any]],
    config: MetricsConfig,
) -> pd.DataFrame:
    rows: list[dict] = []

    for rec in records:
        if rec.get("paradigm") != "pointwise":
            continue

        gk = extract_group_key(rec)
        item_id = rec.get("id", "")

        human_labels_raw = rec.get("human_labels", [])
        llm_labels_raw = rec.get("llm_labels", [])
        if isinstance(human_labels_raw, dict):
            human_labels_raw = [human_labels_raw]
        if isinstance(llm_labels_raw, dict):
            llm_labels_raw = [llm_labels_raw]

        criteria: set[str] = set()
        for d in human_labels_raw:
            if isinstance(d, dict):
                criteria.update(d.keys())
        for d in llm_labels_raw:
            if isinstance(d, dict):
                criteria.update(d.keys())

        for criterion in sorted(criteria):
            human_vals = [
                d[criterion]
                for d in human_labels_raw
                if isinstance(d, dict)
                and criterion in d
                and isinstance(d[criterion], (int, float))
                and not np.isnan(d[criterion])
            ]
            llm_vals = [
                d[criterion]
                for d in llm_labels_raw
                if isinstance(d, dict)
                and criterion in d
                and isinstance(d[criterion], (int, float))
                and not np.isnan(d[criterion])
            ]

            if not human_vals and not llm_vals:
                continue

            h_med = median(human_vals) if human_vals else np.nan
            h_mean = mean(human_vals) if human_vals else np.nan
            h_std = stdev(human_vals) if len(human_vals) > 1 else 0.0
            m_med = median(llm_vals) if llm_vals else np.nan
            m_mean = mean(llm_vals) if llm_vals else np.nan
            m_std = stdev(llm_vals) if len(llm_vals) > 1 else 0.0
            m_mode_share = (
                Counter(llm_vals).most_common(1)[0][1] / len(llm_vals)
                if llm_vals
                else np.nan
            )

            if human_vals:
                med_agreement = sum(1 for v in human_vals if v == h_med) / len(
                    human_vals
                )
            else:
                med_agreement = np.nan

            consensus_bin = _assign_consensus_bin(
                med_agreement if not np.isnan(med_agreement) else 0.0,
                config.consensus_percentage_bins,
            )

            rows.append(
                {
                    "id": item_id,
                    **gk.to_dict(),
                    "criterion": criterion,
                    "human_labels": human_vals,
                    "machine_labels": llm_vals,
                    "human_median": h_med,
                    "human_mean": h_mean,
                    "human_std": h_std,
                    "machine_median": m_med,
                    "machine_mean": m_mean,
                    "machine_seed_std": m_std,
                    "machine_seed_mode_share": m_mode_share,
                    "human_median_agreement": med_agreement,
                    "consensus_bin": consensus_bin,
                }
            )

    df = pd.DataFrame(rows)
    return df


def normalize_pairwise_to_long(
    records: list[dict[str, Any]],
    config: MetricsConfig,
) -> pd.DataFrame:
    rows: list[dict] = []

    for rec in records:
        if rec.get("paradigm") != "pairwise":
            continue

        gk = extract_group_key(rec)
        item_id = rec.get("id", "")

        human_votes_raw = rec.get("human_votes", [])
        llm_votes_raw = rec.get("llm_votes", [])
        llm_swapped_raw = rec.get("llm_votes_swapped", [])

        if isinstance(human_votes_raw, dict):
            human_votes_raw = [human_votes_raw]
        if isinstance(llm_votes_raw, dict):
            llm_votes_raw = [llm_votes_raw]
        if isinstance(llm_swapped_raw, dict):
            llm_swapped_raw = [llm_swapped_raw]

        llm_criteria: set[str] = set()
        for d in llm_votes_raw:
            if isinstance(d, dict):
                llm_criteria.update(d.keys())

        for criterion in sorted(llm_criteria):
            human_votes = [
                normalize_vote(d[criterion])
                for d in human_votes_raw
                if isinstance(d, dict) and criterion in d
            ]
            llm_votes = [
                normalize_vote(d[criterion])
                for d in llm_votes_raw
                if isinstance(d, dict) and criterion in d
            ]
            llm_swapped = [
                normalize_vote(d[criterion])
                for d in llm_swapped_raw
                if isinstance(d, dict) and criterion in d
            ]

            if not human_votes and not llm_votes:
                continue

            def _majority(votes: list[str]) -> tuple[str, float]:
                if not votes:
                    return ("", 0.0)
                c = Counter(votes)
                maj, cnt = c.most_common(1)[0]
                return (maj, cnt / len(votes))

            h_maj, h_share = _majority(human_votes)
            l_maj, l_share = _majority(llm_votes)
            l_maj_sw, _ = _majority(llm_swapped)

            is_clear = h_share >= config.majority_thresh

            consensus_bin = _assign_consensus_bin(
                h_share if not np.isnan(h_share) else 0.0,
                config.consensus_percentage_bins,
            )

            rows.append(
                {
                    "id": item_id,
                    **gk.to_dict(),
                    "criterion": criterion,
                    "human_votes": human_votes,
                    "llm_votes": llm_votes,
                    "llm_votes_swapped": llm_swapped,
                    "human_majority": h_maj,
                    "human_majority_share": h_share,
                    "llm_majority": l_maj,
                    "llm_majority_share": l_share,
                    "llm_majority_swapped": l_maj_sw,
                    "is_clear_majority": is_clear,
                    "is_ambiguous": not is_clear,
                    "consensus_bin": consensus_bin,
                }
            )

    df = pd.DataFrame(rows)
    return df


def _error_codes_to_passfail(codes: list) -> str:
    for code in codes:
        if not isinstance(code, str):
            continue
        parts = code.split(".")
        if parts and parts[-1] in MAJOR_SEVERITIES:
            return "FAIL"
    return "PASS"


def _section_value_to_passfail(val: Any) -> str:
    if isinstance(val, dict):
        val = val.get("errors", "")
    if isinstance(val, str):
        return val.upper()
    if isinstance(val, list):
        if not val:
            return "PASS"
        return _error_codes_to_passfail(val)
    return str(val).upper()


def _collect_section_labels(
    labels_raw: Any,
) -> tuple[dict[str, list[str]], set[str]]:
    if isinstance(labels_raw, dict):
        labels_raw = [labels_raw]
    if not isinstance(labels_raw, list):
        return {}, set()

    sections: set[str] = set()
    for d in labels_raw:
        if isinstance(d, dict):
            sections.update(d.keys())

    section_labels: dict[str, list[str]] = {s: [] for s in sections}
    for d in labels_raw:
        if not isinstance(d, dict):
            continue
        for section in sections:
            if section in d:
                section_labels[section].append(_section_value_to_passfail(d[section]))

    return section_labels, sections


def normalize_passfail_to_long(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict] = []

    for rec in records:
        if rec.get("paradigm") != "passfail":
            continue

        gk = extract_group_key(rec)
        item_id = rec.get("id", "")

        human_section_labels, h_sections = _collect_section_labels(
            rec.get("human_labels", [])
        )
        llm_section_labels, l_sections = _collect_section_labels(
            rec.get("llm_labels", [])
        )

        sections = h_sections | l_sections

        for section in sorted(sections):
            h_labels = human_section_labels.get(section, [])
            l_labels = llm_section_labels.get(section, [])

            def _pf_majority(labels: list[str]) -> str:
                if not labels:
                    return ""
                return Counter(labels).most_common(1)[0][0]

            rows.append(
                {
                    "id": item_id,
                    **gk.to_dict(),
                    "section": section,
                    "human_labels": h_labels,
                    "llm_labels": l_labels,
                    "human_majority": _pf_majority(h_labels),
                    "llm_majority": _pf_majority(l_labels),
                    "human_binary": [1 if lbl == "FAIL" else 0 for lbl in h_labels],
                    "llm_binary": [1 if lbl == "FAIL" else 0 for lbl in l_labels],
                }
            )

    df = pd.DataFrame(rows)
    return df
