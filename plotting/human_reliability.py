from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from metrics.human_threshold import (
    compute_human_accuracy_pairwise_threshold,
    compute_human_accuracy_pointwise_threshold,
    compute_human_mae_threshold,
    mae_threshold_dict,
)


def _parse(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def compute_human_mae(item_pw: pd.DataFrame) -> pd.DataFrame:
    return compute_human_mae_threshold(item_pw)


def compute_human_mae_thresholds(item_pw: pd.DataFrame) -> dict[tuple[str, str], float]:
    return mae_threshold_dict(item_pw)


def compute_human_accuracy_pointwise(item_pw: pd.DataFrame) -> pd.DataFrame:
    return compute_human_accuracy_pointwise_threshold(item_pw)


def compute_human_accuracy_pairwise(item_pair: pd.DataFrame) -> pd.DataFrame:
    return compute_human_accuracy_pairwise_threshold(item_pair)


def _dedupe_items(
    df: pd.DataFrame, value_col: str, key_cols: list[str]
) -> pd.DataFrame:
    return df[[*key_cols, value_col]].drop_duplicates(subset=key_cols)


def compute_human_recall_passfail(item_pf: pd.DataFrame) -> pd.DataFrame:
    if item_pf is None or item_pf.empty or "human_binary" not in item_pf.columns:
        return pd.DataFrame()

    sub = _dedupe_items(item_pf, "human_binary", ["dataset", "task", "section", "id"])
    sub = sub.assign(n_raters=sub["human_binary"].apply(lambda v: len(_parse(v))))
    if (sub["n_raters"] < 2).all():
        return pd.DataFrame()

    def _loo_hits(labels) -> list[bool]:
        labels = [x for x in _parse(labels) if pd.notna(x)]
        n = len(labels)
        if n < 2:
            return []
        hits = []
        for i in range(n):
            others = labels[:i] + labels[i + 1 :]
            if round(np.median(others)) == 1:
                hits.append(round(labels[i]) == 1)
        return hits

    sub = sub.assign(hits=sub["human_binary"].apply(_loo_hits))
    rows = []
    for dataset, gdf in sub.groupby("dataset"):
        all_hits = [h for hits in gdf["hits"] for h in hits]
        if all_hits:
            rows.append(
                {
                    "dataset": dataset,
                    "mean": float(np.mean(all_hits)),
                    "n_items": len(all_hits),
                }
            )
    return pd.DataFrame(rows)
