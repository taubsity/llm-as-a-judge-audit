from __future__ import annotations

import ast
from collections import Counter

import numpy as np
import pandas as pd

from .config import MetricsConfig
from .model_comparison import _prepare


def _parse(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def _loo_mae_errs(labels) -> list[float]:
    labels = [x for x in _parse(labels) if pd.notna(x)]
    n = len(labels)
    if n < 2:
        return []
    return [
        float(abs(labels[i] - np.median(labels[:i] + labels[i + 1 :])))
        for i in range(n)
    ]


def _loo_acc_hits(labels) -> list[float]:
    labels = [x for x in _parse(labels) if pd.notna(x)]
    n = len(labels)
    if n < 2:
        return []
    return [
        float(round(labels[i]) == round(np.median(labels[:i] + labels[i + 1 :])))
        for i in range(n)
    ]


def _loo_pairwise_hits(votes) -> list[float]:
    votes = _parse(votes)
    n = len(votes)
    if n < 2:
        return []
    hits: list[float] = []
    for i in range(n):
        others = votes[:i] + votes[i + 1 :]
        top_vote, top_n = Counter(others).most_common(1)[0]
        if top_n / len(others) <= 0.5:
            continue
        hits.append(float(votes[i] == top_vote))
    return hits


def _pool_and_bootstrap(
    gdf: pd.DataFrame, list_col: str, n_resamples: int, seed: int
) -> dict[str, float]:
    lists = [lst for lst in gdf[list_col] if lst]
    n_clusters = len(lists)
    pooled = [v for lst in lists for v in lst]
    n_items = len(pooled)
    if n_items == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "mean_se": np.nan,
            "n_items": 0,
            "n_clusters": 0,
        }

    mean = float(np.mean(pooled))
    std = float(np.std(pooled, ddof=1)) if n_items > 1 else np.nan

    se = np.nan
    if n_clusters >= 3:
        rng = np.random.default_rng(seed)
        boots = np.empty(n_resamples)
        for b in range(n_resamples):
            pick = rng.integers(0, n_clusters, n_clusters)
            pooled_b = [v for i in pick for v in lists[i]]
            boots[b] = np.mean(pooled_b) if pooled_b else np.nan
        boots = boots[np.isfinite(boots)]
        if len(boots) >= 20:
            se = float(np.std(boots, ddof=1))

    return {
        "mean": mean,
        "std": std,
        "mean_se": se,
        "n_items": n_items,
        "n_clusters": n_clusters,
    }


def _threshold_frame(
    df: pd.DataFrame,
    id_cols: list[str],
    value_col: str,
    loo_fn,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    prepared = _prepare(df, [c for c in id_cols if c != "dataset"])
    sub = prepared[
        ["dataset", *[c for c in id_cols if c != "dataset"], "_cluster", value_col]
    ]
    sub = sub.drop_duplicates(
        subset=["dataset", "_cluster", *[c for c in id_cols if c not in ("dataset",)]]
    )
    sub = sub.assign(_outcomes=sub[value_col].apply(loo_fn))
    sub = sub[sub["_outcomes"].map(len) > 0]
    if sub.empty:
        return pd.DataFrame()

    group_cols = [c for c in id_cols if c != "id"]
    rows: list[dict] = []
    for keys, gdf in sub.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        stats = _pool_and_bootstrap(gdf, "_outcomes", n_resamples, seed)
        rows.append({**dict(zip(group_cols, keys)), **stats})
    return pd.DataFrame(rows)


def compute_human_mae_threshold(
    item_pw: pd.DataFrame,
    config_obj: MetricsConfig | None = None,
    n_resamples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    if item_pw is None or item_pw.empty or "human_labels" not in item_pw.columns:
        return pd.DataFrame()
    config_obj = config_obj or MetricsConfig()
    df = item_pw[~item_pw["criterion"].isin(config_obj.binary_criteria)]
    if df.empty:
        return pd.DataFrame()
    return _threshold_frame(
        df,
        ["dataset", "criterion", "id"],
        "human_labels",
        _loo_mae_errs,
        n_resamples,
        seed,
    )


def compute_human_accuracy_pointwise_threshold(
    item_pw: pd.DataFrame,
    config_obj: MetricsConfig | None = None,
    n_resamples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    if item_pw is None or item_pw.empty or "human_labels" not in item_pw.columns:
        return pd.DataFrame()
    config_obj = config_obj or MetricsConfig()
    df = item_pw[item_pw["criterion"].isin(config_obj.binary_criteria)]
    if df.empty:
        return pd.DataFrame()
    return _threshold_frame(
        df,
        ["dataset", "criterion", "id"],
        "human_labels",
        _loo_acc_hits,
        n_resamples,
        seed,
    )


def compute_human_accuracy_pairwise_threshold(
    item_pair: pd.DataFrame,
    n_resamples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    if item_pair is None or item_pair.empty or "human_votes" not in item_pair.columns:
        return pd.DataFrame()
    return _threshold_frame(
        item_pair,
        ["dataset", "criterion", "id"],
        "human_votes",
        _loo_pairwise_hits,
        n_resamples,
        seed,
    )


def mae_threshold_dict(
    item_pw: pd.DataFrame, config_obj: MetricsConfig | None = None
) -> dict:
    df = compute_human_mae_threshold(item_pw, config_obj)
    if df.empty:
        return {}
    return {(r["dataset"], r["criterion"]): float(r["mean"]) for _, r in df.iterrows()}
