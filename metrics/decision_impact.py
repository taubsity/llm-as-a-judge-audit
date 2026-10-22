from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import bootstrap as scipy_bootstrap

from .stats import _bca_ci_scipy


def _bootstrap_ci(
    values: np.ndarray,
    stat_fn=np.mean,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) < 3:
        return np.nan, np.nan
    for method in ("BCa", "percentile"):
        if method == "BCa" and len(np.unique(values)) == 1:
            continue
        try:
            res = scipy_bootstrap(
                (values,),
                stat_fn,
                n_resamples=n_resamples,
                confidence_level=0.95,
                method=method,
                random_state=seed,
            )
            return float(res.confidence_interval.low), float(
                res.confidence_interval.high
            )
        except Exception:
            continue
    return np.nan, np.nan


def pointwise_decision_impact(
    item_df: pd.DataFrame,
    model: str = "G-120B",
    config: str = "baseline",
    n_resamples: int = 1000,
) -> pd.DataFrame:
    df = item_df[(item_df["model"] == model) & (item_df["config"] == config)].copy()
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["human_median", "machine_median"])

    records: list[dict[str, Any]] = []
    group_cols = ["dataset", "criterion"]

    for (dataset, criterion), gdf in df.groupby(group_cols, dropna=False):
        strata: list[tuple[str, pd.DataFrame]] = [("overall", gdf)]
        for bin_name in gdf["consensus_bin"].dropna().unique():
            strata.append((str(bin_name), gdf[gdf["consensus_bin"] == bin_name]))

        for bin_name, sdf in strata:
            if len(sdf) < 2:
                continue

            h = np.round(sdf["human_median"].values)
            m = np.round(sdf["machine_median"].values)
            diff = m - h

            flips = (h != m).astype(float)
            flip_rate = float(flips.mean())
            flip_ci = _bootstrap_ci(flips, np.mean, n_resamples)

            up_flips = float((diff > 0).mean())
            down_flips = float((diff < 0).mean())

            signed_dev = float(diff.mean())
            signed_dev_ci = _bootstrap_ci(diff, np.mean, n_resamples)

            records.append(
                {
                    "paradigm": "pointwise",
                    "dataset": dataset,
                    "criterion": criterion,
                    "consensus_bin": bin_name,
                    "model": model,
                    "config": config,
                    "n_items": len(sdf),
                    "flip_rate": flip_rate,
                    "flip_rate_ci_lo": flip_ci[0],
                    "flip_rate_ci_hi": flip_ci[1],
                    "up_flip_rate": up_flips,
                    "down_flip_rate": down_flips,
                    "signed_deviation": signed_dev,
                    "signed_deviation_ci_lo": signed_dev_ci[0],
                    "signed_deviation_ci_hi": signed_dev_ci[1],
                }
            )

    return pd.DataFrame(records)


def pairwise_decision_impact(
    item_df: pd.DataFrame,
    model: str = "G-120B",
    config: str = "demo",
    n_resamples: int = 1000,
) -> pd.DataFrame:
    df = item_df[(item_df["model"] == model) & (item_df["config"] == config)].copy()
    if df.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    group_cols = ["dataset", "criterion"]

    for (dataset, criterion), gdf in df.groupby(group_cols, dropna=False):
        strata: list[tuple[str, pd.DataFrame]] = [("overall", gdf)]
        for bin_name in gdf["consensus_bin"].dropna().unique():
            strata.append((str(bin_name), gdf[gdf["consensus_bin"] == bin_name]))

        for bin_name, sdf in strata:
            if len(sdf) < 2:
                continue

            h = sdf["human_majority"].values
            l = sdf["llm_majority"].values

            flips = (h != l).astype(float)
            flip_rate = float(flips.mean())
            flip_ci = _bootstrap_ci(flips, np.mean, n_resamples)

            human_A_rate = float((h == "A").mean())
            llm_A_rate = float((l == "A").mean())
            delta_win_rate = llm_A_rate - human_A_rate

            flow = {}
            for src in ["A", "B", "TIE"]:
                for dst in ["A", "B", "TIE"]:
                    flow[f"flow_{src}_{dst}"] = int(((h == src) & (l == dst)).sum())

            records.append(
                {
                    "paradigm": "pairwise",
                    "dataset": dataset,
                    "criterion": criterion,
                    "consensus_bin": bin_name,
                    "model": model,
                    "config": config,
                    "n_items": len(sdf),
                    "flip_rate": flip_rate,
                    "flip_rate_ci_lo": flip_ci[0],
                    "flip_rate_ci_hi": flip_ci[1],
                    "human_A_rate": human_A_rate,
                    "llm_A_rate": llm_A_rate,
                    "delta_win_rate": delta_win_rate,
                    **flow,
                }
            )

    return pd.DataFrame(records)


def passfail_decision_impact(
    item_df: pd.DataFrame,
    model: str = "Q-235B",
    config: str = "rule",
    n_resamples: int = 1000,
) -> pd.DataFrame:
    df = item_df[
        (item_df["model"] == model)
        & (item_df["config"] == config)
        & (item_df["dataset"] == "mtb_errors")
    ].copy()
    if df.empty:
        return pd.DataFrame()

    df["human_major"] = df["human_majority"] == "FAIL"
    df["llm_major"] = df["llm_majority"] == "FAIL"

    records: list[dict[str, Any]] = []

    strata: list[tuple[str, pd.DataFrame]] = [("overall", df)]

    for bin_name, sdf in strata:
        if len(sdf) < 2:
            continue

        y_true = sdf["human_major"].values.astype(int)
        y_pred = sdf["llm_major"].values.astype(int)

        n_pos = int(y_true.sum())
        if n_pos == 0:
            continue

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        recall = tp / n_pos

        def _recall_stat(yt: np.ndarray, yp: np.ndarray) -> float:
            n = int(yt.sum())
            if n == 0:
                return np.nan
            return float(((yt == 1) & (yp == 1)).sum() / n)

        ci = _bca_ci_scipy(
            _recall_stat, y_true, y_pred, n_resamples=n_resamples, seed=42
        )

        records.append(
            {
                "paradigm": "passfail",
                "dataset": "mtb_errors",
                "criterion": "major_critical_recall",
                "consensus_bin": bin_name,
                "model": model,
                "config": config,
                "n_items": len(sdf),
                "n_human_major": n_pos,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "recall_llm": recall,
                "recall_llm_ci_lo": ci[0],
                "recall_llm_ci_hi": ci[1],
            }
        )

    return pd.DataFrame(records)


def build_decision_impact_frame(
    item_pw: pd.DataFrame,
    item_pair: pd.DataFrame,
    item_pf: pd.DataFrame,
    pw_model: str = "G-120B",
    pw_config: str = "demo",
    pair_model: str = "G-120B",
    pair_config: str = "demo",
    pf_model: str = "Q-235B",
    pf_config: str = "rule",
    n_resamples: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pw_impact = pointwise_decision_impact(item_pw, pw_model, pw_config, n_resamples)
    pair_impact = pairwise_decision_impact(
        item_pair, pair_model, pair_config, n_resamples
    )
    pf_impact = passfail_decision_impact(item_pf, pf_model, pf_config, n_resamples)
    return pw_impact, pair_impact, pf_impact
