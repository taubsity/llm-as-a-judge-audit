from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from .config import MetricsConfig
from .stats import _bca_ci_scipy, compute_gwet_ac1


def _process_passfail_group(group_vals, group_df, config):
    group_cols = ["dataset", "task", "model", "config"]
    group_dict = dict(zip(group_cols, group_vals))
    n_resamples = config.n_resamples

    mask = group_df["human_majority"].notna() & group_df["llm_majority"].notna()
    valid = group_df[mask]
    if valid.empty:
        return []

    y_true_arr = (valid["human_majority"] == "FAIL").astype(int).values
    y_pred_arr = (valid["llm_majority"] == "FAIL").astype(int).values

    n_human_fail = int(y_true_arr.sum())
    n_missed = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
    chr_val = n_missed / n_human_fail if n_human_fail > 0 else np.nan

    prec = precision_score(y_true_arr, y_pred_arr, zero_division=0)
    rec = recall_score(y_true_arr, y_pred_arr, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])

    ac1 = compute_gwet_ac1(y_true_arr, y_pred_arr, [0, 1])
    ac1_ci_lo = ac1_ci_hi = np.nan
    _se = ac1.get("gwet_ac1_se", np.nan)
    _v = ac1.get("gwet_ac1", np.nan)
    if not np.isnan(_se) and not np.isnan(_v):
        ac1_ci_lo = float(np.clip(_v - 1.96 * _se, -1, 1))
        ac1_ci_hi = float(np.clip(_v + 1.96 * _se, -1, 1))

    seed_variability = np.nan
    label_lists = (
        group_df.loc[mask, "llm_labels"].tolist()
        if "llm_labels" in group_df.columns
        else []
    )
    disagreement_rates = []
    for labels in label_lists:
        if not isinstance(labels, list) or len(labels) < 2:
            continue
        n_pairs = len(labels) * (len(labels) - 1) // 2
        n_disagree = sum(a != b for a, b in combinations(labels, 2))
        disagreement_rates.append(n_disagree / n_pairs)
    if disagreement_rates:
        seed_variability = float(np.mean(disagreement_rates))

    chr_ci_lo = chr_ci_hi = np.nan
    rec_ci_lo = rec_ci_hi = np.nan

    if len(y_true_arr) >= 3:

        def _chr_stat(yt: np.ndarray, yp: np.ndarray) -> float:
            nf = int(yt.sum())
            nm = int(((yt == 1) & (yp == 0)).sum())
            return nm / nf if nf > 0 else np.nan

        def _rec_stat(yt: np.ndarray, yp: np.ndarray) -> float:
            return float(recall_score(yt, yp, zero_division=0))

        if not np.isnan(chr_val):
            chr_ci_lo, chr_ci_hi = _bca_ci_scipy(
                _chr_stat,
                y_true_arr,
                y_pred_arr,
                n_resamples=n_resamples,
                confidence_level=0.95,
                seed=42,
            )
        rec_ci_lo, rec_ci_hi = _bca_ci_scipy(
            _rec_stat,
            y_true_arr,
            y_pred_arr,
            n_resamples=n_resamples,
            confidence_level=0.95,
            seed=42,
        )

    return [
        {
            **group_dict,
            "paradigm": "passfail",
            "criterion": "clinical_harm_rate",
            "n_sections": len(y_true_arr),
            "clinical_harm_rate": chr_val,
            "chr_ci_lo": chr_ci_lo,
            "chr_ci_hi": chr_ci_hi,
            "n_human_fail": n_human_fail,
            "n_missed_fail": n_missed,
            "precision_micro": prec,
            "recall_micro": rec,
            "recall_ci_lo": rec_ci_lo,
            "recall_ci_hi": rec_ci_hi,
            "f1_micro": f1,
            "seed_variability": seed_variability,
            **ac1,
            "gwet_ac1_ci_lo": ac1_ci_lo,
            "gwet_ac1_ci_hi": ac1_ci_hi,
            "confusion_tn": int(cm[0, 0]),
            "confusion_fp": int(cm[0, 1]),
            "confusion_fn": int(cm[1, 0]),
            "confusion_tp": int(cm[1, 1]),
        }
    ]


def compute_passfail_metrics(
    df: pd.DataFrame,
    config: MetricsConfig,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = ["dataset", "task", "model", "config"]
    groups = list(df.groupby(group_cols, dropna=False))

    nested = Parallel(n_jobs=-1)(
        delayed(_process_passfail_group)(gv, gdf, config) for gv, gdf in groups
    )
    results = [row for group_results in nested for row in group_results]
    return pd.DataFrame(results)
