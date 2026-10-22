import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import MetricsConfig
from .stats import (
    _bca_ci_scipy,
    compute_gwet_ac1,
    compute_gwet_ac2_quadratic,
)


def _process_pointwise_group(group_vals, group_df, config):
    group_cols = ["dataset", "model", "config", "criterion"]
    group_dict = dict(zip(group_cols, group_vals))
    n_resamples = config.n_resamples

    strata: list[tuple[str, pd.DataFrame]] = [("overall", group_df)]
    bin_names = group_df["consensus_bin"].drop_duplicates().tolist()
    for bin_name in bin_names:
        mask = group_df["consensus_bin"].eq(bin_name)
        strata.append((str(bin_name), group_df.loc[mask, :]))

    criterion = group_dict["criterion"]
    is_binary = criterion in config.binary_criteria
    results = []

    for stratum_name, sdf in strata:
        valid = sdf.dropna(subset=["human_median", "machine_median"])
        if len(valid) < 2:
            continue

        h_med = valid["human_median"].values
        m_med = valid["machine_median"].values

        seed_std_mean = valid["machine_seed_std"].mean()
        seed_mode_share_mean = valid["machine_seed_mode_share"].mean()

        if is_binary:
            accuracy = float(np.mean(np.round(h_med) == np.round(m_med)))
            mae = np.nan
            ac1 = compute_gwet_ac1(h_med, m_med, [0, 1])
            ac2 = {"gwet_ac2": np.nan, "gwet_ac2_se": np.nan, "gwet_ac2_p": np.nan}
        else:
            mae = float(np.mean(np.abs(h_med - m_med)))
            accuracy = np.nan
            ac2 = compute_gwet_ac2_quadratic(h_med, m_med, list(range(1, 6)))
            ac1 = {"gwet_ac1": np.nan, "gwet_ac1_se": np.nan, "gwet_ac1_p": np.nan}

        mae_ci_lo = mae_ci_hi = np.nan
        acc_ci_lo = acc_ci_hi = np.nan
        ac2_ci_lo = ac2_ci_hi = np.nan
        ac1_ci_lo = ac1_ci_hi = np.nan

        if is_binary:
            acc_ci_lo, acc_ci_hi = _bca_ci_scipy(
                lambda h_, m_: float(np.mean(np.round(h_) == np.round(m_))),
                h_med,
                m_med,
                n_resamples=n_resamples,
            )
            _se = ac1.get("gwet_ac1_se", np.nan)
            _v = ac1.get("gwet_ac1", np.nan)
            if not (np.isnan(_se) or np.isnan(_v)):
                ac1_ci_lo = float(np.clip(_v - 1.96 * _se, -1, 1))
                ac1_ci_hi = float(np.clip(_v + 1.96 * _se, -1, 1))
        else:
            mae_ci_lo, mae_ci_hi = _bca_ci_scipy(
                lambda h_, m_: float(np.mean(np.abs(h_ - m_))),
                h_med,
                m_med,
                n_resamples=n_resamples,
            )
            _se = ac2.get("gwet_ac2_se", np.nan)
            _v = ac2.get("gwet_ac2", np.nan)
            if not (np.isnan(_se) or np.isnan(_v)):
                ac2_ci_lo = float(np.clip(_v - 1.96 * _se, -1, 1))
                ac2_ci_hi = float(np.clip(_v + 1.96 * _se, -1, 1))

        results.append(
            {
                **group_dict,
                "paradigm": "pointwise",
                "consensus_bin": stratum_name,
                "n_items": len(valid),
                "is_binary": is_binary,
                "mae": mae,
                "mae_ci_lo": mae_ci_lo,
                "mae_ci_hi": mae_ci_hi,
                "accuracy": accuracy,
                "accuracy_ci_lo": acc_ci_lo,
                "accuracy_ci_hi": acc_ci_hi,
                **ac2,
                "gwet_ac2_ci_lo": ac2_ci_lo,
                "gwet_ac2_ci_hi": ac2_ci_hi,
                **ac1,
                "gwet_ac1_ci_lo": ac1_ci_lo,
                "gwet_ac1_ci_hi": ac1_ci_hi,
                "seed_std_mean": seed_std_mean,
                "seed_mode_share_mean": seed_mode_share_mean,
            }
        )
    return results


def compute_pointwise_metrics(
    df: pd.DataFrame,
    config: MetricsConfig,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = ["dataset", "model", "config", "criterion"]
    groups = list(df.groupby(group_cols, dropna=False))

    nested = Parallel(n_jobs=-1)(
        delayed(_process_pointwise_group)(gv, gdf, config) for gv, gdf in groups
    )
    results = [row for group_results in nested for row in group_results]
    return pd.DataFrame(results)
