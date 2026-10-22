from __future__ import annotations

import itertools
import json
import statistics
import warnings
from collections import Counter
from statistics import mode
from typing import Callable

import numpy as np
import pandas as pd
from irrCAC.raw import CAC
from scipy.spatial.distance import jensenshannon
from scipy.stats import DegenerateDataWarning
from scipy.stats import bootstrap as scipy_bootstrap

from .config import MetricsConfig


def _bca_ci_scipy(
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    h: np.ndarray,
    m: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    for method in ("BCa", "percentile"):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=DegenerateDataWarning)
                warnings.filterwarnings("error", category=RuntimeWarning)
                result = scipy_bootstrap(
                    (h, m),
                    statistic=stat_fn,
                    n_resamples=n_resamples,
                    confidence_level=confidence_level,
                    method=method,
                    random_state=seed,
                    paired=True,
                )
            return float(result.confidence_interval.low), float(
                result.confidence_interval.high
            )
        except (DegenerateDataWarning, RuntimeWarning, ValueError):
            if method == "percentile":
                return np.nan, np.nan
            continue
    return np.nan, np.nan


def compute_gwet_ac2_quadratic(
    human_medians: np.ndarray,
    machine_medians: np.ndarray,
    categories: list[int],
) -> dict[str, float]:
    df_ratings = pd.DataFrame(
        {
            "human_median": np.round(human_medians).astype(int),
            "llm_median": np.round(machine_medians).astype(int),
        }
    )
    cac = CAC(df_ratings, weights="quadratic", categories=categories)
    res = cac.gwet()
    est = res["est"]
    return {
        "gwet_ac2": est["coefficient_value"],
        "gwet_ac2_se": est["se"],
        "gwet_ac2_p": est["p_value"],
    }


def compute_gwet_ac1(
    human_majorities: np.ndarray,
    llm_majorities: np.ndarray,
    categories: list[int],
) -> dict[str, float]:
    df_ratings = pd.DataFrame(
        {
            "human": np.round(human_majorities).astype(int),
            "llm": np.round(llm_majorities).astype(int),
        }
    )
    cac = CAC(df_ratings, weights="identity", categories=categories)
    res = cac.gwet()
    est = res["est"]
    return {
        "gwet_ac1": est["coefficient_value"],
        "gwet_ac1_se": est["se"],
        "gwet_ac1_p": est["p_value"],
    }


def compute_binned_js(
    df, human_labels_column, machine_labels_column, bin_type="median"
):
    result_weighted_js = 0
    result_bin_details = {}

    bin_func = statistics.median if bin_type == "median" else mode

    labels = sorted(
        set(
            itertools.chain(
                *(df[human_labels_column].tolist() + df[machine_labels_column].tolist())
            )
        )
    )

    df_bin_number = df[human_labels_column].apply(lambda x: bin_func(x))

    for label in labels:
        df_bin_items = df[df_bin_number == label]

        bin_weight = len(df_bin_items) / len(df)

        result_bin_details[label] = {
            "bin_number": label,
            "jsd": None,
            "bin_weight": bin_weight,
            "dist_human": [],
            "dist_machine": [],
        }

        if len(df_bin_items) == 0:
            continue

        human_labels_frequencies_in_bin = Counter(
            list(itertools.chain(*df_bin_items[human_labels_column]))
        )
        machine_labels_frequencies_in_bin = Counter(
            list(itertools.chain(*df_bin_items[machine_labels_column]))
        )

        human_total = sum(Counter(human_labels_frequencies_in_bin).values())
        machine_total = sum(Counter(machine_labels_frequencies_in_bin).values())

        if human_total == 0 or machine_total == 0:
            continue

        dist_human = [human_labels_frequencies_in_bin[i] / human_total for i in labels]
        dist_machine = [
            machine_labels_frequencies_in_bin[i] / machine_total for i in labels
        ]

        result_bin_details[label]["dist_machine"] = dist_machine
        result_bin_details[label]["dist_human"] = dist_human

        jsd = jensenshannon(dist_human, dist_machine)

        result_bin_details[label]["jsd"] = jsd

        result_weighted_js += bin_weight * jsd

    return result_weighted_js, result_bin_details


def _bca_ci_jsd(
    df: pd.DataFrame,
    human_col: str,
    machine_col: str,
    bin_type: str,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    n = len(df)
    if n < 3:
        return np.nan, np.nan
    indices = np.arange(n)

    def _stat(idx: np.ndarray, _dummy: np.ndarray) -> float:
        resampled = df.iloc[idx.astype(int)]
        jsd_val, _ = compute_binned_js(resampled, human_col, machine_col, bin_type)
        return float(jsd_val)

    for method in ("BCa", "percentile"):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=DegenerateDataWarning)
                warnings.filterwarnings("error", category=RuntimeWarning)
                result = scipy_bootstrap(
                    (indices, indices),
                    statistic=_stat,
                    n_resamples=n_resamples,
                    confidence_level=confidence_level,
                    method=method,
                    random_state=seed,
                    paired=True,
                )
            return float(result.confidence_interval.low), float(
                result.confidence_interval.high
            )
        except (DegenerateDataWarning, RuntimeWarning, ValueError):
            if method == "percentile":
                return np.nan, np.nan
            continue
    return np.nan, np.nan


def _run_jsd_for_subgroup(
    gdf,
    human_col,
    machine_col,
    bin_type,
    group_dict,
    paradigm,
    consensus_label,
    n_resamples,
):
    if gdf.empty:
        return None
    weighted_jsd, bin_details = compute_binned_js(gdf, human_col, machine_col, bin_type)
    ci_lo, ci_hi = _bca_ci_jsd(
        gdf, human_col, machine_col, bin_type, n_resamples=n_resamples
    )
    serializable_details = {
        str(k): {
            dk: (float(dv) if isinstance(dv, (int, float, np.floating)) else dv)
            for dk, dv in dv_dict.items()
        }
        for k, dv_dict in bin_details.items()
    }
    return {
        **group_dict,
        "paradigm": paradigm,
        "consensus_bin": consensus_label,
        "weighted_jsd": float(weighted_jsd),
        "weighted_jsd_ci_lo": ci_lo,
        "weighted_jsd_ci_hi": ci_hi,
        "n_items": len(gdf),
        "bin_details_json": json.dumps(serializable_details),
    }


def compute_consensus_stratified_jsd(
    df_pw: pd.DataFrame,
    df_pair: pd.DataFrame,
    config: MetricsConfig,
) -> pd.DataFrame:
    from joblib import Parallel, delayed

    _VOTE_MAP = {"A": 0, "B": 1, "TIE": 2}
    n_resamples = config.n_resamples

    jobs = []

    if not df_pw.empty:
        group_cols = ["dataset", "task", "model", "config", "criterion"]
        for group_vals, gdf in df_pw.groupby(group_cols, dropna=False):
            group_dict = dict(zip(group_cols, group_vals))
            is_binary = group_dict["criterion"] in config.binary_criteria
            bin_type = "majority" if is_binary else "median"

            jobs.append(
                (
                    gdf,
                    "human_labels",
                    "machine_labels",
                    bin_type,
                    group_dict,
                    "pointwise",
                    "overall",
                )
            )
            for bin_name in ("high", "medium", "low"):
                subset = gdf[gdf["consensus_bin"] == bin_name]
                jobs.append(
                    (
                        subset,
                        "human_labels",
                        "machine_labels",
                        bin_type,
                        group_dict,
                        "pointwise",
                        bin_name,
                    )
                )

    if not df_pair.empty:
        group_cols = ["dataset", "task", "model", "config", "criterion"]
        df_pair = df_pair.copy()
        df_pair["_human_votes_int"] = df_pair["human_votes"].apply(
            lambda vs: [_VOTE_MAP.get(v, 2) for v in vs if v]
        )
        df_pair["_llm_votes_int"] = df_pair["llm_votes"].apply(
            lambda vs: [_VOTE_MAP.get(v, 2) for v in vs if v]
        )
        for group_vals, gdf in df_pair.groupby(group_cols, dropna=False):
            group_dict = dict(zip(group_cols, group_vals))

            jobs.append(
                (
                    gdf,
                    "_human_votes_int",
                    "_llm_votes_int",
                    "majority",
                    group_dict,
                    "pairwise",
                    "overall",
                )
            )
            for bin_name in ("high", "medium", "low"):
                subset = gdf[gdf["consensus_bin"] == bin_name]
                jobs.append(
                    (
                        subset,
                        "_human_votes_int",
                        "_llm_votes_int",
                        "majority",
                        group_dict,
                        "pairwise",
                        bin_name,
                    )
                )

    results = Parallel(n_jobs=-1)(
        delayed(_run_jsd_for_subgroup)(*job, n_resamples) for job in jobs
    )
    rows = [r for r in results if r is not None]
    return pd.DataFrame(rows)
