from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import MetricsConfig
from .human_threshold import (
    compute_human_accuracy_pairwise_threshold,
    compute_human_accuracy_pointwise_threshold,
    compute_human_mae_threshold,
)
from .model_comparison import (
    EXCLUDED_DATASETS,
    SESOI,
    _ac1_value,
    _ac2_value,
    _as_bool,
    _cluster_bootstrap,
    _gwet,
    _prepare,
    _wilcoxon_paired,
    apply_fdr,
)

_VOTE_CATEGORIES = ["A", "B", "TIE"]


def _drop_excluded(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "dataset" not in df.columns:
        return df
    return df[~df["dataset"].isin(EXCLUDED_DATASETS)]


def _cluster_bootstrap_se(
    frame: pd.DataFrame,
    stat_fn,
    n_resamples: int = 2000,
    seed: int = 42,
) -> float:
    if frame.empty:
        return np.nan
    codes, _ = pd.factorize(frame["_cluster"])
    n_clusters = int(codes.max()) + 1 if len(codes) else 0
    if n_clusters < 3:
        return np.nan

    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.searchsorted(sorted_codes, np.arange(n_clusters + 1))
    idx_by_cluster = [
        order[boundaries[i] : boundaries[i + 1]] for i in range(n_clusters)
    ]

    rng = np.random.default_rng(seed)
    boots = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        pick = rng.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([idx_by_cluster[i] for i in pick])
        try:
            boots[b] = stat_fn(frame.take(idx))
        except Exception:
            boots[b] = np.nan

    boots = boots[np.isfinite(boots)]
    if len(boots) < 20:
        return np.nan
    return float(np.std(boots, ddof=1))


def _pool_hh(hh_rows: pd.DataFrame, value_col: str) -> tuple[float, float]:
    se_col = f"{value_col}_se"
    sub = hh_rows.dropna(subset=[value_col, se_col])
    sub = sub[sub[se_col] > 0]
    if sub.empty:
        return np.nan, np.nan
    w = 1.0 / (sub[se_col].to_numpy() ** 2)
    v_pooled = float(np.sum(w * sub[value_col].to_numpy()) / np.sum(w))
    se_pooled = float(np.sqrt(1.0 / np.sum(w)))
    return v_pooled, se_pooled


def _pointwise_groups(df: pd.DataFrame, config_obj: MetricsConfig):
    for (dataset, cfg, model, criterion), gdf in df.groupby(
        ["dataset", "config", "model", "criterion"], dropna=False
    ):
        yield (
            dataset,
            cfg,
            model,
            criterion,
            gdf,
            criterion in config_obj.binary_criteria,
            [criterion],
        )

    for (dataset, cfg, model), gdf in df.groupby(
        ["dataset", "config", "model"], dropna=False
    ):
        is_bin = gdf["criterion"].isin(config_obj.binary_criteria)
        if is_bin.any():
            sub = gdf[is_bin]
            yield (
                dataset,
                cfg,
                model,
                "__macro_binary__",
                sub,
                True,
                sorted(sub["criterion"].unique().tolist()),
            )
        if (~is_bin).any():
            sub = gdf[~is_bin]
            yield (
                dataset,
                cfg,
                model,
                "__macro_ordinal__",
                sub,
                False,
                sorted(sub["criterion"].unique().tolist()),
            )


def _pairwise_groups(df: pd.DataFrame):
    for (dataset, cfg, model, criterion), gdf in df.groupby(
        ["dataset", "config", "model", "criterion"], dropna=False
    ):
        yield dataset, cfg, model, criterion, gdf, [criterion]

    for (dataset, cfg, model), gdf in df.groupby(
        ["dataset", "config", "model"], dropna=False
    ):
        yield (
            dataset,
            cfg,
            model,
            "__macro__",
            gdf,
            sorted(gdf["criterion"].unique().tolist()),
        )


def _bias_pointwise(
    item_pw: pd.DataFrame, config_obj: MetricsConfig, n_resamples: int
) -> pd.DataFrame:
    if item_pw.empty:
        return pd.DataFrame()

    df = _prepare(item_pw, ["id", "criterion"])

    rows: list[dict] = []
    for dataset, cfg, model, criterion, gdf, is_binary, _ in _pointwise_groups(
        df, config_obj
    ):
        if is_binary:
            continue
        sub = gdf.dropna(subset=["human_median", "machine_median"])
        if len(sub) < 2:
            continue

        diff = sub["machine_median"].to_numpy(dtype=float) - sub[
            "human_median"
        ].to_numpy(dtype=float)
        sub = sub.assign(_diff=diff)

        stats = _wilcoxon_paired(diff, higher_is_better=True)
        ci = _cluster_bootstrap(sub, lambda d: float(d["_diff"].mean()), n_resamples)
        p_raw = stats["p_raw"] if np.isfinite(stats["p_raw"]) else ci["boot_p"]

        rows.append(
            {
                "paradigm": "pointwise",
                "dataset": dataset,
                "config": cfg,
                "model": model,
                "criterion": criterion,
                "metric": "bias",
                "n_items": len(sub),
                "n_clusters": sub["_cluster"].nunique(),
                "mean_bias": float(diff.mean()),
                "n_model_higher": stats["n_favor_b"],
                "n_model_lower": stats["n_favor_a"],
                "n_discordant": stats["n_discordant"],
                "wilcoxon_statistic": stats["statistic"],
                "wilcoxon_p": stats["p_raw"],
                "sign_p": stats["sign_p"],
                "ci_lo": ci["ci_lo"],
                "ci_hi": ci["ci_hi"],
                "bootstrap_p": ci["boot_p"],
                "p_raw": p_raw,
            }
        )

    return pd.DataFrame(rows)


def _mismatch_row(
    sub: pd.DataFrame, mismatch: np.ndarray, common: dict, n_resamples: int
) -> dict:
    sub = sub.assign(_mismatch=mismatch.astype(float))
    ci = _cluster_bootstrap(sub, lambda d: float(d["_mismatch"].mean()), n_resamples)
    return {
        **common,
        "metric": "mismatch_rate",
        "n_items": len(sub),
        "n_clusters": sub["_cluster"].nunique(),
        "mismatch_rate": float(mismatch.mean()),
        "ci_lo": ci["ci_lo"],
        "ci_hi": ci["ci_hi"],
        "ci90_lo": ci["ci90_lo"],
        "ci90_hi": ci["ci90_hi"],
        "ci_excludes_zero": bool(np.isfinite(ci["ci_lo"]) and ci["ci_lo"] > 0),
    }


def _mismatch_pointwise(
    item_pw: pd.DataFrame, config_obj: MetricsConfig, n_resamples: int
) -> pd.DataFrame:
    if item_pw.empty:
        return pd.DataFrame()

    df = _prepare(item_pw, ["id", "criterion"])
    rows: list[dict] = []
    for dataset, cfg, model, criterion, gdf, is_binary, _ in _pointwise_groups(
        df, config_obj
    ):
        sub = gdf.dropna(subset=["human_median", "machine_median"])
        if len(sub) < 2:
            continue
        h = sub["human_median"].to_numpy(dtype=float)
        m = sub["machine_median"].to_numpy(dtype=float)
        mismatch = (np.round(h) != np.round(m)) if is_binary else (h != m)
        common = {
            "paradigm": "pointwise",
            "dataset": dataset,
            "config": cfg,
            "model": model,
            "criterion": criterion,
        }
        rows.append(_mismatch_row(sub, mismatch, common, n_resamples))

    return pd.DataFrame(rows)


def _mismatch_pairwise(item_pair: pd.DataFrame, n_resamples: int) -> pd.DataFrame:
    if item_pair.empty:
        return pd.DataFrame()

    df = _prepare(item_pair, ["id", "criterion"])
    rows: list[dict] = []
    for dataset, cfg, model, criterion, gdf, _ in _pairwise_groups(df):
        sub = gdf.dropna(subset=["human_majority", "llm_majority"])
        if len(sub) < 2:
            continue
        mismatch = (
            sub["human_majority"].astype(str) != sub["llm_majority"].astype(str)
        ).to_numpy()
        common = {
            "paradigm": "pairwise",
            "dataset": dataset,
            "config": cfg,
            "model": model,
            "criterion": criterion,
        }
        rows.append(_mismatch_row(sub, mismatch, common, n_resamples))

    return pd.DataFrame(rows)


def _mismatch_passfail(item_pf: pd.DataFrame, n_resamples: int) -> pd.DataFrame:
    if item_pf.empty:
        return pd.DataFrame()

    df = _prepare(item_pf, ["id", "section"])
    rows: list[dict] = []
    for (dataset, cfg, model), gdf in df.groupby(
        ["dataset", "config", "model"], dropna=False
    ):
        sub = gdf.dropna(subset=["human_majority", "llm_majority"])
        if len(sub) < 2:
            continue
        mismatch = (
            sub["human_majority"].astype(str) != sub["llm_majority"].astype(str)
        ).to_numpy()
        common = {
            "paradigm": "passfail",
            "dataset": dataset,
            "config": cfg,
            "model": model,
            "criterion": "clinical_harm_rate",
        }
        rows.append(_mismatch_row(sub, mismatch, common, n_resamples))

    return pd.DataFrame(rows)


def _gap_row(
    v_model: float,
    se_model: float,
    v_hh: float,
    se_hh: float,
    common: dict,
    n_items: int,
    n_clusters: int,
) -> dict:
    z = (v_model - v_hh) / np.sqrt(se_model**2 + se_hh**2)
    p = float(2 * (1 - norm.cdf(abs(z))))
    delta = v_model - v_hh
    sesoi = SESOI.get("ac", 0.05)
    if abs(delta) < sesoi:
        verdict = "equivalent"
    elif p < 0.05:
        verdict = "human_better" if delta < 0 else "model_better"
    else:
        verdict = "inconclusive"
    return {
        **common,
        "n_items": n_items,
        "n_clusters": n_clusters,
        "value_model": v_model,
        "se_model": se_model,
        "value_hh": v_hh,
        "se_hh": se_hh,
        "delta": delta,
        "z": float(z),
        "p_raw": p,
        "verdict": verdict,
    }


def _gap_pointwise(
    item_pw: pd.DataFrame,
    config_obj: MetricsConfig,
    hh_baseline: pd.DataFrame,
    n_resamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if item_pw.empty or hh_baseline.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = _prepare(item_pw, ["id", "criterion"])
    rows: list[dict] = []
    skipped: list[dict] = []
    for (
        dataset,
        cfg,
        model,
        criterion,
        gdf,
        is_binary,
        criteria_list,
    ) in _pointwise_groups(df, config_obj):
        sub = gdf.dropna(subset=["human_median", "machine_median"])
        if len(sub) < 2:
            continue
        metric = "gwet_ac1" if is_binary else "gwet_ac2"
        hh_key = "gwet_ac1_hh" if is_binary else "gwet_ac2_hh"

        hh_rows = hh_baseline[
            (hh_baseline["dataset"] == dataset)
            & (hh_baseline["criterion"].isin(criteria_list))
        ]
        v_hh, se_hh = (
            _pool_hh(hh_rows, hh_key)
            if hh_key in hh_baseline.columns
            else (
                np.nan,
                np.nan,
            )
        )
        common = {
            "paradigm": "pointwise",
            "dataset": dataset,
            "config": cfg,
            "model": model,
            "criterion": criterion,
            "metric": metric,
        }
        if not (np.isfinite(v_hh) and np.isfinite(se_hh)):
            skipped.append({**common, "reason": "no human-human ceiling"})
            continue

        h = sub["human_median"].to_numpy(dtype=float)
        m = sub["machine_median"].to_numpy(dtype=float)
        fn = _ac1_value if is_binary else _ac2_value
        v_model = fn(h, m)
        if not np.isfinite(v_model):
            continue
        sub2 = sub.assign(_h=h, _m=m)
        se_model = _cluster_bootstrap_se(
            sub2, lambda d: fn(d["_h"].to_numpy(), d["_m"].to_numpy()), n_resamples
        )
        if not np.isfinite(se_model):
            skipped.append(
                {**common, "reason": "bootstrap SE undefined (too few clusters)"}
            )
            continue

        rows.append(
            _gap_row(
                v_model,
                se_model,
                v_hh,
                se_hh,
                common,
                len(sub),
                sub["_cluster"].nunique(),
            )
        )

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def _gap_pairwise(
    item_pair: pd.DataFrame, hh_baseline: pd.DataFrame, n_resamples: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if item_pair.empty or hh_baseline.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = _prepare(item_pair, ["id", "criterion"])
    vote_to_idx = {c: i for i, c in enumerate(_VOTE_CATEGORIES)}
    cats = list(range(len(_VOTE_CATEGORIES)))

    def _ac(h, m):
        return _gwet(h, m, cats, "identity")

    rows: list[dict] = []
    skipped: list[dict] = []
    for dataset, cfg, model, criterion, gdf, criteria_list in _pairwise_groups(df):
        sub = gdf.dropna(subset=["human_majority", "llm_majority"])
        if len(sub) < 2:
            continue

        hh_rows = hh_baseline[
            (hh_baseline["dataset"] == dataset)
            & (hh_baseline["criterion"].isin(criteria_list))
        ]
        v_hh, se_hh = (
            _pool_hh(hh_rows, "gwet_ac1_hh")
            if "gwet_ac1_hh" in hh_baseline.columns
            else (np.nan, np.nan)
        )
        common = {
            "paradigm": "pairwise",
            "dataset": dataset,
            "config": cfg,
            "model": model,
            "criterion": criterion,
            "metric": "gwet_ac1",
        }
        if not (np.isfinite(v_hh) and np.isfinite(se_hh)):
            skipped.append({**common, "reason": "no human-human ceiling"})
            continue

        h = sub["human_majority"].map(vote_to_idx).to_numpy(dtype=float)
        m = sub["llm_majority"].map(vote_to_idx).to_numpy(dtype=float)
        ok = np.isfinite(h) & np.isfinite(m)
        if ok.sum() < 2:
            continue
        sub2 = sub[ok].assign(_h=h[ok], _m=m[ok])
        v_model = _ac(h[ok], m[ok])
        if not np.isfinite(v_model):
            continue
        se_model = _cluster_bootstrap_se(
            sub2, lambda d: _ac(d["_h"].to_numpy(), d["_m"].to_numpy()), n_resamples
        )
        if not np.isfinite(se_model):
            skipped.append(
                {**common, "reason": "bootstrap SE undefined (too few clusters)"}
            )
            continue

        rows.append(
            _gap_row(
                v_model,
                se_model,
                v_hh,
                se_hh,
                common,
                int(ok.sum()),
                sub2["_cluster"].nunique(),
            )
        )

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def _gap_passfail(
    item_pf: pd.DataFrame, hh_baseline: pd.DataFrame, n_resamples: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if item_pf.empty or hh_baseline.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = _prepare(item_pf, ["id", "section"])
    rows: list[dict] = []
    skipped: list[dict] = []
    for (dataset, cfg, model), gdf in df.groupby(
        ["dataset", "config", "model"], dropna=False
    ):
        sub = gdf.dropna(subset=["human_majority", "llm_majority"])
        if len(sub) < 2:
            continue

        hh_rows = hh_baseline[
            (hh_baseline["dataset"] == dataset)
            & (hh_baseline["criterion"] == "clinical_harm_rate")
        ]
        v_hh, se_hh = (
            _pool_hh(hh_rows, "gwet_ac1_hh")
            if "gwet_ac1_hh" in hh_baseline.columns
            else (np.nan, np.nan)
        )
        common = {
            "paradigm": "passfail",
            "dataset": dataset,
            "config": cfg,
            "model": model,
            "criterion": "clinical_harm_rate",
            "metric": "gwet_ac1",
        }
        if not (np.isfinite(v_hh) and np.isfinite(se_hh)):
            skipped.append({**common, "reason": "no human-human ceiling"})
            continue

        h = (sub["human_majority"] == "FAIL").to_numpy(dtype=float)
        m = (sub["llm_majority"] == "FAIL").to_numpy(dtype=float)
        v_model = _ac1_value(h, m)
        if not np.isfinite(v_model):
            continue
        sub2 = sub.assign(_h=h, _m=m)
        se_model = _cluster_bootstrap_se(
            sub2,
            lambda d: _ac1_value(d["_h"].to_numpy(), d["_m"].to_numpy()),
            n_resamples,
        )
        if not np.isfinite(se_model):
            skipped.append(
                {**common, "reason": "bootstrap SE undefined (too few clusters)"}
            )
            continue

        rows.append(
            _gap_row(
                v_model,
                se_model,
                v_hh,
                se_hh,
                common,
                len(sub),
                sub["_cluster"].nunique(),
            )
        )

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def _threshold_row(
    v_model: float,
    se_model: float,
    v_hh: float,
    se_hh: float,
    common: dict,
    n_items: int,
    n_clusters: int,
    worse_is: str,
) -> dict:
    se = np.sqrt(se_model**2 + se_hh**2)
    z = (v_model - v_hh) / se if se > 0 else np.nan
    if not np.isfinite(z):
        p = np.nan
    elif worse_is == "higher":
        p = float(1.0 - norm.cdf(z))
    else:
        p = float(norm.cdf(z))
    return {
        **common,
        "n_items": n_items,
        "n_clusters": n_clusters,
        "value_model": v_model,
        "se_model": se_model,
        "value_human": v_hh,
        "se_human": se_hh,
        "delta": v_model - v_hh,
        "worse_is": worse_is,
        "z": float(z) if np.isfinite(z) else np.nan,
        "p_raw": p,
    }


def _threshold_gap_pointwise(
    item_pw: pd.DataFrame,
    config_obj: MetricsConfig,
    hh_mae: pd.DataFrame,
    hh_acc: pd.DataFrame,
    n_resamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if item_pw.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = _prepare(item_pw, ["id", "criterion"])
    rows: list[dict] = []
    skipped: list[dict] = []
    for (
        dataset,
        cfg,
        model,
        criterion,
        gdf,
        is_binary,
        criteria_list,
    ) in _pointwise_groups(df, config_obj):
        sub = gdf.dropna(subset=["human_median", "machine_median"])
        if len(sub) < 2:
            continue
        metric = "accuracy" if is_binary else "mae"
        hh_source = hh_acc if is_binary else hh_mae
        common = {
            "paradigm": "pointwise",
            "dataset": dataset,
            "config": cfg,
            "model": model,
            "criterion": criterion,
            "metric": metric,
        }
        if hh_source.empty:
            skipped.append({**common, "reason": "no human LOO threshold"})
            continue
        hh_rows = hh_source[
            (hh_source["dataset"] == dataset)
            & (hh_source["criterion"].isin(criteria_list))
        ]
        v_hh, se_hh = _pool_hh(hh_rows, "mean")
        if not (np.isfinite(v_hh) and np.isfinite(se_hh)):
            skipped.append({**common, "reason": "no human LOO threshold"})
            continue

        h = sub["human_median"].to_numpy(dtype=float)
        m = sub["machine_median"].to_numpy(dtype=float)

        def _acc_stat(hh: np.ndarray, mm: np.ndarray) -> float:
            return float(np.mean(np.round(hh) == np.round(mm)))

        def _mae_stat(hh: np.ndarray, mm: np.ndarray) -> float:
            return float(np.mean(np.abs(hh - mm)))

        stat_fn = _acc_stat if is_binary else _mae_stat
        worse_is = "lower" if is_binary else "higher"
        v_model = stat_fn(h, m)
        sub2 = sub.assign(_h=h, _m=m)
        se_model = _cluster_bootstrap_se(
            sub2, lambda d: stat_fn(d["_h"].to_numpy(), d["_m"].to_numpy()), n_resamples
        )
        if not np.isfinite(se_model):
            skipped.append(
                {**common, "reason": "bootstrap SE undefined (too few clusters)"}
            )
            continue

        rows.append(
            _threshold_row(
                v_model,
                se_model,
                v_hh,
                se_hh,
                common,
                len(sub),
                sub["_cluster"].nunique(),
                worse_is,
            )
        )

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def _threshold_gap_pairwise(
    item_pair: pd.DataFrame, hh_acc_pair: pd.DataFrame, n_resamples: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if item_pair.empty or hh_acc_pair.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = _prepare(item_pair, ["id", "criterion"])
    rows: list[dict] = []
    skipped: list[dict] = []
    for dataset, cfg, model, criterion, gdf, criteria_list in _pairwise_groups(df):
        sub = gdf.dropna(subset=["human_majority", "llm_majority"])
        if "is_clear_majority" in sub.columns:
            sub = sub[_as_bool(sub["is_clear_majority"])]
        if len(sub) < 2:
            continue

        common = {
            "paradigm": "pairwise",
            "dataset": dataset,
            "config": cfg,
            "model": model,
            "criterion": criterion,
            "metric": "accuracy_clear",
        }
        hh_rows = hh_acc_pair[
            (hh_acc_pair["dataset"] == dataset)
            & (hh_acc_pair["criterion"].isin(criteria_list))
        ]
        v_hh, se_hh = _pool_hh(hh_rows, "mean")
        if not (np.isfinite(v_hh) and np.isfinite(se_hh)):
            skipped.append({**common, "reason": "no human LOO threshold"})
            continue

        match = (
            sub["human_majority"].astype(str) == sub["llm_majority"].astype(str)
        ).to_numpy(dtype=float)
        v_model = float(match.mean())
        sub2 = sub.assign(_m=match)
        se_model = _cluster_bootstrap_se(
            sub2, lambda d: float(d["_m"].to_numpy().mean()), n_resamples
        )
        if not np.isfinite(se_model):
            skipped.append(
                {**common, "reason": "bootstrap SE undefined (too few clusters)"}
            )
            continue

        rows.append(
            _threshold_row(
                v_model,
                se_model,
                v_hh,
                se_hh,
                common,
                len(sub),
                sub["_cluster"].nunique(),
                "lower",
            )
        )

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def compare_to_human(
    item_pw: pd.DataFrame,
    item_pair: pd.DataFrame,
    item_pf: pd.DataFrame,
    hh_baseline: pd.DataFrame,
    config_obj: MetricsConfig | None = None,
    n_resamples: int = 2000,
) -> dict[str, pd.DataFrame]:
    config_obj = config_obj or MetricsConfig()
    hh_baseline = hh_baseline if hh_baseline is not None else pd.DataFrame()

    item_pw = _drop_excluded(item_pw)
    item_pair = _drop_excluded(item_pair)
    item_pf = _drop_excluded(item_pf)

    bias = _bias_pointwise(item_pw, config_obj, n_resamples)
    mismatch_pw = _mismatch_pointwise(item_pw, config_obj, n_resamples)
    mismatch_pair = _mismatch_pairwise(item_pair, n_resamples)
    mismatch_pf = _mismatch_passfail(item_pf, n_resamples)
    gap_pw, skip_pw = _gap_pointwise(item_pw, config_obj, hh_baseline, n_resamples)
    gap_pair, skip_pair = _gap_pairwise(item_pair, hh_baseline, n_resamples)
    gap_pf, skip_pf = _gap_passfail(item_pf, hh_baseline, n_resamples)

    hh_mae = compute_human_mae_threshold(item_pw, config_obj)
    hh_acc_pw = compute_human_accuracy_pointwise_threshold(item_pw, config_obj)
    hh_acc_pair = compute_human_accuracy_pairwise_threshold(item_pair)
    thresh_pw, skip_thresh_pw = _threshold_gap_pointwise(
        item_pw, config_obj, hh_mae, hh_acc_pw, n_resamples
    )
    thresh_pair, skip_thresh_pair = _threshold_gap_pairwise(
        item_pair, hh_acc_pair, n_resamples
    )

    if not bias.empty:
        bias = apply_fdr(bias, family_cols=("dataset", "config", "metric"))

    if not gap_pw.empty:
        gap_pw = apply_fdr(gap_pw, family_cols=("dataset", "config", "metric"))
    if not gap_pair.empty:
        gap_pair = apply_fdr(gap_pair, family_cols=("dataset", "config", "metric"))
    if not gap_pf.empty:
        gap_pf = apply_fdr(gap_pf, family_cols=("dataset", "config", "metric"))

    if not thresh_pw.empty:
        thresh_pw = apply_fdr(thresh_pw, family_cols=("dataset", "config", "metric"))
        thresh_pw["verdict"] = np.where(
            thresh_pw["p_fdr"] < 0.05, "worse_than_human", "not_significant"
        )
    if not thresh_pair.empty:
        thresh_pair = apply_fdr(
            thresh_pair, family_cols=("dataset", "config", "metric")
        )
        thresh_pair["verdict"] = np.where(
            thresh_pair["p_fdr"] < 0.05, "worse_than_human", "not_significant"
        )

    integrity = (
        pd.concat(
            [s for s in (skip_pw, skip_pair, skip_pf) if not s.empty], ignore_index=True
        )
        if any(not s.empty for s in (skip_pw, skip_pair, skip_pf))
        else pd.DataFrame()
    )
    threshold_integrity = (
        pd.concat(
            [s for s in (skip_thresh_pw, skip_thresh_pair) if not s.empty],
            ignore_index=True,
        )
        if any(not s.empty for s in (skip_thresh_pw, skip_thresh_pair))
        else pd.DataFrame()
    )

    return {
        "human_comparison_pointwise_bias": bias,
        "human_comparison_pointwise_mismatch": mismatch_pw,
        "human_comparison_pairwise_mismatch": mismatch_pair,
        "human_comparison_passfail_mismatch": mismatch_pf,
        "human_comparison_pointwise": gap_pw,
        "human_comparison_pairwise": gap_pair,
        "human_comparison_passfail": gap_pf,
        "human_comparison_integrity": integrity,
        "human_comparison_pointwise_threshold": thresh_pw,
        "human_comparison_pairwise_threshold": thresh_pair,
        "human_comparison_threshold_integrity": threshold_integrity,
    }
