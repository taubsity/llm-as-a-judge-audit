from __future__ import annotations

import json
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binomtest, rankdata, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar as _mcnemar_test
from statsmodels.stats.multitest import multipletests

from .config import MetricsConfig
from .pairwise import _median_unbiased_or

SESOI: dict[str, float] = {
    "mae": 0.25,
    "prop": 0.05,
    "ac": 0.05,
}

METRIC_SPEC: dict[str, tuple[str, bool]] = {
    "mae": ("mae", False),
    "exact_match": ("prop", True),
    "accuracy": ("prop", True),
    "accuracy_clear": ("prop", True),
    "gwet_ac1": ("ac", True),
    "gwet_ac2": ("ac", True),
    "ipi": ("prop", False),
    "clinical_harm_rate": ("prop", False),
    "recall": ("prop", True),
    "precision": ("prop", True),
    "f1": ("prop", True),
    "seed_std": ("aux", False),
    "seed_mode_share": ("prop", True),
    "seed_variability": ("prop", False),
}

OUTPUT_COLUMNS = [
    "paradigm",
    "model_a",
    "model_b",
    "dataset",
    "config",
    "criterion",
    "metric",
    "n_items",
    "n_clusters",
    "n_discordant",
    "n_favor_a",
    "n_favor_b",
    "value_a",
    "value_b",
    "delta",
    "ci_lo",
    "ci_hi",
    "ci90_lo",
    "ci90_hi",
    "test",
    "statistic",
    "p_raw",
    "p_fdr",
    "effect_size",
    "effect_size_name",
    "higher_is_better",
    "sesoi",
    "favors",
    "verdict",
]

_ORDINAL_CATEGORIES = list(range(1, 6))
_VOTE_CATEGORIES = ["A", "B", "TIE"]

EXCLUDED_DATASETS = frozenset({"mtb_nps", "mtb_ppe", "mtb_errors_detailed"})


def _drop_excluded(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "dataset" not in df.columns:
        return df
    return df[~df["dataset"].isin(EXCLUDED_DATASETS)]


def _parse_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def _prepare(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    if "task" not in out.columns:
        out["task"] = ""
    out["task"] = out["task"].fillna("")

    identifying = {
        dataset: bool(gdf.duplicated(subset=["model", "config", *base_cols]).any())
        for dataset, gdf in out.groupby("dataset", dropna=False)
    }
    out["_task_key"] = np.where(
        out["dataset"].map(identifying).fillna(False).to_numpy(),
        out["task"].astype(str),
        "",
    )
    out["_cluster"] = (
        out["dataset"].astype(str)
        + "||"
        + out["_task_key"].astype(str)
        + "||"
        + out["id"].astype(str)
    )
    return out


def check_pairing(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    key_cols: list[str],
    paradigm: str,
    human_col: str | None = None,
    machine_col: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(columns=["paradigm", "dataset", "config", "status"])

    for (dataset, config), gdf in df.groupby(["dataset", "config"], dropna=False):
        a = gdf[gdf["model"] == model_a]
        b = gdf[gdf["model"] == model_b]
        if a.empty or b.empty:
            continue

        keys_a = set(map(tuple, a[key_cols].astype(str).to_numpy()))
        keys_b = set(map(tuple, b[key_cols].astype(str).to_numpy()))
        shared = keys_a & keys_b

        fatal: list[str] = []
        advisory: list[str] = []

        if keys_a - keys_b:
            fatal.append(f"{len(keys_a - keys_b)} keys only in {model_a}")
        if keys_b - keys_a:
            fatal.append(f"{len(keys_b - keys_a)} keys only in {model_b}")

        n_dup_a = len(a) - len(keys_a)
        n_dup_b = len(b) - len(keys_b)
        if n_dup_a:
            fatal.append(f"{n_dup_a} duplicate keys in {model_a}")
        if n_dup_b:
            fatal.append(f"{n_dup_b} duplicate keys in {model_b}")

        n_human_mismatch = 0
        if (
            human_col is not None
            and human_col in gdf.columns
            and not (n_dup_a or n_dup_b)
        ):
            ha = a.set_index([*key_cols])[human_col].astype(str)
            hb = b.set_index([*key_cols])[human_col].astype(str)
            shared_idx = ha.index.intersection(hb.index)
            if len(shared_idx):
                n_human_mismatch = int((ha.loc[shared_idx] != hb.loc[shared_idx]).sum())
            if n_human_mismatch:
                fatal.append(f"{n_human_mismatch} human-label mismatches")

        seed_a = seed_b = np.nan
        if machine_col is not None and machine_col in gdf.columns:
            uniq_a = sorted({len(_parse_list(v)) for v in a[machine_col]})
            uniq_b = sorted({len(_parse_list(v)) for v in b[machine_col]})
            seed_a = uniq_a[0] if len(uniq_a) == 1 else -1
            seed_b = uniq_b[0] if len(uniq_b) == 1 else -1
            if len(uniq_a) > 1:
                advisory.append(f"{model_a} seed counts not uniform: {uniq_a}")
            if len(uniq_b) > 1:
                advisory.append(f"{model_b} seed counts not uniform: {uniq_b}")
            if max(uniq_a) != max(uniq_b):
                fatal.append(
                    f"seed count differs: {model_a}={uniq_a}, {model_b}={uniq_b}"
                )

        problems = fatal + advisory

        rows.append(
            {
                "paradigm": paradigm,
                "dataset": dataset,
                "config": config,
                "model_a": model_a,
                "model_b": model_b,
                "n_keys_a": len(keys_a),
                "n_keys_b": len(keys_b),
                "n_shared": len(shared),
                "n_only_a": len(keys_a - keys_b),
                "n_only_b": len(keys_b - keys_a),
                "n_human_mismatch": n_human_mismatch,
                "n_seeds_a": seed_a,
                "n_seeds_b": seed_b,
                "status": "failed" if fatal else ("warn" if advisory else "ok"),
                "note": "; ".join(problems),
            }
        )

    return pd.DataFrame(rows)


def _usable_strata(report: pd.DataFrame) -> set[tuple[str, str]]:
    if report.empty:
        return set()
    ok = report[report["status"].isin({"ok", "warn"})]
    return set(zip(ok["dataset"], ok["config"]))


def _empty_ci() -> dict[str, float]:
    return {
        "ci_lo": np.nan,
        "ci_hi": np.nan,
        "ci90_lo": np.nan,
        "ci90_hi": np.nan,
        "boot_p": np.nan,
    }


def _cluster_bootstrap(
    frame: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    if frame.empty:
        return _empty_ci()

    codes, _ = pd.factorize(frame["_cluster"])
    n_clusters = int(codes.max()) + 1 if len(codes) else 0
    if n_clusters < 3:
        return _empty_ci()

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
        return _empty_ci()

    ci_lo, ci_hi, ci90_lo, ci90_hi = np.percentile(boots, [2.5, 97.5, 5.0, 95.0])
    p = 2.0 * min(float((boots >= 0).mean()), float((boots <= 0).mean()))
    return {
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "ci90_lo": float(ci90_lo),
        "ci90_hi": float(ci90_hi),
        "boot_p": float(min(1.0, p)),
    }


def _wilcoxon_paired(
    diff: np.ndarray, higher_is_better: bool = False
) -> dict[str, float]:
    diff = diff[np.isfinite(diff)]
    nz = diff[diff != 0]
    n_pos = int((diff > 0).sum())
    n_neg = int((diff < 0).sum())
    out = {
        "statistic": np.nan,
        "p_raw": np.nan,
        "effect_size": np.nan,
        "n_favor_a": n_neg if higher_is_better else n_pos,
        "n_favor_b": n_pos if higher_is_better else n_neg,
        "n_discordant": int(len(nz)),
        "sign_p": np.nan,
    }
    if len(nz) < 1:
        out["p_raw"] = 1.0
        out["sign_p"] = 1.0
        out["effect_size"] = 0.0
        return out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = wilcoxon(nz, alternative="two-sided", zero_method="wilcox")
            out["statistic"] = float(res.statistic)
            out["p_raw"] = float(res.pvalue)
        except ValueError:
            pass

    ranks = rankdata(np.abs(nz))
    w_pos = float(ranks[nz > 0].sum())
    w_neg = float(ranks[nz < 0].sum())
    if w_pos + w_neg > 0:
        out["effect_size"] = (w_pos - w_neg) / (w_pos + w_neg)

    out["sign_p"] = float(binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue)
    return out


def _exact_mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    correct_a = correct_a.astype(bool)
    correct_b = correct_b.astype(bool)
    n_a_only = int((correct_a & ~correct_b).sum())
    n_b_only = int((~correct_a & correct_b).sum())
    n_both = int((correct_a & correct_b).sum())
    n_neither = int((~correct_a & ~correct_b).sum())

    out = {
        "statistic": np.nan,
        "p_raw": 1.0,
        "effect_size": np.nan,
        "n_favor_a": n_a_only,
        "n_favor_b": n_b_only,
        "n_discordant": n_a_only + n_b_only,
    }
    if n_a_only + n_b_only == 0:
        return out

    table = np.array([[n_both, n_a_only], [n_b_only, n_neither]])
    try:
        res = _mcnemar_test(table, exact=True)
        out["statistic"] = float(res.statistic)
        out["p_raw"] = float(res.pvalue)
    except Exception:
        pass
    out["effect_size"] = float(_median_unbiased_or(n_b_only, n_a_only))
    return out


def _make_row(
    *,
    paradigm: str,
    model_a: str,
    model_b: str,
    dataset: str,
    config: str,
    criterion: str,
    metric: str,
    n_items: int,
    n_clusters: int,
    value_a: float,
    value_b: float,
    test: str,
    ci: dict[str, float],
    stats: dict[str, float] | None = None,
    effect_size_name: str = "",
    p_override: float | None = None,
) -> dict[str, Any]:
    stats = stats or {}
    kind, higher_is_better = METRIC_SPEC.get(metric, ("aux", True))
    p_raw = p_override if p_override is not None else stats.get("p_raw", np.nan)
    if p_raw is None or (isinstance(p_raw, float) and not np.isfinite(p_raw)):
        p_raw = ci.get("boot_p", np.nan)

    return {
        "paradigm": paradigm,
        "model_a": model_a,
        "model_b": model_b,
        "dataset": dataset,
        "config": config,
        "criterion": criterion,
        "metric": metric,
        "n_items": int(n_items),
        "n_clusters": int(n_clusters),
        "n_discordant": stats.get("n_discordant", np.nan),
        "n_favor_a": stats.get("n_favor_a", np.nan),
        "n_favor_b": stats.get("n_favor_b", np.nan),
        "value_a": float(value_a) if np.isfinite(value_a) else np.nan,
        "value_b": float(value_b) if np.isfinite(value_b) else np.nan,
        "delta": float(value_b - value_a)
        if np.isfinite(value_a) and np.isfinite(value_b)
        else np.nan,
        "ci_lo": ci.get("ci_lo", np.nan),
        "ci_hi": ci.get("ci_hi", np.nan),
        "ci90_lo": ci.get("ci90_lo", np.nan),
        "ci90_hi": ci.get("ci90_hi", np.nan),
        "test": test,
        "statistic": stats.get("statistic", np.nan),
        "p_raw": p_raw,
        "p_fdr": np.nan,
        "effect_size": stats.get("effect_size", np.nan),
        "effect_size_name": effect_size_name,
        "higher_is_better": higher_is_better,
        "sesoi": SESOI.get(kind, np.nan),
        "favors": "",
        "verdict": "",
        "sign_p": stats.get("sign_p", np.nan),
        "bootstrap_p": ci.get("boot_p", np.nan),
    }


def _pair_frame(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    key_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    keep = [*key_cols, *value_cols]
    a = df[df["model"] == model_a][[*keep, "_cluster"]].copy()
    b = df[df["model"] == model_b][keep].copy()
    merged = a.merge(b, on=key_cols, suffixes=("_a", "_b"), how="inner")
    return merged


def _compare_pointwise_criterion(
    gdf: pd.DataFrame,
    dataset: str,
    config: str,
    criterion: str,
    model_a: str,
    model_b: str,
    is_binary: bool,
    n_resamples: int,
) -> list[dict[str, Any]]:
    frame = _pair_frame(
        gdf,
        model_a,
        model_b,
        key_cols=["_task_key", "id", "criterion"],
        value_cols=[
            "human_median",
            "machine_median",
            "machine_seed_std",
            "machine_seed_mode_share",
        ],
    )
    frame = frame.dropna(
        subset=["human_median_a", "machine_median_a", "machine_median_b"]
    )
    if len(frame) < 2:
        return []

    n = len(frame)
    n_clusters = frame["_cluster"].nunique()
    h = frame["human_median_a"].to_numpy()
    ma = frame["machine_median_a"].to_numpy()
    mb = frame["machine_median_b"].to_numpy()

    common = dict(
        paradigm="pointwise",
        model_a=model_a,
        model_b=model_b,
        dataset=dataset,
        config=config,
        criterion=criterion,
        n_items=n,
        n_clusters=n_clusters,
    )
    rows: list[dict[str, Any]] = []

    if is_binary:
        correct_a = np.round(h) == np.round(ma)
        correct_b = np.round(h) == np.round(mb)
        frame = frame.assign(_ca=correct_a.astype(float), _cb=correct_b.astype(float))
        ci = _cluster_bootstrap(
            frame, lambda d: float(d["_cb"].mean() - d["_ca"].mean()), n_resamples
        )
        rows.append(
            _make_row(
                **common,
                metric="accuracy",
                value_a=float(correct_a.mean()),
                value_b=float(correct_b.mean()),
                test="mcnemar_exact",
                ci=ci,
                stats=_exact_mcnemar(correct_a, correct_b),
                effect_size_name="mcnemar_or_b_vs_a",
            )
        )
        rows.append(
            _ac_row(frame, h, ma, mb, common, "gwet_ac1", n_resamples, binary=True)
        )
    else:
        ae_a = np.abs(h - ma)
        ae_b = np.abs(h - mb)
        frame = frame.assign(_ae_a=ae_a, _ae_b=ae_b)
        ci = _cluster_bootstrap(
            frame, lambda d: float((d["_ae_b"] - d["_ae_a"]).mean()), n_resamples
        )
        rows.append(
            _make_row(
                **common,
                metric="mae",
                value_a=float(ae_a.mean()),
                value_b=float(ae_b.mean()),
                test="wilcoxon_signed_rank",
                ci=ci,
                stats=_wilcoxon_paired(ae_b - ae_a),
                effect_size_name="rank_biserial_b_minus_a",
            )
        )

        exact_a = (ae_a == 0).astype(float)
        exact_b = (ae_b == 0).astype(float)
        frame = frame.assign(_xa=exact_a, _xb=exact_b)
        rows.append(
            _make_row(
                **common,
                metric="exact_match",
                value_a=float(exact_a.mean()),
                value_b=float(exact_b.mean()),
                test="mcnemar_exact",
                ci=_cluster_bootstrap(
                    frame,
                    lambda d: float(d["_xb"].mean() - d["_xa"].mean()),
                    n_resamples,
                ),
                stats=_exact_mcnemar(exact_a.astype(bool), exact_b.astype(bool)),
                effect_size_name="mcnemar_or_b_vs_a",
            )
        )
        rows.append(
            _ac_row(frame, h, ma, mb, common, "gwet_ac2", n_resamples, binary=False)
        )

    for metric, col in (
        ("seed_std", "machine_seed_std"),
        ("seed_mode_share", "machine_seed_mode_share"),
    ):
        sub = frame.dropna(subset=[f"{col}_a", f"{col}_b"])
        if len(sub) < 2:
            continue
        d = sub[f"{col}_b"].to_numpy() - sub[f"{col}_a"].to_numpy()
        rows.append(
            _make_row(
                **{
                    **common,
                    "n_items": len(sub),
                    "n_clusters": sub["_cluster"].nunique(),
                },
                metric=metric,
                value_a=float(sub[f"{col}_a"].mean()),
                value_b=float(sub[f"{col}_b"].mean()),
                test="wilcoxon_signed_rank",
                ci=_cluster_bootstrap(
                    sub,
                    lambda x, c=col: float((x[f"{c}_b"] - x[f"{c}_a"]).mean()),
                    n_resamples,
                ),
                stats=_wilcoxon_paired(d, METRIC_SPEC[metric][1]),
                effect_size_name="rank_biserial_b_minus_a",
            )
        )

    return [r for r in rows if r is not None]


def _ac_row(
    frame: pd.DataFrame,
    h: np.ndarray,
    ma: np.ndarray,
    mb: np.ndarray,
    common: dict[str, Any],
    metric: str,
    n_resamples: int,
    *,
    binary: bool,
) -> dict[str, Any] | None:
    fn = _ac1_value if binary else _ac2_value
    frame = frame.assign(_h=h, _ma=ma, _mb=mb)

    def _stat(d: pd.DataFrame) -> float:
        return fn(d["_h"].to_numpy(), d["_mb"].to_numpy()) - fn(
            d["_h"].to_numpy(), d["_ma"].to_numpy()
        )

    va = fn(h, ma)
    vb = fn(h, mb)
    if not (np.isfinite(va) and np.isfinite(vb)):
        return None
    return _make_row(
        **common,
        metric=metric,
        value_a=va,
        value_b=vb,
        test="cluster_bootstrap",
        ci=_cluster_bootstrap(frame, _stat, n_resamples),
        effect_size_name="",
    )


def _gwet(
    r1: np.ndarray, r2: np.ndarray, categories: list[int], weights: str = "identity"
) -> float:
    q = len(categories)
    index = {c: i for i, c in enumerate(categories)}
    try:
        a = np.array([index[int(round(float(x)))] for x in r1])
        b = np.array([index[int(round(float(x)))] for x in r2])
    except (KeyError, ValueError, TypeError):
        return np.nan
    n = len(a)
    if n == 0 or q < 2:
        return np.nan

    if weights == "quadratic":
        grid = np.arange(q, dtype=float)
        dist = (grid[:, None] - grid[None, :]) ** 2
        w = 1.0 - dist / dist.max()
    else:
        w = np.eye(q)

    pa = float(w[a, b].mean())
    counts = np.bincount(a, minlength=q) + np.bincount(b, minlength=q)
    pi = counts / (2.0 * n)
    pe = (w.sum() / (q * (q - 1))) * float(np.sum(pi * (1.0 - pi)))
    if np.isclose(pe, 1.0):
        return np.nan
    return (pa - pe) / (1.0 - pe)


def _ac1_value(h: np.ndarray, m: np.ndarray) -> float:
    return _gwet(h, m, [0, 1], "identity")


def _ac2_value(h: np.ndarray, m: np.ndarray) -> float:
    return _gwet(h, m, _ORDINAL_CATEGORIES, "quadratic")


def compare_pointwise(
    item_pw: pd.DataFrame,
    model_a: str,
    model_b: str,
    config_obj: MetricsConfig,
    n_resamples: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if item_pw.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = _prepare(item_pw, ["id", "criterion"])

    report = check_pairing(
        df,
        model_a,
        model_b,
        key_cols=["_task_key", "id", "criterion"],
        paradigm="pointwise",
        human_col="human_labels",
        machine_col="machine_labels",
    )
    usable = _usable_strata(report)
    if not usable:
        return pd.DataFrame(), report

    jobs = []
    for (dataset, config, criterion), gdf in df.groupby(
        ["dataset", "config", "criterion"], dropna=False
    ):
        if (dataset, config) not in usable:
            continue
        jobs.append(
            (
                gdf,
                dataset,
                config,
                criterion,
                criterion in config_obj.binary_criteria,
            )
        )

    for (dataset, config), gdf in df.groupby(["dataset", "config"], dropna=False):
        if (dataset, config) not in usable:
            continue
        is_bin = gdf["criterion"].isin(config_obj.binary_criteria)
        if is_bin.any():
            jobs.append((gdf[is_bin], dataset, config, "__macro_binary__", True))
        if (~is_bin).any():
            jobs.append((gdf[~is_bin], dataset, config, "__macro_ordinal__", False))

    nested = Parallel(n_jobs=-1)(
        delayed(_compare_pointwise_criterion)(
            gdf, dataset, config, criterion, model_a, model_b, is_binary, n_resamples
        )
        for gdf, dataset, config, criterion, is_binary in jobs
    )
    rows = [r for group in nested for r in group]
    return pd.DataFrame(rows), report


def _compare_pairwise_criterion(
    gdf: pd.DataFrame,
    dataset: str,
    config: str,
    criterion: str,
    model_a: str,
    model_b: str,
    n_resamples: int,
) -> list[dict[str, Any]]:
    frame = _pair_frame(
        gdf,
        model_a,
        model_b,
        key_cols=["_task_key", "id", "criterion"],
        value_cols=[
            "human_majority",
            "llm_majority",
            "llm_majority_swapped",
            "is_clear_majority",
            "_seed_var",
        ],
    )
    if len(frame) < 2:
        return []

    common = dict(
        paradigm="pairwise",
        model_a=model_a,
        model_b=model_b,
        dataset=dataset,
        config=config,
        criterion=criterion,
    )
    rows: list[dict[str, Any]] = []

    clear = frame[_as_bool(frame["is_clear_majority_a"])]
    if len(clear) >= 2:
        ca = (clear["human_majority_a"] == clear["llm_majority_a"]).to_numpy()
        cb = (clear["human_majority_a"] == clear["llm_majority_b"]).to_numpy()
        clear = clear.assign(_ca=ca.astype(float), _cb=cb.astype(float))
        rows.append(
            _make_row(
                **common,
                n_items=len(clear),
                n_clusters=clear["_cluster"].nunique(),
                metric="accuracy_clear",
                value_a=float(ca.mean()),
                value_b=float(cb.mean()),
                test="mcnemar_exact",
                ci=_cluster_bootstrap(
                    clear,
                    lambda d: float(d["_cb"].mean() - d["_ca"].mean()),
                    n_resamples,
                ),
                stats=_exact_mcnemar(ca, cb),
                effect_size_name="mcnemar_or_b_vs_a",
            )
        )

    vote_to_idx = {c: i for i, c in enumerate(_VOTE_CATEGORIES)}
    h = frame["human_majority_a"].map(vote_to_idx).to_numpy(dtype=float)
    ma = frame["llm_majority_a"].map(vote_to_idx).to_numpy(dtype=float)
    mb = frame["llm_majority_b"].map(vote_to_idx).to_numpy(dtype=float)
    ok = np.isfinite(h) & np.isfinite(ma) & np.isfinite(mb)
    if ok.sum() >= 2:
        sub = frame[ok].assign(_h=h[ok], _ma=ma[ok], _mb=mb[ok])
        cats = list(range(len(_VOTE_CATEGORIES)))

        def _ac(hh, mm):
            return _gwet(hh, mm, cats, "identity")

        va, vb = _ac(h[ok], ma[ok]), _ac(h[ok], mb[ok])
        if np.isfinite(va) and np.isfinite(vb):
            rows.append(
                _make_row(
                    **common,
                    n_items=int(ok.sum()),
                    n_clusters=sub["_cluster"].nunique(),
                    metric="gwet_ac1",
                    value_a=va,
                    value_b=vb,
                    test="cluster_bootstrap",
                    ci=_cluster_bootstrap(
                        sub,
                        lambda d: (
                            _ac(d["_h"].to_numpy(), d["_mb"].to_numpy())
                            - _ac(d["_h"].to_numpy(), d["_ma"].to_numpy())
                        ),
                        n_resamples,
                    ),
                )
            )

    swap = frame.dropna(subset=["llm_majority_swapped_a", "llm_majority_swapped_b"])
    swap = swap[
        (swap["llm_majority_swapped_a"].astype(str) != "")
        & (swap["llm_majority_swapped_b"].astype(str) != "")
    ]
    if len(swap) >= 2:
        inc_a = (swap["llm_majority_a"] != swap["llm_majority_swapped_a"]).to_numpy()
        inc_b = (swap["llm_majority_b"] != swap["llm_majority_swapped_b"]).to_numpy()
        swap = swap.assign(_ia=inc_a.astype(float), _ib=inc_b.astype(float))
        rows.append(
            _make_row(
                **common,
                n_items=len(swap),
                n_clusters=swap["_cluster"].nunique(),
                metric="ipi",
                value_a=float(inc_a.mean()),
                value_b=float(inc_b.mean()),
                test="mcnemar_exact",
                ci=_cluster_bootstrap(
                    swap,
                    lambda d: float(d["_ib"].mean() - d["_ia"].mean()),
                    n_resamples,
                ),
                stats=_exact_mcnemar(~inc_a, ~inc_b),
                effect_size_name="mcnemar_or_stability_b_vs_a",
            )
        )

    sv = frame.dropna(subset=["_seed_var_a", "_seed_var_b"])
    if len(sv) >= 2:
        rows.append(
            _make_row(
                **common,
                n_items=len(sv),
                n_clusters=sv["_cluster"].nunique(),
                metric="seed_variability",
                value_a=float(sv["_seed_var_a"].mean()),
                value_b=float(sv["_seed_var_b"].mean()),
                test="wilcoxon_signed_rank",
                ci=_cluster_bootstrap(
                    sv,
                    lambda d: float((d["_seed_var_b"] - d["_seed_var_a"]).mean()),
                    n_resamples,
                ),
                stats=_wilcoxon_paired(
                    sv["_seed_var_b"].to_numpy() - sv["_seed_var_a"].to_numpy()
                ),
                effect_size_name="rank_biserial_b_minus_a",
            )
        )

    return rows


def _seed_disagreement(votes: Any) -> float:
    v = _parse_list(votes)
    if len(v) < 2:
        return np.nan
    n_pairs = len(v) * (len(v) - 1) // 2
    n_disagree = sum(a != b for i, a in enumerate(v) for b in v[i + 1 :])
    return n_disagree / n_pairs


def compare_pairwise(
    item_pair: pd.DataFrame,
    model_a: str,
    model_b: str,
    n_resamples: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if item_pair.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = _prepare(item_pair, ["id", "criterion"])
    df["_seed_var"] = df["llm_votes"].map(_seed_disagreement)

    report = check_pairing(
        df,
        model_a,
        model_b,
        key_cols=["_task_key", "id", "criterion"],
        paradigm="pairwise",
        human_col="human_votes",
        machine_col="llm_votes",
    )
    usable = _usable_strata(report)
    if not usable:
        return pd.DataFrame(), report

    jobs = [
        (gdf, dataset, config, criterion)
        for (dataset, config, criterion), gdf in df.groupby(
            ["dataset", "config", "criterion"], dropna=False
        )
        if (dataset, config) in usable
    ]
    jobs += [
        (gdf, dataset, config, "__macro__")
        for (dataset, config), gdf in df.groupby(["dataset", "config"], dropna=False)
        if (dataset, config) in usable
    ]
    nested = Parallel(n_jobs=-1)(
        delayed(_compare_pairwise_criterion)(
            gdf, dataset, config, criterion, model_a, model_b, n_resamples
        )
        for gdf, dataset, config, criterion in jobs
    )
    rows = [r for group in nested for r in group]
    return pd.DataFrame(rows), report


def _compare_passfail_stratum(
    gdf: pd.DataFrame,
    dataset: str,
    config: str,
    model_a: str,
    model_b: str,
    n_resamples: int,
) -> list[dict[str, Any]]:
    frame = _pair_frame(
        gdf,
        model_a,
        model_b,
        key_cols=["_task_key", "id", "section"],
        value_cols=["human_majority", "llm_majority"],
    )
    frame = frame.dropna(
        subset=["human_majority_a", "llm_majority_a", "llm_majority_b"]
    )
    if len(frame) < 2:
        return []

    y = (frame["human_majority_a"] == "FAIL").to_numpy()
    pa = (frame["llm_majority_a"] == "FAIL").to_numpy()
    pb = (frame["llm_majority_b"] == "FAIL").to_numpy()
    frame = frame.assign(_y=y, _pa=pa, _pb=pb)

    common = dict(
        paradigm="passfail",
        model_a=model_a,
        model_b=model_b,
        dataset=dataset,
        config=config,
        criterion="clinical_harm_rate",
    )
    rows: list[dict[str, Any]] = []

    def _chr(d: pd.DataFrame, col: str) -> float:
        yy = d["_y"].to_numpy()
        nf = int(yy.sum())
        if nf == 0:
            return np.nan
        return float(((yy) & (~d[col].to_numpy())).sum() / nf)

    def _rate(d: pd.DataFrame, col: str, which: str) -> float:
        yy, pp = d["_y"].to_numpy(), d[col].to_numpy()
        tp = int((yy & pp).sum())
        fp = int((~yy & pp).sum())
        fn = int((yy & ~pp).sum())
        if which == "recall":
            return tp / (tp + fn) if tp + fn else np.nan
        if which == "precision":
            return tp / (tp + fp) if tp + fp else np.nan
        denom = 2 * tp + fp + fn
        return 2 * tp / denom if denom else np.nan

    fail = frame[frame["_y"]]
    if len(fail) >= 2:
        missed_a = ~fail["_pa"].to_numpy()
        missed_b = ~fail["_pb"].to_numpy()
        rows.append(
            _make_row(
                **common,
                n_items=len(fail),
                n_clusters=fail["_cluster"].nunique(),
                metric="clinical_harm_rate",
                value_a=float(missed_a.mean()),
                value_b=float(missed_b.mean()),
                test="mcnemar_exact",
                ci=_cluster_bootstrap(
                    frame, lambda d: _chr(d, "_pb") - _chr(d, "_pa"), n_resamples
                ),
                stats=_exact_mcnemar(~missed_a, ~missed_b),
                effect_size_name="mcnemar_or_catch_b_vs_a",
            )
        )

    for metric in ("recall", "precision", "f1"):
        va, vb = _rate(frame, "_pa", metric), _rate(frame, "_pb", metric)
        if not (np.isfinite(va) and np.isfinite(vb)):
            continue
        rows.append(
            _make_row(
                **common,
                n_items=len(frame),
                n_clusters=frame["_cluster"].nunique(),
                metric=metric,
                value_a=va,
                value_b=vb,
                test="cluster_bootstrap",
                ci=_cluster_bootstrap(
                    frame,
                    lambda d, m=metric: _rate(d, "_pb", m) - _rate(d, "_pa", m),
                    n_resamples,
                ),
            )
        )

    correct_a = y == pa
    correct_b = y == pb
    frame = frame.assign(_ca=correct_a.astype(float), _cb=correct_b.astype(float))
    rows.append(
        _make_row(
            **common,
            n_items=len(frame),
            n_clusters=frame["_cluster"].nunique(),
            metric="accuracy",
            value_a=float(correct_a.mean()),
            value_b=float(correct_b.mean()),
            test="mcnemar_exact",
            ci=_cluster_bootstrap(
                frame, lambda d: float(d["_cb"].mean() - d["_ca"].mean()), n_resamples
            ),
            stats=_exact_mcnemar(correct_a, correct_b),
            effect_size_name="mcnemar_or_b_vs_a",
        )
    )

    ac_row = _ac_row(
        frame,
        y.astype(float),
        pa.astype(float),
        pb.astype(float),
        {**common, "n_items": len(frame), "n_clusters": frame["_cluster"].nunique()},
        "gwet_ac1",
        n_resamples,
        binary=True,
    )
    if ac_row is not None:
        rows.append(ac_row)

    return rows


def compare_passfail(
    item_pf: pd.DataFrame,
    model_a: str,
    model_b: str,
    n_resamples: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if item_pf.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = _prepare(item_pf, ["id", "section"])

    report = check_pairing(
        df,
        model_a,
        model_b,
        key_cols=["_task_key", "id", "section"],
        paradigm="passfail",
        human_col="human_labels",
        machine_col="llm_labels",
    )
    usable = _usable_strata(report)
    if not usable:
        return pd.DataFrame(), report

    jobs = [
        (gdf, dataset, config)
        for (dataset, config), gdf in df.groupby(["dataset", "config"], dropna=False)
        if (dataset, config) in usable
    ]
    nested = Parallel(n_jobs=-1)(
        delayed(_compare_passfail_stratum)(
            gdf, dataset, config, model_a, model_b, n_resamples
        )
        for gdf, dataset, config in jobs
    )
    rows = [r for group in nested for r in group]
    return pd.DataFrame(rows), report


def _gee_sensitivity(frame: pd.DataFrame) -> tuple[float, float]:
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        long = pd.concat(
            [
                frame[["_cluster", "criterion"]].assign(
                    ae=frame["_ae_a"].to_numpy(), is_b=0
                ),
                frame[["_cluster", "criterion"]].assign(
                    ae=frame["_ae_b"].to_numpy(), is_b=1
                ),
            ],
            ignore_index=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.gee(
                "ae ~ is_b + C(criterion)",
                groups="_cluster",
                data=long,
                family=sm.families.Gaussian(),
                cov_struct=sm.cov_struct.Exchangeable(),
            ).fit()
        return float(res.params["is_b"]), float(res.pvalues["is_b"])
    except Exception:
        return np.nan, np.nan


def compare_omnibus(
    item_pw: pd.DataFrame,
    model_a: str,
    model_b: str,
    config_obj: MetricsConfig,
    usable: set[tuple[str, str]],
    n_resamples: int = 2000,
) -> pd.DataFrame:
    if item_pw.empty or not usable:
        return pd.DataFrame()

    df = _prepare(item_pw, ["id", "criterion"])
    df = df[~df["criterion"].isin(config_obj.binary_criteria)]

    rows: list[dict[str, Any]] = []
    for (dataset, config), gdf in df.groupby(["dataset", "config"], dropna=False):
        if (dataset, config) not in usable:
            continue
        frame = _pair_frame(
            gdf,
            model_a,
            model_b,
            key_cols=["_task_key", "id", "criterion"],
            value_cols=["human_median", "machine_median"],
        ).dropna(subset=["human_median_a", "machine_median_a", "machine_median_b"])
        if len(frame) < 3:
            continue

        h = frame["human_median_a"].to_numpy()
        ae_a = np.abs(h - frame["machine_median_a"].to_numpy())
        ae_b = np.abs(h - frame["machine_median_b"].to_numpy())
        frame = frame.assign(_ae_a=ae_a, _ae_b=ae_b)

        ci = _cluster_bootstrap(
            frame, lambda d: float((d["_ae_b"] - d["_ae_a"]).mean()), n_resamples
        )
        stats = _wilcoxon_paired(ae_b - ae_a)
        gee_coef, gee_p = _gee_sensitivity(frame)

        row = _make_row(
            paradigm="pointwise",
            model_a=model_a,
            model_b=model_b,
            dataset=dataset,
            config=config,
            criterion="__pooled_ordinal__",
            metric="mae",
            n_items=len(frame),
            n_clusters=frame["_cluster"].nunique(),
            value_a=float(ae_a.mean()),
            value_b=float(ae_b.mean()),
            test="cluster_bootstrap",
            ci=ci,
            stats=stats,
            effect_size_name="rank_biserial_b_minus_a",
            p_override=ci.get("boot_p", np.nan),
        )
        row["wilcoxon_p"] = stats["p_raw"]
        row["gee_coef"] = gee_coef
        row["gee_p"] = gee_p
        row["n_criteria"] = int(frame["criterion"].nunique())
        rows.append(row)

    return pd.DataFrame(rows)


def apply_fdr(
    df: pd.DataFrame,
    family_cols: tuple[str, ...] = ("dataset", "config", "metric"),
    alpha: float = 0.05,
) -> pd.DataFrame:
    if df.empty or "p_raw" not in df.columns:
        return df
    out = df.copy()
    out["p_fdr"] = np.nan
    for _, idx in out.groupby(list(family_cols), dropna=False).groups.items():
        sub = out.loc[idx, "p_raw"]
        valid = sub[sub.notna()]
        if valid.empty:
            continue
        _, p_adj, _, _ = multipletests(valid.to_numpy(), alpha=alpha, method="fdr_bh")
        out.loc[valid.index, "p_fdr"] = p_adj
    return out


def assign_verdict(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    verdicts, favors = [], []

    for _, r in out.iterrows():
        delta, hib = r["delta"], bool(r["higher_is_better"])
        p = r["p_fdr"] if pd.notna(r.get("p_fdr")) else r.get("p_raw", np.nan)

        if pd.isna(delta) or delta == 0:
            favor = ""
        elif (delta > 0) == hib:
            favor = r["model_b"]
        else:
            favor = r["model_a"]
        favors.append(favor)

        if pd.notna(p) and p < alpha and favor:
            verdicts.append(
                "model_b_better" if favor == r["model_b"] else "model_a_better"
            )
            continue

        sesoi = r.get("sesoi", np.nan)
        lo, hi = r.get("ci90_lo", np.nan), r.get("ci90_hi", np.nan)
        if (
            pd.notna(sesoi)
            and pd.notna(lo)
            and pd.notna(hi)
            and -sesoi < lo
            and hi < sesoi
        ):
            verdicts.append("equivalent")
        else:
            verdicts.append("inconclusive")

    out["favors"] = favors
    out["verdict"] = verdicts
    return out


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = assign_verdict(apply_fdr(df))
    lead = [c for c in OUTPUT_COLUMNS if c in df.columns]
    rest = [c for c in df.columns if c not in lead]
    return (
        df[lead + rest]
        .sort_values(["dataset", "config", "metric", "criterion"], kind="stable")
        .reset_index(drop=True)
    )


def compare_models(
    item_pw: pd.DataFrame,
    item_pair: pd.DataFrame,
    item_pf: pd.DataFrame,
    model_a: str,
    model_b: str,
    config_obj: MetricsConfig | None = None,
    n_resamples: int = 2000,
) -> dict[str, pd.DataFrame]:
    config_obj = config_obj or MetricsConfig()

    item_pw = _drop_excluded(item_pw)
    item_pair = _drop_excluded(item_pair)
    item_pf = _drop_excluded(item_pf)

    pw, rep_pw = compare_pointwise(item_pw, model_a, model_b, config_obj, n_resamples)
    pair, rep_pair = compare_pairwise(item_pair, model_a, model_b, n_resamples)
    pf, rep_pf = compare_passfail(item_pf, model_a, model_b, n_resamples)
    omni = compare_omnibus(
        item_pw, model_a, model_b, config_obj, _usable_strata(rep_pw), n_resamples
    )

    integrity = (
        pd.concat(
            [r for r in (rep_pw, rep_pair, rep_pf) if not r.empty], ignore_index=True
        )
        if any(not r.empty for r in (rep_pw, rep_pair, rep_pf))
        else pd.DataFrame()
    )

    return {
        "model_comparison_pointwise": _finalize(pw),
        "model_comparison_pairwise": _finalize(pair),
        "model_comparison_passfail": _finalize(pf),
        "model_comparison_omnibus": _finalize(omni),
        "model_comparison_integrity": integrity,
    }
