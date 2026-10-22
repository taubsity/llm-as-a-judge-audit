from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import MetricsConfig
from .model_comparison import (
    EXCLUDED_DATASETS,
    _as_bool,
    _cluster_bootstrap,
    _exact_mcnemar,
    _make_row,
    _prepare,
    _wilcoxon_paired,
    apply_fdr,
    assign_verdict,
)

OUTPUT_COLUMNS = [
    "paradigm",
    "dataset",
    "model",
    "config_a",
    "config_b",
    "criterion",
    "metric",
    "n_items",
    "n_clusters",
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


def _drop_excluded(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "dataset" not in df.columns:
        return df
    return df[~df["dataset"].isin(EXCLUDED_DATASETS)]


def _merge_configs(
    ddf: pd.DataFrame,
    config_a: str,
    config_b: str,
    key_cols: list[str],
    value_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    a = ddf[ddf["config"] == config_a]
    b = ddf[ddf["config"] == config_b]
    keep = [*key_cols, *value_cols]
    a2 = a[[*keep, "_cluster"]].copy()
    b2 = b[keep].copy()
    frame = a2.merge(b2, on=key_cols, suffixes=("_a", "_b"), how="inner")

    keys_a = set(map(tuple, a[key_cols].astype(str).to_numpy()))
    keys_b = set(map(tuple, b[key_cols].astype(str).to_numpy()))
    stats = {
        "n_keys_a": len(keys_a),
        "n_keys_b": len(keys_b),
        "n_shared": len(keys_a & keys_b),
        "n_only_a": len(keys_a - keys_b),
        "n_only_b": len(keys_b - keys_a),
        "n_dup_a": len(a) - len(keys_a),
        "n_dup_b": len(b) - len(keys_b),
    }
    return frame, stats


def _integrity_row(
    paradigm: str,
    dataset: str,
    model: str,
    config_a: str,
    config_b: str,
    stats: dict[str, int],
) -> dict[str, Any]:
    return {
        "paradigm": paradigm,
        "dataset": dataset,
        "model": model,
        "config_a": config_a,
        "config_b": config_b,
        **stats,
        "status": "ok" if stats["n_shared"] >= 2 else "skipped",
    }


def _compare_pointwise_mae(
    item_pw: pd.DataFrame,
    config_obj: MetricsConfig,
    models: list[str],
    configs: tuple[str, ...],
    n_resamples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if item_pw.empty:
        return [], []

    df = _prepare(item_pw, ["id", "criterion"])
    df = df[~df["criterion"].isin(config_obj.binary_criteria)]

    rows: list[dict[str, Any]] = []
    integrity: list[dict[str, Any]] = []
    for model in models:
        mdf = df[df["model"] == model]
        if mdf.empty:
            continue
        for dataset, ddf in mdf.groupby("dataset", dropna=False):
            if "baseline" not in ddf["config"].unique():
                continue
            for cfg in configs:
                if cfg not in ddf["config"].unique():
                    continue
                frame, stats = _merge_configs(
                    ddf,
                    "baseline",
                    cfg,
                    key_cols=["_task_key", "id", "criterion"],
                    value_cols=["human_median", "machine_median"],
                )
                integrity.append(
                    _integrity_row("pointwise", dataset, model, "baseline", cfg, stats)
                )
                frame = frame.dropna(
                    subset=["human_median_a", "machine_median_a", "machine_median_b"]
                )
                if len(frame) < 2:
                    continue

                h = frame["human_median_a"].to_numpy()
                ae_a = np.abs(h - frame["machine_median_a"].to_numpy())
                ae_b = np.abs(h - frame["machine_median_b"].to_numpy())
                frame = frame.assign(_ae_a=ae_a, _ae_b=ae_b)

                ci = _cluster_bootstrap(
                    frame,
                    lambda d: float((d["_ae_b"] - d["_ae_a"]).mean()),
                    n_resamples,
                )
                stats_test = _wilcoxon_paired(ae_b - ae_a)

                row = _make_row(
                    paradigm="pointwise",
                    model_a="baseline",
                    model_b=cfg,
                    dataset=dataset,
                    config=f"baseline_vs_{cfg}",
                    criterion="__pooled_ordinal__",
                    metric="mae",
                    n_items=len(frame),
                    n_clusters=frame["_cluster"].nunique(),
                    value_a=float(ae_a.mean()),
                    value_b=float(ae_b.mean()),
                    test="cluster_bootstrap",
                    ci=ci,
                    stats=stats_test,
                    effect_size_name="rank_biserial_b_minus_a",
                    p_override=ci.get("boot_p", np.nan),
                )
                row["model"] = model
                rows.append(row)

    return rows, integrity


def _compare_pairwise_accuracy_clear(
    item_pair: pd.DataFrame,
    models: list[str],
    configs: tuple[str, ...],
    n_resamples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if item_pair.empty:
        return [], []

    df = _prepare(item_pair, ["id", "criterion"])

    rows: list[dict[str, Any]] = []
    integrity: list[dict[str, Any]] = []
    for model in models:
        mdf = df[df["model"] == model]
        if mdf.empty:
            continue
        for dataset, ddf in mdf.groupby("dataset", dropna=False):
            if "baseline" not in ddf["config"].unique():
                continue
            for cfg in configs:
                if cfg not in ddf["config"].unique():
                    continue
                frame, stats = _merge_configs(
                    ddf,
                    "baseline",
                    cfg,
                    key_cols=["_task_key", "id", "criterion"],
                    value_cols=["human_majority", "llm_majority", "is_clear_majority"],
                )
                integrity.append(
                    _integrity_row("pairwise", dataset, model, "baseline", cfg, stats)
                )
                if len(frame) < 2:
                    continue

                clear = frame[_as_bool(frame["is_clear_majority_a"])]
                if len(clear) < 2:
                    continue

                ca = (clear["human_majority_a"] == clear["llm_majority_a"]).to_numpy()
                cb = (clear["human_majority_a"] == clear["llm_majority_b"]).to_numpy()
                clear = clear.assign(_ca=ca.astype(float), _cb=cb.astype(float))

                ci = _cluster_bootstrap(
                    clear,
                    lambda d: float(d["_cb"].mean() - d["_ca"].mean()),
                    n_resamples,
                )
                row = _make_row(
                    paradigm="pairwise",
                    model_a="baseline",
                    model_b=cfg,
                    dataset=dataset,
                    config=f"baseline_vs_{cfg}",
                    criterion="__macro__",
                    metric="accuracy_clear",
                    n_items=len(clear),
                    n_clusters=clear["_cluster"].nunique(),
                    value_a=float(ca.mean()),
                    value_b=float(cb.mean()),
                    test="mcnemar_exact",
                    ci=ci,
                    stats=_exact_mcnemar(ca, cb),
                    effect_size_name="mcnemar_or_b_vs_a",
                )
                row["model"] = model
                rows.append(row)

    return rows, integrity


def _compare_passfail_recall(
    item_pf: pd.DataFrame,
    models: list[str],
    configs: tuple[str, ...],
    n_resamples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if item_pf.empty:
        return [], []

    df = _prepare(item_pf, ["id", "section"])

    rows: list[dict[str, Any]] = []
    integrity: list[dict[str, Any]] = []
    for model in models:
        mdf = df[df["model"] == model]
        if mdf.empty:
            continue
        for dataset, ddf in mdf.groupby("dataset", dropna=False):
            if "baseline" not in ddf["config"].unique():
                continue
            for cfg in configs:
                if cfg not in ddf["config"].unique():
                    continue
                frame, stats = _merge_configs(
                    ddf,
                    "baseline",
                    cfg,
                    key_cols=["_task_key", "id", "section"],
                    value_cols=["human_majority", "llm_majority"],
                )
                integrity.append(
                    _integrity_row("passfail", dataset, model, "baseline", cfg, stats)
                )
                frame = frame.dropna(
                    subset=["human_majority_a", "llm_majority_a", "llm_majority_b"]
                )
                if len(frame) < 2:
                    continue

                y = (frame["human_majority_a"] == "FAIL").to_numpy()
                pa = (frame["llm_majority_a"] == "FAIL").to_numpy()
                pb = (frame["llm_majority_b"] == "FAIL").to_numpy()
                frame = frame.assign(_y=y, _pa=pa, _pb=pb)

                def _recall(d: pd.DataFrame, col: str) -> float:
                    yy, pp = d["_y"].to_numpy(), d[col].to_numpy()
                    tp = int((yy & pp).sum())
                    fn = int((yy & ~pp).sum())
                    return tp / (tp + fn) if tp + fn else np.nan

                va, vb = _recall(frame, "_pa"), _recall(frame, "_pb")
                if not (np.isfinite(va) and np.isfinite(vb)):
                    continue

                ci = _cluster_bootstrap(
                    frame, lambda d: _recall(d, "_pb") - _recall(d, "_pa"), n_resamples
                )
                row = _make_row(
                    paradigm="passfail",
                    model_a="baseline",
                    model_b=cfg,
                    dataset=dataset,
                    config=f"baseline_vs_{cfg}",
                    criterion="clinical_harm_rate",
                    metric="recall",
                    n_items=len(frame),
                    n_clusters=frame["_cluster"].nunique(),
                    value_a=va,
                    value_b=vb,
                    test="cluster_bootstrap",
                    ci=ci,
                )
                row["model"] = model
                rows.append(row)

    return rows, integrity


def compare_configs(
    item_pw: pd.DataFrame,
    item_pair: pd.DataFrame,
    item_pf: pd.DataFrame,
    config_obj: MetricsConfig | None = None,
    models: list[str] | None = None,
    configs: tuple[str, ...] = ("rule", "demo"),
    n_resamples: int = 2000,
) -> dict[str, pd.DataFrame]:
    config_obj = config_obj or MetricsConfig()

    item_pw = _drop_excluded(item_pw)
    item_pair = _drop_excluded(item_pair)
    item_pf = _drop_excluded(item_pf)

    if models is None:
        seen: set[str] = set()
        for frame in (item_pw, item_pair, item_pf):
            if not frame.empty and "model" in frame.columns:
                seen |= set(frame["model"].dropna().unique())
        models = sorted(seen - {"MULTIAGENT"})

    rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []

    for compare_fn in (
        lambda: _compare_pointwise_mae(
            item_pw, config_obj, models, configs, n_resamples
        ),
        lambda: _compare_pairwise_accuracy_clear(
            item_pair, models, configs, n_resamples
        ),
        lambda: _compare_passfail_recall(item_pf, models, configs, n_resamples),
    ):
        r, i = compare_fn()
        rows.extend(r)
        integrity_rows.extend(i)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = apply_fdr(result, family_cols=("model",))
        result = assign_verdict(result)
        result = result.rename(columns={"model_a": "config_a", "model_b": "config_b"})
        lead = [c for c in OUTPUT_COLUMNS if c in result.columns]
        rest = [c for c in result.columns if c not in lead]
        result = (
            result[lead + rest]
            .sort_values(["model", "dataset", "paradigm", "metric"], kind="stable")
            .reset_index(drop=True)
        )

    integrity = pd.DataFrame(integrity_rows)

    return {"config_comparison": result, "config_comparison_integrity": integrity}
