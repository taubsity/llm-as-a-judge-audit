from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _fig2_normalize_criterion(panel_id: str, criterion: str) -> str:
    crit = str(criterion).strip().lower()
    if crit == "organization":
        crit = "organized"
    return crit


def _normal_ci(value: float, se: float, z: float = 1.96) -> tuple[float, float]:
    return max(0.0, value - z * se), min(1.0, value + z * se)


def _pick_ci(
    value: float,
    lo: float | None,
    hi: float | None,
    se: float | None = None,
) -> tuple[float, float]:
    if lo is not None and hi is not None and not (pd.isna(lo) or pd.isna(hi)):
        return float(max(0.0, lo)), float(min(1.0, hi))
    if se is not None and not pd.isna(se):
        return _normal_ci(float(value), float(se))
    v = float(value)
    return v, v


def _fig3_gap_category(
    human: float,
    human_lo: float,
    human_hi: float,
    model: float,
) -> str:
    if human_lo <= model <= human_hi:
        return "within"
    if model < human_lo and (human - model) <= 0.1:
        return "near"
    if model < human_lo and (human - model) > 0.1:
        return "far"
    return "within"


def build_plotting_fig2_records(
    summary_pointwise: pd.DataFrame,
    summary_pairwise: pd.DataFrame,
    summary_passfail: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    if not summary_pointwise.empty:
        pw_overall = (
            summary_pointwise[summary_pointwise["consensus_bin"] == "overall"].copy()
            if "consensus_bin" in summary_pointwise.columns
            else summary_pointwise.copy()
        )
        for _, row in pw_overall.iterrows():
            ds = str(row.get("dataset", ""))
            crit = str(row.get("criterion", ""))

            panel_id: str | None = None
            val_col, lo_col, hi_col = "mae", "mae_ci_lo", "mae_ci_hi"
            metric = "mae"

            if ds == "reports":
                panel_id = "reports"
            elif ds == "nursing":
                if crit in {"stig_original", "stig_summary"}:
                    panel_id = "nursing_stig"
                    metric = "accuracy"
                    val_col, lo_col, hi_col = (
                        "accuracy",
                        "accuracy_ci_lo",
                        "accuracy_ci_hi",
                    )
                else:
                    panel_id = "nursing"

            if panel_id is None:
                continue

            records.append(
                {
                    "panel": panel_id,
                    "criterion": _fig2_normalize_criterion(panel_id, crit),
                    "model": row.get("model"),
                    "config": row.get("config"),
                    "metric": metric,
                    "value": row.get(val_col, np.nan),
                    "lo": row.get(lo_col, np.nan),
                    "hi": row.get(hi_col, np.nan),
                }
            )

    if not summary_pairwise.empty:
        pair_overall = (
            summary_pairwise[summary_pairwise["consensus_bin"] == "overall"].copy()
            if "consensus_bin" in summary_pairwise.columns
            else summary_pairwise.copy()
        )
        for _, row in pair_overall.iterrows():
            if str(row.get("dataset", "")) != "epikrise":
                continue
            records.append(
                {
                    "panel": "epikrise",
                    "criterion": _fig2_normalize_criterion(
                        "epikrise", str(row.get("criterion", ""))
                    ),
                    "model": row.get("model"),
                    "config": row.get("config"),
                    "metric": "accuracy",
                    "value": row.get("accuracy_clear", np.nan),
                    "lo": row.get("accuracy_clear_ci_lo", np.nan),
                    "hi": row.get("accuracy_clear_ci_hi", np.nan),
                }
            )

    if not summary_passfail.empty:
        for _, row in summary_passfail.iterrows():
            if str(row.get("dataset", "")) not in {
                "mtb_errors",
                "mtb_passfail",
                "mtb_pf",
            }:
                continue
            records.append(
                {
                    "panel": "mtb_pf",
                    "criterion": "clinical_harm_rate",
                    "model": row.get("model"),
                    "config": row.get("config"),
                    "metric": "chr",
                    "value": row.get("clinical_harm_rate", np.nan),
                    "lo": row.get(
                        "clinical_harm_rate_ci_lo", row.get("chr_ci_lo", np.nan)
                    ),
                    "hi": row.get(
                        "clinical_harm_rate_ci_hi", row.get("chr_ci_hi", np.nan)
                    ),
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.dropna(subset=["value", "model", "config", "criterion"])
    return df


def build_plotting_passfail_records(summary_passfail: pd.DataFrame) -> pd.DataFrame:
    if summary_passfail.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []

    for _, row in summary_passfail.iterrows():
        if str(row.get("dataset", "")) not in {"mtb_errors", "mtb_passfail", "mtb_pf"}:
            continue

        base = {
            "panel": "mtb_pf",
            "dataset": "mtb_errors",
            "task": row.get("task", ""),
            "model": row.get("model"),
            "config": row.get("config"),
        }

        def _append_metric(
            metric: str,
            value: Any,
            lo: Any = np.nan,
            hi: Any = np.nan,
        ) -> None:
            if pd.isna(value):
                return
            records.append(
                {
                    **base,
                    "metric": metric,
                    "value": float(value),
                    "lo": float(lo) if pd.notna(lo) else np.nan,
                    "hi": float(hi) if pd.notna(hi) else np.nan,
                }
            )

        _append_metric(
            "clinical_harm_rate",
            row.get("clinical_harm_rate", np.nan),
            row.get("chr_ci_lo", np.nan),
            row.get("chr_ci_hi", np.nan),
        )
        _append_metric(
            "recall_micro",
            row.get("recall_micro", np.nan),
            row.get("recall_ci_lo", np.nan),
            row.get("recall_ci_hi", np.nan),
        )
        _append_metric("precision_micro", row.get("precision_micro", np.nan))
        _append_metric("f1_micro", row.get("f1_micro", np.nan))

        ac1 = row.get("gwet_ac1", np.nan)
        ac1_lo = row.get("gwet_ac1_ci_lo", np.nan)
        ac1_hi = row.get("gwet_ac1_ci_hi", np.nan)
        if pd.isna(ac1_lo) or pd.isna(ac1_hi):
            ac1_lo, ac1_hi = _pick_ci(
                float(ac1) if pd.notna(ac1) else np.nan,
                ac1_lo,
                ac1_hi,
                row.get("gwet_ac1_se", np.nan),
            )
        _append_metric("gwet_ac1", ac1, ac1_lo, ac1_hi)

        for col in row.index:
            if not (
                col.startswith("detailed_")
                or col.startswith("type_")
                or col.startswith("severity_")
            ):
                continue
            if col in {"detailed_n_codes"}:
                _append_metric(col, row.get(col, np.nan))
            else:
                _append_metric(col, row.get(col, np.nan))

    return pd.DataFrame(records)


def _build_fig3_model_df(
    summary_pointwise: pd.DataFrame,
    summary_pairwise: pd.DataFrame,
    summary_passfail: pd.DataFrame,
) -> pd.DataFrame:
    panel_names = {
        "reports": "Reports",
        "nursing": "Nursing",
        "epikrise": "Epicrisis",
        "mtb_pf": "MTB (pass/fail)",
    }

    model_rows: list[dict[str, Any]] = []

    ov_pw = (
        summary_pointwise[summary_pointwise["consensus_bin"] == "overall"].copy()
        if not summary_pointwise.empty and "consensus_bin" in summary_pointwise.columns
        else summary_pointwise.copy()
    )
    ov_pair = (
        summary_pairwise[summary_pairwise["consensus_bin"] == "overall"].copy()
        if not summary_pairwise.empty and "consensus_bin" in summary_pairwise.columns
        else summary_pairwise.copy()
    )
    ov_pf = summary_passfail.copy() if not summary_passfail.empty else pd.DataFrame()
    has_mtb_pf = (
        not ov_pf.empty
        and "dataset" in ov_pf.columns
        and ov_pf["dataset"].isin(["mtb_errors", "mtb_passfail", "mtb_pf"]).any()
    )

    if not ov_pw.empty:
        for _, row in ov_pw.iterrows():
            dataset = str(row.get("dataset", ""))
            criterion_raw = str(row.get("criterion", ""))
            panel_id: str | None = None
            metric_col: str | None = None
            ci_lo_col: str | None = None
            ci_hi_col: str | None = None
            se_col: str | None = None

            if dataset == "reports":
                panel_id, metric_col = "reports", "gwet_ac2"
                ci_lo_col, ci_hi_col, se_col = (
                    "gwet_ac2_ci_lo",
                    "gwet_ac2_ci_hi",
                    "gwet_ac2_se",
                )
            elif dataset == "nursing":
                if criterion_raw in {"stig_original", "stig_summary"}:
                    panel_id, metric_col = "nursing", "gwet_ac1"
                    ci_lo_col, ci_hi_col, se_col = (
                        "gwet_ac1_ci_lo",
                        "gwet_ac1_ci_hi",
                        "gwet_ac1_se",
                    )
                else:
                    panel_id, metric_col = "nursing", "gwet_ac2"
                    ci_lo_col, ci_hi_col, se_col = (
                        "gwet_ac2_ci_lo",
                        "gwet_ac2_ci_hi",
                        "gwet_ac2_se",
                    )

            if panel_id is None or metric_col is None:
                continue

            ac = row.get(metric_col, np.nan)
            if pd.isna(ac):
                continue

            criterion = _fig2_normalize_criterion(panel_id, criterion_raw)
            ci_lo, ci_hi = _pick_ci(
                float(ac),
                row.get(ci_lo_col, np.nan),
                row.get(ci_hi_col, np.nan),
                row.get(se_col, np.nan),
            )
            model_rows.append(
                {
                    "panel": panel_id,
                    "panel_name": panel_names.get(panel_id, panel_id),
                    "criterion": criterion,
                    "model": row.get("model"),
                    "config": row.get("config"),
                    "ac": float(ac),
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                }
            )

    if not ov_pair.empty:
        for _, row in ov_pair.iterrows():
            if str(row.get("dataset", "")) != "epikrise":
                continue
            ac = row.get("gwet_ac1", np.nan)
            if pd.isna(ac):
                continue

            criterion = _fig2_normalize_criterion(
                "epikrise",
                str(row.get("criterion", "")),
            )
            ci_lo, ci_hi = _pick_ci(
                float(ac),
                row.get("gwet_ac1_ci_lo", np.nan),
                row.get("gwet_ac1_ci_hi", np.nan),
                row.get("gwet_ac1_se", np.nan),
            )
            model_rows.append(
                {
                    "panel": "epikrise",
                    "panel_name": panel_names["epikrise"],
                    "criterion": criterion,
                    "model": row.get("model"),
                    "config": row.get("config"),
                    "ac": float(ac),
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                }
            )

    if has_mtb_pf:
        for _, row in ov_pf.iterrows():
            if str(row.get("dataset", "")) not in {
                "mtb_errors",
                "mtb_passfail",
                "mtb_pf",
            }:
                continue
            ac = row.get("gwet_ac1", np.nan)
            if pd.isna(ac):
                continue

            criterion = _fig2_normalize_criterion(
                "mtb_pf",
                str(row.get("criterion", "clinical_harm_rate")),
            )
            ci_lo, ci_hi = _pick_ci(
                float(ac),
                row.get("gwet_ac1_ci_lo", np.nan),
                row.get("gwet_ac1_ci_hi", np.nan),
                row.get("gwet_ac1_se", np.nan),
            )
            model_rows.append(
                {
                    "panel": "mtb_pf",
                    "panel_name": panel_names["mtb_pf"],
                    "criterion": criterion,
                    "model": row.get("model"),
                    "config": row.get("config"),
                    "ac": float(ac),
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                }
            )

    if not model_rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(model_rows)
        .groupby(
            ["panel", "panel_name", "criterion", "model", "config"],
            as_index=False,
        )[["ac", "ci_lo", "ci_hi"]]
        .mean()
    )


def build_plotting_fig3_main(
    summary_pointwise: pd.DataFrame,
    summary_pairwise: pd.DataFrame,
    summary_passfail: pd.DataFrame,
    human_baseline: pd.DataFrame,
) -> pd.DataFrame:
    panel_names = {
        "reports": "Reports",
        "nursing": "Nursing",
        "epikrise": "Epicrisis",
        "mtb_pf": "MTB (pass/fail)",
    }

    model_df = _build_fig3_model_df(
        summary_pointwise, summary_pairwise, summary_passfail
    )

    ov_pf = summary_passfail.copy() if not summary_passfail.empty else pd.DataFrame()
    has_mtb_pf = (
        not ov_pf.empty
        and "dataset" in ov_pf.columns
        and ov_pf["dataset"].isin(["mtb_errors", "mtb_passfail", "mtb_pf"]).any()
    )

    human_rows: list[dict[str, Any]] = []

    if not human_baseline.empty:
        for _, row in human_baseline.iterrows():
            dataset = str(row.get("dataset", ""))
            criterion_raw = str(row.get("criterion", ""))
            panel_id: str | None = None
            ac_col: str | None = None
            se_col: str | None = None

            if dataset == "reports":
                panel_id, ac_col, se_col = "reports", "gwet_ac2_hh", "gwet_ac2_hh_se"
            elif dataset == "nursing":
                if criterion_raw in {"stig_original", "stig_summary"}:
                    panel_id, ac_col, se_col = (
                        "nursing",
                        "gwet_ac1_hh",
                        "gwet_ac1_hh_se",
                    )
                else:
                    panel_id, ac_col, se_col = (
                        "nursing",
                        "gwet_ac2_hh",
                        "gwet_ac2_hh_se",
                    )
            elif dataset == "epikrise":
                panel_id, ac_col, se_col = "epikrise", "gwet_ac1_hh", "gwet_ac1_hh_se"
            elif dataset in {"mtb_errors", "mtb_passfail", "mtb_pf"} and has_mtb_pf:
                panel_id, ac_col, se_col = "mtb_pf", "gwet_ac1_hh", "gwet_ac1_hh_se"

            if panel_id is None or ac_col is None:
                continue

            ac = row.get(ac_col, np.nan)
            if pd.isna(ac):
                continue

            criterion = _fig2_normalize_criterion(panel_id, criterion_raw)
            ci_lo, ci_hi = _pick_ci(float(ac), None, None, row.get(se_col, np.nan))
            human_rows.append(
                {
                    "panel": panel_id,
                    "panel_name": panel_names.get(panel_id, panel_id),
                    "criterion": criterion,
                    "human_ac": float(ac),
                    "human_lo": ci_lo,
                    "human_hi": ci_hi,
                }
            )

    human_df = pd.DataFrame(human_rows)
    if model_df.empty or human_df.empty:
        return pd.DataFrame()
    human_df = human_df.groupby(
        ["panel", "panel_name", "criterion"],
        as_index=False,
    )[["human_ac", "human_lo", "human_hi"]].mean()

    avg_model = model_df.groupby(
        ["panel", "panel_name", "model", "config"], as_index=False
    )["ac"].mean()
    avg_model["criterion"] = "average"
    avg_model["ci_lo"] = avg_model["ac"]
    avg_model["ci_hi"] = avg_model["ac"]
    model_df = pd.concat(
        [
            model_df,
            avg_model[
                [
                    "panel",
                    "panel_name",
                    "criterion",
                    "model",
                    "config",
                    "ac",
                    "ci_lo",
                    "ci_hi",
                ]
            ],
        ],
        ignore_index=True,
    )

    avg_human = human_df.groupby(["panel", "panel_name"], as_index=False)[
        "human_ac"
    ].mean()
    avg_human["criterion"] = "average"
    avg_human["human_lo"] = avg_human["human_ac"]
    avg_human["human_hi"] = avg_human["human_ac"]
    human_df = pd.concat(
        [
            human_df,
            avg_human[
                ["panel", "panel_name", "criterion", "human_ac", "human_lo", "human_hi"]
            ],
        ],
        ignore_index=True,
    )

    model_df = model_df.assign(_ci_width=model_df["ci_hi"] - model_df["ci_lo"])
    ranked = model_df.sort_values(
        ["ac", "_ci_width", "model", "config"],
        ascending=[False, True, True, True],
    )
    idx_per_model = (
        ranked.groupby(["panel", "criterion", "model"], sort=False).head(1).index
    )
    model_best_per_model = model_df.loc[idx_per_model].reset_index(drop=True)

    spread_df = model_best_per_model.groupby(["panel", "criterion"], as_index=False)[
        "ac"
    ].agg(spread_lo="min", spread_hi="max")

    ranked_best = model_best_per_model.sort_values(
        ["ac", "_ci_width", "model", "config"],
        ascending=[False, True, True, True],
    )
    idx_best = ranked_best.groupby(["panel", "criterion"], sort=False).head(1).index
    best_df = (
        model_best_per_model.loc[idx_best]
        .reset_index(drop=True)
        .drop(columns=["_ci_width"])
    )

    main_df = best_df.merge(
        human_df,
        on=["panel", "panel_name", "criterion"],
        how="inner",
    )
    main_df = main_df.merge(spread_df, on=["panel", "criterion"], how="left")
    if main_df.empty:
        return pd.DataFrame()

    main_df["gap"] = main_df["human_ac"] - main_df["ac"]
    main_df["gap_cat"] = main_df.apply(
        lambda r: _fig3_gap_category(
            float(r["human_ac"]),
            float(r["human_lo"]),
            float(r["human_hi"]),
            float(r["ac"]),
        ),
        axis=1,
    )
    main_df = main_df.rename(
        columns={
            "ac": "best_ac",
            "ci_lo": "best_lo",
            "ci_hi": "best_hi",
            "model": "best_model",
            "config": "best_config",
        }
    )
    return main_df


def build_plotting_fig3_all_values(
    summary_pointwise: pd.DataFrame,
    summary_pairwise: pd.DataFrame,
    summary_passfail: pd.DataFrame,
) -> pd.DataFrame:
    model_df = _build_fig3_model_df(
        summary_pointwise, summary_pairwise, summary_passfail
    )
    if model_df.empty:
        return pd.DataFrame()

    avg = model_df.groupby(["panel", "panel_name", "model", "config"], as_index=False)[
        ["ac", "ci_lo", "ci_hi"]
    ].mean()
    avg["criterion"] = "average"

    cols = [
        "panel",
        "panel_name",
        "criterion",
        "model",
        "config",
        "ac",
        "ci_lo",
        "ci_hi",
    ]
    return pd.concat(
        [model_df[cols], avg[cols]],
        ignore_index=True,
    )


def build_plotting_fig4_points(consensus_jsd: pd.DataFrame) -> pd.DataFrame:
    if consensus_jsd.empty:
        return pd.DataFrame()

    jsd = consensus_jsd.copy()

    if "paradigm" in jsd.columns:
        jsd["paradigm"] = jsd["paradigm"].replace({"pass_fail": "passfail"})
    else:
        jsd["paradigm"] = "pointwise"

    keep_bins = {"low", "medium", "high", "overall"}
    jsd = jsd[jsd["consensus_bin"].isin(keep_bins)]
    if jsd.empty:
        return pd.DataFrame()

    group_cols = ["paradigm", "model", "config", "dataset", "consensus_bin"]
    for col in ["config", "dataset"]:
        if col not in jsd.columns:
            jsd[col] = ""

    out = (
        jsd.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            mean_jsd=("weighted_jsd", "mean"),
            n_items=("n_items", "sum"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    return out


def build_plotting_fig4b_points(
    summary_pointwise: pd.DataFrame,
    summary_pairwise: pd.DataFrame,
) -> pd.DataFrame:
    frames = []

    if not summary_pointwise.empty:
        pw = summary_pointwise[~summary_pointwise["is_binary"]].copy()

        pw = pw[pw["consensus_bin"].isin({"low", "medium", "high", "overall"})]
        if not pw.empty:
            group_cols = ["model", "config", "dataset", "consensus_bin"]
            for col in ["config", "dataset"]:
                if col not in pw.columns:
                    pw[col] = ""
            pw_agg = pw.groupby(group_cols, dropna=False, as_index=False).agg(
                metric_value=("mae", "mean"), n_items=("n_items", "sum")
            )
            pw_agg["paradigm"] = "pointwise"
            pw_agg["metric"] = "mae"
            frames.append(pw_agg)

    if not summary_pairwise.empty:
        pair = summary_pairwise[summary_pairwise["dataset"] == "epikrise"].copy()
        pair = pair[pair["consensus_bin"].isin({"low", "medium", "high", "overall"})]
        if not pair.empty:
            group_cols = ["model", "config", "dataset", "consensus_bin"]
            for col in ["config"]:
                if col not in pair.columns:
                    pair[col] = ""
            pair_agg = pair.groupby(group_cols, dropna=False, as_index=False).agg(
                metric_value=("accuracy_all", "mean"), n_items=("n_items", "sum")
            )
            pair_agg["paradigm"] = "pairwise"
            pair_agg["metric"] = "accuracy"
            frames.append(pair_agg)

    if not frames:
        return pd.DataFrame()

    col_order = [
        "paradigm",
        "model",
        "config",
        "dataset",
        "consensus_bin",
        "metric_value",
        "metric",
        "n_items",
    ]
    out = pd.concat(frames, ignore_index=True)
    return (
        out[col_order]
        .sort_values(["paradigm", "model", "config", "dataset", "consensus_bin"])
        .reset_index(drop=True)
    )


def write_plotting_inputs(
    summary_pointwise: pd.DataFrame,
    summary_pairwise: pd.DataFrame,
    summary_passfail: pd.DataFrame,
    human_baseline: pd.DataFrame,
    consensus_jsd: pd.DataFrame,
    outdir: Path,
) -> None:
    fig2_records = build_plotting_fig2_records(
        summary_pointwise,
        summary_pairwise,
        summary_passfail,
    )
    if not fig2_records.empty:
        path = outdir / "plotting_fig2_records.csv"
        fig2_records.to_csv(path, index=False)

    fig3_main = build_plotting_fig3_main(
        summary_pointwise,
        summary_pairwise,
        summary_passfail,
        human_baseline,
    )
    if not fig3_main.empty:
        path = outdir / "plotting_fig3_main.csv"
        fig3_main.to_csv(path, index=False)

    fig3_all = build_plotting_fig3_all_values(
        summary_pointwise,
        summary_pairwise,
        summary_passfail,
    )
    if not fig3_all.empty:
        path = outdir / "plotting_fig3_all_values.csv"
        fig3_all.to_csv(path, index=False)

    fig4_points = build_plotting_fig4_points(consensus_jsd)
    if not fig4_points.empty:
        path = outdir / "plotting_fig4_points.csv"
        fig4_points.to_csv(path, index=False)

    fig4b_points = build_plotting_fig4b_points(summary_pointwise, summary_pairwise)
    if not fig4b_points.empty:
        path = outdir / "plotting_fig4b_points.csv"
        fig4b_points.to_csv(path, index=False)

    pf_records = build_plotting_passfail_records(summary_passfail)
    if not pf_records.empty:
        path = outdir / "plotting_passfail_records.csv"
        pf_records.to_csv(path, index=False)
