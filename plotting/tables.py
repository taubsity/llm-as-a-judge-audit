from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .human_reliability import (
    compute_human_accuracy_pairwise,
    compute_human_accuracy_pointwise,
    compute_human_mae,
    compute_human_recall_passfail,
)
from .style import CRITERION_ORDER_CONTENT_TO_STYLE

CONFIG_ORDER = ["baseline", "rule", "demo"]
STRATUM_ORDER = ["overall", "high", "medium", "low"]
BINARY_CRITERIA = frozenset({"stig_original", "stig_summary"})


def _fmt(val: float, ci_lo: float, ci_hi: float) -> str:
    if pd.isna(val):
        return ""
    s = f"{val:.2f}"
    if not pd.isna(ci_lo) and not pd.isna(ci_hi):
        s += f" [{ci_lo:.2f}, {ci_hi:.2f}]"
    return s


def _order_criteria(criteria: list[str]) -> list[str]:
    ordered = [c for c in CRITERION_ORDER_CONTENT_TO_STYLE if c in criteria]
    remainder = sorted(c for c in criteria if c not in ordered)
    return ordered + remainder


def _sort_key(model: str, config: str) -> tuple[str, int]:
    idx = CONFIG_ORDER.index(config) if config in CONFIG_ORDER else 99
    return (model, idx)


def _row_label(model: str, config: str) -> str:
    return f"{model} - {config}"


def _row_label_stratified(model: str, config: str, stratum: str) -> str:
    return f"{model} - {config} - {stratum}"


def _build_sheet(
    df: pd.DataFrame,
    metric_col: str,
    ci_lo_col: str,
    ci_hi_col: str,
    criterion_col: str = "criterion",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    criteria = _order_criteria(df[criterion_col].unique().tolist())
    rows_data: list[dict] = []

    groups = sorted(
        df.groupby(["model", "config"]),
        key=lambda g: _sort_key(g[0][0], g[0][1]),
    )
    for (model, config), gdf in groups:
        row: dict[str, str] = {"": _row_label(model, config)}
        for _, r in gdf.iterrows():
            crit = r[criterion_col]
            row[crit] = _fmt(
                r.get(metric_col, np.nan),
                r.get(ci_lo_col, np.nan),
                r.get(ci_hi_col, np.nan),
            )
        rows_data.append(row)

    result = pd.DataFrame(rows_data)
    cols = [""] + [c for c in criteria if c in result.columns]
    return result.reindex(columns=cols, fill_value="")


def _build_sheet_stratified(
    df: pd.DataFrame,
    metric_col: str,
    ci_lo_col: str,
    ci_hi_col: str,
    criterion_col: str = "criterion",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    criteria = _order_criteria(df[criterion_col].unique().tolist())
    rows_data: list[dict] = []

    model_config_pairs = sorted(
        df[["model", "config"]].drop_duplicates().values.tolist(),
        key=lambda mc: _sort_key(mc[0], mc[1]),
    )
    for model, config in model_config_pairs:
        mc_df = df[(df["model"] == model) & (df["config"] == config)]
        for stratum in STRATUM_ORDER:
            sdf = mc_df[mc_df["consensus_bin"] == stratum]
            if sdf.empty:
                continue
            row: dict[str, str] = {"": _row_label_stratified(model, config, stratum)}
            for _, r in sdf.iterrows():
                crit = r[criterion_col]
                row[crit] = _fmt(
                    r.get(metric_col, np.nan),
                    r.get(ci_lo_col, np.nan),
                    r.get(ci_hi_col, np.nan),
                )
            rows_data.append(row)

    result = pd.DataFrame(rows_data)
    cols = [""] + [c for c in criteria if c in result.columns]
    return result.reindex(columns=cols, fill_value="")


def _build_scalar_sheet(
    df: pd.DataFrame,
    metric_col: str,
    ci_lo_col: str,
    ci_hi_col: str,
    col_label: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows_data: list[dict] = []
    groups = sorted(
        df.groupby(["model", "config"]),
        key=lambda g: _sort_key(g[0][0], g[0][1]),
    )
    for (model, config), gdf in groups:
        r = gdf.iloc[0]
        rows_data.append(
            {
                "": _row_label(model, config),
                col_label: _fmt(
                    r.get(metric_col, np.nan),
                    r.get(ci_lo_col, np.nan),
                    r.get(ci_hi_col, np.nan),
                ),
            }
        )
    return pd.DataFrame(rows_data)


def _pw_overall(pw: pd.DataFrame) -> pd.DataFrame:
    if pw.empty or "consensus_bin" not in pw.columns:
        return pw
    return pw[pw["consensus_bin"] == "overall"].copy()


def _get_pw_dataset(pw: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if pw.empty:
        return pw
    return pw[pw["dataset"] == dataset].copy()


def _get_pair_dataset(pair: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if pair.empty:
        return pair
    return pair[pair["dataset"] == dataset].copy()


def _get_pf_dataset(pf: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if pf.empty:
        return pf
    return pf[pf["dataset"] == dataset].copy()


def _get_jsd_dataset(jsd: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if jsd.empty:
        return jsd
    return jsd[jsd["dataset"] == dataset].copy()


def _build_alignment_sheets(
    pw: pd.DataFrame,
    pair: pd.DataFrame,
    pf: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    pw_ov = _pw_overall(pw)
    pair_ov = _pw_overall(pair)

    sheets: dict[str, pd.DataFrame] = {}

    rp = _get_pw_dataset(pw_ov, "reports")
    if not rp.empty:
        rp = rp[~rp["criterion"].isin(BINARY_CRITERIA)]
        sheets["reports"] = _build_sheet(rp, "mae", "mae_ci_lo", "mae_ci_hi")

    nr = _get_pw_dataset(pw_ov, "nursing")
    if not nr.empty:
        nr_ord = nr[~nr["criterion"].isin(BINARY_CRITERIA)]
        nr_bin = nr[nr["criterion"].isin(BINARY_CRITERIA)]

        sheet_ord = (
            _build_sheet(nr_ord, "mae", "mae_ci_lo", "mae_ci_hi")
            if not nr_ord.empty
            else pd.DataFrame()
        )
        sheet_bin = (
            _build_sheet(nr_bin, "accuracy", "accuracy_ci_lo", "accuracy_ci_hi")
            if not nr_bin.empty
            else pd.DataFrame()
        )

        if not sheet_ord.empty and not sheet_bin.empty:
            merged = sheet_ord.merge(sheet_bin, on="", how="outer")
            criteria = _order_criteria(nr["criterion"].unique().tolist())
            cols = [""] + [c for c in criteria if c in merged.columns]
            sheets["nursing"] = merged.reindex(columns=cols, fill_value="")
        elif not sheet_ord.empty:
            sheets["nursing"] = sheet_ord
        elif not sheet_bin.empty:
            sheets["nursing"] = sheet_bin

    mtb_pf = _get_pf_dataset(pf, "mtb_errors")
    if not mtb_pf.empty:
        sheets["mtb"] = _build_scalar_sheet(
            mtb_pf, "clinical_harm_rate", "chr_ci_lo", "chr_ci_hi", "errors (P/F)"
        )

    ep = _get_pair_dataset(pair_ov, "epikrise")
    if not ep.empty:
        sheets["epikrise"] = _build_sheet(
            ep, "accuracy_clear", "accuracy_clear_ci_lo", "accuracy_clear_ci_hi"
        )

    return sheets


def _build_reliability_sheets(
    pw: pd.DataFrame,
    pair: pd.DataFrame,
    pf: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    pw_ov = _pw_overall(pw)
    pair_ov = _pw_overall(pair)

    sheets: dict[str, pd.DataFrame] = {}

    rp = _get_pw_dataset(pw_ov, "reports")
    if not rp.empty:
        rp = rp[~rp["criterion"].isin(BINARY_CRITERIA)]
        sheets["reports"] = _build_sheet(
            rp, "gwet_ac2", "gwet_ac2_ci_lo", "gwet_ac2_ci_hi"
        )

    nr = _get_pw_dataset(pw_ov, "nursing")
    if not nr.empty:
        nr_ord = nr[~nr["criterion"].isin(BINARY_CRITERIA)]
        nr_bin = nr[nr["criterion"].isin(BINARY_CRITERIA)]

        sheet_ord = (
            _build_sheet(nr_ord, "gwet_ac2", "gwet_ac2_ci_lo", "gwet_ac2_ci_hi")
            if not nr_ord.empty
            else pd.DataFrame()
        )
        sheet_bin = (
            _build_sheet(nr_bin, "gwet_ac1", "gwet_ac1_ci_lo", "gwet_ac1_ci_hi")
            if not nr_bin.empty
            else pd.DataFrame()
        )

        if not sheet_ord.empty and not sheet_bin.empty:
            merged = sheet_ord.merge(sheet_bin, on="", how="outer")
            criteria = _order_criteria(nr["criterion"].unique().tolist())
            cols = [""] + [c for c in criteria if c in merged.columns]
            sheets["nursing"] = merged.reindex(columns=cols, fill_value="")
        elif not sheet_ord.empty:
            sheets["nursing"] = sheet_ord
        elif not sheet_bin.empty:
            sheets["nursing"] = sheet_bin

    mtb_pf = _get_pf_dataset(pf, "mtb_errors")
    if not mtb_pf.empty:
        sheets["mtb"] = _build_scalar_sheet(
            mtb_pf, "gwet_ac1", "gwet_ac1_ci_lo", "gwet_ac1_ci_hi", "errors (P/F)"
        )

    ep = _get_pair_dataset(pair_ov, "epikrise")
    if not ep.empty:
        sheets["epikrise"] = _build_sheet(
            ep, "gwet_ac1", "gwet_ac1_ci_lo", "gwet_ac1_ci_hi"
        )

    return sheets


def _build_uncertainty_sheets(
    pw: pd.DataFrame,
    pair: pd.DataFrame,
    pf: pd.DataFrame,
    jsd: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    sheets: dict[str, pd.DataFrame] = {}

    rp = _get_pw_dataset(pw, "reports")
    rp = rp[~rp["criterion"].isin(BINARY_CRITERIA)]
    if not rp.empty:
        sheets["reports-mae"] = _build_sheet_stratified(
            rp, "mae", "mae_ci_lo", "mae_ci_hi"
        )
        sheets["reports-ac2"] = _build_sheet_stratified(
            rp, "gwet_ac2", "gwet_ac2_ci_lo", "gwet_ac2_ci_hi"
        )
    rp_jsd = _get_jsd_dataset(jsd, "reports")
    if not rp_jsd.empty:
        sheets["reports-jsd_b"] = _build_sheet_stratified(
            rp_jsd, "weighted_jsd", "weighted_jsd_ci_lo", "weighted_jsd_ci_hi"
        )

    nr = _get_pw_dataset(pw, "nursing")
    if not nr.empty:
        nr_ord = nr[~nr["criterion"].isin(BINARY_CRITERIA)]
        nr_bin = nr[nr["criterion"].isin(BINARY_CRITERIA)]
        if not nr_ord.empty:
            sheets["nursing-mae"] = _build_sheet_stratified(
                nr_ord, "mae", "mae_ci_lo", "mae_ci_hi"
            )
            sheets["nursing-ac2"] = _build_sheet_stratified(
                nr_ord, "gwet_ac2", "gwet_ac2_ci_lo", "gwet_ac2_ci_hi"
            )
        if not nr_bin.empty:
            sheets["nursing-acc"] = _build_sheet_stratified(
                nr_bin, "accuracy", "accuracy_ci_lo", "accuracy_ci_hi"
            )
            sheets["nursing-ac1"] = _build_sheet_stratified(
                nr_bin, "gwet_ac1", "gwet_ac1_ci_lo", "gwet_ac1_ci_hi"
            )
    nr_jsd = _get_jsd_dataset(jsd, "nursing")
    if not nr_jsd.empty:
        sheets["nursing-jsd_b"] = _build_sheet_stratified(
            nr_jsd, "weighted_jsd", "weighted_jsd_ci_lo", "weighted_jsd_ci_hi"
        )

    mtb_pf = _get_pf_dataset(pf, "mtb_errors")
    if not mtb_pf.empty:
        sheets["mtb-chr"] = _build_scalar_sheet(
            mtb_pf, "clinical_harm_rate", "chr_ci_lo", "chr_ci_hi", "errors (P/F)"
        )
        sheets["mtb-ac1"] = _build_scalar_sheet(
            mtb_pf, "gwet_ac1", "gwet_ac1_ci_lo", "gwet_ac1_ci_hi", "errors (P/F)"
        )

    ep = _get_pair_dataset(pair, "epikrise")
    if not ep.empty:
        sheets["epikrise-acc"] = _build_sheet_stratified(
            ep, "accuracy_clear", "accuracy_clear_ci_lo", "accuracy_clear_ci_hi"
        )
        sheets["epikrise-ac1"] = _build_sheet_stratified(
            ep, "gwet_ac1", "gwet_ac1_ci_lo", "gwet_ac1_ci_hi"
        )
    ep_jsd = _get_jsd_dataset(jsd, "epikrise")
    if not ep_jsd.empty:
        sheets["epikrise-jsd_b"] = _build_sheet_stratified(
            ep_jsd, "weighted_jsd", "weighted_jsd_ci_lo", "weighted_jsd_ci_hi"
        )

    return sheets


def _fmt_hh(mean: float, std: float, n: int) -> str:
    if pd.isna(mean):
        return ""
    s = f"{mean:.2f}"
    if not pd.isna(std):
        s += f" ± {std:.2f}"
    s += f" (n={int(n)})"
    return s


def _pivot_hh_by_criterion(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    criteria = _order_criteria(df["criterion"].unique().tolist())
    rows_data: list[dict] = []
    for dataset, gdf in df.groupby("dataset"):
        row: dict[str, str] = {"": dataset}
        for _, r in gdf.iterrows():
            row[r["criterion"]] = _fmt_hh(r["mean"], r.get("std", np.nan), r["n_items"])
        rows_data.append(row)

    result = pd.DataFrame(rows_data)
    cols = [""] + [c for c in criteria if c in result.columns]
    return result.reindex(columns=cols, fill_value="")


def _build_human_reliability_sheets(
    item_pw: pd.DataFrame,
    item_pair: pd.DataFrame,
    item_pf: pd.DataFrame,
    human_recall_mtb: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    sheets: dict[str, pd.DataFrame] = {}

    mae = compute_human_mae(item_pw)
    if not mae.empty:
        sheets["mae_pointwise"] = _pivot_hh_by_criterion(mae)

    acc_pw = compute_human_accuracy_pointwise(item_pw)
    if not acc_pw.empty:
        sheets["accuracy_pointwise_binary"] = _pivot_hh_by_criterion(acc_pw)

    acc_pair = compute_human_accuracy_pairwise(item_pair)
    if not acc_pair.empty:
        sheets["accuracy_pairwise"] = _pivot_hh_by_criterion(acc_pair)

    recall_pf = compute_human_recall_passfail(item_pf)
    if recall_pf.empty and human_recall_mtb is not None and not human_recall_mtb.empty:
        recall_pf = human_recall_mtb
    if not recall_pf.empty:
        sheets["recall_passfail"] = recall_pf.rename(
            columns={"dataset": "", "mean": "recall", "n_items": "n"}
        )

    return sheets


def _write_excel(sheets: dict[str, pd.DataFrame], path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=name, index=False)


def export_tables(
    pw: pd.DataFrame,
    pair: pd.DataFrame,
    pf: pd.DataFrame,
    jsd: pd.DataFrame,
    outdir: Path,
    item_pw: pd.DataFrame | None = None,
    item_pair: pd.DataFrame | None = None,
    item_pf: pd.DataFrame | None = None,
    human_recall_mtb: pd.DataFrame | None = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    n_tables = 3

    sheets1 = _build_alignment_sheets(pw, pair, pf)
    _write_excel(sheets1, outdir / "table1_alignment.xlsx")

    sheets2 = _build_reliability_sheets(pw, pair, pf)
    _write_excel(sheets2, outdir / "table2_reliability.xlsx")

    sheets3 = _build_uncertainty_sheets(pw, pair, pf, jsd)
    _write_excel(sheets3, outdir / "table3_uncertainty_robustness.xlsx")

    if item_pw is not None or item_pair is not None or item_pf is not None:
        sheets4 = _build_human_reliability_sheets(
            item_pw if item_pw is not None else pd.DataFrame(),
            item_pair if item_pair is not None else pd.DataFrame(),
            item_pf if item_pf is not None else pd.DataFrame(),
            human_recall_mtb,
        )
        if sheets4:
            _write_excel(sheets4, outdir / "table4_human_human_reliability.xlsx")
            n_tables += 1

    print(f"Exported {n_tables} Excel tables to {outdir}")
