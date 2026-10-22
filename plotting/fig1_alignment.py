from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .style import (
    CM_TO_INCH,
    CONFIG_COLORS,
    CONFIG_ORDER,
    FIG2_CRITERION_LABELS,
    FIG2_PANEL_SPECS,
    FIG_WIDTH_LARGE_CM,
    MARKER_EDGECOLOR,
    MARKER_EDGEWIDTH,
    MODEL_LABELS,
    MODEL_MARKERS,
    _add_panel_label,
    _figsize_cm,
    _order_criteria,
    _sorted_models,
    legends_enabled,
    marker_size,
)

_MAE_YLIM = (-0.14, 3.0)
_RECALL_YLIM = (-0.05, 1.05)
_ACC_YLIM = (-0.05, 1.05)

_THRESHOLD_KW = dict(color="#8D8D8D", linestyle="--", linewidth=0.9, zorder=1)
_ERRORBAR_KW = dict(
    elinewidth=0.5,
    capsize=1.0,
    markeredgecolor=MARKER_EDGECOLOR,
    markeredgewidth=MARKER_EDGEWIDTH,
    alpha=0.9,
    zorder=3,
)


def _build_legend_handles(models: list[str], configs: list[str]) -> tuple[list, list]:
    config_handles = [
        Patch(facecolor=CONFIG_COLORS[c], edgecolor="black", label=c, linewidth=0.7)
        for c in configs
    ]
    model_handles = [
        mlines.Line2D(
            [],
            [],
            color="#999999",
            marker=MODEL_MARKERS.get(m, "o"),
            linestyle="None",
            markeredgecolor=MARKER_EDGECOLOR,
            markeredgewidth=MARKER_EDGEWIDTH,
            markersize=marker_size(MODEL_MARKERS.get(m, "o")),
            label=MODEL_LABELS.get(m, m),
        )
        for m in models
    ]
    return config_handles, model_handles


def _plot_best_worst(
    ax: Axes,
    df: pd.DataFrame,
    x_map: dict[str, float],
    higher_is_better: bool = True,
) -> None:
    for criterion, x_pos in x_map.items():
        sub = (
            df[df["criterion"] == criterion]
            .groupby(["model", "config"], as_index=False)[["value", "lo", "hi"]]
            .mean()
        )
        if sub.empty:
            continue

        vals = sub["value"].dropna().tolist()
        if len(vals) >= 2:
            vp = ax.violinplot(
                vals,
                positions=[x_pos],
                widths=0.5,
                showextrema=False,
                showmedians=False,
            )
            for body in vp["bodies"]:
                body.set_facecolor("#A1A1A1")
                body.set_edgecolor("none")
                body.set_linewidth(0)
                body.set_alpha(0.6)
                body.set_zorder(1)

        idx_best = sub["value"].idxmax() if higher_is_better else sub["value"].idxmin()
        idx_worst = sub["value"].idxmin() if higher_is_better else sub["value"].idxmax()

        rows = [sub.loc[idx_best]]
        if idx_worst != idx_best:
            rows.append(sub.loc[idx_worst])

        for row in rows:
            val = float(row["value"])
            lo = float(row["lo"]) if pd.notna(row["lo"]) else val
            hi = float(row["hi"]) if pd.notna(row["hi"]) else val
            model_marker = MODEL_MARKERS.get(str(row["model"]), "o")
            ax.errorbar(
                x_pos,
                val,
                yerr=[[max(0.0, val - lo)], [max(0.0, hi - val)]],
                fmt=model_marker,
                color=CONFIG_COLORS.get(str(row["config"]), "gray"),
                markersize=marker_size(model_marker),
                **_ERRORBAR_KW,
            )


def _draw_mae_thresholds(
    ax: Axes,
    x_map: dict[str, float],
    dataset: str,
    human_mae: dict[tuple[str, str], float] | None,
) -> None:
    if not human_mae:
        return
    for criterion, x_pos in x_map.items():
        val = human_mae.get((dataset, criterion))
        if val is None:
            continue
        ax.hlines(
            val,
            x_pos - 0.3,
            x_pos + 0.3,
            colors=_THRESHOLD_KW["color"],
            linestyles=_THRESHOLD_KW["linestyle"],
            linewidth=_THRESHOLD_KW["linewidth"],
            zorder=_THRESHOLD_KW["zorder"],
        )


def _draw_strip_panel(
    ax: Axes,
    panel_df: pd.DataFrame,
    spec: dict,
    human_mae: dict[tuple[str, str], float] | None = None,
) -> None:
    criteria = spec["criteria"]
    x_map = {c: float(i) for i, c in enumerate(criteria)}

    _plot_best_worst(ax, panel_df, x_map, higher_is_better=(spec["metric"] != "mae"))

    tick_labels = [
        FIG2_CRITERION_LABELS.get(c, c.replace("_", " ").title()) for c in criteria
    ]
    ax.set_xticks(range(len(criteria)))
    ax.set_xticklabels(tick_labels, rotation=34, ha="right")
    ax.set_xlim(-0.3, len(criteria) - 0.6)
    ax.set_title(spec["name"])

    if spec["metric"] == "mae":
        _draw_mae_thresholds(ax, x_map, spec["dataset"], human_mae)
        ax.set_ylim(*_MAE_YLIM)
        ax.set_ylabel("MAE")
    else:
        ax.axhline(float(spec.get("chance", 0.5)), **_THRESHOLD_KW)
        ax.set_ylim(*_ACC_YLIM)
        ax.set_ylabel("Accuracy")


def _draw_merged_nursing_panel(
    ax: Axes,
    mae_df: pd.DataFrame,
    acc_df: pd.DataFrame,
    spec_mae: dict,
    spec_acc: dict,
    human_mae: dict[tuple[str, str], float] | None = None,
) -> None:
    mae_criteria = spec_mae["criteria"]
    acc_criteria = spec_acc["criteria"]
    all_criteria = mae_criteria + acc_criteria
    full_x_map = {c: float(i) for i, c in enumerate(all_criteria)}
    x_map_mae = {c: full_x_map[c] for c in mae_criteria}
    x_map_acc = {c: full_x_map[c] for c in acc_criteria}

    _plot_best_worst(ax, mae_df, x_map_mae, higher_is_better=False)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    _plot_best_worst(ax2, acc_df, x_map_acc, higher_is_better=True)

    tick_labels = [
        FIG2_CRITERION_LABELS.get(c, c.replace("_", " ").title()) for c in all_criteria
    ]
    n = len(all_criteria)
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=34, ha="right")
    ax.set_xlim(-0.3, n - 0.6)
    ax2.set_xlim(-0.3, n - 0.6)

    sep_x = len(mae_criteria) - 0.5
    ax.axvline(sep_x, color="#cccccc", linewidth=0.8, zorder=0)
    ax2.axvline(sep_x, color="#cccccc", linewidth=0.8, zorder=0)

    _draw_mae_thresholds(ax, x_map_mae, spec_mae["dataset"], human_mae)

    ax.set_title(spec_mae["name"])
    ax.set_ylim(*_MAE_YLIM)
    ax.set_ylabel("MAE")

    ax2.hlines(
        float(spec_acc.get("chance", 0.5)),
        sep_x,
        n - 0.6,
        colors=_THRESHOLD_KW["color"],
        linestyles=_THRESHOLD_KW["linestyle"],
        linewidth=_THRESHOLD_KW["linewidth"],
        zorder=_THRESHOLD_KW["zorder"],
    )
    ax2.set_ylim(*_ACC_YLIM)
    ax2.set_ylabel("Accuracy")


def _add_subtitle_line(fig: Figure, ax: Axes, text: str) -> None:
    renderer = fig.canvas.get_renderer()
    title_bb = ax.title.get_window_extent(renderer=renderer)
    fig_bb = fig.get_window_extent(renderer=renderer)
    x_fig = (title_bb.x0 + title_bb.x1) / 2 / fig_bb.width
    y_fig = title_bb.y0 / fig_bb.height
    fig.text(
        x_fig,
        y_fig,
        text,
        transform=fig.transFigure,
        ha="center",
        va="top",
        fontsize=plt.rcParams["axes.titlesize"],
    )


def _draw_passfail_panel(ax: Axes, pf_df: pd.DataFrame) -> None:
    chr_df = pf_df[pf_df["metric"] == "clinical_harm_rate"].copy()
    if chr_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("MTB")
        ax.set_axis_off()
        return

    lo_orig = chr_df["lo"].copy()
    chr_df["value"] = 1.0 - chr_df["value"]
    chr_df["lo"] = 1.0 - chr_df["hi"]
    chr_df["hi"] = 1.0 - lo_orig
    chr_df["criterion"] = "clinical_harm_rate"

    _plot_best_worst(ax, chr_df, {"clinical_harm_rate": 0.0}, higher_is_better=True)

    ax.set_ylim(*_RECALL_YLIM)
    ax.set_ylabel("Recall")
    ax.set_xticks([0])
    ax.set_xticklabels(["Pass/Fail"], rotation=34, ha="right")
    ax.set_xlim(-0.3, 0.3)
    ax.set_title("MTB")


def plot_fig1(
    records: pd.DataFrame,
    passfail_records: pd.DataFrame,
    human_mae: dict[tuple[str, str], float] | None = None,
) -> Figure:
    if records is None or records.empty:
        fig, ax = plt.subplots(figsize=_figsize_cm(FIG_WIDTH_LARGE_CM, 7.5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    _panels = ["reports", "nursing", "nursing_stig", "epikrise"]
    df = records[records["panel"].astype(str).isin(_panels)].copy()

    panel_by_id: dict[str, dict] = {}
    for spec in FIG2_PANEL_SPECS:
        if spec["id"] not in _panels:
            continue
        panel_df = df[df["panel"] == spec["id"]]
        if panel_df.empty:
            continue
        criteria = _order_criteria(
            [c for c in spec["criteria"] if c in set(panel_df["criterion"].unique())]
        )
        if criteria:
            panel_by_id[spec["id"]] = {**spec, "criteria": criteria}

    pf_df = passfail_records if passfail_records is not None else pd.DataFrame()

    reports_spec = panel_by_id.get("reports")
    epikrise_spec = panel_by_id.get("epikrise")
    nursing_spec = panel_by_id.get("nursing")
    nursing_stig_spec = panel_by_id.get("nursing_stig")

    reports_n = len(reports_spec["criteria"]) if reports_spec else 1
    nursing_n = (
        (len(nursing_spec["criteria"]) if nursing_spec else 0)
        + (len(nursing_stig_spec["criteria"]) if nursing_stig_spec else 0)
    ) or 1
    epikrise_n = len(epikrise_spec["criteria"]) if epikrise_spec else 1

    fig_w_in = FIG_WIDTH_LARGE_CM * CM_TO_INCH
    left_margin = 0.49
    gap_single = 0.36
    gap_double = 0.68
    right_margin = 0.10

    overhead = left_margin + gap_single + gap_double + gap_single + right_margin
    per_unit = (fig_w_in - overhead) / (reports_n + nursing_n + epikrise_n + 1)

    reports_w = reports_n * per_unit
    nursing_w = nursing_n * per_unit
    epikrise_w = epikrise_n * per_unit
    mtb_w = per_unit

    reports_x0 = left_margin
    nursing_x0 = reports_x0 + reports_w + gap_single
    epikrise_x0 = nursing_x0 + nursing_w + gap_double
    mtb_x0 = epikrise_x0 + epikrise_w + gap_single

    fig_h_in = 7.4 * CM_TO_INCH
    bottom_margin_in = 0.95
    top_margin_in = 0.35
    ax_h_in = fig_h_in - bottom_margin_in - top_margin_in
    bottom_frac = bottom_margin_in / fig_h_in
    h_frac = ax_h_in / fig_h_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    def _axes(x0_in: float, w_in: float) -> Axes:
        return fig.add_axes((x0_in / fig_w_in, bottom_frac, w_in / fig_w_in, h_frac))

    ax_reports = _axes(reports_x0, reports_w)
    ax_nursing = _axes(nursing_x0, nursing_w)
    ax_epikrise = _axes(epikrise_x0, epikrise_w)
    ax_pf = _axes(mtb_x0, mtb_w)

    if reports_spec:
        _draw_strip_panel(
            ax_reports, df[df["panel"] == "reports"], reports_spec, human_mae
        )
    if nursing_spec and nursing_stig_spec:
        _draw_merged_nursing_panel(
            ax_nursing,
            df[df["panel"] == "nursing"],
            df[df["panel"] == "nursing_stig"],
            nursing_spec,
            nursing_stig_spec,
            human_mae,
        )
    elif nursing_spec:
        _draw_strip_panel(
            ax_nursing, df[df["panel"] == "nursing"], nursing_spec, human_mae
        )
    if epikrise_spec:
        _draw_strip_panel(ax_epikrise, df[df["panel"] == "epikrise"], epikrise_spec)
    _draw_passfail_panel(ax_pf, pf_df)

    all_models = _sorted_models(df["model"].dropna().unique())
    pf_models = (
        _sorted_models(pf_df["model"].dropna().unique()) if not pf_df.empty else []
    )
    models = _sorted_models(set(all_models) | set(pf_models))
    all_configs = set(df["config"].dropna().unique())
    pf_configs = set(pf_df["config"].dropna().unique()) if not pf_df.empty else set()
    configs = [c for c in CONFIG_ORDER if c in all_configs | pf_configs]

    config_handles, model_handles = (
        _build_legend_handles(models, configs) if legends_enabled() else ([], [])
    )
    leg_kw = dict(
        frameon=False,
        columnspacing=0.8,
        handlelength=1.2,
        handletextpad=0.4,
    )
    gap_between_legends = 0.15
    leg1 = leg2 = None
    if config_handles:
        leg1 = fig.legend(
            handles=config_handles,
            title="Config",
            loc="lower left",
            ncol=-(-len(config_handles) // 2),
            bbox_to_anchor=(0.0, 0.01),
            **leg_kw,
        )
        fig.add_artist(leg1)
    if model_handles:
        leg2 = fig.legend(
            handles=model_handles,
            title="Model",
            loc="lower left",
            ncol=-(-len(model_handles) // 2),
            bbox_to_anchor=(0.0, 0.01),
            **leg_kw,
        )
        fig.add_artist(leg2)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    leg1_w_in = (
        leg1.get_window_extent(renderer=renderer).width / fig.dpi if leg1 else 0.0
    )
    leg2_w_in = (
        leg2.get_window_extent(renderer=renderer).width / fig.dpi if leg2 else 0.0
    )
    combined_w = leg1_w_in + (gap_between_legends if leg1 and leg2 else 0.0) + leg2_w_in
    start_x_in = (fig_w_in - combined_w) / 2.0
    if leg1 is not None:
        leg1.set_bbox_to_anchor(
            (start_x_in / fig_w_in, 0.01), transform=fig.transFigure
        )
    if leg2 is not None:
        leg2_x_in = start_x_in + leg1_w_in + gap_between_legends
        leg2.set_bbox_to_anchor((leg2_x_in / fig_w_in, 0.01), transform=fig.transFigure)

    _add_subtitle_line(fig, ax_pf, "(p/f)")

    _add_panel_label(fig, [ax_reports], "a", dx=-0.025)
    _add_panel_label(fig, [ax_nursing], "b", dx=-0.025)
    _add_panel_label(fig, [ax_epikrise], "c", dx=-0.025)
    _add_panel_label(fig, [ax_pf], "d", dx=-0.025)
    return fig
