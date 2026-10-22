from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .human_reliability import (
    compute_human_accuracy_pairwise,
    compute_human_accuracy_pointwise,
    compute_human_mae_thresholds,
)
from .style import (
    CONFIG_COLORS,
    CONFIG_ORDER,
    FIG2_CRITERION_LABELS,
    FIG2_PANEL_SPECS,
    MODEL_LABELS,
    MODEL_MARKERS,
    _order_criteria,
    _sorted_models,
    legends_enabled,
    marker_size,
)

_MTB_DEFAULT_THRESHOLD = 0.8
_THRESH_COLOR = "#7F7F7F"
_THRESH_SHADE_COLOR = "#7F7F7F"
_SEP_COLOR = "#999999"
_TITLE_FS = 5.0
_TICK_FS = 5.0
_BLOCK_FS = 5.0

_LEFT_PAD = 0.01
_RIGHT_PAD = 0.984
_GUTTER = 0.04
_GUTTER_NARROW = 0.05
_GAP = 0.001
_LABEL_GAP = 0.006
_TOP = 0.965
_BOTTOM = 0.035
_ROW_GAP = 0.09
_FACET_WIDTH_SCALE = 0.90
_ROW_H = (_TOP - _BOTTOM - 2 * _ROW_GAP) / 3
_LEGEND_RATIO = 0.95


def _row_band(row_idx: int) -> tuple[float, float]:
    top = _TOP - row_idx * (_ROW_H + _ROW_GAP)
    return top, top - _ROW_H


def _n_gaps(group_sizes: list[int], legend_ratio: float) -> int:
    n_data = sum(group_sizes)
    n_groups = len(group_sizes)
    return (n_data - n_groups) + (1 if legend_ratio else 0)


def _uniform_facet_width(
    row_specs: list[tuple[list[int], float] | tuple[list[int], float, list[float]]],
) -> float:
    avail = _RIGHT_PAD - _LEFT_PAD
    widths = []
    for spec in row_specs:
        group_sizes, legend_ratio = spec[0], spec[1]
        gutters = spec[2] if len(spec) > 2 else [_GUTTER] * len(group_sizes)
        n_data = sum(group_sizes)
        total_ratio = n_data + legend_ratio
        gaps = _n_gaps(group_sizes, legend_ratio)
        widths.append((avail - sum(gutters) - gaps * _GAP) / total_ratio)
    return min(widths)


def _layout_positions(
    group_sizes: list[int],
    u: float,
    legend_ratio: float = 0.0,
    gutters: list[float] | None = None,
) -> tuple[
    list[tuple[float, float, float, float, list[tuple[float, float]]]],
    tuple[float, float] | None,
]:
    if gutters is None:
        gutters = [_GUTTER] * len(group_sizes)
    n_data = sum(group_sizes)
    gaps = _n_gaps(group_sizes, legend_ratio)
    avail = _RIGHT_PAD - _LEFT_PAD
    consumed = sum(gutters) + (n_data + legend_ratio) * u
    gap = ((avail - consumed) / gaps) if gaps else 0.0

    spans = []
    x = _LEFT_PAD
    for n, gutter in zip(group_sizes, gutters):
        g0, g1 = x, x + gutter
        cols = []
        cx = g1
        for _k in range(n):
            cols.append((cx, cx + u))
            cx += u + gap
        d1 = cx - gap if n else g1
        spans.append((g0, g1, g1, d1, cols))
        x = d1
    if legend_ratio:
        x += gap
        legend_span = (x, x + legend_ratio * u)
    else:
        legend_span = None
    return spans, legend_span


_ROW_LABEL_DISPLAY = {"baseline": "base", "multiagent": "multi-\nagent"}


def _place_row_labels(
    fig: Figure, ax0: Axes, blocks: list[tuple[str, int, int]], label_x: float
) -> None:
    for name, lo_i, hi_i in blocks:
        y_data = (lo_i + hi_i) / 2
        disp = ax0.transData.transform((0, y_data))
        y_fig = fig.transFigure.inverted().transform(disp)[1]
        fig.text(
            label_x,
            y_fig,
            _ROW_LABEL_DISPLAY.get(name, name),
            ha="right",
            va="center",
            fontsize=_BLOCK_FS,
            clip_on=False,
        )


def _build_rows(df: pd.DataFrame) -> tuple[list[tuple[str, str]], list[int]]:
    rows: list[tuple[str, str]] = []
    block_ends: list[int] = []
    for cfg in CONFIG_ORDER:
        models_in_cfg = df.loc[df["config"] == cfg, "model"].dropna().unique().tolist()
        for m in _sorted_models(models_in_cfg):
            rows.append((m, cfg))
        if rows:
            block_ends.append(len(rows) - 1)
    boundaries = block_ends[:-1] if block_ends else []
    return rows, boundaries


def _blocks(rows: list[tuple[str, str]]) -> list[tuple[str, int, int]]:
    blocks: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i][1] != rows[start][1]:
            blocks.append((rows[start][1], start, i - 1))
            start = i
    return blocks


def _sig_dict(df: pd.DataFrame | None) -> dict[tuple[str, str, str, str], bool]:
    if df is None or df.empty or "verdict" not in df.columns:
        return {}
    return {
        (r["dataset"], r["config"], r["model"], r["criterion"]): bool(
            r["verdict"] == "worse_than_human"
        )
        for _, r in df.iterrows()
    }


def _pivot(
    df: pd.DataFrame,
    rows: list[tuple[str, str]],
    criteria: list[str],
    dataset: str,
    sig: dict[tuple[str, str, str, str], bool] | None = None,
) -> dict[tuple[str, str], list[tuple[float, float, float, bool] | None]]:
    agg = df.groupby(["model", "config", "criterion"], as_index=False)[
        ["value", "lo", "hi"]
    ].mean()
    lookup = {
        (r["model"], r["config"], r["criterion"]): (r["value"], r["lo"], r["hi"])
        for _, r in agg.iterrows()
    }
    sig = sig or {}
    table: dict[tuple[str, str], list] = {}
    for row in rows:
        model, config = row
        vals = []
        for c in criteria:
            v = lookup.get((model, config, c))
            vals.append(
                None if v is None else (*v, sig.get((dataset, config, model, c), False))
            )
        table[row] = vals
    return table


def _true_macro(
    thresh: pd.DataFrame | None,
    rows: list[tuple[str, str]],
    dataset: str,
    macro_criterion: str,
    metric: str,
) -> tuple[
    dict[tuple[str, str], tuple[float, float, float, bool] | None], float | None
]:
    out: dict[tuple[str, str], tuple[float, float, float, bool] | None] = {
        row: None for row in rows
    }
    if thresh is None or thresh.empty:
        return out, None
    sub = thresh[
        (thresh["dataset"] == dataset) & (thresh["criterion"] == macro_criterion)
    ]
    lookup = {(r["model"], r["config"]): r for _, r in sub.iterrows()}
    human_vals = sub["value_human"].dropna()
    thresh_human = float(human_vals.iloc[0]) if not human_vals.empty else None
    for row in rows:
        r = lookup.get(row)
        if r is None or pd.isna(r["value_model"]):
            continue
        pt = float(r["value_model"])
        se = float(r["se_model"]) if pd.notna(r["se_model"]) else 0.0
        lo, hi = pt - 1.96 * se, pt + 1.96 * se
        lo = max(0.0, lo)
        if metric == "accuracy":
            hi = min(1.0, hi)
        sig = bool(r["verdict"] == "worse_than_human")
        out[row] = (pt, lo, hi, sig)
    return out, thresh_human


def _span(
    *tables: dict[tuple[str, str], list], pad_frac: float = 0.10
) -> tuple[float, float]:
    vals: list[float] = []
    for table in tables:
        for row_vals in table.values():
            for v in row_vals:
                if v is None:
                    continue
                pt, lo, hi = v[0], v[1], v[2]
                vals.append(lo if pd.notna(lo) else pt)
                vals.append(hi if pd.notna(hi) else pt)
    if not vals:
        return (0.0, 1.0)
    lo_a, hi_a = min(vals), max(vals)
    pad = (hi_a - lo_a) * pad_frac or 0.1
    return (max(0.0, lo_a - pad), hi_a + pad)


def _draw_facet(
    ax: Axes,
    rows: list[tuple[str, str]],
    boundaries: list[int],
    values: list,
    xlim: tuple[float, float],
    title: str,
    thresh: float | None = None,
    better: str | None = "high",
    nbins: int = 3,
    title_bold: bool = False,
) -> None:
    n = len(rows)
    for b in boundaries:
        ax.axhline(b + 0.5, color=_SEP_COLOR, linewidth=0.4, zorder=2)
    if thresh is not None and pd.notna(thresh):
        if better == "low":
            ax.axvspan(xlim[0], thresh, color=_THRESH_SHADE_COLOR, alpha=0.12, zorder=0)
        elif better == "high":
            ax.axvspan(thresh, xlim[1], color=_THRESH_SHADE_COLOR, alpha=0.12, zorder=0)
        ax.axvline(
            thresh,
            color=_THRESH_COLOR,
            linewidth=0.7,
            linestyle=(0, (3, 1.5)),
            zorder=1,
        )
    for i, (model, config) in enumerate(rows):
        v = values[i]
        if v is None:
            continue
        pt, lo, hi, sig = v
        if pd.isna(pt):
            continue
        lo = pt if pd.isna(lo) else lo
        hi = pt if pd.isna(hi) else hi
        marker = MODEL_MARKERS.get(model, "o")
        ax.errorbar(
            pt,
            i,
            xerr=[[max(0.0, pt - lo)], [max(0.0, hi - pt)]],
            fmt=marker,
            color=CONFIG_COLORS.get(config, "gray"),
            ecolor=CONFIG_COLORS.get(config, "gray"),
            elinewidth=0.55,
            capsize=1.0,
            capthick=0.55,
            markersize=marker_size(marker),
            markeredgewidth=0.3,
            markeredgecolor="black",
            zorder=3,
        )
        if sig:
            ax.annotate(
                "*",
                xy=(hi, i),
                xytext=(1.5, -0.5),
                textcoords="offset points",
                fontsize=4.0,
                fontweight="bold",
                color="black",
                va="center",
                ha="left",
                zorder=4,
                clip_on=False,
            )
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    raw = mticker.MaxNLocator(nbins=nbins).tick_values(*xlim)
    eps = (xlim[1] - xlim[0]) * 1e-6
    ax.set_xticks([t for t in raw if xlim[0] - eps <= t <= xlim[1] + eps])
    ax.tick_params(axis="x", labelsize=_TICK_FS, pad=1.2)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(
        title,
        fontsize=_TITLE_FS,
        pad=2,
        linespacing=1.05,
        fontweight="bold" if title_bold else "normal",
    )


def _criteria_for(records: pd.DataFrame, panel_id: str) -> list[str]:
    spec = next(s for s in FIG2_PANEL_SPECS if s["id"] == panel_id)
    present = set(records.loc[records["panel"] == panel_id, "criterion"].unique())
    return _order_criteria([c for c in spec["criteria"] if c in present])


def _label(criterion: str) -> str:
    return FIG2_CRITERION_LABELS.get(criterion, criterion.replace("_", " ").title())


def plot_supp_fig2_forest(
    records: pd.DataFrame,
    passfail_records: pd.DataFrame,
    item_pw: pd.DataFrame,
    item_pair: pd.DataFrame,
    thresh_pw: pd.DataFrame | None = None,
    thresh_pair: pd.DataFrame | None = None,
) -> Figure:
    if records is None or records.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    rows, boundaries = _build_rows(records)
    blocks = _blocks(rows)

    sig_pw = _sig_dict(thresh_pw)
    sig_pair = _sig_dict(thresh_pair)

    mae_thresh = compute_human_mae_thresholds(item_pw)
    acc_pw_thresh = {
        (row["dataset"], row["criterion"]): float(row["mean"])
        for _, row in compute_human_accuracy_pointwise(item_pw).iterrows()
    }
    acc_pair_thresh = {
        (row["dataset"], row["criterion"]): float(row["mean"])
        for _, row in compute_human_accuracy_pairwise(item_pair).iterrows()
    }
    recall_thresh: float = _MTB_DEFAULT_THRESHOLD

    crit_reports = _criteria_for(records, "reports")
    crit_nursing = _criteria_for(records, "nursing")
    crit_nursing_stig = _criteria_for(records, "nursing_stig")
    crit_epikrise = _criteria_for(records, "epikrise")

    reports_df = records[records["panel"] == "reports"]
    nursing_df = records[records["panel"] == "nursing"]
    nursing_stig_df = records[records["panel"] == "nursing_stig"]
    epikrise_df = records[records["panel"] == "epikrise"]

    table_reports = _pivot(reports_df, rows, crit_reports, "reports", sig_pw)
    table_nursing = _pivot(nursing_df, rows, crit_nursing, "nursing", sig_pw)
    table_stig = _pivot(nursing_stig_df, rows, crit_nursing_stig, "nursing", sig_pw)
    table_epikrise = _pivot(epikrise_df, rows, crit_epikrise, "epikrise", sig_pair)

    xlim_reports = xlim_nursing = _span(table_reports, table_nursing)
    xlim_stig = xlim_epikrise = _span(table_stig, table_epikrise)

    avg_reports, avg_thresh_reports = _true_macro(
        thresh_pw, rows, "reports", "__macro_ordinal__", "mae"
    )
    avg_nursing, avg_thresh_nursing = _true_macro(
        thresh_pw, rows, "nursing", "__macro_ordinal__", "mae"
    )
    avg_epikrise, avg_thresh_epikrise = _true_macro(
        thresh_pair, rows, "epikrise", "__macro__", "accuracy"
    )

    table_mtb: dict[tuple[str, str], list] = {row: [None] for row in rows}
    if passfail_records is not None and not passfail_records.empty:
        chr_df = passfail_records[passfail_records["metric"] == "clinical_harm_rate"]
        chr_agg = chr_df.groupby(["model", "config"], as_index=False)[
            ["value", "lo", "hi"]
        ].mean()
        chr_lookup = {
            (r["model"], r["config"]): (
                1.0 - r["value"],
                1.0 - r["hi"],
                1.0 - r["lo"],
                pd.notna(r["hi"]) and (1.0 - r["hi"]) < recall_thresh,
            )
            for _, r in chr_agg.iterrows()
        }
        for row in rows:
            v = chr_lookup.get(row)
            if v is not None:
                table_mtb[row] = [v]

    fig = plt.figure(figsize=(18.0 * (1 / 2.54), 13.2 * (1 / 2.54)))

    row_a_top, row_a_bottom = _row_band(0)
    row_b_top, row_b_bottom = _row_band(1)
    row_c_top, row_c_bottom = _row_band(2)

    row_c_gutters = [_GUTTER, _GUTTER_NARROW, _GUTTER_NARROW]
    legend_ratio = _LEGEND_RATIO if legends_enabled() else 0.0
    u = _uniform_facet_width(
        [
            ([len(crit_reports) + 1], legend_ratio),
            ([len(crit_nursing) + 1], 0.0),
            ([len(crit_nursing_stig), 1, len(crit_epikrise) + 1], 0.0, row_c_gutters),
        ]
    )
    u *= _FACET_WIDTH_SCALE

    (a_span,), legend_span = _layout_positions(
        [len(crit_reports) + 1], u, legend_ratio=legend_ratio
    )
    (b_span,), _ = _layout_positions([len(crit_nursing) + 1], u)
    c_span, d_span, e_span = _layout_positions(
        [len(crit_nursing_stig), 1, len(crit_epikrise) + 1], u, gutters=row_c_gutters
    )[0]

    def _row_axes(
        span: tuple[float, float, float, float, list[tuple[float, float]]],
        top: float,
        bottom: float,
    ) -> list[Axes]:
        return [
            fig.add_axes((left, bottom, right - left, top - bottom))
            for left, right in span[4]
        ]

    axes_a = _row_axes(a_span, row_a_top, row_a_bottom)
    axes_b = _row_axes(b_span, row_b_top, row_b_bottom)
    axes_c = _row_axes(c_span, row_c_top, row_c_bottom)
    axes_d = _row_axes(d_span, row_c_top, row_c_bottom)
    axes_e = _row_axes(e_span, row_c_top, row_c_bottom)

    ax_a0 = None
    for j, crit in enumerate(crit_reports):
        ax = axes_a[j]
        _draw_facet(
            ax,
            rows,
            boundaries,
            [table_reports[r][j] for r in rows],
            xlim_reports,
            _label(crit),
            thresh=mae_thresh.get(("reports", crit)),
            better="low",
        )
        if j == 0:
            ax_a0 = ax
    _draw_facet(
        axes_a[len(crit_reports)],
        rows,
        boundaries,
        [avg_reports[r] for r in rows],
        xlim_reports,
        "Macro",
        thresh=avg_thresh_reports,
        better="low",
        title_bold=True,
    )

    ax_b0 = None
    for j, crit in enumerate(crit_nursing):
        ax = axes_b[j]
        _draw_facet(
            ax,
            rows,
            boundaries,
            [table_nursing[r][j] for r in rows],
            xlim_nursing,
            _label(crit),
            thresh=mae_thresh.get(("nursing", crit)),
            better="low",
        )
        if j == 0:
            ax_b0 = ax
    _draw_facet(
        axes_b[len(crit_nursing)],
        rows,
        boundaries,
        [avg_nursing[r] for r in rows],
        xlim_nursing,
        "Macro",
        thresh=avg_thresh_nursing,
        better="low",
        title_bold=True,
    )

    ax_c0 = None
    for j, crit in enumerate(crit_nursing_stig):
        ax = axes_c[j]
        _draw_facet(
            ax,
            rows,
            boundaries,
            [table_stig[r][j] for r in rows],
            xlim_stig,
            _label(crit),
            thresh=acc_pw_thresh.get(("nursing", crit)),
            better="high",
            nbins=2,
        )
        if j == 0:
            ax_c0 = ax

    ax_d0 = axes_d[0]
    _draw_facet(
        ax_d0,
        rows,
        boundaries,
        [table_mtb[r][0] for r in rows],
        (0.0, 1.0),
        "Recall",
        thresh=recall_thresh,
    )

    ax_e0 = None
    for j, crit in enumerate(crit_epikrise):
        ax = axes_e[j]
        _draw_facet(
            ax,
            rows,
            boundaries,
            [table_epikrise[r][j] for r in rows],
            xlim_epikrise,
            _label(crit),
            thresh=acc_pair_thresh.get(("epikrise", crit)),
            better="high",
        )
        if j == 0:
            ax_e0 = ax
    _draw_facet(
        axes_e[len(crit_epikrise)],
        rows,
        boundaries,
        [avg_epikrise[r] for r in rows],
        xlim_epikrise,
        "Macro",
        thresh=avg_thresh_epikrise,
        better="high",
        title_bold=True,
    )

    if legend_span is not None:
        leg_x0, leg_x1 = legend_span
        ax_leg = fig.add_axes(
            (leg_x0, row_a_bottom, leg_x1 - leg_x0, row_a_top - row_a_bottom)
        )
        ax_leg.axis("off")
        config_handles = [
            mpatches.Patch(facecolor=c, edgecolor="black", linewidth=0.4, label=cfg)
            for cfg, c in CONFIG_COLORS.items()
            if cfg in {cfg for _, cfg in rows}
        ]
        models_present = _sorted_models({m for m, _ in rows})
        model_handles = [
            Line2D(
                [0],
                [0],
                marker=MODEL_MARKERS.get(m, "o"),
                color="black",
                markerfacecolor="#999999",
                linestyle="",
                markersize=marker_size(MODEL_MARKERS.get(m, "o")),
                markeredgewidth=0.4,
                label=MODEL_LABELS.get(m, m),
            )
            for m in models_present
        ]
        leg1 = ax_leg.legend(
            handles=config_handles,
            title="Config",
            loc="upper left",
            fontsize=5,
            title_fontsize=6,
            frameon=False,
            bbox_to_anchor=(-0.4, 1.0),
            handlelength=1.1,
            handletextpad=0.45,
            labelspacing=0.3,
        )
        ax_leg.add_artist(leg1)
        leg2 = ax_leg.legend(
            handles=model_handles,
            title="Model",
            loc="upper left",
            fontsize=5,
            title_fontsize=6,
            frameon=False,
            bbox_to_anchor=(-0.4, 0.62),
            handlelength=1.1,
            handletextpad=0.45,
            labelspacing=0.3,
        )
        ax_leg.add_artist(leg2)

    fig.canvas.draw()
    fh = fig.bbox.height
    panels = [
        (ax_a0, "a", a_span),
        (ax_b0, "b", b_span),
        (ax_c0, "c", c_span),
        (ax_d0, "d", d_span),
        (ax_e0, "e", e_span),
    ]
    for ax0, letter, (g0, g1, _, _, _) in panels:
        title_bb = ax0.title.get_window_extent()
        y = title_bb.y1 / fh + 0.007
        if letter in ("d", "e"):
            fig.text(
                g1 - _LABEL_GAP,
                y,
                letter,
                fontsize=8,
                fontweight="bold",
                va="bottom",
                ha="right",
            )
        else:
            fig.text(g0, y, letter, fontsize=8, fontweight="bold", va="bottom")
            _place_row_labels(fig, ax0, blocks, g1 - _LABEL_GAP)

    return fig
