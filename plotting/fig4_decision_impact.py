from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.path import Path as MplPath

from .style import (
    CM_TO_INCH,
    FIG2_CRITERION_LABELS,
    FIG_WIDTH_LARGE_CM,
    MODEL_LABELS,
    _add_panel_label,
    legends_enabled,
)

COLOR_PW = "#000000"
COLOR_PAIR = "#000000"
COLOR_PF = "#000000"

_SANKEY_PALETTE = {"A": "#56B4E9", "TIE": "#D6D6D6", "B": "#E69F00"}
_DATASET_PREFIX = {"nursing": "N", "reports": "R"}


def plot_fig4(
    pw_impact: pd.DataFrame,
    pair_impact: pd.DataFrame,
    pf_impact: pd.DataFrame,
) -> plt.Figure:
    fig = plt.figure(figsize=(FIG_WIDTH_LARGE_CM * CM_TO_INCH, 6.0 * CM_TO_INCH))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.4, 0.7],
        height_ratios=[1.5, 1.0],
        hspace=0.55,
        wspace=0.35,
    )

    ax_forest = fig.add_subplot(gs[0, 0])
    ax_sankey = fig.add_subplot(gs[1, 0])
    ax_rating = fig.add_subplot(gs[:, 1])

    _plot_forest(ax_forest, pw_impact, pair_impact, pf_impact)
    _plot_sankey(ax_sankey, pair_impact)
    _plot_rating_bars_overall(ax_rating, pw_impact)

    fig.canvas.draw()
    _add_panel_label(fig, [ax_forest], "a")
    _add_panel_label(fig, [ax_sankey], "b")
    _add_panel_label(fig, [ax_rating], "c")

    return fig


def _plot_rating_bars_overall(ax: plt.Axes, pw: pd.DataFrame) -> None:
    if pw.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    pw = pw[~pw["criterion"].isin(["stig_original", "stig_summary"])].copy()
    pw = pw[pw["dataset"].isin(_DATASET_PREFIX)]
    pw = pw[pw["consensus_bin"] == "overall"]
    if pw.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    pw["net"] = (pw["up_flip_rate"] - pw["down_flip_rate"]) * 100

    overall_net = pw.groupby("criterion")["net"].mean()
    crits = sorted(pw["criterion"].unique(), key=lambda c: overall_net.get(c, 0))
    n = len(crits)

    dataset_order = [
        d for d in ("nursing", "reports") if d in set(pw["dataset"].dropna())
    ]
    if not dataset_order or n == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    ytick_labels = [
        FIG2_CRITERION_LABELS.get(c, c.replace("_", " ").title()) for c in crits
    ]

    bin_df = pw.set_index(["dataset", "criterion"])["net"]
    group_bar_h = 0.3
    m = len(dataset_order)
    color_map = {"nursing": "white", "reports": "black"}

    ax.axvline(0, color="#333333", linewidth=0.75, zorder=3)

    for i, criterion in enumerate(crits):
        y_center = n - 1 - i
        for j, dataset in enumerate(dataset_order):
            offset = (j - (m - 1) / 2) * group_bar_h
            y_pos = y_center + offset
            v = bin_df.get((dataset, criterion))
            if v is None or pd.isna(v):
                ax.text(
                    2,
                    y_pos,
                    "n/a",
                    va="center",
                    ha="left",
                    fontsize=5,
                    color="#D1D1D1" if dataset == "nursing" else "black",
                    style="italic",
                )
                continue
            ax.barh(
                y_pos,
                v,
                height=group_bar_h,
                color=color_map.get(dataset, "#AAAAAA"),
                linewidth=0.5,
                zorder=4,
                edgecolor="black",
            )

    ax.set_xlim(-110, 110)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xticklabels(["-100", "-50", "0", "+50", "+100"])
    ax.set_xlabel("Net over/under-rating (%)")
    ax.set_axisbelow(True)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ytick_labels[::-1])
    if legends_enabled():
        ax.legend(
            handles=[
                mpatches.Patch(
                    facecolor=color_map.get(d, "#AAAAAA"),
                    edgecolor="black",
                    label=d.title(),
                    linewidth=0.5,
                )
                for d in dataset_order
            ],
            title="Dataset",
            loc="lower left",
        )


def _plot_forest(
    ax: plt.Axes,
    pw: pd.DataFrame,
    pair: pd.DataFrame,
    pf: pd.DataFrame,
) -> None:
    rows: list[dict] = []

    def _mc(df: pd.DataFrame) -> str:
        if df.empty or "model" not in df.columns or "config" not in df.columns:
            return ""
        m = df["model"].dropna().unique()
        c = df["config"].dropna().unique()
        if len(m) == 0 or len(c) == 0:
            return ""
        return f"{m[0]}/{c[0]}"

    if not pw.empty:
        ov = pw[pw["consensus_bin"] == "overall"]
        if not ov.empty:
            rows.append(
                {
                    "label": f"Pointwise\n(Category Shift)\n{_mc(ov)}",
                    "value": ov["flip_rate"].mean(),
                    "lo": ov["flip_rate_ci_lo"].mean(),
                    "hi": ov["flip_rate_ci_hi"].mean(),
                    "color": COLOR_PW,
                    "risk": "Over/under-rating",
                }
            )

    if not pair.empty:
        ov = pair[pair["consensus_bin"] == "overall"]
        if not ov.empty:
            rows.append(
                {
                    "label": f"Pairwise\n(Win Swap)\n{_mc(ov)}",
                    "value": ov["flip_rate"].mean(),
                    "lo": ov["flip_rate_ci_lo"].mean(),
                    "hi": ov["flip_rate_ci_hi"].mean(),
                    "color": COLOR_PAIR,
                    "risk": "Misleading comparisons",
                }
            )

    if not pf.empty:
        ov = pf[pf["consensus_bin"] == "overall"]
        if not ov.empty:
            miss = 1.0 - ov["recall_llm"].values[0]
            rows.append(
                {
                    "label": f"Pass/Fail\n(1 − Recall)\n{_mc(ov)}",
                    "value": miss,
                    "lo": 1.0 - ov["recall_llm_ci_hi"].values[0],
                    "hi": 1.0 - ov["recall_llm_ci_lo"].values[0],
                    "color": COLOR_PF,
                    "risk": "Harmful content deployed",
                }
            )

    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    n = len(rows)
    for i, r in enumerate(rows):
        y = n - 1 - i
        ax.errorbar(
            r["value"],
            y,
            xerr=[[r["value"] - r["lo"]], [r["hi"] - r["value"]]],
            fmt="o",
            color=r["color"],
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=0.4,
            elinewidth=0.9,
            capsize=2,
            capthick=2.5,
            zorder=3,
        )
        ax.text(
            (r["lo"] - 0.01) if r["hi"] > 0.3 else (r["hi"] + 0.01),
            y,
            f"{r['value']:.1%}\n{r['risk']}",
            va="center",
            ha="left" if r["hi"] < 0.3 else "right",
            color="black",
            fontsize=5,
        )

    ax.set_yticks(list(range(n)))
    ax.set_yticklabels([r["label"] for r in reversed(rows)])
    ax.set_xlabel("Impact metric")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_axisbelow(True)
    ax.set_xlim(0.0, 0.7)
    ax.set_ylim(-0.5, n - 0.5)


def _plot_sankey(ax: plt.Axes, pair: pd.DataFrame) -> None:
    if pair.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    ov = pair[pair["consensus_bin"] == "overall"]
    if ov.empty:
        ax.text(0.5, 0.5, "No overall data", ha="center", va="center")
        return

    flow_cols = [c for c in ov.columns if c.startswith("flow_")]
    if not flow_cols:
        ax.text(0.5, 0.5, "No flow data", ha="center", va="center")
        return

    flows = {col: int(ov[col].sum()) for col in flow_cols}
    categories = ["A", "TIE", "B"]
    category_labels = {"A": "Human", "B": "LLM", "TIE": "TIE"}
    total = sum(flows.values())
    if total == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    y_bottom, y_top, bar_h = -0.05, 1.05, 0.4

    src_totals = {
        s: sum(flows.get(f"flow_{s}_{d}", 0) for d in categories) for s in categories
    }
    dst_totals = {
        d: sum(flows.get(f"flow_{s}_{d}", 0) for s in categories) for d in categories
    }

    def _stack(totals: dict) -> dict:
        pos, y = {}, 0.0
        for cat in categories:
            h = totals[cat] / total
            pos[cat] = (y, h)
            y += h
        return pos

    src_pos = _stack(src_totals)
    dst_pos = _stack(dst_totals)

    for side, pos, totals, y_bar in [
        ("src", src_pos, src_totals, y_top - bar_h),
        ("dst", dst_pos, dst_totals, y_bottom),
    ]:
        for cat in categories:
            x, w = pos[cat]
            ax.bar(
                x + w / 2,
                bar_h,
                width=w,
                bottom=y_bar,
                color=_SANKEY_PALETTE[cat],
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
            if w > 0.05:
                pct = totals[cat] / total
                ax.text(
                    x + w / 2,
                    y_bar + bar_h / 2,
                    f"{category_labels.get(cat, cat)}\n{pct:.1%}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=5,
                    zorder=4,
                )

    src_cursors = {cat: src_pos[cat][0] for cat in categories}
    dst_cursors = {cat: dst_pos[cat][0] for cat in categories}
    y0, y1 = y_top - bar_h, y_bottom + bar_h

    for src in categories:
        for dst in categories:
            count = flows.get(f"flow_{src}_{dst}", 0)
            if count == 0:
                continue
            w = count / total
            x_s, x_d = src_cursors[src], dst_cursors[dst]
            mid = (y0 + y1) / 2
            verts = [
                (x_s, y0),
                (x_s, mid),
                (x_d, mid),
                (x_d, y1),
                (x_d + w, y1),
                (x_d + w, mid),
                (x_s + w, mid),
                (x_s + w, y0),
                (x_s, y0),
            ]
            codes = [
                MplPath.MOVETO,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.LINETO,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CLOSEPOLY,
            ]
            ax.add_patch(
                mpatches.PathPatch(
                    MplPath(verts, codes),
                    facecolor=_SANKEY_PALETTE[src],
                    edgecolor="none",
                    alpha=0.5 if src == dst else 0.3,
                    zorder=2,
                )
            )
            src_cursors[src] += w
            dst_cursors[dst] += w

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([])
    ax.set_yticks([y_bottom + bar_h / 2, y_top - bar_h / 2])
    ax.set_yticklabels([MODEL_LABELS.get("G-120B", "G-120B"), "Human"])
    for spine in ax.spines.values():
        spine.set_visible(False)

    dwr = ov["delta_win_rate"].mean()
    ax.text(
        0.5,
        -0.08,
        f"$\\Delta$ win-rate (Human): {dwr:+.1%}",
        ha="center",
        va="top",
        fontsize=5,
        transform=ax.transAxes,
    )
