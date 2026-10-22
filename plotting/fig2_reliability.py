from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from .style import (
    CONFIG_COLORS,
    CONFIG_ORDER,
    CRITERION_ORDER_CONTENT_TO_STYLE,
    FIG2_CRITERION_LABELS,
    FIG_WIDTH_LARGE_CM,
    FIG_WIDTH_SMALL_CM,
    MARKER_EDGECOLOR,
    MARKER_EDGEWIDTH,
    MODEL_LABELS,
    MODEL_MARKERS,
    TINY_MARKER_ALPHA,
    TINY_MARKER_SIZE,
    _add_panel_label,
    _figsize_cm,
    legends_enabled,
    marker_size,
)

_THRESH_COLOR = "#7F7F7F"
_THRESH_SHADE_COLOR = "#7F7F7F"
_XLIM = (-0.3, 1.05)


def _pretty_criterion(criterion: str) -> str:
    return FIG2_CRITERION_LABELS.get(criterion, criterion.replace("_", " ").title())


def _order_criteria(criteria: list[str]) -> list[str]:
    ordered = [c for c in CRITERION_ORDER_CONTENT_TO_STYLE if c in criteria]
    remainder = sorted(c for c in criteria if c not in ordered)
    return ordered + remainder


_TRUE_MACRO_SOURCE = {
    "reports": ("reports", "__macro_ordinal__"),
    "nursing": ("nursing", "__macro_ordinal__"),
    "epikrise": ("epikrise", "__macro__"),
}


def _true_macro_row(
    hcmp_pw: pd.DataFrame | None,
    hcmp_pair: pd.DataFrame | None,
    panel_id: str,
) -> dict | None:
    source = _TRUE_MACRO_SOURCE.get(panel_id)
    if source is None:
        return None
    dataset, macro_criterion = source
    df = hcmp_pw if panel_id in {"reports", "nursing"} else hcmp_pair
    if df is None or df.empty:
        return None
    sub = df[(df["dataset"] == dataset) & (df["criterion"] == macro_criterion)]
    if sub.empty:
        return None

    human_vals = sub["value_hh"].dropna()
    if human_vals.empty:
        return None
    human_ac = float(human_vals.iloc[0])

    ranked = sub.assign(
        _lo=sub["value_model"] - 1.96 * sub["se_model"].fillna(0.0),
        _hi=sub["value_model"] + 1.96 * sub["se_model"].fillna(0.0),
    )
    ranked = ranked.assign(_ci_width=ranked["_hi"] - ranked["_lo"]).sort_values(
        ["value_model", "_ci_width", "model", "config"],
        ascending=[False, True, True, True],
    )
    best = ranked.iloc[0]
    return {
        "best_ac": float(best["value_model"]),
        "best_lo": float(best["_lo"]),
        "best_hi": float(best["_hi"]),
        "best_model": best["model"],
        "best_config": best["config"],
        "human_ac": human_ac,
        "human_lo": human_ac,
        "human_hi": human_ac,
        "all_values": sub["value_model"].dropna().to_numpy(),
    }


def plot_fig2(
    fig3_df: pd.DataFrame,
    fig3_all_df: pd.DataFrame = None,
    hcmp_pw: pd.DataFrame = None,
    hcmp_pair: pd.DataFrame = None,
) -> plt.Figure:
    if fig3_all_df is None:
        fig3_all_df = pd.DataFrame()
    main_df = fig3_df

    if main_df is None or main_df.empty:
        fig, ax = plt.subplots(figsize=_figsize_cm(FIG_WIDTH_SMALL_CM, 7))
        ax.text(
            0.5, 0.5, "No data\n(reliability inputs missing)", ha="center", va="center"
        )
        ax.set_title("Figure 3 — Reliability dumbbell (RQ2)")
        return fig

    panel_names = {
        "reports": "Reports (pointwise)",
        "nursing": "Nursing (pointwise)",
        "epikrise": "Epicrisis (pairwise)",
        "mtb_pf": "MTB (pass/fail)",
    }
    if "panel_name" in main_df.columns:
        for k, v in (
            main_df[["panel", "panel_name"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ):
            if str(k) not in panel_names:
                panel_names[str(k)] = str(v)
    preferred_order = ["reports", "nursing", "epikrise", "mtb_pf"]
    panel_ids = [p for p in preferred_order if p in set(main_df["panel"].unique())]

    if not panel_ids:
        fig, ax = plt.subplots(figsize=_figsize_cm(FIG_WIDTH_SMALL_CM, 10))
        ax.text(
            0.5, 0.5, "No data\n(reliability inputs missing)", ha="center", va="center"
        )
        ax.set_title("Figure 3 — Reliability dumbbell (RQ2)")
        return fig

    stacked_ids = [p for p in ("epikrise", "mtb_pf") if p in panel_ids]
    solo_ids = [p for p in panel_ids if p not in stacked_ids]
    n_cols = len(solo_ids) + (1 if stacked_ids else 0)

    panel_row_counts = {
        panel_id: max(1, int(len(main_df[main_df["panel"] == panel_id])))
        for panel_id in panel_ids
    }
    max_panel_rows = max(panel_row_counts.values()) if panel_row_counts else 1

    fig = plt.figure(figsize=_figsize_cm(FIG_WIDTH_LARGE_CM, 7))
    gs = fig.add_gridspec(
        2,
        n_cols,
        left=0.105,
        right=0.99,
        wspace=0.44,
        hspace=0.5,
        height_ratios=[6, 1],
    )

    axes_by_id: dict = {}
    for col, panel_id in enumerate(solo_ids):
        axes_by_id[panel_id] = fig.add_subplot(gs[:, col])
    for row, panel_id in enumerate(stacked_ids):
        axes_by_id[panel_id] = fig.add_subplot(gs[row, n_cols - 1])

    panel_axes = [axes_by_id[panel_id] for panel_id in panel_ids]
    for panel_id in panel_ids:
        ax = axes_by_id[panel_id]

        sub_raw = main_df[main_df["panel"] == panel_id].copy()
        sub_regular = sub_raw[sub_raw["criterion"] != "average"]
        criterion_order = _order_criteria(sub_regular["criterion"].unique().tolist())
        sub_regular = (
            sub_regular.set_index("criterion").loc[criterion_order].reset_index()
        )
        sub_avg = (
            sub_raw[sub_raw["criterion"] == "average"]
            if panel_id != "mtb_pf"
            else sub_raw.iloc[0:0]
        )
        true_macro = _true_macro_row(hcmp_pw, hcmp_pair, panel_id)
        if not sub_avg.empty and true_macro is not None:
            sub_avg = sub_avg.copy()
            for col in (
                "best_ac",
                "best_lo",
                "best_hi",
                "best_model",
                "best_config",
                "human_ac",
                "human_lo",
                "human_hi",
            ):
                sub_avg[col] = true_macro[col]
        sub = pd.concat([sub_regular, sub_avg], ignore_index=True)
        y_pos = np.arange(len(sub))
        panel_jitter_width = 0.8 * max(1, len(sub)) / max_panel_rows

        for yi, (_, row) in enumerate(sub.iterrows()):
            color = CONFIG_COLORS.get(str(row["best_config"]), "#888888")

            criterion = str(row["criterion"])
            if criterion == "average" and true_macro is not None:
                vals = np.asarray(true_macro["all_values"])
            elif not fig3_all_df.empty:
                vals = np.array(
                    fig3_all_df[
                        (fig3_all_df["panel"] == panel_id)
                        & (fig3_all_df["criterion"] == criterion)
                    ]["ac"]
                    .dropna()
                    .tolist()
                )
            else:
                vals = np.array([])
            if len(vals) >= 1:
                rng = np.random.default_rng(abs(hash((panel_id, criterion))) % (2**32))
                jitter_half = min(0.35, panel_jitter_width / 2)
                jitter = (
                    rng.uniform(-jitter_half, jitter_half, size=len(vals))
                    if len(vals) > 1
                    else np.zeros(1)
                )
                ax.scatter(
                    vals,
                    yi + jitter,
                    s=TINY_MARKER_SIZE**2,
                    color="#7F7F7F",
                    edgecolor="none",
                    alpha=TINY_MARKER_ALPHA,
                    zorder=1,
                )

            best_ac = float(row["best_ac"])
            m_lo, m_hi = float(row["best_lo"]), float(row["best_hi"])

            has_human = pd.notna(row["human_ac"])
            if has_human:
                human_ac = float(row["human_ac"])
                ax.add_patch(
                    Rectangle(
                        (human_ac, yi - 0.35),
                        _XLIM[1] - human_ac,
                        0.7,
                        facecolor=_THRESH_SHADE_COLOR,
                        edgecolor="none",
                        alpha=0.12,
                        zorder=2,
                    )
                )
                ax.plot(
                    [human_ac, human_ac],
                    [yi - 0.35, yi + 0.35],
                    color=_THRESH_COLOR,
                    linewidth=0.7,
                    linestyle=(0, (3, 1.5)),
                    zorder=3,
                )
            model_marker = MODEL_MARKERS.get(str(row["best_model"]), "o")
            ax.errorbar(
                best_ac,
                yi,
                xerr=[[max(0.0, best_ac - m_lo)], [max(0.0, m_hi - best_ac)]],
                fmt=model_marker,
                color=color,
                ecolor=color,
                markersize=marker_size(model_marker),
                markeredgecolor=MARKER_EDGECOLOR,
                markeredgewidth=MARKER_EDGEWIDTH,
                elinewidth=0.5,
                capsize=2,
                capthick=0.5,
                zorder=5,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [_pretty_criterion(str(c)) for c in sub["criterion"].tolist()]
        )
        for yi, label in enumerate(ax.get_yticklabels()):
            if str(sub["criterion"].iloc[yi]) == "average":
                label.set_fontweight("bold")
        ax.invert_yaxis()
        ax.set_xlim(*_XLIM)
        xlabel = {
            "reports": "Gwet's AC2",
            "nursing": "Gwet's AC2 / AC1",
            "epikrise": "Gwet's AC1",
            "mtb_pf": "Gwet's AC1",
        }.get(panel_id, "Gwet's AC")
        ax.set_xlabel(xlabel)

    models_in_plot = (
        sorted(main_df["best_model"].dropna().unique().tolist())
        if not main_df.empty
        else []
    )
    shape_handles = [
        *[
            Line2D(
                [0],
                [0],
                marker=MODEL_MARKERS.get(m, "o"),
                linestyle="None",
                markerfacecolor="#999999",
                markeredgecolor=MARKER_EDGECOLOR,
                markeredgewidth=MARKER_EDGEWIDTH,
                markersize=marker_size(MODEL_MARKERS.get(m, "o")),
                label=MODEL_LABELS.get(m, m),
            )
            for m in models_in_plot
        ],
    ]

    configs = sorted(
        main_df["best_config"].dropna().unique().tolist(),
        key=lambda c: CONFIG_ORDER.index(c) if c in CONFIG_ORDER else len(CONFIG_ORDER),
    )
    config_handles = [
        Patch(facecolor=CONFIG_COLORS[c], edgecolor="black", label=c, linewidth=0.7)
        for c in configs
    ]

    if legends_enabled():
        markers_legend = panel_axes[0].legend(
            handles=shape_handles,
            title="Markers",
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=1,
            frameon=True,
            facecolor="white",
            framealpha=0.95,
            handlelength=1.2,
            borderpad=0.4,
            labelspacing=0.3,
            handletextpad=0.4,
        )
        markers_legend.get_frame().set_linewidth(0.8)
        panel_axes[0].add_artist(markers_legend)

        config_legend = panel_axes[0].add_artist(
            panel_axes[0].legend(
                handles=config_handles,
                title="Config",
                loc="lower left",
                bbox_to_anchor=(0.0, 0.0),
                ncol=1,
                frameon=True,
                facecolor="white",
                framealpha=0.95,
                handlelength=1.2,
                borderpad=0.4,
                labelspacing=0.3,
                handletextpad=0.4,
            )
        )
        config_legend.get_frame().set_linewidth(0.8)

    fig.canvas.draw()
    for i, ax in enumerate(panel_axes):
        _add_panel_label(fig, [ax], chr(ord("a") + i))

    return fig
