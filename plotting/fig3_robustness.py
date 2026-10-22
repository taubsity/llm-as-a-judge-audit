from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .style import (
    CM_TO_INCH,
    CONFIG_COLORS,
    CONFIG_ORDER,
    FIG_WIDTH_LARGE_CM,
    FIG_WIDTH_SMALL_CM,
    MARKER_EDGECOLOR,
    MARKER_EDGEWIDTH,
    MARKER_SIZE,
    MODEL_LABELS,
    MODEL_MARKERS,
    _add_panel_label,
    _ax_legend,
    _figsize_cm,
    _model_tier_sort_key,
    legends_enabled,
    marker_size,
)

_STRATUM_ORDER = ["low", "medium", "high"]
_STRATUM_X = {s: i for i, s in enumerate(_STRATUM_ORDER)}

_DATASET_ORDER = ["reports", "nursing", "epikrise"]
_DATASET_LABELS = {
    "reports": "Reports (pointw.)",
    "nursing": "Nursing (pointw.)",
    "epikrise": "Epicrisis (pairw.)",
}

_JSD_YLIM = (0.0, 1.0)
_PERF_YLIM = {"reports": (0.0, 3.0), "nursing": (0.0, 3.0), "epikrise": (0.0, 1.0)}
_PERF_YLABEL = {"reports": "MAE", "nursing": "MAE", "epikrise": "Accuracy"}


def _plot_cell(
    ax,
    df: pd.DataFrame,
    value_col: str,
    config: str,
    dataset: str,
    all_models: list[str],
    color: str,
    ylim: tuple[float, float],
) -> None:
    c_df = df[(df["config"] == config) & (df["dataset"] == dataset)]

    for model in all_models:
        m_df = c_df[c_df["model"] == model]
        pts = {
            _STRATUM_X[s]: float(m_df[m_df["consensus_bin"] == s][value_col].mean())
            for s in _STRATUM_ORDER
            if not m_df[m_df["consensus_bin"] == s].empty
        }
        if len(pts) >= 2:
            xs, ys = zip(*sorted(pts.items()))
            ax.plot(
                xs,
                ys,
                color="black",
                linewidth=0.5,
                linestyle="-",
                marker=MODEL_MARKERS.get(model, "o"),
                markersize=marker_size(MODEL_MARKERS.get(model, "o")),
                markerfacecolor=color,
                markeredgecolor=MARKER_EDGECOLOR,
                markeredgewidth=MARKER_EDGEWIDTH,
                alpha=0.9,
                zorder=3,
            )

    ax.set_ylim(*ylim)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Low", "Med.", "High"])
    ax.set_xlim(-0.5, 2.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))


def _draw_panel(
    subfig,
    panel_label: str,
    configs: list[str],
    datasets: list[str],
    df: pd.DataFrame,
    value_col: str,
    ylim_map: dict[str, tuple[float, float]],
    ylabel_map: dict[str, str],
    all_models: list[str],
    right: float = 0.98,
) -> np.ndarray:
    n_rows = len(datasets)
    n_cols = len(configs)

    axes = subfig.subplots(
        n_rows,
        n_cols,
        sharex=True,
        sharey="row",
        gridspec_kw={"hspace": 0.2, "wspace": 0.15, "left": 0.14, "right": right},
    )
    axes = np.atleast_2d(axes)
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    subfig.text(
        0.02,
        0.95,
        panel_label,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        transform=subfig.transSubfigure,
    )

    for col, config in enumerate(configs):
        color = CONFIG_COLORS.get(config, "gray")
        axes[0, col].set_title(config, fontsize=5)

        for row, dataset in enumerate(datasets):
            ax = axes[row, col]
            _plot_cell(
                ax, df, value_col, config, dataset, all_models, color, ylim_map[dataset]
            )
            if col == 0:
                ax.set_ylabel(ylabel_map.get(dataset, dataset), fontsize=5)
            if col == n_cols - 1:
                ax.annotate(
                    _DATASET_LABELS.get(dataset, dataset),
                    xy=(1.05, 0.5),
                    xycoords="axes fraction",
                    ha="left",
                    va="center",
                    fontsize=7,
                    rotation=-90,
                    annotation_clip=False,
                )
            if row == n_rows - 1:
                ax.set_xlabel("Consensus")

    return axes


def _draw_legend(ax, subfig, all_models: list[str]) -> None:
    if not legends_enabled():
        return
    model_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker=MODEL_MARKERS.get(m, "o"),
            linestyle="",
            markersize=marker_size(MODEL_MARKERS.get(m, "o")),
            markeredgecolor=MARKER_EDGECOLOR,
            markeredgewidth=MARKER_EDGEWIDTH,
            label=MODEL_LABELS.get(m, m),
        )
        for m in all_models
    ]
    pos = ax.get_position()
    leg = subfig.legend(
        handles=model_handles,
        title="Model",
        loc="upper right",
        bbox_to_anchor=(pos.x1 - 0.05, pos.y1 + 0.02),
        bbox_transform=subfig.transSubfigure,
        ncol=3,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        columnspacing=0.8,
        handlelength=1.2,
        fontsize=5,
        title_fontsize=5,
    )
    leg.get_frame().set_linewidth(0.5)


def _plot_position_bias(
    pair: pd.DataFrame,
    pw: pd.DataFrame,
    pf: pd.DataFrame,
    fig: plt.Figure | None = None,
    panel_letters: tuple[str, str, str] = ("a", "b", "c"),
    config_comparison: pd.DataFrame | None = None,
) -> plt.Figure:
    def _overall(df: pd.DataFrame, bin_col: str) -> pd.DataFrame:
        if df.empty:
            return df
        return (
            df[df[bin_col] == "overall"].copy() if bin_col in df.columns else df.copy()
        )

    pair_ov = _overall(pair, "consensus_bin")
    if not pair_ov.empty and "ipi_applicable" in pair_ov.columns:
        pair_ov = pair_ov[pair_ov["ipi_applicable"].fillna(True).astype(bool)]
    pw_ov = _overall(pw, "consensus_bin")

    all_models_set: set[str] = set()
    for df in [pair_ov, pw_ov, pf if not pf.empty else pd.DataFrame()]:
        if not df.empty and "model" in df.columns:
            all_models_set.update(str(m) for m in df["model"].dropna().unique())

    if not all_models_set:
        fig, ax = plt.subplots(figsize=_figsize_cm(FIG_WIDTH_SMALL_CM, 7.5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    models = sorted(all_models_set)

    ipi_per_model: dict[str, dict] = {}
    if not pair_ov.empty and "ipi" in pair_ov.columns:
        for model, m_df in pair_ov.groupby("model"):
            vals = m_df["ipi"].dropna()
            if vals.empty:
                continue
            lo_s = (
                m_df["ipi_ci_lo"].dropna()
                if "ipi_ci_lo" in m_df.columns
                else pd.Series(dtype=float)
            )
            hi_s = (
                m_df["ipi_ci_hi"].dropna()
                if "ipi_ci_hi" in m_df.columns
                else pd.Series(dtype=float)
            )
            acc1_s = (
                m_df["acc_order1"].dropna()
                if "acc_order1" in m_df.columns
                else pd.Series(dtype=float)
            )
            acc2_s = (
                m_df["acc_order2"].dropna()
                if "acc_order2" in m_df.columns
                else pd.Series(dtype=float)
            )

            def _col_mean(col: str) -> float:
                s = (
                    m_df[col].dropna()
                    if col in m_df.columns
                    else pd.Series(dtype=float)
                )
                return float(s.mean()) if not s.empty else np.nan

            def _bstrap_ci(arr: np.ndarray) -> tuple[float, float]:
                from scipy.stats import bootstrap as _bs

                arr = arr[~np.isnan(arr)]
                if len(arr) < 3:
                    return np.nan, np.nan
                res = _bs(
                    (arr,),
                    np.mean,
                    n_resamples=1000,
                    confidence_level=0.95,
                    method="BCa",
                    random_state=42,
                )
                return float(res.confidence_interval.low), float(
                    res.confidence_interval.high
                )

            acc1_arr = (
                m_df["acc_order1"].dropna().values
                if "acc_order1" in m_df.columns
                else np.array([])
            )
            acc2_arr = (
                m_df["acc_order2"].dropna().values
                if "acc_order2" in m_df.columns
                else np.array([])
            )
            acc1_ci = _bstrap_ci(acc1_arr)
            acc2_ci = _bstrap_ci(acc2_arr)

            mc_p_s = (
                m_df["mcnemar_p"].dropna()
                if "mcnemar_p" in m_df.columns
                else pd.Series(dtype=float)
            )
            mc_or_s = (
                m_df["mcnemar_or"].dropna()
                if "mcnemar_or" in m_df.columns
                else pd.Series(dtype=float)
            )

            ipi_per_model[str(model)] = {
                "ipi": float(vals.mean()),
                "ipi_lo": float(lo_s.mean()) if not lo_s.empty else np.nan,
                "ipi_hi": float(hi_s.mean()) if not hi_s.empty else np.nan,
                "acc_order1": float(acc1_s.mean()) if not acc1_s.empty else np.nan,
                "acc_order2": float(acc2_s.mean()) if not acc2_s.empty else np.nan,
                "acc_order1_ci": acc1_ci,
                "acc_order2_ci": acc2_ci,
                "flip_AtoB": _col_mean("flip_AtoB"),
                "flip_BtoA": _col_mean("flip_BtoA"),
                "mcnemar_p": float(mc_p_s.min()) if not mc_p_s.empty else np.nan,
                "mcnemar_or": float(mc_or_s.mean()) if not mc_or_s.empty else np.nan,
            }

    CONFIG_ORDER_SENS = ["rule", "demo"]
    PARADIGM_ORDER = ["pointwise", "pairwise", "passfail"]
    sens_data: dict[str, dict[str, dict[str, float]]] = {m: {} for m in models}

    if not pw_ov.empty and "mae" in pw_ov.columns:
        for model, m_df in pw_ov.groupby("model"):
            ms = str(model)
            if ms not in sens_data:
                continue
            config_means = m_df.groupby("config")["mae"].mean()
            if "baseline" not in config_means.index:
                continue
            base_mae = float(config_means["baseline"])
            sens_data[ms]["pointwise"] = {
                cfg: float(config_means[cfg]) - base_mae
                for cfg in CONFIG_ORDER_SENS
                if cfg in config_means.index
            }

    if not pair_ov.empty and "accuracy_clear" in pair_ov.columns:
        for model, m_df in pair_ov.groupby("model"):
            ms = str(model)
            if ms not in sens_data:
                continue
            config_means = m_df.groupby("config")["accuracy_clear"].mean()
            if "baseline" not in config_means.index:
                continue
            base_acc = float(config_means["baseline"])
            sens_data[ms]["pairwise"] = {
                cfg: float(config_means[cfg]) - base_acc
                for cfg in CONFIG_ORDER_SENS
                if cfg in config_means.index
            }

    if not pf.empty and "recall_micro" in pf.columns:
        for model, m_df in pf.groupby("model"):
            ms = str(model)
            if ms not in sens_data:
                continue
            config_means = m_df.groupby("config")["recall_micro"].mean()
            if "baseline" not in config_means.index:
                continue
            base_rec = float(config_means["baseline"])
            sens_data[ms]["passfail"] = {
                cfg: float(config_means[cfg]) - base_rec
                for cfg in CONFIG_ORDER_SENS
                if cfg in config_means.index
            }

    sig_data: dict[str, dict[str, dict[str, bool]]] = {m: {} for m in models}
    if config_comparison is not None and not config_comparison.empty:
        cc = config_comparison[config_comparison["model"].isin(models)]
        for (model, paradigm, cfg), g in cc.groupby(["model", "paradigm", "config_b"]):
            ms = str(model)
            sig_data.setdefault(ms, {}).setdefault(str(paradigm), {})[str(cfg)] = bool(
                (g["verdict"].isin(["model_a_better", "model_b_better"])).any()
            )

    seed_df_pw = pd.DataFrame()
    if not pw_ov.empty and "seed_std_mean" in pw_ov.columns:
        cols = (
            ["model", "config", "seed_std_mean"]
            if "config" in pw_ov.columns
            else ["model", "seed_std_mean"]
        )
        seed_df_pw = pw_ov[cols].dropna(subset=["seed_std_mean"])

    seed_df_pair = pd.DataFrame()
    if not pair_ov.empty and "seed_variability" in pair_ov.columns:
        cols = (
            ["model", "config", "seed_variability"]
            if "config" in pair_ov.columns
            else ["model", "seed_variability"]
        )
        seed_df_pair = pair_ov[cols].dropna(subset=["seed_variability"])

    seed_df_pf = pd.DataFrame()
    if not pf.empty and "seed_variability" in pf.columns:
        cols = (
            ["model", "config", "seed_variability"]
            if "config" in pf.columns
            else ["model", "seed_variability"]
        )
        seed_df_pf = pf[cols].dropna(subset=["seed_variability"])

    has_seed = not seed_df_pw.empty or not seed_df_pair.empty or not seed_df_pf.empty

    if fig is None:
        fig_h = 9.0 if has_seed else 5.5
        fig = plt.figure(figsize=(FIG_WIDTH_LARGE_CM * CM_TO_INCH, fig_h * CM_TO_INCH))

    if has_seed:
        outer = fig.add_gridspec(
            2, 1, height_ratios=[1.8, 1.0], hspace=0.5, left=0.08, right=0.99
        )
        top_spec = outer[0]
        bottom_spec = outer[1]
    else:
        outer = fig.add_gridspec(1, 1, left=0.08, right=0.99)
        top_spec = outer[0]
        bottom_spec = None

    top_gs = top_spec.subgridspec(
        1,
        4,
        width_ratios=[1.1, 0.9, 0.2, 3.0],
        wspace=0.1,
    )
    ax_acc = fig.add_subplot(top_gs[0])
    ax_ipi = fig.add_subplot(top_gs[1])
    c_gs = top_gs[3].subgridspec(1, 3, wspace=0.4)
    ax_b_list = [fig.add_subplot(c_gs[i]) for i in range(3)]

    n_m = len(models)
    y_pos = {m: i for i, m in enumerate(models)}
    models_sens = [m for m in models if m != "MULTIAGENT"]
    n_m_sens = len(models_sens)

    ax_acc.set_ylim(-0.7, n_m + 0.7)
    ax_acc.invert_yaxis()
    ax_acc.spines["right"].set_visible(False)
    ax_acc.spines["top"].set_visible(False)
    ax_acc.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_acc.set_xlabel("Accuracy")
    ax_acc.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    ax_acc.set_axisbelow(True)
    ax_acc.set_yticks(list(range(n_m)))
    ax_acc.set_yticklabels(
        [MODEL_LABELS.get(m, m) for m in models],
        fontsize=5,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    acc_values = []
    for model in models:
        d = ipi_per_model.get(model, {})
        for v in (d.get("acc_order1"), d.get("acc_order2")):
            if v is not None and not pd.isna(v):
                acc_values.append(float(v))
    if acc_values:
        ax_acc.set_xlim(0.05, 0.7)

    COLOR_LEFT = "black"
    COLOR_RIGHT = "white"
    for model in models:
        yi = y_pos[model]
        d = ipi_per_model.get(model, {})
        acc1 = d.get("acc_order1", np.nan)
        acc2 = d.get("acc_order2", np.nan)
        if pd.isna(acc1) or pd.isna(acc2):
            continue

        ci1 = d.get("acc_order1_ci", (np.nan, np.nan))
        ci2 = d.get("acc_order2_ci", (np.nan, np.nan))

        for acc_val, ci in [
            (acc1, ci1),
            (acc2, ci2),
        ]:
            if not pd.isna(ci[0]) and not pd.isna(ci[1]):
                ax_acc.errorbar(
                    acc_val,
                    yi,
                    xerr=[[max(0.0, acc_val - ci[0])], [max(0.0, ci[1] - acc_val)]],
                    fmt="none",
                    ecolor="black",
                    elinewidth=0.8,
                    capsize=2,
                    capthick=0.5,
                    zorder=3,
                )

        ax_acc.scatter(
            [acc1],
            [yi],
            s=MARKER_SIZE**2,
            color=COLOR_LEFT,
            edgecolor=MARKER_EDGECOLOR,
            linewidth=MARKER_EDGEWIDTH,
            zorder=4,
        )
        ax_acc.scatter(
            [acc2],
            [yi],
            s=MARKER_SIZE**2,
            color=COLOR_RIGHT,
            edgecolor=MARKER_EDGECOLOR,
            linewidth=MARKER_EDGEWIDTH,
            zorder=4,
        )

        ipi_val = d.get("ipi", np.nan)
        if not pd.isna(ipi_val):
            x0, x1 = ax_acc.get_xlim()
            ax_acc.text(
                x0 + (x1 - x0) * 0.01,
                yi,
                f"IPI={ipi_val * 100:.0f}%",
                ha="left",
                va="center",
                fontsize=5,
                color="black",
                zorder=5,
            )

    acc_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=COLOR_LEFT,
            markeredgecolor=MARKER_EDGECOLOR,
            markeredgewidth=MARKER_EDGEWIDTH,
            markersize=MARKER_SIZE,
            label="first",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=COLOR_RIGHT,
            markeredgecolor=MARKER_EDGECOLOR,
            markeredgewidth=MARKER_EDGEWIDTH,
            markersize=MARKER_SIZE,
            label="second",
        ),
    ]
    leg_acc = _ax_legend(
        ax_acc,
        title="Human Position",
        handles=acc_handles,
        loc="lower center",
        frameon=True,
        ncols=2,
        framealpha=0.9,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.4,
        fontsize=5,
    )
    if leg_acc is not None:
        leg_acc.get_frame().set_linewidth(0.5)

    def _sig_stars(p: float) -> str:
        return "*" if (not pd.isna(p) and p < 0.05) else ""

    max_flip = 0.0
    for model in models:
        d = ipi_per_model.get(model, {})
        for k in ("flip_AtoB", "flip_BtoA"):
            v = d.get(k, np.nan)
            if not pd.isna(v):
                max_flip = max(max_flip, float(v))
    x_lim = max(0.35, max_flip * 1.25)

    ax_ipi.set_xlim(-x_lim, x_lim)
    ax_ipi.set_ylim(-0.7, n_m + 0.7)
    ax_ipi.invert_yaxis()
    ax_ipi.spines["left"].set_visible(False)
    ax_ipi.spines["right"].set_visible(False)
    ax_ipi.spines["top"].set_visible(False)
    ax_ipi.xaxis.set_ticks_position("bottom")
    ax_ipi.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{abs(x) * 100:.0f}%")
    )
    ax_ipi.set_xlabel("Swap proportion")
    ax_ipi.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    ax_ipi.set_axisbelow(True)
    ax_ipi.axvline(0.0, color="#222222", linewidth=1.2, zorder=5)
    ax_ipi.set_yticks([])
    ax_ipi.set_yticklabels([])
    ax_ipi.text(
        0.2,
        0.03,
        "→ human",
        transform=ax_ipi.transAxes,
        ha="center",
        va="bottom",
        fontsize=5,
        color="black",
    )
    ax_ipi.text(
        0.8,
        0.03,
        "→ model",
        transform=ax_ipi.transAxes,
        ha="center",
        va="bottom",
        fontsize=5,
        color="black",
    )

    COLOR_BTOA = "black"
    COLOR_ATOB = "white"

    for model in models:
        yi = y_pos[model]
        d = ipi_per_model.get(model, {})
        fAB = d.get("flip_AtoB", np.nan)
        fBA = d.get("flip_BtoA", np.nan)
        ipi_val = d.get("ipi", np.nan)
        if pd.isna(ipi_val):
            continue

        if not pd.isna(fBA) and fBA > 0:
            ax_ipi.barh(
                yi,
                -fBA,
                height=0.55,
                color=COLOR_BTOA,
                alpha=0.88,
                edgecolor="black",
                linewidth=0.4,
                zorder=2,
            )
        if not pd.isna(fAB) and fAB > 0:
            ax_ipi.barh(
                yi,
                fAB,
                height=0.55,
                color=COLOR_ATOB,
                alpha=0.88,
                edgecolor="black",
                linewidth=0.4,
                zorder=2,
            )

        or_val = d.get("mcnemar_or", np.nan)
        if not pd.isna(or_val) and or_val > 0:
            ax_ipi.text(
                -x_lim - 0.02,
                yi,
                f"OR={or_val:.1f}",
                ha="left",
                va="center",
                fontsize=5,
                color="black",
                zorder=4,
            )

        p_val = d.get("mcnemar_p", np.nan)
        stars = _sig_stars(p_val)
        if stars:
            right_side = (not pd.isna(fAB) and not pd.isna(fBA) and fAB >= fBA) or (
                pd.isna(fBA) and not pd.isna(fAB)
            )
            if right_side and not pd.isna(fAB) and fAB > 0:
                ax_ipi.text(
                    fAB + x_lim * 0.02,
                    yi,
                    stars,
                    ha="left",
                    va="center",
                    fontsize=5,
                    color="black",
                    zorder=6,
                )
            elif not pd.isna(fBA) and fBA > 0:
                ax_ipi.text(
                    -fBA - x_lim * 0.02,
                    yi,
                    stars,
                    ha="right",
                    va="center",
                    fontsize=5,
                    color="black",
                    zorder=6,
                )

    PARADIGM_LABELS = {
        "pointwise": "Pointwise  (MAE ↓)",
        "pairwise": "Pairwise (Acc. ↑)",
        "passfail": "Pass/Fail (Recall ↑)",
    }
    BAR_W = 0.22
    CONFIG_OFFSETS = {c: (i - 1) * BAR_W for i, c in enumerate(CONFIG_ORDER_SENS)}

    config_handles = [
        Patch(facecolor=CONFIG_COLORS[c], edgecolor="black", linewidth=0.5, label=c)
        for c in CONFIG_ORDER_SENS
        if c in CONFIG_COLORS
    ]

    for pi, (paradigm, ax_b) in enumerate(zip(PARADIGM_ORDER, ax_b_list)):
        all_deltas = [
            sens_data[m].get(paradigm, {}).get(cfg, np.nan)
            for m in models_sens
            for cfg in CONFIG_ORDER_SENS
        ]
        valid = [d for d in all_deltas if not pd.isna(d)]
        _pos = [d for d in valid if d > 0]
        _neg = [d for d in valid if d < 0]
        y_max = max(max(_pos, default=0.05) * 1.2, 0.05)
        y_min = min(min(_neg, default=-0.02) * 1.2, -0.02)

        ax_b.set_xlim(-0.5, n_m_sens - 0.5)
        ax_b.set_ylim(y_min, y_max)
        ax_b.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        if pi == 0:
            ax_b.set_ylabel("Δ metric vs. baseline")
        ax_b.axhline(0.0, color="#222222", linewidth=1.0, zorder=5)
        ax_b.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        ax_b.set_axisbelow(True)
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)
        ax_b.set_xticks(range(n_m_sens))
        ax_b.set_xticklabels(models_sens, rotation=30, ha="right", fontsize=5)
        ax_b.set_title(PARADIGM_LABELS[paradigm])

        for xi, model in enumerate(models_sens):
            for cfg in CONFIG_ORDER_SENS:
                delta = sens_data[model].get(paradigm, {}).get(cfg, np.nan)
                if pd.isna(delta):
                    continue
                x = xi + CONFIG_OFFSETS[cfg]
                ax_b.bar(
                    x,
                    delta,
                    width=BAR_W * 0.9,
                    color=CONFIG_COLORS.get(cfg, "#888888"),
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=2,
                )
                if sig_data.get(model, {}).get(paradigm, {}).get(cfg, False):
                    pad = (y_max - y_min) * 0.01
                    y_star = delta + pad if delta >= 0 else delta - pad
                    ax_b.text(
                        x,
                        y_star,
                        "*",
                        ha="center",
                        va="bottom" if delta >= 0 else "top",
                        fontsize=6,
                        color="black",
                        zorder=6,
                    )

        if pi == len(PARADIGM_ORDER) - 1:
            leg = _ax_legend(
                ax_b,
                title="Config",
                handles=config_handles,
                loc="upper left",
                frameon=True,
                framealpha=0.9,
                handlelength=1.2,
                handletextpad=0.5,
            )
            if leg is not None:
                leg.get_frame().set_linewidth(0.5)

    if has_seed and bottom_spec is not None:
        bot_gs = bottom_spec.subgridspec(1, 3, wspace=0.2)
        seed_panels = [
            (
                fig.add_subplot(bot_gs[0]),
                seed_df_pw,
                "seed_std_mean",
                "Pointwise",
                "Seed std",
            ),
            (
                fig.add_subplot(bot_gs[1]),
                seed_df_pair,
                "seed_variability",
                "Pairwise",
                "Seed variability",
            ),
            (
                fig.add_subplot(bot_gs[2]),
                seed_df_pf,
                "seed_variability",
                "Pass/Fail",
                "Seed variability",
            ),
        ]
        CONFIG_ORDER_SEED = ["baseline", "rule", "demo", "multiagent"]
        config_palette = {c: CONFIG_COLORS.get(c, "#888888") for c in CONFIG_ORDER_SEED}
        seed_handles = [
            Patch(facecolor=CONFIG_COLORS[c], edgecolor="black", linewidth=0.5, label=c)
            for c in CONFIG_ORDER_SEED
            if c in CONFIG_COLORS
        ]

        n_configs = len(CONFIG_ORDER_SEED)
        GROUP_W = 0.8
        slot_w = GROUP_W / n_configs
        cfg_offsets = [(ci - (n_configs - 1) / 2) * slot_w for ci in range(n_configs)]
        violin_w = slot_w * 0.92
        box_w = slot_w * 0.42

        for i, (ax_s, sdf, val_col, title, ylabel) in enumerate(seed_panels):
            if sdf.empty:
                ax_s.axis("off")
                ax_s.set_title(title)
                continue

            sdf_plot = sdf[sdf["model"].isin(models)].copy()
            models_in = [m for m in models if m in sdf_plot["model"].values]
            has_config = "config" in sdf_plot.columns

            for mi, model in enumerate(models_in):
                m_data = sdf_plot[sdf_plot["model"] == model]
                for ci, cfg in enumerate(CONFIG_ORDER_SEED):
                    pos = mi + cfg_offsets[ci]
                    color = config_palette[cfg]
                    vals = (
                        m_data[m_data["config"] == cfg][val_col].dropna().values
                        if has_config
                        else m_data[val_col].dropna().values
                    )
                    if len(vals) == 0:
                        continue

                    if len(vals) > 1:
                        vp = ax_s.violinplot(
                            vals,
                            positions=[pos],
                            showmeans=False,
                            showmedians=False,
                            showextrema=False,
                            widths=violin_w,
                        )
                        for body in vp["bodies"]:
                            body.set_facecolor(color)
                            body.set_alpha(0.25)
                            body.set_edgecolor("none")
                            body.set_zorder(1)

                    bp = ax_s.boxplot(
                        vals,
                        positions=[pos],
                        widths=box_w,
                        patch_artist=True,
                        showfliers=False,
                        zorder=2,
                    )
                    for box in bp["boxes"]:
                        box.set_facecolor(color)
                        box.set_alpha(0.85)
                        box.set_edgecolor("black")
                        box.set_linewidth(0.5)
                    for element in ["medians", "whiskers", "caps"]:
                        for line in bp[element]:
                            line.set_color("black")
                            line.set_linewidth(0.5)

            ax_s.set_xticks(range(len(models_in)))
            ax_s.set_xticklabels(models_in, rotation=30, ha="right", fontsize=5)
            ax_s.set_xlim(-0.5, len(models_in) - 0.5)
            ax_s.set_ylim(bottom=0.0)
            ax_s.set_title(title)
            ax_s.set_ylabel(ylabel)
            ax_s.set_xlabel("")
            ax_s.grid(True, axis="y", alpha=0.25, linewidth=0.5)
            ax_s.set_axisbelow(True)
            ax_s.spines["top"].set_visible(False)
            ax_s.spines["right"].set_visible(False)

            if i == 2:
                leg_seed = _ax_legend(
                    ax_s,
                    title="Config",
                    handles=seed_handles,
                    loc="lower right",
                    ncols=2,
                    frameon=True,
                    framealpha=0.9,
                    handlelength=1.2,
                    handletextpad=0.5,
                )
                if leg_seed is not None:
                    leg_seed.get_frame().set_linewidth(0.5)

        ax_seed_left = seed_panels[0][0] if seed_panels else None
    else:
        ax_seed_left = None

    fig.canvas.draw()
    _add_panel_label(fig, [ax_acc], panel_letters[0])
    _add_panel_label(fig, [ax_b_list[0]], panel_letters[1])
    if ax_seed_left is not None:
        _add_panel_label(fig, [ax_seed_left], panel_letters[2])

    return fig


def plot_fig3(
    fig4_jsd: pd.DataFrame,
    fig4_perf: pd.DataFrame,
    pair: pd.DataFrame,
    pw: pd.DataFrame,
    pf: pd.DataFrame,
    config_comparison: pd.DataFrame | None = None,
) -> Figure:
    def _filter_mtb(df: pd.DataFrame) -> pd.DataFrame:
        if "dataset" in df.columns:
            return df[df["dataset"] != "mtb"].copy()
        return df.copy()

    jsd = _filter_mtb(fig4_jsd)
    perf = _filter_mtb(fig4_perf)

    all_data = pd.concat([jsd, perf], ignore_index=True)
    all_models = sorted(all_data["model"].unique(), key=_model_tier_sort_key)
    configs = [
        c
        for c in CONFIG_ORDER
        if "config" in all_data.columns and c in all_data["config"].unique()
    ]
    datasets = [
        d
        for d in _DATASET_ORDER
        if "dataset" in all_data.columns and d in all_data["dataset"].unique()
    ]

    fig = plt.figure(figsize=_figsize_cm(FIG_WIDTH_LARGE_CM, 17.5))
    sf_top, sf_bottom = fig.subfigures(2, 1, height_ratios=[9.0, 9.5], hspace=-0.05)

    sf_jsd, sf_perf = sf_top.subfigures(1, 2, wspace=0.08)

    _draw_panel(
        sf_jsd,
        "a",
        configs,
        datasets,
        jsd,
        "mean_jsd",
        {d: _JSD_YLIM for d in datasets},
        {d: "Binned-JSD" for d in datasets},
        all_models,
    )

    perf_axes = _draw_panel(
        sf_perf,
        "b",
        configs,
        datasets,
        perf,
        "metric_value",
        {d: _PERF_YLIM.get(d, (0.0, 1.0)) for d in datasets},
        _PERF_YLABEL,
        all_models,
        right=0.95,
    )

    nursing_row = datasets.index("nursing") if "nursing" in datasets else 0
    _draw_legend(perf_axes[nursing_row, -1], sf_perf, all_models)

    _plot_position_bias(
        pair,
        pw,
        pf,
        fig=sf_bottom,
        panel_letters=("c", "d", "e"),
        config_comparison=config_comparison,
    )

    return fig
