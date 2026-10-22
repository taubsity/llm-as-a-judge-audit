from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.pylab import Axes

PALETTE = sns.color_palette(
    [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#000000",
        "#7570B3",
        "#CFDC72",
        "#666666",
    ]
)
MODEL_COLORS: dict[str, Any] = {}

SHOW_LEGENDS = False


def legends_enabled() -> bool:
    return SHOW_LEGENDS


CM_TO_INCH = 1 / 2.54
FIG_WIDTH_SMALL_CM = 9.0
FIG_WIDTH_LARGE_CM = 18.0

SAVE_DPI = 600

CONFIG_ORDER = ["baseline", "rule", "demo", "multiagent"]
CONFIG_COLORS = {
    "baseline": "#0072B2",
    "rule": "#009E73",
    "demo": "#E69F00",
    "multiagent": "#D55E00",
}

MARKER_SIZE = 3
MARKER_EDGEWIDTH = 0.7
MARKER_EDGECOLOR = "black"

MARKER_SIZE_MULT = {
    "o": 1.0,
    "s": 1.0,
    "D": 0.9,
    "v": 1.05,
    "^": 1.1,
    "X": 1.15,
    "p": 1.15,
    "h": 1.15,
    "*": 1.7,
}

TINY_MARKER_SIZE = 2.2
TINY_MARKER_ALPHA = 0.55


def marker_size(marker: str, scale: float = 1.0) -> float:
    return MARKER_SIZE * scale * MARKER_SIZE_MULT.get(marker, 1.0)


MODEL_LABELS: dict[str, str] = {
    "G-120B": "gpt-oss-120b",
    "Q-235B": "Qwen3-235B",
    "Q-80B": "Qwen3-Next-80B",
    "MG-27B": "medgemma-27b",
    "M-24B": "Magistral-Small",
    "Q-235B-Think": "Qwen3-235B-Think",
}

MODEL_MARKERS = {
    "G-120B": "o",
    "Q-122B": "^",
    "Q-235B": "v",
    "L-70B": "s",
    "Q-80B": "D",
    "MG-27B": "s",
    "M-24B": "p",
    "O-32B": "h",
    "MULTIAGENT": "X",
    "Q-235B-Think": "*",
}

MODEL_SIZE_TIER: dict[str, str] = {
    "G-120B": "big",
    "Q-122B": "big",
    "L-70B": "big",
    "Q-80B": "big",
    "M-24B": "small",
    "O-32B": "small",
    "MG-27B": "small",
    "Q-235B-Think": "big",
}

CRITERION_ORDER_CONTENT_TO_STYLE = [
    "clinical_utility",
    "accurate",
    "thorough",
    "citation",
    "abstraction",
    "useful",
    "succinct",
    "comprehensible",
    "organized",
    "completeness",
    "efficiency",
    "organization",
    "overall",
    "human_written",
]

FIG2_PANEL_SPECS = [
    {
        "id": "reports",
        "name": "Reports (pointwise)",
        "metric": "mae",
        "dataset": "reports",
        "criteria": [
            "accurate",
            "thorough",
            "useful",
            "clinical_utility",
            "succinct",
            "organized",
            "comprehensible",
        ],
        "threshold": 1.0,
    },
    {
        "id": "nursing",
        "name": "Nursing (pointwise)",
        "metric": "mae",
        "dataset": "nursing",
        "criteria": [
            "abstraction",
            "accurate",
            "citation",
            "thorough",
            "useful",
            "succinct",
            "organized",
            "comprehensible",
        ],
        "threshold": 1.0,
    },
    {
        "id": "nursing_stig",
        "name": "Nursing Stig (pointwise)",
        "metric": "accuracy",
        "dataset": "nursing",
        "criteria": ["stig_original", "stig_summary"],
        "chance": 0.5,
    },
    {
        "id": "epikrise",
        "name": "Epicrisis (pairwise)",
        "metric": "accuracy",
        "dataset": "epikrise",
        "criteria": [
            "completeness",
            "efficiency",
            "organized",
            "clinical_utility",
            "human_written",
        ],
        "chance": 0.33,
    },
    {
        "id": "mtb_pf",
        "name": "MTB (pass-or-fail)",
        "metric": "chr",
        "dataset": "mtb_errors",
        "criteria": ["clinical_harm_rate"],
        "threshold": 0.1,
    },
]

FIG2_CRITERION_LABELS = {
    "accurate": "Accurate",
    "thorough": "Thorough",
    "useful": "Useful",
    "organized": "Organized",
    "comprehensible": "Comprehensible",
    "succinct": "Succinct",
    "abstraction": "Abstractive",
    "citation": "Cited",
    "stig_original": "Stig. Original",
    "stig_summary": "Stig. Summary",
    "clinical_utility": "Clinical Utility",
    "completeness": "Complete",
    "efficiency": "Efficient",
    "human_written": "Human-Written",
    "clinical_harm_rate": "Recall",
    "average": "Macro",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 5,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
            "legend.title_fontsize": 5,
            "legend.frameon": True,
            "figure.dpi": SAVE_DPI,
            "savefig.dpi": SAVE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def assign_model_colors(models: list[str]) -> None:
    global MODEL_COLORS
    for i, m in enumerate(sorted(models)):
        MODEL_COLORS[m] = PALETTE[i % len(PALETTE)]


def _figsize_cm(width_cm: float, height_cm: float) -> tuple[float, float]:
    return (width_cm * CM_TO_INCH, height_cm * CM_TO_INCH)


def _ax_legend(ax: plt.Axes, *, title: str, **kwargs):
    if not legends_enabled():
        return None
    return ax.legend(title=title, **kwargs)


def _save_exact_width(
    fig: plt.Figure, outdir: Path, name: str, width_cm: float = FIG_WIDTH_LARGE_CM
) -> None:
    from matplotlib.transforms import Bbox

    outdir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight_bbox_in = fig.get_tightbbox(renderer)
    pad_in = plt.rcParams.get("savefig.pad_inches", 0.05)
    width_in = width_cm * CM_TO_INCH
    bbox = Bbox.from_extents(
        0.0,
        tight_bbox_in.y0 - pad_in,
        width_in,
        tight_bbox_in.y1 + pad_in,
    )
    for ext in ("pdf", "png"):
        path = outdir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches=bbox, dpi=SAVE_DPI)
    plt.close(fig)


def _model_tier_sort_key(model: str) -> tuple[int, str]:
    tier = MODEL_SIZE_TIER.get(model, "big")
    return (0 if tier == "big" else 1, model)


def _sorted_models(models, *, include_multiagent: bool = True) -> list[str]:
    return sorted(
        (m for m in models if include_multiagent or m != "MULTIAGENT"),
        key=lambda m: (1, "") if m == "MULTIAGENT" else (0, _model_tier_sort_key(m)),
    )


def _order_criteria(criteria: list[str]) -> list[str]:
    ordered = [c for c in CRITERION_ORDER_CONTENT_TO_STYLE if c in criteria]
    remainder = sorted(c for c in criteria if c not in ordered)
    return ordered + remainder


def _add_panel_label(
    fig: Figure, axes: list[Axes], letter: str, dx: float = -0.07
) -> None:
    if not axes:
        return
    boxes = [ax.get_position() for ax in axes]
    fig.text(
        min(b.x0 for b in boxes) + dx,
        max(b.y1 for b in boxes) + 0.02,
        letter,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        transform=fig.transSubfigure,
        clip_on=False,
    )
