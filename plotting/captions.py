from __future__ import annotations

import re
from pathlib import Path

from .style import (
    CONFIG_COLORS,
    CONFIG_ORDER,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_MARKERS,
    _sorted_models,
)

MIN_WORDS = 100
MAX_WORDS = 300


_COLOR_NAMES = {
    "#E69F00": "orange",
    "#56B4E9": "sky blue",
    "#009E73": "green",
    "#F0E442": "yellow",
    "#0072B2": "blue",
    "#D55E00": "vermillion",
    "#CC79A7": "reddish purple",
    "#000000": "black",
    "#7570B3": "violet",
    "#CFDC72": "pale olive",
    "#666666": "dark grey",
    "#BBBBBB": "light grey",
    "#AAAAAA": "grey",
    "#D6D6D6": "light grey",
}

_MARKER_NAMES = {
    "o": "circle",
    "s": "square",
    "D": "diamond",
    "v": "downward triangle",
    "^": "upward triangle",
    "X": "bold cross",
    "p": "pentagon",
    "h": "hexagon",
    "*": "star",
}


def _color_name(value) -> str:
    if not isinstance(value, str):
        value = "#%02X%02X%02X" % tuple(round(255 * c) for c in value[:3])
    key = value.upper() if value.startswith("#") else value
    return _COLOR_NAMES.get(key, key)


def _models_in_run() -> list[str]:
    return _sorted_models(list(MODEL_COLORS) or list(MODEL_MARKERS))


def _configs() -> str:
    return "; ".join(
        f"{_color_name(CONFIG_COLORS[c])}, {c}"
        for c in CONFIG_ORDER
        if c in CONFIG_COLORS
    )


def _markers() -> str:
    return "; ".join(
        f"{_MARKER_NAMES.get(MODEL_MARKERS.get(m, 'o'), MODEL_MARKERS.get(m, 'o'))}, "
        f"{MODEL_LABELS.get(m, m)}"
        for m in _models_in_run()
    )


def _sankey() -> str:
    from .fig4_decision_impact import _SANKEY_PALETTE

    verdict = {"A": "A", "TIE": "tie", "B": "B"}
    return "; ".join(
        f"{_color_name(c)}, {verdict.get(v, v)}" for v, c in _SANKEY_PALETTE.items()
    )


_CAPTIONS: dict[str, str] = {
    "fig1_absolute_alignment": """\
Fig. 1 | Absolute alignment between LLM judges and human raters, per
criterion. Criterion-level agreement of every judge configuration with the
human reference across the four evaluation settings. a Reports (pointwise):
mean absolute error (MAE) of the judge rating against the human rating, one
column per criterion; lower is better. b Nursing (pointwise): MAE for the
ordinal criteria, followed by the two binary stigma criteria (Stig.
Original, Stig. Summary), which are scored as accuracy. c Epicrisis
(pairwise): accuracy of the judge preference against the human preference.
d MTB (pass-or-fail): recall for clinically harmful content, that is, the
share of human-flagged harmful reports the judge also flags. Small faded
points show every model x configuration combination; the best- and
worst-performing combination per criterion is drawn as a large marker with
its bootstrap confidence interval. Marker shape denotes the judge model
({markers}) and marker fill colour the prompt configuration ({configs}).
Dashed horizontal lines mark the pre-specified acceptability threshold for
each panel's metric: MAE <= 1.0 scale point in a and b, and the chance
level in c (three-way A/tie/B decision) and d.""",
    "supp_fig1_absolute_alignment_human_mae_threshold": """\
Supplementary Fig. 1 | Absolute alignment between LLM judges and human
raters, benchmarked against human-human MAE. Criterion-level agreement of
every judge configuration with the human reference across the four
evaluation settings. a Reports (pointwise): mean absolute error (MAE) of
the judge rating against the human rating, one column per criterion; lower
is better. b Nursing (pointwise): MAE for the ordinal criteria, followed by
the two binary stigma criteria (Stig. Original, Stig. Summary), which are
scored as accuracy. c Epicrisis (pairwise): accuracy of the judge
preference against the human preference, with a dashed line at the chance
level for a three-way A/tie/B decision. d MTB (pass-or-fail): recall for
clinically harmful content, that is, the share of human-flagged harmful
reports the judge also flags. Small faded points show every model x
configuration combination; the best- and worst-performing combination per
criterion is drawn as a large marker with its bootstrap confidence
interval. Marker shape denotes the judge model ({markers}) and marker fill
colour the prompt configuration ({configs}). On the MAE panels a and b,
each criterion carries a dashed horizontal reference line at the mean
leave-one-out MAE between the human raters for that criterion. This places
judge error on the scale of the disagreement the human raters show among
themselves, so a judge falling below the line is within the band of
human-human variability for that criterion.""",
    "supp_fig2_forest_by_criterion": """\
Supplementary Fig. 2 | Per-criterion performance of every judge model x
configuration. Each model x configuration combination is shown as its own
row, with point estimate and bootstrap confidence interval in each
criterion facet. Rows are grouped by prompt configuration and ordered by
model within a group. a Reports (pointwise MAE per criterion, plus a pooled
Macro facet). b Nursing (pointwise MAE per criterion, plus Macro). c
Nursing stigma criteria (accuracy). d MTB pass-or-fail (recall for
clinically harmful content). e Epicrisis (pairwise accuracy per criterion,
plus Macro). Point colour denotes the prompt configuration ({configs}) and
marker shape the judge model ({markers}). The vertical dashed line in each
facet marks the reference threshold for that criterion — the mean
human-human leave-one-out MAE on MAE facets and the chance level on
accuracy facets — and the grey band shades the side of that line which is
better than the human reference, so a confidence interval lying wholly
within the shading indicates performance at or beyond the human-human
level. Macro facets pool all items across a dataset's criteria rather than
averaging the per-criterion values; for nursing the Macro covers the
ordinal criteria only, excluding the two binary stigma criteria.""",
    "fig2_reliability_dumbbell": """\
Fig. 2 | Chance-corrected agreement of the best LLM judge against the
human-human ceiling. Reliability expressed as Gwet's AC, one panel per
evaluation setting: a Reports (pointwise), b Nursing (pointwise), c
Epicrisis (pairwise), d MTB (pass/fail). Each panel has one row per
criterion plus a pooled Macro row where applicable, sorted by the size of
the human-to-model gap. Each row is a dumbbell joining two dots: the
human-human AC, that is, the agreement two human raters reach with each
other and hence the practical ceiling, and the highest model-human AC
observed across all model x configuration combinations; the connecting line
is the remaining gap. Marker shape denotes the judge model achieving that
best value ({markers}) and its fill colour the prompt configuration
({configs}). The best combination is selected on the point estimate and
tie-broken deterministically; it is not a statistically validated ranking,
as many combinations' confidence intervals overlap it. The small grey
jittered points behind each row show the full spread, one per model x
configuration. Macro rows pool all items across a dataset's criteria rather
than averaging the per-criterion AC values; for nursing the Macro covers
the ordinal criteria only, excluding the two binary stigma criteria.""",
    "fig3_robustness": """\
Fig. 3 | Robustness of LLM judges: consensus stratification, position bias,
prompt sensitivity and seed variability. a Judge-human divergence as a
function of human consensus, as a grid with one column per prompt
configuration and one row per dataset (Reports pointwise, Nursing
pointwise, Epicrisis pairwise). Within each cell, items are binned by how
strongly the human raters agreed (low, medium, high) and the y-axis gives
the binned Jensen-Shannon divergence between the judge and human rating
distributions; each connected line is one judge model. b The same grid for
task performance: MAE for the pointwise datasets, accuracy for the pairwise
dataset. c Position bias: pairwise accuracy split by the slot the
human-preferred candidate occupied, as paired dots with bootstrap
confidence intervals (blue, first position; orange, second position); an
offset within a pair is that model's position effect. d Inconsistency under
presentation order, as a diverging butterfly chart — bars left are B-to-A
verdict flips, bars right A-to-B flips, when the same pair is re-presented
in swapped order. e Prompt sensitivity: per-model change relative to the
baseline configuration, one sub-panel per paradigm; the seed sub-panels
give the across-seed standard deviation per model as a violin with a
boxplot in front. Asterisks in d and e mark significance at P < 0.05.
Marker shape denotes the judge model
({markers}) and colour the prompt configuration ({configs}).""",
    "fig4_decision_impact": """\
Fig. 4 | Decision impact of replacing human raters with the best LLM judge.
What changes downstream when the best-performing judge configuration is
substituted for the human raters. a Risk-impact forest plot with one row
per evaluation paradigm, giving the rate at which the substitution changes
the decision with its bootstrap confidence interval: the category-shift
(flip) rate for pointwise, the win-swap rate for pairwise and the miss
rate, 1 - recall, for pass/fail. Each row is labelled with its paradigm,
metric and the judge model and configuration used, and annotated with its
value and the concrete risk it carries — over/under-rating, misleading
comparisons and harmful content deployed, respectively. b Sankey diagram of
the pairwise verdict flow from the human verdict (left) to the LLM verdict
(right); ribbon width is proportional to the number of items on that path
and ribbon colour gives the verdict ({sankey}), so ribbons crossing between
different verdicts are the disagreements. c Net over- versus under-rating
per criterion for the
pointwise task (overall consensus bin). Bars extend right for net
over-rating and left for net under-rating, in percentage points of items,
with the vertical line at zero marking no net bias. The two bars per
criterion are the two pointwise datasets, distinguished by fill (open bars
with black outline, nursing; filled black bars, reports). Criteria are
ordered by mean net rating; "n/a" marks a criterion absent from a
dataset.""",
}


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def render_caption(name: str) -> str:
    try:
        template = _CAPTIONS[name]
    except KeyError:
        raise KeyError(
            f"No caption registered for figure {name!r} — add one to "
            "plotting/captions.py::_CAPTIONS"
        ) from None

    text = " ".join(template.split())
    text = text.format(configs=_configs(), markers=_markers(), sankey=_sankey())

    n = _word_count(text)
    if not MIN_WORDS <= n <= MAX_WORDS:
        raise ValueError(
            f"Caption for {name!r} is {n} words, outside the {MIN_WORDS}-"
            f"{MAX_WORDS} word band — adjust it in plotting/captions.py"
        )
    return text


def write_caption(outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.txt"
    path.write_text(render_caption(name) + "\n", encoding="utf-8")
    return path
