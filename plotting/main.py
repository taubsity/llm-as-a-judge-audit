from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .captions import write_caption
from .data import load_all
from .fig1_alignment import plot_fig1
from .fig2_reliability import plot_fig2
from .fig3_robustness import plot_fig3
from .fig4_decision_impact import plot_fig4
from .human_reliability import compute_human_mae_thresholds
from .style import _save_exact_width, assign_model_colors, setup_style
from .supp_fig2 import plot_supp_fig2_forest
from .tables import export_tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate manuscript figures and supplementary tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("results/metrics"),
        help="Directory containing the metrics CSVs",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for figures and tables",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    setup_style()

    data = load_all(args.metrics_dir)
    pw = data["pw"]
    pair = data["pair"]
    pf = data["pf"]
    jsd = data["jsd"]
    alignment_records = data.get("plot_fig2", pd.DataFrame())
    robustness_jsd = data.get("plot_fig4", pd.DataFrame())
    robustness_perf = data.get("plot_fig4b", pd.DataFrame())
    plot_pf = data.get("plot_pf", pd.DataFrame())
    item_pw = data.get("item_pw", pd.DataFrame())
    item_pair = data.get("item_pair", pd.DataFrame())
    item_pf = data.get("item_pf", pd.DataFrame())
    human_recall_mtb = data.get("human_recall_mtb", pd.DataFrame())
    thresh_pw = data.get("thresh_pw", pd.DataFrame())
    thresh_pair = data.get("thresh_pair", pd.DataFrame())
    hcmp_pw = data.get("hcmp_pw", pd.DataFrame())
    hcmp_pair = data.get("hcmp_pair", pd.DataFrame())
    config_comparison = data.get("config_comparison", pd.DataFrame())

    all_models: set[str] = set()
    for df in [pw, pair, pf, jsd]:
        if not df.empty and "model" in df.columns:
            all_models.update(df["model"].unique())
    assign_model_colors(list(all_models))

    main_outdir = args.outdir / "main"
    main_outdir.mkdir(parents=True, exist_ok=True)
    supp_outdir = args.outdir / "supp"
    supp_outdir.mkdir(parents=True, exist_ok=True)
    export_tables(
        pw,
        pair,
        pf,
        jsd,
        supp_outdir,
        item_pw=item_pw,
        item_pair=item_pair,
        item_pf=item_pf,
        human_recall_mtb=human_recall_mtb,
    )

    fig1 = plot_fig1(alignment_records, plot_pf)
    _save_exact_width(fig1, main_outdir, "fig1_absolute_alignment")
    write_caption(main_outdir, "fig1_absolute_alignment")

    human_mae = compute_human_mae_thresholds(item_pw)
    supp_fig1 = plot_fig1(alignment_records, plot_pf, human_mae=human_mae)
    _save_exact_width(
        supp_fig1, supp_outdir, "supp_fig1_absolute_alignment_human_mae_threshold"
    )
    write_caption(supp_outdir, "supp_fig1_absolute_alignment_human_mae_threshold")

    supp_fig2 = plot_supp_fig2_forest(
        alignment_records,
        plot_pf,
        item_pw,
        item_pair,
        thresh_pw=thresh_pw,
        thresh_pair=thresh_pair,
    )
    _save_exact_width(supp_fig2, supp_outdir, "supp_fig2_forest_by_criterion")
    write_caption(supp_outdir, "supp_fig2_forest_by_criterion")

    reliability_main = data.get("plot_fig3", pd.DataFrame())
    reliability_all = data.get("plot_fig3_all", pd.DataFrame())
    fig2 = plot_fig2(
        reliability_main, reliability_all, hcmp_pw=hcmp_pw, hcmp_pair=hcmp_pair
    )
    _save_exact_width(fig2, main_outdir, "fig2_reliability_dumbbell")
    write_caption(main_outdir, "fig2_reliability_dumbbell")

    fig3 = plot_fig3(
        robustness_jsd,
        robustness_perf,
        pair,
        pw,
        pf,
        config_comparison=config_comparison,
    )
    _save_exact_width(fig3, main_outdir, "fig3_robustness")
    write_caption(main_outdir, "fig3_robustness")

    di_pw = data.get("di_pw", pd.DataFrame())
    di_pair = data.get("di_pair", pd.DataFrame())
    di_pf = data.get("di_pf", pd.DataFrame())
    fig4 = plot_fig4(di_pw, di_pair, di_pf)
    _save_exact_width(fig4, main_outdir, "fig4_decision_impact")
    write_caption(main_outdir, "fig4_decision_impact")

    n_figs = sum(1 for _ in args.outdir.rglob("*.pdf"))
    n_caps = sum(1 for _ in args.outdir.rglob("*.txt"))
    print(f"\n{'=' * 50}")
    print(f"Generated {n_figs} PDF figures, {n_figs} PNG figures, {n_caps} captions")
    print(f"Output: {args.outdir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
