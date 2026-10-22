from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import MetricsConfig
from .decision_impact import build_decision_impact_frame
from .human import compute_human_baseline_metrics
from .io import load_jsonl, write_outputs
from .normalization import (
    normalize_pairwise_to_long,
    normalize_passfail_to_long,
    normalize_pointwise_to_long,
)
from .pairwise import compute_pairwise_metrics
from .passfail import compute_passfail_metrics
from .plotting_inputs import write_plotting_inputs
from .pointwise import compute_pointwise_metrics
from .stats import (
    compute_consensus_stratified_jsd,
)


def _add_mtb_human_baseline(hh_baseline: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    ac1_path = outdir / "human_recall_mtb_ac1.csv"
    if not ac1_path.exists():
        return hh_baseline

    mtb_ac1 = pd.read_csv(ac1_path)
    row = mtb_ac1[mtb_ac1["severity"] == "erheb_krit"]
    if row.empty:
        return hh_baseline

    v = float(row.iloc[0]["gwet_ac1"])
    se = float(row.iloc[0]["gwet_ac1_se"])
    mtb_row = pd.DataFrame(
        [
            {
                "dataset": "mtb_errors",
                "task": "",
                "criterion": "clinical_harm_rate",
                "paradigm": "passfail",
                "is_binary": True,
                "gwet_ac1_hh": v,
                "gwet_ac1_hh_se": se,
                "gwet_ac1_hh_ci_lo": max(0.0, v - 1.96 * se),
                "gwet_ac1_hh_ci_hi": min(1.0, v + 1.96 * se),
                "n_items_hh": int(row.iloc[0]["n_items"]),
                "n_raters": 2,
            }
        ]
    )
    return pd.concat([hh_baseline, mtb_row], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publication-grade evaluation metrics for LLM-as-a-Judge outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../data_audit/judge_outputs.jsonl"),
        help="Input JSONL file",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/metrics"),
        help="Output directory",
    )
    parser.add_argument(
        "--run-beyond-correlation",
        action="store_true",
        help="Run the BeyondCorrelation analysis (requires a BeyondCorrelation/ clone)",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples",
    )
    parser.add_argument(
        "--compare-models",
        nargs=2,
        action="append",
        metavar=("MODEL_A", "MODEL_B"),
        default=None,
        help="Paired comparison of two models, e.g. Q-235B Q-235B-Think (repeatable)",
    )
    parser.add_argument(
        "--comparison-resamples",
        type=int,
        default=2000,
        help="Cluster-bootstrap resamples for --compare-models",
    )
    parser.add_argument(
        "--compare-to-human",
        action="store_true",
        help="Test every model-config combination against the human reference",
    )
    parser.add_argument(
        "--compare-configs",
        action="store_true",
        help="Test each model's rule/demo config against its own baseline (Fig 3d)",
    )

    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    config = MetricsConfig(
        run_beyond_correlation=args.run_beyond_correlation,
        n_resamples=args.n_resamples,
    )

    (args.outdir / "config.json").write_text(config.to_json())

    print("Loading data...")
    records = load_jsonl(args.input)

    print("Normalizing data...")
    df_pw = normalize_pointwise_to_long(records, config)
    print("Pointwise data normalized.")
    df_pair = normalize_pairwise_to_long(records, config)
    print("Pairwise data normalized.")
    df_pf = normalize_passfail_to_long(records)
    print("Pass/fail data normalized.")

    print("Computing metrics...")
    sum_pw = compute_pointwise_metrics(df_pw, config)
    print("Pointwise metrics computed.")
    sum_pair = compute_pairwise_metrics(df_pair, config)
    print("Pairwise metrics computed.")
    sum_pf = compute_passfail_metrics(df_pf, config)
    print("Pass/fail metrics computed.")

    hh_baseline = compute_human_baseline_metrics(df_pw, df_pair, config)
    hh_baseline = _add_mtb_human_baseline(hh_baseline, args.outdir)
    print("Human baseline metrics computed.")
    consensus_stratified_jsd = compute_consensus_stratified_jsd(df_pw, df_pair, config)
    write_outputs(
        df_pw,
        df_pair,
        df_pf,
        sum_pw,
        sum_pair,
        sum_pf,
        args.outdir,
        human_baseline=hh_baseline,
        stratified_jsd=consensus_stratified_jsd,
    )
    write_plotting_inputs(
        sum_pw,
        sum_pair,
        sum_pf,
        hh_baseline,
        consensus_stratified_jsd,
        args.outdir,
    )

    print("Computing decision-impact metrics...")

    def _load(p: Path) -> pd.DataFrame:
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    item_pw = _load(args.outdir / "item_level_pointwise.csv")
    item_pair = _load(args.outdir / "item_level_pairwise.csv")
    item_pf = _load(args.outdir / "item_level_passfail.csv")
    pw_impact, pair_impact, pf_impact = build_decision_impact_frame(
        item_pw, item_pair, item_pf, n_resamples=args.n_resamples
    )
    for name, df in [
        ("decision_impact_pointwise", pw_impact),
        ("decision_impact_pairwise", pair_impact),
        ("decision_impact_passfail", pf_impact),
    ]:
        if not df.empty:
            df.to_csv(args.outdir / f"{name}.csv", index=False)
    print("Decision-impact metrics written.")

    if args.compare_models:
        from .model_comparison import compare_models

        for model_a, model_b in args.compare_models:
            print(f"Comparing {model_a} vs {model_b}...")
            frames = compare_models(
                item_pw,
                item_pair,
                item_pf,
                model_a,
                model_b,
                config_obj=config,
                n_resamples=args.comparison_resamples,
            )
            suffix = "" if len(args.compare_models) == 1 else f"_{model_a}_vs_{model_b}"
            for name, df in frames.items():
                if not df.empty:
                    df.to_csv(args.outdir / f"{name}{suffix}.csv", index=False)

            integrity = frames["model_comparison_integrity"]
            for _, r in integrity[integrity["status"] != "ok"].iterrows():
                print(
                    f"  [{r['status']}] {r['paradigm']}/{r['dataset']}/{r['config']}: "
                    f"{r['note']}"
                )
            n_failed = int((integrity["status"] == "failed").sum())
            if n_failed:
                print(
                    f"  {n_failed} stratum/strata excluded from the comparison "
                    f"(see model_comparison_integrity{suffix}.csv)"
                )
        print("Model-comparison metrics written.")

    if args.compare_to_human:
        from .human_comparison import compare_to_human

        print("Comparing every model-config combination to the human reference...")
        frames = compare_to_human(
            item_pw,
            item_pair,
            item_pf,
            hh_baseline,
            config_obj=config,
            n_resamples=args.comparison_resamples,
        )
        for name, df in frames.items():
            if not df.empty:
                df.to_csv(args.outdir / f"{name}.csv", index=False)

        integrity = frames["human_comparison_integrity"]
        if not integrity.empty:
            print(
                f"  {len(integrity)} stratum/strata skipped for the reliability-gap "
                f"test (see human_comparison_integrity.csv)"
            )
        print("Human-comparison metrics written.")

    if args.compare_configs:
        from .config_comparison import compare_configs

        print("Comparing each model's rule/demo config to its own baseline...")
        frames = compare_configs(
            item_pw,
            item_pair,
            item_pf,
            config_obj=config,
            n_resamples=args.comparison_resamples,
        )
        for name, df in frames.items():
            if not df.empty:
                df.to_csv(args.outdir / f"{name}.csv", index=False)

        integrity = frames["config_comparison_integrity"]
        n_skipped = (
            int((integrity["status"] == "skipped").sum()) if not integrity.empty else 0
        )
        if n_skipped:
            print(
                f"  {n_skipped} stratum/strata skipped (see config_comparison_integrity.csv)"
            )
        print("Config-comparison metrics written.")

    if config.run_beyond_correlation:
        from .beyond_correlation import (
            run_beyond_correlation_binary,
            run_beyond_correlation_categorical,
            run_beyond_correlation_ordinal,
        )

        if not df_pw.empty:
            run_beyond_correlation_ordinal(df_pw, args.outdir, config)
            run_beyond_correlation_binary(df_pw, args.outdir, config)
        if not df_pair.empty:
            run_beyond_correlation_categorical(df_pair, args.outdir, config)


if __name__ == "__main__":
    main()
