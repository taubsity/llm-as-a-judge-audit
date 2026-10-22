from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from .style import MODEL_MARKERS

ALLOWED_MODELS = set(MODEL_MARKERS)
_MODEL_COLUMNS = ("model", "best_model")


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        return _filter_known_models(df, path)
    return pd.DataFrame()


def _filter_known_models(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    for col in _MODEL_COLUMNS:
        if col not in df.columns:
            continue
        unknown = set(df[col].dropna().unique()) - ALLOWED_MODELS
        if unknown:
            warnings.warn(
                f"{path}: dropping rows with unrecognized {col} value(s) "
                f"{sorted(unknown)} (not in MODEL_MARKERS)"
            )
            df = df[df[col].isin(ALLOWED_MODELS) | df[col].isna()]
    return df


def load_all(metrics_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "pw": load_csv(metrics_dir / "summary_pointwise.csv"),
        "pair": load_csv(metrics_dir / "summary_pairwise.csv"),
        "pf": load_csv(metrics_dir / "summary_passfail.csv"),
        "hh": load_csv(metrics_dir / "human_baseline.csv"),
        "jsd": load_csv(metrics_dir / "consensus_stratified_jsd.csv"),
        "plot_fig2": load_csv(metrics_dir / "plotting_fig2_records.csv"),
        "plot_fig3": load_csv(metrics_dir / "plotting_fig3_main.csv"),
        "plot_fig3_all": load_csv(metrics_dir / "plotting_fig3_all_values.csv"),
        "plot_fig4": load_csv(metrics_dir / "plotting_fig4_points.csv"),
        "plot_fig4b": load_csv(metrics_dir / "plotting_fig4b_points.csv"),
        "plot_pf": load_csv(metrics_dir / "plotting_passfail_records.csv"),
        "item_pw": load_csv(metrics_dir / "item_level_pointwise.csv"),
        "item_pair": load_csv(metrics_dir / "item_level_pairwise.csv"),
        "item_pf": load_csv(metrics_dir / "item_level_passfail.csv"),
        "di_pw": load_csv(metrics_dir / "decision_impact_pointwise.csv"),
        "di_pair": load_csv(metrics_dir / "decision_impact_pairwise.csv"),
        "di_pf": load_csv(metrics_dir / "decision_impact_passfail.csv"),
        "human_recall_mtb": load_csv(metrics_dir / "human_recall_mtb.csv"),
        "thresh_pw": load_csv(metrics_dir / "human_comparison_pointwise_threshold.csv"),
        "thresh_pair": load_csv(
            metrics_dir / "human_comparison_pairwise_threshold.csv"
        ),
        "hcmp_pw": load_csv(metrics_dir / "human_comparison_pointwise.csv"),
        "hcmp_pair": load_csv(metrics_dir / "human_comparison_pairwise.csv"),
        "config_comparison": load_csv(metrics_dir / "config_comparison.csv"),
    }
