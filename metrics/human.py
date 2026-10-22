import numpy as np
import pandas as pd
from irrCAC.raw import CAC

from .config import MetricsConfig


def compute_human_baseline_metrics(
    df_pw: pd.DataFrame,
    df_pair: pd.DataFrame,
    config: MetricsConfig,
) -> pd.DataFrame:
    rows: list[dict] = []

    if not df_pw.empty:
        group_cols = ["dataset", "task", "criterion"]
        for group_vals, gdf in df_pw.groupby(group_cols, dropna=False):
            group_dict = dict(zip(group_cols, group_vals))

            criterion = group_dict["criterion"]
            is_binary = criterion in config.binary_criteria

            labels_lists = gdf["human_labels"].tolist()
            if not labels_lists or all(len(ll) < 2 for ll in labels_lists):
                continue

            max_raters = max(len(ll) for ll in labels_lists)
            if max_raters < 2:
                continue

            matrix_rows = [
                ll + [np.nan] * (max_raters - len(ll)) for ll in labels_lists
            ]
            df_ratings = pd.DataFrame(
                matrix_rows,
                columns=[f"rater_{i + 1}" for i in range(max_raters)],
            ).astype(float)

            df_ratings = df_ratings.dropna(how="all")
            if len(df_ratings) < 2:
                continue

            categories = [0, 1] if is_binary else list(range(1, 6))

            weights = "identity" if is_binary else "quadratic"
            cac = CAC(df_ratings, weights=weights, categories=categories)
            res = cac.gwet()
            est = res["est"]
            metric_key = "gwet_ac1_hh" if is_binary else "gwet_ac2_hh"
            _v = est["coefficient_value"]
            _se = est["se"]
            ci_lo = float(np.clip(_v - 1.96 * _se, -1, 1))
            ci_hi = float(np.clip(_v + 1.96 * _se, -1, 1))

            rows.append(
                {
                    **group_dict,
                    "paradigm": "pointwise",
                    "is_binary": is_binary,
                    metric_key: _v,
                    f"{metric_key}_se": _se,
                    f"{metric_key}_ci_lo": ci_lo,
                    f"{metric_key}_ci_hi": ci_hi,
                    "n_items_hh": len(df_ratings),
                    "n_raters": max_raters,
                }
            )

    if not df_pair.empty:
        group_cols = ["dataset", "task", "criterion"]
        for group_vals, gdf in df_pair.groupby(group_cols, dropna=False):
            group_dict = dict(zip(group_cols, group_vals))

            votes_lists = gdf["human_votes"].tolist()
            if not votes_lists:
                continue
            max_raters = max(len(vl) for vl in votes_lists)
            if max_raters < 2:
                continue

            vote_map = {"A": 0, "B": 1, "TIE": 2}
            matrix_rows = [
                [vote_map.get(v, 2) for v in vl] + [np.nan] * (max_raters - len(vl))
                for vl in votes_lists
            ]
            df_ratings = pd.DataFrame(
                matrix_rows,
                columns=[f"rater_{i + 1}" for i in range(max_raters)],
            ).astype(float)
            df_ratings = df_ratings.dropna(how="all")
            if len(df_ratings) < 2:
                continue

            cac = CAC(df_ratings, weights="identity", categories=[0, 1, 2])
            res = cac.gwet()
            est = res["est"]
            _v = est["coefficient_value"]
            _se = est["se"]
            ci_lo = float(np.clip(_v - 1.96 * _se, -1, 1))
            ci_hi = float(np.clip(_v + 1.96 * _se, -1, 1))
            rows.append(
                {
                    **group_dict,
                    "paradigm": "pairwise",
                    "is_binary": False,
                    "gwet_ac1_hh": _v,
                    "gwet_ac1_hh_se": _se,
                    "gwet_ac1_hh_ci_lo": ci_lo,
                    "gwet_ac1_hh_ci_hi": ci_hi,
                    "n_items_hh": len(df_ratings),
                    "n_raters": max_raters,
                }
            )

    df_hh = pd.DataFrame(rows)
    return df_hh
