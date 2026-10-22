from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import brentq
from scipy.stats import binom, binomtest
from statsmodels.stats.contingency_tables import mcnemar as _mcnemar_test

from .config import MetricsConfig
from .stats import _bca_ci_scipy, compute_gwet_ac1


def _median_unbiased_or(n_AB: int, n_BA: int) -> float:
    n = n_AB + n_BA
    if n == 0:
        return np.nan
    if n_AB == 0:
        return 0.0
    if n_BA == 0:
        p = 0.5 ** (1.0 / n)
        return p / (1.0 - p)
    p = brentq(lambda p: binom.sf(n_AB - 1, n, p) - 0.5, 1e-12, 1.0 - 1e-12)
    return p / (1.0 - p)


def _process_pairwise_group(group_vals, group_df, config):
    group_cols = ["dataset", "model", "config", "criterion"]
    group_dict = dict(zip(group_cols, group_vals))
    n_resamples = config.n_resamples
    vote_categories = ["A", "B", "TIE"]

    strata: list[tuple[str, pd.DataFrame]] = [("overall", group_df)]
    bin_names = group_df["consensus_bin"].drop_duplicates().tolist()
    for bin_name in bin_names:
        mask = group_df["consensus_bin"].eq(bin_name)
        strata.append((str(bin_name), group_df.loc[mask, :]))

    results = []
    for stratum_name, sdf in strata:
        n_total = len(sdf)
        if n_total < 2:
            continue

        acc_match_all = (sdf["human_majority"] == sdf["llm_majority"]).astype(float)

        n_clear = int(sdf["is_clear_majority"].sum())
        n_ambig = n_total - n_clear
        clear_df = sdf[sdf["is_clear_majority"]]

        acc_clear = np.nan
        if len(clear_df) > 0:
            acc_clear = float(
                (clear_df["human_majority"] == clear_df["llm_majority"])
                .astype(float)
                .mean()
            )

        acc_all = float(acc_match_all.mean())

        vote_to_idx = {cat: i for i, cat in enumerate(vote_categories)}
        human_numerical = sdf["human_majority"].map(vote_to_idx).to_numpy()
        llm_numerical = sdf["llm_majority"].map(vote_to_idx).to_numpy()

        ac1 = compute_gwet_ac1(
            human_numerical,
            llm_numerical,
            categories=list(range(len(vote_categories))),
        )

        ipi_applicable = True
        swap_df = sdf[sdf["llm_votes_swapped"].apply(lambda x: len(x) > 0)]
        ipi = np.nan
        acc_order1 = np.nan
        acc_order2 = np.nan
        if ipi_applicable and len(swap_df) > 0:
            inconsistent = (
                swap_df["llm_majority"] != swap_df["llm_majority_swapped"]
            ).sum()
            ipi = float(inconsistent / len(swap_df))
            acc_order1 = float(
                (swap_df["human_majority"] == swap_df["llm_majority"]).mean()
            )
            acc_order2 = float(
                (swap_df["human_majority"] == swap_df["llm_majority_swapped"]).mean()
            )

        p_vote_A_orig = np.nan
        p_vote_A_swap = np.nan
        p_tie_orig = np.nan
        p_tie_swap = np.nan
        flip_AtoB = np.nan
        flip_BtoA = np.nan
        mcnemar_stat = np.nan
        mcnemar_p = np.nan
        mcnemar_or = np.nan
        binom_p_identity = np.nan
        binom_p_position = np.nan
        cohens_g_identity = np.nan
        cohens_g_position = np.nan

        if ipi_applicable and len(swap_df) > 0:
            orig_votes = swap_df["llm_majority"].values
            swap_votes = swap_df["llm_majority_swapped"].values
            n_total_swap = len(swap_df)

            n_A_orig = int((orig_votes == "A").sum())
            n_B_orig = int((orig_votes == "B").sum())
            n_TIE_orig = int((orig_votes == "TIE").sum())
            n_A_swap = int((swap_votes == "A").sum())
            n_B_swap = int((swap_votes == "B").sum())
            n_TIE_swap = int((swap_votes == "TIE").sum())

            p_vote_A_orig = n_A_orig / n_total_swap
            p_tie_orig = n_TIE_orig / n_total_swap
            p_vote_A_swap = n_A_swap / n_total_swap
            p_tie_swap = n_TIE_swap / n_total_swap

            n_A_total = n_A_orig + n_A_swap
            n_B_total = n_B_orig + n_B_swap
            n_non_tie_pool = n_A_total + n_B_total
            if n_non_tie_pool >= 2:
                binom_p_identity = binomtest(
                    n_A_total, n_non_tie_pool, p=0.5, alternative="two-sided"
                ).pvalue
                cohens_g_identity = abs(n_A_total / n_non_tie_pool - 0.5)

            n_first = n_A_orig + n_B_swap
            n_second = n_B_orig + n_A_swap
            n_pos_total = n_first + n_second
            if n_pos_total >= 2:
                binom_p_position = binomtest(
                    n_first, n_pos_total, p=0.5, alternative="two-sided"
                ).pvalue
                cohens_g_position = abs(n_first / n_pos_total - 0.5)

            non_tie_mask = (orig_votes != "TIE") & (swap_votes != "TIE")
            n_nt = int(non_tie_mask.sum())
            if n_nt >= 2:
                orig_nt = orig_votes[non_tie_mask]
                swap_nt = swap_votes[non_tie_mask]

                n_AA = int(((orig_nt == "A") & (swap_nt == "A")).sum())
                n_AB = int(((orig_nt == "A") & (swap_nt == "B")).sum())
                n_BA = int(((orig_nt == "B") & (swap_nt == "A")).sum())
                n_BB = int(((orig_nt == "B") & (swap_nt == "B")).sum())

                flip_AtoB = n_AB / n_nt
                flip_BtoA = n_BA / n_nt

                if n_AB + n_BA >= 2:
                    table = np.array([[n_AA, n_AB], [n_BA, n_BB]])
                    mc_result = _mcnemar_test(table, exact=True)
                    mcnemar_stat = float(mc_result.statistic)
                    mcnemar_p = float(mc_result.pvalue)

                mcnemar_or = _median_unbiased_or(n_AB, n_BA)

        seed_variability = np.nan
        vote_lists = sdf["llm_votes"].tolist()
        disagreement_rates = []
        for votes in vote_lists:
            n_votes = len(votes)
            if n_votes >= 2:
                n_pairs = n_votes * (n_votes - 1) // 2
                n_disagree = sum(a != b for a, b in combinations(votes, 2))
                disagreement_rates.append(n_disagree / n_pairs)
        if disagreement_rates:
            seed_variability = float(np.mean(disagreement_rates))

        ambig_df = sdf[sdf["is_ambiguous"]]
        tie_rate_ambig = np.nan
        if len(ambig_df) > 0:
            tie_rate_ambig = float(
                (ambig_df["llm_majority"] == "TIE").sum() / len(ambig_df)
            )

        acc_clear_ci_lo = acc_clear_ci_hi = np.nan
        ipi_ci_lo = ipi_ci_hi = np.nan
        ac1_ci_lo = ac1_ci_hi = np.nan

        _se = ac1.get("gwet_ac1_se", np.nan)
        _v = ac1.get("gwet_ac1", np.nan)
        if not np.isnan(_se) and not np.isnan(_v):
            ac1_ci_lo = float(np.clip(_v - 1.96 * _se, -1, 1))
            ac1_ci_hi = float(np.clip(_v + 1.96 * _se, -1, 1))

        if len(clear_df) >= 3:
            _h = clear_df["human_majority"].values
            _l = clear_df["llm_majority"].values
            acc_clear_ci_lo, acc_clear_ci_hi = _bca_ci_scipy(
                lambda hs, ls: float(np.mean(hs == ls)),
                _h,
                _l,
                n_resamples=n_resamples,
            )

        if ipi_applicable and len(swap_df) >= 3:
            _stable = (
                swap_df["llm_majority"] == swap_df["llm_majority_swapped"]
            ).values.astype(float)
            ipi_ci_lo, ipi_ci_hi = _bca_ci_scipy(
                lambda s, _: float(1.0 - np.mean(s)),
                _stable,
                _stable,
                n_resamples=n_resamples,
            )

        results.append(
            {
                **group_dict,
                "paradigm": "pairwise",
                "consensus_bin": stratum_name,
                "n_items": n_total,
                "n_clear": n_clear,
                "n_ambiguous": n_ambig,
                "accuracy_clear": acc_clear,
                "accuracy_clear_ci_lo": acc_clear_ci_lo,
                "accuracy_clear_ci_hi": acc_clear_ci_hi,
                "accuracy_all": acc_all,
                **ac1,
                "gwet_ac1_ci_lo": ac1_ci_lo,
                "gwet_ac1_ci_hi": ac1_ci_hi,
                "ipi_applicable": ipi_applicable,
                "ipi": ipi,
                "ipi_ci_lo": ipi_ci_lo,
                "ipi_ci_hi": ipi_ci_hi,
                "acc_order1": acc_order1,
                "acc_order2": acc_order2,
                "seed_variability": seed_variability,
                "tie_rate_ambiguous": tie_rate_ambig,
                "p_vote_A_orig": p_vote_A_orig,
                "p_vote_A_swap": p_vote_A_swap,
                "p_tie_orig": p_tie_orig,
                "p_tie_swap": p_tie_swap,
                "flip_AtoB": flip_AtoB,
                "flip_BtoA": flip_BtoA,
                "mcnemar_stat": mcnemar_stat,
                "mcnemar_p": mcnemar_p,
                "mcnemar_or": mcnemar_or,
                "binom_p_identity": binom_p_identity,
                "binom_p_position": binom_p_position,
                "cohens_g_identity": cohens_g_identity,
                "cohens_g_position": cohens_g_position,
            }
        )
    return results


def compute_pairwise_metrics(
    df: pd.DataFrame,
    config: MetricsConfig,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = ["dataset", "model", "config", "criterion"]
    groups = list(df.groupby(group_cols, dropna=False))

    nested = Parallel(n_jobs=-1)(
        delayed(_process_pairwise_group)(gv, gdf, config) for gv, gdf in groups
    )
    results = [row for group_results in nested for row in group_results]
    return pd.DataFrame(results)
