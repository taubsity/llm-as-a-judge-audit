from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from BeyondCorrelation.beyond_correlation import (  # type: ignore
    beyond_correlation_for_categorical_data,
    beyond_correlation_for_ordinal_data,
)

from .config import MetricsConfig


def run_beyond_correlation_ordinal(
    df: pd.DataFrame,
    outdir: Path,
    config: MetricsConfig,
) -> pd.DataFrame | None:
    df = df.copy()
    df.loc[df["dataset"] == "reports", "task"] = "all"
    df.loc[
        (df["dataset"] == "nursing") & (df["task"].isin(["summary", "endpoints"])),
        "task",
    ] = "all"

    bc_dir = outdir / "beyond_correlation" / "ordinal"
    bc_dir.mkdir(parents=True, exist_ok=True)
    group_cols = ["dataset", "task", "model", "config", "criterion"]
    all_results: list[pd.DataFrame] = []
    constant_input_log: list[dict] = []

    for group_vals, group_df in df.groupby(group_cols, dropna=False):
        group_dict = dict(zip(group_cols, group_vals))
        group_id = "__".join(str(v) for v in group_vals if v)

        if group_dict.get("config") == "multiagent":
            print(f"BC ordinal skipped (multiagent): {group_id}")
            continue

        criterion = group_dict.get("criterion", "")
        if criterion in config.binary_criteria:
            print(f"BC ordinal skipped (binary criterion): {group_id}")
            continue

        human_labels_list = group_df["human_labels"].tolist()
        machine_labels_list = group_df["machine_labels"].tolist()

        if not human_labels_list or not machine_labels_list:
            continue

        valid_pairs = [
            (h, m) for h, m in zip(human_labels_list, machine_labels_list) if m
        ]

        if not valid_pairs:
            continue
        human_labels_list, machine_labels_list = zip(*valid_pairs)
        human_labels_list = list(human_labels_list)
        machine_labels_list = list(machine_labels_list)

        all_vals = [v for lst in human_labels_list + machine_labels_list for v in lst]
        if not all_vals:
            continue

        ordinal_min = int(min(all_vals))
        ordinal_max = int(max(all_vals))

        human_labels_int = [[int(v) for v in lst] for lst in human_labels_list]
        machine_labels_int = [[int(v) for v in lst] for lst in machine_labels_list]

        result_subdir = bc_dir / group_id
        result_subdir.mkdir(parents=True, exist_ok=True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res_df = beyond_correlation_for_ordinal_data(
                human_labels=human_labels_int,
                machine_labels=machine_labels_int,
                ordinal_min=ordinal_min,
                ordinal_max=ordinal_max,
                result_dir=str(result_subdir),
                experiment_name=group_dict["criterion"],
                median_percentage_bins=[
                    (lo, hi) for _, lo, hi in config.consensus_percentage_bins
                ],
            )
        n_const = sum(1 for w in caught if "ConstantInputWarning" in str(w.category))
        if n_const:
            constant_input_log.append({**group_dict, "count": n_const})
        for k, v in group_dict.items():
            res_df[k] = v
        all_results.append(res_df)

    if constant_input_log:
        pd.DataFrame(constant_input_log).to_csv(
            bc_dir / "constant_input_warnings.csv", index=False
        )

    if all_results:
        merged = pd.concat(all_results, ignore_index=True)
        merged.to_csv(bc_dir / "beyond_correlation_ordinal_all.csv", index=False)
        return merged
    return None


def run_beyond_correlation_binary(
    df: pd.DataFrame,
    outdir: Path,
    config: MetricsConfig,
) -> pd.DataFrame | None:
    df = df.copy()
    df.loc[df["dataset"] == "reports", "task"] = "all"
    df.loc[
        (df["dataset"] == "nursing") & (df["task"].isin(["summary", "endpoints"])),
        "task",
    ] = "all"

    bin_df = df[df["criterion"].isin(config.binary_criteria)]
    if bin_df.empty:
        return None

    bc_dir = outdir / "beyond_correlation" / "binary"
    bc_dir.mkdir(parents=True, exist_ok=True)
    group_cols = ["dataset", "task", "model", "config", "criterion"]
    unique_labels = [0, 1]
    all_results: list[pd.DataFrame] = []
    constant_input_log: list[dict] = []

    for group_vals, group_df in bin_df.groupby(group_cols, dropna=False):
        group_dict = dict(zip(group_cols, group_vals))
        group_id = "__".join(str(v) for v in group_vals if v)

        if group_dict.get("config") == "multiagent":
            print(f"BC binary skipped (multiagent): {group_id}")
            continue

        human_labels_list = group_df["human_labels"].tolist()
        machine_labels_list = group_df["machine_labels"].tolist()

        valid_pairs = [
            (h, m) for h, m in zip(human_labels_list, machine_labels_list) if m
        ]

        if not valid_pairs:
            print(f"BC binary skipped (no valid machine labels): {group_id}")
            continue

        human_labels_list, machine_labels_list = zip(*valid_pairs)

        human_int = [[int(v) for v in lst] for lst in human_labels_list]
        machine_int = [[int(v) for v in lst] for lst in machine_labels_list]

        result_subdir = bc_dir / group_id
        result_subdir.mkdir(parents=True, exist_ok=True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res_df = beyond_correlation_for_categorical_data(
                human_labels=human_int,
                machine_labels=machine_int,
                unique_label_list=unique_labels,
                result_dir=str(result_subdir),
                experiment_name=group_dict["criterion"],
                majority_percentage_bins=list(config.consensus_percentage_bins),
            )
        n_const = sum(1 for w in caught if "ConstantInputWarning" in str(w.category))
        if n_const:
            constant_input_log.append({**group_dict, "count": n_const})
        for k, v in group_dict.items():
            res_df[k] = v
        all_results.append(res_df)
    if constant_input_log:
        pd.DataFrame(constant_input_log).to_csv(
            bc_dir / "constant_input_warnings.csv", index=False
        )

    if all_results:
        merged = pd.concat(all_results, ignore_index=True)
        merged.to_csv(bc_dir / "beyond_correlation_binary_all.csv", index=False)
        return merged
    return None


def run_beyond_correlation_categorical(
    df: pd.DataFrame,
    outdir: Path,
    config: MetricsConfig,
) -> pd.DataFrame | None:
    df = df.copy()
    df.loc[df["dataset"] == "reports", "task"] = "all"
    df.loc[
        (df["dataset"] == "nursing") & (df["task"].isin(["summary", "endpoints"])),
        "task",
    ] = "all"

    if df.empty:
        return None

    bc_dir = outdir / "beyond_correlation" / "categorical"
    bc_dir.mkdir(parents=True, exist_ok=True)
    group_cols = ["dataset", "task", "model", "config", "criterion"]
    vote_map = {"A": 0, "B": 1, "TIE": 2}
    unique_labels = [0, 1, 2]
    all_results: list[pd.DataFrame] = []
    constant_input_log: list[dict] = []

    for group_vals, group_df in df.groupby(group_cols, dropna=False):
        group_dict = dict(zip(group_cols, group_vals))
        group_id = "__".join(str(v) for v in group_vals if v)

        if group_dict.get("config") == "multiagent":
            print(f"BC categorical skipped (multiagent): {group_id}")
            continue

        valid_pairs = [
            (h, m)
            for h, m in zip(
                group_df["human_votes"].tolist(), group_df["llm_votes"].tolist()
            )
            if m
        ]

        if not valid_pairs:
            print(f"BC categorical skipped (no valid machine votes): {group_id}")
            continue

        human_votes, machine_votes = zip(*valid_pairs)
        human_int = [[vote_map.get(v, 2) for v in lst] for lst in human_votes]
        machine_int = [[vote_map.get(v, 2) for v in lst] for lst in machine_votes]

        if not human_int or not machine_int:
            continue

        result_subdir = bc_dir / group_id
        result_subdir.mkdir(parents=True, exist_ok=True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res_df = beyond_correlation_for_categorical_data(
                human_labels=human_int,
                machine_labels=machine_int,
                unique_label_list=unique_labels,
                result_dir=str(result_subdir),
                experiment_name=group_dict["criterion"],
                majority_percentage_bins=list(config.consensus_percentage_bins),
            )
        n_const = sum(1 for w in caught if "ConstantInputWarning" in str(w.category))
        if n_const:
            constant_input_log.append({**group_dict, "count": n_const})
        for k, v in group_dict.items():
            res_df[k] = v
        all_results.append(res_df)

    if constant_input_log:
        pd.DataFrame(constant_input_log).to_csv(
            bc_dir / "constant_input_warnings.csv", index=False
        )

    if all_results:
        merged = pd.concat(all_results, ignore_index=True)
        merged.to_csv(bc_dir / "beyond_correlation_categorical_all.csv", index=False)
        return merged
    return None
