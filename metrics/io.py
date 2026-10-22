from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .normalization import filter_placeholder_raters, normalize_llm_keys


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec = normalize_llm_keys(rec)
            rec = filter_placeholder_raters(rec)
            records.append(rec)
    return records


def _lists_to_json(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].apply(lambda x: isinstance(x, list)).any():
            out[col] = out[col].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )
    return out


def write_outputs(
    df_pointwise: pd.DataFrame,
    df_pairwise: pd.DataFrame,
    df_passfail: pd.DataFrame,
    summary_pointwise: pd.DataFrame,
    summary_pairwise: pd.DataFrame,
    summary_passfail: pd.DataFrame,
    outdir: Path,
    human_baseline: pd.DataFrame | None = None,
    stratified_jsd: pd.DataFrame | None = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    for name, df in [
        ("item_level_pointwise", df_pointwise),
        ("item_level_pairwise", df_pairwise),
        ("item_level_passfail", df_passfail),
    ]:
        if not df.empty:
            path = outdir / f"{name}.csv"
            _lists_to_json(df).to_csv(path, index=False)

    for name, df in [
        ("summary_pointwise", summary_pointwise),
        ("summary_pairwise", summary_pairwise),
        ("summary_passfail", summary_passfail),
    ]:
        if not df.empty:
            path = outdir / f"{name}.csv"
            df.to_csv(path, index=False)

    if human_baseline is not None and not human_baseline.empty:
        path = outdir / "human_baseline.csv"
        human_baseline.to_csv(path, index=False)

    if stratified_jsd is not None and not stratified_jsd.empty:
        path = outdir / "consensus_stratified_jsd.csv"
        stratified_jsd.to_csv(path, index=False)
