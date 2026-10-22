from __future__ import annotations

import json
from dataclasses import dataclass, field

MAJOR_SEVERITIES: frozenset[str] = frozenset({"ERHEB", "KRIT"})


@dataclass(frozen=True)
class GroupKey:
    dataset: str
    task: str
    paradigm: str
    model: str
    config: str

    def to_dict(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "paradigm": self.paradigm,
            "model": self.model,
            "config": self.config,
        }


@dataclass
class MetricsConfig:
    majority_thresh: float = 0.75
    consensus_percentage_bins: list[tuple[str, float, float]] = field(
        default_factory=lambda: [
            ("low", 0.0, 0.6),
            ("medium", 0.6, 1.0),
            ("high", 1.0, float("inf")),
        ]
    )
    run_beyond_correlation: bool = False
    binary_criteria: frozenset[str] = frozenset({"stig_original", "stig_summary"})
    n_resamples: int = 1000

    def to_json(self) -> str:
        d = {
            "majority_thresh": self.majority_thresh,
            "consensus_percentage_bins": [
                (name, lo, hi if hi != float("inf") else "inf")
                for name, lo, hi in self.consensus_percentage_bins
            ],
            "run_beyond_correlation": self.run_beyond_correlation,
            "n_resamples": self.n_resamples,
        }
        return json.dumps(d, indent=2)
