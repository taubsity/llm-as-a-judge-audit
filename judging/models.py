from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class SamplingConfig:
    max_tokens: int = 20000
    seeds: list[int] = field(default_factory=lambda: [13, 37, 101])


SAMPLING = SamplingConfig()


@dataclass
class ModelConfig:
    name: str
    short_name: str
    endpoint: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"


def _endpoint(short_name: str, default: str) -> str:
    env_key = f"AUDIT_ENDPOINT_{short_name.upper().replace('-', '_')}"
    return os.environ.get(env_key, default)


MODELS: list[ModelConfig] = [
    ModelConfig(
        name="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        short_name="Q-235B",
        endpoint=_endpoint("Q-235B", "http://localhost:8001/v1"),
    ),
    ModelConfig(
        name="Qwen3-Next-80B-A3B-Instruct",
        short_name="Q-80B",
        endpoint=_endpoint("Q-80B", "http://localhost:8002/v1"),
    ),
    ModelConfig(
        name="google/medgemma-27b-it",
        short_name="MG-27B",
        endpoint=_endpoint("MG-27B", "http://localhost:8003/v1"),
    ),
    ModelConfig(
        name="openai/gpt-oss-120b",
        short_name="G-120B",
        endpoint=_endpoint("G-120B", "http://localhost:8004/v1"),
    ),
    ModelConfig(
        name="mistralai/Magistral-Small-2509",
        short_name="M-24B",
        endpoint=_endpoint("M-24B", "http://localhost:8005/v1"),
    ),
    ModelConfig(
        name="Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
        short_name="Q-235B-Think",
        endpoint=_endpoint("Q-235B-Think", "http://localhost:8006/v1"),
    ),
]


MULTIAGENT_MODELS: frozenset[str] = frozenset({"G-120B", "M-24B", "Q-235B"})


CONFIGS: list[str] = ["baseline", "rule", "demo", "multiagent"]


DATASETS: dict[str, dict] = {
    "reports": {
        "paradigm": "pointwise",
        "prompt_type": "reports",
        "file": "reports.jsonl",
    },
    "epikrise": {
        "paradigm": "pairwise",
        "prompt_type": "epikrise",
        "file": "epikrise.jsonl",
    },
    "mtb_errors": {
        "paradigm": "passfail",
        "prompt_type": "mtb_errors",
        "file": "mtb_errors.jsonl",
    },
    "nursing": {
        "paradigm": "pointwise",
        "prompt_type": "nursing",
        "file": "nursing.jsonl",
    },
}


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("AUDIT_DATA_DIR", PROJECT_ROOT.parent / "data_audit"))
PROMPTS_DIR = Path(os.environ.get("AUDIT_PROMPTS_DIR", PROJECT_ROOT / "prompts"))
RESULTS_DIR = Path(os.environ.get("AUDIT_RESULTS_DIR", DATA_DIR))
