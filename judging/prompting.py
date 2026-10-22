import json
import random
from typing import Optional

from .models import PROMPTS_DIR


def get_output_schema(prompt_type: str) -> Optional[dict]:
    schemas = {
        "epikrise": {
            "type": "object",
            "properties": {
                "human_written": {"type": "string", "enum": ["A", "B", "TIE"]},
                "completeness": {"type": "string", "enum": ["A", "B", "TIE"]},
                "efficiency": {"type": "string", "enum": ["A", "B", "TIE"]},
                "organized": {"type": "string", "enum": ["A", "B", "TIE"]},
                "clinical_utility": {"type": "string", "enum": ["A", "B", "TIE"]},
            },
            "required": [
                "human_written",
                "completeness",
                "efficiency",
                "organized",
                "clinical_utility",
            ],
            "additionalProperties": False,
        },
        "reports": {
            "type": "object",
            "properties": {
                "accurate": {"type": "integer", "minimum": 1, "maximum": 5},
                "thorough": {"type": "integer", "minimum": 1, "maximum": 5},
                "useful": {"type": "integer", "minimum": 1, "maximum": 5},
                "organized": {"type": "integer", "minimum": 1, "maximum": 5},
                "comprehensible": {"type": "integer", "minimum": 1, "maximum": 5},
                "succinct": {"type": "integer", "minimum": 1, "maximum": 5},
                "clinical_utility": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": [
                "accurate",
                "thorough",
                "useful",
                "organized",
                "comprehensible",
                "succinct",
                "clinical_utility",
            ],
            "additionalProperties": False,
        },
        "nursing": {
            "type": "object",
            "properties": {
                "citation": {"type": "integer", "minimum": 1, "maximum": 5},
                "accurate": {"type": "integer", "minimum": 1, "maximum": 5},
                "thorough": {"type": "integer", "minimum": 1, "maximum": 5},
                "useful": {"type": "integer", "minimum": 1, "maximum": 5},
                "organized": {"type": "integer", "minimum": 1, "maximum": 5},
                "comprehensible": {"type": "integer", "minimum": 1, "maximum": 5},
                "succinct": {"type": "integer", "minimum": 1, "maximum": 5},
                "abstraction": {"type": "integer", "minimum": 1, "maximum": 5},
                "stig_original": {"type": "integer", "minimum": 0, "maximum": 1},
                "stig_summary": {"type": "integer", "minimum": 0, "maximum": 1},
            },
            "required": [
                "citation",
                "accurate",
                "thorough",
                "useful",
                "organized",
                "comprehensible",
                "succinct",
                "abstraction",
                "stig_original",
                "stig_summary",
            ],
            "additionalProperties": False,
        },
    }
    return schemas.get(prompt_type)


def build_mtb_errors_schema(item: dict) -> dict:
    human_labels = item.get("human_labels", [])
    properties = {}
    required = []

    for key, section_data in human_labels[0].items():
        required.append(key)
        section_text = section_data["text"]
        properties[key] = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "enum": [section_text]},
                "comment": {"type": "string"},
                "errors": {"type": "string", "enum": ["PASS", "FAIL"]},
            },
            "required": ["text", "comment", "errors"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def load_prompt_template(prompt_type: str, config: str) -> str:
    if config in ["baseline", "demo", "multiagent"]:
        suffix = "baseline"
    elif config == "rule":
        suffix = "rules"
    else:
        suffix = "baseline"

    prompt_file = f"{prompt_type}_{suffix}.txt"
    prompt_path = PROMPTS_DIR / prompt_file

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8")


def format_reports_prompt(template: str, item: dict) -> str:
    return template.format(
        notes=item.get("notes", ""),
        summary_to_evaluate=item.get("summary", ""),
        specialty=item.get("specialty", "general medicine"),
    )


def format_nursing_prompt(template: str, item: dict) -> str:
    return template.format(
        notes=item.get("notes", ""),
        summary_to_evaluate=item.get("summary", ""),
    )


def format_epikrise_prompt(template: str, item: dict) -> str:
    return template.format(
        source_text=item.get("source_text", ""),
        text_a=item.get("text_a", ""),
        text_b=item.get("text_b", ""),
    )


def format_mtb_errors_prompt(template: str, item: dict) -> str:
    return template.format(
        source=item.get("source", ""),
        text=item.get("text", ""),
    )


def get_formatter(prompt_type: str):
    formatters = {
        "reports": format_reports_prompt,
        "nursing": format_nursing_prompt,
        "epikrise": format_epikrise_prompt,
        "mtb_errors": format_mtb_errors_prompt,
    }
    return formatters.get(prompt_type, format_reports_prompt)


def _get_first_or_empty(lst: list) -> dict:
    return lst[0] if lst else {}


def create_few_shot_examples(
    items: list[dict], current_item_id: str, dataset_name: str, num_examples: int = 2
) -> str:
    available_items = [
        item
        for item in items
        if item["id"] != current_item_id
        and item.get("human_labels")
        or item.get("human_votes")
    ]

    if len(available_items) < num_examples:
        num_examples = len(available_items)

    if num_examples == 0:
        return ""

    selected = random.sample(available_items, num_examples)

    examples = []
    for i, item in enumerate(selected, 1):
        if dataset_name == "reports":
            human_label = _get_first_or_empty(item.get("human_labels", []))
            example = f"""
<EXAMPLE_{i}>
CLINICAL_NOTES:
{item.get("notes", "")}

CLINICAL_SUMMARY:
{item.get("summary", "")}

SPECIALTY: {item.get("specialty", "general medicine")}

EXPECTED_OUTPUT:
{json.dumps(human_label, indent=2)}
</EXAMPLE_{i}>"""

        elif dataset_name == "nursing":
            human_label = _get_first_or_empty(item.get("human_labels", []))
            example = f"""
<EXAMPLE_{i}>
CLINICAL_NOTES:
{item.get("notes", "")}

CLINICAL_SUMMARY:
{item.get("summary", "")}

EXPECTED_OUTPUT:
{json.dumps(human_label, indent=2)}
</EXAMPLE_{i}>"""

        elif dataset_name == "epikrise":
            human_vote = _get_first_or_empty(item.get("human_votes", []))
            example = f"""
<EXAMPLE_{i}>
SOURCE_TEXT:
{item.get("source_text", "")}

CANDIDATE_A:
{item.get("text_a", "")}

CANDIDATE_B:
{item.get("text_b", "")}

EXPECTED_OUTPUT:
{json.dumps(human_vote, indent=2)}
</EXAMPLE_{i}>"""

        elif dataset_name == "mtb_errors":
            human_label = _get_first_or_empty(item.get("human_labels", []))
            example = f"""
<EXAMPLE_{i}>
MTB_PROTOKOLL:
{item.get("source", "")}

PATIENTENPROTOKOLL:
{item.get("text", "")}

EXPECTED_OUTPUT:
{json.dumps(human_label, indent=2)}
</EXAMPLE_{i}>"""

        else:
            example = ""

        examples.append(example)

    return "\n\n<EXAMPLES>\n" + "\n".join(examples) + "\n</EXAMPLES>\n"
