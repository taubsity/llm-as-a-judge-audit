import argparse
import asyncio
import time
from collections import Counter
from dataclasses import dataclass
from itertools import groupby as _groupby
from typing import Optional

from preprocessing.common import load_jsonl, write_jsonl

from .client import LLMClient, parse_json_response
from .models import (
    CONFIGS,
    DATA_DIR,
    DATASETS,
    MODELS,
    MULTIAGENT_MODELS,
    RESULTS_DIR,
    SAMPLING,
    ModelConfig,
)
from .prompting import (
    build_mtb_errors_schema,
    create_few_shot_examples,
    get_formatter,
    get_output_schema,
    load_prompt_template,
)


def load_dataset(dataset_name: str) -> list[dict]:
    dataset_config = DATASETS.get(dataset_name)
    if not dataset_config:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    filepath = DATA_DIR / dataset_config["file"]
    return load_jsonl(filepath)


@dataclass
class WorkUnit:
    config: str
    dataset_name: str
    item: dict
    seed: int
    prompt: str
    schema: Optional[dict]
    order: str = "original"


def item_identity(item: dict) -> tuple:
    return (str(item.get("id")), str(item.get("task", "")))


@dataclass
class CallResult:
    work: WorkUnit
    response: Optional[str] = None
    metadata: Optional[dict] = None
    error: Optional[str] = None


def _flip_sides(parsed: dict) -> dict:
    return {k: {"A": "B", "B": "A"}.get(v, v) for k, v in parsed.items()}


def _majority_vote_labels(dicts: list[dict]) -> dict:
    all_keys: set[str] = set()
    for d in dicts:
        all_keys.update(d.keys())
    result = {}
    for key in all_keys:
        values = [d[key] for d in dicts if key in d]
        if values:
            result[key] = Counter(values).most_common(1)[0][0]
    return result


def _majority_vote_passfail(dicts: list[dict]) -> dict:
    all_keys: set[str] = set()
    for d in dicts:
        all_keys.update(d.keys())
    result = {}
    for key in all_keys:
        values = [d[key] for d in dicts if key in d]
        if not values:
            continue
        if isinstance(values[0], dict):
            error_vals = [v["errors"] for v in values]
            hashable = [tuple(e) if isinstance(e, list) else e for e in error_vals]
            best = Counter(hashable).most_common(1)[0][0]
            result[key] = {
                "text": values[0]["text"],
                "comment": values[0]["comment"],
                "errors": list(best) if isinstance(best, tuple) else best,
            }
        else:
            result[key] = Counter(values).most_common(1)[0][0]
    return result


def aggregate_multiagent_results(results: list[dict]) -> list[dict]:
    by_item: dict[tuple, list[dict]] = {}
    for result in results:
        if result.get("config") != "baseline":
            continue
        if result.get("model") not in MULTIAGENT_MODELS:
            continue
        item_key = (
            result.get("dataset"),
            result.get("paradigm"),
            *item_identity(result),
        )
        by_item.setdefault(item_key, []).append(result)

    aggregated = []
    skipped_single: list[str] = []

    for (dataset, paradigm, item_id, item_task), item_results in by_item.items():
        if len(item_results) < 2:
            skipped_single.append(
                f"{dataset}/{item_id} (only model: {item_results[0]['model']})"
                if item_results
                else f"{dataset}/{item_id}"
            )
            continue

        models_used = sorted({r["model"] for r in item_results})

        if paradigm == "pointwise":
            seed_labels: dict[int, dict[str, dict]] = {}
            for r in item_results:
                model = r["model"]
                for resp in r.get("responses", []):
                    parsed = resp.get("parsed")
                    if parsed is not None:
                        seed_labels.setdefault(resp["seed"], {})[model] = parsed

            llm_labels = [
                _majority_vote_labels(list(model_dicts.values()))
                for _, model_dicts in sorted(seed_labels.items())
                if model_dicts
            ]
            if not llm_labels:
                continue

            aggregated.append(
                {
                    "id": item_id,
                    "dataset": dataset,
                    "task": item_task,
                    "config": "multiagent",
                    "paradigm": paradigm,
                    "models_aggregated": models_used,
                    "llm_labels": llm_labels,
                    "human_labels": item_results[0].get("human_labels", []),
                    "individual_results": item_results,
                }
            )

        elif paradigm == "pairwise":
            seed_orig: dict[int, dict[str, dict]] = {}
            seed_swap: dict[int, dict[str, dict]] = {}

            for r in item_results:
                model = r["model"]
                for resp in r.get("responses", []):
                    parsed = resp.get("parsed")
                    if parsed is None:
                        continue
                    seed = resp["seed"]
                    if resp.get("order") == "swapped":
                        seed_swap.setdefault(seed, {})[model] = _flip_sides(parsed)
                    else:
                        seed_orig.setdefault(seed, {})[model] = parsed

            llm_votes = [
                _majority_vote_labels(list(model_dicts.values()))
                for _, model_dicts in sorted(seed_orig.items())
                if model_dicts
            ]
            llm_votes_swapped = [
                _majority_vote_labels(list(model_dicts.values()))
                for _, model_dicts in sorted(seed_swap.items())
                if model_dicts
            ]
            if not llm_votes:
                continue

            aggregated.append(
                {
                    "id": item_id,
                    "dataset": dataset,
                    "task": item_task,
                    "config": "multiagent",
                    "paradigm": paradigm,
                    "models_aggregated": models_used,
                    "llm_votes": llm_votes,
                    "llm_votes_swapped": llm_votes_swapped,
                    "human_votes": item_results[0].get("human_votes", []),
                    "individual_results": item_results,
                }
            )

        elif paradigm == "passfail":
            seed_labels_pf: dict[int, dict[str, dict]] = {}
            for r in item_results:
                model = r["model"]
                for resp in r.get("responses", []):
                    parsed = resp.get("parsed")
                    if parsed is not None:
                        seed_labels_pf.setdefault(resp["seed"], {})[model] = parsed

            llm_labels_pf = [
                _majority_vote_passfail(list(model_dicts.values()))
                for _, model_dicts in sorted(seed_labels_pf.items())
                if model_dicts
            ]
            if not llm_labels_pf:
                continue

            aggregated.append(
                {
                    "id": item_id,
                    "dataset": dataset,
                    "task": item_task,
                    "config": "multiagent",
                    "paradigm": paradigm,
                    "models_aggregated": models_used,
                    "llm_labels": llm_labels_pf,
                    "human_labels": item_results[0].get("human_labels", []),
                    "individual_results": item_results,
                }
            )

    if skipped_single:
        print(
            f"  WARNING: {len(skipped_single)} item(s) skipped for multiagent "
            f"(fewer than 2 models produced baseline results):"
        )
        for s in skipped_single[:10]:
            print(f"    - {s}")
        if len(skipped_single) > 10:
            print(f"    ... and {len(skipped_single) - 10} more")

    return aggregated


def _build_work_units(
    model: ModelConfig,
    active_configs: list[str],
    loaded_datasets: dict[str, list[dict]],
    existing_keys: set,
) -> list[WorkUnit]:
    units = []
    for config in active_configs:
        if config == "multiagent":
            continue
        for dataset_name, all_items in loaded_datasets.items():
            dataset_config = DATASETS[dataset_name]
            paradigm = dataset_config["paradigm"]
            prompt_type = dataset_config["prompt_type"]
            template = load_prompt_template(prompt_type, config)
            formatter = get_formatter(prompt_type)
            static_schema = get_output_schema(prompt_type)
            is_mtb_errors = prompt_type == "mtb_errors"

            for item in all_items:
                key = (
                    *item_identity(item),
                    dataset_name,
                    model.short_name,
                    config,
                )
                if key in existing_keys:
                    continue

                prompt = formatter(template, item)
                if config == "demo":
                    examples = create_few_shot_examples(
                        all_items, item["id"], dataset_name, num_examples=2
                    )
                    prompt = prompt + examples

                if is_mtb_errors:
                    schema = build_mtb_errors_schema(item)
                else:
                    schema = static_schema

                for seed in SAMPLING.seeds:
                    units.append(
                        WorkUnit(config, dataset_name, item, seed, prompt, schema)
                    )

                if paradigm == "pairwise":
                    item_swapped = {
                        **item,
                        "text_a": item.get("text_b", ""),
                        "text_b": item.get("text_a", ""),
                    }
                    prompt_swapped = formatter(template, item_swapped)
                    if config == "demo":
                        prompt_swapped = prompt_swapped + examples
                    for seed in SAMPLING.seeds:
                        units.append(
                            WorkUnit(
                                config,
                                dataset_name,
                                item,
                                seed,
                                prompt_swapped,
                                schema,
                                order="swapped",
                            )
                        )
    return units


async def _execute_call(
    sem: asyncio.Semaphore, client: LLMClient, work: WorkUnit, tag: str
) -> CallResult:
    async with sem:
        try:
            response, metadata = await client.generate(
                work.prompt, work.seed, schema=work.schema
            )
            return CallResult(work=work, response=response, metadata=metadata)
        except Exception as e:
            print(
                f"  {tag} ERROR [{work.config}] {work.dataset_name} "
                f"item={work.item.get('id')} seed={work.seed} order={work.order}: {e}",
                flush=True,
            )
            return CallResult(work=work, error=str(e))


def _assemble_results(
    call_results: list[CallResult], model: ModelConfig
) -> tuple[list[dict], list[str]]:
    def item_key(cr: CallResult):
        return (cr.work.config, cr.work.dataset_name, *item_identity(cr.work.item))

    sorted_results = sorted(call_results, key=item_key)
    results = []
    errors = []

    for key, group_iter in _groupby(sorted_results, key=item_key):
        config, dataset_name, item_id, _item_task = key
        calls = list(group_iter)
        item = calls[0].work.item
        paradigm = DATASETS[dataset_name]["paradigm"]

        failed = [c for c in calls if c.error is not None]
        if failed:
            for c in failed:
                errors.append(
                    f"[{model.short_name}] [{config}] {dataset_name} item={item_id} "
                    f"seed={c.work.seed} order={c.work.order}: {c.error}"
                )
            continue

        original_calls = [c for c in calls if c.work.order == "original"]
        raw_responses = []

        llm_labels = []
        for c in original_calls:
            parsed = parse_json_response(c.response)
            raw_responses.append(
                {
                    "seed": c.work.seed,
                    **({"order": "original"} if paradigm == "pairwise" else {}),
                    "raw": c.response,
                    "parsed": parsed,
                    "metadata": c.metadata,
                }
            )
            if parsed is not None:
                llm_labels.append(parsed)

        if not llm_labels:
            errors.append(
                f"[{model.short_name}] [{config}] {dataset_name} item={item_id}: "
                f"all seeds returned unparseable responses"
            )
            continue

        if paradigm == "pairwise":
            swapped_calls = [c for c in calls if c.work.order == "swapped"]
            swapped_votes = []
            for c in swapped_calls:
                parsed = parse_json_response(c.response)
                raw_responses.append(
                    {
                        "seed": c.work.seed,
                        "order": "swapped",
                        "raw": c.response,
                        "parsed": parsed,
                        "metadata": c.metadata,
                    }
                )
                if parsed is not None:
                    swapped_votes.append(_flip_sides(parsed))

            results.append(
                {
                    "id": item["id"],
                    "dataset": dataset_name,
                    "model": model.short_name,
                    "config": config,
                    "paradigm": "pairwise",
                    "llm_votes": llm_labels,
                    "llm_votes_swapped": swapped_votes,
                    "human_votes": item.get("human_votes", []),
                    "responses": raw_responses,
                }
            )
        else:
            label_key = "llm_labels"
            results.append(
                {
                    "id": item["id"],
                    "dataset": dataset_name,
                    "task": item.get("task", ""),
                    "model": model.short_name,
                    "config": config,
                    "paradigm": paradigm,
                    label_key: llm_labels,
                    "human_labels": item.get("human_labels", []),
                    "responses": raw_responses,
                }
            )

    return results, errors


def run_all_judges(
    models: Optional[list[str]] = None,
    configs: Optional[list[str]] = None,
    datasets: Optional[list[str]] = None,
) -> None:
    asyncio.run(run_all_judges_async(models, configs, datasets))


async def run_all_judges_async(
    models: Optional[list[str]] = None,
    configs: Optional[list[str]] = None,
    datasets: Optional[list[str]] = None,
) -> None:
    print("=" * 70)
    print("LLM-as-a-Judge Audit: Running Judges (Parallel Mode)")
    print("=" * 70)

    active_models = MODELS
    if models:
        active_models = [m for m in MODELS if m.short_name in models]

    active_configs = configs or CONFIGS
    active_datasets = datasets or list(DATASETS.keys())

    print(f"\nModels: {[m.short_name for m in active_models]}")
    print(f"Configs: {active_configs}")
    print(f"Datasets: {active_datasets}")
    print(f"Seeds: {SAMPLING.seeds}")
    print(f"Concurrency: 2 calls per model, {len(active_models)} models in parallel")
    print()

    loaded_datasets = {}
    for dataset_name in active_datasets:
        try:
            items = load_dataset(dataset_name)
            loaded_datasets[dataset_name] = items
            print(f"Loaded {dataset_name}: {len(items)} items")
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "judge_outputs.jsonl"

    existing_records = []
    existing_keys = set()
    error_retry_count = 0
    if output_path.exists():
        existing_records = load_jsonl(output_path)
        for res in existing_records:
            if res.get("config") == "multiagent":
                continue
            key = (
                *item_identity(res),
                res.get("dataset"),
                res.get("model"),
                res.get("config"),
            )
            has_error = any(
                r.get("metadata", {}).get("error")
                or str(r.get("raw", "")).startswith("ERROR:")
                for r in res.get("responses", [])
            )
            if has_error:
                error_retry_count += 1
            else:
                existing_keys.add(key)
        print(
            f"Resuming: {len(existing_records)} existing records, "
            f"{len(existing_keys)} completed, {error_retry_count} to retry due to errors"
        )

    start_time = time.time()

    tasks = [
        run_single_model_async(model, active_configs, loaded_datasets, existing_keys)
        for model in active_models
    ]
    model_results = await asyncio.gather(*tasks, return_exceptions=True)

    all_results = [
        r
        for r in existing_records
        if r.get("config") != "multiagent"
        and (*item_identity(r), r.get("dataset"), r.get("model"), r.get("config"))
        in existing_keys
    ]
    all_errors = []

    for i, result in enumerate(model_results):
        if isinstance(result, Exception):
            print(f"\nError in model {active_models[i].short_name}: {result}")
            continue
        model_all, model_errors = result
        all_results.extend(model_all)
        all_errors.extend(model_errors)

    total_time = time.time() - start_time

    if all_errors:
        print(f"\n{'=' * 50}")
        print(f"ERRORS: {len(all_errors)} item(s) failed and will retry on next run:")
        print("=" * 50)
        for err in all_errors:
            print(f"  {err}")

    baseline_ma = [
        r
        for r in all_results
        if r.get("config") == "baseline" and r.get("model") in MULTIAGENT_MODELS
    ]
    if baseline_ma:
        print(f"\n{'=' * 50}")
        print("Aggregating multiagent results...")
        print("=" * 50)
        multiagent_results = aggregate_multiagent_results(all_results)
        all_results.extend(multiagent_results)
        print(f"  Created {len(multiagent_results)} aggregated results")
    else:
        print(
            "\nNo baseline results for multiagent models found; skipping multiagent aggregation."
        )

    write_jsonl(all_results, output_path)

    print(f"\n{'=' * 70}")
    print(f"Complete. Saved {len(all_results)} results to {output_path}")
    print(f"Total execution time: {total_time:.1f}s")
    total_requests = sum(
        len(r.get("responses", [])) for r in all_results if "responses" in r
    )
    if total_requests > 0:
        print(
            f"Total requests: {total_requests} ({total_time / total_requests:.2f}s avg)"
        )
    print("=" * 70)


async def run_single_model_async(
    model: ModelConfig,
    active_configs: list[str],
    loaded_datasets: dict,
    existing_keys: set,
) -> tuple[list[dict], list[str]]:
    tag = f"[{model.short_name}]"
    print(f"{tag} starting ({model.name})", flush=True)

    try:
        client = LLMClient(model)
    except Exception as e:
        print(f"{tag} ERROR initializing client: {e}", flush=True)
        raise

    work_units = _build_work_units(
        model, active_configs, loaded_datasets, existing_keys
    )
    print(f"{tag} {len(work_units)} API calls queued (2 concurrent)", flush=True)

    if not work_units:
        print(f"{tag} nothing to do — all items already completed", flush=True)
        return [], []

    item_work: dict[tuple, list[WorkUnit]] = {}
    for wu in work_units:
        k = (wu.config, wu.dataset_name, *item_identity(wu.item))
        item_work.setdefault(k, []).append(wu)

    sem = asyncio.Semaphore(2)
    all_results = []
    all_errors = []
    output_path = RESULTS_DIR / "judge_outputs.jsonl"

    async def _process_item(item_key: tuple, units: list[WorkUnit]):
        call_results = await asyncio.gather(
            *[_execute_call(sem, client, wu, tag) for wu in units]
        )
        results, errors = _assemble_results(list(call_results), model)
        if results:
            write_jsonl(results, output_path, append=True)
        return results, errors

    tasks = [_process_item(k, units) for k, units in item_work.items()]
    item_outcomes = await asyncio.gather(*tasks)

    for results, errors in item_outcomes:
        all_results.extend(results)
        all_errors.extend(errors)

    print(
        f"{tag} done: {len(all_results)} items completed, {len(all_errors)} errors",
        flush=True,
    )

    return all_results, all_errors


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM judges for LLM-as-a-Judge audit"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[m.short_name for m in MODELS],
        help="Models to run (default: all)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=CONFIGS,
        help="Configurations to run (default: all)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        help="Datasets to run (default: all)",
    )

    args = parser.parse_args()

    run_all_judges(
        models=args.models,
        configs=args.configs,
        datasets=args.datasets,
    )


if __name__ == "__main__":
    main()
