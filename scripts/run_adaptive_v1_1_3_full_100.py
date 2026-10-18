#!/usr/bin/env python3
"""Run the frozen Adaptive v1.1.3 Phase 1 evaluation on 50+50 questions.

This driver deliberately reuses the already-frozen v1.1.3 experiment helpers
and the original Phase 1 sample.  It never reruns the CoR baseline and it
does not import the rejected v1.1.4 implementation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from _bootstrap import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the existing runner first.  It installs the lightweight raw OpenAI
# transport before the CoR modules are imported.
import run_adaptive_v1_1_3_experiment as frozen  # noqa: E402


BASELINE_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "cor_phase1_baseline_20261014_fixed"
SUBSET_ROOT = PROJECT_ROOT / "outputs" / "subsets"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_3" / "full_100"
SAMPLE_PATH = OUTPUT_ROOT / "sample_ids.json"
RESULTS_PATH = OUTPUT_ROOT / "adaptive_results.jsonl"
CONFIG_PATH = OUTPUT_ROOT / "config.json"
DETAIL_ROOT = OUTPUT_ROOT / "details" / "adaptive"

DATASETS = ("cwq_branch_2", "grailqa_branch_2")
FROZEN_SETTINGS = {
    "depth": 3,
    "relation_width": 3,
    "sample_relation_threshold": 1000,
    "remove_unnecessary_rel": True,
    "temperature_exploration": 0.3,
    "temperature_reasoning": 0.1,
    "max_token": 1024,
    "confidence_threshold": 0.75,
    "max_branches": 3,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_and_validate_sample() -> dict[str, list[str]]:
    """Use the original baseline sample without resampling or replacement."""
    source_path = BASELINE_ROOT / "sample_ids.json"
    source = _read_json(source_path)
    sample = {dataset: [str(value) for value in source.get(dataset, [])] for dataset in DATASETS}
    for dataset in DATASETS:
        if len(sample[dataset]) != 50 or len(set(sample[dataset])) != 50:
            raise SystemExit(f"{dataset} does not contain exactly 50 unique frozen IDs")

        subset_path = SUBSET_ROOT / f"{dataset}.jsonl"
        subset_ids = {str(row.get("id", "")) for row in _read_jsonl(subset_path)}
        missing = [item for item in sample[dataset] if item not in subset_ids]
        if missing:
            raise SystemExit(f"{dataset} frozen IDs missing from subset: {missing}")

        baseline_path = BASELINE_ROOT / f"{dataset}_results.jsonl"
        baseline_rows = _read_jsonl(baseline_path)
        baseline_ids = [str(row.get("id", "")) for row in baseline_rows]
        if len(baseline_rows) != 50 or set(baseline_ids) != set(sample[dataset]):
            raise SystemExit(f"{dataset} frozen CoR baseline does not match the 50 frozen IDs")

    if SAMPLE_PATH.exists():
        existing = _read_json(SAMPLE_PATH)
        existing_sample = {dataset: [str(value) for value in existing.get(dataset, [])] for dataset in DATASETS}
        if existing_sample != sample:
            raise SystemExit(f"existing full-100 sample differs from the original frozen sample: {SAMPLE_PATH}")
    else:
        _write_json(SAMPLE_PATH, sample)
    return sample


def load_rows(sample: dict[str, list[str]]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for dataset in DATASETS:
        by_id = {str(row.get("id", "")): row for row in _read_jsonl(SUBSET_ROOT / f"{dataset}.jsonl")}
        for question_id in sample[dataset]:
            rows.append((dataset, by_id[question_id]))
    return rows


def make_frozen_args() -> argparse.Namespace:
    return SimpleNamespace(**FROZEN_SETTINGS)


def _branch_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    traces = result.get("branch_traces")
    proposed = result.get("proposed_branches")
    if isinstance(traces, list) and traces:
        return traces
    if isinstance(proposed, list):
        return proposed
    return []


def enrich_result(dataset: str, source_row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    usage = result.get("token_usage") if isinstance(result.get("token_usage"), dict) else {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    final_prediction = result.get("prediction", [])
    final_hit = result.get("hits_at_1", 0)
    final_f1 = result.get("f1", 0.0)
    validation_failures = result.get("validation_failures")
    if not isinstance(validation_failures, list):
        validation_failures = []
    enriched = {
        **result,
        "dataset": dataset,
        "question_id": str(source_row.get("id", "")),
        "requested_answer_type": result.get("requested_final_answer_type", ""),
        "requested_property": result.get("requested_final_semantic_property", ""),
        "execution_mode": result.get("final_execution_mode", result.get("router_execution_mode", "cor")),
        "branch_count": int(result.get("branches_created", 0) or result.get("branch_count", 0) or 0),
        "branches": _branch_records(result),
        "final_prediction": final_prediction,
        "final_hits_at_1": final_hit,
        "final_f1": final_f1,
        "errors": [result["error"]] if result.get("error") else [],
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "total": int(usage.get("total_tokens", input_tokens + output_tokens) or input_tokens + output_tokens),
        },
        "validation_result": {
            "passed": bool(result.get("validation_passed", False)),
            "failures": validation_failures,
        },
        "merge_result": result.get("merge_outcome", ""),
    }
    return enriched


def update_detail(result: dict[str, Any]) -> None:
    dataset = str(result["dataset"])
    question_id = str(result["question_id"])
    path = DETAIL_ROOT / dataset / f"{question_id}.json"
    if not path.exists():
        return
    detail = _read_json(path)
    detail.update(result)
    _write_json(path, detail)


def existing_results() -> dict[str, dict[str, Any]]:
    if not RESULTS_PATH.exists():
        return {}
    rows = _read_jsonl(RESULTS_PATH)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row.get('dataset')}::{row.get('question_id', row.get('id', ''))}"
        if key in by_id:
            raise SystemExit(f"duplicate full-100 result: {key}")
        by_id[key] = row
    return by_id


def write_config(sample: dict[str, list[str]]) -> None:
    config = {
        "experiment": "adaptive_v1_1_3_full_100_phase1",
        "method": "adaptive_v1.1.3",
        "sample_ids": sample,
        "sample_counts": {dataset: len(sample[dataset]) for dataset in DATASETS},
        "model": os.getenv("MODEL_NAME", ""),
        "freebase_sparql_endpoint": os.getenv("FREEBASE_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql"),
        "settings": FROZEN_SETTINGS,
        "gold_structure_used_at_runtime": False,
        "gold_answers_used_for_routing": False,
        "historical_cor_rerun": False,
        "v1_1_4_used": False,
        "source_sample": str(BASELINE_ROOT / "sample_ids.json"),
        "source_cor_baseline": str(BASELINE_ROOT),
    }
    if CONFIG_PATH.exists():
        old = _read_json(CONFIG_PATH)
        for key in ("sample_ids", "settings", "historical_cor_rerun", "v1_1_4_used"):
            if old.get(key) != config[key]:
                raise SystemExit(f"existing config differs in frozen field {key}: {CONFIG_PATH}")
    else:
        _write_json(CONFIG_PATH, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run the 100-question frozen evaluation")
    parser.add_argument("--resume", action="store_true", help="resume only missing rows in an existing output")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen.load_safe_project_env()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    if not args.run:
        raise SystemExit("choose --run")
    if not os.getenv("MODEL_NAME") or not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("MODEL_NAME and OPENAI_API_KEY are required")

    sample = load_and_validate_sample()
    write_config(sample)
    existing = existing_results()
    if existing and not args.resume:
        raise SystemExit("full-100 results already exist; use --resume to continue")

    rows = load_rows(sample)
    agent = frozen.make_agent("adaptive", make_frozen_args())
    completed = len(existing)
    with RESULTS_PATH.open("a", encoding="utf-8") as output:
        for index, (dataset, source_row) in enumerate(rows, start=1):
            question_id = str(source_row.get("id", ""))
            key = f"{dataset}::{question_id}"
            if key in existing:
                logging.info("[adaptive] %d/%d already complete %s/%s", index, len(rows), dataset, question_id)
                continue
            logging.info("[adaptive] %d/%d running %s/%s", index, len(rows), dataset, question_id)
            raw = frozen.run_one(agent, "adaptive", 1, source_row, DETAIL_ROOT / dataset)
            result = enrich_result(dataset, source_row, raw)
            update_detail(result)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            existing[key] = result
            completed += 1
            logging.info(
                "[adaptive] completed=%d/%d final_h1=%s final_f1=%.4f mode=%s",
                completed,
                len(rows),
                result.get("final_hits_at_1", 0),
                float(result.get("final_f1", 0.0) or 0.0),
                result.get("execution_mode", ""),
            )

    print(json.dumps({"completed": completed, "expected": len(rows), "output": str(RESULTS_PATH)}, indent=2))


if __name__ == "__main__":
    main()
