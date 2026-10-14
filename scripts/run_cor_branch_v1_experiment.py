#!/usr/bin/env python3
"""Run the minimal explicit two-branch CoR Phase 1 experiment.

The evaluation IDs are loaded from the completed baseline and are never
resampled.  The agent receives only the question and inference-safe topic
entities; gold structure metadata and gold answers stay outside inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT  # noqa: E402

if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from run_cor_baseline_experiment import (  # noqa: E402
	evaluate,
	integer_usage,
	load_safe_project_env,
	normalize_topic_entities,
	prediction_names,
	read_jsonl,
	summarize,
)
from chain_of_relations.kg_backend import get_backend  # noqa: E402
from chain_of_relations.methods.branch_v1 import BranchV1Agent  # noqa: E402
from chain_of_relations.methods.cor.agent import CoRAgent  # noqa: E402


BASELINE_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "cor_phase1_baseline_20261014_fixed"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "branch_v1"
SUBSET_ROOT = PROJECT_ROOT / "outputs" / "subsets"
SUBSETS = ("cwq_branch_2", "grailqa_branch_2")


def load_frozen_rows(baseline_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
	selected_ids = json.loads((baseline_root / "sample_ids.json").read_text(encoding="utf-8"))
	selected_rows: dict[str, list[dict[str, Any]]] = {}
	for subset in SUBSETS:
		if subset not in selected_ids:
			raise SystemExit(f"baseline sample IDs missing subset: {subset}")
		path = SUBSET_ROOT / f"{subset}.jsonl"
		if not path.exists():
			raise SystemExit(f"missing inference-safe subset: {path}")
		rows_by_id = {str(row.get("id", "")): row for row in read_jsonl(path)}
		rows: list[dict[str, Any]] = []
		for example_id in selected_ids[subset]:
			if str(example_id) not in rows_by_id:
				raise SystemExit(f"frozen baseline ID not found in {path}: {example_id}")
			rows.append(rows_by_id[str(example_id)])
		selected_rows[subset] = rows
	return selected_rows, {subset: [str(value) for value in selected_ids[subset]] for subset in SUBSETS}


def run_one(agent: BranchV1Agent, row: dict[str, Any], detail_dir: Path) -> dict[str, Any]:
	question = str(row.get("question", ""))
	example_id = str(row.get("id", ""))
	gold_answers = [str(value) for value in row.get("answers", []) if str(value).strip()]
	topic_entities = normalize_topic_entities(row)
	prompt_history: list[dict[str, Any]] = []
	sparql_history: list[dict[str, Any]] = []
	step_history: list[dict[str, Any]] = []
	error: str | None = None
	started = time.perf_counter()

	try:
		if not topic_entities:
			raise ValueError("inference-safe subset has no valid topic entities")
		result = agent.answer(
			question=question,
			topic_entities=topic_entities,
			prompt_history=prompt_history,
			sparql_history=sparql_history,
			step_history=step_history,
		)
	except Exception as exc:  # preserve one row and continue the experiment
		logging.exception("Branch-v1 failed for %s", example_id)
		result = {"action": "ERROR", "results": [], "reasoning_chains": ""}
		error = f"{type(exc).__name__}: {exc}"

	latency_ms = (time.perf_counter() - started) * 1000.0
	prediction = prediction_names(result)
	hit, f1 = evaluate(prediction, gold_answers)
	if str(result.get("action", "")).upper() == "ERROR" and error is None:
		error = str(result.get("reasoning_chains", "LLM/agent error"))
	input_tokens = integer_usage(prompt_history, "input_tokens")
	output_tokens = integer_usage(prompt_history, "output_tokens")
	row_result: dict[str, Any] = {
		"id": example_id,
		"question": question,
		"gold_answers": gold_answers,
		"prediction": prediction,
		"hits_at_1": hit,
		"f1": f1,
		"relation_chain": result.get("reasoning_chains", []),
		"llm_calls": len(prompt_history),
		"kg_queries": len(sparql_history),
		"token_usage": {
			"input_tokens": input_tokens,
			"output_tokens": output_tokens,
			"total_tokens": input_tokens + output_tokens,
		},
		"latency_ms": latency_ms,
		"error": error,
		"branches_created": result.get("branches_created", 0),
		"branch_decomposition": result.get("branch_decomposition", {}),
		"branch_traces": result.get("branch_traces", []),
		"merge_operator": result.get("merge_operator", "AND"),
		"branch_candidate_counts": result.get("branch_candidate_counts", []),
		"merged_candidate_count": result.get("merged_candidate_count", len(prediction)),
		"branch_failures": result.get("branch_failures", []),
	}

	detail = {
		**row_result,
		"action": result.get("action", ""),
		"reasoning_chains": result.get("reasoning_chains", ""),
		"prompt_history": prompt_history,
		"sparql_history": sparql_history,
		"step_history": step_history,
	}
	detail_dir.mkdir(parents=True, exist_ok=True)
	(detail_dir / f"{example_id}.json").write_text(
		json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
	)
	return row_result


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
	parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
	parser.add_argument("--depth", type=int, default=3)
	parser.add_argument("--relation-width", type=int, default=3)
	parser.add_argument("--sample-relation-threshold", type=int, default=1000)
	parser.add_argument("--temperature-exploration", type=float, default=0.3)
	parser.add_argument("--temperature-reasoning", type=float, default=0.1)
	parser.add_argument("--max-token", type=int, default=1024)
	parser.add_argument("--remove-unnecessary-rel", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	load_safe_project_env()
	logging.basicConfig(
		level=getattr(logging, args.log_level),
		format="%(asctime)s | %(levelname)s | %(message)s",
	)
	if not os.getenv("MODEL_NAME"):
		raise SystemExit("MODEL_NAME is required")
	if not os.getenv("OPENAI_API_KEY"):
		raise SystemExit("OPENAI_API_KEY is required")
	if not args.baseline_root.exists():
		raise SystemExit(f"frozen baseline root does not exist: {args.baseline_root}")

	selected_rows, selected_ids = load_frozen_rows(args.baseline_root)
	args.output_root.mkdir(parents=True, exist_ok=True)
	config = {
		"experiment": "phase1_branch_v1",
		"baseline_commit": "af7b513",
		"method": "cor_plus_explicit_two_branch_and",
		"model": os.getenv("MODEL_NAME", ""),
		"openai_base_url": os.getenv("OPENAI_BASE_URL", ""),
		"freebase_sparql_endpoint": os.getenv("FREEBASE_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql"),
		"sparql_timeout": os.getenv("SPARQL_TIMEOUT", "30"),
		"temperature_exploration": args.temperature_exploration,
		"temperature_reasoning": args.temperature_reasoning,
		"max_tokens": args.max_token,
		"max_reasoning_depth": args.depth,
		"relation_width": args.relation_width,
		"sample_relation_threshold": args.sample_relation_threshold,
		"remove_unnecessary_rel": args.remove_unnecessary_rel,
		"retry_settings": {
			"max_retries": int(os.getenv("OPENAI_MAX_RETRIES", "3")),
			"retry_interval": float(os.getenv("OPENAI_RETRY_INTERVAL", "1")),
			"timeout": float(os.getenv("OPENAI_TIMEOUT", "30")),
		},
		"frozen_sample_ids": selected_ids,
		"runtime_structure_inputs": ["question", "topic_entities", "topic_entity_names"],
		"gold_structure_used_at_runtime": False,
		"branch_count": 2,
		"merge_operator": "AND",
	}
	(args.output_root / "branch_v1_config.json").write_text(
		json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
	)
	(args.output_root / "sample_ids.json").write_text(
		json.dumps(selected_ids, ensure_ascii=False, indent=2), encoding="utf-8"
	)

	cor_agent = CoRAgent(
		model_name=os.environ["MODEL_NAME"],
		relation_width=args.relation_width,
		depth=args.depth,
		sample_relation_threshold=args.sample_relation_threshold,
		remove_unnecessary_rel=args.remove_unnecessary_rel,
		temperature_exploration=args.temperature_exploration,
		temperature_reasoning=args.temperature_reasoning,
		max_token=args.max_token,
		backend=get_backend("freebase"),
	)
	agent = BranchV1Agent(cor_agent)

	all_results: dict[str, list[dict[str, Any]]] = {}
	summaries: dict[str, dict[str, Any]] = {}
	for subset, rows in selected_rows.items():
		results_path = args.output_root / f"{subset}_results.jsonl"
		detail_dir = args.output_root / "details" / subset
		existing: dict[str, dict[str, Any]] = {}
		if args.resume and results_path.exists():
			for line in results_path.read_text(encoding="utf-8").splitlines():
				if line.strip():
					saved = json.loads(line)
					existing[str(saved.get("id", ""))] = saved
		elif results_path.exists():
			results_path.unlink()
		results: list[dict[str, Any]] = []
		logging.info("Running %s: %d frozen examples", subset, len(rows))
		for index, row in enumerate(rows, start=1):
			example_id = str(row.get("id", ""))
			if example_id in existing:
				result = existing[example_id]
				logging.info("[%s] %d/%d already complete id=%s", subset, index, len(rows), example_id)
			else:
				logging.info("[%s] %d/%d id=%s", subset, index, len(rows), example_id)
				result = run_one(agent, row, detail_dir)
				with results_path.open("a", encoding="utf-8") as handle:
					handle.write(json.dumps(result, ensure_ascii=False) + "\n")
			results.append(result)
		all_results[subset] = results
		summaries[subset] = summarize(results)

	(args.output_root / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
	print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()

