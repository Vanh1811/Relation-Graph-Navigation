#!/usr/bin/env python3
"""Freeze and run the controlled Adaptive v1.1.3 GrailQA-12 repeats."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

# Importing this module first installs the lightweight official HTTPS transport
# used by the existing experiment runner before CoR imports the OpenAI client.
from run_cor_baseline_experiment import (  # noqa: E402
	evaluate,
	integer_usage,
	load_safe_project_env,
	normalize_topic_entities,
	prediction_names,
	read_jsonl,
	relation_chain_from_trace,
)
from chain_of_relations.kg_backend import get_backend  # noqa: E402
from chain_of_relations.methods.adaptive_v1_1_3 import AdaptiveRoutingAgent  # noqa: E402
from chain_of_relations.methods.cor.agent import CoRAgent  # noqa: E402


SUBSET_ROOT = PROJECT_ROOT / "outputs" / "subsets"
SUBSET = "grailqa_branch_2"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_3"
REPEAT_ROOT = OUTPUT_ROOT / "grailqa_12_repeat"
SAMPLE_PATH = REPEAT_ROOT / "sample_ids.json"
SAMPLE_SIZE = 12
REPEAT_COUNT = 3
SAMPLE_SEED = 20261018


def _collect_ids(value: Any) -> set[str]:
	if isinstance(value, dict):
		result: set[str] = set()
		for item in value.values():
			result.update(_collect_ids(item))
		return result
	if isinstance(value, list):
		return {str(item) for item in value if isinstance(item, (str, int))}
	return set()


def prior_sample_ids() -> set[str]:
	"""Exclude every previously frozen sample plus the v1.1.2 inspected IDs."""
	ids: set[str] = set()
	for path in (PROJECT_ROOT / "outputs" / "experiments").rglob("sample_ids.json"):
		if path == SAMPLE_PATH:
			continue
		try:
			ids.update(_collect_ids(json.loads(path.read_text(encoding="utf-8"))))
		except (OSError, json.JSONDecodeError):
			continue
	ids.update({
		"3201985002000", "3202928006000", "4302129014000",
		"3205391005000", "4300043017000",
	})
	return ids


def freeze_sample() -> dict[str, list[str]]:
	rows = read_jsonl(SUBSET_ROOT / f"{SUBSET}.jsonl")
	excluded = prior_sample_ids()
	candidates = [row for row in rows if str(row.get("id", "")) not in excluded]
	if len(candidates) < SAMPLE_SIZE:
		raise SystemExit(f"only {len(candidates)} fresh GrailQA candidates remain after exclusions")
	rng = random.Random(SAMPLE_SEED)
	chosen = sorted(rng.sample(candidates, SAMPLE_SIZE), key=lambda row: str(row.get("id", "")))
	ids = [str(row["id"]) for row in chosen]
	REPEAT_ROOT.mkdir(parents=True, exist_ok=True)
	SAMPLE_PATH.write_text(json.dumps({SUBSET: ids}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	return {SUBSET: ids}


def load_sample() -> list[str]:
	if not SAMPLE_PATH.exists():
		raise SystemExit(f"sample is not frozen: {SAMPLE_PATH}")
	data = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
	ids = [str(item) for item in data.get(SUBSET, [])]
	if len(ids) != SAMPLE_SIZE or len(set(ids)) != SAMPLE_SIZE:
		raise SystemExit(f"sample must contain exactly {SAMPLE_SIZE} unique IDs")
	overlap = set(ids) & prior_sample_ids()
	if overlap:
		raise SystemExit(f"frozen sample overlaps a prior/inspected sample: {sorted(overlap)}")
	return ids


def load_rows(ids: list[str]) -> list[dict[str, Any]]:
	by_id = {str(row.get("id", "")): row for row in read_jsonl(SUBSET_ROOT / f"{SUBSET}.jsonl")}
	missing = [item for item in ids if item not in by_id]
	if missing:
		raise SystemExit(f"missing frozen IDs from {SUBSET}: {missing}")
	return [by_id[item] for item in ids]


def make_agent(method: str, args: argparse.Namespace) -> Any:
	kwargs = dict(
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
	cor = CoRAgent(**kwargs)
	if method == "cor":
		return cor
	return AdaptiveRoutingAgent(cor, confidence_threshold=args.confidence_threshold, max_branches=3)


def run_one(agent: Any, method: str, repeat: int, row: dict[str, Any], detail_dir: Path) -> dict[str, Any]:
	example_id = str(row.get("id", ""))
	question = str(row.get("question", ""))
	gold_answers = [str(value) for value in row.get("answers", []) if str(value).strip()]
	topic_entities = normalize_topic_entities(row)
	prompt_history: list[dict[str, Any]] = []
	sparql_history: list[dict[str, Any]] = []
	step_history: list[dict[str, Any]] = []
	started = time.perf_counter()
	error: str | None = None
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
	except Exception as exc:
		logging.exception("v1.1.3 failed for %s repeat=%s method=%s", example_id, repeat, method)
		result = {"action": "ERROR", "results": [], "reasoning_chains": ""}
		error = f"{type(exc).__name__}: {exc}"

	latency_ms = (time.perf_counter() - started) * 1000.0
	prediction = prediction_names(result)
	hit, f1 = evaluate(prediction, gold_answers)
	native_result = result.get("adaptive_native_result") if isinstance(result.get("adaptive_native_result"), dict) else result
	native_prediction = prediction_names(native_result)
	native_hit, native_f1 = evaluate(native_prediction, gold_answers)
	branch_source = native_result if method == "adaptive" else result
	if not isinstance(branch_source, dict):
		branch_source = {}
	row_result: dict[str, Any] = {
		"id": example_id,
		"repeat": repeat,
		"method": method,
		"question": question,
		"gold_answers": gold_answers,
		"prediction": prediction,
		"hits_at_1": hit,
		"f1": f1,
		"native_prediction": native_prediction,
		"native_hits_at_1": native_hit,
		"native_f1": native_f1,
		"relation_chain": relation_chain_from_trace(result),
		"llm_calls": len(prompt_history),
		"kg_queries": len(sparql_history),
		"token_usage": {
			"input_tokens": integer_usage(prompt_history, "input_tokens"),
			"output_tokens": integer_usage(prompt_history, "output_tokens"),
		},
		"latency_ms": latency_ms,
		"error": error,
		"router_structure": result.get("router_structure", "BASELINE_UNCHANGED" if method == "cor" else "UNCERTAIN"),
		"router_execution_mode": result.get("router_execution_mode", "cor"),
		"router_confidence": result.get("router_confidence", 0.0),
		"router_reason": result.get("router_reason", ""),
		"branch_relation_mode": result.get("branch_relation_mode", ""),
		"same_final_answer_space": result.get("same_final_answer_space", False),
		"validation_reason": result.get("validation_reason", ""),
		"requested_final_answer_type": result.get("requested_final_answer_type", ""),
		"requested_final_semantic_property": result.get("requested_final_semantic_property", ""),
		"shared_answer_type": result.get("shared_answer_type", ""),
		"proposed_branches": result.get("proposed_branches", []),
		"branch_property_compatibility": result.get("branch_property_compatibility"),
		"branch_constraint_properties": result.get("branch_constraint_properties", []),
		"validation_passed": result.get("validation_passed", True if method == "cor" else False),
		"validation_failed": result.get("validation_failed", False),
		"validation_failures": result.get("validation_failures", []),
		"fallback_used": bool(result.get("fallback_used", False)),
		"fallback_reason": result.get("fallback_reason"),
		"final_execution_mode": result.get("final_execution_mode", "cor"),
		"native_execution_mode": result.get("native_execution_mode", "cor"),
		"branches_created": branch_source.get("branches_created", 0),
		"branch_decomposition": branch_source.get("branch_decomposition", {}),
		"branch_traces": branch_source.get("branch_traces", []),
		"branch_candidate_counts": branch_source.get("branch_candidate_counts", []),
		"merged_candidate_count": branch_source.get("merged_candidate_count", len(prediction)),
		"branch_failures": branch_source.get("branch_failures", []),
		"merge_method": branch_source.get("merge_method", "NONE"),
		"merge_outcome": branch_source.get("merge_outcome", ""),
		"merge_inputs": branch_source.get("merge_inputs", []),
	}
	detail = {
		**row_result,
		"action": result.get("action", ""),
		"reasoning_chains": result.get("reasoning_chains", ""),
		"prompt_history": prompt_history,
		"sparql_history": sparql_history,
		"step_history": step_history,
	}
	if method == "adaptive" and isinstance(result.get("adaptive_native_result"), dict):
		detail["adaptive_native_result"] = result["adaptive_native_result"]
	detail_dir.mkdir(parents=True, exist_ok=True)
	(detail_dir / f"{example_id}.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
	return row_result


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--freeze", action="store_true", help="freeze the fresh 12-question sample and stop")
	parser.add_argument("--run", action="store_true", help="run all configured repeats")
	parser.add_argument("--repeat-count", type=int, default=REPEAT_COUNT)
	parser.add_argument("--depth", type=int, default=3)
	parser.add_argument("--relation-width", type=int, default=3)
	parser.add_argument("--sample-relation-threshold", type=int, default=1000)
	parser.add_argument("--temperature-exploration", type=float, default=0.3)
	parser.add_argument("--temperature-reasoning", type=float, default=0.1)
	parser.add_argument("--max-token", type=int, default=1024)
	parser.add_argument("--confidence-threshold", type=float, default=0.75)
	parser.add_argument("--remove-unnecessary-rel", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	load_safe_project_env()
	logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
	if args.freeze:
		print(json.dumps(freeze_sample(), ensure_ascii=False, indent=2))
		if not args.run:
			return
	if not args.run:
		raise SystemExit("choose --freeze or --run")
	if args.repeat_count <= 0:
		raise SystemExit("--repeat-count must be positive")
	if not os.getenv("MODEL_NAME") or not os.getenv("OPENAI_API_KEY"):
		raise SystemExit("MODEL_NAME and OPENAI_API_KEY are required")
	ids = load_sample()
	rows = load_rows(ids)
	REPEAT_ROOT.mkdir(parents=True, exist_ok=True)
	(REPEAT_ROOT / "config.json").write_text(json.dumps({
		"experiment": "adaptive_v1_1_3_answer_type_stability",
		"subset": SUBSET,
		"sample_size": SAMPLE_SIZE,
		"repeat_count": args.repeat_count,
		"sample_seed": SAMPLE_SEED,
		"model": os.getenv("MODEL_NAME", ""),
		"depth": args.depth,
		"relation_width": args.relation_width,
		"sample_relation_threshold": args.sample_relation_threshold,
		"temperature_exploration": args.temperature_exploration,
		"temperature_reasoning": args.temperature_reasoning,
		"max_token": args.max_token,
		"confidence_threshold": args.confidence_threshold,
		"gold_structure_used_at_runtime": False,
		"excluded_prior_samples": True,
	}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

	paths = {method: REPEAT_ROOT / f"{method}_runs.jsonl" for method in ("cor", "adaptive")}
	existing: dict[tuple[str, int, str], dict[str, Any]] = {}
	if args.resume:
		for method, path in paths.items():
			if path.exists():
				for item in read_jsonl(path):
					existing[(method, int(item.get("repeat", 0)), str(item.get("id", "")))] = item

	for repeat in range(1, args.repeat_count + 1):
		for method in ("cor", "adaptive"):
			agent = make_agent(method, args)
			for index, row in enumerate(rows, start=1):
				case_id = str(row.get("id", ""))
				key = (method, repeat, case_id)
				logging.info("[%s] repeat=%s %d/%d id=%s", method, repeat, index, len(rows), case_id)
				if key in existing:
					continue
				result = run_one(agent, method, repeat, row, REPEAT_ROOT / "details" / f"repeat_{repeat}" / method)
				with paths[method].open("a", encoding="utf-8") as handle:
					handle.write(json.dumps(result, ensure_ascii=False) + "\n")
	print(json.dumps({"rows_per_method": args.repeat_count * len(rows), "total_inference_runs": args.repeat_count * len(rows) * 2, "output": str(REPEAT_ROOT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
