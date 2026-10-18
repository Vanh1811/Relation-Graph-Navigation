#!/usr/bin/env python3
"""Run Adaptive v1.1.2 diagnostic and fresh GrailQA-25 comparisons."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
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
	relation_chain_from_trace,
	summarize,
)
from chain_of_relations.kg_backend import get_backend  # noqa: E402
from chain_of_relations.methods.adaptive_v1_1_2 import AdaptiveRoutingAgent  # noqa: E402
from chain_of_relations.methods.cor.agent import CoRAgent  # noqa: E402


BASELINE_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "cor_phase1_baseline_20261014_fixed"
SUBSET_ROOT = PROJECT_ROOT / "outputs" / "subsets"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_2"
V11_DIAGNOSTIC_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_1" / "diagnostic_36"
SUBSET = "grailqa_branch_2"
FRESH_SAMPLE_PATH = OUTPUT_ROOT / "grailqa_25" / "sample_ids.json"
FRESH_SEED = 20261017


def git_value(args: list[str], default: str = "unknown") -> str:
	try:
		return subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip() or default
	except (OSError, subprocess.CalledProcessError):
		return default


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
	return {str(row.get("id", "")): row for row in read_jsonl(path)}


def selected_rows(ids_by_subset: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
	rows: dict[str, list[dict[str, Any]]] = {}
	for subset, ids in ids_by_subset.items():
		by_id = load_rows(SUBSET_ROOT / f"{subset}.jsonl")
		missing = [item for item in ids if item not in by_id]
		if missing:
			raise SystemExit(f"missing IDs in {subset}: {missing}")
		rows[subset] = [by_id[item] for item in ids]
	return rows


def load_diagnostic_ids() -> dict[str, list[str]]:
	path = V11_DIAGNOSTIC_ROOT / "sample_ids.json"
	data = json.loads(path.read_text(encoding="utf-8"))
	return {str(key): [str(item) for item in value] for key, value in data.items()}


def freeze_fresh_grailqa() -> dict[str, list[str]]:
	all_rows = read_jsonl(SUBSET_ROOT / f"{SUBSET}.jsonl")
	baseline_ids = set(json.loads((BASELINE_ROOT / "sample_ids.json").read_text(encoding="utf-8"))[SUBSET])
	diagnostic_ids = set(load_diagnostic_ids()[SUBSET])
	excluded = {str(item) for item in baseline_ids | diagnostic_ids}
	candidates = [row for row in all_rows if str(row.get("id", "")) not in excluded]
	if len(candidates) < 25:
		raise SystemExit(f"only {len(candidates)} fresh GrailQA candidates remain")
	rng = random.Random(FRESH_SEED)
	chosen = sorted(rng.sample(candidates, 25), key=lambda row: str(row.get("id", "")))
	ids = [str(row["id"]) for row in chosen]
	FRESH_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
	FRESH_SAMPLE_PATH.write_text(json.dumps({SUBSET: ids}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	return {SUBSET: ids}


def load_fresh_ids() -> dict[str, list[str]]:
	if not FRESH_SAMPLE_PATH.exists():
		raise SystemExit(f"fresh sample is not frozen: {FRESH_SAMPLE_PATH}")
	data = json.loads(FRESH_SAMPLE_PATH.read_text(encoding="utf-8"))
	ids = {SUBSET: [str(item) for item in data.get(SUBSET, [])]}
	if len(ids[SUBSET]) != 25 or len(set(ids[SUBSET])) != 25:
		raise SystemExit("fresh GrailQA sample must contain exactly 25 unique IDs")
	return ids


def _native_metrics(result: dict[str, Any], fallback_histories: tuple[list[dict[str, Any]], list[dict[str, Any]]]) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
	native_result = result.get("adaptive_native_result")
	if not isinstance(native_result, dict):
		native_result = result
	native_prompt = result.get("adaptive_native_prompt_history")
	native_sparql = result.get("adaptive_native_sparql_history")
	if not isinstance(native_prompt, list):
		native_prompt = fallback_histories[0]
	if not isinstance(native_sparql, list):
		native_sparql = fallback_histories[1]
	return prediction_names(native_result), native_prompt, native_sparql


def run_one(agent: Any, method: str, row: dict[str, Any], detail_dir: Path) -> dict[str, Any]:
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
	except Exception as exc:  # preserve a row and continue the fixed sample
		logging.exception("v1.1.2 failed for %s", example_id)
		result = {"action": "ERROR", "results": [], "reasoning_chains": ""}
		error = f"{type(exc).__name__}: {exc}"

	latency_ms = (time.perf_counter() - started) * 1000.0
	prediction = prediction_names(result)
	hit, f1 = evaluate(prediction, gold_answers)
	native_prediction, native_prompt, native_sparql = _native_metrics(result, (prompt_history, sparql_history))
	native_hit, native_f1 = evaluate(native_prediction, gold_answers)
	native_result = result.get("adaptive_native_result") if isinstance(result.get("adaptive_native_result"), dict) else result
	if str(result.get("action", "")).upper() == "ERROR" and error is None:
		error = str(result.get("reasoning_chains", "LLM/agent error"))
	native_error = None if str(native_result.get("action", "")).upper() != "ERROR" else str(native_result.get("reasoning_chains", "native execution error"))
	input_tokens = integer_usage(prompt_history, "input_tokens")
	output_tokens = integer_usage(prompt_history, "output_tokens")
	native_input_tokens = integer_usage(native_prompt, "input_tokens")
	native_output_tokens = integer_usage(native_prompt, "output_tokens")
	branch_source = native_result if method == "adaptive" else result
	if not isinstance(branch_source, dict):
		branch_source = {}
	row_result: dict[str, Any] = {
		"id": example_id,
		"question": question,
		"gold_answers": gold_answers,
		"prediction": prediction,
		"hits_at_1": hit,
		"f1": f1,
		"native_prediction": native_prediction,
		"native_hits_at_1": native_hit,
		"native_f1": native_f1,
		"native_error": native_error,
		"relation_chain": relation_chain_from_trace(result),
		"llm_calls": len(prompt_history),
		"kg_queries": len(sparql_history),
		"native_llm_calls": len(native_prompt),
		"native_kg_queries": len(native_sparql),
		"token_usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens},
		"native_token_usage": {"input_tokens": native_input_tokens, "output_tokens": native_output_tokens, "total_tokens": native_input_tokens + native_output_tokens},
		"latency_ms": latency_ms,
		"error": error,
		"method": method,
		"router_structure": result.get("router_structure", "BASELINE_UNCHANGED" if method == "cor" else "UNCERTAIN"),
		"router_execution_mode": result.get("router_execution_mode", "cor"),
		"router_mode": result.get("router_mode", "cor"),
		"router_confidence": result.get("router_confidence", 0.0),
		"router_reason": result.get("router_reason", ""),
		"branch_count": result.get("branch_count", 0),
		"shared_answer_type": result.get("shared_answer_type", ""),
		"proposed_branches": result.get("proposed_branches", []),
		"unsupported_structure": result.get("unsupported_structure", False),
		"validation_passed": result.get("validation_passed", True if method == "cor" else False),
		"validation_failed": result.get("validation_failed", False),
		"validation_failures": result.get("validation_failures", []),
		"fallback_used": bool(result.get("fallback_used", False)),
		"fallback_reason": result.get("fallback_reason"),
		"final_execution_mode": result.get("final_execution_mode", "cor" if method == "cor" else "cor"),
		"native_execution_mode": result.get("native_execution_mode", "cor"),
		"native_result_action": result.get("native_result_action", native_result.get("action", "")),
		"branches_created": branch_source.get("branches_created", 0),
		"branch_decomposition": branch_source.get("branch_decomposition", {}),
		"branch_traces": branch_source.get("branch_traces", []),
		"merge_operator": branch_source.get("merge_operator", "AND"),
		"branch_candidate_counts": branch_source.get("branch_candidate_counts", []),
		"merged_candidate_count": branch_source.get("merged_candidate_count", len(prediction)),
		"branch_failures": branch_source.get("branch_failures", []),
		"branch_statuses": [trace.get("status", "") for trace in branch_source.get("branch_traces", [])],
		"merge_method": branch_source.get("merge_method", "NONE"),
		"merge_methods": branch_source.get("merge_methods", []),
		"merge_method_counts": branch_source.get("merge_method_counts", {}),
		"merge_outcome": branch_source.get("merge_outcome", ""),
		"merge_inputs": branch_source.get("merge_inputs", []),
	}
	detail = {**row_result, "action": result.get("action", ""), "reasoning_chains": result.get("reasoning_chains", ""), "prompt_history": prompt_history, "sparql_history": sparql_history, "step_history": step_history}
	if method == "adaptive" and isinstance(result.get("adaptive_native_result"), dict):
		detail["adaptive_native_result"] = result["adaptive_native_result"]
	detail_dir.mkdir(parents=True, exist_ok=True)
	(detail_dir / f"{example_id}.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
	return row_result


def run_method(method: str, rows: list[dict[str, Any]], output_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
	if method == "cor":
		agent: Any = CoRAgent(
			model_name=os.environ["MODEL_NAME"], relation_width=args.relation_width, depth=args.depth,
			sample_relation_threshold=args.sample_relation_threshold, remove_unnecessary_rel=args.remove_unnecessary_rel,
			temperature_exploration=args.temperature_exploration, temperature_reasoning=args.temperature_reasoning,
			max_token=args.max_token, backend=get_backend("freebase"),
		)
	else:
		cor = CoRAgent(
			model_name=os.environ["MODEL_NAME"], relation_width=args.relation_width, depth=args.depth,
			sample_relation_threshold=args.sample_relation_threshold, remove_unnecessary_rel=args.remove_unnecessary_rel,
			temperature_exploration=args.temperature_exploration, temperature_reasoning=args.temperature_reasoning,
			max_token=args.max_token, backend=get_backend("freebase"),
		)
		agent = AdaptiveRoutingAgent(cor, confidence_threshold=args.confidence_threshold, max_branches=3)
	results_path = output_root / f"{method}_results.jsonl"
	detail_dir = output_root / "details" / method
	existing = {str(item.get("id")): item for item in read_jsonl(results_path)} if args.resume and results_path.exists() else {}
	if not args.resume and results_path.exists():
		results_path.unlink()
	results: list[dict[str, Any]] = []
	for index, row in enumerate(rows, start=1):
		example_id = str(row.get("id", ""))
		logging.info("[%s] %d/%d id=%s", method, index, len(rows), example_id)
		if example_id in existing:
			result = existing[example_id]
		else:
			result = run_one(agent, method, row, detail_dir)
			with results_path.open("a", encoding="utf-8") as handle:
				handle.write(json.dumps(result, ensure_ascii=False) + "\n")
		results.append(result)
	return results


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--run", choices=["diagnostic_36", "grailqa_25", "freeze_grailqa_25"], required=True)
	parser.add_argument("--method", choices=["adaptive", "cor", "both"], default="adaptive")
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
	if args.run == "freeze_grailqa_25":
		print(json.dumps(freeze_fresh_grailqa(), ensure_ascii=False, indent=2))
		return
	if not os.getenv("MODEL_NAME") or not os.getenv("OPENAI_API_KEY"):
		raise SystemExit("MODEL_NAME and OPENAI_API_KEY are required")
	if args.run == "diagnostic_36":
		ids_by_subset = load_diagnostic_ids()
		rows_by_subset = selected_rows(ids_by_subset)
		rows = rows_by_subset["cwq_branch_2"] + rows_by_subset["grailqa_branch_2"]
		root = OUTPUT_ROOT / "diagnostic_36"
		root.mkdir(parents=True, exist_ok=True)
		(root / "sample_ids.json").write_text(json.dumps(ids_by_subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
		(root / "config.json").write_text(json.dumps({"experiment": "adaptive_v1_1_2", "run": args.run, "model": os.getenv("MODEL_NAME", ""), "router_prompt_version": "adaptive-routing-v1.1.2-rule-aware-json-v1", "gold_structure_used_at_runtime": False}, indent=2) + "\n", encoding="utf-8")
		if args.method != "adaptive":
			raise SystemExit("diagnostic_36 is adaptive-only; historical CoR and Branch-v1 outputs are frozen")
		results = run_method("adaptive", rows, root, args)
		(root / "results.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n", encoding="utf-8")
		(root / "summary.json").write_text(json.dumps(summarize(results), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
		print(json.dumps(summarize(results), ensure_ascii=False, indent=2))
		return

	ids_by_subset = load_fresh_ids()
	rows = selected_rows(ids_by_subset)[SUBSET]
	root = OUTPUT_ROOT / "grailqa_25"
	root.mkdir(parents=True, exist_ok=True)
	(root / "sample_ids.json").write_text(json.dumps(ids_by_subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	(root / "config.json").write_text(json.dumps({"experiment": "adaptive_v1_1_2", "run": args.run, "seed": FRESH_SEED, "model": os.getenv("MODEL_NAME", ""), "gold_structure_used_at_runtime": False}, indent=2) + "\n", encoding="utf-8")
	methods = ["cor", "adaptive"] if args.method == "both" else [args.method]
	for method in methods:
		results = run_method(method, rows, root, args)
		(root / f"{method}_summary.json").write_text(json.dumps(summarize(results), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
		print(json.dumps({"method": method, **summarize(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
