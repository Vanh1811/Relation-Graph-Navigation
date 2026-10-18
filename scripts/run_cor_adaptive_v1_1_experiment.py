#!/usr/bin/env python3
"""Run Adaptive Branch Routing v1.1 on frozen CoR evaluation IDs.

The default run is the required diagnostic-13 (10 COR_ONLY_CORRECT and
3 OURS_ONLY_CORRECT cases from the completed paired analysis). The
``expanded_diagnostic`` run is the Adaptive v1.1.1 36-case diagnostic. Use
``--run full`` only after reviewing the diagnostic output. The frozen baseline
is read for IDs and comparison only; it is never rerun.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
from chain_of_relations.methods.adaptive_v1_1 import AdaptiveRoutingAgent  # noqa: E402
from chain_of_relations.methods.cor.agent import CoRAgent  # noqa: E402


BASELINE_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "cor_phase1_baseline_20261014_fixed"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1"
EXPANDED_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_1"
SUBSET_ROOT = PROJECT_ROOT / "outputs" / "subsets"
SUBSETS = ("cwq_branch_2", "grailqa_branch_2")
PAIRED_CASES_PATH = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "comparison" / "paired_cases" / "paired_case_summary.json"

DIAGNOSTIC_IDS = {
	"cwq_branch_2": [
		"WebQTrn-3034_9ad09040a8f87a2abeca21f7311fd71e",
		"WebQTrn-64_984678664419049133d2bd748b68df1b",
		"WebQTrn-3535_2687e08154270a5a87006ee2345f3b60",
		"WebQTest-1307_727248e493325c05836e5bb159c752d5",
		"WebQTest-743_1d6c214af3c3e9fcb3689bec2a8e8336",
		"WebQTest-989_4b6636a049fa767111eef6ce82bf7054",
	],
	"grailqa_branch_2": [
		"3202155015000",
		"3204132008000",
		"4302370011000",
		"4301299003000",
		"4301571012000",
		"3204176003000",
		"3201942006000",
	],
}

COR_ONLY_DIAGNOSTIC_IDS = {
	"WebQTrn-3034_9ad09040a8f87a2abeca21f7311fd71e",
	"WebQTrn-64_984678664419049133d2bd748b68df1b",
	"WebQTrn-3535_2687e08154270a5a87006ee2345f3b60",
	"WebQTest-1307_727248e493325c05836e5bb159c752d5",
	"WebQTest-743_1d6c214af3c3e9fcb3689bec2a8e8336",
	"WebQTest-989_4b6636a049fa767111eef6ce82bf7054",
	"3202155015000",
	"3204132008000",
	"4302370011000",
	"4301299003000",
}

OURS_ONLY_DIAGNOSTIC_IDS = {
	"4301571012000",
	"3204176003000",
	"3201942006000",
}


def load_paired_categories() -> dict[str, dict[str, str]]:
	data = json.loads(PAIRED_CASES_PATH.read_text(encoding="utf-8"))
	categories: dict[str, dict[str, str]] = {}
	for case in data.get("paired_cases", []):
		example_id = str(case.get("id", ""))
		if example_id:
			categories[example_id] = str(case.get("category", "UNKNOWN"))
	return categories


def load_frozen_rows(baseline_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
	selected_ids = json.loads((baseline_root / "sample_ids.json").read_text(encoding="utf-8"))
	selected_rows: dict[str, list[dict[str, Any]]] = {}
	for subset in SUBSETS:
		path = SUBSET_ROOT / f"{subset}.jsonl"
		if not path.exists():
			raise SystemExit(f"missing inference-safe subset: {path}")
		rows_by_id = {str(row.get("id", "")): row for row in read_jsonl(path)}
		rows = []
		for example_id in selected_ids.get(subset, []):
			if str(example_id) not in rows_by_id:
				raise SystemExit(f"frozen baseline ID not found in {path}: {example_id}")
			rows.append(rows_by_id[str(example_id)])
		selected_rows[subset] = rows
	return selected_rows, {subset: [str(value) for value in selected_ids[subset]] for subset in SUBSETS}


def diagnostic_rows(
	selected_rows: dict[str, list[dict[str, Any]]],
	requested_ids: dict[str, list[str]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
	rows_by_subset: dict[str, list[dict[str, Any]]] = {}
	selected_ids_by_subset: dict[str, list[str]] = {}
	for subset in SUBSETS:
		by_id = {str(row.get("id", "")): row for row in selected_rows[subset]}
		ids = (requested_ids or DIAGNOSTIC_IDS)[subset]
		missing = [item for item in ids if item not in by_id]
		if missing:
			raise SystemExit(f"diagnostic IDs missing from frozen sample {subset}: {missing}")
		rows_by_subset[subset] = [by_id[item] for item in ids]
		selected_ids_by_subset[subset] = ids
	return rows_by_subset, selected_ids_by_subset


def git_value(args: list[str], default: str = "unknown") -> str:
	try:
		return subprocess.run(
			["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
		).stdout.strip() or default
	except (OSError, subprocess.CalledProcessError):
		return default


def run_one(agent: AdaptiveRoutingAgent, row: dict[str, Any], detail_dir: Path) -> dict[str, Any]:
	question = str(row.get("question", ""))
	example_id = str(row.get("id", ""))
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
	except Exception as exc:  # preserve one row and continue the experiment
		logging.exception("Adaptive routing failed for %s", example_id)
		result = {"action": "ERROR", "results": [], "reasoning_chains": ""}
		error = f"{type(exc).__name__}: {exc}"

	latency_ms = (time.perf_counter() - started) * 1000.0
	prediction = prediction_names(result)
	hit, f1 = evaluate(prediction, gold_answers)
	if str(result.get("action", "")).upper() == "ERROR" and error is None:
		error = str(result.get("reasoning_chains", "LLM/agent error"))
	input_tokens = integer_usage(prompt_history, "input_tokens")
	output_tokens = integer_usage(prompt_history, "output_tokens")
	branch_result = result.get("adaptive_branch_result", {})
	if not isinstance(branch_result, dict):
		branch_result = {}
	branch_source = branch_result or result
	decision_fields = {
		"router_mode": result.get("router_mode", "cor"),
		"router_confidence": result.get("router_confidence", 0.0),
		"router_reason": result.get("router_reason", ""),
		"branch_count": result.get("branch_count", 0),
		"shared_answer_type": result.get("shared_answer_type", ""),
		"proposed_branches": result.get("proposed_branches", []),
		"validation_passed": result.get("validation_passed", False),
		"validation_failures": result.get("validation_failures", []),
		"fallback_used": result.get("fallback_used", False),
		"fallback_reason": result.get("fallback_reason"),
		"final_execution_mode": result.get("final_execution_mode", "cor_fallback"),
	}
	row_result: dict[str, Any] = {
		"id": example_id,
		"question": question,
		"gold_answers": gold_answers,
		"prediction": prediction,
		"hits_at_1": hit,
		"f1": f1,
		"relation_chain": relation_chain_from_trace(result),
		"llm_calls": len(prompt_history),
		"kg_queries": len(sparql_history),
		"token_usage": {
			"input_tokens": input_tokens,
			"output_tokens": output_tokens,
			"total_tokens": input_tokens + output_tokens,
		},
		"latency_ms": latency_ms,
		"error": error,
		**decision_fields,
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
	detail = {
		**row_result,
		"action": result.get("action", ""),
		"reasoning_chains": result.get("reasoning_chains", ""),
		"prompt_history": prompt_history,
		"sparql_history": sparql_history,
		"step_history": step_history,
	}
	if branch_result:
		detail["adaptive_branch_result"] = branch_result
	detail_dir.mkdir(parents=True, exist_ok=True)
	(detail_dir / f"{example_id}.json").write_text(
		json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
	)
	return row_result


def router_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
	return {
		"cor": sum(row.get("router_mode") == "cor" and not row.get("fallback_used") for row in rows),
		"branch_2": sum(row.get("router_mode") == "branch" and row.get("branch_count") == 2 for row in rows),
		"branch_3": sum(row.get("router_mode") == "branch" and row.get("branch_count") == 3 for row in rows),
		"cor_fallback": sum(row.get("final_execution_mode") == "cor_fallback" for row in rows),
	}


def load_result_map(root: Path) -> dict[str, dict[str, Any]]:
	result: dict[str, dict[str, Any]] = {}
	for subset in SUBSETS:
		path = root / f"{subset}_results.jsonl"
		for row in read_jsonl(path):
			result[str(row.get("id", ""))] = row
	return result


def write_summary(
	rows_by_subset: dict[str, list[dict[str, Any]]],
	all_rows: list[dict[str, Any]],
	output_path: Path,
	mode: str,
	baseline_rows: dict[str, dict[str, Any]],
) -> None:
	metrics = {subset: summarize(rows) for subset, rows in rows_by_subset.items()}
	metrics["ALL"] = summarize(all_rows)
	distribution = {subset: router_distribution(rows) for subset, rows in rows_by_subset.items()}
	distribution["ALL"] = router_distribution(all_rows)
	lines = [
		"# Adaptive Branch Routing v1.1 summary",
		"",
		f"Run: **{mode}**. The frozen CoR baseline was read for comparison only and was not rerun.",
		"",
		"## Effectiveness and efficiency",
		"",
		"| Subset | N | Hits@1 | F1 | Errors | Avg LLM calls | Avg KG queries | Avg tokens | Avg latency (ms) |",
		"|---|---:|---:|---:|---:|---:|---:|---:|---:|",
	]
	for subset in [*SUBSETS, "ALL"]:
		value = metrics[subset]
		lines.append(
			f"| {subset} | {value['n']} | {value['hits_at_1']:.2%} | {value['f1']:.4f} | "
			f"{value['errors']} | {value['average_llm_calls']:.2f} | {value['average_kg_queries']:.2f} | "
			f"{value['average_tokens']:.2f} | {value['average_latency_ms']:.2f} |"
		)
	lines += ["", "## Routing distribution", "", "| Subset | cor | branch_2 | branch_3 | cor_fallback |", "|---|---:|---:|---:|---:|"]
	for subset in [*SUBSETS, "ALL"]:
		value = distribution[subset]
		lines.append(f"| {subset} | {value['cor']} | {value['branch_2']} | {value['branch_3']} | {value['cor_fallback']} |")
	if mode == "diagnostic":
		adaptive_by_id = {row["id"]: row for row in all_rows}
		preserved = sum(
			bool(adaptive_by_id[item].get("hits_at_1"))
			for item in COR_ONLY_DIAGNOSTIC_IDS
			if item in adaptive_by_id
		)
		recovered = sum(
			bool(adaptive_by_id[item].get("hits_at_1"))
			for item in OURS_ONLY_DIAGNOSTIC_IDS
			if item in adaptive_by_id
		)
		lines += [
			"",
			"## Required diagnostic result",
			"",
			f"- COR_ONLY_CORRECT preserved: **{preserved} / 10**",
			f"- OURS_ONLY_CORRECT recovered: **{recovered} / 3**",
			"",
			"The diagnostic must be reviewed before any full-100 run.",
		]
	lines += ["", "## Reproducibility", "", f"- Git commit: `{git_value(['rev-parse', 'HEAD'])}`", f"- Branch: `{git_value(['branch', '--show-current'])}`", f"- Model: `{os.getenv('MODEL_NAME', '')}`", "- Router prompt: `adaptive-routing-v1.1-json-v1`", "- Confidence threshold: `0.75`", "- MAX_BRANCHES: `3`", ""]
	output_path.write_text("\n".join(lines), encoding="utf-8")


def write_router_decisions(rows: list[dict[str, Any]], output_path: Path) -> None:
	lines = [
		"# Adaptive router decisions",
		"",
		"Each row records the runtime router proposal, deterministic validation, fallback behavior, and final execution mode.",
		"",
		"| ID | Router mode | Confidence | Branches | Validated | Fallback | Final mode | Reason |",
		"|---|---|---:|---:|---|---|---|---|",
	]
	for row in rows:
		reason = str(row.get("router_reason", "")).replace("|", "/").replace("\n", " ")
		lines.append(
			f"| `{row['id']}` | {row.get('router_mode')} | {float(row.get('router_confidence', 0.0)):.2f} | "
			f"{row.get('branch_count', 0)} | {row.get('validation_passed')} | {row.get('fallback_used')} | "
			f"{row.get('final_execution_mode')} | {reason} |"
		)
	output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--run", choices=["diagnostic", "expanded_diagnostic", "full"], default="diagnostic")
	parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
	parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
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
	if not os.getenv("MODEL_NAME"):
		raise SystemExit("MODEL_NAME is required")
	if not os.getenv("OPENAI_API_KEY"):
		raise SystemExit("OPENAI_API_KEY is required")
	if not args.baseline_root.exists():
		raise SystemExit(f"frozen baseline root does not exist: {args.baseline_root}")

	selected_rows, selected_ids = load_frozen_rows(args.baseline_root)
	if args.run == "diagnostic":
		selected_rows, selected_ids = diagnostic_rows(selected_rows)
	elif args.run == "expanded_diagnostic":
		categories = load_paired_categories()
		expanded_ids: dict[str, list[str]] = {subset: [] for subset in SUBSETS}
		for subset in SUBSETS:
			available = {str(row.get("id", "")) for row in selected_rows[subset]}
			for example_id, category in categories.items():
				if category in {"COR_ONLY_CORRECT", "OURS_ONLY_CORRECT", "BOTH_WRONG"}:
					# The frozen paired summary is keyed globally; preserve only
					# IDs belonging to this inference-safe subset.
					if example_id in available:
						expanded_ids[subset].append(example_id)
			if len(expanded_ids[subset]) == 0:
				raise SystemExit(f"no expanded diagnostic IDs found for {subset}")
		selected_rows, selected_ids = diagnostic_rows(selected_rows, expanded_ids)
		args.output_root = EXPANDED_OUTPUT_ROOT
	run_root_name = {
		"diagnostic": "diagnostic_13",
		"expanded_diagnostic": "diagnostic_36",
		"full": "full_100",
	}[args.run]
	run_root = args.output_root / run_root_name
	run_root.mkdir(parents=True, exist_ok=True)
	config = {
		"experiment": "adaptive_branch_routing_v1_1_1" if args.run == "expanded_diagnostic" else "adaptive_branch_routing_v1_1",
		"run": args.run,
		"git_commit": git_value(["rev-parse", "HEAD"]),
		"branch": git_value(["branch", "--show-current"]),
		"method": "adaptive_router_with_cor_fallback_and_branch_v1_execution",
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
		"router_prompt_version": "adaptive-routing-v1.1-json-v1",
		"confidence_threshold": args.confidence_threshold,
		"MAX_BRANCHES": 3,
		"runtime_structure_inputs": ["question", "topic_entities", "topic_entity_names"],
		"gold_structure_used_at_runtime": False,
		"gold_answers_used_for_routing": False,
		"frozen_baseline_rerun": False,
		"frozen_sample_ids": selected_ids,
		"paired_case_categories": load_paired_categories() if args.run == "expanded_diagnostic" else {},
	}
	(run_root / "adaptive_v1_1_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
	(run_root / "sample_ids.json").write_text(json.dumps(selected_ids, ensure_ascii=False, indent=2), encoding="utf-8")

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
	agent = AdaptiveRoutingAgent(cor_agent, confidence_threshold=args.confidence_threshold, max_branches=3)
	baseline_rows = load_result_map(args.baseline_root)
	all_rows: list[dict[str, Any]] = []
	rows_for_summary: dict[str, list[dict[str, Any]]] = {}
	for subset, rows in selected_rows.items():
		results_path = run_root / f"{subset}_results.jsonl"
		detail_dir = run_root / "details" / subset
		existing: dict[str, dict[str, Any]] = {}
		if args.resume and results_path.exists():
			for saved in read_jsonl(results_path):
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
		rows_for_summary[subset] = results
		all_rows.extend(results)

	(run_root / "results.jsonl").write_text(
		"\n".join(json.dumps(row, ensure_ascii=False) for row in all_rows) + "\n", encoding="utf-8"
	)
	summaries = {subset: summarize(rows) for subset, rows in rows_for_summary.items()}
	summaries["ALL"] = summarize(all_rows)
	summaries["routing_distribution"] = {subset: router_distribution(rows) for subset, rows in rows_for_summary.items()}
	summaries["routing_distribution"]["ALL"] = router_distribution(all_rows)
	(run_root / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
	write_summary(rows_for_summary, all_rows, run_root / "summary.md", args.run, baseline_rows)
	write_router_decisions(all_rows, run_root / "router_decisions.md")
	print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
