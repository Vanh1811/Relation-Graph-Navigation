#!/usr/bin/env python3
"""Run frozen Adaptive v1.1.3 on the existing 36-case diagnostic set only."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from run_adaptive_v1_1_3_experiment import (  # noqa: E402
	load_safe_project_env,
	make_agent,
	read_jsonl,
	run_one,
)


SUBSETS = ("cwq_branch_2", "grailqa_branch_2")
SOURCE_CONFIG = (
	PROJECT_ROOT
	/ "outputs"
	/ "experiments"
	/ "phase1"
	/ "adaptive_v1_1_1"
	/ "diagnostic_36"
	/ "adaptive_v1_1_config.json"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_3"
DIAGNOSTIC_ROOT = OUTPUT_ROOT / "diagnostic_36"
RESULT_PATH = DIAGNOSTIC_ROOT / "results.jsonl"


def _load_source() -> tuple[list[dict[str, Any]], dict[str, str]]:
	data = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
	frozen = data.get("frozen_sample_ids", {})
	categories = {str(key): str(value) for key, value in data.get("paired_case_categories", {}).items()}
	ids: list[str] = []
	seen: set[str] = set()
	for subset in SUBSETS:
		for value in frozen.get(subset, []):
			case_id = str(value)
			if case_id in seen:
				raise SystemExit(f"duplicate frozen diagnostic ID: {case_id}")
			seen.add(case_id)
			ids.append(case_id)
	if len(ids) != 36:
		raise SystemExit(f"expected 36 frozen diagnostic IDs, found {len(ids)}")
	if {categories.get(case_id) for case_id in ids} != {"COR_ONLY_CORRECT", "OURS_ONLY_CORRECT", "BOTH_WRONG"}:
		raise SystemExit("frozen diagnostic IDs do not contain all required source groups")
	counts = {group: sum(categories[case_id] == group for case_id in ids) for group in {categories[case_id] for case_id in ids}}
	if counts != {"COR_ONLY_CORRECT": 10, "OURS_ONLY_CORRECT": 3, "BOTH_WRONG": 23}:
		raise SystemExit(f"unexpected frozen diagnostic group counts: {counts}")

	rows_by_id: dict[str, dict[str, Any]] = {}
	for subset in SUBSETS:
		path = PROJECT_ROOT / "outputs" / "subsets" / f"{subset}.jsonl"
		for row in read_jsonl(path):
			rows_by_id[str(row.get("id", ""))] = row
	missing = [case_id for case_id in ids if case_id not in rows_by_id]
	if missing:
		raise SystemExit(f"frozen diagnostic IDs missing from subset files: {missing}")
	return [rows_by_id[case_id] for case_id in ids], categories


def _branch_records(result: dict[str, Any]) -> list[dict[str, Any]]:
	decomposition = result.get("branch_decomposition") or {}
	definitions = decomposition.get("branches", []) if isinstance(decomposition, dict) else []
	records: list[dict[str, Any]] = []
	for index, trace in enumerate(result.get("branch_traces", []) or []):
		definition = definitions[index] if index < len(definitions) and isinstance(definitions[index], dict) else {}
		root = trace.get("topic_entity") or definition.get("root_entity", {})
		records.append({
			"branch_id": trace.get("branch_id", definition.get("branch_id", f"branch_{index + 1}")),
			"root": root,
			"constraint_property": trace.get("constraint_property", definition.get("constraint_property", "")),
			"constraint_summary": trace.get("constraint_summary", definition.get("constraint_summary", "")),
			"target_answer_type": trace.get("target_answer_type", definition.get("target_answer_type", "")),
			"relation_chain": trace.get("relation_chain", []),
			"candidate_count": trace.get("candidate_count", 0),
			"final_branch_answer_set": trace.get("final_branch_answers", []),
			"status": trace.get("status", ""),
		})
	return records


def _enrich(row_result: dict[str, Any], source_group: str) -> dict[str, Any]:
	usage = dict(row_result.get("token_usage") or {})
	usage.setdefault(
		"total_tokens",
		int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0),
	)
	row_result["token_usage"] = usage
	row_result["tokens"] = int(usage.get("total_tokens", 0) or 0)
	row_result["question_id"] = row_result.get("id", "")
	row_result["source_group"] = source_group
	row_result["execution_mode"] = row_result.get("final_execution_mode", "cor")
	row_result["branch_count"] = int(
		row_result.get("branches_created", 0)
		or len(row_result.get("branch_traces", []) or [])
		or len((row_result.get("proposed_branches") or []))
	)
	failures = [str(value) for value in row_result.get("validation_failures", []) or []]
	row_result["answer_type_validation"] = {
		"passed": not any("ANSWER_TYPE" in value or "answer type" in value.lower() for value in failures),
		"failures": [value for value in failures if "ANSWER_TYPE" in value or "answer type" in value.lower()],
	}
	row_result["constraint_property_validation"] = {
		"passed": not any("CONSTRAINT_PROPERTY" in value or "constraint property" in value.lower() for value in failures),
		"failures": [value for value in failures if "CONSTRAINT_PROPERTY" in value or "constraint property" in value.lower()],
	}
	row_result["validation_rejected"] = not bool(row_result.get("validation_passed", False))
	row_result["merge_result"] = row_result.get("merge_outcome", "")
	row_result["branches"] = _branch_records(row_result)
	return row_result


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--run", action="store_true", help="run the frozen 36-case diagnostic")
	parser.add_argument("--resume", action="store_true", help="resume an incomplete output file")
	parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if not args.run:
		raise SystemExit("choose --run")
	load_safe_project_env()
	logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
	if not os.getenv("MODEL_NAME") or not os.getenv("OPENAI_API_KEY"):
		raise SystemExit("MODEL_NAME and OPENAI_API_KEY are required")
	rows, categories = _load_source()
	DIAGNOSTIC_ROOT.mkdir(parents=True, exist_ok=True)
	if RESULT_PATH.exists() and not args.resume:
		raise SystemExit(f"output exists; use --resume to continue: {RESULT_PATH}")

	existing: dict[str, dict[str, Any]] = {}
	if args.resume and RESULT_PATH.exists():
		for item in read_jsonl(RESULT_PATH):
			existing[str(item.get("id", ""))] = item

	settings = argparse.Namespace(
		relation_width=3,
		depth=3,
		sample_relation_threshold=1000,
		remove_unnecessary_rel=True,
		temperature_exploration=0.3,
		temperature_reasoning=0.1,
		max_token=1024,
		confidence_threshold=0.75,
	)
	agent = make_agent("adaptive", settings)
	mode = "a" if args.resume and RESULT_PATH.exists() else "w"
	with RESULT_PATH.open(mode, encoding="utf-8") as handle:
		for index, row in enumerate(rows, start=1):
			case_id = str(row.get("id", ""))
			logging.info("[adaptive-v1.1.3] %d/36 id=%s group=%s", index, case_id, categories[case_id])
			if case_id in existing:
				continue
			result = run_one(agent, "adaptive", 1, row, DIAGNOSTIC_ROOT / "details" / "adaptive")
			result = _enrich(result, categories[case_id])
			detail_path = DIAGNOSTIC_ROOT / "details" / "adaptive" / f"{case_id}.json"
			if detail_path.exists():
				detail = json.loads(detail_path.read_text(encoding="utf-8"))
				detail.update(result)
				detail_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
			handle.write(json.dumps(result, ensure_ascii=False) + "\n")
			handle.flush()

	config = {
		"experiment": "adaptive_v1_1_3_diagnostic_36",
		"run": "existing_frozen_diagnostic_only",
		"model": os.getenv("MODEL_NAME", ""),
		"router_prompt_version": "adaptive-routing-v1.1.3-answer-type-aware-json-v1",
		"depth": 3,
		"relation_width": 3,
		"sample_relation_threshold": 1000,
		"remove_unnecessary_rel": True,
		"temperature_exploration": 0.3,
		"temperature_reasoning": 0.1,
		"max_token": 1024,
		"confidence_threshold": 0.75,
		"gold_structure_used_at_runtime": False,
		"gold_answers_used_for_routing": False,
		"historical_methods_rerun": False,
		"source_config": str(SOURCE_CONFIG),
		"source_group_counts": {
			group: sum(categories[str(row.get("id", ""))] == group for row in rows)
			for group in sorted({categories[str(row.get("id", ""))] for row in rows})
		},
	}
	(DIAGNOSTIC_ROOT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	final_rows = read_jsonl(RESULT_PATH)
	print(json.dumps({"rows": len(final_rows), "output": str(RESULT_PATH), "historical_methods_rerun": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
