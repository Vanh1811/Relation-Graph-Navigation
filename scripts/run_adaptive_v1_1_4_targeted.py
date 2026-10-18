#!/usr/bin/env python3
"""Run the frozen v1.1.4 complementary-property targeted validation set."""

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

from run_adaptive_v1_1_3_experiment import load_safe_project_env, read_jsonl, run_one
from chain_of_relations.kg_backend import get_backend
from chain_of_relations.methods.adaptive_v1_1_4 import AdaptiveRoutingAgent
from chain_of_relations.methods.cor.agent import CoRAgent


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_4"
TARGETED_ROOT = OUTPUT_ROOT / "targeted_validation"
RESULT_PATH = TARGETED_ROOT / "results.jsonl"
V113_RESULT_PATH = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_3" / "diagnostic_36" / "results.jsonl"

TARGETED_IDS = [
	"3201942006000",  # GameCube controller + Japan
	"3201942014000",  # Xbox 360 controller + Europe
	"3203433006000",  # Bobby Helm + rock music
	"4301182008000",  # menstrual cramps + SLUDGE syndrome
	"4301990001000",  # Pictures of Lily + Track Record
	"3204176003000",  # same-property comic category control
	"4301299003000",  # same-property computer designer control
	"4300753011000",  # branch-intermediate negative control
]
SUBSETS = ("cwq_branch_2", "grailqa_branch_2")


def make_agent(args: argparse.Namespace) -> AdaptiveRoutingAgent:
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
	return AdaptiveRoutingAgent(
		CoRAgent(**kwargs),
		confidence_threshold=args.confidence_threshold,
		max_branches=3,
	)


def load_rows() -> list[dict[str, Any]]:
	by_id: dict[str, dict[str, Any]] = {}
	for subset in SUBSETS:
		for row in read_jsonl(PROJECT_ROOT / "outputs" / "subsets" / f"{subset}.jsonl"):
			by_id[str(row.get("id", ""))] = row
	missing = [case_id for case_id in TARGETED_IDS if case_id not in by_id]
	if missing:
		raise SystemExit(f"targeted IDs missing from frozen subset files: {missing}")
	return [by_id[case_id] for case_id in TARGETED_IDS]


def load_v113() -> dict[str, dict[str, Any]]:
	return {
		str(row.get("id", "")): row
		for row in read_jsonl(V113_RESULT_PATH)
	}


def branch_records(result: dict[str, Any]) -> list[dict[str, Any]]:
	definitions = result.get("proposed_branches", []) or []
	traces = result.get("branch_traces", []) or []
	records: list[dict[str, Any]] = []
	for index, trace in enumerate(traces):
		definition = definitions[index] if index < len(definitions) and isinstance(definitions[index], dict) else {}
		candidate_count = int(trace.get("candidate_count", 0) or 0)
		records.append({
			"branch_id": trace.get("branch_id", definition.get("branch_id", f"branch_{index + 1}")),
			"root": trace.get("topic_entity", definition.get("root_entity", {})),
			"constraint_property": trace.get("constraint_property", definition.get("constraint_property", "")),
			"constraint_summary": trace.get("constraint_summary", definition.get("constraint_summary", "")),
			"target_answer_type": trace.get("target_answer_type", definition.get("target_answer_type", "")),
			"relation_chain": trace.get("relation_chain", []),
			"candidate_count": candidate_count,
			"final_branch_answer_set": trace.get("final_branch_answers", []),
			"question_supported": bool(definition.get("question_supported", False)),
			"independently_generates_final_candidates": bool(
				definition.get("independently_generates_final_candidates", candidate_count > 0)
			),
			"status": trace.get("status", ""),
		})
	return records


def enrich(result: dict[str, Any], source_row: dict[str, Any], v113: dict[str, dict[str, Any]]) -> dict[str, Any]:
	case_id = str(result.get("id", ""))
	branches = branch_records(result)
	usage = result.get("token_usage") or {}
	result["tokens"] = int(
		usage.get("total_tokens", 0) or 0
		or (int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0))
	)
	result["question_id"] = case_id
	result["v1_1_3_structure"] = v113.get(case_id, {}).get("router_structure", "")
	result["execution_mode"] = result.get("final_execution_mode", "cor")
	result["branch_count"] = len(branches) if result.get("router_structure") == "BRANCH_FINAL" else 0
	result["branches"] = branches
	result["validation_result"] = "PASS" if result.get("validation_passed", True) else "REJECTED"
	result["validation_reason"] = result.get("validation_reason") or (
		"; ".join(str(item) for item in result.get("validation_failures", []))
		if result.get("validation_failures") else ""
	)
	result["final_prediction"] = list(result.get("prediction", []))
	result["final_hits_at_1"] = result.get("hits_at_1", 0)
	result["final_f1"] = result.get("f1", 0.0)
	result["errors"] = result.get("error")
	result["question"] = source_row.get("question", result.get("question", ""))
	return result


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--run", action="store_true")
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
	parser.add_argument("--depth", type=int, default=3)
	parser.add_argument("--relation-width", type=int, default=3)
	parser.add_argument("--sample-relation-threshold", type=int, default=1000)
	parser.add_argument("--temperature-exploration", type=float, default=0.3)
	parser.add_argument("--temperature-reasoning", type=float, default=0.1)
	parser.add_argument("--max-token", type=int, default=1024)
	parser.add_argument("--confidence-threshold", type=float, default=0.75)
	parser.add_argument("--remove-unnecessary-rel", action=argparse.BooleanOptionalAction, default=True)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if not args.run:
		raise SystemExit("choose --run")
	load_safe_project_env()
	logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
	if not os.getenv("MODEL_NAME") or not os.getenv("OPENAI_API_KEY"):
		raise SystemExit("MODEL_NAME and OPENAI_API_KEY are required")
	TARGETED_ROOT.mkdir(parents=True, exist_ok=True)
	rows = load_rows()
	v113 = load_v113()
	existing: dict[str, dict[str, Any]] = {}
	if args.resume and RESULT_PATH.exists():
		for item in read_jsonl(RESULT_PATH):
			existing[str(item.get("id", ""))] = item
	if RESULT_PATH.exists() and not args.resume:
		raise SystemExit(f"output exists; use --resume to continue: {RESULT_PATH}")
	agent = make_agent(args)
	mode = "a" if args.resume and RESULT_PATH.exists() else "w"
	with RESULT_PATH.open(mode, encoding="utf-8") as handle:
		for index, row in enumerate(rows, start=1):
			case_id = str(row.get("id", ""))
			logging.info("[adaptive-v1.1.4] %d/%d id=%s", index, len(rows), case_id)
			if case_id in existing:
				continue
			result = run_one(agent, "adaptive", 1, row, TARGETED_ROOT / "details" / "adaptive")
			result = enrich(result, row, v113)
			detail_path = TARGETED_ROOT / "details" / "adaptive" / f"{case_id}.json"
			if detail_path.exists():
				detail = json.loads(detail_path.read_text(encoding="utf-8"))
				detail.update(result)
				detail_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
			handle.write(json.dumps(result, ensure_ascii=False) + "\n")
			handle.flush()
	config = {
		"experiment": "adaptive_v1_1_4_complementary_property_targeted",
		"run": "targeted_structural_validation_only",
		"targeted_ids": TARGETED_IDS,
		"model": os.getenv("MODEL_NAME", ""),
		"depth": args.depth,
		"relation_width": args.relation_width,
		"sample_relation_threshold": args.sample_relation_threshold,
		"remove_unnecessary_rel": args.remove_unnecessary_rel,
		"temperature_exploration": args.temperature_exploration,
		"temperature_reasoning": args.temperature_reasoning,
		"max_token": args.max_token,
		"confidence_threshold": args.confidence_threshold,
		"historical_methods_rerun": False,
		"full_100_run": False,
	}
	(TARGETED_ROOT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	print(json.dumps({"rows": len(read_jsonl(RESULT_PATH)), "output": str(RESULT_PATH), "historical_methods_rerun": False, "full_100_run": False}, indent=2))


if __name__ == "__main__":
	main()
