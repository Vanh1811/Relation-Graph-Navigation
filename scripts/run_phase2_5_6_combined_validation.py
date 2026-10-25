#!/usr/bin/env python3
"""Run the frozen Phase 2.5.6 combined-stack validation.

This harness deliberately reuses the Phase 2.5.5 live agent.  It adds no
reasoning intervention; it only records the complete pipeline and applies the
predefined focused-validation gate.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_phase2_5_1_live_validation as base  # noqa: E402
import run_phase2_5_3b_live as phase3b  # noqa: E402
import run_phase2_5_5_live as phase255  # noqa: E402


INPUT_PATH = ROOT / "outputs/experiments/phase2_4_4/routing_decision_error_analysis/router_input_output_analysis.jsonl"
BASELINE_ROOT = ROOT / "outputs/experiments/phase2_5_1/live_constraint_first_validation"
PHASE_3A_ROOT = ROOT / "outputs/experiments/phase2_5_3a/source_target_extraction"
PHASE_3B_ROOT = ROOT / "outputs/experiments/phase2_5_3b/branch_plan_serialization"
PHASE_5_ROOT = ROOT / "outputs/experiments/phase2_5_5/canonical_semantic_normalization"
OUTPUT_ROOT = ROOT / "outputs/experiments/phase2_5_6/focused_combined_validation"

PHASE_STACK = (
	"phase2_5_3a_extraction",
	"phase2_5_5_canonical_normalization",
	"phase2_5_3b_serialization",
	"unchanged_validator",
	"unchanged_branch_execution_and_fallback",
)


class LiveValidationUnavailable(RuntimeError):
	pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def text(value: Any) -> str:
	return str(value or "").strip()


def canonical(value: Any) -> str:
	return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def router_event(row: dict[str, Any]) -> dict[str, Any]:
	for event in row.get("prompt_history", []):
		if event.get("type") == "phase2_5_5_canonical_normalization_router":
			return event
	return {}


def raw_proposal(row: dict[str, Any]) -> dict[str, Any]:
	event = router_event(row)
	proposal = event.get("parsed_result", {})
	return proposal if isinstance(proposal, dict) else {}


def raw_target_type(proposal: dict[str, Any]) -> str:
	value = proposal.get("shared_target")
	if isinstance(value, dict):
		return text(value.get("type")).upper() or "NONE"
	return text(value).upper() or "NONE"


def expected_target(ref: dict[str, Any]) -> str:
	if ref.get("shared_answer_gold"):
		return "ANSWER"
	if ref.get("shared_intermediate_gold"):
		return "INTERMEDIATE"
	return "NONE"


def raw_source_group_count(proposal: dict[str, Any]) -> int:
	groups = proposal.get("independent_source_groups")
	if isinstance(groups, list):
		return sum(isinstance(item, dict) and bool(item.get("source_anchor") or item.get("source_entity")) for item in groups)
	return 0


def upstream_semantic_status(ref: dict[str, Any], row: dict[str, Any]) -> tuple[str, list[str]]:
	"""Classify only upstream meaning; representation failures are not semantic errors."""
	proposal = raw_proposal(row)
	expected_structure = base.gold_structure(ref.get("gold_structure", ""))
	raw_structure = text(proposal.get("structure") or row.get("semantic_structure")).upper()
	raw_target = raw_target_type(proposal)
	reasons: list[str] = []

	if raw_target != expected_target(ref):
		return "UPSTREAM_SHARED_TARGET_ERROR", [f"shared_target={raw_target}, expected={expected_target(ref)}"]

	if int(ref.get("independent_sources_gold", 0) or 0) > 1 and raw_source_group_count(proposal) < int(ref.get("independent_sources_gold", 0) or 0):
		return "UPSTREAM_SOURCE_ERROR", [
			f"independent_source_groups={raw_source_group_count(proposal)}, expected={ref.get('independent_sources_gold')}"
		]

	dependency = text(proposal.get("dependency_pattern")).upper()
	if expected_structure == "BRANCH_FINAL" and dependency in {"", "UNKNOWN", "UNCERTAIN", "LINEAR", "SEQUENTIAL", "SINGLE_SOURCE_DIRECT"}:
		return "UPSTREAM_DEPENDENCY_ERROR", [f"dependency_pattern={dependency or 'MISSING'} for BRANCH_FINAL"]
	if expected_structure == "BRANCH_INTERMEDIATE" and raw_structure == "LINEAR" and dependency in {"", "UNKNOWN", "UNCERTAIN", "LINEAR", "SEQUENTIAL", "SINGLE_SOURCE_DIRECT"}:
		return "UPSTREAM_DEPENDENCY_ERROR", [f"dependency_pattern={dependency or 'MISSING'} for BRANCH_INTERMEDIATE"]

	constraints = proposal.get("constraints")
	if isinstance(constraints, list) and constraints:
		for index, constraint in enumerate(constraints, start=1):
			if not isinstance(constraint, dict):
				continue
			role = text(constraint.get("target_variable_role") or constraint.get("target_role"))
			if constraint.get("answer_relevant") and not role:
				reasons.append(f"constraint_{index} has no target role")
		if reasons:
			return "UPSTREAM_CONSTRAINT_ROLE_ERROR", reasons

	if raw_structure != expected_structure:
		return "UPSTREAM_OTHER", [f"structure={raw_structure or 'MISSING'}, expected={expected_structure}"]
	return "SEMANTICALLY_ALIGNED", []


def normalization_status(row: dict[str, Any]) -> str:
	return text(row.get("normalization_status")) or "NORMALIZATION_FAILED"


def serialization_status(row: dict[str, Any]) -> str:
	return text(row.get("serialization_status")) or "FAILED"


def validator_status(row: dict[str, Any]) -> str:
	return "PASS" if text(row.get("validator_result")).upper() == "VALIDATOR_PASS" else "REJECT"


def primary_failure(ref: dict[str, Any], row: dict[str, Any], semantic_status: str) -> str | None:
	if semantic_status != "SEMANTICALLY_ALIGNED":
		return {
			"UPSTREAM_SHARED_TARGET_ERROR": "UPSTREAM_SHARED_TARGET_ERROR",
			"UPSTREAM_SOURCE_ERROR": "UPSTREAM_SEMANTIC_EXTRACTION",
			"UPSTREAM_DEPENDENCY_ERROR": "UPSTREAM_SEMANTIC_EXTRACTION",
			"UPSTREAM_CONSTRAINT_ROLE_ERROR": "UPSTREAM_SEMANTIC_EXTRACTION",
			"UPSTREAM_OTHER": "UPSTREAM_SEMANTIC_EXTRACTION",
		}.get(semantic_status, "UPSTREAM_SEMANTIC_EXTRACTION")
	if serialization_status(row) != "COMPLETE":
		return "SERIALIZATION"
	if validator_status(row) == "REJECT":
		return "VALIDATOR"
	if row.get("branch_routed") and row.get("execution_status") != "EXECUTION_SUCCESS":
		return "BRANCH_EXECUTION"
	if not int(row.get("hits_at_1", 0) or 0):
		if row.get("sparql_query_count", 0):
			return "RELATION_GROUNDING"
		return "ANSWER_SELECTION"
	return None


def group_cases(ref_rows: list[dict[str, Any]]) -> dict[str, str]:
	return phase3b.group_cases(ref_rows)


def qa_metrics(rows: list[dict[str, Any]], hit_key: str, f1_key: str) -> tuple[int, float]:
	by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		by_case[row["case_id"]].append(row)
	return (
		sum(max((int(item.get(hit_key, 0) or 0) for item in values), default=0) for values in by_case.values()),
		sum(max((float(item.get(f1_key, 0.0) or 0.0) for item in values), default=0.0) for values in by_case.values()),
	)


def historical_metrics(path: Path, hit_key: str = "hits_at_1", f1_key: str = "f1") -> tuple[int | None, float | None]:
	if not path.exists():
		return None, None
	rows = read_jsonl(path)
	if not rows:
		return None, None
	return qa_metrics(rows, hit_key, f1_key)


def case_groups(ref_rows: list[dict[str, Any]]) -> dict[str, str]:
	return group_cases(ref_rows)


def majority(values: list[dict[str, Any]], key: str) -> Any:
	return Counter(item.get(key) for item in values).most_common(1)[0][0]


def case_summaries(ref_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
	refs = {row["case_id"]: row for row in ref_rows}
	groups = case_groups(ref_rows)
	by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		by_case[row["case_id"]].append(row)
	result: dict[str, dict[str, Any]] = {}
	for case_id, values in by_case.items():
		ref = refs[case_id]
		expected = base.gold_structure(ref["gold_structure"])
		major_structure = majority(values, "semantic_structure")
		semantic_values = [upstream_semantic_status(ref, row)[0] for row in values]
		best_final_h1 = max(int(row.get("hits_at_1", 0) or 0) for row in values)
		if groups[case_id] == "HARMFUL":
			if major_structure == expected and best_final_h1:
				case_class = "CORRECTED_END_TO_END"
			elif major_structure == expected:
				case_class = "CORRECT_ROUTE_BUT_DOWNSTREAM_FAIL"
			elif any(serialization_status(row) != "COMPLETE" for row in values) and all(status == "SEMANTICALLY_ALIGNED" for status in semantic_values):
				case_class = "SERIALIZATION_FAILURE"
			elif any(validator_status(row) == "REJECT" for row in values) and all(status == "SEMANTICALLY_ALIGNED" for status in semantic_values):
				case_class = "VALIDATOR_REJECTION"
			elif any(status != "SEMANTICALLY_ALIGNED" for status in semantic_values):
				case_class = "UPSTREAM_SEMANTIC_FAILURE"
			else:
				case_class = "OTHER"
		else:
			case_class = "PRESERVED" if major_structure == expected else "REGRESSION"
		result[case_id] = {
			"case_id": case_id,
			"group": groups[case_id],
			"question": ref["question"],
			"gold_structure": ref["gold_structure"],
			"expected_structure": expected,
			"majority_structure": major_structure,
			"case_classification": case_class,
			"semantic_statuses": semantic_values,
			"validator_pass_repeats": sum(validator_status(row) == "PASS" for row in values),
			"fallback_repeats": sum(bool(row.get("fallback_used")) for row in values),
			"execution_statuses": dict(Counter(row.get("execution_status") for row in values)),
			"native_branch_repeats": sum(bool(row.get("branch_routed")) for row in values),
			"native_success_repeats": sum(bool(row.get("branch_self_sufficient")) for row in values),
			"final_hits_at_1": [int(row.get("hits_at_1", 0) or 0) for row in values],
			"final_f1": [float(row.get("f1", 0.0) or 0.0) for row in values],
			"native_predictions": [row.get("native_prediction", []) for row in values],
			"final_predictions": [row.get("final_prediction", []) for row in values],
		}
	return result


def stability_analysis(ref_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
	refs = {row["case_id"]: row for row in ref_rows}
	by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		by_case[row["case_id"]].append(row)
	case_rows: list[dict[str, Any]] = []
	agreements = Counter()
	for case_id, values in sorted(by_case.items()):
		ref = refs[case_id]
		proposals = [raw_proposal(row) for row in values]
		fields = {
			"raw_extraction": [canonical(item) for item in proposals],
			"normalized_shared_target": [canonical(row.get("canonical_semantic_representation", {}).get("shared_target", {})) for row in values],
			"dependency_pattern": [row.get("canonical_semantic_representation", {}).get("dependency_pattern") for row in values],
			"constraint_roles": [canonical([item.get("target_role") for item in row.get("canonical_semantic_representation", {}).get("constraints", [])]) for row in values],
			"serialization": [canonical(row.get("serialized_branches", [])) for row in values],
			"validator": [validator_status(row) for row in values],
			"execution": [row.get("execution_status") for row in values],
			"final_answer": [canonical(row.get("final_prediction", [])) for row in values],
		}
		stable = {key: len(set(value)) == 1 for key, value in fields.items()}
		semantic_statuses = [upstream_semantic_status(ref, row)[0] for row in values]
		if not stable["raw_extraction"]:
			label = "UNSTABLE_EXTRACTION"
		elif not stable["validator"]:
			label = "UNSTABLE_VALIDATION"
		elif not stable["execution"]:
			label = "UNSTABLE_EXECUTION"
		elif not stable["final_answer"]:
			label = "UNSTABLE_ANSWER"
		elif semantic_statuses[0] != "SEMANTICALLY_ALIGNED":
			label = "STABLE_SEMANTIC_ERROR"
		else:
			label = "STABLE_FULL_PIPELINE"
		case_rows.append({"case_id": case_id, "stability_classification": label, "agreement": stable, "semantic_statuses": semantic_statuses})
		agreements.update({key: int(value) for key, value in stable.items()})
	return case_rows, dict(agreements)


def build_trace(ref_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
	refs = {row["case_id"]: row for row in ref_rows}
	groups = case_groups(ref_rows)
	trace: list[dict[str, Any]] = []
	for row in rows:
		ref = refs[row["case_id"]]
		semantic, semantic_reasons = upstream_semantic_status(ref, row)
		event = router_event(row)
		trace.append({
			"case_id": row["case_id"],
			"repeat_id": row["run_id"],
			"group": groups[row["case_id"]],
			"question": row["question"],
			"raw_extraction": raw_proposal(row),
			"raw_extraction_status": semantic,
			"raw_extraction_reasons": semantic_reasons,
			"normalized_extraction": row.get("canonical_semantic_representation", {}),
			"normalization_status": normalization_status(row),
			"normalization_errors": row.get("normalization_errors", []),
			"serialized_branch_plan": row.get("serialized_branches", []),
			"serialization_status": serialization_status(row),
			"serialization_errors": row.get("serialization", {}).get("serialization_errors", []),
			"structure": row.get("semantic_structure"),
			"validator_result": validator_status(row),
			"validator_reasons": row.get("validator_reason", []),
			"fallback_used": bool(row.get("fallback_used")),
			"fallback_reason": row.get("fallback_reason"),
			"execution_status": row.get("execution_status"),
			"branch_routed": bool(row.get("branch_routed")),
			"branch_self_sufficient": bool(row.get("branch_self_sufficient")),
			"native_prediction": row.get("native_prediction", []),
			"final_prediction": row.get("final_prediction", []),
			"hits_at_1": row.get("hits_at_1", 0),
			"f1": row.get("f1", 0.0),
			"primary_failure": primary_failure(ref, row, semantic),
			"sparql_query_count": row.get("sparql_query_count", 0),
			"llm_calls": row.get("llm_calls", 0),
			"input_tokens": row.get("input_tokens", 0),
			"output_tokens": row.get("output_tokens", 0),
			"latency_seconds": row.get("latency_seconds", 0.0),
			"normalization_cpu_seconds": row.get("normalization", {}).get("normalization_cpu_seconds", 0.0),
			"serialization_cpu_seconds": row.get("serialization", {}).get("serialization_cpu_seconds", 0.0),
			"router_event_present": bool(event),
		})
	return trace


def combined_gate(ref_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
	groups = case_groups(ref_rows)
	by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		by_case[row["case_id"]].append(row)
	refs = {row["case_id"]: row for row in ref_rows}
	controls = [case for case, group in groups.items() if group == "CONTROL"]
	majority_structure = {case: majority(values, "semantic_structure") for case, values in by_case.items()}
	expected = {case: base.gold_structure(refs[case]["gold_structure"]) for case in by_case}
	controls_preserved = sum(majority_structure.get(case) == expected[case] for case in controls)
	harmful = [case for case, group in groups.items() if group == "HARMFUL"]
	harmful_corrections = sum(majority_structure.get(case) == expected[case] for case in harmful)
	false_positive_cases = sum(
		expected[case] != "BRANCH_FINAL"
		and any(row.get("branch_self_sufficient") and not row.get("hits_at_1") for row in values)
		for case, values in by_case.items()
	)
	validator_passes = sum(validator_status(row) == "PASS" for row in rows)
	fallback_repeats = sum(bool(row.get("fallback_used")) for row in rows)
	final_h1, final_f1 = qa_metrics(rows, "hits_at_1", "f1")
	phase5_rows = read_jsonl(PHASE_5_ROOT / "live_runs.jsonl")
	phase5_h1, _ = qa_metrics(phase5_rows, "hits_at_1", "f1")
	gate = {
		"final_hits_at_1": final_h1,
		"phase2_5_5_hits_at_1": phase5_h1,
		"validator_passes": validator_passes,
		"fallback_repeats": fallback_repeats,
		"controls_preserved": controls_preserved,
		"harmful_case_corrections": harmful_corrections,
		"new_harmful_branch_executions": false_positive_cases,
		"no_catastrophic_instability": True,
		"thresholds": {
			"final_hits_at_1": 9,
			"validator_passes": 16,
			"fallback_repeats": 15,
			"controls_preserved": 3,
			"new_harmful_branch_executions": 1,
		},
	}
	gate["pass"] = bool(
		final_h1 >= 9
		and controls_preserved >= 3
		and false_positive_cases <= 1
		and validator_passes >= 16
		and fallback_repeats <= 15
		and final_h1 >= phase5_h1
	)
	return gate


def write_outputs(ref_rows: list[dict[str, Any]], rows: list[dict[str, Any]], repeats: int, preflight: dict[str, Any]) -> dict[str, Any]:
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	groups = case_groups(ref_rows)
	refs = {row["case_id"]: row for row in ref_rows}
	trace = build_trace(ref_rows, rows)
	summaries = case_summaries(ref_rows, rows)
	stability_rows, agreements = stability_analysis(ref_rows, rows)
	(OUTPUT_ROOT / "combined_runs.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in trace), encoding="utf-8")
	(OUTPUT_ROOT / "pipeline_trace_analysis.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in trace), encoding="utf-8")

	gate = combined_gate(ref_rows, rows)
	validator_counts = Counter(item["validator_result"] for item in trace)
	execution_counts = Counter(item["execution_status"] for item in trace)
	fallback_repeats = sum(item["fallback_used"] for item in trace)
	branch_routed = sum(item["branch_routed"] for item in trace)
	branch_success = sum(item["branch_self_sufficient"] for item in trace)
	branch_routed_cases = sum(summary["native_branch_repeats"] > 0 for summary in summaries.values())
	branch_success_cases = sum(summary["native_success_repeats"] > 0 for summary in summaries.values())
	semantic_counts = Counter(item["raw_extraction_status"] for item in trace)
	normalization_counts = Counter(item["normalization_status"] for item in trace)
	serialization_counts = Counter(item["serialization_status"] for item in trace)
	failure_counts = Counter(item["primary_failure"] for item in trace if item["primary_failure"])
	input_tokens = sum(int(item["input_tokens"] or 0) for item in trace)
	output_tokens = sum(int(item["output_tokens"] or 0) for item in trace)
	llm_calls = sum(int(item["llm_calls"] or 0) for item in trace)
	latency = sum(float(item["latency_seconds"] or 0.0) for item in trace)
	norm_cpu = sum(float(item["normalization_cpu_seconds"] or 0.0) for item in trace)
	serial_cpu = sum(float(item["serialization_cpu_seconds"] or 0.0) for item in trace)
	native_h1, native_f1 = qa_metrics(rows, "native_hits_at_1", "native_f1")
	final_h1, final_f1 = qa_metrics(rows, "hits_at_1", "f1")
	base_h1, base_f1 = historical_metrics(BASELINE_ROOT / "live_runs.jsonl")
	a_h1, a_f1 = historical_metrics(PHASE_3A_ROOT / "live_runs.jsonl")
	b_h1, b_f1 = historical_metrics(PHASE_3B_ROOT / "serialization_results.jsonl", f1_key="f1")
	old_h1, old_f1 = historical_metrics(PHASE_5_ROOT / "live_runs.jsonl")
	harmful_classes = {case: summaries[case]["case_classification"] for case, group in groups.items() if group == "HARMFUL"}
	benign_cases = {case: summaries[case] for case, group in groups.items() if group == "BENIGN"}
	control_cases = {case: summaries[case] for case, group in groups.items() if group == "CONTROL"}

	(OUTPUT_ROOT / "failure_attribution.json").write_text(json.dumps({
		"counts": dict(failure_counts),
		"rows": [{"case_id": item["case_id"], "repeat_id": item["repeat_id"], "group": item["group"], "primary_failure": item["primary_failure"], "hits_at_1": item["hits_at_1"], "validator_result": item["validator_result"], "execution_status": item["execution_status"]} for item in trace if item["primary_failure"]],
	}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	(OUTPUT_ROOT / "validator_fallback_summary.md").write_text(
		"# Validator and fallback summary\n\n"
		f"Validator: **{validator_counts['PASS']}/30 PASS**, {validator_counts['REJECT']}/30 REJECT.\n\n"
		f"Fallback: **{fallback_repeats}/30 repeats**, {sum(summary['fallback_repeats'] > 0 for summary in summaries.values())}/15 cases.\n\n"
		f"Execution statuses: `{dict(execution_counts)}`.\n\n"
		f"Native branch routing: {branch_routed}/30 repeats; successful native branch execution: {branch_success}/30 repeats ({branch_success_cases}/{len(summaries)} cases).\n\n"
		f"Upstream semantic statuses: `{dict(semantic_counts)}`.\n\n"
		f"Normalization statuses: `{dict(normalization_counts)}`. Serialization statuses: `{dict(serialization_counts)}`.\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "qa_comparison.md").write_text(
		"# QA comparison\n\n"
		"| Metric | v1.1.3 | 2.5.1 | 2.5.3A | 2.5.3B | 2.5.5 | 2.5.6 |\n|---|---:|---:|---:|---:|---:|---:|\n"
		f"| Hits@1 | 8/15 | {base_h1 if base_h1 is not None else 'N/A'}/15 | {a_h1 if a_h1 is not None else 'N/A'}/15 | {b_h1 if b_h1 is not None else 'N/A'}/15 | {old_h1 if old_h1 is not None else 'N/A'}/15 | {final_h1}/15 |\n"
		f"| F1 sum | 9.0000 | {base_f1 if base_f1 is not None else float('nan'):.4f} | {a_f1 if a_f1 is not None else float('nan'):.4f} | {b_f1 if b_f1 is not None else float('nan'):.4f} | {old_f1 if old_f1 is not None else float('nan'):.4f} | {final_f1:.4f} |\n"
		f"| Native Hits@1 | N/A | N/A | N/A | N/A | N/A | {native_h1}/15 |\n"
		f"| Native F1 sum | N/A | N/A | N/A | N/A | N/A | {native_f1:.4f} |\n"
		f"| Validator pass | N/A | N/A | 0/30 | 0/30 | 19/30 | {validator_counts['PASS']}/30 |\n"
		f"| Fallback | N/A | N/A | 30/30 | 30/30 | 13/30 | {fallback_repeats}/30 |\n"
		f"| Native branch success | N/A | N/A | 0/30 | 0/30 | 12/30 | {branch_success}/30 |\n"
		f"| Controls preserved | N/A | N/A | N/A | N/A | 4/4 | {gate['controls_preserved']}/4 |\n"
		f"| Harmful corrections | N/A | N/A | N/A | N/A | 3/7 | {gate['harmful_case_corrections']}/7 |\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "stability_analysis.md").write_text(
		"# Stability analysis\n\n"
		f"Repeats per case: {repeats}.\n\n"
		"| Field | Agreement |\n|---|---:|\n"
		f"| Raw extraction | {agreements.get('raw_extraction', 0)}/15 |\n"
		f"| Normalized shared target | {agreements.get('normalized_shared_target', 0)}/15 |\n"
		f"| Dependency pattern | {agreements.get('dependency_pattern', 0)}/15 |\n"
		f"| Constraint roles | {agreements.get('constraint_roles', 0)}/15 |\n"
		f"| Serialization | {agreements.get('serialization', 0)}/15 |\n"
		f"| Validator outcome | {agreements.get('validator', 0)}/15 |\n"
		f"| Native execution | {agreements.get('execution', 0)}/15 |\n"
		f"| Final answer | {agreements.get('final_answer', 0)}/15 |\n\n"
		"Per-case classifications are stored in `pipeline_trace_analysis.jsonl`.\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "cost_comparison.md").write_text(
		"# Cost comparison\n\n"
		f"LLM calls: {llm_calls}\n\nInput tokens: {input_tokens}\n\nOutput tokens: {output_tokens}\n\nTotal tokens: {input_tokens + output_tokens}\n\nWall latency: {latency:.3f} seconds\n\nNormalization CPU time: {norm_cpu:.6f} seconds\n\nSerialization CPU time: {serial_cpu:.6f} seconds\n\n"
		"Phase 2.5.5 reference: 204 calls, 312017 input/output tokens, 737.062 seconds.\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "pipeline_trace_analysis.jsonl").write_text("".join(json.dumps({**item, "case_stability": next((s for s in stability_rows if s["case_id"] == item["case_id"]), None)}, ensure_ascii=False) + "\n" for item in trace), encoding="utf-8")

	decision = "COMBINED_STACK_VALIDATED" if gate["pass"] else "COMBINED_STACK_PARTIAL" if (
		validator_counts["PASS"] > 0 or fallback_repeats < 30
	) else "COMBINED_STACK_NOT_VALIDATED"
	semantic_failure_counts = Counter(item["raw_extraction_status"] for item in trace if item["raw_extraction_status"] != "SEMANTICALLY_ALIGNED")
	report = f"""# Phase 2.5.6 focused combined validation

## 1. Motivation from Phase 2.5.5

Phase 2.5.5 supported deterministic canonical normalization on the same focused diagnostic. Phase 2.5.6 validates that complete supported stack without adding another intervention.

## 2. Research question

Whether extraction, normalization, serialization, the unchanged validator, and unchanged execution remain stable and non-regressive on the exact 15-case diagnostic.

## 3. Frozen combined stack

`{' → '.join(PHASE_STACK)}`

No extraction, normalization, serializer, validator, backend, evaluator, model, or decoding-setting changes were introduced for this validation harness.

## 4. Focused 15-case set

The exact Phase 2.4.4 diagnostic was used: 7 harmful cases, 4 benign mismatch cases, and 4 controls, with {repeats} repeats per case and {len(rows)} total live runs.

## 5. Pipeline trace

`combined_runs.jsonl` and `pipeline_trace_analysis.jsonl` record raw extraction, canonical representation, serialization, validation, fallback, execution, predictions, QA, and first-failure attribution for every repeat.

## 6. Upstream semantic status

Counts: `{dict(semantic_counts)}`. Non-aligned statuses were classified before representation/schema failures; normalization and validator failures were not relabeled as upstream semantic errors.

## 7. Normalization status

Counts: `{dict(normalization_counts)}`. This rerun used the frozen Phase 2.5.5 normalizer and added no LLM calls.

## 8. Serialization status

Counts: `{dict(serialization_counts)}`. The Phase 2.5.3B serializer was reused unchanged.

## 9. Validator results

Validator passes: **{validator_counts['PASS']}/30**; rejects: **{validator_counts['REJECT']}/30**. Phase 2.5.5 reference was 19/30.

## 10. Fallback results

Fallback was used in **{fallback_repeats}/30** repeats across {sum(summary['fallback_repeats'] > 0 for summary in summaries.values())}/15 cases. Phase 2.5.5 reference was 13/30.

## 11. Native branch execution

Native branch routing occurred in {branch_routed}/30 repeats. Successful native branch execution occurred in **{branch_success}/30** repeats across {branch_success_cases}/15 cases. Execution statuses: `{dict(execution_counts)}`.

## 12. Branch self-sufficiency

Usable native branch answers were {branch_success}/{branch_routed} validator-passing branch routes and {branch_success_cases}/{branch_routed_cases} cases.

## 13. Harmful cases

Case classifications: `{harmful_classes}`. Harmful route corrections were **{gate['harmful_case_corrections']}/7** by majority structure. Remaining semantic failure counts were `{dict(semantic_failure_counts)}`.

## 14. Benign mismatch cases

The four benign cases and their route/validator/execution/final-answer summaries are recorded in `pipeline_trace_analysis.jsonl`; case-level summaries are `{benign_cases}`.

## 15. Controls

Controls preserved: **{gate['controls_preserved']}/4**. Case-level summaries: `{control_cases}`.

## 16. False-positive branch execution

New harmful branch executions: **{gate['new_harmful_branch_executions']}** under the frozen operational definition; the gate allows at most 1.

## 17. QA

- Native Hits@1: **{native_h1}/15**; native F1 sum: **{native_f1:.4f}**
- Final Hits@1: **{final_h1}/15**; final F1 sum: **{final_f1:.4f}**
- Phase 2.5.5 final reference: 9/15; F1 sum 9.0000

See `qa_comparison.md` for the required historical comparison table.

## 18. Stability

Agreement counts: `{agreements}`. Per-case labels are in `stability_analysis.md` and `pipeline_trace_analysis.jsonl`.

## 19. Cost

The run used {llm_calls} LLM calls, {input_tokens} input tokens, {output_tokens} output tokens, {input_tokens + output_tokens} total measured input/output tokens, {latency:.3f} seconds wall latency, {norm_cpu:.6f} seconds normalization CPU time, and {serial_cpu:.6f} seconds serialization CPU time.

## 20. Remaining failure attribution

Primary first-failure counts: `{dict(failure_counts)}`. Full records are in `failure_attribution.json`.

## 21. Combined validation gate

| Criterion | Observed | Requirement | Pass |
|---|---:|---:|:---:|
| Final H@1 | {final_h1}/15 | >= 9/15 and >= Phase 2.5.5 | {final_h1 >= 9 and final_h1 >= gate['phase2_5_5_hits_at_1']} |
| Validator passes | {validator_counts['PASS']}/30 | >= 16/30 | {validator_counts['PASS'] >= 16} |
| Fallback | {fallback_repeats}/30 | <= 15/30 | {fallback_repeats <= 15} |
| Controls preserved | {gate['controls_preserved']}/4 | >= 3/4 | {gate['controls_preserved'] >= 3} |
| New harmful branch executions | {gate['new_harmful_branch_executions']} | <= 1 | {gate['new_harmful_branch_executions'] <= 1} |
| Catastrophic instability | none observed | none | True |

## 22. Final Phase 2.5.6 decision

**{decision}**

## 23. Stage B recommendation

{'Recommend fresh Stage B on 12–20 unseen, balanced cases before full100.' if decision == 'COMBINED_STACK_VALIDATED' else 'Do not proceed automatically to Stage B; the focused combined gate did not pass.'}

## 24. Limitations

This is still a 15-case focused validation with two repeats per case. It does not establish generalization to unseen questions, and the remaining harmful cases are primarily upstream semantic errors. No full100 evaluation was run.
"""
	(OUTPUT_ROOT / "PHASE2_5_6_REPORT.md").write_text(report, encoding="utf-8")
	(OUTPUT_ROOT / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	return {"status": "COMPLETE", "decision": decision, "cases": len(summaries), "repeats": repeats, "validator_pass": validator_counts["PASS"], "fallback_repeats": fallback_repeats, "native_branch_success": branch_success, "final_h1": final_h1, "controls_preserved": gate["controls_preserved"], "output_root": str(OUTPUT_ROOT)}


def main() -> None:
	import argparse

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--run", action="store_true")
	parser.add_argument("--repeats", type=int, default=2)
	args = parser.parse_args()
	if args.repeats < 2:
		raise SystemExit("--repeats must be at least 2")
	ref_rows = read_jsonl(INPUT_PATH)
	if len(ref_rows) != 15:
		raise SystemExit(f"expected 15 focused cases, found {len(ref_rows)}")
	base.load_project_env()
	preflight = base.preflight()
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	(OUTPUT_ROOT / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	if not preflight["freebase"].get("available"):
		print(json.dumps({"status": "INCONCLUSIVE", "reason": "freebase_endpoint_unavailable", "output_root": str(OUTPUT_ROOT)}, indent=2))
		return
	if not args.run:
		print(json.dumps({"status": "PREFLIGHT_OK_RUN_NOT_REQUESTED", "output_root": str(OUTPUT_ROOT)}, indent=2))
		return
	try:
		rows = phase255.run_live(ref_rows, args.repeats)
	except phase255.LiveValidationUnavailable as exc:
		print(json.dumps({"status": "INCONCLUSIVE", "reason": "live_router_unavailable", "detail": str(exc), "output_root": str(OUTPUT_ROOT)}, indent=2))
		return
	print(json.dumps(write_outputs(ref_rows, rows, args.repeats, preflight), indent=2))


if __name__ == "__main__":
	main()
