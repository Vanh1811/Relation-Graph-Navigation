#!/usr/bin/env python3
"""Run the focused Phase 2.5.3B branch-plan compatibility evaluation."""

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
from chain_of_relations.methods.phase2_5_3b.agent import BranchPlanSerializationAgent  # noqa: E402
from chain_of_relations.schema import Entity  # noqa: E402


INPUT_PATH = ROOT / "outputs/experiments/phase2_4_4/routing_decision_error_analysis/router_input_output_analysis.jsonl"
BASELINE_ROOT = ROOT / "outputs/experiments/phase2_5_1/live_constraint_first_validation"
PHASE_3A_ROOT = ROOT / "outputs/experiments/phase2_5_3a/source_target_extraction"
OUTPUT_ROOT = ROOT / "outputs/experiments/phase2_5_3b/branch_plan_serialization"


class LiveValidationUnavailable(RuntimeError):
	pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_agent() -> BranchPlanSerializationAgent:
	from chain_of_relations.kg_backend import get_backend
	from chain_of_relations.methods.cor.agent import CoRAgent

	cor = CoRAgent(
		model_name=os.environ["MODEL_NAME"],
		relation_width=3,
		depth=3,
		sample_relation_threshold=1000,
		remove_unnecessary_rel=True,
		temperature_exploration=0.3,
		temperature_reasoning=0.1,
		max_token=1024,
		backend=get_backend("freebase"),
	)
	return BranchPlanSerializationAgent(cor, confidence_threshold=0.75, max_branches=3)


def classify_validator_failures(failures: list[str]) -> list[str]:
	labels: list[str] = []
	for failure in failures:
		text = str(failure).casefold()
		if "root is not a linked" in text or "root_entity" in text:
			label = "MISSING_ROOT_ENTITY"
		elif "constraint_property_mismatch" in text:
			label = "CONSTRAINT_PROPERTY_MISMATCH"
		elif "no constraint_property" in text or "empty semantic_property" in text:
			label = "MISSING_CONSTRAINT_PROPERTY"
		elif "target_answer_type" in text or "answer_type" in text or "target type" in text:
			label = "MISSING_OR_INCOMPATIBLE_TARGET_TYPE"
		elif "shared_target" in text:
			label = "INVALID_SHARED_TARGET"
		elif "branch_count" in text or "number of branch" in text or "at least two branch" in text:
			label = "INVALID_BRANCH_COUNT"
		elif "type-only" in text or "generic/type-like" in text:
			label = "TYPE_ONLY_BRANCH"
		else:
			label = "OTHER"
		if label not in labels:
			labels.append(label)
	return labels or ["OTHER"]


def execution_status(result: dict[str, Any], validator_passed: bool) -> str:
	if not validator_passed:
		return "NOT_ATTEMPTED_VALIDATOR_REJECT"
	native = result.get("adaptive_native_result") if isinstance(result.get("adaptive_native_result"), dict) else result
	if result.get("fallback_used"):
		if native.get("branch_failures"):
			return "MERGE_FAILURE"
		return "OTHER_EXECUTION_FAILURE"
	if native.get("branch_failures"):
		return "MERGE_FAILURE"
	counts = native.get("branch_candidate_counts", [])
	if isinstance(counts, list) and counts and any(int(value or 0) <= 0 for value in counts):
		return "EMPTY_BRANCH_RESULT"
	if int(native.get("merged_candidate_count", 0) or 0) > 0 or native.get("results"):
		return "EXECUTION_SUCCESS"
	if native.get("merge_outcome") == "NO_INTERSECTION_VALID_DISAGREEMENT":
		return "MERGE_FAILURE"
	return "OTHER_EXECUTION_FAILURE"


def run_live(rows: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
	agent = make_agent()
	output: list[dict[str, Any]] = []
	for row in rows:
		entities = [
			Entity(str(item.get("id", "")), str(item.get("name", "")))
			for item in row.get("router_input", {}).get("runtime_topic_entities", [])
		]
		gold_answers = base.case_gold_answers(row)
		for run_id in range(1, repeats + 1):
			history: list[dict[str, Any]] = []
			sparql_history: list[dict[str, Any]] = []
			step_history: list[dict[str, Any]] = []
			started = time.perf_counter()
			error = None
			try:
				result = agent.answer(
					question=str(row["question"]),
					topic_entities=entities,
					prompt_history=history,
					sparql_history=sparql_history,
					step_history=step_history,
				)
			except Exception as exc:
				result = {"action": "ERROR", "results": []}
				error = f"{type(exc).__name__}: {exc}"
			router_events = [item for item in history if item.get("type") == "phase2_5_3b_branch_plan_serialization_router"]
			if not router_events or not router_events[-1].get("response") or not router_events[-1].get("parsed_result"):
				router_error = (router_events[-1].get("llm_error") if router_events else "") or "empty or invalid live router response"
				raise LiveValidationUnavailable(
					f"live Phase 2.5.3B router unavailable for {row['case_id']} run={run_id}: {router_error}"
				)
			latency = time.perf_counter() - started
			event = router_events[-1]
			decision = result.get("router_decision", result)
			decision = decision if isinstance(decision, dict) else {}
			validator_passed = str(event.get("validator_decision", "")).upper() == "PASS"
			native = result.get("adaptive_native_result") if isinstance(result.get("adaptive_native_result"), dict) else result
			prediction = base.prediction_names(result)
			native_prediction = base.prediction_names(native)
			hit, f1 = base.evaluate(prediction, gold_answers)
			native_hit, native_f1 = base.evaluate(native_prediction, gold_answers)
			serialization = event.get("serialization", {})
			failures = event.get("validation_failures", decision.get("validation_failures", []))
			output.append({
				"case_id": row["case_id"],
				"run_id": run_id,
				"question": row["question"],
				"recognized_entities": [{"id": e.id, "name": e.name} for e in entities],
				"semantic_structure": decision.get("router_structure", "UNKNOWN"),
				"semantic_constraints": event.get("constraints", decision.get("constraints", [])),
				"independent_source_groups": event.get("independent_source_groups", []),
				"shared_target": decision.get("shared_target", {}),
				"dependency_pattern": decision.get("dependency_pattern", "UNKNOWN"),
				"serialized_branches": list(serialization.get("serialized_branches", [])),
				"serialization_status": serialization.get("serialization_status", "FAILED"),
				"serialization_errors": list(serialization.get("serialization_errors", [])),
				"serialization_scope": serialization.get("serialization_scope", "UNKNOWN"),
				"serialization_field_provenance": serialization.get("field_provenance", {}),
				"serialization_cpu_seconds": float(serialization.get("serialization_cpu_seconds", 0.0) or 0.0),
				"validator_result": "VALIDATOR_PASS" if validator_passed else "VALIDATOR_REJECT",
				"validator_reason": list(failures),
				"validator_reason_classes": classify_validator_failures(list(failures)),
				"fallback_used": bool(result.get("fallback_used", False)),
				"fallback_reason": result.get("fallback_reason"),
				"execution_status": execution_status(result, validator_passed),
				"branch_routed": validator_passed,
				"branch_self_sufficient": validator_passed and execution_status(result, validator_passed) == "EXECUTION_SUCCESS",
				"native_prediction": native_prediction,
				"final_prediction": prediction,
				"native_hits_at_1": native_hit,
				"native_f1": native_f1,
				"hits_at_1": hit,
				"f1": f1,
				"input_tokens": sum(int(item.get("input_tokens", 0) or 0) for item in history),
				"output_tokens": sum(int(item.get("output_tokens", 0) or 0) for item in history),
				"total_tokens": sum(int(item.get("total_tokens", 0) or 0) for item in history),
				"llm_calls": len(history),
				"latency_seconds": latency,
				"error": error,
				"prompt_history": history,
				"sparql_query_count": len(sparql_history),
			})
	return output


def canonical(value: Any) -> str:
	return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def case_majority(items: list[dict[str, Any]], field: str) -> Any:
	return Counter(item.get(field) for item in items).most_common(1)[0][0]


def group_cases(ref_rows: list[dict[str, Any]]) -> dict[str, str]:
	return {
		row["case_id"]: (
			"HARMFUL" if row["routing_outcome"] == "HARMFUL_ROUTING_MISMATCH"
			else "BENIGN" if row["routing_outcome"] == "BENIGN_ROUTING_MISMATCH"
			else "CONTROL"
		)
		for row in ref_rows
	}


def qa_case_metrics(rows: list[dict[str, Any]]) -> tuple[int, float]:
	by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		by_case[row["case_id"]].append(row)
	return (
		sum(max((int(item.get("hits_at_1", 0) or 0) for item in values), default=0) for values in by_case.values()),
		sum(max((float(item.get("f1", 0.0) or 0.0) for item in values), default=0.0) for values in by_case.values()),
	)


def build_outputs(ref_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]], repeats: int, preflight: dict[str, Any]) -> str:
	refs = {row["case_id"]: row for row in ref_rows}
	baseline = read_jsonl(BASELINE_ROOT / "live_runs.jsonl")
	phase_3a = read_jsonl(PHASE_3A_ROOT / "live_runs.jsonl")
	base_by_key = {(row["case_id"], row["run_id"]): row for row in baseline}
	new_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for item in new_rows:
		new_by_case[item["case_id"]].append(item)
	groups = group_cases(ref_rows)
	gold_by_case = {row["case_id"]: base.gold_structure(row["gold_structure"]) for row in ref_rows}
	majority_structure = {case_id: case_majority(items, "semantic_structure") for case_id, items in new_by_case.items()}
	harmful = [case_id for case_id, group in groups.items() if group == "HARMFUL"]
	controls = [case_id for case_id, group in groups.items() if group == "CONTROL"]
	corrected_harmful = sum(majority_structure.get(case_id) == gold_by_case[case_id] for case_id in harmful)
	controls_preserved = sum(majority_structure.get(case_id) == gold_by_case[case_id] for case_id in controls)
	false_positive = sum(majority_structure.get(case_id) in {"BRANCH_FINAL", "BRANCH_INTERMEDIATE"} and gold_by_case[case_id] == "LINEAR" for case_id in groups)

	serialization_counts = Counter(row.get("serialization_status") for row in new_rows)
	validator_counts = Counter(row.get("validator_result") for row in new_rows)
	fallback_repeats = sum(bool(row.get("fallback_used")) for row in new_rows)
	fallback_cases = sum(any(row.get("fallback_used") for row in values) for values in new_by_case.values())
	validator_reason_counts = Counter(
		label for row in new_rows for label in row.get("validator_reason_classes", [])
		if row.get("validator_result") == "VALIDATOR_REJECT"
	)
	execution_counts = Counter(row.get("execution_status") for row in new_rows)
	branch_routed_repeats = sum(bool(row.get("branch_routed")) for row in new_rows)
	branch_success_repeats = sum(bool(row.get("branch_self_sufficient")) for row in new_rows)
	branch_routed_cases = sum(any(row.get("branch_routed") for row in values) for values in new_by_case.values())
	branch_success_cases = sum(any(row.get("branch_self_sufficient") for row in values) for values in new_by_case.values())
	serialization_stable = sum(len({canonical(row.get("serialized_branches", [])) for row in values}) == 1 for values in new_by_case.values())
	status_stable = sum(len({row.get("serialization_status") for row in values}) == 1 for values in new_by_case.values())
	validator_stable = sum(len({row.get("validator_result") for row in values}) == 1 for values in new_by_case.values())
	answer_stable = sum(len({(row.get("hits_at_1"), round(float(row.get("f1", 0.0)), 6)) for row in values}) == 1 for values in new_by_case.values())

	baseline_h1, baseline_f1 = qa_case_metrics(baseline)
	phase_3a_h1, phase_3a_f1 = qa_case_metrics(phase_3a)
	new_h1, new_f1 = qa_case_metrics(new_rows)
	input_tokens = sum(int(row.get("input_tokens", 0) or 0) for row in new_rows)
	output_tokens = sum(int(row.get("output_tokens", 0) or 0) for row in new_rows)
	latency = sum(float(row.get("latency_seconds", 0.0) or 0.0) for row in new_rows)
	serialization_cpu = sum(float(row.get("serialization_cpu_seconds", 0.0) or 0.0) for row in new_rows)
	llm_calls = sum(int(row.get("llm_calls", 0) or 0) for row in new_rows)

	comparison_rows: list[dict[str, Any]] = []
	for row in new_rows:
		old = base_by_key.get((row["case_id"], row["run_id"]), {})
		comparison_rows.append({
			"case_id": row["case_id"],
			"repeat_id": row["run_id"],
			"question": row["question"],
			"semantic_structure": row["semantic_structure"],
			"baseline_structure": old.get("structure", "UNKNOWN"),
			"serialized_branches": row["serialized_branches"],
			"serialization_status": row["serialization_status"],
			"serialization_errors": row["serialization_errors"],
			"validator_result": row["validator_result"],
			"validator_reason": row["validator_reason"],
			"validator_reason_classes": row["validator_reason_classes"],
			"fallback_used": row["fallback_used"],
			"execution_status": row["execution_status"],
			"branch_self_sufficient": row["branch_self_sufficient"],
			"native_prediction": row["native_prediction"],
			"final_prediction": row["final_prediction"],
			"native_hits_at_1": row["native_hits_at_1"],
			"hits_at_1": row["hits_at_1"],
			"f1": row["f1"],
			"llm_calls": row["llm_calls"],
			"tokens": row["input_tokens"] + row["output_tokens"],
			"latency_seconds": row["latency_seconds"],
		})
	(OUTPUT_ROOT / "serialization_results.jsonl").write_text(
		"".join(json.dumps(row, ensure_ascii=False) + "\n" for row in comparison_rows),
		encoding="utf-8",
	)

	problem_ids = {
		json.loads(line)["case_id"]
		for line in (ROOT / "outputs/experiments/phase2_5_2/live_extraction_fallback_analysis/harmful_case_analysis.jsonl").read_text().splitlines()
		if line.strip()
	}
	problem_rows = [row for row in new_rows if row["case_id"] in problem_ids]
	upstream_extraction = sum(
		any(row.get("serialization_status") == "COMPLETE" and row.get("validator_result") == "VALIDATOR_REJECT" for row in values)
		for case_id, values in new_by_case.items() if case_id in problem_ids
	)
	serialization_validator_failure = sum(
		any(row.get("serialization_status") in {"PARTIAL", "FAILED"} for row in values)
		for case_id, values in new_by_case.items() if case_id in problem_ids
	)
	new_harmful_execution = sum(
		majority_structure.get(case_id) != "BRANCH_FINAL"
		and any(row.get("branch_self_sufficient") and not row.get("hits_at_1") for row in values)
		for case_id, values in new_by_case.items()
	)

	(OUTPUT_ROOT / "validator_comparison.md").write_text(
		"# Validator comparison\n\n"
		"| Metric | Phase 2.5.1 | Phase 2.5.3A | Phase 2.5.3B |\n|---|---:|---:|---:|\n"
		f"| Validator rejection repeats | not recorded | 30/30 | {validator_counts.get('VALIDATOR_REJECT', 0)}/30 |\n"
		f"| Validator pass repeats | not recorded | 0/30 | {validator_counts.get('VALIDATOR_PASS', 0)}/30 |\n"
		f"| Complete serialization repeats | not recorded | not applicable | {serialization_counts.get('COMPLETE', 0)}/30 |\n"
		f"| Root/property/target/other rejection classes | — | — | {dict(validator_reason_counts)} |\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "fallback_comparison.md").write_text(
		"# Fallback comparison\n\n"
		"| Metric | Phase 2.5.1 | Phase 2.5.3A | Phase 2.5.3B |\n|---|---:|---:|---:|\n"
		"| Fallback repeats | recorded in source report | 30/30 | "
		f"{fallback_repeats}/30 |\n"
		"| Fallback cases | recorded in source report | 15/15 | "
		f"{fallback_cases}/15 |\n"
		f"| Native branch routed repeats | — | 0/30 | {branch_routed_repeats}/30 |\n"
		f"| Native branch success repeats | — | 0/30 | {branch_success_repeats}/30 |\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "qa_comparison.md").write_text(
		"# QA comparison\n\n"
		"| Metric | Frozen v1.1.3 | Phase 2.5.1 | Phase 2.5.3A | Phase 2.5.3B |\n|---|---:|---:|---:|---:|\n"
		f"| Hits@1 | 8/15 | {baseline_h1}/15 | {phase_3a_h1}/15 | {new_h1}/15 |\n"
		f"| F1 sum | 9.0000 | {baseline_f1:.4f} | {phase_3a_f1:.4f} | {new_f1:.4f} |\n"
		"| Native/final predictions | recorded | recorded | recorded | recorded |\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "stability_analysis.md").write_text(
		"# Serialization stability analysis\n\n"
		f"Repeats per case: {repeats}\n\n"
		"| Metric | Agreement |\n|---|---:|\n"
		f"| Serialized branch plan agreement | {serialization_stable}/15 |\n"
		f"| Serialization status agreement | {status_stable}/15 |\n"
		f"| Validator outcome agreement | {validator_stable}/15 |\n"
		f"| Final answer agreement | {answer_stable}/15 |\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "cost_comparison.md").write_text(
		"# Cost comparison\n\n"
		f"Additional LLM calls caused by serialization: 0\n\n"
		f"Frozen 3A extraction/router calls observed in Phase 2.5.3B: {llm_calls}\n\n"
		f"Input tokens: {input_tokens}\n\nOutput tokens: {output_tokens}\n\nTotal tokens: {input_tokens + output_tokens}\n\n"
		f"Serialization CPU time: {serialization_cpu:.6f} seconds\n\n"
		f"Total wall latency: {latency:.3f} seconds\n\n"
		"Phase 2.5.3A reference: 187 calls, 307,312 tokens, 890.409 seconds.\n",
		encoding="utf-8",
	)

	validator_rejection_reduced = validator_counts.get("VALIDATOR_REJECT", 0) <= 15
	fallback_reduced = fallback_repeats < 30
	qa_non_regressive = new_h1 >= phase_3a_h1
	gate_passed = validator_rejection_reduced and fallback_reduced and controls_preserved >= 3 and new_harmful_execution <= 1 and qa_non_regressive
	material_compatibility = validator_counts.get("VALIDATOR_PASS", 0) > 0 or fallback_repeats < 30
	if gate_passed and material_compatibility:
		decision = "SERIALIZATION_COMPATIBILITY_SUPPORTED"
	elif material_compatibility:
		decision = "SERIALIZATION_COMPATIBILITY_PARTIAL"
	else:
		decision = "SERIALIZATION_COMPATIBILITY_NOT_SUPPORTED"

	case_rows = []
	for case_id, values in new_by_case.items():
		ref = refs[case_id]
		case_rows.append({
			"case_id": case_id,
			"group": groups[case_id],
			"gold_structure": gold_by_case[case_id],
			"route": majority_structure.get(case_id),
			"serialization_statuses": sorted({row.get("serialization_status") for row in values}),
			"validator_results": sorted({row.get("validator_result") for row in values}),
			"execution_statuses": sorted({row.get("execution_status") for row in values}),
			"final_prediction": values[0].get("final_prediction", []),
			"hits_at_1": max(int(row.get("hits_at_1", 0) or 0) for row in values),
			"question": ref["question"],
		})
	(OUTPUT_ROOT / "harmful_control_execution.jsonl").write_text(
		"".join(json.dumps(row, ensure_ascii=False) + "\n" for row in case_rows), encoding="utf-8"
	)

	report = f"""# Phase 2.5.3B branch-plan serialization and validator compatibility

## 1. Motivation from Phase 2.5.2 and 2.5.3A

Phase 2.5.2 identified validator rejection as the dominant fallback trigger. Phase 2.5.3A then produced 30/30 validator rejections and 30/30 fallbacks even when the semantic extraction was often branch-shaped.

## 2. Research question

Can a deterministic adapter serialize the frozen Phase 2.5.3A semantic extraction into the exact existing executable branch schema without changing routing, extraction, validation, execution, or fallback semantics?

## 3. Scope boundary

Only the semantic-extraction-to-executable-branch adapter was added. The Phase 2.5.3A extraction prompt and retry path, structure labels, dependency values, validator, branch executor, relation scoring, entity grounding, backend, SPARQL execution, evaluator, model, settings, and fallback policy were frozen.

## 4. Existing execution contract

See `branch_execution_contract.md`. The validator requires `branch_id`, grounded `root_entity_id`/`root_entity_name`, `constraint_property`, `constraint_summary`, and `target_answer_type`; the executor consumes only validated plans with two or three grounded roots.

## 5. Serialization design

The adapter deterministically orders independent source groups by their first extracted constraint position, generates `branch_1`, `branch_2`, ... identifiers, resolves roots only against existing runtime entities, copies semantic properties and summaries, and uses the already available requested/shared answer type. Missing values are recorded rather than fabricated.

## 6. Root entity mapping

`source_anchor` is matched by existing runtime entity ID or normalized existing name. No new entity linking or relation search was added.

## 7. Constraint property mapping

`semantic_property` is copied directly. Empty or multi-property groups are marked unavailable; no synthetic property is created.

## 8. Target type mapping

The adapter uses `requested_final_answer_type`, then existing `shared_answer_type`, then the frozen router's existing lexical safety hint. If none is available it emits `UNKNOWN` and records `MISSING_TARGET_TYPE`.

## 9. Validator unchanged

The production v1.1.3 validator and Phase 2.5 validation checks were executed unchanged. No required-field check, threshold, rejection rule, or fallback branch was relaxed.

## 10. Focused evaluation

The exact Phase 2.4.4 diagnostic set was run: 15 cases × {repeats} repeats. Freebase and OpenAI were live.

## 11. Serialization success rate

- Complete: {serialization_counts.get('COMPLETE', 0)}/30 repeats
- Partial: {serialization_counts.get('PARTIAL', 0)}/30 repeats
- Failed: {serialization_counts.get('FAILED', 0)}/30 repeats

## 12. Validator pass/rejection analysis

- Validator pass: {validator_counts.get('VALIDATOR_PASS', 0)}/30
- Validator rejection: {validator_counts.get('VALIDATOR_REJECT', 0)}/30
- Rejection classes: {dict(validator_reason_counts)}

Among the five previously uncorrected harmful cases, {upstream_extraction} had a complete serialized plan but remained rejected by upstream semantic/validator conditions, while {serialization_validator_failure} had partial/failed serialization.

## 13. Fallback reduction

- Phase 2.5.3A: 30/30 repeat fallbacks
- Phase 2.5.3B: {fallback_repeats}/30 repeat fallbacks and {fallback_cases}/15 cases

## 14. Native branch execution

- Execution statuses: {dict(execution_counts)}
- Native branch routed: {branch_routed_repeats}/30 repeats, {branch_routed_cases}/15 cases
- Native branch execution success: {branch_success_repeats}/30 repeats, {branch_success_cases}/15 cases

## 15. Branch self-sufficiency

Branch self-sufficiency is successful usable native branch answers divided by validator-passing branch routes: {branch_success_repeats}/{branch_routed_repeats} repeats and {branch_success_cases}/{branch_routed_cases} cases.

## 16. Harmful cases and controls

- Harmful cases corrected by semantic route majority: {corrected_harmful}/7
- Controls preserved: {controls_preserved}/4
- New harmful branch executions: {new_harmful_execution}; gate is <=1 and target is 0

Per-case route, serialization, validator, execution, and answer records are in `harmful_control_execution.jsonl`.

## 17. QA

- Frozen v1.1.3 Hits@1: 8/15
- Phase 2.5.1 Hits@1: {baseline_h1}/15
- Phase 2.5.3A Hits@1: {phase_3a_h1}/15
- Phase 2.5.3B Hits@1: {new_h1}/15
- Phase 2.5.3B F1 sum: {new_f1:.4f}

## 18. Stability

- Serialized branch plan agreement: {serialization_stable}/15 cases
- Serialization-status agreement: {status_stable}/15 cases
- Validator-outcome agreement: {validator_stable}/15 cases
- Final-answer agreement: {answer_stable}/15 cases

## 19. Cost

Serialization added 0 LLM calls. The run observed {llm_calls} frozen extraction/router calls, {input_tokens + output_tokens} total input/output tokens, {serialization_cpu:.6f} seconds of serialization CPU time, and {latency:.3f} seconds wall latency.

## 20. Gate decision

- Validator rejection reduced to <=15/30: `{validator_rejection_reduced}`
- Fallback materially reduced: `{fallback_reduced}`
- Controls preserved >=3/4: `{controls_preserved >= 3}`
- New harmful branch executions <=1: `{new_harmful_execution <= 1}`
- Final H@1 >= Phase 2.5.3A: `{qa_non_regressive}`
- Overall gate: `{gate_passed}`

## 21. Final Phase 2.5.3B decision

**{decision}**

## 22. Recommended next step

Do not run Stage B or full100. Perform one focused combined validation of the Phase 2.5.3A extraction plus Phase 2.5.3B serialization on the same 15 cases before deciding whether to proceed.
"""
	(OUTPUT_ROOT / "PHASE2_5_3B_REPORT.md").write_text(report, encoding="utf-8")
	return json.dumps({
		"status": "COMPLETE",
		"decision": decision,
		"cases": len(new_by_case),
		"repeats": repeats,
		"serialization_complete": serialization_counts.get("COMPLETE", 0),
		"validator_pass": validator_counts.get("VALIDATOR_PASS", 0),
		"fallback_repeats": fallback_repeats,
		"branch_success_repeats": branch_success_repeats,
		"controls_preserved": controls_preserved,
		"new_h1": new_h1,
		"output_root": str(OUTPUT_ROOT),
	}, indent=2)


def main() -> None:
	import argparse

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--run", action="store_true")
	parser.add_argument("--repeats", type=int, default=2)
	args = parser.parse_args()
	if args.repeats < 2:
		raise SystemExit("--repeats must be at least 2")
	rows = read_jsonl(INPUT_PATH)
	if len(rows) != 15:
		raise SystemExit(f"expected 15 focused cases, found {len(rows)}")
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
		new_rows = run_live(rows, args.repeats)
	except LiveValidationUnavailable as exc:
		print(json.dumps({"status": "INCONCLUSIVE", "reason": "live_router_unavailable", "detail": str(exc), "output_root": str(OUTPUT_ROOT)}, indent=2))
		return
	(OUTPUT_ROOT / "live_runs.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in new_rows), encoding="utf-8")
	print(build_outputs(rows, new_rows, args.repeats, preflight))


if __name__ == "__main__":
	main()
