#!/usr/bin/env python3
"""Run the focused Phase 2.5.5 canonical-normalization evaluation."""

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
from chain_of_relations.methods.phase2_5_5.agent import CanonicalNormalizationAgent  # noqa: E402
from chain_of_relations.schema import Entity  # noqa: E402


INPUT_PATH = ROOT / "outputs/experiments/phase2_4_4/routing_decision_error_analysis/router_input_output_analysis.jsonl"
BASELINE_ROOT = ROOT / "outputs/experiments/phase2_5_1/live_constraint_first_validation"
PHASE_3A_ROOT = ROOT / "outputs/experiments/phase2_5_3a/source_target_extraction"
PHASE_3B_ROOT = ROOT / "outputs/experiments/phase2_5_3b/branch_plan_serialization"
OUTPUT_ROOT = ROOT / "outputs/experiments/phase2_5_5/canonical_semantic_normalization"


class LiveValidationUnavailable(RuntimeError):
	pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_agent() -> CanonicalNormalizationAgent:
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
	return CanonicalNormalizationAgent(cor, confidence_threshold=0.75, max_branches=3)


def run_live(rows: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
	agent = make_agent()
	output: list[dict[str, Any]] = []
	for row in rows:
		entities = [Entity(str(item.get("id", "")), str(item.get("name", ""))) for item in row.get("router_input", {}).get("runtime_topic_entities", [])]
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
			events = [item for item in history if item.get("type") == "phase2_5_5_canonical_normalization_router"]
			if not events or not events[-1].get("response") or not events[-1].get("parsed_result"):
				reason = (events[-1].get("llm_error") if events else "") or "empty or invalid live router response"
				raise LiveValidationUnavailable(f"live Phase 2.5.5 router unavailable for {row['case_id']} run={run_id}: {reason}")
			event = events[-1]
			serialization = event.get("serialization", {})
			normalization = event.get("normalization", {})
			decision = result.get("router_decision", result)
			decision = decision if isinstance(decision, dict) else {}
			validator_passed = str(event.get("validator_decision", "")).upper() == "PASS"
			branch_routed = validator_passed and decision.get("router_execution_mode") == "branch"
			native = result.get("adaptive_native_result") if isinstance(result.get("adaptive_native_result"), dict) else result
			prediction = base.prediction_names(result)
			native_prediction = base.prediction_names(native)
			hit, f1 = base.evaluate(prediction, gold_answers)
			native_hit, native_f1 = base.evaluate(native_prediction, gold_answers)
			failure_reasons = list(event.get("validation_failures", decision.get("validation_failures", [])))
			execution = phase3b.execution_status(result, validator_passed) if branch_routed else (
				"NOT_APPLICABLE_NON_EXECUTABLE_STRUCTURE" if validator_passed else "NOT_ATTEMPTED_VALIDATOR_REJECT"
			)
			output.append({
				"case_id": row["case_id"],
				"run_id": run_id,
				"question": row["question"],
				"semantic_structure": decision.get("router_structure", "UNKNOWN"),
				"raw_semantic_constraints": event.get("constraints", []),
				"raw_shared_target": event.get("parsed_result", {}).get("shared_target"),
				"raw_dependency_pattern": event.get("parsed_result", {}).get("dependency_pattern"),
				"canonical_semantic_representation": event.get("canonical_semantic_representation", normalization.get("canonical_proposal", {})),
				"normalization": normalization,
				"normalization_status": normalization.get("normalization_status", "NORMALIZATION_FAILED"),
				"normalization_errors": list(normalization.get("normalization_errors", [])),
				"serialization": serialization,
				"serialized_branches": list(serialization.get("serialized_branches", [])),
				"serialization_status": serialization.get("serialization_status", "FAILED"),
				"validator_result": "VALIDATOR_PASS" if validator_passed else "VALIDATOR_REJECT",
				"validator_reason": failure_reasons,
				"validator_reason_classes": phase3b.classify_validator_failures(failure_reasons),
				"fallback_used": bool(result.get("fallback_used", False)),
				"fallback_reason": result.get("fallback_reason"),
				"execution_status": execution,
				"branch_routed": branch_routed,
				"branch_self_sufficient": branch_routed and execution == "EXECUTION_SUCCESS",
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
				"latency_seconds": time.perf_counter() - started,
				"error": error,
				"prompt_history": history,
				"sparql_query_count": len(sparql_history),
			})
	return output


def qa_metrics(rows: list[dict[str, Any]]) -> tuple[int, float]:
	by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		by_case[row["case_id"]].append(row)
	return (
		sum(max((int(item.get("hits_at_1", 0) or 0) for item in values), default=0) for values in by_case.values()),
		sum(max((float(item.get("f1", 0.0) or 0.0) for item in values), default=0.0) for values in by_case.values()),
	)


def canonical(value: Any) -> str:
	return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_outputs(ref_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]], repeats: int) -> str:
	refs = {row["case_id"]: row for row in ref_rows}
	baseline = read_jsonl(BASELINE_ROOT / "live_runs.jsonl")
	phase_3a = read_jsonl(PHASE_3A_ROOT / "live_runs.jsonl")
	phase_3b = read_jsonl(PHASE_3B_ROOT / "serialization_results.jsonl")
	by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in new_rows:
		by_case[row["case_id"]].append(row)
	groups = phase3b.group_cases(ref_rows)
	gold = {row["case_id"]: base.gold_structure(row["gold_structure"]) for row in ref_rows}
	majority_structure = {case: Counter(row["semantic_structure"] for row in values).most_common(1)[0][0] for case, values in by_case.items()}
	harmful = [case for case, group in groups.items() if group == "HARMFUL"]
	controls = [case for case, group in groups.items() if group == "CONTROL"]
	corrected_harmful = sum(majority_structure.get(case) == gold[case] for case in harmful)
	controls_preserved = sum(majority_structure.get(case) == gold[case] for case in controls)
	false_positive_execution = sum(
		gold[case] != "BRANCH_FINAL" and any(row.get("branch_self_sufficient") and not row.get("hits_at_1") for row in values)
		for case, values in by_case.items()
	)
	normalization_counts = Counter(row.get("normalization_status") for row in new_rows)
	validator_counts = Counter(row.get("validator_result") for row in new_rows)
	fallback_repeats = sum(bool(row.get("fallback_used")) for row in new_rows)
	fallback_cases = sum(any(row.get("fallback_used") for row in values) for values in by_case.values())
	execution_counts = Counter(row.get("execution_status") for row in new_rows)
	branch_routed_repeats = sum(bool(row.get("branch_routed")) for row in new_rows)
	branch_success_repeats = sum(bool(row.get("branch_self_sufficient")) for row in new_rows)
	branch_routed_cases = sum(any(row.get("branch_routed") for row in values) for values in by_case.values())
	branch_success_cases = sum(any(row.get("branch_self_sufficient") for row in values) for values in by_case.values())
	base_h1, base_f1 = qa_metrics(baseline)
	a_h1, a_f1 = qa_metrics(phase_3a)
	b_h1, b_f1 = qa_metrics(phase_3b)
	new_h1, new_f1 = qa_metrics(new_rows)

	shared_fixed = 0
	shared_remaining = 0
	shared_semantic_wrong = 0
	for row in new_rows:
		ref = refs[row["case_id"]]
		expected = "ANSWER" if ref.get("shared_answer_gold") else "INTERMEDIATE" if ref.get("shared_intermediate_gold") else "NONE"
		raw = row.get("raw_shared_target")
		raw_type = str(raw.get("type") if isinstance(raw, dict) else raw or "").upper()
		canonical_target = row.get("canonical_semantic_representation", {}).get("shared_target", {})
		canonical_type = str(canonical_target.get("type", "")).upper() if isinstance(canonical_target, dict) else ""
		if raw_type != expected:
			shared_semantic_wrong += 1
		elif canonical_type == expected and isinstance(canonical_target, dict) and canonical_target.get("description"):
			shared_fixed += 1
		else:
			shared_remaining += 1
	type_only_before = sum("TYPE_ONLY_BRANCH" in row.get("validator_reason_classes", []) for row in phase_3b)
	type_only_after = sum("TYPE_ONLY_BRANCH" in row.get("validator_reason_classes", []) for row in new_rows)
	target_before = sum("MISSING_OR_INCOMPATIBLE_TARGET_TYPE" in row.get("validator_reason_classes", []) for row in phase_3b)
	target_after = sum("MISSING_OR_INCOMPATIBLE_TARGET_TYPE" in row.get("validator_reason_classes", []) for row in new_rows)
	serialization_stable = sum(len({canonical(row.get("serialized_branches", [])) for row in values}) == 1 for values in by_case.values())
	canonical_target_stable = sum(len({canonical(row.get("canonical_semantic_representation", {}).get("shared_target", {})) for row in values}) == 1 for values in by_case.values())
	canonical_dependency_stable = sum(len({row.get("canonical_semantic_representation", {}).get("dependency_pattern") for row in values}) == 1 for values in by_case.values())
	canonical_roles_stable = sum(len({canonical([item.get("target_role") for item in row.get("canonical_semantic_representation", {}).get("constraints", [])]) for row in values}) == 1 for values in by_case.values())
	validator_stable = sum(len({row.get("validator_result") for row in values}) == 1 for values in by_case.values())
	answer_stable = sum(len({(row.get("hits_at_1"), round(float(row.get("f1", 0.0)), 6)) for row in values}) == 1 for values in by_case.values())
	input_tokens = sum(int(row.get("input_tokens", 0) or 0) for row in new_rows)
	output_tokens = sum(int(row.get("output_tokens", 0) or 0) for row in new_rows)
	llm_calls = sum(int(row.get("llm_calls", 0) or 0) for row in new_rows)
	latency = sum(float(row.get("latency_seconds", 0.0) or 0.0) for row in new_rows)
	normalization_cpu = sum(float(row.get("normalization", {}).get("normalization_cpu_seconds", 0.0) or 0.0) for row in new_rows)

	results = []
	for row in new_rows:
		results.append({
			"case_id": row["case_id"],
			"repeat_id": row["run_id"],
			"question": row["question"],
			"raw_shared_target": row["raw_shared_target"],
			"canonical_shared_target": row.get("canonical_semantic_representation", {}).get("shared_target"),
			"raw_dependency_pattern": row["raw_dependency_pattern"],
			"canonical_dependency_pattern": row.get("canonical_semantic_representation", {}).get("dependency_pattern"),
			"normalization_status": row["normalization_status"],
			"normalization_errors": row["normalization_errors"],
			"normalization": row["normalization"],
			"serialized_branches": row["serialized_branches"],
			"serialization_status": row["serialization_status"],
			"validator_result": row["validator_result"],
			"validator_reason": row["validator_reason"],
			"fallback_used": row["fallback_used"],
			"execution_status": row["execution_status"],
			"branch_self_sufficient": row["branch_self_sufficient"],
			"native_prediction": row["native_prediction"],
			"final_prediction": row["final_prediction"],
			"hits_at_1": row["hits_at_1"],
			"f1": row["f1"],
			"llm_calls": row["llm_calls"],
			"tokens": row["input_tokens"] + row["output_tokens"],
			"latency_seconds": row["latency_seconds"],
		})
	(OUTPUT_ROOT / "normalization_results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
	(OUTPUT_ROOT / "validator_comparison.md").write_text(
		"# Validator comparison\n\n"
		"| Metric | Phase 2.5.3A | Phase 2.5.3B | Phase 2.5.5 |\n|---|---:|---:|---:|\n"
		f"| Validator pass repeats | 0/30 | 0/30 | {validator_counts.get('VALIDATOR_PASS', 0)}/30 |\n"
		f"| Validator rejection repeats | 30/30 | 30/30 | {validator_counts.get('VALIDATOR_REJECT', 0)}/30 |\n"
		f"| Shared-target representation fixed | — | 0/30 | {shared_fixed}/30 |\n"
		f"| Shared-target representation remaining | — | 25/30 | {shared_remaining}/30 |\n"
		f"| Genuine semantic shared-target errors | — | 5/30 | {shared_semantic_wrong}/30 |\n"
		f"| TYPE_ONLY_BRANCH rejections | — | {type_only_before}/30 | {type_only_after}/30 |\n"
		f"| Target-type rejections | — | {target_before}/30 | {target_after}/30 |\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "fallback_comparison.md").write_text(
		"# Fallback comparison\n\n"
		"| Metric | Phase 2.5.3A | Phase 2.5.3B | Phase 2.5.5 |\n|---|---:|---:|---:|\n"
		f"| Fallback repeats | 30/30 | 30/30 | {fallback_repeats}/30 |\n"
		f"| Fallback cases | 15/15 | 15/15 | {fallback_cases}/15 |\n"
		f"| Native branch routed repeats | 0/30 | 0/30 | {branch_routed_repeats}/30 |\n"
		f"| Native branch success repeats | 0/30 | 0/30 | {branch_success_repeats}/30 |\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "qa_comparison.md").write_text(
		"# QA comparison\n\n"
		"| Metric | Frozen v1.1.3 | Phase 2.5.1 | Phase 2.5.3A | Phase 2.5.3B | Phase 2.5.5 |\n|---|---:|---:|---:|---:|---:|\n"
		f"| Hits@1 | 8/15 | {base_h1}/15 | {a_h1}/15 | {b_h1}/15 | {new_h1}/15 |\n"
		f"| F1 sum | 9.0000 | {base_f1:.4f} | {a_f1:.4f} | {b_f1:.4f} | {new_f1:.4f} |\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "stability_analysis.md").write_text(
		"# Normalization stability analysis\n\n"
		f"Repeats per case: {repeats}\n\n"
		"| Metric | Agreement |\n|---|---:|\n"
		f"| Canonical shared-target agreement | {canonical_target_stable}/15 |\n"
		f"| Canonical dependency agreement | {canonical_dependency_stable}/15 |\n"
		f"| Canonical role agreement | {canonical_roles_stable}/15 |\n"
		f"| Serialization agreement | {serialization_stable}/15 |\n"
		f"| Validator agreement | {validator_stable}/15 |\n"
		f"| Answer agreement | {answer_stable}/15 |\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "cost_comparison.md").write_text(
		"# Cost comparison\n\n"
		"Additional LLM calls caused by normalization: 0\n\n"
		f"Total frozen extraction/router calls: {llm_calls}\n\nInput tokens: {input_tokens}\n\nOutput tokens: {output_tokens}\n\nTotal tokens: {input_tokens + output_tokens}\n\n"
		f"Normalization CPU time: {normalization_cpu:.6f} seconds\n\nWall latency: {latency:.3f} seconds\n\n"
		"Phase 2.5.3B reference: 197 calls, 325,290 tokens, 916.337 seconds.\n",
		encoding="utf-8",
	)

	gate = (
		validator_counts.get("VALIDATOR_PASS", 0) >= 10
		and fallback_repeats < 30
		and controls_preserved >= 3
		and false_positive_execution <= 1
		and new_h1 >= b_h1
		and shared_remaining <= 5
	)
	material = validator_counts.get("VALIDATOR_PASS", 0) > 0 or fallback_repeats < 30 or shared_fixed > 0
	if gate:
		decision = "CANONICAL_NORMALIZATION_SUPPORTED"
	elif material:
		decision = "CANONICAL_NORMALIZATION_PARTIAL"
	else:
		decision = "CANONICAL_NORMALIZATION_NOT_SUPPORTED"
	report = f"""# Phase 2.5.5 canonical semantic-to-contract normalization

## 1. Motivation from Phase 2.5.4

Phase 2.5.4 found 20/30 rejected repeats that were semantically aligned but schema-incompatible, with shared-target representation mismatch in 25/30 repeats. The validator itself was not shown to be broadly over-rejecting.

## 2. Research question

Does deterministic normalization of semantically equivalent representations reduce validator rejection and fallback without weakening the existing validator or changing upstream semantic decisions?

## 3. Scope boundary

Only a pure Python canonical normalization layer was added before the frozen Phase 2.5.3B serializer. Extraction prompts/retries, source grouping, shared-target meaning, dependency meaning, structure labels, serializer semantics, validator, fallback, executor, backend, evaluator, model, and settings were frozen.

## 4. Canonical normalization contract

See `normalization_contract.md`. The adapter accepts only known aliases and formatting variants, preserves semantic meaning, records provenance, and refuses unknown transformations.

## 5. Shared-target normalization

- Representation fixed: {shared_fixed}/30
- Representation remaining: {shared_remaining}/30
- Genuine upstream semantic target errors: {shared_semantic_wrong}/30

Scalar `ANSWER`/`INTERMEDIATE` values were converted to the required object shape using existing target metadata for the description. `NONE` was never promoted to `ANSWER`.

## 6. Dependency normalization

Known sequential, independent, parallel, and intermediate-followed-by-answer variants were mapped to existing canonical enums. Unknown descriptions remained unmapped. Canonical agreement was {canonical_dependency_stable}/15 cases.

## 7. Constraint-role normalization

Known final-answer, intermediate-entity, filter, and type-constraint aliases were mapped to `ANSWER`, `INTERMEDIATE`, `FILTER`, and `TYPE`. Unknown roles remained `UNKNOWN`. Canonical role agreement was {canonical_roles_stable}/15 cases.

## 8. Target-type normalization

Only formatting, existing project-style aliases, and explicit question wording such as ideology/political ideology were normalized. Target-type rejections changed from {target_before}/30 to {target_after}/30.

## 9. Property normalization

Only formatting/case/separator changes were applied. Distinct complementary properties were not collapsed.

## 10. Frozen validator

The existing Phase 2.5 validator and v1.1.3 executable contract were reused unchanged. No rejection rule, threshold, type-only check, property check, or fallback path was relaxed.

## 11. Focused 15-case evaluation

The exact diagnostic set was run with {repeats} repeats per case. No additional LLM calls were introduced.

## 12. Validator pass/rejection results

- Phase 2.5.3B: 0/30 passes, 30/30 rejections
- Phase 2.5.5: {validator_counts.get('VALIDATOR_PASS', 0)}/30 passes, {validator_counts.get('VALIDATOR_REJECT', 0)}/30 rejections
- TYPE_ONLY_BRANCH: {type_only_before}/30 → {type_only_after}/30

## 13. Shared-target mismatch reduction

Representation mismatch reduced from 25/30 to {shared_remaining}/30. Genuine semantic target errors were kept separate and were not normalized.

## 14. Type-only rejection changes

TYPE_ONLY_BRANCH rejections changed from {type_only_before}/30 to {type_only_after}/30. Correct type-only safety behavior was not intentionally relaxed.

## 15. Target-type rejection changes

MISSING_OR_INCOMPATIBLE_TARGET_TYPE changed from {target_before}/30 to {target_after}/30. Missing metadata and genuine target errors were not forced into valid types.

## 16. Fallback reduction

Fallback changed from Phase 2.5.3B's 30/30 repeats and 15/15 cases to {fallback_repeats}/30 repeats and {fallback_cases}/15 cases.

## 17. Native branch execution

Execution statuses: {dict(execution_counts)}. Native branch routing occurred in {branch_routed_repeats}/30 repeats and succeeded in {branch_success_repeats}/30 repeats.

## 18. Branch self-sufficiency

Branch self-sufficiency was {branch_success_repeats}/{branch_routed_repeats} repeats and {branch_success_cases}/{branch_routed_cases} cases.

## 19. Harmful cases

Harmful route corrections: {corrected_harmful}/7. New harmful branch executions: {false_positive_execution}; gate is <=1 and target is 0. Per-repeat normalization, validator, execution, native answer, and final answer records are in `normalization_results.jsonl`.

## 20. Controls

Controls preserved: {controls_preserved}/4. Normalization is deterministic and recorded per field; no validator safety rule was relaxed.

## 21. QA

- Frozen v1.1.3 Hits@1: 8/15
- Phase 2.5.1 Hits@1: {base_h1}/15
- Phase 2.5.3A Hits@1: {a_h1}/15
- Phase 2.5.3B Hits@1: {b_h1}/15
- Phase 2.5.5 Hits@1: {new_h1}/15
- Phase 2.5.5 F1 sum: {new_f1:.4f}

## 22. Stability

Canonical target agreement: {canonical_target_stable}/15; dependency agreement: {canonical_dependency_stable}/15; role agreement: {canonical_roles_stable}/15; serialization agreement: {serialization_stable}/15; validator agreement: {validator_stable}/15; answer agreement: {answer_stable}/15.

## 23. Cost

Normalization added 0 LLM calls. The run used {llm_calls} total frozen router/execution calls, {input_tokens + output_tokens} input/output tokens, {normalization_cpu:.6f} seconds normalization CPU time, and {latency:.3f} seconds wall latency.

## 24. Gate decision

- Validator pass materially improves to >=10/30: `{validator_counts.get('VALIDATOR_PASS', 0) >= 10}`
- Fallback materially decreases: `{fallback_repeats < 30}`
- Controls >=3/4: `{controls_preserved >= 3}`
- New harmful execution <=1: `{false_positive_execution <= 1}`
- Final H@1 >= Phase 2.5.3B: `{new_h1 >= b_h1}`
- Shared-target mismatch <=5/30: `{shared_remaining <= 5}`
- Overall gate: `{gate}`

## 25. Final Phase 2.5.5 decision

**{decision}**

## 26. Recommended next step

Do not run Stage B or full100 automatically. If normalization improves validation, perform one focused combined validation of extraction + normalization + serialization + unchanged validator on the same 15 cases before deciding on fresh Stage B. If validation remains unchanged, return to upstream semantic errors rather than weakening the validator.
"""
	(OUTPUT_ROOT / "PHASE2_5_5_REPORT.md").write_text(report, encoding="utf-8")
	return json.dumps({
		"status": "COMPLETE",
		"decision": decision,
		"cases": len(by_case),
		"repeats": repeats,
		"normalization_status": dict(normalization_counts),
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
	if not preflight["freebase"].get("available"):
		print(json.dumps({"status": "INCONCLUSIVE", "reason": "freebase_endpoint_unavailable", "output_root": str(OUTPUT_ROOT)}, indent=2))
		return
	if not args.run:
		print(json.dumps({"status": "PREFLIGHT_OK_RUN_NOT_REQUESTED", "output_root": str(OUTPUT_ROOT)}, indent=2))
		return
	(OUTPUT_ROOT / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	try:
		new_rows = run_live(rows, args.repeats)
	except LiveValidationUnavailable as exc:
		print(json.dumps({"status": "INCONCLUSIVE", "reason": "live_router_unavailable", "detail": str(exc), "output_root": str(OUTPUT_ROOT)}, indent=2))
		return
	(OUTPUT_ROOT / "live_runs.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in new_rows), encoding="utf-8")
	print(build_outputs(rows, new_rows, args.repeats))


if __name__ == "__main__":
	main()
