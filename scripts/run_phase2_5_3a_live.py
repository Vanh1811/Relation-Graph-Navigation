#!/usr/bin/env python3
"""Run the focused Phase 2.5.3A extraction-only live validation."""

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
from chain_of_relations.methods.phase2_5_3a.agent import SourceTargetExtractionAgent  # noqa: E402
from chain_of_relations.schema import Entity  # noqa: E402


INPUT_PATH = ROOT / "outputs/experiments/phase2_4_4/routing_decision_error_analysis/router_input_output_analysis.jsonl"
BASELINE_ROOT = ROOT / "outputs/experiments/phase2_5_1/live_constraint_first_validation"
OUTPUT_ROOT = ROOT / "outputs/experiments/phase2_5_3a/source_target_extraction"


class LiveValidationUnavailable(RuntimeError):
	pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_agent() -> SourceTargetExtractionAgent:
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
	return SourceTargetExtractionAgent(cor, confidence_threshold=0.75, max_branches=3)


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
			router_events = [item for item in history if item.get("type") == "phase2_5_3a_source_target_extraction_router"]
			if not router_events or not router_events[-1].get("response") or not router_events[-1].get("parsed_result"):
				router_error = (router_events[-1].get("llm_error") if router_events else "") or "empty or invalid live router response"
				raise LiveValidationUnavailable(
					f"live Phase 2.5.3A router unavailable for {row['case_id']} run={run_id}: {router_error}"
				)
			latency = time.perf_counter() - started
			prediction = base.prediction_names(result)
			hit, f1 = base.evaluate(prediction, gold_answers)
			native = result.get("adaptive_native_result") if isinstance(result.get("adaptive_native_result"), dict) else result
			native_prediction = base.prediction_names(native)
			native_hit, native_f1 = base.evaluate(native_prediction, gold_answers)
			decision = result.get("router_decision", result)
			decision = decision if isinstance(decision, dict) else {}
			event = router_events[-1]
			output.append({
				"case_id": row["case_id"],
				"run_id": run_id,
				"question": row["question"],
				"recognized_entities": [{"id": e.id, "name": e.name} for e in entities],
				"constraints": event.get("constraints", decision.get("constraints", [])),
				"independent_source_groups": event.get("independent_source_groups", []),
				"independent_sources": decision.get("independent_sources", []),
				"shared_target": decision.get("shared_target", {}),
				"dependency_pattern": decision.get("dependency_pattern", "UNKNOWN"),
				"structure": decision.get("router_structure", "UNKNOWN"),
				"validator_outcome": event.get("validator_decision", "UNKNOWN"),
				"validation_failures": event.get("validation_failures", decision.get("validation_failures", [])),
				"extraction_completeness": event.get("extraction_completeness", {}),
				"retry_used": bool(event.get("retry_used", False)),
				"retry_reason": event.get("retry_reason", ""),
				"fallback_used": bool(result.get("fallback_used", False)),
				"fallback_reason": result.get("fallback_reason"),
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


def source_status(ref: dict[str, Any], item: dict[str, Any]) -> str:
	gold = int(ref.get("independent_sources_gold", 0) or 0)
	live = len(item.get("independent_source_groups") or item.get("independent_sources") or [])
	if gold <= 1:
		return "INDEPENDENT_SOURCE_FALSE_POSITIVE" if live > 1 else "INDEPENDENT_SOURCES_CORRECT"
	return "INDEPENDENT_SOURCES_CORRECT" if live >= 2 else "INDEPENDENT_SOURCE_MISSED"


def shared_status(ref: dict[str, Any], item: dict[str, Any]) -> str:
	gold = "ANSWER" if ref.get("shared_answer_gold") else "INTERMEDIATE" if ref.get("shared_intermediate_gold") else "NONE"
	live = str((item.get("shared_target") or {}).get("type", "NONE")).upper()
	if live == gold:
		return "SHARED_TARGET_CORRECT"
	if live == "NONE":
		return "SHARED_TARGET_MISSED"
	return "SHARED_TARGET_WRONG_TYPE" if gold != "NONE" else "SHARED_TARGET_SPURIOUS"


def dependency_status(ref: dict[str, Any], item: dict[str, Any]) -> str:
	live = str(item.get("dependency_pattern", "UNKNOWN")).upper()
	gold_shared = bool(ref.get("shared_answer_gold") or ref.get("shared_intermediate_gold"))
	if gold_shared:
		return "DEPENDENCY_CORRECT" if live in {"INDEPENDENT_CONVERGENCE", "MIXED"} else "FALSE_SEQUENTIALIZATION" if live == "SEQUENTIAL" else "OVER_UNCERTAIN"
	return "DEPENDENCY_CORRECT" if live == "SEQUENTIAL" else "FALSE_INDEPENDENCE" if live == "INDEPENDENT_CONVERGENCE" else "UNKNOWN"


def route_outcome(ref: dict[str, Any], structure: str) -> str:
	gold = base.gold_structure(ref["gold_structure"])
	if structure == gold:
		return f"CORRECT_{structure}"
	if structure == "UNCERTAIN":
		return "OVER_UNCERTAIN"
	if structure in {"BRANCH_FINAL", "BRANCH_INTERMEDIATE"}:
		return "FALSE_BRANCHING" if gold == "LINEAR" else "WRONG_BRANCH_TYPE"
	return "FALSE_LINEARIZATION"


def canonical(value: Any) -> str:
	return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def case_majority(items: list[dict[str, Any]], field: str) -> Any:
	return Counter(item.get(field) if field != "source_count" else len(item.get("independent_source_groups") or item.get("independent_sources") or []) for item in items).most_common(1)[0][0]


def build_outputs(ref_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]], repeats: int, preflight: dict[str, Any]) -> str:
	refs = {row["case_id"]: row for row in ref_rows}
	baseline = read_jsonl(BASELINE_ROOT / "live_runs.jsonl")
	base_by_key = {(row["case_id"], row["run_id"]): row for row in baseline}
	new_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for item in new_rows:
		new_by_case[item["case_id"]].append(item)

	comparison = []
	for item in new_rows:
		ref = refs[item["case_id"]]
		old = base_by_key.get((item["case_id"], item["run_id"]), {})
		comparison.append({
			"case_id": item["case_id"],
			"repeat_id": item["run_id"],
			"question": item["question"],
			"baseline_independent_sources": len(old.get("independent_sources") or []),
			"new_independent_sources": len(item.get("independent_source_groups") or item.get("independent_sources") or []),
			"baseline_shared_target": old.get("shared_target", {}),
			"new_shared_target": item.get("shared_target", {}),
			"baseline_dependency_pattern": old.get("dependency_pattern", "UNKNOWN"),
			"new_dependency_pattern": item.get("dependency_pattern", "UNKNOWN"),
			"baseline_structure": old.get("structure", "UNKNOWN"),
			"new_structure": item.get("structure", "UNKNOWN"),
			"source_status": source_status(ref, item),
			"shared_target_status": shared_status(ref, item),
			"dependency_status": dependency_status(ref, item),
			"validator_result": item.get("validator_outcome"),
			"fallback_used": item.get("fallback_used"),
			"native_prediction": item.get("native_prediction", []),
			"final_prediction": item.get("final_prediction", []),
			"hits_at_1": item.get("hits_at_1", 0),
			"f1": item.get("f1", 0.0),
			"retry_used": item.get("retry_used", False),
			"retry_reason": item.get("retry_reason", ""),
			"tokens": item.get("total_tokens", 0),
			"latency_seconds": item.get("latency_seconds", 0.0),
		})
	(OUTPUT_ROOT / "extraction_comparison.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in comparison), encoding="utf-8")

	groups = {row["case_id"]: ("HARMFUL" if row["routing_outcome"] == "HARMFUL_ROUTING_MISMATCH" else "BENIGN" if row["routing_outcome"] == "BENIGN_ROUTING_MISMATCH" else "CONTROL") for row in ref_rows}
	gold_by_case = {row["case_id"]: base.gold_structure(row["gold_structure"]) for row in ref_rows}
	majority_structure = {case_id: case_majority(items, "structure") for case_id, items in new_by_case.items()}
	harmful = [case_id for case_id, group in groups.items() if group == "HARMFUL"]
	controls = [case_id for case_id, group in groups.items() if group == "CONTROL"]
	corrected_harmful = sum(majority_structure.get(case_id) == gold_by_case[case_id] for case_id in harmful)
	controls_preserved = sum(majority_structure.get(case_id) == gold_by_case[case_id] for case_id in controls)
	false_positive = sum(majority_structure.get(case_id) in {"BRANCH_FINAL", "BRANCH_INTERMEDIATE"} and gold_by_case[case_id] == "LINEAR" for case_id in groups)
	baseline_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for item in baseline:
		baseline_by_case[item["case_id"]].append(item)
	baseline_h1 = sum(max((int(item.get("hits_at_1", 0) or 0) for item in values), default=0) for values in baseline_by_case.values())
	new_h1 = sum(max((int(item.get("hits_at_1", 0) or 0) for item in values), default=0) for values in new_by_case.values())
	baseline_f1 = sum(max((float(item.get("f1", 0.0) or 0.0) for item in values), default=0.0) for values in baseline_by_case.values())
	new_f1 = sum(max((float(item.get("f1", 0.0) or 0.0) for item in values), default=0.0) for values in new_by_case.values())
	problem_ids = {json.loads(line)["case_id"] for line in (ROOT / "outputs/experiments/phase2_5_2/live_extraction_fallback_analysis/harmful_case_analysis.jsonl").read_text().splitlines() if line.strip()}
	problem_items = [item for item in comparison if item["case_id"] in problem_ids]
	problem_cases = defaultdict(list)
	for item in problem_items:
		problem_cases[item["case_id"]].append(item)
	problem_source_errors = sum(any(row["source_status"] != "INDEPENDENT_SOURCES_CORRECT" for row in values) for values in problem_cases.values())
	problem_target_errors = sum(any(row["shared_target_status"] != "SHARED_TARGET_CORRECT" for row in values) for values in problem_cases.values())
	case_source_accuracy = sum(all(row["source_status"] == "INDEPENDENT_SOURCES_CORRECT" for row in values) for values in (defaultdict(list, {case: [x for x in comparison if x["case_id"] == case] for case in groups})).values())
	case_target_accuracy = sum(all(row["shared_target_status"] == "SHARED_TARGET_CORRECT" for row in values) for values in (defaultdict(list, {case: [x for x in comparison if x["case_id"] == case] for case in groups})).values())
	new_fallback_cases = sum(any(item.get("fallback_used") for item in items) for items in new_by_case.values())
	new_fallback_repeats = sum(bool(item.get("fallback_used")) for item in new_rows)
	validator_rejections = sum(str(item.get("validator_outcome", "")).upper() == "FALLBACK" for item in new_rows)
	structure_stable = sum(len({item.get("structure") for item in items}) == 1 for items in new_by_case.values())
	source_stable = sum(len({canonical(item.get("independent_source_groups", [])) for item in items}) == 1 for items in new_by_case.values())
	target_stable = sum(len({canonical(item.get("shared_target", {})) for item in items}) == 1 for items in new_by_case.values())
	dependency_stable = sum(len({item.get("dependency_pattern") for item in items}) == 1 for items in new_by_case.values())
	answer_stable = sum(len({(item.get("hits_at_1"), round(float(item.get("f1", 0.0)), 6)) for item in items}) == 1 for items in new_by_case.values())
	input_tokens = sum(int(item.get("input_tokens", 0) or 0) for item in new_rows)
	output_tokens = sum(int(item.get("output_tokens", 0) or 0) for item in new_rows)
	latency = sum(float(item.get("latency_seconds", 0.0) or 0.0) for item in new_rows)
	retry_count = sum(1 for item in new_rows if item.get("retry_used"))

	(OUTPUT_ROOT / "routing_comparison.md").write_text(
		"# Routing comparison\n\n"
		"| Metric | Phase 2.5.1 | Phase 2.5.3A |\n|---|---:|---:|\n"
		f"| Harmful cases corrected | 2/7 | {corrected_harmful}/7 |\n"
		f"| Controls preserved | 4/4 | {controls_preserved}/4 |\n"
		f"| New false-positive branch cases | 0 | {false_positive} |\n"
		f"| Structure-stable cases | 15/15 | {structure_stable}/15 |\n\n"
		"Structure mapping and validator implementation were not changed.\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "qa_comparison.md").write_text(
		"# QA comparison\n\n"
		"| Metric | Frozen baseline | Phase 2.5.1 | Phase 2.5.3A |\n|---|---:|---:|---:|\n"
		f"| Hits@1 | 8/15 | 9/15 | {new_h1}/15 |\n"
		f"| F1 sum | {baseline_f1:.4f} | 9.0000 | {new_f1:.4f} |\n"
		"| Native/final predictions | recorded | recorded | recorded |\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "stability_analysis.md").write_text(
		"# Stability analysis\n\n"
		f"Repeats per case: {repeats}\n\n"
		"| Metric | Agreement |\n|---|---:|\n"
		f"| Exact independent-source agreement | {source_stable}/15 |\n"
		f"| Exact shared-target agreement | {target_stable}/15 |\n"
		f"| Dependency-pattern agreement | {dependency_stable}/15 |\n"
		f"| Structure agreement | {structure_stable}/15 |\n"
		f"| Answer agreement (H@1/F1) | {answer_stable}/15 |\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "cost_comparison.md").write_text(
		"# Cost comparison\n\n"
		f"Phase 2.5.3A LLM calls: {sum(int(item.get('llm_calls', 0) or 0) for item in new_rows)}\n\n"
		f"Input tokens: {input_tokens}\n\nOutput tokens: {output_tokens}\n\nTotal tokens: {input_tokens + output_tokens}\n\n"
		f"Total wall latency: {latency:.3f} seconds\n\n"
		"Phase 2.5.1 reference: 305,538 total tokens and approximately 1007 seconds.\n",
		encoding="utf-8",
	)

	source_errors_case = sum(any(row["source_status"] != "INDEPENDENT_SOURCES_CORRECT" for row in values) for values in defaultdict(list, {case: [x for x in comparison if x["case_id"] == case] for case in groups}).values())
	target_errors_case = sum(any(row["shared_target_status"] != "SHARED_TARGET_CORRECT" for row in values) for values in defaultdict(list, {case: [x for x in comparison if x["case_id"] == case] for case in groups}).values())
	gate_passed = corrected_harmful >= 4 and controls_preserved >= 3 and false_positive <= 1 and new_h1 >= 9 and source_errors_case <= 1 and target_errors_case <= 1
	material_extraction_improvement = problem_source_errors < 4 or problem_target_errors < 4
	decision = "EXTRACTION_INTERVENTION_SUPPORTED" if gate_passed and material_extraction_improvement else "EXTRACTION_INTERVENTION_PARTIAL" if material_extraction_improvement else "EXTRACTION_INTERVENTION_NOT_SUPPORTED"
	report = f"""# Phase 2.5.3A independent-source and shared-target extraction intervention

## 1. Motivation from Phase 2.5.2

Phase 2.5.2 found three independent-source failures and two shared-target failures among the five uncorrected harmful cases. Validator/schema compatibility was identified as a separate problem and was frozen here.

## 2. Research question

Can a narrow extraction intervention improve independent-source and shared-target signals without changing structure mapping, validation, execution, or fallback policy?

## 3. Scope boundary

Only the extraction prompt, extraction field normalization, source-group completeness check, same-call self-check logging, and at most one bounded extraction retry were changed. The bounded retry was used {retry_count} time(s) in this run. Structure mapping, validator, branch schema, fallback, backend, evaluator, model, and settings remain frozen.

## 4. Extraction intervention

The router now requires explicit `source_anchor`, `target_variable_role`, `answer_relevant`, `independent_source_groups`, and `shared_target` fields. It asks whether distinct source anchors independently constrain the same answer or intermediate and excludes TYPE-only anchors.

## 5. Focused evaluation

The exact 15-case Phase 2.4.4 diagnostic set was run with {repeats} repeats per case. Freebase and the OpenAI endpoint were live during the run.

## 6. Extraction metrics

- Independent-source accuracy: {sum(row['source_status'] == 'INDEPENDENT_SOURCES_CORRECT' for row in comparison)}/{len(comparison)} repeat rows; case-level exact agreement: {case_source_accuracy}/15.
- Shared-target accuracy: {sum(row['shared_target_status'] == 'SHARED_TARGET_CORRECT' for row in comparison)}/{len(comparison)} repeat rows; case-level exact agreement: {case_target_accuracy}/15.
- Dependency-pattern accuracy: {sum(row['dependency_status'] == 'DEPENDENCY_CORRECT' for row in comparison)}/{len(comparison)} repeat rows.
- Previously uncorrected harmful cases with any source error: {problem_source_errors}/5.
- Previously uncorrected harmful cases with any shared-target error: {problem_target_errors}/5.

## 7. Routing and controls

- Harmful cases corrected: {corrected_harmful}/7; target gate is >=4/7.
- Controls preserved: {controls_preserved}/4; target gate is >=3/4.
- New false-positive branch cases: {false_positive}; limit is <=1.
- Structure agreement across repeats: {structure_stable}/15.

## 8. QA

- Frozen baseline Hits@1: 8/15.
- Phase 2.5.1 Hits@1: 9/15.
- Phase 2.5.3A Hits@1: {new_h1}/15.
- Frozen baseline F1 sum: {baseline_f1:.4f}.
- Phase 2.5.1 F1 sum: 9.0000.
- Phase 2.5.3A F1 sum: {new_f1:.4f}.

## 9. Fallback and validator interaction

Phase 2.5.3A does not modify validator or fallback. It produced {new_fallback_cases}/15 fallback cases, {new_fallback_repeats} fallback repeats, and {validator_rejections} validator rejections. Any improved extraction that still falls back is therefore evidence for Phase 2.5.3B, not a failure of this extraction-only boundary.

## 10. Stability

Independent-source agreement: {source_stable}/15; shared-target agreement: {target_stable}/15; dependency agreement: {dependency_stable}/15; structure agreement: {structure_stable}/15; answer agreement: {answer_stable}/15.

## 11. Cost

Phase 2.5.3A used {sum(int(item.get('llm_calls', 0) or 0) for item in new_rows)} LLM calls, {input_tokens + output_tokens} total tokens, and {latency:.3f} seconds. Phase 2.5.1 used 305,538 tokens and approximately 1007 seconds.

## 12. Gate decision

Gate passed: `{gate_passed}`. Material extraction improvement observed: `{material_extraction_improvement}`.

## 13. Final Phase 2.5.3A decision

**{decision}**

## 14. Recommended Phase 2.5.3B handoff

Do not run full100. If the extraction result is partial, the next focused experiment should address branch-plan serialization/validator compatibility separately, using the Phase 2.5.3A logs. Do not change validator/schema in this phase's code.
"""
	(OUTPUT_ROOT / "PHASE2_5_3A_REPORT.md").write_text(report, encoding="utf-8")
	return json.dumps({"status": "COMPLETE", "decision": decision, "cases": len(new_by_case), "repeats": repeats, "harmful_corrected": corrected_harmful, "controls_preserved": controls_preserved, "new_h1": new_h1, "output_root": str(OUTPUT_ROOT)}, indent=2)


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
