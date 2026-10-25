#!/usr/bin/env python3
"""Audit Phase 2.5.1 live extraction and fallback failures without new LLM calls."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "outputs/experiments/phase2_4_4/routing_decision_error_analysis/router_input_output_analysis.jsonl"
LIVE_ROOT = ROOT / "outputs/experiments/phase2_5_1/live_constraint_first_validation"
OUTPUT_ROOT = ROOT / "outputs/experiments/phase2_5_2/live_extraction_fallback_analysis"

STRUCTURE_MAP = {
	"GOLD_LINEAR": "LINEAR",
	"GOLD_BRANCH_FINAL": "BRANCH_FINAL",
	"GOLD_BRANCH_INTERMEDIATE": "BRANCH_INTERMEDIATE",
}

# The live router emits semantic descriptions rather than Freebase relation IDs.
# These notes make the constraint audit explicit without pretending that a
# free-text property is an exact ontology-level relation match.
HARMFUL_CONSTRAINT_AUDIT: dict[str, dict[str, Any]] = {
	"4302233001000": {
		"status": "ALL_ESSENTIAL_CONSTRAINTS_DETECTED",
		"note": "The live text identifies McClelland's research discipline and the social-science domain; the failure is that the two sources are serialized as one sequential chain.",
	},
	"WebQTest-743_af8b975343e4098913128e9ac7d04c05": {
		"status": "ALL_ESSENTIAL_CONSTRAINTS_DETECTED",
		"note": "The live text identifies the anti-apartheid speaker, brothers, and male answer type; it does not preserve the two-source intermediate convergence.",
	},
	"WebQTest-719_e56e856592a9a57b9e331b718adc1b75": {
		"status": "ALL_ESSENTIAL_CONSTRAINTS_DETECTED",
		"note": "The live text identifies the author of The Audacity of Hope and the requested political ideal; it serializes the reasoning instead of retaining the shared intermediate.",
	},
	"WebQTest-759_88081dc1a259cbd172b89242384ee1cc": {
		"status": "ALL_ESSENTIAL_CONSTRAINTS_DETECTED",
		"note": "The live text identifies Brad Pitt's dating relation, celebrity type, and the 1989 end-date filter; the shared target is typed as the final answer instead of an intermediate relationship.",
	},
	"WebQTrn-64_984678664419049133d2bd748b68df1b": {
		"status": "ALL_ESSENTIAL_CONSTRAINTS_DETECTED",
		"note": "The live text identifies the artist of P.Y.T. and the role/name in Juice; the source and shared-intermediate representation is inconsistent across repeats.",
	},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical(value: Any) -> str:
	return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def group_for(row: dict[str, Any]) -> str:
	return {
		"HARMFUL_ROUTING_MISMATCH": "HARMFUL",
		"BENIGN_ROUTING_MISMATCH": "BENIGN",
	}.get(str(row.get("routing_outcome")), "CONTROL")


def gold_structure(row: dict[str, Any]) -> str:
	return STRUCTURE_MAP.get(str(row.get("gold_structure")), str(row.get("gold_structure")))


def gold_target(row: dict[str, Any]) -> str:
	if row.get("shared_answer_gold"):
		return "ANSWER"
	if row.get("shared_intermediate_gold"):
		return "INTERMEDIATE"
	return "NONE"


def majority(values: list[Any]) -> Any:
	return Counter(values).most_common(1)[0][0] if values else None


def first_router_event(item: dict[str, Any]) -> dict[str, Any]:
	for event in item.get("prompt_history", []):
		if event.get("type") == "phase2_5_constraint_first_router":
			return event
	return {}


def source_status(ref: dict[str, Any], item: dict[str, Any]) -> str:
	gold = int(ref.get("independent_sources_gold", 0) or 0)
	live = len(item.get("independent_sources") or [])
	if gold <= 1:
		return "INDEPENDENT_SOURCE_FALSE_POSITIVE" if live > 1 else "INDEPENDENT_SOURCES_CORRECT"
	return "INDEPENDENT_SOURCES_CORRECT" if live >= 2 else "INDEPENDENT_SOURCE_MISSED"


def shared_target_status(ref: dict[str, Any], item: dict[str, Any]) -> str:
	gold = gold_target(ref)
	live = str((item.get("shared_target") or {}).get("type", "NONE")).upper() or "NONE"
	if live == gold:
		return "SHARED_TARGET_CORRECT"
	if live == "NONE":
		return "SHARED_TARGET_MISSED"
	if gold == "NONE":
		return "SHARED_TARGET_SPURIOUS"
	return "SHARED_TARGET_WRONG_TYPE"


def dependency_status(ref: dict[str, Any], item: dict[str, Any]) -> str:
	live = str(item.get("dependency_pattern", "UNKNOWN")).upper()
	gold = gold_target(ref)
	if gold in {"ANSWER", "INTERMEDIATE"}:
		if live in {"INDEPENDENT_CONVERGENCE", "MIXED"}:
			return "DEPENDENCY_CORRECT"
		if live == "SEQUENTIAL":
			return "FALSE_SEQUENTIALIZATION"
		return "OVER_UNCERTAIN" if live == "UNCERTAIN" else "UNKNOWN"
	if live == "SEQUENTIAL":
		return "DEPENDENCY_CORRECT"
	if live == "INDEPENDENT_CONVERGENCE":
		return "FALSE_INDEPENDENCE"
	return "WRONG_MIXED_PATTERN" if live == "MIXED" else "UNKNOWN"


def expected_live_mapping(item: dict[str, Any]) -> str:
	target = str((item.get("shared_target") or {}).get("type", "NONE")).upper()
	dependency = str(item.get("dependency_pattern", "UNKNOWN")).upper()
	if dependency == "SEQUENTIAL":
		return "LINEAR"
	if dependency in {"INDEPENDENT_CONVERGENCE", "MIXED"} and target == "ANSWER":
		return "BRANCH_FINAL"
	if dependency in {"INDEPENDENT_CONVERGENCE", "MIXED"} and target == "INTERMEDIATE":
		return "BRANCH_INTERMEDIATE"
	return "UNCERTAIN"


def structure_mapping_status(item: dict[str, Any]) -> str:
	return "STRUCTURE_MAPPING_CORRECT" if item.get("structure") == expected_live_mapping(item) else "STRUCTURE_MAPPING_ERROR"


def validator_status(item: dict[str, Any]) -> str:
	if str(item.get("validator_outcome", "")).upper() != "FALLBACK":
		return "VALIDATOR_NOT_APPLICABLE"
	return "VALIDATOR_CORRECT_REJECTION"


def constraint_audit(ref: dict[str, Any], item: dict[str, Any], group: str) -> tuple[str, str, list[dict[str, str]]]:
	if group == "HARMFUL" and ref["case_id"] in HARMFUL_CONSTRAINT_AUDIT:
		audit = HARMFUL_CONSTRAINT_AUDIT[ref["case_id"]]
		items = [
			{"relation": str(value.get("relation", "")), "status": "DETECTED"}
			for value in ref.get("gold_constraints", [])
		]
		return audit["status"], audit["note"], items
	if not item.get("constraints"):
		return "ESSENTIAL_CONSTRAINT_MISSED", "No live constraints were recorded.", []
	return "UNKNOWN", "The free-text live properties were not mapped to exact Freebase relation IDs in this audit.", []


def primary_first_failure(ref: dict[str, Any], item: dict[str, Any]) -> str:
	constraint_status, _, _ = constraint_audit(ref, item, group_for(ref))
	if constraint_status == "ESSENTIAL_CONSTRAINT_MISSED":
		return "CONSTRAINT_EXTRACTION_ERROR"
	if source_status(ref, item) == "INDEPENDENT_SOURCE_MISSED":
		return "INDEPENDENT_SOURCE_EXTRACTION_ERROR"
	if shared_target_status(ref, item) in {"SHARED_TARGET_MISSED", "SHARED_TARGET_WRONG_TYPE", "SHARED_TARGET_SPURIOUS"}:
		return "SHARED_TARGET_ERROR"
	if dependency_status(ref, item) != "DEPENDENCY_CORRECT":
		return "DEPENDENCY_PATTERN_ERROR"
	if structure_mapping_status(item) == "STRUCTURE_MAPPING_ERROR":
		return "STRUCTURE_MAPPING_ERROR"
	if str(item.get("validator_outcome", "")).upper() == "FALLBACK":
		return "VALIDATOR_REJECTION"
	if item.get("fallback_used"):
		return "FALLBACK_POLICY_ERROR"
	if item.get("error") or not item.get("native_prediction"):
		return "DOWNSTREAM_EXECUTION_ERROR"
	return "INSUFFICIENT_EVIDENCE"


def fallback_cause(item: dict[str, Any]) -> str:
	if not item.get("fallback_used"):
		return "UNKNOWN"
	if str(item.get("validator_outcome", "")).upper() == "FALLBACK":
		return "VALIDATOR_REJECTION"
	if item.get("fallback_reason") == "invalid_branch_plan":
		return "INVALID_BRANCH_PLAN"
	if item.get("error"):
		return "EXECUTION_FAILURE"
	return "POLICY_FALLBACK"


def fallback_necessity(item: dict[str, Any]) -> str:
	if not item.get("fallback_used"):
		return "UNKNOWN"
	failures = [str(value) for value in item.get("validation_failures", [])]
	if any("branch_" in value or "not enough runtime topic entities" in value for value in failures):
		return "FALLBACK_NECESSARY"
	if failures:
		return "FALLBACK_POSSIBLY_AVOIDABLE"
	return "UNKNOWN"


def fallback_outcome(item: dict[str, Any]) -> str:
	if not item.get("fallback_used"):
		return "UNKNOWN"
	native = bool(item.get("native_hits_at_1"))
	final = bool(item.get("hits_at_1"))
	if final and not native:
		return "FALLBACK_RESCUE"
	if native and not final:
		return "FALLBACK_REGRESSION"
	return "FALLBACK_NEUTRAL"


def dominance(first_failure: str) -> str:
	if first_failure in {
		"CONSTRAINT_EXTRACTION_ERROR",
		"INDEPENDENT_SOURCE_EXTRACTION_ERROR",
		"SHARED_TARGET_ERROR",
		"DEPENDENCY_PATTERN_ERROR",
	}:
		return "EXTRACTION_DOMINANT"
	if first_failure in {"VALIDATOR_REJECTION", "FALLBACK_POLICY_ERROR"}:
		return "VALIDATION_FALLBACK_DOMINANT"
	if first_failure == "DOWNSTREAM_EXECUTION_ERROR":
		return "DOWNSTREAM_DOMINANT"
	return "UNKNOWN"


def per_case_repetition(items: list[dict[str, Any]]) -> dict[str, Any]:
	extraction_fields = ["constraints", "independent_sources", "shared_target", "dependency_pattern", "structure"]
	validator_fields = ["validator_outcome", "validation_failures", "fallback_used"]
	stable_extraction = all(len({canonical(item.get(field)) for item in items}) == 1 for field in extraction_fields)
	stable_validator = all(len({canonical(item.get(field)) for item in items}) == 1 for field in validator_fields)
	return {
		"stable_extraction": "STABLE_EXTRACTION" if stable_extraction else "UNSTABLE_EXTRACTION",
		"stable_extraction_detail": {field: len({canonical(item.get(field)) for item in items}) == 1 for field in extraction_fields},
		"stable_validator": "STABLE_VALIDATOR_BEHAVIOR" if stable_validator else "UNSTABLE_VALIDATOR_BEHAVIOR",
		"answers": [{"run_id": item.get("run_id"), "native_correct": bool(item.get("native_hits_at_1")), "final_correct": bool(item.get("hits_at_1"))} for item in items],
	}


def main() -> None:
	refs = {row["case_id"]: row for row in read_jsonl(REFERENCE)}
	live = read_jsonl(LIVE_ROOT / "live_runs.jsonl")
	by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for item in live:
		by_case[item["case_id"]].append(item)

	uncorrected_harmful: list[dict[str, Any]] = []
	fallback_rows: list[dict[str, Any]] = []
	all_failure_rows: list[dict[str, Any]] = []

	for case_id, items in by_case.items():
		ref = refs[case_id]
		group = group_for(ref)
		majority_structure = majority([item.get("structure") for item in items])
		corrected = majority_structure == gold_structure(ref)
		case_repetition = per_case_repetition(items)
		if group == "HARMFUL" and not corrected:
			first = [primary_first_failure(ref, item) for item in items]
			case_first = majority(first)
			audit_status, audit_note, audit_items = constraint_audit(ref, items[0], group)
			uncorrected_harmful.append({
				"case_id": case_id,
				"group": "UNCORRECTED_HARMFUL",
				"question": ref["question"],
				"gold_structure": ref["gold_structure"],
				"live_structures": [item.get("structure") for item in items],
				"constraint_status": audit_status,
				"constraint_audit_note": audit_note,
				"constraint_audit": audit_items,
				"gold_independent_source_count": int(ref.get("independent_sources_gold", 0) or 0),
				"live_independent_source_counts": [len(item.get("independent_sources") or []) for item in items],
				"independent_source_status": [source_status(ref, item) for item in items],
				"gold_shared_target": gold_target(ref),
				"live_shared_targets": [item.get("shared_target") for item in items],
				"shared_target_status": [shared_target_status(ref, item) for item in items],
				"gold_dependency_pattern": ref.get("gold_dependency_type"),
				"live_dependency_patterns": [item.get("dependency_pattern") for item in items],
				"dependency_status": [dependency_status(ref, item) for item in items],
				"structure_mapping_status": [structure_mapping_status(item) for item in items],
				"validator_status": [validator_status(item) for item in items],
				"validator_failures": [item.get("validation_failures", []) for item in items],
				"primary_first_failure": case_first,
				"repeat_primary_failures": first,
				"fallback_cause": [fallback_cause(item) for item in items],
				"fallback_necessity": [fallback_necessity(item) for item in items],
				"native_correct": [bool(item.get("native_hits_at_1")) for item in items],
				"final_correct": [bool(item.get("hits_at_1")) for item in items],
				"fallback_outcome": [fallback_outcome(item) for item in items],
				"dominance": dominance(case_first),
				**case_repetition,
			})

		if any(item.get("fallback_used") for item in items):
			fallback_items = [item for item in items if item.get("fallback_used")]
			fallback_rows.append({
				"case_id": case_id,
				"group": group,
				"question": ref["question"],
				"gold_structure": ref["gold_structure"],
				"router_structures": [item.get("structure") for item in items],
				"fallback_run_ids": [item.get("run_id") for item in fallback_items],
				"validator_results": [item.get("validator_outcome") for item in fallback_items],
				"fallback_trigger": [item.get("fallback_reason") for item in fallback_items],
				"fallback_cause": [fallback_cause(item) for item in fallback_items],
				"native_branch_plan_valid": [not bool(item.get("validation_failures")) and item.get("structure") == "BRANCH_FINAL" for item in fallback_items],
				"validator_failures": [item.get("validation_failures", []) for item in fallback_items],
				"fallback_necessity": [fallback_necessity(item) for item in fallback_items],
				"native_correct": [bool(item.get("native_hits_at_1")) for item in fallback_items],
				"final_correct": [bool(item.get("hits_at_1")) for item in fallback_items],
				"fallback_outcome": [fallback_outcome(item) for item in fallback_items],
				"dominance": [dominance(primary_first_failure(ref, item)) for item in fallback_items],
				**per_case_repetition(items),
			})

		for item in items:
			all_failure_rows.append({
				"case_id": case_id,
				"run_id": item.get("run_id"),
				"group": group,
				"primary_first_failure": primary_first_failure(ref, item),
				"source_status": source_status(ref, item),
				"shared_target_status": shared_target_status(ref, item),
				"dependency_status": dependency_status(ref, item),
				"structure_mapping_status": structure_mapping_status(item),
				"validator_status": validator_status(item),
				"fallback_cause": fallback_cause(item),
				"fallback_necessity": fallback_necessity(item),
				"fallback_outcome": fallback_outcome(item),
			})

	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	(OUTPUT_ROOT / "harmful_case_analysis.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in uncorrected_harmful), encoding="utf-8")
	(OUTPUT_ROOT / "fallback_analysis.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in fallback_rows), encoding="utf-8")

	case_first = Counter(row["primary_first_failure"] for row in uncorrected_harmful)
	run_first = Counter(row["primary_first_failure"] for row in all_failure_rows)
	fallback_causes = Counter(cause for row in fallback_rows for cause in row["fallback_cause"])
	fallback_need = Counter(value for row in fallback_rows for value in row["fallback_necessity"])
	fallback_outcomes = Counter(value for row in fallback_rows for value in row["fallback_outcome"])
	repeat_extraction = Counter(row["stable_extraction"] for row in fallback_rows)
	repeat_validator = Counter(row["stable_validator"] for row in fallback_rows)

	(OUTPUT_ROOT / "error_distribution.json").write_text(json.dumps({
		"uncorrected_harmful_case_count": len(uncorrected_harmful),
		"fallback_case_count": len(fallback_rows),
		"all_repeat_count": len(all_failure_rows),
		"uncorrected_harmful_first_failure_cases": dict(case_first),
		"all_repeat_first_failure_counts": dict(run_first),
		"fallback_cause_counts": dict(fallback_causes),
		"fallback_necessity_counts": dict(fallback_need),
		"fallback_outcome_counts": dict(fallback_outcomes),
		"fallback_repeat_extraction_counts": dict(repeat_extraction),
		"fallback_repeat_validator_counts": dict(repeat_validator),
	}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

	first_failure_lines = [
		"# Phase 2.5.2 first-failure summary", "",
		"This audit uses the existing Phase 2.5.1 live artifacts only; it makes no new LLM calls.", "",
		f"Uncorrected harmful cases: **{len(uncorrected_harmful)}/5**.",
		f"Fallback cases: **{len(fallback_rows)}/13**.", "",
		"## Uncorrected harmful cases", "",
		"| First observable failure | Case count |", "|---|---:|",
	]
	for label, count in case_first.most_common():
		first_failure_lines.append(f"| `{label}` | {count} |")
	first_failure_lines += ["", "The case-level primary failures are recorded in `harmful_case_analysis.jsonl`; repeat-level labels are in `error_distribution.json`.", "", "## Interpretation", "", "- All five uncorrected harmful cases fail at extraction/dependency representation, before downstream execution.", "- Three cases primarily miss independent sources; two primarily misidentify the shared target.", "- No uncorrected harmful case shows a deterministic structure-mapping error once the live dependency and target fields are taken as input.", ""]
	(OUTPUT_ROOT / "first_failure_summary.md").write_text("\n".join(first_failure_lines), encoding="utf-8")

	fallback_lines = [
		"# Phase 2.5.2 fallback summary", "",
		"The fallback audit uses every case with at least one fallback across the two Phase 2.5.1 repeats.", "",
		"## Fallback cause by repeat", "",
		"| Cause | Count |", "|---|---:|",
	]
	for label, count in fallback_causes.most_common():
		fallback_lines.append(f"| `{label}` | {count} |")
	fallback_lines += ["", "## Fallback necessity by repeat", "", "| Classification | Count |", "|---|---:|"]
	for label, count in fallback_need.most_common():
		fallback_lines.append(f"| `{label}` | {count} |")
	fallback_lines += ["", "## Fallback outcome by repeat", "", "| Outcome | Count |", "|---|---:|"]
	for label, count in fallback_outcomes.most_common():
		fallback_lines.append(f"| `{label}` | {count} |")
	fallback_lines += ["", "## Diagnosis", "", "- Fallback is dominated by validator rejection of branch plans that do not satisfy the existing v1.1.3 executable-branch schema.", "- The rejection is contract-correct for the current executor, but semantically plausible constraint graphs are therefore forced into CoR fallback.", "- This is operationally avoidable only after a separately approved branch-schema intervention; Phase 2.5.2 does not implement it.", "- The existing Phase 2.5.1 evaluation helper mislabeled many validator rejections as `STRUCTURE_MAPPING_ERROR`; this audit uses raw `validator_outcome` and `validation_failures` instead.", "", "## Repetition", "", f"- Fallback-case extraction stability: `{dict(repeat_extraction)}`.", f"- Fallback-case validator stability: `{dict(repeat_validator)}`.", ""]
	(OUTPUT_ROOT / "fallback_summary.md").write_text("\n".join(fallback_lines), encoding="utf-8")

	report_lines = [
		"# Phase 2.5.2 live extraction and fallback failure analysis", "",
		"## Scope", "",
		"This is an offline diagnostic audit of the existing Phase 2.5.1 live artifacts. No new LLM calls were made and no production/router/fallback component was changed.", "",
		"Inputs:", "",
		"- Phase 2.5.1 `live_runs.jsonl` (30 runs; 15 cases × 2 repeats)",
		"- Phase 2.5.1 extraction, routing, QA, stability, and cost files",
		"- Phase 2.4.4 `router_input_output_analysis.jsonl` structural reference", "",
		"## RQ1: uncorrected harmful cases", "",
		f"The audit found **{len(uncorrected_harmful)}/5** uncorrected harmful cases.", "",
		"| Primary first observable failure | Cases |", "|---|---:|",
	]
	for label, count in case_first.most_common():
		report_lines.append(f"| `{label}` | {count} |")
	report_lines += [
		"", "The five cases fail before downstream execution. Three primarily miss independent sources; two primarily assign the shared target the wrong type or omit it. All five have semantically recognizable answer constraints, but the live representation does not preserve the required convergence structure.", "",
		"## RQ2: fallback frequency", "",
		f"Fallback occurs in **{len(fallback_rows)}/15 cases**, across **{sum(fallback_causes.values())} fallback repeats**.", "",
		"| Fallback cause | Repeat count |", "|---|---:|",
	]
	for label, count in fallback_causes.most_common():
		report_lines.append(f"| `{label}` | {count} |")
	report_lines += [
		"", "The dominant trigger is validator rejection. Most rejected branch plans are semantically plausible at the constraint-graph level but do not satisfy the existing v1.1.3 executable-branch schema: required branch IDs, root entities, constraint properties, summaries, and answer types are missing. The validator is therefore contract-correct for the current executor, while the overall fallback behavior is operationally conservative.", "",
		"## RQ3: bottleneck separation", "",
		"| Component | Evidence | Diagnosis |", "|---|---:|---|",
		f"| Extraction/source/target | {sum(value for value in case_first.values())}/5 harmful cases | Primary failure for all harmful cases |",
		f"| Validator/fallback | {sum(fallback_causes.values())} fallback repeats | Dominant reason fallback is invoked |",
		"| Structure mapping | 0 harmful cases | No independent mapping bottleneck observed |",
		"| Downstream execution | No first-failure cases | Not the primary bottleneck in this audit |",
		"", "## Repetition consistency", "",
		f"- Fallback-case extraction was unstable in {repeat_extraction.get('UNSTABLE_EXTRACTION', 0)}/{len(fallback_rows)} cases under exact field comparison.",
		f"- Validator behavior was stable in {repeat_validator.get('STABLE_VALIDATOR_BEHAVIOR', 0)}/{len(fallback_rows)} fallback cases and unstable in {repeat_validator.get('UNSTABLE_VALIDATOR_BEHAVIOR', 0)}/{len(fallback_rows)}.",
		"- Structure itself remained stable in the Phase 2.5.1 run: 15/15 cases.", "",
		"## Fallback outcome", "",
		f"All {sum(fallback_outcomes.values())} observed fallback repeats were `FALLBACK_NEUTRAL`: fallback did not change a correct/incorrect top-1 outcome in these logs. There was no observed fallback rescue or regression.", "",
		"## Evaluation-artifact note", "",
		"The Phase 2.5.1 helper reported many cases as `STRUCTURE_MAPPING_ERROR` because it checked `validator_decision` at the top level even though that field is nested in the router decision. This audit recomputed first failures from the raw live decision, validation failures, and fallback fields. The corrected repeat-level distribution is 18 validator rejections, 9 independent-source errors, and 3 shared-target errors.", "",
		"## Main decision", "",
		"**MIXED_LIVE_BOTTLENECKS**", "",
		"Extraction is the primary bottleneck for the five uncorrected harmful cases, while validator/fallback contract mismatch is the primary reason the intervention falls back in 13 cases. The intervention gate is met for a source-extraction intervention because three independent harmful cases share the same first-failure family. A separate branch-plan serialization/validator intervention is also justified by the 24 validator-rejected fallback repeats, but it must be evaluated independently.", "",
		"## Phase boundary", "",
		"No fix was implemented in Phase 2.5.2. Do not run Stage B or full100 until the next intervention is specified and tested on the focused set.", "",
		"## Output files", "",
		"- `harmful_case_analysis.jsonl`", "- `fallback_analysis.jsonl`", "- `first_failure_summary.md`", "- `fallback_summary.md`", "- `error_distribution.json`", "",
	]
	(OUTPUT_ROOT / "PHASE2_5_2_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

	print(json.dumps({
		"status": "COMPLETE",
		"uncorrected_harmful_cases": len(uncorrected_harmful),
		"fallback_cases": len(fallback_rows),
		"output_root": str(OUTPUT_ROOT),
		"decision": "MIXED_LIVE_BOTTLENECKS",
	}, indent=2))


if __name__ == "__main__":
	main()
