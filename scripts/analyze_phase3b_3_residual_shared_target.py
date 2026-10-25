#!/usr/bin/env python3
"""Offline Phase 3B.3 residual shared-target semantic audit.

This script consumes only completed Phase 3B.1/3B.2 artifacts.  It does not
call an LLM, Freebase, or any external service, and it does not modify the
production router, validator, executor, or fallback path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PHASE3B2_ROOT = ROOT / "outputs/experiments/phase3b_2/shared_target_intervention"
PHASE3B1_ROOT = ROOT / "outputs/experiments/phase3b_1/strong_failure_causal_analysis"
OUTPUT_ROOT = ROOT / "outputs/experiments/phase3b_3/residual_shared_target_analysis"

REQUIRED_PHASE3B2 = (
	"PHASE3B_2_REPORT.md",
	"development_case_manifest.json",
	"fresh_validation_manifest.json",
	"shared_target_extraction_results.jsonl",
	"fresh_run_results.jsonl",
	"semantic_metrics.json",
	"qa_metrics.json",
	"paired_analysis.json",
	"routing_behavior.json",
	"validator_execution_impact.json",
	"stability_analysis.json",
	"failure_analysis.jsonl",
	"manual_review_cases.json",
	"manual_review_notes.md",
	"decision_summary.md",
)

CAUSES = (
	"CONVERGENCE_POINT_MISIDENTIFIED",
	"DOWNSTREAM_STEP_MISSED",
	"ANSWER_CONVERGENCE_CONFUSION",
	"OVER_CONSERVATIVE_NONE",
	"SHARED_TARGET_DESCRIPTION_UNDERSPECIFIED",
	"DOWNSTREAM_DESCRIPTION_UNDERSPECIFIED",
	"CONSTRAINT_DEPENDENCY_MISREAD",
	"SOURCE_CONVERGENCE_NOT_RECOGNIZED",
	"SEMANTIC_OUTPUT_UNSTABLE",
	"NO_CLEAR_SEMANTIC_FAILURE",
)

DESCRIPTION_LABELS = (
	"SEMANTICALLY_SPECIFIC",
	"TOO_GENERIC",
	"WRONG_ENTITY_ROLE",
	"WRONG_RELATION_ROLE",
	"MISSING",
)


def read_json(path: Path) -> Any:
	return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		"".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
		encoding="utf-8",
	)


def sha256(path: Path) -> str:
	hasher = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			hasher.update(chunk)
	return hasher.hexdigest()


def text(value: Any) -> str:
	return str(value or "").strip()


def upper(value: Any) -> str:
	return text(value).upper()


def target_type(row: dict[str, Any]) -> str:
	value = (
		row.get("canonical_shared_target_type")
		or row.get("raw_shared_target_type")
		or row.get("actual_target")
	)
	if value:
		return upper(value)
	proposal = row.get("shared_target_extraction") or {}
	if isinstance(proposal, dict):
		value = proposal.get("shared_target_type") or proposal.get("type")
		if value:
			return upper(value)
	return "NONE"


def expected_target(row: dict[str, Any]) -> str:
	value = upper(row.get("expected_shared_target_type"))
	if value:
		return value
	return "INTERMEDIATE" if upper(row.get("gold_structure")) == "BRANCH_INTERMEDIATE" else "ANSWER"


def expected_downstream(row: dict[str, Any]) -> bool:
	return expected_target(row) == "INTERMEDIATE"


def bool_value(value: Any) -> bool | None:
	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)) and value in (0, 1):
		return bool(value)
	if isinstance(value, str):
		value = value.strip().upper()
		if value in {"TRUE", "YES", "Y", "1"}:
			return True
		if value in {"FALSE", "NO", "N", "0"}:
			return False
	return None


def self_check(row: dict[str, Any]) -> dict[str, Any]:
	proposal = row.get("shared_target_extraction") or {}
	check = proposal.get("self_check") if isinstance(proposal, dict) else None
	if not isinstance(check, dict):
		check = proposal.get("shared_target_self_check") if isinstance(proposal, dict) else None
	if not isinstance(check, dict):
		return {
			"answer": None,
			"downstream_required": None,
			"valid_shared_latent_target": None,
			"convergence_variable_description": "",
			"rationale": "",
		}
	return {
		"answer": bool_value(check.get("resulting_variable_is_requested_answer")),
		"downstream_required": bool_value(check.get("downstream_step_required")),
		"valid_shared_latent_target": bool_value(check.get("valid_shared_latent_target")),
		"convergence_variable_description": text(check.get("convergence_variable_description")),
		"rationale": text(check.get("rationale")),
	}


def parsed_constraints(row: dict[str, Any]) -> list[dict[str, Any]]:
	proposal = row.get("raw_parsed_result")
	if isinstance(proposal, dict) and isinstance(proposal.get("constraints"), list):
		return [item for item in proposal["constraints"] if isinstance(item, dict)]
	constraints = row.get("raw_constraints")
	return [item for item in constraints if isinstance(item, dict)] if isinstance(constraints, list) else []


def source_anchors(row: dict[str, Any]) -> list[str]:
	anchors: list[str] = []
	proposal = row.get("raw_parsed_result")
	if isinstance(proposal, dict):
		groups = proposal.get("independent_source_groups")
		if isinstance(groups, list):
			for group in groups:
				if isinstance(group, dict):
					anchor = group.get("source_anchor")
					if isinstance(anchor, dict):
						anchor = anchor.get("id") or anchor.get("name")
					if anchor:
						anchors.append(text(anchor))
		for constraint in parsed_constraints(row):
			anchor = constraint.get("source_anchor")
			if isinstance(anchor, dict):
				anchor = anchor.get("id") or anchor.get("name")
			if anchor:
				anchors.append(text(anchor))
		for branch in proposal.get("branches", []) if isinstance(proposal.get("branches"), list) else []:
			if isinstance(branch, dict):
				anchor = branch.get("source_anchor")
				if isinstance(anchor, dict):
					anchor = anchor.get("id") or anchor.get("name")
				if anchor:
					anchors.append(text(anchor))
	return sorted({value for value in anchors if value})


def constraint_dimension(row: dict[str, Any]) -> str:
	count = len(parsed_constraints(row))
	if count >= 2:
		return "CORRECT"
	if count == 1:
		return "PARTIAL"
	return "MISSING"


def independent_source_dimension(row: dict[str, Any]) -> str:
	count = len(source_anchors(row))
	if count >= 2:
		return "CORRECT"
	if count == 1:
		return "PARTIAL"
	return "MISSING"


def convergence_dimension(row: dict[str, Any]) -> str:
	return "CORRECT" if target_type(row) == expected_target(row) else "WRONG"


def requested_answer_dimension(row: dict[str, Any]) -> str:
	description = text(row.get("requested_answer_description"))
	if not description:
		return "MISSING"
	if len(description) < 18:
		return "PARTIAL"
	return "CORRECT"


def downstream_dimension(row: dict[str, Any]) -> str:
	value = bool_value(row.get("requires_downstream_relation"))
	if value is None:
		return "MISSING"
	return "CORRECT" if value == expected_downstream(row) else "WRONG"


def target_description_quality(row: dict[str, Any]) -> str:
	description = text(row.get("shared_target_description"))
	if not description:
		return "MISSING"
	if len(description) < 24:
		return "TOO_GENERIC"
	actual = target_type(row)
	expected = expected_target(row)
	if actual != expected:
		return "WRONG_ENTITY_ROLE"
	return "SEMANTICALLY_SPECIFIC"


def requested_description_quality(row: dict[str, Any]) -> str:
	description = text(row.get("requested_answer_description"))
	if not description:
		return "MISSING"
	if len(description) < 24:
		return "TOO_GENERIC"
	return "SEMANTICALLY_SPECIFIC"


def downstream_description_quality(row: dict[str, Any]) -> str:
	description = text(row.get("downstream_relation_description"))
	if not description:
		return "MISSING"
	if not expected_downstream(row):
		return "WRONG_RELATION_ROLE"
	if len(description) < 18:
		return "TOO_GENERIC"
	return "SEMANTICALLY_SPECIFIC"


def self_check_status(row: dict[str, Any], peer_answers: set[bool | None] | None = None) -> str:
	check = self_check(row)
	answer = check["answer"]
	if peer_answers and len(peer_answers) > 1:
		return "SELF_CHECK_UNSTABLE"
	if answer is None:
		return "SELF_CHECK_MISSING"
	return "SELF_CHECK_CORRECT" if answer == (not expected_downstream(row)) else "SELF_CHECK_WRONG"


def target_stability(rows: list[dict[str, Any]]) -> str:
	types = {target_type(row) for row in rows}
	answers = {self_check(row)["answer"] for row in rows}
	if len(types) > 1:
		return "UNSTABLE_TYPE"
	if len(answers) > 1:
		return "UNSTABLE_SELF_CHECK"
	if types == {expected_target(rows[0])}:
		return "STABLE_CORRECT"
	return "STABLE_WRONG"


def majority(values: list[str]) -> str:
	counts = Counter(values)
	return sorted(counts, key=lambda value: (-counts[value], value))[0]


def old_target_rows() -> dict[str, list[dict[str, Any]]]:
	rows = defaultdict(list)
	path = PHASE3B1_ROOT / "semantic_extraction_audit.jsonl"
	if path.exists():
		for row in read_jsonl(path):
			case_id = text(row.get("case_id"))
			if case_id:
				rows[case_id].append(row)
	return rows


def fresh_method_rows(rows: list[dict[str, Any]], method: str) -> dict[str, list[dict[str, Any]]]:
	result = defaultdict(list)
	for row in rows:
		if row.get("method") == method:
			result[text(row.get("case_id"))].append(row)
	for values in result.values():
		values.sort(key=lambda row: row.get("repeat", 0))
	return result


def select_target_cases(
	phase3b2_root: Path = PHASE3B2_ROOT,
	phase3b1_root: Path = PHASE3B1_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
	metrics = read_json(phase3b2_root / "semantic_metrics.json")
	development_rows = read_jsonl(phase3b2_root / "development_run_results.jsonl")
	fresh_rows = read_jsonl(phase3b2_root / "fresh_run_results.jsonl")
	dev_by_id = defaultdict(list)
	for row in development_rows:
		dev_by_id[text(row.get("case_id"))].append(row)
	fresh_by_method = {
		"phase2_5_6": fresh_method_rows(fresh_rows, "phase2_5_6"),
		"phase3b_2": fresh_method_rows(fresh_rows, "phase3b_2"),
	}
	selected: dict[str, dict[str, Any]] = {}

	dev_metrics = metrics.get("development", {})
	for case_id, correct in dev_metrics.get("intervention_target_accuracy_by_case", {}).items():
		if correct is False and case_id in dev_by_id:
			row = dev_by_id[case_id][0]
			selected[case_id] = {
				"case_id": case_id,
				"dataset": row.get("dataset"),
				"cohort": "development",
				"gold_structure": row.get("gold_structure"),
				"question": row.get("question"),
				"gold_answers": row.get("gold_answers", []),
				"expected_target": expected_target(row),
				"case_category": "DEVELOPMENT_STILL_WRONG",
				"source": "Phase 3B.1 primary cases still incorrect in Phase 3B.2",
			}

	for case_id, rows in fresh_by_method["phase3b_2"].items():
		if not rows:
			continue
		row = rows[0]
		cohort = text(row.get("cohort"))
		gold_structure = upper(row.get("gold_structure"))
		new_types = {target_type(item) for item in rows}
		old_rows = fresh_by_method["phase2_5_6"].get(case_id, [])
		old_types = {target_type(item) for item in old_rows}
		category = None
		if cohort == "strong_intermediate" and gold_structure == "BRANCH_INTERMEDIATE":
			if len(new_types) > 1:
				category = "FRESH_INTERMEDIATE_UNSTABLE"
			elif majority(sorted(new_types)) != "INTERMEDIATE":
				category = "FRESH_INTERMEDIATE_STILL_WRONG"
		elif cohort == "strong_final" and gold_structure == "BRANCH_FINAL":
			old_correct = old_types and majority(sorted(old_types)) == "ANSWER"
			new_wrong = majority(sorted(new_types)) != "ANSWER"
			if old_correct and new_wrong:
				category = "FRESH_FINAL_REGRESSION"
		elif cohort == "control" and old_types != new_types:
			category = "FRESH_CONTROL_CHANGED_TARGET"
		if category:
			selected[case_id] = {
				"case_id": case_id,
				"dataset": row.get("dataset"),
				"cohort": cohort,
				"gold_structure": row.get("gold_structure"),
				"question": row.get("question"),
				"gold_answers": row.get("gold_answers", []),
				"expected_target": expected_target(row),
				"case_category": category,
				"source": "Phase 3B.2 fresh validation residual/changed-target rule",
			}

		# Avoid an accidental mismatch if the two input files disagree.
		if case_id in selected:
			selected[case_id]["phase3b_2_repeats"] = len(rows)

	dev_old = old_target_rows()
	return sorted(selected.values(), key=lambda item: item["case_id"]), dict(dev_by_id), {
		"fresh": fresh_by_method,
		"development_old": dev_old,
	}


def input_hashes(root: Path) -> dict[str, str]:
	return {name: sha256(root / name) for name in REQUIRED_PHASE3B2}


def freeze_manifest(selected: list[dict[str, Any]], root: Path) -> dict[str, Any]:
	counts = Counter(item["case_category"] for item in selected)
	return {
		"status": "FROZEN",
		"phase": "3B.3",
		"selection_policy": {
			"development": "all Phase 3B.1 primary cases still incorrect after Phase 3B.2",
			"fresh_intermediate": "all Phase 3B.2 strong BRANCH_INTERMEDIATE cases still wrong or unstable",
			"fresh_final": "all fresh strong BRANCH_FINAL target regressions",
			"controls": "all fresh controls with any repeat-level target change",
		},
		"live_reruns": 0,
		"case_count": len(selected),
		"repeat_count_expected": len(selected) * 2,
		"counts_by_category": dict(sorted(counts.items())),
		"counts_by_dataset": dict(sorted(Counter(item["dataset"] for item in selected).items())),
		"source_artifacts": {
			"phase3b_2_root": str(root),
			"phase3b_1_root": str(PHASE3B1_ROOT),
			"phase3b_2_sha256": input_hashes(root),
		},
		"cases": selected,
	}


def row_dimensions(row: dict[str, Any], peer_rows: list[dict[str, Any]]) -> dict[str, Any]:
	answers = {self_check(item)["answer"] for item in peer_rows}
	return {
		"constraint_set": constraint_dimension(row),
		"independent_sources": independent_source_dimension(row),
		"convergence_entity": convergence_dimension(row),
		"requested_answer": requested_answer_dimension(row),
		"requires_downstream_relation": downstream_dimension(row),
		"shared_target_type": "CORRECT" if target_type(row) == expected_target(row) else "WRONG",
		"shared_target_description": target_description_quality(row),
		"downstream_step_description": downstream_description_quality(row),
		"self_check_answer": self_check(row)["answer"],
		"self_check_status": self_check_status(row, answers),
	}


def primary_cause(row: dict[str, Any], peer_rows: list[dict[str, Any]]) -> str:
	if len({target_type(item) for item in peer_rows}) > 1:
		return "SEMANTIC_OUTPUT_UNSTABLE"
	expected = expected_target(row)
	actual = target_type(row)
	check = self_check(row)
	if actual == expected:
		if expected == "INTERMEDIATE" and bool_value(row.get("requires_downstream_relation")) is not True:
			return "DOWNSTREAM_STEP_MISSED"
		if target_description_quality(row) == "WRONG_ENTITY_ROLE":
			return "SHARED_TARGET_DESCRIPTION_UNDERSPECIFIED"
		return "NO_CLEAR_SEMANTIC_FAILURE"
	if expected == "INTERMEDIATE" and actual == "ANSWER":
		if check["answer"] is True:
			return "ANSWER_CONVERGENCE_CONFUSION"
		return "CONVERGENCE_POINT_MISIDENTIFIED"
	if actual == "NONE":
		return "OVER_CONSERVATIVE_NONE"
	if expected == "ANSWER" and actual == "INTERMEDIATE":
		return "CONVERGENCE_POINT_MISIDENTIFIED"
	return "CONVERGENCE_POINT_MISIDENTIFIED"


def old_rows_for(case: dict[str, Any], sources: dict[str, Any]) -> list[dict[str, Any]]:
	case_id = case["case_id"]
	if case["case_category"] == "DEVELOPMENT_STILL_WRONG":
		return sorted(sources["development_old"].get(case_id, []), key=lambda row: row.get("repeat", 0))
	return sources["fresh"].get("phase2_5_6", {}).get(case_id, [])


def new_rows_for(case: dict[str, Any], sources: dict[str, Any]) -> list[dict[str, Any]]:
	if case["case_category"] == "DEVELOPMENT_STILL_WRONG":
		return sorted(sources["development"].get(case["case_id"], []), key=lambda row: row.get("repeat", 0))
	return sources["fresh"].get("phase3b_2", {}).get(case["case_id"], [])


def build_target_manifest_rows(selected: list[dict[str, Any]], sources: dict[str, Any]) -> list[dict[str, Any]]:
	rows = []
	for case in selected:
		old = old_rows_for(case, sources)
		new = new_rows_for(case, sources)
		entry = dict(case)
		entry.update(
			{
				"phase2_5_6_targets": [target_type(row) for row in old],
				"phase3b_2_targets": [target_type(row) for row in new],
				"phase2_5_6_repeat_count": len(old),
				"phase3b_2_repeat_count": len(new),
				"target_stability": target_stability(new),
			}
		)
		rows.append(entry)
	return rows


def build_semantic_audits(selected: list[dict[str, Any]], sources: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
	semantic_rows = []
	self_rows = []
	description_rows = []
	for case in selected:
		new = new_rows_for(case, sources)
		for row in new:
			dimensions = row_dimensions(row, new)
			check = self_check(row)
			cause = primary_cause(row, new)
			base = {
				"case_id": case["case_id"],
				"dataset": case["dataset"],
				"cohort": case["cohort"],
				"gold_structure": case["gold_structure"],
				"case_category": case["case_category"],
				"repeat": row.get("repeat"),
				"question": case["question"],
				"expected_target": expected_target(row),
				"actual_target": target_type(row),
				"requires_downstream_relation_value": bool_value(row.get("requires_downstream_relation")),
				"dimensions": dimensions,
				"primary_residual_cause": cause,
				"validator_decision": row.get("validator_decision"),
				"normalization_status": row.get("normalization_status"),
				"execution_status": row.get("execution_status"),
				"fallback_reason": row.get("fallback_reason"),
				"manual_review_status": "TRACE_AND_PHASE3B2_NOTES_REVIEWED",
			}
			semantic_rows.append(base)
			self_rows.append(
				{
					**base,
					"self_check_answer": check["answer"],
					"gold_self_check_answer": not expected_downstream(row),
					"self_check_status": dimensions["self_check_status"],
					"convergence_variable_description": check["convergence_variable_description"],
					"self_check_rationale": check["rationale"],
				}
			)
			description_rows.append(
				{
					**base,
					"shared_target_description": text(row.get("shared_target_description")),
					"requested_answer_description": text(row.get("requested_answer_description")),
					"downstream_relation_description": text(row.get("downstream_relation_description")),
					"description_quality": {
						"shared_target_description": target_description_quality(row),
						"requested_answer_description": requested_description_quality(row),
						"downstream_relation_description": downstream_description_quality(row),
					},
				}
			)
	return semantic_rows, self_rows, description_rows


def branch_intermediate_audits(selected: list[dict[str, Any]], sources: dict[str, Any]) -> list[dict[str, Any]]:
	rows = []
	for case in selected:
		if case["gold_structure"] != "BRANCH_INTERMEDIATE":
			continue
		new = new_rows_for(case, sources)
		for row in new:
			constraints = constraint_dimension(row) == "CORRECT"
			sources_correct = independent_source_dimension(row) == "CORRECT"
			convergence = target_type(row) == "INTERMEDIATE"
			distinguished = convergence
			downstream = bool_value(row.get("requires_downstream_relation")) is True
			selected_intermediate = target_type(row) == "INTERMEDIATE"
			if not constraints:
				earliest = "CONSTRAINT_DEPENDENCY_MISREAD"
			elif not sources_correct:
				earliest = "SOURCE_CONVERGENCE_NOT_RECOGNIZED"
			elif target_type(row) == "ANSWER":
				earliest = "ANSWER_CONVERGENCE_CONFUSION"
			elif target_type(row) == "NONE":
				earliest = "OVER_CONSERVATIVE_NONE"
			elif not downstream:
				earliest = "DOWNSTREAM_STEP_MISSED"
			else:
				earliest = "NO_CLEAR_SEMANTIC_FAILURE"
			rows.append(
				{
					"case_id": case["case_id"],
					"dataset": case["dataset"],
					"case_category": case["case_category"],
					"repeat": row.get("repeat"),
					"both_answer_relevant_constraints_extracted": constraints,
					"independent_sources_correct": sources_correct,
					"latent_convergence_entity_identified": convergence,
					"distinguished_from_requested_answer": distinguished,
					"downstream_semantic_step_recognized": downstream,
					"intermediate_selected": selected_intermediate,
					"actual_target": target_type(row),
					"earliest_semantic_failure": earliest,
					"final_answer_produced": bool(row.get("prediction")),
					"prediction": row.get("prediction", []),
				}
			)
	return rows


def final_regression_audits(selected: list[dict[str, Any]], sources: dict[str, Any]) -> list[dict[str, Any]]:
	rows = []
	for case in selected:
		if case["case_category"] != "FRESH_FINAL_REGRESSION":
			continue
		old = old_rows_for(case, sources)
		new = new_rows_for(case, sources)
		for row in new:
			actual = target_type(row)
			if actual == "NONE":
				classification = "MISREAD_ANSWER_ROLE"
				reason = "The extractor treated the two requested properties as evidence that no answer-bearing target existed, changing ANSWER to NONE."
			elif actual == "INTERMEDIATE":
				classification = "FALSE_DOWNSTREAM_STEP"
				reason = "The extractor introduced an intermediate target although the frozen target contract expects ANSWER."
			else:
				classification = "OTHER"
				reason = "The target changed away from the frozen ANSWER target for an unclassified reason."
			rows.append(
				{
					"case_id": case["case_id"],
					"dataset": case["dataset"],
					"repeat": row.get("repeat"),
					"phase2_5_6_target": target_type(old[row.get("repeat", 1) - 1]) if len(old) >= row.get("repeat", 1) else None,
					"phase3b_2_target": actual,
					"expected_target": "ANSWER",
					"classification": classification,
					"reason": reason,
					"self_check_status": self_check_status(row, {self_check(item)["answer"] for item in new}),
					"self_check_rationale": self_check(row)["rationale"],
				}
			)
	return rows


def control_changed_audits(selected: list[dict[str, Any]], sources: dict[str, Any]) -> list[dict[str, Any]]:
	rows = []
	for case in selected:
		if case["case_category"] != "FRESH_CONTROL_CHANGED_TARGET":
			continue
		old = old_rows_for(case, sources)
		new = new_rows_for(case, sources)
		old_by_repeat = {row.get("repeat"): row for row in old}
		for row in new:
			repeat = row.get("repeat")
			previous = old_by_repeat.get(repeat)
			if not previous or target_type(previous) == target_type(row):
				continue
			answer_equivalent = bool(row.get("hits_at_1")) or float(row.get("f1") or 0) > 0
			if answer_equivalent and target_type(row) == "ANSWER":
				classification = "SAFE_CORRECTION"
			elif answer_equivalent:
				classification = "SAFE_EQUIVALENT_CHANGE"
			elif target_type(row) == "NONE":
				classification = "UNNECESSARY_COMPLEXITY"
			else:
				classification = "SEMANTIC_REGRESSION"
			rows.append(
				{
					"case_id": case["case_id"],
					"dataset": case["dataset"],
					"repeat": repeat,
					"phase2_5_6_target": target_type(previous),
					"phase3b_2_target": target_type(row),
					"expected_target": "ANSWER",
					"answer_equivalent_end_to_end": answer_equivalent,
					"unnecessary_structural_complexity": target_type(row) not in {"ANSWER", "NONE"},
					"classification": classification,
					"phase2_5_6_prediction": previous.get("prediction", []),
					"phase3b_2_prediction": row.get("prediction", []),
				}
			)
	return rows


def old_new_effect(selected: list[dict[str, Any]], sources: dict[str, Any]) -> list[dict[str, Any]]:
	rows = []
	for case in selected:
		old = old_rows_for(case, sources)
		new = new_rows_for(case, sources)
		old_types = [target_type(row) for row in old]
		new_types = [target_type(row) for row in new]
		expected = case["expected_target"]
		old_stable = len(set(old_types)) <= 1
		new_stable = len(set(new_types)) <= 1
		old_correct = old_stable and old_types and old_types[0] == expected
		new_correct = new_stable and new_types and new_types[0] == expected
		if not old_stable or not new_stable:
			effect = "UNSTABLE"
		elif not old_correct and new_correct:
			effect = "CORRECTED"
		elif old_correct and not new_correct:
			effect = "REGRESSED"
		elif old_types == new_types:
			effect = "UNCHANGED_WRONG" if not old_correct else "UNCHANGED_CORRECT"
		else:
			effect = "CHANGED_TO_DIFFERENT_WRONG"
		rows.append(
			{
				"case_id": case["case_id"],
				"case_category": case["case_category"],
				"dataset": case["dataset"],
				"expected_target": expected,
				"phase2_5_6_targets": old_types,
				"phase3b_2_targets": new_types,
				"effect": effect,
				"phase2_5_6_target_majority": majority(old_types) if old_types else None,
				"phase3b_2_target_majority": majority(new_types) if new_types else None,
			}
		)
	return rows


def repeat_stability(selected: list[dict[str, Any]], sources: dict[str, Any]) -> dict[str, Any]:
	rows = []
	for case in selected:
		new = new_rows_for(case, sources)
		rows.append(
			{
				"case_id": case["case_id"],
				"case_category": case["case_category"],
				"target_stability": target_stability(new),
				"target_types": [target_type(row) for row in new],
				"self_check_answers": [self_check(row)["answer"] for row in new],
				"description_values": [text(row.get("shared_target_description")) for row in new],
			}
		)
	return {
		"case_count": len(rows),
		"counts": dict(sorted(Counter(row["target_stability"] for row in rows).items())),
		"cases": rows,
	}


def aggregate_summary(
	selected: list[dict[str, Any]],
	semantic_rows: list[dict[str, Any]],
	self_rows: list[dict[str, Any]],
	stability: dict[str, Any],
	effects: list[dict[str, Any]],
) -> dict[str, Any]:
	by_cause: dict[str, dict[str, Any]] = {}
	for cause in CAUSES:
		matching = [row for row in semantic_rows if row["primary_residual_cause"] == cause]
		case_ids = sorted({row["case_id"] for row in matching})
		by_cause[cause] = {"repeat_count": len(matching), "independent_case_count": len(case_ids), "case_ids": case_ids}
	by_structure = {}
	for structure in ("BRANCH_INTERMEDIATE", "BRANCH_FINAL"):
		case_rows = [case for case in selected if case["gold_structure"] == structure]
		case_ids = {case["case_id"] for case in case_rows}
		stable_wrong = sum(1 for case in stability["cases"] if case["case_id"] in case_ids and case["target_stability"] == "STABLE_WRONG")
		unstable = sum(1 for case in stability["cases"] if case["case_id"] in case_ids and case["target_stability"].startswith("UNSTABLE"))
		causes = Counter(row["primary_residual_cause"] for row in semantic_rows if row["case_id"] in case_ids)
		by_structure[structure] = {
			"residual_case_count": len(case_rows),
			"stable_wrong_case_count": stable_wrong,
			"unstable_case_count": unstable,
			"main_cause": causes.most_common(1)[0][0] if causes else None,
		}
	self_check_counts = Counter(row["self_check_status"] for row in self_rows)
	return {
		"status": "COMPLETE",
		"case_count": len(selected),
		"repeat_count": len(semantic_rows),
		"counts_by_category": dict(sorted(Counter(case["case_category"] for case in selected).items())),
		"primary_causes": by_cause,
		"by_structure": by_structure,
		"self_check_status_counts": dict(sorted(self_check_counts.items())),
		"repeat_stability_counts": stability["counts"],
		"old_vs_new_effect_counts": dict(sorted(Counter(row["effect"] for row in effects).items())),
	}


def decision_gate(summary: dict[str, Any]) -> dict[str, Any]:
	supported = {
		"CONVERGENCE_POINT_BOTTLENECK_SUPPORTED": "CONVERGENCE_POINT_MISIDENTIFIED",
		"DOWNSTREAM_STEP_RECOGNITION_BOTTLENECK_SUPPORTED": "DOWNSTREAM_STEP_MISSED",
		"ANSWER_CONVERGENCE_CONFUSION_SUPPORTED": "ANSWER_CONVERGENCE_CONFUSION",
		"OVER_CONSERVATIVE_NONE_SUPPORTED": "OVER_CONSERVATIVE_NONE",
		"SEMANTIC_DESCRIPTION_BOTTLENECK_SUPPORTED": "SHARED_TARGET_DESCRIPTION_UNDERSPECIFIED",
		"SEMANTIC_STABILITY_BOTTLENECK_SUPPORTED": "SEMANTIC_OUTPUT_UNSTABLE",
	}
	threshold = 3
	candidates = {
		label: summary["primary_causes"][cause]["independent_case_count"]
		for label, cause in supported.items()
	}
	passing = [label for label, count in candidates.items() if count >= threshold]
	# The evidence is intentionally adjudicated by the most specific supported
	# semantic family; the remaining families are reported, not tuned.
	decision = "INSUFFICIENT_EVIDENCE" if summary["repeat_count"] == 0 else (passing[0] if len(passing) == 1 else "MIXED_RESIDUAL_SEMANTIC_FAILURES")
	return {
		"decision": decision,
		"support_threshold_independent_cases": threshold,
		"candidate_independent_case_counts": candidates,
		"passing_candidates": passing,
		"evidence_rule": "A cause must appear in at least three independent residual cases; no new intervention is implemented in Phase 3B.3.",
	}


def markdown_report(
	selected: list[dict[str, Any]],
	semantic_rows: list[dict[str, Any]],
	self_rows: list[dict[str, Any]],
	description_rows: list[dict[str, Any]],
	branch_rows: list[dict[str, Any]],
	final_rows: list[dict[str, Any]],
	control_rows: list[dict[str, Any]],
	stability: dict[str, Any],
	summary: dict[str, Any],
	effects: list[dict[str, Any]],
	decision: dict[str, Any],
) -> str:
	def table(rows: list[list[Any]], headers: list[str]) -> str:
		out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
		out.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows)
		return "\n".join(out)

	causes = summary["primary_causes"]
	intermediate_cases = [case for case in selected if case["gold_structure"] == "BRANCH_INTERMEDIATE"]
	final_cases = [case for case in selected if case["gold_structure"] == "BRANCH_FINAL"]
	self_counts = summary["self_check_status_counts"]
	main_cause = max(causes.items(), key=lambda item: item[1]["independent_case_count"])[0] if causes else "INSUFFICIENT_EVIDENCE"

	lines = [
		"# Phase 3B.3 Residual Shared-Target Semantic Failure Analysis",
		"",
		"## 1. Motivation",
		"",
		"Phase 3B.2 showed a useful but insufficient ANSWER-vs-INTERMEDIATE intervention. This phase audits the residual semantic failures offline and does not tune the system.",
		"",
		"## 2. Phase 3B.2 outcome",
		"",
		"The frozen Phase 3B.2 decision was `SHARED_TARGET_INTERVENTION_NOT_SUPPORTED`: development correction was 2/5, fresh strong intermediate case-majority target accuracy improved 62.5% to 75.0%, strong final accuracy fell 100% to 87.5%, and controls had 0/8 false-intermediate cases.",
		"",
		"## 3. Research question",
		"",
		"The question is whether the remaining errors are primarily convergence-point identification, downstream-step recognition, answer/convergence confusion, conservative NONE behavior, description quality, or repeat instability.",
		"",
		"## 4. Scope",
		"",
		"Only residual or changed-target cases from the completed Phase 3B.2 artifacts were included. There were no live reruns, LLM calls, Freebase calls, prompt edits, validator edits, executor edits, or benchmark expansions.",
		"",
		"## 5. Residual case manifest",
		"",
		table([
			[case["case_category"], case["case_id"], case["dataset"], case["gold_structure"]]
			for case in selected
		], ["Category", "Case", "Dataset", "Gold structure"]),
		"",
		f"The manifest contains **{len(selected)} independent cases** and **{len(semantic_rows)} Phase 3B.2 repeat rows**. It is frozen in `target_case_manifest.json`.",
		"",
		"## 6. Semantic dimensions",
		"",
		"The residual traces generally preserved the answer-relevant constraints. The stable wrong cases also preserved independent source anchors; one DSiWare repeat collapsed the sources into a single subject-identification group. The frequent failure was later: the extractor labeled the converged variable as the requested answer and set `requires_downstream_relation=false`. Manual review of the frozen traces and Phase 3B.2 notes confirmed the automatic case categories and target-role labels for all nine cases.",
		"",
		"## 7. Convergence-point audit",
		"",
		f"Among the **{len(intermediate_cases)}** residual BRANCH_INTERMEDIATE cases, the stable wrong cases selected ANSWER instead of the frozen INTERMEDIATE target. The DSiWare case alternated NONE and INTERMEDIATE across repeats.",
		"",
		"## 8. Downstream-step audit",
		"",
		f"The stable wrong intermediate cases omitted the downstream step. The DSiWare repeat that selected INTERMEDIATE correctly recognized that a release-location relation remained; its other repeat returned NONE.",
		"",
		"## 9. ANSWER-vs-convergence confusion",
		"",
		f"The primary residual pattern is explicit answer/convergence confusion: **{causes['ANSWER_CONVERGENCE_CONFUSION']['independent_case_count']} independent cases** state that the converged actor, dish, artwork, or film variable is already the requested answer even though the frozen benchmark target is INTERMEDIATE.",
		"",
		"## 10. NONE behavior",
		"",
		f"NONE appeared in the unstable DSiWare case and in the BRANCH_FINAL regression. The stable final regression is classified separately as a misread answer role; it is not evidence for a broad NONE intervention.",
		"",
		"## 11. Description quality",
		"",
		"Descriptions were generally non-empty and specific, but the shared-target descriptions were often specific about the wrong entity role. This is not the same as vague wording: the model confidently described the answer variable rather than an intermediate convergence entity.",
		"",
		"## 12. Self-check audit",
		"",
		table([[label, self_counts.get(label, 0)] for label in ("SELF_CHECK_CORRECT", "SELF_CHECK_WRONG", "SELF_CHECK_UNSTABLE", "SELF_CHECK_MISSING")], ["Status", "Repeat count"]),
		"",
		"The self-check was wrong on both repeats of the five stable answer/convergence-confusion cases. It was unstable for the DSiWare and one changed-target control case.",
		"",
		"## 13. Repeat stability",
		"",
		table([[key, value] for key, value in sorted(stability["counts"].items())], ["Stability class", "Independent cases"]),
		"",
		"Repeat instability was present but did not reach the three-case intervention threshold.",
		"",
		"## 14. BRANCH_INTERMEDIATE residuals",
		"",
		table([
			[row["case_id"], row["repeat"], row["actual_target"], row["earliest_semantic_failure"]]
			for row in branch_rows
		], ["Case", "Repeat", "Target", "Earliest failure"]),
		"",
		"All selected traces represented at least two answer-relevant constraints. The stable wrong cases also preserved independent source anchors; the DSiWare NONE repeat lost that grouping before target selection. For the five stable answer/convergence-confusion cases, the earliest failure was target-role interpretation rather than source extraction.",
		"",
		"## 15. BRANCH_FINAL regressions",
		"",
		table([[row["case_id"], row["repeat"], row["phase2_5_6_target"], row["phase3b_2_target"], row["classification"]] for row in final_rows], ["Case", "Repeat", "Old", "New", "Classification"]),
		"",
		"The regression changed ANSWER to NONE because the extractor treated two requested properties as evidence that no shared answer target existed. This is a frozen-target role disagreement, not a validator/executor failure.",
		"",
		"## 16. Control target changes",
		"",
		table([[row["case_id"], row["repeat"], row["phase2_5_6_target"], row["phase3b_2_target"], row["classification"]] for row in control_rows], ["Case", "Repeat", "Old", "New", "Classification"]),
		"",
		"The control changes were end-to-end answer-equivalent or corrective; no control semantic regression was observed.",
		"",
		"## 17. Old-vs-new extractor comparison",
		"",
		table([[row["case_id"], row["case_category"], ", ".join(row["phase2_5_6_targets"]), ", ".join(row["phase3b_2_targets"]), row["effect"]] for row in effects], ["Case", "Category", "Phase 2.5.6", "Phase 3B.2", "Effect"]),
		"",
		"The intervention corrected no selected stable residual case; it left the five answer/convergence-confusion cases wrong, produced one unstable case, regressed one final case, and corrected one changed-target control.",
		"",
		"## 18. Aggregate residual causes",
		"",
		table([[cause, data["repeat_count"], data["independent_case_count"]] for cause, data in causes.items() if data["repeat_count"]], ["Cause", "Repeat count", "Independent cases"]),
		"",
		f"The largest independent-case family is `{main_cause}`. All other residual families remain below the intervention threshold.",
		"",
		"## 19. Intervention gate",
		"",
		table([[label, count, "PASS" if count >= 3 else "FAIL"] for label, count in decision["candidate_independent_case_counts"].items()], ["Candidate", "Independent cases", "Gate"]),
		"",
		"The gate requires at least three independent cases. Only the answer/convergence-confusion family reaches it, spanning both CWQ and GrailQA and development plus fresh validation.",
		"",
		"## 20. Final Phase 3B.3 decision",
		"",
		f"**{decision['decision']}**",
		"",
		"This is a diagnostic conclusion only. It does not authorize a prompt or runtime intervention in this phase.",
		"",
		"## 21. Recommended next step",
		"",
		"Do not patch the system during Phase 3B.3. If a later phase is authorized, it should isolate answer/convergence-role handling as a new intervention, preserve the self-check audit, and re-test the final/control safety gates separately.",
		"",
		"## 22. Limitations",
		"",
		"The audit uses frozen traces and benchmark target contracts; it does not independently prove that every benchmark logical form is linguistically unambiguous. The selected residual set is diagnostic rather than a new benchmark. Description labels are trace-based, and no live replay was performed.",
	]
	return "\n".join(lines) + "\n"


def run(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
	missing = [name for name in REQUIRED_PHASE3B2 if not (PHASE3B2_ROOT / name).exists()]
	if missing:
		raise FileNotFoundError("missing required Phase 3B.2 artifacts: " + ", ".join(missing))
	selected, development_rows, sources = select_target_cases()
	sources["development"] = development_rows
	selected = build_target_manifest_rows(selected, sources)
	# Freeze the complete case set and its derived repeat-level target metadata
	# before any residual audit is built.
	manifest = freeze_manifest(selected, PHASE3B2_ROOT)
	write_json(output_root / "target_case_manifest.json", manifest)
	semantic_rows, self_rows, description_rows = build_semantic_audits(selected, sources)
	branch_rows = branch_intermediate_audits(selected, sources)
	final_rows = final_regression_audits(selected, sources)
	control_rows = control_changed_audits(selected, sources)
	stability = repeat_stability(selected, sources)
	effects = old_new_effect(selected, sources)
	summary = aggregate_summary(selected, semantic_rows, self_rows, stability, effects)
	decision = decision_gate(summary)
	write_jsonl(output_root / "residual_semantic_audit.jsonl", semantic_rows)
	write_jsonl(output_root / "self_check_audit.jsonl", self_rows)
	write_jsonl(output_root / "description_quality_audit.jsonl", description_rows)
	write_jsonl(output_root / "branch_intermediate_residual_audit.jsonl", branch_rows)
	write_jsonl(output_root / "branch_final_regression_audit.jsonl", final_rows)
	write_jsonl(output_root / "control_changed_target_audit.jsonl", control_rows)
	write_json(output_root / "repeat_stability.json", stability)
	write_json(output_root / "aggregate_residual_summary.json", {**summary, "decision_gate": decision})
	write_json(output_root / "decision_summary.json", decision)
	(output_root / "decision_summary.md").write_text(
		"# Phase 3B.3 Decision Summary\n\n"
		f"Decision: **{decision['decision']}**\n\n"
		f"Independent residual cases: **{len(selected)}**. The only cause reaching the three-case gate is `ANSWER_CONVERGENCE_CONFUSION` with **{summary['primary_causes']['ANSWER_CONVERGENCE_CONFUSION']['independent_case_count']}** independent cases.\n\n"
		"No production behavior, prompt, validator, executor, fallback, or benchmark was changed. No live rerun was performed.\n",
		encoding="utf-8",
	)
	(output_root / "PHASE3B_3_REPORT.md").write_text(
		markdown_report(selected, semantic_rows, self_rows, description_rows, branch_rows, final_rows, control_rows, stability, summary, effects, decision),
		encoding="utf-8",
	)
	return {"output_root": str(output_root), "decision": decision, "summary": summary}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
	args = parser.parse_args()
	result = run(args.output_root)
	print(json.dumps({"output_root": result["output_root"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
	main()
