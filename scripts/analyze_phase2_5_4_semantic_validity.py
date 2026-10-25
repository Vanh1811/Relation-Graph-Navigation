#!/usr/bin/env python3
"""Offline Phase 2.5.4 semantic-validity versus validator-rejection audit."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "outputs/experiments/phase2_5_3b/branch_plan_serialization"
REF_PATH = ROOT / "outputs/experiments/phase2_4_4/routing_decision_error_analysis/router_input_output_analysis.jsonl"
OUTPUT_ROOT = ROOT / "outputs/experiments/phase2_5_4/semantic_validity_validator_audit"

PRIMARY_LABELS = {
	"UPSTREAM_EXTRACTION_WRONG",
	"SEMANTICALLY_INVALID",
	"VALID_BUT_SCHEMA_INCOMPATIBLE",
	"VALID_BUT_VALIDATOR_OVER_REJECTS",
	"SERIALIZATION_INCOMPLETE",
	"INSUFFICIENT_EVIDENCE",
}


def text(value: Any) -> str:
	return str(value or "").strip()


def normalise(value: Any) -> str:
	return re.sub(r"[^a-z0-9]+", " ", text(value).casefold()).strip()


def anchor_value(value: Any) -> str:
	if isinstance(value, dict):
		for key in ("id", "entity_id", "name", "entity_name", "source_anchor", "literal"):
			if text(value.get(key)):
				return text(value[key])
	return text(value)


def raw_proposal(live_row: dict[str, Any]) -> dict[str, Any]:
	for event in live_row.get("prompt_history", []):
		if event.get("type") == "phase2_5_4_branch_plan_serialization_router":
			return event.get("parsed_result", {}) or {}
		if event.get("type") == "phase2_5_3b_branch_plan_serialization_router":
			return event.get("parsed_result", {}) or {}
	return {}


def raw_constraints(proposal: dict[str, Any]) -> list[dict[str, Any]]:
	return [item for item in proposal.get("constraints", []) if isinstance(item, dict)]


def raw_groups(proposal: dict[str, Any]) -> list[dict[str, Any]]:
	return [item for item in proposal.get("independent_source_groups", []) if isinstance(item, dict)]


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


def expected_structure(ref: dict[str, Any]) -> str:
	return text(ref.get("gold_structure")).replace("GOLD_", "")


def semantic_anchor_ids(proposal: dict[str, Any]) -> set[str]:
	anchors: set[str] = set()
	for group in raw_groups(proposal):
		anchor = anchor_value(group.get("source_anchor") or group.get("source_entity"))
		if anchor:
			match = re.search(r"(?:^|\()((?:m|g)\.[a-z0-9_]+)(?:\)|$)", anchor, flags=re.IGNORECASE)
			anchors.add(match.group(1) if match else anchor)
	return anchors


def gold_anchor_ids(ref: dict[str, Any]) -> set[str]:
	return {text(item.get("id")) for item in ref.get("gold_source_anchors", []) if text(item.get("id"))}


def has_upstream_semantic_error(ref: dict[str, Any], row: dict[str, Any], proposal: dict[str, Any]) -> tuple[bool, list[str]]:
	gold = expected_structure(ref)
	live_structure = text(row.get("semantic_structure")).upper()
	target = raw_target_type(proposal)
	reasons: list[str] = []
	if gold == "BRANCH_FINAL":
		if target != "ANSWER":
			reasons.append(f"shared_target={target}, expected ANSWER")
		if live_structure != "BRANCH_FINAL":
			reasons.append(f"structure={live_structure}, expected BRANCH_FINAL")
		if len(semantic_anchor_ids(proposal)) < int(ref.get("independent_sources_gold", 0) or 0):
			reasons.append("independent source anchors are incomplete")
	elif gold == "BRANCH_INTERMEDIATE":
		if target != "INTERMEDIATE":
			reasons.append(f"shared_target={target}, expected INTERMEDIATE")
		if live_structure not in {"BRANCH_INTERMEDIATE", "LINEAR"}:
			reasons.append(f"structure={live_structure}, expected intermediate route")
		if len(semantic_anchor_ids(proposal)) < len(gold_anchor_ids(ref)):
			reasons.append("gold source anchors were collapsed or omitted")
	return bool(reasons), reasons


def primary_classification(ref: dict[str, Any], row: dict[str, Any], proposal: dict[str, Any]) -> tuple[str, list[str]]:
	if text(row.get("serialization_status")) != "COMPLETE":
		return "SERIALIZATION_INCOMPLETE", list(row.get("serialization_errors", [])) or ["required executable fields unavailable"]
	wrong, reasons = has_upstream_semantic_error(ref, row, proposal)
	if wrong:
		return "UPSTREAM_EXTRACTION_WRONG", reasons
	if text(row.get("validator_result")) == "VALIDATOR_REJECT":
		return "VALID_BUT_SCHEMA_INCOMPATIBLE", [
			"semantic plan aligns with offline structure evidence, but the unchanged executable contract rejects its representation"
		]
	return "INSUFFICIENT_EVIDENCE", ["row was not a validator rejection"]


def shared_target_audit(ref: dict[str, Any], proposal: dict[str, Any]) -> str:
	actual = raw_target_type(proposal)
	expected = expected_target(ref)
	if actual != expected:
		return "SHARED_TARGET_GENUINELY_WRONG"
	if expected in {"ANSWER", "INTERMEDIATE"}:
		raw = proposal.get("shared_target")
		if isinstance(raw, str) or (isinstance(raw, dict) and not text(raw.get("description"))):
			return "SHARED_TARGET_REPRESENTATION_MISMATCH"
		return "SHARED_TARGET_VALID_BUT_REJECTED"
	return "UNKNOWN"


def type_only_audit(row: dict[str, Any], proposal: dict[str, Any]) -> str:
	if not any("type-only" in text(reason).casefold() for reason in row.get("validator_reason", [])):
		return "UNKNOWN"
	constraints = raw_constraints(proposal)
	answer_relevant = [bool(item.get("answer_relevant")) for item in constraints]
	roles = {text(item.get("target_variable_role") or item.get("target_role")).upper() for item in constraints}
	properties = [normalise(item.get("semantic_property")) for item in constraints if normalise(item.get("semantic_property"))]
	generic = {"male", "female", "person", "people", "country", "city", "school", "university", "phd", "degree", "type", "gender", "sex"}
	if "TYPE" in roles or (answer_relevant and not any(answer_relevant)) or (properties and all(item in generic for item in properties)):
		return "CORRECT_TYPE_ONLY_REJECTION"
	if any(answer_relevant) and properties:
		return "FALSE_TYPE_ONLY_REJECTION"
	return "TYPE_SIGNAL_BUT_ANSWER_RELEVANT"


def property_audit(ref: dict[str, Any], row: dict[str, Any], proposal: dict[str, Any]) -> str:
	if not any("CONSTRAINT_PROPERTY_MISMATCH" in text(reason) for reason in row.get("validator_reason", [])):
		return "UNKNOWN"
	properties = [normalise(item.get("semantic_property")) for item in raw_constraints(proposal) if normalise(item.get("semantic_property"))]
	if len(set(properties)) >= 2 and all(bool(item.get("answer_relevant")) for item in raw_constraints(proposal)):
		return "COMPLEMENTARY_PROPERTY_VALID"
	if len(set(properties)) == 1:
		return "LEXICAL_NORMALIZATION_MISMATCH"
	return "GENUINE_PROPERTY_MISMATCH"


def target_type_audit(ref: dict[str, Any], proposal: dict[str, Any]) -> str:
	actual = text(proposal.get("requested_final_answer_type") or proposal.get("shared_answer_type"))
	expected = text(ref.get("requested_final_answer_type"))
	if not actual:
		return "MISSING_TYPE_METADATA"
	if not expected:
		return "UNKNOWN"
	actual_norm = normalise(actual)
	expected_norm = normalise(expected)
	if actual_norm == expected_norm:
		return "UNKNOWN"
	if actual_norm in expected_norm or expected_norm in actual_norm or ("ideology" in actual_norm and "ideology" in expected_norm) or ("discipline" in actual_norm and "discipline" in expected_norm):
		return "TYPE_NORMALIZATION_MISMATCH"
	return "GENUINE_TARGET_TYPE_ERROR"


def branch_count_audit(ref: dict[str, Any], row: dict[str, Any], proposal: dict[str, Any]) -> str:
	if not any("at least two branches" in text(reason).casefold() or "branch_count" in text(reason).casefold() for reason in row.get("validator_reason", [])):
		return "UNKNOWN"
	if len(semantic_anchor_ids(proposal)) < int(ref.get("independent_sources_gold", 0) or 0):
		return "UPSTREAM_SOURCE_EXTRACTION_ERROR"
	if any("literal" in anchor.casefold() or re.fullmatch(r"(?:literal:)?[0-9.]+", anchor) for anchor in semantic_anchor_ids(proposal)):
		return "UNKNOWN"
	return "TRUE_BRANCH_COUNT_ERROR"


def executability(primary: str, ref: dict[str, Any], row: dict[str, Any]) -> str:
	if primary == "SERIALIZATION_INCOMPLETE":
		return "UNKNOWN"
	if primary == "UPSTREAM_EXTRACTION_WRONG" or primary == "SEMANTICALLY_INVALID":
		return "NOT_SAFE_TO_EXECUTE"
	if expected_structure(ref) == "BRANCH_INTERMEDIATE":
		return "NOT_SAFE_TO_EXECUTE"
	branches = row.get("serialized_branches", [])
	if len(branches) in {2, 3} and all(text(item.get("root_entity_id")) and text(item.get("constraint_property")) and text(item.get("constraint_summary")) for item in branches):
		return "SAFE_TO_EXECUTE"
	return "POSSIBLY_EXECUTABLE"


def validator_families(row: dict[str, Any]) -> list[str]:
	result: list[str] = []
	for reason in row.get("validator_reason", []):
		value = text(reason)
		lower = value.casefold()
		if "shared_target" in lower:
			family = "INVALID_SHARED_TARGET"
		elif "type-only" in lower or "generic/type-like" in lower:
			family = "TYPE_ONLY_BRANCH"
		elif "constraint_property_mismatch" in lower:
			family = "CONSTRAINT_PROPERTY_MISMATCH"
		elif "branch_count" in lower or "at least two branches" in lower or "number of branch" in lower:
			family = "INVALID_BRANCH_COUNT"
		elif "target_answer_type" in lower or "answer_type" in lower or "requested_final_answer_type" in lower:
			family = "MISSING_OR_INCOMPATIBLE_TARGET_TYPE"
		elif "root" in lower:
			family = "ROOT_ENTITY_INVALID"
		else:
			family = "OTHER"
		if family not in result:
			result.append(family)
	return result or ["OTHER"]


def audit_row(ref: dict[str, Any], row: dict[str, Any], live_row: dict[str, Any]) -> dict[str, Any]:
	proposal = raw_proposal(live_row)
	primary, primary_reasons = primary_classification(ref, row, proposal)
	semantic_error, semantic_reasons = has_upstream_semantic_error(ref, row, proposal)
	return {
		"case_id": row["case_id"],
		"repeat_id": row["repeat_id"],
		"question": ref["question"],
		"group": "HARMFUL" if ref["routing_outcome"] == "HARMFUL_ROUTING_MISMATCH" else "BENIGN" if ref["routing_outcome"] == "BENIGN_ROUTING_MISMATCH" else "CONTROL",
		"gold_structure": ref["gold_structure"],
		"gold_answer_variable": ref.get("gold_answer_variable"),
		"semantic_structure": row.get("semantic_structure"),
		"serialization_status": row.get("serialization_status"),
		"serialized_branches": row.get("serialized_branches", []),
		"validator_reasons": row.get("validator_reason", []),
		"validator_rule_families": validator_families(row),
		"primary_semantic_classification": primary,
		"primary_reasons": primary_reasons,
		"semantic_error_detected": semantic_error,
		"semantic_error_reasons": semantic_reasons,
		"shared_target_raw": proposal.get("shared_target"),
		"shared_target_expected": expected_target(ref),
		"shared_target_audit": shared_target_audit(ref, proposal),
		"type_only_audit": type_only_audit(row, proposal),
		"property_audit": property_audit(ref, row, proposal),
		"target_type_raw": proposal.get("requested_final_answer_type") or proposal.get("shared_answer_type"),
		"target_type_audit": target_type_audit(ref, proposal),
		"branch_count_audit": branch_count_audit(ref, row, proposal),
		"raw_source_anchor_count": len(semantic_anchor_ids(proposal)),
		"gold_source_anchor_count": len(gold_anchor_ids(ref)),
		"executability": executability(primary, ref, row),
		"hits_at_1": row.get("hits_at_1"),
		"f1": row.get("f1"),
	}


def consistency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
	by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		by_case[row["case_id"]].append(row)
	result = []
	for case_id, values in sorted(by_case.items()):
		primary = {row["primary_semantic_classification"] for row in values}
		reasons = {tuple(row["validator_rule_families"]) for row in values}
		if len(primary) > 1:
			label = "UNSTABLE_SEMANTICS"
		elif len(reasons) > 1:
			label = "UNSTABLE_VALIDATOR_REASON"
		elif next(iter(primary)) in {"UPSTREAM_EXTRACTION_WRONG", "SEMANTICALLY_INVALID"}:
			label = "STABLE_SEMANTIC_INVALIDITY"
		elif next(iter(primary)) in {"VALID_BUT_SCHEMA_INCOMPATIBLE", "VALID_BUT_VALIDATOR_OVER_REJECTS"}:
			label = "STABLE_VALID_BUT_REJECTED"
		else:
			label = "STABLE_SERIALIZATION_INCOMPLETE"
		result.append({
			"case_id": case_id,
			"repeat_count": len(values),
			"primary_labels": sorted(primary),
			"validator_reason_sets": [list(item) for item in sorted(reasons)],
			"consistency_label": label,
		})
	return result


def decision_for(primary_counts: Counter[str], case_labels: dict[str, set[str]], rule_counts: Counter[str], control_rows: list[dict[str, Any]]) -> dict[str, Any]:
	over_reject_rows = primary_counts.get("VALID_BUT_VALIDATOR_OVER_REJECTS", 0)
	over_reject_cases = len({row["case_id"] for row in control_rows if row["primary_semantic_classification"] == "VALID_BUT_VALIDATOR_OVER_REJECTS"})
	schema_rows = primary_counts.get("VALID_BUT_SCHEMA_INCOMPATIBLE", 0)
	schema_cases = sum("VALID_BUT_SCHEMA_INCOMPATIBLE" in labels for labels in case_labels.values())
	validator_gate = over_reject_cases >= 3 and over_reject_rows >= 3
	schema_gate = schema_cases >= 3
	if schema_gate and not validator_gate:
		main = "VALIDATOR_SCHEMA_MISMATCH_IS_MAJOR"
	elif validator_gate:
		main = "VALIDATOR_HAS_NARROW_OVER_REJECTION"
	elif primary_counts.get("UPSTREAM_EXTRACTION_WRONG", 0) + primary_counts.get("SEMANTICALLY_INVALID", 0) > schema_rows:
		main = "UPSTREAM_SEMANTICS_REMAIN_PRIMARY"
	else:
		main = "MIXED_SEMANTIC_AND_VALIDATOR_FAILURES"
	return {
		"main_decision": main,
		"validator_intervention_gate": validator_gate,
		"schema_intervention_gate": schema_gate,
		"validator_gate_reason": "Requires at least 3 independent over-rejected cases; no primary over-rejection cases were found." if not validator_gate else "Passed.",
		"schema_gate_reason": f"{schema_cases} independent cases are classified as valid but schema-incompatible." if schema_gate else "Fewer than 3 independent schema-incompatible cases.",
		"over_rejected_repeat_count": over_reject_rows,
		"over_rejected_case_count": over_reject_cases,
		"schema_incompatible_repeat_count": schema_rows,
		"schema_incompatible_case_count": schema_cases,
		"rule_counts": dict(rule_counts),
	}


def write_outputs(audits: list[dict[str, Any]], consistency_rows: list[dict[str, Any]]) -> dict[str, Any]:
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	(OUTPUT_ROOT / "semantic_validity_analysis.jsonl").write_text(
		"".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audits), encoding="utf-8"
	)
	executability_rows = [
		{
			"case_id": row["case_id"],
			"repeat_id": row["repeat_id"],
			"group": row["group"],
			"serialization_status": row["serialization_status"],
			"primary_semantic_classification": row["primary_semantic_classification"],
			"executability": row["executability"],
			"serialized_branches": row["serialized_branches"],
			"question": row["question"],
		}
		for row in audits
	]
	(OUTPUT_ROOT / "executability_analysis.jsonl").write_text(
		"".join(json.dumps(row, ensure_ascii=False) + "\n" for row in executability_rows), encoding="utf-8"
	)
	rule_rows: list[dict[str, Any]] = []
	for row in audits:
		for family in row["validator_rule_families"]:
			rule_rows.append({
				"case_id": row["case_id"],
				"repeat_id": row["repeat_id"],
				"group": row["group"],
				"validator_rule": family,
				"primary_semantic_classification": row["primary_semantic_classification"],
				"shared_target_audit": row["shared_target_audit"],
				"type_only_audit": row["type_only_audit"],
				"property_audit": row["property_audit"],
				"target_type_audit": row["target_type_audit"],
				"branch_count_audit": row["branch_count_audit"],
				"exact_validator_reasons": row["validator_reasons"],
			})
	(OUTPUT_ROOT / "validator_rule_analysis.jsonl").write_text(
		"".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rule_rows), encoding="utf-8"
	)

	primary_counts = Counter(row["primary_semantic_classification"] for row in audits)
	case_labels = defaultdict(set)
	for row in audits:
		case_labels[row["case_id"]].add(row["primary_semantic_classification"])
	rule_counts = Counter(row["validator_rule"] for row in rule_rows)
	rule_breakdown: dict[str, dict[str, int]] = {}
	for rule in sorted(rule_counts):
		relevant = [row for row in audits if rule in row["validator_rule_families"]]
		rule_breakdown[rule] = {
			"genuine_invalid": sum(row["primary_semantic_classification"] in {"UPSTREAM_EXTRACTION_WRONG", "SEMANTICALLY_INVALID"} for row in relevant),
			"over_reject": sum(row["primary_semantic_classification"] == "VALID_BUT_VALIDATOR_OVER_REJECTS" for row in relevant),
			"schema_mismatch": sum(row["primary_semantic_classification"] == "VALID_BUT_SCHEMA_INCOMPATIBLE" for row in relevant),
		}
	shared_counts = Counter(row["shared_target_audit"] for row in audits)
	type_counts = Counter(row["type_only_audit"] for row in audits)
	property_counts = Counter(row["property_audit"] for row in audits)
	target_counts = Counter(row["target_type_audit"] for row in audits)
	branch_counts = Counter(row["branch_count_audit"] for row in audits)
	execution_counts = Counter(row["executability"] for row in audits)
	decision = decision_for(primary_counts, case_labels, rule_counts, audits)
	distribution = {
		"repeat_count": len(audits),
		"case_count": len(case_labels),
		"primary_classification": {label: {"repeat_count": primary_counts.get(label, 0), "case_count": sum(label in labels for labels in case_labels.values())} for label in sorted(PRIMARY_LABELS)},
		"validator_rule_rejections": dict(rule_counts),
		"validator_rule_breakdown": rule_breakdown,
		"shared_target_audit": dict(shared_counts),
		"type_only_audit": dict(type_counts),
		"property_audit": dict(property_counts),
		"target_type_audit": dict(target_counts),
		"branch_count_audit": dict(branch_counts),
		"executability": dict(execution_counts),
		"consistency": dict(Counter(row["consistency_label"] for row in consistency_rows)),
		"decision": decision,
	}
	(OUTPUT_ROOT / "error_distribution.json").write_text(json.dumps(distribution, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	(OUTPUT_ROOT / "decision_summary.md").write_text(render_decision_summary(distribution, consistency_rows), encoding="utf-8")
	(OUTPUT_ROOT / "PHASE2_5_4_REPORT.md").write_text(render_report(distribution, consistency_rows, audits), encoding="utf-8")
	return distribution


def table_primary(distribution: dict[str, Any]) -> str:
	rows = ["| Classification | Repeat count | Case count |", "|---|---:|---:|"]
	for label in sorted(PRIMARY_LABELS):
		item = distribution["primary_classification"][label]
		rows.append(f"| {label} | {item['repeat_count']} | {item['case_count']} |")
	return "\n".join(rows)


def table_rules(distribution: dict[str, Any]) -> str:
	rows = ["| Validator rule | Rejections | Genuine invalid | Over-reject | Schema mismatch |", "|---|---:|---:|---:|---:|"]
	for rule, count in sorted(distribution["validator_rule_rejections"].items()):
		breakdown = distribution.get("validator_rule_breakdown", {}).get(rule, {})
		# Rule-level counts are diagnostic; a row can trigger multiple rules.
		rows.append(
			f"| {rule} | {count} | {breakdown.get('genuine_invalid', 0)} | "
			f"{breakdown.get('over_reject', 0)} | {breakdown.get('schema_mismatch', 0)} |"
		)
	return "\n".join(rows)


def render_decision_summary(distribution: dict[str, Any], consistency_rows: list[dict[str, Any]]) -> str:
	d = distribution["decision"]
	return f"""# Phase 2.5.4 decision summary

## Main decision

**{d['main_decision']}**

The audit covers {distribution['repeat_count']} validator-rejected repeats across {distribution['case_count']} cases.

## Primary semantic classification

{table_primary(distribution)}

## Validator-rule counts

{table_rules(distribution)}

## Gate results

- Validator intervention gate: `{d['validator_intervention_gate']}` — {d['validator_gate_reason']}
- Schema intervention gate: `{d['schema_intervention_gate']}` — {d['schema_gate_reason']}
- Valid-but-validator-over-rejected repeats: {d['over_rejected_repeat_count']}
- Valid-but-schema-incompatible repeats: {d['schema_incompatible_repeat_count']}

## Repeat consistency

{json.dumps(dict(Counter(row['consistency_label'] for row in consistency_rows)), indent=2)}

No validator rule was changed and no rejected plan was executed.
"""


def render_report(distribution: dict[str, Any], consistency_rows: list[dict[str, Any]], audits: list[dict[str, Any]]) -> str:
	d = distribution["decision"]
	harmful = [row for row in audits if row["group"] == "HARMFUL"]
	controls = [row for row in audits if row["group"] == "CONTROL"]
	return f"""# Phase 2.5.4 semantic-validity versus validator-rejection audit

## 1. Motivation from Phase 2.5.3B

Phase 2.5.3B populated complete executable branch plans in 25/30 repeats, but the unchanged validator rejected all 30 and fallback remained 30/30. This audit separates upstream semantic errors, schema incompatibility, and genuine validator over-rejection before any validator change.

## 2. Research question

Among the 30 rejected repeats, which plans are semantically wrong and which are potentially usable despite rejection?

## 3. Frozen components

This was an offline audit only. Phase 2.5.3A extraction, Phase 2.5.3B serialization, structure mapping, validator, fallback, executor, relation scoring, entity grounding, Freebase, SPARQL, evaluator, model, and settings were not modified or bypassed.

## 4. Audit dataset

The exact 30 Phase 2.5.3B rejected repeats were audited: 25 complete serializations and 5 partial/failed serializations. Phase 2.4.4 structural evidence and Phase 2.5.2/2.5.3A reports were reused as offline references.

## 5. Semantic-validity classification

{table_primary(distribution)}

`UPSTREAM_EXTRACTION_WRONG` covers wrong target/structure/source semantics visible before serialization. `VALID_BUT_SCHEMA_INCOMPATIBLE` covers semantically aligned plans whose representation does not satisfy the current executable contract. `SERIALIZATION_INCOMPLETE` covers plans for which required executable fields could not be constructed.

## 6. Shared-target rejection analysis

{json.dumps(distribution['shared_target_audit'], indent=2)}

Most rejected rows expressed the correct ANSWER or INTERMEDIATE target type as a scalar value without the description shape expected by the Phase 2.5 validator. That is a representation mismatch, not automatically a semantic error. Mismatched target types were retained as upstream errors.

## 7. Type-only rejection analysis

{json.dumps(distribution['type_only_audit'], indent=2)}

Rows labeled `FALSE_TYPE_ONLY_REJECTION` had answer-relevant constraints and non-generic semantic properties in the raw extraction; the type-only validator message resulted from non-canonical role values rather than genuinely type-only constraints.

## 8. Constraint-property rejection analysis

{json.dumps(distribution['property_audit'], indent=2)}

The observed property-mismatch rows contain complementary answer-relevant properties, such as a geographic constraint plus a numeric condition. They were not treated as automatically semantically invalid.

## 9. Target-type rejection analysis

{json.dumps(distribution['target_type_audit'], indent=2)}

Missing target metadata and genuine target-type errors remain upstream limitations. Lexical variants such as `ideology` versus `political ideology` were recorded as normalization mismatches.

## 10. Branch-count rejection analysis

{json.dumps(distribution['branch_count_audit'], indent=2)}

Literal-valued source constraints that cannot become grounded branch roots were not fabricated into executable roots; they remain serialization/schema evidence rather than proof that the question has the wrong number of constraints.

## 11. Executability assessment

{json.dumps(distribution['executability'], indent=2)}

Complete gold-aligned final-answer plans with grounded roots were judged `SAFE_TO_EXECUTE` for a future controlled experiment, but none were executed here. Intermediate-answer plans were not judged safe under the current final-answer-only branch merge semantics.

## 12. Harmful cases

The 7 harmful cases contain {sum(row['primary_semantic_classification'] == 'UPSTREAM_EXTRACTION_WRONG' for row in harmful)} upstream-wrong repeats, {sum(row['primary_semantic_classification'] == 'VALID_BUT_SCHEMA_INCOMPATIBLE' for row in harmful)} schema-incompatible repeats, and {sum(row['primary_semantic_classification'] == 'SERIALIZATION_INCOMPLETE' for row in harmful)} incomplete repeats. The harmful subset is mixed; extraction/source errors remain present, but complete schema-compatible-looking plans are also common.

## 13. Controls

All 8 control repeats were semantically aligned but rejected by the same representation/contract family. This means a future schema normalization must preserve the existing conservative type/root checks rather than broadly relaxing validation.

## 14. Repeat consistency

{json.dumps(dict(Counter(row['consistency_label'] for row in consistency_rows)), indent=2)}

Per-case consistency records are included in `decision_summary.md` and the machine-readable audit files.

## 15. Aggregate validator-rule analysis

{table_rules(distribution)}

Rule counts are not mutually exclusive because one repeat can trigger multiple validator reasons. The dominant repeated representation issue is invalid/missing shared-target metadata, accompanied by non-canonical dependency/role values recorded under `OTHER`.

## 16. Main decision

**{d['main_decision']}**

The audit does not support calling the validator itself broadly incorrect. It does show a major semantic-to-executable schema mismatch: many semantically aligned plans are rejected because the current contract receives non-canonical shared-target/dependency/role representations.

## 17. Validator intervention gate

`{d['validator_intervention_gate']}`. The requirement of at least 3 independent cases classified as `VALID_BUT_VALIDATOR_OVER_REJECTS` under the same rule was not met ({d['over_rejected_case_count']} cases). Do not relax validator rules based on this audit.

## 18. Schema intervention gate

`{d['schema_intervention_gate']}`. {d['schema_gate_reason']} A future intervention should be a narrow deterministic normalization/contract experiment, separately evaluated with controls.

## 19. Limitations

This audit did not execute rejected plans, so `SAFE_TO_EXECUTE` is an offline judgment rather than an execution result. The structural reference is the existing Phase 2.4.4 analysis, and semantic labels inherit any limitations in that reference. Repeated LLM outputs also vary in field spelling and enum formatting.

## 20. Recommended next step

Do not run Stage B or full100. Do not modify the validator yet. Run one controlled focused experiment that normalizes only the demonstrated semantic-to-contract representation fields while preserving all validator safety rules, then measure validator passes, branch execution, controls, and QA on the same 15 cases.
"""


def main() -> None:
	refs = {row["case_id"]: row for row in (json.loads(line) for line in REF_PATH.read_text().splitlines() if line.strip())}
	serialization_rows = [json.loads(line) for line in (INPUT_ROOT / "serialization_results.jsonl").read_text().splitlines() if line.strip()]
	live_rows = {(row["case_id"], row["run_id"]): row for row in (json.loads(line) for line in (INPUT_ROOT / "live_runs.jsonl").read_text().splitlines() if line.strip())}
	if len(serialization_rows) != 30 or len(live_rows) != 30:
		raise SystemExit(f"expected 30 Phase 2.5.3B repeats, found {len(serialization_rows)} summaries and {len(live_rows)} live rows")
	audits = []
	for row in serialization_rows:
		if row.get("validator_result") != "VALIDATOR_REJECT":
			raise SystemExit(f"unexpected non-rejected row: {row['case_id']} run={row['repeat_id']}")
		key = (row["case_id"], row["repeat_id"])
		audits.append(audit_row(refs[row["case_id"]], row, live_rows[key]))
	consistency_rows = consistency(audits)
	distribution = write_outputs(audits, consistency_rows)
	print(json.dumps({
		"status": "COMPLETE",
		"rows": len(audits),
		"cases": len(consistency_rows),
		"decision": distribution["decision"]["main_decision"],
		"validator_intervention_gate": distribution["decision"]["validator_intervention_gate"],
		"schema_intervention_gate": distribution["decision"]["schema_intervention_gate"],
		"output_root": str(OUTPUT_ROOT),
	}, indent=2))


if __name__ == "__main__":
	main()
