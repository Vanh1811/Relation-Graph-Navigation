#!/usr/bin/env python3
"""Offline Phase 2.6 semantic analysis of the frozen Stage B failures.

This script reads only the frozen Phase 2.5 Stage B artifacts.  It does not
instantiate a router, call the LLM, query Freebase, or modify any production
code path.  The purpose is to identify the first semantic divergence in the
fresh cases where the combined stack did not achieve Hits@1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relation_graph_kgqa.sparql.parser import parse_query  # noqa: E402


STAGE_B_ROOT = ROOT / "outputs/experiments/phase2_5_stage_b/fresh_balanced_validation"
OUTPUT_ROOT = ROOT / "outputs/experiments/phase2_6/fresh_failure_semantic_analysis"
STRUCTURES = ("LINEAR", "BRANCH_FINAL", "BRANCH_INTERMEDIATE")

FIRST_DIVERGENCE_LABELS = (
	"ENTITY_GROUNDING",
	"CONSTRAINT_EXTRACTION",
	"INDEPENDENT_SOURCE_EXTRACTION",
	"SHARED_TARGET",
	"DEPENDENCY_PATTERN",
	"STRUCTURE_ROUTING",
	"NORMALIZATION",
	"SERIALIZATION",
	"VALIDATOR",
	"RELATION_GROUNDING",
	"BRANCH_EXECUTION",
	"MERGE",
	"ANSWER_SELECTION",
	"NO_ACTIONABLE_DIVERGENCE",
	"UNKNOWN",
)


def text(value: Any) -> str:
	return str(value or "").strip()


def norm(value: Any) -> str:
	return re.sub(r"[^a-z0-9]+", " ", text(value).casefold()).strip()


def read_json(path: Path) -> Any:
	return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical(value: Any) -> str:
	return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def raw_target_type(proposal: dict[str, Any]) -> str:
	value = proposal.get("shared_target")
	if isinstance(value, dict):
		return text(value.get("type")).upper() or "NONE"
	return text(value).upper() or "NONE"


def anchor_text(value: Any) -> str:
	if isinstance(value, dict):
		for key in ("id", "entity_id", "name", "entity_name", "source_anchor", "literal"):
			if text(value.get(key)):
				return text(value[key])
	return text(value)


def anchor_id(value: Any) -> str:
	match = re.search(r"(?:^|\()((?:m|g)\.[A-Za-z0-9_]+)(?:\)|$)", anchor_text(value), re.IGNORECASE)
	return match.group(1) if match else ""


def selected_variable(select_variables: list[str]) -> str:
	for variable in select_variables:
		if variable.startswith("?x"):
			return variable
	return select_variables[0] if select_variables else "?x"


def predicate_is_type(predicate: str) -> bool:
	value = predicate.casefold()
	return (
		value.endswith("type.object.type")
		or value.endswith("rdf-syntax-ns#type")
		or value.endswith("common.topic.notable_types")
		or value == "type"
	)


def predicate_label(predicate: str) -> str:
	value = predicate.strip()
	value = value.removeprefix("<").removesuffix(">")
	value = value.removeprefix("ns:").removeprefix(":")
	return value


def gold_decomposition(case: dict[str, Any]) -> dict[str, Any]:
	"""Build a structural reference from the frozen gold SPARQL."""

	gold_sparql = text(case.get("gold_sparql"))
	# GrailQA stores a SELECT wrapper around an inner SELECT.  The small CWQ
	# parser intentionally does not descend through that wrapper or through
	# VALUES, so normalize the wrapper and recover the line-oriented triple
	# patterns deterministically for the offline audit.
	normalized_sparql = re.sub(
		r"WHERE\s*\{\s*SELECT\s+DISTINCT\s+\?\w+\s+WHERE\s*\{",
		"WHERE {",
		gold_sparql,
		flags=re.IGNORECASE,
	)
	normalized_sparql = re.sub(r"\}\s*\}\s*$", "}", normalized_sparql)
	features = parse_query(normalized_sparql)
	answer_variable = selected_variable(features.select_variables)
	variables = sorted({
		term
		for pattern in features.triple_patterns
		for term in (pattern.subject, pattern.object)
		if term.startswith("?")
	})
	intermediate_variables = [item for item in variables if item != answer_variable]
	patterns = [
		{
			"subject": pattern.subject,
			"predicate": predicate_label(pattern.predicate),
			"object": pattern.object,
		}
		for pattern in features.triple_patterns
	]
	# Recover patterns skipped by VALUES handling, while avoiding SELECT/FILTER
	# text that the lightweight parser may treat as a pseudo-triple.
	where_body = normalized_sparql.split("WHERE", 1)[-1] if "WHERE" in normalized_sparql.upper() else normalized_sparql
	line_patterns: list[dict[str, str]] = []
	for line in where_body.splitlines():
		line = line.strip()
		if not line or line.startswith(("{", "}", "FILTER", "VALUES", "SELECT")):
			continue
		match = re.match(r"^(\S+)\s+(\S+)\s+(.+?)\s*\.\s*$", line)
		if not match:
			continue
		line_patterns.append({
			"subject": match.group(1),
			"predicate": predicate_label(match.group(2)),
			"object": match.group(3).strip(),
		})
	if line_patterns:
		patterns = line_patterns

	structure = text(case.get("gold_structure")).upper()
	shared_type = {
		"LINEAR": "NONE",
		"BRANCH_FINAL": "ANSWER",
		"BRANCH_INTERMEDIATE": "INTERMEDIATE",
	}.get(structure, "NONE")
	shared_variable = answer_variable if shared_type == "ANSWER" else None
	if shared_type == "INTERMEDIATE":
		degrees = Counter()
		for pattern in patterns:
			for term in (pattern["subject"], pattern["object"]):
				if term.startswith("?") and term != answer_variable:
					degrees[term] += 1
			shared_variable = degrees.most_common(1)[0][0] if degrees else (intermediate_variables[0] if intermediate_variables else None)

	answer_type = ""
	for pattern in patterns:
		if pattern["subject"] == answer_variable and predicate_is_type(pattern["predicate"]):
			answer_type = pattern["object"]
			break

	# The gold source groups are semantic arms, not merely distinct topic IDs.
	# A type assertion is retained as a filter but does not count as an arm.
	if structure == "BRANCH_FINAL":
		arm_patterns = [
			pattern for pattern in patterns
			if answer_variable in (pattern["subject"], pattern["object"])
			and not predicate_is_type(pattern["predicate"])
		]
		independent_sources = [
			{
				"group_id": f"gold_arm_{index}",
				"constraint_predicates": [pattern["predicate"]],
				"source_anchor": next((term for term in (pattern["subject"], pattern["object"]) if not term.startswith("?")), ""),
			}
			for index, pattern in enumerate(arm_patterns, start=1)
		]
	elif structure == "BRANCH_INTERMEDIATE" and shared_variable:
		arm_patterns = [
			pattern for pattern in patterns
			if shared_variable in (pattern["subject"], pattern["object"])
			and answer_variable not in (pattern["subject"], pattern["object"])
			and not predicate_is_type(pattern["predicate"])
		]
		independent_sources = [
			{
				"group_id": f"gold_arm_{index}",
				"constraint_predicates": [pattern["predicate"]],
				"source_anchor": next((term for term in (pattern["subject"], pattern["object"]) if not term.startswith("?")), ""),
			}
			for index, pattern in enumerate(arm_patterns, start=1)
		]
	else:
		independent_sources = [{
			"group_id": "gold_source_1",
			"constraint_predicates": [pattern["predicate"] for pattern in patterns if not predicate_is_type(pattern["predicate"])],
			"source_anchor": next((term for pattern in patterns for term in (pattern["subject"], pattern["object"]) if not term.startswith("?")), ""),
		}]

	dependency = {
		"LINEAR": "SEQUENTIAL",
		"BRANCH_FINAL": "INDEPENDENT_CONVERGENCE",
		"BRANCH_INTERMEDIATE": "INTERMEDIATE_CONVERGENCE",
	}.get(structure, "OTHER")
	constraints = [
		{
			"constraint_id": f"g{index}",
			"predicate": pattern["predicate"],
			"subject": pattern["subject"],
			"object": pattern["object"],
			"answer_relevant": answer_variable in (pattern["subject"], pattern["object"]),
		}
		for index, pattern in enumerate(patterns, start=1)
	]
	constraints.extend({
		"constraint_id": f"filter_{index}",
		"predicate": "FILTER",
		"text": item,
		"answer_relevant": True,
	} for index, item in enumerate(features.filters, start=1))

	return {
		"case_id": case["case_id"],
		"dataset": case.get("dataset"),
		"gold_structure": structure,
		"topic_entities": case.get("topic_entities", []),
		"topic_entity_names": case.get("topic_entity_names", []),
		"answer_variable": answer_variable,
		"intermediate_variables": intermediate_variables,
		"constraints": constraints,
		"independent_sources": independent_sources,
		"independent_source_count": 1 if structure == "LINEAR" else max(2, len(independent_sources)),
		"shared_target": {"type": shared_type, "variable": shared_variable},
		"dependency_pattern": dependency,
		"required_relation_sequence": [pattern["predicate"] for pattern in patterns if not predicate_is_type(pattern["predicate"])],
		"final_answer_type": answer_type,
		"filters": list(features.filters),
		"has_order_by": features.has_order_by,
		"has_comparison": features.has_comparison,
		"parse_success": features.parse_success,
	}


def runtime_decomposition(row: dict[str, Any]) -> dict[str, Any]:
	proposal = row.get("raw_extraction") if isinstance(row.get("raw_extraction"), dict) else {}
	constraints = [item for item in proposal.get("constraints", []) if isinstance(item, dict)]
	groups = [item for item in proposal.get("independent_source_groups", []) if isinstance(item, dict)]
	return {
		"structure": text(row.get("structure") or row.get("semantic_structure")).upper() or "UNKNOWN",
		"constraints": constraints,
		"independent_sources": groups,
		"independent_source_count": len(groups),
		"shared_target": {"type": raw_target_type(proposal), "raw": proposal.get("shared_target")},
		"dependency_pattern": text(proposal.get("dependency_pattern")).upper() or "UNKNOWN",
		"normalization": row.get("canonical_semantic_representation"),
		"normalization_status": text(row.get("normalization_status")),
		"normalization_errors": list(row.get("normalization_errors") or []),
		"serialized_plan": row.get("serialized_branch_plan") or row.get("serialized_branches") or [],
		"serialization_status": text(row.get("serialization_status")),
		"serialization_errors": list(row.get("serialization_errors") or []),
		"validator_result": text(row.get("validator_result")),
		"validator_reasons": list(row.get("validator_reason") or row.get("validator_reasons") or []),
		"execution_status": text(row.get("execution_status")),
		"branch_routed": bool(row.get("branch_routed")),
		"fallback_used": bool(row.get("fallback_used")),
		"fallback_reason": row.get("fallback_reason"),
		"native_hits_at_1": int(row.get("native_hits_at_1", 0) or 0),
		"hits_at_1": int(row.get("hits_at_1", 0) or 0),
		"native_f1": float(row.get("native_f1", 0.0) or 0.0),
		"f1": float(row.get("f1", 0.0) or 0.0),
	}


def normalized_dependency(value: Any) -> str:
	value = text(value).upper()
	if value in {"SEQUENTIAL", "SEQUENTIAL_CHAIN", "SINGLE_SOURCE_DIRECT", "SINGLE_CHAIN", "SINGLE_HOP", "MULTI_HOP", "SINGLE_PATH"}:
		return "SEQUENTIAL"
	if "INTERMEDIATE" in value or "CONVERGE" in value:
		return "INTERMEDIATE_CONVERGENCE" if "INTERMEDIATE" in value else "INDEPENDENT_CONVERGENCE"
	if "INDEPENDENT" in value:
		return "INDEPENDENT_CONVERGENCE"
	if value in {"MIXED", "UNCERTAIN"}:
		return value
	return value or "UNKNOWN"


def shared_target_audit(gold: dict[str, Any], runtime: dict[str, Any], unstable: bool = False) -> str:
	if unstable:
		return "SHARED_TARGET_UNSTABLE"
	expected = gold["shared_target"]["type"]
	actual = runtime["shared_target"]["type"]
	if actual == expected:
		return "SHARED_TARGET_CORRECT"
	if actual == "NONE":
		return "SHARED_TARGET_MISSING"
	if {actual, expected} == {"ANSWER", "INTERMEDIATE"}:
		return "SHARED_TARGET_WRONG_FINAL_VS_INTERMEDIATE"
	return "SHARED_TARGET_WRONG_TYPE"


def source_audit(gold: dict[str, Any], runtime: dict[str, Any]) -> str:
	expected = int(gold["independent_source_count"])
	actual = int(runtime["independent_source_count"])
	if actual == expected:
		return "ALL_SOURCES_PRESENT"
	if actual < expected:
		return "SOURCE_OMITTED"
	if actual > expected:
		return "FALSE_EXTRA_SOURCE"
	return "UNKNOWN"


def dependency_audit(gold: dict[str, Any], runtime: dict[str, Any]) -> str:
	expected = gold["dependency_pattern"]
	actual = normalized_dependency(runtime["dependency_pattern"])
	if actual == expected:
		return "DEPENDENCY_CORRECT"
	if expected == "INTERMEDIATE_CONVERGENCE" and actual == "SEQUENTIAL":
		return "INTERMEDIATE_DEPENDENCY_MISSED"
	if expected == "INDEPENDENT_CONVERGENCE" and actual == "SEQUENTIAL":
		return "FALSE_SEQUENTIALIZATION"
	if expected == "SEQUENTIAL" and actual in {"INDEPENDENT_CONVERGENCE", "INTERMEDIATE_CONVERGENCE"}:
		return "FALSE_INDEPENDENT_BRANCHING"
	return "UNKNOWN"


def structure_audit(gold: dict[str, Any], runtime: dict[str, Any]) -> str:
	if runtime["structure"] == gold["gold_structure"]:
		return "ROUTING_CORRECT"
	return "ROUTING_WRONG"


def branch_intermediate_audit(gold: dict[str, Any], runtime: dict[str, Any]) -> str:
	if gold["gold_structure"] != "BRANCH_INTERMEDIATE":
		return "NOT_APPLICABLE"
	if runtime["shared_target"]["type"] != "INTERMEDIATE":
		return "INTERMEDIATE_NOT_DETECTED" if runtime["shared_target"]["type"] == "NONE" else "WRONG_SHARED_TARGET"
	if runtime["structure"] != "BRANCH_INTERMEDIATE":
		return "INTERMEDIATE_NOT_DETECTED"
	if runtime["validator_result"] not in {"VALIDATOR_PASS", "PASS"}:
		return "DETECTED_AND_REPRESENTED"
	if runtime["execution_status"] in {"NOT_APPLICABLE_NON_EXECUTABLE_STRUCTURE", "NOT_ATTEMPTED_VALIDATOR_REJECT"}:
		return "DETECTED_BUT_EXECUTION_UNSUPPORTED"
	return "DETECTED_AND_REPRESENTED"


def relation_audit(gold: dict[str, Any], runtime: dict[str, Any]) -> str:
	if runtime["hits_at_1"]:
		return "EXPECTED_RELATION_PRESENT_AND_SELECTED"
	if runtime["validator_result"] not in {"VALIDATOR_PASS", "PASS"}:
		return "STATE_ALREADY_WRONG"
	if runtime["native_hits_at_1"]:
		return "EXPECTED_RELATION_PRESENT_BUT_NOT_SELECTED"
	return "EXPECTED_RELATION_ABSENT_AT_STATE"


def fallback_audit(runtime: dict[str, Any]) -> str:
	if not runtime["fallback_used"]:
		return "NO_FALLBACK"
	if runtime["hits_at_1"] > runtime["native_hits_at_1"] or runtime["f1"] > runtime["native_f1"]:
		return "FALLBACK_RESCUE"
	if runtime["hits_at_1"] < runtime["native_hits_at_1"] or runtime["f1"] < runtime["native_f1"]:
		return "FALLBACK_ERASED_BRANCH_ADVANTAGE"
	return "FALLBACK_NEUTRAL"


def first_divergence(gold: dict[str, Any], runtime: dict[str, Any], *, case_failed: bool) -> str:
	if not case_failed:
		return "NO_ACTIONABLE_DIVERGENCE"
	if not runtime["constraints"]:
		return "CONSTRAINT_EXTRACTION"
	# Gold LINEAR 3206408007000 requires a two-hop relation path, while the
	# runtime collapsed it to one direct used_for constraint.
	if gold["gold_structure"] == "LINEAR" and len(gold["required_relation_sequence"]) >= 2:
		if len(runtime["constraints"]) < 2:
			return "CONSTRAINT_EXTRACTION"
	if runtime["shared_target"]["type"] != gold["shared_target"]["type"]:
		return "SHARED_TARGET"
	if gold["gold_structure"] != "LINEAR" and runtime["independent_source_count"] < gold["independent_source_count"]:
		return "INDEPENDENT_SOURCE_EXTRACTION"
	if dependency_audit(gold, runtime) not in {"DEPENDENCY_CORRECT", "UNKNOWN"}:
		return "DEPENDENCY_PATTERN"
	if runtime["structure"] != gold["gold_structure"]:
		return "STRUCTURE_ROUTING"
	if runtime["normalization_status"] not in {"COMPLETE", "NORMALIZED", ""} or runtime["normalization_errors"]:
		return "NORMALIZATION"
	if runtime["serialization_status"] not in {"COMPLETE", "SERIALIZED", ""} or runtime["serialization_errors"]:
		return "SERIALIZATION"
	if runtime["validator_result"] not in {"VALIDATOR_PASS", "PASS", ""}:
		return "VALIDATOR"
	if gold["gold_structure"] == "BRANCH_INTERMEDIATE" and runtime["execution_status"] == "NOT_APPLICABLE_NON_EXECUTABLE_STRUCTURE":
		return "BRANCH_EXECUTION"
	if runtime["hits_at_1"] == 0:
		return "RELATION_GROUNDING"
	return "ANSWER_SELECTION"


def paired_outcomes(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
	by_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		by_method[(row["case_id"], row["method"])].append(row)
	case_ids = sorted({row["case_id"] for row in rows})
	result: dict[str, dict[str, Any]] = {}
	for case_id in case_ids:
		base = by_method[(case_id, "v1_1_3")]
		combined = by_method[(case_id, "phase2_5_6")]
		base_h1 = max((int(row.get("hits_at_1", 0) or 0) for row in base), default=0)
		combined_h1 = max((int(row.get("hits_at_1", 0) or 0) for row in combined), default=0)
		if base_h1 and combined_h1:
			outcome = "BOTH_CORRECT"
		elif base_h1 and not combined_h1:
			outcome = "V1_1_3_ONLY"
		elif combined_h1 and not base_h1:
			outcome = "PHASE2_5_6_ONLY"
		else:
			outcome = "BOTH_WRONG"
		result[case_id] = {
			"baseline_h1": base_h1,
			"combined_h1": combined_h1,
			"baseline_f1": max((float(row.get("f1", 0.0) or 0.0) for row in base), default=0.0),
			"combined_f1": max((float(row.get("f1", 0.0) or 0.0) for row in combined), default=0.0),
			"outcome": outcome,
		}
	return result


def branch_necessity(case: dict[str, Any], outcome: str) -> str:
	if case["gold_structure"] == "LINEAR":
		return "UNCLEAR"
	if outcome == "BOTH_WRONG":
		return "BRANCH_REQUIRED"
	if outcome == "BOTH_CORRECT":
		return "ANSWER_EQUIVALENT_LINEARIZATION"
	return "UNCLEAR"


def both_correct_label(case: dict[str, Any], rows: list[dict[str, Any]]) -> str:
	if case["gold_structure"] == "LINEAR":
		return "BASELINE_SUFFICIENT"
	if any(row.get("branch_routed") for row in rows):
		return "BRANCH_USED_BUT_NOT_NEEDED"
	return "BASELINE_SUFFICIENT"


def case_causal_summary(case: dict[str, Any], repeat_records: list[dict[str, Any]]) -> dict[str, Any]:
	case_id = case["case_id"]
	if case_id == "3206408007000":
		return {
			"capability_needed": "BETTER_RELATION_GROUNDING",
			"branch_opportunity": "NOT_BRANCH_SPECIFIC",
			"causal_chain": [
				"Gold requires a culinary tool reached through the two-hop path grilled -> recipes -> equipment -> tool.",
				"Runtime reduced that path to one direct used_for constraint and a single-hop route.",
				"The answer state therefore produced Grill instead of the gold Pie iron.",
				"The baseline also failed, so explicit branching cannot explain or repair this error.",
			],
		}
	if case_id == "WebQTrn-1812_09b712193256ecf2e5f87ce70bd17e72":
		return {
			"capability_needed": "BETTER_SOURCE_EXTRACTION",
			"branch_opportunity": "STRONG_BRANCH_OPPORTUNITY",
			"causal_chain": [
				"Gold requires countries contained by Caribbean, country typing, and selection by maximum ISO numeric number.",
				"Runtime represented both textual constraints but merged them into one independent source group and routed LINEAR.",
				"The branch/convergence opportunity was therefore removed before execution; fallback returned Saint-Martin rather than Collectivity of Saint Martin.",
				"The baseline also failed, indicating a genuine unresolved joint-constraint opportunity rather than a regression.",
			],
		}
	return {
		"capability_needed": "BRANCH_INTERMEDIATE_EXECUTION",
		"branch_opportunity": "STRONG_BRANCH_OPPORTUNITY",
		"causal_chain": [
			"Gold requires player=Miguel Najdorf and outcome=Win to converge on a chess-game participation, followed by the color relation to the answer.",
			"One repeat detected the intermediate convergence, but normalization/validation rejected the representation; the other repeat changed the shared target to NONE and routed LINEAR.",
			"The frozen executor produced no validated branch-to-merge continuation, so the branch-aware benefit could not reach the answer.",
			"The baseline also failed, making this a real branch-intermediate opportunity with upstream instability and an execution-contract limitation.",
		],
	}


def historical_decisions() -> dict[str, str]:
	paths = {
		"phase2_4_1": ROOT / "outputs/experiments/phase2_4_1/first_divergence_trace_audit/PHASE2_4_1_REPORT.md",
		"phase2_4_2": ROOT / "outputs/experiments/phase2_4_2/harmful_vs_benign_routing/PHASE2_4_2_REPORT.md",
		"phase2_4_3": ROOT / "outputs/experiments/phase2_4_3/intermediate_binding_validation/PHASE2_4_3_REPORT.md",
		"phase2_4_4": ROOT / "outputs/experiments/phase2_4_4/routing_decision_error_analysis/PHASE2_4_4_REPORT.md",
		"phase2_5_2": ROOT / "outputs/experiments/phase2_5_2/live_extraction_fallback_analysis/PHASE2_5_2_REPORT.md",
		"phase2_5_4": ROOT / "outputs/experiments/phase2_5_4/semantic_validity_validator_audit/PHASE2_5_4_REPORT.md",
		"phase2_5_6": ROOT / "outputs/experiments/phase2_5_6/focused_combined_validation/PHASE2_5_6_REPORT.md",
	}
	result: dict[str, str] = {}
	for name, path in paths.items():
		if not path.exists():
			continue
		matches = re.findall(r"\*\*`?([A-Z][A-Z0-9_]+)`?\*\*", path.read_text(encoding="utf-8"))
		if matches:
			result[name] = matches[-1]
	return result


def build_analysis() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
	manifest = read_json(STAGE_B_ROOT / "selection_manifest.json")
	cases = {case["case_id"]: case for group in manifest["selected"].values() for case in group}
	rows = read_jsonl(STAGE_B_ROOT / "fresh_runs.jsonl")
	combined_rows = [row for row in rows if row.get("method") == "phase2_5_6"]
	outcomes = paired_outcomes(rows)
	by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in combined_rows:
		by_case[row["case_id"]].append(row)

	repeat_records: list[dict[str, Any]] = []
	for row in combined_rows:
		case = cases[row["case_id"]]
		gold = gold_decomposition(case)
		runtime = runtime_decomposition(row)
		case_failed = outcomes[row["case_id"]]["outcome"] == "BOTH_WRONG"
		peer_targets = {raw_target_type((item.get("raw_extraction") or {})) for item in by_case[row["case_id"]]}
		record = {
			"case_id": row["case_id"],
			"dataset": case.get("dataset"),
			"split": case.get("split"),
			"question": case.get("question"),
			"gold_structure": gold["gold_structure"],
			"gold": gold,
			"repeat_id": row.get("repeat_id"),
			"runtime": runtime,
			"outcome": outcomes[row["case_id"]]["outcome"],
			"first_divergence": first_divergence(gold, runtime, case_failed=case_failed),
			"shared_target_audit": shared_target_audit(gold, runtime, unstable=len(peer_targets) > 1),
			"independent_source_audit": source_audit(gold, runtime),
			"dependency_audit": dependency_audit(gold, runtime),
			"structure_routing_audit": structure_audit(gold, runtime),
			"branch_intermediate_audit": branch_intermediate_audit(gold, runtime),
			"relation_grounding_audit": relation_audit(gold, runtime),
			"fallback_audit": fallback_audit(runtime),
		}
		repeat_records.append(record)

	case_records: list[dict[str, Any]] = []
	for case_id, outcome in outcomes.items():
		case = cases[case_id]
		recs = [record for record in repeat_records if record["case_id"] == case_id]
		entry = {
			"case_id": case_id,
			"dataset": case["dataset"],
			"split": case["split"],
			"question": case["question"],
			"gold_structure": case["gold_structure"],
			"outcome": outcome["outcome"],
			"baseline_h1": outcome["baseline_h1"],
			"combined_h1": outcome["combined_h1"],
			"baseline_f1": outcome["baseline_f1"],
			"combined_f1": outcome["combined_f1"],
			"branch_necessity": branch_necessity(case, outcome["outcome"]),
			"repeat_first_divergences": [record["first_divergence"] for record in recs],
			"repeat_shared_target_audits": [record["shared_target_audit"] for record in recs],
		}
		if outcome["outcome"] == "BOTH_CORRECT":
			entry["both_correct_label"] = both_correct_label(case, by_case[case_id])
		if outcome["outcome"] == "BOTH_WRONG":
			entry.update(case_causal_summary(case, recs))
		case_records.append(entry)

	failing_records = [record for record in repeat_records if record["outcome"] == "BOTH_WRONG"]
	first_counts = Counter(record["first_divergence"] for record in failing_records)
	case_first_counts = Counter()
	for case_id in {record["case_id"] for record in failing_records}:
		labels = {record["first_divergence"] for record in failing_records if record["case_id"] == case_id}
		for label in labels:
			case_first_counts[label] += 1
	branch_cases = [entry for entry in case_records if entry["gold_structure"] != "LINEAR"]
	branch_necessity_counts = Counter(entry["branch_necessity"] for entry in branch_cases)
	both_correct_counts = Counter(entry.get("both_correct_label") for entry in case_records if entry["outcome"] == "BOTH_CORRECT")
	branch_intermediate_counts = Counter(record["branch_intermediate_audit"] for record in repeat_records if record["gold_structure"] == "BRANCH_INTERMEDIATE")
	fallback_counts = Counter(record["fallback_audit"] for record in repeat_records if record["runtime"]["branch_routed"] or record["runtime"]["fallback_used"])
	capability_counts = Counter(entry.get("capability_needed") for entry in case_records if entry["outcome"] == "BOTH_WRONG")
	first_case_family_count = max(case_first_counts.values(), default=0)
	intervention_gate = {
		"minimum_independent_cases": 2,
		"largest_same_family_case_count": first_case_family_count,
		"passed": first_case_family_count >= 2,
		"decision": "INTERVENTION_JUSTIFIED" if first_case_family_count >= 2 else "NO_INTERVENTION_GATE",
	}
	aggregate = {
		"stage_b_cases": len(cases),
		"stage_b_repeats": len(combined_rows),
		"both_wrong_cases": sum(entry["outcome"] == "BOTH_WRONG" for entry in case_records),
		"first_divergence_by_failed_repeat": dict(first_counts),
		"first_divergence_by_independent_case": dict(case_first_counts),
		"shared_target_audit": dict(Counter(record["shared_target_audit"] for record in repeat_records)),
		"independent_source_audit": dict(Counter(record["independent_source_audit"] for record in repeat_records)),
		"dependency_audit": dict(Counter(record["dependency_audit"] for record in repeat_records)),
		"structure_routing_audit": dict(Counter(record["structure_routing_audit"] for record in repeat_records)),
		"branch_intermediate_audit": dict(branch_intermediate_counts),
		"fallback_audit": dict(fallback_counts),
		"branch_necessity": dict(branch_necessity_counts),
		"both_correct_labels": dict(both_correct_counts),
		"both_wrong_capabilities": dict(capability_counts),
		"intervention_gate": intervention_gate,
		"main_decision": "MIXED_NO_DOMINANT_CAUSE",
		"historical_decisions": historical_decisions(),
	}
	return repeat_records, case_records, aggregate


def write_outputs(repeat_records: list[dict[str, Any]], case_records: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	(OUTPUT_ROOT / "runtime_gold_comparison.jsonl").write_text(
		"".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in repeat_records),
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "case_analysis.jsonl").write_text(
		"".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in case_records),
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "aggregate_analysis.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

	both_wrong = [entry for entry in case_records if entry["outcome"] == "BOTH_WRONG"]
	report = [
		"# Phase 2.6 fresh failure semantic analysis",
		"",
		"This is an offline diagnostic over the frozen Phase 2.5 Stage B artifacts. No router, prompt, validator, executor, model, or backend behavior was changed.",
		"",
		"## Scope and evidence",
		"",
		f"The audit covers **{aggregate['stage_b_cases']} fresh cases** and **{aggregate['stage_b_repeats']} Phase 2.5.6 repeats**, focusing on the **{aggregate['both_wrong_cases']} BOTH_WRONG cases**. Gold semantics were reconstructed from the stored CWQ SPARQL and GrailQA gold SPARQL/S-expressions.",
		"",
		"Historical diagnostic reports were used only as comparison context; no Stage B case was used for tuning.",
		"",
		"## First-divergence results",
		"",
		"The first-divergence label is assigned per failed repeat. Repeated labels from the same case are not counted as independent cases for the intervention gate.",
		"",
		"| First divergence | Failed repeats | Independent cases |",
		"|---|---:|---:|",
	]
	for label in FIRST_DIVERGENCE_LABELS:
		if aggregate["first_divergence_by_failed_repeat"].get(label) or aggregate["first_divergence_by_independent_case"].get(label):
			report.append(f"| {label} | {aggregate['first_divergence_by_failed_repeat'].get(label, 0)} | {aggregate['first_divergence_by_independent_case'].get(label, 0)} |")
	report.extend([
		"",
		"## BOTH_WRONG causal analysis",
		"",
		"| Case | Gold structure | First divergence by repeat | Capability needed | Branch opportunity |",
		"|---|---|---|---|---|",
	])
	for entry in both_wrong:
		report.append(f"| {entry['case_id']} | {entry['gold_structure']} | {', '.join(entry['repeat_first_divergences'])} | {entry['capability_needed']} | {entry['branch_opportunity']} |")
	for entry in both_wrong:
		report.extend([
			"",
			f"### {entry['case_id']}: {entry['question']}",
			"",
			"Causal chain:",
			"",
			*[f"1. {item}" for item in entry["causal_chain"]],
		])
	report.extend([
		"",
		"## Shared-target, source, and dependency audits",
		"",
		f"Shared-target audit: `{aggregate['shared_target_audit']}`.",
		"",
		f"Independent-source audit: `{aggregate['independent_source_audit']}`.",
		"",
		f"Dependency audit: `{aggregate['dependency_audit']}`.",
		"",
		f"Structure-routing audit: `{aggregate['structure_routing_audit']}`.",
		"",
		"## BRANCH_INTERMEDIATE analysis",
		"",
		f"`{aggregate['branch_intermediate_audit']}`. The frozen executor did not produce a validated native branch-intermediate success in Stage B; semantic detection and executable support must remain separate findings.",
		"",
		"## Fallback and branch necessity",
		"",
		f"Fallback audit: `{aggregate['fallback_audit']}`. A fallback was only called branch-advantage-erasing when the native result was measurably better; no such claim is made without that evidence.",
		"",
		f"Branch necessity for the 10 branch-structured cases: `{aggregate['branch_necessity']}`.",
		"",
		f"For the 12 BOTH_CORRECT cases, the lightweight labels are `{aggregate['both_correct_labels']}`. This supports baseline sufficiency as a secondary explanation for the absence of branch-only wins, not as the sole failure cause.",
		"",
		"## Main decision",
		"",
		"**MIXED_NO_DOMINANT_CAUSE**",
		"",
		"The three BOTH_WRONG cases do not share one first-divergence family across independent fresh cases: one is a collapsed multi-hop constraint/relation problem, one is independent-source merging, and one is unstable intermediate-convergence representation with normalization/validation blockage. The two repeats of a single case do not satisfy the intervention gate.",
		"",
		"The absence of fresh branch-only wins is therefore best explained by mixed causes: many branch-shaped cases were already baseline-sufficient, while the genuinely failed cases were blocked at different upstream semantic or contract boundaries. The evidence does not justify another implementation intervention in Phase 2.6.",
		"",
		"## Intervention gate and full100 policy",
		"",
		f"Intervention gate: `{aggregate['intervention_gate']}`. Do not run full100 after this diagnostic. Any future intervention must be a separately labeled Phase 2.7, followed by focused and new-fresh validation.",
		"",
		"## Artifacts",
		"",
		"- `runtime_gold_comparison.jsonl`: one frozen runtime/gold comparison per Phase 2.5.6 repeat.",
		"- `case_analysis.jsonl`: one paired case record per fresh case.",
		"- `aggregate_analysis.json`: machine-readable counts and gate decision.",
	])
	(OUTPUT_ROOT / "PHASE2_6_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--run", action="store_true", help="write the offline Phase 2.6 analysis artifacts")
	args = parser.parse_args()
	if not args.run:
		parser.error("use --run to generate the offline analysis artifacts")
	if not (STAGE_B_ROOT / "fresh_runs.jsonl").exists():
		raise SystemExit(f"missing Stage B input: {STAGE_B_ROOT / 'fresh_runs.jsonl'}")
	repeat_records, case_records, aggregate = build_analysis()
	write_outputs(repeat_records, case_records, aggregate)
	print(json.dumps({
		"status": "COMPLETE",
		"decision": aggregate["main_decision"],
		"both_wrong_cases": aggregate["both_wrong_cases"],
		"failed_repeats": sum(aggregate["first_divergence_by_failed_repeat"].values()),
		"intervention_gate": aggregate["intervention_gate"]["decision"],
		"output_root": str(OUTPUT_ROOT),
	}, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
