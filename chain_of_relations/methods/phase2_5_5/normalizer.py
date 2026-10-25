"""Pure deterministic normalization of Phase 2.5 semantic representations."""

from __future__ import annotations

import copy
import re
from typing import Any


ACCEPTED_TARGET_TYPES = {"ANSWER", "INTERMEDIATE", "NONE"}
ACCEPTED_DEPENDENCIES = {"SEQUENTIAL", "INDEPENDENT_CONVERGENCE", "MIXED", "UNCERTAIN"}
ACCEPTED_ROLES = {"ANSWER", "INTERMEDIATE", "FILTER", "TYPE"}


def _text(value: Any) -> str:
	return str(value or "").strip()


def _clean(value: Any) -> str:
	return re.sub(r"\s+", " ", _text(value).replace("_", " ").replace("-", " ")).strip().casefold()


def _canonical_property(value: Any) -> tuple[str, str]:
	raw = _text(value)
	if not raw:
		return "", "NORMALIZATION_MISSING_VALUE"
	canonical = re.sub(r"\s+", " ", raw.replace("_", " ").replace("-", " ")).strip().casefold()
	if canonical == raw:
		return raw, "NO_CHANGE"
	return canonical, "FORMAT_CASE_SEPARATOR_NORMALIZATION"


def _canonical_target_type(value: Any, question: str, semantic_property: Any = "") -> tuple[str, str]:
	raw = _text(value)
	if not raw or _clean(raw) in {"unknown", "none", "null"}:
		return ("UNKNOWN" if not raw or _clean(raw) == "unknown" else raw.upper()), "NORMALIZATION_MISSING_VALUE"
	clean = _clean(raw)
	known = {
		"sex": "gender",
		"parent sex": "gender",
		"parent gender": "gender",
		"profession": "occupation",
		"shared occupation": "occupation",
		"higher classification": "genus",
	}
	if clean in known:
		return known[clean], "KNOWN_PROJECT_TYPE_ALIAS"
	question_clean = _clean(question)
	property_clean = _clean(semantic_property)
	if "political ideology" in question_clean and clean == "ideology":
		return "political ideology", "EXPLICIT_QUESTION_TYPE_ALIAS"
	if clean == "political ideology" or clean == "ideology":
		return clean, "NO_CHANGE"
	canonical = clean
	if canonical == raw:
		return canonical, "NO_CHANGE"
	return canonical, "FORMAT_CASE_SEPARATOR_NORMALIZATION"


def _canonical_target(value: Any, proposal: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
	raw = value
	if isinstance(raw, dict):
		raw_type = raw.get("type") if raw.get("type") is not None else raw.get("target")
		raw_description = _text(raw.get("description"))
	elif isinstance(raw, str):
		raw_type = raw
		raw_description = ""
	else:
		raw_type = "UNKNOWN"
		raw_description = ""
	clean_type = _clean(raw_type)
	type_aliases = {
		"answer": "ANSWER",
		"final answer": "ANSWER",
		"final answer target": "ANSWER",
		"final answer entity": "ANSWER",
		"intermediate": "INTERMEDIATE",
		"intermediate target": "INTERMEDIATE",
		"intermediate entity": "INTERMEDIATE",
		"none": "NONE",
		"no shared target": "NONE",
		"uncertain": "UNCERTAIN",
		"unknown": "UNCERTAIN",
	}
	canonical_type = type_aliases.get(clean_type)
	rule = "SCALAR_OR_ALIAS_TO_CANONICAL_TARGET"
	if canonical_type is None:
		canonical_type = _text(raw_type).upper() or "UNCERTAIN"
		if canonical_type not in ACCEPTED_TARGET_TYPES and canonical_type != "UNCERTAIN":
			return {"type": canonical_type, "description": raw_description}, {
				"raw_value": raw,
				"canonical_value": {"type": canonical_type, "description": raw_description},
				"normalization_rule": "NORMALIZATION_REFUSED_UNKNOWN_TARGET_TYPE",
				"semantic_change": False,
				"status": "REFUSED",
			}
	if canonical_type in {"ANSWER", "INTERMEDIATE"} and not raw_description:
		existing = _text(proposal.get("shared_answer_type") or proposal.get("requested_final_answer_type") or proposal.get("requested_final_semantic_property"))
		if existing:
			raw_description = f"shared {existing} target"
			rule = "TARGET_TYPE_TO_CANONICAL_OBJECT_USING_EXISTING_METADATA"
		else:
			return {"type": canonical_type, "description": ""}, {
				"raw_value": raw,
				"canonical_value": {"type": canonical_type, "description": ""},
				"normalization_rule": "NORMALIZATION_MISSING_VALUE",
				"semantic_change": False,
				"status": "MISSING",
			}
	if canonical_type == "NONE":
		raw_description = ""
	return {"type": canonical_type, "description": raw_description}, {
		"raw_value": raw,
		"canonical_value": {"type": canonical_type, "description": raw_description},
		"normalization_rule": rule,
		"semantic_change": False,
		"status": "NORMALIZED" if {"type": canonical_type, "description": raw_description} != (raw if isinstance(raw, dict) else {"type": raw, "description": ""}) else "NO_CHANGE",
	}


def _canonical_dependency(value: Any, target_type: str, structure: str) -> tuple[str, dict[str, Any]]:
	raw = _text(value)
	clean = _clean(raw)
	if clean in {"sequential", "linear", "linear chain", "sequential chain", "sequence", "single source direct relation"}:
		canonical = "SEQUENTIAL"
		rule = "KNOWN_SEQUENTIAL_VARIANT"
	elif clean in {"uncertain", "unknown", "ambiguous"} or not clean:
		canonical = "UNCERTAIN" if clean else "UNKNOWN"
		rule = "KNOWN_UNCERTAIN_VARIANT" if clean else "NORMALIZATION_MISSING_VALUE"
	elif "independent" in clean or "parallel" in clean or "branch" in clean:
		if target_type == "INTERMEDIATE" or "intermediate" in clean or "then" in clean or "followed" in clean:
			canonical = "MIXED"
			rule = "INDEPENDENT_INTERMEDIATE_TO_MIXED"
		else:
			canonical = "INDEPENDENT_CONVERGENCE"
			rule = "KNOWN_INDEPENDENT_VARIANT"
	elif "convergent" in clean or "converge" in clean:
		canonical = "MIXED" if target_type == "INTERMEDIATE" or "then" in clean or "followed" in clean else "INDEPENDENT_CONVERGENCE"
		rule = "KNOWN_CONVERGENCE_VARIANT"
	else:
		canonical = "UNKNOWN"
		rule = "NORMALIZATION_REFUSED_UNKNOWN_DEPENDENCY"
	return canonical, {
		"raw_value": value,
		"canonical_value": canonical,
		"normalization_rule": rule,
		"semantic_change": False,
		"status": "NORMALIZED" if canonical in ACCEPTED_DEPENDENCIES else "REFUSED",
	}


def _canonical_role(value: Any) -> tuple[str, dict[str, Any]]:
	raw = _text(value)
	clean = _clean(raw)
	aliases = {
		"answer": "ANSWER",
		"answer target": "ANSWER",
		"final": "ANSWER",
		"final answer": "ANSWER",
		"final answer entity": "ANSWER",
		"final answer target": "ANSWER",
		"intermediate": "INTERMEDIATE",
		"intermediate entity": "INTERMEDIATE",
		"intermediate author": "INTERMEDIATE",
		"intermediate artist": "INTERMEDIATE",
		"intermediate book": "INTERMEDIATE",
		"filter": "FILTER",
		"filter constraint": "FILTER",
		"type": "TYPE",
		"type constraint": "TYPE",
	}
	canonical = aliases.get(clean)
	if canonical is None:
		return "UNKNOWN", {
			"raw_value": value,
			"canonical_value": "UNKNOWN",
			"normalization_rule": "NORMALIZATION_REFUSED_UNKNOWN_ROLE",
			"semantic_change": False,
			"status": "REFUSED",
		}
	return canonical, {
		"raw_value": value,
		"canonical_value": canonical,
		"normalization_rule": "KNOWN_ROLE_ALIAS" if canonical != raw else "NO_CHANGE",
		"semantic_change": False,
		"status": "NORMALIZED" if canonical != raw else "NO_CHANGE",
	}


def normalize_proposal(proposal: Any, question: str) -> dict[str, Any]:
	"""Return a canonical proposal plus auditable, per-field provenance."""
	if not isinstance(proposal, dict):
		return {
			"canonical_proposal": proposal,
			"normalization_status": "NORMALIZATION_FAILED",
			"normalization_errors": ["NORMALIZATION_MISSING_VALUE"],
			"fields": {},
		}
	canonical = copy.deepcopy(proposal)
	fields: dict[str, Any] = {}
	errors: list[str] = []
	changed = False

	canonical_target, target_log = _canonical_target(proposal.get("shared_target"), proposal)
	canonical["shared_target"] = canonical_target
	fields["shared_target"] = target_log
	changed |= target_log["status"] == "NORMALIZED"
	if target_log["status"] in {"MISSING", "REFUSED"}:
		errors.append(target_log["normalization_rule"])

	target_type_for_dependency = canonical_target.get("type", "UNKNOWN")
	canonical_dependency, dependency_log = _canonical_dependency(proposal.get("dependency_pattern"), target_type_for_dependency, _text(proposal.get("structure")))
	canonical["dependency_pattern"] = canonical_dependency
	fields["dependency_pattern"] = dependency_log
	changed |= dependency_log["status"] == "NORMALIZED"
	if dependency_log["status"] == "REFUSED":
		errors.append(dependency_log["normalization_rule"])

	for field in ("requested_final_answer_type", "shared_answer_type"):
		raw_value = proposal.get(field)
		alternate_value = proposal.get("shared_answer_type" if field == "requested_final_answer_type" else "requested_final_answer_type")
		canonical_value, rule = _canonical_target_type(raw_value, question, proposal.get("requested_final_semantic_property"))
		canonical[field] = canonical_value
		fields[field] = {
			"raw_value": raw_value,
			"canonical_value": canonical_value,
			"normalization_rule": rule,
			"semantic_change": False,
			"status": "NORMALIZED" if canonical_value != raw_value else "NO_CHANGE",
		}
		changed |= canonical_value != raw_value and rule != "NORMALIZATION_MISSING_VALUE"
		if rule == "NORMALIZATION_MISSING_VALUE" and not raw_value and not _text(alternate_value):
			errors.append(f"{field}:NORMALIZATION_MISSING_VALUE")

		canonical_property, property_rule = _canonical_property(proposal.get("requested_final_semantic_property"))
		canonical["requested_final_semantic_property"] = canonical_property
		fields["requested_final_semantic_property"] = {
			"raw_value": proposal.get("requested_final_semantic_property"),
			"canonical_value": canonical_property,
			"normalization_rule": property_rule,
			"semantic_change": False,
			"status": "NORMALIZED" if property_rule != "NO_CHANGE" else "NO_CHANGE",
		}
		changed |= property_rule not in {"NO_CHANGE", "NORMALIZATION_MISSING_VALUE"}
		break

	normalized_constraints: list[dict[str, Any]] = []
	for index, item in enumerate(proposal.get("constraints") or [], start=1):
		if not isinstance(item, dict):
			normalized_constraints.append(item)
			continue
		updated = copy.deepcopy(item)
		raw_role = item.get("target_variable_role") if item.get("target_variable_role") is not None else item.get("target_role")
		canonical_role, role_log = _canonical_role(raw_role)
		updated["target_variable_role"] = canonical_role
		updated["target_role"] = canonical_role
		fields[f"constraint_{index}.target_role"] = role_log
		changed |= role_log["status"] == "NORMALIZED"
		if role_log["status"] == "REFUSED":
			errors.append(f"constraint_{index}:{role_log['normalization_rule']}")
		canonical_constraint_property, constraint_property_rule = _canonical_property(item.get("semantic_property") or item.get("property"))
		if "semantic_property" in item or "property" in item:
			updated["semantic_property"] = canonical_constraint_property
		fields[f"constraint_{index}.semantic_property"] = {
			"raw_value": item.get("semantic_property") or item.get("property"),
			"canonical_value": canonical_constraint_property,
			"normalization_rule": constraint_property_rule,
			"semantic_change": False,
			"status": "NORMALIZED" if constraint_property_rule != "NO_CHANGE" else "NO_CHANGE",
		}
		changed |= constraint_property_rule not in {"NO_CHANGE", "NORMALIZATION_MISSING_VALUE"}
		normalized_constraints.append(updated)
	canonical["constraints"] = normalized_constraints

	if errors:
		status = "PARTIAL_NORMALIZATION" if changed else "NORMALIZATION_FAILED"
	elif changed:
		status = "NORMALIZED_SUCCESSFULLY"
	else:
		status = "NO_NORMALIZATION_NEEDED"
	return {
		"canonical_proposal": canonical,
		"normalization_status": status,
		"normalization_errors": errors,
		"fields": fields,
	}
