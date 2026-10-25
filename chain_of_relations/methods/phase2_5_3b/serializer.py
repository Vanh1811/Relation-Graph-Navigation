"""Deterministic adapter from Phase 2.5.3A semantics to the v1.1.3 branch contract."""

from __future__ import annotations

from typing import Any

from chain_of_relations.methods.adaptive_v1_1_3.router import (
	extract_requested_answer_target,
	_normalise,
)
from chain_of_relations.schema import Entity


STATUS_COMPLETE = "COMPLETE"
STATUS_PARTIAL = "PARTIAL"
STATUS_FAILED = "FAILED"


def _text(value: Any) -> str:
	return str(value or "").strip()


def _shared_target(value: Any) -> dict[str, str]:
	"""Normalize the existing semantic value without changing its meaning."""
	if isinstance(value, dict):
		return {
			"type": _text(value.get("type")).upper() or "NONE",
			"description": _text(value.get("description")),
		}
	if isinstance(value, str):
		return {"type": value.strip().upper() or "NONE", "description": ""}
	return {"type": "NONE", "description": ""}


def _dedupe(values: list[str]) -> list[str]:
	result: list[str] = []
	for value in values:
		if value and value not in result:
			result.append(value)
	return result


def _resolve_entity(anchor: str, topic_entities: list[Entity]) -> Entity | None:
	anchor_text = _text(anchor)
	if not anchor_text:
		return None
	by_id = {str(entity.id).strip(): entity for entity in topic_entities if str(entity.id).strip()}
	by_name = {_normalise(entity.name): entity for entity in topic_entities if _normalise(entity.name)}
	return by_id.get(anchor_text) or by_name.get(_normalise(anchor_text))


def _constraint_sort_key(item: dict[str, Any], positions: dict[str, int]) -> tuple[int, str, str]:
	constraint_id = _text(item.get("constraint_id"))
	return (positions.get(constraint_id, 10**6), _normalise(item.get("source_anchor")), constraint_id)


def serialize_branch_plan(
	proposal: Any,
	constraints: list[dict[str, Any]],
	groups: list[dict[str, Any]],
	topic_entities: list[Entity],
	question: str,
) -> dict[str, Any]:
	"""Create only fields supported by the existing executable branch contract."""
	if not isinstance(proposal, dict):
		return {
			"semantic_plan_exists": False,
			"serialization_scope": "UNKNOWN",
			"serialization_status": STATUS_FAILED,
			"serialization_errors": ["NO_EXECUTABLE_BRANCHES"],
			"serialized_branches": [],
			"normalized_shared_target": {"type": "NONE", "description": ""},
			"target_answer_type": "UNKNOWN",
			"field_provenance": {},
		}

	structure = _text(proposal.get("structure")).upper()
	shared_target = _shared_target(proposal.get("shared_target"))
	if structure != "BRANCH_FINAL":
		return {
			"semantic_plan_exists": bool(constraints or groups),
			"serialization_scope": "NON_EXECUTABLE_STRUCTURE",
			"serialization_status": STATUS_COMPLETE,
			"serialization_errors": [],
			"serialized_branches": [],
			"normalized_shared_target": shared_target,
			"target_answer_type": _text(proposal.get("requested_final_answer_type"))
				or _text(proposal.get("shared_answer_type"))
				or "UNKNOWN",
			"field_provenance": {},
		}

	requested_type = _text(proposal.get("requested_final_answer_type"))
	target_type_source = "DIRECT_FROM_EXTRACTION"
	if not requested_type:
		requested_type = _text(proposal.get("shared_answer_type"))
	if not requested_type:
		hint = extract_requested_answer_target(question)
		requested_type = _text(hint.get("requested_final_answer_type"))
		target_type_source = "LOOKED_UP_FROM_EXISTING_ROUTER_STATE"
	if not requested_type:
		requested_type = "UNKNOWN"
		target_type_source = "UNAVAILABLE"

	by_id = {item.get("constraint_id"): item for item in constraints if item.get("constraint_id")}
	positions = {item.get("constraint_id"): index for index, item in enumerate(constraints)}
	ordered_groups = sorted(
		[group for group in groups if isinstance(group, dict)],
		key=lambda group: min(
			(positions.get(_text(value), 10**6) for value in (group.get("constraint_ids") or [])),
			default=10**6,
		),
	)
	errors: list[str] = []
	serialized: list[dict[str, Any]] = []
	provenance: dict[str, str] = {
		"branch_id": "DETERMINISTICALLY_DERIVED",
		"root_entity_id": "LOOKED_UP_FROM_EXISTING_RUNTIME_STATE",
		"root_entity_name": "LOOKED_UP_FROM_EXISTING_RUNTIME_STATE",
		"constraint_property": "DIRECT_FROM_EXTRACTION",
		"constraint_summary": "DIRECT_FROM_EXTRACTION",
		"target_answer_type": target_type_source,
	}

	if len(ordered_groups) > 3:
		errors.append("INVALID_BRANCH_COUNT")

	for index, group in enumerate(ordered_groups[:3], start=1):
		constraint_ids = [_text(value) for value in (group.get("constraint_ids") or []) if _text(value)]
		selected = [by_id[value] for value in constraint_ids if value in by_id]
		if not selected:
			errors.append("NO_EXECUTABLE_BRANCHES")
			continue
		selected.sort(key=lambda item: _constraint_sort_key(item, positions))
		anchor = _text(group.get("source_anchor")) or _text(selected[0].get("source_anchor"))
		entity = _resolve_entity(anchor, topic_entities)
		if entity is None:
			errors.append("ROOT_ENTITY_UNRESOLVED")
			continue

		properties = _dedupe([_text(item.get("semantic_property")) for item in selected])
		if not properties:
			errors.append("MISSING_CONSTRAINT_PROPERTY")
			continue
		if len(properties) > 1:
			errors.append("CONSTRAINT_PROPERTY_UNAVAILABLE")
			continue

		summaries = _dedupe([_text(item.get("constraint_text") or item.get("text")) for item in selected])
		if not summaries:
			errors.append("MISSING_SUMMARY")
			continue
		if requested_type == "UNKNOWN":
			errors.append("MISSING_TARGET_TYPE")

		serialized.append({
			"branch_id": f"branch_{index}",
			"root_entity_id": str(entity.id),
			"root_entity_name": str(entity.name),
			"constraint_property": properties[0],
			"constraint_summary": " AND ".join(summaries),
			"target_answer_type": requested_type,
		})

	if len(serialized) < 2:
		errors.append("NO_EXECUTABLE_BRANCHES")
	status = STATUS_COMPLETE
	if errors:
		status = STATUS_PARTIAL if serialized else STATUS_FAILED
	if len(serialized) < 2:
		status = STATUS_FAILED if not serialized else STATUS_PARTIAL

	return {
		"semantic_plan_exists": bool(constraints or groups),
		"serialization_scope": "EXECUTABLE_BRANCH_PLAN",
		"serialization_status": status,
		"serialization_errors": _dedupe(errors),
		"serialized_branches": serialized,
		"normalized_shared_target": shared_target,
		"target_answer_type": requested_type,
		"field_provenance": provenance,
	}
