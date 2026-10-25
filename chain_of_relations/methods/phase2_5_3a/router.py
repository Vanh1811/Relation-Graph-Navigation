"""Phase 2.5.3A independent-source/shared-target extraction intervention.

Only the extraction representation and its prompt are changed here.  Structure
mapping and the inherited v1.1.3 validator remain frozen through the parent
ConstraintFirstRouter implementation.
"""

from __future__ import annotations

import json
from typing import Any

from chain_of_relations.methods.phase2_5.router import (
	CONSTRAINT_ROLES,
	DEPENDENCY_PATTERNS,
	STRUCTURES,
	TARGET_TYPES,
	ConstraintFirstRouter,
	_text,
)
from chain_of_relations.methods.adaptive_v1_1_3.router import extract_requested_answer_target
from chain_of_relations.schema import Entity


ROUTER_PROMPT_VERSION = "phase2.5.3a-source-target-extraction-json-v1"


def _anchor_text(value: Any) -> str:
	if isinstance(value, dict):
		return _text(value.get("id") or value.get("name") or value.get("source_anchor"))
	return _text(value)


class SourceTargetExtractionRouter(ConstraintFirstRouter):
	"""Use explicit source groups and shared-target verification before routing."""

	@staticmethod
	def _normalise_constraint(value: Any, index: int) -> dict[str, Any]:
		if not isinstance(value, dict):
			return {}
		role = _text(value.get("target_variable_role") or value.get("target_role")).upper() or "FILTER"
		text = _text(value.get("constraint_text") or value.get("text"))
		source_anchor = _anchor_text(value.get("source_anchor") or value.get("source_entity"))
		answer_relevant = value.get("answer_relevant")
		if answer_relevant is None:
			answer_relevant = role != "TYPE"
		return {
			"constraint_id": _text(value.get("constraint_id")) or f"c{index}",
			"text": text,
			"constraint_text": text,
			"source_entity": source_anchor,
			"source_anchor": source_anchor,
			"target_role": role,
			"target_variable_role": role,
			"answer_relevant": bool(answer_relevant),
			"semantic_property": _text(value.get("semantic_property") or value.get("property")),
			"depends_on": [
				_text(item) for item in (value.get("depends_on") or []) if _text(item)
			],
		}

	@classmethod
	def _normalise_source_groups(
		cls,
		proposal: dict[str, Any],
		constraints: list[dict[str, Any]],
	) -> list[dict[str, Any]]:
		by_id = {item.get("constraint_id"): item for item in constraints}
		raw_groups = proposal.get("independent_source_groups")
		groups: list[dict[str, Any]] = []
		if isinstance(raw_groups, list):
			for index, raw in enumerate(raw_groups, start=1):
				if not isinstance(raw, dict):
					continue
				anchor = _anchor_text(raw.get("source_anchor") or raw.get("source_entity"))
				ids = [_text(item) for item in (raw.get("constraint_ids") or []) if _text(item)]
				if not ids and anchor:
					ids = [item["constraint_id"] for item in constraints if item.get("source_anchor") == anchor]
				relevant_ids = [item for item in ids if by_id.get(item, {}).get("answer_relevant", True)]
				if anchor and relevant_ids:
					groups.append({
						"source_anchor": anchor,
						"constraint_ids": relevant_ids,
						"group_id": _text(raw.get("group_id")) or f"g{index}",
					})
		if groups:
			return groups

		# Compatibility with the Phase 2.5.1 field while keeping source anchors
		# explicit for the new diagnostic output.
		legacy = proposal.get("independent_sources")
		if isinstance(legacy, list):
			for index, value in enumerate(legacy, start=1):
				anchor = _anchor_text(value)
				if anchor:
					ids = [item["constraint_id"] for item in constraints if item.get("source_anchor") == anchor and item.get("answer_relevant", True)]
					groups.append({"source_anchor": anchor, "constraint_ids": ids, "group_id": f"g{index}"})
		return groups

	@classmethod
	def _extract_analysis(cls, proposal: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], dict[str, str], str, list[str]]:
		constraints = [
			cls._normalise_constraint(item, index)
			for index, item in enumerate(proposal.get("constraints") or [], start=1)
			if isinstance(item, dict)
		]
		groups = cls._normalise_source_groups(proposal, constraints)
		sources = [group["source_anchor"] for group in groups if group.get("source_anchor")]
		raw_target = proposal.get("shared_target")
		shared_target = {
			"type": _text(raw_target.get("type")).upper() if isinstance(raw_target, dict) else "NONE",
			"description": _text(raw_target.get("description")) if isinstance(raw_target, dict) else "",
		}
		failures: list[str] = []
		if not constraints:
			failures.append("constraint extraction returned no constraints")
		if not sources and len([item for item in constraints if item.get("answer_relevant")]) >= 2:
			failures.append("no independent source groups were extracted")
		if shared_target["type"] == "UNCERTAIN":
			# Keep the frozen parent mapping contract: uncertainty cannot create a
			# branch, but preserve it as a diagnostic warning in route history.
			shared_target["type"] = "NONE"
			failures.append("shared target was uncertain")
		if shared_target["type"] not in TARGET_TYPES:
			failures.append("shared_target.type must be ANSWER, INTERMEDIATE, or NONE")
		dependency = _text(proposal.get("dependency_pattern")).upper() or "UNCERTAIN"
		if dependency not in DEPENDENCY_PATTERNS:
			failures.append("dependency_pattern must be SEQUENTIAL, INDEPENDENT_CONVERGENCE, MIXED, or UNCERTAIN")
		for index, item in enumerate(constraints, start=1):
			if item.get("target_role") not in CONSTRAINT_ROLES:
				failures.append(f"constraint_{index} has invalid target_role")
			if not item.get("text"):
				failures.append(f"constraint_{index} has empty constraint_text")
			if not item.get("semantic_property"):
				failures.append(f"constraint_{index} has empty semantic_property")
		if shared_target["type"] in {"ANSWER", "INTERMEDIATE"} and not shared_target["description"]:
			failures.append("shared_target has no description")
		return constraints, sources, shared_target, dependency, failures

	@staticmethod
	def _completeness_check(
		proposal: Any,
		constraints: list[dict[str, Any]],
		groups: list[dict[str, Any]],
		topic_entities: list[Entity],
	) -> dict[str, Any]:
		check = proposal.get("extraction_self_check") if isinstance(proposal, dict) else {}
		if not isinstance(check, dict):
			check = {}
		answer_constraints = [item for item in constraints if item.get("answer_relevant")]
		raw_target = proposal.get("shared_target") if isinstance(proposal, dict) else None
		default_target_verified = isinstance(raw_target, dict) and bool(raw_target.get("type"))
		return {
			"all_answer_relevant_constraints_present": bool(check.get("all_answer_relevant_constraints_present", bool(answer_constraints))),
			"all_independent_sources_present": bool(check.get("all_independent_sources_present", len(groups) >= 2 or len(topic_entities) <= 1)),
			"shared_target_verified": bool(check.get("shared_target_verified", default_target_verified)),
			"source_group_count": len(groups),
			"answer_relevant_constraint_count": len(answer_constraints),
			"suspicious": bool(check.get("correction_needed", False)) or (
				len(topic_entities) >= 2 and len(groups) < 2 and len(answer_constraints) >= 2
			),
			"self_check_note": _text(check.get("correction_note")),
		}

	def _build_prompt(self, question: str, topic_entities: list[Entity]) -> str:
		entities = [{"id": str(entity.id), "name": str(entity.name)} for entity in topic_entities]
		hint = extract_requested_answer_target(question)
		return (
			"You are the Phase 2.5.3A extraction-first router for a relation-centric knowledge-graph QA system.\n"
			"Use only the question and linked runtime entities. Never use gold answers, gold SPARQL, S-expressions, or dataset labels.\n\n"
			"Do extraction before structure classification. First list every explicit answer-relevant constraint.\n"
			"For each constraint, identify its source_anchor: a grounded runtime entity or an explicit literal/date/value condition.\n"
			"Do not use generic answer-type words as source anchors. TYPE-only descriptions are not independent answer sources.\n"
			"Then group constraints by independent source anchor. Ask: which constraints begin from different source anchors, and do they independently restrict the same answer or intermediate variable?\n"
			"Do not collapse independent constraints merely because their semantic properties differ. Do not create a source group for a generic type word, answer-type word, or non-answer-relevant modifier.\n"
			"After source grouping, explicitly identify the convergence target. Use shared_target ANSWER only when independent constraints directly identify the final requested entity/value. Use INTERMEDIATE when they converge before a later answer step. Use NONE when there is no meaningful convergence. Use UNCERTAIN only when the evidence is insufficient.\n"
			"For a sequential chain such as Entity A -> intermediate -> answer, keep one dependent chain rather than inventing independent sources.\n\n"
			"Return JSON only with these extraction fields: constraints, independent_source_groups, shared_target, dependency_pattern, extraction_self_check.\n"
			"Each constraint must contain constraint_id, source_anchor, constraint_text, target_variable_role, answer_relevant, semantic_property, depends_on.\n"
			"Each independent_source_group must contain source_anchor and constraint_ids.\n"
			"extraction_self_check must contain all_answer_relevant_constraints_present, all_independent_sources_present, shared_target_verified, correction_needed, correction_note.\n"
			"Also return the existing routing fields: structure, execution_mode, confidence, reason, requested_final_answer_type, requested_final_semantic_property, shared_answer_type, branches, unsupported_structure.\n"
			"Keep the existing structure labels exactly: LINEAR, BRANCH_FINAL, BRANCH_INTERMEDIATE, or UNCERTAIN.\n"
			"For BRANCH_FINAL, branches use the existing v1.1.3 fields; this phase does not change branch validation or execution.\n\n"
			f"Question: {question}\n"
			f"Lexical safety hint (supporting evidence only): {json.dumps(hint, ensure_ascii=False)}\n"
			f"Runtime topic entities: {json.dumps(entities, ensure_ascii=False)}\n"
		)

	def route(self, question: str, topic_entities: list[Entity], prompt_history: list[dict[str, Any]]) -> Any:
		prompt = self._build_prompt(question, topic_entities)
		response, usage = self.llm_generate(
			prompt,
			temperature=0.1,
			max_tokens=1024,
			system_prompt="Return one Phase 2.5.3A JSON extraction and routing object and no markdown.",
		)
		parsed = self._extract_json(response)
		retry_used = False
		retry_reason = ""
		if not isinstance(parsed, dict):
			retry_used = True
			retry_reason = "empty_model_response" if not response else "invalid_extraction_json"
			retry_prompt = (
				prompt
				+ "\n\nExtraction self-check failed because the previous response was empty or invalid. "
				"Retry exactly once. Return the complete JSON object, including constraints, "
				"independent_source_groups, shared_target, dependency_pattern, and extraction_self_check."
			)
			retry_response, retry_usage = self.llm_generate(
				retry_prompt,
				temperature=0.1,
				max_tokens=1024,
				system_prompt="Return one Phase 2.5.3A JSON extraction and routing object and no markdown.",
			)
			response = retry_response
			parsed = self._extract_json(response)
			usage = {
				"input_tokens": int((usage or {}).get("input_tokens", 0) or 0) + int((retry_usage or {}).get("input_tokens", 0) or 0),
				"output_tokens": int((usage or {}).get("output_tokens", 0) or 0) + int((retry_usage or {}).get("output_tokens", 0) or 0),
				"total_tokens": int((usage or {}).get("total_tokens", 0) or 0) + int((retry_usage or {}).get("total_tokens", 0) or 0),
			}
			if not isinstance(parsed, dict) and retry_usage and retry_usage.get("error"):
				usage["error"] = retry_usage["error"]
		decision = self._validate(parsed, question, topic_entities)
		constraints = []
		groups = []
		if isinstance(parsed, dict):
			constraints = [
				self._normalise_constraint(item, index)
				for index, item in enumerate(parsed.get("constraints") or [], start=1)
				if isinstance(item, dict)
			]
			groups = self._normalise_source_groups(parsed, constraints)
		completeness = self._completeness_check(parsed, constraints, groups, topic_entities)
		prompt_history.append({
			"type": "phase2_5_3a_source_target_extraction_router",
			"prompt": prompt,
			"response": response or "",
			"llm_error": _text((usage or {}).get("error")),
			"input_tokens": int((usage or {}).get("input_tokens", 0) or 0),
			"output_tokens": int((usage or {}).get("output_tokens", 0) or 0),
			"total_tokens": int((usage or {}).get("total_tokens", 0) or 0),
			"parsed_result": parsed if parsed is not None else {},
			"constraints": constraints,
			"independent_source_groups": groups,
			"independent_source_count": len(groups),
			"shared_target": dict(decision.shared_target),
			"dependency_pattern": decision.dependency_pattern,
			"validator_decision": decision.validator_decision,
			"validation_failures": list(decision.validation_failures or []),
			"extraction_completeness": completeness,
			"retry_used": retry_used,
			"retry_reason": retry_reason,
			"prompt_version": ROUTER_PROMPT_VERSION,
		})
		return decision
