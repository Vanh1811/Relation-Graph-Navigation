"""Constraint-first routing for Phase 2.5.

The intervention keeps the v1.1.3 structure labels and execution contract, but
requires the model to emit a compact constraint/dependency representation
before the final route.  It is deliberately isolated from the v1.1.3 router.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from chain_of_relations.methods.adaptive_v1_1_3.router import (
	AdaptiveRouter as AdaptiveRouterV113,
	RouterConfig,
	RouterDecision as RouterDecisionV113,
	STRUCTURES,
	_normalise,
	extract_requested_answer_target,
)
from chain_of_relations.schema import Entity


ROUTER_PROMPT_VERSION = "phase2.5-constraint-first-routing-json-v1"
CONSTRAINT_ROLES = {"ANSWER", "INTERMEDIATE", "FILTER", "TYPE"}
DEPENDENCY_PATTERNS = {"SEQUENTIAL", "INDEPENDENT_CONVERGENCE", "MIXED", "UNCERTAIN"}
TARGET_TYPES = {"ANSWER", "INTERMEDIATE", "NONE"}


def _text(value: Any) -> str:
	return str(value or "").strip()


def _normalise_target(value: Any) -> dict[str, str]:
	if not isinstance(value, dict):
		return {"type": "NONE", "description": ""}
	target_type = _text(value.get("type")).upper() or "NONE"
	return {
		"type": target_type,
		"description": _text(value.get("description")),
	}


@dataclass
class ConstraintFirstDecision(RouterDecisionV113):
	"""v1.1.3 decision plus the pre-classification representation."""

	constraints: list[dict[str, Any]] = field(default_factory=list)
	independent_sources: list[str] = field(default_factory=list)
	shared_target: dict[str, str] = field(default_factory=lambda: {"type": "NONE", "description": ""})
	dependency_pattern: str = "UNCERTAIN"
	validator_decision: str = "FALLBACK"

	def as_dict(self) -> dict[str, Any]:
		result = super().as_dict()
		result.update({
			"constraints": list(self.constraints),
			"independent_sources": list(self.independent_sources),
			"shared_target": dict(self.shared_target),
			"dependency_pattern": self.dependency_pattern,
			"validator_decision": self.validator_decision,
		})
		return result


class ConstraintFirstRouter(AdaptiveRouterV113):
	"""Ask for constraints first, then map the extracted graph to a route."""

	@staticmethod
	def _normalise_constraint(value: Any, index: int) -> dict[str, Any]:
		if not isinstance(value, dict):
			return {}
		role = _text(value.get("target_role")).upper() or "FILTER"
		return {
			"constraint_id": _text(value.get("constraint_id")) or f"c{index}",
			"text": _text(value.get("text")),
			"source_entity": _text(value.get("source_entity")),
			"target_role": role,
			"semantic_property": _text(value.get("semantic_property")),
			"depends_on": [
				_text(item) for item in (value.get("depends_on") or []) if _text(item)
			],
		}

	@classmethod
	def _extract_analysis(cls, proposal: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], dict[str, str], str, list[str]]:
		constraints = [
			cls._normalise_constraint(item, index)
			for index, item in enumerate(proposal.get("constraints") or [], start=1)
			if isinstance(item, dict)
		]
		sources = [_text(item) for item in (proposal.get("independent_sources") or []) if _text(item)]
		shared_target = _normalise_target(proposal.get("shared_target"))
		dependency = _text(proposal.get("dependency_pattern")).upper() or "UNCERTAIN"
		failures: list[str] = []
		if not constraints:
			failures.append("constraint extraction returned no constraints")
		if dependency not in DEPENDENCY_PATTERNS:
			failures.append("dependency_pattern must be SEQUENTIAL, INDEPENDENT_CONVERGENCE, MIXED, or UNCERTAIN")
		if shared_target["type"] not in TARGET_TYPES:
			failures.append("shared_target.type must be ANSWER, INTERMEDIATE, or NONE")
		for index, item in enumerate(constraints, start=1):
			if item.get("target_role") not in CONSTRAINT_ROLES:
				failures.append(f"constraint_{index} has invalid target_role")
			if not item.get("text"):
				failures.append(f"constraint_{index} has empty text")
			if not item.get("semantic_property"):
				failures.append(f"constraint_{index} has empty semantic_property")
		if shared_target["type"] in {"ANSWER", "INTERMEDIATE"} and not shared_target["description"]:
			failures.append("shared_target has no description")
		return constraints, sources, shared_target, dependency, failures

	@staticmethod
	def _structure_from_extraction(
		proposal_structure: str,
		constraints: list[dict[str, Any]],
		independent_sources: list[str],
		shared_target: dict[str, str],
		dependency_pattern: str,
	) -> str:
		roles = [item.get("target_role") for item in constraints]
		# FILTER includes explicit literal/date/value constraints.  Those can be
		# answer-disambiguating even though generic TYPE information usually is not.
		meaningful = [role for role in roles if role != "TYPE"]
		if shared_target["type"] == "INTERMEDIATE" and len(constraints) >= 2:
			return "BRANCH_INTERMEDIATE"
		if (
			shared_target["type"] == "ANSWER"
			and dependency_pattern == "INDEPENDENT_CONVERGENCE"
			and len(constraints) >= 2
			and len(independent_sources) >= 2
			and meaningful
		):
			return "BRANCH_FINAL"
		if dependency_pattern == "SEQUENTIAL":
			return "LINEAR"
		if dependency_pattern == "MIXED" and shared_target["type"] == "NONE":
			return "UNCERTAIN"
		return proposal_structure if proposal_structure in STRUCTURES else "UNCERTAIN"

	def _validate(self, proposal: Any, question: str, topic_entities: list[Entity]) -> ConstraintFirstDecision:
		hint = extract_requested_answer_target(question)
		if not isinstance(proposal, dict):
			return ConstraintFirstDecision(
				requested_final_answer_type=hint["requested_final_answer_type"],
				requested_final_semantic_property=hint["requested_final_semantic_property"],
				validation_passed=False,
				validation_failures=["router output was not a JSON object"],
				reason="Invalid constraint-first output; execute unchanged CoR.",
			)

		constraints, sources, shared_target, dependency, extraction_failures = self._extract_analysis(proposal)
		proposal_structure = _text(proposal.get("structure")).upper()
		structure = self._structure_from_extraction(
			proposal_structure, constraints, sources, shared_target, dependency,
		)

		# A lexical hint is supporting evidence for the answer type, not a reason
		# to discard an otherwise valid constraint graph.  It also prevents the
		# old parent-gender/genus inversion without making type conflict the route.
		answer_type = hint["requested_final_answer_type"] or _text(proposal.get("requested_final_answer_type"))
		if not answer_type:
			answer_type = _text(proposal.get("shared_answer_type")) or shared_target.get("description", "")
		semantic_property = hint["requested_final_semantic_property"] or _text(proposal.get("requested_final_semantic_property"))
		if not semantic_property:
			properties = [item.get("semantic_property", "") for item in constraints if item.get("semantic_property")]
			semantic_property = properties[0] if properties else "constraint graph"
		shared_type = _text(proposal.get("shared_answer_type")) or answer_type
		confidence = self._confidence(proposal.get("confidence"))
		reason = _text(proposal.get("reason"))
		raw_branches = proposal.get("branches")
		branches = [self._normalise_branch(item) for item in raw_branches] if isinstance(raw_branches, list) else []

		# Reuse the deterministic v1.1.3 checks for branch identity, runtime
		# roots, confidence, and required metadata.  Constraint-first extraction
		# deliberately relaxes only the two checks that previously short-circuited
		# structural evidence: answer-type conflict and property mismatch.
		base_proposal = dict(proposal)
		base_proposal.update({
			"structure": structure,
			"requested_final_answer_type": answer_type,
			"requested_final_semantic_property": semantic_property,
			"shared_answer_type": shared_type,
			"branches": branches,
			"branch_count": len(branches),
		})
		base = super()._validate(base_proposal, question, topic_entities)
		failures = list(base.validation_failures or [])
		if structure == "BRANCH_FINAL" and shared_target["type"] == "ANSWER" and len(sources) >= 2:
			failures = [
				item for item in failures
				if not item.startswith("ANSWER_TYPE_MISMATCH:")
				and not item.startswith("CONSTRAINT_PROPERTY_MISMATCH:")
			]
		failures.extend(extraction_failures)
		if structure == "BRANCH_FINAL":
			if shared_target["type"] != "ANSWER":
				failures.append("BRANCH_FINAL requires shared_target.type=ANSWER")
			if dependency not in {"INDEPENDENT_CONVERGENCE", "MIXED"}:
				failures.append("BRANCH_FINAL requires an independent or mixed dependency pattern")
			if len(sources) < 2:
				failures.append("BRANCH_FINAL requires at least two independent source descriptions")
			if len(constraints) < 2:
				failures.append("BRANCH_FINAL requires at least two extracted constraints")
			if not any(item.get("target_role") in {"ANSWER", "INTERMEDIATE", "FILTER"} for item in constraints):
				failures.append("branch plan cannot be supported by type-only constraints")
		elif structure == "BRANCH_INTERMEDIATE":
			if shared_target["type"] != "INTERMEDIATE":
				failures.append("BRANCH_INTERMEDIATE requires shared_target.type=INTERMEDIATE")
		elif structure == "LINEAR":
			if dependency != "SEQUENTIAL":
				failures.append("LINEAR requires dependency_pattern=SEQUENTIAL")

		# For non-branch routes, executable branches are never allowed.  The
		# inherited validator already checks this, but keep the diagnostic cause
		# explicit when extraction itself is invalid.
		if structure in {"LINEAR", "UNCERTAIN", "BRANCH_INTERMEDIATE"} and branches:
			failures.append(f"{structure} must not include executable branch objects")
		validation_passed = not failures
		execution_mode = "branch" if structure == "BRANCH_FINAL" and validation_passed else "cor"
		return ConstraintFirstDecision(
			structure=structure if structure in STRUCTURES else "UNCERTAIN",
			execution_mode=execution_mode,
			confidence=confidence,
			reason=reason,
			requested_final_answer_type=answer_type,
			requested_final_semantic_property=semantic_property,
			shared_answer_type=shared_type,
			proposed_branches=branches if structure == "BRANCH_FINAL" and validation_passed else [],
			branch_property_compatibility=(True if structure == "BRANCH_FINAL" and validation_passed else None),
			unsupported_structure=structure == "BRANCH_INTERMEDIATE",
			validation_passed=validation_passed,
			validation_failures=failures,
			constraints=constraints,
			independent_sources=sources,
			shared_target=shared_target,
			dependency_pattern=dependency,
			validator_decision="PASS" if validation_passed else "FALLBACK",
		)

	def _build_prompt(self, question: str, topic_entities: list[Entity]) -> str:
		entities = [{"id": str(entity.id), "name": str(entity.name)} for entity in topic_entities]
		hint = extract_requested_answer_target(question)
		return (
			"You are the Phase 2.5 constraint-first router for a relation-centric knowledge-graph QA system.\n"
			"Use only the question and linked runtime entities. Never use gold answers, gold SPARQL, S-expressions, or dataset labels.\n\n"
			"First extract every essential answer-relevant constraint, then classify the dependency structure. Do not treat every noun phrase as a branch.\n"
			"A constraint may be an entity relation, a type/filter, or a literal/date/value condition.\n"
			"Use target_role TYPE or FILTER for generic/type information unless it independently constrains the answer.\n"
			"independent_sources must describe two or more distinct constraint sources; a grounded entity and a literal/date/value condition may be distinct sources.\n"
			"shared_target.type is ANSWER when constraints converge on the final answer, INTERMEDIATE when they converge before a later answer step, otherwise NONE.\n"
			"dependency_pattern is SEQUENTIAL, INDEPENDENT_CONVERGENCE, MIXED, or UNCERTAIN.\n"
			"Do not reject a branch solely because properties are complementary or because a lexical answer-type cue is imperfect; represent the structure first.\n\n"
			"Map the extracted graph to exactly one existing structure: LINEAR, BRANCH_FINAL, BRANCH_INTERMEDIATE, or UNCERTAIN.\n"
			"BRANCH_FINAL requires independent constraints converging on the final answer. BRANCH_INTERMEDIATE requires convergence on an intermediate and remains CoR-executed.\n"
			"Return JSON only with fields: constraints, independent_sources, shared_target, dependency_pattern, structure, execution_mode, confidence, reason, requested_final_answer_type, requested_final_semantic_property, shared_answer_type, branches, unsupported_structure.\n"
			"Each constraint must contain constraint_id, text, source_entity, target_role, semantic_property, depends_on.\n"
			"For BRANCH_FINAL, branches use the existing v1.1.3 fields. For other structures branches must be [].\n\n"
			f"Question: {question}\n"
			f"Lexical safety hint (supporting evidence only): {json.dumps(hint, ensure_ascii=False)}\n"
			f"Runtime topic entities: {json.dumps(entities, ensure_ascii=False)}\n"
		)

	def route(self, question: str, topic_entities: list[Entity], prompt_history: list[dict[str, Any]]) -> ConstraintFirstDecision:
		prompt = self._build_prompt(question, topic_entities)
		response, usage = self.llm_generate(
			prompt,
			temperature=0.1,
			max_tokens=1024,
			system_prompt="Return one Phase 2.5 JSON routing object and no markdown.",
		)
		parsed = self._extract_json(response)
		decision = self._validate(parsed, question, topic_entities)
		prompt_history.append({
			"type": "phase2_5_constraint_first_router",
			"prompt": prompt,
			"response": response or "",
			"llm_error": str((usage or {}).get("error", "")),
			"input_tokens": int((usage or {}).get("input_tokens", 0) or 0),
			"output_tokens": int((usage or {}).get("output_tokens", 0) or 0),
			"total_tokens": int((usage or {}).get("total_tokens", 0) or 0),
			"parsed_result": parsed if parsed is not None else {},
			"requested_final_answer_type": decision.requested_final_answer_type,
			"requested_final_semantic_property": decision.requested_final_semantic_property,
			"constraint_count": len(decision.constraints),
			"independent_source_count": len(decision.independent_sources),
			"shared_target": dict(decision.shared_target),
			"dependency_pattern": decision.dependency_pattern,
			"validator_decision": decision.validator_decision,
			"warnings": list(decision.validation_failures or []),
			"prompt_version": ROUTER_PROMPT_VERSION,
		})
		return decision
