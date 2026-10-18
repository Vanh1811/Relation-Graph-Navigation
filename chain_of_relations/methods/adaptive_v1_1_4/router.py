"""Adaptive v1.1.4 routing with explicit complementary-property validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from chain_of_relations.schema import Entity
from chain_of_relations.methods.adaptive_v1_1_3.router import (
	AdaptiveRouter as AdaptiveRouterV113,
	RouterConfig,
	RouterDecision as RouterDecisionV113,
	STRUCTURES,
	_TYPE_ALIASES,
	_INCOMPATIBLE_PROPERTY_PAIRS,
	_normalise,
	_semantic_group,
	extract_requested_answer_target,
)


MAX_BRANCHES = 3
ROUTER_PROMPT_VERSION = "adaptive-routing-v1.1.4-complementary-property-json-v1"
BRANCH_RELATION_MODES = {"SAME_PROPERTY", "COMPLEMENTARY_PROPERTIES"}


def _as_bool(value: Any) -> bool:
	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)):
		return bool(value)
	return str(value or "").strip().casefold() in {"true", "1", "yes", "y"}


@dataclass
class RouterDecision(RouterDecisionV113):
	branch_relation_mode: str = ""
	same_final_answer_space: bool = False
	validation_reason: str = ""

	def as_dict(self) -> dict[str, Any]:
		result = super().as_dict()
		result.update({
			"branch_relation_mode": self.branch_relation_mode,
			"same_final_answer_space": self.same_final_answer_space,
			"validation_reason": self.validation_reason,
		})
		return result


class AdaptiveRouter(AdaptiveRouterV113):
	"""v1.1.3 router with a narrow complementary-property relaxation."""

	def _normalise_branch(self, branch: Any) -> dict[str, Any]:
		result = super()._normalise_branch(branch)
		if not result:
			return result
		result.update({
			"question_supported": _as_bool(branch.get("question_supported")),
			"independently_generates_final_candidates": _as_bool(
				branch.get("independently_generates_final_candidates")
			),
		})
		return result

	@staticmethod
	def _same_property(properties: list[str], requested: str) -> bool:
		groups = [_semantic_group(value) for value in properties]
		if not groups or len(groups) != len(properties) or len(set(groups)) != 1:
			return False
		requested_group = _semantic_group(requested)
		return not requested_group or groups[0] == requested_group

	@staticmethod
	def _complementary_properties(properties: list[str]) -> bool:
		groups = [_semantic_group(value) for value in properties]
		if not groups or len(groups) != len(properties) or len(set(groups)) < 2:
			return False
		for left_index, left in enumerate(groups):
			for right in groups[left_index + 1:]:
				if frozenset({left, right}) in _INCOMPATIBLE_PROPERTY_PAIRS:
					return False
		return True

	def _validate(self, proposal: Any, question: str, topic_entities: list[Entity]) -> RouterDecision:
		hint = extract_requested_answer_target(question)
		if not isinstance(proposal, dict):
			return RouterDecision(
				requested_final_answer_type=hint["requested_final_answer_type"],
				requested_final_semantic_property=hint["requested_final_semantic_property"],
				validation_passed=False,
				validation_failures=["router output was not a JSON object"],
				reason="Invalid router output; execute unchanged CoR.",
				validation_reason="invalid_branch_plan",
			)

		structure = str(proposal.get("structure", "")).strip().upper()
		if not structure and str(proposal.get("mode", "")).casefold() == "cor":
			structure = "LINEAR"
		confidence = self._confidence(proposal.get("confidence"))
		reason = str(proposal.get("reason", "")).strip()
		type_value = str(proposal.get("requested_final_answer_type", "")).strip() or str(proposal.get("shared_answer_type", "")).strip()
		property_value = str(proposal.get("requested_final_semantic_property", "")).strip()
		shared_type = str(proposal.get("shared_answer_type", "")).strip() or type_value
		if not type_value:
			type_value = hint["requested_final_answer_type"]
		if not property_value:
			property_value = hint["requested_final_semantic_property"]
		raw_branches = proposal.get("branches", [])
		branches = [self._normalise_branch(item) for item in raw_branches] if isinstance(raw_branches, list) else []
		failures: list[str] = []
		branch_relation_mode = str(proposal.get("branch_relation_mode", "")).strip().upper()
		same_final_answer_space = _as_bool(proposal.get("same_final_answer_space"))

		if structure not in STRUCTURES:
			failures.append("structure must be LINEAR, BRANCH_FINAL, BRANCH_INTERMEDIATE, or UNCERTAIN")
		if confidence < self.config.confidence_threshold:
			failures.append(f"confidence {confidence:.3f} is below threshold {self.config.confidence_threshold:.3f}")
		if hint["requested_final_answer_type"] and _semantic_group(type_value) and _semantic_group(type_value) != _semantic_group(hint["requested_final_answer_type"]):
			failures.append("ANSWER_TYPE_MISMATCH: router target conflicts with explicit question wording")
		if not type_value:
			failures.append("requested_final_answer_type is empty")
		if not property_value:
			failures.append("requested_final_semantic_property is empty")

		if structure in {"LINEAR", "UNCERTAIN", "BRANCH_INTERMEDIATE"}:
			if branches:
				failures.append(f"{structure} must not include executable branch objects")
			execution_mode = "cor"
		else:
			execution_mode = "branch"
			try:
				proposed_count = int(proposal.get("branch_count", len(branches)))
			except (TypeError, ValueError):
				proposed_count = -1
			if proposed_count < 2:
				failures.append("BRANCH_FINAL requires at least two branches")
			if proposed_count > self.config.max_branches:
				failures.append(f"branch_count exceeds MAX_BRANCHES={self.config.max_branches}")
			if len(branches) != proposed_count:
				failures.append("number of branch objects does not equal branch_count")
			if not shared_type:
				failures.append("shared_answer_type is empty")
			if len(topic_entities) < max(proposed_count, 0):
				failures.append("not enough runtime topic entities for proposed branches")
			if branch_relation_mode not in BRANCH_RELATION_MODES:
				failures.append("branch_relation_mode must be SAME_PROPERTY or COMPLEMENTARY_PROPERTIES")
			if not same_final_answer_space:
				failures.append("branches do not explicitly claim the same final answer space")

		known_by_id = {str(entity.id).strip(): entity for entity in topic_entities if str(entity.id).strip()}
		known_by_name = {_normalise(entity.name): entity for entity in topic_entities if _normalise(entity.name)}
		root_keys: list[str] = []
		branch_signatures: list[tuple[str, str]] = []
		branch_properties: list[str] = []
		branch_types: list[str] = []
		if structure == "BRANCH_FINAL":
			for index, branch in enumerate(branches):
				prefix = f"branch_{index + 1}"
				if not branch.get("branch_id"):
					failures.append(f"{prefix} has no branch_id")
				if not branch.get("constraint_property"):
					failures.append(f"{prefix} has no constraint_property")
				if not branch.get("constraint_summary"):
					failures.append(f"{prefix} has an empty constraint_summary")
				if not branch.get("target_answer_type"):
					failures.append(f"{prefix} has no target_answer_type")
				if not branch.get("question_supported"):
					failures.append(f"{prefix} constraint is not explicitly marked question-supported")
				if not branch.get("independently_generates_final_candidates"):
					failures.append(f"{prefix} does not independently generate final-answer candidates")
				entity = known_by_id.get(branch.get("root_entity_id")) or known_by_name.get(_normalise(branch.get("root_entity_name")))
				if entity is None:
					failures.append(f"{prefix} root is not a linked runtime topic entity")
					continue
				if self.is_generic_root(entity.name):
					failures.append(f"{prefix} root is generic/type-like: {entity.name}")
				root_key = str(entity.id).strip() or _normalise(entity.name)
				root_keys.append(root_key)
				branch_signatures.append((root_key, _normalise(branch.get("constraint_summary"))))
				branch_properties.append(branch.get("constraint_property", ""))
				branch_types.append(branch.get("target_answer_type", ""))
			if len(root_keys) != len(set(root_keys)):
				failures.append("branch roots are duplicated")
			if len(branch_signatures) != len(set(branch_signatures)):
				failures.append("duplicate or redundant branch root and constraint")
			if any(_semantic_group(value) != _semantic_group(type_value) for value in branch_types if value and type_value):
				failures.append("branch targets do not match requested final answer type")
			if branch_relation_mode == "SAME_PROPERTY" and not self._same_property(branch_properties, property_value):
				failures.append("CONSTRAINT_PROPERTY_MISMATCH: SAME_PROPERTY branches are not semantically equivalent")
			if branch_relation_mode == "COMPLEMENTARY_PROPERTIES" and not self._complementary_properties(branch_properties):
				failures.append("CONSTRAINT_PROPERTY_MISMATCH: complementary branches are not distinct compatible properties")

		unsupported = structure == "BRANCH_INTERMEDIATE"
		if structure == "BRANCH_FINAL" and failures:
			execution_mode = "cor"
		if structure not in STRUCTURES:
			execution_mode = "cor"
		validation_reason = ""
		if failures:
			validation_reason = "invalid_complementary_branch" if branch_relation_mode == "COMPLEMENTARY_PROPERTIES" else "invalid_branch_plan"
		return RouterDecision(
			structure=structure if structure in STRUCTURES else "UNCERTAIN",
			execution_mode=execution_mode,
			confidence=confidence,
			reason=reason,
			requested_final_answer_type=type_value,
			requested_final_semantic_property=property_value,
			shared_answer_type=shared_type,
			proposed_branches=branches if structure == "BRANCH_FINAL" and not failures else [],
			branch_property_compatibility=(not any("CONSTRAINT_PROPERTY_MISMATCH" in item for item in failures)) if structure == "BRANCH_FINAL" else None,
			unsupported_structure=unsupported,
			validation_passed=not failures,
			validation_failures=failures,
			branch_relation_mode=branch_relation_mode,
			same_final_answer_space=same_final_answer_space,
			validation_reason=validation_reason,
		)

	def _build_prompt(self, question: str, topic_entities: list[Entity]) -> str:
		entities = [{"id": str(entity.id), "name": str(entity.name)} for entity in topic_entities]
		hint = extract_requested_answer_target(question)
		return (
			"You are the Adaptive v1.1.4 reasoning-structure router for a relation-centric knowledge-graph QA system.\n"
			"Use only the question and linked runtime entities. Never use gold answers, gold SPARQL, S-expressions, or dataset labels.\n\n"
			"First identify the exact requested final answer type and semantic property. Preserve v1.1.3 answer-type safety.\n"
			"For BRANCH_FINAL, branches must independently constrain the SAME final answer variable/answer space.\n"
			"Use SAME_PROPERTY when branches use equivalent semantic properties. Use COMPLEMENTARY_PROPERTIES when properties differ but each is explicitly supported by the question and independently produces candidates of the same final answer type.\n"
			"Do not use complementary branching for an intermediate convergence, an unrelated relation, a generic/type-like root, or a different answer type. Such cases are BRANCH_INTERMEDIATE, LINEAR, or UNCERTAIN.\n\n"
			"Every BRANCH_FINAL response must include same_final_answer_space=true, branch_relation_mode, and for each branch question_supported=true and independently_generates_final_candidates=true.\n"
			"For BRANCH_FINAL, every branch must state constraint_property, constraint_summary, and target_answer_type. All branch target answer types must match the requested final answer type.\n\n"
			"Structures: LINEAR, BRANCH_FINAL, BRANCH_INTERMEDIATE, UNCERTAIN. BRANCH_INTERMEDIATE is logged but executed as CoR because branch-to-merge-to-continue is not implemented.\n"
			"Return JSON only with fields: structure, execution_mode, branch_relation_mode, same_final_answer_space, confidence, reason, requested_final_answer_type, requested_final_semantic_property, shared_answer_type, branches, unsupported_structure.\n"
			"For BRANCH_FINAL each branch must contain branch_id, root_entity_id, root_entity_name, constraint_property, constraint_summary, target_answer_type, question_supported, independently_generates_final_candidates. For other structures branches must be [].\n\n"
			f"Question: {question}\n"
			f"Lexical safety hint (verify semantically, do not blindly copy): {json.dumps(hint, ensure_ascii=False)}\n"
			f"Runtime topic entities: {json.dumps(entities, ensure_ascii=False)}\n"
		)

	def route(self, question: str, topic_entities: list[Entity], prompt_history: list[dict[str, Any]]) -> RouterDecision:
		prompt = self._build_prompt(question, topic_entities)
		response, usage = self.llm_generate(
			prompt,
			temperature=0.1,
			max_tokens=768,
			system_prompt="Return one JSON routing object and no markdown.",
		)
		parsed = self._extract_json(response)
		decision = self._validate(parsed, question, topic_entities)
		prompt_history.append({
			"type": "adaptive_router_v1_1_4",
			"prompt": prompt,
			"response": response or "",
			"input_tokens": int((usage or {}).get("input_tokens", 0) or 0),
			"output_tokens": int((usage or {}).get("output_tokens", 0) or 0),
			"total_tokens": int((usage or {}).get("total_tokens", 0) or 0),
			"parsed_result": parsed if parsed is not None else {},
			"requested_final_answer_type": decision.requested_final_answer_type,
			"requested_final_semantic_property": decision.requested_final_semantic_property,
			"branch_relation_mode": decision.branch_relation_mode,
			"same_final_answer_space": decision.same_final_answer_space,
			"warnings": list(decision.validation_failures or []),
			"prompt_version": ROUTER_PROMPT_VERSION,
		})
		return decision
