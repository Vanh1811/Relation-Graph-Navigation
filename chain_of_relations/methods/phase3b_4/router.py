"""Phase 3B.4 explicit answer-vs-convergence role separation.

The model supplies two semantic roles and a contrastive comparison.  The
shared-target enum is then derived deterministically from those fields before
the unchanged Phase 2.5.5/2.5.3B validator and execution stack runs.
"""

from __future__ import annotations

import json
from typing import Any

from chain_of_relations.methods.adaptive_v1_1_3.router import extract_requested_answer_target
from chain_of_relations.methods.phase2_5_3a.router import _text
from chain_of_relations.methods.phase2_5_5.router import CanonicalNormalizationRouter
from chain_of_relations.schema import Entity


ROUTER_PROMPT_VERSION = "phase3b.4-answer-convergence-role-separation-json-v1"
TARGET_TYPES = {"ANSWER", "INTERMEDIATE", "NONE"}


def _bool_or_none(value: Any) -> bool | None:
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		clean = value.strip().casefold()
		if clean in {"true", "yes", "1"}:
			return True
		if clean in {"false", "no", "0"}:
			return False
	return None


def _role(value: Any) -> dict[str, str]:
	if isinstance(value, dict):
		return {
			"description": _text(value.get("description") or value.get("role")),
			"entity_type": _text(value.get("entity_type") or value.get("type")),
		}
	if isinstance(value, str):
		return {"description": _text(value), "entity_type": ""}
	return {"description": "", "entity_type": ""}


def derive_shared_target_type(roles_same_entity: bool | None, requires_downstream_relation: bool | None) -> str:
	"""Derive the enum without reading gold labels or question text."""
	if roles_same_entity is True:
		return "ANSWER" if requires_downstream_relation is not True else "NONE"
	if roles_same_entity is False and requires_downstream_relation is True:
		return "INTERMEDIATE"
	return "NONE"


def role_consistency(
	requested_answer_role: dict[str, str],
	convergence_role: dict[str, str],
	roles_same_entity: bool | None,
	requires_downstream_relation: bool | None,
) -> tuple[str, list[str]]:
	issues: list[str] = []
	if not requested_answer_role.get("description") or not requested_answer_role.get("entity_type"):
		issues.append("requested_answer_role_incomplete")
	if not convergence_role.get("description") or not convergence_role.get("entity_type"):
		issues.append("convergence_role_incomplete")
	if roles_same_entity is None:
		issues.append("roles_same_entity_missing_or_uncertain")
	if requires_downstream_relation is None:
		issues.append("requires_downstream_relation_missing_or_uncertain")
	if roles_same_entity is True and requires_downstream_relation is True:
		issues.append("same_roles_cannot_require_downstream_relation")
	return ("CONSISTENT" if not issues else "INCONSISTENT", issues)


def separate_roles(proposal: Any) -> tuple[dict[str, Any], dict[str, Any]]:
	"""Normalize role fields and deterministically derive the target enum."""
	enriched = dict(proposal) if isinstance(proposal, dict) else {}
	requested = _role(enriched.get("requested_answer_role"))
	convergence = _role(enriched.get("convergence_role"))
	same = _bool_or_none(enriched.get("roles_same_entity"))
	requires = _bool_or_none(enriched.get("requires_downstream_relation"))
	status, issues = role_consistency(requested, convergence, same, requires)
	derived = derive_shared_target_type(same, requires)
	original_target = enriched.get("shared_target")
	original_description = _text(enriched.get("shared_target_description"))
	if isinstance(original_target, dict) and not original_description:
		original_description = _text(original_target.get("description"))
	if derived == "INTERMEDIATE" and not original_description:
		original_description = convergence["description"]
	if derived == "ANSWER" and not original_description:
		original_description = requested["description"]
	if derived == "NONE":
		original_description = ""
	enriched["requested_answer_role"] = requested
	enriched["convergence_role"] = convergence
	enriched["roles_same_entity"] = same
	enriched["role_comparison"] = "YES" if same is True else "NO" if same is False else "UNCERTAIN"
	enriched["role_comparison_reason"] = _text(enriched.get("role_comparison_reason"))
	enriched["requires_downstream_relation"] = requires
	enriched["shared_target_type"] = derived
	enriched["shared_target"] = {"type": derived, "description": original_description}
	enriched["shared_target_description"] = original_description
	enriched["role_separation_consistency"] = status
	enriched["role_separation_issues"] = issues
	metadata = {
		"requested_answer_role": requested,
		"convergence_role": convergence,
		"roles_same_entity": same,
		"role_comparison": enriched["role_comparison"],
		"role_comparison_reason": enriched["role_comparison_reason"],
		"requires_downstream_relation": requires,
		"derived_shared_target_type": derived,
		"role_separation_consistency": status,
		"role_separation_issues": issues,
		"input_shared_target": original_target,
		"input_shared_target_type": (
			_text(original_target.get("type") or original_target.get("target")).upper()
			if isinstance(original_target, dict)
			else _text(original_target).upper()
		),
	}
	return enriched, metadata


class RoleSeparationRouter(CanonicalNormalizationRouter):
	"""Ask for distinct roles, then derive the executable semantic target."""

	def _build_prompt(self, question: str, topic_entities: list[Entity]) -> str:
		entities = [{"id": str(entity.id), "name": str(entity.name)} for entity in topic_entities]
		hint = extract_requested_answer_target(question)
		return (
			"You are the Phase 3B.4 answer-vs-convergence role-separation router for a relation-centric knowledge-graph QA system.\n"
			"Use only the question and linked runtime entities. Never use gold answers, gold SPARQL, S-expressions, dataset labels, or benchmark examples.\n\n"
			"First extract every essential answer-relevant constraint and group constraints by independent source anchor.\n"
			"Then perform these semantic steps in order:\n"
			"A. Identify the requested_answer_role: the semantic entity or property the question asks to return.\n"
			"B. Identify the convergence_role: the entity on which the independent constraints jointly converge.\n"
			"C. Decide whether those two roles refer to the same semantic entity.\n"
			"D. If they differ, decide whether a downstream semantic relation is required to reach the requested answer.\n"
			"E. Report the fields before the system derives shared_target.type.\n\n"
			"requested_answer_role and convergence_role must each contain description and entity_type. roles_same_entity must be true, false, or null. requires_downstream_relation must be a JSON boolean or null.\n"
			"The final contrastive self-check is: The constraints jointly identify X. The question asks for Y. Are X and Y the same semantic role? Answer YES, NO, or UNCERTAIN and give a short role_comparison_reason.\n"
			"Do not assign shared_target.type independently of these fields. The runtime deterministically derives it: same=true with no downstream relation -> ANSWER; same=false with downstream relation=true -> INTERMEDIATE; otherwise -> NONE.\n"
			"Do not label INTERMEDIATE merely because the question is multi-hop, contains multiple entities, contains a CVT/reified node, or has an indirect answer relation.\n"
			"Keep existing structure labels and branch fields. Do not redesign normalization, serialization, validation, execution, fallback, relation grounding, or answer selection.\n\n"
			"Return JSON only with fields: constraints, independent_source_groups, requested_answer_role, convergence_role, roles_same_entity, role_comparison, role_comparison_reason, requires_downstream_relation, downstream_relation_description, shared_target, shared_target_description, requested_answer_description, shared_target_self_check, dependency_pattern, structure, execution_mode, confidence, reason, requested_final_answer_type, requested_final_semantic_property, shared_answer_type, branches, unsupported_structure, extraction_self_check.\n"
			"Each constraint must contain constraint_id, source_anchor, constraint_text, target_variable_role, answer_relevant, semantic_property, depends_on. target_variable_role MUST be exactly ANSWER, INTERMEDIATE, FILTER, or TYPE.\n"
			"shared_target_self_check must preserve the role comparison and state whether the resulting variable is already the requested answer.\n"
			"For non-branch structures branches must be [].\n\n"
			f"Question: {question}\n"
			f"Lexical safety hint (supporting evidence only): {json.dumps(hint, ensure_ascii=False)}\n"
			f"Runtime topic entities: {json.dumps(entities, ensure_ascii=False)}\n"
		)

	def _validate(self, proposal: Any, question: str, topic_entities: list[Entity]):
		enriched, metadata = separate_roles(proposal)
		decision = super()._validate(enriched, question, topic_entities)
		setattr(decision, "role_separation", metadata)
		setattr(decision, "role_enriched_proposal", enriched)
		return decision

	def route(self, question: str, topic_entities: list[Entity], prompt_history: list[dict[str, Any]]) -> Any:
		decision = super().route(question, topic_entities, prompt_history)
		if prompt_history:
			event = prompt_history[-1]
			if event.get("type") == "phase2_5_5_canonical_normalization_router":
				metadata = getattr(decision, "role_separation", {})
				event["type"] = "phase3b_4_answer_convergence_role_separation_router"
				event["prompt_version"] = ROUTER_PROMPT_VERSION
				event["role_separation"] = metadata
				event["requested_answer_role"] = metadata.get("requested_answer_role", {})
				event["convergence_role"] = metadata.get("convergence_role", {})
				event["roles_same_entity"] = metadata.get("roles_same_entity")
				event["role_comparison"] = metadata.get("role_comparison")
				event["role_comparison_reason"] = metadata.get("role_comparison_reason", "")
				event["requires_downstream_relation"] = metadata.get("requires_downstream_relation")
				event["derived_shared_target_type"] = metadata.get("derived_shared_target_type", "NONE")
				event["role_separation_consistency"] = metadata.get("role_separation_consistency")
				event["role_separation_issues"] = metadata.get("role_separation_issues", [])
				parsed = event.get("parsed_result") if isinstance(event.get("parsed_result"), dict) else {}
				event["role_enriched_parsed_result"] = getattr(decision, "role_enriched_proposal", parsed)
				self_check = parsed.get("shared_target_self_check") if isinstance(parsed, dict) else {}
				if not isinstance(self_check, dict):
					self_check = {}
				event["shared_target_extraction"] = {
					"shared_target_type": metadata.get("derived_shared_target_type", "NONE"),
					"shared_target_description": _text(parsed.get("shared_target_description")),
					"requested_answer_description": _text(parsed.get("requested_answer_description")),
					"requires_downstream_relation": metadata.get("requires_downstream_relation"),
					"downstream_relation_description": _text(parsed.get("downstream_relation_description")),
					"requested_answer_role": metadata.get("requested_answer_role", {}),
					"convergence_role": metadata.get("convergence_role", {}),
					"roles_same_entity": metadata.get("roles_same_entity"),
					"role_comparison": metadata.get("role_comparison"),
					"role_comparison_reason": metadata.get("role_comparison_reason", ""),
					"self_check": self_check,
				}
		return decision
