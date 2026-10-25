"""Phase 3B.2 shared-target extraction intervention.

Only the semantic extraction instructions are changed here.  The inherited
Phase 2.5.5 normalization, Phase 2.5.3B serialization, validator, execution,
and fallback paths remain unchanged.
"""

from __future__ import annotations

import json
from typing import Any

from chain_of_relations.methods.phase2_5_5.router import CanonicalNormalizationRouter
from chain_of_relations.methods.phase2_5_3a.router import _text
from chain_of_relations.methods.adaptive_v1_1_3.router import extract_requested_answer_target
from chain_of_relations.schema import Entity


ROUTER_PROMPT_VERSION = "phase3b.2-shared-target-semantic-extraction-json-v2"
TARGET_TYPES = {"ANSWER", "INTERMEDIATE", "NONE"}


def _target_type(value: Any) -> str:
	if isinstance(value, dict):
		return _text(value.get("type") or value.get("target")).upper() or "NONE"
	return _text(value).upper() or "NONE"


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


class SharedTargetInterventionRouter(CanonicalNormalizationRouter):
	"""Use an explicit convergence-point self-check before the frozen stack."""

	def _build_prompt(self, question: str, topic_entities: list[Entity]) -> str:
		entities = [{"id": str(entity.id), "name": str(entity.name)} for entity in topic_entities]
		hint = extract_requested_answer_target(question)
		return (
			"You are the Phase 3B.2 shared-target semantic extraction router for a relation-centric knowledge-graph QA system.\n"
			"Use only the question and linked runtime entities. Never use gold answers, gold SPARQL, S-expressions, dataset labels, or benchmark-specific examples.\n\n"
			"First extract every essential answer-relevant constraint and group constraints by independent source anchor.\n"
			"Then identify the point where the independent constraints converge before the final answer is produced.\n"
			"Do not infer ANSWER merely because the question requests a final entity or value.\n\n"
			"Set shared_target.type to ANSWER only when the independent constraints directly constrain the requested final answer variable/value.\n"
			"Set shared_target.type to INTERMEDIATE only when all of these are true: (1) at least two answer-relevant constraints exist, (2) they converge on the same latent entity/variable, (3) that latent target is not the requested final answer, and (4) at least one downstream semantic relation/step is still required to reach the requested answer.\n"
			"Set shared_target.type to NONE for a linear question, an ambiguous case, or when there is no valid shared convergence target. When uncertain, prefer NONE over forcing ANSWER or INTERMEDIATE.\n"
			"Do not label INTERMEDIATE merely because the query is multi-hop, contains multiple entities, contains a CVT/reified node, or has an indirect answer relation. Multiple independent constraints must converge before the answer.\n\n"
			"Perform this self-check explicitly: If the independent constraints are satisfied, is the resulting variable already the requested final answer? YES means ANSWER. NO means INTERMEDIATE only when a genuine shared latent target and downstream step are present.\n"
			"Return shared_target_description, requested_answer_description, requires_downstream_relation, downstream_relation_description, and shared_target_self_check with the semantic extraction. These fields describe semantics only; do not guess KG relation IDs.\n"
			"Keep the existing structure labels exactly: LINEAR, BRANCH_FINAL, BRANCH_INTERMEDIATE, or UNCERTAIN. For BRANCH_FINAL, preserve the existing branch fields. Do not redesign normalization, serialization, validation, execution, fallback, relation grounding, or answer selection.\n\n"
			"Return JSON only with fields: constraints, independent_source_groups, shared_target, shared_target_description, requested_answer_description, requires_downstream_relation, downstream_relation_description, shared_target_self_check, dependency_pattern, structure, execution_mode, confidence, reason, requested_final_answer_type, requested_final_semantic_property, shared_answer_type, branches, unsupported_structure, extraction_self_check.\n"
			"Each constraint must contain constraint_id, source_anchor, constraint_text, target_variable_role, answer_relevant, semantic_property, depends_on. target_variable_role MUST be exactly one of ANSWER, INTERMEDIATE, FILTER, or TYPE; answer_relevant MUST be a JSON boolean; semantic_property MUST be a concise non-empty string. Do not use variants such as FINAL_ANSWER, FINAL, ANSWER_ENTITY, or INTERMEDIATE_ENTITY.\n"
			"Each independent_source_group must contain source_anchor and constraint_ids.\n"
			"shared_target_self_check must contain convergence_variable_description, resulting_variable_is_requested_answer, valid_shared_latent_target, downstream_step_required, and rationale.\n"
			"For non-branch structures branches must be [].\n\n"
			f"Question: {question}\n"
			f"Lexical safety hint (supporting evidence only): {json.dumps(hint, ensure_ascii=False)}\n"
			f"Runtime topic entities: {json.dumps(entities, ensure_ascii=False)}\n"
		)

	def route(self, question: str, topic_entities: list[Entity], prompt_history: list[dict[str, Any]]) -> Any:
		decision = super().route(question, topic_entities, prompt_history)
		if prompt_history:
			event = prompt_history[-1]
			if event.get("type") == "phase2_5_5_canonical_normalization_router":
				parsed = event.get("parsed_result") if isinstance(event.get("parsed_result"), dict) else {}
				raw_target = parsed.get("shared_target")
				self_check = parsed.get("shared_target_self_check")
				if not isinstance(self_check, dict):
					self_check = {}
				extraction = {
					"shared_target_type": _target_type(raw_target),
					"shared_target_description": _text(parsed.get("shared_target_description")) or _text(raw_target.get("description") if isinstance(raw_target, dict) else ""),
					"requested_answer_description": _text(parsed.get("requested_answer_description")),
					"requires_downstream_relation": _bool_or_none(parsed.get("requires_downstream_relation")),
					"downstream_relation_description": _text(parsed.get("downstream_relation_description")),
					"self_check": self_check,
				}
				event["type"] = "phase3b_2_shared_target_intervention_router"
				event["prompt_version"] = ROUTER_PROMPT_VERSION
				event["shared_target_extraction"] = extraction
				event["shared_target_self_check"] = self_check
				event["intervention_scope"] = "shared_target_semantic_extraction_only"
		return decision
