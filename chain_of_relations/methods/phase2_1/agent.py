"""Phase 2.1 Adaptive wrapper with isolated relation grounding."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from chain_of_relations.methods.adaptive_v1_1_3.router import AdaptiveRouter, RouterConfig, extract_requested_answer_target
from chain_of_relations.methods.branch_v1.agent import BranchV1Agent
from chain_of_relations.methods.cor.agent import CoRAgent
from chain_of_relations.methods.phase2_1.relation_grounding import (
	SemanticTargetRelationGrounder,
	SemanticTargetState,
	relation_history,
	relation_records,
)
from chain_of_relations.schema import Entity, Relation

if TYPE_CHECKING:
	from typing import List


class SemanticTargetCoRAgent(CoRAgent):
	"""CoR with only its relation-prune decision replaced."""

	def __init__(self, *args: Any, **kwargs: Any) -> None:
		super().__init__(*args, **kwargs)
		self.semantic_relation_grounder = SemanticTargetRelationGrounder(
			self._llm_generate,
			temperature=self.temperature_exploration,
			max_tokens=self.max_token,
		)
		self._semantic_target_context: dict[str, Any] = {}

	@staticmethod
	def _branch_context(context: dict[str, Any], topic_entities: list[Entity]) -> dict[str, Any]:
		branches = context.get("branch_plan", [])
		root = topic_entities[0] if topic_entities else None
		if root is None or not isinstance(branches, list):
			return context
		for branch in branches:
			if not isinstance(branch, dict):
				continue
			if str(branch.get("root_entity_id", "")) == str(root.id) or str(branch.get("root_entity_name", "")).casefold() == str(root.name).casefold():
				return {
					**context,
					"requested_final_answer_type": branch.get("target_answer_type") or context.get("requested_final_answer_type", ""),
					"requested_final_semantic_property": branch.get("constraint_property") or context.get("requested_final_semantic_property", ""),
					"current_subgoal": branch.get("constraint_summary") or context.get("current_subgoal", ""),
					"remaining_question_constraint": branch.get("constraint_summary") or context.get("remaining_question_constraint", ""),
				}
		return context

	def answer(self, question: str, topic_entities: list[Entity], **kwargs: Any) -> dict[str, Any]:
		old_context = self._semantic_target_context
		context = kwargs.pop("semantic_target_context", None)
		if not isinstance(context, dict):
			context = {}
		context = self._branch_context(context, topic_entities)
		self._semantic_target_context = context
		try:
			return super().answer(question=question, topic_entities=topic_entities, **kwargs)
		finally:
			self._semantic_target_context = old_context

	def _state(self, question: str, topic_entity: Entity, relation_chain: list[Relation]) -> SemanticTargetState:
		hint = extract_requested_answer_target(question)
		context = self._semantic_target_context
		answer_type = str(context.get("requested_final_answer_type") or hint.get("requested_final_answer_type", "")).strip()
		semantic_property = str(context.get("requested_final_semantic_property") or hint.get("requested_final_semantic_property", "")).strip()
		history = relation_history(relation_chain)
		current_subgoal = str(context.get("current_subgoal", "")).strip()
		if not current_subgoal:
			current_subgoal = (
				f"Advance from {topic_entity.name} toward the requested {answer_type or 'answer'}"
				if not relation_chain else f"Continue the relation path toward {answer_type or semantic_property or 'the answer'}"
			)
		remaining = str(context.get("remaining_question_constraint", "")).strip() or question
		entity_type = str(context.get("current_entity_type", "")).strip()
		if not entity_type:
			entity_type = f"runtime entity named {topic_entity.name}"
		return SemanticTargetState(
			requested_final_answer_type=answer_type,
			requested_final_semantic_property=semantic_property,
			current_subgoal=current_subgoal,
			current_entity_type=entity_type,
			remaining_question_constraint=remaining,
			reasoning_step_index=len(relation_chain),
			relation_history=tuple(history),
		)

	def relation_prune(
		self,
		question: str,
		topic_entity: Entity,
		relation_chain: list[Relation],
		head_relations: list[str],
		tail_relations: list[str],
		prompt_history: list[dict[str, Any]],
	) -> list[Relation]:
		baseline_selected = super().relation_prune(
			question=question,
			topic_entity=topic_entity,
			relation_chain=relation_chain,
			head_relations=head_relations,
			tail_relations=tail_relations,
			prompt_history=prompt_history,
		)
		capped_head, capped_tail = self._cap_relations_by_threshold(
			head_relations=head_relations,
			tail_relations=tail_relations,
			threshold=self.sample_relation_threshold,
		)
		display_names = self.kg_backend.relation_ids2labels(capped_head + capped_tail)
		state = self._state(question, topic_entity, relation_chain)
		decision = self.semantic_relation_grounder.select(
			question=question,
			topic_entity=topic_entity,
			relation_chain=relation_chain,
			head_relations=capped_head,
			tail_relations=capped_tail,
			relation_display_names=display_names,
			state=state,
			top_k=self.relation_width,
		)
		phase2_selected = decision.selected_relations if decision.success else list(baseline_selected)
		prompt_history.append({
			"type": "phase2_1_semantic_relation_grounding",
			"prompt": decision.prompt,
			"response": decision.response,
			"input_tokens": int(decision.usage.get("input_tokens", 0) or 0),
			"output_tokens": int(decision.usage.get("output_tokens", 0) or 0),
			"total_tokens": int(decision.usage.get("total_tokens", 0) or 0),
			"candidate_relations": [item.as_dict() for item in decision_candidates(capped_head, capped_tail, display_names)],
			"candidate_directions": {
				"outgoing": list(capped_head),
				"incoming": list(capped_tail),
			},
			"baseline_selected_relation": relation_records(baseline_selected),
			"phase2_selected_relation": relation_records(phase2_selected),
			"selected_direction": decision.selected_direction,
			"requested_final_answer_type": state.requested_final_answer_type,
			"requested_final_semantic_property": state.requested_final_semantic_property,
			"current_subgoal": state.current_subgoal,
			"current_entity_type": state.current_entity_type,
			"remaining_question_constraint": state.remaining_question_constraint,
			"reasoning_step_index": state.reasoning_step_index,
			"relation_history": list(state.relation_history),
			"confidence": decision.confidence,
			"selection_reason": decision.selection_reason,
			"warnings": list(decision.warnings),
			"phase2_success": decision.success,
			"fallback_to_baseline": not decision.success,
		})
		return phase2_selected


def decision_candidates(head: list[str], tail: list[str], display: dict[str, str]) -> list[Any]:
	from chain_of_relations.methods.phase2_1.relation_grounding import RelationCandidate
	return SemanticTargetRelationGrounder.candidates(head, tail, display)


class SemanticTargetBranchV1Agent(BranchV1Agent):
	"""Pass validated branch semantic metadata to the Phase 2 CoR agent."""

	def answer(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
		branch_plan = kwargs.get("branch_plan")
		requested_type = kwargs.get("requested_final_answer_type", "")
		requested_property = kwargs.get("requested_final_semantic_property", "")
		old_context = getattr(self.cor_agent, "_semantic_target_context", {})
		self.cor_agent._semantic_target_context = {
			"requested_final_answer_type": requested_type,
			"requested_final_semantic_property": requested_property,
			"branch_plan": branch_plan if isinstance(branch_plan, list) else [],
		}
		try:
			return super().answer(*args, **kwargs)
		finally:
			self.cor_agent._semantic_target_context = old_context


class SemanticTargetAdaptiveAgent:
	"""Frozen v1.1.3 routing with Phase 2.1 relation grounding only."""

	def __init__(self, cor_agent: SemanticTargetCoRAgent, confidence_threshold: float = 0.75, max_branches: int = 3) -> None:
		self.cor_agent = cor_agent
		self.branch_agent = SemanticTargetBranchV1Agent(cor_agent)
		self.router = AdaptiveRouter(
			cor_agent._llm_generate,
			config=RouterConfig(max_branches=max_branches, confidence_threshold=confidence_threshold),
		)

	@staticmethod
	def _branch_result_is_unusable(result: dict[str, Any]) -> tuple[bool, str | None]:
		if str(result.get("action", "")).upper() == "ERROR":
			return True, "branch_execution_error"
		if result.get("branch_failures"):
			return True, "branch_execution_failure"
		counts = result.get("branch_candidate_counts", [])
		if isinstance(counts, list) and counts and any(int(value or 0) <= 0 for value in counts):
			return True, "branch_empty"
		if result.get("merged_candidate_count", 0) == 0 and counts:
			return True, "empty_strict_merge"
		return False, None

	@staticmethod
	def _attach(result: dict[str, Any], decision: dict[str, Any], fallback_used: bool, fallback_reason: str | None, final_execution_mode: str, native_result: dict[str, Any] | None = None) -> dict[str, Any]:
		out = dict(result)
		native = native_result if native_result is not None else result
		out.update(decision)
		out.update({
			"fallback_used": fallback_used,
			"fallback_reason": fallback_reason,
			"final_execution_mode": final_execution_mode,
			"native_execution_mode": decision.get("router_execution_mode", "cor"),
			"native_result_action": native.get("action", ""),
			"adaptive_native_result": dict(native),
		})
		out["router_decision"] = dict(decision)
		return out

	def _cor_answer(self, question: str, topic_entities: list[Entity], decision: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
		return self.cor_agent.answer(
			question=question,
			topic_entities=topic_entities,
			semantic_target_context={
				"requested_final_answer_type": decision.get("requested_final_answer_type", ""),
				"requested_final_semantic_property": decision.get("requested_final_semantic_property", ""),
			},
			**kwargs,
		)

	def answer(self, question: str, topic_entities: list[Entity], **kwargs: Any) -> dict[str, Any]:
		prompt_history = kwargs.get("prompt_history", [])
		decision = self.router.route(question, topic_entities, prompt_history)
		decision_dict = decision.as_dict()
		if not decision.validation_passed or decision.execution_mode != "branch":
			result = self._cor_answer(question, topic_entities, decision_dict, **kwargs)
			return self._attach(result, decision_dict, not decision.validation_passed, "invalid_branch_plan" if not decision.validation_passed else None, "cor_fallback" if not decision.validation_passed else "cor")
		try:
			result = self.branch_agent.answer(
			question=question,
			topic_entities=topic_entities,
			branch_plan=decision.proposed_branches,
			shared_answer_type=decision.shared_answer_type,
			requested_final_answer_type=decision.requested_final_answer_type,
			requested_final_semantic_property=decision.requested_final_semantic_property,
			prompt_history=prompt_history,
			sparql_history=kwargs.get("sparql_history", []),
			step_history=kwargs.get("step_history", []),
			semantic_target_context=decision_dict,
			)
			unusable, reason = self._branch_result_is_unusable(result)
		except Exception as exc:
			result = {"action": "ERROR", "results": [], "reasoning_chains": ""}
			unusable, reason = True, f"branch_exception:{type(exc).__name__}"
		if not unusable:
			return self._attach(result, decision_dict, False, None, "branch")
		fallback_result = self._cor_answer(question, topic_entities, decision_dict, **kwargs)
		fallback_result["adaptive_branch_result"] = result
		return self._attach(fallback_result, decision_dict, True, reason, "cor_fallback", native_result=result)
