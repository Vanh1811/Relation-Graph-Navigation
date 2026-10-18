"""Adaptive v1.1.3 execution with answer-type-aware branch metadata."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from chain_of_relations.methods.branch_v1.agent import BranchV1Agent
from chain_of_relations.methods.adaptive_v1_1_3.router import AdaptiveRouter, RouterConfig
from chain_of_relations.schema import Entity

if TYPE_CHECKING:
	from chain_of_relations.methods.cor.agent import CoRAgent


class AdaptiveRoutingAgent:
	"""Use CoR unless a validated answer-type-compatible branch plan exists."""

	def __init__(self, cor_agent: CoRAgent, confidence_threshold: float = 0.75, max_branches: int = 3) -> None:
		self.cor_agent = cor_agent
		self.branch_agent = BranchV1Agent(cor_agent)
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
	def _attach(
		result: dict[str, Any],
		decision: dict[str, Any],
		fallback_used: bool,
		fallback_reason: str | None,
		final_execution_mode: str,
		native_result: dict[str, Any] | None = None,
	) -> dict[str, Any]:
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

	def answer(self, question: str, topic_entities: list[Entity], **kwargs: Any) -> dict[str, Any]:
		prompt_history = kwargs.get("prompt_history", [])
		sparql_history = kwargs.get("sparql_history", [])
		step_history = kwargs.get("step_history", [])
		decision = self.router.route(question, topic_entities, prompt_history)
		decision_dict = decision.as_dict()

		if not decision.validation_passed or decision.execution_mode != "branch":
			result = self.cor_agent.answer(
				question=question,
				topic_entities=topic_entities,
				prompt_history=prompt_history,
				sparql_history=sparql_history,
				step_history=step_history,
			)
			return self._attach(
				result,
				decision_dict,
				fallback_used=not decision.validation_passed,
				fallback_reason="invalid_branch_plan" if not decision.validation_passed else None,
				final_execution_mode="cor_fallback" if not decision.validation_passed else "cor",
			)

		try:
			result = self.branch_agent.answer(
				question=question,
				topic_entities=topic_entities,
				branch_plan=decision.proposed_branches,
				shared_answer_type=decision.shared_answer_type,
				requested_final_answer_type=decision.requested_final_answer_type,
				requested_final_semantic_property=decision.requested_final_semantic_property,
				prompt_history=prompt_history,
				sparql_history=sparql_history,
				step_history=step_history,
			)
			unusable, reason = self._branch_result_is_unusable(result)
		except Exception as exc:
			result = {"action": "ERROR", "results": [], "reasoning_chains": ""}
			unusable, reason = True, f"branch_exception:{type(exc).__name__}"

		if not unusable:
			return self._attach(result, decision_dict, False, None, "branch")

		fallback_result = self.cor_agent.answer(
			question=question,
			topic_entities=topic_entities,
			prompt_history=prompt_history,
			sparql_history=sparql_history,
			step_history=step_history,
		)
		fallback_result["adaptive_branch_result"] = result
		return self._attach(fallback_result, decision_dict, True, reason, "cor_fallback", native_result=result)
