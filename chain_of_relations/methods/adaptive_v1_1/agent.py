"""Adaptive routing wrapper around unchanged CoR and Branch-v1 execution."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from chain_of_relations.methods.branch_v1.agent import BranchV1Agent
from chain_of_relations.schema import Entity
from chain_of_relations.methods.adaptive_v1_1.router import (
	AdaptiveRouter,
	RouterConfig,
)

if TYPE_CHECKING:
	from chain_of_relations.methods.cor.agent import CoRAgent


class AdaptiveRoutingAgent:
	"""Route each question conservatively to CoR or explicit branch execution."""

	def __init__(
		self,
		cor_agent: CoRAgent,
		confidence_threshold: float = 0.75,
		max_branches: int = 3,
	) -> None:
		self.cor_agent = cor_agent
		self.branch_agent = BranchV1Agent(cor_agent)
		self.router = AdaptiveRouter(
			cor_agent._llm_generate,
			config=RouterConfig(
				max_branches=max_branches,
				confidence_threshold=confidence_threshold,
			),
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
	def _attach_route_metadata(
		result: dict[str, Any],
		decision: dict[str, Any],
		fallback_used: bool,
		fallback_reason: str | None,
		final_execution_mode: str,
		prompt_history: list[dict[str, Any]],
		sparql_history: list[dict[str, Any]],
		step_history: list[dict[str, Any]],
	) -> dict[str, Any]:
		result = dict(result)
		result.update(decision)
		result.update(
			{
				"fallback_used": fallback_used,
				"fallback_reason": fallback_reason,
				"final_execution_mode": final_execution_mode,
				"prompt_history": prompt_history,
				"sparql_history": sparql_history,
				"step_history": step_history,
			}
		)
		result["router_decision"] = dict(decision)
		return result

	def answer(self, question: str, topic_entities: list[Entity], **kwargs: Any) -> dict[str, Any]:
		prompt_history = kwargs.get("prompt_history", [])
		sparql_history = kwargs.get("sparql_history", [])
		step_history = kwargs.get("step_history", [])
		decision = self.router.route(question, topic_entities, prompt_history)
		decision_dict = decision.as_dict()

		if not decision.validation_passed:
			result = self.cor_agent.answer(
				question=question,
				topic_entities=topic_entities,
				prompt_history=prompt_history,
				sparql_history=sparql_history,
				step_history=step_history,
			)
			return self._attach_route_metadata(
				result,
				decision_dict,
				fallback_used=True,
				fallback_reason="invalid_branch_plan",
				final_execution_mode="cor_fallback",
				prompt_history=prompt_history,
				sparql_history=sparql_history,
				step_history=step_history,
			)

		if decision.router_mode == "cor":
			result = self.cor_agent.answer(
				question=question,
				topic_entities=topic_entities,
				prompt_history=prompt_history,
				sparql_history=sparql_history,
				step_history=step_history,
			)
			return self._attach_route_metadata(
				result,
				decision_dict,
				fallback_used=False,
				fallback_reason=None,
				final_execution_mode="cor",
				prompt_history=prompt_history,
				sparql_history=sparql_history,
				step_history=step_history,
			)

		try:
			result = self.branch_agent.answer(
				question=question,
				topic_entities=topic_entities,
				branch_plan=decision.proposed_branches,
				shared_answer_type=decision.shared_answer_type,
				prompt_history=prompt_history,
				sparql_history=sparql_history,
				step_history=step_history,
			)
			unusable, reason = self._branch_result_is_unusable(result)
		except Exception as exc:
			result = {"action": "ERROR", "results": [], "reasoning_chains": ""}
			unusable, reason = True, f"branch_exception:{type(exc).__name__}"

		if not unusable:
			return self._attach_route_metadata(
				result,
				decision_dict,
				fallback_used=False,
				fallback_reason=None,
				final_execution_mode="branch",
				prompt_history=prompt_history,
				sparql_history=sparql_history,
				step_history=step_history,
			)

		fallback_result = self.cor_agent.answer(
			question=question,
			topic_entities=topic_entities,
			prompt_history=prompt_history,
			sparql_history=sparql_history,
			step_history=step_history,
		)
		fallback_result.setdefault("adaptive_branch_result", result)
		return self._attach_route_metadata(
			fallback_result,
			decision_dict,
			fallback_used=True,
			fallback_reason=reason,
			final_execution_mode="cor_fallback",
			prompt_history=prompt_history,
			sparql_history=sparql_history,
			step_history=step_history,
		)
