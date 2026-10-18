"""Adaptive v1.1.2 execution with native/fallback separation."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from chain_of_relations.methods.branch_v1.agent import BranchV1Agent
from chain_of_relations.methods.adaptive_v1_1_2.router import AdaptiveRouter, RouterConfig
from chain_of_relations.schema import Entity

if TYPE_CHECKING:
	from chain_of_relations.methods.cor.agent import CoRAgent


class AdaptiveRoutingAgent:
	"""Use CoR unless a validated BRANCH_FINAL plan is available."""

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
		outer_prompt_history: list[dict[str, Any]],
		outer_sparql_history: list[dict[str, Any]],
		outer_step_history: list[dict[str, Any]],
		native_result: dict[str, Any],
		native_prompt_history: list[dict[str, Any]],
		native_sparql_history: list[dict[str, Any]],
		native_step_history: list[dict[str, Any]],
		fallback_used: bool,
		fallback_reason: str | None,
		final_execution_mode: str,
	) -> dict[str, Any]:
		out = dict(result)
		out.update(decision)
		out.update(
			{
				"fallback_used": fallback_used,
				"fallback_reason": fallback_reason,
				"final_execution_mode": final_execution_mode,
				"native_execution_mode": decision.get("router_execution_mode", "cor"),
				"native_result_action": native_result.get("action", ""),
				"adaptive_native_result": dict(native_result),
				"adaptive_native_prompt_history": list(native_prompt_history),
				"adaptive_native_sparql_history": list(native_sparql_history),
				"adaptive_native_step_history": list(native_step_history),
				"prompt_history": outer_prompt_history,
				"sparql_history": outer_sparql_history,
				"step_history": outer_step_history,
			}
		)
		out["router_decision"] = dict(decision)
		return out

	def answer(self, question: str, topic_entities: list[Entity], **kwargs: Any) -> dict[str, Any]:
		outer_prompt = kwargs.get("prompt_history", [])
		outer_sparql = kwargs.get("sparql_history", [])
		outer_steps = kwargs.get("step_history", [])
		decision = self.router.route(question, topic_entities, outer_prompt)
		decision_dict = decision.as_dict()

		if not decision.validation_passed or decision.execution_mode != "branch":
			native_prompt: list[dict[str, Any]] = []
			native_sparql: list[dict[str, Any]] = []
			native_steps: list[dict[str, Any]] = []
			result = self.cor_agent.answer(
				question=question,
				topic_entities=topic_entities,
				prompt_history=native_prompt,
				sparql_history=native_sparql,
				step_history=native_steps,
			)
			outer_prompt.extend(native_prompt)
			outer_sparql.extend(native_sparql)
			outer_steps.extend(native_steps)
			return self._attach(
				result, decision_dict, outer_prompt, outer_sparql, outer_steps,
				result, native_prompt, native_sparql, native_steps,
				False, None, "cor",
			)

		native_prompt = []
		native_sparql = []
		native_steps = []
		try:
			native_result = self.branch_agent.answer(
				question=question,
				topic_entities=topic_entities,
				branch_plan=decision.proposed_branches,
				shared_answer_type=decision.shared_answer_type,
				prompt_history=native_prompt,
				sparql_history=native_sparql,
				step_history=native_steps,
			)
			unusable, reason = self._branch_result_is_unusable(native_result)
		except Exception as exc:
			native_result = {"action": "ERROR", "results": [], "reasoning_chains": ""}
			unusable, reason = True, f"branch_exception:{type(exc).__name__}"
		outer_prompt.extend(native_prompt)
		outer_sparql.extend(native_sparql)
		outer_steps.extend(native_steps)
		if not unusable:
			return self._attach(
				native_result, decision_dict, outer_prompt, outer_sparql, outer_steps,
				native_result, native_prompt, native_sparql, native_steps,
				False, None, "branch",
			)

		fallback_prompt: list[dict[str, Any]] = []
		fallback_sparql: list[dict[str, Any]] = []
		fallback_steps: list[dict[str, Any]] = []
		fallback_result = self.cor_agent.answer(
			question=question,
			topic_entities=topic_entities,
			prompt_history=fallback_prompt,
			sparql_history=fallback_sparql,
			step_history=fallback_steps,
		)
		outer_prompt.extend(fallback_prompt)
		outer_sparql.extend(fallback_sparql)
		outer_steps.extend(fallback_steps)
		fallback_result["adaptive_branch_result"] = native_result
		return self._attach(
			fallback_result, decision_dict, outer_prompt, outer_sparql, outer_steps,
			native_result, native_prompt, native_sparql, native_steps,
			True, reason, "cor_fallback",
		)
