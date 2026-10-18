"""Adaptive v1.1.4 execution using complementary-property branch validation."""

from __future__ import annotations

from typing import Any

from chain_of_relations.methods.adaptive_v1_1_3.agent import AdaptiveRoutingAgent as AdaptiveRoutingAgentV113
from chain_of_relations.methods.adaptive_v1_1_4.router import AdaptiveRouter, RouterConfig
from chain_of_relations.methods.branch_v1.agent import BranchV1Agent


class AdaptiveRoutingAgent(AdaptiveRoutingAgentV113):
	"""Use CoR unless the v1.1.4 branch plan is deterministically valid."""

	def __init__(self, cor_agent: Any, confidence_threshold: float = 0.75, max_branches: int = 3) -> None:
		self.cor_agent = cor_agent
		self.branch_agent = BranchV1Agent(cor_agent)
		self.router = AdaptiveRouter(
			cor_agent._llm_generate,
			config=RouterConfig(max_branches=max_branches, confidence_threshold=confidence_threshold),
		)

	def answer(self, question: str, topic_entities: list[Any], **kwargs: Any) -> dict[str, Any]:
		result = super().answer(question, topic_entities, **kwargs)
		if result.get("validation_reason") == "invalid_complementary_branch" and result.get("fallback_used"):
			result["fallback_reason"] = "invalid_complementary_branch"
		return result
