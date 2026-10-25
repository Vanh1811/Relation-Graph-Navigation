"""Adaptive execution wrapper using the Phase 2.5 constraint-first router."""

from __future__ import annotations

from typing import Any

from chain_of_relations.methods.adaptive_v1_1_3.agent import AdaptiveRoutingAgent
from chain_of_relations.methods.phase2_5.router import ConstraintFirstRouter


class ConstraintFirstRoutingAgent(AdaptiveRoutingAgent):
	"""Reuse v1.1.3 execution, replacing only router input preparation."""

	def __init__(self, cor_agent: Any, confidence_threshold: float = 0.75, max_branches: int = 3) -> None:
		super().__init__(cor_agent, confidence_threshold=confidence_threshold, max_branches=max_branches)
		self.router = ConstraintFirstRouter(
			cor_agent._llm_generate,
			config=self.router.config,
		)
