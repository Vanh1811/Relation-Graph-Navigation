"""Execution wrapper for the Phase 2.5.5 normalization intervention."""

from __future__ import annotations

from typing import Any

from chain_of_relations.methods.adaptive_v1_1_3.agent import AdaptiveRoutingAgent
from .router import CanonicalNormalizationRouter


class CanonicalNormalizationAgent(AdaptiveRoutingAgent):
	"""Use frozen extraction, normalize representations, then reuse frozen 3B execution."""

	def __init__(self, cor_agent: Any, confidence_threshold: float = 0.75, max_branches: int = 3) -> None:
		super().__init__(cor_agent, confidence_threshold=confidence_threshold, max_branches=max_branches)
		self.router = CanonicalNormalizationRouter(
			cor_agent._llm_generate,
			config=self.router.config,
		)
