"""Execution wrapper for the Phase 2.5.3A extraction-only intervention."""

from __future__ import annotations

from typing import Any

from chain_of_relations.methods.adaptive_v1_1_3.agent import AdaptiveRoutingAgent
from chain_of_relations.methods.phase2_5_3a.router import SourceTargetExtractionRouter


class SourceTargetExtractionAgent(AdaptiveRoutingAgent):
	"""Reuse frozen execution, mapping, and validation with a new extractor."""

	def __init__(self, cor_agent: Any, confidence_threshold: float = 0.75, max_branches: int = 3) -> None:
		super().__init__(cor_agent, confidence_threshold=confidence_threshold, max_branches=max_branches)
		self.router = SourceTargetExtractionRouter(
			cor_agent._llm_generate,
			config=self.router.config,
		)
