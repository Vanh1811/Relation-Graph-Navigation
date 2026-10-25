"""Execution wrapper for the Phase 3B.4 role-separation intervention."""

from __future__ import annotations

from typing import Any

from chain_of_relations.methods.phase2_5_5.agent import CanonicalNormalizationAgent

from .router import RoleSeparationRouter


class RoleSeparationAgent(CanonicalNormalizationAgent):
	"""Change only semantic role extraction; reuse the frozen downstream stack."""

	def __init__(self, cor_agent: Any, confidence_threshold: float = 0.75, max_branches: int = 3) -> None:
		super().__init__(cor_agent, confidence_threshold=confidence_threshold, max_branches=max_branches)
		self.router = RoleSeparationRouter(
			cor_agent._llm_generate,
			config=self.router.config,
		)
