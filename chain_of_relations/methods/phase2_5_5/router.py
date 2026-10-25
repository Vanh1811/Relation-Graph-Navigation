"""Phase 2.5.5 normalization before the frozen Phase 2.5.3B router."""

from __future__ import annotations

import time
from typing import Any

from chain_of_relations.methods.phase2_5_3b.router import BranchPlanSerializationRouter
from .normalizer import normalize_proposal


ROUTER_PROMPT_VERSION = "phase2.5.5-canonical-semantic-normalization-json-v1"


class CanonicalNormalizationRouter(BranchPlanSerializationRouter):
	"""Normalize representations only, then reuse the unchanged 3B validator path."""

	def _validate(self, proposal: Any, question: str, topic_entities: list[Any]):
		started = time.perf_counter()
		normalization = normalize_proposal(proposal, question)
		normalization["normalization_cpu_seconds"] = time.perf_counter() - started
		decision = super()._validate(normalization["canonical_proposal"], question, topic_entities)
		setattr(decision, "normalization", normalization)
		return decision

	def route(self, question: str, topic_entities: list[Any], prompt_history: list[dict[str, Any]]) -> Any:
		decision = super().route(question, topic_entities, prompt_history)
		if prompt_history:
			event = prompt_history[-1]
			if event.get("type") == "phase2_5_3b_branch_plan_serialization_router":
				event["type"] = "phase2_5_5_canonical_normalization_router"
				normalization = getattr(decision, "normalization", normalize_proposal(event.get("parsed_result", {}), question))
				event["normalization"] = normalization
				event["normalization_status"] = normalization.get("normalization_status")
				event["normalization_errors"] = list(normalization.get("normalization_errors", []))
				event["canonical_semantic_representation"] = normalization.get("canonical_proposal")
				event["prompt_version"] = ROUTER_PROMPT_VERSION
		return decision
