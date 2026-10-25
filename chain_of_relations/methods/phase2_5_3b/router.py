"""Phase 2.5.3B branch-plan serialization before the unchanged validator."""

from __future__ import annotations

import json
import time
from typing import Any

from chain_of_relations.methods.phase2_5_3a.router import SourceTargetExtractionRouter, _text
from .serializer import serialize_branch_plan


ROUTER_PROMPT_VERSION = "phase2.5.3b-branch-plan-serialization-json-v1"


class BranchPlanSerializationRouter(SourceTargetExtractionRouter):
	"""Reuse the frozen 3A extraction and adapt only its executable branch plan."""

	@classmethod
	def _serialization_for(cls, proposal: Any, topic_entities: list[Any], question: str) -> dict[str, Any]:
		if not isinstance(proposal, dict):
			return serialize_branch_plan(proposal, [], [], topic_entities, question)
		constraints = [
			cls._normalise_constraint(item, index)
			for index, item in enumerate(proposal.get("constraints") or [], start=1)
			if isinstance(item, dict)
		]
		groups = cls._normalise_source_groups(proposal, constraints)
		return serialize_branch_plan(proposal, constraints, groups, topic_entities, question)

	def _validate(self, proposal: Any, question: str, topic_entities: list[Any]):
		started = time.perf_counter()
		serialization = self._serialization_for(proposal, topic_entities, question)
		serialization["serialization_cpu_seconds"] = time.perf_counter() - started
		enriched = dict(proposal) if isinstance(proposal, dict) else proposal
		if isinstance(enriched, dict):
			enriched["branches"] = list(serialization.get("serialized_branches", []))
			enriched["branch_count"] = len(serialization.get("serialized_branches", []))
			# This is a shape normalization of the existing semantic value. The
			# original raw value remains in parsed_result for auditability.
			enriched["shared_target"] = dict(serialization.get("normalized_shared_target", {}))
		decision = super()._validate(enriched, question, topic_entities)
		setattr(decision, "serialization", serialization)
		return decision

	def route(self, question: str, topic_entities: list[Any], prompt_history: list[dict[str, Any]]) -> Any:
		# This call/retry path is intentionally identical to Phase 2.5.3A. Only
		# the parsed object passed into the validator receives serialized branches.
		prompt = self._build_prompt(question, topic_entities)
		response, usage = self.llm_generate(
			prompt,
			temperature=0.1,
			max_tokens=1024,
			system_prompt="Return one Phase 2.5.3A JSON extraction and routing object and no markdown.",
		)
		parsed = self._extract_json(response)
		retry_used = False
		retry_reason = ""
		if not isinstance(parsed, dict):
			retry_used = True
			retry_reason = "empty_model_response" if not response else "invalid_extraction_json"
			retry_prompt = (
				prompt
				+ "\n\nExtraction self-check failed because the previous response was empty or invalid. "
				"Retry exactly once. Return the complete JSON object, including constraints, "
				"independent_source_groups, shared_target, dependency_pattern, and extraction_self_check."
			)
			retry_response, retry_usage = self.llm_generate(
				retry_prompt,
				temperature=0.1,
				max_tokens=1024,
				system_prompt="Return one Phase 2.5.3A JSON extraction and routing object and no markdown.",
			)
			response = retry_response
			parsed = self._extract_json(response)
			usage = {
				"input_tokens": int((usage or {}).get("input_tokens", 0) or 0) + int((retry_usage or {}).get("input_tokens", 0) or 0),
				"output_tokens": int((usage or {}).get("output_tokens", 0) or 0) + int((retry_usage or {}).get("output_tokens", 0) or 0),
				"total_tokens": int((usage or {}).get("total_tokens", 0) or 0) + int((retry_usage or {}).get("total_tokens", 0) or 0),
			}
			if not isinstance(parsed, dict) and retry_usage and retry_usage.get("error"):
				usage["error"] = retry_usage["error"]

		decision = self._validate(parsed, question, topic_entities)
		constraints = []
		groups = []
		if isinstance(parsed, dict):
			constraints = [
				self._normalise_constraint(item, index)
				for index, item in enumerate(parsed.get("constraints") or [], start=1)
				if isinstance(item, dict)
			]
			groups = self._normalise_source_groups(parsed, constraints)
		completeness = self._completeness_check(parsed, constraints, groups, topic_entities)
		serialization = getattr(decision, "serialization", self._serialization_for(parsed, topic_entities, question))
		prompt_history.append({
			"type": "phase2_5_3b_branch_plan_serialization_router",
			"prompt": prompt,
			"response": response or "",
			"llm_error": _text((usage or {}).get("error")),
			"input_tokens": int((usage or {}).get("input_tokens", 0) or 0),
			"output_tokens": int((usage or {}).get("output_tokens", 0) or 0),
			"total_tokens": int((usage or {}).get("total_tokens", 0) or 0),
			"parsed_result": parsed if parsed is not None else {},
			"constraints": constraints,
			"independent_source_groups": groups,
			"independent_source_count": len(groups),
			"shared_target": dict(decision.shared_target),
			"dependency_pattern": decision.dependency_pattern,
			"validator_decision": decision.validator_decision,
			"validation_failures": list(decision.validation_failures or []),
			"extraction_completeness": completeness,
			"serialization": serialization,
			"serialized_branches": list(serialization.get("serialized_branches", [])),
			"retry_used": retry_used,
			"retry_reason": retry_reason,
			"prompt_version": ROUTER_PROMPT_VERSION,
		})
		return decision
