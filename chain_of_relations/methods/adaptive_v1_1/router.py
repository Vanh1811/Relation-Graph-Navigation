"""LLM proposal and deterministic validation for Adaptive Routing v1.1.

The router sees only the question and inference-safe runtime topic entities.
It never receives gold answers, gold SPARQL, S-expressions, or structural
labels.  The deterministic validator is deliberately conservative: any
uncertain branch plan is sent to unchanged CoR.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from chain_of_relations.schema import Entity


MAX_BRANCHES = 3
DEFAULT_BRANCH_CONFIDENCE_THRESHOLD = 0.75
ROUTER_PROMPT_VERSION = "adaptive-routing-v1.1-json-v1"

_GENERIC_ROOTS = {
	"male",
	"female",
	"person",
	"people",
	"human",
	"school",
	"university",
	"college",
	"country",
	"city",
	"place",
	"location",
	"organization",
	"organisation",
	"date",
	"year",
	"phd",
	"degree",
	"profession",
	"occupation",
	"language",
	"type",
}


@dataclass(frozen=True)
class RouterConfig:
	"""Conservative routing settings recorded in experiment metadata."""

	max_branches: int = MAX_BRANCHES
	confidence_threshold: float = DEFAULT_BRANCH_CONFIDENCE_THRESHOLD


@dataclass
class RouterDecision:
	"""Validated or rejected route proposal plus reproducibility metadata."""

	router_mode: str = "cor"
	router_confidence: float = 0.0
	router_reason: str = ""
	branch_count: int = 0
	shared_answer_type: str = ""
	proposed_branches: list[dict[str, Any]] | None = None
	validation_passed: bool = True
	validation_failures: list[str] | None = None

	def __post_init__(self) -> None:
		if self.proposed_branches is None:
			self.proposed_branches = []
		if self.validation_failures is None:
			self.validation_failures = []

	def as_dict(self) -> dict[str, Any]:
		return {
			"router_mode": self.router_mode,
			"router_confidence": self.router_confidence,
			"router_reason": self.router_reason,
			"branch_count": self.branch_count,
			"shared_answer_type": self.shared_answer_type,
			"proposed_branches": list(self.proposed_branches or []),
			"validation_passed": self.validation_passed,
			"validation_failures": list(self.validation_failures or []),
		}


class AdaptiveRouter:
	"""Ask the model for a route and validate it without gold structure."""

	def __init__(
		self,
		llm_generate: Callable[..., tuple[str | None, dict[str, Any]]],
		config: RouterConfig | None = None,
	) -> None:
		self.llm_generate = llm_generate
		self.config = config or RouterConfig()
		if self.config.max_branches != MAX_BRANCHES:
			raise ValueError(f"Adaptive Routing v1.1 requires MAX_BRANCHES={MAX_BRANCHES}")

	@staticmethod
	def _normalise(value: Any) -> str:
		return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()

	@classmethod
	def is_generic_root(cls, name: str) -> bool:
		normalized = cls._normalise(name)
		if not normalized:
			return True
		if normalized in _GENERIC_ROOTS:
			return True
		tokens = set(normalized.split())
		# Semantic guard for common type/metadata concepts, while allowing
		# meaningful content entities such as "Country Music Hall of Fame".
		if tokens.intersection({"phd", "degree", "profession", "occupation"}):
			return True
		if len(tokens) <= 2 and tokens.intersection({"school", "university", "country", "city", "place", "person", "people"}):
			return True
		return False

	@staticmethod
	def _extract_json(response: str | None) -> Any:
		if not response:
			return None
		text = response.strip()
		if text.startswith("```"):
			text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
			text = re.sub(r"\s*```$", "", text)
		try:
			return json.loads(text)
		except json.JSONDecodeError:
			match = re.search(r"\{.*\}", text, flags=re.DOTALL)
			if not match:
				return None
			try:
				return json.loads(match.group(0))
			except json.JSONDecodeError:
				return None

	@staticmethod
	def _as_confidence(value: Any) -> float:
		try:
			return max(0.0, min(1.0, float(value)))
		except (TypeError, ValueError):
			return 0.0

	@classmethod
	def _normalise_branch(cls, branch: Any) -> dict[str, Any]:
		if not isinstance(branch, dict):
			return {}
		return {
			"branch_id": str(branch.get("branch_id", "")).strip(),
			"root_entity_id": str(branch.get("root_entity_id", "")).strip(),
			"root_entity_name": str(branch.get("root_entity_name", "")).strip(),
			"constraint_summary": str(branch.get("constraint_summary", "")).strip(),
		}

	def _validate(
		self,
		proposal: Any,
		question: str,
		topic_entities: list[Entity],
	) -> RouterDecision:
		if not isinstance(proposal, dict):
			return RouterDecision(
				validation_passed=False,
				validation_failures=["router output was not a JSON object"],
				router_reason="Invalid router output; conservative CoR fallback.",
			)

		mode = str(proposal.get("mode", "")).strip().casefold()
		confidence = self._as_confidence(proposal.get("confidence"))
		reason = str(proposal.get("reason", "")).strip()
		shared_answer_type = str(proposal.get("shared_answer_type", "")).strip()
		raw_branches = proposal.get("branches", [])
		branches = [self._normalise_branch(item) for item in raw_branches] if isinstance(raw_branches, list) else []
		try:
			proposed_count = int(proposal.get("branch_count", 0))
		except (TypeError, ValueError):
			proposed_count = -1

		if mode == "cor":
			failures = []
			if proposed_count not in (0,):
				failures.append("cor mode must declare branch_count=0")
			if branches:
				failures.append("cor mode must not include branch objects")
			if failures:
				return RouterDecision(
					router_mode="cor",
					router_confidence=confidence,
					router_reason=reason,
					validation_passed=False,
					validation_failures=failures,
				)
			return RouterDecision(
				router_mode="cor",
				router_confidence=confidence,
				router_reason=reason,
				validation_passed=True,
			)

		failures: list[str] = []
		if mode != "branch":
			failures.append("mode must be cor or branch")
		if proposed_count < 2:
			failures.append("branch_count must be at least 2")
		if proposed_count > self.config.max_branches:
			failures.append(f"branch_count exceeds MAX_BRANCHES={self.config.max_branches}")
		if len(branches) != proposed_count:
			failures.append("number of branch objects does not equal branch_count")
		if confidence < self.config.confidence_threshold:
			failures.append(
				f"confidence {confidence:.3f} is below threshold {self.config.confidence_threshold:.3f}"
			)
		if not shared_answer_type:
			failures.append("shared_answer_type is empty")
		if len(topic_entities) < proposed_count:
			failures.append("not enough runtime topic entities for proposed branches")

		known_by_id = {str(entity.id).strip(): entity for entity in topic_entities if str(entity.id).strip()}
		known_by_name = {self._normalise(entity.name): entity for entity in topic_entities if self._normalise(entity.name)}
		root_keys: list[str] = []
		branch_signatures: list[tuple[str, str]] = []
		for index, branch in enumerate(branches):
			prefix = f"branch_{index + 1}"
			if not branch.get("branch_id"):
				failures.append(f"{prefix} has no branch_id")
			if not branch.get("constraint_summary"):
				failures.append(f"{prefix} has an empty constraint_summary")
			root_id = branch.get("root_entity_id", "")
			root_name = branch.get("root_entity_name", "")
			entity = known_by_id.get(root_id) or known_by_name.get(self._normalise(root_name))
			if entity is None:
				failures.append(f"{prefix} root is not a linked runtime topic entity")
				continue
			if self.is_generic_root(entity.name):
				failures.append(f"{prefix} root is generic/type-like: {entity.name}")
			root_key = str(entity.id).strip() or self._normalise(entity.name)
			root_keys.append(root_key)
			branch_signatures.append((root_key, self._normalise(branch.get("constraint_summary"))))

		if len(root_keys) != len(set(root_keys)):
			failures.append("branch roots are duplicated")
		if len(branch_signatures) != len(set(branch_signatures)):
			failures.append("duplicate branch root and constraint")

		return RouterDecision(
			router_mode="branch" if mode == "branch" else "cor",
			router_confidence=confidence,
			router_reason=reason,
			branch_count=proposed_count if proposed_count >= 0 else 0,
			shared_answer_type=shared_answer_type,
			proposed_branches=branches,
			validation_passed=not failures,
			validation_failures=failures,
		)

	def _build_prompt(self, question: str, topic_entities: list[Entity]) -> str:
		entities = [
			{"id": str(entity.id), "name": str(entity.name)}
			for entity in topic_entities
		]
		return (
			"You are the reasoning-structure router for a relation-centric knowledge-graph QA system.\n"
			"Choose whether the question should use unchanged sequential CoR or independent shared-answer branches.\n\n"
			"Rules:\n"
			"- Use mode=cor for sequential paths such as A -> B -> C -> answer.\n"
			"- Use mode=branch only when at least two meaningful entities independently constrain the same answer.\n"
			"- A branch root must be a content entity grounded in the question, not a generic type, metadata word, degree, profession, school, gender, or location type.\n"
			"- Propose at most 3 branches. Do not invent entity IDs.\n"
			"- Be conservative: uncertainty means mode=cor.\n"
			"- Return JSON only, with exactly these top-level fields: mode, branch_count, shared_answer_type, confidence, reason, branches.\n"
			"- Each branch must contain branch_id, root_entity_id, root_entity_name, constraint_summary.\n\n"
			f"Question: {question}\n"
			f"Runtime topic entities: {json.dumps(entities, ensure_ascii=False)}\n"
		)

	def route(
		self,
		question: str,
		topic_entities: list[Entity],
		prompt_history: list[dict[str, Any]],
	) -> RouterDecision:
		prompt = self._build_prompt(question, topic_entities)
		response, usage = self.llm_generate(
			prompt,
			temperature=0.1,
			max_tokens=512,
			system_prompt="Return a single JSON routing object and no markdown.",
		)
		parsed = self._extract_json(response)
		decision = self._validate(parsed, question, topic_entities)
		prompt_history.append(
			{
				"type": "adaptive_router",
				"prompt": prompt,
				"response": response or "",
				"input_tokens": int((usage or {}).get("input_tokens", 0) or 0),
				"output_tokens": int((usage or {}).get("output_tokens", 0) or 0),
				"total_tokens": int((usage or {}).get("total_tokens", 0) or 0),
				"parsed_result": parsed if parsed is not None else {},
				"warnings": list(decision.validation_failures or []),
				"prompt_version": ROUTER_PROMPT_VERSION,
			}
		)
		return decision
