"""Rule-aware structural routing for Adaptive v1.1.2.

The model proposes a structure, while this module deterministically validates
the proposal using only the question and linked runtime topic entities.  Gold
answers, gold SPARQL, and dataset structure labels never enter routing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from chain_of_relations.schema import Entity


MAX_BRANCHES = 3
DEFAULT_BRANCH_CONFIDENCE_THRESHOLD = 0.75
ROUTER_PROMPT_VERSION = "adaptive-routing-v1.1.2-rule-aware-json-v1"
STRUCTURES = {"LINEAR", "BRANCH_FINAL", "BRANCH_INTERMEDIATE", "UNCERTAIN"}
EXECUTION_MODES = {"cor", "branch"}

_GENERIC_ROOTS = {
	"male", "female", "person", "people", "human", "school", "university",
	"college", "country", "city", "place", "location", "organization",
	"organisation", "date", "year", "phd", "degree", "profession",
	"occupation", "language", "type",
}


@dataclass(frozen=True)
class RouterConfig:
	max_branches: int = MAX_BRANCHES
	confidence_threshold: float = DEFAULT_BRANCH_CONFIDENCE_THRESHOLD


@dataclass
class RouterDecision:
	structure: str = "UNCERTAIN"
	execution_mode: str = "cor"
	confidence: float = 0.0
	reason: str = ""
	shared_answer_type: str = ""
	proposed_branches: list[dict[str, Any]] | None = None
	unsupported_structure: bool = False
	validation_passed: bool = True
	validation_failures: list[str] | None = None

	def __post_init__(self) -> None:
		if self.proposed_branches is None:
			self.proposed_branches = []
		if self.validation_failures is None:
			self.validation_failures = []

	@property
	def router_mode(self) -> str:
		return "branch" if self.execution_mode == "branch" else "cor"

	def as_dict(self) -> dict[str, Any]:
		return {
			"router_structure": self.structure,
			"router_execution_mode": self.execution_mode,
			"router_mode": self.router_mode,
			"router_confidence": self.confidence,
			"router_reason": self.reason,
			"branch_count": len(self.proposed_branches or []) if self.structure == "BRANCH_FINAL" else 0,
			"shared_answer_type": self.shared_answer_type,
			"proposed_branches": list(self.proposed_branches or []),
			"unsupported_structure": self.unsupported_structure,
			"validation_passed": self.validation_passed,
			"validation_failed": not self.validation_passed,
			"validation_failures": list(self.validation_failures or []),
		}


class AdaptiveRouter:
	"""Ask for a structural proposal and apply conservative validation."""

	def __init__(
		self,
		llm_generate: Callable[..., tuple[str | None, dict[str, Any]]],
		config: RouterConfig | None = None,
	) -> None:
		self.llm_generate = llm_generate
		self.config = config or RouterConfig()
		if self.config.max_branches != MAX_BRANCHES:
			raise ValueError(f"Adaptive Routing v1.1.2 requires MAX_BRANCHES={MAX_BRANCHES}")

	@staticmethod
	def _normalise(value: Any) -> str:
		return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()

	@classmethod
	def is_generic_root(cls, name: str) -> bool:
		normalized = cls._normalise(name)
		if not normalized or normalized in _GENERIC_ROOTS:
			return True
		tokens = set(normalized.split())
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
	def _confidence(value: Any) -> float:
		try:
			return max(0.0, min(1.0, float(value)))
		except (TypeError, ValueError):
			return 0.0

	@staticmethod
	def _text(value: Any) -> str:
		return str(value or "").strip()

	@classmethod
	def _normalise_branch(cls, branch: Any) -> dict[str, Any]:
		if not isinstance(branch, dict):
			return {}
		return {
			"branch_id": cls._text(branch.get("branch_id")),
			"root_entity_id": cls._text(branch.get("root_entity_id")),
			"root_entity_name": cls._text(branch.get("root_entity_name")),
			"constraint_summary": cls._text(branch.get("constraint_summary")),
			"target_answer_type": cls._text(branch.get("target_answer_type")),
		}

	def _validate(self, proposal: Any, topic_entities: list[Entity]) -> RouterDecision:
		if not isinstance(proposal, dict):
			return RouterDecision(
				validation_passed=False,
				validation_failures=["router output was not a JSON object"],
				reason="Invalid router output; execute unchanged CoR.",
			)

		structure = self._text(proposal.get("structure")).upper()
		if not structure and self._text(proposal.get("mode")).casefold() == "cor":
			structure = "LINEAR"
		confidence = self._confidence(proposal.get("confidence"))
		reason = self._text(proposal.get("reason"))
		shared_type = self._text(proposal.get("shared_answer_type"))
		raw_branches = proposal.get("branches", [])
		branches = [self._normalise_branch(item) for item in raw_branches] if isinstance(raw_branches, list) else []
		failures: list[str] = []

		if structure not in STRUCTURES:
			failures.append("structure must be LINEAR, BRANCH_FINAL, BRANCH_INTERMEDIATE, or UNCERTAIN")
		if confidence < self.config.confidence_threshold:
			failures.append(f"confidence {confidence:.3f} is below threshold {self.config.confidence_threshold:.3f}")

		if structure in {"LINEAR", "UNCERTAIN", "BRANCH_INTERMEDIATE"}:
			if branches:
				failures.append(f"{structure} must not include executable branch objects")
			if proposal.get("branch_count", 0) not in (0, None, "0", ""):
				failures.append(f"{structure} must declare branch_count=0")
			execution_mode = "cor"
		else:
			execution_mode = "branch"
			try:
				proposed_count = int(proposal.get("branch_count", len(branches)))
			except (TypeError, ValueError):
				proposed_count = -1
			if proposed_count < 2:
				failures.append("BRANCH_FINAL requires at least two branches")
			if proposed_count > self.config.max_branches:
				failures.append(f"branch_count exceeds MAX_BRANCHES={self.config.max_branches}")
			if len(branches) != proposed_count:
				failures.append("number of branch objects does not equal branch_count")
			if not shared_type:
				failures.append("shared_answer_type is empty")
			if len(topic_entities) < max(proposed_count, 0):
				failures.append("not enough runtime topic entities for proposed branches")

		known_by_id = {self._text(entity.id): entity for entity in topic_entities if self._text(entity.id)}
		known_by_name = {self._normalise(entity.name): entity for entity in topic_entities if self._normalise(entity.name)}
		root_keys: list[str] = []
		branch_signatures: list[tuple[str, str]] = []
		answer_types: list[str] = []
		if structure == "BRANCH_FINAL":
			for index, branch in enumerate(branches):
				prefix = f"branch_{index + 1}"
				if not branch.get("branch_id"):
					failures.append(f"{prefix} has no branch_id")
				if not branch.get("constraint_summary"):
					failures.append(f"{prefix} has an empty constraint_summary")
				entity = known_by_id.get(branch.get("root_entity_id")) or known_by_name.get(self._normalise(branch.get("root_entity_name")))
				if entity is None:
					failures.append(f"{prefix} root is not a linked runtime topic entity")
					continue
				if self.is_generic_root(entity.name):
					failures.append(f"{prefix} root is generic/type-like: {entity.name}")
				root_key = self._text(entity.id) or self._normalise(entity.name)
				root_keys.append(root_key)
				branch_signatures.append((root_key, self._normalise(branch.get("constraint_summary"))))
				if branch.get("target_answer_type"):
					answer_types.append(self._normalise(branch["target_answer_type"]))
			if len(root_keys) != len(set(root_keys)):
				failures.append("branch roots are duplicated")
			if len(branch_signatures) != len(set(branch_signatures)):
				failures.append("duplicate or redundant branch root and constraint")
			if answer_types and any(item != self._normalise(shared_type) for item in answer_types):
				failures.append("branch constraints target different final answer spaces")

		unsupported = structure == "BRANCH_INTERMEDIATE"
		if structure == "BRANCH_INTERMEDIATE":
			execution_mode = "cor"
		if structure in {"LINEAR", "UNCERTAIN"}:
			execution_mode = "cor"
		if structure == "BRANCH_FINAL" and failures:
			execution_mode = "cor"
		if structure not in STRUCTURES:
			execution_mode = "cor"
		return RouterDecision(
			structure=structure if structure in STRUCTURES else "UNCERTAIN",
			execution_mode=execution_mode,
			confidence=confidence,
			reason=reason,
			shared_answer_type=shared_type,
			proposed_branches=branches if structure == "BRANCH_FINAL" and not failures else [],
			unsupported_structure=unsupported,
			validation_passed=not failures,
			validation_failures=failures,
		)

	def _build_prompt(self, question: str, topic_entities: list[Entity]) -> str:
		entities = [{"id": self._text(entity.id), "name": self._text(entity.name)} for entity in topic_entities]
		generic = ", ".join(sorted(_GENERIC_ROOTS))
		return (
			"You are the reasoning-structure router for a relation-centric knowledge-graph QA system.\n"
			"Choose the structure before execution using only the question and linked runtime entities.\n"
			"Do not use gold answers, gold SPARQL, S-expressions, or dataset labels.\n\n"
			"Structural rules:\n"
			"- LINEAR: use for sequential reasoning A -> B -> C -> answer, one meaningful content path, or concepts that do not independently constrain the same final answer. Generic/type-like roots also imply LINEAR.\n"
			"- BRANCH_FINAL: use only when at least two independent, non-redundant semantic constraints from grounded content entities directly converge on the requested final answer. Each branch must be independently explorable and target the same final answer type.\n"
			"- BRANCH_INTERMEDIATE: use when independent constraints converge on an intermediate entity and an additional relation is needed to reach the requested answer. This structure is logged but executed as CoR; do not propose executable branches for it.\n"
			"- UNCERTAIN: use when independence, roots, or shared answer compatibility is unclear. UNCERTAIN executes as CoR.\n"
			f"Generic/type-like roots usually invalid for branching: {generic}. This is semantic guidance, not a strict blacklist.\n"
			"Return JSON only with exactly: structure, execution_mode, confidence, reason, shared_answer_type, branches, unsupported_structure.\n"
			"For BRANCH_FINAL, execution_mode must be branch and branches must contain branch_id, root_entity_id, root_entity_name, constraint_summary, and optional target_answer_type. For all other structures, execution_mode must be cor and branches must be [].\n\n"
			f"Question: {question}\n"
			f"Runtime topic entities: {json.dumps(entities, ensure_ascii=False)}\n"
		)

	def route(self, question: str, topic_entities: list[Entity], prompt_history: list[dict[str, Any]]) -> RouterDecision:
		prompt = self._build_prompt(question, topic_entities)
		response, usage = self.llm_generate(
			prompt,
			temperature=0.1,
			max_tokens=768,
			system_prompt="Return one JSON routing object and no markdown.",
		)
		parsed = self._extract_json(response)
		decision = self._validate(parsed, topic_entities)
		prompt_history.append(
			{
				"type": "adaptive_router_v1_1_2",
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
