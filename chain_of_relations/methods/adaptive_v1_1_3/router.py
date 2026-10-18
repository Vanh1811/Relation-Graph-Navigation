"""Answer-type-aware routing and conservative branch validation for v1.1.3."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from chain_of_relations.schema import Entity


MAX_BRANCHES = 3
DEFAULT_BRANCH_CONFIDENCE_THRESHOLD = 0.75
ROUTER_PROMPT_VERSION = "adaptive-routing-v1.1.3-answer-type-aware-json-v1"
STRUCTURES = {"LINEAR", "BRANCH_FINAL", "BRANCH_INTERMEDIATE", "UNCERTAIN"}

_GENERIC_ROOTS = {
	"male", "female", "person", "people", "human", "school", "university",
	"college", "country", "city", "place", "location", "organization",
	"organisation", "date", "year", "phd", "degree", "profession",
	"occupation", "language", "type",
}

_TYPE_ALIASES = {
	"sex": "gender",
	"parent sex": "gender",
	"parent gender": "gender",
	"parent_gender": "gender",
	"parent_sex": "gender",
	"gender": "gender",
	"genus": "taxonomy",
	"taxonomy": "taxonomy",
	"higher classification": "taxonomy",
	"higher_classification": "taxonomy",
	"occupation": "occupation",
	"profession": "occupation",
	"shared occupation": "occupation",
	"cheese": "cheese",
	"primary election": "primary election",
}

_INCOMPATIBLE_PROPERTY_PAIRS = {
	frozenset({"occupation", "rank"}),
	frozenset({"occupation", "taxonomy"}),
	frozenset({"gender", "taxonomy"}),
	frozenset({"gender", "rank"}),
}


def _normalise(value: Any) -> str:
	return re.sub(r"[^a-z0-9_]+", " ", str(value or "").casefold()).strip()


def _semantic_group(value: Any) -> str:
	text = _normalise(value).replace("_", " ")
	if not text:
		return ""
	for alias, group in sorted(_TYPE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
		if alias.replace("_", " ") in text:
			return group
	return text


def extract_requested_answer_target(question: str) -> dict[str, str]:
	"""Extract a conservative answer-type/property hint from question wording.

	This is not a gold-structure lookup.  It is a small semantic safety layer
	used to catch high-cost answer-type inversions such as parent gender→genus.
	Unknown questions intentionally remain broad so the router can choose
	LINEAR/UNCERTAIN rather than being forced into a fabricated type.
	"""
	text = _normalise(question).replace("_", " ")
	if "parent gender" in text or "parent sex" in text or "hybrid parent gender" in text:
		return {
			"requested_final_answer_type": "gender",
			"requested_final_semantic_property": "hybrid parent gender / parent sex",
		}
	if re.search(r"\b(gender|sex)\b", text):
		return {
			"requested_final_answer_type": "gender",
			"requested_final_semantic_property": "gender / sex",
		}
	if "genus" in text or "taxonomy" in text or "higher classification" in text:
		return {
			"requested_final_answer_type": "genus",
			"requested_final_semantic_property": "taxonomy / higher classification",
		}
	if "occupation" in text or "profession" in text:
		return {
			"requested_final_answer_type": "occupation",
			"requested_final_semantic_property": "occupation / profession",
		}
	if "primary election" in text:
		return {
			"requested_final_answer_type": "primary election",
			"requested_final_semantic_property": "associated primary election",
		}
	if "cheese" in text:
		return {
			"requested_final_answer_type": "cheese",
			"requested_final_semantic_property": "cheese satisfying the stated constraints",
		}
	if "what do " in text and " have in common" in text:
		return {
			"requested_final_answer_type": "shared property or entity",
			"requested_final_semantic_property": "shared property explicitly grounded by both entities",
		}
	return {
		"requested_final_answer_type": "",
		"requested_final_semantic_property": "",
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
	requested_final_answer_type: str = ""
	requested_final_semantic_property: str = ""
	shared_answer_type: str = ""
	proposed_branches: list[dict[str, Any]] | None = None
	branch_property_compatibility: bool | None = None
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
			"requested_final_answer_type": self.requested_final_answer_type,
			"requested_final_semantic_property": self.requested_final_semantic_property,
			"shared_answer_type": self.shared_answer_type,
			"branch_count": len(self.proposed_branches or []) if self.structure == "BRANCH_FINAL" else 0,
			"proposed_branches": list(self.proposed_branches or []),
			"branch_property_compatibility": self.branch_property_compatibility,
			"unsupported_structure": self.unsupported_structure,
			"validation_passed": self.validation_passed,
			"validation_failed": not self.validation_passed,
			"validation_failures": list(self.validation_failures or []),
		}


class AdaptiveRouter:
	"""Ask for a route and reject semantically incompatible branch plans."""

	def __init__(self, llm_generate: Callable[..., tuple[str | None, dict[str, Any]]], config: RouterConfig | None = None) -> None:
		self.llm_generate = llm_generate
		self.config = config or RouterConfig()
		if self.config.max_branches != MAX_BRANCHES:
			raise ValueError(f"Adaptive Routing v1.1.3 requires MAX_BRANCHES={MAX_BRANCHES}")

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

	@classmethod
	def is_generic_root(cls, name: str) -> bool:
		normalized = _normalise(name)
		if not normalized or normalized in _GENERIC_ROOTS:
			return True
		tokens = set(normalized.split())
		return bool(tokens.intersection({"phd", "degree", "profession", "occupation"})) or (
			len(tokens) <= 2 and bool(tokens.intersection({"school", "university", "country", "city", "place", "person", "people"}))
		)

	@classmethod
	def _normalise_branch(cls, branch: Any) -> dict[str, Any]:
		if not isinstance(branch, dict):
			return {}
		return {
			"branch_id": str(branch.get("branch_id", "")).strip(),
			"root_entity_id": str(branch.get("root_entity_id", "")).strip(),
			"root_entity_name": str(branch.get("root_entity_name", "")).strip(),
			"constraint_property": str(branch.get("constraint_property", "")).strip(),
			"constraint_summary": str(branch.get("constraint_summary", "")).strip(),
			"target_answer_type": str(branch.get("target_answer_type", "")).strip(),
		}

	@staticmethod
	def _compatible_properties(properties: list[str], requested: str) -> bool:
		groups = [_semantic_group(value) for value in properties]
		groups = [value for value in groups if value]
		if len(groups) != len(properties) or not groups:
			return False
		requested_group = _semantic_group(requested)
		if requested_group in {"gender", "occupation", "taxonomy", "primary election"}:
			return all(group == requested_group for group in groups)
		for left_index, left in enumerate(groups):
			for right in groups[left_index + 1:]:
				if frozenset({left, right}) in _INCOMPATIBLE_PROPERTY_PAIRS:
					return False
		# Multiple independent constraints such as milk-source and texture are
		# compatible when they target a concrete answer space such as cheese.
		return True

	def _validate(self, proposal: Any, question: str, topic_entities: list[Entity]) -> RouterDecision:
		hint = extract_requested_answer_target(question)
		if not isinstance(proposal, dict):
			return RouterDecision(
				requested_final_answer_type=hint["requested_final_answer_type"],
				requested_final_semantic_property=hint["requested_final_semantic_property"],
				validation_passed=False,
				validation_failures=["router output was not a JSON object"],
				reason="Invalid router output; execute unchanged CoR.",
			)

		structure = str(proposal.get("structure", "")).strip().upper()
		if not structure and str(proposal.get("mode", "")).casefold() == "cor":
			structure = "LINEAR"
		confidence = self._confidence(proposal.get("confidence"))
		reason = str(proposal.get("reason", "")).strip()
		type_value = str(proposal.get("requested_final_answer_type", "")).strip() or str(proposal.get("shared_answer_type", "")).strip()
		property_value = str(proposal.get("requested_final_semantic_property", "")).strip()
		shared_type = str(proposal.get("shared_answer_type", "")).strip() or type_value
		if not type_value:
			type_value = hint["requested_final_answer_type"]
		if not property_value:
			property_value = hint["requested_final_semantic_property"]
		raw_branches = proposal.get("branches", [])
		branches = [self._normalise_branch(item) for item in raw_branches] if isinstance(raw_branches, list) else []
		failures: list[str] = []

		if structure not in STRUCTURES:
			failures.append("structure must be LINEAR, BRANCH_FINAL, BRANCH_INTERMEDIATE, or UNCERTAIN")
		if confidence < self.config.confidence_threshold:
			failures.append(f"confidence {confidence:.3f} is below threshold {self.config.confidence_threshold:.3f}")
		if hint["requested_final_answer_type"] and _semantic_group(type_value) and _semantic_group(type_value) != _semantic_group(hint["requested_final_answer_type"]):
			failures.append("ANSWER_TYPE_MISMATCH: router target conflicts with explicit question wording")
		if not type_value:
			failures.append("requested_final_answer_type is empty")
		if not property_value:
			failures.append("requested_final_semantic_property is empty")

		if structure in {"LINEAR", "UNCERTAIN", "BRANCH_INTERMEDIATE"}:
			if branches:
				failures.append(f"{structure} must not include executable branch objects")
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

		known_by_id = {str(entity.id).strip(): entity for entity in topic_entities if str(entity.id).strip()}
		known_by_name = {_normalise(entity.name): entity for entity in topic_entities if _normalise(entity.name)}
		root_keys: list[str] = []
		branch_signatures: list[tuple[str, str]] = []
		branch_properties: list[str] = []
		branch_types: list[str] = []
		if structure == "BRANCH_FINAL":
			for index, branch in enumerate(branches):
				prefix = f"branch_{index + 1}"
				if not branch.get("branch_id"):
					failures.append(f"{prefix} has no branch_id")
				if not branch.get("constraint_property"):
					failures.append(f"{prefix} has no constraint_property")
				if not branch.get("constraint_summary"):
					failures.append(f"{prefix} has an empty constraint_summary")
				if not branch.get("target_answer_type"):
					failures.append(f"{prefix} has no target_answer_type")
				entity = known_by_id.get(branch.get("root_entity_id")) or known_by_name.get(_normalise(branch.get("root_entity_name")))
				if entity is None:
					failures.append(f"{prefix} root is not a linked runtime topic entity")
					continue
				if self.is_generic_root(entity.name):
					failures.append(f"{prefix} root is generic/type-like: {entity.name}")
				root_key = str(entity.id).strip() or _normalise(entity.name)
				root_keys.append(root_key)
				branch_signatures.append((root_key, _normalise(branch.get("constraint_summary"))))
				branch_properties.append(branch.get("constraint_property", ""))
				branch_types.append(branch.get("target_answer_type", ""))
			if len(root_keys) != len(set(root_keys)):
				failures.append("branch roots are duplicated")
			if len(branch_signatures) != len(set(branch_signatures)):
				failures.append("duplicate or redundant branch root and constraint")
			if not self._compatible_properties(branch_properties, property_value):
				failures.append("CONSTRAINT_PROPERTY_MISMATCH: branch constraint properties are incompatible")
			if any(_semantic_group(value) != _semantic_group(type_value) for value in branch_types if value and type_value):
				failures.append("branch targets do not match requested final answer type")

		unsupported = structure == "BRANCH_INTERMEDIATE"
		if structure == "BRANCH_FINAL" and failures:
			execution_mode = "cor"
		if structure not in STRUCTURES:
			execution_mode = "cor"
		return RouterDecision(
			structure=structure if structure in STRUCTURES else "UNCERTAIN",
			execution_mode=execution_mode,
			confidence=confidence,
			reason=reason,
			requested_final_answer_type=type_value,
			requested_final_semantic_property=property_value,
			shared_answer_type=shared_type,
			proposed_branches=branches if structure == "BRANCH_FINAL" and not failures else [],
			branch_property_compatibility=(not any("CONSTRAINT_PROPERTY_MISMATCH" in item for item in failures)) if structure == "BRANCH_FINAL" else None,
			unsupported_structure=unsupported,
			validation_passed=not failures,
			validation_failures=failures,
		)

	def _build_prompt(self, question: str, topic_entities: list[Entity]) -> str:
		entities = [{"id": str(entity.id), "name": str(entity.name)} for entity in topic_entities]
		hint = extract_requested_answer_target(question)
		generic = ", ".join(sorted(_GENERIC_ROOTS))
		return (
			"You are the v1.1.3 reasoning-structure router for a relation-centric knowledge-graph QA system.\n"
			"Use only the question and linked runtime entities. Never use gold answers, gold SPARQL, S-expressions, or dataset labels.\n\n"
			"First identify the exact requested final answer type and semantic property. Do not substitute a related taxonomy concept.\n"
			"For example, hybrid parent gender means answer type gender and property parent_sex/parent_gender, not genus or higher classification.\n"
			"For BRANCH_FINAL, every branch must state constraint_property, constraint_summary, and target_answer_type.\n"
			"All branch constraint_property values must describe the same semantic property; occupation and rank are incompatible.\n"
			"If answer type or branch-property compatibility is unclear, choose UNCERTAIN or LINEAR and execute CoR.\n\n"
			"Structures:\n"
			"- LINEAR: sequential path or semantically aligned constraints that are not independent final-answer constraints.\n"
			"- BRANCH_FINAL: at least two meaningful grounded entities independently constrain the same final answer/property.\n"
			"- BRANCH_INTERMEDIATE: constraints meet at an intermediate; it is logged but executed as CoR.\n"
			"- UNCERTAIN: ambiguity or incompatible constraints; execute CoR.\n"
			f"Generic/type-like roots usually invalid for branching: {generic}.\n\n"
			"Return JSON only with exactly these top-level fields: structure, execution_mode, confidence, reason, requested_final_answer_type, requested_final_semantic_property, shared_answer_type, branches, unsupported_structure.\n"
			"For BRANCH_FINAL, branches contain branch_id, root_entity_id, root_entity_name, constraint_property, constraint_summary, target_answer_type. For other structures, branches must be [].\n\n"
			f"Question: {question}\n"
			f"Lexical safety hint (verify semantically, do not blindly copy): {json.dumps(hint, ensure_ascii=False)}\n"
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
		decision = self._validate(parsed, question, topic_entities)
		prompt_history.append(
			{
				"type": "adaptive_router_v1_1_3",
				"prompt": prompt,
				"response": response or "",
				"input_tokens": int((usage or {}).get("input_tokens", 0) or 0),
				"output_tokens": int((usage or {}).get("output_tokens", 0) or 0),
				"total_tokens": int((usage or {}).get("total_tokens", 0) or 0),
				"parsed_result": parsed if parsed is not None else {},
				"requested_final_answer_type": decision.requested_final_answer_type,
				"requested_final_semantic_property": decision.requested_final_semantic_property,
				"branch_property_compatibility": decision.branch_property_compatibility,
				"warnings": list(decision.validation_failures or []),
				"prompt_version": ROUTER_PROMPT_VERSION,
			}
		)
		return decision
