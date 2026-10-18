"""Semantic-target-aware relation selection for Phase 2.1.

This module deliberately sits beside the frozen Phase 1 methods.  It does not
change entity linking, relation discovery, routing, branch execution, or
answer evaluation.  It only ranks the already discovered directional relation
candidates using the current semantic target state.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from chain_of_relations.schema import Entity, Relation


@dataclass(frozen=True)
class SemanticTargetState:
	requested_final_answer_type: str = ""
	requested_final_semantic_property: str = ""
	current_subgoal: str = ""
	current_entity_type: str = ""
	remaining_question_constraint: str = ""
	reasoning_step_index: int = 0
	relation_history: Tuple[str, ...] = ()

	def as_dict(self) -> Dict[str, Any]:
		return {
			"requested_final_answer_type": self.requested_final_answer_type,
			"requested_final_semantic_property": self.requested_final_semantic_property,
			"current_subgoal": self.current_subgoal,
			"current_entity_type": self.current_entity_type,
			"remaining_question_constraint": self.remaining_question_constraint,
			"reasoning_step_index": self.reasoning_step_index,
			"relation_history": list(self.relation_history),
		}


@dataclass(frozen=True)
class RelationCandidate:
	relation: str
	direction: str
	label: str = ""

	def as_dict(self) -> Dict[str, str]:
		return {
			"relation": self.relation,
			"direction": self.direction,
			"label": self.label or self.relation,
		}


@dataclass
class RelationGroundingDecision:
	success: bool
	selected_relations: List[Relation]
	selected_direction: str = ""
	selection_reason: str = ""
	confidence: float = 0.0
	response: str = ""
	prompt: str = ""
	usage: Dict[str, Any] = field(default_factory=dict)
	warnings: List[str] = field(default_factory=list)


def _extract_json(response: Optional[str]) -> Any:
	if not response:
		return None
	text = str(response).strip()
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


def _normalise_direction(value: Any) -> str:
	text = str(value or "").strip().casefold()
	if text in {"outgoing", "out", "forward", "head", "left", "->"}:
		return "outgoing"
	if text in {"incoming", "in", "backward", "tail", "right", "<-"}:
		return "incoming"
	return ""


def _normalise_confidence(value: Any) -> float:
	try:
		return max(0.0, min(1.0, float(value)))
	except (TypeError, ValueError):
		return 0.0


def _history_value(relation_chain: List[Relation]) -> Tuple[str, ...]:
	return tuple(
		(f"{'->' if relation.left else '<-'}{relation.id}")
		for relation in relation_chain
	)


def build_semantic_target_prompt(
	question: str,
	topic_entity: Entity,
	state: SemanticTargetState,
	candidates: List[RelationCandidate],
) -> str:
	lines = [
		"You are a semantic-target-aware Freebase relation selector.",
		"Select the relation that best advances the current semantic subgoal toward the requested final answer.",
		"Prefer semantic compatibility over lexical similarity. Distinguish outgoing and incoming direction.",
		"Use only the question, runtime context, and candidate relations. Do not use gold answers, gold relations, or dataset labels.",
		"Return JSON only with fields: ranked_relations, selected_relation, direction, reason, confidence.",
		"ranked_relations must be an ordered list of candidate relation IDs or objects with relation and direction.",
		"Do not invent relation IDs. The selected relation must be one of the candidates.",
		"",
		f"Question: {question}",
		f"Current entity: {topic_entity.name} ({topic_entity.id})",
		f"Requested final answer type: {state.requested_final_answer_type}",
		f"Requested final semantic property: {state.requested_final_semantic_property}",
		f"Current subgoal: {state.current_subgoal}",
		f"Current entity type: {state.current_entity_type}",
		f"Remaining question constraint: {state.remaining_question_constraint}",
		f"Reasoning step index: {state.reasoning_step_index}",
		f"Relation history: {json.dumps(list(state.relation_history), ensure_ascii=False)}",
		"Candidate relations:",
	]
	for index, candidate in enumerate(candidates, start=1):
		lines.append(
			f"{index}. relation={candidate.relation} | direction={candidate.direction} | label={candidate.label or candidate.relation}"
		)
	return "\n".join(lines)


class SemanticTargetRelationGrounder:
	"""LLM relation scorer with strict candidate/direction validation."""

	def __init__(
		self,
		llm_generate: Callable[..., Any],
		temperature: float = 0.3,
		max_tokens: int = 512,
		max_retries: int = 3,
	) -> None:
		self.llm_generate = llm_generate
		self.temperature = temperature
		self.max_tokens = max_tokens
		self.max_retries = max(1, int(max_retries))

	@staticmethod
	def candidates(
		head_relations: List[str],
		tail_relations: List[str],
		relation_display_names: Optional[Dict[str, str]] = None,
	) -> List[RelationCandidate]:
		display = relation_display_names or {}
		result: List[RelationCandidate] = []
		seen: set[tuple[str, str]] = set()
		for relation in head_relations:
			key = (str(relation), "outgoing")
			if key not in seen:
				result.append(RelationCandidate(str(relation), "outgoing", str(display.get(relation, relation))))
				seen.add(key)
		for relation in tail_relations:
			key = (str(relation), "incoming")
			if key not in seen:
				result.append(RelationCandidate(str(relation), "incoming", str(display.get(relation, relation))))
				seen.add(key)
		return result

	@staticmethod
	def _entry_values(payload: Any) -> List[Tuple[str, str]]:
		if isinstance(payload, str):
			return [(payload.strip(), "")]
		if not isinstance(payload, dict):
			return []
		relation = payload.get("relation") or payload.get("relation_id") or payload.get("id") or ""
		direction = _normalise_direction(payload.get("direction"))
		return [(str(relation).strip(), direction)] if str(relation).strip() else []

	@classmethod
	def _ranked_entries(cls, parsed: Any) -> List[Tuple[str, str]]:
		if not isinstance(parsed, dict):
			return []
		value = parsed.get("ranked_relations")
		if value is None:
			value = parsed.get("relation_by_relevance")
		if value is None and parsed.get("selected_relation"):
			value = [
				{
					"relation": parsed.get("selected_relation"),
					"direction": parsed.get("direction", ""),
				}
			]
		if isinstance(value, str):
			value = [value]
		if not isinstance(value, list):
			return []
		entries: List[Tuple[str, str]] = []
		for item in value:
			entries.extend(cls._entry_values(item))
		return entries

	def _call(self, prompt: str) -> Tuple[Optional[str], Dict[str, Any]]:
		usage: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
		for _ in range(self.max_retries):
			try:
				try:
					response, usage = self.llm_generate(
						prompt,
						temperature=self.temperature,
						max_tokens=self.max_tokens,
						system_prompt="Return one JSON relation-selection object and no markdown.",
					)
				except TypeError:
					response, usage = self.llm_generate(
						prompt,
						temperature=self.temperature,
						max_tokens=self.max_tokens,
					)
				if response:
					return str(response), dict(usage or {})
			except Exception:
				continue
		return None, usage

	def select(
		self,
		question: str,
		topic_entity: Entity,
		relation_chain: List[Relation],
		head_relations: List[str],
		tail_relations: List[str],
		relation_display_names: Optional[Dict[str, str]],
		state: SemanticTargetState,
		top_k: int = 3,
	) -> RelationGroundingDecision:
		candidates = self.candidates(head_relations, tail_relations, relation_display_names)
		prompt = build_semantic_target_prompt(question, topic_entity, state, candidates)
		response, usage = self._call(prompt)
		warnings: List[str] = []
		if not response:
			return RelationGroundingDecision(False, [], response="", prompt=prompt, usage=usage, warnings=["empty model response"])
		parsed = _extract_json(response)
		if not isinstance(parsed, dict):
			return RelationGroundingDecision(False, [], response=response, prompt=prompt, usage=usage, warnings=["invalid JSON response"])

		by_key = {(item.relation, item.direction): item for item in candidates}
		by_relation: Dict[str, List[RelationCandidate]] = {}
		for item in candidates:
			by_relation.setdefault(item.relation, []).append(item)
		entries = self._ranked_entries(parsed)
		selected: List[Relation] = []
		seen: set[tuple[str, str]] = set()
		for relation_id, direction in entries:
			matches = [by_key.get((relation_id, direction))] if direction else by_relation.get(relation_id, [])
			matches = [item for item in matches if item is not None]
			if len(matches) != 1:
				if not matches:
					# Allow a display label only when it maps unambiguously.
					label_matches = [item for item in candidates if item.label == relation_id]
					matches = label_matches if direction == "" else [item for item in label_matches if item.direction == direction]
				if len(matches) != 1:
					continue
			item = matches[0]
			key = (item.relation, item.direction)
			if key in seen:
				continue
			seen.add(key)
			selected.append(Relation(item.relation, score=max(0.0, 1.0 - 0.01 * len(selected)), left=item.direction == "outgoing"))
			if len(selected) >= max(1, int(top_k)):
				break

		if relation_chain:
			previous = relation_chain[-1]
			selected = [
				item for item in selected
				if not (item.id == previous.id and item.left != previous.left)
			]
		if not selected:
			warnings.append("no valid candidate relation was selected")
			return RelationGroundingDecision(False, [], response=response, prompt=prompt, usage=usage, warnings=warnings)
		selected_relation = str(parsed.get("selected_relation", selected[0].id)).strip()
		selected_direction = _normalise_direction(parsed.get("direction"))
		if not selected_direction:
			selected_direction = "outgoing" if selected[0].left else "incoming"
		return RelationGroundingDecision(
			success=True,
			selected_relations=selected,
			selected_direction=selected_direction,
			selection_reason=str(parsed.get("reason") or parsed.get("selection_reason") or "").strip(),
			confidence=_normalise_confidence(parsed.get("confidence")),
			response=response,
			prompt=prompt,
			usage=usage,
			warnings=warnings,
		)


def relation_records(relations: List[Relation]) -> List[Dict[str, Any]]:
	return [
		{
			"relation": relation.id,
			"direction": "outgoing" if relation.left else "incoming",
			"score": relation.score,
		}
		for relation in relations
	]


def relation_history(relation_chain: List[Relation]) -> List[str]:
	return list(_history_value(relation_chain))
