"""Minimal explicit branch wrapper around the unchanged CoR agent.

Branch-v1 deliberately reuses CoR's relation search, relation pruning, entity
search, reasoning, and prompts.  Its default behavior remains the original
two-runtime-root execution.  Adaptive v1.1 may pass a validated three-root
plan through the same executor.  It never consumes gold SPARQL, S-expressions,
structure labels, answer IDs, or a gold branch count.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from typing import Any, TYPE_CHECKING

from chain_of_relations.methods.branch_v1.state import BranchState, RelationGraphState
from chain_of_relations.schema import Entity

if TYPE_CHECKING:
	from chain_of_relations.methods.cor.agent import CoRAgent


class BranchV1Agent:
	"""Run CoR on validated runtime topic roots and merge final candidates."""

	def __init__(self, cor_agent: CoRAgent):
		self.cor_agent = cor_agent

	@staticmethod
	def _normalise_text(value: Any) -> str:
		text = unicodedata.normalize("NFKC", str(value or "")).strip()
		text = text.translate(str.maketrans({
			"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "…": "...",
		}))
		text = re.sub(r"\s+", " ", text).strip()
		for left, right in (("\"", "\""), ("'", "'")):
			while len(text) >= 2 and text.startswith(left) and text.endswith(right):
				text = text[1:-1].strip()
		return text.casefold()

	@staticmethod
	def _candidate_id(item: Any) -> str:
		return str(item.get("id", "")).strip() if isinstance(item, dict) else ""

	@staticmethod
	def _candidate_name(item: Any) -> str:
		if isinstance(item, dict):
			return str(item.get("name", "")).strip()
		return str(item or "").strip()

	@classmethod
	def _candidate_snapshot(cls, items: list[Any]) -> list[dict[str, str]]:
		return [
			{"id": cls._candidate_id(item), "name": cls._candidate_name(item)}
			for item in items
			if cls._candidate_id(item) or cls._candidate_name(item)
		]

	@classmethod
	def _result_key(cls, item: Any) -> tuple[str, str] | None:
		item_id = cls._candidate_id(item)
		name = cls._candidate_name(item)
		if item_id:
			return ("id", item_id)
		if name:
			return ("text", cls._normalise_text(name))
		return None

	@classmethod
	def _merge_candidates(cls, branch_results: list[list[Any]]) -> tuple[list[Any], list[str]]:
		if len(branch_results) not in (2, 3):
			raise ValueError("MERGE_AND requires two or three branch result sets")

		merged: list[Any] = []
		methods: list[str] = []
		used_indices = [set() for _ in branch_results]

		# Preserve exact Freebase identity whenever every branch provides it.
		for first_index, first_item in enumerate(branch_results[0]):
			item_id = cls._candidate_id(first_item)
			if not item_id:
				continue
			matches: list[int] = []
			for branch_index in range(1, len(branch_results)):
				match = next(
					(
						candidate_index
						for candidate_index, candidate in enumerate(branch_results[branch_index])
						if candidate_index not in used_indices[branch_index]
						and cls._candidate_id(candidate) == item_id
					),
					None,
				)
				if match is None:
					break
				matches.append(match)
			if len(matches) != len(branch_results) - 1:
				continue
			used_indices[0].add(first_index)
			for branch_index, match in enumerate(matches, start=1):
				used_indices[branch_index].add(match)
			merged.append(first_item)
			methods.append("ID")

		# Then allow only exact normalized-label equality, including the safe case
		# where one branch has an ID and another only has a generated label.
		for first_index, first_item in enumerate(branch_results[0]):
			if first_index in used_indices[0]:
				continue
			label = cls._normalise_text(cls._candidate_name(first_item))
			if not label:
				continue
			matches = []
			for branch_index in range(1, len(branch_results)):
				match = next(
					(
						candidate_index
						for candidate_index, candidate in enumerate(branch_results[branch_index])
						if candidate_index not in used_indices[branch_index]
						and cls._normalise_text(cls._candidate_name(candidate)) == label
					),
					None,
				)
				if match is None:
					break
				matches.append(match)
			if len(matches) != len(branch_results) - 1:
				continue
			used_indices[0].add(first_index)
			for branch_index, match in enumerate(matches, start=1):
				used_indices[branch_index].add(match)
			merged.append(first_item)
			methods.append("NORMALIZED_LABEL")

		return merged, methods

	@classmethod
	def _merge_and(cls, branch_results: list[list[Any]]) -> list[Any]:
		"""Backward-compatible list-only merge used by existing callers/tests."""
		merged, _ = cls._merge_candidates(branch_results)
		return merged

	@staticmethod
	def _relation_chain_from_result(result: dict[str, Any]) -> list[str]:
		"""Extract the final relation path from the unchanged CoR result."""
		direct_chain = result.get("relation_chain", [])
		if isinstance(direct_chain, list) and direct_chain:
			return [str(item) for item in direct_chain if str(item).strip()]

		reasoning_chains = result.get("reasoning_chains", "")
		if isinstance(reasoning_chains, list):
			reasoning_chains = "\n".join(str(item) for item in reasoning_chains)
		if not isinstance(reasoning_chains, str):
			return []

		relations: list[str] = []
		for line in reasoning_chains.splitlines():
			match = re.search(r"\)\s+(\S+)\s+e\d+\s+\.", line)
			if match:
				relations.append(match.group(1))
				continue
			match = re.search(r"e\d+\s+(\S+)\s+e\d+\s+\.", line)
			if match:
				relations.append(match.group(1))
		return relations

	@staticmethod
	def _branch_trace(
		branch: BranchState,
		topic_entity: Entity,
		result: dict[str, Any],
		sparql_history: list[dict[str, Any]],
		step_history: list[dict[str, Any]],
		failure: str | None = None,
	) -> dict[str, Any]:
		retrieved_candidates: list[dict[str, Any]] = []
		for event in sparql_history:
			if event.get("search_type") != "entity_search":
				continue
			ids = [str(value).strip() for value in event.get("candidate_entity_ids", [])]
			names = [str(value).strip() for value in event.get("candidate_entity_names", [])]
			literals = [str(value).strip() for value in event.get("literal_candidates", [])]
			candidates = [
				{"id": item_id, "name": name}
				for item_id, name in zip(ids, names)
				if item_id or name
			]
			candidates.extend({"id": "", "name": value} for value in literals if value)
			retrieved_candidates.append(
				{
					"query": event.get("sparql", ""),
					"result_count": event.get("result_count", 0),
					"candidates": candidates,
				}
			)
		post_filter_candidates = [
			item.get("answers", [])
			for item in step_history
			if item.get("step_type") == "llm" and item.get("answers")
		]
		final_candidates = BranchV1Agent._candidate_snapshot(result.get("results", []) or [])
		if failure:
			status = "EXECUTION_ERROR"
		elif final_candidates:
			status = "SUCCESS"
		elif retrieved_candidates and all(int(item.get("result_count", 0) or 0) == 0 for item in retrieved_candidates):
			status = "EMPTY_AFTER_RETRIEVAL"
		else:
			status = "EMPTY_AFTER_FILTERING"
		return {
			"branch_id": branch.branch_id,
			"topic_entity": {"id": topic_entity.id, "name": topic_entity.name},
			"relation_chain": list(branch.relation_chain),
			"status": status,
			"candidate_count": len(result.get("results", []) or []),
			"candidate_ids": [
				str(item.get("id", ""))
				for item in result.get("results", []) or []
				if isinstance(item, dict) and str(item.get("id", "")).strip()
			],
			"failure": failure,
			"action": result.get("action", ""),
			"history": list(branch.history),
			"retrieved_candidates": retrieved_candidates,
			"post_filter_candidates": post_filter_candidates,
			"final_branch_answers": final_candidates,
			"merge_inputs": final_candidates,
		}

	def answer(
		self,
		question: str,
		topic_entities: list[Entity],
		branch_plan: list[dict[str, Any]] | None = None,
		shared_answer_type: str = "",
		requested_final_answer_type: str = "",
		requested_final_semantic_property: str = "",
		**kwargs: Any,
	) -> dict[str, Any]:
		if not topic_entities:
			return {
				"action": "ERROR",
				"question": question,
				"results": [],
				"reasoning_chains": "No topic_entities provided",
				"branch_failures": ["no topic entities"],
			}

		# Without an explicit plan this preserves the frozen Branch-v1 behavior:
		# use at most the first two runtime entities and reuse a single root.  The
		# adaptive wrapper passes a validated two- or three-branch plan here.
		explicit_plan = branch_plan is not None
		shared_root_mode = False
		if explicit_plan:
			entities_by_id = {str(entity.id): entity for entity in topic_entities}
			entities_by_name = {str(entity.name).casefold(): entity for entity in topic_entities}
			execution_entities: list[Entity] = []
			for item in branch_plan or []:
				entity = entities_by_id.get(str(item.get("root_entity_id", "")))
				if entity is None:
					entity = entities_by_name.get(str(item.get("root_entity_name", "")).casefold())
				if entity is None:
					raise ValueError("validated branch plan contains an unknown runtime root")
				execution_entities.append(entity)
			if len(execution_entities) not in (2, 3):
				raise ValueError("explicit branch plan must contain two or three roots")
		else:
			execution_entities = list(topic_entities[:2])
			if not execution_entities:
				raise ValueError("no runtime topic entities available")
		shared_root_mode = len(execution_entities) == 1
		if shared_root_mode:
			execution_entities.append(execution_entities[0])
		branch_count = len(execution_entities)

		graph = RelationGraphState(
			shared_answer_variable="?answer",
			branches=[BranchState(f"branch_{chr(ord('a') + index)}") for index in range(branch_count)],
			merge_operator="AND",
		)
		all_prompt_history: list[dict[str, Any]] = []
		all_sparql_history: list[dict[str, Any]] = []
		all_step_history: list[dict[str, Any]] = []
		branch_results: list[list[Any]] = []
		branch_traces: list[dict[str, Any]] = []
		branch_failures: list[str] = []
		decomposition = {
			"source": "validated_adaptive_plan" if explicit_plan else "runtime_topic_entities_and_question",
			"shared_answer_variable": "?answer",
			"branch_count": branch_count,
			"shared_root_mode": shared_root_mode,
			"branches": [
				{
					"branch_id": branch.branch_id,
					"root_entity": {"id": entity.id, "name": entity.name},
					"question": question,
					"constraint_summary": (
						(branch_plan[index].get("constraint_summary", "") if explicit_plan else "")
					),
					"constraint_property": (
						(branch_plan[index].get("constraint_property", "") if explicit_plan else "")
					),
					"target_answer_type": (
						(branch_plan[index].get("target_answer_type", requested_final_answer_type) if explicit_plan else requested_final_answer_type)
					),
					"focus_instruction": None if shared_root_mode else "anchor relation search at this root",
				}
				for index, (branch, entity) in enumerate(zip(graph.branches, execution_entities))
			],
		}

		for index, (branch, root_entity) in enumerate(zip(graph.branches, execution_entities)):
			if shared_root_mode and index == 1:
				branch.status = "COMPLETED"
				branch.relation_chain = list(graph.branches[0].relation_chain)
				branch.history.append({"event": "shared_root_reuse", "source_branch": "branch_a"})
				branch_results.append(list(branch_results[0]))
				branch_traces.append(self._branch_trace(branch, root_entity, first_result, [], []))
				continue

			branch_prompt_history: list[dict[str, Any]] = []
			branch_sparql_history: list[dict[str, Any]] = []
			branch_step_history: list[dict[str, Any]] = []
			branch_question = question
			if not shared_root_mode:
				other_entities = [
					f"{entity.name} ({entity.id})"
					for other_index, entity in enumerate(execution_entities)
					if other_index != index
				]
				constraint_summary = ""
				constraint_property = ""
				target_answer_type = requested_final_answer_type or shared_answer_type
				if explicit_plan:
					constraint_summary = str(branch_plan[index].get("constraint_summary", "")).strip()
					constraint_summary = f" Focus on this proposed constraint: {constraint_summary}."
					constraint_property = str(branch_plan[index].get("constraint_property", "")).strip()
					target_answer_type = str(branch_plan[index].get("target_answer_type", target_answer_type)).strip()
				semantic_guidance = ""
				if requested_final_semantic_property or constraint_property:
					semantic_guidance = (
						f" Required semantic property for this branch: {constraint_property or requested_final_semantic_property}."
						f" Requested final answer type: {target_answer_type}."
					)
				branch_question = (
					f"Original question: {question}\n\n"
					f"This is {branch.branch_id} of an explicit {branch_count}-branch AND graph. "
					f"Start from runtime topic entity {root_entity.name} ({root_entity.id}) and "
					"explore only the relation-centric path/constraint anchored at this entity."
					f"{constraint_summary}{semantic_guidance} "
					f"Do not solve the other branches rooted at {', '.join(other_entities)}; "
					f"all branch candidate sets will be intersected for the shared answer type {shared_answer_type or '?answer'}. "
					"Return candidates in the shared answer space."
				)
			try:
				result = self.cor_agent.answer(
					question=branch_question,
					topic_entities=[root_entity],
					prompt_history=branch_prompt_history,
					sparql_history=branch_sparql_history,
					step_history=branch_step_history,
				)
			except Exception as exc:  # preserve the paired experiment row
				logging.exception("Branch-v1 failed for %s", branch.branch_id)
				result = {
					"action": "ERROR",
					"results": [],
					"reasoning_chains": "",
				}
				failure = f"{type(exc).__name__}: {exc}"
			else:
				failure = None

			branch.history.extend(branch_step_history)
			branch.relation_chain = self._relation_chain_from_result(result)
			if result.get("action", "").upper() == "ERROR":
				failure = failure or str(result.get("reasoning_chains", "branch CoR error"))
			if failure:
				branch.status = "FAILED"
				branch_failures.append(f"{branch.branch_id}: {failure}")
			else:
				branch.status = "COMPLETED"
			all_prompt_history.extend(branch_prompt_history)
			all_sparql_history.extend(branch_sparql_history)
			all_step_history.extend(branch_step_history)
			branch_results.append(list(result.get("results", []) or []))
			branch_trace = self._branch_trace(branch, root_entity, result, branch_sparql_history, branch_step_history, failure)
			branch_trace.update({
				"constraint_property": constraint_property if not shared_root_mode else "",
				"constraint_summary": str(branch_plan[index].get("constraint_summary", "")) if explicit_plan else "",
				"target_answer_type": target_answer_type if not shared_root_mode else requested_final_answer_type,
			})
			branch_traces.append(branch_trace)
			if index == 0:
				first_result = result

		if shared_root_mode:
			merged_results = branch_results[0]
			merge_status = "SHARED_ROOT_REUSE"
		elif branch_failures:
			merged_results = []
			merge_status = "FAILED_BRANCH"
		else:
			merged_results, merge_methods = self._merge_candidates(branch_results)
			merge_status = "MERGE_AND"

		if shared_root_mode:
			merge_methods = ["OTHER_SAFE_ALIGNMENT"] * len(merged_results)
		elif branch_failures:
			merge_methods = []
		merge_method_counts = dict(Counter(merge_methods))
		if merge_methods and len(set(merge_methods)) == 1:
			merge_method = merge_methods[0]
		elif merge_methods:
			merge_method = "MIXED"
		else:
			merge_method = "NONE"
		if branch_failures:
			merge_outcome = "FAILED_BRANCH_FALLBACK"
		elif merged_results:
			merge_outcome = "MERGED"
		else:
			merge_outcome = "NO_INTERSECTION_VALID_DISAGREEMENT"

		# Preserve the normal CoR runner contract: callers provide history lists
		# and receive one combined trace for token/query accounting.
		kwargs.get("prompt_history", []).extend(all_prompt_history)
		kwargs.get("sparql_history", []).extend(all_sparql_history)
		kwargs.get("step_history", []).extend(all_step_history)

		return {
			"action": merge_status,
			"question": question,
			"results": merged_results,
			"reasoning_chains": [trace.get("relation_chain", []) for trace in branch_traces],
			"prompt_history": all_prompt_history,
			"sparql_history": all_sparql_history,
			"step_history": all_step_history,
			"branches_created": branch_count,
			"branch_decomposition": decomposition,
			"branch_traces": branch_traces,
			"merge_operator": graph.merge_operator,
			"merge_method": merge_method,
			"merge_methods": merge_methods,
			"merge_method_counts": merge_method_counts,
			"merge_outcome": merge_outcome,
			"merge_inputs": [self._candidate_snapshot(items) for items in branch_results],
			"branch_candidate_counts": [len(items) for items in branch_results],
			"merged_candidate_count": len(merged_results),
			"branch_failures": branch_failures,
			"requested_final_answer_type": requested_final_answer_type,
			"requested_final_semantic_property": requested_final_semantic_property,
			"branch_constraint_properties": [
				str(item.get("constraint_property", "")) for item in (branch_plan or [])
			],
			"branch_property_compatibility": bool(branch_plan) and all(
				str(item.get("constraint_property", "")).strip() for item in (branch_plan or [])
			),
		}
