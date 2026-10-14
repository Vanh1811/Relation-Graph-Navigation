"""Minimal explicit two-branch wrapper around the unchanged CoR agent.

Branch-v1 deliberately reuses CoR's relation search, relation pruning, entity
search, reasoning, and prompts.  Its only research intervention is to execute
two runtime-linked topic roots independently and merge their answer
candidates.  It never consumes gold SPARQL, S-expressions, structure labels,
answer IDs, or a gold branch count.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from chain_of_relations.methods.branch_v1.state import BranchState, RelationGraphState
from chain_of_relations.methods.cor.agent import CoRAgent
from chain_of_relations.schema import Entity


class BranchV1Agent:
	"""Run CoR on two runtime topic roots and merge the final candidates."""

	def __init__(self, cor_agent: CoRAgent):
		self.cor_agent = cor_agent

	@staticmethod
	def _normalise_text(value: Any) -> str:
		return re.sub(r"\s+", " ", str(value or "").strip()).casefold()

	@classmethod
	def _result_key(cls, item: Any) -> tuple[str, str] | None:
		if isinstance(item, dict):
			item_id = str(item.get("id", "")).strip()
			name = str(item.get("name", "")).strip()
		else:
			item_id = ""
			name = str(item or "").strip()
		if item_id:
			return ("id", item_id)
		if name:
			return ("text", cls._normalise_text(name))
		return None

	@classmethod
	def _merge_and(cls, branch_results: list[list[Any]]) -> list[Any]:
		if len(branch_results) != 2:
			raise ValueError("MERGE_AND requires exactly two branch result sets")

		left = {key: item for item in branch_results[0] if (key := cls._result_key(item))}
		right = {key: item for item in branch_results[1] if (key := cls._result_key(item))}
		shared_keys = set(left).intersection(right)

		merged: list[Any] = []
		seen: set[tuple[str, str]] = set()
		for item in branch_results[0] + branch_results[1]:
			key = cls._result_key(item)
			if key in shared_keys and key not in seen:
				seen.add(key)
				merged.append(left.get(key, item))
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
		failure: str | None = None,
	) -> dict[str, Any]:
		return {
			"branch_id": branch.branch_id,
			"topic_entity": {"id": topic_entity.id, "name": topic_entity.name},
			"relation_chain": list(branch.relation_chain),
			"status": branch.status,
			"candidate_count": len(result.get("results", []) or []),
			"candidate_ids": [
				str(item.get("id", ""))
				for item in result.get("results", []) or []
				if isinstance(item, dict) and str(item.get("id", "")).strip()
			],
			"failure": failure,
			"action": result.get("action", ""),
			"history": list(branch.history),
		}

	def answer(self, question: str, topic_entities: list[Entity], **kwargs: Any) -> dict[str, Any]:
		if not topic_entities:
			return {
				"action": "ERROR",
				"question": question,
				"results": [],
				"reasoning_chains": "No topic_entities provided",
				"branch_failures": ["no topic entities"],
			}

		# Runtime entity linking is already part of the inference-safe input.  A
		# second runtime topic entity gives v1 a natural independent branch root.
		# One-topic linear questions reuse the unchanged CoR execution once and
		# record a completed shared-root branch so they are not artificially
		# penalised for lacking a second linked root.
		execution_entities = list(topic_entities[:2])
		shared_root_mode = len(execution_entities) == 1
		if shared_root_mode:
			execution_entities.append(execution_entities[0])

		graph = RelationGraphState(
			shared_answer_variable="?answer",
			branches=[BranchState("branch_a"), BranchState("branch_b")],
			merge_operator="AND",
		)
		all_prompt_history: list[dict[str, Any]] = []
		all_sparql_history: list[dict[str, Any]] = []
		all_step_history: list[dict[str, Any]] = []
		branch_results: list[list[Any]] = []
		branch_traces: list[dict[str, Any]] = []
		branch_failures: list[str] = []
		decomposition = {
			"source": "runtime_topic_entities_and_question",
			"shared_answer_variable": "?answer",
			"branch_count": 2,
			"shared_root_mode": shared_root_mode,
			"branches": [
				{
					"branch_id": "branch_a",
					"root_entity": {"id": execution_entities[0].id, "name": execution_entities[0].name},
					"question": question,
					"focus_instruction": None if shared_root_mode else "anchor relation search at this root",
				},
				{
					"branch_id": "branch_b",
					"root_entity": {"id": execution_entities[1].id, "name": execution_entities[1].name},
					"question": question,
					"focus_instruction": None if shared_root_mode else "anchor relation search at this root",
				},
			],
		}

		for index, (branch, root_entity) in enumerate(zip(graph.branches, execution_entities)):
			if shared_root_mode and index == 1:
				branch.status = "COMPLETED"
				branch.relation_chain = list(graph.branches[0].relation_chain)
				branch.history.append({"event": "shared_root_reuse", "source_branch": "branch_a"})
				branch_results.append(list(branch_results[0]))
				branch_traces.append(self._branch_trace(branch, root_entity, first_result))
				continue

			branch_prompt_history: list[dict[str, Any]] = []
			branch_sparql_history: list[dict[str, Any]] = []
			branch_step_history: list[dict[str, Any]] = []
			branch_question = question
			if not shared_root_mode:
				other_entity = execution_entities[1 - index]
				branch_question = (
					f"Original question: {question}\n\n"
					f"This is {branch.branch_id} of an explicit two-branch AND graph. "
					f"Start from runtime topic entity {root_entity.name} ({root_entity.id}) and "
					"explore only the relation-centric path/constraint anchored at this entity. "
					f"Do not solve the other branch rooted at {other_entity.name} ({other_entity.id}); "
					"the two branch answer candidate sets will be intersected after execution. "
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
			branch_traces.append(self._branch_trace(branch, root_entity, result, failure))
			if index == 0:
				first_result = result

		if shared_root_mode:
			merged_results = branch_results[0]
			merge_status = "SHARED_ROOT_REUSE"
		elif branch_failures:
			merged_results = []
			merge_status = "FAILED_BRANCH"
		else:
			merged_results = self._merge_and(branch_results)
			merge_status = "MERGE_AND"

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
			"branches_created": 2,
			"branch_decomposition": decomposition,
			"branch_traces": branch_traces,
			"merge_operator": graph.merge_operator,
			"branch_candidate_counts": [len(items) for items in branch_results],
			"merged_candidate_count": len(merged_results),
			"branch_failures": branch_failures,
		}
