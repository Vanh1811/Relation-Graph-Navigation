#!/usr/bin/env python3
"""Diagnostic answer-equivalent path audit for the four Phase 2.2 misses.

This script is deliberately outside the runtime reasoning pipeline.  It
replays the exact recorded current entities, executes the live raw/nearly-raw
one-hop neighborhood, and performs a bounded two-hop query only when no
one-hop relation reaches a gold answer.  It never changes routing, ranking,
candidate discovery, prompts, or evaluation behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from _bootstrap import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from chain_of_relations.eval.accuracy import normalize


PHASE21_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase2_1" / "relation_grounding"
PHASE22_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase2_2" / "relation_candidate_recall"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase2_2b" / "answer_equivalent_path_audit"
SUBSET_ROOT = PROJECT_ROOT / "outputs" / "subsets"

FREEBASE_PREFIX = "http://rdf.freebase.com/ns/"
RDF_TYPE_RELATION = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
ONE_HOP_RESULT_LIMIT = 1000
TWO_HOP_PATH_LIMIT = 10000


SEMANTIC_ASSESSMENTS: dict[str, dict[str, Any]] = {
	"4301990001000": {
		"value": "UNCLEAR",
		"reason": "music.recording.releases reaches a musical release from the recording, but answer agreement alone does not prove that it expresses the gold producers relation.",
	},
	"4302233001000": {
		"value": "NO",
		"reason": "education.field_of_study.subdisciplines is a different relation from the gold research_areas relation; it reaches a candidate answer from the recorded constraint entity.",
	},
	"WebQTrn-1812_8126be584feae3f358a930823db22c4e": {
		"value": "NO",
		"reason": "location.location.contains identifies countries in the region, but it does not express the gold GDP-deflator relation.",
	},
	"WebQTest-743_af8b975343e4098913128e9ac7d04c05": {
		"value": "NO",
		"reason": "Any answer-reaching alternative from Anti-Apartheid would not by itself establish the gold public-speaker/sibling reasoning relation.",
	},
}


def read_json(path: Path) -> Any:
	return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def normalize_relation(value: str) -> str:
	return str(value or "").replace(FREEBASE_PREFIX, "").strip()


def is_entity_id(value: str) -> bool:
	return str(value or "").startswith(("m.", "g."))


def relation_term(relation: str) -> str:
	text = str(relation or "").strip()
	if text.startswith(("http://", "https://")):
		return f"<{text}>"
	return f"ns:{text}"


def entity_term(entity_id: str) -> str:
	return f"ns:{entity_id}"


def value_key(value: dict[str, Any], gold_by_id: dict[str, str], gold_names: dict[str, str]) -> str:
	"""Create a stable identity key for set comparison.

	Entity IDs are preferred. Literals are compared with the repository's
	normalization. A literal that exactly matches one gold name is mapped to its
	gold entity identity for diagnostics.
	"""
	raw = str(value.get("raw_value", "")).strip()
	value_type = str(value.get("value_type", ""))
	entity_id = str(value.get("entity_id", "")).strip()
	if entity_id:
		return f"entity:{entity_id}"
	normalized = normalize(raw)
	for gold_id, gold_name in gold_by_id.items():
		if normalized == normalize(gold_name):
			return f"entity:{gold_id}"
	return f"{value_type or 'value'}:{normalized}"


def parse_value(binding: dict[str, Any]) -> dict[str, Any]:
	raw = str(binding.get("value", "")).strip()
	value_type = str(binding.get("type", ""))
	entity_id = ""
	if value_type == "uri" and raw.startswith(FREEBASE_PREFIX):
		candidate = raw[len(FREEBASE_PREFIX):]
		if is_entity_id(candidate):
			entity_id = candidate
	return {
		"raw_value": raw,
		"value_type": value_type,
		"entity_id": entity_id,
		"language": binding.get("xml:lang", ""),
		"datatype": binding.get("datatype", ""),
	}


def answer_metrics(
	values: list[dict[str, Any]],
	gold_ids: list[str],
	gold_names: list[str],
) -> dict[str, Any]:
	gold_by_id = {str(gold_id): str(name) for gold_id, name in zip(gold_ids, gold_names)}
	gold_keys = {f"entity:{gold_id}" for gold_id in gold_ids if gold_id}
	if not gold_keys:
		gold_keys = {f"literal:{normalize(name)}" for name in gold_names if normalize(name)}
	returned_keys = {value_key(value, gold_by_id, {normalize(name): name for name in gold_names}) for value in values}
	overlap = returned_keys.intersection(gold_keys)
	return {
		"gold_answer_count": len(gold_keys),
		"returned_answer_count": len(returned_keys),
		"gold_overlap_count": len(overlap),
		"gold_recall": len(overlap) / len(gold_keys) if gold_keys else None,
		"exact_answer_set_match": bool(gold_keys) and returned_keys == gold_keys,
		"returned_identity_keys": sorted(returned_keys),
		"gold_identity_keys": sorted(gold_keys),
	}


def query_endpoint(endpoint: str, sparql: str, timeout: int) -> dict[str, Any]:
	started = time.time()
	url = endpoint + "?" + urllib.parse.urlencode({"query": sparql, "format": "json"})
	request = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json"})
	try:
		with urllib.request.urlopen(request, timeout=timeout) as response:
			payload = json.load(response)
		return {
			"status": "ok",
			"timed_out": False,
			"rows": payload.get("results", {}).get("bindings", []) or [],
			"elapsed_seconds": round(time.time() - started, 3),
		}
	except Exception as exc:  # pragma: no cover - endpoint-dependent
		message = str(exc)
		return {
			"status": "timeout" if "timed out" in message.lower() else "error",
			"timed_out": "timed out" in message.lower(),
			"rows": [],
			"error": message,
			"elapsed_seconds": round(time.time() - started, 3),
		}


def freebase_relation_is_excluded(relation: str) -> bool:
	relation = normalize_relation(relation)
	return (
		relation in {RDF_TYPE_RELATION, "type.object.type", "type.object.name"}
		or relation.startswith("common.")
		or relation.startswith("freebase.")
		or "sameAs" in relation
	)


def load_gold_record(dataset: str, case_id: str) -> dict[str, Any] | None:
	path = SUBSET_ROOT / f"{dataset}.jsonl"
	if not path.exists():
		return None
	with path.open(encoding="utf-8") as handle:
		for line in handle:
			row = json.loads(line)
			if str(row.get("id")) == case_id:
				return row
	return None


def load_cases() -> list[dict[str, Any]]:
	traces = read_jsonl(PHASE22_ROOT / "candidate_pipeline_traces.jsonl")
	missing = [trace for trace in traces if trace.get("failure_type") == "RELATION_CANDIDATE_MISSING"]
	result: list[dict[str, Any]] = []
	for trace in missing:
		case_id = str(trace["question_id"])
		gold = load_gold_record(str(trace["dataset"]), case_id)
		if gold is None:
			result.append({"trace": trace, "gold_status": "INSUFFICIENT_GOLD_ANSWER_EVIDENCE", "gold_reason": "Subset record was not found."})
			continue
		gold_ids = [str(value) for value in gold.get("answer_ids", []) if str(value)]
		gold_names = [str(value) for value in gold.get("answers", []) if str(value)]
		if len(gold_ids) != len(gold_names) or not gold_names:
			result.append({
				"trace": trace,
				"gold_status": "INSUFFICIENT_GOLD_ANSWER_EVIDENCE",
				"gold_reason": "Gold answer IDs and names were missing or not aligned in the subset record.",
			})
			continue
		result.append({
			"trace": trace,
			"gold_status": "OK",
			"gold_reason": "Gold answer IDs and names loaded from the frozen dataset subset.",
			"gold": {
				"ids": gold_ids,
				"names": gold_names,
				"question": gold.get("question", ""),
				"answer_representation": "Freebase entity IDs plus answer labels",
			},
		})
	return result


def eligible_relations(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
	raw = trace.get("all_raw_neighbor_relations_from_KG") or {}
	eligible: list[dict[str, Any]] = []
	excluded: list[dict[str, str]] = []
	relation_sources: dict[str, dict[str, Any]] = {}
	for direction in ("outgoing", "incoming"):
		for item in raw.get(direction, []) or []:
			relation = normalize_relation(item.get("relation", ""))
			entry = {"relation": relation, "direction": direction, "raw_uri": item.get("raw_uri", "")}
			if freebase_relation_is_excluded(relation):
				excluded.append(entry)
				continue
			source = relation_sources.setdefault(
				relation,
				{"raw_uri": item.get("raw_uri", ""), "raw_directions": []},
			)
			if direction not in source["raw_directions"]:
				source["raw_directions"].append(direction)
	for relation, source in relation_sources.items():
		for direction in ("outgoing", "incoming"):
			eligible.append({
				"relation": relation,
				"direction": direction,
				"raw_uri": source["raw_uri"],
				"raw_directions": source["raw_directions"],
			})
	return eligible, {
		"raw_relation_count": sum(len(raw.get(direction, []) or []) for direction in ("outgoing", "incoming")),
		"eligible_relation_count": len(eligible),
		"excluded_relation_count": len(excluded),
		"excluded_relations": excluded,
		"direction_rule": "Every schema-eligible relation identity was executed once outgoing and once incoming, regardless of the direction in which it appeared in the Phase 2.2 raw neighborhood.",
		"exclusion_rule": "Frozen Freebase abandon_rels plus RDF schema type: rdf:type, type.object.type, type.object.name, common.*, freebase.*, and sameAs.",
	}


def one_hop_query(entity_id: str, relation: str, direction: str) -> str:
	term = relation_term(relation)
	entity = entity_term(entity_id)
	pattern = f"{entity} {term} ?value ." if direction == "outgoing" else f"?value {term} {entity} ."
	return (
		"PREFIX ns: <http://rdf.freebase.com/ns/>\n"
		"SELECT DISTINCT ?value\n"
		"WHERE {\n"
		f"  {pattern}\n"
		f"}} LIMIT {ONE_HOP_RESULT_LIMIT}\n"
	)


def execute_one_hop(
	case: dict[str, Any],
	entry: dict[str, str],
	endpoint: str,
	timeout: int,
) -> dict[str, Any]:
	trace = case["trace"]
	gold = case["gold"]
	query = one_hop_query(trace["current_entity"]["id"], entry["relation"], entry["direction"])
	result = query_endpoint(endpoint, query, timeout)
	values = [parse_value(row.get("value", {})) for row in result.get("rows", [])]
	metrics = answer_metrics(values, gold["ids"], gold["names"])
	returned_answers = []
	gold_names_by_id = dict(zip(gold["ids"], gold["names"]))
	for value in values:
		item = dict(value)
		if value.get("entity_id") in gold_names_by_id:
			item["display_name"] = gold_names_by_id[value["entity_id"]]
		returned_answers.append(item)
	return {
		"case_id": trace["question_id"],
		"dataset": trace["dataset"],
		"question": gold["question"],
		"current_entity": trace["current_entity"],
		"relation": entry["relation"],
		"direction": entry["direction"],
		"query": query,
		"status": result.get("status"),
		"timed_out": result.get("timed_out", False),
		"error": result.get("error", ""),
		"elapsed_seconds": result.get("elapsed_seconds", 0.0),
		"result_count": len(values),
		"result_limit": ONE_HOP_RESULT_LIMIT,
		"result_truncated": len(values) >= ONE_HOP_RESULT_LIMIT,
		"returned_answers": returned_answers,
		**metrics,
		"answer_equivalent": metrics["gold_overlap_count"] > 0,
		"semantic_equivalent": "UNCLEAR",
		"semantic_interpretation": "Execution overlap is evidence of answer reachability only; semantic equivalence is adjudicated separately.",
	}


def classify_one_hop(rows: list[dict[str, Any]]) -> str:
	if any(row.get("exact_answer_set_match") for row in rows):
		return "ONE_HOP_EXACT_EQUIVALENT"
	if any(row.get("gold_overlap_count", 0) > 0 for row in rows):
		return "ONE_HOP_PARTIAL_GOLD_REACHABLE"
	return "NO_ONE_HOP_GOLD_REACHABILITY"


def values_clause(variable: str, values: list[str]) -> str:
	return " ".join(f"{variable}:{value}" if not value.startswith(("http://", "https://")) else f"<{value}>" for value in values)


def two_hop_query(entity_id: str, first_relations: list[dict[str, str]], gold_ids: list[str]) -> str:
	relations = sorted({item["relation"] for item in first_relations})
	first_values = " ".join(relation_term(relation) for relation in relations)
	gold_values = " ".join(entity_term(gold_id) for gold_id in gold_ids)
	entity = entity_term(entity_id)
	return f"""PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?r1 ?mid ?r2 ?gold ?direction1 ?direction2
WHERE {{
  VALUES ?r1 {{ {first_values} }}
  VALUES ?gold {{ {gold_values} }}
  {{
    {entity} ?r1 ?mid .
    {{ ?mid ?r2 ?gold . BIND("outgoing" AS ?direction2) }}
    UNION
    {{ ?gold ?r2 ?mid . BIND("incoming" AS ?direction2) }}
    BIND("outgoing" AS ?direction1)
  }}
  UNION
  {{
    ?mid ?r1 {entity} .
    {{ ?mid ?r2 ?gold . BIND("outgoing" AS ?direction2) }}
    UNION
    {{ ?gold ?r2 ?mid . BIND("incoming" AS ?direction2) }}
    BIND("incoming" AS ?direction1)
  }}
}}
LIMIT {TWO_HOP_PATH_LIMIT}
"""


def parse_relation_binding(binding: dict[str, Any]) -> str:
	return normalize_relation(binding.get("value", ""))


def parse_entity_binding(binding: dict[str, Any]) -> dict[str, Any]:
	value = parse_value(binding)
	return value


def inspect_intermediate_types(endpoint: str, intermediate_ids: list[str], timeout: int) -> dict[str, list[str]]:
	ids = sorted({value for value in intermediate_ids if is_entity_id(value)})
	if not ids:
		return {}
	query = (
		"PREFIX ns: <http://rdf.freebase.com/ns/>\n"
		"SELECT DISTINCT ?mid ?type WHERE {\n"
		f"VALUES ?mid {{ {' '.join(entity_term(value) for value in ids)} }}\n"
		"?mid ns:type.object.type ?type .\n"
		"} LIMIT 1000\n"
	)
	result = query_endpoint(endpoint, query, timeout)
	types: dict[str, list[str]] = defaultdict(list)
	for row in result.get("rows", []):
		mid = normalize_relation(row.get("mid", {}).get("value", ""))
		type_id = normalize_relation(row.get("type", {}).get("value", ""))
		if mid and type_id and type_id not in types[mid]:
			types[mid].append(type_id)
	return dict(types)


def possible_mediator(types: list[str]) -> bool:
	# This is intentionally a weak diagnostic flag, not a semantic assertion.
	return any(type_id.startswith("base.") and type_id != "base.type_ontology.topic" for type_id in types)


def run_two_hop(case: dict[str, Any], endpoint: str, timeout: int, first_relations: list[dict[str, str]]) -> dict[str, Any]:
	trace = case["trace"]
	gold = case["gold"]
	query = two_hop_query(trace["current_entity"]["id"], first_relations, gold["ids"])
	result = query_endpoint(endpoint, query, timeout)
	raw_paths: list[dict[str, Any]] = []
	for row in result.get("rows", []):
		gold_id = normalize_relation(row.get("gold", {}).get("value", ""))
		mid = parse_entity_binding(row.get("mid", {}))
		r1 = parse_relation_binding(row.get("r1", {}))
		r2 = parse_relation_binding(row.get("r2", {}))
		d1 = str(row.get("direction1", {}).get("value", ""))
		d2 = str(row.get("direction2", {}).get("value", ""))
		if not r1 or not r2 or freebase_relation_is_excluded(r2) or gold_id not in gold["ids"]:
			continue
		raw_paths.append({
			"case_id": trace["question_id"],
			"dataset": trace["dataset"],
			"current_entity": trace["current_entity"],
			"relation_1": r1,
			"direction_1": d1,
			"intermediate_entity": mid,
			"relation_2": r2,
			"direction_2": d2,
			"gold_entity_id": gold_id,
			"returned_result": gold["names"][gold["ids"].index(gold_id)],
		})

	intermediate_types = inspect_intermediate_types(
		endpoint,
		[str(path["intermediate_entity"].get("entity_id", "")) for path in raw_paths],
		timeout,
	)
	grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
	for path in raw_paths:
		mid = path["intermediate_entity"]
		mid_id = str(mid.get("entity_id", ""))
		key = (
			path["relation_1"], path["direction_1"], mid_id or str(mid.get("raw_value", "")),
			path["relation_2"], path["direction_2"],
		)
		item = grouped.setdefault(key, {**path, "gold_entity_ids": [], "gold_answer_names": []})
		if path["gold_entity_id"] not in item["gold_entity_ids"]:
			item["gold_entity_ids"].append(path["gold_entity_id"])
			item["gold_answer_names"].append(path["returned_result"])

	rows: list[dict[str, Any]] = []
	for item in grouped.values():
		matched = set(item["gold_entity_ids"])
		gold_set = set(gold["ids"])
		types = intermediate_types.get(str(item["intermediate_entity"].get("entity_id", "")), [])
		rows.append({
			**item,
			"result_count": len(item["gold_entity_ids"]),
			"gold_answer_count": len(gold_set),
			"gold_overlap_count": len(matched),
			"gold_recall": len(matched) / len(gold_set) if gold_set else None,
			"exact_answer_set_match": matched == gold_set,
			"answer_equivalent": bool(matched),
			"intermediate_entity_label": "",
			"intermediate_types": types,
			"possible_mediator_path": possible_mediator(types),
			"mediator_cvt_indication": "possible_from_base_type" if possible_mediator(types) else "not_indicated",
			"semantic_equivalent": "UNCLEAR",
			"semantic_interpretation": "A gold-reaching two-hop path does not by itself establish semantic equivalence.",
		})
	return {
		"case_id": trace["question_id"],
		"dataset": trace["dataset"],
		"current_entity": trace["current_entity"],
		"query": query,
		"status": result.get("status"),
		"timed_out": result.get("timed_out", False),
		"error": result.get("error", ""),
		"elapsed_seconds": result.get("elapsed_seconds", 0.0),
		"first_hop_relations_considered": len(first_relations),
		"raw_path_rows": len(result.get("rows", [])),
		"eligible_gold_reaching_path_groups": len(rows),
		"path_limit": TWO_HOP_PATH_LIMIT,
		"search_truncated": len(result.get("rows", [])) >= TWO_HOP_PATH_LIMIT,
		"bounded_strategy": "All Phase 2.2 schema-eligible first-hop relations; exact frozen gold entity IDs; both directions for each edge; second-hop obvious Freebase schema exclusions applied after retrieval.",
		"paths": rows,
	}


def semantic_for_winner(case_id: str, relation: str) -> dict[str, str]:
	assessment = SEMANTIC_ASSESSMENTS.get(case_id)
	if assessment:
		return {"semantic_equivalent": assessment["value"], "semantic_interpretation": assessment["reason"]}
	return {"semantic_equivalent": "UNCLEAR", "semantic_interpretation": "No automatic semantic equivalence inference was applied."}


def final_case_diagnosis(case_id: str, one_class: str, one_rows: list[dict[str, Any]], two: dict[str, Any] | None) -> tuple[str, str]:
	winners = [row for row in one_rows if row.get("gold_overlap_count", 0) > 0]
	if winners:
		if any(semantic_for_winner(case_id, row["relation"])["semantic_equivalent"] == "YES" for row in winners):
			return "ALT_ONE_HOP_SEMANTIC_EQUIVALENT", "A one-hop relation reaches gold, and the conservative semantic assessment marks it equivalent."
		return "ALT_ONE_HOP_ANSWER_ONLY", "A one-hop relation reaches gold answers, but semantic equivalence is not established."
	if one_class != "NO_ONE_HOP_GOLD_REACHABILITY":
		return "INSUFFICIENT_EVIDENCE", "One-hop classification was inconsistent with its execution rows."
	if not two or two.get("status") != "ok" or two.get("search_truncated"):
		return "INSUFFICIENT_EVIDENCE", "The bounded two-hop query did not complete without truncation."
	paths = two.get("paths", [])
	if any(path.get("exact_answer_set_match") for path in paths):
		if any(path.get("semantic_equivalent") == "YES" for path in paths):
			return "ALT_TWO_HOP_SEMANTIC_EQUIVALENT", "A bounded two-hop path reproduces the complete gold answer set and has a separate positive semantic assessment."
		return "ALT_TWO_HOP_ANSWER_ONLY", "A bounded two-hop path reproduces the complete gold answer set, but semantic equivalence is not established."
	if any(path.get("gold_overlap_count", 0) > 0 for path in paths):
		return "ALT_TWO_HOP_ANSWER_ONLY", "A bounded two-hop path reaches part of the gold answer set, but semantic equivalence is not established."
	return "NO_SHORT_PATH_FROM_RECORDED_STATE", "No gold-reaching path was found in the completed bounded one-/two-hop audit."


def write_report(cases: list[dict[str, Any]], one_rows: list[dict[str, Any]], two_results: dict[str, dict[str, Any]], summaries: list[dict[str, Any]], runtime: float, query_count: int) -> str:
	one_exact = sum(summary["one_hop_classification"] == "ONE_HOP_EXACT_EQUIVALENT" for summary in summaries)
	one_partial = sum(summary["one_hop_classification"] == "ONE_HOP_PARTIAL_GOLD_REACHABLE" for summary in summaries)
	two_exact = sum(summary.get("two_hop_classification") == "TWO_HOP_EXACT_EQUIVALENT" for summary in summaries)
	two_partial = sum(summary.get("two_hop_classification") == "TWO_HOP_PARTIAL_GOLD_REACHABLE" for summary in summaries)
	no_short = sum(summary["final_diagnosis"] == "NO_SHORT_PATH_FROM_RECORDED_STATE" for summary in summaries)
	insufficient = sum(summary["final_diagnosis"] == "INSUFFICIENT_EVIDENCE" for summary in summaries)
	one_reachable = sum(summary["one_hop_answer_reachable"] for summary in summaries)
	within_two = sum(summary["answer_reachable_within_two_hops"] for summary in summaries)
	finals = Counter(summary["final_diagnosis"] for summary in summaries)
	if finals.get("ALT_ONE_HOP_SEMANTIC_EQUIVALENT") or finals.get("ALT_TWO_HOP_SEMANTIC_EQUIVALENT"):
		overall = "ANSWER_EQUIVALENT_ALTERNATIVE_PATH_SIGNAL"
	elif finals.get("ALT_ONE_HOP_ANSWER_ONLY") or finals.get("ALT_TWO_HOP_ANSWER_ONLY"):
		overall = "ANSWER_REACHABLE_BUT_REASONING_MISMATCH"
	elif insufficient:
		overall = "INCONCLUSIVE_ANSWER_REACHABILITY"
	else:
		overall = "NO_SHORT_PATH_FROM_RECORDED_STATE"
	lines = [
		"# Phase 2.2b report: answer-equivalent relation / short-path audit",
		"",
		"## Research question",
		"",
		"For each Phase 2.2 candidate-missing case, can the gold answer still be reached from the exact recorded current entity through another available relation or a bounded two-hop path?",
		"",
		"## Why Phase 2.2 was insufficient",
		"",
		"Phase 2.2 measured exact relation availability. Phase 2.2b measures executed gold-answer reachability through alternative relations and short paths.",
		"",
		"## Frozen components",
		"",
		"Adaptive v1.1.3 routing, entity lookup, Freebase execution semantics, branch handling, evaluator normalization, Phase 2.1 semantic scoring/ranking, Phase 2.2 filters, top-k settings, prompts, and runtime inference were unchanged. No full benchmark or production intervention was run.",
		"",
		"## Method",
		"",
	"The exact recorded entity was used. Every Phase 2.2 schema-eligible raw relation identity was executed once outgoing and once incoming with a 1,000-result query bound. The existing `normalize` function was reused; entity IDs were matched exactly and literals/labels were normalized conservatively. Two-hop search ran only for cases with no one-hop gold overlap, using all eligible first-hop relation identities, both directions, exact frozen gold entity IDs, and a 10,000-path result bound.",
		"",
		"## Aggregate results",
		"",
		"| Metric | Count |",
		"|---|---:|",
		f"| Audited cases | {len(summaries)} |",
		"| Exact gold-relation availability from Phase 2.2 | 0/4 |",
		f"| One-hop exact answer-equivalent cases | {one_exact}/4 |",
		f"| One-hop partial gold-reachable cases | {one_partial}/4 |",
		f"| Two-hop exact answer-equivalent cases | {two_exact}/4 |",
		f"| Two-hop partial gold-reachable cases | {two_partial}/4 |",
		f"| Gold answer reachable at <=1 hop | {one_reachable}/4 |",
		f"| Gold answer reachable at <=2 hops | {within_two}/4 |",
		f"| No-short-path cases | {no_short}/4 |",
		f"| Insufficient-evidence cases | {insufficient}/4 |",
		f"| SPARQL queries | {query_count} |",
		f"| Total reported runtime | {runtime:.2f} seconds |",
		"",
		"## Required case table",
		"",
		"| Case | Current entity | Gold relation | Gold relation present? | Alternative 1-hop reaches gold? | Alternative <=2-hop reaches gold? | Answer-equivalent path | Semantic equivalence | Final diagnosis |",
		"|---|---|---|---:|---:|---:|---|---|---|",
	]
	for summary in summaries:
		lines.append(
			f"| `{summary['case_id']}` | `{summary['current_entity']['name']}` (`{summary['current_entity']['id']}`) | `{summary['expected_relation']}` / {summary['expected_direction']} | no | "
			f"{'yes' if summary['one_hop_answer_reachable'] else 'no'} | "
			f"{'yes' if summary['answer_reachable_within_two_hops'] else 'no'} | "
			f"{summary['answer_equivalent_path']} | {summary['semantic_equivalence']} | {summary['final_diagnosis']} |"
		)
	lines += [
		"",
		"## Per-case findings",
		"",
	]
	for summary in summaries:
		lines += [
			f"### {summary['case_id']}",
			"",
			f"- Dataset: `{summary['dataset']}`",
			f"- Question: {summary['question']}",
			f"- Recorded current entity: `{summary['current_entity']['name']}` (`{summary['current_entity']['id']}`)",
			f"- Recorded hop/state: step `{summary['step_index']}`, Phase 2.2 source trace `{summary['source_trace_index']}`",
			f"- Gold answer(s): {', '.join(summary['gold_names'])}",
			f"- Expected relation: `{summary['expected_relation']}` / {summary['expected_direction']}",
			f"- Phase 2.2 raw outgoing/incoming counts: {summary['raw_outgoing_count']} / {summary['raw_incoming_count']}",
			f"- Phase 2.2b one-hop relations executed: {summary['one_hop_relations_executed']}",
			f"- Case runtime: {summary['case_runtime_seconds']:.2f} seconds",
			f"- One-hop relation(s) reaching gold: {', '.join(summary['one_hop_winners']) or 'none'}",
			f"- One-hop classification: `{summary['one_hop_classification']}`",
			f"- Two-hop paths explored: {summary['two_hop_paths_explored']}",
			f"- Gold-reaching two-hop paths: {summary['two_hop_gold_path_groups']}",
			f"- Mediator/CVT evidence: {summary['mediator_cvt_evidence']}",
			f"- Final diagnosis: `{summary['final_diagnosis']}` — {summary['diagnosis_reason']}",
			"",
		]
	lines += [
		"## Interpretation",
		"",
		"An answer-equivalent result is not automatically a reasoning-equivalent result. The audit reports answer overlap separately from semantic equivalence; semantic labels were not inferred from relation-name similarity or answer overlap alone.",
		"",
		f"## Decision\n\n**{overall}**",
		"",
		"No runtime code was changed. The evidence is limited to reachability from the recorded state and does not prove that the answer is unreachable elsewhere in Freebase.",
		"",
		"## Next evidence required",
		"",
		"If the cases are answer-reachable only through semantically different relations, investigate actual-versus-gold hop/entity-state divergence before considering any runtime intervention. Do not add aliases, widen top-k, or force two-hop expansion from this audit alone.",
	]
	(OUTPUT_ROOT / "PHASE2_2B_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
	analysis_lines = [
		"# Phase 2.2b answer-equivalent analysis",
		"",
		"This companion analysis separates executable answer reachability from semantic equivalence. A relation/path is marked answer-equivalent when its returned Freebase entity IDs overlap the frozen gold IDs; semantic equivalence remains `YES`, `NO`, or `UNCLEAR` and is never inferred from overlap alone.",
		"",
		"| Case | One-hop classification | Two-hop classification | Answer-equivalent path | Semantic equivalence | Final diagnosis |",
		"|---|---|---|---|---|---|",
	]
	for summary in summaries:
		analysis_lines.append(
			f"| `{summary['case_id']}` | `{summary['one_hop_classification']}` | `{summary['two_hop_classification']}` | "
			f"{summary['answer_equivalent_path']} | {summary['semantic_equivalence']} | `{summary['final_diagnosis']}` |"
		)
	analysis_lines += ["", "## Evidence by case", ""]
	for summary in summaries:
		analysis_lines += [
			f"### `{summary['case_id']}`",
			"",
			f"- Current state: `{summary['current_entity']['name']}` (`{summary['current_entity']['id']}`)",
			f"- Gold answers: {', '.join(summary.get('gold_names', [])) or 'unavailable'}",
			f"- One-hop winners: {', '.join(summary.get('one_hop_winners', [])) or 'none'}",
			f"- Two-hop gold-reaching path groups: {summary.get('two_hop_gold_path_groups', 0)}",
			f"- Semantic assessment: {summary['semantic_equivalence']}",
			f"- Diagnosis: `{summary['final_diagnosis']}` — {summary['diagnosis_reason']}",
			"",
		]
	analysis_lines += [
		"## Interpretation",
		"",
		"The audit does not establish that an alternative path is a valid substitute for the expected relation. It only establishes whether the frozen gold answer IDs are reachable under the bounded diagnostic query. The production CoR pipeline remains unchanged.",
	]
	(OUTPUT_ROOT / "answer_equivalent_analysis.md").write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")
	return overall


def run() -> None:
	global OUTPUT_ROOT
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--endpoint", default=os.getenv("FREEBASE_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql"))
	parser.add_argument("--timeout", type=int, default=180)
	parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
	args = parser.parse_args()
	OUTPUT_ROOT = args.output_root
	started = time.time()
	query_count = 0
	cases = load_cases()
	one_rows: list[dict[str, Any]] = []
	two_results: dict[str, dict[str, Any]] = {}
	summaries: list[dict[str, Any]] = []

	for index, case in enumerate(cases, start=1):
		case_started = time.time()
		trace = case["trace"]
		case_id = str(trace["question_id"])
		if case.get("gold_status") != "OK":
			summaries.append({
				"case_id": case_id,
				"dataset": trace["dataset"],
				"question": "",
				"current_entity": trace["current_entity"],
				"expected_relation": trace["expected_relation"],
				"expected_direction": trace["expected_direction"],
				"step_index": trace.get("step_index"),
				"source_trace_index": trace.get("relation_search", {}).get("trace_index"),
				"gold_names": [],
				"one_hop_classification": "INSUFFICIENT_GOLD_ANSWER_EVIDENCE",
				"one_hop_answer_reachable": False,
				"answer_reachable_within_two_hops": False,
				"answer_equivalent_path": "none",
				"semantic_equivalence": "UNCLEAR",
				"final_diagnosis": "INSUFFICIENT_EVIDENCE",
				"diagnosis_reason": case.get("gold_reason", "Gold evidence unavailable."),
				"case_runtime_seconds": round(time.time() - case_started, 3),
			})
			continue
		eligible, exclusions = eligible_relations(trace)
		print(f"CASE {index}/{len(cases)} {trace['dataset']}:{case_id} one-hop relations={len(eligible)}", flush=True)
		case["eligible"] = eligible
		case["exclusions"] = exclusions
		case_one_rows: list[dict[str, Any]] = []
		for relation_index, entry in enumerate(eligible, start=1):
			row = execute_one_hop(case, entry, args.endpoint, max(1, int(args.timeout)))
			case_one_rows.append(row)
			one_rows.append(row)
			query_count += 1
			if relation_index % 50 == 0:
				print(f"  one-hop {relation_index}/{len(eligible)}", flush=True)
		one_class = classify_one_hop(case_one_rows)
		winners = [row for row in case_one_rows if row.get("gold_overlap_count", 0) > 0]
		for winner in winners:
			winner.update(semantic_for_winner(case_id, winner["relation"]))
		two_result: dict[str, Any] | None = None
		if one_class == "NO_ONE_HOP_GOLD_REACHABILITY":
			print(f"  two-hop search first_relations={len(eligible)}", flush=True)
			two_result = run_two_hop(case, args.endpoint, max(1, int(args.timeout)), eligible)
			two_results[case_id] = two_result
			query_count += 1
			if two_result.get("paths"):
				query_count += 1  # mediator/type inspection query
			for path in two_result.get("paths", []):
				path.update(semantic_for_winner(case_id, path.get("relation_1", "")))
		else:
			two_results[case_id] = {"status": "NOT_REQUIRED", "paths": [], "raw_path_rows": 0, "eligible_gold_reaching_path_groups": 0, "elapsed_seconds": 0.0}
		final_diagnosis, reason = final_case_diagnosis(case_id, one_class, case_one_rows, two_result)
		answer_reachable = bool(winners)
		two_reachable = answer_reachable or bool(two_result and any(path.get("gold_overlap_count", 0) > 0 for path in two_result.get("paths", [])))
		semantic_values = [row.get("semantic_equivalent", "UNCLEAR") for row in winners]
		semantic = "YES" if "YES" in semantic_values else ("NO" if "NO" in semantic_values else "UNCLEAR")
		summaries.append({
			"case_id": case_id,
			"dataset": trace["dataset"],
			"question": case["gold"]["question"],
			"current_entity": trace["current_entity"],
			"expected_relation": trace["expected_relation"],
			"expected_direction": trace["expected_direction"],
			"step_index": trace.get("step_index"),
			"source_trace_index": trace.get("relation_search", {}).get("trace_index"),
			"gold_ids": case["gold"]["ids"],
			"gold_names": case["gold"]["names"],
			"gold_answer_representation": case["gold"]["answer_representation"],
			"phase2_2_exact_relation_present": False,
			"raw_outgoing_count": trace.get("live_query_status", {}).get("outgoing", {}).get("rows_count"),
			"raw_incoming_count": trace.get("live_query_status", {}).get("incoming", {}).get("rows_count"),
			"one_hop_relations_executed": len(case_one_rows),
			"one_hop_classification": one_class,
			"one_hop_winners": [f"{row['relation']} ({row['direction']})" for row in winners],
			"one_hop_answer_reachable": answer_reachable,
			"two_hop_classification": (
				"TWO_HOP_EXACT_EQUIVALENT" if two_result and any(path.get("exact_answer_set_match") for path in two_result.get("paths", []))
				else "TWO_HOP_PARTIAL_GOLD_REACHABLE" if two_result and any(path.get("gold_overlap_count", 0) > 0 for path in two_result.get("paths", []))
				else "NOT_REQUIRED" if one_class != "NO_ONE_HOP_GOLD_REACHABILITY" else "NO_BOUNDED_2HOP_GOLD_REACHABILITY"
			),
			"two_hop_paths_explored": (two_result or {}).get("raw_path_rows", 0),
			"two_hop_gold_path_groups": (two_result or {}).get("eligible_gold_reaching_path_groups", 0),
			"answer_reachable_within_two_hops": two_reachable,
			"answer_equivalent_path": "one-hop" if answer_reachable else "two-hop" if two_reachable else "none",
			"semantic_equivalence": semantic,
			"mediator_cvt_evidence": [path.get("mediator_cvt_indication") for path in (two_result or {}).get("paths", [])] or "not required",
			"final_diagnosis": final_diagnosis,
			"diagnosis_reason": reason,
			"exclusions": exclusions,
			"case_runtime_seconds": round(time.time() - case_started, 3),
		})

	write_jsonl(OUTPUT_ROOT / "one_hop_execution_traces.jsonl", one_rows)
	two_rows = [path for result in two_results.values() for path in result.get("paths", [])]
	write_jsonl(OUTPUT_ROOT / "two_hop_execution_traces.jsonl", two_rows)
	write_json(OUTPUT_ROOT / "case_summary.json", summaries)
	write_json(OUTPUT_ROOT / "audit_metadata.json", {
		"endpoint": args.endpoint,
		"one_hop_result_limit": ONE_HOP_RESULT_LIMIT,
		"two_hop_path_limit": TWO_HOP_PATH_LIMIT,
		"query_count": query_count,
		"runtime_seconds": round(time.time() - started, 3),
		"frozen_components": ["routing", "entity_lookup", "Freebase execution", "evaluator", "Phase2.1 scorer", "Phase2.2 filters", "runtime inference"],
		"production_intervention": False,
	})
	overall = write_report(cases, one_rows, two_results, summaries, time.time() - started, query_count)
	print(f"Wrote Phase 2.2b audit to {OUTPUT_ROOT}")
	print(f"Overall decision: {overall}")
	print(f"SPARQL queries: {query_count}")


if __name__ == "__main__":
	run()
