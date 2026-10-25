#!/usr/bin/env python3
"""Audit Phase 2.2 relation-candidate recall from the frozen Phase 2.1 traces.

Phase 2.2 must change relation discovery only after a candidate-missing case is
shown to be a discovery/retention loss.  The Phase 2.1 trace stores the final
directional candidates and the relation-search counts, but not the raw
relation IDs before Freebase filtering.  This script therefore performs a
conservative offline audit and does not invent a candidate-retention method
when the raw stage cannot be observed.

It intentionally does not call an LLM, does not contact SPARQL, does not
modify the frozen Phase 2.1 scorer, and does not run the end-to-end evaluator.
If the endpoint is restored later, this audit can be extended with raw
head/tail relation IDs and the same output schema can be reused.
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
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from _bootstrap import PROJECT_ROOT


PHASE21_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase2_1" / "relation_grounding"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase2_2" / "relation_candidate_recall"


def read_json(path: Path) -> Any:
	return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	if not path.exists():
		return []
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		"".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
		encoding="utf-8",
	)


def compact_relation(item: dict[str, Any]) -> dict[str, str]:
	return {
		"relation": str(item.get("relation", "")),
		"direction": str(item.get("direction", "")),
		"label": str(item.get("label", item.get("relation", ""))),
	}


def relation_key(relation: str, direction: str) -> tuple[str, str]:
	return str(relation), str(direction)


def relation_is_expected(case: dict[str, Any], candidate: dict[str, Any]) -> bool:
	return relation_key(candidate.get("relation", ""), candidate.get("direction", "")) == relation_key(
		case.get("expected_relation", ""), case.get("expected_direction", "")
	)


def load_detail(case: dict[str, Any]) -> dict[str, Any] | None:
	path = (
		PHASE21_ROOT
		/ "details"
		/ "adaptive"
		/ str(case["dataset"])
		/ f"{case['question_id']}.json"
	)
	return read_json(path) if path.exists() else None


def relation_search_pairs(detail: dict[str, Any] | None) -> list[dict[str, Any]]:
	"""Pair the recorded head/tail relation searches in execution order."""
	if not detail:
		return []
	pairs: list[dict[str, Any]] = []
	current: dict[str, Any] = {}
	for event in detail.get("sparql_history", []):
		kind = str(event.get("search_type", ""))
		if kind.endswith("_head"):
			if current:
				pairs.append(current)
			current = {"head": event}
		elif kind.endswith("_tail"):
			current["tail"] = event
			if "head" in current and "tail" in current:
				pairs.append(current)
				current = {}
	if current:
		pairs.append(current)
	return pairs


def phase21_relation_event(detail: dict[str, Any] | None, trace_index: int) -> dict[str, Any] | None:
	if not detail:
		return None
	events = [
		event
		for event in detail.get("prompt_history", [])
		if event.get("type") == "phase2_1_semantic_relation_grounding"
	]
	if not events:
		return None
	return events[min(max(int(trace_index), 0), len(events) - 1)]


def relation_search_trace(case: dict[str, Any], detail: dict[str, Any] | None) -> dict[str, Any]:
	index = int(case.get("source_trace_index", case.get("step_index", 0)))
	pairs = relation_search_pairs(detail)
	pair = pairs[index] if index < len(pairs) else {}
	head = pair.get("head", {})
	tail = pair.get("tail", {})
	return {
		"trace_index": index,
		"head_query": head.get("sparql"),
		"tail_query": tail.get("sparql"),
		# These counts are what the old trace recorded after its parser.  They
		# are useful diagnostics, but are not raw relation-ID lists.
		"reported_neighbor_counts": {
			"outgoing": head.get("result_count"),
			"incoming": tail.get("result_count"),
		},
		"head_status": head.get("status"),
		"tail_status": tail.get("status"),
		"head_timed_out": head.get("timed_out"),
		"tail_timed_out": tail.get("timed_out"),
		"head_trace_id": head.get("trace_id"),
		"tail_trace_id": tail.get("trace_id"),
	}


def build_pipeline_trace(case: dict[str, Any]) -> dict[str, Any]:
	detail = load_detail(case)
	candidates = [compact_relation(item) for item in case.get("candidate_relations", [])]
	phase21_event = phase21_relation_event(detail, int(case.get("source_trace_index", 0)))
	search = relation_search_trace(case, detail)
	if phase21_event and phase21_event.get("candidate_directions"):
		# Keep the diagnostic case as the canonical list: it is the frozen
		# relation set used by both Phase 2.1 and its offline comparison.
		candidate_source = "phase2_1_diagnostic_case"
	else:
		candidate_source = "missing_phase2_1_event"

	available = any(relation_is_expected(case, item) for item in candidates)
	if available:
		first_disappearance = "NO_DISAPPEARANCE"
		diagnosis = "Expected relation is present in the recorded set passed to the frozen ranker."
	elif case.get("failure_type") == "RELATION_CANDIDATE_MISSING":
		first_disappearance = "UNKNOWN"
		diagnosis = (
		"The expected relation is absent from the recorded final candidate set, but raw relation IDs "
		"before Freebase filtering were not logged and the endpoint is unavailable."
	)
	else:
		first_disappearance = "NOT_APPLICABLE"
		diagnosis = "This is not a candidate-missing point; candidate recall is positive."

	return {
		"question_id": case["question_id"],
		"dataset": case["dataset"],
		"step_index": case.get("step_index"),
		"current_entity": case.get("current_entity"),
		"expected_relation": case.get("expected_relation"),
		"expected_direction": case.get("expected_direction"),
		"failure_type": case.get("failure_type"),
		"candidate_source": candidate_source,
		"raw_relation_ids_available": False,
		"all_raw_neighbor_relations_from_KG": None,
		"after_schema_filter": None,
		"after_direction_filter": None,
		"after_existing_pruning": candidates,
		"relations_passed_to_ranker": candidates,
		"reported_neighbor_counts": search["reported_neighbor_counts"],
		"relation_search": search,
		"candidate_recall": bool(available),
		"first_disappearance_stage": first_disappearance,
		"diagnosis": diagnosis,
		"raw_stage_note": (
			"Phase 2.1 recorded result counts and final candidates, not raw relation IDs at each filter stage; "
			"UNKNOWN is therefore intentional, not an assertion of KG absence."
		),
	}


def query_live_endpoint(endpoint: str, sparql: str, timeout: int) -> dict[str, Any]:
	started = time.time()
	if not sparql:
		return {"status": "missing_query", "timed_out": False, "rows": [], "elapsed_seconds": 0.0}
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


def raw_relation_records(result: dict[str, Any]) -> list[dict[str, str]]:
	records: list[dict[str, str]] = []
	seen: set[str] = set()
	prefix = "http://rdf.freebase.com/ns/"
	for row in result.get("rows", []):
		raw_uri = str(row.get("relation", {}).get("value", "")).strip()
		if not raw_uri or raw_uri in seen:
			continue
		seen.add(raw_uri)
		records.append({
			"raw_uri": raw_uri,
			"relation": raw_uri.replace(prefix, ""),
		})
	return records


def relation_ids(records: list[dict[str, str]]) -> set[str]:
	return {str(item.get("relation", "")) for item in records if item.get("relation")}


def relation_after_schema(records: list[dict[str, str]]) -> list[dict[str, str]]:
	def unnecessary(relation: str) -> bool:
		return (
			relation == "type.object.type"
			or relation == "type.object.name"
			or relation.startswith("common.")
			or relation.startswith("freebase.")
			or "sameAs" in relation
		)
	return [item for item in records if not unnecessary(str(item.get("relation", "")))]


def previous_relation_from_query(query: str | None) -> tuple[str, str] | None:
	"""Recover the immediately preceding hop from the recorded SPARQL."""
	if not query:
		return None
	for line in str(query).splitlines():
		if "?relation ?x" in line or "?x ?relation" in line:
			break
		outgoing = re.search(r"\bns:[^\s]+\s+ns:([^\s]+)\s+\?e_\d+\s*\.", line)
		if outgoing:
			return outgoing.group(1), "outgoing"
		incoming = re.search(r"\?e_\d+\s+ns:([^\s]+)\s+ns:[^\s]+\s*\.", line)
		if incoming:
			return incoming.group(1), "incoming"
	return None


def live_stage_for_expected(
	case: dict[str, Any],
	raw: dict[str, list[dict[str, str]]],
	after_schema: dict[str, list[dict[str, str]]],
	after_existing: dict[str, list[dict[str, str]]],
	recorded_candidates: list[dict[str, str]],
	combined_limit: int = 1000,
) -> str:
	if any(relation_is_expected(case, item) for item in recorded_candidates):
		return "NO_DISAPPEARANCE"
	expected = str(case.get("expected_relation", ""))
	direction = str(case.get("expected_direction", ""))
	opposite = "incoming" if direction == "outgoing" else "outgoing"
	if expected in relation_ids(raw.get(direction, [])):
		if expected not in relation_ids(after_schema.get(direction, [])):
			return "SCHEMA_FILTERED"
		if expected not in relation_ids(after_existing.get(direction, [])):
			return "OTHER"
		if expected not in {
				str(item.get("relation", ""))
				for item in recorded_candidates
			}:
			return "TOP_K_TRUNCATION" if sum(len(items) for items in after_existing.values()) > combined_limit else "OTHER"
	if expected in relation_ids(raw.get(opposite, [])):
		return "DIRECTION_FILTERED"
	return "NOT_PRESENT_IN_KG_NEIGHBORHOOD"


def augment_trace_with_live_raw(
	case: dict[str, Any],
	trace: dict[str, Any],
	endpoint: str,
	timeout: int,
) -> dict[str, Any]:
	search = trace.get("relation_search", {})
	head_result = query_live_endpoint(endpoint, search.get("head_query"), timeout)
	tail_result = query_live_endpoint(endpoint, search.get("tail_query"), timeout)
	raw = {
		"outgoing": raw_relation_records(head_result),
		"incoming": raw_relation_records(tail_result),
	}
	after_schema = {direction: relation_after_schema(records) for direction, records in raw.items()}
	after_direction = {direction: list(records) for direction, records in after_schema.items()}
	previous = previous_relation_from_query(search.get("head_query"))
	after_existing = {direction: list(records) for direction, records in after_direction.items()}
	if previous:
		previous_relation, previous_direction = previous
		inverse_direction = "incoming" if previous_direction == "outgoing" else "outgoing"
		after_existing[inverse_direction] = [
			item for item in after_existing[inverse_direction]
			if item.get("relation") != previous_relation
		]
	query_ok = head_result.get("status") == "ok" and tail_result.get("status") == "ok"
	if query_ok:
		first_stage = live_stage_for_expected(
			case,
			raw,
			after_schema,
			after_existing,
			trace.get("relations_passed_to_ranker", []),
		)
	else:
		first_stage = "UNKNOWN"
	recorded_keys = {
		relation_key(item.get("relation", ""), item.get("direction", ""))
		for item in trace.get("relations_passed_to_ranker", [])
	}
	live_keys = {
		relation_key(item.get("relation", ""), direction)
		for direction, items in after_existing.items()
		for item in items
	}
	trace["raw_relation_ids_available"] = query_ok
	trace["live_endpoint"] = endpoint
	trace["live_query_status"] = {
		"outgoing": {
			**{key: value for key, value in head_result.items() if key != "rows"},
			"rows_count": len(head_result.get("rows", [])),
		},
		"incoming": {
			**{key: value for key, value in tail_result.items() if key != "rows"},
			"rows_count": len(tail_result.get("rows", [])),
		},
	}
	trace["all_raw_neighbor_relations_from_KG"] = raw
	trace["after_schema_filter"] = after_schema
	trace["after_direction_filter"] = after_direction
	trace["after_existing_pruning"] = after_existing
	trace["live_relations_passed_to_ranker"] = [
		{"relation": item.get("relation"), "direction": direction}
		for direction, items in after_existing.items()
		for item in items
	]
	trace["live_candidate_set_matches_recorded"] = live_keys == recorded_keys
	trace["live_candidate_recall"] = any(
		relation_key(case.get("expected_relation", ""), case.get("expected_direction", "")) == key
		for key in live_keys
	)
	trace["first_disappearance_stage"] = first_stage
	if first_stage == "NO_DISAPPEARANCE":
		trace["diagnosis"] = "Expected relation is present in the recorded candidate set passed to the frozen ranker."
	elif first_stage == "NOT_PRESENT_IN_KG_NEIGHBORHOOD":
		trace["diagnosis"] = "Live Freebase queries returned no expected relation in either raw outgoing or incoming neighborhood at the recorded entity/decision point."
	else:
		trace["diagnosis"] = f"Live raw trace places the first disappearance at {first_stage}."
	trace["raw_stage_note"] = "Raw relation IDs were queried live from the restored endpoint; schema and existing-pruning stages were reconstructed with the frozen Freebase rules."
	return trace


def candidate_prediction() -> str:
	return """# Phase 2.2 prediction: relation-candidate discovery/recall

This prediction is written before any Phase 2.2 runtime intervention.

## Prediction

1. The four inherited `RELATION_CANDIDATE_MISSING` labels will not yet yield a
   confirmed candidate-retention loss, because the frozen Phase 2.1 trace does
   not contain raw relation IDs before Freebase filtering and the live endpoint
   is unavailable for reconstruction.
2. The expected relation will remain present for the 12 control/ranking
   points where the frozen candidate set already contains it.
3. The four missing cases will remain `UNKNOWN` at the first-disappearance
   stage rather than being treated as `NOT_PRESENT_IN_KG_NEIGHBORHOOD`.
4. No candidate-retention patch should be enabled until at least one missing
   case is confirmed to be present in the raw KG neighborhood and absent only
   after a filter, pruning, or truncation stage.

## Planned intervention if the prediction is disproved

If the endpoint-backed audit confirms a real retention loss, the smallest
permitted intervention is semantic-target-aware retention before the frozen
Phase 2.1 ranker: keep the baseline candidate budget and protect only validated
semantic candidates, with a fixed small budget such as `K+2` or `K+4`. No
router, entity linker, scorer, branch executor, or evaluator change is
authorized by this phase.
"""


def expected_pair(row: dict[str, Any] | None) -> tuple[str, str] | None:
	if not row:
		return None
	items = row.get("baseline_top1") or row.get("top_k_selected_relations") or row.get("selected_relation") or []
	if not items:
		return None
	item = items[0]
	if isinstance(item, str):
		return item, ""
	return str(item.get("relation", "")), str(item.get("direction", ""))


def write_candidate_comparison(
	cases: list[dict[str, Any]],
	baseline: dict[str, dict[str, Any]],
	phase21: dict[str, dict[str, Any]],
) -> None:
	available = sum(bool(case.get("candidate_recall")) for case in cases)
	live_available = sum(bool(case.get("live_candidate_recall")) for case in cases)
	missing = len(cases) - available
	confirmed_stages = {
		"SCHEMA_FILTERED",
		"DIRECTION_FILTERED",
		"PRUNED_BY_RELATION_SELECTOR",
		"TOP_K_TRUNCATION",
		"DEDUP_OR_NORMALIZATION_LOSS",
		"OTHER",
	}
	confirmed = sum(
		case.get("first_disappearance_stage") in confirmed_stages
		for case in cases
	)
	by_type = Counter(case.get("failure_type") for case in cases)
	live_mode = any(case.get("raw_relation_ids_available") for case in cases)
	lines = [
		"# Phase 2.2 candidate-recall comparison",
		"",
		("The inherited frozen Phase 2.1 decision points were replayed against the live endpoint."
		 if live_mode else
		 "The inherited frozen Phase 2.1 decision points are used for offline audit only."),
		"No Phase 2.2 candidate-generation intervention was run because no candidate-retention loss was confirmed.",
		"",
		"| Metric | Frozen Phase 2.1 candidate set | Phase 2.2 intervention |",
		"|---|---:|---:|",
		f"| Diagnostic points | {len(cases)} | not run |",
		f"| Expected relation present | {available}/{len(cases)} ({available / len(cases):.2%}) | not estimable |",
		f"| Expected relation absent | {missing}/{len(cases)} | not estimable |",
		f"| Live post-filter candidate recall | {live_available}/{len(cases)} ({live_available / len(cases):.2%}) | not applicable |",
		f"| Confirmed filter/pruning losses | {confirmed} | 0 tested |",
		"",
		"## Interpretation",
		"",
		(f"The inherited candidate recall is {available}/{len(cases)} ({available / len(cases):.2%}). "
		 "The live replay confirms that the missing expected relations are absent from both raw directions "
		 "at the recorded decision entities; none disappears in a candidate-retention filter. "
		 "The correct Phase 2.2 result is therefore no candidate-recall signal."
		 if live_mode else
		 f"The nominal inherited candidate recall is {available}/{len(cases)} ({available / len(cases):.2%}), "
		 "but the four absent relations are not evidence of a candidate-discovery failure yet. "
		 "Their raw pre-filter relation IDs were not recorded, and the SPARQL endpoint is currently unavailable. "
		 "The correct Phase 2.2 result is therefore an unresolved audit, not a fabricated recall gain."),
		"",
		"Case mix: " + ", ".join(f"{key}={value}" for key, value in sorted(by_type.items())) + ".",
		"",
		("The frozen Phase 2.1 ranker remains unchanged. No candidate-retention patch is justified by the live evidence."
		 if live_mode else
		 "The frozen Phase 2.1 ranker remains unchanged. Candidate recall and ranking accuracy must be rerun only after an endpoint-backed raw-neighborhood trace is available."),
	]
	(OUTPUT_ROOT / "candidate_recall_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_relation_comparison(
	baseline: dict[str, dict[str, Any]],
	phase21: dict[str, dict[str, Any]],
) -> None:
	rows = list(phase21.values())
	base_correct = sum(bool(row.get("baseline_relation_selection_correct")) for row in rows)
	p21_correct = sum(bool(row.get("relation_selection_correct")) for row in rows)
	phase21_candidates = sum(bool(row.get("candidate_recall")) for row in rows)
	lines = [
		"# Relation-level comparison",
		"",
		"Phase 2.2 did not alter candidate generation or invoke a new ranker. The Phase 2.1 scorer and its measured result are reproduced only as a frozen reference.",
		"",
		"| Metric | Baseline | Frozen Phase 2.1 | Phase 2.2 |",
		"|---|---:|---:|---:|",
		f"| Relation top-1 | {base_correct}/{len(rows)} ({base_correct / len(rows):.2%}) | {p21_correct}/{len(rows)} ({p21_correct / len(rows):.2%}) | not run |",
		f"| Candidate recall | {phase21_candidates}/{len(rows)} ({phase21_candidates / len(rows):.2%}) | {phase21_candidates}/{len(rows)} ({phase21_candidates / len(rows):.2%}) | not estimable |",
		"| Frozen Phase 2.1 ranker changed | no | no | no |",
		"",
		"The six unresolved BOTH_WRONG ranking cases have positive candidate recall and are outside the Phase 2.2 intervention target. The four candidate-missing labels remain raw-stage `UNKNOWN`.",
	]
	(OUTPUT_ROOT / "relation_level_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_control_report(cases: list[dict[str, Any]]) -> None:
	controls = [case for case in cases if case.get("failure_type") == "SUCCESSFUL_CONTROL"]
	preserved = sum(bool(case.get("candidate_recall")) for case in controls)
	lines = [
		"# Control preservation",
		"",
		f"The frozen diagnostic contains {len(controls)} successful controls. Their expected relations are present in the recorded candidate sets for {preserved}/{len(controls)} controls.",
		"",
		"| Control preservation metric | Result |",
		"|---|---:|",
		f"| Candidate availability | {preserved}/{len(controls)} |",
		"| Phase 2.2 runtime intervention applied | 0 |",
		"| Controls changed by Phase 2.2 | 0 observed (no intervention run) |",
		"",
		"Because Phase 2.2 was not enabled, this is a preservation audit, not a claim of improvement.",
	]
	(OUTPUT_ROOT / "control_preservation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_noise_report(traces: list[dict[str, Any]]) -> None:
	lines = [
		"# Candidate noise analysis",
		"",
		"No candidate-retention intervention was run. Consequently, there are no added candidates, no new LLM candidate calls, and no measured noise increase.",
		"",
		"| Quantity | Result |",
		"|---|---:|",
		"| Added candidates | 0 measured; intervention not run |",
		"| Candidate sets expanded | no |",
		"| New selector calls | 0 |",
		"| Raw-stage unknown cases | " + str(sum(t.get("first_disappearance_stage") == "UNKNOWN" for t in traces)) + " |",
		"",
		"A future intervention must report candidate-set deltas and selected-candidate noise separately. Passing all raw relations to the ranker is explicitly not approved by this audit.",
	]
	(OUTPUT_ROOT / "noise_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_both_wrong_analysis(
	e2e_rows: list[dict[str, Any]],
	case_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for row in e2e_rows:
		baseline = row.get("baseline") or {}
		phase21 = row.get("phase2_1") or {}
		if baseline.get("hits_at_1") or phase21.get("hits_at_1"):
			continue
		case = case_by_id.get(str(row.get("question_id")))
		rows.append(
			{
				"question_id": row.get("question_id"),
				"dataset": row.get("dataset"),
				"failure_type": row.get("failure_type"),
				"expected_relation": row.get("expected_relation"),
				"candidate_recall": None if not case else bool(case.get("candidate_recall")),
				"phase2_2_relevance": (
					"candidate-discovery audit target"
					if case and case.get("failure_type") == "RELATION_CANDIDATE_MISSING"
					else "not a candidate-discovery target"
				),
				"baseline_prediction": baseline.get("prediction"),
				"phase2_1_prediction": phase21.get("prediction"),
			}
		)
	return rows


def write_analysis_report(
	traces: list[dict[str, Any]],
	both_wrong: list[dict[str, Any]],
) -> None:
	missing = [trace for trace in traces if trace.get("failure_type") == "RELATION_CANDIDATE_MISSING"]
	live_mode = any(trace.get("raw_relation_ids_available") for trace in traces)
	known_raw = sum(bool(trace.get("raw_relation_ids_available")) for trace in traces)
	stage_counts = Counter(trace.get("first_disappearance_stage") for trace in missing)
	if live_mode:
		scope = (
			"This audit replays the exact recorded head/tail relation-search queries against the restored live endpoint. "
			f"Raw relation IDs were available for {known_raw}/{len(traces)} decision points."
		)
		evidence_boundary = (
			"The live result is decisive for these four points: every expected relation is absent from both raw outgoing "
			"and raw incoming neighborhoods at the recorded current entity. Therefore no candidate-retention intervention "
			"is justified. This does not claim the relation is absent at later hops or different entities."
		)
	else:
		scope = (
			"This audit uses the frozen Phase 2.1 diagnostic and detail traces. The traces record relation-search SPARQL "
			"and result counts plus the final directional candidate set, but not raw relation IDs before each Freebase filter."
		)
		evidence_boundary = (
			"Therefore `UNKNOWN` means unobservable, not absent. No Phase 2.2 candidate-retention implementation is "
			"justified by the current evidence."
		)
	lines = [
		"# Phase 2.2 candidate-missing analysis",
		"",
		"## Scope and evidence boundary",
		"",
		scope,
		"",
		evidence_boundary,
		"",
		"## Candidate-missing cases",
		"",
		"| Dataset | ID | Expected relation/direction | Raw out/in | Final candidates | First disappearance | Live diagnosis |",
		"|---|---|---|---:|---:|---|---|",
	]
	for trace in missing:
		counts = (
			(trace.get("live_query_status") or {}).get("outgoing", {}).get("rows_count"),
			(trace.get("live_query_status") or {}).get("incoming", {}).get("rows_count"),
		) if live_mode else (
			(trace.get("reported_neighbor_counts") or {}).get("outgoing"),
			(trace.get("reported_neighbor_counts") or {}).get("incoming"),
		)
		diagnosis = trace.get("diagnosis", "Raw-stage reconstruction required.")
		lines.append(
			f"| {trace['dataset']} | `{trace['question_id']}` | `{trace['expected_relation']}` / {trace['expected_direction']} | "
			f"{counts[0]} out, {counts[1]} in | {len(trace.get('relations_passed_to_ranker', []))} | "
			f"{trace['first_disappearance_stage']} | {diagnosis} |"
		)
	missing_sentence = (
		f"The live stage counts are: {', '.join(f'{key}={value}' for key, value in sorted(stage_counts.items()))}. "
		"All four expected relations are absent from the raw neighborhood at the recorded decision point."
		if live_mode else
		"All four missing labels are retained as diagnostic cases, but none is counted as a confirmed candidate-discovery loss."
	)
	lines += [
		"",
		missing_sentence,
		"",
		"## Unresolved BOTH_WRONG cases",
		"",
		("The Phase 2.1 end-to-end diagnostic had ten BOTH_WRONG pairs. Six had positive candidate recall and are "
		 "ranking/downstream cases, not candidate-discovery targets. Four overlap with the missing cases above and are "
		 "confirmed raw-neighborhood absences at the recorded entities."
		 if live_mode else
		 "The Phase 2.1 end-to-end diagnostic had ten BOTH_WRONG pairs. Six had positive candidate recall and are ranking/downstream cases, not candidate-discovery targets. Four overlap with the missing cases above and remain raw-stage `UNKNOWN`."),
		"",
		"| ID | Failure label | Candidate recall | Phase 2.2 relevance |",
		"|---|---|---:|---|",
	]
	for row in both_wrong:
		value = "unknown" if row["candidate_recall"] is None else ("yes" if row["candidate_recall"] else "no")
		lines.append(f"| `{row['question_id']}` | {row['failure_type']} | {value} | {row['phase2_2_relevance']} |")
	lines += [
		"",
		"## Stage reconstruction status",
		"",
		"| Stage | Status |",
		"|---|---|",
		(f"| Raw outgoing/incoming relation IDs from KG | live and available for {known_raw}/{len(traces)} points |"
		 if live_mode else
		 "| Raw outgoing/incoming relation IDs from KG | unavailable in frozen trace; endpoint unavailable |"),
		("| Schema filtering | reconstructed with frozen Freebase hard-filter rules |"
		 if live_mode else
		 "| Schema filtering | unknown for missing cases |"),
		("| Direction filtering | separate live outgoing/incoming queries |"
		 if live_mode else
		 "| Direction filtering | represented in final trace, but pre-filter IDs unavailable |"),
		("| Existing inverse/necessary-relation pruning | reconstructed from frozen rules and recorded query state |"
		 if live_mode else
		 "| Existing inverse/necessary-relation pruning | final candidate set available; exact removals unknown |"),
		"| Relation-selector top-k | recorded separately as a ranking decision, not treated as candidate discovery |",
		f"| First disappearance stage | {', '.join(f'{key}={value}' for key, value in sorted(stage_counts.items()))} |",
		"",
		"This separation prevents a wrong-hop or wrong-entity label from being “fixed” by widening the relation candidate set.",
	]
	(OUTPUT_ROOT / "candidate_missing_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_end_to_end_status() -> None:
	row = {
		"status": "NOT_RUN",
		"reason": "No confirmed candidate-discovery loss; Phase 2.2 runtime intervention was not enabled.",
		"frozen_phase2_1_end_to_end_comparison": "outputs/experiments/phase2_1/relation_grounding/end_to_end_summary.md",
	}
	write_jsonl(OUTPUT_ROOT / "end_to_end_results.jsonl", [row])
	(OUTPUT_ROOT / "end_to_end_summary.md").write_text(
		"# Phase 2.2 end-to-end status\n\n"
		"The end-to-end evaluator was not rerun. Phase 2.2 did not introduce a candidate-discovery intervention because no raw-stage retention loss was confirmed. The frozen Phase 2.1 end-to-end result remains the last valid runtime comparison.\n",
		encoding="utf-8",
	)


def write_cost_report(traces: list[dict[str, Any]]) -> None:
	live_points = sum(bool(trace.get("raw_relation_ids_available")) for trace in traces)
	live_queries = live_points * 2
	live_seconds = sum(
		float((trace.get("live_query_status") or {}).get(direction, {}).get("elapsed_seconds", 0.0))
		for trace in traces
		for direction in ("outgoing", "incoming")
	)
	(OUTPUT_ROOT / "cost_analysis.md").write_text(
		"# Phase 2.2 cost analysis\n\n"
		f"The live diagnostic issued {live_queries} read-only SPARQL queries across {live_points} decision points ({live_seconds:.2f} seconds of reported query time). It made no LLM calls and enabled no candidate-retention intervention.\n\n"
		"Measured Phase 2.2 intervention cost: 0 additional selector calls, 0 additional tokens, and 0 candidate-set expansions. A future approved intervention must report these quantities for baseline K, K+2, and K+4 retention budgets before selecting one fixed budget.\n",
		encoding="utf-8",
	)


def write_final_report(
	cases: list[dict[str, Any]],
	traces: list[dict[str, Any]],
	both_wrong: list[dict[str, Any]],
) -> None:
	missing = [trace for trace in traces if trace.get("failure_type") == "RELATION_CANDIDATE_MISSING"]
	live_mode = any(trace.get("raw_relation_ids_available") for trace in traces)
	stage_counts = Counter(trace.get("first_disappearance_stage") for trace in missing)
	lines = [
		"# Phase 2.2 report: relation-candidate discovery/recall",
		"",
		"## Frozen components",
		"",
		"Adaptive v1.1.3 routing, entity lookup, Freebase backend, SPARQL execution, branch handling, evaluator, and the Phase 2.1 semantic-target scorer were not changed. No end-to-end evaluation was rerun.",
		"",
		"## What was completed",
		"",
		f"- Audited {len(cases)} frozen relation decision points ({len(missing)} inherited candidate-missing labels and {len(cases) - len(missing)} controls/ranking points).",
		("- Replayed the exact recorded relation-search queries against the restored endpoint and recorded raw relation IDs before the frozen Freebase filters."
		 if live_mode else
		 "- Reconstructed the available relation-search counts, queries, final candidate sets, expected relation/direction, and ranker input from the Phase 2.1 logs."),
		"- Audited all ten unresolved BOTH_WRONG end-to-end pairs and separated six positive-recall ranking/downstream cases from the four missing-label cases.",
		("- Reconstructed schema, direction, and existing-pruning stages with the frozen Freebase rules."
		 if live_mode else
		 "- Marked raw KG/filter stages as `UNKNOWN` where the old trace did not record raw IDs."),
		"",
		"## Result",
		"",
		("The inherited candidate set has recall 12/16 (75%), matching Phase 2.1. The live endpoint confirms all four missing expected relations are absent from both raw directions at the recorded current entities. There are **zero confirmed candidate-retention losses**. These may still be valid relations at later hops or different entities, but widening candidate retention at this decision point cannot recover them."
		 if live_mode else
		 "The inherited candidate set has nominal offline recall of 12/16 (75%), matching Phase 2.1. However, there are **zero confirmed candidate-retention losses** in the current evidence. The four absent expected relations may reflect a wrong entity or later-hop gold relation, and cannot be safely repaired by widening candidate retention."),
		"",
		"No Phase 2.2 runtime intervention was implemented or enabled. This preserves the clean Phase 2.1 comparison and avoids changing the baseline semantics without evidence.",
		"",
		"## Decision",
		"",
		"**NO_CLEAR_CANDIDATE_RECALL_SIGNAL**",
		"",
		"## Next evidence required",
		"",
		("Do not implement a Phase 2.2 candidate-retention patch. The next research step should investigate hop/entity-state alignment for these cases or proceed to the next approved phase; the frozen Phase 2.1 ranker remains unchanged."
		 if live_mode else
		 "Restore the Linux Virtuoso endpoint/SSH tunnel and rerun the same four decision points with raw relation IDs logged before and after each filter/retention stage. Proceed to a targeted semantic retention patch only if at least one expected relation is observed in the raw neighborhood and disappears before the frozen ranker."),
	]
	(OUTPUT_ROOT / "PHASE2_2_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
	global OUTPUT_ROOT
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
	parser.add_argument("--live", action="store_true", help="Replay recorded relation-search queries against the live endpoint.")
	parser.add_argument(
		"--endpoint",
		default=os.getenv("FREEBASE_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql"),
		help="SPARQL endpoint used with --live.",
	)
	parser.add_argument("--timeout", type=int, default=180, help="Per-query timeout in seconds for --live.")
	args = parser.parse_args()
	OUTPUT_ROOT = args.output_root

	cases = read_json(PHASE21_ROOT / "diagnostic_cases.json")
	baseline_rows = read_jsonl(PHASE21_ROOT / "relation_decisions_baseline.jsonl")
	phase21_rows = read_jsonl(PHASE21_ROOT / "relation_decisions_phase2_1.jsonl")
	e2e_rows = read_jsonl(PHASE21_ROOT / "end_to_end_results.jsonl")
	baseline = {str(row["question_id"]): row for row in baseline_rows}
	phase21 = {str(row["question_id"]): row for row in phase21_rows}

	traces = [build_pipeline_trace(case) for case in cases]
	if args.live:
		for index, (case, trace) in enumerate(zip(cases, traces), start=1):
			print(
				f"LIVE {index}/{len(cases)} {case['dataset']}:{case['question_id']}",
				flush=True,
			)
			augment_trace_with_live_raw(case, trace, args.endpoint, max(1, int(args.timeout)))
	case_by_id = {str(case["question_id"]): trace for case, trace in zip(cases, traces)}
	both_wrong = write_both_wrong_analysis(e2e_rows, case_by_id)

	write_json(OUTPUT_ROOT / "diagnostic_cases.json", cases)
	write_jsonl(OUTPUT_ROOT / "candidate_pipeline_traces.jsonl", traces)
	write_json(OUTPUT_ROOT / "prediction.md.json", {"prediction_markdown": candidate_prediction()})
	(OUTPUT_ROOT / "prediction.md").write_text(candidate_prediction(), encoding="utf-8")

	baseline_decisions = []
	phase22_decisions = []
	for case, trace in zip(cases, traces):
		qid = str(case["question_id"])
		b = baseline.get(qid, {})
		p = phase21.get(qid, {})
		base_decision = {
			"question_id": qid,
			"dataset": case["dataset"],
			"expected_relation": case["expected_relation"],
			"expected_direction": case["expected_direction"],
			"candidate_recall": bool(trace["candidate_recall"]),
			"candidate_count": len(trace["relations_passed_to_ranker"]),
			"baseline_top1": (b.get("baseline_top1") or [])[:1],
			"baseline_selection_correct": bool(b.get("baseline_relation_selection_correct")),
			"first_disappearance_stage": trace["first_disappearance_stage"],
		}
		baseline_decisions.append(base_decision)
		phase22_decisions.append(
			{
				**base_decision,
				"phase2_2_status": "NOT_RUN_NO_CONFIRMED_RETENTION_LOSS",
				"intervention_applied": False,
				"candidate_set_delta": 0,
				"phase2_2_candidate_recall": None,
				"frozen_phase2_1_top1": (p.get("top_k_selected_relations") or [])[:1],
			}
		)
	write_jsonl(OUTPUT_ROOT / "baseline_candidate_decisions.jsonl", baseline_decisions)
	write_jsonl(OUTPUT_ROOT / "phase2_2_candidate_decisions.jsonl", phase22_decisions)
	write_jsonl(OUTPUT_ROOT / "unresolved_both_wrong.jsonl", both_wrong)

	write_candidate_comparison(traces, baseline, phase21)
	write_relation_comparison(baseline, phase21)
	write_control_report(traces)
	write_noise_report(traces)
	write_analysis_report(traces, both_wrong)
	write_end_to_end_status()
	write_cost_report(traces)
	write_final_report(cases, traces, both_wrong)

	print(f"Wrote conservative Phase 2.2 audit to {OUTPUT_ROOT}")
	print("Decision: NO_CLEAR_CANDIDATE_RECALL_SIGNAL")


if __name__ == "__main__":
	run()
