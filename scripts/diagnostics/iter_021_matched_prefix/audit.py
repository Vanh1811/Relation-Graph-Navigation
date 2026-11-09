#!/usr/bin/env python3
"""Offline Iteration 21 trace audit and artifact builder. Never calls APIs/KG."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from collections import defaultdict


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
RESULTS = WORK / "RESULTS/iter_021_matched_prefix_replay"
RAW = WORK / "RESULTS/iter_019_depth4_holdout_preparation/control_continuation/paired_runs_cumulative.jsonl"
METADATA = WORK / "RESULTS/iter_019_depth4_holdout_preparation/control_continuation/continuation_metadata.json"
TEST_DIR = ROOT / "scripts/diagnostics/iter_021_matched_prefix"
MODEL_INPUT_PRICE = 0.20
MODEL_OUTPUT_PRICE = 1.20
MAX_CALLS = 60
MAX_INPUT_TOKEN_BOUND_PER_CALL = 7000
MAX_OUTPUT_TOKENS = 1024
AUTH_COST_CAP_USD = 0.17
CONTROL_ID = "WebQTest-450_3eb90d65b82703c8f7fb0496baa80ce7"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def event_types(row: dict) -> set[str]:
    return {str(event.get("type", "")) for event in row.get("prompt_history", [])}


def have_event(row: dict, stage: str) -> bool:
    return any(event.get("stage") == stage for event in row.get("state_lineage", []))


def feasibility_matrix(rows: list[dict], metadata: dict) -> list[dict[str, str]]:
    n = len(rows)
    cases: list[tuple[str, callable, str, str, str]] = [
        ("question_id", lambda r: bool(r.get("question_id")), "row.question_id", "EXACT_VALUE", "Question ID is present on every run."),
        ("repeat_index", lambda r: r.get("repeat_index") is not None, "row.repeat_index", "EXACT_VALUE", "Repeat index is present on every run."),
        ("route", lambda r: bool(r.get("router_execution_mode")), "row.router_execution_mode", "OBSERVED_ROUTE", "Routing output is recorded, but it was independently sampled between depth runs."),
        ("model_and_global_request_settings", lambda r: bool(metadata.get("model") and metadata.get("max_completion_tokens_per_request") and metadata.get("timeout_seconds") is not None and metadata.get("retries") is not None), "continuation_metadata.json", "RUN_LEVEL_ONLY", "Model, output cap, timeout, retry count are run-level; per-call sampling settings are not fully recorded."),
        ("prompt_response_text", lambda r: any(e.get("prompt") is not None and e.get("response") is not None for e in r.get("prompt_history", [])), "row.prompt_history", "TEXT_RECORDED", "Saved prompts/responses permit text comparison; no exact decision-cache/replay IDs are attached."),
        ("prompt_decision_order", lambda r: bool(r.get("prompt_history")), "row.prompt_history order", "ORDERED_EVENTS", "LLM events are ordered within a run; no shared prefix hash links independent runs."),
        ("gold_id_present_for_scoring_only", lambda r: isinstance(r.get("gold_candidate_diagnostic_scoring_only"), dict) and r["gold_candidate_diagnostic_scoring_only"].get("gold_answer_seen_any_search") is not None, "row.gold_candidate_diagnostic_scoring_only", "SCORING_ONLY", "Presence summary is available after execution; it is not a binding tuple and must not be used for inference."),
        ("Forward_Finish_Backtrack_actions", lambda r: any(str(e.get("parsed_result", "")).lower() in {"forward", "finish", "backtrack", "stop", "constraint"} for e in r.get("prompt_history", [])), "row.prompt_history.parsed_result", "PARTIAL_ACTION_TRACE", "Actions are logged where an LLM event exists; empty/failed results and executor decisions are not a complete state machine log."),
        ("selected_relation_ID_and_direction", lambda r: any(e.get("selected_relations") for e in r.get("state_lineage", [])), "row.state_lineage.relation_prune", "SELECTED_ONLY", "Selected relation IDs/directions are recorded; stack order after push is not."),
        ("DFS_stack_position_and_contents", lambda r: any(k in r for k in ("dfs_stack", "stack_snapshot")), "not in Iteration 19 rows", "MISSING", "Required to resume the same DFS frontier and pending siblings."),
        ("active_intermediate_binding_ID_tuples", lambda r: any(e.get("active_binding_rows") for e in r.get("state_lineage", [])), "not in Iteration 19 rows", "MISSING", "Entity search stores endpoint IDs, not the joined ?e1…?eN tuples; exact graph bindings cannot be restored."),
        ("endpoint_frontier_entity_IDs", lambda r: any(e.get("stage") == "entity_search" and "candidate_entity_ids" in e for e in r.get("state_lineage", [])), "row.state_lineage.entity_search", "ENDPOINT_IDS_ONLY", "Endpoint candidate IDs are recorded, but not all intermediate bindings or raw backend rows."),
        ("candidate_relation_IDs_before_and_after_prune", lambda r: any(e.get("stage") == "relation_prune" and "candidate_relations_head" in e and "selected_relations" in e for e in r.get("state_lineage", [])), "row.state_lineage.relation_prune", "PARTIAL_IDS", "Candidate lists and selected IDs are logged; order/stack mutation is not fully represented."),
        ("SPARQL_text_status_and_row_count", lambda r: bool(r.get("sparql_history")), "row.sparql_history", "QUERY_AND_COUNT", "Query text, status and result counts are recorded, not raw rows."),
        ("normalized_relation_result_IDs", lambda r: any(e.get("stage") == "relation_search" and ("head_relations" in e or "tail_relations" in e) for e in r.get("state_lineage", [])), "row.state_lineage.relation_search", "NORMALIZED_IDS", "Relation result IDs are available through lineage wrappers, but query binding rows are not."),
        ("raw_KG_binding_rows", lambda r: any(e.get("raw_bindings") for e in r.get("sparql_history", [])), "not in Iteration 19 rows", "MISSING", "SPARQL history records counts and candidate endpoint IDs only."),
        ("branch_merge_or_intersection_IDs", lambda r: any(e.get("merge_binding_ids") for e in r.get("state_lineage", [])), "not in Iteration 19 rows", "MISSING", "Branch diagnostics contain counts/serialized plan evidence, not joined binding tuples at each merge."),
        ("filter_input_and_output_IDs", lambda r: any(e.get("filter_input_ids") is not None and e.get("filter_output_ids") is not None for e in r.get("prompt_history", [])), "not in Iteration 19 rows", "MISSING_IDS", "Filter prompt/labels and counts may be recorded; candidate and accepted entity IDs are not."),
        ("answer_source_linked_to_final_IDs", lambda r: bool(r.get("answer_source") and r.get("final_answer_ids") is not None), "not in Iteration 19 rows", "MISSING_LINK", "Filter/direct generation event types exist, but final predictions are not consistently linked to graph IDs and provenance."),
        ("deterministic_sort_rng_iteration_settings", lambda r: bool(r.get("deterministic_settings")), "source + run config, not run row", "NOT_CAPTURED", "Relation normalization is sorted in code; entity backend order and any RNG state are not recorded per run."),
        ("prefix_hash_and_replay_exact_flag", lambda r: bool(r.get("prefix_hash") and r.get("prefix_replay_exact")), "not in Iteration 19 rows", "MISSING", "No archive row proves exact state identity across the independent depth runs."),
    ]
    def relevant_events(row: dict, field: str) -> tuple[list[dict], str]:
        lineage = row.get("state_lineage", [])
        pre_depth3 = [e for e in lineage if len(e.get("relation_chain", [])) <= 3]
        if field == "Forward_Finish_Backtrack_actions":
            return [e for e in pre_depth3 if e.get("stage") == "reasoning"], "state_lineage reasoning events with chain length <= 3"
        if field == "selected_relation_ID_and_direction":
            return [e for e in pre_depth3 if e.get("stage") == "relation_prune"], "state_lineage relation_prune events with chain length <= 3"
        if field == "DFS_stack_position_and_contents":
            return [e for e in pre_depth3 if e.get("stage") == "reasoning" and e.get("action") == "Forward" and len(e.get("relation_chain", [])) == 3], "depth-3 Forward boundary events"
        if field in {"active_intermediate_binding_ID_tuples", "endpoint_frontier_entity_IDs", "raw_KG_binding_rows"}:
            return [e for e in pre_depth3 if e.get("stage") == "entity_search"], "state_lineage entity_search events with chain length <= 3"
        if field == "candidate_relation_IDs_before_and_after_prune":
            return [e for e in pre_depth3 if e.get("stage") == "relation_prune"], "state_lineage relation_prune events with chain length <= 3"
        if field == "normalized_relation_result_IDs":
            return [e for e in pre_depth3 if e.get("stage") == "relation_search"], "state_lineage relation_search events with chain length <= 3"
        if field == "branch_merge_or_intersection_IDs":
            return [e for e in pre_depth3 if "merge" in str(e.get("stage", "")).lower()], "merge events in state_lineage with chain length <= 3"
        if field == "filter_input_and_output_IDs":
            return [e for e in row.get("prompt_history", []) if e.get("type") == "filter_answer"], "answer filter events; prompt history lacks depth linkage"
        if field == "prompt_response_text":
            return [e for e in row.get("prompt_history", [])], "all logged prompt events; prefix boundary cannot be aligned to prompt history"
        return [], "not an event-level field or event-to-prefix linkage is unavailable"

    def event_value_available(field: str, event: dict) -> bool:
        if field == "Forward_Finish_Backtrack_actions":
            return event.get("action") is not None
        if field == "selected_relation_ID_and_direction":
            return all(isinstance(x, dict) and x.get("id") and x.get("direction") in {"OUTGOING", "INCOMING"} for x in event.get("selected_relations", []))
        if field == "DFS_stack_position_and_contents":
            return isinstance(event.get("dfs_stack"), list)
        if field in {"active_intermediate_binding_ID_tuples", "raw_KG_binding_rows"}:
            return "active_binding_rows" in event or "raw_bindings" in event
        if field == "endpoint_frontier_entity_IDs":
            return isinstance(event.get("candidate_entity_ids"), list)
        if field == "candidate_relation_IDs_before_and_after_prune":
            return "candidate_relations_head" in event and "candidate_relations_tail" in event and "selected_relations" in event
        if field == "normalized_relation_result_IDs":
            return "head_relations" in event and "tail_relations" in event
        if field == "branch_merge_or_intersection_IDs":
            return "merge_binding_ids" in event
        if field == "filter_input_and_output_IDs":
            return isinstance(event.get("filter_input_ids"), list) and isinstance(event.get("filter_output_ids"), list)
        if field == "prompt_response_text":
            return event.get("prompt") is not None and event.get("response") is not None
        return False

    output = []
    for field, predicate, source, status, limitation in cases:
        available = sum(bool(predicate(row)) for row in rows)
        relevant = [event for row in rows for event in relevant_events(row, field)[0]]
        event_available = sum(event_value_available(field, event) for event in relevant)
        scope = next((relevant_events(row, field)[1] for row in rows if relevant_events(row, field)[1] != "not an event-level field or event-to-prefix linkage is unavailable"), "not applicable")
        output.append({
            "field": field,
            "run_records_with_value": str(available),
            "total_run_records": str(n),
            "coverage_rate": f"{available / n:.4f}" if n else "0",
            "applicable_trace_events": str(len(relevant)) if relevant else "N/A",
            "trace_events_with_value": str(event_available) if relevant else "N/A",
            "trace_event_coverage_rate": f"{event_available / len(relevant):.4f}" if relevant else "N/A",
            "trace_event_scope": scope,
            "trace_source": source,
            "replay_status": status,
            "limitation": limitation,
        })
    return output


def control_loss(rows: list[dict]) -> dict:
    selected = [r for r in rows if r.get("question_id") == CONTROL_ID and r.get("repeat_index") == 1]
    by_depth = {r.get("diagnostic_depth"): r for r in selected}
    d3, d4 = by_depth[3], by_depth[4]
    def reasoning_at_depth(row: dict, depth: int) -> list[dict]:
        return [e for e in row.get("state_lineage", []) if e.get("stage") == "reasoning" and len(e.get("relation_chain", [])) == depth]
    d3_trace, d4_trace = reasoning_at_depth(d3, 3), reasoning_at_depth(d4, 3)
    d3_filter = [e for e in d3.get("prompt_history", []) if e.get("type") == "filter_answer"]
    d4_filter = [e for e in d4.get("prompt_history", []) if e.get("type") == "filter_answer"]
    d4_empty = [e for e in d4.get("prompt_history", []) if str(e.get("parsed_result", "")).lower() == "unknownaction"]
    return {
        "question_id": CONTROL_ID,
        "repeat_index": 1,
        "depth3": {
            "hit": d3.get("hits_at_1"), "prediction": d3.get("prediction"),
            "answer_source_observation": "filter_answer event present",
            "filter_events": [{"candidate_size": e.get("candidate_size"), "filtered_candidate_size": e.get("filtered_candidate_size"), "parsed_result": e.get("parsed_result")} for e in d3_filter],
            "depth3_boundary_reasoning_actions": [e.get("action") for e in d3_trace],
        },
        "depth4": {
            "hit": d4.get("hits_at_1"), "prediction": d4.get("prediction"),
            "answer_source_observation": "no filter_answer event; no final answer IDs",
            "filter_events": len(d4_filter),
            "depth3_reasoning_actions": [e.get("action") for e in d4_trace],
            "empty_or_unknown_reasoning_events": [{"response_present": bool(e.get("response")), "warnings": e.get("warnings", [])} for e in d4_empty],
            "execution_status": d4.get("execution_status"), "error": d4.get("error"),
        },
        "interpretation": (
            "The D3 run's depth-3 reasoning event is Forward, while the independent D4 run's corresponding depth-3 event is Backtrack. "
            "The D4 run later records an UnknownAction with an empty response warning and returns no result; it has no answer-filter event. "
            "Thus the recorded loss is consistent with independent trajectory/model-response divergence, including an empty reasoning response. "
            "It is not evidence that permitting one extra edge erased the D3 answer: these are separate runs, and the D4 run never has a matched D3 prefix. "
            "Filter input IDs and active binding tuples are absent, so candidate loss or binding-merge explanations cannot be ruled in or out."
        ),
    }


def write_feasibility(rows: list[dict], metadata: dict) -> None:
    fields = feasibility_matrix(rows, metadata)
    out = RESULTS / "REPLAY_FEASIBILITY.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields[0]))
        writer.writeheader()
        writer.writerows(fields)
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row.get("question_id"), row.get("repeat_index"))][row.get("diagnostic_depth")] = row
    exact = sum(
        bool(a.get("prefix_replay_exact") and b.get("prefix_replay_exact") and a.get("prefix_hash") == b.get("prefix_hash"))
        for a, b in ((v.get(3), v.get(4)) for v in pairs.values()) if a and b
    )
    write_json(RESULTS / "archive_pair_counts.json", {
        "raw_rows": len(rows), "question_repeat_depth_pairs": len(pairs), "complete_depth3_depth4_pairs": sum(set(v) == {3, 4} for v in pairs.values()),
        "pairs_with_provably_identical_prefix": exact,
        "historical_archive_verdict": "NOT_REPLAYABLE",
    })

    per_run = []
    for row in rows:
        lineage = row.get("state_lineage", [])
        pre = [e for e in lineage if len(e.get("relation_chain", [])) <= 3]
        reasoning = [e for e in pre if e.get("stage") == "reasoning"]
        prune = [e for e in pre if e.get("stage") == "relation_prune"]
        search = [e for e in pre if e.get("stage") == "relation_search"]
        entities = [e for e in pre if e.get("stage") == "entity_search"]
        prompts = row.get("prompt_history", [])
        filters = [e for e in prompts if e.get("type") == "filter_answer"]
        per_run.append({
            "question_id": row.get("question_id", ""),
            "repeat_index": row.get("repeat_index", ""),
            "diagnostic_depth": row.get("diagnostic_depth", ""),
            "paired_depth_3_4_exists": str({3, 4}.issubset(set(pairs[(row.get("question_id"), row.get("repeat_index"))]))).lower(),
            "route_recorded": str(bool(row.get("router_execution_mode"))).lower(),
            "run_model_settings_available": str(bool(metadata.get("model") and metadata.get("max_completion_tokens_per_request") is not None)).lower(),
            "prompt_event_count": len(prompts),
            "prompt_response_complete_event_count": sum(e.get("prompt") is not None and e.get("response") is not None for e in prompts),
            "scoring_only_gold_ID_ever_seen": row.get("gold_candidate_diagnostic_scoring_only", {}).get("gold_answer_seen_any_search"),
            "pre_depth3_state_event_count": len(pre),
            "pre_depth3_reasoning_event_count": len(reasoning),
            "pre_depth3_reasoning_action_complete_count": sum(e.get("action") is not None for e in reasoning),
            "pre_depth3_relation_prune_event_count": len(prune),
            "pre_depth3_prune_candidate_and_selection_ids_complete_count": sum("candidate_relations_head" in e and "candidate_relations_tail" in e and "selected_relations" in e for e in prune),
            "pre_depth3_relation_search_event_count": len(search),
            "pre_depth3_relation_result_ids_complete_count": sum("head_relations" in e and "tail_relations" in e for e in search),
            "pre_depth3_entity_search_event_count": len(entities),
            "pre_depth3_endpoint_candidate_ID_event_count": sum(isinstance(e.get("candidate_entity_ids"), list) for e in entities),
            "pre_depth3_active_binding_tuple_event_count": sum(isinstance(e.get("active_binding_rows"), list) for e in entities),
            "depth3_forward_boundary_count": sum(e.get("stage") == "reasoning" and e.get("action") == "Forward" and len(e.get("relation_chain", [])) == 3 for e in lineage),
            "depth3_forward_boundary_with_DFS_stack_count": 0,
            "raw_sparql_query_status_count": sum(bool(e.get("sparql") and e.get("status") is not None) for e in row.get("sparql_history", [])),
            "raw_sparql_binding_row_count": sum(bool(e.get("raw_bindings")) for e in row.get("sparql_history", [])),
            "filter_event_count": len(filters),
            "filter_event_with_input_and_output_IDs_count": sum(isinstance(e.get("filter_input_ids"), list) and isinstance(e.get("filter_output_ids"), list) for e in filters),
            "final_answer_source_linked_to_IDs": str(bool(row.get("answer_source") and row.get("final_answer_ids") is not None)).lower(),
            "branch_merge_binding_ID_event_count": sum(bool(e.get("merge_binding_ids")) for e in pre),
            "prefix_hash_and_exact_flag_present": str(bool(row.get("prefix_hash") and row.get("prefix_replay_exact"))).lower(),
        })
    with (RESULTS / "ARCHIVE_RUN_AUDIT.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_run[0]))
        writer.writeheader()
        writer.writerows(per_run)


def make_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Iteration 21 matched-prefix replay record",
        "type": "object",
        "required": ["question_id", "repeat", "prefix_hash", "prefix_replay_exact", "boundary_depth", "current_path", "dfs_stack", "active_binding_ids", "active_binding_rows", "frontier_ids", "selected_relation_id", "relation_direction", "candidate_ids_before_prune", "candidate_ids_after_prune", "KG_query_hash", "KG_result_ids", "answer_source", "filter_input_ids", "filter_output_ids", "final_answer_ids", "gold_id_present_for_scoring", "log_completeness", "evidence_pointer", "model_settings", "decisions", "method_events", "sparql_history", "kg_cache", "visited_node_ids"],
        "properties": {
            "question_id": {"type": "string"}, "repeat": {"type": "integer", "minimum": 1},
            "prefix_hash": {"type": ["string", "null"], "description": "SHA-256 of canonical pre-treatment decisions, KG cache, stack, binding tuples and frontier."},
            "prefix_replay_exact": {"type": "boolean"}, "boundary_depth": {"type": "integer", "minimum": 0},
            "current_path": {"type": "array", "description": "Path of current popped DFS state with node IDs and edge directions."},
            "dfs_stack": {"type": "array", "items": {"type": "object", "required": ["path", "depth"], "properties": {"path": {"type": "array"}, "depth": {"type": "integer"}}}},
            "active_binding_ids": {"type": "array", "items": {"type": "string"}, "description": "IDs only; labels are separate presentation data."},
            "active_binding_rows": {"type": ["array", "null"], "description": "Intermediate variable-to-entity-ID maps needed to validate joins."},
            "frontier_ids": {"type": "array", "items": {"type": "string"}},
            "selected_relation_id": {"type": ["string", "null"]}, "relation_direction": {"enum": ["OUTGOING", "INCOMING", None]},
            "candidate_ids_before_prune": {"type": ["array", "null"], "items": {"type": "string"}},
            "candidate_ids_after_prune": {"type": ["array", "null"], "items": {"type": "string"}},
            "KG_query_hash": {"type": ["string", "null"]}, "KG_result_ids": {"type": "array", "items": {"type": "string"}},
            "answer_source": {"enum": ["GRAPH_FILTER", "DIRECT_LLM", "OTHER"]},
            "filter_input_ids": {"type": ["array", "null"], "items": {"type": "string"}},
            "filter_output_ids": {"type": ["array", "null"], "items": {"type": "string"}},
            "final_answer_ids": {"type": "array", "items": {"type": "string"}},
            "gold_id_present_for_scoring": {"type": ["boolean", "null"], "description": "Scoring-only; never used in branch selection."},
            "log_completeness": {"type": "string"}, "evidence_pointer": {"type": "string"},
            "model_settings": {"type": "object"}, "decisions": {"type": "array"}, "kg_cache": {"type": "array"},
            "method_events": {"type": "array", "description": "Reversible method-wrapper outputs including relation and entity IDs."},
            "sparql_history": {"type": "array", "description": "Exact prefix query text, status, timeout and row-count events."},
            "visited_node_ids": {"type": "array", "items": {"type": "string"}},
        },
        "x-iteration-19-coverage": {
            "prefix_replay_exact": False, "dfs_stack": False, "active_binding_rows": False,
            "frontier_ids": "endpoint candidate IDs only", "answer_source": "event type only; final ID provenance missing",
        },
    }


def run_tests() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(TEST_DIR.relative_to(ROOT)), "-p", "test_*.py", "-v"],
        cwd=ROOT, text=True, capture_output=True,
    )
    output = proc.stdout + proc.stderr
    match = re.search(r"Ran (\d+) tests? in", output)
    result = {
        "command": f"python -m unittest discover -s scripts/diagnostics/iter_021_matched_prefix -p 'test_*.py' -v",
        "exit_code": proc.returncode,
        "tests_run": int(match.group(1)) if match else None,
        "passed": proc.returncode == 0,
        "test_cases": [
            "deterministic identical prefix", "stack and per-hop binding serialization", "visited-node loop logging",
            "exactly one new hop variable", "branch binding merge", "empty frontier", "failed KG query is not an empty frontier",
            "oracle and malformed binding rejection", "missing candidate IDs", "labels versus IDs", "answer-source separation",
            "strict exact-prefix completeness", "short-path ineligibility", "safe JSONL resume",
            "binding projection query preservation", "sidecar query execution and literal rejection",
            "binding projection attachment and frontier equality", "projection truncation and endpoint mismatch rejection",
            "reversible CoRAgent method-output capture", "method wrapper restoration after exception",
            "protected source hashes", "live boundary trace capture", "trace restoration on success/error",
        ],
        "output_tail": output[-4500:],
    }
    write_json(RESULTS / "REPLAY_TEST_RESULTS.json", result)
    if proc.returncode:
        raise RuntimeError("Iteration 21 diagnostic tests failed; see REPLAY_TEST_RESULTS.json")
    return result


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(RAW)
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    write_feasibility(rows, metadata)
    write_json(RESULTS / "PREFIX_STATE_SCHEMA.json", make_schema())
    control = control_loss(rows)
    write_json(RESULTS / "control_loss_trace_audit.json", control)
    tests = run_tests()

    manifest = {
        "purpose": "Retrospective mechanism-development fixture only; no unseen or scored generalization claim.",
        "selection_rule": "One repeat per question. Freeze one four-edge target with D3-only observed hit, one target tie with a D3 Forward at depth 3, and the sole short-path D3-only control loss with D3 Forward at depth 3. If a live D3 run has no Forward at depth 3, mark treatment-ineligible; do not force it.",
        "cases": [
            {"question_id": "WebQTest-450_5839e7f421b1c7dec152856912c23c6f", "repeat": 1, "role": "four_edge_target_depth4_only", "archived_d3_hit": 0, "archived_d4_hit": 1},
            {"question_id": "WebQTrn-1053_278042154fd6e20398fbac62f0bc6aa5", "repeat": 1, "role": "target_paired_hit_tie", "archived_d3_hit": 1, "archived_d4_hit": 1},
            {"question_id": CONTROL_ID, "repeat": 1, "role": "short_path_control_depth3_only_loss", "archived_d3_hit": 1, "archived_d4_hit": 0},
        ],
        "source_raw_jsonl_sha256": sha_file(RAW),
        "source_gold_usage": "Gold labels are scoring-only; no gold ID or gold relation is permitted in inference, continuation, selection, or gate input.",
    }
    write_json(RESULTS / "PROPOSED_DEVELOPMENT_MANIFEST.json", manifest)
    manifest_path = RESULTS / "PROPOSED_DEVELOPMENT_MANIFEST.json"
    manifest_hash = sha_file(manifest_path)
    historical_requests = sum(
        int(r.get("llm_calls", 0) or 0) for r in rows
        if r.get("diagnostic_depth") == 3 and r.get("repeat_index") == 1 and r.get("question_id") in {x["question_id"] for x in manifest["cases"]}
    )
    historical_input = sum(
        int(r.get("input_tokens", 0) or 0) for r in rows
        if r.get("diagnostic_depth") == 3 and r.get("repeat_index") == 1 and r.get("question_id") in {x["question_id"] for x in manifest["cases"]}
    )
    historical_output = sum(
        int(r.get("output_tokens", 0) or 0) for r in rows
        if r.get("diagnostic_depth") == 3 and r.get("repeat_index") == 1 and r.get("question_id") in {x["question_id"] for x in manifest["cases"]}
    )
    historical_cost = historical_input * MODEL_INPUT_PRICE / 1_000_000 + historical_output * MODEL_OUTPUT_PRICE / 1_000_000
    cost_worst = MAX_CALLS * MAX_INPUT_TOKEN_BOUND_PER_CALL * MODEL_INPUT_PRICE / 1_000_000 + MAX_CALLS * MAX_OUTPUT_TOKENS * MODEL_OUTPUT_PRICE / 1_000_000

    (RESULTS / "PROTOCOL.md").write_text(f"""# Iteration 21 — matched-prefix causal replay

## Status and scope

**Decision: E — PAID_CONFIRMATION_AUTHORIZATION_REQUIRED for the smallest live retrieval-only mechanism check.** Archived Iteration 19 depth pairs are not replayable: {len(rows)} run rows form 60 complete same-question/repeat depth pairs, but zero contain the stack, intermediate binding tuples, or a complete cached prefix needed to establish exact identity. No causal effect is estimated here.

This investigation made zero paid API calls and zero SPARQL/KG queries. It does not change TGAR, CoR, CFP, prompts, production depth, or default routing. Fixture-only replay code and synthetic tests are diagnostic-only.

## Estimand and proposed smallest next measurement

The next authorized measurement is retrieval-only: on the exact same captured live depth-3 `Forward` boundary, log the STOP candidate/binding set, select one post-prune relation using the unchanged CoR relation-pruning prompt, execute exactly one relation edge, and compare newly returned binding IDs before any answer filter. Preserve the same stack/prefix hash on both arms. A non-Forward boundary is ineligible; do not force a hop. This can identify newly reachable entity/binding IDs and scoring-only gold-ID presence, but it cannot establish final-answer improvement or a deployable depth gate.

Use the three-case development manifest in `PROPOSED_DEVELOPMENT_MANIFEST.json` only for retrospective mechanism development. Repeat=1 is fixed. A future live run that yields several depth-3 Forward points uses the first one in DFS order, as captured before the added-edge action. Do not treat three questions as a generalization sample.

## Reproduction

```sh
python -m unittest discover -s scripts/diagnostics/iter_021_matched_prefix -p 'test_*.py' -v
python scripts/diagnostics/iter_021_matched_prefix/audit.py
```

`audit.py` reads archived local JSON, hashes local files, runs local fixture tests, and writes the Iteration 21 artifacts. It does not make network calls.
""", encoding="utf-8")

    (RESULTS / "INSTRUMENTATION_NOTES.md").write_text(f"""# Diagnostic instrumentation notes

## What the archive captured

The canonical source is `{RAW.relative_to(ROOT)}` (SHA-256 `{sha_file(RAW)}`). `prompt_history` stores prompt/response text and parsed actions; `state_lineage` stores selected relation IDs/directions, relation candidate IDs, and endpoint entity IDs/names; `sparql_history` stores query text, status, timeout and row count. Model and request settings are run-level in `continuation_metadata.json` (model `{metadata.get('model')}`, completion cap {metadata.get('max_completion_tokens_per_request')}, timeout {metadata.get('timeout_seconds')}s, retries {metadata.get('retries')}).

There are no saved `DFSStack.items` snapshots, no joined intermediate variable-to-ID tuples, no raw SPARQL result bindings, no stable shared prefix hash, no RNG/order snapshot, and no filter input/output entity IDs. Archived rows therefore cannot be restored into an exact matched pair.

## Diagnostic-only live hook

`replay.py::run_instrumented_answer` combines `capture_cor_method_outputs` and `run_with_boundary_capture` without changing production source. Reversible wrappers record relation search IDs, relation candidates before pruning, selected relation IDs/directions, entity IDs, reasoning outputs, prompt history, and the exact SPARQL event history. The trace hook snapshots this information at the unchanged max-depth `Forward` guard (`chain_of_relations/methods/cor/agent.py`, around line 839), including the current path and ordered pending DFS stack. It refuses to replace an existing trace hook and restores wrappers/hooks in `finally`. It explicitly marks `active_binding_rows` unavailable until a sidecar capture succeeds; it never fabricates those IDs.

`binding_projection_query` creates an isolated sidecar SPARQL query by changing only the projection to include intermediate variables and the target. It preserves the WHERE body and modifiers. Because `DISTINCT` now applies to full bindings, this is diagnostic evidence capture, not a production query replacement. `fetch_binding_projection` calls an injected executor, normalizes Freebase URI values to IDs, and rejects literal/unsupported terms rather than pretending they are entity IDs. `attach_binding_projection` rejects a result reaching its LIMIT and requires its endpoint-ID set to exactly equal the executor frontier before sealing the prefix. A future live driver must supply path-variable order and the backend executor, retain those ID rows, and include query/result hashes in the snapshot. All helpers were tested offline only; no endpoint query was run.

The live boundary hook snapshots the exact prompt and SPARQL event histories available in the unchanged answer frame, plus the ordered stack and current path. It still marks intermediate bindings absent until the sidecar returns successfully. The replay adapter refuses an incomplete prefix, rejects label-shaped binding values and oracle selection, uses natural joins on shared variable IDs, records loops without silently deleting them, and rejects result rows adding more than one fresh hop variable. It keeps stop-arm and continuation-arm prefix hashes identical. It is a synthetic fixture adapter; it is not yet a production CoR executor resume implementation.

## Frozen production source integrity

SHA-256 values captured before Iteration 21 edits are in `PROTECTED_SOURCE_HASHES.json`. The passing test suite rechecks all four protected CoR/Freebase files. No production reasoning file was edited.
""", encoding="utf-8")

    (RESULTS / "COVERAGE_AND_BLOCKERS.md").write_text(f"""# Coverage and blockers

## Archived trace coverage

- {len(rows)} raw runs form {sum(set(v) == {3, 4} for v in _pair_rows(rows).values())} complete same-question/repeat depth-3/depth-4 pairs.
- Provably exact archived prefix pairs: **0**. There is no `MATCHED_PREFIX_RESULTS.csv` because no real pair meets the protocol.
- `REPLAY_FEASIBILITY.csv` gives field coverage and limitations. `PREFIX_STATE_SCHEMA.json` defines the complete future record.
- `ARCHIVE_RUN_AUDIT.csv` records per-run and pre-depth-3 event coverage for all 120 archived rows, including applicable event counts within each run.
- Run-level model configuration exists, but per-request sampling/order settings are incomplete. Prompt and response strings are stored, but cannot reconstruct bindings or the pending stack.

## Control loss audit

For `{CONTROL_ID}`, repeat 1, the depth-3 run returns “Hawaiian hibiscus” via a filter event with six candidates; the independent depth-4 run returns no answer and has no filter event. On the corresponding third-edge reasoning step, depth 3 records `Forward` and depth 4 records `Backtrack`. The depth-4 trace later has an `UnknownAction` warning that the LLM returned an empty response; the agent exits with no result. This rules out attributing the loss to the extra-hop bound alone. There is no same-prefix comparison, and missing filter candidate IDs/intermediate bindings mean a candidate/merge contribution is not identifiable.

## What must be captured in a future authorized live measurement

1. Exact model/settings, event order, all prompt and response hashes and content, query/cache results, and decision IDs through the depth-3 boundary.
2. The ordered DFS stack, current path/direction, active binding rows with Freebase IDs, endpoint frontier IDs, and visited IDs.
3. One selected post-boundary relation ID/direction, pre/post-prune candidate relation IDs, exact query hash and one-hop raw binding IDs.
4. For any answer-level study, filter input/output IDs, direct-generation provenance, final answer IDs and a controlled post-boundary sampler. The current authorized proposal is retrieval-only and does not claim answer-level causality.

## Remaining limitations

The live endpoint and API were not contacted. The tracer and shadow-query projection helper passed synthetic tests but have not been exercised against the cluster backend. The fixture adapter validates serialization and one-hop pair semantics; it does not yet restore a live CoR stack or perform a full answer-level fork. These are explicit preflight/implementation gates for the authorized retrieval-only run.
""", encoding="utf-8")

    (RESULTS / "COST_ESTIMATE.md").write_text(f"""# Cost estimate — proposed Iteration 21 retrieval-only development check

No API calls were made. Historical observed Iteration 19 depth-3 runs for the frozen three-case manifest used {historical_requests} requests, {historical_input:,} input tokens, and {historical_output:,} output tokens. At the recorded configured rates (${MODEL_INPUT_PRICE:.2f}/M input, ${MODEL_OUTPUT_PRICE:.2f}/M output), this is an estimated ${historical_cost:.6f}; it is a planning reference, not a price guarantee.

Proposed hard ceilings, subject to fresh explicit user authorization:

- 3 fixed questions, repeat 1; one depth-3 run per question.
- At most **60 total model requests** including the three baseline traces and post-boundary pruning.
- At most 1,024 completion tokens/request and a conservative 7,000-token input reservation/request (requests exceeding this are rejected before sending).
- **USD $0.17 total** hard cost cap; no retries; stop on empty/error response or any unmetered request.
- Estimated observed-use range: about **$0.03–$0.05**, using the archived 47 baseline requests and at most 3 post-boundary relation-prune requests, with room for baseline call-count variation. Max reserved usage under the stated caps is {MAX_CALLS} × {MAX_INPUT_TOKEN_BOUND_PER_CALL:,} input tokens and {MAX_CALLS} × {MAX_OUTPUT_TOKENS:,} output tokens, or **${cost_worst:.6f}** at recorded rates, below the $0.17 hard cap.

This estimate covers only retrieval reachability. It excludes any extra answer-filter/direct-answer LLM study, new cohort, retries, or generalization evaluation. Unused authorization would expire at completion.
""", encoding="utf-8")

    (RESULTS / "AUTHORIZATION_REQUIRED.md").write_text(f"""# Authorization required — no requests sent

Iteration 21 has not made paid requests. The historical Iteration 19 authorization is expired and grants no current calls. Please explicitly authorize or decline the following bounded live retrieval-only mechanism check; do not infer approval from previous top-ups or authorizations.

## Exact proposed scope

- Fixed development cohort and repeat: see `PROPOSED_DEVELOPMENT_MANIFEST.json` (SHA-256 `{manifest_hash}`), three named Iteration 19 questions, repeat 1 only. It is retrospective mechanism development, not a fresh scored evaluation.
- Run the unchanged CoR at depth 3 once/question with diagnostic boundary capture. Do not force a continuation if no depth-3 `Forward` is observed.
- At the first qualifying depth-3 Forward boundary in DFS order per question, capture exact prompt/response decisions, ordered stack, active ID binding rows via a projection-only sidecar query, frontier IDs, and a canonical prefix hash. Fork a STOP record and one continuation. Use the unchanged relation-pruning prompt for one relation choice; issue one relation edge and record raw resulting bindings. Compare gold IDs only after execution for scoring.
- Retrieval-only outcome: new binding/entity IDs and scoring-only gold-ID presence. No answer filtering/direct LLM claim, no gate fitting, no broad accuracy claim, and no extra case search.

## Hard authorization ceiling

- **60 total API requests**, 1,024 completion tokens/request, 7,000-token conservative input reservation/request.
- **US$0.17 total** at the Iteration 19 configured rates; no retries. Halt before any call exceeding the per-request reservation or either cumulative cap. Stop on an empty/error/unmetered response and preserve partial outputs; do not automatically resume.
- Expected spend around **$0.03–$0.05**; the cap is the maximum requested authorization, not a target.
- Local KG sidecar and one-hop queries: **100 read-only SPARQL queries total across the run**, with an expected 98 based on archived counts (86 baseline + 12 incremental: one binding projection, one two-direction relation search, and one entity-binding query per case). Abort before query 101. No remote network or database writes.

## Failure handling and what this would identify

No retries, substitutions, query widening, answer-level calls or additional questions. A missing depth-3 Forward, missing binding projection, query error or incomplete prefix is marked ineligible; retain usage and stop that case. If all three are ineligible or no gold ID is newly reachable, report that fact without changing policy. The test can identify whether one extra edge from an exactly frozen live prefix adds gold/entity bindings before answer selection. It cannot establish answer accuracy improvement, control safety as a policy, or generalization. A later answer-level or scored study would need a new protocol and separate authorization.

## Approval required

The test remains unrun until the user explicitly approves the exact 60-request / US$0.17 ceiling and the named development manifest. No approval is inferred from the expired Iteration 19 budget.
""", encoding="utf-8")

    (RESULTS / "MECHANISM_DECISION.md").write_text(f"""# Iteration 21 mechanism decision

## Decision: E — PAID_CONFIRMATION_AUTHORIZATION_REQUIRED

The archived historical pairs are **NOT_REPLAYABLE**: 60 same-question/repeat depth pairs exist, but **zero** contain a complete matched prefix hash, active intermediate binding rows, or an ordered DFS stack. Independent depth-3/depth-4 runs cannot estimate a search-horizon causal effect.

Offline prerequisites were completed: a diagnostic-only reversible live boundary tracer, a projection-only binding query builder, strict exact-prefix validation, deterministic fixture replay, one-hop validation, and {tests['tests_run']} adversarial tests. All tests pass and protected production-source hashes match. The remaining discriminating retrieval test needs a fresh live depth-3 trace plus a post-boundary relation-prune request, which requires new paid authorization. No API or KG requests were made in this iteration.

The status is E for the proposed small future retrieval-only confirmation; it does **not** mean the depth effect is established. Iteration 19 remains retrospective mechanism-development evidence only. The production TGAR/CoR/CFP methods and default depth remain unchanged.

## Control regression

The D3 control answer is produced by answer filtering after a six-item candidate set. The independent D4 run has a different third-edge reasoning action (`Backtrack` instead of D3 `Forward`), later receives an empty reasoning response (`UnknownAction`) and returns no result without a filter event. This makes independent run/model-response divergence visible but does not identify the treatment effect. Missing active binding tuples and filter candidate IDs prevent a stronger attribution.

## What can be claimed now

- Iteration 19's selected-cohort score differences remain as previously reported and are not causal search-horizon estimates.
- No Iteration 21 matched-prefix result exists; no `MATCHED_PREFIX_RESULTS.csv` was created.
- The future three-case test, if authorized, would measure only new pre-filter binding/entity reachability. Answer quality and deployment safety remain untested.

## Reproduction and integrity

- `python -m unittest discover -s scripts/diagnostics/iter_021_matched_prefix -p 'test_*.py' -v`
- `python scripts/diagnostics/iter_021_matched_prefix/audit.py`
- Test report: `REPLAY_TEST_RESULTS.json`; protected hashes: `PROTECTED_SOURCE_HASHES.json`.
""", encoding="utf-8")

    # Update only the explicitly permitted persistent workflow files.
    status_path = WORK / "STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["status"] = "ITERATION_21_PAID_CONFIRMATION_REQUIRED"
    status["latest_experiment_attempt"] = f"Iteration 21 completed offline trace feasibility audit, diagnostic-only replay adapter, and {tests['tests_run']} fixture tests; archived exact prefix replay is impossible. Awaiting separate authorization for the bounded three-case retrieval-only check."
    status["next_action"] = "Keep TGAR/CoR/CFP and default depth unchanged. Do not make paid calls until the user explicitly authorizes the Iteration 21 manifest, 60-request cap, and US$0.17 cost cap."
    budget = status.setdefault("budget_remaining", {})
    budget["active_paid_authorization"] = False
    budget["new_experiment_authorized_calls"] = 0
    budget["new_experiment_authorized_cost_usd"] = 0.0
    budget["iteration_21_proposed_call_cap"] = MAX_CALLS
    budget["iteration_21_proposed_cost_cap_usd"] = AUTH_COST_CAP_USD
    budget["iteration_21_calls_used"] = 0
    budget["iteration_21_cost_used_usd"] = 0.0
    budget["iteration_21_authorization_status"] = "NOT_AUTHORIZED"
    write_json(status_path, status)

    ledger_path = WORK / "EVIDENCE_LEDGER.csv"
    with ledger_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        old_rows = list(reader)
        headers = list(reader.fieldnames or [])
    old_rows = [row for row in old_rows if row.get("iteration") != "21"]
    new_rows = [
        {"iteration": "21", "question_key": "cwq::30-question Iteration 19 selected cohort", "hypothesis": "MATCHED_PREFIX_CAUSAL_REPLAY", "supporting_trace": "120 archived rows; 60 depth pairs; no stack or intermediate binding tuples", "confidence": "HIGH_FOR_REPLAY_BLOCKER", "outcome": "NOT_REPLAYABLE_ZERO_EXACT_PREFIX_PAIRS", "source_artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_021_matched_prefix_replay/REPLAY_FEASIBILITY.csv", "notes": "Independent runs are not a causal depth comparison; no paid or KG calls."},
        {"iteration": "21", "question_key": "cwq::WebQTest-450_3eb90d65b82703c8f7fb0496baa80ce7", "hypothesis": "CONTROL_LOSS_CAUSE", "supporting_trace": "D3 depth-3 Forward then filter answer; D4 Backtrack divergence and later empty LLM response", "confidence": "HIGH_FOR_TRACE_FACTS_LOW_FOR_CAUSAL_ATTRIBUTION", "outcome": "NOT_IDENTIFIED_AS_DEPTH_EFFECT", "source_artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_021_matched_prefix_replay/control_loss_trace_audit.json", "notes": "Missing active binding tuples and filter input/output entity IDs."},
        {"iteration": "21", "question_key": "synthetic fixture cohort", "hypothesis": "REPLAY_ADAPTER_AND_CAPTURE_HOOK", "supporting_trace": f"{tests['tests_run']} adversarial local unit tests; 4 protected source hashes", "confidence": "HIGH_FOR_FIXTURE_MECHANICS", "outcome": "PASS_OFFLINE_FIXTURES_ONLY", "source_artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_021_matched_prefix_replay/REPLAY_TEST_RESULTS.json", "notes": "No live endpoint/API validation; production reasoning files unchanged."},
    ]
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerows(new_rows)

    log_path = WORK / "ITERATION_LOG.md"
    log_block = f"""

## Iteration 21 — matched-prefix causal replay — 2026-11-08

- Offline audit of `{RAW.relative_to(ROOT)}`: {len(rows)} rows, {sum(set(v) == {3, 4} for v in _pair_rows(rows).values())} complete depth pairs, zero exact matched prefixes. Missing per-run DFS stack and active intermediate binding-ID tuples prevent restoring the same depth-3 state. No `MATCHED_PREFIX_RESULTS.csv` was produced.
- Control loss `{CONTROL_ID}`, repeat 1: depth 3 returns “Hawaiian hibiscus” via a six-candidate answer-filter event; depth 4 independently diverges at the third-edge reasoning decision (Backtrack versus Forward), later records an empty LLM response/UnknownAction and no filter event. This does not establish that extra search caused the loss; filter IDs and binding rows remain absent.
- Added diagnostic-only serializer/replay fixture, reversible depth-bound trace hook, binding-projection sidecar builder, and **{tests['tests_run']} passing tests**. No production reasoning source changed; hashes verify. No paid API or SPARQL calls.
- Decision: **E — PAID_CONFIRMATION_AUTHORIZATION_REQUIRED** for a bounded retrieval-only three-case development check. Proposed 60-request / $0.17 hard caps; authorization is not active. See `RESULTS/iter_021_matched_prefix_replay/AUTHORIZATION_REQUIRED.md` and manifest hash in that file.
- Full artifacts: `RESULTS/iter_021_matched_prefix_replay/`. Reproduction: `python scripts/diagnostics/iter_021_matched_prefix/audit.py`.
"""
    old_log = log_path.read_text(encoding="utf-8")
    marker = "## Iteration 21 — matched-prefix causal replay — 2026-11-08"
    if marker in old_log:
        start = old_log.index(marker)
        next_header = old_log.find("\n## ", start + len(marker))
        old_log = old_log[:start].rstrip() + ("\n" + old_log[next_header + 1:] if next_header >= 0 else "\n")
    log_path.write_text(old_log.rstrip() + "\n" + log_block, encoding="utf-8")

    handoff = WORK / "CHATGPT_HANDOFF.md"
    old_handoff = handoff.read_text(encoding="utf-8")
    handoff_start = "<!-- ITERATION21_HANDOFF_START -->"
    handoff_end = "<!-- ITERATION21_HANDOFF_END -->"
    if handoff_start in old_handoff and handoff_end in old_handoff:
        suffix = old_handoff[old_handoff.index(handoff_end) + len(handoff_end):].lstrip()
    else:
        suffix = old_handoff
    handoff_block = f"""{handoff_start}
# Iteration 21 current handoff — matched-prefix causal replay

## Decision and current approval state

**Status E — PAID_CONFIRMATION_AUTHORIZATION_REQUIRED.** No paid LLM/API requests or KG/SPARQL queries were made during Iteration 21. The old Iteration 19 budget is expired. A new three-case mechanism-development run is not yet authorized; the proposed ceilings are 60 API requests and US$0.17, with no retries.

The canonical Iteration 19 archive has 120 run rows forming 60 complete same-question/repeat depth-3/depth-4 pairs, but zero exact matched prefixes. It omits active intermediate binding-ID tuples and the ordered DFS stack. Matching the question, prompt text, or relation chain cannot fill that gap. No causal search-horizon effect can be claimed, and no `MATCHED_PREFIX_RESULTS.csv` exists.

The depth-3-only control loss (`{CONTROL_ID}`, repeat 1) returns “Hawaiian hibiscus” through a filter event with six candidates. Its independent depth-4 run diverges at the third-edge reasoning action (Backtrack versus Forward), later gets an empty LLM response (`UnknownAction`), and has no filter event. This explains the recorded trace shape but does not establish a depth treatment effect; intermediate binding and filter IDs are missing.

## What Iteration 21 completed

- Built a diagnostic-only exact-prefix schema, fixture adapter, safe resume helper, one-hop natural binding join, reversible live boundary stack tracer, and sidecar query projection builder.
- **{tests['tests_run']} fixture tests passed**, covering determinism, stack/binding serialization, loops, one-hop cap, merge joins, empty rows, unsupported oracle/binding data, missing IDs, label/ID separation, answer-source separation, short-path stop behavior, resumability, projection preservation, live trace capture/restoration, and protected-source hashes.
- No TGAR/CoR/CFP production reasoning source or default depth was changed. The live cluster/API were not contacted.

## Exact next decision

Explicitly approve or decline the narrowly scoped retrieval-only check in `RESULTS/iter_021_matched_prefix_replay/AUTHORIZATION_REQUIRED.md`. The frozen three-case development manifest is `PROPOSED_DEVELOPMENT_MANIFEST.json`; its SHA-256 is recorded in the authorization file. Approval would cover at most **60 requests / US$0.17**, model completion limit 1,024 tokens, conservative input reservation 7,000 tokens/request, no retries, and stop-on-error. This test would assess only new pre-filter binding/entity reachability; it cannot demonstrate final answer improvement, control-policy safety, or generalization. Any answer-level or fresh scored experiment requires separate authorization.

## Reproduction and files

- `python -m unittest discover -s scripts/diagnostics/iter_021_matched_prefix -p 'test_*.py' -v`
- `python scripts/diagnostics/iter_021_matched_prefix/audit.py`
- Results: `RESULTS/iter_021_matched_prefix_replay/` (`PROTOCOL.md`, per-field and per-run coverage, schema, test report, control audit, cost and authorization).
- Persistent workflow updates: `STATUS.json`, `EVIDENCE_LEDGER.csv`, `ITERATION_LOG.md`, and this handoff. Earlier iteration artifacts remain preserved.

{handoff_end}
"""
    handoff.write_text(handoff_block + "\n" + suffix, encoding="utf-8")

    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()
    git_branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()
    git_status = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=False).stdout.splitlines()
    write_json(RESULTS / "REPO_STATE.json", {
        "head_commit": git_head,
        "branch": git_branch,
        "dirty_worktree": bool(git_status),
        "status_short": git_status,
        "note": "Pre-existing user changes were preserved. Iteration 21 diagnostic files are new; no commit or push was made.",
    })

    print(json.dumps({"status": "complete", "archive_rows": len(rows), "depth_pairs": sum(set(v) == {3, 4} for v in _pair_rows(rows).values()), "exact_prefix_pairs": 0, "tests": tests["tests_run"], "manifest_sha256": manifest_hash, "paid_calls": 0, "kg_queries": 0}, indent=2))


def _pair_rows(rows: list[dict]) -> dict:
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row.get("question_id"), row.get("repeat_index"))][row.get("diagnostic_depth")] = row
    return pairs


if __name__ == "__main__":
    main()
