#!/usr/bin/env python3
"""Summarize Iteration 30 failure partitions against retained runtime trace shape."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS"
ITER30 = RESULTS / "iter_030_results_first"
RUN = RESULTS / "iter_028_terminal_boundary_repair/RUNS/iter29_terminal_candidate_selector_attempt3"
PRUNER = ROOT / "chain_of_relations/tools/relation_prune.py"
OUT = RESULTS / "iter_033_scalar_generalization_and_upstream_decision/UPSTREAM_EVIDENCE_AUDIT.json"
EXPECTED_MISS_HASH = "486197ce21f81a297e67203a0cdd6d1be53e3eef1fc5476c59bef478f06edc95"
EXPECTED_TRACE_HASH = "17bdae03a002a541d329f37b3e87702fd7535032f6dbf9989b8f51d3f3bb272f"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    miss_path = ITER30 / "ENTITY_MISS_TRACE_AUDIT.csv"
    trace_path = RUN / "CASE_TRACES_NON_GOLD.jsonl"
    miss_hash, trace_hash = digest(miss_path), digest(trace_path)
    if miss_hash != EXPECTED_MISS_HASH or trace_hash != EXPECTED_TRACE_HASH:
        raise SystemExit("Iteration 30 audit or archived runtime trace hash changed")
    with miss_path.open(encoding="utf-8-sig", newline="") as stream:
        misses = list(csv.DictReader(stream))
    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(misses) != 154 or len(traces) != 800:
        raise SystemExit("unexpected frozen entity-miss or trace row count")

    partition = Counter(row["failure_partition"] for row in misses)
    first_boundary = Counter(row["first_failure_boundary"] for row in misses)
    absent = [row for row in misses if row["failure_partition"] == "GOLD_ABSENT_FROM_RECORDED_CANDIDATE_POOL"]
    absent_boundary = Counter(row["first_failure_boundary"] for row in absent)
    expected_partition = {
        "GOLD_ABSENT_FROM_RECORDED_CANDIDATE_POOL": 84,
        "NO_ENTITY_CANDIDATE_POOL": 39,
        "GOLD_IN_RECORDED_CANDIDATE_POOL": 31,
    }
    expected_absent = {
        "FIRST_REQUIRED_RELATION_OMITTED_BY_RELATION_PRUNING": 27,
        "FIRST_REQUIRED_RELATION_ABSENT_FROM_LOGGED_INITIAL_PROMPT_CANDIDATES": 1,
        "SELECTED_FIRST_RELATION_NOT_EXECUTED_AS_REQUIRED_PATH_EDGE": 42,
        "LATER_PATH_STOP_JOIN_OR_TERMINAL_QUERY_DIVERGENCE": 14,
    }
    if dict(partition) != expected_partition or dict(absent_boundary) != expected_absent:
        raise SystemExit(f"frozen Iteration 30 partitions changed: {partition}, {absent_boundary}")

    relation_searches = []
    relation_prunes = []
    trace_keys: set[str] = set()
    step_keys: set[str] = set()
    sparql_keys: set[str] = set()
    for trace in traces:
        trace_keys.update(trace.keys())
        for item in trace.get("step_history", []):
            if isinstance(item, dict):
                step_keys.update(item.keys())
                if item.get("search_type", "").startswith("relation_search"):
                    relation_searches.append(item)
        for item in trace.get("sparql_history", []):
            if isinstance(item, dict):
                sparql_keys.update(item.keys())
        for item in trace.get("prompt_history", []):
            if isinstance(item, dict) and item.get("type") == "relation_prune":
                relation_prunes.append(item)

    relation_search_row_payload_fields = sorted(
        key for key in step_keys | sparql_keys
        if any(token in key.casefold() for token in ("binding", "relation_ids", "relation_rows", "returned_relations", "result_rows"))
    )
    relation_search_schema = sorted(step_keys)
    relation_prune_schema = sorted({key for item in relation_prunes for key in item})
    no_returned_relation_rows = not any(
        key in step_keys | sparql_keys
        for key in ("candidate_relation_ids", "returned_relation_rows", "result_bindings", "bindings")
    )
    no_step_binding_tuples = not any("binding" in key.casefold() for key in step_keys | sparql_keys)
    if not relation_searches or not relation_prunes or not no_returned_relation_rows or not no_step_binding_tuples:
        raise SystemExit("trace-shape assumptions no longer hold; review upstream evidence before decision")

    output: dict[str, Any] = {
        "status": "UPSTREAM_INTERVENTION_NOT_READY",
        "iteration": 33,
        "gold_used_only_for_failure_partition_audit": True,
        "runtime_source_calls": 0,
        "api_requests": 0,
        "sparql_attempts": 0,
        "source_hashes": {
            str(miss_path.relative_to(ROOT)): miss_hash,
            str(trace_path.relative_to(ROOT)): trace_hash,
            str(PRUNER.relative_to(ROOT)): digest(PRUNER),
        },
        "entity_miss_partition": dict(partition),
        "entity_miss_first_failure_boundary": dict(first_boundary),
        "gold_absent_first_boundary": dict(absent_boundary),
        "retained_runtime_trace_shape": {
            "trace_rows": len(traces),
            "relation_search_events": len(relation_searches),
            "relation_prune_events": len(relation_prunes),
            "step_history_keys": relation_search_schema,
            "relation_prune_event_keys": relation_prune_schema,
            "relation_searches_store_query_text_status_and_counts_but_not_returned_relation_rows": no_returned_relation_rows,
            "relation_searches_store_active_binding_tuples": not no_step_binding_tuples,
            "selected_relation_ids_include_explicit_direction": any(
                isinstance(selected, dict) and any(key in selected for key in ("direction", "left", "is_head"))
                for item in relation_prunes
                for selected in (item.get("parsed_result") if isinstance(item.get("parsed_result"), list) else [])
            ),
            "relation_candidates_survive_only_as_prompt_text_and_untyped_parsed_id_lists": True,
        },
        "implementation_check": {
            "current_relation_pruner_maps_selected_ids_by_candidate_membership": True,
            "if_relation_id_occurs_in_both_head_and_tail_pools_head_membership_wins": True,
            "evidence_sufficient_for_safe_behavior_change": False,
            "reason": "The gold-only boundary counts identify where paths diverged but do not identify a unique gold-free cause. Archived relation queries retain only query text/status/count, prompts retain candidates as directional text, parsed selections omit direction, and traces omit intermediate binding tuples. A repair based on gold first edges would be oracle leakage; a logging-only addition would not be a validated correction.",
        },
        "decision": "Stop this upstream intervention path until a genuinely independent study has complete runtime relation-row, selected-direction, and active-binding provenance or another gold-free mechanism source. No production change or optional behavior flag was added.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "partition": dict(partition), "gold_absent_first_boundary": dict(absent_boundary), "relation_search_events": len(relation_searches), "relation_prune_events": len(relation_prunes), "output": str(OUT.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
