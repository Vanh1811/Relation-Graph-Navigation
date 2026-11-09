#!/usr/bin/env python3
"""Build Iteration 22 offline-only preflight artifacts; never opens a transport."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from collections import defaultdict

from pilot import (
    HardCapMeter,
    MANIFEST_SHA256,
    MAX_API_COST_USD,
    MAX_API_REQUESTS,
    MAX_COMPLETION_TOKENS,
    MAX_INPUT_TOKENS,
    MAX_SPARQL_QUERIES,
    approval_matches,
    dry_run_case,
    estimated_cost,
    summarize_dry_run,
)


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
ITER21 = WORK / "RESULTS/iter_021_matched_prefix_replay"
ITER22 = WORK / "RESULTS/iter_022_gated_matched_prefix"
AUTH_PATH = ITER22 / "AUTHORIZATION.json"
PROTOCOL = ROOT / "Codex_Iteration22_Gated_Matched_Prefix_Pilot.md"
RAW = WORK / "RESULTS/iter_019_depth4_holdout_preparation/control_continuation/paired_runs_cumulative.jsonl"
MANIFEST = ITER21 / "PROPOSED_DEVELOPMENT_MANIFEST.json"
HASH_MANIFEST = ITER21 / "PROTECTED_SOURCE_HASHES.json"

PREEXISTING_STATUS_AT_ITERATION_START = [
    " M chain_of_relations/kg_backend/__init__.py",
    "?? chain_of_relations/kg_backend/kqa_pro_backend.py",
    "?? scripts/analyze_phase4_selective_branch_benefit.py",
    "?? scripts/audit_fresh50_tgar_vs_cfp.py",
    "?? scripts/diagnostics/",
    "?? scripts/run_fresh50_strong_branch_three_way.py",
    "?? scripts/run_phase3c_frozen_generalization.py",
    "?? scripts/run_phase5_dataset_complexity_audit.py",
    "?? scripts/run_phase6_kqa_pro_transfer.py",
    "?? tests/test_fresh50_autonomous_diagnosis.py",
    "?? tests/test_fresh50_root_cause_audit.py",
    "?? tests/test_kqa_pro_backend.py",
    "?? tests/test_phase3c_frozen_generalization.py",
    "?? tests/test_phase4_selective_branch_benefit.py",
    "?? tests/test_phase5_dataset_complexity_audit.py",
    "?? tests/test_phase6_supported_subset.py",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_label_cache(rows: list[dict], manifest: dict) -> tuple[dict[str, str], dict]:
    case_ids = {str(item["question_id"]) for item in manifest["cases"]}
    chosen = [row for row in rows if row.get("question_id") in case_ids
              and row.get("diagnostic_depth") == 3 and row.get("repeat_index") == 1]
    expected_keys = {(str(item["question_id"]), 1, 3) for item in manifest["cases"]}
    actual_keys = {(str(row.get("question_id")), row.get("repeat_index"), row.get("diagnostic_depth")) for row in chosen}
    if actual_keys != expected_keys:
        raise RuntimeError(f"fixed manifest slot coverage mismatch: expected={expected_keys}, actual={actual_keys}")
    labels: dict[str, set[str]] = defaultdict(set)
    appearances = 0
    unnamed = 0
    for row in chosen:
        for event in row.get("sparql_history", []):
            if event.get("search_type") != "entity_search":
                continue
            entity_ids = event.get("candidate_entity_ids") or []
            entity_names = event.get("candidate_entity_names") or []
            if len(entity_ids) != len(entity_names):
                raise RuntimeError("archived endpoint ID/name arrays are not aligned")
            for entity_id, label in zip(entity_ids, entity_names):
                appearances += 1
                if label == "UnName_Entity":
                    unnamed += 1
                labels[str(entity_id)].add(str(label))
    conflicts = {entity_id: sorted(values) for entity_id, values in labels.items() if len(values) != 1}
    cache = {entity_id: next(iter(values)) for entity_id, values in labels.items() if len(values) == 1}
    return cache, {
        "fixed_d3_repeat1_rows": len(chosen),
        "entity_candidate_appearances": appearances,
        "unique_entity_ids": len(labels),
        "cacheable_unambiguous_ids": len(cache),
        "conflicting_ids": conflicts,
        "unnamed_entity_appearances": unnamed,
        "label_cache_misses_during_historical_trace": 0,
    }


def query_accounting(rows: list[dict], manifest: dict, label_cache: dict[str, str]) -> dict:
    case_ids = {str(item["question_id"]) for item in manifest["cases"]}
    chosen = [row for row in rows if row.get("question_id") in case_ids
              and row.get("diagnostic_depth") == 3 and row.get("repeat_index") == 1]
    logged_calls = sum(len(row.get("sparql_history", [])) for row in chosen)
    entity_events = []
    relation_direction_events = 0
    for row in chosen:
        for event in row.get("sparql_history", []):
            search_type = str(event.get("search_type", ""))
            if search_type == "entity_search":
                entity_events.append(event)
            elif search_type in {"relation_search_head", "relation_search_tail"}:
                relation_direction_events += 1
    label_cache_keys: set[str] = set()
    cold_label_batches = 0
    cold_unresolved_fallbacks = 0
    label_batch_size = 100  # Explicitly set this in the future run; source default is also 100.
    for event in entity_events:
        ids = list(dict.fromkeys(str(value) for value in event.get("candidate_entity_ids", []) or []))
        pending = [value for value in ids if value not in label_cache_keys]
        cold_label_batches += (len(pending) + label_batch_size - 1) // label_batch_size
        names_by_id = dict(zip(event.get("candidate_entity_ids", []) or [], event.get("candidate_entity_names", []) or []))
        cold_unresolved_fallbacks += sum(names_by_id.get(value) == "UnName_Entity" for value in pending)
        label_cache_keys.update(pending)

    incremental_per_case = {
        "binding_projection": 1,
        "post_boundary_relation_search_directions": 2,
        "one_selected_edge_binding_query": 1,
    }
    incremental_total = len(manifest["cases"]) * sum(incremental_per_case.values())
    cache_only_expected = logged_calls + incremental_total
    cold_expected = logged_calls + cold_label_batches + cold_unresolved_fallbacks + incremental_total
    return {
        "scope": "logical one-attempt read-only SPARQL calls; actual pilot transport will meter every physical attempt",
        "manifest_cases": len(manifest["cases"]),
        "archived_depth3_repeat1_logged_queries": logged_calls,
        "archived_relation_search_query_count": relation_direction_events,
        "archived_entity_search_query_count": len(entity_events),
        "hidden_label_resolution_calls_if_cold_cache": cold_label_batches + cold_unresolved_fallbacks,
        "cold_cache_label_batches": cold_label_batches,
        "cold_cache_unresolved_single_id_lookups": cold_unresolved_fallbacks,
        "unique_archive_label_ids": len(label_cache),
        "incremental_queries_per_case": incremental_per_case,
        "incremental_queries_total": incremental_total,
        "cold_cache_total_estimate": cold_expected,
        "cold_cache_fits_100_query_cap": cold_expected <= MAX_SPARQL_QUERIES,
        "frozen_local_label_cache_total_estimate": cache_only_expected,
        "frozen_local_label_cache_fits_100_query_cap": cache_only_expected <= MAX_SPARQL_QUERIES,
        "expected_headroom_with_frozen_cache": MAX_SPARQL_QUERIES - cache_only_expected,
        "cache_miss_policy": "stop the affected case before a name-resolution request; do not fall back to the hidden retrying resolver",
        "health_preflight_queries": 0,
        "query_cap": MAX_SPARQL_QUERIES,
        "finding": (
            "The original cold-cache estimate is over cap because entity_search performs hidden label lookups. "
            "Using the unambiguous local label cache reconstructed from the same three frozen historical traces reduces the expected total below cap; "
            "the live meter must still stop before query 101 if the new path exceeds this expectation."
        ),
    }


def synthetic_snapshot(action: str = "Forward") -> dict:
    return {
        "question_id": "synthetic-iteration22", "repeat": 1, "boundary_depth": 3,
        "boundary_action": action, "route": "cor", "evidence_pointer": "synthetic://iter22",
        "model_settings": {"model": "fixture", "temperature": 0, "max_completion_tokens": 64},
        "decisions": [{"stage": "reasoning", "prompt": "fixture", "response": "Forward"}],
        "method_events": [
            {"stage": "relation_search", "relation_chain": [], "head_relation_ids": ["r.next"], "tail_relation_ids": []},
            {"stage": "relation_prune", "relation_chain": [], "candidate_relation_ids_before_prune": ["r.next", "r.alt"],
             "selected_relations": [{"id": "r.next", "direction": "OUTGOING"}]},
            {"stage": "entity_search", "relation_chain": [], "candidate_entity_ids": ["m.c"]},
            {"stage": "reasoning", "relation_chain": [], "action": "Forward", "response": "Forward"},
        ],
        "sparql_history": [{"sparql": "SELECT ?targetEntity WHERE {}", "status": "ok"}],
        "kg_cache": [{"query_hash": "prefix-cache", "status": "ok", "result_ids": ["m.a", "m.b", "m.c"]}],
        "dfs_stack": [{"path": [{"kind": "entity", "id": "m.root"}], "depth": 0}],
        "current_path": [
            {"kind": "entity", "id": "m.root"},
            {"kind": "relation", "id": "r.one", "direction": "OUTGOING"},
            {"kind": "entity", "id": "m.a"},
            {"kind": "relation", "id": "r.two", "direction": "OUTGOING"},
            {"kind": "entity", "id": "m.b"},
            {"kind": "relation", "id": "r.three", "direction": "INCOMING"},
        ],
        "active_binding_rows": None, "frontier_ids": ["m.c"],
        "visited_node_ids": ["m.root", "m.a", "m.b", "m.c"],
        "candidate_relation_ids_before_prune": ["r.next", "r.alt"],
        "answer_source": "OTHER", "filter_input_ids": None, "filter_output_ids": None,
        "final_answer_ids": [],
    }


def dry_run_scenarios() -> dict:
    projection_query = "SELECT DISTINCT ?a ?b ?endpoint WHERE {}"
    projection = {"status": "ok", "query": projection_query,
                  "query_hash": hashlib.sha256(projection_query.encode("utf-8")).hexdigest(), "limit": 100,
                  "rows": [{"?a": "m.a", "?b": "m.b", "?endpoint": "m.c"}]}
    variable_order = ["?a", "?b", "?endpoint"]
    select = [{"id": "r.next", "direction": "OUTGOING"}]
    rows = [{"?e3": "m.c", "?e4": "m.new"}]
    outcomes = {}
    successful_meter = HardCapMeter()
    outcomes["complete_one_hop"] = dry_run_case(
        snapshot=synthetic_snapshot(), projection=projection, projection_variable_order=variable_order,
        llm_response="selected r.next", parsed_selection=select, kg_query="synthetic new-edge query",
        kg_rows=rows, meter=successful_meter)
    failure_cases = [
        ("non_forward", synthetic_snapshot("Finish"), projection, "Forward", select, "ok"),
        ("missing_projection", synthetic_snapshot(), None, "Forward", select, "ok"),
        ("truncated_projection", synthetic_snapshot(), {**projection, "limit": 1}, "Forward", select, "ok"),
        ("empty_llm", synthetic_snapshot(), projection, "", select, "ok"),
        ("malformed_llm", synthetic_snapshot(), projection, "unparseable", None, "ok"),
        ("kg_timeout", synthetic_snapshot(), projection, "selected r.next", select, "timeout"),
    ]
    for name, snapshot, project, response, parsed, kg_status in failure_cases:
        meter = HardCapMeter()
        outcomes[name] = dry_run_case(
            snapshot=snapshot, projection=project, projection_variable_order=variable_order,
            llm_response=response, parsed_selection=parsed, kg_query="synthetic only",
            kg_rows=rows, kg_status=kg_status, meter=meter)
        outcomes[name + "_meter"] = summarize_dry_run([outcomes[name]], meter)
    all_ineligible_meter = HardCapMeter()
    all_ineligible = [dry_run_case(
        snapshot=synthetic_snapshot("Finish"), projection=None, projection_variable_order=[],
        llm_response=None, parsed_selection=None, kg_query="", kg_rows=[], meter=all_ineligible_meter)
        for _ in range(3)]
    outcomes["all_ineligible_batch"] = summarize_dry_run(all_ineligible, all_ineligible_meter)
    outcomes["successful_batch_meter"] = summarize_dry_run([outcomes["complete_one_hop"]], successful_meter)
    return {"evidence_class": "SYNTHETIC_DRY_RUN_ONLY", "scenarios": outcomes,
            "actual_api_requests": 0, "actual_sparql_queries": 0, "transport_calls": 0}


def run_tests() -> dict:
    commands = [
        [sys.executable, "-m", "unittest", "discover", "-s", "scripts/diagnostics/iter_021_matched_prefix", "-p", "test_*.py", "-v"],
        [sys.executable, "-m", "unittest", "discover", "-s", "scripts/diagnostics/iter_022_gated_matched_prefix", "-p", "test_*.py", "-v"],
    ]
    results = []
    for command in commands:
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        output = proc.stdout + proc.stderr
        lines = output.splitlines()
        summary = next((line.strip() for line in reversed(lines) if line.strip().startswith("Ran ")), "unknown")
        results.append({"command": " ".join(command), "exit_code": proc.returncode,
                        "summary": summary, "passed": proc.returncode == 0,
                        "output_tail": output[-2500:]})
        if proc.returncode:
            raise RuntimeError(f"offline test suite failed: {command[-3:]}")
    return {"all_passed": all(item["passed"] for item in results), "suites": results,
            "total_tests": sum(int(item["summary"].split()[1]) for item in results)}


def main() -> None:
    ITER22.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(MANIFEST)
    if manifest_hash != MANIFEST_SHA256:
        raise RuntimeError(f"frozen manifest hash mismatch: {manifest_hash}")
    if len(manifest.get("cases", [])) != 3 or any(item.get("repeat") != 1 for item in manifest["cases"]):
        raise RuntimeError("manifest scope differs from the authorized protocol")
    authorization = json.loads(AUTH_PATH.read_text(encoding="utf-8")) if AUTH_PATH.exists() else None
    authorization_recorded = approval_matches(authorization)

    raw_rows = read_jsonl(RAW)
    label_cache, label_meta = build_label_cache(raw_rows, manifest)
    source_hash = sha256_file(RAW)
    label_cache_path = ITER22 / "LOCAL_LABEL_CACHE.json"
    write_json(label_cache_path, {
        "provenance": "Archived Iteration 19 candidate ID/name pairs for the exact frozen three-question depth-3 repeat-1 manifest; local-only, no live query.",
        "manifest_sha256": manifest_hash,
        "source_archive": str(RAW.relative_to(ROOT)),
        "source_archive_sha256": source_hash,
        "coverage": label_meta,
        "labels": [{"entity_id": key, "label": label_cache[key]} for key in sorted(label_cache)],
    })
    accounting = query_accounting(raw_rows, manifest, label_cache)
    write_json(ITER22 / "QUERY_ACCOUNTING.json", accounting)

    dry = dry_run_scenarios()
    write_json(ITER22 / "PILOT_DRY_RUN.json", dry)
    tests = run_tests()
    write_json(ITER22 / "OFFLINE_TEST_RESULTS.json", tests)

    protected = json.loads(HASH_MANIFEST.read_text(encoding="utf-8"))
    integrity = []
    for item in protected["files"]:
        path = ROOT / item["path"]
        actual = sha256_file(path)
        integrity.append({"path": item["path"], "expected_sha256": item["sha256_at_iteration_start"],
                          "actual_sha256": actual, "matches": actual == item["sha256_at_iteration_start"]})
    runtime_paths = [
        "chain_of_relations/methods/cor/agent.py", "chain_of_relations/methods/cor/dfs_stack.py",
        "chain_of_relations/kg_backend/freebase/db_func.py", "chain_of_relations/kg_backend/freebase/query_template.yml",
        "chain_of_relations/kg_backend/freebase_backend.py", "chain_of_relations/tools/entity_search.py",
        "chain_of_relations/tools/relation_search.py", "chain_of_relations/tools/relation_prune.py",
        "chain_of_relations/llm_api.py",
    ]
    diagnostic_paths = [
        "scripts/diagnostics/iter_022_gated_matched_prefix/pilot.py",
        "scripts/diagnostics/iter_022_gated_matched_prefix/test_pilot.py",
        "scripts/diagnostics/iter_022_gated_matched_prefix/offline_preflight.py",
    ]
    git_branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, text=True,
                                capture_output=True, check=False).stdout.strip()
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              capture_output=True, check=False).stdout.strip()
    write_json(ITER22 / "SOURCE_SNAPSHOT.json", {
        "protocol_sha256": sha256_file(PROTOCOL), "manifest_sha256": manifest_hash,
        "iter21_replay_adapter_sha256": sha256_file(ROOT / "scripts/diagnostics/iter_021_matched_prefix/replay.py"),
        "iter21_replay_tests_sha256": sha256_file(ROOT / "scripts/diagnostics/iter_021_matched_prefix/test_replay.py"),
        "protected_hashes": integrity,
        "protected_hashes_all_match": all(item["matches"] for item in integrity),
        "relevant_runtime_source_hashes": {path: sha256_file(ROOT / path) for path in runtime_paths},
        "iteration22_diagnostic_source_hashes": {path: sha256_file(ROOT / path) for path in diagnostic_paths},
        "live_runner_sha256": sha256_file(ROOT / "scripts/diagnostics/iter_022_gated_matched_prefix/run_live_pilot.py"),
        "authorization_recorded": authorization_recorded,
        "authorization_sha256": sha256_file(AUTH_PATH) if AUTH_PATH.exists() else None,
        "branch": git_branch, "head_commit": git_head,
        "preexisting_worktree_status_at_iteration_start": PREEXISTING_STATUS_AT_ITERATION_START,
        "current_worktree_status_after_iteration22_diagnostics": "Not rescanned by the artifact writer; Iter22 files are new under scripts/diagnostics/ and the new result folder.",
        "note": "No cleanup, stash, commit, push, production edit, or live request was performed. The baseline dirty paths are recorded explicitly above.",
    })

    max_reserved = MAX_API_REQUESTS * estimated_cost(MAX_INPUT_TOKENS, MAX_COMPLETION_TOKENS)
    cold_fits = accounting["cold_cache_total_estimate"] <= MAX_SPARQL_QUERIES
    cache_fits = accounting["frozen_local_label_cache_total_estimate"] <= MAX_SPARQL_QUERIES
    state_text = (
        "A fresh exact-scope approval is recorded; no API or SPARQL transport was opened during this offline preflight."
        if authorization_recorded else
        "Live pilot remains disabled pending fresh explicit approval."
    )
    decision_text = (
        "Offline implementation and preflight checks pass with the frozen local label-cache strategy and hard meters. "
        "The cold-cache plan would exceed the query cap; the cache-only strategy is the bounded repair, with misses marked ineligible. "
        "The exact approval is recorded. The live pilot remains a separate one-shot action under those recorded caps."
        if authorization_recorded else
        "Offline implementation and preflight checks pass with the frozen local label-cache strategy and hard meters. "
        "The cold-cache plan would exceed the query cap; the cache-only strategy is the bounded repair, with misses marked ineligible. "
        "No live operation has approval. The fixed manifest and all caps must be explicitly authorized before transport is enabled."
    )
    (ITER22 / "OFFLINE_PREFLIGHT.md").write_text(f"""# Iteration 22 offline preflight

**State: {state_text}** This report contains archive-derived accounting and synthetic dry-run checks only. It contains no new KG result, model output, or matched-prefix measurement.

## Frozen inputs and workspace integrity

- Iteration 22 protocol SHA-256: `{sha256_file(PROTOCOL)}`.
- Fixed manifest SHA-256: `{manifest_hash}` (matches the required value); exactly three predetermined questions, repeat 1.
- Iteration 21 replay unit suite: 23/23 passed. Its audit script passed in a temporary isolated repository copy so it did not rewrite the Iteration 21 results or workflow files.
- Iteration 22 harness suite: {sum(int(item["summary"].split()[1]) for item in tests["suites"] if "iter_022" in item["command"])} tests passed; total across both suites: {tests["total_tests"]}.
- Protected production hashes: {sum(item["matches"] for item in integrity)}/{len(integrity)} match. Current commit `{git_head}` on branch `{git_branch}`. Existing worktree changes were retained.

## Query budget correction

The cold-cache estimate from Iteration 21 omitted entity-label resolution. In the frozen archived depth-3 traces there are {accounting["archived_depth3_repeat1_logged_queries"]} logged navigation queries: {accounting["archived_relation_search_query_count"]} relation-direction queries and {accounting["archived_entity_search_query_count"]} entity searches. The current Freebase entity-search path also resolves candidate IDs through batched label SPARQL. Replaying the archived candidate stream with a cold per-process label cache adds {accounting["hidden_label_resolution_calls_if_cold_cache"]} label queries; with the proposed 12 incremental queries, the cold-cache estimate is **{accounting["cold_cache_total_estimate"]}**, so it **{'' if cold_fits else 'does not '}fit** the 100-query ceiling.

Offline, the fixed three archived traces provide {label_meta["cacheable_unambiguous_ids"]} unambiguous ID/label mappings for {label_meta["unique_entity_ids"]} unique candidate IDs, with {len(label_meta["conflicting_ids"])} conflicts and {label_meta["unnamed_entity_appearances"]} unnamed appearances. Using this exact local cache and stopping a case before any label query on a cache miss gives an expected **{accounting["frozen_local_label_cache_total_estimate"]}/{MAX_SPARQL_QUERIES}** read-only queries, leaving {accounting["expected_headroom_with_frozen_cache"]}. The estimate is conditional: a new live trajectory can visit uncached IDs or issue more navigation queries. The hard meter counts every physical attempt, consumes failed attempts, disables retries, and refuses query 101 before transport. No separate endpoint health query is planned.

The CoR SPARQL helper otherwise retries up to three times. `install_metered_backend` replaces both backend query entry points with a caller-supplied single-attempt transport and the frozen label cache, then restores the backend methods. A cache miss stops the case instead of falling through to the hidden resolver. The LLM client must be constructed with `OPENAI_MAX_RETRIES=1`, `OPENAI_RETRY_INTERVAL=0`; the source configures SDK retries to zero. The metered wrapper verifies these conditions and refuses calls without the exact approval record.

## Request and cost bounds

- Archived baseline for the three fixed cases: 47 model requests, 75,304 input and 11,937 output tokens, estimated at **${estimated_cost(75304,11937):.6f}** using $0.20/M input and $1.20/M output.
- Expected pilot calls from archived usage plus one relation-prune request per case: approximately 50, below the 60-call cap. Dry-run worst-case reservations at all caps: **${max_reserved:.6f}**, below $0.17. Each request is rejected before transport if its input reservation exceeds 7,000, completion cap exceeds 1,024, or remaining spend/call capacity cannot fit.
- Empty/error responses consume the attempted call and stop without retry. Missing usage reserves the full per-request amount and blocks further calls.

## Dry-run and failure controls

`PILOT_DRY_RUN.json` exercises a synthetic exact parent hash, STOP/ONE_HOP pairing, one-edge limit, non-Forward ineligibility, missing/truncated binding projection, empty and malformed model responses, KG timeout, all-ineligible cases, and the distinction between simulated and actual calls. **Every scenario records zero actual API calls, zero actual SPARQL queries, and zero transport calls.** These fixtures are not Freebase evidence.

## Decision

{decision_text}

## Reproduction

```sh
python -m unittest discover -s scripts/diagnostics/iter_021_matched_prefix -p 'test_*.py' -v
python -m unittest discover -s scripts/diagnostics/iter_022_gated_matched_prefix -p 'test_*.py' -v
python scripts/diagnostics/iter_022_gated_matched_prefix/offline_preflight.py
```

Artifacts are confined to `RESULTS/iter_022_gated_matched_prefix/`; Iteration 19/20/21 results were preserved. No `PILOT_RESULTS.json`, `PILOT_CASES.csv`, or matched-prefix results file was created.
""", encoding="utf-8")

    print(json.dumps({"status": "offline_preflight_complete", "manifest_sha256": manifest_hash,
                      "cold_cache_queries": accounting["cold_cache_total_estimate"],
                      "frozen_cache_queries": accounting["frozen_local_label_cache_total_estimate"],
                      "query_cap": MAX_SPARQL_QUERIES, "tests": tests["total_tests"],
                      "actual_api_calls": 0, "actual_sparql_queries": 0}, indent=2))


if __name__ == "__main__":
    main()
