#!/usr/bin/env python3
"""Independently audit the authorized Iteration 22 live pilot artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "scripts/diagnostics/iter_022_gated_matched_prefix") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts/diagnostics/iter_022_gated_matched_prefix"))
if str(ROOT / "scripts/diagnostics/iter_021_matched_prefix") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts/diagnostics/iter_021_matched_prefix"))

from pilot import (  # noqa: E402
    INPUT_USD_PER_MILLION,
    MAX_API_COST_USD,
    MAX_API_REQUESTS,
    MAX_COMPLETION_TOKENS,
    MAX_INPUT_TOKENS,
    MAX_SPARQL_QUERIES,
    OUTPUT_USD_PER_MILLION,
    approval_matches,
    estimated_cost,
    require_read_only_select,
)
import run_live_pilot as runner  # noqa: E402

OUT = runner.OUT
MANIFEST_PATH = runner.MANIFEST_PATH
PROTECTED_PATH = runner.PROTECTED_PATH


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    runner.atomic_json(path, value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def audit() -> dict[str, Any]:
    manifest_bytes_hash = sha256(MANIFEST_PATH)
    manifest = read_json(MANIFEST_PATH)
    approval = read_json(runner.AUTH_PATH)
    results = read_json(runner.RESULTS_PATH)
    state = read_json(runner.RUN_STATE_PATH)
    cache_doc = read_json(runner.LABEL_CACHE_PATH)
    gold_doc = read_json(runner.GOLD_PATH)
    with runner.CASE_CSV_PATH.open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    traces = read_jsonl(runner.TRACE_JSONL_PATH)
    request_events = read_jsonl(runner.REQUEST_JSONL_PATH)
    query_records = read_jsonl(runner.QUERY_JSONL_PATH)

    expected_cases = [
        (str(item["question_id"]), int(item["repeat"]), str(item.get("role", "")))
        for item in manifest["cases"]
    ]
    observed_cases = [
        (str(row["question_id"]), int(row["repeat"]), str(row["role"]))
        for row in cases
    ]
    assert manifest_bytes_hash == runner.MANIFEST_SHA256
    assert len(expected_cases) == 3 and all(item[1] == 1 for item in expected_cases)
    assert observed_cases == expected_cases
    assert len(traces) == 3
    assert [str(item["case"]["question_id"]) for item in traces] == [item[0] for item in expected_cases]
    assert approval_matches(approval)
    assert approval.get("manifest_sha256") == manifest_bytes_hash
    assert approval.get("transport_opened") is True
    assert approval.get("live_execution_started") is True

    current_sources = runner.source_hashes()
    saved_sources = approval.get("code_version", {}).get("relevant_source_sha256", {})
    source_hashes_match = current_sources == saved_sources
    protected = read_json(PROTECTED_PATH)["files"]
    protected_results = [
        {"path": item["path"], "expected_sha256": item["sha256_at_iteration_start"],
         "actual_sha256": sha256(ROOT / item["path"]),
         "matches": sha256(ROOT / item["path"]) == item["sha256_at_iteration_start"]}
        for item in protected
    ]

    llm_starts = [e for e in request_events if e.get("kind") == "LLM" and e.get("status") == "ATTEMPT_STARTED"]
    llm_metered = [e for e in request_events if e.get("kind") == "LLM" and e.get("status") == "METERED"]
    sparql_starts = [e for e in request_events if e.get("kind") == "SPARQL" and e.get("status") == "ATTEMPT_STARTED"]
    sparql_metered = [e for e in request_events if e.get("kind") == "SPARQL" and e.get("status") == "ok"]
    assert [int(e["request_index"]) for e in llm_starts] == list(range(1, len(llm_starts) + 1))
    assert [int(e["request_index"]) for e in llm_metered] == list(range(1, len(llm_metered) + 1))
    assert [int(e["query_index"]) for e in sparql_starts] == list(range(1, len(sparql_starts) + 1))
    assert [int(e["query_index"]) for e in sparql_metered] == list(range(1, len(sparql_metered) + 1))
    assert len(llm_starts) == len(llm_metered)
    assert len(sparql_starts) == len(sparql_metered)
    assert all(e.get("retries") == 0 for e in llm_starts + llm_metered + sparql_starts + sparql_metered)
    llm_by_index = {int(e["request_index"]): e for e in llm_metered}
    for attempt in llm_starts:
        idx = int(attempt["request_index"])
        measured = llm_by_index[idx]
        assert attempt.get("input_token_reservation") == MAX_INPUT_TOKENS
        assert attempt.get("completion_token_cap") == MAX_COMPLETION_TOKENS
        assert attempt.get("prompt_sha256") and len(attempt["prompt_sha256"]) == 64
        assert 0 <= int(measured["input_tokens"]) <= MAX_INPUT_TOKENS
        assert 0 <= int(measured["output_tokens"]) <= MAX_COMPLETION_TOKENS

    query_starts = [q for q in query_records if q.get("status") == "ATTEMPT_STARTED"]
    query_finals = [q for q in query_records if q.get("status") != "ATTEMPT_STARTED"]
    assert len(query_starts) == len(query_finals) == len(sparql_starts)
    finals_by_index = {int(q["query_index"]): q for q in query_finals}
    for start in query_starts:
        idx = int(start["query_index"])
        final = finals_by_index[idx]
        assert start["query"] == final["query"]
        assert start["query_sha256"] == final["query_sha256"]
        assert hashlib.sha256(start["query"].encode("utf-8")).hexdigest() == start["query_sha256"]
        require_read_only_select(start["query"])
        assert final.get("status") == "ok"

    api_requests = len(llm_starts)
    sparql_queries = len(sparql_starts)
    input_tokens = sum(int(e["input_tokens"]) for e in llm_metered)
    output_tokens = sum(int(e["output_tokens"]) for e in llm_metered)
    actual_cost = sum(float(e["cost_usd"]) for e in llm_metered)
    recomputed_cost = estimated_cost(input_tokens, output_tokens)
    assert abs(actual_cost - recomputed_cost) < 1e-12
    assert api_requests <= MAX_API_REQUESTS and sparql_queries <= MAX_SPARQL_QUERIES
    assert actual_cost <= MAX_API_COST_USD
    assert results["api_requests_attempted"] == api_requests
    assert results["sparql_queries_attempted"] == sparql_queries
    assert abs(float(results["estimated_api_cost_usd"]) - actual_cost) < 1e-12
    assert abs(float(state["actual_cost_usd"]) - actual_cost) < 1e-12
    assert float(results["uncertain_reserved_cost_usd"]) == 0.0
    assert float(state["uncertain_cost_usd"]) == 0.0

    cache = {str(x["entity_id"]): str(x["label"]) for x in cache_doc["labels"]}
    assert len(cache) == 198 and cache_doc["manifest_sha256"] == manifest_bytes_hash
    misses: set[str] = set()
    for trace in traces:
        misses.update(str(x) for x in trace.get("missing_ids", []))
    assert misses and not (misses & set(cache))
    case_statuses = [str(row["status"]) for row in cases]
    case_reasons = [str(row["reason"]) for row in cases]
    all_cache_miss = all(status == "INELIGIBLE" and reason == "LABEL_CACHE_MISS"
                         for status, reason in zip(case_statuses, case_reasons))
    no_prefix = all(not row.get("prefix_hash") and row.get("prefix_complete") == "False" for row in cases)
    no_pairs_or_projection = all("pair" not in trace and "binding_projection" not in trace
                                 and not trace.get("snapshot") for trace in traces)

    gold_ids = {str(gid) for values in gold_doc["cases"].values() for gid in values}
    prompts = [str(p.get("prompt", "")) for trace in traces for p in trace.get("prompt_history", [])]
    prompt_gold_leaks = sorted(gid for gid in gold_ids if any(gid in prompt for prompt in prompts))
    answer_stage_events = []
    for trace in traces:
        answer_stage_events.extend(
            str(event.get("type")) for event in trace.get("prompt_history", [])
            if event.get("type") in {"filter_answer", "generate_without_explored_paths"}
        )
        answer_stage_events.extend(
            str(event.get("stage")) for event in trace.get("method_events", [])
            if event.get("stage") in {"validate", "generate_directly"}
        )

    worst_case_reservation = MAX_API_REQUESTS * (
        MAX_INPUT_TOKENS * INPUT_USD_PER_MILLION / 1_000_000
        + MAX_COMPLETION_TOKENS * OUTPUT_USD_PER_MILLION / 1_000_000
    )
    assert abs(worst_case_reservation - 0.157728) < 1e-12
    artifact_paths = [
        runner.AUTH_PATH, runner.RUN_STATE_PATH, runner.RESULTS_PATH, runner.CASE_CSV_PATH,
        runner.TRACE_JSONL_PATH, runner.REQUEST_JSONL_PATH, runner.QUERY_JSONL_PATH,
        runner.GOLD_PATH, runner.LABEL_CACHE_PATH,
        OUT / "OFFLINE_PREFLIGHT.md", OUT / "QUERY_ACCOUNTING.json", OUT / "PILOT_DRY_RUN.json",
    ]
    artifact_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in artifact_paths}

    integrity = {
        "iteration": 22,
        "audited_at_utc": utc_now(),
        "status": "PASS_ACCOUNTING_INTEGRITY_PARTIAL_OR_INELIGIBLE" if (
            source_hashes_match and all(x["matches"] for x in protected_results)
            and all_cache_miss and no_prefix and no_pairs_or_projection
            and not prompt_gold_leaks and not answer_stage_events
        ) else "INTEGRITY_BLOCKED",
        "evidence_class": "LIVE_RETRIEVAL_ONLY_PILOT_NO_VALID_MATCHED_PREFIX",
        "manifest_sha256": manifest_bytes_hash,
        "approval": {
            "matches_exact_scope_and_caps": approval_matches(approval),
            "approved_at": approval.get("approved_at"),
            "transport_opened": approval.get("transport_opened"),
            "live_execution_started": approval.get("live_execution_started"),
            "authorization_consumed": True,
            "no_budget_rollover": True,
        },
        "source_hashes_match_authorized_code_version": source_hashes_match,
        "source_sha256": current_sources,
        "protected_source_integrity": {
            "all_match": all(item["matches"] for item in protected_results),
            "files": protected_results,
        },
        "fixed_slots": {
            "expected": expected_cases,
            "observed": observed_cases,
            "all_three_processed_once": len(cases) == 3 and observed_cases == expected_cases,
            "repeat": 1,
        },
        "matched_prefix": {
            "eligible_cases": sum(status.startswith("VALID_") for status in case_statuses),
            "valid_exact_prefix_cases": 0 if no_prefix else None,
            "all_cases_ineligible_due_to_label_cache_miss": all_cache_miss,
            "cases_with_prefix_hash": sum(bool(row.get("prefix_hash")) for row in cases),
            "binding_projection_or_pair_records": sum(
                "binding_projection" in trace or "pair" in trace for trace in traces
            ),
            "unique_uncached_ids": sorted(misses),
            "unique_uncached_id_count": len(misses),
            "all_misses_absent_from_frozen_cache": not bool(misses & set(cache)),
        },
        "gold_separation": {
            "gold_sidecar_evidence_class": gold_doc.get("evidence_class"),
            "prompt_gold_id_leaks": prompt_gold_leaks,
            "filter_or_final_generation_events": answer_stage_events,
            "post_boundary_answer_stage_calls": 0 if not answer_stage_events else None,
        },
        "accounting": {
            "api_requests_attempted": api_requests,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_api_cost_usd": round(actual_cost, 12),
            "cost_rate_basis_usd_per_million": {
                "input": INPUT_USD_PER_MILLION, "output": OUTPUT_USD_PER_MILLION,
            },
            "worst_case_cost_reservation_at_60_full_caps_usd": round(worst_case_reservation, 12),
            "sparql_physical_attempts": sparql_queries,
            "successful_read_only_selects": len(query_finals),
            "api_retries": 0,
            "sparql_retries": 0,
            "api_cap": MAX_API_REQUESTS,
            "input_token_cap_per_request": MAX_INPUT_TOKENS,
            "completion_token_cap_per_request": MAX_COMPLETION_TOKENS,
            "estimated_cost_cap_usd": MAX_API_COST_USD,
            "sparql_cap": MAX_SPARQL_QUERIES,
            "all_caps_respected": api_requests <= MAX_API_REQUESTS and sparql_queries <= MAX_SPARQL_QUERIES
                and actual_cost <= MAX_API_COST_USD
                and all(int(e["input_tokens"]) <= MAX_INPUT_TOKENS for e in llm_metered)
                and all(int(e["output_tokens"]) <= MAX_COMPLETION_TOKENS for e in llm_metered),
        },
        "query_integrity": {
            "query_attempts": len(query_starts),
            "all_queries_read_only_select": True,
            "all_queries_have_matching_text_hash_and_success_record": True,
            "query_attempt_indices_contiguous": True,
        },
        "case_statuses": [
            {"question_id": row["question_id"], "status": row["status"], "reason": row["reason"],
             "missing_label_ids": trace.get("missing_ids", []),
             "api_attempts": int(row["api_attempts"]), "sparql_attempts": int(row["query_attempts"])}
            for row, trace in zip(cases, traces)
        ],
        "artifact_sha256": artifact_hashes,
    }
    assert integrity["status"] != "INTEGRITY_BLOCKED", "one or more independent integrity checks failed"

    # Mark the execution outcome separately from the absence of scientific cases.
    results["execution_status"] = "ALL_THREE_FIXED_CASES_PROCESSED"
    results["status"] = "PARTIAL_OR_INELIGIBLE"
    results["eligible_cases"] = 0
    results["valid_matched_prefix_cases"] = 0
    results["scientific_decision"] = "NO_VALID_MATCHED_PREFIX_MEASUREMENT"
    results["label_cache_miss_unique_ids"] = sorted(misses)
    results["authorization_consumed"] = True
    write_json(runner.RESULTS_PATH, results)

    state["scientific_status"] = "PARTIAL_OR_INELIGIBLE"
    state["valid_matched_prefix_cases"] = 0
    state["authorization_consumed"] = True
    state["authorization_remaining_rollover"] = False
    write_json(runner.RUN_STATE_PATH, state)

    approval["authorization_status"] = "CONSUMED"
    approval["authorization_consumed_at_utc"] = utc_now()
    approval["actual_usage"] = {
        "api_requests": api_requests, "input_tokens": input_tokens,
        "completion_tokens": output_tokens, "estimated_cost_usd": round(actual_cost, 12),
        "read_only_sparql_queries": sparql_queries, "retries": 0,
        "valid_matched_prefix_cases": 0,
        "remaining_authorization_rollover": False,
    }
    write_json(runner.AUTH_PATH, approval)

    # Recompute artifact hashes after the audited summaries above were updated.
    artifact_hashes[str(runner.RESULTS_PATH.relative_to(ROOT))] = sha256(runner.RESULTS_PATH)
    artifact_hashes[str(runner.RUN_STATE_PATH.relative_to(ROOT))] = sha256(runner.RUN_STATE_PATH)
    artifact_hashes[str(runner.AUTH_PATH.relative_to(ROOT))] = sha256(runner.AUTH_PATH)
    integrity["artifact_sha256"] = artifact_hashes
    write_json(OUT / "PILOT_INTEGRITY.json", integrity)

    decision = f"""# Iteration 22 live pilot decision

**Decision: `PARTIAL_OR_INELIGIBLE` — no causal matched-prefix result.** The approved run processed all three fixed repeat-1 questions, but **0/3** reached an eligible depth-3 prefix. Each case stopped at the frozen local entity-label cache gate before binding projection or the one-edge continuation.

## What happened

The live trajectories produced {len(misses)} unique Freebase entity IDs absent from the approved 198-ID local label cache: {', '.join(f'`{value}`' for value in sorted(misses))}. The fail-closed cache policy stopped each affected case before any hidden label-resolution query. The raw SPARQL journal contains the exact successful read-only SELECT query text, result rows and query hashes; the raw case traces retain the prompts, decisions and cache-miss IDs.

| Case | Status | API requests | SPARQL queries | Missing label IDs |
| --- | --- | ---: | ---: | --- |
""" + "\n".join(
        f"| `{row['question_id']}` | `{row['status']}: {row['reason']}` | {row['api_attempts']} | {row['query_attempts']} | {', '.join(f'`{x}`' for x in trace.get('missing_ids', []))} |"
        for row, trace in zip(cases, traces)
    ) + f"""

## Accounting and integrity

- Model: `{runner.MODEL_EXPECTED}`; **{api_requests}/{MAX_API_REQUESTS}** API attempts, {input_tokens:,} input and {output_tokens:,} completion tokens, **${actual_cost:.6f} estimated** at $0.20/M input and $1.20/M output. This is a rate-based estimate, not an invoice.
- SPARQL: **{sparql_queries}/{MAX_SPARQL_QUERIES}** physical attempts; all {len(query_finals)} completed as read-only SELECTs. API retries: **0**; SPARQL retries: **0**.
- Approved ceilings: 60 attempts; 7,000 input and 1,024 completion tokens/request; US$0.17 estimated; 100 read-only SPARQL queries. Every ceiling was respected. The full 60-request worst-case reservation is US$0.157728.
- Exact manifest, recorded approval, contiguous attempt journals, query text/hash provenance, scoring-only gold separation, and all {len(protected_results)}/{len(protected_results)} protected production source hashes passed audit.
- No answer filter, final generation, binding projection, or one-edge arm ran. Gold IDs were absent from logged inference prompts.

## Interpretation

This is an **instrumentation/cache-coverage blocker**, not evidence for or against the benefit of a fourth edge. The observed trajectories left the historical label cache before the depth-3 decision boundary; therefore the pilot measured no identical-prefix STOP/ONE_HOP comparison and produced no new-gold-ID outcome. It cannot support claims about search depth, final-answer accuracy, adaptive routing, or generalization.

The authorization is consumed. Unused request, cost and query headroom does not roll over, and none of these repeat-1 slots may be rerun under this approval.

## Next defensible step

Prepare a **new manifest and separate authorization** for a cache-coverage repair: allow bounded, read-only batched label resolution for newly encountered IDs, freeze the resulting label map and prompts, and use new repeat slots. Do not rerun these completed repeat-1 slots or claim a matched-prefix effect. The decision requiring researcher approval is whether to authorize that new, separately budgeted pilot after its exact manifest and caps are prepared.

## Artifacts

- `PILOT_RESULTS.json`, `PILOT_CASES.csv`, `PILOT_RUN_STATE.json`
- `RAW_CASE_TRACES.jsonl`, `SPARQL_QUERY_JOURNAL.jsonl`, `REQUEST_JOURNAL.jsonl`
- `GOLD_SCORING_ONLY.json`, `PILOT_INTEGRITY.json`, `AUTHORIZATION.json`
"""
    (OUT / "DECISION.md").write_text(decision, encoding="utf-8")
    print(json.dumps({
        "status": integrity["status"], "eligible_cases": 0,
        "api_requests": api_requests, "input_tokens": input_tokens,
        "output_tokens": output_tokens, "estimated_cost_usd": round(actual_cost, 12),
        "sparql_queries": sparql_queries, "unique_cache_miss_ids": len(misses),
    }, indent=2))
    return integrity


if __name__ == "__main__":
    audit()
