"""Offline independent accounting/integrity audit for the consumed Iteration 23 pilot."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
OUT = WORK / "RESULTS/iter_023_label_coverage_and_prefix_readiness"
MANIFEST_PATH = OUT / "PROPOSED_NEW_MANIFEST.json"
AUTH_PATH = OUT / "AUTHORIZATION.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def check() -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    manifest_sha = sha256(MANIFEST_PATH)
    expected_sha = (OUT / "PROPOSED_NEW_MANIFEST.sha256").read_text(encoding="utf-8").split()[0]
    assert manifest_sha == expected_sha == "ff57d84266295e7778b00e733fdba9e8a9232f4eed87971f7b0e029cefc331f2"
    assert len(manifest["cases"]) == 3 and [c["repeat"] for c in manifest["cases"]] == [2, 2, 2]

    result = read_json(OUT / "PILOT_RESULTS.json")
    state = read_json(OUT / "LIVE_RUN_STATE.json")
    approval = read_json(AUTH_PATH)
    assert result["status"] == "PILOT_COMPLETE"
    assert result["manifest_sha256"] == manifest_sha
    assert len(result["case_results"]) == 3
    assert state["status"] == "PILOT_COMPLETE" and state["case_results_written"] == 3
    assert approval["authorization_status"] == "APPROVED"
    assert approval["manifest_sha256"] == manifest_sha
    assert approval["authorization_consumed"] is True
    assert approval["transport_opened"] is True and approval["live_execution_started"] is True

    request_events = read_jsonl(OUT / "REQUEST_JOURNAL.jsonl")
    query_events = read_jsonl(OUT / "SPARQL_QUERY_JOURNAL.jsonl")
    sparql_requests = [e for e in request_events if e.get("kind") == "SPARQL"]
    api_requests = [e for e in request_events if e.get("kind") == "LLM"]
    api_starts = [e for e in api_requests if e.get("status") == "ATTEMPT_STARTED"]
    api_success = [e for e in api_requests if e.get("status") == "METERED"]
    api_errors = [e for e in api_requests if e.get("status") == "TRANSPORT_ERROR"]
    assert len(sparql_requests) == 38
    assert [e["status"] for e in sparql_requests].count("ATTEMPT_STARTED") == 19
    assert [e["status"] for e in sparql_requests].count("METERED") == 19
    assert len(query_events) == 38
    query_starts = [e for e in query_events if e.get("status") == "ATTEMPT_STARTED"]
    query_ends = [e for e in query_events if e.get("status") != "ATTEMPT_STARTED"]
    assert len(query_starts) == len(query_ends) == 19
    assert [e["query_index"] for e in query_starts] == list(range(1, 20))
    assert all(e.get("retries") == 0 for e in query_events + sparql_requests)
    assert all(str(e.get("query", "")).lstrip().upper().startswith(("SELECT", "PREFIX")) for e in query_starts)
    assert all(e.get("status") == "ok" for e in query_ends)
    assert all(e.get("status") == "ATTEMPT_STARTED" for e in sparql_requests[::2])
    assert all(e.get("status") == "METERED" for e in sparql_requests[1::2])
    assert [e.get("category") for e in sparql_requests[0::2]][0] == "endpoint_preflight"

    assert len(api_starts) == 7 and len(api_success) == 6 and len(api_errors) == 1
    assert [e["request_index"] for e in api_starts] == list(range(1, 8))
    assert all(e.get("retries") == 0 for e in api_requests)
    assert all(e["input_tokens"] <= 7000 and e["output_tokens"] <= 1024 for e in api_success)
    error_event = api_errors[0]
    assert error_event["request_index"] == 4 and "KeyboardInterrupt" in error_event.get("error", "")
    start_by_request = {e["request_index"]: e for e in api_starts}
    end_by_request = {e["request_index"]: e for e in api_success}
    end_by_request[4] = error_event
    api_durations = {
        index: (parse_time(end_by_request[index]["recorded_at_utc"])
                - parse_time(start["recorded_at_utc"])).total_seconds()
        for index, start in start_by_request.items() if index in end_by_request
    }
    assert start_by_request[4]["prompt_sha256"] == start_by_request[5]["prompt_sha256"]
    assert api_durations[1] == 163.0
    assert api_durations[1] > approval["per_call_timeout_seconds"]

    input_tokens = sum(int(e["input_tokens"]) for e in api_success)
    output_tokens = sum(int(e["output_tokens"]) for e in api_success)
    actual_cost = round(input_tokens * 0.20 / 1_000_000 + output_tokens * 1.20 / 1_000_000, 12)
    uncertain = round(float(start_by_request[4]["reserved_usd"]), 12)
    usage = approval["actual_usage"]
    assert (usage["api_requests"], usage["input_tokens"], usage["completion_tokens"],
            usage["read_only_sparql_attempts"]) == (7, input_tokens, output_tokens, 19)
    assert usage.get("wrapper_sdk_retries", 0) == 0
    assert usage.get("same_prompt_retries", 0) == 1
    assert usage["estimated_cost_usd"] == actual_cost
    assert usage["uncertain_reserved_cost_usd"] == uncertain
    assert actual_cost + uncertain < 0.16
    assert result["api_requests_attempted"] == 7 and result["sparql_physical_attempts"] == 19

    labels = read_jsonl(OUT / "RAW_CASE_TRACES.jsonl")
    assert len(labels) == 3
    missing_by_case: dict[str, list[str]] = {}
    for trace in labels:
        case = trace.get("case", {})
        case_id = str(case.get("question_id", ""))
        assert trace.get("status") == "INELIGIBLE" and trace.get("reason") == "LABEL_RESOLUTION_STOP"
        assert not trace.get("snapshot")
        events = trace.get("label_resolution", {}).get("events", [])
        missing = sorted({str(e["entity_id"]) for e in events
                          if e.get("status") == "UNRESOLVED_LABEL_ABSENT"})
        missing_by_case[case_id] = missing
    all_missing = sorted({entity_id for ids in missing_by_case.values() for entity_id in ids})
    expected_missing = sorted({
        "m.04st848", "m.04stk8g", "m.04stpzt", "m.04strb6",
        "m.0hz9k44", "m.0hzdjww", "m.0ndvpvf",
    })
    assert all_missing == expected_missing
    label_queries = [e for e in query_ends if e.get("case_id") and e.get("query_index") in {5, 9, 19}]
    assert len(label_queries) == 3
    label_query_rows = {
        e["query_index"]: {
            "rows": len(e.get("rows", [])),
            "entity_bound_rows": sum("entity" in row for row in e.get("rows", [])),
            "label_bound_rows": sum("targetEntity" in row for row in e.get("rows", [])),
        }
        for e in label_queries
    }
    assert label_query_rows == {
        5: {"rows": 1, "entity_bound_rows": 1, "label_bound_rows": 0},
        9: {"rows": 1, "entity_bound_rows": 1, "label_bound_rows": 0},
        19: {"rows": 6, "entity_bound_rows": 6, "label_bound_rows": 0},
    }
    assert all(c.get("status") == "INELIGIBLE" for c in result["case_results"])
    assert result["eligible_cases"] == result["valid_matched_prefix_cases"] == 0

    tests = read_json(OUT / "TEST_RESULTS.json")
    assert tests["status"] == "PASS" and tests["total_tests"] == 78
    assert [suite["tests_run"] for suite in tests["suites"]] == [23, 19, 36]
    protected = read_json(OUT / "PROTECTED_PRIOR_ARTIFACTS.json")
    old_files = sum(sha256(ROOT / item["path"]) == item["sha256"] for item in protected["files"])
    assert old_files == len(protected["files"]) == 350
    snapshot = read_json(OUT / "SOURCE_SNAPSHOT.json")
    prod = snapshot["protected_production_sources"]
    prod_ok = sum(sha256(ROOT / path) == item["expected_sha256"] == item["actual_sha256"]
                  for path, item in prod.items())
    assert prod_ok == len(prod) == 4

    result_doc = {
        "iteration": 23,
        "audit_status": "ACCOUNTING_RECONCILED_WITH_PROTOCOL_DEVIATIONS",
        "research_decision": "REPAIR_NOT_READY",
        "manifest_sha256": manifest_sha,
        "fixed_slots": len(manifest["cases"]),
        "eligible_cases": 0,
        "live_matched_prefix_pairs": 0,
        "unique_unresolved_label_ids": all_missing,
        "label_resolution_selects": 3,
        "label_lookup_rows": label_query_rows,
        "label_lookup_rows_with_bound_labels": 0,
        "api_attempts": len(api_starts),
        "api_metered_attempts": len(api_success),
        "api_uncertain_attempts": len(api_errors),
        "input_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "actual_rate_based_api_cost_usd": actual_cost,
        "uncertain_reserved_cost_usd": uncertain,
        "maximum_rate_based_exposure_usd": round(actual_cost + uncertain, 12),
        "sparql_physical_attempts_including_preflight": len(query_starts),
        "sparql_attempt_categories": {
            category: sum(e.get("category") == category for e in query_starts)
            for category in sorted({str(e.get("category")) for e in query_starts})
        },
        "wrapper_sdk_retries_recorded": 0,
        "same_prompt_retries_after_interruption": 1,
        "no_retry_authorization_compliant": False,
        "api_attempt_elapsed_seconds_by_index": api_durations,
        "max_api_attempt_elapsed_seconds": max(api_durations.values()),
        "authorization_per_call_timeout_seconds": approval["per_call_timeout_seconds"],
        "protocol_deviations": [
            "API attempt 1 lasted 163 seconds by journal timestamps, exceeding the authorized 120-second per-call ceiling.",
            "API attempt 4 was locally interrupted and recorded as TRANSPORT_ERROR; request 5 immediately reused the identical prompt hash. Conservatively count this as one repeated-prompt retry, despite zero SDK/wrapper retry metadata.",
        ],
        "prior_result_artifacts_unchanged": old_files,
        "protected_production_sources_unchanged": prod_ok,
        "offline_tests_passed": tests["total_tests"],
        "audited_at_utc": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (OUT / "PILOT_INTEGRITY.json").write_text(json.dumps(result_doc, indent=2) + "\n", encoding="utf-8")

    slots = {str(case["question_id"]): case for case in manifest["cases"]}
    rows: list[dict[str, Any]] = []
    for question_id, case in slots.items():
        case_api = [e for e in api_starts if e.get("case_id") == question_id]
        case_query = [e for e in query_starts if e.get("case_id") == question_id]
        idxs = {e["request_index"] for e in case_api}
        case_actual = sum(float(e.get("cost_usd", 0.0)) for e in api_success if e.get("request_index") in idxs)
        case_reserved = sum(float(e.get("reserved_usd", 0.0)) for e in api_errors if e.get("request_index") in idxs)
        rows.append({
            "slot_id": case["slot_id"], "question_id": question_id, "repeat": case["repeat"],
            "case_status": "INELIGIBLE_LABEL_RESOLUTION_STOP",
            "unresolved_ids": "|".join(missing_by_case.get(question_id, [])),
            "api_attempts": len(case_api), "api_metered": sum(e["request_index"] in idxs for e in api_success),
            "api_uncertain": sum(e["request_index"] in idxs for e in api_errors),
            "sparql_attempts": len(case_query), "metered_api_cost_usd": round(case_actual, 12),
            "uncertain_reserved_usd": round(case_reserved, 12),
        })
    with (OUT / "CASE_USAGE_AUDIT.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "iteration": 23,
        "status": result_doc["audit_status"],
        "manifest_sha256": manifest_sha,
        "api_attempts": len(api_starts), "api_metered": len(api_success),
        "api_uncertain": len(api_errors), "sparql_attempts": len(query_starts),
        "eligible_cases": 0, "live_matched_prefix_pairs": 0,
        "unique_unresolved_label_ids": len(all_missing), "actual_rate_based_api_cost_usd": actual_cost,
        "uncertain_reserved_cost_usd": uncertain,
        "prior_result_artifacts_unchanged": old_files,
        "protected_production_sources_unchanged": prod_ok,
        "protocol_deviation_count": len(result_doc["protocol_deviations"]),
    }
    (OUT / "ITER23_INTEGRITY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return result_doc


if __name__ == "__main__":
    check()
