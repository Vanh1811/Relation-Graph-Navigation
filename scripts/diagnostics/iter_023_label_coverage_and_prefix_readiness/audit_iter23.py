"""Independent integrity audit for Iteration 23 offline readiness artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
OUT = WORK / "RESULTS/iter_023_label_coverage_and_prefix_readiness"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check() -> dict:
    manifest_path = OUT / "PROPOSED_NEW_MANIFEST.json"
    manifest_hash = digest(manifest_path)
    expected_hash = (OUT / "PROPOSED_NEW_MANIFEST.sha256").read_text(encoding="utf-8").split()[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_hash == expected_hash
    assert len(manifest["cases"]) == 3
    assert [case["repeat"] for case in manifest["cases"]] == [2, 2, 2]
    assert len({case["slot_id"] for case in manifest["cases"]}) == 3
    assert len({case["question_id"] for case in manifest["cases"]}) == 3

    tests = json.loads((OUT / "TEST_RESULTS.json").read_text(encoding="utf-8"))
    assert tests["status"] == "PASS" and tests["total_tests"] == 76
    assert [suite["tests_run"] for suite in tests["suites"]] == [23, 19, 34]
    assert tests["paid_api_requests"] == 0 and tests["live_sparql_attempts"] == 0

    synthetic = json.loads((OUT / "SYNTHETIC_PILOT.json").read_text(encoding="utf-8"))
    assert synthetic["status"] == "PASS"
    assert synthetic["live_api_requests"] == 0 and synthetic["live_sparql_attempts"] == 0
    assert synthetic["matched_prefix_fixture"]["eligible_pairs"] == 1
    assert synthetic["matched_prefix_fixture"]["same_parent_hash"] is True
    assert synthetic["matched_prefix_fixture"]["new_edge_count"] == 1
    assert synthetic["non_forward_fixture"]["reason"] == "NON_FORWARD_BOUNDARY"

    query = json.loads((OUT / "QUERY_BUDGET_MODEL.json").read_text(encoding="utf-8"))
    assert query["expected_physical_attempts"] == sum(query["expected_query_components"].values()) == 109
    assert query["physical_attempt_cap"] == sum(query["planning_allocation_not_individually_enforced"].values()) == 140
    assert query["enforced_limits"]["global_physical_select_attempt_cap"] == 140
    assert query["enforced_limits"]["dynamic_label_query_attempt_cap"] == 12
    assert query["dynamic_label_bounds"]["max_ids_per_batch"] == 100
    assert query["dynamic_label_bounds"]["max_label_queries"] == 12
    assert query["retries"] == 0 and query["failed_attempts_consume_cap"]

    with (OUT / "LABEL_COVERAGE_MATRIX.csv").open(newline="", encoding="utf-8") as handle:
        matrix = list(csv.DictReader(handle))
    cached = [row for row in matrix if row["frozen_cache_status"] == "CACHED"]
    missed = [row for row in matrix if row["frozen_cache_status"] == "UNSEEN_IN_FROZEN_CACHE"]
    assert len(cached) == 198 and len(missed) == 7 and len(matrix) == 205
    assert all(not row["frozen_label"] for row in missed)

    protected = json.loads((OUT / "PROTECTED_PRIOR_ARTIFACTS.json").read_text(encoding="utf-8"))
    prior_unchanged = 0
    for item in protected["files"]:
        assert digest(ROOT / item["path"]) == item["sha256"], item["path"]
        prior_unchanged += 1
    sources = json.loads((OUT / "SOURCE_SNAPSHOT.json").read_text(encoding="utf-8"))
    production_matches = sum(
        digest(ROOT / path) == value["expected_sha256"] == value["actual_sha256"]
        for path, value in sources["protected_production_sources"].items()
    )
    assert production_matches == len(sources["protected_production_sources"])
    for path, expected in sources["iteration23_diagnostic_sources"].items():
        assert digest(ROOT / path) == expected, path

    auth_request = (OUT / "AUTHORIZATION_REQUIRED.md").read_text(encoding="utf-8")
    assert manifest_hash in auth_request and "US$0.16" in auth_request and "140 physical" in auth_request
    assert "no retries" in auth_request
    approval = json.loads((OUT / "AUTHORIZATION.json").read_text(encoding="utf-8"))
    assert approval["authorization_status"] == "APPROVED"
    assert approval["manifest_sha256"] == manifest_hash
    assert approval["transport_opened"] is False and approval["live_execution_started"] is False
    assert approval["authorization_consumed"] is False
    preflight = json.loads((OUT / "LIVE_PREFLIGHT.json").read_text(encoding="utf-8"))
    assert preflight["status"] == "PASS_NO_TRANSPORT_PREFLIGHT"
    assert preflight["api_or_sparql_transport_opened"] is False
    forbidden_live_files = ["PILOT_RESULTS.json", "RAW_CASE_TRACES.jsonl", "REQUEST_JOURNAL.jsonl", "SPARQL_QUERY_JOURNAL.jsonl"]
    assert not any((OUT / name).exists() for name in forbidden_live_files)

    status = json.loads((WORK / "STATUS.json").read_text(encoding="utf-8"))
    cost = json.loads((WORK / "COST_LEDGER.json").read_text(encoding="utf-8"))
    assert status["status"] == "ITERATION_23_APPROVED_AWAITING_LIVE_PREFLIGHT"
    assert status["budget_remaining"]["active_paid_authorization"] is True
    assert status["budget_remaining"]["iteration_23_authorization_status"] == "APPROVED_NOT_YET_CONSUMED"
    assert status["budget_remaining"]["cumulative_llm_calls"] == cost["llm_calls"] == 1457
    assert status["budget_remaining"]["cumulative_input_tokens"] == cost["input_tokens"] == 1880696
    assert status["budget_remaining"]["cumulative_output_tokens"] == cost["output_tokens"] == 360423
    assert status["budget_remaining"]["read_only_kg_queries"] == cost["read_only_kg_queries"] == 2252
    assert cost["paid_experiment_authorized"] is True
    assert cost["configured_call_budget"] == 60
    assert cost["configured_cost_budget_usd"] == 0.16
    handoff = (WORK / "CHATGPT_HANDOFF.md").read_text(encoding="utf-8")
    assert handoff.startswith("<!-- ITERATION23_HANDOFF_START -->")
    assert "<!-- ITERATION22_HANDOFF_START -->" in handoff
    assert "preserved verbatim as historical evidence" in handoff

    result = {
        "iteration": 23,
        "status": "PASS_APPROVED_PRELIVE_INTEGRITY",
        "manifest_sha256": manifest_hash,
        "test_count": 76,
        "synthetic_valid_pairs": 1,
        "archive_cache_entries": len(cached),
        "known_cache_misses": len(missed),
        "prior_result_artifacts_unchanged": prior_unchanged,
        "protected_production_sources_unchanged": production_matches,
        "live_api_requests": 0,
        "live_sparql_attempts": 0,
        "authorization_recorded": True,
        "authorization_status": "APPROVED_NOT_YET_CONSUMED",
    }
    (OUT / "ITER23_INTEGRITY.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    result = check()
    raise SystemExit(0 if result["status"] == "PASS_APPROVED_PRELIVE_INTEGRITY" else 1)
