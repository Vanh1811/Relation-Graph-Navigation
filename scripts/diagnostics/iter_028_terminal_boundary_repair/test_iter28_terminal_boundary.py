from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from scripts.diagnostics.iter_028_terminal_boundary_repair.scoring_gate import (
    candidate_snapshot_hash,
    manifest_digest,
    validate_for_scoring,
)
from scripts.diagnostics.iter_028_terminal_boundary_repair.terminal_audit import (
    build_iter27_matrix,
    observe_terminal_boundary,
)


ROOT = Path(__file__).resolve().parents[3]


def direct_fixture() -> dict:
    return {
        "case_id": "case-1",
        "question_key": "grailqa::1",
        "baseline_status": "BASELINE_COMPLETED",
        "route": "cor",
        "result": {"action": "Stop", "results": [{"id": "m.a"}]},
        "prompt_history": [{
            "type": "reasoning", "trace_id": 5, "parsed_result": "Stop",
            "answers": ["Alpha"],
        }],
        "sparql_history": [{
            "search_type": "entity_search", "trace_id": 4, "status": "ok",
            "sparql": "SELECT ?entity WHERE { ns:m.root ns:people.person.profession ?entity }",
            "candidate_entity_ids": ["m.a", "m.b"],
            "candidate_entity_names": ["Alpha", "Beta"],
            "literal_candidates": [],
        }],
        "question": "which entity is linked to the topic?",
    }


def valid_run() -> tuple[dict, list[str]]:
    ids = ["m.a", "m.b"]
    labels = ["Alpha", "Beta"]
    digest = candidate_snapshot_hash(ids, labels)
    question_keys = ["grailqa::1", "grailqa::2"]
    manifest = {
        "question_keys": question_keys,
        "failure_treatment": "ITT_EMPTY_METHOD_FAILURE_AND_SELECTOR_BASELINE_FALLBACK_V1",
        "api_caps": {
            "model": "gpt-5.6-luna",
            "api_request_cap": 10,
            "cost_cap_usd": 1.0,
            "per_request_input_token_cap": 7000,
            "per_request_completion_token_cap": 1024,
            "per_request_timeout_seconds": 120,
            "rate_assumption_input_uncached_usd_per_million": 0.2,
            "rate_assumption_output_usd_per_million": 1.2,
        },
        "sparql_caps": {"physical_read_only_select_cap": 10,"per_case_select_cap": 4,"endpoint":"http://127.0.0.1:8890/sparql"},
        "time_caps": {"wall_clock_cap_hours": 4},
        "analysis_plan": {"minimum_eligible_cases_for_scoring": 1},
        "intervention": {
            "max_candidate_count": 100,
            "max_input_tokens": 7000,
            "prompt_sha256": "p" * 64,
            "fallback": "invalid or ambiguous selector result preserves baseline",
        },
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)
    cases = [{
        "case_id": "case-1",
        "question_key": question_keys[0],
        "case_state": "PAIRED_VALID",
        "baseline_status": "BASELINE_COMPLETED",
        "baseline_response": {"finish_reason": "stop", "content_nonempty": True},
        "question": "which entity is linked to the topic?",
        "route": "cor",
        "selector_applied": True,
        "selector_eligible": True,
        "terminal_source": "direct-LLM-reasoning-STOP",
        "answer_kind": "entity",
        "executed_relation_ids": ["people.person.place_of_birth"],
        "filter_event_count": 0,
        "candidate_query_status": "ok",
        "candidate_query_attempt_id": "kg-1",
        "candidate_query_sha256": "qhash-1",
        "candidate_ids": ids,
        "candidate_labels": labels,
        "literal_candidates": [],
        "candidate_snapshot_sha256": digest,
        "baseline_prediction_ids": ["m.a"],
        "baseline_answer_names": ["Alpha"],
        "selector_prompt_input_token_estimate": 100,
        "intervention_prediction_ids": ["m.b"],
        "selector_request_count": 1,
        "selector_result": {
            "status": "VALID_SELECTION", "finish_reason": "stop",
            "content_nonempty": True, "selected_ids": ["m.b"],
        },
    }, {
        "case_id": "case-2",
        "question_key": question_keys[1],
        "case_state": "PAIRED_VALID",
        "baseline_status": "BASELINE_COMPLETED",
        "baseline_response": {"finish_reason": "stop", "content_nonempty": True},
        "question": "which entity is linked to the topic?",
        "route": "cor",
        "selector_applied": False,
        "selector_eligible": False,
        "terminal_source": "direct-LLM-reasoning-STOP",
        "answer_kind": "entity",
        "executed_relation_ids": ["people.person.place_of_birth"],
        "filter_event_count": 0,
        "candidate_query_status": "ok",
        "candidate_query_attempt_id": "kg-2",
        "candidate_query_sha256": "qhash-2",
        "candidate_ids": ["m.c"],
        "candidate_labels": ["Gamma"],
        "literal_candidates": [],
        "candidate_snapshot_sha256": candidate_snapshot_hash(["m.c"], ["Gamma"]),
        "baseline_prediction_ids": ["m.c"],
        "baseline_answer_names": ["Gamma"],
        "selector_prompt_input_token_estimate": 100,
        "intervention_prediction_ids": ["m.c"],
        "selector_request_count": 0,
        "selector_not_called_reason": "TRIGGER_INELIGIBLE",
    }]
    raw_requests = [
        ("case-1", "baseline", "request-1"),
        ("case-1", "selector", "request-2"),
        ("case-2", "baseline", "request-3"),
    ]
    request_events = []
    for index, (case_id, role, label) in enumerate(raw_requests):
        timestamp_minute = 1 + index
        request_events.append({
            "case_id": case_id,
            "request_role": role,
            "model": "gpt-5.6-luna",
            "request_sha256": __import__("hashlib").sha256(label.encode()).hexdigest(),
            "attempt_number": 1,
            "started_at_utc": f"2026-11-08T22:{timestamp_minute:02d}:00Z",
            "finished_at_utc": f"2026-11-08T22:{timestamp_minute:02d}:01Z",
            "input_tokens": 100,
            "uncached_input_tokens": 100,
            "output_tokens": 10,
            "max_completion_tokens": 100,
            "finish_reason": "stop",
            "content_nonempty": True,
            "transport_outcome": "HTTP_OK",
            **({"prompt_sha256": "p" * 64} if role == "selector" else {}),
        })
    sparql_events = [
        {"attempt_id": "preflight", "attempt_kind":"endpoint_preflight","endpoint":"http://127.0.0.1:8890/sparql","query_type": "SELECT", "read_only": True, "status": "ok", "attempt_number": 1, "started_at_utc": "2026-11-08T22:00:00Z", "query_sha256": "preflight-hash"},
        {"attempt_id": "kg-1", "attempt_kind":"case","case_id":"case-1","endpoint":"http://127.0.0.1:8890/sparql","query_type": "SELECT", "read_only": True, "status": "ok", "attempt_number": 1, "started_at_utc": "2026-11-08T22:01:00Z", "query_sha256": "qhash-1"},
        {"attempt_id": "kg-2", "attempt_kind":"case","case_id":"case-2","endpoint":"http://127.0.0.1:8890/sparql","query_type": "SELECT", "read_only": True, "status": "ok", "attempt_number": 1, "started_at_utc": "2026-11-08T22:02:00Z", "query_sha256": "qhash-2"},
    ]
    measured_cost = 3 * (100 * 0.2 + 10 * 1.2) / 1_000_000
    run = {
        "manifest_sha256": manifest["manifest_sha256"],
        "model": "gpt-5.6-luna",
        "sparql_endpoint":"http://127.0.0.1:8890/sparql",
        "runner_state": "COMPLETE",
        "failure_treatment": "ITT_EMPTY_METHOD_FAILURE_AND_SELECTOR_BASELINE_FALLBACK_V1",
        "run_started_at_utc": "2026-11-08T22:00:00Z",
        "hard_deadline_utc": "2026-11-09T00:00:00Z",
        "run_finished_at_utc": "2026-11-08T23:00:00Z",
        "accounting": {
            "api_requests_started": 3, "api_request_cap": 10,
            "physical_sparql_attempts": 3, "sparql_attempt_cap": 10,
            "input_tokens": 300, "output_tokens": 30,
            "rate_based_cost_usd": measured_cost,
        },
        "request_events": request_events,
        "sparql_events": sparql_events,
        "cases": cases,
    }
    authorization = {
        "authorized": True,
        "manifest_sha256": manifest["manifest_sha256"],
        "authorization_text": "test fixture exact-hash authorization",
    }
    return run, manifest, authorization


class TerminalBoundaryTests(unittest.TestCase):
    def observe(self, **updates):
        payload = direct_fixture()
        payload.update(updates)
        return observe_terminal_boundary(**payload)

    def test_direct_stop_exposes_exact_candidate_ids(self):
        row = self.observe()
        self.assertEqual(row["terminal_answer_source"], "direct-LLM-reasoning-STOP")
        self.assertEqual(row["candidate_source_search_trace_id"], 4)
        self.assertTrue(row["answer_id_mapping_valid"])
        self.assertTrue(row["candidate_selector_strict_eligible"])

    def test_filter_answer_is_a_distinct_terminal_path(self):
        payload = direct_fixture()
        payload["prompt_history"] = [{
            "type": "filter_answer", "trace_id": 7, "parsed_result": "Constraint",
            "answers": ["Beta"],
        }]
        payload["sparql_history"][0]["trace_id"] = 6
        payload["result"] = {"action": "Constraint", "results": [{"id": "m.b"}]}
        row = observe_terminal_boundary(**payload)
        self.assertEqual(row["terminal_answer_source"], "graph-filter")
        self.assertEqual(row["filter_event_count"], 1)
        self.assertTrue(row["answer_id_mapping_valid"])

    def test_search_exhaustion_direct_generation_has_no_bound_candidates(self):
        payload = direct_fixture()
        payload["prompt_history"] = [{"type": "generate_without_explored_paths", "trace_id": 8}]
        payload["result"] = {"action": "generate_directly", "results": [{"id": ""}]}
        row = observe_terminal_boundary(**payload)
        self.assertEqual(row["terminal_answer_source"], "direct-LLM-search-exhaustion-fallback")
        self.assertFalse(row["candidate_selector_strict_eligible"])

    def test_aggregate_scalar_is_not_entity_selectable(self):
        payload = direct_fixture()
        payload["question"] = "how many entities are there?"
        payload["prompt_history"][0]["answers"] = ["3"]
        payload["result"] = {"action": "Stop", "results": [{"id": ""}]}
        row = observe_terminal_boundary(**payload)
        self.assertFalse(row["answer_id_mapping_valid"])
        self.assertFalse(row["candidate_selector_strict_eligible"])

    def test_truncated_filter_call_is_visible_from_request_ledger(self):
        payload = direct_fixture()
        payload["baseline_status"] = "UPSTREAM_EMPTY_OR_TRUNCATED_FAILURE"
        payload["prompt_history"][0]["parsed_result"] = "Constraint"
        payload["request_events"] = [
            {"event": "REQUEST_STARTED", "request_sha256": "r1", "estimated_input_tokens": 100,
             "request_body": {"max_completion_tokens": 1024}},
            {"event": "REQUEST_FINISHED", "request_sha256": "r1", "finish_reason": "length",
             "raw_response_body": json.dumps({"choices": [{"finish_reason": "length",
             "message": {"content": ""}}]})},
        ]
        row = observe_terminal_boundary(**payload)
        self.assertEqual(row["terminal_answer_source"], "empty/error")
        self.assertEqual(row["next_path_or_failure"], "filter-request-failed-before-filter-event-append")
        self.assertEqual(row["request_prompt_event_gap"], 0)

    def test_branch_merge_is_not_misclassified_as_single_path_selection(self):
        payload = direct_fixture()
        payload["result"] = {"action": "branch_merge", "branch_traces": [], "results": [{"id": "m.a"}]}
        row = observe_terminal_boundary(**payload)
        self.assertEqual(row["terminal_answer_source"], "branch-merge")
        self.assertFalse(row["candidate_selector_strict_eligible"])

    def test_duplicate_candidate_labels_are_ambiguous(self):
        payload = direct_fixture()
        payload["sparql_history"][0]["candidate_entity_names"] = ["Same", "Same"]
        payload["prompt_history"][0]["answers"] = ["Same"]
        row = observe_terminal_boundary(**payload)
        self.assertEqual(row["candidate_id_set_status"], "INVALID_OR_AMBIGUOUS_ID_LABEL_MAPPING")
        self.assertFalse(row["answer_id_mapping_valid"])

    def test_actual_iter27_trace_inventory_and_event_gaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = build_iter27_matrix(ROOT, Path(temporary) / "matrix.csv")
        rows = result["rows"]
        self.assertEqual(len(rows), 40)
        self.assertEqual(sum(row["baseline_status"] == "BASELINE_COMPLETED" for row in rows), 38)
        self.assertEqual(sum(row["filter_event_count"] == 1 for row in rows), 4)
        self.assertEqual(sum(row["candidate_selector_strict_eligible"] for row in rows), 13)
        self.assertEqual(sum(row["request_prompt_event_gap"] == 1 for row in rows), 2)
        self.assertEqual(sum(row["next_path_or_failure"] == "filter-request-failed-before-filter-event-append" for row in rows), 2)

    def test_runtime_source_wiring_matches_terminal_audit(self):
        cor_source = (ROOT / "chain_of_relations/methods/cor/agent.py").read_text(encoding="utf-8")
        adaptive_source = (ROOT / "chain_of_relations/methods/adaptive_v1_1_3/agent.py").read_text(encoding="utf-8")
        run_source = (ROOT / "scripts/diagnostics/iter_027_results_first_answer_repair/run_live.py").read_text(encoding="utf-8")
        tree = ast.parse(cor_source)
        cor_answer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CoRAgent")
        answer_fn = next(node for node in cor_answer.body if isinstance(node, ast.FunctionDef) and node.name == "answer")
        source = ast.get_source_segment(cor_source, answer_fn) or ""
        self.assertIn('if action.lower() == "stop"', source)
        self.assertIn('if action.lower() == "constraint"', source)
        self.assertIn("self.validate(", source)
        self.assertIn("self.generate_directly(", source)
        validate_fn = next(node for node in cor_answer.body if isinstance(node, ast.FunctionDef) and node.name == "validate")
        validate_source = ast.get_source_segment(cor_source, validate_fn) or ""
        run_filter_at = validate_source.index("run_filter(")
        normal_append_at = validate_source.index('"type": "filter_answer"', run_filter_at)
        self.assertLess(run_filter_at, normal_append_at)
        self.assertLess(validate_source.index('"type": "filter_answer"'), run_filter_at)
        self.assertIn("if not decision.validation_passed or decision.execution_mode != \"branch\"", adaptive_source)
        self.assertIn('event.get("type") == "filter_answer"', run_source)
        self.assertIn('baseline.get("router_execution_mode", "cor")', run_source)


class ScoringGateTests(unittest.TestCase):
    def validate(self, run, manifest, authorization):
        return validate_for_scoring(run, manifest, authorization)

    def test_valid_pair_passes_external_gate(self):
        run, manifest, authorization = valid_run()
        result = self.validate(run, manifest, authorization)
        self.assertEqual(result["verdict"], "SCOREABLE")
        self.assertFalse(result["gold_scoring_performed"])

    def test_missing_exact_hash_authorization_blocks_scoring(self):
        run, manifest, _ = valid_run()
        result = validate_for_scoring(run, manifest, None)
        self.assertEqual(result["verdict"], "NOT_SCOREABLE")
        self.assertIn("EXACT_HASH_AUTHORIZATION_RECORD_MISSING", result["reasons"])

    def test_manifest_tampering_blocks_scoring(self):
        run, manifest, authorization = valid_run()
        manifest["api_caps"]["api_request_cap"] = 999
        result = self.validate(run, manifest, authorization)
        self.assertIn("MANIFEST_HASH_INVALID", result["reasons"])

    def test_runner_cannot_self_select_a_lower_accounting_cap(self):
        run, manifest, authorization = valid_run()
        run["accounting"]["api_request_cap"] = 1
        result = self.validate(run, manifest, authorization)
        self.assertIn("API_REQUEST_CAP_DIFFERS_FROM_MANIFEST", result["reasons"])

    def test_non_read_only_sparql_event_blocks_scoring(self):
        run, manifest, authorization = valid_run()
        run["sparql_events"][1]["read_only"] = False
        result = self.validate(run, manifest, authorization)
        self.assertIn("NON_READ_ONLY_OR_NON_SELECT_ATTEMPT", result["reasons"])

    def test_per_case_sparql_cap_is_enforced(self):
        run, manifest, authorization = valid_run()
        for index in range(4):
            event = dict(run["sparql_events"][1])
            event["attempt_id"] = f"kg-extra-{index}"
            run["sparql_events"].append(event)
        run["accounting"]["physical_sparql_attempts"] = 7
        result = self.validate(run, manifest, authorization)
        self.assertIn("PER_CASE_SPARQL_CAP_INVALID_OR_EXCEEDED", result["reasons"])

    def test_run_endpoint_must_match_frozen_manifest(self):
        run, manifest, authorization = valid_run()
        run["sparql_endpoint"] = "http://127.0.0.1:9999/sparql"
        result = self.validate(run, manifest, authorization)
        self.assertIn("RUN_SPARQL_ENDPOINT_DIFFERS_FROM_MANIFEST", result["reasons"])

    def test_selector_prompt_hash_must_match_frozen_manifest(self):
        run, manifest, authorization = valid_run()
        run["request_events"][1]["prompt_sha256"] = "x" * 64
        result = self.validate(run, manifest, authorization)
        self.assertIn("SELECTOR_PROMPT_HASH_DIFFERS_FROM_MANIFEST", result["reasons"])

    def test_40_rows_with_38_valid_pairs_fails_manifest_and_prediction_checks(self):
        run, manifest, authorization = valid_run()
        template = run["cases"][1]
        run["cases"] = []
        expected = [f"grailqa::{index:02d}" for index in range(1, 41)]
        manifest["question_keys"] = expected
        manifest["manifest_sha256"] = manifest_digest(manifest)
        run["manifest_sha256"] = manifest["manifest_sha256"]
        authorization["manifest_sha256"] = manifest["manifest_sha256"]
        for index, question_key in enumerate(expected, 1):
            case = dict(template)
            case["case_id"] = f"case-{index:02d}"
            case["question_key"] = question_key
            case["baseline_prediction_ids"] = ["m.c"]
            case["intervention_prediction_ids"] = ["m.c"]
            case["case_state"] = "PAIRED_VALID" if index <= 38 else "ATTEMPTED"
            case["baseline_status"] = "BASELINE_COMPLETED" if index <= 38 else "BASELINE_ATTEMPTED"
            case["baseline_response"] = {"finish_reason": "stop", "content_nonempty": True} if index <= 38 else {}
            case["selector_eligible"] = False
            case["selector_applied"] = False
            case["selector_request_count"] = 0
            run["cases"].append(case)
        result = self.validate(run, manifest, authorization)
        self.assertEqual(result["verdict"], "NOT_SCOREABLE")
        self.assertIn("case-39:CASE_NOT_PAIRED_VALID", result["reasons"])
        self.assertIn("case-39:BASELINE_STATUS_INVALID", result["reasons"])

    def test_two_unhandled_length_completions_are_not_scored(self):
        run, manifest, authorization = valid_run()
        run["request_events"][0]["finish_reason"] = "length"
        run["request_events"][1]["finish_reason"] = "length"
        result = self.validate(run, manifest, authorization)
        self.assertEqual(result["verdict"], "NOT_SCOREABLE")
        self.assertIn("UNHANDLED_NONSTOP_RESPONSE", result["reasons"])

    def test_omitted_candidate_ids_fail_closed(self):
        run, manifest, authorization = valid_run()
        run["cases"][0]["candidate_ids"] = []
        result = self.validate(run, manifest, authorization)
        self.assertTrue(any("CANDIDATE_IDS_OR_LABELS_INVALID" in reason for reason in result["reasons"]))

    def test_repeated_request_hash_is_rejected_as_possible_retry(self):
        run, manifest, authorization = valid_run()
        run["request_events"][1]["request_sha256"] = run["request_events"][0]["request_sha256"]
        result = self.validate(run, manifest, authorization)
        self.assertIn("REPEATED_REQUEST_HASH_POSSIBLE_RETRY", result["reasons"])

    def test_malformed_selector_without_explicit_baseline_fallback_is_rejected(self):
        run, manifest, authorization = valid_run()
        run["cases"][0]["selector_result"] = {"status": "INVALID_JSON", "selected_ids": []}
        result = self.validate(run, manifest, authorization)
        self.assertIn("case-1:SELECTOR_RESULT_STATUS_INVALID", result["reasons"])

    def test_preregistered_invalid_selector_falls_back_exactly_to_baseline(self):
        run, manifest, authorization = valid_run()
        case = run["cases"][0]
        case["intervention_prediction_ids"] = list(case["baseline_prediction_ids"])
        case["selector_fallback_preregistered"] = True
        case["selector_result"] = {
            "status": "INVALID_FALLBACK_BASELINE", "selected_ids": [],
            "fallback_reason": "malformed_json", "finish_reason": "stop",
            "content_nonempty": True,
        }
        result = self.validate(run, manifest, authorization)
        self.assertEqual(result["verdict"], "SCOREABLE")

    def test_zero_eligible_cases_is_mechanism_failure_not_scoreable(self):
        run, manifest, authorization = valid_run()
        run["cases"][0]["question"] = "how many entities are linked to the topic?"
        run["cases"][0]["selector_applied"] = False
        run["cases"][0]["selector_eligible"] = False
        run["cases"][0]["selector_request_count"] = 0
        run["cases"][0]["selector_not_called_reason"] = "TRIGGER_INELIGIBLE"
        run["cases"][0]["intervention_prediction_ids"] = list(run["cases"][0]["baseline_prediction_ids"])
        run["request_events"] = [event for event in run["request_events"] if event["request_role"] != "selector"]
        run["accounting"]["api_requests_started"] = 2
        run["accounting"]["input_tokens"] = 200
        run["accounting"]["output_tokens"] = 20
        run["accounting"]["rate_based_cost_usd"] = 2 * (100 * 0.2 + 10 * 1.2) / 1_000_000
        result = self.validate(run, manifest, authorization)
        self.assertIn("ZERO_ELIGIBLE_CASES_IS_MECHANISM_FAILURE_NOT_ACCURACY_RESULT", result["reasons"])

    def test_multiple_filters_cannot_trigger_selector(self):
        run, manifest, authorization = valid_run()
        run["cases"][0]["terminal_source"] = "graph-filter"
        run["cases"][0]["filter_event_count"] = 2
        result = self.validate(run, manifest, authorization)
        self.assertTrue(any("FILTER_BOUNDARY_INVALID" in reason for reason in result["reasons"]))

    def test_aggregate_answer_cannot_trigger_selector(self):
        run, manifest, authorization = valid_run()
        run["cases"][0]["answer_kind"] = "aggregate"
        result = self.validate(run, manifest, authorization)
        self.assertTrue(any("ANSWER_NOT_ENTITY" in reason for reason in result["reasons"]))

    def test_failed_candidate_query_cannot_trigger_selector(self):
        run, manifest, authorization = valid_run()
        run["cases"][0]["candidate_query_status"] = "error"
        result = self.validate(run, manifest, authorization)
        self.assertTrue(any("CANDIDATE_QUERY_PROVENANCE_INVALID" in reason for reason in result["reasons"]))

    def test_baseline_method_failure_requires_preregistered_empty_itt_pair(self):
        run, manifest, authorization = valid_run()
        case = run["cases"][1]
        case["baseline_status"] = "BASELINE_FAILURE_ITT"
        case["baseline_failure_kind"] = "input_cap"
        case["baseline_response"] = {"finish_reason": "not_sent", "content_nonempty": False}
        case["baseline_prediction_ids"] = []
        case["intervention_prediction_ids"] = []
        run["request_events"] = [event for event in run["request_events"] if not (event["case_id"] == "case-2" and event["request_role"] == "baseline")]
        run["accounting"]["api_requests_started"] = 2
        run["accounting"]["input_tokens"] = 200
        run["accounting"]["output_tokens"] = 20
        run["accounting"]["rate_based_cost_usd"] = 2 * (100 * 0.2 + 10 * 1.2) / 1_000_000
        result = self.validate(run, manifest, authorization)
        self.assertEqual(result["verdict"], "SCOREABLE")

    def test_hard_deadline_rejects_late_run_or_request(self):
        run, manifest, authorization = valid_run()
        run["request_events"][0]["finished_at_utc"] = "2026-11-09T00:01:00Z"
        result = self.validate(run, manifest, authorization)
        self.assertIn("REQUEST_FINISHED_AFTER_HARD_DEADLINE", result["reasons"])

    def test_late_run_completion_is_rejected(self):
        run, manifest, authorization = valid_run()
        run["run_finished_at_utc"] = "2026-11-09T00:01:00Z"
        result = self.validate(run, manifest, authorization)
        self.assertIn("RUN_FINISHED_AFTER_HARD_DEADLINE", result["reasons"])

    def test_request_cap_is_enforced(self):
        run, manifest, authorization = valid_run()
        run["accounting"]["api_requests_started"] = 11
        result = self.validate(run, manifest, authorization)
        self.assertIn("API_REQUESTS_STARTED_MANIFEST_CAP_INVALID_OR_EXCEEDED", result["reasons"])


if __name__ == "__main__":
    unittest.main()
