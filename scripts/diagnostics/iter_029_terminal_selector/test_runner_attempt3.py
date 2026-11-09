from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scripts.diagnostics.iter_029_terminal_selector.scoring_gate_attempt3 import validate_for_scoring, _parse_executed_relation_path
from scripts.diagnostics.iter_029_terminal_selector.runner_attempt3 import (
    FAILURE_TREATMENT,
    _build_run_document,
    CaseLocalApiFailure,
    execute_cohort,
    manifest_digest,
    run_live,
    sha256_bytes,
    validate_authorization,
    write_json_fsync,
)


PROMPT = (
    'Select only IDs from the supplied candidate list. Return '
    '{"status":"SELECT","selected_ids":["m.beta"]} or UNSURE.'
)
PROMPT_SHA = hashlib.sha256(PROMPT.encode()).hexdigest()
VALID_SELECTOR_RESPONSE = '{"status":"SELECT","selected_ids":["m.beta"]}'


def _stamp(offset: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset)).isoformat()


def make_manifest(keys: list[str]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "iteration": 29,
        "dataset": "GrailQA",
        "split": "dev",
        "question_keys": keys,
        "failure_treatment": FAILURE_TREATMENT,
        "analysis_plan": {"minimum_eligible_cases_for_scoring": 1},
        "time_caps": {"wall_clock_cap_hours": 1},
        "api_caps": {
            "model": "gpt-5.6-luna",
            "api_request_cap": 10,
            "per_request_input_token_cap": 100,
            "per_request_completion_token_cap": 16,
            "cost_cap_usd": 0.01,
            "max_retries": 0,
            "resume_allowed": False,
            "per_request_timeout_seconds": 120,
            "rate_assumption_input_uncached_usd_per_million": 0.2,
            "rate_assumption_output_usd_per_million": 1.2,
        },
        "sparql_caps": {
            "endpoint": "http://127.0.0.1:8890/sparql",
            "mutations_allowed": False,
            "per_case_select_cap": 16,
            "physical_read_only_select_cap": 20,
            "preflight_selects_included": 1,
        },
        "intervention": {
            "max_candidate_count": 100,
            "max_input_tokens": 100,
            "max_completion_tokens": 16,
            "prompt_sha256": PROMPT_SHA,
            "fallback": "invalid selector preserves baseline exactly",
            "reasoning_effort": "low",
        },
        "status": "FROZEN_WITH_STANDING_USER_AUTHORIZATION",
    }
    manifest["api_caps"]["cost_cap_usd"] = 10 * (100 * 0.2 + 16 * 1.2) / 1_000_000
    manifest["manifest_sha256"] = manifest_digest(manifest)
    return manifest


def make_authorization(manifest: dict[str, Any]) -> dict[str, Any]:
    api, sparql = manifest["api_caps"], manifest["sparql_caps"]
    return {
        "authorized": True,
        "manifest_sha256": manifest["manifest_sha256"],
        "authorization_text": "Synthetic test authorization; no external call is made.",
        "accepted_caps": {
            "model": api["model"],
            "api_request_cap": api["api_request_cap"],
            "per_request_input_token_cap": api["per_request_input_token_cap"],
            "per_request_completion_token_cap": api["per_request_completion_token_cap"],
            "cost_cap_usd": api["cost_cap_usd"],
            "physical_read_only_select_cap": sparql["physical_read_only_select_cap"],
            "endpoint": sparql["endpoint"],
            "per_request_timeout_seconds": api["per_request_timeout_seconds"],
            "wall_clock_cap_hours": manifest["time_caps"]["wall_clock_cap_hours"],
            "max_retries": 0,
            "resume_allowed": False,
        },
    }


class FakeGateway:
    def __init__(self, selector_content: str = '{"status":"SELECT","selected_ids":["m.beta"]}') -> None:
        self.halted = False
        self.active_case_id: str | None = None
        self.request_role = "baseline"
        self.normalized_prompt_sha = PROMPT_SHA
        self.token_counter = lambda _messages: 24
        self.request_events: list[dict[str, Any]] = []
        self.sparql_events: list[dict[str, Any]] = []
        self.successful_query_by_sha: dict[str, dict[str, Any]] = {}
        self.case_select_counts: dict[str, int] = {}
        self.per_case_select_cap = 4
        self.selector_content = selector_content
        self.selector_calls = 0
        self.fail_next_baseline_http = False
        self.fail_next_selector_http = False

    def set_case_context(self, case_id: str | None) -> None:
        self.active_case_id = case_id

    def select_once(self, query: str) -> dict[str, Any]:
        digest = sha256_bytes(query.encode())
        case_id = self.active_case_id or ""
        if case_id:
            self.case_select_counts[case_id] = self.case_select_counts.get(case_id, 0) + 1
        event = {
            "attempt_id": f"sparql-{len(self.sparql_events) + 1}",
            "attempt_kind": "case" if case_id else "endpoint_preflight",
            "case_id": case_id,
            "endpoint": "http://127.0.0.1:8890/sparql",
            "query_type": "SELECT",
            "read_only": True,
            "query_sha256": digest,
            "attempt_number": 1,
            "retry_of": None,
            "started_at_utc": _stamp(),
            "status": "ok",
            "transport_outcome": "HTTP_OK",
        }
        self.sparql_events.append(event)
        self.successful_query_by_sha[digest] = event
        return {"bindings": [], "raw_response_sha256": digest}

    def call_model_once(self, messages: list[dict[str, str]], *, max_completion_tokens: int, reasoning_effort: str) -> dict[str, Any]:
        role = self.request_role
        baseline_length = role == "baseline" and "length failure" in messages[-1]["content"]
        content = self.selector_content if role == "selector" else ("truncated baseline" if baseline_length else "baseline answer")
        if role == "selector":
            self.selector_calls += 1
        prompt_digest = sha256_bytes(json.dumps(messages, sort_keys=True, separators=(",", ":")).encode())
        event = {
            "request_sha256": hashlib.sha256(f"req-{len(self.request_events)}".encode()).hexdigest(),
            "prompt_sha256": self.normalized_prompt_sha if role == "selector" else prompt_digest,
            "case_id": str(self.active_case_id or ""),
            "request_role": role,
            "model": "gpt-5.6-luna",
            "attempt_number": 1,
            "retry_of": None,
            "started_at_utc": _stamp(),
            "finished_at_utc": _stamp(1),
            "input_tokens": 24,
            "uncached_input_tokens": 24,
            "output_tokens": 3,
            "usage_known": True,
            "reserved_cost_usd": 0.0000392,
            "max_completion_tokens": max_completion_tokens,
            "finish_reason": "length" if baseline_length else "stop",
            "content_nonempty": bool(content),
            "transport_outcome": "HTTP_OK",
            "http_status": 200,
            "failure_policy_applied": baseline_length,
            "failure_outcome": "BASELINE_METHOD_FAILURE" if baseline_length else None,
        }
        should_fail = (role == "baseline" and self.fail_next_baseline_http) or (role == "selector" and self.fail_next_selector_http)
        if should_fail:
            self.fail_next_baseline_http = False
            self.fail_next_selector_http = False
            event.update({
                "input_tokens": 0, "uncached_input_tokens": 0, "output_tokens": 0,
                "usage_known": False, "finish_reason": "http_error", "raw_finish_reason": None,
                "content_nonempty": False, "transport_outcome": "HTTP_ERROR", "http_status": 429,
                "raw_response_sha256": "a" * 64, "failure_policy_applied": True,
                "failure_outcome": "BASELINE_METHOD_FAILURE" if role == "baseline" else "SELECTOR_BASELINE_FALLBACK",
                "error_type": "HTTPStatusError", "error_code": "rate_limit_exceeded",
            })
        self.request_events.append(event)
        if should_fail:
            raise CaseLocalApiFailure(
                "API_HTTP_ERROR", http_status=429, error_code="rate_limit_exceeded",
                error_type="HTTPStatusError", raw_response_sha256="a" * 64,
            )
        return {"content": content, "finish_reason": "length" if baseline_length else "stop", "usage": {"input_tokens": 24, "output_tokens": 3}}


class FakeAgent:
    def __init__(self, gateway: FakeGateway) -> None:
        self.gateway = gateway


def fake_answer_once(agent: FakeAgent, *, question: str, topic_entities: list[Any], prompt_history: list[dict[str, Any]], sparql_history: list[dict[str, Any]], step_history: list[dict[str, Any]]) -> tuple[dict[str, Any], None]:
    agent.gateway.request_role = "baseline"
    agent.gateway.call_model_once([{"role": "user", "content": "baseline:" + question}], max_completion_tokens=16, reasoning_effort="low")
    if "length failure" in question:
        return None, {"failure_kind": "EMPTY_OR_TRUNCATED_RESPONSE", "finish_reason": "length", "content_empty": False}
    ids = ["m.alpha", "m.beta"] if "eligible" in question else ["m.alpha"]
    labels = ["Alpha", "Beta"] if len(ids) == 2 else ["Alpha"]
    query = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 10"
    agent.gateway.select_once(query)
    sparql_history.append({
        "search_type": "entity_search", "trace_id": 1, "sparql": query,
        "status": "ok", "timed_out": False,
        "candidate_entity_ids": ids, "candidate_entity_names": labels,
        "literal_candidates": [],
    })
    prompt_history.append({"type": "reasoning", "trace_id": 2, "parsed_result": "Stop"})
    baseline_id = "m.alpha"
    return ({
        "results": [{"id": baseline_id, "name": "Alpha"}],
        "final_execution_mode": "cor",
        "reasoning_chains": "TopicEntity(Topic) people.person.place_of_birth e1 .\ne1 candidate.target TargetEntity(Alpha, Beta) .",
    }, None)


class Iter29RunnerOfflineTests(unittest.TestCase):
    def _run(
        self, questions: list[str], selector_content: str,
        *, fail_baseline_http: bool = False, fail_selector_http: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        manifest = make_manifest([f"grailqa::{index}" for index in range(len(questions))])
        cohort = [
            {"case_id": f"case-{i}", "question_key": manifest["question_keys"][i], "question": question,
             "topic_entities": ["m.topic"], "topic_entity_names": ["Topic"]}
            for i, question in enumerate(questions)
        ]
        gateway = FakeGateway(selector_content)
        gateway.fail_next_baseline_http = fail_baseline_http
        gateway.fail_next_selector_http = fail_selector_http
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            output.mkdir()
            cases, abort = execute_cohort(
                manifest, cohort, gateway, FakeAgent(gateway), fake_answer_once,
                lambda entity_id, label: (entity_id, label), PROMPT, output,
            )
            self.assertIsNone(abort)
            started = _stamp(-1)
            deadline = (datetime.fromisoformat(started) + timedelta(hours=1)).isoformat()
            finished = _stamp(2)
            run = _build_run_document(manifest, started, deadline, finished, cases, gateway.request_events, gateway.sparql_events, "COMPLETE")
            run_path = output / "run.json"
            manifest_path = output / "manifest.json"
            auth_path = output / "authorization.json"
            write_json_fsync(run_path, run)
            write_json_fsync(manifest_path, manifest)
            authorization = make_authorization(manifest)
            write_json_fsync(auth_path, authorization)
            from scripts.diagnostics.iter_029_terminal_selector.runner_attempt3 import _invoke_external_gate
            verdict = _invoke_external_gate(run_path, manifest_path, auth_path)
            return run, authorization, verdict

    def test_mocked_full_pairing_validates_through_external_gate(self) -> None:
        run, authorization, verdict = self._run(["eligible question", "ordinary question"], VALID_SELECTOR_RESPONSE)
        self.assertEqual(len(run["cases"]), 2)
        self.assertEqual(run["cases"][0]["intervention_prediction_ids"], ["m.beta"])
        self.assertEqual(run["cases"][1]["intervention_prediction_ids"], run["cases"][1]["baseline_prediction_ids"])
        self.assertEqual(verdict["verdict"], "SCOREABLE", verdict)
        self.assertFalse(verdict["gold_scoring_performed"])

    def test_invalid_selector_preserves_baseline_exactly_and_is_scoreable(self) -> None:
        run, _, verdict = self._run(["eligible question"], '{"status":"SELECT","selected_ids":["m.unknown"]}')
        case = run["cases"][0]
        self.assertEqual(case["selector_result"]["status"], "INVALID_FALLBACK_BASELINE")
        self.assertEqual(case["intervention_prediction_ids"], case["baseline_prediction_ids"])
        self.assertEqual(verdict["verdict"], "SCOREABLE", verdict)

    def test_zero_eligibility_is_mechanism_failure_not_scoreable(self) -> None:
        run, _, verdict = self._run(["ordinary one", "ordinary two"], VALID_SELECTOR_RESPONSE)
        self.assertEqual(sum(case["selector_eligible"] for case in run["cases"]), 0)
        self.assertTrue(all(case["intervention_prediction_ids"] == case["baseline_prediction_ids"] for case in run["cases"]))
        self.assertEqual(verdict["verdict"], "NOT_SCOREABLE")
        self.assertIn("ZERO_ELIGIBLE_CASES_IS_MECHANISM_FAILURE_NOT_ACCURACY_RESULT", verdict["reasons"])

    def test_length_failure_is_empty_paired_itt_and_other_cases_continue(self) -> None:
        run, _, verdict = self._run(["length failure case", "eligible question"], VALID_SELECTOR_RESPONSE)
        failed = run["cases"][0]
        self.assertEqual(failed["baseline_status"], "BASELINE_FAILURE_ITT")
        self.assertEqual(failed["baseline_failure_kind"], "length")
        self.assertEqual(failed["baseline_prediction_ids"], [])
        self.assertEqual(failed["intervention_prediction_ids"], [])
        self.assertEqual(verdict["verdict"], "SCOREABLE", verdict)

    def test_baseline_http_error_is_empty_itt_and_later_cases_continue(self) -> None:
        run, _, verdict = self._run(
            ["first baseline fails", "eligible question"], VALID_SELECTOR_RESPONSE,
            fail_baseline_http=True,
        )
        failed = run["cases"][0]
        self.assertEqual(failed["baseline_status"], "BASELINE_FAILURE_ITT")
        self.assertEqual(failed["baseline_failure_kind"], "http_error")
        self.assertEqual(failed["baseline_response"]["http_status"], 429)
        self.assertEqual(failed["baseline_prediction_ids"], [])
        self.assertEqual(failed["intervention_prediction_ids"], [])
        self.assertEqual(run["cases"][1]["baseline_status"], "BASELINE_COMPLETED")
        self.assertEqual(verdict["verdict"], "SCOREABLE", verdict)

    def test_selector_http_error_falls_back_exactly_to_baseline(self) -> None:
        run, _, verdict = self._run(
            ["eligible question"], VALID_SELECTOR_RESPONSE,
            fail_selector_http=True,
        )
        case = run["cases"][0]
        self.assertEqual(case["selector_result"]["status"], "INVALID_FALLBACK_BASELINE")
        self.assertEqual(case["selector_result"]["fallback_reason"], "API_HTTP_ERROR")
        self.assertEqual(case["intervention_prediction_ids"], case["baseline_prediction_ids"])
        self.assertEqual(verdict["verdict"], "SCOREABLE", verdict)

    def test_aggregate_question_does_not_trigger_selector(self) -> None:
        run, _, verdict = self._run(["how many eligible entities?", "eligible question"], VALID_SELECTOR_RESPONSE)
        aggregate = run["cases"][0]
        self.assertFalse(aggregate["selector_eligible"])
        self.assertEqual(aggregate["selector_request_count"], 0)
        self.assertEqual(aggregate["intervention_prediction_ids"], aggregate["baseline_prediction_ids"])
        self.assertEqual(verdict["verdict"], "SCOREABLE", verdict)

    def test_mandatory_first_hundred_zero_trigger_stop(self) -> None:
        manifest = make_manifest([f"grailqa::{index}" for index in range(100)])
        cohort = [
            {"case_id": f"case-{i}", "question_key": manifest["question_keys"][i], "question": f"ordinary {i}",
             "topic_entities": ["m.topic"], "topic_entity_names": ["Topic"]}
            for i in range(100)
        ]
        gateway = FakeGateway(VALID_SELECTOR_RESPONSE)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            output.mkdir()
            cases, abort = execute_cohort(
                manifest, cohort, gateway, FakeAgent(gateway), fake_answer_once,
                lambda entity_id, label: (entity_id, label), PROMPT, output,
            )
            self.assertEqual(len(cases), 100)
            self.assertEqual(abort, {"status": "MECHANISM_FAILURE_NO_ELIGIBILITY", "completed_cases": 100})
            self.assertEqual(gateway.selector_calls, 0)

    def test_exact_hash_and_caps_required_before_authorization_acceptance(self) -> None:
        manifest = make_manifest(["grailqa::1"])
        auth = make_authorization(manifest)
        auth["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "exact manifest hash"):
            validate_authorization({**manifest, "status": "FROZEN_WITH_STANDING_USER_AUTHORIZATION"}, manifest["manifest_sha256"], auth)

    def test_live_wiring_checks_hash_and_authorization_before_transport_setup(self) -> None:
        source = inspect.getsource(run_live)
        self.assertLess(source.index("manifest_digest(manifest)"), source.index("validate_authorization("))
        self.assertLess(source.index("validate_authorization("), source.index("load_safe_project_env()"))
        self.assertLess(source.index("verify_inputs("), source.index("load_safe_project_env()"))
        self.assertIn("make_attempt4_tgar_agent(gateway)", source)
        self.assertIn("execute_cohort(", source)
        self.assertIn("_invoke_external_gate(", source)
        self.assertNotIn("score_completed_run", source)


class Iter29Attempt2PathParserTests(unittest.TestCase):
    def test_outgoing_incoming_and_branch_edges_parse_in_path_order(self) -> None:
        path = (
            "TopicEntity(A topic with spaces) people.person.parents e1 .\n"
            "e2 people.person.children e1 .\n"
            "e1 people.person.parents e3 .\n"
            "e3 candidate.target TargetEntity(Answer labels) ."
        )
        expected = ["people.person.parents", "people.person.children"]
        from scripts.diagnostics.iter_029_terminal_selector.path_parser_attempt2 import parse_executed_relation_ids
        self.assertEqual(parse_executed_relation_ids(path), expected)
        self.assertEqual(_parse_executed_relation_path(path), expected)

    def test_target_annotation_is_never_a_relation_and_empty_path_stays_empty(self) -> None:
        from scripts.diagnostics.iter_029_terminal_selector.path_parser_attempt2 import parse_executed_relation_ids
        self.assertEqual(parse_executed_relation_ids("e1 candidate.target TargetEntity(Answer) ."), [])
        self.assertEqual(parse_executed_relation_ids(""), [])
        self.assertEqual(_parse_executed_relation_path(""), [])

    def test_gate_detects_runner_recording_that_disagrees_with_raw_path(self) -> None:
        path = "TopicEntity(Topic) people.person.place_of_birth e1 .\ne1 candidate.target TargetEntity(Alpha) ."
        self.assertEqual(_parse_executed_relation_path(path), ["people.person.place_of_birth"])
        self.assertNotEqual([""], _parse_executed_relation_path(path))


if __name__ == "__main__":
    unittest.main()
