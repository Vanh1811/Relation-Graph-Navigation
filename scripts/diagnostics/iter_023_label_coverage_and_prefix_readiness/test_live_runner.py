"""Offline tests for Iteration 23's live-run gates and shared meters."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_iter23_pilot import (  # noqa: E402
    MODEL,
    PilotRuntime,
    perform_approved_endpoint_preflight,
)


MANIFEST = "ff57d84266295e7778b00e733fdba9e8a9232f4eed87971f7b0e029cefc331f2"


def approval_record() -> dict:
    return {
        "approval_verbatim": (
            "I authorize Iteration 23 manifest " + MANIFEST +
            ": three fixed repeat-2 development slots using gpt-5.6-luna at "
            "US$0.20 per million uncached input tokens and US$1.20 per million output tokens; "
            "up to 60 API requests, 7000 input tokens/request, 1024 completion tokens/request, "
            "up to US$0.16 estimated API spend, up to 140 physical read-only SPARQL SELECT "
            "attempts including one endpoint preflight, up to 8 hours total, 120-second maximum "
            "per request, and no retries."
        ),
        "approved_at": "2026-11-08T04:08:12Z",
        "manifest_sha256": MANIFEST,
        "authorization_status": "APPROVED",
        "model": MODEL,
        "api_request_cap": 60,
        "input_tokens_per_request_cap": 7000,
        "completion_tokens_per_request_cap": 1024,
        "estimated_api_spend_cap_usd": 0.16,
        "input_usd_per_million": 0.2,
        "output_usd_per_million": 1.2,
        "read_only_sparql_query_cap": 140,
        "endpoint_preflight_selects_included": 1,
        "time_cap_seconds": 28800,
        "per_call_timeout_seconds": 120,
        "retries": 0,
        "transport_opened": False,
        "live_execution_started": False,
    }


class FakeRecorder:
    def __init__(self) -> None:
        self.current_case_id = "test"
        self.events = []

    def meter_event(self, event) -> None:
        self.events.append(dict(event))

    @staticmethod
    def query_transport(transport):
        def audited(query):
            return transport(query)
        return audited


class Iter23LiveRunnerTests(unittest.TestCase):
    def runtime(self) -> PilotRuntime:
        runtime = PilotRuntime(approval_record(), {}, "http://fixture.invalid/sparql")
        runtime.recorder = FakeRecorder()
        return runtime

    def test_one_read_only_query_is_metered_before_transport(self):
        runtime = self.runtime()
        calls = []

        def transport(query):
            calls.append(query)
            return {"status": "ok", "rows": []}

        result = runtime.call_sparql_once(
            approval=runtime.approval,
            query_text="SELECT ?s WHERE { ?s ?p ?o } LIMIT 1",
            one_attempt_transport=transport,
        )
        self.assertEqual("ok", result["status"])
        self.assertEqual(1, runtime.live_sparql_attempts)
        self.assertEqual(1, len(calls))
        self.assertEqual("ATTEMPT_STARTED", runtime.recorder.events[0]["status"])

    def test_successful_preflight_releases_transport_for_case_adapter(self):
        runtime = self.runtime()
        calls = []
        runtime.recorder.sparql_transport = lambda query: calls.append(query) or {"status": "ok", "rows": []}
        result = perform_approved_endpoint_preflight(runtime, runtime.recorder)
        self.assertEqual("ok", result["status"])
        self.assertEqual(1, runtime.live_sparql_attempts)
        self.assertEqual(["SELECT * WHERE { ?s ?p ?o } LIMIT 1"], calls)
        self.assertIsNone(runtime.current_transport)
        self.assertIsNone(runtime.recorder.current_case_id)

    def test_resumed_meter_starts_after_saved_preflight_without_replay(self):
        previous = {
            "event": "REUSED_SUCCESSFUL_AUTHORIZED_PREFLIGHT",
            "query_index": 1,
            "query_sha256": "d" * 64,
            "category": "endpoint_preflight",
            "retries": 0,
        }
        runtime = PilotRuntime(
            approval_record(), {}, "http://fixture.invalid/sparql",
            initial_query_attempts=1,
            initial_query_events=[previous],
            initial_query_categories={"endpoint_preflight": 1},
        )
        runtime.recorder = FakeRecorder()
        calls = []
        runtime.call_sparql_once(
            approval=runtime.approval,
            query_text="SELECT ?s WHERE { ?s ?p ?o } LIMIT 1",
            one_attempt_transport=lambda query: calls.append(query) or {"status": "ok", "rows": []},
        )
        self.assertEqual(2, runtime.live_sparql_attempts)
        self.assertEqual(2, runtime.query_meter.events[-1]["query_index"])
        self.assertEqual(1, len(calls))

    def test_failed_query_consumes_attempt_and_is_not_retried(self):
        runtime = self.runtime()
        calls = []

        def fail_once(_query):
            calls.append("one")
            raise TimeoutError("fixture timeout")

        with self.assertRaises(Exception):
            runtime.call_sparql_once(
                approval=runtime.approval,
                query_text="SELECT ?s WHERE { ?s ?p ?o } LIMIT 1",
                one_attempt_transport=fail_once,
            )
        self.assertEqual(1, runtime.live_sparql_attempts)
        self.assertEqual(["one"], calls)

    def test_global_query_cap_blocks_before_a_second_transport(self):
        runtime = self.runtime()
        runtime.query_meter.hard_cap = 1
        calls = []
        send = lambda query: calls.append(query) or {"status": "ok", "rows": []}
        runtime.call_sparql_once(approval=runtime.approval,
                                 query_text="SELECT ?s WHERE { ?s ?p ?o } LIMIT 1",
                                 one_attempt_transport=send)
        with self.assertRaises(Exception):
            runtime.call_sparql_once(approval=runtime.approval,
                                     query_text="SELECT ?p WHERE { ?s ?p ?o } LIMIT 1",
                                     one_attempt_transport=send)
        self.assertEqual(1, runtime.live_sparql_attempts)
        self.assertEqual(1, len(calls))

    def test_non_select_is_blocked_without_transport(self):
        runtime = self.runtime()
        calls = []
        with self.assertRaises(Exception):
            runtime.call_sparql_once(approval=runtime.approval,
                                     query_text="DELETE WHERE { ?s ?p ?o }",
                                     one_attempt_transport=lambda q: calls.append(q))
        self.assertEqual(0, runtime.live_sparql_attempts)
        self.assertEqual([], calls)

    def test_deadline_blocks_before_transport(self):
        runtime = self.runtime()
        runtime.deadline.deadline_at = 0
        calls = []
        with self.assertRaises(Exception):
            runtime.call_sparql_once(approval=runtime.approval,
                                     query_text="SELECT ?s WHERE { ?s ?p ?o } LIMIT 1",
                                     one_attempt_transport=lambda q: calls.append(q))
        self.assertEqual(0, runtime.live_sparql_attempts)
        self.assertEqual([], calls)

    def test_model_request_uses_actual_usage_and_exact_rate(self):
        runtime = self.runtime()
        content, usage = runtime.call_llm_once(
            approval=runtime.approval,
            input_tokens=7000,
            completion_tokens=1024,
            prompt_sha256="a" * 64,
            transport=lambda: ("Forward", {"input_tokens": 100, "output_tokens": 20}),
        )
        self.assertEqual("Forward", content)
        self.assertEqual(1, runtime.live_api_attempts)
        self.assertEqual(0.000044, round(runtime.live_cost_usd, 12))

    def test_model_failure_is_charged_at_reservation_and_not_retried(self):
        runtime = self.runtime()
        calls = []

        def fail_once():
            calls.append("one")
            raise TimeoutError("fixture timeout")

        with self.assertRaises(Exception):
            runtime.call_llm_once(
                approval=runtime.approval, input_tokens=7000,
                completion_tokens=1024, prompt_sha256="b" * 64,
                transport=fail_once,
            )
        self.assertEqual(1, runtime.live_api_attempts)
        self.assertEqual(["one"], calls)
        self.assertEqual(0.0026288, round(runtime.uncertain_live_cost_usd, 12))

    def test_model_token_and_spend_caps_block_before_transport(self):
        runtime = self.runtime()
        calls = []
        with self.assertRaises(Exception):
            runtime.call_llm_once(
                approval=runtime.approval, input_tokens=7001,
                completion_tokens=1024, prompt_sha256="c" * 64,
                transport=lambda: calls.append("sent"),
            )
        self.assertEqual(0, runtime.live_api_attempts)
        self.assertEqual([], calls)


if __name__ == "__main__":
    unittest.main()
