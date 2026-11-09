from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts.diagnostics.iter_026_answer_filter.gateway import BudgetExhausted
from scripts.diagnostics.iter_026_answer_filter.gateway_attempt4 import Attempt4Gateway, InputTokenLimitExceeded
from scripts.diagnostics.iter_026_answer_filter.strict_runtime_attempt4 import (
    Attempt4LLMAPI,
    InputCapCaseFailure,
    answer_case_once,
    selector_call_or_cap_fallback,
)


class FakeModelTransport:
    max_retries = 0

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(body)
        return {
            "http_status": 200,
            "content": "ok",
            "finish_reason": "stop",
            "raw_response_sha256": "a" * 64,
            "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
        }


class FakeSparqlTransport:
    max_retries = 0

    def __call__(self, _query: str) -> dict[str, Any]:
        return {"http_status": 200, "bindings": []}


def make_gateway(directory: str, model: FakeModelTransport, token_count: int) -> Attempt4Gateway:
    return Attempt4Gateway(
        Path(directory),
        manifest_sha256="a" * 64,
        model_transport=model,
        sparql_transport=FakeSparqlTransport(),
        api_request_cap=2,
        api_cost_cap_usd=1.0,
        input_tokens_per_request=7,
        output_tokens_per_request=4,
        input_usd_per_million=0.20,
        output_usd_per_million=1.20,
        sparql_attempt_cap=2,
        timeout_seconds=2,
        wall_seconds=20,
        token_counter=lambda _messages: token_count,
    )


class InputCapHandlingTests(unittest.TestCase):
    def test_oversize_is_local_unreserved_and_does_not_halt_or_send(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModelTransport()
            gateway = make_gateway(directory, model, token_count=8)
            with self.assertRaises(InputTokenLimitExceeded) as caught:
                gateway.call_model_once([{"role": "user", "content": "too large"}], max_completion_tokens=4)
            self.assertEqual(caught.exception.metadata()["estimated_input_tokens"], 8)
            self.assertFalse(caught.exception.metadata()["api_request_sent"])
            self.assertEqual(gateway.api_starts, 0)
            self.assertEqual(gateway.api_reserved_usd, 0.0)
            self.assertFalse(gateway.halted)
            self.assertEqual(model.calls, [])

    def test_baseline_llm_maps_only_input_cap_to_case_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            gateway = make_gateway(directory, FakeModelTransport(), token_count=8)
            llm = Attempt4LLMAPI(gateway)
            with self.assertRaises(InputCapCaseFailure) as caught:
                llm.generate("too large", max_tokens=4)
            self.assertEqual(caught.exception.metadata["failure_kind"], "INPUT_CAP_EXCEEDED")
            self.assertEqual(gateway.api_starts, 0)
            self.assertFalse(gateway.halted)

    def test_selector_cap_falls_back_and_next_small_request_can_run(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModelTransport()
            gateway = make_gateway(directory, model, token_count=8)
            result, failure = selector_call_or_cap_fallback(
                gateway, [{"role": "user", "content": "too large"}], max_completion_tokens=4,
            )
            self.assertIsNone(result)
            self.assertEqual(failure["status"], "INPUT_CAP_EXCEEDED_BEFORE_TRANSPORT")
            self.assertEqual(model.calls, [])
            self.assertFalse(gateway.halted)
            # Simulate a later question whose local token estimate is under cap.
            gateway.token_counter = lambda _messages: 4
            ok = gateway.call_model_once([{"role": "user", "content": "small"}], max_completion_tokens=4)
            self.assertEqual(ok["content"], "ok")
            self.assertEqual(gateway.api_starts, 1)
            request_events = [
                line for line in (Path(directory) / "REQUEST_JOURNAL.jsonl").read_text().splitlines()
                if '"event":"REQUEST_STARTED"' in line
            ]
            self.assertEqual(len(request_events), 1)

    def test_global_budget_exhaustion_is_not_converted_to_case_failure(self):
        class GlobalBudgetFailure:
            def call_model_once(self, *_args: Any, **_kwargs: Any) -> Any:
                raise BudgetExhausted("global cap reached")

        with self.assertRaises(BudgetExhausted):
            selector_call_or_cap_fallback(
                GlobalBudgetFailure(), [{"role": "user", "content": "small"}],
                max_completion_tokens=4,
            )

    def test_answer_case_records_input_cap_failure_and_preserves_runtime_boundary(self):
        class OversizedAgent:
            def answer(self, **_kwargs: Any) -> Any:
                raise InputCapCaseFailure({
                    "failure_kind": "INPUT_CAP_EXCEEDED",
                    "estimated_input_tokens": 8,
                    "input_token_limit": 7,
                })

        result, failure = answer_case_once(OversizedAgent())
        self.assertIsNone(result)
        self.assertEqual(failure["failure_kind"], "INPUT_CAP_EXCEEDED")
        self.assertEqual(failure["estimated_input_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
