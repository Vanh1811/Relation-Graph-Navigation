from __future__ import annotations

import unittest

from scripts.diagnostics.iter_026_answer_filter.strict_runtime import (
    EmptyOrTruncatedModelResponse,
    StrictGatewayLLMAPI,
    answer_case_once,
    selector_response_metadata,
)


class StubGateway:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def call_model_once(self, messages, *, max_completion_tokens, reasoning_effort):
        self.calls.append((messages, max_completion_tokens, reasoning_effort))
        return self.results.pop(0)


def response(content, finish_reason, digest):
    return {
        "content": content,
        "finish_reason": finish_reason,
        "raw_response_sha256": digest,
        "usage": {"input_tokens": 13, "output_tokens": 1024, "total_tokens": 1037},
    }


class StrictRuntimeTests(unittest.TestCase):
    def test_truncated_response_preserves_metadata_and_is_one_call(self):
        gateway = StubGateway([response("", "length", "raw-sha")])
        api = StrictGatewayLLMAPI(gateway)
        with self.assertRaises(EmptyOrTruncatedModelResponse) as raised:
            api.generate("fixed prompt", max_tokens=1024)
        self.assertEqual(len(gateway.calls), 1)
        self.assertEqual(raised.exception.metadata["finish_reason"], "length")
        self.assertEqual(raised.exception.metadata["raw_response_sha256"], "raw-sha")
        self.assertEqual(raised.exception.metadata["usage"]["output_tokens"], 1024)

    def test_empty_stopped_response_is_also_case_failure(self):
        gateway = StubGateway([response("  ", "stop", "empty-sha")])
        with self.assertRaises(EmptyOrTruncatedModelResponse):
            StrictGatewayLLMAPI(gateway).generate("prompt")
        self.assertEqual(len(gateway.calls), 1)

    def test_case_boundary_records_failure_and_next_case_can_continue(self):
        from chain_of_relations.methods.cor.tools import filter as filter_module

        class Agent:
            def __init__(self, api):
                self.api = api

            def answer(self, question, **_):
                text, _usage = filter_module._run_llm(
                    self.api.generate,
                    question,
                    temperature=0.01,
                    max_tokens=1024,
                    max_retries=3,
                )
                return {"results": [{"id": text}], "router_execution_mode": "cor"}

        gateway = StubGateway([
            response("", "length", "first-sha"),
            response("m.good", "stop", "second-sha"),
        ])
        agent = Agent(StrictGatewayLLMAPI(gateway))
        result, failure = answer_case_once(agent, question="case one")
        self.assertIsNone(result)
        self.assertEqual(failure["finish_reason"], "length")
        self.assertEqual(len(gateway.calls), 1)

        result, failure = answer_case_once(agent, question="case two")
        self.assertIsNone(failure)
        self.assertEqual(result["results"], [{"id": "m.good"}])
        self.assertEqual(len(gateway.calls), 2)

    def test_selector_metadata_keeps_actual_transport_fields(self):
        details = selector_response_metadata(response('{"ok":true}', "stop", "body-sha"))
        self.assertEqual(details["finish_reason"], "stop")
        self.assertEqual(details["raw_response_sha256"], "body-sha")
        self.assertEqual(details["usage"]["input_tokens"], 13)


if __name__ == "__main__":
    unittest.main()
