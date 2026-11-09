from __future__ import annotations

import hashlib
import json
import ssl
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from urllib.parse import parse_qs

from gateway import (
    AttemptUncertain,
    BudgetExhausted,
    DuplicatePrompt,
    ResearchCallOnceGateway,
    ResumeForbidden,
)
import transport
from transport import FreebaseSelectTransport, OpenAIChatCompletionsTransport


class FakeResponse:
    def __init__(self, status: int, body: bytes):
        self.status = status
        self.body = body

    def read(self):
        return self.body


class FakeConnection:
    def __init__(self, response: FakeResponse, calls: list[dict]):
        self.response = response
        self.calls = calls
        self.closed = False

    def request(self, method, path, body=None, headers=None):
        self.calls.append({"method": method, "path": path, "body": body, "headers": headers})

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class FileCountingModelTransport:
    max_retries = 0

    def __init__(self, call_file: Path, request_journal: Path, attempt_journal: Path):
        self.call_file = call_file
        self.request_journal = request_journal
        self.attempt_journal = attempt_journal

    def __call__(self, request_body):
        with self.call_file.open("a") as stream:
            stream.write("call\n")
        request_events = [json.loads(line)["event"] for line in self.request_journal.read_text().splitlines()]
        attempt_events = [json.loads(line)["event"] for line in self.attempt_journal.read_text().splitlines()]
        assert request_events[-1] == "REQUEST_STARTED"
        assert attempt_events[-1] == "ATTEMPT_STARTED"
        return {
            "http_status": 200,
            "raw_response_sha256": "b" * 64,
            "raw_response_body": '{"choices":[]}',
            "content": "OK",
            "finish_reason": "stop",
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        }


class FakeSparqlTransport:
    max_retries = 0

    def __init__(self, call_file: Path):
        self.call_file = call_file

    def __call__(self, query):
        with self.call_file.open("a") as stream:
            stream.write(query + "\n")
        return {
            "http_status": 200,
            "raw_response_sha256": "c" * 64,
            "raw_response_body": '{"results":{"bindings":[]}}',
            "bindings": [],
        }


class MissingUsageModelTransport:
    max_retries = 0

    def __call__(self, request_body):
        return {
            "http_status": 200,
            "raw_response_sha256": "d" * 64,
            "raw_response_body": '{"choices":[]}',
            "content": "OK",
            "finish_reason": "stop",
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }


def make_gateway(directory: Path, *, transport=None, sparql=None, calls=2, cost=1.0, queries=2):
    output = directory / "live"
    return ResearchCallOnceGateway(
        output,
        manifest_sha256="a" * 64,
        model_transport=transport or FileCountingModelTransport(
            directory / "model.calls", output / "REQUEST_JOURNAL.jsonl", output / "ATTEMPT_JOURNAL.jsonl"
        ),
        sparql_transport=sparql or FakeSparqlTransport(directory / "sparql.calls"),
        api_request_cap=calls,
        api_cost_cap_usd=cost,
        input_tokens_per_request=7000,
        output_tokens_per_request=1024,
        input_usd_per_million=0.20,
        output_usd_per_million=1.20,
        sparql_attempt_cap=queries,
        timeout_seconds=2,
        wall_seconds=30,
        token_counter=lambda _: 20,
    )


class TransportAdapterTests(unittest.TestCase):
    def test_default_openai_connection_uses_installed_verified_certifi_bundle(self):
        response_body = json.dumps({
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
        }).encode()
        calls = []
        contexts = []

        def factory(host, timeout, context):
            self.assertEqual(host, "api.openai.com")
            self.assertEqual(timeout, 120.0)
            contexts.append(context)
            return FakeConnection(FakeResponse(200, response_body), calls)

        body = {
            "model": "gpt-5.6-luna",
            "messages": [{"role": "user", "content": "Reply OK"}],
            "max_completion_tokens": 16,
            "reasoning_effort": "low",
            "stream": False,
        }
        with patch.object(transport.http.client, "HTTPSConnection", side_effect=factory):
            result = OpenAIChatCompletionsTransport("secret-test-value")(body)

        self.assertEqual(result["content"], "OK")
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0].verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(contexts[0].check_hostname)
        self.assertGreater(contexts[0].cert_store_stats()["x509_ca"], 0)

    def test_openai_adapter_sends_one_supported_request_and_captures_finish_metadata(self):
        response_body = json.dumps({
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
        }).encode()
        calls = []
        connections = []

        def factory(host, timeout):
            connection = FakeConnection(FakeResponse(200, response_body), calls)
            connections.append((host, timeout, connection))
            return connection

        transport = OpenAIChatCompletionsTransport("secret-test-value", connection_factory=factory)
        result = transport({
            "model": "gpt-5.6-luna",
            "messages": [{"role": "user", "content": "Reply OK"}],
            "max_completion_tokens": 16,
            "reasoning_effort": "low",
            "stream": False,
        })
        self.assertEqual(transport.max_retries, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["method"], "POST")
        self.assertEqual(calls[0]["path"], "/v1/chat/completions")
        self.assertNotIn("temperature", json.loads(calls[0]["body"]))
        self.assertEqual(result["content"], "OK")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["raw_response_sha256"], hashlib.sha256(response_body).hexdigest())
        self.assertEqual(result["usage"]["input_tokens"], 12)
        self.assertTrue(connections[0][2].closed)
        self.assertNotIn("secret-test-value", json.dumps(result))

    def test_openai_adapter_rejects_unfrozen_parameters_before_connection(self):
        calls = []
        transport = OpenAIChatCompletionsTransport(
            "secret-test-value", connection_factory=lambda *_args, **_kwargs: calls.append("opened")
        )
        with self.assertRaises(ValueError):
            transport({"model": "gpt-5.6-luna", "messages": [], "temperature": 0.1})
        self.assertEqual(calls, [])

    def test_sparql_adapter_is_one_direct_post_to_only_the_local_tunnel(self):
        body = b'{"results":{"bindings":[{"s":{"type":"uri","value":"urn:x"}}]}}'
        calls = []

        def factory(host, port, timeout):
            self.assertEqual((host, port), ("127.0.0.1", 8890))
            return FakeConnection(FakeResponse(200, body), calls)

        transport = FreebaseSelectTransport(connection_factory=factory)
        result = transport("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")
        self.assertEqual(transport.max_retries, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["method"], "POST")
        self.assertEqual(parse_qs(calls[0]["body"].decode())["format"], ["json"])
        self.assertEqual(len(result["bindings"]), 1)
        with self.assertRaises(ValueError):
            FreebaseSelectTransport("https://example.com/sparql")


class GatewayAttemptTests(unittest.TestCase):
    def test_durable_attempt_journal_precedes_transport_and_duplicate_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gateway = make_gateway(root)
            messages = [{"role": "user", "content": "Reply exactly OK."}]
            result = gateway.call_model_once(messages, max_completion_tokens=16)
            self.assertEqual(result["finish_reason"], "stop")
            with self.assertRaises(DuplicatePrompt):
                gateway.call_model_once(messages, max_completion_tokens=16)
            self.assertTrue(gateway.halted)
            self.assertEqual((root / "model.calls").read_text().splitlines(), ["call"])
            request_events = [json.loads(line) for line in (root / "live/REQUEST_JOURNAL.jsonl").read_text().splitlines()]
            self.assertEqual(request_events[0]["event"], "REQUEST_STARTED")
            self.assertEqual(request_events[1]["event"], "REQUEST_FINISHED")
            self.assertEqual(request_events[1]["finish_reason"], "stop")
            self.assertEqual(request_events[1]["raw_response_sha256"], "b" * 64)
            self.assertEqual(request_events[2]["event"], "DUPLICATE_PROMPT_BLOCKED")

    def test_incomplete_usage_metadata_fails_closed_and_halts_after_one_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gateway = make_gateway(root, transport=MissingUsageModelTransport())
            with self.assertRaises(AttemptUncertain):
                gateway.call_model_once([{"role": "user", "content": "one"}], max_completion_tokens=8)
            self.assertEqual(gateway.api_starts, 1)
            self.assertTrue(gateway.halted)
            events = [json.loads(line) for line in (root / "live/REQUEST_JOURNAL.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["status"], "UNCERTAIN_OR_FAILED_NO_RETRY")

    def test_full_worst_case_reservation_blocks_before_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gateway = make_gateway(root, cost=0.001)
            with self.assertRaises(BudgetExhausted):
                gateway.call_model_once([{"role": "user", "content": "one"}], max_completion_tokens=16)
            self.assertFalse((root / "model.calls").exists())
            self.assertEqual(gateway.api_starts, 0)

    def test_select_only_query_is_journaled_and_successful_repeats_use_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gateway = make_gateway(root)
            query = "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?s WHERE { ?s ?p ?o } LIMIT 1"
            gateway.select_once(query)
            gateway.select_once(query)
            self.assertEqual((root / "sparql.calls").read_text(), query + "\n")
            events = [json.loads(line) for line in (root / "live/SPARQL_QUERY_JOURNAL.jsonl").read_text().splitlines()]
            self.assertEqual([row["event"] for row in events], ["QUERY_STARTED", "QUERY_FINISHED", "QUERY_CACHE_HIT"])
            with self.assertRaises(ValueError):
                gateway.select_once("DELETE WHERE { ?s ?p ?o }")

    def test_no_retry_declaration_and_nonempty_journal_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                make_gateway(root, transport=lambda _: {})
            gateway = make_gateway(root)
            with self.assertRaises(ResumeForbidden):
                make_gateway(root)


if __name__ == "__main__":
    unittest.main()
