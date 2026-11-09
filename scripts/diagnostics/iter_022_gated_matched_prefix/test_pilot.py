from __future__ import annotations

import hashlib
import unittest
from types import SimpleNamespace

from pilot import (
    HardCapMeter,
    LabelCacheMiss,
    MAX_API_COST_USD,
    MAX_API_REQUESTS,
    MAX_COMPLETION_TOKENS,
    MAX_INPUT_TOKENS,
    MAX_SPARQL_QUERIES,
    PilotStop,
    approval_matches,
    dry_run_case,
    estimated_cost,
    install_metered_backend,
    install_metered_llm,
    require_read_only_select,
    sparqlwrapper_one_attempt,
    summarize_dry_run,
)


def approval_fixture():
    return {
        "approval_verbatim": (
            "I authorize Iteration 22 manifest `f8a0a15de59e7a4cd3a20d351ad549e41d4135ad114f2d8527bbda9e0ce953d9`: "
            "three fixed questions, repeat 1 only; up to 60 API requests, 7,000 input tokens/request, "
            "1,024 completion tokens/request, up to US$0.17, up to 100 read-only SPARQL queries, and no retries."
        ),
        "approved_at": "2026-11-08T00:00:00Z",
        "manifest_sha256": "f8a0a15de59e7a4cd3a20d351ad549e41d4135ad114f2d8527bbda9e0ce953d9",
        "api_request_cap": 60,
        "input_tokens_per_request_cap": 7000,
        "completion_tokens_per_request_cap": 1024,
        "estimated_api_spend_cap_usd": 0.17,
        "read_only_sparql_query_cap": 100,
        "retries": 0,
    }


def prefix_fixture(**updates):
    snapshot = {
        "question_id": "synthetic-q1", "repeat": 1, "boundary_depth": 3,
        "boundary_action": "Forward", "route": "cor", "evidence_pointer": "synthetic://fixture",
        "model_settings": {"model": "fixture-model", "temperature": 0, "max_completion_tokens": 64},
        "decisions": [{"stage": "reasoning", "prompt": "synthetic prompt", "response": "Forward"}],
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
        "active_binding_rows": None,
        "frontier_ids": ["m.c"],
        "visited_node_ids": ["m.root", "m.a", "m.b", "m.c"],
        "candidate_relation_ids_before_prune": ["r.next", "r.alt"],
        "answer_source": "OTHER", "filter_input_ids": None, "filter_output_ids": None,
        "final_answer_ids": [],
    }
    snapshot.update(updates)
    return snapshot


def projection_fixture():
    query = "SELECT DISTINCT ?a ?b ?endpoint WHERE {}"
    return {"status": "ok", "query": query,
            "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest(), "limit": 100,
            "rows": [{"?a": "m.a", "?b": "m.b", "?endpoint": "m.c"}]}


class Iteration22PilotTests(unittest.TestCase):
    def test_manifest_approval_requires_exact_frozen_caps(self):
        record = approval_fixture()
        self.assertTrue(approval_matches(record))
        record["read_only_sparql_query_cap"] = 101
        self.assertFalse(approval_matches(record))
        self.assertFalse(approval_matches(None))

    def test_verbatim_approval_must_name_all_hard_caps(self):
        record = approval_fixture()
        record["approval_verbatim"] = "I approve this experiment."
        self.assertFalse(approval_matches(record))

    def test_read_only_guard_accepts_select_and_rejects_updates(self):
        require_read_only_select("# comment\nPREFIX ns: <http://example.org/>\nSELECT ?s WHERE {}")
        for query in ("INSERT DATA { <s> <p> <o> }", "DELETE WHERE { ?s ?p ?o }", "ASK {}", ""):
            with self.subTest(query=query):
                with self.assertRaises(PilotStop):
                    require_read_only_select(query)

    def test_dry_run_builds_same_parent_one_hop_pair(self):
        meter = HardCapMeter()
        outcome = dry_run_case(
            snapshot=prefix_fixture(), projection=projection_fixture(),
            projection_variable_order=["?a", "?b", "?endpoint"],
            llm_response="r.next", parsed_selection=[{"id": "r.next", "direction": "OUTGOING"}],
            kg_query="synthetic one-edge query", kg_rows=[{"?e3": "m.c", "?e4": "m.new"}], meter=meter,
        )
        self.assertEqual("SYNTHETIC_PAIR_VALID", outcome["status"])
        self.assertEqual(outcome["stop_parent_hash"], outcome["one_hop_parent_hash"])
        self.assertEqual(1, outcome["new_edge_count"])
        self.assertEqual(0, meter.live_api_attempts)
        self.assertEqual(0, meter.live_sparql_attempts)

    def test_non_forward_and_incomplete_projection_are_ineligible_without_calls(self):
        meter = HardCapMeter()
        stopped = dry_run_case(
            snapshot=prefix_fixture(boundary_action="Finish"), projection=projection_fixture(),
            projection_variable_order=["?a", "?b", "?endpoint"], llm_response="r.next",
            parsed_selection=[{"id": "r.next", "direction": "OUTGOING"}],
            kg_query="synthetic", kg_rows=[], meter=meter)
        truncated = dry_run_case(
            snapshot=prefix_fixture(), projection={**projection_fixture(), "limit": 1},
            projection_variable_order=["?a", "?b", "?endpoint"], llm_response="r.next",
            parsed_selection=[{"id": "r.next", "direction": "OUTGOING"}],
            kg_query="synthetic", kg_rows=[], meter=meter)
        self.assertEqual("NON_FORWARD_BOUNDARY", stopped["reason"])
        self.assertEqual("INCOMPLETE_OR_MISMATCHED_BINDINGS", truncated["reason"])
        self.assertEqual(0, meter.simulated_api_attempts)
        self.assertEqual(0, meter.simulated_sparql_attempts)

    def test_empty_and_malformed_model_responses_are_metered_once_then_stopped(self):
        for response, selection, expected in [
            ("", [{"id": "r.next", "direction": "OUTGOING"}], "EMPTY_LLM_RESPONSE"),
            ("unparseable", None, "MALFORMED_OR_UNPARSED_RESPONSE"),
            ("[]", [], "EMPTY_SELECTION"),
            ("bad id", [{"id": "r.not-a-candidate", "direction": "OUTGOING"}], "MALFORMED_LLM_SELECTION"),
        ]:
            with self.subTest(expected=expected):
                meter = HardCapMeter()
                result = dry_run_case(
                    snapshot=prefix_fixture(), projection=projection_fixture(),
                    projection_variable_order=["?a", "?b", "?endpoint"], llm_response=response,
                    parsed_selection=selection, kg_query="synthetic", kg_rows=[], meter=meter)
                self.assertEqual(expected, result["reason"])
                self.assertEqual(1, meter.simulated_api_attempts)
                self.assertEqual(0, meter.simulated_sparql_attempts)

    def test_kg_error_is_counted_once_and_not_treated_as_empty_result(self):
        meter = HardCapMeter()
        result = dry_run_case(
            snapshot=prefix_fixture(), projection=projection_fixture(),
            projection_variable_order=["?a", "?b", "?endpoint"], llm_response="r.next",
            parsed_selection=[{"id": "r.next", "direction": "OUTGOING"}],
            kg_query="synthetic", kg_rows=[], kg_status="timeout", meter=meter)
        self.assertEqual("KG_QUERY_FAILURE_OR_CAP", result["reason"])
        self.assertEqual(1, meter.simulated_api_attempts)
        self.assertEqual(1, meter.simulated_sparql_attempts)

    def test_all_ineligible_cases_do_not_make_simulated_post_boundary_calls(self):
        meter = HardCapMeter()
        outcomes = [dry_run_case(
            snapshot=prefix_fixture(boundary_action="Finish"), projection=None,
            projection_variable_order=[], llm_response=None, parsed_selection=None,
            kg_query="", kg_rows=[], meter=meter) for _ in range(3)]
        report = summarize_dry_run(outcomes, meter)
        self.assertEqual("ALL_CASES_INELIGIBLE", report["status"])
        self.assertEqual(0, report["simulated_api_attempts"])
        self.assertEqual(0, report["simulated_sparql_attempts"])
        self.assertEqual(0, report["transport_calls"])

    def test_api_caps_allow_sixty_worst_case_reservations_then_stop(self):
        meter = HardCapMeter()
        for _ in range(MAX_API_REQUESTS):
            status = meter.simulate_api_attempt(input_tokens=MAX_INPUT_TOKENS,
                                                completion_tokens=MAX_COMPLETION_TOKENS,
                                                content="ok", parsed_selection=[{"id": "r", "direction": "OUTGOING"}])
            self.assertEqual("SCRIPTED_RESPONSE_PRESENT", status)
        self.assertAlmostEqual(0.157728, meter.simulated_worst_case_reserved_usd)
        self.assertLessEqual(meter.simulated_worst_case_reserved_usd, MAX_API_COST_USD)
        with self.assertRaisesRegex(PilotStop, "60-request cap"):
            meter.simulate_api_attempt(input_tokens=1, completion_tokens=1, content="x", parsed_selection=[])
        self.assertEqual(MAX_API_REQUESTS, meter.simulated_api_attempts)

    def test_api_input_and_output_reservations_reject_before_counting(self):
        meter = HardCapMeter()
        with self.assertRaisesRegex(PilotStop, "input reservation"):
            meter.simulate_api_attempt(input_tokens=MAX_INPUT_TOKENS + 1, completion_tokens=1, content="x", parsed_selection=[])
        with self.assertRaisesRegex(PilotStop, "completion cap"):
            meter.simulate_api_attempt(input_tokens=1, completion_tokens=MAX_COMPLETION_TOKENS + 1, content="x", parsed_selection=[])
        self.assertEqual(0, meter.simulated_api_attempts)

    def test_sparql_cap_allows_one_hundred_then_refuses_before_call_101(self):
        meter = HardCapMeter()
        for i in range(MAX_SPARQL_QUERIES):
            meter.simulate_sparql_attempt(f"synthetic-{i}")
        with self.assertRaisesRegex(PilotStop, "100-query cap"):
            meter.simulate_sparql_attempt("must-not-be-issued")
        self.assertEqual(MAX_SPARQL_QUERIES, meter.simulated_sparql_attempts)
        self.assertEqual(0, meter.live_sparql_attempts)

    def test_live_llm_call_requires_approval_before_transport(self):
        meter = HardCapMeter()
        transports = []
        with self.assertRaisesRegex(PilotStop, "approval record"):
            meter.call_llm_once(approval=None, input_tokens=5, completion_tokens=10,
                                transport=lambda: transports.append("called"))
        self.assertEqual([], transports)
        self.assertEqual(0, meter.live_api_attempts)

    def test_single_attempt_llm_meters_empty_error_and_does_not_retry(self):
        meter = HardCapMeter()
        attempts = []
        def fake_transport():
            attempts.append(1)
            return None, {"input_tokens": 0, "output_tokens": 0, "error": "empty"}
        with self.assertRaisesRegex(PilotStop, "full reservation"):
            meter.call_llm_once(approval=approval_fixture(), input_tokens=100, completion_tokens=16,
                                transport=fake_transport)
        self.assertEqual(1, len(attempts))
        self.assertEqual(1, meter.live_api_attempts)
        self.assertEqual(0, meter.live_cost_usd)
        self.assertGreater(meter.uncertain_live_cost_usd, 0)

    def test_single_attempt_sparql_error_consumes_one_slot_without_retry(self):
        meter = HardCapMeter()
        attempts = []
        def fake_transport(query):
            attempts.append(query)
            return {"status": "timeout", "rows": []}
        with self.assertRaisesRegex(PilotStop, "status is not ok"):
            meter.call_sparql_once(approval=approval_fixture(), query_text="SELECT * WHERE {}",
                                   one_attempt_transport=fake_transport)
        self.assertEqual(1, len(attempts))
        self.assertEqual(1, meter.live_sparql_attempts)

    def test_update_query_is_rejected_before_transport_or_metering(self):
        meter = HardCapMeter()
        attempts = []
        with self.assertRaisesRegex(PilotStop, "only read-only"):
            meter.call_sparql_once(approval=approval_fixture(), query_text="DROP ALL",
                                   one_attempt_transport=lambda query: attempts.append(query))
        self.assertEqual([], attempts)
        self.assertEqual(0, meter.live_sparql_attempts)

    def test_sparqlwrapper_transport_executes_one_select_without_retry(self):
        import sys
        import types
        from unittest.mock import patch

        client_instances = []
        class FakeClient:
            def __init__(self, endpoint):
                self.endpoint = endpoint
                self.query_calls = 0
                client_instances.append(self)
            def setQuery(self, query): self.query_text = query
            def setReturnFormat(self, fmt): self.return_format = fmt
            def setTimeout(self, timeout): self.timeout = timeout
            def query(self):
                self.query_calls += 1
                return types.SimpleNamespace(convert=lambda: {"results": {"bindings": [{"x": {"value": "m.a"}}]}})
        fake_module = types.ModuleType("SPARQLWrapper")
        fake_module.JSON = object()
        fake_module.SPARQLWrapper = FakeClient
        with self.assertRaisesRegex(PilotStop, "approval record"):
            sparqlwrapper_one_attempt("http://localhost:8890/sparql", approval=None)
        transport = sparqlwrapper_one_attempt("http://localhost:8890/sparql", approval=approval_fixture(), timeout_seconds=7)
        with patch.dict(sys.modules, {"SPARQLWrapper": fake_module}):
            result = transport("SELECT ?x WHERE {}")
            with self.assertRaises(PilotStop):
                transport("LOAD <file:///tmp/input.ttl>")
        self.assertEqual("ok", result["status"])
        self.assertEqual(1, len(client_instances))
        self.assertEqual(1, client_instances[0].query_calls)
        self.assertEqual(7, client_instances[0].timeout)

    def test_metered_backend_restores_methods_and_uses_only_cached_labels(self):
        class Backend:
            def execute_sparql_with_meta(self, query): return {"status": "old", "rows": []}
            def execute_sparql(self, query): return []
            def id2entity_names(self, ids): return {}
            def id2entity_name_or_type(self, entity_id): return "old"
        backend = Backend()
        original = Backend.execute_sparql_with_meta
        meter = HardCapMeter()
        calls = []
        with install_metered_backend(backend, meter, approval=approval_fixture(), label_cache={"m.a": "Cached A"},
                                     one_attempt_transport=lambda query: calls.append(query) or {"status": "ok", "rows": [{"x": 1}]}):
            self.assertEqual({"status": "ok", "rows": [{"x": 1}]}, backend.execute_sparql_with_meta("SELECT ?x WHERE {}"))
            self.assertEqual({"m.a": "Cached A"}, backend.id2entity_names(["m.a"]))
            with self.assertRaises(LabelCacheMiss):
                try:
                    backend.id2entity_names(["m.unknown"])
                except Exception:
                    pass  # Mirrors entity_search's broad fallback handler.
            self.assertEqual(1, meter.live_sparql_attempts)
        self.assertEqual(["SELECT ?x WHERE {}"], calls)
        self.assertNotIn("execute_sparql_with_meta", backend.__dict__)
        self.assertIs(original, backend.execute_sparql_with_meta.__func__)

    def test_metered_llm_checks_retry_configuration_and_restores_wrapper(self):
        class Api:
            max_retries = 1
            retry_interval = 0
            def __init__(self):
                self.client = SimpleNamespace(max_retries=0)
                self.calls = 0
            def generate(self, **kwargs):
                self.calls += 1
                return "ok", {"input_tokens": 2, "output_tokens": 1}
        class Agent:
            def __init__(self): self.llm_api = Api()
        agent, meter = Agent(), HardCapMeter()
        original = Api.generate
        with install_metered_llm(agent, meter, approval=approval_fixture(), token_counter=lambda text: 2):
            response = agent.llm_api.generate(user_prompt="fixture", max_tokens=8)
            self.assertEqual("ok", response[0])
        self.assertEqual(1, agent.llm_api.calls)
        self.assertEqual(1, meter.live_api_attempts)
        self.assertEqual(estimated_cost(2, 1), meter.live_cost_usd)
        self.assertNotIn("generate", agent.llm_api.__dict__)
        self.assertIs(original, agent.llm_api.generate.__func__)

    def test_metered_llm_fails_closed_if_sdk_retries_are_unknown(self):
        agent = SimpleNamespace(llm_api=SimpleNamespace(max_retries=1, retry_interval=0,
                                                        client=SimpleNamespace()))
        with self.assertRaisesRegex(PilotStop, "SDK retries"):
            with install_metered_llm(agent, HardCapMeter(), approval=approval_fixture(), token_counter=lambda _: 1):
                self.fail("must not install a wrapper with unknown SDK retry state")


if __name__ == "__main__":
    unittest.main()
