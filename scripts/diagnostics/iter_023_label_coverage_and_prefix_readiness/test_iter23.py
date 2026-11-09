from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/diagnostics/iter_021_matched_prefix"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics/iter_022_gated_matched_prefix"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_resolver import (  # noqa: E402
    BoundedLabelResolver,
    LabelResolutionStop,
    PhysicalQueryMeter,
    QueryBudgetExceeded,
    QueryExecutionFailed,
    UnmeteredQueryPath,
    build_label_query,
    install_metered_freebase_backend,
)
from model_budget import (  # noqa: E402
    INPUT_USD_PER_MILLION,
    OUTPUT_USD_PER_MILLION,
    MODEL_NAME,
    ModelBudgetStop,
    ModelCaps,
    NoRetryModelMeter,
    estimate_cost,
)
from deadline import DeadlineGuard  # noqa: E402
from pilot import dry_run_case  # noqa: E402
from replay import build_pair  # noqa: E402
from test_pilot import prefix_fixture, projection_fixture  # noqa: E402


KNOWN_MISSES = {
    "m.04st848", "m.04stk8g", "m.04stpzt", "m.04strb6",
    "m.0hz9k44", "m.0hzdjww", "m.0ndvpvf",
}


def binding(entity_id: str, label: str | None, language: str = "en") -> dict:
    entity = {"type": "uri", "value": f"http://rdf.freebase.com/ns/{entity_id}"}
    if label is None:
        return {"entity": entity}
    target = {"type": "literal", "value": label}
    if language:
        target["xml:lang"] = language
    return {"entity": entity, "targetEntity": target}


class Iteration23LabelResolverTests(unittest.TestCase):
    def make_resolver(self, answers, *, cap=20, max_batch_size=100, max_label_queries=12):
        calls = []

        def transport(query):
            calls.append(query)
            if callable(answers):
                return answers(query)
            return {"status": "ok", "rows": answers}

        meter = PhysicalQueryMeter(hard_cap=cap, transport=transport, transport_kind="synthetic_fixture")
        resolver = BoundedLabelResolver(
            label_cache={"m.cached": "Trusted label"}, meter=meter,
            max_batch_size=max_batch_size, max_label_queries=max_label_queries,
        )
        return resolver, meter, calls

    def test_archived_pilot_reproduces_exact_seven_unique_cache_misses(self):
        path = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_022_gated_matched_prefix/RAW_CASE_TRACES.jsonl"
        traces = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        observed = {entity_id for trace in traces for entity_id in trace.get("missing_ids", [])}
        self.assertEqual(KNOWN_MISSES, observed)
        self.assertEqual(3, len(traces))

    def test_original_production_hashes_and_prior_result_artifacts_are_unchanged(self):
        snapshot_path = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_021_matched_prefix_replay/PROTECTED_SOURCE_HASHES.json"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        for item in snapshot["files"]:
            path = ROOT / item["path"]
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(item["sha256_at_iteration_start"], actual, item["path"])
        artifacts_path = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/PROTECTED_PRIOR_ARTIFACTS.json"
        if artifacts_path.exists():
            protected = json.loads(artifacts_path.read_text(encoding="utf-8"))
            self.assertTrue(protected["captured_before_iteration_023_reports"])
            for item in protected["files"]:
                actual = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
                self.assertEqual(item["sha256"], actual, item["path"])

    def test_deduplicates_inputs_and_returns_stable_unicode_labels_with_provenance(self):
        resolver, meter, calls = self.make_resolver([
            binding("m.new", "New Ω 東京", "en"),
            binding("m.new", "unranked name", ""),
        ])
        first = resolver.resolve(["m.new", "m.new", "m.cached"])
        second = resolver.resolve(["m.new", "m.cached"])
        self.assertEqual({"m.new": "New Ω 東京", "m.cached": "Trusted label"}, first)
        self.assertEqual(first, second)
        self.assertEqual(1, len(calls))
        self.assertEqual(1, meter.physical_attempts)
        provenance = resolver.provenance["m.new"]
        self.assertEqual("bounded_read_only_label_select", provenance["source"])
        self.assertEqual(hashlib.sha256(calls[0].encode()).hexdigest(), provenance["query_sha256"])
        self.assertEqual("en", provenance["language"])

    def test_builder_uses_a_read_only_values_batch(self):
        query = build_label_query(["m.alpha", "g.beta"])
        self.assertIn("SELECT DISTINCT ?entity ?targetEntity", query)
        self.assertIn("VALUES ?entity { ns:m.alpha ns:g.beta }", query)
        self.assertIn("langMatches(lang(?targetEntity), 'en')", query)
        self.assertNotRegex(query, r"(?i)\b(?:INSERT|DELETE|LOAD|CLEAR|DROP|CREATE)\b")

    def test_english_label_wins_over_untagged_label_deterministically(self):
        resolver, _, _ = self.make_resolver([
            binding("m.new", "English", "en"), binding("m.new", "Default", ""),
        ])
        self.assertEqual("English", resolver.resolve_one("m.new"))

    def test_conflicting_best_rank_labels_are_explicitly_ambiguous(self):
        resolver, meter, _ = self.make_resolver([
            binding("m.new", "Name A", "en"), binding("m.new", "Name B", "en"),
        ])
        with self.assertRaises(LabelResolutionStop) as caught:
            resolver.resolve_one("m.new")
        self.assertIn("ambiguous", caught.exception.reason)
        self.assertNotIn("m.new", resolver.labels)
        self.assertIn("AMBIGUOUS_LABELS", [event["status"] for event in resolver.outcomes])
        self.assertEqual(1, meter.physical_attempts)

    def test_missing_and_empty_label_results_are_not_guessed_or_cached(self):
        for rows in ([], [binding("m.new", None)]):
            with self.subTest(rows=rows):
                resolver, meter, _ = self.make_resolver(rows)
                with self.assertRaises(LabelResolutionStop):
                    resolver.resolve_one("m.new")
                self.assertNotIn("m.new", resolver.labels)
                self.assertEqual(1, meter.physical_attempts)
                self.assertIn("UNRESOLVED_LABEL_ABSENT", [event["status"] for event in resolver.outcomes])

    def test_malformed_ids_and_literal_values_stop_before_query(self):
        for value in ("", "m.bad id", "not-an-id", "a literal label"):
            with self.subTest(value=value):
                resolver, meter, calls = self.make_resolver([])
                with self.assertRaises(LabelResolutionStop):
                    resolver.resolve([value])
                self.assertEqual(0, meter.physical_attempts)
                self.assertEqual([], calls)

    def test_malformed_or_unrequested_response_ids_fail_closed(self):
        for rows in (
            [{"entity": {"type": "literal", "value": "m.new"}, "targetEntity": {"type": "literal", "value": "Name"}}],
            [binding("m.other", "Unexpected")],
        ):
            with self.subTest(rows=rows):
                resolver, _, _ = self.make_resolver(rows)
                with self.assertRaises(LabelResolutionStop):
                    resolver.resolve_one("m.new")
                self.assertNotIn("m.new", resolver.labels)

    def test_failed_physical_attempt_is_counted_before_transport_and_never_retried(self):
        calls = []

        def failing_transport(_query):
            calls.append("attempt")
            raise TimeoutError("synthetic timeout")

        meter = PhysicalQueryMeter(hard_cap=2, transport=failing_transport, transport_kind="synthetic_fixture")
        with self.assertRaises(QueryExecutionFailed):
            meter.execute("SELECT ?s WHERE {}", category="navigation")
        self.assertEqual(1, meter.physical_attempts)
        self.assertEqual(["attempt"], calls)
        self.assertEqual("TRANSPORT_ERROR", meter.events[0]["status"])

    def test_non_ok_status_is_counted_and_not_misread_as_empty(self):
        meter = PhysicalQueryMeter(
            hard_cap=2, transport=lambda _q: {"status": "timeout", "rows": []},
            transport_kind="synthetic_fixture",
        )
        with self.assertRaises(QueryExecutionFailed):
            meter.execute("SELECT ?s WHERE {}", category="navigation")
        self.assertEqual(1, meter.physical_attempts)
        self.assertEqual("QUERY_ERROR", meter.events[0]["status"])

    def test_query_past_cap_is_blocked_before_transport(self):
        calls = []
        meter = PhysicalQueryMeter(
            hard_cap=1,
            transport=lambda _q: calls.append("called") or {"status": "ok", "rows": []},
            transport_kind="synthetic_fixture",
        )
        meter.execute("SELECT ?s WHERE {}", category="navigation")
        with self.assertRaises(QueryBudgetExceeded):
            meter.execute("SELECT ?p WHERE {}", category="projection")
        self.assertEqual(1, meter.physical_attempts)
        self.assertEqual(1, meter.blocked_requests)
        self.assertEqual(["called"], calls)
        self.assertFalse(meter.events[-1]["transport_invoked"])

    def test_deadline_blocks_query_before_transport(self):
        now = [0.0]
        deadline = DeadlineGuard(5, clock=lambda: now[0])
        calls = []
        meter = PhysicalQueryMeter(
            hard_cap=2, transport=lambda q: calls.append(q) or {"status": "ok", "rows": []},
            transport_kind="synthetic_fixture", deadline=deadline,
        )
        now[0] = 5.0
        with self.assertRaises(Exception):
            meter.execute("SELECT ?s WHERE {}", category="navigation")
        self.assertEqual(0, meter.physical_attempts)
        self.assertEqual([], calls)
        self.assertEqual("BLOCKED_BY_DEADLINE", meter.events[-1]["event"])

    def test_invalid_query_is_rejected_before_meter_or_transport(self):
        calls = []
        meter = PhysicalQueryMeter(
            hard_cap=3, transport=lambda q: calls.append(q) or {"status": "ok", "rows": []},
            transport_kind="synthetic_fixture",
        )
        with self.assertRaises(Exception):
            meter.execute("DELETE WHERE {?s ?p ?o}", category="navigation")
        self.assertEqual(0, meter.physical_attempts)
        self.assertEqual([], calls)

    def test_label_batch_bound_and_query_allowance_fail_closed(self):
        def answer_for(query):
            values = query.split("VALUES ?entity {", 1)[1].split("}", 1)[0].split()
            ids = [value[3:] for value in values]
            return {"status": "ok", "rows": [binding(value, f"Label {value}") for value in ids]}

        resolver, meter, calls = self.make_resolver(
            answer_for, cap=5, max_batch_size=100, max_label_queries=2,
        )
        ids = [f"m.id{i}" for i in range(201)]
        with self.assertRaises(LabelResolutionStop) as caught:
            resolver.resolve(ids)
        self.assertIn("allowance exhausted", caught.exception.reason)
        self.assertEqual(2, len(calls))
        self.assertEqual(2, meter.physical_attempts)
        self.assertEqual(200, len(resolver.labels) - 1)  # Includes one seeded cache mapping.
        self.assertIn("LABEL_QUERY_BUDGET_EXHAUSTED", [event["status"] for event in resolver.outcomes])

    def test_entity_search_uses_metered_dynamic_labels_in_normal_candidate_path(self):
        from chain_of_relations.kg_backend.freebase_backend import FreebaseBackend
        from chain_of_relations.schema import Entity, Relation
        from chain_of_relations.tools.entity_search import EntitySearchInput, entity_search

        def transport(query):
            if "VALUES ?entity" in query:
                return {"status": "ok", "rows": [binding("m.unseen", "Resolved Ω", "en")]}
            return {"status": "ok", "rows": [{
                "targetEntity": {"type": "uri", "value": "http://rdf.freebase.com/ns/m.unseen"}
            }]}

        meter = PhysicalQueryMeter(hard_cap=10, transport=transport, transport_kind="synthetic_fixture")
        resolver = BoundedLabelResolver(label_cache={}, meter=meter)
        backend = FreebaseBackend()
        with install_metered_freebase_backend(backend, meter=meter, resolver=resolver):
            output = entity_search(
                EntitySearchInput(topic_entity=Entity("m.root", "Root"), relation_chain=[Relation("r.rel")]),
                backend=backend,
            )
        self.assertEqual(["m.unseen"], output.entity_ids)
        self.assertEqual(["Resolved Ω"], output.entity_names)
        self.assertEqual(1, meter.category_counts["navigation"])
        self.assertEqual(1, meter.category_counts["label_resolution"])

    def test_all_backend_entrypoints_route_through_meter_and_restore(self):
        from chain_of_relations.kg_backend import freebase_backend as backend_module
        from chain_of_relations.kg_backend.freebase import db_func
        from chain_of_relations.kg_backend.freebase_backend import FreebaseBackend

        calls = []

        def transport(query):
            calls.append(query)
            return {"status": "ok", "rows": []}

        meter = PhysicalQueryMeter(hard_cap=12, transport=transport, transport_kind="synthetic_fixture")
        resolver = BoundedLabelResolver(label_cache={"m.cached": "Cache"}, meter=meter)
        backend = FreebaseBackend()
        original_db_execute = db_func.execurte_sparql_with_meta
        original_db_rows = db_func.execurte_sparql
        original_db_wrapper = db_func.SPARQLWrapper
        original_alias = backend_module.freebase_id2entity_names
        original_backend_execute = backend_module.execurte_sparql
        original_backend_meta = backend_module.execurte_sparql_with_meta
        original_backend_label = backend_module.id2entity_name_or_type

        with install_metered_freebase_backend(backend, meter=meter, resolver=resolver):
            backend.execute_sparql("SELECT ?s WHERE {}")
            backend.execute_sparql_with_meta("SELECT ?p WHERE {}")
            db_func.execurte_sparql("SELECT ?o WHERE {}")
            db_func.execurte_sparql_with_meta("SELECT ?q WHERE {}")
            backend_module.execurte_sparql("SELECT ?x WHERE {}")
            backend_module.execurte_sparql_with_meta("SELECT ?y WHERE {}")
            self.assertEqual("Cache", backend_module.id2entity_name_or_type("m.cached"))
            self.assertEqual({"m.cached": "Cache"}, backend.id2entity_names(["m.cached"]))
            self.assertEqual("Cache", backend.id2entity_name_or_type("m.cached"))
            with self.assertRaises(UnmeteredQueryPath):
                db_func.SPARQLWrapper("http://should-not-be-used")

        self.assertEqual(6, meter.physical_attempts)
        self.assertEqual(6, len(calls))
        self.assertNotIn("execute_sparql", backend.__dict__)
        self.assertIs(original_db_execute, db_func.execurte_sparql_with_meta)
        self.assertIs(original_db_rows, db_func.execurte_sparql)
        self.assertIs(original_db_wrapper, db_func.SPARQLWrapper)
        self.assertIs(original_alias, backend_module.freebase_id2entity_names)
        self.assertIs(original_backend_execute, backend_module.execurte_sparql)
        self.assertIs(original_backend_meta, backend_module.execurte_sparql_with_meta)
        self.assertIs(original_backend_label, backend_module.id2entity_name_or_type)

    def test_unresolved_label_in_normal_path_never_falls_back_to_single_id_lookup(self):
        from chain_of_relations.kg_backend.freebase_backend import FreebaseBackend
        from chain_of_relations.schema import Entity, Relation
        from chain_of_relations.tools.entity_search import EntitySearchInput, entity_search

        calls = []
        def transport(query):
            calls.append(query)
            if "VALUES ?entity" in query:
                return {"status": "ok", "rows": []}
            return {"status": "ok", "rows": [{
                "targetEntity": {"type": "uri", "value": "http://rdf.freebase.com/ns/m.unseen"}
            }]}

        meter = PhysicalQueryMeter(hard_cap=10, transport=transport, transport_kind="synthetic_fixture")
        resolver = BoundedLabelResolver(label_cache={}, meter=meter)
        backend = FreebaseBackend()
        with install_metered_freebase_backend(backend, meter=meter, resolver=resolver):
            with self.assertRaises(LabelResolutionStop):
                entity_search(
                    EntitySearchInput(topic_entity=Entity("m.root", "Root"), relation_chain=[Relation("r.rel")]),
                    backend=backend,
                )
        self.assertEqual(2, meter.physical_attempts)
        self.assertEqual(2, len(calls))
        self.assertEqual(["navigation", "label_resolution"], [event["category"] for event in meter.events])

    def test_backend_wrappers_restore_after_baseexception(self):
        from chain_of_relations.kg_backend.freebase import db_func
        from chain_of_relations.kg_backend.freebase_backend import FreebaseBackend

        meter = PhysicalQueryMeter(hard_cap=2, transport=lambda _q: {"status": "ok", "rows": []},
                                   transport_kind="synthetic_fixture")
        resolver = BoundedLabelResolver(label_cache={}, meter=meter)
        backend = FreebaseBackend()
        original = db_func.SPARQLWrapper
        with self.assertRaises(KeyboardInterrupt):
            with install_metered_freebase_backend(backend, meter=meter, resolver=resolver):
                raise KeyboardInterrupt()
        self.assertIs(original, db_func.SPARQLWrapper)
        self.assertNotIn("execute_sparql", backend.__dict__)


class Iteration23BudgetAndPairTests(unittest.TestCase):
    def caps_and_authorization(self):
        caps = ModelCaps(
            manifest_sha256="a" * 64, request_cap=2,
            input_tokens_per_request=100, completion_tokens_per_request=20,
            estimated_spend_cap_usd=0.00003,
        )
        auth = {
            "authorization_status": "APPROVED", "approved_at": "synthetic-fixture",
            "manifest_sha256": "a" * 64, "model": MODEL_NAME,
            "api_request_cap": 2, "input_tokens_per_request_cap": 100,
            "completion_tokens_per_request_cap": 20,
            "estimated_api_spend_cap_usd": 0.00003,
            "input_usd_per_million": INPUT_USD_PER_MILLION,
            "output_usd_per_million": OUTPUT_USD_PER_MILLION,
            "retries": 0,
            "approval_verbatim": "I approve manifest " + "a" * 64 + "; no retries",
        }
        return caps, auth

    def test_model_meter_uses_manifest_specific_caps_and_one_call(self):
        caps, auth = self.caps_and_authorization()
        meter = NoRetryModelMeter(caps)
        calls = []
        result = meter.call_once(
            authorization=auth, input_token_reservation=60, completion_token_cap=10,
            prompt="fixture prompt",
            transport=lambda: calls.append(1) or ("selected", {"input_tokens": 40, "output_tokens": 5}),
        )
        self.assertEqual("selected", result[0])
        self.assertEqual([1], calls)
        self.assertEqual(1, meter.attempts)
        self.assertEqual(0, meter.events[0]["retries"])
        self.assertAlmostEqual(estimate_cost(40, 5), meter.estimated_cost_usd)

    def test_model_request_over_token_or_cost_reservation_is_not_sent(self):
        caps, auth = self.caps_and_authorization()
        meter = NoRetryModelMeter(caps)
        calls = []
        with self.assertRaises(ModelBudgetStop):
            meter.call_once(authorization=auth, input_token_reservation=101, completion_token_cap=10,
                            prompt="x", transport=lambda: calls.append(1))
        with self.assertRaises(ModelBudgetStop):
            meter.call_once(authorization=auth, input_token_reservation=100, completion_token_cap=20,
                            prompt="x", transport=lambda: calls.append(1))
        self.assertEqual([], calls)
        self.assertEqual(0, meter.attempts)

    def test_model_requires_matching_new_manifest_and_stops_at_call_cap(self):
        caps, auth = self.caps_and_authorization()
        meter = NoRetryModelMeter(caps)
        calls = []
        with self.assertRaises(ModelBudgetStop):
            meter.call_once(authorization={**auth, "manifest_sha256": "b" * 64},
                            input_token_reservation=10, completion_token_cap=5,
                            prompt="x", transport=lambda: calls.append(1))
        self.assertEqual([], calls)
        for _ in range(caps.request_cap):
            meter.call_once(
                authorization=auth, input_token_reservation=1, completion_token_cap=1,
                prompt="x", transport=lambda: ("ok", {"input_tokens": 1, "output_tokens": 1}),
            )
        with self.assertRaises(ModelBudgetStop):
            meter.call_once(authorization=auth, input_token_reservation=1, completion_token_cap=1,
                            prompt="x", transport=lambda: calls.append(1))
        self.assertEqual(caps.request_cap, meter.attempts)
        self.assertEqual([], calls)

    def test_deadline_blocks_model_call_before_transport(self):
        now = [0.0]
        caps, auth = self.caps_and_authorization()
        meter = NoRetryModelMeter(caps, deadline=DeadlineGuard(5, clock=lambda: now[0]))
        calls = []
        now[0] = 5.0
        with self.assertRaises(ModelBudgetStop):
            meter.call_once(authorization=auth, input_token_reservation=10, completion_token_cap=5,
                            prompt="x", transport=lambda: calls.append(1))
        self.assertEqual(0, meter.attempts)
        self.assertEqual([], calls)

    def test_empty_error_missing_usage_and_transport_error_consume_once_without_retry(self):
        cases = [
            ("", {"input_tokens": 1, "output_tokens": 1}),
            ("x", {"error": "provider error"}),
            ("x", None),
        ]
        for content, usage in cases:
            with self.subTest(content=content, usage=usage):
                caps, auth = self.caps_and_authorization()
                meter = NoRetryModelMeter(caps)
                calls = []
                with self.assertRaises(ModelBudgetStop):
                    meter.call_once(authorization=auth, input_token_reservation=10, completion_token_cap=5,
                                    prompt="x", transport=lambda: calls.append(1) or (content, usage))
                self.assertEqual(1, meter.attempts)
                self.assertEqual([1], calls)
        caps, auth = self.caps_and_authorization()
        meter = NoRetryModelMeter(caps)
        calls = []
        def fail():
            calls.append(1)
            raise TimeoutError("fixture")
        with self.assertRaises(ModelBudgetStop):
            meter.call_once(authorization=auth, input_token_reservation=10, completion_token_cap=5,
                            prompt="x", transport=fail)
        self.assertEqual(1, meter.attempts)
        self.assertEqual([1], calls)
        self.assertEqual(1, len(meter.events))

    def test_synthetic_unknown_label_then_exact_stop_one_hop_pair(self):
        resolver, meter, _ = Iteration23LabelResolverTests().make_resolver(
            [binding("m.new", "A real fixture label", "en")], cap=10,
        )
        labels = resolver.resolve(["m.new"])
        prompt_candidate = f"TargetEntity({labels['m.new']})"
        self.assertEqual("TargetEntity(A real fixture label)", prompt_candidate)
        outcome = dry_run_case(
            snapshot=prefix_fixture(), projection=projection_fixture(),
            projection_variable_order=["?a", "?b", "?endpoint"],
            llm_response="r.next", parsed_selection=[{"id": "r.next", "direction": "OUTGOING"}],
            kg_query="SELECT ?e3 ?e4 WHERE {}",
            kg_rows=[{"?e3": "m.c", "?e4": "m.new"}],
        )
        self.assertEqual("SYNTHETIC_PAIR_VALID", outcome["status"])
        self.assertEqual(outcome["stop_parent_hash"], outcome["one_hop_parent_hash"])
        self.assertEqual(1, outcome["new_edge_count"])
        self.assertEqual(["m.new"], outcome["new_entity_ids"])
        self.assertEqual("synthetic_fixture", meter.transport_kind,
                         "fixture label transport is synthetic, not live SPARQL")

    def test_synthetic_nonforward_stays_ineligible_without_fork(self):
        result = dry_run_case(
            snapshot=prefix_fixture(boundary_action="Finish"), projection=projection_fixture(),
            projection_variable_order=["?a", "?b", "?endpoint"], llm_response="r.next",
            parsed_selection=[{"id": "r.next", "direction": "OUTGOING"}],
            kg_query="SELECT ?e3 ?e4 WHERE {}", kg_rows=[],
        )
        self.assertEqual("NON_FORWARD_BOUNDARY", result["reason"])
        self.assertNotEqual("SYNTHETIC_PAIR_VALID", result["status"])


if __name__ == "__main__":
    unittest.main()
