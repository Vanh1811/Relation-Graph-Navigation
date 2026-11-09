from __future__ import annotations

import hashlib
import multiprocessing
import ast
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from chain_of_relations.methods.iter_031_count_readiness.evidence import EvidenceRef
from chain_of_relations.methods.iter_031_count_readiness.path_completeness import (
    PathEvidence,
    PathStep,
    check_path_completeness,
)
from chain_of_relations.methods.iter_031_count_readiness.runtime_contract import (
    CONTRACT_VERSION,
    CountPlanContract,
    ReadOnlyQueryBudget,
    RuntimeCallsite,
    callsite_from_cor_shape,
    execute_bounded_select,
    is_single_read_only_count_select,
    run_count_adapter,
)
from chain_of_relations.methods.iter_031_count_readiness.type_resolution import (
    TargetTypeEvidence,
    resolve_target_type,
)


def ref(name: str = "runtime-plan.json", *, source_class: str = "runtime", raw: bytes = b"fixture") -> EvidenceRef:
    return EvidenceRef(name, hashlib.sha256(raw).hexdigest(), "count_plan.path", "case-1", source_class)


def type_evidence(type_id="rail.railway_terminus", variable="?v1", *, kind="explicit_target_type_constraint", provenance=None):
    return TargetTypeEvidence(type_id, variable, kind, provenance if provenance is not None else ref("runtime-plan.json"), "fixture field")


def steps_for(relations=("rail.railway.terminuses",), directions=None):
    directions = directions or tuple("out" for _ in relations)
    return tuple(PathStep(rel, direction, f"?v{i}", f"?v{i + 1}") for i, (rel, direction) in enumerate(zip(relations, directions)))


def make_plan(**overrides):
    steps = steps_for()
    values = {
        "target_variable": "?v1",
        "query_target_variable": "?v1",
        "type_filter_variable": "?v1",
        "expected_steps": steps,
        "executed_steps": steps,
        "query_status": "ok",
        "finalized": True,
        "target_type_evidence": (type_evidence(),),
        "provenance": (ref("runtime-plan.json"),),
        "plan_shape": "linear",
        "operator": "COUNT_DISTINCT_ENTITY",
        "semantics": "distinct_target_entities",
        "required_external_binding_variables": ("?v0",),
        "active_binding_tuples": ({"?v0": "m.topic"},),
        "required_constraints_complete": True,
        "all_pairwise_node_inequalities_represented": True,
    }
    values.update(overrides)
    return CountPlanContract(**values)


def make_callsite(*, plan=None, question="How many railway termini does the line have?", baseline=None, prompt_history=None):
    baseline = baseline if baseline is not None else {
        "action": "Stop",
        "question": question,
        "results": [{"id": "", "name": "2"}],
        "reasoning_chains": "original CoR output",
        "prompt_history": prompt_history or [],
    }
    return callsite_from_cor_shape(
        question,
        [{"id": "m.topic", "name": "Example Line", "type": "entity"}],
        baseline,
        count_plan=plan,
        callsite_provenance=(ref("cor-result.json"),),
    )


def count_payload(value="2", *, status="ok", binding_type="typed-literal", bindings_marker=True):
    bindings = [{"count": {"type": binding_type, "datatype": "http://www.w3.org/2001/XMLSchema#integer", "value": value}}] if bindings_marker else []
    return {"status": status, "results": {"bindings": bindings}}


def slow_executor(query):
    time.sleep(5)
    return count_payload("1")


class TypeResolverTests(unittest.TestCase):
    def test_single_explicit_target_constraint_with_hash_resolves(self):
        result = resolve_target_type("?v1", [type_evidence()])
        self.assertEqual((result.status, result.resolved_type_id), ("resolved", "rail.railway_terminus"))
        self.assertEqual(result.evidence_grade, "DIRECT_RUNTIME_TARGET_CONSTRAINT")

    def test_natural_language_answer_type_is_not_freebase_target_type(self):
        result = resolve_target_type("?v1", [], answer_type_hint="number")
        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.resolved_type_id)

    def test_relation_range_is_only_a_coarse_candidate(self):
        result = resolve_target_type("?v1", [type_evidence(kind="relation_range")])
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.evidence_grade, "COARSE_RELATION_RANGE")

    def test_absent_schema_range_and_entity_alias_labels_do_not_supply_target_class(self):
        absent_schema = resolve_target_type("?v1", [], answer_type_hint="railway terminus")
        alias = resolve_target_type("?v1", [type_evidence(kind="localized_alias")])
        self.assertEqual(absent_schema.status, "ambiguous")
        self.assertEqual(alias.status, "not_observed")

    def test_stale_alias_or_label_does_not_resolve_type(self):
        result = resolve_target_type("?v1", [type_evidence(kind="cached_label")])
        self.assertEqual(result.status, "not_observed")

    def test_intermediate_type_does_not_resolve_final_target(self):
        result = resolve_target_type("?v2", [type_evidence(variable="?v1")])
        self.assertEqual(result.status, "not_observed")
        self.assertEqual(result.evidence_grade, "INTERMEDIATE_TYPE_ONLY")

    def test_multiple_nested_or_competing_types_remain_ambiguous(self):
        result = resolve_target_type("?v1", [type_evidence(), type_evidence("common.topic", provenance=ref("nested.json"))])
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(len(result.type_ids), 2)

    def test_conflict_with_relation_range_fails_closed(self):
        result = resolve_target_type("?v1", [type_evidence(), type_evidence("common.topic", kind="relation_range", provenance=ref("schema.json", source_class="schema"))])
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.evidence_grade, "CONFLICTING_SCHEMA_EVIDENCE")

    def test_missing_invalid_or_gold_provenance_is_not_accepted(self):
        missing = resolve_target_type("?v1", [TargetTypeEvidence("rail.railway_terminus", "?v1", "explicit_target_type_constraint", None)])
        bad_ref = EvidenceRef("gold-sidecar.json", "0" * 63, "target_type")
        bad_hash = resolve_target_type("?v1", [type_evidence(provenance=bad_ref)])
        gold_ref = EvidenceRef("scoring.json", hashlib.sha256(b"gold").hexdigest(), "gold.type", source_class="gold")
        gold = resolve_target_type("?v1", [type_evidence(provenance=gold_ref)])
        self.assertEqual(missing.status, "unsupported")
        self.assertEqual(bad_hash.status, "unsupported")
        self.assertEqual(gold.status, "unsupported")

    def test_malformed_variable_and_type_ids_are_unsupported(self):
        self.assertEqual(resolve_target_type("target", []).status, "unsupported")
        self.assertEqual(resolve_target_type("?v1", [type_evidence("not-a-freebase-id")]).status, "unsupported")

    def test_scalar_literal_target_is_not_silently_mapped_to_entity_type(self):
        result = resolve_target_type("?v1", [type_evidence(kind="literal_range")])
        self.assertEqual(result.status, "not_observed")


class PathCompletenessTests(unittest.TestCase):
    def resolved_type(self):
        return resolve_target_type("?v1", [type_evidence()])

    def complete_evidence(self, **overrides):
        steps = steps_for()
        values = {
            "target_variable": "?v1",
            "query_target_variable": "?v1",
            "expected_steps": steps,
            "executed_steps": steps,
            "query_status": "ok",
            "plan_finalized": True,
            "required_external_binding_variables": ("?v0",),
            "active_binding_tuples": ({"?v0": "m.topic"},),
            "provenance": (ref(),),
            "required_constraints_complete": True,
            "all_pairwise_node_inequalities_represented": True,
            "type_filter_variable": "?v1",
        }
        values.update(overrides)
        return PathEvidence(**values)

    def test_exact_forward_and_inverse_paths(self):
        for path, target in ((steps_for(), "?v1"), (steps_for(("rail.r",), ("in",)), "?v1")):
            evidence = self.complete_evidence(target_variable=target, query_target_variable=target, expected_steps=path, executed_steps=path, type_filter_variable=target)
            result = check_path_completeness(evidence, self.resolved_type())
            self.assertTrue(result.ready, result.reason)

    def test_two_and_three_hop_join_identity(self):
        for relations in (("a.r1", "b.r2"), ("a.r1", "b.r2", "c.r3")):
            steps = steps_for(relations, tuple("out" for _ in relations))
            target = f"?v{len(steps)}"
            evidence = self.complete_evidence(target_variable=target, query_target_variable=target, type_filter_variable=target, expected_steps=steps, executed_steps=steps)
            resolved = resolve_target_type(target, [type_evidence(variable=target)])
            result = check_path_completeness(evidence, resolved)
            self.assertTrue(result.ready, result.reason)

    def test_unknown_if_expected_plan_or_provenance_missing(self):
        self.assertEqual(check_path_completeness(self.complete_evidence(expected_steps=None), self.resolved_type()).status, "UNKNOWN_MISSING_PROVENANCE")
        self.assertEqual(check_path_completeness(self.complete_evidence(provenance=()), self.resolved_type()).status, "UNKNOWN_MISSING_PROVENANCE")
        self.assertEqual(check_path_completeness(self.complete_evidence(plan_finalized=None), self.resolved_type()).status, "UNKNOWN_MISSING_PROVENANCE")

    def test_wrong_relation_and_direction(self):
        wrong_rel = steps_for(("other.relation",))
        wrong_dir = steps_for(("rail.railway.terminuses",), ("in",))
        self.assertEqual(check_path_completeness(self.complete_evidence(executed_steps=wrong_rel), self.resolved_type()).status, "WRONG_RELATION_OR_DIRECTION")
        self.assertEqual(check_path_completeness(self.complete_evidence(executed_steps=wrong_dir), self.resolved_type()).status, "WRONG_RELATION_OR_DIRECTION")

    def test_missing_hop_extra_hop_and_join_mismatch(self):
        three = steps_for(("a.r", "b.r", "c.r"))
        two = steps_for(("a.r", "b.r"))
        disconnected = (PathStep("a.r", "out", "?v0", "?v1"), PathStep("b.r", "out", "?v9", "?v2"))
        self.assertEqual(check_path_completeness(self.complete_evidence(expected_steps=three, executed_steps=two, target_variable="?v3", query_target_variable="?v3", type_filter_variable="?v3"), self.resolved_type()).status, "MISSING_HOP")
        self.assertEqual(check_path_completeness(self.complete_evidence(expected_steps=two, executed_steps=three, target_variable="?v3", query_target_variable="?v3", type_filter_variable="?v3"), self.resolved_type()).status, "WRONG_RELATION_OR_DIRECTION")
        self.assertEqual(check_path_completeness(self.complete_evidence(expected_steps=disconnected, executed_steps=disconnected, target_variable="?v2", query_target_variable="?v2", type_filter_variable="?v2"), self.resolved_type()).status, "MISSING_SHARED_VARIABLE_JOIN")

    def test_dangling_target_and_wrong_target_type_variable(self):
        result = check_path_completeness(self.complete_evidence(query_target_variable="?v0"), self.resolved_type())
        self.assertEqual(result.status, "DANGLING_OR_AMBIGUOUS_TARGET")
        result = check_path_completeness(self.complete_evidence(type_filter_variable="?v0"), self.resolved_type())
        self.assertEqual(result.status, "TARGET_TYPE_FILTER_MISALIGNED")

    def test_unsupported_branch_operator_and_failed_query(self):
        self.assertEqual(check_path_completeness(self.complete_evidence(plan_shape="branch"), self.resolved_type()).status, "UNSUPPORTED_OPERATOR_OR_ORDERING")
        self.assertEqual(check_path_completeness(self.complete_evidence(operator="COUNT_EVENTS"), self.resolved_type()).status, "UNSUPPORTED_OPERATOR_OR_ORDERING")
        self.assertEqual(check_path_completeness(self.complete_evidence(query_status="error"), self.resolved_type()).status, "QUERY_FAILED")

    def test_missing_and_unstable_active_bindings(self):
        self.assertEqual(check_path_completeness(self.complete_evidence(active_binding_tuples=None), self.resolved_type()).status, "MISSING_ACTIVE_BINDING_TUPLES")
        self.assertEqual(check_path_completeness(self.complete_evidence(active_binding_tuples=({"?v0": "not-an-entity"},)), self.resolved_type()).status, "MISSING_ACTIVE_BINDING_TUPLES")

    def test_incomplete_constraints_or_node_inequality_fail_closed(self):
        self.assertEqual(check_path_completeness(self.complete_evidence(required_constraints_complete=False), self.resolved_type()).status, "MISSING_REQUIRED_CONSTRAINT")
        self.assertEqual(check_path_completeness(self.complete_evidence(all_pairwise_node_inequalities_represented=False), self.resolved_type()).status, "NODE_DISTINCTNESS_UNPROVEN")
        self.assertEqual(check_path_completeness(self.complete_evidence(required_constraints_complete=None), self.resolved_type()).status, "UNKNOWN_MISSING_PROVENANCE")


class RuntimeAdapterTests(unittest.TestCase):
    def test_unsupported_and_non_count_paths_preserve_original_object_identity(self):
        baseline = {"answer": "2", "results": [{"name": "2"}], "opaque": object()}
        no_plan = make_callsite(plan=None, baseline=baseline)
        outcome = run_count_adapter(no_plan)
        self.assertIs(outcome.output, baseline)
        self.assertEqual(outcome.status, "UNKNOWN_MISSING_PROVENANCE")
        entity = make_callsite(plan=make_plan(), question="Which railway serves this line?", baseline=baseline)
        entity_outcome = run_count_adapter(entity, lambda _: self.fail("query must not run"))
        self.assertIs(entity_outcome.output, baseline)
        self.assertEqual(entity_outcome.physical_attempts, 0)

    def test_actual_cor_shape_adapter_extracts_router_text_and_ignores_gold_fields(self):
        result = {
            "action": "Stop", "question": "How many?", "results": [],
            "prompt_history": [{"type": "adaptive_router_v1_1_3", "parsed_result": {"requested_final_answer_type": "number"}}],
            "gold_answer": "999", "gold_target_type": "invented.type",
        }
        callsite = callsite_from_cor_shape("How many?", [{"id": "m.topic"}], result)
        self.assertEqual(callsite.requested_answer_type, "number")
        self.assertIs(callsite.baseline_output, result)
        self.assertIsNone(callsite.count_plan)
        self.assertEqual(run_count_adapter(callsite).status, "UNKNOWN_MISSING_PROVENANCE")
        clean = dict(result)
        clean.pop("gold_answer")
        clean.pop("gold_target_type")
        clean_callsite = callsite_from_cor_shape("How many?", [{"id": "m.topic"}], clean)
        self.assertEqual(run_count_adapter(clean_callsite).status, run_count_adapter(callsite).status)

    def test_runtime_package_has_no_dataset_or_scoring_file_access(self):
        package = Path(__file__).resolve().parents[1] / "chain_of_relations/methods/iter_031_count_readiness"
        forbidden_literals = (
            "data/interim", "gold_sparql", "gold_answer_ids", "gold_target_type",
            "COUNT_OPERATOR_AUDIT.csv", "PAIRED_EVAL_CASES_GOLD_ONLY.csv",
        )
        for source in package.glob("*.py"):
            text = source.read_text(encoding="utf-8")
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imported = ast.unparse(node)
                    self.assertNotIn("relation_graph_kgqa", imported)
                    self.assertNotIn("data.interim", imported)
            for token in forbidden_literals:
                self.assertNotIn(token, text, source.name)

    def test_prompt_over_limit_prevents_query_and_preserves_baseline(self):
        plan = make_plan(upstream_prompt_tokens=8001, upstream_prompt_limit=7000)
        baseline = {"raw": "unchanged"}
        outcome = run_count_adapter(make_callsite(plan=plan, baseline=baseline), lambda _: self.fail("must not query"))
        self.assertIs(outcome.output, baseline)
        self.assertEqual(outcome.status, "PROMPT_BUDGET_EXCEEDED")

    def test_complete_mocked_callsite_compiles_executes_and_normalizes_scalar(self):
        plan = make_plan()
        callsite = make_callsite(plan=plan)
        calls = multiprocessing.get_context("fork").Value("i", 0)

        def executor(query):
            with calls.get_lock():
                calls.value += 1
            return count_payload("2")

        outcome = run_count_adapter(callsite, executor, timeout_seconds=5)
        self.assertEqual(outcome.status, "COUNT_EXECUTED")
        self.assertEqual(outcome.output["answer_kind"], "aggregate")
        self.assertEqual(outcome.output["scalar_answer"], "2")
        self.assertEqual(outcome.output["results"][0]["datatype"], "http://www.w3.org/2001/XMLSchema#integer")
        self.assertIn("COUNT(DISTINCT ?iter30_node_1)", outcome.query)
        self.assertIn("?iter30_node_1 ns:type.object.type ns:rail.railway_terminus", outcome.query)
        self.assertEqual(outcome.path_status, "SUPPORTED_EXACT_PATH")
        self.assertEqual(outcome.physical_attempts, 1)
        self.assertEqual(calls.value, 1)
        self.assertEqual(len(outcome.query_sha256), 64)
        self.assertEqual(outcome.result_sha256, outcome.output["count_provenance"]["result"]["source_sha256"])
        self.assertEqual(outcome.output["contract_version"], CONTRACT_VERSION)

    def test_duplicate_rows_count_distinct_endpoint_ids(self):
        # Semantic fixture: the SPARQL DISTINCT aggregate must count endpoint
        # IDs, not duplicate RDF rows. This does not claim Virtuoso execution.
        edges = [("m.topic", "m.a"), ("m.topic", "m.a"), ("m.topic", "m.b")]
        distinct_count = len({target for source, target in edges if source == "m.topic"})
        self.assertEqual(distinct_count, 2)
        def executor(query):
            return count_payload(str(distinct_count))
        outcome = run_count_adapter(make_callsite(plan=make_plan()), executor, timeout_seconds=5)
        self.assertEqual(outcome.output["scalar_answer"], "2")
        self.assertIn("COUNT(DISTINCT", outcome.query)

    def test_empty_solution_is_zero_but_missing_or_null_binding_is_unknown(self):
        zero = run_count_adapter(make_callsite(plan=make_plan()), lambda _: count_payload("0"), timeout_seconds=5)
        self.assertEqual(zero.output["scalar_answer"], "0")
        for payload in (count_payload("", bindings_marker=False), {"status": "ok", "results": {"bindings": None}}, None):
            outcome = run_count_adapter(make_callsite(plan=make_plan()), lambda _, p=payload: p, timeout_seconds=5)
            self.assertEqual(outcome.status, "UNKNOWN_RESULT")
            self.assertEqual(outcome.output["action"], "Stop")
            self.assertTrue(outcome.baseline_preserved)

    def test_non_integer_datatype_is_unknown_not_a_scalar_answer(self):
        payload = count_payload("2")
        payload["results"]["bindings"][0]["count"]["datatype"] = "http://www.w3.org/2001/XMLSchema#decimal"
        outcome = run_count_adapter(make_callsite(plan=make_plan()), lambda _: payload, timeout_seconds=5)
        self.assertEqual(outcome.status, "UNKNOWN_RESULT")
        self.assertTrue(outcome.baseline_preserved)

    def test_type_conflict_wrong_type_and_missing_type_all_preserve_baseline(self):
        plans = [
            make_plan(target_type_evidence=(type_evidence("rail.railway"), type_evidence("common.topic", provenance=ref("alt.json")))),
            make_plan(target_type_evidence=()),
            make_plan(target_type_evidence=(type_evidence(variable="?v0"),)),
            make_plan(target_type_evidence=(type_evidence("bad id"),)),
        ]
        for plan in plans:
            outcome = run_count_adapter(make_callsite(plan=plan), lambda _: self.fail("must not query"))
            self.assertTrue(outcome.baseline_preserved)
            self.assertNotEqual(outcome.status, "COUNT_EXECUTED")

    def test_mutating_any_single_path_type_binding_or_semantics_field_fails_closed(self):
        changes = [
            {"executed_steps": steps_for(("wrong.relation",))},
            {"executed_steps": steps_for(("rail.railway.terminuses",), ("in",))},
            {"expected_steps": None},
            {"query_target_variable": "?v0"},
            {"type_filter_variable": "?v0"},
            {"active_binding_tuples": None},
            {"active_binding_tuples": ({"?v0": "m.other"},)},
            {"required_constraints_complete": False},
            {"all_pairwise_node_inequalities_represented": False},
            {"plan_shape": "many_to_many"},
            {"semantics": "count_relation_events"},
        ]
        for change in changes:
            with self.subTest(change=change):
                plan = make_plan(**change)
                outcome = run_count_adapter(make_callsite(plan=plan), lambda _: self.fail("mutated plan must not query"))
                self.assertTrue(outcome.baseline_preserved)
                self.assertNotEqual(outcome.status, "COUNT_EXECUTED")

    def test_external_join_bindings_not_supported_by_single_root_compiler(self):
        plan = make_plan(required_external_binding_variables=("?v0", "?constraint"), active_binding_tuples=({"?v0": "m.topic", "?constraint": "m.other"},))
        outcome = run_count_adapter(make_callsite(plan=plan), lambda _: self.fail("must not query"))
        self.assertEqual(outcome.status, "UNSUPPORTED_OPERATOR_OR_ORDERING")

    def test_malformed_topic_or_relation_ids_never_reach_executor(self):
        for callsite in (
            replace(make_callsite(plan=make_plan()), topic_entity_id="m.bad id"),
            make_callsite(plan=make_plan(executed_steps=steps_for(("bad relation",)), expected_steps=steps_for(("bad relation",)))),
        ):
            outcome = run_count_adapter(callsite, lambda _: self.fail("invalid IDs must not query"))
            self.assertTrue(outcome.baseline_preserved)

    def test_query_shape_rejects_multiple_select_update_service_and_non_count(self):
        good = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT (COUNT(DISTINCT ?x) AS ?count) WHERE { ?x ns:type.object.type ns:some.type . }"
        self.assertTrue(is_single_read_only_count_select(good))
        for bad in (
            good + " SELECT * WHERE {?s ?p ?o}",
            good + "; INSERT DATA { <a> <b> <c> }",
            good.replace("WHERE", "WHERE { SERVICE <http://evil> { SELECT * WHERE { ?s ?p ?o } } #"),
            "SELECT ?x WHERE {?s ?p ?o}",
        ):
            self.assertFalse(is_single_read_only_count_select(bad))

    def test_parser_rejection_and_backend_error_are_one_shot_fallbacks(self):
        with patch("chain_of_relations.methods.iter_031_count_readiness.runtime_contract.build_count_query", return_value="DELETE WHERE {?s ?p ?o}"):
            outcome = run_count_adapter(make_callsite(plan=make_plan()), lambda _: self.fail("invalid compiled query must not run"))
            self.assertEqual(outcome.status, "QUERY_REJECTED")
            self.assertEqual(outcome.physical_attempts, 0)
        budget = ReadOnlyQueryBudget()
        outcome = run_count_adapter(make_callsite(plan=make_plan()), lambda _: (_ for _ in ()).throw(ConnectionError()), budget=budget, timeout_seconds=5)
        self.assertEqual(outcome.status, "QUERY_FAILED")
        self.assertEqual(outcome.physical_attempts, 1)
        self.assertEqual(budget.attempts_used, 1)

    def test_budget_is_exactly_one_and_cannot_be_reused(self):
        outcome = run_count_adapter(make_callsite(plan=make_plan()), lambda _: count_payload("1"), budget=ReadOnlyQueryBudget(), timeout_seconds=5)
        self.assertEqual(outcome.physical_attempts, 1)
        rejected = run_count_adapter(make_callsite(plan=make_plan()), lambda _: self.fail("2-slot budget must be rejected"), budget=ReadOnlyQueryBudget(max_attempts=2), timeout_seconds=5)
        self.assertEqual(rejected.status, "QUERY_FAILED")
        budget = ReadOnlyQueryBudget()
        budget.attempts_used = 1
        exhausted = run_count_adapter(make_callsite(plan=make_plan()), lambda _: self.fail("exhausted budget must not query"), budget=budget, timeout_seconds=5)
        self.assertEqual(exhausted.status, "QUERY_FAILED")

    def test_timeout_is_bounded_kills_child_and_never_retries(self):
        budget = ReadOnlyQueryBudget()
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            execute_bounded_select(
                "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT (COUNT(DISTINCT ?x) AS ?count) WHERE { ?x ns:type.object.type ns:some.type . }",
                slow_executor, budget, timeout_seconds=0.15,
            )
        self.assertLess(time.monotonic() - start, 2.0)
        self.assertEqual(budget.attempts_used, 1)
        with self.assertRaises(ValueError):
            execute_bounded_select("SELECT ?s WHERE {?s ?p ?o}", slow_executor, ReadOnlyQueryBudget(), timeout_seconds=121)

    def test_hashes_and_sorted_serialization_are_reproducible(self):
        first = run_count_adapter(make_callsite(plan=make_plan()), lambda _: count_payload("3"), timeout_seconds=5)
        second = run_count_adapter(make_callsite(plan=make_plan()), lambda _: count_payload("3"), timeout_seconds=5)
        self.assertEqual(first.query_sha256, second.query_sha256)
        self.assertEqual(first.result_sha256, second.result_sha256)


if __name__ == "__main__":
    unittest.main()
