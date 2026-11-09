from __future__ import annotations

import ast
import json
import unittest
from dataclasses import fields
from pathlib import Path

from chain_of_relations.methods.iter_030_operator_recovery.relation_aware_count import (
    RelationStep,
    RuntimeCountContext,
    build_count_query,
    classify_count_intent,
    recover_count,
)
from chain_of_relations.methods.iter_030_operator_recovery.runtime_features import (
    GoldFreeRuntimeFeatures,
    RuntimeRelationStep,
)


def make_context(**overrides):
    values = {
        "question": "How many video games did Randy Condon design?",
        "topic_entity_id": "m.02hptvh",
        "relation_chain": (RelationStep("cvg.cvg_designer.games_designed", True),),
        "grounded_query_status": "ok",
        "target_type_id": "cvg.computer_videogame",
        "baseline_answer": "baseline-preserved",
    }
    values.update(overrides)
    return RuntimeCountContext(**values)


def sparql_result(value="3", *, status="ok", binding_type="literal"):
    return {"status": status, "results": {"bindings": [{"count": {"type": binding_type, "value": value}}]}}


class RelationAwareCountTests(unittest.TestCase):
    def test_forward_query_has_type_filter_distinct_count_and_node_inequality(self):
        query = build_count_query(make_context())
        self.assertIn("COUNT(DISTINCT ?iter30_node_1)", query)
        self.assertIn("VALUES ?iter30_node_0 { ns:m.02hptvh }", query)
        self.assertIn("?iter30_node_0 ns:cvg.cvg_designer.games_designed ?iter30_node_1 .", query)
        self.assertIn("?iter30_node_1 ns:type.object.type ns:cvg.computer_videogame .", query)
        self.assertIn("FILTER (?iter30_node_0 != ?iter30_node_1)", query)

    def test_inverse_direction(self):
        query = build_count_query(make_context(relation_chain=(RelationStep("people.person.place_of_birth", False),)))
        self.assertIn("?iter30_node_1 ns:people.person.place_of_birth ?iter30_node_0 .", query)

    def test_two_and_three_hop_joins_and_all_distinct_node_pairs(self):
        paths = [
            (RelationStep("a.r1", True), RelationStep("b.r2", False)),
            (RelationStep("a.r1", True), RelationStep("b.r2", True), RelationStep("c.r3", False)),
        ]
        for path in paths:
            with self.subTest(path=path):
                query = build_count_query(make_context(relation_chain=path, target_type_id="answer.type"))
                node_count = len(path) + 1
                self.assertTrue(all(f"?iter30_node_{i}" in query for i in range(node_count)))
                self.assertEqual(query.count("FILTER ("), node_count * (node_count - 1) // 2)
                self.assertIn("COUNT(DISTINCT", query)

    def test_loop_back_to_topic_is_excluded_by_distinct_path_constraint(self):
        query = build_count_query(make_context(relation_chain=(RelationStep("a.r", True), RelationStep("a.r", False))))
        self.assertIn("FILTER (?iter30_node_0 != ?iter30_node_2)", query)

    def test_successful_and_zero_count_each_use_one_executor_call(self):
        for value in ("4", "0"):
            calls = []
            recovered = recover_count(make_context(), lambda query, timeout: (calls.append((query, timeout)) or sparql_result(value)))
            self.assertEqual((recovered.status, recovered.answer), ("recovered", value))
            self.assertTrue(recovered.used_extension)
            self.assertEqual(recovered.executor_calls, 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], 120.0)

    def test_count_intent_rejects_non_targets_and_ambiguous_measurements(self):
        cases = [
            ("How many games did the designer make?", True),
            ("What is the total amount of games created?", True),
            ("Which trial has the least number of participants?", False),
            ("How much does the car cost?", False),
            ("How many miles long is the road?", False),
            ("What is the total amount of money held by the company?", False),
            ("What was the number of the jersey?", False),
            ("What number is assigned to the album?", False),
            ("Who designed the game?", False),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                self.assertEqual(classify_count_intent(question)[0], expected)

    def test_entity_question_preserves_baseline_without_transport(self):
        calls = []
        result = recover_count(make_context(question="Who designed the game?"), lambda *args: calls.append(args))
        self.assertEqual((result.status, result.answer, result.executor_calls), ("preserved", "baseline-preserved", 0))
        self.assertFalse(calls)

    def test_missing_type_or_executor_fails_closed(self):
        missing_type = recover_count(make_context(target_type_id=""), lambda *_: self.fail("must not query"))
        missing_executor = recover_count(make_context())
        self.assertEqual(missing_type.reason, "target_type_id_unavailable")
        self.assertEqual(missing_executor.reason, "no_executor_injected")
        self.assertEqual(missing_type.answer, "baseline-preserved")
        self.assertEqual(missing_executor.executor_calls, 0)

    def test_failed_grounding_and_bad_timeout_do_not_query(self):
        calls = []
        failed = recover_count(make_context(grounded_query_status="error"), lambda *args: calls.append(args))
        timeout = recover_count(make_context(), lambda *args: calls.append(args), timeout_seconds=120.1)
        self.assertEqual(failed.reason, "grounded_path_not_successful")
        self.assertEqual(timeout.reason, "timeout_out_of_bounds")
        self.assertFalse(calls)

    def test_timeout_transport_exception_and_query_errors_do_not_retry(self):
        for exception in (TimeoutError(), OSError("connection reset")):
            calls = []

            def fail_once(*args):
                calls.append(args)
                raise exception

            result = recover_count(make_context(), fail_once)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result.executor_calls, 1)
            self.assertEqual(result.answer, "baseline-preserved")
        query_error = recover_count(make_context(), lambda *_: sparql_result(status="error"))
        self.assertEqual(query_error.status, "preserved")

    def test_malformed_results_preserve_baseline(self):
        bad_payloads = [
            {"status": "ok", "results": {"bindings": []}},
            {"status": "ok", "results": {"bindings": [{"count": {"type": "uri", "value": "3"}}]}},
            sparql_result("3.0"),
            {"status": "ok", "results": {"bindings": [
                {"count": {"type": "literal", "value": "1"}},
                {"count": {"type": "literal", "value": "2"}},
            ]}},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                result = recover_count(make_context(), lambda *_args, payload=payload: payload)
                self.assertEqual(result.status, "preserved")
                self.assertEqual(result.answer, "baseline-preserved")

    def test_invalid_path_ids_depth_and_empty_path_are_rejected(self):
        invalid_contexts = [
            make_context(topic_entity_id="m.bad id"),
            make_context(relation_chain=()),
            make_context(target_type_id=""),
            make_context(relation_chain=(RelationStep("bad relation"),)),
            make_context(relation_chain=tuple(RelationStep(f"r.p{i}") for i in range(7))),
        ]
        for context in invalid_contexts:
            with self.subTest(context=context):
                with self.assertRaises(ValueError):
                    build_count_query(context)

    def test_gold_free_runtime_feature_allow_list(self):
        expected = {
            "question", "topic_entity_id", "relation_chain", "requested_answer_type",
            "target_type_id", "grounded_query_status", "baseline_answer",
        }
        self.assertEqual({field.name for field in fields(GoldFreeRuntimeFeatures)}, expected)
        payload = GoldFreeRuntimeFeatures(
            question="How many games?", topic_entity_id="m.topic",
            relation_chain=(RuntimeRelationStep("games.designed", True),),
        ).to_runtime_dict()
        self.assertEqual(set(payload), expected)
        self.assertNotIn("gold_answer", payload)
        self.assertNotIn("question_id", payload)

    def test_runtime_code_has_no_gold_dataset_import_or_scoring_file_access(self):
        package = Path(__file__).resolve().parents[1] / "chain_of_relations/methods/iter_030_operator_recovery"
        forbidden = ("data/interim", "gold_sparql", "logical_form", "answer_ids", "GOLD_ONLY.csv")
        for source in package.glob("*.py"):
            text = source.read_text(encoding="utf-8")
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self.assertNotIn("relation_graph_kgqa", ast.unparse(node))
            self.assertFalse(any(token in text for token in forbidden), source.name)
        schema_path = Path(__file__).resolve().parents[1] / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_030_results_first/GOLD_FREE_RUNTIME_FEATURES.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertTrue(schema["prohibited_fields"])
        self.assertIn("no verified target_type_id resolver", schema["runtime_status"]["blocking_capability"])


if __name__ == "__main__":
    unittest.main()
