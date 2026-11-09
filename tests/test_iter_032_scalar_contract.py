from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path

from chain_of_relations.methods.iter_031_count_readiness import (
    CountPlanContract,
    PathStep,
    RuntimeCallsite,
    TargetTypeEvidence,
    callsite_from_cor_shape,
    run_count_adapter,
)
from chain_of_relations.methods.iter_031_count_readiness.evidence import EvidenceRef
from chain_of_relations.methods.iter_032_scalar_contract import (
    CONTRACT_VERSION,
    XSD_INTEGER,
    RequestEvidence,
    score_typed_answer,
    serialize_callsite_result,
    serialize_executor_count,
)


ROOT = Path(__file__).resolve().parents[1]


def runtime_result(
    *, question="How many railway termini does this line have?", value="2",
    router_type="number", rationale="The candidates can be aggregated to answer the count question.",
    action="Stop", decision="stop", candidate_labels=None, literal_candidates=None,
    result_id="", finish_reason="stop", response=None,
):
    prompt = "Q: frozen question and path"
    body = response or (
        '{"rationale":' + __import__("json").dumps(rationale)
        + ',"decision":' + __import__("json").dumps(decision)
        + ',"answer":[' + __import__("json").dumps(value) + ']}'
    )
    prompt_history = [
        {
            "type": "adaptive_router_v1_1_3",
            "parsed_result": {
                "requested_final_answer_type": router_type,
                "requested_final_semantic_property": "railway terminus count",
            },
        },
        {
            "type": "reasoning",
            "trace_id": 5,
            "parsed_result": "Stop" if decision == "stop" else decision,
            "prompt": prompt,
            "response": body,
            "output_tokens": 7,
        },
    ]
    return {
        "action": action,
        "question": question,
        "results": [{"id": result_id, "name": value}],
        "prompt_history": prompt_history,
        "sparql_history": [{
            "search_type": "entity_search",
            "trace_id": 4,
            "candidate_entity_ids": ["m.candidate"],
            "candidate_entity_names": candidate_labels if candidate_labels is not None else ["Platform A"],
            "literal_candidates": literal_candidates or [],
        }],
    }


def request_for(result, *, finish="stop", content=True, output_tokens=7):
    prompt = result["prompt_history"][-1]["prompt"]
    return RequestEvidence(
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        finish_reason=finish,
        content_nonempty=content,
        http_status=200,
        output_tokens=output_tokens,
        request_sha256="1" * 64,
        raw_response_sha256="2" * 64,
    )


def provenance_ref(name="synthetic-runtime-plan.json"):
    raw = name.encode()
    return EvidenceRef(name, hashlib.sha256(raw).hexdigest(), "count_plan", "fixture-case", "runtime")


def synthetic_complete_plan(**overrides):
    ref = provenance_ref()
    step = PathStep("rail.railway.terminuses", "out", "?v0", "?v1")
    values = {
        "target_variable": "?v1",
        "query_target_variable": "?v1",
        "type_filter_variable": "?v1",
        "expected_steps": (step,),
        "executed_steps": (step,),
        "query_status": "ok",
        "finalized": True,
        "target_type_evidence": (TargetTypeEvidence("rail.railway_terminus", "?v1", "explicit_target_type_constraint", ref, "synthetic fixture constraint"),),
        "provenance": (ref,),
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


def count_response(value="0"):
    return {"status": "ok", "results": {"bindings": [{"count": {
        "type": "typed-literal",
        "datatype": XSD_INTEGER,
        "value": value,
    }}]}}


def fake_executor(query):
    if "COUNT(DISTINCT ?iter30_node_1)" not in query:
        raise AssertionError("fake executor received a query outside the count compiler")
    return count_response("0")


class ScalarAnswerContractTests(unittest.TestCase):
    def test_disabled_mode_is_exact_object_passthrough(self):
        result = runtime_result()
        outcome = serialize_callsite_result(result, enabled=False, request_evidence=request_for(result))
        self.assertIs(outcome.output, result)
        self.assertTrue(outcome.baseline_preserved)
        self.assertIsNone(outcome.typed_answer)

    def test_terminal_scalar_is_typed_only_with_complete_runtime_provenance(self):
        result = runtime_result(value="0")
        outcome = serialize_callsite_result(result, enabled=True, request_evidence=request_for(result))
        self.assertEqual(outcome.status, "AGGREGATE")
        self.assertEqual(outcome.typed_answer["answer_kind"], "aggregate")
        self.assertEqual(outcome.typed_answer["datatype"], XSD_INTEGER)
        self.assertEqual(outcome.typed_answer["lexical_value"], "0")
        self.assertEqual(outcome.typed_answer["provenance"]["raw_answer_field_path"], "prompt_history[terminal].response.answer[0]")
        self.assertFalse(outcome.baseline_preserved)

    def test_router_alias_count_phrase_is_accepted_but_entity_answer_is_not_coerced(self):
        result = runtime_result(router_type="count of electric locomotive classes")
        outcome = serialize_callsite_result(result, enabled=True, request_evidence=request_for(result))
        self.assertEqual(outcome.status, "AGGREGATE")
        entity = runtime_result(result_id="m.numeric_entity")
        entity_outcome = serialize_callsite_result(entity, enabled=True, request_evidence=request_for(entity))
        self.assertEqual(entity_outcome.status, "ENTITY")
        self.assertEqual(entity_outcome.typed_answer["entities"][0]["id"], "m.numeric_entity")

    def test_rejects_measurement_year_ordinal_minmax_and_noncount_number_property(self):
        cases = [
            ("How many miles long is Route 66?", "number", "The measured length is 66 miles."),
            ("What year did the 1998 tournament begin?", "number", "The tournament began in 1998."),
            ("Who was the number one ranked player?", "number", "The number one player is the answer."),
            ("Which trial has the least number of expected participants?", "number", "This is the least number, 4."),
            ("What number was assigned to the album?", "number", "The assigned number is 2."),
            ("How much revenue did the company earn?", "number", "The revenue amount is 2."),
        ]
        for question, answer_type, rationale in cases:
            with self.subTest(question=question):
                result = runtime_result(question=question, value="2", router_type=answer_type, rationale=rationale)
                outcome = serialize_callsite_result(result, enabled=True, request_evidence=request_for(result))
                self.assertIn(outcome.status, {"UNKNOWN", "INVALID"})
                self.assertIs(outcome.output, result)
                self.assertEqual(outcome.typed_answer["answer_kind"], "unknown")

    def test_rejects_candidate_label_and_literal_ambiguity(self):
        for kwargs in (
            {"candidate_labels": ["2"]},
            {"literal_candidates": ["2"]},
        ):
            result = runtime_result(**kwargs)
            outcome = serialize_callsite_result(result, enabled=True, request_evidence=request_for(result))
            self.assertEqual(outcome.status, "UNKNOWN")
            self.assertEqual(outcome.reason, "scalar_ambiguous_with_candidate_or_literal")
            self.assertIs(outcome.output, result)

    def test_rejects_malformed_negative_leading_zero_multiple_and_mixed_answer(self):
        responses = [
            '{"decision":"stop","rationale":"count","answer":["02"]}',
            '{"decision":"stop","rationale":"count","answer":["-2"]}',
            '{"decision":"stop","rationale":"count","answer":["2","3"]}',
            '{"decision":"stop","rationale":"count","answer":["Station A: 2"]}',
            '{"decision":"stop","rationale":"count","answer":[null]}',
        ]
        for response in responses:
            result = runtime_result(value="2", response=response)
            outcome = serialize_callsite_result(result, enabled=True, request_evidence=request_for(result))
            self.assertEqual(outcome.status, "INVALID")
            self.assertIs(outcome.output, result)

    def test_rejects_empty_length_failed_or_unlinked_request(self):
        result = runtime_result()
        for evidence in (
            request_for(result, finish="length"),
            request_for(result, content=False),
            RequestEvidence("0" * 64, "stop", True, 200, 7),
            request_for(result, output_tokens=8),
        ):
            outcome = serialize_callsite_result(result, enabled=True, request_evidence=evidence)
            self.assertEqual(outcome.status, "UNKNOWN")
            self.assertIs(outcome.output, result)

    def test_entity_envelope_preserves_entity_ids(self):
        result = runtime_result(result_id="m.railway")
        envelope = serialize_callsite_result(result, enabled=True, request_evidence=request_for(result)).typed_answer
        self.assertEqual(envelope["answer_kind"], "entity")
        self.assertEqual(envelope["entities"], [{"id": "m.railway", "name": "2"}])

    def test_runtime_modules_have_no_gold_file_access(self):
        path = ROOT / "chain_of_relations/methods/iter_032_scalar_contract/answer_contract.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.assertNotIn("data.interim", ast.unparse(node))
                self.assertNotIn("gold", ast.unparse(node).casefold())
        for forbidden in ("Path(", "open(", "COUNT_OPERATOR_AUDIT.csv", "PAIRED_EVAL_CASES_GOLD_ONLY.csv"):
            self.assertNotIn(forbidden, source)


class IndependentScorerTests(unittest.TestCase):
    def test_aggregate_zero_and_entity_answers_score_by_distinct_kinds(self):
        scalar = {
            "contract_version": CONTRACT_VERSION,
            "answer_kind": "aggregate",
            "status": "accepted",
            "datatype": XSD_INTEGER,
            "lexical_value": "0",
        }
        self.assertEqual(score_typed_answer(scalar, ["0"])["hits_at_1"], 1)
        entity = {
            "contract_version": CONTRACT_VERSION,
            "answer_kind": "entity",
            "entities": [{"id": "m.a", "name": "A"}],
        }
        self.assertEqual(score_typed_answer(entity, ["m.a"])["f1"], 1.0)

    def test_invalid_typed_values_are_never_scored_as_answers(self):
        for value in ("", "00", "-1", "+1", "1.0", "١"):
            answer = {
                "contract_version": CONTRACT_VERSION,
                "answer_kind": "aggregate",
                "status": "accepted",
                "datatype": XSD_INTEGER,
                "lexical_value": value,
            }
            self.assertEqual(score_typed_answer(answer, [value])["hits_at_1"], 0)


class ConsumerIntegrationFixtureTests(unittest.TestCase):
    def test_synthetic_complete_plan_runs_fake_count_then_typed_score(self):
        question = "How many railway termini does this line have?"
        baseline = {"action": "Stop", "question": question, "results": [{"id": "m.old", "name": "Legacy"}]}
        callsite = callsite_from_cor_shape(
            question,
            [{"id": "m.topic", "name": "Example Line", "type": "entity"}],
            baseline,
            count_plan=synthetic_complete_plan(),
            callsite_provenance=(provenance_ref("synthetic-cor-callsite.json"),),
        )
        adapter = run_count_adapter(callsite, fake_executor, timeout_seconds=5)
        self.assertEqual(adapter.status, "COUNT_EXECUTED")
        self.assertEqual(adapter.physical_attempts, 1)
        serialized = serialize_executor_count(adapter, baseline)
        self.assertEqual(serialized.status, "AGGREGATE")
        self.assertEqual(serialized.typed_answer["lexical_value"], "0")
        self.assertEqual(score_typed_answer(serialized.typed_answer, ["0"])["hits_at_1"], 1)

    def test_plausible_but_incomplete_plan_preserves_exact_baseline(self):
        question = "How many railway termini does this line have?"
        baseline = {"action": "Stop", "question": question, "results": [{"id": "", "name": "2"}]}
        incomplete = synthetic_complete_plan(required_constraints_complete=False)
        callsite = callsite_from_cor_shape(
            question,
            [{"id": "m.topic", "name": "Example Line", "type": "entity"}],
            baseline,
            count_plan=incomplete,
            callsite_provenance=(provenance_ref("synthetic-cor-callsite.json"),),
        )
        adapter = run_count_adapter(callsite, lambda _: self.fail("incomplete plan must not execute"), timeout_seconds=5)
        self.assertNotEqual(adapter.status, "COUNT_EXECUTED")
        self.assertIs(adapter.output, baseline)
        serialized = serialize_executor_count(adapter, baseline)
        self.assertTrue(serialized.baseline_preserved)
        self.assertIs(serialized.output, baseline)


if __name__ == "__main__":
    unittest.main()
