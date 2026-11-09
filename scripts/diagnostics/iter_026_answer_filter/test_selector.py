from __future__ import annotations

import copy
import json
import unittest

from selector import (
    SCHEMA_VERSION,
    build_inference_payload,
    candidate_snapshot_sha256,
    paired_outcome,
    score_id_sets,
    validate_selector_response,
)


class SelectorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            {"id": "m.alpha", "label": "Alpha", "source": {"event": 4, "position": 0}},
            {"id": "m.beta", "label": "Bêta 東京", "source": {"event": 4, "position": 1}},
            {"id": "m.gamma", "label": "Gamma", "source": {"event": 4, "position": 2}},
        ]
        self.snapshot_hash = candidate_snapshot_sha256(self.candidates)
        self.baseline = ["m.gamma"]

    def response(self, ids, *, decision="select", schema=SCHEMA_VERSION, **extra):
        value = {"schema_version": schema, "decision": decision, "candidate_ids": ids}
        value.update(extra)
        return json.dumps(value, ensure_ascii=False)

    def validate(self, raw, candidates=None, expected_hash=None, baseline=None):
        return validate_selector_response(
            raw,
            candidates=self.candidates if candidates is None else candidates,
            expected_candidate_snapshot_sha256=self.snapshot_hash if expected_hash is None else expected_hash,
            baseline_prediction=self.baseline if baseline is None else baseline,
        )

    def test_valid_single_candidate_id(self):
        result = self.validate(self.response(["m.alpha"]))
        self.assertTrue(result.valid)
        self.assertEqual(result.selected_ids, ("m.alpha",))
        self.assertEqual(result.prediction_source, "INTERVENTION_SELECTOR")

    def test_valid_multi_answer_output_preserves_order(self):
        result = self.validate(self.response(["m.beta", "m.alpha"]))
        self.assertTrue(result.valid)
        self.assertEqual(result.selected_ids, ("m.beta", "m.alpha"))

    def test_empty_response_falls_back_unchanged(self):
        baseline = {"legacy_answer": ["Gamma"]}
        result = self.validate("", baseline=baseline)
        self.assertEqual(result.status, "EMPTY_RESPONSE")
        self.assertEqual(result.final_prediction, baseline)
        self.assertEqual(result.prediction_source, "UNCHANGED_BASELINE_FALLBACK")
        self.assertIsNot(result.final_prediction, baseline)

    def test_duplicate_selected_ids_fall_back(self):
        result = self.validate(self.response(["m.alpha", "m.alpha"]))
        self.assertEqual(result.status, "DUPLICATE_IDS")
        self.assertEqual(result.final_prediction, self.baseline)

    def test_out_of_candidate_id_falls_back(self):
        result = self.validate(self.response(["m.not_in_pool"]))
        self.assertEqual(result.status, "OUT_OF_CANDIDATE_ID")
        self.assertEqual(result.final_prediction, self.baseline)

    def test_absent_candidate_label_rejects_intervention(self):
        candidates = copy.deepcopy(self.candidates)
        candidates[0]["label"] = " "
        result = self.validate(self.response(["m.alpha"]), candidates=candidates)
        self.assertEqual(result.status, "INVALID_FROZEN_CANDIDATE_SNAPSHOT")
        self.assertEqual(result.final_prediction, self.baseline)

    def test_duplicate_alias_label_is_not_selected_ambiguously(self):
        candidates = copy.deepcopy(self.candidates)
        candidates[2]["label"] = "alpha"
        result = self.validate(
            self.response(["m.alpha"]), candidates=candidates,
            expected_hash=candidate_snapshot_sha256(candidates),
        )
        self.assertEqual(result.status, "AMBIGUOUS_DISPLAY_LABEL")
        self.assertEqual(result.final_prediction, self.baseline)

    def test_mislabeled_or_reordered_frozen_pair_is_detected_by_snapshot_hash(self):
        swapped = copy.deepcopy(self.candidates)
        swapped[0]["label"], swapped[1]["label"] = swapped[1]["label"], swapped[0]["label"]
        result = self.validate(self.response(["m.alpha"]), candidates=swapped)
        self.assertEqual(result.status, "CANDIDATE_SNAPSHOT_MISMATCH")
        self.assertEqual(result.final_prediction, self.baseline)

    def test_wrong_cardinality_is_scored_as_precision_recall_tradeoff(self):
        scores = score_id_sets(["m.alpha"], ["m.alpha", "m.beta"])
        self.assertEqual(scores["precision"], 1.0)
        self.assertEqual(scores["recall"], 0.5)
        self.assertAlmostEqual(scores["f1"], 2 / 3)
        self.assertEqual(scores["predicted_count"], 1)
        self.assertEqual(scores["gold_count"], 2)

    def test_unicode_labels_are_preserved_while_ids_remain_output(self):
        payload = build_inference_payload(
            question="Which entity?",
            frozen_filter_context="Reasoning: Bêta 東京 was returned as a candidate.",
            frozen_filter_system_context="Use only the question and candidate reasoning context.",
            candidates=self.candidates,
        )
        self.assertIn("Bêta 東京", json.dumps(payload, ensure_ascii=False))
        self.assertEqual(
            payload["constraints"]["frozen_tgar_filter_system_context"],
            "Use only the question and candidate reasoning context.",
        )
        self.assertNotIn("gold", json.dumps(payload, ensure_ascii=False).casefold())
        result = self.validate(self.response(["m.beta"]))
        self.assertEqual(result.selected_ids, ("m.beta",))

    def test_code_fence_is_unexpected_format_and_falls_back(self):
        result = self.validate("```json\n" + self.response(["m.alpha"]) + "\n```")
        self.assertEqual(result.status, "INVALID_JSON")
        self.assertEqual(result.final_prediction, self.baseline)

    def test_extra_rationale_field_is_rejected(self):
        result = self.validate(self.response(["m.alpha"], rationale="invented text"))
        self.assertEqual(result.status, "INVALID_SCHEMA")

    def test_partial_json_falls_back_without_retry(self):
        result = self.validate('{"schema_version":"iter26.answer_selector.v1","decision":"select"')
        self.assertEqual(result.status, "INVALID_JSON")
        self.assertEqual(result.prediction_source, "UNCHANGED_BASELINE_FALLBACK")

    def test_wrong_schema_version_falls_back(self):
        result = self.validate(self.response(["m.alpha"], schema="iter25.answer_selector.v0"))
        self.assertEqual(result.status, "INVALID_SCHEMA_VERSION")

    def test_select_with_empty_list_is_invalid_and_falls_back(self):
        result = self.validate(self.response([]))
        self.assertEqual(result.status, "INVALID_EMPTY_SELECTION")
        self.assertEqual(result.final_prediction, self.baseline)

    def test_explicit_abstention_preserves_baseline_and_is_not_a_rescue(self):
        baseline = ["m.gamma"]
        result = self.validate(self.response([], decision="abstain"), baseline=baseline)
        self.assertEqual(result.status, "VALID_ABSTAIN_BASELINE_FALLBACK")
        self.assertEqual(result.final_prediction, baseline)
        self.assertEqual(result.prediction_source, "UNCHANGED_BASELINE_FALLBACK")
        self.assertEqual(score_id_sets(result.final_prediction, ["m.alpha"])["hits_at_1"], 0)

    def test_duplicate_candidate_ids_in_frozen_snapshot_are_rejected(self):
        candidates = copy.deepcopy(self.candidates)
        candidates.append(copy.deepcopy(candidates[0]))
        result = self.validate(self.response(["m.alpha"]), candidates=candidates)
        self.assertEqual(result.status, "INVALID_FROZEN_CANDIDATE_SNAPSHOT")

    def test_empty_candidate_snapshot_is_ineligible(self):
        result = self.validate(self.response(["m.alpha"]), candidates=[])
        self.assertEqual(result.status, "INVALID_FROZEN_CANDIDATE_SNAPSHOT")

    def test_malformed_freebase_candidate_id_is_ineligible(self):
        candidates = copy.deepcopy(self.candidates)
        candidates[0]["id"] = "not-an-entity-id"
        result = self.validate(
            self.response(["not-an-entity-id"]), candidates=candidates,
            expected_hash=candidate_snapshot_sha256(self.candidates),
        )
        self.assertEqual(result.status, "INVALID_FROZEN_CANDIDATE_SNAPSHOT")

    def test_non_string_decision_is_rejected_without_parser_exception(self):
        result = self.validate(self.response(["m.alpha"], decision=[]))
        self.assertEqual(result.status, "INVALID_DECISION")

    def test_inference_payload_preserves_exact_filter_context_and_has_no_gold(self):
        context = "Q: Which?\nReasoning Chain (SPARQL-style):\nTopicEntity(m.x) relation e1.\nAnswer:"
        payload = build_inference_payload(
            question="Which?", frozen_filter_context=context,
            frozen_filter_system_context="Filter-system prompt fixture.", candidates=self.candidates,
        )
        self.assertEqual(payload["constraints"]["frozen_tgar_filter_context"], context)
        self.assertEqual(list(payload["candidate_entities"][0]), ["id", "label"])
        serialized = json.dumps(payload, ensure_ascii=False).casefold()
        for forbidden in ("gold_answer", "gold_ids", "scoring_only", "cohort_role"):
            self.assertNotIn(forbidden, serialized)

    def test_baseline_correct_fallback_is_not_counted_as_selector_success(self):
        gold = ["m.gamma"]
        result = self.validate(None, baseline=gold)
        paired = paired_outcome(gold, result.final_prediction, gold)
        self.assertEqual(result.prediction_source, "UNCHANGED_BASELINE_FALLBACK")
        self.assertEqual(paired["paired_hits"], "TIE")
        self.assertEqual(paired["hit_delta"], 0)

    def test_paired_metrics_include_f1_delta_and_controls(self):
        paired = paired_outcome(["m.gamma"], ["m.alpha", "m.beta"], ["m.alpha", "m.beta"])
        self.assertEqual(paired["paired_hits"], "WIN")
        self.assertEqual(paired["hit_delta"], 1)
        self.assertEqual(paired["intervention"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
