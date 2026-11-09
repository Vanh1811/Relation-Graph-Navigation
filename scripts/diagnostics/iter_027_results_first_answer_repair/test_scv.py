from __future__ import annotations

import hashlib
import json
import unittest

from scripts.diagnostics.iter_026_answer_filter.selector import candidate_snapshot_sha256
from scripts.diagnostics.iter_027_results_first_answer_repair.scv import (
    SCHEMA_VERSION,
    build_scv_payload,
    eligible_for_scv_empty_only,
    empty_baseline_non_regression_invariant,
    validate_scv_response,
)


def candidates():
    qsha = hashlib.sha256(b"frozen read-only SELECT").hexdigest()
    return [
        {"id": "m.one", "label": "Entity One", "source": {
            "entity_search_trace_id": 7, "entity_search_query_sha256": qsha, "position": 0,
        }},
        {"id": "m.two", "label": "Entity Two", "source": {
            "entity_search_trace_id": 7, "entity_search_query_sha256": qsha, "position": 1,
        }},
    ]


class SCVEmptyOnlyTests(unittest.TestCase):
    def eligibility(self, **overrides):
        values = {
            "baseline_status": "BASELINE_COMPLETED",
            "baseline_entity_only": True,
            "baseline_prediction_ids": [],
            "router_execution_mode": "cor",
            "final_filter_event_count": 1,
            "candidates": candidates(),
        }
        values.update(overrides)
        return eligible_for_scv_empty_only(**values)

    def test_empty_completed_baseline_with_graph_provenance_is_eligible(self):
        self.assertEqual(self.eligibility(), (True, "SCV_EMPTY_BASELINE_WITH_FROZEN_PROVENANCE"))

    def test_nonempty_baseline_always_falls_back(self):
        self.assertEqual(self.eligibility(baseline_prediction_ids=["m.one"]),
                         (False, "BASELINE_NONEMPTY_SAFE_FALLBACK"))

    def test_upstream_failure_is_not_treated_as_empty_answer(self):
        self.assertFalse(self.eligibility(baseline_status="BASELINE_INPUT_CAP_EXCEEDED_NO_OUTPUT")[0])

    def test_branch_route_is_not_eligible(self):
        self.assertEqual(self.eligibility(router_execution_mode="branch"),
                         (False, "NON_COR_ROUTER_MODE"))

    def test_multiple_filter_boundaries_are_not_eligible(self):
        self.assertEqual(self.eligibility(final_filter_event_count=2),
                         (False, "FILTER_BOUNDARY_COUNT_NOT_ONE"))

    def test_candidate_provenance_is_required(self):
        pool = candidates()
        del pool[0]["source"]["entity_search_query_sha256"]
        self.assertFalse(self.eligibility(candidates=pool)[0])

    def test_payload_contains_provenance_but_no_gold_or_scoring_fields(self):
        payload = build_scv_payload(
            question="Which entity matches the constraint?",
            frozen_filter_context="Topic ?relation ?candidate; preserve the constraint.",
            router_execution_mode="cor", filter_trace_id=8,
            selected_relation_evidence=["domain.relation.property"], candidates=candidates(),
        )
        rendered = json.dumps(payload).casefold()
        self.assertIn("entity_search_query_sha256", rendered)
        self.assertIn("domain.relation.property", rendered)
        self.assertNotIn("gold_answer", rendered)
        self.assertNotIn("scoring_only", rendered)

    def test_valid_output_returns_only_candidate_ids(self):
        pool = candidates()
        raw = json.dumps({"schema_version": SCHEMA_VERSION, "decision": "select", "candidate_ids": ["m.two"]})
        result = validate_scv_response(raw, candidates=pool,
                                       expected_candidate_snapshot_sha256=candidate_snapshot_sha256(pool))
        self.assertTrue(result["valid"])
        self.assertEqual(result["selected_ids"], ["m.two"])
        self.assertEqual(result["prediction_source"], "SCV_EMPTY_BASELINE_REPAIR")

    def test_non_stop_finish_reason_fails_closed_even_if_json_is_valid(self):
        pool = candidates()
        raw = json.dumps({"schema_version": SCHEMA_VERSION, "decision": "select", "candidate_ids": ["m.one"]})
        result = validate_scv_response(
            raw, candidates=pool,
            expected_candidate_snapshot_sha256=candidate_snapshot_sha256(pool),
            finish_reason="length",
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "NON_TERMINAL_OR_TRUNCATED_RESPONSE")
        self.assertEqual(result["selected_ids"], [])
        self.assertEqual(result["prediction_source"], "UNCHANGED_EMPTY_BASELINE")

    def test_out_of_pool_id_fails_closed(self):
        pool = candidates()
        raw = json.dumps({"schema_version": SCHEMA_VERSION, "decision": "select", "candidate_ids": ["m.outside"]})
        result = validate_scv_response(raw, candidates=pool,
                                       expected_candidate_snapshot_sha256=candidate_snapshot_sha256(pool))
        self.assertFalse(result["valid"])
        self.assertEqual(result["selected_ids"], [])
        self.assertEqual(result["prediction_source"], "UNCHANGED_EMPTY_BASELINE")

    def test_abstention_preserves_empty_baseline(self):
        pool = candidates()
        raw = json.dumps({"schema_version": SCHEMA_VERSION, "decision": "abstain", "candidate_ids": []})
        result = validate_scv_response(raw, candidates=pool,
                                       expected_candidate_snapshot_sha256=candidate_snapshot_sha256(pool))
        self.assertTrue(result["valid"])
        self.assertEqual(result["selected_ids"], [])
        self.assertEqual(result["prediction_source"], "UNCHANGED_EMPTY_BASELINE")

    def test_malformed_empty_and_duplicate_ids_fail_closed(self):
        pool = candidates()
        for selected in [[], ["m.one", "m.one"]]:
            raw = json.dumps({"schema_version": SCHEMA_VERSION, "decision": "select", "candidate_ids": selected})
            result = validate_scv_response(raw, candidates=pool,
                                           expected_candidate_snapshot_sha256=candidate_snapshot_sha256(pool))
            self.assertFalse(result["valid"])
            self.assertEqual(result["prediction_source"], "UNCHANGED_EMPTY_BASELINE")

    def test_empty_baseline_metric_invariant_exhaustive_small_pools(self):
        pool = ["a", "b", "c"]
        for mask in range(1 << len(pool)):
            gold = [value for i, value in enumerate(pool) if mask & (1 << i)]
            for pred_mask in range(1 << len(pool)):
                proposed = [value for i, value in enumerate(pool) if pred_mask & (1 << i)]
                self.assertTrue(empty_baseline_non_regression_invariant(gold, proposed))


if __name__ == "__main__":
    unittest.main()
