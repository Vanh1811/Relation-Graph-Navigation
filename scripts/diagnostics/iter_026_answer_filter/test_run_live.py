from __future__ import annotations

import unittest

from run_live import _selector_eligibility


class SelectorForkEligibilityTests(unittest.TestCase):
    def test_uses_nearest_entity_search_before_single_filter_boundary(self):
        history = [{
            "type": "filter_answer", "trace_id": 7,
            "prompt": "Q: Which?\nReasoning Chain (SPARQL-style): Topic relation TargetEntity(Alpha)",
            "candidate_size": 2,
        }]
        sparql = [
            {"search_type": "entity_search", "trace_id": 2, "candidate_entity_ids": ["m.old"],
             "candidate_entity_names": ["Old"], "literal_candidates": [], "sparql": "SELECT ?s WHERE {}"},
            {"search_type": "entity_search", "trace_id": 4, "candidate_entity_ids": ["m.alpha", "m.beta"],
             "candidate_entity_names": ["Alpha", "Beta"], "literal_candidates": [], "sparql": "SELECT ?s WHERE { ?s ?p ?o }"},
            {"search_type": "entity_search", "trace_id": 8, "candidate_entity_ids": ["m.later"],
             "candidate_entity_names": ["Later"], "literal_candidates": [], "sparql": "SELECT ?s WHERE { ?s ?p ?o }"},
        ]
        eligible, reason, event, candidates = _selector_eligibility(
            {"router_execution_mode": "cor"}, "Which?", history, sparql,
        )
        self.assertTrue(eligible)
        self.assertEqual(reason, "ELIGIBLE_SINGLE_ENTITY_COR_FILTER")
        self.assertIs(event, history[0])
        self.assertEqual([row["id"] for row in candidates], ["m.alpha", "m.beta"])
        self.assertEqual(candidates[0]["source"]["entity_search_trace_id"], 4)

    def test_non_cor_or_multiple_filter_boundaries_are_ineligible(self):
        event = {"type": "filter_answer", "trace_id": 7, "prompt": "context", "candidate_size": 1}
        eligible, reason, _, _ = _selector_eligibility(
            {"router_execution_mode": "branch"}, "Which?", [event], [],
        )
        self.assertFalse(eligible)
        self.assertEqual(reason, "NON_COR_ROUTER_MODE")
        eligible, reason, _, _ = _selector_eligibility(
            {"router_execution_mode": "cor"}, "Which?", [event, dict(event, trace_id=9)], [],
        )
        self.assertFalse(eligible)
        self.assertEqual(reason, "FILTER_BOUNDARY_COUNT_NOT_ONE")

    def test_aggregate_or_non_entity_baseline_is_ineligible(self):
        event = {"type": "filter_answer", "trace_id": 7, "prompt": "context", "candidate_size": 1}
        result = {"router_execution_mode": "cor"}
        eligible, reason, _, _ = _selector_eligibility(
            result, "How many awards did this person win?", [event], [],
        )
        self.assertFalse(eligible)
        self.assertEqual(reason, "AGGREGATE_ANSWER_NOT_ENTITY_SELECTABLE")
        eligible, reason, _, _ = _selector_eligibility(
            {"router_execution_mode": "cor", "results": [{"id": "", "name": "4"}]},
            "Which awards?", [event], [],
        )
        self.assertFalse(eligible)
        self.assertEqual(reason, "BASELINE_RESULT_NOT_ENTITY_ID_ONLY")


if __name__ == "__main__":
    unittest.main()
