import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_phase2_4_4_routing_decision_error_analysis import (  # noqa: E402
    aggregate,
    branch_type_label,
    choose_decision,
    independent_source_label,
    primary_error,
    router_detected_source_count,
    sequential_label,
    shared_answer_label,
    shared_intermediate_label,
    uncertainty_label,
)


def router(structure="LINEAR", reason="", branches=None):
    return {
        "structure": structure,
        "reason": reason,
        "branches": branches or [],
        "requested_final_answer_type": "answer",
        "requested_final_semantic_property": "property",
        "shared_answer_type": "answer",
    }


class Phase244RoutingDecisionTests(unittest.TestCase):
    def test_negative_source_statement_is_not_positive_detection(self):
        value = router(reason="not independent constraints from multiple grounded entities")
        self.assertEqual(router_detected_source_count(value), 1)

    def test_independent_source_detection_and_shared_answer(self):
        gold = {
            "gold_independent_sources": 2,
            "gold_shared_answer": True,
            "gold_shared_intermediate": False,
            "gold_structure": "GOLD_BRANCH_FINAL",
        }
        branches = [
            {"target_answer_type": "answer"},
            {"target_answer_type": "answer"},
        ]
        value = router("BRANCH_FINAL", "two grounded entities constrain the same answer", branches)
        self.assertEqual(independent_source_label(gold, value), "INDEPENDENT_SOURCES_DETECTED")
        self.assertEqual(shared_answer_label(gold, value), "SHARED_ANSWER_DETECTED")

    def test_shared_intermediate_and_sequentialization_labels(self):
        gold = {
            "gold_independent_sources": 2,
            "gold_shared_answer": False,
            "gold_shared_intermediate": True,
            "gold_structure": "GOLD_BRANCH_INTERMEDIATE",
        }
        value = router(reason="This is a sequential path rather than independent constraints")
        self.assertEqual(shared_intermediate_label(gold, value), "SHARED_INTERMEDIATE_MISINTERPRETED")
        self.assertEqual(sequential_label(gold, value), "FALSE_SEQUENTIALIZATION")
        self.assertEqual(branch_type_label(gold, value), "INTERMEDIATE_AS_LINEAR")

    def test_uncertainty_and_primary_error_are_distinct(self):
        gold = {
            "gold_independent_sources": 2,
            "gold_shared_answer": True,
            "gold_shared_intermediate": False,
            "gold_structure": "GOLD_BRANCH_FINAL",
        }
        value = router("UNCERTAIN", "different semantic properties are incompatible")
        self.assertEqual(uncertainty_label(gold, value), "CONFLICTING_SIGNALS")
        primary, secondary = primary_error(gold, value, "HARMFUL_ROUTING_MISMATCH")
        self.assertEqual(primary, "ANSWER_TYPE_CONFLICT")
        self.assertIn("BRANCH_TYPE_CONFUSION", secondary)

    def test_aggregate_and_decision_count_primary_errors(self):
        rows = []
        for index in range(3):
            rows.append({
                "routing_outcome": "HARMFUL_ROUTING_MISMATCH",
                "primary_router_error": "INDEPENDENT_SOURCE_MISSED",
                "independent_source_detection": "INDEPENDENT_SOURCE_MISSED",
                "shared_answer_detection": "SHARED_ANSWER_MISSED",
                "shared_intermediate_detection": "UNKNOWN",
                "sequential_vs_independent": "FALSE_SEQUENTIALIZATION",
                "uncertainty_analysis": "UNKNOWN",
                "reused_from_phase24": True,
            })
        rows.append({
            "routing_outcome": "BENIGN_ROUTING_MISMATCH",
            "primary_router_error": "FALSE_SEQUENTIALIZATION",
            "independent_source_detection": "INDEPENDENT_SOURCE_MISSED",
            "shared_answer_detection": "SHARED_ANSWER_MISSED",
            "shared_intermediate_detection": "UNKNOWN",
            "sequential_vs_independent": "FALSE_SEQUENTIALIZATION",
            "uncertainty_analysis": "UNKNOWN",
            "reused_from_phase24": True,
        })
        rows.append({
            "routing_outcome": "CORRECT_ROUTING_CONTROL",
            "primary_router_error": "NONE",
            "independent_source_detection": "INDEPENDENT_SOURCES_DETECTED",
            "shared_answer_detection": "SHARED_ANSWER_DETECTED",
            "shared_intermediate_detection": "UNKNOWN",
            "sequential_vs_independent": "UNKNOWN",
            "uncertainty_analysis": "UNKNOWN",
            "reused_from_phase24": False,
        })
        summary = aggregate(rows)
        self.assertEqual(summary["harmful_primary_error_counts"]["INDEPENDENT_SOURCE_MISSED"], 3)
        self.assertEqual(choose_decision(rows, summary), "CLEAR_ROUTING_ERROR_PATTERN")


if __name__ == "__main__":
    unittest.main()
