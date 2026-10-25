import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_phase2_4_mixed_failure_validation import (  # noqa: E402
    classify_relation_failure,
    classify_structure_pair,
    dominant_decision,
    failure_rate,
    gold_structure_from_s_expression,
)


class Phase24MixedFailureValidationTests(unittest.TestCase):
    def test_gold_structure_family_classification(self):
        final = "(AND answer_type (AND (JOIN relation a) (JOIN relation b)))"
        intermediate = "(AND answer_type (JOIN relation (AND (JOIN relation a) (JOIN relation b))))"
        self.assertEqual(gold_structure_from_s_expression(final), "GOLD_BRANCH_FINAL")
        self.assertEqual(gold_structure_from_s_expression(intermediate), "GOLD_BRANCH_INTERMEDIATE")

    def test_structure_false_negative_and_control(self):
        self.assertEqual(classify_structure_pair("GOLD_BRANCH_FINAL", "LINEAR"), "STRUCTURE_FALSE_NEGATIVE")
        self.assertEqual(classify_structure_pair("GOLD_BRANCH_FINAL", "BRANCH_FINAL"), "CORRECT_STRUCTURE")
        self.assertEqual(classify_structure_pair("GOLD_LINEAR", "BRANCH_FINAL"), "STRUCTURE_FALSE_POSITIVE")

    def test_relation_track_requires_candidate_available(self):
        row = {
            "expected_relation": "celebrities.celebrity.sexual_relationships",
            "expected_direction": "outgoing",
            "selected_relation": [{"relation": "base.popstra.celebrity.dated", "direction": "outgoing"}],
            "correct_relation_available_in_candidates": True,
        }
        self.assertEqual(classify_relation_failure(row), "GENERIC_RELATION_PREFERRED")
        row["correct_relation_available_in_candidates"] = False
        self.assertEqual(classify_relation_failure(row), "CANDIDATE_MISSING_NOT_TRACK_B")

    def test_rate_and_dominant_decision_are_conservative(self):
        rows = {
            "A": [{"suspected_failure": True}, {"suspected_failure": False}],
            "B": [{"suspected_failure": True}, {"suspected_failure": True}],
            "C": [{"suspected_failure": True}, {"suspected_failure": True}],
        }
        self.assertEqual(failure_rate(rows["A"]), 0.5)
        self.assertEqual(dominant_decision(rows), "MIXED_NO_DOMINANT_FAILURE")

    def test_insufficient_evidence_is_inconclusive(self):
        rows = {
            "A": [{"suspected_failure": False}],
            "B": [{"suspected_failure": False}],
            "C": [{"suspected_failure": False}],
        }
        self.assertEqual(dominant_decision(rows), "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
