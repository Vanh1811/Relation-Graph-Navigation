import unittest

from scripts.run_phase2_5_1_live_validation import (
	gold_structure,
	routing_outcome,
)


class TestPhase251EvaluationHelpers(unittest.TestCase):
	def test_gold_structure_mapping(self):
		self.assertEqual(gold_structure("GOLD_BRANCH_FINAL"), "BRANCH_FINAL")
		self.assertEqual(gold_structure("GOLD_BRANCH_INTERMEDIATE"), "BRANCH_INTERMEDIATE")

	def test_routing_outcome_labels(self):
		row = {"gold_structure": "GOLD_BRANCH_FINAL"}
		self.assertEqual(routing_outcome(row, "BRANCH_FINAL"), "CORRECT_BRANCH_FINAL")
		self.assertEqual(routing_outcome(row, "LINEAR"), "FALSE_LINEARIZATION")
		self.assertEqual(routing_outcome({"gold_structure": "GOLD_LINEAR"}, "BRANCH_FINAL"), "FALSE_BRANCHING")


if __name__ == "__main__":
	unittest.main()
