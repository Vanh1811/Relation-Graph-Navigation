import unittest

from scripts.run_phase2_5_stage_b import (
	STRUCTURES,
	canonical,
	is_example_id,
)


class StageBSelectionTests(unittest.TestCase):
	def test_example_id_filter(self):
		self.assertTrue(is_example_id("WebQTest-1_abc"))
		self.assertTrue(is_example_id("3201234567000"))
		self.assertFalse(is_example_id("m.0123"))

	def test_structures_are_exact_stage_b_targets(self):
		self.assertEqual(STRUCTURES, ("LINEAR", "BRANCH_FINAL", "BRANCH_INTERMEDIATE"))

	def test_canonical_selection_records_are_deterministic(self):
		self.assertEqual(canonical({"b": 1, "a": 2}), '{"a":2,"b":1}')


if __name__ == "__main__":
	unittest.main()
