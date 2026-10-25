import unittest

from scripts.run_phase3a_1_safety_cap_sensitivity import (
	balanced_ids,
	classify_stability,
	metric_signature,
	recompute_class,
)


class Phase3A1SafetyCapTests(unittest.TestCase):
	def test_recompute_necessity_classification(self):
		self.assertEqual(
			recompute_class([{"status": "ok", "material_effect": True}, {"status": "ok", "material_effect": True}]),
			"STRONGLY_BRANCH_REQUIRED",
		)
		self.assertEqual(
			recompute_class([{"status": "ok", "material_effect": True}, {"status": "ok", "material_effect": False}]),
			"PARTIALLY_BRANCH_REQUIRED",
		)
		self.assertEqual(
			recompute_class([{"status": "ok", "material_effect": False}, {"status": "ok", "material_effect": False}]),
			"ANSWER_EQUIVALENT_LINEARIZATION",
		)

	def test_stability_classification(self):
		stable_metric = {"status": "ok", "truncated": False, "f1": 0.5}
		self.assertEqual(
			classify_stability({"1000": stable_metric, "2000": stable_metric, "5000": stable_metric}, {"1000": "STRONGLY_BRANCH_REQUIRED", "2000": "STRONGLY_BRANCH_REQUIRED", "5000": "STRONGLY_BRANCH_REQUIRED"}, [1000, 2000, 5000]),
			"STABLE_ALL_CAPS",
		)
		self.assertEqual(
			classify_stability({"1000": stable_metric, "2000": {**stable_metric, "f1": 0.4}, "5000": {**stable_metric, "f1": 0.4}}, {"1000": "STRONGLY_BRANCH_REQUIRED", "2000": "STRONGLY_BRANCH_REQUIRED", "5000": "STRONGLY_BRANCH_REQUIRED"}, [1000, 2000, 5000]),
			"STABLE_MATERIAL_EFFECT",
		)
		self.assertEqual(
			classify_stability({"1000": stable_metric, "2000": {**stable_metric, "truncated": True}, "5000": {**stable_metric, "truncated": True}}, {"1000": "STRONGLY_BRANCH_REQUIRED", "2000": "STRONGLY_BRANCH_REQUIRED", "5000": "STRONGLY_BRANCH_REQUIRED"}, [1000, 2000, 5000]),
			"UNRESOLVED_STILL_TRUNCATED",
		)

	def test_same_cap_change_does_not_change_query_semantics(self):
		self.assertEqual(
			metric_signature({"status": "ok", "predicted_answer_count": 2, "truncated": False, "gold_overlap": 1, "gold_loss": 0, "false_positive_count": 1, "precision": 0.5, "recall": 1.0, "f1": 0.66, "material_effect": True, "cap": 1000}),
			metric_signature({"status": "ok", "predicted_answer_count": 2, "truncated": False, "gold_overlap": 1, "gold_loss": 0, "false_positive_count": 1, "precision": 0.5, "recall": 1.0, "f1": 0.66, "material_effect": True, "cap": 5000}),
		)

	def test_balanced_sampling_is_deterministic(self):
		metadata = {
			"a": {"dataset": "cwq", "gold_structure": "BRANCH_FINAL"},
			"b": {"dataset": "grailqa", "gold_structure": "BRANCH_FINAL"},
			"c": {"dataset": "grailqa", "gold_structure": "BRANCH_INTERMEDIATE"},
		}
		self.assertEqual(balanced_ids(["c", "a", "b"], metadata, 3), ["a", "b", "c"])


if __name__ == "__main__":
	unittest.main()
