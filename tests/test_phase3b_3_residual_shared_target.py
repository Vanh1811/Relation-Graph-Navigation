import unittest

from scripts.analyze_phase3b_3_residual_shared_target import (
	CAUSES,
	aggregate_summary,
	decision_gate,
	primary_cause,
	select_target_cases,
	self_check_status,
	target_type,
)


def row(target="ANSWER", expected="INTERMEDIATE", check_answer=True, requires=False):
	return {
		"case_id": "synthetic",
		"canonical_shared_target_type": target,
		"expected_shared_target_type": expected,
		"requires_downstream_relation": requires,
		"shared_target_description": "A specific convergence entity description.",
		"requested_answer_description": "A specific requested answer description.",
		"downstream_relation_description": "A downstream relation description." if requires else "",
		"raw_parsed_result": {
			"constraints": [
				{"source_anchor": "source-a"},
				{"source_anchor": "source-b"},
			]
		},
		"shared_target_extraction": {
			"self_check": {
				"resulting_variable_is_requested_answer": check_answer,
				"downstream_step_required": requires,
				"valid_shared_latent_target": requires,
				"convergence_variable_description": "convergence entity",
				"rationale": "synthetic rationale",
			}
		},
	}


class TestPhase3B3ResidualSharedTarget(unittest.TestCase):
	def test_residual_case_extraction_freezes_expected_nine_cases(self):
		cases, _, _ = select_target_cases()
		self.assertEqual(len(cases), 9)
		categories = {case["case_category"] for case in cases}
		self.assertEqual(
			categories,
			{
				"DEVELOPMENT_STILL_WRONG",
				"FRESH_INTERMEDIATE_STILL_WRONG",
				"FRESH_INTERMEDIATE_UNSTABLE",
				"FRESH_FINAL_REGRESSION",
				"FRESH_CONTROL_CHANGED_TARGET",
			},
		)

	def test_target_type_reads_phase3b1_actual_target_for_old_audit_rows(self):
		self.assertEqual(target_type({"actual_target": "INTERMEDIATE"}), "INTERMEDIATE")

	def test_semantic_cause_precedence_prefers_unstable_type(self):
		first = row(target="NONE", expected="INTERMEDIATE", check_answer=True, requires=False)
		second = row(target="INTERMEDIATE", expected="INTERMEDIATE", check_answer=False, requires=True)
		self.assertEqual(primary_cause(first, [first, second]), "SEMANTIC_OUTPUT_UNSTABLE")

	def test_answer_convergence_confusion_requires_wrong_answer_self_check(self):
		wrong = row(target="ANSWER", expected="INTERMEDIATE", check_answer=True, requires=False)
		self.assertEqual(primary_cause(wrong, [wrong, wrong]), "ANSWER_CONVERGENCE_CONFUSION")
		misidentified = row(target="ANSWER", expected="INTERMEDIATE", check_answer=False, requires=False)
		self.assertNotEqual(primary_cause(misidentified, [misidentified, misidentified]), "ANSWER_CONVERGENCE_CONFUSION")

	def test_self_check_classification(self):
		correct = row(target="INTERMEDIATE", expected="INTERMEDIATE", check_answer=False, requires=True)
		wrong = row(target="ANSWER", expected="INTERMEDIATE", check_answer=True, requires=False)
		self.assertEqual(self_check_status(correct), "SELF_CHECK_CORRECT")
		self.assertEqual(self_check_status(wrong), "SELF_CHECK_WRONG")
		self.assertEqual(self_check_status(correct, {True, False}), "SELF_CHECK_UNSTABLE")

	def test_decision_gate_counts_independent_cases_not_repeats(self):
		summary = {
			"repeat_count": 6,
			"primary_causes": {
				cause: {"repeat_count": 0, "independent_case_count": 0, "case_ids": []}
				for cause in CAUSES
			},
		}
		summary["primary_causes"]["ANSWER_CONVERGENCE_CONFUSION"] = {
			"repeat_count": 6,
			"independent_case_count": 3,
			"case_ids": ["a", "b", "c"],
		}
		decision = decision_gate(summary)
		self.assertEqual(decision["decision"], "ANSWER_CONVERGENCE_CONFUSION_SUPPORTED")

	def test_decision_gate_is_mixed_below_threshold(self):
		summary = {
			"repeat_count": 4,
			"primary_causes": {
				cause: {"repeat_count": 0, "independent_case_count": 0, "case_ids": []}
				for cause in CAUSES
			},
		}
		summary["primary_causes"]["OVER_CONSERVATIVE_NONE"] = {
			"repeat_count": 4,
			"independent_case_count": 2,
			"case_ids": ["a", "b"],
		}
		self.assertEqual(decision_gate(summary)["decision"], "MIXED_RESIDUAL_SEMANTIC_FAILURES")


if __name__ == "__main__":
	unittest.main()
