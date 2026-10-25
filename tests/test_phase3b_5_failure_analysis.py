import unittest

import scripts.analyze_phase3b_5_role_separation_failure as analysis


class TestPhase3B5FailureAnalysis(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.manifest = analysis.frozen_case_manifest()
		cls.audits = analysis.build_role_audits(cls.manifest)
		cls.contrast = analysis.contrastive_summary(cls.audits)
		cls.summary = analysis.failure_summary(cls.audits)

	def test_target_case_selection_is_exactly_six_cases(self):
		self.assertEqual(self.manifest["case_count"], 6)
		self.assertEqual(
			{case["category"] for case in self.manifest["cases"]},
			{"UNCORRECTED_CONFUSION", "CORRECTED_CONFUSION", "FALSE_INTERMEDIATE_REGRESSION"},
		)
		self.assertEqual(sum(case["category"] == "UNCORRECTED_CONFUSION" for case in self.manifest["cases"]), 3)
		self.assertEqual(sum(case["category"] == "CORRECTED_CONFUSION" for case in self.manifest["cases"]), 2)
		self.assertEqual(sum(case["category"] == "FALSE_INTERMEDIATE_REGRESSION" for case in self.manifest["cases"]), 1)

	def test_corrected_and_uncorrected_categorization(self):
		by_id = {row["case_id"]: row for row in self.audits}
		self.assertEqual(by_id["4302283003000"]["category"], "CORRECTED_CONFUSION")
		self.assertEqual(by_id["WebQTrn-1053_1d93ad4e3448e762133bda682c0693a2"]["category"], "CORRECTED_CONFUSION")
		self.assertEqual(by_id["4301356014000"]["category"], "UNCORRECTED_CONFUSION")
		self.assertEqual(by_id["3200506013000"]["category"], "FALSE_INTERMEDIATE_REGRESSION")

	def test_role_field_audit_uses_semantic_cause_precedence(self):
		by_id = {row["case_id"]: row for row in self.audits}
		self.assertEqual(by_id["4301356014000"]["manual_adjudication"]["primary_cause"], "CONVERGENCE_ROLE_WRONG")
		self.assertEqual(by_id["WebQTrn-1053_4c8649801d197e6eb5377d292c0b6ca8"]["manual_adjudication"]["primary_cause"], "SEMANTIC_OUTPUT_UNSTABLE")
		self.assertEqual(by_id["WebQTrn-436_d63127b099ed581855690cb5f1524332"]["manual_adjudication"]["primary_cause"], "REQUESTED_ANSWER_ROLE_WRONG")

	def test_false_intermediate_regression_is_classified(self):
		row = next(item for item in self.audits if item["case_id"] == "3200506013000")
		manual = row["manual_adjudication"]
		self.assertEqual(manual["primary_cause"], "MULTI_ENTITY_FALSE_INTERMEDIATE")
		self.assertEqual(manual["roles_same_entity"]["status"], "WRONG_FALSE")
		self.assertEqual(manual["requires_downstream_relation"]["status"], "FALSE_POSITIVE")

	def test_contrastive_aggregation_and_intervention_gate(self):
		self.assertEqual(self.contrast["feature_comparison"]["reason_consistent"]["CORRECTED_CONFUSION"]["count"], 2)
		self.assertEqual(self.contrast["feature_comparison"]["reason_consistent"]["UNCORRECTED_CONFUSION"]["count"], 3)
		self.assertEqual(self.summary["repeated_family_among_uncorrected"], [])
		self.assertEqual(self.summary["intervention_gate"], "NO_REPEATED_FAILURE_FAMILY")
		self.assertEqual(self.summary["decision"], "MIXED_ROLE_SEPARATION_FAILURES")


if __name__ == "__main__":
	unittest.main()
