import json
import unittest
from pathlib import Path

import scripts.run_phase3b_4_role_separation as experiment


OUTPUT_ROOT = Path("outputs/experiments/phase3b_4/answer_convergence_role_separation")


class TestPhase3B4ExperimentProtocol(unittest.TestCase):
	def test_development_manifest_is_the_frozen_nine_case_set(self):
		manifest = experiment.build_development_manifest()
		self.assertEqual(manifest["case_count"], 9)
		self.assertEqual(manifest["focus_counts"].get("ANSWER_CONVERGENCE_CONFUSION"), 5)

	def test_completed_development_run_fails_before_fresh_stage(self):
		manifest = experiment.read_json(OUTPUT_ROOT / "development_case_manifest.json")
		new_rows = experiment.read_jsonl(OUTPUT_ROOT / "development_run_results.jsonl")
		baseline_rows = experiment.baseline_development_rows(manifest)
		gate = experiment.development_gate(manifest, new_rows, baseline_rows)
		self.assertFalse(gate["passed"])
		self.assertEqual(gate["primary_confusion_cases_corrected"], 2)
		self.assertEqual(gate["false_intermediate_answer_case_count"], 1)
		self.assertEqual(gate["role_consistent_case_count"], 8)

	def test_failure_decision_prevents_fresh_validation(self):
		gate = json.loads((OUTPUT_ROOT / "development_gate.json").read_text())
		decision, data = experiment.decision_label(gate, None, None, None)
		self.assertEqual(decision, "ROLE_SEPARATION_DEVELOPMENT_FAILED")
		self.assertEqual(data["development_gate"], gate)
		self.assertFalse((OUTPUT_ROOT / "fresh_validation_manifest.json").exists())

	def test_failure_report_has_all_protocol_sections(self):
		report = (OUTPUT_ROOT / "PHASE3B_4_REPORT.md").read_text()
		for section in range(1, 24):
			self.assertIn(f"## {section}. ", report)
		self.assertIn("ROLE_SEPARATION_DEVELOPMENT_FAILED", report)
		self.assertIn("Fresh validation sampling", report)


if __name__ == "__main__":
	unittest.main()
