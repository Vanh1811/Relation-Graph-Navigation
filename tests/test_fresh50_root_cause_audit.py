import json
import unittest
from pathlib import Path

from scripts.audit_fresh50_tgar_vs_cfp import (
    AUDIT_ROOT,
    METHODS,
    SOURCE_ROOT,
    integrity,
    load_bundle,
    primary_diagnosis,
)


class Fresh50RootCauseAuditTests(unittest.TestCase):
    def test_frozen_source_has_exact_slots_and_reproduces_pattern(self):
        bundle = load_bundle()
        checks = integrity(bundle)
        self.assertEqual(checks["question_count"], 50)
        self.assertEqual(checks["unique_question_count"], 50)
        self.assertEqual(checks["run_count"], 150)
        self.assertEqual(checks["duplicate_slots"], [])
        self.assertEqual(checks["missing_slots"], [])
        self.assertEqual(checks["missing_gold_keys"], [])
        self.assertEqual(checks["disagreement_question_count"], 11)
        self.assertEqual(checks["all_three_pattern_counts"].get("1|1|1"), 22)
        self.assertEqual(checks["all_three_pattern_counts"].get("0|0|0"), 17)

    def test_method_metrics_match_frozen_report(self):
        checks = integrity(load_bundle())
        metrics = json.loads((SOURCE_ROOT / "metrics.json").read_text())
        for method in METHODS:
            self.assertEqual(checks["recomputed_methods"][method]["run_count"], metrics["methods"][method]["run_count"])
            self.assertEqual(checks["recomputed_methods"][method]["correct_questions"], metrics["methods"][method]["correct_questions"])
            self.assertAlmostEqual(checks["recomputed_methods"][method]["mean_f1"], metrics["methods"][method]["mean_f1"], places=12)

    def test_fallback_reason_is_not_conflated_with_native_success(self):
        bundle = load_bundle()
        row = bundle["runs"][("cwq", "WebQTest-712_67f5f717564163742ca2588b7e186fd8", "phase2_5_6")][1]
        self.assertTrue(row["branch_attempted"])
        self.assertTrue(row["fallback_used"])
        self.assertEqual(row["fallback_reason"], "empty_strict_merge")
        self.assertFalse(row["native_branch_success"])

    def test_missing_candidate_state_is_explicitly_unavailable(self):
        bundle = load_bundle()
        manifest = bundle["manifest"][("cwq", "WebQTest-450_8928d9663aabbda9525d0b8cbf2e3e39")]
        gold = bundle["gold"][("cwq", "WebQTest-450_8928d9663aabbda9525d0b8cbf2e3e39")]
        tgar = bundle["runs"][("cwq", "WebQTest-450_8928d9663aabbda9525d0b8cbf2e3e39", "v1_1_3")][1]
        diagnosis = primary_diagnosis(tgar, manifest, gold)
        self.assertIn(diagnosis["primary_root_cause"], {"UNDETERMINED", "RELATION_RANKING_OR_DIRECTION", "METHOD_EXPRESSIVITY_LIMIT"})
        self.assertNotIn("GOLD_ORACLE", diagnosis["primary_root_cause"])

    def test_audit_outputs_have_required_row_counts(self):
        self.assertTrue((AUDIT_ROOT / "integrity_checks.json").exists())
        traces = [line for line in (AUDIT_ROOT / "method_traces.jsonl").read_text().splitlines() if line.strip()]
        cases = (AUDIT_ROOT / "case_matrix.csv").read_text().splitlines()
        divergences = (AUDIT_ROOT / "first_divergence_audit.csv").read_text().splitlines()
        self.assertEqual(len(traces), 150)
        self.assertEqual(len(cases) - 1, 50)
        self.assertEqual(len(divergences) - 1, 50)


if __name__ == "__main__":
    unittest.main()
