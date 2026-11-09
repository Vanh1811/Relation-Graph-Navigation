import unittest

from scripts.run_phase6_kqa_pro_transfer import (
    TIER_A_OPERATORS,
    TIER_B_OPERATORS,
    decision,
    majority_case_rows,
    source_rows_and_features,
    subset_manifest,
)


class Phase6SupportedSubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_rows, cls.features, cls.manifest = source_rows_and_features()
        cls.subset, cls.supported, cls.unsupported = subset_manifest(cls.source_rows, cls.features, cls.manifest)

    def test_exact_phase5_sample_is_recovered(self):
        selected = [index for values in self.manifest["selected_source_indices"].values() for index in values]
        self.assertEqual(len(selected), 180)
        self.assertEqual(len(set(selected)), 180)
        self.assertEqual(len(self.features), 180)

    def test_supported_subset_is_frozen_and_operator_bounded(self):
        self.assertEqual(self.subset["supported_case_count"], 32)
        self.assertEqual(self.subset["unsupported_case_count"], 148)
        self.assertEqual(self.subset["supported_subset_viability"], "SUPPORTED_SUBSET_SMALL")
        self.assertEqual(self.subset["tier_counts"], {"EXTENDED_EXPRESSIBLE": 11, "NATIVE_RELATIONAL": 21})
        for row in self.supported:
            self.assertTrue(set(row["operator_set"]) <= TIER_B_OPERATORS)
            self.assertIn("Find", row["operator_set"])
            if row["tier"] == "NATIVE_RELATIONAL":
                self.assertTrue(set(row["operator_set"]) <= TIER_A_OPERATORS)

    def test_case_majority_and_decision_gate(self):
        run_rows = []
        for row in self.supported[:2]:
            run_rows.extend([
                {"case_id": row["case_id"], "method": "v1_1_3", "hits_at_1": 1, "f1": 1.0},
                {"case_id": row["case_id"], "method": "v1_1_3", "hits_at_1": 0, "f1": 0.0},
                {"case_id": row["case_id"], "method": "phase2_5_6", "hits_at_1": 1, "f1": 1.0},
                {"case_id": row["case_id"], "method": "phase2_5_6", "hits_at_1": 1, "f1": 1.0},
            ])
        cases = majority_case_rows(run_rows)
        self.assertTrue(cases[(self.supported[0]["case_id"], "v1_1_3")]["case_majority_correct"])
        self.assertTrue(cases[(self.supported[0]["case_id"], "phase2_5_6")]["case_majority_correct"])
        result = decision(32, {"v1_1_3": {"case_majority_h1": 0.0}, "phase2_5_6": {"case_majority_h1": 0.0}}, {"paired_net": 0}, [], {}, "NOT_RUN_USE_--RUN")
        self.assertEqual(result["main_phase6_decision"], "KQA_TRANSFER_INCONCLUSIVE")
        self.assertEqual(result["secondary_complexity_decision"], "INSUFFICIENT_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
