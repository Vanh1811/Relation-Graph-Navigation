from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import score as score_module


class OfflineScorerTests(unittest.TestCase):
    def _write_rows(self, root: Path, *, regression: bool = False):
        gold = []
        results = []
        for index in range(12):
            case_id = f"iter26_case_{index + 1:02d}"
            gold.append({
                "case_id": case_id,
                "question_key": f"cwq::q{index}",
                "dataset": "cwq",
                "gold_answer_ids_scoring_only": [f"m.gold{index}"],
            })
            baseline = [f"m.gold{index}"] if regression and index == 0 else [f"m.wrong{index}"]
            intervention = [f"m.wrong{index}"] if regression and index == 0 else [f"m.gold{index}"]
            results.append({
                "case_id": case_id,
                "baseline_prediction_ids": baseline,
                "intervention_prediction_ids": intervention,
                "selector_eligible": True,
                "selector_invoked": True,
                "candidate_ids_snapshot": [f"m.gold{index}", f"m.wrong{index}"],
                "selector": {
                    "status": "VALID_SELECTION",
                    "valid": True,
                    "candidate_membership_valid": True,
                    "prediction_source": "INTERVENTION_SELECTOR",
                },
            })
        (root / "gold.jsonl").write_text("".join(json.dumps(row) + "\n" for row in gold))
        (root / "run").mkdir()
        (root / "run/PAIRED_RESULTS_NON_GOLD.jsonl").write_text("".join(json.dumps(row) + "\n" for row in results))
        (root / "run/RUN_METADATA.json").write_text(json.dumps({
            "status": "COMPLETE", "evidence_class": "FIXTURE_TEST",
        }))
        return root / "run"

    def test_preregistered_gate_passes_only_with_net_rescues_and_no_control_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_rows(root)
            old_gold = score_module.GOLD
            score_module.GOLD = root / "gold.jsonl"
            try:
                report = score_module.score(root / "run", root / "scored.json")
            finally:
                score_module.GOLD = old_gold
            self.assertEqual(report["paired_hits_at_1"], {"wins": 12, "losses": 0, "ties": 0, "net_wins": 12})
            self.assertEqual(report["promotion_gate"], "PASS_TO_NEW_AUTHORIZATION_FOR_LARGER_TEST")
            self.assertEqual(report["evidence_class"], "FIXTURE_TEST")

    def test_any_baseline_correct_regression_fails_promotion_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_rows(root, regression=True)
            old_gold = score_module.GOLD
            score_module.GOLD = root / "gold.jsonl"
            try:
                report = score_module.score(root / "run", root / "scored.json")
            finally:
                score_module.GOLD = old_gold
            self.assertEqual(report["baseline_correct_regressions"], 1)
            self.assertEqual(report["promotion_gate"], "FAIL_OR_INCONCLUSIVE_NO_TUNING_ON_THIS_COHORT")


if __name__ == "__main__":
    unittest.main()
