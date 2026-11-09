from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.diagnostics.iter_026_answer_filter.score_attempt3 import GOLD, score


class Attempt3ScorerTests(unittest.TestCase):
    def test_upstream_truncation_is_a_paired_no_output_tie_in_full_denominator(self):
        gold_rows = [
            json.loads(line)
            for line in GOLD.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            result_rows = []
            for gold in gold_rows:
                result_rows.append({
                    "case_id": gold["case_id"],
                    "question_key": gold["question_key"],
                    "baseline_prediction_ids": [],
                    "intervention_prediction_ids": [],
                    "selector_eligible": False,
                    "selector_invoked": False,
                    "selector_eligibility_reason": "UPSTREAM_BASELINE_INCOMPLETE_NO_SELECTOR_COMPARISON",
                    "prediction_source": "PAIRED_UPSTREAM_FAILURE_NO_OUTPUT",
                    "execution_status": "UPSTREAM_EMPTY_OR_TRUNCATED_FAILURE",
                    "upstream_failure_metadata": {"finish_reason": "length"},
                })
            (run_dir / "PAIRED_RESULTS_NON_GOLD.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in result_rows), encoding="utf-8"
            )
            (run_dir / "RUN_METADATA.json").write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
            report = score(run_dir, run_dir / "scored.json")

        self.assertEqual(report["question_unit_count_scored"], 12)
        self.assertEqual(report["paired_hits_at_1"], {"wins": 0, "losses": 0, "ties": 12, "net_wins": 0})
        self.assertEqual(report["upstream_empty_or_truncated_case_count"], 12)
        self.assertEqual(report["promotion_gate"], "FAIL_OR_INCONCLUSIVE_NO_TUNING_ON_THIS_COHORT")


if __name__ == "__main__":
    unittest.main()
