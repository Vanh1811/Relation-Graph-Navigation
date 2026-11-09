import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/fresh50_autonomous_causal_diagnosis"


class Fresh50AutonomousDiagnosisTests(unittest.TestCase):
    def test_frozen_question_and_targeted_replay_counts(self):
        with (OUT / "case_inventory.csv").open(encoding="utf-8", newline="") as handle:
            inventory = list(csv.DictReader(handle))
        with (OUT / "question_attribution.csv").open(encoding="utf-8", newline="") as handle:
            attribution = list(csv.DictReader(handle))
        with (OUT / "gold_kg_replay.csv").open(encoding="utf-8", newline="") as handle:
            kg = list(csv.DictReader(handle))
        with (OUT / "validator_replay.csv").open(encoding="utf-8", newline="") as handle:
            validators = list(csv.DictReader(handle))
        with (OUT / "answer_selection_counterfactuals.csv").open(encoding="utf-8", newline="") as handle:
            selections = list(csv.DictReader(handle))
        with (OUT / "entity_candidate_reachability.csv").open(encoding="utf-8", newline="") as handle:
            reachability = list(csv.DictReader(handle))
        self.assertEqual(len(inventory), 50)
        self.assertEqual(len(attribution), 50)
        self.assertEqual(sum("ALL_THREE_WRONG" in row["outcome_tags"] for row in inventory), 17)
        self.assertEqual(len(kg), 28)
        self.assertEqual(len(validators), 100)
        self.assertEqual(len(selections), 26)
        self.assertEqual(len(reachability), 21)

    def test_gold_replay_is_separate_from_oracle_executor(self):
        summary = json.loads((OUT / "quantitative_summary.json").read_text(encoding="utf-8"))
        gate = json.loads((OUT / "stop_gate.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["shared_failures_with_gold_ids_reproduced"], 17)
        self.assertEqual(summary["oracle_executor_replay_cases_completed"], 0)
        self.assertEqual(summary["oracle_executor_blocked_cases"], 22)
        self.assertEqual(summary["llm_calls"], 0)
        self.assertEqual(summary["confirmed_causal_counts_among_tgar_failures"]["ANSWER_SELECTION"], 7)
        self.assertEqual(summary["provisionally_suspected_failures"], 11)
        self.assertEqual(summary["unknown_causal_failures"], 3)
        self.assertEqual(summary["answer_selection_counterfactuals"]["wrong_candidate_selected"], 6)
        self.assertEqual(summary["answer_selection_counterfactuals"]["filter_output_unmapped"], 1)
        self.assertEqual(gate["status"], "PASSED")
        self.assertTrue(gate["scientific_completion_claimed"])
        reachability = summary["entity_candidate_reachability"]
        self.assertEqual(reachability["gold_ids_seen_any_entity_search"], 9)
        self.assertEqual(reachability["gold_ids_in_final_candidate_set"], 7)
        self.assertEqual(reachability["gold_ids_earlier_only"], 2)
        self.assertEqual(reachability["gold_ids_never_seen"], 12)

    def test_fixed_candidate_selection_counterfactual_and_controls(self):
        with (OUT / "answer_selection_counterfactuals.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rescued = [row for row in rows if row["counterfactual_status"] == "RESCUED_BY_GOLD_ID_SELECTION_FROM_FIXED_CANDIDATES"]
        controls = [row for row in rows if row["cohort_role"] == "BOTH_CORRECT_CONTROL"]
        self.assertEqual(len(rescued), 7)
        self.assertEqual(len(controls), 5)
        self.assertTrue(all(row["gold_ids_in_final_candidate_set"] != "[]" for row in rescued))
        self.assertTrue(all(json.loads(row["gold_candidate_labels_exposed_to_selector"]) for row in rescued))
        self.assertTrue(all(row["counterfactual_hits_at_1"] == "1" for row in rescued))
        self.assertEqual(sum(row["filter_failure_mode"] == "NON_GOLD_CANDIDATE_SELECTED" for row in rescued), 6)
        self.assertEqual(sum(row["filter_failure_mode"] == "FILTER_OUTPUT_MATCHED_NO_CANDIDATE" for row in rescued), 1)
        self.assertTrue(all(row["counterfactual_status"] == "NEGATIVE_CONTROL_CORRECT_UNCHANGED" for row in controls))

    def test_path_only_oracle_adapter_would_change_selected_gold_graphs(self):
        with (OUT / "oracle_adapter_feasibility.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 22)
        self.assertEqual(sum(row["graph_topology"] == "PARSE_UNAVAILABLE" for row in rows), 8)
        self.assertEqual(sum(row["single_path_adapter_semantics_preserving"] == "YES" for row in rows), 0)
        self.assertTrue(all("shared" in row["required_capability"] for row in rows))


if __name__ == "__main__":
    unittest.main()
