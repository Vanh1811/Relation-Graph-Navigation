import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_phase2_3_constraint_state_alignment import (  # noqa: E402
    case_blueprints,
    one_hop_query,
    parse_entity_from_prompt,
    parse_relation_history,
    selected_relation_value,
)


class Phase23ConstraintStateAlignmentTests(unittest.TestCase):
    def test_runtime_prompt_entity_and_history_are_recovered(self):
        prompt = (
            "Current entity: Caribbean (m.0261m)\n"
            "Current subgoal: Continue toward country\n"
            "Relation history: [\"->location.location.contains\"]\n"
        )
        self.assertEqual(parse_entity_from_prompt(prompt)["id"], "m.0261m")
        self.assertEqual(parse_entity_from_prompt(prompt)["name"], "Caribbean")
        self.assertEqual(parse_relation_history(prompt), ["->location.location.contains"])

    def test_historical_selected_relation_list_uses_top_relation(self):
        value = [
            {"relation": "location.location.contains", "direction": "outgoing", "score": 1.0},
            {"relation": "location.location.containedby", "direction": "incoming", "score": 0.99},
        ]
        self.assertEqual(selected_relation_value(value), "location.location.contains")

    def test_relation_verification_query_is_one_hop_and_bounded(self):
        outgoing = one_hop_query("m.0261m", "location.statistical_region.gdp_deflator_change", "outgoing")
        incoming = one_hop_query("m.0261m", "location.statistical_region.gdp_deflator_change", "incoming")
        self.assertIn("ns:m.0261m ns:location.statistical_region.gdp_deflator_change ?value", outgoing)
        self.assertIn("?value ns:location.statistical_region.gdp_deflator_change ns:m.0261m", incoming)
        self.assertNotIn("?middle", outgoing)
        self.assertIn("LIMIT 20", outgoing)

    def test_caribbean_blueprint_keeps_shared_country_target_explicit(self):
        case = case_blueprints()["WebQTrn-1812_8126be584feae3f358a930823db22c4e"]
        gdp = next(item for item in case["constraints"] if item["constraint_id"] == "gdp_value")
        self.assertEqual(gdp["intended_target_variable"], "?country")
        self.assertEqual(gdp["intended_target_state"]["id"], "m.05r7t")
        self.assertEqual(gdp["actual_decision_index"], 1)
        self.assertEqual(case["diagnosis"], "TARGET_VARIABLE_ALIGNMENT_ERROR")

    def test_gold_constraints_are_not_injected_as_runtime_context(self):
        case = case_blueprints()["WebQTrn-1812_8126be584feae3f358a930823db22c4e"]
        for item in case["constraints"]:
            self.assertIn("evidence", item["intended_target_state"])
            self.assertNotIn("runtime_selected_entity", item["intended_target_state"])


if __name__ == "__main__":
    unittest.main()
