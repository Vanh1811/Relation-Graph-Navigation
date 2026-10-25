import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_phase2_4_2_harmful_vs_benign_routing import (  # noqa: E402
    classify_branch_requirement,
    classify_linearization,
    classify_routing_outcome,
    choose_main_decision,
    constraint_preservation,
    feature_summary,
    gold_structure_features,
)


class Phase242HarmfulBenignRoutingTests(unittest.TestCase):
    def test_routing_outcome_labels(self):
        self.assertEqual(classify_routing_outcome("GOLD_BRANCH_FINAL", "LINEAR", False), "HARMFUL_ROUTING_MISMATCH")
        self.assertEqual(classify_routing_outcome("GOLD_BRANCH_FINAL", "LINEAR", True), "BENIGN_ROUTING_MISMATCH")
        self.assertEqual(classify_routing_outcome("GOLD_BRANCH_FINAL", "BRANCH_FINAL", True), "CORRECT_ROUTING_CONTROL")

    def test_constraint_preservation_keeps_relation_multiplicity(self):
        rows = constraint_preservation(["r", "r", "s"], ["r", "s"])
        self.assertEqual([row["status"] for row in rows], ["PRESERVED", "LOST", "PRESERVED"])

    def test_linearization_invalid_when_navigation_constraint_is_lost(self):
        preservation = constraint_preservation(["r1", "r2"], ["r1"])
        self.assertEqual(classify_linearization("LINEAR", preservation), "LINEARIZATION_INVALID")
        self.assertEqual(classify_branch_requirement({
            "independent_anchor_count": 2,
            "constraint_convergence": True,
            "requires_new_binding_transition": False,
        }, preservation, "BENIGN_ROUTING_MISMATCH"), "BRANCH_REQUIRED")
        self.assertEqual(classify_branch_requirement({
            "independent_anchor_count": 2,
            "constraint_convergence": True,
            "requires_new_binding_transition": False,
        }, preservation, "CORRECT_ROUTING_CONTROL"), "BRANCH_REQUIRED")

    def test_shared_anchor_and_variable_features(self):
        cwq = {
            "WebQTest-191_ed2cadafdbc0ac07e3fedfdfc504e54b": {
                "id": "WebQTest-191_ed2cadafdbc0ac07e3fedfdfc504e54b",
                "question": "q",
                "sparql": """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?x WHERE {
  ns:m.a ns:r.one ?y .
  ?y ns:r.two ?x .
  ?x ns:r.three ns:m.b .
  ?x ns:r.four ns:m.c .
}""",
                "topic_entity": [{"id": "m.a", "name": "A"}, {"id": "m.b", "name": "B"}, {"id": "m.c", "name": "C"}],
                "answer": [],
            }
        }
        features = gold_structure_features("WebQTest-191_ed2cadafdbc0ac07e3fedfdfc504e54b", cwq, {})
        self.assertEqual(features["independent_anchor_count"], 3)
        self.assertEqual(features["shared_variable_type"], "ANSWER")
        self.assertTrue(features["requires_new_binding_transition"])

    def test_cvt_marker_is_conservative(self):
        cwq = {
            "x": {
                "id": "x",
                "question": "q",
                "sparql": """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?x WHERE {
  ns:m.a ns:celebrities.celebrity.sexual_relationships ?y .
  ?y ns:celebrities.romantic_relationship.celebrity ?x .
}""",
                "topic_entity": [{"id": "m.a", "name": "A"}],
                "answer": [],
            }
        }
        features = gold_structure_features("x", cwq, {})
        self.assertTrue(features["contains_cvt"])

    def test_decision_is_partial_when_signals_have_exceptions(self):
        rows = [
            {
                "routing_outcome": "HARMFUL_ROUTING_MISMATCH",
                "independent_anchor_count": 2,
                "shared_variable_type": "INTERMEDIATE",
                "requires_new_binding_transition": True,
                "contains_cvt": True,
                "lost_relation_count": 1,
                "binding_dependency_lost": True,
                "branch_merge_required": True,
            },
            {
                "routing_outcome": "HARMFUL_ROUTING_MISMATCH",
                "independent_anchor_count": 1,
                "shared_variable_type": "ANSWER",
                "requires_new_binding_transition": False,
                "contains_cvt": False,
                "lost_relation_count": 1,
                "binding_dependency_lost": False,
                "branch_merge_required": False,
            },
            {
                "routing_outcome": "BENIGN_ROUTING_MISMATCH",
                "independent_anchor_count": 2,
                "shared_variable_type": "INTERMEDIATE",
                "requires_new_binding_transition": True,
                "contains_cvt": True,
                "lost_relation_count": 1,
                "binding_dependency_lost": True,
                "branch_merge_required": True,
            },
            {
                "routing_outcome": "BENIGN_ROUTING_MISMATCH",
                "independent_anchor_count": 1,
                "shared_variable_type": "NONE",
                "requires_new_binding_transition": False,
                "contains_cvt": False,
                "lost_relation_count": 0,
                "binding_dependency_lost": False,
                "branch_merge_required": False,
            },
        ]
        features = feature_summary(rows)
        self.assertEqual(choose_main_decision(rows, features), "PARTIAL_BRANCH_NECESSITY_SIGNAL")


if __name__ == "__main__":
    unittest.main()
