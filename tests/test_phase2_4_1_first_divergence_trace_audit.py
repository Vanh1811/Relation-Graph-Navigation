import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_phase2_4_1_first_divergence_trace_audit import (  # noqa: E402
    downstream_status,
    first_divergence_label,
    gold_structure_from_sparql,
    interaction_labels,
)


class Phase241FirstDivergenceAuditTests(unittest.TestCase):
    def test_structure_precedes_relation_and_state(self):
        checks = {
            "structure_match": False,
            "target_variable_match": False,
            "candidate_available": True,
            "relation_selection_match": False,
        }
        self.assertEqual(first_divergence_label(checks), "STRUCTURE_ROUTING")

    def test_relation_is_selected_when_structure_and_state_match(self):
        checks = {
            "structure_match": True,
            "constraint_decomposition_match": True,
            "target_variable_match": True,
            "entity_state_match": True,
            "candidate_available": True,
            "relation_selection_match": False,
        }
        self.assertEqual(first_divergence_label(checks), "RELATION_GROUNDING")

    def test_downstream_after_first_failure(self):
        self.assertEqual(downstream_status("STRUCTURE_ROUTING", "RELATION_GROUNDING"), "DOWNSTREAM_AFTER_FIRST_FAILURE")
        self.assertEqual(downstream_status("RELATION_GROUNDING", "RELATION_GROUNDING"), "FIRST_FAILURE_CANDIDATE")

    def test_unknown_evidence_does_not_create_failure(self):
        self.assertEqual(first_divergence_label({"structure_match": True}), "INSUFFICIENT_EVIDENCE")

    def test_gold_cwq_branch_is_classified_from_graph(self):
        row = {
            "id": "example",
            "question": "q",
            "topic_entity": [{"id": "m.topic"}],
            "sparql": """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?x WHERE {
  ns:m.topic ns:r.one ?x .
  ?x ns:r.two ns:m.constraint .
  ?x ns:r.three ?y .
}""",
        }
        self.assertEqual(gold_structure_from_sparql(row), "GOLD_BRANCH_FINAL")

    def test_interaction_classification_preserves_causal_order(self):
        labels = interaction_labels("STRUCTURE_ROUTING", True, True, False)
        self.assertEqual(labels["routing_to_relation"], "ROUTING_CAUSED_DOWNSTREAM_RELATION_FAILURE")
        self.assertEqual(labels["relation_to_state"], "STATE_ALREADY_WRONG_BEFORE_RELATION")


if __name__ == "__main__":
    unittest.main()
