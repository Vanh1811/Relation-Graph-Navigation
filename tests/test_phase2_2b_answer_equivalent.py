import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_phase2_2b_answer_equivalent_audit import (  # noqa: E402
    ONE_HOP_RESULT_LIMIT,
    TWO_HOP_PATH_LIMIT,
    answer_metrics,
    classify_one_hop,
    one_hop_query,
    semantic_for_winner,
    two_hop_query,
)


class Phase22bAnswerEquivalentTests(unittest.TestCase):
    def test_exact_answer_match_deduplicates_results_and_prefers_entity_ids(self):
        values = [
            {"raw_value": "http://rdf.freebase.com/ns/m.answer", "value_type": "uri", "entity_id": "m.answer"},
            {"raw_value": "http://rdf.freebase.com/ns/m.answer", "value_type": "uri", "entity_id": "m.answer"},
        ]
        metrics = answer_metrics(values, ["m.answer"], ["Gold Answer"])
        self.assertTrue(metrics["exact_answer_set_match"])
        self.assertEqual(metrics["returned_answer_count"], 1)

    def test_literal_label_can_match_gold_entity_after_normalization(self):
        values = [{"raw_value": "The Gold Answer!", "value_type": "literal", "entity_id": ""}]
        metrics = answer_metrics(values, ["m.answer"], ["gold answer"])
        self.assertEqual(metrics["gold_overlap_count"], 1)
        self.assertTrue(metrics["exact_answer_set_match"])

    def test_missing_gold_evidence_is_not_treated_as_match(self):
        metrics = answer_metrics([], [], [])
        self.assertEqual(metrics["gold_overlap_count"], 0)
        self.assertIsNone(metrics["gold_recall"])
        self.assertFalse(metrics["exact_answer_set_match"])

    def test_one_hop_classifications(self):
        self.assertEqual(
            classify_one_hop([{"exact_answer_set_match": True, "gold_overlap_count": 1}]),
            "ONE_HOP_EXACT_EQUIVALENT",
        )
        self.assertEqual(
            classify_one_hop([{"exact_answer_set_match": False, "gold_overlap_count": 1}]),
            "ONE_HOP_PARTIAL_GOLD_REACHABLE",
        )
        self.assertEqual(
            classify_one_hop([{"exact_answer_set_match": False, "gold_overlap_count": 0}]),
            "NO_ONE_HOP_GOLD_REACHABILITY",
        )

    def test_queries_are_bounded_and_cover_both_directions(self):
        outgoing = one_hop_query("m.current", "people.person.parents", "outgoing")
        incoming = one_hop_query("m.current", "people.person.parents", "incoming")
        self.assertIn("ns:m.current ns:people.person.parents ?value", outgoing)
        self.assertIn("?value ns:people.person.parents ns:m.current", incoming)
        self.assertIn(f"LIMIT {ONE_HOP_RESULT_LIMIT}", outgoing)

        two_hop = two_hop_query(
            "m.current",
            [
                {"relation": "people.person.parents", "direction": "outgoing"},
                {"relation": "people.person.children", "direction": "incoming"},
            ],
            ["m.gold"],
        )
        self.assertIn("VALUES ?gold { ns:m.gold }", two_hop)
        self.assertIn(f"LIMIT {TWO_HOP_PATH_LIMIT}", two_hop)
        self.assertIn("BIND(\"incoming\" AS ?direction1)", two_hop)

    def test_semantic_equivalence_is_not_inferred_for_unknown_relation(self):
        assessment = semantic_for_winner("unknown-case", "people.person.parents")
        self.assertEqual(assessment["semantic_equivalent"], "UNCLEAR")


if __name__ == "__main__":
    unittest.main()
