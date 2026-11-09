import unittest

from scripts.diagnostics.run_fresh50_continuous_agent import (
    build_bgp_step_query,
    anchor_gold_bgp_to_root,
    find_gold_entity_path,
    ordered_bgp,
    parse_directional_relation_prompt,
    relation_edge_match_kind,
    root_anchor,
    resolve_labels_to_candidate_ids,
    validate_id_payload,
)
from scripts.diagnostics.run_fresh50_autonomous_diagnosis import graph_summary


class StrictCandidateIdAdapterTests(unittest.TestCase):
    def test_unique_normalized_label_maps_to_allowed_id(self):
        result = resolve_labels_to_candidate_ids(
            ["  ALPHA   CITY "], [("m.a", "Alpha City"), ("m.b", "Beta City")]
        )
        self.assertEqual(result["status"], "RESOLVED_UNIQUE_CANDIDATE_IDS")
        self.assertEqual(result["answer_ids"], ["m.a"])
        self.assertTrue(result["membership_valid"])

    def test_ambiguous_label_is_rejected(self):
        result = resolve_labels_to_candidate_ids(
            ["Springfield"], [("m.a", "Springfield"), ("m.b", "Springfield")]
        )
        self.assertEqual(result["status"], "AMBIGUOUS_LABEL_REJECTED")
        self.assertEqual(result["answer_ids"], [])

    def test_unmapped_label_is_rejected(self):
        result = resolve_labels_to_candidate_ids(["None"], [("m.a", "Alpha")])
        self.assertEqual(result["status"], "UNMAPPED_LABEL_REJECTED")
        self.assertEqual(result["answer_ids"], [])

    def test_id_payload_enforces_schema_membership_and_deduplication(self):
        valid = validate_id_payload({"answer_ids": ["m.a", "m.a"]}, {"m.a", "m.b"})
        self.assertTrue(valid["valid"])
        self.assertEqual(valid["answer_ids"], ["m.a"])
        self.assertEqual(valid["duplicates_removed"], 1)
        invalid_id = validate_id_payload({"answer_ids": ["m.outside"]}, {"m.a"})
        self.assertEqual(invalid_id["status"], "ID_OUTSIDE_CANDIDATE_SET")
        invalid_schema = validate_id_payload({"answer": ["m.a"]}, {"m.a"})
        self.assertEqual(invalid_schema["status"], "INVALID_SCHEMA")
        invalid_json = validate_id_payload("not json", {"m.a"})
        self.assertEqual(invalid_json["status"], "INVALID_JSON")

    def test_root_anchor_detects_outgoing_and_incoming_ids(self):
        self.assertEqual(
            root_anchor("SELECT ?relation WHERE { ns:m.topic ?relation ?x . }", "relation_search_head"),
            ("m.topic", "OUTGOING"),
        )
        self.assertEqual(
            root_anchor("SELECT ?relation WHERE { ?x ?relation ns:g.topic . }", "relation_search_tail"),
            ("g.topic", "INCOMING"),
        )

    def test_bgp_order_and_query_preserve_shared_intermediate_bindings(self):
        triples = [
            {"subject": "?c", "predicate": "ns:government.political_district.representatives", "object": "?k"},
            {"subject": "?k", "predicate": "ns:government.government_position_held.appointed_by", "object": "ns:m.topic"},
            {"subject": "?c", "predicate": "ns:government.governmental_jurisdiction.official_symbols", "object": "?y"},
        ]
        ordered = ordered_bgp("m.topic", triples)
        self.assertEqual(ordered[0]["object"], "ns:m.topic")
        query = build_bgp_step_query(
            {"subject": "?c", "predicate": "ns:government.governmental_jurisdiction.official_symbols", "object": "?y"},
            [{"c": "ns:m.county"}],
        )
        self.assertIn("SELECT DISTINCT ?c ?y", query)
        self.assertIn("VALUES (?c)", query)
        self.assertIn("?y", query)

    def test_gold_path_follows_inverse_edges_from_root(self):
        triples = [
            {"subject": "?k", "predicate": "ns:appointed_by", "object": "ns:m.root"},
            {"subject": "?c", "predicate": "ns:represents", "object": "?k"},
            {"subject": "?c", "predicate": "ns:symbols", "object": "?x"},
        ]
        path = find_gold_entity_path("m.root", "?x", triples)
        self.assertEqual(len(path), 3)
        self.assertEqual([left for _, left in path], [False, False, True])

    def test_graph_summary_parses_default_prefix_used_by_grailqa(self):
        summary = graph_summary(
            "PREFIX : <http://rdf.freebase.com/ns/> SELECT (?x0 AS ?value) WHERE { "
            "?x0 :food.type_of_dish.dishes ?x1 . VALUES ?x2 { :m.0m4v } "
            "?x1 :food.dish.ingredients ?x2 . }"
        )
        self.assertEqual(summary["answer_variable"], "?x0")
        self.assertEqual(len(summary["triple_patterns"]), 2)
        self.assertEqual(summary["triple_patterns"][1]["object"], "?x2")
        self.assertEqual(summary["values_bindings"]["x2"], ["ns:m.0m4v"])
        anchored = anchor_gold_bgp_to_root(
            summary["triple_patterns"], summary["values_bindings"], "m.0m4v")
        self.assertEqual(anchored[1]["object"], "ns:m.0m4v")

    def test_directional_prompt_separates_head_and_tail_candidate_pools(self):
        prompt = (
            "# TopicEntity(Alpha) ?relation ?x\n"
            "people.person.parents; common.topic;\n"
            "# ?x ?relation TopicEntity(Alpha)\n"
            "government.position.appointed_by; people.person.parents;\n"
            "Answer: "
        )
        head, tail = parse_directional_relation_prompt(prompt)
        self.assertEqual(head, {"people.person.parents", "common.topic"})
        self.assertEqual(tail, {"government.position.appointed_by", "people.person.parents"})

    def test_explicit_inverse_matches_only_with_opposite_direction(self):
        inverse_map = {"food.dish.ingredients": ["food.ingredient.dishes"]}
        self.assertEqual(
            relation_edge_match_kind(
                ("food.dish.ingredients", "INCOMING"),
                ("food.ingredient.dishes", "OUTGOING"), inverse_map),
            "EXPLICIT_INVERSE_EQUIVALENT",
        )
        self.assertEqual(
            relation_edge_match_kind(
                ("food.dish.ingredients", "INCOMING"),
                ("food.ingredient.dishes", "INCOMING"), inverse_map),
            "NO_MATCH",
        )


if __name__ == "__main__":
    unittest.main()
