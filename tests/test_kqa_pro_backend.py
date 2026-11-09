import json
import tempfile
import unittest
from pathlib import Path

from chain_of_relations.kg_backend.kqa_pro_backend import KQAProBackend, UnsupportedKQAOperator


class KQAProBackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "kb.json"
        path.write_text(
            json.dumps(
                {
                    "concepts": {
                        "Q2": {"name": "human", "instanceOf": []},
                        "Q4": {"name": "city", "instanceOf": []},
                    },
                    "entities": {
                        "Q1": {
                            "name": "Alice",
                            "instanceOf": ["Q2"],
                            "attributes": [{"key": "nickname", "value": {"type": "string", "value": "Al"}, "qualifiers": {}}],
                            "relations": [
                                {"predicate": "friend", "object": "Q3", "direction": "forward", "qualifiers": {}},
                                {"predicate": "born in", "object": "Q4", "direction": "forward", "qualifiers": {}},
                            ],
                        },
                        "Q3": {
                            "name": "Bob",
                            "instanceOf": ["Q2"],
                            "attributes": [],
                            "relations": [
                                {"predicate": "born in", "object": "Q4", "direction": "forward", "qualifiers": {}}
                            ],
                        },
                        "Q4": {"name": "Paris", "instanceOf": ["Q4"], "attributes": [], "relations": []},
                    },
                }
            ),
            encoding="utf-8",
        )
        self.backend = KQAProBackend(path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_grounding_and_directional_traversal(self):
        self.assertEqual(self.backend.lookup_entities("alice"), ["Q1"])
        self.assertEqual(self.backend.traverse(["Q1"], "friend", "forward"), ["Q3"])
        self.assertEqual(self.backend.traverse(["Q1"], "born in", "forward"), ["Q4"])
        self.assertTrue(self.backend.is_instance_of("Q1", "human"))

    def test_supported_program_execution_and_serialization(self):
        result = self.backend.execute_program(
            [
                {"function": "Find", "dependencies": [], "inputs": ["Alice"]},
                {"function": "Relate", "dependencies": [0], "inputs": ["friend", "forward"]},
                {"function": "What", "dependencies": [1], "inputs": []},
            ]
        )
        self.assertEqual(result["values"], ["Bob"])

        attr_result = self.backend.execute_program(
            [
                {"function": "Find", "dependencies": [], "inputs": ["Alice"]},
                {"function": "QueryAttr", "dependencies": [0], "inputs": ["nickname"]},
            ]
        )
        self.assertEqual(attr_result["values"], ["Al"])

    def test_relation_query_and_generic_search_shape(self):
        self.assertEqual(self.backend.relation_between(["Q1"], ["Q3"]), ["friend"])
        query = """PREFIX kqa: <urn:kqa:>
        SELECT DISTINCT ?relation WHERE { kqa:Q1 ?relation ?x . } LIMIT 1000"""
        rows = self.backend.execute_sparql(query)
        self.assertIn("friend", self.backend.parse_relations(rows))

    def test_protocol_builders_wrap_relation_and_entity_patterns(self):
        relation_query = self.backend.build_sparql_relations("kqa:Q1 ?relation ?x .")
        relation_rows = self.backend.execute_sparql(relation_query)
        self.assertIn("friend", self.backend.parse_relations(relation_rows))

        entity_query = self.backend.build_sparql_entities(
            f"kqa:Q1 {self.backend.format_relation_node('friend')} ?targetEntity ."
        )
        entity_rows = self.backend.execute_sparql(entity_query)
        _, entity_ids = self.backend.parse_entities(entity_rows)
        self.assertEqual(entity_ids, ["Q3"])

        multiword_query = self.backend.build_sparql_entities(
            f"kqa:Q1 {self.backend.format_relation_node('born in')} ?targetEntity ."
        )
        _, multiword_ids = self.backend.parse_entities(self.backend.execute_sparql(multiword_query))
        self.assertEqual(multiword_ids, ["Q4"])

    def test_bounded_multi_hop_query_uses_normalized_graph_semantics(self):
        query = self.backend.build_sparql_entities(
            "?person <urn:kqa:rel:friend> ?friend .\n"
            "?friend <urn:kqa:rel:born%20in> ?targetEntity ."
        )
        _, entity_ids = self.backend.parse_entities(self.backend.execute_sparql(query))
        self.assertEqual(entity_ids, ["Q4"])

    def test_unsupported_operator_is_rejected(self):
        with self.assertRaises(UnsupportedKQAOperator):
            self.backend.execute_program(
                [{"function": "FindAll", "dependencies": [], "inputs": []}, {"function": "Count", "dependencies": [0], "inputs": []}]
            )


if __name__ == "__main__":
    unittest.main()
