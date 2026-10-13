import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relation_graph_kgqa.sparql.parser import parse_query  # noqa: E402


class SparqlParserTests(unittest.TestCase):
    def test_extracts_linear_triples_and_select_variable(self):
        query = """
        SELECT DISTINCT ?x WHERE {
          ns:m.topic ns:people.person.parents ?parent .
          ?parent ns:people.person.child ?x .
        }
        """
        features = parse_query(query)
        self.assertTrue(features.parse_success)
        self.assertEqual(features.select_variables, ["?x"])
        self.assertEqual(len(features.triple_patterns), 2)
        self.assertFalse(features.has_comparison)

    def test_ignores_generic_anti_cycle_and_language_filters(self):
        query = """
        SELECT ?x WHERE {
          FILTER (?x != ns:m.topic)
          FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))
          ns:m.topic ns:location.contains ?x .
        }
        """
        features = parse_query(query)
        self.assertTrue(features.parse_success)
        self.assertFalse(features.has_comparison)

    def test_detects_ordered_comparison_and_typed_literal(self):
        query = """
        SELECT ?x WHERE {
          ns:m.topic ns:people.person.birthdate ?date .
          FILTER (xsd:datetime(?date) <= "2015-08-10"^^xsd:dateTime)
          ns:m.topic ns:people.person.profession ?x .
        }
        """
        features = parse_query(query)
        self.assertTrue(features.parse_success)
        self.assertTrue(features.has_comparison)
        self.assertEqual(len(features.triple_patterns), 2)
        self.assertIn('"2015-08-10"', features.filters[0])
        self.assertIn("^^xsd:dateTime", features.filters[0])

    def test_handles_missing_whitespace_between_triples(self):
        query = """
        SELECT ?x WHERE {
          ns:m.topic ns:people.person.parents ?y .?y ns:people.person.child ?x .
        }
        """
        features = parse_query(query)
        self.assertTrue(features.parse_success)
        self.assertEqual(len(features.triple_patterns), 2)


if __name__ == "__main__":
    unittest.main()
