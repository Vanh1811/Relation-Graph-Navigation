from __future__ import annotations

import unittest

from scripts.diagnostics.iter_027_results_first_answer_repair.provenance import (
    observed_relation_provenance,
    relation_ids_from_executed_entity_search,
)


class LiveProvenanceExtractionTests(unittest.TestCase):
    def test_extracts_predicate_from_executed_candidate_query(self):
        query = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?targetEntity WHERE {
  ?targetEntity ns:food.cheese.country_of_origin ns:m.07ssc .
} LIMIT 1000"""
        self.assertEqual(
            relation_ids_from_executed_entity_search(query),
            ["food.cheese.country_of_origin"],
        )

    def test_ignores_prefix_and_entity_terms_as_predicates(self):
        query = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?x WHERE { ns:m.topic ?relation ?x . }"""
        self.assertEqual(relation_ids_from_executed_entity_search(query), [])

    def test_relation_provenance_is_limited_to_pre_filter_entity_searches(self):
        events = [
            {"search_type": "entity_search", "trace_id": 4, "sparql": "?s ns:book.author ?o .", "result_count": 2, "status": "ok"},
            {"search_type": "entity_search", "trace_id": 8, "sparql": "?s ns:music.artist ?o .", "result_count": 1, "status": "ok"},
            {"search_type": "relation_search_head", "trace_id": 3, "sparql": "?s ?p ?o .", "result_count": 4, "status": "ok"},
        ]
        result = observed_relation_provenance(events, filter_trace_id=7)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["observed_relation_ids_in_executed_query"], ["book.author"])


if __name__ == "__main__":
    unittest.main()
