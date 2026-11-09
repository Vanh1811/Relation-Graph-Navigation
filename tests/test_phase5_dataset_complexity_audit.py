import json
import unittest
from pathlib import Path

from scripts.run_phase5_dataset_complexity_audit import (
    KQA_FAMILIES,
    depth_distribution,
    kqa_depth,
    kqa_family,
    percentile,
    sample_kqa,
    sparql_triples,
    sparql_operator_counts,
)


class Phase5AuditTests(unittest.TestCase):
    def test_percentile_and_depth_distribution(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertEqual(depth_distribution([1, 2, 5, 6])["counts"], {"1": 1, "2": 1, "5+": 2})

    def test_sparql_triple_extraction_skips_control_lines(self):
        query = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x WHERE {\nns:m.a ns:foo ?x .\n?x ns:bar ns:m.b .\nFILTER (?x != ns:m.a)\n}"""
        self.assertEqual(len(sparql_triples(query)), 2)

    def test_kqa_family_mapping(self):
        self.assertEqual(kqa_family("What"), "QueryName")
        self.assertEqual(kqa_family("QueryAttrUnderCondition"), "QueryAttr")
        self.assertEqual(kqa_family("VerifyDate"), "Verify")
        self.assertEqual(len(KQA_FAMILIES), 9)

    def test_kqa_dependency_depth(self):
        program = [
            {"function": "Find", "dependencies": []},
            {"function": "Relate", "dependencies": [0]},
            {"function": "QueryAttr", "dependencies": [1]},
        ]
        self.assertEqual(kqa_depth(program), 3)

    def test_entity_inequality_is_not_numeric_filter(self):
        query = """SELECT ?x WHERE {
        ?x <http://rdf.freebase.com/ns/type.object.type> <http://rdf.freebase.com/ns/common.topic> .
        FILTER (?x != <http://rdf.freebase.com/ns/m.01>)
        }"""
        triples = sparql_triples(query)
        counts = sparql_operator_counts(query, triples, branch_count=1)
        self.assertNotIn("COMPARISON", counts)

    def test_sampling_is_deterministic_and_balanced(self):
        rows = []
        terminal_functions = {
            "Count": "Count",
            "QueryAttr": "QueryAttr",
            "QueryAttrQualifier": "QueryAttrQualifier",
            "QueryName": "What",
            "QueryRelation": "QueryRelation",
            "QueryRelationQualifier": "QueryRelationQualifier",
            "SelectAmong": "SelectAmong",
            "SelectBetween": "SelectBetween",
            "Verify": "VerifyStr",
        }
        for family_index, function in enumerate(terminal_functions.values()):
            for offset in range(25):
                rows.append(
                    {
                        "question": f"q-{family_index}-{offset}",
                        "program": [
                            {"function": "Find", "dependencies": []},
                            {"function": function, "dependencies": [0]},
                        ],
                    }
                )
        first, first_manifest = sample_kqa(rows)
        second, second_manifest = sample_kqa(rows)
        self.assertEqual([row["sample_index"] for row in first], [row["sample_index"] for row in second])
        self.assertEqual(first_manifest["selected_source_indices"], second_manifest["selected_source_indices"])
        self.assertEqual(first_manifest["sample_size"], 180)
        self.assertEqual(first_manifest["family_counts"], {family: 20 for family in KQA_FAMILIES})

    def test_phase3c_manifest_is_exactly_60_unique_cases(self):
        path = Path("outputs/experiments/phase3c/frozen_branch_required_generalization/phase3c_eval_manifest.json")
        manifest = json.loads(path.read_text())
        ids = [row["case_id"] for row in manifest["strong_cases"]]
        self.assertEqual(len(ids), 60)
        self.assertEqual(len(set(ids)), 60)


if __name__ == "__main__":
    unittest.main()
