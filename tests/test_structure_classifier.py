import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relation_graph_kgqa.datasets.schemas import QAExample  # noqa: E402
from relation_graph_kgqa.sparql.structure_classifier import classify_example  # noqa: E402


def make_example(query: str, topic: str = "m.topic") -> QAExample:
    return QAExample(
        id="example",
        question="example",
        answers=["answer"],
        answer_ids=["m.answer"],
        gold_sparql=query,
        dataset_type="conjunction",
        topic_entities=[topic],
        split="train",
    )


class StructureClassifierTests(unittest.TestCase):
    def test_linear(self):
        query = """
        SELECT ?x WHERE {
          ns:m.topic ns:relation.first ?y .
          ?y ns:relation.second ?x .
        }
        """
        metadata, _, _ = classify_example(make_example(query))
        self.assertEqual(metadata.structure_type, "linear")

    def test_branch_2(self):
        query = """
        SELECT ?x WHERE {
          ns:m.topic ns:relation.first ?x .
          ?x ns:relation.second ns:m.condition_a .
          ?x ns:relation.third ns:m.condition_b .
        }
        """
        metadata, _, _ = classify_example(make_example(query))
        self.assertEqual(metadata.structure_type, "branch_2")
        self.assertEqual(metadata.branch_count, 2)

    def test_comparative_and_superlative(self):
        comparative = """
        SELECT ?x WHERE {
          ns:m.topic ns:relation.value ?value .
          ns:m.topic ns:relation.answer ?x .
          FILTER (?value > 10)
        }
        """
        superlative = """
        SELECT ?x WHERE {
          ns:m.topic ns:relation.answer ?x .
          ?x ns:relation.score ?score .
        }
        ORDER BY DESC(?score)
        LIMIT 1
        """
        comparative_metadata, _, _ = classify_example(make_example(comparative))
        superlative_metadata, _, _ = classify_example(make_example(superlative))
        self.assertEqual(comparative_metadata.structure_type, "comparative")
        self.assertEqual(superlative_metadata.structure_type, "superlative")


if __name__ == "__main__":
    unittest.main()
