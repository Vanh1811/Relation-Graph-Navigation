import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relation_graph_kgqa.datasets.schemas import QAExample  # noqa: E402
from relation_graph_kgqa.sparql.graph_builder import build_query_graph  # noqa: E402
from relation_graph_kgqa.sparql.parser import parse_query  # noqa: E402


class GraphBuilderTests(unittest.TestCase):
    def _example(self):
        return QAExample(
            id="example",
            question="Which person satisfies both conditions?",
            answers=["Person"],
            answer_ids=["m.answer"],
            gold_sparql=None,
            dataset_type="conjunction",
            topic_entities=["m.topic"],
            split="train",
        )

    def test_detects_two_arms_at_shared_variable(self):
        query = """
        SELECT ?x WHERE {
          ns:m.topic ns:relation.parent ?x .
          ?x ns:relation.birthplace ns:m.place .
          ?x ns:relation.profession ns:m.profession .
        }
        """
        features = parse_query(query)
        graph = build_query_graph(features, self._example())
        self.assertEqual(graph.branch_candidates().get("?x"), 2)
        self.assertEqual(graph.max_reasoning_depth(), 1)
        self.assertEqual(len(graph.connected_components()), 1)


if __name__ == "__main__":
    unittest.main()
