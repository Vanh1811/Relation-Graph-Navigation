import json
import tempfile
import unittest
from pathlib import Path

from relation_graph_kgqa.datasets.grailqa_loader import load_grailqa


class GrailQALoaderTests(unittest.TestCase):
    def test_normalizes_official_fields_and_preserves_missing_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = {
                "qid": 1,
                "question": "which thing?",
                "answer": [{"answer_type": "Entity", "answer_argument": "m.a", "entity_name": "A"}],
                "domains": ["example"],
                "function": "none",
                "num_node": 2,
                "num_edge": 1,
                "graph_query": {
                    "nodes": [
                        {"nid": 0, "node_type": "class", "id": "example.thing", "question_node": 1},
                        {"nid": 1, "node_type": "entity", "id": "m.topic", "friendly_name": "Topic"},
                    ],
                    "edges": [],
                },
                "sparql_query": "SELECT * WHERE {}",
                "s_expression": "(AND example.thing (JOIN relation m.topic))",
            }
            (root / "grailqa_v1.0_train.json").write_text(json.dumps([base]))
            dev = dict(base)
            dev["qid"] = 2
            dev["level"] = "zero-shot"
            (root / "grailqa_v1.0_dev.json").write_text(json.dumps([dev]))
            (root / "grailqa_v1.0_test_public.json").write_text(
                json.dumps([{"qid": "test-0", "question": "hidden?"}])
            )

            examples, report = load_grailqa(root)

        self.assertEqual(len(examples), 3)
        self.assertEqual(report["split_counts"], {"dev": 1, "test": 1, "train": 1})
        train = next(item for item in examples if item.split == "train")
        dev_example = next(item for item in examples if item.split == "dev")
        test = next(item for item in examples if item.split == "test")
        self.assertEqual(train.logical_form, base["s_expression"])
        self.assertIsNone(train.generalization_type)
        self.assertEqual(dev_example.generalization_type, "zero-shot")
        self.assertEqual(train.topic_entities, ["m.topic"])
        self.assertIsNone(test.logical_form)
        self.assertEqual(report["examples_without_logical_form"], 1)
