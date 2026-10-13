import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relation_graph_kgqa.datasets.schemas import QAExample, ProcessedExample, StructuralMetadata  # noqa: E402


class PipelineIntegrityTests(unittest.TestCase):
    def test_inference_record_excludes_gold_sparql(self):
        qa = QAExample(
            id="example",
            question="question",
            answers=["answer"],
            gold_sparql="SELECT ?x WHERE {}",
            dataset_type="composition",
            topic_entities=["m.topic"],
            split="train",
        )
        metadata = StructuralMetadata(
            structure_type="linear",
            branch_count=0,
            max_depth=1,
            variable_count=1,
            shared_variable_count=0,
            has_comparison=False,
            has_superlative=False,
            has_union=False,
            parse_success=True,
        )
        analysis = ProcessedExample(qa, metadata).to_analysis_dict()
        inference = ProcessedExample(qa, metadata).to_inference_dict()
        self.assertEqual(analysis["gold_sparql"], qa.gold_sparql)
        self.assertNotIn("gold_sparql", inference)

    def test_generated_inference_subsets_do_not_contain_gold_sparql(self):
        subset_dir = Path(__file__).resolve().parents[1] / "outputs/subsets"
        files = sorted(subset_dir.glob("*.jsonl"))
        if not files:
            self.skipTest("Run export_subsets.py before checking generated artifacts")
        for path in files:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.assertNotIn("gold_sparql", json.loads(line))

    def test_structural_subsets_are_mutually_exclusive(self):
        subset_dir = Path(__file__).resolve().parents[1] / "outputs/subsets"
        labels = ["linear", "branch_2", "branch_3_plus", "nested_branch"]
        ids_by_label = {}
        for label in labels:
            path = subset_dir / f"cwq_{label}.jsonl"
            if not path.exists():
                self.skipTest("Run export_subsets.py before checking generated artifacts")
            ids_by_label[label] = {
                json.loads(line)["id"]
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        for index, label in enumerate(labels):
            for other in labels[index + 1 :]:
                self.assertTrue(ids_by_label[label].isdisjoint(ids_by_label[other]))

    def test_branch_count_matches_subset_labels(self):
        subset_dir = Path(__file__).resolve().parents[1] / "outputs/subsets"
        expectations = {
            "branch_2": lambda value: value == 2,
            "branch_3_plus": lambda value: value >= 3,
            "nested_branch": lambda value: value >= 2,
        }
        for label, predicate in expectations.items():
            path = subset_dir / f"cwq_{label}.jsonl"
            if not path.exists():
                self.skipTest("Run export_subsets.py before checking generated artifacts")
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.assertTrue(predicate(json.loads(line)["branch_count"]))


if __name__ == "__main__":
    unittest.main()
