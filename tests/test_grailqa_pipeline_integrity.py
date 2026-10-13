import json
import unittest
from pathlib import Path

from relation_graph_kgqa.utils.io import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


class GrailQAPipelineIntegrityTests(unittest.TestCase):
    def _analysis_records(self) -> list[dict]:
        records = []
        for split in ("train", "dev", "test"):
            path = ROOT / "data/processed" / f"grailqa_{split}_analysis.jsonl"
            if path.exists():
                records.extend(read_jsonl(path))
        return records

    def test_analysis_records_include_logical_forms_only_when_source_has_them(self) -> None:
        records = self._analysis_records()
        self.assertGreater(len(records), 0)
        parsed = [record for record in records if record["structure"]["parse_success"]]
        missing = [record for record in records if not record["structure"]["parse_success"]]
        self.assertGreater(len(parsed), 0)
        self.assertTrue(all(record.get("logical_form") for record in parsed))
        self.assertTrue(all("gold S-expression is missing" in record["structure"]["parser_errors"] for record in missing))

    def test_inference_subsets_exclude_gold_logical_form_and_sparql(self) -> None:
        subset_dir = ROOT / "outputs/subsets"
        for path in subset_dir.glob("grailqa_*.jsonl"):
            for record in read_jsonl(path):
                self.assertNotIn("logical_form", record, path.name)
                self.assertNotIn("gold_sparql", record, path.name)
                self.assertNotIn("source_metadata", record, path.name)

    def test_primary_structural_subsets_are_disjoint_and_complete_for_labeled_rows(self) -> None:
        labels = ("linear", "branch_2", "branch_3_plus", "nested_branch", "comparative", "superlative")
        paths = [ROOT / "outputs/subsets" / f"grailqa_{label}.jsonl" for label in labels]
        id_sets = [{record["id"] for record in read_jsonl(path)} for path in paths]
        for index, current in enumerate(id_sets):
            for other in id_sets[index + 1 :]:
                self.assertTrue(current.isdisjoint(other))
        labeled = {
            record["id"]
            for record in self._analysis_records()
            if record["structure"]["structure_type"] in labels
        }
        self.assertEqual(set().union(*id_sets), labeled)

    def test_summary_counts_match_analysis_rows(self) -> None:
        summary = json.loads((ROOT / "outputs/reports/grailqa_structure_summary.json").read_text())
        self.assertEqual(summary["total_examples"], len(self._analysis_records()))
        self.assertEqual(sum(summary["structure_counts"].values()), summary["total_examples"])
