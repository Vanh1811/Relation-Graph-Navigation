#!/usr/bin/env python3
"""Export inference-safe CWQ structural subsets without gold SPARQL."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: E402
from relation_graph_kgqa.datasets.schemas import QAExample, ProcessedExample, StructuralMetadata  # noqa: E402
from relation_graph_kgqa.utils.io import read_jsonl, write_jsonl  # noqa: E402


def _processed_from_dict(record: dict) -> ProcessedExample:
    qa = QAExample(
        id=str(record["id"]),
        question=str(record["question"]),
        answers=list(record.get("answers", [])),
        answer_ids=list(record.get("answer_ids", [])),
        answer_entities=list(record.get("answer_entities", [])),
        gold_sparql=record.get("gold_sparql"),
        dataset_type=record.get("dataset_type"),
        topic_entities=list(record.get("topic_entities", [])),
        topic_entity_names=list(record.get("topic_entity_names", [])),
        split=str(record.get("split", "unknown")),
    )
    structure = StructuralMetadata(**record["structure"])
    return ProcessedExample(qa, structure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "data/processed"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs/subsets"))
    args = parser.parse_args()

    records = []
    for split in ("train", "dev", "test"):
        path = Path(args.input_dir) / f"cwq_{split}_analysis.jsonl"
        if path.exists():
            records.extend(_processed_from_dict(row) for row in read_jsonl(path))

    labels = {
        "cwq_linear": {"linear"},
        "cwq_branch_2": {"branch_2"},
        "cwq_branch_3_plus": {"branch_3_plus"},
        "cwq_nested_branch": {"nested_branch"},
        "cwq_branching_all": {"branch_2", "branch_3_plus", "nested_branch"},
    }
    output_dir = Path(args.output_dir)
    for filename, accepted in labels.items():
        selected = [item.to_inference_dict() for item in records if item.structure.structure_type in accepted]
        write_jsonl(output_dir / f"{filename}.jsonl", selected)
        print(f"{filename}: {len(selected)}")


if __name__ == "__main__":
    main()
