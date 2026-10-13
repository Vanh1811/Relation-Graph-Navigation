#!/usr/bin/env python3
"""Export GrailQA structural and generalization inference-safe subsets."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: E402
from relation_graph_kgqa.utils.io import read_jsonl, write_jsonl  # noqa: E402


STRUCTURAL_LABELS = (
    "linear",
    "branch_2",
    "branch_3_plus",
    "nested_branch",
    "comparative",
    "superlative",
)
GENERALIZATION_FILES = {
    "i.i.d.": "iid",
    "compositional": "compositional",
    "zero-shot": "zero_shot",
}


def _safe_record(record: dict) -> dict:
    structure = record.get("structure", {})
    raw_generalization = record.get("generalization_type")
    return {
        "id": record.get("id"),
        "question": record.get("question"),
        "answers": list(record.get("answers", [])),
        "answer_ids": list(record.get("answer_ids", [])),
        "topic_entities": list(record.get("topic_entities", [])),
        "topic_entity_names": list(record.get("topic_entity_names", [])),
        "split": record.get("split"),
        "generalization_type": raw_generalization,
        "generalization_type_normalized": GENERALIZATION_FILES.get(raw_generalization),
        "structure_type": structure.get("structure_type", "other_or_unknown"),
        "branch_count": structure.get("branch_count", 0),
        "max_depth": structure.get("max_depth", 0),
        "nested_branch_count": structure.get("nested_branch_count", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "data/processed"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs/subsets"))
    args = parser.parse_args()

    records = []
    for split in ("train", "dev", "test"):
        path = Path(args.input_dir) / f"grailqa_{split}_analysis.jsonl"
        if path.exists():
            records.extend(read_jsonl(path))
    safe = [_safe_record(record) for record in records]
    output_dir = Path(args.output_dir)

    for label in STRUCTURAL_LABELS:
        rows = [record for record in safe if record["structure_type"] == label]
        write_jsonl(output_dir / f"grailqa_{label}.jsonl", rows)

    branching = [record for record in safe if record["structure_type"] in {"branch_2", "branch_3_plus", "nested_branch"}]
    write_jsonl(output_dir / "grailqa_branching_all.jsonl", branching)

    for raw_label, filename_label in GENERALIZATION_FILES.items():
        rows = [record for record in safe if record["generalization_type"] == raw_label]
        write_jsonl(output_dir / f"grailqa_{filename_label}.jsonl", rows)
        branching_rows = [record for record in rows if record in branching]
        write_jsonl(output_dir / f"grailqa_branching_{filename_label}.jsonl", branching_rows)

    print(f"Exported inference-safe subsets from {len(safe)} records to {output_dir}")


if __name__ == "__main__":
    main()
