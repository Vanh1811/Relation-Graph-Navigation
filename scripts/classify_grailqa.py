#!/usr/bin/env python3
"""Parse GrailQA S-expressions and emit structural analysis artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: E402
from relation_graph_kgqa.analysis.grailqa import classify_grailqa_example, logical_graph_dict  # noqa: E402
from relation_graph_kgqa.datasets.schemas import QAExample  # noqa: E402
from relation_graph_kgqa.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402


SPLITS = ("train", "dev", "test")


def _inventory(records: list[dict]) -> dict[str, dict]:
    inventory: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "example_logical_forms": [], "example_questions": []}
    )
    for record in records:
        structure = record.get("structure", {})
        for operator in structure.get("operators", []):
            item = inventory[operator]
            item["count"] += 1
            if len(item["example_logical_forms"]) < 5 and record.get("logical_form"):
                item["example_logical_forms"].append(record["logical_form"])
            if len(item["example_questions"]) < 5:
                item["example_questions"].append(record.get("question", ""))
    return dict(sorted(inventory.items()))


def _inventory_markdown(inventory: dict[str, dict]) -> str:
    lines = [
        "# GrailQA operator inventory",
        "",
        "Counts are the number of parsed records containing each operator. Example logical forms and questions are capped at five per operator.",
        "",
        "| Operator | Records |",
        "| --- | ---: |",
    ]
    for operator, item in inventory.items():
        lines.append(f"| `{operator}` | {item['count']} |")
    for operator, item in inventory.items():
        lines.extend(["", f"## `{operator}`", "", "### Example logical forms", ""])
        for logical_form in item["example_logical_forms"]:
            lines.extend(["```text", logical_form, "```", ""])
        lines.extend(["### Example questions", ""])
        lines.extend(f"- {question}" for question in item["example_questions"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "data/interim"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data/processed"))
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "outputs/reports"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_records: list[dict] = []
    for split in SPLITS:
        source = Path(args.input_dir) / f"grailqa_{split}.jsonl"
        if not source.exists():
            continue
        output: list[dict] = []
        for raw in read_jsonl(source):
            example = QAExample(**raw)
            processed, graph = classify_grailqa_example(example)
            record = processed.to_analysis_dict()
            record["logical_graph"] = logical_graph_dict(graph)
            output.append(record)
        write_jsonl(output_dir / f"grailqa_{split}_analysis.jsonl", output)
        all_records.extend(output)
        print(f"Classified {len(output)} {split} records")

    inventory = _inventory(all_records)
    report_dir = Path(args.report_dir)
    write_json(report_dir / "grailqa_operator_inventory.json", inventory)
    (report_dir / "grailqa_operator_inventory.md").parent.mkdir(parents=True, exist_ok=True)
    (report_dir / "grailqa_operator_inventory.md").write_text(
        _inventory_markdown(inventory), encoding="utf-8"
    )
    print(f"Wrote operator inventory for {len(inventory)} operators")


if __name__ == "__main__":
    main()
