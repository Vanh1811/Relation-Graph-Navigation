#!/usr/bin/env python3
"""Export manually inspectable GrailQA structural examples."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: E402
from relation_graph_kgqa.utils.io import read_jsonl  # noqa: E402


TARGETS = {
    "linear": 50,
    "branch_2": 50,
    "branch_3_plus": 30,
    "nested_branch": 30,
    "comparative": 20,
    "superlative": 20,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "data/processed"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs/examples/grailqa_manual_validation.md"))
    parser.add_argument(
        "--report-output",
        default=str(PROJECT_ROOT / "outputs/reports/grailqa_structure_examples.md"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    for split in ("train", "dev", "test"):
        path = Path(args.input_dir) / f"grailqa_{split}_analysis.jsonl"
        if path.exists():
            records.extend(read_jsonl(path))

    rng = random.Random(args.seed)
    by_label: dict[str, list[dict]] = {}
    for record in records:
        label = record.get("structure", {}).get("structure_type", "other_or_unknown")
        by_label.setdefault(label, []).append(record)

    lines = [
        "# GrailQA manual structural validation",
        "",
        f"Random seed: `{args.seed}`",
        "",
        "These examples are for human inspection. Gold logical forms and SPARQL are retained because this is an analysis artifact, not an inference input.",
        "",
    ]
    for label, count in TARGETS.items():
        candidates = list(by_label.get(label, []))
        rng.shuffle(candidates)
        selected = candidates[:count]
        lines.extend([f"## {label} ({len(selected)} examples)", ""])
        if not selected:
            lines.extend(["No examples available.", ""])
            continue
        for index, record in enumerate(selected, start=1):
            structure = record.get("structure", {})
            lines.extend(
                [
                    f"### {index}. {record['id']}",
                    "",
                    f"- Question: {record['question']}",
                    f"- Split: {record.get('split')}",
                    f"- Generalization type: {record.get('generalization_type') or '<missing>'}",
                    f"- Predicted structure: {structure.get('structure_type')}",
                    f"- Branch count: {structure.get('branch_count')}",
                    f"- Max depth: {structure.get('max_depth')}",
                    f"- Nested branch count: {structure.get('nested_branch_count')}",
                    f"- Operators: {', '.join(structure.get('operators', []))}",
                    f"- Reasons: {'; '.join(structure.get('classifier_reasons', []))}",
                    "",
                    "```text",
                    str(record.get("logical_form", "")),
                    "```",
                    "",
                    "```sparql",
                    str(record.get("gold_sparql", "")),
                    "```",
                    "",
                    "```text",
                    record.get("logical_graph", {}).get("ascii", "(no logical-form graph)"),
                    "```",
                    "",
                ]
            )

    rendered = "\n".join(lines)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    report_output = Path(args.report_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Wrote {report_output}")


if __name__ == "__main__":
    main()
