#!/usr/bin/env python3
"""Compare the unified structural summaries for CWQ and GrailQA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: E402


LABELS = (
    "linear",
    "branch_2",
    "branch_3_plus",
    "nested_branch",
    "comparative",
    "superlative",
    "other_or_unknown",
)


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _percentage(count: int, denominator: int) -> str:
    return f"{(count / denominator):.2%}" if denominator else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwq", default=str(PROJECT_ROOT / "outputs/reports/cwq_structure_summary.json"))
    parser.add_argument("--grailqa", default=str(PROJECT_ROOT / "outputs/reports/grailqa_structure_summary.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs/reports/cwq_vs_grailqa_structure.md"))
    args = parser.parse_args()

    cwq = _load(Path(args.cwq))
    grailqa = _load(Path(args.grailqa))
    lines = [
        "# CWQ vs GrailQA structural comparison",
        "",
        "The same structural taxonomy is used for both datasets. GrailQA's public test split contains no public logical forms, so its structural percentages below use the parse-success denominator; the total source size is reported separately.",
        "",
        "| Dataset | Total records | Parse-success records | Average depth | Average branch count |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in (("CWQ", cwq), ("GrailQA", grailqa)):
        lines.append(
            f"| {name} | {summary['total_examples']} | {summary['parse_success_count']} | "
            f"{summary['average_max_depth']:.2f} | {summary['average_branch_count']:.2f} |"
        )
    lines.extend(["", "| Structure | CWQ count | CWQ % | GrailQA count | GrailQA % |", "| --- | ---: | ---: | ---: | ---: |"])
    for label in LABELS:
        cwq_count = cwq["structure_counts"].get(label, 0)
        grailqa_count = grailqa["structure_counts"].get(label, 0)
        lines.append(
            f"| {label} | {cwq_count} | {_percentage(cwq_count, cwq['parse_success_count'])} | "
            f"{grailqa_count} | {_percentage(grailqa_count, grailqa['parse_success_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Do not equate CWQ's source compositionality labels with GrailQA structural labels.",
            "- The structural taxonomy is derived from each dataset's logical representation.",
            "- GrailQA public-test records are retained for schema completeness but are not included in parse-success structural percentages.",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
