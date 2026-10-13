"""Deterministic summary statistics for structural classification."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Iterable, Mapping


STRUCTURE_LABELS = [
    "linear",
    "branch_2",
    "branch_3_plus",
    "nested_branch",
    "comparative",
    "superlative",
    "other_or_unknown",
]


def summarize_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    label_counts = Counter(str(row.get("structure", {}).get("structure_type", "other_or_unknown")) for row in rows)
    split_counts = Counter(str(row.get("split", "unknown")) for row in rows)
    original_counts = Counter(str(row.get("dataset_type") or "<missing>") for row in rows)
    cross_tab: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        original = str(row.get("dataset_type") or "<missing>")
        label = str(row.get("structure", {}).get("structure_type", "other_or_unknown"))
        cross_tab[original][label] += 1

    parse_success = [bool(row.get("structure", {}).get("parse_success")) for row in rows]
    depths = [int(row.get("structure", {}).get("max_depth", 0)) for row in rows]
    branches = [int(row.get("structure", {}).get("branch_count", 0)) for row in rows]
    parser_errors = Counter(
        error
        for row in rows
        for error in row.get("structure", {}).get("parser_errors", [])
    )

    return {
        "total_examples": len(rows),
        "parse_success_count": sum(parse_success),
        "parse_failure_count": len(rows) - sum(parse_success),
        "parse_success_rate": (sum(parse_success) / len(rows)) if rows else 0.0,
        "structure_counts": {label: label_counts.get(label, 0) for label in STRUCTURE_LABELS},
        "split_counts": dict(sorted(split_counts.items())),
        "original_compositionality_counts": dict(sorted(original_counts.items())),
        "original_type_by_structure": {
            original: {label: counts.get(label, 0) for label in STRUCTURE_LABELS}
            for original, counts in sorted(cross_tab.items())
        },
        "average_max_depth": mean(depths) if depths else 0.0,
        "maximum_max_depth": max(depths, default=0),
        "average_branch_count": mean(branches) if branches else 0.0,
        "parser_error_counts": dict(parser_errors.most_common()),
    }


def summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# CWQ structural summary",
        "",
        f"- Total examples: {summary['total_examples']}",
        f"- Parse success: {summary['parse_success_count']} ({summary['parse_success_rate']:.2%})",
        f"- Parse failures: {summary['parse_failure_count']}",
        f"- Average max depth: {summary['average_max_depth']:.2f}",
        f"- Maximum max depth: {summary['maximum_max_depth']}",
        f"- Average branch count: {summary['average_branch_count']:.2f}",
        "",
        "## Structure counts",
        "",
        "| Structure | Count |",
        "| --- | ---: |",
    ]
    for label, count in summary["structure_counts"].items():
        lines.append(f"| {label} | {count} |")

    lines.extend(["", "## Source splits", "", "| Split | Count |", "| --- | ---: |"])
    for split, count in summary["split_counts"].items():
        lines.append(f"| {split} | {count} |")

    lines.extend(
        [
            "",
            "## Original compositionality type × structural type",
            "",
            "| Original type | " + " | ".join(STRUCTURE_LABELS) + " |",
            "| --- | " + " | ".join("---:" for _ in STRUCTURE_LABELS) + " |",
        ]
    )
    for original, row in summary["original_type_by_structure"].items():
        lines.append("| " + original + " | " + " | ".join(str(row[label]) for label in STRUCTURE_LABELS) + " |")

    if summary["parser_error_counts"]:
        lines.extend(["", "## Parser errors", ""])
        for error, count in summary["parser_error_counts"].items():
            lines.append(f"- `{error}`: {count}")
    return "\n".join(lines) + "\n"
