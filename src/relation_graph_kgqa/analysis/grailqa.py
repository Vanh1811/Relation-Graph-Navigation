"""GrailQA-specific structural analysis and report helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Iterable, Mapping

from relation_graph_kgqa.datasets.schemas import ProcessedExample, QAExample, StructuralMetadata
from relation_graph_kgqa.logical_forms.graph_builder import LogicalFormGraph
from relation_graph_kgqa.logical_forms.structure_classifier import analyze_s_expression


STRUCTURE_LABELS = [
    "linear",
    "branch_2",
    "branch_3_plus",
    "nested_branch",
    "comparative",
    "superlative",
    "other_or_unknown",
]


def classify_grailqa_example(example: QAExample) -> tuple[ProcessedExample, LogicalFormGraph | None]:
    logical, graph = analyze_s_expression(example.logical_form)
    metadata = StructuralMetadata(
        structure_type=logical.structure_type,
        branch_count=logical.branch_count,
        max_depth=logical.max_depth,
        variable_count=0,
        shared_variable_count=0,
        has_comparison=logical.has_comparison,
        has_superlative=logical.has_argmax or logical.has_argmin,
        has_union=logical.has_or,
        parse_success=logical.parse_success,
        answer_variables=[],
        triple_count=0,
        connected_components=0,
        has_filter=False,
        has_aggregation=logical.has_count,
        has_order_by=False,
        has_limit=False,
        nested_branch=logical.nested_branch_count > 0,
        classifier_confidence=logical.classifier_confidence,
        classifier_reasons=list(logical.classifier_reasons),
        parser_errors=list(logical.parser_errors),
        unsupported_constructs=list(logical.unsupported_constructs),
        operators=list(logical.operators),
        nested_branch_count=logical.nested_branch_count,
        max_branch_depth=logical.max_branch_depth,
        has_and=logical.has_and,
        has_or=logical.has_or,
        has_argmax=logical.has_argmax,
        has_argmin=logical.has_argmin,
        has_count=logical.has_count,
    )
    return ProcessedExample(example, metadata), graph


def logical_graph_dict(graph: LogicalFormGraph | None) -> dict[str, Any]:
    if graph is None:
        return {"nodes": [], "edges": [], "ascii": "(no logical-form graph)"}
    return {
        "nodes": [
            {"id": node.node_id, "label": node.label, "kind": node.kind}
            for node in graph.nodes
        ],
        "edges": [{"source": edge.source, "target": edge.target} for edge in graph.edges],
        "ascii": graph.ascii,
    }


def summarize_grailqa_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    label_counts = Counter(
        str(row.get("structure", {}).get("structure_type", "other_or_unknown")) for row in rows
    )
    split_counts = Counter(str(row.get("split", "unknown")) for row in rows)
    generalization_counts = Counter(str(row.get("generalization_type") or "<missing>") for row in rows)
    cross_tab: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        generalization = str(row.get("generalization_type") or "<missing>")
        label = str(row.get("structure", {}).get("structure_type", "other_or_unknown"))
        cross_tab[generalization][label] += 1

    parse_success = [bool(row.get("structure", {}).get("parse_success")) for row in rows]
    depths = [int(row.get("structure", {}).get("max_depth", 0)) for row in rows]
    branches = [int(row.get("structure", {}).get("branch_count", 0)) for row in rows]
    parser_errors = Counter(
        error for row in rows for error in row.get("structure", {}).get("parser_errors", [])
    )
    operator_counts = Counter(
        operator for row in rows for operator in row.get("structure", {}).get("operators", [])
    )
    return {
        "total_examples": len(rows),
        "parse_success_count": sum(parse_success),
        "parse_failure_count": len(rows) - sum(parse_success),
        "parse_success_rate": (sum(parse_success) / len(rows)) if rows else 0.0,
        "structure_counts": {label: label_counts.get(label, 0) for label in STRUCTURE_LABELS},
        "split_counts": dict(sorted(split_counts.items())),
        "generalization_type_counts": dict(sorted(generalization_counts.items())),
        "generalization_by_structure": {
            generalization: {label: counts.get(label, 0) for label in STRUCTURE_LABELS}
            for generalization, counts in sorted(cross_tab.items())
        },
        "average_max_depth": mean(depths) if depths else 0.0,
        "maximum_max_depth": max(depths, default=0),
        "average_branch_count": mean(branches) if branches else 0.0,
        "operator_counts": dict(sorted(operator_counts.items())),
        "parser_error_counts": dict(parser_errors.most_common()),
    }


def summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# GrailQA structural summary",
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
    lines.extend(f"| {label} | {summary['structure_counts'][label]} |" for label in STRUCTURE_LABELS)
    lines.extend(["", "## Source splits", "", "| Split | Count |", "| --- | ---: |"])
    lines.extend(f"| {split} | {count} |" for split, count in summary["split_counts"].items())
    lines.extend(["", "## Generalization values", "", "| Value | Count |", "| --- | ---: |"])
    lines.extend(
        f"| {label} | {count} |" for label, count in summary["generalization_type_counts"].items()
    )

    lines.extend(["", "## Generalization × structure", ""])
    lines.append("| Generalization | " + " | ".join(STRUCTURE_LABELS) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in STRUCTURE_LABELS) + " |")
    for generalization, row in summary["generalization_by_structure"].items():
        lines.append(
            "| " + generalization + " | " + " | ".join(str(row[label]) for label in STRUCTURE_LABELS) + " |"
        )
    if summary["parser_error_counts"]:
        lines.extend(["", "## Parser errors", ""])
        lines.extend(
            f"- `{error}`: {count}" for error, count in summary["parser_error_counts"].items()
        )
    return "\n".join(lines) + "\n"


def generalization_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# GrailQA structure by generalization",
        "",
        "Counts and row percentages are calculated within each observed generalization value.",
        "Missing labels are retained as `<missing>` rather than inferred.",
        "",
    ]
    for generalization, row in summary["generalization_by_structure"].items():
        total = sum(row.values())
        lines.extend([f"## {generalization} (n={total})", ""])
        lines.extend(["| Structure | Count | Within-group percentage |", "| --- | ---: | ---: |"])
        for label in STRUCTURE_LABELS:
            count = row[label]
            percentage = (count / total) if total else 0.0
            lines.append(f"| {label} | {count} | {percentage:.2%} |")
        lines.append("")
    return "\n".join(lines)
