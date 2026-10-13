#!/usr/bin/env python3
"""Parse and classify normalized CWQ records using gold SPARQL offline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: E402
from relation_graph_kgqa.datasets.schemas import QAExample, ProcessedExample  # noqa: E402
from relation_graph_kgqa.sparql.graph_builder import build_query_graph  # noqa: E402
from relation_graph_kgqa.sparql.parser import parse_query  # noqa: E402
from relation_graph_kgqa.sparql.structure_classifier import (  # noqa: E402
    build_structural_metadata,
    classify_structure,
)
from relation_graph_kgqa.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402


def _qa_from_dict(record: dict) -> QAExample:
    return QAExample(
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


def classify_file(path: Path) -> tuple[list[dict], dict]:
    processed: list[dict] = []
    parser_failures = 0
    label_counts: dict[str, int] = {}
    for raw in read_jsonl(path):
        example = _qa_from_dict(raw)
        features = parse_query(example.gold_sparql or "")
        graph = build_query_graph(features, example)
        result = classify_structure(features, graph)
        metadata = build_structural_metadata(features, graph, result)
        record = ProcessedExample(example, metadata).to_analysis_dict()
        record["triple_patterns"] = [asdict(pattern) for pattern in features.triple_patterns]
        record["graph"] = {
            "nodes": sorted(graph.nodes),
            "variables": sorted(graph.variables),
            "constants": sorted(graph.constants),
            "edges": [asdict(edge) for edge in graph.edges],
            "ascii": graph.ascii_view(),
            "branch_candidates": graph.branch_candidates(),
        }
        processed.append(record)
        label_counts[metadata.structure_type] = label_counts.get(metadata.structure_type, 0) + 1
        parser_failures += int(not metadata.parse_success)
    return processed, {
        "source": str(path),
        "records": len(processed),
        "parser_failure_count": parser_failures,
        "structure_counts": dict(sorted(label_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "data/interim"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data/processed"))
    parser.add_argument("--report", default=str(PROJECT_ROOT / "outputs/reports/cwq_parse_report.json"))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    reports = []
    for split in ("train", "dev", "test"):
        input_path = input_dir / f"cwq_{split}.jsonl"
        output_path = output_dir / f"cwq_{split}_analysis.jsonl"
        if not input_path.exists():
            raise SystemExit(f"Missing input file: {input_path}. Run preprocess_cwq.py first.")
        records, report = classify_file(input_path)
        write_jsonl(output_path, records)
        reports.append(report)
        print(f"Classified {len(records)} records from {split}")

    write_json(args.report, {"splits": reports})


if __name__ == "__main__":
    main()
