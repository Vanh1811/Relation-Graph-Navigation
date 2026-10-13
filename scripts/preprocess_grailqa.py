#!/usr/bin/env python3
"""Normalize official GrailQA train/dev/public-test files into JSONL."""

from __future__ import annotations

import argparse

from _bootstrap import PROJECT_ROOT  # noqa: E402
from relation_graph_kgqa.datasets.grailqa_loader import load_grailqa, write_grailqa_interim  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "datasets/grailqa/official"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data/interim"))
    args = parser.parse_args()

    examples, report = load_grailqa(args.input_dir)
    write_grailqa_interim(examples, args.output_dir, report)
    print(f"Normalized {len(examples)} GrailQA records from {args.input_dir}")
    print(f"Split counts: {report['split_counts']}")
    print(f"Generalization counts: {report['generalization_type_counts']}")
    if report["validation_error_count"]:
        raise SystemExit(f"Validation errors: {report['validation_error_count']}")


if __name__ == "__main__":
    main()
