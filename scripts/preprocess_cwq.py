#!/usr/bin/env python3
"""Normalize the repository's CWQ JSON into deterministic split JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: E402
from relation_graph_kgqa.datasets.cwq_loader import load_cwq, write_cwq_interim  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(PROJECT_ROOT / "datasets/cwq/cwq.json"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data/interim"))
    args = parser.parse_args()

    examples, report = load_cwq(args.input)
    write_cwq_interim(examples, args.output_dir, report)
    print(f"Normalized {len(examples)} CWQ records from {args.input}")
    print(f"Split counts: {report['split_counts']}")
    if report["missing_expected_splits"]:
        print(f"Missing source splits: {report['missing_expected_splits']}")
    if report["validation_error_count"]:
        raise SystemExit(f"Validation errors: {report['validation_error_count']}")


if __name__ == "__main__":
    main()
