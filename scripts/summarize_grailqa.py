#!/usr/bin/env python3
"""Generate GrailQA structural and generalization reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: E402
from relation_graph_kgqa.analysis.grailqa import (  # noqa: E402
    generalization_markdown,
    summarize_grailqa_records,
    summary_markdown,
)
from relation_graph_kgqa.utils.io import read_jsonl, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "data/processed"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs/reports"))
    args = parser.parse_args()

    records = []
    for split in ("train", "dev", "test"):
        path = Path(args.input_dir) / f"grailqa_{split}_analysis.jsonl"
        if path.exists():
            records.extend(read_jsonl(path))
    summary = summarize_grailqa_records(records)
    output_dir = Path(args.output_dir)
    write_json(output_dir / "grailqa_structure_summary.json", summary)
    (output_dir / "grailqa_structure_summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    (output_dir / "grailqa_structure_by_generalization.md").write_text(
        generalization_markdown(summary), encoding="utf-8"
    )
    print(summary_markdown(summary))


if __name__ == "__main__":
    main()
