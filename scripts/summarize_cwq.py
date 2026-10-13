#!/usr/bin/env python3
"""Generate CWQ structural summary reports from analysis JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: E402
from relation_graph_kgqa.analysis.summarize import summary_markdown, summarize_records  # noqa: E402
from relation_graph_kgqa.utils.io import read_jsonl, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "data/processed"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs/reports"))
    args = parser.parse_args()

    records = []
    for split in ("train", "dev", "test"):
        path = Path(args.input_dir) / f"cwq_{split}_analysis.jsonl"
        if path.exists():
            records.extend(read_jsonl(path))
    summary = summarize_records(records)
    output_dir = Path(args.output_dir)
    write_json(output_dir / "cwq_structure_summary.json", summary)
    (output_dir / "cwq_structure_summary.md").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "cwq_structure_summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    print(summary_markdown(summary))


if __name__ == "__main__":
    main()
