#!/usr/bin/env python3
"""Download an explicitly supplied CWQ source into the raw-data area.

The repository already contains the processed CWQ source used by the current
pipeline. This utility intentionally requires a URL instead of inventing a
dataset source or silently replacing the checked-in data.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Explicit CWQ source URL")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data/raw/cwq/cwq.json"),
        help="Destination JSON path",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(args.url, headers={"User-Agent": "relation-graph-kgqa/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    with partial.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, (list, dict)):
        raise SystemExit("Downloaded CWQ source must be a JSON list or object")
    partial.replace(output)
    print(f"Downloaded CWQ source to {output}")


if __name__ == "__main__":
    main()
