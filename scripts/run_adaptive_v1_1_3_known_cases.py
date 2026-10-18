#!/usr/bin/env python3
"""Run the three v1.1.3 known-case regression checks once each."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from run_adaptive_v1_1_3_experiment import (
	OUTPUT_ROOT,
	load_rows,
	load_safe_project_env,
	make_agent,
	run_one,
)


KNOWN_IDS = ["4300043017000", "3205391005000", "4302129014000"]


def main() -> None:
	load_safe_project_env()
	if not os.getenv("MODEL_NAME") or not os.getenv("OPENAI_API_KEY"):
		raise SystemExit("MODEL_NAME and OPENAI_API_KEY are required")
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
	root = OUTPUT_ROOT / "regression_checks"
	root.mkdir(parents=True, exist_ok=True)
	rows = {str(row["id"]): row for row in load_rows(KNOWN_IDS)}
	args = type("Args", (), {
		"relation_width": 3,
		"depth": 3,
		"sample_relation_threshold": 1000,
		"remove_unnecessary_rel": True,
		"temperature_exploration": 0.3,
		"temperature_reasoning": 0.1,
		"max_token": 1024,
		"confidence_threshold": 0.75,
	})()
	agent = make_agent("adaptive", args)
	results = []
	for case_id in KNOWN_IDS:
		results.append(run_one(agent, "adaptive", 1, rows[case_id], root / "details" / "adaptive"))
	(root / "known_case_results.jsonl").write_text(
		"".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results), encoding="utf-8"
	)
	(root / "known_case_ids.json").write_text(json.dumps(KNOWN_IDS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	print(json.dumps({"cases": len(results), "output": str(root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
