#!/usr/bin/env python3
"""Run the bounded stability check for the five v1.1.2 disagreements.

The original GrailQA-25 run counts as run 0.  This script adds at most two
independent repetitions per method and case, so each method/case has at most
three total observations without selecting a favorable output.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from _bootstrap import PROJECT_ROOT  # noqa: E402

if str(PROJECT_ROOT) not in os.sys.path:

    os.sys.path.insert(0, str(PROJECT_ROOT))

from run_adaptive_v1_1_2_experiment import (  # noqa: E402
    OUTPUT_ROOT,
    SUBSET,
    AdaptiveRoutingAgent,
    CoRAgent,
    get_backend,
    load_fresh_ids,
    load_safe_project_env,
    read_jsonl,
    run_one,
    selected_rows,
)


CASE_IDS = [
    "3201985002000",
    "3202928006000",
    "4302129014000",
    "3205391005000",
    "4300043017000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2, choices=[1, 2])
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--relation-width", type=int, default=3)
    parser.add_argument("--sample-relation-threshold", type=int, default=1000)
    parser.add_argument("--temperature-exploration", type=float, default=0.3)
    parser.add_argument("--temperature-reasoning", type=float, default=0.1)
    parser.add_argument("--max-token", type=int, default=1024)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--remove-unnecessary-rel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def make_agent(method: str, args: argparse.Namespace) -> Any:
    kwargs = dict(
        model_name=os.environ["MODEL_NAME"],
        relation_width=args.relation_width,
        depth=args.depth,
        sample_relation_threshold=args.sample_relation_threshold,
        remove_unnecessary_rel=args.remove_unnecessary_rel,
        temperature_exploration=args.temperature_exploration,
        temperature_reasoning=args.temperature_reasoning,
        max_token=args.max_token,
        backend=get_backend("freebase"),
    )
    cor = CoRAgent(**kwargs)
    return cor if method == "cor" else AdaptiveRoutingAgent(
        cor, confidence_threshold=args.confidence_threshold, max_branches=3
    )


def main() -> None:
    args = parse_args()
    load_safe_project_env()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if not os.getenv("MODEL_NAME") or not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("MODEL_NAME and OPENAI_API_KEY are required")

    rows = selected_rows(load_fresh_ids())[SUBSET]
    rows_by_id = {str(row["id"]): row for row in rows}
    missing = [case_id for case_id in CASE_IDS if case_id not in rows_by_id]
    if missing:
        raise SystemExit(f"stability cases are not in the frozen sample: {missing}")

    output_root = OUTPUT_ROOT / "disagreement_analysis" / "stability"
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for repeat in range(1, args.repeats + 1):
        for method in ("cor", "adaptive"):
            agent = make_agent(method, args)
            for case_id in CASE_IDS:
                logging.info("[stability] repeat=%s method=%s id=%s", repeat, method, case_id)
                detail_dir = output_root / "details" / f"repeat_{repeat}" / method
                row = run_one(agent, method, rows_by_id[case_id], detail_dir)
                results.append(
                    {
                        "repeat": repeat,
                        "method": method,
                        "id": case_id,
                        "prediction": row.get("prediction", []),
                        "native_prediction": row.get("native_prediction", []),
                        "hits_at_1": row.get("hits_at_1", 0),
                        "f1": row.get("f1", 0.0),
                        "native_hits_at_1": row.get("native_hits_at_1", 0),
                        "native_f1": row.get("native_f1", 0.0),
                        "router_structure": row.get("router_structure", ""),
                        "final_execution_mode": row.get("final_execution_mode", ""),
                        "fallback_used": row.get("fallback_used", False),
                        "fallback_reason": row.get("fallback_reason"),
                        "relation_chain": row.get("relation_chain", []),
                        "error": row.get("error"),
                    }
                )

    path = output_root / "results.jsonl"
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
        encoding="utf-8",
    )
    (output_root / "config.json").write_text(
        json.dumps(
            {
                "experiment": "adaptive_v1_1_2_disagreement_stability",
                "case_ids": CASE_IDS,
                "historical_run_count": 1,
                "new_repeats": args.repeats,
                "max_total_runs_per_method_case": 1 + args.repeats,
                "model": os.getenv("MODEL_NAME", ""),
                "depth": args.depth,
                "relation_width": args.relation_width,
                "sample_relation_threshold": args.sample_relation_threshold,
                "temperature_exploration": args.temperature_exploration,
                "temperature_reasoning": args.temperature_reasoning,
                "max_token": args.max_token,
                "confidence_threshold": args.confidence_threshold,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(results), "path": str(path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
