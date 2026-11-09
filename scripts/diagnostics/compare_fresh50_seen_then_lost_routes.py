#!/usr/bin/env python3
"""Compare frozen TGAR and CFP traces for residual cases with early gold IDs.

This is a retrospective, offline comparison. Gold IDs are scoring-only, and no
inference, API call, or KG query is made.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256"
WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
FIRST_LOSS = WORK / "RESULTS/iter_011_residual_candidate_first_loss/candidate_first_loss.csv"
RUNS = EXP / "runs.jsonl"
GOLD = EXP / "evaluation_gold.jsonl"
DEFAULT_OUT = WORK / "RESULTS/iter_012_seen_then_lost_route_comparison"


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)

    first_loss = list(csv.DictReader(FIRST_LOSS.open(encoding="utf-8", newline="")))
    keys = {
        row["question_key"]
        for row in first_loss
        if row["candidate_presence_status"] == "SEEN_THEN_ABSENT_FROM_LATER_SEARCHES"
    }
    gold_rows = {
        f"{row['dataset']}::{row['question_id']}": row
        for row in jsonl(GOLD)
        if f"{row['dataset']}::{row['question_id']}" in keys
    }
    runs = [row for row in jsonl(RUNS) if row.get("question_key") in keys]

    comparisons: list[dict[str, Any]] = []
    scoring: list[dict[str, Any]] = []
    for key in sorted(keys):
        gold_ids = set(gold_rows[key].get("answer_ids", []))
        scoring.append({"question_key": key, "gold_answer_ids_scoring_only": sorted(gold_ids)})
        methods = {row.get("method"): row for row in runs if row.get("question_key") == key}
        if not {"v1_1_3", "phase2_5_6"}.issubset(methods):
            raise RuntimeError(f"Missing paired method trace for {key}: {sorted(methods)}")

        pair: dict[str, Any] = {"question_key": key, "gold_visible_to_test_time_method": False}
        for method in ("v1_1_3", "phase2_5_6"):
            run = methods[method]
            route_events = run.get("prompt_history", [])
            router = next(
                (event.get("parsed_result") for event in route_events
                 if isinstance(event.get("parsed_result"), dict)
                 and ("execution_mode" in event["parsed_result"] or "branches" in event["parsed_result"])),
                {},
            )
            branch_summaries = []
            for branch in run.get("branch_traces", []):
                candidates = set(branch.get("candidate_ids") or [])
                branch_summaries.append(
                    {
                        "branch_id": branch.get("branch_id"),
                        "action": branch.get("action"),
                        "constraint_summary": branch.get("constraint_summary"),
                        "candidate_count": branch.get("candidate_count"),
                        "gold_ids_in_branch_candidates_scoring_only": sorted(candidates & gold_ids),
                        "final_branch_answer_names": [
                            answer.get("name") for answer in branch.get("final_branch_answers", [])
                        ],
                    }
                )
            entity_searches = [
                event for event in run.get("sparql_history", [])
                if event.get("search_type") == "entity_search"
            ]
            entity_hits = [
                {
                    "trace_id": event.get("trace_id"),
                    "candidate_count": event.get("result_count"),
                    "gold_ids_present_scoring_only": sorted(
                        set(event.get("candidate_entity_ids") or []) & gold_ids
                    ),
                    "query_sha256": hashlib.sha256((event.get("sparql") or "").encode()).hexdigest(),
                }
                for event in entity_searches
            ]
            parsed_types = [
                {"type": event.get("type"), "parsed_result": event.get("parsed_result")}
                for event in route_events
                if event.get("type") in {
                    "adaptive_router_v1_1_3",
                    "phase2_5_5_canonical_normalization_router",
                    "relation_prune",
                    "reasoning",
                    "filter_answer",
                    "generate_without_explored_paths",
                }
            ]
            pair[method] = {
                "hits_at_1": run.get("hits_at_1"),
                "prediction": run.get("prediction"),
                "route": run.get("route"),
                "router_execution_mode": run.get("router_execution_mode"),
                "router_structure": run.get("router_structure"),
                "fallback_reason": run.get("fallback_reason"),
                "validation_failures": run.get("validation_failures"),
                "router_proposal_summary": {
                    key_name: router.get(key_name)
                    for key_name in (
                        "execution_mode", "structure", "dependency_pattern", "constraints", "branches"
                    )
                    if key_name in router
                },
                "entity_searches": entity_hits,
                "branch_summaries": branch_summaries,
                "archived_prompt_decisions": parsed_types,
            }
        comparisons.append(pair)

    (out / "paired_route_trace_comparison.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "gold_scoring_only.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scoring), encoding="utf-8"
    )
    summary = {
        "experiment": "paired TGAR versus CFP trace comparison for archived earlier-only gold candidates",
        "claim_scope": "This is a frozen-run association, not a controlled intervention. It identifies a promising planning/constraint-routing direction but does not establish that changing TGAR would reproduce CFP outcomes.",
        "command": "PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/compare_fresh50_seen_then_lost_routes.py",
        "cohort_count": len(comparisons),
        "question_keys": sorted(keys),
        "tgar_hits": sum(row["v1_1_3"]["hits_at_1"] == 1 for row in comparisons),
        "cfp_hits": sum(row["phase2_5_6"]["hits_at_1"] == 1 for row in comparisons),
        "source_runs_sha256": sha256(RUNS),
        "gold_sha256": sha256(GOLD),
        "candidate_loss_audit_sha256": sha256(FIRST_LOSS),
        "llm_calls": 0,
        "read_only_kg_queries": 0,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    hashes = {
        path.name: sha256(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
