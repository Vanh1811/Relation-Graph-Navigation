#!/usr/bin/env python3
"""Offline first-loss audit of archived TGAR entity-search candidate IDs.

This script reads frozen Fresh50 run traces and the already-frozen cohort from
iteration 10. Gold answer IDs are used only for retrospective scoring; no model
is called and no KG request is made.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256"
WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
SOURCE = EXP / "runs.jsonl"
GOLD = EXP / "evaluation_gold.jsonl"
COHORT = WORK / "RESULTS/iter_010_semantic_archived_path_audit/semantic_archived_path_alignment.csv"
OUT = WORK / "RESULTS/iter_011_residual_candidate_first_loss"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="new output directory (existing iteration artifacts are never overwritten)",
    )
    args = parser.parse_args()
    out = args.output_dir
    cohort_rows = list(csv.DictReader(COHORT.open(encoding="utf-8", newline="")))
    cohort = {
        row["question_key"]: row["cohort_role"]
        for row in cohort_rows
        if row["cohort_role"] in {"TGAR_FAILURE", "BOTH_CORRECT_CONTROL"}
    }
    gold_rows = {f"{row['dataset']}::{row['question_id']}": row for row in read_jsonl(GOLD)}
    runs = {
        row["question_key"]: row
        for row in read_jsonl(SOURCE)
        if row.get("method") == "v1_1_3" and row.get("question_key") in cohort
    }
    if set(runs) != set(cohort):
        raise RuntimeError(f"Cohort/run mismatch: missing={sorted(set(cohort)-set(runs))}")

    out.mkdir(parents=True, exist_ok=False)
    scoring_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for key, role in sorted(cohort.items()):
        run = runs[key]
        gold = gold_rows[key]
        gold_ids = set(gold.get("answer_ids", []))
        searches = [
            event
            for event in run.get("sparql_history", [])
            if event.get("search_type") == "entity_search"
        ]
        presence: list[tuple[int, set[str]]] = []
        for event_index, event in enumerate(searches, start=1):
            candidate_ids = set(event.get("candidate_entity_ids") or [])
            matched = candidate_ids & gold_ids
            presence.append((event_index, matched))
            query = event.get("sparql") or ""
            event_rows.append(
                {
                    "question_key": key,
                    "event_index": event_index,
                    "trace_id": event.get("trace_id"),
                    "status": event.get("status"),
                    "result_count": event.get("result_count"),
                    "candidate_id_count": len(candidate_ids),
                    "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                    "candidate_entity_ids": sorted(candidate_ids),
                    "candidate_entity_names": event.get("candidate_entity_names") or [],
                    "sparql": query,
                }
            )

        seen_indexes = [index for index, matched in presence if matched]
        first_seen = seen_indexes[0] if seen_indexes else None
        last_seen = seen_indexes[-1] if seen_indexes else None
        last_event_match = sorted(presence[-1][1]) if presence else []
        first_loss = None
        if first_seen is not None:
            first_loss = next(
                (index for index, matched in presence if index > first_seen and not matched),
                None,
            )
        if not seen_indexes:
            status = "NEVER_SEEN"
        elif last_event_match:
            status = "PRESENT_IN_LAST_ENTITY_SEARCH"
        else:
            status = "SEEN_THEN_ABSENT_FROM_LATER_SEARCHES"

        parsed = [event.get("parsed_result") for event in run.get("prompt_history", [])]
        router = next(
            (value for value in parsed if isinstance(value, dict) and "execution_mode" in value),
            {},
        )
        relation_prunes = [
            value for value in parsed
            if isinstance(value, list) and all(isinstance(item, str) for item in value)
        ]
        reasoning = [
            value for value in parsed
            if isinstance(value, str) and value in {"Forward", "Backtrack", "Constraint", "Stop"}
        ]
        final_generation = next(
            (event.get("parsed_result") for event in reversed(run.get("prompt_history", []))
             if event.get("type") == "generate_without_explored_paths"),
            None,
        )
        decision = {
            "question_key": key,
            "cohort_role": role,
            "tgar_route": run.get("route"),
            "tgar_hits_at_1": run.get("hits_at_1"),
            "router_structure": router.get("structure"),
            "router_execution_mode": router.get("execution_mode"),
            "relation_prune_outputs": relation_prunes,
            "reasoning_decisions": reasoning,
            "ungrounded_final_generation": final_generation,
        }
        decisions.append(decision)
        scoring_rows.append(
            {
                "question_key": key,
                "cohort_role": role,
                "gold_answer_ids_scoring_only": sorted(gold_ids),
                "gold_answer_text_scoring_only": gold.get("gold_answers", []),
            }
        )
        audit_rows.append(
            {
                "question_key": key,
                "cohort_role": role,
                "tgar_hits_at_1": run.get("hits_at_1"),
                "entity_search_count": len(searches),
                "gold_seen_any_search": bool(seen_indexes),
                "first_gold_seen_event_index": first_seen,
                "last_gold_seen_event_index": last_seen,
                "first_later_search_without_gold_event_index": first_loss,
                "gold_ids_in_last_entity_search": ";".join(last_event_match),
                "candidate_presence_status": status,
                "entity_search_trace_ids": ";".join(str(event.get("trace_id")) for event in searches),
                "query_hashes": ";".join(
                    hashlib.sha256((event.get("sparql") or "").encode()).hexdigest()
                    for event in searches
                ),
            }
        )

    with (out / "archived_entity_search_events.jsonl").open("w", encoding="utf-8") as stream:
        for row in event_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "case_decisions_non_gold.jsonl").open("w", encoding="utf-8") as stream:
        for row in decisions:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "gold_scoring_only.jsonl").open("w", encoding="utf-8") as stream:
        for row in scoring_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "candidate_first_loss.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    failures = [row for row in audit_rows if row["cohort_role"] == "TGAR_FAILURE"]
    controls = [row for row in audit_rows if row["cohort_role"] == "BOTH_CORRECT_CONTROL"]
    failure_counts = Counter(row["candidate_presence_status"] for row in failures)
    control_counts = Counter(row["candidate_presence_status"] for row in controls)
    result = {
        "experiment": "retrospective first-loss audit of archived entity-search candidate IDs",
        "claim_scope": "The audit identifies whether gold IDs were ever present in archived entity-search outputs and records the first later query without them. It does not establish that an intervention on route selection would repair accuracy.",
        "command": "PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/audit_fresh50_residual_candidate_loss.py",
        "source_trace": str(SOURCE),
        "source_trace_sha256": sha256(SOURCE),
        "gold_file_sha256": sha256(GOLD),
        "cohort_file_sha256": sha256(COHORT),
        "failure_count": len(failures),
        "control_count": len(controls),
        "failure_presence_status_counts": dict(failure_counts),
        "control_presence_status_counts": dict(control_counts),
        "gold_ids_seen_any_search_failures": sum(row["gold_seen_any_search"] for row in failures),
        "gold_ids_seen_in_last_search_failures": sum(bool(row["gold_ids_in_last_entity_search"]) for row in failures),
        "failure_case_keys_seen_then_lost": [
            row["question_key"] for row in failures
            if row["candidate_presence_status"] == "SEEN_THEN_ABSENT_FROM_LATER_SEARCHES"
        ],
        "read_only_kg_queries": 0,
        "llm_calls": 0,
        "gold_visible_to_test_time_method": False,
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    artifacts = [
        path for path in out.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    ]
    hashes = {path.name: sha256(path) for path in sorted(artifacts)}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
