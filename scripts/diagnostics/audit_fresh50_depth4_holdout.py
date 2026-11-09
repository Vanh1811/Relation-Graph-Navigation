#!/usr/bin/env python3
"""Offline integrity and paired-score audit for Iteration 19.

This script reads the frozen holdout inputs and completed run journals only.
It makes no API or SPARQL requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cor_baseline_experiment import evaluate  # noqa: E402

WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
OUT = WORK / "RESULTS/iter_019_depth4_holdout_preparation"
SNAPSHOT = WORK / "RESULTS/iter_000_baseline/SNAPSHOT_MANIFEST.json"
TARGET = "STRONG_DEPTH4_TARGET"
CONTROL = "STRONG_SHORTER_PATH_NEGATIVE_CONTROL"
ALLOWED_INFERENCE_FIELDS = {
    "question_key", "case_id", "dataset", "split", "question",
    "topic_entities", "topic_entity_names",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_artifact_hashes(out: Path) -> None:
    names = (
        "call_usage.jsonl", "holdout_manifest.json", "holdout_report.md",
        "holdout_results.json", "inference_cases.jsonl", "offline_audit.json",
        "paired_analysis.json", "paired_case_results.csv", "paired_runs.jsonl",
        "preflight_holdout.json", "report.md", "run_authorization.json",
        "run_metadata.json", "scoring_gold_only.jsonl", "control_continuation_budget_request.json",
    )
    hashes = {name: sha(out / name) for name in names}
    hashes["runner_source_sha256"] = sha(ROOT / "scripts/diagnostics/run_fresh50_depth4_holdout.py")
    hashes["instrumentation_source_sha256"] = sha(ROOT / "scripts/diagnostics/run_fresh50_state_lineage.py")
    hashes["offline_auditor_source_sha256"] = sha(Path(__file__).resolve())
    hashes["control_continuation_runner_source_sha256"] = sha(ROOT / "scripts/diagnostics/run_fresh50_depth4_control_continuation.py")
    (out / "artifact_hashes.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output_dir

    manifest = read_json(out / "holdout_manifest.json")
    metadata = read_json(out / "run_metadata.json")
    authorization = read_json(out / "run_authorization.json")
    results = read_json(out / "holdout_results.json")
    paired_report = read_json(out / "paired_analysis.json")
    inference = read_jsonl(out / "inference_cases.jsonl")
    scoring_rows = read_jsonl(out / "scoring_gold_only.jsonl")
    runs = read_jsonl(out / "paired_runs.jsonl")
    usage = read_jsonl(out / "call_usage.jsonl")

    scoring = {row["question_key"]: row for row in scoring_rows}
    role_by_key = {key: row["cohort_role_scoring_only"] for key, row in scoring.items()}
    expected_inputs = {
        "holdout_manifest.json": sha(out / "holdout_manifest.json"),
        "inference_cases.jsonl": sha(out / "inference_cases.jsonl"),
        "scoring_gold_only.jsonl": sha(out / "scoring_gold_only.jsonl"),
    }
    frozen_input_hashes_ok = metadata.get("frozen_input_hashes") == expected_inputs
    auth_ok = (
        authorization.get("authorized") is True
        and authorization.get("holdout_manifest_sha256") == expected_inputs["holdout_manifest.json"]
        and authorization.get("request_cap") == 1000
        and authorization.get("cost_cap_usd") == 0.70
        and authorization.get("model") == "gpt-5.6-luna"
    )
    inference_keys = {row.get("question_key") for row in inference}
    inference_schema_ok = (
        len(inference) == 30
        and inference_keys == set(scoring)
        and all(set(row) <= ALLOWED_INFERENCE_FIELDS for row in inference)
    )

    schedule = {
        (row["question_key"], int(row["repeat_index"]), int(row["depth"]))
        for row in manifest["treatment_protocol"]["randomized_run_schedule"]
    }
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    duplicate_slots = []
    score_mismatches = []
    for row in runs:
        slot = (row["question_key"], int(row["repeat_index"]), int(row["diagnostic_depth"]))
        if slot in cells:
            duplicate_slots.append(slot)
        cells[slot] = row
        expected_hit, expected_f1 = evaluate(row.get("prediction", []), scoring[row["question_key"]].get("gold_answers", []))
        if expected_hit != int(row.get("hits_at_1", 0)) or abs(expected_f1 - float(row.get("f1", 0.0))) > 1e-10:
            score_mismatches.append({
                "slot": list(slot), "stored": [row.get("hits_at_1"), row.get("f1")],
                "recomputed": [expected_hit, expected_f1],
            })

    unexpected_slots = sorted(cells.keys() - schedule)
    all_runs_completed = all(row.get("case_status") == "COMPLETED" for row in runs)
    by_case_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        by_case_runs[row["question_key"]].append(row)
    coverage = {
        role: dict(Counter(len(by_case_runs.get(key, [])) for key, row in scoring.items()
                           if row["cohort_role_scoring_only"] == role))
        for role in (TARGET, CONTROL)
    }

    paired_rows = []
    pair_counts: dict[str, Counter[str]] = defaultdict(Counter)
    by_case_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, score in scoring.items():
        role = score["cohort_role_scoring_only"]
        for repeat in (1, 2):
            d3 = cells.get((key, repeat, 3))
            d4 = cells.get((key, repeat, 4))
            if d3 is None or d4 is None:
                continue
            h3, h4 = int(d3.get("hits_at_1", 0)), int(d4.get("hits_at_1", 0))
            outcome = "TIE" if h3 == h4 else "DEPTH4_ONLY" if h4 else "DEPTH3_ONLY"
            row = {
                "question_key": key, "cohort_role_scoring_only": role,
                "repeat_index": repeat, "paired_outcome": outcome,
                "depth3_hits_at_1": h3, "depth4_hits_at_1": h4,
                "depth3_f1": float(d3.get("f1", 0.0)),
                "depth4_f1": float(d4.get("f1", 0.0)),
            }
            pair_counts[role][outcome] += 1
            paired_rows.append(row)
            by_case_pairs[key].append(row)

    case_consistency: dict[str, Counter[str]] = defaultdict(Counter)
    for key, rows in by_case_pairs.items():
        if len(rows) != 2:
            continue
        outcomes = [row["paired_outcome"] for row in sorted(rows, key=lambda r: r["repeat_index"])]
        if outcomes == ["DEPTH4_ONLY", "DEPTH4_ONLY"]:
            case_consistency[role_by_key[key]]["BOTH_DEPTH4_ONLY"] += 1
        if outcomes == ["DEPTH3_ONLY", "DEPTH3_ONLY"]:
            case_consistency[role_by_key[key]]["BOTH_DEPTH3_ONLY"] += 1
        if len(set(outcomes) - {"TIE"}) > 1:
            case_consistency[role_by_key[key]]["MIXED_DIRECTION"] += 1
        if "DEPTH4_ONLY" not in outcomes:
            case_consistency[role_by_key[key]]["NO_DEPTH4_ONLY"] += 1

    condition_metrics = {}
    candidate_coverage = {}
    paired_deltas = {}
    for role in (TARGET, CONTROL):
        condition_metrics[role] = {}
        candidate_coverage[role] = {}
        for depth in (3, 4):
            subset = [row for row in runs if role_by_key[row["question_key"]] == role and int(row["diagnostic_depth"]) == depth]
            condition_metrics[role][str(depth)] = {
                "runs": len(subset),
                "mean_hits_at_1": mean([float(row.get("hits_at_1", 0)) for row in subset]),
                "mean_f1": mean([float(row.get("f1", 0.0)) for row in subset]),
            }
            gold_seen = sum(
                bool(row.get("gold_candidate_diagnostic_scoring_only", {}).get("gold_answer_seen_any_search"))
                for row in subset
            )
            candidate_coverage[role][str(depth)] = {
                "runs_with_gold_in_any_entity_search": gold_seen,
                "runs": len(subset),
                "rate": gold_seen / len(subset) if subset else None,
            }
        role_pairs = [row for row in paired_rows if row["cohort_role_scoring_only"] == role]
        paired_deltas[role] = {
            "paired_repeats": len(role_pairs),
            "mean_hits_at_1_delta_depth4_minus_depth3": mean([
                row["depth4_hits_at_1"] - row["depth3_hits_at_1"] for row in role_pairs
            ]),
            "mean_f1_delta_depth4_minus_depth3": mean([
                row["depth4_f1"] - row["depth3_f1"] for row in role_pairs
            ]),
        }

    starts = [row for row in usage if row.get("status") == "REQUEST_STARTED"]
    metered = [row for row in usage if row.get("status") == "REQUEST_METERED"]
    unmetered = [row for row in usage if row.get("status") == "REQUEST_UNMETERED_ERROR"]
    started_ids = [row.get("request_id") for row in starts]
    metered_ids = [row.get("request_id") for row in metered]
    usage_accounting_ok = (
        len(starts) == len(metered) == 1000
        and not unmetered
        and len(set(started_ids)) == 1000
        and set(started_ids) == set(metered_ids)
    )
    total_input = sum(int(row.get("input_tokens", 0) or 0) for row in metered)
    total_output = sum(int(row.get("output_tokens", 0) or 0) for row in metered)
    total_cost = sum(float(row.get("cost_usd", 0.0) or 0.0) for row in metered)
    empty_responses = sum(row.get("has_content") is False for row in metered)
    usage_matches_summary = (
        total_input == results.get("input_tokens")
        and total_output == results.get("output_tokens")
        and abs(total_cost - float(results.get("cost_usd", 0.0))) < 1e-10
    )

    query_status = Counter(
        query.get("status", "UNKNOWN")
        for run in runs for query in run.get("sparql_history", [])
    )
    case_query_count = sum(int(row.get("kg_queries", 0) or 0) for row in runs)

    # Verify each frozen production file against the original baseline snapshot.
    snapshot = read_json(SNAPSHOT)
    source_failures = []
    source_hashes = snapshot.get("production_reasoning_source_hashes", {})
    for relative, expected in source_hashes.items():
        path = ROOT / relative
        if not path.exists() or sha(path) != expected:
            source_failures.append(relative)

    expected_pairs = {
        role: {key: value for key, value in counts.items()}
        for role, counts in paired_report.get("paired_outcomes_by_role", {}).items()
    }
    actual_pairs = {role: dict(counts) for role, counts in pair_counts.items()}
    paired_summary_matches = (
        len(paired_rows) == len(paired_report.get("paired_rows", []))
        and actual_pairs == expected_pairs
        and condition_metrics == results.get("condition_metrics_by_role")
        and paired_deltas == results.get("paired_mean_deltas_by_role")
    )

    checks = {
        "authorization_matches_frozen_manifest_and_caps": auth_ok,
        "frozen_input_hashes_match": frozen_input_hashes_ok,
        "inference_schema_excludes_gold_fields": inference_schema_ok,
        "all_completed_run_slots_are_unique_and_expected": not duplicate_slots and not unexpected_slots and all_runs_completed,
        "scores_recompute_without_mismatch": not score_mismatches,
        "request_usage_is_fully_metered_and_within_cap": usage_accounting_ok,
        "token_and_cost_totals_match_run_summary": usage_matches_summary,
        "paired_summary_recomputes": paired_summary_matches,
        "production_source_integrity_passes": not source_failures and len(source_hashes) == 7,
        "experiment_status_correctly_marks_budget_capped_partial": results.get("status") == "INCOMPLETE_BUDGET_CAPPED" and len(runs) < 120 and len(starts) == 1000,
    }
    audit = {
        "experiment": results.get("experiment"),
        "audit_status": "PASS" if all(checks.values()) else "FAIL",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "network_requests_made_by_auditor": 0,
        "checks": checks,
        "cohort_coverage_by_role_and_completed_run_count": coverage,
        "method_runs_completed": len(runs),
        "method_runs_expected": len(schedule),
        "complete_depth_pairs": len(paired_rows),
        "paired_outcomes_by_role": actual_pairs,
        "condition_metrics_by_role": condition_metrics,
        "gold_candidate_coverage_by_role_and_depth_scoring_only": candidate_coverage,
        "paired_mean_deltas_by_role": paired_deltas,
        "case_repeat_consistency_by_role": {role: dict(counts) for role, counts in case_consistency.items()},
        "score_mismatch_count": len(score_mismatches),
        "score_mismatches": score_mismatches,
        "requests_started": len(starts),
        "requests_metered": len(metered),
        "unmetered_requests": len(unmetered),
        "empty_content_responses": empty_responses,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cost_usd": total_cost,
        "request_cap": 1000,
        "cost_cap_usd": 0.70,
        "budget_stop_reason": results.get("budget_stop_reason"),
        "case_sparql_queries": case_query_count,
        "case_sparql_status_counts": dict(query_status),
        "preflight_sparql_queries": results.get("preflight_kg_queries"),
        "production_source_files_checked": len(source_hashes),
        "production_source_failures": source_failures,
        "unrun_slots": len(schedule) - len(runs),
        "notes": [
            "All 20 four-edge target cases completed at both depths for both repeats.",
            "Only three of ten shorter-path controls had any completed runs: two controls have both repeats; one has one paired repeat; seven controls are unrun.",
            "One case SPARQL query is recorded as timed out; its read-only backend attempted it three times before returning an empty result. All other case queries returned status ok.",
            "The 1,000-request cap was reached exactly; the $0.70 cap was not reached. The result is partial because the control schedule was not completed.",
        ],
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "offline_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    refresh_artifact_hashes(out)
    if not all(checks.values()):
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
