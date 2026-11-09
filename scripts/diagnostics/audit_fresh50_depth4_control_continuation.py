#!/usr/bin/env python3
"""Offline reconciliation for the completed Iteration 19 control continuation."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts/diagnostics")]

import run_fresh50_depth4_control_continuation as continuation  # noqa: E402
import run_fresh50_depth4_holdout as holdout  # noqa: E402
import run_fresh50_state_lineage as instrumentation  # noqa: E402
from run_cor_baseline_experiment import evaluate  # noqa: E402

OUT = continuation.OUT
CONT = continuation.CONT_OUT
AUTH = continuation.AUTH_PATH
TARGET = continuation.TARGET_ROLE
CONTROL = continuation.CONTROL_ROLE
REQUEST_CAP = 1400
COST_CAP_USD = 0.82
ADDITIONAL_REQUEST_CAP = 400
ADDITIONAL_COST_CAP_USD = 0.251658


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slot_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return row["question_key"], int(row["repeat_index"]), int(row["diagnostic_depth"])


def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": len(rows),
        "mean_hits_at_1": sum(int(r.get("hits_at_1", 0)) for r in rows) / len(rows) if rows else None,
        "mean_f1": sum(float(r.get("f1", 0.0)) for r in rows) / len(rows) if rows else None,
    }


def cumulative_rows(base_bytes: bytes, extra_rows: list[dict[str, Any]], base_cost: float) -> bytes:
    base_rows = [json.loads(line) for line in base_bytes.decode("utf-8").splitlines() if line.strip()]
    base_starts = sum(row.get("status") == "REQUEST_STARTED" for row in base_rows)
    local_to_global: dict[str, str] = {}
    for row in extra_rows:
        if row.get("status") == "REQUEST_STARTED":
            local_id = str(row["request_id"])
            number = int(local_id.rsplit("_", 1)[1])
            local_to_global[local_id] = f"request_{base_starts + number:04d}"
    local_cost = 0.0
    appended: list[dict[str, Any]] = []
    for original in extra_rows:
        row = dict(original)
        local_id = str(row["request_id"])
        if local_id not in local_to_global:
            raise ValueError(f"Continuation usage row has no matching request start: {local_id}")
        row["continuation_local_request_id"] = local_id
        row["request_id"] = local_to_global[local_id]
        row["continuation_added"] = True
        if row.get("status") == "REQUEST_METERED":
            local_cost += float(row.get("cost_usd", 0.0) or 0.0)
            row["cumulative_cost_usd"] = base_cost + local_cost
        appended.append(row)
    prefix = base_bytes if base_bytes.endswith(b"\n") else base_bytes + b"\n"
    suffix = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in appended
    )
    return prefix + suffix


def main() -> int:
    plan = continuation.inspect_plan()
    auth = read_json(AUTH)
    base_bytes = (OUT / "call_usage.jsonl").read_bytes()
    base_runs = read_jsonl(OUT / "paired_runs.jsonl")
    base_usage = read_jsonl(OUT / "call_usage.jsonl")
    extra_runs = read_jsonl(CONT / "continuation_runs.jsonl")
    extra_usage = read_jsonl(CONT / "call_usage.jsonl")
    all_runs = base_runs + extra_runs
    scoring = plan["scores"]
    manifest = plan["manifest"]
    schedule_rows = manifest["treatment_protocol"]["randomized_run_schedule"]
    expected_slots = {
        (row["question_key"], int(row["repeat_index"]), int(row["depth"]))
        for row in schedule_rows
    }
    all_slots = [slot_key(row) for row in all_runs]
    slot_counts = Counter(all_slots)
    completed = [row for row in all_runs if row.get("case_status") == "COMPLETED"]
    completed_slots = {slot_key(row) for row in completed}
    missing_initial = {
        (row["question_key"], int(row["repeat_index"]), int(row["depth"]))
        for row in plan["missing_slots"]
    }
    extra_slots = [slot_key(row) for row in extra_runs]

    score_mismatches = []
    for row in completed:
        hit, f1 = evaluate(row.get("prediction", []), scoring[row["question_key"]].get("gold_answers", []))
        if hit != int(row.get("hits_at_1", 0)) or abs(f1 - float(row.get("f1", 0.0))) > 1e-10:
            score_mismatches.append({"slot": list(slot_key(row)), "stored": [row.get("hits_at_1"), row.get("f1")], "recomputed": [hit, f1]})

    role_violations = [row["question_key"] for row in extra_runs if row.get("cohort_role_scoring_only") != CONTROL]
    extra_scope_violations = [list(key) for key in extra_slots if key not in missing_initial]
    base_starts = [row for row in base_usage if row.get("status") == "REQUEST_STARTED"]
    base_metered = [row for row in base_usage if row.get("status") == "REQUEST_METERED"]
    extra_starts = [row for row in extra_usage if row.get("status") == "REQUEST_STARTED"]
    extra_metered = [row for row in extra_usage if row.get("status") == "REQUEST_METERED"]
    unmetered_errors = [row for row in base_usage + extra_usage if row.get("status") == "REQUEST_UNMETERED_ERROR"]

    def pending(starts: list[dict[str, Any]], finishes: list[dict[str, Any]]) -> set[str]:
        finished = {row.get("request_id") for row in finishes}
        return {str(row.get("request_id")) for row in starts} - {str(value) for value in finished}

    base_pending = pending(base_starts, [r for r in base_usage if r.get("status") in {"REQUEST_METERED", "REQUEST_UNMETERED_ERROR"}])
    extra_pending = pending(extra_starts, [r for r in extra_usage if r.get("status") in {"REQUEST_METERED", "REQUEST_UNMETERED_ERROR"}])
    unique_base_ids = len({row.get("request_id") for row in base_starts}) == len(base_starts)
    unique_extra_ids = len({row.get("request_id") for row in extra_starts}) == len(extra_starts)

    base_cost = sum(float(row.get("cost_usd", 0.0) or 0.0) for row in base_metered)
    extra_cost = sum(float(row.get("cost_usd", 0.0) or 0.0) for row in extra_metered)
    base_input = sum(int(row.get("input_tokens", 0) or 0) for row in base_metered)
    base_output = sum(int(row.get("output_tokens", 0) or 0) for row in base_metered)
    extra_input = sum(int(row.get("input_tokens", 0) or 0) for row in extra_metered)
    extra_output = sum(int(row.get("output_tokens", 0) or 0) for row in extra_metered)
    total_calls = len(base_starts) + len(extra_starts)
    total_cost = base_cost + extra_cost
    all_empty = sum(row.get("has_content") is False for row in base_metered + extra_metered)
    extra_empty = sum(row.get("has_content") is False for row in extra_metered)

    # Preserve the raw runner-produced summary before correcting its stale usage totals.
    raw_result_path = CONT / "continuation_results_runner_output.json"
    if not raw_result_path.exists():
        shutil.copyfile(CONT / "continuation_results.json", raw_result_path)

    expected_cumulative = cumulative_rows(base_bytes, extra_usage, base_cost)
    cumulative_path = CONT / "cumulative_call_usage.jsonl"
    existing_cumulative = cumulative_path.read_bytes()
    if existing_cumulative == base_bytes:
        cumulative_path.write_bytes(expected_cumulative)
    elif existing_cumulative != expected_cumulative:
        raise SystemExit("Cumulative request journal differs from both the frozen base and its deterministic continuation merge")
    cumulative = read_jsonl(cumulative_path)
    cumulative_starts = [row for row in cumulative if row.get("status") == "REQUEST_STARTED"]
    cumulative_metered = [row for row in cumulative if row.get("status") == "REQUEST_METERED"]
    cumulative_ids = [row.get("request_id") for row in cumulative_starts]
    cumulative_pending = pending(cumulative_starts, [r for r in cumulative if r.get("status") in {"REQUEST_METERED", "REQUEST_UNMETERED_ERROR"}])
    cumulative_cost_from_journal = sum(float(row.get("cost_usd", 0.0) or 0.0) for row in cumulative_metered)
    cumulative_input_from_journal = sum(int(row.get("input_tokens", 0) or 0) for row in cumulative_metered)
    cumulative_output_from_journal = sum(int(row.get("output_tokens", 0) or 0) for row in cumulative_metered)

    paired = holdout.paired_summary(completed, scoring)
    paired_counts = paired["paired_outcomes_by_role"]
    paired_outcomes_ok = (
        len(paired["paired_rows"]) == 60
        and sum(1 for row in paired["paired_rows"] if row["cohort_role_scoring_only"] == TARGET) == 40
        and sum(1 for row in paired["paired_rows"] if row["cohort_role_scoring_only"] == CONTROL) == 20
        and paired_counts.get(TARGET) == {"TIE": 26, "DEPTH3_ONLY": 1, "DEPTH4_ONLY": 13}
        and paired_counts.get(CONTROL) == {"TIE": 19, "DEPTH3_ONLY": 1}
    )
    condition_metrics: dict[str, dict[str, Any]] = {}
    candidate_coverage: dict[str, dict[str, Any]] = {}
    paired_deltas: dict[str, dict[str, Any]] = {}
    for role in (TARGET, CONTROL):
        condition_metrics[role] = {}
        candidate_coverage[role] = {}
        for depth in (3, 4):
            group = [row for row in completed if row.get("cohort_role_scoring_only") == role and int(row.get("diagnostic_depth", 0)) == depth]
            condition_metrics[role][str(depth)] = metric(group)
            gold_seen = sum(bool(row.get("gold_candidate_diagnostic_scoring_only", {}).get("gold_answer_seen_any_search")) for row in group)
            candidate_coverage[role][str(depth)] = {
                "runs_with_gold_in_any_entity_search": gold_seen,
                "runs": len(group),
                "rate": gold_seen / len(group) if group else None,
            }
        pairs = [row for row in paired["paired_rows"] if row["cohort_role_scoring_only"] == role]
        paired_deltas[role] = {
            "paired_repeats": len(pairs),
            "mean_hits_at_1_delta_depth4_minus_depth3": sum(row["depth4_hits_at_1"] - row["depth3_hits_at_1"] for row in pairs) / len(pairs) if pairs else None,
            "mean_f1_delta_depth4_minus_depth3": sum(row["depth4_f1"] - row["depth3_f1"] for row in pairs) / len(pairs) if pairs else None,
        }

    source_integrity = instrumentation.production_source_integrity()
    previous_hashes_path = CONT / "continuation_hashes.json"
    previous_hashes = read_json(previous_hashes_path) if previous_hashes_path.exists() else {}
    run_hashes_ok = (
        auth.get("continuation_runner_sha256") == previous_hashes.get("continuation_runner_sha256_at_execution")
        and auth.get("manifest_sha256") == sha(holdout.MANIFEST_PATH)
        and auth.get("inference_sha256") == sha(holdout.INFERENCE_PATH)
        and auth.get("scoring_sha256") == sha(holdout.SCORING_PATH)
        and auth.get("base_paired_runs_sha256") == sha(OUT / "paired_runs.jsonl")
        and auth.get("base_call_usage_sha256") == sha(OUT / "call_usage.jsonl")
    )
    checks = {
        "all_120_frozen_slots_completed_once": len(completed) == 120 and completed_slots == expected_slots and all(count == 1 for count in slot_counts.values()),
        "continuation_contains_only_30_authorized_control_slots": len(extra_runs) == 30 and set(extra_slots) == missing_initial and not role_violations and not extra_scope_violations,
        "all_scores_recompute_without_mismatch": not score_mismatches,
        "paired_outcomes_recompute_for_40_targets_and_20_controls": paired_outcomes_ok,
        "base_and_continuation_requests_fully_metered": len(base_starts) == len(base_metered) == 1000 and len(extra_starts) == len(extra_metered) == 237 and not base_pending and not extra_pending and not unmetered_errors and unique_base_ids and unique_extra_ids,
        "cumulative_journal_preserves_base_and_uses_unique_sequential_ids": cumulative_path.read_bytes().startswith(base_bytes) and len(cumulative_starts) == len(cumulative_metered) == total_calls and len(set(cumulative_ids)) == total_calls and not cumulative_pending,
        "cumulative_usage_matches_source_journals": abs(cumulative_cost_from_journal - total_cost) < 1e-10 and cumulative_input_from_journal == base_input + extra_input and cumulative_output_from_journal == base_output + extra_output,
        "actual_usage_within_authorized_caps": total_calls <= REQUEST_CAP and len(extra_starts) <= ADDITIONAL_REQUEST_CAP and total_cost <= COST_CAP_USD + 1e-10 and extra_cost <= ADDITIONAL_COST_CAP_USD + 1e-10,
        "authorization_hashes_and_scope_match": run_hashes_ok and auth.get("authorized") is True and auth.get("scope") == "Iteration 19 remaining control slots only",
        "production_reasoning_source_integrity_passes": bool(source_integrity.get("passed")) and source_integrity.get("source_count") == 7,
    }
    if not all(checks.values()):
        raise SystemExit("Continuation audit failed: " + ", ".join(key for key, ok in checks.items() if not ok))

    result = {
        "experiment": manifest["experiment"],
        "status": "COMPLETED",
        "cases": 30,
        "method_runs_expected": len(expected_slots),
        "method_runs_completed": len(completed),
        "continuation_runs_completed": len(extra_runs),
        "paired_repeats_complete": len(paired["paired_rows"]),
        "paired_outcomes_by_role": paired["paired_outcomes_by_role"],
        "case_repeat_consistency": paired["case_repeat_consistency"],
        "condition_metrics_by_role": condition_metrics,
        "candidate_coverage_by_role": candidate_coverage,
        "paired_mean_deltas_by_role": paired_deltas,
        "requests_started_cumulative": len(cumulative_starts),
        "requests_metered_cumulative": len(cumulative_metered),
        "additional_requests_metered": len(extra_metered),
        "input_tokens_cumulative": cumulative_input_from_journal,
        "output_tokens_cumulative": cumulative_output_from_journal,
        "cost_usd_cumulative": cumulative_cost_from_journal,
        "additional_input_tokens": extra_input,
        "additional_output_tokens": extra_output,
        "additional_cost_usd": extra_cost,
        "empty_content_responses_cumulative": all_empty,
        "empty_content_responses_continuation": extra_empty,
        "cumulative_request_cap": REQUEST_CAP,
        "additional_request_cap": ADDITIONAL_REQUEST_CAP,
        "cumulative_cost_cap_usd": COST_CAP_USD,
        "additional_cost_cap_usd": ADDITIONAL_COST_CAP_USD,
        "case_kg_queries_cumulative": int(read_json(OUT / "holdout_results.json")["case_kg_queries"]) + sum(int(row.get("kg_queries", 0) or 0) for row in extra_runs),
        "preflight_kg_queries_cumulative": int(read_json(OUT / "holdout_results.json")["preflight_kg_queries"]) + int(bool(read_json(CONT / "preflight.json").get("freebase_available"))),
        "production_source_integrity": source_integrity,
        "production_algorithm_changed": False,
        "independent_holdout": True,
        "unrun_slots": 0,
        "budget_stop_reason": None,
        "accounting_reconciliation": {
            "original_runner_summary_was_stale": True,
            "cause": "The runner's Meter writes continuation-local events to call_usage.jsonl; the finalizer read cumulative_call_usage.jsonl before those local events were merged.",
            "corrected_offline_without_network": True,
            "actual_caps_respected": True,
            "raw_runner_summary": "continuation_results_runner_output.json",
        },
    }
    write_json(CONT / "paired_analysis_complete.json", paired)
    with (CONT / "paired_runs_cumulative.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_runs:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (CONT / "paired_case_results_complete.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired["paired_rows"][0]) if paired["paired_rows"] else [])
        if paired["paired_rows"]:
            writer.writeheader()
            writer.writerows(paired["paired_rows"])
    write_json(CONT / "continuation_results.json", result)

    audit = {
        "audit_status": "PASS",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "offline_only": True,
        "network_calls": 0,
        "checks": checks,
        "score_mismatches": score_mismatches,
        "duplicate_slots": [list(key) for key, count in slot_counts.items() if count > 1],
        "unexpected_slots": [list(key) for key in set(all_slots) - expected_slots],
        "base_usage": {"requests": len(base_metered), "input_tokens": base_input, "output_tokens": base_output, "cost_usd": base_cost},
        "continuation_usage": {"requests": len(extra_metered), "input_tokens": extra_input, "output_tokens": extra_output, "cost_usd": extra_cost, "empty_content_responses": extra_empty},
        "cumulative_usage": {"requests": total_calls, "input_tokens": base_input + extra_input, "output_tokens": base_output + extra_output, "cost_usd": total_cost, "empty_content_responses": all_empty},
        "caps": {"cumulative_requests": REQUEST_CAP, "additional_requests": ADDITIONAL_REQUEST_CAP, "cumulative_cost_usd": COST_CAP_USD, "additional_cost_usd": ADDITIONAL_COST_CAP_USD},
        "continuation_slots": {"expected": len(missing_initial), "completed": len(extra_runs), "control_role_violations": role_violations, "scope_violations": extra_scope_violations},
        "source_integrity": source_integrity,
        "runner_accounting_note": result["accounting_reconciliation"],
    }
    write_json(CONT / "continuation_offline_audit.json", audit)

    report = f"""# Iteration 19 — completed control continuation

## Result

**Completed:** all 120/120 method slots across 30 frozen CWQ cases, including all 80 target runs from the original execution and the 30 missing short-path control runs. The continuation executed only the authorized control slots; no target slot was rerun. Production reasoning code stayed frozen.

| Cohort | Paired outcome | Count |
| --- | --- | ---:|
| Four-edge targets | Depth 4 only | 13 |
| Four-edge targets | Depth 3 only | 1 |
| Four-edge targets | Tie | 26 |
| Short-path controls | Depth 4 only | 0 |
| Short-path controls | Depth 3 only | 1 |
| Short-path controls | Tie | 19 |

The target cohort's mean paired changes remain **+0.300 Hits@1** and **+0.228 F1** (40 repeated pairs across 20 questions). The completed controls have a mean paired depth4-minus-depth3 change of **−0.050 Hits@1** and **−0.050 F1** (20 pairs across 10 controls). All controls had Hits@1=1.0 at depth 3 (20/20); depth 4 had 19/20. One control regressed on the first repeat and tied on its second; the other nine controls tied on both repeats.

Gold answer IDs appeared in entity-search results on **2/40 depth-3 target runs** and **29/40 depth-4 target runs**. Among controls, the corresponding counts were 16/20 at depth 3 and 15/20 at depth 4. These IDs remained scoring-only and were not included in inference prompts.

## Usage and integrity

- Continuation: **{len(extra_metered)}** metered requests, {extra_input:,} input and {extra_output:,} output tokens, estimated **${extra_cost:.6f}**; {extra_empty} metered empty-content response.
- Cumulative Iteration 19: **{total_calls}** metered requests, {base_input + extra_input:,} input and {base_output + extra_output:,} output tokens, estimated **${total_cost:.6f}**.
- Approved ceilings: {REQUEST_CAP} cumulative / {ADDITIONAL_REQUEST_CAP} additional requests and ${COST_CAP_USD:.2f} cumulative / ${ADDITIONAL_COST_CAP_USD:.6f} additional. Actual totals were below all ceilings.
- SPARQL: {result['case_kg_queries_cumulative']:,} case queries plus {result['preflight_kg_queries_cumulative']} read-only preflights. Frozen production reasoning source integrity passed 7/7.
- Offline audit: all 120 scores recomputed with zero mismatches; slot coverage, scope, metering, caps, hashes, and source integrity passed. No network calls were made by this audit.

## Accounting reconciliation

The runner's raw final summary omitted continuation-local usage because the meter wrote it to `call_usage.jsonl`, while the finalizer read the cumulative journal before merging that local file. The run's raw summary is preserved as `continuation_results_runner_output.json`. This audit deterministically appended the 237 continuation request records to the unchanged base journal, assigned IDs 1001–1237, recomputed totals, and updated the final summary. Actual observed usage was 237 additional requests / ${extra_cost:.6f}, so it remained within the approved 400-request / ${ADDITIONAL_COST_CAP_USD:.6f} additional limits and the 1,400-request / ${COST_CAP_USD:.2f} cumulative limits.

## Interpretation

Depth 4 shows a clear positive signal on this targeted 20-question four-edge cohort, while the completed short-path controls include one observed regression. The control mean is mildly negative and does not support a universal depth-4 default. Repeated runs are not independent questions; this remains a targeted CWQ depth challenge, not a representative benchmark. Keep production TGAR/CoR unchanged pending broader, independently sampled evaluation.

## Artifacts

- `continuation_results.json`: corrected cumulative result summary.
- `paired_analysis_complete.json`, `paired_case_results_complete.csv`, `paired_runs_cumulative.jsonl`: combined paired analysis and traces.
- `continuation_runs.jsonl`, `call_usage.jsonl`, `cumulative_call_usage.jsonl`: continuation-only runs, continuation usage, and cumulative request journal.
- `continuation_offline_audit.json`: offline integrity and accounting audit.
- `continuation_results_runner_output.json`: unmodified raw final summary produced by the runner, retained to document the accounting reconciliation.
"""
    (CONT / "continuation_report.md").write_text(report, encoding="utf-8")

    metadata_path = CONT / "continuation_metadata.json"
    metadata = read_json(metadata_path)
    metadata.update({
        "status": "COMPLETED",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "requests_started_cumulative": total_calls,
        "requests_metered_cumulative": total_calls,
        "requests_metered_continuation": len(extra_metered),
        "input_tokens_cumulative": base_input + extra_input,
        "output_tokens_cumulative": base_output + extra_output,
        "cost_usd_cumulative": total_cost,
        "method_runs_completed_cumulative": len(completed),
        "accounting_reconciled_offline": True,
        "audit_path": str((CONT / "continuation_offline_audit.json").relative_to(ROOT)),
    })
    write_json(metadata_path, metadata)

    hashes = {
        "manifest_sha256": sha(holdout.MANIFEST_PATH),
        "initial_runs_sha256": sha(OUT / "paired_runs.jsonl"),
        "initial_usage_sha256": sha(OUT / "call_usage.jsonl"),
        "initial_missing_slots_sha256": continuation.slots_sha(plan["missing_slots"]),
        "continuation_runner_sha256_at_execution": auth["continuation_runner_sha256"],
        "continuation_runner_sha256_current": sha(ROOT / "scripts/diagnostics/run_fresh50_depth4_control_continuation.py"),
        "continuation_auditor_sha256": sha(Path(__file__).resolve()),
        "continuation_runs_sha256": sha(CONT / "continuation_runs.jsonl"),
        "continuation_usage_sha256": sha(CONT / "call_usage.jsonl"),
        "cumulative_usage_sha256": sha(cumulative_path),
        "combined_runs_sha256": sha(CONT / "paired_runs_cumulative.jsonl"),
        "paired_analysis_sha256": sha(CONT / "paired_analysis_complete.json"),
        "results_sha256": sha(CONT / "continuation_results.json"),
        "offline_audit_sha256": sha(CONT / "continuation_offline_audit.json"),
        "report_sha256": sha(CONT / "continuation_report.md"),
    }
    write_json(CONT / "continuation_hashes.json", hashes)
    print(json.dumps({"status": "PASS", "checks": checks, "result": result, "hashes": hashes}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
