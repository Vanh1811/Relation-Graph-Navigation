#!/usr/bin/env python3
"""Run the frozen paired depth-3/depth-4 CWQ holdout after explicit authorization."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts/diagnostics")]

import run_fresh50_state_lineage as instrumentation  # noqa: E402

WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
OUT = WORK / "RESULTS/iter_019_depth4_holdout_preparation"
MANIFEST_PATH = OUT / "holdout_manifest.json"
INFERENCE_PATH = OUT / "inference_cases.jsonl"
SCORING_PATH = OUT / "scoring_gold_only.jsonl"
AUTH_PATH = OUT / "run_authorization.json"
REQUEST_CAP = 1000
COST_CAP_USD = 0.70
MODEL = "gpt-5.6-luna"
TIMEOUT_SECONDS = 120
MAX_COMPLETION_TOKENS = 1024


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest = read_json(MANIFEST_PATH)
    if manifest.get("status") != "FROZEN_PREPARED_NOT_RUN":
        raise SystemExit("Holdout manifest is not in the expected frozen, unrun state")
    inference = read_jsonl(INFERENCE_PATH)
    scoring_rows = read_jsonl(SCORING_PATH)
    score = {row["question_key"]: row for row in scoring_rows}
    schedule = manifest["treatment_protocol"]["randomized_run_schedule"]
    keys = manifest["case_keys_scoring_only"]
    if len(keys) != 30 or len(set(keys)) != 30 or len(inference) != 30 or set(keys) != {row["question_key"] for row in inference}:
        raise SystemExit("Expected 30 unique, aligned inference/scoring cases")
    if len(score) != 30 or set(keys) != set(score):
        raise SystemExit("Scoring sidecar does not align exactly with the frozen case list")
    if len(schedule) != 120 or {(row["question_key"], row["repeat_index"], row["depth"]) for row in schedule} != {
        (key, repeat, depth) for key in keys for repeat in (1, 2) for depth in (3, 4)
    }:
        raise SystemExit("Frozen schedule must contain exactly two paired repeats per depth and case")
    allowed_inference_fields = {"question_key", "case_id", "dataset", "split", "question", "topic_entities", "topic_entity_names"}
    if any(set(row) - allowed_inference_fields for row in inference):
        raise SystemExit("Inference file contains unexpected fields; gold or cohort metadata may have leaked")
    if len({row["question_key"] for row in inference}) != len(inference):
        raise SystemExit("Duplicate inference case key")
    if any(row.get("cohort_role_scoring_only") not in {"STRONG_DEPTH4_TARGET", "STRONG_SHORTER_PATH_NEGATIVE_CONTROL"} for row in scoring_rows):
        raise SystemExit("Unexpected scoring-only cohort label")
    if sum(row["cohort_role_scoring_only"] == "STRONG_DEPTH4_TARGET" for row in scoring_rows) != 20:
        raise SystemExit("Expected exactly 20 targeted depth-challenge cases")
    if sum(row["cohort_role_scoring_only"] == "STRONG_SHORTER_PATH_NEGATIVE_CONTROL" for row in scoring_rows) != 10:
        raise SystemExit("Expected exactly 10 shorter-path negative controls")
    for relative, expected in manifest.get("input_hashes", {}).items():
        path = ROOT / relative
        if not path.exists() or sha(path) != expected:
            raise SystemExit(f"Frozen input hash mismatch: {relative}")
    return manifest, inference, score


def paired_summary(runs: list[dict[str, Any]], scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    for run in runs:
        cells[(run["question_key"], int(run["repeat_index"]), int(run["diagnostic_depth"]))] = run
    pair_rows = []
    role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for key, scoring in scores.items():
        role = scoring["cohort_role_scoring_only"]
        for repeat in (1, 2):
            depth3 = cells.get((key, repeat, 3))
            depth4 = cells.get((key, repeat, 4))
            if not depth3 or not depth4:
                continue
            h3, h4 = int(depth3.get("hits_at_1", 0)), int(depth4.get("hits_at_1", 0))
            outcome = "TIE" if h3 == h4 else "DEPTH4_ONLY" if h4 else "DEPTH3_ONLY"
            role_counts[role][outcome] += 1
            pair_rows.append({
                "question_key": key, "cohort_role_scoring_only": role, "repeat_index": repeat,
                "depth3_hits_at_1": h3, "depth4_hits_at_1": h4, "paired_outcome": outcome,
                "depth3_f1": float(depth3.get("f1", 0.0)), "depth4_f1": float(depth4.get("f1", 0.0)),
                "depth3_requests": depth3.get("request_count_case", 0), "depth4_requests": depth4.get("request_count_case", 0),
            })
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_case[row["question_key"]].append(row)
    consistency = Counter()
    per_case = []
    for key, rows in sorted(by_case.items()):
        outcomes = [row["paired_outcome"] for row in sorted(rows, key=lambda item: item["repeat_index"])]
        if len(outcomes) == 2:
            consistency["BOTH_DEPTH4_ONLY"] += outcomes == ["DEPTH4_ONLY", "DEPTH4_ONLY"]
            consistency["BOTH_DEPTH3_ONLY"] += outcomes == ["DEPTH3_ONLY", "DEPTH3_ONLY"]
            consistency["MIXED_DIRECTION"] += len(set(outcomes) - {"TIE"}) > 1
            consistency["NO_DEPTH4_ONLY"] += "DEPTH4_ONLY" not in outcomes
        per_case.append({"question_key": key, "repeat_outcomes": outcomes})
    return {
        "paired_rows": pair_rows,
        "paired_outcomes_by_role": {role: dict(counts) for role, counts in role_counts.items()},
        "case_repeat_consistency": dict(consistency),
        "per_case_repeat_outcomes": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true", help="validate frozen inputs without network or model calls")
    action.add_argument("--run", action="store_true", help="run only after a matching authorization record is present")
    action.add_argument("--resume", action="store_true", help="resume after review; refuses any incomplete prior run")
    args = parser.parse_args()
    manifest, inference, scores = validate_inputs()
    expected_inputs = {"holdout_manifest.json": sha(MANIFEST_PATH), "inference_cases.jsonl": sha(INFERENCE_PATH), "scoring_gold_only.jsonl": sha(SCORING_PATH)}
    if args.dry_run:
        instrumentation.OUT = OUT
        integrity = instrumentation.production_source_integrity()
        if not integrity.get("passed"):
            raise SystemExit(f"Frozen production source hashes failed: {integrity.get('failed_paths')}")
        print(json.dumps({
            "status": "DRY_RUN_PASS", "cases": len(inference), "depths": [3, 4],
            "repeats_per_depth": 2, "method_runs": len(manifest["treatment_protocol"]["randomized_run_schedule"]),
            "gold_fields_in_inference": False, "source_hashes": "pass",
            "production_reasoning_integrity": integrity,
            "request_cap": REQUEST_CAP, "cost_cap_usd": COST_CAP_USD, "network_calls": 0,
        }, indent=2))
        return

    if not AUTH_PATH.exists():
        raise SystemExit("Missing run_authorization.json; this holdout has no paid-run authorization")
    authorization = read_json(AUTH_PATH)
    if authorization.get("authorized") is not True or authorization.get("request_cap") != REQUEST_CAP or authorization.get("cost_cap_usd") != COST_CAP_USD:
        raise SystemExit("Run authorization must explicitly match the frozen 1000-request/$0.70 caps")
    if (authorization.get("model") != MODEL
            or authorization.get("max_completion_tokens") != MAX_COMPLETION_TOKENS
            or authorization.get("timeout_seconds") != TIMEOUT_SECONDS
            or authorization.get("retries") != 0
            or authorization.get("holdout_manifest_sha256") != sha(MANIFEST_PATH)):
        raise SystemExit("Run authorization must match the model, token/time limits, retry policy, and frozen cohort manifest")

    instrumentation.load_safe_project_env()
    if os.environ.get("MODEL_NAME") != MODEL or not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(f"Expected MODEL_NAME={MODEL} and a configured API key")
    base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    if "api.openai.com" not in base_url:
        raise SystemExit("This budget is for direct OpenAI API requests only")
    calls_cap = int(os.environ.get("RESEARCH_MAX_LLM_CALLS", "0"))
    cost_cap = float(os.environ.get("RESEARCH_MAX_LLM_COST_USD", "0"))
    if calls_cap != REQUEST_CAP or abs(cost_cap - COST_CAP_USD) > 1e-9:
        raise SystemExit("Environment budget must exactly match the separately authorized caps")
    os.environ["OPENAI_MAX_RETRIES"] = "1"  # one logical attempt; SDK retries are disabled in LLMAPI
    os.environ["OPENAI_TIMEOUT"] = str(TIMEOUT_SECONDS)
    os.environ["OPENAI_RETRY_INTERVAL"] = "0"
    os.environ.setdefault("OPENAI_REASONING_EFFORT", "low")

    run_path = OUT / "paired_runs.jsonl"
    usage_path = OUT / "call_usage.jsonl"
    metadata_path = OUT / "run_metadata.json"
    if args.run and any(path.exists() for path in (run_path, usage_path, metadata_path)):
        raise SystemExit("Run artifacts already exist; use --resume only after reviewing journals")
    if args.resume and not metadata_path.exists():
        raise SystemExit("Cannot resume: run metadata is missing")

    instrumentation.OUT = OUT
    instrumentation.MAX_REQUESTS = REQUEST_CAP
    instrumentation.MAX_COST_USD = COST_CAP_USD
    integrity_before = instrumentation.production_source_integrity()
    if not integrity_before.get("passed"):
        raise SystemExit(f"Frozen production source hashes failed: {integrity_before.get('failed_paths')}")
    metadata = {
        "experiment": manifest["experiment"], "status": "RUNNING", "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "method_runs_expected": 120, "request_cap": REQUEST_CAP, "cost_cap_usd": COST_CAP_USD,
        "max_completion_tokens_per_request": MAX_COMPLETION_TOKENS, "timeout_seconds": TIMEOUT_SECONDS,
        "logical_attempts_per_request": 1, "retries": 0, "openai_base_url_host": base_url.split("/")[2],
        "authorization": authorization, "frozen_input_hashes": expected_inputs,
        "production_source_integrity_before": integrity_before, "independent_holdout": True,
        "inference_gold_fields": False,
    }
    if args.run:
        write_json(metadata_path, metadata)
    else:
        metadata = read_json(metadata_path)
        if metadata.get("frozen_input_hashes") != expected_inputs or metadata.get("request_cap") != REQUEST_CAP or metadata.get("cost_cap_usd") != COST_CAP_USD:
            raise SystemExit("Resume inputs or caps do not match the original run")

    preflight = instrumentation.endpoint_preflight()
    write_json(OUT / "preflight_holdout.json", preflight)
    if not preflight.get("ready_for_run"):
        metadata.update({"status": "BLOCKED_PREFLIGHT", "preflight": preflight})
        write_json(metadata_path, metadata)
        raise SystemExit("API or Freebase preflight failed; no case runs submitted")

    existing = read_jsonl(run_path) if args.resume and run_path.exists() else []
    if any(row.get("case_status") != "COMPLETED" for row in existing):
        raise SystemExit("An incomplete case/run exists; review before any further paid requests")
    completed_slots = {(row["question_key"], int(row["repeat_index"]), int(row["diagnostic_depth"])) for row in existing}
    allowed_slots = {(row["question_key"], int(row["repeat_index"]), int(row["depth"])) for row in manifest["treatment_protocol"]["randomized_run_schedule"]}
    if not completed_slots.issubset(allowed_slots):
        raise SystemExit("Existing journal contains an unexpected run slot")
    inference_by_key = {row["question_key"]: row for row in inference}
    all_runs = existing[:]
    budget_stop: str | None = None
    for schedule_index, slot in enumerate(manifest["treatment_protocol"]["randomized_run_schedule"], start=1):
        key, repeat, depth = slot["question_key"], int(slot["repeat_index"]), int(slot["depth"])
        if (key, repeat, depth) in completed_slots:
            continue
        infer_case = dict(inference_by_key[key])
        infer_case["question_id"] = infer_case["case_id"]
        # Gold is joined only for the scorer called after inference; it is never
        # included in the question, topic entities, prompt history, or router input.
        infer_case["gold_answers"] = list(scores[key].get("gold_answers", []))
        agent = instrumentation.make_agent("v1_1_3")
        initial_depth = int(agent.cor_agent.depth)
        if initial_depth != 3:
            raise RuntimeError(f"Expected frozen TGAR depth=3, got {initial_depth}")
        agent.cor_agent.depth = depth
        events: list[dict[str, Any]] = []
        meter = instrumentation.install_state_lineage(agent, events)
        original_guard = meter.guarded_generate

        def token_bounded_guard(original: Any, user_prompt: str, temperature: float = 0.01,
                                max_tokens: int = 256, system_prompt: str | None = None):
            if int(max_tokens) > MAX_COMPLETION_TOKENS:
                raise instrumentation.BudgetStop(
                    f"request completion-token limit {max_tokens} exceeds frozen cap {MAX_COMPLETION_TOKENS}"
                )
            return original_guard(original, user_prompt, temperature, max_tokens, system_prompt)

        meter.guarded_generate = token_bounded_guard
        request_start = meter.calls
        run = instrumentation.run_one(agent, "v1_1_3", infer_case, schedule_index)
        run.update({
            "question_key": key, "repeat_index": repeat, "diagnostic_depth": depth,
            "diagnostic_depth_before": initial_depth, "cohort_role_scoring_only": scores[key]["cohort_role_scoring_only"],
            "state_lineage": events, "request_count_case": meter.calls - request_start,
            "case_status": "COMPLETED" if not run.get("error") else "INCOMPLETE_ERROR",
        })
        run["gold_candidate_diagnostic_scoring_only"] = instrumentation.case_candidate_summary(run, set(map(str, scores[key].get("answer_ids", []))))
        instrumentation.append_jsonl(run_path, run)
        all_runs.append(run)
        if run["case_status"] != "COMPLETED":
            budget_stop = run.get("error") or "case incomplete"
            break
        completed_slots.add((key, repeat, depth))
        print(json.dumps({"slot": schedule_index, "slots_total": len(allowed_slots), "question_key": key,
                          "repeat": repeat, "depth": depth, "requests": meter.calls,
                          "cost_usd": round(meter.spent, 7), "status": run["case_status"]}), flush=True)
        if meter.halted:
            budget_stop = meter.unmetered_error or "request or cost cap reached"
            break

    all_runs = read_jsonl(run_path) if run_path.exists() else all_runs
    usage = read_jsonl(usage_path) if usage_path.exists() else []
    metered = [row for row in usage if row.get("status") == "REQUEST_METERED"]
    starts = [row for row in usage if row.get("status") == "REQUEST_STARTED"]
    paired = paired_summary([row for row in all_runs if row.get("case_status") == "COMPLETED"], scores)
    write_json(OUT / "paired_analysis.json", paired)
    write_csv_rows = paired["paired_rows"]
    if write_csv_rows:
        import csv
        with (OUT / "paired_case_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(write_csv_rows[0]))
            writer.writeheader()
            writer.writerows(write_csv_rows)
    total_input = sum(int(row.get("input_tokens", 0) or 0) for row in metered)
    total_output = sum(int(row.get("output_tokens", 0) or 0) for row in metered)
    total_cost = sum(float(row.get("cost_usd", 0) or 0) for row in metered)
    case_queries = sum(int(row.get("kg_queries", 0) or 0) for row in all_runs)
    status = "COMPLETED" if len(completed_slots) == len(allowed_slots) else "INCOMPLETE"
    integrity_after = instrumentation.production_source_integrity()
    condition_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    for role in ("STRONG_DEPTH4_TARGET", "STRONG_SHORTER_PATH_NEGATIVE_CONTROL"):
        condition_metrics[role] = {}
        role_runs = [row for row in all_runs if row.get("cohort_role_scoring_only") == role]
        for depth in (3, 4):
            group = [row for row in role_runs if int(row.get("diagnostic_depth", 0)) == depth]
            condition_metrics[role][str(depth)] = {
                "runs": len(group),
                "mean_hits_at_1": sum(int(row.get("hits_at_1", 0)) for row in group) / len(group) if group else 0.0,
                "mean_f1": sum(float(row.get("f1", 0.0)) for row in group) / len(group) if group else 0.0,
            }
    pair_delta_by_role: dict[str, dict[str, float | int]] = {}
    for role, rows in ((role, [r for r in paired["paired_rows"] if r["cohort_role_scoring_only"] == role])
                       for role in ("STRONG_DEPTH4_TARGET", "STRONG_SHORTER_PATH_NEGATIVE_CONTROL")):
        pair_delta_by_role[role] = {
            "paired_repeats": len(rows),
            "mean_hits_at_1_delta_depth4_minus_depth3": sum(r["depth4_hits_at_1"] - r["depth3_hits_at_1"] for r in rows) / len(rows) if rows else 0.0,
            "mean_f1_delta_depth4_minus_depth3": sum(r["depth4_f1"] - r["depth3_f1"] for r in rows) / len(rows) if rows else 0.0,
        }
    results = {
        "experiment": manifest["experiment"], "status": status,
        "cases": len(inference), "method_runs_expected": len(allowed_slots), "method_runs_completed": len(completed_slots),
        "paired_repeats_complete": len(paired["paired_rows"]), "paired_outcomes_by_role": paired["paired_outcomes_by_role"],
        "case_repeat_consistency": paired["case_repeat_consistency"],
        "condition_metrics_by_role": condition_metrics, "paired_mean_deltas_by_role": pair_delta_by_role,
        "requests_started": len(starts), "metered_requests": len(metered),
        "input_tokens": total_input, "output_tokens": total_output, "cost_usd": total_cost,
        "request_cap": REQUEST_CAP, "cost_cap_usd": COST_CAP_USD,
        "case_kg_queries": case_queries, "preflight_kg_queries": 1,
        "budget_stop_reason": budget_stop, "production_source_integrity_after": integrity_after,
        "production_algorithm_changed": False, "independent_holdout": True,
        "interpretation_limit": "A targeted census of eligible four-edge CWQ cases with within-pool shorter-path negative controls; results do not estimate broad KGQA performance.",
    }
    write_json(OUT / "holdout_results.json", results)
    metadata.update({"status": status, "preflight": preflight, "requests_started": len(starts),
                    "metered_requests": len(metered), "input_tokens": total_input, "output_tokens": total_output,
                    "cost_usd": total_cost, "case_kg_queries": case_queries,
                    "production_source_integrity_after": integrity_after, "budget_stop_reason": budget_stop})
    write_json(metadata_path, metadata)
    lines = [
        "# Iteration 19 — independent depth-4 holdout results", "",
        "## Run status", "",
        f"Status: **{status}**. Completed **{len(completed_slots)}/120** method runs across 30 unused CWQ cases, two depths, and two repeats.",
        f"API usage: **{len(starts)} requests**, {total_input:,} input / {total_output:,} output tokens, estimated **${total_cost:.6f}**. SPARQL: {case_queries} case queries plus the preflight. Frozen production source integrity: {integrity_after.get('source_count', 0)}/7.",
        "", "## Depth 4 targets", "",
        f"Paired repeat outcomes: `{paired['paired_outcomes_by_role'].get('STRONG_DEPTH4_TARGET', {})}`. Mean paired Hits@1 delta: {pair_delta_by_role['STRONG_DEPTH4_TARGET']['mean_hits_at_1_delta_depth4_minus_depth3']:.3f}; mean paired F1 delta: {pair_delta_by_role['STRONG_DEPTH4_TARGET']['mean_f1_delta_depth4_minus_depth3']:.3f}.",
        f"Depth 3 metrics: `{condition_metrics['STRONG_DEPTH4_TARGET']['3']}`. Depth 4 metrics: `{condition_metrics['STRONG_DEPTH4_TARGET']['4']}`.",
        "", "## Shorter-path negative controls", "",
        f"Paired repeat outcomes: `{paired['paired_outcomes_by_role'].get('STRONG_SHORTER_PATH_NEGATIVE_CONTROL', {})}`. Mean paired Hits@1 delta: {pair_delta_by_role['STRONG_SHORTER_PATH_NEGATIVE_CONTROL']['mean_hits_at_1_delta_depth4_minus_depth3']:.3f}; mean paired F1 delta: {pair_delta_by_role['STRONG_SHORTER_PATH_NEGATIVE_CONTROL']['mean_f1_delta_depth4_minus_depth3']:.3f}.",
        f"Depth 3 metrics: `{condition_metrics['STRONG_SHORTER_PATH_NEGATIVE_CONTROL']['3']}`. Depth 4 metrics: `{condition_metrics['STRONG_SHORTER_PATH_NEGATIVE_CONTROL']['4']}`.",
        "", "## Interpretation limits", "",
        "The 20 target cases are the available, previously unused CWQ strongly-branch-required cases whose gold query graph places the answer four edges from at least one linked topic. The 10 negative controls are sampled from the same strong pool at three or fewer edges. This is a targeted depth challenge, not a representative sample of CWQ or KGQA. Repeats expose model variability but do not make individual calls independent observations. Treat results as evidence about this frozen cohort only; do not claim broad benchmark superiority from this design.",
        "", "Question-level paired outcomes and traces are in `paired_analysis.json` and `paired_runs.jsonl`; request accounting is in `call_usage.jsonl`; input, output, and source hashes are recorded in `run_metadata.json` and `artifact_hashes.json`.", "",
    ]
    (OUT / "holdout_report.md").write_text("\n".join(lines), encoding="utf-8")
    artifacts = {path.name: sha(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "artifact_hashes.json"}
    artifacts["runner_source_sha256"] = sha(Path(__file__))
    artifacts["instrumentation_source_sha256"] = sha(Path(instrumentation.__file__))
    write_json(OUT / "artifact_hashes.json", artifacts)
    if status != "COMPLETED":
        print(json.dumps(results, indent=2), flush=True)
        raise SystemExit("Holdout is incomplete; report and partial evidence were saved without a completion claim")
    if not integrity_after.get("passed"):
        raise SystemExit("Frozen production source integrity mismatch after holdout")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
