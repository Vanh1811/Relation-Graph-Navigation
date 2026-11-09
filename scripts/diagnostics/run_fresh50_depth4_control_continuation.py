#!/usr/bin/env python3
"""Continue only Iteration 19's unrun short-path control slots.

The initial holdout journals remain immutable. This runner copies the metered
request journal into a continuation workspace and applies new caps to cumulative
usage, so request IDs and spend remain continuous across both runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts/diagnostics")]

import run_fresh50_depth4_holdout as holdout  # noqa: E402
import run_fresh50_state_lineage as instrumentation  # noqa: E402

OUT = holdout.OUT
CONT_OUT = OUT / "control_continuation"
AUTH_PATH = OUT / "control_continuation_authorization.json"
RUN_PATH = CONT_OUT / "continuation_runs.jsonl"
USAGE_PATH = CONT_OUT / "cumulative_call_usage.jsonl"
METADATA_PATH = CONT_OUT / "continuation_metadata.json"
MODEL = holdout.MODEL
TIMEOUT_SECONDS = holdout.TIMEOUT_SECONDS
MAX_COMPLETION_TOKENS = holdout.MAX_COMPLETION_TOKENS
CONTROL_ROLE = "STRONG_SHORTER_PATH_NEGATIVE_CONTROL"
TARGET_ROLE = "STRONG_DEPTH4_TARGET"
RECOMMENDED_ADDITIONAL_REQUEST_CAP = 400
RECOMMENDED_CUMULATIVE_COST_CAP_USD = 0.82


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    instrumentation.append_jsonl(path, value)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slots_sha(slots: list[dict[str, Any]]) -> str:
    normalized = [
        {"question_key": row["question_key"], "repeat_index": int(row["repeat_index"]), "depth": int(row["depth"])}
        for row in slots
    ]
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def slot_key(row: dict[str, Any], *, depth_field: str) -> tuple[str, int, int]:
    return row["question_key"], int(row["repeat_index"]), int(row[depth_field])


def inspect_plan() -> dict[str, Any]:
    manifest, inference, scores = holdout.validate_inputs()
    base_runs = read_jsonl(OUT / "paired_runs.jsonl")
    base_usage = read_jsonl(OUT / "call_usage.jsonl")
    base_results = read_json(OUT / "holdout_results.json")
    base_audit = read_json(OUT / "offline_audit.json")

    if base_results.get("status") != "INCOMPLETE_BUDGET_CAPPED":
        raise SystemExit("The original holdout must remain in its verified budget-capped state")
    if base_audit.get("audit_status") != "PASS" or not all(base_audit.get("checks", {}).values()):
        raise SystemExit("The initial holdout offline audit must pass before continuation")
    if any(row.get("case_status") != "COMPLETED" for row in base_runs):
        raise SystemExit("The initial run contains a non-completed slot; review it before continuing")

    schedule = manifest["treatment_protocol"]["randomized_run_schedule"]
    allowed = {slot_key(row, depth_field="depth") for row in schedule}
    base_done = {slot_key(row, depth_field="diagnostic_depth") for row in base_runs}
    if len(base_done) != len(base_runs) or not base_done <= allowed:
        raise SystemExit("Initial run journal has duplicate or unexpected slots")

    missing = [
        row for row in schedule
        if slot_key(row, depth_field="depth") not in base_done
        and scores[row["question_key"]]["cohort_role_scoring_only"] == CONTROL_ROLE
    ]
    non_control_missing = [
        row for row in schedule
        if slot_key(row, depth_field="depth") not in base_done
        and scores[row["question_key"]]["cohort_role_scoring_only"] != CONTROL_ROLE
    ]
    if len(base_done) != 90 or len(missing) != 30 or non_control_missing:
        raise SystemExit(
            f"Expected exactly 90 completed slots and 30 missing control slots; got {len(base_done)} and {len(missing)}"
        )
    target_slots = [row for row in base_runs if row.get("cohort_role_scoring_only") == TARGET_ROLE]
    if len(target_slots) != 80:
        raise SystemExit("All 80 target runs must remain completed before a control-only continuation")

    starts = [row for row in base_usage if row.get("status") == "REQUEST_STARTED"]
    metered = [row for row in base_usage if row.get("status") == "REQUEST_METERED"]
    finished_ids = {row.get("request_id") for row in base_usage if row.get("status") in {"REQUEST_METERED", "REQUEST_UNMETERED_ERROR"}}
    pending = [row.get("request_id") for row in starts if row.get("request_id") not in finished_ids]
    if pending or len(starts) != 1000 or len(metered) != 1000:
        raise SystemExit("Initial request journal is not fully metered at its 1,000-request cap")
    if len({row.get("request_id") for row in starts}) != len(starts):
        raise SystemExit("Initial request journal has duplicate request IDs")

    previous_cost = sum(float(row.get("cost_usd", 0.0) or 0.0) for row in metered)
    control_runs = [row for row in base_runs if row.get("cohort_role_scoring_only") == CONTROL_ROLE]
    control_calls = sum(int(row.get("request_count_case", 0) or 0) for row in control_runs)
    control_input = sum(int(row.get("input_tokens", 0) or 0) for row in control_runs)
    control_output = sum(int(row.get("output_tokens", 0) or 0) for row in control_runs)
    control_cost = control_input * instrumentation.INPUT_PRICE / 1_000_000 + control_output * instrumentation.OUTPUT_PRICE / 1_000_000
    missing_hash = slots_sha(missing)
    hashes = {
        "manifest_sha256": sha(holdout.MANIFEST_PATH),
        "inference_sha256": sha(holdout.INFERENCE_PATH),
        "scoring_sha256": sha(holdout.SCORING_PATH),
        "base_paired_runs_sha256": sha(OUT / "paired_runs.jsonl"),
        "base_call_usage_sha256": sha(OUT / "call_usage.jsonl"),
        "initial_missing_slots_sha256": missing_hash,
        "continuation_runner_sha256": sha(Path(__file__).resolve()),
    }
    integrity = instrumentation.production_source_integrity()
    if not integrity.get("passed"):
        raise SystemExit(f"Production source integrity failed: {integrity.get('failed_paths')}")

    return {
        "manifest": manifest,
        "inference": inference,
        "scores": scores,
        "base_runs": base_runs,
        "base_usage": base_usage,
        "base_results": base_results,
        "missing_slots": missing,
        "hashes": hashes,
        "production_integrity": integrity,
        "base_calls": len(starts),
        "base_cost_usd": previous_cost,
        "base_case_queries": sum(int(row.get("kg_queries", 0) or 0) for row in base_runs),
        "control_runs_completed": len(control_runs),
        "control_requests_observed": control_calls,
        "control_cost_observed_usd": control_cost,
        "estimated_additional_requests": round(control_calls / len(control_runs) * len(missing)) if control_runs else None,
        "estimated_additional_cost_usd": control_cost / len(control_runs) * len(missing) if control_runs else None,
    }


def validate_authorization(plan: dict[str, Any], *, cumulative_calls: int, cumulative_cost: float) -> dict[str, Any]:
    if not AUTH_PATH.exists():
        raise SystemExit("No continuation authorization exists. The 1,000-request/$0.70 initial authorization is exhausted.")
    auth = read_json(AUTH_PATH)
    exact_fields = {
        **plan["hashes"],
        "base_requests": plan["base_calls"],
        "base_cost_usd": round(plan["base_cost_usd"], 10),
        "model": MODEL,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "retries": 0,
    }
    for key, expected in exact_fields.items():
        if auth.get(key) != expected:
            raise SystemExit(f"Continuation authorization mismatch for {key}")
    if auth.get("authorized") is not True or auth.get("scope") != "Iteration 19 remaining control slots only":
        raise SystemExit("Continuation authorization does not explicitly cover only the missing control slots")
    if not auth.get("authorized_at_utc") or not auth.get("authorized_by"):
        raise SystemExit("Continuation authorization must record who approved it and when")
    if int(auth.get("cumulative_request_cap", 0)) <= cumulative_calls:
        raise SystemExit("Authorized cumulative request cap must exceed the already metered request count")
    if float(auth.get("cumulative_cost_cap_usd", 0)) <= cumulative_cost:
        raise SystemExit("Authorized cumulative cost cap must exceed the already metered spend")
    if int(auth["cumulative_request_cap"]) - cumulative_calls > int(auth.get("additional_request_cap", -1)):
        raise SystemExit("Additional request cap does not reconcile with the cumulative cap")
    if abs(float(auth["cumulative_cost_cap_usd"]) - cumulative_cost - float(auth.get("additional_cost_cap_usd", -1))) > 1e-8:
        raise SystemExit("Additional cost cap does not reconcile with cumulative authorized spend")
    return auth


def validate_cumulative_journal(plan: dict[str, Any], *, initialize: bool) -> list[dict[str, Any]]:
    base_bytes = (OUT / "call_usage.jsonl").read_bytes()
    if initialize:
        if CONT_OUT.exists():
            raise SystemExit("Continuation workspace already exists; use --resume after reviewing it")
        CONT_OUT.mkdir(parents=True)
        USAGE_PATH.write_bytes(base_bytes)
    elif not USAGE_PATH.exists():
        raise SystemExit("Continuation cumulative usage journal is missing")
    if not USAGE_PATH.read_bytes().startswith(base_bytes):
        raise SystemExit("Cumulative continuation journal does not preserve the original usage journal byte-for-byte")
    usage = read_jsonl(USAGE_PATH)
    started = [row for row in usage if row.get("status") == "REQUEST_STARTED"]
    finished = {row.get("request_id") for row in usage if row.get("status") in {"REQUEST_METERED", "REQUEST_UNMETERED_ERROR"}}
    if any(row.get("request_id") not in finished for row in started):
        raise SystemExit("Cumulative journal has an unresolved request; refusing to continue")
    if len({row.get("request_id") for row in started}) != len(started):
        raise SystemExit("Cumulative journal contains duplicate request IDs")
    return usage


def merge_continuation_usage(plan: dict[str, Any]) -> None:
    """Append the local continuation meter to the immutable base journal."""
    base_path = OUT / "call_usage.jsonl"
    base_bytes = base_path.read_bytes()
    base_starts = [row for row in plan["base_usage"] if row.get("status") == "REQUEST_STARTED"]
    base_metered = [row for row in plan["base_usage"] if row.get("status") == "REQUEST_METERED"]
    base_cost = sum(float(row.get("cost_usd", 0.0) or 0.0) for row in base_metered)
    local_rows = read_jsonl(CONT_OUT / "call_usage.jsonl") if (CONT_OUT / "call_usage.jsonl").exists() else []

    local_ids: dict[str, str] = {}
    for row in local_rows:
        if row.get("status") == "REQUEST_STARTED":
            local_id = str(row["request_id"])
            local_ids[local_id] = f"request_{len(base_starts) + int(local_id.rsplit('_', 1)[1]):04d}"

    additional_cost = 0.0
    merged_rows = []
    for original in local_rows:
        row = dict(original)
        local_id = str(row["request_id"])
        if local_id not in local_ids:
            raise RuntimeError(f"Continuation usage row has no request start: {local_id}")
        row["continuation_local_request_id"] = local_id
        row["request_id"] = local_ids[local_id]
        row["continuation_added"] = True
        if row.get("status") == "REQUEST_METERED":
            additional_cost += float(row.get("cost_usd", 0.0) or 0.0)
            row["cumulative_cost_usd"] = base_cost + additional_cost
        merged_rows.append(row)

    prefix = base_bytes if base_bytes.endswith(b"\n") else base_bytes + b"\n"
    expected = prefix + b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in merged_rows
    )
    current = USAGE_PATH.read_bytes()
    if current == base_bytes:
        USAGE_PATH.write_bytes(expected)
    elif current != expected:
        raise RuntimeError("Cumulative usage journal is neither the frozen base nor its exact continuation merge")


def finalize(plan: dict[str, Any], authorization: dict[str, Any], metadata: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    base_runs = plan["base_runs"]
    continuation_runs = read_jsonl(RUN_PATH) if RUN_PATH.exists() else []
    all_runs = base_runs + continuation_runs
    scores = plan["scores"]
    completed_runs = [row for row in all_runs if row.get("case_status") == "COMPLETED"]
    paired = holdout.paired_summary(completed_runs, scores)
    cumulative_usage = read_jsonl(USAGE_PATH) if USAGE_PATH.exists() else plan["base_usage"]
    started = [row for row in cumulative_usage if row.get("status") == "REQUEST_STARTED"]
    metered = [row for row in cumulative_usage if row.get("status") == "REQUEST_METERED"]
    total_input = sum(int(row.get("input_tokens", 0) or 0) for row in metered)
    total_output = sum(int(row.get("output_tokens", 0) or 0) for row in metered)
    total_cost = sum(float(row.get("cost_usd", 0.0) or 0.0) for row in metered)
    complete_slots = {
        (row["question_key"], int(row["repeat_index"]), int(row["diagnostic_depth"]))
        for row in all_runs if row.get("case_status") == "COMPLETED"
    }
    expected = len(plan["manifest"]["treatment_protocol"]["randomized_run_schedule"])
    continuation_complete = len(complete_slots) == expected
    stop_reason = metadata.get("budget_stop_reason")
    if continuation_complete:
        status = "COMPLETED"
    elif stop_reason and "cap" not in str(stop_reason).lower():
        status = "INCOMPLETE_ERROR"
    else:
        status = "INCOMPLETE_BUDGET_CAPPED"
    case_queries = sum(int(row.get("kg_queries", 0) or 0) for row in all_runs)
    condition_metrics = {}
    paired_deltas = {}
    for role in (TARGET_ROLE, CONTROL_ROLE):
        condition_metrics[role] = {}
        for depth in (3, 4):
            group = [row for row in all_runs if row.get("cohort_role_scoring_only") == role and int(row.get("diagnostic_depth", 0)) == depth and row.get("case_status") == "COMPLETED"]
            condition_metrics[role][str(depth)] = {
                "runs": len(group),
                "mean_hits_at_1": sum(int(row.get("hits_at_1", 0)) for row in group) / len(group) if group else 0.0,
                "mean_f1": sum(float(row.get("f1", 0.0)) for row in group) / len(group) if group else 0.0,
            }
        pairs = [row for row in paired["paired_rows"] if row["cohort_role_scoring_only"] == role]
        paired_deltas[role] = {
            "paired_repeats": len(pairs),
            "mean_hits_at_1_delta_depth4_minus_depth3": sum(row["depth4_hits_at_1"] - row["depth3_hits_at_1"] for row in pairs) / len(pairs) if pairs else 0.0,
            "mean_f1_delta_depth4_minus_depth3": sum(row["depth4_f1"] - row["depth3_f1"] for row in pairs) / len(pairs) if pairs else 0.0,
        }

    write_json(CONT_OUT / "paired_analysis_complete.json", paired)
    with (CONT_OUT / "paired_runs_cumulative.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_runs:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (CONT_OUT / "paired_case_results_complete.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired["paired_rows"][0]) if paired["paired_rows"] else [])
        if paired["paired_rows"]:
            writer.writeheader()
            writer.writerows(paired["paired_rows"])

    initial_preflight = int(plan["base_results"].get("preflight_kg_queries", 1))
    continuation_preflight = int(preflight.get("freebase_available", False))
    results = {
        "experiment": plan["manifest"]["experiment"],
        "status": status,
        "cases": 30,
        "method_runs_expected": expected,
        "method_runs_completed": len(complete_slots),
        "continuation_runs_completed": sum(row.get("case_status") == "COMPLETED" for row in continuation_runs),
        "paired_repeats_complete": len(paired["paired_rows"]),
        "paired_outcomes_by_role": paired["paired_outcomes_by_role"],
        "case_repeat_consistency": paired["case_repeat_consistency"],
        "condition_metrics_by_role": condition_metrics,
        "paired_mean_deltas_by_role": paired_deltas,
        "requests_started_cumulative": len(started),
        "requests_metered_cumulative": len(metered),
        "additional_requests_metered": len(metered) - plan["base_calls"],
        "input_tokens_cumulative": total_input,
        "output_tokens_cumulative": total_output,
        "cost_usd_cumulative": total_cost,
        "additional_cost_usd": total_cost - plan["base_cost_usd"],
        "cumulative_request_cap": authorization["cumulative_request_cap"],
        "cumulative_cost_cap_usd": authorization["cumulative_cost_cap_usd"],
        "case_kg_queries_cumulative": case_queries,
        "preflight_kg_queries_cumulative": initial_preflight + continuation_preflight,
        "production_source_integrity": instrumentation.production_source_integrity(),
        "production_algorithm_changed": False,
        "independent_holdout": True,
        "unrun_slots": expected - len(complete_slots),
        "budget_stop_reason": None if continuation_complete else metadata.get("budget_stop_reason", "continuation cap reached or run stopped"),
    }
    write_json(CONT_OUT / "continuation_results.json", results)
    metadata.update({
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "requests_started_cumulative": len(started),
        "requests_metered_cumulative": len(metered),
        "input_tokens_cumulative": total_input,
        "output_tokens_cumulative": total_output,
        "cost_usd_cumulative": total_cost,
        "method_runs_completed_cumulative": len(complete_slots),
        "case_kg_queries_cumulative": case_queries,
    })
    write_json(METADATA_PATH, metadata)
    report = [
        "# Iteration 19 — control continuation results", "",
        f"Status: **{status}**. Completed {len(complete_slots)}/{expected} method slots and {len(paired['paired_rows'])} paired repeats.",
        f"Cumulative API usage: {len(metered)} metered requests, {total_input:,} input / {total_output:,} output tokens, estimated ${total_cost:.6f}; continuation added {len(metered) - plan['base_calls']} requests and ${total_cost - plan['base_cost_usd']:.6f}.",
        f"Target results and controls are recomputed together in `paired_analysis_complete.json`; cumulative traces are in `paired_runs_cumulative.jsonl`.",
        "", "The initial 1,000-request run artifacts remain unchanged. The cumulative usage journal is `cumulative_call_usage.jsonl`.",
    ]
    (CONT_OUT / "continuation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashes = {
        "manifest_sha256": plan["hashes"]["manifest_sha256"],
        "initial_runs_sha256": plan["hashes"]["base_paired_runs_sha256"],
        "initial_usage_sha256": plan["hashes"]["base_call_usage_sha256"],
        "initial_missing_slots_sha256": plan["hashes"]["initial_missing_slots_sha256"],
        "continuation_runner_sha256": sha(Path(__file__).resolve()),
        "continuation_runs_sha256": sha(RUN_PATH) if RUN_PATH.exists() else None,
        "cumulative_usage_sha256": sha(USAGE_PATH),
        "combined_runs_sha256": sha(CONT_OUT / "paired_runs_cumulative.jsonl"),
        "paired_analysis_sha256": sha(CONT_OUT / "paired_analysis_complete.json"),
    }
    write_json(CONT_OUT / "continuation_hashes.json", hashes)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true", help="inspect the exact missing slots; no network or file writes")
    action.add_argument("--run", action="store_true", help="execute after a matching continuation authorization is present")
    action.add_argument("--resume", action="store_true", help="resume a continuation after reviewing its journals")
    args = parser.parse_args()

    plan = inspect_plan()
    if args.dry_run:
        completed_result_path = CONT_OUT / "continuation_results.json"
        completed_result = read_json(completed_result_path) if completed_result_path.exists() else None
        if completed_result and completed_result.get("status") == "COMPLETED" and int(completed_result.get("unrun_slots", -1)) == 0:
            dry_status = "DRY_RUN_PASS_CONTINUATION_ALREADY_COMPLETE"
            remaining_slots = 0
        elif AUTH_PATH.exists():
            validate_authorization(plan, cumulative_calls=plan["base_calls"], cumulative_cost=plan["base_cost_usd"])
            dry_status = "DRY_RUN_PASS_AUTHORIZATION_PRESENT"
            remaining_slots = len(plan["missing_slots"])
        else:
            dry_status = "DRY_RUN_PASS_AWAITING_NEW_AUTHORIZATION"
            remaining_slots = len(plan["missing_slots"])
        print(json.dumps({
            "status": dry_status,
            "continuation_scope": "remaining short-path control slots only",
            "base_completed_runs": len(plan["base_runs"]),
            "base_metered_requests": plan["base_calls"],
            "base_cost_usd": plan["base_cost_usd"],
            "missing_control_method_slots": remaining_slots,
            "authorized_initial_missing_slots": len(plan["missing_slots"]),
            "continuation_slots_completed": int((completed_result or {}).get("continuation_runs_completed", 0)),
            "missing_slots": plan["missing_slots"],
            "control_method_runs_completed": plan["control_runs_completed"],
            "control_requests_observed": plan["control_requests_observed"],
            "estimated_additional_requests": plan["estimated_additional_requests"],
            "estimated_additional_cost_usd": plan["estimated_additional_cost_usd"],
            "recommended_additional_request_cap": RECOMMENDED_ADDITIONAL_REQUEST_CAP,
            "recommended_cumulative_request_cap": plan["base_calls"] + RECOMMENDED_ADDITIONAL_REQUEST_CAP,
            "recommended_additional_cost_cap_usd": round(RECOMMENDED_CUMULATIVE_COST_CAP_USD - plan["base_cost_usd"], 10),
            "recommended_cumulative_cost_cap_usd": RECOMMENDED_CUMULATIVE_COST_CAP_USD,
            "hashes": plan["hashes"],
            "production_source_integrity": plan["production_integrity"],
            "network_calls": 0,
            "authorization_file_present": AUTH_PATH.exists(),
        }, ensure_ascii=False, indent=2))
        return

    initialize = bool(args.run)
    if args.run and CONT_OUT.exists():
        raise SystemExit("Continuation artifacts already exist; inspect them and use --resume")
    if args.resume and not METADATA_PATH.exists():
        raise SystemExit("Cannot resume without continuation metadata")

    # Validate budget and environment before creating any continuation journal.
    if not AUTH_PATH.exists():
        raise SystemExit("Missing continuation_authorization.json; no additional paid requests are authorized")
    auth = read_json(AUTH_PATH)
    calls_cap = int(auth.get("cumulative_request_cap", 0))
    cost_cap = float(auth.get("cumulative_cost_cap_usd", 0.0))
    auth = validate_authorization(plan, cumulative_calls=plan["base_calls"], cumulative_cost=plan["base_cost_usd"])

    instrumentation.load_safe_project_env()
    if os.environ.get("MODEL_NAME") != MODEL or not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(f"Expected MODEL_NAME={MODEL} and a configured API key")
    base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    if "api.openai.com" not in base_url:
        raise SystemExit("The continuation authorization is priced for direct OpenAI API requests")
    if int(os.environ.get("RESEARCH_MAX_LLM_CALLS", "0")) != calls_cap:
        raise SystemExit("RESEARCH_MAX_LLM_CALLS must equal the authorized cumulative request cap")
    if abs(float(os.environ.get("RESEARCH_MAX_LLM_COST_USD", "0")) - cost_cap) > 1e-9:
        raise SystemExit("RESEARCH_MAX_LLM_COST_USD must equal the authorized cumulative cost cap")

    os.environ["OPENAI_MAX_RETRIES"] = "1"
    os.environ["OPENAI_TIMEOUT"] = str(TIMEOUT_SECONDS)
    os.environ["OPENAI_RETRY_INTERVAL"] = "0"
    os.environ.setdefault("OPENAI_REASONING_EFFORT", "low")

    integrity = instrumentation.production_source_integrity()
    if not integrity.get("passed"):
        raise SystemExit(f"Frozen production source integrity failed: {integrity.get('failed_paths')}")
    preflight = instrumentation.endpoint_preflight()
    if not preflight.get("ready_for_run"):
        raise SystemExit("Model or Freebase endpoint preflight failed; no case generation started")

    if initialize:
        CONT_OUT.mkdir(parents=True, exist_ok=False)
        USAGE_PATH.write_bytes((OUT / "call_usage.jsonl").read_bytes())
    write_json(CONT_OUT / "preflight.json", preflight)

    if initialize:
        metadata = {
            "experiment": "Iteration 19 remaining control slots continuation",
            "status": "RUNNING",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "authorization": auth,
            "model": MODEL,
            "max_completion_tokens_per_request": MAX_COMPLETION_TOKENS,
            "timeout_seconds": TIMEOUT_SECONDS,
            "logical_attempts_per_request": 1,
            "retries": 0,
            "cumulative_request_cap": calls_cap,
            "cumulative_cost_cap_usd": cost_cap,
            "base_requests": plan["base_calls"],
            "base_cost_usd": plan["base_cost_usd"],
            "base_hashes": plan["hashes"],
            "missing_slots_initial": plan["missing_slots"],
            "preflight": preflight,
        }
        write_json(METADATA_PATH, metadata)
    else:
        metadata = read_json(METADATA_PATH)
        if metadata.get("authorization") != auth or metadata.get("base_hashes") != plan["hashes"]:
            raise SystemExit("Continuation metadata no longer matches its authorization or frozen base journals")
        if any(row.get("case_status") != "COMPLETED" for row in (read_jsonl(RUN_PATH) if RUN_PATH.exists() else [])):
            raise SystemExit("Continuation contains an incomplete method run; review before resuming")

    validate_cumulative_journal(plan, initialize=False)
    instrumentation.OUT = CONT_OUT
    # Meter reads the continuation-local call_usage.jsonl. Convert cumulative
    # authorization ceilings into the remaining additional budget here.
    additional_request_cap = calls_cap - plan["base_calls"]
    additional_cost_cap = cost_cap - plan["base_cost_usd"]
    if additional_request_cap != int(auth["additional_request_cap"]):
        raise SystemExit("Continuation request cap does not match the approved additional budget")
    if abs(additional_cost_cap - float(auth["additional_cost_cap_usd"])) > 1e-8:
        raise SystemExit("Continuation cost cap does not match the approved additional budget")
    instrumentation.MAX_REQUESTS = additional_request_cap
    instrumentation.MAX_COST_USD = additional_cost_cap

    base_done = {slot_key(row, depth_field="diagnostic_depth") for row in plan["base_runs"]}
    continued = read_jsonl(RUN_PATH) if RUN_PATH.exists() else []
    continued_done = {slot_key(row, depth_field="diagnostic_depth") for row in continued if row.get("case_status") == "COMPLETED"}
    if len(continued_done) != sum(row.get("case_status") == "COMPLETED" for row in continued):
        raise SystemExit("Continuation run journal contains duplicate slots")
    if any(slot_key(row, depth_field="diagnostic_depth") not in {slot_key(s, depth_field="depth") for s in plan["missing_slots"]} for row in continued):
        raise SystemExit("Continuation journal contains a slot outside the frozen missing-control set")

    inference_by_key = {row["question_key"]: row for row in plan["inference"]}
    allowed_initial = {slot_key(row, depth_field="depth") for row in plan["missing_slots"]}
    schedule = plan["manifest"]["treatment_protocol"]["randomized_run_schedule"]
    schedule_index = {slot_key(row, depth_field="depth"): index for index, row in enumerate(schedule, start=1)}
    missing_now = [row for row in plan["missing_slots"] if slot_key(row, depth_field="depth") not in continued_done]
    if not set(slot_key(row, depth_field="depth") for row in missing_now) <= allowed_initial:
        raise SystemExit("Continuation schedule drifted from the authorized slot set")

    stop_reason = None
    for slot in missing_now:
        key = slot["question_key"]
        repeat = int(slot["repeat_index"])
        depth = int(slot["depth"])
        infer_case = dict(inference_by_key[key])
        infer_case["question_id"] = infer_case["case_id"]
        infer_case["gold_answers"] = list(plan["scores"][key].get("gold_answers", []))
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
        run = instrumentation.run_one(agent, "v1_1_3", infer_case, schedule_index[(key, repeat, depth)])
        run.update({
            "question_key": key,
            "repeat_index": repeat,
            "diagnostic_depth": depth,
            "diagnostic_depth_before": initial_depth,
            "cohort_role_scoring_only": CONTROL_ROLE,
            "state_lineage": events,
            "request_count_case": meter.calls - request_start,
            "case_status": "COMPLETED" if not run.get("error") else "INCOMPLETE_BUDGET_OR_RUNTIME_ERROR",
            "continuation_run": True,
        })
        run["gold_candidate_diagnostic_scoring_only"] = instrumentation.case_candidate_summary(
            run, set(map(str, plan["scores"][key].get("answer_ids", [])))
        )
        append_jsonl(RUN_PATH, run)
        if run["case_status"] != "COMPLETED":
            stop_reason = run.get("error") or meter.unmetered_error or "continuation run incomplete"
            metadata["budget_stop_reason"] = stop_reason
            write_json(METADATA_PATH, metadata)
            break
        print(json.dumps({
            "slot": schedule_index[(key, repeat, depth)],
            "continuation_slots_completed": sum(r.get("case_status") == "COMPLETED" for r in read_jsonl(RUN_PATH)),
            "continuation_slots_total": len(plan["missing_slots"]),
            "cumulative_requests": plan["base_calls"] + meter.calls,
            "cumulative_cost_usd": round(plan["base_cost_usd"] + meter.spent, 7),
            "status": "COMPLETED",
        }), flush=True)
        if meter.halted or meter.calls >= calls_cap or meter.spent >= cost_cap:
            stop_reason = meter.unmetered_error or "cumulative request or cost cap reached"
            metadata["budget_stop_reason"] = stop_reason
            write_json(METADATA_PATH, metadata)
            break

    merge_continuation_usage(plan)
    final = finalize(plan, auth, metadata, preflight)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    if final["status"] != "COMPLETED":
        raise SystemExit("Control continuation remains incomplete; cumulative report and journals were saved")


if __name__ == "__main__":
    main()
