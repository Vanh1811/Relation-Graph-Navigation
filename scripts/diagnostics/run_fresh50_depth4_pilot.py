#!/usr/bin/env python3
"""Run the authorized eight-case, depth-only TGAR diagnostic pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts" / "diagnostics") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

import run_fresh50_state_lineage as instrumentation  # noqa: E402

WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
OUT = WORK / "RESULTS/iter_018_depth4_diagnostic_pilot"
PREVIOUS_RUNS = WORK / "RESULTS/iter_016_instrumented_tgar_rerun/runs_with_state_lineage.jsonl"
FAILURE_KEYS = [
    "cwq::WebQTest-450_8928d9663aabbda9525d0b8cbf2e3e39",
    "cwq::WebQTrn-2486_bcd294e8b78226fb373c0a54e4dcba01",
    "cwq::WebQTrn-436_b301288f1064d4357ae5d8271061e1d1",
]
CONTROL_KEYS = [
    "cwq::WebQTest-1348_3f44e0c4fd23a8334a972ebb812e0e35",
    "cwq::WebQTest-1348_c0fe53f8950b9ae1ce9b57622bd27ece",
    "cwq::WebQTest-450_5fc1f8bfcdc0eb7f09c9d976ed42f1c4",
    "cwq::WebQTrn-1259_1997cb4922db71983be26e6a509950f4",
    "cwq::WebQTrn-1259_7d99f28971c0ff3aa8437737f81f16b2",
]
ALL_KEYS = FAILURE_KEYS + CONTROL_KEYS
REQUEST_CAP = 120
COST_CAP_USD = 0.15
DEPTH = 4
MODEL = "gpt-5.6-luna"
MAX_COMPLETION_TOKENS = 1024
TIMEOUT_SECONDS = 120


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_pilot_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    cases, scoring, roles = instrumentation.load_cases()
    by_key = {f"{row['dataset']}::{row['question_id']}": row for row in cases}
    score_by_key = {row["question_key"]: row for row in scoring}
    expected = set(ALL_KEYS)
    if set(by_key) & expected != expected:
        raise RuntimeError(f"Frozen source missing pilot cases: {sorted(expected - set(by_key))}")
    chosen_roles = {key: roles[key] for key in ALL_KEYS}
    if any(chosen_roles[key] != "TGAR_FAILURE" for key in FAILURE_KEYS):
        raise RuntimeError("A specified failure case is not labeled TGAR_FAILURE in the frozen scoring sidecar")
    if any(chosen_roles[key] != "BOTH_CORRECT_CONTROL" for key in CONTROL_KEYS):
        raise RuntimeError("A specified control is not labeled BOTH_CORRECT_CONTROL in the frozen scoring sidecar")
    chosen_cases = [by_key[key] for key in ALL_KEYS]
    chosen_scores = [score_by_key[key] for key in ALL_KEYS]
    if len(chosen_cases) != 8 or len({f"{r['dataset']}::{r['question_id']}" for r in chosen_cases}) != 8:
        raise RuntimeError("Pilot must contain exactly eight unique frozen cases")
    return chosen_cases, chosen_scores, chosen_roles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true", help="execute the approved development pilot")
    action.add_argument("--resume", action="store_true", help="resume only fully journaled cases")
    args = parser.parse_args()

    instrumentation.load_safe_project_env()
    calls_cap = int(os.environ.get("RESEARCH_MAX_LLM_CALLS", "0"))
    cost_cap = float(os.environ.get("RESEARCH_MAX_LLM_COST_USD", "0"))
    if calls_cap != REQUEST_CAP or abs(cost_cap - COST_CAP_USD) > 1e-9:
        raise SystemExit("Approved pilot requires RESEARCH_MAX_LLM_CALLS=120 and RESEARCH_MAX_LLM_COST_USD=0.15")
    if os.environ.get("MODEL_NAME") != MODEL:
        raise SystemExit(f"Expected MODEL_NAME={MODEL}; refusing a different model")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing")
    base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    if "api.openai.com" not in base_url:
        raise SystemExit("Approved cost budget is for direct OpenAI API requests only")

    os.environ["OPENAI_MAX_RETRIES"] = "1"  # one wrapper attempt; SDK retries are disabled in LLMAPI
    os.environ["OPENAI_TIMEOUT"] = str(TIMEOUT_SECONDS)
    os.environ["OPENAI_RETRY_INTERVAL"] = "0"
    os.environ.setdefault("OPENAI_REASONING_EFFORT", "low")

    if not (OUT / "EXPERIMENT_SPEC.md").exists():
        raise SystemExit(f"Missing preregistration: {OUT / 'EXPERIMENT_SPEC.md'}")
    metadata_path = OUT / "run_metadata.json"
    if args.run and metadata_path.exists():
        raise SystemExit("Existing pilot metadata found; use --resume after reviewing the request journal")
    if args.resume and not metadata_path.exists():
        raise SystemExit("No pilot metadata found to resume")

    cases, scoring, roles = load_pilot_cases()
    score_by_key = {row["question_key"]: row for row in scoring}
    integrity_before = instrumentation.production_source_integrity()
    if not integrity_before["passed"]:
        raise SystemExit(f"Frozen production source integrity failed: {integrity_before['failed_paths']}")

    instrumentation.OUT = OUT
    instrumentation.MAX_REQUESTS = REQUEST_CAP
    instrumentation.MAX_COST_USD = COST_CAP_USD
    OUT.mkdir(parents=True, exist_ok=True)
    if args.run and any((OUT / name).exists() for name in ("runs_with_state_lineage.jsonl", "call_usage.jsonl")):
        raise SystemExit("Existing run journal found; use --resume to prevent duplicate calls")

    input_files = [instrumentation.CANDIDATE_AUDIT, instrumentation.FROZEN_MANIFEST,
                   instrumentation.EVALUATION_GOLD, instrumentation.BASELINE_RUNS, PREVIOUS_RUNS]
    input_hashes = {str(path.relative_to(ROOT)): sha_file(path) for path in input_files}
    key_hash = hashlib.sha256("\n".join(ALL_KEYS).encode()).hexdigest()
    previous_runs = {row["question_key"]: row for row in read_jsonl(PREVIOUS_RUNS)}
    if args.run:
        metadata = {
            "experiment": "single-variable TGAR depth 3-to-4 diagnostic pilot",
            "status": "RUNNING", "created_utc": utc_now(), "model": MODEL,
            "depth_before": 3, "depth_after": DEPTH, "cases": len(ALL_KEYS),
            "failure_case_count": len(FAILURE_KEYS), "control_case_count": len(CONTROL_KEYS),
            "case_keys_sha256": key_hash, "request_cap": REQUEST_CAP, "cost_cap_usd": COST_CAP_USD,
            "max_completion_tokens_per_request": MAX_COMPLETION_TOKENS,
            "timeout_seconds": TIMEOUT_SECONDS, "logical_attempts_per_call": 1,
            "retry_policy": "No retries; OPENAI_MAX_RETRIES=1 is one wrapper attempt and SDK retries are disabled.",
            "prices_usd_per_million": {"input": instrumentation.INPUT_PRICE,
                                       "output": instrumentation.OUTPUT_PRICE,
                                       "source": instrumentation.PRICE_SOURCE},
            "openai_base_url_host": base_url.split("/")[2],
            "production_algorithm_changed": False, "independent_holdout": False,
            "inference_scope": "Question and topic entities only; gold answers, paths, and cohort roles are scoring-only.",
            "input_hashes": input_hashes,
            "frozen_production_source_integrity_before": integrity_before,
            "case_keys_failure": FAILURE_KEYS, "case_keys_control": CONTROL_KEYS,
            "previous_depth3_runs_sha256": sha_file(PREVIOUS_RUNS),
        }
        write_json(metadata_path, metadata)
    else:
        metadata = read_json(metadata_path)
        expected = {"model": MODEL, "depth_before": 3, "depth_after": DEPTH,
                    "request_cap": REQUEST_CAP, "cost_cap_usd": COST_CAP_USD,
                    "input_hashes": input_hashes, "case_keys_sha256": key_hash}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise SystemExit("Resume metadata does not match the approved pilot inputs/configuration")

    preflight = instrumentation.endpoint_preflight()
    write_json(OUT / "preflight.json", preflight)
    if not preflight.get("ready_for_run"):
        metadata["status"] = "BLOCKED_PREFLIGHT"
        metadata["preflight"] = preflight
        write_json(metadata_path, metadata)
        raise SystemExit("OpenAI or Freebase preflight failed; no case inference was submitted")

    runs_path = OUT / "runs_with_state_lineage.jsonl"
    existing = read_jsonl(runs_path) if args.resume and runs_path.exists() else []
    if any(row.get("case_status") != "COMPLETED" for row in existing):
        raise SystemExit("An incomplete case exists. Review request/case journals manually; refusing duplicate case inference.")
    done = {row["question_key"] for row in existing}
    if not done.issubset(set(ALL_KEYS)):
        raise SystemExit("Run journal contains an unexpected case key")

    results = existing[:]
    budget_stop = None
    for index, case in enumerate(cases, start=1):
        key = f"{case['dataset']}::{case['question_id']}"
        if key in done:
            continue
        agent = instrumentation.make_agent("v1_1_3")
        before_depth = int(agent.cor_agent.depth)
        if before_depth != 3:
            raise RuntimeError(f"Expected frozen depth=3 before the treatment, found {before_depth}")
        agent.cor_agent.depth = DEPTH
        if int(agent.cor_agent.depth) != DEPTH:
            raise RuntimeError("Could not apply diagnostic depth=4 setting")
        events: list[dict[str, Any]] = []
        meter = instrumentation.install_state_lineage(agent, events)
        request_start = meter.calls
        run = instrumentation.run_one(agent, "v1_1_3", case, index)
        run["question_key"] = key
        run["cohort_role_scoring_only"] = roles[key]
        run["state_lineage"] = events
        run["diagnostic_depth_before"] = before_depth
        run["diagnostic_depth"] = int(agent.cor_agent.depth)
        run["request_count_case"] = meter.calls - request_start
        run["case_status"] = "COMPLETED" if not run.get("error") else "INCOMPLETE_ERROR"
        score = score_by_key[key]
        run["gold_candidate_diagnostic_scoring_only"] = instrumentation.case_candidate_summary(
            run, set(map(str, score.get("answer_ids", []))))
        append_jsonl(runs_path, [run])
        results.append(run)
        done.add(key)
        print(json.dumps({"case": index, "total": len(ALL_KEYS), "question_key": key,
                          "status": run["case_status"], "requests": meter.calls,
                          "cost_usd": round(meter.spent, 7),
                          "depth": run["diagnostic_depth"]}), flush=True)
        if meter.halted or run["case_status"] != "COMPLETED":
            budget_stop = meter.unmetered_error or run.get("error") or "request/cost cap reached"
            break

    completed = [row for row in results if row.get("case_status") == "COMPLETED"]
    rows = []
    for run in completed:
        key = run["question_key"]
        old = previous_runs[key]
        rows.append({
            "question_key": key, "cohort_role": roles[key],
            "depth3_hits_at_1": int(old.get("hits_at_1", 0)),
            "depth4_hits_at_1": int(run.get("hits_at_1", 0)),
            "depth4_vs_depth3": ("TIE" if int(old.get("hits_at_1", 0)) == int(run.get("hits_at_1", 0))
                                 else ("DEPTH4_ONLY" if int(run.get("hits_at_1", 0)) else "DEPTH3_ONLY")),
            "depth3_f1": float(old.get("f1", 0.0)), "depth4_f1": float(run.get("f1", 0.0)),
            "depth4_candidate_status": run.get("gold_candidate_diagnostic_scoring_only", {}).get("candidate_presence_status"),
            "depth4_route": run.get("relation_chain", []),
            "depth4_requests": run.get("request_count_case", 0),
        })
    write_json(OUT / "scoring_gold_only.json", {"case_keys": ALL_KEYS,
                "scores": [{"question_key": s["question_key"], "role": roles[s["question_key"]],
                            "answer_ids": s.get("answer_ids", []), "gold_answers": s.get("gold_answers", [])}
                           for s in scoring]})
    write_json(OUT / "paired_depth_comparison.json", {"rows": rows})
    write_csv(OUT / "case_audit.csv", rows)
    usage = read_jsonl(OUT / "call_usage.jsonl") if (OUT / "call_usage.jsonl").exists() else []
    starts = [row for row in usage if row.get("status") == "REQUEST_STARTED"]
    metered = [row for row in usage if row.get("status") == "REQUEST_METERED"]
    total_cost = sum(float(row.get("cost_usd", 0) or 0) for row in metered)
    total_input = sum(int(row.get("input_tokens", 0) or 0) for row in metered)
    total_output = sum(int(row.get("output_tokens", 0) or 0) for row in metered)
    failures = [row for row in rows if row["cohort_role"] == "TGAR_FAILURE"]
    controls = [row for row in rows if row["cohort_role"] == "BOTH_CORRECT_CONTROL"]
    failure_gains = sum(row["depth4_vs_depth3"] == "DEPTH4_ONLY" for row in failures)
    control_regressions = sum(row["depth4_vs_depth3"] == "DEPTH3_ONLY" for row in controls)
    query_count = sum(int(run.get("kg_queries", 0) or 0) for run in completed)
    status = "COMPLETED" if len(completed) == len(ALL_KEYS) else ("STOPPED_AT_BUDGET_CAP" if budget_stop else "INCOMPLETE")
    integrity_after = instrumentation.production_source_integrity()
    result_summary = {
        "experiment": "single-variable TGAR depth 3-to-4 diagnostic pilot", "status": status,
        "cases_expected": len(ALL_KEYS), "cases_completed": len(completed),
        "failure_cases_completed": len(failures), "controls_completed": len(controls),
        "depth4_only_failure_hits": failure_gains, "depth3_only_failure_hits": sum(r["depth4_vs_depth3"] == "DEPTH3_ONLY" for r in failures),
        "control_regressions": control_regressions,
        "failure_depth3_hits": sum(r["depth3_hits_at_1"] for r in failures),
        "failure_depth4_hits": sum(r["depth4_hits_at_1"] for r in failures),
        "control_depth3_hits": sum(r["depth3_hits_at_1"] for r in controls),
        "control_depth4_hits": sum(r["depth4_hits_at_1"] for r in controls),
        "requests_started": len(starts), "metered_requests": len(metered),
        "input_tokens": total_input, "output_tokens": total_output, "cost_usd": total_cost,
        "request_cap": REQUEST_CAP, "cost_cap_usd": COST_CAP_USD,
        "kg_queries_in_cases": query_count, "preflight_count": 1,
        "budget_stop_reason": budget_stop,
        "frozen_production_source_integrity_after": integrity_after,
        "production_algorithm_changed": False, "independent_holdout": False,
        "interpretation_limit": "One stochastic run per case on a reused development cohort. Paired differences are diagnostic, not independent validation or a generalization claim.",
    }
    write_json(OUT / "results.json", result_summary)
    metadata.update({"status": status, "preflight": preflight, "requests_started": len(starts),
                    "metered_requests": len(metered), "input_tokens": total_input,
                    "output_tokens": total_output, "cost_usd": total_cost,
                    "kg_queries_in_cases": query_count, "frozen_production_source_integrity_after": integrity_after,
                    "budget_stop_reason": budget_stop})
    write_json(metadata_path, metadata)

    report_lines = [
        "# Iteration 18 — depth-4 diagnostic pilot", "",
        "## Results", "",
        f"Status: **{status}**. Cases completed: **{len(completed)}/8** ({len(failures)}/3 failures, {len(controls)}/5 controls).",
        f"Model requests: **{len(starts)}**; tokens: **{total_input:,} input / {total_output:,} output**; estimated cost: **${total_cost:.7f}** (hard caps: 120 requests / $0.15).",
        f"Read-only SPARQL: **{query_count}** case queries plus one preflight. Production source integrity: **{integrity_after.get('source_count', 0)}/7**.",
        "", "| Case | Role | Depth 3 Hits@1 | Depth 4 Hits@1 | Paired result | Depth 4 F1 | Requests |", "|---|---|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        report_lines.append(f"| `{row['question_key']}` | {row['cohort_role']} | {row['depth3_hits_at_1']} | {row['depth4_hits_at_1']} | {row['depth4_vs_depth3']} | {row['depth4_f1']:.3f} | {row['depth4_requests']} |")
    gate = "PASS" if failure_gains >= 1 and control_regressions == 0 and len(completed) == 8 else "FAIL_OR_INCOMPLETE"
    report_lines += [
        "", "## Preregistered development gate", "",
        f"Gate: **{gate}** — requires at least one depth-4-only hit among the three targeted failures and zero control regressions. Observed {failure_gains} targeted paired gains and {control_regressions} control regressions.",
        "", "## Interpretation", "",
        "Depth is the only configured method variable changed: the diagnostic runner constructs frozen TGAR at depth 3 and sets `cor_agent.depth=4` before inference. Other prompts, model settings, relation width, routing, executor, filtering, and scoring are unchanged. Gold data and cohort labels are used only by the runner/scorer after inference.",
        "",
        "A paired gain is evidence consistent with a depth contribution on this development case, but model responses are stochastic and the cohort has been repeatedly inspected. Do not call the paired gain a validated general effect. A positive development result still requires a separately budgeted untouched evaluation. A negative result does not eliminate early continuation decisions as another mechanism.",
        "", "## Reproduction", "",
        "```bash",
        "RESEARCH_MAX_LLM_CALLS=120 RESEARCH_MAX_LLM_COST_USD=0.15 PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/run_fresh50_depth4_pilot.py --run",
        "```", "",
        "Full traces, usage journal, scoring-only data, source hashes, and exact inputs are stored in this directory. No production inference code or official benchmark output was modified.", "",
    ]
    (OUT / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    hashes = {path.name: sha_file(path) for path in sorted(OUT.iterdir())
              if path.is_file() and path.name != "artifact_hashes.json"}
    hashes["runner_source_sha256"] = sha_file(Path(__file__))
    hashes["instrumentation_source_sha256"] = sha_file(Path(instrumentation.__file__))
    write_json(OUT / "artifact_hashes.json", hashes)
    print(json.dumps(result_summary, ensure_ascii=False, indent=2), flush=True)
    if not integrity_after.get("passed"):
        raise SystemExit("Frozen production source integrity mismatch after pilot")


if __name__ == "__main__":
    main()
