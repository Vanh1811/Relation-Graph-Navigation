#!/usr/bin/env python3
"""Instrument frozen TGAR execution on the preregistered 19-case diagnostic cohort."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_cor_baseline_experiment import load_safe_project_env  # noqa: E402
from run_fresh50_strong_branch_three_way import (  # noqa: E402
    OUTPUT_ROOT as FRESH50_ROOT,
    endpoint_preflight,
    make_agent,
    run_one,
)

WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
OUT = WORK / "RESULTS/iter_016_instrumented_tgar_rerun"
CANDIDATE_AUDIT = WORK / "RESULTS/iter_011_residual_candidate_first_loss/candidate_first_loss.csv"
FROZEN_MANIFEST = FRESH50_ROOT / "manifest_frozen.json"
EVALUATION_GOLD = FRESH50_ROOT / "evaluation_gold.jsonl"
BASELINE_RUNS = FRESH50_ROOT / "runs.jsonl"
SNAPSHOT = WORK / "RESULTS/iter_000_baseline/SNAPSHOT_MANIFEST.json"
MODEL = "gpt-5.6-luna"
INPUT_PRICE = 0.20
OUTPUT_PRICE = 1.20
PRICE_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5.6-luna"
MAX_REQUESTS = 160
MAX_COST_USD = 0.20
TIMEOUT_SECONDS = 120
MAX_CASES = 19


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, row: dict[str, Any]) -> None:
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def relation_chain(relations: list[Any]) -> list[dict[str, str]]:
    return [
        {"id": str(getattr(rel, "id", "")), "direction": "OUTGOING" if bool(getattr(rel, "left", False)) else "INCOMING"}
        for rel in relations
    ]


def entity_payload(entity: Any) -> dict[str, str]:
    return {"id": str(getattr(entity, "id", "")), "name": str(getattr(entity, "name", ""))}


class BudgetStop(RuntimeError):
    pass


class Meter:
    def __init__(self, call_path: Path, call_cap: int, cost_cap: float) -> None:
        self.call_path = call_path
        self.call_cap = call_cap
        self.cost_cap = cost_cap
        self.calls = 0
        self.spent = 0.0
        self.halted = False
        self.unmetered_error: str | None = None
        if call_path.exists():
            previous = read_jsonl(call_path)
            started = [r for r in previous if r.get("status") == "REQUEST_STARTED"]
            finished = {r.get("request_id") for r in previous if r.get("status") in {"REQUEST_METERED", "REQUEST_UNMETERED_ERROR"}}
            pending = [r for r in started if r.get("request_id") not in finished]
            if pending:
                raise RuntimeError("A prior model request has no recorded response; refusing an unsafe paid resume.")
            self.calls = len(started)
            self.spent = sum(float(r.get("cost_usd", 0.0) or 0.0) for r in previous if r.get("status") == "REQUEST_METERED")
            self.halted = any(r.get("status") == "REQUEST_UNMETERED_ERROR" for r in previous)
            self.unmetered_error = "previous unmetered request" if self.halted else None

    @staticmethod
    def conservative_input_token_bound(user_prompt: str, system_prompt: str | None) -> int:
        # Token count cannot exceed the UTF-8 byte count for these text messages;
        # add 32 tokens for Chat Completions message framing.
        raw = (user_prompt + (system_prompt or "")).encode("utf-8")
        return len(raw) + 32

    def guarded_generate(self, original: Any, user_prompt: str, temperature: float = 0.01,
                         max_tokens: int = 256, system_prompt: str | None = None):
        if self.halted:
            raise BudgetStop(self.unmetered_error or "the paid request budget is halted")
        if self.calls >= self.call_cap:
            self.halted = True
            raise BudgetStop("request cap reached before next API call")
        input_bound = self.conservative_input_token_bound(user_prompt, system_prompt)
        reserve = input_bound * INPUT_PRICE / 1_000_000 + int(max_tokens) * OUTPUT_PRICE / 1_000_000
        if self.spent + reserve > self.cost_cap:
            self.halted = True
            raise BudgetStop(f"next-call worst-case reserve ${reserve:.8f} exceeds remaining cost cap")

        self.calls += 1
        request_id = f"request_{self.calls:04d}"
        append_jsonl(self.call_path, {
            "request_id": request_id, "status": "REQUEST_STARTED", "recorded_utc": utc_now(),
            "prompt_sha256": sha_bytes(user_prompt.encode("utf-8")),
            "system_prompt_sha256": sha_bytes((system_prompt or "").encode("utf-8")),
            "input_token_reserve": input_bound, "output_token_reserve": int(max_tokens),
            "reserved_cost_usd": reserve,
        })
        started = time.perf_counter()
        text, usage = original(
            user_prompt=user_prompt, temperature=temperature,
            max_tokens=max_tokens, system_prompt=system_prompt,
        )
        latency = time.perf_counter() - started
        usage = usage if isinstance(usage, dict) else {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        usage_error = usage.get("error")
        metered = input_tokens > 0 or output_tokens > 0
        cost = input_tokens * INPUT_PRICE / 1_000_000 + output_tokens * OUTPUT_PRICE / 1_000_000
        if metered:
            self.spent += cost
            status = "REQUEST_METERED"
        else:
            status = "REQUEST_UNMETERED_ERROR"
            self.halted = True
            self.unmetered_error = str(usage_error or "request returned no token usage")
        append_jsonl(self.call_path, {
            "request_id": request_id, "status": status, "recorded_utc": utc_now(),
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cost_usd": cost, "cumulative_cost_usd": self.spent,
            "latency_seconds": latency, "has_content": bool(text),
            "api_error": str(usage_error)[:500] if usage_error else None,
        })
        if self.spent > self.cost_cap + 1e-12:
            self.halted = True
            raise BudgetStop("metered usage exceeded the configured cost cap; no further calls will be sent")
        return text, usage


def install_state_lineage(agent: Any, state_events: list[dict[str, Any]]) -> Meter:
    cor = agent.cor_agent
    context = {"events": state_events}
    for name in ("relation_search", "relation_prune", "entity_search", "reasoning"):
        original = getattr(cor, name)

        def make_wrapper(method_name: str, original_method: Any):
            def wrapped(*args: Any, **kwargs: Any):
                import inspect
                bound = inspect.signature(original_method).bind_partial(*args, **kwargs)
                values = bound.arguments
                root_entity = values.get("topic_entity")
                chain = values.get("relation_chain", [])
                event: dict[str, Any] = {
                    "sequence": len(context["events"]) + 1,
                    "stage": method_name,
                    "root_entity": entity_payload(root_entity) if root_entity is not None else {},
                    "relation_chain": relation_chain(chain),
                    "recorded_utc": utc_now(),
                }
                if method_name == "relation_prune":
                    event["candidate_relations_head"] = [str(x) for x in values.get("head_relations", [])]
                    event["candidate_relations_tail"] = [str(x) for x in values.get("tail_relations", [])]
                elif method_name == "reasoning":
                    event["candidate_entities"] = [str(x) for x in values.get("target_entity_candidates", [])]
                result = original_method(*args, **kwargs)
                if method_name == "relation_search":
                    head, tail = result
                    event["head_relations"] = [str(x) for x in head]
                    event["tail_relations"] = [str(x) for x in tail]
                elif method_name == "relation_prune":
                    event["selected_relations"] = [
                        {"id": str(getattr(x, "id", "")), "direction": "OUTGOING" if bool(getattr(x, "left", False)) else "INCOMING"}
                        for x in result
                    ]
                elif method_name == "entity_search":
                    ids, names, literals = result
                    event.update({"candidate_entity_ids": list(ids), "candidate_entity_names": list(names), "literal_candidates": list(literals)})
                elif method_name == "reasoning":
                    action, response, answers = result
                    event.update({"action": action, "response": response or "", "selected_answers": list(answers or [])})
                context["events"].append(event)
                return result
            return wrapped

        setattr(cor, name, make_wrapper(name, original))

    meter = Meter(OUT / "call_usage.jsonl", MAX_REQUESTS, MAX_COST_USD)
    original_generate = cor.llm_api.generate

    def guarded_generate(user_prompt: str, temperature: float = 0.01, max_tokens: int = 256,
                         system_prompt: str | None = None):
        return meter.guarded_generate(original_generate, user_prompt, temperature, max_tokens, system_prompt)

    cor.llm_api.generate = guarded_generate
    return meter


def production_source_integrity() -> dict[str, Any]:
    manifest = read_json(SNAPSHOT)
    failures = []
    checked = manifest.get("production_reasoning_source_hashes", {})
    for relative, expected in checked.items():
        path = ROOT / relative
        if not path.exists() or sha_file(path) != expected:
            failures.append(relative)
    return {"source_count": len(checked), "failed_paths": failures, "passed": not failures}


def load_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    with CANDIDATE_AUDIT.open(newline="", encoding="utf-8") as f:
        audit = list(csv.DictReader(f))
    selected = [r for r in audit if r["cohort_role"] in {"TGAR_FAILURE", "BOTH_CORRECT_CONTROL"}]
    if len(selected) != MAX_CASES or sum(r["cohort_role"] == "TGAR_FAILURE" for r in selected) != 14:
        raise ValueError("Expected exactly 14 residual failures and 5 controls")
    manifest = read_json(FROZEN_MANIFEST)
    manifest_by_key = {f"{r['dataset']}::{r['question_id']}": r for r in manifest["cases"]}
    gold_rows = read_jsonl(EVALUATION_GOLD)
    gold_by_key = {f"{r['dataset']}::{r['question_id']}": r for r in gold_rows}
    baseline = read_jsonl(BASELINE_RUNS)
    baseline_by_key = {r["question_key"]: r for r in baseline if r.get("method") == "v1_1_3"}
    cases, scoring, roles = [], [], {}
    for row in selected:
        key = row["question_key"]
        if key not in manifest_by_key or key not in gold_by_key or key not in baseline_by_key:
            raise ValueError(f"Missing frozen manifest, gold, or baseline row for {key}")
        case = dict(manifest_by_key[key]); case.update(gold_by_key[key])
        case["case_id"] = case["question_id"]
        cases.append(case)
        score = dict(gold_by_key[key])
        score.update({"question_key": key, "cohort_role_scoring_only": row["cohort_role"],
                      "archived_tgar_hits_at_1": int(baseline_by_key[key].get("hits_at_1", 0)),
                      "archived_tgar_f1": float(baseline_by_key[key].get("f1", 0.0))})
        scoring.append(score)
        roles[key] = row["cohort_role"]
    return cases, scoring, roles


def case_candidate_summary(run: dict[str, Any], gold_ids: set[str]) -> dict[str, Any]:
    searches = [e for e in run.get("sparql_history", []) if e.get("search_type") == "entity_search"]
    first = None
    for i, event in enumerate(searches, start=1):
        if gold_ids.intersection(str(x) for x in event.get("candidate_entity_ids", [])):
            first = i
            break
    last_has = bool(searches and gold_ids.intersection(str(x) for x in searches[-1].get("candidate_entity_ids", [])))
    return {
        "entity_search_count": len(searches),
        "gold_answer_seen_any_search": first is not None,
        "first_search_index_with_gold": first,
        "gold_answer_in_last_search": last_has,
        "candidate_presence_status": "NEVER_SEEN" if first is None else ("PRESENT_IN_LAST" if last_has else "SEEN_THEN_ABSENT"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute the authorized instrumented rerun")
    parser.add_argument("--resume", action="store_true", help="continue after fully logged cases; never resend an uncertain request")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("No requests sent. Use --run after explicitly setting the approved limits.")

    load_safe_project_env()
    calls_cap = int(os.environ.get("RESEARCH_MAX_LLM_CALLS", "0"))
    cost_cap = float(os.environ.get("RESEARCH_MAX_LLM_COST_USD", "0"))
    if calls_cap != MAX_REQUESTS or abs(cost_cap - MAX_COST_USD) > 1e-9:
        raise SystemExit("This approved run requires RESEARCH_MAX_LLM_CALLS=160 and RESEARCH_MAX_LLM_COST_USD=0.20")
    if os.environ.get("MODEL_NAME") != MODEL:
        raise SystemExit(f"Expected configured model {MODEL}; refusing other model")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing")
    base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    if "api.openai.com" not in base_url:
        raise SystemExit("This cost cap is priced for direct OpenAI API requests only")

    # One logical request per generation. The SDK itself has retries disabled.
    os.environ["OPENAI_MAX_RETRIES"] = "1"
    os.environ["OPENAI_TIMEOUT"] = str(TIMEOUT_SECONDS)
    os.environ["OPENAI_RETRY_INTERVAL"] = "0"
    os.environ.setdefault("OPENAI_REASONING_EFFORT", "low")

    cases, scoring, roles = load_cases()
    if args.resume:
        if not OUT.is_dir() or not (OUT / "run_metadata.json").exists():
            raise SystemExit("No existing run metadata to resume")
    else:
        OUT.mkdir(parents=True, exist_ok=False)
    integrity_before = production_source_integrity()
    if not integrity_before["passed"]:
        raise SystemExit(f"Frozen production source hashes differ: {integrity_before['failed_paths']}")
    preflight = endpoint_preflight()
    write_json(OUT / "preflight.json", preflight)
    if not preflight.get("ready_for_run"):
        raise SystemExit("OpenAI model or Freebase SPARQL preflight failed; no case requests submitted")

    request_path = OUT / "call_usage.jsonl"
    meter = Meter(request_path, calls_cap, cost_cap)
    scoring_by_key = {r["question_key"]: r for r in scoring}
    lineage_path = OUT / "runs_with_state_lineage.jsonl"
    existing = read_jsonl(lineage_path) if args.resume and lineage_path.exists() else []
    done = {r["question_key"] for r in existing if r.get("case_status") == "COMPLETED"}
    if args.resume and request_path.exists():
        usage_rows = read_jsonl(request_path)
        starts = {r["request_id"] for r in usage_rows if r.get("status") == "REQUEST_STARTED"}
        finals = {r.get("request_id") for r in usage_rows if r.get("status") in {"REQUEST_METERED", "REQUEST_UNMETERED_ERROR"}}
        if starts - finals:
            raise SystemExit("Found an unclosed API request; refusing to risk a duplicate charge")

    metadata = {
        "experiment": "instrumented frozen-TGAR rerun for candidate/state lineage",
        "status": "RUNNING", "created_utc": utc_now(), "model": MODEL,
        "cases": MAX_CASES, "case_keys_sha256": sha_bytes("\n".join(sorted(roles)).encode()),
        "cohort_counts_scoring_only": {"TGAR_FAILURE": 14, "BOTH_CORRECT_CONTROL": 5},
        "request_cap": calls_cap, "cost_cap_usd": cost_cap,
        "max_completion_tokens_per_request": 1024, "timeout_seconds": TIMEOUT_SECONDS,
        "logical_attempts_per_call": 1, "retry_policy": "disabled; OPENAI_MAX_RETRIES=1 means one total SDK attempt",
        "prices_usd_per_million": {"input": INPUT_PRICE, "output": OUTPUT_PRICE, "source": PRICE_SOURCE},
        "openai_base_url_host": base_url.split("/")[2] if "://" in base_url else base_url,
        "request_scope": "Question, initial entities, and ordinary runtime state only. Gold and cohort role are scoring-only.",
        "production_algorithm_changed": False, "independent_holdout": False,
        "frozen_production_source_integrity_before": integrity_before,
        "input_hashes": {str(CANDIDATE_AUDIT.relative_to(ROOT)): sha_file(CANDIDATE_AUDIT),
                         str(FROZEN_MANIFEST.relative_to(ROOT)): sha_file(FROZEN_MANIFEST),
                         str(EVALUATION_GOLD.relative_to(ROOT)): sha_file(EVALUATION_GOLD),
                         str(BASELINE_RUNS.relative_to(ROOT)): sha_file(BASELINE_RUNS)},
        "preflight_read_only_kg_queries": 1,
        "preflight": preflight,
    }
    if args.resume:
        prev = read_json(OUT / "run_metadata.json")
        for key in ("model", "request_cap", "cost_cap_usd", "input_hashes", "case_keys_sha256"):
            if prev.get(key) != metadata.get(key):
                raise SystemExit(f"Resume metadata mismatch: {key}")
        metadata["created_utc"] = prev.get("created_utc", metadata["created_utc"])
    write_json(OUT / "run_metadata.json", metadata)

    all_case_rows = existing[:]
    all_scoring = scoring
    budget_stop = None
    for ix, case in enumerate(cases, start=1):
        key = f"{case['dataset']}::{case['question_id']}"
        if key in done:
            continue
        case_events: list[dict[str, Any]] = []
        agent = make_agent("v1_1_3")
        meter = install_state_lineage(agent, case_events)
        run = run_one(agent, "v1_1_3", case, ix)
        run["question_key"] = key
        run["cohort_role_scoring_only"] = roles[key]
        run["state_lineage"] = case_events
        run["case_status"] = "COMPLETED"
        if meter.halted and run.get("error"):
            run["case_status"] = "INCOMPLETE_BUDGET_OR_METERING_STOP"
            budget_stop = run.get("error")
        score = scoring_by_key[key]
        candidate_summary = case_candidate_summary(run, set(score.get("answer_ids", [])))
        run["gold_candidate_diagnostic_scoring_only"] = candidate_summary
        append_jsonl(lineage_path, run)
        all_case_rows.append(run)
        done.add(key)
        print(json.dumps({"case": ix, "total_cases": MAX_CASES, "status": run["case_status"],
                          "llm_calls_so_far": meter.calls, "cost_usd_so_far": round(meter.spent, 7),
                          "gold_candidate_status_scoring_only": candidate_summary["candidate_presence_status"]}), flush=True)
        if meter.halted or run["case_status"] != "COMPLETED":
            budget_stop = budget_stop or meter.unmetered_error or "budget cap reached"
            break

    # Isolate gold and cohort labels in a sidecar, never in request prompts.
    write_jsonl(OUT / "scoring_gold_only.jsonl", all_scoring)
    completed = [r for r in all_case_rows if r.get("case_status") == "COMPLETED"]
    paired = []
    for run in completed:
        score = scoring_by_key[run["question_key"]]
        baseline_hit = int(score.get("archived_tgar_hits_at_1", 0))
        new_hit = int(run.get("hits_at_1", 0))
        paired.append({"question_key": run["question_key"], "role": roles[run["question_key"]],
                       "archived_hits_at_1": baseline_hit, "rerun_hits_at_1": new_hit,
                       "outcome": "TIE" if baseline_hit == new_hit else ("RERUN_ONLY" if new_hit else "ARCHIVE_ONLY"),
                       **run.get("gold_candidate_diagnostic_scoring_only", {})})
    usage = read_jsonl(request_path) if request_path.exists() else []
    starts = sum(r.get("status") == "REQUEST_STARTED" for r in usage)
    metered = [r for r in usage if r.get("status") == "REQUEST_METERED"]
    total_in = sum(int(r.get("input_tokens", 0) or 0) for r in metered)
    total_out = sum(int(r.get("output_tokens", 0) or 0) for r in metered)
    total_cost = sum(float(r.get("cost_usd", 0.0) or 0.0) for r in metered)
    status = "COMPLETED" if len(completed) == MAX_CASES else ("STOPPED_AT_BUDGET_CAP" if budget_stop else "INCOMPLETE")
    integrity_after = production_source_integrity()
    summary = {
        "experiment": metadata["experiment"], "status": status,
        "cases_expected": MAX_CASES, "cases_completed": len(completed),
        "failure_cases_completed": sum(r.get("cohort_role_scoring_only") == "TGAR_FAILURE" for r in completed),
        "controls_completed": sum(r.get("cohort_role_scoring_only") == "BOTH_CORRECT_CONTROL" for r in completed),
        "rerun_hits_at_1": sum(int(r.get("hits_at_1", 0)) for r in completed),
        "rerun_mean_f1": sum(float(r.get("f1", 0.0)) for r in completed) / len(completed) if completed else None,
        "candidate_presence_status_counts_scoring_only": {},
        "archived_vs_rerun_hit_outcomes_scoring_only": {},
        "requests_started": starts, "metered_requests": len(metered),
        "input_tokens": total_in, "output_tokens": total_out, "cost_usd": total_cost,
        "request_cap": calls_cap, "cost_cap_usd": cost_cap,
        "kg_queries": 1, "kg_query_scope": "single read-only endpoint preflight; no KG queries during cases",
        "budget_stop_reason": budget_stop,
        "frozen_production_source_integrity_after": integrity_after,
        "production_algorithm_changed": False, "independent_holdout": False,
        "limitations": ["Archived baseline comparison is a rerun/stability comparison, not a method treatment comparison.",
                        "The cases were already analyzed and are development data.",
                        "The diagnostic does not itself prove a causal intervention."],
    }
    from collections import Counter
    summary["candidate_presence_status_counts_scoring_only"] = dict(Counter(r["gold_candidate_diagnostic_scoring_only"]["candidate_presence_status"] for r in completed))
    summary["archived_vs_rerun_hit_outcomes_scoring_only"] = dict(Counter(r["outcome"] for r in paired))
    write_jsonl(OUT / "paired_rerun_scoring_only.jsonl", paired)
    write_json(OUT / "results.json", summary)
    metadata["status"] = status
    metadata["requests_started"] = starts
    metadata["cost_usd"] = total_cost
    metadata["input_tokens"] = total_in
    metadata["output_tokens"] = total_out
    metadata["frozen_production_source_integrity_after"] = integrity_after
    write_json(OUT / "run_metadata.json", metadata)
    hashes = {p.name: sha_file(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "artifact_hashes.json"}
    hashes["runner_source_sha256"] = sha_file(Path(__file__))
    write_json(OUT / "artifact_hashes.json", hashes)
    print(json.dumps(summary, indent=2), flush=True)
    if not integrity_after["passed"]:
        raise SystemExit("Frozen production source hash mismatch after diagnostic run")


if __name__ == "__main__":
    main()
