#!/usr/bin/env python3
"""Run the pre-registered 12-case fixed-candidate selector pilot once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
PREP = WORK / "RESULTS/iter_013_selector_pilot_preparation"
REQUESTS = PREP / "requests_non_gold.jsonl"
SCORING = PREP / "scoring_gold_only.jsonl"
DEFAULT_OUT = WORK / "RESULTS/iter_014_selector_pilot"

MODEL_INPUT_PRICE_PER_MILLION = 0.20
MODEL_OUTPUT_PRICE_PER_MILLION = 1.20
MODEL_PRICE_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5.6-luna"
EXPECTED_MODEL = "gpt-5.6-luna"
MAX_CASES = 12
MAX_COMPLETION_TOKENS = 128
TIMEOUT_SECONDS = 120


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def estimate_cost_upper_bound(request: dict[str, Any]) -> float:
    # Reserve four input tokens per UTF-8 character, plus the full output cap.
    # This is deliberately much larger than the actual token count for this
    # mostly ASCII JSON prompt and keeps the cap safe before each request.
    chars = sum(len(message["content"]) for message in request["messages"])
    input_tokens_upper = 4 * chars
    return (
        input_tokens_upper * MODEL_INPUT_PRICE_PER_MILLION / 1_000_000
        + MAX_COMPLETION_TOKENS * MODEL_OUTPUT_PRICE_PER_MILLION / 1_000_000
    )


def response_ids(content: str, allowed: set[str]) -> tuple[list[str], str]:
    try:
        payload = json.loads(content)
    except Exception:
        return [], "INVALID_JSON"
    if not isinstance(payload, dict) or set(payload) != {"answer_ids"}:
        return [], "INVALID_SCHEMA"
    values = payload["answer_ids"]
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        return [], "INVALID_ID_LIST"
    if len(values) != len(set(values)):
        return [], "DUPLICATE_IDS"
    if not set(values).issubset(allowed):
        return [], "OUT_OF_CANDIDATE_ID"
    return values, "VALID"


def f1_score(predicted: set[str], gold: set[str]) -> float:
    if not predicted or not gold:
        return 0.0
    overlap = len(predicted & gold)
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def call_api(
    model: str,
    messages: list[dict[str, str]],
    base_url: str,
    api_key: str,
) -> tuple[str | None, dict[str, int], str | None]:
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning_effort": os.environ.get("OPENAI_REASONING_EFFORT", "low"),
        "stream": False,
    }
    request = Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        with urlopen(request, timeout=TIMEOUT_SECONDS, context=context) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return None, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, f"HTTP_{exc.code}"
    except (URLError, TimeoutError, OSError) as exc:
        return None, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, type(exc).__name__
    except Exception as exc:
        return None, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, type(exc).__name__

    usage_obj = data.get("usage") or {}
    usage = {
        "input_tokens": int(usage_obj.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage_obj.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage_obj.get("total_tokens", 0) or 0),
    }
    choices = data.get("choices") or []
    message = (choices[0].get("message") if choices else {}) or {}
    text = message.get("content")
    if not usage_obj or "prompt_tokens" not in usage_obj or "completion_tokens" not in usage_obj:
        return (text if isinstance(text, str) else None), usage, "MISSING_USAGE"
    return (text if isinstance(text, str) and text.strip() else None), usage, (
        None if isinstance(text, str) and text.strip() else "EMPTY_RESPONSE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="submit the prepared API requests")
    parser.add_argument("--resume", action="store_true", help="continue only cases with no recorded request attempt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("Refusing to submit requests without --run")

    configured_calls = int(os.environ.get("RESEARCH_MAX_LLM_CALLS", "0"))
    configured_cost = float(os.environ.get("RESEARCH_MAX_LLM_COST_USD", "0"))
    if configured_calls <= 0 or configured_calls > MAX_CASES:
        raise SystemExit(f"RESEARCH_MAX_LLM_CALLS must be 1..{MAX_CASES}")
    if configured_cost <= 0:
        raise SystemExit("RESEARCH_MAX_LLM_COST_USD must be explicitly positive")

    # The caller loads the project's .env in memory and passes only its
    # variables through the process environment; this script never prints or
    # writes the API key. Call/cost authorization comes from the invoking process.
    model = os.environ.get("MODEL_NAME", "")
    if model != EXPECTED_MODEL:
        raise SystemExit(f"Expected {EXPECTED_MODEL}; refusing to run with configured model {model or 'UNSET'}")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing from the process environment")
    base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    if "api.openai.com" not in base_url:
        raise SystemExit("Refusing to apply OpenAI's rate table to a non-OpenAI endpoint")

    requests = jsonl(REQUESTS)
    scoring_rows = {row["case_id"]: row for row in jsonl(SCORING)}
    if len(requests) != MAX_CASES or len(scoring_rows) != MAX_CASES:
        raise SystemExit("Prepared packet is not exactly the frozen 12-case cohort")
    if configured_calls != MAX_CASES:
        raise SystemExit("For this preregistered pilot RESEARCH_MAX_LLM_CALLS must equal 12")
    reserve = sum(estimate_cost_upper_bound(request) for request in requests)
    if reserve > configured_cost:
        raise SystemExit(
            f"Conservative full-pilot reserve ${reserve:.6f} exceeds configured cap ${configured_cost:.6f}"
        )

    out = args.output_dir
    os.environ["OPENAI_TIMEOUT"] = str(TIMEOUT_SECONDS)
    os.environ["OPENAI_MAX_RETRIES"] = "1"  # one logical attempt, no wrapper retries
    os.environ["OPENAI_RETRY_INTERVAL"] = "0"
    os.environ.setdefault("OPENAI_REASONING_EFFORT", "low")

    metadata = {
        "experiment": "fresh50 fixed-candidate non-oracle ID selector development pilot",
        "status": "RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "request_count_cap": configured_calls,
        "cost_cap_usd": configured_cost,
        "cost_reserve_upper_bound_usd": reserve,
        "pricing": {
            "input_usd_per_million_tokens": MODEL_INPUT_PRICE_PER_MILLION,
            "output_usd_per_million_tokens": MODEL_OUTPUT_PRICE_PER_MILLION,
            "source": MODEL_PRICE_SOURCE,
        },
        "max_completion_tokens_per_case": MAX_COMPLETION_TOKENS,
        "timeout_seconds_per_case": TIMEOUT_SECONDS,
        "retries": 0,
        "request_sha256": sha256_file(REQUESTS),
        "scoring_file_sha256": sha256_file(SCORING),
        "prompt_gold_leakage": False,
        "independent_holdout": False,
    }
    outputs_path = out / "selector_outputs_non_gold.jsonl"
    trace_path = out / "call_trace.jsonl"
    if args.resume:
        if not out.is_dir() or not (out / "run_metadata.json").exists() or not trace_path.exists():
            raise SystemExit("Cannot resume: existing pilot metadata and call trace are required")
        previous_metadata = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
        for key in ("model", "cost_cap_usd", "request_sha256", "scoring_file_sha256"):
            if previous_metadata.get(key) != metadata.get(key):
                raise SystemExit(f"Cannot resume: pilot {key} differs from the recorded run")
        trace_rows = jsonl(trace_path)
        started_cases = {row["case_id"] for row in trace_rows if row.get("status") == "REQUEST_STARTED"}
        terminal_cases = {
            row["case_id"] for row in trace_rows
            if row.get("status") in {"REQUEST_COMPLETED", "REQUEST_FAILED"}
        }
        if started_cases - terminal_cases:
            raise SystemExit("Cannot resume safely: at least one request has no recorded completion status")
        finished_case_ids = terminal_cases
        calls = len(started_cases)
        spend = sum(float(row.get("cost_usd", 0.0) or 0.0) for row in jsonl(outputs_path)) if outputs_path.exists() else 0.0
        metadata = previous_metadata
        metadata["status"] = "RUNNING"
    else:
        out.mkdir(parents=True, exist_ok=False)
        finished_case_ids: set[str] = set()
        calls = 0
        spend = 0.0
        (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    session_started = time.perf_counter()
    pending = [request for request in requests if request["case_id"] not in finished_case_ids]
    started_all = time.perf_counter()

    for position, request in enumerate(pending, start=1):
        remaining_reserve = sum(estimate_cost_upper_bound(row) for row in pending[position - 1 :])
        if calls >= configured_calls or spend + remaining_reserve > configured_cost:
            append_jsonl(trace_path, {
                "case_id": request["case_id"],
                "status": "STOPPED_BEFORE_REQUEST_BUDGET_RESERVE",
                "calls_attempted": calls,
                "spend_usd": spend,
                "remaining_reserve_usd": remaining_reserve,
            })
            break

        append_jsonl(trace_path, {
            "case_id": request["case_id"],
            "status": "REQUEST_STARTED",
            "prompt_sha256": request["prompt_sha256"],
            "attempt_number": 1,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
        })
        calls += 1
        case_started = time.perf_counter()
        text, usage, api_error = call_api(
            model=model,
            messages=request["messages"],
            base_url=base_url,
            api_key=os.environ["OPENAI_API_KEY"],
        )
        latency = time.perf_counter() - case_started
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        known_cost = (
            input_tokens * MODEL_INPUT_PRICE_PER_MILLION / 1_000_000
            + output_tokens * MODEL_OUTPUT_PRICE_PER_MILLION / 1_000_000
        )
        spend += known_cost
        allowed = set(request["allowed_candidate_ids"])
        selected, validation = response_ids(text or "", allowed) if text else ([], "API_ERROR")
        result_row = {
            "case_id": request["case_id"],
            "answer_ids": selected,
            "validation": validation,
            "prompt_sha256": request["prompt_sha256"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": int(usage.get("total_tokens", input_tokens + output_tokens) or 0),
            "cost_usd": known_cost,
            "latency_seconds": latency,
            "api_error": api_error,
        }
        append_jsonl(outputs_path, result_row)
        metered_response = api_error in {None, "EMPTY_RESPONSE"}
        append_jsonl(trace_path, {
            "case_id": request["case_id"],
            "status": "REQUEST_COMPLETED" if metered_response else "REQUEST_FAILED",
            "api_error": api_error,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": known_cost,
            "cumulative_cost_usd": spend,
            "validation": validation,
            "latency_seconds": latency,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(json.dumps({
            "case": calls,
            "case_cap": configured_calls,
            "status": "completed" if text and not api_error else (
                "empty_metered_response" if api_error == "EMPTY_RESPONSE" else "failed_or_unmetered"
            ),
            "validation": validation,
            "cumulative_cost_usd": round(spend, 8),
        }), flush=True)
        if not metered_response:
            # A transport/HTTP error without usage makes further spend
            # accounting uncertain. Metered empty responses count as a failed
            # case and the preregistered remaining cases may still proceed.
            metadata["status"] = "STOPPED_AFTER_UNMETERED_OR_FAILED_REQUEST"
            metadata["stopped_case_id"] = request["case_id"]
            break
        if spend > configured_cost:
            metadata["status"] = "STOPPED_COST_CAP_REACHED"
            break
    output_rows = jsonl(outputs_path) if outputs_path.exists() else []
    outputs_by_case = {row["case_id"]: row for row in output_rows}
    scored: list[dict[str, Any]] = []
    for request in requests:
        result = outputs_by_case.get(request["case_id"])
        if result is None:
            continue
        score = scoring_rows[request["case_id"]]
        gold = set(score["gold_answer_ids_scoring_only"])
        predicted = set(result.get("answer_ids") or [])
        scored.append({
            "case_id": request["case_id"],
            "question_key": score["question_key"],
            "cohort_role_scoring_only": score["cohort_role_scoring_only"],
            "gold_answer_ids_scoring_only": sorted(gold),
            "predicted_answer_ids": result.get("answer_ids", []),
            "valid_schema_and_membership": result.get("validation") == "VALID",
            "hits_at_1_id_overlap": int(bool(gold & predicted)),
            "f1_id_set": f1_score(predicted, gold),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "cost_usd": result.get("cost_usd", 0.0),
            "api_error": result.get("api_error"),
        })
    scored_path = out / "scored_results_gold_only.jsonl"
    scored_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scored),
        encoding="utf-8",
    )
    failures = [row for row in scored if row["cohort_role_scoring_only"] == "TGAR_FAILURE"]
    controls = [row for row in scored if row["cohort_role_scoring_only"] == "BOTH_CORRECT_CONTROL"]
    valid_count = sum(row["valid_schema_and_membership"] for row in scored)
    if len(scored) == MAX_CASES and calls == MAX_CASES:
        final_status = "COMPLETED"
    else:
        final_status = metadata.get("status") if metadata.get("status") != "RUNNING" else "INCOMPLETE"
    summary = {
        "experiment": metadata["experiment"],
        "status": final_status,
        "created_utc": metadata["created_utc"],
        "model": model,
        "cases_prepared": len(requests),
        "calls_attempted": calls,
        "cases_scored": len(scored),
        "failures_scored": len(failures),
        "controls_scored": len(controls),
        "valid_schema_and_candidate_membership": valid_count,
        "invalid_or_failed_outputs": len(scored) - valid_count,
        "failure_rescues_hits_at_1": sum(row["hits_at_1_id_overlap"] for row in failures),
        "failure_mean_f1": sum(row["f1_id_set"] for row in failures) / len(failures) if failures else None,
        "control_losses_hits_at_1": sum(not row["hits_at_1_id_overlap"] for row in controls),
        "control_mean_f1": sum(row["f1_id_set"] for row in controls) / len(controls) if controls else None,
        "mean_f1_all_scored": sum(row["f1_id_set"] for row in scored) / len(scored) if scored else None,
        "input_tokens": sum(row["input_tokens"] for row in scored),
        "output_tokens": sum(row["output_tokens"] for row in scored),
        "cost_usd": spend,
        "cost_cap_usd": configured_cost,
        "cost_reserve_upper_bound_usd": reserve,
        "mean_latency_seconds": sum(row.get("latency_seconds", 0.0) for row in output_rows) / calls if calls else None,
        "development_gate": "PASS" if len(failures) == 7 and len(controls) == 5 and sum(row["hits_at_1_id_overlap"] for row in failures) >= 5 and sum(not row["hits_at_1_id_overlap"] for row in controls) == 0 else ("INCOMPLETE" if len(scored) != MAX_CASES else "FAIL"),
        "independent_holdout": False,
        "total_wall_seconds": time.perf_counter() - started_all,
        "this_session_wall_seconds": time.perf_counter() - session_started,
        "pricing_source": MODEL_PRICE_SOURCE,
        "llm_calls": calls,
        "read_only_kg_queries": 0,
    }
    metadata["status"] = summary["status"]
    metadata["calls_attempted"] = calls
    metadata["cost_usd"] = spend
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
