"""Authorization-gated, one-shot Iteration 26 attempt-4 paired pilot.

This runner reads only the gold-free cohort. It first generates one frozen
TGAR v1.1.3 trajectory, then forks the same final filter context/candidates to
the candidate-ID selector when the result has exactly one entity-only CoR
filter boundary. All other cases retain the original baseline prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chain_of_relations.schema import Entity  # noqa: E402
from scripts.diagnostics.iter_026_answer_filter.gateway_attempt4 import Attempt4Gateway  # noqa: E402
from scripts.diagnostics.iter_026_answer_filter.strict_runtime_attempt4 import (  # noqa: E402
    answer_case_once,
    make_attempt4_tgar_agent,
    selector_call_or_cap_fallback,
    selector_response_metadata,
)
from scripts.diagnostics.iter_026_answer_filter.selector import (  # noqa: E402
    build_inference_payload,
    candidate_snapshot_sha256,
    validate_selector_response,
)
from scripts.diagnostics.iter_026_answer_filter.transport import (  # noqa: E402
    FreebaseSelectTransport,
    OpenAIChatCompletionsTransport,
)

WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
RESULTS = WORK / "RESULTS/iter_026_answer_selector_readiness"
PREPARATION = RESULTS / "ATTEMPT4_PREPARATION"
COHORT = PREPARATION / "INFERENCE_COHORT_NON_GOLD.jsonl"
MANIFEST = RESULTS / "LIVE_MANIFEST_ATTEMPT4.json"
MANIFEST_SHA_FILE = RESULTS / "LIVE_MANIFEST_ATTEMPT4.sha256"
CONSUMPTION_MARKER = RESULTS / "AUTHORIZATION_CONSUMPTION_ATTEMPT4.json"
MAX_INPUT_TOKENS = 7000
MAX_COMPLETION_TOKENS = 1024
INPUT_PRICE = 0.20
OUTPUT_PRICE = 1.20


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_fsync(path: Path, row: dict[str, Any]) -> None:
    data = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    with path.open("ab", buffering=0) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _approved(manifest_sha: str, approval_sha: str) -> dict[str, Any]:
    frozen_sha = MANIFEST_SHA_FILE.read_text(encoding="utf-8").split()[0]
    if manifest_sha != frozen_sha or sha256_file(MANIFEST) != frozen_sha:
        raise SystemExit("manifest hash mismatch; no transport permitted")
    if approval_sha != frozen_sha:
        raise SystemExit("explicit approval must name the exact frozen manifest SHA-256")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_AWAITING_EXPLICIT_AUTHORIZATION":
        raise SystemExit("manifest is not frozen for authorization")
    frozen_values = {
        "model": "gpt-5.6-luna",
        "attempt_number": 4,
        "api_request_cap": 140,
        "api_cost_cap_usd": 0.368032,
        "api_input_price_usd_per_million_uncached": INPUT_PRICE,
        "api_output_price_usd_per_million": OUTPUT_PRICE,
        "per_request_worst_case_reservation_usd": 0.0026288,
        "full_request_cap_worst_case_reservation_usd": 0.368032,
        "input_tokens_per_request": MAX_INPUT_TOKENS,
        "completion_tokens_per_request": MAX_COMPLETION_TOKENS,
        "sparql_attempt_cap": 300,
        "sparql_endpoint": "http://127.0.0.1:8890/sparql",
        "timeout_seconds_per_request": 120,
        "wall_clock_cap_seconds": 8 * 60 * 60,
        "max_retries": 0,
        "resume_allowed": False,
        "run_directory": "RUNS/iter26_live_attempt4",
        "one_shot_authorization_consumption_marker": "AUTHORIZATION_CONSUMPTION_ATTEMPT4.json",
        "authorization_consumed": False,
    }
    if any(manifest.get(key) != value for key, value in frozen_values.items()):
        raise SystemExit("manifest configuration differs from the frozen attempt-4 caps")
    if sha256_file(PREPARATION / "COHORT_MANIFEST.json") != manifest.get("cohort_manifest_sha256"):
        raise SystemExit("cohort manifest hash mismatch; no transport permitted")
    if sha256_file(COHORT) != manifest.get("inference_cohort_sha256"):
        raise SystemExit("gold-free inference cohort hash mismatch; no transport permitted")
    for relative, expected in {
        **manifest.get("source_files_sha256", {}),
        **manifest.get("diagnostic_files_sha256", {}),
    }.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"frozen source hash mismatch for {relative}; no transport permitted")
    return manifest


def _selector_eligibility(
    result: dict[str, Any], question: str,
    history: list[dict[str, Any]], sparql: list[dict[str, Any]],
) -> tuple[bool, str, dict[str, Any] | None, list[dict[str, Any]]]:
    filters = [row for row in history if row.get("type") == "filter_answer"]
    if str(result.get("router_execution_mode", "cor")).casefold() != "cor":
        return False, "NON_COR_ROUTER_MODE", None, []
    if len(filters) != 1:
        return False, "FILTER_BOUNDARY_COUNT_NOT_ONE", None, []
    filter_event = filters[0]
    if re.search(r"\b(how many|how much|number of|total number|count of|average|sum of|maximum|minimum)\b", question, re.IGNORECASE):
        return False, "AGGREGATE_ANSWER_NOT_ENTITY_SELECTABLE", filter_event, []
    baseline_items = result.get("results")
    if isinstance(baseline_items, list) and baseline_items and any(
        not str(item.get("id", "")).strip()
        for item in baseline_items if isinstance(item, dict)
    ):
        return False, "BASELINE_RESULT_NOT_ENTITY_ID_ONLY", filter_event, []
    if not isinstance(filter_event.get("prompt"), str) or not filter_event["prompt"].strip():
        return False, "FILTER_CONTEXT_UNAVAILABLE", None, []
    try:
        filter_trace_id = int(filter_event["trace_id"])
    except (KeyError, TypeError, ValueError):
        return False, "FILTER_TRACE_ID_UNAVAILABLE", filter_event, []
    searches = [
        row for row in sparql
        if row.get("search_type") == "entity_search"
        and isinstance(row.get("trace_id"), int)
        and int(row["trace_id"]) < filter_trace_id
    ]
    if not searches:
        return False, "ENTITY_SEARCH_SNAPSHOT_UNAVAILABLE", None, []
    search = searches[-1]
    ids = search.get("candidate_entity_ids") or []
    labels = search.get("candidate_entity_names") or []
    literals = search.get("literal_candidates") or []
    if literals:
        return False, "LITERAL_CANDIDATES_NOT_REPRESENTABLE_AS_ENTITY_IDS", filter_event, []
    if not ids or len(ids) != len(labels):
        return False, "CANDIDATE_ID_LABEL_ALIGNMENT_INVALID", filter_event, []
    if len(ids) != len(set(ids)) or any(not str(value).strip() for value in ids + labels):
        return False, "CANDIDATE_IDS_OR_LABELS_AMBIGUOUS", filter_event, []
    if int(filter_event.get("candidate_size", -1)) != len(ids):
        return False, "FILTER_AND_ENTITY_CANDIDATE_COUNTS_DIFFER", filter_event, []
    query_hash = hashlib.sha256(str(search.get("sparql", "")).encode("utf-8")).hexdigest()
    source_event = {
        "entity_search_trace_id": search.get("trace_id"),
        "entity_search_query_sha256": query_hash,
    }
    candidates = [
        {"id": entity_id, "label": label, "source": {**source_event, "position": position}}
        for position, (entity_id, label) in enumerate(zip(ids, labels))
    ]
    try:
        candidate_snapshot_sha256(candidates)
    except (TypeError, ValueError):
        return False, "CANDIDATE_SNAPSHOT_INVALID", filter_event, []
    return True, "ELIGIBLE_SINGLE_ENTITY_COR_FILTER", filter_event, candidates


def _baseline_ids(result: dict[str, Any]) -> list[str]:
    items = result.get("results")
    if not isinstance(items, list):
        return []
    return list(dict.fromkeys(
        str(item.get("id", "")).strip()
        for item in items if isinstance(item, dict) and str(item.get("id", "")).strip()
    ))


def run(manifest_sha: str, approved_sha: str, output_dir: Path) -> dict[str, Any]:
    manifest = _approved(manifest_sha, approved_sha)
    output_dir = output_dir.resolve()
    expected_output = (RESULTS / manifest["run_directory"]).resolve()
    if output_dir.resolve() != expected_output:
        raise SystemExit("output directory differs from the exact frozen run directory")
    if output_dir.exists():
        raise SystemExit("output directory already exists; this run has no resume mode")
    if CONSUMPTION_MARKER.exists():
        raise SystemExit("this frozen authorization has already been consumed; no resume or rerun")
    rows = read_jsonl(COHORT)
    if len(rows) != 12 or [row["question_key"] for row in rows] != manifest["question_keys"]:
        raise SystemExit("gold-free cohort differs from the frozen authorization manifest")
    if any(any("gold" in key.casefold() or "scoring" in key.casefold() for key in row) for row in rows):
        raise SystemExit("gold/scoring field appeared in inference input")
    if any(
        not isinstance(row.get("topic_entities"), list)
        or not isinstance(row.get("topic_entity_names"), list)
        or len(row["topic_entities"]) != len(row["topic_entity_names"])
        or not row["topic_entities"]
        for row in rows
    ):
        raise SystemExit("topic entity IDs/names are missing or misaligned in the frozen cohort")

    # Read local credentials only after exact manifest approval is verified.
    from scripts.run_cor_baseline_experiment import load_safe_project_env
    load_safe_project_env()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is unavailable; no transport has started")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")
    if base_url and base_url != "https://api.openai.com/v1":
        raise SystemExit("Iteration 26 is frozen to the official OpenAI API base URL")

    RESULTS.mkdir(parents=True, exist_ok=True)
    marker = {
        "iteration": 26,
        "manifest_sha256": manifest_sha,
        "approved_manifest_sha256": approved_sha,
        "status": "CONSUMED_BEFORE_FIRST_TRANSPORT; NO_RESUME",
        "run_directory": str(output_dir.relative_to(RESULTS)),
        "recorded_time_epoch": time.time(),
    }
    marker_bytes = (json.dumps(marker, sort_keys=True, indent=2) + "\n").encode("utf-8")
    try:
        descriptor = os.open(CONSUMPTION_MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise SystemExit("this frozen authorization was consumed concurrently; no transport") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(marker_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(RESULTS, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    output_dir.mkdir(parents=True, exist_ok=False)
    model_transport = OpenAIChatCompletionsTransport(api_key, timeout_seconds=120)
    sparql_transport = FreebaseSelectTransport(timeout_seconds=120)
    gateway = Attempt4Gateway(
        output_dir,
        manifest_sha256=manifest_sha,
        model_transport=model_transport,
        sparql_transport=sparql_transport,
        api_request_cap=manifest["api_request_cap"],
        api_cost_cap_usd=manifest["api_cost_cap_usd"],
        input_tokens_per_request=manifest["input_tokens_per_request"],
        output_tokens_per_request=manifest["completion_tokens_per_request"],
        input_usd_per_million=INPUT_PRICE,
        output_usd_per_million=OUTPUT_PRICE,
        sparql_attempt_cap=manifest["sparql_attempt_cap"],
        timeout_seconds=manifest["timeout_seconds_per_request"],
        wall_seconds=manifest["wall_clock_cap_seconds"],
    )
    run_info = {
        "iteration": 26,
        "manifest_sha256": manifest_sha,
        "cohort_sha256": sha256_file(COHORT),
        "model": manifest["model"],
        "evidence_class": "NEW_DEVELOPMENT_EVALUATION",
        "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gold_loaded_by_live_process": False,
        "status": "RUNNING",
    }
    (output_dir / "RUN_METADATA.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    run_rows: list[dict[str, Any]] = []
    current_case_id: str | None = None
    trace_path = output_dir / "CASE_TRACES.jsonl"
    result_path = output_dir / "PAIRED_RESULTS_NON_GOLD.jsonl"
    progress_path = output_dir / "CASE_PROGRESS.jsonl"
    try:
        agent = make_attempt4_tgar_agent(gateway)
        preflight = gateway.select_once("SELECT * WHERE { ?s ?p ?o } LIMIT 1")
        (output_dir / "PREFLIGHT.json").write_text(json.dumps({
            "status": "HTTP_OK", "binding_count": len(preflight["bindings"]),
            "raw_response_sha256": preflight.get("raw_response_sha256"),
        }, indent=2) + "\n", encoding="utf-8")
        for item in rows:
            if gateway.halted:
                raise SystemExit("gateway halted; do not resume or submit later cases")
            current_case_id = item["case_id"]
            gateway.set_case_context(current_case_id)
            append_fsync(progress_path, {"event": "CASE_STARTED", "case_id": current_case_id})
            entities = [
                Entity(str(entity_id), str(name))
                for entity_id, name in zip(item["topic_entities"], item["topic_entity_names"])
                if str(entity_id).strip() and str(name).strip()
            ]
            history: list[dict[str, Any]] = []
            sparql_history: list[dict[str, Any]] = []
            step_history: list[dict[str, Any]] = []
            api_before, sparql_before = gateway.api_starts, gateway.sparql_starts
            input_before, output_before = gateway.api_input_tokens, gateway.api_output_tokens
            cost_before, reserved_before = gateway.api_actual_cost_usd, gateway.api_reserved_usd
            started = time.monotonic()
            result, upstream_failure = answer_case_once(
                agent,
                question=item["question"], topic_entities=entities,
                prompt_history=history, sparql_history=sparql_history,
                step_history=step_history,
            )
            if upstream_failure is not None:
                failure_kind = upstream_failure.get("failure_kind")
                case_failure_status = (
                    "BASELINE_INPUT_CAP_EXCEEDED_NO_OUTPUT"
                    if failure_kind == "INPUT_CAP_EXCEEDED"
                    else "UPSTREAM_EMPTY_OR_TRUNCATED_FAILURE"
                )
                result = {
                    "results": [],
                    "router_execution_mode": "cor",
                    "error": case_failure_status,
                }
            else:
                case_failure_status = "BASELINE_COMPLETED"
            baseline_ids = _baseline_ids(result)
            if upstream_failure is not None:
                eligible, eligibility_reason, filter_event, candidates = (
                    False, "UPSTREAM_BASELINE_INCOMPLETE_NO_SELECTOR_COMPARISON", None, []
                )
            else:
                eligible, eligibility_reason, filter_event, candidates = _selector_eligibility(
                    result, item["question"], history, sparql_history,
                )
            selector_row: dict[str, Any] | None = None
            intervention_ids = list(baseline_ids)
            if eligible and filter_event is not None:
                payload = build_inference_payload(
                    question=item["question"],
                    frozen_filter_context=filter_event["prompt"],
                    frozen_filter_system_context=agent.cor_agent._get_system_prompt("filter"),
                    candidates=candidates,
                )
                messages = [{
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                }]
                api_result, selector_input_cap_failure = selector_call_or_cap_fallback(
                    gateway, messages,
                    max_completion_tokens=manifest["completion_tokens_per_request"],
                )
                if selector_input_cap_failure is not None:
                    selector_row = {
                        "status": "SKIPPED_INPUT_CAP_EXCEEDED",
                        "valid": False,
                        "selected_ids": list(baseline_ids),
                        "prediction_source": "UNCHANGED_BASELINE_FALLBACK",
                        "response_sha256": None,
                        "error": selector_input_cap_failure["status"],
                        "candidate_snapshot_sha256": candidate_snapshot_sha256(candidates),
                        "candidate_membership_valid": False,
                        "api_request_sent": False,
                        "input_cap_failure": selector_input_cap_failure,
                    }
                else:
                    selector_result = validate_selector_response(
                        api_result.get("content"), candidates=candidates,
                        expected_candidate_snapshot_sha256=candidate_snapshot_sha256(candidates),
                        baseline_prediction=baseline_ids,
                    )
                    if selector_result.prediction_source == "INTERVENTION_SELECTOR":
                        intervention_ids = list(selector_result.selected_ids)
                    selector_row = {
                        "status": selector_result.status,
                        "valid": selector_result.valid,
                        "selected_ids": list(selector_result.selected_ids),
                        "prediction_source": selector_result.prediction_source,
                        "response_sha256": selector_result.response_sha256,
                        "error": selector_result.error,
                        "candidate_snapshot_sha256": candidate_snapshot_sha256(candidates),
                        "candidate_membership_valid": (
                            selector_result.status == "VALID_SELECTION"
                            and set(selector_result.selected_ids).issubset({candidate["id"] for candidate in candidates})
                        ),
                        "api_request_sent": True,
                        "response_metadata": selector_response_metadata(api_result),
                    }
            row = {
                "case_id": item["case_id"],
                "question_key": item["question_key"],
                "dataset": item["dataset"],
                "baseline_prediction_ids": baseline_ids,
                "intervention_prediction_ids": intervention_ids,
                "router_execution_mode": result.get("router_execution_mode", "cor"),
                "filter_event_count": sum(event.get("type") == "filter_answer" for event in history),
                "selector_eligible": eligible,
                "selector_eligibility_reason": eligibility_reason,
                "candidate_ids_snapshot": [candidate["id"] for candidate in candidates] if candidates else [],
                "selector_invoked": selector_row is not None,
                "selector_api_request_sent": bool(selector_row and selector_row.get("api_request_sent")),
                "selector": selector_row,
                "prediction_source": (
                    selector_row["prediction_source"] if selector_row else
                    "PAIRED_UPSTREAM_FAILURE_NO_OUTPUT" if upstream_failure is not None else
                    "UNCHANGED_BASELINE_NOT_ELIGIBLE"
                ),
                "execution_status": (
                    case_failure_status
                ),
                "upstream_failure_metadata": upstream_failure,
                "api_requests_started": gateway.api_starts - api_before,
                "api_input_tokens": gateway.api_input_tokens - input_before,
                "api_output_tokens": gateway.api_output_tokens - output_before,
                "api_actual_cost_usd": gateway.api_actual_cost_usd - cost_before,
                "api_reserved_cost_usd": gateway.api_reserved_usd - reserved_before,
                "sparql_attempts_started": gateway.sparql_starts - sparql_before,
                "latency_seconds": time.monotonic() - started,
                "error": result.get("error"),
            }
            trace = {**row, "prompt_history": history, "sparql_history": sparql_history, "step_history": step_history}
            append_fsync(trace_path, trace)
            append_fsync(result_path, row)
            run_rows.append(row)
            append_fsync(progress_path, {"event": "CASE_FINISHED", "case_id": current_case_id})
            if gateway.halted:
                raise SystemExit("gateway halted after an uncertain call; later questions were not submitted")
            current_case_id = None
            gateway.set_case_context(None)
    except BaseException as exc:
        incomplete_case = current_case_id if current_case_id and current_case_id not in {
            row["case_id"] for row in run_rows
        } else None
        if incomplete_case:
            append_fsync(output_dir / "INCOMPLETE_CASES.jsonl", {
                "case_id": incomplete_case,
                "status": "INTERRUPTED_NO_RESUME",
                "error_type": type(exc).__name__,
                "api_requests_started": gateway.api_starts,
                "physical_sparql_attempts": gateway.sparql_starts,
            })
        run_info.update({
            "status": "ABORTED_NO_RESUME",
            "evidence_class": "INCOMPLETE_LIVE_ATTEMPT",
            "cases_completed": len(run_rows),
            "cases_started": len(run_rows) + int(incomplete_case is not None),
            "current_case_id": incomplete_case,
            "abort_error_type": type(exc).__name__,
            "api_requests_started": gateway.api_starts,
            "api_input_tokens": gateway.api_input_tokens,
            "api_output_tokens": gateway.api_output_tokens,
            "api_actual_cost_usd": gateway.api_actual_cost_usd,
            "api_reserved_cost_usd": gateway.api_reserved_usd,
            "physical_sparql_attempts": gateway.sparql_starts,
        })
        (output_dir / "RUN_METADATA.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
        raise
    run_info.update({
        "status": "COMPLETE" if len(run_rows) == len(rows) else "INCOMPLETE_NO_RESUME",
        "evidence_class": "NEW_DEVELOPMENT_EVALUATION" if len(run_rows) == len(rows) else "INCOMPLETE_LIVE_ATTEMPT",
        "cases_completed": len(run_rows),
        "cases_started": len(rows),
        "current_case_id": None,
        "api_requests_started": gateway.api_starts,
        "api_cost_reserved_usd": gateway.api_reserved_usd,
        "api_input_tokens": gateway.api_input_tokens,
        "api_output_tokens": gateway.api_output_tokens,
        "api_actual_cost_usd": gateway.api_actual_cost_usd,
        "physical_sparql_attempts": gateway.sparql_starts,
        "gold_loaded_by_live_process": False,
        "completed_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "upstream_empty_or_truncated_case_count": sum(
            row.get("execution_status") == "UPSTREAM_EMPTY_OR_TRUNCATED_FAILURE" for row in run_rows
        ),
        "input_cap_exceeded_case_count": sum(
            row.get("execution_status") == "BASELINE_INPUT_CAP_EXCEEDED_NO_OUTPUT" for row in run_rows
        ),
        "selector_input_cap_skip_count": sum(
            (row.get("selector") or {}).get("status") == "SKIPPED_INPUT_CAP_EXCEEDED"
            for row in run_rows
        ),
    })
    (output_dir / "RUN_METADATA.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    return run_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="submit the separately authorized attempt-4 one-shot pilot")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--approved-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("no transport: pass --run only after explicit authorization of this exact manifest")
    print(json.dumps(run(args.manifest_sha256, args.approved_manifest_sha256, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
