"""One-shot Iteration 29 terminal selector runner, fresh attempt 3.

This module contains no gold-data reader or scorer. The live path requires a
separate authorization record naming the exact frozen manifest hash. Imports
of model/KG runtime code, credentials, and transports happen only after that
check. Request-level HTTP and transport failures are recorded as one-shot ITT
failures or selector fallbacks; they never trigger retries. An independent
attempt-3 scoring gate is invoked after durable run artifacts are written.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from scripts.diagnostics.iter_029_terminal_selector.path_parser_attempt2 import parse_executed_relation_ids


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_028_terminal_boundary_repair"
MANIFEST_PATH = RESULTS / "NEXT_MANIFEST_DRAFT_ATTEMPT3.json"
COHORT_PATH = RESULTS / "NEXT_INFERENCE_COHORT_ATTEMPT3_NON_GOLD.jsonl"
PROMPT_PATH = RESULTS / "TERMINAL_SELECTOR_PROMPT_V1.txt"
GATE_PATH = ROOT / "scripts/diagnostics/iter_029_terminal_selector/scoring_gate_attempt3.py"
RUN_RELATIVE = "RUNS/iter29_terminal_candidate_selector_attempt3"
FAILURE_TREATMENT = "ITT_EMPTY_BASELINE_REQUEST_FAILURE_AND_SELECTOR_BASELINE_FALLBACK_V2"
AGGREGATE_QUESTION = re.compile(
    r"\b(how many|how much|number of|total number|count of|average|sum of|maximum|minimum)\b",
    re.IGNORECASE,
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def manifest_digest(manifest: dict[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return sha256_bytes(canonical_json(unsigned))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_fsync(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    existed = path.exists()
    with path.open("ab", buffering=0) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if not existed:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_json_fsync(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def write_json_replace(path: Path, value: Any) -> None:
    """Durably replace an existing mutable run-status file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with temp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def candidate_snapshot_sha256(ids: list[str], labels: list[str]) -> str:
    if len(ids) != len(labels) or not ids:
        return ""
    return sha256_bytes(canonical_json([
        {"id": str(entity_id), "label": str(label), "position": index}
        for index, (entity_id, label) in enumerate(zip(ids, labels))
    ]))


class CaseLocalApiFailure(RuntimeError):
    """One API attempt failed; its reservation is charged and never retried."""

    def __init__(self, failure_kind: str, *, http_status: int | None = None,
                 error_code: str | None = None, error_type: str | None = None,
                 raw_response_sha256: str | None = None) -> None:
        self.failure_kind = failure_kind
        self.http_status = http_status
        self.error_code = error_code
        self.error_type = error_type
        self.raw_response_sha256 = raw_response_sha256
        super().__init__(failure_kind)

    def failure_record(self) -> dict[str, Any]:
        return {
            "failure_kind": self.failure_kind,
            "response_metadata": {
                "finish_reason": "http_error" if self.failure_kind == "API_HTTP_ERROR" else "transport_error",
                "content_nonempty": False,
                **({"http_status": self.http_status} if self.http_status is not None else {}),
                **({"error_code": self.error_code} if self.error_code else {}),
                **({"error_type": self.error_type} if self.error_type else {}),
                **({"raw_response_sha256": self.raw_response_sha256} if self.raw_response_sha256 else {}),
            },
        }


def _manifest_sources_ok(manifest: dict[str, Any]) -> None:
    implementation = manifest.get("implementation", {})
    for relative, expected in implementation.get("source_files_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen Iteration 29 source hash mismatch: {relative}")


def verify_inputs(manifest: dict[str, Any], cohort: list[dict[str, Any]]) -> None:
    expected_hash = manifest.get("manifest_sha256")
    if not expected_hash or expected_hash != manifest_digest(manifest):
        raise RuntimeError("manifest hash is missing or invalid")
    if manifest.get("iteration") != 29 or manifest.get("dataset") != "GrailQA" or manifest.get("split") != "dev":
        raise RuntimeError("manifest dataset/iteration differs from the frozen Iteration 29 protocol")
    if manifest.get("failure_treatment") != FAILURE_TREATMENT:
        raise RuntimeError("frozen failure treatment differs")
    if manifest.get("inference_cohort_path") != COHORT_PATH.relative_to(ROOT).as_posix():
        raise RuntimeError("manifest inference cohort path differs from the runner's fixed gold-free cohort")
    keys = manifest.get("question_keys")
    if not isinstance(keys, list) or len(keys) != 800 or len(keys) != len(set(keys)):
        raise RuntimeError("manifest must contain exactly 800 unique ordered question keys")
    if sha256_bytes("\n".join(keys).encode("utf-8")) != manifest.get("question_keys_sha256"):
        raise RuntimeError("ordered question-key digest mismatch")
    if [row.get("question_key") for row in cohort] != keys:
        raise RuntimeError("gold-free cohort identity/order does not match the manifest")
    if sha256_file(COHORT_PATH) != manifest.get("inference_cohort_sha256"):
        raise RuntimeError("gold-free inference cohort hash mismatch")
    if sha256_file(PROMPT_PATH) != manifest.get("intervention", {}).get("prompt_sha256"):
        raise RuntimeError("selector prompt hash mismatch")
    expected_rows = manifest.get("per_question_safe_row_sha256")
    if not isinstance(expected_rows, list) or len(expected_rows) != len(cohort):
        raise RuntimeError("safe-row hash inventory missing or malformed")
    for index, row in enumerate(cohort):
        if row.get("safe_row_sha256") != expected_rows[index]:
            raise RuntimeError(f"safe-row hash mismatch at cohort position {index}")
        if any(any(term in str(key).casefold() for term in ("gold", "answer_id", "logical_form", "sparql_gold", "scoring")) for key in row):
            raise RuntimeError("gold/scoring field found in live inference cohort")
        if not isinstance(row.get("topic_entities"), list) or not row["topic_entities"]:
            raise RuntimeError("topic entity IDs missing from cohort row")
        if len(row["topic_entities"]) != len(row.get("topic_entity_names", [])):
            raise RuntimeError("topic entity IDs/names misaligned")
    # Do not open the source dataset at execution time: it contains gold fields.
    # The exact non-gold inference cohort and its per-row hashes are the only
    # question inputs read by this runner.
    api = manifest.get("api_caps", {})
    sparql = manifest.get("sparql_caps", {})
    per_request_reservation = (
        api.get("per_request_input_token_cap", 0) * api.get("rate_assumption_input_uncached_usd_per_million", 0)
        + api.get("per_request_completion_token_cap", 0) * api.get("rate_assumption_output_usd_per_million", 0)
    ) / 1_000_000
    if (
        api.get("model") != "gpt-5.6-luna"
        or api.get("max_retries") != 0
        or api.get("resume_allowed") is not False
        or api.get("per_request_timeout_seconds") != 120
        or api.get("api_request_cap", 0) <= 0
        or api.get("cost_cap_usd", -1) + 1e-10 < api.get("api_request_cap", 0) * per_request_reservation
        or sparql.get("endpoint") != "http://127.0.0.1:8890/sparql"
        or sparql.get("mutations_allowed") is not False
        or sparql.get("preflight_selects_included") != 1
        or sparql.get("physical_read_only_select_cap", 0) < 1
        or sparql.get("per_case_select_cap", 0) < 1
    ):
        raise RuntimeError("manifest resource limits differ from the frozen one-shot protocol")
    _manifest_sources_ok(manifest)


def validate_authorization(manifest: dict[str, Any], manifest_sha: str, record: dict[str, Any]) -> None:
    if manifest.get("status") != "FROZEN_WITH_STANDING_USER_AUTHORIZATION":
        raise RuntimeError("manifest is not frozen under the standing user authorization")
    if record.get("authorized") is not True:
        raise RuntimeError("authorization record does not explicitly grant the run")
    if record.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("authorization record does not name the exact manifest hash")
    if not isinstance(record.get("authorization_text"), str) or not record["authorization_text"].strip():
        raise RuntimeError("verbatim authorization text is required")
    api = manifest["api_caps"]
    sparql = manifest["sparql_caps"]
    caps = record.get("accepted_caps")
    expected = {
        "model": api["model"],
        "api_request_cap": api["api_request_cap"],
        "per_request_input_token_cap": api["per_request_input_token_cap"],
        "per_request_completion_token_cap": api["per_request_completion_token_cap"],
        "cost_cap_usd": api["cost_cap_usd"],
        "physical_read_only_select_cap": sparql["physical_read_only_select_cap"],
        "endpoint": sparql["endpoint"],
        "per_request_timeout_seconds": api["per_request_timeout_seconds"],
        "wall_clock_cap_hours": manifest["time_caps"]["wall_clock_cap_hours"],
        "max_retries": api["max_retries"],
        "resume_allowed": api["resume_allowed"],
    }
    if caps != expected:
        raise RuntimeError("authorization record caps differ from the manifest")


def _baseline_ids(result: dict[str, Any]) -> list[str]:
    items = result.get("results")
    if not isinstance(items, list):
        return []
    ids: list[str] = []
    for item in items:
        if isinstance(item, dict):
            entity_id = str(item.get("id", "")).strip()
            if entity_id and entity_id not in ids:
                ids.append(entity_id)
    return ids


def _baseline_names(result: dict[str, Any]) -> list[str]:
    items = result.get("results")
    if not isinstance(items, list):
        return []
    names: list[str] = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name and name not in names:
                names.append(name)
    return names


def _route(result: dict[str, Any]) -> str:
    mode = str(result.get("final_execution_mode", result.get("router_execution_mode", "cor"))).casefold()
    if mode in {"cor", "cor_fallback"}:
        return "cor"
    return mode


def _terminal(prompt_history: list[dict[str, Any]]) -> tuple[str, int | None, dict[str, Any] | None]:
    events = [event for event in prompt_history if isinstance(event, dict) and isinstance(event.get("trace_id"), int)]
    if not events:
        return "empty/error", None, None
    event = max(events, key=lambda item: item["trace_id"])
    event_type = event.get("type")
    if event_type == "reasoning" and str(event.get("parsed_result", "")).casefold() == "stop":
        return "direct-LLM-reasoning-STOP", event["trace_id"], event
    if event_type == "filter_answer":
        return "graph-filter", event["trace_id"], event
    if event_type == "generate_without_explored_paths":
        return "direct-LLM-generate-directly", event["trace_id"], event
    return "unsupported-terminal", event["trace_id"], event


def _derive_observation(
    *, row: dict[str, Any], result: dict[str, Any] | None, failure: dict[str, Any] | None,
    prompt_history: list[dict[str, Any]], sparql_history: list[dict[str, Any]], gateway: Any,
    manifest: dict[str, Any], prompt_text: str,
) -> dict[str, Any]:
    case_id = str(row["case_id"])
    if failure:
        raw_kind = str(failure.get("failure_kind", "")).upper()
        response_meta = failure.get("response_metadata", failure)
        finish = str(response_meta.get("finish_reason", "")).casefold()
        if raw_kind == "INPUT_CAP_EXCEEDED":
            failure_kind, finish = "input_cap", "not_sent"
        elif raw_kind == "API_HTTP_ERROR":
            failure_kind, finish = "http_error", "http_error"
        elif raw_kind == "API_TRANSPORT_ERROR":
            failure_kind, finish = "transport_error", "transport_error"
        elif finish == "length":
            failure_kind = "length"
        elif not response_meta.get("content_nonempty", not response_meta.get("content_empty", False)) or finish == "empty":
            failure_kind, finish = "empty", "empty"
        else:
            raise RuntimeError("baseline failure is not covered by the preregistered ITT policy")
        result = {"results": [], "final_execution_mode": "cor"}
        baseline_status = "BASELINE_FAILURE_ITT"
        baseline_response = {
            "finish_reason": finish,
            "content_nonempty": False,
            **{key: response_meta[key] for key in ("http_status", "error_code", "error_type") if key in response_meta},
        }
    else:
        result = result or {}
        if str(result.get("action", "")).casefold() == "error":
            raise RuntimeError("baseline returned an unclassified runtime error; abort without a scored placeholder")
        baseline_status = "BASELINE_COMPLETED"
        baseline_response = {"finish_reason": "stop", "content_nonempty": True}
        failure_kind = None

    ids = _baseline_ids(result)
    names = _baseline_names(result)
    terminal_source, terminal_trace, terminal_event = _terminal(prompt_history) if not failure else ("empty/error", None, None)
    filters = [event for event in prompt_history if event.get("type") == "filter_answer"]
    candidate_searches = [
        event for event in sparql_history
        if event.get("search_type") == "entity_search"
        and isinstance(event.get("trace_id"), int)
        and terminal_trace is not None and event["trace_id"] < terminal_trace
    ]
    search = candidate_searches[-1] if candidate_searches else {}
    candidate_ids = [str(value).strip() for value in search.get("candidate_entity_ids", [])]
    candidate_labels = [str(value).strip() for value in search.get("candidate_entity_names", [])]
    literals = list(search.get("literal_candidates", []) or [])
    query = str(search.get("sparql", ""))
    query_sha = sha256_bytes(query.encode("utf-8")) if query else ""
    query_event = gateway.successful_query_by_sha.get(query_sha) if query_sha else None
    query_status = str(search.get("status", "missing"))
    route = _route(result)
    relations = parse_executed_relation_ids(result.get("reasoning_chains", ""))
    unique_ids = len(candidate_ids) == len(set(candidate_ids)) and all(candidate_ids)
    unique_labels = len(candidate_labels) == len(set(candidate_labels)) and all(candidate_labels)
    baseline_names_to_ids_valid = bool(ids and names and len(ids) == len(names) and len(set(names)) == len(names))
    if baseline_names_to_ids_valid:
        by_label = dict(zip(candidate_labels, candidate_ids))
        baseline_names_to_ids_valid = all(name in by_label for name in names) and [by_label.get(name) for name in names] == ids
    prompt_payload = {
        "question": row["question"],
        "executed_relation_path": str(result.get("reasoning_chains", "")),
        "executed_relation_ids": relations,
        "baseline_answer_ids": ids,
        "candidate_entities": [
            {"id": entity_id, "label": label}
            for entity_id, label in zip(candidate_ids, candidate_labels)
        ],
    }
    message = [{"role": "user", "content": prompt_text.rstrip() + "\n\nINPUT:\n" + json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}]
    prompt_estimate = int(gateway.token_counter(message))
    max_candidates = manifest["intervention"]["max_candidate_count"]
    eligible = (
        baseline_status == "BASELINE_COMPLETED"
        and route == "cor"
        and terminal_source in {"direct-LLM-reasoning-STOP", "graph-filter"}
        and (terminal_source != "graph-filter" or len(filters) == 1)
        and len(filters) <= 1
        and not AGGREGATE_QUESTION.search(str(row["question"]))
        and bool(ids) and all(entity_id in candidate_ids for entity_id in ids)
        and baseline_names_to_ids_valid
        and bool(relations)
        and unique_ids and unique_labels
        and len(candidate_ids) >= 2 and len(candidate_ids) <= max_candidates
        and len(candidate_ids) == len(candidate_labels)
        and not literals
        and query_status == "ok" and not search.get("timed_out", False)
        and query_event is not None and query_event.get("status") == "ok"
        and prompt_estimate <= manifest["intervention"]["max_input_tokens"]
    )
    return {
        "case_id": case_id,
        "question_key": row["question_key"],
        "question": row["question"],
        "route": route,
        "terminal_source": terminal_source,
        "terminal_trace_id": terminal_trace,
        "terminal_event": terminal_event or {},
        "filter_event_count": len(filters),
        "baseline_status": baseline_status,
        "baseline_failure_kind": failure_kind,
        "baseline_response": baseline_response,
        "baseline_prediction_ids": ids,
        "baseline_answer_names": names,
        "answer_kind": "entity" if ids and baseline_names_to_ids_valid and not literals else "unknown",
        "candidate_ids": candidate_ids,
        "candidate_labels": candidate_labels,
        "literal_candidates": literals,
        "candidate_query_attempt_id": query_event.get("attempt_id", "") if query_event else "",
        "candidate_query_status": query_status,
        "candidate_query_sha256": query_sha,
        "candidate_snapshot_sha256": candidate_snapshot_sha256(candidate_ids, candidate_labels),
        "selector_prompt_input_token_estimate": prompt_estimate,
        "selector_eligible": eligible,
        "selector_eligibility_reason": "ELIGIBLE" if eligible else "FROZEN_TRIGGER_NOT_MET",
        "executed_relation_path": str(result.get("reasoning_chains", "")),
        "executed_relation_ids": relations,
        "prompt_payload": prompt_payload,
    }


def _parse_selector(content: Any, candidate_ids: list[str], baseline_ids: list[str]) -> tuple[dict[str, Any], list[str]]:
    fallback = copy.deepcopy(baseline_ids)
    if not isinstance(content, str) or not content.strip():
        return {"status": "INVALID_FALLBACK_BASELINE", "fallback_reason": "EMPTY_RESPONSE", "selected_ids": []}, fallback
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return {"status": "INVALID_FALLBACK_BASELINE", "fallback_reason": "INVALID_JSON", "selected_ids": []}, fallback
    if not isinstance(payload, dict) or set(payload) != {"status", "selected_ids"}:
        return {"status": "INVALID_FALLBACK_BASELINE", "fallback_reason": "INVALID_SCHEMA", "selected_ids": []}, fallback
    status, selected = payload.get("status"), payload.get("selected_ids")
    if status == "UNSURE" and selected == []:
        return {"status": "INVALID_FALLBACK_BASELINE", "fallback_reason": "UNSURE", "selected_ids": []}, fallback
    if status != "SELECT" or not isinstance(selected, list) or not selected:
        return {"status": "INVALID_FALLBACK_BASELINE", "fallback_reason": "INVALID_SELECTION", "selected_ids": []}, fallback
    if any(not isinstance(value, str) or not value.strip() for value in selected):
        return {"status": "INVALID_FALLBACK_BASELINE", "fallback_reason": "INVALID_ID", "selected_ids": []}, fallback
    if len(selected) != len(set(selected)):
        return {"status": "INVALID_FALLBACK_BASELINE", "fallback_reason": "DUPLICATE_ID", "selected_ids": []}, fallback
    if not set(selected).issubset(set(candidate_ids)):
        return {"status": "INVALID_FALLBACK_BASELINE", "fallback_reason": "UNKNOWN_ID", "selected_ids": []}, fallback
    return {"status": "VALID_SELECTION", "selected_ids": list(selected)}, list(selected)


def _invoke_external_gate(run_path: Path, manifest_path: Path, authorization_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(GATE_PATH), "--run-json", str(run_path), "--manifest-json", str(manifest_path), "--authorization-json", str(authorization_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=60, check=False,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = {"verdict": "NOT_SCOREABLE", "score_invocation_allowed": False, "reasons": ["EXTERNAL_GATE_OUTPUT_INVALID", completed.stderr[-1000:]]}
    return result


def _build_run_document(
    manifest: dict[str, Any], started: str, deadline: str, finished: str,
    cases: list[dict[str, Any]], request_events: list[dict[str, Any]], sparql_events: list[dict[str, Any]],
    runner_state: str,
) -> dict[str, Any]:
    api = manifest["api_caps"]
    sparql = manifest["sparql_caps"]
    total_input = sum(int(event.get("input_tokens", 0)) for event in request_events)
    total_output = sum(int(event.get("output_tokens", 0)) for event in request_events)
    cost = sum(
        int(event.get("uncached_input_tokens", 0)) * float(api["rate_assumption_input_uncached_usd_per_million"]) / 1_000_000
        + int(event.get("output_tokens", 0)) * float(api["rate_assumption_output_usd_per_million"]) / 1_000_000
        for event in request_events
    )
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "model": api["model"],
        "sparql_endpoint": sparql["endpoint"],
        "runner_state": runner_state,
        "failure_treatment": manifest["failure_treatment"],
        "run_started_at_utc": started,
        "hard_deadline_utc": deadline,
        "run_finished_at_utc": finished,
        "cases": cases,
        "request_events": request_events,
        "sparql_events": sparql_events,
        "accounting": {
            "api_requests_started": len(request_events),
            "api_request_cap": api["api_request_cap"],
            "physical_sparql_attempts": len(sparql_events),
            "sparql_attempt_cap": sparql["physical_read_only_select_cap"],
            "input_tokens": total_input,
            "output_tokens": total_output,
            "rate_based_cost_usd": round(cost, 10),
            "reserved_api_exposure_usd": round(sum(float(event.get("reserved_cost_usd", 0.0)) for event in request_events), 10),
        },
    }


def execute_cohort(
    manifest: dict[str, Any], cohort: list[dict[str, Any]], gateway: Any, agent: Any,
    answer_once: Callable[..., tuple[dict[str, Any] | None, dict[str, Any] | None]],
    entity_factory: Callable[[str, str], Any], prompt_text: str, output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Run the paired state machine against supplied (possibly mocked) I/O."""
    run_cases: list[dict[str, Any]] = []
    snapshots_path = output_dir / "FROZEN_CASE_SNAPSHOTS_NON_GOLD.jsonl"
    paired_path = output_dir / "PAIRED_CASES_NON_GOLD.jsonl"
    trace_path = output_dir / "CASE_TRACES_NON_GOLD.jsonl"
    progress_path = output_dir / "CASE_STATE_EVENTS.jsonl"
    abort: dict[str, Any] | None = None
    try:
        preflight = gateway.select_once("SELECT * WHERE { ?s ?p ?o } LIMIT 1")
        write_json_fsync(output_dir / "PREFLIGHT.json", {
            "status": "HTTP_OK", "binding_count": len(preflight.get("bindings", [])),
            "raw_response_sha256": preflight.get("raw_response_sha256"),
        })
        for row in cohort:
            case_id = str(row["case_id"])
            if gateway.halted:
                raise RuntimeError("one-shot gateway halted; no retries or resume")
            gateway.set_case_context(case_id)
            gateway.request_role = "baseline"
            append_fsync(progress_path, {
                "event": "CASE_ATTEMPTED", "state": "ATTEMPTED",
                "case_id": case_id, "question_key": row["question_key"], "at_utc": utc_now(),
            })
            prompt_history: list[dict[str, Any]] = []
            sparql_history: list[dict[str, Any]] = []
            step_history: list[dict[str, Any]] = []
            entities = [entity_factory(str(eid), str(label)) for eid, label in zip(row["topic_entities"], row["topic_entity_names"])]
            api_before, sparql_before = len(gateway.request_events), len(gateway.sparql_events)
            try:
                baseline_result, failure = answer_once(
                    agent, question=row["question"], topic_entities=entities,
                    prompt_history=prompt_history, sparql_history=sparql_history, step_history=step_history,
                )
            except CaseLocalApiFailure as exc:
                baseline_result, failure = None, exc.failure_record()
            observation = _derive_observation(
                row=row, result=baseline_result, failure=failure,
                prompt_history=prompt_history, sparql_history=sparql_history,
                gateway=gateway, manifest=manifest, prompt_text=prompt_text,
            )
            # Baseline predictions and candidate provenance are durable before
            # any possible selector request.
            append_fsync(snapshots_path, {
                key: observation[key] for key in (
                    "case_id", "question_key", "baseline_status", "baseline_prediction_ids",
                    "baseline_answer_names", "terminal_source", "terminal_trace_id", "route",
                    "filter_event_count", "candidate_ids", "candidate_labels", "literal_candidates",
                    "candidate_query_attempt_id", "candidate_query_status", "candidate_query_sha256",
                    "candidate_snapshot_sha256", "executed_relation_path", "executed_relation_ids",
                    "selector_eligible", "selector_eligibility_reason",
                )
            })
            append_fsync(progress_path, {
                "event": "BASELINE_RECORDED",
                "state": observation["baseline_status"],
                "case_id": case_id, "question_key": row["question_key"], "at_utc": utc_now(),
            })
            intervention_ids = copy.deepcopy(observation["baseline_prediction_ids"])
            selector_result = None
            selector_not_called_reason = "TRIGGER_NOT_MET"
            if observation["selector_eligible"]:
                messages = [{
                    "role": "user",
                    "content": prompt_text.rstrip() + "\n\nINPUT:\n" + json.dumps(
                        observation["prompt_payload"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ),
                }]
                gateway.request_role = "selector"
                gateway.normalized_prompt_sha = manifest["intervention"]["prompt_sha256"]
                try:
                    response = gateway.call_model_once(
                        messages,
                        max_completion_tokens=manifest["intervention"]["max_completion_tokens"],
                        reasoning_effort=manifest["intervention"]["reasoning_effort"],
                    )
                except Exception as exc:
                    from scripts.diagnostics.iter_026_answer_filter.gateway_attempt4 import InputTokenLimitExceeded
                    if isinstance(exc, InputTokenLimitExceeded):
                        selector_result = {
                            "status": "INVALID_FALLBACK_BASELINE",
                            "fallback_reason": "LOCAL_PROMPT_INPUT_CAP",
                            "selected_ids": [],
                        }
                        selector_not_called_reason = "LOCAL_PROMPT_INPUT_CAP"
                        selected = copy.deepcopy(observation["baseline_prediction_ids"])
                    elif isinstance(exc, CaseLocalApiFailure):
                        selector_result = {
                            "status": "INVALID_FALLBACK_BASELINE",
                            "fallback_reason": exc.failure_kind,
                            "selected_ids": [],
                        }
                        selected = copy.deepcopy(observation["baseline_prediction_ids"])
                    else:
                        raise
                else:
                    checked, selected = _parse_selector(
                        response.get("content"), observation["candidate_ids"], observation["baseline_prediction_ids"]
                    )
                    selector_result = checked
                    if response.get("finish_reason") != "stop" or not isinstance(response.get("content"), str) or not response["content"].strip():
                        selector_result = {
                            "status": "INVALID_FALLBACK_BASELINE",
                            "fallback_reason": "LENGTH_OR_EMPTY_RESPONSE",
                            "selected_ids": [],
                        }
                        selected = copy.deepcopy(observation["baseline_prediction_ids"])
                if selector_result and selector_result.get("status") == "INVALID_FALLBACK_BASELINE":
                    selected = copy.deepcopy(observation["baseline_prediction_ids"])
                if selector_result is None:
                    selector_result = {
                        "status": "INVALID_FALLBACK_BASELINE",
                        "fallback_reason": "UNCLASSIFIED_SELECTOR_FAILURE",
                        "selected_ids": [],
                    }
                    selected = copy.deepcopy(observation["baseline_prediction_ids"])
                intervention_ids = selected
            case_events = gateway.request_events[api_before:]
            selector_requests = [event for event in case_events if event["request_role"] == "selector"]
            case = {
                "case_id": case_id,
                "question_key": row["question_key"],
                "question": row["question"],
                "case_state": "PAIRED_VALID",
                "route": observation["route"],
                "terminal_source": observation["terminal_source"],
                "filter_event_count": observation["filter_event_count"],
                "baseline_status": observation["baseline_status"],
                "baseline_failure_kind": observation["baseline_failure_kind"],
                "baseline_response": observation["baseline_response"],
                "baseline_prediction_ids": observation["baseline_prediction_ids"],
                "baseline_answer_names": observation["baseline_answer_names"],
                "intervention_prediction_ids": intervention_ids,
                "answer_kind": observation["answer_kind"],
                "executed_relation_path": observation["executed_relation_path"],
                "executed_relation_ids": observation["executed_relation_ids"],
                "candidate_ids": observation["candidate_ids"],
                "candidate_labels": observation["candidate_labels"],
                "literal_candidates": observation["literal_candidates"],
                "candidate_query_attempt_id": observation["candidate_query_attempt_id"],
                "candidate_query_status": observation["candidate_query_status"],
                "candidate_query_sha256": observation["candidate_query_sha256"],
                "candidate_snapshot_sha256": observation["candidate_snapshot_sha256"],
                "selector_prompt_input_token_estimate": observation["selector_prompt_input_token_estimate"],
                "selector_eligible": observation["selector_eligible"],
                "selector_eligibility_reason": observation["selector_eligibility_reason"],
                "selector_request_count": len(selector_requests),
                "selector_applied": bool(selector_requests),
                "selector_not_called_reason": None if selector_requests else selector_not_called_reason,
                "selector_result": selector_result,
                "baseline_api_request_count": sum(event["request_role"] == "baseline" for event in case_events),
                "case_sparql_attempt_count": len(gateway.sparql_events[sparql_before:]),
            }
            if observation["selector_eligible"] and len(selector_requests) != 1 and selector_not_called_reason != "LOCAL_PROMPT_INPUT_CAP":
                raise RuntimeError("eligible case did not receive exactly one selector request")
            if not observation["selector_eligible"] and intervention_ids != observation["baseline_prediction_ids"]:
                raise RuntimeError("ineligible case changed the frozen baseline prediction")
            append_fsync(trace_path, {
                **case, "prompt_history": prompt_history,
                "sparql_history": sparql_history, "step_history": step_history,
            })
            append_fsync(paired_path, case)
            run_cases.append(case)
            append_fsync(progress_path, {
                "event": "PAIRED_VALID", "state": "PAIRED_VALID",
                "case_id": case_id, "question_key": row["question_key"], "at_utc": utc_now(),
            })
            gateway.set_case_context(None)
            gateway.request_role = "baseline"
            if len(run_cases) == 100 and not any(item["selector_eligible"] for item in run_cases):
                abort = {"status": "MECHANISM_FAILURE_NO_ELIGIBILITY", "completed_cases": 100}
                break
            if gateway.halted:
                raise RuntimeError("one-shot gateway halted; no retries or resume")
    except BaseException as exc:
        abort = {"status": "ABORTED_INCOMPLETE_NO_RESUME", "error_type": type(exc).__name__, "detail": str(exc)[:1000]}
    return run_cases, abort


def run_live(manifest_path: Path, authorization_path: Path, expected_manifest_sha: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != expected_manifest_sha or manifest_digest(manifest) != expected_manifest_sha:
        raise RuntimeError("exact frozen manifest hash verification failed; no transport")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    validate_authorization(manifest, expected_manifest_sha, authorization)
    cohort = read_jsonl(COHORT_PATH)
    verify_inputs(manifest, cohort)

    # Credential availability is checked after exact authorization and hashes,
    # but before consuming the one-shot marker or creating a run directory.
    from scripts.run_cor_baseline_experiment import load_safe_project_env
    load_safe_project_env()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY unavailable; no transport started")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")
    if base_url and base_url != "https://api.openai.com/v1":
        raise RuntimeError("frozen transport requires the official OpenAI API base URL")

    # Import and validate local runtime/transport dependencies before consuming
    # the one-shot marker. These imports are inert and make no network calls.
    from scripts.diagnostics.iter_026_answer_filter.gateway_attempt4 import Attempt4Gateway, InputTokenLimitExceeded
    from scripts.diagnostics.iter_026_answer_filter.strict_runtime_attempt4 import answer_case_once, make_attempt4_tgar_agent
    from scripts.diagnostics.iter_026_answer_filter.transport import FreebaseSelectTransport, OpenAIChatCompletionsTransport
    from chain_of_relations.schema import Entity

    def answer_case_once_attempt3(agent: Any, **kwargs: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        try:
            return answer_case_once(agent, **kwargs)
        except CaseLocalApiFailure as exc:
            return None, exc.failure_record()

    output_dir = (RESULTS / manifest.get("run_directory", RUN_RELATIVE)).resolve()
    expected_output = (RESULTS / RUN_RELATIVE).resolve()
    if output_dir != expected_output:
        raise RuntimeError("run directory differs from the fixed Iteration 29 location")
    marker_path = RESULTS / "ITER29_ATTEMPT3_AUTHORIZATION_CONSUMPTION.json"
    if output_dir.exists() or marker_path.exists():
        raise RuntimeError("one-shot manifest already consumed or run files exist; no resume")
    RESULTS.mkdir(parents=True, exist_ok=True)
    marker = {
        "iteration": 29,
        "attempt": 3,
        "manifest_sha256": expected_manifest_sha,
        "status": "CONSUMED_BEFORE_PREFLIGHT_ONE_SHOT_NO_RESUME",
        "consumed_at_utc": utc_now(),
        "run_directory": RUN_RELATIVE,
    }
    write_json_fsync(marker_path, marker)
    output_dir.mkdir(parents=True, exist_ok=False)
    auth_copy = output_dir / "AUTHORIZATION_RECORD.json"
    write_json_fsync(auth_copy, authorization)

    class Iter29Gateway(Attempt4Gateway):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.request_role = "baseline"
            self.request_events: list[dict[str, Any]] = []
            self.sparql_events: list[dict[str, Any]] = []
            self.successful_query_by_sha: dict[str, dict[str, Any]] = {}
            self.case_select_counts: dict[str, int] = {}
            self.normalized_prompt_sha = ""
            self.endpoint = manifest["sparql_caps"]["endpoint"]
            self.request_events_path = output_dir / "REQUEST_EVENTS.jsonl"
            self.sparql_events_path = output_dir / "SPARQL_EVENTS.jsonl"
            self.per_case_select_cap = manifest["sparql_caps"]["per_case_select_cap"]

        def call_model_once(self, messages: list[dict[str, str]], *, max_completion_tokens: int, reasoning_effort: str = "low") -> dict[str, Any]:
            from scripts.diagnostics.iter_026_answer_filter.gateway import (
                AttemptUncertain, BudgetExhausted, DuplicatePrompt,
                _append_fsync, canonical_json as _canonical,
                sha256_bytes as _sha,
            )
            from scripts.diagnostics.transport_safety_repair.hard_deadline import call_with_hard_deadline

            self._check_live()
            if getattr(self.model_transport, "max_retries", None) != 0:
                raise RuntimeError("model transport retry policy changed")
            if not isinstance(messages, list) or not messages or any(
                not isinstance(message, dict) or set(message) != {"role", "content"}
                or not isinstance(message["role"], str) or not isinstance(message["content"], str)
                for message in messages
            ):
                raise ValueError("messages must be a non-empty list of role/content text objects")
            prompt_hash = _sha(_canonical(messages))
            if prompt_hash in self.prompt_hashes:
                self.halted = True
                raise DuplicatePrompt("same model messages already attempted; duplicate transport blocked")
            if not 1 <= int(max_completion_tokens) <= self.output_tokens_per_request:
                raise BudgetExhausted("completion cap exceeds the authorized per-request limit")
            if self.api_starts >= self.api_request_cap:
                raise BudgetExhausted("API request cap reached")
            reservation = (
                self.input_tokens_per_request * self.input_price
                + self.output_tokens_per_request * self.output_price
            ) / 1_000_000
            if self.api_reserved_usd + reservation > self.api_cost_cap_usd + 1e-12:
                raise BudgetExhausted("worst-case API reservation would exceed the total cost cap")
            estimate = int(self.token_counter(messages))
            if estimate > self.input_tokens_per_request:
                raise InputTokenLimitExceeded(
                    estimate=estimate, limit=self.input_tokens_per_request,
                    prompt_sha256=prompt_hash, case_id=self.active_case_id,
                )

            request_body = {
                "model": manifest["api_caps"]["model"], "messages": messages,
                "max_completion_tokens": int(max_completion_tokens),
                "reasoning_effort": reasoning_effort, "stream": False,
            }
            request_hash = _sha(_canonical(request_body))
            start = utc_now()
            _append_fsync(self.request_journal, {
                "event": "REQUEST_STARTED", "manifest_sha256": self.manifest_sha256,
                "case_id": self.active_case_id, "request_sha256": request_hash,
                "prompt_sha256": prompt_hash, "estimated_input_tokens": estimate,
                "reserved_cost_usd": reservation, "request_body": request_body,
                "recorded_at_utc": start,
            })
            attempt_id = self._record_attempt_start(
                kind="api", digest=request_hash,
                details={"reserved_cost_usd": reservation,
                         "reserved_input_tokens": self.input_tokens_per_request,
                         "reserved_completion_tokens": self.output_tokens_per_request},
            )
            self.prompt_hashes.add(prompt_hash)
            self.api_starts += 1
            self.api_reserved_usd += reservation
            try:
                response = call_with_hard_deadline(
                    lambda: self.model_transport(request_body), self.timeout_seconds,
                )
            except Exception as exc:
                end = utc_now()
                failure_kind = "API_TRANSPORT_ERROR"
                outcome = "TRANSPORT_ERROR"
                role_failure = "BASELINE_METHOD_FAILURE" if self.request_role == "baseline" else "SELECTOR_BASELINE_FALLBACK"
                event = {
                    "request_sha256": request_hash,
                    "prompt_sha256": self.normalized_prompt_sha if self.request_role == "selector" else prompt_hash,
                    "payload_prompt_sha256": prompt_hash,
                    "case_id": str(self.active_case_id or ""), "request_role": self.request_role,
                    "model": manifest["api_caps"]["model"], "attempt_number": 1, "retry_of": None,
                    "started_at_utc": start, "finished_at_utc": end,
                    "input_tokens": 0, "uncached_input_tokens": 0, "output_tokens": 0,
                    "usage_known": False, "reserved_cost_usd": reservation,
                    "max_completion_tokens": int(max_completion_tokens),
                    "finish_reason": "transport_error", "raw_finish_reason": None,
                    "content_nonempty": False, "transport_outcome": outcome,
                    "failure_policy_applied": True, "failure_outcome": role_failure,
                    "error_type": type(exc).__name__,
                }
                self.request_events.append(event)
                append_fsync(self.request_events_path, event)
                _append_fsync(self.request_journal, {
                    "event": "REQUEST_FINISHED", "manifest_sha256": self.manifest_sha256,
                    "case_id": self.active_case_id, "request_sha256": request_hash,
                    "status": outcome, "error_type": type(exc).__name__,
                    "reserved_cost_usd": reservation, "recorded_at_utc": end,
                })
                _append_fsync(self.attempt_journal, {
                    "event": "ATTEMPT_FINISHED", "attempt_id": attempt_id,
                    "outcome": "TRANSPORT_ERROR_RESERVED_NO_RETRY", "error_type": type(exc).__name__,
                    "recorded_at_utc": end,
                })
                raise CaseLocalApiFailure(failure_kind, error_type=type(exc).__name__) from None

            end = utc_now()
            if not isinstance(response, dict) or response.get("http_status") != 200:
                status = response.get("http_status") if isinstance(response, dict) else None
                error_code = response.get("error_code") if isinstance(response, dict) else None
                response_hash = response.get("raw_response_sha256") if isinstance(response, dict) else None
                role_failure = "BASELINE_METHOD_FAILURE" if self.request_role == "baseline" else "SELECTOR_BASELINE_FALLBACK"
                event = {
                    "request_sha256": request_hash,
                    "prompt_sha256": self.normalized_prompt_sha if self.request_role == "selector" else prompt_hash,
                    "payload_prompt_sha256": prompt_hash,
                    "case_id": str(self.active_case_id or ""), "request_role": self.request_role,
                    "model": manifest["api_caps"]["model"], "attempt_number": 1, "retry_of": None,
                    "started_at_utc": start, "finished_at_utc": end,
                    "input_tokens": 0, "uncached_input_tokens": 0, "output_tokens": 0,
                    "usage_known": False, "reserved_cost_usd": reservation,
                    "max_completion_tokens": int(max_completion_tokens),
                    "finish_reason": "http_error", "raw_finish_reason": None,
                    "content_nonempty": False, "transport_outcome": "HTTP_ERROR",
                    "http_status": status, "error_code": error_code,
                    "raw_response_sha256": response_hash,
                    "failure_policy_applied": True, "failure_outcome": role_failure,
                    "error_type": "HTTPStatusError",
                }
                self.request_events.append(event)
                append_fsync(self.request_events_path, event)
                _append_fsync(self.request_journal, {
                    "event": "REQUEST_FINISHED", "manifest_sha256": self.manifest_sha256,
                    "case_id": self.active_case_id, "request_sha256": request_hash,
                    "status": "HTTP_ERROR", "http_status": status, "error_code": error_code,
                    "raw_response_sha256": response_hash, "reserved_cost_usd": reservation,
                    "recorded_at_utc": end,
                })
                _append_fsync(self.attempt_journal, {
                    "event": "ATTEMPT_FINISHED", "attempt_id": attempt_id,
                    "outcome": "HTTP_ERROR_RESERVED_NO_RETRY", "http_status": status,
                    "recorded_at_utc": end,
                })
                raise CaseLocalApiFailure(
                    "API_HTTP_ERROR", http_status=status if isinstance(status, int) else None,
                    error_code=error_code, error_type="HTTPStatusError",
                    raw_response_sha256=response_hash,
                ) from None

            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            if not all(key in usage for key in ("input_tokens", "output_tokens", "total_tokens")):
                self.halted = True
                raise AttemptUncertain("OpenAI usage metadata is missing; no retry is allowed")
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            if input_tokens > self.input_tokens_per_request or output_tokens > self.output_tokens_per_request:
                self.halted = True
                raise AttemptUncertain("reported usage exceeded the manifest request cap")
            actual_cost = (input_tokens * self.input_price + output_tokens * self.output_price) / 1_000_000
            self.api_input_tokens += input_tokens
            self.api_output_tokens += output_tokens
            self.api_actual_cost_usd += actual_cost
            content = response.get("content")
            finish = response.get("finish_reason")
            nonempty = isinstance(content, str) and bool(content.strip())
            normalized_finish = "empty" if not nonempty and finish != "length" else finish
            failure_policy = normalized_finish in {"length", "empty"}
            role_failure = ("BASELINE_METHOD_FAILURE" if self.request_role == "baseline" else "SELECTOR_BASELINE_FALLBACK") if failure_policy else None
            event = {
                "request_sha256": request_hash,
                "prompt_sha256": self.normalized_prompt_sha if self.request_role == "selector" else prompt_hash,
                "payload_prompt_sha256": prompt_hash,
                "case_id": str(self.active_case_id or ""), "request_role": self.request_role,
                "model": manifest["api_caps"]["model"], "attempt_number": 1, "retry_of": None,
                "started_at_utc": start, "finished_at_utc": end,
                "input_tokens": input_tokens, "uncached_input_tokens": input_tokens,
                "output_tokens": output_tokens, "usage_known": True,
                "reserved_cost_usd": reservation, "max_completion_tokens": int(max_completion_tokens),
                "finish_reason": normalized_finish, "raw_finish_reason": finish,
                "content_nonempty": nonempty, "transport_outcome": "HTTP_OK",
                "http_status": 200, "raw_response_sha256": response.get("raw_response_sha256"),
                "failure_policy_applied": failure_policy, "failure_outcome": role_failure,
                "error_type": None,
            }
            self.request_events.append(event)
            append_fsync(self.request_events_path, event)
            _append_fsync(self.request_journal, {
                "event": "REQUEST_FINISHED", "manifest_sha256": self.manifest_sha256,
                "case_id": self.active_case_id, "request_sha256": request_hash,
                "status": "HTTP_OK", "finish_reason": finish,
                "raw_response_sha256": response.get("raw_response_sha256"),
                "usage": usage, "actual_cost_usd": actual_cost, "recorded_at_utc": end,
            })
            _append_fsync(self.attempt_journal, {
                "event": "ATTEMPT_FINISHED", "attempt_id": attempt_id,
                "outcome": "HTTP_OK", "actual_cost_usd": actual_cost,
                "recorded_at_utc": end,
            })
            return response

        def select_once(self, query: str) -> dict[str, Any]:
            case_id = str(self.active_case_id or "")
            if case_id and self.case_select_counts.get(case_id, 0) >= self.per_case_select_cap:
                raise RuntimeError("per-case physical SPARQL SELECT cap reached; run aborts without resume")
            query_sha = sha256_bytes(query.encode("utf-8"))
            start = utc_now()
            before = self.sparql_starts
            try:
                response = super().select_once(query)
                status = "ok"
                outcome = "HTTP_OK"
                error = None
            except BaseException as exc:
                response = {}
                status = "error"
                outcome = "HTTP_ERROR"
                error = type(exc).__name__
                raise
            finally:
                if self.sparql_starts > before:
                    if case_id:
                        self.case_select_counts[case_id] = self.case_select_counts.get(case_id, 0) + 1
                    event = {
                        "attempt_id": str(uuid.uuid4()),
                        "attempt_kind": "case" if case_id else "endpoint_preflight",
                        "case_id": case_id,
                        "endpoint": self.endpoint,
                        "query_type": "SELECT",
                        "read_only": True,
                        "query_sha256": query_sha,
                        "attempt_number": 1,
                        "retry_of": None,
                        "started_at_utc": start,
                        "status": status,
                        "transport_outcome": outcome,
                        "error_type": error,
                    }
                    self.sparql_events.append(event)
                    append_fsync(self.sparql_events_path, event)
                    if status == "ok":
                        self.successful_query_by_sha[query_sha] = event
            return response

    started = utc_now()
    wall = float(manifest["time_caps"]["wall_clock_cap_hours"]) * 3600
    deadline = (datetime.fromisoformat(started) + timedelta(seconds=wall)).isoformat()
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
    gateway = Iter29Gateway(
        output_dir,
        manifest_sha256=expected_manifest_sha,
        model_transport=OpenAIChatCompletionsTransport(api_key, timeout_seconds=manifest["api_caps"]["per_request_timeout_seconds"]),
        sparql_transport=FreebaseSelectTransport(endpoint=manifest["sparql_caps"]["endpoint"], timeout_seconds=manifest["api_caps"]["per_request_timeout_seconds"]),
        api_request_cap=manifest["api_caps"]["api_request_cap"],
        api_cost_cap_usd=manifest["api_caps"]["cost_cap_usd"],
        input_tokens_per_request=manifest["api_caps"]["per_request_input_token_cap"],
        output_tokens_per_request=manifest["api_caps"]["per_request_completion_token_cap"],
        input_usd_per_million=manifest["api_caps"]["rate_assumption_input_uncached_usd_per_million"],
        output_usd_per_million=manifest["api_caps"]["rate_assumption_output_usd_per_million"],
        sparql_attempt_cap=manifest["sparql_caps"]["physical_read_only_select_cap"],
        timeout_seconds=manifest["api_caps"]["per_request_timeout_seconds"],
        wall_seconds=wall,
    )
    metadata_path = output_dir / "RUN_METADATA.json"
    write_json_fsync(metadata_path, {
        "iteration": 29, "manifest_sha256": expected_manifest_sha,
        "cohort_sha256": sha256_file(COHORT_PATH), "model": manifest["api_caps"]["model"],
        "status": "RUNNING", "gold_loaded_by_live_process": False,
        "evidence_class": manifest["evidence_class"], "run_started_at_utc": started,
    })
    try:
        agent = make_attempt4_tgar_agent(gateway)
        run_cases, abort = execute_cohort(
            manifest, cohort, gateway, agent, answer_case_once_attempt3,
            lambda entity_id, label: Entity(entity_id, label), prompt_text, output_dir,
        )
    except BaseException as exc:
        run_cases = []
        abort = {"status": "ABORTED_INCOMPLETE_NO_RESUME", "error_type": type(exc).__name__, "detail": str(exc)[:1000]}

    finished = utc_now()
    expected_keys = manifest.get("question_keys", [])
    paired_keys = [case.get("question_key") for case in run_cases]
    all_pairs_valid = (
        len(run_cases) == len(cohort)
        and paired_keys == expected_keys
        and all(case.get("case_state") == "PAIRED_VALID" for case in run_cases)
    )
    runner_state = "COMPLETE" if all_pairs_valid and abort is None else (abort or {}).get("status", "ABORTED_INCOMPLETE_NO_RESUME")
    run_doc = _build_run_document(
        manifest, started, deadline, finished, run_cases,
        gateway.request_events, gateway.sparql_events, runner_state,
    )
    run_path = output_dir / "VALIDATION_INPUT_NON_GOLD.json"
    write_json_fsync(run_path, run_doc)
    auth_run_path = output_dir / "AUTHORIZATION_RECORD.json"
    gate_result = _invoke_external_gate(run_path, manifest_path, auth_run_path)
    write_json_fsync(output_dir / "SCORING_GATE_VERDICT.json", gate_result)
    write_json_replace(metadata_path, {
        "iteration": 29, "manifest_sha256": expected_manifest_sha,
        "cohort_sha256": sha256_file(COHORT_PATH), "model": manifest["api_caps"]["model"],
        "status": runner_state, "cases_expected": len(cohort), "cases_paired": len(run_cases),
        "eligible_cases": sum(bool(case.get("selector_eligible")) for case in run_cases),
        "api_requests_started": len(gateway.request_events),
        "physical_sparql_attempts": len(gateway.sparql_events),
        "gold_loaded_by_live_process": False, "gold_scoring_performed": False,
        "abort": abort, "external_gate_verdict": gate_result.get("verdict"),
        "run_started_at_utc": started, "run_finished_at_utc": finished,
    })
    return {"runner_state": runner_state, "paired_cases": len(run_cases), "abort": abort, "gate": gate_result, "output_dir": str(output_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify frozen hashes only; make no API/SPARQL calls")
    mode.add_argument("--run", action="store_true", help="one-shot run; requires exact-hash authorization JSON")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--manifest-sha256", default="")
    parser.add_argument("--authorization-json", type=Path)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        cohort = read_jsonl(COHORT_PATH)
        if args.check:
            verify_inputs(manifest, cohort)
            print(json.dumps({"status": "FROZEN_INPUTS_VALID_NO_TRANSPORT", "manifest_sha256": manifest["manifest_sha256"], "cases": len(cohort), "api_requests": 0, "sparql_attempts": 0}, indent=2))
            return 0
        if args.authorization_json is None or not args.manifest_sha256:
            parser.error("--run requires --manifest-sha256 and --authorization-json")
        result = run_live(args.manifest, args.authorization_json, args.manifest_sha256)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if result["gate"].get("verdict") == "SCOREABLE" else 2
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "REFUSED_OR_FAILED_CLOSED", "error_type": type(exc).__name__, "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
