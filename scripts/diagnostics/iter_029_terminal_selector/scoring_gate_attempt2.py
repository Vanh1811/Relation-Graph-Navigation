"""Independent fail-closed release gate for a paired KGQA run.

The validator reads no gold labels and imports no dataset scorer. It checks
the run against the hash-frozen manifest and a separate authorization record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


FAILURE_TREATMENT = "ITT_EMPTY_METHOD_FAILURE_AND_SELECTOR_BASELINE_FALLBACK_V1"
ENTITY_TERMINALS = {"direct-LLM-reasoning-STOP", "graph-filter"}
AGGREGATE_QUESTION = re.compile(
    r"\b(how many|how much|number of|total number|count of|average|sum of|maximum|minimum)\b",
    re.IGNORECASE,
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_digest(manifest: dict[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_json(unsigned)).hexdigest()


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def candidate_snapshot_hash(ids: list[str], labels: list[str]) -> str:
    if len(ids) != len(labels):
        return ""
    payload = [
        {"id": str(entity_id), "label": str(label), "position": index}
        for index, (entity_id, label) in enumerate(zip(ids, labels))
    ]
    return hashlib.sha256(canonical_json(payload)).hexdigest() if payload else ""


def _unique_nonempty(values: Any) -> bool:
    return (
        isinstance(values, list)
        and bool(values)
        and all(isinstance(value, str) and value.strip() for value in values)
        and len(values) == len(set(values))
    )


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _parse_executed_relation_path(path: Any) -> list[str]:
    """Independently parse CoR's triple-line path; omit candidate.target."""
    if not isinstance(path, str) or not path.strip():
        return []
    node = re.compile(r"^(?:e\d+|TopicEntity\(.*\))$", re.DOTALL)
    triple = re.compile(r"^\s*((?:e\d+|TopicEntity\(.*\)))\s+(\S+)\s+((?:e\d+|TopicEntity\(.*\)))\s+\.\s*$")
    relation_ids: list[str] = []
    seen: set[str] = set()
    for line in path.splitlines():
        match = triple.fullmatch(line)
        if not match:
            continue
        subject, predicate, object_ = (part.strip() for part in match.groups())
        if predicate == "candidate.target":
            continue
        if not node.fullmatch(subject) or not node.fullmatch(object_):
            continue
        if predicate not in seen:
            seen.add(predicate)
            relation_ids.append(predicate)
    return relation_ids


def _selector_eligibility(case: dict[str, Any], manifest: dict[str, Any], sparql_by_id: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    """Recompute the frozen gold-free trigger from recorded provenance."""
    reasons: list[str] = []
    intervention = manifest.get("intervention", {})
    source = case.get("terminal_source")
    route = str(case.get("route", "")).casefold()
    if case.get("baseline_status") != "BASELINE_COMPLETED":
        reasons.append("BASELINE_NOT_COMPLETED")
    if route != "cor":
        reasons.append("ROUTE_NOT_COR")
    if source not in ENTITY_TERMINALS:
        reasons.append("TERMINAL_SOURCE_UNSUPPORTED")
    filter_count = case.get("filter_event_count")
    if not _int(filter_count) or filter_count > 1 or (source == "graph-filter" and filter_count != 1):
        reasons.append("FILTER_BOUNDARY_INVALID")
    if case.get("answer_kind") != "entity":
        reasons.append("ANSWER_NOT_ENTITY")
    parsed_relation_ids = _parse_executed_relation_path(case.get("executed_relation_path"))
    if case.get("executed_relation_ids") != parsed_relation_ids:
        reasons.append("EXECUTED_RELATION_PATH_ID_MISMATCH")
    if not _unique_nonempty(parsed_relation_ids):
        reasons.append("EXECUTED_RELATION_PATH_MISSING")
    question = str(case.get("question", ""))
    if not question or AGGREGATE_QUESTION.search(question):
        reasons.append("AGGREGATE_OR_QUESTION_MISSING")

    query_id = str(case.get("candidate_query_attempt_id", ""))
    query_event = sparql_by_id.get(query_id)
    if (
        not query_event
        or query_event.get("query_type") != "SELECT"
        or query_event.get("read_only") is not True
        or query_event.get("status") != "ok"
        or case.get("candidate_query_status") != "ok"
        or not query_id
        or case.get("candidate_query_sha256") != query_event.get("query_sha256")
    ):
        reasons.append("CANDIDATE_QUERY_PROVENANCE_INVALID")

    ids = case.get("candidate_ids")
    labels = case.get("candidate_labels")
    max_candidates = intervention.get("max_candidate_count")
    if (
        not _unique_nonempty(ids)
        or not _unique_nonempty(labels)
        or len(ids) != len(labels)
        or not isinstance(max_candidates, int)
        or len(ids) < 2
        or len(ids) > max_candidates
    ):
        reasons.append("CANDIDATE_IDS_OR_LABELS_INVALID")
    else:
        expected_snapshot = candidate_snapshot_hash(ids, labels)
        if case.get("candidate_snapshot_sha256") != expected_snapshot:
            reasons.append("CANDIDATE_SNAPSHOT_HASH_MISMATCH")

    literals = case.get("literal_candidates", [])
    if not isinstance(literals, list) or literals:
        reasons.append("LITERAL_CANDIDATES_PRESENT_OR_INVALID")
    baseline_ids = case.get("baseline_prediction_ids")
    baseline_names = case.get("baseline_answer_names")
    if not _unique_nonempty(baseline_ids) or not isinstance(ids, list) or not set(baseline_ids).issubset(set(ids)):
        reasons.append("BASELINE_ENTITY_IDS_INVALID_OR_OUTSIDE_CANDIDATES")
    if not isinstance(baseline_names, list) or not baseline_names or not isinstance(labels, list):
        reasons.append("BASELINE_ANSWER_NAMES_MISSING")
    elif _unique_nonempty(labels):
        label_to_id = {label: entity_id for entity_id, label in zip(ids, labels)}
        mapped = [label_to_id.get(name) for name in baseline_names]
        if any(not isinstance(name, str) or name not in label_to_id for name in baseline_names) or mapped != baseline_ids:
            reasons.append("BASELINE_NAME_TO_ID_MAPPING_INVALID")

    prompt_tokens = case.get("selector_prompt_input_token_estimate")
    max_input = intervention.get("max_input_tokens")
    if not _int(prompt_tokens) or not isinstance(max_input, int) or prompt_tokens > max_input:
        reasons.append("SELECTOR_PROMPT_EXCEEDS_OR_LACKS_FROZEN_INPUT_CAP")
    return not reasons, reasons


def validate_for_scoring(
    run: dict[str, Any],
    manifest: dict[str, Any],
    authorization_record: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate a paired run against immutable protocol, authorization and ledgers."""
    reasons: list[str] = []
    expected_manifest_hash = str(manifest.get("manifest_sha256", ""))
    recomputed_manifest_hash = manifest_digest(manifest)
    if not expected_manifest_hash or expected_manifest_hash != recomputed_manifest_hash:
        reasons.append("MANIFEST_HASH_INVALID")
    if not isinstance(authorization_record, dict):
        reasons.append("EXACT_HASH_AUTHORIZATION_RECORD_MISSING")
    else:
        if authorization_record.get("authorized") is not True:
            reasons.append("AUTHORIZATION_NOT_EXPLICITLY_GRANTED")
        if authorization_record.get("manifest_sha256") != expected_manifest_hash:
            reasons.append("AUTHORIZATION_MANIFEST_HASH_MISMATCH")
        if not str(authorization_record.get("authorization_text", "")).strip():
            reasons.append("VERBATIM_AUTHORIZATION_TEXT_MISSING")
    if run.get("manifest_sha256") != expected_manifest_hash:
        reasons.append("RUN_MANIFEST_HASH_MISMATCH")

    expected = manifest.get("question_keys")
    if not isinstance(expected, list) or not expected or not all(isinstance(item, str) for item in expected):
        expected = []
        reasons.append("MANIFEST_QUESTION_KEYS_INVALID")
    cases = run.get("cases")
    if not isinstance(cases, list):
        cases = []
        reasons.append("CASES_NOT_A_LIST")
    observed = [str(case.get("question_key", "")) for case in cases if isinstance(case, dict)]
    if len(expected) != len(set(expected)):
        reasons.append("EXPECTED_MANIFEST_QUESTION_KEYS_NOT_UNIQUE")
    if observed != expected:
        reasons.append("CASE_QUESTION_KEYS_OR_ORDER_DO_NOT_MATCH_MANIFEST")
    if run.get("runner_state") != "COMPLETE":
        reasons.append("RUNNER_STATE_NOT_COMPLETE")
    if run.get("failure_treatment") != manifest.get("failure_treatment") or run.get("failure_treatment") != FAILURE_TREATMENT:
        reasons.append("FAILURE_TREATMENT_NOT_FROZEN")

    started = _parse_time(run.get("run_started_at_utc"))
    deadline = _parse_time(run.get("hard_deadline_utc"))
    finished = _parse_time(run.get("run_finished_at_utc"))
    wall_hours = manifest.get("time_caps", {}).get("wall_clock_cap_hours")
    if started is None or deadline is None or finished is None or not isinstance(wall_hours, (int, float)):
        reasons.append("RUN_DEADLINE_OR_TIME_CAP_INVALID")
    else:
        if deadline > started + timedelta(hours=float(wall_hours)):
            reasons.append("HARD_DEADLINE_EXCEEDS_MANIFEST_WALL_CAP")
        if finished > deadline:
            reasons.append("RUN_FINISHED_AFTER_HARD_DEADLINE")

    accounting = run.get("accounting") if isinstance(run.get("accounting"), dict) else {}
    api_caps = manifest.get("api_caps", {})
    sparql_caps = manifest.get("sparql_caps", {})
    expected_caps = {
        "api_request_cap": api_caps.get("api_request_cap"),
        "sparql_attempt_cap": sparql_caps.get("physical_read_only_select_cap"),
    }
    for count_key, cap_key in (("api_requests_started", "api_request_cap"), ("physical_sparql_attempts", "sparql_attempt_cap")):
        count = accounting.get(count_key)
        claimed_cap = accounting.get(cap_key)
        manifest_cap = expected_caps[cap_key]
        if not _int(count) or not isinstance(manifest_cap, int) or count > manifest_cap:
            reasons.append(f"{count_key.upper()}_MANIFEST_CAP_INVALID_OR_EXCEEDED")
        if claimed_cap != manifest_cap:
            reasons.append(f"{cap_key.upper()}_DIFFERS_FROM_MANIFEST")

    request_events = run.get("request_events")
    if not isinstance(request_events, list):
        request_events = []
        reasons.append("REQUEST_EVENT_LEDGER_MISSING")
    if _int(accounting.get("api_requests_started")) and len(request_events) != accounting["api_requests_started"]:
        reasons.append("API_REQUEST_LEDGER_COUNT_MISMATCH")
    seen_request_hashes: set[str] = set()
    requests_by_case: dict[str, list[dict[str, Any]]] = {}
    total_input = total_output = 0
    total_cost = 0.0
    input_cap = api_caps.get("per_request_input_token_cap")
    completion_cap = api_caps.get("per_request_completion_token_cap")
    timeout_cap = api_caps.get("per_request_timeout_seconds")
    input_rate = api_caps.get("rate_assumption_input_uncached_usd_per_million")
    output_rate = api_caps.get("rate_assumption_output_usd_per_million")
    if run.get("model") != api_caps.get("model"):
        reasons.append("RUN_MODEL_DIFFERS_FROM_MANIFEST")
    endpoint = sparql_caps.get("endpoint")
    if run.get("sparql_endpoint") != endpoint:
        reasons.append("RUN_SPARQL_ENDPOINT_DIFFERS_FROM_MANIFEST")
    for event in request_events:
        if not isinstance(event, dict):
            reasons.append("REQUEST_EVENT_INVALID")
            continue
        request_hash = str(event.get("request_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", request_hash):
            reasons.append("REQUEST_HASH_INVALID")
        elif request_hash in seen_request_hashes:
            reasons.append("REPEATED_REQUEST_HASH_POSSIBLE_RETRY")
        seen_request_hashes.add(request_hash)
        if event.get("attempt_number") != 1 or event.get("retry_of"):
            reasons.append("RETRY_OR_RESUME_DETECTED")
        case_id = str(event.get("case_id", ""))
        role = event.get("request_role")
        if not case_id or role not in {"baseline", "selector"}:
            reasons.append("REQUEST_CASE_OR_ROLE_INVALID")
        if event.get("model") != api_caps.get("model"):
            reasons.append("REQUEST_MODEL_DIFFERS_FROM_MANIFEST")
        requests_by_case.setdefault(case_id, []).append(event)
        if role == "selector" and event.get("prompt_sha256") != manifest.get("intervention", {}).get("prompt_sha256"):
            reasons.append("SELECTOR_PROMPT_HASH_DIFFERS_FROM_MANIFEST")
        req_start = _parse_time(event.get("started_at_utc"))
        req_end = _parse_time(event.get("finished_at_utc"))
        if req_start is None or req_end is None or req_end < req_start:
            reasons.append("REQUEST_TIME_INVALID")
        else:
            duration = (req_end - req_start).total_seconds()
            if started is not None and req_start < started:
                reasons.append("REQUEST_STARTED_BEFORE_RUN")
            if finished is not None and req_end > finished:
                reasons.append("REQUEST_FINISHED_AFTER_RUN")
            if deadline is not None and req_end > deadline:
                reasons.append("REQUEST_FINISHED_AFTER_HARD_DEADLINE")
            if not isinstance(timeout_cap, (int, float)) or duration > float(timeout_cap):
                reasons.append("REQUEST_EXCEEDS_MANIFEST_TIMEOUT")
        input_tokens = event.get("input_tokens")
        uncached_tokens = event.get("uncached_input_tokens")
        output_tokens = event.get("output_tokens")
        requested_completion = event.get("max_completion_tokens")
        if not all(_int(value) for value in (input_tokens, uncached_tokens, output_tokens, requested_completion)):
            reasons.append("REQUEST_TOKEN_USAGE_MISSING_OR_INVALID")
        else:
            if not isinstance(input_cap, int) or input_tokens > input_cap:
                reasons.append("REQUEST_INPUT_CAP_EXCEEDED")
            if not isinstance(completion_cap, int) or requested_completion > completion_cap or output_tokens > completion_cap:
                reasons.append("REQUEST_COMPLETION_CAP_EXCEEDED")
            if uncached_tokens > input_tokens:
                reasons.append("UNCACHED_INPUT_TOKENS_EXCEED_INPUT_TOKENS")
            total_input += input_tokens
            total_output += output_tokens
            if isinstance(input_rate, (int, float)) and isinstance(output_rate, (int, float)):
                total_cost += uncached_tokens * float(input_rate) / 1_000_000
                total_cost += output_tokens * float(output_rate) / 1_000_000
        finish = event.get("finish_reason")
        if event.get("transport_outcome") != "HTTP_OK":
            reasons.append("REQUEST_TRANSPORT_NOT_SUCCESSFUL")
        if finish != "stop" and not (
            event.get("failure_policy_applied") is True
            and finish in {"length", "empty"}
            and event.get("failure_outcome") in {"BASELINE_METHOD_FAILURE", "SELECTOR_BASELINE_FALLBACK"}
        ):
            reasons.append("UNHANDLED_NONSTOP_RESPONSE")

    if accounting.get("input_tokens") != total_input or accounting.get("output_tokens") != total_output:
        reasons.append("TOKEN_ACCOUNTING_DIFFERS_FROM_REQUEST_LEDGER")
    reported_cost = accounting.get("rate_based_cost_usd")
    cost_cap = api_caps.get("cost_cap_usd")
    if not isinstance(reported_cost, (int, float)) or abs(float(reported_cost) - total_cost) > 1e-8:
        reasons.append("COST_ACCOUNTING_DIFFERS_FROM_REQUEST_LEDGER")
    if not isinstance(cost_cap, (int, float)) or total_cost > float(cost_cap) + 1e-8:
        reasons.append("RATE_BASED_COST_CAP_EXCEEDED_OR_INVALID")

    sparql_events = run.get("sparql_events")
    if not isinstance(sparql_events, list):
        sparql_events = []
        reasons.append("SPARQL_EVENT_LEDGER_MISSING")
    if _int(accounting.get("physical_sparql_attempts")) and len(sparql_events) != accounting["physical_sparql_attempts"]:
        reasons.append("SPARQL_LEDGER_COUNT_MISMATCH")
    sparql_by_id: dict[str, dict[str, Any]] = {}
    sparql_counts_by_case: dict[str, int] = {}
    per_case_select_cap = sparql_caps.get("per_case_select_cap")
    for event in sparql_events:
        if not isinstance(event, dict):
            reasons.append("SPARQL_EVENT_INVALID")
            continue
        attempt_id = str(event.get("attempt_id", ""))
        if not attempt_id or attempt_id in sparql_by_id:
            reasons.append("SPARQL_ATTEMPT_ID_MISSING_OR_DUPLICATED")
        else:
            sparql_by_id[attempt_id] = event
        if event.get("query_type") != "SELECT" or event.get("read_only") is not True:
            reasons.append("NON_READ_ONLY_OR_NON_SELECT_ATTEMPT")
        if event.get("endpoint") != endpoint:
            reasons.append("SPARQL_ATTEMPT_ENDPOINT_DIFFERS_FROM_MANIFEST")
        attempt_kind = event.get("attempt_kind")
        case_id = str(event.get("case_id", ""))
        if attempt_kind == "case" and case_id:
            sparql_counts_by_case[case_id] = sparql_counts_by_case.get(case_id, 0) + 1
        elif attempt_kind != "endpoint_preflight" or case_id:
            reasons.append("SPARQL_ATTEMPT_CASE_OR_KIND_INVALID")
        if not isinstance(per_case_select_cap, int) or (
            case_id and sparql_counts_by_case.get(case_id, 0) > per_case_select_cap
        ):
            reasons.append("PER_CASE_SPARQL_CAP_INVALID_OR_EXCEEDED")
        if event.get("attempt_number") != 1 or event.get("retry_of"):
            reasons.append("SPARQL_RETRY_OR_RESUME_DETECTED")
        event_time = _parse_time(event.get("started_at_utc"))
        if event_time is None:
            reasons.append("SPARQL_ATTEMPT_TIME_INVALID")
        else:
            if started is not None and event_time < started:
                reasons.append("SPARQL_ATTEMPT_BEFORE_RUN")
            if finished is not None and event_time > finished:
                reasons.append("SPARQL_ATTEMPT_AFTER_RUN")
            if deadline is not None and event_time > deadline:
                reasons.append("SPARQL_ATTEMPT_AFTER_HARD_DEADLINE")

    cases_by_id = {str(case.get("case_id", "")): case for case in cases if isinstance(case, dict)}
    if len(cases_by_id) != len(cases):
        reasons.append("CASE_IDS_MISSING_OR_DUPLICATED")
    if any(case_id not in cases_by_id for case_id in requests_by_case):
        reasons.append("REQUEST_LEDGER_REFERENCES_UNKNOWN_CASE")
    minimum_eligible = manifest.get("analysis_plan", {}).get("minimum_eligible_cases_for_scoring")
    if not isinstance(minimum_eligible, int) or minimum_eligible < 1:
        reasons.append("MINIMUM_ELIGIBILITY_FOR_SCORING_NOT_IN_MANIFEST")
        minimum_eligible = 1
    eligible_count = 0
    for case in cases:
        if not isinstance(case, dict):
            reasons.append("CASE_ROW_INVALID")
            continue
        case_id = str(case.get("case_id", ""))
        baseline = case.get("baseline_prediction_ids")
        intervention = case.get("intervention_prediction_ids")
        if not isinstance(baseline, list) or not isinstance(intervention, list):
            reasons.append(f"{case_id}:PAIRED_PREDICTIONS_MISSING")
            continue
        if case.get("case_state") != "PAIRED_VALID":
            reasons.append(f"{case_id}:CASE_NOT_PAIRED_VALID")
        baseline_status = case.get("baseline_status")
        baseline_response = case.get("baseline_response") if isinstance(case.get("baseline_response"), dict) else {}
        case_request_events = requests_by_case.get(case_id, [])
        baseline_requests = [event for event in case_request_events if event.get("request_role") == "baseline"]
        if baseline_status == "BASELINE_COMPLETED":
            if baseline_response.get("finish_reason") != "stop" or baseline_response.get("content_nonempty") is not True:
                reasons.append(f"{case_id}:BASELINE_RESPONSE_NOT_VALID")
            if not baseline_requests:
                reasons.append(f"{case_id}:BASELINE_REQUEST_LEDGER_MISSING")
        elif baseline_status == "BASELINE_FAILURE_ITT":
            if case.get("baseline_failure_kind") not in {"length", "empty", "input_cap"}:
                reasons.append(f"{case_id}:BASELINE_FAILURE_KIND_UNPREREGISTERED")
            if baseline or intervention or case.get("selector_applied"):
                reasons.append(f"{case_id}:BASELINE_FAILURE_NOT_EMPTY_PAIRED_ITT")
            if baseline_response.get("finish_reason") not in {"length", "empty", "not_sent"}:
                reasons.append(f"{case_id}:BASELINE_FAILURE_RESPONSE_UNSUPPORTED")
            if case.get("baseline_failure_kind") in {"length", "empty"} and not baseline_requests:
                reasons.append(f"{case_id}:BASELINE_FAILURE_REQUEST_LEDGER_MISSING")
        else:
            reasons.append(f"{case_id}:BASELINE_STATUS_INVALID")

        recomputed_eligible, trigger_reasons = _selector_eligibility(case, manifest, sparql_by_id)
        if case.get("selector_eligible") is not recomputed_eligible:
            reasons.append(f"{case_id}:SELECTOR_ELIGIBILITY_DOES_NOT_MATCH_FROZEN_TRIGGER")
        if recomputed_eligible:
            eligible_count += 1
        selector_requests = [event for event in case_request_events if event.get("request_role") == "selector"]
        if case.get("selector_request_count") != len(selector_requests):
            reasons.append(f"{case_id}:SELECTOR_REQUEST_LEDGER_COUNT_MISMATCH")
        selector_applied = case.get("selector_applied") is True
        if selector_applied != (len(selector_requests) == 1):
            reasons.append(f"{case_id}:SELECTOR_APPLIED_FLAG_MISMATCH")
        if not selector_applied:
            if intervention != baseline:
                reasons.append(f"{case_id}:UNCALLED_INTERVENTION_CHANGED_BASELINE")
            if selector_requests:
                reasons.append(f"{case_id}:UNAUTHORIZED_SELECTOR_REQUEST_ON_UNCALLED_CASE")
            if recomputed_eligible and case.get("selector_not_called_reason") != "LOCAL_PROMPT_INPUT_CAP":
                reasons.append(f"{case_id}:ELIGIBLE_SELECTOR_NOT_CALLED_WITHOUT_FROZEN_REASON")
            continue

        if not recomputed_eligible:
            reasons.append(f"{case_id}:SELECTOR_APPLIED_OUTSIDE_FROZEN_TRIGGER:{','.join(trigger_reasons)}")
        selector_event = selector_requests[0]
        selector = case.get("selector_result") if isinstance(case.get("selector_result"), dict) else {}
        status = selector.get("status")
        if status == "VALID_SELECTION":
            selected = selector.get("selected_ids")
            ids = case.get("candidate_ids")
            if selector_event.get("finish_reason") != "stop" or selector_event.get("content_nonempty") is not True:
                reasons.append(f"{case_id}:SELECTOR_RESPONSE_NOT_VALIDLY_TERMINATED")
            if not _unique_nonempty(selected):
                reasons.append(f"{case_id}:SELECTOR_IDS_INVALID")
            elif isinstance(ids, list) and not set(selected).issubset(set(ids)):
                reasons.append(f"{case_id}:SELECTOR_ID_OUTSIDE_CANDIDATE_SNAPSHOT")
            elif intervention != selected:
                reasons.append(f"{case_id}:INTERVENTION_DIFFERS_FROM_VALIDATED_SELECTOR")
        elif status == "INVALID_FALLBACK_BASELINE":
            if not manifest.get("intervention", {}).get("fallback"):
                reasons.append(f"{case_id}:SELECTOR_FALLBACK_NOT_PREREGISTERED")
            if intervention != baseline or not selector.get("fallback_reason"):
                reasons.append(f"{case_id}:INVALID_SELECTOR_DID_NOT_EXACTLY_FALL_BACK")
        else:
            reasons.append(f"{case_id}:SELECTOR_RESULT_STATUS_INVALID")

    if eligible_count < minimum_eligible:
        reasons.append("MINIMUM_ELIGIBILITY_GATE_NOT_MET")
    if eligible_count == 0:
        reasons.append("ZERO_ELIGIBLE_CASES_IS_MECHANISM_FAILURE_NOT_ACCURACY_RESULT")

    verdict = "SCOREABLE" if not reasons else "NOT_SCOREABLE"
    return {
        "verdict": verdict,
        "gold_scoring_performed": False,
        "score_invocation_allowed": verdict == "SCOREABLE",
        "manifest_sha256": expected_manifest_hash,
        "expected_case_count": len(expected),
        "observed_case_count": len(cases),
        "eligible_case_count": eligible_count,
        "request_count": len(request_events),
        "sparql_attempt_count": len(sparql_events),
        "rate_based_cost_usd": round(total_cost, 10),
        "reasons": sorted(set(reasons)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--authorization-json", required=True, type=Path)
    args = parser.parse_args()
    try:
        run = json.loads(args.run_json.read_text(encoding="utf-8"))
        manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
        authorization = json.loads(args.authorization_json.read_text(encoding="utf-8"))
        result = validate_for_scoring(run, manifest, authorization)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        result = {
            "verdict": "NOT_SCOREABLE",
            "gold_scoring_performed": False,
            "score_invocation_allowed": False,
            "reasons": [f"INDEPENDENT_VALIDATOR_INPUT_ERROR:{type(exc).__name__}"],
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["verdict"] == "SCOREABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
