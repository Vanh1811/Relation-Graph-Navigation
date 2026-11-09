"""Gold-free typed answer envelope, isolated from production CoR/TGAR.

This module only accepts scalar model answers when the archived/runtime-shaped
callsite contains a direct COUNT intent, a numerical router hint, an explicit
terminal count rationale, a canonical single integer, and matching CoR result
and request provenance. Rejections preserve the exact baseline object.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from chain_of_relations.methods.iter_030_operator_recovery.relation_aware_count import (
    classify_count_intent,
)


CONTRACT_VERSION = "iter32-typed-answer-v1"
XSD_INTEGER = "http://www.w3.org/2001/XMLSchema#integer"
_CANONICAL_NONNEGATIVE = re.compile(r"^(?:0|[1-9][0-9]*)$")
_NUMERIC_HINTS = {"number", "count", "quantity", "numeric"}
_AGGREGATE_RATIONALE = re.compile(
    r"\b(?:count|counting|aggregate|aggregated|number of|total number)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class RequestEvidence:
    """Evidence joined to a terminal prompt by its exact request prompt hash."""

    prompt_sha256: str
    finish_reason: str
    content_nonempty: bool
    http_status: int | None
    output_tokens: int | None
    request_sha256: str = ""
    raw_response_sha256: str = ""


@dataclass(frozen=True)
class SerializationOutcome:
    status: str
    output: Any
    typed_answer: Mapping[str, Any] | None
    reason: str
    baseline_preserved: bool


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _terminal_event(result: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    history = result.get("prompt_history")
    if not isinstance(history, list):
        return None, "missing_prompt_history"
    events = [item for item in history if isinstance(item, dict)]
    events = [item for item in events if item.get("type") in {"reasoning", "filter_answer"}]
    if not events:
        return None, "missing_terminal_event"
    with_ids = [item for item in events if isinstance(item.get("trace_id"), int)]
    event = max(with_ids, key=lambda item: item["trace_id"]) if with_ids else events[-1]
    return event, ""


def _router_event(result: Mapping[str, Any]) -> dict[str, Any] | None:
    history = result.get("prompt_history")
    if not isinstance(history, list):
        return None
    for item in history:
        if isinstance(item, dict) and item.get("type") == "adaptive_router_v1_1_3":
            return item
    return None


def _candidate_evidence(result: Mapping[str, Any], terminal: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    history = result.get("sparql_history")
    if not isinstance(history, list):
        return [], []
    terminal_id = terminal.get("trace_id")
    eligible = [
        item for item in history
        if isinstance(item, dict)
        and item.get("search_type") == "entity_search"
        and (
            not isinstance(terminal_id, int)
            or not isinstance(item.get("trace_id"), int)
            or item.get("trace_id") < terminal_id
        )
    ]
    if not eligible:
        return [], []
    search = max(
        eligible,
        key=lambda item: item.get("trace_id", -1) if isinstance(item.get("trace_id"), int) else -1,
    )
    names = search.get("candidate_entity_names", [])
    literals = search.get("literal_candidates", [])
    return (
        [str(value).strip() for value in names] if isinstance(names, list) else [],
        [str(value).strip() for value in literals] if isinstance(literals, list) else [],
    )


def _typed_entity(result: Mapping[str, Any]) -> dict[str, Any] | None:
    rows = result.get("results")
    if not isinstance(rows, list) or not rows:
        return None
    entities: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        entity_id = str(row.get("id", "")).strip()
        name = str(row.get("name", "")).strip()
        if not entity_id or not name:
            return None
        entities.append({"id": entity_id, "name": name})
    return {
        "contract_version": CONTRACT_VERSION,
        "answer_kind": "entity",
        "entities": entities,
    }


def _request_matches(terminal: Mapping[str, Any], evidence: RequestEvidence | None) -> bool:
    if evidence is None:
        return False
    prompt = terminal.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        return False
    return (
        evidence.prompt_sha256 == _sha256(prompt)
        and evidence.finish_reason == "stop"
        and evidence.content_nonempty is True
        and evidence.http_status == 200
        and isinstance(evidence.output_tokens, int)
        and evidence.output_tokens == terminal.get("output_tokens")
    )


def serialize_callsite_result(
    result: Any,
    *,
    enabled: bool = False,
    request_evidence: RequestEvidence | None = None,
) -> SerializationOutcome:
    """Return a typed sidecar only when explicitly enabled and fully supported.

    Disabled mode returns the original object itself. Any rejected scalar also
    returns that same object, with an explicit unknown diagnostic sidecar.
    """
    if not enabled:
        return SerializationOutcome("DISABLED", result, None, "feature_disabled", True)
    if not isinstance(result, Mapping):
        return SerializationOutcome("UNKNOWN", result, None, "result_not_mapping", True)

    entity = _typed_entity(result)
    if entity is not None:
        return SerializationOutcome("ENTITY", entity, entity, "entity_ids_preserved", False)

    terminal, event_error = _terminal_event(result)
    if terminal is None:
        unknown = {"contract_version": CONTRACT_VERSION, "answer_kind": "unknown", "reason": event_error}
        return SerializationOutcome("UNKNOWN", result, unknown, event_error, True)

    def reject(reason: str, *, invalid: bool = False) -> SerializationOutcome:
        sidecar = {
            "contract_version": CONTRACT_VERSION,
            "answer_kind": "invalid" if invalid else "unknown",
            "reason": reason,
        }
        return SerializationOutcome("INVALID" if invalid else "UNKNOWN", result, sidecar, reason, True)

    if str(result.get("action", "")).casefold() != "stop":
        return reject("result_action_not_stop")
    if terminal.get("type") != "reasoning":
        return reject("terminal_is_not_direct_reasoning")
    if str(terminal.get("parsed_result", "")).casefold() != "stop":
        return reject("terminal_decision_not_stop")
    if not _request_matches(terminal, request_evidence):
        return reject("request_finish_or_prompt_provenance_missing")

    response = terminal.get("response")
    if not isinstance(response, str) or not response.strip():
        return reject("empty_terminal_response")
    try:
        payload = json.loads(response)
    except (TypeError, ValueError):
        return reject("terminal_response_not_json", invalid=True)
    if not isinstance(payload, dict) or str(payload.get("decision", "")).casefold() != "stop":
        return reject("terminal_json_decision_not_stop", invalid=True)

    answers = payload.get("answer")
    if not isinstance(answers, list) or len(answers) != 1 or not isinstance(answers[0], str):
        return reject("answer_not_single_string", invalid=True)
    value = answers[0]
    if not _CANONICAL_NONNEGATIVE.fullmatch(value):
        return reject("answer_not_canonical_nonnegative_integer", invalid=True)

    rows = result.get("results")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        return reject("returned_result_not_single_item")
    returned_name = str(rows[0].get("name", "")).strip()
    returned_id = str(rows[0].get("id", "")).strip()
    if returned_name != value:
        return reject("terminal_answer_differs_from_returned_name")
    if returned_id:
        return reject("numeric_result_has_entity_id")

    candidate_labels, literal_candidates = _candidate_evidence(result, terminal)
    if not candidate_labels:
        return reject("candidate_label_evidence_missing")
    if value in candidate_labels or value in literal_candidates:
        return reject("scalar_ambiguous_with_candidate_or_literal")

    question = str(result.get("question", ""))
    count_intent, intent_reason = classify_count_intent(question)
    if not count_intent:
        return reject(f"count_intent_not_supported:{intent_reason}")

    router = _router_event(result)
    parsed_router = router.get("parsed_result") if isinstance(router, dict) else None
    if not isinstance(parsed_router, dict):
        return reject("router_evidence_missing")
    answer_type = str(parsed_router.get("requested_final_answer_type", "")).strip().casefold()
    numeric_type = answer_type in _NUMERIC_HINTS or answer_type.startswith("count of ") or answer_type.startswith("number of ")
    if not numeric_type:
        return reject("router_did_not_request_numeric_answer_type")
    rationale = str(payload.get("rationale", ""))
    if not _AGGREGATE_RATIONALE.search(rationale):
        return reject("terminal_rationale_does_not_support_aggregation")

    provenance = {
        "source_kind": "archived_or_runtime_cor_terminal_answer",
        "raw_answer_field_path": "prompt_history[terminal].response.answer[0]",
        "cor_return_field_path": "results[0].name",
        "router_field_path": "prompt_history[adaptive_router_v1_1_3].parsed_result.requested_final_answer_type",
        "question_field_path": "question",
        "intent_rule": intent_reason,
        "terminal_trace_id": terminal.get("trace_id"),
        "terminal_prompt_sha256": request_evidence.prompt_sha256,
        "terminal_response_text_sha256": _sha256(response),
        "request_sha256": request_evidence.request_sha256,
        "raw_response_sha256": request_evidence.raw_response_sha256,
        "request_finish_reason": request_evidence.finish_reason,
        "request_output_tokens": request_evidence.output_tokens,
        "result_entity_id": returned_id,
    }
    typed = {
        "contract_version": CONTRACT_VERSION,
        "answer_kind": "aggregate",
        "status": "accepted",
        "datatype": XSD_INTEGER,
        "lexical_value": value,
        "source_text": value,
        "provenance": provenance,
    }
    return SerializationOutcome("AGGREGATE", typed, typed, "runtime_count_intent_and_terminal_scalar_supported", False)


def serialize_executor_count(adapter_outcome: Any, baseline: Any) -> SerializationOutcome:
    """Validate an Iteration 31 consumer result into the shared scalar schema."""
    if getattr(adapter_outcome, "status", None) != "COUNT_EXECUTED":
        return SerializationOutcome(
            "UNKNOWN", baseline,
            {"contract_version": CONTRACT_VERSION, "answer_kind": "unknown", "reason": "count_adapter_not_executed"},
            "count_adapter_not_executed", True,
        )
    output = getattr(adapter_outcome, "output", None)
    if not isinstance(output, Mapping) or output.get("answer_kind") != "aggregate":
        return SerializationOutcome("INVALID", baseline, None, "count_adapter_output_shape_invalid", True)
    value = output.get("scalar_answer")
    result_rows = output.get("results")
    provenance = output.get("count_provenance")
    if (
        not isinstance(value, str)
        or not _CANONICAL_NONNEGATIVE.fullmatch(value)
        or not isinstance(result_rows, list)
        or len(result_rows) != 1
        or not isinstance(result_rows[0], Mapping)
        or result_rows[0].get("value") != value
        or result_rows[0].get("datatype") != XSD_INTEGER
        or not isinstance(provenance, Mapping)
        or not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("query_sha256", "")))
        or not isinstance(provenance.get("result"), Mapping)
        or not re.fullmatch(r"[0-9a-f]{64}", str(provenance["result"].get("source_sha256", "")))
    ):
        return SerializationOutcome("INVALID", baseline, None, "count_adapter_scalar_or_provenance_invalid", True)
    typed = {
        "contract_version": CONTRACT_VERSION,
        "answer_kind": "aggregate",
        "status": "accepted",
        "datatype": XSD_INTEGER,
        "lexical_value": value,
        "source_text": value,
        "provenance": {
            "source_kind": "iter31_count_adapter",
            "field_path": "output.scalar_answer",
            "query_sha256": provenance["query_sha256"],
            "result_source_sha256": provenance["result"]["source_sha256"],
            "operator": provenance.get("operator"),
            "target_variable": provenance.get("target_variable"),
            "target_type_id": provenance.get("target_type_id"),
        },
    }
    return SerializationOutcome("AGGREGATE", typed, typed, "typed_count_adapter_output_valid", False)
