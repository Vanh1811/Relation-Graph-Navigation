"""Gold-free trigger, payload and response contract for SCV-EmptyOnly.

This module has no transport, database, credential or gold-label access. The
only intervention trigger is a successfully completed, entity-only TGAR case
whose baseline answer is empty and whose final filter candidate pool has
verifiable graph-query provenance. Because it acts only on an empty baseline,
it cannot lower baseline Hits@1 or F1 under the repository's set-overlap
metrics; any proposed output still has to pass the frozen candidate-ID check.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

try:
    from scripts.diagnostics.iter_026_answer_filter.selector import (
        candidate_snapshot_sha256,
        validate_selector_response,
    )
except ImportError:  # pragma: no cover - direct script test invocation
    from iter_026_answer_filter.selector import (  # type: ignore
        candidate_snapshot_sha256,
        validate_selector_response,
    )


SCHEMA_VERSION = "iter27.scv_empty_only.v1"
ID_RE = re.compile(r"(?:m|g)\.[A-Za-z0-9_]+\Z")
INSTRUCTIONS = (
    "Choose only answer entities from the supplied frozen graph candidate pool. "
    "Use the question, the unchanged TGAR filter reasoning context, and the "
    "recorded graph-query provenance. Do not invent IDs or use outside facts. "
    "Return all distinct answers needed by the question. If evidence is "
    "insufficient, abstain. Return exactly one JSON object with schema_version "
    f"{SCHEMA_VERSION!r}, decision 'select' or 'abstain', and candidate_ids "
    "as a list of verbatim Freebase IDs. An abstention must use an empty list. "
    "Return no additional keys, prose, or markdown."
)


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def eligible_for_scv_empty_only(
    *,
    baseline_status: str,
    baseline_entity_only: bool,
    baseline_prediction_ids: Sequence[str],
    router_execution_mode: str,
    final_filter_event_count: int,
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[bool, str]:
    """Return a deterministic gold-free intervention decision and reason."""
    if baseline_status != "BASELINE_COMPLETED":
        return False, "BASELINE_NOT_COMPLETED"
    if router_execution_mode.casefold() != "cor":
        return False, "NON_COR_ROUTER_MODE"
    if final_filter_event_count != 1:
        return False, "FILTER_BOUNDARY_COUNT_NOT_ONE"
    if not baseline_entity_only:
        return False, "BASELINE_NOT_ENTITY_ONLY"
    if baseline_prediction_ids:
        return False, "BASELINE_NONEMPTY_SAFE_FALLBACK"
    if not candidates:
        return False, "CANDIDATE_POOL_EMPTY"
    try:
        candidate_snapshot_sha256(candidates)
    except (TypeError, ValueError):
        return False, "CANDIDATE_SNAPSHOT_INVALID"
    for position, candidate in enumerate(candidates):
        source = candidate.get("source")
        if not isinstance(source, Mapping):
            return False, "CANDIDATE_PROVENANCE_MISSING"
        trace_id = source.get("entity_search_trace_id")
        query_sha = source.get("entity_search_query_sha256")
        source_position = source.get("position")
        if not isinstance(trace_id, int) or trace_id < 0:
            return False, "CANDIDATE_PROVENANCE_MISSING"
        if not isinstance(query_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", query_sha):
            return False, "CANDIDATE_PROVENANCE_MISSING"
        if source_position != position:
            return False, "CANDIDATE_PROVENANCE_ORDER_MISMATCH"
    return True, "SCV_EMPTY_BASELINE_WITH_FROZEN_PROVENANCE"


def build_scv_payload(
    *,
    question: str,
    frozen_filter_context: str,
    router_execution_mode: str,
    filter_trace_id: int,
    selected_relation_evidence: Sequence[str],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Serialize only inference-time question, captured evidence and candidates."""
    if not question.strip() or not frozen_filter_context.strip():
        raise ValueError("question and frozen filter context must be present")
    if router_execution_mode.casefold() != "cor":
        raise ValueError("SCV-EmptyOnly only accepts a completed CoR route")
    if not candidates:
        raise ValueError("candidate pool must be nonempty")
    snapshot_sha = candidate_snapshot_sha256(candidates)
    first_source = candidates[0]["source"]
    query_hash = first_source["entity_search_query_sha256"]
    trace_id = first_source["entity_search_trace_id"]
    for candidate in candidates:
        if candidate["source"].get("entity_search_query_sha256") != query_hash:
            raise ValueError("candidates must share one frozen source query")
        if candidate["source"].get("entity_search_trace_id") != trace_id:
            raise ValueError("candidates must share one frozen source trace")
    return {
        "schema_version": SCHEMA_VERSION,
        "question": question,
        "baseline_state": "completed_entity_answer_empty",
        "frozen_filter_context": frozen_filter_context,
        "graph_provenance": {
            "router_execution_mode": router_execution_mode,
            "filter_trace_id": int(filter_trace_id),
            "entity_search_trace_id": int(trace_id),
            "entity_search_query_sha256": query_hash,
            "candidate_snapshot_sha256": snapshot_sha,
            "selected_relation_evidence": list(selected_relation_evidence),
        },
        "candidate_entities": [
            {"id": item["id"], "label": item["label"], "position": index}
            for index, item in enumerate(candidates)
        ],
        "instructions": INSTRUCTIONS,
    }


def validate_scv_response(
    raw_response: str | None,
    *,
    candidates: Sequence[Mapping[str, Any]],
    expected_candidate_snapshot_sha256: str,
    finish_reason: str | None = "stop",
) -> dict[str, Any]:
    """Strictly validate SCV output; invalid/empty output leaves baseline empty."""
    if finish_reason != "stop":
        return {
            "status": "NON_TERMINAL_OR_TRUNCATED_RESPONSE", "valid": False,
            "selected_ids": [], "prediction_source": "UNCHANGED_EMPTY_BASELINE",
            "error": f"finish_reason must be 'stop', got {finish_reason!r}",
        }
    try:
        payload = json.loads(raw_response or "")
    except (json.JSONDecodeError, TypeError):
        payload = None
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "decision", "candidate_ids"}:
        return {
            "status": "INVALID_SCHEMA", "valid": False, "selected_ids": [],
            "prediction_source": "UNCHANGED_EMPTY_BASELINE", "error": "invalid exact JSON schema",
        }
    if payload.get("schema_version") != SCHEMA_VERSION:
        return {
            "status": "INVALID_SCHEMA_VERSION", "valid": False, "selected_ids": [],
            "prediction_source": "UNCHANGED_EMPTY_BASELINE", "error": "schema version mismatch",
        }
    # Reuse the already-tested ID membership, duplicate, ambiguity, and
    # snapshot-integrity contract, then enforce this iteration's namespace.
    old = validate_selector_response(
        json.dumps({
            "schema_version": "iter26.answer_selector.v1",
            "decision": payload.get("decision"),
            "candidate_ids": payload.get("candidate_ids"),
        }, ensure_ascii=False),
        candidates=candidates,
        expected_candidate_snapshot_sha256=expected_candidate_snapshot_sha256,
        baseline_prediction=[],
    )
    if old.valid and old.status == "VALID_SELECTION":
        return {
            "status": "VALID_SELECTION", "valid": True,
            "selected_ids": list(old.selected_ids),
            "prediction_source": "SCV_EMPTY_BASELINE_REPAIR", "error": None,
        }
    if old.valid and old.status == "VALID_ABSTAIN_BASELINE_FALLBACK":
        return {
            "status": "VALID_ABSTENTION", "valid": True, "selected_ids": [],
            "prediction_source": "UNCHANGED_EMPTY_BASELINE", "error": None,
        }
    return {
        "status": old.status, "valid": False, "selected_ids": [],
        "prediction_source": "UNCHANGED_EMPTY_BASELINE", "error": old.error,
    }


def metric_pair(gold_ids: Sequence[str], prediction_ids: Sequence[str]) -> tuple[int, float]:
    """Hits@1 and set F1 matching the experiment's entity-set scorer."""
    gold, prediction = set(gold_ids), set(prediction_ids)
    if not prediction or not gold:
        return 0, 0.0
    overlap = len(gold & prediction)
    precision = overlap / len(prediction)
    recall = overlap / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return int(bool(overlap)), f1


def empty_baseline_non_regression_invariant(
    gold_ids: Sequence[str], proposed_ids: Sequence[str]
) -> bool:
    """True exactly when intervention metrics are no worse than empty baseline."""
    base = metric_pair(gold_ids, [])
    proposed = metric_pair(gold_ids, proposed_ids)
    return proposed[0] >= base[0] and proposed[1] >= base[1]


__all__ = [
    "SCHEMA_VERSION", "build_scv_payload", "eligible_for_scv_empty_only",
    "empty_baseline_non_regression_invariant", "metric_pair", "validate_scv_response",
]
