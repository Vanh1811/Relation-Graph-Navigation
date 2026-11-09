"""Offline contract for Iteration 26's context-preserving candidate-ID selector.

This module has no model, network, database, or credential code. A future
authorized caller may supply its transport; this module validates only its
local result and preserves the original baseline output on failure.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "iter26.answer_selector.v1"
INSTRUCTIONS = (
    "Choose the candidate entities that answer the question using the supplied "
    "question and the complete frozen TGAR answer-filter reasoning context. "
    "Return candidate IDs only. Preserve every distinct answer needed by the "
    "question. Do not invent, rewrite, or expand entities. If uncertain or no "
    "candidate answers the question, abstain. Return exactly one JSON object "
    "with schema_version \"iter26.answer_selector.v1\", decision \"select\" "
    "or \"abstain\", and candidate_ids as a list of verbatim IDs. An abstention "
    "must use an empty candidate_ids list. Include no other keys or text."
)


@dataclass(frozen=True)
class SelectionResult:
    status: str
    valid: bool
    selected_ids: tuple[str, ...]
    final_prediction: Any
    prediction_source: str
    response_sha256: str | None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["selected_ids"] = list(self.selected_ids)
        return result


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def candidate_snapshot_sha256(candidates: Sequence[Mapping[str, Any]]) -> str:
    """Hash ordered candidate IDs, exact display labels, and source provenance."""
    normalized = []
    if not candidates:
        raise ValueError("candidate snapshot must contain at least one eligible answer")
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate entry must be an object")
        entity_id = candidate.get("id")
        label = candidate.get("label")
        if not isinstance(entity_id, str) or re.fullmatch(r"(?:m|g)\.[A-Za-z0-9_]+", entity_id) is None:
            raise ValueError("candidate ID must be a well-formed Freebase entity ID")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("candidate label is absent or blank")
        source = candidate.get("source")
        normalized.append({"id": entity_id, "label": label, "source": source})
    ids = [candidate["id"] for candidate in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate snapshot contains duplicate IDs")
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def build_inference_payload(
    *, question: str, frozen_filter_context: str,
    frozen_filter_system_context: str | None,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build model-visible fields; gold, roles, and evaluation metadata are excluded."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty")
    if not isinstance(frozen_filter_context, str) or not frozen_filter_context.strip():
        raise ValueError("the original filter reasoning context must be non-empty")
    candidate_snapshot_sha256(candidates)
    visible_candidates = [{"id": c["id"], "label": c["label"]} for c in candidates]
    return {
        "question": question,
        "constraints": {
            "frozen_tgar_filter_system_context": frozen_filter_system_context or "",
            "frozen_tgar_filter_context": frozen_filter_context,
        },
        "candidate_entities": visible_candidates,
        "instructions": INSTRUCTIONS,
    }


def _label_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _fallback(
    *, status: str, error: str, response_sha256: str | None,
    baseline_prediction: Any,
) -> SelectionResult:
    return SelectionResult(
        status=status,
        valid=False,
        selected_ids=(),
        final_prediction=copy.deepcopy(baseline_prediction),
        prediction_source="UNCHANGED_BASELINE_FALLBACK",
        response_sha256=response_sha256,
        error=error,
    )


def validate_selector_response(
    raw_response: str | None,
    *, candidates: Sequence[Mapping[str, Any]],
    expected_candidate_snapshot_sha256: str,
    baseline_prediction: Any,
) -> SelectionResult:
    """Validate exact JSON and candidate membership; otherwise preserve baseline."""
    response_sha256 = (
        hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
        if isinstance(raw_response, str) else None
    )
    try:
        actual_snapshot_hash = candidate_snapshot_sha256(candidates)
    except (TypeError, ValueError) as exc:
        return _fallback(
            status="INVALID_FROZEN_CANDIDATE_SNAPSHOT", error=str(exc),
            response_sha256=response_sha256, baseline_prediction=baseline_prediction,
        )
    if actual_snapshot_hash != expected_candidate_snapshot_sha256:
        return _fallback(
            status="CANDIDATE_SNAPSHOT_MISMATCH",
            error="candidate ID/label/order/provenance differs from the frozen snapshot",
            response_sha256=response_sha256, baseline_prediction=baseline_prediction,
        )
    if raw_response is None or not raw_response.strip():
        return _fallback(
            status="EMPTY_RESPONSE", error="response content is empty",
            response_sha256=response_sha256, baseline_prediction=baseline_prediction,
        )
    try:
        payload = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        return _fallback(
            status="INVALID_JSON", error="response is not a single JSON object",
            response_sha256=response_sha256, baseline_prediction=baseline_prediction,
        )
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "decision", "candidate_ids"}:
        return _fallback(
            status="INVALID_SCHEMA", error="unexpected top-level keys or non-object output",
            response_sha256=response_sha256, baseline_prediction=baseline_prediction,
        )
    if payload["schema_version"] != SCHEMA_VERSION:
        return _fallback(
            status="INVALID_SCHEMA_VERSION", error="schema_version is not the frozen contract",
            response_sha256=response_sha256, baseline_prediction=baseline_prediction,
        )
    decision = payload["decision"]
    selected = payload["candidate_ids"]
    if not isinstance(decision, str) or decision not in {"select", "abstain"} or not isinstance(selected, list):
        return _fallback(
            status="INVALID_DECISION", error="decision or candidate_ids has the wrong type",
            response_sha256=response_sha256, baseline_prediction=baseline_prediction,
        )
    if any(not isinstance(value, str) or not value.strip() for value in selected):
        return _fallback(
            status="INVALID_ID_LIST", error="candidate_ids must contain non-empty strings",
            response_sha256=response_sha256, baseline_prediction=baseline_prediction,
        )
    if len(selected) != len(set(selected)):
        return _fallback(
            status="DUPLICATE_IDS", error="candidate_ids contains duplicates",
            response_sha256=response_sha256, baseline_prediction=baseline_prediction,
        )
    allowed_ids = {candidate["id"] for candidate in candidates}
    if not set(selected).issubset(allowed_ids):
        return _fallback(
            status="OUT_OF_CANDIDATE_ID", error="selector returned an ID outside the frozen pool",
            response_sha256=response_sha256, baseline_prediction=baseline_prediction,
        )
    if decision == "abstain":
        if selected:
            return _fallback(
                status="INVALID_ABSTENTION", error="abstention must have an empty candidate_ids list",
                response_sha256=response_sha256, baseline_prediction=baseline_prediction,
            )
        # A valid abstention retains the pre-existing baseline prediction.
        return SelectionResult(
            status="VALID_ABSTAIN_BASELINE_FALLBACK", valid=True,
            selected_ids=(), final_prediction=copy.deepcopy(baseline_prediction),
            prediction_source="UNCHANGED_BASELINE_FALLBACK",
            response_sha256=response_sha256,
        )
    if not selected:
        return _fallback(
            status="INVALID_EMPTY_SELECTION", error="select decision requires at least one ID",
            response_sha256=response_sha256, baseline_prediction=baseline_prediction,
        )

    labels_by_key: dict[str, set[str]] = {}
    label_for_id = {}
    for candidate in candidates:
        label_key = _label_key(candidate["label"])
        labels_by_key.setdefault(label_key, set()).add(candidate["id"])
        label_for_id[candidate["id"]] = candidate["label"]
    ambiguous_ids = [entity_id for entity_id in selected if len(labels_by_key[_label_key(label_for_id[entity_id])]) > 1]
    if ambiguous_ids:
        return _fallback(
            status="AMBIGUOUS_DISPLAY_LABEL", error="selected ID has a duplicated/aliased display label",
            response_sha256=response_sha256, baseline_prediction=baseline_prediction,
        )
    return SelectionResult(
        status="VALID_SELECTION", valid=True, selected_ids=tuple(selected),
        final_prediction=list(selected), prediction_source="INTERVENTION_SELECTOR",
        response_sha256=response_sha256,
    )


def score_id_sets(predicted: Sequence[str], gold: Sequence[str]) -> dict[str, Any]:
    """Deterministic set metrics for local fixtures and future gold-sidecar scoring."""
    pred = set(predicted)
    truth = set(gold)
    overlap = pred & truth
    precision = len(overlap) / len(pred) if pred else 0.0
    recall = len(overlap) / len(truth) if truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "hits_at_1": int(bool(overlap)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_count": len(pred),
        "gold_count": len(truth),
    }


def paired_outcome(baseline_ids: Sequence[str], intervention_ids: Sequence[str], gold_ids: Sequence[str]) -> dict[str, Any]:
    baseline = score_id_sets(baseline_ids, gold_ids)
    intervention = score_id_sets(intervention_ids, gold_ids)
    b, i = baseline["hits_at_1"], intervention["hits_at_1"]
    return {
        "baseline": baseline,
        "intervention": intervention,
        "paired_hits": "WIN" if i > b else "LOSS" if i < b else "TIE",
        "hit_delta": i - b,
        "f1_delta": intervention["f1"] - baseline["f1"],
    }
