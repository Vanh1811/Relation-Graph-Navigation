"""Pure offline scorer for explicitly typed answer envelopes.

This module does not read gold or data files. Gold atoms enter only as an
explicit scoring argument after a prediction snapshot has been frozen.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .answer_contract import CONTRACT_VERSION, XSD_INTEGER


def prediction_atoms(answer: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(answer, Mapping) or answer.get("contract_version") != CONTRACT_VERSION:
        return set()
    kind = answer.get("answer_kind")
    if kind == "entity":
        entities = answer.get("entities")
        if not isinstance(entities, list):
            return set()
        ids = {str(item.get("id", "")) for item in entities if isinstance(item, Mapping)}
        return ids if ids and all(ids) else set()
    if kind == "aggregate":
        value = answer.get("lexical_value")
        if answer.get("status") != "accepted" or answer.get("datatype") != XSD_INTEGER:
            return set()
        if not isinstance(value, str) or not (value == "0" or (value.isascii() and value.isdigit() and not value.startswith("0"))):
            return set()
        return {value}
    return set()


def score_typed_answer(answer: Mapping[str, Any] | None, gold_atoms: Sequence[str]) -> dict[str, Any]:
    """Score one envelope under a caller-declared entity-or-scalar contract."""
    gold = {str(value) for value in gold_atoms if str(value)}
    predicted = prediction_atoms(answer)
    overlap = len(gold & predicted)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "hits_at_1": int(bool(overlap)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "answer_kind": answer.get("answer_kind", "unknown") if isinstance(answer, Mapping) else "unknown",
        "prediction_atoms": sorted(predicted),
    }
