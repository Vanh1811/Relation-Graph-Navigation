"""Small type-aware set scorer for the Iteration 33 audit.

This diagnostic helper is intentionally isolated from production inference.
It keeps entity identifiers and literal values in separate answer namespaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, FrozenSet, Tuple


Atom = Tuple[str, str]


@dataclass(frozen=True)
class SetScore:
    hit: int
    precision: float
    recall: float
    f1: float


def atom(kind: str, value: object) -> Atom:
    """Return an exact, type-tagged answer atom; no label normalization."""
    if kind not in {"Entity", "Value"}:
        raise ValueError(f"unsupported answer kind: {kind!r}")
    text = str(value)
    if not text:
        raise ValueError("answer atom must be nonempty")
    return kind, text


def answer_set(items: Iterable[Atom]) -> FrozenSet[Atom]:
    result = frozenset(items)
    if any(kind not in {"Entity", "Value"} or not value for kind, value in result):
        raise ValueError("malformed typed answer atom")
    return result


def score_sets(predicted: Iterable[Atom], gold: Iterable[Atom]) -> SetScore:
    """Score exact typed denotations with set semantics.

    Empty predictions or empty gold sets score zero here. WebQSP's official
    empty-answer convention is handled separately by its official evaluator.
    """
    pred = answer_set(predicted)
    truth = answer_set(gold)
    overlap = pred & truth
    if not pred or not truth:
        return SetScore(int(bool(overlap)), 0.0, 0.0, 0.0)
    precision = len(overlap) / len(pred)
    recall = len(overlap) / len(truth)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return SetScore(int(bool(overlap)), precision, recall, f1)
