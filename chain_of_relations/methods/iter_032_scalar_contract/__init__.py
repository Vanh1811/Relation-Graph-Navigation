"""Isolated, opt-in typed answer serialization for Iteration 32."""

from .answer_contract import (
    CONTRACT_VERSION,
    XSD_INTEGER,
    RequestEvidence,
    SerializationOutcome,
    serialize_callsite_result,
    serialize_executor_count,
)
from .scorer_gate import score_typed_answer

__all__ = [
    "CONTRACT_VERSION",
    "XSD_INTEGER",
    "RequestEvidence",
    "SerializationOutcome",
    "score_typed_answer",
    "serialize_callsite_result",
    "serialize_executor_count",
]
