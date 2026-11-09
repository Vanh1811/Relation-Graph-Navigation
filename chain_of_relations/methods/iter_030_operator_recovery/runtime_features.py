"""Gold-free inference feature contract for the Iteration 30 prototype.

This module deliberately accepts only fields available at inference time. It
does not load datasets, traces, labels, logical forms, expected answers, or
question identifiers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeRelationStep:
    relation_id: str
    forward_from_current: bool


@dataclass(frozen=True)
class GoldFreeRuntimeFeatures:
    question: str
    topic_entity_id: str
    relation_chain: tuple[RuntimeRelationStep, ...]
    requested_answer_type: str = ""
    target_type_id: str = ""
    grounded_query_status: str = "unknown"
    baseline_answer: str = ""

    def to_runtime_dict(self) -> dict[str, Any]:
        """Serialize only the allow-listed inference-time feature fields."""
        return asdict(self)
