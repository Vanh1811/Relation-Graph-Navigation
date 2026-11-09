"""Conservative target type resolver; no question-to-ID guessing."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .evidence import EvidenceRef


_VAR = re.compile(r"^\?[A-Za-z_][A-Za-z0-9_]*$")
_SCHEMA_ID = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+$")


@dataclass(frozen=True)
class TargetTypeEvidence:
    type_id: str
    target_variable: str
    evidence_kind: str
    provenance: EvidenceRef | None
    detail: str = ""


@dataclass(frozen=True)
class TargetTypeResolution:
    status: str  # resolved | ambiguous | unsupported | not_observed
    target_variable: str
    type_ids: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...]
    evidence_grade: str
    justification: str

    @property
    def resolved_type_id(self) -> str | None:
        if self.status == "resolved" and len(self.type_ids) == 1:
            return self.type_ids[0]
        return None

    @property
    def confidence_label(self) -> str:
        """Categorical evidence strength; deliberately not a numeric probability."""
        if self.status == "resolved":
            return "DIRECT_CONSTRAINT"
        if self.evidence_grade in {"COARSE_RELATION_RANGE", "NATURAL_LANGUAGE_HINT_ONLY"}:
            return "CANDIDATE_ONLY"
        if self.status == "ambiguous":
            return "CONFLICTED_OR_UNRESOLVED"
        if self.status == "unsupported":
            return "INVALID_EVIDENCE"
        return "NO_EVIDENCE"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "target_variable": self.target_variable,
            "type_ids": list(self.type_ids),
            "evidence": [item.as_dict() for item in self.evidence],
            "evidence_grade": self.evidence_grade,
            "confidence_label": self.confidence_label,
            "confidence_is_calibrated_probability": False,
            "justification": self.justification,
        }


def _valid_variable(value: str) -> bool:
    return bool(_VAR.fullmatch(str(value or "")))


def _valid_schema_id(value: str) -> bool:
    return bool(_SCHEMA_ID.fullmatch(str(value or "")))


def resolve_target_type(
    target_variable: str,
    evidence: Iterable[TargetTypeEvidence] = (),
    *,
    answer_type_hint: str = "",
) -> TargetTypeResolution:
    """Resolve only an explicit, provenance-backed target constraint.

    Natural-language answer types, relation ranges and cached labels are useful
    clues but do not prove the requested class. Multiple nested type IDs are
    left ambiguous because the current count query contract accepts one class.
    """
    target = str(target_variable or "").strip()
    entries = tuple(evidence)
    if not _valid_variable(target):
        return TargetTypeResolution("unsupported", target, (), (), "NONE", "Target variable is missing or malformed.")

    relevant = tuple(item for item in entries if item.target_variable == target)
    explicit = tuple(item for item in relevant if item.evidence_kind == "explicit_target_type_constraint")
    other_variables = tuple(item for item in entries if item.evidence_kind == "explicit_target_type_constraint" and item.target_variable != target)
    if any(not _valid_schema_id(item.type_id) for item in relevant if item.type_id):
        return TargetTypeResolution("unsupported", target, (), (), "INVALID", "Runtime type evidence contains a malformed Freebase schema ID.")
    if any(item.provenance is None or not item.provenance.valid for item in explicit):
        return TargetTypeResolution("unsupported", target, (), (), "INVALID", "An explicit type constraint lacks a valid field-level source hash and pointer.")

    explicit_ids = tuple(sorted({item.type_id for item in explicit if item.type_id}))
    relation_ranges = tuple(sorted({item.type_id for item in relevant if item.evidence_kind == "relation_range" and item.type_id}))
    explicit_refs = tuple(sorted({item.provenance for item in explicit if item.provenance is not None}))
    range_refs = tuple(sorted({item.provenance for item in relevant if item.evidence_kind == "relation_range" and item.provenance is not None and item.provenance.valid}))

    if len(explicit_ids) > 1:
        return TargetTypeResolution(
            "ambiguous", target, explicit_ids, explicit_refs, "CONFLICTING_EXPLICIT_TYPES",
            "Multiple explicit target constraints disagree; the one-type count compiler cannot safely choose or intersect them.",
        )
    if len(explicit_ids) == 1:
        chosen = explicit_ids[0]
        if relation_ranges and chosen not in relation_ranges:
            return TargetTypeResolution(
                "ambiguous", target, tuple(sorted(set(explicit_ids + relation_ranges))),
                tuple(sorted(set(explicit_refs + range_refs))), "CONFLICTING_SCHEMA_EVIDENCE",
                "The explicit target constraint conflicts with the available relation range metadata.",
            )
        return TargetTypeResolution(
            "resolved", target, explicit_ids, explicit_refs, "DIRECT_RUNTIME_TARGET_CONSTRAINT",
            "One provenance-backed canonical type constraint is attached to the final target variable.",
        )
    if relation_ranges:
        return TargetTypeResolution(
            "ambiguous", target, relation_ranges, range_refs, "COARSE_RELATION_RANGE",
            "A relation range narrows possible values but does not prove the question's requested target class or subtype.",
        )
    if answer_type_hint.strip():
        return TargetTypeResolution(
            "ambiguous", target, (), (), "NATURAL_LANGUAGE_HINT_ONLY",
            "The router's answer-type phrase is not a canonical Freebase target type ID.",
        )
    if other_variables:
        return TargetTypeResolution(
            "not_observed", target, (), (), "INTERMEDIATE_TYPE_ONLY",
            "Type evidence is attached to an intermediate variable, not the requested final target.",
        )
    return TargetTypeResolution("not_observed", target, (), (), "NONE", "No target-type evidence was observed at runtime.")
