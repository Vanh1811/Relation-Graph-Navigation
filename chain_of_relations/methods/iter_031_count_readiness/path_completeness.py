"""Exact path, join, target-variable and binding checks for COUNT plans."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

from .evidence import EvidenceRef
from .type_resolution import TargetTypeResolution, _valid_schema_id, _valid_variable


_ENTITY_ID = re.compile(r"^(?:m|g)\.[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class PathStep:
    relation_id: str
    direction: str  # out | in
    input_variable: str
    output_variable: str


@dataclass(frozen=True)
class PathEvidence:
    target_variable: str
    query_target_variable: str
    expected_steps: tuple[PathStep, ...] | None
    executed_steps: tuple[PathStep, ...]
    query_status: str
    plan_finalized: bool | None
    plan_shape: str = "linear"
    required_external_binding_variables: tuple[str, ...] = ()
    active_binding_tuples: tuple[Mapping[str, str], ...] | None = None
    provenance: tuple[EvidenceRef, ...] = ()
    operator: str = "COUNT_DISTINCT_ENTITY"
    required_constraints_complete: bool | None = None
    all_pairwise_node_inequalities_represented: bool | None = None
    type_filter_variable: str = ""


@dataclass(frozen=True)
class PathCompletenessResult:
    status: str
    ready: bool
    reason: str
    target_variable: str
    path_length: int
    provenance: tuple[EvidenceRef, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "ready": self.ready,
            "reason": self.reason,
            "target_variable": self.target_variable,
            "path_length": self.path_length,
            "provenance": [item.as_dict() for item in self.provenance],
        }


def _result(status: str, reason: str, evidence: PathEvidence) -> PathCompletenessResult:
    return PathCompletenessResult(status, status == "SUPPORTED_EXACT_PATH", reason, evidence.target_variable, len(evidence.executed_steps), evidence.provenance)


def _bindings_cover(evidence: PathEvidence) -> bool:
    required = set(evidence.required_external_binding_variables)
    if not required:
        return True
    if evidence.active_binding_tuples is None or not evidence.active_binding_tuples:
        return False
    for row in evidence.active_binding_tuples:
        if not required.issubset(row):
            return False
        if any(not _ENTITY_ID.fullmatch(str(row[key] or "")) for key in required):
            return False
    return True


def check_path_completeness(
    evidence: PathEvidence,
    type_resolution: TargetTypeResolution,
) -> PathCompletenessResult:
    if evidence.operator != "COUNT_DISTINCT_ENTITY":
        return _result("UNSUPPORTED_OPERATOR_OR_ORDERING", "Only the explicitly supported COUNT_DISTINCT_ENTITY operator is compiled.", evidence)
    if evidence.plan_shape != "linear":
        return _result("UNSUPPORTED_OPERATOR_OR_ORDERING", "Branching, union and event-multiplicity plans are outside the linear path compiler.", evidence)
    if evidence.query_status not in {"ok", "success"}:
        return _result("QUERY_FAILED", "The observed grounded path query did not succeed.", evidence)
    if not evidence.provenance or any(not item.valid for item in evidence.provenance):
        return _result("UNKNOWN_MISSING_PROVENANCE", "The expected path or execution source lacks valid field-level provenance.", evidence)
    if evidence.expected_steps is None or evidence.plan_finalized is None:
        return _result("UNKNOWN_MISSING_PROVENANCE", "No finalized expected path plan was emitted; a selected CoR path alone cannot prove completeness.", evidence)
    if not evidence.plan_finalized:
        return _result("UNKNOWN_MISSING_PROVENANCE", "The runtime planner did not finalize a complete path.", evidence)
    expected = evidence.expected_steps
    executed = evidence.executed_steps
    if not expected:
        return _result("MISSING_HOP", "The finalized count plan contains no relation hop.", evidence)
    if len(executed) < len(expected):
        return _result("MISSING_HOP", "The executed path is shorter than the finalized expected path.", evidence)
    if len(executed) != len(expected):
        return _result("WRONG_RELATION_OR_DIRECTION", "The executed path has extra hops relative to the finalized plan.", evidence)

    for steps, label in ((expected, "expected"), (executed, "executed")):
        for index, step in enumerate(steps):
            if not _valid_schema_id(step.relation_id) or step.direction not in {"out", "in"}:
                return _result("UNSUPPORTED_OPERATOR_OR_ORDERING", f"The {label} path contains a malformed relation or direction.", evidence)
            if not _valid_variable(step.input_variable) or not _valid_variable(step.output_variable):
                return _result("DANGLING_OR_AMBIGUOUS_TARGET", f"The {label} path contains an invalid variable.", evidence)
            if index and steps[index - 1].output_variable != step.input_variable:
                return _result("MISSING_SHARED_VARIABLE_JOIN", f"Adjacent {label} hops do not reuse the same variable.", evidence)

    for wanted, actual in zip(expected, executed):
        if wanted.relation_id != actual.relation_id or wanted.direction != actual.direction:
            return _result("WRONG_RELATION_OR_DIRECTION", "The executed relation or traversal direction differs from the finalized plan.", evidence)
        if wanted.input_variable != actual.input_variable or wanted.output_variable != actual.output_variable:
            return _result("MISSING_SHARED_VARIABLE_JOIN", "The executed hop does not preserve the planned binding identity.", evidence)

    final_target = expected[-1].output_variable
    if not _valid_variable(evidence.target_variable) or evidence.target_variable != final_target or evidence.query_target_variable != final_target:
        return _result("DANGLING_OR_AMBIGUOUS_TARGET", "The aggregate, plan and query do not identify the same final target variable.", evidence)
    if type_resolution.status != "resolved" or type_resolution.target_variable != evidence.target_variable or type_resolution.resolved_type_id is None:
        return _result("AMBIGUOUS_TARGET_TYPE", "A canonical type is not resolved for this exact final target variable.", evidence)
    if not _bindings_cover(evidence):
        return _result("MISSING_ACTIVE_BINDING_TUPLES", "One or more required external join variables lack stable entity-ID bindings.", evidence)
    if evidence.required_constraints_complete is not True:
        status = "MISSING_REQUIRED_CONSTRAINT" if evidence.required_constraints_complete is False else "UNKNOWN_MISSING_PROVENANCE"
        return _result(status, "The plan does not prove that every required constraint is represented in the aggregate query.", evidence)
    if evidence.all_pairwise_node_inequalities_represented is not True:
        status = "NODE_DISTINCTNESS_UNPROVEN" if evidence.all_pairwise_node_inequalities_represented is False else "UNKNOWN_MISSING_PROVENANCE"
        return _result(status, "The supported GrailQA COUNT fragment requires the complete node-distinctness contract.", evidence)
    if evidence.type_filter_variable != evidence.target_variable:
        return _result("TARGET_TYPE_FILTER_MISALIGNED", "The canonical target type is not applied to the exact final target binding.", evidence)
    return _result("SUPPORTED_EXACT_PATH", "Finalized path, directions, join-variable identities, target type and required source bindings agree.", evidence)
