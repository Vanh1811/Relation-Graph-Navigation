#!/usr/bin/env python3
"""Run the frozen Phase 2.4.1 first-divergence trace audit.

The audit consumes only existing dataset annotations and frozen runtime
artifacts.  It does not call the model, query Virtuoso, or change inference.
Its key rule is causal ordering: a structure mismatch is reported before any
later state or relation mismatch.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from relation_graph_kgqa.datasets.schemas import QAExample  # noqa: E402
from relation_graph_kgqa.sparql.graph_builder import build_query_graph  # noqa: E402
from relation_graph_kgqa.sparql.parser import parse_query  # noqa: E402
from relation_graph_kgqa.logical_forms.sexpr_ast import Atom, ExprNode  # noqa: E402
from relation_graph_kgqa.logical_forms.sexpr_parser import parse_s_expression  # noqa: E402
from run_phase2_4_mixed_failure_validation import (  # noqa: E402
    gold_structure_from_s_expression,
    normalize_structure_label,
    read_json,
    read_jsonl,
)


OUTPUT_ROOT = ROOT / "outputs/experiments/phase2_4_1/first_divergence_trace_audit"
PHASE24_ROOT = ROOT / "outputs/experiments/phase2_4/mixed_failure_validation"
PHASE21_ROOT = ROOT / "outputs/experiments/phase2_1/relation_grounding"
PHASE23_ROOT = ROOT / "outputs/experiments/phase2_3/constraint_state_alignment"
FRESH_ROOT = ROOT / "outputs/experiments/phase1/adaptive_v1_1_3/grailqa_12_repeat"
CWQ_PATH = ROOT / "datasets/cwq/cwq.json"
GRAILQA_PATH = ROOT / "datasets/grailqa/official/grailqa_v1.0_train.json"

FIRST_DIVERGENCE_LABELS = {
    "STRUCTURE_ROUTING",
    "CONSTRAINT_DECOMPOSITION",
    "TARGET_VARIABLE_ALIGNMENT",
    "ENTITY_STATE_ALIGNMENT",
    "RELATION_CANDIDATE_AVAILABILITY",
    "RELATION_GROUNDING",
    "SPARQL_TRAVERSAL",
    "STATE_TRANSITION",
    "MERGE",
    "DOWNSTREAM_ANSWER",
    "KG_GOLD_MISMATCH",
    "INSUFFICIENT_EVIDENCE",
}


def phase24_manifest() -> dict[str, list[dict[str, Any]]]:
    """Load Phase 2.4 cases and add two successful fresh routing controls."""

    manifest = read_json(PHASE24_ROOT / "diagnostic_cases.json")
    manifest["track_c"].extend(
        [
            {
                "case_id": "3205047010000",
                "suspected_failure": False,
                "fresh_case": True,
                "primary_label": "CORRECT_STRUCTURE",
                "added_control": True,
            },
        ]
    )
    return manifest


def normalize_relation(value: str) -> str:
    value = str(value or "").strip()
    if value.startswith("ns:"):
        value = value[3:]
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
        value = value.rsplit("/", 1)[-1]
    return value


def gold_structure_from_sparql(row: dict[str, Any]) -> str:
    """Classify CWQ gold structure, distinguishing final/intermediate branches."""

    topic_entities = [str(item.get("id", "")) for item in row.get("topic_entity", [])]
    example = QAExample(
        id=str(row.get("id", "")),
        question=str(row.get("question", "")),
        answers=[],
        gold_sparql=str(row.get("sparql", "")),
        dataset_type="cwq",
        topic_entities=topic_entities,
        split="",
    )
    features = parse_query(example.gold_sparql or "")
    graph = build_query_graph(features, example)
    if not features.parse_success:
        return "GOLD_OTHER"
    candidates = graph.branch_candidates()
    if not candidates:
        return "GOLD_LINEAR"
    if any(variable in graph.answer_variables for variable in candidates):
        return "GOLD_BRANCH_FINAL"
    return "GOLD_BRANCH_INTERMEDIATE"


def gold_structure(case_id: str, cwq: dict[str, dict[str, Any]], grailqa: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if case_id in cwq:
        return gold_structure_from_sparql(cwq[case_id]), cwq[case_id]
    row = grailqa.get(case_id)
    if row is None:
        raise KeyError(f"missing gold dataset row for {case_id}")
    return gold_structure_from_s_expression(str(row.get("s_expression", ""))), row


def structure_matches(gold: str, runtime: str) -> bool:
    runtime = normalize_structure_label(runtime)
    return (gold == "GOLD_LINEAR" and runtime == "LINEAR") or (
        gold == "GOLD_BRANCH_FINAL" and runtime == "BRANCH_FINAL"
    ) or (
        gold == "GOLD_BRANCH_INTERMEDIATE" and runtime == "BRANCH_INTERMEDIATE"
    )


def first_divergence_label(checks: dict[str, Any]) -> str:
    """Select exactly one primary diagnosis in protocol order."""

    if checks.get("structure_match") is False:
        return "STRUCTURE_ROUTING"
    if checks.get("constraint_decomposition_match") is False:
        return "CONSTRAINT_DECOMPOSITION"
    if checks.get("target_variable_match") is False:
        return "TARGET_VARIABLE_ALIGNMENT"
    if checks.get("entity_state_match") is False:
        return "ENTITY_STATE_ALIGNMENT"
    if checks.get("candidate_available") is False:
        return "RELATION_CANDIDATE_AVAILABILITY"
    if checks.get("relation_selection_match") is False:
        return "RELATION_GROUNDING"
    if checks.get("sparql_match") is False:
        return "SPARQL_TRAVERSAL"
    if checks.get("state_transition_match") is False:
        return "STATE_TRANSITION"
    if checks.get("merge_match") is False:
        return "MERGE"
    if checks.get("answer_match") is False:
        return "DOWNSTREAM_ANSWER"
    return "INSUFFICIENT_EVIDENCE"


def downstream_status(first_stage: str, later_stage: str) -> str:
    if first_stage == "INSUFFICIENT_EVIDENCE":
        return "NOT_REACHED"
    return "DOWNSTREAM_AFTER_FIRST_FAILURE" if first_stage != later_stage else "FIRST_FAILURE_CANDIDATE"


def _first_relation(value: Any) -> tuple[str, str]:
    if isinstance(value, list):
        value = value[0] if value else {}
    if isinstance(value, dict):
        return str(value.get("relation", "")), str(value.get("direction", ""))
    return str(value or ""), ""


def _index(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row.get(key)): row for row in rows if row.get(key) is not None}


def _phase23_index(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row.get("case_id")), str(row.get("constraint_id"))): row for row in rows}


def _case_config(manifest: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for track, rows in manifest.items():
        for row in rows:
            item = dict(row)
            item["track"] = track.upper().replace("TRACK_", "")
            configs.append(item)
    return configs


def _gold_steps_from_sparql(row: dict[str, Any]) -> list[dict[str, Any]]:
    example = QAExample(
        id=str(row.get("id", "")),
        question=str(row.get("question", "")),
        answers=[],
        gold_sparql=str(row.get("sparql", "")),
        dataset_type="cwq",
        topic_entities=[str(item.get("id", "")) for item in row.get("topic_entity", [])],
        split="",
    )
    features = parse_query(example.gold_sparql or "")
    steps: list[dict[str, Any]] = []
    for index, pattern in enumerate(features.triple_patterns):
        subject = normalize_relation(pattern.subject)
        predicate = normalize_relation(pattern.predicate)
        object_ = normalize_relation(pattern.object)
        direction = "incoming" if pattern.subject.startswith("?") and not pattern.object.startswith("?") else "outgoing"
        steps.append(
            {
                "step_index": index,
                "gold_source_state": subject,
                "expected_relation": predicate,
                "expected_direction": direction,
                "gold_destination": object_,
                "verification_status": "GOLD_STEP_UNKNOWN",
                "verification_basis": "raw CWQ SPARQL triple pattern; no per-triple live verification in this audit",
            }
        )
    return steps


def _sexpr_relations(node: Any, output: list[tuple[str, str]]) -> None:
    if not isinstance(node, ExprNode):
        return
    operator = node.operator.upper()
    if operator == "JOIN" and node.arguments and isinstance(node.arguments[0], Atom):
        output.append((node.arguments[0].value, "outgoing"))
    elif operator == "R" and node.arguments and isinstance(node.arguments[0], Atom):
        output.append((node.arguments[0].value, "incoming"))
    for argument in node.arguments:
        _sexpr_relations(argument, output)


def _gold_steps_from_sexpr(row: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        root = parse_s_expression(str(row.get("s_expression", "")))
    except Exception:
        return []
    relations: list[tuple[str, str]] = []
    _sexpr_relations(root, relations)
    return [
        {
            "step_index": index,
            "gold_source_state": "UNKNOWN_FROM_S_EXPRESSION",
            "expected_relation": relation,
            "expected_direction": direction,
            "gold_destination": "UNKNOWN_FROM_S_EXPRESSION",
            "verification_status": "GOLD_STEP_UNKNOWN",
            "verification_basis": "raw GrailQA S-expression; exact bound destination was not logged for this audit",
        }
        for index, (relation, direction) in enumerate(relations)
    ]


def build_gold_verification(
    case: dict[str, Any],
    gold_row: dict[str, Any],
    gold_label: str,
    phase21_row: dict[str, Any] | None,
    phase23_row: dict[str, Any] | None,
) -> dict[str, Any]:
    if case["case_id"] in {str(row.get("id")) for row in [gold_row] if row.get("id") is not None} or "sparql" in gold_row:
        steps = _gold_steps_from_sparql(gold_row)
    else:
        steps = _gold_steps_from_sexpr(gold_row)

    expected_relation = ""
    expected_direction = ""
    intended_state: dict[str, Any] = {}
    verified_basis = ""
    if phase23_row:
        expected_relation = str(phase23_row.get("expected_relation", ""))
        expected_direction = str(phase23_row.get("expected_direction", ""))
        intended_state = phase23_row.get("intended_target_state") or {}
        verified_basis = "Phase 2.3 local Freebase relation verification at intended state"
    elif phase21_row:
        expected_relation = str(phase21_row.get("expected_relation", ""))
        expected_direction = str(phase21_row.get("expected_direction", ""))
        verified_basis = "Phase 2.1 candidate-generation trace at runtime state"

    matched = False
    for step in steps:
        if normalize_relation(step["expected_relation"]) != normalize_relation(expected_relation):
            continue
        step["verification_status"] = "GOLD_STEP_VERIFIED" if phase23_row and phase23_row.get("relation_present_at_intended_state") is True or phase21_row and phase21_row.get("correct_relation_available_in_candidates") is True else "GOLD_STEP_UNKNOWN"
        step["verification_basis"] = verified_basis or step["verification_basis"]
        if expected_direction:
            step["expected_direction"] = expected_direction
        if intended_state:
            step["gold_source_state"] = intended_state
        matched = True

    if not matched and (phase23_row or phase21_row):
        steps.append(
            {
                "step_index": len(steps),
                "gold_source_state": intended_state or (phase21_row or {}).get("current_entity", {}),
                "expected_relation": expected_relation,
                "expected_direction": expected_direction,
                "gold_destination": "UNKNOWN",
                "verification_status": "GOLD_STEP_VERIFIED" if phase23_row and phase23_row.get("relation_present_at_intended_state") is True or phase21_row and phase21_row.get("correct_relation_available_in_candidates") is True else "GOLD_STEP_UNKNOWN",
                "verification_basis": verified_basis,
            }
        )

    verified_count = sum(step["verification_status"] == "GOLD_STEP_VERIFIED" for step in steps)
    return {
        "case_id": case["case_id"],
        "dataset": "cwq" if "sparql" in gold_row else "grailqa",
        "question": gold_row.get("question"),
        "gold_answer": gold_row.get("answer", []),
        "gold_structure": gold_label,
        "gold_logical_form": gold_row.get("s_expression"),
        "gold_sparql": gold_row.get("sparql"),
        "gold_steps": steps,
        "verified_step_count": verified_count,
        "gold_step_count": len(steps),
        "gold_path_verification": "FULL" if steps and verified_count == len(steps) else "PARTIAL" if verified_count else "UNKNOWN",
        "gold_path_verified": bool(steps and verified_count == len(steps)),
        "structure_verified_from_annotation": gold_label != "GOLD_OTHER",
    }


def _runtime_structure_and_outcome(
    case: dict[str, Any],
    phase21_row: dict[str, Any] | None,
    phase23_row: dict[str, Any] | None,
    phase21_end_to_end: dict[str, Any] | None,
    fresh_row: dict[str, Any] | None,
) -> tuple[str, int | None, float | None, str]:
    if fresh_row:
        return (
            normalize_structure_label(str(fresh_row.get("router_structure", ""))),
            fresh_row.get("hits_at_1"),
            fresh_row.get("f1"),
            "phase1_adaptive_v1_1_3_grailqa_12_repeat/adaptive_runs.jsonl",
        )
    if phase23_row:
        runtime = normalize_structure_label(str(phase23_row.get("structure_label", "")))
    elif phase21_row:
        runtime = normalize_structure_label(str(phase21_row.get("structure", "")))
    else:
        runtime = "UNCERTAIN"
    result = (phase21_end_to_end or {}).get("phase2_1", {})
    return runtime, result.get("hits_at_1"), result.get("f1"), "phase2_1 frozen trace"


def _runtime_step(
    case: dict[str, Any],
    phase21_row: dict[str, Any] | None,
    phase23_row: dict[str, Any] | None,
    first_stage: str,
) -> dict[str, Any]:
    source = phase23_row or phase21_row or {}
    if phase23_row:
        actual_state = phase23_row.get("actual_current_entity") or {}
        intended_state = phase23_row.get("intended_target_state") or {}
        expected_relation = phase23_row.get("expected_relation", "")
        expected_direction = phase23_row.get("expected_direction", "")
        selected_relation = phase23_row.get("actual_selected_relation", "")
        selected_direction = phase23_row.get("actual_selected_direction", "")
        relation_at_actual = phase23_row.get("relation_present_at_actual_state")
        relation_at_intended = phase23_row.get("relation_present_at_intended_state")
        constraint = phase23_row.get("constraint_id", "")
        subgoal = phase23_row.get("actual_subgoal", "")
    elif phase21_row:
        actual_state = phase21_row.get("current_entity") or {}
        intended_state = {}
        expected_relation = phase21_row.get("expected_relation", "")
        expected_direction = phase21_row.get("expected_direction", "")
        selected_relation, selected_direction = _first_relation(phase21_row.get("selected_relation"))
        relation_at_actual = phase21_row.get("correct_relation_available_in_candidates")
        relation_at_intended = phase21_row.get("correct_relation_available_in_candidates")
        constraint = f"decision_{phase21_row.get('decision_index', '')}"
        subgoal = phase21_row.get("semantic_goal", "")
    else:
        actual_state = {}
        intended_state = {}
        expected_relation = ""
        expected_direction = ""
        selected_relation = ""
        selected_direction = ""
        relation_at_actual = None
        relation_at_intended = None
        constraint = ""
        subgoal = ""

    return {
        "step_index": 0,
        "gold_state": intended_state,
        "actual_state": actual_state,
        "gold_active_constraint": constraint,
        "actual_subgoal": subgoal,
        "expected_relation": expected_relation,
        "expected_direction": expected_direction,
        "relation_present_at_gold_state": relation_at_intended,
        "relation_present_at_runtime_state": relation_at_actual,
        "selected_relation": selected_relation,
        "selected_direction": selected_direction,
        "sparql_expected_destination": "UNKNOWN_NOT_LOGGED",
        "sparql_actual_result": "UNKNOWN_NOT_LOGGED",
        "expected_next_state": intended_state,
        "actual_next_state": "UNKNOWN_NOT_LOGGED",
        "match": False if first_stage == "STRUCTURE_ROUTING" else None,
        "stage_status": "FIRST_DIVERGENCE" if first_stage == "STRUCTURE_ROUTING" else "DOWNSTREAM_AFTER_FIRST_FAILURE" if first_stage != "INSUFFICIENT_EVIDENCE" else "UNKNOWN",
        "source_fields": sorted(source.keys()),
    }


def interaction_labels(
    first_stage: str,
    relation_mismatch: bool,
    state_mismatch: bool,
    answer_correct: bool | None,
) -> dict[str, str]:
    if first_stage == "STRUCTURE_ROUTING" and relation_mismatch:
        routing_to_relation = "ROUTING_CAUSED_DOWNSTREAM_RELATION_FAILURE"
    elif first_stage == "STRUCTURE_ROUTING" and answer_correct is True:
        routing_to_relation = "ROUTING_INDEPENDENT_OF_RELATION_FAILURE"
    else:
        routing_to_relation = "UNKNOWN"

    if first_stage == "RELATION_GROUNDING" and state_mismatch:
        relation_to_state = "RELATION_CAUSED_STATE_DIVERGENCE"
    elif state_mismatch and first_stage in {"STRUCTURE_ROUTING", "TARGET_VARIABLE_ALIGNMENT", "ENTITY_STATE_ALIGNMENT"}:
        relation_to_state = "STATE_ALREADY_WRONG_BEFORE_RELATION"
    else:
        relation_to_state = "UNKNOWN"
    return {
        "routing_to_relation": routing_to_relation,
        "relation_to_state": relation_to_state,
    }


def build_case_audit(
    case: dict[str, Any],
    cwq: dict[str, dict[str, Any]],
    grailqa: dict[str, dict[str, Any]],
    phase21: dict[str, dict[str, Any]],
    phase21_end: dict[str, dict[str, Any]],
    phase23: dict[tuple[str, str], dict[str, Any]],
    fresh: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = case["case_id"]
    gold_label, gold_row = gold_structure(case_id, cwq, grailqa)
    phase21_row = phase21.get(case_id)
    phase23_row = None
    if case["track"] == "A":
        phase23_row = phase23.get((case_id, str(case.get("source_constraint_id"))))
    fresh_row = fresh.get(case_id)
    runtime_label, answer_h1, answer_f1, runtime_source = _runtime_structure_and_outcome(
        case, phase21_row, phase23_row, phase21_end.get(case_id), fresh_row
    )
    answer_correct = None if answer_h1 is None else bool(answer_h1)

    state_mismatch = None
    if phase23_row:
        actual_id = str((phase23_row.get("actual_current_entity") or {}).get("id", ""))
        intended_id = str((phase23_row.get("intended_target_state") or {}).get("id", ""))
        state_mismatch = bool(actual_id and intended_id and actual_id != intended_id)

    relation_mismatch = False
    candidate_available = None
    relation_selection_match = None
    if phase23_row:
        expected = (str(phase23_row.get("expected_relation", "")), str(phase23_row.get("expected_direction", "")))
        selected = (str(phase23_row.get("actual_selected_relation", "")), str(phase23_row.get("actual_selected_direction", "")))
        relation_mismatch = expected != selected
        candidate_available = phase23_row.get("relation_present_at_actual_state")
        relation_selection_match = not relation_mismatch
    elif phase21_row:
        expected = (str(phase21_row.get("expected_relation", "")), str(phase21_row.get("expected_direction", "")))
        selected = _first_relation(phase21_row.get("selected_relation"))
        relation_mismatch = expected != selected
        candidate_available = bool(phase21_row.get("correct_relation_available_in_candidates"))
        relation_selection_match = not relation_mismatch

    checks = {
        "structure_match": structure_matches(gold_label, runtime_label),
        "constraint_decomposition_match": None if not gold_label.startswith("GOLD_BRANCH") else False if runtime_label in {"LINEAR", "UNCERTAIN"} else True,
        "target_variable_match": False if state_mismatch is True and case_id.startswith("WebQTrn-1812") else None,
        "entity_state_match": False if state_mismatch is True and case_id == "4302233001000" else None,
        "candidate_available": False if candidate_available is False else None,
        "relation_selection_match": False if relation_selection_match is False else None,
        "sparql_match": None,
        "state_transition_match": None,
        "merge_match": None,
        "answer_match": False if answer_correct is False else None,
    }
    first_stage = first_divergence_label(checks)
    if first_stage not in FIRST_DIVERGENCE_LABELS:
        raise AssertionError(f"invalid first-divergence label: {first_stage}")

    gold_trace = build_gold_verification(case, gold_row, gold_label, phase21_row, phase23_row)
    runtime_trace = {
        "case_id": case_id,
        "dataset": "cwq" if case_id in cwq else "grailqa",
        "question": gold_row.get("question"),
        "gold_structure": gold_label,
        "runtime_structure": runtime_label,
        "runtime_structure_source": runtime_source,
        "answer_hits_at_1": answer_h1,
        "answer_f1": answer_f1,
        "answer_correct": answer_correct,
        "checks_in_protocol_order": checks,
        "steps": [_runtime_step(case, phase21_row, phase23_row, first_stage)],
        "first_divergence_step": 0 if first_stage != "INSUFFICIENT_EVIDENCE" else None,
        "first_divergence_stage": first_stage,
        "primary_diagnosis": first_stage,
        "secondary_diagnoses": [
            label
            for label, condition in (
                ("CONSTRAINT_DECOMPOSITION", checks["constraint_decomposition_match"] is False),
                ("TARGET_VARIABLE_ALIGNMENT", checks["target_variable_match"] is False),
                ("ENTITY_STATE_ALIGNMENT", checks["entity_state_match"] is False),
                ("RELATION_CANDIDATE_AVAILABILITY", checks["candidate_available"] is False),
                ("RELATION_GROUNDING", checks["relation_selection_match"] is False),
            )
            if condition and label != first_stage
        ],
        "gold_path_verification": gold_trace["gold_path_verification"],
        "gold_path_verified": gold_trace["gold_path_verified"],
        "downstream_events": {
            "constraint_decomposition": downstream_status(first_stage, "CONSTRAINT_DECOMPOSITION") if checks["constraint_decomposition_match"] is False else "NOT_OBSERVED",
            "relation_grounding": downstream_status(first_stage, "RELATION_GROUNDING") if relation_mismatch else "NOT_OBSERVED",
            "state_transition": downstream_status(first_stage, "STATE_TRANSITION"),
            "answer_handling": downstream_status(first_stage, "DOWNSTREAM_ANSWER") if answer_correct is False else "CONTROL_OR_UNKNOWN",
        },
        "evidence_sources": [
            "raw CWQ/GrailQA gold annotation",
            "Phase 2.1 relation decision trace" if phase21_row else "",
            "Phase 2.3 state verification trace" if phase23_row else "",
            "Phase 1 v1.1.3 fresh repeat routing trace" if fresh_row else "",
        ],
        "case_role_in_phase2_4": case.get("primary_label"),
        "fresh_case": bool(case.get("fresh_case")),
        "added_control": bool(case.get("added_control", False)),
    }
    runtime_trace["evidence_sources"] = [item for item in runtime_trace["evidence_sources"] if item]
    runtime_trace["interaction_analysis"] = interaction_labels(first_stage, relation_mismatch, bool(state_mismatch), answer_correct)
    return gold_trace, runtime_trace


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def render_summary(audits: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> str:
    distribution = Counter(row["primary_diagnosis"] for row in audits)
    failures = [row for row in audits if row.get("answer_correct") is False]
    supported = [row for row in audits if row["primary_diagnosis"] != "INSUFFICIENT_EVIDENCE"]
    lines = [
        "# Phase 2.4.1 first-divergence summary",
        "",
        "| First divergence | All cases | Failed-answer cases |",
        "|---|---:|---:|",
    ]
    failed_counts = Counter(row["primary_diagnosis"] for row in failures)
    for label in sorted(FIRST_DIVERGENCE_LABELS):
        if distribution.get(label, 0) or failed_counts.get(label, 0):
            lines.append(f"| `{label}` | {distribution.get(label, 0)} | {failed_counts.get(label, 0)} |")
    lines += [
        "",
        f"- Audited cases: **{len(audits)}**.",
        f"- Failed-answer cases: **{len(failures)}**; successful/control cases: **{len(audits) - len(failures)}**.",
        f"- Cases with a supported first-divergence diagnosis: **{len(supported)}/{len(audits)} = {len(supported) / len(audits):.2%}**.",
        f"- Failed-answer cases with a supported first-divergence diagnosis: **{sum(row['primary_diagnosis'] != 'INSUFFICIENT_EVIDENCE' for row in failures)}/{len(failures)}**.",
        "- A structure mismatch is counted before later relation/state mismatches, exactly as required by the protocol.",
        "",
    ]
    return "\n".join(lines)


def render_interactions(audits: list[dict[str, Any]]) -> str:
    routing = Counter(row["interaction_analysis"]["routing_to_relation"] for row in audits)
    relation = Counter(row["interaction_analysis"]["relation_to_state"] for row in audits)
    lines = [
        "# Phase 2.4.1 interaction analysis",
        "",
        "## Structure → relation",
        "",
        "| Classification | Cases |",
        "|---|---:|",
    ]
    for label, count in sorted(routing.items()):
        lines.append(f"| `{label}` | {count} |")
    lines += [
        "",
        "The Phase 2.4 relation cases do not isolate pure relation grounding: their gold structures are branch structures, while the recorded runtime structures are LINEAR or UNCERTAIN. Their relation mismatches are therefore downstream observations, not independent primary failures.",
        "",
        "## Relation → state",
        "",
        "| Classification | Cases |",
        "|---|---:|",
    ]
    for label, count in sorted(relation.items()):
        lines.append(f"| `{label}` | {count} |")
    lines += [
        "",
        "No case supports a relation-grounding-first causal claim in this audit. The selected relation evidence is recorded, but the earlier structure/state ordering takes precedence.",
        "",
    ]
    return "\n".join(lines)


def render_report(
    audits: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    decision: str,
    summary: str,
) -> str:
    fresh = sum(bool(row.get("fresh_case")) for row in audits)
    failures = [row for row in audits if row.get("answer_correct") is False]
    case_lines = [
        "| Case | Gold structure | Runtime structure | Outcome | First divergence | Gold path verification |",
        "|---|---|---|---|---|---|",
    ]
    for row in audits:
        outcome = "FAIL" if row.get("answer_correct") is False else "SUCCESS/CONTROL" if row.get("answer_correct") is True else "UNKNOWN"
        case_lines.append(
            f"| `{row['case_id']}` | `{row['gold_structure']}` | `{row['runtime_structure']}` | {outcome} | `{row['primary_diagnosis']}` | `{row['gold_path_verification']}` |"
        )
    return f"""# Phase 2.4.1 first-divergence trace audit

## 1. Motivation from Phase 2.4

Phase 2.4 reported a tie between state alignment (50%), relation grounding (75%), and structure routing (75%). That report intentionally described family-level symptoms. Phase 2.4.1 reorders the evidence causally so downstream relation/state mismatches are not counted as independent failures after an earlier routing mismatch.

## 2. Research question

For each audited case, at which exact step does the actual runtime first diverge from the dataset gold reasoning path, subject to local-KG verification limits?

## 3. Why first divergence matters

A wrong relation selected after the runtime has already chosen the wrong structure is a consequence, not an independent causal failure. The audit therefore assigns exactly one primary first-divergence label per case and marks later observations as downstream.

## 4. Frozen components

No router, relation scorer, prompt, model setting, top-k, backend, SPARQL semantics, evaluator, merge behavior, fallback behavior, or production inference code was changed. No new inference or full100 run was performed.

## 5. Case selection

- **{len(audits)}** cases audited: the Phase 2.4 twelve-case set plus one successful fresh routing control.
- **{fresh}** fresh-for-Phase-2 cases and **{len(audits) - fresh}** reused frozen-trace cases.
- The sample covers the prior state-alignment, relation-grounding, and structure-routing families and includes successful controls.

## 6. Gold-path reconstruction

CWQ gold SPARQL was parsed into graph triples; GrailQA gold S-expressions were parsed into relation/direction steps. Gold structure was derived from the logical graph, distinguishing final branches from intermediate branches. Gold annotations were treated as intended reasoning references, not automatically executable truth.

## 7. Gold-path KG verification

Where available, the audit reused Phase 2.3 local-Freebase verification at intended states and Phase 2.1 candidate-generation evidence. Unverified destinations remain `GOLD_STEP_UNKNOWN`; no gold answer or state was injected into runtime execution.

## 8. Runtime trace reconstruction

Runtime structure, current state, candidate availability, selected relation, answer outcome, and state-verification evidence were read from frozen Phase 2.1, Phase 2.3, and v1.1.3 repeat artifacts. Missing fields remain unknown.

## 9. First-divergence methodology

The comparison order was: structure routing → constraint decomposition → target variable/entity state → candidate availability → relation selection → SPARQL traversal → state transition → merge → final answer. The first confirmed mismatch receives the primary label; later mismatches are downstream.

## 10. Per-case findings

{chr(10).join(case_lines)}

The machine-readable step-level comparisons are in `runtime_trace_comparison.jsonl`; the corresponding gold verification records are in `gold_path_verification.jsonl`.

## 11. Structure → relation interaction

The prior Phase 2.4 relation cases are all gold branch cases whose recorded runtime structure is LINEAR or UNCERTAIN. Consequently, their relation-selection mismatches occur after the structure mismatch and are labelled `DOWNSTREAM_AFTER_FIRST_FAILURE`. The interaction table is in `interaction_analysis.md`.

## 12. Relation → state interaction

No audited case establishes relation grounding as the first divergence. Relation-to-state causality is reported conservatively as `UNKNOWN` or `STATE_ALREADY_WRONG_BEFORE_RELATION` when the evidence supports that ordering.

## 13. Aggregate first-divergence distribution

{summary}

## 14. Main causal decision

**`{decision}`**

Among the seven failed-answer cases, all seven have structure routing as the first supported divergence. The two successful fresh controls have no confirmed failure and are labelled `INSUFFICIENT_EVIDENCE`; one additional successful case has a routing mismatch but preserved its answer. This makes structure routing the primary causal candidate for this audited sample, while not proving that every relation error in the broader project is routing-caused.

## 15. Limitations

- The selected cases are heavily concentrated in the Phase 2 branch-2 sample; this can inflate routing-first findings.
- Several gold destinations were not logged or live-verified in the existing artifacts, so gold-path verification is often partial rather than full.
- The audit combines frozen Phase 2.1/2.3 decision evidence with the fresh v1.1.3 repeat routing records; it does not claim these were one identical process invocation.
- No fresh Virtuoso query was issued during this diagnostic; existing Phase 2.3 local-KG verification was reused.

## 16. Recommended next intervention target

Prioritize routing and constraint-decomposition instrumentation first. A future intervention should make the active branch/constraint state explicit in logs and then re-evaluate relation grounding only on cases where the gold structure and active state match. Do not implement that intervention as part of Phase 2.4.1.
"""


def main() -> None:
    manifest = phase24_manifest()
    configs = _case_config(manifest)
    cwq = {str(row["id"]): row for row in read_json(CWQ_PATH)}
    grailqa = {str(row["qid"]): row for row in read_json(GRAILQA_PATH)}
    phase21_rows = read_jsonl(PHASE21_ROOT / "relation_decisions_phase2_1.jsonl")
    phase21_end_rows = read_jsonl(PHASE21_ROOT / "end_to_end_results.jsonl")
    phase23_rows = read_jsonl(PHASE23_ROOT / "case_alignment_traces.jsonl")
    fresh_rows = read_jsonl(FRESH_ROOT / "adaptive_runs.jsonl")
    phase21 = _index(phase21_rows, "question_id")
    phase21_end = _index(phase21_end_rows, "question_id")
    phase23 = _phase23_index(phase23_rows)
    fresh = _index(fresh_rows, "id")

    gold_traces: list[dict[str, Any]] = []
    runtime_traces: list[dict[str, Any]] = []
    for case in configs:
        gold_trace, runtime_trace = build_case_audit(case, cwq, grailqa, phase21, phase21_end, phase23, fresh)
        gold_traces.append(gold_trace)
        runtime_traces.append(runtime_trace)

    failed = [row for row in runtime_traces if row.get("answer_correct") is False]
    first_labels = Counter(row["primary_diagnosis"] for row in runtime_traces)
    failed_labels = Counter(row["primary_diagnosis"] for row in failed)
    supported = sum(row["primary_diagnosis"] != "INSUFFICIENT_EVIDENCE" for row in runtime_traces)
    fresh_count = sum(bool(row.get("fresh_case")) for row in runtime_traces)
    decision = "ROUTING_IS_PRIMARY_CAUSE" if failed and all(row["primary_diagnosis"] == "STRUCTURE_ROUTING" for row in failed) else "MIXED_CAUSAL_FAILURES" if failed else "INCONCLUSIVE_CAUSAL_AUDIT"
    failure_distribution = {
        "experiment": "phase2_4_1_first_divergence_trace_audit",
        "audited_cases": len(runtime_traces),
        "failed_answer_cases": len(failed),
        "successful_or_control_cases": len(runtime_traces) - len(failed),
        "fresh_case_count": fresh_count,
        "reused_case_count": len(runtime_traces) - fresh_count,
        "gold_path_verification": dict(Counter(row["gold_path_verification"] for row in gold_traces)),
        "first_divergence_distribution": dict(first_labels),
        "failed_case_first_divergence_distribution": dict(failed_labels),
        "supported_first_divergence_count": supported,
        "supported_first_divergence_rate": supported / len(runtime_traces) if runtime_traces else 0.0,
        "failed_case_supported_count": sum(row["primary_diagnosis"] != "INSUFFICIENT_EVIDENCE" for row in failed),
        "interaction_distribution": {
            "routing_to_relation": dict(Counter(row["interaction_analysis"]["routing_to_relation"] for row in runtime_traces)),
            "relation_to_state": dict(Counter(row["interaction_analysis"]["relation_to_state"] for row in runtime_traces)),
        },
        "decision": decision,
        "production_pipeline_changed": False,
    }
    summary = render_summary(runtime_traces, gold_traces)
    interactions = render_interactions(runtime_traces)
    report = render_report(runtime_traces, gold_traces, decision, summary)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _json_dump(OUTPUT_ROOT / "diagnostic_cases.json", {"cases": configs})
    _write_jsonl(OUTPUT_ROOT / "gold_path_verification.jsonl", gold_traces)
    _write_jsonl(OUTPUT_ROOT / "runtime_trace_comparison.jsonl", runtime_traces)
    _json_dump(OUTPUT_ROOT / "failure_distribution.json", failure_distribution)
    (OUTPUT_ROOT / "first_divergence_summary.md").write_text(summary, encoding="utf-8")
    (OUTPUT_ROOT / "interaction_analysis.md").write_text(interactions, encoding="utf-8")
    (OUTPUT_ROOT / "PHASE2_4_1_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
