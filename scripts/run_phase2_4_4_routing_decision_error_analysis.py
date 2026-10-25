#!/usr/bin/env python3
"""Run the frozen Phase 2.4.4 routing-decision error analysis.

The audit compares offline gold structure with the recorded v1.1.3 router
input/output and execution traces.  It is diagnostic only: no model, prompt,
router, SPARQL endpoint, or inference behavior is changed.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_phase2_4_2_harmful_vs_benign_routing import (  # noqa: E402
    _cwq_graph,
    gold_structure_features,
    read_json,
    read_jsonl,
)


OUTPUT_ROOT = ROOT / "outputs/experiments/phase2_4_4/routing_decision_error_analysis"
PHASE241_TRACE = ROOT / "outputs/experiments/phase2_4_1/first_divergence_trace_audit/runtime_trace_comparison.jsonl"
CWQ_PATH = ROOT / "datasets/cwq/cwq.json"
GRAILQA_PATH = ROOT / "datasets/grailqa/official/grailqa_v1.0_train.json"


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    detail_path: str
    source_experiment: str
    reused_from_phase24: bool


CASE_SPECS = (
    CaseSpec("WebQTrn-1812_8126be584feae3f358a930823db22c4e", "outputs/experiments/phase2_1/relation_grounding/details/adaptive/cwq_branch_2/WebQTrn-1812_8126be584feae3f358a930823db22c4e.json", "phase2_1_relation_grounding", True),
    CaseSpec("4302233001000", "outputs/experiments/phase2_1/relation_grounding/details/adaptive/grailqa_branch_2/4302233001000.json", "phase2_1_relation_grounding", True),
    CaseSpec("4301990001000", "outputs/experiments/phase2_1/relation_grounding/details/adaptive/grailqa_branch_2/4301990001000.json", "phase2_1_relation_grounding", True),
    CaseSpec("WebQTest-743_af8b975343e4098913128e9ac7d04c05", "outputs/experiments/phase2_1/relation_grounding/details/adaptive/cwq_branch_2/WebQTest-743_af8b975343e4098913128e9ac7d04c05.json", "phase2_1_relation_grounding", True),
    CaseSpec("WebQTest-719_e56e856592a9a57b9e331b718adc1b75", "outputs/experiments/phase2_1/relation_grounding/details/adaptive/cwq_branch_2/WebQTest-719_e56e856592a9a57b9e331b718adc1b75.json", "phase2_1_relation_grounding", True),
    CaseSpec("WebQTest-759_88081dc1a259cbd172b89242384ee1cc", "outputs/experiments/phase2_1/relation_grounding/details/adaptive/cwq_branch_2/WebQTest-759_88081dc1a259cbd172b89242384ee1cc.json", "phase2_1_relation_grounding", True),
    CaseSpec("WebQTrn-64_984678664419049133d2bd748b68df1b", "outputs/experiments/phase2_1/relation_grounding/details/adaptive/cwq_branch_2/WebQTrn-64_984678664419049133d2bd748b68df1b.json", "phase2_1_relation_grounding", True),
    CaseSpec("WebQTest-191_ed2cadafdbc0ac07e3fedfdfc504e54b", "outputs/experiments/phase2_1/relation_grounding/details/adaptive/cwq_branch_2/WebQTest-191_ed2cadafdbc0ac07e3fedfdfc504e54b.json", "phase2_1_relation_grounding", True),
    CaseSpec("3201049003000", "outputs/experiments/phase1/adaptive_v1_1_3/grailqa_12_repeat/details/repeat_1/adaptive/3201049003000.json", "phase1_adaptive_v1_1_3_grailqa_repeat_1", True),
    CaseSpec("3201277005000", "outputs/experiments/phase1/adaptive_v1_1_3/grailqa_12_repeat/details/repeat_1/adaptive/3201277005000.json", "phase1_adaptive_v1_1_3_grailqa_repeat_1", True),
    CaseSpec("3202841012000", "outputs/experiments/phase1/adaptive_v1_1_3/grailqa_12_repeat/details/repeat_1/adaptive/3202841012000.json", "phase1_adaptive_v1_1_3_grailqa_repeat_1", True),
    CaseSpec("3203234013000", "outputs/experiments/phase1/adaptive_v1_1_3/grailqa_12_repeat/details/repeat_1/adaptive/3203234013000.json", "phase1_adaptive_v1_1_3_grailqa_repeat_1", True),
    CaseSpec("3205047010000", "outputs/experiments/phase1/adaptive_v1_1_3/grailqa_12_repeat/details/repeat_1/adaptive/3205047010000.json", "phase1_adaptive_v1_1_3_grailqa_repeat_1", True),
    CaseSpec("3205365004000", "outputs/experiments/phase1/adaptive_v1_1_3/grailqa_12_repeat/details/repeat_1/adaptive/3205365004000.json", "phase1_adaptive_v1_1_3_grailqa_repeat_1", False),
    CaseSpec("4302370014000", "outputs/experiments/phase1/adaptive_v1_1_3/grailqa_12_repeat/details/repeat_1/adaptive/4302370014000.json", "phase1_adaptive_v1_1_3_grailqa_repeat_1", False),
)


PRIMARY_ERRORS = {
    "CONSTRAINT_OMISSION",
    "FALSE_SEQUENTIALIZATION",
    "SHARED_ANSWER_MISSED",
    "SHARED_INTERMEDIATE_MISSED",
    "INDEPENDENT_SOURCE_MISSED",
    "BRANCH_TYPE_CONFUSION",
    "OVER_CONSERVATIVE_UNCERTAINTY",
    "TYPE_CONSTRAINT_CONFUSION",
    "ANSWER_TYPE_CONFLICT",
    "OTHER",
    "INSUFFICIENT_EVIDENCE",
}


def normalize_structure(value: Any) -> str:
    value = str(value or "").upper()
    if value in {"BRANCH", "BRANCH_FINAL", "PARALLEL"}:
        return "BRANCH_FINAL"
    if value in {"BRANCH_INTERMEDIATE"}:
        return "BRANCH_INTERMEDIATE"
    if value in {"LINEAR", "DIRECT", "COR"}:
        return "LINEAR"
    if value in {"UNCERTAIN", "UNCERTAINTY"}:
        return "UNCERTAIN"
    return value or "UNKNOWN"


def load_router_record(detail: dict[str, Any]) -> dict[str, Any]:
    entries = [
        item for item in detail.get("prompt_history", [])
        if isinstance(item, dict) and "router" in str(item.get("type", "")).lower()
    ]
    entry = entries[-1] if entries else {}
    parsed = entry.get("parsed_result")
    if not isinstance(parsed, dict):
        try:
            parsed = json.loads(entry.get("response", "{}"))
        except (TypeError, json.JSONDecodeError):
            parsed = {}
    prompt = str(entry.get("prompt", ""))
    topic_entities: Any = "UNKNOWN_NOT_LOGGED"
    match = re.search(r"Runtime topic entities:\s*(\[.*?\])\s*$", prompt, flags=re.MULTILINE | re.DOTALL)
    if match:
        try:
            topic_entities = json.loads(match.group(1))
        except json.JSONDecodeError:
            topic_entities = "UNKNOWN_NOT_LOGGED"
    branches = parsed.get("branches") if isinstance(parsed.get("branches"), list) else []
    return {
        "prompt": prompt or "UNKNOWN_NOT_LOGGED",
        "parsed": parsed,
        "structure": normalize_structure(parsed.get("structure") or detail.get("router_structure")),
        "execution_mode": parsed.get("execution_mode", "UNKNOWN_NOT_LOGGED"),
        "confidence": parsed.get("confidence", "UNKNOWN_NOT_LOGGED"),
        "reason": str(parsed.get("reason") or detail.get("router_reason") or "UNKNOWN_NOT_LOGGED"),
        "requested_final_answer_type": parsed.get("requested_final_answer_type") or detail.get("requested_final_answer_type") or "UNKNOWN_NOT_LOGGED",
        "requested_final_semantic_property": parsed.get("requested_final_semantic_property") or detail.get("requested_final_semantic_property") or "UNKNOWN_NOT_LOGGED",
        "shared_answer_type": parsed.get("shared_answer_type") or detail.get("shared_answer_type") or "UNKNOWN_NOT_LOGGED",
        "branches": branches,
        "unsupported_structure": parsed.get("unsupported_structure", "UNKNOWN_NOT_LOGGED"),
        "runtime_topic_entities": topic_entities,
        "validator_outcome": {
            "fallback_reason": detail.get("fallback_reason", "UNKNOWN_NOT_LOGGED"),
            "validation_failures": detail.get("validation_failures", "UNKNOWN_NOT_LOGGED"),
            "validation_reason": detail.get("validation_reason", "UNKNOWN_NOT_LOGGED"),
            "branch_property_compatibility": detail.get("branch_property_compatibility", "UNKNOWN_NOT_LOGGED"),
        },
    }


def gold_source_count(case_id: str, cwq: dict[str, Any], grailqa: dict[str, Any], features: dict[str, Any]) -> int:
    if case_id in cwq:
        _, graph = _cwq_graph(cwq[case_id])
        candidates = graph.branch_candidates()
        if candidates:
            return max(candidates.values())
        return len({item.get("id") for item in cwq[case_id].get("topic_entity", []) if item.get("id")})
    return len({item["id"] for item in features.get("anchor_ids", []) if item.get("id")})


def gold_reference(case_id: str, cwq: dict[str, Any], grailqa: dict[str, Any]) -> dict[str, Any]:
    features = gold_structure_features(case_id, cwq, grailqa)
    sources = gold_source_count(case_id, cwq, grailqa, features)
    branch_final = features["gold_structure"] == "GOLD_BRANCH_FINAL"
    branch_intermediate = features["gold_structure"] == "GOLD_BRANCH_INTERMEDIATE"
    return {
        "dataset": features["dataset"],
        "question": features["question"],
        "gold_structure": features["gold_structure"],
        "gold_constraints": [
            {"constraint_id": f"constraint_{index}", "relation": relation}
            for index, relation in enumerate(features["relations"])
        ],
        "gold_relation_names": features["relations"],
        "gold_source_anchors": features["anchor_ids"],
        "gold_independent_sources": sources,
        "gold_shared_answer": branch_final and features["shared_variable_type"] == "ANSWER",
        "gold_shared_intermediate": branch_intermediate or features["shared_variable_type"] == "INTERMEDIATE",
        "gold_answer_variable": features["shared_variable_type"],
        "gold_dependency_type": "SHARED_ANSWER" if branch_final else "SHARED_INTERMEDIATE" if branch_intermediate else "LINEAR_CHAIN",
        "gold_explicit_conjunction": features["constraint_convergence"],
        "gold_contains_cvt": features["contains_cvt"],
    }


def load_phase241() -> dict[str, dict[str, Any]]:
    return {str(item["case_id"]): item for item in read_jsonl(PHASE241_TRACE)}


def router_evidence_text(router: dict[str, Any]) -> str:
    branches = json.dumps(router.get("branches", []), ensure_ascii=False)
    return " ".join([
        router.get("reason", ""),
        str(router.get("requested_final_answer_type", "")),
        str(router.get("requested_final_semantic_property", "")),
        str(router.get("shared_answer_type", "")),
        branches,
    ]).lower()


def router_detected_source_count(router: dict[str, Any]) -> int:
    branches = router.get("branches", [])
    if len(branches) >= 2:
        return len(branches)
    evidence = router_evidence_text(router)
    if any(token in evidence for token in ("not independent", "no independent", "independently invalid")):
        return 1
    if any(token in evidence for token in ("two grounded", "two distinct", "two independent", "multiple grounded")):
        return 2
    if any(token in evidence for token in ("only one", "single identified", "single entity", "one linked", "not independent")):
        return 1
    return 0


def independent_source_label(gold: dict[str, Any], router: dict[str, Any]) -> str:
    if gold["gold_independent_sources"] < 2:
        return "UNKNOWN"
    detected = router_detected_source_count(router)
    evidence = router_evidence_text(router)
    if detected >= 2:
        return "INDEPENDENT_SOURCES_DETECTED"
    if detected == 1 and "two" in evidence:
        return "SOURCE_DEPENDENCY_MISREAD"
    if detected <= 1:
        return "INDEPENDENT_SOURCE_MISSED"
    return "UNKNOWN"


def shared_answer_label(gold: dict[str, Any], router: dict[str, Any]) -> str:
    if not gold["gold_shared_answer"]:
        return "UNKNOWN"
    branches = router.get("branches", [])
    evidence = router_evidence_text(router)
    if len(branches) >= 2:
        targets = {str(item.get("target_answer_type", "")).lower() for item in branches}
        if len(targets) == 1 and "" not in targets:
            return "SHARED_ANSWER_DETECTED"
        return "SHARED_ANSWER_MISINTERPRETED"
    if "same final" in evidence or "same semantic" in evidence or "same requested" in evidence:
        return "SHARED_ANSWER_MISINTERPRETED"
    return "SHARED_ANSWER_MISSED"


def shared_intermediate_label(gold: dict[str, Any], router: dict[str, Any]) -> str:
    if not gold["gold_shared_intermediate"]:
        return "UNKNOWN"
    evidence = router_evidence_text(router)
    branches = router.get("branches", [])
    if any(token in evidence for token in ("shared intermediate", "converge", "intermediate variable")):
        return "SHARED_INTERMEDIATE_DETECTED"
    if any(token in evidence for token in ("sequential", "sequentially", "path rather than")):
        return "SHARED_INTERMEDIATE_MISINTERPRETED"
    if branches:
        return "SHARED_INTERMEDIATE_MISINTERPRETED"
    return "SHARED_INTERMEDIATE_MISSED"


def branch_type_label(gold: dict[str, Any], router: dict[str, Any]) -> str:
    predicted = router["structure"]
    expected = gold["gold_structure"]
    if expected == "GOLD_BRANCH_FINAL":
        if predicted == "BRANCH_FINAL":
            return "CORRECT_BRANCH_TYPE"
        if predicted == "LINEAR":
            return "FINAL_AS_LINEAR"
        if predicted == "UNCERTAIN":
            return "UNCERTAIN_INSTEAD_OF_BRANCH"
        return "OTHER"
    if expected == "GOLD_BRANCH_INTERMEDIATE":
        if predicted == "BRANCH_INTERMEDIATE":
            return "CORRECT_BRANCH_TYPE"
        if predicted == "LINEAR":
            return "INTERMEDIATE_AS_LINEAR"
        if predicted == "BRANCH_FINAL":
            return "INTERMEDIATE_AS_FINAL"
        if predicted == "UNCERTAIN":
            return "UNCERTAIN_INSTEAD_OF_BRANCH"
        return "OTHER"
    return "CORRECT_BRANCH_TYPE" if predicted == "LINEAR" else "OTHER"


def sequential_label(gold: dict[str, Any], router: dict[str, Any]) -> str:
    if gold["gold_structure"] not in {"GOLD_BRANCH_FINAL", "GOLD_BRANCH_INTERMEDIATE"}:
        return "TRUE_SEQUENTIAL"
    if router["structure"] == "BRANCH_FINAL":
        return "UNKNOWN"
    evidence = router_evidence_text(router)
    if any(token in evidence for token in ("sequential", "sequentially", "path rather than", "single entity", "single identified", "only one")):
        return "FALSE_SEQUENTIALIZATION"
    return "UNKNOWN"


def uncertainty_label(gold: dict[str, Any], router: dict[str, Any]) -> str:
    if router["structure"] != "UNCERTAIN":
        return "UNKNOWN"
    evidence = router_evidence_text(router)
    if any(token in evidence for token in ("different semantic", "same semantic property", "semantic property required", "differently typed", "incompatible", "different properties")):
        return "CONFLICTING_SIGNALS"
    if any(token in evidence for token in ("generic", "type-like", "unsupported")):
        return "OVER_CONSERVATIVE_UNCERTAINTY"
    return "UNKNOWN"


def type_constraint_label(gold: dict[str, Any], router: dict[str, Any]) -> str:
    evidence = router_evidence_text(router)
    if any(token in evidence for token in ("generic", "type-like")):
        return "TYPE_CONSTRAINT_FALSE_LINEARIZATION"
    if "type" in evidence and gold["gold_structure"] == "GOLD_LINEAR":
        return "TYPE_CONSTRAINT_CORRECTLY_IGNORED"
    return "UNKNOWN"


def answer_type_label(gold: dict[str, Any], router: dict[str, Any], routing_outcome: str) -> str:
    evidence = router_evidence_text(router)
    if routing_outcome == "CORRECT_ROUTING_CONTROL" and router.get("branches"):
        return "ANSWER_TYPE_SUPPORTIVE"
    if any(token in evidence for token in ("different semantic", "same semantic property", "semantic property required", "differently typed", "different properties", "incompatible")):
        return "ANSWER_TYPE_CONFLICTING"
    if routing_outcome in {"HARMFUL_ROUTING_MISMATCH", "BENIGN_ROUTING_MISMATCH"}:
        return "ANSWER_TYPE_NEUTRAL"
    return "UNKNOWN"


def constraint_detection(gold: dict[str, Any], router: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = router_evidence_text(router)
    branches = router.get("branches", [])
    output: list[dict[str, Any]] = []
    semantic_stop = {"type", "object", "common", "base", "people", "person", "location", "music", "book", "film", "government", "fictional", "universe"}
    for item in gold["gold_constraints"]:
        relation = str(item["relation"])
        tokens = [token for token in re.split(r"[._]", relation.lower()) if len(token) >= 5 and token not in semantic_stop]
        token_hit = any(token in evidence for token in tokens)
        if len(branches) >= gold["gold_independent_sources"] >= 2:
            status = "DETECTED"
        elif token_hit:
            status = "DETECTED"
        elif any(token in evidence for token in ("sequential", "different properties", "not independent", "single entity")):
            status = "MERGED_WITH_OTHER"
        elif router["structure"] in {"LINEAR", "UNCERTAIN"}:
            status = "OMITTED"
        else:
            status = "UNKNOWN"
        output.append({"constraint_id": item["constraint_id"], "relation": relation, "status": status})
    return output


def primary_error(gold: dict[str, Any], router: dict[str, Any], outcome: str) -> tuple[str, list[str]]:
    if outcome == "CORRECT_ROUTING_CONTROL":
        return "NONE", []
    evidence = router_evidence_text(router)
    secondary: list[str] = []
    if any(token in evidence for token in ("generic", "type-like")):
        primary = "TYPE_CONSTRAINT_CONFUSION"
    elif any(token in evidence for token in ("different semantic", "same semantic property", "semantic property required", "differently typed", "different properties", "incompatible")):
        primary = "ANSWER_TYPE_CONFLICT"
    elif any(token in evidence for token in ("sequential", "sequentially", "path rather than")):
        primary = "FALSE_SEQUENTIALIZATION"
    elif any(token in evidence for token in ("only one", "single identified", "single entity", "one linked", "not independent")):
        primary = "INDEPENDENT_SOURCE_MISSED"
    elif gold["gold_shared_intermediate"]:
        primary = "SHARED_INTERMEDIATE_MISSED"
    elif gold["gold_shared_answer"]:
        primary = "SHARED_ANSWER_MISSED"
    else:
        primary = "INSUFFICIENT_EVIDENCE"
    if primary != "FALSE_SEQUENTIALIZATION" and sequential_label(gold, router) == "FALSE_SEQUENTIALIZATION":
        secondary.append("FALSE_SEQUENTIALIZATION")
    if primary != "INDEPENDENT_SOURCE_MISSED" and independent_source_label(gold, router) == "INDEPENDENT_SOURCE_MISSED":
        secondary.append("INDEPENDENT_SOURCE_MISSED")
    if branch_type_label(gold, router) in {"FINAL_AS_LINEAR", "INTERMEDIATE_AS_LINEAR", "UNCERTAIN_INSTEAD_OF_BRANCH"}:
        if "BRANCH_TYPE_CONFUSION" not in secondary:
            secondary.append("BRANCH_TYPE_CONFUSION")
    if primary not in PRIMARY_ERRORS:
        primary = "OTHER"
    return primary, secondary


def routing_outcome(gold: dict[str, Any], execution_structure: str, answer_correct: bool) -> str:
    mismatch = gold["gold_structure"] != f"GOLD_{execution_structure}"
    if mismatch and not answer_correct:
        return "HARMFUL_ROUTING_MISMATCH"
    if mismatch and answer_correct:
        return "BENIGN_ROUTING_MISMATCH"
    if not mismatch and answer_correct:
        return "CORRECT_ROUTING_CONTROL"
    return "UNCLEAR"


def audit_case(spec: CaseSpec, cwq: dict[str, Any], grailqa: dict[str, Any], phase241: dict[str, dict[str, Any]]) -> dict[str, Any]:
    detail = read_json(ROOT / spec.detail_path)
    gold = gold_reference(spec.case_id, cwq, grailqa)
    router = load_router_record(detail)
    trace = phase241.get(spec.case_id)
    if trace:
        execution_structure = normalize_structure(trace.get("runtime_structure"))
        answer_correct = bool(trace.get("answer_correct"))
    else:
        execution_structure = normalize_structure(detail.get("router_structure"))
        answer_correct = bool(detail.get("hits_at_1"))
    outcome = routing_outcome(gold, execution_structure, answer_correct)
    primary, secondary = primary_error(gold, router, outcome)
    return {
        "case_id": spec.case_id,
        "dataset": gold["dataset"],
        "question": gold["question"],
        "source_experiment": spec.source_experiment,
        "reused_from_phase24": spec.reused_from_phase24,
        "gold_structure": gold["gold_structure"],
        "runtime_structure": execution_structure,
        "router_structure": router["structure"],
        "routing_outcome": outcome,
        "answer_correct": answer_correct,
        "gold_constraints": gold["gold_constraints"],
        "router_constraints": [
            {
                "branch_id": item.get("branch_id", "UNKNOWN_NOT_LOGGED"),
                "root_entity_id": item.get("root_entity_id", "UNKNOWN_NOT_LOGGED"),
                "root_entity_name": item.get("root_entity_name", "UNKNOWN_NOT_LOGGED"),
                "constraint_property": item.get("constraint_property", "UNKNOWN_NOT_LOGGED"),
                "constraint_summary": item.get("constraint_summary", "UNKNOWN_NOT_LOGGED"),
                "target_answer_type": item.get("target_answer_type", "UNKNOWN_NOT_LOGGED"),
            }
            for item in router["branches"]
        ],
        "router_constraint_evidence": router["reason"],
        "constraint_detection": constraint_detection(gold, router),
        "gold_source_anchors": gold["gold_source_anchors"],
        "independent_sources_gold": gold["gold_independent_sources"],
        "independent_sources_router": router_detected_source_count(router),
        "independent_source_detection": independent_source_label(gold, router),
        "shared_answer_gold": gold["gold_shared_answer"],
        "shared_answer_detection": shared_answer_label(gold, router),
        "shared_intermediate_gold": gold["gold_shared_intermediate"],
        "shared_intermediate_detection": shared_intermediate_label(gold, router),
        "sequential_vs_independent": sequential_label(gold, router),
        "branch_type_confusion": branch_type_label(gold, router),
        "uncertainty_analysis": uncertainty_label(gold, router),
        "type_constraint_analysis": type_constraint_label(gold, router),
        "answer_type_influence": answer_type_label(gold, router, outcome),
        "gold_dependency_type": gold["gold_dependency_type"],
        "gold_answer_variable": gold["gold_answer_variable"],
        "gold_explicit_conjunction": gold["gold_explicit_conjunction"],
        "gold_contains_cvt": gold["gold_contains_cvt"],
        "requested_final_answer_type": router["requested_final_answer_type"],
        "requested_final_semantic_property": router["requested_final_semantic_property"],
        "router_confidence": router["confidence"],
        "router_reason": router["reason"],
        "router_input": {
            "question": gold["question"],
            "runtime_topic_entities": router["runtime_topic_entities"],
            "prompt": router["prompt"],
        },
        "router_output": {
            "structure": router["structure"],
            "execution_mode": router["execution_mode"],
            "confidence": router["confidence"],
            "reason": router["reason"],
            "branches": router["branches"],
            "unsupported_structure": router["unsupported_structure"],
        },
        "validator_outcome": router["validator_outcome"],
        "primary_router_error": primary,
        "secondary_router_errors": secondary,
        "router_trace_available": bool(router["prompt"] != "UNKNOWN_NOT_LOGGED"),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "HARMFUL_ROUTING_MISMATCH": [row for row in rows if row["routing_outcome"] == "HARMFUL_ROUTING_MISMATCH"],
        "BENIGN_ROUTING_MISMATCH": [row for row in rows if row["routing_outcome"] == "BENIGN_ROUTING_MISMATCH"],
        "CORRECT_ROUTING_CONTROL": [row for row in rows if row["routing_outcome"] == "CORRECT_ROUTING_CONTROL"],
    }
    primary_counts = Counter(row["primary_router_error"] for row in rows if row["routing_outcome"] != "CORRECT_ROUTING_CONTROL")
    harmful_counts = Counter(row["primary_router_error"] for row in groups["HARMFUL_ROUTING_MISMATCH"])
    benign_counts = Counter(row["primary_router_error"] for row in groups["BENIGN_ROUTING_MISMATCH"])
    controls_branch_detected = sum(row["independent_source_detection"] == "INDEPENDENT_SOURCES_DETECTED" for row in groups["CORRECT_ROUTING_CONTROL"])
    return {
        "total_cases": len(rows),
        "fresh_case_count": sum(not row["reused_from_phase24"] for row in rows),
        "reused_case_count": sum(row["reused_from_phase24"] for row in rows),
        "groups": {key: {"count": len(value), "primary_errors": dict(Counter(row["primary_router_error"] for row in value))} for key, value in groups.items()},
        "primary_error_counts": dict(primary_counts),
        "harmful_primary_error_counts": dict(harmful_counts),
        "benign_primary_error_counts": dict(benign_counts),
        "shared_answer_detection_counts": dict(Counter(row["shared_answer_detection"] for row in rows)),
        "shared_intermediate_detection_counts": dict(Counter(row["shared_intermediate_detection"] for row in rows)),
        "independent_source_detection_counts": dict(Counter(row["independent_source_detection"] for row in rows)),
        "sequential_classification_counts": dict(Counter(row["sequential_vs_independent"] for row in rows)),
        "uncertainty_counts": dict(Counter(row["uncertainty_analysis"] for row in rows)),
        "correct_controls_with_independent_source_signal": controls_branch_detected,
    }


def choose_decision(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    harmful = summary["groups"]["HARMFUL_ROUTING_MISMATCH"]["count"]
    benign = summary["groups"]["BENIGN_ROUTING_MISMATCH"]["count"]
    if not rows or not harmful or not benign:
        return "INCONCLUSIVE"
    counts = Counter(summary["harmful_primary_error_counts"])
    if not counts:
        return "INCONCLUSIVE"
    top = counts.most_common()
    if top[0][1] >= max(3, (harmful + 1) // 2) and len(top) <= 2:
        return "CLEAR_ROUTING_ERROR_PATTERN"
    if top[0][1] >= 2:
        return "PARTIAL_ROUTING_ERROR_PATTERN"
    return "NO_CLEAR_ROUTING_ERROR_PATTERN"


def render_constraint_analysis(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Constraint detection analysis",
        "",
        "`router_constraints` contains only structured branch constraints emitted by the router. For LINEAR/UNCERTAIN outputs, the narrative reason is preserved separately and is not promoted to a structured constraint graph.",
        "",
        "| Case | Gold structure | Router structure | Gold constraints | Detected | Misstated/merged | Omitted |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        counts = Counter(item["status"] for item in row["constraint_detection"])
        lines.append(f"| `{row['case_id']}` | `{row['gold_structure']}` | `{row['router_structure']}` | {len(row['gold_constraints'])} | {counts['DETECTED']} | {counts['MISSTATED'] + counts['MERGED_WITH_OTHER']} | {counts['OMITTED']} |")
    lines += ["", "## Router narrative evidence", ""]
    for row in rows:
        lines.append(f"- `{row['case_id']}`: {row['router_constraint_evidence']}")
    return "\n".join(lines) + "\n"


def render_error_summary(rows: list[dict[str, Any]], summary: dict[str, Any], decision: str) -> str:
    lines = [
        "# Routing error summary",
        "",
        f"- Total cases: **{summary['total_cases']}**.",
        f"- Harmful mismatches: **{summary['groups']['HARMFUL_ROUTING_MISMATCH']['count']}**.",
        f"- Benign mismatches: **{summary['groups']['BENIGN_ROUTING_MISMATCH']['count']}**.",
        f"- Correct-routing controls: **{summary['groups']['CORRECT_ROUTING_CONTROL']['count']}**.",
        "",
        "| Primary router error | Harmful | Benign |",
        "|---|---:|---:|",
    ]
    keys = sorted(set(summary["harmful_primary_error_counts"]) | set(summary["benign_primary_error_counts"]))
    for key in keys:
        lines.append(f"| `{key}` | {summary['harmful_primary_error_counts'].get(key, 0)} | {summary['benign_primary_error_counts'].get(key, 0)} |")
    lines += [
        "",
        f"## Decision: `{decision}`",
        "",
        "The harmful group contains several recurring but non-identical router weaknesses. False sequentialization appears in both harmful and benign mismatches, while property-compatibility conflicts also appear in both. Independent-source misses and false sequentialization are tied as the largest harmful subgroups, but neither explains most harmful cases alone.",
        "",
    ]
    return "\n".join(lines)


def render_report(rows: list[dict[str, Any]], summary: dict[str, Any], decision: str) -> str:
    counts = summary["groups"]
    case_lines = [
        "| Case | Outcome | Gold | Router output | Runtime | Primary error | Secondary |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        case_lines.append(f"| `{row['case_id']}` | `{row['routing_outcome']}` | `{row['gold_structure']}` | `{row['router_structure']}` | `{row['runtime_structure']}` | `{row['primary_router_error']}` | `{', '.join(row['secondary_router_errors']) or 'none'}` |")
    return f"""# Phase 2.4.4 routing decision error analysis

## 1. Motivation from Phase 2.4.3

Phase 2.4.3 found that intermediate-binding loss explains only a minority of the audited failures. Phase 2.4.4 therefore returns to the router decision and asks which observable structural signal was missed or misunderstood before the wrong route was selected.

## 2. Research question

Why does the v1.1.3 router choose LINEAR or UNCERTAIN when the gold structure is branch-shaped, and which error mode distinguishes harmful from benign mismatches?

## 3. Frozen components

No router, router prompt, validator, structure label, branch construction, merge logic, fallback, relation scorer, candidate generation, entity grounding, Freebase backend, SPARQL execution, evaluator, model, decoding setting, or top-k was changed. No full100 or live backend run was performed.

## 4. Case selection

The audit contains **{len(rows)}** cases: **{counts['HARMFUL_ROUTING_MISMATCH']['count']}** harmful mismatches, **{counts['BENIGN_ROUTING_MISMATCH']['count']}** benign mismatches, and **{counts['CORRECT_ROUTING_CONTROL']['count']}** correct-routing controls. **{summary['fresh_case_count']}** cases are fresh relative to Phase 2.4/2.4.2 and **{summary['reused_case_count']}** are reused frozen traces. Reuse is explicit in `diagnostic_cases.json`.

## 5. Router input reconstruction

For each case the audit stores the exact recorded router prompt, runtime topic-entity JSON when available, router response, parsed structure, confidence, reason, requested answer type, requested semantic property, branch list, and validator/fallback outcome. Missing fields remain `UNKNOWN_NOT_LOGGED`.

## 6. Gold structure reference

Gold CWQ SPARQL and GrailQA S-expressions were parsed offline. The audit records gold relations, source anchors, independent-source count, shared answer/intermediate status, answer-variable type, conjunction, and dependency type. Gold structure is used only as the offline diagnostic reference.

## 7. Constraint detection analysis

{chr(10).join(case_lines)}

Structured router constraints are counted only when emitted in the router branch list. Narrative explanations remain evidence for diagnosing the decision but are not treated as a complete constraint graph.

## 8. Independent-source analysis

The router correctly exposed multiple independent sources in the four controls. It missed or collapsed independent source evidence in several harmful and benign mismatches. The exact per-case labels are in `router_input_output_analysis.jsonl`.

## 9. Shared-answer analysis

Correct controls explicitly represented two branches with a common answer type. Several wrong routes mentioned the answer type but did not construct branches; this is classified as missed or misinterpreted shared-answer structure rather than absence of an answer-type cue.

## 10. Shared-intermediate analysis

The four reused CWQ intermediate cases were described as sequential paths or were routed as LINEAR/UNCERTAIN. Their recorded router outputs did not expose a shared intermediate-variable graph. This supports a shared-intermediate detection weakness, but the sample is too small to isolate it from relation/structure errors.

## 11. Sequential-vs-independent confusion

False sequentialization occurs when the router explicitly describes a branch-shaped query as a sequential path or says that constraints with different surface properties cannot form branches. It appears in both harmful and benign mismatches, so it is not sufficient by itself to predict answer failure.

## 12. Branch-type confusion

The wrong-route cases are primarily `FINAL_AS_LINEAR`, `INTERMEDIATE_AS_LINEAR`, or `UNCERTAIN_INSTEAD_OF_BRANCH`. Correct controls use `BRANCH_FINAL` and expose branch roots and shared target types.

## 13. Uncertainty analysis

UNCERTAIN outputs were not uniform. Property-compatibility conflicts were labelled `CONFLICTING_SIGNALS`; the generic/type-like rejection in the Anti-Apartheid case was labelled `OVER_CONSERVATIVE_UNCERTAINTY`. These are separate observable failure modes.

## 14. Harmful-vs-benign comparison

{render_error_summary(rows, summary, decision)}

The same broad mistakes occur in both groups: false sequentialization, source/constraint collapse, and answer-property compatibility conflicts. Therefore a routing mismatch becomes harmful only when the omitted structure carries answer-disambiguating information.

## 15. Correct-routing controls

The four controls expose the recurring positive signal: multiple grounded entities, explicit independent constraints, a common requested answer type/property, and structured branch objects. One control also records validator fallback despite a correct BRANCH_FINAL prediction; routing correctness and downstream validator success are kept separate.

## 16. Primary router error distribution

See `routing_error_summary.md` and `feature_signal_summary.json`. Independent-source misses and false sequentialization are tied as the largest harmful subgroups in this sample; false sequentialization is distributed across harmful and benign mismatches.

## 17. Candidate router signals

The most useful observable candidates are: independent source count, explicit conjunction, shared answer type, shared intermediate dependency, and a sequential-vs-independent distinction before property-compatibility gating. CVT/mediator and answer-type cues remain secondary.

## 18. Main decision

**`{decision}`**

Several recurring routing weaknesses are present, but no single error explains most harmful mismatches. The evidence is therefore partial rather than a clear global router diagnosis.

## 19. Limitations

- The sample is 15 cases and is concentrated in previously diagnosed branch-2 examples.
- Router reasons are observable outputs, not hidden chain-of-thought.
- Independent-source counts are reconstructed from gold graph structure; the router does not log a comparable source-count field.
- Correct answer in benign cases does not prove that every gold constraint was executed.
- The analysis is sample-specific and does not establish global router behavior.

## 20. Recommended next intervention

The intervention gate is met for a narrow diagnostic direction: test an explicit independent-source/constraint extraction stage before structure classification. Three wrong-routing cases share an independent-source miss—two harmful and one benign—and all four controls expose the corresponding positive signal. This should be tested as a separate future experiment; it is not implemented here.
"""


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    cwq = {str(row["id"]): row for row in read_json(CWQ_PATH)}
    grailqa = {str(row["qid"]): row for row in read_json(GRAILQA_PATH)}
    phase241 = load_phase241()
    rows = [audit_case(spec, cwq, grailqa, phase241) for spec in CASE_SPECS]
    summary = aggregate(rows)
    decision = choose_decision(rows, summary)
    summary["decision"] = decision
    summary["selection"] = [spec.__dict__ for spec in CASE_SPECS]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _json_dump(OUTPUT_ROOT / "diagnostic_cases.json", {"cases": rows})
    _write_jsonl(OUTPUT_ROOT / "router_input_output_analysis.jsonl", rows)
    _json_dump(OUTPUT_ROOT / "feature_signal_summary.json", summary)
    (OUTPUT_ROOT / "constraint_detection_analysis.md").write_text(render_constraint_analysis(rows), encoding="utf-8")
    (OUTPUT_ROOT / "routing_error_summary.md").write_text(render_error_summary(rows, summary, decision), encoding="utf-8")
    report = render_report(rows, summary, decision)
    (OUTPUT_ROOT / "PHASE2_4_4_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
