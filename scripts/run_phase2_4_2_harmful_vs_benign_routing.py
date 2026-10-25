#!/usr/bin/env python3
"""Run the frozen Phase 2.4.2 harmful-vs-benign routing audit.

This is an offline structural audit.  It consumes the frozen Phase 2.4.1
traces and dataset annotations, does not call an LLM or Virtuoso, and does not
modify routing or execution behavior.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from relation_graph_kgqa.datasets.schemas import QAExample  # noqa: E402
from relation_graph_kgqa.logical_forms.sexpr_ast import Atom, ExprNode  # noqa: E402
from relation_graph_kgqa.logical_forms.sexpr_parser import parse_s_expression  # noqa: E402
from relation_graph_kgqa.sparql.graph_builder import build_query_graph  # noqa: E402
from relation_graph_kgqa.sparql.parser import parse_query  # noqa: E402
from run_phase2_4_mixed_failure_validation import read_json, read_jsonl  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs/experiments/phase2_4_2/harmful_vs_benign_routing"
PHASE241_ROOT = ROOT / "outputs/experiments/phase2_4_1/first_divergence_trace_audit"
PHASE21_DETAIL_ROOT = ROOT / "outputs/experiments/phase2_1/relation_grounding/details/adaptive"
FRESH_DETAIL_ROOT = ROOT / "outputs/experiments/phase1/adaptive_v1_1_3/grailqa_12_repeat/details/repeat_1/adaptive"
CWQ_PATH = ROOT / "datasets/cwq/cwq.json"
GRAILQA_PATH = ROOT / "datasets/grailqa/official/grailqa_v1.0_train.json"

CVT_MARKERS = (
    "romantic_relationship",
    "film.performance",
    "speech_or_presentation",
    "dated_percentage",
)


def normalize_relation(value: str) -> str:
    value = str(value or "").strip()
    if value.startswith("ns:"):
        value = value[3:]
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].rsplit("/", 1)[-1]
    return value


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


def _cwq_graph(row: dict[str, Any]):
    example = QAExample(
        id=str(row["id"]),
        question=str(row["question"]),
        answers=[],
        gold_sparql=str(row["sparql"]),
        dataset_type="cwq",
        topic_entities=[str(item.get("id", "")) for item in row.get("topic_entity", [])],
        split="",
    )
    features = parse_query(example.gold_sparql or "")
    return features, build_query_graph(features, example)


def gold_structure_features(case_id: str, cwq: dict[str, dict[str, Any]], grailqa: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Extract branch-necessity features from the gold dependency graph."""

    if case_id in cwq:
        row = cwq[case_id]
        features, graph = _cwq_graph(row)
        relations = [normalize_relation(edge.predicate) for edge in graph.edges]
        anchors = [
            {"id": str(item.get("id", "")), "name": str(item.get("name", ""))}
            for item in row.get("topic_entity", [])
        ]
        candidates = graph.branch_candidates()
        if any(variable in graph.answer_variables for variable in candidates):
            shared_type = "ANSWER"
        elif candidates:
            shared_type = "INTERMEDIATE"
        else:
            shared_type = "NONE"
        contains_cvt = any(any(marker in relation for marker in CVT_MARKERS) for relation in relations)
        return {
            "dataset": "cwq",
            "question": row["question"],
            "gold_answer": row.get("answer", []),
            "gold_structure": "GOLD_BRANCH_FINAL" if shared_type == "ANSWER" else "GOLD_BRANCH_INTERMEDIATE" if shared_type == "INTERMEDIATE" else "GOLD_LINEAR",
            "gold_form": row.get("sparql"),
            "relations": relations,
            "relation_directions": ["outgoing" if not edge.subject.startswith("?") or edge.object.startswith("?") else "incoming" for edge in graph.edges],
            "anchor_ids": anchors,
            "independent_anchor_count": len({item["id"] for item in anchors if item["id"]}),
            "shared_variable_type": shared_type,
            "constraint_count": len(graph.edges),
            "requires_new_binding_transition": bool(graph.variables - set(graph.answer_variables)),
            "contains_cvt": contains_cvt,
            "constraint_convergence": shared_type != "NONE",
            "branch_candidates": candidates,
            "variable_count": len(graph.variables),
        }

    row = grailqa[case_id]
    try:
        root = parse_s_expression(str(row.get("s_expression", "")))
        relations_with_direction: list[tuple[str, str]] = []
        _sexpr_relations(root, relations_with_direction)
    except Exception:
        relations_with_direction = []
        root = None
    query = str(row.get("sparql_query", ""))
    anchor_ids = sorted(set(re.findall(r"VALUES\s+\?\w+\s+\{\s*:([mg]\.[^}\s]+)", query)))
    body = root.arguments[1] if isinstance(root, ExprNode) and len(root.arguments) > 1 else None
    body_branch = isinstance(body, ExprNode) and body.operator.upper() == "AND"
    shared_type = "ANSWER" if body_branch else "INTERMEDIATE" if any(item[1] == "incoming" for item in relations_with_direction) and len(relations_with_direction) > 1 else "NONE"
    nested_join = False

    def has_join_inside_join(node: Any, under_join: bool = False) -> bool:
        nonlocal nested_join
        if not isinstance(node, ExprNode):
            return False
        current = under_join or node.operator.upper() == "JOIN"
        if under_join and node.operator.upper() == "JOIN":
            nested_join = True
        return any(has_join_inside_join(arg, current) for arg in node.arguments)

    has_join_inside_join(root)
    return {
        "dataset": "grailqa",
        "question": row["question"],
        "gold_answer": row.get("answer", []),
        "gold_structure": "GOLD_BRANCH_FINAL" if shared_type == "ANSWER" else "GOLD_BRANCH_INTERMEDIATE" if shared_type == "INTERMEDIATE" else "GOLD_LINEAR",
        "gold_form": row.get("s_expression"),
        "relations": [item[0] for item in relations_with_direction],
        "relation_directions": [item[1] for item in relations_with_direction],
        "anchor_ids": [{"id": item, "name": ""} for item in anchor_ids],
        "independent_anchor_count": len(anchor_ids),
        "shared_variable_type": shared_type,
        "constraint_count": len(relations_with_direction),
        "requires_new_binding_transition": nested_join,
        "contains_cvt": any(any(marker in relation for marker in CVT_MARKERS) for relation, _ in relations_with_direction),
        "constraint_convergence": shared_type != "NONE",
        "branch_candidates": {},
        "variable_count": 0,
    }


def extract_runtime_relations(detail: dict[str, Any]) -> list[str]:
    relations: list[str] = []
    for relation in detail.get("relation_chain") or []:
        if isinstance(relation, str):
            relations.append(normalize_relation(relation.lstrip("->")))
    for branch in detail.get("branch_traces") or []:
        for relation in branch.get("relation_chain") or []:
            if isinstance(relation, str):
                relations.append(normalize_relation(relation.lstrip("->")))
    for branch in (detail.get("adaptive_branch_result") or {}).get("branch_traces") or []:
        for relation in branch.get("relation_chain") or []:
            if isinstance(relation, str):
                relations.append(normalize_relation(relation.lstrip("->")))
    return relations


def load_runtime_detail(case_id: str, fresh: bool) -> dict[str, Any] | None:
    root = FRESH_DETAIL_ROOT if fresh else PHASE21_DETAIL_ROOT
    matches = list(root.glob(f"**/{case_id}.json"))
    if not matches:
        return None
    return read_json(matches[0])


def constraint_preservation(gold_relations: list[str], runtime_relations: list[str]) -> list[dict[str, Any]]:
    """Compare navigation constraints with multiplicity, preserving duplicates."""

    remaining = Counter(normalize_relation(item) for item in runtime_relations)
    result: list[dict[str, Any]] = []
    for index, relation in enumerate(gold_relations):
        normalized = normalize_relation(relation)
        if remaining[normalized] > 0:
            status = "PRESERVED"
            remaining[normalized] -= 1
        else:
            status = "LOST"
        result.append({"constraint_id": f"relation_{index}", "relation": normalized, "status": status})
    return result


def classify_routing_outcome(gold_structure: str, runtime_structure: str, answer_correct: bool | None) -> str:
    mismatch = gold_structure != f"GOLD_{runtime_structure}"
    if mismatch and answer_correct is False:
        return "HARMFUL_ROUTING_MISMATCH"
    if mismatch and answer_correct is True:
        return "BENIGN_ROUTING_MISMATCH"
    if not mismatch and answer_correct is True:
        return "CORRECT_ROUTING_CONTROL"
    return "UNCLEAR"


def classify_branch_requirement(features: dict[str, Any], preservation: list[dict[str, Any]], routing_outcome: str) -> str:
    lost = any(item["status"] == "LOST" for item in preservation)
    has_independent_join = features["independent_anchor_count"] >= 2 and features["constraint_convergence"]
    if has_independent_join and (lost or routing_outcome == "CORRECT_ROUTING_CONTROL"):
        return "BRANCH_REQUIRED"
    if routing_outcome == "BENIGN_ROUTING_MISMATCH" and not lost:
        return "SAFE_TO_LINEARIZE"
    return "UNCLEAR"


def classify_linearization(runtime_structure: str, preservation: list[dict[str, Any]]) -> str:
    if runtime_structure not in {"LINEAR", "UNCERTAIN"}:
        return "LINEARIZATION_UNCLEAR"
    if any(item["status"] == "LOST" for item in preservation):
        return "LINEARIZATION_INVALID"
    if all(item["status"] == "PRESERVED" for item in preservation):
        return "LINEARIZATION_VALID"
    return "LINEARIZATION_UNCLEAR"


def semantic_success_status(routing_outcome: str, preservation: list[dict[str, Any]], detail: dict[str, Any] | None) -> str:
    if routing_outcome == "HARMFUL_ROUTING_MISMATCH":
        return "NOT_APPLICABLE"
    if routing_outcome == "CORRECT_ROUTING_CONTROL":
        branch_merges = bool((detail or {}).get("merge_method") or (detail or {}).get("merge_outcome"))
        return "SEMANTICALLY_VALID_SUCCESS" if branch_merges else "UNKNOWN"
    if routing_outcome == "BENIGN_ROUTING_MISMATCH":
        if preservation and all(item["status"] == "PRESERVED" for item in preservation):
            return "SEMANTICALLY_VALID_SUCCESS"
        return "ANSWER_ONLY_SUCCESS"
    return "UNKNOWN"


def primary_loss_mechanism(features: dict[str, Any], preservation: list[dict[str, Any]], routing_outcome: str) -> str:
    if routing_outcome == "CORRECT_ROUTING_CONTROL":
        return "NONE_OBSERVED"
    if not any(item["status"] == "LOST" for item in preservation):
        return "UNKNOWN"
    if features["independent_anchor_count"] >= 2:
        return "INDEPENDENT_SOURCE_OMITTED"
    if features["requires_new_binding_transition"]:
        return "NEW_BINDING_NOT_PROPAGATED"
    return "CONSTRAINT_LOST"


def build_case_audit(
    trace: dict[str, Any],
    cwq: dict[str, dict[str, Any]],
    grailqa: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    case_id = str(trace["case_id"])
    features = gold_structure_features(case_id, cwq, grailqa)
    fresh = bool(trace.get("fresh_case"))
    detail = load_runtime_detail(case_id, fresh)
    runtime_relations = extract_runtime_relations(detail or {})
    preservation = constraint_preservation(features["relations"], runtime_relations)
    runtime_structure = str(trace["runtime_structure"])
    answer_correct = trace.get("answer_correct")
    outcome = classify_routing_outcome(features["gold_structure"], runtime_structure, answer_correct)
    linearization = classify_linearization(runtime_structure, preservation)
    requirement = classify_branch_requirement(features, preservation, outcome)
    success_status = semantic_success_status(outcome, preservation, detail)
    lost_relations = [item["relation"] for item in preservation if item["status"] == "LOST"]
    preserved_relations = [item["relation"] for item in preservation if item["status"] == "PRESERVED"]
    binding_loss = bool(features["requires_new_binding_transition"] and lost_relations)
    branch_merge_observed = bool((detail or {}).get("merge_method") or (detail or {}).get("merge_outcome"))
    return {
        "case_id": case_id,
        "dataset": features["dataset"],
        "question": features["question"],
        "gold_answer": features["gold_answer"],
        "gold_structure": features["gold_structure"],
        "runtime_structure": runtime_structure,
        "answer_correct": answer_correct,
        "routing_outcome": outcome,
        "shared_variable_type": features["shared_variable_type"],
        "independent_anchor_count": features["independent_anchor_count"],
        "independent_anchor_ids": features["anchor_ids"],
        "constraint_count": features["constraint_count"],
        "constraint_convergence": features["constraint_convergence"],
        "requires_new_binding_transition": features["requires_new_binding_transition"],
        "binding_dependency_lost": binding_loss,
        "contains_cvt": features["contains_cvt"],
        "gold_relations": features["relations"],
        "runtime_relations": runtime_relations,
        "constraint_preservation": preservation,
        "preserved_relation_count": len(preserved_relations),
        "lost_relation_count": len(lost_relations),
        "preserved_relations": preserved_relations,
        "lost_relations": lost_relations,
        "linearization_result": linearization,
        "branch_requirement": requirement,
        "semantic_success_validation": success_status,
        "primary_loss_mechanism": primary_loss_mechanism(features, preservation, outcome),
        "branch_merge_required": features["independent_anchor_count"] >= 2 and features["constraint_convergence"],
        "branch_merge_observed": branch_merge_observed,
        "runtime_detail_available": detail is not None,
        "fresh_case": fresh,
        "phase241_first_divergence": trace.get("first_divergence_stage"),
        "phase241_case_role": trace.get("case_role_in_phase2_4"),
        "evidence_limit": "Answer-only success; missing constraints were not independently verified" if success_status == "ANSWER_ONLY_SUCCESS" else "",
    }


def feature_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = ["HARMFUL_ROUTING_MISMATCH", "BENIGN_ROUTING_MISMATCH", "CORRECT_ROUTING_CONTROL"]
    signals = [
        "independent_anchor_count_ge_2",
        "shared_answer_variable",
        "shared_intermediate_variable",
        "requires_new_binding_transition",
        "contains_cvt",
        "constraint_loss",
        "binding_dependency_lost",
        "branch_merge_required",
    ]
    result: dict[str, Any] = {"groups": {}, "signals": signals}
    for group in groups:
        subset = [row for row in rows if row["routing_outcome"] == group]
        values: dict[str, Any] = {"case_count": len(subset)}
        for signal in signals:
            count = 0
            for row in subset:
                if signal == "independent_anchor_count_ge_2": value = row["independent_anchor_count"] >= 2
                elif signal == "shared_answer_variable": value = row["shared_variable_type"] == "ANSWER"
                elif signal == "shared_intermediate_variable": value = row["shared_variable_type"] == "INTERMEDIATE"
                elif signal == "constraint_loss": value = row["lost_relation_count"] > 0
                else: value = bool(row[signal])
                count += int(value)
            values[signal] = {"count": count, "rate": count / len(subset) if subset else 0.0}
        result["groups"][group] = values
    result["discriminative_signals"] = [
        signal
        for signal in signals
        if result["groups"]["HARMFUL_ROUTING_MISMATCH"][signal]["count"] > result["groups"]["BENIGN_ROUTING_MISMATCH"][signal]["count"]
        and result["groups"]["BENIGN_ROUTING_MISMATCH"][signal]["count"] == 0
    ]
    return result


def choose_main_decision(rows: list[dict[str, Any]], features: dict[str, Any]) -> str:
    if not rows:
        return "INCONCLUSIVE"
    harmful = features["groups"]["HARMFUL_ROUTING_MISMATCH"]
    benign = features["groups"]["BENIGN_ROUTING_MISMATCH"]
    if not harmful["case_count"] or not benign["case_count"]:
        return "INCONCLUSIVE"
    potential = [
        signal
        for signal in features["signals"]
        if harmful[signal]["rate"] >= 0.5 and harmful[signal]["rate"] > benign[signal]["rate"]
    ]
    if not potential:
        return "NO_CLEAR_BRANCH_NECESSITY_SIGNAL"
    clear = [
        signal
        for signal in potential
        if harmful[signal]["rate"] == 1.0 and benign[signal]["rate"] == 0.0
    ]
    return "CLEAR_BRANCH_NECESSITY_SIGNAL" if clear else "PARTIAL_BRANCH_NECESSITY_SIGNAL"


def render_constraint_preservation(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Constraint preservation audit",
        "",
        "`PRESERVED` means the relation appears in the recorded runtime navigation chain. `LOST` means it does not. Type/literal filters that were not represented as navigation relations remain outside this table; answer correctness does not upgrade them to preserved.",
        "",
        "| Case | Outcome | Gold navigation constraints | Preserved | Lost |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| `{row['case_id']}` | `{row['routing_outcome']}` | {len(row['gold_relations'])} | {row['preserved_relation_count']} | {row['lost_relation_count']} |")
    lines += ["", "## Per-case losses", ""]
    for row in rows:
        lines.append(f"- `{row['case_id']}`: lost `{', '.join(row['lost_relations']) or 'none'}`.")
    return "\n".join(lines) + "\n"


def render_linearization(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Linearization analysis",
        "",
        "The audit treats a linearization as semantically valid only when all essential navigation relations remain represented. A correct answer with lost constraints is recorded as `ANSWER_ONLY_SUCCESS`, not as proof of safe linearization.",
        "",
        "| Case | Gold structure | Runtime | Linearization | Branch requirement | Semantic success validation |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| `{row['case_id']}` | `{row['gold_structure']}` | `{row['runtime_structure']}` | `{row['linearization_result']}` | `{row['branch_requirement']}` | `{row['semantic_success_validation']}` |")
    lines += [
        "",
        "The benign mismatch cases returned correct answers despite recorded constraint loss. They are therefore benign routing outcomes, not validated safe-linearization examples.",
        "",
    ]
    return "\n".join(lines)


def render_branch_summary(rows: list[dict[str, Any]], features: dict[str, Any], decision: str) -> str:
    group_counts = Counter(row["routing_outcome"] for row in rows)
    requirement_counts = Counter(row["branch_requirement"] for row in rows)
    lines = [
        "# Branch-requirement summary",
        "",
        f"- Harmful routing mismatches: **{group_counts['HARMFUL_ROUTING_MISMATCH']}**.",
        f"- Benign routing mismatches: **{group_counts['BENIGN_ROUTING_MISMATCH']}**.",
        f"- Correct-routing controls: **{group_counts['CORRECT_ROUTING_CONTROL']}**.",
        f"- `SAFE_TO_LINEARIZE`: **{requirement_counts['SAFE_TO_LINEARIZE']}**.",
        f"- `BRANCH_REQUIRED`: **{requirement_counts['BRANCH_REQUIRED']}**.",
        f"- `UNCLEAR`: **{requirement_counts['UNCLEAR']}**.",
        "",
        "| Signal | Harmful | Benign | Correct-routing |",
        "|---|---:|---:|---:|",
    ]
    for signal in features["signals"]:
        values = []
        for group in ("HARMFUL_ROUTING_MISMATCH", "BENIGN_ROUTING_MISMATCH", "CORRECT_ROUTING_CONTROL"):
            values.append(f"{features['groups'][group][signal]['count']}/{features['groups'][group]['case_count']}")
        lines.append(f"| `{signal}` | {values[0]} | {values[1]} | {values[2]} |")
    if decision == "PARTIAL_BRANCH_NECESSITY_SIGNAL":
        interpretation = "Shared-intermediate, mediator, and new-binding signals are more common in harmful cases, but they have exceptions and do not fully separate fatality. Independent anchors and observed constraint loss occur in both groups."
    elif decision == "CLEAR_BRANCH_NECESSITY_SIGNAL":
        interpretation = "At least one structural signal consistently separates the harmful and benign groups in this audit."
    else:
        interpretation = "The harmful and benign groups remain structurally indistinguishable under the tested signals."
    lines += [
        "",
        f"## Decision: `{decision}`",
        "",
        interpretation,
        "",
    ]
    return "\n".join(lines) + "\n"


def render_report(rows: list[dict[str, Any]], features: dict[str, Any], decision: str) -> str:
    outcome_counts = Counter(row["routing_outcome"] for row in rows)
    requirement_counts = Counter(row["branch_requirement"] for row in rows)
    case_table = [
        "| Case | Routing outcome | Gold/runtime | Anchors | Shared variable | Constraint loss | Linearization | Branch requirement |",
        "|---|---|---|---:|---|---:|---|---|",
    ]
    for row in rows:
        case_table.append(
            f"| `{row['case_id']}` | `{row['routing_outcome']}` | `{row['gold_structure']}` → `{row['runtime_structure']}` | {row['independent_anchor_count']} | `{row['shared_variable_type']}` | {row['lost_relation_count']} | `{row['linearization_result']}` | `{row['branch_requirement']}` |"
        )
    return f"""# Phase 2.4.2 harmful vs benign routing mismatch analysis

## 1. Motivation from Phase 2.4.1

Phase 2.4.1 found routing as the first divergence for all seven failed-answer cases, but also observed successful gold-branch → LINEAR cases. Phase 2.4.2 asks whether those successful mismatches were genuinely safe linearizations or merely answer-only successes.

## 2. Research question

What distinguishes a routing mismatch that causes answer failure from one that still returns the correct answer?

## 3. Why routing mismatch is not always fatal

A structure mismatch is not itself an answer failure. A linear path may happen to reach the same answer as one branch, or may find a unique candidate before the omitted constraint is needed. The audit therefore separates routing outcome from semantic branch requirement.

## 4. Frozen components

No router, structure label, branch merge, fallback, relation scorer, candidate generation, Freebase backend, SPARQL execution, evaluator, model, prompt, top-k, or inference setting was changed. No new full100 run was performed.

## 5. Case selection

The frozen Phase 2.4.1 set was reused: **{len(rows)}** cases consisting of **{outcome_counts['HARMFUL_ROUTING_MISMATCH']}** harmful mismatches, **{outcome_counts['BENIGN_ROUTING_MISMATCH']}** benign mismatches, and **{outcome_counts['CORRECT_ROUTING_CONTROL']}** correct-routing controls. Freshness and provenance remain those recorded by Phase 2.4.1.

## 6. Gold dependency reconstruction

CWQ SPARQL graphs and GrailQA S-expressions were parsed into relations, directions, entity anchors, shared variables, and convergence points. Independent anchors were counted from CWQ topic entities and GrailQA `VALUES` anchors. CVT/mediator flags were conservative relation-name indicators, not proof of a special backend node.

## 7. Harmful routing mismatches

The seven failed-answer cases all lost at least one essential gold navigation relation in the recorded linear/uncertain chain. The losses include omitted independent-source constraints, missing newly bound transitions, and dropped answer filters. Their primary structural mechanisms are recorded in `routing_mismatch_analysis.jsonl`.

## 8. Benign routing mismatches

The four successful mismatch cases also show recorded constraint loss. Their correct answers therefore cannot be treated as evidence that all gold constraints were executed. They are classified as `ANSWER_ONLY_SUCCESS` unless the frozen trace preserves every navigation constraint.

## 9. Constraint preservation

See `constraint_preservation.md`. Correct answer equality does not convert an unlogged or omitted constraint into `PRESERVED`.

## 10. Linearization validity

See `linearization_analysis.md`. The audit labels current linear traces invalid when an essential navigation relation is absent, even if the final answer is correct.

## 11. Shared-variable analysis

The cases converge on answer or intermediate variables. The same convergence pattern is present in harmful and benign mismatch groups; it does not separate fatal from benign outcomes.

## 12. Newly-bound-variable dependency

New-binding dependencies occur in both groups. They are candidate branch-necessity signals, but the sample does not show them as a fatality separator by themselves.

## 13. Independent-source analysis

Independent anchors are common in both groups. In the harmful group they are often omitted by the linear trace; in the benign group the answer is still found, but the second source constraint is not independently evidenced.

## 14. CVT / mediator effects

Mediator-like relation families occur in several harmful cases and do not explain all benign cases. They should remain a future feature, not a production rule.

## 15. Harmful-vs-benign comparison

{chr(10).join(case_table)}

## 16. Candidate branch-necessity signals

{render_branch_summary(rows, features, decision)}

## 17. Main decision

**`{decision}`**

{("Some signals are promising but incomplete: harmful cases more often involve shared intermediates, mediator-like relations, and binding dependency loss. However, independent anchors and constraint loss also occur in benign cases, and benign answers remain answer-only successes rather than validated safe linearizations." if decision == "PARTIAL_BRANCH_NECESSITY_SIGNAL" else "No tested signal cleanly separates harmful from benign routing mismatches. Benign answers are not validated safe linearizations; they are successful outcomes with incomplete trace-level constraint preservation." if decision == "NO_CLEAR_BRANCH_NECESSITY_SIGNAL" else "The available evidence supports a consistent structural separator in this sample.")}

## 18. Limitations

- The sample is only 13 cases and is concentrated in branch-2 examples.
- Several successful answers lack independent local-KG verification of every gold constraint.
- Constraint preservation is based on recorded navigation relations; unlogged type/literal filters remain unknown.
- This audit does not prove that every branch-shaped query requires runtime branching.

## 19. Recommended future router objective

If a future router feature is tested, predict whether the proposed linear execution would lose an essential constraint or variable dependency—not merely whether the gold logical form contains a branch. The next experiment should explicitly log constraint preservation and distinguish semantically validated success from answer-only success.

No such intervention is implemented here.
"""


def main() -> None:
    cwq = {str(row["id"]): row for row in read_json(CWQ_PATH)}
    grailqa = {str(row["qid"]): row for row in read_json(GRAILQA_PATH)}
    phase241_rows = read_jsonl(PHASE241_ROOT / "runtime_trace_comparison.jsonl")
    audits = [build_case_audit(row, cwq, grailqa) for row in phase241_rows]
    features = feature_summary(audits)
    decision = choose_main_decision(audits, features)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _json_dump(OUTPUT_ROOT / "diagnostic_cases.json", {"cases": audits})
    _write_jsonl(OUTPUT_ROOT / "routing_mismatch_analysis.jsonl", audits)
    _json_dump(OUTPUT_ROOT / "feature_signal_summary.json", features)
    (OUTPUT_ROOT / "constraint_preservation.md").write_text(render_constraint_preservation(audits), encoding="utf-8")
    (OUTPUT_ROOT / "linearization_analysis.md").write_text(render_linearization(audits), encoding="utf-8")
    (OUTPUT_ROOT / "branch_requirement_summary.md").write_text(render_branch_summary(audits, features, decision), encoding="utf-8")
    report = render_report(audits, features, decision)
    (OUTPUT_ROOT / "PHASE2_4_2_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
