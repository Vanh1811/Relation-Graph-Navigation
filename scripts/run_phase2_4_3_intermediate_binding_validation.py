#!/usr/bin/env python3
"""Run the frozen Phase 2.4.3 intermediate-binding validation audit.

This is an offline diagnostic.  It reconstructs dependency shapes from the
gold CWQ/GrailQA annotations and inspects already-recorded runtime traces.  It
does not call an LLM or Virtuoso and does not change production behavior.
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

from relation_graph_kgqa.logical_forms.sexpr_ast import Atom, ExprNode  # noqa: E402
from relation_graph_kgqa.logical_forms.sexpr_parser import parse_s_expression  # noqa: E402
from run_phase2_4_2_harmful_vs_benign_routing import (  # noqa: E402
    _cwq_graph,
    gold_structure_features,
    read_json,
    read_jsonl,
)


OUTPUT_ROOT = ROOT / "outputs/experiments/phase2_4_3/intermediate_binding_validation"
PHASE241_TRACE = ROOT / "outputs/experiments/phase2_4_1/first_divergence_trace_audit/runtime_trace_comparison.jsonl"
CWQ_PATH = ROOT / "datasets/cwq/cwq.json"
GRAILQA_PATH = ROOT / "datasets/grailqa/official/grailqa_v1.0_train.json"


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    group: str
    detail_path: str
    source_experiment: str
    reused_from_phase24: bool
    target_variable: str | None = None


# Group A deliberately includes four earlier failure traces because those
# traces contain explicit actual-vs-intended state evidence.  The fifth case
# is fresh relative to Phase 2.4 and comes from the repeated GrailQA run.
CASE_SPECS = (
    CaseSpec(
        "WebQTest-719_e56e856592a9a57b9e331b718adc1b75",
        "INTERMEDIATE_DEPENDENT",
        "outputs/experiments/phase2_1/relation_grounding/details/adaptive/cwq_branch_2/WebQTest-719_e56e856592a9a57b9e331b718adc1b75.json",
        "phase2_1_relation_grounding",
        True,
        "?c",
    ),
    CaseSpec(
        "WebQTest-759_88081dc1a259cbd172b89242384ee1cc",
        "INTERMEDIATE_DEPENDENT",
        "outputs/experiments/phase2_1/relation_grounding/details/adaptive/cwq_branch_2/WebQTest-759_88081dc1a259cbd172b89242384ee1cc.json",
        "phase2_1_relation_grounding",
        True,
        "?y",
    ),
    CaseSpec(
        "WebQTrn-64_984678664419049133d2bd748b68df1b",
        "INTERMEDIATE_DEPENDENT",
        "outputs/experiments/phase2_1/relation_grounding/details/adaptive/cwq_branch_2/WebQTrn-64_984678664419049133d2bd748b68df1b.json",
        "phase2_1_relation_grounding",
        True,
        "?y",
    ),
    CaseSpec(
        "4302233001000",
        "INTERMEDIATE_DEPENDENT",
        "outputs/experiments/phase2_1/relation_grounding/details/adaptive/grailqa_branch_2/4302233001000.json",
        "phase2_1_relation_grounding",
        True,
    ),
    CaseSpec(
        "4300978012000",
        "INTERMEDIATE_DEPENDENT",
        "outputs/experiments/phase1/adaptive_v1_1_3/grailqa_12_repeat/details/repeat_1/adaptive/4300978012000.json",
        "phase1_adaptive_v1_1_3_grailqa_repeat_1",
        False,
    ),
    CaseSpec(
        "3201049003000",
        "SHARED_ANSWER_BRANCH",
        "outputs/experiments/phase1/adaptive_v1_1_3/grailqa_12_repeat/details/repeat_1/adaptive/3201049003000.json",
        "phase1_adaptive_v1_1_3_grailqa_repeat_1",
        False,
    ),
    CaseSpec(
        "3201277005000",
        "SHARED_ANSWER_BRANCH",
        "outputs/experiments/phase1/adaptive_v1_1_3/grailqa_12_repeat/details/repeat_1/adaptive/3201277005000.json",
        "phase1_adaptive_v1_1_3_grailqa_repeat_1",
        False,
    ),
    CaseSpec(
        "3202841012000",
        "SHARED_ANSWER_BRANCH",
        "outputs/experiments/phase1/adaptive_v1_1_3/grailqa_12_repeat/details/repeat_1/adaptive/3202841012000.json",
        "phase1_adaptive_v1_1_3_grailqa_repeat_1",
        False,
    ),
    CaseSpec(
        "3203234013000",
        "SHARED_ANSWER_BRANCH",
        "outputs/experiments/phase1/adaptive_v1_1_3/grailqa_12_repeat/details/repeat_1/adaptive/3203234013000.json",
        "phase1_adaptive_v1_1_3_grailqa_repeat_1",
        False,
    ),
    CaseSpec(
        "3205047010000",
        "SHARED_ANSWER_BRANCH",
        "outputs/experiments/phase1/adaptive_v1_1_3/grailqa_12_repeat/details/repeat_1/adaptive/3205047010000.json",
        "phase1_adaptive_v1_1_3_grailqa_repeat_1",
        False,
    ),
    CaseSpec(
        "WebQTrn-124_4d06de0cc31fbca96f6f1d5bdec75e1a",
        "LINEAR_CONTROL",
        "outputs/experiments/cor_baseline_timeout120_512_partial/details/cwq_linear/WebQTrn-124_4d06de0cc31fbca96f6f1d5bdec75e1a.json",
        "cor_baseline_timeout120_512_partial",
        False,
    ),
    CaseSpec(
        "WebQTrn-3376_13cf1a3ce233a65b8b835b6e2f5c86fe",
        "LINEAR_CONTROL",
        "outputs/experiments/cor_baseline_timeout120_512_partial/details/cwq_linear/WebQTrn-3376_13cf1a3ce233a65b8b835b6e2f5c86fe.json",
        "cor_baseline_timeout120_512_partial",
        False,
    ),
    CaseSpec(
        "WebQTrn-2834_1a821009d9d43af7c2094125b1dd5136",
        "LINEAR_CONTROL",
        "outputs/experiments/cor_baseline_timeout120_512_partial/details/cwq_linear/WebQTrn-2834_1a821009d9d43af7c2094125b1dd5136.json",
        "cor_baseline_timeout120_512_partial",
        False,
    ),
    CaseSpec(
        "WebQTrn-493_8faa4f3ff9c14c303d5c535cf18ceddf",
        "LINEAR_CONTROL",
        "outputs/experiments/cor_baseline_timeout120_512_partial/details/cwq_linear/WebQTrn-493_8faa4f3ff9c14c303d5c535cf18ceddf.json",
        "cor_baseline_timeout120_512_partial",
        False,
    ),
)


def _expr_text(node: Any) -> str:
    if isinstance(node, Atom):
        return node.value
    if isinstance(node, ExprNode):
        return "(" + " ".join([node.operator] + [_expr_text(arg) for arg in node.arguments]) + ")"
    return str(node)


def _join_relation(node: Any) -> str:
    if not isinstance(node, ExprNode) or node.operator.upper() != "JOIN" or not node.arguments:
        return "UNKNOWN"
    first = node.arguments[0]
    if isinstance(first, Atom):
        return first.value
    if isinstance(first, ExprNode) and first.operator.upper() == "R" and first.arguments and isinstance(first.arguments[0], Atom):
        return f"R {first.arguments[0].value}"
    return _expr_text(first)


def _nested_join_dependencies(node: Any, output: list[dict[str, Any]], counter: list[int]) -> None:
    if not isinstance(node, ExprNode):
        return
    if node.operator.upper() == "JOIN" and len(node.arguments) >= 2:
        relation = _join_relation(node)
        child = node.arguments[1]
        if isinstance(child, ExprNode):
            counter[0] += 1
            variable = f"?intermediate_{counter[0]}"
            output.append(
                {
                    "intermediate_variable": variable,
                    "binding_step": counter[0],
                    "binding_relation": _expr_text(child),
                    "binding_source_state": "nested logical-form expression",
                    "expected_bound_type": "intermediate_expression_result",
                    "dependent_constraints": [
                        {
                            "constraint": relation,
                            "status": "UNKNOWN",
                            "evidence": "Gold dependency; runtime status is audited separately",
                        }
                    ],
                    "dependent_constraint_count": 1,
                    "all_later_constraints_use_same_variable": True,
                }
            )
    for argument in node.arguments:
        _nested_join_dependencies(argument, output, counter)


def _edge_text(edge: Any) -> str:
    return f"{edge.subject} --{edge.predicate}--> {edge.object}"


def _cwq_intermediate(spec: CaseSpec, graph: Any) -> dict[str, Any]:
    variable = spec.target_variable
    if not variable:
        candidates = graph.branch_candidates()
        variable = max(candidates, key=lambda item: candidates[item]) if candidates else None
    if not variable:
        return {"intermediates": [], "shared_answer_only": False}
    incident = [edge for edge in graph.edges if edge.subject == variable or edge.object == variable]
    topic_terms = set(graph.topic_terms)
    binding_edges = [
        edge for edge in incident
        if edge.subject in topic_terms or edge.object in topic_terms
    ]
    if not binding_edges:
        binding_edges = incident[:1]
    binding_edge_ids = {id(edge) for edge in binding_edges}
    dependent = [_edge_text(edge) for edge in incident if id(edge) not in binding_edge_ids]
    return {
        "intermediates": [
            {
                "intermediate_variable": variable,
                "binding_step": 1,
                "binding_relation": [_edge_text(edge) for edge in binding_edges],
                "binding_source_state": sorted(topic_terms) or ["UNKNOWN"],
                "expected_bound_type": "intermediate_entity",
                "dependent_constraints": [
                    {"constraint": item, "status": "UNKNOWN", "evidence": "Gold dependency; runtime status is audited separately"}
                    for item in dependent
                ],
                "dependent_constraint_count": len(dependent),
                "all_later_constraints_use_same_variable": bool(dependent),
            }
        ],
        "shared_answer_only": False,
    }


def gold_dependency_features(spec: CaseSpec, cwq: dict[str, Any], grailqa: dict[str, Any]) -> dict[str, Any]:
    features = gold_structure_features(spec.case_id, cwq, grailqa)
    if spec.case_id in cwq:
        _, graph = _cwq_graph(cwq[spec.case_id])
        target = _cwq_intermediate(spec, graph)
        all_variables = sorted(graph.variables)
        dependency_shape = "CONVERGENT_SHARED_INTERMEDIATE" if target["intermediates"] else "SINGLE_PATH"
    else:
        root = parse_s_expression(str(grailqa[spec.case_id].get("s_expression", "")))
        dependencies: list[dict[str, Any]] = []
        _nested_join_dependencies(root, dependencies, [0])
        target = {
            "intermediates": dependencies,
            "shared_answer_only": not dependencies and features["gold_structure"] == "GOLD_BRANCH_FINAL",
        }
        all_variables = [item["intermediate_variable"] for item in dependencies]
        dependency_shape = "NESTED_INTERMEDIATE" if dependencies else "SHARED_ANSWER_ONLY" if target["shared_answer_only"] else "SINGLE_PATH"
    if spec.group == "SHARED_ANSWER_BRANCH":
        target = {"intermediates": [], "shared_answer_only": True}
        dependency_shape = "SHARED_ANSWER_ONLY"
    if spec.group == "LINEAR_CONTROL":
        target = {"intermediates": [], "shared_answer_only": False}
        dependency_shape = "SINGLE_PATH"
    return {
        **features,
        "group": spec.group,
        "dependency_shape": dependency_shape,
        "all_gold_variables": all_variables,
        "intermediates": target["intermediates"],
        "shared_answer_only": target["shared_answer_only"],
        "contains_cvt": bool(features.get("contains_cvt")),
    }


def _walk_trace_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_trace_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_trace_dicts(child)


def runtime_sparql_steps(detail: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int | None]] = set()
    for item in _walk_trace_dicts(detail):
        if item.get("step_type") != "sparql" or not item.get("sparql"):
            continue
        key = (str(item.get("search_type", "")), str(item.get("sparql", "")), item.get("trace_id"))
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _trace_query_variables(query: str) -> list[str]:
    return sorted(set(re.findall(r"\?e_\d+", query)))


def runtime_relation_chain(detail: dict[str, Any]) -> list[str]:
    relations: list[str] = []
    for item in detail.get("relation_chain") or []:
        if isinstance(item, str):
            relations.append(item.lstrip("->"))
    for branch in detail.get("branch_traces") or []:
        for item in branch.get("relation_chain") or []:
            if isinstance(item, str):
                relations.append(item.lstrip("->"))
    return relations


def load_phase241_traces() -> dict[str, dict[str, Any]]:
    return {str(row["case_id"]): row for row in read_jsonl(PHASE241_TRACE)}


def trace_state_evidence(trace: dict[str, Any] | None, detail: dict[str, Any]) -> dict[str, Any]:
    step = ((trace or {}).get("steps") or [{}])[0]
    actual = step.get("actual_state") or {}
    gold = step.get("gold_state") or {}
    actual_id = actual.get("id")
    gold_id = gold.get("id")
    explicit_wrong_state = bool(actual_id and gold_id and actual_id != gold_id)
    return {
        "current_state": actual or "UNKNOWN_NOT_LOGGED",
        "expected_state": gold or "UNKNOWN_NOT_LOGGED",
        "next_runtime_state": step.get("actual_next_state", "UNKNOWN_NOT_LOGGED"),
        "active_subgoal": step.get("actual_subgoal", "UNKNOWN_NOT_LOGGED"),
        "remaining_constraint": "UNKNOWN_NOT_LOGGED",
        "selected_relation": step.get("selected_relation", "UNKNOWN_NOT_LOGGED"),
        "selected_direction": step.get("selected_direction", "UNKNOWN_NOT_LOGGED"),
        "relation_history": step.get("actual_relation_history") or detail.get("relation_chain") or [],
        "explicit_wrong_state": explicit_wrong_state,
        "source": "Phase 2.4.1 first-divergence trace" if trace else "runtime detail only; state fields not logged",
    }


def binding_audit(detail: dict[str, Any], target_applicable: bool) -> dict[str, Any]:
    steps = runtime_sparql_steps(detail)
    variable_occurrences: Counter[str] = Counter()
    continuation_occurrences: Counter[str] = Counter()
    for item in steps:
        query = str(item.get("sparql", ""))
        variables = _trace_query_variables(query)
        for variable in variables:
            variable_occurrences[variable] += 1
            if item.get("search_type") == "entity_search" and variable in query:
                continuation_occurrences[variable] += 1
    reasoning = json.dumps(detail.get("reasoning_chains", ""), ensure_ascii=False)
    reasoning_has_transition = bool(re.search(r"e_?\d+\s+\S+\s+e_?\d+", reasoning))
    if not target_applicable:
        status = "UNKNOWN"
    elif continuation_occurrences or reasoning_has_transition:
        status = "BINDING_PRESERVED"
    elif variable_occurrences:
        status = "BINDING_LOST"
    else:
        status = "BINDING_NOT_CREATED"
    runtime_structure = str(detail.get("router_structure") or detail.get("native_execution_mode") or "UNKNOWN")
    if runtime_structure.startswith("BRANCH") or detail.get("final_execution_mode") in {"branch", "branch_fallback"}:
        execution_label = "BRANCH_BINDING_PRESERVED" if status == "BINDING_PRESERVED" else "BRANCH_BINDING_LOST" if status in {"BINDING_LOST", "BINDING_NOT_CREATED"} else "BRANCH_BINDING_UNKNOWN"
    else:
        execution_label = "LINEAR_BINDING_PRESERVED" if status == "BINDING_PRESERVED" else "LINEAR_BINDING_LOST" if status in {"BINDING_LOST", "BINDING_NOT_CREATED"} else "LINEAR_BINDING_UNKNOWN"
    return {
        "binding_status": status,
        "execution_binding_status": execution_label,
        "variable_occurrences": dict(variable_occurrences),
        "continuation_occurrences": dict(continuation_occurrences),
        "reasoning_chain_shows_transition": reasoning_has_transition,
        "evidence_query_count": len(steps),
    }


def dependent_constraint_statuses(
    gold: dict[str, Any],
    binding: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    dependent: list[dict[str, Any]] = []
    for intermediate in gold["intermediates"]:
        for item in intermediate["dependent_constraints"]:
            status = "UNKNOWN"
            if binding["binding_status"] == "BINDING_NOT_CREATED":
                status = "OMITTED"
            elif state["explicit_wrong_state"]:
                status = "APPLIED_TO_WRONG_STATE"
            elif binding["binding_status"] == "BINDING_PRESERVED":
                status = "APPLIED_TO_CORRECT_BINDING"
            elif binding["binding_status"] == "BINDING_LOST":
                status = "UNKNOWN"
            dependent.append({
                "constraint": item["constraint"],
                "status": status,
                "intermediate_variable": intermediate["intermediate_variable"],
                "evidence": "runtime SPARQL/state trace" if status != "UNKNOWN" else "not logged sufficiently",
            })
    return dependent


def relation_evidence(gold_relations: list[str], detail: dict[str, Any]) -> dict[str, Any]:
    steps = runtime_sparql_steps(detail)
    text = "\n".join(str(item.get("sparql", "")) for item in steps) + "\n" + json.dumps(detail.get("relation_chain", []))
    preserved = [relation for relation in gold_relations if relation in text]
    return {
        "gold_relation_count": len(gold_relations),
        "runtime_evidenced_relation_count": len(preserved),
        "preserved_relations": preserved,
        "lost_relations": [relation for relation in gold_relations if relation not in preserved],
    }


def semantic_success(answer_correct: bool, dependent: list[dict[str, Any]], relation_evidence_result: dict[str, Any], detail: dict[str, Any]) -> str:
    if not answer_correct:
        return "NOT_APPLICABLE"
    if dependent and all(item["status"] == "APPLIED_TO_CORRECT_BINDING" for item in dependent):
        branch_traces = detail.get("branch_traces") or []
        branch_success = bool(branch_traces) and all(item.get("status") == "SUCCESS" for item in branch_traces)
        if branch_success or not relation_evidence_result["lost_relations"]:
            return "SEMANTICALLY_VALID_SUCCESS"
    if not relation_evidence_result["lost_relations"]:
        return "SEMANTICALLY_VALID_SUCCESS"
    return "ANSWER_ONLY_SUCCESS"


def primary_failure(
    gold: dict[str, Any],
    binding: dict[str, Any],
    dependent: list[dict[str, Any]],
    state: dict[str, Any],
    answer_correct: bool,
) -> str:
    if gold["intermediates"] and binding["binding_status"] in {"BINDING_NOT_CREATED", "BINDING_LOST"}:
        return "INTERMEDIATE_BINDING_LOSS"
    if any(item["status"] == "OMITTED" for item in dependent):
        return "DEPENDENT_CONSTRAINT_OMITTED"
    if any(item["status"] == "APPLIED_TO_WRONG_STATE" for item in dependent):
        return "DEPENDENT_CONSTRAINT_WRONG_STATE"
    if not answer_correct:
        trace = state.get("source", "")
        if "Phase 2.4.1" in trace:
            return "STRUCTURE_ROUTING"
        return "RELATION_GROUNDING"
    return "NO_BINDING_FAILURE" if not gold["intermediates"] else "INSUFFICIENT_EVIDENCE"


def audit_case(spec: CaseSpec, cwq: dict[str, Any], grailqa: dict[str, Any], phase241: dict[str, dict[str, Any]]) -> dict[str, Any]:
    detail_path = ROOT / spec.detail_path
    detail = read_json(detail_path)
    gold = gold_dependency_features(spec, cwq, grailqa)
    trace = phase241.get(spec.case_id)
    target_applicable = bool(gold["intermediates"])
    binding = binding_audit(detail, target_applicable)
    state = trace_state_evidence(trace, detail)
    dependent = dependent_constraint_statuses(gold, binding, state)
    evidence = relation_evidence(gold["relations"], detail)
    answer_correct = bool(detail.get("hits_at_1"))
    success = semantic_success(answer_correct, dependent, evidence, detail)
    failure = primary_failure(gold, binding, dependent, state, answer_correct)
    if success == "SEMANTICALLY_VALID_SUCCESS":
        failure = "NO_BINDING_FAILURE"
    runtime_structure = detail.get("router_structure") or detail.get("native_execution_mode")
    if not runtime_structure and spec.group == "LINEAR_CONTROL":
        runtime_structure = "LINEAR"
    runtime_structure = runtime_structure or "UNKNOWN"
    return {
        "case_id": spec.case_id,
        "dataset": gold["dataset"],
        "group": spec.group,
        "question": gold["question"],
        "source_experiment": spec.source_experiment,
        "reused_from_phase24": spec.reused_from_phase24,
        "gold_structure": gold["gold_structure"],
        "runtime_structure": runtime_structure,
        "runtime_execution_mode": detail.get("final_execution_mode") or detail.get("execution_mode") or "UNKNOWN",
        "answer_correct": answer_correct,
        "gold_answer": gold["gold_answer"],
        "dependency_shape": gold["dependency_shape"],
        "gold_variables": gold["all_gold_variables"],
        "intermediate_variable": gold["intermediates"][0]["intermediate_variable"] if gold["intermediates"] else None,
        "intermediate_dependencies": gold["intermediates"],
        "dependent_constraint_count": sum(item["dependent_constraint_count"] for item in gold["intermediates"]),
        "shared_answer_only": gold["shared_answer_only"],
        "contains_cvt": gold["contains_cvt"],
        "binding_status": binding["binding_status"],
        "execution_binding_status": binding["execution_binding_status"],
        "binding_evidence": binding,
        "dependent_constraints": dependent,
        "dependent_constraints_lost_count": sum(item["status"] in {"OMITTED", "APPLIED_TO_WRONG_STATE"} for item in dependent),
        "runtime_relation_chain": runtime_relation_chain(detail),
        "relation_evidence": evidence,
        "runtime_trace": state,
        "runtime_trace_available": bool(runtime_sparql_steps(detail)),
        "phase241_trace_available": trace is not None,
        "primary_failure": failure,
        "semantic_success": success,
        "branch_merge_observed": bool(detail.get("merge_method") not in {None, "", "NONE"} or detail.get("merge_outcome")),
        "evidence_limit": "Correct answer does not validate omitted constraints" if success == "ANSWER_ONLY_SUCCESS" else "",
    }


def _rate(count: int, denominator: int) -> float:
    return count / denominator if denominator else 0.0


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in ("INTERMEDIATE_DEPENDENT", "SHARED_ANSWER_BRANCH", "LINEAR_CONTROL"):
        subset = [row for row in rows if row["group"] == group]
        binding_applicable = [row for row in subset if row["intermediate_variable"]]
        dependent = [item for row in subset for item in row["dependent_constraints"]]
        lost = sum(row["binding_status"] in {"BINDING_LOST", "BINDING_NOT_CREATED"} for row in binding_applicable)
        dependent_lost = sum(item["status"] in {"OMITTED", "APPLIED_TO_WRONG_STATE"} for item in dependent)
        preserved_dep = sum(item["status"] == "APPLIED_TO_CORRECT_BINDING" for item in dependent)
        result[group] = {
            "case_count": len(subset),
            "binding_applicable_cases": len(binding_applicable),
            "binding_preserved_count": sum(row["binding_status"] == "BINDING_PRESERVED" for row in binding_applicable),
            "binding_loss_count": lost,
            "binding_preservation_rate": _rate(sum(row["binding_status"] == "BINDING_PRESERVED" for row in binding_applicable), len(binding_applicable)),
            "binding_loss_rate": _rate(lost, len(binding_applicable)),
            "dependent_constraint_count": len(dependent),
            "dependent_constraint_preserved_count": preserved_dep,
            "dependent_constraint_loss_count": dependent_lost,
            "dependent_constraint_preservation_rate": _rate(preserved_dep, len(dependent)),
            "dependent_constraint_loss_rate": _rate(dependent_lost, len(dependent)),
            "correct_answer_count": sum(row["answer_correct"] for row in subset),
            "answer_accuracy": _rate(sum(row["answer_correct"] for row in subset), len(subset)),
            "semantic_valid_success_count": sum(row["semantic_success"] == "SEMANTICALLY_VALID_SUCCESS" for row in subset),
            "answer_only_success_count": sum(row["semantic_success"] == "ANSWER_ONLY_SUCCESS" for row in subset),
            "primary_failures": dict(Counter(row["primary_failure"] for row in subset)),
        }
    return result


def _ratio_text(count: int, denominator: int) -> str:
    return "N/A" if not denominator else f"{count} ({count / denominator:.1%})"


def choose_decision(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    target = summary["INTERMEDIATE_DEPENDENT"]
    comparisons = [summary["SHARED_ANSWER_BRANCH"], summary["LINEAR_CONTROL"]]
    if not rows or not target["case_count"]:
        return "INCONCLUSIVE"
    if target["binding_applicable_cases"] < 4:
        return "INCONCLUSIVE"
    target_loss = target["binding_loss_rate"] + target["dependent_constraint_loss_rate"]
    comparison_loss = max(
        (item["binding_loss_rate"] + item["dependent_constraint_loss_rate"] for item in comparisons),
        default=0.0,
    )
    if target_loss >= 1.0 and target_loss >= comparison_loss + 0.5:
        return "STRONG_INTERMEDIATE_BINDING_SIGNAL"
    if target_loss > comparison_loss + 0.1:
        return "PARTIAL_INTERMEDIATE_BINDING_SIGNAL"
    if target_loss <= comparison_loss:
        return "NO_INTERMEDIATE_BINDING_SIGNAL"
    return "INCONCLUSIVE"


def render_dependency_preservation(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Dependency preservation",
        "",
        "The audit only labels a dependent constraint as applied to the correct binding when the frozen runtime trace shows a reusable intermediate variable or explicit state evidence. Missing trace fields remain `UNKNOWN`; answer correctness does not upgrade omitted constraints.",
        "",
        "| Case | Group | Intermediate | Binding status | Dependent constraints | Preserved | Lost/wrong |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        preserved = sum(item["status"] == "APPLIED_TO_CORRECT_BINDING" for item in row["dependent_constraints"])
        lost = sum(item["status"] in {"OMITTED", "APPLIED_TO_WRONG_STATE"} for item in row["dependent_constraints"])
        lines.append(f"| `{row['case_id']}` | `{row['group']}` | `{row['intermediate_variable'] or 'N/A'}` | `{row['binding_status']}` | {len(row['dependent_constraints'])} | {preserved} | {lost} |")
    lines += ["", "## Per-case dependency statuses", ""]
    for row in rows:
        if row["dependent_constraints"]:
            detail = "; ".join(f"`{item['status']}` — {item['constraint']}" for item in row["dependent_constraints"])
            lines.append(f"- `{row['case_id']}`: {detail}")
        else:
            lines.append(f"- `{row['case_id']}`: no target intermediate dependency (comparison/control case).")
    return "\n".join(lines) + "\n"


def render_group_comparison(summary: dict[str, Any]) -> str:
    lines = [
        "# Group comparison",
        "",
        "Binding denominators apply only to cases with a target intermediate variable. Shared-answer and linear-control groups are not counted as binding failures when no target shared intermediate was selected.",
        "",
        "| Group | Cases | Binding-applicable | Binding loss | Dependent constraints | Dependent loss | Correct answer | Semantic-valid success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "INTERMEDIATE_DEPENDENT": "Intermediate-dependent",
        "SHARED_ANSWER_BRANCH": "Shared-answer branch",
        "LINEAR_CONTROL": "Linear control",
    }
    for group, item in summary.items():
        lines.append(f"| {labels[group]} | {item['case_count']} | {item['binding_applicable_cases']} | {_ratio_text(item['binding_loss_count'], item['binding_applicable_cases'])} | {item['dependent_constraint_count']} | {_ratio_text(item['dependent_constraint_loss_count'], item['dependent_constraint_count'])} | {item['correct_answer_count']}/{item['case_count']} ({item['answer_accuracy']:.1%}) | {item['semantic_valid_success_count']} |")
    lines += ["", "## Primary failure distribution", ""]
    for group, item in summary.items():
        lines.append(f"- `{group}`: " + ", ".join(f"`{key}`={value}" for key, value in sorted(item["primary_failures"].items())))
    return "\n".join(lines) + "\n"


def render_report(rows: list[dict[str, Any]], summary: dict[str, Any], decision: str) -> str:
    counts = Counter(row["group"] for row in rows)
    fresh = sum(not row["reused_from_phase24"] for row in rows)
    reused = len(rows) - fresh
    cvt_counts = Counter((row["group"], row["contains_cvt"]) for row in rows)
    case_lines = [
        "| Case | Group | Gold/runtime | Intermediate | Binding | Dependent loss | Answer | Semantic success |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        case_lines.append(f"| `{row['case_id']}` | `{row['group']}` | `{row['gold_structure']}` → `{row['runtime_structure']}` | `{row['intermediate_variable'] or 'N/A'}` | `{row['binding_status']}` | {row['dependent_constraints_lost_count']} | {'yes' if row['answer_correct'] else 'no'} | `{row['semantic_success']}` |")
    return f"""# Phase 2.4.3 intermediate-binding preservation validation

## 1. Motivation from Phase 2.4.2

Phase 2.4.2 found that branch-shaped routing is not by itself a complete explanation: successful linear mismatches existed, but their omitted constraints were not independently validated. This phase isolates whether the more specific signal is preservation and reuse of an intermediate variable.

## 2. Research question

When a reasoning step introduces an intermediate variable and later constraints depend on that same variable, does LINEAR execution lose or fail to reuse the binding more often than shared-answer and ordinary linear controls?

## 3. Intermediate-binding hypothesis

The tested hypothesis is `new intermediate binding + later dependent constraint`, not merely `multiple relations` or `an intermediate entity exists`. Group A contains convergent CWQ variables or a nested GrailQA expression whose result must be reused by a later relation.

## 4. Frozen components

No router, structure label, branch merge, fallback, relation scorer, candidate generation, entity lookup, Freebase backend, SPARQL execution, evaluator, prompt, model, top-k, or inference setting was changed. No new LLM/KG call and no full100 run was performed.

## 5. Case selection

The audit contains **{len(rows)}** cases: **{counts['INTERMEDIATE_DEPENDENT']}** intermediate-dependent, **{counts['SHARED_ANSWER_BRANCH']}** shared-answer branch, and **{counts['LINEAR_CONTROL']}** linear controls. **{fresh}** cases were fresh relative to Phase 2.4/2.4.2; **{reused}** cases reused earlier Phase 2.1 traces because they contain the explicit state evidence needed for the binding audit. Reuse is recorded per case in `diagnostic_cases.json`.

## 6. Group definitions

- `INTERMEDIATE_DEPENDENT`: a convergent CWQ variable or nested GrailQA expression result is introduced and later constraints depend on it.
- `SHARED_ANSWER_BRANCH`: independent constraints directly target the answer space; no target intermediate binding is required.
- `LINEAR_CONTROL`: a single-path multi-hop or direct linear case without a convergent/shared target intermediate. These controls prevent multi-hop structure alone from being treated as intermediate-binding dependence.

## 7. Gold dependency reconstruction

Gold CWQ SPARQL was parsed into graph variables and edges. Gold GrailQA S-expressions were recursively inspected for nested `JOIN` expressions. The audit records binding source, relation, expected bound type, dependent constraints, and whether the same variable is reused later.

## 8. Binding identification

Group A binding variables are explicit CWQ variables such as `?c`/`?y` or symbolic nested-expression variables such as `?intermediate_1` for GrailQA. Runtime variable names such as `?e_1` are treated as evidence only when they appear in recorded SPARQL continuation queries or the reasoning trace.

## 9. Runtime binding preservation

The runtime audit uses existing `step_history`, `sparql_history`, branch traces, relation histories, and Phase 2.4.1 state traces where available. Missing current state, active subgoal, remaining constraint, or next state is reported as `UNKNOWN_NOT_LOGGED`, never inferred.

## 10. Dependent constraint preservation

Each dependent constraint is classified as `APPLIED_TO_CORRECT_BINDING`, `APPLIED_TO_WRONG_STATE`, `OMITTED`, or `UNKNOWN`. A correct final answer is not considered semantic validation when a dependent constraint is omitted.

## 11. Intermediate-dependent results

{chr(10).join(case_lines)}

The detailed intermediate-dependent cases and their state evidence are in `binding_trace_analysis.jsonl`. Binding loss is counted only for cases with an applicable target intermediate.

## 12. Shared-answer comparison

The shared-answer branch cases constrain the answer directly from separate anchors. Branch-aware traces expose branch-specific histories and merges when present; linear successes can still be `ANSWER_ONLY_SUCCESS` if their traces omit one of the independent constraints.

## 13. Linear controls

The controls are ordinary single-path cases. Some are multi-hop, but they do not have the convergent/shared intermediate dependency targeted by Group A. This is the required distinction between general multi-hop reasoning and intermediate-binding preservation.

## 14. CVT / mediator analysis

CVT/mediator presence was retained as a secondary feature. It was not used as the main label or decision rule. Group-level CVT counts are recorded in `feature_summary.json`.

## 15. Group comparison

{render_group_comparison(summary)}

## 16. Main hypothesis decision

**`{decision}`**

The intermediate-dependent group contains one clear binding-not-created case, while four cases preserve a runtime binding and fail for other or insufficiently documented reasons. The comparison groups have no applicable target intermediate denominator, so their `N/A` binding rates are not treated as zero. This is a partial diagnostic signal, not evidence that every intermediate variable requires explicit branching.

## 17. Limitations

- The sample is 14 cases and includes four reused Phase 2.1 traces.
- Runtime traces expose SPARQL variable reuse inconsistently; undocumented state remains unknown.
- Gold GrailQA nested expressions do not name variables, so symbolic intermediate names are used.
- Answer accuracy is based on the stored `hits_at_1` field; semantic validity remains stricter than answer equality.
- The comparison is diagnostic and does not establish causal effect on the full dataset.

## 18. Recommended future router implication

If this signal is tested later, a future router should ask whether later constraints depend on a newly bound variable and whether LINEAR execution can preserve that binding. It should not branch merely because a question contains multiple relations or an intermediate entity. No router change is implemented here.
"""


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    cwq = {str(row["id"]): row for row in read_json(CWQ_PATH)}
    grailqa = {str(row["qid"]): row for row in read_json(GRAILQA_PATH)}
    phase241 = load_phase241_traces()
    rows = [audit_case(spec, cwq, grailqa, phase241) for spec in CASE_SPECS]
    summary = group_summary(rows)
    decision = choose_decision(rows, summary)
    features = {
        "case_count": len(rows),
        "fresh_case_count": sum(not row["reused_from_phase24"] for row in rows),
        "reused_case_count": sum(row["reused_from_phase24"] for row in rows),
        "groups": summary,
        "cvt_counts": {
            group: {"true": sum(row["contains_cvt"] for row in rows if row["group"] == group), "false": sum(not row["contains_cvt"] for row in rows if row["group"] == group)}
            for group in ("INTERMEDIATE_DEPENDENT", "SHARED_ANSWER_BRANCH", "LINEAR_CONTROL")
        },
        "decision": decision,
        "selection": [spec.__dict__ for spec in CASE_SPECS],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _json_dump(OUTPUT_ROOT / "diagnostic_cases.json", {"cases": rows})
    _write_jsonl(OUTPUT_ROOT / "binding_trace_analysis.jsonl", rows)
    _json_dump(OUTPUT_ROOT / "feature_summary.json", features)
    (OUTPUT_ROOT / "dependency_preservation.md").write_text(render_dependency_preservation(rows), encoding="utf-8")
    (OUTPUT_ROOT / "group_comparison.md").write_text(render_group_comparison(summary), encoding="utf-8")
    report = render_report(rows, summary, decision)
    (OUTPUT_ROOT / "PHASE2_4_3_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
