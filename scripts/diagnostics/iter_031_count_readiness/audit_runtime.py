#!/usr/bin/env python3
"""Create Iteration 31 runtime-only audits, then separate gold-only joins.

The first phase reads only `question_key` from the retrospective COUNT audit
to select the already-studied 68 cases. All per-case inference fields are then
derived exclusively from Iteration 29 NON_GOLD traces and cohort rows. It writes
those audits before opening any scoring-only columns. The second phase adds
explicitly suffixed gold-only comparisons and never revises the runtime rows.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from chain_of_relations.methods.iter_030_operator_recovery.relation_aware_count import classify_count_intent
from chain_of_relations.methods.iter_031_count_readiness.evidence import EvidenceRef
from chain_of_relations.methods.iter_031_count_readiness.path_completeness import PathEvidence, PathStep, check_path_completeness
from chain_of_relations.methods.iter_031_count_readiness.type_resolution import TargetTypeEvidence, resolve_target_type


RESULTS = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS"
I30 = RESULTS / "iter_030_results_first"
I29 = RESULTS / "iter_028_terminal_boundary_repair"
RUN = I29 / "RUNS/iter29_terminal_candidate_selector_attempt3"
TRACE_PATH = RUN / "CASE_TRACES_NON_GOLD.jsonl"
COHORT_PATH = I29 / "NEXT_INFERENCE_COHORT_ATTEMPT3_NON_GOLD.jsonl"
COUNT_AUDIT_PATH = I30 / "COUNT_OPERATOR_AUDIT.csv"
PAIRED_REFERENCE_PATH = I30 / "ITER29_PAIRED_CASE_REFERENCE_GOLD_ONLY.csv"
OUT = RESULTS / "iter_031_gold_free_count_readiness"
TARGET_FIELDS = ("target_type_id", "resolved_target_type_id", "final_target_type_id")
PLAN_KEYS = ("count_plan", "expected_relation_path", "finalized_graph_plan", "finalized_plan")
TYPE_VAR_KEYS = ("target_variable", "final_target_variable", "type_target_variable")
_TRIPLE = re.compile(r"^\s*(\S+)\s+ns:([A-Za-z0-9_.-]+)\s+(\S+)\s*\.\s*$")
_PROJECTED = re.compile(r"\bSELECT\s+(?:DISTINCT\s+)?(\?[A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_INEQUALITY = re.compile(r"FILTER\s*\(\s*\?[A-Za-z_][A-Za-z0-9_]*\s*!=\s*\?[A-Za-z_][A-Za-z0-9_]*\s*\)", re.IGNORECASE)
_TYPE_FILTER = re.compile(r"(\?[A-Za-z_][A-Za-z0-9_]*)\s+ns:type\.object\.type\s+ns:[A-Za-z0-9_.-]+", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_selected_keys_only(path: Path) -> tuple[list[str], dict[str, Any]]:
    """The runtime phase is permitted the cohort key list, and no gold values."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if "question_key" not in (reader.fieldnames or []):
            raise ValueError("COUNT audit lacks question_key selector column")
        keys = [str(row["question_key"]) for row in reader if row.get("question_key")]
    if len(keys) != 68 or len(set(keys)) != 68:
        raise ValueError(f"expected 68 unique retrospective case keys, got {len(keys)} / {len(set(keys))}")
    return keys, {
        "source": str(path.relative_to(ROOT)),
        "source_sha256": sha256(path),
        "rows_selected": len(keys),
        "columns_read": ["question_key"],
        "gold_values_read_for_runtime_classification": False,
    }


def _router_hint(trace: Mapping[str, Any]) -> tuple[str, str]:
    history = trace.get("prompt_history")
    if not isinstance(history, list):
        return "", ""
    for index in range(len(history) - 1, -1, -1):
        event = history[index]
        if not isinstance(event, Mapping):
            continue
        parsed = event.get("parsed_result")
        if isinstance(parsed, Mapping):
            hint = parsed.get("requested_final_answer_type") or parsed.get("shared_answer_type")
            if hint:
                field = "requested_final_answer_type" if parsed.get("requested_final_answer_type") else "shared_answer_type"
                return str(hint).strip(), f"prompt_history[{index}].parsed_result.{field}"
        if event.get("requested_final_answer_type"):
            return str(event["requested_final_answer_type"]).strip(), f"prompt_history[{index}].requested_final_answer_type"
    return "", ""


def _find_explicit_type_fields(value: Any, path: str = "trace") -> list[tuple[str, str, str]]:
    """Find only structured runtime type IDs explicitly scoped to a variable."""
    found: list[tuple[str, str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in TARGET_FIELDS and isinstance(child, str):
                variable = next((str(value.get(name, "")) for name in TYPE_VAR_KEYS if value.get(name)), "")
                constraint_kind = str(value.get("constraint_kind", value.get("evidence_kind", ""))).casefold()
                if variable and constraint_kind in {"explicit_target_type_constraint", "target_type_constraint"}:
                    found.append((child, variable, child_path))
            if isinstance(child, (Mapping, list)):
                found.extend(_find_explicit_type_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, (Mapping, list)):
                found.extend(_find_explicit_type_fields(child, f"{path}[{index}]"))
    return found


def _last_entity_search(trace: Mapping[str, Any]) -> Mapping[str, Any] | None:
    history = trace.get("sparql_history")
    if not isinstance(history, list):
        return None
    events = [event for event in history if isinstance(event, Mapping) and event.get("search_type") == "entity_search"]
    return events[-1] if events else None


def _linear_path_from_query(query: str, topic_id: str) -> tuple[tuple[PathStep, ...], bool, str, str, list[str]]:
    triples: list[tuple[str, str, str]] = []
    for line in str(query or "").splitlines():
        match = _TRIPLE.match(line)
        if match and match.group(2) != "type.object.type":
            triples.append(match.groups())
    root_token = f"ns:{topic_id}"
    current = root_token
    node_variable: dict[str, str] = {root_token: "?observed_0"}
    steps: list[PathStep] = []
    remaining = list(triples)
    valid = bool(triples)
    while remaining:
        matches = [(i, item) for i, item in enumerate(remaining) if item[0] == current or item[2] == current]
        if len(matches) != 1:
            valid = False
            break
        index, (subject, relation, obj) = matches[0]
        direction = "out" if subject == current else "in"
        nxt = obj if direction == "out" else subject
        current_variable = node_variable.setdefault(current, f"?observed_{len(node_variable) - 1}")
        next_variable = node_variable.setdefault(nxt, f"?observed_{len(node_variable)}")
        steps.append(PathStep(relation, direction, current_variable, next_variable))
        current = nxt
        remaining.pop(index)
    if remaining or len(steps) != len(triples):
        valid = False
    projected = _PROJECTED.search(str(query or ""))
    target_variable = projected.group(1) if projected else ""
    if target_variable and steps and current != target_variable:
        # The path parser names variables by graph position; retain the query's
        # explicit terminal variable as the observed target identity.
        if current.startswith("?") and current == target_variable:
            pass
    relation_ids = [step.relation_id for step in steps]
    return tuple(steps), valid, target_variable, current, relation_ids


def _plan_fields_present(trace: Mapping[str, Any]) -> bool:
    def walk(value: Any) -> bool:
        if isinstance(value, Mapping):
            if any(key in value for key in PLAN_KEYS):
                return True
            return any(walk(child) for child in value.values() if isinstance(child, (Mapping, list)))
        if isinstance(value, list):
            return any(walk(child) for child in value if isinstance(child, (Mapping, list)))
        return False
    return walk(trace)


def build_runtime_rows(keys: list[str], traces: list[dict[str, Any]], cohort: list[dict[str, Any]], trace_hash: str, cohort_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    trace_by_key = {str(row.get("question_key", "")): (index, row) for index, row in enumerate(traces, 1)}
    cohort_by_key = {str(row.get("question_key", "")): (index, row) for index, row in enumerate(cohort, 1)}
    if len(trace_by_key) != len(traces) or len(cohort_by_key) != len(cohort):
        raise ValueError("duplicate question_key in frozen non-gold artifacts")
    target_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    runtime_base: list[dict[str, Any]] = []
    for key in keys:
        if key not in trace_by_key or key not in cohort_by_key:
            raise ValueError(f"retrospective case missing non-gold source row: {key}")
        trace_line, trace = trace_by_key[key]
        cohort_line, cohort_row = cohort_by_key[key]
        question = str(trace.get("question") or cohort_row.get("question") or "")
        if question != str(cohort_row.get("question", "")):
            raise ValueError(f"question text mismatch between frozen trace/cohort for {key}")
        topic_ids = cohort_row.get("topic_entities") or []
        topic_id = str(topic_ids[0]) if topic_ids else ""
        answer_hint, answer_hint_field = _router_hint(trace)
        typed = _find_explicit_type_fields(trace)
        evidence = tuple(
            TargetTypeEvidence(
                type_id=type_id,
                target_variable=target_var,
                evidence_kind="explicit_target_type_constraint",
                provenance=EvidenceRef("CASE_TRACES_NON_GOLD.jsonl", trace_hash, field_path, key, "runtime"),
                detail="structured runtime field; gold-free audit",
            )
            for type_id, target_var, field_path in typed
        )
        entity_query = _last_entity_search(trace)
        query = str((entity_query or {}).get("sparql", ""))
        query_status = str((entity_query or {}).get("status", "missing"))
        steps, path_parse_ok, projected, terminal_term, observed_relations = _linear_path_from_query(query, topic_id)
        target_var = projected or "?targetEntity"
        type_filter_match = _TYPE_FILTER.search(query)
        type_filter_var = type_filter_match.group(1) if type_filter_match else ""
        resolver = resolve_target_type(target_var, evidence, answer_type_hint=answer_hint)
        trace_ref = EvidenceRef("CASE_TRACES_NON_GOLD.jsonl", trace_hash, f"jsonl[{trace_line}].sparql_history", key, "runtime")
        inequality_count = len(_INEQUALITY.findall(query)) if query else 0
        expected_inequality_count = (len(steps) + 1) * len(steps) // 2
        path_evidence = PathEvidence(
            target_variable=target_var,
            query_target_variable=projected or target_var,
            expected_steps=None,
            executed_steps=steps,
            query_status=query_status,
            plan_finalized=None,
            plan_shape="linear" if path_parse_ok else "unknown",
            required_external_binding_variables=(),
            active_binding_tuples=(),
            provenance=(trace_ref, EvidenceRef("NEXT_INFERENCE_COHORT_ATTEMPT3_NON_GOLD.jsonl", cohort_hash, f"jsonl[{cohort_line}].topic_entities", key, "runtime")),
            required_constraints_complete=None,
            all_pairwise_node_inequalities_represented=(inequality_count == expected_inequality_count and expected_inequality_count > 0) if query else None,
            type_filter_variable=type_filter_var,
        )
        path_check = check_path_completeness(path_evidence, resolver)
        triggered, trigger_reason = classify_count_intent(question)
        expected_plan_present = _plan_fields_present(trace)
        range_fields_seen = "relation_range" in json.dumps(trace, ensure_ascii=False).casefold() or "range_type" in json.dumps(trace, ensure_ascii=False).casefold()
        target_type_fields_seen = bool(typed)
        target_rows.append({
            "question_key": key,
            "question": question,
            "router_answer_type_text": answer_hint,
            "router_answer_type_hint_field_path": answer_hint_field,
            "router_answer_type_hint_source_sha256": trace_hash if answer_hint_field else "",
            "target_variable_observed": target_var,
            "explicit_target_type_constraint_count": len(typed),
            "canonical_target_type_ids_runtime_only": json.dumps(sorted({item.type_id for item in evidence}), separators=(",", ":")),
            "relation_range_metadata_observed": range_fields_seen,
            "target_type_id_field_observed": target_type_fields_seen,
            "resolution_status": resolver.status,
            "evidence_grade": resolver.evidence_grade,
            "confidence_label": resolver.confidence_label,
            "resolution_justification": resolver.justification,
            "provenance": json.dumps([item.as_dict() for item in resolver.evidence], sort_keys=True, separators=(",", ":")),
            "trace_line_1based": trace_line,
            "trace_sha256": trace_hash,
        })
        path_rows.append({
            "question_key": key,
            "question": question,
            "observed_relation_ids_runtime_only": json.dumps(observed_relations, separators=(",", ":")),
            "observed_directions_runtime_only": json.dumps([step.direction for step in steps], separators=(",", ":")),
            "observed_path_parse_complete": path_parse_ok,
            "expected_finalized_plan_present": expected_plan_present,
            "query_status": query_status,
            "query_target_variable": projected,
            "query_projects_final_target_only": bool(projected and re.search(r"SELECT\s+DISTINCT\s+\?targetEntity\b", query, re.IGNORECASE)),
            "intermediate_binding_tuples_persisted": False,
            "target_type_filter_variable": type_filter_var,
            "pairwise_node_inequality_filters_in_query": inequality_count,
            "path_completeness_status": path_check.status,
            "path_completeness_reason": path_check.reason,
            "entity_search_result_count": (entity_query or {}).get("result_count", ""),
            "result_count_at_or_above_1000": int((entity_query or {}).get("result_count", 0) or 0) >= 1000,
            "trace_line_1based": trace_line,
            "trace_sha256": trace_hash,
        })
        runtime_base.append({
            "question_key": key,
            "question": question,
            "count_intent_triggered_by_frozen_wording_rule": triggered,
            "count_intent_reason": trigger_reason,
            "target_type_status": resolver.status,
            "path_status": path_check.status,
            "runtime_plan_present": expected_plan_present,
            "runtime_ready": bool(triggered and resolver.status == "resolved" and path_check.ready),
            "runtime_blocker": ";".join(reason for reason, active in (
                ("target_type_not_resolved", resolver.status != "resolved"),
                ("finalized_expected_path_missing", not expected_plan_present),
                ("path_completeness_not_proven", not path_check.ready),
            ) if active),
            "observed_query_status": query_status,
            "observed_relation_ids": json.dumps(observed_relations, separators=(",", ":")),
            "trace_line_1based": trace_line,
            "cohort_line_1based": cohort_line,
        })
    return target_rows, path_rows, runtime_base


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty audit {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def score_only_join(runtime_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Second phase: attach posthoc scoring labels without editing runtime rows."""
    with COUNT_AUDIT_PATH.open(encoding="utf-8", newline="") as stream:
        count_rows = list(csv.DictReader(stream))
    with PAIRED_REFERENCE_PATH.open(encoding="utf-8", newline="") as stream:
        paired_rows = list(csv.DictReader(stream))
    gold_by_key = {row["question_key"]: row for row in count_rows}
    paired_by_key = {row["question_key"]: row for row in paired_rows}
    matrix: list[dict[str, Any]] = []
    for runtime in runtime_rows:
        key = runtime["question_key"]
        analytic = gold_by_key[key]
        paired = paired_by_key[key]
        matrix.append({
            **runtime,
            "gold_only_operator_root_scoring_only": analytic["gold_operator_root"],
            "gold_only_target_class_scoring_only": analytic["gold_target_class_id_scoring_only"],
            "gold_only_relation_ids_scoring_only": analytic["gold_path_relation_ids_scoring_only"],
            "gold_only_binary_edge_count_scoring_only": analytic["gold_binary_edge_count"],
            "gold_only_requires_shared_join_scoring_only": analytic["requires_shared_variable_join"],
            "gold_only_relation_set_agreement_scoring_only": analytic["runtime_relation_set_matches_gold_scoring_only"],
            "gold_only_gold_type_filter_present_scoring_only": analytic["gold_type_filter_present"],
            "gold_only_gold_pairwise_inequalities_present_scoring_only": int(analytic["gold_query_inequality_pair_count"]),
            "gold_only_candidate_rows_equal_gold_count_scoring_only": analytic["runtime_count_rows_equals_gold_scoring_only"],
            "gold_only_baseline_hits_at_1_scoring_only": paired["baseline_hits_at_1_scoring_only"],
            "gold_only_baseline_scalar_answer_match_scoring_only": analytic["runtime_count_rows_exact_scalar_answer_scoring_only"],
            "analytic_scope_warning": "gold-only fields are retrospective comparison columns; they were not inputs to runtime_ready",
        })
    if len(matrix) != 68 or len({row["question_key"] for row in matrix}) != 68:
        raise ValueError("gold-only comparison join did not preserve 68 unique rows")
    with PAIRED_REFERENCE_PATH.open(encoding="utf-8", newline="") as stream:
        full_reference = list(csv.DictReader(stream))
    if len(full_reference) != 800:
        raise ValueError("expected 800 rows for trigger-control audit")
    full_traces = {str(row.get("question_key", "")): row for row in read_jsonl(TRACE_PATH)}
    trigger_rows = []
    for row in full_reference:
        trace = full_traces.get(row["question_key"])
        if trace is None:
            raise ValueError(f"missing non-gold trace for control key {row['question_key']}")
        triggered, reason = classify_count_intent(str(trace.get("question", "")))
        is_count = row["gold_root_operator_scoring_only"] == "COUNT"
        trigger_rows.append({
            "question_key": row["question_key"],
            "triggered": triggered,
            "trigger_reason": reason,
            "gold_root_is_count_scoring_only": is_count,
            "baseline_hit_scoring_only": row["baseline_hits_at_1_scoring_only"] == "1",
        })
    trigger_counts = Counter({
        "triggered_all": 0,
        "triggered_count_root": 0,
        "triggered_non_count_root": 0,
        "triggered_previously_correct_non_count": 0,
        "non_count_root_total": 0,
    })
    for row in trigger_rows:
        if not row["gold_root_is_count_scoring_only"]:
            trigger_counts["non_count_root_total"] += 1
        if row["triggered"]:
            trigger_counts["triggered_all"] += 1
            if row["gold_root_is_count_scoring_only"]:
                trigger_counts["triggered_count_root"] += 1
            else:
                trigger_counts["triggered_non_count_root"] += 1
                if row["baseline_hit_scoring_only"]:
                    trigger_counts["triggered_previously_correct_non_count"] += 1
    alignment = {
        "runtime_rows": len(matrix),
        "gold_only_root_count_rows": sum(row["gold_only_operator_root_scoring_only"] == "COUNT" for row in matrix),
        "gold_only_multiedge_rows": sum(int(row["gold_only_binary_edge_count_scoring_only"]) > 1 for row in matrix),
        "gold_only_two_hop_rows": sum(int(row["gold_only_binary_edge_count_scoring_only"]) == 2 for row in matrix),
        "gold_only_three_hop_rows": sum(int(row["gold_only_binary_edge_count_scoring_only"]) == 3 for row in matrix),
        "gold_only_distinct_target_class_count": len({row["gold_only_target_class_scoring_only"] for row in matrix}),
        "gold_only_relation_set_agreement_count": sum(row["gold_only_relation_set_agreement_scoring_only"] == "True" for row in matrix),
        "gold_only_candidate_rows_equal_gold_count": sum(row["gold_only_candidate_rows_equal_gold_count_scoring_only"] == "True" for row in matrix),
        "gold_only_baseline_scalar_match_count": sum(row["gold_only_baseline_scalar_answer_match_scoring_only"] == "True" for row in matrix),
        "gold_only_target_type_filter_count": sum(row["gold_only_gold_type_filter_present_scoring_only"] == "True" for row in matrix),
        "frozen_wording_trigger_on_800": dict(trigger_counts),
        "trigger_control_evidence_class": "posthoc development diagnostic; not independent validation",
    }
    control_rows = []
    for row in trigger_rows:
        control_rows.append({
            "question_key": row["question_key"],
            "count_triggered_by_frozen_runtime_wording_rule": row["triggered"],
            "trigger_reason_runtime_text_only": row["trigger_reason"],
            "gold_only_root_operator_scoring_only": "COUNT" if row["gold_root_is_count_scoring_only"] else "OTHER",
            "gold_only_baseline_hit_at_1_scoring_only": row["baseline_hit_scoring_only"],
            "gold_only_interpretation": "trigger/control join is retrospective development diagnostics; no runtime classification used these fields",
        })
    return matrix, alignment, control_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys, selector_audit = read_selected_keys_only(COUNT_AUDIT_PATH)
    traces = read_jsonl(TRACE_PATH)
    cohort = read_jsonl(COHORT_PATH)
    if len(traces) != 800 or len(cohort) != 800:
        raise ValueError("expected frozen 800-row trace and inference cohort")
    trace_hash, cohort_hash = sha256(TRACE_PATH), sha256(COHORT_PATH)
    target_rows, path_rows, runtime_rows = build_runtime_rows(keys, traces, cohort, trace_hash, cohort_hash)
    # Persist the strictly gold-free material before opening any scoring fields.
    write_csv(OUT / "TARGET_TYPE_RESOLVER_AUDIT.csv", target_rows)
    write_csv(OUT / "PATH_COMPLETENESS_AUDIT.csv", path_rows)
    write_csv(OUT / "RUNTIME_READINESS_BASE.csv", runtime_rows)
    matrix, diagnostics, control_rows = score_only_join(runtime_rows)
    write_csv(OUT / "COUNT_READINESS_MATRIX.csv", matrix)
    write_csv(OUT / "TRIGGER_CONTROL_AUDIT.csv", control_rows)
    runtime_summary = {
        "iteration": 31,
        "scope": "offline audit of 68 retrospective GrailQA dev COUNT-error keys",
        "network_calls": {"openai": 0, "sparql": 0},
        "gold_free_selector_audit": selector_audit,
        "runtime_input_hashes": {
            "trace_non_gold_sha256": trace_hash,
            "cohort_non_gold_sha256": cohort_hash,
        },
        "runtime_only_counts": {
            "cases": len(runtime_rows),
            "triggered": sum(row["count_intent_triggered_by_frozen_wording_rule"] for row in runtime_rows),
            "type_status": dict(Counter(row["target_type_status"] for row in runtime_rows)),
            "path_status": dict(Counter(row["path_status"] for row in runtime_rows)),
            "runtime_ready": sum(row["runtime_ready"] for row in runtime_rows),
        },
        "gold_only_diagnostics_separate": diagnostics,
        "selection_warning": "The 68 case keys were selected by a retrospective gold COUNT audit; no gold values or structures were read for runtime classifications. This is not independent validation.",
    }
    (OUT / "RUNTIME_AUDIT_SUMMARY.json").write_text(json.dumps(runtime_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(runtime_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
