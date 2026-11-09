"""Reproduce gold-only failure taxonomy for Iteration 29 attempt 3.

Offline-only: reads the frozen run artifacts and GrailQA dev labels, writes
analysis artifacts under RESULTS/iter_030_results_first, and performs no
model, endpoint, or other network calls.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from relation_graph_kgqa.logical_forms.sexpr_ast import walk_expr
from relation_graph_kgqa.logical_forms.sexpr_parser import parse_s_expression
from chain_of_relations.methods.iter_030_operator_recovery.relation_aware_count import classify_count_intent


ROOT = Path(__file__).resolve().parents[3]
I29 = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_028_terminal_boundary_repair"
RUN = I29 / "RUNS/iter29_terminal_candidate_selector_attempt3"
OUT = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_030_results_first"
MANIFEST = I29 / "NEXT_MANIFEST_DRAFT_ATTEMPT3.json"
COHORT = I29 / "NEXT_INFERENCE_COHORT_ATTEMPT3_NON_GOLD.jsonl"
GOLD_SOURCE = ROOT / "data/interim/grailqa_dev.jsonl"
NUMERIC = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
VALUES_RE = re.compile(r"VALUES\s+(\?\w+)\s*\{\s*:(m\.[A-Za-z0-9_]+|g\.[A-Za-z0-9_]+)\s*\}", re.I)
TRIPLE_RE = re.compile(r"(\?\w+|:[A-Za-z0-9_.-]+)\s+(:[A-Za-z0-9_.-]+)\s+(\?\w+|:[A-Za-z0-9_.-]+)\s*\.")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_jsonl_indexed(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            rows.append((line_number, json.loads(line)))
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_numeric_answer(values: Any) -> bool:
    return bool(values) and all(NUMERIC.fullmatch(str(value).strip().replace(",", "")) for value in values)


def int_value(values: Any) -> int | None:
    if not is_numeric_answer(values) or len(values) != 1:
        return None
    try:
        number = float(str(values[0]).replace(",", ""))
        return int(number) if number.is_integer() else None
    except (TypeError, ValueError, OverflowError):
        return None


def normalize_relation(value: str) -> str:
    return str(value).replace("http://rdf.freebase.com/ns/", "").lstrip(":").strip()


def runtime_first_hop(trace: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return the first executed CoR predicate and its topic-relative direction."""
    path = str(trace.get("executed_relation_path") or "")
    for line in path.splitlines():
        if "candidate.target" in line:
            continue
        match = re.match(r"\s*TopicEntity\([^)]*\)\s+(\S+)\s+e1\s+\.", line)
        if match:
            return normalize_relation(match.group(1)), "out"
        match = re.match(r"\s*e1\s+(\S+)\s+TopicEntity\([^)]*\)\s+\.", line)
        if match:
            return normalize_relation(match.group(1)), "in"
    return None, None


def gold_topic_first_edges(raw: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract gold-only first binary edges from topic VALUES in Gold SPARQL."""
    query = str(raw.get("gold_sparql") or "")
    topic_variables = {match.group(1): match.group(2) for match in VALUES_RE.finditer(query)}
    edges: list[tuple[str, str]] = []
    for match in TRIPLE_RE.finditer(query):
        subject, predicate, obj = match.groups()
        relation = normalize_relation(predicate)
        if relation == "type.object.type":
            continue
        for topic_var in topic_variables:
            if subject == topic_var and obj.startswith("?"):
                edges.append((relation, "out"))
            elif obj == topic_var and subject.startswith("?"):
                edges.append((relation, "in"))
    return list(dict.fromkeys(edges))


def initial_relation_candidates(trace: dict[str, Any]) -> list[tuple[str, set[str], set[str]]]:
    """Read directional candidate IDs and selected IDs from first-hop prompts."""
    result = []
    for event in trace.get("step_history") or []:
        if event.get("step_type") != "llm" or event.get("operation_type") != "relation_prune":
            continue
        prompt = str(event.get("prompt") or "")
        selected_raw = event.get("parsed_result") or []
        selected = {normalize_relation(value) for value in selected_raw} if isinstance(selected_raw, list) else set()
        lines = prompt.splitlines()
        for index, header in enumerate(lines):
            lowered = header.lower()
            if not header.strip().startswith("#") or "topicentity(" not in lowered:
                continue
            if re.search(r"\be\d+\s+", header):
                continue
            if "?relation ?x" in lowered:
                direction = "out"
            elif "?x ?relation" in lowered:
                direction = "in"
            else:
                continue
            if index + 1 >= len(lines):
                continue
            candidates = {
                normalize_relation(item)
                for item in lines[index + 1].split(";")
                if "." in item and normalize_relation(item)
            }
            result.append((direction, candidates, selected))
    return result


def final_entity_query(trace: dict[str, Any]) -> dict[str, Any]:
    return next(
        (event for event in reversed(trace.get("sparql_history") or []) if event.get("search_type") == "entity_search"),
        {},
    )


def last_reasoning_action(trace: dict[str, Any]) -> str:
    return str(next(
        (event.get("parsed_result") for event in reversed(trace.get("step_history") or [])
         if event.get("step_type") == "llm" and event.get("operation_type") == "reasoning"),
        "",
    ))


def audit() -> dict[str, Any]:
    manifest = read_json(MANIFEST)
    metadata = read_json(RUN / "RUN_METADATA.json")
    gate = read_json(RUN / "SCORING_GATE_VERDICT.json")
    auth = read_json(RUN / "AUTHORIZATION_RECORD.json")
    score = read_json(RUN / "PAIRED_ANALYSIS_GOLD_ONLY.json")
    cohort = read_jsonl(COHORT)
    paired = read_jsonl(RUN / "PAIRED_CASES_NON_GOLD.jsonl")
    trace_indexed = read_jsonl_indexed(RUN / "CASE_TRACES_NON_GOLD.jsonl")
    snapshot = read_jsonl(RUN / "FROZEN_CASE_SNAPSHOTS_NON_GOLD.jsonl")
    gold_rows = read_jsonl(GOLD_SOURCE)

    checks = {
        "cohort_size_800": len(cohort) == 800,
        "paired_size_800": len(paired) == 800,
        "trace_size_800": len(trace_indexed) == 800,
        "snapshot_size_800": len(snapshot) == 800,
        "unique_cohort_ids": len({row["question_key"] for row in cohort}) == 800,
        "unique_paired_ids": len({row["question_key"] for row in paired}) == 800,
        "cohort_hash_matches_manifest": sha256(COHORT) == manifest["inference_cohort_sha256"],
        "source_hash_matches_manifest": sha256(GOLD_SOURCE) == manifest["source_dataset_sha256"],
        "auth_metadata_gate_hashes_match": (
            manifest["manifest_sha256"] == auth.get("manifest_sha256")
            == metadata.get("manifest_sha256") == gate.get("manifest_sha256")
        ),
        "run_complete": metadata.get("status") == "COMPLETE" and metadata.get("cases_paired") == 800,
        "independent_gate_scoreable": gate.get("verdict") == "SCOREABLE" and gate.get("observed_case_count") == 800,
        "gold_scoring_after_gate": (RUN / "PAIRED_ANALYSIS_GOLD_ONLY.json").stat().st_mtime > (RUN / "SCORING_GATE_VERDICT.json").stat().st_mtime,
        "score_case_rows_800": len(score.get("case_results") or []) == 800,
    }
    if not all(checks.values()):
        raise SystemExit(f"integrity checks failed: {checks}")

    gold_by_id = {
        str(row["id"]): row for row in gold_rows
        if str(row.get("split", "")).casefold() == "dev"
    }
    paired_by_key = {row["question_key"]: row for row in paired}
    trace_by_key = {row["question_key"]: (line, row) for line, row in trace_indexed}
    score_by_key = {row["question_key"]: row for row in score["case_results"]}
    request_events_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in read_jsonl(RUN / "REQUEST_EVENTS.jsonl"):
        if event.get("request_role") == "baseline":
            request_events_by_case[str(event.get("case_id"))].append(event)
    _require_keys = set(row["question_key"] for row in cohort)
    if not (_require_keys == set(paired_by_key) == set(trace_by_key) == set(score_by_key)):
        raise SystemExit("integrity checks failed: cohort/run question key sets differ")

    matrix: list[dict[str, Any]] = []
    count_audit: list[dict[str, Any]] = []
    empty_audit: list[dict[str, Any]] = []
    entity_audit: list[dict[str, Any]] = []
    partition = Counter()
    operator_counts = Counter()
    numeric_runtime = Counter()
    entity_group_empty = Counter()
    entity_first_hop = Counter()
    empty_reason = Counter()
    scalar_matches = 0
    query_count_matches = 0
    uncapped_query_path_count = 0

    for cohort_row in cohort:
        key = cohort_row["question_key"]
        dataset_id = key.split("::", 1)[1]
        raw = gold_by_id.get(dataset_id)
        if raw is None or raw.get("question") != cohort_row.get("question"):
            raise SystemExit(f"gold source identity/text mismatch: {key}")
        live = paired_by_key[key]
        line, trace = trace_by_key[key]
        scored = score_by_key[key]
        root_operator = "UNKNOWN"
        operators: list[str] = []
        sexpr = str(raw.get("logical_form") or "")
        try:
            ast = parse_s_expression(sexpr)
            root_operator = ast.operator.upper()
            operators = [node.operator.upper() for node in walk_expr(ast)]
        except Exception:
            operators = ["PARSE_ERROR"]
        answers = raw.get("answers") or []
        answer_ids = list(map(str, raw.get("answer_ids") or []))
        numeric_gold = is_numeric_answer(answers)
        is_count = root_operator == "COUNT"
        query = str(raw.get("gold_sparql") or "")
        has_count_query = bool(re.search(r"\bCOUNT\s*\(", query, re.I))
        comparison_ops = [op for op in operators if op.lower() in {"gt", "ge", "lt", "le", "argmax", "argmin", "argmax_n", "argmin_n"}]
        has_date = bool(re.search(r"\b(?:date|year|day|month)\b|XMLSchema#date|XMLSchema#gYear", query + " " + sexpr, re.I))
        graph_query = (raw.get("source_metadata") or {}).get("graph_query") or {}
        gold_edges = graph_query.get("edges") or []
        gold_relation_ids = [normalize_relation(edge.get("relation", "")) for edge in gold_edges if edge.get("relation")]
        first_edges = gold_topic_first_edges(raw)
        first_candidates = initial_relation_candidates(trace)
        first_available = any(
            direction == wanted_direction and wanted_relation in candidates
            for wanted_relation, wanted_direction in first_edges
            for direction, candidates, _ in first_candidates
        )
        first_selected = any(
            direction == wanted_direction and wanted_relation in candidates and wanted_relation in selected
            for wanted_relation, wanted_direction in first_edges
            for direction, candidates, selected in first_candidates
        )
        actual_first_relation, actual_first_direction = runtime_first_hop(trace)
        first_executed = any(
            actual_first_relation == wanted_relation and actual_first_direction == wanted_direction
            for wanted_relation, wanted_direction in first_edges
        )

        pool_ids = set(map(str, trace.get("candidate_ids") or []))
        gold_in_pool = bool(set(answer_ids) & pool_ids)
        entity_pool_present = bool(pool_ids)
        hit = bool(scored.get("baseline_hits_at_1"))
        pred_ids = list(map(str, trace.get("baseline_prediction_ids") or []))
        pred_names = [str(value).strip() for value in trace.get("baseline_answer_names") or []]
        gold_answers = [str(value).strip() for value in answers]
        entity_labels = [str(item.get("name", "")).strip() for item in raw.get("answer_entities") or [] if isinstance(item, dict)]
        gold_names = {value.casefold() for value in (entity_labels or gold_answers) if value}
        names_overlap_gold = bool(gold_names & {value.casefold() for value in pred_names})

        final_query = final_entity_query(trace)
        final_query_status = str(final_query.get("status", "missing"))
        final_result_count = final_query.get("result_count")
        relation_chain = list(map(normalize_relation, trace.get("executed_relation_ids") or []))
        scalar_value = int_value(answers)
        scalar_prediction = next((int(float(value)) for value in pred_names if NUMERIC.fullmatch(value.strip()) and float(value).is_integer()), None)
        scalar_match = numeric_gold and scalar_value is not None and scalar_prediction == scalar_value
        if scalar_match:
            scalar_matches += 1
        query_count_match = numeric_gold and scalar_value is not None and final_query_status == "ok" and final_result_count == scalar_value
        if query_count_match:
            query_count_matches += 1
        query_uncapped = final_query_status == "ok" and isinstance(final_result_count, int) and final_result_count < 1000 and bool(relation_chain)
        if query_uncapped:
            uncapped_query_path_count += 1

        if hit:
            category = "BASELINE_HIT"
            boundary = "NONE"
            evidence = "DIRECT"
            pool_group = "not_applicable_hit"
        elif numeric_gold:
            category = "NUMERIC_GOLD_MISS"
            pool_group = "NUMERIC_ANSWER"
            if is_count and scalar_match:
                boundary = "SCALAR_ANSWER_PRESENT_BUT_NOT_REPRESENTED_IN_ENTITY_ID_PREDICTION"
                evidence = "DIRECT"
            elif is_count and final_query_status == "error":
                boundary = "COUNT_PATH_QUERY_ERROR_AND_NUMERIC_OUTPUT_CONTRACT"
                evidence = "DIRECT"
            elif is_count:
                boundary = "COUNT_VALUE_OR_TYPED_SCALAR_OUTPUT_FAILURE"
                evidence = "STRONG"
            else:
                boundary = "NUMERIC_GOLD_OPERATOR_NOT_COUNT_OR_ANNOTATION_MISMATCH"
                evidence = "DIRECT" if root_operator != "UNKNOWN" else "LIMITED"
        elif gold_in_pool:
            pool_group = "GOLD_IN_RECORDED_CANDIDATE_POOL"
            boundary = "TERMINAL_ANSWER_SELECTION_OR_MAPPING"
            evidence = "DIRECT"
        elif not entity_pool_present:
            pool_group = "NO_ENTITY_CANDIDATE_POOL"
            if final_query_status == "error" or trace.get("candidate_query_status") == "error":
                boundary = "SPARQL_EXECUTION_ERROR_BEFORE_CANDIDATE_SET"
                evidence = "DIRECT"
            elif final_query_status == "ok" and final_query.get("filter_caused_empty"):
                boundary = "ENTITY_TYPE_OR_LABEL_FILTER_REMOVED_ALL_RESULTS"
                evidence = "DIRECT"
            elif final_query_status == "ok" and final_result_count == 0:
                boundary = "SUCCESSFUL_EMPTY_TERMINAL_ENTITY_QUERY"
                evidence = "DIRECT_RESULT_UNKNOWN_CAUSE"
            elif not relation_chain:
                boundary = "NO_GROUNDED_TERMINAL_RELATION_PATH_OR_DIRECT_GENERATION"
                evidence = "DIRECT_TRACE_WITH_UNLOCALIZED_CAUSE"
            else:
                boundary = "NO_RECORDED_CANDIDATES_WITH_INCOMPLETE_QUERY_PROVENANCE"
                evidence = "LIMITED"
            pool_group = "NO_ENTITY_CANDIDATE_POOL"
        else:
            pool_group = "GOLD_ABSENT_FROM_RECORDED_CANDIDATE_POOL"
            if first_available and not first_selected:
                boundary = "FIRST_REQUIRED_RELATION_OMITTED_BY_RELATION_PRUNING"
                evidence = "DIRECT_PROMPT_CANDIDATE_AND_SELECTION_TRACE"
            elif not first_available:
                boundary = "FIRST_REQUIRED_RELATION_ABSENT_FROM_LOGGED_INITIAL_PROMPT_CANDIDATES"
                evidence = "DIRECT_PROMPT_CAPTURE; KG_CANDIDATE_ROW_DETAILS_NOT_PERSISTED"
            elif first_selected and not first_executed:
                boundary = "SELECTED_FIRST_RELATION_NOT_EXECUTED_AS_REQUIRED_PATH_EDGE"
                evidence = "STRONG; DIRECTION_OR_DFS_TRANSITION_NOT_FULLY_INSTRUMENTED"
            elif first_executed:
                boundary = "LATER_PATH_STOP_JOIN_OR_TERMINAL_QUERY_DIVERGENCE"
                evidence = "STRONG; GOLD_ABSENCE_FROM_FINAL_POOL_IS_DIRECT"
            else:
                boundary = "UPSTREAM_PATH_OR_RELATION_GROUNDING_DIVERGENCE"
                evidence = "LIMITED"

        partition[pool_group] += int(not hit)
        if not hit and not numeric_gold:
            entity_first_hop[boundary] += 1
            entity_group_empty[(pool_group, not bool(pred_ids))] += 1

        runtime_type = next((str(item.get("requested_final_answer_type", "")) for item in trace.get("prompt_history") or []
                             if item.get("prompt_version") == "adaptive-routing-v1.1.3-answer-type-aware-json-v1"), "")
        path_len = len(relation_chain)
        query_capped = isinstance(final_result_count, int) and final_result_count >= 1000
        evidence_pointer = f"RUNS/iter29_terminal_candidate_selector_attempt3/CASE_TRACES_NON_GOLD.jsonl:{line}"
        row = {
            "case_id": trace.get("case_id"),
            "question_key": key,
            "question": raw.get("question"),
            "gold_only_analytic": True,
            "gold_answers_scoring_only": json.dumps(answers, ensure_ascii=False),
            "gold_answer_ids_scoring_only": json.dumps(answer_ids, ensure_ascii=False),
            "gold_answer_type_analytic": "numeric_literal" if numeric_gold else "entity_or_non_numeric",
            "gold_operator_root_scoring_only": root_operator,
            "gold_operators_scoring_only": json.dumps(operators),
            "gold_has_count_aggregate_scoring_only": has_count_query,
            "gold_comparison_operators_scoring_only": json.dumps(comparison_ops),
            "gold_date_or_year_constraint_scoring_only": has_date,
            "gold_logical_form_scoring_only": sexpr,
            "gold_sparql_scoring_only": query,
            "baseline_hits_at_1_scoring_only": hit,
            "numeric_gold_miss": bool(not hit and numeric_gold),
            "failure_partition": pool_group,
            "gold_in_recorded_pool_scoring_only": gold_in_pool if not numeric_gold else None,
            "baseline_prediction_ids": json.dumps(pred_ids, ensure_ascii=False),
            "baseline_answer_names_gold_only_comparison": json.dumps(pred_names, ensure_ascii=False),
            "scalar_answer_exact_match_scoring_only": bool(scalar_match) if numeric_gold else None,
            "gold_query_result_count_match_scoring_only": bool(query_count_match) if numeric_gold else None,
            "baseline_ids_empty": not bool(pred_ids),
            "baseline_status": trace.get("baseline_status"),
            "baseline_failure_kind": trace.get("baseline_failure_kind"),
            "runtime_route": trace.get("route"),
            "runtime_answer_type_hint": runtime_type,
            "runtime_terminal_source": trace.get("terminal_source"),
            "runtime_last_reasoning_action": last_reasoning_action(trace),
            "runtime_relation_chain_ids": json.dumps(relation_chain),
            "runtime_relation_chain_length": path_len,
            "runtime_candidate_ids": json.dumps(trace.get("candidate_ids") or []),
            "runtime_candidate_labels": json.dumps(trace.get("candidate_labels") or [], ensure_ascii=False),
            "runtime_literal_candidates": json.dumps(trace.get("literal_candidates") or [], ensure_ascii=False),
            "runtime_candidate_query_status": trace.get("candidate_query_status"),
            "runtime_final_entity_query_status": final_query_status,
            "runtime_final_entity_query_result_count": final_result_count,
            "runtime_final_query_at_1000_row_cap": query_capped,
            "runtime_uncapped_successful_path_count_available": query_uncapped,
            "gold_first_hop_relation_candidates": json.dumps(first_edges),
            "runtime_first_hop_gold_relation_candidate_available": first_available,
            "runtime_first_hop_gold_relation_selected": first_selected,
            "runtime_first_hop_gold_relation_executed": first_executed,
            "first_failure_boundary": boundary,
            "evidence_tier": evidence,
            "trace_pointer": evidence_pointer,
        }
        matrix.append(row)

        if numeric_gold:
            target_class_filter = bool(re.search(r"type\.object\.type", query, re.I))
            target_class_nodes = [node for node in (graph_query.get("nodes") or []) if node.get("question_node") and node.get("node_type") == "class"]
            relation_set_match = set(gold_relation_ids) == set(relation_chain)
            count_audit.append({
                "question_key": key,
                "question": raw.get("question"),
                "gold_only_analytic": True,
                "gold_value_scoring_only": answers[0] if len(answers) == 1 else json.dumps(answers),
                "gold_answer_ids_scoring_only": json.dumps(answer_ids),
                "gold_sexpr_scoring_only": sexpr,
                "gold_operator_root": root_operator,
                "gold_operator_nodes": json.dumps(operators),
                "gold_query_uses_count": has_count_query,
                "gold_query_uses_select_distinct_subquery": bool(re.search(r"SELECT\s+DISTINCT", query, re.I)),
                "gold_query_has_filter": bool(re.search(r"\bFILTER\s*\([^)]*!=", query, re.I)),
                "gold_query_inequality_pair_count": len(re.findall(r"!=", query)),
                "gold_comparison_operators": json.dumps(comparison_ops),
                "gold_date_or_year_constraint": has_date,
                "gold_type_filter_present": target_class_filter,
                "gold_target_class_id_scoring_only": (target_class_nodes[0].get("id") if target_class_nodes else None),
                "gold_path_relation_ids_scoring_only": json.dumps(gold_relation_ids),
                "gold_binary_edge_count": len(gold_edges),
                "requires_shared_variable_join": len(gold_edges) > 1,
                "gold_graph_max_node_degree": max((sum(edge.get(side) == node.get("nid") for edge in gold_edges for side in ("start", "end")) for node in (graph_query.get("nodes") or [])), default=0),
                "topic_entity_count": len(raw.get("topic_entities") or []),
                "runtime_count_intent_phrase": classify_count_intent(str(raw.get("question") or ""))[0],
                "runtime_count_intent_reason": classify_count_intent(str(raw.get("question") or ""))[1],
                "runtime_answer_type_hint": runtime_type,
                "runtime_path_relation_ids": json.dumps(relation_chain),
                "runtime_relation_set_matches_gold_scoring_only": relation_set_match,
                "runtime_path_length": path_len,
                "runtime_final_query_status": final_query_status,
                "runtime_entity_search_result_count": final_result_count,
                "runtime_query_result_count_below_1000": isinstance(final_result_count, int) and final_result_count < 1000,
                "runtime_count_rows_equals_gold_scoring_only": bool(query_count_match),
                "runtime_count_rows_exact_scalar_answer_scoring_only": bool(scalar_match),
                "runtime_final_query_has_type_filter": bool(re.search(r"type\.object\.type", str(final_query.get("sparql", "")), re.I)),
                "runtime_baseline_prediction_ids_empty": not bool(pred_ids),
                "runtime_baseline_answer_names": json.dumps(pred_names, ensure_ascii=False),
                "trace_pointer": evidence_pointer,
                "interpretation": "Gold-only comparison of persisted entity-search row count and scalar text; not a method result. Candidate query can be limited to 1000 rows and does not add the gold target type filter.",
            })

        if not hit and not pred_ids:
            request_events = request_events_by_case.get(str(trace.get("case_id")), [])
            finish_reasons = Counter(str(event.get("finish_reason", "missing")) for event in request_events)
            if numeric_gold:
                empty_category = "NUMERIC_SCALAR_GOLD_TEXT_NOT_TYPED_AS_PREDICTION_ID" if scalar_match else "NUMERIC_COUNT_VALUE_OR_TYPED_OUTPUT_FAILURE"
            elif not pool_ids:
                empty_category = "NO_RECORDED_ENTITY_CANDIDATE_POOL"
            else:
                empty_category = "CANDIDATE_POOL_PRESENT_BUT_NO_ENTITY_ID_PREDICTION"
            empty_reason[empty_category] += 1
            empty_audit.append({
                "question_key": key,
                "question": raw.get("question"),
                "gold_only_analytic": True,
                "gold_numeric_answer": numeric_gold,
                "gold_answer_values_scoring_only": json.dumps(answers, ensure_ascii=False),
                "gold_operator_root_scoring_only": root_operator,
                "gold_answer_ids_scoring_only": json.dumps(answer_ids, ensure_ascii=False),
                "baseline_prediction_ids": json.dumps(pred_ids),
                "baseline_answer_names_gold_only_comparison": json.dumps(pred_names, ensure_ascii=False),
                "scalar_output_matches_gold_scoring_only": bool(scalar_match) if numeric_gold else None,
                "candidate_ids_count": len(trace.get("candidate_ids") or []),
                "literal_candidates_count": len(trace.get("literal_candidates") or []),
                "candidate_query_status": trace.get("candidate_query_status"),
                "final_entity_query_status": final_query_status,
                "final_entity_query_result_count": final_result_count,
                "baseline_response_nonempty": (trace.get("baseline_response") or {}).get("content_nonempty"),
                "baseline_response_finish_reason": (trace.get("baseline_response") or {}).get("finish_reason"),
                "baseline_request_finish_reasons": json.dumps(dict(finish_reasons)),
                "terminal_source": trace.get("terminal_source"),
                "empty_prediction_reason": empty_category,
                "trace_pointer": evidence_pointer,
            })

        if not hit and not numeric_gold:
            group = pool_group
            entity_audit.append({
                "question_key": key,
                "question": raw.get("question"),
                "gold_only_analytic": True,
                "gold_answer_ids_scoring_only": json.dumps(answer_ids, ensure_ascii=False),
                "gold_answers_scoring_only": json.dumps(answers, ensure_ascii=False),
                "failure_partition": group,
                "gold_in_candidate_pool_scoring_only": gold_in_pool,
                "prediction_ids_empty": not bool(pred_ids),
                "gold_name_appears_in_baseline_answer_names_scoring_only": names_overlap_gold,
                "baseline_answer_names_scoring_only": json.dumps(pred_names, ensure_ascii=False),
                "candidate_ids_count": len(pool_ids),
                "candidate_query_status": trace.get("candidate_query_status"),
                "final_entity_query_status": final_query_status,
                "final_entity_query_result_count": final_result_count,
                "terminal_source": trace.get("terminal_source"),
                "last_reasoning_action": last_reasoning_action(trace),
                "gold_path_relation_ids_scoring_only": json.dumps(gold_relation_ids),
                "runtime_path_relation_ids": json.dumps(relation_chain),
                "runtime_relation_set_matches_gold_scoring_only": set(gold_relation_ids) == set(relation_chain),
                "first_gold_relation_candidate_available": first_available,
                "first_gold_relation_selected": first_selected,
                "first_gold_relation_executed": first_executed,
                "first_failure_boundary": boundary,
                "evidence_tier": evidence,
                "trace_pointer": evidence_pointer,
            })

    # Save independent case-level ledgers. Gold columns are explicitly tagged and
    # live inference modules do not import or read these files.
    OUT.mkdir(parents=True, exist_ok=True)
    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    write_csv(OUT / "FAILURE_MECHANISM_MATRIX.csv", matrix)
    write_csv(OUT / "COUNT_OPERATOR_AUDIT.csv", count_audit)
    write_csv(OUT / "ENTITY_MISS_TRACE_AUDIT.csv", entity_audit)
    write_csv(OUT / "EMPTY_PREDICTION_AUDIT.csv", empty_audit)

    entity_groups = Counter(row["failure_partition"] for row in entity_audit)
    entity_empty_cross = Counter((row["failure_partition"], row["prediction_ids_empty"]) for row in entity_audit)
    count_filter_classes = Counter()
    for row in count_audit:
        if row["gold_date_or_year_constraint"] and row["gold_comparison_operators"] != "[]":
            count_filter_classes["COUNT_WITH_DATE_OR_YEAR_COMPARISON"] += 1
        elif row["gold_comparison_operators"] != "[]":
            count_filter_classes["COUNT_WITH_NONDATE_COMPARISON"] += 1
        elif row["gold_date_or_year_constraint"]:
            count_filter_classes["COUNT_WITH_DATE_OR_YEAR_MENTION"] += 1
        else:
            count_filter_classes["COUNT_WITHOUT_DETECTED_COMPARISON"] += 1
    numeric_runtime["numeric_count_cases"] = len(count_audit)
    numeric_runtime["count_intent_triggered"] = sum(bool(row["runtime_count_intent_phrase"]) for row in count_audit)
    numeric_runtime["count_intent_not_triggered"] = len(count_audit) - numeric_runtime["count_intent_triggered"]
    numeric_runtime["router_answer_type_number_count_amount"] = sum(
        str(row["runtime_answer_type_hint"]).strip().casefold() in {"number", "count", "quantity", "amount"}
        or str(row["runtime_answer_type_hint"]).strip().casefold().startswith("count of ")
        for row in count_audit
    )
    numeric_runtime["runtime_relation_path_nonempty"] = sum(bool(json.loads(row["runtime_path_relation_ids"])) for row in count_audit)
    numeric_runtime["successful_uncapped_path_query"] = sum(row["runtime_query_result_count_below_1000"] and row["runtime_final_query_status"] == "ok" and row["runtime_path_length"] > 0 for row in count_audit)
    numeric_runtime["gold_scalar_text_matches_current_baseline_name_scoring_only"] = sum(bool(row["runtime_count_rows_exact_scalar_answer_scoring_only"]) for row in count_audit)
    numeric_runtime["persisted_entity_search_rows_equal_gold_count_scoring_only"] = sum(bool(row["runtime_count_rows_equals_gold_scoring_only"]) for row in count_audit)
    numeric_runtime["multi_edge_gold_paths_need_shared_variables"] = sum(bool(row["requires_shared_variable_join"]) for row in count_audit)
    numeric_runtime["runtime_entity_search_includes_gold_type_filter"] = sum(bool(row["runtime_final_query_has_type_filter"]) for row in count_audit)
    numeric_runtime["gold_queries_include_target_type_filter"] = sum(bool(row["gold_type_filter_present"]) for row in count_audit)
    numeric_runtime["gold_queries_with_select_distinct_subquery"] = sum(bool(row["gold_query_uses_select_distinct_subquery"]) for row in count_audit)
    numeric_runtime["gold_queries_with_node_filter"] = sum(bool(row["gold_query_has_filter"]) for row in count_audit)
    numeric_runtime["gold_queries_with_all_pairwise_node_inequalities"] = sum(
        row["gold_query_inequality_pair_count"] == (int(row["gold_binary_edge_count"]) + 1) * int(row["gold_binary_edge_count"]) // 2
        for row in count_audit
    )
    numeric_runtime["gold_path_relation_set_matches_runtime_path_scoring_only"] = sum(bool(row["runtime_relation_set_matches_gold_scoring_only"]) for row in count_audit)
    target_class_ids = {str(row["gold_target_class_id_scoring_only"]) for row in count_audit if row["gold_target_class_id_scoring_only"]}

    summary = {
        "status": "OFFLINE_PHASE_A_COMPLETE",
        "evidence_class": "ITER29_GRAILQA_DEV_POSTHOC_ANALYSIS_NOT_LIVE_METHOD_RESULT",
        "gold_only_analytic": True,
        "network_calls": 0,
        "integrity_checks": checks,
        "source_hashes": {
            "manifest_declared_sha256": manifest.get("manifest_sha256"),
            "cohort_sha256": sha256(COHORT),
            "source_dataset_sha256": sha256(GOLD_SOURCE),
            "score_summary_sha256": sha256(RUN / "PAIRED_ANALYSIS_GOLD_ONLY.json"),
            "score_case_csv_sha256": sha256(RUN / "PAIRED_EVAL_CASES_GOLD_ONLY.csv"),
        },
        "n": {"cohort": len(cohort), "paired": len(paired), "unique_ids": len({row["question_key"] for row in cohort})},
        "primary_recomputed": {
            "baseline_hits": sum(bool(row["baseline_hits_at_1_scoring_only"]) for row in matrix),
            "intervention_hits": sum(bool(score_by_key[row["question_key"]]["intervention_hits_at_1"]) for row in matrix),
            "baseline_misses": sum(not bool(row["baseline_hits_at_1_scoring_only"]) for row in matrix),
            "numeric_gold_misses": sum(bool(row["numeric_gold_miss"]) for row in matrix),
            "entity_answer_misses": sum(not row["baseline_hits_at_1_scoring_only"] and row["gold_answer_type_analytic"] != "numeric_literal" for row in matrix),
            "failure_partition_counts_exclusive": dict(partition),
        },
        "numeric_operator_taxonomy": {
            "root_operator_counts": dict(Counter(row["gold_operator_root"] for row in count_audit)),
            "gold_query_count_aggregate_count": sum(bool(row["gold_query_uses_count"]) for row in count_audit),
            "comparison_date_strata": dict(count_filter_classes),
            "gold_path_edge_count_distribution": dict(Counter(row["gold_binary_edge_count"] for row in count_audit)),
            "gold_path_max_degree_distribution": dict(Counter(row["gold_graph_max_node_degree"] for row in count_audit)),
            "unique_gold_target_class_count_scoring_only": len(target_class_ids),
            "requires_shared_variable_join": numeric_runtime["multi_edge_gold_paths_need_shared_variables"],
            "target_type_filter_in_gold_count": numeric_runtime["gold_queries_include_target_type_filter"],
            "target_type_filter_in_current_runtime_query": numeric_runtime["runtime_entity_search_includes_gold_type_filter"],
            "gold_queries_with_select_distinct_subquery": numeric_runtime["gold_queries_with_select_distinct_subquery"],
            "gold_queries_with_node_inequality_filter": numeric_runtime["gold_queries_with_node_filter"],
            "gold_queries_with_all_pairwise_node_inequalities": numeric_runtime["gold_queries_with_all_pairwise_node_inequalities"],
            "runtime_relation_set_matches_gold_scoring_only": numeric_runtime["gold_path_relation_set_matches_runtime_path_scoring_only"],
            "gold_scalar_text_exact_matches_scoring_only": scalar_matches,
            "persisted_entity_search_row_count_matches_gold_count_scoring_only": query_count_matches,
            "successful_uncapped_grounded_path_query_count": numeric_runtime["successful_uncapped_path_query"],
            "runtime_features": dict(numeric_runtime),
            "interpretation": "The answer strings and persisted row counts are gold-compared offline diagnostics only. They are not rescored method outputs; current entity query omits the Gold target type filter and can stop at the 1000-row cap.",
        },
        "entity_miss_taxonomy": {
            "candidate_pool_groups": dict(entity_groups),
            "prediction_empty_by_pool_group": {f"{group}|empty={empty}": count for (group, empty), count in entity_empty_cross.items()},
            "boundary_counts": dict(entity_first_hop),
            "gold_present_pool_misses_have_candidate_query_status": dict(Counter(row["candidate_query_status"] for row in entity_audit if row["failure_partition"] == "GOLD_IN_RECORDED_CANDIDATE_POOL")),
            "interpretation": "Gold presence in a recorded pool supports a terminal-selection diagnosis. Gold absence from a recorded pool does not establish Freebase absence; persisted traces lack raw relation-search rows, so the precise discovery-vs-query boundary remains unknown for some cases.",
        },
        "empty_prediction_taxonomy": {
            "empty_prediction_count": len(empty_audit),
            "category_counts": dict(empty_reason),
            "all_empty_entity_candidate_ids_absent": sum(not row["candidate_ids_count"] for row in empty_audit),
            "interpretation": "Numeric scalar text, entity candidate availability, and response/parser/terminal conditions are separated. A numeric text match is a scoring-only observation, not a prediction repair.",
        },
        "artifacts": {
            "failure_matrix": str((OUT / "FAILURE_MECHANISM_MATRIX.csv").relative_to(ROOT)),
            "count_audit": str((OUT / "COUNT_OPERATOR_AUDIT.csv").relative_to(ROOT)),
            "entity_audit": str((OUT / "ENTITY_MISS_TRACE_AUDIT.csv").relative_to(ROOT)),
            "empty_audit": str((OUT / "EMPTY_PREDICTION_AUDIT.csv").relative_to(ROOT)),
        },
    }
    (OUT / "PHASE_A_AUDIT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2, sort_keys=True))
