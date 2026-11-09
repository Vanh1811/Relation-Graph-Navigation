"""Gold-free terminal-boundary observer for CoR/TGAR diagnostic runs.

This module reads only already captured runtime outputs. It does not import the
KG backend, model client, dataset scorer, or any transport implementation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


AGGREGATE_QUESTION = re.compile(
    r"\b(how many|how much|number of|total number|count of|average|sum of|maximum|minimum)\b",
    re.IGNORECASE,
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _entity_search_before(
    sparql_history: list[dict[str, Any]], event_trace_id: int | None,
) -> dict[str, Any] | None:
    searches = [
        event for event in sparql_history
        if event.get("search_type") == "entity_search"
        and isinstance(event.get("trace_id"), int)
        and (event_trace_id is None or event["trace_id"] < event_trace_id)
    ]
    return searches[-1] if searches else None


def _request_summaries(request_events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    starts: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for event in request_events:
        request_hash = str(event.get("request_sha256", ""))
        if event.get("event") == "REQUEST_STARTED":
            starts[request_hash] = event
            continue
        if event.get("event") != "REQUEST_FINISHED":
            continue
        start = starts.get(request_hash, {})
        raw = event.get("raw_response_body")
        content_chars: int | None = None
        parsed_finish = event.get("finish_reason")
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
                choice = (payload.get("choices") or [{}])[0]
                parsed_finish = choice.get("finish_reason", parsed_finish)
                content = choice.get("message", {}).get("content")
                content_chars = len(content) if isinstance(content, str) else 0
            except (json.JSONDecodeError, TypeError, AttributeError):
                content_chars = None
        ordered.append({
            "request_sha256": request_hash,
            "started_at_utc": start.get("recorded_at_utc"),
            "finished_at_utc": event.get("recorded_at_utc"),
            "estimated_input_tokens": start.get("estimated_input_tokens"),
            "max_completion_tokens": (start.get("request_body") or {}).get("max_completion_tokens"),
            "status": event.get("status"),
            "finish_reason": parsed_finish,
            "content_chars": content_chars,
        })
    return ordered


def observe_terminal_boundary(
    *,
    case_id: str,
    question_key: str,
    baseline_status: str,
    route: str,
    result: dict[str, Any],
    prompt_history: list[dict[str, Any]],
    sparql_history: list[dict[str, Any]],
    request_events: list[dict[str, Any]] | None = None,
    question: str = "",
    legacy_gate_reason: str = "",
    captured_candidate_ids: list[str] | None = None,
    snapshot_candidate_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Return an auditable, prompt-free terminal record from one completed row.

    Candidate membership is derived only from the latest successful entity
    search that precedes the final CoR decision. Names are mapped exactly; the
    observer never guesses or normalizes labels.
    """
    prompts = list(prompt_history)
    requests = _request_summaries(request_events or [])
    last_prompt = prompts[-1] if prompts else {}
    last_type = str(last_prompt.get("type", ""))
    last_action = str(last_prompt.get("parsed_result", ""))
    answer_event = last_prompt
    terminal_source = "unknown"
    next_path = "unknown"

    final_mode = str(result.get("final_execution_mode", ""))
    has_branch_merge = (
        final_mode == "branch"
        or "branch_traces" in result
        or "merged_candidate_count" in result
        or str(result.get("action", "")).casefold() == "branch_merge"
    )
    if baseline_status != "BASELINE_COMPLETED" or str(result.get("action", "")).upper() == "ERROR":
        terminal_source = "empty/error"
        if (
            last_type == "reasoning"
            and last_action.casefold() in {"constraint", "filter"}
            and requests
            and requests[-1].get("finish_reason") == "length"
        ):
            next_path = "filter-request-failed-before-filter-event-append"
    elif has_branch_merge:
        terminal_source = "branch-merge"
    elif last_type == "filter_answer":
        terminal_source = "graph-filter"
    elif last_type == "reasoning" and last_action.casefold() == "stop":
        terminal_source = "direct-LLM-reasoning-STOP"
    elif last_type == "generate_without_explored_paths" or str(result.get("action", "")).casefold() == "generate_directly":
        terminal_source = "direct-LLM-search-exhaustion-fallback"
    elif last_type == "reasoning" and last_action.casefold() in {"constraint", "filter"}:
        failed = bool(requests and requests[-1].get("finish_reason") == "length")
        if failed:
            terminal_source = "empty/error"
            next_path = "filter-request-failed-before-filter-event-append"
        else:
            terminal_source = "filter-path-instrumentation-uncertain"
            next_path = "filter"
    else:
        terminal_source = "empty/error" if not result.get("results") else "unclassified-result"

    trace_id = last_prompt.get("trace_id") if isinstance(last_prompt.get("trace_id"), int) else None
    search = _entity_search_before(sparql_history, trace_id)
    candidate_ids = list(search.get("candidate_entity_ids") or []) if search else []
    candidate_labels = list(search.get("candidate_entity_names") or []) if search else []
    literals = list(search.get("literal_candidates") or []) if search else []
    predictions = [
        str(item.get("id", "")).strip()
        for item in (result.get("results") if isinstance(result.get("results"), list) else [])
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    ]
    answer_names = [str(x).strip() for x in (answer_event.get("answers") or []) if str(x).strip()]
    label_counts: dict[str, int] = {}
    for label in candidate_labels:
        label_counts[str(label).strip()] = label_counts.get(str(label).strip(), 0) + 1
    label_to_id: dict[str, str] = {}
    if len(candidate_ids) == len(candidate_labels):
        for candidate_id, label in zip(candidate_ids, candidate_labels):
            if label_counts.get(str(label).strip()) == 1:
                label_to_id[str(label).strip()] = str(candidate_id).strip()
    mapped_answer_ids = [label_to_id.get(label, "") for label in answer_names]
    ids_aligned = bool(candidate_ids) and len(candidate_ids) == len(candidate_labels)
    ids_unique = len(candidate_ids) == len(set(candidate_ids))
    labels_unique = len(candidate_labels) == len(set(candidate_labels))
    answer_mapping_complete = bool(answer_names) and all(mapped_answer_ids)
    output_ids_are_candidates = bool(predictions) and set(predictions).issubset(set(candidate_ids))
    candidate_set_valid = (
        bool(search)
        and search.get("status") == "ok"
        and not search.get("timed_out", False)
        and ids_aligned
        and ids_unique
        and labels_unique
        and not literals
    )
    answer_id_mapping_valid = (
        answer_mapping_complete
        and set(mapped_answer_ids) == set(predictions)
        and output_ids_are_candidates
    )
    filter_events = [event for event in prompts if event.get("type") == "filter_answer"]
    terminal_event_kind = terminal_source
    aggregate_lexical_gate = bool(AGGREGATE_QUESTION.search(question))
    selector_eligible = bool(
        baseline_status == "BASELINE_COMPLETED"
        and route.casefold() == "cor"
        and terminal_source in {"direct-LLM-reasoning-STOP", "graph-filter"}
        and candidate_set_valid
        and len(candidate_ids) >= 2
        and answer_id_mapping_valid
        and not aggregate_lexical_gate
        and len(filter_events) <= 1
    )

    query = str(search.get("sparql", "")) if search else ""
    relation_tokens = sorted({
        value for value in re.findall(r"\bns:([A-Za-z0-9_.]+)", query)
        if not value.startswith("m.")
    })
    candidate_objects = [
        {"id": str(entity_id), "label": str(label), "position": index}
        for index, (entity_id, label) in enumerate(zip(candidate_ids, candidate_labels))
    ] if len(candidate_ids) == len(candidate_labels) else []
    candidate_hash = sha256_text(_compact_json(candidate_objects)) if candidate_objects else ""
    prompt_types = [str(event.get("type", "")) for event in prompts]
    prompt_event_order = ">".join(
        f"{event.get('type', '')}@{event.get('trace_id', '?')}" for event in prompts
    )
    filter_event_order = ";".join(
        f"{index}:{event.get('trace_id', '?')}" for index, event in enumerate(prompts)
        if event.get("type") == "filter_answer"
    )
    request_event_gap = len(requests) - len(prompts)
    last_request = requests[-1] if requests else {}
    source_trace_id = search.get("trace_id") if search else None
    source_query_hash = sha256_text(query) if query else ""

    return {
        "case_id": case_id,
        "question_key": question_key,
        "route": route,
        "final_execution_mode_observed": final_mode or ("cor_inferred_from_router_route" if route == "cor" else "not_persisted"),
        "terminal_answer_source": terminal_source,
        "next_path_or_failure": next_path,
        "baseline_status": baseline_status,
        "last_prompt_type": last_type,
        "last_prompt_action": last_action,
        "prompt_event_count": len(prompts),
        "prompt_event_order": prompt_event_order,
        "filter_event_count": len(filter_events),
        "filter_event_order": filter_event_order,
        "request_count": len(requests),
        "request_prompt_event_gap": request_event_gap,
        "last_request_status": last_request.get("status", ""),
        "last_request_finish_reason": last_request.get("finish_reason", ""),
        "last_request_input_tokens": last_request.get("estimated_input_tokens", ""),
        "last_request_completion_cap": last_request.get("max_completion_tokens", ""),
        "terminal_event_trace_id": trace_id,
        "candidate_source_search_trace_id": source_trace_id,
        "candidate_source_query_sha256": source_query_hash,
        "candidate_source_relations": _compact_json(relation_tokens),
        "terminal_binding_available": bool(search and search.get("status") == "ok" and candidate_ids),
        "candidate_id_set_status": (
            "VALID_ALIGNED_ENTITY_SET" if candidate_set_valid else
            "LITERAL_ONLY_OR_MIXED" if literals else
            "NO_ENTITY_IDS" if not candidate_ids else
            "INVALID_OR_AMBIGUOUS_ID_LABEL_MAPPING"
        ),
        "candidate_count": len(candidate_ids),
        "candidate_ids": _compact_json(candidate_ids),
        "candidate_labels": _compact_json(candidate_labels),
        "literal_candidate_count": len(literals),
        "candidate_snapshot_sha256_recomputed": candidate_hash,
        "candidate_snapshot_ids_captured_by_iter27": _compact_json(captured_candidate_ids or []),
        "candidate_snapshot_ids_in_frozen_row": _compact_json(snapshot_candidate_ids or []),
        "baseline_prediction_ids": _compact_json(predictions),
        "terminal_answer_names": _compact_json(answer_names),
        "answer_id_mapping_valid": answer_id_mapping_valid,
        "candidate_selector_strict_eligible": selector_eligible,
        "aggregate_lexical_gate": aggregate_lexical_gate,
        "legacy_gate_reason": legacy_gate_reason,
        "fallback_path": (
            "COR_ROUTE"
            if route.casefold() == "cor" and terminal_source != "direct-LLM-search-exhaustion-fallback"
            else terminal_source
        ),
        "evidence_pointer": f"CASE_TRACES.jsonl#{case_id}; REQUEST_JOURNAL.jsonl#{case_id}; source=chain_of_relations/methods/cor/agent.py",
    }


def build_iter27_matrix(repo_root: Path, output_csv: Path) -> dict[str, Any]:
    run_dir = repo_root / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_027_results_first_answer_repair/RUNS/iter27_scv_empty_only"
    traces = _read_jsonl(run_dir / "CASE_TRACES.jsonl")
    paired_rows = {row["case_id"]: row for row in _read_jsonl(run_dir / "PAIRED_RESULTS_NON_GOLD.jsonl")}
    snapshots = {row["case_id"]: row for row in _read_jsonl(run_dir / "FROZEN_CASE_SNAPSHOTS_NON_GOLD.jsonl")}
    requests_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(run_dir / "REQUEST_JOURNAL.jsonl"):
        case_id = row.get("case_id")
        if case_id:
            requests_by_case.setdefault(str(case_id), []).append(row)
    cohort_path = repo_root / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_027_results_first_answer_repair/INFERENCE_COHORT_NON_GOLD.jsonl"
    questions = {str(row.get("question_key")): str(row.get("question", "")) for row in _read_jsonl(cohort_path)}
    matrix: list[dict[str, Any]] = []
    for trace in traces:
        case_id = str(trace["case_id"])
        paired = paired_rows[case_id]
        snap = snapshots[case_id]
        # The captured live baseline is represented by IDs in the non-gold pair row.
        result = {
            "action": trace.get("prompt_history", [{}])[-1].get("parsed_result", "") if trace.get("prompt_history") else "ERROR",
            "results": [{"id": value} for value in paired.get("baseline_prediction_ids", [])],
            "final_execution_mode": "",
        }
        matrix.append(observe_terminal_boundary(
            case_id=case_id,
            question_key=str(trace.get("question_key", "")),
            baseline_status=str(trace.get("baseline_status", "")),
            route=str(trace.get("router_execution_mode", "")),
            result=result,
            prompt_history=list(trace.get("prompt_history") or []),
            sparql_history=list(trace.get("sparql_history") or []),
            request_events=requests_by_case.get(case_id, []),
            question=questions.get(str(trace.get("question_key", "")), ""),
            legacy_gate_reason=str(trace.get("scv_eligibility_reason", "")),
            captured_candidate_ids=list(paired.get("candidate_ids_snapshot") or []),
            snapshot_candidate_ids=[str(item.get("id", "")) for item in (snap.get("candidate_snapshot") or [])],
        ))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(matrix[0]) if matrix else []
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(matrix)
    return {"rows": matrix, "row_count": len(matrix)}
