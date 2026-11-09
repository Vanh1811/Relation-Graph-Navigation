#!/usr/bin/env python3
"""Freeze gold-free Iteration 32 scalar serialization over Iteration 29 traces."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from chain_of_relations.methods.iter_032_scalar_contract import RequestEvidence, serialize_callsite_result


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS"
RUN = RESULTS / "iter_028_terminal_boundary_repair/RUNS/iter29_terminal_candidate_selector_attempt3"
ITER31 = RESULTS / "iter_031_gold_free_count_readiness"
OUTPUT = RESULTS / "iter_032_count_producer_and_scalar_contract"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_count_keys() -> set[str]:
    path = ITER31 / "TARGET_TYPE_RESOLVER_AUDIT.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if any("gold" in name.casefold() for name in reader.fieldnames or []):
            raise RuntimeError("runtime key-selection source unexpectedly contains gold columns")
        return {str(row["question_key"]) for row in reader}


def request_evidence_by_prompt() -> tuple[dict[tuple[str, str], RequestEvidence], dict[tuple[str, str], int]]:
    started = read_jsonl(RUN / "REQUEST_JOURNAL.jsonl")
    finished = read_jsonl(RUN / "REQUEST_EVENTS.jsonl")
    user_hashes: dict[tuple[str, str], list[str]] = defaultdict(list)
    for event in started:
        if event.get("event") != "REQUEST_STARTED":
            continue
        messages = event.get("request_body", {}).get("messages", [])
        user_text = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        )
        user_hashes[(str(event.get("case_id", "")), str(event.get("prompt_sha256", "")))].append(
            sha256_bytes(user_text.encode("utf-8"))
        )

    evidence_candidates: dict[tuple[str, str], list[RequestEvidence]] = defaultdict(list)
    journal_link_counts: dict[tuple[str, str], int] = defaultdict(int)
    for event in finished:
        if event.get("request_role") != "baseline":
            continue
        case_id = str(event.get("case_id", ""))
        prompt_hash = str(event.get("prompt_sha256", ""))
        links = user_hashes.get((case_id, prompt_hash), [])
        item = RequestEvidence(
            prompt_sha256="",
            finish_reason=str(event.get("finish_reason", "")),
            content_nonempty=event.get("content_nonempty") is True,
            http_status=event.get("http_status") if isinstance(event.get("http_status"), int) else None,
            output_tokens=event.get("output_tokens") if isinstance(event.get("output_tokens"), int) else None,
            request_sha256=str(event.get("request_sha256", "")),
            raw_response_sha256=str(event.get("raw_response_sha256", "")),
        )
        for user_hash in links:
            key = (case_id, user_hash)
            journal_link_counts[key] += 1
            evidence_candidates[key].append(
                RequestEvidence(
                    prompt_sha256=user_hash,
                    finish_reason=item.finish_reason,
                    content_nonempty=item.content_nonempty,
                    http_status=item.http_status,
                    output_tokens=item.output_tokens,
                    request_sha256=item.request_sha256,
                    raw_response_sha256=item.raw_response_sha256,
                )
            )
    # Keep only unique request entries whose exact user prompt body was
    # independently reconstructed from the append-only request journal.
    evidence = {key: values[0] for key, values in evidence_candidates.items() if len(values) == 1}
    return evidence, journal_link_counts


def trace_as_cor_result(trace: dict[str, Any]) -> dict[str, Any]:
    ids = trace.get("baseline_prediction_ids")
    names = trace.get("baseline_answer_names")
    ids = ids if isinstance(ids, list) else []
    names = names if isinstance(names, list) else []
    mapped: list[dict[str, str]] = []
    for index, name in enumerate(names):
        entity_id = ids[index] if len(ids) == len(names) and index < len(ids) else ""
        mapped.append({"id": str(entity_id), "name": str(name)})
    action = "Stop" if trace.get("terminal_source") == "direct-LLM-reasoning-STOP" else "Unsupported"
    return {
        "action": action,
        "question": str(trace.get("question", "")),
        "results": mapped,
        "prompt_history": trace.get("prompt_history", []),
        "sparql_history": trace.get("sparql_history", []),
    }


def serialize_trace(
    trace: dict[str, Any], evidence_by_prompt: dict[tuple[str, str], RequestEvidence],
    journal_links: dict[tuple[str, str], int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = trace_as_cor_result(trace)
    history = result["prompt_history"]
    terminal = next(
        (event for event in reversed(history) if isinstance(event, dict) and event.get("type") in {"reasoning", "filter_answer"}),
        {},
    )
    prompt = terminal.get("prompt") if isinstance(terminal, dict) else None
    prompt_hash = sha256_bytes(str(prompt or "").encode("utf-8"))
    key = (str(trace.get("case_id", "")), prompt_hash)
    evidence = evidence_by_prompt.get(key)
    link_count = journal_links.get(key, 0)
    outcome = serialize_callsite_result(result, enabled=True, request_evidence=evidence if link_count == 1 else None)
    typed = dict(outcome.typed_answer or {
        "contract_version": "iter32-typed-answer-v1",
        "answer_kind": "unknown",
        "reason": outcome.reason,
    })
    runtime_row = {
        "case_id": trace.get("case_id", ""),
        "question_key": trace.get("question_key", ""),
        "question": trace.get("question", ""),
        "baseline_status": trace.get("baseline_status", ""),
        "terminal_source": trace.get("terminal_source", ""),
        "terminal_trace_id": terminal.get("trace_id", "") if isinstance(terminal, dict) else "",
        "terminal_prompt_sha256": prompt_hash,
        "request_journal_prompt_body_matches": link_count == 1,
        "request_finish_reason": evidence.finish_reason if evidence else "missing_or_unlinked",
        "request_content_nonempty": evidence.content_nonempty if evidence else False,
        "request_output_tokens": evidence.output_tokens if evidence else "",
        "request_sha256": evidence.request_sha256 if evidence else "",
        "raw_response_sha256": evidence.raw_response_sha256 if evidence else "",
        "router_requested_final_answer_type": _router_value(history, "requested_final_answer_type"),
        "router_requested_final_semantic_property": _router_value(history, "requested_final_semantic_property"),
        "terminal_response": terminal.get("response", "") if isinstance(terminal, dict) else "",
        "returned_names": json.dumps(trace.get("baseline_answer_names", []), ensure_ascii=False),
        "returned_ids": json.dumps(trace.get("baseline_prediction_ids", []), ensure_ascii=False),
        "serializer_status": outcome.status,
        "serializer_reason": outcome.reason,
        "answer_kind": typed.get("answer_kind", "unknown"),
        "lexical_value": typed.get("lexical_value", ""),
        "source_text": typed.get("source_text", ""),
        "provenance_json": json.dumps(typed.get("provenance", {}), ensure_ascii=False, sort_keys=True),
        "baseline_preserved_on_rejection": outcome.baseline_preserved,
    }
    snapshot = {
        "case_id": trace.get("case_id", ""),
        "question_key": trace.get("question_key", ""),
        "question": trace.get("question", ""),
        "baseline_status": trace.get("baseline_status", ""),
        "baseline_prediction_ids": trace.get("baseline_prediction_ids", []),
        "baseline_answer_names": trace.get("baseline_answer_names", []),
        "legacy_answer_kind": trace.get("answer_kind", "unknown"),
        "typed_answer": typed,
        "serializer_status": outcome.status,
        "serializer_reason": outcome.reason,
        "selector_prediction_ids": trace.get("intervention_prediction_ids", []),
    }
    return runtime_row, snapshot


def _router_value(history: Any, name: str) -> str:
    if not isinstance(history, list):
        return ""
    for event in history:
        if not isinstance(event, dict) or event.get("type") != "adaptive_router_v1_1_3":
            continue
        parsed = event.get("parsed_result")
        return str(parsed.get(name, "")) if isinstance(parsed, dict) else ""
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty Iteration 32 output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    count_keys = load_count_keys()
    traces = read_jsonl(RUN / "CASE_TRACES_NON_GOLD.jsonl")
    if len(traces) != 800 or len({row.get("question_key") for row in traces}) != 800:
        raise SystemExit("frozen Iteration 29 non-gold trace is not 800 unique cases")
    if not count_keys or not count_keys.issubset({row.get("question_key") for row in traces}):
        raise SystemExit("Iteration 31 runtime audit key set does not align with archived traces")

    evidence, journal_links = request_evidence_by_prompt()
    audit_rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for trace in traces:
        row, snapshot = serialize_trace(trace, evidence, journal_links)
        snapshots.append(snapshot)
        if trace.get("question_key") in count_keys:
            audit_rows.append(row)

    if len(audit_rows) != len(count_keys):
        raise SystemExit(f"expected {len(count_keys)} count audit rows, got {len(audit_rows)}")
    counts = Counter(row["typed_answer"].get("answer_kind", "unknown") for row in snapshots)
    summary = {
        "status": "FROZEN_NON_GOLD_RUNTIME_AUDIT",
        "case_count": len(snapshots),
        "posthoc_count_subset_size_from_iter31_runtime_audit_keys": len(count_keys),
        "count_subset_scalar_acceptances": sum(row["answer_kind"] == "aggregate" for row in audit_rows),
        "all_cohort_answer_kind_counts": dict(sorted(counts.items())),
        "accepted_scalar_question_keys": [row["question_key"] for row in snapshots if row["typed_answer"].get("answer_kind") == "aggregate"],
        "gold_read": False,
        "network_calls": 0,
    }
    fields = list(audit_rows[0]) if audit_rows else []
    with (args.output / "SCALAR_SOURCE_AUDIT.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)
    with (args.output / "PREDICTION_SNAPSHOT_NON_GOLD.jsonl").open("w", encoding="utf-8") as handle:
        for row in snapshots:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output / "RUNTIME_AUDIT_SUMMARY_NON_GOLD.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    sources = {
        "trace_sha256": sha256_bytes((RUN / "CASE_TRACES_NON_GOLD.jsonl").read_bytes()),
        "request_events_sha256": sha256_bytes((RUN / "REQUEST_EVENTS.jsonl").read_bytes()),
        "request_journal_sha256": sha256_bytes((RUN / "REQUEST_JOURNAL.jsonl").read_bytes()),
        "iter31_target_type_audit_sha256": sha256_bytes((ITER31 / "TARGET_TYPE_RESOLVER_AUDIT.csv").read_bytes()),
        "prediction_snapshot_sha256": sha256_bytes((args.output / "PREDICTION_SNAPSHOT_NON_GOLD.jsonl").read_bytes()),
        "negative_controls_sha256": sha256_bytes((args.output / "PREDECLARED_NEGATIVE_CONTROLS_NON_GOLD.json").read_bytes()),
    }
    (args.output / "RUNTIME_SOURCE_HASHES.json").write_text(
        json.dumps(sources, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
