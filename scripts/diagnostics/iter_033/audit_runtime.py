#!/usr/bin/env python3
"""Verify Iteration 32 scalar serialization using runtime artifacts only.

This stage deliberately does not open any gold dataset, logical form, or
gold-only scoring artifact. It links accepted scalars to their archived CoR
terminal answer, prompt, request journal event, and frozen prediction object.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[3]
ITER32 = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract"
RUN = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_028_terminal_boundary_repair/RUNS/iter29_terminal_candidate_selector_attempt3"
OUT = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_033_scalar_generalization_and_upstream_decision/RUNTIME_REPRESENTATION_AUDIT_NON_GOLD.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    runtime_hashes = json.loads((ITER32 / "RUNTIME_SOURCE_HASHES.json").read_text())
    snapshot_path = ITER32 / "PREDICTION_SNAPSHOT_NON_GOLD.jsonl"
    trace_path = RUN / "CASE_TRACES_NON_GOLD.jsonl"
    request_journal_path = RUN / "REQUEST_JOURNAL.jsonl"
    request_events_path = RUN / "REQUEST_EVENTS.jsonl"
    scalar_audit_path = ITER32 / "SCALAR_SOURCE_AUDIT.csv"

    checked_hashes = {
        str(snapshot_path.relative_to(ROOT)): {
            "expected": runtime_hashes["iter32_non_gold_snapshot_sha256"],
            "actual": sha256_file(snapshot_path),
        },
        str(trace_path.relative_to(ROOT)): {
            "expected": runtime_hashes["iter29_trace_sha256"],
            "actual": sha256_file(trace_path),
        },
        str(request_journal_path.relative_to(ROOT)): {
            "expected": runtime_hashes["iter29_request_journal_sha256"],
            "actual": sha256_file(request_journal_path),
        },
        str(request_events_path.relative_to(ROOT)): {
            "expected": runtime_hashes["iter29_request_events_sha256"],
            "actual": sha256_file(request_events_path),
        },
    }
    for path, values in checked_hashes.items():
        if values["expected"] != values["actual"]:
            raise SystemExit(f"runtime source hash mismatch: {path}")

    snapshots = read_jsonl(snapshot_path)
    traces = read_jsonl(trace_path)
    request_journal = read_jsonl(request_journal_path)
    request_events = read_jsonl(request_events_path)
    with scalar_audit_path.open(encoding="utf-8", newline="") as stream:
        scalar_rows = list(csv.DictReader(stream))

    def unique_index(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for row in rows:
            value = str(row.get(key, ""))
            if not value or value in indexed:
                raise SystemExit(f"missing/duplicate {label} key {value!r}")
            indexed[value] = row
        return indexed

    snapshots_by_case = unique_index(snapshots, "case_id", "snapshot case_id")
    traces_by_case = unique_index(traces, "case_id", "trace case_id")
    scalar_by_case = unique_index(scalar_rows, "case_id", "scalar audit case_id")
    if len(snapshots) != 800 or len(traces) != 800 or len(scalar_rows) != 68:
        raise SystemExit(
            "expected 800 snapshot/trace rows and the 68-case COUNT scalar audit; "
            f"got {len(snapshots)}/{len(traces)}/{len(scalar_rows)}"
        )
    if set(snapshots_by_case) != set(traces_by_case):
        raise SystemExit("snapshot and trace case identity sets differ")
    if not set(scalar_by_case).issubset(snapshots_by_case):
        raise SystemExit("COUNT scalar audit contains a case absent from the frozen runtime cohort")

    accepted = []
    accepted_case_ids = set()
    for row in scalar_rows:
        if row.get("serializer_status") != "AGGREGATE":
            continue
        case_id = row["case_id"]
        if case_id in accepted_case_ids:
            raise SystemExit(f"duplicate accepted case {case_id}")
        accepted_case_ids.add(case_id)
        snapshot = snapshots_by_case[case_id]
        trace = traces_by_case[case_id]
        typed = snapshot.get("typed_answer") or {}
        if typed.get("answer_kind") != "aggregate" or typed.get("status") != "accepted":
            raise SystemExit(f"accepted sidecar missing for {case_id}")
        lexical = str(typed.get("lexical_value", ""))
        if not lexical or row.get("lexical_value") != lexical or row.get("source_text") != lexical:
            raise SystemExit(f"scalar text mismatch for {case_id}")
        if snapshot.get("baseline_answer_names") != [lexical]:
            raise SystemExit(f"CoR result projection mismatch for {case_id}")
        if snapshot.get("baseline_prediction_ids"):
            raise SystemExit(f"accepted scalar unexpectedly carries entity IDs for {case_id}")

        response_text = row.get("terminal_response", "")
        try:
            terminal = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid terminal JSON for {case_id}: {exc}")
        if terminal.get("decision", "").lower() != "stop" or terminal.get("answer") != [lexical]:
            raise SystemExit(f"terminal answer[0] / Stop mismatch for {case_id}")
        response_text_hash = sha256_bytes(response_text.encode("utf-8"))
        provenance = json.loads(row.get("provenance_json") or "{}")
        if response_text_hash != provenance.get("terminal_response_text_sha256"):
            raise SystemExit(f"terminal response text hash mismatch for {case_id}")

        trace_id = str(row.get("terminal_trace_id", ""))
        trace_events = [event for event in trace.get("prompt_history", []) if str(event.get("trace_id", "")) == trace_id]
        if len(trace_events) != 1:
            raise SystemExit(f"expected one archived terminal trace event for {case_id}")
        trace_event = trace_events[0]
        prompt_hash = sha256_bytes(str(trace_event.get("prompt", "")).encode("utf-8"))
        if prompt_hash != row.get("terminal_prompt_sha256"):
            raise SystemExit(f"terminal prompt hash mismatch for {case_id}")
        if trace_event.get("response") != response_text or trace_event.get("decision", "").lower() != "stop":
            raise SystemExit(f"trace-to-terminal response mismatch for {case_id}")
        if trace_event.get("answers") != [lexical]:
            raise SystemExit(f"raw terminal answer to trace projection mismatch for {case_id}")

        matching_events = [
            event for event in request_events
            if event.get("case_id") == case_id
            and event.get("request_sha256") == row.get("request_sha256")
            and event.get("raw_response_sha256") == row.get("raw_response_sha256")
        ]
        if len(matching_events) != 1:
            raise SystemExit(f"request event linkage missing/ambiguous for {case_id}")
        event = matching_events[0]
        if event.get("http_status") != 200 or event.get("finish_reason") != "stop" or not event.get("content_nonempty"):
            raise SystemExit(f"request completion provenance invalid for {case_id}")

        matching_journal = [
            entry for entry in request_journal
            if entry.get("case_id") == case_id and entry.get("request_sha256") == row.get("request_sha256")
        ]
        if not matching_journal:
            raise SystemExit(f"request journal linkage missing for {case_id}")
        prompt_hashes = {entry.get("prompt_sha256") for entry in matching_journal if entry.get("prompt_sha256")}
        body_entries = [entry for entry in matching_journal if isinstance(entry.get("request_body"), dict)]
        if len(prompt_hashes) != 1 or len(body_entries) != 1:
            raise SystemExit(f"request journal start/finish records disagree for {case_id}")
        journal_entry = body_entries[0]
        body = journal_entry.get("request_body", {})
        user_messages = [m.get("content") for m in body.get("messages", []) if m.get("role") == "user"]
        if user_messages != [trace_event.get("prompt")]:
            raise SystemExit(f"journal user prompt differs from archived trace for {case_id}")
        if event.get("prompt_sha256") != journal_entry.get("prompt_sha256"):
            raise SystemExit(f"request event/journal prompt hash mismatch for {case_id}")

        accepted.append({
            "case_id": case_id,
            "question_key": row["question_key"],
            "question": row["question"],
            "answer_kind": "Value",
            "datatype": typed.get("datatype"),
            "lexical_value": lexical,
            "baseline_prediction_ids": snapshot.get("baseline_prediction_ids", []),
            "baseline_answer_names": snapshot.get("baseline_answer_names", []),
            "serializer_reason": snapshot.get("serializer_reason"),
            "terminal_trace_id": trace_id,
            "terminal_prompt_sha256": prompt_hash,
            "raw_response_sha256": row.get("raw_response_sha256"),
            "terminal_response_text_sha256": response_text_hash,
            "request_sha256": row["request_sha256"],
            "finish_reason": event["finish_reason"],
            "http_status": event["http_status"],
            "prompt_body_linked": True,
            "identical_request_journal_records": len(matching_journal),
        })

    if len(accepted) != 39:
        raise SystemExit(f"expected 39 accepted runtime scalars, got {len(accepted)}")

    output = {
        "status": "PASS_NON_GOLD_RUNTIME_REPRODUCTION",
        "iteration": 33,
        "gold_accessed": False,
        "network_calls": 0,
        "expected_cases": 800,
        "observed_cases": 800,
        "accepted_scalars": len(accepted),
        "accepted_scalar_evidence_all_linked": True,
        "source_hashes": checked_hashes,
        "scalar_source_audit_sha256": sha256_file(scalar_audit_path),
        "accepted_cases": accepted,
        "non_gold_sources_read": [
            str(snapshot_path.relative_to(ROOT)),
            str(scalar_audit_path.relative_to(ROOT)),
            str(trace_path.relative_to(ROOT)),
            str(request_journal_path.relative_to(ROOT)),
            str(request_events_path.relative_to(ROOT)),
            str((ITER32 / "RUNTIME_SOURCE_HASHES.json").relative_to(ROOT)),
        ],
        "gold_sources_opened": [],
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "cases": 800, "accepted": len(accepted), "output": str(OUT.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
