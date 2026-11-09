#!/usr/bin/env python3
"""Approval-gated, two-query Iteration 24 label-source audit.

Default invocation is an offline preflight. Live traffic requires --run and an
exact approved AUTHORIZATION.json matching the immutable manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_024_label_source_audit"
MANIFEST_PATH = OUT / "PROPOSED_MANIFEST.json"
MANIFEST_HASH_PATH = OUT / "PROPOSED_MANIFEST.sha256"
AUTH_PATH = OUT / "AUTHORIZATION.json"
ENDPOINT = "http://127.0.0.1:8890/sparql"
MAX_ATTEMPTS = 2
PER_CALL_SECONDS = 120
TOTAL_SECONDS = 1800

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SAFETY_PATH = ROOT / "scripts/diagnostics/transport_safety_repair"
if str(SAFETY_PATH) not in sys.path:
    sys.path.insert(0, str(SAFETY_PATH))
from hard_deadline import call_with_hard_deadline  # noqa: E402


class AuditStop(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def append_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_frozen_inputs() -> tuple[dict[str, Any], str]:
    raw = MANIFEST_PATH.read_bytes()
    digest = sha256_bytes(raw)
    recorded = MANIFEST_HASH_PATH.read_text(encoding="utf-8").split()[0]
    if digest != recorded:
        raise AuditStop("manifest digest mismatch")
    manifest = json.loads(raw)
    if manifest.get("iteration") != 24 or manifest.get("endpoint") != ENDPOINT:
        raise AuditStop("manifest identity or endpoint mismatch")
    if manifest.get("api") != {"enabled": False, "requests": 0, "spend_cap_usd": 0}:
        raise AuditStop("manifest unexpectedly enables paid API calls")
    sparql = manifest.get("sparql", {})
    if (sparql.get("maximum_physical_attempts") != MAX_ATTEMPTS
            or sparql.get("preflight_attempts_included") != 1
            or sparql.get("retries") != 0
            or sparql.get("maximum_seconds_per_attempt") != PER_CALL_SECONDS
            or sparql.get("maximum_elapsed_seconds") != TOTAL_SECONDS
            or len(sparql.get("queries", [])) != MAX_ATTEMPTS):
        raise AuditStop("manifest query and time bounds differ from the runner")
    ids = manifest.get("fixed_entity_ids", [])
    if len(ids) != 7 or len(set(ids)) != 7:
        raise AuditStop("manifest must contain exactly seven distinct fixed IDs")
    categories = [query.get("category") for query in sparql["queries"]]
    if categories != ["endpoint_preflight", "label_source_evidence"]:
        raise AuditStop("manifest query categories/order differ from the fixed audit")
    for query in sparql["queries"]:
        text = query.get("query", "")
        is_select = re.match(r"(?is)^\s*(?:PREFIX\s+[^\s:]+:\s*<[^>]+>\s*)*SELECT\b", text)
        has_update = re.search(r"(?i)\b(INSERT|DELETE|LOAD|CLEAR|CREATE|DROP|MOVE|COPY|ADD|WITH)\b", text)
        if (not is_select or has_update
                or query.get("sha256") != sha256_bytes(text.encode())):
            raise AuditStop("manifest contains a non-SELECT or hash-mismatched query")
    for rel, expected in manifest.get("runtime_source_sha256", {}).items():
        if sha256_file(ROOT / rel) != expected:
            raise AuditStop(f"runtime source hash mismatch: {rel}")
    return manifest, digest


def expected_approval_text(manifest_sha256: str) -> str:
    return (
        f"I authorize Iteration 24 manifest {manifest_sha256}: one read-only Freebase label-source audit "
        "for the seven fixed Iteration 23 unresolved IDs using at most 2 SPARQL SELECT attempts total "
        "including 1 endpoint preflight, no OpenAI/API requests, no retries, max 120 seconds per request, "
        "max 30 minutes total, endpoint http://127.0.0.1:8890/sparql."
    )


def validate_approval(manifest_sha256: str) -> dict[str, Any]:
    if not AUTH_PATH.is_file():
        raise AuditStop("approved AUTHORIZATION.json is absent")
    approval = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    if (approval.get("authorization_status") != "APPROVED"
            or approval.get("manifest_sha256") != manifest_sha256
            or approval.get("approval_verbatim") != expected_approval_text(manifest_sha256)
            or approval.get("transport_opened") is not False):
        raise AuditStop("authorization does not exactly approve this immutable manifest")
    return approval


def query_once(query_text: str, timeout_seconds: float) -> dict[str, Any]:
    """Issue one HTTP request in a killable process; do not retry."""
    def transport() -> dict[str, Any]:
        from SPARQLWrapper import JSON, SPARQLWrapper

        client = SPARQLWrapper(ENDPOINT)
        client.setQuery(query_text)
        client.setReturnFormat(JSON)
        client.setTimeout(timeout_seconds + 5)
        return client.query().convert()

    value = call_with_hard_deadline(transport, timeout_seconds)
    if not isinstance(value, dict) or not isinstance(value.get("results", {}).get("bindings"), list):
        raise AuditStop("endpoint returned an unexpected SPARQL JSON result")
    return value


def run_live(manifest: dict[str, Any], manifest_sha256: str,
             approval: dict[str, Any]) -> dict[str, Any]:
    state_path = OUT / "RUN_STATE.json"
    journal_path = OUT / "QUERY_JOURNAL.jsonl"
    result_path = OUT / "RESULT.json"
    for path in (state_path, journal_path, result_path):
        if path.exists():
            raise AuditStop(f"live output already exists; this authorization cannot be replayed: {path.name}")

    started = time.monotonic()
    state = {
        "iteration": 24,
        "status": "RUNNING",
        "manifest_sha256": manifest_sha256,
        "started_at_utc": utc_now(),
        "transport_opened": True,
        "api_requests": 0,
        "sparql_attempts": 0,
        "retries": 0,
    }
    # Consume authorization durably before any network operation. A crash after
    # this point requires a new manifest and approval, even if no bytes arrived.
    approval["transport_opened"] = True
    approval["consumed_at_utc"] = utc_now()
    approval["actual_usage"] = {"api_requests": 0, "read_only_sparql_attempts": 0, "retries": 0}
    write_json(AUTH_PATH, approval)
    write_json(state_path, state)

    results: list[dict[str, Any]] = []
    try:
        for query_spec in manifest["sparql"]["queries"]:
            elapsed = time.monotonic() - started
            remaining = TOTAL_SECONDS - elapsed
            if remaining <= 0:
                raise AuditStop("30-minute overall deadline reached")
            index = state["sparql_attempts"] + 1
            allowance = min(PER_CALL_SECONDS, remaining)
            event = {
                "query_index": index,
                "category": query_spec["category"],
                "query_sha256": query_spec["sha256"],
                "status": "ATTEMPT_STARTED",
                "retries": 0,
                "started_at_utc": utc_now(),
                "timeout_seconds": allowance,
            }
            state["sparql_attempts"] = index
            approval["actual_usage"]["read_only_sparql_attempts"] = index
            write_json(AUTH_PATH, approval)
            write_json(state_path, state)
            append_jsonl(journal_path, event)
            try:
                response = query_once(query_spec["query"], allowance)
            except BaseException as exc:
                event.update({"status": "UNCERTAIN_OR_FAILED", "error_type": type(exc).__name__,
                              "finished_at_utc": utc_now()})
                append_jsonl(journal_path, event)
                raise AuditStop(f"query {index} failed or timed out; stopped without retry") from exc
            bindings = response["results"]["bindings"]
            event.update({"status": "OK", "row_count": len(bindings), "rows": bindings,
                          "finished_at_utc": utc_now()})
            append_jsonl(journal_path, event)
            results.append({"category": query_spec["category"], "row_count": len(bindings), "rows": bindings})

        final = {
            "iteration": 24,
            "status": "COMPLETE",
            "manifest_sha256": manifest_sha256,
            "endpoint": ENDPOINT,
            "api_requests": 0,
            "sparql_attempts": state["sparql_attempts"],
            "retries": 0,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "results": results,
            "label_policy": "evidence_only_no_automatic_cache_update",
            "finished_at_utc": utc_now(),
        }
        state.update({"status": "COMPLETE", "finished_at_utc": final["finished_at_utc"]})
        write_json(state_path, state)
        write_json(result_path, final)
        return final
    except BaseException as exc:
        state.update({"status": "STOPPED", "stop_reason": f"{type(exc).__name__}: {exc}",
                      "finished_at_utc": utc_now()})
        write_json(state_path, state)
        stopped = {
            "iteration": 24, "status": "STOPPED", "manifest_sha256": manifest_sha256,
            "api_requests": 0, "sparql_attempts": state["sparql_attempts"], "retries": 0,
            "stop_reason": state["stop_reason"], "finished_at_utc": state["finished_at_utc"],
        }
        write_json(result_path, stopped)
        return stopped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="open the two explicitly approved read-only SELECT transports")
    args = parser.parse_args()
    try:
        manifest, digest = load_frozen_inputs()
        if not args.run:
            print(json.dumps({"status": "OFFLINE_PREFLIGHT_PASS_NO_TRANSPORT",
                              "manifest_sha256": digest,
                              "fixed_ids": manifest["fixed_entity_ids"],
                              "maximum_sparql_attempts": MAX_ATTEMPTS,
                              "api_requests": 0,
                              "transport_opened": False}, indent=2))
            return 0
        approval = validate_approval(digest)
        result = run_live(manifest, digest, approval)
        print(json.dumps({key: result.get(key) for key in (
            "status", "manifest_sha256", "api_requests", "sparql_attempts", "retries", "elapsed_seconds"
        )}, indent=2))
        return 0 if result.get("status") == "COMPLETE" else 1
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
