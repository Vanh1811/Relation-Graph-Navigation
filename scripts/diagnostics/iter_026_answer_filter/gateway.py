"""Fail-closed attempt accounting for a future Iteration 26 run.

No network request occurs unless ``call_model_once`` or ``select_once`` is
explicitly called with a separately supplied transport. Attempt records and
the exact non-secret request/query are fsynced before the call. Any uncertain
transport outcome halts the gateway and makes journal reuse impossible.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
SAFETY_DIR = HERE.parent / "transport_safety_repair"
if str(SAFETY_DIR) not in sys.path:
    sys.path.insert(0, str(SAFETY_DIR))
from hard_deadline import call_with_hard_deadline  # noqa: E402


class GatewayError(RuntimeError):
    pass


class ResumeForbidden(GatewayError):
    pass


class DuplicatePrompt(GatewayError):
    pass


class BudgetExhausted(GatewayError):
    pass


class AttemptUncertain(GatewayError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _append_fsync(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    data = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("ab", buffering=0) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if not existed:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class ResearchCallOnceGateway:
    """Meter and journal unique model prompts and physical read-only SELECTs."""

    def __init__(
        self,
        output_dir: Path,
        *,
        manifest_sha256: str,
        model_transport: Callable[[dict[str, Any]], dict[str, Any]],
        sparql_transport: Callable[[str], dict[str, Any]],
        api_request_cap: int = 180,
        api_cost_cap_usd: float = 0.50,
        input_tokens_per_request: int = 7000,
        output_tokens_per_request: int = 1024,
        input_usd_per_million: float = 0.20,
        output_usd_per_million: float = 1.20,
        sparql_attempt_cap: int = 320,
        timeout_seconds: float = 120.0,
        wall_seconds: float = 8 * 60 * 60,
        token_counter: Callable[[list[dict[str, str]]], int] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
            raise ValueError("manifest SHA-256 must be lowercase hexadecimal")
        if getattr(model_transport, "max_retries", None) != 0:
            raise ValueError("model transport must explicitly declare max_retries=0")
        if getattr(sparql_transport, "max_retries", None) != 0:
            raise ValueError("SPARQL transport must explicitly declare max_retries=0")
        if min(api_request_cap, sparql_attempt_cap, input_tokens_per_request, output_tokens_per_request) <= 0:
            raise ValueError("request and token caps must be positive")
        if min(api_cost_cap_usd, input_usd_per_million, output_usd_per_million) < 0:
            raise ValueError("prices and API cap cannot be negative")
        if timeout_seconds <= 0 or wall_seconds <= 0:
            raise ValueError("timeouts must be positive")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.request_journal = self.output_dir / "REQUEST_JOURNAL.jsonl"
        self.sparql_journal = self.output_dir / "SPARQL_QUERY_JOURNAL.jsonl"
        self.attempt_journal = self.output_dir / "ATTEMPT_JOURNAL.jsonl"
        for path in (self.request_journal, self.sparql_journal, self.attempt_journal):
            if path.exists() and path.stat().st_size:
                raise ResumeForbidden("Iteration 26 journals must be new and empty for every authorized manifest")

        self.manifest_sha256 = manifest_sha256
        self.model_transport = model_transport
        self.sparql_transport = sparql_transport
        self.api_request_cap = api_request_cap
        self.api_cost_cap_usd = float(api_cost_cap_usd)
        self.input_tokens_per_request = input_tokens_per_request
        self.output_tokens_per_request = output_tokens_per_request
        self.input_price = float(input_usd_per_million)
        self.output_price = float(output_usd_per_million)
        self.sparql_attempt_cap = sparql_attempt_cap
        self.timeout_seconds = float(timeout_seconds)
        self.wall_seconds = float(wall_seconds)
        if token_counter is None:
            try:
                from .transport import count_input_tokens
            except ImportError:  # direct unittest/script invocation
                from transport import count_input_tokens
            token_counter = count_input_tokens
        self.token_counter = token_counter
        self.clock = clock
        self.started_monotonic = clock()
        self.started_epoch = time.time()
        self.api_starts = 0
        self.api_reserved_usd = 0.0
        self.api_input_tokens = 0
        self.api_output_tokens = 0
        self.api_actual_cost_usd = 0.0
        self.sparql_starts = 0
        self.prompt_hashes: set[str] = set()
        self.query_results: dict[str, dict[str, Any]] = {}
        self.active_case_id: str | None = None
        self.halted = False
        _append_fsync(self.attempt_journal, {
            "event": "RUN_STARTED",
            "manifest_sha256": manifest_sha256,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "recorded_at_epoch": self.started_epoch,
        })

    def _check_live(self) -> None:
        if self.halted:
            raise GatewayError("gateway halted after an uncertain/failed transport; no later calls are allowed")
        elapsed = max(self.clock() - self.started_monotonic, time.time() - self.started_epoch)
        if elapsed < 0 or elapsed >= self.wall_seconds:
            self.halted = True
            raise BudgetExhausted("overall wall-clock cap reached")

    def set_case_context(self, case_id: str | None) -> None:
        """Attach subsequent one-shot requests to their frozen question slot."""
        if case_id is not None and (not isinstance(case_id, str) or not case_id.strip()):
            raise ValueError("case_id must be non-empty text or None")
        self.active_case_id = case_id

    def _record_attempt_start(self, *, kind: str, digest: str, details: dict[str, Any]) -> str:
        self._check_live()
        attempt_id = str(uuid.uuid4())
        row = {
            "event": "ATTEMPT_STARTED",
            "kind": kind,
            "attempt_id": attempt_id,
            "manifest_sha256": self.manifest_sha256,
            "payload_sha256": digest,
            "case_id": self.active_case_id,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        _append_fsync(self.attempt_journal, row)
        return attempt_id

    def call_model_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_completion_tokens: int,
        reasoning_effort: str = "low",
    ) -> dict[str, Any]:
        self._check_live()
        if getattr(self.model_transport, "max_retries", None) != 0:
            raise GatewayError("model transport retry policy changed")
        if not isinstance(messages, list) or not messages or any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message["role"], str)
            or not isinstance(message["content"], str)
            for message in messages
        ):
            raise ValueError("messages must be a non-empty list of role/content text objects")
        if not 1 <= int(max_completion_tokens) <= self.output_tokens_per_request:
            raise BudgetExhausted("completion cap exceeds the authorized per-request limit")
        prompt_hash = sha256_bytes(canonical_json(messages))
        if prompt_hash in self.prompt_hashes:
            self.halted = True
            _append_fsync(self.request_journal, {
                "event": "DUPLICATE_PROMPT_BLOCKED",
                "manifest_sha256": self.manifest_sha256,
                "prompt_sha256": prompt_hash,
                "case_id": self.active_case_id,
                "gateway_halted": True,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            raise DuplicatePrompt("same model messages already attempted; duplicate transport blocked")
        estimated_input = int(self.token_counter(messages))
        if estimated_input > self.input_tokens_per_request:
            raise BudgetExhausted(f"estimated input {estimated_input} exceeds per-request cap")
        if self.api_starts >= self.api_request_cap:
            raise BudgetExhausted("API request cap reached")
        reservation = (
            self.input_tokens_per_request * self.input_price
            + self.output_tokens_per_request * self.output_price
        ) / 1_000_000
        if self.api_reserved_usd + reservation > self.api_cost_cap_usd + 1e-12:
            raise BudgetExhausted("worst-case API reservation would exceed the total cost cap")

        request_body = {
            "model": "gpt-5.6-luna",
            "messages": messages,
            "max_completion_tokens": int(max_completion_tokens),
            "reasoning_effort": reasoning_effort,
            "stream": False,
        }
        request_sha = sha256_bytes(canonical_json(request_body))
        # Persist the exact non-secret request and the attempt reservation before transport.
        _append_fsync(self.request_journal, {
            "event": "REQUEST_STARTED",
            "manifest_sha256": self.manifest_sha256,
            "case_id": self.active_case_id,
            "request_sha256": request_sha,
            "prompt_sha256": prompt_hash,
            "estimated_input_tokens": estimated_input,
            "reserved_cost_usd": reservation,
            "request_body": request_body,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        attempt_id = self._record_attempt_start(
            kind="api", digest=request_sha,
            details={
                "reserved_cost_usd": reservation,
                "reserved_input_tokens": self.input_tokens_per_request,
                "reserved_completion_tokens": self.output_tokens_per_request,
            },
        )
        self.prompt_hashes.add(prompt_hash)
        self.api_starts += 1
        self.api_reserved_usd += reservation
        try:
            result = call_with_hard_deadline(
                lambda: self.model_transport(request_body), self.timeout_seconds,
            )
            if not isinstance(result, dict) or result.get("http_status") != 200:
                raise AttemptUncertain("OpenAI did not return HTTP 200; no retry is allowed")
            usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
            if not all(key in usage for key in ("input_tokens", "output_tokens", "total_tokens")):
                raise AttemptUncertain("OpenAI usage metadata is missing; no retry is allowed")
            actual_input = int(usage.get("input_tokens", 0) or 0)
            actual_output = int(usage.get("output_tokens", 0) or 0)
            actual_cost = (
                actual_input * self.input_price + actual_output * self.output_price
            ) / 1_000_000
            if actual_input > self.input_tokens_per_request or actual_output > self.output_tokens_per_request:
                raise AttemptUncertain("reported usage exceeded the manifest request cap")
            self.api_input_tokens += actual_input
            self.api_output_tokens += actual_output
            self.api_actual_cost_usd += actual_cost
            finished = {
                "event": "REQUEST_FINISHED",
                "manifest_sha256": self.manifest_sha256,
                "case_id": self.active_case_id,
                "request_sha256": request_sha,
                "status": "HTTP_OK",
                "finish_reason": result.get("finish_reason"),
                "raw_response_sha256": result.get("raw_response_sha256"),
                "raw_response_body": result.get("raw_response_body"),
                "content": result.get("content"),
                "usage": usage,
                "actual_cost_usd": actual_cost,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _append_fsync(self.request_journal, finished)
            _append_fsync(self.attempt_journal, {
                "event": "ATTEMPT_FINISHED", "attempt_id": attempt_id,
                "outcome": "HTTP_OK", "actual_cost_usd": actual_cost,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            return result
        except BaseException as exc:
            self.halted = True
            _append_fsync(self.request_journal, {
                "event": "REQUEST_FINISHED",
                "manifest_sha256": self.manifest_sha256,
                "case_id": self.active_case_id,
                "request_sha256": request_sha,
                "status": "UNCERTAIN_OR_FAILED_NO_RETRY",
                "error_type": type(exc).__name__,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            _append_fsync(self.attempt_journal, {
                "event": "ATTEMPT_FINISHED", "attempt_id": attempt_id,
                "outcome": "UNCERTAIN_OR_FAILED_FULL_RESERVATION",
                "error_type": type(exc).__name__,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            raise

    @staticmethod
    def _validate_select(query: str) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("SPARQL query must be non-empty text")
        without_prefixes = re.sub(r"(?im)^\s*PREFIX\s+[\w:-]+\s*<[^>]+>\s*", "", query)
        if not re.match(r"\s*SELECT\b", without_prefixes, re.IGNORECASE):
            raise ValueError("only read-only SELECT queries are authorized")
        if re.search(r"\b(SERVICE|INSERT|DELETE|LOAD|CLEAR|CREATE|DROP|MOVE|COPY|ADD|WITH)\b", query, re.IGNORECASE):
            raise ValueError("query contains a forbidden remote-service or update operation")

    def select_once(self, query: str) -> dict[str, Any]:
        self._check_live()
        if getattr(self.sparql_transport, "max_retries", None) != 0:
            raise GatewayError("SPARQL transport retry policy changed")
        self._validate_select(query)
        query_hash = sha256_bytes(query.encode("utf-8"))
        if query_hash in self.query_results:
            _append_fsync(self.sparql_journal, {
                "event": "QUERY_CACHE_HIT",
                "manifest_sha256": self.manifest_sha256,
                "case_id": self.active_case_id,
                "query_sha256": query_hash,
                "physical_attempt": False,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            return self.query_results[query_hash]
        if self.sparql_starts >= self.sparql_attempt_cap:
            raise BudgetExhausted("physical SPARQL attempt cap reached")
        _append_fsync(self.sparql_journal, {
            "event": "QUERY_STARTED",
            "manifest_sha256": self.manifest_sha256,
            "case_id": self.active_case_id,
            "query_sha256": query_hash,
            "query": query,
            "physical_attempt": True,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        attempt_id = self._record_attempt_start(kind="sparql", digest=query_hash, details={})
        self.sparql_starts += 1
        try:
            result = call_with_hard_deadline(
                lambda: self.sparql_transport(query), self.timeout_seconds,
            )
            if (
                not isinstance(result, dict) or result.get("http_status") != 200
                or not isinstance(result.get("bindings"), list)
            ):
                raise AttemptUncertain("SPARQL did not return a valid SELECT result; no retry is allowed")
            _append_fsync(self.sparql_journal, {
                "event": "QUERY_FINISHED",
                "manifest_sha256": self.manifest_sha256,
                "case_id": self.active_case_id,
                "query_sha256": query_hash,
                "status": "HTTP_OK",
                "raw_response_sha256": result.get("raw_response_sha256"),
                "raw_response_body": result.get("raw_response_body"),
                "binding_count": len(result["bindings"]),
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            _append_fsync(self.attempt_journal, {
                "event": "ATTEMPT_FINISHED", "attempt_id": attempt_id,
                "outcome": "HTTP_OK", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            self.query_results[query_hash] = result
            return result
        except BaseException as exc:
            self.halted = True
            _append_fsync(self.sparql_journal, {
                "event": "QUERY_FINISHED",
                "manifest_sha256": self.manifest_sha256,
                "case_id": self.active_case_id,
                "query_sha256": query_hash,
                "status": "UNCERTAIN_OR_FAILED_NO_RETRY",
                "error_type": type(exc).__name__,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            _append_fsync(self.attempt_journal, {
                "event": "ATTEMPT_FINISHED", "attempt_id": attempt_id,
                "outcome": "UNCERTAIN_OR_FAILED_NO_RETRY",
                "error_type": type(exc).__name__,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            raise
