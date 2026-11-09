"""Fixture-only future-pilot safety wrapper; never performs transport itself.

All transport callables are injected by a separately authorized caller. This
module records durable pre-transport reservations, enforces one-shot semantics
across process restarts, and delegates elapsed deadlines to the existing
killable-worker implementation.
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
from hard_deadline import (  # noqa: E402
    HardDeadlineExceeded,
    HardDeadlineUnavailable,
    RemoteTransportError,
    call_with_hard_deadline,
)
from model_call_guard import DuplicatePromptAttempt, OneShotModelCaller  # noqa: E402


class SafetyError(RuntimeError):
    pass


class ResumeForbidden(SafetyError):
    pass


class DuplicatePrompt(SafetyError):
    pass


class CacheMissNoTransport(SafetyError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _valid_digest(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _load_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SafetyError(f"journal line {number} is corrupt; refusing transport") from exc
            if not isinstance(row, dict):
                raise SafetyError(f"journal line {number} is not an object; refusing transport")
            records.append(row)
    return records


def _append_fsync(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    existed = path.exists()
    with path.open("ab", buffering=0) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if not existed:
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def _require_single_attempt_transport(transport: Callable[..., Any]) -> None:
    if getattr(transport, "max_retries", None) != 0:
        raise SafetyError("transport must explicitly declare max_retries=0; unknown SDK/wrapper retry policy is blocked")


def _contains_gold_metadata(value: Any) -> bool:
    forbidden = {"gold", "gold_ids", "gold_answer_ids", "reference_answers", "scoring_only", "cohort_role_scoring_only"}
    if isinstance(value, dict):
        return any(str(key).lower() in forbidden or _contains_gold_metadata(child) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_gold_metadata(child) for child in value)
    return False


class SafeDiagnosticRunner:
    """Durable no-resume/no-repeat reservations around injected fake transports."""

    def __init__(self, journal_path: Path, *, manifest_sha256: str,
                 api_request_cap: int, api_cost_cap_usd: float,
                 input_tokens_per_request: int, output_tokens_per_request: int,
                 input_usd_per_million: float, output_usd_per_million: float,
                 sparql_attempt_cap: int, timeout_seconds: float = 120.0,
                 wall_seconds: float = 8 * 60 * 60,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not _valid_digest(manifest_sha256):
            raise ValueError("manifest SHA-256 must be lowercase hexadecimal")
        if min(api_request_cap, sparql_attempt_cap, input_tokens_per_request, output_tokens_per_request) < 0:
            raise ValueError("caps must be non-negative")
        if timeout_seconds <= 0 or wall_seconds <= 0 or api_cost_cap_usd < 0:
            raise ValueError("timeouts and cost cap must be valid")
        if input_usd_per_million < 0 or output_usd_per_million < 0:
            raise ValueError("token prices must be non-negative")
        self.journal_path = Path(journal_path)
        self.manifest_sha256 = manifest_sha256
        self.api_request_cap = api_request_cap
        self.api_cost_cap_usd = api_cost_cap_usd
        self.input_tokens_per_request = input_tokens_per_request
        self.output_tokens_per_request = output_tokens_per_request
        self.input_usd_per_million = input_usd_per_million
        self.output_usd_per_million = output_usd_per_million
        self.sparql_attempt_cap = sparql_attempt_cap
        self.timeout_seconds = timeout_seconds
        self.wall_seconds = wall_seconds
        self.clock = clock
        self.started_monotonic = clock()
        previous = _load_journal(self.journal_path)
        if any(row.get("event") == "ATTEMPT_STARTED" for row in previous):
            raise ResumeForbidden("journal contains a submitted or uncertain attempt; create a new authorized manifest and journal")
        if any(row.get("manifest_sha256") not in (None, manifest_sha256) for row in previous):
            raise ResumeForbidden("journal manifest does not match this run")
        run_records = [row for row in previous if row.get("event") == "RUN_STARTED"]
        if run_records:
            if len(run_records) != 1:
                raise SafetyError("journal has invalid run-start markers")
            start_epoch = run_records[0].get("recorded_at_epoch")
            if not isinstance(start_epoch, (int, float)):
                raise SafetyError("journal run-start timestamp is invalid")
            self.started_epoch = float(start_epoch)
            elapsed_wall = time.time() - self.started_epoch
            if elapsed_wall < 0:
                raise SafetyError("wall clock moved backwards; refusing to extend run deadline")
            self.started_monotonic = self.clock() - elapsed_wall
        else:
            if previous:
                raise SafetyError("non-empty journal lacks a valid run-start marker")
            self.started_epoch = time.time()
            _append_fsync(self.journal_path, {
                "event": "RUN_STARTED", "manifest_sha256": self.manifest_sha256,
                "recorded_at_utc": _utc_now(), "recorded_at_epoch": self.started_epoch,
            })
            previous = _load_journal(self.journal_path)
        self.records = previous
        self.prompt_digests: set[str] = set()
        self.process_local_model_guard = OneShotModelCaller()
        self.api_reservations = 0.0
        self.api_starts = 0
        self.sparql_starts = 0

    def _check_wall(self) -> None:
        elapsed_mono = self.clock() - self.started_monotonic
        elapsed_epoch = time.time() - self.started_epoch
        if elapsed_mono < 0 or elapsed_epoch < 0:
            raise SafetyError("clock moved backwards; refusing to extend run deadline")
        if max(elapsed_mono, elapsed_epoch) >= self.wall_seconds:
            raise SafetyError("overall wall-clock cap reached; refusing transport")

    def _start(self, *, kind: str, digest: str, reserve_usd: float = 0.0,
               reserve_tokens: dict[str, int] | None = None) -> str:
        self._check_wall()
        if not _valid_digest(digest):
            raise ValueError("transport digest must be lowercase SHA-256")
        if kind == "api":
            if self.api_starts >= self.api_request_cap:
                raise SafetyError("API request cap reached")
            if self.api_reservations + reserve_usd > self.api_cost_cap_usd + 1e-12:
                raise SafetyError("worst-case API reservation exceeds cost cap")
            self.api_starts += 1
            self.api_reservations += reserve_usd
        else:
            if self.sparql_starts >= self.sparql_attempt_cap:
                raise SafetyError("physical SPARQL attempt cap reached")
            self.sparql_starts += 1
        attempt_id = str(uuid.uuid4())
        record = {
            "event": "ATTEMPT_STARTED",
            "kind": kind,
            "attempt_id": attempt_id,
            "manifest_sha256": self.manifest_sha256,
            "payload_sha256": digest,
            "reserved_cost_usd": round(reserve_usd, 12),
            "reserved_tokens": reserve_tokens or {},
            "recorded_at_utc": _utc_now(),
        }
        # Durable journal write precedes the only point that may launch a worker.
        _append_fsync(self.journal_path, record)
        self.records.append(record)
        return attempt_id

    @staticmethod
    def render_inference_prompt(inference_payload: dict[str, Any]) -> str:
        allowed = {"question", "answer_type", "candidate_entities", "candidate_paths", "constraints", "instructions"}
        if not isinstance(inference_payload, dict) or not inference_payload:
            raise SafetyError("inference payload must be a non-empty object")
        if _contains_gold_metadata(inference_payload):
            raise SafetyError("gold/scoring metadata is forbidden in inference payload")
        extra = set(inference_payload) - allowed
        if extra:
            raise SafetyError(f"inference payload contains non-allowlisted fields: {sorted(extra)}")
        return json.dumps(inference_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def call_model_once(self, inference_payload: dict[str, Any], transport: Callable[[str], Any]) -> Any:
        _require_single_attempt_transport(transport)
        prompt = self.render_inference_prompt(inference_payload)
        prompt_digest = sha256_bytes(prompt.encode("utf-8"))
        if prompt_digest in self.prompt_digests:
            raise DuplicatePrompt("prompt digest has already been reserved in this run")
        try:
            self.process_local_model_guard.claim(prompt_digest)
        except DuplicatePromptAttempt as exc:
            raise DuplicatePrompt(str(exc)) from exc
        self.prompt_digests.add(prompt_digest)
        reserve = (
            self.input_tokens_per_request * self.input_usd_per_million
            + self.output_tokens_per_request * self.output_usd_per_million
        ) / 1_000_000
        attempt_id = self._start(
            kind="api", digest=prompt_digest, reserve_usd=reserve,
            reserve_tokens={"input": self.input_tokens_per_request, "completion": self.output_tokens_per_request},
        )
        try:
            value = call_with_hard_deadline(lambda: transport(prompt), self.timeout_seconds)
        except HardDeadlineExceeded:
            self.process_local_model_guard.outcomes[prompt_digest] = "UNCERTAIN_OR_FAILED"
            self._finish(attempt_id, "UNCERTAIN_TIMEOUT_FULL_RESERVATION")
            raise
        except HardDeadlineUnavailable:
            self.process_local_model_guard.outcomes[prompt_digest] = "UNCERTAIN_OR_FAILED"
            self._finish(attempt_id, "UNCERTAIN_WORKER_UNAVAILABLE_FULL_RESERVATION")
            raise
        except BaseException as exc:
            self.process_local_model_guard.outcomes[prompt_digest] = "UNCERTAIN_OR_FAILED"
            self._finish(attempt_id, "UNCERTAIN_OR_FAILED_FULL_RESERVATION", error_type=type(exc).__name__)
            raise
        self.process_local_model_guard.outcomes[prompt_digest] = "RETURNED"
        self._finish(attempt_id, "RETURNED_FULL_RESERVATION_RETAINED")
        return value

    def select_once(self, query: str, transport: Callable[[], Any]) -> Any:
        _require_single_attempt_transport(transport)
        if not isinstance(query, str) or not re.match(r"\s*SELECT\b", query, re.IGNORECASE):
            raise SafetyError("only SELECT queries are allowed")
        if re.search(r"\b(SERVICE|INSERT|DELETE|LOAD|CLEAR|CREATE|DROP|MOVE|COPY|ADD|WITH)\b", query, re.IGNORECASE):
            raise SafetyError("query contains a forbidden remote or mutating operation")
        query_digest = sha256_bytes(query.encode("utf-8"))
        attempt_id = self._start(kind="sparql", digest=query_digest)
        try:
            value = call_with_hard_deadline(transport, self.timeout_seconds)
        except BaseException as exc:
            self._finish(attempt_id, "UNCERTAIN_OR_FAILED_NO_RETRY", error_type=type(exc).__name__)
            raise
        self._finish(attempt_id, "RETURNED")
        return value

    def resolve_label_cache_only(self, entity_id: str, cache_path: Path, *,
                                 expected_cache_sha256: str,
                                 expected_source_archive_sha256: str,
                                 expected_provenance_manifest_sha256: str) -> str:
        if not isinstance(entity_id, str) or not re.fullmatch(r"(?:m|g)\.[A-Za-z0-9_]+", entity_id):
            raise SafetyError("malformed Freebase entity ID")
        cache_path = Path(cache_path)
        raw = cache_path.read_bytes()
        if not _valid_digest(expected_cache_sha256) or sha256_bytes(raw) != expected_cache_sha256:
            raise SafetyError("label-cache content hash does not match the manifest-bound cache digest")
        document = json.loads(raw)
        if document.get("source_archive_sha256") != expected_source_archive_sha256:
            raise SafetyError("label-cache source archive provenance mismatch")
        if document.get("manifest_sha256") != expected_provenance_manifest_sha256:
            raise SafetyError("label-cache provenance manifest mismatch")
        labels = document.get("labels")
        if not isinstance(labels, list):
            raise SafetyError("label-cache entries are malformed")
        mapping: dict[str, str] = {}
        for entry in labels:
            if not isinstance(entry, dict) or not isinstance(entry.get("entity_id"), str) or not isinstance(entry.get("label"), str):
                raise SafetyError("label-cache entry is malformed")
            if entry["entity_id"] in mapping and mapping[entry["entity_id"]] != entry["label"]:
                raise SafetyError("conflicting labels in trusted cache")
            mapping[entry["entity_id"]] = entry["label"]
        value = mapping.get(entity_id)
        if not isinstance(value, str) or not value.strip():
            raise CacheMissNoTransport(f"trusted offline label unavailable for {entity_id}")
        return value

    def _finish(self, attempt_id: str, outcome: str, *, error_type: str | None = None) -> None:
        record = {"event": "ATTEMPT_FINISHED", "attempt_id": attempt_id,
                  "outcome": outcome, "error_type": error_type, "recorded_at_utc": _utc_now()}
        _append_fsync(self.journal_path, record)
        self.records.append(record)


REQUIRED_PREFIX_FIELDS = (
    "prompt_sha256", "prior_response_sha256", "ordered_dfs_stack",
    "active_binding_entity_ids", "frontier_entity_ids", "kg_query_result_sha256s",
    "prefix_sha256", "replay_status",
)


def validate_prefix_capture(state: dict[str, Any]) -> None:
    if not isinstance(state, dict) or any(field not in state for field in REQUIRED_PREFIX_FIELDS):
        raise SafetyError("prefix capture is incomplete")
    if state["replay_status"] != "COMPLETE":
        raise SafetyError("partial or failed prefix replay is ineligible")
    if not isinstance(state["ordered_dfs_stack"], list):
        raise SafetyError("DFS stack order must be captured as a list")
    for field in ("active_binding_entity_ids", "frontier_entity_ids"):
        values = state[field]
        if not isinstance(values, list) or any(not isinstance(value, str) or not re.fullmatch(r"(?:m|g)\.[A-Za-z0-9_]+", value) for value in values):
            raise SafetyError(f"{field} must contain only well-formed Freebase entity IDs")
    for field in ("prompt_sha256", "prior_response_sha256", "prefix_sha256"):
        if not _valid_digest(state[field]):
            raise SafetyError(f"{field} must be lowercase SHA-256")
    if not isinstance(state["kg_query_result_sha256s"], list) or any(not _valid_digest(x) for x in state["kg_query_result_sha256s"]):
        raise SafetyError("all prefix KG query results must have recorded hashes")
    expected = prefix_state_sha256(state)
    if state["prefix_sha256"] != expected:
        raise SafetyError("prefix hash does not match the captured decision, stack, binding, frontier and KG evidence")


def prefix_state_sha256(state: dict[str, Any]) -> str:
    """Hash only the common captured state, excluding arm-specific additions."""
    payload = {field: state[field] for field in REQUIRED_PREFIX_FIELDS if field != "prefix_sha256"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(canonical)


def validate_matched_pair(stop_state: dict[str, Any], one_hop_state: dict[str, Any]) -> None:
    validate_prefix_capture(stop_state)
    validate_prefix_capture(one_hop_state)
    for field in REQUIRED_PREFIX_FIELDS:
        if stop_state[field] != one_hop_state[field]:
            raise SafetyError(f"matched arms diverge at prefix field {field}")
    if stop_state.get("arm") != "STOP" or one_hop_state.get("arm") != "ONE_HOP":
        raise SafetyError("pair must contain STOP and ONE_HOP arms")
    if one_hop_state.get("added_relation_traversals") != 1:
        raise SafetyError("ONE_HOP must add exactly one relation traversal")


def empty_frontier_allowed(*, kg_status: str, result_ids: list[str]) -> bool:
    """An empty frontier is meaningful only after successful KG execution."""
    if kg_status != "ok":
        raise SafetyError("failed KG execution cannot be represented as an empty frontier")
    return len(result_ids) == 0
