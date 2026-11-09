#!/usr/bin/env python3
"""Approval-bound Iteration 23 live retrieval-only matched-prefix pilot.

The Iteration 22 runner is imported only for its frozen CoR instrumentation and
case logic. This entrypoint binds those helpers to the distinct repeat-2
manifest, the saved Iteration 23 approval, and shared no-retry meters.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
OUT = WORK / "RESULTS/iter_023_label_coverage_and_prefix_readiness"
MANIFEST_PATH = OUT / "PROPOSED_NEW_MANIFEST.json"
MANIFEST_SHA_PATH = OUT / "PROPOSED_NEW_MANIFEST.sha256"
AUTH_PATH = OUT / "AUTHORIZATION.json"
PROTECTED_PATH = WORK / "RESULTS/iter_021_matched_prefix_replay/PROTECTED_SOURCE_HASHES.json"
LABEL_CACHE_PATH = WORK / "RESULTS/iter_022_gated_matched_prefix/LOCAL_LABEL_CACHE.json"
ENDPOINT = "http://127.0.0.1:8890/sparql"
MODEL = "gpt-5.6-luna"
INPUT_RATE = 0.20
OUTPUT_RATE = 1.20
API_REQUEST_CAP = 60
INPUT_TOKEN_CAP = 7000
COMPLETION_TOKEN_CAP = 1024
SPEND_CAP_USD = 0.16
SPARQL_CAP = 140
TIME_CAP_SECONDS = 8 * 60 * 60
CALL_TIMEOUT_SECONDS = 120
RETRIES = 0

for path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "scripts/diagnostics/iter_021_matched_prefix",
             ROOT / "scripts/diagnostics/iter_022_gated_matched_prefix", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pilot as i22_pilot  # noqa: E402
from deadline import DeadlineGuard  # noqa: E402
from label_resolver import (  # noqa: E402
    BoundedLabelResolver,
    LabelResolutionStop,
    PhysicalQueryMeter,
    install_metered_freebase_backend,
)
from model_budget import ModelCaps, authorization_matches as model_authorization_matches  # noqa: E402

i22 = None


def _load_i22_helpers():
    """Defer CoRAgent/OpenAI SDK imports until after offline runner tests."""
    global i22
    if i22 is None:
        import importlib
        i22 = importlib.import_module("run_live_pilot")
    return i22


class Iter23Stop(RuntimeError):
    """Fail-closed stop raised by the new pilot entrypoint."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def relevant_source_hashes() -> dict[str, str]:
    names = [
        "scripts/diagnostics/iter_023_label_coverage_and_prefix_readiness/run_iter23_pilot.py",
        "scripts/diagnostics/iter_023_label_coverage_and_prefix_readiness/label_resolver.py",
        "scripts/diagnostics/iter_023_label_coverage_and_prefix_readiness/model_budget.py",
        "scripts/diagnostics/iter_023_label_coverage_and_prefix_readiness/deadline.py",
        "scripts/diagnostics/iter_022_gated_matched_prefix/run_live_pilot.py",
        "scripts/diagnostics/iter_022_gated_matched_prefix/pilot.py",
        "scripts/diagnostics/iter_021_matched_prefix/replay.py",
        "chain_of_relations/methods/cor/agent.py",
        "chain_of_relations/methods/cor/dfs_stack.py",
        "chain_of_relations/kg_backend/freebase/db_func.py",
        "chain_of_relations/kg_backend/freebase/query_template.yml",
        "chain_of_relations/llm_api.py",
    ]
    return {name: sha256_file(ROOT / name) for name in names}


def git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return result.stdout.strip()


def safe_load_project_env() -> None:
    """Read only the needed allowlisted keys; never shell-source .env."""
    allowed = {"OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL_NAME", "FREEBASE_SPARQL_ENDPOINT"}
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[7:].split("=", 1)
        key, value = key.strip(), value.strip()
        if key not in allowed or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]
        os.environ[key] = value


def frozen_inputs(*, allow_resume: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    digest = sha256_file(MANIFEST_PATH)
    expected = MANIFEST_SHA_PATH.read_text(encoding="utf-8").split()[0]
    if digest != expected:
        raise Iter23Stop("Iteration 23 frozen manifest file hash differs from its recorded digest")
    if digest != "ff57d84266295e7778b00e733fdba9e8a9232f4eed87971f7b0e029cefc331f2":
        raise Iter23Stop("Iteration 23 manifest is not the specifically approved manifest")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cases = manifest.get("cases", [])
    if (manifest.get("iteration") != 23 or len(cases) != 3
            or [case.get("repeat") for case in cases] != [2, 2, 2]
            or len({case.get("slot_id") for case in cases}) != 3
            or len({case.get("question_id") for case in cases}) != 3):
        raise Iter23Stop("manifest no longer matches the three unique fixed repeat-2 slots")
    approval = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    caps = ModelCaps(
        manifest_sha256=digest, request_cap=API_REQUEST_CAP,
        input_tokens_per_request=INPUT_TOKEN_CAP,
        completion_tokens_per_request=COMPLETION_TOKEN_CAP,
        estimated_spend_cap_usd=SPEND_CAP_USD,
        model=MODEL, input_usd_per_million=INPUT_RATE,
        output_usd_per_million=OUTPUT_RATE, retries=RETRIES,
    )
    if not model_authorization_matches(approval, caps):
        raise Iter23Stop("saved approval does not exactly authorize this manifest and API cap")
    exact_query_fields = {
        "read_only_sparql_query_cap": SPARQL_CAP,
        "endpoint_preflight_selects_included": 1,
        "time_cap_seconds": TIME_CAP_SECONDS,
        "per_call_timeout_seconds": CALL_TIMEOUT_SECONDS,
        "retries": 0,
    }
    if any(approval.get(key) != value for key, value in exact_query_fields.items()):
        raise Iter23Stop("saved approval does not exactly authorize the SPARQL and time limits")
    if not allow_resume and (approval.get("transport_opened") or approval.get("live_execution_started")):
        raise Iter23Stop("an earlier transport attempt consumed this authorization; refusing restart")
    if not allow_resume:
        for name in ("LIVE_RUN_STATE.json", "LIVE_CASES.csv", "RAW_CASE_TRACES.jsonl",
                     "REQUEST_JOURNAL.jsonl", "SPARQL_QUERY_JOURNAL.jsonl", "PILOT_RESULTS.json",
                     "GOLD_SCORING_ONLY.json", "PILOT_INTEGRITY.json", "DECISION.md"):
            if (OUT / name).exists():
                raise Iter23Stop(f"Iteration 23 live output already exists; do not rerun: {name}")
    protected = json.loads(PROTECTED_PATH.read_text(encoding="utf-8"))
    mismatches = [item["path"] for item in protected["files"]
                  if sha256_file(ROOT / item["path"]) != item["sha256_at_iteration_start"]]
    if mismatches:
        raise Iter23Stop(f"protected production source hash mismatch: {mismatches}")
    cache_doc = json.loads(LABEL_CACHE_PATH.read_text(encoding="utf-8"))
    labels_list = cache_doc.get("labels", [])
    label_cache = {str(item["entity_id"]): str(item["label"]) for item in labels_list}
    if len(labels_list) != 198 or len(label_cache) != 198:
        raise Iter23Stop("expected exactly 198 unique frozen Iteration 22 cache labels")
    if not all(str(k).startswith(("m.", "g.")) for k in label_cache):
        raise Iter23Stop("frozen label cache contains an invalid Freebase entity key")
    return manifest, approval, label_cache


def exact_iteration23_approval(record: Mapping[str, Any] | None) -> bool:
    caps = ModelCaps(
        manifest_sha256="ff57d84266295e7778b00e733fdba9e8a9232f4eed87971f7b0e029cefc331f2",
        request_cap=API_REQUEST_CAP, input_tokens_per_request=INPUT_TOKEN_CAP,
        completion_tokens_per_request=COMPLETION_TOKEN_CAP,
        estimated_spend_cap_usd=SPEND_CAP_USD, model=MODEL,
        input_usd_per_million=INPUT_RATE, output_usd_per_million=OUTPUT_RATE,
        retries=RETRIES,
    )
    return bool(
        model_authorization_matches(record, caps)
        and record.get("read_only_sparql_query_cap") == SPARQL_CAP
        and record.get("endpoint_preflight_selects_included") == 1
        and record.get("time_cap_seconds") == TIME_CAP_SECONDS
        and record.get("per_call_timeout_seconds") == CALL_TIMEOUT_SECONDS
    )


def configure_runtime(*, allow_resume: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, str]]:
    safe_load_project_env()
    if os.environ.get("MODEL_NAME") != MODEL:
        raise Iter23Stop("MODEL_NAME must equal the frozen gpt-5.6-luna model")
    if not os.environ.get("OPENAI_API_KEY"):
        raise Iter23Stop("OPENAI_API_KEY is unavailable; key value is never displayed")
    if (os.environ.get("OPENAI_BASE_URL") or "").strip():
        raise Iter23Stop("OPENAI_BASE_URL must be blank so this pilot uses the approved direct OpenAI endpoint")
    endpoint = os.environ.get("FREEBASE_SPARQL_ENDPOINT", ENDPOINT)
    if endpoint != ENDPOINT:
        raise Iter23Stop("FREEBASE_SPARQL_ENDPOINT must be the local forwarded endpoint http://127.0.0.1:8890/sparql")

    os.environ["OPENAI_MAX_RETRIES"] = "1"  # LLMAPI uses one total attempt; SDK retries are zero.
    os.environ["OPENAI_RETRY_INTERVAL"] = "0"
    os.environ["OPENAI_TIMEOUT"] = str(CALL_TIMEOUT_SECONDS)
    os.environ["SPARQL_TIMEOUT"] = str(CALL_TIMEOUT_SECONDS)
    os.environ["OPENAI_REASONING_EFFORT"] = "low"

    manifest, approval, label_cache = frozen_inputs(allow_resume=allow_resume)
    sources = relevant_source_hashes()
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("o200k_base")
        if len(encoding.encode("Iteration 23 tokenizer preflight")) + 128 > INPUT_TOKEN_CAP:
            raise Iter23Stop("tokenizer reservation preflight exceeds the approved input-token cap")
        if importlib.util.find_spec("SPARQLWrapper") is None or importlib.util.find_spec("openai") is None:
            raise Iter23Stop("OpenAI SDK or SPARQLWrapper dependency is not installed")
    except Iter23Stop:
        raise
    except Exception as exc:
        raise Iter23Stop(f"local runtime dependency preflight failed: {type(exc).__name__}: {exc}") from exc
    llm_source = (ROOT / "chain_of_relations/llm_api.py").read_text(encoding="utf-8")
    if ('"max_retries": 0' not in llm_source
            or "for attempt in range(1, self.max_retries + 1):" not in llm_source
            or 'self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES"' not in llm_source
            or 'self.timeout = float(os.getenv("OPENAI_TIMEOUT"' not in llm_source):
        raise Iter23Stop("cannot statically verify wrapper and SDK no-retry configuration in LLMAPI")
    return manifest, approval, label_cache, sources


def prepare_saved_authorization(approval: dict[str, Any], sources: Mapping[str, str]) -> None:
    """Persist the audited code version before transport; do not change approval scope."""
    if approval.get("transport_opened") or approval.get("live_execution_started"):
        raise Iter23Stop("cannot refresh code hashes after the first transport attempt")
    approval["code_version"] = {
        "git_head": git_head(),
        "relevant_source_sha256": dict(sources),
    }
    approval["runtime_preflight"] = {
        "recorded_at_utc": utc_now(),
        "model": MODEL,
        "openai_base_url": "direct_default_https://api.openai.com/v1",
        "model_wrapper_attempts": 1,
        "sdk_retries": 0,
        "sparql_retries": 0,
        "per_call_timeout_seconds": CALL_TIMEOUT_SECONDS,
        "pilot_deadline_seconds": TIME_CAP_SECONDS,
        "transport_opened": False,
    }
    write_json(AUTH_PATH, approval)


def verify_runtime_source_record(approval: Mapping[str, Any], sources: Mapping[str, str]) -> None:
    version = approval.get("code_version", {})
    if version.get("relevant_source_sha256") != dict(sources):
        raise Iter23Stop("saved pre-transport source hashes do not match the runtime implementation")
    if not version.get("git_head"):
        raise Iter23Stop("saved runtime code version has no Git HEAD record")


class AuthorizedQueryView:
    """Make label/navigation adapter calls pass exact approval before the shared meter."""

    def __init__(self, run: "PilotRuntime") -> None:
        self.run = run

    def execute(self, query: str, *, category: str, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self.run.require_authorization()
        return self.run.query_meter.execute(query, category=category, context=context)


class PilotRuntime:
    def __init__(self, approval: dict[str, Any], label_cache: Mapping[str, str], endpoint: str,
                 *, deadline_seconds: float = TIME_CAP_SECONDS,
                 initial_query_attempts: int = 0,
                 initial_query_events: list[dict[str, Any]] | None = None,
                 initial_query_categories: Mapping[str, int] | None = None) -> None:
        self.approval = approval
        self.endpoint = endpoint
        self.deadline = DeadlineGuard(deadline_seconds)
        self.recorder: i22.DurableRecorder | None = None
        self.current_transport: Callable[[str], Mapping[str, Any]] | None = None
        self.live_api_attempts = 0
        self.live_cost_usd = 0.0
        self.uncertain_live_cost_usd = 0.0
        self.api_events: list[dict[str, Any]] = []
        self.query_meter = PhysicalQueryMeter(
            hard_cap=SPARQL_CAP,
            transport=self._dispatch_query,
            transport_kind="approved_live_sparql_single_attempt",
            deadline=self.deadline,
        )
        self.query_meter.physical_attempts = initial_query_attempts
        self.query_meter.events = list(initial_query_events or [])
        self.query_meter.category_counts = dict(initial_query_categories or {})
        self.resolver = BoundedLabelResolver(
            label_cache=label_cache,
            meter=AuthorizedQueryView(self),
            max_batch_size=100,
            max_label_queries=12,
            cache_provenance="iteration22_archived_local_label_cache",
        )
        self.labels_outcome_start = 0

    @property
    def live_sparql_attempts(self) -> int:
        return self.query_meter.physical_attempts

    def require_authorization(self) -> None:
        caps = ModelCaps(
            manifest_sha256="ff57d84266295e7778b00e733fdba9e8a9232f4eed87971f7b0e029cefc331f2",
            request_cap=API_REQUEST_CAP, input_tokens_per_request=INPUT_TOKEN_CAP,
            completion_tokens_per_request=COMPLETION_TOKEN_CAP,
            estimated_spend_cap_usd=SPEND_CAP_USD, model=MODEL,
            input_usd_per_million=INPUT_RATE, output_usd_per_million=OUTPUT_RATE,
        )
        if (not model_authorization_matches(self.approval, caps)
                or self.approval.get("read_only_sparql_query_cap") != SPARQL_CAP
                or self.approval.get("time_cap_seconds") != TIME_CAP_SECONDS):
            raise Iter23Stop("exact saved Iteration 23 approval check failed")
        self.deadline.assert_open()

    def _dispatch_query(self, query: str) -> Mapping[str, Any]:
        self.require_authorization()
        if self.recorder is None or self.current_transport is None:
            raise Iter23Stop("SPARQL transport unavailable before durable recorder setup")
        latest = self.query_meter.events[-1]
        index = int(latest["query_index"])
        digest = str(latest["query_sha256"])
        category = str(latest["category"])
        self.recorder.meter_event({
            "kind": "SPARQL", "status": "ATTEMPT_STARTED", "query_index": index,
            "query_sha256": digest, "category": category, "retries": 0,
        })
        try:
            result = self.current_transport(query)
        except BaseException as exc:
            self.recorder.meter_event({
                "kind": "SPARQL", "status": "TRANSPORT_ERROR", "query_index": index,
                "query_sha256": digest, "category": category, "retries": 0,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        self.recorder.meter_event({
            "kind": "SPARQL", "status": "METERED" if result.get("status") == "ok" else "QUERY_ERROR",
            "query_index": index, "query_sha256": digest, "category": category, "retries": 0,
            "row_count": len(result.get("rows", [])),
        })
        return result

    def call_sparql_once(self, *, approval: Mapping[str, Any] | None, query_text: str,
                         one_attempt_transport: Callable[[str], Mapping[str, Any]],
                         category: str = "sidecar") -> Mapping[str, Any]:
        self.require_authorization()
        if approval is not self.approval and dict(approval or {}) != self.approval:
            raise Iter23Stop("SPARQL call received a different authorization object")
        previous = self.current_transport
        self.current_transport = one_attempt_transport
        try:
            result = self.query_meter.execute(query_text, category=category)
        except BaseException as exc:
            raise i22_pilot.PilotStop(f"one metered SPARQL attempt failed; no retry: {exc}") from exc
        finally:
            self.current_transport = previous
        return result

    def call_llm_once(self, *, approval: Mapping[str, Any] | None, input_tokens: int,
                      completion_tokens: int, transport: Callable[[], tuple[str | None, Mapping[str, Any]]],
                      prompt_sha256: str | None = None) -> tuple[str, Mapping[str, Any]]:
        self.require_authorization()
        if approval is not self.approval and dict(approval or {}) != self.approval:
            raise Iter23Stop("model call received a different authorization object")
        if not isinstance(input_tokens, int) or not 0 <= input_tokens <= INPUT_TOKEN_CAP:
            raise i22_pilot.PilotStop("input token reservation is outside the approved per-request cap")
        if not isinstance(completion_tokens, int) or not 1 <= completion_tokens <= COMPLETION_TOKEN_CAP:
            raise i22_pilot.PilotStop("completion cap is outside the approved per-request cap")
        if self.live_api_attempts >= API_REQUEST_CAP:
            raise i22_pilot.PilotStop("API request cap reached before transport")
        reserve = input_tokens * INPUT_RATE / 1_000_000 + completion_tokens * OUTPUT_RATE / 1_000_000
        if self.live_cost_usd + self.uncertain_live_cost_usd + reserve > SPEND_CAP_USD + 1e-12:
            raise i22_pilot.PilotStop("request worst-case reservation exceeds remaining approved spend")
        self.live_api_attempts += 1
        index = self.live_api_attempts
        if self.recorder is None:
            raise Iter23Stop("API transport blocked until durable recorder is ready")
        self.recorder.meter_event({
            "kind": "LLM", "status": "ATTEMPT_STARTED", "request_index": index,
            "input_token_reservation": input_tokens,
            "completion_token_cap": completion_tokens,
            "reserved_usd": reserve, "prompt_sha256": prompt_sha256, "retries": 0,
        })
        event = {"request_index": index, "prompt_sha256": prompt_sha256,
                 "input_token_reservation": input_tokens,
                 "completion_token_cap": completion_tokens,
                 "worst_case_reservation_usd": reserve, "retries": 0}
        self.api_events.append(event)
        try:
            content, usage = transport()
        except BaseException as exc:
            self.uncertain_live_cost_usd += reserve
            event.update({"status": "TRANSPORT_ERROR", "error": f"{type(exc).__name__}: {exc}"})
            self.recorder.meter_event({"kind": "LLM", "status": "TRANSPORT_ERROR",
                                       "request_index": index, "reserved_usd": reserve,
                                       "retries": 0, "error": event["error"]})
            raise i22_pilot.PilotStop("model call failed once; reservation held and retries disabled") from exc
        if not isinstance(usage, Mapping) or usage.get("error"):
            self.uncertain_live_cost_usd += reserve
            event.update({"status": "UNMETERED_OR_API_ERROR", "error": str(usage.get("error") if isinstance(usage, Mapping) else "missing usage")})
            self.recorder.meter_event({"kind": "LLM", "status": "API_ERROR", "request_index": index,
                                       "reserved_usd": reserve, "retries": 0, "error": event["error"]})
            raise i22_pilot.PilotStop("model response has missing/error usage; stop without retry")
        used_in, used_out = usage.get("input_tokens"), usage.get("output_tokens")
        if (not isinstance(used_in, int) or not isinstance(used_out, int)
                or used_in < 0 or used_out < 0 or used_in > input_tokens or used_out > completion_tokens):
            self.uncertain_live_cost_usd += reserve
            event.update({"status": "INVALID_OR_OVER_RESERVATION_USAGE"})
            self.recorder.meter_event({"kind": "LLM", "status": "INVALID_USAGE",
                                       "request_index": index, "reserved_usd": reserve, "retries": 0})
            raise i22_pilot.PilotStop("observed model usage is invalid or exceeds its reservation")
        actual = used_in * INPUT_RATE / 1_000_000 + used_out * OUTPUT_RATE / 1_000_000
        if self.live_cost_usd + self.uncertain_live_cost_usd + actual > SPEND_CAP_USD + 1e-12:
            self.uncertain_live_cost_usd += reserve
            raise i22_pilot.PilotStop("observed model cost exceeds approved spend cap")
        self.live_cost_usd += actual
        event.update({"status": "METERED", "input_tokens": used_in,
                      "output_tokens": used_out, "estimated_cost_usd": actual})
        self.recorder.meter_event({"kind": "LLM", "status": "METERED", "request_index": index,
                                   "input_tokens": used_in, "output_tokens": used_out,
                                   "cost_usd": actual, "retries": 0})
        if not isinstance(content, str) or not content.strip():
            raise i22_pilot.PilotStop("empty response was charged once; no retry")
        return content, usage


def install_iter23_backend(runtime: PilotRuntime):
    @contextmanager
    def install(backend: Any, _meter: Any, *, approval: Mapping[str, Any] | None,
                label_cache: Mapping[str, str],
                one_attempt_transport: Callable[[str], Mapping[str, Any]]):
        if dict(approval or {}) != runtime.approval:
            raise Iter23Stop("backend adapter received nonmatching approval")
        if dict(label_cache) != runtime.resolver.labels and not set(label_cache).issubset(runtime.resolver.labels):
            raise Iter23Stop("legacy cache and shared bounded resolver disagree")
        if runtime.current_transport is not None:
            raise Iter23Stop("nested transport context is prohibited")
        runtime.current_transport = one_attempt_transport
        try:
            with install_metered_freebase_backend(
                backend, meter=AuthorizedQueryView(runtime), resolver=runtime.resolver,
            ):
                yield backend
        finally:
            runtime.current_transport = None
    return install


def _case_for_runtime(case: Mapping[str, Any]) -> dict[str, Any]:
    """Keep fixed slot metadata, while ensuring the prior I22 helper has its role key."""
    return {
        "question_id": str(case["question_id"]),
        "repeat": int(case["repeat"]),
        "slot_id": str(case["slot_id"]),
        "role": str(case["selection_role"]),
        "evidence_pointer": f"iteration23:{case['slot_id']}",
    }


def _protected_hash_check() -> dict[str, Any]:
    protected = json.loads(PROTECTED_PATH.read_text(encoding="utf-8"))
    rows = [{"path": item["path"], "expected_sha256": item["sha256_at_iteration_start"],
             "actual_sha256": sha256_file(ROOT / item["path"]),
             "unchanged": sha256_file(ROOT / item["path"]) == item["sha256_at_iteration_start"]}
            for item in protected["files"]]
    return {"status": "PASS" if all(row["unchanged"] for row in rows) else "FAIL", "files": rows}


def run_preflight_only(*, prepare_record: bool) -> dict[str, Any]:
    manifest, approval, _cache, sources = configure_runtime()
    verify_runtime_source_record(approval, sources) if not prepare_record else None
    if prepare_record:
        prepare_saved_authorization(approval, sources)
        approval = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
        verify_runtime_source_record(approval, sources)
    check = _protected_hash_check()
    if check["status"] != "PASS":
        raise Iter23Stop("protected source hash audit failed")
    report = {
        "iteration": 23, "status": "PASS_NO_TRANSPORT_PREFLIGHT",
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "fixed_slots": [_case_for_runtime(case) for case in manifest["cases"]],
        "model": MODEL, "endpoint": ENDPOINT, "api_key_present": True,
        "openai_base_url": "direct_default_https://api.openai.com/v1",
        "wrapper_attempts": 1, "sdk_retries": 0, "sparql_retries": 0,
        "api_request_cap": API_REQUEST_CAP, "input_tokens_per_request_cap": INPUT_TOKEN_CAP,
        "completion_tokens_per_request_cap": COMPLETION_TOKEN_CAP,
        "estimated_spend_cap_usd": SPEND_CAP_USD, "sparql_physical_attempt_cap": SPARQL_CAP,
        "endpoint_preflight_selects_included": 1,
        "deadline_seconds": TIME_CAP_SECONDS, "per_call_timeout_seconds": CALL_TIMEOUT_SECONDS,
        "protected_production_sources": check,
        "git_head": git_head(), "relevant_source_sha256": sources,
        "api_or_sparql_transport_opened": False,
    }
    write_json(OUT / "LIVE_PREFLIGHT.json", report)
    print(json.dumps({key: report[key] for key in (
        "status", "manifest_sha256", "model", "api_key_present", "wrapper_attempts",
        "sdk_retries", "sparql_retries", "sparql_physical_attempt_cap",
        "deadline_seconds", "api_or_sparql_transport_opened",
    )}, indent=2))
    return report


def run_resume_preflight_only() -> dict[str, Any]:
    """Validate a no-replay continuation without opening either transport."""
    manifest, approval, _cache, sources = configure_runtime(allow_resume=True)
    _load_i22_helpers()
    resumed = verify_successful_preflight_resume(manifest, approval, sources)
    protected = _protected_hash_check()
    if protected["status"] != "PASS":
        raise Iter23Stop("protected production source audit failed before continuation")
    report = {
        "iteration": 23,
        "status": "PASS_RESUME_WITHOUT_TRANSPORT",
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "first_preflight_reused": True,
        "first_preflight_query_sha256": resumed["query_hash"],
        "physical_sparql_attempts_already_used": resumed["query_count"],
        "api_requests_already_used": 0,
        "fixed_question_slots_started": 0,
        "retries": 0,
        "elapsed_before_resume_seconds": round(resumed["elapsed_before_resume_seconds"], 3),
        "remaining_deadline_seconds": round(resumed["remaining_deadline_seconds"], 3),
        "changed_runtime_sources": resumed["changed_sources"],
        "protected_production_sources": protected,
        "transport_opened_by_this_check": False,
        "validated_at_utc": utc_now(),
    }
    write_json(OUT / "RESUME_PREFLIGHT.json", report)
    print(json.dumps({key: report[key] for key in (
        "status", "physical_sparql_attempts_already_used", "api_requests_already_used",
        "fixed_question_slots_started", "retries", "remaining_deadline_seconds",
        "transport_opened_by_this_check",
    )}, indent=2))
    return report


def _patch_i22_outputs() -> None:
    i22.OUT = OUT
    i22.AUTH_PATH = AUTH_PATH
    i22.LABEL_CACHE_PATH = LABEL_CACHE_PATH
    i22.RUN_STATE_PATH = OUT / "LIVE_RUN_STATE.json"
    i22.CASE_CSV_PATH = OUT / "LIVE_CASES.csv"
    i22.TRACE_JSONL_PATH = OUT / "RAW_CASE_TRACES.jsonl"
    i22.REQUEST_JSONL_PATH = OUT / "REQUEST_JOURNAL.jsonl"
    i22.QUERY_JSONL_PATH = OUT / "SPARQL_QUERY_JOURNAL.jsonl"
    i22.RESULTS_PATH = OUT / "PILOT_RESULTS.json"
    i22.INTEGRITY_PATH = OUT / "PILOT_INTEGRITY.json"
    i22.DECISION_PATH = OUT / "DECISION.md"
    i22.GOLD_PATH = OUT / "GOLD_SCORING_ONLY.json"
    i22.MANIFEST_SHA256 = sha256_file(MANIFEST_PATH)
    i22.CASE_FIELDS = [*i22.CASE_FIELDS, "slot_id"] if "slot_id" not in i22.CASE_FIELDS else i22.CASE_FIELDS
    i22_pilot.MANIFEST_SHA256 = i22.MANIFEST_SHA256
    i22_pilot.MAX_API_REQUESTS = API_REQUEST_CAP
    i22_pilot.MAX_INPUT_TOKENS = INPUT_TOKEN_CAP
    i22_pilot.MAX_COMPLETION_TOKENS = COMPLETION_TOKEN_CAP
    i22_pilot.MAX_API_COST_USD = SPEND_CAP_USD
    i22_pilot.MAX_SPARQL_QUERIES = SPARQL_CAP
    i22_pilot.INPUT_USD_PER_MILLION = INPUT_RATE
    i22_pilot.OUTPUT_USD_PER_MILLION = OUTPUT_RATE
    i22_pilot.approval_matches = exact_iteration23_approval
    i22.approval_matches = exact_iteration23_approval
    original_case_result = i22._case_result

    def case_result_with_slot(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_case_result(*args, **kwargs)
        case = args[1] if len(args) > 1 else kwargs.get("case", {})
        result["slot_id"] = case.get("slot_id") if isinstance(case, Mapping) else None
        return result

    i22._case_result = case_result_with_slot

    original_llm_adapter = i22.install_metered_llm

    @contextmanager
    def checked_llm_adapter(agent: Any, meter: Any, *, approval: Mapping[str, Any] | None,
                            token_counter: Callable[[str], int]):
        api = getattr(agent, "llm_api", None)
        if (api is None or getattr(api, "model_name", None) != MODEL
                or (getattr(api, "base_url", "") or "").strip()
                or getattr(api, "max_retries", None) != 1
                or getattr(api, "retry_interval", None) != 0
                or getattr(api, "timeout", CALL_TIMEOUT_SECONDS + 1) != CALL_TIMEOUT_SECONDS
                or getattr(getattr(api, "client", None), "max_retries", None) != 0):
            raise Iter23Stop("live LLM client differs from the approved model, direct endpoint, retry, or timeout settings")
        with original_llm_adapter(agent, meter, approval=approval, token_counter=token_counter):
            yield api

    i22.install_metered_llm = checked_llm_adapter


def _install_recorder_trace_provenance(recorder: Any, runtime: PilotRuntime) -> None:
    original = recorder.write_trace

    def write_trace(record: Mapping[str, Any]) -> None:
        enriched = dict(record)
        start = runtime.labels_outcome_start
        outcomes = runtime.resolver.outcomes[start:]
        ids = sorted({str(item.get("entity_id")) for item in outcomes if item.get("entity_id")})
        enriched["label_resolution"] = {
            "events": outcomes,
            "provenance_by_entity_id": {
                entity_id: dict(runtime.resolver.provenance[entity_id])
                for entity_id in ids if entity_id in runtime.resolver.provenance
            },
            "label_query_attempts_total": runtime.resolver.label_query_attempts,
        }
        original(enriched)
        runtime.labels_outcome_start = len(runtime.resolver.outcomes)

    recorder.write_trace = write_trace


def _install_consumption_journal(recorder: Any, approval: dict[str, Any]) -> None:
    original = recorder.meter_event

    def meter_event(event: Mapping[str, Any]) -> None:
        original(event)
        if event.get("status") == "ATTEMPT_STARTED" and event.get("kind") in {"LLM", "SPARQL"}:
            approval["authorization_consumed"] = True
            write_json(AUTH_PATH, approval)

    recorder.meter_event = meter_event


def _utc_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def verify_successful_preflight_resume(manifest: Mapping[str, Any], approval: Mapping[str, Any],
                                       current_sources: Mapping[str, str]) -> dict[str, Any]:
    """Allow only continuation after the successful preflight, with no request replay."""
    state_path = OUT / "LIVE_RUN_STATE.json"
    results_path = OUT / "PILOT_RESULTS.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    stopped = json.loads(results_path.read_text(encoding="utf-8"))
    saved_source = approval.get("code_version", {}).get("relevant_source_sha256", {})
    state_source = state.get("source_sha256", {})
    if saved_source != state_source:
        raise Iter23Stop("first-attempt source hashes do not match the saved approved runtime record")
    snapshot = json.loads((OUT / "SOURCE_SNAPSHOT.json").read_text(encoding="utf-8"))
    original_runner = "scripts/diagnostics/iter_023_label_coverage_and_prefix_readiness/run_iter23_pilot.py"
    if snapshot.get("iteration23_diagnostic_sources", {}).get(original_runner) != saved_source.get(original_runner):
        raise Iter23Stop("saved source snapshot does not match the first-attempt runner")
    changed = {
        name: {"first_attempt_sha256": digest, "continuation_sha256": current_sources.get(name)}
        for name, digest in saved_source.items()
        if current_sources.get(name) != digest
    }
    if set(changed) != {original_runner}:
        raise Iter23Stop(f"continuation may only change the runner after diagnosis; changed sources={sorted(changed)}")
    if state.get("status") != "PILOT_STOPPED" or "nested transport context is prohibited" not in state.get("stop_reason", ""):
        raise Iter23Stop("saved stop is not the known pre-case adapter-state failure")
    if stopped.get("status") != "PILOT_STOPPED" or stopped.get("api_requests_attempted") != 0:
        raise Iter23Stop("resume is allowed only before any model request or case execution")
    if stopped.get("sparql_physical_attempts") != 1 or stopped.get("manifest_sha256") != sha256_file(MANIFEST_PATH):
        raise Iter23Stop("first stopped run must have used only the one approved endpoint preflight")
    if approval.get("authorization_consumed") is not True or approval.get("transport_opened") is not True:
        raise Iter23Stop("saved authorization must already be consumed by exactly the successful preflight")
    actual = approval.get("actual_usage", {})
    if (actual.get("api_requests") != 0 or actual.get("read_only_sparql_attempts") != 1
            or actual.get("retries") != 0 or actual.get("estimated_cost_usd") != 0):
        raise Iter23Stop("saved authorization usage does not match the one-query, zero-API preflight")

    query_path = OUT / "SPARQL_QUERY_JOURNAL.jsonl"
    query_events = [json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(query_events) != 2:
        raise Iter23Stop("resume requires exactly one started and one completed SPARQL preflight record")
    expected_query = "SELECT * WHERE { ?s ?p ?o } LIMIT 1"
    if (query_events[0].get("status") != "ATTEMPT_STARTED"
            or query_events[1].get("status") != "ok"
            or query_events[0].get("query") != expected_query
            or query_events[1].get("query_sha256") != query_events[0].get("query_sha256")
            or query_events[0].get("retries") != 0
            or query_events[1].get("retries") != 0):
        raise Iter23Stop("endpoint preflight was not one successful, non-retried SELECT")
    request_events = [json.loads(line) for line in (OUT / "REQUEST_JOURNAL.jsonl").read_text().splitlines() if line.strip()]
    if (len(request_events) != 2
            or any(event.get("kind") != "SPARQL" for event in request_events)
            or [event.get("status") for event in request_events] != ["ATTEMPT_STARTED", "METERED"]):
        raise Iter23Stop("resume is prohibited because the request journal contains activity beyond preflight")
    with (OUT / "LIVE_CASES.csv").open(newline="", encoding="utf-8") as handle:
        if list(csv.DictReader(handle)):
            raise Iter23Stop("a fixed question slot already started; do not replay it")
    if (OUT / "RAW_CASE_TRACES.jsonl").read_text(encoding="utf-8").strip():
        raise Iter23Stop("question traces exist; refusing to repeat any case")
    gold = json.loads((OUT / "GOLD_SCORING_ONLY.json").read_text(encoding="utf-8"))
    if gold.get("manifest_sha256") != sha256_file(MANIFEST_PATH):
        raise Iter23Stop("saved scoring-only file belongs to a different manifest")

    original_start = _utc_datetime(str(state["started_at_utc"]))
    elapsed_before_resume = max(0.0, (datetime.now(timezone.utc) - original_start).total_seconds())
    remaining = TIME_CAP_SECONDS - elapsed_before_resume
    if remaining <= 0:
        raise Iter23Stop("the original eight-hour authorization deadline has expired")
    return {
        "state": state, "first_result": stopped,
        "started_at_utc": state["started_at_utc"],
        "elapsed_before_resume_seconds": elapsed_before_resume,
        "remaining_deadline_seconds": remaining,
        "query_hash": query_events[0]["query_sha256"],
        "query_count": 1,
        "changed_sources": changed,
    }


def record_resume_start(approval: dict[str, Any], current_sources: Mapping[str, str],
                        resumed: Mapping[str, Any]) -> None:
    first_result = OUT / "PILOT_RESULTS_FIRST_STOP.json"
    first_state = OUT / "LIVE_RUN_STATE_FIRST_STOP.json"
    if first_result.exists() or first_state.exists() or (OUT / "CONTINUATION_CODE_CHANGE.json").exists():
        raise Iter23Stop("continuation record already exists; refusing to start another resume")
    first_result.write_bytes((OUT / "PILOT_RESULTS.json").read_bytes())
    first_state.write_bytes((OUT / "LIVE_RUN_STATE.json").read_bytes())
    runner_path = "scripts/diagnostics/iter_023_label_coverage_and_prefix_readiness/run_iter23_pilot.py"
    saved = approval["code_version"]["relevant_source_sha256"]
    write_json(OUT / "CONTINUATION_CODE_CHANGE.json", {
        "iteration": 23,
        "reason": "The initial runner retained the endpoint-preflight transport in shared state and blocked the first case as nested.",
        "first_attempt_runner_sha256": saved[runner_path],
        "continuation_runner_sha256": current_sources[runner_path],
        "only_changed_runtime_source": runner_path,
        "first_endpoint_preflight_reused": True,
        "request_replayed": False,
        "api_requests_before_resume": 0,
        "sparql_attempts_before_resume": 1,
        "authorization_manifest_sha256": sha256_file(MANIFEST_PATH),
        "recorded_at_utc": utc_now(),
    })
    approval["continuation"] = {
        "started_at_utc": utc_now(),
        "reason": "resume same authorized run after local pre-case runner bug",
        "preflight_replayed": False,
        "question_slots_started_before_resume": 0,
        "previous_api_requests": 0,
        "previous_sparql_attempts": 1,
        "previous_attempt_code_sha256": saved[runner_path],
        "continuation_code_sha256": current_sources[runner_path],
    }
    write_json(AUTH_PATH, approval)


def resume_recorder(manifest: dict[str, Any], approval: dict[str, Any],
                    current_sources: Mapping[str, str], resumed: Mapping[str, Any]) -> Any:
    """Reattach to durable journals; preserve prior attempt and append at index 2."""
    recorder = object.__new__(i22.DurableRecorder)
    recorder.approval = approval
    recorder.api_attempts = 0
    recorder.sparql_attempts = 1
    recorder.actual_cost_usd = 0.0
    recorder.uncertain_cost_usd = 0.0
    recorder.current_case_id = None
    recorder.state = dict(resumed["state"])
    # The first stopped attempt is separately preserved. Do not carry its
    # terminal reason into the resumed execution's live/final state.
    recorder.state.pop("stop_reason", None)
    recorder.state.update({
        "status": "RUNNING",
        "resumed_at_utc": utc_now(),
        "resume_reason": "continue fixed slots after successful endpoint preflight without reissuing it",
        "continuation_source_sha256": dict(current_sources),
        "retries": 0,
    })
    i22.atomic_json(i22.RUN_STATE_PATH, recorder.state)
    return recorder


def _make_case_transport(runtime: PilotRuntime, recorder: Any) -> None:
    recorder.sparql_transport = i22.sparqlwrapper_one_attempt(
        ENDPOINT, approval=runtime.approval, timeout_seconds=CALL_TIMEOUT_SECONDS,
    )


def perform_approved_endpoint_preflight(runtime: PilotRuntime, recorder: Any) -> Mapping[str, Any]:
    """Run the sole preflight SELECT and always release its temporary transport."""
    recorder.current_case_id = "__endpoint_preflight__"
    runtime.current_transport = recorder.sparql_transport
    try:
        return runtime.call_sparql_once(
            approval=runtime.approval,
            query_text="SELECT * WHERE { ?s ?p ?o } LIMIT 1",
            one_attempt_transport=recorder.query_transport(recorder.sparql_transport),
            category="endpoint_preflight",
        )
    finally:
        runtime.current_transport = None
        recorder.current_case_id = None


def run_live(*, resume_after_preflight: bool = False) -> dict[str, Any]:
    manifest, approval, label_cache, sources = configure_runtime(allow_resume=resume_after_preflight)
    _load_i22_helpers()
    resumed = verify_successful_preflight_resume(manifest, approval, sources) if resume_after_preflight else None
    if not resume_after_preflight:
        verify_runtime_source_record(approval, sources)
    check = _protected_hash_check()
    if check["status"] != "PASS":
        raise Iter23Stop("protected production source hash audit failed before transport")
    rows = i22.load_case_rows(manifest)
    _patch_i22_outputs()
    preflight_hash = (resumed or {}).get("query_hash")
    initial_events = ([{
        "event": "REUSED_SUCCESSFUL_AUTHORIZED_PREFLIGHT",
        "query_index": 1,
        "query_sha256": preflight_hash,
        "category": "endpoint_preflight",
        "transport_invoked": True,
        "retries": 0,
    }] if resumed else [])
    runtime = PilotRuntime(
        approval, label_cache, ENDPOINT,
        deadline_seconds=(resumed or {}).get("remaining_deadline_seconds", TIME_CAP_SECONDS),
        initial_query_attempts=(resumed or {}).get("query_count", 0),
        initial_query_events=initial_events,
        initial_query_categories=({"endpoint_preflight": 1} if resumed else {}),
    )
    model_settings = {
        "model": MODEL, "temperature": i22.TEMPERATURE_REASONING,
        "relation_width": i22.RELATION_WIDTH,
        "sample_relation_threshold": i22.SAMPLE_RELATION_THRESHOLD,
        "remove_unnecessary_rel": True,
        "temperature_exploration": i22.TEMPERATURE_EXPLORATION,
        "temperature_reasoning": i22.TEMPERATURE_REASONING,
        "max_completion_tokens": COMPLETION_TOKEN_CAP,
        "reasoning_effort": "low",
    }
    scoring_only = {
        str(row["id"]): [str(value) for value in row.get("answer_ids", []) if str(value).strip()]
        for row in rows
    }
    gold_document = {
        "evidence_class": "SCORING_ONLY_NEVER_PASSED_TO_INFERENCE",
        "manifest_sha256": i22.MANIFEST_SHA256,
        "cases": scoring_only,
    }
    if resumed:
        existing_gold = json.loads(i22.GOLD_PATH.read_text(encoding="utf-8"))
        if existing_gold != gold_document:
            raise Iter23Stop("scoring-only file changed during resume; refusing to overwrite it")
    else:
        i22.atomic_json(i22.GOLD_PATH, gold_document)
    if resumed:
        record_resume_start(approval, sources, resumed)
        recorder = resume_recorder(manifest, approval, sources, resumed)
    else:
        recorder = i22.DurableRecorder(approval, manifest, sources)
    runtime.recorder = recorder
    _install_consumption_journal(recorder, approval)
    recorder.sparql_transport = i22_pilot.sparqlwrapper_one_attempt(
        ENDPOINT, approval=approval, timeout_seconds=CALL_TIMEOUT_SECONDS,
    )
    runtime.current_transport = None
    _install_recorder_trace_provenance(recorder, runtime)
    outcomes: list[dict[str, Any]] = []
    started = time.monotonic()
    overall = "PILOT_PARTIAL"
    try:
        if not resumed:
            # The one approved endpoint preflight is the first physical SELECT.
            perform_approved_endpoint_preflight(runtime, recorder)
            if time.monotonic() - started > TIME_CAP_SECONDS:
                raise Iter23Stop("pilot deadline reached during endpoint preflight")
        for index, (frozen_case, row) in enumerate(zip(manifest["cases"], rows), start=1):
            runtime.require_authorization()
            case = _case_for_runtime(frozen_case)
            runtime.labels_outcome_start = len(runtime.resolver.outcomes)
            recorder.current_case_id = str(case["question_id"])
            _make_case_transport(runtime, recorder)
            i22.install_metered_backend = install_iter23_backend(runtime)
            try:
                outcome = i22.run_case(
                    index, case, row, scoring_only[str(row["id"])], model_settings,
                    approval, label_cache, runtime, recorder, ENDPOINT,
                )
            except LabelResolutionStop as exc:
                # Missing/ambiguous labels invalidate only this slot; transport,
                # deadline, and global-budget failures terminate the whole run.
                outcomes.append({
                    "case_index": index, "question_id": case["question_id"],
                    "repeat": case["repeat"], "slot_id": case["slot_id"],
                    "status": "INELIGIBLE", "reason": "LABEL_RESOLUTION_STOP",
                    "error": str(exc), "missing_ids": list(exc.ids),
                    "label_resolution_events": runtime.resolver.outcomes[runtime.labels_outcome_start:],
                })
                recorder.write_trace({"case": case, "status": "INELIGIBLE",
                                      "reason": "LABEL_RESOLUTION_STOP", "error": str(exc),
                                      "missing_ids": list(exc.ids)})
                recorder.write_case(outcomes[-1])
                if any(token in exc.reason.lower() for token in ("failed", "budget", "deadline")):
                    raise Iter23Stop(f"label resolver stopped on fatal condition: {exc.reason}") from exc
                continue
            outcomes.append(dict(outcome))
            if outcome.get("status") == "INTEGRITY_BLOCKED":
                raise Iter23Stop("gold-ID prompt integrity guard stopped the run")
        overall = "PILOT_COMPLETE" if len(outcomes) == 3 else "PILOT_PARTIAL"
    except BaseException as exc:
        overall = "PILOT_STOPPED"
        recorder.state["status"] = "PILOT_STOPPED"
        recorder.state["stop_reason"] = f"{type(exc).__name__}: {exc}"
        recorder.state["finished_at_utc"] = i22.utc_now()
        i22.atomic_json(i22.RUN_STATE_PATH, recorder.state)
        if (i22.CASE_CSV_PATH).exists():
            with i22.CASE_CSV_PATH.open(newline="", encoding="utf-8") as handle:
                outcomes = list(csv.DictReader(handle))
    finally:
        elapsed = time.monotonic() - started
    results = {
        "iteration": 23, "status": overall, "evidence_class": "LIVE_RETRIEVAL_ONLY_DEVELOPMENT_PILOT",
        "manifest_sha256": i22.MANIFEST_SHA256,
        "fixed_slots": [_case_for_runtime(case) for case in manifest["cases"]],
        "case_results": outcomes,
        "manifest_case_count": 3,
        "eligible_cases": sum(str(case.get("status", "")).startswith("VALID_") for case in outcomes),
        "valid_matched_prefix_cases": sum(str(case.get("status", "")).startswith("VALID_") for case in outcomes),
        "api_requests_attempted": runtime.live_api_attempts,
        "sparql_physical_attempts": runtime.live_sparql_attempts,
        "label_query_attempts": runtime.resolver.label_query_attempts,
        "estimated_api_cost_usd": round(runtime.live_cost_usd, 12),
        "uncertain_reserved_cost_usd": round(runtime.uncertain_live_cost_usd, 12),
        "elapsed_seconds": round((datetime.now(timezone.utc) - _utc_datetime(
            (resumed or {}).get("started_at_utc", recorder.state.get("started_at_utc", utc_now()))
        )).total_seconds(), 3),
        "resumed_after_preflight_without_replay": bool(resumed),
        "first_attempt_stop_reason": (resumed or {}).get("state", {}).get("stop_reason"),
        "authorized_caps": {
            "api_requests": API_REQUEST_CAP, "input_tokens_per_request": INPUT_TOKEN_CAP,
            "completion_tokens_per_request": COMPLETION_TOKEN_CAP,
            "estimated_api_spend_usd": SPEND_CAP_USD,
            "read_only_sparql_physical_attempts": SPARQL_CAP,
            "endpoint_preflight_selects_included": 1,
            "wall_clock_seconds": TIME_CAP_SECONDS,
            "per_call_timeout_seconds": CALL_TIMEOUT_SECONDS,
            "retries": 0,
        },
        "no_answer_filter_or_answer_generation_after_boundary": True,
        "gold_ids_scoring_only": True,
        "question_identities_overlap_prior_development": True,
        "finished_at_utc": i22.utc_now(),
    }
    approval["authorization_consumed"] = bool(approval.get("transport_opened"))
    approval["actual_usage"] = {
        "api_requests": runtime.live_api_attempts,
        "input_tokens": sum(int(item.get("input_tokens", 0)) for item in runtime.api_events),
        "completion_tokens": sum(int(item.get("output_tokens", 0)) for item in runtime.api_events),
        "estimated_cost_usd": round(runtime.live_cost_usd, 12),
        "uncertain_reserved_cost_usd": round(runtime.uncertain_live_cost_usd, 12),
        "read_only_sparql_attempts": runtime.live_sparql_attempts,
        "retries": 0,
        "valid_matched_prefix_cases": results["valid_matched_prefix_cases"],
    }
    write_json(AUTH_PATH, approval)
    i22.atomic_json(i22.RESULTS_PATH, results)
    recorder.state["status"] = overall
    recorder.state["finished_at_utc"] = results["finished_at_utc"]
    i22.atomic_json(i22.RUN_STATE_PATH, recorder.state)
    print(json.dumps({key: results[key] for key in (
        "status", "manifest_case_count", "eligible_cases", "api_requests_attempted",
        "sparql_physical_attempts", "label_query_attempts", "estimated_api_cost_usd",
        "uncertain_reserved_cost_usd", "elapsed_seconds",
    )}, indent=2))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate local configuration and authorization without transport")
    parser.add_argument("--prepare-authorization-record", action="store_true",
                        help="record source hashes before transport; does not call API or SPARQL")
    parser.add_argument("--run", action="store_true", help="consume the approved Iteration 23 pilot once")
    parser.add_argument("--resume-after-preflight", action="store_true",
                        help="continue only after the single successful endpoint SELECT, without replaying it")
    parser.add_argument("--resume-preflight-only", action="store_true",
                        help="verify a saved successful preflight and continuation safety without transport")
    args = parser.parse_args()
    if sum((args.preflight_only, args.prepare_authorization_record, args.run,
            args.resume_after_preflight, args.resume_preflight_only)) != 1:
        parser.error("choose exactly one preflight, prepare, run, or resume mode")
    try:
        if args.prepare_authorization_record:
            run_preflight_only(prepare_record=True)
        elif args.preflight_only:
            run_preflight_only(prepare_record=False)
        elif args.resume_after_preflight:
            run_live(resume_after_preflight=True)
        elif args.resume_preflight_only:
            run_resume_preflight_only()
        else:
            run_live()
    except (Iter23Stop, i22_pilot.PilotStop, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ITERATION23_STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
