#!/usr/bin/env python3
"""One-shot, approval-gated driver for the fixed Iteration 22 live pilot."""
from __future__ import annotations

import csv
import hashlib
import json
import linecache
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
ITER21_DIR = ROOT / "scripts/diagnostics/iter_021_matched_prefix"
if str(ITER21_DIR) not in sys.path:
    sys.path.insert(0, str(ITER21_DIR))

from chain_of_relations.schema import Entity, Relation  # noqa: E402
from chain_of_relations.kg_backend import get_backend  # noqa: E402
from chain_of_relations.methods.cor.agent import CoRAgent  # noqa: E402
from chain_of_relations.tools.entity_search import _build_entities_extract  # noqa: E402
from replay import (  # noqa: E402
    attach_binding_projection,
    binding_projection_query,
    build_pair,
    canonical_json,
    capture_cor_method_outputs,
    fetch_binding_projection,
    is_complete_prefix,
    sha256_json,
    _node_payload,
)
from pilot import (  # noqa: E402
    HardCapMeter,
    LabelCacheMiss,
    MANIFEST_SHA256,
    MAX_API_COST_USD,
    MAX_API_REQUESTS,
    MAX_COMPLETION_TOKENS,
    MAX_INPUT_TOKENS,
    MAX_SPARQL_QUERIES,
    PilotStop,
    approval_matches,
    estimated_cost,
    install_metered_backend,
    install_metered_llm,
    sparqlwrapper_one_attempt,
)

WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
OUT = WORK / "RESULTS/iter_022_gated_matched_prefix"
MANIFEST_PATH = WORK / "RESULTS/iter_021_matched_prefix_replay/PROPOSED_DEVELOPMENT_MANIFEST.json"
PROTECTED_PATH = WORK / "RESULTS/iter_021_matched_prefix_replay/PROTECTED_SOURCE_HASHES.json"
AUTH_PATH = OUT / "AUTHORIZATION.json"
LABEL_CACHE_PATH = OUT / "LOCAL_LABEL_CACHE.json"
DATA_PATH = ROOT / "outputs/subsets/cwq_branch_2.jsonl"
MODEL_EXPECTED = "gpt-5.6-luna"
MAX_DEPTH = 3
RELATION_WIDTH = 3
SAMPLE_RELATION_THRESHOLD = 1000
TEMPERATURE_EXPLORATION = 0.3
TEMPERATURE_REASONING = 0.1
SPARQL_TIMEOUT_SECONDS = 120
OPENAI_TIMEOUT_SECONDS = 120
RUN_STATE_PATH = OUT / "PILOT_RUN_STATE.json"
CASE_CSV_PATH = OUT / "PILOT_CASES.csv"
TRACE_JSONL_PATH = OUT / "RAW_CASE_TRACES.jsonl"
REQUEST_JSONL_PATH = OUT / "REQUEST_JOURNAL.jsonl"
QUERY_JSONL_PATH = OUT / "SPARQL_QUERY_JOURNAL.jsonl"
RESULTS_PATH = OUT / "PILOT_RESULTS.json"
INTEGRITY_PATH = OUT / "PILOT_INTEGRITY.json"
DECISION_PATH = OUT / "DECISION.md"
GOLD_PATH = OUT / "GOLD_SCORING_ONLY.json"

CASE_FIELDS = [
    "case_index", "question_id", "repeat", "role", "status", "reason",
    "boundary_action", "boundary_depth", "prefix_hash", "prefix_complete",
    "selected_relation_id", "relation_direction", "candidate_ids_before_prune",
    "candidate_ids_after_prune", "stop_binding_id_count", "one_hop_entity_ids",
    "new_entity_ids", "new_gold_ids", "query_attempts", "api_attempts",
    "estimated_cost_usd", "error",
]


class BoundaryCaptured(BaseException):
    def __init__(self, snapshot: Mapping[str, Any]):
        self.snapshot = dict(snapshot)
        super().__init__("first depth-3 reasoning decision captured")


class PreBoundaryAnswerStage(BaseException):
    def __init__(self, stage: str):
        self.stage = stage
        super().__init__(f"pilot stopped before prohibited answer stage: {stage}")


class CaseIneligible(Exception):
    """A case cannot support exact-prefix measurement; the next fixed case may proceed."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(dict(value)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def safe_load_project_env() -> None:
    """Read only the allowlisted single-line settings; never shell-source .env."""
    allowed = {
        "OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL_NAME",
        "FREEBASE_SPARQL_ENDPOINT", "WIKIDATA_SPARQL_ENDPOINT",
    }
    env_path = ROOT / ".env"
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[7:].split("=", 1)
        key = key.strip()
        if key not in allowed or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]
        os.environ[key] = value


def source_hashes() -> dict[str, str]:
    paths = [
        "scripts/diagnostics/iter_022_gated_matched_prefix/pilot.py",
        "scripts/diagnostics/iter_022_gated_matched_prefix/run_live_pilot.py",
        "scripts/diagnostics/iter_021_matched_prefix/replay.py",
        "chain_of_relations/methods/cor/agent.py",
        "chain_of_relations/methods/cor/dfs_stack.py",
        "chain_of_relations/kg_backend/freebase/db_func.py",
        "chain_of_relations/kg_backend/freebase/query_template.yml",
        "chain_of_relations/llm_api.py",
    ]
    return {name: sha256_file(ROOT / name) for name in paths}


def verify_preflight() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    if sha256_file(MANIFEST_PATH) != MANIFEST_SHA256:
        raise PilotStop("frozen manifest hash changed")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if len(manifest.get("cases", [])) != 3 or any(c.get("repeat") != 1 for c in manifest["cases"]):
        raise PilotStop("fixed manifest scope is not exactly three cases, repeat 1")
    approval = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    if not approval_matches(approval):
        raise PilotStop("saved approval does not match the frozen manifest and all exact caps")
    if approval.get("manifest_sha256") != sha256_file(MANIFEST_PATH):
        raise PilotStop("authorization and manifest hashes differ")
    if approval.get("transport_opened") is not False or approval.get("live_execution_started") is not False:
        raise PilotStop("authorization journal shows an earlier live attempt; audit before resuming")
    saved_code = approval.get("code_version", {}).get("relevant_source_sha256", {})
    current_code = source_hashes()
    if saved_code != current_code:
        raise PilotStop("authorization code hashes do not match the current pilot implementation")
    protected = json.loads(PROTECTED_PATH.read_text(encoding="utf-8"))
    protected_results = []
    for item in protected["files"]:
        actual = sha256_file(ROOT / item["path"])
        protected_results.append(actual == item["sha256_at_iteration_start"])
    if not all(protected_results):
        raise PilotStop("protected production source hash mismatch")
    if any(path.exists() for path in (RUN_STATE_PATH, CASE_CSV_PATH, TRACE_JSONL_PATH,
                                      REQUEST_JSONL_PATH, QUERY_JSONL_PATH, RESULTS_PATH,
                                      INTEGRITY_PATH, DECISION_PATH, GOLD_PATH)):
        raise PilotStop("pilot execution artifacts already exist; audit them before any resume")
    cache_document = json.loads(LABEL_CACHE_PATH.read_text(encoding="utf-8"))
    if cache_document.get("manifest_sha256") != MANIFEST_SHA256:
        raise PilotStop("frozen local label cache was made from a different manifest")
    labels = {str(item["entity_id"]): str(item["label"]) for item in cache_document.get("labels", [])}
    if len(labels) != 198 or len(cache_document.get("labels", [])) != 198:
        raise PilotStop("frozen label cache must contain exactly 198 unambiguous mappings")
    return manifest, approval, labels


def load_case_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    with DATA_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("id") in {str(case["question_id"]) for case in manifest["cases"]}:
                by_id[str(row["id"])] = row
    ordered = []
    for case in manifest["cases"]:
        case_id = str(case["question_id"])
        if case_id not in by_id:
            raise PilotStop(f"frozen manifest question is missing from inference-safe CWQ subset: {case_id}")
        row = by_id[case_id]
        if not row.get("topic_entities") or len(row.get("topic_entities", [])) != len(row.get("topic_entity_names", [])):
            raise PilotStop(f"inference-safe topic entities are missing or misaligned for {case_id}")
        ordered.append(row)
    return ordered


def conservative_token_reservation(text: str) -> int:
    """Reserve the full cap; reject if a conservative prompt estimate exceeds it."""
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("o200k_base")
        estimated = len(encoding.encode(text)) + 128
    except Exception as exc:
        raise PilotStop("cannot verify the input-token reservation without tiktoken") from exc
    if estimated > MAX_INPUT_TOKENS:
        return MAX_INPUT_TOKENS + 1
    return MAX_INPUT_TOKENS


class DurableRecorder:
    def __init__(self, approval: dict[str, Any], manifest: dict[str, Any], source: dict[str, str]):
        self.approval = approval
        self.api_attempts = 0
        self.sparql_attempts = 0
        self.actual_cost_usd = 0.0
        self.uncertain_cost_usd = 0.0
        self.current_case_id: str | None = None
        RUN_STATE_PATH.touch(exist_ok=False)
        CASE_CSV_PATH.touch(exist_ok=False)
        TRACE_JSONL_PATH.touch(exist_ok=False)
        REQUEST_JSONL_PATH.touch(exist_ok=False)
        QUERY_JSONL_PATH.touch(exist_ok=False)
        with CASE_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=CASE_FIELDS).writeheader()
            handle.flush()
            os.fsync(handle.fileno())
        self.state = {
            "status": "RUNNING", "started_at_utc": utc_now(),
            "manifest_sha256": MANIFEST_SHA256, "git_head": approval["code_version"]["git_head"],
            "source_sha256": source, "case_order": [x["question_id"] for x in manifest["cases"]],
            "case_results_written": 0, "api_attempts": 0, "sparql_attempts": 0,
            "actual_cost_usd": 0.0, "uncertain_cost_usd": 0.0,
            "retries": 0, "transport_opened": False,
        }
        atomic_json(RUN_STATE_PATH, self.state)

    def meter_event(self, event: Mapping[str, Any]) -> None:
        item = {**dict(event), "recorded_at_utc": utc_now(), "case_id": self.current_case_id}
        append_jsonl(REQUEST_JSONL_PATH, item)
        if item.get("kind") == "LLM" and item.get("status") == "ATTEMPT_STARTED":
            self.api_attempts = max(self.api_attempts, int(item["request_index"]))
        if item.get("kind") == "SPARQL" and item.get("status") == "ATTEMPT_STARTED":
            self.sparql_attempts = max(self.sparql_attempts, int(item["query_index"]))
        if item.get("status") == "METERED":
            self.actual_cost_usd += float(item.get("cost_usd", 0.0))
        if item.get("kind") == "LLM" and item.get("status") in {
            "TRANSPORT_ERROR", "UNMETERED", "API_ERROR", "INVALID_USAGE", "RESERVATION_EXCEEDED"
        }:
            self.uncertain_cost_usd += float(item.get("reserved_usd", 0.0))
        if item.get("status") == "ATTEMPT_STARTED":
            self.approval["transport_opened"] = True
            self.approval["live_execution_started"] = True
            self.approval.setdefault("first_transport_attempt_at_utc", item["recorded_at_utc"])
            atomic_json(AUTH_PATH, self.approval)
            self.state["transport_opened"] = True
        self.state.update({
            "last_updated_at_utc": item["recorded_at_utc"],
            "api_attempts": self.api_attempts,
            "sparql_attempts": self.sparql_attempts,
            "actual_cost_usd": round(self.actual_cost_usd, 12),
            "uncertain_cost_usd": round(self.uncertain_cost_usd, 12),
        })
        atomic_json(RUN_STATE_PATH, self.state)

    def query_transport(self, transport):
        def audited(query_text: str) -> Mapping[str, Any]:
            query_hash = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
            start = {
                "status": "ATTEMPT_STARTED", "query_index": self.sparql_attempts,
                "case_id": self.current_case_id, "query_sha256": query_hash,
                "query": query_text, "recorded_at_utc": utc_now(), "retries": 0,
            }
            append_jsonl(QUERY_JSONL_PATH, start)
            try:
                result = transport(query_text)
            except BaseException as exc:
                append_jsonl(QUERY_JSONL_PATH, {
                    **start, "status": "TRANSPORT_ERROR", "error": f"{type(exc).__name__}: {exc}",
                    "finished_at_utc": utc_now(),
                })
                raise
            append_jsonl(QUERY_JSONL_PATH, {
                **start, "status": result.get("status"), "rows": result.get("rows", []),
                "row_count": len(result.get("rows", [])), "finished_at_utc": utc_now(),
            })
            return result
        return audited

    def write_case(self, row: Mapping[str, Any]) -> None:
        with CASE_CSV_PATH.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CASE_FIELDS)
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (list, dict))
                             else row.get(key) for key in CASE_FIELDS})
            handle.flush()
            os.fsync(handle.fileno())
        self.state["case_results_written"] += 1
        atomic_json(RUN_STATE_PATH, self.state)

    def write_trace(self, record: Mapping[str, Any]) -> None:
        append_jsonl(TRACE_JSONL_PATH, record)


def snapshot_at_first_depth_decision(frame, *, question_id: str, repeat: int,
                                     evidence_pointer: str, model_settings: dict[str, Any],
                                     method_events: list[dict[str, Any]], prompt_history: list[dict[str, Any]],
                                     sparql_history: list[dict[str, Any]]) -> dict[str, Any]:
    loc = frame.f_locals
    path = list(getattr(loc.get("current_state"), "path", []) or [])
    stack = loc.get("stack")
    stack_rows = [{
        "path": [_node_payload(node) for node in getattr(item, "path", [])],
        "depth": getattr(item, "depth", None),
    } for item in getattr(stack, "items", []) or []]
    current_path = [_node_payload(node) for node in path]
    captured_events = [dict(event) for event in method_events
                       if len(event.get("relation_chain", [])) <= MAX_DEPTH]
    kg_cache = [dict(item) for event in captured_events for item in event.get("kg_cache", [])]
    visited = {item["id"] for item in current_path if item.get("kind") == "entity" and item.get("id")}
    for state in stack_rows:
        visited.update(node["id"] for node in state["path"] if node.get("kind") == "entity" and node.get("id"))
    for event in sparql_history:
        visited.update(str(value) for value in event.get("candidate_entity_ids", []) if value)
    visited.update(str(value) for value in loc.get("entity_ids", []) or [])
    return {
        "question_id": question_id, "repeat": repeat, "boundary_depth": MAX_DEPTH,
        "boundary_action": loc.get("action"), "boundary_response": loc.get("response"),
        "route": "cor", "model_settings": dict(model_settings),
        "temperature": TEMPERATURE_REASONING,
        "decisions": json.loads(json.dumps(prompt_history)),
        "method_events": captured_events,
        "sparql_history": json.loads(json.dumps(sparql_history)),
        "kg_cache": kg_cache, "current_path": current_path, "dfs_stack": stack_rows,
        "active_binding_rows": None,
        "frontier_ids": [str(value) for value in loc.get("entity_ids", []) or []],
        "visited_node_ids": sorted(visited),
        "selected_relation_id": current_path[-1].get("id") if current_path else None,
        "relation_direction": current_path[-1].get("direction") if current_path else None,
        "candidate_relation_ids_before_prune": None,
        "candidate_relation_ids_after_prune": None,
        "answer_source": "OTHER", "filter_input_ids": None,
        "filter_output_ids": None, "final_answer_ids": [],
        "log_completeness": "BOUNDARY_CAPTURED_BINDING_PROJECTION_PENDING",
        "evidence_pointer": evidence_pointer,
    }


def capture_first_depth_decision(agent: CoRAgent, row: Mapping[str, Any], topic_entities: list[Entity],
                                 case_meta: Mapping[str, Any], model_settings: dict[str, Any]) -> dict[str, Any]:
    prompt_history: list[dict[str, Any]] = []
    sparql_history: list[dict[str, Any]] = []
    step_history: list[dict[str, Any]] = []
    method_events: list[dict[str, Any]] = []
    if sys.gettrace() is not None:
        raise PilotStop("refusing to replace an existing Python trace hook")

    def stop_at_boundary(frame, event, arg):
        if event != "line" or frame.f_code.co_name != "answer":
            return stop_at_boundary
        if not frame.f_code.co_filename.endswith("chain_of_relations/methods/cor/agent.py"):
            return stop_at_boundary
        loc = frame.f_locals
        if loc.get("current_depth") != MAX_DEPTH:
            return stop_at_boundary
        if linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip() != "if action is None:":
            return stop_at_boundary
        snapshot = snapshot_at_first_depth_decision(
            frame, question_id=str(case_meta["question_id"]), repeat=int(case_meta["repeat"]),
            evidence_pointer=str(case_meta["evidence_pointer"]), model_settings=model_settings,
            method_events=method_events, prompt_history=prompt_history, sparql_history=sparql_history,
        )
        raise BoundaryCaptured(snapshot)

    with capture_cor_method_outputs(agent, method_events):
        original_validate = agent.validate
        original_generate = agent.generate_directly

        def blocked_validate(*args, **kwargs):
            raise PreBoundaryAnswerStage("answer_filter")

        def blocked_generate(*args, **kwargs):
            raise PreBoundaryAnswerStage("direct_answer_generation")

        agent.validate = blocked_validate
        agent.generate_directly = blocked_generate
        try:
            sys.settrace(stop_at_boundary)
            try:
                result = agent.answer(
                    question=str(row["question"]), topic_entities=topic_entities,
                    prompt_history=prompt_history, sparql_history=sparql_history,
                    step_history=step_history,
                )
            except BoundaryCaptured as captured:
                return {
                    "snapshot": captured.snapshot, "prompt_history": prompt_history,
                    "sparql_history": sparql_history, "step_history": step_history,
                    "method_events": method_events, "result": None,
                    "capture_status": "BOUNDARY_CAPTURED",
                }
            except LabelCacheMiss as exc:
                return {
                    "snapshot": None, "prompt_history": prompt_history,
                    "sparql_history": sparql_history, "step_history": step_history,
                    "method_events": method_events, "result": None,
                    "capture_status": "LABEL_CACHE_MISS", "error": str(exc),
                    "missing_ids": exc.missing_ids,
                }
            except PreBoundaryAnswerStage as exc:
                return {
                    "snapshot": None, "prompt_history": prompt_history,
                    "sparql_history": sparql_history, "step_history": step_history,
                    "method_events": method_events, "result": None,
                    "capture_status": "PRE_BOUNDARY_ANSWER_STAGE", "error": exc.stage,
                }
            except PilotStop as exc:
                return {
                    "snapshot": None, "prompt_history": prompt_history,
                    "sparql_history": sparql_history, "step_history": step_history,
                    "method_events": method_events, "result": None,
                    "capture_status": "PILOT_STOP", "error": str(exc),
                }
            finally:
                sys.settrace(None)
        finally:
            agent.validate = original_validate
            agent.generate_directly = original_generate
    return {
        "snapshot": None, "prompt_history": prompt_history,
        "sparql_history": sparql_history, "step_history": step_history,
        "method_events": method_events, "result": result,
        "capture_status": "NO_DEPTH3_DECISION",
    }


def relation_chain_from_snapshot(snapshot: Mapping[str, Any]) -> list[Relation]:
    relations = [item for item in snapshot["current_path"] if item.get("kind") == "relation"]
    if len(relations) != MAX_DEPTH:
        raise PilotStop("captured current path does not contain exactly three relation edges")
    return [Relation(id=str(item["id"]), left=item["direction"] == "OUTGOING") for item in relations]


def query_projection(agent: CoRAgent, meter: HardCapMeter, approval: Mapping[str, Any],
                     query: str, variables: list[str], recorder: DurableRecorder) -> dict[str, Any]:
    sidecar = binding_projection_query(query, variables)
    meta = meter.call_sparql_once(
        approval=approval, query_text=sidecar,
        one_attempt_transport=recorder.query_transport(recorder.sparql_transport),
    )
    # Reuse parser normalization without dispatching another query.
    from replay import fetch_binding_projection
    cached_meta = {"status": "ok", "rows": meta["rows"]}
    projected = fetch_binding_projection(query, variables, lambda _: cached_meta)
    if projected["query_hash"] != hashlib.sha256(sidecar.encode("utf-8")).hexdigest():
        raise PilotStop("sidecar projection query hash mismatch")
    return projected


def build_agent(model_name: str) -> CoRAgent:
    return CoRAgent(
        model_name=model_name, relation_width=RELATION_WIDTH, depth=MAX_DEPTH,
        sample_relation_threshold=SAMPLE_RELATION_THRESHOLD, remove_unnecessary_rel=True,
        temperature_exploration=TEMPERATURE_EXPLORATION,
        temperature_reasoning=TEMPERATURE_REASONING, max_token=MAX_COMPLETION_TOKENS,
        backend=get_backend("freebase"),
    )


def _case_result(case_index: int, case: Mapping[str, Any], status: str, reason: str,
                 meter: HardCapMeter, *, snapshot: Mapping[str, Any] | None = None,
                 pair: Mapping[str, Any] | None = None, selected: Mapping[str, Any] | None = None,
                 error: str | None = None, new_gold: list[str] | None = None,
                 query_start: int = 0, api_start: int = 0, cost_start: float = 0.0) -> dict[str, Any]:
    stop = (pair or {}).get("STOP_AT_3", {})
    hop = (pair or {}).get("CONTINUE_ONE_HOP", {})
    treatment = hop.get("treatment", {}) if hop else {}
    return {
        "case_index": case_index, "question_id": case["question_id"], "repeat": case["repeat"],
        "role": case["role"], "status": status, "reason": reason,
        "boundary_action": (snapshot or {}).get("boundary_action"),
        "boundary_depth": (snapshot or {}).get("boundary_depth"),
        "prefix_hash": (snapshot or {}).get("prefix_hash"),
        "prefix_complete": bool((snapshot or {}).get("prefix_replay_exact", False)),
        "selected_relation_id": (selected or {}).get("id"),
        "relation_direction": (selected or {}).get("direction"),
        "candidate_ids_before_prune": treatment.get("candidate_ids_before_prune"),
        "candidate_ids_after_prune": treatment.get("candidate_ids_after_prune"),
        "stop_binding_id_count": len(stop.get("active_binding_ids", [])),
        "one_hop_entity_ids": hop.get("KG_result_ids", []),
        "new_entity_ids": sorted(set(hop.get("KG_result_ids", [])) - set(stop.get("active_binding_ids", []))),
        "new_gold_ids": new_gold or [],
        "query_attempts": meter.live_sparql_attempts - query_start,
        "api_attempts": meter.live_api_attempts - api_start,
        "estimated_cost_usd": round(meter.live_cost_usd - cost_start, 12),
        "error": error,
    }


def run_case(case_index: int, case: Mapping[str, Any], row: Mapping[str, Any], gold_ids: list[str],
             model_settings: dict[str, Any], approval: dict[str, Any], label_cache: dict[str, str],
             meter: HardCapMeter, recorder: DurableRecorder, endpoint: str) -> dict[str, Any]:
    recorder.current_case_id = str(case["question_id"])
    query_start, api_start = meter.live_sparql_attempts, meter.live_api_attempts
    cost_start = meter.live_cost_usd
    topics = [Entity(str(i), str(n)) for i, n in zip(row["topic_entities"], row["topic_entity_names"])]
    agent = build_agent(model_settings["model"])
    recorder.sparql_transport = sparqlwrapper_one_attempt(
        endpoint, approval=approval, timeout_seconds=SPARQL_TIMEOUT_SECONDS
    )
    with install_metered_backend(
        agent.kg_backend, meter, approval=approval, label_cache=label_cache,
        one_attempt_transport=recorder.query_transport(recorder.sparql_transport),
    ), install_metered_llm(
        agent, meter, approval=approval, token_counter=conservative_token_reservation,
    ):
        try:
            captured = capture_first_depth_decision(
                agent, row, topics, case,
                {**model_settings, "llm_timeout_seconds": OPENAI_TIMEOUT_SECONDS,
                 "sparql_timeout_seconds": SPARQL_TIMEOUT_SECONDS, "retries": 0},
            )
        except LabelCacheMiss as exc:
            result = _case_result(case_index, case, "INELIGIBLE", "LABEL_CACHE_MISS", meter,
                                  error=str(exc), query_start=query_start, api_start=api_start, cost_start=cost_start)
            recorder.write_trace({"case": dict(case), "status": "INELIGIBLE", "reason": "LABEL_CACHE_MISS",
                                  "missing_ids": exc.missing_ids, "query_attempts": result["query_attempts"],
                                  "api_attempts": result["api_attempts"]})
            recorder.write_case(result)
            return result
        except PreBoundaryAnswerStage as exc:
            result = _case_result(case_index, case, "INELIGIBLE", "PRE_BOUNDARY_ANSWER_STAGE", meter,
                                  error=exc.stage, query_start=query_start, api_start=api_start, cost_start=cost_start)
            recorder.write_trace({"case": dict(case), "status": "INELIGIBLE", "reason": result["reason"],
                                  "prompt_history": captured_history_if_any(locals()),
                                  "sparql_history": captured_sparql_if_any(locals())})
            recorder.write_case(result)
            return result
        except PilotStop as exc:
            result = _case_result(case_index, case, "STOPPED", "PILOT_STOP", meter,
                                  error=str(exc), query_start=query_start, api_start=api_start, cost_start=cost_start)
            recorder.write_trace({"case": dict(case), "status": "STOPPED", "reason": "PILOT_STOP",
                                  "error": str(exc), "query_attempts": result["query_attempts"],
                                  "api_attempts": result["api_attempts"]})
            recorder.write_case(result)
            raise

        if captured["capture_status"] == "LABEL_CACHE_MISS":
            result = _case_result(case_index, case, "INELIGIBLE", "LABEL_CACHE_MISS", meter,
                                  error=captured.get("error"), query_start=query_start,
                                  api_start=api_start, cost_start=cost_start)
            recorder.write_trace({"case": dict(case), "status": result["status"], "reason": result["reason"],
                                  "missing_ids": captured.get("missing_ids", []),
                                  "prompt_history": captured["prompt_history"],
                                  "sparql_history": captured["sparql_history"],
                                  "method_events": captured["method_events"]})
            recorder.write_case(result)
            return result
        if captured["capture_status"] == "PRE_BOUNDARY_ANSWER_STAGE":
            result = _case_result(case_index, case, "INELIGIBLE", "PRE_BOUNDARY_ANSWER_STAGE", meter,
                                  error=captured.get("error"), query_start=query_start,
                                  api_start=api_start, cost_start=cost_start)
            recorder.write_trace({"case": dict(case), "status": result["status"], "reason": result["reason"],
                                  "blocked_stage": captured.get("error"),
                                  "prompt_history": captured["prompt_history"],
                                  "sparql_history": captured["sparql_history"],
                                  "method_events": captured["method_events"]})
            recorder.write_case(result)
            return result
        if captured["capture_status"] == "PILOT_STOP":
            result = _case_result(case_index, case, "STOPPED", "LIVE_ATTEMPT_OR_CAP_FAILURE", meter,
                                  error=captured.get("error"), query_start=query_start,
                                  api_start=api_start, cost_start=cost_start)
            recorder.write_trace({"case": dict(case), "status": result["status"], "reason": result["reason"],
                                  "error": captured.get("error"),
                                  "prompt_history": captured["prompt_history"],
                                  "sparql_history": captured["sparql_history"],
                                  "method_events": captured["method_events"]})
            recorder.write_case(result)
            raise PilotStop(captured.get("error", "live request stopped"))

        snapshot = captured.get("snapshot")
        if snapshot is None:
            result = _case_result(case_index, case, "INELIGIBLE", "NO_DEPTH3_DECISION", meter,
                                  query_start=query_start, api_start=api_start, cost_start=cost_start)
            recorder.write_trace({"case": dict(case), "status": result["status"], "reason": result["reason"],
                                  "prompt_history": captured["prompt_history"],
                                  "sparql_history": captured["sparql_history"],
                                  "method_events": captured["method_events"]})
            recorder.write_case(result)
            return result
        if snapshot.get("boundary_action") != "Forward":
            result = _case_result(case_index, case, "INELIGIBLE", "NON_FORWARD_BOUNDARY", meter,
                                  snapshot=snapshot, query_start=query_start, api_start=api_start, cost_start=cost_start)
            recorder.write_trace({"case": dict(case), "status": result["status"], "reason": result["reason"],
                                  "snapshot": snapshot, "prompt_history": captured["prompt_history"],
                                  "sparql_history": captured["sparql_history"],
                                  "method_events": captured["method_events"]})
            recorder.write_case(result)
            return result
        try:
            chain = relation_chain_from_snapshot(snapshot)
            last_entity = next((e for e in reversed(snapshot["sparql_history"])
                                if e.get("search_type") == "entity_search"), None)
            if not last_entity or not last_entity.get("sparql"):
                raise CaseIneligible("MISSING_DEPTH3_ENTITY_QUERY")
            prefix_variables = [f"?e_{i}" for i in range(1, MAX_DEPTH)] + ["?targetEntity"]
            prefix_projection = query_projection(
                agent, meter, approval, last_entity["sparql"], prefix_variables, recorder,
            )
            try:
                sealed_prefix = attach_binding_projection(snapshot, prefix_projection, prefix_variables)
            except (KeyError, RuntimeError, ValueError) as exc:
                raise CaseIneligible(f"INCOMPLETE_BINDING_PROJECTION: {exc}") from exc
            if not is_complete_prefix(sealed_prefix) or not sealed_prefix.get("prefix_replay_exact"):
                raise CaseIneligible("PREFIX_INCOMPLETE_AFTER_BINDING_PROJECTION")
            post_events: list[dict[str, Any]] = []
            post_prompt_start = len(captured["prompt_history"])
            post_sparql_start = len(captured["sparql_history"])
            with capture_cor_method_outputs(agent, post_events):
                head_relations, tail_relations = agent.relation_search(
                    topic_entity=topics[0], relation_chain=chain, sparql_history=captured["sparql_history"],
                )
                before_all = list(dict.fromkeys([*head_relations, *tail_relations]))
                if not before_all:
                    raise CaseIneligible("NO_POST_BOUNDARY_RELATION_CANDIDATES")
                selected_relations = agent.relation_prune(
                    question=str(row["question"]), topic_entity=topics[0], relation_chain=chain,
                    head_relations=head_relations, tail_relations=tail_relations,
                    prompt_history=captured["prompt_history"],
                )
            if not selected_relations:
                raise PilotStop("post-boundary relation pruning returned no selected relation")
            selected_relations = selected_relations[:agent.relation_width]
            selected_relations = agent._expand_bidirectional_relations(
                selected_relations=selected_relations,
                head_relations=head_relations,
                tail_relations=tail_relations,
            )
            chosen = selected_relations[0]
            direction = "OUTGOING" if bool(chosen.left) else "INCOMING"
            valid_by_direction = set(head_relations if direction == "OUTGOING" else tail_relations)
            if str(chosen.id) not in valid_by_direction:
                raise PilotStop("selected relation direction is absent from its candidate direction set")
            after_all = list(dict.fromkeys(str(rel.id) for rel in selected_relations))
            if not set(after_all).issubset(before_all):
                raise PilotStop("post-prune relation IDs are not a subset of pre-prune candidate IDs")
            extended_chain = chain + [chosen]
            extract, err = _build_entities_extract(topics[0], extended_chain, agent.kg_backend)
            if err:
                raise PilotStop(f"cannot build one-hop entity query: {err}")
            edge_query = agent.kg_backend.build_sparql_entities(extract)
            edge_variables = [f"?e_{i}" for i in range(1, MAX_DEPTH)] + ["?targetEntity"]
            edge_meta = meter.call_sparql_once(
                approval=approval,
                query_text=binding_projection_query(edge_query, edge_variables),
                one_attempt_transport=recorder.query_transport(recorder.sparql_transport),
            )
            edge_projection = fetch_binding_projection(
                edge_query, edge_variables, lambda _: {"status": "ok", "rows": edge_meta["rows"]},
            )
            if len(edge_projection["rows"]) >= 1000:
                raise PilotStop("one-hop edge projection reached LIMIT; completeness is unknown")
            extension_rows = [{
                "?e1": row["?e_1"], "?e2": row["?e_2"], "?e3": row["?e_3"], "?e4": row["?targetEntity"],
            } for row in edge_projection["rows"]]
            pair = build_pair(
                sealed_prefix, selected_relation_id=str(chosen.id), relation_direction=direction,
                candidate_relation_ids_before_prune=before_all,
                candidate_relation_ids_after_prune=after_all,
                kg_query=edge_projection["query"], kg_rows=extension_rows, kg_status="ok",
            )
            stop_ids = set(pair["STOP_AT_3"]["active_binding_ids"])
            new_ids = sorted(set(pair["CONTINUE_ONE_HOP"]["KG_result_ids"]) - stop_ids)
            new_gold_ids = sorted(set(new_ids).intersection(gold_ids))
            # Scoring-only labels are checked after the inference and graph traversal have ended.
            prompts = [str(item.get("prompt", "")) for item in captured["prompt_history"] if item.get("prompt")]
            leaked = sorted(gold_id for gold_id in gold_ids if any(gold_id in prompt for prompt in prompts))
            status = "VALID_NEW_GOLD_IDS" if new_gold_ids else ("VALID_NEW_IDS_NO_GOLD" if new_ids else "VALID_NO_NEW_IDS")
            result = _case_result(case_index, case, status, "EXACT_PREFIX_ONE_HOP_COMPLETE", meter,
                                  snapshot=sealed_prefix, pair=pair,
                                  selected={"id": str(chosen.id), "direction": direction},
                                  new_gold=new_gold_ids, query_start=query_start, api_start=api_start,
                                  cost_start=cost_start)
            trace = {
                "case": dict(case), "status": status, "reason": result["reason"],
                "snapshot": sealed_prefix,
                "binding_projection": prefix_projection,
                "post_boundary_relation_search": {
                    "head_relation_ids": list(head_relations), "tail_relation_ids": list(tail_relations),
                    "pre_prune_relation_ids": before_all,
                    "post_prune_relations": [{"id": str(x.id), "direction": "OUTGOING" if x.left else "INCOMING"}
                                              for x in selected_relations],
                },
                "one_hop_projection": edge_projection,
                "pair": pair,
                "gold_scoring_only": {"answer_ids": list(gold_ids), "new_gold_ids": new_gold_ids,
                                      "prompt_id_leak_check": leaked},
                "prompt_history": captured["prompt_history"],
                "sparql_history_before_boundary": captured["sparql_history"],
                "method_events_before_boundary": captured["method_events"],
                "post_boundary_prompt_history": captured["prompt_history"][post_prompt_start:],
                "post_boundary_sparql_events": captured["sparql_history"][post_sparql_start:],
                "post_boundary_method_events": post_events,
            }
            if leaked:
                result["status"] = "INTEGRITY_BLOCKED"
                result["reason"] = "GOLD_ID_APPEARED_IN_INFERENCE_PROMPT"
                trace["status"] = result["status"]
                trace["reason"] = result["reason"]
            recorder.write_trace(trace)
            recorder.write_case(result)
            if leaked:
                raise PilotStop("gold ID appeared in an inference prompt; stop without further calls")
            return result
        except CaseIneligible as exc:
            result = _case_result(case_index, case, "INELIGIBLE", str(exc).split(":", 1)[0], meter,
                                  snapshot=snapshot, error=str(exc), query_start=query_start,
                                  api_start=api_start, cost_start=cost_start)
            recorder.write_trace({"case": dict(case), "status": result["status"],
                                  "reason": result["reason"], "error": result["error"],
                                  "snapshot": snapshot,
                                  "binding_projection": locals().get("prefix_projection"),
                                  "prompt_history": captured["prompt_history"],
                                  "sparql_history": captured["sparql_history"],
                                  "method_events": captured["method_events"]})
            recorder.write_case(result)
            return result
        except LabelCacheMiss as exc:
            result = _case_result(case_index, case, "INELIGIBLE", "LABEL_CACHE_MISS", meter,
                                  snapshot=snapshot, error=str(exc), query_start=query_start,
                                  api_start=api_start, cost_start=cost_start)
            recorder.write_trace({"case": dict(case), "status": result["status"], "reason": result["reason"],
                                  "missing_ids": exc.missing_ids, "snapshot": snapshot,
                                  "prompt_history": captured["prompt_history"],
                                  "sparql_history": captured["sparql_history"],
                                  "method_events": captured["method_events"]})
            recorder.write_case(result)
            return result
        except (PilotStop, ValueError, RuntimeError) as exc:
            result = _case_result(case_index, case, "STOPPED", "PILOT_INVARIANT_OR_QUERY_FAILURE", meter,
                                  snapshot=snapshot, error=f"{type(exc).__name__}: {exc}",
                                  query_start=query_start, api_start=api_start, cost_start=cost_start)
            recorder.write_trace({"case": dict(case), "status": result["status"], "reason": result["reason"],
                                  "error": result["error"], "snapshot": snapshot,
                                  "prompt_history": captured["prompt_history"],
                                  "sparql_history": captured["sparql_history"],
                                  "method_events": captured["method_events"]})
            recorder.write_case(result)
            raise


def captured_history_if_any(local_values: Mapping[str, Any]) -> list[Any]:
    captured = local_values.get("captured")
    return captured.get("prompt_history", []) if isinstance(captured, dict) else []


def captured_sparql_if_any(local_values: Mapping[str, Any]) -> list[Any]:
    captured = local_values.get("captured")
    return captured.get("sparql_history", []) if isinstance(captured, dict) else []


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] != "--run":
        raise SystemExit("usage: python run_live_pilot.py --run")
    safe_load_project_env()
    os.environ["OPENAI_MAX_RETRIES"] = "1"  # LLMAPI counts this as one total attempt; SDK retries remain zero.
    os.environ["OPENAI_RETRY_INTERVAL"] = "0"
    os.environ["OPENAI_TIMEOUT"] = str(OPENAI_TIMEOUT_SECONDS)
    os.environ["SPARQL_TIMEOUT"] = str(SPARQL_TIMEOUT_SECONDS)
    os.environ.setdefault("OPENAI_REASONING_EFFORT", "low")
    if os.environ.get("MODEL_NAME") != MODEL_EXPECTED:
        raise SystemExit(f"frozen model mismatch; expected {MODEL_EXPECTED}")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is unavailable after safe .env parsing")
    endpoint = os.environ.get("FREEBASE_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql")
    if endpoint != "http://127.0.0.1:8890/sparql":
        raise SystemExit("Iteration 22 requires the configured local forwarded Freebase endpoint")

    try:
        import tiktoken
        tiktoken.get_encoding("o200k_base")
        conservative_token_reservation("Iteration 22 tokenizer preflight")
        from SPARQLWrapper import SPARQLWrapper as _SPARQLWrapper  # noqa: F401
    except Exception as exc:
        raise SystemExit(f"offline runtime dependency preflight failed: {type(exc).__name__}: {exc}") from exc
    probe_agent = build_agent(MODEL_EXPECTED)
    probe_api = probe_agent.llm_api
    if (getattr(probe_api, "max_retries", None) != 1
            or getattr(probe_api, "retry_interval", None) != 0
            or getattr(getattr(probe_api, "client", None), "max_retries", None) != 0):
        raise SystemExit("offline runtime preflight could not verify one LLM attempt and zero SDK retries")
    del probe_agent

    manifest, approval, label_cache = verify_preflight()
    rows = load_case_rows(manifest)
    source = source_hashes()
    model_settings = {
        "model": MODEL_EXPECTED, "temperature": TEMPERATURE_REASONING,
        "relation_width": RELATION_WIDTH,
        "sample_relation_threshold": SAMPLE_RELATION_THRESHOLD,
        "remove_unnecessary_rel": True, "temperature_exploration": TEMPERATURE_EXPLORATION,
        "temperature_reasoning": TEMPERATURE_REASONING,
        "max_completion_tokens": MAX_COMPLETION_TOKENS, "reasoning_effort": "low",
    }
    recorder = DurableRecorder(approval, manifest, source)
    meter = HardCapMeter(event_sink=recorder.meter_event)
    recorder.sparql_transport = sparqlwrapper_one_attempt(endpoint, approval=approval,
                                                           timeout_seconds=SPARQL_TIMEOUT_SECONDS)

    # Keep gold answer IDs isolated from the inference argument objects.
    scoring_only = {str(row["id"]): [str(x) for x in row.get("answer_ids", []) if str(x).strip()]
                    for row in rows}
    atomic_json(GOLD_PATH, {
        "evidence_class": "SCORING_ONLY_NEVER_PASSED_TO_INFERENCE",
        "manifest_sha256": MANIFEST_SHA256,
        "cases": scoring_only,
    })
    outcomes = []
    try:
        for i, (case, row) in enumerate(zip(manifest["cases"], rows), start=1):
            outcome = run_case(i, case, row, scoring_only[str(row["id"])], model_settings,
                               approval, label_cache, meter, recorder, endpoint)
            outcomes.append(outcome)
        overall = "PILOT_COMPLETE" if len(outcomes) == len(manifest["cases"]) else "PILOT_PARTIAL"
    except BaseException as exc:
        recorder.state["status"] = "PILOT_STOPPED"
        recorder.state["stop_reason"] = f"{type(exc).__name__}: {exc}"
        recorder.state["finished_at_utc"] = utc_now()
        atomic_json(RUN_STATE_PATH, recorder.state)
        overall = "PILOT_PARTIAL"
        outcomes = []
        if CASE_CSV_PATH.exists():
            with CASE_CSV_PATH.open(newline="", encoding="utf-8") as handle:
                outcomes = list(csv.DictReader(handle))

    estimated_cap = MAX_API_REQUESTS * estimated_cost(MAX_INPUT_TOKENS, MAX_COMPLETION_TOKENS)
    if estimated_cap > MAX_API_COST_USD:
        raise AssertionError("authorized all-request reservation unexpectedly exceeds the approved dollar cap")
    results = {
        "iteration": 22, "status": overall, "evidence_class": "LIVE_RETRIEVAL_ONLY_PILOT",
        "manifest_sha256": MANIFEST_SHA256, "case_results": outcomes,
        "manifest_case_count": len(manifest["cases"]),
        "eligible_cases": sum(isinstance(x, dict) and str(x.get("status", "")).startswith("VALID_") for x in outcomes),
        "valid_matched_prefix_cases": sum(isinstance(x, dict) and str(x.get("status", "")).startswith("VALID_") for x in outcomes),
        "api_requests_attempted": meter.live_api_attempts,
        "sparql_queries_attempted": meter.live_sparql_attempts,
        "estimated_api_cost_usd": round(meter.live_cost_usd, 12),
        "uncertain_reserved_cost_usd": round(meter.uncertain_live_cost_usd, 12),
        "authorized_caps": {
            "api_requests": MAX_API_REQUESTS, "input_tokens_per_request": MAX_INPUT_TOKENS,
            "completion_tokens_per_request": MAX_COMPLETION_TOKENS,
            "estimated_api_spend_usd": MAX_API_COST_USD,
            "read_only_sparql_queries": MAX_SPARQL_QUERIES, "retries": 0,
        },
        "all_request_cap_reservation_usd": round(estimated_cap, 12),
        "no_answer_filter_or_answer_generation_after_boundary": True,
        "finished_at_utc": utc_now(),
    }
    atomic_json(RESULTS_PATH, results)
    recorder.state["status"] = overall
    recorder.state["finished_at_utc"] = results["finished_at_utc"]
    atomic_json(RUN_STATE_PATH, recorder.state)
    print(json.dumps({k: results[k] for k in (
        "status", "manifest_case_count", "eligible_cases", "api_requests_attempted",
        "sparql_queries_attempted", "estimated_api_cost_usd", "uncertain_reserved_cost_usd",
    )}, indent=2))


if __name__ == "__main__":
    main()
