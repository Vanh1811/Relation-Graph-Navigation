"""Offline dry-run and hard-cap guards for the Iteration 22 retrieval pilot.

No transport is created here. Live calls require an exact, persisted approval
record plus caller-supplied single-attempt transports. All fixture outputs are
explicitly marked synthetic and cannot be mistaken for measured KG evidence.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

ITER21_DIR = Path(__file__).resolve().parents[1] / "iter_021_matched_prefix"
if str(ITER21_DIR) not in sys.path:
    sys.path.insert(0, str(ITER21_DIR))

from replay import attach_binding_projection, build_pair, is_complete_prefix, sha256_json


MANIFEST_SHA256 = "f8a0a15de59e7a4cd3a20d351ad549e41d4135ad114f2d8527bbda9e0ce953d9"
MAX_API_REQUESTS = 60
MAX_INPUT_TOKENS = 7000
MAX_COMPLETION_TOKENS = 1024
MAX_API_COST_USD = 0.17
MAX_SPARQL_QUERIES = 100
INPUT_USD_PER_MILLION = 0.20
OUTPUT_USD_PER_MILLION = 1.20


class PilotStop(RuntimeError):
    """A hard cap, missing authorization, or invalid live response stopped work."""


class LabelCacheMiss(BaseException):
    """Fatal cache-only stop that bypasses entity_search's broad Exception handler."""

    def __init__(self, missing_ids: Sequence[str]):
        self.missing_ids = sorted(set(str(value) for value in missing_ids))
        super().__init__(f"frozen local label cache misses: {self.missing_ids}")


def approval_matches(record: Mapping[str, Any] | None) -> bool:
    """Accept only a recorded approval for the frozen manifest and exact caps."""
    if not isinstance(record, Mapping):
        return False
    approval_text = str(record.get("approval_verbatim", "")).lower()
    if not approval_text.strip():
        return False
    if not str(record.get("approved_at", "")).strip():
        return False
    expected = {
        "manifest_sha256": MANIFEST_SHA256,
        "api_request_cap": MAX_API_REQUESTS,
        "input_tokens_per_request_cap": MAX_INPUT_TOKENS,
        "completion_tokens_per_request_cap": MAX_COMPLETION_TOKENS,
        "estimated_api_spend_cap_usd": MAX_API_COST_USD,
        "read_only_sparql_query_cap": MAX_SPARQL_QUERIES,
        "retries": 0,
    }
    if not all(record.get(key) == value for key, value in expected.items()):
        return False
    normalized_text = re.sub(r"(?<=\d),(?=\d)", "", approval_text)
    required_text = (
        MANIFEST_SHA256,
        "three fixed questions",
        "repeat 1 only",
        "60",
        "7000",
        "1024",
        "0.17",
        "100",
        "no retries",
    )
    return all(value in normalized_text for value in required_text)


def require_read_only_select(query_text: str) -> None:
    """Reject non-SELECT text before it consumes a live SPARQL attempt."""
    if not isinstance(query_text, str) or not query_text.strip():
        raise PilotStop("SPARQL query must be a nonempty read-only SELECT")
    query = re.sub(r"(?m)^\s*#.*$", "", query_text.lstrip("\ufeff"))
    prefix = r"(?:PREFIX\s+[A-Za-z_][\w.-]*:\s*<[^>]*>|BASE\s*<[^>]*>)"
    while True:
        updated = re.sub(rf"(?is)^\s*{prefix}\s*", "", query, count=1)
        if updated == query:
            break
        query = updated
    if re.match(r"(?is)^\s*SELECT\b", query) is None:
        raise PilotStop("only read-only SPARQL SELECT queries are allowed")


def sparqlwrapper_one_attempt(endpoint: str, *, approval: Mapping[str, Any] | None,
                              timeout_seconds: int = 120) -> Callable[[str], Mapping[str, Any]]:
    """Create an explicit SPARQLWrapper transport with one HTTP attempt per call.

    This only constructs a callable; it performs no network I/O until invoked by
    ``HardCapMeter.call_sparql_once`` after approval and cap checks.
    """
    if not approval_matches(approval):
        raise PilotStop("explicit Iteration 22 approval record is absent or does not match the frozen caps")
    if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
        raise ValueError("SPARQL endpoint must be an explicit http(s) URL")
    if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
        raise ValueError("timeout_seconds must be a positive integer")

    def one_attempt(query_text: str) -> Mapping[str, Any]:
        require_read_only_select(query_text)
        from SPARQLWrapper import JSON, SPARQLWrapper

        client = SPARQLWrapper(endpoint)
        client.setQuery(query_text)
        client.setReturnFormat(JSON)
        client.setTimeout(timeout_seconds)
        response = client.query().convert()
        bindings = response.get("results", {}).get("bindings", [])
        if not isinstance(bindings, list):
            return {"status": "malformed_response", "rows": []}
        return {"status": "ok", "rows": bindings}

    return one_attempt


def estimated_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * INPUT_USD_PER_MILLION / 1_000_000 + output_tokens * OUTPUT_USD_PER_MILLION / 1_000_000


class HardCapMeter:
    """Track simulated activity separately from authorized live attempts."""

    def __init__(self, event_sink: Callable[[Mapping[str, Any]], None] | None = None) -> None:
        self.live_api_attempts = 0
        self.live_sparql_attempts = 0
        self.live_cost_usd = 0.0
        self.uncertain_live_cost_usd = 0.0
        self.simulated_api_attempts = 0
        self.simulated_sparql_attempts = 0
        self.simulated_worst_case_reserved_usd = 0.0
        self.events: list[dict[str, Any]] = []
        self.event_sink = event_sink

    def _emit(self, event: Mapping[str, Any]) -> None:
        item = dict(event)
        self.events.append(item)
        if self.event_sink is not None:
            self.event_sink(item)

    def _check_llm_reservation(self, input_tokens: int, completion_tokens: int, *, live: bool) -> float:
        if not isinstance(input_tokens, int) or input_tokens < 0 or input_tokens > MAX_INPUT_TOKENS:
            raise PilotStop(f"input reservation must be an integer from 0 to {MAX_INPUT_TOKENS}")
        if not isinstance(completion_tokens, int) or completion_tokens < 1 or completion_tokens > MAX_COMPLETION_TOKENS:
            raise PilotStop(f"completion cap must be an integer from 1 to {MAX_COMPLETION_TOKENS}")
        if live and self.live_api_attempts >= MAX_API_REQUESTS:
            raise PilotStop("refusing API request above the 60-request cap")
        if not live and self.simulated_api_attempts >= MAX_API_REQUESTS:
            raise PilotStop("dry-run reached the 60-request cap; no request was sent")
        reserve = estimated_cost(input_tokens, completion_tokens)
        spent = self.live_cost_usd + self.uncertain_live_cost_usd if live else self.simulated_worst_case_reserved_usd
        if spent + reserve > MAX_API_COST_USD + 1e-12:
            raise PilotStop("refusing request because its worst-case reservation exceeds the remaining $0.17 cap")
        return reserve

    def simulate_api_attempt(self, *, input_tokens: int, completion_tokens: int,
                             content: str | None, parsed_selection: Sequence[Mapping[str, str]] | None = None) -> str:
        """Account a scripted fixture attempt; never accepts or invokes a transport."""
        reserve = self._check_llm_reservation(input_tokens, completion_tokens, live=False)
        self.simulated_api_attempts += 1
        self.simulated_worst_case_reserved_usd += reserve
        if not isinstance(content, str) or not content.strip():
            status = "EMPTY_RESPONSE"
        elif parsed_selection is None:
            status = "MALFORMED_OR_UNPARSED_RESPONSE"
        elif not parsed_selection:
            status = "EMPTY_SELECTION"
        else:
            status = "SCRIPTED_RESPONSE_PRESENT"
        self._emit({"kind": "SIMULATED_LLM", "status": status,
                    "request_index": self.simulated_api_attempts,
                    "reserved_usd": reserve, "transport_invoked": False})
        return status

    def simulate_sparql_attempt(self, query_text: str, *, status: str = "ok") -> str:
        """Account one scripted query attempt without a network-capable callback."""
        if self.simulated_sparql_attempts >= MAX_SPARQL_QUERIES:
            raise PilotStop("refusing SPARQL query above the 100-query cap")
        self.simulated_sparql_attempts += 1
        event = {"kind": "SIMULATED_SPARQL", "query_index": self.simulated_sparql_attempts,
                 "query_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
                 "status": status, "transport_invoked": False}
        self._emit(event)
        if status != "ok":
            raise PilotStop(f"simulated SPARQL query failed with status={status}; no retry")
        return status

    def call_llm_once(self, *, approval: Mapping[str, Any] | None, input_tokens: int,
                      completion_tokens: int, transport: Callable[[], tuple[str | None, Mapping[str, Any]]],
                      prompt_sha256: str | None = None) -> tuple[str, Mapping[str, Any]]:
        """Run one approved model attempt; account errors and never retry."""
        if not approval_matches(approval):
            raise PilotStop("explicit Iteration 22 approval record is absent or does not match the frozen caps")
        reserve = self._check_llm_reservation(input_tokens, completion_tokens, live=True)
        self.live_api_attempts += 1  # Attempt is consumed before transport, including failures.
        self._emit({"kind": "LLM", "status": "ATTEMPT_STARTED", "request_index": self.live_api_attempts,
                    "input_token_reservation": input_tokens, "completion_token_cap": completion_tokens,
                    "reserved_usd": reserve, "prompt_sha256": prompt_sha256, "retries": 0})
        try:
            content, usage = transport()
        except BaseException as exc:
            self.uncertain_live_cost_usd += reserve
            self._emit({"kind": "LLM", "status": "TRANSPORT_ERROR", "request_index": self.live_api_attempts,
                        "reserved_usd": reserve, "error": str(exc), "retries": 0})
            raise PilotStop("single model attempt failed; its full reservation remains charged and no retry is allowed") from exc
        if not isinstance(usage, Mapping):
            self.uncertain_live_cost_usd += reserve
            self._emit({"kind": "LLM", "status": "UNMETERED", "request_index": self.live_api_attempts,
                        "reserved_usd": reserve, "retries": 0})
            raise PilotStop("model response has no usage meter; stop without retry")
        if usage.get("error"):
            self.uncertain_live_cost_usd += reserve
            self._emit({"kind": "LLM", "status": "API_ERROR", "request_index": self.live_api_attempts,
                        "reserved_usd": reserve, "error": str(usage["error"]), "retries": 0})
            raise PilotStop("model API error was metered at its full reservation; stop without retry")
        used_input = usage.get("input_tokens")
        used_output = usage.get("output_tokens")
        if not isinstance(used_input, int) or not isinstance(used_output, int):
            self.uncertain_live_cost_usd += reserve
            self._emit({"kind": "LLM", "status": "INVALID_USAGE", "request_index": self.live_api_attempts,
                        "reserved_usd": reserve, "retries": 0})
            raise PilotStop("model usage fields are missing or invalid; stop without retry")
        if used_input > input_tokens or used_output > completion_tokens:
            self.uncertain_live_cost_usd += reserve
            self._emit({"kind": "LLM", "status": "RESERVATION_EXCEEDED", "request_index": self.live_api_attempts,
                        "reserved_usd": reserve, "input_tokens": used_input, "output_tokens": used_output,
                        "retries": 0})
            raise PilotStop("observed usage exceeded its pre-request reservation")
        actual = estimated_cost(used_input, used_output)
        self.live_cost_usd += actual
        self._emit({"kind": "LLM", "status": "METERED", "request_index": self.live_api_attempts,
                    "input_tokens": used_input, "output_tokens": used_output,
                    "cost_usd": actual, "retries": 0})
        if not isinstance(content, str) or not content.strip():
            raise PilotStop("empty model response was metered; stop without retry")
        return content, usage

    def call_sparql_once(self, *, approval: Mapping[str, Any] | None, query_text: str,
                         one_attempt_transport: Callable[[str], Mapping[str, Any]]) -> Mapping[str, Any]:
        """Execute exactly one metered query; a failed query consumes its slot."""
        if not approval_matches(approval):
            raise PilotStop("explicit Iteration 22 approval record is absent or does not match the frozen caps")
        require_read_only_select(query_text)
        if self.live_sparql_attempts >= MAX_SPARQL_QUERIES:
            raise PilotStop("refusing SPARQL query above the 100-query cap")
        self.live_sparql_attempts += 1
        query_hash = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
        self._emit({"kind": "SPARQL", "status": "ATTEMPT_STARTED", "query_index": self.live_sparql_attempts,
                    "query_sha256": query_hash, "retries": 0})
        try:
            result = one_attempt_transport(query_text)
        except BaseException as exc:
            self._emit({"kind": "SPARQL", "status": "TRANSPORT_ERROR",
                        "query_index": self.live_sparql_attempts, "query_sha256": query_hash,
                        "retries": 0, "error": str(exc)})
            raise PilotStop("single SPARQL attempt failed; stop without retry") from exc
        if not isinstance(result, Mapping) or result.get("status") != "ok":
            self._emit({"kind": "SPARQL", "status": "QUERY_ERROR",
                        "query_index": self.live_sparql_attempts, "query_sha256": query_hash,
                        "retries": 0})
            raise PilotStop("SPARQL status is not ok; stop without retry")
        self._emit({"kind": "SPARQL", "status": "ok", "query_index": self.live_sparql_attempts,
                    "query_sha256": query_hash, "result_rows": len(result.get("rows", [])), "retries": 0})
        return result


@contextmanager
def install_metered_backend(backend: Any, meter: HardCapMeter, *, approval: Mapping[str, Any] | None,
                            label_cache: Mapping[str, str],
                            one_attempt_transport: Callable[[str], Mapping[str, Any]]):
    """Temporarily meter both backend query methods and use frozen labels only.

    The supplied transport is called once per query. A label-cache miss raises
    and stops the case before FreebaseBackend's usual hidden label-query path.
    """
    names = ("execute_sparql_with_meta", "execute_sparql", "id2entity_names", "id2entity_name_or_type")
    instance_dict = getattr(backend, "__dict__", {})
    originals = {name: (getattr(backend, name), name in instance_dict, instance_dict.get(name))
                 for name in names if hasattr(backend, name)}

    def execute_meta(query: str):
        return meter.call_sparql_once(approval=approval, query_text=query,
                                      one_attempt_transport=one_attempt_transport)

    def execute_rows(query: str):
        return meter.call_sparql_once(approval=approval, query_text=query,
                                      one_attempt_transport=one_attempt_transport).get("rows", [])

    def labels(entity_ids: Sequence[str]):
        missing = [str(entity_id) for entity_id in entity_ids if str(entity_id) not in label_cache]
        if missing:
            # entity_search currently catches Exception around this lookup and
            # silently falls back to per-ID labels. BaseException is deliberate:
            # fail closed before its hidden, retrying SPARQL resolver can run.
            raise LabelCacheMiss(missing)
        return {str(entity_id): str(label_cache[str(entity_id)]) for entity_id in entity_ids}

    def label(entity_id: str):
        key = str(entity_id)
        if key not in label_cache:
            raise LabelCacheMiss([key])
        return str(label_cache[key])

    replacements = {
        "execute_sparql_with_meta": execute_meta,
        "execute_sparql": execute_rows,
        "id2entity_names": labels,
        "id2entity_name_or_type": label,
    }
    try:
        for name, replacement in replacements.items():
            if hasattr(backend, name):
                setattr(backend, name, replacement)
        yield backend
    finally:
        for name, (original, was_instance_attr, previous_value) in originals.items():
            if was_instance_attr:
                setattr(backend, name, previous_value)
            else:
                delattr(backend, name)


@contextmanager
def install_metered_llm(agent: Any, meter: HardCapMeter, *, approval: Mapping[str, Any] | None,
                        token_counter: Callable[[str], int]):
    """Wrap the agent's existing LLM client with a strict one-attempt meter.

    Construct the CoRAgent only after setting OPENAI_MAX_RETRIES=1 and
    OPENAI_RETRY_INTERVAL=0. The SDK-level retry count must already be zero.
    """
    api = getattr(agent, "llm_api", None)
    client = getattr(api, "client", None)
    if api is None or getattr(api, "max_retries", None) != 1 or getattr(api, "retry_interval", None) != 0:
        raise PilotStop("LLMAPI must be configured for one wrapper attempt and zero delay before the agent is created")
    if getattr(client, "max_retries", None) != 0:
        raise PilotStop("OpenAI SDK retries are not verifiably disabled")
    original = api.generate
    instance_dict = getattr(api, "__dict__", {})
    was_instance_attr = "generate" in instance_dict
    previous_value = instance_dict.get("generate")

    def metered_generate(user_prompt: str, temperature: float = 0.01, max_tokens: int = 256,
                         system_prompt: str | None = None):
        text = "\n".join(value for value in (system_prompt, user_prompt) if value is not None)
        input_reservation = token_counter(text)
        return meter.call_llm_once(
            approval=approval,
            input_tokens=input_reservation,
            completion_tokens=max_tokens,
            prompt_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            transport=lambda: original(user_prompt=user_prompt, temperature=temperature,
                                       max_tokens=max_tokens, system_prompt=system_prompt),
        )

    try:
        api.generate = metered_generate
        yield api
    finally:
        if was_instance_attr:
            api.generate = previous_value
        else:
            delattr(api, "generate")


def _selection_ids(parsed_selection: Sequence[Mapping[str, str]] | None,
                   before_prune: Sequence[str]) -> tuple[list[str], list[dict[str, str]]]:
    if not isinstance(parsed_selection, (list, tuple)) or not parsed_selection:
        raise PilotStop("relation-pruning response was malformed or selected no relation")
    allowed = set(before_prune)
    ids: list[str] = []
    normalized: list[dict[str, str]] = []
    for item in parsed_selection:
        if not isinstance(item, Mapping):
            raise PilotStop("relation-pruning parser returned a malformed selection")
        relation_id = item.get("id")
        direction = item.get("direction")
        if not isinstance(relation_id, str) or relation_id not in allowed:
            raise PilotStop("relation-pruning selected an ID absent from saved pre-prune candidates")
        if direction not in {"OUTGOING", "INCOMING"}:
            raise PilotStop("relation-pruning selection has no valid direction")
        ids.append(relation_id)
        normalized.append({"id": relation_id, "direction": direction})
    if len(ids) != len(set(ids)):
        raise PilotStop("relation-pruning returned duplicate relation IDs")
    return ids, normalized


def dry_run_case(*, snapshot: Mapping[str, Any], projection: Mapping[str, Any] | None,
                 projection_variable_order: Sequence[str], llm_response: str | None,
                 parsed_selection: Sequence[Mapping[str, str]] | None, kg_query: str,
                 kg_rows: Sequence[Mapping[str, str]], kg_status: str = "ok",
                 meter: HardCapMeter | None = None) -> dict[str, Any]:
    """Run a scripted case through the exact gates; no network/API is possible."""
    meter = meter or HardCapMeter()
    result: dict[str, Any] = {"evidence_class": "SYNTHETIC_DRY_RUN_ONLY", "status": "INELIGIBLE"}
    if str(snapshot.get("boundary_action", "")).lower() != "forward":
        result["reason"] = "NON_FORWARD_BOUNDARY"
        return result
    if not projection:
        result["reason"] = "MISSING_BINDING_PROJECTION"
        return result
    try:
        frozen = attach_binding_projection(snapshot, projection, projection_variable_order)
    except (TypeError, ValueError, KeyError) as exc:
        result["reason"] = "INCOMPLETE_OR_MISMATCHED_BINDINGS"
        result["detail"] = str(exc)
        return result
    if not is_complete_prefix(frozen) or not frozen.get("prefix_replay_exact"):
        result["reason"] = "PREFIX_INCOMPLETE"
        return result

    response_status = meter.simulate_api_attempt(input_tokens=256, completion_tokens=64,
                                                 content=llm_response, parsed_selection=parsed_selection)
    if response_status == "EMPTY_RESPONSE":
        result.update({"reason": "EMPTY_LLM_RESPONSE", "api_attempts_simulated": 1})
        return result
    if response_status in {"MALFORMED_OR_UNPARSED_RESPONSE", "EMPTY_SELECTION"}:
        result.update({"reason": response_status, "api_attempts_simulated": 1})
        return result
    try:
        selected_ids, normalized = _selection_ids(parsed_selection, frozen["candidate_relation_ids_before_prune"])
    except PilotStop as exc:
        result.update({"reason": "MALFORMED_LLM_SELECTION", "detail": str(exc), "api_attempts_simulated": 1})
        return result

    try:
        meter.simulate_sparql_attempt(kg_query, status=kg_status)
    except PilotStop as exc:
        result.update({"reason": "KG_QUERY_FAILURE_OR_CAP", "detail": str(exc), "api_attempts_simulated": 1})
        return result
    chosen = normalized[0]
    try:
        pair = build_pair(
            frozen,
            selected_relation_id=chosen["id"],
            relation_direction=chosen["direction"],
            candidate_relation_ids_before_prune=frozen["candidate_relation_ids_before_prune"],
            candidate_relation_ids_after_prune=selected_ids,
            kg_query=kg_query,
            kg_rows=kg_rows,
            kg_status=kg_status,
        )
    except (TypeError, ValueError, KeyError) as exc:
        result.update({"reason": "ONE_HOP_INVARIANT_FAILURE", "detail": str(exc), "api_attempts_simulated": 1})
        return result
    stop = pair["STOP_AT_3"]
    hop = pair["CONTINUE_ONE_HOP"]
    if stop["prefix_hash"] != hop["prefix_hash"] or hop["new_edge_count"] not in {0, 1}:
        result.update({"reason": "PAIR_INTEGRITY_FAILURE", "api_attempts_simulated": 1})
        return result
    result.update({
        "status": "SYNTHETIC_PAIR_VALID",
        "reason": None,
        "prefix_hash": stop["prefix_hash"],
        "stop_parent_hash": stop["prefix_hash"],
        "one_hop_parent_hash": hop["prefix_hash"],
        "new_edge_count": hop["new_edge_count"],
        "new_entity_ids": hop["KG_result_ids"],
        "api_attempts_simulated": 1,
        "sparql_attempts_simulated": 1,
    })
    return result


def summarize_dry_run(outcomes: Sequence[Mapping[str, Any]], meter: HardCapMeter) -> dict[str, Any]:
    eligible = sum(item.get("status") == "SYNTHETIC_PAIR_VALID" for item in outcomes)
    return {
        "evidence_class": "SYNTHETIC_DRY_RUN_ONLY",
        "status": "DRY_RUN_PASS" if eligible else "ALL_CASES_INELIGIBLE",
        "case_count": len(outcomes),
        "eligible_cases": eligible,
        "ineligible_reasons": [item.get("reason") for item in outcomes if item.get("status") != "SYNTHETIC_PAIR_VALID"],
        "actual_api_requests": meter.live_api_attempts,
        "actual_sparql_queries": meter.live_sparql_attempts,
        "simulated_api_attempts": meter.simulated_api_attempts,
        "simulated_sparql_attempts": meter.simulated_sparql_attempts,
        "transport_calls": 0,
    }
