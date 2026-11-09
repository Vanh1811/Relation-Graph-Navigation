"""Fail-closed COUNT adapter contract, isolated from production CoR/TGAR."""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import signal
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from chain_of_relations.methods.iter_030_operator_recovery.relation_aware_count import (
    RelationStep,
    RuntimeCountContext,
    _parse_sparql_json_count,
    build_count_query,
    classify_count_intent,
)

from .evidence import EvidenceRef
from .path_completeness import PathEvidence, PathStep, check_path_completeness
from .type_resolution import TargetTypeEvidence, TargetTypeResolution, resolve_target_type


CONTRACT_VERSION = "iter31-count-runtime-contract-v1"
MAX_PHYSICAL_SELECT_ATTEMPTS = 1
MAX_TIMEOUT_SECONDS = 120.0
_SELECT_WORD = re.compile(r"\bSELECT\b", re.IGNORECASE)
_FORBIDDEN_QUERY_WORD = re.compile(
    r"\b(?:SERVICE|INSERT|DELETE|CLEAR|DROP|LOAD|CREATE|MOVE|COPY|ADD|WITH)\b",
    re.IGNORECASE,
)
_COUNT_DISTINCT = re.compile(r"COUNT\s*\(\s*DISTINCT\s+\?[A-Za-z_][A-Za-z0-9_]*\s*\)", re.IGNORECASE)
_ENTITY_ID = re.compile(r"^(?:m|g)\.[A-Za-z0-9_]+$")
_INTEGER = re.compile(r"^(?:0|[1-9][0-9]*)$")
_XSD_INTEGER = "http://www.w3.org/2001/XMLSchema#integer"
_ACCEPTED_INTEGER_DATATYPES = {_XSD_INTEGER, "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"}


@dataclass(frozen=True)
class CountPlanContract:
    """The minimum complete, runtime-produced plan consumed by this adapter."""

    target_variable: str
    query_target_variable: str
    type_filter_variable: str
    expected_steps: tuple[PathStep, ...]
    executed_steps: tuple[PathStep, ...]
    query_status: str
    finalized: bool
    target_type_evidence: tuple[TargetTypeEvidence, ...]
    provenance: tuple[EvidenceRef, ...]
    plan_shape: str = "linear"
    operator: str = "COUNT_DISTINCT_ENTITY"
    semantics: str = "distinct_target_entities"
    required_external_binding_variables: tuple[str, ...] = ()
    active_binding_tuples: tuple[Mapping[str, str], ...] | None = None
    required_constraints_complete: bool | None = None
    all_pairwise_node_inequalities_represented: bool | None = None
    upstream_prompt_tokens: int | None = None
    upstream_prompt_limit: int | None = None


@dataclass(frozen=True)
class RuntimeCallsite:
    """A CoRAgent-shaped callsite plus an optional future typed COUNT plan.

    The current CoRAgent result object is retained by reference. If any gate
    fails, `output` is that exact object. `count_plan` is not currently emitted
    by CoRAgent or TGAR; its absence is a normal fail-closed outcome.
    """

    question: str
    topic_entity_id: str
    baseline_output: Any
    requested_answer_type: str = ""
    count_plan: CountPlanContract | None = None
    callsite_provenance: tuple[EvidenceRef, ...] = ()


@dataclass(frozen=True)
class AdapterOutcome:
    status: str
    output: Any
    reason: str
    query: str = ""
    query_sha256: str = ""
    result_sha256: str = ""
    target_type_resolution: TargetTypeResolution | None = None
    path_status: str = "NOT_CHECKED"
    physical_attempts: int = 0
    baseline_preserved: bool = True


@dataclass
class ReadOnlyQueryBudget:
    """A one-case physical-attempt meter. Retrying is not supported."""

    max_attempts: int = MAX_PHYSICAL_SELECT_ATTEMPTS
    attempts_used: int = 0
    query_sha256: str = ""

    def reserve(self, query: str) -> None:
        if self.max_attempts != MAX_PHYSICAL_SELECT_ATTEMPTS:
            raise ValueError("count_adapter_requires_exactly_one_attempt_budget")
        if self.attempts_used >= self.max_attempts:
            raise RuntimeError("read_only_select_budget_exhausted")
        if not is_single_read_only_count_select(query):
            raise ValueError("query_rejected_by_single_select_contract")
        self.attempts_used += 1
        self.query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()


def is_single_read_only_count_select(query: str) -> bool:
    """Accept only the fixed aggregate shape; no user text enters its syntax."""
    text = str(query or "").strip()
    if not text or len(_SELECT_WORD.findall(text)) != 1:
        return False
    if _FORBIDDEN_QUERY_WORD.search(text) or not _COUNT_DISTINCT.search(text):
        return False
    if not re.match(r"^PREFIX\s+ns:\s*<http://rdf\.freebase\.com/ns/>\s+SELECT\b", text, re.IGNORECASE):
        return False
    if not re.search(r"\bWHERE\s*\{", text, re.IGNORECASE) or not text.endswith("}"):
        return False
    if "?count" not in text or "type.object.type" not in text:
        return False
    return True


def _worker(pipe, executor: Callable[[str], Mapping[str, Any]], query: str) -> None:
    try:
        if hasattr(os, "setsid"):
            os.setsid()
        result = executor(query)
        pipe.send(("ok", result))
    except BaseException as exc:  # returned to parent; there is no retry path
        try:
            pipe.send(("error", type(exc).__name__, str(exc)[:500]))
        except BaseException:
            pass
    finally:
        pipe.close()


def _kill_process_tree(process: multiprocessing.Process) -> None:
    if process.pid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            process.terminate()
    else:
        process.terminate()
    process.join(timeout=0.25)
    if process.is_alive():
        if process.pid is not None and hasattr(os, "killpg"):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                process.kill()
        else:
            process.kill()
        process.join(timeout=1.0)


def execute_bounded_select(
    query: str,
    executor: Callable[[str], Mapping[str, Any]],
    budget: ReadOnlyQueryBudget,
    *,
    timeout_seconds: float = 30.0,
) -> Mapping[str, Any]:
    """Run one injected SELECT in a killable child; the executor must not retry.

    The process boundary enforces a hard wall-clock limit even if the injected
    executor ignores socket timeouts. A production executor still must issue
    exactly one HTTP request and disable its own retries.
    """
    if timeout_seconds <= 0 or timeout_seconds > MAX_TIMEOUT_SECONDS:
        raise ValueError("timeout_out_of_bounds")
    budget.reserve(query)
    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in methods else "spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(child, executor, query), daemon=False)
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            _kill_process_tree(process)
            raise TimeoutError("bounded_select_timeout")
        message = parent.recv()
        process.join(timeout=1.0)
        if process.is_alive():
            _kill_process_tree(process)
        if not message or message[0] != "ok":
            kind = message[1] if len(message) > 1 else "empty_worker_result"
            raise RuntimeError(f"bounded_executor_error:{kind}")
        return message[1]
    finally:
        parent.close()
        if process.is_alive():
            _kill_process_tree(process)


def _router_answer_type(prompt_history: Any) -> str:
    if not isinstance(prompt_history, list):
        return ""
    for event in reversed(prompt_history):
        if not isinstance(event, Mapping):
            continue
        parsed = event.get("parsed_result")
        if isinstance(parsed, Mapping):
            value = parsed.get("requested_final_answer_type") or parsed.get("shared_answer_type")
            if value:
                return str(value).strip()
        value = event.get("requested_final_answer_type")
        if value:
            return str(value).strip()
    return ""


def callsite_from_cor_shape(
    question: str,
    topic_entities: list[Any],
    cor_result: Mapping[str, Any],
    *,
    count_plan: CountPlanContract | None = None,
    callsite_provenance: tuple[EvidenceRef, ...] = (),
) -> RuntimeCallsite:
    """Read only the actual public CoRAgent inputs/output and optional plan."""
    topic_id = ""
    if topic_entities:
        first = topic_entities[0]
        topic_id = str(first.get("id", "") if isinstance(first, Mapping) else getattr(first, "id", ""))
    return RuntimeCallsite(
        question=str(question or ""),
        topic_entity_id=topic_id,
        baseline_output=cor_result,
        requested_answer_type=_router_answer_type(cor_result.get("prompt_history")),
        count_plan=count_plan,
        callsite_provenance=callsite_provenance,
    )


def _fallback(
    callsite: RuntimeCallsite,
    status: str,
    reason: str,
    *,
    type_resolution: TargetTypeResolution | None = None,
    path_status: str = "NOT_CHECKED",
    query: str = "",
    budget: ReadOnlyQueryBudget | None = None,
) -> AdapterOutcome:
    return AdapterOutcome(
        status=status,
        output=callsite.baseline_output,
        reason=reason,
        query=query,
        query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest() if query else "",
        target_type_resolution=type_resolution,
        path_status=path_status,
        physical_attempts=budget.attempts_used if budget else 0,
        baseline_preserved=True,
    )


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_integer_count(payload: Mapping[str, Any]) -> str | None:
    results = payload.get("results")
    bindings = results.get("bindings") if isinstance(results, Mapping) else None
    if not isinstance(bindings, list) or len(bindings) != 1 or not isinstance(bindings[0], Mapping):
        return None
    binding = bindings[0].get("count")
    if not isinstance(binding, Mapping):
        return None
    datatype = str(binding.get("datatype", ""))
    if datatype and datatype not in _ACCEPTED_INTEGER_DATATYPES:
        return None
    value = _parse_sparql_json_count(payload)
    return value if value is not None and _INTEGER.fullmatch(value) else None


def run_count_adapter(
    callsite: RuntimeCallsite,
    executor: Callable[[str], Mapping[str, Any]] | None = None,
    *,
    budget: ReadOnlyQueryBudget | None = None,
    timeout_seconds: float = 30.0,
) -> AdapterOutcome:
    """Count only when every typed runtime contract gate is satisfied."""
    triggered, trigger_reason = classify_count_intent(callsite.question)
    if not triggered:
        return _fallback(callsite, "PRESERVED_NO_COUNT_TRIGGER", trigger_reason)
    if callsite.count_plan is None:
        return _fallback(callsite, "UNKNOWN_MISSING_PROVENANCE", "runtime_emitted_no_finalized_count_plan")
    plan = callsite.count_plan
    if plan.upstream_prompt_tokens is not None and plan.upstream_prompt_limit is not None:
        if plan.upstream_prompt_tokens > plan.upstream_prompt_limit:
            return _fallback(callsite, "PROMPT_BUDGET_EXCEEDED", "recorded upstream prompt estimate exceeds its frozen limit")
    type_resolution = resolve_target_type(
        plan.target_variable,
        plan.target_type_evidence,
        answer_type_hint=callsite.requested_answer_type,
    )
    path_evidence = PathEvidence(
        target_variable=plan.target_variable,
        query_target_variable=plan.query_target_variable,
        expected_steps=plan.expected_steps,
        executed_steps=plan.executed_steps,
        query_status=plan.query_status,
        plan_finalized=plan.finalized,
        plan_shape=plan.plan_shape,
        required_external_binding_variables=plan.required_external_binding_variables,
        active_binding_tuples=plan.active_binding_tuples,
        provenance=plan.provenance,
        operator=plan.operator,
        required_constraints_complete=plan.required_constraints_complete,
        all_pairwise_node_inequalities_represented=plan.all_pairwise_node_inequalities_represented,
        type_filter_variable=plan.type_filter_variable,
    )
    path_check = check_path_completeness(path_evidence, type_resolution)
    if not path_check.ready:
        return _fallback(
            callsite,
            path_check.status,
            path_check.reason,
            type_resolution=type_resolution,
            path_status=path_check.status,
        )
    if plan.semantics != "distinct_target_entities":
        return _fallback(callsite, "UNSUPPORTED_OPERATOR_OR_ORDERING", "only distinct target-entity counts are supported", type_resolution=type_resolution, path_status=path_check.status)
    if not _ENTITY_ID.fullmatch(callsite.topic_entity_id):
        return _fallback(callsite, "UNSUPPORTED_OPERATOR_OR_ORDERING", "malformed or missing topic entity ID", type_resolution=type_resolution, path_status=path_check.status)
    if not plan.expected_steps or plan.expected_steps[0].input_variable not in plan.required_external_binding_variables:
        return _fallback(callsite, "MISSING_ACTIVE_BINDING_TUPLES", "the compiled path is not anchored to an explicit bound topic variable", type_resolution=type_resolution, path_status=path_check.status)
    if plan.required_external_binding_variables != (plan.expected_steps[0].input_variable,):
        return _fallback(callsite, "UNSUPPORTED_OPERATOR_OR_ORDERING", "the current compiler supports one fixed topic binding and no additional external binding variables", type_resolution=type_resolution, path_status=path_check.status)
    bindings = plan.active_binding_tuples or ()
    if not bindings or any(str(row.get(plan.expected_steps[0].input_variable, "")) != callsite.topic_entity_id for row in bindings):
        return _fallback(callsite, "MISSING_ACTIVE_BINDING_TUPLES", "the explicit source binding does not match the runtime topic entity", type_resolution=type_resolution, path_status=path_check.status)
    relation_chain = tuple(RelationStep(step.relation_id, step.direction == "out") for step in plan.expected_steps)
    context = RuntimeCountContext(
        question=callsite.question,
        topic_entity_id=callsite.topic_entity_id,
        relation_chain=relation_chain,
        grounded_query_status=plan.query_status,
        target_type_id=str(type_resolution.resolved_type_id),
        requested_answer_type=callsite.requested_answer_type,
        baseline_answer="",
    )
    try:
        query = build_count_query(context)
    except (TypeError, ValueError) as exc:
        return _fallback(callsite, "UNSUPPORTED_OPERATOR_OR_ORDERING", f"query_context_rejected:{exc}", type_resolution=type_resolution, path_status=path_check.status)
    if not is_single_read_only_count_select(query):
        return _fallback(callsite, "QUERY_REJECTED", "compiled query violates the one-read-only-SELECT contract", type_resolution=type_resolution, path_status=path_check.status, query=query)
    if executor is None:
        return _fallback(callsite, "READY_NO_EXECUTOR", "all offline gates pass but no executor was injected", type_resolution=type_resolution, path_status=path_check.status, query=query)
    query_budget = budget or ReadOnlyQueryBudget()
    try:
        payload = execute_bounded_select(query, executor, query_budget, timeout_seconds=timeout_seconds)
    except TimeoutError:
        return _fallback(callsite, "QUERY_TIMEOUT", "bounded child executor timed out; no retry", type_resolution=type_resolution, path_status=path_check.status, query=query, budget=query_budget)
    except Exception as exc:
        return _fallback(callsite, "QUERY_FAILED", f"single SELECT attempt failed:{type(exc).__name__}", type_resolution=type_resolution, path_status=path_check.status, query=query, budget=query_budget)
    value = _normalize_integer_count(payload) if isinstance(payload, Mapping) else None
    if value is None:
        return _fallback(callsite, "UNKNOWN_RESULT", "missing, null, malformed or non-integer aggregate binding; unknown is not zero", type_resolution=type_resolution, path_status=path_check.status, query=query, budget=query_budget)
    try:
        response_hash = _hash_json(payload)
    except (TypeError, ValueError):
        return _fallback(callsite, "UNKNOWN_RESULT", "aggregate response cannot be deterministically serialized", type_resolution=type_resolution, path_status=path_check.status, query=query, budget=query_budget)
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    result_ref = EvidenceRef(
        source_name="single_read_only_select_response",
        source_sha256=response_hash,
        field_path="results.bindings[0].count.value",
        source_class="runtime",
    )
    output = {
        "contract_version": CONTRACT_VERSION,
        "action": "COUNT",
        "answer_kind": "aggregate",
        "question": callsite.question,
        "scalar_answer": value,
        "results": [{"type": "literal", "value": value, "datatype": _XSD_INTEGER}],
        "count_provenance": {
            "query_sha256": query_hash,
            "result": result_ref.as_dict(),
            "operator": plan.operator,
            "target_variable": plan.target_variable,
            "target_type_id": type_resolution.resolved_type_id,
        },
    }
    return AdapterOutcome(
        status="COUNT_EXECUTED",
        output=output,
        reason="one typed COUNT(DISTINCT target) result normalized to xsd:integer",
        query=query,
        query_sha256=query_hash,
        result_sha256=response_hash,
        target_type_resolution=type_resolution,
        path_status=path_check.status,
        physical_attempts=query_budget.attempts_used,
        baseline_preserved=False,
    )
