"""Fail-closed, isolated SPARQL COUNT operator prototype.

No network client is constructed here. Callers must inject a single-attempt
read-only executor, which makes offline behavior and request bounds explicit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping


_FREEBASE_ID = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_INTEGER = re.compile(r"^(?:0|[1-9][0-9]*)$")
_COUNT_INTENT = re.compile(
    r"\b(?:how many|number of|count of|what amount of|what is the amount of|"
    r"what is the total amount of|total amount of)\b",
    re.IGNORECASE,
)
_RANKED_NUMBER = re.compile(
    r"\b(?:least|lowest|greatest|highest|maximum|minimum)\s+(?:possible\s+)?number of\b",
    re.IGNORECASE,
)
_HOW_MANY_MEASURE = re.compile(
    r"\bhow many\s+(?:miles?|kilomet(?:er|re)s?|meters?|metres?|feet|inches?|"
    r"dollars?|euros?|pounds?|kilograms?|grams?|degrees?|percent(?:age)?|"
    r"seconds?|minutes?|hours?|days?|months?|years?)\b",
    re.IGNORECASE,
)
_AMOUNT_MEASURE = re.compile(
    r"\bamount of (?:money|cash|revenue|budget|cost|debt|salary|energy|water|fuel|electricity)\b",
    re.IGNORECASE,
)
_NUMBER_IDENTIFIER = re.compile(
    r"\b(?:what|which)\s+(?:is|was)\s+(?:the\s+)?number of the\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RelationStep:
    """A grounded relation traversed from the current node.

    `forward_from_current=True` means current --relation--> next; False means
    next --relation--> current.
    """

    relation_id: str
    forward_from_current: bool = True


@dataclass(frozen=True)
class RuntimeCountContext:
    question: str
    topic_entity_id: str
    relation_chain: tuple[RelationStep, ...]
    grounded_query_status: str
    target_type_id: str = ""
    requested_answer_type: str = ""
    baseline_answer: str = ""


@dataclass(frozen=True)
class CountRecoveryResult:
    status: str
    answer: str
    query: str
    reason: str
    used_extension: bool = False
    executor_calls: int = 0


def classify_count_intent(question: str) -> tuple[bool, str]:
    """Return a conservative deterministic count trigger and its reason.

    Ranking questions such as “least number of enrolled patients” are numeric
    property comparisons, not COUNT of path endpoints, and are excluded.
    """
    text = str(question or "").strip()
    if not text:
        return False, "empty_question"
    if _RANKED_NUMBER.search(text):
        return False, "numeric_property_ranking"
    if _HOW_MANY_MEASURE.search(text):
        return False, "measured_numeric_property"
    if _AMOUNT_MEASURE.search(text):
        return False, "measured_numeric_property"
    if _NUMBER_IDENTIFIER.search(text):
        return False, "numeric_identifier"
    if _COUNT_INTENT.search(text):
        return True, "explicit_count_wording"
    return False, "no_explicit_count_wording"


def _valid_id(value: str) -> bool:
    return bool(_FREEBASE_ID.fullmatch(str(value or "")))


def build_count_query(context: RuntimeCountContext) -> str:
    """Build a typed path aggregate with joins, direction and node inequality."""
    if not _valid_id(context.topic_entity_id):
        raise ValueError("invalid_topic_entity_id")
    if not context.relation_chain:
        raise ValueError("empty_relation_chain")
    if len(context.relation_chain) > 6:
        raise ValueError("relation_chain_too_deep")
    if not _valid_id(context.target_type_id):
        raise ValueError("missing_or_invalid_target_type_id")
    if any(not _valid_id(step.relation_id) for step in context.relation_chain):
        raise ValueError("invalid_relation_id")

    node_count = len(context.relation_chain) + 1
    nodes = [f"?iter30_node_{index}" for index in range(node_count)]
    triples = [f"VALUES {nodes[0]} {{ ns:{context.topic_entity_id} }}"]
    for index, step in enumerate(context.relation_chain):
        subject, obj = (nodes[index], nodes[index + 1]) if step.forward_from_current else (nodes[index + 1], nodes[index])
        triples.append(f"{subject} ns:{step.relation_id} {obj} .")
    triples.append(f"{nodes[-1]} ns:type.object.type ns:{context.target_type_id} .")

    # GrailQA's COUNT gold forms use a DISTINCT subquery and exclude repeated
    # graph nodes. Preserve that set/path contract for the supported fragment.
    inequalities = [
        f"FILTER ({nodes[left]} != {nodes[right]})"
        for left in range(node_count)
        for right in range(left + 1, node_count)
    ]
    body = "\n  ".join(triples + inequalities)
    return (
        "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
        f"SELECT (COUNT(DISTINCT {nodes[-1]}) AS ?count) WHERE {{\n  {body}\n}}"
    )


def _parse_sparql_json_count(payload: Mapping[str, Any]) -> str | None:
    if payload.get("status") not in (None, "ok"):
        return None
    bindings = ((payload.get("results") or {}).get("bindings"))
    if not isinstance(bindings, list) or len(bindings) != 1 or not isinstance(bindings[0], dict):
        return None
    binding = bindings[0].get("count")
    if not isinstance(binding, dict) or binding.get("type") not in {"literal", "typed-literal"}:
        return None
    value = str(binding.get("value", "")).strip()
    if not _INTEGER.fullmatch(value):
        return None
    return value


def recover_count(
    context: RuntimeCountContext,
    query_executor: Callable[[str, float], Mapping[str, Any]] | None = None,
    *,
    timeout_seconds: float = 120.0,
) -> CountRecoveryResult:
    """Run at most one injected query, otherwise preserve the baseline answer."""
    baseline = str(context.baseline_answer or "")
    matched, trigger_reason = classify_count_intent(context.question)
    if not matched:
        return CountRecoveryResult("preserved", baseline, "", trigger_reason)
    if context.grounded_query_status != "ok":
        return CountRecoveryResult("preserved", baseline, "", "grounded_path_not_successful")
    if not context.relation_chain:
        return CountRecoveryResult("preserved", baseline, "", "missing_grounded_relation_chain")
    if not _valid_id(context.target_type_id):
        return CountRecoveryResult("preserved", baseline, "", "target_type_id_unavailable")
    if query_executor is None:
        return CountRecoveryResult("preserved", baseline, "", "no_executor_injected")
    if timeout_seconds <= 0 or timeout_seconds > 120:
        return CountRecoveryResult("preserved", baseline, "", "timeout_out_of_bounds")

    try:
        query = build_count_query(context)
    except (TypeError, ValueError) as error:
        return CountRecoveryResult("preserved", baseline, "", f"invalid_runtime_context:{error}")
    try:
        payload = query_executor(query, timeout_seconds)
    except TimeoutError:
        return CountRecoveryResult("preserved", baseline, query, "executor_timeout", executor_calls=1)
    except Exception as error:  # an injected transport/parser error is a case failure
        return CountRecoveryResult("preserved", baseline, query, f"executor_error:{type(error).__name__}", executor_calls=1)

    answer = _parse_sparql_json_count(payload) if isinstance(payload, Mapping) else None
    if answer is None:
        return CountRecoveryResult("preserved", baseline, query, "malformed_or_failed_count_result", executor_calls=1)
    return CountRecoveryResult("recovered", answer, query, "typed_count_success", used_extension=True, executor_calls=1)
