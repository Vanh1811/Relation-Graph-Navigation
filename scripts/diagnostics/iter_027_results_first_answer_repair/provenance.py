"""Pure helpers for preserving executed Freebase query provenance."""
from __future__ import annotations

import hashlib
import re
from typing import Any


def relation_ids_from_executed_entity_search(query: str) -> list[str]:
    """Return Freebase predicates in executed entity-search triple patterns."""
    body = re.sub(r"(?im)^\s*PREFIX\s+[\w:-]+\s*<[^>]+>\s*", "", query)
    term = r"(?:\?[A-Za-z_][A-Za-z0-9_]*|ns:[A-Za-z0-9_.]+)"
    predicate = re.compile(rf"{term}\s+(ns:[A-Za-z0-9_.]+)\s+{term}\s*\.")
    return sorted({match.group(1)[3:] for match in predicate.finditer(body)})


def observed_relation_provenance(
    sparql_history: list[dict[str, Any]], filter_trace_id: int,
) -> list[dict[str, Any]]:
    """Retain only prior executed entity-search queries with content hashes."""
    evidence: list[dict[str, Any]] = []
    for event in sparql_history:
        if event.get("search_type") != "entity_search":
            continue
        trace_id = event.get("trace_id")
        if not isinstance(trace_id, int) or trace_id >= filter_trace_id:
            continue
        query = str(event.get("sparql", ""))
        evidence.append({
            "entity_search_trace_id": trace_id,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "observed_relation_ids_in_executed_query": relation_ids_from_executed_entity_search(query),
            "result_count": event.get("result_count"),
            "status": event.get("status"),
        })
    return evidence
