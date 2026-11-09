"""Bounded, fail-closed Freebase label resolution for Iteration 23 diagnostics.

This module is isolated from production code. It has no default transport and
cannot contact a SPARQL endpoint unless a caller explicitly injects one.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import re
from typing import Any, Callable, Mapping, Sequence

from pilot import require_read_only_select
from deadline import DeadlineExceeded, DeadlineGuard


FREEBASE_NS = "http://rdf.freebase.com/ns/"
ENTITY_ID_RE = re.compile(r"^(?:m|g)\.[A-Za-z0-9_]+$")
MAX_LABEL_BATCH_SIZE = 100
MAX_LABEL_QUERY_ATTEMPTS = 12


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class QueryBudgetExceeded(RuntimeError):
    pass


class QueryExecutionFailed(RuntimeError):
    pass


class UnmeteredQueryPath(RuntimeError):
    pass


class LabelResolutionStop(BaseException):
    """Fail closed through entity_search's broad Exception fallback."""

    def __init__(self, reason: str, ids: Sequence[str] = ()) -> None:
        self.reason = reason
        self.ids = tuple(sorted(set(str(value) for value in ids)))
        super().__init__(f"{reason}: {', '.join(self.ids)}" if self.ids else reason)


class PhysicalQueryMeter:
    """A one-transport-attempt meter; failures consume a physical query slot."""

    def __init__(self, *, hard_cap: int, transport: Callable[[str], Mapping[str, Any]],
                 transport_kind: str = "injected_single_attempt",
                 deadline: DeadlineGuard | None = None) -> None:
        if not isinstance(hard_cap, int) or hard_cap < 1:
            raise ValueError("hard_cap must be a positive integer")
        if not callable(transport):
            raise TypeError("an explicit single-attempt transport is required")
        self.hard_cap = hard_cap
        self.transport = transport
        self.transport_kind = str(transport_kind)
        self.deadline = deadline
        self.physical_attempts = 0
        self.blocked_requests = 0
        self.events: list[dict[str, Any]] = []
        self.category_counts: dict[str, int] = {}

    def execute(self, query: str, *, category: str, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        require_read_only_select(query)
        category = str(category or "unspecified")
        query_hash = sha256_text(query)
        if self.deadline is not None:
            try:
                self.deadline.assert_open()
            except DeadlineExceeded:
                self.blocked_requests += 1
                self.events.append({
                    "event": "BLOCKED_BY_DEADLINE", "query_sha256": query_hash,
                    "category": category, "physical_attempts": self.physical_attempts,
                    "transport_invoked": False, "transport_kind": self.transport_kind,
                })
                raise
        if self.physical_attempts >= self.hard_cap:
            self.blocked_requests += 1
            self.events.append({
                "event": "BLOCKED_BEFORE_TRANSPORT", "query_sha256": query_hash,
                "category": category, "physical_attempts": self.physical_attempts,
                "hard_cap": self.hard_cap, "transport_invoked": False,
                "transport_kind": self.transport_kind,
            })
            raise QueryBudgetExceeded(f"physical SPARQL cap {self.hard_cap} reached")

        # Reserve the physical slot before invoking transport. The callback is
        # contractually a single attempt; this meter never retries it.
        self.physical_attempts += 1
        index = self.physical_attempts
        self.category_counts[category] = self.category_counts.get(category, 0) + 1
        event = {
            "event": "ATTEMPT_STARTED", "query_index": index,
            "query_sha256": query_hash, "query": query,
            "category": category, "context": dict(context or {}),
            "retries": 0, "transport_invoked": True,
            "transport_kind": self.transport_kind,
        }
        self.events.append(event)
        try:
            response = self.transport(query)
        except BaseException as exc:
            event.update({"status": "TRANSPORT_ERROR", "error": str(exc)})
            raise QueryExecutionFailed(f"single SPARQL attempt failed at query {index}") from exc

        if not isinstance(response, Mapping) or response.get("status") != "ok":
            status = response.get("status") if isinstance(response, Mapping) else "malformed_response"
            event.update({"status": "QUERY_ERROR", "response_status": status})
            raise QueryExecutionFailed(f"SPARQL query {index} returned status={status}")
        rows = response.get("rows")
        if not isinstance(rows, list):
            event.update({"status": "MALFORMED_ROWS"})
            raise QueryExecutionFailed(f"SPARQL query {index} returned non-list rows")
        event.update({"status": "ok", "row_count": len(rows)})
        return {"status": "ok", "rows": rows, "query_sha256": query_hash, "query_index": index}


def build_label_query(entity_ids: Sequence[str]) -> str:
    if not entity_ids:
        raise ValueError("entity_ids must not be empty")
    if any(not isinstance(value, str) or not ENTITY_ID_RE.fullmatch(value) for value in entity_ids):
        raise ValueError("label query accepts only well-formed Freebase m./g. IDs")
    values = " ".join(f"ns:{entity_id}" for entity_id in entity_ids)
    return (
        "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
        "SELECT DISTINCT ?entity ?targetEntity\n"
        "WHERE {\n"
        f"VALUES ?entity {{ {values} }}\n"
        "OPTIONAL {\n"
        "  ?entity ns:type.object.name ?targetEntity .\n"
        "  FILTER (lang(?targetEntity) = '' OR langMatches(lang(?targetEntity), 'en'))\n"
        "}\n"
        "}"
    )


def _entity_id_from_binding(binding: Mapping[str, Any]) -> str:
    entity = binding.get("entity", {})
    if not isinstance(entity, Mapping) or entity.get("type") != "uri":
        raise ValueError("label row entity must be a URI binding")
    uri = str(entity.get("value", ""))
    if not uri.startswith(FREEBASE_NS):
        raise ValueError("label row entity is outside the Freebase namespace")
    entity_id = uri[len(FREEBASE_NS):]
    if not ENTITY_ID_RE.fullmatch(entity_id):
        raise ValueError("label row contains a malformed Freebase ID")
    return entity_id


def _language_rank(language: str) -> int:
    language = str(language or "").strip().lower()
    if language == "en":
        return 0
    if not language:
        return 1
    return 2


class BoundedLabelResolver:
    """Resolve cache misses in bounded SPARQL batches and preserve provenance."""

    def __init__(self, *, label_cache: Mapping[str, str], meter: PhysicalQueryMeter,
                 max_batch_size: int = MAX_LABEL_BATCH_SIZE,
                 max_label_queries: int = MAX_LABEL_QUERY_ATTEMPTS,
                 cache_provenance: str = "frozen_iteration19_archive") -> None:
        if not 1 <= max_batch_size <= MAX_LABEL_BATCH_SIZE:
            raise ValueError(f"max_batch_size must be between 1 and {MAX_LABEL_BATCH_SIZE}")
        if not 1 <= max_label_queries <= MAX_LABEL_QUERY_ATTEMPTS:
            raise ValueError(f"max_label_queries must be between 1 and {MAX_LABEL_QUERY_ATTEMPTS}")
        self.labels: dict[str, str] = {}
        self.provenance: dict[str, dict[str, Any]] = {}
        self.meter = meter
        self.max_batch_size = max_batch_size
        self.max_label_queries = max_label_queries
        self.label_query_attempts = 0
        self.outcomes: list[dict[str, Any]] = []
        for raw_id, raw_label in label_cache.items():
            entity_id = str(raw_id).strip()
            label = str(raw_label).strip()
            if not ENTITY_ID_RE.fullmatch(entity_id):
                raise ValueError(f"invalid ID in trusted cache: {entity_id!r}")
            if not label or label == "UnName_Entity":
                raise ValueError(f"trusted cache contains a blank or placeholder label for {entity_id}")
            if entity_id in self.labels and self.labels[entity_id] != label:
                raise ValueError(f"trusted cache has conflicting labels for {entity_id}")
            self.labels[entity_id] = label
            self.provenance[entity_id] = {"source": cache_provenance}

    def _event(self, entity_id: str, status: str, **details: Any) -> None:
        self.outcomes.append({"entity_id": entity_id, "status": status, **details})

    def resolve(self, entity_ids: Sequence[str]) -> dict[str, str]:
        ordered: list[str] = []
        seen: set[str] = set()
        malformed: list[str] = []
        for raw_id in entity_ids:
            entity_id = str(raw_id or "").strip()
            if not ENTITY_ID_RE.fullmatch(entity_id):
                malformed.append(entity_id)
                self._event(entity_id, "MALFORMED_ID_OR_LITERAL_REJECTED")
                continue
            if entity_id not in seen:
                seen.add(entity_id)
                ordered.append(entity_id)
        if malformed:
            raise LabelResolutionStop("malformed entity ID or literal passed to label resolver", malformed)

        pending: list[str] = []
        for entity_id in ordered:
            if entity_id in self.labels:
                self._event(entity_id, "CACHE_HIT", provenance=self.provenance[entity_id]["source"])
            else:
                pending.append(entity_id)

        for start in range(0, len(pending), self.max_batch_size):
            chunk = pending[start:start + self.max_batch_size]
            if self.label_query_attempts >= self.max_label_queries:
                for entity_id in chunk:
                    self._event(entity_id, "LABEL_QUERY_BUDGET_EXHAUSTED")
                raise LabelResolutionStop("bounded label-query allowance exhausted", chunk)
            query = build_label_query(chunk)
            try:
                response = self.meter.execute(
                    query, category="label_resolution",
                    context={"entity_ids": chunk, "batch_size": len(chunk)},
                )
                self.label_query_attempts += 1
            except QueryBudgetExceeded as exc:
                for entity_id in chunk:
                    self._event(entity_id, "GLOBAL_QUERY_BUDGET_EXHAUSTED")
                raise LabelResolutionStop("global SPARQL query budget exhausted", chunk) from exc
            except DeadlineExceeded as exc:
                for entity_id in chunk:
                    self._event(entity_id, "PILOT_DEADLINE_EXHAUSTED")
                raise LabelResolutionStop("pilot deadline reached before label lookup", chunk) from exc
            except QueryExecutionFailed as exc:
                # Count label-resolution attempts even when transport/status fails.
                self.label_query_attempts += 1
                for entity_id in chunk:
                    self._event(entity_id, "LABEL_QUERY_FAILED", query_sha256=sha256_text(query))
                raise LabelResolutionStop("label lookup failed; no retry", chunk) from exc

            query_hash = response["query_sha256"]
            rows = response["rows"]
            row_hash = sha256_text(canonical_json(rows))
            grouped: dict[str, list[tuple[int, str, str]]] = {entity_id: [] for entity_id in chunk}
            unexpected: list[str] = []
            malformed_rows: list[str] = []
            try:
                for row in rows:
                    if not isinstance(row, Mapping):
                        raise ValueError("label result row is not an object")
                    entity_id = _entity_id_from_binding(row)
                    if entity_id not in grouped:
                        unexpected.append(entity_id)
                        continue
                    label_binding = row.get("targetEntity", {})
                    if not isinstance(label_binding, Mapping):
                        raise ValueError("targetEntity label binding is malformed")
                    if not label_binding:
                        continue  # OPTIONAL matched an entity with no name.
                    if label_binding.get("type") not in {"literal", "typed-literal"}:
                        raise ValueError("targetEntity label must be a literal")
                    label = str(label_binding.get("value", "")).strip()
                    if not label:
                        continue
                    language = str(label_binding.get("xml:lang", ""))
                    grouped[entity_id].append((_language_rank(language), label, language))
            except (TypeError, ValueError) as exc:
                malformed_rows.append(str(exc))

            if unexpected or malformed_rows:
                for entity_id in chunk:
                    self._event(entity_id, "MALFORMED_OR_UNEXPECTED_LABEL_RESPONSE",
                                query_sha256=query_hash, unexpected_ids=unexpected,
                                errors=malformed_rows)
                raise LabelResolutionStop("malformed or unexpected label-query response", chunk)

            unresolved: list[str] = []
            ambiguous: list[str] = []
            resolved_now: dict[str, tuple[str, str]] = {}
            for entity_id in chunk:
                candidates = grouped[entity_id]
                if not candidates:
                    unresolved.append(entity_id)
                    self._event(entity_id, "UNRESOLVED_LABEL_ABSENT", query_sha256=query_hash)
                    continue
                best_rank = min(candidate[0] for candidate in candidates)
                best = sorted({candidate[1] for candidate in candidates if candidate[0] == best_rank})
                if len(best) != 1:
                    ambiguous.append(entity_id)
                    self._event(entity_id, "AMBIGUOUS_LABELS", query_sha256=query_hash,
                                language_rank=best_rank, conflicting_labels=best)
                    continue
                chosen_label = best[0]
                chosen_language = sorted(
                    {candidate[2] for candidate in candidates
                     if candidate[0] == best_rank and candidate[1] == chosen_label}
                )[0]
                resolved_now[entity_id] = (chosen_label, chosen_language)

            # Keep validated results in-session, but never cache unresolved or
            # ambiguous IDs and never overwrite an existing trusted mapping.
            for entity_id, (label, language) in resolved_now.items():
                if entity_id in self.labels and self.labels[entity_id] != label:
                    raise LabelResolutionStop("attempt to change an immutable ID/label mapping", [entity_id])
                self.labels[entity_id] = label
                self.provenance[entity_id] = {
                    "source": "bounded_read_only_label_select",
                    "query_sha256": query_hash,
                    "result_rows_sha256": row_hash,
                    "language": language,
                    "batch_ids": chunk,
                }
                self._event(entity_id, "RESOLVED_DYNAMIC", label=label,
                            language=language, query_sha256=query_hash,
                            result_rows_sha256=row_hash)
            if unresolved or ambiguous:
                failed = unresolved + ambiguous
                not_attempted = pending[start + len(chunk):]
                for entity_id in not_attempted:
                    self._event(entity_id, "NOT_ATTEMPTED_AFTER_BATCH_FAILURE")
                raise LabelResolutionStop("one or more labels are absent or ambiguous", failed)

        return {entity_id: self.labels[entity_id] for entity_id in ordered}

    def resolve_one(self, entity_id: str) -> str:
        return self.resolve([entity_id])[entity_id]


@contextmanager
def install_metered_freebase_backend(backend: Any, *, meter: PhysicalQueryMeter,
                                     resolver: BoundedLabelResolver):
    """Temporarily route every Freebase backend/query-template path through meters.

    Direct SPARQLWrapper use inside db_func is blocked while installed. Methods
    and module globals are always restored, including after BaseException.
    """
    from chain_of_relations.kg_backend import freebase_backend as backend_module
    from chain_of_relations.kg_backend.freebase import db_func

    changes: list[tuple[Any, str, bool, Any]] = []

    def patch(target: Any, name: str, replacement: Any) -> None:
        had_own = name in getattr(target, "__dict__", {})
        previous = getattr(target, name, None)
        changes.append((target, name, had_own, previous))
        setattr(target, name, replacement)

    def query_meta(query: str) -> dict[str, Any]:
        return meter.execute(query, category="navigation")

    def query_rows(query: str) -> list[dict[str, Any]]:
        return meter.execute(query, category="navigation")["rows"]

    def forbidden_wrapper(*_args: Any, **_kwargs: Any) -> Any:
        raise UnmeteredQueryPath("unmetered SPARQLWrapper path blocked by Iteration 23 adapter")

    try:
        patch(backend, "execute_sparql_with_meta", query_meta)
        patch(backend, "execute_sparql", query_rows)
        patch(backend, "id2entity_names", resolver.resolve)
        patch(backend, "id2entity_name_or_type", resolver.resolve_one)
        # FreebaseBackend imported these names directly from db_func, so patch
        # the aliases as well as the defining module to close import-time paths.
        patch(backend_module, "execurte_sparql_with_meta", query_meta)
        patch(backend_module, "execurte_sparql", query_rows)
        patch(backend_module, "id2entity_name_or_type", resolver.resolve_one)
        patch(backend_module, "id2entity_names", resolver.resolve)
        patch(backend_module, "freebase_id2entity_names", resolver.resolve)
        patch(db_func, "id2entity_names", resolver.resolve)
        patch(db_func, "id2entity_name_or_type", resolver.resolve_one)
        patch(db_func, "execurte_sparql_with_meta", query_meta)
        patch(db_func, "execurte_sparql", query_rows)
        patch(db_func, "SPARQLWrapper", forbidden_wrapper)
        yield backend
    finally:
        for target, name, had_own, previous in reversed(changes):
            if had_own:
                setattr(target, name, previous)
            else:
                try:
                    delattr(target, name)
                except AttributeError:
                    pass
