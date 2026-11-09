"""Fixture-only matched-prefix replay primitives for Iteration 21.

This module is intentionally independent of production TGAR/CoR execution.
It serializes IDs and cached graph rows, and refuses incomplete state when an
exact replay is requested.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import inspect
import json
import re
from typing import Any, Callable, Iterable, Mapping, Sequence


ANSWER_SOURCES = {"GRAPH_FILTER", "DIRECT_LLM", "OTHER"}
ARMS = {"STOP_AT_3", "CONTINUE_ONE_HOP"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty ID string")
    if any(ch.isspace() for ch in value):
        raise ValueError(f"{label} must be an ID, not a label")
    return value


def _ids(values: Iterable[Any], label: str) -> list[str]:
    result = [_require_id(value, label) for value in values]
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate IDs")
    return result


def prefix_payload(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return the fields that define a branch point, excluding arm outputs."""
    fields = (
        "question_id", "repeat", "model_settings", "boundary_depth", "route",
        "decisions", "method_events", "sparql_history", "kg_cache", "dfs_stack",
        "active_binding_rows", "current_path", "frontier_ids", "visited_node_ids",
        "binding_projection_query", "binding_projection_query_hash", "evidence_pointer",
    )
    return {field: deepcopy(snapshot.get(field)) for field in fields}


def is_complete_prefix(snapshot: Mapping[str, Any]) -> bool:
    if any(snapshot.get(key) for key in ("oracle_relation_id", "gold_id_used_for_selection", "gold_relation_used_for_selection")):
        return False
    required_nonempty = ("question_id", "repeat", "model_settings", "route", "evidence_pointer")
    if any(snapshot.get(key) in (None, "") for key in required_nonempty):
        return False
    settings = snapshot.get("model_settings")
    if not isinstance(settings, dict) or not settings.get("model"):
        return False
    if "temperature" not in settings or not any(key in settings for key in ("max_completion_tokens", "max_tokens")):
        return False
    if not isinstance(snapshot.get("boundary_depth"), int) or snapshot["boundary_depth"] < 0:
        return False
    for key in ("decisions", "method_events", "sparql_history", "kg_cache", "dfs_stack", "active_binding_rows", "current_path", "frontier_ids", "visited_node_ids"):
        if not isinstance(snapshot.get(key), list):
            return False
    if not snapshot["decisions"] or any(
        not isinstance(event, dict) or "prompt" not in event or "response" not in event
        for event in snapshot["decisions"]
    ):
        return False
    if not snapshot["kg_cache"] or any(
        not isinstance(item, dict) or not item.get("query_hash") or item.get("status") != "ok"
        or not isinstance(item.get("result_ids"), list)
        for item in snapshot["kg_cache"]
    ):
        return False
    if not snapshot["method_events"] or any(
        not isinstance(event, dict) or not event.get("stage")
        or not isinstance(event.get("relation_chain"), list)
        for event in snapshot["method_events"]
    ):
        return False
    methods = {event["stage"]: event for event in snapshot["method_events"]}
    if not {"relation_search", "relation_prune", "entity_search", "reasoning"}.issubset(methods):
        return False
    if not all(key in methods["relation_search"] for key in ("head_relation_ids", "tail_relation_ids")):
        return False
    if not all(key in methods["relation_prune"] for key in ("candidate_relation_ids_before_prune", "selected_relations")):
        return False
    if not isinstance(methods["entity_search"].get("candidate_entity_ids"), list):
        return False
    if not all(key in methods["reasoning"] for key in ("action", "response")):
        return False
    if not snapshot["sparql_history"] or any(
        not isinstance(event, dict) or not event.get("sparql") or event.get("status") is None
        for event in snapshot["sparql_history"]
    ):
        return False
    try:
        for item in snapshot["kg_cache"]:
            _ids(item["result_ids"], "cached KG result_ids")
    except (TypeError, ValueError):
        return False
    if not snapshot["active_binding_rows"]:
        return False
    try:
        _ids(snapshot["frontier_ids"], "frontier_ids")
        _ids(snapshot["visited_node_ids"], "visited_node_ids")
        for row in snapshot["active_binding_rows"]:
            if not isinstance(row, dict) or not row:
                return False
            for variable, entity_id in row.items():
                if not isinstance(variable, str) or not variable.startswith("?"):
                    return False
                _require_id(entity_id, f"binding {variable}")
            if len(row) < snapshot["boundary_depth"]:
                return False
        for state in snapshot["dfs_stack"]:
            if not isinstance(state, dict) or not isinstance(state.get("path"), list):
                return False
            if not isinstance(state.get("depth"), int):
                return False
            relation_count = 0
            for node in state["path"]:
                if not isinstance(node, dict) or node.get("kind") not in {"entity", "relation"}:
                    return False
                _require_id(node.get("id"), "DFS stack node ID")
                relation_count += node.get("kind") == "relation"
                if node.get("kind") == "relation" and node.get("direction") not in {"OUTGOING", "INCOMING"}:
                    return False
            if relation_count != state["depth"]:
                return False
        if not snapshot["current_path"]:
            return False
        current_relation_count = 0
        for node in snapshot["current_path"]:
            if not isinstance(node, dict) or node.get("kind") not in {"entity", "relation"}:
                return False
            _require_id(node.get("id"), "current path node ID")
            current_relation_count += node.get("kind") == "relation"
            if node.get("kind") == "relation" and node.get("direction") not in {"OUTGOING", "INCOMING"}:
                return False
        if current_relation_count != snapshot["boundary_depth"]:
            return False
        if snapshot.get("answer_source") not in ANSWER_SOURCES:
            return False
        if snapshot.get("filter_input_ids") is not None:
            _ids(snapshot["filter_input_ids"], "filter_input_ids")
        if snapshot.get("filter_output_ids") is not None:
            _ids(snapshot["filter_output_ids"], "filter_output_ids")
        _ids(snapshot.get("final_answer_ids", []), "final_answer_ids")
    except (TypeError, ValueError):
        return False
    return True


def seal_prefix(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a snapshot and add a reproducible content hash/completeness flag."""
    record = deepcopy(dict(snapshot))
    record["prefix_replay_exact"] = is_complete_prefix(record)
    record["prefix_hash"] = sha256_json(prefix_payload(record))
    record.setdefault("log_completeness", "COMPLETE" if record["prefix_replay_exact"] else "INCOMPLETE")
    return record


def binding_join(left: Sequence[Mapping[str, str]], right: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Natural join on shared variable IDs. Labels are never join keys."""
    joined: list[dict[str, str]] = []
    for lhs in left:
        for rhs in right:
            shared = set(lhs).intersection(rhs)
            if not shared:
                continue
            if any(lhs[key] != rhs[key] for key in shared):
                continue
            joined.append({**lhs, **rhs})
    deduped = {canonical_json(row): row for row in joined}
    return [deduped[key] for key in sorted(deduped)]


def build_pair(
    snapshot: Mapping[str, Any],
    *,
    selected_relation_id: str,
    relation_direction: str,
    candidate_relation_ids_before_prune: Sequence[str],
    candidate_relation_ids_after_prune: Sequence[str],
    kg_query: str,
    kg_rows: Sequence[Mapping[str, str]],
    kg_status: str = "ok",
    labels: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fork a complete prefix into stop and exactly-one-edge fixture arms."""
    base = seal_prefix(snapshot)
    if not base["prefix_replay_exact"]:
        raise ValueError("cannot claim matched replay from incomplete prefix state")
    if str(base.get("boundary_action", "Forward")).lower() != "forward":
        raise ValueError("a frozen non-Forward/short-path boundary is treatment-ineligible")
    if relation_direction not in {"OUTGOING", "INCOMING"}:
        raise ValueError("relation_direction must be OUTGOING or INCOMING")
    if kg_status != "ok":
        raise ValueError("continuation KG query must have status=ok; failed queries are not empty frontiers")
    selected_relation_id = _require_id(selected_relation_id, "selected_relation_id")
    before = _ids(candidate_relation_ids_before_prune, "candidate_relation_ids_before_prune")
    after = _ids(candidate_relation_ids_after_prune, "candidate_relation_ids_after_prune")
    if selected_relation_id not in after:
        raise ValueError("selected relation was not in the saved post-prune candidate IDs")
    if not set(after).issubset(set(before)):
        raise ValueError("post-prune relation IDs must be present before pruning")

    stop = deepcopy(base)
    active_binding_ids = sorted({value for row in base["active_binding_rows"] for value in row.values()})
    stop.update({
        "arm": "STOP_AT_3", "new_edge_count": 0, "new_binding_rows": [],
        "active_binding_ids": active_binding_ids,
        "selected_relation_id": None, "relation_direction": None,
        "candidate_ids_before_prune": None, "candidate_ids_after_prune": None,
        "KG_query_hash": None, "KG_result_ids": [], "gold_id_present_for_scoring": None,
    })

    continued = deepcopy(base)
    query_hash = hashlib.sha256(kg_query.encode("utf-8")).hexdigest()
    new_rows = [dict(row) for row in kg_rows]
    if new_rows:
        if any(not isinstance(row, dict) or not row for row in new_rows):
            raise ValueError("KG rows must be non-empty ID binding maps")
        for row in new_rows:
            for variable, entity_id in row.items():
                if not isinstance(variable, str) or not variable.startswith("?"):
                    raise ValueError("KG row keys must be SPARQL variable names")
                _require_id(entity_id, f"KG result {variable}")
        extensions = binding_join(base["active_binding_rows"], new_rows)
    else:
        extensions = []
    if new_rows and not extensions:
        raise ValueError("KG rows do not join the frozen active binding IDs")
    active_variables = set().union(*(set(row) for row in base["active_binding_rows"]))
    if any(len(set(row) - active_variables) != 1 for row in new_rows):
        raise ValueError("continuation KG rows must add exactly one hop variable")
    new_variable_values = [
        row[key]
        for row in new_rows
        for key in set(row) - active_variables
    ]

    visited = set(base["visited_node_ids"])
    loop_targets = sorted({value for value in new_variable_values if value in visited})
    continued.update({
        "arm": "CONTINUE_ONE_HOP",
        "new_edge_count": 1 if extensions else 0,
        "treatment": {
            "selected_relation_id": selected_relation_id,
            "relation_direction": relation_direction,
            "candidate_ids_before_prune": before,
            "candidate_ids_after_prune": after,
            "KG_query_hash": query_hash,
            "KG_result_ids": sorted(set(new_variable_values)),
            "new_binding_rows": extensions,
            "loop_target_ids": loop_targets,
            "visited_node_ids_after_hop": sorted(visited.union(new_variable_values)),
        },
        "gold_id_present_for_scoring": None,
        "labels_for_display_only": dict(labels or {}),
    })
    continued.update({
        "active_binding_ids": active_binding_ids,
        "selected_relation_id": selected_relation_id,
        "relation_direction": relation_direction,
        "candidate_ids_before_prune": before,
        "candidate_ids_after_prune": after,
        "KG_query_hash": query_hash,
        "KG_result_ids": sorted(set(new_variable_values)),
        "answer_source": "OTHER",
        "filter_input_ids": None,
        "filter_output_ids": None,
        "final_answer_ids": [],
    })
    continued["prefix_hash"] = base["prefix_hash"]
    if stop["prefix_hash"] != continued["prefix_hash"]:
        raise AssertionError("matched arms diverged before treatment")
    return {"STOP_AT_3": stop, "CONTINUE_ONE_HOP": continued}


def resume_jsonl(path: str, records: Sequence[Mapping[str, Any]]) -> int:
    """Append unseen record IDs; refuse partial lines and conflicting duplicates."""
    import os
    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if output.exists():
        raw = output.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("cannot resume a partial JSONL output")
        for line in raw.splitlines():
            row = json.loads(line)
            key = str(row["record_id"])
            digest = sha256_json(row)
            if key in existing and existing[key] != digest:
                raise ValueError(f"conflicting existing record_id: {key}")
            existing[key] = digest
    appended = 0
    with output.open("a", encoding="utf-8") as handle:
        for record in records:
            row = deepcopy(dict(record))
            key = str(row["record_id"])
            digest = sha256_json(row)
            if key in existing:
                if existing[key] != digest:
                    raise ValueError(f"resume record conflicts with existing record_id: {key}")
                continue
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            existing[key] = digest
            appended += 1
    return appended


def binding_projection_query(query: str, variables: Sequence[str]) -> str:
    """Make an isolated sidecar query that exposes intermediate binding IDs.

    WHERE text and modifiers are preserved. Because projection changes DISTINCT
    granularity, this query is diagnostic evidence capture, not a replacement
    for the production entity query.
    """
    vars_clean = list(dict.fromkeys(variables))
    if not vars_clean or any(not re.fullmatch(r"\?[A-Za-z_][A-Za-z0-9_]*", v) for v in vars_clean):
        raise ValueError("variables must be valid, non-empty SPARQL variable names")
    matches = list(re.finditer(r"(?is)\bSELECT\s+DISTINCT\s+.+?\s+WHERE\s*\{", query))
    if len(matches) != 1:
        raise ValueError("expected one SELECT DISTINCT ... WHERE clause")
    match = matches[0]
    prefix = match.group(0)
    where_pos = re.search(r"(?is)\s+WHERE\s*\{", prefix)
    assert where_pos is not None
    replacement = "SELECT DISTINCT " + " ".join(vars_clean) + prefix[where_pos.start():]
    return query[:match.start()] + replacement + query[match.end():]


def fetch_binding_projection(query: str, variables: Sequence[str], execute_query: Callable[[str], Any]) -> dict[str, Any]:
    """Execute a caller-supplied sidecar query and return ID-only bindings.

    `execute_query` is injected so this module has no implicit network/KG access.
    Only Freebase entity URIs or already-normalized Freebase IDs are accepted;
    literal/non-Freebase terms make the exact entity-binding snapshot ineligible.
    """
    sidecar = binding_projection_query(query, variables)
    meta = execute_query(sidecar)
    if not isinstance(meta, dict) or meta.get("status") != "ok":
        raise RuntimeError("binding projection query did not return status=ok")
    rows = meta.get("rows")
    if not isinstance(rows, list):
        raise ValueError("binding projection result must contain a rows list")
    normalized: list[dict[str, str]] = []
    ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("binding projection rows must be mappings")
        mapped: dict[str, str] = {}
        for variable in variables:
            raw = row.get(variable.lstrip("?"))
            if not isinstance(raw, dict) or raw.get("type") != "uri":
                raise ValueError(f"unsupported or missing RDF binding for {variable}; exact ID replay is unavailable")
            value = str(raw.get("value", "")).strip()
            prefix = "http://rdf.freebase.com/ns/"
            if value.startswith(prefix):
                value = value[len(prefix):]
            mapped[variable] = _require_id(value, f"binding {variable}")
            ids.add(value)
        normalized.append(mapped)
    normalized.sort(key=canonical_json)
    return {
        "query": sidecar,
        "query_hash": hashlib.sha256(sidecar.encode("utf-8")).hexdigest(),
        "rows": normalized,
        "result_ids": sorted(ids),
        "status": "ok",
        "limit": _query_limit(sidecar),
    }


def _query_limit(query: str) -> int | None:
    match = re.search(r"(?is)\bLIMIT\s+(\d+)", query)
    return int(match.group(1)) if match else None


def attach_binding_projection(snapshot: Mapping[str, Any], projection: Mapping[str, Any],
                              variable_order: Sequence[str]) -> dict[str, Any]:
    """Attach a complete sidecar binding result and seal a replayable prefix.

    `variable_order` lists the raw query variables in path order, ending at the
    endpoint variable. The helper renames them to ?e1…?eN so adjacent path
    records can join by IDs despite CoR using ?targetEntity at each query end.
    """
    if projection.get("status") != "ok" or not projection.get("query_hash"):
        raise ValueError("successful binding projection with query hash is required")
    rows = projection.get("rows")
    if not isinstance(rows, list):
        raise ValueError("binding projection rows are unavailable")
    limit = projection.get("limit")
    if isinstance(limit, int) and len(rows) >= limit:
        raise ValueError("binding projection reached LIMIT; completeness cannot be established")
    depth = snapshot.get("boundary_depth")
    if not isinstance(depth, int) or len(variable_order) != depth:
        raise ValueError("variable_order must contain one endpoint variable per traversed edge")
    canonical_vars = [f"?e{i}" for i in range(1, depth + 1)]
    if any(variable not in row for row in rows for variable in variable_order):
        raise ValueError("binding projection omitted a required path variable")
    canonical_rows = [
        {canonical: row[raw] for canonical, raw in zip(canonical_vars, variable_order)}
        for row in rows
    ]
    canonical_rows.sort(key=canonical_json)
    frontier = _ids(snapshot.get("frontier_ids", []), "frontier_ids")
    observed_frontier = sorted({row[canonical_vars[-1]] for row in canonical_rows}) if canonical_rows else []
    if observed_frontier != sorted(frontier):
        raise ValueError("sidecar endpoints do not exactly match the executor's saved frontier IDs")
    record = deepcopy(dict(snapshot))
    record["active_binding_rows"] = canonical_rows
    record["active_binding_ids"] = sorted({value for row in canonical_rows for value in row.values()})
    record["binding_projection_query"] = str(projection.get("query", ""))
    record["binding_projection_query_hash"] = projection["query_hash"]
    if (not record["binding_projection_query"]
            or hashlib.sha256(record["binding_projection_query"].encode("utf-8")).hexdigest()
            != record["binding_projection_query_hash"]):
        raise ValueError("binding projection query text and hash do not match")
    record["log_completeness"] = "COMPLETE"
    return seal_prefix(record)


@contextmanager
def capture_cor_method_outputs(agent: Any, event_log: list[dict[str, Any]]):
    """Reversibly record CoRAgent method outputs without changing their returns."""
    methods = ("relation_search", "relation_prune", "entity_search", "reasoning", "validate", "generate_directly")
    instance_dict = getattr(agent, "__dict__", {})
    originals = {
        name: (getattr(agent, name), name in instance_dict, instance_dict.get(name))
        for name in methods if hasattr(agent, name)
    }

    def make_wrapper(name: str, original: Callable[..., Any]):
        def wrapped(*args: Any, **kwargs: Any):
            bound = inspect.signature(original).bind_partial(*args, **kwargs)
            values = bound.arguments
            sparql_history = values.get("sparql_history", [])
            sparql_before = len(sparql_history)
            result = original(*args, **kwargs)
            relations = values.get("relation_chain", [])
            event: dict[str, Any] = {
                "sequence": len(event_log) + 1,
                "stage": name,
                "relation_chain": [
                    {"id": str(getattr(rel, "id", "")), "direction": "OUTGOING" if bool(getattr(rel, "left", False)) else "INCOMING"}
                    for rel in relations
                ],
            }
            if name == "relation_search":
                head, tail = result
                event["head_relation_ids"] = [str(x) for x in head]
                event["tail_relation_ids"] = [str(x) for x in tail]
            elif name == "relation_prune":
                event["candidate_relation_ids_before_prune"] = [str(x) for x in values.get("head_relations", []) + values.get("tail_relations", [])]
                event["selected_relations"] = [
                    {"id": str(getattr(rel, "id", "")), "direction": "OUTGOING" if bool(getattr(rel, "left", False)) else "INCOMING"}
                    for rel in result
                ]
            elif name == "entity_search":
                ids, names, literals = result
                event.update({"candidate_entity_ids": list(ids), "candidate_entity_names": list(names), "literal_candidates": list(literals)})
            elif name == "reasoning":
                action, response, answers = result
                event.update({"action": action, "response": response, "selected_answer_labels": list(answers or [])})
            elif name == "validate":
                event.update({"filter_input_labels": list(values.get("target_entity_names", [])), "filter_output_labels": list(result or [])})
            elif name == "generate_directly":
                event.update({"direct_llm_answer": str(result or "")})

            if name in {"relation_search", "entity_search"}:
                new_sparql = sparql_history[sparql_before:]
                if name == "relation_search":
                    relation_results = {
                        "relation_search_head": event.get("head_relation_ids", []),
                        "relation_search_tail": event.get("tail_relation_ids", []),
                    }
                    for query_event in new_sparql:
                        search_type = query_event.get("search_type", "")
                        result_ids = relation_results.get(search_type, [])
                        event.setdefault("kg_cache", []).append({
                            "query_hash": hashlib.sha256(str(query_event.get("sparql", "")).encode("utf-8")).hexdigest(),
                            "status": query_event.get("status"),
                            "result_ids": list(result_ids),
                            "search_type": search_type,
                        })
                else:
                    query_event = new_sparql[-1] if new_sparql else {}
                    event["kg_cache"] = [{
                        "query_hash": hashlib.sha256(str(query_event.get("sparql", "")).encode("utf-8")).hexdigest(),
                        "status": query_event.get("status"),
                        "result_ids": list(result[0]),
                        "search_type": "entity_search",
                    }]
                event["sparql_events"] = deepcopy(new_sparql)
            event_log.append(event)
            return result
        return wrapped

    try:
        for name, (original, _, _) in originals.items():
            setattr(agent, name, make_wrapper(name, original))
        yield event_log
    finally:
        for name, (original, was_instance_attr, previous_value) in originals.items():
            if was_instance_attr:
                setattr(agent, name, previous_value)
            else:
                delattr(agent, name)


@dataclass
class BoundaryCapture:
    question_id: str
    repeat: int
    depth: int
    evidence_pointer: str
    snapshots: list[dict[str, Any]]
    model_settings: dict[str, Any] | None = None
    route: str = "cor"
    method_events: list[dict[str, Any]] | None = None

    def trace(self, frame: Any, event: str, arg: Any):
        if event != "line" or frame.f_code.co_name != "answer":
            return self.trace
        loc = frame.f_locals
        action = loc.get("action")
        if not isinstance(action, str) or action.lower() != "forward":
            return self.trace
        if loc.get("current_depth") != self.depth:
            return self.trace
        source_line = frame.f_code.co_filename
        if source_line.endswith("chain_of_relations/methods/cor/agent.py") is False:
            return self.trace
        # The target line is the max-depth branch in the live CoR answer loop.
        import linecache

        text = linecache.getline(source_line, frame.f_lineno).strip()
        if text != "if current_depth >= self.depth:":
            return self.trace
        state = loc.get("current_state")
        path = list(getattr(state, "path", []) or [])
        stack = loc.get("stack")
        serialized_stack = []
        for item in getattr(stack, "items", []) or []:
            serialized_stack.append({
                "path": [_node_payload(node) for node in getattr(item, "path", [])],
                "depth": getattr(item, "depth", None),
            })
        chain = [_node_payload(node) for node in path]
        prompt_history = deepcopy(loc.get("prompt_history", []))
        sparql_history = deepcopy(loc.get("sparql_history", []))
        method_events = deepcopy(self.method_events or [])
        method_events = [e for e in method_events if len(e.get("relation_chain", [])) <= self.depth]
        kg_cache = [deepcopy(item) for event_record in method_events for item in event_record.get("kg_cache", [])]
        observed_ids = set()
        for item in [*path, *(node for pending in (getattr(stack, "items", []) or []) for node in getattr(pending, "path", []))]:
            payload = _node_payload(item)
            if payload.get("kind") == "entity" and payload.get("id"):
                observed_ids.add(payload["id"])
        for event in sparql_history:
            observed_ids.update(str(value) for value in event.get("candidate_entity_ids", []) if value)
        observed_ids.update(str(value) for value in loc.get("entity_ids", []) if value)
        self.snapshots.append({
            "question_id": self.question_id,
            "repeat": self.repeat,
            "boundary_depth": self.depth,
            "prefix_replay_exact": False,
            "route": self.route,
            "model_settings": deepcopy(self.model_settings),
            "decisions": prompt_history,
            "kg_cache": kg_cache,
            "method_events": method_events,
            "sparql_history": sparql_history,
            "current_path": chain,
            "dfs_stack": serialized_stack,
            "active_binding_rows": None,
            "frontier_ids": list(loc.get("entity_ids") or []),
            "visited_node_ids": sorted(observed_ids),
            "selected_relation_id": chain[-1].get("id") if chain else None,
            "relation_direction": chain[-1].get("direction") if chain else None,
            "candidate_relation_ids_before_prune": None,
            "candidate_relation_ids_after_prune": None,
            "answer_source": "OTHER",
            "filter_input_ids": None,
            "filter_output_ids": None,
            "final_answer_ids": [],
            "log_completeness": "INCOMPLETE_INTERMEDIATE_BINDINGS_AND_PREFIX_CACHE",
            "evidence_pointer": self.evidence_pointer,
        })
        return self.trace


def _node_payload(node: Any) -> dict[str, Any]:
    name = type(node).__name__
    if name == "Entity":
        return {"kind": "entity", "id": str(getattr(node, "id", "")), "label": str(getattr(node, "name", ""))}
    if name == "Relation":
        return {"kind": "relation", "id": str(getattr(node, "id", "")),
                "direction": "OUTGOING" if bool(getattr(node, "left", False)) else "INCOMING"}
    return {"kind": name, "id": str(getattr(node, "id", ""))}


def run_with_boundary_capture(call: Callable[[], Any], *, question_id: str, repeat: int, depth: int,
                              evidence_pointer: str, model_settings: dict[str, Any] | None = None,
                              route: str = "cor", method_events: list[dict[str, Any]] | None = None) -> tuple[Any, list[dict[str, Any]]]:
    """Run a supplied unchanged answer call with reversible line tracing."""
    import sys

    previous = sys.gettrace()
    if previous is not None:
        raise RuntimeError("refusing to replace an existing Python trace hook")
    capture = BoundaryCapture(question_id, repeat, depth, evidence_pointer, [], deepcopy(model_settings), route, method_events)
    try:
        sys.settrace(capture.trace)
        result = call()
    finally:
        sys.settrace(previous)
    return result, capture.snapshots


def run_instrumented_answer(agent: Any, question: str, topic_entities: Sequence[Any], *, question_id: str,
                            repeat: int, depth: int, evidence_pointer: str,
                            model_settings: dict[str, Any]) -> dict[str, Any]:
    """Run one supplied CoRAgent locally with reversible method and stack capture."""
    prompt_history: list[dict[str, Any]] = []
    sparql_history: list[dict[str, Any]] = []
    step_history: list[dict[str, Any]] = []
    method_events: list[dict[str, Any]] = []
    with capture_cor_method_outputs(agent, method_events):
        result, snapshots = run_with_boundary_capture(
            lambda: agent.answer(question, list(topic_entities), prompt_history=prompt_history,
                                 sparql_history=sparql_history, step_history=step_history),
            question_id=question_id, repeat=repeat, depth=depth, evidence_pointer=evidence_pointer,
            model_settings=model_settings, method_events=method_events,
        )
    return {
        "result": result,
        "prompt_history": prompt_history,
        "sparql_history": sparql_history,
        "step_history": step_history,
        "method_events": method_events,
        "boundary_snapshots": snapshots,
    }
