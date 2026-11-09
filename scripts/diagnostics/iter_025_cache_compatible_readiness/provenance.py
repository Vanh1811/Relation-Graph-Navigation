"""Rebuild Iterations 22–24 label-miss provenance from offline journals."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    from .coverage import REPO, WORK, sha256_file
except ImportError:  # unittest discovery with this directory as its top-level
    from coverage import REPO, WORK, sha256_file

IDS = ["m.04st848", "m.04stk8g", "m.04stpzt", "m.04strb6", "m.0hz9k44", "m.0hzdjww", "m.0ndvpvf"]
I22 = WORK / "RESULTS/iter_022_gated_matched_prefix"
I23 = WORK / "RESULTS/iter_023_label_coverage_and_prefix_readiness"
I24 = WORK / "RESULTS/iter_024_label_source_audit"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rows_for_query_journal(path: Path, kind: str) -> list[dict[str, Any]]:
    out = []
    for row in _jsonl(path):
        if row.get("status") != "ok":
            continue
        query = str(row.get("query", ""))
        if kind == "candidate" and re.search(r"SELECT\s+DISTINCT\s+\?targetEntity\b", query, re.I) and "VALUES ?entity" not in query:
            out.append(row)
        if kind == "label" and "VALUES ?entity" in query and "?targetEntity" in query:
            out.append(row)
    return out


def _prompt_strings(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "prompt" and isinstance(child, str):
                yield child
            else:
                yield from _prompt_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _prompt_strings(child)


def build_label_audit() -> dict[str, Any]:
    cache_doc = json.loads((I22 / "LOCAL_LABEL_CACHE.json").read_text(encoding="utf-8"))
    cache_ids = {entry["entity_id"] for entry in cache_doc["labels"]}
    observations: dict[str, list[dict[str, Any]]] = {entity_id: [] for entity_id in IDS}
    for iteration, directory in ((22, I22), (23, I23)):
        journal = directory / "SPARQL_QUERY_JOURNAL.jsonl"
        for event in _rows_for_query_journal(journal, "candidate"):
            query = str(event.get("query", ""))
            for row_index, bindings in enumerate(event.get("rows", []) or []):
                binding = bindings.get("targetEntity") if isinstance(bindings, dict) else None
                value = binding.get("value", "") if isinstance(binding, dict) else ""
                entity_id = value.rsplit("/", 1)[-1] if isinstance(value, str) else ""
                if entity_id in observations:
                    observations[entity_id].append({
                        "iteration": iteration,
                        "question_id": event.get("case_id"),
                        "query_index": event.get("query_index"),
                        "query_sha256": event.get("query_sha256"),
                        "query_predicate_context": query.strip().replace("\n", " "),
                        "binding_name": "targetEntity",
                        "binding_position_zero_based": row_index,
                        "query_status": event.get("status"),
                    })

    label_lookup: dict[str, list[dict[str, Any]]] = {entity_id: [] for entity_id in IDS}
    for iteration, directory in ((23, I23),):
        journal = directory / "SPARQL_QUERY_JOURNAL.jsonl"
        for event in _rows_for_query_journal(journal, "label"):
            query = str(event.get("query", ""))
            ids = re.findall(r"ns:(m\.[A-Za-z0-9_]+|g\.[A-Za-z0-9_]+)", query.split("}", 1)[0])
            for entity_id in IDS:
                if entity_id not in ids:
                    continue
                rows = event.get("rows", []) or []
                matching = [r for r in rows if isinstance(r, dict) and r.get("entity", {}).get("value", "").endswith("/" + entity_id)]
                has_value = any("targetEntity" in r and r["targetEntity"].get("value") for r in matching)
                label_lookup[entity_id].append({
                    "iteration": iteration,
                    "case_id": event.get("case_id"),
                    "query_index": event.get("query_index"),
                    "query_sha256": event.get("query_sha256"),
                    "status": event.get("status"),
                    "entity_rows": len(matching),
                    "targetEntity_value_bound": has_value,
                })

    prompt_mentions: dict[str, list[dict[str, Any]]] = {entity_id: [] for entity_id in IDS}
    for iteration, directory in ((22, I22), (23, I23)):
        traces = _jsonl(directory / "RAW_CASE_TRACES.jsonl")
        for row in traces:
            case_id = row.get("case_id") or row.get("question_id") or row.get("question_key")
            for prompt in _prompt_strings(row):
                for entity_id in IDS:
                    if entity_id in prompt:
                        prompt_mentions[entity_id].append({"iteration": iteration, "case_id": case_id,
                                                           "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})

    scoring_ids: set[str] = set()
    for directory in (I22, I23):
        sidecar = json.loads((directory / "GOLD_SCORING_ONLY.json").read_text(encoding="utf-8"))
        for values in sidecar.get("cases", {}).values():
            scoring_ids.update(str(value) for value in values)

    archive = WORK / "RESULTS/iter_019_depth4_holdout_preparation/control_continuation/paired_runs_cumulative.jsonl"
    archive_candidate_name_observations = set()
    for row in _jsonl(archive):
        for event in row.get("sparql_history", []) or []:
            if event.get("search_type") != "entity_search":
                continue
            for entity_id, name in zip(event.get("candidate_entity_ids", []) or [], event.get("candidate_entity_names", []) or []):
                if entity_id in IDS and isinstance(name, str) and name.strip():
                    archive_candidate_name_observations.add((entity_id, name))

    with (I24 / "LABEL_SOURCE_COVERAGE.csv").open(newline="", encoding="utf-8") as stream:
        import csv
        iter24_rows = list(csv.DictReader(stream))
    return {
        "ids": IDS,
        "iteration22_candidate_observations": {key: value for key, value in observations.items() if any(x["iteration"] == 22 for x in value)},
        "candidate_source_observations": observations,
        "iteration23_label_lookup_observations": label_lookup,
        "prompt_mentions": prompt_mentions,
        "frozen_cache_contains_ids": sorted(set(IDS) & cache_ids),
        "archived_inference_candidate_name_observations": sorted([list(x) for x in archive_candidate_name_observations]),
        "gold_scoring_id_overlap": sorted(set(IDS) & scoring_ids),
        "iter24_queried_predicates": ["type.object.name", "rdfs:label", "common.topic.alias", "type.object.key"],
        "iter24_all_four_predicates_zero_for_all_seven": len(iter24_rows) == 7 and all(
            row.get("type.object.name_observed") == "False"
            and row.get("rdfs:label_observed") == "False"
            and row.get("common.topic.alias_observed") == "False"
            and row.get("type.object.key_observed") == "False"
            for row in iter24_rows
        ),
        "sha256": {
            "iter22_query_journal": sha256_file(I22 / "SPARQL_QUERY_JOURNAL.jsonl"),
            "iter23_query_journal": sha256_file(I23 / "SPARQL_QUERY_JOURNAL.jsonl"),
            "iter22_cache": sha256_file(I22 / "LOCAL_LABEL_CACHE.json"),
            "iter24_label_source_csv": sha256_file(I24 / "LABEL_SOURCE_COVERAGE.csv"),
        },
    }


if __name__ == "__main__":
    print(json.dumps(build_label_audit(), indent=2, sort_keys=True))
