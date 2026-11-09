#!/usr/bin/env python3
"""Bounded read-only oracle probe: append one missed gold edge to a reached prefix."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_cor_baseline_experiment import load_safe_project_env  # noqa: E402
from chain_of_relations.kg_backend import get_backend  # noqa: E402
from chain_of_relations.schema import Entity, Relation  # noqa: E402
from chain_of_relations.tools.entity_search import _build_entities_extract  # noqa: E402

WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
RESULTS = WORK / "RESULTS"
SOURCE_RUNS = RESULTS / "iter_016_instrumented_tgar_rerun/runs_with_state_lineage.jsonl"
SEMANTIC_PATHS = RESULTS / "iter_010_semantic_archived_path_audit/semantic_archived_path_alignment.csv"
MANIFEST = ROOT / "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256/manifest_frozen.json"
GOLD = ROOT / "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256/evaluation_gold.jsonl"
OUT = RESULTS / "iter_017_oracle_one_edge_continuation"
QUERY_CAP = 7
TIMEOUT_SECONDS = 20
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def best_reached_prefix(run: dict, expected: list[list[str]]) -> int:
    observed: list[list[tuple[str, str]]] = []
    for event in run.get("state_lineage", []):
        chain = [(str(x.get("id", "")), str(x.get("direction", ""))) for x in event.get("relation_chain", [])]
        observed.append(chain)
        if event.get("stage") == "relation_prune":
            observed.extend(chain + [(str(x.get("id", "")), str(x.get("direction", "")))]
                            for x in event.get("selected_relations", []))
    goal = [(str(edge[0]), str(edge[1])) for edge in expected]
    return max((sum(1 for a, b in zip(path, goal) if a == b) for path in observed), default=0)


def select_cases() -> tuple[list[dict], dict[str, dict]]:
    with SEMANTIC_PATHS.open(newline="", encoding="utf-8") as handle:
        semantic = {row["question_key"]: row for row in csv.DictReader(handle)}
    runs = read_jsonl(SOURCE_RUNS)
    chosen = []
    for run in runs:
        if run.get("cohort_role_scoring_only") != "TGAR_FAILURE" or int(run.get("hits_at_1", 0)):
            continue
        if run.get("gold_candidate_diagnostic_scoring_only", {}).get("candidate_presence_status") != "NEVER_SEEN":
            continue
        row = semantic[run["question_key"]]
        prefix_len, gold_len = int(row["max_archived_semantic_gold_prefix"]), int(row["gold_path_length"])
        if not (0 < prefix_len < gold_len):
            continue
        event = json.loads(row["best_archived_path_event"])
        prefix = event.get("path_sequence", [])[:prefix_len]
        if len(prefix) != prefix_len or best_reached_prefix(run, prefix) < prefix_len:
            continue
        next_edge = json.loads(row["next_unreached_gold_edge"])
        chosen.append({"run": run, "semantic": row, "prefix": prefix, "next_edge": next_edge})
    chosen.sort(key=lambda item: item["run"]["question_key"])
    if len(chosen) != QUERY_CAP:
        raise RuntimeError(f"Expected exactly {QUERY_CAP} eligible cases, found {len(chosen)}")
    return chosen, semantic


def run_query(endpoint: str, query: str, backend: object) -> dict:
    request = Request(
        endpoint,
        data=urlencode({"query": query, "format": "json"}).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/sparql-results+json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            status = response.status
            body = response.read(MAX_RESPONSE_BYTES + 1)
        elapsed = time.perf_counter() - started
        if len(body) > MAX_RESPONSE_BYTES:
            return {"status": "RESPONSE_LIMIT_EXCEEDED", "http_status": status, "elapsed_seconds": elapsed, "candidate_ids": []}
        payload = json.loads(body.decode("utf-8"))
        bindings = payload.get("results", {}).get("bindings", [])
        ids = []
        for binding in bindings:
            target = binding.get("targetEntity", {})
            if target.get("type") == "uri":
                entity_id = backend.normalize_uri(target.get("value", ""))
                if backend.is_entity_id(entity_id):
                    ids.append(entity_id)
        return {"status": "HTTP_OK", "http_status": status, "elapsed_seconds": elapsed,
                "candidate_ids": list(dict.fromkeys(ids)), "raw_binding_count": len(bindings)}
    except HTTPError as exc:
        return {"status": "HTTP_ERROR", "http_status": exc.code,
                "elapsed_seconds": time.perf_counter() - started, "candidate_ids": [], "error": str(exc)[:300]}
    except (TimeoutError, URLError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"status": "QUERY_ERROR", "http_status": None,
                "elapsed_seconds": time.perf_counter() - started, "candidate_ids": [],
                "error": f"{type(exc).__name__}: {exc}"[:300]}


def main() -> None:
    load_safe_project_env()
    endpoint = os.environ.get("FREEBASE_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql")
    backend = get_backend("freebase")
    selected, _ = select_cases()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_key = {f"{row['dataset']}::{row['question_id']}": row for row in manifest["cases"]}
    gold_by_key = {f"{row['dataset']}::{row['question_id']}": row for row in read_jsonl(GOLD)}
    if OUT.exists():
        if any(OUT.iterdir()):
            raise SystemExit(f"Output already exists and is non-empty: {OUT}")
        OUT.rmdir()
    OUT.mkdir(parents=True)
    records = []
    scores = []
    for index, item in enumerate(selected, start=1):
        run, key = item["run"], item["run"]["question_key"]
        case = by_key[key]
        inference_entities = gold_by_key[key]
        root = Entity(id=str(inference_entities["topic_entities"][0]),
                      name=str(inference_entities["topic_entity_names"][0]))
        edges = item["prefix"] + [[item["next_edge"]["id"], item["next_edge"]["direction"]]]
        relation_chain = [Relation(id=str(rel_id), left=(direction == "OUTGOING")) for rel_id, direction in edges]
        extract, error = _build_entities_extract(root, relation_chain, backend)
        if error:
            raise RuntimeError(f"Could not build path for {key}: {error}")
        query = backend.build_sparql_entities(extract)
        result = run_query(endpoint, query, backend)
        gold_ids = set(map(str, gold_by_key[key].get("answer_ids", [])))
        scores.append({"question_key": key, "gold_ids_scoring_only": sorted(gold_ids),
                       "gold_answer_in_candidates": bool(gold_ids.intersection(result.get("candidate_ids", [])))})
        records.append({"index": index, "question_key": key,
                        "archived_semantic_prefix_length": len(item["prefix"]),
                        "one_edge_extension": {"id": item["next_edge"]["id"], "direction": item["next_edge"]["direction"]},
                        "relation_path": [{"id": r.id, "direction": "OUTGOING" if r.left else "INCOMING"} for r in relation_chain],
                        "query": query, "query_sha256": hashlib.sha256(query.encode()).hexdigest(), **result})
        print(json.dumps({"query": index, "max_queries": QUERY_CAP, "question_key": key,
                          "status": result["status"], "elapsed_seconds": round(result["elapsed_seconds"], 2),
                          "candidate_count": len(result.get("candidate_ids", []))}), flush=True)

    write_jsonl(OUT / "query_results.jsonl", records)
    write_jsonl(OUT / "scoring_gold_only.jsonl", scores)
    success = [r for r in records if r.get("status") == "HTTP_OK"]
    found = sum(s["gold_answer_in_candidates"] for s in scores)
    input_hashes = {str(path.relative_to(ROOT)): sha_file(path) for path in (SOURCE_RUNS, SEMANTIC_PATHS, MANIFEST, GOLD)}
    results = {"experiment": "single oracle gold-edge continuation from reached semantic prefix",
               "status": "COMPLETED" if len(success) == QUERY_CAP else "INCOMPLETE_QUERY_ERRORS",
               "cases_selected": QUERY_CAP, "queries_attempted": len(records), "queries_http_ok": len(success),
               "gold_answer_present_after_one_edge_extension": found,
               "oracle_inference": True, "gold_relations_used_for_query": True,
               "gold_ids_used_only_for_post_query_scoring": True,
               "method_changed": False, "llm_calls": 0, "kg_queries": len(records),
               "query_cap": QUERY_CAP, "timeout_seconds_per_query": TIMEOUT_SECONDS,
               "max_response_bytes_per_query": MAX_RESPONSE_BYTES,
               "endpoint_host": endpoint.split("/")[2] if "://" in endpoint else endpoint,
               "input_hashes": input_hashes,
               "interpretation_limit": "A positive answer candidate shows that appending this oracle edge to the reached semantic prefix retrieves the gold ID through the unchanged CoR query builder. It does not prove the ordinary router should have selected it or validate an inference-time intervention."}
    write_json(OUT / "results.json", results)
    manifest_hashes = {path.name: sha_file(path) for path in sorted(OUT.iterdir()) if path.is_file()}
    manifest_hashes["runner_source_sha256"] = sha_file(Path(__file__))
    write_json(OUT / "artifact_hashes.json", manifest_hashes)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
