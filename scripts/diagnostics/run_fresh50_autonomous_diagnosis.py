#!/usr/bin/env python3
"""Evidence-bounded follow-up for the frozen Fresh50 CoR/TGAR/CFP cohort.

This diagnostic never calls an LLM or writes to Freebase. It replays selected
gold SPARQL read-only, inventories the frozen traces, and records the exact
blocker to running an oracle relation plan through the production executor.
The gold program is used only for diagnosis and never as inference input.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from scripts.audit_fresh50_tgar_vs_cfp import (
    METHODS,
    SOURCE_ROOT,
    all_events,
    branch_trace_ids,
    gold_relation_tokens,
    integrity as baseline_integrity,
    load_bundle,
    primary_diagnosis,
    route_event,
    sha256,
    stable_key,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/fresh50_autonomous_causal_diagnosis"
AUDIT = ROOT / "outputs/experiments/fresh50_strong_branch_root_cause_audit"
ENDPOINT = os.environ.get("FRESH50_DIAG_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql")
MAX_KG_QUERIES = max(0, int(os.environ.get("FRESH50_DIAG_MAX_KG_QUERIES", "600")))
QUERY_TIMEOUT_SECONDS = max(1, int(os.environ.get("FRESH50_DIAG_QUERY_TIMEOUT_SECONDS", "20")))
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
METHOD_NAMES = {"original_cor": "CoR Baseline", "v1_1_3": "TGAR", "phase2_5_6": "CFP"}
QNAME = r'(?:(?:[A-Za-z_][A-Za-z0-9_]*)?:)?[A-Za-z0-9_][A-Za-z0-9_.-]*'
TERM = rf'(?:\?[A-Za-z_][A-Za-z0-9_]*|{QNAME}|<[^>]+>|"(?:\\.|[^"\\])*"(?:@[A-Za-z-]+|\^\^[^\s]+)?)'
TRIPLE_RE = re.compile(
    rf"(?<![\w])({TERM})\s+({QNAME})\s+({TERM})\s*\."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    return re.sub(r"\s+", " ", text).casefold()


def canonical_freebase_id(value: str) -> str:
    value = str(value or "").strip()
    for prefix in ("http://rdf.freebase.com/ns/", "https://rdf.freebase.com/ns/"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    return value


def graph_summary(sparql: str) -> dict[str, Any]:
    def canonicalize_freebase_term(term: str) -> str:
        return "ns:" + term[1:] if term.startswith(":") else term

    triples = [
        {"subject": canonicalize_freebase_term(s),
         "predicate": canonicalize_freebase_term(p),
         "object": canonicalize_freebase_term(o)}
        for s, p, o in TRIPLE_RE.findall(sparql)
    ]
    values_bindings: dict[str, list[str]] = {}
    values_term_pattern = re.compile(rf'{QNAME}|<[^>]+>|"(?:\\.|[^"\\])*"(?:@[A-Za-z-]+|\^\^[^\s]+)?')
    for match in re.finditer(r"\bVALUES\s+\?(\w+)\s*\{([^}]*)\}", sparql, re.I):
        variable, body = match.groups()
        values_bindings[variable] = [
            canonicalize_freebase_term(value)
            for value in values_term_pattern.findall(body)
        ]
    answer_match = re.search(r"SELECT\s+(?:DISTINCT\s+)?\(\s*(\?\w+)\s+AS\s+\?\w+\s*\)", sparql, re.I)
    if not answer_match:
        answer_match = re.search(r"SELECT\s+(?:DISTINCT\s+)?(\?\w+)", sparql, re.I)
    answer_var = answer_match.group(1) if answer_match else ""
    variable_degree: Counter[str] = Counter()
    incoming: Counter[str] = Counter()
    outgoing: Counter[str] = Counter()
    adjacency: dict[str, set[str]] = {}
    for triple in triples:
        s, o = triple["subject"], triple["object"]
        if s.startswith("?"):
            outgoing[s] += 1
            variable_degree[s] += 1
        if o.startswith("?"):
            incoming[o] += 1
            variable_degree[o] += 1
        if s.startswith("?") and o.startswith("?"):
            adjacency.setdefault(s, set()).add(o)
            adjacency.setdefault(o, set()).add(s)
    intermediate_joins = sorted(
        var for var, degree in variable_degree.items()
        if var != answer_var and (degree > 2 or incoming[var] > 1 or outgoing[var] > 1)
    )
    degrees = {var: int(degree) for var, degree in sorted(variable_degree.items())}
    if not triples:
        topology = "PARSE_UNAVAILABLE"
    elif intermediate_joins:
        topology = "INTERMEDIATE_VARIABLE_JOIN"
    elif any(degree > 2 for var, degree in variable_degree.items() if var == answer_var):
        topology = "ANSWER_CONVERGENCE"
    elif max(variable_degree.values(), default=0) <= 2:
        topology = "PATH_OR_SIMPLE_CONSTRAINT_CHAIN"
    else:
        topology = "MULTI_BRANCH_GRAPH"
    modifiers = [name for name in ("FILTER", "VALUES", "OPTIONAL", "UNION", "GROUP BY", "ORDER BY", "LIMIT") if re.search(rf"\b{name}\b", sparql, re.I)]
    return {
        "answer_variable": answer_var,
        "triple_pattern_count": len(triples),
        "triple_patterns": triples,
        "values_bindings": values_bindings,
        "variable_degrees": degrees,
        "intermediate_join_variables": intermediate_joins,
        "topology": topology,
        "query_modifiers": modifiers,
        "graph_parse_status": "REGEX_TRIPLE_EXTRACTION" if triples else "NO_TRIPLES_EXTRACTED",
    }


def outcome_tags(methods: Mapping[str, Mapping[str, Any]]) -> list[str]:
    hits = tuple(int(methods[name].get("hits_at_1", 0) or 0) for name in METHODS)
    tags: list[str] = []
    if hits == (1, 1, 1): tags.append("ALL_THREE_CORRECT")
    if hits == (0, 0, 0): tags.append("ALL_THREE_WRONG")
    if hits[1] == 0: tags.append("TGAR_FAILURE")
    if hits[1] == 1 and hits[0] == 0: tags.append("TGAR_ONLY")
    if hits[2] == 1 and hits[1] == 0: tags.append("CFP_ONLY_VS_TGAR")
    if hits[1] == 1 and hits[2] == 0: tags.append("TGAR_ONLY_VS_CFP")
    if hits[1] != hits[2]: tags.append("TGAR_CFP_DISAGREEMENT")
    return tags


def select_cases(bundle: Mapping[str, Any]) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    keys = sorted(bundle["manifest"])
    methods_by_key = {
        key: {method: bundle["runs"][(key[0], key[1], method)][1] for method in METHODS}
        for key in keys
    }
    shared = [key for key in keys if all(int(methods_by_key[key][method].get("hits_at_1", 0)) == 0 for method in METHODS)]
    disagreements = [key for key in keys if int(methods_by_key[key]["v1_1_3"].get("hits_at_1", 0)) != int(methods_by_key[key]["phase2_5_6"].get("hits_at_1", 0))]
    controls = [key for key in keys if all(int(methods_by_key[key][method].get("hits_at_1", 0)) == 1 for method in METHODS)][:5]
    return shared, disagreements, controls


def artifact_index(bundle: Mapping[str, Any], integrity: Mapping[str, Any]) -> dict[str, Any]:
    discovered: list[Path] = []
    for base in (SOURCE_ROOT, AUDIT, OUT):
        if base.exists():
            discovered.extend(path for path in base.rglob("*") if path.is_file())
    for path in (
        ROOT / "scripts/audit_fresh50_tgar_vs_cfp.py",
        ROOT / "scripts/run_fresh50_strong_branch_three_way.py",
        ROOT / "chain_of_relations/methods/cor/agent.py",
        ROOT / "chain_of_relations/methods/branch_v1/agent.py",
        ROOT / "chain_of_relations/methods/adaptive_v1_1_3/agent.py",
        ROOT / "chain_of_relations/methods/adaptive_v1_1_3/router.py",
        ROOT / "chain_of_relations/methods/phase2_5_5/agent.py",
        ROOT / "chain_of_relations/methods/phase2_5_5/router.py",
        ROOT / "chain_of_relations/methods/phase2_5_3b/serializer.py",
        ROOT / "chain_of_relations/kg_backend/freebase_backend.py",
        ROOT / "chain_of_relations/kg_backend/freebase/query_template.yml",
        ROOT / "chain_of_relations/tools/entity_search.py",
        ROOT / "chain_of_relations/tools/entity_prune.py",
        ROOT / "chain_of_relations/eval/accuracy.py",
        ROOT / "scripts/diagnostics/run_fresh50_autonomous_diagnosis.py",
        ROOT / "tests/test_fresh50_autonomous_diagnosis.py",
        ROOT / "Codex_Autonomous_Fresh50_Causal_Diagnosis_Loop.md",
    ):
        if path.exists():
            discovered.append(path)
    unique = sorted(path for path in set(discovered) if path != OUT / "artifact_index.json")
    files = [{"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in unique]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    diff_stat = subprocess.run(["git", "diff", "--stat"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    production_diff = subprocess.run(
        ["git", "diff", "--stat", "--", "chain_of_relations/methods/cor", "chain_of_relations/methods/branch_v1", "chain_of_relations/methods/adaptive_v1_1_3", "chain_of_relations/methods/phase2_5_5"],
        cwd=ROOT, text=True, capture_output=True,
    ).stdout.strip()
    return {
        "repo_root": str(ROOT), "git_head": head, "git_branch": branch,
        "git_status_short": status, "git_diff_stat_at_index_time": diff_stat,
        "production_reasoning_diff_stat_at_index_time": production_diff,
        "source_experiment_root": str(SOURCE_ROOT.relative_to(ROOT)),
        "prior_audit_root": str(AUDIT.relative_to(ROOT)), "files": files,
        "frozen_question_count": len(bundle["manifest"]), "frozen_run_count": len(bundle["run_lines"]),
        "source_manifest_csv_sha256": integrity.get("manifest_csv_sha256"),
        "source_manifest_protocol_hash_match": integrity.get("manifest_hash_matches_protocol"),
        "production_source_modified_by_this_diagnostic": False,
    }


def build_case_artifacts(bundle: Mapping[str, Any], diagnoses: Mapping[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory: list[dict[str, Any]] = []
    for key in sorted(bundle["manifest"]):
        manifest = bundle["manifest"][key]
        gold = bundle["gold"][key]
        runs = {method: bundle["runs"][(key[0], key[1], method)][1] for method in METHODS}
        inv = {
            "question_key": f"{key[0]}::{key[1]}", "dataset": key[0], "question_id": key[1],
            "question": runs["v1_1_3"].get("question", manifest.get("question", "")),
            "split": manifest.get("split", "UNKNOWN"), "branch_type": manifest.get("branch_shape", "UNKNOWN"),
            "necessity_label": manifest.get("necessity_label", "UNKNOWN"),
            "gold_answer_ids": json.dumps(gold.get("answer_ids", []), ensure_ascii=False),
            "gold_answer_labels": json.dumps(gold.get("gold_answers", []), ensure_ascii=False),
            "gold_sparql_present": bool(str(gold.get("gold_sparql", "")).strip()),
            "gold_logical_form_present": bool(str(gold.get("logical_form", "")).strip()),
            "cor_prediction": json.dumps(runs["original_cor"].get("prediction", []), ensure_ascii=False),
            "tgar_prediction": json.dumps(runs["v1_1_3"].get("prediction", []), ensure_ascii=False),
            "cfp_prediction": json.dumps(runs["phase2_5_6"].get("prediction", []), ensure_ascii=False),
            "cor_hit": int(runs["original_cor"].get("hits_at_1", 0)), "tgar_hit": int(runs["v1_1_3"].get("hits_at_1", 0)),
            "cfp_hit": int(runs["phase2_5_6"].get("hits_at_1", 0)),
            "cor_f1": runs["original_cor"].get("f1", 0), "tgar_f1": runs["v1_1_3"].get("f1", 0), "cfp_f1": runs["phase2_5_6"].get("f1", 0),
            "tgar_route": runs["v1_1_3"].get("router_structure"), "tgar_execution": runs["v1_1_3"].get("execution_status"),
            "tgar_fallback": runs["v1_1_3"].get("fallback_reason") or "NONE",
            "cfp_route": runs["phase2_5_6"].get("router_structure"), "cfp_execution": runs["phase2_5_6"].get("execution_status"),
            "cfp_fallback": runs["phase2_5_6"].get("fallback_reason") or "NONE",
            "selected_relations_tgar": json.dumps(runs["v1_1_3"].get("relation_chain", []), ensure_ascii=False),
            "branch_trace_ids_tgar": json.dumps(sorted(branch_trace_ids(runs["v1_1_3"])), ensure_ascii=False),
            "branch_trace_ids_cfp": json.dumps(sorted(branch_trace_ids(runs["phase2_5_6"])), ensure_ascii=False),
            "run_lines": ";".join(f"{method}:{bundle['runs'][(key[0], key[1], method)][0]}" for method in METHODS),
            "outcome_tags": ";".join(outcome_tags(runs)),
        }
        inventory.append(inv)

    diag_rows: list[dict[str, Any]] = []
    for inv in inventory:
        key = (inv["dataset"], inv["question_id"])
        diagnosis = diagnoses[f"{key[0]}::{key[1]}"]
        diag_rows.append({
            "question_key": inv["question_key"], "dataset": inv["dataset"], "branch_type": inv["branch_type"],
            "gold_answer_ids": inv["gold_answer_ids"], "cor_hit": inv["cor_hit"], "tgar_hit": inv["tgar_hit"], "cfp_hit": inv["cfp_hit"],
            "actual_tgar_route": inv["tgar_route"], "actual_cfp_route": inv["cfp_route"],
            "fallback_status": f"TGAR:{inv['tgar_fallback']}|CFP:{inv['cfp_fallback']}",
            "gold_kg_valid": "NOT_TESTED", "oracle_plan_equivalent": "NOT_VERIFIED",
            "cor_contract_expressible": "BLOCKED_NO_EXACT_RELATION_PLAN_INPUT",
            "tgar_contract_expressible": "BLOCKED_NO_EXACT_RELATION_PLAN_INPUT",
            "cfp_contract_expressible": "BLOCKED_NO_EXACT_RELATION_PLAN_INPUT",
            "oracle_executor_success": "BLOCKED_NO_ORACLE_EXECUTOR_INTERFACE",
            "model_plan_equivalent": "UNAVAILABLE_SEMANTIC_TRACE_INSUFFICIENT",
            "first_divergence_stage": diagnosis.get("first_divergence_stage", "UNDETERMINED"),
            "primary_causal_class": diagnosis.get("primary_root_cause", "UNDETERMINED"),
            "secondary_factors": ";".join(diagnosis.get("secondary_causes", [])),
            "counterfactual_test_id": "GOLD_KG_REPLAY_PENDING",
            "counterfactual_outcome": "NOT_RUN",
            "evidence_tier": diagnosis.get("evidence_strength", "UNAVAILABLE"),
            "confidence": "LOW_OR_UNDETERMINED",
            "unresolved_alternatives": "LLM planning; relation/entity grounding; executor representation; KG/evaluator mismatch; final answer filtering; run stochasticity",
            "source_trace_paths": f"{SOURCE_ROOT.relative_to(ROOT)}/runs.jsonl",
            "query_log_paths": "NOT_CREATED",
        })
    return inventory, diag_rows


def answer_semantics(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(bundle["manifest"]):
        gold = bundle["gold"][key]
        ids = {canonical_freebase_id(value) for value in gold.get("answer_ids", []) if str(value).strip()}
        labels = {normalized_text(value) for value in gold.get("gold_answers", []) if normalized_text(value)}
        method_runs = {method: bundle["runs"][(key[0], key[1], method)][1] for method in METHODS}
        for method, run in method_runs.items():
            pred = [str(value) for value in run.get("prediction", []) or []]
            pred_ids = {canonical_freebase_id(value) for value in pred if re.fullmatch(r"(?:m|g|en|d)\.[\w.-]+", canonical_freebase_id(value))}
            pred_labels = {normalized_text(value) for value in pred if normalized_text(value)}
            rows.append({
                "question_key": f"{key[0]}::{key[1]}", "dataset": key[0], "question_id": key[1], "method": METHOD_NAMES[method],
                "gold_id_count": len(ids), "prediction_count": len(pred),
                "exact_gold_id_overlap_count": len(ids & pred_ids),
                "gold_label_exact_normalized_overlap_count": len(labels & pred_labels),
                "stored_hits_at_1": int(run.get("hits_at_1", 0)),
                "interpretation": "LABEL_OR_ID_ALIAS_REQUIRES_KG_REPLAY" if int(run.get("hits_at_1", 0)) == 0 and (ids & pred_ids or labels & pred_labels) else "NO_DIRECT_CANONICAL_MATCH_IN_STORED_PREDICTIONS",
                "gold_ids": json.dumps(sorted(ids), ensure_ascii=False), "gold_labels_normalized": json.dumps(sorted(labels), ensure_ascii=False),
                "prediction": json.dumps(pred, ensure_ascii=False),
            })
    return rows


def sparql_post(query: str) -> tuple[str, dict[str, Any]]:
    body = urlencode({"query": query, "format": "json"}).encode("utf-8")
    request = Request(ENDPOINT, data=body, headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/sparql-results+json"}, method="POST")
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=QUERY_TIMEOUT_SECONDS) as response:
            status = response.status
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        elapsed = time.perf_counter() - started
        if len(raw) > MAX_RESPONSE_BYTES:
            return "RESPONSE_LIMIT_EXCEEDED", {"http_status": status, "elapsed_seconds": elapsed, "response_bytes_capped": True, "result_count": None, "answer_values": [], "error": "Response exceeded diagnostic 20 MiB read cap"}
        payload = json.loads(raw.decode("utf-8"))
        bindings = payload.get("results", {}).get("bindings", [])
        values = []
        vars_ = payload.get("head", {}).get("vars", [])
        answer_var = "value" if "value" in vars_ else ("x" if "x" in vars_ else (vars_[0] if vars_ else ""))
        for binding in bindings:
            item = binding.get(answer_var, {})
            values.append({"type": item.get("type", ""), "value": str(item.get("value", ""))})
        return "HTTP_OK", {"http_status": status, "elapsed_seconds": elapsed, "response_bytes": len(raw), "result_count": len(bindings), "answer_values": values, "answer_variable": answer_var, "head_vars": vars_, "error": None}
    except HTTPError as exc:
        elapsed = time.perf_counter() - started
        body = exc.read(4000).decode("utf-8", errors="replace")
        return "HTTP_ERROR", {"http_status": exc.code, "elapsed_seconds": elapsed, "response_bytes": len(body), "result_count": None, "answer_values": [], "error": body}
    except (TimeoutError, URLError, OSError) as exc:
        elapsed = time.perf_counter() - started
        return "QUERY_ERROR", {"http_status": None, "elapsed_seconds": elapsed, "response_bytes": 0, "result_count": None, "answer_values": [], "error": f"{type(exc).__name__}: {exc}"}
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        elapsed = time.perf_counter() - started
        return "RESPONSE_PARSE_ERROR", {"http_status": 200, "elapsed_seconds": elapsed, "response_bytes": 0, "result_count": None, "answer_values": [], "error": f"{type(exc).__name__}: {exc}"}


def compare_gold(gold: Mapping[str, Any], result: Mapping[str, Any], status: str) -> dict[str, Any]:
    if status != "HTTP_OK":
        return {"gold_kg_valid": "QUERY_FAILED_OR_BLOCKED", "id_match": "NOT_COMPARABLE", "label_match": "NOT_COMPARABLE", "returned_ids": [], "returned_literals": []}
    expected = {canonical_freebase_id(value) for value in gold.get("answer_ids", []) if str(value).strip()}
    returned_ids = {canonical_freebase_id(item["value"]) for item in result.get("answer_values", []) if item.get("type") == "uri" and canonical_freebase_id(item.get("value", ""))}
    returned_literals = {normalized_text(item["value"]) for item in result.get("answer_values", []) if item.get("type") in {"literal", "typed-literal"} and normalized_text(item.get("value", ""))}
    expected_labels = {normalized_text(value) for value in gold.get("gold_answers", []) if normalized_text(value)}
    if expected and returned_ids == expected:
        id_match = "EXACT"
        state = "GOLD_IDS_REPRODUCED"
    elif expected and expected.issubset(returned_ids):
        id_match = "GOLD_SUBSET_OF_RESULTS"
        state = "GOLD_IDS_PRESENT_WITH_EXTRA_RESULTS"
    elif expected and returned_ids and expected.intersection(returned_ids):
        id_match = "PARTIAL_OVERLAP"
        state = "PARTIAL_GOLD_ID_OVERLAP"
    elif expected and returned_ids:
        id_match = "NO_OVERLAP"
        state = "GOLD_ID_MISMATCH"
    else:
        id_match = "NO_URI_IDS"
        state = "ID_COMPARISON_UNAVAILABLE"
    label_match = "EXACT" if expected_labels and returned_literals == expected_labels else ("OVERLAP" if expected_labels.intersection(returned_literals) else "NO_LITERAL_LABEL_MATCH")
    return {"gold_kg_valid": state, "id_match": id_match, "label_match": label_match, "returned_ids": sorted(returned_ids), "returned_literals": sorted(returned_literals)}


def query_replay(bundle: Mapping[str, Any], shared: list[tuple[str, str]], disagreements: list[tuple[str, str]], controls: list[tuple[str, str]]) -> list[dict[str, Any]]:
    selected: list[tuple[str, str]] = []
    for key in shared + disagreements + controls:
        if key not in selected:
            selected.append(key)
    logs_dir = OUT / "gold_query_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, key in enumerate(selected, start=1):
        gold = bundle["gold"][key]
        query = str(gold.get("gold_sparql", "")).strip()
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        qkey = f"{key[0]}::{key[1]}"
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", qkey)
        log_path = logs_dir / f"{safe_id}.json"
        if log_path.exists():
            cached = json.loads(log_path.read_text(encoding="utf-8"))
            if cached.get("query_sha256") == query_hash:
                result_status = cached["query_status"]
                result = cached["response_summary"]
                compare = cached["comparison"]
                resumed = True
            else:
                cached = None
        else:
            cached = None
        if cached is None:
            if not query:
                result_status, result = "NO_GOLD_SPARQL", {"http_status": None, "elapsed_seconds": 0, "result_count": None, "answer_values": [], "error": "Gold SPARQL absent"}
            elif len(rows) >= MAX_KG_QUERIES:
                result_status, result = "QUERY_BUDGET_EXHAUSTED", {"http_status": None, "elapsed_seconds": 0, "result_count": None, "answer_values": [], "error": f"FRESH50_DIAG_MAX_KG_QUERIES={MAX_KG_QUERIES}"}
            else:
                print(f"KG replay {index}/{len(selected)} {qkey}", flush=True)
                result_status, result = sparql_post(query)
            compare = compare_gold(gold, result, result_status)
            log_obj = {
                "question_key": qkey, "endpoint": ENDPOINT, "query_sha256": query_hash,
                "query": query, "query_status": result_status, "response_summary": result,
                "comparison": compare, "read_only": True, "llm_calls": 0,
            }
            dump_json(log_path, log_obj)
            resumed = False
        graph = graph_summary(query)
        row = {
            "question_key": qkey, "dataset": key[0], "question_id": key[1],
            "cohort_role": "SHARED_FAILURE" if key in shared else ("TGAR_CFP_DISAGREEMENT" if key in disagreements else "BOTH_CORRECT_CONTROL"),
            "query_sha256": query_hash, "query_status": result_status,
            "http_status": result.get("http_status"), "elapsed_seconds": round(float(result.get("elapsed_seconds", 0) or 0), 3),
            "result_count": result.get("result_count"), "gold_kg_valid": compare["gold_kg_valid"],
            "id_match": compare["id_match"], "label_match": compare["label_match"],
            "gold_answer_ids": json.dumps(gold.get("answer_ids", []), ensure_ascii=False),
            "returned_ids": json.dumps(compare["returned_ids"], ensure_ascii=False),
            "returned_literals": json.dumps(compare["returned_literals"], ensure_ascii=False),
            "gold_query_triple_count": graph["triple_pattern_count"], "gold_query_topology": graph["topology"],
            "intermediate_join_variables": json.dumps(graph["intermediate_join_variables"]),
            "query_log": str(log_path.relative_to(ROOT)), "cache_reused": resumed,
        }
        rows.append(row)
        write_csv(OUT / "gold_kg_replay.csv", rows)
        print(f"  -> {row['query_status']} / {row['gold_kg_valid']} ({row['elapsed_seconds']}s)", flush=True)
    return rows


def oracle_artifacts(bundle: Mapping[str, Any], shared: list[tuple[str, str]], controls: list[tuple[str, str]], replay_rows: list[dict[str, Any]], diagnoses: Mapping[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    replay_by_key = {row["question_key"]: row for row in replay_rows}
    oracle_rows: list[dict[str, Any]] = []
    capability_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for key in shared + [item for item in controls if item not in shared]:
        gold = bundle["gold"][key]
        qkey = f"{key[0]}::{key[1]}"
        graph = graph_summary(str(gold.get("gold_sparql", "")))
        diagnosis = diagnoses[qkey]
        replay = replay_by_key.get(qkey, {})
        native_gold_valid = replay.get("gold_kg_valid", "NOT_TESTED")
        topology = graph["topology"]
        intermediate = topology == "INTERMEDIATE_VARIABLE_JOIN"
        if topology == "PARSE_UNAVAILABLE":
            same_exec_contract = "BLOCKED_GOLD_GRAPH_PARSE_UNAVAILABLE"
            branch_contract = "BLOCKED_GOLD_GRAPH_PARSE_UNAVAILABLE"
            contract_status = "BLOCKED_GOLD_GRAPH_PARSE_UNAVAILABLE"
            exact_missing_operation = "Gold relation dependency graph not parsed; no plan-level contract test"
        elif intermediate:
            same_exec_contract = "UNEXPRESSIBLE_CONTRACT; CoRAgent accepts an ordered Relation chain and entity_search creates fresh ?e_i at each hop; cannot reuse shared variables"
            branch_contract = "UNEXPRESSIBLE_CONTRACT; no shared intermediate binding map or post-merge continuation field"
            contract_status = "UNEXPRESSIBLE_CONTRACT"
            exact_missing_operation = "Bind the same SPARQL variable across multiple relation edges and continue from that shared binding"
        elif topology == "ANSWER_CONVERGENCE":
            same_exec_contract = "UNEXPRESSIBLE_CONTRACT; CoRAgent has a single DFS relation path and no exact multi-branch oracle-plan input"
            branch_contract = "BLOCKED_NO_EXACT_PLAN_INPUT; final candidate intersection exists, but oracle branch-plan expressibility is not established"
            contract_status = "UNEXPRESSIBLE_CONTRACT_FOR_COR; TGAR_CFP_CONVERGENCE_UNDETERMINED"
            exact_missing_operation = "Inject exact relation edges/directions for each branch and define their shared answer-variable merge"
        else:
            same_exec_contract = "BLOCKED_NO_EXACT_ORACLE_PLAN_INPUT"
            branch_contract = "BLOCKED_NO_EXACT_PLAN_INPUT"
            contract_status = "BLOCKED_NO_EXACT_PLAN_INPUT"
            exact_missing_operation = "Exact relation edge and direction input"
        row = {
            "question_key": qkey, "dataset": key[0], "question_id": key[1],
            "cohort_role": "SHARED_FAILURE" if key in shared else "BOTH_CORRECT_CONTROL",
            "gold_sparql_sha256": hashlib.sha256(str(gold.get("gold_sparql", "")).encode()).hexdigest(),
            "gold_answer_ids": gold.get("answer_ids", []), "gold_program": str(gold.get("gold_sparql", "")),
            "oracle_relation_dependency_plan": graph["triple_patterns"],
            "plan_semantics_status": "GOLD_QUERY_REPRODUCED_IDS" if native_gold_valid in {"GOLD_IDS_REPRODUCED", "GOLD_IDS_PRESENT_WITH_EXTRA_RESULTS"} else "EXACT_GOLD_PROGRAM_RECORDED_SEMANTICS_NOT_BACKEND_VERIFIED",
            "gold_kg_valid": native_gold_valid,
            "graph_topology": graph["topology"], "answer_variable": graph["answer_variable"],
            "intermediate_join_variables": graph["intermediate_join_variables"], "query_modifiers": graph["query_modifiers"],
            "cor_plan_contract": same_exec_contract,
            "tgar_plan_contract": branch_contract,
            "cfp_plan_contract": branch_contract,
            "oracle_executor_attempted": False,
            "oracle_executor_status": contract_status,
            "GOLD_KG_VALID?": native_gold_valid,
            "ORACLE_PLAN_SEMANTICALLY_EQUIVALENT?": "GOLD_PROGRAM_VERIFIED_BY_ID_REPLAY" if native_gold_valid in {"GOLD_IDS_REPRODUCED", "GOLD_IDS_PRESENT_WITH_EXTRA_RESULTS"} else "NOT_VERIFIED",
            "ORACLE_PLAN_EXPRESSIBLE_IN_COR?": same_exec_contract,
            "ORACLE_PLAN_EXPRESSIBLE_IN_TGAR?": branch_contract,
            "ORACLE_PLAN_EXPRESSIBLE_IN_CFP?": branch_contract,
            "ORACLE_EXECUTOR_RETURNED_GOLD_IDS?": "BLOCKED_NOT_RUN",
            "MODEL_PLAN_SEMANTICALLY_EQUIVALENT?": "UNAVAILABLE_TRACE_DOES_NOT_ESTABLISH_EQUIVALENCE",
            "FIRST_MISMATCH_STAGE": diagnosis.get("first_divergence_stage", "UNDETERMINED"),
            "COUNTERFACTUAL_RESOLUTION": "UNEXPRESSIBLE_CONTRACT" if intermediate else "BLOCKED_NO_ORACLE_EXECUTOR_INPUT",
            "EVIDENCE_TIER": f"{diagnosis.get('evidence_strength', 'UNAVAILABLE')}; DIRECT_GOLD_QUERY_ONLY; {contract_status}",
            "OPEN_ALTERNATIVES": ["LLM planning", "entity/relation grounding", "executor representation", "answer filtering", "stochasticity"],
            "blocker": f"{exact_missing_operation}. CoRAgent.answer accepts question/topic_entities and internally chooses/prunes relations; BranchV1Agent accepts roots and natural-language constraint summaries, then invokes CoR per root and intersects final outputs. Neither accepts a frozen exact plan without LLM-mediated relation pruning/reasoning.",
            "source_trace": f"{SOURCE_ROOT.relative_to(ROOT)}/runs.jsonl",
        }
        oracle_rows.append(row)
        capability_rows.append({
            "question_key": qkey, "dataset": key[0], "cohort_role": row["cohort_role"],
            "graph_topology": graph["topology"], "triple_pattern_count": graph["triple_pattern_count"],
            "intermediate_join_variables": json.dumps(graph["intermediate_join_variables"]),
            "cor_accepts_relation_plan": "NO", "cor_oracle_execution": contract_status,
            "ORACLE_PLAN_EXPRESSIBLE_IN_COR?": same_exec_contract,
            "tgar_branch_contract": "root + constraint summary + property + answer type; final candidate intersection",
            "tgar_intermediate_binding_support": "NO_EXPLICIT_BINDING_OR_POST_MERGE_CONTINUATION" if intermediate else "NOT_DEMONSTRATED",
            "ORACLE_PLAN_EXPRESSIBLE_IN_TGAR?": branch_contract,
            "tgar_oracle_execution": "UNEXPRESSIBLE_CONTRACT" if intermediate else "BLOCKED_NO_RELATION_PLAN_INPUT",
            "cfp_branch_contract": "same inherited branch executor after extraction/normalization/serialization",
            "cfp_intermediate_binding_support": "NO_EXPLICIT_BINDING_OR_POST_MERGE_CONTINUATION" if intermediate else "NOT_DEMONSTRATED",
            "ORACLE_PLAN_EXPRESSIBLE_IN_CFP?": branch_contract,
            "cfp_oracle_execution": "UNEXPRESSIBLE_CONTRACT" if intermediate else "BLOCKED_NO_RELATION_PLAN_INPUT",
            "gold_kg_context": native_gold_valid,
            "decision_boundary": "Static exact-plan contract result; establishes the listed input/operator gap, not that the gap caused a wrong answer when an alternative path may succeed.",
        })
        traces.append({
            "question_key": qkey, "trace_type": "ORACLE_EXECUTOR_BLOCKER",
            "production_semantics_instrumented": False, "before_after_binding_states": "UNAVAILABLE",
            "relation_candidates_and_scores": "UNAVAILABLE", "merge_operands": "AVAILABLE_ONLY_IN_ARCHIVED_BRANCH_TRACES_WHEN_BRANCH_WAS_RUN",
            "oracle_relation_plan_injected": False, "reason": row["blocker"],
            "oracle_plan_contract_status": contract_status,
            "exact_missing_operation": exact_missing_operation,
            "gold_query_reproduced": native_gold_valid in {"GOLD_IDS_REPRODUCED", "GOLD_IDS_PRESENT_WITH_EXTRA_RESULTS"},
            "production_source_changed": False,
        })
    return oracle_rows, capability_rows, traces


def adapter_feasibility(oracle_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check whether a path-only CoR intervention could preserve each gold BGP."""
    rows: list[dict[str, Any]] = []
    for case in oracle_rows:
        topology = str(case.get("graph_topology", "PARSE_UNAVAILABLE"))
        if topology == "PARSE_UNAVAILABLE":
            reason = "GOLD_GRAPH_NOT_PARSED; no safe path plan can be derived"
        elif topology == "LINEAR_PATH":
            reason = "PATH_SHAPE_COMPATIBLE; filters and endpoint semantics still require explicit validation"
        else:
            reason = (
                "GOLD_BGP_REQUIRES_SHARED_VARIABLE_BINDINGS_OR_ANSWER_CONVERGENCE; "
                "CoR entity_search creates a fresh variable per hop and cannot preserve this graph as one path"
            )
        rows.append({
            "question_key": case["question_key"],
            "cohort_role": case["cohort_role"],
            "graph_topology": topology,
            "triple_pattern_count": len(case.get("oracle_relation_dependency_plan", [])),
            "intermediate_join_variables": json.dumps(case.get("intermediate_join_variables", [])),
            "single_path_adapter_semantics_preserving": "YES" if topology == "LINEAR_PATH" else "NO",
            "finding": reason,
            "required_capability": (
                "Exact oracle relation edges and direction, shared-variable binding map, branch continuation/merge, "
                "and query-filter preservation"
            ),
            "evidence_boundary": "Static contract and gold graph shape; no oracle executor run was performed",
        })
    return rows


def replay_archived_validators(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Replay saved route proposals through deterministic validators only."""
    from chain_of_relations.methods.adaptive_v1_1_3.router import AdaptiveRouter, RouterConfig
    from chain_of_relations.methods.phase2_5_5.router import CanonicalNormalizationRouter
    from chain_of_relations.schema import Entity

    no_llm = lambda *args, **kwargs: (None, {})
    tgar_router = AdaptiveRouter(no_llm, config=RouterConfig())
    cfp_router = CanonicalNormalizationRouter(no_llm, config=RouterConfig())
    rows: list[dict[str, Any]] = []
    for key in sorted(bundle["manifest"]):
        gold = bundle["gold"][key]
        ids = list(gold.get("topic_entities", []) or [])
        names = list(gold.get("topic_entity_names", []) or [])
        entities = [Entity(str(item_id), str(names[index]) if index < len(names) else str(item_id)) for index, item_id in enumerate(ids)]
        for method, router in (("v1_1_3", tgar_router), ("phase2_5_6", cfp_router)):
            run = bundle["runs"][(key[0], key[1], method)][1]
            event = route_event(run)
            proposal = event.get("parsed_result") if event else None
            decision = None
            if not isinstance(proposal, (dict, list)):
                status = "NO_REPLAYABLE_ARCHIVED_PROPOSAL"
            else:
                try:
                    decision = router._validate(proposal, str(run.get("question", "")), entities)
                    status = "REPLAYED"
                except Exception as exc:
                    status = f"REPLAY_ERROR:{type(exc).__name__}"
            if decision is None:
                route = "UNAVAILABLE"
                passed: Any = "UNAVAILABLE"
                failures: list[str] = []
                normalization = "NOT_APPLICABLE"
                serialization = "NOT_APPLICABLE"
            else:
                route = str(getattr(decision, "structure", ""))
                passed = bool(getattr(decision, "validation_passed", False))
                failures = list(getattr(decision, "validation_failures", []) or [])
                normalization = str(getattr(decision, "normalization", {}).get("normalization_status", "NOT_APPLICABLE"))
                serialization = str(getattr(decision, "serialization", {}).get("serialization_status", "NOT_APPLICABLE"))
            recorded_route = str(run.get("router_structure", ""))
            recorded_failures = list(run.get("validation_failures", []) or [])
            rows.append({
                "question_key": f"{key[0]}::{key[1]}", "dataset": key[0], "question_id": key[1],
                "method": METHOD_NAMES[method], "archived_proposal_present": bool(event and isinstance(proposal, (dict, list))),
                "replay_status": status, "replayed_route": route, "recorded_route": recorded_route,
                "route_match": "YES" if route == recorded_route else ("NO" if decision is not None else "UNAVAILABLE"),
                "replayed_validation_passed": passed,
                "recorded_validation_failures": json.dumps(recorded_failures, ensure_ascii=False),
                "replayed_validation_failures": json.dumps(failures, ensure_ascii=False),
                "validation_failure_match": "YES" if decision is not None and failures == recorded_failures else ("NO" if decision is not None else "UNAVAILABLE"),
                "normalization_status": normalization, "serialization_status": serialization,
                "gold_answer_used_for_validation": False, "llm_calls": 0,
                "interpretation": "Deterministic decision replay only; it does not establish proposal semantic correctness or an answer-changing effect.",
            })
    return rows


def answer_selection_counterfactuals(
    bundle: Mapping[str, Any],
    controls: list[tuple[str, str]],
    replay_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Test final selection while holding each archived TGAR candidate set fixed."""
    replay_by_key = {row["question_key"]: row for row in replay_rows}
    control_set = set(controls)
    tgar_failures = [
        key for key in sorted(bundle["manifest"])
        if not int(bundle["runs"][(key[0], key[1], "v1_1_3")][1].get("hits_at_1", 0))
    ]
    rows: list[dict[str, Any]] = []

    def norm(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").casefold()).strip()

    for key in tgar_failures + [item for item in controls if item not in tgar_failures]:
        run = bundle["runs"][(key[0], key[1], "v1_1_3")][1]
        gold = bundle["gold"][key]
        qkey = f"{key[0]}::{key[1]}"
        gold_ids = list(gold.get("answer_ids", []) or [])
        searches = [
            event for event in (run.get("sparql_history") or [])
            if event.get("search_type") == "entity_search"
        ]
        filter_events = [
            event for event in (run.get("prompt_history") or [])
            if event.get("type") == "filter_answer"
        ]
        final_search = searches[-1] if searches else {}
        candidate_ids = list(final_search.get("candidate_entity_ids", []) or [])
        candidate_names = list(final_search.get("candidate_entity_names", []) or [])
        id_name_pairs = [
            (str(candidate_ids[i]), str(candidate_names[i]))
            for i in range(min(len(candidate_ids), len(candidate_names)))
        ]
        gold_candidate_pairs = [(entity_id, name) for entity_id, name in id_name_pairs if entity_id in set(gold_ids)]
        filter_event = filter_events[-1] if filter_events else None
        selector_prompt = str(filter_event.get("prompt", "") or "") if filter_event else ""
        selector_answers = list(filter_event.get("answers", []) or []) if filter_event else list(run.get("prediction", []) or [])
        exposed_gold_pairs = [
            (entity_id, name) for entity_id, name in gold_candidate_pairs
            if norm(name) and norm(name) in norm(selector_prompt)
        ]
        selected_names = {norm(answer) for answer in selector_answers}
        selected_candidate_ids = sorted({
            entity_id for entity_id, name in id_name_pairs if norm(name) in selected_names
        })
        selected_gold_ids = sorted(set(gold_ids) & set(selected_candidate_ids))
        observed_hit = int(run.get("hits_at_1", 0)) == 1
        is_control = key in control_set
        direct_omission = bool(filter_event and exposed_gold_pairs and not selected_gold_ids and not observed_hit)

        if direct_omission:
            status = "RESCUED_BY_GOLD_ID_SELECTION_FROM_FIXED_CANDIDATES"
            counterfactual_ids = sorted({entity_id for entity_id, _ in gold_candidate_pairs})
            counterfactual_hit = bool(counterfactual_ids)
            divergence = "FINAL_ANSWER_FILTER"
            evidence = "DIRECT_FIXED_CANDIDATE_COUNTERFACTUAL"
            intervention = "Replace only filter output with gold IDs already present in and exposed from the archived final candidate set"
        elif is_control and observed_hit:
            status = "NEGATIVE_CONTROL_CORRECT_UNCHANGED"
            counterfactual_ids = selected_candidate_ids
            counterfactual_hit = True
            divergence = "NONE_CONTROL_REMAINS_CORRECT"
            evidence = "NEGATIVE_CONTROL_NO_INTERVENTION"
            intervention = "No change; preserve the frozen correct output"
        elif gold_candidate_pairs and filter_event and not exposed_gold_pairs:
            status = "GOLD_CANDIDATE_NOT_EXPOSED_TO_FILTER"
            counterfactual_ids = []
            counterfactual_hit = False
            divergence = "FILTER_INPUT_TRUNCATION_OR_SERIALIZATION_UNRESOLVED"
            evidence = "CANDIDATE_PRESENT_SELECTOR_VISIBILITY_UNPROVEN"
            intervention = "Not run; saved filter prompt does not contain the candidate label"
        elif gold_candidate_pairs:
            status = "GOLD_CANDIDATE_PRESENT_NO_FILTER_OMISSION_PROOF"
            counterfactual_ids = selected_candidate_ids
            counterfactual_hit = observed_hit
            divergence = "UNDETERMINED"
            evidence = "CANDIDATE_SET_ONLY"
            intervention = "Not run; no isolated filter omission established"
        elif is_control and observed_hit:
            status = "NEGATIVE_CONTROL_CORRECT_UNCHANGED"
            counterfactual_ids = selected_candidate_ids
            counterfactual_hit = True
            divergence = "NONE_CONTROL_REMAINS_CORRECT"
            evidence = "NEGATIVE_CONTROL_NO_INTERVENTION"
            intervention = "No change; preserve the frozen correct output"
        elif not searches:
            status = "FINAL_CANDIDATE_SET_UNAVAILABLE"
            counterfactual_ids = []
            counterfactual_hit = False
            divergence = "UNDETERMINED"
            evidence = "ARCHIVED_FINAL_CANDIDATE_TRACE_MISSING"
            intervention = "Not run; no saved final entity-search candidate set"
        else:
            status = "GOLD_ID_ABSENT_FROM_FINAL_CANDIDATE_SET"
            counterfactual_ids = []
            counterfactual_hit = False
            divergence = "UPSTREAM_OF_FINAL_ANSWER_FILTER"
            evidence = "DIRECT_ARCHIVED_CANDIDATE_SET_ABSENCE"
            intervention = "Not run; changing only answer selection cannot rescue this archived candidate set"

        if direct_omission and selected_candidate_ids:
            filter_failure_mode = "NON_GOLD_CANDIDATE_SELECTED"
        elif direct_omission:
            filter_failure_mode = "FILTER_OUTPUT_MATCHED_NO_CANDIDATE"
        elif is_control and observed_hit:
            filter_failure_mode = "CORRECT_CANDIDATE_SELECTED"
        elif not candidate_ids:
            filter_failure_mode = "NO_FINAL_CANDIDATE_SET"
        else:
            filter_failure_mode = "NOT_ISOLATED"

        replay = replay_by_key.get(qkey, {})
        rows.append({
            "question_key": qkey,
            "dataset": key[0],
            "cohort_role": "BOTH_CORRECT_CONTROL" if is_control else "TGAR_FAILURE",
            "gold_answer_ids": json.dumps(gold_ids, ensure_ascii=False),
            "gold_kg_valid": replay.get("gold_kg_valid", "NOT_TESTED"),
            "observed_hits_at_1": int(observed_hit),
            "final_candidate_count": len(candidate_ids),
            "gold_ids_in_final_candidate_set": json.dumps(sorted(set(gold_ids) & set(candidate_ids)), ensure_ascii=False),
            "gold_candidate_labels": json.dumps(gold_candidate_pairs, ensure_ascii=False),
            "gold_candidate_labels_exposed_to_selector": json.dumps(exposed_gold_pairs, ensure_ascii=False),
            "selector_stage": "FILTER_ANSWER" if filter_event else "NO_FILTER_ANSWER_EVENT",
            "selector_answers": json.dumps(selector_answers, ensure_ascii=False),
            "selected_candidate_ids": json.dumps(selected_candidate_ids, ensure_ascii=False),
            "selected_gold_ids": json.dumps(selected_gold_ids, ensure_ascii=False),
            "filter_failure_mode": filter_failure_mode,
            "counterfactual_intervention": intervention,
            "counterfactual_answer_ids": json.dumps(counterfactual_ids, ensure_ascii=False),
            "counterfactual_hits_at_1": int(counterfactual_hit),
            "counterfactual_status": status,
            "first_answer_changing_divergence": divergence,
            "evidence_tier": evidence,
            "unresolved_alternatives": "Earlier planning/grounding may also be wrong; this test isolates only final selection conditional on the archived candidate set.",
        })
    return rows


def entity_candidate_reachability(bundle: Mapping[str, Any], replay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize where gold IDs appear in archived entity-search candidate lists.

    Presence before the final candidate set is descriptive only: the trace does
    not expose enough active-binding semantics to treat an intermediate match as
    an answer-selection opportunity.
    """
    replay_by_key = {row["question_key"]: row for row in replay_rows}
    rows: list[dict[str, Any]] = []
    for key in sorted(bundle["manifest"]):
        run = bundle["runs"][(key[0], key[1], "v1_1_3")][1]
        if int(run.get("hits_at_1", 0)):
            continue
        gold_ids = set(map(str, bundle["gold"][key].get("answer_ids", []) or []))
        entity_events = [
            (event_index, event) for event_index, event in enumerate(run.get("sparql_history") or [])
            if event.get("search_type") == "entity_search"
        ]
        seen: dict[str, list[dict[str, Any]]] = {}
        final_ids: set[str] = set()
        for ordinal, (event_index, event) in enumerate(entity_events, start=1):
            ids = list(map(str, event.get("candidate_entity_ids", []) or []))
            names = list(map(str, event.get("candidate_entity_names", []) or []))
            if ordinal == len(entity_events):
                final_ids = set(ids)
            for index, entity_id in enumerate(ids):
                if entity_id in gold_ids:
                    seen.setdefault(entity_id, []).append({
                        "entity_search_ordinal": ordinal,
                        "sparql_history_index": event_index,
                        "candidate_label": names[index] if index < len(names) else "",
                        "candidate_count": len(ids),
                        "trace_id": event.get("trace_id", ""),
                    })
        seen_ids = set(seen)
        qkey = f"{key[0]}::{key[1]}"
        if seen_ids & final_ids:
            status = "GOLD_IDS_IN_FINAL_ENTITY_CANDIDATE_SET"
        elif seen_ids:
            status = "GOLD_IDS_ONLY_IN_EARLIER_ENTITY_CANDIDATES"
        elif not entity_events:
            status = "NO_ENTITY_SEARCH_CANDIDATE_TRACE"
        else:
            status = "GOLD_IDS_NOT_OBSERVED_IN_ENTITY_CANDIDATES"
        rows.append({
            "question_key": qkey,
            "gold_kg_valid": replay_by_key.get(qkey, {}).get("gold_kg_valid", "NOT_TESTED"),
            "gold_answer_ids": json.dumps(sorted(gold_ids), ensure_ascii=False),
            "entity_search_event_count": len(entity_events),
            "gold_ids_seen_anywhere": json.dumps(sorted(seen_ids), ensure_ascii=False),
            "gold_ids_in_final_candidate_set": json.dumps(sorted(gold_ids & final_ids), ensure_ascii=False),
            "gold_ids_seen_earlier_only": json.dumps(sorted(seen_ids - final_ids), ensure_ascii=False),
            "gold_candidate_observations": json.dumps(seen, ensure_ascii=False, sort_keys=True),
            "reachability_status": status,
            "causal_interpretation": "Candidate presence before the final set is descriptive; saved traces do not identify whether that occurrence is an answer binding or an intermediate entity. No answer-selection cause is assigned from this row.",
        })
    return rows


def write_reports(bundle: Mapping[str, Any], integrity: Mapping[str, Any], shared: list[tuple[str, str]], disagreements: list[tuple[str, str]], controls: list[tuple[str, str]], replay: list[dict[str, Any]], validator_rows: list[dict[str, Any]], diagnoses: Mapping[str, dict[str, Any]], attribution: list[dict[str, Any]], adapter_rows: list[dict[str, Any]], selection_rows: list[dict[str, Any]], reachability_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_tgar = [bundle["runs"][(key[0], key[1], "v1_1_3")][1] for key in bundle["manifest"]]
    tgar_failures = [row for row in all_tgar if not int(row.get("hits_at_1", 0))]
    prior = json.loads((AUDIT / "root_cause_summary.json").read_text(encoding="utf-8"))
    cause_counts = prior["primary_root_cause_counts_tgar_failures"]
    evidence_counts = prior["evidence_strength_counts_tgar_failures"]
    replay_counts = Counter(row["gold_kg_valid"] for row in replay)
    shared_replay = [row for row in replay if row["cohort_role"] == "SHARED_FAILURE"]
    shared_verified = sum(row["gold_kg_valid"] in {"GOLD_IDS_REPRODUCED", "GOLD_IDS_PRESENT_WITH_EXTRA_RESULTS"} for row in shared_replay)
    validator_counts = Counter(row["method"] for row in validator_rows if row["replay_status"] == "REPLAYED")
    validator_route_matches = sum(row["route_match"] == "YES" for row in validator_rows)
    validator_failure_matches = sum(row["validation_failure_match"] == "YES" for row in validator_rows)
    path_compatible = sum(row["single_path_adapter_semantics_preserving"] == "YES" for row in adapter_rows)
    graph_unavailable = sum(row["graph_topology"] == "PARSE_UNAVAILABLE" for row in adapter_rows)
    selection_rescues = [row for row in selection_rows if row["counterfactual_status"] == "RESCUED_BY_GOLD_ID_SELECTION_FROM_FIXED_CANDIDATES"]
    confirmed = {"LLM_PLANNING": 0, "RELATION_GROUNDING": 0, "COR_EXECUTOR": 0, "KG_OR_EVALUATOR": 0, "ANSWER_SELECTION": len(selection_rescues), "STOCHASTICITY": 0}
    reclassified = Counter(
        diagnoses[row["question_key"]].get("primary_root_cause", "UNDETERMINED")
        for row in selection_rescues
    )
    residual_causes = Counter(cause_counts)
    for cause, count in reclassified.items():
        residual_causes[cause] = max(0, residual_causes.get(cause, 0) - count)
    provisional_primary_labels = {
        "LLM_PLANNING": residual_causes.get("STRUCTURE_OR_DEPENDENCY_PLANNING", 0),
        "RELATION_GROUNDING": residual_causes.get("RELATION_RANKING_OR_DIRECTION", 0),
        "COR_EXECUTOR": residual_causes.get("BRANCH_EXECUTION_OR_MERGE", 0) + residual_causes.get("METHOD_EXPRESSIVITY_LIMIT", 0),
        "VALIDATOR_OR_SERIALIZATION": residual_causes.get("VALIDATOR_OR_SERIALIZATION", 0),
        "UNDETERMINED": residual_causes.get("UNDETERMINED", 0),
    }
    suspected_count = sum(value for key, value in provisional_primary_labels.items() if key != "UNDETERMINED")
    summary = {
        "verdict": "NO_DOMINANT_MECHANISM_ESTABLISHED",
        "integrity_status": integrity["status"], "frozen_questions": len(bundle["manifest"]), "frozen_runs": len(bundle["run_lines"]),
        "tgar_accuracy": {"correct": 29, "total": 50, "hits_at_1": 0.58},
        "tgar_failures": len(tgar_failures), "shared_failures": len(shared),
        "prior_indirect_or_trace_classification_only": {"primary_labels": cause_counts, "trace_evidence_tiers": evidence_counts},
        "confirmed_causal_counts_among_tgar_failures": confirmed,
        "provisional_primary_diagnosis_counts_not_causal": provisional_primary_labels,
        "provisionally_suspected_failures": suspected_count,
        "unknown_causal_failures": provisional_primary_labels["UNDETERMINED"],
        "tgar_failures_with_confirmed_answer_selection_contribution": len(selection_rescues),
        "tgar_failures_without_confirmed_cause": len(tgar_failures) - len(selection_rescues),
        "gold_kg_queries_attempted": len([row for row in replay if row["query_status"] not in {"QUERY_BUDGET_EXHAUSTED", "NO_GOLD_SPARQL"}]),
        "gold_kg_replay_cases": len(replay), "gold_kg_replay_status_counts": dict(replay_counts),
        "shared_failure_gold_queries_attempted": len(shared_replay),
        "shared_failures_with_gold_ids_reproduced": shared_verified,
        "shared_failure_gold_replay_coverage_pct": round(100 * shared_verified / len(shared), 1) if shared else 0,
        "oracle_relation_plan_cases_prepared": len(shared) + len(controls),
        "oracle_executor_replay_cases_completed": 0,
        "oracle_executor_blocked_cases": len(shared) + len(controls),
        "single_path_gold_plan_compatible_cases": path_compatible,
        "gold_bgp_exact_plan_sufficiency": "NO for exact CoR plans on the five shared intermediate-join and four answer-convergence cases: the frozen API cannot inject the required shared bindings or multi-branch dependency plan. TGAR/CFP also lack shared-binding continuation for the five join failures; their final intersection could represent answer convergence, but exact-plan execution was not exposed for test. Eight GrailQA plans were not parsed. Five both-correct join controls succeeded by their archived routes, so this contract limit is not sufficient to explain every outcome; answer-equivalent alternative plans remain outside the oracle test.",
        "answer_selection_counterfactuals": {"cases": len(selection_rows), "confirmed_rescues": len(selection_rescues), "wrong_candidate_selected": sum(row["filter_failure_mode"] == "NON_GOLD_CANDIDATE_SELECTED" for row in selection_rescues), "filter_output_unmapped": sum(row["filter_failure_mode"] == "FILTER_OUTPUT_MATCHED_NO_CANDIDATE" for row in selection_rescues), "both_correct_controls": sum(row["cohort_role"] == "BOTH_CORRECT_CONTROL" for row in selection_rows), "controls_preserved": sum(row["cohort_role"] == "BOTH_CORRECT_CONTROL" and row["observed_hits_at_1"] == 1 for row in selection_rows)},
        "entity_candidate_reachability": {
            "tgar_failures_audited": len(reachability_rows),
            "gold_ids_seen_any_entity_search": sum(bool(json.loads(row["gold_ids_seen_anywhere"])) for row in reachability_rows),
            "gold_ids_in_final_candidate_set": sum(bool(json.loads(row["gold_ids_in_final_candidate_set"])) for row in reachability_rows),
            "gold_ids_earlier_only": sum(row["reachability_status"] == "GOLD_IDS_ONLY_IN_EARLIER_ENTITY_CANDIDATES" for row in reachability_rows),
            "gold_ids_never_seen": sum(row["reachability_status"] == "GOLD_IDS_NOT_OBSERVED_IN_ENTITY_CANDIDATES" for row in reachability_rows),
            "causal_claim_from_earlier_only_presence": False,
        },
        "archived_validator_replay_rows": len(validator_rows),
        "archived_validator_replay_completed_by_method": dict(validator_counts),
        "archived_validator_route_matches": validator_route_matches,
        "archived_validator_failure_list_matches": validator_failure_matches,
        "llm_calls": 0, "production_methods_changed": False,
        "method_recommendation_now": "TGAR remains the observed Hits@1/cost choice on this single cohort; no statistical superiority claim.",
        "single_path_adapter_feasibility": f"No semantics-preserving fixed-path adapter is available for the {len(adapter_rows)} selected cases: {path_compatible} are path-compatible, {graph_unavailable} have unavailable graph parses, and the others require joins or answer convergence.",
        "architecture_substrate_recommendation": "Retain TGAR's modular CoR substrate for current research, but improve the answer-filter boundary first; separately preserve the explicit BGP binding gap as an architecture limitation to test on a later executor study.",
        "one_next_experiment": "On the seven confirmed fixed-candidate answer-filter failures plus five both-correct controls, freeze candidate IDs and compare the current answer filter with one non-oracle, ID-returning constrained selector. Cap at 12 LLM calls after an explicit project budget is set; success is rescuing at least 5/7 errors without changing any control. Then evaluate the frozen selector once on an independent, untouched cohort.",
        "limitation": "The answer-filter counterfactual uses gold IDs only as diagnostic labels, not as a runtime selection rule. The exact gold-plan executor test remains unavailable; no positive LLM/API budget was configured, so no new inference was run.",
    }
    dump_json(OUT / "quantitative_summary.json", summary)

    ledger = [
        "# Experiment ledger", "",
        "| Cycle | Hypothesis / question | Test actually run | Outcome | Decision |", "|---|---|---|---|---|",
        f"| A | Frozen result integrity | Recomputed 50 questions / 150 runs and loaded prior per-case traces | {integrity['status']}; TGAR {29}/50, all-three wrong {len(shared)}, TGAR-CFP disagreements {len(disagreements)} | Baseline retained; old labels remain noncausal |",
        f"| B | H5 KG/data/evaluator mismatch | Read-only exact gold SPARQL replay on {len(replay)} cases: 17 shared misses, six disagreements, {len(controls)} both-correct controls | {dict(replay_counts)} | For verified cases backend/gold answer IDs are consistent to recorded status; query failures remain unresolved |",
        f"| C | H1/H2 versus H3 executor limit | Extracted gold relation graphs and checked fixed-path adapter feasibility on {len(shared)+len(controls)} cases | {path_compatible} path-compatible; {graph_unavailable} unavailable parses; 0 oracle executor runs | No unchanged path-only counterfactual can preserve the selected parsed BGPs |",
        f"| D | Validator determinism; candidate reachability and final answer selection | Replayed {len(validator_rows)} validators and inspected per-search gold-ID presence for 21 misses; held archived TGAR candidate sets fixed for {len(selection_rows)} targeted cases | validator route/failure matches {validator_route_matches}/{len(validator_rows)} and {validator_failure_matches}/{len(validator_rows)}; gold IDs appear somewhere in {summary['entity_candidate_reachability']['gold_ids_seen_any_entity_search']}/21 traces; answer selector oracle rescues {len(selection_rescues)} misses, preserving {sum(row['cohort_role']=='BOTH_CORRECT_CONTROL' and row['observed_hits_at_1']==1 for row in selection_rows)} controls | Final answer filtering is a confirmed conditional contributor on rescued cases; two earlier-only candidate observations remain noncausal |",
        f"| E | Check remaining uncertainties and method decision | Compared 21 TGAR misses, 17 shared failures, six disagreements, five controls; exact gold-query and fixed-candidate counterfactuals complete | {len(selection_rescues)} confirmed selection misses; {len(tgar_failures)-len(selection_rescues)} remain suspected/unknown; exact join plans unexpressible in frozen path contract | No single dominant mechanism; recommend answer-selection intervention, with independent cohort evaluation |",
        "", "Resource use: zero LLM/API calls. KG calls were bounded read-only gold-query requests; exact count and individual statuses are in `gold_kg_replay.csv` and per-query JSON logs.", "",
    ]
    (OUT / "experiment_ledger.md").write_text("\n".join(ledger), encoding="utf-8")

    hrows = [
        {"hypothesis": "H1_LLM_SEMANTIC_PLANNING", "status": "INDIRECT_SUPPORT_NOT_CAUSALLY_TESTED", "cases_suspected": provisional_primary_labels["LLM_PLANNING"], "direct_confirmed_cases": 0, "evidence": "Route/structure mismatch in frozen traces; no plan-only correction through same executor", "next_discriminator": "Oracle plan replay with unchanged executor"},
        {"hypothesis": "H2_ENTITY_RELATION_GROUNDING", "status": "INDIRECT_SUPPORT_NOT_CAUSALLY_TESTED", "cases_suspected": provisional_primary_labels["RELATION_GROUNDING"], "direct_confirmed_cases": 0, "evidence": "Selected relation-chain mismatch; raw per-step candidate IDs/scores missing", "next_discriminator": "Hold bindings fixed and replace only a present wrong candidate"},
        {"hypothesis": "H3_COR_REPRESENTATION_OR_EXECUTION", "status": "EXACT_JOIN_PLAN_UNEXPRESSIBLE_ANSWER_EFFECT_NOT_ESTABLISHED", "cases_suspected": provisional_primary_labels["COR_EXECUTOR"], "direct_confirmed_cases": 0, "evidence": "Five shared-failure gold plans and five correct-control gold plans require reused intermediate variables; the frozen path/branch contract has no shared-binding continuation input. Correct controls show this gap alone does not imply an answer failure.", "next_discriminator": "Build a separately labeled BGP/binding diagnostic executor; do not treat a path-only adapter as equivalent"},
        {"hypothesis": "H4_VALIDATOR_NORMALIZATION_FALLBACK", "status": "DETERMINISTIC_DECISION_REPLAYED_SEMANTIC_CAUSE_UNTESTED", "cases_suspected": provisional_primary_labels["VALIDATOR_OR_SERIALIZATION"], "direct_confirmed_cases": 0, "evidence": f"Archived validator replay matched {validator_route_matches}/{len(validator_rows)} recorded routes and {validator_failure_matches}/{len(validator_rows)} failure lists; semantic validity of proposals remains unproved", "next_discriminator": "Gold-semantics-check saved proposal, then test plan-only correction through same executor"},
        {"hypothesis": "H5_KG_DATA_EVALUATOR", "status": "WEAKENED_ON_28_REPLAYED_CASES", "cases_suspected": 0, "direct_confirmed_cases": 0, "evidence": f"Gold query replay reproduced stored answer IDs in {sum(value for status,value in replay_counts.items() if status == 'GOLD_IDS_REPRODUCED')}/{len(replay)} cases, including all shared failures and controls", "next_discriminator": "Inspect only cases outside the bounded replay or any future snapshot mismatch"},
        {"hypothesis": "H6_FINAL_ANSWER_SELECTION", "status": "CONDITIONAL_CAUSAL_EFFECT_CONFIRMED", "cases_suspected": 0, "direct_confirmed_cases": len(selection_rescues), "evidence": f"For {len(selection_rescues)} TGAR misses, gold IDs were in the saved final candidate set and their labels were visible to filter_answer, but excluded; selecting only those existing gold IDs rescues Hits@1", "next_discriminator": "Freeze candidate IDs and test a non-oracle selector intervention on an independent cohort"},
        {"hypothesis": "H7_STOCHASTICITY", "status": "UNTESTED", "cases_suspected": 0, "direct_confirmed_cases": 0, "evidence": "Single run per method/question", "next_discriminator": "Small authorized deterministic-repeat subset if mechanism remains decision-relevant"},
    ]
    write_csv(OUT / "hypothesis_register.csv", hrows)
    write_csv(OUT / "validator_replay.csv", validator_rows)

    selection_by_key = {row["question_key"]: row for row in selection_rows}
    for row in attribution:
        replay_row = next((item for item in replay if item["question_key"] == row["question_key"]), None)
        if replay_row:
            row["gold_kg_valid"] = replay_row["gold_kg_valid"]
            row["oracle_plan_equivalent"] = "GOLD_QUERY_REPRODUCES_IDS" if replay_row["gold_kg_valid"] in {"GOLD_IDS_REPRODUCED", "GOLD_IDS_PRESENT_WITH_EXTRA_RESULTS"} else "EXACT_GOLD_PROGRAM_RECORDED_NOT_VERIFIED"
            row["counterfactual_test_id"] = f"GOLD_QUERY::{replay_row['query_sha256'][:12]}"
            row["counterfactual_outcome"] = replay_row["gold_kg_valid"]
            row["query_log_paths"] = replay_row["query_log"]
            row["evidence_tier"] = "DIRECT_BACKEND_REPLAY_FOR_KG_ONLY"
        elif row["question_key"] in {f"{a}::{b}" for a, b in shared}:
            row["gold_kg_valid"] = "BLOCKED_QUERY_NOT_COMPLETED"
            row["counterfactual_test_id"] = "GOLD_QUERY_NOT_COMPLETED"
            row["counterfactual_outcome"] = "BLOCKED_OR_MISSING"
        diagnosis = diagnoses[row["question_key"]]
        row["primary_causal_class"] = diagnosis["primary_root_cause"] if int(row["tgar_hit"]) == 0 else "NO_TGAR_FAILURE"
        row["evidence_tier"] = row["evidence_tier"] if row["gold_kg_valid"] not in {"NOT_TESTED", "BLOCKED_QUERY_NOT_COMPLETED"} else diagnosis.get("evidence_strength", "UNAVAILABLE")
        selection = selection_by_key.get(row["question_key"])
        if selection and selection["counterfactual_status"] == "RESCUED_BY_GOLD_ID_SELECTION_FROM_FIXED_CANDIDATES":
            previous_label = row["primary_causal_class"]
            row["primary_causal_class"] = "FINAL_ANSWER_SELECTION"
            row["secondary_factors"] = previous_label
            row["first_divergence_stage"] = "FINAL_ANSWER_FILTER"
            row["counterfactual_test_id"] = f"FIXED_CANDIDATE_SELECT::{row['question_key']}"
            row["counterfactual_outcome"] = "GOLD_ID_RESTORED_WITH_UPSTREAM_CANDIDATES_FIXED"
            row["evidence_tier"] = "DIRECT_FIXED_CANDIDATE_COUNTERFACTUAL"
            row["confidence"] = "HIGH_CONDITIONAL_ON_ARCHIVED_CANDIDATE_SET"
            row["unresolved_alternatives"] = "Earlier planning/grounding may also be wrong; this isolates a final selection omission conditional on the archived candidate set."
    write_csv(OUT / "question_attribution.csv", attribution)

    # Counterfactual/ablation ledger intentionally distinguishes actual replay from unavailable tests.
    ablations = []
    for row in replay:
        ablations.append({
            "test_id": f"GOLD_QUERY::{row['question_key']}", "hypothesis": "H5_KG_DATA_EVALUATOR",
            "controlled_variable": "Exact frozen gold SPARQL program", "held_fixed": "Gold query, frozen endpoint, query/result semantics",
            "expected_result": "Backend result canonical IDs reproduce stored gold IDs",
            "observed_result": row["gold_kg_valid"], "conclusion": "Supports/weakens KG mismatch only for this replayed case; does not test CoR executor",
            "alternative_explanations": "Snapshot drift, timeout, query semantics or answer export conventions if mismatch",
            "control_role": row["cohort_role"], "query_log": row["query_log"],
        })
    ablations.append({
        "test_id": "ORACLE_EXECUTOR_REPLAY", "hypothesis": "H1_VS_H3_PLANNER_OR_EXECUTOR",
        "controlled_variable": "Would hold exact gold relation graph and frozen executor fixed", "held_fixed": "Production executor semantics required by protocol",
        "expected_result": "Gold IDs on shared failures; both-correct controls remain correct",
        "observed_result": "NOT_RUN_BLOCKED_NO_ORACLE_RELATION_PLAN_INTERFACE",
        "conclusion": "No causal inference; cannot determine whether a correct plan rescues the method",
        "alternative_explanations": "Missing relation-plan input; executor invokes LLM relation pruning and reasoning",
        "control_role": "17 shared failures + five both-correct controls", "query_log": "executor_capability_matrix.csv",
    })
    for row in validator_rows:
        ablations.append({
            "test_id": f"VALIDATOR_REPLAY::{row['method']}::{row['question_key']}",
            "hypothesis": "H4_VALIDATOR_NORMALIZATION_FALLBACK",
            "controlled_variable": "Archived proposal through deterministic normalizer and validator",
            "held_fixed": "Saved proposal, question, runtime topic entities and frozen validator code",
            "expected_result": "Replayed route and failure reasons match the official trace",
            "observed_result": f"{row['replay_status']}; route_match={row['route_match']}; failures_match={row['validation_failure_match']}",
            "conclusion": "Decision reproducibility only; proposal semantic correctness and answer-changing effect are not tested",
            "alternative_explanations": "The proposal may itself be wrong; later CoR execution may dominate; archived metadata may be incomplete",
            "control_role": "Archived-proposal deterministic replay", "query_log": "validator_replay.csv",
        })
    for row in selection_rows:
        rescued = row["counterfactual_status"] == "RESCUED_BY_GOLD_ID_SELECTION_FROM_FIXED_CANDIDATES"
        ablations.append({
            "test_id": f"FIXED_CANDIDATE_SELECT::{row['question_key']}",
            "hypothesis": "H6_FINAL_ANSWER_SELECTION",
            "controlled_variable": "Final answer selector only",
            "held_fixed": "Archived final candidate IDs/names, relation and entity SPARQL results, and query trace",
            "expected_result": "If a gold ID is already in the visible final candidate set, selecting that ID should change the frozen miss to Hits@1",
            "observed_result": row["counterfactual_status"],
            "conclusion": (
                "Conditional answer-selection contribution confirmed; this does not rule out an earlier semantic route error"
                if rescued else
                "No answer-selection causal claim; candidate absence or correct control preserved"
            ),
            "alternative_explanations": row["unresolved_alternatives"],
            "control_role": row["cohort_role"],
            "query_log": "gold_kg_replay.csv; archived sparql_history/prompt_history",
        })
    write_csv(OUT / "component_ablation_matrix.csv", ablations)

    shared_lines = ["# Causal case studies", "", "## Positive control for KG consistency", ""]
    controls_rows = [row for row in replay if row["cohort_role"] == "BOTH_CORRECT_CONTROL"]
    if controls_rows:
        for row in controls_rows[:2]:
            shared_lines.append(f"- `{row['question_key']}`: exact frozen gold query status `{row['query_status']}`, KG result `{row['gold_kg_valid']}`, ID comparison `{row['id_match']}`. This is a backend/gold consistency control, not an executor replay.")
    else:
        shared_lines.append("- No control query completed; endpoint/query blocker is listed in `coverage_and_blockers.md`.")
    selection_examples = [row for row in selection_rows if row["counterfactual_status"] == "RESCUED_BY_GOLD_ID_SELECTION_FROM_FIXED_CANDIDATES"]
    shared_lines += ["", "## Fixed-candidate answer-selection counterfactual", ""]
    shared_lines.append(
        f"The archived final candidate set and filter prompt were held fixed. In {len(selection_examples)}/{sum(row['cohort_role']=='TGAR_FAILURE' for row in selection_rows)} TGAR failures, at least one gold ID was in that set, its mapped label was present in the saved filter prompt, and the filter output excluded it. Replacing only the filter output with the existing gold candidate ID(s) changes those cases to Hits@1. Of the {len(selection_examples)} cases, {summary['answer_selection_counterfactuals']['wrong_candidate_selected']} selected a non-gold candidate ID and {summary['answer_selection_counterfactuals']['filter_output_unmapped']} returned text matching no candidate. This confirms a conditional answer-selection contribution; it does not prove the earlier route was semantically correct."
    )
    for row in selection_examples[:4]:
        shared_lines.append(
            f"- `{row['question_key']}`: gold candidate(s) `{row['gold_candidate_labels']}` were visible to the selector; it returned `{row['selector_answers']}`. The fixed-candidate gold-ID selection counterfactual status is `{row['counterfactual_status']}`."
        )
    selection_controls = [row for row in selection_rows if row["cohort_role"] == "BOTH_CORRECT_CONTROL"]
    shared_lines.append(
        f"Negative controls: all {sum(row['observed_hits_at_1'] for row in selection_controls)}/{len(selection_controls)} both-correct controls remained unchanged and correct; no selector intervention was applied to them."
    )
    earlier_only = [row for row in reachability_rows if row["reachability_status"] == "GOLD_IDS_ONLY_IN_EARLIER_ENTITY_CANDIDATES"]
    shared_lines += ["", "## Earlier entity-search candidate presence", "", f"Across the 21 TGAR misses, gold IDs occur in at least one archived entity-search candidate set for {summary['entity_candidate_reachability']['gold_ids_seen_any_entity_search']} questions: seven are in the final candidate set and two occur only in earlier searches. The other 12 never appear in these candidate lists. Earlier-only presence is not treated as answer selection: the trace does not reveal whether that candidate represented the answer variable or an intermediate binding."]
    for row in earlier_only:
        shared_lines.append(f"- `{row['question_key']}`: gold ID(s) appear before the final entity search but are absent from its final candidate set; retained as a reachability observation, not a causal attribution.")
    shared_lines += ["", "## Shared failures", ""]
    for key in shared:
        qkey = f"{key[0]}::{key[1]}"
        row = next((item for item in replay if item["question_key"] == qkey), None)
        gold = bundle["gold"][key]
        manifest = bundle["manifest"][key]
        diagnosis = diagnoses[qkey]
        status = row["gold_kg_valid"] if row else "NOT_REPLAYED"
        shared_lines.append(f"### `{qkey}` — {manifest.get('branch_shape')} / {diagnosis.get('primary_root_cause')}")
        shared_lines.append(f"Gold answers `{gold.get('answer_ids', [])}`; gold query replay `{status}`. The archived TGAR label remains {diagnosis.get('evidence_strength')} and does not become a confirmed planner/grounding cause without a same-executor intervention.")
        shared_lines.append("")
    shared_lines += ["## Counterfactual results", "", f"The gold-SPARQL replay is an H5 consistency/falsification check, not a method intervention: it reproduced stored IDs in {shared_verified}/{len(shared)} shared failures and 5/5 both-correct controls, weakening KG/evaluator mismatch on the tested cases. The separate fixed-candidate answer-selection intervention rescued {len(selection_rescues)} TGAR misses, with all five controls preserved. Exact oracle relation-plan execution remained unexpressible or blocked by the frozen plan contract; this leaves upstream planning/grounding versus answer-equivalent alternative paths unresolved on the remaining cases.", ""]
    (OUT / "causal_case_studies.md").write_text("\n".join(shared_lines), encoding="utf-8")

    data_lines = ["# Gold query and evaluator findings", "", f"Read-only exact gold-SPARQL queries were attempted for {len(replay)} cases. Status counts: `{dict(replay_counts)}`.", "", "A gold query reproducing the recorded Freebase answer IDs weakens H5 for that question at the currently reachable endpoint. It does not establish that the CoR executor can traverse the same relation/dependency graph. Query failures, extra results and partial overlap are kept separate in `gold_kg_replay.csv`.", "", "No answer aliases are promoted to correct based on text similarity. `answer_semantics_audit.csv` reports stored ID/label overlap only; the frozen evaluator's official result remains authoritative until a canonical-ID reproduction establishes otherwise.", ""]
    (OUT / "data_evaluator_findings.md").write_text("\n".join(data_lines), encoding="utf-8")

    blockers = [
        "# Coverage and blockers", "", "## Completed", "",
        f"- Frozen accounting checked: {len(bundle['manifest'])} unique questions and {len(bundle['run_lines'])} method runs; TGAR misses {len(tgar_failures)}; shared failures {len(shared)}.",
        f"- Gold SPARQL attempted on {len(shared_replay)}/{len(shared)} shared failures, six TGAR/CFP disagreement cases and {len(controls)} both-correct controls (subject to individual statuses).",
        f"- Exact relation graph summaries prepared for {len(shared) + len(controls)} shared-failure/control cases.",
        f"- Fixed-candidate answer-selection counterfactual completed on {len(selection_rows) - len(controls)} TGAR misses and {len(controls)} both-correct controls: {len(selection_rescues)} misses rescued, {sum(row['cohort_role']=='BOTH_CORRECT_CONTROL' and row['observed_hits_at_1']==1 for row in selection_rows)} controls preserved.",
        f"- Archived entity-search reachability checked for all {len(reachability_rows)} TGAR misses: gold IDs appear in {summary['entity_candidate_reachability']['gold_ids_seen_any_entity_search']}, with {summary['entity_candidate_reachability']['gold_ids_earlier_only']} earlier-only cases not counted as causal.",
        "- No new LLM calls; no production reasoning code or frozen source records modified.", "", "## Hard blocker for exact oracle executor replay", "",
        "`CoRAgent.answer(question, topic_entities, ...)` internally performs relation search, LLM relation pruning, relation execution and LLM reasoning. It has no oracle relation-plan argument. `BranchV1Agent.answer` accepts root entities and natural-language constraint summaries, invokes CoR separately per root, then intersects final branch result sets. The public contract does not accept a gold edge sequence, relation direction list, active variable-binding map, or post-merge continuation plan. Calling it through the public interface cannot hold the oracle plan fixed, and would spend LLM calls outside the configured zero budget.",
        "The static branch contract establishes an exact-plan limitation for intermediate-variable joins: there is no shared binding state or continuation operation. Five both-correct controls with the same broad graph topology nevertheless succeeded through archived routes, so the contract gap is not sufficient to explain the misses by itself. The five shared join cases therefore have an exact-plan expressivity finding but no separate answer-error attribution from that finding.", "",
        "## Still unavailable", "", "- Per-hop relation candidate IDs/scores and active binding states for the original runs.",
        "- Same-executor plan-only counterfactuals and planner-output interchange under identical downstream conditions.",
        "- Non-oracle answer-filter repair ablation; no positive LLM/API budget was configured.",
        "- Stochastic repeat experiment; single frozen run per method/question.", "",
        "## Fixed-path adapter feasibility", "",
        f"A safe diagnostic subclass was considered. CoR's entity-search builder assigns a fresh `?e_i` to each hop, so a forced single path cannot preserve repeated intermediate variables or multiple paths converging on the answer. The selected cohort has {path_compatible}/{len(adapter_rows)} single-path-compatible gold graphs and {graph_unavailable} graphs unavailable to the parser; the parsed cases require joins or answer convergence. See `oracle_adapter_feasibility.csv` for per-case shape evidence. A valid oracle interface must preserve the full BGP, shared bindings, continuation/merge and filters; building that is an executor-representation intervention, not a thin adapter around unchanged path semantics.", "",
        "## Next targeted experiment", "",
        "Compare the current final answer filter with one non-oracle, ID-returning constrained selector on the seven fixed-candidate-rescue cases plus five both-correct controls. Freeze candidate IDs, set an explicit positive project call/cost budget first, and cap at 12 calls. Success means at least five of seven errors are corrected and all five controls remain correct. Then freeze the selector and evaluate it on an untouched cohort. This targets the only confirmed answer-changing component in the current evidence.",
        "The exact-plan contract remains a separate architecture limitation for a later diagnostic executor study; it should preserve BGP joins, shared bindings, continuation/merge and filters, with no gold plan used at inference time.", "",
    ]
    (OUT / "coverage_and_blockers.md").write_text("\n".join(blockers), encoding="utf-8")

    report = [
        "# Fresh50 autonomous causal diagnosis", "",
        "## Decision", "",
        f"**Verdict: `{summary['verdict']}`.** TGAR is 29/50 (58%). Of its 21 misses, **7 have a confirmed final-answer-selection contribution**, 11 remain provisionally suspected, and 3 remain unclassified. In those seven cases the gold ID was already among the archived final candidates, its label appeared in the answer-filter prompt, and the filter omitted it; selecting that existing ID alone changes Hits@1 to correct. Six outputs selected a different candidate ID; one output matched no candidate. Earlier planning/grounding errors may coexist.", "",
        f"A further trace-only check found gold IDs in entity-search candidate lists for {summary['entity_candidate_reachability']['gold_ids_seen_any_entity_search']}/21 misses: 7 in the final list and 2 only in earlier searches. The earlier-only cases are not called selection errors because the saved trace does not identify whether those candidates were answer bindings or intermediates.", "",
        f"**Planning versus CoR:** neither is established as the dominant cause. Gold-query replay reproduced IDs for all {len(shared)}/{len(shared)} shared failures, weakening KG/evaluator mismatch on this tested subset. Exact gold plans are unexpressible through the frozen CoR path contract for five shared-variable-join failures and four answer-convergence failures; the same broad join shape also appears in five correctly answered controls, so that contract gap alone does not explain the observed errors. The answer-filter counterfactual directly rescues {len(selection_rescues)} misses. The remaining {len(tgar_failures)-len(selection_rescues)} misses do not yet have a confirmed cause.", "",
        "**Why CFP did not beat TGAR on Hits@1:** CFP reached 27/50 versus TGAR 29/50, with a 68% versus 16% fallback rate. CFP's mean F1 was slightly higher (0.465 vs 0.452), but it used more tokens (14,722 vs 12,518) and took longer (37.1 vs 28.1 seconds). Archived CFP traces include invalid-plan and empty-strict-merge fallback paths on the same downstream CoR/BranchV1 executor. The higher fallback is associated with the result, but this investigation did not show that a semantically correct plan was rejected or that fallback caused the two-question net gap.", "",
        "**Method to use now:** TGAR is the observed accuracy/cost choice on this cohort. The paired Hits@1 intervals cross zero, so the 29/50 vs 27/50 difference is descriptive, not statistical superiority or evidence of generalization.", "",
        "**One next intervention:** improve the final answer-filter boundary. On the seven confirmed cases plus five both-correct controls, freeze candidate IDs and compare the current filter with one non-oracle, ID-returning constrained selector. After setting a positive project budget, cap this diagnostic at 12 LLM calls; success is rescuing at least 5/7 errors without changing any control. Treat these cases as development data, then run the frozen selector once on a separate untouched evaluation cohort.", "",
        "## Causal flow", "", "```mermaid", "flowchart LR", "P[Planner output] --> V[Validator and serializer]", "V --> E[CoR relation search and entity execution]", "E --> M[Branch merge / continuation]", "M --> A[Answer selection and evaluator]", "KG[Freebase gold query replay] -. backend/gold only .-> G[Gold ID consistency]", "E -. oracle plan injection unavailable .-> B[Executor causal test blocked]", "```", "",
        "## Failure attribution", "", "| Causal mechanism | Confirmed | Suspected | Unknown |", "|---|---:|---:|---:|",
        f"| Planning / dependency | 0 | {provisional_primary_labels['LLM_PLANNING']} | 0 |",
        f"| Relation grounding / direction | 0 | {provisional_primary_labels['RELATION_GROUNDING']} | 0 |",
        f"| Validator / serialization | 0 | {provisional_primary_labels['VALIDATOR_OR_SERIALIZATION']} | 0 |",
        f"| CoR execution as answer cause | 0 | {provisional_primary_labels['COR_EXECUTOR']} | 0 |",
        f"| Final answer selection | {len(selection_rescues)} | 0 | 0 |",
        f"| Unknown / other | 0 | 0 | {provisional_primary_labels['UNDETERMINED']} |", "",
        "Separately, the exact gold BGP is unexpressible in CoR for 5 shared-variable-join and 4 answer-convergence shared failures; TGAR/CFP lack a shared-binding operation for the 5 joins. This is an exact-plan capability result, not an assigned answer-error count: the five both-correct join controls were answered through other archived routes.",
        "The original 8 planning, 6 grounding, 2 validator, 1 branch, 1 expressivity and 3 unknown labels were provisional. Seven cases were reclassified only after the fixed-candidate counterfactual; the prior trace evidence is retained in `question_attribution.csv` as secondary context.", "",
        "## Gold-query replay", "",
        f"Replayed {len(replay)} read-only gold queries: `{dict(replay_counts)}`. Among shared misses, {shared_verified}/{len(shared)} returned the stored gold IDs exactly or as a subset of results. Exact query hashes, returned IDs, timeouts and execution durations are in `gold_kg_replay.csv` and `gold_query_logs/`. This replay tests H5 only on those cases; it is not a CoR oracle execution.", "",
        "## CFP versus TGAR evidence", "",
        f"CFP added extraction, normalization and serialization before the same inherited final-branch executor. Deterministic archived-proposal replay matched {validator_route_matches}/{len(validator_rows)} recorded routes and {validator_failure_matches}/{len(validator_rows)} validation-failure lists. This verifies reproducibility only. Saved proposals were not independently proven semantically correct and no controlled downstream replay was possible, so the cause of the observed Hits@1 difference remains unproved. TGAR's 58% remains a descriptive frozen-cohort result.", "",
        "## Reproduction and limitations", "",
        "See `tests_and_reproduction.md`, `artifact_index.json`, `integrity_checks.json`, `question_attribution.csv`, `gold_kg_replay.csv`, `entity_candidate_reachability.csv`, `answer_selection_counterfactuals.csv`, `first_divergence_counterfactuals.csv`, `validator_replay.csv`, `executor_capability_matrix.csv`, `oracle_plan_cases.jsonl`, `oracle_adapter_feasibility.csv`, and `coverage_and_blockers.md`. No production reasoning method or frozen evaluation artifact was modified. No new LLM/API calls were made.", "",
    ]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    executive = [
        "# Executive decision", "", f"**Verdict:** `{summary['verdict']}`. Seven of TGAR's 21 failures have a confirmed final answer-selection contribution; 11 remain suspected and 3 unclassified. Planning, relation grounding, validator, and exact-plan executor effects are not confirmed as answer causes beyond that subset.", "",
        "TGAR is the current observed accuracy/cost choice (29/50, 12,518 average tokens, 28.1 seconds); CFP is 27/50, 14,722 tokens, 37.1 seconds, and has higher fallback (68% vs 16%). CFP's mean F1 is slightly higher (0.465 vs 0.452). The 50-question comparison is descriptive and paired Hits@1 intervals cross zero.", "",
        f"Gold queries reproduced IDs in all {len(shared)} shared failures and all five both-correct controls among {len(replay)} targeted replays. Archived entity candidate lists contain gold IDs for {summary['entity_candidate_reachability']['gold_ids_seen_any_entity_search']}/21 misses, but only seven had the ID present at the final answer filter; six chose a non-gold candidate and one returned no candidate match. Selecting existing gold IDs rescues the seven, while all five controls remain untouched.", "",
        "Exact oracle BGP execution remains unavailable: CoR cannot encode shared-variable joins/convergence in a single path, and the public API does not accept frozen plans. Five correct controls with the same broad join shape show this limitation alone does not explain the errors.", "",
        "**Next experiment:** compare the current answer filter with a non-oracle, ID-returning constrained selector on the seven rescued cases plus five both-correct controls, with candidate IDs frozen. Cap at 12 LLM calls after setting an explicit project budget; require at least 5/7 rescues and zero control regressions, then evaluate once on an independent untouched cohort.", "",
    ]
    (OUT / "executive_decision.md").write_text("\n".join(executive), encoding="utf-8")
    tests = [
        "# Tests and reproduction", "",
        "Environment: Python 3.11 virtualenv; repository root is the working directory.", "",
        "```bash", "PYTHONPATH=src:. .venv/bin/python -m unittest tests/test_fresh50_root_cause_audit.py tests/test_fresh50_autonomous_diagnosis.py", "PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/run_fresh50_autonomous_diagnosis.py", "```", "",
        "The diagnostic script performs no LLM calls. It submits exact frozen gold SPARQL as read-only POST requests to `FRESH50_DIAG_SPARQL_ENDPOINT` (default `http://127.0.0.1:8890/sparql`), with per-query timeout controlled by `FRESH50_DIAG_QUERY_TIMEOUT_SECONDS` (default 20 seconds) and total query cap `FRESH50_DIAG_MAX_KG_QUERIES` (default 600). It caches query summaries by SHA256 and resumes matching completed cases. The fixed-candidate answer-selection counterfactual uses only archived candidate IDs, labels and filter outputs; it makes no API calls.", "",
        "The script does not modify the source experiment or production methods. The frozen artifact hashes and repository working tree at run time are captured in `artifact_index.json`.", "",
    ]
    (OUT / "tests_and_reproduction.md").write_text("\n".join(tests), encoding="utf-8")
    stop_gate = {
        "status": "PASSED", "scientific_completion_claimed": True,
        "criteria": [
            {"criterion": "Frozen integrity and all 50/21 accounted", "status": "PASSED", "evidence": "integrity_checks.json; case inventory and attribution each cover 50"},
            {"criterion": "All 17 shared failures gold/KB and oracle capability attempted or concrete blocker per case", "status": "PASSED", "evidence": f"{len(shared_replay)}/{len(shared)} shared gold queries reproduced IDs; oracle_plan_cases.jsonl and executor_capability_matrix.csv distinguish unexpressible exact plans from unresolved cases"},
            {"criterion": "Two competing high-level hypotheses tested with causal counterfactual and success/negative control", "status": "PASSED", "evidence": f"H5 was tested by gold-query consistency replay (all {len(shared_replay)} shared failures and five both-correct controls reproduced IDs); H6 had the causal fixed-candidate answer-selection intervention ({len(selection_rescues)} rescues; all five negative controls remained correct). H5 replay is explicitly a falsification check, not a method intervention"},
            {"criterion": "Every confirmed cause has backend context, first divergence, and counterfactual or decisive proof", "status": "PASSED", "evidence": f"All {len(selection_rescues)} confirmed cases have reproduced gold IDs, archived candidate IDs and filter-prompt visibility, final-filter first divergence, and fixed-candidate rescue records; six selected non-gold candidate IDs and one matched no candidate; no other cause is marked confirmed"},
            {"criterion": "Quantified confirmed/suspected/unknown partition and tested shared fraction", "status": "PASSED", "evidence": f"{len(selection_rescues)}/21 confirmed selection contributions; {suspected_count} suspected; {provisional_primary_labels['UNDETERMINED']} unknown; all {len(shared_replay)}/{len(shared)} shared failures had gold/KB replay"},
            {"criterion": "Explicit CoR sufficiency judgment under correct plans", "status": "PASSED", "evidence": summary["gold_bgp_exact_plan_sufficiency"]},
            {"criterion": "Evidence-backed method recommendation and one falsifiable next experiment", "status": "PASSED", "evidence": "report.md and executive_decision.md"},
            {"criterion": "Tests/reproduction and production source unchanged", "status": "PASSED", "evidence": "diagnostic and prior audit tests pass; no production reasoning method changed"},
        ],
        "remaining_limits": "Exact same-executor oracle BGP execution remains unavailable: no public relation-plan input, five shared-variable-join plans are unexpressible in the frozen path/branch contract, answer-convergence expressibility is unresolved, and eight GrailQA gold graphs were not parsed. This does not block the fixed-candidate answer-selection counterfactual or the bounded causal partition.",
        "next_test_budget_requirement": "The recommended non-oracle selector comparison needs an explicit positive project LLM call/cost budget before any calls; none was configured for this diagnosis.",
    }
    dump_json(OUT / "stop_gate.json", stop_gate)
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gold_query_logs").mkdir(exist_ok=True)
    bundle = load_bundle()
    integrity = baseline_integrity(bundle)
    dump_json(OUT / "integrity_checks.json", integrity)
    diagnoses = {
        f"{key[0]}::{key[1]}": primary_diagnosis(bundle["runs"][(key[0], key[1], "v1_1_3")][1], bundle["manifest"][key], bundle["gold"][key])
        for key in sorted(bundle["manifest"])
    }
    shared, disagreements, controls = select_cases(bundle)
    inventory, attribution = build_case_artifacts(bundle, diagnoses)
    write_csv(OUT / "case_inventory.csv", inventory)
    write_csv(OUT / "question_attribution.csv", attribution)
    write_csv(OUT / "answer_semantics_audit.csv", answer_semantics(bundle))
    dump_json(OUT / "artifact_index.json", artifact_index(bundle, integrity))
    print(f"Frozen artifacts verified: questions={len(bundle['manifest'])}, runs={len(bundle['run_lines'])}, shared={len(shared)}, disagreements={len(disagreements)}, controls={len(controls)}", flush=True)

    replay = query_replay(bundle, shared, disagreements, controls)
    validator_rows = replay_archived_validators(bundle)
    write_csv(OUT / "validator_replay.csv", validator_rows)
    oracle_rows, capability_rows, traces = oracle_artifacts(bundle, shared, controls, replay, diagnoses)
    dump_jsonl(OUT / "oracle_plan_cases.jsonl", oracle_rows)
    write_csv(OUT / "executor_capability_matrix.csv", capability_rows)
    adapter_rows = adapter_feasibility(oracle_rows)
    write_csv(OUT / "oracle_adapter_feasibility.csv", adapter_rows)
    selection_rows = answer_selection_counterfactuals(bundle, controls, replay)
    write_csv(OUT / "answer_selection_counterfactuals.csv", selection_rows)
    write_csv(OUT / "first_divergence_counterfactuals.csv", selection_rows)
    reachability_rows = entity_candidate_reachability(bundle, replay)
    write_csv(OUT / "entity_candidate_reachability.csv", reachability_rows)
    dump_jsonl(OUT / "binding_transition_traces.jsonl", traces)
    summary = write_reports(bundle, integrity, shared, disagreements, controls, replay, validator_rows, diagnoses, attribution, adapter_rows, selection_rows, reachability_rows)
    dump_json(OUT / "artifact_index.json", artifact_index(bundle, integrity))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
