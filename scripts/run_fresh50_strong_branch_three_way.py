#!/usr/bin/env python3
"""Run the fresh 50 strongly branch-required three-way evaluation.

This is orchestration and analysis only.  The original CoR, adaptive v1.1.3,
and Phase 2.5.6 method implementations are imported unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import certifi

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_cor_baseline_experiment import (  # noqa: E402
    evaluate,
    integer_usage,
    load_safe_project_env,
    normalize_topic_entities,
    prediction_names,
    relation_chain_from_trace,
)
from run_phase3c_frozen_generalization import (  # noqa: E402
    STRONG_BENCHMARK,
    SOURCE_MANIFEST,
    benchmark_cases,
    collect_prior_ids,
    example_id,
    raw_cases,
)
from chain_of_relations.kg_backend import get_backend  # noqa: E402
from chain_of_relations.methods.adaptive_v1_1_3.agent import AdaptiveRoutingAgent  # noqa: E402
from chain_of_relations.methods.cor.agent import CoRAgent  # noqa: E402
from chain_of_relations.methods.phase2_5_5.agent import CanonicalNormalizationAgent  # noqa: E402


EXPERIMENT = "fresh50_strong_branch_cor_vs_v113_vs_p256"
OUTPUT_ROOT = ROOT / "outputs/experiments" / EXPERIMENT
SEED = 20261029
METHODS = ("original_cor", "v1_1_3", "phase2_5_6")
STRATA = (
    ("cwq", "BRANCH_FINAL"),
    ("cwq", "BRANCH_INTERMEDIATE"),
    ("grailqa", "BRANCH_FINAL"),
    ("grailqa", "BRANCH_INTERMEDIATE"),
)
REQUESTED_QUOTAS = {
    ("cwq", "BRANCH_FINAL"): 13,
    ("cwq", "BRANCH_INTERMEDIATE"): 12,
    ("grailqa", "BRANCH_FINAL"): 12,
    ("grailqa", "BRANCH_INTERMEDIATE"): 13,
}
EXPECTED_CASES = 50


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_key(case: Mapping[str, Any]) -> str:
    case_id = case.get("case_id", case.get("question_id"))
    if case_id is None:
        raise KeyError("case record has neither case_id nor question_id")
    return f"{case['dataset']}::{case_id}"


def prior_ids_and_sources() -> tuple[set[str], dict[str, Any]]:
    prior, source_counts = collect_prior_ids()
    extra: set[str] = set()
    extra_sources: list[str] = []
    phase3c_manifest = ROOT / "outputs/experiments/phase3c/frozen_branch_required_generalization/phase3c_eval_manifest.json"
    if phase3c_manifest.exists():
        data = read_json(phase3c_manifest)
        ids = {str(item.get("case_id")) for item in data.get("strong_cases", []) + data.get("control_cases", []) if item.get("case_id")}
        extra.update(ids)
        extra_sources.append(str(phase3c_manifest))

    # Selected-ID manifests from earlier experiments are safe to scan.  Source
    # benchmark manifests are deliberately not scanned because they define the
    # eligible population rather than prior evaluation use.
    import re
    pattern = re.compile(r"(?:WebQ(?:Trn|Test)-[A-Za-z0-9_]+|\b[0-9]{10,13}\b)")
    for path in sorted((ROOT / "outputs/experiments").rglob("sample_ids.json")):
        try:
            found = {value for value in pattern.findall(path.read_text(encoding="utf-8", errors="ignore")) if example_id(value)}
        except OSError:
            continue
        if found:
            extra.update(found)
            extra_sources.append(str(path))
    prior.update(extra)
    return prior, {
        "phase3c_and_selected_manifest_extra_count": len(extra),
        "extra_sources": sorted(set(extra_sources)),
        "phase2_to_phase3b_source_counts": source_counts,
    }


def load_fresh_pool() -> tuple[list[dict[str, Any]], dict[str, Any], set[str], dict[str, Any]]:
    source = read_json(SOURCE_MANIFEST)
    raw = raw_cases()
    allowed = set(str(value) for value in source.get("strong_branch_benchmark_ids", []))
    strong = benchmark_cases(STRONG_BENCHMARK, allowed, raw, "STRONGLY_BRANCH_REQUIRED")
    prior, prior_meta = prior_ids_and_sources()
    fresh = [case for case in strong if case["case_id"] not in prior]
    if len(fresh) < EXPECTED_CASES:
        raise RuntimeError(f"fewer than {EXPECTED_CASES} fresh strong cases remain: {len(fresh)}")
    return fresh, source, prior, {"prior": prior_meta, "strong_pool_before_exclusion": len(strong)}


def freeze_manifest() -> dict[str, Any]:
    fresh, source, prior, exclusion_meta = load_fresh_pool()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in fresh:
        groups[(case["dataset"], case["gold_structure"])].append(case)
    for values in groups.values():
        values.sort(key=canonical_key)

    rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    allocation: dict[str, int] = {}
    deficits: dict[str, int] = {}
    for stratum in STRATA:
        quota = REQUESTED_QUOTAS[stratum]
        pool = groups.get(stratum, [])
        take = min(quota, len(pool))
        if take < quota:
            deficits[f"{stratum[0]}|{stratum[1]}"] = quota - take
        selected.extend(rng.sample(pool, take))
        allocation[f"{stratum[0]}|{stratum[1]}"] = take

    # Deterministic redistribution: first preserve dataset balance by filling a
    # short stratum from the same dataset's opposite branch shape; then use
    # remaining capacity in canonical stratum order.
    deficit_total = sum(deficits.values())
    selected_keys = {canonical_key(item) for item in selected}
    redistribution: list[dict[str, Any]] = []
    for missing_stratum, missing_count in sorted(deficits.items()):
        dataset, structure = missing_stratum.split("|", 1)
        preferred = [(dataset, other) for other in ("BRANCH_FINAL", "BRANCH_INTERMEDIATE") if other != structure]
        fallback = [stratum for stratum in STRATA if stratum[0] != dataset]
        order = preferred + fallback
        for target in order:
            if missing_count <= 0:
                break
            pool = [item for item in groups.get(target, []) if canonical_key(item) not in selected_keys]
            if not pool:
                continue
            take = min(missing_count, len(pool))
            chosen = rng.sample(pool, take)
            selected.extend(chosen)
            selected_keys.update(canonical_key(item) for item in chosen)
            key = f"{target[0]}|{target[1]}"
            allocation[key] = allocation.get(key, 0) + take
            redistribution.append({"from_missing_stratum": missing_stratum, "to_stratum": key, "count": take})
            missing_count -= take
    if len(selected) != EXPECTED_CASES or deficit_total != sum(item["count"] for item in redistribution):
        raise RuntimeError(f"deterministic redistribution failed: selected={len(selected)} redistribution={redistribution}")

    selected.sort(key=lambda item: (item["dataset"], item["gold_structure"], item["case_id"]))
    if len({canonical_key(item) for item in selected}) != EXPECTED_CASES:
        raise RuntimeError("fresh manifest contains duplicate dataset-qualified IDs")
    selected_manifest_rows = []
    evaluation_rows = []
    for index, case in enumerate(selected, start=1):
        selected_manifest_rows.append({
            "sample_index": index,
            "dataset": case["dataset"],
            "question_id": case["case_id"],
            "question": case["question"],
            "branch_shape": case["gold_structure"],
            "necessity_label": case.get("necessity", "STRONGLY_BRANCH_REQUIRED"),
            "necessity_strength": case.get("necessity_strength"),
            "phase3a_manifest_source": str(SOURCE_MANIFEST),
            "split": case["split"],
            "validity_gate": case.get("validity_gate"),
            "safety_cap_certification": "POST_PHASE3A_1_FILTERED_FUTURE_MANIFEST",
        })
        evaluation_rows.append({
            "sample_index": index,
            "dataset": case["dataset"],
            "question_id": case["case_id"],
            "gold_answers": case["gold_answers"],
            "answer_ids": case.get("answer_ids", []),
            "gold_sparql": case.get("gold_sparql", ""),
            "logical_form": case.get("logical_form", ""),
            "topic_entities": case.get("topic_entities", []),
            "topic_entity_names": case.get("topic_entity_names", []),
        })

    manifest = {
        "status": "FROZEN",
        "experiment": EXPERIMENT,
        "selection_seed": SEED,
        "sample_size": EXPECTED_CASES,
        "methods": list(METHODS),
        "runs_per_method": 1,
        "expected_run_slots": EXPECTED_CASES * len(METHODS),
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "strong_benchmark_source": str(STRONG_BENCHMARK),
        "strong_benchmark_sha256": sha256(STRONG_BENCHMARK),
        "post_cap_strong_pool_count": len(strong_ids := source.get("strong_branch_benchmark_ids", [])),
        "verified_strong_records_loaded": exclusion_meta["strong_pool_before_exclusion"],
        "prior_id_count": len(prior),
        "fresh_eligible_count": len(fresh),
        "requested_stratum_quotas": {f"{key[0]}|{key[1]}": value for key, value in REQUESTED_QUOTAS.items()},
        "final_stratum_allocation": allocation,
        "redistribution": redistribution,
        "exclusion_metadata": exclusion_meta,
        "selection_policy": "stable dataset-qualified ID sort, seeded sampling, no model outcomes, post-cap STRONGLY_BRANCH_REQUIRED records only",
        "gold_not_in_inference_manifest": True,
        "cases": selected_manifest_rows,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_ROOT / "manifest_frozen.json", manifest)
    write_jsonl(OUTPUT_ROOT / "manifest_frozen.jsonl", selected_manifest_rows)
    write_jsonl(OUTPUT_ROOT / "evaluation_gold.jsonl", evaluation_rows)
    with (OUTPUT_ROOT / "manifest_frozen.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(selected_manifest_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_manifest_rows)
    manifest["manifest_frozen_csv_sha256"] = sha256(OUTPUT_ROOT / "manifest_frozen.csv")
    manifest["manifest_frozen_jsonl_sha256"] = sha256(OUTPUT_ROOT / "manifest_frozen.jsonl")
    manifest["evaluation_gold_sha256"] = sha256(OUTPUT_ROOT / "evaluation_gold.jsonl")
    write_json(OUTPUT_ROOT / "manifest_frozen.json", manifest)
    return manifest


def method_files() -> dict[str, list[Path]]:
    return {
        "original_cor": [ROOT / "chain_of_relations/methods/cor/agent.py", ROOT / "chain_of_relations/methods/cor/prompt.yml"],
        "v1_1_3": [ROOT / "chain_of_relations/methods/adaptive_v1_1_3/agent.py", ROOT / "chain_of_relations/methods/adaptive_v1_1_3/router.py"],
        "phase2_5_6": [ROOT / "chain_of_relations/methods/phase2_5_5/agent.py", ROOT / "chain_of_relations/methods/phase2_5_5/router.py", ROOT / "chain_of_relations/methods/phase2_5_3b"],
    }


def code_fingerprints() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.splitlines()
    except Exception as exc:
        commit, status = "UNAVAILABLE", [f"{type(exc).__name__}: {exc}"]
    files: dict[str, str] = {}
    for method, paths in method_files().items():
        for path in paths:
            if path.is_file():
                files[f"{method}:{path.relative_to(ROOT)}"] = sha256(path)
            elif path.is_dir():
                digest = hashlib.sha256()
                for child in sorted(path.rglob("*")):
                    if child.is_file():
                        digest.update(str(child.relative_to(ROOT)).encode())
                        digest.update(child.read_bytes())
                files[f"{method}:{path.relative_to(ROOT)}/"] = digest.hexdigest()
    return {"git_head": commit, "working_tree_dirty": bool(status), "working_tree_status_count": len(status), "method_file_sha256": files}


def historical_estimate() -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for path in (
        ROOT / "outputs/experiments/phase3c/frozen_branch_required_generalization/cost_analysis.json",
        ROOT / "outputs/experiments/phase3b/frozen_strong_vs_control/cost_analysis.json",
    ):
        if not path.exists():
            continue
        try:
            data = read_json(path)
        except Exception:
            continue
        values.append({"source": str(path), "data": data})
    return {"sources": values, "note": "Descriptive token/call estimate only; no monetary price applied without a verified rate table."}


def endpoint_preflight() -> dict[str, Any]:
    load_safe_project_env()
    result: dict[str, Any] = {
        "model_name": os.getenv("MODEL_NAME", ""),
        "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "openai_base_url": os.getenv("OPENAI_BASE_URL", "") or "https://api.openai.com/v1",
        "freebase_endpoint": os.getenv("FREEBASE_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql"),
        "openai_available": False,
        "freebase_available": False,
    }
    if result["openai_api_key_present"]:
        try:
            request = urllib.request.Request(
                result["openai_base_url"].rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Accept": "application/json"},
            )
            tls_context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(request, timeout=15, context=tls_context) as response:
                payload = json.loads(response.read().decode("utf-8"))
            result["openai_available"] = bool(payload.get("data"))
            result["configured_model_listed"] = any(item.get("id") == result["model_name"] for item in payload.get("data", []))
        except Exception as exc:
            result["openai_error"] = f"{type(exc).__name__}: {exc}"
    endpoint = result["freebase_endpoint"]
    try:
        query = urllib.parse.quote("SELECT * WHERE { ?s ?p ?o } LIMIT 1")
        with urllib.request.urlopen(f"{endpoint}?query={query}&format=json", timeout=15) as response:
            body = response.read(4000).decode("utf-8", "replace")
        result["freebase_available"] = response.status == 200 and "bindings" in body
    except Exception as exc:
        result["freebase_error"] = f"{type(exc).__name__}: {exc}"
    result["ready_for_run"] = bool(result["openai_available"] and result.get("configured_model_listed") and result["freebase_available"])
    return result


def make_agent(method: str) -> Any:
    cor = CoRAgent(
        model_name=os.environ["MODEL_NAME"], relation_width=3, depth=3,
        sample_relation_threshold=1000, remove_unnecessary_rel=True,
        temperature_exploration=0.3, temperature_reasoning=0.1, max_token=1024,
        backend=get_backend("freebase"),
    )
    if method == "original_cor":
        return cor
    if method == "v1_1_3":
        return AdaptiveRoutingAgent(cor, confidence_threshold=0.75, max_branches=3)
    if method == "phase2_5_6":
        return CanonicalNormalizationAgent(cor, confidence_threshold=0.75, max_branches=3)
    raise ValueError(method)


def router_event(history: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    event_type = {
        "v1_1_3": "adaptive_router_v1_1_3",
        "phase2_5_6": "phase2_5_5_canonical_normalization_router",
    }.get(method)
    if not event_type:
        return {}
    for event in reversed(history):
        if event.get("type") == event_type:
            return dict(event)
    return {}


def run_one(agent: Any, method: str, case: Mapping[str, Any], schedule_index: int) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    sparql_history: list[dict[str, Any]] = []
    step_history: list[dict[str, Any]] = []
    started = time.perf_counter()
    error = None
    try:
        topic_entities = normalize_topic_entities({"topic_entities": case["topic_entities"], "topic_entity_names": case["topic_entity_names"]})
        if not topic_entities:
            raise ValueError("selected case has no valid topic entities")
        result = agent.answer(
            question=str(case["question"]), topic_entities=topic_entities,
            prompt_history=history, sparql_history=sparql_history, step_history=step_history,
        )
    except Exception as exc:
        result = {"action": "ERROR", "results": [], "reasoning_chains": ""}
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - started) * 1000.0
    prediction = prediction_names(result)
    hit, f1 = evaluate(prediction, list(case["gold_answers"]))
    native = result.get("adaptive_native_result") if isinstance(result.get("adaptive_native_result"), dict) else result
    native_prediction = prediction_names(native)
    native_hit, native_f1 = evaluate(native_prediction, list(case["gold_answers"]))
    event = router_event(history, method)
    decision = result.get("router_decision", result) if isinstance(result, dict) else {}
    decision = decision if isinstance(decision, dict) else {}
    serialization = event.get("serialization", {}) if isinstance(event.get("serialization"), dict) else {}
    normalization = event.get("normalization", {}) if isinstance(event.get("normalization"), dict) else {}
    branch_attempted = method != "original_cor" and decision.get("router_execution_mode") == "branch"
    fallback = bool(result.get("fallback_used", False))
    native_success = bool(branch_attempted and not fallback and native.get("results"))
    if method == "original_cor":
        execution_status = "COR_EXECUTION" if not error else "RUNTIME_ERROR"
    elif fallback:
        execution_status = "BRANCH_ATTEMPTED_FALLBACK" if branch_attempted else "COR_FALLBACK"
    elif branch_attempted:
        execution_status = "NATIVE_BRANCH_SUCCESS" if native_success else "BRANCH_EXECUTION_FAILURE"
    else:
        execution_status = "LINEAR_COR_PATH"
    return {
        "experiment": EXPERIMENT, "schedule_index": schedule_index, "run_index": 1,
        "method": method, "dataset": case["dataset"], "question_id": case.get("case_id", case.get("question_id")),
        "question_key": canonical_key(case), "question": case["question"], "split": case["split"],
        "prediction": prediction, "normalized_prediction": [str(item).casefold() for item in prediction],
        "hits_at_1": hit, "f1": f1, "native_prediction": native_prediction,
        "native_hits_at_1": native_hit, "native_f1": native_f1,
        "relation_chain": relation_chain_from_trace(result),
        "router_structure": result.get("router_structure", "ORIGINAL_COR" if method == "original_cor" else "UNKNOWN"),
        "router_execution_mode": result.get("router_execution_mode", "cor"),
        "router_confidence": result.get("router_confidence", 0.0),
        "router_reason": result.get("router_reason", ""),
        "validation_passed": result.get("validation_passed", None if method == "original_cor" else False),
        "validation_failures": result.get("validation_failures", event.get("validation_failures", [])),
        "validator_decision": event.get("validator_decision", "NOT_APPLICABLE" if method == "original_cor" else "UNKNOWN"),
        "normalization_status": event.get("normalization_status", "NOT_APPLICABLE" if method == "original_cor" else "UNKNOWN"),
        "serialization_status": serialization.get("serialization_status", "NOT_APPLICABLE" if method == "original_cor" else "UNKNOWN"),
        "canonical_semantic_representation": event.get("canonical_semantic_representation", normalization.get("canonical_proposal", {})),
        "serialized_branches": serialization.get("serialized_branches", []),
        "fallback_used": fallback, "fallback_reason": result.get("fallback_reason"),
        "branch_attempted": branch_attempted, "native_branch_success": native_success,
        "execution_status": execution_status, "branches_created": native.get("branches_created", 0),
        "branch_traces": native.get("branch_traces", []), "branch_failures": native.get("branch_failures", []),
        "merged_candidate_count": native.get("merged_candidate_count", 0),
        "llm_calls": len(history), "kg_queries": len(sparql_history),
        "input_tokens": integer_usage(history, "input_tokens"), "output_tokens": integer_usage(history, "output_tokens"),
        "total_tokens": integer_usage(history, "input_tokens") + integer_usage(history, "output_tokens"),
        "latency_ms": latency_ms, "retry_count": sum(1 for item in history if item.get("retry")),
        "error": error, "prompt_history": history, "sparql_history": sparql_history, "step_history": step_history,
    }


def run_evaluation(manifest: Mapping[str, Any], resume: bool) -> dict[str, Any]:
    preflight = endpoint_preflight()
    write_json(OUTPUT_ROOT / "preflight.json", preflight)
    if not preflight.get("ready_for_run"):
        return {"status": "NOT_RUN_PREFLIGHT_FAILED", "preflight": preflight, "rows": len(read_jsonl(OUTPUT_ROOT / "runs.jsonl")) if (OUTPUT_ROOT / "runs.jsonl").exists() else 0}
    gold_rows = read_jsonl(OUTPUT_ROOT / "evaluation_gold.jsonl")
    gold_by_key = {(row["dataset"], row["question_id"]): row for row in gold_rows}
    cases: list[dict[str, Any]] = []
    for manifest_case in manifest["cases"]:
        gold = gold_by_key.get((manifest_case["dataset"], manifest_case["question_id"]))
        if gold is None:
            raise RuntimeError(f"missing separate evaluation gold for {canonical_key(manifest_case)}")
        merged = dict(manifest_case)
        merged.update(gold)
        cases.append(merged)
    path = OUTPUT_ROOT / "runs.jsonl"
    existing = read_jsonl(path) if resume and path.exists() else []
    done = {(row.get("question_key"), row.get("method"), int(row.get("run_index", 0))) for row in existing}
    agents = {method: make_agent(method) for method in METHODS}
    rows = list(existing)
    schedule_index = len(rows)
    for case in cases:
        for method in METHODS:
            key = (canonical_key(case), method, 1)
            if key in done:
                continue
            schedule_index += 1
            row = run_one(agents[method], method, case, schedule_index)
            rows.append(row)
            done.add(key)
            write_jsonl(path, rows)
            print(json.dumps({"schedule_index": schedule_index, "question_key": key[0], "method": method, "hit": row["hits_at_1"], "error": row["error"]}), flush=True)
    rows.sort(key=lambda row: (int(row.get("schedule_index", 0)), row.get("method", "")))
    write_jsonl(path, rows)
    return {"status": "COMPLETE" if len(rows) == EXPECTED_CASES * len(METHODS) else "INCOMPLETE", "rows": len(rows), "expected_rows": EXPECTED_CASES * len(METHODS), "preflight": preflight}


def case_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["question_key"]), str(row["method"])): row for row in rows}


def bootstrap_mean(values: Sequence[float], seed: int = SEED, samples: int = 5000) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0}
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(list(values), k=len(values))) for _ in range(samples))
    return {"mean": statistics.mean(values), "lower_95": means[int(0.025 * samples)], "upper_95": means[int(0.975 * samples) - 1]}


def paired_comparisons(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by = case_rows(rows)
    questions = sorted({str(row["question_key"]) for row in rows})
    output: dict[str, Any] = {"question_count": len(questions), "comparisons": {}}
    for left, right in (("original_cor", "v1_1_3"), ("original_cor", "phase2_5_6"), ("v1_1_3", "phase2_5_6")):
        records = []
        outcomes = Counter()
        hit_deltas: list[float] = []
        f1_deltas: list[float] = []
        for question in questions:
            a, b = by.get((question, left)), by.get((question, right))
            if not a or not b:
                continue
            ah, bh = int(a.get("hits_at_1", 0)), int(b.get("hits_at_1", 0))
            outcome = "TIE" if ah == bh else f"{left}_ONLY" if ah > bh else f"{right}_ONLY"
            outcomes[outcome] += 1
            hit_deltas.append(bh - ah)
            f1_deltas.append(float(b.get("f1", 0.0)) - float(a.get("f1", 0.0)))
            records.append({"question_key": question, "outcome": outcome, "left_hit": ah, "right_hit": bh, "left_f1": a.get("f1", 0.0), "right_f1": b.get("f1", 0.0)})
        output["comparisons"][f"{left}_vs_{right}"] = {"left": left, "right": right, "counts": dict(outcomes), "mean_hit_delta_right_minus_left": statistics.mean(hit_deltas) if hit_deltas else 0.0, "mean_f1_delta_right_minus_left": statistics.mean(f1_deltas) if f1_deltas else 0.0, "bootstrap_hit_delta_95": bootstrap_mean(hit_deltas), "bootstrap_f1_delta_95": bootstrap_mean(f1_deltas), "records": records}
    return output


def metrics(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"sample_size": len(manifest["cases"]), "methods": {}}
    for method in METHODS:
        values = [row for row in rows if row["method"] == method]
        result["methods"][method] = {
            "run_count": len(values), "correct_questions": sum(int(row.get("hits_at_1", 0)) for row in values),
            "hits_at_1": statistics.mean(int(row.get("hits_at_1", 0)) for row in values) if values else 0.0,
            "mean_f1": statistics.mean(float(row.get("f1", 0.0)) for row in values) if values else 0.0,
            "average_tokens": statistics.mean(int(row.get("total_tokens", 0)) for row in values) if values else 0.0,
            "median_tokens": statistics.median(int(row.get("total_tokens", 0)) for row in values) if values else 0.0,
            "average_latency_ms": statistics.mean(float(row.get("latency_ms", 0.0)) for row in values) if values else 0.0,
            "median_latency_ms": statistics.median(float(row.get("latency_ms", 0.0)) for row in values) if values else 0.0,
            "fallback_count": sum(bool(row.get("fallback_used")) for row in values),
            "fallback_rate": sum(bool(row.get("fallback_used")) for row in values) / len(values) if values else 0.0,
            "error_count": sum(bool(row.get("error")) for row in values),
            "retry_count": sum(int(row.get("retry_count", 0)) for row in values),
        }
    for label, predicate in {
        "cwq": lambda c: c["dataset"] == "cwq",
        "grailqa": lambda c: c["dataset"] == "grailqa",
        "BRANCH_FINAL": lambda c: c["branch_shape"] == "BRANCH_FINAL",
        "BRANCH_INTERMEDIATE": lambda c: c["branch_shape"] == "BRANCH_INTERMEDIATE",
    }.items():
        result[label] = {}
        for method in METHODS:
            keys = {canonical_key(c) for c in manifest["cases"] if predicate(c)}
            values = [row for row in rows if row["method"] == method and row["question_key"] in keys]
            result[label][method] = {"n": len(values), "hits_at_1": statistics.mean(int(row.get("hits_at_1", 0)) for row in values) if values else 0.0, "mean_f1": statistics.mean(float(row.get("f1", 0.0)) for row in values) if values else 0.0}
    return result


def mechanism_audit(rows: Sequence[Mapping[str, Any]], comparisons: Mapping[str, Any]) -> list[dict[str, Any]]:
    by = case_rows(rows)
    records: list[dict[str, Any]] = []
    for question in sorted({str(row["question_key"]) for row in rows}):
        p256, cor, v113 = by.get((question, "phase2_5_6")), by.get((question, "original_cor")), by.get((question, "v1_1_3"))
        if not p256 or not cor or not v113:
            continue
        if int(p256.get("hits_at_1", 0)) and not int(cor.get("hits_at_1", 0)):
            comparator = "original_cor"
        elif int(p256.get("hits_at_1", 0)) and not int(v113.get("hits_at_1", 0)):
            comparator = "v1_1_3"
        elif int(cor.get("hits_at_1", 0)) and not int(p256.get("hits_at_1", 0)):
            comparator = "original_cor"
        elif int(v113.get("hits_at_1", 0)) and not int(p256.get("hits_at_1", 0)):
            comparator = "v1_1_3"
        else:
            continue
        p_success = bool(p256.get("native_branch_success"))
        if int(p256.get("hits_at_1", 0)) and p_success and not p256.get("fallback_used") and p256.get("branch_traces"):
            classification = "VERIFIED_NATIVE_BRANCH_WIN" if not int(by[(question, comparator)].get("hits_at_1", 0)) else "AMBIGUOUS_OR_UNVERIFIED"
        elif p256.get("fallback_used"):
            classification = "FALLBACK_ASSISTED"
        elif int(p256.get("hits_at_1", 0)):
            classification = "LINEAR_OR_GROUNDING_VARIATION"
        else:
            classification = "AMBIGUOUS_OR_UNVERIFIED"
        records.append({"question_key": question, "comparator": comparator, "outcome": "PHASE2_5_6_ONLY" if p256.get("hits_at_1") else "COMPARATOR_ONLY", "classification": classification, "phase2_5_6_execution_status": p256.get("execution_status"), "phase2_5_6_branch_attempted": p256.get("branch_attempted"), "phase2_5_6_native_branch_success": p_success, "phase2_5_6_fallback": p256.get("fallback_used"), "phase2_5_6_prediction": p256.get("prediction"), "comparator_prediction": by[(question, comparator)].get("prediction")})
    return records


def write_analysis(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    metrics_data = metrics(rows, manifest)
    paired = paired_comparisons(rows)
    audit = mechanism_audit(rows, paired)
    write_json(OUTPUT_ROOT / "metrics.json", metrics_data)
    write_json(OUTPUT_ROOT / "paired_comparison.json", paired)
    with (OUTPUT_ROOT / "paired_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["question_key", "left", "right", "outcome", "left_hit", "right_hit", "left_f1", "right_f1"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for comparison in paired["comparisons"].values():
            for record in comparison["records"]:
                writer.writerow({**record, "left": comparison["left"], "right": comparison["right"]})
    with (OUTPUT_ROOT / "mechanism_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for record in audit for key in record}) or ["question_key"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit)
    method_stats = metrics_data["methods"]
    report = [
        f"# Fresh 50 Strong Branch Three-Way Evaluation", "",
        f"Experiment: `{EXPERIMENT}`. The sample contains {len(manifest['cases'])} post-cap strongly branch-required questions and one official run per method.", "",
        "## Executive summary", "",
        "This report compares original CoR, adaptive v1.1.3, and Phase 2.5.6 on the same frozen questions. No method implementation, prompt, routing, validator, fallback, or metric was changed.", "",
        "## Dataset integrity", "",
        f"Post-cap strong records: **{manifest['post_cap_strong_pool_count']}**; fresh eligible after exclusions: **{manifest['fresh_eligible_count']}**; selected: **{len(manifest['cases'])}**.", "",
        f"Sampling seed: `{manifest['selection_seed']}`. Manifest CSV SHA256: `{manifest['manifest_frozen_csv_sha256']}`.", "",
        f"Final strata: `{manifest['final_stratum_allocation']}`. Redistribution: `{manifest['redistribution']}`.", "",
        "Gold answers and programs are stored separately in `evaluation_gold.jsonl` and were not passed to inference.", "",
        "## Three-way performance", "",
        "| Method | Hits@1 | Correct/50 | Mean F1 | Avg tokens | Avg latency ms | Fallbacks | Errors |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = method_stats[method]
        report.append(f"| {method} | {item['hits_at_1']:.3f} | {item['correct_questions']}/50 | {item['mean_f1']:.3f} | {item['average_tokens']:.1f} | {item['average_latency_ms']:.1f} | {item['fallback_count']} ({item['fallback_rate']:.3f}) | {item['error_count']} |")
    report += ["", "## Paired comparisons", "", "See `paired_comparison.json` and `paired_comparison.csv`. Results are single-run question-level comparisons; no repeat or majority aggregation is used.", ""]
    for name, item in paired["comparisons"].items():
        report.append(f"- `{name}`: {item['counts']}; mean H@1 delta (right-left) `{item['mean_hit_delta_right_minus_left']:+.3f}`; bootstrap 95% CI `{item['bootstrap_hit_delta_95']}`.")
    report += ["", "## Mechanism and branch benefit", "", f"Mechanism audit records: **{len(audit)}**. Verified native branch-only classifications: **{sum(row['classification'] == 'VERIFIED_NATIVE_BRANCH_WIN' for row in audit)}**. A branch router decision alone was not treated as evidence of causal benefit.", "", "## Exploratory breakdowns", ""]
    for label in ("cwq", "grailqa", "BRANCH_FINAL", "BRANCH_INTERMEDIATE"):
        report.append(f"### {label}")
        for method in METHODS:
            item = metrics_data[label][method]
            report.append(f"- {method}: n={item['n']}, H@1={item['hits_at_1']:.3f}, F1={item['mean_f1']:.3f}")
    report += ["", "## Limitations", "", "This is a fresh 50-question sample with one run per method per question. It does not establish LLM repeatability. Subgroup analyses are exploratory, and any unresolved prior-ID coverage is recorded in `sampling_audit.json`.", "", "## Reproduction", "", "```bash", "PYTHONPATH=src:. .venv/bin/python scripts/run_fresh50_strong_branch_three_way.py --select", "PYTHONPATH=src:. .venv/bin/python scripts/run_fresh50_strong_branch_three_way.py --preflight", "PYTHONPATH=src:. .venv/bin/python scripts/run_fresh50_strong_branch_three_way.py --run --resume", "```", ""]
    (OUTPUT_ROOT / "report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "manifest_frozen.json"
    if args.select or not manifest_path.exists():
        manifest = freeze_manifest()
        write_json(OUTPUT_ROOT / "sampling_audit.json", {"manifest": manifest, "code_fingerprints": code_fingerprints(), "historical_estimate": historical_estimate()})
        print(json.dumps({"status": "SELECTION_FROZEN", "sample_size": len(manifest["cases"]), "final_strata": manifest["final_stratum_allocation"], "manifest": str(manifest_path)}, indent=2))
        if not (args.preflight or args.run):
            return
    manifest = read_json(manifest_path)
    if args.preflight or args.run:
        preflight = endpoint_preflight()
        write_json(OUTPUT_ROOT / "preflight.json", preflight)
        print(json.dumps(preflight, indent=2))
        if args.preflight and not args.run:
            return
    if not args.run:
        return
    run_config = {
        "experiment": EXPERIMENT, "manifest_sha256": sha256(manifest_path), "methods": list(METHODS),
        "method_fidelity": {"original_cor": "CoRAgent directly; not adaptive LINEAR", "v1_1_3": "AdaptiveRoutingAgent", "phase2_5_6": "CanonicalNormalizationAgent"},
        "shared_cor_settings": {"depth": 3, "relation_width": 3, "sample_relation_threshold": 1000, "remove_unnecessary_rel": True, "temperature_exploration": 0.3, "temperature_reasoning": 0.1, "max_token": 1024, "confidence_threshold": 0.75, "max_branches": 3},
        "model_name": os.getenv("MODEL_NAME", ""), "openai_base_url": os.getenv("OPENAI_BASE_URL", "") or "https://api.openai.com/v1", "freebase_endpoint": os.getenv("FREEBASE_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql"),
        "retry_settings": {"max_retries": os.getenv("OPENAI_MAX_RETRIES", "3"), "retry_interval": os.getenv("OPENAI_RETRY_INTERVAL", "1"), "timeout": os.getenv("OPENAI_TIMEOUT", "30")},
        "schedule": "question-major interleaved original_cor, v1_1_3, phase2_5_6; one run per slot; checkpoint after every row",
        "gold_not_in_inference": True, "code_fingerprints": code_fingerprints(), "historical_estimate": historical_estimate(),
    }
    write_json(OUTPUT_ROOT / "run_config.json", run_config)
    execution = run_evaluation(manifest, args.resume)
    write_json(OUTPUT_ROOT / "execution_summary.json", execution)
    rows = read_jsonl(OUTPUT_ROOT / "runs.jsonl") if (OUTPUT_ROOT / "runs.jsonl").exists() else []
    if rows:
        write_analysis(manifest, rows)
    print(json.dumps(execution, indent=2))


if __name__ == "__main__":
    main()
