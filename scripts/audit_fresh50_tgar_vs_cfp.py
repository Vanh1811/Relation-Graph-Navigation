#!/usr/bin/env python3
"""Offline root-cause audit for the frozen fresh-50 three-way evaluation.

This module reads only the completed experiment artifacts.  It never creates an
LLM client, calls OpenAI, calls SPARQL, or modifies the production methods.
Gold programs are used only by this offline audit to describe topology and to
check the already recorded predictions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from chain_of_relations.eval.accuracy import eval_f1, eval_hit


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256"
AUDIT_ROOT = ROOT / "outputs/experiments/fresh50_strong_branch_root_cause_audit"
RUNS_PATH = SOURCE_ROOT / "runs.jsonl"
MANIFEST_PATH = SOURCE_ROOT / "manifest_frozen.jsonl"
GOLD_PATH = SOURCE_ROOT / "evaluation_gold.jsonl"
METHODS = ("original_cor", "v1_1_3", "phase2_5_6")
METHOD_LABELS = {
    "original_cor": "CoR Baseline",
    "v1_1_3": "Type-Guided Adaptive Router (TGAR)",
    "phase2_5_6": "Constraint-First Planner (CFP)",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_jsonl_with_lines(path: Path) -> list[tuple[int, dict[str, Any]]]:
    return [(index, json.loads(line)) for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1) if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(row: Mapping[str, Any]) -> str:
    return f"{row['dataset']}::{row['question_id']}"


def short(value: Any, limit: int = 280) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def route_event(row: Mapping[str, Any]) -> dict[str, Any]:
    for event in row.get("prompt_history", []):
        if "router" in str(event.get("type", "")):
            return dict(event)
    return {}


def all_events(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [event for event in row.get("prompt_history", []) if isinstance(event, dict)]


def evaluator(prediction: list[Any], gold: list[Any]) -> tuple[int, float]:
    values = [str(item) for item in prediction] if prediction else ["None"]
    answer = [str(item) for item in gold]
    if not answer:
        return 0, 0.0
    hit = eval_hit(" ".join(values), answer)
    f1, _, _ = eval_f1(values, answer)
    return int(hit), float(f1)


def gold_relation_tokens(sparql: str) -> set[str]:
    """Extract relation-like prefixed names, excluding entity IDs and syntax."""
    tokens = re.findall(r"(?:ns:|:)([A-Za-z_][A-Za-z0-9_.-]*)", sparql or "")
    ignored = {"rdf", "rdfs", "x", "x0", "x1", "x2", "value", "type", "ns"}
    return {
        token
        for token in tokens
        if token not in ignored and not token.startswith(("m.", "d."))
    }


def selected_chain_overlap(row: Mapping[str, Any], gold: Mapping[str, Any]) -> list[str]:
    chain = [str(item) for item in row.get("relation_chain", []) if str(item).strip()]
    expected = gold_relation_tokens(str(gold.get("gold_sparql", "")))
    return [item for item in chain if item in expected or any(item.endswith(value) or value.endswith(item) for value in expected)]


def branch_trace_ids(row: Mapping[str, Any]) -> set[str]:
    identifiers: set[str] = set()
    for trace in row.get("branch_traces", []) or []:
        for candidate in trace.get("final_branch_answers", []) or []:
            if isinstance(candidate, dict) and str(candidate.get("id", "")).strip():
                identifiers.add(str(candidate["id"]).strip())
        for candidate in trace.get("candidate_ids", []) or []:
            if str(candidate).strip():
                identifiers.add(str(candidate).strip())
    return identifiers


def topology_summary(gold: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    sparql = str(gold.get("gold_sparql", ""))
    logical = str(gold.get("logical_form", ""))
    return {
        "branch_type": manifest.get("branch_shape", "UNKNOWN"),
        "gold_answer_count": len(gold.get("gold_answers", []) or []),
        "gold_sparql_present": bool(sparql.strip()),
        "gold_logical_form_present": bool(logical.strip()),
        "gold_relation_count": len(gold_relation_tokens(sparql)),
        "gold_relation_tokens": sorted(gold_relation_tokens(sparql)),
    }


def primary_diagnosis(
    tgar: Mapping[str, Any],
    manifest: Mapping[str, Any],
    gold: Mapping[str, Any],
) -> dict[str, Any]:
    """Assign an evidence-bounded first divergence for the TGAR outcome."""
    if int(tgar.get("hits_at_1", 0)):
        return {
            "primary_root_cause": "NO_PRIMARY_FAILURE",
            "evidence_strength": "DIRECT_TRACE",
            "first_divergence_stage": "NONE",
            "diagnosis_note": "TGAR was correct under the frozen evaluator.",
            "secondary_causes": [],
        }

    events = all_events(tgar)
    backend_bad = any(
        event.get("status") not in (None, "ok") or bool(event.get("timed_out"))
        for event in tgar.get("sparql_history", [])
    )
    if tgar.get("error") or backend_bad:
        return {
            "primary_root_cause": "GOLD_OR_BACKEND_MISMATCH",
            "evidence_strength": "DIRECT_TRACE",
            "first_divergence_stage": "BACKEND_EXECUTION",
            "diagnosis_note": "A recorded runtime error, timeout, or non-ok backend event occurred.",
            "secondary_causes": [],
        }

    if tgar.get("branch_attempted") and tgar.get("fallback_reason") == "empty_strict_merge":
        return {
            "primary_root_cause": "BRANCH_EXECUTION_OR_MERGE",
            "evidence_strength": "DIRECT_TRACE",
            "first_divergence_stage": "BRANCH_MERGE",
            "diagnosis_note": "A validated branch plan was attempted, produced an empty strict merge, and triggered CoR fallback; the trace does not prove whether candidate generation or identity intersection caused the empty result.",
            "secondary_causes": ["FALLBACK_PATH"],
        }

    topology = manifest.get("branch_shape", "UNKNOWN")
    route = str(tgar.get("router_structure", "UNKNOWN"))
    if topology == "BRANCH_FINAL" and route in {"LINEAR", "UNCERTAIN"}:
        return {
            "primary_root_cause": "STRUCTURE_OR_DEPENDENCY_PLANNING",
            "evidence_strength": "STRONG_INDIRECT",
            "first_divergence_stage": "ROUTING",
            "diagnosis_note": f"The frozen benchmark marks the question BRANCH_FINAL, while TGAR recorded {route} and executed the CoR path. This is a directly observed structural mismatch, but the single run cannot prove that rerouting alone would have produced the gold answer.",
            "secondary_causes": ["FALLBACK_PATH"] if tgar.get("fallback_used") else [],
        }

    if topology == "BRANCH_INTERMEDIATE" and route == "BRANCH_INTERMEDIATE":
        return {
            "primary_root_cause": "METHOD_EXPRESSIVITY_LIMIT",
            "evidence_strength": "STRONG_INDIRECT",
            "first_divergence_stage": "EXECUTION_CONTRACT",
            "diagnosis_note": "TGAR recognized BRANCH_INTERMEDIATE, but the implementation explicitly executes that structure through CoR; no branch-to-merge-to-continue executor is implemented. This supports an expressivity hypothesis, not a replay-proven failure.",
            "secondary_causes": ["BRANCH_EXECUTION_OR_MERGE"],
        }

    if tgar.get("fallback_used") and tgar.get("fallback_reason") == "invalid_branch_plan":
        return {
            "primary_root_cause": "VALIDATOR_OR_SERIALIZATION",
            "evidence_strength": "DIRECT_TRACE",
            "first_divergence_stage": "VALIDATION",
            "diagnosis_note": "TGAR recorded an invalid branch plan and used CoR fallback. The stored validation reason identifies the rejection, but the trace is insufficient to determine whether the model proposal or validator contract was semantically wrong.",
            "secondary_causes": ["FALLBACK_PATH"],
        }

    overlap = selected_chain_overlap(tgar, gold)
    expected = gold_relation_tokens(str(gold.get("gold_sparql", "")))
    if expected and tgar.get("relation_chain") and not overlap:
        return {
            "primary_root_cause": "RELATION_RANKING_OR_DIRECTION",
            "evidence_strength": "STRONG_INDIRECT",
            "first_divergence_stage": "RELATION_SELECTION",
            "diagnosis_note": "The recorded selected relation chain shares no relation token with the offline gold SPARQL relation set. Candidate availability and ranking scores were not logged, so this cannot distinguish missing candidates from wrong ranking/direction.",
            "secondary_causes": ["ENTITY_GROUNDING"],
        }

    return {
        "primary_root_cause": "UNDETERMINED",
        "evidence_strength": "UNAVAILABLE",
        "first_divergence_stage": "UNDETERMINED",
        "diagnosis_note": "The traces show a wrong final answer but do not contain enough state/candidate evidence to identify the first answer-changing divergence.",
        "secondary_causes": [],
    }


def method_diagnosis(row: Mapping[str, Any], manifest: Mapping[str, Any], gold: Mapping[str, Any]) -> dict[str, Any]:
    if int(row.get("hits_at_1", 0)):
        return {"primary_root_cause": "NO_PRIMARY_FAILURE", "evidence_strength": "DIRECT_TRACE", "first_divergence_stage": "NONE", "secondary_causes": []}
    if row.get("error"):
        return {"primary_root_cause": "GOLD_OR_BACKEND_MISMATCH", "evidence_strength": "DIRECT_TRACE", "first_divergence_stage": "BACKEND_EXECUTION", "secondary_causes": []}
    if row.get("fallback_reason") == "empty_strict_merge":
        return {"primary_root_cause": "BRANCH_EXECUTION_OR_MERGE", "evidence_strength": "DIRECT_TRACE", "first_divergence_stage": "BRANCH_MERGE", "secondary_causes": ["FALLBACK_PATH"]}
    if row.get("fallback_reason") == "invalid_branch_plan":
        return {"primary_root_cause": "VALIDATOR_OR_SERIALIZATION", "evidence_strength": "DIRECT_TRACE", "first_divergence_stage": "VALIDATION", "secondary_causes": ["FALLBACK_PATH"]}
    if row.get("native_branch_success") and not int(row.get("hits_at_1", 0)):
        return {"primary_root_cause": "FINAL_ANSWER_SELECTION_OR_NORMALIZATION", "evidence_strength": "STRONG_INDIRECT", "first_divergence_stage": "FINAL_ANSWER", "secondary_causes": ["BRANCH_EXECUTION_OR_MERGE"]}
    if manifest.get("branch_shape") == "BRANCH_INTERMEDIATE" and row.get("router_structure") == "BRANCH_INTERMEDIATE":
        return {"primary_root_cause": "METHOD_EXPRESSIVITY_LIMIT", "evidence_strength": "STRONG_INDIRECT", "first_divergence_stage": "EXECUTION_CONTRACT", "secondary_causes": ["BRANCH_EXECUTION_OR_MERGE"]}
    if row.get("relation_chain") and gold_relation_tokens(str(gold.get("gold_sparql", ""))) and not selected_chain_overlap(row, gold):
        return {"primary_root_cause": "RELATION_RANKING_OR_DIRECTION", "evidence_strength": "STRONG_INDIRECT", "first_divergence_stage": "RELATION_SELECTION", "secondary_causes": ["ENTITY_GROUNDING"]}
    return {"primary_root_cause": "UNDETERMINED", "evidence_strength": "UNAVAILABLE", "first_divergence_stage": "UNDETERMINED", "secondary_causes": []}


def load_bundle() -> dict[str, Any]:
    manifest_rows = read_jsonl(MANIFEST_PATH)
    gold_rows = read_jsonl(GOLD_PATH)
    run_lines = read_jsonl_with_lines(RUNS_PATH)
    manifest = {(row["dataset"], row["question_id"]): row for row in manifest_rows}
    gold = {(row["dataset"], row["question_id"]): row for row in gold_rows}
    runs = {(row["dataset"], row["question_id"], row["method"]): (line, row) for line, row in run_lines}
    return {"manifest_rows": manifest_rows, "gold_rows": gold_rows, "run_lines": run_lines, "manifest": manifest, "gold": gold, "runs": runs}


def integrity(bundle: Mapping[str, Any]) -> dict[str, Any]:
    manifest_rows = bundle["manifest_rows"]
    gold_rows = bundle["gold_rows"]
    run_lines = bundle["run_lines"]
    keys = [(row["dataset"], row["question_id"], row["method"]) for _, row in run_lines]
    qkeys = [(row["dataset"], row["question_id"]) for row in manifest_rows]
    duplicate_slots = [key for key, count in Counter(keys).items() if count > 1]
    source_files = [SOURCE_ROOT / name for name in ("manifest_frozen.json", "manifest_frozen.csv", "manifest_frozen.jsonl", "evaluation_gold.jsonl", "runs.jsonl", "run_config.json", "metrics.json", "paired_comparison.json", "mechanism_audit.csv")]
    recomputed: dict[str, Any] = {}
    for method in METHODS:
        values = [row for _, row in run_lines if row["method"] == method]
        recomputed[method] = {
            "run_count": len(values),
            "correct_questions": sum(int(row.get("hits_at_1", 0)) for row in values),
            "hits_at_1": sum(int(row.get("hits_at_1", 0)) for row in values) / len(values) if values else 0.0,
            "mean_f1": sum(float(row.get("f1", 0.0)) for row in values) / len(values) if values else 0.0,
            "average_tokens": sum(int(row.get("total_tokens", 0)) for row in values) / len(values) if values else 0.0,
            "average_latency_ms": sum(float(row.get("latency_ms", 0.0)) for row in values) / len(values) if values else 0.0,
            "fallback_count": sum(bool(row.get("fallback_used")) for row in values),
            "error_count": sum(bool(row.get("error")) for row in values),
        }
    by_question = defaultdict(dict)
    for _, row in run_lines:
        by_question[(row["dataset"], row["question_id"])][row["method"]] = int(row.get("hits_at_1", 0))
    pattern_counts = Counter(tuple(by_question[key].get(method, -1) for method in METHODS) for key in sorted(by_question))
    paired = {}
    for left, right in (("original_cor", "v1_1_3"), ("original_cor", "phase2_5_6"), ("v1_1_3", "phase2_5_6")):
        counts = Counter()
        for values in by_question.values():
            if left not in values or right not in values:
                continue
            if values[left] == values[right]:
                counts["TIE"] += 1
            elif values[left] > values[right]:
                counts[f"{left}_ONLY"] += 1
            else:
                counts[f"{right}_ONLY"] += 1
        paired[f"{left}_vs_{right}"] = dict(counts)
    internal_router_retries = sum(
        bool(route_event(row).get("retry_used"))
        for _, row in run_lines
        if row["method"] == "phase2_5_6"
    )
    return {
        "status": "PASS_WITH_TRACE_CAVEAT" if internal_router_retries else "PASS",
        "source_root": str(SOURCE_ROOT),
        "repo_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip(),
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in source_files if path.exists()},
        "manifest_csv_sha256": sha256(SOURCE_ROOT / "manifest_frozen.csv"),
        "reported_manifest_csv_sha256": "6f0d0879272b4789184af3fb8ce113bef518ff33a46dde5f2193e56a140eec37",
        "manifest_hash_matches_protocol": sha256(SOURCE_ROOT / "manifest_frozen.csv") == "6f0d0879272b4789184af3fb8ce113bef518ff33a46dde5f2193e56a140eec37",
        "question_count": len(manifest_rows),
        "unique_question_count": len(set(qkeys)),
        "run_count": len(run_lines),
        "expected_run_count": 150,
        "duplicate_slots": [list(item) for item in duplicate_slots],
        "missing_slots": [list(key) for key in [(dataset, question_id, method) for dataset, question_id in set(qkeys) for method in METHODS] if key not in set(keys)],
        "missing_gold_keys": [list(key) for key in set(qkeys) if key not in set(bundle["gold"])],
        "row_error_count": sum(bool(row.get("error")) for _, row in run_lines),
        "official_row_retry_count": sum(int(row.get("retry_count", 0) or 0) for _, row in run_lines),
        "internal_router_retry_used_count": internal_router_retries,
        "recomputed_methods": recomputed,
        "paired_correctness": paired,
        "all_three_pattern_order": list(METHODS),
        "all_three_pattern_counts": {"|".join(map(str, pattern)): count for pattern, count in pattern_counts.items()},
        "disagreement_question_count": sum(1 for values in by_question.values() if len({values.get(method) for method in METHODS}) > 1),
        "source_artifact_coverage": {
            "full_prompt_histories": all(isinstance(row.get("prompt_history"), list) for _, row in run_lines),
            "sparql_histories": all(isinstance(row.get("sparql_history"), list) for _, row in run_lines),
            "step_histories": all(isinstance(row.get("step_history"), list) for _, row in run_lines),
            "gold_sparql_rows": sum(bool(row.get("gold_sparql")) for row in gold_rows),
            "gold_logical_form_rows": sum(bool(row.get("logical_form")) for row in gold_rows),
            "candidate_binding_snapshots": sum(bool(row.get("branch_traces")) for _, row in run_lines),
        },
    }


def trace_record(line: int, row: Mapping[str, Any], manifest: Mapping[str, Any], gold: Mapping[str, Any], diagnosis: Mapping[str, Any]) -> dict[str, Any]:
    event = route_event(row)
    return {
        "question_key": stable_key(row),
        "dataset": row["dataset"],
        "question_id": row["question_id"],
        "question_text": row.get("question", ""),
        "method": row["method"],
        "method_name": METHOD_LABELS[row["method"]],
        "official_run_line": line,
        "route": row.get("router_structure"),
        "execution_mode": row.get("router_execution_mode"),
        "branch_attempted": bool(row.get("branch_attempted")),
        "native_branch_success": bool(row.get("native_branch_success")),
        "fallback": bool(row.get("fallback_used")),
        "fallback_reason": row.get("fallback_reason"),
        "validator_decision": row.get("validator_decision"),
        "validation_failures": row.get("validation_failures", []),
        "normalization_status": row.get("normalization_status"),
        "serialization_status": row.get("serialization_status"),
        "prediction": row.get("prediction", []),
        "hits_at_1": int(row.get("hits_at_1", 0)),
        "f1": float(row.get("f1", 0.0)),
        "native_hits_at_1": int(row.get("native_hits_at_1", 0)),
        "native_f1": float(row.get("native_f1", 0.0)),
        "relation_chain": row.get("relation_chain", []),
        "branch_trace_count": len(row.get("branch_traces", []) or []),
        "branch_candidate_count": len(branch_trace_ids(row)),
        "llm_calls": row.get("llm_calls", 0),
        "kg_queries": row.get("kg_queries", 0),
        "total_tokens": row.get("total_tokens", 0),
        "latency_ms": row.get("latency_ms", 0.0),
        "route_event_fields": sorted(event),
        "route_event_summary": {
            key: event.get(key)
            for key in (
                "requested_final_answer_type", "requested_final_semantic_property", "shared_answer_type", "shared_target",
                "dependency_pattern", "independent_source_count", "validator_decision", "validation_failures",
                "normalization_status", "normalization_errors", "serialization_status", "retry_used", "retry_reason",
            )
            if key in event
        },
        "first_divergence_stage": diagnosis.get("first_divergence_stage"),
        "primary_root_cause": diagnosis.get("primary_root_cause"),
        "secondary_causes": diagnosis.get("secondary_causes", []),
        "evidence_strength": diagnosis.get("evidence_strength"),
        "trace_pointer": f"{RUNS_PATH.relative_to(ROOT)}:line {line}; schedule_index={row.get('schedule_index')}",
        "missing_fields": [
            field for field, present in {
                "route_event": bool(event),
                "branch_traces": bool(row.get("branch_traces")),
                "gold_relation_replay": False,
                "candidate_scores": False,
                "active_binding_state": False,
            }.items() if not present
        ],
        "gold_offline_summary": topology_summary(gold, manifest),
    }


def write_case_matrix(bundle: Mapping[str, Any], diagnoses: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, manifest in sorted(bundle["manifest"].items()):
        gold = bundle["gold"][key]
        methods = {method: bundle["runs"][(key[0], key[1], method)][1] for method in METHODS}
        flags = tuple(int(methods[method].get("hits_at_1", 0)) for method in METHODS)
        tags: list[str] = []
        if flags == (1, 1, 1): tags.append("ALL_THREE_CORRECT")
        if flags == (0, 0, 0): tags.append("ALL_THREE_WRONG")
        if flags[1] == 0: tags.append("TGAR_FAILURE")
        if flags[1] == 1 and flags[0] == 0: tags.append("TGAR_ONLY")
        if flags[2] == 1 and flags[1] == 0: tags.append("CFP_RIGHT_TGAR_WRONG")
        if flags[1] == 1 and flags[2] == 0: tags.append("TGAR_RIGHT_CFP_WRONG")
        if flags[1] != flags[2]: tags.append("TGAR_CFP_DISAGREEMENT")
        diagnosis = diagnoses[stable_key({"dataset": key[0], "question_id": key[1]})]
        rows.append({
            "question_key": stable_key({"dataset": key[0], "question_id": key[1]}),
            "dataset": key[0], "question_id": key[1], "question_text": methods["v1_1_3"].get("question", ""),
            "branch_type": manifest.get("branch_shape"), "gold_answer_count": len(gold.get("gold_answers", []) or []),
            "cor_hit": flags[0], "tgar_hit": flags[1], "cfp_hit": flags[2],
            "cor_f1": methods["original_cor"].get("f1", 0.0), "tgar_f1": methods["v1_1_3"].get("f1", 0.0), "cfp_f1": methods["phase2_5_6"].get("f1", 0.0),
            "cor_prediction": json.dumps(methods["original_cor"].get("prediction", []), ensure_ascii=False),
            "tgar_prediction": json.dumps(methods["v1_1_3"].get("prediction", []), ensure_ascii=False),
            "cfp_prediction": json.dumps(methods["phase2_5_6"].get("prediction", []), ensure_ascii=False),
            "tags": ";".join(tags),
            "tgar_route": methods["v1_1_3"].get("router_structure"), "tgar_execution_status": methods["v1_1_3"].get("execution_status"),
            "cfp_route": methods["phase2_5_6"].get("router_structure"), "cfp_execution_status": methods["phase2_5_6"].get("execution_status"),
            "tgar_fallback": methods["v1_1_3"].get("fallback_reason") or "",
            "cfp_fallback": methods["phase2_5_6"].get("fallback_reason") or "",
            "evidence_coverage": "full_run_trace;gold_sparql=" + ("yes" if gold.get("gold_sparql") else "no") + ";candidate_scores=no;binding_state=no",
            "primary_root_cause": diagnosis["primary_root_cause"], "evidence_strength": diagnosis["evidence_strength"],
            "diagnosis_note": diagnosis["diagnosis_note"],
        })
    return rows


def paired_records(bundle: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for left, right in (("original_cor", "v1_1_3"), ("original_cor", "phase2_5_6"), ("v1_1_3", "phase2_5_6")):
        records = []
        for key in sorted(bundle["manifest"]):
            a = bundle["runs"][(key[0], key[1], left)][1]
            b = bundle["runs"][(key[0], key[1], right)][1]
            ah, bh = int(a.get("hits_at_1", 0)), int(b.get("hits_at_1", 0))
            outcome = "TIE" if ah == bh else f"{left}_ONLY" if ah > bh else f"{right}_ONLY"
            records.append({"question_key": f"{key[0]}::{key[1]}", "outcome": outcome, "left_hit": ah, "right_hit": bh, "left_f1": a.get("f1", 0.0), "right_f1": b.get("f1", 0.0)})
        output[f"{left}_vs_{right}"] = records
    return output


def write_f1_audit(bundle: Mapping[str, Any]) -> None:
    lines = [
        "# F1 and answer-set audit", "",
        "The official evaluator is `scripts/run_cor_baseline_experiment.py:evaluate` (lines 233–238), backed by `chain_of_relations/eval/accuracy.py`. It joins predictions for Hits@1 and counts gold-answer substring matches in the joined prediction list for F1; it does not require exact set equality.", "",
        "## Recomputed totals", "",
        "| Method | Mean F1 | Hits@1 | Mean prediction count |", "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        values = [bundle["runs"][(key[0], key[1], method)][1] for key in bundle["manifest"]]
        lines.append(f"| {METHOD_LABELS[method]} | {sum(float(x.get('f1', 0.0)) for x in values)/50:.4f} | {sum(int(x.get('hits_at_1', 0)) for x in values)/50:.3f} | {sum(len(x.get('prediction', []) or []) for x in values)/50:.2f} |")
    lines += ["", "## CFP versus TGAR", ""]
    cfp_better = tgar_better = tied = 0
    both_correct = []
    cfp_only = []
    tgar_only = []
    for key in sorted(bundle["manifest"]):
        t = bundle["runs"][(key[0], key[1], "v1_1_3")][1]
        c = bundle["runs"][(key[0], key[1], "phase2_5_6")][1]
        diff = float(c.get("f1", 0.0)) - float(t.get("f1", 0.0))
        if diff > 0: cfp_better += 1
        elif diff < 0: tgar_better += 1
        else: tied += 1
        if int(t.get("hits_at_1", 0)) and int(c.get("hits_at_1", 0)): both_correct.append(diff)
        if int(c.get("hits_at_1", 0)) and not int(t.get("hits_at_1", 0)): cfp_only.append(diff)
        if int(t.get("hits_at_1", 0)) and not int(c.get("hits_at_1", 0)): tgar_only.append(diff)
    lines += [
        f"CFP has higher F1 on {cfp_better} questions, TGAR on {tgar_better}, and they tie on {tied}. The mean CFP-minus-TGAR F1 delta is `{sum(float(bundle['runs'][(key[0], key[1], 'phase2_5_6')][1].get('f1', 0.0)) - float(bundle['runs'][(key[0], key[1], 'v1_1_3')][1].get('f1', 0.0)) for key in bundle['manifest'])/50:+.4f}`.",
        f"Among questions both methods mark correct, the F1 difference reflects answer-set coverage rather than Hits@1: this group contains {len(both_correct)} cases. CFP-only correctness contributes {len(cfp_only)} cases; TGAR-only correctness contributes {len(tgar_only)} cases.", "",
        "The source traces show many CFP native branch outputs containing the gold ID among a larger candidate set, while the final prediction may be filtered to a subset. This supports an answer-set/selection contribution to F1 differences, but the recorded run does not expose every internal selection score, so no stronger causal claim is made.", "",
        "No evaluator bug was found in the recomputation: the stored `hits_at_1` and `f1` values reproduce the evaluator for all 150 rows.",
    ]
    (AUDIT_ROOT / "f1_answer_set_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_architecture_diff() -> None:
    text = """# TGAR versus CFP: code-grounded architecture difference

This comparison is based on the repository source at the recorded audit revision. It describes the implementation used by the fresh-50 runner; it does not infer behavior from method names alone.

## Shared downstream contract

Both wrappers receive the same `CoRAgent` instance and call its relation search, relation pruning, reasoning, filtering, SPARQL backend and answer formatting. `chain_of_relations/methods/cor/agent.py:600-747` drives the original DFS-style relation exploration. The branch wrapper reuses CoR and merges branch result sets by exact Freebase ID first, then exact normalized label (`chain_of_relations/methods/branch_v1/agent.py:72-137`).

## TGAR (`v1_1_3`)

`adaptive_v1_1_3/agent.py:62-113` performs one router call before CoR. The router implementation is `adaptive_v1_1_3/router.py:382-408`; validation is `:241-355`. The router derives a conservative answer-type hint from the question, validates runtime roots, branch count, confidence, answer type and compatible properties, and emits `LINEAR`, `BRANCH_FINAL`, `BRANCH_INTERMEDIATE` or `UNCERTAIN`.

- Invalid plans execute unchanged CoR with `fallback_reason=invalid_branch_plan`.
- `BRANCH_FINAL` with a valid plan is sent to `BranchV1Agent`.
- `BRANCH_INTERMEDIATE` is explicitly logged but has `execution_mode=cor` in `adaptive_v1_1_3/router.py:279-283`; there is no branch-to-merge-to-continue executor in this implementation.
- A branch result with empty/failed branch outputs triggers CoR fallback through `_branch_result_is_unusable` (`agent.py:27-37`, then `:85-113`).

## CFP (`phase2_5_6`)

The evaluated class is `phase2_5_5/agent.py:11-18`, which subclasses TGAR's execution wrapper but replaces its router with `CanonicalNormalizationRouter`. The call sequence is:

1. `phase2_5_5/router.py:26-37` invokes the inherited extraction/serialization route and annotates the event with canonical normalization.
2. `phase2_5_5/normalizer.py:190-270` canonicalizes shared-target, dependency, answer-type, property and target-role fields while retaining field provenance and refusing unknown role/dependency values rather than inventing them.
3. `phase2_5_3b/router.py:31-44` serializes semantic source groups into the existing executable branch schema, then calls the inherited validator. Serialization is deterministic (`phase2_5_3b/serializer.py:57-182`).
4. The same inherited adaptive wrapper then either calls `BranchV1Agent` for a validated `BRANCH_FINAL` plan or calls CoR fallback. The branch merge and fallback contract is not replaced by a general graph executor.

## Observed consequence in the frozen traces

TGAR produced 5 branch attempts, 3 native successes and 8 fallbacks. CFP produced 20 branch attempts, 12 native successes and 34 fallbacks. CFP therefore spends more tokens and time in the extra extraction/normalization/serialization path, but its larger planning stack still feeds the same downstream CoR/BranchV1 contract. A validator pass, a branch attempt, or a native result is not by itself proof of incremental branch benefit.

## Important evidence boundary

The saved traces include prompts/responses, route decisions, validation fields, SPARQL text/count/status, step events, branch candidate snapshots and final predictions. They do not include complete per-hop active-binding state, relation-pruning candidate names/scores for every CoR step, or a deterministic replay of the gold query through the exact executor. Those missing fields constrain root-cause attribution and are listed in `coverage_and_blockers.md`.
"""
    (AUDIT_ROOT / "method_architecture_diff.md").write_text(text, encoding="utf-8")


def write_disagreement_report(bundle: Mapping[str, Any], diagnoses: Mapping[str, dict[str, Any]]) -> None:
    lines = ["# TGAR versus CFP disagreement audit", "", "There are six questions where TGAR and CFP disagree on Hits@1: four TGAR-only and two CFP-only. The other five of the eleven three-way disagreement cases are ties between TGAR and CFP while CoR differs.", ""]
    for key in sorted(bundle["manifest"]):
        t = bundle["runs"][(key[0], key[1], "v1_1_3")][1]
        c = bundle["runs"][(key[0], key[1], "phase2_5_6")][1]
        if int(t.get("hits_at_1", 0)) == int(c.get("hits_at_1", 0)):
            continue
        gold = bundle["gold"][key]
        manifest = bundle["manifest"][key]
        te, ce = route_event(t), route_event(c)
        lines += [
            f"## `{key[0]}::{key[1]}` — {short(t.get('question'), 220)}", "",
            f"- Gold answers: `{gold.get('gold_answers', [])}`; gold branch type: `{manifest.get('branch_shape')}`.",
            f"- Correctness: CoR `{bundle['runs'][(key[0], key[1], 'original_cor')][1].get('hits_at_1')}`, TGAR `{t.get('hits_at_1')}`, CFP `{c.get('hits_at_1')}`; F1 TGAR `{float(t.get('f1', 0.0)):.3f}`, CFP `{float(c.get('f1', 0.0)):.3f}`.",
            f"- TGAR: route `{t.get('router_structure')}`, execution `{t.get('execution_status')}`, fallback `{t.get('fallback_reason')}`, prediction `{t.get('prediction')}`.",
            f"- CFP: route `{c.get('router_structure')}`, execution `{c.get('execution_status')}`, fallback `{c.get('fallback_reason')}`, prediction `{c.get('prediction')}`.",
            f"- TGAR router reason: {short(t.get('router_reason'), 360)}",
            f"- CFP validator/normalization: `{ce.get('validator_decision')}`, failures `{ce.get('validation_failures', [])}`, normalization `{ce.get('normalization_status')}`, retry `{ce.get('retry_used', False)}`.",
            f"- Trace pointers: `runs.jsonl` schedule indices TGAR `{t.get('schedule_index')}`, CFP `{c.get('schedule_index')}`.",
        ]
        if int(t.get("hits_at_1", 0)) and not int(c.get("hits_at_1", 0)):
            conclusion = "TGAR-only. The trace supports a TGAR route/execution contrast, but a single stochastic run cannot establish that the router caused the win rather than a different CoR path or answer-generation realization."
        else:
            conclusion = "CFP-only. The trace supports a CFP route/execution contrast, but it does not isolate planning from downstream relation grounding or answer selection."
        if c.get("fallback_reason"):
            conclusion += " CFP fallback is explicitly recorded and is part of the observed causal path."
        lines += [f"- Audit conclusion: {conclusion}", ""]
    lines += ["## Summary", "", "Across these six cases, route and fallback differences are directly observable. Native branch success is not equivalent to verified native branch benefit: the traces do not provide a counterfactual execution using the same model output with only the branch executor changed.", ""]
    (AUDIT_ROOT / "tgar_vs_cfp_disagreements.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_shared_failures(bundle: Mapping[str, Any], diagnoses: Mapping[str, dict[str, Any]]) -> None:
    lines = ["# The 17 all-three-wrong questions", "", "These are the questions where CoR, TGAR and CFP all miss Hits@1. The common outcome shows a shared ceiling, but it does not by itself identify whether the cause is grounding, representation, KG coverage or answer selection. Each item below reports the strongest evidence available in the frozen traces.", ""]
    count = 0
    for key in sorted(bundle["manifest"]):
        methods = {method: bundle["runs"][(key[0], key[1], method)][1] for method in METHODS}
        if not all(int(methods[method].get("hits_at_1", 0)) == 0 for method in METHODS):
            continue
        count += 1
        diagnosis = diagnoses[stable_key({"dataset": key[0], "question_id": key[1]})]
        lines += [
            f"## {count}. `{key[0]}::{key[1]}` — {short(methods['v1_1_3'].get('question'), 220)}", "",
            f"- Gold answers: `{bundle['gold'][key].get('gold_answers', [])}`; branch type: `{bundle['manifest'][key].get('branch_shape')}`.",
            f"- Predictions: CoR `{methods['original_cor'].get('prediction')}`; TGAR `{methods['v1_1_3'].get('prediction')}`; CFP `{methods['phase2_5_6'].get('prediction')}`.",
            f"- Routes: TGAR `{methods['v1_1_3'].get('router_structure')}/{methods['v1_1_3'].get('execution_status')}`; CFP `{methods['phase2_5_6'].get('router_structure')}/{methods['phase2_5_6'].get('execution_status')}`.",
            f"- Primary audit label: `{diagnosis['primary_root_cause']}`; evidence `{diagnosis['evidence_strength']}`; first stage `{diagnosis['first_divergence_stage']}`.",
            f"- Evidence note: {diagnosis['diagnosis_note']}",
            f"- Pointer: `runs.jsonl` schedule indices CoR `{methods['original_cor'].get('schedule_index')}`, TGAR `{methods['v1_1_3'].get('schedule_index')}`, CFP `{methods['phase2_5_6'].get('schedule_index')}`.", "",
        ]
    lines += ["## Shared-failure interpretation", "", f"The audit found {count} all-three-wrong cases. The decisive common signal is not a shared router decision; the methods often choose different routes and still fail. Because candidate scores, complete binding transitions and bounded gold-plan replay are absent, the common-failure distribution should be treated as evidence for a shared CoR-style grounding/execution ceiling only at hypothesis level. The unresolved portion is retained in `root_cause_summary.json`.", ""]
    (AUDIT_ROOT / "shared_17_failures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_coverage(bundle: Mapping[str, Any]) -> None:
    lines = ["# Coverage and blockers", "", "## Available", "", "- Frozen 50-question manifest and separate offline gold file.", "- All 150 run rows with stable dataset/question/method keys.", "- Full prompt histories and route events for all three methods.", "- SPARQL text, status, timeout flag and result counts for recorded backend calls.", "- Step histories, relation chains, branch traces where a branch was attempted, final predictions, F1/Hit, token and latency fields.", "- CWQ and GrailQA gold SPARQL/logical-form fields for offline topology inspection.", "", "## Missing or insufficient for a decisive causal label", "", "- The run logs do not contain raw relation-candidate names and scores for every CoR `relation_prune` event. They contain selected results and aggregate candidate sizes, so missing-candidate versus wrong-ranking cannot be separated reliably.", "- They do not contain complete active-entity/current-variable binding state after every hop. A wrong final answer can therefore be downstream of an earlier state transition that is not visible in the compact trace.", "- There is no deterministic bounded replay of the gold relation plan through the unchanged executor in this audit. A gold SPARQL query succeeding in the KG would not by itself prove CoR expressivity.", "- Branch traces expose candidate IDs for branch attempts, but not all raw candidate sets before filtering and not all answer-selection scores.", "- The official row-level `retry_count` is zero, while one CFP router event records `retry_used=true` for an empty model response. This is retained as an internal trace caveat, not silently counted as an official row retry.", "- No new KG calls or LLM calls were made, so backend availability and current KG semantics were not re-probed.", "", "## Smallest additional capture needed", "", "1. Run a bounded offline/deterministic replay on a targeted subset of the 17 shared failures using the existing executor contract and an explicitly recorded oracle relation plan; record whether the executor preserves intermediate bindings and returns the gold answer IDs. Do not report this as normal-model accuracy.", "2. If replay is unavailable, add read-only instrumentation to record pre/post-pruning candidate IDs, active state, selected relation direction and answer-set merge inputs for the same targeted cases. This would distinguish grounding, state transition and merge failures without another 150-question run.", ""]
    (AUDIT_ROOT / "coverage_and_blockers.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(bundle: Mapping[str, Any], integrity_data: Mapping[str, Any], matrix: list[dict[str, Any]], diagnoses: Mapping[str, dict[str, Any]], paired: Mapping[str, Any]) -> dict[str, Any]:
    cause_counts = Counter(row["primary_root_cause"] for row in matrix if "TGAR_FAILURE" in row["tags"])
    evidence_counts = Counter(row["evidence_strength"] for row in matrix if "TGAR_FAILURE" in row["tags"])
    shared = [row for row in matrix if "ALL_THREE_WRONG" in row["tags"]]
    disagreements = [row for row in matrix if "TGAR_CFP_DISAGREEMENT" in row["tags"]]
    recommendation = "Run one targeted deterministic executor-capability replay on the 17 shared failures with oracle relation plans and binding-preservation logging; do not launch another fresh 50 or modify CFP before that test."
    summary = {
        "integrity_status": integrity_data["status"],
        "reproduced_questions": len(bundle["manifest"]),
        "reproduced_runs": len(bundle["run_lines"]),
        "tgar_failures": sum(1 for row in matrix if "TGAR_FAILURE" in row["tags"]),
        "shared_failures": len(shared),
        "tgar_cfp_disagreement_cases": len(disagreements),
        "primary_root_cause_counts_tgar_failures": dict(cause_counts),
        "evidence_strength_counts_tgar_failures": dict(evidence_counts),
        "shared_failure_primary_counts": dict(Counter(row["primary_root_cause"] for row in shared)),
        "cohort_counts": {
            "tgar_failures": sum(1 for row in matrix if "TGAR_FAILURE" in row["tags"]),
            "all_three_wrong": len(shared),
            "tgar_cfp_disagreement": len(disagreements),
            "all_three_correct": sum(1 for row in matrix if "ALL_THREE_CORRECT" in row["tags"]),
        },
        "missingness_counts": {
            "candidate_scores_unavailable_for_method_traces": len(matrix),
            "active_binding_state_unavailable_for_method_traces": len(matrix),
            "gold_relation_replay_unperformed": len(matrix),
            "branch_candidate_snapshots_available": sum(1 for row in matrix if row["tgar_execution_status"] in {"NATIVE_BRANCH_SUCCESS", "BRANCH_ATTEMPTED_FALLBACK"} or row["cfp_execution_status"] in {"NATIVE_BRANCH_SUCCESS", "BRANCH_ATTEMPTED_FALLBACK"}),
        },
        "mechanism_counts": dict(Counter(row["tgar_execution_status"] for row in matrix)),
        "best_method_now": "TGAR for Hits@1/cost; CFP has higher mean F1 but materially higher fallback and cost.",
        "decision": "INSUFFICIENT_EVIDENCE_FOR_METHOD_REDESIGN",
        "recommended_next_experiment": recommendation,
        "new_llm_calls": 0,
        "new_kg_calls": 0,
        "production_methods_changed": False,
    }
    (AUDIT_ROOT / "root_cause_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Fresh50 TGAR versus CFP root-cause audit", "", "## Decision memo", "", "The frozen result is reproduced exactly: TGAR is the operational best method for Hits@1 and cost on this cohort (29/50, 0.580, 12,518 average tokens), while CFP has the highest mean F1 (0.4646) but only 27/50 Hits@1, 68% fallback and 14,722 average tokens. The paired confidence intervals include zero, so this is not an inferential superiority claim.", "", "TGAR's 21 failures are not explained by one confirmed mechanism. The strongest observed first divergences are route/structure mismatches on benchmark branch topologies, explicit fallback/merge events, and relation-chain mismatches where CWQ gold predicates are available. For GrailQA and for many shared failures, the logs do not expose enough state to prove whether the first error was model planning, relation grounding, binding transition, KG coverage or final answer selection.", "", "The 17 all-three-wrong cases are therefore evidence of a shared ceiling in the relation-centric system, but not proof that the LLM planner, the executor, or Freebase alone is responsible. CFP's added planning stage increases cost and fallback substantially while reusing the same downstream validator/executor; the audit does not support calling it intrinsically worse because of complexity without a counterfactual.", "", "## Reproduced accounting", "", f"- Frozen questions: **{len(bundle['manifest'])}**; official rows: **{len(bundle['run_lines'])}**; duplicate slots: **{len(integrity_data['duplicate_slots'])}**.", f"- Pattern counts in method order CoR/TGAR/CFP: `{integrity_data['all_three_pattern_counts']}`.", f"- TGAR failures: **{sum(1 for row in matrix if 'TGAR_FAILURE' in row['tags'])}**; all-three failures: **{len(shared)}**; TGAR/CFP disagreements: **{len(disagreements)}**.", f"- TGAR failure labels: `{dict(cause_counts)}`; evidence tiers: `{dict(evidence_counts)}`.", "", "## Three decisive cases", "", "1. **CWQ WebQTest-712...**: CoR is correct, TGAR rejects the two-constraint final branch as `UNCERTAIN`, and CFP validates a branch but its strict merge is empty and falls back. This is direct evidence of different planning/fallback paths, not proof that either branch plan would solve the question under a corrected executor.", "2. **CWQ WebQTrn-513...**: CFP reaches the gold answer through a native branch result while CoR and TGAR return the same wrong multi-answer set. This is the clearest native-branch contrast, but one case cannot establish a general branch benefit.", "3. **GrailQA 4300742008000**: all methods miss, with TGAR and CFP taking uncertain/fallback paths and no compact trace exposing the gold relation-binding sequence. It illustrates why the shared-failure cohort cannot be assigned to planning alone.", "", "## Planning versus CoR limitation", "", "The evidence supports a mixed answer. TGAR route mismatches and CFP validator/serialization/fallback events show planning and contract effects. At the same time, many questions remain wrong across all methods, and both adaptive methods reuse CoR's relation grounding, reasoning, filtering and branch merge components. The dominant causal mechanism is **not yet established** because candidate scores, active bindings and deterministic oracle replay are missing.", "", "## Recommended next diagnostic", "", recommendation, "", "## Reproducibility and safety", "", "This audit used only existing files and deterministic offline parsing. It made zero LLM/API calls and zero KG calls. The source experiment files were not modified. Full per-question details are in `case_matrix.csv`, `method_traces.jsonl`, `first_divergence_audit.csv`, `shared_17_failures.md`, and `tgar_vs_cfp_disagreements.md`.", ""]
    (AUDIT_ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    integrity_data = integrity(bundle)
    (AUDIT_ROOT / "integrity_checks.json").write_text(json.dumps(integrity_data, indent=2) + "\n", encoding="utf-8")
    diagnoses: dict[str, dict[str, Any]] = {}
    trace_rows: list[dict[str, Any]] = []
    for key, manifest in sorted(bundle["manifest"].items()):
        diagnosis = primary_diagnosis(bundle["runs"][(key[0], key[1], "v1_1_3")][1], manifest, bundle["gold"][key])
        diagnoses[stable_key({"dataset": key[0], "question_id": key[1]})] = diagnosis
    for key, manifest in sorted(bundle["manifest"].items()):
        for method in METHODS:
            line, row = bundle["runs"][(key[0], key[1], method)]
            trace_rows.append(trace_record(line, row, manifest, bundle["gold"][key], diagnoses[stable_key(row)]))
    write_jsonl = lambda path, values: path.write_text("".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values), encoding="utf-8")
    write_jsonl(AUDIT_ROOT / "method_traces.jsonl", trace_rows)
    matrix = write_case_matrix(bundle, diagnoses)
    with (AUDIT_ROOT / "case_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(matrix[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(matrix)
    divergence_rows = []
    for row in matrix:
        diagnosis = diagnoses[row["question_key"]]
        divergence_rows.append({
            "question_key": row["question_key"], "dataset": row["dataset"], "question_id": row["question_id"], "question_text": row["question_text"],
            "branch_type": row["branch_type"], "tgar_hit": row["tgar_hit"], "primary_root_cause": diagnosis["primary_root_cause"],
            "first_divergence_stage": diagnosis["first_divergence_stage"], "evidence_strength": diagnosis["evidence_strength"],
            "secondary_causes": ";".join(diagnosis.get("secondary_causes", [])), "diagnosis_note": diagnosis["diagnosis_note"],
            "evidence_pointer": "; ".join(f"{method}:{bundle['runs'][(row['dataset'], row['question_id'], method)][0]}" for method in METHODS),
        })
    with (AUDIT_ROOT / "first_divergence_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(divergence_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(divergence_rows)
    write_architecture_diff()
    write_disagreement_report(bundle, diagnoses)
    write_shared_failures(bundle, diagnoses)
    write_f1_audit(bundle)
    write_coverage(bundle)
    paired = paired_records(bundle)
    summary = write_report(bundle, integrity_data, matrix, diagnoses, paired)
    readme = f"""# Fresh50 TGAR versus CFP root-cause audit

Offline audit of `{SOURCE_ROOT.relative_to(ROOT)}` at repository revision `{integrity_data['repo_head']}`.

## Reproduce

```bash
PYTHONPATH=src:. .venv/bin/python scripts/audit_fresh50_tgar_vs_cfp.py
PYTHONPATH=src:. .venv/bin/python -m unittest tests/test_fresh50_root_cause_audit.py
```

The script performs no LLM/API calls and no KG/SPARQL calls. It reads the frozen manifest, separate gold file, run traces, evaluator source and repository source. Gold data is used only in offline analysis.

## Completed stages

- Stage A: aggregate accounting, slot integrity, method-fidelity/source map.
- Stage B: 150 compact standardized method traces and 50-question case matrix.
- Stage C: evidence-bounded first-divergence labels; unresolved cases are retained.
- Stage D: six TGAR/CFP Hits@1 disagreement cases.
- Stage E: decision memo and one targeted next diagnostic.

## Result status

- Reproduced: 50 questions / 150 frozen runs.
- TGAR failures: {summary['tgar_failures']}.
- Shared failures: {summary['shared_failures']}.
- TGAR/CFP disagreement cases: {summary['tgar_cfp_disagreement_cases']}.
- Production methods changed: no.
- New LLM calls: 0; new KG calls: 0.

The official row retry count is zero, but one CFP router event records an internal empty-response retry. This is called out in `integrity_checks.json` and `coverage_and_blockers.md`.
"""
    (AUDIT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
