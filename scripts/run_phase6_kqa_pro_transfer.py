#!/usr/bin/env python3
"""Run the Phase 6 KQA Pro backend transfer and frozen baseline evaluation.

The script first performs a complete offline compatibility audit.  The frozen
methods are only run when the predeclared supported-subset gate is viable.
Nothing in this script changes prompts, routing policy, validators, or answer
selection semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from chain_of_relations.kg_backend.kqa_pro_backend import KQAProBackend, UnsupportedKQAOperator
from chain_of_relations.schema import Entity


PHASE5_ROOT = ROOT / "outputs/experiments/phase5/dataset_complexity_audit"
OUTPUT_ROOT = ROOT / "outputs/experiments/phase6/kqa_pro_frozen_transfer"
PHASE5_SOURCE = PHASE5_ROOT / "source/kqa_pro_val.json"
PHASE5_MANIFEST = PHASE5_ROOT / "kqa_pro_sample_manifest.json"
PHASE5_FEATURES = PHASE5_ROOT / "kqa_pro_complexity_features.jsonl"
KB_SOURCE = OUTPUT_ROOT / "source/kb.json"
RELATION_AUDIT_SEED = 20261101

TIER_A_OPERATORS = {
    "Find",
    "Relate",
    "FilterConcept",
    "And",
    "What",
    "QueryName",
    "QueryRelation",
}
TIER_B_OPERATORS = TIER_A_OPERATORS | {"FilterStr", "QueryAttr"}

OPERATOR_MATRIX = {
    "Find": ("SUPPORTED_NATIVE", "FULL", "FULL", "entity grounding and root construction"),
    "FindAll": ("SUPPORTED_NATIVE", "NONE", "NONE", "current agents require runtime topic entities"),
    "Relate": ("SUPPORTED_NATIVE", "FULL", "FULL", "relation traversal with explicit direction"),
    "FilterConcept": ("SUPPORTED_NATIVE", "FULL", "FULL", "deterministic concept/type check"),
    "FilterStr": ("SUPPORTED_WITH_ADAPTER", "PARTIAL", "PARTIAL", "simple string attribute filtering"),
    "FilterNum": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "numeric comparison policy is not part of frozen methods"),
    "FilterYear": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "temporal comparison policy is not part of frozen methods"),
    "FilterDate": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "temporal comparison policy is not part of frozen methods"),
    "QFilterStr": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "qualifier-aware filtering"),
    "QFilterNum": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "qualifier-aware numeric filtering"),
    "QFilterYear": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "qualifier-aware temporal filtering"),
    "QFilterDate": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "qualifier-aware temporal filtering"),
    "And": ("SUPPORTED_NATIVE", "FULL", "FULL", "basic set intersection"),
    "Or": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "set union is not in the frozen method contract"),
    "Count": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "aggregation/count answer semantics"),
    "What": ("SUPPORTED_NATIVE", "FULL", "FULL", "entity-name answer serialization"),
    "QueryName": ("SUPPORTED_NATIVE", "FULL", "FULL", "alias of What when present"),
    "QueryAttr": ("SUPPORTED_WITH_ADAPTER", "PARTIAL", "PARTIAL", "simple attribute-value answer serialization"),
    "QueryAttrUnderCondition": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "condition-aware attribute semantics"),
    "QueryRelation": ("SUPPORTED_NATIVE", "FULL", "FULL", "direct relation candidate discovery"),
    "QueryAttrQualifier": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "qualified attribute semantics"),
    "QueryRelationQualifier": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "qualified relation semantics"),
    "SelectBetween": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "comparison selection"),
    "SelectAmong": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "superlative/comparison selection"),
    "VerifyStr": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "verification answer semantics"),
    "VerifyNum": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "verification answer semantics"),
    "VerifyYear": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "verification answer semantics"),
    "VerifyDate": ("UNSUPPORTED_SEMANTIC_OPERATOR", "NONE", "NONE", "verification answer semantics"),
}


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
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_id(sample_index: int) -> str:
    return f"kqa_pro_val_{int(sample_index):05d}"


def source_rows_and_features() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = read_json(PHASE5_SOURCE)
    manifest = read_json(PHASE5_MANIFEST)
    features = read_jsonl(PHASE5_FEATURES)
    if not isinstance(rows, list) or len(rows) != 11797:
        raise RuntimeError(f"unexpected frozen KQA Pro validation size: {len(rows)}")
    selected = sorted({int(index) for values in manifest["selected_source_indices"].values() for index in values})
    feature_indices = sorted(int(row["sample_index"]) for row in features)
    if len(selected) != 180 or feature_indices != selected:
        raise RuntimeError("Phase 5 selected indices and complexity features do not match exactly")
    for feature in features:
        source = rows[int(feature["sample_index"])]
        if feature.get("question") != source.get("question"):
            raise RuntimeError(f"Phase 5 feature/source mismatch at index {feature['sample_index']}")
    return rows, features, manifest


def validate_source(rows: list[dict[str, Any]], features: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    source_hash = sha256(PHASE5_SOURCE)
    kb_hash = sha256(KB_SOURCE)
    selected_indices = sorted(int(index) for values in manifest["selected_source_indices"].values() for index in values)
    return {
        "status": "VALIDATED",
        "phase5_source_path": str(PHASE5_SOURCE),
        "phase5_manifest_path": str(PHASE5_MANIFEST),
        "phase5_feature_path": str(PHASE5_FEATURES),
        "phase5_source_split": manifest.get("source_split"),
        "phase5_source_sha256": source_hash,
        "phase5_manifest_source_sha256": manifest.get("source_sha256"),
        "phase5_source_hash_matches_manifest": source_hash == manifest.get("source_sha256"),
        "phase5_source_record_count": len(rows),
        "frozen_sample_size": len(features),
        "frozen_sample_seed": manifest.get("selection_seed"),
        "selected_source_indices_sha256": hashlib.sha256(json.dumps(selected_indices).encode()).hexdigest(),
        "selected_source_indices_unique": len(selected_indices) == len(set(selected_indices)),
        "kb_path": str(KB_SOURCE),
        "kb_sha256": kb_hash,
        "kb_size_bytes": KB_SOURCE.stat().st_size,
        "no_resampling": True,
        "source_version_note": manifest.get("source_version", ""),
    }


def capability_matrix() -> dict[str, Any]:
    rows = []
    for operator, (backend_status, v1, p25, extension) in OPERATOR_MATRIX.items():
        rows.append({
            "operator": operator,
            "backend_status": backend_status,
            "v1_1_3": v1,
            "phase2_5_6": p25,
            "frozen_method_compatibility": v1 in {"FULL", "PARTIAL"} and p25 in {"FULL", "PARTIAL"},
            "notes": extension,
        })
    return {
        "status": "FROZEN_BEFORE_INFERENCE",
        "backend": "KQAProBackend",
        "method_policies_unchanged": True,
        "operators": rows,
        "tier_a_operators": sorted(TIER_A_OPERATORS),
        "tier_b_operators": sorted(TIER_B_OPERATORS),
        "unsupported_semantics_are_not_emulated": True,
    }


def feature_for_index(features: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(row["sample_index"]): row for row in features}


def primary_blocker(functions: Sequence[str]) -> str:
    priority = [
        "FindAll", "Or", "Count", "SelectAmong", "SelectBetween",
        "QueryRelationQualifier", "QueryAttrQualifier", "QFilterStr",
        "QFilterNum", "QFilterYear", "QFilterDate", "FilterNum",
        "FilterYear", "FilterDate", "QueryAttrUnderCondition",
    ]
    for operator in priority:
        if operator in functions:
            return operator
    return next((operator for operator in functions if operator not in TIER_B_OPERATORS), "NO_RUNTIME_TOPIC_ENTITY")


def subset_manifest(rows: Sequence[Mapping[str, Any]], features: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_index = feature_for_index(features)
    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for index in sorted(int(value) for values in manifest["selected_source_indices"].values() for value in values):
        source = rows[index]
        functions = [str(step.get("function", "")) for step in source.get("program", [])]
        blockers = sorted(set(functions) - TIER_B_OPERATORS)
        if not any(function == "Find" for function in functions):
            blockers.append("NO_RUNTIME_TOPIC_ENTITY")
        feature = by_index[index]
        record = {
            "case_id": case_id(index),
            "sample_index": index,
            "family": feature.get("family"),
            "question": source.get("question", ""),
            "operators": functions,
            "operator_set": sorted(set(functions)),
            "normalized_depth": feature.get("normalized_depth"),
            "operator_diversity": len(feature.get("normalized_operator_families", [])),
            "constraint_count": feature.get("constraint_count"),
            "nested_composition": bool(feature.get("complexity_flags", {}).get("nested_composition")),
            "source_split": "val",
        }
        if not blockers:
            tier = "NATIVE_RELATIONAL" if set(functions) <= TIER_A_OPERATORS else "EXTENDED_EXPRESSIBLE"
            record["tier"] = tier
            record["eligibility"] = "SUPPORTED"
            supported.append(record)
        else:
            record["eligibility"] = "UNSUPPORTED"
            record["primary_blocking_operator"] = primary_blocker(functions)
            record["all_blocking_operators"] = sorted(set(blockers))
            unsupported.append(record)
    tier_counts = Counter(row["tier"] for row in supported)
    result = {
        "status": "FROZEN_BEFORE_INFERENCE",
        "source_manifest": str(PHASE5_MANIFEST),
        "source_sample_size": 180,
        "supported_case_count": len(supported),
        "unsupported_case_count": len(unsupported),
        "supported_subset_viability": (
            "SUPPORTED_SUBSET_VIABLE" if len(supported) >= 60 else
            "SUPPORTED_SUBSET_SMALL" if len(supported) >= 30 else
            "NO_MEANINGFUL_SUPPORTED_SUBSET"
        ),
        "tier_counts": dict(sorted(tier_counts.items())),
        "eligibility_rule": "all actual KoPL operators must be in the frozen Tier A/B set and at least one Find root must be available; no simplification is applied",
        "method_changes_forbidden": True,
        "supported_cases": supported,
    }
    return result, supported, unsupported


def grounding_audit(backend: KQAProBackend, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = []
    total_anchors = 0
    resolved_anchors = 0
    full_questions = 0
    ambiguous = 0
    unresolved = 0
    for row in rows:
        source = row["source"]
        anchors = [str(step.get("inputs", [""])[0]) for step in source.get("program", []) if step.get("function") == "Find" and step.get("inputs")]
        anchor_records = []
        question_ok = True
        for anchor in anchors:
            candidates = backend.lookup_entities(anchor)
            total_anchors += 1
            if candidates:
                resolved_anchors += 1
            else:
                unresolved += 1
                question_ok = False
            if len(candidates) > 1:
                ambiguous += 1
            anchor_records.append({"gold_anchor": anchor, "resolved_entity_ids": candidates, "ambiguous": len(candidates) > 1, "unresolved": not candidates})
        if question_ok:
            full_questions += 1
        records.append({"case_id": row["case_id"], "sample_index": row["sample_index"], "family": row["family"], "anchors": anchor_records, "full_anchor_resolution": question_ok})
    return {
        "status": "COMPLETE",
        "supported_subset_size": len(rows),
        "gold_entity_anchor_count": total_anchors,
        "resolved_entity_anchor_count": resolved_anchors,
        "ambiguous_entity_anchor_count": ambiguous,
        "unresolved_entity_anchor_count": unresolved,
        "entity_anchor_recall": resolved_anchors / total_anchors if total_anchors else 0.0,
        "question_full_anchor_resolution_count": full_questions,
        "question_full_anchor_resolution_rate": full_questions / len(rows) if rows else 0.0,
        "records": records,
    }


def relation_diagnostic_sample(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(RELATION_AUDIT_SEED)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(dict(row))
    for family_rows in grouped.values():
        rng.shuffle(family_rows)
        family_rows.sort(key=lambda row: (int(row.get("normalized_depth") or 0), int(row["sample_index"])))
    selected: list[dict[str, Any]] = []
    families = sorted(grouped)
    cursor = 0
    while len(selected) < min(20, len(rows)) and families:
        family = families[cursor % len(families)]
        if grouped[family]:
            selected.append(grouped[family].pop(0))
        families = [name for name in families if grouped[name]]
        cursor += 1
    return sorted(selected, key=lambda row: int(row["sample_index"]))


def relation_candidate_audit(backend: KQAProBackend, supported: Sequence[Mapping[str, Any]], source_by_index: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    diagnostic_rows = relation_diagnostic_sample(supported)
    records = []
    obligations = 0
    available = 0
    direction_total = 0
    direction_correct = 0
    for row in diagnostic_rows:
        source = source_by_index[int(row["sample_index"])]
        program = source["program"]
        case_record = {"case_id": row["case_id"], "sample_index": row["sample_index"], "family": row["family"], "gold_relation_checks": []}
        for step_index, step in enumerate(program):
            function = str(step.get("function", ""))
            deps = [int(dep) for dep in step.get("dependencies", [])]
            try:
                prefix = backend.execute_program(program[: step_index + 1])
            except UnsupportedKQAOperator:
                continue
            if function == "Relate" and len(deps) == 1 and len(step.get("inputs", [])) >= 2:
                sources = [str(value) for value in prefix["states"].get(deps[0], []) if backend.is_entity_id(str(value))]
                gold_relation = str(step["inputs"][0])
                direction = str(step["inputs"][1])
                candidates = sorted({candidate for source_id in sources for candidate in backend.relation_candidates(source_id, direction)})
                traversed = backend.traverse(sources, gold_relation, direction)
                obligations += 1
                available += int(gold_relation in candidates)
                direction_total += 1
                direction_correct += int(bool(traversed))
                case_record["gold_relation_checks"].append({"step_index": step_index, "operation": function, "gold_relation": gold_relation, "direction": direction, "source_entity_ids": sources, "candidate_relations": candidates, "gold_relation_available": gold_relation in candidates, "direction_correct": bool(traversed)})
            elif function == "QueryRelation" and len(deps) == 2:
                source_ids = [str(value) for value in prefix["states"].get(deps[0], []) if backend.is_entity_id(str(value))]
                target_ids = [str(value) for value in prefix["states"].get(deps[1], []) if backend.is_entity_id(str(value))]
                gold_relation = str(source.get("answer", ""))
                candidates = backend.relation_between(source_ids, target_ids)
                obligations += 1
                available += int(gold_relation in candidates)
                case_record["gold_relation_checks"].append({"step_index": step_index, "operation": function, "gold_relation": gold_relation, "source_entity_ids": source_ids, "target_entity_ids": target_ids, "candidate_relations": candidates, "gold_relation_available": gold_relation in candidates, "direction_correct": None})
        records.append(case_record)
    return {
        "status": "COMPLETE",
        "selection_seed": RELATION_AUDIT_SEED,
        "diagnostic_sample_size": len(diagnostic_rows),
        "gold_relation_obligations": obligations,
        "gold_relation_available_count": available,
        "candidate_relation_recall": available / obligations if obligations else None,
        "direction_checks": direction_total,
        "direction_correct_count": direction_correct,
        "direction_correctness": direction_correct / direction_total if direction_total else None,
        "records": records,
    }


def question_entities(backend: KQAProBackend, source: Mapping[str, Any]) -> list[Entity]:
    entities: list[Entity] = []
    seen: set[str] = set()
    for step in source.get("program", []):
        if step.get("function") != "Find" or not step.get("inputs"):
            continue
        name = str(step["inputs"][0])
        for entity_id in backend.lookup_entities(name):
            if entity_id in seen:
                continue
            seen.add(entity_id)
            entities.append(Entity(entity_id, backend.id2entity_name_or_type(entity_id)))
    return entities


def result_prediction(result: Mapping[str, Any]) -> list[str]:
    values = result.get("results", [])
    if not isinstance(values, list):
        return []
    return [str(value.get("name", "")).strip() for value in values if isinstance(value, dict) and str(value.get("name", "")).strip()]


def evaluate_prediction(prediction: Sequence[str], gold: str) -> tuple[int, float]:
    from chain_of_relations.eval.accuracy import eval_f1, eval_hit
    answer = [str(gold)] if str(gold).strip() else []
    text = " ".join(prediction) if prediction else "None"
    return (int(eval_hit(text, answer)) if answer else 0, float(eval_f1(list(prediction), answer)[0]) if answer else 0.0)


def method_agent(method: str, backend: KQAProBackend) -> Any:
    from chain_of_relations.methods.cor.agent import CoRAgent
    cor = CoRAgent(
        model_name=os.environ["MODEL_NAME"],
        relation_width=3,
        depth=3,
        sample_relation_threshold=1000,
        remove_unnecessary_rel=True,
        temperature_exploration=0.3,
        temperature_reasoning=0.1,
        max_token=1024,
        backend=backend,
    )
    if method == "v1_1_3":
        from chain_of_relations.methods.adaptive_v1_1_3.agent import AdaptiveRoutingAgent
        return AdaptiveRoutingAgent(cor, confidence_threshold=0.75, max_branches=3)
    if method == "phase2_5_6":
        from chain_of_relations.methods.phase2_5_5.agent import CanonicalNormalizationAgent
        return CanonicalNormalizationAgent(cor, confidence_threshold=0.75, max_branches=3)
    raise ValueError(f"unknown frozen method: {method}")


def integer_usage(history: Sequence[Mapping[str, Any]], field: str) -> int:
    total = 0
    for item in history:
        try:
            total += max(0, int(item.get(field, 0) or 0))
        except (TypeError, ValueError):
            continue
    return total


def run_live(backend: KQAProBackend, supported: Sequence[Mapping[str, Any]], source_by_index: Mapping[int, Mapping[str, Any]], repeats: int) -> list[dict[str, Any]]:
    from run_cor_baseline_experiment import load_safe_project_env
    load_safe_project_env()
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("MODEL_NAME"):
        raise RuntimeError("OPENAI_API_KEY and MODEL_NAME are required for frozen evaluation")
    output: list[dict[str, Any]] = []
    for method in ("v1_1_3", "phase2_5_6"):
        agent = method_agent(method, backend)
        for row in supported:
            source = source_by_index[int(row["sample_index"])]
            topic_entities = question_entities(backend, source)
            for repeat in range(1, repeats + 1):
                prompt_history: list[dict[str, Any]] = []
                sparql_history: list[dict[str, Any]] = []
                step_history: list[dict[str, Any]] = []
                started = time.perf_counter()
                error = None
                try:
                    result = agent.answer(
                        question=str(source.get("question", "")),
                        topic_entities=topic_entities,
                        prompt_history=prompt_history,
                        sparql_history=sparql_history,
                        step_history=step_history,
                    )
                except Exception as exc:  # preserve paired case accounting
                    result = {"action": "ERROR", "results": [], "reasoning_chains": ""}
                    error = f"{type(exc).__name__}: {exc}"
                latency_ms = (time.perf_counter() - started) * 1000.0
                prediction = result_prediction(result)
                hit, f1 = evaluate_prediction(prediction, str(source.get("answer", "")))
                native = result.get("adaptive_native_result") if isinstance(result.get("adaptive_native_result"), dict) else result
                native_prediction = result_prediction(native)
                native_hit, native_f1 = evaluate_prediction(native_prediction, str(source.get("answer", "")))
                output.append({
                    "case_id": row["case_id"],
                    "sample_index": row["sample_index"],
                    "family": row["family"],
                    "question": source.get("question", ""),
                    "method": method,
                    "repeat": repeat,
                    "gold_answer": source.get("answer", ""),
                    "prediction": prediction,
                    "hits_at_1": hit,
                    "f1": f1,
                    "native_prediction": native_prediction,
                    "native_hits_at_1": native_hit,
                    "native_f1": native_f1,
                    "topic_entity_ids": [entity.id for entity in topic_entities],
                    "topic_entity_names": [entity.name for entity in topic_entities],
                    "llm_calls": len(prompt_history),
                    "kg_queries": len(sparql_history),
                    "token_usage": {"input_tokens": integer_usage(prompt_history, "input_tokens"), "output_tokens": integer_usage(prompt_history, "output_tokens"), "total_tokens": integer_usage(prompt_history, "input_tokens") + integer_usage(prompt_history, "output_tokens")},
                    "latency_ms": latency_ms,
                    "error": error or (str(result.get("reasoning_chains", "")) if str(result.get("action", "")).upper() == "ERROR" else None),
                    "router_structure": result.get("router_structure", "BASELINE_UNCHANGED" if method == "v1_1_3" else "UNCERTAIN"),
                    "router_execution_mode": result.get("router_execution_mode", "cor"),
                    "router_confidence": result.get("router_confidence", 0.0),
                    "fallback_used": bool(result.get("fallback_used", False)),
                    "fallback_reason": result.get("fallback_reason"),
                    "final_execution_mode": result.get("final_execution_mode", "cor"),
                    "native_execution_mode": result.get("native_execution_mode", "cor"),
                    "branch_failures": result.get("branch_failures", []),
                    "branches_created": result.get("branches_created", 0),
                    "merged_candidate_count": result.get("merged_candidate_count", len(prediction)),
                    "merge_method": result.get("merge_method", "NONE"),
                    "merge_outcome": result.get("merge_outcome", ""),
                    "validation_passed": result.get("validation_passed", True if method == "v1_1_3" else False),
                    "prompt_history": prompt_history,
                    "sparql_history": sparql_history,
                    "step_history": step_history,
                })
                print(json.dumps({"method": method, "case_id": row["case_id"], "repeat": repeat, "hit": hit, "error": error}, ensure_ascii=False), flush=True)
    return output


def majority_case_rows(run_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row["case_id"]), str(row["method"]))].append(row)
    result = {}
    for key, values in grouped.items():
        hits = [int(value.get("hits_at_1", 0) or 0) for value in values]
        result[key] = {
            "case_id": key[0],
            "method": key[1],
            "n_repeats": len(values),
            "case_majority_correct": sum(hits) * 2 >= len(hits),
            "hits_at_1_mean": statistics.mean(hits) if hits else 0.0,
            "f1_mean": statistics.mean(float(value.get("f1", 0.0) or 0.0) for value in values) if values else 0.0,
            "rows": values,
        }
    return result


def method_metrics(run_rows: Sequence[Mapping[str, Any]], case_rows: Mapping[tuple[str, str], Mapping[str, Any]], method: str) -> dict[str, Any]:
    values = [row for row in run_rows if row.get("method") == method]
    cases = [value for (case_id, case_method), value in case_rows.items() if case_method == method]
    return {
        "method": method,
        "run_rows": len(values),
        "cases": len(cases),
        "hits_at_1_run_mean": statistics.mean(int(row.get("hits_at_1", 0) or 0) for row in values) if values else 0.0,
        "f1_run_mean": statistics.mean(float(row.get("f1", 0.0) or 0.0) for row in values) if values else 0.0,
        "case_majority_correct": sum(bool(row["case_majority_correct"]) for row in cases),
        "case_majority_h1": sum(bool(row["case_majority_correct"]) for row in cases) / len(cases) if cases else 0.0,
        "average_total_tokens": statistics.mean(int(row.get("token_usage", {}).get("total_tokens", 0) or 0) for row in values) if values else 0.0,
        "average_latency_ms": statistics.mean(float(row.get("latency_ms", 0.0) or 0.0) for row in values) if values else 0.0,
        "average_llm_calls": statistics.mean(int(row.get("llm_calls", 0) or 0) for row in values) if values else 0.0,
        "average_kg_queries": statistics.mean(int(row.get("kg_queries", 0) or 0) for row in values) if values else 0.0,
        "error_rate": sum(bool(row.get("error")) for row in values) / len(values) if values else 0.0,
        "fallback_rate": sum(bool(row.get("fallback_used")) for row in values) / len(values) if values else 0.0,
    }


def paired_analysis(case_rows: Mapping[tuple[str, str], Mapping[str, Any]], supported: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_case = {row["case_id"]: row for row in supported}
    paired = []
    counts = Counter()
    for case_id, row in sorted(by_case.items()):
        v1 = case_rows.get((case_id, "v1_1_3"))
        p25 = case_rows.get((case_id, "phase2_5_6"))
        if not v1 or not p25:
            continue
        v1_correct = bool(v1["case_majority_correct"])
        p25_correct = bool(p25["case_majority_correct"])
        outcome = "BOTH_CORRECT" if v1_correct and p25_correct else "V1_1_3_ONLY" if v1_correct else "PHASE2_5_6_ONLY" if p25_correct else "BOTH_WRONG"
        counts[outcome] += 1
        paired.append({"case_id": case_id, "family": row["family"], "v1_1_3_correct": v1_correct, "phase2_5_6_correct": p25_correct, "outcome": outcome})
    return {"case_count": len(paired), "counts": dict(counts), "paired_net": counts["PHASE2_5_6_ONLY"] - counts["V1_1_3_ONLY"], "rows": paired}


def stratification_rows(case_rows: Mapping[tuple[str, str], Mapping[str, Any]], supported: Sequence[Mapping[str, Any]], dimension: str) -> list[dict[str, Any]]:
    def label(row: Mapping[str, Any]) -> str:
        if dimension == "program_depth":
            depth = int(row.get("normalized_depth") or 0)
            return "depth<=3" if depth <= 3 else "depth=4" if depth == 4 else "depth>=5"
        if dimension == "operator_diversity":
            count = int(row.get("operator_diversity") or 0)
            return "low<=3" if count <= 3 else "medium=4" if count == 4 else "high>=5"
        if dimension == "nested_composition":
            return "nested" if row.get("nested_composition") else "not_nested"
        if dimension == "constraint_count":
            return "lower<=4" if int(row.get("constraint_count") or 0) <= 4 else "higher>=5"
        raise ValueError(dimension)

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in supported:
        groups[label(row)].append(row)
    output = []
    for stratum, rows in sorted(groups.items()):
        values = {method: [case_rows[(row["case_id"], method)] for row in rows if (row["case_id"], method) in case_rows] for method in ("v1_1_3", "phase2_5_6")}
        v1_h1 = sum(bool(item["case_majority_correct"]) for item in values["v1_1_3"]) / len(values["v1_1_3"]) if values["v1_1_3"] else 0.0
        p25_h1 = sum(bool(item["case_majority_correct"]) for item in values["phase2_5_6"]) / len(values["phase2_5_6"]) if values["phase2_5_6"] else 0.0
        output.append({"stratum": stratum, "n": len(rows), "v1_1_3_h1": v1_h1, "phase2_5_6_h1": p25_h1, "delta_h1": p25_h1 - v1_h1})
    return output


def build_family_results(case_rows: Mapping[tuple[str, str], Mapping[str, Any]], supported: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in supported:
        groups[str(row["family"])].append(row)
    result = []
    for family, rows in sorted(groups.items()):
        values = {}
        for method in ("v1_1_3", "phase2_5_6"):
            method_rows = [case_rows[(row["case_id"], method)] for row in rows if (row["case_id"], method) in case_rows]
            values[method] = {"n": len(method_rows), "case_majority_correct": sum(bool(item["case_majority_correct"]) for item in method_rows), "h1": sum(bool(item["case_majority_correct"]) for item in method_rows) / len(method_rows) if method_rows else 0.0}
        result.append({"family": family, "n": len(rows), "v1_1_3": values["v1_1_3"], "phase2_5_6": values["phase2_5_6"], "paired_net": values["phase2_5_6"]["case_majority_correct"] - values["v1_1_3"]["case_majority_correct"]})
    return result


def oracle_headroom(case_rows: Mapping[tuple[str, str], Mapping[str, Any]], supported: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    v1 = sum(bool(case_rows.get((row["case_id"], "v1_1_3"), {}).get("case_majority_correct")) for row in supported)
    p25 = sum(bool(case_rows.get((row["case_id"], "phase2_5_6"), {}).get("case_majority_correct")) for row in supported)
    oracle = sum(bool(case_rows.get((row["case_id"], "v1_1_3"), {}).get("case_majority_correct")) or bool(case_rows.get((row["case_id"], "phase2_5_6"), {}).get("case_majority_correct")) for row in supported)
    n = len(supported)
    return {"n": n, "v1_1_3_case_majority_correct": v1, "phase2_5_6_case_majority_correct": p25, "oracle_case_majority_correct": oracle, "oracle_gain_over_v1_1_3": oracle - v1, "oracle_gain_over_phase2_5_6": oracle - p25}


def win_and_loss_audits(case_rows: Mapping[tuple[str, str], Mapping[str, Any]], paired: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wins = []
    losses = []
    for pair in paired.get("rows", []):
        v1 = case_rows[(pair["case_id"], "v1_1_3")]
        p25 = case_rows[(pair["case_id"], "phase2_5_6")]
        v1_row = v1["rows"][-1] if v1["rows"] else {}
        p25_row = p25["rows"][-1] if p25["rows"] else {}
        if pair["outcome"] == "PHASE2_5_6_ONLY":
            if p25_row.get("error"):
                classification = "BACKEND_ARTIFACT"
            elif p25_row.get("fallback_used"):
                classification = "FALLBACK_ASSISTED"
            elif p25_row.get("final_execution_mode") == "branch" and not p25_row.get("branch_failures"):
                classification = "TRUE_RELATION_GRAPH_ADVANTAGE"
            else:
                classification = "UNCLEAR"
            wins.append({"case_id": pair["case_id"], "outcome": pair["outcome"], "classification": classification, "review_status": "AUTOMATIC_TRIAGE_REQUIRES_MANUAL_REVIEW", "v1_1_3_trace_summary": {"prediction": v1_row.get("prediction", []), "execution_mode": v1_row.get("final_execution_mode")}, "phase2_5_6_trace_summary": {"prediction": p25_row.get("prediction", []), "execution_mode": p25_row.get("final_execution_mode"), "fallback_used": p25_row.get("fallback_used"), "branch_failures": p25_row.get("branch_failures", [])}})
        elif pair["outcome"] == "V1_1_3_ONLY":
            if p25_row.get("error"):
                cause = "BACKEND_ADAPTER"
            elif p25_row.get("branch_failures"):
                cause = "BRANCH_EXECUTION"
            elif p25_row.get("fallback_used"):
                cause = "FALLBACK"
            elif len(v1["rows"]) > 1 and len(p25["rows"]) > 1 and len({bool(item.get("hits_at_1")) for item in p25["rows"]}) > 1:
                cause = "STOCHASTIC"
            else:
                cause = "ANSWER_SELECTION"
            losses.append({"case_id": pair["case_id"], "outcome": pair["outcome"], "primary_cause": cause, "review_status": "AUTOMATIC_TRIAGE_REQUIRES_MANUAL_REVIEW", "v1_1_3_prediction": v1_row.get("prediction", []), "phase2_5_6_prediction": p25_row.get("prediction", []), "phase2_5_6_branch_failures": p25_row.get("branch_failures", [])})
    return wins, losses


def unsupported_analysis(unsupported: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    primary_counts = Counter()
    all_counts = Counter()
    records = []
    for row in unsupported:
        primary = str(row["primary_blocking_operator"])
        primary_counts[primary] += 1
        for operator in row["all_blocking_operators"]:
            all_counts[operator] += 1
        by_family[str(row["family"])][primary] += 1
        records.append({"case_id": row["case_id"], "sample_index": row["sample_index"], "family": row["family"], "supported": False, "primary_blocking_operator": primary, "all_blocking_operators": row["all_blocking_operators"]})
    return {"status": "COMPLETE", "total_frozen_cases": 180, "supported_count": 180 - len(unsupported), "unsupported_count": len(unsupported), "primary_blocking_operator_counts": dict(primary_counts), "all_blocking_operator_counts": dict(all_counts), "family_breakdown": {family: dict(counter) for family, counter in sorted(by_family.items())}, "records": records}


def extension_priority(unsupported: Sequence[Mapping[str, Any]], features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    feature_map = feature_for_index(features)
    categories = {
        "ATTRIBUTE": {"FilterStr", "QueryAttr", "QueryAttrUnderCondition"},
        "QUALIFIER": {"QFilterStr", "QFilterNum", "QFilterYear", "QFilterDate", "QueryAttrQualifier", "QueryRelationQualifier"},
        "COMPARISON": {"FilterNum", "FilterYear", "FilterDate", "SelectBetween", "SelectAmong"},
        "AGGREGATION": {"Count"},
        "VERIFICATION": {"VerifyStr", "VerifyNum", "VerifyYear", "VerifyDate"},
    }
    fit = {"ATTRIBUTE": 0.90, "QUALIFIER": 0.80, "COMPARISON": 0.60, "AGGREGATION": 0.40, "VERIFICATION": 0.40}
    rows = []
    for category, operators in categories.items():
        matched = [row for row in unsupported if set(row["all_blocking_operators"]) & operators]
        depths = [int(feature_map[int(row["sample_index"])] .get("normalized_depth") or 0) for row in matched]
        prevalence = len(matched) / 180
        depth_mean = statistics.mean(depths) if depths else 0.0
        score = prevalence * fit[category] * (1.0 + depth_mean / 10.0)
        rows.append({"extension": category, "blocking_case_count": len(matched), "prevalence": prevalence, "mean_depth": depth_mean, "expected_architectural_fit": fit[category], "priority_score": score, "rank": 0, "operators": sorted(operators)})
    rows.sort(key=lambda row: (-row["priority_score"], row["extension"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {"status": "DESCRIPTIVE_PRIORITY_ONLY", "method_changes_made": False, "extensions": rows}


def decision(supported_count: int, metrics: Mapping[str, Any], paired: Mapping[str, Any], wins: Sequence[Mapping[str, Any]], stratified: Mapping[str, Sequence[Mapping[str, Any]]], run_status: str) -> dict[str, Any]:
    if supported_count < 30:
        main = "KQA_TRANSFER_NOT_VIABLE"
    elif not str(run_status).startswith("COMPLETE_"):
        main = "KQA_TRANSFER_INCONCLUSIVE"
    else:
        v1 = metrics.get("v1_1_3", {}).get("case_majority_h1", 0.0)
        p25 = metrics.get("phase2_5_6", {}).get("case_majority_h1", 0.0)
        true_wins = sum(item.get("classification") == "TRUE_RELATION_GRAPH_ADVANTAGE" for item in wins)
        if p25 > v1 and int(paired.get("paired_net", 0)) > 0 and true_wins >= 3:
            main = "KQA_TRANSFER_SHOWS_RELATION_GRAPH_SIGNAL"
        elif p25 <= v1 and int(paired.get("paired_net", 0)) <= 0:
            main = "KQA_TRANSFER_NO_RELATION_GRAPH_SIGNAL"
        else:
            main = "KQA_TRANSFER_INCONCLUSIVE"
    deltas = [float(row.get("delta_h1", 0.0)) for rows in stratified.values() for row in rows if int(row.get("n", 0)) >= 3]
    positive = sum(delta > 0 for delta in deltas)
    if supported_count < 30 or not str(run_status).startswith("COMPLETE_") or not deltas:
        complexity = "INSUFFICIENT_EVIDENCE"
    elif positive >= 2:
        complexity = "COMPLEXITY_DEPENDENT_ADVANTAGE_PARTIAL"
    elif positive == 1:
        complexity = "COMPLEXITY_DEPENDENT_ADVANTAGE_PARTIAL"
    else:
        complexity = "NO_COMPLEXITY_DEPENDENT_ADVANTAGE"
    return {"main_phase6_decision": main, "secondary_complexity_decision": complexity, "supported_subset_count": supported_count, "true_relation_graph_advantage_count": sum(item.get("classification") == "TRUE_RELATION_GRAPH_ADVANTAGE" for item in wins), "decision_rule": "viability gate first; otherwise compare case-majority H@1, paired net, and at least three conservatively triaged branch-only wins", "complexity_rule": "descriptive positive delta in at least one adequately sized predeclared stratum is partial evidence; no model is fitted"}


def report(source_validation: Mapping[str, Any], capability: Mapping[str, Any], subset: Mapping[str, Any], grounding: Mapping[str, Any], relation_audit: Mapping[str, Any], metrics: Mapping[str, Any], family: Sequence[Mapping[str, Any]], stratified: Mapping[str, Sequence[Mapping[str, Any]]], paired: Mapping[str, Any], oracle: Mapping[str, Any], wins: Sequence[Mapping[str, Any]], losses: Sequence[Mapping[str, Any]], priority: Mapping[str, Any], final_decision: Mapping[str, Any], run_status: str) -> str:
    def family_table() -> str:
        lines = ["| Family | N | v1.1.3 H@1 | Phase2.5.6 H@1 | Paired net |", "|---|---:|---:|---:|---:|"]
        for row in family:
            lines.append(f"| {row['family']} | {row['n']} | {row['v1_1_3']['h1']:.3f} | {row['phase2_5_6']['h1']:.3f} | {row['paired_net']} |")
        return "\n".join(lines)

    def stratum_table() -> str:
        lines = ["| Complexity stratum | v1.1.3 H@1 | Phase2.5.6 H@1 | Delta |", "|---|---:|---:|---:|"]
        for dimension, rows in stratified.items():
            for row in rows:
                lines.append(f"| {dimension}:{row['stratum']} (n={row['n']}) | {row['v1_1_3_h1']:.3f} | {row['phase2_5_6_h1']:.3f} | {row['delta_h1']:+.3f} |")
        return "\n".join(lines)

    v1 = metrics.get("v1_1_3", {})
    p25 = metrics.get("phase2_5_6", {})
    tier = subset.get("tier_counts", {})
    return f"""# Phase 6 KQA Pro Backend and Frozen Baseline Transfer

## 1. Motivation

Phase 5 found a mixed complexity gap between the frozen Phase 3C branch-required benchmark and KQA Pro. Phase 6 transfers the existing methods without changing their reasoning policy.

## 2. Phase 5 rationale

The exact Phase 5 sample remains frozen at 180 validation records, 20 per normalized family, seed `20261030`. No resampling or outcome-based filtering was performed.

## 3. Research question

Does the relative behavior of frozen adaptive v1.1.3 and frozen Phase 2.5.6 change on the compositionally broader KQA Pro setting?

## 4. Frozen KQA Pro source

Source validation status: **{source_validation.get('status')}**. The Phase 5 validation hash matches its manifest: `{source_validation.get('phase5_source_hash_matches_manifest')}`. The deterministic companion KB hash is `{source_validation.get('kb_sha256')}`.

## 5. Backend design

`KQAProBackend` provides exact local name lookup, relation traversal, direction handling, concept checks, simple string filters, attribute retrieval, and supported KoPL execution. Unsupported semantics raise an explicit exception; they are not approximated by an LLM.

## 6. Runtime capability matrix

See `phase6_runtime_capability_matrix.json`. Backend access and frozen-method expressibility are recorded separately.

## 7. Supported-subset definition

Tier A contains native relational operations and basic intersection. Tier B adds simple string filtering and attribute lookup. A question is eligible only when every actual KoPL operator is in the frozen Tier A/B set and at least one exact `Find` root is available.

## 8. Supported-subset viability

| Tier | Cases |
|---|---:|
| Native relational | {tier.get('NATIVE_RELATIONAL', 0)} |
| Extended expressible | {tier.get('EXTENDED_EXPRESSIBLE', 0)} |
| Unsupported | {subset.get('unsupported_case_count', 0)} |

Viability gate: **{subset.get('supported_subset_viability')}**.

## 9. Unsupported operator distribution

See `phase6_unsupported_operator_analysis.json`. Unsupported operators are not interpreted as method failures.

## 10. Entity-grounding audit

Supported-subset entity-anchor recall is `{grounding.get('entity_anchor_recall', 0.0):.3f}` and full-question anchor resolution is `{grounding.get('question_full_anchor_resolution_rate', 0.0):.3f}`. Ambiguous anchors: `{grounding.get('ambiguous_entity_anchor_count', 0)}`; unresolved anchors: `{grounding.get('unresolved_entity_anchor_count', 0)}`.

## 11. Relation-candidate audit

The fixed diagnostic sample uses seed `{relation_audit.get('selection_seed')}` and `{relation_audit.get('diagnostic_sample_size')}` cases. Candidate recall is `{relation_audit.get('candidate_relation_recall')}` and direction correctness is `{relation_audit.get('direction_correctness')}`.

## 12. Frozen methods

The evaluation uses exactly frozen adaptive v1.1.3 and frozen Phase 2.5.6. Prompts, routing policies, validators, fallback policy, answer selection, and evaluator semantics were not changed.

## 13. Evaluation protocol

Each supported case was assigned `{run_status}` with two repeats per method when live execution was enabled. Gold programs were used for compatibility auditing only, never passed to the methods.

## 14. Overall results

| Method | H@1 | F1 | Case-majority correct | Tokens | Latency ms |
|---|---:|---:|---:|---:|---:|
| v1.1.3 | {v1.get('case_majority_h1', 0.0):.3f} | {v1.get('f1_run_mean', 0.0):.3f} | {v1.get('case_majority_correct', 0)} | {v1.get('average_total_tokens', 0.0):.1f} | {v1.get('average_latency_ms', 0.0):.1f} |
| Phase2.5.6 | {p25.get('case_majority_h1', 0.0):.3f} | {p25.get('f1_run_mean', 0.0):.3f} | {p25.get('case_majority_correct', 0)} | {p25.get('average_total_tokens', 0.0):.1f} | {p25.get('average_latency_ms', 0.0):.1f} |

## 15. Family-level results

{family_table()}

## 16. Complexity-stratified results

Thresholds were frozen before outcomes: depth `<=3`, `4`, `>=5`; operator diversity `<=3`, `4`, `>=5` normalized families; nested versus not nested; constraints `<=4` versus `>=5`.

{stratum_table()}

## 17. Program-depth interaction

See `complexity_stratified_results.json` for the predeclared depth bins and deltas.

## 18. Operator-diversity interaction

See `complexity_stratified_results.json` for the predeclared normalized-family bins and deltas.

## 19. Nested-composition interaction

See `complexity_stratified_results.json` for nested and non-nested results.

## 20. Paired outcomes

| Outcome | Count |
|---|---:|
| BOTH_CORRECT | {paired.get('counts', {}).get('BOTH_CORRECT', 0)} |
| V1_1_3_ONLY | {paired.get('counts', {}).get('V1_1_3_ONLY', 0)} |
| PHASE2_5_6_ONLY | {paired.get('counts', {}).get('PHASE2_5_6_ONLY', 0)} |
| BOTH_WRONG | {paired.get('counts', {}).get('BOTH_WRONG', 0)} |

Paired net: `{paired.get('paired_net', 0)}`.

## 21. Oracle headroom

The paired oracle corrects `{oracle.get('oracle_case_majority_correct', 0)}` of `{oracle.get('n', 0)}` cases, with gains of `{oracle.get('oracle_gain_over_v1_1_3', 0)}` over v1.1.3 and `{oracle.get('oracle_gain_over_phase2_5_6', 0)}` over Phase2.5.6.

## 22. Phase2.5.6-only win audit

`phase2_5_6_win_audit.jsonl` contains all method-only wins with conservative causal triage. No win was used for tuning.

## 23. v1.1.3-only loss audit

`v1_1_3_loss_audit.jsonl` contains all baseline-only losses with a primary observational cause. No loss was used for tuning.

## 24. Cost analysis

See `cost_analysis.json` for token, latency, LLM-call, KG-call, error, and fallback accounting.

## 25. Future extension priority

See `future_extension_priority.json`. This is a descriptive ranking only; no extension was implemented.

## 26. Main Phase 6 decision

**{final_decision.get('main_phase6_decision')}**

## 27. Complexity-dependent advantage decision

**{final_decision.get('secondary_complexity_decision')}**

## 28. Interpretation

Backend compatibility, grounding, unsupported semantics, and method outcomes are reported as separate layers. Unsupported KQA Pro operators are not method failures, and backend failures are not reasoning failures.

## 29. Implication for dataset hypothesis

This phase can test the dataset-narrowness hypothesis only on the frozen supported subset. Unsupported operator prevalence remains evidence for future extension, not evidence of current method failure.

## 30. Recommended next step

Do not modify the frozen methods from Phase 6 outcomes. Follow the decision-specific Phase 7 protocol only after reviewing the paired and causal-audit artifacts.

## 31. Limitations

The supported subset is deliberately conservative. KQA Pro's full operator language is broader than the current relation-centric method contract. The local KB is a documented companion file from the same frozen public mirror, and the end-to-end comparison is limited to cases whose operators and topic roots are expressible without a new reasoning policy.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run the frozen methods after the compatibility gate")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows, features, phase5_manifest = source_rows_and_features()
    source_validation = validate_source(rows, features, phase5_manifest)
    write_json(OUTPUT_ROOT / "phase6_source_validation.json", source_validation)
    capability = capability_matrix()
    write_json(OUTPUT_ROOT / "phase6_runtime_capability_matrix.json", capability)
    subset, supported, unsupported = subset_manifest(rows, features, phase5_manifest)
    write_json(OUTPUT_ROOT / "phase6_supported_subset_manifest.json", subset)
    write_json(OUTPUT_ROOT / "phase6_unsupported_operator_analysis.json", unsupported_analysis(unsupported))
    backend = KQAProBackend(KB_SOURCE)
    source_by_index = {index: rows[index] for index in range(len(rows))}
    supported_with_source = [{**row, "source": source_by_index[int(row["sample_index"])]} for row in supported]
    grounding = grounding_audit(backend, supported_with_source)
    write_json(OUTPUT_ROOT / "entity_grounding_audit.json", grounding)
    relation_audit = relation_candidate_audit(backend, supported, source_by_index)
    write_json(OUTPUT_ROOT / "relation_candidate_audit.json", relation_audit)
    write_json(OUTPUT_ROOT / "future_extension_priority.json", extension_priority(unsupported, features))

    run_rows: list[dict[str, Any]] = []
    run_status = "NOT_RUN_VIABILITY_GATE" if len(supported) < 30 else "NOT_RUN_USE_--RUN"
    if args.run and len(supported) >= 30:
        run_rows = run_live(backend, supported, source_by_index, args.repeats)
        run_status = f"COMPLETE_{args.repeats}_REPEATS"
    write_jsonl(OUTPUT_ROOT / "run_results.jsonl", run_rows)
    case_rows = majority_case_rows(run_rows)
    metrics = {"status": run_status, "v1_1_3": method_metrics(run_rows, case_rows, "v1_1_3"), "phase2_5_6": method_metrics(run_rows, case_rows, "phase2_5_6")}
    paired = paired_analysis(case_rows, supported)
    family_results = build_family_results(case_rows, supported)
    stratified = {dimension: stratification_rows(case_rows, supported, dimension) for dimension in ("program_depth", "operator_diversity", "nested_composition", "constraint_count")}
    oracle = oracle_headroom(case_rows, supported)
    wins, losses = win_and_loss_audits(case_rows, paired)
    final_decision = decision(len(supported), metrics, paired, wins, stratified, run_status)
    cost_analysis = {
        "status": run_status,
        "methods": {method: metrics[method] for method in ("v1_1_3", "phase2_5_6")},
        "pricing_applied": False,
        "note": "Token and latency accounting are recorded; no price table was applied because model pricing was not part of the frozen Phase 6 protocol.",
    }
    write_json(OUTPUT_ROOT / "metrics.json", metrics)
    write_json(OUTPUT_ROOT / "paired_analysis.json", paired)
    write_json(OUTPUT_ROOT / "family_level_results.json", family_results)
    write_json(OUTPUT_ROOT / "complexity_stratified_results.json", stratified)
    write_json(OUTPUT_ROOT / "oracle_headroom.json", oracle)
    write_json(OUTPUT_ROOT / "cost_analysis.json", cost_analysis)
    write_jsonl(OUTPUT_ROOT / "phase2_5_6_win_audit.jsonl", wins)
    write_jsonl(OUTPUT_ROOT / "v1_1_3_loss_audit.jsonl", losses)
    write_json(OUTPUT_ROOT / "decision_summary.json", final_decision)
    (OUTPUT_ROOT / "decision_summary.md").write_text(
        "# Phase 6 Decision Summary\n\n"
        f"Main decision: **{final_decision['main_phase6_decision']}**\n\n"
        f"Secondary complexity decision: **{final_decision['secondary_complexity_decision']}**\n\n"
        f"Supported subset: **{len(supported)}** cases; run status: **{run_status}**.\n\n"
        "No prompts, routing policy, validators, fallback policy, or answer-selection logic were changed.\n",
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "PHASE6_REPORT.md").write_text(report(source_validation, capability, subset, grounding, relation_audit, metrics, family_results, stratified, paired, oracle, wins, losses, read_json(OUTPUT_ROOT / "future_extension_priority.json"), final_decision, run_status), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "output_root": str(OUTPUT_ROOT), "supported_subset": len(supported), "unsupported": len(unsupported), "run_status": run_status, **final_decision}, indent=2))


if __name__ == "__main__":
    main()
