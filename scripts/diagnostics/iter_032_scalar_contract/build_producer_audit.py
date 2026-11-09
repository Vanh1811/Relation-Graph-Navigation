#!/usr/bin/env python3
"""Build the gold-free producer matrix from Iter31 runtime audits."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from chain_of_relations.methods.iter_030_operator_recovery.relation_aware_count import classify_count_intent


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS"
ITER31 = RESULTS / "iter_031_gold_free_count_readiness"
OUTPUT = RESULTS / "iter_032_count_producer_and_scalar_contract"
SNAPSHOT = OUTPUT / "PREDICTION_SNAPSHOT_NON_GOLD.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if not SNAPSHOT.exists():
        raise SystemExit("freeze the non-gold prediction snapshot first")
    snapshots = [json.loads(line) for line in SNAPSHOT.read_text(encoding="utf-8").splitlines() if line.strip()]
    targets = read_csv(ITER31 / "TARGET_TYPE_RESOLVER_AUDIT.csv")
    paths = read_csv(ITER31 / "PATH_COMPLETENESS_AUDIT.csv")
    if len(targets) != 68 or len(paths) != 68:
        raise SystemExit("Iteration 31 must provide the frozen 68-case runtime-only producer audit")
    if any("gold" in key.casefold() for row in targets + paths for key in row):
        raise SystemExit("producer matrix runtime source unexpectedly contains gold-only fields")

    snapshot_by_key = {row["question_key"]: row for row in snapshots}
    target_by_key = {row["question_key"]: row for row in targets}
    path_by_key = {row["question_key"]: row for row in paths}
    if set(target_by_key) != set(path_by_key) or not set(target_by_key).issubset(snapshot_by_key):
        raise SystemExit("Iteration 31 producer audit keys do not match the frozen Iteration 29 cohort")

    controls: list[dict[str, Any]] = []
    for row in snapshots:
        triggered, reason = classify_count_intent(str(row.get("question", "")))
        if not triggered:
            controls.append({
                "case_id": row["case_id"],
                "question_key": row["question_key"],
                "question": row["question"],
                "trigger_reason": reason,
            })
    control_path = OUTPUT / "PREDECLARED_NEGATIVE_CONTROLS_NON_GOLD.json"
    control_payload = {
        "status": "FROZEN_ITER31_RUNTIME_TRIGGER_NEGATIVE_COHORT",
        "selection_rule": "all 800 frozen Iteration 29 questions rejected by the existing Iteration 30 count-intent rule; no posthoc subsampling",
        "source_prediction_snapshot_sha256": sha256(SNAPSHOT),
        "n": len(controls),
        "gold_read": False,
        "controls": controls,
    }
    if control_path.exists():
        existing = json.loads(control_path.read_text(encoding="utf-8"))
        if existing != control_payload:
            raise SystemExit("existing trigger-negative cohort differs; refusing overwrite")
    else:
        control_path.write_text(json.dumps(control_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    runtime_sources = {
        "iter29_trace_sha256": sha256(RESULTS / "iter_028_terminal_boundary_repair/RUNS/iter29_terminal_candidate_selector_attempt3/CASE_TRACES_NON_GOLD.jsonl"),
        "iter29_request_events_sha256": sha256(RESULTS / "iter_028_terminal_boundary_repair/RUNS/iter29_terminal_candidate_selector_attempt3/REQUEST_EVENTS.jsonl"),
        "iter29_request_journal_sha256": sha256(RESULTS / "iter_028_terminal_boundary_repair/RUNS/iter29_terminal_candidate_selector_attempt3/REQUEST_JOURNAL.jsonl"),
        "iter31_target_type_audit_sha256": sha256(ITER31 / "TARGET_TYPE_RESOLVER_AUDIT.csv"),
        "iter31_path_audit_sha256": sha256(ITER31 / "PATH_COMPLETENESS_AUDIT.csv"),
        "iter32_non_gold_snapshot_sha256": sha256(SNAPSHOT),
        "iter32_frozen_negative_controls_sha256": sha256(control_path),
        "iter32_runtime_audit_source_sha256": sha256(Path(__file__).with_name("audit_runtime.py")),
        "iter32_producer_audit_source_sha256": sha256(Path(__file__)),
    }
    (OUTPUT / "RUNTIME_SOURCE_HASHES.json").write_text(
        json.dumps(runtime_sources, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fields = [
        "row_type", "case_id", "question_key", "question", "runtime_count_intent",
        "target_type_status", "target_type_evidence_count", "target_type_ids_runtime_only",
        "path_status", "query_status", "expected_plan_present", "plan_producer_present",
        "producer_status", "readiness", "rejection_reason", "source_class",
    ]
    matrix: list[dict[str, Any]] = []
    for key in sorted(target_by_key, key=lambda item: int(target_by_key[item].get("trace_line_1based", "0"))):
        t = target_by_key[key]
        p = path_by_key[key]
        s = snapshot_by_key[key]
        matrix.append({
            "row_type": "posthoc_selected_count_dev_case",
            "case_id": s["case_id"],
            "question_key": key,
            "question": s["question"],
            "runtime_count_intent": classify_count_intent(s["question"])[0],
            "target_type_status": t.get("resolution_status", ""),
            "target_type_evidence_count": t.get("explicit_target_type_constraint_count", ""),
            "target_type_ids_runtime_only": t.get("canonical_target_type_ids_runtime_only", ""),
            "path_status": p.get("path_completeness_status", ""),
            "query_status": p.get("query_status", ""),
            "expected_plan_present": p.get("expected_finalized_plan_present", ""),
            "plan_producer_present": False,
            "producer_status": "PRODUCER_SOURCE_UNAVAILABLE",
            "readiness": False,
            "rejection_reason": ";".join(filter(None, [
                "no_runtime_count_plan_producer",
                "target_type_" + t.get("resolution_status", "unknown"),
                p.get("path_completeness_status", "path_status_unknown"),
            ])),
            "source_class": "archived_non_gold_trace_and_iter31_runtime_audit",
        })
    for control in controls:
        matrix.append({
            "row_type": "frozen_trigger_negative_control",
            "case_id": control["case_id"],
            "question_key": control["question_key"],
            "question": control["question"],
            "runtime_count_intent": False,
            "target_type_status": "not_resolved",
            "target_type_evidence_count": 0,
            "target_type_ids_runtime_only": "[]",
            "path_status": "not_applicable_trigger_not_met",
            "query_status": "not_executed_by_count_extension",
            "expected_plan_present": False,
            "plan_producer_present": False,
            "producer_status": "PRODUCER_SOURCE_UNAVAILABLE",
            "readiness": False,
            "rejection_reason": "frozen_trigger_negative_control; no_count_plan_producer",
            "source_class": "frozen_non_gold_runtime_trace",
        })

    synthetic_cases = [
        ("SCALAR_NEG_01", "What year was the 1998 tournament held?", "year_in_question"),
        ("SCALAR_NEG_02", "Which entity has numeric identifier m.123?", "numeric_entity_id"),
        ("SCALAR_NEG_03", "How much revenue did the company earn?", "money_amount"),
        ("SCALAR_NEG_04", "What year did the tournament begin?", "year_date"),
        ("SCALAR_NEG_05", "Who was the number one ranked player?", "ordinal"),
        ("SCALAR_NEG_06", "Which trial has the least number of participants?", "minimum_property"),
        ("SCALAR_NEG_07", "How many teams are in the league?", "entity_answer_not_scalar"),
        ("SCALAR_NEG_08", "What number was assigned to the album?", "numeric_identifier_property"),
        ("SCALAR_NEG_09", "How many people appeared in the series?", "malformed_decimal"),
        ("SCALAR_NEG_10", "How many people appeared in the series?", "negative_integer"),
        ("SCALAR_NEG_11", "How many people appeared in the series?", "leading_zero"),
        ("SCALAR_NEG_12", "How many people appeared in the series?", "conflicting_answers"),
        ("SCALAR_NEG_13", "How many people appeared in the series?", "label_and_number"),
        ("SCALAR_NEG_14", "How many people appeared in the series?", "finish_reason_length"),
        ("SCALAR_NEG_15", "How many people appeared in the series?", "empty_response"),
        ("SCALAR_NEG_16", "How many people appeared in the series?", "candidate_literal_collision"),
        ("SCALAR_POS_00", "How many people appeared in the series?", "valid_zero_vs_missing"),
    ]
    for control_id, question, rejection_reason in synthetic_cases:
        matrix.append({
            "row_type": "synthetic_scalar_contract_control",
            "case_id": control_id,
            "question_key": control_id,
            "question": question,
            "runtime_count_intent": classify_count_intent(question)[0],
            "target_type_status": "synthetic_test_only",
            "target_type_evidence_count": "",
            "target_type_ids_runtime_only": "[]",
            "path_status": "synthetic_test_only",
            "query_status": "not_run",
            "expected_plan_present": False,
            "plan_producer_present": False,
            "producer_status": "NOT_A_PRODUCER_CASE",
            "readiness": False,
            "rejection_reason": rejection_reason,
            "source_class": "synthetic_adversarial_fixture",
        })

    with (OUTPUT / "PRODUCER_READINESS_MATRIX.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(matrix)

    path_counts = Counter(row.get("path_completeness_status", "") for row in paths)
    type_counts = Counter(row.get("resolution_status", "") for row in targets)
    rejection_counts = Counter(row["rejection_reason"] for row in matrix)
    result = {
        "status": "GOLD_FREE_PRODUCER_AUDIT",
        "count_dev_cases": len(targets),
        "runtime_type_resolution_status_counts": dict(sorted(type_counts.items())),
        "path_completeness_status_counts": dict(sorted(path_counts.items())),
        "real_gold_free_producer_cases": 0,
        "count_dev_readiness": "0/68",
        "trigger_negative_control_cases": len(controls),
        "synthetic_scalar_contract_controls": len(synthetic_cases),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "gold_read": False,
        "network_calls": 0,
        "sources": {
            "target_type_audit_sha256": sha256(ITER31 / "TARGET_TYPE_RESOLVER_AUDIT.csv"),
            "path_completeness_audit_sha256": sha256(ITER31 / "PATH_COMPLETENESS_AUDIT.csv"),
            "runtime_snapshot_sha256": sha256(SNAPSHOT),
        },
    }
    (OUTPUT / "REJECTION_REASON_COUNTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
