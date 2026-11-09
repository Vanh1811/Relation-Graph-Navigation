#!/usr/bin/env python3
"""Score an already-frozen Iteration 32 prediction snapshot, gold-only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from chain_of_relations.methods.iter_032_scalar_contract import score_typed_answer
from chain_of_relations.methods.iter_032_scalar_contract.answer_contract import CONTRACT_VERSION


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS"
RUN = RESULTS / "iter_028_terminal_boundary_repair/RUNS/iter29_terminal_candidate_selector_attempt3"
OUTPUT = RESULTS / "iter_032_count_producer_and_scalar_contract"
COUNT_AUDIT = RESULTS / "iter_030_results_first/COUNT_OPERATOR_AUDIT.csv"


def score_sets(gold: set[str], predicted: set[str]) -> tuple[int, float]:
    overlap = len(gold & predicted)
    if not gold or not predicted:
        return 0, 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return int(overlap > 0), 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    snapshot_path = args.output / "PREDICTION_SNAPSHOT_NON_GOLD.jsonl"
    if not snapshot_path.exists():
        raise SystemExit("the non-gold snapshot must be frozen before gold-only scoring")
    snapshot_rows = [json.loads(line) for line in snapshot_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(snapshot_rows) != 800 or len({row["question_key"] for row in snapshot_rows}) != 800:
        raise SystemExit("the frozen prediction snapshot must contain 800 unique cases")

    gold_path = RUN / "PAIRED_EVAL_CASES_GOLD_ONLY.csv"
    with gold_path.open(newline="", encoding="utf-8") as handle:
        gold_rows = {row["question_key"]: row for row in csv.DictReader(handle)}
    if set(gold_rows) != {row["question_key"] for row in snapshot_rows}:
        raise SystemExit("gold-only scoring source does not exactly match frozen snapshot IDs")

    with COUNT_AUDIT.open(newline="", encoding="utf-8") as handle:
        count_audit_rows = {row["question_key"]: row for row in csv.DictReader(handle)}
    count_keys = set(count_audit_rows)

    scored: list[dict[str, Any]] = []
    for row in snapshot_rows:
        gold = set(map(str, json.loads(gold_rows[row["question_key"]]["gold_answer_ids_scoring_only"])))
        baseline_ids = set(map(str, row.get("baseline_prediction_ids", [])))
        typed_sidecar = row.get("typed_answer")
        if isinstance(typed_sidecar, dict) and typed_sidecar.get("answer_kind") == "aggregate":
            scoring_answer = typed_sidecar
        elif baseline_ids:
            # Compatibility contract: every scalar rejection keeps the exact
            # historic entity-ID prediction, even if its diagnostic sidecar is
            # unknown or invalid.
            scoring_answer = {
                "contract_version": CONTRACT_VERSION,
                "answer_kind": "entity",
                "entities": [{"id": entity_id, "name": ""} for entity_id in sorted(baseline_ids)],
            }
        else:
            scoring_answer = {"contract_version": CONTRACT_VERSION, "answer_kind": "unknown"}
        typed_score = score_typed_answer(scoring_answer, sorted(gold))
        old_hit, old_f1 = score_sets(gold, baseline_ids)
        scored.append({
            "case_id": row["case_id"],
            "question_key": row["question_key"],
            "gold_answer_ids_scoring_only": json.dumps(sorted(gold)),
            "gold_root_count_key_scoring_only": row["question_key"] in count_keys,
            "legacy_entity_prediction_ids": json.dumps(sorted(baseline_ids)),
            "legacy_entity_hits_at_1": old_hit,
            "legacy_entity_f1": old_f1,
            "typed_answer_kind": scoring_answer.get("answer_kind", "unknown"),
            "serializer_sidecar_kind": typed_sidecar.get("answer_kind", "unknown") if isinstance(typed_sidecar, dict) else "unknown",
            "typed_prediction_atoms": json.dumps(typed_score["prediction_atoms"]),
            "typed_hits_at_1": typed_score["hits_at_1"],
            "typed_f1": typed_score["f1"],
            "serializer_status": row.get("serializer_status", ""),
        })

    with (args.output / "DEVELOPMENT_SCORING_GOLD_ONLY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)

    with (args.output / "SCALAR_SOURCE_AUDIT.csv").open(newline="", encoding="utf-8") as handle:
        runtime_rows = list(csv.DictReader(handle))
    joined_audit = []
    for runtime in runtime_rows:
        gold_row = count_audit_rows[runtime["question_key"]]
        try:
            gold_names = json.loads(gold_row.get("runtime_baseline_answer_names", "[]"))
        except (TypeError, ValueError):
            gold_names = []
        gold_value = str(gold_row.get("gold_value_scoring_only", ""))
        runtime_accept = runtime.get("answer_kind") == "aggregate"
        joined_audit.append({
            **runtime,
            "gold_value_scoring_only": gold_value,
            "gold_answer_ids_scoring_only": gold_row.get("gold_answer_ids_scoring_only", ""),
            "baseline_name_equals_gold_value_scoring_only": bool(gold_value and gold_value in gold_names),
            "serialized_scalar_equals_gold_value_scoring_only": bool(runtime_accept and runtime.get("lexical_value") == gold_value),
            "gold_only_fields_used_for_serialization": False,
        })
    with (args.output / "SCALAR_SOURCE_AUDIT_GOLD_ONLY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(joined_audit[0]))
        writer.writeheader()
        writer.writerows(joined_audit)

    def metrics(prefix: str) -> dict[str, Any]:
        hit_key = f"{prefix}_hits_at_1"
        f1_key = f"{prefix}_f1"
        return {
            "hits_at_1": sum(row[hit_key] for row in scored),
            "hits_at_1_rate": sum(row[hit_key] for row in scored) / len(scored),
            "mean_set_f1": sum(row[f1_key] for row in scored) / len(scored),
        }

    kind_counts: dict[str, int] = {}
    for row in scored:
        kind_counts[row["serializer_sidecar_kind"]] = kind_counts.get(row["serializer_sidecar_kind"], 0) + 1
    count_rows = [row for row in scored if row["gold_root_count_key_scoring_only"]]
    accepted_count_rows = [row for row in count_rows if row["typed_answer_kind"] == "aggregate"]
    noncount_aggregates = [row for row in scored if not row["gold_root_count_key_scoring_only"] and row["typed_answer_kind"] == "aggregate"]
    count_trigger_misses = [row for row in count_rows if row["typed_answer_kind"] != "aggregate"]
    typed_hits_delta = [row["typed_hits_at_1"] - row["legacy_entity_hits_at_1"] for row in scored]
    summary = {
        "status": "DEVELOPMENTAL_RETROSPECTIVE_SCORING_ONLY",
        "prediction_snapshot_sha256": __import__("hashlib").sha256(snapshot_path.read_bytes()).hexdigest(),
        "gold_source_sha256": __import__("hashlib").sha256(gold_path.read_bytes()).hexdigest(),
        "count_operator_audit_gold_only_sha256": __import__("hashlib").sha256(COUNT_AUDIT.read_bytes()).hexdigest(),
        "n": len(scored),
        "legacy_entity_only_contract": metrics("legacy_entity"),
        "typed_mixed_contract": metrics("typed"),
        "serializer_sidecar_kind_counts": dict(sorted(kind_counts.items())),
        "unchanged_entity_cases": sum(bool(row.get("baseline_prediction_ids")) for row in snapshot_rows),
        "entity_compatibility_passthrough_cases": sum(
            bool(row.get("baseline_prediction_ids"))
            and row.get("typed_answer", {}).get("answer_kind") != "entity"
            for row in snapshot_rows
        ),
        "converted_scalar_cases": sum(row["serializer_sidecar_kind"] == "aggregate" for row in scored),
        "typed_contract_hit_changes_vs_entity_only": {
            "wins": sum(delta > 0 for delta in typed_hits_delta),
            "losses": sum(delta < 0 for delta in typed_hits_delta),
            "ties": sum(delta == 0 for delta in typed_hits_delta),
        },
        "count_operator_subset_scoring_only": {
            "n": len(count_rows),
            "legacy_hits_at_1": sum(row["legacy_entity_hits_at_1"] for row in count_rows),
            "legacy_f1_sum": sum(row["legacy_entity_f1"] for row in count_rows),
            "runtime_supported_scalar_conversions": len(accepted_count_rows),
            "typed_hits_at_1": sum(row["typed_hits_at_1"] for row in count_rows),
            "typed_f1_sum": sum(row["typed_f1"] for row in count_rows),
            "unsupported_or_unconverted_count_cases": len(count_trigger_misses),
            "exact_gold_numeric_matches_among_converted": sum(row["typed_hits_at_1"] for row in accepted_count_rows),
        },
        "hard_negative_false_conversions_gold_only": {
            "non_count_count_operator_cases": len(scored) - len(count_rows),
            "converted_scalar_cases": len(noncount_aggregates),
            "question_keys": [row["question_key"] for row in noncount_aggregates],
        },
        "apparent_numeric_name_match_audit_scoring_only": {
            "apparent_matches_of_68": sum(row["baseline_name_equals_gold_value_scoring_only"] for row in joined_audit),
            "runtime_supported_scalars_among_apparent_matches": sum(
                row["baseline_name_equals_gold_value_scoring_only"] and row["answer_kind"] == "aggregate"
                for row in joined_audit
            ),
            "correct_typed_scalars_among_apparent_matches": sum(
                row["baseline_name_equals_gold_value_scoring_only"] and row["serialized_scalar_equals_gold_value_scoring_only"]
                for row in joined_audit
            ),
            "apparent_matches_rejected_by_frozen_runtime_contract": sum(
                row["baseline_name_equals_gold_value_scoring_only"] and row["answer_kind"] != "aggregate"
                for row in joined_audit
            ),
            "accepted_scalars_incorrect_under_gold_scoring": sum(
                row["answer_kind"] == "aggregate" and not row["serialized_scalar_equals_gold_value_scoring_only"]
                for row in joined_audit
            ),
        },
        "scores_are_not_comparable_as_method_improvement": True,
        "historical_iter29_artifacts_modified": False,
    }
    (args.output / "DEVELOPMENT_SCORING_SUMMARY_GOLD_ONLY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
