#!/usr/bin/env python3
"""Gold-only scorer for Iteration 33's frozen representation audit.

Run only after audit_runtime.py has completed successfully. The runtime audit
does not open gold files; this separate process consumes its non-gold linkage
record and independently scores the unchanged 800 frozen predictions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from typed_scoring import atom, score_sets  # noqa: E402

RESULTS = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS"
ITER30 = RESULTS / "iter_030_results_first"
ITER32 = RESULTS / "iter_032_count_producer_and_scalar_contract"
OUTDIR = RESULTS / "iter_033_scalar_generalization_and_upstream_decision"
SNAPSHOT = ITER32 / "PREDICTION_SNAPSHOT_NON_GOLD.jsonl"
FROZEN_SCORE = ITER32 / "DEVELOPMENT_SCORING_GOLD_ONLY.csv"
SCALAR_GOLD = ITER32 / "SCALAR_SOURCE_AUDIT_GOLD_ONLY.csv"
COUNT_AUDIT = ITER30 / "COUNT_OPERATOR_AUDIT.csv"
OFFICIAL_GOLD = ROOT / "datasets/grailqa/official/grailqa_v1.0_dev.json"
PAIRED_GOLD = RESULTS / "iter_028_terminal_boundary_repair/RUNS/iter29_terminal_candidate_selector_attempt3/PAIRED_EVAL_CASES_GOLD_ONLY.csv"
RUNTIME_AUDIT = OUTDIR / "RUNTIME_REPRESENTATION_AUDIT_NON_GOLD.json"

EXPECTED = {
    "prediction_snapshot_sha256": "1db208b74b418a9e623b7ea9cf79c1a338b33649d375f64feb1dafceb559fd26",
    "gold_source_sha256": "7304cef55bec08ea7601e818960166b81df0e23353423ce877ed34fc1632ef3e",
    "frozen_scoring_csv_sha256": "4415ce55e6afcde51c04cc9b3d8995b5f16832d62e35599f2e5d1aa0d0e1d582",
    "count_operator_audit_gold_only_sha256": "9e01b691a1050b8e9f0ed26db7170dc13018e938afcdd3b2bd80dd15f423e42d",
    "official_grailqa_dev_sha256": "55f3c2f937fee12bde3242247f5a06c6d595a898aa291b62fd3c2abf502f748c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    hashes = {key: {"expected": value, "actual": sha256(path)} for key, value, path in [
        ("prediction_snapshot_sha256", EXPECTED["prediction_snapshot_sha256"], SNAPSHOT),
        ("gold_source_sha256", EXPECTED["gold_source_sha256"], PAIRED_GOLD),
        ("frozen_scoring_csv_sha256", EXPECTED["frozen_scoring_csv_sha256"], FROZEN_SCORE),
        ("count_operator_audit_gold_only_sha256", EXPECTED["count_operator_audit_gold_only_sha256"], COUNT_AUDIT),
        ("official_grailqa_dev_sha256", EXPECTED["official_grailqa_dev_sha256"], OFFICIAL_GOLD),
    ]}
    for name, values in hashes.items():
        if values["actual"] != values["expected"]:
            raise SystemExit(f"frozen source hash mismatch for {name}: {values}")

    runtime = json.loads(RUNTIME_AUDIT.read_text(encoding="utf-8"))
    if runtime.get("status") != "PASS_NON_GOLD_RUNTIME_REPRODUCTION" or runtime.get("gold_accessed") is not False:
        raise SystemExit("the independent non-gold runtime audit is absent or failed")
    runtime_cases = {row["case_id"]: row for row in runtime["accepted_cases"]}

    snapshot_rows = jsonl(SNAPSHOT)
    snapshot_by_case = {row["case_id"]: row for row in snapshot_rows}
    frozen = csv_rows(FROZEN_SCORE)
    scalar_rows = csv_rows(SCALAR_GOLD)
    count_rows = csv_rows(COUNT_AUDIT)
    official_rows = json.loads(OFFICIAL_GOLD.read_text(encoding="utf-8"))
    official_by_qid: dict[str, dict[str, Any]] = {}
    for row in official_rows:
        qid = str(row["qid"])
        if qid in official_by_qid:
            raise SystemExit(f"duplicate official GrailQA qid {qid}")
        official_by_qid[qid] = row

    if not (len(snapshot_rows) == len(frozen) == 800):
        raise SystemExit("expected exactly 800 frozen prediction/scoring rows")
    if len({row["case_id"] for row in frozen}) != 800:
        raise SystemExit("duplicate case IDs in frozen scoring rows")
    frozen_by_case = {row["case_id"]: row for row in frozen}
    scalar_by_key = {row["question_key"]: row for row in scalar_rows}
    count_by_key = {row["question_key"]: row for row in count_rows}

    legacy_hits = 0
    typed_hits = 0
    legacy_f1: list[float] = []
    typed_f1: list[float] = []
    wins = losses = ties = 0
    cases: list[dict[str, Any]] = []
    accepted_correct = 0
    gold_atom_types: Counter[str] = Counter()

    for frozen_row in frozen:
        case_id = frozen_row["case_id"]
        snapshot = snapshot_by_case.get(case_id)
        if snapshot is None:
            raise SystemExit(f"missing non-gold snapshot for {case_id}")
        question_key = frozen_row["question_key"]
        prefix, sep, qid = question_key.partition("::")
        if prefix != "grailqa" or not sep or qid not in official_by_qid:
            raise SystemExit(f"official GrailQA crosswalk failed for {question_key}")
        official = official_by_qid[qid]
        if official.get("question") != snapshot.get("question"):
            raise SystemExit(f"official question text differs for {question_key}")

        gold_items = official.get("answer") or []
        typed_gold = {
            atom(str(item["answer_type"]), item["answer_argument"])
            for item in gold_items
        }
        flat_gold = {str(item["answer_argument"]) for item in gold_items}
        stored_gold = set(json.loads(frozen_row["gold_answer_ids_scoring_only"]))
        if flat_gold != stored_gold:
            raise SystemExit(f"official typed gold / frozen gold argument crosswalk differs for {question_key}")
        for value in typed_gold:
            gold_atom_types[value[0]] += 1

        baseline_ids = {str(x) for x in snapshot.get("baseline_prediction_ids", [])}
        stored_prediction_ids = set(json.loads(frozen_row["legacy_entity_prediction_ids"]))
        if baseline_ids != stored_prediction_ids:
            raise SystemExit(f"snapshot / frozen entity prediction differs for {question_key}")
        legacy_overlap = baseline_ids & flat_gold
        legacy_precision = len(legacy_overlap) / len(baseline_ids) if baseline_ids else 0.0
        legacy_recall = len(legacy_overlap) / len(flat_gold) if flat_gold else 0.0
        legacy_f1_value = (
            2 * legacy_precision * legacy_recall / (legacy_precision + legacy_recall)
            if legacy_precision + legacy_recall else 0.0
        )
        old_hit = int(bool(legacy_overlap))

        sidecar = snapshot.get("typed_answer") or {}
        if sidecar.get("answer_kind") == "aggregate" and sidecar.get("status") == "accepted":
            lexical = str(sidecar.get("lexical_value", ""))
            if not lexical or case_id not in runtime_cases:
                raise SystemExit(f"accepted scalar lacks exact non-gold provenance for {case_id}")
            typed_pred = {atom("Value", lexical)}
            if typed_pred & typed_gold:
                accepted_correct += 1
        else:
            typed_pred = {atom("Entity", value) for value in baseline_ids}
        new_score = score_sets(typed_pred, typed_gold)

        frozen_old_hit = int(frozen_row["legacy_entity_hits_at_1"])
        frozen_typed_hit = int(frozen_row["typed_hits_at_1"])
        if old_hit != frozen_old_hit or new_score.hit != frozen_typed_hit:
            raise SystemExit(f"recomputed hit label differs from frozen score for {question_key}")
        legacy_hits += old_hit
        typed_hits += new_score.hit
        legacy_f1.append(legacy_f1_value)
        typed_f1.append(new_score.f1)
        if new_score.hit > old_hit:
            wins += 1
        elif new_score.hit < old_hit:
            losses += 1
        else:
            ties += 1
        cases.append({
            "case_id": case_id,
            "question_key": question_key,
            "legacy_prediction_strings": sorted(baseline_ids),
            "typed_prediction_atoms": sorted(typed_pred),
            "gold_atoms": sorted(typed_gold),
            "legacy_gold_argument_strings": sorted(flat_gold),
            "legacy_hit": old_hit,
            "typed_hit": new_score.hit,
            "legacy_f1": legacy_f1_value,
            "typed_f1": new_score.f1,
            "representation_changed": typed_pred != {atom("Entity", value) for value in baseline_ids},
        })

    aggregates = {
        "n": len(cases),
        "legacy_entity_only": {
            "hits_at_1": legacy_hits,
            "hits_at_1_rate": legacy_hits / len(cases),
            "mean_set_f1": mean(legacy_f1),
        },
        "typed_mixed_supplementary": {
            "hits_at_1": typed_hits,
            "hits_at_1_rate": typed_hits / len(cases),
            "mean_typed_set_f1": mean(typed_f1),
        },
        "paired_hit_outcomes": {"wins": wins, "losses": losses, "ties": ties},
        "accepted_scalars": len(runtime_cases),
        "accepted_scalar_correct": accepted_correct,
        "accepted_scalar_incorrect": len(runtime_cases) - accepted_correct,
        "accepted_scalar_precision": accepted_correct / len(runtime_cases) if runtime_cases else 0.0,
        "gold_answer_atom_counts_by_type": dict(gold_atom_types),
        "cases_with_baseline_entity_predictions": sum(bool(row.get("baseline_prediction_ids")) for row in snapshot_rows),
    }
    expected_metrics = (legacy_hits, typed_hits, wins, losses, ties, len(runtime_cases), accepted_correct)
    if expected_metrics != (578, 614, 36, 0, 764, 39, 36):
        raise SystemExit(f"metric hard gate mismatch: {expected_metrics}")
    if abs(aggregates["legacy_entity_only"]["mean_set_f1"] - 0.6854957005036327) > 1e-12:
        raise SystemExit("legacy mean F1 hard gate mismatch")
    if abs(aggregates["typed_mixed_supplementary"]["mean_typed_set_f1"] - 0.7304957005036329) > 1e-12:
        raise SystemExit("typed mean F1 hard gate mismatch")

    count_items = [row for row in scalar_rows if row.get("serializer_status") == "AGGREGATE"]
    incorrect = [row for row in count_items if row.get("lexical_value") != row.get("gold_value_scoring_only")]
    if len(count_items) != 39 or len(incorrect) != 3:
        raise SystemExit("accepted scalar / incorrect scalar count mismatch")
    noncount_rows = [row for row in frozen if row.get("gold_root_count_key_scoring_only") == "False"]
    count_root_rows = [row for row in frozen if row.get("gold_root_count_key_scoring_only") == "True"]
    hard_negative_conversions = sum(1 for row in noncount_rows if row.get("serializer_sidecar_kind") == "aggregate")
    if len(count_rows) != 68 or len(noncount_rows) != 732 or len(count_root_rows) != 68 or hard_negative_conversions != 0:
        raise SystemExit("COUNT/non-COUNT negative-control audit mismatch")

    wrong_keys = {row["question_key"] for row in incorrect}
    expected_wrong = {
        "grailqa::2105476009000",
        "grailqa::2103115005000",
        "grailqa::4301524012000",
    }
    if wrong_keys != expected_wrong:
        raise SystemExit(f"unexpected development scalar error set: {sorted(wrong_keys)}")
    six_rejected = [
        row for row in scalar_rows
        if row.get("baseline_name_equals_gold_value_scoring_only") == "True"
        and row.get("serializer_status") != "AGGREGATE"
    ]
    if len(six_rejected) != 6:
        raise SystemExit(f"expected six apparent numeric matches rejected by frozen serializer, got {len(six_rejected)}")
    apparent_matches = [row for row in scalar_rows if row.get("baseline_name_equals_gold_value_scoring_only") == "True"]
    if len(apparent_matches) != 42:
        raise SystemExit(f"expected 42 apparent numeric name matches, got {len(apparent_matches)}")

    wrong_detail = []
    for row in incorrect:
        count = count_by_key[row["question_key"]]
        wrong_detail.append({
            "question_key": row["question_key"],
            "question": row["question"],
            "predicted_value": row["lexical_value"],
            "gold_value": row["gold_value_scoring_only"],
            "runtime_entity_search_result_count": count["runtime_entity_search_result_count"],
            "runtime_query_result_count_below_1000": count["runtime_query_result_count_below_1000"],
            "relation_set_matches_gold": count["runtime_relation_set_matches_gold_scoring_only"],
            "target_type_filter_present": count["runtime_final_query_has_type_filter"],
            "interpretation_limit": "Development-only symptom description; these records do not prove a unique causal mechanism.",
        })

    audit = {
        "status": "PASS_RETROSPECTIVE_REPRESENTATION_REPRODUCTION",
        "iteration": 33,
        "claim_class": "DEV_REPRESENTATION_RECOVERY",
        "prospective_evaluation": "NOT_RUN",
        "network_calls": 0,
        "gold_scoring_process_separate_from_runtime_audit": True,
        "gold_type_source": "official GrailQA dev per-answer answer_type/answer_argument",
        "scoring_contract_note": "The typed mixed metric is supplementary and uses exact type-tagged answer atoms; it is not a new inference result or an official GrailQA benchmark score.",
        "source_hashes": hashes,
        "runtime_audit_sha256": sha256(RUNTIME_AUDIT),
        "frozen_score_csv_sha256": sha256(FROZEN_SCORE),
        "scalar_gold_audit_sha256": sha256(SCALAR_GOLD),
        "metrics": aggregates,
        "scalar_error_review": wrong_detail,
        "six_apparent_numeric_matches_rejected": [
            {"question_key": row["question_key"], "question": row["question"], "router_answer_type": row["router_requested_final_answer_type"], "reason": row["serializer_reason"]}
            for row in six_rejected
        ],
        "gold_answer_files_opened_only_in_this_process": [str(OFFICIAL_GOLD.relative_to(ROOT)), str(PAIRED_GOLD.relative_to(ROOT)), str(FROZEN_SCORE.relative_to(ROOT)), str(SCALAR_GOLD.relative_to(ROOT)), str(COUNT_AUDIT.relative_to(ROOT))],
        "runtime_process_gold_files_opened": [],
        "all_800_question_crosswalk_and_frozen_hit_labels_verified": True,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "REPRESENTATION_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUTDIR / "REPRESENTATION_CASES.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        for row in cases:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    print(json.dumps({"status": audit["status"], "metrics": aggregates, "output": str((OUTDIR / "REPRESENTATION_AUDIT.json").relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
