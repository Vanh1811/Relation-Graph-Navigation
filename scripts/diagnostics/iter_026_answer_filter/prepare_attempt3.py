"""Freeze a disjoint, balanced I26 attempt-3 cohort offline."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.iter_026_answer_filter import cohort


ROOT = cohort.ROOT
I26 = cohort.OUT
PREP = I26 / "ATTEMPT3_PREPARATION"
ATTEMPT2_COHORT = I26 / "COHORT_MANIFEST.json"
ATTEMPT2_MANIFEST = I26 / "LIVE_MANIFEST_ATTEMPT2.json"
SEED = "iter26-attempt3-independent-grailqa-dev-balanced-v2-2026-11-08"
TARGET_PER_CELL = 3
BRANCH_FAMILIES = ("BRANCH_FINAL", "BRANCH_INTERMEDIATE")
CARDINALITIES = ("SINGLE", "MULTI")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rank_key(cardinality: str, branch_family: str, case_id: str) -> str:
    text = f"{SEED}|grailqa|dev|{cardinality}|{branch_family}|{case_id}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parent_manifest = json.loads(ATTEMPT2_COHORT.read_text(encoding="utf-8"))
    attempt2_ids = {
        cohort._id_from_key(key) for key in parent_manifest.get("case_keys", [])
    }
    attempt2_ids.discard(None)
    if len(attempt2_ids) != 12:
        raise RuntimeError(f"expected 12 frozen attempt-2 identities, got {len(attempt2_ids)}")

    fresh_pool, _, prior_ids, _ = cohort.load_fresh_pool()
    fresh50_ids = cohort._fresh50_ids()
    prior14_25_ids = cohort._continuous_agent_ids()
    known_ids = set(prior_ids) | fresh50_ids | prior14_25_ids | attempt2_ids
    raw = cohort.raw_cases()
    prior_texts = [
        (case_id, raw[case_id]["question"])
        for case_id in sorted(known_ids)
        if case_id in raw and raw[case_id].get("question")
    ]
    prior_normalized = {
        cohort.normalize_question(question): case_id
        for case_id, question in prior_texts
    }

    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {
        (cardinality, family): []
        for cardinality in CARDINALITIES
        for family in BRANCH_FAMILIES
    }
    for row in fresh_pool:
        if row.get("dataset") != "grailqa" or row.get("split") != "dev":
            continue
        if row["case_id"] in known_ids:
            continue
        answer_count = len(set(row.get("gold_answers", [])))
        if answer_count < 1:
            continue
        cardinality = "MULTI" if answer_count > 1 else "SINGLE"
        family = str(row.get("gold_structure", ""))
        if (cardinality, family) in candidates:
            candidates[(cardinality, family)].append(row)

    for (cardinality, family), values in candidates.items():
        values.sort(key=lambda row: _rank_key(cardinality, family, row["case_id"]))

    exclusions = Counter()
    selected: list[dict[str, Any]] = []
    exclusion_examples: list[dict[str, Any]] = []
    for cell in (
        (cardinality, family)
        for cardinality in CARDINALITIES
        for family in BRANCH_FAMILIES
    ):
        chosen: list[dict[str, Any]] = []
        for row in candidates[cell]:
            question = row["question"]
            normalized = cohort.normalize_question(question)
            if normalized in prior_normalized:
                exclusions["normalized_exact_prior"] += 1
                exclusion_examples.append({
                    "candidate_id": row["case_id"],
                    "reason": "NORMALIZED_EXACT_PRIOR",
                    "matched_prior_id": prior_normalized[normalized],
                })
                continue
            near = None
            for prior_id, prior_question in prior_texts:
                duplicate, jaccard, sequence = cohort._lexical_duplicate(question, prior_question)
                if duplicate:
                    near = {
                        "matched_prior_id": prior_id,
                        "token_jaccard": jaccard,
                        "sequence_ratio": sequence,
                    }
                    break
            if near:
                exclusions["lexical_near_duplicate_prior"] += 1
                exclusion_examples.append({
                    "candidate_id": row["case_id"],
                    "reason": "LEXICAL_NEAR_DUPLICATE_PRIOR",
                    **near,
                })
                continue
            duplicate_selected = False
            for previous in selected + chosen:
                duplicate, _, _ = cohort._lexical_duplicate(question, previous["question"])
                if duplicate:
                    duplicate_selected = True
                    break
            if duplicate_selected:
                exclusions["lexical_duplicate_within_cohort"] += 1
                continue
            chosen.append(row)
            if len(chosen) == TARGET_PER_CELL:
                break
        if len(chosen) != TARGET_PER_CELL:
            raise RuntimeError(
                f"insufficient clean records in {cell}: required {TARGET_PER_CELL}, found {len(chosen)}"
            )
        selected.extend(chosen)

    selected.sort(key=lambda row: (
        "SINGLE" if len(set(row.get("gold_answers", []))) == 1 else "MULTI",
        str(row.get("gold_structure", "")),
        row["case_id"],
    ))
    if len(selected) != 12 or len({row["case_id"] for row in selected}) != 12:
        raise RuntimeError("attempt-3 cohort must contain exactly twelve unique questions")

    inference_rows: list[dict[str, Any]] = []
    scoring_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, 1):
        key = f"grailqa::{row['case_id']}"
        answer_count = len(set(row.get("gold_answers", [])))
        cardinality = "MULTI" if answer_count > 1 else "SINGLE"
        source_case = raw.get(row["case_id"])
        if not source_case or not source_case.get("topic_entities"):
            raise RuntimeError(f"selected question lacks inference-safe topic entities: {key}")
        inference_rows.append({
            "case_id": f"iter26_attempt3_case_{index:02d}",
            "question_key": key,
            "dataset": "grailqa",
            "source_split": "dev",
            "question": row["question"],
            "topic_entities": source_case["topic_entities"],
            "topic_entity_names": source_case["topic_entity_names"],
        })
        scoring_rows.append({
            "case_id": f"iter26_attempt3_case_{index:02d}",
            "question_key": key,
            "gold_answer_ids_scoring_only": row.get("answer_ids", []),
            "gold_answers_scoring_only": row.get("gold_answers", []),
            "answer_cardinality_stratum_scoring_only": cardinality,
            "branch_family_scoring_only": row.get("gold_structure"),
        })
        audit_rows.append({
            "case_id": f"iter26_attempt3_case_{index:02d}",
            "question_key": key,
            "dataset": "grailqa",
            "source_split": "dev",
            "question": row["question"],
            "answer_cardinality_stratum_scoring_only": cardinality,
            "branch_family_scoring_only": row.get("gold_structure"),
            "gold_answer_count_scoring_only": answer_count,
            "stable_id_prior_overlap": False,
            "normalized_exact_text_overlap_with_prior": False,
            "lexical_near_duplicate_overlap_with_prior": False,
            "prior_archive_candidate_snapshot_exists": False,
            "selection_rank_sha256": _rank_key(cardinality, str(row.get("gold_structure", "")), row["case_id"]),
        })

    PREP.mkdir(parents=True, exist_ok=True)
    _write_jsonl(PREP / "INFERENCE_COHORT_NON_GOLD.jsonl", inference_rows)
    _write_jsonl(PREP / "SCORING_GOLD_ONLY.jsonl", scoring_rows)
    with (PREP / "INDEPENDENT_COHORT_AUDIT.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    source_files = {
        str(path.relative_to(ROOT)): cohort.sha256_file(path)
        for paths in cohort.POOL_PATHS.values()
        for path in paths
        if path.is_file()
    }
    cardinality_counts = Counter(row["answer_cardinality_stratum_scoring_only"] for row in audit_rows)
    branch_counts = Counter(row["branch_family_scoring_only"] for row in audit_rows)
    cell_counts = Counter(
        (row["answer_cardinality_stratum_scoring_only"], row["branch_family_scoring_only"])
        for row in audit_rows
    )
    if any(cell_counts[(cardinality, family)] != TARGET_PER_CELL
           for cardinality in CARDINALITIES for family in BRANCH_FAMILIES):
        raise RuntimeError(f"stratum allocation is not balanced: {dict(cell_counts)}")

    parent_hash = cohort.sha256_file(ATTEMPT2_COHORT)
    attempt2_manifest_hash = cohort.sha256_file(ATTEMPT2_MANIFEST)
    fresh50_manifest = ROOT / "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256/manifest_frozen.json"
    manifest = {
        "iteration": 26,
        "attempt_number": 3,
        "status": "FROZEN_OFFLINE_COHORT_NO_TRANSPORT",
        "manifest_kind": "FRESH_QUESTION_COHORT_AND_UPSTREAM_FORK_PLAN; CANDIDATE_IDS_REQUIRE_NEW_AUTHORIZED_UPSTREAM_RUN",
        "evidence_class": "OFFLINE_DIAGNOSTIC; NO_NEW_INDEPENDENT_EVALUATION",
        "evaluation_scope": "Twelve previously unused GrailQA dev strong-branch questions for feasibility only; balanced by gold cardinality and branch family; not a broad benchmark.",
        "selection_seed": SEED,
        "sample_size": 12,
        "allocation": {f"{cardinality}|{family}": TARGET_PER_CELL for cardinality in CARDINALITIES for family in BRANCH_FAMILIES},
        "source_dataset": "GrailQA",
        "source_split_policy": {"grailqa": "dev"},
        "selection_policy": "Within each of four predeclared GrailQA dev strata (single/multi gold cardinality x final/intermediate branch family), select the first three clean records in SHA-256 rank order from the eligible source pool. Exclude all historical source IDs, Fresh50 selected IDs, Iterations 14-25 IDs, and all 12 frozen Iteration 26 attempt-2 IDs. Exclude normalized exact and high-threshold lexical near-duplicate question texts against resolvable prior records and within the new cohort. Gold is isolated to the scoring sidecar and is used only to form prespecified strata.",
        "candidate_pool_source": "Fresh50 post-cap STRONGLY_BRANCH_REQUIRED source population, restricted to GrailQA dev; the source pool's existing exclusions are retained and all prior-used identities plus the complete attempt-2 cohort are removed.",
        "source_manifest_sha256": cohort.sha256_file(ROOT / "outputs/experiments/phase3a_1/safety_cap_sensitivity/frozen_future_eval_manifest.json"),
        "strong_benchmark_sha256": cohort.sha256_file(ROOT / "outputs/experiments/phase3a/branch_necessity_benchmark/strong_branch_benchmark.jsonl"),
        "frozen_fresh50_manifest_sha256": cohort.sha256_file(fresh50_manifest),
        "source_dataset_files_sha256": source_files,
        "attempt2_cohort_manifest_sha256": parent_hash,
        "attempt2_live_manifest_sha256": attempt2_manifest_hash,
        "attempt2_question_identities_excluded": sorted(attempt2_ids),
        "attempt2_question_count_excluded": len(attempt2_ids),
        "attempt2_all_selected_ids_excluded_regardless_of_run_status": True,
        "source_eligible_count_after_identity_exclusions": len([
            row for row in fresh_pool
            if row.get("dataset") == "grailqa" and row.get("split") == "dev" and row["case_id"] not in known_ids
        ]),
        "clean_candidate_counts_by_cardinality_and_branch_family": {
            f"{cardinality}|{family}": len(candidates[(cardinality, family)])
            for cardinality in CARDINALITIES for family in BRANCH_FAMILIES
        },
        "selected_cardinality_counts": dict(cardinality_counts),
        "selected_branch_family_counts": dict(branch_counts),
        "selected_cell_counts": {f"{cardinality}|{family}": cell_counts[(cardinality, family)] for cardinality in CARDINALITIES for family in BRANCH_FAMILIES},
        "prior_question_id_count_from_iterations_14_to_25": len(prior14_25_ids),
        "fresh50_selected_id_count": len(fresh50_ids),
        "prior_source_id_count": len(prior_ids),
        "resolvable_prior_text_count": len(prior_texts),
        "stable_id_overlaps": 0,
        "normalized_exact_prior_text_overlaps": 0,
        "lexical_near_duplicate_prior_text_overlaps": 0,
        "excluded_candidates_by_reason": dict(exclusions),
        "candidate_snapshots_for_selected_questions": 0,
        "new_authorized_upstream_generation_required": True,
        "new_paid_api_calls": 0,
        "new_sparql_attempts": 0,
        "new_spend_usd": 0.0,
        "gold_not_in_inference_file": True,
        "inference_file": "INFERENCE_COHORT_NON_GOLD.jsonl",
        "gold_file": "SCORING_GOLD_ONLY.jsonl",
        "case_keys": [row["question_key"] for row in inference_rows],
    }
    manifest_path = PREP / "COHORT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_summary = {
        "status": "PASS_FRESH_BALANCED_COHORT; TEXT_SCREEN_IS_LEXICAL_NOT_SEMANTIC",
        "sample_size": len(selected),
        "source_eligible_count_after_identity_exclusions": manifest["source_eligible_count_after_identity_exclusions"],
        "clean_candidate_counts_by_cardinality_and_branch_family": manifest["clean_candidate_counts_by_cardinality_and_branch_family"],
        "selected_cardinality_counts": dict(cardinality_counts),
        "selected_branch_family_counts": dict(branch_counts),
        "selected_cell_counts": manifest["selected_cell_counts"],
        "selected_all_have_gold_sidecar": all(bool(row.get("gold_answer_ids_scoring_only")) for row in scoring_rows),
        "stable_id_overlaps": 0,
        "normalized_exact_prior_text_overlaps": 0,
        "lexical_near_duplicate_prior_text_overlaps": 0,
        "excluded_candidates_by_reason": dict(exclusions),
        "prior_text_equivalence_limit": "Exact normalized and high-threshold lexical near-duplicate screen only; semantic paraphrase identity cannot be proven by this audit.",
        "candidate_snapshot_limit": "No archived final candidate snapshots exist for these selected IDs. A new authorized run must generate one shared upstream trajectory per question and fork only the final answer selector.",
    }
    (PREP / "COHORT_AUDIT_SUMMARY.json").write_text(json.dumps(audit_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme = (
        "# Iteration 26 attempt 3 cohort preparation\n\n"
        "Offline only. The entire attempt-2 cohort is treated as development material, including its unstarted questions. "
        "This new feasibility cohort uses twelve previously unused GrailQA dev questions: three in each of the four "
        "predeclared cells formed by single/multi gold cardinality and final/intermediate branch family. All stable IDs, "
        "normalized exact questions, and high-threshold lexical near duplicates screened against the available history. "
        "Gold appears only in the scoring sidecar. Candidate snapshots do not exist, so any live run must generate a new "
        "shared upstream trajectory per question. This is not an independent scientific result until separately authorized "
        "and run.\n\n"
        f"Cohort manifest SHA-256: `{cohort.sha256_file(manifest_path)}`\n\n"
        "No API or SPARQL calls were made during preparation.\n"
    )
    (PREP / "COHORT_PREPARATION.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "preparation_dir": str(PREP.relative_to(ROOT)),
        "sample_size": len(selected),
        "case_keys": manifest["case_keys"],
        "selected_cell_counts": manifest["selected_cell_counts"],
        "clean_candidate_counts": manifest["clean_candidate_counts_by_cardinality_and_branch_family"],
        "text_exclusions": dict(exclusions),
        "manifest_sha256": cohort.sha256_file(manifest_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
