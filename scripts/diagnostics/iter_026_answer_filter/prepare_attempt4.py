"""Prepare an offline, disjoint I26 attempt-4 GrailQA dev cohort."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.iter_026_answer_filter import cohort


ROOT = cohort.ROOT
RESULTS = cohort.OUT
PREP = RESULTS / "ATTEMPT4_PREPARATION"
SEED = "iter26-attempt4-grailqa-dev-cap-aware-2026-11-08"
ALLOCATION = {
    ("SINGLE", "BRANCH_FINAL"): 6,
    ("SINGLE", "BRANCH_INTERMEDIATE"): 1,
    ("MULTI", "BRANCH_FINAL"): 5,
    ("MULTI", "BRANCH_INTERMEDIATE"): 0,
}


def _rank(cardinality: str, family: str, case_id: str) -> str:
    value = f"{SEED}|grailqa|dev|{cardinality}|{family}|{case_id}"
    return hashlib.sha256(value.encode()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _near_duplicate_precomputed(
    normalized: str,
    tokens: set[str],
    previous: list[tuple[str, set[str]]],
) -> bool:
    for previous_normalized, previous_tokens in previous:
        union_size = len(tokens | previous_tokens)
        if not union_size:
            continue
        jaccard = len(tokens & previous_tokens) / union_size
        if jaccard >= 0.85 and SequenceMatcher(
            None, normalized, previous_normalized, autojunk=False,
        ).ratio() >= 0.88:
            return True
    return False


def main() -> None:
    fresh_pool, _, prior_ids, _ = cohort.load_fresh_pool()
    raw = cohort.raw_cases()
    known_ids = set(prior_ids) | cohort._fresh50_ids() | cohort._continuous_agent_ids()
    prior_cohorts = (
        RESULTS / "COHORT_MANIFEST.json",
        RESULTS / "ATTEMPT3_PREPARATION/COHORT_MANIFEST.json",
    )
    for path in prior_cohorts:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        known_ids.update(
            case_id for case_id in (cohort._id_from_key(key) for key in manifest.get("case_keys", []))
            if case_id
        )
    attempt3_ids = {
        case_id for case_id in (
            cohort._id_from_key(key)
            for key in json.loads((RESULTS / "ATTEMPT3_PREPARATION/COHORT_MANIFEST.json").read_text(encoding="utf-8")).get("case_keys", [])
        ) if case_id
    }
    if len(attempt3_ids) != 12 or not attempt3_ids.issubset(known_ids):
        raise RuntimeError("all twelve attempt-3 identities must be excluded")

    prior_texts = [
        (case_id, raw[case_id]["question"])
        for case_id in sorted(known_ids)
        if case_id in raw and raw[case_id].get("question")
    ]
    prior_norms = {cohort.normalize_question(text) for _, text in prior_texts}
    prior_precomputed = [
        (normalized, set(normalized.split()))
        for _, text in prior_texts
        if (normalized := cohort.normalize_question(text))
    ]
    pool: dict[tuple[str, str], list[dict[str, Any]]] = {key: [] for key in ALLOCATION}
    for row in fresh_pool:
        if row.get("dataset") != "grailqa" or row.get("split") != "dev":
            continue
        if row["case_id"] in known_ids or not row.get("question"):
            continue
        answer_count = len(set(row.get("gold_answers", [])))
        if answer_count < 1:
            continue
        cardinality = "MULTI" if answer_count > 1 else "SINGLE"
        family = str(row.get("gold_structure", ""))
        if (cardinality, family) in pool:
            pool[(cardinality, family)].append(row)
    for values in pool.values():
        values.sort(key=lambda row: _rank(
            "MULTI" if len(set(row.get("gold_answers", []))) > 1 else "SINGLE",
            str(row.get("gold_structure", "")), row["case_id"],
        ))

    excluded = Counter()
    selected: list[dict[str, Any]] = []
    clean_counts: Counter = Counter()
    for cell, required in ALLOCATION.items():
        clean_rows: list[dict[str, Any]] = []
        for row in pool[cell]:
            question = row["question"]
            normalized = cohort.normalize_question(question)
            if normalized in prior_norms:
                excluded["normalized_exact_prior"] += 1
                continue
            tokens = set(normalized.split())
            if _near_duplicate_precomputed(normalized, tokens, prior_precomputed):
                excluded["lexical_near_duplicate_prior"] += 1
                continue
            if any(cohort._lexical_duplicate(question, other["question"])[0] for other in selected + clean_rows):
                excluded["lexical_duplicate_within_cohort"] += 1
                continue
            clean_rows.append(row)
        clean_counts[cell] = len(clean_rows)
        accepted = clean_rows[:required]
        if len(accepted) != required:
            raise RuntimeError(f"insufficient clean candidates for {cell}: need {required}, got {len(accepted)}")
        selected.extend(accepted)

    selected.sort(key=lambda row: (
        "MULTI" if len(set(row.get("gold_answers", []))) > 1 else "SINGLE",
        str(row.get("gold_structure", "")), row["case_id"],
    ))
    if len(selected) != 12 or len({row["case_id"] for row in selected}) != 12:
        raise RuntimeError("attempt-4 cohort must have twelve unique questions")
    if {row["case_id"] for row in selected} & known_ids:
        raise RuntimeError("attempt-4 cohort overlaps a previously used identity")

    inference: list[dict[str, Any]] = []
    scoring: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    cell_counts = Counter()
    for index, row in enumerate(selected, 1):
        case_id = f"iter26_attempt4_case_{index:02d}"
        key = f"grailqa::{row['case_id']}"
        count = len(set(row.get("gold_answers", [])))
        cardinality = "MULTI" if count > 1 else "SINGLE"
        family = str(row.get("gold_structure", ""))
        cell_counts[(cardinality, family)] += 1
        source = raw.get(row["case_id"])
        if not source or not source.get("topic_entities"):
            raise RuntimeError(f"missing inference-safe topic entity data for {key}")
        inference.append({
            "case_id": case_id, "question_key": key, "dataset": "grailqa",
            "source_split": "dev", "question": row["question"],
            "topic_entities": source["topic_entities"],
            "topic_entity_names": source["topic_entity_names"],
        })
        scoring.append({
            "case_id": case_id, "question_key": key,
            "gold_answer_ids_scoring_only": row.get("answer_ids", []),
            "gold_answers_scoring_only": row.get("gold_answers", []),
            "answer_cardinality_stratum_scoring_only": cardinality,
            "branch_family_scoring_only": family,
        })
        audit.append({
            "case_id": case_id, "question_key": key, "question": row["question"],
            "answer_cardinality_stratum_scoring_only": cardinality,
            "branch_family_scoring_only": family,
            "gold_answer_count_scoring_only": count,
            "attempt3_identity_excluded": row["case_id"] not in attempt3_ids,
            "stable_id_prior_overlap": False,
            "normalized_exact_prior_text_overlap": False,
            "lexical_near_duplicate_prior_text_overlap": False,
            "selection_rank_sha256": _rank(cardinality, family, row["case_id"]),
        })

    PREP.mkdir(parents=True, exist_ok=True)
    _write_jsonl(PREP / "INFERENCE_COHORT_NON_GOLD.jsonl", inference)
    _write_jsonl(PREP / "SCORING_GOLD_ONLY.jsonl", scoring)
    with (PREP / "INDEPENDENT_COHORT_AUDIT.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)

    source_hashes = {
        str(path.relative_to(ROOT)): cohort.sha256_file(path)
        for paths in cohort.POOL_PATHS.values() for path in paths if path.is_file()
    }
    allocation = {f"{card}|{family}": required for (card, family), required in ALLOCATION.items()}
    selected_allocation = {f"{card}|{family}": cell_counts[(card, family)] for card, family in ALLOCATION}
    manifest = {
        "iteration": 26, "attempt_number": 4,
        "status": "FROZEN_OFFLINE_COHORT_NO_TRANSPORT",
        "evidence_class": "OFFLINE_DIAGNOSTIC; NO_NEW_INDEPENDENT_EVALUATION",
        "selection_seed": SEED, "sample_size": 12,
        "allocation": allocation, "selected_allocation": selected_allocation,
        "source_dataset": "GrailQA", "source_split": "dev",
        "selection_policy": "Hash-rank selection from the previously unused GrailQA dev source pool, excluding all prior identities including all twelve attempt-3 IDs; retain a 6/6 single/multi answer split. Remaining source counts do not permit multi-answer intermediate-branch cases, so include the only clean single-answer intermediate case and allocate the other eleven records to final-branch cells. Exact normalized and high-threshold lexical near-duplicate text screens are applied against resolvable history and within cohort. Gold is isolated in the scoring sidecar.",
        "source_eligible_counts_after_identity_exclusion": {f"{card}|{family}": len(values) for (card, family), values in pool.items()},
        "clean_candidate_counts_by_stratum": {f"{card}|{family}": clean_counts[(card, family)] for card, family in ALLOCATION},
        "selected_cell_counts": selected_allocation,
        "attempt3_question_ids_excluded": sorted(attempt3_ids),
        "attempt3_question_count_excluded": 12,
        "all_historical_question_ids_excluded": len(known_ids),
        "prior_text_rows_resolvable": len(prior_texts),
        "excluded_candidates_by_reason": dict(excluded),
        "stable_id_overlaps": 0, "normalized_exact_prior_text_overlaps": 0,
        "lexical_near_duplicate_prior_text_overlaps": 0,
        "source_dataset_files_sha256": source_hashes,
        "attempt3_cohort_manifest_sha256": cohort.sha256_file(RESULTS / "ATTEMPT3_PREPARATION/COHORT_MANIFEST.json"),
        "attempt3_manifest_sha256": cohort.sha256_file(RESULTS / "LIVE_MANIFEST_ATTEMPT3.json"),
        "gold_not_in_inference_file": True,
        "inference_file": "INFERENCE_COHORT_NON_GOLD.jsonl",
        "gold_file": "SCORING_GOLD_ONLY.jsonl",
        "case_keys": [row["question_key"] for row in inference],
        "new_paid_api_calls": 0, "new_sparql_attempts": 0, "new_spend_usd": 0.0,
    }
    (PREP / "COHORT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "status": "PASS_DISJOINT_COHORT; IMBALANCED_BRANCH_FAMILY_DUE_TO_SOURCE_AVAILABILITY",
        "sample_size": 12, "selected_cell_counts": selected_allocation,
        "source_eligible_counts_after_identity_exclusion": manifest["source_eligible_counts_after_identity_exclusion"],
        "clean_candidate_counts_by_stratum": manifest["clean_candidate_counts_by_stratum"],
        "attempt3_identity_overlap": 0,
        "exact_and_lexical_text_overlap": 0,
        "semantic_paraphrase_identity_proven": False,
        "gold_isolated_from_inference": True,
    }
    (PREP / "COHORT_AUDIT_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (PREP / "COHORT_PREPARATION.md").write_text(
        "# Iteration 26 attempt 4 cohort preparation\n\n"
        "Offline only; zero API and SPARQL requests. This cohort excludes all twelve attempt-3 identities and every resolvable historical identity. It contains six single-answer and six multi-answer GrailQA dev questions. The remaining source pool has no multi-answer intermediate-branch case and only one clean single-answer intermediate-branch case, so branch-family allocation is 11 final and 1 intermediate. Gold remains in a separate scoring file. Exact and high-threshold lexical duplicate checks do not establish semantic non-duplication.\n\n"
        f"Cohort manifest SHA-256: `{cohort.sha256_file(PREP / 'COHORT_MANIFEST.json')}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "preparation_dir": str(PREP.relative_to(ROOT)),
        "case_keys": manifest["case_keys"],
        "selected_cell_counts": selected_allocation,
        "clean_candidate_counts": manifest["clean_candidate_counts_by_stratum"],
        "manifest_sha256": cohort.sha256_file(PREP / "COHORT_MANIFEST.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
