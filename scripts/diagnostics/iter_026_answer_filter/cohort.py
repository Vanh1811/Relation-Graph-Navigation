"""Freeze an offline, untouched 12-question feasibility cohort.

The pool is the verified unused remainder of the Fresh50 strong-branch source
population. Selection is outcome-blind and stratified by dataset and gold
answer cardinality; gold fields are written only to a separate scorer file.
No inference or transport occurs here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
OUT = WORK / "RESULTS/iter_026_answer_selector_readiness"
SEED = "iter26-independent-answer-selector-heldout-splits-v2-2026-11-08"
# Use benchmark evaluation splits only. Four previously unused CWQ test
# questions remain; GrailQA dev supplies the other eight while preserving
# both answer-cardinality strata. The resulting 4:8 dataset mix is explicit.
STRATA_ALLOCATION = {
    ("cwq", "SINGLE"): 4,
    ("cwq", "MULTI"): 0,
    ("grailqa", "SINGLE"): 4,
    ("grailqa", "MULTI"): 4,
}
SPLIT_POLICY = {"cwq": "test", "grailqa": "dev"}

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from run_fresh50_strong_branch_three_way import load_fresh_pool  # noqa: E402
from run_phase3c_frozen_generalization import POOL_PATHS, raw_cases  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _id_from_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(?:cwq::)?(WebQ(?:Trn|Test)-[A-Za-z0-9_]+)", value)
    if match:
        return match.group(1)
    match = re.fullmatch(r"grailqa::([0-9]{10,13})", value)
    if match:
        return match.group(1)
    if re.fullmatch(r"[0-9]{10,13}", value):
        return value
    return None


def _collect_ids_in_object(value: Any, output: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_lower = str(key).casefold()
            if key_lower in {"question_key", "question_id", "qid", "case_id", "example_id"}:
                found = _id_from_key(child)
                if found:
                    output.add(found)
            _collect_ids_in_object(child, output)
    elif isinstance(value, list):
        for child in value:
            _collect_ids_in_object(child, output)


def _iter_records(path: Path):
    try:
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8", errors="ignore") as stream:
                for line in stream:
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        elif path.suffix == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except (json.JSONDecodeError, OSError):
                return
            if isinstance(value, list):
                yield from value
            else:
                yield value
        elif path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8", errors="ignore") as stream:
                yield from csv.DictReader(stream)
    except OSError:
        return


def _continuous_agent_ids() -> set[str]:
    """Explicitly include Iterations 14–25 even if their IDs also occur earlier."""
    found: set[str] = set()
    result_root = WORK / "RESULTS"
    for iteration_dir in sorted(result_root.glob("iter_*")):
        try:
            number = int(iteration_dir.name.split("_", 2)[1])
        except (IndexError, ValueError):
            continue
        if not 14 <= number <= 25:
            continue
        for path in iteration_dir.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".csv"}:
                for record in _iter_records(path):
                    _collect_ids_in_object(record, found)
    return found


def _fresh50_ids() -> set[str]:
    path = ROOT / "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256/manifest_frozen.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["question_id"]) for row in data.get("cases", [])}


def _lexical_duplicate(left: str, right: str) -> tuple[bool, float, float]:
    left_norm, right_norm = normalize_question(left), normalize_question(right)
    if not left_norm or not right_norm:
        return False, 0.0, 0.0
    if left_norm == right_norm:
        return True, 1.0, 1.0
    left_tokens, right_tokens = set(left_norm.split()), set(right_norm.split())
    jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens | right_tokens else 0.0
    # SequenceMatcher is comparatively expensive. A candidate below the
    # required token-overlap threshold cannot be a near duplicate, so skip it.
    if jaccard < 0.85:
        return False, jaccard, 0.0
    sequence = SequenceMatcher(None, left_norm, right_norm, autojunk=False).ratio()
    return jaccard >= 0.85 and sequence >= 0.88, jaccard, sequence


def _rank_key(dataset: str, cardinality: str, case_id: str, branch: str) -> str:
    value = f"{SEED}|{dataset}|{cardinality}|{branch}|{case_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def freeze_cohort() -> dict[str, Any]:
    fresh_pool, source_manifest, prior_ids, prior_meta = load_fresh_pool()
    if len(fresh_pool) != 1398:
        raise RuntimeError(f"Fresh50 source-pool size changed: expected 1398, got {len(fresh_pool)}")
    known_previous_ids = set(prior_ids) | _fresh50_ids() | _continuous_agent_ids()
    pre_iter26_pool_count = len(fresh_pool)
    source_pool_id_overlap_count = sum(row["case_id"] in known_previous_ids for row in fresh_pool)
    fresh_pool = [row for row in fresh_pool if row["case_id"] not in known_previous_ids]
    if source_pool_id_overlap_count:
        print(f"excluded {source_pool_id_overlap_count} previously used stable IDs from the 1398-case source remainder", flush=True)

    raw = raw_cases()
    source_data_sha256 = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for paths in POOL_PATHS.values() for path in paths if path.is_file()
    }
    prior_texts = []
    for case_id in sorted(known_previous_ids):
        record = raw.get(case_id)
        if record and record.get("question"):
            prior_texts.append((case_id, record["question"]))
    prior_text_norms = {normalize_question(question): case_id for case_id, question in prior_texts}

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fresh_pool:
        if row.get("split") != SPLIT_POLICY.get(row["dataset"]):
            continue
        gold_count = len(set(row.get("gold_answers", [])))
        if gold_count < 1:
            continue
        cardinality = "MULTI" if gold_count > 1 else "SINGLE"
        groups[(row["dataset"], cardinality, row["gold_structure"])].append(row)
    for values in groups.values():
        values.sort(key=lambda row: _rank_key(row["dataset"], "MULTI" if len(set(row.get("gold_answers", []))) > 1 else "SINGLE", row["case_id"], row["gold_structure"]))

    all_ranked = {}
    for dataset in ("cwq", "grailqa"):
        for cardinality in ("SINGLE", "MULTI"):
            eligible = [row for key, rows in groups.items() if key[:2] == (dataset, cardinality) for row in rows]
            all_ranked[(dataset, cardinality)] = sorted(
                eligible,
                key=lambda row: _rank_key(dataset, cardinality, row["case_id"], row["gold_structure"]),
            )

    selected: list[dict[str, Any]] = []
    overlap_notes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (dataset, cardinality), required_count in STRATA_ALLOCATION.items():
        ranked = all_ranked[(dataset, cardinality)]
        group_selected: list[dict[str, Any]] = []
        if required_count == 0:
            continue
        for candidate in ranked:
            question = candidate["question"]
            normalized = normalize_question(question)
            if normalized in prior_text_norms:
                overlap_notes["normalized_exact"].append({
                    "candidate_id": candidate["case_id"],
                    "matched_prior_id": prior_text_norms[normalized],
                })
                continue
            near_match = None
            for prior_id, prior_question in prior_texts:
                duplicated, jaccard, sequence = _lexical_duplicate(question, prior_question)
                if duplicated:
                    near_match = {"matched_prior_id": prior_id, "token_jaccard": jaccard, "sequence_ratio": sequence}
                    break
            if near_match:
                overlap_notes["lexical_near_duplicate_excluded"].append({"candidate_id": candidate["case_id"], **near_match})
                continue
            if any(normalize_question(question) == normalize_question(other["question"]) for other in selected + group_selected):
                overlap_notes["within_cohort_duplicate_excluded"].append({"candidate_id": candidate["case_id"]})
                continue
            group_selected.append(candidate)
            if len(group_selected) == required_count:
                break
        if len(group_selected) != required_count:
            raise RuntimeError(
                f"insufficient clean cases in {dataset}/{cardinality}: "
                f"needed {required_count}, found {len(group_selected)}"
            )
        selected.extend(group_selected)

    selected.sort(key=lambda row: (row["dataset"], "MULTI" if len(set(row.get("gold_answers", []))) > 1 else "SINGLE", row["case_id"]))
    selected_ids = {row["case_id"] for row in selected}
    if len(selected) != 12 or len(selected_ids) != 12:
        raise RuntimeError("the frozen feasibility cohort must have exactly 12 unique identities")
    for row in selected:
        question = row["question"]
        if any(_lexical_duplicate(question, other["question"])[0] for other in selected if other["case_id"] != row["case_id"]):
            raise RuntimeError("selected cohort contains lexical duplicates")

    OUT.mkdir(parents=True, exist_ok=True)
    inference_rows = []
    scoring_rows = []
    audit_rows = []
    for index, row in enumerate(selected, 1):
        key = f"{row['dataset']}::{row['case_id']}"
        cardinality = "MULTI" if len(set(row.get("gold_answers", []))) > 1 else "SINGLE"
        source_case = raw.get(row["case_id"])
        if not source_case or not source_case.get("topic_entities"):
            raise RuntimeError(f"selected case lacks inference-safe topic entity input: {key}")
        inference_rows.append({
            "case_id": f"iter26_case_{index:02d}",
            "question_key": key,
            "dataset": row["dataset"],
            "source_split": row.get("split", ""),
            "question": row["question"],
            "topic_entities": source_case["topic_entities"],
            "topic_entity_names": source_case["topic_entity_names"],
        })
        scoring_rows.append({
            "case_id": f"iter26_case_{index:02d}",
            "question_key": key,
            "gold_answer_ids_scoring_only": row.get("answer_ids", []),
            "gold_answers_scoring_only": row.get("gold_answers", []),
            "answer_cardinality_stratum_scoring_only": cardinality,
            "branch_family_scoring_only": row.get("gold_structure"),
        })
        audit_rows.append({
            "case_id": f"iter26_case_{index:02d}",
            "question_key": key,
            "dataset": row["dataset"],
            "source_split": row.get("split", ""),
            "question": row["question"],
            "answer_cardinality_stratum_scoring_only": cardinality,
            "branch_family_scoring_only": row.get("gold_structure"),
            "gold_answer_count_scoring_only": len(set(row.get("gold_answers", []))),
            "stable_id_prior_overlap": False,
            "normalized_exact_text_overlap_with_prior": False,
            "lexical_near_duplicate_overlap_with_prior": False,
            "prior_archive_candidate_snapshot_exists": False,
            "selection_rank_sha256": _rank_key(row["dataset"], cardinality, row["case_id"], row["gold_structure"]),
        })

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    write_jsonl(OUT / "INFERENCE_COHORT_NON_GOLD.jsonl", inference_rows)
    write_jsonl(OUT / "SCORING_GOLD_ONLY.jsonl", scoring_rows)
    with (OUT / "INDEPENDENT_COHORT_AUDIT.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    fresh50_manifest = ROOT / "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256/manifest_frozen.json"
    fresh50_data = json.loads(fresh50_manifest.read_text(encoding="utf-8"))
    candidate_manifest = {
        "iteration": 26,
        "status": "FROZEN_OFFLINE_COHORT_NO_TRANSPORT",
        "manifest_kind": "QUESTION_COHORT_AND_UPSTREAM_FORK_PLAN; CANDIDATE_IDS_AWAIT_NEWLY_AUTHORIZED_UPSTREAM_RUN",
        "evaluation_scope": "12-question feasibility slice sampled only from CWQ test and GrailQA dev among unused strong-branch records; not a representative broad KGQA benchmark",
        "selection_seed": SEED,
        "sample_size": len(inference_rows),
        "allocation": {f"{k[0]}|{k[1]}": required for k, required in STRATA_ALLOCATION.items()},
        "source_split_policy": SPLIT_POLICY,
        "dataset_allocation_implication": "CWQ contributes 4 test questions and GrailQA contributes 8 dev questions because only four previously unused CWQ test identities remain after project-history exclusion.",
        "unavailable_strata": [f"{dataset}|{cardinality}" for (dataset, cardinality), count in STRATA_ALLOCATION.items() if count == 0],
        "selection_policy": "Hash-ranked sample among unused CWQ test and GrailQA dev cases only; dataset split chosen before inference, then stratified by dataset and answer cardinality wherever eligible cases exist. The CWQ test multi-answer stratum had zero eligible cases after prior-question exclusion; gold remains in the scoring sidecar; exact and lexical near-duplicate questions excluded before inference.",
        "candidate_pool_source": "Fresh50 post-cap STRONGLY_BRANCH_REQUIRED source population, after its original exclusions and a second exclusion of Fresh50-selected and Iteration 14–25 identities.",
        "source_manifest_sha256": sha256_file(ROOT / "outputs/experiments/phase3a_1/safety_cap_sensitivity/frozen_future_eval_manifest.json"),
        "strong_benchmark_sha256": sha256_file(ROOT / "outputs/experiments/phase3a/branch_necessity_benchmark/strong_branch_benchmark.jsonl"),
        "source_dataset_files_sha256": source_data_sha256,
        "original_fresh50_prior_id_count": len(prior_ids),
        "fresh50_source_remainder_count_before_iter26_exclusions": pre_iter26_pool_count,
        "overlap_ids_removed_from_source_remainder": source_pool_id_overlap_count,
        "eligible_source_count_after_all_known_identity_exclusions": len(fresh_pool),
        "heldout_split_eligible_source_count": sum(len(values) for values in all_ranked.values()),
        "source_split_policy": SPLIT_POLICY,
        "eligible_source_counts_by_dataset_and_cardinality": {
            f"{dataset}|{cardinality}": len(all_ranked[(dataset, cardinality)])
            for dataset, cardinality in STRATA_ALLOCATION
        },
        "prior_question_ids_from_iter14_to_iter25": len(_continuous_agent_ids()),
        "identity_audit_source_scope": "Fresh50's frozen prior-ID exclusions, its 50 selected IDs, and all stable question IDs parsed from Iterations 14–25 JSON/JSONL/CSV result artifacts; exact and high-threshold lexical comparison against resolvable prior question text.",
        "fresh50_selected_id_count": len(fresh50_data.get("cases", [])),
        "frozen_fresh50_manifest_sha256": sha256_file(fresh50_manifest),
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
    (OUT / "COHORT_MANIFEST.json").write_text(json.dumps(candidate_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    audit_summary = {
        "status": "PASS_WITH_SCOPE_LIMITATION",
        "fresh50_source_remainder_count_before_iter26_exclusions": pre_iter26_pool_count,
        "overlap_ids_removed_from_source_remainder": source_pool_id_overlap_count,
        "eligible_source_pool_count_after_all_known_identity_exclusions": len(fresh_pool),
        "prior_id_count": len(prior_ids),
        "fresh50_selected_count": len(fresh50_data.get("cases", [])),
        "continuous_agent_iter14_25_id_count": len(_continuous_agent_ids()),
        "selected_count": len(selected),
        "stable_id_overlaps": 0,
        "normalized_exact_text_overlaps": len(overlap_notes["normalized_exact"]),
        "normalized_exact_text_exclusions": overlap_notes["normalized_exact"],
        "lexical_near_duplicate_exclusions": overlap_notes["lexical_near_duplicate_excluded"],
        "within_cohort_exact_text_exclusions": overlap_notes["within_cohort_duplicate_excluded"],
        "selected_all_have_gold_sidecar": all(bool(row.get("answer_ids")) for row in selected),
        "historical_candidate_snapshot_count_for_selected_ids": 0,
        "answer_cardinality_allocation": dict(Counter(x["answer_cardinality_stratum_scoring_only"] for x in audit_rows)),
        "dataset_allocation": dict(Counter(x["dataset"] for x in audit_rows)),
        "selected_allocation_by_dataset_and_cardinality": {
            f"{dataset}|{cardinality}": sum(
                1 for row in audit_rows
                if row["dataset"] == dataset and row["answer_cardinality_stratum_scoring_only"] == cardinality
            )
            for dataset, cardinality in STRATA_ALLOCATION
        },
        "branch_family_allocation_scoring_only": dict(Counter(x["branch_family_scoring_only"] for x in audit_rows)),
        "prior_text_resolvable_id_count": len(prior_texts),
        "prior_text_equivalence_limit": "Exact normalized and high-threshold lexical near-duplicate screen only; it cannot prove semantic paraphrase identity.",
        "candidate_pool_scope_limit": "All sampled questions are from the strong-branch subset and benchmark evaluation splits; results will be conditional on this feasibility cohort, not broad KGQA.",
        "split_scope_limit": "Only CWQ test and GrailQA dev were eligible. Four unused CWQ test cases remain, so the sample is intentionally 4 CWQ / 8 GrailQA rather than dataset-balanced. Training split cases were excluded.",
    }
    (OUT / "COHORT_AUDIT_SUMMARY.json").write_text(json.dumps(audit_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": candidate_manifest, "audit": audit_summary, "prior_sources": prior_meta}


if __name__ == "__main__":
    result = freeze_cohort()
    print(json.dumps({"sample_size": result["manifest"]["sample_size"], "allocation": result["manifest"]["allocation"], "audit": result["audit"]}, indent=2, ensure_ascii=False))
