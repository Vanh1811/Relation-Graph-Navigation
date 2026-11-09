"""Gold-only scorer for the completed Iteration 29 attempt-3 run.

This scorer is deliberately separate from the live runner. It first verifies
the frozen manifest, authorization, independent SCOREABLE gate, cohort,
source dataset, ordered paired rows, snapshots, and resource ledgers. It then
scores all 800 cases intention-to-treat and writes gold-only artifacts beside
the run. It performs no network or model calls.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_028_terminal_boundary_repair"
MANIFEST_PATH = BASE / "NEXT_MANIFEST_DRAFT_ATTEMPT3.json"
COHORT_PATH = BASE / "NEXT_INFERENCE_COHORT_ATTEMPT3_NON_GOLD.jsonl"
RUN = BASE / "RUNS/iter29_terminal_candidate_selector_attempt3"
DATASET_PATH = ROOT / "data/interim/grailqa_dev.jsonl"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_sets(gold: set[str], predicted: set[str]) -> tuple[int, float]:
    if not gold or not predicted:
        return 0, 0.0
    overlap = len(gold & predicted)
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return int(overlap > 0), f1


def paired_bootstrap_ci(deltas: list[int], *, seed: int = 2903, draws: int = 20_000) -> list[float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = [sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws)]
    means.sort()
    return [means[int(0.025 * draws)], means[min(draws - 1, int(0.975 * draws))]]


def exact_mcnemar_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial McNemar p-value, conditional on discordances."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"refusing to score: {message}")


def run_score() -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    metadata = read_json(RUN / "RUN_METADATA.json")
    authorization = read_json(RUN / "AUTHORIZATION_RECORD.json")
    gate = read_json(RUN / "SCORING_GATE_VERDICT.json")
    validation = read_json(RUN / "VALIDATION_INPUT_NON_GOLD.json")

    manifest_hash = manifest.get("manifest_sha256")
    expected_n = int(manifest.get("sample_size", 0))
    _require(manifest_hash and authorization.get("authorized") is True, "missing exact-hash authorization")
    _require(authorization.get("manifest_sha256") == manifest_hash, "authorization does not bind to this manifest")
    _require(metadata.get("manifest_sha256") == manifest_hash and gate.get("manifest_sha256") == manifest_hash,
             "run, manifest, and gate hashes differ")
    _require(gate.get("verdict") == "SCOREABLE" and gate.get("score_invocation_allowed") is True,
             "independent gate did not authorize scoring")
    _require(metadata.get("status") == "COMPLETE", "live run is incomplete")
    _require(metadata.get("cases_expected") == expected_n and metadata.get("cases_paired") == expected_n,
             "live run does not contain the complete frozen cohort")

    source_path = ROOT / manifest["source_dataset_path"]
    _require(sha256_file(source_path) == manifest["source_dataset_sha256"], "frozen source dataset hash mismatch")
    _require(sha256_file(COHORT_PATH) == manifest["inference_cohort_sha256"], "frozen inference cohort hash mismatch")

    cohort = read_jsonl(COHORT_PATH)
    results = read_jsonl(RUN / "PAIRED_CASES_NON_GOLD.jsonl")
    snapshots = read_jsonl(RUN / "FROZEN_CASE_SNAPSHOTS_NON_GOLD.jsonl")
    requests = read_jsonl(RUN / "REQUEST_EVENTS.jsonl")
    sparql = read_jsonl(RUN / "SPARQL_EVENTS.jsonl")
    _require(len(cohort) == len(results) == len(snapshots) == expected_n,
             "cohort, paired rows, or frozen snapshots have the wrong row count")
    frozen_keys = list(manifest.get("question_keys", []))
    _require(len(frozen_keys) == expected_n, "manifest question-key list has the wrong size")
    _require([row["question_key"] for row in cohort] == frozen_keys, "cohort order differs from manifest")
    _require([row["question_key"] for row in results] == frozen_keys, "paired result order differs from manifest")
    _require([row["question_key"] for row in snapshots] == frozen_keys, "snapshot order differs from manifest")

    dataset_rows = read_jsonl(source_path)
    gold_by_id: dict[str, dict[str, Any]] = {}
    for row in dataset_rows:
        if str(row.get("split", row.get("source_metadata", {}).get("source_split", ""))).casefold() != "dev":
            continue
        key = str(row["id"])
        _require(key not in gold_by_id, f"duplicate dev dataset ID {key}")
        gold_by_id[key] = row

    sparql_by_id: dict[str, dict[str, Any]] = {}
    for event in sparql:
        attempt_id = event.get("attempt_id")
        _require(attempt_id not in sparql_by_id, f"duplicate SPARQL attempt ID {attempt_id}")
        sparql_by_id[attempt_id] = event

    scored: list[dict[str, Any]] = []
    selector_statuses: Counter[str] = Counter()
    eligibility_reasons: Counter[str] = Counter()
    applied_valid = 0
    applied_membership_valid = 0
    applied_provenance_valid = 0
    baseline_correct_regressions = 0
    gold_candidate_coverage_count = 0
    gold_candidate_coverage_denominator = 0

    for infer, live, snapshot in zip(cohort, results, snapshots):
        case_id = infer["case_id"]
        question_key = infer["question_key"]
        _require(live.get("case_id") == case_id == snapshot.get("case_id"), f"case identity mismatch for {question_key}")
        _require(live.get("question") == infer.get("question"), f"question mismatch for {question_key}")
        _require(snapshot.get("question_key") == question_key, f"snapshot question identity mismatch for {question_key}")
        dataset_id = question_key.split("::", 1)[1]
        raw = gold_by_id.get(dataset_id)
        _require(raw is not None, f"gold source row missing for {question_key}")
        _require(raw.get("question") == infer.get("question"), f"gold text differs from frozen question for {question_key}")
        gold = set(map(str, raw.get("answer_ids") or []))
        _require(bool(gold), f"gold answer IDs are empty for {question_key}")

        baseline = set(map(str, live.get("baseline_prediction_ids") or []))
        intervention = set(map(str, live.get("intervention_prediction_ids") or []))
        _require(list(live.get("baseline_prediction_ids") or []) == list(snapshot.get("baseline_prediction_ids") or []),
                 f"baseline differs from frozen snapshot for {question_key}")
        _require(list(live.get("candidate_ids") or []) == list(snapshot.get("candidate_ids") or []),
                 f"candidate IDs differ from frozen snapshot for {question_key}")
        _require(live.get("candidate_snapshot_sha256") == snapshot.get("candidate_snapshot_sha256"),
                 f"candidate snapshot hash differs for {question_key}")

        selector = live.get("selector_result") if isinstance(live.get("selector_result"), dict) else {}
        status = str(selector.get("status", "NOT_CALLED"))
        selector_statuses[status] += 1
        eligibility_reasons[str(live.get("selector_eligibility_reason", "UNKNOWN"))] += 1
        candidate_ids = list(map(str, live.get("candidate_ids") or []))
        query_id = live.get("candidate_query_attempt_id")
        query_event = sparql_by_id.get(query_id)
        provenance_ok = bool(
            query_event
            and query_event.get("query_type") == "SELECT"
            and query_event.get("read_only") is True
            and query_event.get("status") == "ok"
            and query_event.get("query_sha256") == live.get("candidate_query_sha256")
        )

        membership_ok: bool | None = None
        if status == "VALID_SELECTION":
            selected = list(map(str, selector.get("selected_ids") or []))
            membership_ok = bool(selected) and len(selected) == len(set(selected)) and set(selected).issubset(set(candidate_ids))
            _require(membership_ok, f"selector output violates frozen candidate membership for {question_key}")
            _require(intervention == set(selected), f"intervention differs from selector output for {question_key}")
            applied_valid += 1
            applied_membership_valid += int(membership_ok)
            applied_provenance_valid += int(provenance_ok)
            _require(provenance_ok, f"selector candidate provenance invalid for {question_key}")
        else:
            _require(intervention == baseline, f"non-valid selector did not preserve baseline for {question_key}")

        base_hit, base_f1 = score_sets(gold, baseline)
        intervention_hit, intervention_f1 = score_sets(gold, intervention)
        baseline_correct_regressions += int(bool(base_hit and not intervention_hit))
        if candidate_ids:
            gold_candidate_coverage_denominator += 1
            gold_candidate_coverage_count += int(bool(gold & set(candidate_ids)))
        outcome = "WIN" if intervention_hit > base_hit else "LOSS" if intervention_hit < base_hit else "TIE"
        scored.append({
            "case_id": case_id,
            "question_key": question_key,
            "source_split": "dev",
            "gold_answer_ids_scoring_only": sorted(gold),
            "baseline_prediction_ids": sorted(baseline),
            "intervention_prediction_ids": sorted(intervention),
            "baseline_hits_at_1": base_hit,
            "intervention_hits_at_1": intervention_hit,
            "paired_hits_outcome": outcome,
            "baseline_f1": base_f1,
            "intervention_f1": intervention_f1,
            "delta_f1": intervention_f1 - base_f1,
            "selector_eligible": bool(live.get("selector_eligible")),
            "selector_status": status,
            "candidate_membership_valid": membership_ok,
            "candidate_provenance_valid": provenance_ok if live.get("selector_eligible") else None,
            "gold_in_candidate_pool_scoring_only": bool(gold & set(candidate_ids)) if candidate_ids else None,
        })

    wins = sum(row["paired_hits_outcome"] == "WIN" for row in scored)
    losses = sum(row["paired_hits_outcome"] == "LOSS" for row in scored)
    ties = len(scored) - wins - losses
    hit_deltas = [row["intervention_hits_at_1"] - row["baseline_hits_at_1"] for row in scored]
    f1_deltas = [row["delta_f1"] for row in scored]
    hit_ci = paired_bootstrap_ci(hit_deltas)
    mcnemar_p = exact_mcnemar_p(wins, losses)

    eligible_count = int(metadata.get("eligible_cases", 0))
    request_count = len(requests)
    sparql_count = len(sparql)
    selector_requests = sum(event.get("request_role") == "selector" for event in requests)
    request_retries = sum(event.get("retry_of") is not None or int(event.get("attempt_number", 1)) > 1 for event in requests)
    sparql_retries = sum(event.get("retry_of") is not None or int(event.get("attempt_number", 1)) > 1 for event in sparql)
    selector_valid_rate = applied_valid / selector_requests if selector_requests else 0.0
    membership_rate = applied_membership_valid / applied_valid if applied_valid else 1.0
    provenance_rate = applied_provenance_valid / applied_valid if applied_valid else 1.0
    estimated_cost = float(gate["rate_based_cost_usd"])
    api_cap = int(manifest["api_caps"]["api_request_cap"])
    select_cap = int(manifest["sparql_caps"]["physical_read_only_select_cap"])
    eligible_cap = int(manifest["sparql_caps"]["per_case_select_cap"])
    per_case_sparql = Counter(event.get("case_id") for event in sparql if event.get("case_id"))
    resource_checks = {
        "api_request_cap": request_count <= api_cap,
        "api_cost_cap": estimated_cost <= float(manifest["api_caps"]["cost_cap_usd"]),
        "sparql_select_cap": sparql_count <= select_cap,
        "per_case_sparql_cap": max(per_case_sparql.values(), default=0) <= eligible_cap,
        "zero_api_retries": request_retries == 0,
        "zero_sparql_retries": sparql_retries == 0,
        "all_requests_within_120_seconds": True,
        "all_sparql_attempts_read_only_select": all(e.get("query_type") == "SELECT" and e.get("read_only") is True for e in sparql),
    }
    all_http_api_durations: list[float] = []
    for event in requests:
        start = datetime.fromisoformat(str(event["started_at_utc"]).replace("Z", "+00:00"))
        finish = datetime.fromisoformat(str(event["finished_at_utc"]).replace("Z", "+00:00"))
        all_http_api_durations.append((finish - start).total_seconds())
    max_request_seconds = max(all_http_api_durations, default=0.0)
    resource_checks["all_requests_within_120_seconds"] = max_request_seconds <= int(manifest["api_caps"]["per_request_timeout_seconds"])

    promotion_checks = {
        "minimum_150_eligible_cases": eligible_count >= int(manifest["analysis_plan"]["minimum_eligible_cases_for_promotion"]),
        "valid_nonfallback_selector_rate_at_least_90pct": selector_valid_rate >= 0.90,
        "candidate_membership_100pct": membership_rate == 1.0,
        "candidate_provenance_100pct": provenance_rate == 1.0,
        "zero_baseline_correct_regressions": baseline_correct_regressions == 0,
        "positive_all_cohort_hits_delta": sum(hit_deltas) > 0,
        "paired_bootstrap_ci_lower_above_zero": hit_ci[0] > 0.0,
        "exact_mcnemar_p_below_0_05": mcnemar_p < 0.05,
        "resource_compliance": all(resource_checks.values()),
    }

    request_role_counts = dict(Counter(str(e.get("request_role")) for e in requests))
    request_status_counts = dict(Counter(str(e.get("http_status")) for e in requests))
    sparql_status_counts = dict(Counter(str(e.get("status")) for e in sparql))
    tokens_in = sum(int(e.get("uncached_input_tokens", e.get("input_tokens", 0)) or 0) for e in requests)
    tokens_out = sum(int(e.get("output_tokens", 0) or 0) for e in requests)
    status = "PROMOTION_CRITERIA_MET_DEVELOPMENT_ONLY" if all(promotion_checks.values()) else "NO_PROMOTION_CRITERIA_NOT_MET"
    summary = {
        "iteration": 29,
        "attempt": 3,
        "status": "SCORED_COMPLETE_PAIRED_DEVELOPMENT_EVALUATION",
        "evidence_class": manifest["evidence_class"],
        "interpretation_limit": "Fresh GrailQA dev development slice; this is not the official test holdout or confirmatory evidence.",
        "manifest_sha256": manifest_hash,
        "cohort_sha256": manifest["inference_cohort_sha256"],
        "source_dataset_sha256": manifest["source_dataset_sha256"],
        "n_intended": expected_n,
        "n_scored": len(scored),
        "primary_hits_at_1": {
            "baseline_hits": sum(row["baseline_hits_at_1"] for row in scored),
            "intervention_hits": sum(row["intervention_hits_at_1"] for row in scored),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "net_wins": wins - losses,
            "paired_delta": sum(hit_deltas) / len(scored),
            "paired_bootstrap_95pct_ci": hit_ci,
            "bootstrap_draws": 20000,
            "bootstrap_seed": 2903,
            "exact_two_sided_mcnemar_p": mcnemar_p,
        },
        "secondary": {
            "mean_baseline_f1": sum(row["baseline_f1"] for row in scored) / len(scored),
            "mean_intervention_f1": sum(row["intervention_f1"] for row in scored) / len(scored),
            "mean_paired_f1_delta": sum(f1_deltas) / len(f1_deltas),
            "eligible_count": eligible_count,
            "eligible_rate": eligible_count / len(scored),
            "selector_request_count": selector_requests,
            "selector_status_counts": dict(selector_statuses),
            "valid_nonfallback_selector_rate": selector_valid_rate,
            "candidate_membership_valid_rate_among_applied": membership_rate,
            "candidate_provenance_valid_rate_among_applied": provenance_rate,
            "baseline_correct_regressions": baseline_correct_regressions,
            "gold_in_candidate_pool_count_scoring_only": gold_candidate_coverage_count,
            "gold_candidate_pool_denominator_scoring_only": gold_candidate_coverage_denominator,
            "gold_in_candidate_pool_rate_scoring_only": gold_candidate_coverage_count / gold_candidate_coverage_denominator if gold_candidate_coverage_denominator else None,
            "eligibility_reason_counts": dict(eligibility_reasons),
        },
        "operations": {
            "api_requests": request_count,
            "api_request_cap": api_cap,
            "api_request_role_counts": request_role_counts,
            "api_http_status_counts": request_status_counts,
            "api_retries": request_retries,
            "input_tokens_uncached_or_reported": tokens_in,
            "output_tokens": tokens_out,
            "rate_based_cost_usd": estimated_cost,
            "rate_based_cost_cap_usd": manifest["api_caps"]["cost_cap_usd"],
            "request_reservation_usd": validation["accounting"].get("reserved_api_exposure_usd"),
            "sparql_attempts": sparql_count,
            "sparql_attempt_cap": select_cap,
            "sparql_status_counts": sparql_status_counts,
            "sparql_retries": sparql_retries,
            "maximum_api_request_seconds": max_request_seconds,
            "maximum_sparql_attempts_per_case": max(per_case_sparql.values(), default=0),
            "run_started_at_utc": metadata.get("run_started_at_utc"),
            "run_finished_at_utc": metadata.get("run_finished_at_utc"),
        },
        "promotion_screen": {
            "decision": status,
            "criteria": promotion_checks,
            "interpretation": "Even if every development promotion criterion passes, a fresh authorized confirmatory holdout is required before claiming generalization.",
        },
        "case_results": scored,
    }

    (RUN / "PAIRED_ANALYSIS_GOLD_ONLY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (RUN / "PAIRED_EVAL_CASES_GOLD_ONLY.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scored[0]), extrasaction="ignore")
        writer.writeheader()
        for row in scored:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    return {key: value for key, value in summary.items() if key != "case_results"}


if __name__ == "__main__":
    print(json.dumps(run_score(), indent=2, sort_keys=True))
