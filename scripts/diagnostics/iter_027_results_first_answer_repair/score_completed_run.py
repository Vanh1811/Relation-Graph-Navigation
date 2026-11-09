"""Gold-only scoring for a fully completed, frozen Iteration 27 run.

This script refuses partial runs and never runs during inference. It reads the
gold GrailQA source only after all 40 paired outputs are present and saved.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_027_results_first_answer_repair"
RUN = OUT / "RUNS/iter27_scv_empty_only"
DATA = ROOT / "data/processed/grailqa_dev_analysis.jsonl"


def score(gold: set[str], predicted: set[str]) -> tuple[int, float]:
    if not gold or not predicted:
        return 0, 0.0
    overlap = len(gold & predicted)
    precision, recall = overlap / len(predicted), overlap / len(gold)
    return int(overlap > 0), (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _bootstrap_mean_ci(deltas: list[float], *, seed: int = 270027, draws: int = 20000) -> list[float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = [sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws)]
    means.sort()
    return [means[int(0.025 * draws)], means[min(draws - 1, int(0.975 * draws))]]


def run_score() -> dict[str, Any]:
    metadata = json.loads((RUN / "RUN_METADATA.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "COMPLETE" or metadata.get("cases_completed") != 40:
        raise SystemExit("refusing to score: live run is not complete with all 40 cases")
    cohort_manifest = json.loads((OUT / "COHORT_MANIFEST.json").read_text(encoding="utf-8"))
    cohort = [json.loads(line) for line in (OUT / "INFERENCE_COHORT_NON_GOLD.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold_by_id = {str(row["id"]): row for row in raw_rows if str(row.get("split", "")).casefold() == "dev"}
    live_rows = [json.loads(line) for line in (RUN / "PAIRED_RESULTS_NON_GOLD.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    snapshots = [json.loads(line) for line in (RUN / "FROZEN_CASE_SNAPSHOTS_NON_GOLD.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cohort) != 40 or len(live_rows) != 40:
        raise SystemExit("cohort or result row count is not 40")
    if len(snapshots) != 40:
        raise SystemExit("frozen baseline/candidate snapshot count is not 40")
    if [row["question_key"] for row in cohort] != cohort_manifest["selected_question_keys"]:
        raise SystemExit("inference cohort order differs from its frozen manifest")
    if [row["question_key"] for row in live_rows] != cohort_manifest["selected_question_keys"]:
        raise SystemExit("paired result order differs from the frozen question identities")

    cases: list[dict[str, Any]] = []
    from scripts.diagnostics.iter_026_answer_filter.selector import candidate_snapshot_sha256

    for infer, live, snapshot in zip(cohort, live_rows, snapshots):
        if live["case_id"] != snapshot["case_id"] or live["question_key"] != snapshot["question_key"]:
            raise SystemExit(f"baseline/snapshot identity mismatch for {infer['question_key']}")
        if list(live.get("baseline_prediction_ids", [])) != list(snapshot.get("baseline_prediction_ids", [])):
            raise SystemExit(f"baseline output differs from its pre-intervention snapshot for {infer['question_key']}")
        frozen_candidates = snapshot.get("candidate_snapshot") or []
        frozen_hash = candidate_snapshot_sha256(frozen_candidates) if frozen_candidates else None
        if frozen_hash != snapshot.get("candidate_snapshot_sha256") or frozen_hash != live.get("candidate_snapshot_sha256"):
            raise SystemExit(f"candidate snapshot hash mismatch for {infer['question_key']}")
        if [candidate["id"] for candidate in frozen_candidates] != list(live.get("candidate_ids_snapshot", [])):
            raise SystemExit(f"candidate IDs differ from frozen snapshot for {infer['question_key']}")
        dataset_id = infer["question_key"].split("::", 1)[1]
        raw = gold_by_id.get(dataset_id)
        if raw is None or raw.get("question") != infer["question"]:
            raise SystemExit(f"gold source identity/text mismatch for {infer['question_key']}")
        gold = set(map(str, raw.get("answer_ids", []) or []))
        baseline = set(map(str, live.get("baseline_prediction_ids", []) or []))
        intervention = set(map(str, live.get("intervention_prediction_ids", []) or []))
        base_hit, base_f1 = score(gold, baseline)
        treatment_hit, treatment_f1 = score(gold, intervention)
        candidates = set(map(str, live.get("candidate_ids_snapshot", []) or []))
        accepted = live.get("scv", {})
        candidate_membership = (
            set(map(str, accepted.get("selected_ids", []) or [])).issubset(candidates)
            if accepted and accepted.get("status") == "VALID_SELECTION" else None
        )
        cases.append({
            "case_id": live["case_id"],
            "question_key": live["question_key"],
            "source_split": "dev",
            "router_execution_mode": live.get("router_execution_mode"),
            "answer_cardinality_scoring_only": len(gold),
            "gold_answer_ids_scoring_only": sorted(gold),
            "baseline_prediction_ids": sorted(baseline),
            "intervention_prediction_ids": sorted(intervention),
            "gold_in_candidate_pool_scoring_only": bool(gold & candidates) if candidates else None,
            "baseline_hits_at_1": base_hit,
            "intervention_hits_at_1": treatment_hit,
            "paired_hits_outcome": "WIN" if treatment_hit > base_hit else "LOSS" if treatment_hit < base_hit else "TIE",
            "delta_f1": treatment_f1 - base_f1,
            "baseline_f1": base_f1,
            "intervention_f1": treatment_f1,
            "baseline_correct_regression": bool(base_hit and not treatment_hit),
            "scv_eligible": bool(live.get("scv_eligible")),
            "scv_status": accepted.get("status") if accepted else None,
            "scv_valid_nonempty": bool(accepted and accepted.get("status") == "VALID_SELECTION" and accepted.get("selected_ids")),
            "accepted_candidate_membership_valid": candidate_membership,
        })

    wins = sum(case["paired_hits_outcome"] == "WIN" for case in cases)
    losses = sum(case["paired_hits_outcome"] == "LOSS" for case in cases)
    ties = len(cases) - wins - losses
    hit_deltas = [case["intervention_hits_at_1"] - case["baseline_hits_at_1"] for case in cases]
    f1_deltas = [case["delta_f1"] for case in cases]
    eligible = [case for case in cases if case["scv_eligible"]]
    valid_nonempty = sum(case["scv_valid_nonempty"] for case in eligible)
    accepted = [case for case in cases if case["scv_status"] == "VALID_SELECTION"]
    membership_ok = sum(case["accepted_candidate_membership_valid"] is True for case in accepted)
    gold_covered = [case for case in cases if case["gold_in_candidate_pool_scoring_only"] is not None]
    hit_ci = _bootstrap_mean_ci(hit_deltas)
    actual_cost = float(metadata.get("api_actual_cost_usd", 0.0) or 0.0)
    reserved_cost = float(metadata.get("api_reserved_cost_usd", 0.0) or 0.0)
    api_count = int(metadata.get("api_requests_started", 0) or 0)
    select_count = int(metadata.get("physical_sparql_attempts", 0) or 0)
    valid_rate = valid_nonempty / len(eligible) if eligible else 0.0
    membership_rate = membership_ok / len(accepted) if accepted else 0.0
    operational_gates = {
        "all_40_completed": len(cases) == 40,
        "api_request_cap": api_count <= 400,
        "actual_cost_cap": actual_cost <= 1.05152 + 1e-12,
        "reserved_cost_cap": reserved_cost <= 1.05152 + 1e-12,
        "physical_select_cap": select_count <= 650,
        "zero_retries": metadata.get("retries") == 0,
        "no_resume": metadata.get("resumed") is False,
        "no_baseline_correct_regressions": sum(case["baseline_correct_regression"] for case in cases) == 0,
        "accepted_output_membership_100pct": not accepted or membership_rate == 1.0,
        "valid_nonempty_rate_at_least_95pct_when_eligible": not eligible or valid_rate >= 0.95,
    }
    benefit_gates = {
        "eligible_cases_at_least_8": len(eligible) >= 8,
        "positive_hits_delta": sum(hit_deltas) > 0,
        "net_hits_wins_at_least_3": wins - losses >= 3,
        "paired_bootstrap_ci_lower_gt_zero": hit_ci[0] > 0,
    }
    hard_operational_gates = {
        key: value for key, value in operational_gates.items()
        if key != "valid_nonempty_rate_at_least_95pct_when_eligible"
    }
    if not all(hard_operational_gates.values()):
        screen_decision = "NO_PROMOTION_OPERATIONAL_OR_SAFETY_GATE_FAILED"
    elif not benefit_gates["eligible_cases_at_least_8"]:
        screen_decision = "INCONCLUSIVE_LOW_ELIGIBILITY"
    elif not operational_gates["valid_nonempty_rate_at_least_95pct_when_eligible"]:
        screen_decision = "NO_PROMOTION_VALID_NONEMPTY_RATE_BELOW_95PCT"
    elif not benefit_gates["positive_hits_delta"]:
        screen_decision = "REJECT_NO_POSITIVE_PAIRED_HITS_EFFECT"
    elif all(benefit_gates.values()):
        screen_decision = "ELIGIBLE_FOR_CAUTIOUS_PROMOTION_CONSIDERATION"
    else:
        screen_decision = "EXPLORATORY_ONLY_BENEFIT_UNCERTAIN"
    route_strata: dict[str, dict[str, int]] = {}
    cardinality_strata: dict[str, dict[str, int]] = {}
    for case in cases:
        for strata, name in ((route_strata, str(case["router_execution_mode"])),
                             (cardinality_strata, str(case["answer_cardinality_scoring_only"]))):
            counts = strata.setdefault(name, {"wins": 0, "losses": 0, "ties": 0, "n": 0})
            counts["n"] += 1
            counts["wins"] += int(case["paired_hits_outcome"] == "WIN")
            counts["losses"] += int(case["paired_hits_outcome"] == "LOSS")
            counts["ties"] += int(case["paired_hits_outcome"] == "TIE")
    summary = {
        "iteration": 27,
        "status": "SCORED_COMPLETE_PAIRED_DEVELOPMENT_EVALUATION",
        "evidence_class": "GRAILQA_DEV_DEVELOPMENT_NOT_CONFIRMATORY_HOLDOUT",
        "manifest_sha256": metadata["manifest_sha256"],
        "cohort_manifest_sha256": hashlib.sha256((OUT / "COHORT_MANIFEST.json").read_bytes()).hexdigest(),
        "n_intended": 40,
        "n_completed": len(cases),
        "primary_hits_at_1": {
            "baseline_hits": sum(case["baseline_hits_at_1"] for case in cases),
            "intervention_hits": sum(case["intervention_hits_at_1"] for case in cases),
            "wins": wins, "losses": losses, "ties": ties,
            "net_wins": wins - losses,
            "paired_delta": sum(hit_deltas) / len(cases),
            "question_level_paired_bootstrap_95pct_ci": hit_ci,
        },
        "secondary": {
            "mean_baseline_f1": sum(case["baseline_f1"] for case in cases) / len(cases),
            "mean_intervention_f1": sum(case["intervention_f1"] for case in cases) / len(cases),
            "mean_paired_f1_delta": sum(f1_deltas) / len(cases),
            "eligible_count": len(eligible),
            "eligible_fraction_of_intended": len(eligible) / len(cases),
            "valid_nonempty_selector_outputs": valid_nonempty,
            "valid_nonempty_rate_among_eligible": valid_nonempty / len(eligible) if eligible else None,
            "accepted_selector_outputs": len(accepted),
            "accepted_output_candidate_membership_rate": membership_ok / len(accepted) if accepted else None,
            "baseline_correct_regressions": sum(case["baseline_correct_regression"] for case in cases),
            "gold_in_candidate_pool_count_scoring_only": sum(bool(case["gold_in_candidate_pool_scoring_only"]) for case in gold_covered),
            "gold_coverage_denominator_scoring_only": len(gold_covered),
            "gold_in_candidate_pool_rate_scoring_only": sum(bool(case["gold_in_candidate_pool_scoring_only"]) for case in gold_covered) / len(gold_covered) if gold_covered else None,
        },
        "operations": {
            key: metadata.get(key) for key in (
                "eligible_cases", "selector_requests", "api_requests_started", "api_input_tokens",
                "api_output_tokens", "api_actual_cost_usd", "api_reserved_cost_usd",
                "physical_sparql_attempts", "retries", "resumed",
            )
        },
        "posthoc_descriptive_strata": {
            "router_execution_mode": route_strata,
            "gold_answer_cardinality": cardinality_strata,
        },
        "promotion_screen": {
            "decision": screen_decision,
            "operational_and_safety_gates": operational_gates,
            "benefit_gates": benefit_gates,
            "interpretation": "Development evidence only; passing this screen would permit cautious promotion consideration, not establish confirmatory generalization.",
        },
        "case_results": cases,
    }
    (RUN / "PAIRED_ANALYSIS_GOLD_ONLY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (RUN / "PAIRED_EVAL_CASES.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]), extrasaction="ignore")
        writer.writeheader()
        for case in cases:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in case.items()})
    return {key: value for key, value in summary.items() if key != "case_results"}


if __name__ == "__main__":
    print(json.dumps(run_score(), indent=2))
