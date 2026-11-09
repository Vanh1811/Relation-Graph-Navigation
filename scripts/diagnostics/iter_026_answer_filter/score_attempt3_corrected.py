"""Corrected offline scorer preserving the frozen attempt-3 scorer artifact."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:  # direct script invocation and package-based fixture tests
    from .selector import paired_outcome
except ImportError:
    from selector import paired_outcome

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness"
GOLD = RESULTS / "ATTEMPT3_PREPARATION/SCORING_GOLD_ONLY.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(run_dir: Path, output: Path | None = None) -> dict[str, Any]:
    # This scorer is deliberately separate from run_live.py. It is the only
    # I26 runtime component that opens the gold sidecar.
    gold_rows = {row["case_id"]: row for row in read_jsonl(GOLD)}
    result_rows = read_jsonl(run_dir / "PAIRED_RESULTS_NON_GOLD.jsonl")
    metadata_path = run_dir / "RUN_METADATA.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    request_journal_path = run_dir / "REQUEST_JOURNAL.jsonl"
    request_journal = read_jsonl(request_journal_path) if request_journal_path.is_file() else []
    by_id = {row["case_id"]: row for row in result_rows}
    scored = []
    outcomes = Counter()
    selector_attempts = selector_valid = 0
    selector_accepted_membership = 0
    eligible = 0
    baseline_correct_regressions = 0
    candidate_conditioned = []
    eligibility_reasons = Counter()
    selector_statuses = Counter()
    baseline_fallbacks = 0
    prediction_sources = Counter()
    execution_statuses = Counter()
    for case_id, gold_row in gold_rows.items():
        run_row = by_id.get(case_id)
        if run_row is None:
            continue
        gold_ids = list(gold_row["gold_answer_ids_scoring_only"])
        baseline_ids = list(run_row.get("baseline_prediction_ids") or [])
        intervention_ids = list(run_row.get("intervention_prediction_ids") or [])
        paired = paired_outcome(baseline_ids, intervention_ids, gold_ids)
        outcomes[paired["paired_hits"]] += 1
        eligible += int(bool(run_row.get("selector_eligible")))
        prediction_sources[str(run_row.get("prediction_source", "UNKNOWN"))] += 1
        execution_statuses[str(run_row.get("execution_status", "UNKNOWN"))] += 1
        eligibility_reasons[str(run_row.get("selector_eligibility_reason", "UNKNOWN"))] += 1
        if run_row.get("selector_eligible") and set(gold_ids) & set(run_row.get("candidate_ids_snapshot") or []):
            candidate_conditioned.append(paired)
        if run_row.get("selector_invoked"):
            selector_attempts += 1
            selector_record = run_row.get("selector") or {}
            selector_statuses[str(selector_record.get("status", "UNKNOWN"))] += 1
            baseline_fallbacks += int(selector_record.get("prediction_source") == "UNCHANGED_BASELINE_FALLBACK")
            if selector_record.get("status") == "VALID_SELECTION" and selector_record.get("valid"):
                selector_valid += 1
                selector_accepted_membership += int(bool(selector_record.get("candidate_membership_valid")))
        if paired["baseline"]["hits_at_1"] and not paired["intervention"]["hits_at_1"]:
            baseline_correct_regressions += 1
        scored.append({
            "case_id": case_id,
            "question_key": gold_row["question_key"],
            "dataset_scoring_only": gold_row.get("dataset"),
            "gold_answer_count_scoring_only": len(set(gold_ids)),
            "gold_present_in_candidates_diagnostic_only": bool(
                set(gold_ids) & set(run_row.get("candidate_ids_snapshot") or [])
            ),
            "baseline": paired["baseline"],
            "intervention": paired["intervention"],
            "paired_hits": paired["paired_hits"],
            "hit_delta": paired["hit_delta"],
            "f1_delta": paired["f1_delta"],
            "selector_eligible": bool(run_row.get("selector_eligible")),
            "selector_invoked": bool(run_row.get("selector_invoked")),
            "selector_status": (run_row.get("selector") or {}).get("status"),
            "prediction_source": run_row.get("prediction_source"),
            "execution_status": run_row.get("execution_status"),
            "upstream_failure_metadata": run_row.get("upstream_failure_metadata"),
        })
    complete = (
        len(result_rows) == len(gold_rows)
        and len(scored) == len(gold_rows)
        and len(by_id) == len(gold_rows)
        and set(by_id) == set(gold_rows)
    )
    wins, losses = outcomes["WIN"], outcomes["LOSS"]
    net_hits = wins - losses
    mean_f1_delta = sum(row["f1_delta"] for row in scored) / len(scored) if scored else 0.0
    latencies = [float(row.get("latency_seconds", 0.0) or 0.0) for row in result_rows]
    finish_reasons = Counter(
        str(row.get("finish_reason") or "MISSING")
        for row in request_journal
        if row.get("event") == "REQUEST_FINISHED" and row.get("status") == "HTTP_OK"
    )
    conditional_baseline_hits = sum(row["baseline"]["hits_at_1"] for row in candidate_conditioned)
    conditional_intervention_hits = sum(row["intervention"]["hits_at_1"] for row in candidate_conditioned)
    valid_rate = selector_valid / selector_attempts if selector_attempts else 0.0
    membership_rate = selector_accepted_membership / selector_valid if selector_valid else 0.0
    gate_pass = bool(
        complete
        and metadata.get("status") == "COMPLETE"
        and eligible >= 8
        and selector_attempts > 0
        and valid_rate >= 0.90
        and membership_rate == 1.0
        and net_hits >= 3
        and baseline_correct_regressions == 0
        and mean_f1_delta >= 0.0
    )
    report = {
        "evidence_class": metadata.get("evidence_class", "FIXTURE_TEST" if result_rows else "OFFLINE_DIAGNOSTIC"),
        "status": "COMPLETE" if complete and metadata.get("status") == "COMPLETE" else "INCOMPLETE",
        "question_unit_count_scored": len(scored),
        "question_unit_count_frozen": len(gold_rows),
        "paired_hits_at_1": {"wins": wins, "losses": losses, "ties": outcomes["TIE"], "net_wins": net_hits},
        "mean_f1_delta_intervention_minus_baseline": mean_f1_delta,
        "gold_in_candidate_conditional": {
            "eligible_questions_with_gold_in_candidates": len(candidate_conditioned),
            "baseline_hits_at_1": conditional_baseline_hits,
            "intervention_hits_at_1": conditional_intervention_hits,
            "baseline_accuracy": conditional_baseline_hits / len(candidate_conditioned) if candidate_conditioned else None,
            "intervention_accuracy": conditional_intervention_hits / len(candidate_conditioned) if candidate_conditioned else None,
            "interpretation": "diagnostic conditional subset; not the primary denominator",
        },
        "baseline_correct_regressions": baseline_correct_regressions,
        "selector_eligible_count": eligible,
        "selector_attempt_count": selector_attempts,
        "selector_status_counts": dict(selector_statuses),
        "baseline_fallback_count": baseline_fallbacks,
        "prediction_source_counts": dict(prediction_sources),
        "execution_status_counts": dict(execution_statuses),
        "upstream_empty_or_truncated_case_count": execution_statuses["UPSTREAM_EMPTY_OR_TRUNCATED_FAILURE"],
        "selector_invalid_or_empty_count": sum(
            count for status, count in selector_statuses.items()
            if status not in {"VALID_SELECTION", "VALID_ABSTAIN_BASELINE_FALLBACK"}
        ),
        "valid_nonempty_selector_rate": valid_rate,
        "candidate_membership_rate_among_accepted_outputs": membership_rate,
        "eligibility_reason_counts": dict(eligibility_reasons),
        "usage_and_execution": {
            "api_requests_started_including_uncertain": metadata.get("api_requests_started", 0),
            "input_tokens_reported": metadata.get("api_input_tokens", 0),
            "output_tokens_reported": metadata.get("api_output_tokens", 0),
            "rate_estimated_actual_cost_usd": metadata.get("api_actual_cost_usd", 0.0),
            "worst_case_reserved_cost_usd": metadata.get(
                "api_cost_reserved_usd", metadata.get("api_reserved_cost_usd", 0.0)
            ),
            "physical_read_only_sparql_attempts_including_preflight": metadata.get("physical_sparql_attempts", 0),
            "model_finish_reason_counts": dict(finish_reasons),
            "mean_case_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
            "latency_seconds_by_question": {row["case_id"]: row.get("latency_seconds") for row in result_rows},
        },
        "promotion_gate": "PASS_TO_NEW_AUTHORIZATION_FOR_LARGER_TEST" if gate_pass else "FAIL_OR_INCONCLUSIVE_NO_TUNING_ON_THIS_COHORT",
        "scored_cases": scored,
        "gold_was_read_only_by_this_offline_scorer": True,
    }
    destination = output or (run_dir / "PAIRED_ANALYSIS_GOLD_ONLY.json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = score(args.run_dir, args.output)
    print(json.dumps({key: report[key] for key in (
        "status", "question_unit_count_scored", "paired_hits_at_1",
        "baseline_correct_regressions", "promotion_gate",
    )}, indent=2))


if __name__ == "__main__":
    main()
