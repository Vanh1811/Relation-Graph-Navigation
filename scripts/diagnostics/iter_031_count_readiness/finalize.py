#!/usr/bin/env python3
"""Validate frozen evidence and update the Iteration 31 handoff ledgers."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "outputs/experiments/fresh50_continuous_agent"
RESULTS = LEDGER / "RESULTS"
OUT = RESULTS / "iter_031_gold_free_count_readiness"
I30 = RESULTS / "iter_030_results_first"
I29 = RESULTS / "iter_028_terminal_boundary_repair"
RUN = I29 / "RUNS/iter29_terminal_candidate_selector_attempt3"
BASELINE = OUT / "_ITER31_BASELINE_INTEGRITY.json"
MUTABLE_LEDGERS = {
    "outputs/experiments/fresh50_continuous_agent/STATUS.json",
    "outputs/experiments/fresh50_continuous_agent/COST_LEDGER.json",
    "outputs/experiments/fresh50_continuous_agent/EVIDENCE_LEDGER.csv",
    "outputs/experiments/fresh50_continuous_agent/ITERATION_LOG.md",
    "outputs/experiments/fresh50_continuous_agent/CHATGPT_HANDOFF.md",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_prior_evidence() -> dict[str, Any]:
    summary = read_json(I30 / "RESULTS_SUMMARY.json")
    phase_a = read_json(I30 / "PHASE_A_AUDIT.json")
    gate = read_json(RUN / "SCORING_GATE_VERDICT.json")
    metadata = read_json(RUN / "RUN_METADATA.json")
    trace_path = RUN / "CASE_TRACES_NON_GOLD.jsonl"
    cohort_path = I29 / "NEXT_INFERENCE_COHORT_ATTEMPT3_NON_GOLD.jsonl"
    score_cases_path = RUN / "PAIRED_EVAL_CASES_GOLD_ONLY.csv"
    score_summary_path = RUN / "PAIRED_ANALYSIS_GOLD_ONLY.json"
    paired_ref_path = I30 / "ITER29_PAIRED_CASE_REFERENCE_GOLD_ONLY.csv"
    expected = {
        "cohort": summary["source_hashes"]["cohort_sha256"],
        "score_case_csv": phase_a["source_hashes"]["score_case_csv_sha256"],
        "score_summary": summary["source_hashes"]["iter29_scored_cases_sha256"],
    }
    actual = {
        "cohort": digest(cohort_path),
        "score_case_csv": digest(score_cases_path),
        "score_summary": digest(score_summary_path),
    }
    if actual != expected:
        raise ValueError(f"Iter29 artifact hash mismatch: expected={expected}, actual={actual}")
    if gate.get("verdict") != "SCOREABLE" or gate.get("score_invocation_allowed") is not True:
        raise ValueError("Iteration 29 attempt 3 was not independently SCOREABLE")
    if gate.get("expected_case_count") != 800 or gate.get("observed_case_count") != 800:
        raise ValueError("Iteration 29 scoring gate did not verify exactly 800 cases")
    if gate.get("manifest_sha256") != metadata.get("manifest_sha256"):
        raise ValueError("Iteration 29 manifest hash differs between run metadata and gate")
    if gate.get("manifest_sha256") != phase_a["source_hashes"]["manifest_declared_sha256"]:
        raise ValueError("Iteration 29 manifest hash differs from the Iteration 30 audit")
    trace_rows = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cohort_rows = [line for line in cohort_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    score_rows = rows(score_cases_path)
    reference_rows = rows(paired_ref_path)
    if any(len(x) != 800 for x in (trace_rows, cohort_rows, score_rows, reference_rows)):
        raise ValueError("Iteration 29 trace/cohort/score/reference row count mismatch")
    for name, collection, key in (
        ("trace", [json.loads(line) for line in trace_rows], "question_key"),
        ("cohort", [json.loads(line) for line in cohort_rows], "question_key"),
        ("score", score_rows, "question_key"),
        ("reference", reference_rows, "question_key"),
    ):
        values = [str(item.get(key, "")) for item in collection]
        if len(set(values)) != 800 or "" in values:
            raise ValueError(f"Iteration 29 {name} keys are not 800 unique nonempty IDs")
    key_sets = [
        {str(item["question_key"]) for item in collection}
        for collection in (
            [json.loads(line) for line in trace_rows],
            [json.loads(line) for line in cohort_rows],
            score_rows,
            reference_rows,
        )
    ]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("Iteration 29 trace/cohort/score/reference keys are not paired exactly")
    hits_baseline = sum(int(row["baseline_hits_at_1"]) for row in score_rows)
    hits_selector = sum(int(row["intervention_hits_at_1"]) for row in score_rows)
    gate_mtime = (RUN / "SCORING_GATE_VERDICT.json").stat().st_mtime
    gate_before_gold = gate_mtime < score_cases_path.stat().st_mtime and gate_mtime < score_summary_path.stat().st_mtime
    if not gate_before_gold:
        raise ValueError("Iteration 29 gold-scored cases do not postdate the independent gate")
    if (hits_baseline, hits_selector) != (578, 578):
        raise ValueError(f"Iteration 29 paired Hits@1 mismatch: {hits_baseline}, {hits_selector}")
    pair_outcomes = Counter(row.get("paired_hits_outcome", "") for row in score_rows)
    baseline_f1 = sum(float(row["baseline_f1"]) for row in score_rows) / len(score_rows)
    selector_f1 = sum(float(row["intervention_f1"]) for row in score_rows) / len(score_rows)
    recomputed = summary["iter29_recomputed"]
    if pair_outcomes != {"TIE": 800}:
        raise ValueError(f"Iteration 29 paired Hits@1 outcomes mismatch: {pair_outcomes}")
    if not math.isclose(baseline_f1, recomputed["mean_f1_baseline"], abs_tol=1e-12):
        raise ValueError("Iteration 29 baseline mean F1 does not match the I30 recomputation")
    if not math.isclose(selector_f1, recomputed["mean_f1_terminal_selector"], abs_tol=1e-12):
        raise ValueError("Iteration 29 selector mean F1 does not match the I30 recomputation")
    if not all(summary["iter29_integrity_checks"].values()):
        raise ValueError("Iteration 30 summary recorded a failed Iteration 29 integrity check")
    return {
        "prior_iteration": 29,
        "manifest_sha256": gate["manifest_sha256"],
        "gate_verdict": gate["verdict"],
        "gate_score_invocation_allowed": gate["score_invocation_allowed"],
        "gate_verified_cases": gate["observed_case_count"],
        "trace_rows_unique": 800,
        "cohort_rows_unique": 800,
        "score_rows_unique": 800,
        "paired_reference_rows_unique": 800,
        "baseline_hits_at_1_recomputed": hits_baseline,
        "selector_hits_at_1_recomputed": hits_selector,
        "paired_hits_outcomes_recomputed": dict(pair_outcomes),
        "mean_f1_recomputed": {"baseline": baseline_f1, "selector": selector_f1},
        "score_artifacts_mtime_after_gate": gate_before_gold,
        "hashes_expected": expected,
        "hashes_actual": actual,
        "trace_sha256": digest(trace_path),
        "paired_reference_sha256": digest(paired_ref_path),
        "iter30_summary_checks_all_true": True,
        "scope": "GrailQA development evidence; not a confirmatory holdout",
    }


def update_ledgers(test_summary: dict[str, Any], runtime_summary: dict[str, Any], timestamp: str) -> dict[str, Any]:
    status_path = LEDGER / "STATUS.json"
    status = read_json(status_path)
    if "iteration31_status" not in status:
        status["pre_iter31_status_snapshot"] = {
            key: status.get(key)
            for key in ("iteration", "status", "last_completed_experiment", "latest_experiment_attempt", "next_action")
        }
    status.update({
        "iteration": 31,
        "iteration31_status": "RUNTIME_CONTRACT_BLOCKED_OFFLINE_COMPLETE",
        "iteration31_decision": "RUNTIME_CONTRACT_BLOCKED",
        "iteration31_completed_at_utc": timestamp,
        "iteration31_cases_audited": 68,
        "iteration31_runtime_ready_cases": 0,
        "iteration31_target_type_status_counts": runtime_summary["runtime_only_counts"]["type_status"],
        "iteration31_path_status_counts": runtime_summary["runtime_only_counts"]["path_status"],
        "iteration31_count_triggered_cases": runtime_summary["runtime_only_counts"]["triggered"],
        "iteration31_api_requests": 0,
        "iteration31_physical_sparql_attempts": 0,
        "iteration31_cost_usd": 0.0,
        "iteration31_offline_test_methods": test_summary["total_test_methods"],
        "iteration31_offline_test_status": test_summary["status"],
        "iteration31_manifest_created": False,
        "iteration31_pilot_authorization_requested": False,
        "iteration31_source_audit": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_031_gold_free_count_readiness/SOURCE_AND_HISTORY_INTEGRITY.json",
        "last_completed_experiment": "Iteration 31 gold-free COUNT readiness audit completed offline; 0/68 executable; no live calls.",
        "latest_experiment_attempt": "Iteration 31 decision RUNTIME_CONTRACT_BLOCKED; no pilot manifest or authorization requested.",
        "next_action": "Add a provenance-backed target type and finalized complete CountPlanContract producer plus scalar evaluator support; rerun offline acceptance before proposing any live study.",
        "status": "ITER31_RUNTIME_CONTRACT_BLOCKED_OFFLINE_COMPLETE",
    })
    write_json(status_path, status)

    cost_path = LEDGER / "COST_LEDGER.json"
    cost = read_json(cost_path)
    old_totals = {k: cost.get(k) for k in ("cost_usd", "llm_calls", "read_only_kg_queries", "input_tokens", "output_tokens")}
    entry = {
        "iteration": 31,
        "experiment": "gold-free COUNT execution readiness, isolated contract and retrospective trace audit",
        "llm_calls": 0,
        "kg_queries": 0,
        "cost_usd": 0.0,
        "offline_test_methods": test_summary["total_test_methods"],
        "decision": "RUNTIME_CONTRACT_BLOCKED",
        "artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_031_gold_free_count_readiness/ITER31_DECISION.md",
        "recorded_at_utc": timestamp,
    }
    for list_name in ("offline_iterations", "offline_experiment_log"):
        values = cost.setdefault(list_name, [])
        if not any(isinstance(item, dict) and item.get("iteration") == 31 for item in values):
            values.append(entry if list_name == "offline_experiment_log" else {**entry, "test_methods": test_summary["total_test_methods"]})
    cost.update({
        "iteration31_api_requests": 0,
        "iteration31_physical_sparql_attempts": 0,
        "iteration31_cost_usd": 0.0,
        "iteration31_offline_test_methods": test_summary["total_test_methods"],
        "iteration31_decision": "RUNTIME_CONTRACT_BLOCKED",
        "iteration31_authorization_status": "NO_LIVE_RUN; NO_AUTHORIZATION_REQUESTED",
        "iteration31_manifest_sha256": None,
        "iteration31_scope": "offline only; zero API and SPARQL transport calls",
    })
    budget = cost.setdefault("budget_remaining", {})
    budget.update({
        "iteration31_authorization_status": "NO_NEW_AUTHORIZATION_REQUESTED",
        "iteration31_new_experiment_authorized_calls": 0,
        "iteration31_new_experiment_authorized_cost_usd": 0.0,
        "iteration31_unused_caps_rollover": False,
    })
    if {k: cost.get(k) for k in old_totals} != old_totals:
        raise ValueError("cumulative cost/token/query totals changed while adding zero-cost Iteration 31")
    write_json(cost_path, cost)

    evidence_path = LEDGER / "EVIDENCE_LEDGER.csv"
    with evidence_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        evidence_rows = list(reader)
    if not fieldnames:
        raise ValueError("evidence ledger has no header")
    if not any(row.get("iteration") == "31" and row.get("question_key") == "aggregate::iter31" for row in evidence_rows):
        evidence_rows.append({
            "iteration": "31",
            "question_key": "aggregate::iter31",
            "hypothesis": "Safe COUNT execution requires a provenance-backed final target class and finalized complete runtime graph plan.",
            "supporting_trace": "68/68 target types ambiguous; 66/68 paths lack finalized-plan provenance; 2/68 candidate queries failed; 0/68 ready.",
            "confidence": "HIGH for the runtime contract gap; no accuracy claim",
            "outcome": "RUNTIME_CONTRACT_BLOCKED; isolated offline contract passed; 0 API / 0 SPARQL / US$0",
            "source_artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_031_gold_free_count_readiness/ITER31_DECISION.md",
            "notes": "Retrospective gold-defined 68-case selection; inference fields came from non-gold traces; scoring-only columns remain separate.",
        })
    with evidence_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(evidence_rows)

    log_path = LEDGER / "ITERATION_LOG.md"
    log = log_path.read_text(encoding="utf-8")
    marker = "## Iteration 31 — gold-free COUNT readiness"
    if marker not in log:
        addition = (
            f"\n{marker} — {timestamp}\n\n"
            "- Scope: offline inspection of the 68 retrospective COUNT cases, with per-case runtime classifications generated from Iteration 29 non-gold traces/cohort only.\n"
            "- Result: 68/68 target type resolutions ambiguous; 66 paths unknown for missing finalized-plan provenance; 2 candidate-query errors; 0/68 runtime-ready.\n"
            "- Implemented an isolated typed resolver, path checker, single-SELECT COUNT adapter and killable one-attempt executor contract. No production CoR/TGAR code was changed.\n"
            f"- Tests: Iteration 31 {test_summary['methods_by_suite']['iteration31_contract']}/{test_summary['methods_by_suite']['iteration31_contract']} and Iteration 30 regression {test_summary['methods_by_suite']['iteration30_regression']}/{test_summary['methods_by_suite']['iteration30_regression']} passed.\n"
            "- Decision: RUNTIME_CONTRACT_BLOCKED. No pilot manifest or authorization request. API requests 0; physical SPARQL attempts 0; cost US$0.\n"
            "- Artifacts: `outputs/experiments/fresh50_continuous_agent/RESULTS/iter_031_gold_free_count_readiness/`.\n"
        )
        log_path.write_text(log.rstrip() + "\n" + addition, encoding="utf-8")

    handoff_path = LEDGER / "CHATGPT_HANDOFF.md"
    old_handoff = handoff_path.read_text(encoding="utf-8")
    start_marker, end_marker = "<!-- ITER31_HANDOFF_START -->", "<!-- ITER31_HANDOFF_END -->"
    current = (
        "# Current state — Iteration 31 COUNT readiness blocked by missing runtime contract\n\n"
        f"**Decision:** `RUNTIME_CONTRACT_BLOCKED` ({timestamp}). 0/68 audited retrospective COUNT cases are executable from actual runtime evidence. All 68 target class resolutions are ambiguous; 66 path checks are unknown because no finalized expected plan is recorded; 2 candidate queries failed.\n\n"
        "The isolated Iteration 31 resolver, completeness checker and one-shot COUNT adapter passed 37 local tests; the Iteration 30 regression suite passed 14. These were mock/offline tests, not Freebase execution. Iteration 31 made 0 OpenAI requests, 0 SPARQL attempts, and spent US$0. The adapter is not wired into production, and the current Iteration 29 scorer rejects aggregate output.\n\n"
        "Gold-only diagnostics remain explicitly separate: 13/68 gold plans need multi-edge joins; runtime relation-set agreement is 39/68 posthoc, not a rescue ceiling. The trigger audit is development-only and posthoc.\n\n"
        "**Next:** emit a provenance-backed target class and finalized complete plan from the runtime, support a typed aggregate result in the evaluator, and pass authentic callsite-shaped offline acceptance tests. No pilot manifest or authorization request exists.\n\n"
        "Report: `RESULTS/iter_031_gold_free_count_readiness/ITER31_DECISION.md`. Runtime contract: `RESULTS/iter_031_gold_free_count_readiness/COUNT_RUNTIME_CONTRACT.md`.\n"
    )
    block = f"{start_marker}\n{current}{end_marker}"
    if start_marker in old_handoff and end_marker in old_handoff:
        left = old_handoff.split(start_marker, 1)[0]
        right = old_handoff.split(end_marker, 1)[1]
        handoff_path.write_text(left + block + right, encoding="utf-8")
    else:
        handoff_path.write_text(block + "\n\n--- Archived prior handoff ---\n\n" + old_handoff, encoding="utf-8")
    return {"cumulative_totals_preserved": old_totals, "status_keys_updated": True}


def main() -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    baseline = read_json(BASELINE)
    prior = verify_prior_evidence()
    runtime_summary = read_json(OUT / "RUNTIME_AUDIT_SUMMARY.json")
    test_summary = read_json(OUT / "OFFLINE_TEST_RESULTS.json")
    if test_summary.get("status") != "PASS" or test_summary.get("total_test_methods") != 51:
        raise ValueError("offline test suite is not the expected PASS/51 result")
    if runtime_summary["runtime_only_counts"]["runtime_ready"] != 0:
        raise ValueError("runtime audit no longer supports the blocked decision")
    update_note = update_ledgers(test_summary, runtime_summary, timestamp)

    immutable_before = {
        name: value for name, value in baseline["protected_sha256_before_iter31_writes"].items()
        if name not in MUTABLE_LEDGERS
    }
    immutable_after = {}
    for name in immutable_before:
        path = ROOT / name
        immutable_after[name] = digest(path) if path.exists() else None
    matching = {name: immutable_before[name] == immutable_after[name] for name in immutable_before}
    if not all(matching.values()):
        failed = [name for name, okay in matching.items() if not okay]
        raise ValueError(f"protected production/history files changed: {failed}")
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if git_head != baseline["git_head"]:
        raise ValueError("git HEAD changed during Iteration 31")
    git_status = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    new_sources = [
        "chain_of_relations/methods/iter_031_count_readiness/__init__.py",
        "chain_of_relations/methods/iter_031_count_readiness/evidence.py",
        "chain_of_relations/methods/iter_031_count_readiness/type_resolution.py",
        "chain_of_relations/methods/iter_031_count_readiness/path_completeness.py",
        "chain_of_relations/methods/iter_031_count_readiness/runtime_contract.py",
        "tests/test_iter_031_count_readiness.py",
        "scripts/diagnostics/iter_031_count_readiness/audit_runtime.py",
        "scripts/diagnostics/iter_031_count_readiness/finalize.py",
    ]
    output_files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "SOURCE_AND_HISTORY_INTEGRITY.json")
    integrity = {
        "iteration": 31,
        "decision": "RUNTIME_CONTRACT_BLOCKED",
        "created_at_utc": timestamp,
        "network_calls": {"openai": 0, "sparql": 0},
        "prior_iter29_validation": prior,
        "git": {
            "head_before": baseline["git_head"],
            "head_after": git_head,
            "head_unchanged": git_head == baseline["git_head"],
            "staged_paths": staged,
            "commit_or_push_performed": False,
            "status_before": baseline["git_status_before_iter31_writes"],
            "status_after": git_status,
        },
        "preexisting_protected_files_unchanged": matching,
        "preexisting_protected_files_all_unchanged": all(matching.values()),
        "mutable_handoff_ledgers_updated_by_design": sorted(MUTABLE_LEDGERS),
        "root_ledger_update_summary": update_note,
        "new_iteration31_source_hashes": {name: digest(ROOT / name) for name in new_sources},
        "iteration31_artifact_hashes": {str(path.relative_to(ROOT)): digest(path) for path in output_files},
        "historical_iter29_iter30_outputs_modified": False,
        "production_cor_tgar_sources_modified": False,
        "score_created_in_iteration31": False,
        "test_summary": test_summary["methods_by_suite"],
    }
    write_json(OUT / "SOURCE_AND_HISTORY_INTEGRITY.json", integrity)
    print(json.dumps({
        "decision": integrity["decision"],
        "iter29_gate": prior["gate_verdict"],
        "iter29_recomputed_hits": [prior["baseline_hits_at_1_recomputed"], prior["selector_hits_at_1_recomputed"]],
        "protected_files_unchanged": sum(matching.values()),
        "protected_files_checked": len(matching),
        "new_source_files": len(new_sources),
        "artifacts": len(output_files) + 1,
        "git_head_unchanged": integrity["git"]["head_unchanged"],
        "staged_paths": staged,
        "ledger_totals_preserved": update_note["cumulative_totals_preserved"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
