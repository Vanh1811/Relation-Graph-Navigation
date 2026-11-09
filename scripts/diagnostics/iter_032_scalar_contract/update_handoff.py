#!/usr/bin/env python3
"""Append the Iteration 32 decision to the shared research ledgers."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HANDOFF = ROOT / "outputs/experiments/fresh50_continuous_agent/CHATGPT_HANDOFF.md"
STATUS = ROOT / "outputs/experiments/fresh50_continuous_agent/STATUS.json"
COST = ROOT / "outputs/experiments/fresh50_continuous_agent/COST_LEDGER.json"
EVIDENCE = ROOT / "outputs/experiments/fresh50_continuous_agent/EVIDENCE_LEDGER.csv"
LOG = ROOT / "outputs/experiments/fresh50_continuous_agent/ITERATION_LOG.md"
OUT = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract"
STAMP = "2026-11-09T06:10:28Z"
BLOCK_START = "<!-- ITER32_HANDOFF_START -->"
BLOCK_END = "<!-- ITER32_HANDOFF_END -->"


def append_once(path: Path, text: str, start: str, end: str) -> None:
    current = path.read_text(encoding="utf-8")
    if start in current:
        before = current.split(start, 1)[0]
        after = current.split(end, 1)[1] if end in current else ""
        current = before + after.lstrip("\n")
    path.write_text(text + current, encoding="utf-8")


def main() -> None:
    result_paths = [
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/ITER32_DECISION.md",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/SCALAR_CONTRACT_REPORT.md",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/SCALAR_SOURCE_AUDIT.csv",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/SCALAR_SOURCE_AUDIT_GOLD_ONLY.csv",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/DEVELOPMENT_SCORING_GOLD_ONLY.csv",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/TYPED_SCALAR_SCORING_SPEC.md",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/PLAN_PRODUCER_PROVENANCE.md",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/PRODUCER_READINESS_MATRIX.csv",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/GOLD_SEPARATION_AUDIT.md",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/THESIS_CHECKPOINT.md",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/TEST_RESULTS.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/RESOURCE_ACCOUNTING.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/RUNTIME_AUDIT_SUMMARY_NON_GOLD.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/DEVELOPMENT_SCORING_SUMMARY_GOLD_ONLY.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/REJECTION_REASON_COUNTS.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/INTEGRATED_FIXTURE_RESULTS.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/SCORER_COMPATIBILITY_TESTS.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/PREDICTION_SNAPSHOT_NON_GOLD.jsonl",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/PREDECLARED_NEGATIVE_CONTROLS_NON_GOLD.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/RUNTIME_SOURCE_HASHES.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/SOURCE_HASH_MANIFEST.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/SOURCE_AND_HISTORY_INTEGRITY.json",
    ]
    block = f"""{BLOCK_START}
# Current state — Iteration 32 `SCALAR_CONTRACT_RESULT_ONLY`

**Decision:** a strict, opt-in typed scalar contract recovered supported values from the existing frozen CoR final results. It does not establish a COUNT query producer. Real gold-free COUNT readiness remains **0/68**: 68/68 target types ambiguous, 66 paths missing finalized-plan provenance, and 2 candidate-query errors.

**Same 800 frozen GrailQA dev baseline predictions, separate contracts:** entity-only = **578/800 Hits@1**, mean F1 **0.6854957**; opt-in typed mixed entity/scalar = **614/800 Hits@1**, mean F1 **0.7304957**. The typed contract accepted 39 scalar outputs; 36 were correct and 3 were wrong. Of the 42 retrospective name/gold-value matches, 36 passed strict runtime provenance and were correct. This is output representation and scoring, not a new reasoning or database-execution result. Iteration 29’s authoritative score is unchanged.

**No live calls:** 0 OpenAI requests, 0 physical SPARQL attempts, US$0. Tests: Iterations 30/31/32, 64/64 passed with `PYTHONPATH=.`. A synthetic complete-plan fixture passed the consumer/scorer shape, but no real plan producer was found and no pilot is ready.

Decision report: `RESULTS/iter_032_count_producer_and_scalar_contract/ITER32_DECISION.md`. Source audit, separated scoring, producer audit, and integrity files are in that directory. Thesis checkpoint: `RESULTS/iter_032_count_producer_and_scalar_contract/THESIS_CHECKPOINT.md`.
{BLOCK_END}

"""
    append_once(HANDOFF, block, BLOCK_START, BLOCK_END)

    status = json.loads(STATUS.read_text(encoding="utf-8"))
    for path in result_paths:
        if path not in status.setdefault("artifact_paths", []):
            status["artifact_paths"].append(path)
    status.update({
        "iteration32_status": "SCALAR_CONTRACT_RESULT_ONLY_OFFLINE_COMPLETE",
        "iteration32_decision": "SCALAR_CONTRACT_RESULT_ONLY",
        "iteration32_completed_at_utc": STAMP,
        "iteration32_cases_audited": 800,
        "iteration32_posthoc_count_cases_audited": 68,
        "iteration32_typed_scalar_acceptances": 39,
        "iteration32_typed_scalar_gold_correct_scoring_only": 36,
        "iteration32_typed_scalar_gold_incorrect_scoring_only": 3,
        "iteration32_count_producer_ready_cases": 0,
        "iteration32_api_requests": 0,
        "iteration32_physical_sparql_attempts": 0,
        "iteration32_cost_usd": 0.0,
        "iteration32_offline_tests_passed": 64,
        "iteration32_offline_tests_failed": 0,
        "iteration32_live_manifest_created": False,
        "iteration32_authorization_requested": False,
        "last_completed_experiment": "Iteration 32 typed scalar source audit and separate 800-case development rescoring completed offline; no COUNT producer found.",
        "latest_experiment_attempt": "SCALAR_CONTRACT_RESULT_ONLY; 39/68 scalar conversions; 36 correct and 3 incorrect by gold-only scoring; real COUNT producer 0/68.",
        "best_supported_conclusion": "The same frozen GrailQA dev predictions score 578/800 under the historic entity-only contract and 614/800 under an opt-in typed mixed contract; this recovers existing output representation and is not a new reasoning gain. No gold-free COUNT plan producer exists in the inspected runtime.",
        "next_action": "Consolidate the scalar representation result and COUNT producer limitation in the thesis. A future COUNT execution study requires an independent gold-free target-type and complete-plan source; no pilot is ready.",
        "status": "ITER32_SCALAR_CONTRACT_RESULT_ONLY_OFFLINE_COMPLETE",
        "remaining_questions": [
            "Can an independent, gold-free schema/type source and complete-plan producer be established outside the inspected runtime?",
            "No COUNT execution accuracy was measured; do not infer it from the typed scalar rescoring."
        ],
    })
    STATUS.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cost = json.loads(COST.read_text(encoding="utf-8"))
    cost.update({
        "iteration32_decision": "SCALAR_CONTRACT_RESULT_ONLY",
        "iteration32_api_requests": 0,
        "iteration32_physical_sparql_attempts": 0,
        "iteration32_cost_usd": 0.0,
        "iteration32_cases_audited": 800,
        "iteration32_count_cases_audited": 68,
        "iteration32_scalar_conversions": 39,
        "iteration32_correct_scalars_scoring_only": 36,
        "iteration32_offline_test_methods": 64,
        "iteration32_manifest_created": False,
        "iteration32_authorization_requested": False,
        "iteration32_cumulative_totals_incremented": False,
    })
    offline_iteration = {
        "artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/ITER32_DECISION.md",
        "cost_usd": 0.0,
        "decision": "SCALAR_CONTRACT_RESULT_ONLY",
        "experiment": "typed scalar source audit, separate 800-case development scoring, and COUNT producer provenance audit",
        "iteration": 32,
        "kg_queries": 0,
        "llm_calls": 0,
        "offline_test_methods": 64,
        "recorded_at_utc": STAMP,
        "test_methods": 64,
    }
    # This historical field mixes integer IDs with detailed records; preserve
    # every prior entry and replace only a possible Iteration 32 entry.
    retained_iterations = []
    for row in cost.get("offline_iterations", []):
        if row == 32 or (isinstance(row, dict) and row.get("iteration") == 32):
            continue
        retained_iterations.append(row)
    cost["offline_iterations"] = retained_iterations + [offline_iteration]
    offline_experiment = {
        "artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/ITER32_DECISION.md",
        "cost_usd": 0.0,
        "experiment": "scalar representation audit and independent typed scorer; no live calls",
        "input_tokens": 0,
        "iteration": 32,
        "kg_queries": 0,
        "llm_calls": 0,
        "offline_test_methods": 64,
        "output_tokens": 0,
        "recorded_at_utc": STAMP,
        "status": "COMPLETE; result-only scalar contract; COUNT producer unavailable",
    }
    cost["offline_experiment_log"] = [
        row for row in cost.get("offline_experiment_log", []) if row.get("iteration") != 32
    ] + [offline_experiment]
    COST.write_text(json.dumps(cost, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with EVIDENCE.open(newline="", encoding="utf-8") as handle:
        existing_evidence = list(csv.reader(handle))
    if not any(row and row[0] == "32" for row in existing_evidence[1:]):
        with EVIDENCE.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
            "32", "GrailQA-dev-800; posthoc COUNT subset 68",
            "Existing scalar text can be recovered with a strict typed contract while the complete COUNT producer remains absent",
            "39/68 typed scalar outputs; 36 gold-correct, 3 wrong; 0/68 gold-free plan producers",
            "HIGH for output-representation mismatch; NO new reasoning or COUNT execution evidence",
            "SCALAR_CONTRACT_RESULT_ONLY; 614/800 under a distinct mixed contract; historical entity score unchanged",
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/ITER32_DECISION.md",
            "Same frozen predictions; retrospective development scoring only; 0 API; 0 SPARQL; US$0.",
            ])

    log = LOG.read_text(encoding="utf-8")
    marker = "## Iteration 32 — typed scalar contract and COUNT producer audit"
    section = f"""\n{marker} — {STAMP}

- Decision: `SCALAR_CONTRACT_RESULT_ONLY`. A strict serializer accepted 39/68 existing numeric outputs; separate gold-only scoring found 36 correct and 3 wrong. It recovered output representation only; no COUNT query or reasoning method ran.
- Same 800 frozen baseline predictions: entity-only 578/800 Hits@1, mean F1 0.6854957; typed mixed contract 614/800, mean F1 0.7304957. This alternate contract is not directly comparable as a method gain. Iteration 29 remains unchanged.
- Producer audit: 0/68 real gold-free plans; 68 target types ambiguous; 66 path plans lack provenance; 2 query errors. Synthetic consumer integration passed but does not demonstrate a real producer.
- API 0; SPARQL 0; cost US$0; offline suites 64/64 passed. No manifest or authorization request.
- Report: `outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/ITER32_DECISION.md`.
"""
    if marker not in log:
        LOG.write_text(log.rstrip() + "\n" + section, encoding="utf-8")


if __name__ == "__main__":
    main()
