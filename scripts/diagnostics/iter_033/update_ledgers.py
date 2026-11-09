#!/usr/bin/env python3
"""Idempotently append Iteration 33's offline decision to project ledgers."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROJECT = ROOT / "outputs/experiments/fresh50_continuous_agent"
OUTDIR = PROJECT / "RESULTS/iter_033_scalar_generalization_and_upstream_decision"
RECORDED = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

ARTIFACTS = [
    "ITER33_DECISION.md",
    "REPRESENTATION_RESULT_VALIDITY.md",
    "REPRESENTATION_AUDIT.json",
    "REPRESENTATION_CASES.csv",
    "RUNTIME_REPRESENTATION_AUDIT_NON_GOLD.json",
    "WEBQSP_COHORT_AUDIT.json",
    "INDEPENDENT_COHORT_FEASIBILITY.md",
    "EVALUATION_PROTOCOL.md",
    "UPSTREAM_GAP_DECISION.md",
    "UPSTREAM_EVIDENCE_AUDIT.json",
    "COUNT_ON_HOLD.md",
    "NO_LIVE_MANIFEST.md",
    "TEST_RESULTS.json",
    "SOURCE_AND_HISTORY_INTEGRITY.json",
    "COST_AND_SCOPE.json",
    "THESIS_CHECKPOINT.md",
]
SCRIPTS = [
    "scripts/diagnostics/iter_033/typed_scoring.py",
    "scripts/diagnostics/iter_033/audit_runtime.py",
    "scripts/diagnostics/iter_033/score_representation.py",
    "scripts/diagnostics/iter_033/audit_webqsp_cohort.py",
    "scripts/diagnostics/iter_033/audit_upstream_evidence.py",
    "scripts/diagnostics/iter_033/update_ledgers.py",
    "scripts/diagnostics/iter_033/finalize_integrity.py",
    "tests/test_iter_033_typed_representation.py",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    cost_path = PROJECT / "COST_LEDGER.json"
    cost = read_json(cost_path)
    cost.update({
        "iteration33_status": "OFFLINE_COMPLETE_NO_CLEAN_SCALAR_GENERALIZATION_COHORT",
        "iteration33_api_requests": 0,
        "iteration33_physical_sparql_attempts": 0,
        "iteration33_cost_usd": 0.0,
        "iteration33_offline_tests_passed": 74,
        "iteration33_manifest_created": False,
        "iteration33_authorization_requested": False,
        "iteration33_cost_scope_note": "Iteration 33 added zero-cost offline work only. Existing top-level cumulative ledger figures remain at their historical Iteration 27 scope; later STATUS snapshots and Iteration 29 attempt-level records are not reconciled and were not overwritten or summed.",
    })
    logs = cost.setdefault("offline_experiment_log", [])
    if not any(entry.get("iteration") == 33 for entry in logs if isinstance(entry, dict)):
        logs.append({
            "iteration": 33,
            "recorded_at_utc": RECORDED,
            "experiment": "independent typed-representation reproduction, WebQSP cohort feasibility, and upstream provenance decision",
            "status": "OFFLINE_COMPLETE; NO_CLEAN_INDEPENDENT_COHORT; UPSTREAM_INTERVENTION_NOT_READY",
            "artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_033_scalar_generalization_and_upstream_decision/ITER33_DECISION.md",
            "api_requests": 0,
            "llm_calls": 0,
            "kg_queries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "offline_test_methods": 74,
        })
    offline = cost.setdefault("offline_iterations", [])
    if not any(entry.get("iteration") == 33 for entry in offline if isinstance(entry, dict)):
        offline.append({
            "iteration": 33,
            "recorded_at_utc": RECORDED,
            "decision": "NO_CLEAN_INDEPENDENT_COHORT; UPSTREAM_INTERVENTION_NOT_READY; COUNT_PRODUCER_BLOCKED",
            "experiment": "retrospective representation reproduction and independent-source feasibility audit",
            "artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_033_scalar_generalization_and_upstream_decision/ITER33_DECISION.md",
            "llm_calls": 0,
            "kg_queries": 0,
            "cost_usd": 0.0,
            "test_methods": 74,
        })
    write_json(cost_path, cost)

    status_path = PROJECT / "STATUS.json"
    status = read_json(status_path)
    conclusion = (
        "Iteration 33 independently reproduced the 578/800 historic entity-only and 614/800 supplementary typed-mixed scores on identical frozen GrailQA dev predictions; 39 scalar conversions include 36 correct and 3 incorrect. This is representation recovery, not method gain. No clean prospective trigger-positive cohort, real COUNT producer, or evidence-supported upstream intervention is available."
    )
    status.update({
        "iteration": 33,
        "status": "ITER33_OFFLINE_COMPLETE_NO_CLEAN_SCALAR_GENERALIZATION_COHORT",
        "best_supported_conclusion": conclusion,
        "last_completed_experiment": "Iteration 33 offline representation reproduction, independent WebQSP cohort feasibility audit, and upstream evidence decision completed; no live run or method intervention.",
        "latest_experiment_attempt": "DEV_REPRESENTATION_RECOVERY reproduced; PROSPECTIVE_EVALUATION_NOT_RUN; COUNT_PRODUCER_BLOCKED; UPSTREAM_INTERVENTION_NOT_READY.",
        "next_action": "Consolidate the bounded development representation result and limitations in the thesis. Reopen experiments only with an untouched sufficiently trigger-positive cohort or an independently sourced gold-free complete-plan/step-provenance source.",
        "iteration33_status": "OFFLINE_COMPLETE_NO_CLEAN_SCALAR_GENERALIZATION_COHORT",
        "iteration33_decision": "DEV_REPRESENTATION_RECOVERY; NO_CLEAN_INDEPENDENT_COHORT; UPSTREAM_INTERVENTION_NOT_READY; COUNT_PRODUCER_BLOCKED",
        "iteration33_completed_at_utc": RECORDED,
        "iteration33_cases_reproduced": 800,
        "iteration33_legacy_hits_at_1": 578,
        "iteration33_typed_supplementary_hits_at_1": 614,
        "iteration33_typed_scalars_accepted": 39,
        "iteration33_typed_scalars_correct_scoring_only": 36,
        "iteration33_typed_scalars_incorrect_scoring_only": 3,
        "iteration33_webqsp_exposure_eligible_rows": 1628,
        "iteration33_webqsp_frozen_trigger_positive_rows": 4,
        "iteration33_webqsp_status_after_label_audit": "DEVELOPMENT_EXPOSED; NOT A HOLDOUT",
        "iteration33_count_producer_ready_cases": 0,
        "iteration33_upstream_intervention": "NOT_READY; no runtime behavior change",
        "iteration33_api_requests": 0,
        "iteration33_physical_sparql_attempts": 0,
        "iteration33_cost_usd": 0.0,
        "iteration33_offline_tests_passed": 74,
        "iteration33_live_manifest_created": False,
        "iteration33_authorization_requested": False,
        "iteration33_cost_scope_note": "Historical top-level and nested cumulative ledger snapshots are differently scoped; Iteration 33 records zero and does not rewrite or sum their counters.",
    })
    status["budget_remaining"].update({
        "iteration33_api_requests": 0,
        "iteration33_physical_sparql_attempts": 0,
        "iteration33_cost_usd": 0.0,
        "iteration33_manifest_created": False,
        "iteration33_paid_authorization_active": False,
        "iteration33_scope_note": "No cumulative budget counter was incremented or rewritten; see the Iteration 33 COST_AND_SCOPE.json for historical scope mismatch.",
    })
    status["iteration33_remaining_research_questions"] = [
        "Can a fresh, officially scoreable dataset provide substantially more examples that pass the frozen scalar trigger without prior label exposure?",
        "Can an independent runtime trace source retain directional relation rows and active bindings well enough to support one non-oracle upstream intervention?",
        "Can a real gold-free type source and complete semantic-plan producer be identified for COUNT?",
    ]
    paths = status.setdefault("artifact_paths", [])
    for path in [f"outputs/experiments/fresh50_continuous_agent/RESULTS/iter_033_scalar_generalization_and_upstream_decision/{name}" for name in ARTIFACTS] + SCRIPTS:
        if path not in paths:
            paths.append(path)
    write_json(status_path, status)

    evidence_path = PROJECT / "EVIDENCE_LEDGER.csv"
    with evidence_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    question_key = "aggregate::iteration_33"
    if not any(row.get("iteration") == "33" and row.get("question_key") == question_key for row in rows):
        rows.append({
            "iteration": "33",
            "question_key": question_key,
            "hypothesis": "The strict typed serializer can recover runtime-grounded scalar values omitted by an entity-ID-only projection; transfer and upstream intervention still require independent evidence.",
            "supporting_trace": "39/39 accepted scalar provenance links verified against archived runtime artifacts; all 800 frozen predictions crosswalked and rescored with official per-answer types.",
            "confidence": "HIGH for retrospective reproduction; LOW for generalization",
            "outcome": "578/800 historic vs 614/800 supplementary typed mixed on identical outputs; 36 correct/3 incorrect conversions; no prospective cohort, COUNT producer, or supported upstream correction.",
            "source_artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_033_scalar_generalization_and_upstream_decision/ITER33_DECISION.md",
            "notes": "DEV_REPRESENTATION_RECOVERY only; PROSPECTIVE_EVALUATION_NOT_RUN; WebQSP labels opened; COUNT_PRODUCER_BLOCKED; UPSTREAM_INTERVENTION_NOT_READY; 0 API, 0 SPARQL, $0.",
        })
    with evidence_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    log_path = PROJECT / "ITERATION_LOG.md"
    log = log_path.read_text(encoding="utf-8")
    if "## Iteration 33 —" not in log:
        log += f"""

## Iteration 33 — scalar generalization and upstream research decision — {RECORDED}

- Decision: `DEV_REPRESENTATION_RECOVERY`; `PROSPECTIVE_EVALUATION_NOT_RUN`; `COUNT_PRODUCER_BLOCKED`; `UPSTREAM_INTERVENTION_NOT_READY`.
- Independently linked all 39 accepted scalar outputs to frozen runtime prompts, terminal responses, request events, and the 800-case snapshot. Separate gold scoring reproduced 578/800 historic entity-only versus 614/800 supplementary typed mixed, with 36 correct and 3 incorrect scalar conversions. This is a contract/representation result on identical outputs, not a method gain.
- WebQSP audit: 1,628 IDs remain by historical exposure, but only four pass the frozen count-intent trigger. Official labels were inspected; the test split is development-exposed and no manifest was created.
- Upstream audit reproduced the Iteration 30 miss partitions but found no uniquely supported gold-free correction from the retained trace fields. COUNT plan producer remains 0/68.
- Tests: 74 passed. API requests 0; SPARQL attempts 0; cost US$0. Historical cost snapshots were not reconciled by overwriting or summing them.
- Report: `outputs/experiments/fresh50_continuous_agent/RESULTS/iter_033_scalar_generalization_and_upstream_decision/ITER33_DECISION.md`.
"""
        log_path.write_text(log, encoding="utf-8")

    handoff_path = PROJECT / "CHATGPT_HANDOFF.md"
    handoff = handoff_path.read_text(encoding="utf-8")
    if "<!-- ITER33_HANDOFF_START -->" not in handoff:
        current = f"""<!-- ITER33_HANDOFF_START -->
# Current state — Iteration 33 `OFFLINE_COMPLETE`

**Decision:** `DEV_REPRESENTATION_RECOVERY`; `PROSPECTIVE_EVALUATION_NOT_RUN`; `COUNT_PRODUCER_BLOCKED`; `UPSTREAM_INTERVENTION_NOT_READY`.

On the same 800 frozen GrailQA dev predictions, Iteration 33 independently reproduced 578/800 Hits@1 and mean F1 0.685496 for the historic entity-only contract, and 614/800 and mean typed set F1 0.730496 for a supplementary typed mixed contract. The serializer accepted 39 existing scalar outputs; 36 were correct and 3 incorrect. This is representation recovery and a scoring-contract change, not new inference or a KGQA reasoning/count-execution gain. Iteration 29 remains unchanged.

The source/exposure audit found 1,628 WebQSP test IDs remaining by historical exposure, but only four pass the frozen count-intent trigger. Official labels were opened during the audit, so WebQSP is now development-exposed. No prospective manifest was created. COUNT remains 0/68 producer-ready. Archived first-hop evidence does not justify a safe non-oracle upstream correction.

Iteration 33: 74 tests passed; 0 paid API requests; 0 SPARQL attempts; US$0. Historical ledger snapshots remain unreconciled and unchanged. No commit or push was made.

**Next:** consolidate this bounded development result into the thesis. Reopen experiment work only with an untouched, sufficiently trigger-positive cohort or an independent gold-free plan/step-provenance source.

Primary report: `outputs/experiments/fresh50_continuous_agent/RESULTS/iter_033_scalar_generalization_and_upstream_decision/ITER33_DECISION.md`.

<!-- ITER33_HANDOFF_END -->

"""
        handoff_path.write_text(current + handoff, encoding="utf-8")

    print(json.dumps({
        "status": "LEDGERS_UPDATED",
        "recorded_at_utc": RECORDED,
        "cost_log_entries": len(logs),
        "status_artifact_paths": len(status["artifact_paths"]),
        "evidence_rows": len(rows),
        "iteration_log_appended": True,
        "current_first_handoff_added": True,
    }, indent=2))


if __name__ == "__main__":
    main()
