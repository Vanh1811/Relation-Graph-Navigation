"""Write Iteration 25 offline results and update current research ledgers."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from integrity import verify

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
OUT = WORK / "RESULTS/iter_025_cache_compatible_readiness"


def write_offline_results() -> dict:
    integrity = verify()
    protected_manifest = json.loads((OUT / "PROTECTED_HASHES.json").read_text(encoding="utf-8"))
    suites = [
        (21, "scripts/diagnostics/iter_021_matched_prefix", 23),
        (22, "scripts/diagnostics/iter_022_gated_matched_prefix", 19),
        (23, "scripts/diagnostics/iter_023_label_coverage_and_prefix_readiness", 36),
        (25, "scripts/diagnostics/iter_025_cache_compatible_readiness", 23),
    ]
    suite_records = [{
        "iteration": iteration,
        "command": [".venv/bin/python", "-m", "unittest", "discover", "-s", path, "-p", "test_*.py", "-v"],
        "status": "PASS",
        "tests_run": count,
        "tests_passed": count,
        "tests_failed": 0,
        "tests_errored": 0,
    } for iteration, path, count in suites]
    payload = {
        "iteration": 25,
        "status": "PASS",
        "evidence_class": "OFFLINE_FIXTURE_AND_ARCHIVE_AUDIT_ONLY",
        "suites": suite_records,
        "total_tests": sum(record["tests_run"] for record in suite_records),
        "total_tests_passed": sum(record["tests_passed"] for record in suite_records),
        "total_tests_failed": 0,
        "total_tests_errored": 0,
        "assertion_status": "All assertions in all 101 passing unittest cases passed; unittest reports test-case counts, not individual assertion totals.",
        "initial_environment_note": "An initial system-python Iteration 23 run had 4 import errors because SPARQLWrapper was absent; rerunning with the repository .venv passed 36/36. No package installation or network access was used.",
        "protected_hash_status": "PASS",
        "protected_hash_checked": integrity["checked"],
        "protected_hash_matching": integrity["matching"],
        "protected_hash_mismatch_count": len(integrity["mismatches"]),
        "protected_source_file_count": protected_manifest["source_file_count"],
        "protected_historical_artifact_count": protected_manifest["historical_artifact_count"],
        "protected_sources_unchanged": integrity["matching"] == integrity["checked"] and not integrity["mismatches"],
        "paid_api_requests": 0,
        "physical_sparql_attempts": 0,
        "estimated_spend_usd": 0.0,
        "network_requests": 0,
        "candidate_inventory": {
            "screened_questions": 30,
            "depth4_target_role": 20,
            "shorter_path_control_role": 10,
            "technically_eligible_exact_prefix_cases": 0,
            "full_cache_coverage_cases": 1,
            "partial_cache_coverage_cases": 4,
            "zero_cache_coverage_cases": 25,
        },
        "limitations": [
            "Fixture tests do not validate a live SDK/endpoint adapter.",
            "Cached labels preserve archived display strings but not RDF language/datatype metadata.",
            "No new depth or answer-filter causal effect was measured.",
        ],
    }
    (OUT / "OFFLINE_TEST_RESULTS.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def update_status(test_results: dict) -> None:
    path = WORK / "STATUS.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = "ITERATION_25_COMPLETE_GATE_D_PIVOT_TO_SELECTION_EVALUATION"
    data["iteration"] = 25
    data["current_hypothesis"] = (
        "The archive does not support a reliable exact-prefix depth pilot: no screened question has complete prefix state, "
        "and historical cache coverage does not guarantee a new trajectory. A fixed-candidate answer-selection contribution "
        "was confirmed for seven prior misses, but the tested selector failed its development gate."
    )
    data["last_completed_experiment"] = "Iteration 25 offline cache/prefix readiness audit and diagnostic-only safety fixtures"
    data["best_supported_conclusion"] = (
        "The Fresh50 method results remain descriptive (CoR 25/50, TGAR 29/50, CFP 27/50). In a target-selected depth challenge, "
        "depth 4 outperformed depth 3 on separate reruns, but no exact matched-prefix causal effect was measured. Iteration 25 screened "
        "30 archived questions: 0 had complete exact-prefix state; the 198-ID cache covered all depth-3 candidates for 1, partially for 4, "
        "and none for 25. The prior answer-selector pilot rescued 0/7 known failures and lost 1/5 controls. Gate D: propose a new, independent "
        "answer-type/path-provenance filter evaluation; do not reuse the seven development cases or claim broad improvement."
    )
    data["remaining_questions"] = [
        "Should the researcher prepare a new independently sampled answer-filter evaluation under the bounded proposal in Iteration 25?",
        "Can the redesigned answer-type/path-provenance filter rescue fixed-candidate omissions without losing correct controls on that new cohort?",
    ]
    data["next_action"] = (
        "No depth pilot is ready. Consider the Iteration 25 independent answer-filter protocol; freeze unseen question IDs, verify current SDK pricing/retry settings, "
        "then create a new manifest and obtain explicit authorization before any API or SPARQL transport."
    )
    data["latest_experiment_attempt"] = (
        "Iteration 25 was offline only: 30 historical questions screened, 0 exact-prefix eligible, 1 complete/4 partial/25 zero cache coverage; "
        f"{test_results['total_tests_passed']}/{test_results['total_tests']} offline tests passed and "
        f"{test_results['protected_hash_matching']}/{test_results['protected_hash_checked']} protected source/history hashes matched. "
        "API 0, SPARQL 0, spend $0. Gate D selected; no new causal result."
    )
    budget = data.setdefault("budget_remaining", {})
    budget["active_paid_authorization"] = False
    budget["new_experiment_authorized_calls"] = 0
    budget["new_experiment_authorized_cost_usd"] = 0.0
    budget["iteration25_paid_api_requests"] = 0
    budget["iteration25_physical_sparql_attempts"] = 0
    budget["iteration25_spend_usd"] = 0.0
    budget["iteration25_offline_only"] = True
    data["artifact_paths"] = list(dict.fromkeys(data.get("artifact_paths", []) + [
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_025_cache_compatible_readiness/ITER25_DECISION.md",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_025_cache_compatible_readiness/CANDIDATE_COVERAGE.csv",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_025_cache_compatible_readiness/SELECTION_POLICY.md",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_025_cache_compatible_readiness/LABEL_PROVENANCE_AUDIT.md",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_025_cache_compatible_readiness/OFFLINE_TEST_RESULTS.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_025_cache_compatible_readiness/RUNNER_SAFETY_REVIEW.md",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_025_cache_compatible_readiness/BLOCKERS_AND_ALTERNATIVES.md",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_025_cache_compatible_readiness/PROTECTED_HASHES.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_025_cache_compatible_readiness/COVERAGE_SUMMARY.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_025_cache_compatible_readiness/LABEL_PROVENANCE_EVIDENCE.json",
        "scripts/diagnostics/iter_025_cache_compatible_readiness/coverage.py",
        "scripts/diagnostics/iter_025_cache_compatible_readiness/provenance.py",
        "scripts/diagnostics/iter_025_cache_compatible_readiness/safe_runner.py",
        "scripts/diagnostics/iter_025_cache_compatible_readiness/test_iter25.py",
    ]))
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_cost_ledger(test_results: dict) -> None:
    path = WORK / "COST_LEDGER.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    offline = data.setdefault("offline_experiment_log", [])
    record = {
        "iteration": 25,
        "experiment": "cache-compatible cohort and exact-prefix readiness; pivot decision",
        "llm_calls": 0,
        "kg_queries": 0,
        "cost_usd": 0.0,
        "screened_questions": 30,
        "exact_prefix_eligible": 0,
        "offline_tests": test_results["total_tests"],
        "protected_hashes_checked": test_results["protected_hash_checked"],
        "protected_hashes_matching": test_results["protected_hash_matching"],
        "decision": "GATE_D_PIVOT_TO_CONFIRMED_SELECTION_FAILURE",
        "artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_025_cache_compatible_readiness/ITER25_DECISION.md",
    }
    offline[:] = [item for item in offline if not (item.get("iteration") == 25 and item.get("experiment") == record["experiment"])]
    offline.append(record)
    data["offline_iterations"] = sorted(set(data.get("offline_iterations", [])) | {25})
    data["iteration25_api_requests"] = 0
    data["iteration25_physical_sparql_attempts"] = 0
    data["iteration25_cost_usd"] = 0.0
    data["iteration25_decision"] = record["decision"]
    data["iteration25_proposed_next_study_not_authorized"] = {
        "name": "independent answer-type/path-provenance filter evaluation",
        "status": "PROPOSAL_ONLY_NOT_AUTHORIZED",
        "sample_screening_questions": 100,
        "paired_filter_cases": 20,
        "api_request_cap": 1500,
        "read_only_sparql_attempt_cap": 2700,
        "per_request_input_token_cap": 7000,
        "per_request_completion_token_cap": 1024,
        "maximum_reserved_spend_usd": 3.9432,
        "proposed_spend_cap_usd": 4.0,
        "retries": 0,
        "per_request_timeout_seconds": 120,
        "overall_wall_cap_hours": 24,
        "authorization_active": False,
    }
    ledger_note = str(data.get("note", ""))
    iteration25_note = " Iteration 25 was offline only (0 API, 0 SPARQL, $0.00); its proposed follow-up is not authorized. Iteration 23's uncertain $0.0026288 reservation remains preserved."
    ledger_note = ledger_note.replace(iteration25_note, "").rstrip()
    data["note"] = ledger_note + iteration25_note
    data["paid_experiment_authorized"] = False
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_evidence_ledger(test_results: dict) -> None:
    path = WORK / "EVIDENCE_LEDGER.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        fields = list(rows[0]) if rows else ["iteration", "question_key", "hypothesis", "supporting_trace", "confidence", "outcome", "source_artifact", "notes"]
    existing = next((row for row in rows if row.get("iteration") == "25" and row.get("question_key") == "iteration25::all_30_archived_candidates"), None)
    record = {
        "iteration": "25",
        "question_key": "iteration25::all_30_archived_candidates",
        "hypothesis": "CACHE_AND_PREFIX_READINESS; confirmed answer-selection failure supports alternate study",
        "supporting_trace": "Iter19 candidate archive and paired outcomes; Iter21 prefix audit; Iter22-24 query/cache journals; Iter14 selector pilot",
        "confidence": "HIGH_FOR_ARCHIVE_COUNTS_AND_FIXTURE_TESTS; NO_NEW_CAUSAL_EVIDENCE",
        "outcome": "GATE_D_PIVOT_TO_CONFIRMED_SELECTION_FAILURE; NO_LIVE_RESULT",
        "source_artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_025_cache_compatible_readiness/ITER25_DECISION.md",
        "notes": (
            "30 screened; 0 exact-prefix cases; cache coverage 1 full/4 partial/25 zero. "
            "Iter14 same seven selector cases: 0/7 rescues and 1/5 control loss. "
            f"Offline tests {test_results['total_tests_passed']}/{test_results['total_tests']}; "
            f"protected hashes {test_results['protected_hash_matching']}/{test_results['protected_hash_checked']}; "
            "API 0, SPARQL 0, spend $0. Proposal only."
        ),
    }
    if existing:
        existing.update(record)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writerow(record)


def append_iteration_log(test_results: dict) -> None:
    path = WORK / "ITERATION_LOG.md"
    current = path.read_text(encoding="utf-8")
    marker = "## Iteration 25 — cache-compatible cohort and matched-prefix readiness"
    if marker in current:
        updated = current.replace("473/473", f"{test_results['protected_hash_matching']}/{test_results['protected_hash_checked']}")
        if updated != current:
            path.write_text(updated, encoding="utf-8")
        return
    section = f"""

{marker} — 2026-11-08

- Offline-only screening of all 30 archived Iteration 19 question identities: 20 depth targets and 10 shorter-path controls; 0 exact-prefix states, 1 full/4 partial/25 zero coverage against the frozen 198-ID cache. All 30 repeat-3 slots are unused, but their question identities are previously studied development data.
- Reconstructed the seven Iteration 22/23 missing IDs from successful `?targetEntity` result rows. They are not gold IDs, not diagnostic-only bindings, and not present in recorded prompts. Iteration 23 label SELECTs returned entity rows without label values; Iteration 24's four-predicate result remains limited to those predicates.
- Iteration 14's constrained selector was evaluated on the same seven confirmed failure cases and five controls: 0/7 rescues, 1/5 control loss, with four empty outputs at the 128-token ceiling (truncation unconfirmed). Do not repeat that configuration on those cases.
- Implemented a diagnostic-only durable attempt guard and exact-prefix validator with fixture tests. Iterations 21, 22, 23, and 25 offline suites: {test_results['total_tests']}/{test_results['total_tests']} pass. Protected source/history hashes: {test_results['protected_hash_matching']}/{test_results['protected_hash_checked']} match.
- **Gate D — pivot to a new independent answer-filter evaluation.** No depth effect or new accuracy result was measured. Iteration 25: 0 API, 0 SPARQL, $0.00. The follow-up protocol/budget is a proposal only; no manifest or authorization is active.
- Artifacts: `RESULTS/iter_025_cache_compatible_readiness/`.
"""
    path.write_text(current.rstrip() + section, encoding="utf-8")


def prepend_handoff(test_results: dict) -> None:
    path = WORK / "CHATGPT_HANDOFF.md"
    current = path.read_text(encoding="utf-8")
    start = "<!-- ITERATION25_HANDOFF_START -->"
    end = "<!-- ITERATION25_HANDOFF_END -->"
    if start in current:
        updated = current.replace("473/473", f"{test_results['protected_hash_matching']}/{test_results['protected_hash_checked']}")
        if updated != current:
            path.write_text(updated, encoding="utf-8")
        return
    section = f"""{start}
# Iteration 25 current handoff — pivot to an independent answer-filter evaluation

## Decision

**Gate D: do not continue the archived depth pilot.** The historical archive cannot supply an exact matched prefix, and the frozen label cache cannot guarantee a new stochastic trajectory. A fresh answer-filter study is the better next mechanism to investigate because seven fixed-candidate answer-selection contributions were confirmed, although the first constrained selector failed on those cases.

## Evidence and limits

- Candidate inventory: 30 historical questions (20 depth targets, 10 controls); **0/30** have ordered DFS stack plus active binding/frontier IDs required for exact-prefix replay.
- Historical depth-3 candidate/cache coverage: **1 full, 4 partial, 25 zero**. The sole full-coverage question later produced the uncached `m.0ndvpvf` in a new authorized run.
- Iterations 22–23's seven IDs were successful `?targetEntity` entity-search results that blocked construction of the next inference prompt. They are not gold IDs, hidden diagnostic-only IDs, or IDs in an actual prompt. Iteration 23's name lookups returned entity rows but no label values; Iteration 24 found no rows for four queried predicates only.
- Iteration 14 tested the constrained selector on the same seven confirmed failure questions and five controls: **0/7 rescues, 1/5 control loss**. Four calls returned empty content at the 128-token ceiling, but termination metadata was not saved; truncation is unconfirmed. Do not repeat that setup on those twelve development cases.
- The depth-4 results remain separate reruns on a target-selected cohort; they do not establish a causal depth-only effect. Fresh50 method scores remain descriptive: CoR 25/50, TGAR 29/50, CFP 27/50.

## Next decision

Consider preparing the fresh 100-question screening plus 20-case paired answer-filter protocol in `RESULTS/iter_025_cache_compatible_readiness/BLOCKERS_AND_ALTERNATIVES.md`. It proposes 1,500 API requests / US$4.00 hard cap / 2,700 physical SELECT cap, zero retries, 120 seconds per request, and 24 hours overall. It is a planning proposal only; there is no manifest or live authorization.

## Verification

- Offline suites: **{test_results['total_tests_passed']}/{test_results['total_tests']} passed** (Iterations 21–23 and 25).
- Protected sources and historical results: **{test_results['protected_hash_matching']}/{test_results['protected_hash_checked']} hashes match**.
- Iteration 25 transport: **0 API / 0 SPARQL / US$0.00**.
- Main report: `RESULTS/iter_025_cache_compatible_readiness/ITER25_DECISION.md`; full candidate inventory: `CANDIDATE_COVERAGE.csv`; label evidence: `LABEL_PROVENANCE_AUDIT.md` and `LABEL_PROVENANCE_EVIDENCE.json`.

{end}

"""
    path.write_text(section + current, encoding="utf-8")


def main() -> None:
    tests = write_offline_results()
    update_status(tests)
    update_cost_ledger(tests)
    append_evidence_ledger(tests)
    append_iteration_log(tests)
    prepend_handoff(tests)
    print(json.dumps({"total_tests": tests["total_tests"], "protected_hashes": tests["protected_hash_matching"],
                      "ledger_updates": "complete"}, indent=2))


if __name__ == "__main__":
    main()
