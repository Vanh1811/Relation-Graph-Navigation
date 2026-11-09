#!/usr/bin/env python3
"""Hash Iteration 32 inputs/outputs and verify frozen history without network access."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS"
OUT = RESULTS / "iter_032_count_producer_and_scalar_contract"
HANDOFF_ROOT = ROOT / "outputs/experiments/fresh50_continuous_agent"
BASELINE = RESULTS / "iter_031_gold_free_count_readiness/_ITER31_BASELINE_INTEGRITY.json"
ITER31_INTEGRITY = RESULTS / "iter_031_gold_free_count_readiness/SOURCE_AND_HISTORY_INTEGRITY.json"
MANIFEST = OUT / "SOURCE_HASH_MANIFEST.json"
INTEGRITY = OUT / "SOURCE_AND_HISTORY_INTEGRITY.json"

MUTABLE_HANDOFFS = {
    "outputs/experiments/fresh50_continuous_agent/CHATGPT_HANDOFF.md",
    "outputs/experiments/fresh50_continuous_agent/COST_LEDGER.json",
    "outputs/experiments/fresh50_continuous_agent/EVIDENCE_LEDGER.csv",
    "outputs/experiments/fresh50_continuous_agent/ITERATION_LOG.md",
    "outputs/experiments/fresh50_continuous_agent/STATUS.json",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    baseline = read_json(BASELINE)
    iter31 = read_json(ITER31_INTEGRITY)

    # Recheck all previously protected immutable sources against the Iteration 31
    # pre-write hashes. Mutable handoff ledgers are intentionally updated here.
    protected_expected = baseline["protected_sha256_before_iter31_writes"]
    protected_checks = {}
    for path, expected in protected_expected.items():
        if path in MUTABLE_HANDOFFS:
            continue
        target = ROOT / path
        protected_checks[path] = {
            "expected_sha256": expected,
            "actual_sha256": digest(target),
            "unchanged": digest(target) == expected,
        }

    iter31_artifact_checks = {
        path: {
            "expected_sha256": expected,
            "actual_sha256": digest(ROOT / path),
            "unchanged": digest(ROOT / path) == expected,
        }
        for path, expected in iter31["iteration31_artifact_hashes"].items()
    }
    iter31_source_checks = {
        path: {
            "expected_sha256": expected,
            "actual_sha256": digest(ROOT / path),
            "unchanged": digest(ROOT / path) == expected,
        }
        for path, expected in iter31["new_iteration31_source_hashes"].items()
    }

    run = RESULTS / "iter_028_terminal_boundary_repair/RUNS/iter29_terminal_candidate_selector_attempt3"
    prior_iter29_paths = {
        "cohort": RESULTS / "iter_028_terminal_boundary_repair/NEXT_INFERENCE_COHORT_ATTEMPT3_NON_GOLD.jsonl",
        "trace": run / "CASE_TRACES_NON_GOLD.jsonl",
        "score_case_csv": run / "PAIRED_EVAL_CASES_GOLD_ONLY.csv",
        "score_summary": run / "PAIRED_ANALYSIS_GOLD_ONLY.json",
        "scoring_gate": run / "SCORING_GATE_VERDICT.json",
    }
    expected_prior = iter31["prior_iter29_validation"]["hashes_expected"]
    prior_iter29_checks = {
        key: {
            "path": rel(path),
            "expected_sha256": expected_prior[key] if key in expected_prior else iter31["prior_iter29_validation"]["trace_sha256"],
            "actual_sha256": digest(path),
        }
        for key, path in prior_iter29_paths.items()
    }
    prior_iter29_checks["trace"]["expected_sha256"] = iter31["prior_iter29_validation"]["trace_sha256"]
    prior_iter29_checks["scoring_gate"]["expected_sha256"] = baseline["protected_sha256_before_iter31_writes"][rel(prior_iter29_paths["scoring_gate"])]
    for value in prior_iter29_checks.values():
        value["unchanged"] = value["actual_sha256"] == value["expected_sha256"]

    iter30_audit = RESULTS / "iter_030_results_first/COUNT_OPERATOR_AUDIT.csv"
    iter30_report = RESULTS / "iter_030_results_first/ITER30_RESEARCH_REPORT.md"

    sources = {
        "frozen_non_gold_runtime": {
            rel(run / "CASE_TRACES_NON_GOLD.jsonl"): digest(run / "CASE_TRACES_NON_GOLD.jsonl"),
            rel(run / "REQUEST_EVENTS.jsonl"): digest(run / "REQUEST_EVENTS.jsonl"),
            rel(run / "REQUEST_JOURNAL.jsonl"): digest(run / "REQUEST_JOURNAL.jsonl"),
            rel(RESULTS / "iter_031_gold_free_count_readiness/TARGET_TYPE_RESOLVER_AUDIT.csv"): digest(RESULTS / "iter_031_gold_free_count_readiness/TARGET_TYPE_RESOLVER_AUDIT.csv"),
            rel(RESULTS / "iter_031_gold_free_count_readiness/PATH_COMPLETENESS_AUDIT.csv"): digest(RESULTS / "iter_031_gold_free_count_readiness/PATH_COMPLETENESS_AUDIT.csv"),
            rel(iter30_audit): digest(iter30_audit),
        },
        "gold_only_scoring_inputs_not_used_at_runtime": {
            rel(run / "PAIRED_EVAL_CASES_GOLD_ONLY.csv"): digest(run / "PAIRED_EVAL_CASES_GOLD_ONLY.csv"),
            rel(iter30_audit): digest(iter30_audit),
        },
    }
    code_roots = [
        ROOT / "chain_of_relations/methods/iter_032_scalar_contract",
        ROOT / "scripts/diagnostics/iter_032_scalar_contract",
        ROOT / "tests/test_iter_032_scalar_contract.py",
    ]
    code_hashes: dict[str, str] = {}
    for item in code_roots:
        candidates = [item] if item.is_file() else sorted(item.rglob("*.py"))
        for path in candidates:
            code_hashes[rel(path)] = digest(path)

    output_hashes = {
        rel(path): digest(path)
        for path in sorted(OUT.iterdir())
        if path.is_file() and path.name not in {MANIFEST.name, INTEGRITY.name}
    }
    resource = read_json(OUT / "RESOURCE_ACCOUNTING.json")
    score = read_json(OUT / "DEVELOPMENT_SCORING_SUMMARY_GOLD_ONLY.json")
    producer = read_json(OUT / "REJECTION_REASON_COUNTS.json")

    manifest = {
        "iteration": 32,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "offline scalar source audit, frozen prediction serialization, retrospective gold-only scoring, and producer provenance audit",
        "runtime_gold_access": False,
        "runtime_sources": sources["frozen_non_gold_runtime"],
        "gold_only_scoring_sources": sources["gold_only_scoring_inputs_not_used_at_runtime"],
        "iteration32_code_sha256": dict(sorted(code_hashes.items())),
        "iteration32_artifact_sha256": output_hashes,
        "frozen_prediction_snapshot_sha256": score["prediction_snapshot_sha256"],
        "gold_only_scoring_source_sha256": score["gold_source_sha256"],
        "iteration30_gold_only_audit_sha256": score["count_operator_audit_gold_only_sha256"],
        "producer_audit_runtime_sources": producer["sources"],
        "resource_accounting": {
            "openai_api_requests": resource["openai_api_requests"],
            "physical_sparql_attempts": resource["physical_sparql_attempts"],
            "network_requests": resource["network_requests"],
            "measured_cost_usd": resource["measured_cost_usd"],
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    tracked_changed = subprocess.run(["git", "diff", "--name-only"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()

    cost = read_json(HANDOFF_ROOT / "COST_LEDGER.json")
    budget = cost["budget_remaining"]
    cumulative = {
        "cost_usd": budget["cumulative_llm_cost_usd"],
        "input_tokens": budget["cumulative_input_tokens"],
        "llm_calls": budget["cumulative_llm_calls"],
        "output_tokens": budget["cumulative_output_tokens"],
        "read_only_kg_queries": cost["read_only_kg_queries"],
    }
    baseline_cumulative = iter31["root_ledger_update_summary"]["cumulative_totals_preserved"]
    ledger_counts_unchanged = cumulative == baseline_cumulative

    all_history_checks = (
        all(row["unchanged"] for row in protected_checks.values())
        and all(row["unchanged"] for row in iter31_artifact_checks.values())
        and all(row["unchanged"] for row in iter31_source_checks.values())
        and all(row["unchanged"] for row in prior_iter29_checks.values())
    )
    integrity = {
        "iteration": 32,
        "primary_decision": "SCALAR_CONTRACT_RESULT_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "head_expected": baseline["git_head"],
            "head_actual": head,
            "head_unchanged": head == baseline["git_head"],
            "staged_paths": staged,
            "tracked_changed_paths": tracked_changed,
            "status_after": status,
            "commit_or_push_performed": False,
            "reset_clean_stash_performed": False,
        },
        "historical_iter29_artifacts_modified": not all(row["unchanged"] for row in prior_iter29_checks.values()),
        "historical_iter30_artifacts_modified": not all(protected_checks[path]["unchanged"] for path in protected_checks if "/iter_030_results_first/" in path),
        "iteration31_artifacts_unchanged": all(row["unchanged"] for row in iter31_artifact_checks.values()),
        "iteration31_sources_unchanged": all(row["unchanged"] for row in iter31_source_checks.values()),
        "preexisting_protected_files_all_unchanged": all(row["unchanged"] for row in protected_checks.values()),
        "preexisting_protected_files_unchanged": protected_checks,
        "iteration31_artifact_hash_checks": iter31_artifact_checks,
        "iteration31_source_hash_checks": iter31_source_checks,
        "prior_iter29_validation": {
            "checks": prior_iter29_checks,
            "all_unchanged": all(row["unchanged"] for row in prior_iter29_checks.values()),
            "authoritative_baseline_hits_at_1": 578,
            "typed_contract_hits_at_1": score["typed_mixed_contract"]["hits_at_1"],
            "typed_score_is_separate_development_contract": True,
        },
        "root_ledgers_updated_by_design": sorted(MUTABLE_HANDOFFS),
        "cumulative_ledger_totals_before_and_after_iter32": {
            "baseline_iter31": baseline_cumulative,
            "current": cumulative,
            "unchanged": ledger_counts_unchanged,
        },
        "iteration32_artifact_manifest_sha256": digest(MANIFEST),
        "iteration32_prediction_snapshot_sha256": score["prediction_snapshot_sha256"],
        "iteration32_gold_source_sha256_scoring_only": score["gold_source_sha256"],
        "iteration32_counts": {
            "frozen_cases": score["n"],
            "typed_scalar_acceptances": score["converted_scalar_cases"],
            "typed_scalar_correct_scoring_only": score["count_operator_subset_scoring_only"]["typed_hits_at_1"],
            "typed_scalar_incorrect_scoring_only": score["count_operator_subset_scoring_only"]["runtime_supported_scalar_conversions"] - score["count_operator_subset_scoring_only"]["typed_hits_at_1"],
            "real_gold_free_count_producer_cases": producer["real_gold_free_producer_cases"],
        },
        "network_calls": {
            "openai_api_requests": resource["openai_api_requests"],
            "paid_model_calls": resource["paid_model_calls"],
            "physical_sparql_attempts": resource["physical_sparql_attempts"],
            "network_requests": resource["network_requests"],
            "retries": resource["retries"],
            "measured_cost_usd": resource["measured_cost_usd"],
        },
        "overall_history_integrity_pass": all_history_checks and head == baseline["git_head"] and not staged and ledger_counts_unchanged,
        "notes": [
            "Iteration 32's five root handoff ledgers were intentionally updated; immutable Iteration 29/30/31 sources and artifacts were hash-checked.",
            "The current worktree included user/previous-iteration changes before Iteration 32. They were neither staged nor reverted.",
            "Gold-only sources were read only by the separate development scoring step, after the non-gold prediction snapshot was frozen.",
        ],
    }
    INTEGRITY.write_text(json.dumps(integrity, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    failures = []
    if not all_history_checks:
        failures.append("one or more frozen Iteration 29/30/31 source/artifact hashes differ")
    if head != baseline["git_head"]:
        failures.append("git HEAD changed")
    if staged:
        failures.append("staged paths exist")
    if not ledger_counts_unchanged:
        failures.append("cumulative accounting totals changed")
    if resource["openai_api_requests"] or resource["physical_sparql_attempts"] or resource["network_requests"]:
        failures.append("resource accounting reports an external call")
    if failures:
        raise SystemExit("Integrity checks failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
