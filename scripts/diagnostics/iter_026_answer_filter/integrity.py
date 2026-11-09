"""Capture/verify Iteration 26 inputs and its diagnostic-only source files."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness"
PROTECTED = OUT / "PROTECTED_HASHES.json"
I25_DIR = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_025_cache_compatible_readiness"
I25_PROTECTED = I25_DIR / "PROTECTED_HASHES.json"

HISTORICAL_PATHS = [
    "Codex_Iteration26_Independent_Answer_Selector_Readiness.md",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_025_cache_compatible_readiness/ITER25_DECISION.md",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_025_cache_compatible_readiness/BLOCKERS_AND_ALTERNATIVES.md",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_025_cache_compatible_readiness/RUNNER_SAFETY_REVIEW.md",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_025_cache_compatible_readiness/PROTECTED_HASHES.json",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_013_selector_pilot_preparation/requests_non_gold.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_013_selector_pilot_preparation/archived_filter_reference_non_gold.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_013_selector_pilot_preparation/scoring_gold_only.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_014_selector_pilot/call_trace.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_014_selector_pilot/selector_outputs_non_gold.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_014_selector_pilot/scored_results_gold_only.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_014_selector_pilot/case_results.csv",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_014_selector_pilot/run_metadata.json",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_014_selector_pilot/results.json",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_014_selector_pilot/report.md",
    "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256/manifest_frozen.json",
    "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256/run_config.json",
    "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256/runs.jsonl",
    "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256/evaluation_gold.jsonl",
    "outputs/experiments/phase3a_1/safety_cap_sensitivity/frozen_future_eval_manifest.json",
    "outputs/experiments/phase3a/branch_necessity_benchmark/strong_branch_benchmark.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_000_baseline/source_artifacts/outputs/experiments/fresh50_autonomous_causal_diagnosis/answer_selection_counterfactuals.csv",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_000_baseline/source_artifacts/outputs/experiments/fresh50_autonomous_causal_diagnosis/executive_decision.md",
    "data/processed/cwq_train_analysis.jsonl",
    "data/processed/cwq_test_analysis.jsonl",
    "data/processed/grailqa_dev_analysis.jsonl",
    "data/processed/grailqa_train_analysis.jsonl",
    "chain_of_relations/methods/adaptive_v1_1_3/agent.py",
    "chain_of_relations/methods/adaptive_v1_1_3/router.py",
    "chain_of_relations/methods/cor/agent.py",
    "chain_of_relations/methods/cor/prompt.yml",
    "chain_of_relations/methods/cor/tools/filter.py",
    "chain_of_relations/tools/entity_search.py",
    "chain_of_relations/tools/relation_search.py",
    "chain_of_relations/tools/reasoning.py",
    "chain_of_relations/tools/relation_prune.py",
    "chain_of_relations/tools/generate_directly.py",
    "chain_of_relations/methods/branch_v1/agent.py",
    "chain_of_relations/llm_api.py",
    "chain_of_relations/kg_backend/freebase_backend.py",
    "chain_of_relations/kg_backend/freebase/db_func.py",
    "chain_of_relations/kg_backend/freebase/query_template.yml",
    "chain_of_relations/kg_backend/base.py",
    "chain_of_relations/schema.py",
    "scripts/run_fresh50_strong_branch_three_way.py",
    "scripts/run_phase3c_frozen_generalization.py",
    "scripts/run_cor_baseline_experiment.py",
    "scripts/diagnostics/prepare_fresh50_selector_pilot.py",
    "scripts/diagnostics/run_fresh50_selector_pilot.py",
    "scripts/diagnostics/transport_safety_repair/hard_deadline.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _i25_verify() -> dict[str, Any]:
    payload = json.loads(I25_PROTECTED.read_text(encoding="utf-8"))
    rows = []
    for item in payload["files"]:
        path = ROOT / item["path"]
        actual = sha256_file(path) if path.is_file() else None
        rows.append({"path": item["path"], "matches": actual == item["sha256"]})
    return {
        "checked": len(rows), "matching": sum(row["matches"] for row in rows),
        "mismatches": [row["path"] for row in rows if not row["matches"]],
    }


def capture() -> dict[str, Any]:
    previous_protected = json.loads(PROTECTED.read_text(encoding="utf-8")) if PROTECTED.is_file() else {}
    missing = [relative for relative in HISTORICAL_PATHS if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"missing protected inputs: {missing}")
    inherited = _i25_verify()
    if inherited["mismatches"]:
        raise RuntimeError(f"Iteration 25 protected hash mismatch: {inherited['mismatches'][:5]}")
    worktree_start_path = OUT / "ITER26_WORKTREE_START.json"
    start_snapshot = json.loads(worktree_start_path.read_text(encoding="utf-8"))
    paths = [{
        "path": relative,
        "sha256": sha256_file(ROOT / relative),
        "bytes": (ROOT / relative).stat().st_size,
    } for relative in HISTORICAL_PATHS]
    diagnostic_files = sorted((ROOT / "scripts/diagnostics/iter_026_answer_filter").glob("*.py"))
    diagnostics = [{
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    } for path in diagnostic_files]
    status = subprocess.run(["git", "status", "--porcelain=v1"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    frozen_artifacts = sorted(
        path for path in OUT.rglob("*")
        if path.is_file() and path != PROTECTED
    )
    artifacts = [{
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    } for path in frozen_artifacts]
    payload = {
        "iteration": 26,
        "status": "OFFLINE_INPUT_AND_CODE_HASHES_CAPTURED; NO_TRANSPORT",
        "repository_head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "iteration_start_worktree_snapshot": start_snapshot,
        "current_worktree_status_at_capture": status,
        "iteration25_inherited_protected_hash_verification": inherited,
        "protected_historical_and_source_file_count": len(paths),
        "protected_historical_and_source_files": paths,
        "iteration26_diagnostic_source_file_count": len(diagnostics),
        "iteration26_diagnostic_source_files": diagnostics,
        "iteration26_artifact_file_count": len(artifacts),
        "iteration26_artifacts": artifacts,
        "transport_counts": {"api_requests": 0, "physical_sparql_attempts": 0, "spend_usd": 0.0},
        "attempt3_preparation_transport_counts": {
            "api_requests": 0, "physical_sparql_attempts": 0, "spend_usd": 0.0,
        },
    }
    # Keep the immutable accounting for already-consumed attempts when the
    # protected file is refreshed after additional offline preparation.
    for key in (
        "attempt1_live_accounting",
        "attempt2_live_accounting",
        "iteration26_total_live_accounting",
        "attempt2_preparation_transport_counts",
        "preparation_transport_counts",
        "capture_operation_transport_counts",
        "branch",
    ):
        if key in previous_protected:
            payload[key] = previous_protected[key]
    PROTECTED.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify() -> dict[str, Any]:
    payload = json.loads(PROTECTED.read_text(encoding="utf-8"))
    inherited = _i25_verify()
    old_checks = []
    for item in payload["protected_historical_and_source_files"]:
        path = ROOT / item["path"]
        actual = sha256_file(path) if path.is_file() else None
        old_checks.append(actual == item["sha256"])
    code_checks = []
    for item in payload["iteration26_diagnostic_source_files"]:
        path = ROOT / item["path"]
        actual = sha256_file(path) if path.is_file() else None
        code_checks.append(actual == item["sha256"])
    artifact_checks = []
    for item in payload.get("iteration26_artifacts", []):
        path = ROOT / item["path"]
        actual = sha256_file(path) if path.is_file() else None
        artifact_checks.append(actual == item["sha256"])
    return {
        "iteration25_checked": inherited["checked"],
        "iteration25_matching": inherited["matching"],
        "iteration25_mismatches": inherited["mismatches"],
        "iteration26_protected_inputs_checked": len(old_checks),
        "iteration26_protected_inputs_matching": sum(old_checks),
        "iteration26_code_checked": len(code_checks),
        "iteration26_code_matching": sum(code_checks),
        "iteration26_artifacts_checked": len(artifact_checks),
        "iteration26_artifacts_matching": sum(artifact_checks),
        "mismatches": (len(old_checks) - sum(old_checks)) + (len(code_checks) - sum(code_checks)) + (len(artifact_checks) - sum(artifact_checks)) + len(inherited["mismatches"]),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("capture", "verify"))
    args = parser.parse_args()
    result = capture() if args.action == "capture" else verify()
    if args.action == "capture":
        result = {key: value for key, value in result.items() if key not in {"protected_historical_and_source_files", "iteration26_diagnostic_source_files", "iteration26_artifacts", "current_worktree_status_at_capture"}}
    print(json.dumps(result, indent=2))
