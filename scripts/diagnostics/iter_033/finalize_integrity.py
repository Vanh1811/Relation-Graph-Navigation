#!/usr/bin/env python3
"""Verify immutable Iterations 29–32 and record Iteration 33 artifact hashes."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROJECT = ROOT / "outputs/experiments/fresh50_continuous_agent"
RESULTS = PROJECT / "RESULTS"
ITER32 = RESULTS / "iter_032_count_producer_and_scalar_contract"
ITER31 = RESULTS / "iter_031_gold_free_count_readiness"
OUTDIR = RESULTS / "iter_033_scalar_generalization_and_upstream_decision"
OUT = OUTDIR / "SOURCE_AND_HISTORY_INTEGRITY.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_mapping(mapping: dict[str, str], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative, expected in mapping.items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        result[relative] = {"expected_sha256": expected, "actual_sha256": actual, "unchanged": actual == expected}
        if actual != expected:
            raise SystemExit(f"{label} hash mismatch: {relative}")
    return result


def command(args: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=10, check=False)
        return {"argv": args, "exit_code": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except Exception as exc:  # Git status can be unavailable if the host cannot map its index.
        return {"argv": args, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    old_integrity_path = ITER32 / "SOURCE_AND_HISTORY_INTEGRITY.json"
    old_integrity = json.loads(old_integrity_path.read_text(encoding="utf-8"))
    manifest_path = ITER32 / "SOURCE_HASH_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    artifact_checks = verify_manifest_mapping(manifest["iteration32_artifact_sha256"], "Iteration 32 artifact")
    code_checks = verify_manifest_mapping(manifest["iteration32_code_sha256"], "Iteration 32 source")
    gold_checks = verify_manifest_mapping(manifest["gold_only_scoring_sources"], "gold scoring source")
    producer_sources = manifest["producer_audit_runtime_sources"]
    producer_path_map = {
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_031_gold_free_count_readiness/PATH_COMPLETENESS_AUDIT.csv": producer_sources["path_completeness_audit_sha256"],
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_032_count_producer_and_scalar_contract/PREDICTION_SNAPSHOT_NON_GOLD.jsonl": producer_sources["runtime_snapshot_sha256"],
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_031_gold_free_count_readiness/TARGET_TYPE_RESOLVER_AUDIT.csv": producer_sources["target_type_audit_sha256"],
    }
    producer_checks = verify_manifest_mapping(producer_path_map, "Iteration 32 producer source")
    runtime_checks = verify_manifest_mapping(manifest["runtime_sources"], "archived runtime source")

    prior_checks: dict[str, dict[str, dict[str, Any]]] = {}
    for section, filename in [
        ("iteration31_artifact_hash_checks", "artifact"),
        ("iteration31_source_hash_checks", "source"),
        ("preexisting_protected_files_unchanged", "preexisting protected file"),
    ]:
        source = old_integrity.get(section, {})
        mapping: dict[str, str] = {}
        for relative, record in source.items():
            expected = record.get("expected_sha256") or record.get("expected")
            if expected:
                mapping[relative] = expected
        prior_checks[filename] = verify_manifest_mapping(mapping, f"Iteration 32-recorded {filename}")

    head = command(["git", "rev-parse", "HEAD"])
    git_status = command(["git", "status", "--short", "--untracked-files=all"])
    expected_head = old_integrity.get("git", {}).get("head_expected")
    actual_head = head.get("stdout") if head.get("exit_code") == 0 else None

    new_artifact_paths = sorted(
        path for path in OUTDIR.rglob("*")
        if path.is_file() and path != OUT
    )
    new_artifacts = {str(path.relative_to(ROOT)): sha256(path) for path in new_artifact_paths}
    central_paths = [
        PROJECT / "STATUS.json",
        PROJECT / "COST_LEDGER.json",
        PROJECT / "EVIDENCE_LEDGER.csv",
        PROJECT / "ITERATION_LOG.md",
        PROJECT / "CHATGPT_HANDOFF.md",
    ]
    central_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in central_paths}
    script_paths = [
        ROOT / "scripts/diagnostics/iter_033/typed_scoring.py",
        ROOT / "scripts/diagnostics/iter_033/audit_runtime.py",
        ROOT / "scripts/diagnostics/iter_033/score_representation.py",
        ROOT / "scripts/diagnostics/iter_033/audit_webqsp_cohort.py",
        ROOT / "scripts/diagnostics/iter_033/audit_upstream_evidence.py",
        ROOT / "scripts/diagnostics/iter_033/update_ledgers.py",
        ROOT / "scripts/diagnostics/iter_033/finalize_integrity.py",
        ROOT / "tests/test_iter_033_typed_representation.py",
    ]
    script_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in script_paths}
    checked_sections = [artifact_checks, code_checks, gold_checks, producer_checks, runtime_checks, *prior_checks.values()]
    source_files_checked = sum(len(section) for section in checked_sections)
    source_files_unchanged = all(record["unchanged"] for section in checked_sections for record in section.values())
    out = {
        "iteration": 33,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "PASS_HISTORICAL_HASHES_AND_ITER33_INTEGRITY" if source_files_unchanged and actual_head == expected_head else "FAIL",
        "repo": {
            "head_expected": expected_head,
            "head_actual": actual_head,
            "head_unchanged": actual_head == expected_head,
            "status_command": git_status,
            "commit_or_push_performed": False,
            "reset_clean_stash_performed": False,
        },
        "preexisting_worktree_snapshot_from_iteration32": old_integrity.get("git", {}).get("status_after", []),
        "preexisting_staged_paths_from_iteration32": old_integrity.get("git", {}).get("staged_paths", []),
        "preexisting_protected_files_unchanged": all(record["unchanged"] for record in prior_checks["preexisting protected file"].values()),
        "preexisting_protected_file_hash_checks": prior_checks["preexisting protected file"],
        "iteration31_recorded_artifact_checks": prior_checks["artifact"],
        "iteration31_recorded_source_checks": prior_checks["source"],
        "iteration32_frozen_artifact_checks": artifact_checks,
        "iteration32_frozen_code_checks": code_checks,
        "frozen_gold_source_checks": gold_checks,
        "iteration32_producer_runtime_source_checks": producer_checks,
        "archived_iter29_runtime_source_checks": runtime_checks,
        "source_files_checked": source_files_checked,
        "all_checked_historical_sources_unchanged": source_files_unchanged,
        "iteration33_new_artifact_hashes_excluding_this_file": new_artifacts,
        "iteration33_diagnostic_script_and_test_hashes": script_hashes,
        "central_ledger_hashes_after_iteration33": central_hashes,
        "iteration33_transport_accounting": {"paid_model_api_requests": 0, "sparql_attempts": 0, "spend_usd": 0.0},
        "scope_note": "The Iteration 32 Git status snapshot is the pre-existing worktree baseline. If the current Git status command reports an index mapping error, HEAD and immutable source hashes are still checked; no Git write/mutation command was run.",
    }
    if out["status"] != "PASS_HISTORICAL_HASHES_AND_ITER33_INTEGRITY":
        raise SystemExit("Iteration 33 source integrity gate failed")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "historical_sources_checked": source_files_checked,
        "all_historical_sources_unchanged": source_files_unchanged,
        "head_unchanged": actual_head == expected_head,
        "git_status": git_status,
        "new_artifacts": len(new_artifacts),
        "output": str(OUT.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
