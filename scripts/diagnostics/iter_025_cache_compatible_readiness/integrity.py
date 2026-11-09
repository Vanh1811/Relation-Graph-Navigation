"""Capture and compare protected production/history hashes for Iteration 25."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[3]
WORK = REPO / "outputs/experiments/fresh50_continuous_agent"
OUT = WORK / "RESULTS/iter_025_cache_compatible_readiness"
PROTECTED_FILE = OUT / "PROTECTED_HASHES.json"

EVIDENCE = [
    "Codex_Iteration25_Cache_Compatible_Matched_Prefix_Readiness.md",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_019_depth4_holdout_preparation/control_continuation/paired_runs_cumulative.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_019_depth4_holdout_preparation/control_continuation/paired_case_results_complete.csv",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_022_gated_matched_prefix/DECISION.md",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_022_gated_matched_prefix/PILOT_INTEGRITY.json",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_022_gated_matched_prefix/LOCAL_LABEL_CACHE.json",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_022_gated_matched_prefix/RAW_CASE_TRACES.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_022_gated_matched_prefix/SPARQL_QUERY_JOURNAL.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_022_gated_matched_prefix/REQUEST_JOURNAL.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_022_gated_matched_prefix/PILOT_CASES.csv",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/PILOT_INTEGRITY.json",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/CASE_USAGE_AUDIT.csv",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/RAW_CASE_TRACES.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/SPARQL_QUERY_JOURNAL.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/REQUEST_JOURNAL.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/TRANSPORT_SAFETY_REPAIR.md",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_024_label_source_audit/ITER24_REPORT.md",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_024_label_source_audit/LABEL_SOURCE_COVERAGE.csv",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_024_label_source_audit/QUERY_JOURNAL.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_014_selector_pilot/results.json",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_014_selector_pilot/case_results.csv",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_014_selector_pilot/selector_outputs_non_gold.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_014_selector_pilot/report.md",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_000_baseline/source_artifacts/outputs/experiments/fresh50_autonomous_causal_diagnosis/answer_selection_counterfactuals.csv",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_000_baseline/source_artifacts/outputs/experiments/fresh50_autonomous_causal_diagnosis/executive_decision.md",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_021_matched_prefix_replay/MECHANISM_DECISION.md",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_021_matched_prefix_replay/REPLAY_FEASIBILITY.csv",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_021_matched_prefix_replay/PROTECTED_SOURCE_HASHES.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files() -> set[Path]:
    result = set()
    code_root = REPO / "chain_of_relations"
    for path in code_root.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".yml", ".yaml"} and "__pycache__" not in path.parts:
            result.add(path)
    for relative in ("scripts/diagnostics/transport_safety_repair",):
        for path in (REPO / relative).rglob("*"):
            if path.is_file() and path.suffix == ".py" and "__pycache__" not in path.parts:
                result.add(path)
    diagnostics_root = REPO / "scripts/diagnostics"
    for pattern in ("iter_014*", "iter_021*", "iter_022*", "iter_023*", "iter_024*"):
        for directory in diagnostics_root.glob(pattern):
            if not directory.is_dir():
                continue
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix in {".py", ".yml", ".yaml"} and "__pycache__" not in path.parts:
                    result.add(path)
    return result


def _history_files() -> set[Path]:
    result = set()
    results = WORK / "RESULTS"
    for iteration_dir in results.glob("iter_*"):
        if iteration_dir.name == "iter_025_cache_compatible_readiness":
            continue
        for path in iteration_dir.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                result.add(path)
    return result


def capture() -> dict:
    evidence_paths = [REPO / path for path in EVIDENCE]
    missing = [str(path.relative_to(REPO)) for path in evidence_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing evidence inputs: {missing}")
    sources = _source_files()
    histories = _history_files()
    all_paths = sorted(sources | histories | set(evidence_paths))
    payload = {
        "iteration": 25,
        "capture_status": "ITERATION25_PROTECTED_SOURCE_AND_HISTORY_SNAPSHOT; protected paths were not modified",
        "source_file_count": len(sources),
        "historical_artifact_count": len(histories),
        "protected_file_count": len(all_paths),
        "files": [{"path": str(path.relative_to(REPO)), "sha256": sha256_file(path), "bytes": path.stat().st_size,
                   "category": "production_or_safety_source" if path in sources else "historical_or_evidence_artifact"}
                  for path in all_paths],
        "evidence_inputs": [{"path": str(path.relative_to(REPO)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
                           for path in evidence_paths],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    PROTECTED_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify(path: Path = PROTECTED_FILE) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = []
    for item in payload["files"]:
        file_path = REPO / item["path"]
        actual = sha256_file(file_path) if file_path.is_file() else None
        checks.append({"path": item["path"], "expected": item["sha256"], "actual": actual,
                       "matches": actual == item["sha256"]})
    return {"checked": len(checks), "matching": sum(row["matches"] for row in checks),
            "mismatches": [row for row in checks if not row["matches"]], "checks": checks}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("capture", "verify"))
    args = parser.parse_args()
    result = capture() if args.action == "capture" else verify()
    print(json.dumps({k: v for k, v in result.items() if k not in {"files", "checks", "evidence_inputs"}}, indent=2))
