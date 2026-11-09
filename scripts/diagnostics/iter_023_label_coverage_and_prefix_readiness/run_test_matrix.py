"""Run Iterations 21–23 offline suites and write a concise test record."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/TEST_RESULTS.json"
PYTHON = ROOT / ".venv/bin/python"
SUITES = [
    (21, "scripts/diagnostics/iter_021_matched_prefix", "test_*.py"),
    (22, "scripts/diagnostics/iter_022_gated_matched_prefix", "test_*.py"),
    (23, "scripts/diagnostics/iter_023_label_coverage_and_prefix_readiness", "test_*.py"),
]


def main() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")))
    records = []
    total = 0
    for iteration, start_dir, pattern in SUITES:
        cmd = [str(PYTHON), "-m", "unittest", "discover", "-s", start_dir, "-p", pattern, "-v"]
        result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        match = re.search(r"Ran (\d+) tests?", result.stdout)
        count = int(match.group(1)) if match else None
        if result.returncode or count is None or "OK" not in result.stdout:
            records.append({"iteration": iteration, "command": cmd, "status": "FAIL",
                            "returncode": result.returncode, "tests_run": count,
                            "output_tail": result.stdout[-5000:]})
            payload = {"iteration": 23, "status": "FAIL", "suites": records,
                       "paid_api_requests": 0, "live_sparql_attempts": 0}
            OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(payload, indent=2))
            return payload
        total += count
        records.append({"iteration": iteration, "command": cmd, "status": "PASS",
                        "returncode": 0, "tests_run": count,
                        "output_tail": result.stdout[-1200:]})

    payload = {
        "iteration": 23,
        "status": "PASS",
        "suites": records,
        "total_tests": total,
        "paid_api_requests": 0,
        "live_sparql_attempts": 0,
        "evidence_class": "OFFLINE_TESTS_ONLY",
        "note": "The Iteration 23 full synthetic STOP/ONE_HOP scenario is recorded separately in SYNTHETIC_PILOT.json.",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    result = main()
    raise SystemExit(0 if result["status"] == "PASS" else 1)
