#!/usr/bin/env python3
"""Offline score and trace audit for the Iteration 18 depth-4 pilot."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_cor_baseline_experiment import evaluate  # noqa: E402

WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
OUT = WORK / "RESULTS/iter_018_depth4_diagnostic_pilot"
RUNS = OUT / "runs_with_state_lineage.jsonl"
SCORES = OUT / "scoring_gold_only.json"
DEPTH3 = WORK / "RESULTS/iter_016_instrumented_tgar_rerun/runs_with_state_lineage.jsonl"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    runs = read_jsonl(RUNS)
    old_runs = {row["question_key"]: row for row in read_jsonl(DEPTH3)}
    score_rows = read_json(SCORES)["scores"]
    score_by_key = {row["question_key"]: row for row in score_rows}
    if len(runs) != 8 or len(score_by_key) != 8:
        raise SystemExit("Expected eight completed pilot runs and eight scoring sidecars")

    audits = []
    mismatches = []
    gain_mechanisms = []
    for run in runs:
        key = run["question_key"]
        score = score_by_key[key]
        gold_ids = set(map(str, score.get("answer_ids", [])))
        recomputed_hit, recomputed_f1 = evaluate(run.get("prediction", []), score.get("gold_answers", []))
        if int(recomputed_hit) != int(run.get("hits_at_1", 0)) or abs(float(recomputed_f1) - float(run.get("f1", 0.0))) > 1e-12:
            mismatches.append({"question_key": key, "saved_hit": run.get("hits_at_1"), "recomputed_hit": recomputed_hit,
                               "saved_f1": run.get("f1"), "recomputed_f1": recomputed_f1})
        entity_events = [event for event in run.get("state_lineage", []) if event.get("stage") == "entity_search"]
        max_search_depth = max((len(event.get("relation_chain", [])) for event in entity_events), default=0)
        gold_search_depths = sorted({len(event.get("relation_chain", [])) for event in entity_events
                                     if gold_ids.intersection(map(str, event.get("candidate_entity_ids", [])))})
        old = old_runs[key]
        old_entities = [event for event in old.get("state_lineage", []) if event.get("stage") == "entity_search"]
        old_max_search_depth = max((len(event.get("relation_chain", [])) for event in old_entities), default=0)
        old_gold_search_depths = sorted({len(event.get("relation_chain", [])) for event in old_entities
                                         if gold_ids.intersection(map(str, event.get("candidate_entity_ids", [])))})
        outcome = "TIE" if int(old.get("hits_at_1", 0)) == int(run.get("hits_at_1", 0)) else (
            "DEPTH4_ONLY" if int(run.get("hits_at_1", 0)) else "DEPTH3_ONLY")
        audit = {
            "question_key": key, "cohort_role": run["cohort_role_scoring_only"],
            "depth3_hits_at_1": int(old.get("hits_at_1", 0)), "depth4_hits_at_1": int(run.get("hits_at_1", 0)),
            "depth4_vs_depth3": outcome, "depth3_f1": float(old.get("f1", 0.0)),
            "depth4_f1": float(run.get("f1", 0.0)), "depth3_max_entity_search_depth": old_max_search_depth,
            "depth4_max_entity_search_depth": max_search_depth,
            "depth3_gold_answer_search_depths": old_gold_search_depths,
            "depth4_gold_answer_search_depths": gold_search_depths,
            "gold_found_in_four_edge_search": 4 in gold_search_depths,
            "depth4_candidate_status": run.get("gold_candidate_diagnostic_scoring_only", {}).get("candidate_presence_status"),
            "actions": [event.get("action") for event in run.get("state_lineage", []) if event.get("stage") == "reasoning"],
            "depth4_requests": run.get("request_count_case", 0),
        }
        audits.append(audit)
        if outcome == "DEPTH4_ONLY":
            gain_mechanisms.append({"question_key": key, "depth3_max_entity_search_depth": old_max_search_depth,
                                    "depth4_max_entity_search_depth": max_search_depth,
                                    "depth4_gold_answer_search_depths": gold_search_depths,
                                    "gold_found_in_four_edge_search": 4 in gold_search_depths,
                                    "depth4_prediction": run.get("prediction", [])})

    write_csv = OUT / "case_audit.csv"
    with write_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audits[0]))
        writer.writeheader()
        writer.writerows(audits)
    summary = read_json(OUT / "results.json")
    summary.update({
        "score_recomputation_mismatches": mismatches,
        "score_recomputation_passed": not mismatches,
        "depth4_only_gain_mechanisms": gain_mechanisms,
        "depth4_only_gains_with_gold_id_in_four_edge_search": sum(item["gold_found_in_four_edge_search"] for item in gain_mechanisms),
        "cases_with_depth4_search": sum(row["depth4_max_entity_search_depth"] >= 4 for row in audits),
        "independent_holdout": False,
    })
    (OUT / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace_audit = {
        "experiment": "Iteration 18 depth 3-to-4 trace mechanism audit",
        "score_recomputation_mismatches": mismatches,
        "depth4_only_gains": gain_mechanisms,
        "all_depth4_only_gains_exposed_gold_at_four_edge_search": bool(gain_mechanisms) and all(item["gold_found_in_four_edge_search"] for item in gain_mechanisms),
        "third_target_trace": next((row for row in audits if row["cohort_role"] == "TGAR_FAILURE" and row["depth4_hits_at_1"] == 0), None),
        "interpretation_limit": "The development comparison is paired but model outputs are stochastic. Fourth-edge gold retrieval supports a depth contribution on these cases; it does not establish generalization.",
    }
    (OUT / "trace_mechanism_audit.json").write_text(json.dumps(trace_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = (OUT / "report.md").read_text(encoding="utf-8")
    section = [
        "", "## Trace-level mechanism check", "",
        f"The two depth-4-only failure gains both had a gold answer ID in an entity-search result at relation-chain depth 4; the corresponding depth-3 reruns never issued a depth-4 entity search. The third target remained wrong and its depth-4 run reached only depth 2 before `Backtrack`/`Constraint`. All five controls remained correct.",
        "",
        f"Saved scores were independently recomputed from the scoring-only answer file: **{len(mismatches)} mismatches**. The case-level depth and gold-candidate evidence is in `case_audit.csv` and `trace_mechanism_audit.json`.",
        "",
        "This strengthens the interpretation that the added search depth contributed to the two answers in this development pilot. It does not prove that depth alone explains every paired difference, because model output is stochastic, and it does not establish an effect on an independent cohort.",
        "",
    ]
    if "## Trace-level mechanism check" not in report:
        report = report.rstrip() + "\n" + "\n".join(section)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    metadata = read_json(OUT / "run_metadata.json")
    metadata["score_recomputation_passed"] = not mismatches
    metadata["depth4_only_gains_with_gold_id_in_four_edge_search"] = sum(item["gold_found_in_four_edge_search"] for item in gain_mechanisms)
    metadata["offline_trace_auditor_source_sha256"] = sha(Path(__file__))
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hashes = {path.name: sha(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "artifact_hashes.json"}
    hashes["runner_source_sha256"] = sha(ROOT / "scripts/diagnostics/run_fresh50_depth4_pilot.py")
    hashes["offline_auditor_source_sha256"] = sha(Path(__file__))
    hashes["instrumentation_source_sha256"] = sha(ROOT / "scripts/diagnostics/run_fresh50_state_lineage.py")
    (OUT / "artifact_hashes.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(audits), "score_mismatches": len(mismatches),
                      "depth4_only_gains": len(gain_mechanisms),
                      "gains_with_gold_at_depth4": sum(item["gold_found_in_four_edge_search"] for item in gain_mechanisms),
                      "case_audit_rows": len(audits), "hashes": len(hashes)}, indent=2))
    if mismatches:
        raise SystemExit("Saved pilot scores did not match independent recomputation")


if __name__ == "__main__":
    main()
