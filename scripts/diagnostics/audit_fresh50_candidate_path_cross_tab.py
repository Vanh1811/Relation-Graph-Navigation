#!/usr/bin/env python3
"""Cross-tab archived candidate reachability against semantic gold-path prefixes."""
from __future__ import annotations

import csv
import hashlib
import json
import argparse
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
RESULTS = WORK / "RESULTS"
OUT = RESULTS / "iter_015_candidate_reachability_path_cross_tab"
CANDIDATE = RESULTS / "iter_011_residual_candidate_first_loss/candidate_first_loss.csv"
SEMANTIC = RESULTS / "iter_010_semantic_archived_path_audit/semantic_archived_path_alignment.csv"
DIRECTION = RESULTS / "iter_008_archived_path_direction_audit/archived_gold_path_direction_alignment.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = {row["question_key"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate question key in {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output_dir = args.output_dir
    candidate = read_csv(CANDIDATE)
    semantic = read_csv(SEMANTIC)
    direction = read_csv(DIRECTION)
    if set(candidate) != set(semantic) or set(candidate) != set(direction):
        raise ValueError("Iteration 8/10/11 case manifests do not align exactly")

    rows: list[dict[str, Any]] = []
    for key in sorted(candidate):
        c, s, d = candidate[key], semantic[key], direction[key]
        rows.append({
            "question_key": key,
            "cohort_role": c["cohort_role"],
            "candidate_presence_status": c["candidate_presence_status"],
            "entity_search_count": int(c["entity_search_count"]),
            "max_semantic_gold_prefix": int(s["max_archived_semantic_gold_prefix"]),
            "semantic_divergence_status": s["semantic_divergence_status"],
            "first_gold_relation_semantically_offered": s["first_relation_semantically_offered"],
            "first_gold_relation_semantically_selected": s["first_relation_semantically_selected"],
            "first_gold_relation_selected_in_matching_direction": d["gold_first_relation_selected_in_matching_direction"],
            "gold_answer_in_oracle_path_candidates": s["gold_answer_in_oracle_path_candidates"],
            "gold_answer_seen_anywhere_archived": c["gold_seen_any_search"],
        })

    failures = [r for r in rows if r["cohort_role"] == "TGAR_FAILURE"]
    controls = [r for r in rows if r["cohort_role"] == "BOTH_CORRECT_CONTROL"]
    never = [r for r in failures if r["candidate_presence_status"] == "NEVER_SEEN"]
    seen_lost = [r for r in failures if r["candidate_presence_status"] == "SEEN_THEN_ABSENT_FROM_LATER_SEARCHES"]
    prefix = Counter(r["max_semantic_gold_prefix"] for r in never)
    summary = {
        "experiment": "cross-tab archived candidate first-loss status against semantic gold-path alignment",
        "iteration": 15,
        "status": "COMPLETED_OFFLINE",
        "cases": len(rows),
        "tgar_failures": len(failures),
        "both_correct_controls": len(controls),
        "never_seen_gold_answer": len(never),
        "seen_then_lost_gold_answer": len(seen_lost),
        "never_seen_with_zero_semantic_gold_prefix": prefix.get(0, 0),
        "never_seen_with_nonzero_semantic_gold_prefix": sum(n for k, n in prefix.items() if k > 0),
        "never_seen_prefix_distribution": {str(k): prefix[k] for k in sorted(prefix)},
        "never_seen_first_gold_relation_semantically_selected": sum(r["first_gold_relation_semantically_selected"] == "True" for r in never),
        "never_seen_first_gold_relation_selected_in_matching_direction": sum(r["first_gold_relation_selected_in_matching_direction"] == "True" for r in never),
        "seen_then_lost_full_semantic_gold_path_executed": sum(r["semantic_divergence_status"] == "FULL_SEMANTIC_GOLD_PATH_EXECUTED" for r in seen_lost),
        "controls_gold_answer_in_last_entity_search": sum(r["candidate_presence_status"] == "PRESENT_IN_LAST_ENTITY_SEARCH" for r in controls),
        "controls_full_semantic_gold_path_executed": sum(r["semantic_divergence_status"] == "FULL_SEMANTIC_GOLD_PATH_EXECUTED" for r in controls),
        "oracle_path_candidate_coverage": sum(r["gold_answer_in_oracle_path_candidates"] == "True" for r in failures),
        "kg_queries": 0,
        "llm_calls": 0,
        "gold_visible_to_archived_test_time_methods": False,
        "production_code_changed": False,
        "inputs": {
            str(CANDIDATE.relative_to(ROOT)): sha(CANDIDATE),
            str(SEMANTIC.relative_to(ROOT)): sha(SEMANTIC),
            str(DIRECTION.relative_to(ROOT)): sha(DIRECTION),
        },
        "claim_scope": "Descriptive cross-tab of already archived traces. It localizes where saved paths leave semantic gold prefixes relative to candidate presence, but does not establish why the path diverged or whether changing it would change answers.",
    }
    if (len(failures), len(controls), len(never), len(seen_lost)) != (14, 5, 12, 2):
        raise ValueError(f"unexpected cohort sizes: {summary}")
    if summary["never_seen_with_nonzero_semantic_gold_prefix"] != 8:
        raise ValueError("unexpected gold-prefix cross-tab; inspect source artifacts")

    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "candidate_path_cross_tab.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = f"""# Iteration 15 — candidate reachability × semantic path cross-tab

## Question

Among 14 archived TGAR misses outside the final-answer-selector cohort, how does gold-answer candidate reachability relate to the longest semantic gold-path prefix visible in the archived TGAR route? This was selected as a zero-call discriminator after the Iteration 14 selector no-go.

## Results

- **12/14** misses never exposed a gold answer ID in any archived entity-search output; **2/14** exposed it earlier and later lost it.
- Among the 12 never-seen misses, **8** had a nonzero semantic gold-path prefix before the archived route diverged; **4** diverged at the first semantic edge.
- The first gold relation was semantically selected in **{summary['never_seen_first_gold_relation_semantically_selected']}/12** never-seen cases (direction-aware ID alignment: **{summary['never_seen_first_gold_relation_selected_in_matching_direction']}/12**).
- Both seen-then-lost cases had the full semantic gold path in archived search history, yet neither retained the gold answer in its final entity-search results.
- All **5/5** both-correct controls retained gold answers in the last archived entity search; **4/5** executed a full semantic gold path. The remaining control answered through a different route.
- The oracle-path candidate replay covered **14/14** residual failure answers, confirming that the stored database/query path could retrieve them under supplied gold relations. This is an oracle diagnostic, not ordinary inference.
- Cost: **0 LLM calls, 0 KG queries**. No production method or source was changed.

## Interpretation

Initial root-relation choice alone is unlikely to explain most NEVER_SEEN failures: the first gold relation was semantically selected in 11/12. Eight routes followed at least one semantic gold edge, then diverged before the gold answer appeared. This shifts the next measurement toward downstream active-state/entity progression, later relation choice, or binding/query transitions. The logs do not identify which of these caused the divergence: gold-path alignment is retrospective and not a controlled intervention.

The two seen-then-lost cases differ: their archived routes executed a full semantic gold path, but the answer candidate disappeared later. They remain the best targeted cases for checking whether candidate selection or final filtering acts after a successful path. The Iteration 14 ID selector did not rescue the fixed-candidate failures, so do not repeat it unchanged.

## Reproduction

```bash
PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/audit_fresh50_candidate_path_cross_tab.py
```

Inputs are the Iteration 8 direction audit, Iteration 10 semantic path audit, and Iteration 11 candidate first-loss audit. Their SHA256 values and this output manifest are in `results.json` and `artifact_hashes.json`.

This cross-tab is descriptive; it does not prove the cause of divergence or validate a route/executor change.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    hashes = {p.name: sha(p) for p in sorted(output_dir.iterdir()) if p.is_file() and p.name != "artifact_hashes.json"}
    (output_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
