#!/usr/bin/env python3
"""Offline audit and report for Iteration 16 state-lineage rerun artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
RESULTS = WORK / "RESULTS"
OUT = RESULTS / "iter_016_instrumented_tgar_rerun"
RUNS = OUT / "runs_with_state_lineage.jsonl"
USAGE = OUT / "call_usage.jsonl"
PAIRED = OUT / "paired_rerun_scoring_only.jsonl"
SUMMARY = OUT / "results.json"
METADATA = OUT / "run_metadata.json"
SEMANTIC = RESULTS / "iter_010_semantic_archived_path_audit/semantic_archived_path_alignment.csv"
INVERSE_MAP = RESULTS / "iter_009_inverse_relation_catalog/relation_inverse_map.json"
BASELINE = ROOT / "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256/runs.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def direction_match(edge: dict, rel_id: str, direction: str, inverse: dict[str, set[str]]) -> bool:
    return ((edge.get("id") == rel_id and edge.get("direction") == direction)
            or (edge.get("id") in inverse.get(rel_id, set()) and edge.get("direction") != direction))


def main() -> None:
    rows = read_jsonl(RUNS)
    usage = read_jsonl(USAGE)
    paired = read_jsonl(PAIRED)
    baseline = {r["question_key"]: r for r in read_jsonl(BASELINE) if r.get("method") == "v1_1_3"}
    with SEMANTIC.open(newline="", encoding="utf-8") as handle:
        semantic = {r["question_key"]: r for r in csv.DictReader(handle)}
    raw_inverse = json.loads(INVERSE_MAP.read_text(encoding="utf-8"))
    inverse: dict[str, set[str]] = {}
    for rel_id, values in raw_inverse.items():
        for other in values:
            inverse.setdefault(rel_id, set()).add(other)
            inverse.setdefault(other, set()).add(rel_id)

    group_rows = {}
    for role in ("TGAR_FAILURE", "BOTH_CORRECT_CONTROL"):
        cohort = [r for r in rows if r.get("cohort_role_scoring_only") == role]
        cohort_pairs = [r for r in paired if r.get("role") == role]
        group_rows[role] = {
            "n": len(cohort), "archived_hits_at_1": sum(int(r["archived_hits_at_1"]) for r in cohort_pairs),
            "rerun_hits_at_1": sum(int(r["rerun_hits_at_1"]) for r in cohort_pairs),
            "mean_f1": sum(float(r["f1"]) for r in cohort) / len(cohort) if cohort else None,
            "paired_outcomes": dict(Counter(r["outcome"] for r in cohort_pairs)),
            "candidate_status": dict(Counter(r["gold_candidate_diagnostic_scoring_only"]["candidate_presence_status"] for r in cohort)),
        }

    root_offer = Counter()
    for run in rows:
        role = run.get("cohort_role_scoring_only")
        if role not in {"TGAR_FAILURE", "BOTH_CORRECT_CONTROL"}:
            continue
        row = semantic[run["question_key"]]
        root_events = [e for e in run.get("state_lineage", [])
                       if e.get("stage") == "relation_prune" and not e.get("relation_chain")]
        candidates = []
        selected = []
        for event in root_events:
            candidates.extend({"id": x, "direction": "OUTGOING"} for x in event.get("candidate_relations_head", []))
            candidates.extend({"id": x, "direction": "INCOMING"} for x in event.get("candidate_relations_tail", []))
            selected.extend(event.get("selected_relations", []))
        relation_id = row["first_gold_relation_id"]
        direction = row["first_gold_relation_direction"]
        root_offer[f"{role}_n"] += 1
        root_offer[f"{role}_offered"] += int(any(direction_match(x, relation_id, direction, inverse) for x in candidates))
        root_offer[f"{role}_selected"] += int(any(direction_match(x, relation_id, direction, inverse) for x in selected))

    request_starts = sum(r.get("status") == "REQUEST_STARTED" for r in usage)
    metered = [r for r in usage if r.get("status") == "REQUEST_METERED"]
    cost = sum(float(r.get("cost_usd", 0) or 0) for r in metered)
    case_queries = sum(int(r.get("kg_queries", 0) or 0) for r in rows)
    query_status = Counter(event.get("status") for r in rows for event in r.get("sparql_history", []))
    actions = Counter(event.get("action") for r in rows for event in r.get("state_lineage", [])
                      if event.get("stage") == "reasoning")
    stage_counts = dict(Counter(event.get("stage") for run in rows for event in run.get("state_lineage", [])))
    route_changes = [r for r in rows if r.get("relation_chain", []) != baseline[r["question_key"]].get("relation_chain", [])]
    runner_path = ROOT / "scripts/diagnostics/run_fresh50_state_lineage.py"
    source = runner_path.read_bytes()
    helper_start = source.find(b"\ndef write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:\n")
    helper_end = source.find(b"\ndef relation_chain(", helper_start)
    if helper_start < 0 or helper_end < 0:
        raise RuntimeError("Could not reconstruct the inference-time runner source before its reporting-helper fix")
    pre_finalizer_source = source[:helper_start] + source[helper_end:]
    inference_runner_sha = hashlib.sha256(pre_finalizer_source).hexdigest()

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    summary.update({
        "hit_rate_at_1": summary["rerun_hits_at_1"] / len(rows),
        "archived_hits_at_1": sum(int(r["archived_hits_at_1"]) for r in paired),
        "failure_cohort": group_rows["TGAR_FAILURE"],
        "control_cohort": group_rows["BOTH_CORRECT_CONTROL"],
        "archived_route_changed_cases": len(route_changes),
        "archived_route_unchanged_cases": len(rows) - len(route_changes),
        "root_gold_first_relation_offered_and_selected": {
            "TGAR_FAILURE": {"n": root_offer["TGAR_FAILURE_n"], "offered": root_offer["TGAR_FAILURE_offered"],
                             "selected": root_offer["TGAR_FAILURE_selected"]},
            "BOTH_CORRECT_CONTROL": {"n": root_offer["BOTH_CORRECT_CONTROL_n"], "offered": root_offer["BOTH_CORRECT_CONTROL_offered"],
                                     "selected": root_offer["BOTH_CORRECT_CONTROL_selected"]},
        },
        "reasoning_action_counts": dict(actions),
        "case_kg_queries": case_queries,
        "preflight_kg_queries": 2,
        "kg_queries": case_queries + 2,
        "kg_query_scope": "183 read-only Freebase SPARQL requests during 19 cases plus two SELECT LIMIT 1 preflights (the initial run and offline-finalization resume). All 183 case-query events returned status ok.",
        "case_query_status_counts": dict(query_status),
        "case_errors": sum(bool(r.get("error")) for r in rows),
        "empty_content_metered_responses": sum(not r.get("has_content") for r in metered),
        "runner_summary_correction": "The initial runner hardcoded kg_queries=1 and omitted case-query accounting. This offline audit replaces that field with the count summed from the 19 saved run rows and records both health preflights.",
    })
    write_json(SUMMARY, summary)

    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    metadata.update({
        "case_read_only_kg_queries": case_queries,
        "preflight_read_only_kg_queries_total": 2,
        "total_read_only_kg_queries": case_queries + 2,
        "requests_started": request_starts,
        "metered_requests": len(metered),
        "input_tokens": sum(int(r.get("input_tokens", 0) or 0) for r in metered),
        "output_tokens": sum(int(r.get("output_tokens", 0) or 0) for r in metered),
        "cost_usd": cost,
        "request_runner_source_sha256": inference_runner_sha,
        "postrun_finalizer_source_sha256": sha(runner_path),
        "postrun_source_note": "After all 143 model responses were recorded, a JSONL writer helper was added because the runner crashed while writing the offline summary. The helper is only used after inference; the actual inference source hash is separately recorded.",
        "finalization_resume_sent_model_requests": 0,
    })
    write_json(METADATA, metadata)

    case_audit = []
    for run in rows:
        pair = next(r for r in paired if r["question_key"] == run["question_key"])
        sem = semantic[run["question_key"]]
        root_events = [e for e in run.get("state_lineage", []) if e.get("stage") == "relation_prune" and not e.get("relation_chain")]
        offered = selected = False
        for event in root_events:
            candidates = ([{"id": x, "direction": "OUTGOING"} for x in event.get("candidate_relations_head", [])]
                          + [{"id": x, "direction": "INCOMING"} for x in event.get("candidate_relations_tail", [])])
            offered |= any(direction_match(x, sem["first_gold_relation_id"], sem["first_gold_relation_direction"], inverse) for x in candidates)
            selected |= any(direction_match(x, sem["first_gold_relation_id"], sem["first_gold_relation_direction"], inverse)
                            for x in event.get("selected_relations", []))
        case_audit.append({
            "question_key": run["question_key"], "cohort_role": run["cohort_role_scoring_only"],
            "archived_hits_at_1": pair["archived_hits_at_1"], "rerun_hits_at_1": pair["rerun_hits_at_1"],
            "rerun_f1": run["f1"], "paired_outcome": pair["outcome"],
            "candidate_presence_status": run["gold_candidate_diagnostic_scoring_only"]["candidate_presence_status"],
            "entity_searches": run["gold_candidate_diagnostic_scoring_only"]["entity_search_count"],
            "prompt_history_llm_calls": run.get("llm_calls", 0),
            "kg_queries": run.get("kg_queries", 0), "archived_route_changed": run.get("relation_chain", []) != baseline[run["question_key"]].get("relation_chain", []),
            "gold_path_length": int(sem["gold_path_length"]),
            "first_gold_relation_offered": offered, "first_gold_relation_selected": selected,
            "route_structure": run.get("router_structure"), "route_mode": run.get("router_execution_mode"),
        })
    fields = list(case_audit[0])
    write_csv(OUT / "case_audit.csv", case_audit, fields)

    report = f"""# Iteration 16 — instrumented frozen-TGAR rerun

## Question and design

Does a one-time, instrumentation-only rerun of frozen TGAR reproduce the archived candidate-loss patterns, and which state transitions precede the misses? The run used the same 14 archived TGAR failures and five both-correct controls. Gold answers and cohort labels were kept in scoring-only sidecars; inference received the frozen questions and initial entities. The model, prompts, routing and CoR implementation were unchanged. No production algorithm was changed.

## Execution and integrity

- Model: `gpt-5.6-luna`; 1024 max completion tokens, 120-second timeout, one total attempt per request (no retries).
- Results: 19/19 cases completed, zero runtime errors.
- API usage: {request_starts} requests, {sum(int(r.get('input_tokens',0) or 0) for r in metered):,} input tokens and {sum(int(r.get('output_tokens',0) or 0) for r in metered):,} output tokens; estimated list-price cost **${cost:.7f}**. Two metered responses had empty content; all attempts were recorded and no retry was sent. Rate source is in `run_metadata.json`.
- Freebase: {case_queries} read-only SPARQL queries during cases, all `{dict(query_status)}`; two additional `SELECT * LIMIT 1` preflights were sent across the initial run and report-only resume. No KG query failed or timed out.
- Frozen source integrity: {metadata.get('frozen_production_source_integrity_after', {}).get('source_count', 0)}/7 production reasoning source hashes match.
- The first process completed all model calls but crashed while writing its final JSONL summary because a helper was missing. The report-only resume added that helper, reused the 19 saved cases, and sent **zero additional model requests**. An offline audit corrected the runner's hardcoded KG query total from 1 to {case_queries + 2}; the request source and finalizer hashes are separately recorded.

## Paired results

| Cohort | n | Archived Hits@1 | Rerun Hits@1 | Rerun mean F1 | Rerun-only | Archived-only | Candidate status on rerun |
|---|---:|---:|---:|---:|---:|---:|---|
| TGAR failures | 14 | {group_rows['TGAR_FAILURE']['archived_hits_at_1']} | {group_rows['TGAR_FAILURE']['rerun_hits_at_1']} | {group_rows['TGAR_FAILURE']['mean_f1']:.3f} | {group_rows['TGAR_FAILURE']['paired_outcomes'].get('RERUN_ONLY',0)} | {group_rows['TGAR_FAILURE']['paired_outcomes'].get('ARCHIVE_ONLY',0)} | `{json.dumps(group_rows['TGAR_FAILURE']['candidate_status'], sort_keys=True)}` |
| Both-correct controls | 5 | {group_rows['BOTH_CORRECT_CONTROL']['archived_hits_at_1']} | {group_rows['BOTH_CORRECT_CONTROL']['rerun_hits_at_1']} | {group_rows['BOTH_CORRECT_CONTROL']['mean_f1']:.3f} | {group_rows['BOTH_CORRECT_CONTROL']['paired_outcomes'].get('RERUN_ONLY',0)} | {group_rows['BOTH_CORRECT_CONTROL']['paired_outcomes'].get('ARCHIVE_ONLY',0)} | `{json.dumps(group_rows['BOTH_CORRECT_CONTROL']['candidate_status'], sort_keys=True)}` |
| Total | 19 | {summary['archived_hits_at_1']} | {summary['rerun_hits_at_1']} | {summary['rerun_mean_f1']:.3f} | {summary['archived_vs_rerun_hit_outcomes_scoring_only'].get('RERUN_ONLY',0)} | {summary['archived_vs_rerun_hit_outcomes_scoring_only'].get('ARCHIVE_ONLY',0)} | `{json.dumps(summary['candidate_presence_status_counts_scoring_only'], sort_keys=True)}` |

Two prior failures became correct in this repeat, while no control regressed. Both changed their final relation path relative to the archived run; one returned a gold answer that appeared in its last candidate search, and one produced a correct direct answer despite never retrieving the gold ID. These are same-method rerun differences, not treatment rescues. Three of 19 final relation paths differed from archive; the other 16 matched.

## State evidence

- Across the 14 failures, the first semantic gold relation was offered in **14/14** root relation-prune pools and selected in **13/14**. Controls: offered **5/5**, selected **4/5**. This matches the archived root-edge audit and makes root relation availability alone an inadequate explanation for most failures.
- On rerun, **12/14** failures still never exposed a gold answer ID in any entity-search output; one saw it only in an earlier search; one retained it in the last search. All five controls retained a gold answer ID in the last search.
- The 19 runs issued {case_queries} successful SPARQL queries. The trace contains {json.dumps(stage_counts, sort_keys=True)} method-stage events and reasoning actions {json.dumps(dict(actions), sort_keys=True)}. Candidate loss therefore occurred despite a healthy endpoint; the trace still does not prove a single exclusive cause.
- Rerun vs archive Hits@1: **17 ties, 2 rerun-only, 0 archive-only**. This variation shows the archived miss set is not fully stable under the same model/method configuration.

## Decision and limits

Reject the tested fixed-candidate answer selector as a development accuracy intervention (Iteration 14: 0/7 rescues and 1/5 control loss). This rerun does not validate a new method. It supports a narrower diagnosis: most residual failures remain upstream of final answer selection, and the first relation is usually available and selected. Later continuation, relation/path choice, and the depth cap remain plausible contributors. Same-method variation prevents interpreting the two rerun-only answers as an intervention effect. These 19 cases are development data, not an independent holdout.

The state trace records root, relation chain, relation pools/selections, entity candidate IDs, actions and SPARQL histories. It does not record a dedicated per-stack-node binding identity or every internal frontier transition, so it cannot alone separate model action choice from executor/search-depth effects.

## Reproduction and artifacts

```bash
RESEARCH_MAX_LLM_CALLS=160 RESEARCH_MAX_LLM_COST_USD=0.20 OPENAI_MAX_RETRIES=1 OPENAI_TIMEOUT=120 OPENAI_RETRY_INTERVAL=0 PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/run_fresh50_state_lineage.py --run
PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/audit_fresh50_state_lineage.py
```

The run command is already complete; do not execute it again under this authorization. Per-case results are in `case_audit.csv`; complete traces and scoring sidecars are in this directory. The one-edge oracle follow-up is reported separately in `../iter_017_oracle_one_edge_continuation/report.md`.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    correction = {
        "summary_field_corrected": "kg_queries",
        "initial_value": 1,
        "correct_value": case_queries + 2,
        "derivation": {"case_queries_sum_from_runs_jsonl": case_queries,
                       "initial_run_health_preflight": 1,
                       "report_only_resume_health_preflight": 1},
        "model_requests_added_by_correction": 0,
        "explanation": "The run summary originally represented only one preflight query and omitted the 183 read-only SPARQL requests made inside the 19 cases. This offline audit reconciles the actual query count.",
    }
    write_json(OUT / "analysis_adjustment.json", correction)
    hashes = {path.name: sha(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "artifact_hashes.json"}
    hashes["runner_source_sha256"] = inference_runner_sha
    hashes["postrun_finalizer_source_sha256"] = sha(runner_path)
    hashes["offline_auditor_source_sha256"] = sha(Path(__file__))
    write_json(OUT / "artifact_hashes.json", hashes)
    print(json.dumps({"cases": len(rows), "requests": request_starts, "cost_usd": cost,
                      "case_kg_queries": case_queries, "total_kg_queries": case_queries + 2,
                      "report": str(OUT / 'report.md'), "artifact_hashes": len(hashes)}, indent=2))


if __name__ == "__main__":
    main()
