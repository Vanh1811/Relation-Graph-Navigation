"""Build Iteration 30 handoff and reproducible report artifacts offline."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from chain_of_relations.methods.iter_030_operator_recovery.relation_aware_count import classify_count_intent
from relation_graph_kgqa.logical_forms.sexpr_parser import parse_s_expression


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_030_results_first"
BASE = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_028_terminal_boundary_repair"
RUN = BASE / "RUNS/iter29_terminal_candidate_selector_attempt3"
ITER29_REPORT = BASE / "ITER29_ATTEMPT3_FINAL_REPORT.md"
COHORT = BASE / "NEXT_INFERENCE_COHORT_ATTEMPT3_NON_GOLD.jsonl"
GOLD = ROOT / "data/interim/grailqa_dev.jsonl"


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def root_operator(expression: str) -> str:
    try:
        return parse_s_expression(expression).operator.upper()
    except Exception:
        return "PARSE_ERROR"


def build() -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    audit = json.loads((RESULTS / "PHASE_A_AUDIT.json").read_text(encoding="utf-8"))
    score = json.loads((RUN / "PAIRED_ANALYSIS_GOLD_ONLY.json").read_text(encoding="utf-8"))
    run_metadata = json.loads((RUN / "RUN_METADATA.json").read_text(encoding="utf-8"))
    historical_report = ITER29_REPORT.read_text(encoding="utf-8")
    cost_match = re.search(r"Rate-based cost estimate:\s+\*\*US\$([0-9.]+)", historical_report)
    historical_cost_usd = float(cost_match.group(1)) if cost_match else None
    cohort = jsonl(COHORT)
    gold_by_id = {str(row["id"]): row for row in jsonl(GOLD) if row.get("split") == "dev"}
    score_by_key = {row["question_key"]: row for row in score["case_results"]}

    paired_rows = []
    trigger_rows = []
    trigger_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for case in cohort:
        key = case["question_key"]
        raw = gold_by_id[key.split("::", 1)[1]]
        scored = score_by_key[key]
        op = root_operator(str(raw.get("logical_form") or ""))
        triggered, reason = classify_count_intent(str(case.get("question") or ""))
        is_count = op == "COUNT"
        outcome = "tp" if triggered and is_count else "fp" if triggered else "fn" if is_count else "tn"
        trigger_counts[outcome] += 1
        paired_rows.append({
            "question_key": key,
            "source_split": "grailqa_dev",
            "gold_only_analytic": True,
            "baseline_hits_at_1_scoring_only": scored["baseline_hits_at_1"],
            "terminal_selector_hits_at_1_scoring_only": scored["intervention_hits_at_1"],
            "paired_hits_outcome_scoring_only": scored["paired_hits_outcome"],
            "baseline_f1_scoring_only": scored["baseline_f1"],
            "terminal_selector_f1_scoring_only": scored["intervention_f1"],
            "delta_f1_scoring_only": scored["delta_f1"],
            "gold_root_operator_scoring_only": op,
            "runtime_count_triggered_gold_only_audit": triggered,
            "runtime_count_trigger_reason": reason,
        })
        if triggered and not is_count:
            trigger_rows.append({
                "control_id": key,
                "source": "GRAILQA_DEV_GOLD_ONLY_FALSE_POSITIVE",
                "question": raw.get("question"),
                "classifier_triggered": triggered,
                "classifier_reason": reason,
                "gold_root_operator_scoring_only": op,
                "expected_runtime_action_scoring_only": "DO_NOT_APPLY_COUNT_OPERATOR",
                "limitation": "Gold label used only to audit trigger behavior; never supplied to runtime.",
            })
        elif not triggered and is_count:
            trigger_rows.append({
                "control_id": key,
                "source": "GRAILQA_DEV_GOLD_ONLY_FALSE_NEGATIVE",
                "question": raw.get("question"),
                "classifier_triggered": triggered,
                "classifier_reason": reason,
                "gold_root_operator_scoring_only": op,
                "expected_runtime_action_scoring_only": "COUNT_OPERATOR_WOULD_BE_MISSED",
                "limitation": "Gold label used only to audit trigger behavior; never supplied to runtime.",
            })

    synthetic_controls = [
        ("How much did the company earn in revenue?", "numeric magnitude/property"),
        ("How many miles long is Route 66?", "measurement, not entity cardinality"),
        ("How many years old is the building?", "age property"),
        ("What number was assigned to the album?", "identifier property"),
        ("Which trial has the least number of expected participants?", "argmin over a numeric property"),
        ("What is the maximum number of entrants allowed?", "numeric bound"),
        ("Who was the number one ranked player?", "entity answer"),
        ("What is the total amount of money held by the company?", "money magnitude"),
        ("How many dollars did the organization donate?", "currency magnitude"),
        ("What was the number of the jersey?", "identifier property"),
    ]
    for index, (question, expected_action) in enumerate(synthetic_controls, start=1):
        triggered, reason = classify_count_intent(question)
        trigger_rows.append({
            "control_id": f"SYNTHETIC_CONTROL_{index:02d}",
            "source": "SYNTHETIC_GOLD_FREE_NEGATIVE_CONTROL",
            "question": question,
            "classifier_triggered": triggered,
            "classifier_reason": reason,
            "gold_root_operator_scoring_only": "NOT_APPLICABLE",
            "expected_runtime_action_scoring_only": "DO_NOT_APPLY_COUNT_OPERATOR: " + expected_action,
            "limitation": "Manually authored negative control; expected action is semantic review, not benchmark gold.",
        })
    write_csv(RESULTS / "COUNT_TRIGGER_CONTROL_AUDIT.csv", trigger_rows)
    write_csv(RESULTS / "ITER29_PAIRED_CASE_REFERENCE_GOLD_ONLY.csv", paired_rows)

    baseline_hits = sum(int(row["baseline_hits_at_1_scoring_only"]) for row in paired_rows)
    selector_hits = sum(int(row["terminal_selector_hits_at_1_scoring_only"]) for row in paired_rows)
    baseline_f1 = mean(float(row["baseline_f1_scoring_only"]) for row in paired_rows)
    selector_f1 = mean(float(row["terminal_selector_f1_scoring_only"]) for row in paired_rows)
    tpr_denominator = trigger_counts["tp"] + trigger_counts["fn"]
    precision_denominator = trigger_counts["tp"] + trigger_counts["fp"]
    trigger_metrics = {
        "cohort_size": len(cohort),
        "gold_count_operator_questions_scoring_only": tpr_denominator,
        "trigger_confusion_counts_gold_only": trigger_counts,
        "trigger_recall_gold_only": trigger_counts["tp"] / tpr_denominator if tpr_denominator else None,
        "trigger_precision_gold_only": trigger_counts["tp"] / precision_denominator if precision_denominator else None,
        "scope_warning": "Posthoc development-set audit only; the heuristic was developed after inspecting this same cohort, so this is not held-out trigger performance. Gold labels never enter runtime.",
    }
    summary = {
        "iteration": 30,
        "decision": "ARCHITECTURAL BLOCKER REQUIRES NEW STUDY",
        "evidence_class": "OFFLINE_POSTHOC_GRAILQA_DEV_ANALYSIS_AND_UNVALIDATED_ISOLATED_PROTOTYPE",
        "network_calls": {"openai": 0, "sparql": 0},
        "iter29_integrity_checks": audit["integrity_checks"],
        "iter29_recomputed": {
            "n": len(paired_rows),
            "baseline_hits_at_1": baseline_hits,
            "terminal_selector_hits_at_1": selector_hits,
            "paired_hits_wins_losses_ties": {
                "wins": sum(row["paired_hits_outcome_scoring_only"] == "INTERVENTION_WIN" for row in paired_rows),
                "losses": sum(row["paired_hits_outcome_scoring_only"] == "INTERVENTION_LOSS" for row in paired_rows),
                "ties": sum(row["paired_hits_outcome_scoring_only"] == "TIE" for row in paired_rows),
            },
            "mean_f1_baseline": baseline_f1,
            "mean_f1_terminal_selector": selector_f1,
            "baseline_misses": 800 - baseline_hits,
            "api_requests_started": run_metadata.get("api_requests_started"),
            "physical_sparql_attempts": run_metadata.get("physical_sparql_attempts"),
            "rate_based_cost_estimate_usd": historical_cost_usd,
        },
        "count_operator_audit": audit["numeric_operator_taxonomy"],
        "gold_free_trigger_audit": trigger_metrics,
        "operator_prototype": {
            "implementation": "isolated optional module; no production defaults changed",
            "supported_query_contract": ["linear grounded paths up to six edges", "directional relations", "shared-variable joins", "COUNT(DISTINCT)", "runtime target type filter", "all-node inequality constraints", "single injected read-only executor call", "typed numeric scalar"],
            "fails_closed_when": ["no explicit conservative count trigger", "grounded path did not succeed", "target_type_id unavailable", "no executor supplied", "invalid query context", "query timeout/error/malformed result"],
            "live_or_end_to_end_validated": False,
            "reason_not_ready": "Current CoR runtime exposes natural-language answer type and a relation path, but not a reliable gold-free canonical target type ID or a gold-free proof that the grounded path is complete and answers the requested class.",
        },
        "prototype_test_suite": {
            "command": "PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests -p 'test_iter_030_relation_aware_count.py' -v",
            "test_methods": 14,
            "result": "PASS",
            "network_calls": 0,
        },
        "source_hashes": {
            "gold_source_sha256": sha256(GOLD),
            "cohort_sha256": sha256(COHORT),
            "iter29_scored_cases_sha256": sha256(RUN / "PAIRED_ANALYSIS_GOLD_ONLY.json"),
            "iter29_final_report_sha256": sha256(ITER29_REPORT),
            "audit_script_sha256": sha256(ROOT / "scripts/diagnostics/iter_030_operator_candidate_recovery/audit_attempt3.py"),
            "runtime_operator_sha256": sha256(ROOT / "chain_of_relations/methods/iter_030_operator_recovery/relation_aware_count.py"),
            "runtime_schema_sha256": sha256(ROOT / "chain_of_relations/methods/iter_030_operator_recovery/runtime_features.py"),
        },
        "artifacts": [
            "PHASE_A_AUDIT.json", "FAILURE_MECHANISM_MATRIX.csv", "COUNT_OPERATOR_AUDIT.csv",
            "ENTITY_MISS_TRACE_AUDIT.csv", "EMPTY_PREDICTION_AUDIT.csv", "GOLD_FREE_RUNTIME_FEATURES.json",
            "COUNT_TRIGGER_CONTROL_AUDIT.csv", "ITER29_PAIRED_CASE_REFERENCE_GOLD_ONLY.csv",
            "ITER30_RESEARCH_REPORT.md", "CHATGPT_HANDOFF.md", "THESIS_CHECKPOINT.md",
            "REPRODUCTION_README.md", "ITER30_PRESENTATION_TABLE.md", "ITER30_TESTS.log",
        ],
    }
    (RESULTS / "RESULTS_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    counts = audit["entity_miss_taxonomy"]["candidate_pool_groups"]
    count_tax = audit["numeric_operator_taxonomy"]
    entity_bounds = audit["entity_miss_taxonomy"]["boundary_counts"]
    empty_tax = audit["empty_prediction_taxonomy"]["category_counts"]
    report = f"""# Iteration 30 — Operator and Candidate Recovery Research

**Decision: ARCHITECTURAL BLOCKER REQUIRES NEW STUDY.** This offline campaign confirms a repeatable COUNT failure class and implements an isolated prototype, but the current runtime cannot reliably supply the canonical target class or certify that its relation path is complete. No new live evaluation is valid yet.

## Scope and evidence boundary

The audit independently revalidated Iteration 29 attempt 3: all 800 cohort, paired, trace, snapshot, hash, manifest, authorization and `SCOREABLE` checks passed; gold scoring was written after the gate. This is GrailQA **dev** evidence. Iteration 29's baseline and terminal selector each scored {baseline_hits}/800 Hits@1, with 0 paired wins, 0 losses and 800 ties. Mean F1 was {baseline_f1:.5f} and {selector_f1:.5f}, respectively. Historical Iteration 29 usage was {run_metadata.get('api_requests_started')} API requests, {run_metadata.get('physical_sparql_attempts')} physical SPARQL attempts, and US${historical_cost_usd:.7f} recorded rate-based cost. Iteration 30 made **zero OpenAI requests and zero SPARQL calls**; it did not run or score a new method.

## Error mechanism results

The baseline has 222 misses: 68 numeric answers and 154 entity answers. Within the entity misses, gold appears in the recorded candidate pool in {counts.get('GOLD_IN_RECORDED_CANDIDATE_POOL', 0)}, is absent from that recorded pool in {counts.get('GOLD_ABSENT_FROM_RECORDED_CANDIDATE_POOL', 0)}, and no entity pool was recorded in {counts.get('NO_ENTITY_CANDIDATE_POOL', 0)}. Recorded-pool absence does not prove the graph lacks an answer. The 31 gold-present cases localize the failure to terminal selection or ID/name mapping, but Iteration 29's selector did not recover them.

For the 84 gold-absent entity misses, persisted first-hop prompts show {entity_bounds.get('FIRST_REQUIRED_RELATION_OMITTED_BY_RELATION_PRUNING', 0)} required relations present in the logged directional candidate prompt but omitted from its selected relation IDs; 1 required relation was absent from that logged prompt; 42 were selected but not executed as the required first edge; and 14 diverged later in path/stop/join/terminal execution. Direction and raw relation-search bindings are not fully persisted, so the 42 and 14 are strong trace-level diagnoses, not proof that Freebase has or lacks those answers. Among the 39 no-pool cases, 19 have a recorded SPARQL error, 11 have no grounded terminal relation path, and 9 have incomplete query provenance.

Of 136 empty baseline predictions, the gold-only diagnostic split is {empty_tax.get('NUMERIC_SCALAR_GOLD_TEXT_NOT_TYPED_AS_PREDICTION_ID', 0)} numeric cases where the baseline text matches the gold number but the prediction ID contract is empty, {empty_tax.get('NUMERIC_COUNT_VALUE_OR_TYPED_OUTPUT_FAILURE', 0)} numeric cases with no exact scalar-text match, {empty_tax.get('NO_RECORDED_ENTITY_CANDIDATE_POOL', 0)} entity cases without a recorded pool, and {empty_tax.get('CANDIDATE_POOL_PRESENT_BUT_NO_ENTITY_ID_PREDICTION', 0)} entity cases with a pool but no predicted IDs. These are posthoc labels, not runtime features.

## COUNT operator audit and prototype

All 68 numeric misses have root logical-form operator `COUNT` and a SPARQL `COUNT` aggregate. Their gold query shapes use a `SELECT DISTINCT` subquery, a target `type.object.type` constraint and all pairwise node-inequality filters in all 68 cases. Paths are 55 one-edge and 13 multi-edge linear joins (8 two-edge, 5 three-edge); target class IDs span 32 different classes. The gold-free wording heuristic triggers on {count_tax['runtime_features']['count_intent_triggered']}/68 of this miss subset and misses {count_tax['runtime_features']['count_intent_not_triggered']}; the unmatched question is “what is the unit of volume that is associated with a measurement system that has a magnetic field strength unit of ampere per metre?” with numeric gold `7`. This question/LF alignment needs manual semantic review. The wording rule was developed after inspecting this same dev cohort, so its 800-case trigger precision/recall are posthoc diagnostics, not held-out performance or deployment estimates.

The optional module `chain_of_relations/methods/iter_030_operator_recovery/` builds a directional, joined `COUNT(DISTINCT ...)` query, preserves pairwise node inequalities, adds a runtime target-type filter, parses only a typed nonnegative integer, makes at most one injected executor call, and otherwise preserves the baseline. It has no default network client and is not connected to production CoR/TGAR.

Two independent issues prevent promotion. First, the current router's natural-language answer-type hint is not a canonical Freebase type ID; a safe resolver for the 32 target classes is not present. The prototype therefore correctly fails closed on all real current-runtime cases without that field. Second, the gold-only audit finds the existing CoR relation set matches the gold relation set in only 39/68 count misses. The current trace does not provide a gold-free completeness check. A COUNT over an incorrect or incomplete path would simply produce a confident wrong number. Persisted entity-search row counts equal the gold count in 45/68 cases, but those queries omit target-type constraints and can hit a 1000-row cap, so these values are not valid aggregate predictions.

Question wording is not sufficient by itself: a count trigger must avoid argmin/argmax-style numeric property questions, measurement questions, IDs and money magnitudes. The full-cohort trigger confusion counts, {len(trigger_rows)} control rows (including {len(synthetic_controls)} manually authored synthetic negatives), are included as audit material. Synthetic controls are explicitly marked and do not count as benchmark evidence.

## Tests and reproducibility

The prototype test suite has 14 passing test methods. It covers forward/inverse paths, two/three-edge joins, duplicate-safe distinct count, node loops, zero, malformed/error/timeout responses, no-retry behavior, disabled-by-default transport, target-type fail-closed behavior, baseline preservation, malformed IDs and the gold-free runtime input allow-list. Run it with the command in `REPRODUCTION_README.md`. The test is synthetic/local: it verifies query construction and output handling, not Virtuoso's response to the query.

The authoritative per-case inputs and mechanism partitions are the paired reference CSV and the four audit CSVs in this directory. Gold fields are explicitly marked scoring-only/analytic. `GOLD_FREE_RUNTIME_FEATURES.json` is the runtime allow-list. Historical Iteration 29 outputs were not modified.

## Decision and next step

Do not launch a new paid/live evaluation. The next research task is to build and offline-validate a gold-free target-type resolver **and** a path-completeness/answer-type check, then test its aggregate query on a small separately authorized cohort. Until those exist, this extension has no defensible end-to-end trigger and cannot establish accuracy lift. The current result is an architectural blocker, not a negative measured accuracy result.
"""
    (RESULTS / "ITER30_RESEARCH_REPORT.md").write_text(report, encoding="utf-8")

    handoff = f"""# ChatGPT handoff — Iteration 30

## Result

Iteration 30 completed its offline audit and isolated operator prototype. **Decision: ARCHITECTURAL BLOCKER REQUIRES NEW STUDY.** It did not run a new benchmark and made no OpenAI or live SPARQL requests.

## Verified foundation

- Iteration 29 attempt 3 remains the authoritative {len(paired_rows)}-case GrailQA dev comparison. The independent scoring gate was `SCOREABLE`; the new audit reran integrity checks successfully.
- Baseline and TerminalCandidateSelector each had {baseline_hits}/800 Hits@1; paired outcome was 0 wins / 0 losses / 800 ties. Mean F1: {baseline_f1:.5f} vs {selector_f1:.5f}.
- Baseline misses: 222 = 68 numeric + 154 entity. Entity misses split into 84 gold absent from the recorded pool, 39 no recorded entity pool, 31 gold present in the pool. Pool absence is not evidence of KG absence.
- All 68 numeric misses are gold COUNT operations. Gold queries require target type and distinct-node semantics; 13 paths require multi-edge joins. Gold classes span 32 types.

## What changed

Added `chain_of_relations/methods/iter_030_operator_recovery/`, an optional no-network prototype that compiles an aggregate over a grounded, directional relation chain. It preserves joins and node inequalities, applies a required runtime type ID, accepts a single injected executor call, returns typed scalar text, and preserves baseline output for unsupported/error cases. Production defaults were not changed. Offline gold-only artifacts are separated from the runtime allow-list.

## Why it cannot be evaluated honestly yet

1. CoR supplies a natural-language answer-type hint, not a verified Freebase target class ID. Without a target class, the prototype refuses to query; this is necessary because all 68 gold COUNT forms constrain the target class and there are 32 distinct classes.
2. The existing relation set matches the gold relation set in only 39/68 count misses (posthoc gold-only diagnostic). Runtime traces contain no independent gold-free path-completeness test. Aggregating the current path could produce the wrong count.
3. Persisted entity-search row counts matching gold in 45/68 cases is not valid COUNT evidence: the query lacks a target type filter and may be capped at 1000.
4. Therefore Iteration 30 has no new method score, no estimated improvement, and no live endpoint validation. The trigger audit and fixture tests are not benchmark results; the trigger rule was developed on this same dev cohort and is not independently validated.

## Most important artifacts

- `ITER30_RESEARCH_REPORT.md` — full result and decision.
- `RESULTS_SUMMARY.json` — recomputed metrics, integrity, hashes and limitations.
- `FAILURE_MECHANISM_MATRIX.csv` — all 800 question-level posthoc labels.
- `COUNT_OPERATOR_AUDIT.csv`, `ENTITY_MISS_TRACE_AUDIT.csv`, `EMPTY_PREDICTION_AUDIT.csv` — mechanism details.
- `COUNT_TRIGGER_CONTROL_AUDIT.csv` — gold-only false-trigger/miss audit plus separately marked synthetic hard negatives.
- `ITER29_PAIRED_CASE_REFERENCE_GOLD_ONLY.csv` — paired case reference; does not include Iteration 30 predictions.
- `GOLD_FREE_RUNTIME_FEATURES.json` — runtime allow-list and prohibited gold fields.
- `THESIS_CHECKPOINT.md`, `REPRODUCTION_README.md`, and `ITER30_PRESENTATION_TABLE.md`.

## One next decision

Do not spend on another run. The next study must supply and validate a gold-free target-type resolver and a path-completeness check. After those components work offline, a small new live evaluation can be designed with a separate exact-manifest authorization. The current result is an architectural blocker, not evidence that COUNT recovery cannot work.
"""
    (RESULTS / "CHATGPT_HANDOFF.md").write_text(handoff, encoding="utf-8")

    thesis = f"""# Thesis checkpoint — Iteration 30

## Research question

Can relation-centric KGQA recover numeric/count answers by executing a typed aggregate over CoR's grounded relation path?

## Evidence available

The completed Iteration 29 GrailQA dev run (800 questions, independently `SCOREABLE`) found 222 baseline misses. Gold-only review of all 68 numeric misses verified root `COUNT` operators. Gold semantics require a target class, DISTINCT endpoints, and distinct graph nodes; 13 cases require multi-hop variable joins. The isolated Iteration 30 prototype encodes this supported query fragment and is covered by local deterministic tests.

## What cannot be claimed

No Iteration 30 benchmark prediction was generated or scored. No accuracy improvement has been measured. Gold-derived path comparisons and row-count coincidences are diagnostics only. The GrailQA dev cohort is not a confirmatory holdout. No live SPARQL or API query was made in this phase.

## Current blocker

The runtime lacks a verified gold-free mapping from natural-language answer type to a Freebase target class ID. It also lacks a signal that the current relation path is complete and semantically aligned. Posthoc, only 39/68 relation sets match gold, illustrating why blindly counting the current path would be unsafe.

## Thesis-ready conclusion

Iteration 30 identifies a concrete representation gap between relation-path entity retrieval and numeric aggregate answers. A COUNT operator is technically expressible, including joins and set semantics, but safe activation requires runtime target-class grounding and path validation. The research outcome is an architectural blocker with a precise next capability requirement, not a failed or successful accuracy comparison.
"""
    (RESULTS / "THESIS_CHECKPOINT.md").write_text(thesis, encoding="utf-8")

    readme = """# Iteration 30 reproduction

All commands are local/offline. No OpenAI key or SPARQL endpoint is needed.

From the repository root:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/iter_030_operator_candidate_recovery/audit_attempt3.py
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests -p 'test_iter_030_relation_aware_count.py' -v
PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/iter_030_operator_candidate_recovery/finalize_report.py
```

The first script verifies Iteration 29 cohort, gate, hashes and score timing, then writes gold-only analytic CSV/JSON outputs. The tests exercise the isolated query compiler and injected executor contract with local mocks. The finalizer writes the handoff and thesis artifacts. Runtime code lives under `chain_of_relations/methods/iter_030_operator_recovery/`; it has no default network client and is not imported by production CoR/TGAR.

Inputs: Iteration 29 attempt 3 artifacts under `outputs/experiments/fresh50_continuous_agent/RESULTS/iter_028_terminal_boundary_repair/` and `data/interim/grailqa_dev.jsonl`. The audit checks their declared hashes before producing results. Gold-only artifacts must never be passed to runtime inference.
"""
    (RESULTS / "REPRODUCTION_README.md").write_text(readme, encoding="utf-8")

    table = f"""# Iteration 30 evidence table

| Evidence | Result | Interpretation |
|---|---:|---|
| Iteration 29 paired dev cohort | 800 | `SCOREABLE`; development only |
| CoR/TGAR baseline Hits@1 | {baseline_hits}/800 ({baseline_hits/8:.2f}%) | Frozen baseline reference |
| Terminal selector Hits@1 | {selector_hits}/800 ({selector_hits/8:.2f}%) | 0 wins, 0 losses, 800 ties |
| Baseline misses | 222 | 68 numeric + 154 entity |
| Numeric misses with gold COUNT LF | 68/68 | Gold-only operator taxonomy |
| Gold paths needing multi-edge joins | {count_tax['requires_shared_variable_join']}/68 | Sidecar must preserve shared variables |
| Gold queries with target type constraint | {count_tax['target_type_filter_in_gold_count']}/68 | Runtime currently has no verified type ID |
| Runtime relation sets matching gold (posthoc) | {count_tax['runtime_relation_set_matches_gold_scoring_only']}/68 | No gold-free completeness check |
| Iteration 30 API / SPARQL calls | 0 / 0 | Offline-only prototype and audit |
| Iteration 30 method accuracy | Not measured | No defensible live trigger / end-to-end validation |

**Decision: ARCHITECTURAL BLOCKER REQUIRES NEW STUDY.**
"""
    (RESULTS / "ITER30_PRESENTATION_TABLE.md").write_text(table, encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2, sort_keys=True))
