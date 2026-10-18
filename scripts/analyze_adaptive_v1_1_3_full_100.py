#!/usr/bin/env python3
"""Create the required reports for the frozen Adaptive v1.1.3 full-100 run."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from _bootstrap import PROJECT_ROOT


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_3" / "full_100"
BASELINE_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "cor_phase1_baseline_20261014_fixed"
DATASETS = ("cwq_branch_2", "grailqa_branch_2")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_text(name: str, content: str) -> None:
    (OUTPUT_ROOT / name).write_text(content.rstrip() + "\n", encoding="utf-8")


def mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    return sum(values) / len(values) if values else 0.0


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def f4(value: float) -> str:
    return f"{value:.4f}"


def correct(row: dict[str, Any], key: str = "hits_at_1") -> bool:
    return bool(row.get(key, 0))


def rows_for(rows: list[dict[str, Any]], dataset: str | None = None) -> list[dict[str, Any]]:
    return [row for row in rows if dataset is None or row.get("dataset") == dataset]


def metric_line(label: str, rows: list[dict[str, Any]], hit_key: str = "hits_at_1", f1_key: str = "f1") -> str:
    errors = sum(1 for row in rows if row.get("error"))
    return f"| {label} | {len(rows)} | {pct(mean(rows, hit_key))} | {f4(mean(rows, f1_key))} | {errors} | {mean(rows, 'llm_calls'):.2f} | {mean(rows, 'kg_queries'):.2f} | {mean_tokens(rows):.1f} | {mean(rows, 'latency_ms'):.1f} |"


def mean_tokens(rows: list[dict[str, Any]]) -> float:
    values = []
    for row in rows:
        tokens = row.get("tokens")
        if isinstance(tokens, dict):
            values.append(float(tokens.get("total", 0) or 0))
            continue
        usage = row.get("token_usage") or {}
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
            if total in (None, 0, ""):
                total = (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)
            values.append(float(total or 0))
        else:
            values.append(0.0)
    return sum(values) / len(values) if values else 0.0


def grouped_metrics(adaptive: list[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Dataset | N | CoR H@1 | CoR F1 | Adaptive native H@1 | Adaptive native F1 | Adaptive final H@1 | Adaptive final F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in (*DATASETS, "overall"):
        current = rows_for(adaptive, None if dataset == "overall" else dataset)
        base = [baseline[str(row["question_id"])] for row in current]
        lines.append(
            f"| {dataset} | {len(current)} | {pct(mean(base, 'hits_at_1'))} | {f4(mean(base, 'f1'))} | "
            f"{pct(mean(current, 'native_hits_at_1'))} | {f4(mean(current, 'native_f1'))} | "
            f"{pct(mean(current, 'final_hits_at_1'))} | {f4(mean(current, 'final_f1'))} |"
        )
    return "\n".join(lines)


def paired_rows(adaptive: list[dict[str, Any]], baseline: dict[str, dict[str, Any]], native: bool = False) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    key = "native_hits_at_1" if native else "final_hits_at_1"
    for row in adaptive:
        base = baseline[str(row["question_id"])]
        base_ok = correct(base)
        adaptive_ok = correct(row, key)
        if base_ok and adaptive_ok:
            group = "BOTH_CORRECT"
        elif base_ok and not adaptive_ok:
            group = "COR_ONLY_CORRECT"
        elif not base_ok and adaptive_ok:
            group = "ADAPTIVE_ONLY_CORRECT"
        else:
            group = "BOTH_WRONG"
        groups[group].append({"adaptive": row, "baseline": base})
    return groups


def cause_for(item: dict[str, Any], native: bool = False) -> str:
    row = item["adaptive"]
    base = item["baseline"]
    if row.get("error"):
        return "runtime_error"
    if not native and row.get("fallback_used"):
        if correct(row, "final_hits_at_1") and not correct(row, "native_hits_at_1"):
            return "fallback_rescue"
        return "fallback_or_native_merge_failure"
    structure = str(row.get("router_structure", ""))
    if structure == "BRANCH_FINAL":
        if row.get("branch_failures"):
            return "branch_execution_failure"
        if row.get("merged_candidate_count", 0) == 0:
            return "empty_branch_merge"
        return "branch_routing_or_grounding"
    if structure == "BRANCH_INTERMEDIATE":
        return "unsupported_branch_intermediate_continuation"
    if not row.get("relation_chain"):
        return "entity_or_relation_grounding"
    if not correct(row, "native_hits_at_1") and correct(base):
        return "native_relation_or_answer_grounding"
    return "answer_selection_or_grounding"


def disagreement_table(items: list[dict[str, Any]], native: bool = False) -> str:
    if not items:
        return "_None._"
    lines = [
        "| Dataset | ID | Structure | Native/final mode | CoR prediction | Adaptive prediction | Cause |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in items:
        row, base = item["adaptive"], item["baseline"]
        pred = row.get("native_prediction", []) if native else row.get("final_prediction", row.get("prediction", []))
        lines.append(
            f"| {row.get('dataset')} | {row.get('question_id')} | {row.get('router_structure')} | "
            f"{row.get('native_execution_mode')}/{row.get('final_execution_mode')} | "
            f"{str(base.get('prediction', []))[:120]} | {str(pred)[:120]} | {cause_for(item, native)} |"
        )
    return "\n".join(lines)


def routing_report(adaptive: list[dict[str, Any]]) -> str:
    lines = ["# Routing analysis", "", "Counts are computed from the 100 frozen Adaptive v1.1.3 rows.", ""]
    lines += ["| Dataset | LINEAR | BRANCH_FINAL | BRANCH_INTERMEDIATE | UNCERTAIN |", "|---|---:|---:|---:|---:|"]
    for dataset in (*DATASETS, "overall"):
        rows = rows_for(adaptive, None if dataset == "overall" else dataset)
        counts = Counter(str(row.get("router_structure", "")) for row in rows)
        lines.append(f"| {dataset} | {counts['LINEAR']} | {counts['BRANCH_FINAL']} | {counts['BRANCH_INTERMEDIATE']} | {counts['UNCERTAIN']} |")
    lines += ["", "## Branch-final execution", "", "| Dataset | Branch-final rows | Native H@1 correct | Final H@1 correct | Fallback rows | Self-sufficient rows | Self-sufficiency |", "|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in (*DATASETS, "overall"):
        rows = [row for row in rows_for(adaptive, None if dataset == "overall" else dataset) if row.get("router_structure") == "BRANCH_FINAL"]
        native_success = sum(correct(row, "native_hits_at_1") for row in rows)
        final_success = sum(correct(row, "final_hits_at_1") for row in rows)
        fallbacks = sum(bool(row.get("fallback_used")) for row in rows)
        self_sufficient = sum(not bool(row.get("fallback_used")) for row in rows)
        rate = self_sufficient / len(rows) if rows else 0.0
        lines.append(f"| {dataset} | {len(rows)} | {native_success} | {final_success} | {fallbacks} | {self_sufficient} | {pct(rate)} |")
    return "\n".join(lines)


def fallback_report(adaptive: list[dict[str, Any]]) -> str:
    lines = ["# Fallback analysis", "", "Fallback is reported separately from native routing. A fallback rescue is a row where native H@1 is wrong and final H@1 is correct.", "", "| Dataset | Adaptive rows | Fallback rows | Fallback rate | Fallback correct | Fallback rescue |", "|---|---:|---:|---:|---:|---:|"]
    for dataset in (*DATASETS, "overall"):
        rows = rows_for(adaptive, None if dataset == "overall" else dataset)
        fallback = [row for row in rows if row.get("fallback_used")]
        rescue = [row for row in fallback if not correct(row, "native_hits_at_1") and correct(row, "final_hits_at_1")]
        lines.append(f"| {dataset} | {len(rows)} | {len(fallback)} | {pct(len(fallback) / len(rows) if rows else 0)} | {sum(correct(row, 'final_hits_at_1') for row in fallback)} | {len(rescue)} |")
    reasons = Counter(str(row.get("fallback_reason") or "unspecified") for row in adaptive if row.get("fallback_used"))
    lines += ["", "## Fallback reasons", ""]
    if reasons:
        lines += ["| Reason | Count |", "|---|---:|"] + [f"| {reason} | {count} |" for reason, count in reasons.most_common()]
    else:
        lines.append("_No fallback rows._")
    return "\n".join(lines)


def validation_report(adaptive: list[dict[str, Any]]) -> str:
    failures = [failure for row in adaptive for failure in (row.get("validation_failures") or [])]
    answer_type = sum("answer" in str(value).casefold() or "type" in str(value).casefold() for value in failures)
    property_mismatch = sum("property" in str(value).casefold() for value in failures)
    rejected = sum(not bool(row.get("validation_passed", False)) for row in adaptive)
    safe_redirect = sum(not bool(row.get("validation_passed", False)) and str(row.get("final_execution_mode", "")).startswith("cor") for row in adaptive)
    lines = ["# Semantic validation analysis", "", "| Measure | Count |", "|---|---:|", f"| Rows | {len(adaptive)} |", f"| Validation failures | {len(failures)} |", f"| Answer-type mismatch mentions | {answer_type} |", f"| Constraint-property mismatch mentions | {property_mismatch} |", f"| Rejected branch plans | {rejected} |", f"| Safe redirects to CoR | {safe_redirect} |", "", "## Failure details", ""]
    if failures:
        lines += ["| ID | Structure | Failures |", "|---|---|---|"]
        lines += [f"| {row.get('question_id')} | {row.get('router_structure')} | {str(row.get('validation_failures'))[:240]} |" for row in adaptive if row.get("validation_failures")]
    else:
        lines.append("No semantic validation failures were recorded.")
    return "\n".join(lines)


def intermediate_report(adaptive: list[dict[str, Any]]) -> str:
    rows = [row for row in adaptive if row.get("router_structure") == "BRANCH_INTERMEDIATE"]
    lines = ["# BRANCH_INTERMEDIATE analysis", "", "The frozen v1.1.3 executor does not implement branch→merge→continue. These rows are reported as observed behavior, not retrofitted during Phase 1.", "", f"Total BRANCH_INTERMEDIATE rows: **{len(rows)}**", ""]
    if rows:
        lines += ["| Dataset | ID | Native mode | Final mode | H@1 | F1 | Fallback | Relation chain |", "|---|---|---|---|---:|---:|---|---|"]
        for row in rows:
            lines.append(f"| {row.get('dataset')} | {row.get('question_id')} | {row.get('native_execution_mode')} | {row.get('final_execution_mode')} | {row.get('final_hits_at_1')} | {f4(float(row.get('final_f1', 0) or 0))} | {row.get('fallback_used')} | {str(row.get('relation_chain', []))[:160]} |")
    else:
        lines.append("No BRANCH_INTERMEDIATE rows were routed.")
    return "\n".join(lines)


def grounding_category(row: dict[str, Any]) -> str:
    if row.get("error"):
        return "runtime_error"
    if row.get("validation_failures"):
        return "semantic_validation"
    if row.get("fallback_used") and not correct(row, "final_hits_at_1"):
        return "fallback_wrong"
    if row.get("router_structure") == "BRANCH_FINAL" and (row.get("branch_failures") or int(row.get("merged_candidate_count", 0) or 0) == 0):
        return "branch_execution_or_merge"
    if not row.get("relation_chain"):
        return "entity_grounding"
    if row.get("router_structure") == "BRANCH_INTERMEDIATE":
        return "unsupported_branch_intermediate"
    return "relation_or_answer_grounding"


def grounding_report(adaptive: list[dict[str, Any]]) -> str:
    lines = ["# Grounding failure analysis", "", "Categories are post-hoc diagnostic labels based on recorded traces; they are not gold structure labels and were not available to the runtime router.", "", "| Dataset | Runtime error | Semantic validation | Fallback wrong | Branch/merge | Entity grounding | Intermediate | Relation/answer grounding |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in (*DATASETS, "overall"):
        rows = [row for row in rows_for(adaptive, None if dataset == "overall" else dataset) if not correct(row, "final_hits_at_1")]
        counts = Counter(grounding_category(row) for row in rows)
        lines.append(f"| {dataset} | {counts['runtime_error']} | {counts['semantic_validation']} | {counts['fallback_wrong']} | {counts['branch_execution_or_merge']} | {counts['entity_grounding']} | {counts['unsupported_branch_intermediate']} | {counts['relation_or_answer_grounding']} |")
    wrong = [row for row in adaptive if not correct(row, "final_hits_at_1")]
    dominant = Counter(grounding_category(row) for row in wrong).most_common(1)
    lines += ["", f"Dominant remaining final-error category: **{dominant[0][0] if dominant else 'none'}** ({dominant[0][1] if dominant else 0} rows).", "", "## Wrong final-answer cases", ""]
    if wrong:
        lines += ["| Dataset | ID | Structure | Category | Prediction | Gold |", "|---|---|---|---|---|---|"]
        for row in wrong:
            lines.append(f"| {row.get('dataset')} | {row.get('question_id')} | {row.get('router_structure')} | {grounding_category(row)} | {str(row.get('final_prediction', row.get('prediction', [])))[:140]} | {str(row.get('gold_answers', []))[:140]} |")
    else:
        lines.append("No final-answer errors.")
    return "\n".join(lines)


def cost_report(adaptive: list[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> str:
    lines = ["# Cost and efficiency analysis", "", "Token counts are the recorded prompt/output usage. No API dollar price is inferred because the configured model/provider price can change; this report compares resource consumption.", "", "| Dataset | Method | N | Avg LLM calls | Avg KG queries | Avg tokens | Avg latency (ms) |", "|---|---|---:|---:|---:|---:|---:|"]
    for dataset in (*DATASETS, "overall"):
        ar = rows_for(adaptive, None if dataset == "overall" else dataset)
        br = [baseline[str(row["question_id"])] for row in ar]
        lines.append(f"| {dataset} | CoR | {len(br)} | {mean(br, 'llm_calls'):.2f} | {mean(br, 'kg_queries'):.2f} | {mean_tokens(br):.1f} | {mean(br, 'latency_ms'):.1f} |")
        lines.append(f"| {dataset} | Adaptive v1.1.3 final | {len(ar)} | {mean(ar, 'llm_calls'):.2f} | {mean(ar, 'kg_queries'):.2f} | {mean_tokens(ar):.1f} | {mean(ar, 'latency_ms'):.1f} |")
    return "\n".join(lines)


def historical_report(adaptive: list[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> str:
    overall = rows_for(adaptive)
    cor_h1 = mean(list(baseline.values()), "hits_at_1")
    cor_f1 = mean(list(baseline.values()), "f1")
    lines = [
        "# Historical method-development comparison",
        "",
        "The full-100 values below are computed from the frozen paired sample. Historical diagnostic values are included for context only and were not rerun here.",
        "",
        "| Method/version | Scope | H@1 | F1 | Status |",
        "|---|---|---:|---:|---|",
        f"| CoR baseline | frozen 100 | {pct(cor_h1)} | {f4(cor_f1)} | paired baseline, not rerun |",
        "| Branch-v1 | earlier Phase 1 diagnostics | see prior report | see prior report | historical comparison only |",
        "| Adaptive v1.1.1 | diagnostic-36 | see prior report | see prior report | historical comparison only |",
        "| Adaptive v1.1.2 | diagnostic/fresh subsets | see prior reports | see prior reports | historical comparison only |",
        f"| Adaptive v1.1.3 native | frozen 100 | {pct(mean(overall, 'native_hits_at_1'))} | {f4(mean(overall, 'native_f1'))} | current run |",
        f"| Adaptive v1.1.3 final | frozen 100 | {pct(mean(overall, 'final_hits_at_1'))} | {f4(mean(overall, 'final_f1'))} | current run |",
        "",
        "Prior reports remain the authoritative source for their differently sized diagnostic samples; those values must not be treated as directly pooled with this 100-question result.",
    ]
    return "\n".join(lines)


def summary_report(adaptive: list[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> str:
    overall = rows_for(adaptive)
    cor_h1, cor_f1 = mean(list(baseline.values()), "hits_at_1"), mean(list(baseline.values()), "f1")
    final_h1, final_f1 = mean(overall, "final_hits_at_1"), mean(overall, "final_f1")
    native_h1, native_f1 = mean(overall, "native_hits_at_1"), mean(overall, "native_f1")
    paired = paired_rows(adaptive, baseline)
    gain = final_h1 - cor_h1
    if gain >= 0.05 and native_h1 >= cor_h1 - 0.02 and len(paired["ADAPTIVE_ONLY_CORRECT"]) >= len(paired["COR_ONLY_CORRECT"]):
        classification = "PHASE1_SUCCESS"
    elif gain > 0 or any(mean(rows_for(adaptive, dataset), "final_hits_at_1") > mean([baseline[str(row["question_id"])] for row in rows_for(adaptive, dataset)], "hits_at_1") for dataset in DATASETS):
        classification = "PARTIAL_SUCCESS"
    else:
        classification = "NO_CLEAR_IMPROVEMENT"
    branch_rows = [row for row in adaptive if row.get("router_structure") == "BRANCH_FINAL"]
    fallback_rows = [row for row in adaptive if row.get("fallback_used")]
    return f"""# Adaptive v1.1.3 full-100 summary

## Scope and protocol

This is the single frozen Phase 1 evaluation on the original 50 CWQ branch_2 and 50 GrailQA branch_2 questions. The CoR baseline was read from `{BASELINE_ROOT}` and was not rerun. v1.1.4 was excluded from the main experiment.

## Accuracy

{grouped_metrics(adaptive, baseline)}

## Paired outcomes

| Outcome | Count |
|---|---:|
| BOTH_CORRECT | {len(paired['BOTH_CORRECT'])} |
| COR_ONLY_CORRECT | {len(paired['COR_ONLY_CORRECT'])} |
| ADAPTIVE_ONLY_CORRECT | {len(paired['ADAPTIVE_ONLY_CORRECT'])} |
| BOTH_WRONG | {len(paired['BOTH_WRONG'])} |

## Routing and fallback

- BRANCH_FINAL rows: **{len(branch_rows)}** / 100.
- Fallback rows: **{len(fallback_rows)}** / 100 ({pct(len(fallback_rows) / len(adaptive) if adaptive else 0)}).
- Native Adaptive v1.1.3: **{pct(native_h1)} H@1**, **{f4(native_f1)} F1**.
- Final Adaptive v1.1.3: **{pct(final_h1)} H@1**, **{f4(final_f1)} F1**.
- Frozen CoR: **{pct(cor_h1)} H@1**, **{f4(cor_f1)} F1**.

## Required interpretation

The final-vs-CoR H@1 difference is **{gain * 100:+.2f} percentage points**. Native and fallback-assisted results are reported separately; paired gains and regressions are documented in `paired_analysis.md`.

Phase 1 is classified as **{classification}** because the full-100 paired result, native performance, routing self-sufficiency, and fallback dependence must be considered together rather than accuracy alone.

## Limitations and stop condition

This run does not implement branch→merge→continue, does not use gold structure at runtime, and does not retune prompts or thresholds after inspecting results. The v1.1.4 targeted extension remains a negative exploratory result and is not part of this Phase 1 score.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=OUTPUT_ROOT / "adaptive_results.jsonl")
    args = parser.parse_args()
    adaptive = read_jsonl(args.results)
    if len(adaptive) != 100:
        raise SystemExit(f"expected 100 adaptive rows, found {len(adaptive)}")
    ids = {(str(row.get("dataset")), str(row.get("question_id", row.get("id", "")))) for row in adaptive}
    if len(ids) != 100:
        raise SystemExit("adaptive result IDs are not unique")
    sample = read_json(OUTPUT_ROOT / "sample_ids.json")
    expected = {(dataset, str(question_id)) for dataset in DATASETS for question_id in sample[dataset]}
    if ids != expected:
        raise SystemExit("adaptive result IDs do not match sample_ids.json")

    baseline: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        for row in read_jsonl(BASELINE_ROOT / f"{dataset}_results.jsonl"):
            baseline[str(row["id"])] = row
    if len(baseline) != 100:
        raise SystemExit(f"expected 100 baseline rows, found {len(baseline)}")

    grouped = paired_rows(adaptive, baseline)
    native_grouped = paired_rows(adaptive, baseline, native=True)
    all_disagreements = grouped["COR_ONLY_CORRECT"] + grouped["ADAPTIVE_ONLY_CORRECT"]
    native_disagreements = native_grouped["COR_ONLY_CORRECT"] + native_grouped["ADAPTIVE_ONLY_CORRECT"]

    paired_text = ["# Paired CoR vs Adaptive v1.1.3 analysis", "", "Final/post-fallback paired H@1 classification is based on the exact frozen question IDs.", "", "| Outcome | Count |", "|---|---:|"]
    paired_text += [f"| {name} | {len(grouped[name])} |" for name in ("BOTH_CORRECT", "COR_ONLY_CORRECT", "ADAPTIVE_ONLY_CORRECT", "BOTH_WRONG")]
    paired_text += ["", "## Final disagreements", "", disagreement_table(all_disagreements), "", "## Cause counts for final disagreements", ""]
    cause_counts = Counter(cause_for(item) for item in all_disagreements)
    paired_text += ["| Cause | Count |", "|---|---:|"] + [f"| {cause} | {count} |" for cause, count in cause_counts.most_common()]
    write_text("paired_analysis.md", "\n".join(paired_text))

    native_text = ["# Native versus final Adaptive v1.1.3", "", "| Dataset | Native H@1 | Native F1 | Final H@1 | Final F1 | Native→final rescues | Final regressions from native |", "|---|---:|---:|---:|---:|---:|"]
    for dataset in (*DATASETS, "overall"):
        rows = rows_for(adaptive, None if dataset == "overall" else dataset)
        rescues = sum(not correct(row, "native_hits_at_1") and correct(row, "final_hits_at_1") for row in rows)
        regressions = sum(correct(row, "native_hits_at_1") and not correct(row, "final_hits_at_1") for row in rows)
        native_text.append(f"| {dataset} | {pct(mean(rows, 'native_hits_at_1'))} | {f4(mean(rows, 'native_f1'))} | {pct(mean(rows, 'final_hits_at_1'))} | {f4(mean(rows, 'final_f1'))} | {rescues} | {regressions} |")
    native_text += ["", "## Native disagreements with CoR", "", disagreement_table(native_disagreements, native=True)]
    write_text("native_vs_final.md", "\n".join(native_text))

    write_text("routing_analysis.md", routing_report(adaptive))
    write_text("fallback_analysis.md", fallback_report(adaptive))
    write_text("semantic_validation_analysis.md", validation_report(adaptive))
    write_text("branch_intermediate_analysis.md", intermediate_report(adaptive))
    write_text("grounding_analysis.md", grounding_report(adaptive))
    write_text("cost_analysis.md", cost_report(adaptive, baseline))
    write_text("historical_comparison.md", historical_report(adaptive, baseline))
    write_text("summary.md", summary_report(adaptive, baseline))
    (OUTPUT_ROOT.parent / "FULL_100_PHASE1_REPORT.md").write_text(summary_report(adaptive, baseline) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(adaptive), "output": str(OUTPUT_ROOT), "report": str(OUTPUT_ROOT.parent / "FULL_100_PHASE1_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
