#!/usr/bin/env python3
"""Create the paired Phase 1 comparison for baseline CoR and branch-v1."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "outputs/experiments/cor_phase1_baseline_20261014_fixed"
BRANCH_ROOT = REPO_ROOT / "outputs/experiments/phase1/branch_v1"
OUTPUT_ROOT = REPO_ROOT / "outputs/experiments/phase1/comparison"

DATASETS = ("cwq_branch_2", "grailqa_branch_2")


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
	rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
	ids = [row["id"] for row in rows]
	if len(ids) != len(set(ids)):
		raise ValueError(f"Duplicate IDs in {path}")
	return {row["id"]: row for row in rows}


def metric(rows: list[dict[str, Any]], field: str) -> float:
	return statistics.mean(float(row.get(field, 0.0)) for row in rows)


def usage_metric(rows: list[dict[str, Any]], field: str) -> float:
	return statistics.mean(float(row.get("token_usage", {}).get(field, 0.0)) for row in rows)


def metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
	return {
		"n": len(rows),
		"hits_at_1": metric(rows, "hits_at_1"),
		"f1": metric(rows, "f1"),
		"error_rate": statistics.mean(bool(row.get("error")) for row in rows),
		"llm_calls": metric(rows, "llm_calls"),
		"kg_queries": metric(rows, "kg_queries"),
		"tokens": usage_metric(rows, "total_tokens"),
		"latency_ms": metric(rows, "latency_ms"),
		"branch_failures": sum(bool(row.get("branch_failures")) for row in rows),
		"merged_zero": sum(row.get("merged_candidate_count", 0) == 0 for row in rows),
		"shared_root_rows": sum(bool(row.get("branch_decomposition", {}).get("shared_root_mode")) for row in rows),
	}


def outcome(baseline: dict[str, Any], branch: dict[str, Any]) -> str:
	base_correct = bool(baseline.get("hits_at_1"))
	branch_correct = bool(branch.get("hits_at_1"))
	if base_correct and branch_correct:
		return "BOTH_CORRECT"
	if not base_correct and not branch_correct:
		return "BOTH_WRONG"
	if branch_correct:
		return "OURS_ONLY_CORRECT"
	return "COR_ONLY_CORRECT"


def fmt(value: float, percent: bool = False) -> str:
	if percent:
		return f"{value * 100:.1f}%"
	return f"{value:.2f}"


def table_for_dataset(name: str, base: dict[str, float], branch: dict[str, float]) -> str:
	rows = [
		("Examples", str(int(base["n"])), str(int(branch["n"])), ""),
		("Hits@1", fmt(base["hits_at_1"], True), fmt(branch["hits_at_1"], True), fmt(branch["hits_at_1"] - base["hits_at_1"], True)),
		("F1", fmt(base["f1"]), fmt(branch["f1"]), fmt(branch["f1"] - base["f1"])),
		("Error rate", fmt(base["error_rate"], True), fmt(branch["error_rate"], True), fmt(branch["error_rate"] - base["error_rate"], True)),
		("Avg LLM calls", fmt(base["llm_calls"]), fmt(branch["llm_calls"]), fmt(branch["llm_calls"] - base["llm_calls"])),
		("Avg KG queries", fmt(base["kg_queries"]), fmt(branch["kg_queries"]), fmt(branch["kg_queries"] - base["kg_queries"])),
		("Avg total tokens", fmt(base["tokens"]), fmt(branch["tokens"]), fmt(branch["tokens"] - base["tokens"])),
		("Avg latency (ms)", fmt(base["latency_ms"]), fmt(branch["latency_ms"]), fmt(branch["latency_ms"] - base["latency_ms"])),
	]
	lines = [f"### {name}", "", "| Metric | Frozen CoR baseline | Branch-v1 | Branch-v1 − baseline |", "|---|---:|---:|---:|"]
	lines.extend(f"| {label} | {left} | {right} | {delta} |" for label, left, right, delta in rows)
	return "\n".join(lines)


def representative_block(
	name: str,
	base: dict[str, dict[str, Any]],
	branch: dict[str, dict[str, Any]],
) -> str:
	paired = [(item_id, base[item_id], branch[item_id]) for item_id in base]
	ours_only = [(item_id, b, o) for item_id, b, o in paired if outcome(b, o) == "OURS_ONLY_CORRECT"]
	cor_only = [(item_id, b, o) for item_id, b, o in paired if outcome(b, o) == "COR_ONLY_CORRECT"]
	f1_improved = [(item_id, b, o) for item_id, b, o in paired if bool(b.get("hits_at_1")) == bool(o.get("hits_at_1")) and o.get("f1", 0) > b.get("f1", 0)]
	f1_regressed = [(item_id, b, o) for item_id, b, o in paired if bool(b.get("hits_at_1")) == bool(o.get("hits_at_1")) and o.get("f1", 0) < b.get("f1", 0)]

	def render(title: str, entries: list[tuple[str, dict[str, Any], dict[str, Any]]]) -> list[str]:
		lines = [f"#### {title}", ""]
		if not entries:
			return lines + ["None.", ""]
		for item_id, old, new in entries[:5]:
			lines.extend([
				f"- `{item_id}` — {new.get('question', old.get('question', ''))}",
				f"  - baseline: Hits@1={old.get('hits_at_1')}, F1={old.get('f1'):.4f}; branch-v1: Hits@1={new.get('hits_at_1')}, F1={new.get('f1'):.4f}",
				f"  - branch candidates: {new.get('branch_candidate_counts')}, merged: {new.get('merged_candidate_count')}",
			])
		return lines + [""]

	return "\n".join(
		[f"### {name} representative paired cases", ""]
		+ render("Branch-v1 wins", ours_only)
		+ render("Baseline-only wins", cor_only)
		+ render("F1 improved with unchanged Hits@1", f1_improved)
		+ render("F1 regressed with unchanged Hits@1", f1_regressed)
	)


def main() -> None:
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	all_paired: list[dict[str, Any]] = []
	sections: list[str] = []
	summary_sections: list[str] = []
	analysis_sections: list[str] = []

	for dataset in DATASETS:
		base = load_rows(BASELINE_ROOT / f"{dataset}_results.jsonl")
		branch = load_rows(BRANCH_ROOT / f"{dataset}_results.jsonl")
		if set(base) != set(branch):
			raise ValueError(f"Frozen ID mismatch for {dataset}: {set(base) ^ set(branch)}")

		for item_id in base:
			old = base[item_id]
			new = branch[item_id]
			all_paired.append({
				"dataset": dataset,
				"id": item_id,
				"question": new.get("question", old.get("question", "")),
				"outcome": outcome(old, new),
				"baseline": {
					"hits_at_1": old.get("hits_at_1"),
					"f1": old.get("f1"),
					"llm_calls": old.get("llm_calls"),
					"kg_queries": old.get("kg_queries"),
					"token_usage": old.get("token_usage", {}),
					"latency_ms": old.get("latency_ms"),
					"error": old.get("error"),
				},
				"branch_v1": {
					"hits_at_1": new.get("hits_at_1"),
					"f1": new.get("f1"),
					"llm_calls": new.get("llm_calls"),
					"kg_queries": new.get("kg_queries"),
					"token_usage": new.get("token_usage", {}),
					"latency_ms": new.get("latency_ms"),
					"error": new.get("error"),
					"branches_created": new.get("branches_created"),
					"branch_candidate_counts": new.get("branch_candidate_counts"),
					"merged_candidate_count": new.get("merged_candidate_count"),
					"branch_failures": new.get("branch_failures"),
				},
			})

		base_metrics = metrics(list(base.values()))
		branch_metrics = metrics(list(branch.values()))
		counts = {label: 0 for label in ("BOTH_CORRECT", "BOTH_WRONG", "OURS_ONLY_CORRECT", "COR_ONLY_CORRECT")}
		for item_id in base:
			counts[outcome(base[item_id], branch[item_id])] += 1
		table_section = table_for_dataset(dataset, base_metrics, branch_metrics)
		sections.append(table_section)
		sections.append("")
		outcome_section = (
			"Paired outcomes: "
			+ ", ".join(f"{key}={value}" for key, value in counts.items())
			+ "."
		)
		sections.append(outcome_section)
		sections.append("")
		sections.append(representative_block(dataset, base, branch))
		summary_sections.extend([table_section, outcome_section])
		analysis_sections.append(
			f"### {dataset}\n\n"
			f"The branch-v1 run used {int(branch_metrics['shared_root_rows'])} shared-root compatibility rows and "
			f"{int(branch_metrics['n'] - branch_metrics['shared_root_rows'])} explicit two-root rows. "
			f"It recorded {int(branch_metrics['branch_failures'])} branch failures and "
			f"{int(branch_metrics['merged_zero'])} rows with zero merged candidates.\n"
		)

	(OUTPUT_ROOT / "paired_results.jsonl").write_text(
		"\n".join(json.dumps(row, ensure_ascii=False) for row in all_paired) + "\n",
		encoding="utf-8",
	)

	comparison = """# Branch-v1 versus frozen CoR baseline

This is the paired comparison for the exact 50 CWQ and 50 GrailQA IDs used by the frozen Phase 1 baseline. Every ID is present exactly once in both systems. The baseline is the unchanged CoR implementation; branch-v1 adds explicit two-branch state, focused branch execution for two runtime topic entities, and an `AND` candidate merge.

""" + "\n\n".join(sections) + "\n\n## Interpretation\n\nThe prototype does not support the hypothesis on this frozen sample. It produces no CWQ branch-v1-only correct cases and loses six baseline-correct CWQ cases. On GrailQA it gains three cases and loses four, leaving a one-example net decrease. The average LLM calls, KG queries, tokens, and latency all increase because explicit branches execute CoR separately. These results do not justify Phase 2 expansion.\n"
	(OUTPUT_ROOT / "branch_v1_vs_cor.md").write_text(comparison, encoding="utf-8")

	paired_analysis = """# Paired Phase 1 analysis

## Outcome definitions

- `BOTH_CORRECT`: both systems have Hits@1=1.
- `BOTH_WRONG`: both systems have Hits@1=0.
- `OURS_ONLY_CORRECT`: branch-v1 is correct and the frozen baseline is wrong.
- `COR_ONLY_CORRECT`: the frozen baseline is correct and branch-v1 is wrong.

""" + "\n".join(analysis_sections) + """

## Decision

The minimal branch-aware prototype is operational and produces explicit branch traces, but the Phase 1 result is negative: branch-v1 is not a drop-in improvement over unchanged CoR. The main next step should be diagnosis of branch decomposition and answer-set preservation, not branch_3, nested branching, or broader experimentation.
"""
	(OUTPUT_ROOT / "paired_analysis.md").write_text(paired_analysis, encoding="utf-8")

	# The final report is intentionally concise; detailed paired rows remain in JSONL.
	final_report = """# First two-branch Phase 1 experiment report

## Scope

- Frozen evaluation: 50 CWQ `branch_2` examples and 50 GrailQA `branch_2` examples.
- Baseline: unchanged CoR, using the frozen baseline outputs.
- Intervention: minimal explicit two-branch relation-centric v1 with `BranchState`, `RelationGraphState`, and `MERGE_AND`.
- Runtime uses only linked topic entities and the question; no gold SPARQL, S-expression, structural label, answer ID, or gold branch count.
- No branch_3, nested, OR, NOT, aggregation, or Phase 2 work was started.

## Results

""" + "\n\n".join(summary_sections) + """

## Overall conclusion

The prototype runs end-to-end with zero row-level errors and records explicit branch decomposition, traces, candidate counts, and merges. However, it does not improve the frozen Phase 1 evaluation: CWQ decreases from 86.0% to 74.0% Hits@1 and GrailQA changes from 62.0% to 60.0%. It also increases average execution cost. The current hypothesis is therefore not supported, and the implementation should not advance to Phase 2 without a targeted diagnosis of regressions and a pre-registered refinement.

Detailed paired outcomes are in `paired_results.jsonl`, with expanded analysis in `paired_analysis.md` and `branch_v1_vs_cor.md`.
"""
	(OUTPUT_ROOT.parent / "FIRST_2_BRANCH_EXPERIMENT_REPORT.md").write_text(final_report, encoding="utf-8")

	print(f"Wrote {len(all_paired)} paired rows to {OUTPUT_ROOT}")


if __name__ == "__main__":
	main()
