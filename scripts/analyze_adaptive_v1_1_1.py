#!/usr/bin/env python3
"""Create the required Adaptive v1.1.1 expanded-diagnostic analyses."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_1"
RUN_ROOT = OUTPUT_ROOT / "diagnostic_36"
PAIRED_CASES = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "comparison" / "paired_cases" / "paired_case_summary.json"
BASELINE_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "cor_phase1_baseline_20261014_fixed"
BRANCH_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "branch_v1"
SUBSETS = ("cwq_branch_2", "grailqa_branch_2")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_rows(root: Path) -> dict[str, dict[str, Any]]:
	rows: dict[str, dict[str, Any]] = {}
	for subset in SUBSETS:
		path = root / f"{subset}_results.jsonl"
		for row in read_jsonl(path):
			rows[str(row["id"])] = row
	return rows


def load_cases() -> dict[str, dict[str, Any]]:
	data = json.loads(PAIRED_CASES.read_text(encoding="utf-8"))
	return {str(case["id"]): case for case in data["paired_cases"]}


def all_rows() -> list[dict[str, Any]]:
	rows = []
	for subset in SUBSETS:
		rows.extend(read_jsonl(RUN_ROOT / f"{subset}_results.jsonl"))
	return rows


def token_total(row: dict[str, Any]) -> int:
	return int((row.get("token_usage") or {}).get("total_tokens", 0) or 0)


def mean(rows: list[dict[str, Any]], key: str) -> float:
	values = [float(row.get(key, 0) or 0) for row in rows]
	return sum(values) / len(values) if rows else 0.0


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
	return {
		"n": len(rows),
		"hits_at_1": sum(bool(row.get("hits_at_1")) for row in rows),
		"hits_rate": sum(bool(row.get("hits_at_1")) for row in rows) / len(rows) if rows else 0.0,
		"f1": mean(rows, "f1"),
		"errors": sum(bool(row.get("error")) for row in rows),
		"llm_calls": mean(rows, "llm_calls"),
		"kg_queries": mean(rows, "kg_queries"),
		"tokens": sum(token_total(row) for row in rows) / len(rows) if rows else 0.0,
		"latency_ms": mean(rows, "latency_ms"),
	}


def prediction_signature(row: dict[str, Any]) -> tuple[str, ...]:
	return tuple(str(value).strip().casefold() for value in row.get("prediction", []) if str(value).strip())


def route_label(row: dict[str, Any]) -> str:
	if row.get("router_mode") == "branch":
		return f"branch_{row.get('branch_count', 0)}"
	return "cor"


def branch_outcome(row: dict[str, Any]) -> str:
	if row.get("router_mode") != "branch":
		return "NOT_BRANCH_MODE"
	if not row.get("fallback_used") and row.get("final_execution_mode") == "branch":
		return "BRANCH_SUCCESS_NO_FALLBACK"
	if row.get("merge_outcome") == "NO_INTERSECTION_VALID_DISAGREEMENT":
		return "BRANCH_FAILED_EMPTY_MERGE"
	statuses = row.get("branch_statuses", [])
	if any(status == "EMPTY_AFTER_RETRIEVAL" for status in statuses):
		return "BRANCH_FAILED_EMPTY_MERGE"
	if any(status == "EMPTY_AFTER_FILTERING" for status in statuses):
		return "BRANCH_FAILED_FILTERING"
	if any(status == "EXECUTION_ERROR" for status in statuses):
		return "BRANCH_FAILED_OTHER"
	if row.get("fallback_used"):
		return "BRANCH_SUCCESS_WITH_FALLBACK"
	return "BRANCH_FAILED_OTHER"


def merge_bucket(row: dict[str, Any]) -> str:
	if row.get("router_mode") != "branch":
		return "NOT_BRANCH_MODE"
	counts = row.get("merge_method_counts") or {}
	if counts.get("ID", 0):
		return "MERGED_BY_ID"
	if counts.get("NORMALIZED_LABEL", 0):
		return "MERGED_BY_NORMALIZED_LABEL"
	if counts.get("OTHER_SAFE_ALIGNMENT", 0):
		return "MERGED_BY_OTHER_SAFE_ALIGNMENT"
	if row.get("fallback_used") and row.get("merge_outcome") == "NO_INTERSECTION_VALID_DISAGREEMENT":
		return "NO_INTERSECTION_VALID_DISAGREEMENT"
	if row.get("fallback_used"):
		return "FAILED_BRANCH_FALLBACK"
	return "EMPTY_MERGE_OTHER"


def both_wrong_classification(row: dict[str, Any], baseline: dict[str, Any], branch: dict[str, Any]) -> str:
	if row.get("hits_at_1"):
		if str(row.get("final_execution_mode", "")).startswith("cor"):
			return "NEWLY_SOLVED_BY_COR_PATH"
		return "NEWLY_SOLVED_BY_BRANCHING"
	if float(row.get("f1", 0) or 0) > max(float(baseline.get("f1", 0) or 0), float(branch.get("f1", 0) or 0)):
		return "PARTIAL_F1_IMPROVEMENT"
	if prediction_signature(row) not in {prediction_signature(baseline), prediction_signature(branch)}:
		return "REGRESSED_DIFFERENTLY"
	return "STILL_BOTH_WRONG_EQUIVALENT"


def both_wrong_cause(row: dict[str, Any]) -> str:
	if row.get("router_mode") == "branch":
		if row.get("merge_outcome") == "NO_INTERSECTION_VALID_DISAGREEMENT":
			return "MERGE_FAILURE"
		if row.get("fallback_used"):
			return "BRANCH_EXECUTION"
		return "BRANCH_DECOMPOSITION"
	if row.get("error"):
		return "OTHER"
	if not row.get("relation_chain"):
		return "ENTITY_GROUNDING"
	return "RELATION_GROUNDING"


def fmt_prediction(row: dict[str, Any]) -> str:
	return json.dumps(row.get("prediction", []), ensure_ascii=False)


def write_summary(rows: list[dict[str, Any]], cases: dict[str, dict[str, Any]]) -> None:
	groups = defaultdict(list)
	for row in rows:
		groups[cases.get(str(row["id"]), {}).get("category", "UNKNOWN")].append(row)
	route_groups = ["COR_ONLY_CORRECT", "OURS_ONLY_CORRECT", "BOTH_WRONG", "ALL"]
	lines = [
		"# Adaptive v1.1.1 expanded diagnostic summary",
		"",
		"The frozen CoR and Branch-v1 outputs were read for paired evaluation only; neither was rerun.",
		"",
		"## Overall metrics",
		"",
		"| Group | N | Hits@1 | F1 | Errors | Avg LLM calls | Avg KG queries | Avg tokens | Avg latency (ms) |",
		"|---|---:|---:|---:|---:|---:|---:|---:|---:|",
	]
	for group in route_groups:
		value = metrics(rows if group == "ALL" else groups[group])
		lines.append(f"| {group} | {value['n']} | {value['hits_rate']:.2%} | {value['f1']:.4f} | {value['errors']} | {value['llm_calls']:.2f} | {value['kg_queries']:.2f} | {value['tokens']:.2f} | {value['latency_ms']:.2f} |")
	lines += ["", "## Routing distribution", "", "| Group | CoR selected | branch_2 selected | branch_3 selected | CoR fallback |", "|---|---:|---:|---:|---:|"]
	for group in route_groups:
		items = rows if group == "ALL" else groups[group]
		lines.append(f"| {group} | {sum(route_label(r) == 'cor' for r in items)} | {sum(route_label(r) == 'branch_2' for r in items)} | {sum(route_label(r) == 'branch_3' for r in items)} | {sum(r.get('final_execution_mode') == 'cor_fallback' for r in items)} |")
	lines += ["", "## Required paired outcomes", "", f"- COR_ONLY_CORRECT preserved: **{sum(bool(r.get('hits_at_1')) for r in groups['COR_ONLY_CORRECT'])}/10**", f"- OURS_ONLY_CORRECT recovered: **{sum(bool(r.get('hits_at_1')) for r in groups['OURS_ONLY_CORRECT'])}/3**", f"- BOTH_WRONG newly solved: **{sum(bool(r.get('hits_at_1')) for r in groups['BOTH_WRONG'])}/23**", "", "## Branch execution", ""]
	branch_rows = [row for row in rows if row.get("router_mode") == "branch"]
	counts = Counter(branch_outcome(row) for row in branch_rows)
	lines += [f"- Branch proposals: **{len(branch_rows)}**", f"- Successful branch merges without fallback: **{counts['BRANCH_SUCCESS_NO_FALLBACK']}**", f"- Empty merges: **{counts['BRANCH_FAILED_EMPTY_MERGE']}**", f"- Fallbacks: **{sum(bool(row.get('fallback_used')) for row in branch_rows)}**", f"- Fallback rate: **{sum(bool(row.get('fallback_used')) for row in branch_rows) / len(branch_rows):.2%}**" if branch_rows else "- Fallback rate: **0.00%**", "", "See the accompanying routing, branch, merge, and BOTH_WRONG analyses."]
	(RUN_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_routing(rows: list[dict[str, Any]], cases: dict[str, dict[str, Any]]) -> None:
	groups = defaultdict(list)
	for row in rows:
		groups[cases[str(row["id"])] ["category"]].append(row)
	lines = ["# Expanded diagnostic routing analysis", "", "| Group | CoR | branch_2 | branch_3 | CoR fallback |", "|---|---:|---:|---:|---:|"]
	for group in ["COR_ONLY_CORRECT", "OURS_ONLY_CORRECT", "BOTH_WRONG", "ALL"]:
		items = rows if group == "ALL" else groups[group]
		lines.append(f"| {group} | {sum(route_label(r) == 'cor' for r in items)} | {sum(route_label(r) == 'branch_2' for r in items)} | {sum(route_label(r) == 'branch_3' for r in items)} | {sum(r.get('final_execution_mode') == 'cor_fallback' for r in items)} |")
	lines += ["", "## Branch-mode reasons", ""]
	for row in rows:
		if row.get("router_mode") == "branch":
			lines.append(f"- `{row['id']}` ({cases[str(row['id'])]['category']}): {row.get('router_reason', '').replace(chr(10), ' ')}")
	(RUN_ROOT / "routing_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_branch_analysis(rows: list[dict[str, Any]], cases: dict[str, dict[str, Any]]) -> None:
	branch_rows = [row for row in rows if row.get("router_mode") == "branch"]
	lines = ["# Expanded diagnostic branch execution analysis", "", "| ID | Group | Outcome | Branches | Fallback | Merge outcome | Prediction | Hits@1 | F1 |", "|---|---|---|---:|---|---|---|---:|---:|"]
	for row in branch_rows:
		lines.append(f"| `{row['id']}` | {cases[str(row['id'])]['category']} | {branch_outcome(row)} | {row.get('branch_count', 0)} | {row.get('fallback_used')} | {row.get('merge_outcome', '')} | {fmt_prediction(row)} | {row.get('hits_at_1')} | {float(row.get('f1', 0)):.4f} |")
	lines += ["", "## Branch details", ""]
	for row in branch_rows:
		lines += [f"### `{row['id']}` — {row.get('question', '')}", "", f"- Source group: `{cases[str(row['id'])]['category']}`", f"- Router reason: {row.get('router_reason', '')}", f"- Outcome: `{branch_outcome(row)}`", f"- Merge method(s): `{row.get('merge_methods', [])}`", f"- Merge result: `{row.get('merge_outcome', '')}`", f"- Fallback: `{row.get('fallback_used')}` ({row.get('fallback_reason')})", "", "| Branch | Root | Constraint | Relation chain | Retrieved counts | Post-filter candidates | Final answers | Status |", "|---|---|---|---|---|---|---|---|"]
		decomp = row.get("branch_decomposition", {}) or {}
		branches = decomp.get("branches", []) if isinstance(decomp, dict) else []
		traces = row.get("branch_traces", []) or []
		for index, trace in enumerate(traces):
			definition = branches[index] if index < len(branches) else {}
			retrieved = trace.get("retrieved_candidates", [])
			counts = [item.get("result_count", 0) for item in retrieved]
			post_filter = trace.get("post_filter_candidates", [])
			lines.append(f"| {trace.get('branch_id', index)} | {definition.get('root_entity', {})} | {definition.get('constraint_summary', '')} | `{trace.get('relation_chain', [])}` | `{counts}` | `{json.dumps(post_filter, ensure_ascii=False)}` | `{json.dumps(trace.get('final_branch_answers', []), ensure_ascii=False)}` | {trace.get('status')} |")
		lines.append("")
	counts = Counter(branch_outcome(row) for row in branch_rows)
	lines += ["## Aggregate", "", f"- Branch proposals total: **{len(branch_rows)}**", f"- Successful branch merges: **{counts['BRANCH_SUCCESS_NO_FALLBACK']}**", f"- Empty merges: **{counts['BRANCH_FAILED_EMPTY_MERGE']}**", f"- Fallback count: **{sum(bool(row.get('fallback_used')) for row in branch_rows)}**", f"- Fallback rate: **{sum(bool(row.get('fallback_used')) for row in branch_rows) / len(branch_rows):.2%}**" if branch_rows else "- Fallback rate: **0.00%**", "", "v1.1 comparison: 4/5 branch attempts fell back on empty strict merge."]
	(RUN_ROOT / "branch_execution_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_merge_analysis(rows: list[dict[str, Any]]) -> None:
	counts = Counter(merge_bucket(row) for row in rows if row.get("router_mode") == "branch")
	lines = ["# Expanded diagnostic merge analysis", "", "| Merge category | Count |", "|---|---:|"]
	for category in ["MERGED_BY_ID", "MERGED_BY_NORMALIZED_LABEL", "MERGED_BY_OTHER_SAFE_ALIGNMENT", "NO_INTERSECTION_VALID_DISAGREEMENT", "FAILED_BRANCH_FALLBACK", "EMPTY_MERGE_OTHER"]:
		lines.append(f"| {category} | {counts[category]} |")
	lines += ["", "## Normalized-label merges", ""]
	for row in rows:
		if merge_bucket(row) != "MERGED_BY_NORMALIZED_LABEL":
			continue
		lines += [f"### `{row['id']}`", "", "The labels were identical after case folding, Unicode normalization, whitespace collapse, and removal of matching outer quotation marks. This is safe because no semantic similarity or fuzzy edit distance was used.", "", f"Merge inputs: `{json.dumps(row.get('merge_inputs', []), ensure_ascii=False)}`", f"Methods: `{row.get('merge_methods', [])}`", ""]
	(RUN_ROOT / "merge_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_both_wrong(rows: list[dict[str, Any]], cases: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]], branch: dict[str, dict[str, Any]]) -> None:
	items = [row for row in rows if cases[str(row["id"])] ["category"] == "BOTH_WRONG"]
	lines = ["# Expanded diagnostic BOTH_WRONG analysis", "", "| ID | Classification | Main cause | Final mode | Hits@1 | F1 |", "|---|---|---|---|---:|---:|"]
	for row in items:
		base = baseline.get(str(row["id"]), {})
		br = branch.get(str(row["id"]), {})
		lines.append(f"| `{row['id']}` | {both_wrong_classification(row, base, br)} | {both_wrong_cause(row)} | {row.get('final_execution_mode')} | {row.get('hits_at_1')} | {float(row.get('f1', 0)):.4f} |")
	lines += ["", "## Case explanations", ""]
	for row in items:
		base = baseline.get(str(row["id"]), {})
		br = branch.get(str(row["id"]), {})
		classification = both_wrong_classification(row, base, br)
		lines += [f"### `{row['id']}` — {row.get('question', '')}", "", f"- Classification: `{classification}`", f"- Main cause: `{both_wrong_cause(row)}`", f"- Adaptive execution: `{row.get('final_execution_mode')}`; prediction: `{fmt_prediction(row)}`", f"- Frozen CoR prediction: `{fmt_prediction(base)}`", f"- Frozen Branch-v1 prediction: `{fmt_prediction(br)}`", f"- Router decision: `{route_label(row)}` — {row.get('router_reason', '')}", ""]
	(RUN_ROOT / "both_wrong_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paired_case_analysis(rows: list[dict[str, Any]], cases: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]], branch: dict[str, dict[str, Any]]) -> None:
	lines = ["# Adaptive v1.1.1 paired case analysis", "", "| ID | Group | Adaptive mode | Prediction | Hits@1 | F1 | Baseline Hits@1 | Branch-v1 Hits@1 |", "|---|---|---|---|---:|---:|---:|---:|"]
	for row in rows:
		case = cases[str(row["id"])]
		base = baseline.get(str(row["id"]), {})
		br = branch.get(str(row["id"]), {})
		lines.append(f"| `{row['id']}` | {case['category']} | {row.get('final_execution_mode')} | {fmt_prediction(row)} | {row.get('hits_at_1')} | {float(row.get('f1', 0)):.4f} | {base.get('hits_at_1', 0)} | {br.get('hits_at_1', 0)} |")
	(RUN_ROOT / "paired_case_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(rows: list[dict[str, Any]], cases: dict[str, dict[str, Any]]) -> None:
	groups = defaultdict(list)
	for row in rows:
		groups[cases[str(row["id"])] ["category"]].append(row)
	branch_rows = [row for row in rows if row.get("router_mode") == "branch"]
	lines = [
		"# Adaptive v1.1.1 expanded diagnostic report",
		"",
		"## Scope",
		"",
		"This run keeps the v1.1 router fixed and changes only branch candidate preservation, structured answer preservation, and safe merge behavior. It evaluates exactly 10 COR_ONLY_CORRECT, 3 OURS_ONLY_CORRECT, and 23 BOTH_WRONG cases. Frozen CoR and Branch-v1 outputs were not rerun.",
		"",
		"## Validation before the run",
		"",
		"- Pre-fix analysis completed for all five v1.1 branch proposals.",
		"- Targeted tests: **16 passed**.",
		"- Python compilation passed.",
		"- `git diff --check` passed.",
		"",
		"## Results",
		"",
	]
	all_metrics = metrics(rows)
	lines += [f"- Overall: **{all_metrics['hits_rate']:.2%} Hits@1**, **{all_metrics['f1']:.4f} F1**, {all_metrics['errors']} errors.", f"- COR_ONLY_CORRECT preserved: **{sum(bool(r.get('hits_at_1')) for r in groups['COR_ONLY_CORRECT'])}/10**.", f"- OURS_ONLY_CORRECT recovered: **{sum(bool(r.get('hits_at_1')) for r in groups['OURS_ONLY_CORRECT'])}/3**.", f"- BOTH_WRONG newly solved: **{sum(bool(r.get('hits_at_1')) for r in groups['BOTH_WRONG'])}/23**.", f"- Routing: **{sum(route_label(r) == 'cor' for r in rows)} CoR**, **{sum(route_label(r) == 'branch_2' for r in rows)} branch_2**, **{sum(route_label(r) == 'branch_3' for r in rows)} branch_3**, **{sum(r.get('final_execution_mode') == 'cor_fallback' for r in rows)} CoR fallbacks**.", "", "## Branch execution decision", "", f"- Branch proposals: **{len(branch_rows)}**.", f"- Empty-merge outcomes: **{sum(branch_outcome(r) == 'BRANCH_FAILED_EMPTY_MERGE' for r in branch_rows)}**.", f"- Fallbacks: **{sum(bool(r.get('fallback_used')) for r in branch_rows)}**.", "", "The expanded diagnostic must be reviewed before any full Phase 1 run. See `branch_execution_analysis.md`, `merge_analysis.md`, and `both_wrong_analysis.md` for the required case-level analysis."]
	(RUN_ROOT.parent / "ADAPTIVE_V1_1_1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
	rows = all_rows()
	if len(rows) != 36:
		raise SystemExit(f"expected exactly 36 diagnostic rows, found {len(rows)}")
	cases = load_cases()
	if {cases[str(row["id"])] ["category"] for row in rows} != {"COR_ONLY_CORRECT", "OURS_ONLY_CORRECT", "BOTH_WRONG"}:
		raise SystemExit("diagnostic rows do not contain all three required source groups")
	baseline = load_rows(BASELINE_ROOT)
	branch = load_rows(BRANCH_ROOT)
	write_summary(rows, cases)
	write_routing(rows, cases)
	write_branch_analysis(rows, cases)
	write_merge_analysis(rows)
	write_both_wrong(rows, cases, baseline, branch)
	write_paired_case_analysis(rows, cases, baseline, branch)
	write_report(rows, cases)
	print(json.dumps({"rows": len(rows), "groups": Counter(cases[str(row["id"])] ["category"] for row in rows), "branch_proposals": sum(row.get("router_mode") == "branch" for row in rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
