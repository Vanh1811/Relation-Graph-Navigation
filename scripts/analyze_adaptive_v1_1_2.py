#!/usr/bin/env python3
"""Analyze Adaptive v1.1.2 diagnostic-36 and fresh GrailQA-25 outputs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_2"
DIAGNOSTIC_ROOT = ROOT / "diagnostic_36"
FRESH_ROOT = ROOT / "grailqa_25"
V11_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_1" / "diagnostic_36"
PAIRED_CASES = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "comparison" / "paired_cases" / "paired_case_summary.json"
BASELINE_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "cor_phase1_baseline_20261014_fixed"
BRANCH_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "branch_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_map(path: Path) -> dict[str, dict[str, Any]]:
	return {str(row.get("id", "")): row for row in read_jsonl(path)}


def token_total(row: dict[str, Any], native: bool = False) -> int:
	key = "native_token_usage" if native else "token_usage"
	return int((row.get(key) or {}).get("total_tokens", 0) or 0)


def metric(rows: list[dict[str, Any]], native: bool = False) -> dict[str, Any]:
	if not rows:
		return {"n": 0, "hits_at_1": 0.0, "f1": 0.0, "errors": 0, "llm_calls": 0.0, "kg_queries": 0.0, "tokens": 0.0, "latency_ms": 0.0}
	hit_key = "native_hits_at_1" if native else "hits_at_1"
	f1_key = "native_f1" if native else "f1"
	llm_key = "native_llm_calls" if native else "llm_calls"
	kg_key = "native_kg_queries" if native else "kg_queries"
	return {
		"n": len(rows),
		"hits_at_1": sum(bool(row.get(hit_key)) for row in rows) / len(rows),
		"f1": sum(float(row.get(f1_key, 0) or 0) for row in rows) / len(rows),
		"errors": sum(bool(row.get("native_error" if native else "error")) for row in rows),
		"llm_calls": sum(int(row.get(llm_key, 0) or 0) for row in rows) / len(rows),
		"kg_queries": sum(int(row.get(kg_key, 0) or 0) for row in rows) / len(rows),
		"tokens": sum(token_total(row, native) for row in rows) / len(rows),
		"latency_ms": sum(float(row.get("latency_ms", 0) or 0) for row in rows) / len(rows),
	}


def fmt_metric(value: dict[str, Any]) -> str:
	return f"{value['n']} | {value['hits_at_1']:.2%} | {value['f1']:.4f} | {value['errors']} | {value['llm_calls']:.2f} | {value['kg_queries']:.2f} | {value['tokens']:.2f} | {value['latency_ms']:.2f}"


def load_cases() -> dict[str, dict[str, Any]]:
	data = json.loads(PAIRED_CASES.read_text(encoding="utf-8"))
	return {str(item["id"]): item for item in data.get("paired_cases", [])}


def structure_counts(rows: list[dict[str, Any]]) -> Counter[str]:
	return Counter(str(row.get("router_structure", "UNKNOWN")) for row in rows)


def execution_counts(rows: list[dict[str, Any]]) -> Counter[str]:
	counts = Counter()
	for row in rows:
		if row.get("router_structure") == "BRANCH_FINAL":
			counts[f"branch_{row.get('branch_count', 0)}"] += 1
		else:
			counts["CoR"] += 1
	return counts


def branch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
	return [row for row in rows if row.get("router_structure") == "BRANCH_FINAL" and row.get("router_execution_mode") == "branch" and row.get("validation_passed")]


def branch_self_sufficiency(rows: list[dict[str, Any]]) -> dict[str, float | int]:
	selected = branch_rows(rows)
	successes = sum(not bool(row.get("fallback_used")) and row.get("final_execution_mode") == "branch" for row in selected)
	fallbacks = sum(bool(row.get("fallback_used")) for row in selected)
	return {
		"branch_final_selections": len(selected),
		"native_branch_successes": successes,
		"branch_fallbacks": fallbacks,
		"fallback_rate": fallbacks / len(selected) if selected else 0.0,
		"branch_self_sufficiency_rate": successes / len(selected) if selected else 0.0,
	}


def old_route_map() -> dict[str, str]:
	return {str(row["id"]): ("BRANCH_FINAL" if row.get("router_mode") == "branch" else "LINEAR") for row in read_jsonl(V11_ROOT / "results.jsonl")}


def cause(row: dict[str, Any], comparison: dict[str, Any] | None = None) -> str:
	if row.get("error") or row.get("native_error"):
		return "OTHER"
	if row.get("fallback_used"):
		if row.get("fallback_reason") in {"empty_strict_merge", "branch_empty"}:
			return "MERGE_FAILURE"
		return "BRANCH_EXECUTION"
	if comparison is not None and row.get("prediction") != comparison.get("prediction"):
		if not row.get("router_structure") or row.get("router_structure") in {"UNCERTAIN", "BASELINE_UNCHANGED"}:
			return "ENTITY_GROUNDING"
		return "RELATION_GROUNDING"
	return "OTHER"


def prediction(row: dict[str, Any], native: bool = False) -> list[str]:
	return list(row.get("native_prediction", []) if native else row.get("prediction", []))


def write_diagnostic(rows: list[dict[str, Any]], cases: dict[str, dict[str, Any]]) -> None:
	old_routes = old_route_map()
	groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		groups[cases.get(str(row["id"]), {}).get("category", "UNKNOWN")].append(row)
	all_final = metric(rows)
	all_native = metric(rows, native=True)
	ss = branch_self_sufficiency(rows)
	lines = [
		"# Adaptive v1.1.2 diagnostic-36 summary", "",
		"Historical CoR and Branch-v1 outputs were read for paired evaluation only; they were not rerun.", "",
		"## Native versus final metrics", "",
		"| View | N | Hits@1 | F1 | Errors | Avg LLM calls | Avg KG queries | Avg tokens | Avg latency (ms) |",
		"|---|---:|---:|---:|---:|---:|---:|---:|---:|",
		f"| Native / pre-fallback | {fmt_metric(all_native)} |",
		f"| Final / post-fallback | {fmt_metric(all_final)} |",
		"",
		"## Group outcomes", "", "| Group | N | Final Hits@1 | Final F1 | Native Hits@1 | Native F1 |", "|---|---:|---:|---:|---:|---:|",
	]
	for group in ["COR_ONLY_CORRECT", "OURS_ONLY_CORRECT", "BOTH_WRONG"]:
		final = metric(groups[group]); native = metric(groups[group], native=True)
		lines.append(f"| {group} | {final['n']} | {final['hits_at_1']:.2%} | {final['f1']:.4f} | {native['hits_at_1']:.2%} | {native['f1']:.4f} |")
	lines += ["", "- COR_ONLY_CORRECT preserved: **%d/10**" % sum(bool(row.get("hits_at_1")) for row in groups["COR_ONLY_CORRECT"]), "- OURS_ONLY_CORRECT recovered: **%d/3**" % sum(bool(row.get("hits_at_1")) for row in groups["OURS_ONLY_CORRECT"]), "- BOTH_WRONG newly solved: **%d/23**" % sum(bool(row.get("hits_at_1")) for row in groups["BOTH_WRONG"]), "", "## Structural routing", "", "| Structure | Count |", "|---|---:|"]
	for key in ["LINEAR", "BRANCH_FINAL", "BRANCH_INTERMEDIATE", "UNCERTAIN"]:
		lines.append(f"| {key} | {structure_counts(rows)[key]} |")
	lines += ["", "## Execution and fallback", "", f"- Execution: **{dict(execution_counts(rows))}**", f"- BRANCH_FINAL selections: **{ss['branch_final_selections']}**", f"- Native branch successes: **{ss['native_branch_successes']}**", f"- Branch fallbacks: **{ss['branch_fallbacks']}**", f"- Fallback rate: **{ss['fallback_rate']:.2%}**", f"- Branch self-sufficiency rate: **{ss['branch_self_sufficiency_rate']:.2%}**"]
	(DIAGNOSTIC_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
	(DIAGNOSTIC_ROOT / "summary.json").write_text(json.dumps({"final": all_final, "native": all_native, "structures": dict(structure_counts(rows)), "execution": dict(execution_counts(rows)), "branch_self_sufficiency": ss}, indent=2) + "\n", encoding="utf-8")

	changed: list[str] = []
	change_lines = ["# Diagnostic routing changes versus v1.1.1", "", "| ID | v1.1.1 route | v1.1.2 structure | Validation | Fallback | Reason |", "|---|---|---|---|---|---|"]
	for row in rows:
		old = old_routes.get(str(row["id"]), "UNKNOWN")
		new = str(row.get("router_structure", "UNKNOWN"))
		if old != new:
			changed.append(str(row["id"]))
		change_lines.append(f"| `{row['id']}` | {old} | {new} | {row.get('validation_passed')} | {row.get('fallback_used')} | {str(row.get('router_reason', '')).replace('|', '/')} |")
	(DIAGNOSTIC_ROOT / "routing_analysis.md").write_text("\n".join(change_lines + ["", f"Changed decisions: **{len(changed)}**.", f"Previous branch decisions changed to LINEAR: **{sum(old_routes.get(str(r['id'])) == 'BRANCH_FINAL' and r.get('router_structure') == 'LINEAR' for r in rows)}**.", f"Previous branch decisions changed to BRANCH_INTERMEDIATE: **{sum(old_routes.get(str(r['id'])) == 'BRANCH_FINAL' and r.get('router_structure') == 'BRANCH_INTERMEDIATE' for r in rows)}**.", f"BRANCH_FINAL retained: **{sum(old_routes.get(str(r['id'])) == 'BRANCH_FINAL' and r.get('router_structure') == 'BRANCH_FINAL' for r in rows)}**.", f"New BRANCH_FINAL selections: **{sum(old_routes.get(str(r['id'])) != 'BRANCH_FINAL' and r.get('router_structure') == 'BRANCH_FINAL' for r in rows)}**."]) + "\n", encoding="utf-8")

	branch = branch_rows(rows)
	fallback_lines = ["# Diagnostic fallback analysis", "", "Fallback is a safety mechanism, not evidence that the original branch route was correct.", "", "| ID | Structure | Native Hits@1 | Final Hits@1 | Native result | Fallback reason |", "|---|---|---:|---:|---|---|"]
	for row in branch:
		fallback_lines.append(f"| `{row['id']}` | {row.get('router_structure')} | {row.get('native_hits_at_1')} | {row.get('hits_at_1')} | {row.get('native_result_action')} | {row.get('fallback_reason') or '-'} |")
	(DIAGNOSTIC_ROOT / "fallback_analysis.md").write_text("\n".join(fallback_lines + ["", f"Branch self-sufficiency rate: **{ss['branch_self_sufficiency_rate']:.2%}**."]) + "\n", encoding="utf-8")

	paired_lines = ["# Diagnostic paired analysis", "", "| ID | Group | Structure | Native Hits@1 | Final Hits@1 | Native F1 | Final F1 |", "|---|---|---|---:|---:|---:|---:|"]
	for row in rows:
		paired_lines.append(f"| `{row['id']}` | {cases.get(str(row['id']), {}).get('category', 'UNKNOWN')} | {row.get('router_structure')} | {row.get('native_hits_at_1')} | {row.get('hits_at_1')} | {float(row.get('native_f1', 0)):.4f} | {float(row.get('f1', 0)):.4f} |")
	(DIAGNOSTIC_ROOT / "paired_analysis.md").write_text("\n".join(paired_lines) + "\n", encoding="utf-8")

	ground_counts = Counter(cause(row) for row in rows if not row.get("hits_at_1"))
	ground_lines = ["# Diagnostic grounding and failure analysis", "", "These are post-hoc diagnostic labels; they do not enter runtime routing.", "", "| Category | Count |", "|---|---:|"]
	for key in ["ENTITY_GROUNDING", "RELATION_GROUNDING", "STRUCTURAL_ROUTING", "BRANCH_EXECUTION", "MERGE_FAILURE", "FILTERING_LOSS", "BACKTRACK_FAILURE", "ANSWER_GENERATION", "OTHER"]:
		ground_lines.append(f"| {key} | {ground_counts[key]} |")
	ground_lines += ["", "## Cases", ""]
	for row in rows:
		if not row.get("hits_at_1"):
			ground_lines.append(f"- `{row['id']}`: **{cause(row)}** — {row.get('router_reason', '')}")
	(DIAGNOSTIC_ROOT / "grounding_analysis.md").write_text("\n".join(ground_lines) + "\n", encoding="utf-8")


def write_fresh(rows_cor: list[dict[str, Any]], rows_adaptive: list[dict[str, Any]]) -> None:
	cor = {str(row["id"]): row for row in rows_cor}
	adaptive = {str(row["id"]): row for row in rows_adaptive}
	if set(cor) != set(adaptive) or len(cor) != 25:
		raise SystemExit("fresh CoR/adaptive outputs do not contain the same 25 IDs")
	adapt_rows = list(adaptive.values())
	ss = branch_self_sufficiency(adapt_rows)
	lines = ["# Fresh GrailQA-25 comparison", "", "The sample was frozen before either method was run and excludes the original Phase 1 GrailQA-50 and v1.1 diagnostic IDs.", "", "## CoR versus Adaptive v1.1.2", "", "| Method / view | N | Hits@1 | F1 | Errors | Avg LLM calls | Avg KG queries | Avg tokens | Avg latency (ms) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
	for name, rows, native in [("Unchanged CoR", rows_cor, False), ("Adaptive native / pre-fallback", adapt_rows, True), ("Adaptive final / post-fallback", adapt_rows, False)]:
		lines.append(f"| {name} | {fmt_metric(metric(rows, native=native))} |")
	lines += ["", "## Adaptive structure and fallback", "", f"- Structures: **{dict(structure_counts(adapt_rows))}**", f"- Execution: **{dict(execution_counts(adapt_rows))}**", f"- BRANCH_FINAL selections: **{ss['branch_final_selections']}**", f"- Native branch successes: **{ss['native_branch_successes']}**", f"- Branch fallbacks: **{ss['branch_fallbacks']}**", f"- Fallback rate: **{ss['fallback_rate']:.2%}**", f"- Branch self-sufficiency rate: **{ss['branch_self_sufficiency_rate']:.2%}"]
	(FRESH_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

	paired_counts = Counter()
	paired_lines = ["# Fresh GrailQA-25 paired analysis", "", "| ID | Classification | Structure | Execution | Fallback | CoR prediction | Adaptive prediction | Cause if different |", "|---|---|---|---|---|---|---|---|"]
	for item_id in sorted(cor):
		base = cor[item_id]; new = adaptive[item_id]
		base_ok = bool(base.get("hits_at_1")); new_ok = bool(new.get("hits_at_1"))
		if base_ok and new_ok: label = "BOTH_CORRECT"
		elif base_ok: label = "COR_ONLY_CORRECT"
		elif new_ok: label = "ADAPTIVE_ONLY_CORRECT"
		else: label = "BOTH_WRONG"
		paired_counts[label] += 1
		different = prediction(base) != prediction(new)
		paired_lines.append(f"| `{item_id}` | {label} | {new.get('router_structure')} | {new.get('final_execution_mode')} | {new.get('fallback_used')} | {json.dumps(prediction(base), ensure_ascii=False)} | {json.dumps(prediction(new), ensure_ascii=False)} | {cause(new, base) if different else '-'} |")
	(FRESH_ROOT / "paired_analysis.md").write_text("\n".join(paired_lines + ["", f"Outcome counts: **{dict(paired_counts)}**."]) + "\n", encoding="utf-8")

	routing_lines = ["# Fresh GrailQA-25 routing analysis", "", "| Structure | Count |", "|---|---:|"]
	for key in ["LINEAR", "BRANCH_FINAL", "BRANCH_INTERMEDIATE", "UNCERTAIN"]:
		routing_lines.append(f"| {key} | {structure_counts(adapt_rows)[key]} |")
	routing_lines += ["", f"Execution: **{dict(execution_counts(adapt_rows))}**.", f"Routing prompt version: **adaptive-routing-v1.1.2-rule-aware-json-v1**."]
	(FRESH_ROOT / "routing_analysis.md").write_text("\n".join(routing_lines) + "\n", encoding="utf-8")

	fallback_lines = ["# Fresh GrailQA-25 fallback analysis", "", "| ID | Structure | Native Hits@1 | Final Hits@1 | Fallback | Reason |", "|---|---|---:|---:|---|---|"]
	for row in adapt_rows:
		if row.get("router_structure") == "BRANCH_FINAL" or row.get("fallback_used"):
			fallback_lines.append(f"| `{row['id']}` | {row.get('router_structure')} | {row.get('native_hits_at_1')} | {row.get('hits_at_1')} | {row.get('fallback_used')} | {row.get('fallback_reason') or '-'} |")
	(FRESH_ROOT / "fallback_analysis.md").write_text("\n".join(fallback_lines + ["", f"Branch self-sufficiency rate: **{ss['branch_self_sufficiency_rate']:.2%}**."]) + "\n", encoding="utf-8")

	ground_counts = Counter(cause(adaptive[item_id], cor[item_id]) for item_id in cor if prediction(adaptive[item_id]) != prediction(cor[item_id]))
	ground_lines = ["# Fresh GrailQA-25 grounding analysis", "", "Post-hoc failure labels are kept separate from structural routing.", "", "| Category | Count |", "|---|---:|"]
	for key in ["ENTITY_GROUNDING", "RELATION_GROUNDING", "STRUCTURAL_ROUTING", "BRANCH_EXECUTION", "MERGE_FAILURE", "FILTERING_LOSS", "BACKTRACK_FAILURE", "ANSWER_GENERATION", "OTHER"]:
		ground_lines.append(f"| {key} | {ground_counts[key]} |")
	(FRESH_ROOT / "grounding_analysis.md").write_text("\n".join(ground_lines) + "\n", encoding="utf-8")
	(FRESH_ROOT / "summary.json").write_text(json.dumps({"cor": metric(rows_cor), "adaptive_native": metric(adapt_rows, native=True), "adaptive_final": metric(adapt_rows), "paired_outcomes": dict(paired_counts), "structures": dict(structure_counts(adapt_rows)), "branch_self_sufficiency": ss}, indent=2) + "\n", encoding="utf-8")


def write_combined(diagnostic: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> None:
	all_rows = diagnostic + fresh
	ss = branch_self_sufficiency(all_rows)
	counts = structure_counts(all_rows)
	lines = ["# Adaptive v1.1.2 combined 61-case descriptive summary", "", "The diagnostic-36 and fresh GrailQA-25 sets are intentionally different and are not treated as one statistically homogeneous benchmark.", "", "## Separate effectiveness results", "", "- Diagnostic-36: see `diagnostic_36/summary.md`.", "- Fresh GrailQA-25: see `grailqa_25/summary.md`.", "", "## Combined routing and fallback counts only", "", f"- Total rows: **{len(all_rows)}**", f"- LINEAR: **{counts['LINEAR']}**", f"- BRANCH_FINAL: **{counts['BRANCH_FINAL']}**", f"- BRANCH_INTERMEDIATE: **{counts['BRANCH_INTERMEDIATE']}**", f"- UNCERTAIN: **{counts['UNCERTAIN']}**", f"- BRANCH_FINAL selections: **{ss['branch_final_selections']}**", f"- Native branch successes: **{ss['native_branch_successes']}**", f"- Branch fallbacks: **{ss['branch_fallbacks']}**", f"- Overall branch self-sufficiency: **{ss['branch_self_sufficiency_rate']:.2%}**"]
	(ROOT / "combined_61_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(diagnostic: list[dict[str, Any]], fresh_cor: list[dict[str, Any]], fresh_adaptive: list[dict[str, Any]]) -> None:
	dm = metric(diagnostic); dn = metric(diagnostic, native=True); fm = metric(fresh_adaptive); fn = metric(fresh_adaptive, native=True); cm = metric(fresh_cor); ss = branch_self_sufficiency(diagnostic + fresh_adaptive)
	lines = [
		"# Adaptive v1.1.2 report", "", "## Motivation", "", "v1.1.1 completed the 36-case diagnostic with 0 runtime errors, preserved 10/10 CoR-only wins and recovered 3/3 prior branch wins, but 10 of 15 branch selections fell back to CoR. Fallback success therefore could not be treated as evidence that the branch route was correct.", "", "## Method", "", "The router now receives explicit rules for `LINEAR`, `BRANCH_FINAL`, `BRANCH_INTERMEDIATE`, and `UNCERTAIN`. Deterministic validation remains after the LLM proposal. `BRANCH_INTERMEDIATE` is recorded and executed as CoR because branch→merge→continue is not implemented.", "", "## Previous fallback-case reanalysis", "", "Nine of the ten previous branch/fallback cases are direct-final branch candidates. The methane/staged-combustion case is `BRANCH_INTERMEDIATE`: both constraints identify an engine before a later designer relation. See `router_reanalysis/both_wrong_branch_10.md`.", "", "## Diagnostic-36", "", f"- Final/post-fallback: **{dm['hits_at_1']:.2%} Hits@1**, **{dm['f1']:.4f} F1**, **{dm['errors']} errors**.", f"- Native/pre-fallback: **{dn['hits_at_1']:.2%} Hits@1**, **{dn['f1']:.4f} F1**.", "- Group preservation and paired outcomes are in `diagnostic_36/summary.md` and `diagnostic_36/paired_analysis.md`.", "- Routing changes and failure categories are in `diagnostic_36/routing_analysis.md` and `diagnostic_36/grounding_analysis.md`.", "", "## Fresh GrailQA-25", "", f"- Unchanged CoR: **{cm['hits_at_1']:.2%} Hits@1**, **{cm['f1']:.4f} F1**, **{cm['errors']} errors**.", f"- Adaptive native: **{fn['hits_at_1']:.2%} Hits@1**, **{fn['f1']:.4f} F1**.", f"- Adaptive final: **{fm['hits_at_1']:.2%} Hits@1**, **{fm['f1']:.4f} F1**, **{fm['errors']} errors**.", "- The frozen sample excludes the original Phase 1 GrailQA-50 and all v1.1 diagnostic IDs. Paired outcomes are in `grailqa_25/paired_analysis.md`.", "", "## Branch self-sufficiency", "", f"Across the 61 Adaptive rows: **{ss['branch_final_selections']}** BRANCH_FINAL selections, **{ss['native_branch_successes']}** native branch successes, **{ss['branch_fallbacks']}** fallbacks, and **{ss['branch_self_sufficiency_rate']:.2%}** self-sufficiency.", "", "## Combined routing", "", "Only routing/fallback counts are combined across the intentionally selected diagnostic-36 and fresh GrailQA-25 sets. Accuracy is kept separate in `combined_61_summary.md`.", "", "## Recommendation", "", "Do not run the full Phase 1 100 automatically. The next decision should be based on whether the fresh GrailQA comparison shows a genuine native branch benefit and whether fallback decreases without losing CoR wins. If branch-final decisions are semantically correct but native execution remains weak, prioritize branch execution; if structural decisions are wrong, continue routing work; if most differences are entity/relation grounding, shift focus to grounding.", "", "> Adaptive v1.1.2 suggests that the next research priority should be determined from the fresh GrailQA-25 native-versus-CoR comparison because fallback-assisted accuracy alone cannot validate structural routing.",
	]
	(ROOT / "ADAPTIVE_V1_1_2_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
	diagnostic_path = DIAGNOSTIC_ROOT / "results.jsonl"
	if not diagnostic_path.exists():
		raise SystemExit(f"missing diagnostic output: {diagnostic_path}")
	diagnostic = read_jsonl(diagnostic_path)
	if len(diagnostic) != 36:
		raise SystemExit(f"expected 36 diagnostic rows, found {len(diagnostic)}")
	if any(row.get("error") for row in diagnostic):
		raise SystemExit("diagnostic contains runtime errors")
	cases = load_cases()
	write_diagnostic(diagnostic, cases)
	if not (FRESH_ROOT / "cor_results.jsonl").exists() or not (FRESH_ROOT / "adaptive_results.jsonl").exists():
		raise SystemExit("fresh CoR and Adaptive outputs are both required")
	fresh_cor = read_jsonl(FRESH_ROOT / "cor_results.jsonl")
	fresh_adaptive = read_jsonl(FRESH_ROOT / "adaptive_results.jsonl")
	if len(fresh_cor) != 25 or len(fresh_adaptive) != 25:
		raise SystemExit("fresh outputs must contain exactly 25 rows each")
	write_fresh(fresh_cor, fresh_adaptive)
	write_combined(diagnostic, fresh_adaptive)
	write_report(diagnostic, fresh_cor, fresh_adaptive)
	print(json.dumps({"diagnostic": metric(diagnostic), "fresh_cor": metric(fresh_cor), "fresh_adaptive": metric(fresh_adaptive), "combined_branch_self_sufficiency": branch_self_sufficiency(diagnostic + fresh_adaptive)}, indent=2))


if __name__ == "__main__":
	main()
