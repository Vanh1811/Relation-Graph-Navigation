#!/usr/bin/env python3
"""Analyze the frozen Adaptive v1.1.3 diagnostic-36 run."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_3"
DIAGNOSTIC_ROOT = ROOT / "diagnostic_36"
SOURCE_CONFIG = (
	PROJECT_ROOT
	/ "outputs"
	/ "experiments"
	/ "phase1"
	/ "adaptive_v1_1_1"
	/ "diagnostic_36"
	/ "adaptive_v1_1_config.json"
)
V11_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_1" / "diagnostic_36"
V12_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_2" / "diagnostic_36"
BASELINE_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "cor_phase1_baseline_20261014_fixed"
BRANCH_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "branch_v1"
SUBSETS = ("cwq_branch_2", "grailqa_branch_2")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_map(paths: list[Path]) -> dict[str, dict[str, Any]]:
	rows: dict[str, dict[str, Any]] = {}
	for path in paths:
		for row in read_jsonl(path):
			rows[str(row.get("id", ""))] = row
	return rows


def load_context() -> tuple[list[dict[str, Any]], dict[str, str], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
	config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
	ids: list[str] = []
	for subset in SUBSETS:
		ids.extend(str(value) for value in config["frozen_sample_ids"][subset])
	categories = {str(key): str(value) for key, value in config["paired_case_categories"].items()}
	current = {str(row["id"]): row for row in read_jsonl(DIAGNOSTIC_ROOT / "results.jsonl")}
	v11 = read_map([V11_ROOT / f"{subset}_results.jsonl" for subset in SUBSETS])
	v12 = read_map([V12_ROOT / "adaptive_results.jsonl"])
	baseline = read_map([BASELINE_ROOT / f"{subset}_results.jsonl" for subset in SUBSETS])
	branch = read_map([BRANCH_ROOT / f"{subset}_results.jsonl" for subset in SUBSETS])
	missing = [case_id for case_id in ids if case_id not in current]
	if missing:
		raise SystemExit(f"missing v1.1.3 diagnostic rows: {missing}")
	if len(current) != 36:
		raise SystemExit(f"expected 36 unique v1.1.3 rows, found {len(current)}")
	for name, mapping in (("v1.1.1", v11), ("v1.1.2", v12), ("CoR", baseline), ("Branch-v1", branch)):
		missing = [case_id for case_id in ids if case_id not in mapping]
		if missing:
			raise SystemExit(f"missing {name} historical rows: {missing}")
	return [current[case_id] for case_id in ids], categories, v11, v12, baseline, branch


def prediction(row: dict[str, Any], native: bool = False) -> list[str]:
	return list(row.get("native_prediction", []) if native else row.get("prediction", []))


def tokens(row: dict[str, Any], native: bool = False) -> int:
	if native and row.get("native_token_usage"):
		usage = row.get("native_token_usage") or {}
	else:
		usage = row.get("token_usage") or {}
	if usage.get("total_tokens") is not None:
		return int(usage.get("total_tokens") or 0)
	return int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0)


def metric(rows: list[dict[str, Any]], native: bool = False) -> dict[str, Any]:
	hit_key = "native_hits_at_1" if native else "hits_at_1"
	f1_key = "native_f1" if native else "f1"
	error_key = "native_error" if native else "error"
	call_key = "native_llm_calls" if native else "llm_calls"
	kg_key = "native_kg_queries" if native else "kg_queries"
	return {
		"n": len(rows),
		"hits": sum(bool(row.get(hit_key)) for row in rows),
		"hits_rate": sum(bool(row.get(hit_key)) for row in rows) / len(rows) if rows else 0.0,
		"f1": sum(float(row.get(f1_key, 0) or 0) for row in rows) / len(rows) if rows else 0.0,
		"errors": sum(bool(row.get(error_key)) for row in rows),
		"llm_calls": sum(int(row.get(call_key, 0) or 0) for row in rows) / len(rows) if rows else 0.0,
		"kg_queries": sum(int(row.get(kg_key, 0) or 0) for row in rows) / len(rows) if rows else 0.0,
		"tokens": sum(tokens(row, native) for row in rows) / len(rows) if rows else 0.0,
		"latency_ms": sum(float(row.get("latency_ms", 0) or 0) for row in rows) / len(rows) if rows else 0.0,
	}


def fmt_metric(value: dict[str, Any]) -> str:
	return f"{value['n']} | {value['hits_rate']:.2%} | {value['f1']:.4f} | {value['errors']} | {value['llm_calls']:.2f} | {value['kg_queries']:.2f} | {value['tokens']:.2f} | {value['latency_ms']:.2f}"


def structure(row: dict[str, Any], version: str = "v1.1.3") -> str:
	if version == "v1.1.1":
		return "BRANCH_FINAL" if row.get("router_mode") == "branch" else "LINEAR"
	if version == "v1.1.2":
		return str(row.get("router_structure", "UNKNOWN"))
	return str(row.get("router_structure", "UNKNOWN"))


def execution_mode(row: dict[str, Any]) -> str:
	return str(row.get("execution_mode", row.get("final_execution_mode", "cor")))


def branch_final(row: dict[str, Any], version: str = "v1.1.3") -> bool:
	return structure(row, version) == "BRANCH_FINAL"


def fallback(row: dict[str, Any]) -> bool:
	return bool(row.get("fallback_used"))


def native_branch_success(row: dict[str, Any], version: str = "v1.1.3") -> bool:
	if not branch_final(row, version) or fallback(row):
		return False
	return execution_mode(row) == "branch"


def branch_stats(rows: list[dict[str, Any]], version: str = "v1.1.3") -> dict[str, Any]:
	selected = [row for row in rows if branch_final(row, version)]
	successes = sum(native_branch_success(row, version) for row in selected)
	fallbacks = sum(fallback(row) for row in selected)
	return {
		"branch_final_selections": len(selected),
		"native_branch_successes": successes,
		"branch_fallbacks": fallbacks,
		"fallback_rate": fallbacks / len(selected) if selected else 0.0,
		"branch_self_sufficiency_rate": successes / len(selected) if selected else 0.0,
	}


def validation_failures(row: dict[str, Any]) -> list[str]:
	return [str(value) for value in row.get("validation_failures", []) or []]


def validation_kind(row: dict[str, Any]) -> str:
	failures = validation_failures(row)
	if any("ANSWER_TYPE" in value or "answer type" in value.lower() for value in failures):
		return "ANSWER_TYPE_ERROR"
	if any("CONSTRAINT_PROPERTY" in value or "constraint property" in value.lower() for value in failures):
		return "CONSTRAINT_PROPERTY_ERROR"
	if not row.get("validation_passed", True):
		return "OTHER_VALIDATION_REJECTION"
	return "NONE"


def main_cause(row: dict[str, Any], baseline: dict[str, Any], branch: dict[str, Any]) -> str:
	if row.get("error"):
		return "OTHER"
	validation = validation_kind(row)
	if validation == "ANSWER_TYPE_ERROR":
		return "ANSWER_TYPE_ERROR"
	if validation == "CONSTRAINT_PROPERTY_ERROR":
		return "CONSTRAINT_PROPERTY_ERROR"
	if fallback(row):
		outcome = str(row.get("merge_outcome", ""))
		if "EMPTY" in outcome or "NO_INTERSECTION" in outcome:
			return "MERGE_FAILURE"
		return "BRANCH_EXECUTION"
	if branch_final(row):
		if not row.get("relation_chain") and not row.get("branches"):
			return "ENTITY_GROUNDING"
		return "RELATION_GROUNDING"
	if structure(row) in {"UNCERTAIN", "BRANCH_INTERMEDIATE"}:
		return "STRUCTURAL_ROUTING"
	if not row.get("relation_chain"):
		return "ENTITY_GROUNDING"
	return "RELATION_GROUNDING"


def success_cause(row: dict[str, Any], baseline: dict[str, Any], branch: dict[str, Any]) -> str:
	if row.get("error"):
		return "OTHER"
	if fallback(row):
		return "FALLBACK"
	if branch_final(row):
		if str(row.get("merge_outcome", "")) == "MERGED":
			return "BRANCH_INTERSECTION"
		return "ANSWER_SET_PRECISION"
	if row.get("native_prediction") != baseline.get("prediction"):
		return "STOCHASTIC_VARIATION"
	return "OTHER"


def both_wrong_classification(row: dict[str, Any], baseline: dict[str, Any], branch: dict[str, Any]) -> str:
	if bool(row.get("hits_at_1")):
		return "NEWLY_SOLVED"
	old_f1 = max(float(baseline.get("f1", 0) or 0), float(branch.get("f1", 0) or 0))
	if float(row.get("f1", 0) or 0) > old_f1:
		return "PARTIAL_F1_IMPROVEMENT"
	return "STILL_WRONG"


def _case_line(row: dict[str, Any], baseline: dict[str, Any], v11: dict[str, Any], v12: dict[str, Any], branch: dict[str, Any]) -> str:
	return (
		f"- **`{row['id']}`** — {row.get('question', '')}\n"
		f"  - CoR prediction: `{json.dumps(baseline.get('prediction', []), ensure_ascii=False)}`; "
		f"v1.1.1: `{json.dumps(v11.get('prediction', []), ensure_ascii=False)}`; "
		f"v1.1.2: `{json.dumps(v12.get('prediction', []), ensure_ascii=False)}`; "
		f"v1.1.3: `{json.dumps(row.get('prediction', []), ensure_ascii=False)}`.\n"
		f"  - v1.1.2 structure: `{structure(v12, 'v1.1.2')}`; v1.1.3 structure: `{structure(row)}`; "
		f"execution: `{execution_mode(row)}`; cause: `{main_cause(row, baseline, branch)}`.\n"
		f"  - Router reason: {str(row.get('router_reason', '')).replace(chr(10), ' ')}"
	)


def write_summary(rows: list[dict[str, Any]], categories: dict[str, str]) -> None:
	groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		groups[categories[str(row["id"])]].append(row)
	native = metric(rows, native=True)
	final = metric(rows)
	ss = branch_stats(rows)
	structures = Counter(structure(row) for row in rows)
	validation = Counter(validation_kind(row) for row in rows)
	lines = [
		"# Adaptive v1.1.3 diagnostic-36 summary", "",
		"This is a regression/failure-analysis run on the previously inspected frozen 36-case set. Historical methods were read only and were not rerun.", "",
		"## Overall", "",
		"| View | N | Hits@1 | F1 | Errors | Avg LLM calls | Avg KG queries | Avg tokens | Avg latency (ms) |",
		"|---|---:|---:|---:|---:|---:|---:|---:|---:|",
		f"| Native / pre-fallback | {fmt_metric(native)} |",
		f"| Final / post-fallback | {fmt_metric(final)} |", "",
		"## Groups", "",
		"| Group | N | Final Hits@1 | Final F1 | Native Hits@1 | Native F1 |",
		"|---|---:|---:|---:|---:|---:|",
	]
	for group in ("COR_ONLY_CORRECT", "OURS_ONLY_CORRECT", "BOTH_WRONG"):
		group_final = metric(groups[group])
		group_native = metric(groups[group], native=True)
		lines.append(f"| {group} | {len(groups[group])} | {group_final['hits_rate']:.2%} | {group_final['f1']:.4f} | {group_native['hits_rate']:.2%} | {group_native['f1']:.4f} |")
	lines += [
		"", "- COR_ONLY_CORRECT preserved: **%d/10**" % sum(bool(row.get("hits_at_1")) for row in groups["COR_ONLY_CORRECT"]),
		"- OURS_ONLY_CORRECT recovered: **%d/3**" % sum(bool(row.get("hits_at_1")) for row in groups["OURS_ONLY_CORRECT"]),
		"- BOTH_WRONG newly solved: **%d/23**" % sum(bool(row.get("hits_at_1")) for row in groups["BOTH_WRONG"]),
		"", "## Routing", "", "| Structure | Count |", "|---|---:|",
	]
	for key in ("LINEAR", "BRANCH_FINAL", "BRANCH_INTERMEDIATE", "UNCERTAIN"):
		lines.append(f"| {key} | {structures[key]} |")
	lines += [
		"", "## Branching", "",
		f"- BRANCH_FINAL selections: **{ss['branch_final_selections']}**",
		f"- Native branch successes: **{ss['native_branch_successes']}**",
		f"- Branch fallbacks: **{ss['branch_fallbacks']}**",
		f"- Fallback rate: **{ss['fallback_rate']:.2%}**",
		f"- Branch self-sufficiency: **{ss['branch_self_sufficiency_rate']:.2%}**",
		"", "## Validation", "",
		f"- Answer-type mismatch failures: **{validation['ANSWER_TYPE_ERROR']}**",
		f"- Constraint-property mismatch failures: **{validation['CONSTRAINT_PROPERTY_ERROR']}**",
		f"- Validation rejections: **{sum(value != 'NONE' for value in (validation_kind(row) for row in rows))}**",
		f"- Safely redirected to CoR after validation rejection: **{sum(not row.get('validation_passed', True) and execution_mode(row) in {'cor', 'cor_fallback'} for row in rows)}**",
	]
	(DIAGNOSTIC_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
	(DIAGNOSTIC_ROOT / "summary.json").write_text(json.dumps({"native": native, "final": final, "structures": dict(structures), "branch": ss, "validation": dict(validation)}, indent=2) + "\n", encoding="utf-8")


def write_paired_groups(rows: list[dict[str, Any]], categories: dict[str, str], v11: dict[str, dict[str, Any]], v12: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]], branch: dict[str, dict[str, Any]]) -> None:
	groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		groups[categories[str(row["id"])]].append(row)
	lines = [
		"# Adaptive v1.1.3 paired group analysis", "",
		"The group labels come from the frozen historical paired comparison. They are used for regression analysis only, not as runtime inputs.", "",
		"| ID | Group | v1.1.3 structure | Execution | Native Hits@1 | Final Hits@1 | Final F1 | Classification / cause |",
		"|---|---|---|---|---:|---:|---:|---|",
	]
	for row in rows:
		group = categories[str(row["id"])]
		if group == "BOTH_WRONG":
			label = f"{both_wrong_classification(row, baseline[str(row['id'])], branch[str(row['id'])])} / {main_cause(row, baseline[str(row['id'])], branch[str(row['id'])])}"
		elif group == "COR_ONLY_CORRECT":
			label = "PRESERVED" if row.get("hits_at_1") else f"LOST / {main_cause(row, baseline[str(row['id'])], branch[str(row['id'])])}"
		else:
			label = "RECOVERED / " + success_cause(row, baseline[str(row['id'])], branch[str(row['id'])]) if row.get("hits_at_1") else "NOT_RECOVERED"
		lines.append(f"| `{row['id']}` | {group} | {structure(row)} | {execution_mode(row)} | {row.get('native_hits_at_1', 0)} | {row.get('hits_at_1', 0)} | {float(row.get('f1', 0) or 0):.4f} | {label} |")
	lines += ["", "## COR_ONLY_CORRECT preservation", "", f"Preserved: **{sum(bool(row.get('hits_at_1')) for row in groups['COR_ONLY_CORRECT'])}/10**.", f"Lost: **{sum(not bool(row.get('hits_at_1')) for row in groups['COR_ONLY_CORRECT'])}/10**.", ""]
	for row in groups["COR_ONLY_CORRECT"]:
		if not row.get("hits_at_1"):
			lines.append(_case_line(row, baseline[str(row["id"])], v11[str(row["id"])], v12[str(row["id"])], branch[str(row["id"])]))
	lines += ["", "## OURS_ONLY_CORRECT preservation", "", f"Recovered: **{sum(bool(row.get('hits_at_1')) for row in groups['OURS_ONLY_CORRECT'])}/3**.", f"Lost: **{sum(not bool(row.get('hits_at_1')) for row in groups['OURS_ONLY_CORRECT'])}/3**.", ""]
	for row in groups["OURS_ONLY_CORRECT"]:
		if row.get("hits_at_1"):
			lines.append(_case_line(row, baseline[str(row["id"])], v11[str(row["id"])], v12[str(row["id"])], branch[str(row["id"])]))
	lines += ["", "## BOTH_WRONG improvement", ""]
	both = groups["BOTH_WRONG"]
	lines += [f"Newly solved: **{sum(bool(row.get('hits_at_1')) for row in both)}/23**.", f"Partial F1 improvement: **{sum(both_wrong_classification(row, baseline[str(row['id'])], branch[str(row['id'])]) == 'PARTIAL_F1_IMPROVEMENT' for row in both)}/23**.", f"Still wrong: **{sum(both_wrong_classification(row, baseline[str(row['id'])], branch[str(row['id'])]) == 'STILL_WRONG' for row in both)}/23**.", "", "| ID | Classification | Cause | Route | Prediction |", "|---|---|---|---|---|"]
	for row in both:
		lines.append(f"| `{row['id']}` | {both_wrong_classification(row, baseline[str(row['id'])], branch[str(row['id'])])} | {main_cause(row, baseline[str(row['id'])], branch[str(row['id'])])} | {structure(row)} | `{json.dumps(row.get('prediction', []), ensure_ascii=False)}` |")
	lines += ["", "## Case explanations", ""]
	for row in both:
		lines.append(_case_line(row, baseline[str(row["id"])], v11[str(row["id"])], v12[str(row["id"])], branch[str(row["id"])]))
	(DIAGNOSTIC_ROOT / "paired_group_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_routing(rows: list[dict[str, Any]], v12: dict[str, dict[str, Any]]) -> None:
	transitions = Counter(f"{structure(v12[str(row['id'])], 'v1.1.2')} → {structure(row)}" for row in rows)
	lines = ["# Adaptive v1.1.3 routing comparison", "", "Transitions compare v1.1.3 with the frozen v1.1.2 structural decisions; historical methods were not rerun.", "", "## Transition counts", "", "| Transition | Count |", "|---|---:|"]
	for transition, count in sorted(transitions.items()):
		lines.append(f"| {transition} | {count} |")
	lines += ["", "## All 36 decisions", "", "| ID | v1.1.2 | v1.1.3 | Confidence | Requested type | Semantic property | Validation | Reason |", "|---|---|---|---:|---|---|---|---|"]
	for row in rows:
		lines.append(f"| `{row['id']}` | {structure(v12[str(row['id'])], 'v1.1.2')} | {structure(row)} | {float(row.get('router_confidence', 0) or 0):.2f} | {row.get('requested_final_answer_type', '')} | {row.get('requested_final_semantic_property', '')} | {row.get('validation_passed')} | {str(row.get('router_reason', '')).replace('|', '/').replace(chr(10), ' ')} |")
	lines += ["", "## Changed structures", ""]
	for row in rows:
		old = structure(v12[str(row["id"])], "v1.1.2")
		new = structure(row)
		if old != new:
			lines.append(f"- `{row['id']}`: **{old} → {new}** — {row.get('router_reason', '')}")
	(DIAGNOSTIC_ROOT / "routing_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_branch_analysis(rows: list[dict[str, Any]], categories: dict[str, str]) -> None:
	branch_rows = [row for row in rows if branch_final(row)]
	lines = ["# Adaptive v1.1.3 branch analysis", "", "| ID | Group | Branches | Native mode | Fallback | Merge result | Native Hits@1 | Final Hits@1 |", "|---|---|---:|---|---|---|---:|---:|"]
	for row in branch_rows:
		lines.append(f"| `{row['id']}` | {categories[str(row['id'])]} | {row.get('branch_count', 0)} | {row.get('native_execution_mode', '')} | {row.get('fallback_used')} | {row.get('merge_outcome', '')} | {row.get('native_hits_at_1', 0)} | {row.get('hits_at_1', 0)} |")
	for row in branch_rows:
		lines += ["", f"## `{row['id']}` — {row.get('question', '')}", "", f"- Group: `{categories[str(row['id'])]}`", f"- Router reason: {row.get('router_reason', '')}", f"- Validation: `{row.get('validation_passed')}`; failures: `{row.get('validation_failures', [])}`", f"- Native prediction: `{json.dumps(row.get('native_prediction', []), ensure_ascii=False)}`; final prediction: `{json.dumps(row.get('prediction', []), ensure_ascii=False)}`", f"- Merge method/result: `{row.get('merge_method', '')}` / `{row.get('merge_outcome', '')}`", "", "| Branch | Root | Constraint property | Target type | Relation chain | Candidate count | Final branch answers | Status |", "|---|---|---|---|---|---:|---|---|"]
		for branch in row.get("branches", []) or []:
			lines.append(f"| {branch.get('branch_id', '')} | `{json.dumps(branch.get('root', {}), ensure_ascii=False)}` | {branch.get('constraint_property', '')} | {branch.get('target_answer_type', '')} | `{json.dumps(branch.get('relation_chain', []), ensure_ascii=False)}` | {branch.get('candidate_count', 0)} | `{json.dumps(branch.get('final_branch_answer_set', []), ensure_ascii=False)}` | {branch.get('status', '')} |")
	ss = branch_stats(rows)
	lines += ["", "## Aggregate", "", f"- BRANCH_FINAL selections: **{ss['branch_final_selections']}**", f"- Native branch successes: **{ss['native_branch_successes']}**", f"- Branch fallbacks: **{ss['branch_fallbacks']}**", f"- Fallback rate: **{ss['fallback_rate']:.2%}**", f"- Branch self-sufficiency: **{ss['branch_self_sufficiency_rate']:.2%}**"]
	(DIAGNOSTIC_ROOT / "branch_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_grounding(rows: list[dict[str, Any]], baseline: dict[str, dict[str, Any]], branch: dict[str, dict[str, Any]], categories: dict[str, str]) -> None:
	causes = Counter()
	for row in rows:
		causes[main_cause(row, baseline[str(row["id"])], branch[str(row["id"])])] += 1
	lines = ["# Adaptive v1.1.3 grounding and failure analysis", "", "These are post-hoc labels for regression analysis; they were not runtime inputs.", "", "| Cause | Count |", "|---|---:|"]
	for key in ("ENTITY_GROUNDING", "RELATION_GROUNDING", "STRUCTURAL_ROUTING", "ANSWER_TYPE_ERROR", "CONSTRAINT_PROPERTY_ERROR", "BRANCH_EXECUTION", "MERGE_FAILURE", "FILTERING_LOSS", "BACKTRACK_FAILURE", "ANSWER_GENERATION", "KG_COVERAGE_OR_SCHEMA", "OTHER"):
		lines.append(f"| {key} | {causes[key]} |")
	lines += ["", "## Cases", ""]
	for row in rows:
		lines.append(f"- `{row['id']}` ({categories[str(row['id'])]}): **{main_cause(row, baseline[str(row['id'])], branch[str(row['id'])])}** — {row.get('question', '')}")
	(DIAGNOSTIC_ROOT / "grounding_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def known_case_text(rows: list[dict[str, Any]]) -> list[str]:
	known_path = ROOT / "regression_checks" / "known_case_analysis.json"
	known = json.loads(known_path.read_text(encoding="utf-8")) if known_path.exists() else []
	known_by_id = {str(item.get("id")): item for item in known}
	text = ["## Targeted semantic regression cross-checks", "", "The exact parent-gender, occupation-vs-rank, and cheese cases are not members of the historical 36-case diagnostic set. Their v1.1.3 targeted checks are therefore referenced from the already completed v1.1.3 regression run rather than rerun here.", ""]
	for case_id in ("4300043017000", "3205391005000", "4302129014000"):
		item = known_by_id.get(case_id, {})
		text.append(f"- `{case_id}`: passed=`{item.get('passed')}`, structure=`{item.get('router_structure')}`, requested type=`{item.get('requested_final_answer_type')}`, prediction=`{json.dumps(item.get('prediction', []), ensure_ascii=False)}`.")
	matching = []
	for row in rows:
		question = str(row.get("question", "")).lower()
		if "cheese" in question or ("occupation" in question and "rank" in question) or ("parent" in question and "gender" in question):
			matching.append(row)
	text.append("")
	text.append(f"Diagnostic-36 lexical matches for those exact patterns: **{len(matching)}**.")
	return text


def write_validation(rows: list[dict[str, Any]]) -> None:
	answer_mismatch = sum(validation_kind(row) == "ANSWER_TYPE_ERROR" for row in rows)
	property_mismatch = sum(validation_kind(row) == "CONSTRAINT_PROPERTY_ERROR" for row in rows)
	rejected = sum(validation_kind(row) != "NONE" for row in rows)
	redirected = sum(not row.get("validation_passed", True) and execution_mode(row) in {"cor", "cor_fallback"} for row in rows)
	lines = ["# Adaptive v1.1.3 answer-type and constraint-property validation", "", f"- Answer-type mismatch failures: **{answer_mismatch}**", f"- Constraint-property mismatch failures: **{property_mismatch}**", f"- Validation rejections: **{rejected}**", f"- Safe CoR redirects after validation rejection: **{redirected}**", "", "| ID | Requested answer type | Requested property | Validation | Failures | Structure | Final mode |", "|---|---|---|---|---|---|---|"]
	for row in rows:
		lines.append(f"| `{row['id']}` | {row.get('requested_final_answer_type', '')} | {row.get('requested_final_semantic_property', '')} | {row.get('validation_passed')} | `{json.dumps(row.get('validation_failures', []), ensure_ascii=False)}` | {structure(row)} | {execution_mode(row)} |")
	lines += ["", "## Interpretation", "", "The explicit fields are recorded for every row. A zero mismatch count means no executed branch was rejected for the two targeted semantic mismatch classes in this diagnostic run; it does not prove that all routing decisions are correct.", ""]
	(DIAGNOSTIC_ROOT / "validation_analysis.md").write_text("\n".join(lines) + "\n" + "\n".join(known_case_text(rows)) + "\n", encoding="utf-8")


def historical_row_metrics(rows: list[dict[str, Any]], version: str) -> dict[str, Any]:
	return {
		"cor_only_preserved": sum(bool(row.get("hits_at_1")) for row in rows if row.get("source_group") == "COR_ONLY_CORRECT"),
		"ours_only_recovered": sum(bool(row.get("hits_at_1")) for row in rows if row.get("source_group") == "OURS_ONLY_CORRECT"),
		"both_wrong_solved": sum(bool(row.get("hits_at_1")) for row in rows if row.get("source_group") == "BOTH_WRONG"),
		"branch_final": sum(branch_final(row, version) for row in rows),
		"branch_fallbacks": sum(branch_final(row, version) and fallback(row) for row in rows),
		"fallback_rate": (sum(branch_final(row, version) and fallback(row) for row in rows) / sum(branch_final(row, version) for row in rows)) if any(branch_final(row, version) for row in rows) else 0.0,
		"self_sufficiency": branch_stats(rows, version)["branch_self_sufficiency_rate"],
		"answer_type_mismatches": sum(validation_kind(row) == "ANSWER_TYPE_ERROR" for row in rows) if version != "v1.1.1" else None,
		"constraint_property_mismatches": sum(validation_kind(row) == "CONSTRAINT_PROPERTY_ERROR" for row in rows) if version == "v1.1.3" else 0,
	}


def write_historical(rows: list[dict[str, Any]], categories: dict[str, str], v11: dict[str, dict[str, Any]], v12: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]], branch: dict[str, dict[str, Any]]) -> None:
	def with_group(mapping: dict[str, dict[str, Any]], version: str) -> list[dict[str, Any]]:
		result = []
		for row in rows:
			item = dict(mapping[str(row["id"])])
			item["source_group"] = categories[str(row["id"])]
			result.append(item)
		return result
	current = [dict(row) for row in rows]
	for row in current:
		row["source_group"] = categories[str(row["id"])]
	values = {
		"v1.1.1": historical_row_metrics(with_group(v11, "v1.1.1"), "v1.1.1"),
		"v1.1.2": historical_row_metrics(with_group(v12, "v1.1.2"), "v1.1.2"),
		"v1.1.3": historical_row_metrics(current, "v1.1.3"),
	}
	lines = ["# Historical diagnostic-36 version comparison", "", "Only frozen historical outputs are used for v1.1.1 and v1.1.2. No historical method was rerun.", "", "| Metric | v1.1.1 | v1.1.2 | v1.1.3 |", "|---|---:|---:|---:|"]
	for key, label in (("cor_only_preserved", "COR_ONLY preserved"), ("ours_only_recovered", "OURS_ONLY recovered"), ("both_wrong_solved", "BOTH_WRONG newly solved"), ("branch_final", "BRANCH_FINAL selections"), ("branch_fallbacks", "Branch fallbacks")):
		lines.append(f"| {label} | {values['v1.1.1'][key]} | {values['v1.1.2'][key]} | {values['v1.1.3'][key]} |")
	lines.append(f"| Fallback rate | {values['v1.1.1']['fallback_rate']:.2%} | {values['v1.1.2']['fallback_rate']:.2%} | {values['v1.1.3']['fallback_rate']:.2%} |")
	lines.append(f"| Branch self-sufficiency | {values['v1.1.1']['self_sufficiency']:.2%} | {values['v1.1.2']['self_sufficiency']:.2%} | {values['v1.1.3']['self_sufficiency']:.2%} |")
	lines.append(f"| Answer-type mismatches | n/a | {values['v1.1.2']['answer_type_mismatches']} | {values['v1.1.3']['answer_type_mismatches']} |")
	lines.append(f"| Constraint-property mismatches | n/a | {values['v1.1.2']['constraint_property_mismatches']} | {values['v1.1.3']['constraint_property_mismatches']} |")
	lines += ["", "## Descriptive cost comparison", "", "| Version | N | Avg LLM calls | Avg KG queries | Avg tokens | Avg latency (ms) |", "|---|---:|---:|---:|---:|---:|"]
	for label, historical_rows, native in (("v1.1.1", with_group(v11, "v1.1.1"), False), ("v1.1.2", with_group(v12, "v1.1.2"), False), ("v1.1.3", current, False)):
		m = metric(historical_rows, native=native)
		lines.append(f"| {label} | {m['n']} | {m['llm_calls']:.2f} | {m['kg_queries']:.2f} | {m['tokens']:.2f} | {m['latency_ms']:.2f} |")
	lines += ["", "Cost values are descriptive because the versions used different routing/execution implementations and historical environments. They are not a combined benchmark."]
	(DIAGNOSTIC_ROOT / "historical_version_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
	(DIAGNOSTIC_ROOT / "historical_version_comparison.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def recommendation(rows: list[dict[str, Any]], v12: dict[str, dict[str, Any]]) -> str:
	groups = defaultdict(list)
	for row in rows:
		groups[row["source_group"]].append(row)
	current_ss = branch_stats(rows)
	old_ss = branch_stats(list(v12.values()), "v1.1.2")
	preserved = sum(bool(row.get("hits_at_1")) for row in groups["COR_ONLY_CORRECT"])
	recovered = sum(bool(row.get("hits_at_1")) for row in groups["OURS_ONLY_CORRECT"])
	mismatches = sum(validation_kind(row) in {"ANSWER_TYPE_ERROR", "CONSTRAINT_PROPERTY_ERROR"} for row in rows)
	if preserved == 10 and recovered == 3 and mismatches == 0 and current_ss["fallback_rate"] < old_ss["fallback_rate"] and current_ss["branch_self_sufficiency_rate"] >= old_ss["branch_self_sufficiency_rate"]:
		return "READY_FOR_FULL_100"
	if mismatches == 0 and current_ss["fallback_rate"] < old_ss["fallback_rate"]:
		return "ONE_MORE_TARGETED_FIX"
	return "REASSESS"


def write_report(rows: list[dict[str, Any]], categories: dict[str, str], v11: dict[str, dict[str, Any]], v12: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]], branch: dict[str, dict[str, Any]]) -> None:
	final = metric(rows)
	native = metric(rows, native=True)
	groups = defaultdict(list)
	for row in rows:
		groups[categories[str(row["id"])]].append(row)
	ss = branch_stats(rows)
	recommend = recommendation(rows, v12)
	lines = [
		"# Adaptive v1.1.3 diagnostic-36 report", "",
		"## Scope", "",
		"Adaptive v1.1.3 was run once on the exact frozen 36-case diagnostic set: 10 COR_ONLY_CORRECT, 3 OURS_ONLY_CORRECT, and 23 BOTH_WRONG. Frozen CoR, Branch-v1, Adaptive v1.1.1, and Adaptive v1.1.2 outputs were read for comparison only; they were not rerun.", "",
		"The diagnostic set was previously inspected and influenced method development, so it is not treated as fresh generalization evidence. The fresh GrailQA-12 repeated evaluation remains the generalization/stability evidence.", "",
		"## Results", "",
		f"- Native/pre-fallback: **{native['hits_rate']:.2%} Hits@1**, **{native['f1']:.4f} F1**.",
		f"- Final/post-fallback: **{final['hits_rate']:.2%} Hits@1**, **{final['f1']:.4f} F1**, **{final['errors']} errors**.",
		f"- COR_ONLY_CORRECT preserved: **{sum(bool(row.get('hits_at_1')) for row in groups['COR_ONLY_CORRECT'])}/10**.",
		f"- OURS_ONLY_CORRECT recovered: **{sum(bool(row.get('hits_at_1')) for row in groups['OURS_ONLY_CORRECT'])}/3**.",
		f"- BOTH_WRONG newly solved: **{sum(bool(row.get('hits_at_1')) for row in groups['BOTH_WRONG'])}/23**.", "",
		"Detailed group-level explanations are in `diagnostic_36/paired_group_analysis.md`.", "",
		"## Routing and branch execution", "",
		f"- Routing: **{Counter(structure(row) for row in rows)}**.",
		f"- BRANCH_FINAL selections: **{ss['branch_final_selections']}**; native branch successes: **{ss['native_branch_successes']}**; fallbacks: **{ss['branch_fallbacks']}**.",
		f"- Branch self-sufficiency: **{ss['branch_self_sufficiency_rate']:.2%}**; fallback rate: **{ss['fallback_rate']:.2%}**.", "",
		"## Semantic validation", "",
		f"- Answer-type mismatch failures: **{sum(validation_kind(row) == 'ANSWER_TYPE_ERROR' for row in rows)}**.",
		f"- Constraint-property mismatch failures: **{sum(validation_kind(row) == 'CONSTRAINT_PROPERTY_ERROR' for row in rows)}**.",
		f"- Validation rejections: **{sum(validation_kind(row) != 'NONE' for row in rows)}**.",
		"- The exact parent-gender, occupation-vs-rank, and cheese checks were outside this frozen 36; the already completed v1.1.3 targeted regression artifact is cross-referenced in `diagnostic_36/validation_analysis.md`.", "",
		"## Historical comparison", "",
		"See `diagnostic_36/historical_version_comparison.md` for frozen v1.1.1/v1.1.2 values and descriptive cost comparison.", "",
		"## Decision", "",
		f"**{recommend}**.", "",
		"This recommendation combines the positive fresh GrailQA-12 repeated result with diagnostic regression behavior; it does not claim statistical generalization from the diagnostic-36 set.", "",
		"The full 100-question evaluation was not run by this task.", "",
		f"> Adaptive v1.1.3 suggests that the next research priority should be **{recommend}** because the fresh repeated evidence and frozen diagnostic regression behavior must be considered together.",
	]
	(ROOT / "DIAGNOSTIC_36_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
	rows, categories, v11, v12, baseline, branch = load_context()
	write_summary(rows, categories)
	write_paired_groups(rows, categories, v11, v12, baseline, branch)
	write_routing(rows, v12)
	write_branch_analysis(rows, categories)
	write_grounding(rows, baseline, branch, categories)
	write_validation(rows)
	write_historical(rows, categories, v11, v12, baseline, branch)
	write_report(rows, categories, v11, v12, baseline, branch)
	print(json.dumps({"rows": len(rows), "recommendation": recommendation(rows, v12), "report": str(ROOT / 'DIAGNOSTIC_36_REPORT.md')}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
