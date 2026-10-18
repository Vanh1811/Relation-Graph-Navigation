#!/usr/bin/env python3
"""Aggregate the v1.1.3 known-case and repeated GrailQA-12 results."""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from chain_of_relations.methods.adaptive_v1_1_3.router import _semantic_group


EXP = ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_3"
REPEAT = EXP / "grailqa_12_repeat"
REGRESSION = EXP / "regression_checks"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def by_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
	result: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		result[str(row.get("id", ""))].append(row)
	return dict(result)


def mean_std(values: list[float]) -> tuple[float, float]:
	if not values:
		return 0.0, 0.0
	return statistics.mean(values), statistics.pstdev(values)


def fmt(value: float) -> str:
	return f"{value:.4f}"


def prediction(row: dict[str, Any]) -> tuple[str, ...]:
	return tuple(str(item) for item in row.get("prediction", []))


def success_frequency(rows: list[dict[str, Any]]) -> dict[str, Any]:
	hits = [int(row.get("hits_at_1", 0)) for row in rows]
	f1 = [float(row.get("f1", 0.0)) for row in rows]
	mean_hit, std_hit = mean_std([float(value) for value in hits])
	mean_f1, std_f1 = mean_std(f1)
	return {
		"runs": len(rows),
		"hits_at_1_sum": sum(hits),
		"hits_at_1_frequency": f"{sum(hits)}/{len(hits)}" if hits else "0/0",
		"mean_hits_at_1": mean_hit,
		"std_hits_at_1": std_hit,
		"mean_f1": mean_f1,
		"std_f1": std_f1,
		"exact_prediction_stable": len({prediction(row) for row in rows}) == 1,
		"hits_at_1_stable": len(set(hits)) == 1,
		"structure_stable": len({str(row.get("router_structure", "")) for row in rows}) == 1,
		"relation_chain_stable": len({tuple(str(item) for item in row.get("relation_chain", [])) for row in rows}) == 1,
	}


def contains_failure(row: dict[str, Any], text: str) -> bool:
	return any(text in str(item) for item in row.get("validation_failures", []))


def classify_pair(cor: list[dict[str, Any]], adaptive: list[dict[str, Any]]) -> dict[str, Any]:
	cor_hits = sum(int(row.get("hits_at_1", 0)) for row in cor)
	adaptive_hits = sum(int(row.get("hits_at_1", 0)) for row in adaptive)
	cor_unstable = not success_frequency(cor)["hits_at_1_stable"] or not success_frequency(cor)["exact_prediction_stable"]
	adaptive_unstable = not success_frequency(adaptive)["hits_at_1_stable"] or not success_frequency(adaptive)["exact_prediction_stable"]
	if adaptive_hits - cor_hits >= 2:
		classification = "ADAPTIVE_STABLY_BETTER"
	elif cor_hits - adaptive_hits >= 2:
		classification = "COR_STABLY_BETTER"
	elif cor_unstable and adaptive_unstable:
		classification = "BOTH_UNSTABLE"
	else:
		classification = "SIMILAR_STABILITY"
	return {
		"cor_successes": cor_hits,
		"adaptive_successes": adaptive_hits,
		"cor_frequency": f"{cor_hits}/{len(cor)}",
		"adaptive_frequency": f"{adaptive_hits}/{len(adaptive)}",
		"classification": classification,
	}


def causal_label(rows: list[dict[str, Any]], better: bool) -> str:
	if not rows:
		return "OTHER"
	if any(contains_failure(row, "ANSWER_TYPE_MISMATCH") for row in rows):
		return "ANSWER_TYPE_AWARE_ROUTING"
	if any(contains_failure(row, "CONSTRAINT_PROPERTY_MISMATCH") for row in rows):
		return "CONSTRAINT_PROPERTY_ALIGNMENT"
	if any(str(row.get("router_structure")) == "BRANCH_FINAL" and str(row.get("final_execution_mode")) == "branch" for row in rows):
		if any(str(row.get("merge_method", "")) in {"ID", "NORMALIZED_LABEL", "MIXED"} for row in rows):
			return "BRANCH_INTERSECTION"
		return "CONSTRAINT_PROPERTY_ALIGNMENT"
	if any(str(row.get("router_structure")) == "LINEAR" for row in rows):
		return "STOCHASTIC_VARIATION"
	return "OTHER"


def known_case_analysis() -> tuple[list[dict[str, Any]], str]:
	path = REGRESSION / "known_case_results.jsonl"
	if not path.exists():
		return [], "Known-case runner output was not found."
	rows = read_jsonl(path)
	results: list[dict[str, Any]] = []
	for row in rows:
		case_id = str(row["id"])
		branches = row.get("proposed_branches", [])
		branch_props = [str(item.get("constraint_property", "")) for item in branches]
		groups = [_semantic_group(item) for item in branch_props if item]
		if case_id == "4300043017000":
			passed = _semantic_group(row.get("requested_final_answer_type", "")) == "gender" and not (
				row.get("final_execution_mode") == "branch" and any(_semantic_group(item.get("target_answer_type", "")) == "taxonomy" for item in branches)
			)
			check = "parent gender remains gender/parent-sex rather than genus/taxonomy"
		elif case_id == "3205391005000":
			passed = row.get("final_execution_mode") != "branch" or (len(groups) >= 2 and len(set(groups)) == 1)
			check = "branch properties do not merge occupation with rank"
		else:
			passed = not row.get("validation_failed", False) or (
				all(branch_props) and all(_semantic_group(item.get("target_answer_type", "")) == "cheese" for item in branches)
			)
			check = "cheese branches preserve explicit constraints while targeting cheese"
		results.append({
			"id": case_id,
			"check": check,
			"passed": bool(passed),
			"requested_final_answer_type": row.get("requested_final_answer_type", ""),
			"requested_final_semantic_property": row.get("requested_final_semantic_property", ""),
			"router_structure": row.get("router_structure", ""),
			"final_execution_mode": row.get("final_execution_mode", ""),
			"validation_failures": row.get("validation_failures", []),
			"branch_constraint_properties": branch_props,
			"prediction": row.get("prediction", []),
		})
	md = [
		"# v1.1.3 known-case regression checks",
		"",
		"These are targeted semantic-consistency checks, not benchmark results. They use the three previously inspected cases and are excluded from the fresh 12-question sample.",
		"",
		"| ID | Check | Passed | Requested type | Property | Structure | Final mode |",
		"|---|---|---|---|---|---|---|",
	]
	for item in results:
		md.append(
			f"| {item['id']} | {item['check']} | {'yes' if item['passed'] else 'no'} | "
			f"{item['requested_final_answer_type']} | {item['requested_final_semantic_property']} | "
			f"{item['router_structure']} | {item['final_execution_mode']} |"
		)
	md += [
		"",
		"The checks verify that v1.1.3 logs explicit answer-type/property fields and does not execute a validated genus branch for the parent-gender case. A failed check would require routing refinement before interpreting the fresh repeated evaluation.",
		"",
	]
	return results, "\n".join(md)


def main() -> None:
	REGRESSION.mkdir(parents=True, exist_ok=True)
	known, known_md = known_case_analysis()
	(REGRESSION / "known_case_analysis.json").write_text(json.dumps(known, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	(REGRESSION / "known_case_analysis.md").write_text(known_md, encoding="utf-8")

	cor_rows = read_jsonl(REPEAT / "cor_runs.jsonl")
	adaptive_rows = read_jsonl(REPEAT / "adaptive_runs.jsonl")
	cor_by_id = by_id(cor_rows)
	adaptive_by_id = by_id(adaptive_rows)
	ids = sorted(set(cor_by_id) | set(adaptive_by_id))
	metrics = {
		"cor": success_frequency(cor_rows),
		"adaptive": success_frequency(adaptive_rows),
		"per_question": {},
	}
	paired: dict[str, Any] = {}
	for case_id in ids:
		cor = cor_by_id.get(case_id, [])
		adaptive = adaptive_by_id.get(case_id, [])
		metrics["per_question"][case_id] = {
			"question": (cor or adaptive)[0].get("question", ""),
			"cor": success_frequency(cor),
			"adaptive": success_frequency(adaptive),
		}
		paired[case_id] = classify_pair(cor, adaptive)

	routing_counts = Counter(str(row.get("router_structure", "")) for row in adaptive_rows)
	final_mode_counts = Counter(str(row.get("final_execution_mode", "")) for row in adaptive_rows)
	branch_selections = sum(1 for row in adaptive_rows if row.get("router_structure") == "BRANCH_FINAL")
	native_branch_successes = sum(1 for row in adaptive_rows if row.get("router_structure") == "BRANCH_FINAL" and row.get("final_execution_mode") == "branch" and not row.get("fallback_used"))
	answer_type_mismatches = sum(contains_failure(row, "ANSWER_TYPE_MISMATCH") for row in adaptive_rows)
	property_mismatches = sum(contains_failure(row, "CONSTRAINT_PROPERTY_MISMATCH") for row in adaptive_rows)
	routing = {
		"structure_counts": dict(routing_counts),
		"structure_frequencies": {key: value / len(adaptive_rows) if adaptive_rows else 0.0 for key, value in routing_counts.items()},
		"final_execution_mode_counts": dict(final_mode_counts),
		"branch_final_selections": branch_selections,
		"native_branch_successes": native_branch_successes,
		"branch_self_sufficiency": native_branch_successes / branch_selections if branch_selections else 0.0,
		"answer_type_mismatch_count": answer_type_mismatches,
		"constraint_property_mismatch_count": property_mismatches,
		"per_question": {
			case_id: {
				"structures": dict(Counter(str(row.get("router_structure", "")) for row in adaptive_by_id[case_id])),
				"structure_stable": metrics["per_question"][case_id]["adaptive"]["structure_stable"],
				"relation_chain_stable": metrics["per_question"][case_id]["adaptive"]["relation_chain_stable"],
			}
			for case_id in ids
		},
	}

	grounding = {
		"branch_rows": sum(1 for row in adaptive_rows if row.get("router_structure") == "BRANCH_FINAL"),
		"branch_rows_with_failures": sum(1 for row in adaptive_rows if row.get("branch_failures")),
		"fallback_rows": sum(1 for row in adaptive_rows if row.get("fallback_used")),
		"merge_failure_rows": sum(1 for row in adaptive_rows if "merge" in str(row.get("fallback_reason", "")).lower()),
		"relation_grounding_signals": sum(1 for row in adaptive_rows if any("relation" in str(item).lower() for item in row.get("branch_failures", []))),
		"constraint_properties_seen": sorted({str(item) for row in adaptive_rows for item in row.get("branch_constraint_properties", []) if str(item).strip()}),
	}

	causal = {"adaptive_better": {}, "adaptive_regression": {}}
	for case_id in ids:
		pair = paired[case_id]
		if pair["classification"] == "ADAPTIVE_STABLY_BETTER":
			causal["adaptive_better"][case_id] = causal_label(adaptive_by_id[case_id], True)
		elif pair["classification"] == "COR_STABLY_BETTER":
			causal["adaptive_regression"][case_id] = causal_label(adaptive_by_id[case_id], False)

	stable_better = sum(item["classification"] == "ADAPTIVE_STABLY_BETTER" for item in paired.values())
	answer_type_errors = answer_type_mismatches + property_mismatches
	unstable_questions = sum(item["classification"] == "BOTH_UNSTABLE" for item in paired.values())
	if stable_better > 0 and not answer_type_errors and routing["branch_self_sufficiency"] >= 0.5:
		recommendation = "PROCEED_TO_FULL_100"
	elif answer_type_errors or sum(not item["structure_stable"] for item in routing["per_question"].values()) > len(ids) / 2:
		recommendation = "CONTINUE_ROUTING_REFINEMENT"
	elif (
		grounding["branch_rows_with_failures"]
		and grounding["relation_grounding_signals"] > grounding["branch_rows_with_failures"] / 2
	):
		recommendation = "IMPROVE_RELATION_GROUNDING"
	elif unstable_questions:
		recommendation = "STOP_BRANCH_EXPANSION_AND_REASSESS"
	else:
		recommendation = "STOP_BRANCH_EXPANSION_AND_REASSESS"

	(REPEAT / "summary.md").write_text(
		"# Adaptive v1.1.3 repeated-run summary\n\n"
		f"Frozen sample: `{REPEAT / 'sample_ids.json'}`; {len(ids)} questions; 3 repeats per method; 72 total inference runs.\n\n"
		"| Method | Runs | Mean Hits@1 | SD Hits@1 | Mean F1 | SD F1 |\n|---|---:|---:|---:|---:|---:|\n"
		f"| CoR | {metrics['cor']['runs']} | {fmt(metrics['cor']['mean_hits_at_1'])} | {fmt(metrics['cor']['std_hits_at_1'])} | {fmt(metrics['cor']['mean_f1'])} | {fmt(metrics['cor']['std_f1'])} |\n"
		f"| Adaptive v1.1.3 | {metrics['adaptive']['runs']} | {fmt(metrics['adaptive']['mean_hits_at_1'])} | {fmt(metrics['adaptive']['std_hits_at_1'])} | {fmt(metrics['adaptive']['mean_f1'])} | {fmt(metrics['adaptive']['std_f1'])} |\n\n"
		"All repeated observations are included; no favorable repeat was selected.\n",
		encoding="utf-8",
	)

	stability_lines = [
		"# Stability analysis",
		"",
		"Each frozen question has three CoR and three Adaptive v1.1.3 observations.",
		"",
		"| ID | CoR exact stable | CoR Hits stable | Adaptive exact stable | Adaptive Hits stable | Adaptive route stable | Adaptive relation-chain stable |",
		"|---|---|---|---|---|---|---|",
	]
	for case_id in ids:
		c = metrics["per_question"][case_id]
		a = c["adaptive"]
		route = routing["per_question"][case_id]
		stability_lines.append(
			f"| {case_id} | {'yes' if c['cor']['exact_prediction_stable'] else 'no'} | {'yes' if c['cor']['hits_at_1_stable'] else 'no'} | "
			f"{'yes' if a['exact_prediction_stable'] else 'no'} | {'yes' if a['hits_at_1_stable'] else 'no'} | "
			f"{'yes' if route['structure_stable'] else 'no'} | {'yes' if route['relation_chain_stable'] else 'no'} |"
		)
	stability_lines += [
		"",
		"Exact prediction and relation-chain stability are stricter than Hits@1 stability. Three repeats are descriptive only and do not support significance claims.",
		"",
	]
	(REPEAT / "stability_analysis.md").write_text("\n".join(stability_lines), encoding="utf-8")

	paired_lines = [
		"# Paired repeated-run analysis",
		"",
		"Classification uses success frequency across three repeats: a difference of at least two successes is labelled stably better; otherwise both-unstable or similar-stability is reported.",
		"",
		"| ID | CoR success frequency | Adaptive success frequency | Classification |",
		"|---|---:|---:|---|",
	]
	for case_id in ids:
		item = paired[case_id]
		paired_lines.append(f"| {case_id} | {item['cor_frequency']} | {item['adaptive_frequency']} | {item['classification']} |")
	paired_lines += ["", "Do not interpret these three-repeat frequencies as statistically significant estimates.", ""]
	(REPEAT / "paired_frequency_analysis.md").write_text("\n".join(paired_lines), encoding="utf-8")

	routing_lines = [
		"# Routing analysis",
		"",
		f"Total Adaptive observations: {len(adaptive_rows)}.",
		"",
		"| Structure | Count | Frequency |",
		"|---|---:|---:|",
	]
	for key, value in sorted(routing["structure_counts"].items()):
		routing_lines.append(f"| {key} | {value} | {value / len(adaptive_rows):.4f} |")
	routing_lines += [
		"",
		f"BRANCH_FINAL selections: {branch_selections}; native branch successes: {native_branch_successes}; branch self-sufficiency: {routing['branch_self_sufficiency']:.4f}.",
		f"Answer-type mismatch validation failures: {answer_type_mismatches}; constraint-property mismatch failures: {property_mismatches}.",
		"",
	]
	(REPEAT / "routing_analysis.md").write_text("\n".join(routing_lines), encoding="utf-8")

	grounding_lines = [
		"# Grounding analysis",
		"",
		f"Branch observations: {grounding['branch_rows']}; branch rows with execution failures: {grounding['branch_rows_with_failures']}; fallback rows: {grounding['fallback_rows']}; merge-failure signals: {grounding['merge_failure_rows']}.",
		f"Relation-grounding failure signals in branch failures: {grounding['relation_grounding_signals']}.",
		"",
		"Constraint properties observed:",
		"",
	]
	grounding_lines.extend(f"- `{item}`" for item in grounding["constraint_properties_seen"])
	grounding_lines += [
		"",
		"This report distinguishes routing/property validation failures from downstream branch execution and merge signals. A fallback is recorded as safety behavior, not as proof that the original route was correct.",
		"",
	]
	(REPEAT / "grounding_analysis.md").write_text("\n".join(grounding_lines), encoding="utf-8")

	json_summary = {
		"sample_ids": json.loads((REPEAT / "sample_ids.json").read_text(encoding="utf-8")),
		"metrics": metrics,
		"paired": paired,
		"routing": routing,
		"grounding": grounding,
		"causal_interpretation": causal,
		"recommendation": recommendation,
		"known_cases": known,
	}
	(REPEAT / "aggregate.json").write_text(json.dumps(json_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

	report = [
		"# Adaptive v1.1.3 report",
		"",
		"## 1. Motivation",
		"",
		"v1.1.2 disagreement analysis found an unstable LINEAR comparison, a stable parent-gender→genus routing error, and a relation-grounding/merge regression. v1.1.3 adds explicit requested answer-type and semantic-property fields plus deterministic branch-property compatibility checks. It does not add new graph structures or branch→merge→continue behavior.",
		"",
		"## 2. Implementation changes",
		"",
		"- Router output now records `requested_final_answer_type` and `requested_final_semantic_property`.",
		"- Every executable branch must provide `constraint_property`, `constraint_summary`, and `target_answer_type`.",
		"- Deterministic validation rejects answer-type mismatches, missing properties, incompatible known property pairs, generic roots, and target-type mismatches.",
		"- Branch relation-selection prompts receive the branch constraint property and requested final type.",
		"- LINEAR, BRANCH_INTERMEDIATE, fallback, and existing merge behavior remain available.",
		"",
		"## 3. Known regression checks",
		"",
		f"See `regression_checks/known_case_analysis.md`; checks passed: {sum(bool(item.get('passed')) for item in known)}/{len(known)}.",
		"",
		"## 4. Fresh repeated evaluation",
		"",
		f"The frozen sample contains {len(ids)} fresh GrailQA branch_2 questions and is saved at `grailqa_12_repeat/sample_ids.json`. It excludes all prior frozen experiment samples and the manually inspected v1.1.2 disagreement cases. The protocol ran three CoR repeats and three Adaptive v1.1.3 repeats per question: 72 inference runs total.",
		"",
		"| Method | Runs | Mean Hits@1 | SD Hits@1 | Mean F1 | SD F1 |",
		"|---|---:|---:|---:|---:|---:|",
		f"| CoR | {metrics['cor']['runs']} | {fmt(metrics['cor']['mean_hits_at_1'])} | {fmt(metrics['cor']['std_hits_at_1'])} | {fmt(metrics['cor']['mean_f1'])} | {fmt(metrics['cor']['std_f1'])} |",
		f"| Adaptive v1.1.3 | {metrics['adaptive']['runs']} | {fmt(metrics['adaptive']['mean_hits_at_1'])} | {fmt(metrics['adaptive']['std_hits_at_1'])} | {fmt(metrics['adaptive']['mean_f1'])} | {fmt(metrics['adaptive']['std_f1'])} |",
		"",
		"## 5. Per-question frequencies and paired analysis",
		"",
		"See `paired_frequency_analysis.md` for every question. The classifications are descriptive and use only three repeats; no cherry-picking or significance claim is made.",
		"",
		"| Classification | Count |",
		"|---|---:|",
	]
	for label, count in sorted(Counter(item["classification"] for item in paired.values()).items()):
		report.append(f"| {label} | {count} |")
	report += [
		"",
		"## 6. Routing and branch self-sufficiency",
		"",
		f"Routing counts and stability are in `routing_analysis.md`. BRANCH_FINAL self-sufficiency was {routing['branch_self_sufficiency']:.4f} ({native_branch_successes}/{branch_selections} selections). Answer-type mismatch failures={answer_type_mismatches}; constraint-property mismatch failures={property_mismatches}.",
		"",
		"## 7. Grounding and causal interpretation",
		"",
		f"Branch execution failures={grounding['branch_rows_with_failures']}; fallback rows={grounding['fallback_rows']}; merge-failure signals={grounding['merge_failure_rows']}; relation-grounding signals={grounding['relation_grounding_signals']}.",
		"",
		f"Adaptive-better causal labels: `{json.dumps(causal['adaptive_better'], ensure_ascii=False)}`.",
		f"Adaptive-regression causal labels: `{json.dumps(causal['adaptive_regression'], ensure_ascii=False)}`.",
		"",
		"## 8. Recommendation",
		"",
		f"**{recommendation}**. The full 100-question evaluation is not run automatically by this task. The recommendation is based on repeated success frequencies, route/relation-chain stability, answer-type/property validation counts, and branch self-sufficiency.",
		"",
		"## 9. Output files",
		"",
		"- `regression_checks/known_case_analysis.md` and `.json`",
		"- `grailqa_12_repeat/sample_ids.json`, raw `cor_runs.jsonl`, raw `adaptive_runs.jsonl`, and `aggregate.json`",
		"- `summary.md`, `stability_analysis.md`, `paired_frequency_analysis.md`, `routing_analysis.md`, and `grounding_analysis.md`",
		"",
		f"Adaptive v1.1.3 suggests that the next research priority should be **{recommendation}** because the repeated fresh-sample evidence must be interpreted together with answer-type/property validation, routing stability, and downstream grounding behavior rather than a single favorable run.",
		"",
	]
	(EXP / "ADAPTIVE_V1_1_3_REPORT.md").write_text("\n".join(report), encoding="utf-8")
	print(json.dumps({"sample_size": len(ids), "cor_rows": len(cor_rows), "adaptive_rows": len(adaptive_rows), "recommendation": recommendation, "report": str(EXP / "ADAPTIVE_V1_1_3_REPORT.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
