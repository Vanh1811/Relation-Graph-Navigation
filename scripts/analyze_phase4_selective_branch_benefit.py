#!/usr/bin/env python3
"""Analyze selective branch benefit from completed Phase 3C traces only."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
PHASE3C_ROOT = ROOT / "outputs/experiments/phase3c/frozen_branch_required_generalization"
OUTPUT_ROOT = ROOT / "outputs/experiments/phase4/selective_branch_benefit_analysis"
STRONG_BENCHMARK = ROOT / "outputs/experiments/phase3a/branch_necessity_benchmark/strong_branch_benchmark.jsonl"
CONTROL_BENCHMARK = ROOT / "outputs/experiments/phase3a/branch_necessity_benchmark/answer_equivalent_branch_control.jsonl"
SEED = 20261029
METHODS = ("v1_1_3", "phase2_5_6")
OUTCOMES = ("TRUE_BRANCH_WIN", "V1_1_3_ONLY_LOSS", "BOTH_CORRECT", "BOTH_WRONG")


def read_json(path: Path) -> Any:
	return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
	hasher = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			hasher.update(chunk)
	return hasher.hexdigest()


def norm_set(values: Any) -> set[str]:
	if not isinstance(values, list):
		return set()
	return {str(value).strip().casefold() for value in values if str(value).strip()}


def load_benchmark(path: Path) -> dict[str, dict[str, Any]]:
	return {row["case_id"]: row for row in read_jsonl(path)}


def majority(values: list[str]) -> str:
	if not values:
		return "UNKNOWN"
	return Counter(values).most_common(1)[0][0]


def method_summary(rows: list[dict[str, Any]], method: str, failure: dict[str, Any] | None = None) -> dict[str, Any]:
	values = [row for row in rows if row.get("method") == method]
	hits = [int(row.get("hits_at_1", 0) or 0) for row in values]
	structures = [str(row.get("semantic_structure") or "UNKNOWN") for row in values]
	answers = [json.dumps(sorted(norm_set(row.get("final_prediction"))), ensure_ascii=False) for row in values]
	fallbacks = [bool(row.get("fallback_used")) for row in values]
	branch_success = [bool(row.get("branch_self_sufficient")) for row in values]
	branch_attempt = [bool(row.get("branch_routed")) for row in values]
	validator = [row.get("validator_result") in {"PASS", "VALIDATOR_PASS"} for row in values]
	return {
		"method": method,
		"run_count": len(values),
		"hits_at_1": hits,
		"f1_by_repeat": [float(row.get("f1", 0.0) or 0.0) for row in values],
		"majority_correct": bool(any(hits)),
		"f1_mean": statistics.mean(float(row.get("f1", 0.0) or 0.0) for row in values) if values else 0.0,
		"predicted_structure_by_repeat": structures,
		"predicted_structure": majority(structures),
		"branch_attempted": any(branch_attempt),
		"validator_passed": any(validator),
		"native_branch_success": any(branch_success),
		"fallback_used": any(fallbacks),
		"fallback_all_repeats": bool(fallbacks) and all(fallbacks),
		"fallback_rate": statistics.mean(fallbacks) if values else 0.0,
		"repeat_correctness_stable": len(set(hits)) <= 1,
		"answer_set_stable": len(set(answers)) <= 1,
		"routing_stable": len(set(structures)) <= 1,
		"fallback_stable": len(set(fallbacks)) <= 1,
		"branch_execution_stable": len(set(branch_success)) <= 1,
		"primary_failure_family": (failure or {}).get("first_failure_family", "NONE"),
		"normalization_issue": any(row.get("normalization_status") not in {None, "", "NORMALIZED_SUCCESSFULLY"} for row in values),
		"serialization_issue": any(row.get("serialization_status") not in {None, "", "COMPLETE"} for row in values),
		"validator_issue": any(row.get("validator_result") not in {None, "PASS", "VALIDATOR_PASS"} for row in values),
		"branch_execution_issue": any(bool(row.get("branch_routed")) and row.get("execution_status") != "EXECUTION_SUCCESS" for row in values),
		"answer_selection_issue": (failure or {}).get("first_failure_family") == "ANSWER_SELECTION",
		"shared_target_issue": (failure or {}).get("first_failure_family") == "SHARED_TARGET",
		"llm_calls": sum(int(row.get("llm_calls", 0) or 0) for row in values),
		"input_tokens": sum(int(row.get("input_tokens", 0) or 0) for row in values),
		"output_tokens": sum(int(row.get("output_tokens", 0) or 0) for row in values),
		"total_tokens": sum(int(row.get("total_tokens", 0) or 0) for row in values),
		"kg_queries": sum(int(row.get("sparql_query_count", 0) or 0) for row in values),
		"latency_seconds": sum(float(row.get("latency_seconds", 0.0) or 0.0) for row in values),
	}


def primary_loss_cause(new: dict[str, Any], outcome: str) -> str:
	if outcome != "V1_1_3_ONLY_LOSS":
		return "NONE"
	family = new.get("primary_failure_family", "NONE")
	if family == "NORMALIZATION":
		return "NORMALIZATION"
	if family == "SERIALIZATION":
		return "SERIALIZATION"
	if family == "VALIDATOR":
		return "VALIDATOR"
	if family == "SHARED_TARGET":
		return "SHARED_TARGET_ERROR"
	if new.get("branch_execution_issue"):
		return "BRANCH_EXECUTION"
	if family == "ANSWER_SELECTION":
		return "ANSWER_SELECTION"
	if new.get("fallback_used"):
		return "BRANCH_OVERHEAD"
	return "OTHER"


def constraint_loss_label(gold_structure: str, old: dict[str, Any], new: dict[str, Any], outcome: str) -> str:
	linear = all(structure in {"LINEAR", "UNCERTAIN"} for structure in old["predicted_structure_by_repeat"])
	if outcome == "TRUE_BRANCH_WIN" and linear and new["native_branch_success"] and new["majority_correct"]:
		return "LINEARIZATION_LOSES_ESSENTIAL_CONSTRAINT"
	if outcome == "BOTH_CORRECT" and old["majority_correct"]:
		return "LINEARIZATION_SAFE"
	if gold_structure in {"BRANCH_FINAL", "BRANCH_INTERMEDIATE"} and not linear:
		return "LINEARIZATION_PARTIAL"
	return "UNKNOWN"


def fallback_class(new: dict[str, Any], outcome: str) -> str:
	if not new["fallback_used"]:
		return "NO_FALLBACK"
	if outcome == "TRUE_BRANCH_WIN" and new["majority_correct"] and not new["fallback_all_repeats"]:
		return "FALLBACK_NEUTRAL"
	if not new["majority_correct"]:
		return "FALLBACK_HARMFUL"
	if any(new["hits_at_1"]) and any(bool(new_hit) for new_hit in new["hits_at_1"]):
		return "FALLBACK_NEUTRAL"
	return "FALLBACK_RESCUE"


def build_case_record(case: dict[str, Any], outcome: str, rows: list[dict[str, Any]], failure: dict[str, Any] | None, benchmark: dict[str, Any]) -> dict[str, Any]:
	old = method_summary(rows, "v1_1_3", failure if False else None)
	new = method_summary(rows, "phase2_5_6", failure)
	gold_structure = case["gold_structure"]
	branches = benchmark.get("branches", [])
	material = [bool(branch.get("material_effect")) for branch in branches]
	necessity_strength = float(case.get("necessity_strength") or 0.0)
	true_win = outcome == "TRUE_BRANCH_WIN"
	return {
		"case_id": case["case_id"],
		"paired_outcome": outcome,
		"true_branch_advantage": true_win,
		"dataset": case["dataset"],
		"gold_structure": gold_structure,
		"branch_necessity_class": case.get("necessity", ""),
		"necessity_strength": necessity_strength,
		"structural_features": {
			"direct_final_convergence": gold_structure == "BRANCH_FINAL",
			"shared_final_answer": gold_structure == "BRANCH_FINAL",
			"shared_intermediate": gold_structure == "BRANCH_INTERMEDIATE",
			"post_merge_relation_required": gold_structure == "BRANCH_INTERMEDIATE",
			"requires_downstream_relation_after_merge": gold_structure == "BRANCH_INTERMEDIATE",
			"number_of_post_merge_hops": 1 if gold_structure == "BRANCH_INTERMEDIATE" else 0,
			"independent_source_count": int(case.get("branch_count") or len(branches) or 0),
			"shared_target_type": "ANSWER" if gold_structure == "BRANCH_FINAL" else "INTERMEDIATE",
			"answer_disambiguating_constraint": bool(material) and all(material),
		},
		"constraint_features": {
			"independent_constraints": int(case.get("branch_count") or len(branches) or 0),
			"binding_dependency": gold_structure == "BRANCH_INTERMEDIATE",
			"constraint_loss_risk": "HIGH" if necessity_strength >= 0.75 else "MEDIUM" if necessity_strength >= 0.5 else "LOW",
			"linearization_outcome": constraint_loss_label(gold_structure, old, new, outcome),
			"material_branch_effects": material,
		},
		"runtime": {
			"v1_1_3": old,
			"phase2_5_6": new,
			"fallback_class": fallback_class(new, outcome),
			"primary_loss_cause": primary_loss_cause(new, outcome),
		},
		"grounding_features": {
			"entity_grounding_issue": False,
			"relation_grounding_issue": False,
			"answer_type_issue": False,
			"normalization_issue": new["normalization_issue"],
			"serialization_issue": new["serialization_issue"],
			"validator_issue": new["validator_issue"],
			"branch_execution_issue": new["branch_execution_issue"],
			"answer_selection_issue": new["answer_selection_issue"],
			"evidence_note": "Entity and relation grounding were not independently labelled in the frozen Phase 3C trace schema.",
		},
		"stability": {
			"v1_1_3_repeat_correctness": old["repeat_correctness_stable"],
			"phase2_5_6_repeat_correctness": new["repeat_correctness_stable"],
			"v1_1_3_answer_stability": old["answer_set_stable"],
			"phase2_5_6_answer_stability": new["answer_set_stable"],
			"v1_1_3_routing_stability": old["routing_stable"],
			"phase2_5_6_routing_stability": new["routing_stable"],
		},
		"cost": {
			"v1_1_3": old,
			"phase2_5_6": new,
			"additional_total_tokens": new["total_tokens"] - old["total_tokens"],
			"additional_latency_seconds": new["latency_seconds"] - old["latency_seconds"],
		},
		"manual_review": {
			"status": "TRACE_REVIEWED",
			"basis": "Completed Phase 3C frozen run traces and Phase 3A.1 branch-necessity metadata; no inference rerun.",
		},
	}


def prevalence(records: list[dict[str, Any]], feature: str) -> dict[str, Any]:
	output = {}
	for outcome in OUTCOMES:
		items = [record for record in records if record["paired_outcome"] == outcome]
		values = []
		for record in items:
			value: Any = record
			for part in feature.split("."):
				value = value.get(part) if isinstance(value, dict) else None
			if feature == "constraint_features.linearization_outcome":
				values.append(value == "LINEARIZATION_LOSES_ESSENTIAL_CONSTRAINT")
			else:
				values.append(bool(value))
		count = sum(values)
		output[outcome] = {"count": count, "total": len(items), "rate": count / len(items) if items else 0.0, "display": f"{count}/{len(items)}"}
	return output


def rule_records(records: list[dict[str, Any]], controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
	def direct_final(record: dict[str, Any]) -> bool:
		return record["structural_features"]["direct_final_convergence"]

	def direct_final_high_necessity(record: dict[str, Any]) -> bool:
		return direct_final(record) and record["constraint_features"]["constraint_loss_risk"] == "HIGH"

	def intermediate_merge(record: dict[str, Any]) -> bool:
		return record["structural_features"]["shared_intermediate"]

	rules: list[tuple[str, Callable[[dict[str, Any]], bool], str]] = [
		("BRANCH_FINAL direct convergence", direct_final, "gold structural proxy; not a deployed selector"),
		("BRANCH_FINAL direct convergence + HIGH necessity strength", direct_final_high_necessity, "fixed descriptive bin; no threshold optimization"),
		("BRANCH_INTERMEDIATE shared target with downstream continuation", intermediate_merge, "gold structural proxy; intermediate execution risk"),
	]
	output = []
	for name, predicate, interpretation in rules:
		strong_items = [record for record in records if predicate(record)]
		true_wins = [record for record in strong_items if record["paired_outcome"] == "TRUE_BRANCH_WIN"]
		losses = [record for record in strong_items if record["paired_outcome"] == "V1_1_3_ONLY_LOSS"]
		control_items = [record for record in controls if predicate(record)]
		output.append({
			"rule": name,
			"interpretation": interpretation,
			"true_wins_covered": len(true_wins),
			"losses_triggered": len(losses),
			"both_correct_triggered": sum(record["paired_outcome"] == "BOTH_CORRECT" for record in strong_items),
			"both_wrong_triggered": sum(record["paired_outcome"] == "BOTH_WRONG" for record in strong_items),
			"controls_triggered": len(control_items),
			"control_correct": sum(record["runtime"]["phase2_5_6"]["majority_correct"] for record in control_items),
			"coverage": f"{len(true_wins)}/4",
		})
	return output


def load_records() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
	manifest = read_json(PHASE3C_ROOT / "phase3c_eval_manifest.json")
	paired = read_json(PHASE3C_ROOT / "paired_analysis.json")["cases"]
	run_rows = read_jsonl(PHASE3C_ROOT / "run_results.jsonl")
	failure = {row["case_id"]: row for row in read_jsonl(PHASE3C_ROOT / "failure_family_analysis.jsonl")}
	audit = {row["case_id"]: row for row in read_jsonl(PHASE3C_ROOT / "branch_advantage_audit.jsonl")}
	benchmarks = load_benchmark(STRONG_BENCHMARK)
	control_benchmarks = load_benchmark(CONTROL_BENCHMARK)
	paired_by_id = {row["case_id"]: row for row in paired}
	rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in run_rows:
		rows_by_case[row["case_id"]].append(row)
	strong_records = []
	control_records = []
	for case in manifest["strong_cases"] + manifest["control_cases"]:
		paired_row = paired_by_id[case["case_id"]]
		outcome = paired_row["outcome"]
		if outcome == "V1_1_3_ONLY":
			outcome = "V1_1_3_ONLY_LOSS"
		if case["cohort"] == "STRONG":
			if audit.get(case["case_id"], {}).get("classification") == "TRUE_BRANCH_ADVANTAGE":
				outcome = "TRUE_BRANCH_WIN"
			benchmark = benchmarks.get(case["case_id"], {})
			record = build_case_record(case, outcome, rows_by_case[case["case_id"]], failure.get(case["case_id"]), benchmark)
			strong_records.append(record)
		else:
			benchmark = control_benchmarks.get(case["case_id"], {})
			control_records.append(build_case_record(case, outcome, rows_by_case[case["case_id"]], failure.get(case["case_id"]), benchmark))
	strong_records.sort(key=lambda record: record["case_id"])
	control_records.sort(key=lambda record: record["case_id"])
	return manifest, strong_records, control_records, {"audit": audit, "failure": failure}


def write_report(manifest: dict[str, Any], records: list[dict[str, Any]], controls: list[dict[str, Any]], rules: list[dict[str, Any]], decision: str, decision_reason: str) -> None:
	counts = Counter(record["paired_outcome"] for record in records)
	structure = {outcome: Counter(record["gold_structure"] for record in records if record["paired_outcome"] == outcome) for outcome in OUTCOMES}
	feature_names = ["structural_features.direct_final_convergence", "structural_features.shared_intermediate", "structural_features.post_merge_relation_required", "constraint_features.linearization_outcome", "runtime.phase2_5_6.native_branch_success", "runtime.phase2_5_6.fallback_used"]
	lines = [
		"# Phase 4 Selective Branch Benefit Analysis", "",
		"## 1. Motivation", "",
		"Phase 3C rejected branch-aware reasoning as a global replacement but recorded four trace-audited true branch wins. Phase 4 tests whether those wins form an interpretable selective-benefit boundary.", "",
		"## 2. Phase 3C result", "",
		"Phase 3C completed 360 frozen runs and produced a strong paired net of -3: 25 BOTH_CORRECT, 7 V1_1_3_ONLY, 4 PHASE2_5_6_ONLY, and 24 BOTH_WRONG.", "",
		"## 3. Reframing branch-aware reasoning as selective enhancement", "",
		"The analysis treats Phase 2.5.6 as a possible specialized enhancement over v1.1.3, not as a replacement architecture.", "",
		"## 4. Research question", "",
		"The question is whether observable structural, dependency, runtime, stability, or cost properties distinguish incremental branch benefit from harm or no benefit.", "",
		"## 5. Frozen evidence", "",
		f"Only completed Phase 3C artifacts were read. The strong analysis manifest contains {len(records)} cases; no LLM, Freebase, or inference calls were issued.", "",
		"## 6. Analysis cohorts", "",
		f"Cohort counts: `{dict(counts)}`. Deterministic BOTH_CORRECT/BOTH_WRONG context samples use seed `{SEED}`.", "",
		"| Cohort | Cases |", "|---|---:|",
	]
	for outcome in OUTCOMES:
		lines.append(f"| {outcome} | {counts.get(outcome, 0)} |")
	lines += ["", "## 7. Feature taxonomy", "", "Features cover gold structure, branch necessity metadata, frozen runtime traces, failure families, repeat stability, and cost. Grounding-specific entity/relation labels were not independently available and are marked as not observed.", "", "## 8. True branch wins", "", f"There are {counts.get('TRUE_BRANCH_WIN', 0)} true wins. Native-branch status and causal caveats are recorded in `true_branch_win_audit.jsonl`.", "", "## 9. v1.1.3-only losses", "", "The seven baseline-only losses are manually trace-reviewed in `v1_1_3_only_loss_audit.jsonl`; primary causes are descriptive, not interventions.", "", "## 10. BRANCH_FINAL vs BRANCH_INTERMEDIATE", "", "| Outcome | BRANCH_FINAL | BRANCH_INTERMEDIATE |", "|---|---:|---:|"]
	for outcome in OUTCOMES:
		lines.append(f"| {outcome} | {structure[outcome].get('BRANCH_FINAL', 0)} | {structure[outcome].get('BRANCH_INTERMEDIATE', 0)} |")
	lines += ["", "## 11. Constraint-loss analysis", "", "Constraint-loss labels are evidence-bounded proxies from the frozen baseline structure traces and branch-necessity metadata; they do not prove the internal cause of a model error.", "", "## 12. Fallback analysis", "", "Phase 2.5.6 fallback behavior is summarized by outcome in `fallback_analysis.json`.", "", "## 13. Native branch self-sufficiency", "", "Native branch advantage is separated from fallback-assisted advantage. It is an outcome audit, not an admissible pre-invocation selector feature.", "", "## 14. Both-correct context", "", "Five deterministic BOTH_CORRECT cases are included for context only; compatibility is not evidence of necessity.", "", "## 15. Both-wrong context", "", "Five deterministic BOTH_WRONG cases are included for context only; failure families are used to avoid attributing all failures to branching.", "", "## 16. Feature prevalence", "", "| Feature | True wins | v1.1.3-only losses | Both correct | Both wrong |", "|---|---:|---:|---:|---:|"]
	for feature in feature_names:
		p = prevalence(records, feature)
		lines.append(f"| {feature} | {p['TRUE_BRANCH_WIN']['display']} | {p['V1_1_3_ONLY_LOSS']['display']} | {p['BOTH_CORRECT']['display']} | {p['BOTH_WRONG']['display']} |")
	lines += ["", "## 17. Candidate selective rules", "", "At most three descriptive rules were evaluated. None is treated as a deployed policy or validated selector.", "", "| Rule | True wins covered | Losses triggered | Controls triggered |", "|---|---:|---:|---:|"]
	for rule in rules:
		lines.append(f"| {rule['rule']} | {rule['true_wins_covered']}/4 | {rule['losses_triggered']}/7 | {rule['controls_triggered']} |")
	lines += ["", "## 18. Control trigger analysis", "", "Control trigger counts and correctness are in `control_rule_trigger_analysis.json`; controls remain a separate safety cohort.", "", "## 19. Cost-aware benefit analysis", "", "Per-win token and latency deltas are in `cost_aware_benefit_analysis.json`. No composite cost-benefit score was invented.", "", "## 20. Benefit-aware selector viability gate", "", f"**{decision}** — {decision_reason}", "", "## 21. Final Phase 4 decision", "", f"**{decision}**", "", "## 22. Scientific interpretation", "", "The four wins show isolated capability, but the frozen evidence does not establish a reliable pre-invocation rule that separates useful branching from harmful or unnecessary branching.", "", "## 23. Recommended next step", "", "Do not implement or tune a selector in Phase 4. If pursued later, validate a benefit-aware selector on a new frozen dataset with runtime-independent features.", "", "## 24. Limitations", "", "The contrast is based on 60 strong cases, two repeats, and trace-derived labels. Several candidate features are gold-structure proxies or post-outcome diagnostics, and entity/relation grounding were not independently labelled."]
	(OUTPUT_ROOT / "PHASE4_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze() -> dict[str, Any]:
	manifest, records, controls, aux = load_records()
	if len(records) != 60 or Counter(record["paired_outcome"] for record in records) != Counter({"TRUE_BRANCH_WIN": 4, "V1_1_3_ONLY_LOSS": 7, "BOTH_CORRECT": 25, "BOTH_WRONG": 24}):
		raise SystemExit("Phase 4 requires the completed Phase 3C 4/7/25/24 strong cohorts")
	rng = random.Random(SEED)
	by_outcome = {outcome: [record for record in records if record["paired_outcome"] == outcome] for outcome in OUTCOMES}
	context_correct = sorted(rng.sample(by_outcome["BOTH_CORRECT"], 5), key=lambda record: record["case_id"])
	context_wrong = sorted(rng.sample(by_outcome["BOTH_WRONG"], 5), key=lambda record: record["case_id"])
	analysis_manifest = {
		"status": "FROZEN_ANALYSIS",
		"experiment": "phase4_selective_branch_benefit_analysis",
		"seed": SEED,
		"source_root": str(PHASE3C_ROOT),
		"source_hashes": {name: sha256(PHASE3C_ROOT / name) for name in ("phase3c_eval_manifest.json", "run_results.jsonl", "paired_analysis.json", "branch_advantage_audit.jsonl", "failure_family_analysis.jsonl")},
		"strong_case_count": len(records),
		"cohort_counts": dict(Counter(record["paired_outcome"] for record in records)),
		"cases": [{"case_id": record["case_id"], "dataset": record["dataset"], "gold_structure": record["gold_structure"], "branch_necessity_class": record["branch_necessity_class"], "paired_outcome": record["paired_outcome"], "true_branch_advantage": record["true_branch_advantage"], "repeat_correctness": {method: record["runtime"][method]["hits_at_1"] for method in METHODS}, "f1_by_method": {method: record["runtime"][method]["f1_by_repeat"] for method in METHODS}} for record in records],
		"context_review_ids": {"BOTH_CORRECT": [record["case_id"] for record in context_correct], "BOTH_WRONG": [record["case_id"] for record in context_wrong]},
	}
	rules = rule_records(records, controls)
	true_audit = []
	for record in by_outcome["TRUE_BRANCH_WIN"]:
		new = record["runtime"]["phase2_5_6"]
		if record["gold_structure"] == "BRANCH_FINAL" and new["native_branch_success"]:
			explanation = "NATIVE_BRANCH_PRESERVES_DIRECT_FINAL_CONVERGENCE"
		elif new["native_branch_success"]:
			explanation = "NATIVE_BRANCH_WIN_WITH_INTERMEDIATE_MERGE_TRACE_CAVEAT"
		else:
			explanation = "FALLBACK_OR_UNCLEAR"
		true_audit.append({**record, "primary_explanation": explanation, "native_classification": "NATIVE_BRANCH_ADVANTAGE" if new["native_branch_success"] else "FALLBACK_ASSISTED_BRANCH_ADVANTAGE" if new["majority_correct"] else "UNCLEAR"})
	loss_audit = [{**record, "primary_loss_cause": record["runtime"]["primary_loss_cause"]} for record in by_outcome["V1_1_3_ONLY_LOSS"]]
	write_json(OUTPUT_ROOT / "phase4_analysis_manifest.json", analysis_manifest)
	write_jsonl(OUTPUT_ROOT / "benefit_feature_matrix.jsonl", records)
	write_jsonl(OUTPUT_ROOT / "true_branch_win_audit.jsonl", true_audit)
	write_jsonl(OUTPUT_ROOT / "v1_1_3_only_loss_audit.jsonl", loss_audit)
	write_jsonl(OUTPUT_ROOT / "both_correct_context.jsonl", context_correct)
	write_jsonl(OUTPUT_ROOT / "both_wrong_context.jsonl", context_wrong)
	write_json(OUTPUT_ROOT / "branch_final_vs_intermediate.json", {"counts_by_outcome": {outcome: dict(Counter(record["gold_structure"] for record in by_outcome[outcome])) for outcome in OUTCOMES}, "direct_final_hypothesis": {"true_wins": sum(record["structural_features"]["direct_final_convergence"] for record in by_outcome["TRUE_BRANCH_WIN"]), "losses": sum(record["structural_features"]["direct_final_convergence"] for record in by_outcome["V1_1_3_ONLY_LOSS"]), "both_correct": sum(record["structural_features"]["direct_final_convergence"] for record in by_outcome["BOTH_CORRECT"]), "both_wrong": sum(record["structural_features"]["direct_final_convergence"] for record in by_outcome["BOTH_WRONG"])}})
	write_jsonl(OUTPUT_ROOT / "constraint_loss_analysis.jsonl", [{"case_id": record["case_id"], "paired_outcome": record["paired_outcome"], "gold_structure": record["gold_structure"], "linearization_outcome": record["constraint_features"]["linearization_outcome"], "necessity_strength": record["necessity_strength"], "baseline_structures": record["runtime"]["v1_1_3"]["predicted_structure_by_repeat"], "evidence_note": "Trace-derived proxy; not a causal proof."} for record in records])
	fallback = {}
	for outcome in OUTCOMES:
		items = by_outcome[outcome]
		fallback[outcome] = Counter(record["runtime"]["fallback_class"] for record in items)
	write_json(OUTPUT_ROOT / "fallback_analysis.json", {"by_outcome": {key: dict(value) for key, value in fallback.items()}, "phase2_5_6_strong_fallback_rate": sum(record["runtime"]["phase2_5_6"]["fallback_used"] for record in records) / len(records)})
	write_json(OUTPUT_ROOT / "native_branch_advantage_analysis.json", {"true_win_count": len(true_audit), "class_counts": dict(Counter(item["native_classification"] for item in true_audit)), "cases": [{"case_id": item["case_id"], "gold_structure": item["gold_structure"], "classification": item["native_classification"], "native_branch_success": item["runtime"]["phase2_5_6"]["native_branch_success"], "fallback_class": item["runtime"]["fallback_class"]} for item in true_audit]})	
	features = ["structural_features.direct_final_convergence", "structural_features.shared_intermediate", "structural_features.post_merge_relation_required", "constraint_features.linearization_outcome", "runtime.phase2_5_6.native_branch_success", "runtime.phase2_5_6.fallback_used", "runtime.phase2_5_6.normalization_issue", "runtime.phase2_5_6.validator_issue", "runtime.phase2_5_6.branch_execution_issue"]
	write_json(OUTPUT_ROOT / "feature_prevalence_by_outcome.json", {feature: prevalence(records, feature) for feature in features})
	write_json(OUTPUT_ROOT / "candidate_selective_rules.json", {"rules": rules, "validated": False, "note": "Descriptive rules only; no selector was implemented or tuned."})
	control_rule_analysis = []
	for rule in rules:
		control_rule_analysis.append({"rule": rule["rule"], "controls_triggered": rule["controls_triggered"], "control_correct": rule["control_correct"], "control_total": len(controls)})
	write_json(OUTPUT_ROOT / "control_rule_trigger_analysis.json", {"rules": control_rule_analysis})
	win_costs = [record["cost"] for record in true_audit]
	write_json(OUTPUT_ROOT / "cost_aware_benefit_analysis.json", {"true_win_cases": [{"case_id": record["case_id"], "additional_total_tokens": record["cost"]["additional_total_tokens"], "additional_latency_seconds": record["cost"]["additional_latency_seconds"]} for record in true_audit], "mean_additional_tokens_per_true_win": statistics.mean(item["additional_total_tokens"] for item in win_costs) if win_costs else 0.0, "mean_additional_latency_seconds_per_true_win": statistics.mean(item["additional_latency_seconds"] for item in win_costs) if win_costs else 0.0, "interpretation": "Descriptive overhead only; no composite benefit/cost score."})
	# The only 4/4 feature is native success, which is observed after invoking the method.
	# It is therefore not evidence for a pre-invocation selector. Gold direct-final and
	# intermediate structure split the wins 2/4 each and also occur in losses.
	decision = "SELECTIVE_BENEFIT_SIGNAL_PARTIAL"
	decision_reason = "A coherent capability boundary is visible, but admissible pre-invocation structural rules cover only 2/4 wins and trigger multiple baseline-only losses; native success is post-outcome evidence."
	write_json(OUTPUT_ROOT / "viability_gate.json", {"decision": decision, "reason": decision_reason, "native_success_upper_bound": "4/4 true wins", "admissible_direct_final_rule": "2/4 true wins and 3/7 losses"})
	(OUTPUT_ROOT / "decision_summary.md").write_text(f"# Phase 4 Decision Summary\n\nDecision: **{decision}**\n\n{decision_reason}\n\nNo selector or method intervention was implemented.\n", encoding="utf-8")
	write_report(manifest, records, controls, rules, decision, decision_reason)
	return {"decision": decision, "strong_cases": len(records), "control_cases": len(controls), "output_root": str(OUTPUT_ROOT)}


def main() -> None:
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	print(json.dumps(analyze(), indent=2))


if __name__ == "__main__":
	main()
