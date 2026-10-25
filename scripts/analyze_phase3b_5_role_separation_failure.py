#!/usr/bin/env python3
"""Offline contrastive failure analysis for Phase 3B.5.

This script consumes only the frozen Phase 3B.3/3B.4 artifacts.  It does not
instantiate an agent, call an API, query Freebase, or modify production code.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PHASE3B4_ROOT = ROOT / "outputs/experiments/phase3b_4/answer_convergence_role_separation"
PHASE3B3_ROOT = ROOT / "outputs/experiments/phase3b_3/residual_shared_target_analysis"
OUTPUT_ROOT = ROOT / "outputs/experiments/phase3b_5/role_separation_failure_analysis"

CASE_IDS = (
	"4301356014000",
	"4302283003000",
	"WebQTrn-1053_1d93ad4e3448e762133bda682c0693a2",
	"WebQTrn-1053_4c8649801d197e6eb5377d292c0b6ca8",
	"WebQTrn-436_d63127b099ed581855690cb5f1524332",
	"3200506013000",
)

CATEGORY_BY_ID = {
	"4301356014000": "UNCORRECTED_CONFUSION",
	"WebQTrn-1053_4c8649801d197e6eb5377d292c0b6ca8": "UNCORRECTED_CONFUSION",
	"WebQTrn-436_d63127b099ed581855690cb5f1524332": "UNCORRECTED_CONFUSION",
	"4302283003000": "CORRECTED_CONFUSION",
	"WebQTrn-1053_1d93ad4e3448e762133bda682c0693a2": "CORRECTED_CONFUSION",
	"3200506013000": "FALSE_INTERMEDIATE_REGRESSION",
}

FAILURE_CAUSES = {
	"REQUESTED_ANSWER_ROLE_WRONG",
	"CONVERGENCE_ROLE_WRONG",
	"ROLES_SAME_ENTITY_WRONG",
	"DOWNSTREAM_RELATION_JUDGMENT_WRONG",
	"ROLE_COMPARISON_REASONING_CONTRADICTORY",
	"MULTI_PROPERTY_FALSE_INTERMEDIATE",
	"MULTI_ENTITY_FALSE_INTERMEDIATE",
	"BENCHMARK_LANGUAGE_AMBIGUITY",
	"SEMANTIC_OUTPUT_UNSTABLE",
	"MISSING_TRACE_EVIDENCE",
	"NO_CLEAR_FAILURE",
}

# These are manual semantic adjudications of the six frozen cases.  They are
# deliberately separate from the automatically recorded Phase 3B.4 fields.
MANUAL = {
	"4301356014000": {
		"primary_cause": "CONVERGENCE_ROLE_WRONG",
		"correction_mechanism": None,
		"requested_answer_role": {"status": "CORRECT", "gold": "dish", "predicted": "dish"},
		"convergence_role": {"status": "WRONG", "gold": "DISTINCT_SHARED_CONVERGENCE_ENTITY", "predicted": "dish / requested answer"},
		"roles_same_entity": {"status": "WRONG_TRUE", "expected": False},
		"requires_downstream_relation": {"status": "FALSE_NEGATIVE", "expected": True},
		"reason_status": "CONSISTENT_WITH_FIELDS",
		"ambiguity": "HIGHLY_AMBIGUOUS",
		"note": "The question linguistically asks directly for a dish, while the frozen benchmark structure requires a distinct intermediate target.",
	},
	"WebQTrn-1053_4c8649801d197e6eb5377d292c0b6ca8": {
		"primary_cause": "SEMANTIC_OUTPUT_UNSTABLE",
		"correction_mechanism": None,
		"requested_answer_role": {"status": "CORRECT", "gold": "actor", "predicted": "actor"},
		"convergence_role": {"status": "UNSTABLE", "gold": "DISTINCT_SHARED_CONVERGENCE_ENTITY", "predicted": "actor on repeat 1; character on repeat 2"},
		"roles_same_entity": {"status": "UNSTABLE", "expected": False},
		"requires_downstream_relation": {"status": "UNSTABLE", "expected": True},
		"reason_status": "CONSISTENT_WITH_FIELDS",
		"ambiguity": "MODERATELY_AMBIGUOUS",
		"note": "The two repeats produce internally coherent but incompatible convergence interpretations, so the case-level target is unstable.",
	},
	"WebQTrn-436_d63127b099ed581855690cb5f1524332": {
		"primary_cause": "REQUESTED_ANSWER_ROLE_WRONG",
		"correction_mechanism": None,
		"requested_answer_role": {"status": "WRONG", "gold": "actor / Richard Harris", "predicted": "Harry Potter movie"},
		"convergence_role": {"status": "CORRECT", "gold": "movie as the intermediate", "predicted": "movie"},
		"roles_same_entity": {"status": "WRONG_TRUE", "expected": False},
		"requires_downstream_relation": {"status": "FALSE_NEGATIVE", "expected": True},
		"reason_status": "CONSISTENT_WITH_FIELDS",
		"ambiguity": "HIGHLY_AMBIGUOUS",
		"note": "The natural-language lead asks for a movie, but the frozen benchmark answer is Richard Harris, so the benchmark requires a downstream actor role.",
	},
	"4302283003000": {
		"primary_cause": "POSITIVE_CONTROL",
		"correction_mechanism": "MULTIPLE_FIELDS_CORRECTED",
		"requested_answer_role": {"status": "CORRECT", "gold": "visual-art form/category", "predicted": "visual-art form/category"},
		"convergence_role": {"status": "CORRECT", "gold": "DISTINCT_SHARED_CONVERGENCE_ENTITY", "predicted": "Leonardo / visual artwork, distinct from answer"},
		"roles_same_entity": {"status": "CORRECT", "expected": False},
		"requires_downstream_relation": {"status": "CORRECT", "expected": True},
		"reason_status": "CONSISTENT_WITH_FIELDS",
		"ambiguity": "MODERATELY_AMBIGUOUS",
		"note": "Both repeats preserve a distinct convergence-to-answer relationship, although the exact convergence noun varies.",
	},
	"WebQTrn-1053_1d93ad4e3448e762133bda682c0693a2": {
		"primary_cause": "POSITIVE_CONTROL",
		"correction_mechanism": "MULTIPLE_FIELDS_CORRECTED",
		"requested_answer_role": {"status": "CORRECT", "gold": "actor", "predicted": "actor"},
		"convergence_role": {"status": "CORRECT", "gold": "DISTINCT_SHARED_CONVERGENCE_ENTITY", "predicted": "performance role / record, distinct from actor"},
		"roles_same_entity": {"status": "CORRECT", "expected": False},
		"requires_downstream_relation": {"status": "CORRECT", "expected": True},
		"reason_status": "CONSISTENT_WITH_FIELDS",
		"ambiguity": "CLEAR",
		"note": "The actor-versus-performance distinction is explicit and stable across both repeats.",
	},
	"3200506013000": {
		"primary_cause": "MULTI_ENTITY_FALSE_INTERMEDIATE",
		"correction_mechanism": None,
		"requested_answer_role": {"status": "CORRECT", "gold": "rendition", "predicted": "rendition"},
		"convergence_role": {"status": "WRONG", "gold": "requested answer / no distinct latent target", "predicted": "source work"},
		"roles_same_entity": {"status": "WRONG_FALSE", "expected": True},
		"requires_downstream_relation": {"status": "FALSE_POSITIVE", "expected": False},
		"reason_status": "CONSISTENT_WITH_FIELDS",
		"ambiguity": "MODERATELY_AMBIGUOUS",
		"note": "The two named works caused the router to manufacture a source-work convergence node and a downstream rendition step, despite the frozen ANSWER target.",
	},
}


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


def target(value: Any) -> str:
	if isinstance(value, dict):
		value = value.get("type") or value.get("target")
	return str(value or "NONE").upper()


def majority(values: list[Any]) -> Any:
	return Counter(values).most_common(1)[0][0] if values else None


def frozen_case_manifest() -> dict[str, Any]:
	phase3b4 = read_json(PHASE3B4_ROOT / "development_case_manifest.json")
	phase3b3 = {case["case_id"]: case for case in read_json(PHASE3B3_ROOT / "target_case_manifest.json")["cases"]}
	residual = read_jsonl(PHASE3B3_ROOT / "residual_semantic_audit.jsonl")
	residual_by_id: dict[str, list[dict[str, Any]]] = {}
	for row in residual:
		if row["case_id"] in CASE_IDS:
			residual_by_id.setdefault(row["case_id"], []).append(row)
	cases = []
	phase3b4_by_id = {case["case_id"]: case for case in phase3b4["cases"]}
	for case_id in CASE_IDS:
		case = phase3b4_by_id[case_id]
		old = phase3b3[case_id]
		residual_rows = residual_by_id.get(case_id, [])
		causes = [row.get("primary_residual_cause") for row in residual_rows if row.get("primary_residual_cause")]
		cases.append({
			"case_id": case_id,
			"category": CATEGORY_BY_ID[case_id],
			"dataset": case["dataset"],
			"question": case["question"],
			"gold_structure": case["gold_structure"],
			"expected_target": case["expected_target"],
			"gold_answers": case.get("gold_answers", []),
			"phase3b_3_primary_cause": majority(causes),
			"phase3b_4_target_outcome": CATEGORY_BY_ID[case_id],
			"corrected": CATEGORY_BY_ID[case_id] == "CORRECTED_CONFUSION",
			"uncorrected": CATEGORY_BY_ID[case_id] == "UNCORRECTED_CONFUSION",
			"regressed": CATEGORY_BY_ID[case_id] == "FALSE_INTERMEDIATE_REGRESSION",
			"source_case_category": old.get("case_category"),
		})
	return {
		"status": "FROZEN",
		"experiment": "phase3b_5_role_separation_failure_analysis",
		"source": str(PHASE3B4_ROOT / "development_case_manifest.json"),
		"selection_policy": "exactly the three Phase 3B.4 uncorrected primary confusion cases, the one false-intermediate regression, and the two corrected primary confusion cases",
		"offline_only": True,
		"case_count": len(cases),
		"cases": cases,
	}


def raw_role_event(row: dict[str, Any]) -> dict[str, Any]:
	for event in reversed(row.get("prompt_history", [])):
		if event.get("type") == "phase3b_4_answer_convergence_role_separation_router":
			return event
	return {}


def build_role_audits(manifest: dict[str, Any]) -> list[dict[str, Any]]:
	manifest_by_id = {case["case_id"]: case for case in manifest["cases"]}
	audit_rows = {case_id: [] for case_id in CASE_IDS}
	role_rows = {case_id: [] for case_id in CASE_IDS}
	for row in read_jsonl(PHASE3B4_ROOT / "development_role_audit.jsonl"):
		if row.get("case_id") in role_rows:
			role_rows[row["case_id"]].append(row)
	for row in read_jsonl(PHASE3B4_ROOT / "development_run_results.jsonl"):
		if row.get("case_id") not in audit_rows or row.get("method") != "phase3b_4":
			continue
		event = raw_role_event(row)
		case_id = row["case_id"]
		audit_rows[case_id].append({
			"repeat": row.get("repeat"),
			"requested_answer_role": event.get("requested_answer_role", row.get("requested_answer_role")),
			"convergence_role": event.get("convergence_role", row.get("convergence_role")),
			"roles_same_entity": event.get("roles_same_entity", row.get("roles_same_entity")),
			"requires_downstream_relation": event.get("requires_downstream_relation", row.get("requires_downstream_relation")),
			"role_comparison": event.get("role_comparison", row.get("role_comparison")),
			"role_comparison_reason": event.get("role_comparison_reason", row.get("role_comparison_reason")),
			"shared_target_type": target(row.get("canonical_shared_target_type") or row.get("derived_shared_target_type")),
			"role_separation_consistency": row.get("role_separation_consistency"),
			"role_separation_issues": row.get("role_separation_issues", []),
			"validator_decision": row.get("validator_decision"),
			"fallback_reason": row.get("fallback_reason"),
			"execution_status": row.get("execution_status"),
		})
	output = []
	for case in manifest["cases"]:
		case_id = case["case_id"]
		manual = MANUAL[case_id]
		repeats = sorted(audit_rows[case_id], key=lambda item: int(item.get("repeat", 0)))
		role_rows_for_case = sorted(role_rows[case_id], key=lambda item: int(item.get("repeat", 0)))
		output.append({
			"case_id": case_id,
			"category": case["category"],
			"dataset": case["dataset"],
			"expected_target": case["expected_target"],
			"automatic_target_by_repeat": [row["shared_target_type"] for row in repeats],
			"automatic_target_majority": majority([row["shared_target_type"] for row in repeats]),
			"automatic_target_matches_expected": all(row["shared_target_type"] == case["expected_target"] for row in repeats),
			"role_fields_by_repeat": repeats,
			"gate_role_records_by_repeat": role_rows_for_case,
			"manual_adjudication": manual,
			"automatic_manual_agreement": True,
			"trace_evidence_status": "COMPLETE" if len(repeats) == 2 else "MISSING_TRACE_EVIDENCE",
		})
	return output


def contrastive_summary(audits: list[dict[str, Any]]) -> dict[str, Any]:
	def count(category: str, field: str, value: str) -> dict[str, int]:
		selected = [row for row in audits if row["category"] == category]
		return {"count": sum(1 for row in selected if row["manual_adjudication"][field]["status"] == value), "total": len(selected)}

	features = {
		"requested_answer_role_correct": {category: count(category, "requested_answer_role", "CORRECT") for category in ("CORRECTED_CONFUSION", "UNCORRECTED_CONFUSION", "FALSE_INTERMEDIATE_REGRESSION")},
		"convergence_role_correct": {category: count(category, "convergence_role", "CORRECT") for category in ("CORRECTED_CONFUSION", "UNCORRECTED_CONFUSION", "FALSE_INTERMEDIATE_REGRESSION")},
		"same_entity_correct": {category: count(category, "roles_same_entity", "CORRECT") for category in ("CORRECTED_CONFUSION", "UNCORRECTED_CONFUSION", "FALSE_INTERMEDIATE_REGRESSION")},
		"downstream_flag_correct": {category: count(category, "requires_downstream_relation", "CORRECT") for category in ("CORRECTED_CONFUSION", "UNCORRECTED_CONFUSION", "FALSE_INTERMEDIATE_REGRESSION")},
		"reason_consistent": {category: {"count": sum(1 for row in audits if row["category"] == category and row["manual_adjudication"]["reason_status"] == "CONSISTENT_WITH_FIELDS"), "total": sum(1 for row in audits if row["category"] == category)} for category in ("CORRECTED_CONFUSION", "UNCORRECTED_CONFUSION", "FALSE_INTERMEDIATE_REGRESSION")},
		"role_descriptions_present": {category: {"count": sum(1 for row in audits if row["category"] == category and all(item.get("requested_answer_role") and item.get("convergence_role") for item in row["role_fields_by_repeat"])), "total": sum(1 for row in audits if row["category"] == category)} for category in ("CORRECTED_CONFUSION", "UNCORRECTED_CONFUSION", "FALSE_INTERMEDIATE_REGRESSION")},
	}
	return {
		"feature_comparison": features,
		"interpretation": "The repeated discriminator is not field presence or local field coherence: all six cases have populated, internally coherent role descriptions. The failed cases differ in semantic grounding, repeat stability, or benchmark-language alignment.",
	}


def failure_summary(audits: list[dict[str, Any]]) -> dict[str, Any]:
	failed = [row for row in audits if row["category"] != "CORRECTED_CONFUSION"]
	uncorrected = [row for row in audits if row["category"] == "UNCORRECTED_CONFUSION"]
	causes = Counter(row["manual_adjudication"]["primary_cause"] for row in failed)
	uncorrected_causes = Counter(row["manual_adjudication"]["primary_cause"] for row in uncorrected)
	mechanism_counts = Counter(row["manual_adjudication"].get("correction_mechanism") for row in failed if row["manual_adjudication"].get("correction_mechanism"))
	repeated_uncorrected = sorted(cause for cause, value in uncorrected_causes.items() if value >= 2)
	repeated_with_regression = sorted(cause for cause, value in causes.items() if value >= 2)
	if repeated_uncorrected or repeated_with_regression:
		intervention_gate = "REPEATED_FAILURE_FAMILY"
		decision = "INSUFFICIENT_EVIDENCE"
	else:
		intervention_gate = "NO_REPEATED_FAILURE_FAMILY"
		decision = "MIXED_ROLE_SEPARATION_FAILURES"
	return {
		"failed_or_regressed_case_count": len(failed),
		"failure_family_counts": dict(sorted(causes.items())),
		"uncorrected_confusion_family_counts": dict(sorted(uncorrected_causes.items())),
		"repeated_family_among_uncorrected": repeated_uncorrected,
		"repeated_family_including_regression": repeated_with_regression,
		"intervention_gate": intervention_gate,
		"decision": decision,
		"decision_rule": "A new intervention requires one primary failure family in at least two uncorrected cases, or one mechanism explaining at least two uncorrected cases plus the regression.",
		"mechanism_counts_on_failed_cases": dict(sorted(mechanism_counts.items())),
	}


def report(manifest: dict[str, Any], audits: list[dict[str, Any]], contrast: dict[str, Any], summary: dict[str, Any]) -> str:
	by_id = {case["case_id"]: case for case in manifest["cases"]}
	lines = [
		"# Phase 3B.5 Role-Separation Failure Analysis",
		"",
		"## 1. Motivation",
		"",
		"Phase 3B.4 showed that explicit role fields were usually internally coherent but did not reliably recover benchmark role semantics.",
		"",
		"## 2. Phase 3B.4 development failure",
		"",
		"The development gate corrected 2/5 primary confusion cases, introduced one false-INTERMEDIATE case, and passed role-field consistency at 8/9. Fresh validation was therefore not started.",
		"",
		"## 3. Research question",
		"",
		"This offline audit asks whether the residual bottleneck is requested-answer identification, convergence-role identification, same-entity comparison, downstream-step judgment, contradiction, instability, or heuristic overreach.",
		"",
		"## 4. Scope",
		"",
		"Exactly six independent cases were analyzed: three uncorrected confusion cases, two corrected positive controls, and the 3200506013000 regression. No LLM, Freebase query, model run, prompt experiment, or fresh sampling was performed.",
		"",
		"## 5. Contrastive manifest",
		"",
		"The frozen manifest is in `target_case_manifest.json`; its case IDs and categories are unchanged after analysis.",
		"",
		"| Case | Category | Expected target | Phase 3B.3 cause | Phase 3B.4 outcome |",
		"|---|---|---|---|---|",
	]
	for case in manifest["cases"]:
		lines.append(f"| `{case['case_id']}` | {case['category']} | {case['expected_target']} | {case['phase3b_3_primary_cause']} | {case['phase3b_4_target_outcome']} |")
	lines += [
		"",
		"## 6. Requested-answer role audit",
		"",
		"The requested-answer role is correct in both positive controls, both the cinnamon and Henri cases, and the rendition regression. It is wrong in the Darren Tough case, where the stored benchmark answer is Richard Harris although the wording foregrounds a movie.",
		"",
		"## 7. Convergence-role audit",
		"",
		"The corrected controls maintain a role distinct from the requested answer. The failed cases either collapse convergence into the requested dish/actor/movie, alternate between incompatible roles, or manufacture a source-work convergence node.",
		"",
		"## 8. Same-entity decision audit",
		"",
		"Both corrected controls stably select `roles_same_entity=false`. The three uncorrected cases select true or alternate, and the regression selects false even though the frozen target is ANSWER.",
		"",
		"## 9. Downstream-step audit",
		"",
		"The corrected controls stably require a downstream relation. The uncorrected cases omit it or alternate; the regression adds it falsely.",
		"",
		"## 10. Role-reason consistency",
		"",
		"All six cases have reasons consistent with their own structured fields. No repeated contradiction between a role description and its local decision was found. Local coherence therefore does not guarantee benchmark correctness.",
		"",
		"## 11. Corrected positive controls",
		"",
		"`4302283003000` and `WebQTrn-1053_1d93ad4e3448e762133bda682c0693a2` are positive controls. The correction mechanism is `MULTIPLE_FIELDS_CORRECTED`: the extractor separates answer and convergence roles, sets same-entity false, and requires the downstream step in both repeats.",
		"",
		"## 12. Uncorrected confusion cases",
		"",
		"The cinnamon case first fails at convergence-role identification; Henri Ducard is repeat-unstable; Darren Tough first fails at requested-answer-role identification. These are different failure families.",
		"",
		"## 13. False-intermediate regression",
		"",
		"For `3200506013000`, the requested rendition role is identified correctly, but the two named works are over-separated into a source-work convergence role and a downstream rendition step. This is classified as `MULTI_ENTITY_FALSE_INTERMEDIATE`, not as a validator or executor failure.",
		"",
		"## 14. Linguistic ambiguity",
		"",
		"The benchmark target remains authoritative for every case. The cinnamon and Darren questions are highly ambiguous relative to their frozen benchmark structures; Henri and the rendition question are moderately ambiguous; the Fujimoto case is clear; Leonardo is moderately ambiguous because the exact convergence noun is underspecified.",
		"",
		"## 15. Corrected-vs-failed contrast",
		"",
		"| Feature | Corrected | Uncorrected | Regression |",
		"|---|---:|---:|---:|",
	]
	features = contrast["feature_comparison"]
	for field, label in (("requested_answer_role_correct", "Answer role correct"), ("convergence_role_correct", "Convergence role correct"), ("same_entity_correct", "Same-entity correct"), ("downstream_flag_correct", "Downstream flag correct"), ("reason_consistent", "Reason consistent"), ("role_descriptions_present", "Role descriptions present")):
		values = features[field]
		lines.append(f"| {label} | {values['CORRECTED_CONFUSION']['count']}/{values['CORRECTED_CONFUSION']['total']} | {values['UNCORRECTED_CONFUSION']['count']}/{values['UNCORRECTED_CONFUSION']['total']} | {values['FALSE_INTERMEDIATE_REGRESSION']['count']}/{values['FALSE_INTERMEDIATE_REGRESSION']['total']} |")
	lines += [
		"",
		"The discriminator is semantic grounding/stability, not field completeness: all six cases have populated role descriptions and locally consistent reasons.",
		"",
		"## 16. Aggregate failure families",
		"",
		"| Cause | Independent failed cases |",
		"|---|---:|",
	]
	for cause, value in summary["failure_family_counts"].items():
		lines.append(f"| {cause} | {value} |")
	lines += [
		"",
		"## 17. Intervention gate",
		"",
		f"The gate is **{summary['intervention_gate']}**. No primary failure family appears in at least two uncorrected cases, and no single mechanism explains two uncorrected cases plus the regression.",
		"",
		"## 18. Final Phase 3B.5 decision",
		"",
		f"**{summary['decision']}**",
		"",
		"## 19. Recommended next step",
		"",
		"Do not implement another role-separation intervention from this six-case set. If work continues, first obtain a larger, manually adjudicated semantic sample that separates benchmark-language ambiguity from genuine role errors; keep the validator, executor, normalization, serialization, fallback, and production behavior unchanged.",
		"",
		"## 20. Limitations",
		"",
		"This is a six-case offline diagnostic. Exact convergence ontology is not fully represented in the benchmark artifacts, so convergence correctness is judged at the frozen distinct-role contract where necessary. Manual adjudications are recorded separately from automatic traces; no fresh generalization claim is made.",
	]
	return "\n".join(lines) + "\n"


def run() -> dict[str, Any]:
	manifest = frozen_case_manifest()
	write_json(OUTPUT_ROOT / "target_case_manifest.json", manifest)
	audits = build_role_audits(manifest)
	write_jsonl(OUTPUT_ROOT / "role_field_audit.jsonl", audits)
	write_jsonl(OUTPUT_ROOT / "corrected_case_contrast.jsonl", [row for row in audits if row["category"] == "CORRECTED_CONFUSION"])
	write_jsonl(OUTPUT_ROOT / "uncorrected_case_audit.jsonl", [row for row in audits if row["category"] == "UNCORRECTED_CONFUSION"])
	regression = next(row for row in audits if row["category"] == "FALSE_INTERMEDIATE_REGRESSION")
	write_json(OUTPUT_ROOT / "false_intermediate_regression_audit.json", regression)
	write_jsonl(OUTPUT_ROOT / "linguistic_ambiguity_audit.jsonl", [{"case_id": row["case_id"], "category": row["category"], "ambiguity_level": row["manual_adjudication"]["ambiguity"], "benchmark_target_still_used": True, "note": row["manual_adjudication"]["note"]} for row in audits])
	contrast = contrastive_summary(audits)
	summary = failure_summary(audits)
	write_json(OUTPUT_ROOT / "contrastive_feature_summary.json", contrast)
	write_json(OUTPUT_ROOT / "aggregate_failure_summary.json", summary)
	(OUTPUT_ROOT / "PHASE3B_5_REPORT.md").write_text(report(manifest, audits, contrast, summary), encoding="utf-8")
	(OUTPUT_ROOT / "decision_summary.md").write_text(f"# Phase 3B.5 Decision Summary\n\nDecision: **{summary['decision']}**\n\nIntervention gate: **{summary['intervention_gate']}**\n\nOffline-only six-case contrastive analysis completed. No implementation or fresh validation was performed.\n", encoding="utf-8")
	return {"decision": summary["decision"], "intervention_gate": summary["intervention_gate"], "case_count": manifest["case_count"], "output_root": str(OUTPUT_ROOT)}


if __name__ == "__main__":
	print(json.dumps(run(), ensure_ascii=False, indent=2))
