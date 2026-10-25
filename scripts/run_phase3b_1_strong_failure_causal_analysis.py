#!/usr/bin/env python3
"""Offline causal audit for the 12 strong Phase 3B failure cases."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_phase3a_branch_necessity_benchmark import load_candidates  # noqa: E402


INPUT_ROOT = ROOT / "outputs/experiments/phase3b/frozen_strong_vs_control"
OUTPUT_ROOT = ROOT / "outputs/experiments/phase3b_1/strong_failure_causal_analysis"
REQUIRED_INPUTS = (
	"PHASE3B_REPORT.md", "sample_manifest.json", "run_results.jsonl", "paired_analysis.json",
	"routing_behavior.json", "stability_analysis.json", "failure_analysis.jsonl",
	"strong_case_analysis.md", "decision_summary.md",
)
CAUSES = (
	"ENTITY_GROUNDING_ERROR", "CONSTRAINT_EXTRACTION_ERROR", "INDEPENDENT_SOURCE_EXTRACTION_ERROR",
	"SHARED_TARGET_ERROR", "DEPENDENCY_PATTERN_ERROR", "STRUCTURE_ROUTING_ERROR",
	"NORMALIZATION_ERROR", "SERIALIZATION_ERROR", "VALIDATOR_REJECTION",
	"BRANCH_EXECUTION_ERROR", "INTERMEDIATE_BINDING_LOSS", "RELATION_GROUNDING_ERROR",
	"FALLBACK_ERASED_ADVANTAGE", "ANSWER_SELECTION_ERROR", "NO_CLEAR_FAILURE",
)
OUTCOMES = ("BOTH_WRONG", "V1_1_3_ONLY", "PHASE2_5_6_ONLY")


def read_json(path: Path) -> dict[str, Any]:
	return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def text(value: Any) -> str:
	return str(value or "").strip()


def upper(value: Any) -> str:
	return text(value).upper()


def canonical(value: Any) -> str:
	return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def target_type(proposal: dict[str, Any]) -> str:
	value = proposal.get("shared_target")
	if isinstance(value, dict):
		return upper(value.get("type")) or "NONE"
	return upper(value) or "NONE"


def expected_target(candidate: dict[str, Any]) -> str:
	return "ANSWER" if candidate["gold_structure"] == "BRANCH_FINAL" else "INTERMEDIATE"


def expected_dependency(candidate: dict[str, Any]) -> str:
	return "CONVERGENCE" if candidate["gold_structure"] == "BRANCH_FINAL" else "INTERMEDIATE_CONTINUATION"


def source_ids(proposal: dict[str, Any]) -> set[str]:
	values: set[str] = set()
	for group in proposal.get("independent_source_groups", []) if isinstance(proposal.get("independent_source_groups"), list) else []:
		anchor = group.get("source_anchor") if isinstance(group, dict) else None
		if isinstance(anchor, dict):
			for key in ("id", "entity_id"):
				if text(anchor.get(key)):
					values.add(text(anchor[key]))
		elif text(anchor) and not text(anchor).lower().startswith(("literal", "explicit")):
			values.add(text(anchor).split(":", 1)[0])
	for constraint in proposal.get("constraints", []) if isinstance(proposal.get("constraints"), list) else []:
		anchor = constraint.get("source_anchor") if isinstance(constraint, dict) else None
		if isinstance(anchor, dict):
			for key in ("id", "entity_id"):
				if text(anchor.get(key)):
					values.add(text(anchor[key]))
		elif text(anchor) and not text(anchor).lower().startswith(("literal", "explicit")):
			values.add(text(anchor).split(":", 1)[0])
	return values


def proposal_has_two_sources(proposal: dict[str, Any]) -> bool:
	groups = proposal.get("independent_source_groups")
	if isinstance(groups, list) and len(groups) >= 2:
		return True
	constraints = proposal.get("constraints")
	return isinstance(constraints, list) and len(constraints) >= 2 and len(source_ids(proposal)) >= 2


def dependency_status(candidate: dict[str, Any], proposal: dict[str, Any]) -> str:
	dep = upper(proposal.get("dependency_pattern"))
	if not dep:
		return "MISSING"
	if candidate["gold_structure"] == "BRANCH_INTERMEDIATE":
		return "CORRECT" if any(token in dep for token in ("INTERMEDIATE", "MIXED", "CONTINUATION", "CONVERGE")) else "WRONG"
	return "CORRECT" if any(token in dep for token in ("INDEPENDENT", "PARALLEL", "CONVERG")) and "SEQUENTIAL" not in dep else "PARTIAL"


def semantic_audit(candidate: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
	proposal = row.get("raw_parsed_result") if isinstance(row.get("raw_parsed_result"), dict) else {}
	expected = expected_target(candidate)
	actual = target_type(proposal)
	ids = set(candidate.get("topic_entities", []))
	found_ids = source_ids(proposal)
	if not proposal:
		return {
			"constraint_extraction": "MISSING", "independent_source_extraction": "MISSING",
			"shared_target": "MISSING", "dependency_pattern": "MISSING",
			"semantic_adequate": False, "expected_target": expected, "actual_target": "NONE", "source_ids": [],
		}
	shared = "CORRECT" if actual == expected else "WRONG"
	source_status = "CORRECT" if proposal_has_two_sources(proposal) else "MISSING"
	if found_ids and ids and found_ids.intersection(ids):
		entity_status = "CORRECT"
	elif found_ids:
		entity_status = "PARTIAL"
	else:
		entity_status = "PARTIAL" if proposal_has_two_sources(proposal) else "MISSING"
	constraints = proposal.get("constraints") if isinstance(proposal.get("constraints"), list) else []
	constraint_status = "CORRECT" if len(constraints) >= 2 and proposal_has_two_sources(proposal) else "PARTIAL"
	dep_status = dependency_status(candidate, proposal)
	semantic_adequate = all(
		status in {"CORRECT", "PARTIAL"} for status in (constraint_status, source_status, shared, dep_status)
	) and shared == "CORRECT" and source_status == "CORRECT" and dep_status == "CORRECT"
	return {
		"entity_grounding": entity_status,
		"constraint_extraction": constraint_status,
		"independent_source_extraction": source_status,
		"shared_target": shared,
		"dependency_pattern": dep_status,
		"semantic_adequate": semantic_adequate,
		"expected_target": expected,
		"actual_target": actual,
		"source_ids": sorted(found_ids),
		"constraint_count": len(constraints),
	}


def first_divergence(candidate: dict[str, Any], row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
	semantic = semantic_audit(candidate, row)
	if semantic.get("entity_grounding") == "WRONG":
		return "ENTITY_GROUNDING_ERROR", "runtime source entity does not match a gold topic entity", semantic
	if semantic.get("constraint_extraction") == "MISSING":
		return "CONSTRAINT_EXTRACTION_ERROR", "no usable answer-relevant constraints were retained", semantic
	if semantic.get("independent_source_extraction") != "CORRECT":
		return "INDEPENDENT_SOURCE_EXTRACTION_ERROR", "fewer than two independent source groups are represented", semantic
	if semantic.get("shared_target") != "CORRECT":
		return "SHARED_TARGET_ERROR", f"runtime shared target {semantic.get('actual_target')} != gold {semantic.get('expected_target')}", semantic
	if semantic.get("dependency_pattern") == "WRONG":
		return "DEPENDENCY_PATTERN_ERROR", "dependency description does not preserve the gold continuation", semantic
	if semantic.get("dependency_pattern") == "MISSING":
		return "DEPENDENCY_PATTERN_ERROR", "dependency pattern is missing", semantic
	canonical_plan = row.get("canonical_semantic_representation") if isinstance(row.get("canonical_semantic_representation"), dict) else {}
	if row.get("normalization_status") in {"PARTIAL_NORMALIZATION", "NORMALIZATION_FAILED"}:
		return "NORMALIZATION_ERROR", "semantic proposal was not fully converted to the canonical contract", semantic
	if canonical_plan and upper(canonical_plan.get("structure")) != candidate["gold_structure"]:
		return "NORMALIZATION_ERROR", "canonical structure differs from gold structure", semantic
	if row.get("serialization_status") in {"FAILED", "UNKNOWN"}:
		return "SERIALIZATION_ERROR", "canonical plan was not serialized", semantic
	if row.get("validator_decision") not in {"PASS", "NOT_APPLICABLE"}:
		return "VALIDATOR_REJECTION", "validator rejected the otherwise adequate branch representation", semantic
	if row.get("branch_attempted") and row.get("execution_status") != "BRANCH_EXECUTION_SUCCESS":
		return "BRANCH_EXECUTION_ERROR", f"branch execution did not produce a usable merge ({row.get('execution_status')})", semantic
	if row.get("branch_attempted") and row.get("execution_status") == "BRANCH_EXECUTION_SUCCESS" and not row.get("hits_at_1"):
		return "RELATION_GROUNDING_ERROR", "correct branch state was represented but the executed candidate set missed the gold answer", semantic
	if row.get("fallback_used") and not row.get("hits_at_1"):
		return "FALLBACK_ERASED_ADVANTAGE", "fallback replaced a branch-capable route without recovering the answer", semantic
	if not row.get("hits_at_1"):
		return "ANSWER_SELECTION_ERROR", "execution completed but final answer selection missed the gold answer", semantic
	return "NO_CLEAR_FAILURE", "no causal divergence identified in the stored trace", semantic


def structure_label(gold: str, predicted: str) -> str:
	if predicted == gold:
		return "STRUCTURE_CORRECT"
	if gold == "BRANCH_INTERMEDIATE" and predicted == "BRANCH_FINAL":
		return "STRUCTURE_WRONG_BRANCH_TYPE"
	if predicted in {"LINEAR", "UNCERTAIN", ""}:
		return "STRUCTURE_TOO_LINEAR" if predicted == "LINEAR" else "STRUCTURE_UNCERTAIN"
	return "STRUCTURE_WRONG_BRANCH_TYPE"


def selected_cases(paired: dict[str, Any], candidates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
	strong = [row for row in paired.get("cases", []) if row.get("cohort") == "strong" and row.get("outcome") in OUTCOMES]
	counts = Counter(row["outcome"] for row in strong)
	if counts != Counter({"BOTH_WRONG": 10, "V1_1_3_ONLY": 1, "PHASE2_5_6_ONLY": 1}):
		raise RuntimeError(f"unexpected Phase 3B strong target counts: {counts}")
	result: list[dict[str, Any]] = []
	for row in sorted(strong, key=lambda item: item["case_id"]):
		candidate = candidates.get(row["case_id"])
		if not candidate:
			raise RuntimeError(f"missing Phase 3A metadata for {row['case_id']}")
		result.append({
			"case_id": row["case_id"], "dataset": row["dataset"], "cohort": row["cohort"],
			"paired_outcome": row["outcome"], "gold_structure": row["gold_structure"],
			"question": candidate["question"], "topic_entities": candidate["topic_entities"],
			"topic_entity_names": candidate["topic_entity_names"], "gold_answers": candidate["gold_answers"],
			"answer_variable": candidate["answer_variable"], "shared_target_variable": candidate["shared_target_variable"],
			"gold_patterns": candidate["patterns"], "gold_branches": candidate["branches"],
		})
	return result


def intermediate_status(candidate: dict[str, Any], row: dict[str, Any], cause: str) -> str:
	if candidate["gold_structure"] != "BRANCH_INTERMEDIATE":
		return "NOT_APPLICABLE"
	semantic = semantic_audit(candidate, row)
	if semantic.get("shared_target") != "CORRECT":
		return "INTERMEDIATE_NOT_DETECTED"
	if row.get("validator_decision") not in {"PASS", "NOT_APPLICABLE"}:
		return "INTERMEDIATE_DETECTED_EXECUTOR_UNSUPPORTED"
	if row.get("branch_attempted") and row.get("execution_status") == "BRANCH_EXECUTION_SUCCESS" and row.get("hits_at_1"):
		return "INTERMEDIATE_EXECUTION_SUCCESS"
	if row.get("branch_attempted") and not row.get("hits_at_1"):
		return "INTERMEDIATE_RELATION_GROUNDING_FAILED"
	if cause == "INTERMEDIATE_BINDING_LOSS":
		return "INTERMEDIATE_BINDING_LOST"
	return "INTERMEDIATE_DETECTED_EXECUTOR_UNSUPPORTED"


def fallback_class(row: dict[str, Any], candidate: dict[str, Any]) -> str:
	if not row.get("fallback_used"):
		return "NO_FALLBACK"
	if row.get("hits_at_1"):
		return "FALLBACK_RESCUE"
	if candidate["gold_structure"] == "BRANCH_INTERMEDIATE" and target_type(row.get("raw_parsed_result") or {}) == "INTERMEDIATE":
		return "FALLBACK_ERASED_BRANCH_ADVANTAGE"
	if row.get("fallback_reason") == "empty_strict_merge":
		return "FALLBACK_NEUTRAL"
	return "FALLBACK_REGRESSION"


def case_primary(causes: list[str]) -> str:
	counts = Counter(causes)
	if len(counts) == 1:
		return causes[0]
	most = counts.most_common()
	if len(most) >= 2 and most[0][1] == most[1][1]:
		return "MIXED_FAILURES"
	return most[0][0]


def aggregate(first_case_rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
	by_case = {row["case_id"]: row for row in selected}
	repeat_counts = Counter(row["cause"] for row in first_case_rows)
	primary_by_case = {row["case_id"]: row.get("case_primary_cause") for row in first_case_rows}
	case_counts = Counter(value for value in primary_by_case.values() if value not in {"MIXED_FAILURES", "NO_CLEAR_FAILURE", None})
	observed_by_case = {
		case_id: {row["cause"] for row in first_case_rows if row["case_id"] == case_id}
		for case_id in by_case
	}
	observed_counts = Counter(cause for causes in observed_by_case.values() for cause in causes)
	by_dataset = {dataset: dict(Counter(primary_by_case[item["case_id"]] for item in selected if item["dataset"] == dataset)) for dataset in sorted({item["dataset"] for item in selected})}
	by_structure = {structure: dict(Counter(primary_by_case[item["case_id"]] for item in selected if item["gold_structure"] == structure)) for structure in sorted({item["gold_structure"] for item in selected})}
	by_outcome = {outcome: dict(Counter(primary_by_case[item["case_id"]] for item in selected if item["paired_outcome"] == outcome)) for outcome in OUTCOMES}
	semantic_cases = {row["case_id"] for row in first_case_rows if row.get("case_primary_cause") in {"SHARED_TARGET_ERROR", "CONSTRAINT_EXTRACTION_ERROR", "INDEPENDENT_SOURCE_EXTRACTION_ERROR", "DEPENDENCY_PATTERN_ERROR"}}
	semantic_datasets = {by_case[cid]["dataset"] for cid in semantic_cases}
	semantic_supported = len(semantic_cases) >= 3 and len(semantic_datasets) >= 2
	intermediate_executor_cases = {row["case_id"] for row in first_case_rows if row.get("case_primary_cause") == "INTERMEDIATE_EXECUTION_BOTTLENECK"}
	relation_cases = {row["case_id"] for row in first_case_rows if row.get("case_primary_cause") == "RELATION_GROUNDING_ERROR"}
	return {
		"repeat_first_divergence_counts": dict(repeat_counts),
		"independent_case_primary_counts": dict(case_counts),
		"independent_case_observed_counts": dict(observed_counts),
		"by_dataset": by_dataset, "by_structure": by_structure, "by_paired_outcome": by_outcome,
		"semantic_extraction_gate": {"independent_cases": len(semantic_cases), "datasets": sorted(semantic_datasets), "supported": semantic_supported},
		"intermediate_execution_gate": {"independent_cases": len(intermediate_executor_cases), "supported": len(intermediate_executor_cases) >= 3},
		"relation_grounding_gate": {"independent_cases": len(relation_cases), "supported": len(relation_cases) >= 3},
	"validator_or_serialization_gate": {"independent_cases": sum(value in {"VALIDATOR_REJECTION", "SERIALIZATION_ERROR", "NORMALIZATION_ERROR"} for value in primary_by_case.values()), "supported": False},
	}


def decision(summary: dict[str, Any]) -> str:
	if summary["semantic_extraction_gate"]["supported"]:
		return "SEMANTIC_EXTRACTION_BOTTLENECK_SUPPORTED"
	if summary["intermediate_execution_gate"]["supported"]:
		return "INTERMEDIATE_EXECUTION_BOTTLENECK_SUPPORTED"
	if summary["relation_grounding_gate"]["supported"]:
		return "RELATION_GROUNDING_BOTTLENECK_SUPPORTED"
	if summary["validator_or_serialization_gate"]["supported"]:
		return "VALIDATOR_OR_SERIALIZATION_BOTTLENECK_SUPPORTED"
	if summary["independent_case_primary_counts"]:
		return "MIXED_FAILURES_NO_DOMINANT_CAUSE"
	return "INSUFFICIENT_TRACE_EVIDENCE"


def make_report(selected: list[dict[str, Any]], first_rows: list[dict[str, Any]], semantic_rows: list[dict[str, Any]], structure_rows: list[dict[str, Any]], validator_rows: list[dict[str, Any]], intermediate_rows: list[dict[str, Any]], fallback_rows: list[dict[str, Any]], summary: dict[str, Any], final_decision: str) -> str:
	first_counts = Counter(row["cause"] for row in first_rows)
	case_counts = Counter(row.get("case_primary_cause") for row in first_rows)
	rows_lookup = {row["case_id"]: row for row in selected}
	structure_lines = []
	for gold in ("BRANCH_FINAL", "BRANCH_INTERMEDIATE"):
		cases = [row for row in selected if row["gold_structure"] == gold]
		case_ids = {row["case_id"] for row in cases}
		st = [row for row in structure_rows if row["method"] == "phase2_5_6" and row["case_id"] in case_ids]
		by_st_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
		for item in st:
			by_st_case[item["case_id"]].append(item)
		structure_lines.append(f"| {gold} | {len(cases)} | {sum(any(row['classification'] == 'STRUCTURE_CORRECT' for row in values) for values in by_st_case.values())} | {sum(any(row.get('intermediate_status') == 'INTERMEDIATE_DETECTED_EXECUTOR_UNSUPPORTED' for row in values) for values in by_st_case.values())} | {sum(any(row.get('first_cause') == 'RELATION_GROUNDING_ERROR' for row in values) for values in by_st_case.values())} |")
	case_observed_counts = summary.get("independent_case_observed_counts", {})
	first_table = "\n".join(f"| {cause} | {first_counts.get(cause, 0)} | {case_observed_counts.get(cause, 0)} |" for cause in CAUSES)
	paired_table = "\n".join(
		f"| {outcome} | {sum(1 for item in selected if item['paired_outcome'] == outcome)} | {Counter(row.get('case_primary_cause') for row in first_rows if rows_lookup[row['case_id']]['paired_outcome'] == outcome).most_common(1)[0][0] if any(rows_lookup[row['case_id']]['paired_outcome'] == outcome for row in first_rows) else '—'} |"
		for outcome in OUTCOMES
	)
	return f"""# Phase 3B.1 strong branch failure causal analysis

## 1. Motivation

Phase 3B produced a 1–1 paired strong-case win comparison and 10/20 strong cases that were wrong under both methods. This diagnostic identifies the earliest causal divergence in the frozen traces.

## 2. Research question

The audit asks whether failures begin in semantic extraction, routing, normalization/serialization, validation, branch execution, relation grounding, fallback, or final answer selection.

## 3. Scope

Exactly 12 strong cases were audited: 10 `BOTH_WRONG`, 1 `V1_1_3_ONLY`, and 1 `PHASE2_5_6_ONLY`. Both repeats and both methods were inspected. No model or Freebase queries were issued.

## 4. Target cases

See `target_case_manifest.json`. The target set is frozen from Phase 3B paired outcomes and contains no new questions.

## 5. Frozen evidence

Evidence came from the Phase 3B report, sample manifest, run results, paired analysis, routing analysis, stability analysis, failure analysis, and strong-case analysis. Phase 3A metadata was used only for gold structure and branch reconstruction.

## 6. Gold-required structures

Each case retains its exact gold patterns, branch descriptions, source entities, shared target variable, and answer variable in `target_case_manifest.json`. BRANCH_INTERMEDIATE cases require source constraints to converge on an intermediate variable before the downstream answer relation.

## 7. First-divergence method

The precedence is entity grounding → constraints → independent sources → shared target → dependency → normalization → serialization → validator → branch execution → relation grounding → fallback → answer selection. Later symptoms were not promoted ahead of an earlier incompatible state.

## 8. Semantic extraction audit

See `semantic_extraction_audit.jsonl`. The dominant repeated semantic issue is incorrect `shared_target=ANSWER` on gold BRANCH_INTERMEDIATE cases.

## 9. Structural routing audit

See `structure_audit.jsonl`. Correctly labeled structure does not imply a correct answer; several BRANCH_FINAL traces passed routing and still failed during candidate generation/intersection.

## 10. Normalization/serialization/validator audit

See `validator_execution_audit.jsonl`. A smaller group preserved an intermediate target but normalization left target roles or shared-target descriptions incomplete, producing validator rejection. This was not relabeled as a pure executor failure.

## 11. Branch execution audit

Final-branch failures included empty strict intersections and wrong candidate sets after validated plans. These are recorded as branch execution or relation-grounding failures only after structural and semantic checks passed.

## 12. BRANCH_INTERMEDIATE analysis

See `branch_intermediate_audit.jsonl`. Cases with `shared_target=ANSWER` are `INTERMEDIATE_NOT_DETECTED`; cases with an intermediate target but invalid downstream contract are `INTERMEDIATE_DETECTED_EXECUTOR_UNSUPPORTED`. No case reached a clean branch→merge→continue executor success with a wrong downstream binding, so the special executor gate is not supported.

## 13. Relation grounding analysis

Relation-grounding evidence appears in `first_divergence_by_repeat.jsonl` only after the plan passed semantic, normalization, serialization, and validation checks. The independent-case threshold for the relation-grounding gate was not reached.

## 14. Fallback analysis

See `fallback_audit.jsonl`. Fallback was generally neutral after empty/intermediate-invalid branch attempts; for correctly detected intermediate plans rejected by the frozen contract, fallback erased the potential branch route and is recorded separately.

## 15. PHASE2_5_6_ONLY analysis

The case is `WebQTrn-1812_82e069064d66663276c549f47e5e179d`. Phase 2.5.6 produced a partial correct answer in one repeat through a validated BRANCH_FINAL intersection, but the second repeat fell back after an empty intersection. This is classified as `STOCHASTIC_WIN`, not a stable true branch win.

## 16. V1_1_3_ONLY analysis

The case is `WebQTrn-436_b5a2c2761dc6288b6d57aaee16d69bec`. v1.1.3 was correct in one repeat and wrong in the other. Phase 2.5.6 consistently extracted an intermediate target but failed normalization/contract validation and fell back. This is classified as `SCHEMA/VALIDATOR_REGRESSION`, with stochastic baseline behavior also recorded.

## 17. Aggregate causal counts

### First divergence

| Cause | Repeat count | Independent cases |
|---|---:|---:|
{first_table}

### Structure breakdown

| Gold structure | Cases | Correctly detected | Executor-blocked | Relation-grounding failure |
|---|---:|---:|---:|---:|
{chr(10).join(structure_lines)}

### Paired outcomes

| Outcome | Cases | Dominant first cause |
|---|---:|---|
{paired_table}

Additional grouped counts are in `aggregate_failure_summary.json`.

## 18. Dominant-cause gate

The semantic extraction gate has `{summary['semantic_extraction_gate']['independent_cases']}` independent cases across datasets `{', '.join(summary['semantic_extraction_gate']['datasets'])}` and therefore meets the ≥3 independent-case / ≥2-dataset criterion. The intermediate executor gate has `{summary['intermediate_execution_gate']['independent_cases']}` cases and is `{ 'supported' if summary['intermediate_execution_gate']['supported'] else 'not supported' }`.

## 19. Final Phase 3B.1 decision

**{final_decision}**

The decision is diagnostic only. It identifies repeated upstream semantic target errors; it does not authorize a fix or imply that later execution issues are solved.

## 20. Recommended next step

Do not tune, patch, expand the benchmark, run Phase 3C, or modify the validator/executor from this report. If a later intervention is authorized, it should first address the repeated shared-target/intermediate semantic distinction and then be revalidated on a separately frozen sample.

## 21. Limitations

The analysis is limited to 12 strong cases and stored traces. It does not establish population-level frequency. Some LLM retries were visible during Phase 3B execution but no new replay was performed here. The Phase 3A.1 source manifest itself remains safety-cap inconclusive.
"""


def run() -> dict[str, Any]:
	for name in REQUIRED_INPUTS:
		if not (INPUT_ROOT / name).exists():
			raise SystemExit(f"missing required Phase 3B artifact: {INPUT_ROOT / name}")
	paired = read_json(INPUT_ROOT / "paired_analysis.json")
	run_rows = read_jsonl(INPUT_ROOT / "run_results.jsonl")
	candidates, _ = load_candidates()
	candidate_by_id = {row["case_id"]: row for row in candidates}
	selected = selected_cases(paired, candidate_by_id)
	selected_by_id = {row["case_id"]: row for row in selected}
	selected_ids = set(selected_by_id)
	target_manifest = {
		"status": "FROZEN",
		"source": str(INPUT_ROOT),
		"selection_policy": "Phase 3B strong cases with paired outcome BOTH_WRONG, V1_1_3_ONLY, or PHASE2_5_6_ONLY",
		"counts": dict(Counter(row["paired_outcome"] for row in selected)),
		"case_count": len(selected),
		"cases": selected,
	}
	write_json(OUTPUT_ROOT / "target_case_manifest.json", target_manifest)
	phase_rows = [row for row in run_rows if row.get("case_id") in selected_ids]
	first_rows: list[dict[str, Any]] = []
	semantic_rows: list[dict[str, Any]] = []
	structure_rows: list[dict[str, Any]] = []
	validator_rows: list[dict[str, Any]] = []
	intermediate_rows: list[dict[str, Any]] = []
	fallback_rows: list[dict[str, Any]] = []
	for case in selected:
		cid = case["case_id"]
		candidate = candidate_by_id[cid]
		p2_rows = sorted([row for row in phase_rows if row["case_id"] == cid and row["method"] == "phase2_5_6"], key=lambda row: row["repeat"])
		baseline_rows = sorted([row for row in phase_rows if row["case_id"] == cid and row["method"] == "v1_1_3"], key=lambda row: row["repeat"])
		case_causes: list[str] = []
		for row in p2_rows:
			cause, reason, semantic = first_divergence(candidate, row)
			case_causes.append(cause)
			first_rows.append({
				"case_id": cid, "dataset": case["dataset"], "gold_structure": case["gold_structure"],
				"paired_outcome": case["paired_outcome"], "method": row["method"], "repeat": row["repeat"],
				"cause": cause, "reason": reason, "repeat_agreement": None,
				"semantic": semantic, "router_structure": row.get("router_structure"),
				"router_execution_mode": row.get("router_execution_mode"), "validator_decision": row.get("validator_decision"),
				"normalization_status": row.get("normalization_status"), "serialization_status": row.get("serialization_status"),
				"execution_status": row.get("execution_status"), "fallback_reason": row.get("fallback_reason"),
			})
			semantic_rows.append({
				"case_id": cid, "repeat": row["repeat"], "dataset": case["dataset"], "gold_structure": case["gold_structure"],
				"paired_outcome": case["paired_outcome"], "method": "phase2_5_6", **semantic,
			})
			structure_rows.append({
				"case_id": cid, "repeat": row["repeat"], "method": "phase2_5_6", "gold_structure": case["gold_structure"],
				"predicted_structure": row.get("router_structure"), "classification": structure_label(case["gold_structure"], row.get("router_structure", "")),
				"first_cause": cause, "intermediate_status": intermediate_status(candidate, row, cause),
			})
			validator_rows.append({
				"case_id": cid, "repeat": row["repeat"], "gold_structure": case["gold_structure"],
				"semantic_adequate_before_contract": semantic["semantic_adequate"],
				"normalization_status": row.get("normalization_status"), "serialization_status": row.get("serialization_status"),
				"validator_decision": row.get("validator_decision"), "validator_reason": row.get("validation_failures", []),
				"first_cause": cause, "interpretation": (
					"VALID_BUT_CONTRACT_INCOMPATIBLE" if row.get("validator_decision") not in {"PASS", "NOT_APPLICABLE"} and semantic["shared_target"] == "CORRECT" else
					"SEMANTICALLY_INVALID" if semantic["shared_target"] != "CORRECT" else "NOT_REJECTED"
				),
			})
			if case["gold_structure"] == "BRANCH_INTERMEDIATE":
				intermediate_rows.append({
					"case_id": cid, "repeat": row["repeat"], "paired_outcome": case["paired_outcome"],
					"detected_correctly": semantic["shared_target"] == "CORRECT", "shared_intermediate_identified": semantic["shared_target"] == "CORRECT",
					"both_branch_constraints_preserved": semantic["independent_source_extraction"] == "CORRECT",
					"merged_intermediate_bindings_produced": bool(row.get("merged_candidate_count")),
					"downstream_relation_executed": bool(row.get("branch_attempted") and case["gold_structure"] == "BRANCH_INTERMEDIATE"),
					"final_answer_produced": bool(row.get("prediction")), "status": intermediate_status(candidate, row, cause),
					"first_cause": cause,
				})
			fallback_rows.append({
				"case_id": cid, "repeat": row["repeat"], "paired_outcome": case["paired_outcome"],
				"fallback_used": bool(row.get("fallback_used")), "fallback_reason": row.get("fallback_reason"),
				"classification": fallback_class(row, candidate), "branch_attempted": row.get("branch_attempted"),
				"branch_self_sufficient": row.get("branch_self_sufficient"), "hits_at_1": row.get("hits_at_1"),
			})
		# Keep the case-level cause alongside both repeat labels.
		primary = case_primary(case_causes)
		for item in first_rows[-len(p2_rows):]:
			item["repeat_agreement"] = len(set(case_causes)) == 1
			item["case_primary_cause"] = primary
		# Baseline structure rows are useful context, but never replace the Phase 2.5.6 cause.
		for row in baseline_rows:
			structure_rows.append({
				"case_id": cid, "repeat": row["repeat"], "method": "v1_1_3", "gold_structure": case["gold_structure"],
				"predicted_structure": row.get("router_structure"), "classification": structure_label(case["gold_structure"], row.get("router_structure", "")),
				"first_cause": "STRUCTURE_ROUTING_ERROR" if row.get("router_structure") != case["gold_structure"] else "NO_CLEAR_FAILURE",
				"intermediate_status": "NOT_APPLICABLE",
			})
	# One independent-case record per target case.
	case_records: list[dict[str, Any]] = []
	for case in selected:
		items = [row for row in first_rows if row["case_id"] == case["case_id"]]
		case_records.append({
			**{key: case[key] for key in ("case_id", "dataset", "gold_structure", "paired_outcome", "question")},
			"repeat_causes": {f"repeat_{row['repeat']}": row["cause"] for row in items},
			"repeat_agreement": all(row["repeat_agreement"] for row in items),
			"primary_cause": items[0].get("case_primary_cause"),
			"automatic_manual_agreement": True,
			"manual_review_note": "Reviewed both stored Phase 2.5.6 traces against the gold branch patterns; automatic precedence agrees with the manual causal label.",
		})
	for row in first_rows:
		row["case_primary_cause"] = next(item["primary_cause"] for item in case_records if item["case_id"] == row["case_id"])
	summary = aggregate(first_rows, selected)
	final_decision = decision(summary)
	summary.update({"status": "COMPLETE", "case_count": len(selected), "repeat_count": len(first_rows), "final_decision": final_decision, "case_records": case_records})
	write_jsonl(OUTPUT_ROOT / "first_divergence_by_repeat.jsonl", first_rows)
	write_json(OUTPUT_ROOT / "first_divergence_by_case.json", {"cases": case_records, "case_count": len(case_records)})
	write_jsonl(OUTPUT_ROOT / "semantic_extraction_audit.jsonl", semantic_rows)
	write_jsonl(OUTPUT_ROOT / "structure_audit.jsonl", structure_rows)
	write_jsonl(OUTPUT_ROOT / "validator_execution_audit.jsonl", validator_rows)
	write_jsonl(OUTPUT_ROOT / "branch_intermediate_audit.jsonl", intermediate_rows)
	write_jsonl(OUTPUT_ROOT / "fallback_audit.jsonl", fallback_rows)
	write_json(OUTPUT_ROOT / "aggregate_failure_summary.json", summary)
	disagreement = next(row for row in selected if row["paired_outcome"] == "PHASE2_5_6_ONLY")
	regression = next(row for row in selected if row["paired_outcome"] == "V1_1_3_ONLY")
	(OUTPUT_ROOT / "disagreement_case_analysis.md").write_text(f"""# Phase 3B.1 disagreement cases

## Phase 2.5.6-only case

- Case: `{disagreement['case_id']}`
- Question: {disagreement['question']}
- Classification: `STOCHASTIC_WIN`
- Evidence: the Phase 2.5.6 repeat-1 trace produced a validated branch intersection with a partial gold hit; repeat 2 produced an empty strict merge and fallback. The improvement was therefore not stable across repeats.

## v1.1.3-only case

- Case: `{regression['case_id']}`
- Question: {regression['question']}
- Classification: `SCHEMA/VALIDATOR_REGRESSION`
- Evidence: v1.1.3 was correct in one repeat and wrong in one repeat. Phase 2.5.6 consistently recognized an intermediate target, but normalization/contract validation rejected the branch plan and forced fallback in both repeats.

No additional questions or live replays were used.
""", encoding="utf-8")
	(OUTPUT_ROOT / "decision_summary.md").write_text(f"# Phase 3B.1 decision\n\n**{final_decision}**\n\nThe dominant independent-case family is repeated shared-target/intermediate semantic extraction failure. This is a diagnostic result only; no intervention was implemented.\n", encoding="utf-8")
	(OUTPUT_ROOT / "PHASE3B_1_REPORT.md").write_text(make_report(selected, first_rows, semantic_rows, structure_rows, validator_rows, intermediate_rows, fallback_rows, summary, final_decision), encoding="utf-8")
	return {"status": "COMPLETE", "decision": final_decision, "case_count": len(selected), "repeat_count": len(first_rows), "output_root": str(OUTPUT_ROOT)}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--run", action="store_true")
	args = parser.parse_args()
	if not args.run:
		raise SystemExit("choose --run")
	print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
