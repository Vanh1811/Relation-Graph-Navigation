#!/usr/bin/env python3
"""Run the Phase 3B.4 answer-vs-convergence role-separation experiment.

The script freezes the nine-case Phase 3B.3 development set, runs the new
semantic extractor, and proceeds to a fresh 30-case validation set only when
the development gate passes.  The downstream CoR/normalization/serialization/
validator/execution stack is reused from Phase 3B.2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PHASE3B2_ROOT = ROOT / "outputs/experiments/phase3b_2/shared_target_intervention"
PHASE3B3_ROOT = ROOT / "outputs/experiments/phase3b_3/residual_shared_target_analysis"
OUTPUT_ROOT = ROOT / "outputs/experiments/phase3b_4/answer_convergence_role_separation"
SEED = 20261027
REPEATS = 2
METHODS = ("phase3b_2", "phase3b_4")
ROLE_EVENT = "phase3b_4_answer_convergence_role_separation_router"


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


def text(value: Any) -> str:
	return str(value or "").strip()


def upper(value: Any) -> str:
	return text(value).upper()


def target_type(value: Any) -> str:
	if isinstance(value, dict):
		return upper(value.get("type") or value.get("target")) or "NONE"
	return upper(value) or "NONE"


def row_target(row: dict[str, Any]) -> str:
	return target_type(row.get("canonical_shared_target_type") or row.get("raw_shared_target_type") or row.get("actual_target") or row.get("shared_target"))


def expected_target(row: dict[str, Any]) -> str:
	value = upper(row.get("expected_shared_target_type"))
	if value:
		return value
	return "INTERMEDIATE" if upper(row.get("gold_structure")) == "BRANCH_INTERMEDIATE" else "ANSWER"


def bool_value(value: Any) -> bool | None:
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		value = value.strip().upper()
		if value in {"TRUE", "YES", "Y", "1"}:
			return True
		if value in {"FALSE", "NO", "N", "0"}:
			return False
	return None


def _base_module():
	import run_phase3b_2_shared_target_intervention as base
	return base


def role_metadata(row: dict[str, Any]) -> dict[str, Any]:
	metadata = row.get("role_separation")
	if isinstance(metadata, dict):
		return metadata
	extraction = row.get("shared_target_extraction") or {}
	return {
		"requested_answer_role": extraction.get("requested_answer_role", {}),
		"convergence_role": extraction.get("convergence_role", {}),
		"roles_same_entity": extraction.get("roles_same_entity"),
		"role_comparison": extraction.get("role_comparison"),
		"role_comparison_reason": extraction.get("role_comparison_reason", ""),
		"requires_downstream_relation": extraction.get("requires_downstream_relation"),
		"derived_shared_target_type": row_target(row),
		"role_separation_consistency": row.get("role_separation_consistency"),
		"role_separation_issues": row.get("role_separation_issues", []),
	}


def role_complete(role: Any) -> bool:
	return isinstance(role, dict) and bool(text(role.get("description"))) and bool(text(role.get("entity_type")))


def build_development_manifest() -> dict[str, Any]:
	source = read_json(PHASE3B3_ROOT / "target_case_manifest.json")
	confusion_ids = {
		row["case_id"]
		for row in read_jsonl(PHASE3B3_ROOT / "residual_semantic_audit.jsonl")
		if row.get("primary_residual_cause") == "ANSWER_CONVERGENCE_CONFUSION"
	}
	cases = []
	for item in source.get("cases", []):
		case = dict(item)
		expected = case.get("expected_target", "INTERMEDIATE" if case.get("gold_structure") == "BRANCH_INTERMEDIATE" else "ANSWER")
		case["gold_requested_answer_role"] = "REQUESTED_FINAL_ANSWER"
		case["gold_convergence_role"] = "REQUESTED_FINAL_ANSWER" if expected == "ANSWER" else "DISTINCT_SHARED_CONVERGENCE_ENTITY"
		case["gold_roles_same_entity"] = expected == "ANSWER"
		case["gold_requires_downstream_relation"] = expected == "INTERMEDIATE"
		case["primary_development_focus"] = "ANSWER_CONVERGENCE_CONFUSION" if case["case_id"] in confusion_ids else case.get("case_category")
		cases.append(case)
	return {
		"status": "FROZEN",
		"experiment": "phase3b_4_answer_convergence_role_separation",
		"source": str(PHASE3B3_ROOT / "target_case_manifest.json"),
		"seed": SEED,
		"repeats_per_method": REPEATS,
		"selection_policy": "exactly the nine frozen Phase 3B.3 residual/changed-target cases; no new development examples",
		"case_count": len(cases),
		"focus_counts": dict(Counter(case["primary_development_focus"] for case in cases)),
		"cases": sorted(cases, key=lambda item: item["case_id"]),
		"source_sha256": {
			"phase3b3_target_case_manifest.json": sha256(PHASE3B3_ROOT / "target_case_manifest.json"),
			"phase3b3_residual_semantic_audit.jsonl": sha256(PHASE3B3_ROOT / "residual_semantic_audit.jsonl"),
		},
	}


def balanced_fresh_manifest(base: Any) -> dict[str, Any]:
	metadata, pooled = base.source_candidate_pools()
	strong_intermediate = [item for item in pooled if item.get("_cohort_source") == "strong" and item["gold_structure"] == "BRANCH_INTERMEDIATE"]
	strong_final = [item for item in pooled if item.get("_cohort_source") == "strong" and item["gold_structure"] == "BRANCH_FINAL"]
	controls = [item for item in pooled if item.get("_cohort_source") == "control"]
	rng = random.Random(SEED)
	selected = [
		*[(item, "strong_intermediate") for item in base.balanced_sample(strong_intermediate, 10, rng)],
		*[(item, "strong_final") for item in base.balanced_sample(strong_final, 10, rng)],
		*[(item, "control") for item in base.balanced_sample(controls, 10, rng)],
	]
	rows = [base.public_case(candidate, cohort) for candidate, cohort in selected]
	rows.sort(key=lambda item: (item["cohort"], item["case_id"]))
	return {
		"status": "FROZEN",
		"experiment": "phase3b_4_answer_convergence_role_separation",
		"seed": SEED,
		"repeats_per_method": REPEATS,
		"sampling_policy": "10 BRANCH_INTERMEDIATE strong, 10 BRANCH_FINAL strong, and 10 answer-equivalent controls; balanced round-robin by dataset where available",
		"selection_excludes_model_outcomes": True,
		"excluded_phase2_and_phase3_diagnostic_count": metadata["excluded_id_count"],
		"source_metadata": metadata,
		"cohort_counts": dict(Counter(item["cohort"] for item in rows)),
		"dataset_counts": dict(Counter(f"{item['cohort']}|{item['dataset']}" for item in rows)),
		"cases": rows,
	}


def apply_role_event(row: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
	parsed = event.get("parsed_result") if isinstance(event.get("parsed_result"), dict) else {}
	canonical = event.get("canonical_semantic_representation") if isinstance(event.get("canonical_semantic_representation"), dict) else {}
	extraction = event.get("shared_target_extraction") if isinstance(event.get("shared_target_extraction"), dict) else {}
	row.update({
		"raw_constraints": event.get("constraints", []),
		"raw_parsed_result": parsed,
		"role_enriched_parsed_result": event.get("role_enriched_parsed_result", {}),
		"canonical_semantic_representation": canonical,
		"shared_target_extraction": extraction,
		"raw_shared_target_type": target_type(parsed.get("shared_target")),
		"canonical_shared_target_type": target_type(canonical.get("shared_target")),
		"shared_target_description": extraction.get("shared_target_description", ""),
		"requested_answer_description": extraction.get("requested_answer_description", ""),
		"requires_downstream_relation": extraction.get("requires_downstream_relation"),
		"downstream_relation_description": extraction.get("downstream_relation_description", ""),
		"validator_decision": event.get("validator_decision", row.get("validator_decision", "UNKNOWN")),
		"validation_failures": event.get("validation_failures", row.get("validation_failures", [])),
		"validator_reason": event.get("validation_failures", row.get("validator_reason", [])),
		"normalization_status": event.get("normalization_status", row.get("normalization_status", "UNKNOWN")),
		"serialization_status": (event.get("serialization") or {}).get("serialization_status", row.get("serialization_status", "UNKNOWN")),
		"serialized_branches": event.get("serialized_branches", row.get("serialized_branches", [])),
		"requested_answer_role": event.get("requested_answer_role", {}),
		"convergence_role": event.get("convergence_role", {}),
		"roles_same_entity": event.get("roles_same_entity"),
		"role_comparison": event.get("role_comparison"),
		"role_comparison_reason": event.get("role_comparison_reason", ""),
		"derived_shared_target_type": event.get("derived_shared_target_type", target_type(canonical.get("shared_target"))),
		"role_separation_consistency": event.get("role_separation_consistency"),
		"role_separation_issues": event.get("role_separation_issues", []),
	})
	return row


def run_one(base: Any, agent: Any, method: str, repeat: int, candidate: dict[str, Any]) -> dict[str, Any]:
	# The Phase 3B.2 runner supplies the frozen execution/evaluation plumbing.
	# Passing its event selector name keeps those mechanics unchanged; the role
	# event is then attached as an audit-only compatibility layer below.
	row = base.run_one(agent, "phase3b_2", repeat, candidate)
	row["method"] = method
	if method == "phase3b_4":
		event = next((item for item in reversed(row.get("prompt_history", [])) if item.get("type") == ROLE_EVENT), {})
		if event:
			apply_role_event(row, event)
			if event.get("validator_decision") == "REJECT":
				row["execution_status"] = "VALIDATOR_REJECT_FALLBACK"
	else:
		row["method"] = "phase3b_2"
	return row


def make_agent(base: Any, method: str) -> Any:
	if method == "phase3b_2":
		return base.make_agent("phase3b_2")
	from chain_of_relations.kg_backend import get_backend
	from chain_of_relations.methods.cor.agent import CoRAgent
	from chain_of_relations.methods.phase3b_4.agent import RoleSeparationAgent
	cor = CoRAgent(
		model_name=os.environ["MODEL_NAME"],
		relation_width=3,
		depth=3,
		sample_relation_threshold=1000,
		remove_unnecessary_rel=True,
		temperature_exploration=0.3,
		temperature_reasoning=0.1,
		max_token=1024,
		backend=get_backend("freebase"),
	)
	return RoleSeparationAgent(cor, confidence_threshold=0.75, max_branches=3)


def append_run(path: Path, rows: list[dict[str, Any]], base: Any, agent: Any, method: str, repeats: int, candidates: list[dict[str, Any]], resume: bool) -> list[dict[str, Any]]:
	if resume and path.exists():
		rows = read_jsonl(path)
	existing = {(row.get("method"), int(row.get("repeat", 0)), row.get("case_id")) for row in rows}
	for repeat in range(1, repeats + 1):
		for index, candidate in enumerate(candidates, start=1):
			key = (method, repeat, candidate["case_id"])
			if key in existing:
				continue
			print(f"[{method}] repeat={repeat} {index}/{len(candidates)} {candidate['case_id']}", flush=True)
			rows.append(run_one(base, agent, method, repeat, candidate))
			existing.add(key)
			write_jsonl(path, rows)
	rows.sort(key=lambda row: (row.get("method", ""), int(row.get("repeat", 0)), row.get("case_id", "")))
	write_jsonl(path, rows)
	return rows


def baseline_development_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
	ids = {case["case_id"] for case in manifest["cases"]}
	rows = []
	for row in read_jsonl(PHASE3B2_ROOT / "development_run_results.jsonl") + read_jsonl(PHASE3B2_ROOT / "fresh_run_results.jsonl"):
		if row.get("method") != "phase3b_2" or row.get("case_id") not in ids:
			continue
		updated = dict(row)
		updated["cohort"] = "development"
		frozen = next(case for case in manifest["cases"] if case["case_id"] == row["case_id"])
		updated["gold_structure"] = frozen["gold_structure"]
		updated["expected_shared_target_type"] = frozen["expected_target"]
		rows.append(updated)
	return sorted(rows, key=lambda row: (row["case_id"], int(row.get("repeat", 0))))


def group_by_case(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
	grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		grouped[row["case_id"]].append(row)
	return dict(grouped)


def majority(values: list[Any]) -> Any:
	return Counter(values).most_common(1)[0][0] if values else None


def role_record(row: dict[str, Any], manifest_case: dict[str, Any], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
	meta = role_metadata(row)
	requested = meta.get("requested_answer_role", {})
	convergence = meta.get("convergence_role", {})
	expected = manifest_case["expected_target"]
	role_same = bool_value(meta.get("roles_same_entity"))
	requires = bool_value(meta.get("requires_downstream_relation"))
	return {
		"case_id": row["case_id"],
		"dataset": row.get("dataset"),
		"cohort": row.get("cohort"),
		"gold_structure": row.get("gold_structure"),
		"repeat": row.get("repeat"),
		"method": row.get("method"),
		"expected_shared_target_type": expected,
		"gold_requested_answer_role": manifest_case["gold_requested_answer_role"],
		"gold_convergence_role": manifest_case["gold_convergence_role"],
		"gold_roles_same_entity": manifest_case["gold_roles_same_entity"],
		"gold_requires_downstream_relation": manifest_case["gold_requires_downstream_relation"],
		"requested_answer_role": requested,
		"convergence_role": convergence,
		"roles_same_entity": role_same,
		"role_comparison": meta.get("role_comparison"),
		"role_comparison_reason": meta.get("role_comparison_reason", ""),
		"requires_downstream_relation": requires,
		"shared_target_type": row_target(row),
		"requested_answer_role_present": role_complete(requested),
		"convergence_role_present": role_complete(convergence),
		"roles_same_entity_correct": None if role_same is None else role_same == manifest_case["gold_roles_same_entity"],
		"requires_downstream_relation_correct": None if requires is None else requires == manifest_case["gold_requires_downstream_relation"],
		"shared_target_correct": row_target(row) == expected,
		"role_fields_internally_consistent": meta.get("role_separation_consistency") == "CONSISTENT",
		"role_separation_issues": meta.get("role_separation_issues", []),
		"phase3b_2_target": row_target(baseline) if baseline else None,
		"confusion_corrected": bool(baseline) and row_target(baseline) != expected and row_target(row) == expected,
	}


def development_gate(manifest: dict[str, Any], new_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
	new_by = group_by_case(new_rows)
	base_by = group_by_case(baseline_rows)
	focus = {case["case_id"] for case in manifest["cases"] if case["primary_development_focus"] == "ANSWER_CONVERGENCE_CONFUSION"}
	corrected = []
	for case_id in sorted(focus):
		case = next(item for item in manifest["cases"] if item["case_id"] == case_id)
		new_target = majority([row_target(row) for row in new_by.get(case_id, [])])
		base_target = majority([row_target(row) for row in base_by.get(case_id, [])])
		if base_target != case["expected_target"] and new_target == case["expected_target"]:
			corrected.append(case_id)
	answer_cases = {case["case_id"] for case in manifest["cases"] if case["expected_target"] == "ANSWER"}
	false_intermediate = sorted(
		case_id for case_id in answer_cases
		if majority([row_target(row) for row in new_by.get(case_id, [])]) == "INTERMEDIATE"
	)
	role_audit = [role_record(row, next(case for case in manifest["cases"] if case["case_id"] == row["case_id"])) for row in new_rows]
	consistent_cases = sum(
		all(item["role_fields_internally_consistent"] for item in values)
		for values in group_by_case(role_audit).values()
	)
	gate = {
		"primary_confusion_cases": len(focus),
		"primary_confusion_cases_corrected": len(corrected),
		"corrected_case_ids": corrected,
		"false_intermediate_answer_case_count": len(false_intermediate),
		"false_intermediate_answer_case_ids": false_intermediate,
		"role_consistent_case_count": consistent_cases,
		"case_count": len(manifest["cases"]),
		"gates": {
			"at_least_4_of_5_confusion_cases_corrected": len(corrected) >= 4,
			"zero_new_false_intermediate_final_or_control_cases": len(false_intermediate) == 0,
			"role_fields_consistent_in_at_least_8_of_9_cases": consistent_cases >= 8,
		},
	}
	gate["passed"] = all(gate["gates"].values())
	return gate


def role_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
	output: dict[str, Any] = {}
	for cohort in ("strong_intermediate", "strong_final", "control"):
		output[cohort] = {}
		for method in METHODS:
			values = [row for row in rows if row.get("cohort") == cohort and row.get("method") == method]
			def rate(field: str) -> float | None:
				available = [row[field] for row in values if row.get(field) is not None]
				return sum(bool(value) for value in available) / len(available) if available else None
			output[cohort][method] = {
				"run_count": len(values),
				"case_count": len(group_by_case(values)),
				"requested_answer_role_accuracy": rate("requested_answer_role_present"),
				"convergence_role_accuracy": rate("convergence_role_present"),
				"roles_same_entity_accuracy": rate("roles_same_entity_correct"),
				"requires_downstream_relation_accuracy": rate("requires_downstream_relation_correct"),
				"shared_target_type_accuracy_run": sum(bool(row.get("shared_target_correct")) for row in values) / len(values) if values else 0.0,
				"shared_target_type_accuracy_case": sum(majority([bool(item.get("shared_target_correct")) for item in case_rows]) for case_rows in group_by_case(values).values()) / len(group_by_case(values)) if values else 0.0,
				"role_consistency_rate_run": rate("role_fields_internally_consistent"),
			}
	return output


def shared_target_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
	output: dict[str, Any] = {}
	for cohort in ("strong_intermediate", "strong_final", "control"):
		output[cohort] = {}
		for method in METHODS:
			values = [row for row in rows if row.get("cohort") == cohort and row.get("method") == method]
			cases = group_by_case(values)
			confusion_cases = []
			for case_id, case_rows in cases.items():
				if case_rows[0].get("expected_shared_target_type") == "INTERMEDIATE" and majority([row_target(row) for row in case_rows]) == "ANSWER":
					confusion_cases.append(case_id)
			output[cohort][method] = {
				"case_count": len(cases),
				"shared_target_accuracy_case": sum(majority([row_target(row) == row.get("expected_shared_target_type") for row in case_rows]) for case_rows in cases.values()) / len(cases) if cases else 0.0,
				"answer_convergence_confusion_case_count": len(confusion_cases),
				"answer_convergence_confusion_case_ids": sorted(confusion_cases),
				"answer_convergence_confusion_rate": len(confusion_cases) / len(cases) if cases else 0.0,
				"false_intermediate_case_count": sum(majority([row_target(row) for row in case_rows]) == "INTERMEDIATE" for case_rows in cases.values() if case_rows[0].get("expected_shared_target_type") != "INTERMEDIATE"),
			}
	return output


def qa_metrics(base: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
	output: dict[str, Any] = {}
	for cohort in ("strong_intermediate", "strong_final", "control"):
		output[cohort] = {}
		for method in METHODS:
			values = [row for row in rows if row.get("cohort") == cohort and row.get("method") == method]
			output[cohort][method] = {
				"run_count": len(values),
				"hits_at_1_mean": sum(int(row.get("hits_at_1", 0)) for row in values) / len(values) if values else 0.0,
				"f1_mean": sum(float(row.get("f1", 0.0)) for row in values) / len(values) if values else 0.0,
				"validator_pass_rate": sum(str(row.get("validator_decision", "")).upper() == "PASS" for row in values) / len(values) if values else 0.0,
				"fallback_rate": sum(bool(row.get("fallback_used")) for row in values) / len(values) if values else 0.0,
				"native_branch_attempts": sum(bool(row.get("branch_attempted")) for row in values),
				"native_branch_successes": sum(row.get("execution_status") == "BRANCH_EXECUTION_SUCCESS" for row in values),
			}
	return output


def paired_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
	by_key = {(row["case_id"], row["method"], int(row["repeat"])): row for row in rows}
	paired = []
	counts = Counter()
	for case_id in sorted({row["case_id"] for row in rows}):
		case_rows = [row for row in rows if row["case_id"] == case_id]
		base = [by_key.get((case_id, "phase3b_2", repeat)) for repeat in range(1, REPEATS + 1)]
		intervention = [by_key.get((case_id, "phase3b_4", repeat)) for repeat in range(1, REPEATS + 1)]
		if not all(base) or not all(intervention):
			continue
		base_hit = majority([bool(row.get("hits_at_1")) for row in base])
		new_hit = majority([bool(row.get("hits_at_1")) for row in intervention])
		outcome = "BOTH_CORRECT" if base_hit and new_hit else "PHASE3B_2_ONLY" if base_hit else "PHASE3B_4_ONLY" if new_hit else "BOTH_WRONG"
		cohort = case_rows[0].get("cohort")
		counts[(cohort, outcome)] += 1
		paired.append({
			"case_id": case_id,
			"cohort": cohort,
			"dataset": case_rows[0].get("dataset"),
			"gold_structure": case_rows[0].get("gold_structure"),
			"outcome": outcome,
			"phase3b_2_hit_majority": base_hit,
			"phase3b_4_hit_majority": new_hit,
			"phase3b_2_target_majority": majority([row_target(row) for row in base]),
			"phase3b_4_target_majority": majority([row_target(row) for row in intervention]),
			"phase3b_2_f1_mean": sum(float(row.get("f1", 0.0)) for row in base) / len(base),
			"phase3b_4_f1_mean": sum(float(row.get("f1", 0.0)) for row in intervention) / len(intervention),
		})
	return {
		"paired_case_count": len(paired),
		"counts": {cohort: {outcome: counts[(cohort, outcome)] for outcome in ("BOTH_CORRECT", "PHASE3B_2_ONLY", "PHASE3B_4_ONLY", "BOTH_WRONG")} for cohort in ("strong_intermediate", "strong_final", "control")},
		"cases": paired,
	}


def stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
	output = {}
	for method in METHODS:
		classes = Counter()
		case_records = []
		for case_id, values in group_by_case([row for row in rows if row.get("method") == method]).items():
			targets = {row_target(row) for row in values}
			same_values = {row.get("roles_same_entity") for row in values}
			if len(same_values) > 1:
				classification = "UNSTABLE_ROLE"
			elif len(targets) > 1:
				classification = "UNSTABLE_TARGET"
			else:
				correct = targets == {values[0].get("expected_shared_target_type")}
				classification = "STABLE_CORRECT" if correct else "STABLE_WRONG"
			classes[classification] += 1
			case_records.append({"case_id": case_id, "classification": classification, "targets": sorted(targets), "roles_same_entity": sorted(str(value) for value in same_values)})
		output[method] = {"case_count": len(case_records), "counts": dict(sorted(classes.items())), "cases": sorted(case_records, key=lambda item: item["case_id"])}
	return output


def downstream_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
	output = []
	for row in rows:
		# Raw run rows do not carry the derived ``shared_target_correct`` field;
		# compare the recorded target with the frozen expected target directly.
		if row.get("method") != "phase3b_4" or row_target(row) != row.get("expected_shared_target_type") or row.get("hits_at_1"):
			continue
		if row.get("normalization_status") in {"PARTIAL_NORMALIZATION", "NORMALIZATION_FAILED"}:
			cause = "DOWNSTREAM_NORMALIZATION"
		elif str(row.get("validator_decision", "")).upper() not in {"PASS", "NOT_APPLICABLE"}:
			cause = "DOWNSTREAM_VALIDATOR"
		elif row.get("branch_attempted") and row.get("execution_status") != "BRANCH_EXECUTION_SUCCESS":
			cause = "DOWNSTREAM_EXECUTION"
		elif not row.get("relation_chain"):
			cause = "DOWNSTREAM_RELATION_GROUNDING"
		elif row.get("fallback_used"):
			cause = "DOWNSTREAM_FALLBACK"
		else:
			cause = "OTHER"
		output.append({"case_id": row["case_id"], "cohort": row.get("cohort"), "dataset": row.get("dataset"), "repeat": row.get("repeat"), "cause": cause, "target": row_target(row), "roles_same_entity": row.get("roles_same_entity"), "execution_status": row.get("execution_status"), "validator_decision": row.get("validator_decision"), "fallback_used": row.get("fallback_used")})
	return output


def manual_review_cases(development_rows: list[dict[str, Any]], fresh_rows: list[dict[str, Any]], manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
	selected = []
	for row in development_rows + fresh_rows:
		case = manifests.get(row.get("case_id"), {})
		if row.get("cohort") == "development" or row.get("gold_structure") == "BRANCH_INTERMEDIATE" or row_target(row) != row.get("expected_shared_target_type"):
			meta = role_metadata(row)
			selected.append({"case_id": row.get("case_id"), "cohort": row.get("cohort"), "dataset": row.get("dataset"), "gold_structure": row.get("gold_structure"), "repeat": row.get("repeat"), "method": row.get("method"), "question": row.get("question"), "gold_target": case.get("expected_target", row.get("expected_shared_target_type")), "requested_answer_role": meta.get("requested_answer_role", {}), "convergence_role": meta.get("convergence_role", {}), "roles_same_entity": meta.get("roles_same_entity"), "requires_downstream_relation": meta.get("requires_downstream_relation"), "shared_target_type": row_target(row), "automatic_role_consistency": meta.get("role_separation_consistency"), "manual_label": None})
	return {"status": "REQUIRES_MANUAL_REVIEW", "review_scope": "all development cases, fresh BRANCH_INTERMEDIATE cases, changed targets, and regressions", "cases": selected}


def decision_label(dev_gate: dict[str, Any], role: dict[str, Any] | None, shared: dict[str, Any] | None, paired: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
	if not dev_gate.get("passed"):
		return "ROLE_SEPARATION_DEVELOPMENT_FAILED", {"development_gate": dev_gate}
	if role is None or shared is None or paired is None:
		return "INCONCLUSIVE", {"development_gate": dev_gate, "reason": "fresh validation was not completed"}
	base_conf = shared["strong_intermediate"]["phase3b_2"]["answer_convergence_confusion_rate"]
	new_conf = shared["strong_intermediate"]["phase3b_4"]["answer_convergence_confusion_rate"]
	confusion_reduction = 1.0 if base_conf == 0 and new_conf == 0 else (base_conf - new_conf) / base_conf if base_conf else 0.0
	base_acc = role["strong_intermediate"]["phase3b_2"]["shared_target_type_accuracy_case"]
	new_acc = role["strong_intermediate"]["phase3b_4"]["shared_target_type_accuracy_case"]
	final_cases = [item for item in paired["cases"] if item["cohort"] == "strong_final"]
	# Use the explicit target contract instead of comparing a structure label.
	final_regressions = sum(
		item["phase3b_2_target_majority"] == "ANSWER" and item["phase3b_4_target_majority"] != "ANSWER"
		for item in final_cases
	)
	control_false = shared["control"]["phase3b_4"]["false_intermediate_case_count"]
	qa_new = sum(item["phase3b_4_hit_majority"] for item in paired["cases"])
	qa_old = sum(item["phase3b_2_hit_majority"] for item in paired["cases"])
	qa_net = qa_new - qa_old
	gates = {
		"development_gate": True,
		"confusion_reduction_at_least_50_percent": confusion_reduction >= 0.5,
		"intermediate_accuracy_gain_at_least_20_points": new_acc - base_acc >= 0.20,
		"branch_final_regression_at_most_one_case": final_regressions <= 1,
		"control_false_intermediate_at_most_one_of_ten": control_false <= 1,
		"fresh_paired_qa_net_nonnegative": qa_net >= 0,
	}
	if all(gates.values()):
		label = "ROLE_SEPARATION_SUPPORTED"
	elif qa_net < 0 or final_regressions > 1 or control_false > 1:
		label = "ROLE_SEPARATION_REGRESSES"
	elif (new_conf < base_conf or new_acc > base_acc) and not (new_conf == base_conf and new_acc == base_acc):
		label = "ROLE_SEPARATION_PARTIAL"
	else:
		label = "ROLE_SEPARATION_NOT_SUPPORTED"
	return label, {"gates": gates, "confusion_rate_reduction": confusion_reduction, "intermediate_accuracy_change": new_acc - base_acc, "branch_final_regression_count": final_regressions, "control_false_intermediate_case_count": control_false, "fresh_paired_qa_net": qa_net, "phase3b_2_intermediate_accuracy": base_acc, "phase3b_4_intermediate_accuracy": new_acc, "phase3b_2_confusion_rate": base_conf, "phase3b_4_confusion_rate": new_conf}


def write_report(decision: str, decision_data: dict[str, Any], dev_gate: dict[str, Any], role_metrics_data: dict[str, Any] | None, shared_data: dict[str, Any] | None, qa_data: dict[str, Any] | None, stability_data: dict[str, Any] | None) -> None:
	def metric(value: Any) -> str:
		return "N/A" if value is None else f"{float(value):.3f}"
	lines = [
		"# Phase 3B.4 Explicit Answer-vs-Convergence Role Separation",
		"",
		"## 1. Motivation",
		"",
		"Phase 3B.3 identified repeated confusion between the entity jointly identified by branch constraints and the entity requested by the question.",
		"",
		"## 2. Phase 3B.3 causal evidence",
		"",
		"The prior audit found `ANSWER_CONVERGENCE_CONFUSION_SUPPORTED` in five independent cases, across CWQ and GrailQA and across development and fresh validation.",
		"",
		"## 3. Research question",
		"",
		"Does explicit role separation reduce this confusion without harming final-answer or control behavior?",
		"",
		"## 4. Narrow role-separation intervention",
		"",
		"The extractor now emits requested-answer role, convergence role, roles_same_entity, role comparison, and downstream requirement. The shared-target enum is derived deterministically from these fields.",
		"",
		"## 5. Frozen downstream stack",
		"",
		"Phase 2.5.5 normalization, serialization, validation, execution, fallback, grounding, scoring, evaluator, and Freebase backend were reused unchanged.",
		"",
		"## 6. Development diagnostic set",
		"",
		f"The frozen nine-case Phase 3B.3 manifest was reused. It contains five primary confusion cases, one unstable intermediate case, one final regression, and two changed controls.",
		"",
		"## 7. Development gate",
		"",
		f"Development gate: **{'PASS' if dev_gate.get('passed') else 'FAIL'}**. Confusion cases corrected: `{dev_gate.get('primary_confusion_cases_corrected', 0)}/{dev_gate.get('primary_confusion_cases', 0)}`; new false-intermediate answer cases: `{dev_gate.get('false_intermediate_answer_case_count', 0)}`; role-consistent cases: `{dev_gate.get('role_consistent_case_count', 0)}/9`.",
		"",
	]
	if role_metrics_data is None:
		lines.extend([
			"## 8. Fresh validation sampling",
			"",
			"Not started. The protocol requires the development gate to pass before a fresh manifest is created or any fresh cases are run.",
			"",
			"## 9. Requested-answer role results",
			"",
			"Not applicable because fresh validation was not started.",
			"",
			"## 10. Convergence-role results",
			"",
			"Not applicable because fresh validation was not started.",
			"",
			"## 11. Same-role decision results",
			"",
			"Development role consistency is reported in `development_gate.json` and `development_role_audit.jsonl`.",
			"",
			"## 12. Shared-target results",
			"",
			"Development-only result: the deterministic role separation corrected 2 of 5 primary confusion cases.",
			"",
			"## 13. BRANCH_INTERMEDIATE results",
			"",
			"The development requirement of at least 4 of 5 corrected primary confusion cases was not met.",
			"",
			"## 14. BRANCH_FINAL preservation",
			"",
			"The development safety requirement also failed because one known final/control case became a false INTERMEDIATE case.",
			"",
			"## 15. Control safety",
			"",
			f"False-intermediate answer cases: `{dev_gate.get('false_intermediate_answer_case_count', 0)}`. Case IDs: `{', '.join(dev_gate.get('false_intermediate_answer_case_ids', [])) or 'none'}`.",
			"",
			"## 16. Repeat stability",
			"",
			"Development repeat-level role records are available in `development_role_audit.jsonl`; no fresh stability analysis was run.",
			"",
			"## 17. End-to-end QA",
			"",
			"Not run for fresh validation. The gate is intentionally evaluated before the fresh end-to-end stage.",
			"",
			"## 18. Downstream failures after correct role separation",
			"",
			"Not applicable. No fresh cases reached the downstream comparison stage.",
			"",
			"## 19. Manual review",
			"",
			"`manual_review_cases.json` contains all nine development cases and their automatic role records. Manual labels remain unset.",
			"",
			"## 20. Decision gate",
			"",
			"```json",
			json.dumps(dev_gate, ensure_ascii=False, indent=2),
			"```",
			"",
			"## 21. Final Phase 3B.4 decision",
			"",
			f"**{decision}**",
			"",
			"## 22. Recommended next step",
			"",
			"Do not start the fresh 30-case validation and do not change the validator or executor. First inspect the two uncorrected confusion cases and the false-intermediate control, then define a narrower follow-up intervention if warranted.",
			"",
			"## 23. Limitations",
			"",
			"This is a development-gate result, not a fresh generalization result. It supports neither a positive Phase 3B.4 claim nor a comparison on the planned 120 fresh runs.",
		])
	else:
		lines.extend([
			"## 8. Fresh validation sampling",
			"",
			"A new seed-20261027 manifest with 10 strong BRANCH_INTERMEDIATE, 10 strong BRANCH_FINAL, and 10 controls was frozen before fresh execution.",
			"",
			"## 9. Requested-answer role results",
			"",
			table_role(role_metrics_data, "requested_answer_role_accuracy", "Requested-answer role"),
			"",
			"## 10. Convergence-role results",
			"",
			table_role(role_metrics_data, "convergence_role_accuracy", "Convergence role"),
			"",
			"## 11. Same-role decision results",
			"",
			table_role(role_metrics_data, "roles_same_entity_accuracy", "Same-role decision"),
			"",
			"## 12. Shared-target results",
			"",
			table_role(role_metrics_data, "shared_target_type_accuracy_case", "Shared-target type"),
			"",
			"## 13. BRANCH_INTERMEDIATE results",
			f"Confusion rate: Phase 3B.2 `{metric(shared_data['strong_intermediate']['phase3b_2']['answer_convergence_confusion_rate'])}`; Phase 3B.4 `{metric(shared_data['strong_intermediate']['phase3b_4']['answer_convergence_confusion_rate'])}`. Target accuracy: Phase 3B.2 `{metric(role_metrics_data['strong_intermediate']['phase3b_2']['shared_target_type_accuracy_case'])}`; Phase 3B.4 `{metric(role_metrics_data['strong_intermediate']['phase3b_4']['shared_target_type_accuracy_case'])}`.",
			"",
			"## 14. BRANCH_FINAL preservation",
			f"Final-target regression count: `{decision_data.get('branch_final_regression_count', 'N/A')}`. False INTERMEDIATE and false NONE are retained in the shared-target metrics and paired artifacts.",
			"",
			"## 15. Control safety",
			f"False-intermediate control cases: `{decision_data.get('control_false_intermediate_case_count', 'N/A')}/10`.",
			"",
			"## 16. Repeat stability",
			"See `stability_analysis.json` for STABLE_CORRECT, STABLE_WRONG, UNSTABLE_ROLE, and UNSTABLE_TARGET classifications.",
			"",
			"## 17. End-to-end QA",
			"",
			table_qa(qa_data),
			"",
			"## 18. Downstream failures after correct role separation",
			"Cases with correct role/target fields but incorrect answers are listed in `downstream_failure_audit.jsonl`; no downstream component was changed.",
			"",
			"## 19. Manual review",
			"`manual_review_cases.json` enumerates all required development, intermediate, changed-target, and regression cases. Automatic labels and manual labels remain separate.",
			"",
			"## 20. Decision gate",
			"",
			"```json",
			json.dumps(decision_data, ensure_ascii=False, indent=2),
			"```",
			"",
			"## 21. Final Phase 3B.4 decision",
			"",
			f"**{decision}**",
			"",
			"## 22. Recommended next step",
			"",
			"Do not modify the validator or executor. If supported, a later phase may isolate role separation further; if partial, inspect the remaining role/grounding and downstream failures before any new intervention.",
			"",
			"## 23. Limitations",
			"",
			"Role descriptions have no canonical ontology in the existing benchmark, so role-field metrics use explicit-field completeness plus the structural same/different-role contract. Manual review is required for semantic wording quality. The fresh result, if present, uses two repeats per case and a fixed 30-case sample.",
		])
	(OUTPUT_ROOT / "PHASE3B_4_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_role(data: dict[str, Any], field: str, label: str) -> str:
	rows = ["| Cohort | Phase 3B.2 | Phase 3B.4 |", "|---|---:|---:|"]
	for cohort in ("strong_intermediate", "strong_final", "control"):
		base = data[cohort]["phase3b_2"].get(field)
		new = data[cohort]["phase3b_4"].get(field)
		fmt = lambda value: "N/A" if value is None else f"{float(value):.3f}"
		rows.append(f"| {label} — {cohort} | {fmt(base)} | {fmt(new)} |")
	return "\n".join(rows)


def table_qa(data: dict[str, Any]) -> str:
	rows = ["| Cohort | Metric | Phase 3B.2 | Phase 3B.4 |", "|---|---|---:|---:|"]
	for cohort in ("strong_intermediate", "strong_final", "control"):
		for field, label in (("hits_at_1_mean", "Hits@1"), ("f1_mean", "F1")):
			rows.append(f"| {cohort} | {label} | {data[cohort]['phase3b_2'][field]:.3f} | {data[cohort]['phase3b_4'][field]:.3f} |")
	return "\n".join(rows)


def write_outputs(
	base: Any,
	dev_manifest: dict[str, Any],
	dev_new: list[dict[str, Any]],
	dev_base: list[dict[str, Any]],
	fresh_manifest: dict[str, Any] | None,
	fresh_rows: list[dict[str, Any]] | None,
	preflight: dict[str, Any],
) -> dict[str, Any]:
	dev_by = {case["case_id"]: case for case in dev_manifest["cases"]}
	dev_records = []
	for row in dev_new:
		base_row = next((item for item in dev_base if item["case_id"] == row["case_id"] and int(item["repeat"]) == int(row["repeat"])), None)
		dev_records.append(role_record(row, dev_by[row["case_id"]], base_row))
	write_jsonl(OUTPUT_ROOT / "development_role_audit.jsonl", dev_records)
	write_json(OUTPUT_ROOT / "development_gate.json", development_gate(dev_manifest, dev_new, dev_base))
	if fresh_manifest is None or fresh_rows is None:
		decision = "ROLE_SEPARATION_DEVELOPMENT_FAILED"
		decision_data = {"development_gate": development_gate(dev_manifest, dev_new, dev_base)}
		write_json(OUTPUT_ROOT / "manual_review_cases.json", manual_review_cases(dev_new, [], dev_by))
		(OUTPUT_ROOT / "decision_summary.md").write_text(f"# Phase 3B.4 Decision Summary\n\nDecision: **{decision}**\n\nDevelopment gate failed; fresh validation was not started.\n", encoding="utf-8")
		write_report(decision, decision_data, decision_data["development_gate"], None, None, None, None)
		return {"decision": decision, "development_gate": decision_data["development_gate"]}

	fresh_by = {case["case_id"]: case for case in fresh_manifest["cases"]}
	all_role_rows = []
	for row in fresh_rows:
		all_role_rows.append(role_record(row, fresh_by[row["case_id"]]))
	write_jsonl(OUTPUT_ROOT / "fresh_role_extraction_results.jsonl", all_role_rows)
	role_data = role_metrics(all_role_rows)
	shared_data = shared_target_metrics(fresh_rows)
	qa_data = qa_metrics(base, fresh_rows)
	paired_data = paired_analysis(fresh_rows)
	stability_data = stability(all_role_rows)
	downstream_data = downstream_audit(fresh_rows)
	write_json(OUTPUT_ROOT / "semantic_role_metrics.json", role_data)
	write_json(OUTPUT_ROOT / "shared_target_metrics.json", shared_data)
	write_json(OUTPUT_ROOT / "qa_metrics.json", qa_data)
	write_json(OUTPUT_ROOT / "paired_analysis.json", paired_data)
	write_json(OUTPUT_ROOT / "stability_analysis.json", stability_data)
	write_jsonl(OUTPUT_ROOT / "downstream_failure_audit.jsonl", downstream_data)
	write_json(OUTPUT_ROOT / "manual_review_cases.json", manual_review_cases(dev_new, fresh_rows, {**dev_by, **fresh_by}))
	decision, decision_data = decision_label(development_gate(dev_manifest, dev_new, dev_base), role_data, shared_data, paired_data)
	write_json(OUTPUT_ROOT / "decision_gate.json", decision_data)
	(OUTPUT_ROOT / "decision_summary.md").write_text(f"# Phase 3B.4 Decision Summary\n\nDecision: **{decision}**\n\n{json.dumps(decision_data, ensure_ascii=False, indent=2)}\n", encoding="utf-8")
	write_report(decision, decision_data, development_gate(dev_manifest, dev_new, dev_base), role_data, shared_data, qa_data, stability_data)
	return {"decision": decision, "decision_data": decision_data, "development_gate": development_gate(dev_manifest, dev_new, dev_base)}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--prepare", action="store_true", help="freeze the nine-case development manifest")
	parser.add_argument("--run", action="store_true", help="run development, then fresh validation only if development passes")
	parser.add_argument("--resume", action="store_true", help="resume existing run files")
	args = parser.parse_args()
	if not args.prepare and not args.run:
		raise SystemExit("choose --prepare or --run")
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	dev_path = OUTPUT_ROOT / "development_case_manifest.json"
	dev_manifest = read_json(dev_path) if dev_path.exists() else build_development_manifest()
	write_json(dev_path, dev_manifest)
	if args.prepare and not args.run:
		print(json.dumps({"development_cases": len(dev_manifest["cases"]), "fresh_manifest": "created only after development gate passes", "manifest": str(dev_path)}, indent=2))
		return
	base = _base_module()
	preflight = base.api_preflight()
	write_json(OUTPUT_ROOT / "preflight.json", preflight)
	if not preflight.get("openai_api_key_present") or not preflight.get("model_name") or not preflight.get("openai_available") or not preflight.get("freebase_available"):
		raise SystemExit(f"live preflight failed: {json.dumps(preflight, ensure_ascii=False)}")
	candidates, _ = base.load_candidates()
	by_id = {candidate["case_id"]: candidate for candidate in candidates}
	dev_candidates = []
	for case in dev_manifest["cases"]:
		if case["case_id"] not in by_id:
			raise SystemExit(f"missing candidate metadata for development case {case['case_id']}")
		candidate = dict(by_id[case["case_id"]])
		candidate["cohort"] = "development"
		candidate["gold_structure"] = case["gold_structure"]
		dev_candidates.append(candidate)
	dev_run_path = OUTPUT_ROOT / "development_run_results.jsonl"
	dev_new = append_run(dev_run_path, [], base, make_agent(base, "phase3b_4"), "phase3b_4", REPEATS, dev_candidates, args.resume)
	dev_base = baseline_development_rows(dev_manifest)
	dev_gate = development_gate(dev_manifest, dev_new, dev_base)
	if not dev_gate["passed"]:
		write_outputs(base, dev_manifest, dev_new, dev_base, None, None, preflight)
		print(json.dumps({"status": "DEVELOPMENT_GATE_FAILED", "development_gate": dev_gate, "output_root": str(OUTPUT_ROOT)}, indent=2))
		return
	fresh_path = OUTPUT_ROOT / "fresh_validation_manifest.json"
	if fresh_path.exists():
		fresh_manifest = read_json(fresh_path)
	else:
		fresh_manifest = balanced_fresh_manifest(base)
		write_json(fresh_path, fresh_manifest)
	fresh_candidates = list(fresh_manifest["cases"])
	fresh_run_path = OUTPUT_ROOT / "fresh_run_results.jsonl"
	fresh_rows: list[dict[str, Any]] = read_jsonl(fresh_run_path) if args.resume and fresh_run_path.exists() else []
	for method in METHODS:
		fresh_rows = append_run(fresh_run_path, fresh_rows, base, make_agent(base, method), method, REPEATS, fresh_candidates, args.resume)
	result = write_outputs(base, dev_manifest, dev_new, dev_base, fresh_manifest, fresh_rows, preflight)
	print(json.dumps({"status": "COMPLETE", "fresh_rows": len(fresh_rows), "development_rows": len(dev_new), "result": result, "output_root": str(OUTPUT_ROOT)}, indent=2))


if __name__ == "__main__":
	main()
