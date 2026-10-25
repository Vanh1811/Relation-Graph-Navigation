#!/usr/bin/env python3
"""Run the frozen Phase 2.5 Stage B fresh balanced validation."""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_phase2_5_1_live_validation as base  # noqa: E402
import run_phase2_5_3b_live as phase3b  # noqa: E402
import run_phase2_5_5_live as phase255  # noqa: E402
import run_phase2_5_6_combined_validation as phase256  # noqa: E402
from chain_of_relations.schema import Entity  # noqa: E402
from relation_graph_kgqa.datasets.schemas import QAExample  # noqa: E402
from relation_graph_kgqa.sparql.graph_builder import build_query_graph  # noqa: E402
from relation_graph_kgqa.sparql.parser import parse_query  # noqa: E402
from run_phase2_4_mixed_failure_validation import gold_structure_from_s_expression  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs/experiments/phase2_5_stage_b/fresh_balanced_validation"
MANIFEST_PATH = OUTPUT_ROOT / "selection_manifest.json"
SEED = 20261025
REPEATS = 2
STRUCTURES = ("LINEAR", "BRANCH_FINAL", "BRANCH_INTERMEDIATE")
POOL_PATHS = {
	"cwq": (ROOT / "data/processed/cwq_train_analysis.jsonl", ROOT / "data/processed/cwq_test_analysis.jsonl"),
	"grailqa": (ROOT / "data/processed/grailqa_dev_analysis.jsonl", ROOT / "data/processed/grailqa_train_analysis.jsonl"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def text(value: Any) -> str:
	return str(value or "").strip()


def canonical(value: Any) -> str:
	return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cwq_gold_structure(row: dict[str, Any]) -> str:
	example = QAExample(
		id=text(row.get("id")),
		question=text(row.get("question")),
		answers=list(row.get("answers", [])),
		gold_sparql=text(row.get("gold_sparql")),
		dataset_type="cwq",
		topic_entities=[text(item) for item in row.get("topic_entities", [])],
		split=text(row.get("split")),
	)
	features = parse_query(example.gold_sparql or "")
	graph = build_query_graph(features, example)
	if not features.parse_success:
		return "GOLD_OTHER"
	candidates = graph.branch_candidates()
	if any(variable in graph.answer_variables for variable in candidates):
		return "GOLD_BRANCH_FINAL"
	if candidates:
		return "GOLD_BRANCH_INTERMEDIATE"
	return "GOLD_LINEAR"


def gold_structure(dataset: str, row: dict[str, Any]) -> str:
	if dataset == "cwq":
		return cwq_gold_structure(row)
	return gold_structure_from_s_expression(text(row.get("logical_form")))


def is_example_id(value: Any) -> bool:
	value = text(value)
	return value.startswith("WebQ") or (value.isdigit() and len(value) >= 10)


def known_prior_ids() -> set[str]:
	"""Collect prior sample IDs and Phase 2.4/2.5 diagnostic IDs only."""
	known: set[str] = set()
	pattern = re.compile(r"(?:WebQ(?:Trn|Test)-[A-Za-z0-9_]+|\b[0-9]{10,13}\b)")
	for path in (ROOT / "outputs/experiments").rglob("sample_ids.json"):
		try:
			known.update(item for item in pattern.findall(path.read_text(encoding="utf-8")) if is_example_id(item))
		except OSError:
			continue
	for directory in list((ROOT / "outputs/experiments").glob("phase2_4*")) + list((ROOT / "outputs/experiments").glob("phase2_5_[0-9]*")):
		for path in directory.rglob("*"):
			if path.suffix not in {".json", ".jsonl"}:
				continue
			try:
				for match in pattern.findall(path.read_text(encoding="utf-8", errors="ignore")):
					if is_example_id(match):
						known.add(match)
			except OSError:
				continue
	return known


def normalize_case(dataset: str, path: Path, row: dict[str, Any], structure: str) -> dict[str, Any] | None:
	case_id = text(row.get("id"))
	if not is_example_id(case_id) or structure not in STRUCTURES:
		return None
	if not text(row.get("question")) or not row.get("answers") or not row.get("topic_entities"):
		return None
	if dataset == "cwq" and not text(row.get("gold_sparql")):
		return None
	return {
		"case_id": case_id,
		"dataset": dataset,
		"split": text(row.get("split")),
		"question": text(row.get("question")),
		"gold_answers": [text(item) for item in row.get("answers", []) if text(item)],
		"answer_ids": [text(item) for item in row.get("answer_ids", []) if text(item)],
		"gold_structure": structure,
		"gold_form": text(row.get("gold_sparql") or row.get("logical_form")),
		"gold_sparql": text(row.get("gold_sparql")),
		"logical_form": text(row.get("logical_form")),
		"topic_entities": [text(item) for item in row.get("topic_entities", []) if text(item)],
		"topic_entity_names": [text(item) for item in row.get("topic_entity_names", [])],
		"source_path": str(path),
	}


def collect_pools(known: set[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
	pools = {structure: [] for structure in STRUCTURES}
	all_counts = Counter()
	for dataset, paths in POOL_PATHS.items():
		for path in paths:
			if not path.exists():
				continue
			for row in read_jsonl(path):
				gold_label = gold_structure(dataset, row)
				structure = gold_label.replace("GOLD_", "")
				all_counts[structure] += 1
				if text(row.get("id")) in known:
					continue
				case = normalize_case(dataset, path, row, structure)
				if case:
					pools[structure].append(case)
	for structure in STRUCTURES:
		pools[structure].sort(key=lambda item: (item["dataset"], item["case_id"]))
	return pools, dict(all_counts)


def freeze_selection() -> dict[str, Any]:
	known = known_prior_ids()
	pools, all_counts = collect_pools(known)
	if any(len(pools[structure]) < 5 for structure in STRUCTURES):
		raise SystemExit(f"insufficient fresh candidates: { {key: len(value) for key, value in pools.items()} }")
	rng = random.Random(SEED)
	# Keep both datasets represented in every structure while keeping the total
	# split close to balanced (8 CWQ, 7 GrailQA).
	targets = {
		"LINEAR": {"cwq": 3, "grailqa": 2},
		"BRANCH_FINAL": {"cwq": 2, "grailqa": 3},
		"BRANCH_INTERMEDIATE": {"cwq": 3, "grailqa": 2},
	}
	selected: dict[str, list[dict[str, Any]]] = {}
	for structure in STRUCTURES:
		selected[structure] = []
		for dataset, count in targets[structure].items():
			available = [item for item in pools[structure] if item["dataset"] == dataset]
			if len(available) < count:
				raise SystemExit(f"insufficient {dataset} candidates for {structure}: need {count}, found {len(available)}")
			selected[structure].extend(rng.sample(available, count))
		selected[structure].sort(key=lambda item: item["case_id"])
	selected_ids = {item["case_id"] for items in selected.values() for item in items}
	excluded_overlap = sorted(known & {item["case_id"] for values in pools.values() for item in values})
	manifest = {
		"selection_seed": SEED,
		"repeats": REPEATS,
		"eligible_pool_size": {structure: len(pools[structure]) for structure in STRUCTURES},
		"eligible_pool_size_before_exclusion": all_counts,
		"selected_ids": {structure: [item["case_id"] for item in selected[structure]] for structure in STRUCTURES},
		"selected": selected,
		"selected_case_count": len(selected_ids),
		"dataset_counts": dict(Counter(item["dataset"] for values in selected.values() for item in values)),
		"excluded_known_prior_ids_in_pool": excluded_overlap,
		"excluded_known_prior_id_count": len(excluded_overlap),
		"known_prior_id_count": len(known),
		"selection_policy": "sorted candidate IDs, fixed seed, 5 per gold structure, both CWQ and GrailQA represented per structure",
	}
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	return manifest


def load_manifest() -> tuple[dict[str, Any], list[dict[str, Any]]]:
	if not MANIFEST_PATH.exists():
		raise SystemExit(f"missing selection manifest: {MANIFEST_PATH}; run with --select first")
	manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
	selected = [item for values in manifest.get("selected", {}).values() for item in values]
	if len(selected) != 15 or len({item["case_id"] for item in selected}) != 15:
		raise SystemExit("selection manifest must contain 15 unique cases")
	if any(len(manifest.get("selected_ids", {}).get(structure, [])) != 5 for structure in STRUCTURES):
		raise SystemExit("selection manifest must contain exactly 5 cases per structure")
	known = known_prior_ids()
	overlap = {item["case_id"] for item in selected} & known
	if overlap:
		raise SystemExit(f"selection overlaps prior experiment IDs: {sorted(overlap)}")
	return manifest, sorted(selected, key=lambda item: (STRUCTURES.index(item["gold_structure"]), item["case_id"]))



def make_cor() -> Any:
	from chain_of_relations.kg_backend import get_backend
	from chain_of_relations.methods.cor.agent import CoRAgent

	return CoRAgent(
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


def entities_for(case: dict[str, Any]) -> list[Entity]:
	ids = case.get("topic_entities", [])
	names = case.get("topic_entity_names", [])
	return [Entity(str(item), str(names[index] if index < len(names) and names[index] else item)) for index, item in enumerate(ids)]


def usage_totals(history: list[dict[str, Any]]) -> tuple[int, int, int]:
	input_tokens = sum(int(item.get("input_tokens", 0) or 0) for item in history)
	output_tokens = sum(int(item.get("output_tokens", 0) or 0) for item in history)
	return input_tokens, output_tokens, input_tokens + output_tokens


def run_combined(agent: Any, case: dict[str, Any], repeat: int) -> dict[str, Any]:
	history: list[dict[str, Any]] = []
	sparql_history: list[dict[str, Any]] = []
	step_history: list[dict[str, Any]] = []
	started = time.perf_counter()
	error = None
	try:
		result = agent.answer(
			question=case["question"],
			topic_entities=entities_for(case),
			prompt_history=history,
			sparql_history=sparql_history,
			step_history=step_history,
		)
	except Exception as exc:
		result = {"action": "ERROR", "results": []}
		error = f"{type(exc).__name__}: {exc}"
	event = next((item for item in history if item.get("type") == "phase2_5_5_canonical_normalization_router"), {})
	decision = result.get("router_decision", result) if isinstance(result, dict) else {}
	decision = decision if isinstance(decision, dict) else {}
	serialization = event.get("serialization", {}) if isinstance(event.get("serialization"), dict) else {}
	normalization = event.get("normalization", {}) if isinstance(event.get("normalization"), dict) else {}
	validator_passed = text(event.get("validator_decision")).upper() == "PASS"
	branch_routed = validator_passed and decision.get("router_execution_mode") == "branch"
	if branch_routed:
		execution = phase3b.execution_status(result, validator_passed)
	else:
		execution = "NOT_APPLICABLE_NON_EXECUTABLE_STRUCTURE" if validator_passed else "NOT_ATTEMPTED_VALIDATOR_REJECT"
	native = result.get("adaptive_native_result") if isinstance(result.get("adaptive_native_result"), dict) else result
	prediction = base.prediction_names(result)
	native_prediction = base.prediction_names(native)
	hit, f1 = base.evaluate(prediction, case["gold_answers"])
	native_hit, native_f1 = base.evaluate(native_prediction, case["gold_answers"])
	input_tokens, output_tokens, total_tokens = usage_totals(history)
	return {
		"case_id": case["case_id"], "dataset": case["dataset"], "split": case["split"], "question": case["question"],
		"gold_answers": case["gold_answers"], "gold_structure": case["gold_structure"], "gold_form": case["gold_form"],
		"repeat_id": repeat, "method": "phase2_5_6", "semantic_structure": decision.get("router_structure", "UNKNOWN"),
		"raw_extraction": event.get("parsed_result", {}), "raw_shared_target": event.get("parsed_result", {}).get("shared_target") if isinstance(event.get("parsed_result"), dict) else None,
		"raw_dependency_pattern": event.get("parsed_result", {}).get("dependency_pattern") if isinstance(event.get("parsed_result"), dict) else None,
		"canonical_semantic_representation": event.get("canonical_semantic_representation", normalization.get("canonical_proposal", {})),
		"normalization": normalization, "normalization_status": event.get("normalization_status", "NORMALIZATION_FAILED"),
		"normalization_errors": event.get("normalization_errors", []), "serialization": serialization,
		"serialized_branches": serialization.get("serialized_branches", []), "serialization_status": serialization.get("serialization_status", "FAILED"),
		"validator_result": "VALIDATOR_PASS" if validator_passed else "VALIDATOR_REJECT", "validator_reason": event.get("validation_failures", []),
		"fallback_used": bool(result.get("fallback_used")), "fallback_reason": result.get("fallback_reason"),
		"execution_status": execution, "branch_routed": branch_routed, "branch_self_sufficient": branch_routed and execution == "EXECUTION_SUCCESS",
		"native_prediction": native_prediction, "final_prediction": prediction, "native_hits_at_1": native_hit, "native_f1": native_f1,
		"hits_at_1": hit, "f1": f1, "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens,
		"llm_calls": len(history), "sparql_query_count": len(sparql_history), "latency_seconds": time.perf_counter() - started,
		"prompt_history": history, "error": error,
	}


def run_baseline(agent: Any, case: dict[str, Any], repeat: int) -> dict[str, Any]:
	history: list[dict[str, Any]] = []
	sparql_history: list[dict[str, Any]] = []
	step_history: list[dict[str, Any]] = []
	started = time.perf_counter()
	error = None
	try:
		result = agent.answer(question=case["question"], topic_entities=entities_for(case), prompt_history=history, sparql_history=sparql_history, step_history=step_history)
	except Exception as exc:
		result = {"action": "ERROR", "results": []}
		error = f"{type(exc).__name__}: {exc}"
	decision = result.get("router_decision", result) if isinstance(result, dict) else {}
	decision = decision if isinstance(decision, dict) else {}
	native = result.get("adaptive_native_result") if isinstance(result.get("adaptive_native_result"), dict) else result
	prediction = base.prediction_names(result)
	native_prediction = base.prediction_names(native)
	hit, f1 = base.evaluate(prediction, case["gold_answers"])
	native_hit, native_f1 = base.evaluate(native_prediction, case["gold_answers"])
	input_tokens, output_tokens, total_tokens = usage_totals(history)
	return {
		"case_id": case["case_id"], "dataset": case["dataset"], "split": case["split"], "question": case["question"],
		"gold_answers": case["gold_answers"], "gold_structure": case["gold_structure"], "gold_form": case["gold_form"],
		"repeat_id": repeat, "method": "v1_1_3", "semantic_structure": decision.get("router_structure", "UNKNOWN"),
		"router_execution_mode": decision.get("router_execution_mode", "cor"), "validator_result": "PASS" if decision.get("validation_passed") else "REJECT",
		"fallback_used": bool(result.get("fallback_used")), "fallback_reason": result.get("fallback_reason"),
		"execution_status": "EXECUTION_SUCCESS" if decision.get("router_execution_mode") == "branch" and not result.get("fallback_used") else "FALLBACK_OR_COR",
		"native_prediction": native_prediction, "final_prediction": prediction, "native_hits_at_1": native_hit, "native_f1": native_f1,
		"hits_at_1": hit, "f1": f1, "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens,
		"llm_calls": len(history), "sparql_query_count": len(sparql_history), "latency_seconds": time.perf_counter() - started,
		"prompt_history": history, "error": error,
	}


def stage_ref(case: dict[str, Any]) -> dict[str, Any]:
	return {
		"case_id": case["case_id"], "question": case["question"], "gold_structure": f"GOLD_{case['gold_structure']}",
		"shared_answer_gold": case["gold_structure"] == "BRANCH_FINAL", "shared_intermediate_gold": case["gold_structure"] == "BRANCH_INTERMEDIATE",
		"independent_sources_gold": 2 if case["gold_structure"] != "LINEAR" else 0,
		"routing_outcome": "FRESH_CASE", "dataset": case["dataset"],
	}


def trace_row(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
	ref = stage_ref(case)
	semantic, semantic_reasons = phase256.upstream_semantic_status(ref, row)
	primary = phase256.primary_failure(ref, row, semantic)
	return {
		"case_id": case["case_id"], "dataset": case["dataset"], "split": case["split"], "question": case["question"],
		"gold_structure": case["gold_structure"], "gold_answers": case["gold_answers"], "gold_form": case["gold_form"],
		"repeat_id": row["repeat_id"], "raw_extraction": row.get("raw_extraction", {}), "raw_extraction_status": semantic,
		"raw_extraction_reasons": semantic_reasons, "normalized_extraction": row.get("canonical_semantic_representation", {}),
		"normalization_status": row.get("normalization_status"), "normalization_errors": row.get("normalization_errors", []),
		"serialized_branch_plan": row.get("serialized_branches", []), "serialization_status": row.get("serialization_status"),
		"serialization_errors": row.get("serialization", {}).get("serialization_errors", []), "structure": row.get("semantic_structure"),
		"validator_result": "PASS" if row.get("validator_result") in {"VALIDATOR_PASS", "PASS"} else "REJECT",
		"validator_reasons": row.get("validator_reason", []), "fallback_used": bool(row.get("fallback_used")),
		"fallback_reason": row.get("fallback_reason"), "execution_status": row.get("execution_status"),
		"branch_routed": bool(row.get("branch_routed")), "branch_self_sufficient": bool(row.get("branch_self_sufficient")),
		"native_prediction": row.get("native_prediction", []), "final_prediction": row.get("final_prediction", []),
		"native_hits_at_1": row.get("native_hits_at_1", 0), "native_f1": row.get("native_f1", 0.0),
		"hits_at_1": row.get("hits_at_1", 0), "f1": row.get("f1", 0.0), "primary_failure": primary,
		"input_tokens": row.get("input_tokens", 0), "output_tokens": row.get("output_tokens", 0), "total_tokens": row.get("total_tokens", 0),
		"llm_calls": row.get("llm_calls", 0), "sparql_query_count": row.get("sparql_query_count", 0), "latency_seconds": row.get("latency_seconds", 0.0),
	}


def qa(rows: list[dict[str, Any]], key: str = "hits_at_1", f1_key: str = "f1") -> tuple[int, float]:
	by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		by_case[row["case_id"]].append(row)
	return sum(max(int(item.get(key, 0) or 0) for item in values) for values in by_case.values()), sum(max(float(item.get(f1_key, 0.0) or 0.0) for item in values) for values in by_case.values())


def group_metrics(rows: list[dict[str, Any]], field: str, value: str) -> dict[str, Any]:
	items = [row for row in rows if row.get(field) == value]
	if not items:
		return {"runs": 0}
	h1, f1 = qa(items)
	return {
		"runs": len(items), "cases": len({row["case_id"] for row in items}), "hits_at_1": h1, "f1": f1,
		"routing_accuracy": sum(row.get("semantic_structure") == row.get("gold_structure") for row in items) / len(items),
		"validator_pass": sum(row.get("validator_result") == "VALIDATOR_PASS" for row in items),
		"fallback": sum(bool(row.get("fallback_used")) for row in items),
		"native_branch_success": sum(bool(row.get("branch_self_sufficient")) for row in items),
	}


def paired_analysis(cases: list[dict[str, Any]], baseline: list[dict[str, Any]], combined: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
	base_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
	comb_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in baseline: base_by[row["case_id"]].append(row)
	for row in combined: comb_by[row["case_id"]].append(row)
	rows = []
	for case in cases:
		cid = case["case_id"]
		b = base_by[cid]; c = comb_by[cid]
		b_h1, b_f1 = qa(b); c_h1, c_f1 = qa(c)
		if b_h1 and c_h1: outcome = "BOTH_CORRECT"
		elif b_h1: outcome = "V1_1_3_ONLY"
		elif c_h1: outcome = "PHASE2_5_6_ONLY"
		else: outcome = "BOTH_WRONG"
		f1_class = "IMPROVED" if c_f1 > b_f1 else "REGRESSED" if c_f1 < b_f1 else "SAME"
		rows.append({"case_id": cid, "dataset": case["dataset"], "gold_structure": case["gold_structure"], "baseline_hits_at_1": b_h1, "combined_hits_at_1": c_h1, "baseline_f1": b_f1, "combined_f1": c_f1, "outcome": outcome, "f1_class": f1_class})
	return rows, {
		"both_correct": sum(item["outcome"] == "BOTH_CORRECT" for item in rows), "v1_1_3_only": sum(item["outcome"] == "V1_1_3_ONLY" for item in rows),
		"phase2_5_6_only": sum(item["outcome"] == "PHASE2_5_6_ONLY" for item in rows), "both_wrong": sum(item["outcome"] == "BOTH_WRONG" for item in rows),
		"paired_net": sum(item["outcome"] == "PHASE2_5_6_ONLY" for item in rows) - sum(item["outcome"] == "V1_1_3_ONLY" for item in rows),
	}


def stability(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
	by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows: by_case[row["case_id"]].append(row)
	case_labels = []
	agreement = Counter()
	for case in cases:
		values = by_case[case["case_id"]]
		fields = {
			"structure": [row.get("structure") for row in values], "shared_target": [canonical(row.get("normalized_extraction", {}).get("shared_target", {})) for row in values],
			"dependency": [row.get("normalized_extraction", {}).get("dependency_pattern") for row in values], "validator": [row.get("validator_result") for row in values],
			"execution": [row.get("execution_status") for row in values], "answer": [canonical(row.get("final_prediction", [])) for row in values],
		}
		stable = {key: len(set(value)) <= 1 for key, value in fields.items()}
		agreement.update({key: int(value) for key, value in stable.items()})
		if not stable["structure"]:
			label = "UNSTABLE_ROUTING"
		elif not stable["validator"]:
			label = "UNSTABLE_VALIDATION"
		elif not stable["execution"]:
			label = "UNSTABLE_EXECUTION"
		elif not stable["answer"]:
			label = "UNSTABLE_ANSWER"
		elif values and values[0].get("hits_at_1"):
			label = "STABLE_CORRECT"
		else:
			label = "STABLE_WRONG"
		case_labels.append({"case_id": case["case_id"], "structure": case["gold_structure"], "label": label, "agreement": stable})
	return case_labels, dict(agreement)


def write_outputs(manifest: dict[str, Any], cases: list[dict[str, Any]], baseline: list[dict[str, Any]], combined: list[dict[str, Any]], preflight: dict[str, Any]) -> dict[str, Any]:
	base_trace = [dict(row, method="v1_1_3") for row in baseline]
	combined_trace = [trace_row(case, row) for case in cases for row in combined if row["case_id"] == case["case_id"]]
	all_trace = [{**row, "method": "phase2_5_6"} for row in combined_trace] + base_trace
	paired_rows, paired_counts = paired_analysis(cases, baseline, combined)
	stability_rows, agreement = stability(cases, combined_trace)
	refs = {case["case_id"]: case for case in cases}
	structure_metrics = {structure: group_metrics(combined, "gold_structure", structure) for structure in STRUCTURES}
	dataset_metrics = {dataset: group_metrics(combined, "dataset", dataset) for dataset in ("cwq", "grailqa")}
	all_h1, all_f1 = qa(combined); base_h1, base_f1 = qa(baseline)
	false_positive_branch_cases = sum(
		case["gold_structure"] == "LINEAR" and any(row.get("branch_self_sufficient") and not row.get("hits_at_1") for row in combined if row["case_id"] == case["case_id"])
		for case in cases
	)
	linear_regressions = sum(item["gold_structure"] == "LINEAR" and item["baseline_hits_at_1"] and not item["combined_hits_at_1"] for item in paired_rows)
	branch_only = sum(item["gold_structure"] == "BRANCH_FINAL" and item["outcome"] == "PHASE2_5_6_ONLY" for item in paired_rows)
	gate = {
		"overall_paired_net": paired_counts["paired_net"], "linear_regressions": linear_regressions,
		"phase2_5_6_only_branch_final_cases": branch_only, "new_harmful_branch_executions": false_positive_branch_cases,
		"overall_final_hits_at_1": all_h1, "v1_1_3_final_hits_at_1": base_h1,
		"pass": paired_counts["paired_net"] >= 0 and linear_regressions <= 1 and branch_only >= 1 and false_positive_branch_cases <= 1 and all_h1 >= base_h1,
	}
	decision = "STAGE_B_GENERALIZATION_SUPPORTED" if gate["pass"] else "STAGE_B_GENERALIZATION_PARTIAL" if branch_only >= 1 else "STAGE_B_GENERALIZATION_NOT_SUPPORTED"
	failures = Counter()
	for row in combined_trace:
		if not row.get("hits_at_1"):
			failures[row.get("primary_failure") or "UNKNOWN"] += 1
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	(OUTPUT_ROOT / "fresh_runs.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_trace), encoding="utf-8")
	(OUTPUT_ROOT / "routing_analysis.jsonl").write_text("".join(json.dumps({"case_id": row["case_id"], "method": row["method"], "repeat_id": row["repeat_id"], "gold_structure": row["gold_structure"], "predicted_structure": row.get("semantic_structure", row.get("structure")), "validator_result": row.get("validator_result"), "fallback_used": row.get("fallback_used"), "execution_status": row.get("execution_status"), "hits_at_1": row.get("hits_at_1"), "f1": row.get("f1")}, ensure_ascii=False) + "\n" for row in all_trace), encoding="utf-8")
	(OUTPUT_ROOT / "qa_comparison.md").write_text(
		"# Stage B QA comparison\n\n"
		"| Metric | v1.1.3 | Phase 2.5.6 |\n|---|---:|---:|\n"
		f"| Overall Hits@1 | {base_h1}/15 | {all_h1}/15 |\n| Overall F1 | {base_f1:.4f} | {all_f1:.4f} |\n"
		+ "\n| Structure | v1.1.3 H@1 | Phase 2.5.6 H@1 |\n|---|---:|---:|\n"
		+ "\n".join(f"| {structure} | {group_metrics(baseline, 'gold_structure', structure).get('hits_at_1', 0)}/5 | {structure_metrics[structure].get('hits_at_1', 0)}/5 |" for structure in STRUCTURES)
		+ "\n\n| Dataset | v1.1.3 H@1 | Phase 2.5.6 H@1 |\n|---|---:|---:|\n"
		+ "\n".join(f"| {dataset} | {group_metrics(baseline, 'dataset', dataset).get('hits_at_1', 0)}/{sum(row.get('dataset') == dataset for row in cases)} | {dataset_metrics[dataset].get('hits_at_1', 0)}/{sum(row.get('dataset') == dataset for row in cases)} |" for dataset in ("cwq", "grailqa")) + "\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "stability_analysis.md").write_text(
		"# Stage B stability analysis\n\n| Field | Agreement |\n|---|---:|\n"
		+ "\n".join(f"| {key} | {agreement.get(key, 0)}/15 |" for key in ("structure", "shared_target", "dependency", "validator", "execution", "answer"))
		+ "\n\nPer-case labels:\n\n" + "\n".join(f"- `{item['case_id']}`: **{item['label']}**" for item in stability_rows) + "\n",
		encoding="utf-8",
	)
	input_tokens = sum(int(row.get("input_tokens", 0) or 0) for row in combined) + sum(int(row.get("input_tokens", 0) or 0) for row in baseline)
	output_tokens = sum(int(row.get("output_tokens", 0) or 0) for row in combined) + sum(int(row.get("output_tokens", 0) or 0) for row in baseline)
	(OUTPUT_ROOT / "cost_comparison.md").write_text(
		"# Stage B cost comparison\n\n"
		f"v1.1.3 calls: {sum(row.get('llm_calls', 0) for row in baseline)}; Phase 2.5.6 calls: {sum(row.get('llm_calls', 0) for row in combined)}.\n\n"
		f"Combined input tokens: {input_tokens}; output tokens: {output_tokens}; total: {input_tokens + output_tokens}.\n\n"
		f"Combined wall latency: {sum(float(row.get('latency_seconds', 0.0) or 0.0) for row in combined):.3f}s; v1.1.3 wall latency: {sum(float(row.get('latency_seconds', 0.0) or 0.0) for row in baseline):.3f}s.\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "failure_analysis.md").write_text(
		"# Stage B failure analysis\n\n"
		f"Phase 2.5.6 failed-run primary families: `{dict(failures)}`.\n\n"
		f"Paired outcomes: `{paired_counts}`.\n\n"
		"Fresh-case failures are recorded without tuning the frozen stack.\n",
		encoding="utf-8",
	)
	(OUTPUT_ROOT / "PHASE2_5_STAGE_B_REPORT.md").write_text(f"""# Phase 2.5 Stage B fresh balanced generalization validation

## 1. Motivation from Phase 2.5.6

Phase 2.5.6 validated the combined stack on its focused diagnostic. Stage B tests generalization on fresh cases.

## 2. Research question

Whether the frozen combined stack generalizes to unseen LINEAR, BRANCH_FINAL, and BRANCH_INTERMEDIATE cases without unsafe over-branching or LINEAR regression.

## 3. Frozen stack

`Phase 2.5 extraction → canonical normalization → branch-plan serialization → unchanged validator → unchanged execution/fallback`

No router, normalizer, serializer, validator, backend, evaluator, model, or decoding-setting changes were made during Stage B.

## 4. Fresh-case selection

Selection seed: **{manifest['selection_seed']}**. Prior IDs were excluded before selection. The exact manifest is in `selection_manifest.json`.

## 5. Structural balance

Selected exactly 15 cases: 5 LINEAR, 5 BRANCH_FINAL, 5 BRANCH_INTERMEDIATE. Dataset counts: `{manifest['dataset_counts']}`.

## 6. Dataset balance

The selected cases contain both CWQ and GrailQA in every structural group; no exact dataset equality was forced because the total sample is odd.

## 7. Run configuration

Two repeats per case were run for both frozen v1.1.3 and Phase 2.5.6, using the same current model, settings, Freebase backend, entity grounding, and evaluator.

## 8. Routing results

Phase 2.5.6 routing metrics by structure are in `routing_analysis.jsonl`; the selected gold distribution is exactly balanced.

## 9. LINEAR preservation

LINEAR metrics: `{structure_metrics['LINEAR']}`. False-positive branch cases: **{false_positive_branch_cases}**.

## 10. BRANCH_FINAL results

BRANCH_FINAL metrics: `{structure_metrics['BRANCH_FINAL']}`. Fresh Phase 2.5.6-only correct cases: **{branch_only}**.

## 11. BRANCH_INTERMEDIATE results

BRANCH_INTERMEDIATE metrics: `{structure_metrics['BRANCH_INTERMEDIATE']}`. Structure detection is reported separately from execution support; the frozen executor does not claim unsupported branch-to-merge continuation as successful native execution.

## 12. Validator results

Overall Phase 2.5.6 validator pass: {sum(row.get('validator_result') == 'VALIDATOR_PASS' for row in combined)}/{len(combined)}. Per-group records are in `routing_analysis.jsonl`.

## 13. Fallback results

Overall Phase 2.5.6 fallback: {sum(bool(row.get('fallback_used')) for row in combined)}/{len(combined)}.

## 14. Native branch execution

Native branch successes: {sum(bool(row.get('branch_self_sufficient')) for row in combined)}. Execution details are in `fresh_runs.jsonl`.

## 15. QA

- v1.1.3: Hits@1 **{base_h1}/15**, F1 **{base_f1:.4f}**
- Phase 2.5.6: Hits@1 **{all_h1}/15**, F1 **{all_f1:.4f}**

See `qa_comparison.md` for structure and dataset breakdowns.

## 16. Paired comparison

`{paired_counts}`. Per-case results are in `routing_analysis.jsonl` and `fresh_runs.jsonl`.

## 17. Stability

Agreement: `{agreement}`. Per-case repeat labels are in `stability_analysis.md`.

## 18. Cost

See `cost_comparison.md`. The two methods were run with the same model/settings; their LLM and latency totals are recorded separately.

## 19. Failure analysis

Primary Phase 2.5.6 failure families: `{dict(failures)}`. No failures were used to tune the stack during this evaluation.

## 20. Stage B gate

| Criterion | Observed | Requirement | Pass |
|---|---:|---:|:---:|
| Overall paired net | {paired_counts['paired_net']} | >= 0 | {paired_counts['paired_net'] >= 0} |
| Overall H@1 | {all_h1}/15 vs {base_h1}/15 | non-regressive | {all_h1 >= base_h1} |
| LINEAR regressions | {linear_regressions} | <= 1 | {linear_regressions <= 1} |
| Fresh BRANCH_FINAL-only correct cases | {branch_only} | >= 1 | {branch_only >= 1} |
| New harmful branch executions | {false_positive_branch_cases} | <= 1 | {false_positive_branch_cases <= 1} |

## 21. Final Stage B decision

**{decision}**

## 22. Full100 recommendation

{'Recommend a carefully frozen full100 evaluation as the next stage.' if decision == 'STAGE_B_GENERALIZATION_SUPPORTED' else 'Do not automatically run full100; Stage B evidence is not sufficient.'}

## 23. Limitations

This is a 15-case, two-repeat generalization test. It is larger than the focused diagnostic but still too small for broad benchmark claims. Branch-intermediate execution remains limited by the frozen executor contract.
""", encoding="utf-8")
	(OUTPUT_ROOT / "selection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	(OUTPUT_ROOT / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	return {"status": "COMPLETE", "decision": decision, "cases": len(cases), "repeats": REPEATS, "baseline_h1": base_h1, "combined_h1": all_h1, "paired_net": paired_counts["paired_net"], "branch_only": branch_only, "linear_regressions": linear_regressions, "output_root": str(OUTPUT_ROOT)}


def main() -> None:
	import argparse
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--select", action="store_true")
	parser.add_argument("--run", action="store_true")
	parser.add_argument("--repeats", type=int, default=REPEATS)
	args = parser.parse_args()
	if args.repeats != REPEATS:
		raise SystemExit(f"Stage B is frozen at {REPEATS} repeats")
	base.load_project_env()
	if args.select or not MANIFEST_PATH.exists():
		manifest = freeze_selection()
		print(json.dumps({"status": "SELECTION_FROZEN", "manifest": str(MANIFEST_PATH), "selected_ids": manifest["selected_ids"], "eligible_pool_size": manifest["eligible_pool_size"]}, indent=2))
		if not args.run:
			return
	if not args.run:
		print(json.dumps({"status": "SELECTION_READY_RUN_NOT_REQUESTED", "manifest": str(MANIFEST_PATH)}, indent=2))
		return
	manifest, cases = load_manifest()
	preflight = base.preflight()
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	(OUTPUT_ROOT / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	if not preflight["freebase"].get("available"):
		print(json.dumps({"status": "INCONCLUSIVE", "reason": "freebase_endpoint_unavailable", "output_root": str(OUTPUT_ROOT)}, indent=2))
		return
	from chain_of_relations.methods.adaptive_v1_1_3.agent import AdaptiveRoutingAgent

	cor_combined = phase255.make_agent()
	cor_baseline = make_cor()
	baseline_agent = AdaptiveRoutingAgent(cor_baseline, confidence_threshold=0.75, max_branches=3)
	baseline_rows: list[dict[str, Any]] = []
	combined_rows: list[dict[str, Any]] = []
	for case in cases:
		for repeat in range(1, REPEATS + 1):
			baseline_rows.append(run_baseline(baseline_agent, case, repeat))
			combined_rows.append(run_combined(cor_combined, case, repeat))
	print(json.dumps(write_outputs(manifest, cases, baseline_rows, combined_rows, preflight), indent=2))


if __name__ == "__main__":
	main()
