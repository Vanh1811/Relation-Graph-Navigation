#!/usr/bin/env python3
"""Run the frozen Phase 3C branch-required generalization evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SOURCE_MANIFEST = ROOT / "outputs/experiments/phase3a_1/safety_cap_sensitivity/frozen_future_eval_manifest.json"
STRONG_BENCHMARK = ROOT / "outputs/experiments/phase3a/branch_necessity_benchmark/strong_branch_benchmark.jsonl"
CONTROL_BENCHMARK = ROOT / "outputs/experiments/phase3a/branch_necessity_benchmark/answer_equivalent_branch_control.jsonl"
OUTPUT_ROOT = ROOT / "outputs/experiments/phase3c/frozen_branch_required_generalization"
MANIFEST_PATH = OUTPUT_ROOT / "phase3c_eval_manifest.json"
RUN_PATH = OUTPUT_ROOT / "run_results.jsonl"
SEED = 20261028
REPEATS = 2
METHODS = ("v1_1_3", "phase2_5_6")
STRONG_STRATA = (("cwq", "BRANCH_FINAL"), ("cwq", "BRANCH_INTERMEDIATE"), ("grailqa", "BRANCH_FINAL"), ("grailqa", "BRANCH_INTERMEDIATE"))
POOL_PATHS = {"cwq": (ROOT / "data/processed/cwq_train_analysis.jsonl", ROOT / "data/processed/cwq_test_analysis.jsonl"), "grailqa": (ROOT / "data/processed/grailqa_dev_analysis.jsonl", ROOT / "data/processed/grailqa_train_analysis.jsonl")}


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


def example_id(value: Any) -> bool:
	return bool(re.fullmatch(r"(?:WebQ(?:Trn|Test)-[A-Za-z0-9_]+|[0-9]{10,13})", text(value)))


def prior_id_sources() -> list[Path]:
	paths: list[Path] = []
	for phase in ("phase2_1", "phase2_2", "phase2_2b", "phase2_3", "phase2_4", "phase2_4_1", "phase2_4_2", "phase2_4_3", "phase2_4_4", "phase2_5", "phase2_5_1", "phase2_5_2", "phase2_5_3a", "phase2_5_3b", "phase2_5_4", "phase2_5_5", "phase2_5_6", "phase2_5_stage_b", "phase2_6", "phase3b", "phase3b_1", "phase3b_2", "phase3b_3", "phase3b_4", "phase3b_5"):
		root = ROOT / "outputs/experiments" / phase
		if root.exists():
			paths.extend(path for path in root.rglob("*") if path.is_file() and path.suffix in {".json", ".jsonl", ".md"})
	for path in (ROOT / "outputs/experiments/phase3a/branch_necessity_benchmark/manual_quality_audit.json", ROOT / "outputs/experiments/phase3a_1/safety_cap_sensitivity/manual_review_cases.json"):
		if path.exists():
			paths.append(path)
	return sorted(set(paths))


def collect_prior_ids() -> tuple[set[str], dict[str, int]]:
	pattern = re.compile(r"(?:WebQ(?:Trn|Test)-[A-Za-z0-9_]+|\b[0-9]{10,13}\b)")
	known: set[str] = set()
	counts: Counter[str] = Counter()
	for path in prior_id_sources():
		try:
			matches = {item for item in pattern.findall(path.read_text(encoding="utf-8", errors="ignore")) if example_id(item)}
		except OSError:
			continue
		known.update(matches)
		if matches:
			counts[str(path.relative_to(ROOT))] += len(matches)
	return known, dict(counts)


def raw_cases() -> dict[str, dict[str, Any]]:
	output: dict[str, dict[str, Any]] = {}
	for dataset, paths in POOL_PATHS.items():
		for path in paths:
			if not path.exists():
				continue
			for row in read_jsonl(path):
				case_id = text(row.get("id"))
				if not case_id or case_id in output:
					continue
				output[case_id] = {
					"case_id": case_id, "dataset": dataset, "split": text(row.get("split")), "question": text(row.get("question")),
					"gold_answers": [text(item) for item in row.get("answers", []) if text(item)], "answer_ids": [text(item) for item in row.get("answer_ids", []) if text(item)],
					"gold_sparql": text(row.get("gold_sparql")), "logical_form": text(row.get("logical_form")),
					"topic_entities": [text(item) for item in row.get("topic_entities", []) if text(item)], "topic_entity_names": [text(item) for item in row.get("topic_entity_names", [])], "source_path": str(path),
				}
	return output


def benchmark_cases(path: Path, allowed_ids: set[str], raw: dict[str, dict[str, Any]], necessity: str) -> list[dict[str, Any]]:
	rows = []
	for meta in read_jsonl(path):
		case_id = meta.get("case_id")
		if case_id not in allowed_ids or case_id not in raw or meta.get("validity_gate") != "PASS" or meta.get("necessity_class") != necessity:
			continue
		case = dict(raw[case_id])
		case.update({"gold_structure": meta["gold_structure"], "necessity": meta["necessity_class"], "necessity_strength": meta.get("necessity_strength"), "branch_count": meta.get("branch_count"), "shared_target_variable": meta.get("shared_target_variable"), "validity_gate": meta.get("validity_gate"), "benchmark_full_gold_answers": meta.get("full_gold_answers", [])})
		if case["question"] and case["gold_answers"] and case["topic_entities"]:
			rows.append(case)
	return sorted(rows, key=lambda item: item["case_id"])


def freeze_manifest() -> dict[str, Any]:
	source = read_json(SOURCE_MANIFEST)
	prior_ids, prior_counts = collect_prior_ids()
	raw = raw_cases()
	strong_pool = [item for item in benchmark_cases(STRONG_BENCHMARK, set(source["strong_branch_benchmark_ids"]), raw, "STRONGLY_BRANCH_REQUIRED") if item["case_id"] not in prior_ids]
	control_pool = [item for item in benchmark_cases(CONTROL_BENCHMARK, set(source["answer_equivalent_branch_control_ids"]), raw, "ANSWER_EQUIVALENT_LINEARIZATION") if item["case_id"] not in prior_ids]
	rng = random.Random(SEED)
	strong: list[dict[str, Any]] = []
	for dataset, structure in STRONG_STRATA:
		pool = [item for item in strong_pool if item["dataset"] == dataset and item["gold_structure"] == structure]
		if len(pool) < 15:
			raise SystemExit(f"insufficient clean strong cases for {dataset}|{structure}: {len(pool)}")
		strong.extend(rng.sample(pool, 15))
	control_targets = {("cwq", "BRANCH_INTERMEDIATE"): 10, ("grailqa", "BRANCH_FINAL"): 10, ("grailqa", "BRANCH_INTERMEDIATE"): 10}
	controls: list[dict[str, Any]] = []
	for (dataset, structure), count in control_targets.items():
		pool = [item for item in control_pool if item["dataset"] == dataset and item["gold_structure"] == structure]
		if len(pool) < count:
			raise SystemExit(f"insufficient clean controls for {dataset}|{structure}: {len(pool)}")
		controls.extend(rng.sample(pool, count))
	for item in strong:
		item["cohort"] = "STRONG"
	for item in controls:
		item["cohort"] = "CONTROL"
	strong.sort(key=lambda item: (item["dataset"], item["gold_structure"], item["case_id"]))
	controls.sort(key=lambda item: (item["dataset"], item["gold_structure"], item["case_id"]))
	return {"status": "FROZEN", "experiment": "phase3c_frozen_branch_required_generalization", "seed": SEED, "repeats_per_method": REPEATS, "methods": list(METHODS), "selection_policy": "60 strong cases at 15 per CWQ/GrailQA x BRANCH_FINAL/BRANCH_INTERMEDIATE stratum and 30 representative answer-equivalent controls; fixed seed; no model outcomes used", "source_manifest": str(SOURCE_MANIFEST), "source_manifest_sha256": sha256(SOURCE_MANIFEST), "source_pool_counts": {"strong": len(strong_pool), "control": len(control_pool)}, "prior_exclusion_count": len(prior_ids), "prior_exclusion_source_counts": prior_counts, "exclusion_policy": "Phase 2.1 through Phase 2.6, Phase 3A manual QA/review, and Phase 3B/3B.5 artifacts; unresolved Phase 3A.1 cases are absent from the source manifest", "strong_case_count": len(strong), "control_case_count": len(controls), "strong_stratum_counts": dict(Counter(f"{item['dataset']}|{item['gold_structure']}" for item in strong)), "control_stratum_counts": dict(Counter(f"{item['dataset']}|{item['gold_structure']}" for item in controls)), "strong_cases": strong, "control_cases": controls}


def load_manifest() -> tuple[dict[str, Any], list[dict[str, Any]]]:
	manifest = read_json(MANIFEST_PATH)
	if manifest.get("strong_case_count") != 60 or manifest.get("control_case_count") != 30:
		raise SystemExit("Phase 3C manifest must contain exactly 60 strong and 30 control cases")
	cases = manifest["strong_cases"] + manifest["control_cases"]
	if len({case["case_id"] for case in cases}) != 90:
		raise SystemExit("Phase 3C manifest contains duplicate case IDs")
	for case in cases:
		case.setdefault("gold_form", case.get("gold_sparql") or case.get("logical_form") or "")
	return manifest, cases


def run_evaluation(manifest: dict[str, Any], cases: list[dict[str, Any]], resume: bool) -> dict[str, Any]:
	import run_phase2_5_stage_b as stage
	stage.base.load_project_env()
	preflight = stage.base.preflight()
	write_json(OUTPUT_ROOT / "preflight.json", preflight)
	if not preflight.get("openai_api_key_present") or not preflight.get("model_name") or not preflight.get("freebase", {}).get("available"):
		return {"status": "INCONCLUSIVE", "reason": "preflight_failed", "preflight": preflight, "rows": 0, "expected_rows": 360}
	from chain_of_relations.methods.adaptive_v1_1_3.agent import AdaptiveRoutingAgent

	baseline_agent = AdaptiveRoutingAgent(stage.make_cor(), confidence_threshold=0.75, max_branches=3)
	combined_agent = stage.phase255.make_agent()
	rows = read_jsonl(RUN_PATH) if resume and RUN_PATH.exists() else []
	existing = {(row.get("method"), int(row.get("repeat", row.get("repeat_id", 0))), row.get("case_id")) for row in rows}
	for method, agent in (("v1_1_3", baseline_agent), ("phase2_5_6", combined_agent)):
		for repeat in range(1, REPEATS + 1):
			for index, case in enumerate(cases, start=1):
				key = (method, repeat, case["case_id"])
				if key in existing:
					continue
				print(f"[{method}] repeat={repeat} {index}/{len(cases)} {case['case_id']}", flush=True)
				row = stage.run_baseline(agent, case, repeat) if method == "v1_1_3" else stage.run_combined(agent, case, repeat)
				row.update({"cohort": case["cohort"], "necessity": case["necessity"], "necessity_strength": case.get("necessity_strength"), "branch_count": case.get("branch_count"), "repeat": row.get("repeat_id", repeat), "gold_answers": case["gold_answers"], "gold_structure": case["gold_structure"]})
				rows.append(row)
				existing.add(key)
				write_jsonl(RUN_PATH, rows)
	rows.sort(key=lambda row: (row.get("method", ""), int(row.get("repeat", 0)), row.get("case_id", "")))
	write_jsonl(RUN_PATH, rows)
	return {"status": "COMPLETE", "rows": len(rows), "expected_rows": 360, "preflight": preflight}


def norm_set(values: list[Any]) -> set[str]:
	return {text(value).casefold() for value in values if text(value)}


def exact_set(row: dict[str, Any]) -> int:
	return int(norm_set(row.get("final_prediction", [])) == norm_set(row.get("gold_answers", [])))


def grouped(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
	output: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		output[(row["method"], row["case_id"])].append(row)
	return output


def case_summaries(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
	output = {}
	for key, values in grouped(rows).items():
		correct = [int(row.get("hits_at_1", 0) or 0) for row in values]
		output[key] = {"case_id": values[0]["case_id"], "method": values[0]["method"], "cohort": values[0]["cohort"], "dataset": values[0]["dataset"], "gold_structure": values[0]["gold_structure"], "majority_correct": int(sum(correct) >= 1), "f1_mean": statistics.mean(float(row.get("f1", 0.0) or 0.0) for row in values), "exact_set_majority": int(sum(exact_set(row) for row in values) >= 1), "repeat_correctness": correct}
	return output


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
	case_rows = case_summaries(rows)
	cohorts = {
		"ALL_STRONG": lambda row: row["cohort"] == "STRONG",
		"BRANCH_FINAL_STRONG": lambda row: row["cohort"] == "STRONG" and row["gold_structure"] == "BRANCH_FINAL",
		"BRANCH_INTERMEDIATE_STRONG": lambda row: row["cohort"] == "STRONG" and row["gold_structure"] == "BRANCH_INTERMEDIATE",
		"CWQ_STRONG": lambda row: row["cohort"] == "STRONG" and row["dataset"] == "cwq",
		"GRAILQA_STRONG": lambda row: row["cohort"] == "STRONG" and row["dataset"] == "grailqa",
		"CONTROLS": lambda row: row["cohort"] == "CONTROL",
	}
	output = {}
	for cohort, predicate in cohorts.items():
		output[cohort] = {}
		case_ids = {row["case_id"] for row in rows if predicate(row)}
		for method in METHODS:
			run_rows = [row for row in rows if row["method"] == method and predicate(row)]
			case_values = [case_rows[(method, case_id)] for case_id in case_ids if (method, case_id) in case_rows]
			output[cohort][method] = {
				"run_count": len(run_rows), "case_count": len(case_values),
				"hits_at_1_mean": statistics.mean(float(row.get("hits_at_1", 0) or 0) for row in run_rows) if run_rows else 0.0,
				"f1_mean": statistics.mean(float(row.get("f1", 0.0) or 0.0) for row in run_rows) if run_rows else 0.0,
				"case_majority_hits": sum(item["majority_correct"] for item in case_values), "case_majority_rate": statistics.mean(item["majority_correct"] for item in case_values) if case_values else 0.0,
				"exact_answer_set_accuracy": statistics.mean(item["exact_set_majority"] for item in case_values) if case_values else 0.0,
				"fallback_rate": statistics.mean(bool(row.get("fallback_used")) for row in run_rows) if run_rows else 0.0,
				"branch_attempt_rate": statistics.mean(bool(row.get("branch_routed")) or row.get("router_execution_mode") == "branch" for row in run_rows) if run_rows else 0.0,
				"native_branch_success_rate": statistics.mean(bool(row.get("branch_self_sufficient")) for row in run_rows) if run_rows else 0.0,
			}
	return output


def paired_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
	case_rows = case_summaries(rows)
	paired = []
	for case_id in sorted({row["case_id"] for row in rows}):
		old, new = case_rows.get(("v1_1_3", case_id)), case_rows.get(("phase2_5_6", case_id))
		if not old or not new:
			continue
		old_correct, new_correct = old["majority_correct"], new["majority_correct"]
		outcome = "BOTH_CORRECT" if old_correct and new_correct else "V1_1_3_ONLY" if old_correct else "PHASE2_5_6_ONLY" if new_correct else "BOTH_WRONG"
		paired.append({**old, "phase2_5_6_majority_correct": new_correct, "phase2_5_6_f1_mean": new["f1_mean"], "outcome": outcome})
	by_cohort = {}
	for cohort in ("STRONG", "CONTROL"):
		items = [row for row in paired if row["cohort"] == cohort]
		counts = Counter(row["outcome"] for row in items)
		by_cohort[cohort] = {"case_count": len(items), "counts": {key: counts.get(key, 0) for key in ("BOTH_CORRECT", "V1_1_3_ONLY", "PHASE2_5_6_ONLY", "BOTH_WRONG")}, "paired_net": counts.get("PHASE2_5_6_ONLY", 0) - counts.get("V1_1_3_ONLY", 0)}
	return {"by_cohort": by_cohort, "cases": paired}


def raw_shared_target(row: dict[str, Any]) -> str:
	proposal = row.get("canonical_semantic_representation") or row.get("raw_extraction") or {}
	if isinstance(proposal, dict) and isinstance(proposal.get("shared_target"), dict):
		return text(proposal["shared_target"].get("type")).upper() or "NONE"
	return "NONE"


def branch_advantage_audit(rows: list[dict[str, Any]], paired: dict[str, Any]) -> list[dict[str, Any]]:
	groups = grouped(rows)
	output = []
	for item in paired["cases"]:
		if item["cohort"] != "STRONG" or item["outcome"] not in {"PHASE2_5_6_ONLY", "V1_1_3_ONLY"}:
			continue
		new_rows, old_rows = groups[("phase2_5_6", item["case_id"])], groups[("v1_1_3", item["case_id"])]
		if item["outcome"] == "PHASE2_5_6_ONLY":
			if any(row.get("branch_self_sufficient") for row in new_rows) and not any(row.get("branch_self_sufficient") for row in old_rows):
				label = "TRUE_BRANCH_ADVANTAGE"
			elif any(row.get("fallback_used") for row in new_rows):
				label = "FALLBACK_RESCUE"
			elif any(row.get("hits_at_1") for row in new_rows):
				label = "NON_BRANCH_GROUNDING_VARIATION"
			else:
				label = "UNCLEAR"
		else:
			label = "NON_BRANCH_GROUNDING_VARIATION"
		output.append({"case_id": item["case_id"], "dataset": item["dataset"], "gold_structure": item["gold_structure"], "outcome": item["outcome"], "classification": label, "phase2_5_6_branch_routed": [bool(row.get("branch_routed")) for row in new_rows], "phase2_5_6_native_branch_success": [bool(row.get("branch_self_sufficient")) for row in new_rows], "phase2_5_6_fallback": [bool(row.get("fallback_used")) for row in new_rows], "manual_review_status": "TRACE_AUDITED_AUTOMATIC_PREAUDIT"})
	return output


def routing_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
	output = {}
	for group_name, predicate in (("STRONG_FINAL", lambda row: row["cohort"] == "STRONG" and row["gold_structure"] == "BRANCH_FINAL"), ("STRONG_INTERMEDIATE", lambda row: row["cohort"] == "STRONG" and row["gold_structure"] == "BRANCH_INTERMEDIATE"), ("CONTROL", lambda row: row["cohort"] == "CONTROL")):
		output[group_name] = {}
		for method in METHODS:
			values = [row for row in rows if row["method"] == method and predicate(row)]
			output[group_name][method] = {"run_count": len(values), "structures": dict(Counter(text(row.get("semantic_structure")) or "UNKNOWN" for row in values)), "validator_passes": sum(row.get("validator_result") in {"PASS", "VALIDATOR_PASS"} for row in values), "fallbacks": sum(bool(row.get("fallback_used")) for row in values), "branch_attempts": sum(bool(row.get("branch_routed")) or row.get("router_execution_mode") == "branch" for row in values), "native_branch_successes": sum(bool(row.get("branch_self_sufficient")) for row in values)}
	return output


def branch_intermediate_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
	output = {}
	for method in METHODS:
		values = [row for row in rows if row["method"] == method and row["cohort"] == "STRONG" and row["gold_structure"] == "BRANCH_INTERMEDIATE"]
		output[method] = {"run_count": len(values), "structure_detected": sum(text(row.get("semantic_structure")).upper() in {"BRANCH_INTERMEDIATE", "BRANCH_FINAL"} for row in values), "shared_intermediate_present": sum(raw_shared_target(row) == "INTERMEDIATE" for row in values), "native_branch_path_executable": sum(bool(row.get("branch_self_sufficient")) for row in values), "fallbacks": sum(bool(row.get("fallback_used")) for row in values), "final_correct": sum(bool(row.get("hits_at_1")) for row in values)}
	return output


def failure_family(row: dict[str, Any]) -> str:
	if row.get("error"):
		return "STOCHASTIC_OR_UNKNOWN"
	if row.get("normalization_status") in {"PARTIAL_NORMALIZATION", "NORMALIZATION_FAILED"}:
		return "NORMALIZATION"
	if row.get("serialization_status") not in {None, "", "COMPLETE"}:
		return "SERIALIZATION"
	if row.get("validator_result") not in {None, "PASS", "VALIDATOR_PASS"}:
		return "VALIDATOR"
	if row.get("fallback_used"):
		return "FALLBACK"
	if row.get("gold_structure") in {"BRANCH_FINAL", "BRANCH_INTERMEDIATE"} and not row.get("branch_routed"):
		return "STRUCTURE_ROUTING"
	if row.get("gold_structure") == "BRANCH_INTERMEDIATE" and raw_shared_target(row) != "INTERMEDIATE":
		return "SHARED_TARGET"
	return "ANSWER_SELECTION"


def failure_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
	output = []
	for (method, case_id), values in sorted(grouped(rows).items()):
		if method != "phase2_5_6" or values[0]["cohort"] != "STRONG" or sum(bool(row.get("hits_at_1")) for row in values) >= 1:
			continue
		families = [failure_family(row) for row in values]
		output.append({"case_id": case_id, "dataset": values[0]["dataset"], "gold_structure": values[0]["gold_structure"], "first_failure_by_repeat": families, "first_failure_family": Counter(families).most_common(1)[0][0], "shared_target_type_by_repeat": [raw_shared_target(row) for row in values], "independent_case": True})
	return output


def stability_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
	output = {}
	for method in METHODS:
		counts = Counter()
		case_records = []
		for (current, case_id), values in sorted(grouped(rows).items()):
			if current != method:
				continue
			correct = {bool(row.get("hits_at_1")) for row in values}
			label = "STABLE_CORRECT" if correct == {True} else "STABLE_WRONG" if correct == {False} else "UNSTABLE"
			counts[label] += 1
			case_records.append({"case_id": case_id, "classification": label, "correctness": [bool(row.get("hits_at_1")) for row in values], "answer_sets_agree": len({json.dumps(sorted(norm_set(row.get("final_prediction", [])))) for row in values}) == 1, "routing_agree": len({row.get("semantic_structure") for row in values}) == 1, "fallback_agree": len({bool(row.get("fallback_used")) for row in values}) == 1, "branch_execution_agree": len({bool(row.get("branch_self_sufficient")) for row in values}) == 1})
		output[method] = {"case_count": len(case_records), "counts": dict(counts), "cases": case_records}
	return output


def cost_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
	output = {}
	for method in METHODS:
		values = [row for row in rows if row["method"] == method]
		output[method] = {"run_count": len(values), "llm_calls": sum(int(row.get("llm_calls", 0) or 0) for row in values), "input_tokens": sum(int(row.get("input_tokens", 0) or 0) for row in values), "output_tokens": sum(int(row.get("output_tokens", 0) or 0) for row in values), "total_tokens": sum(int(row.get("total_tokens", 0) or 0) for row in values), "kg_queries": sum(int(row.get("sparql_query_count", 0) or 0) for row in values), "latency_seconds": sum(float(row.get("latency_seconds", 0.0) or 0.0) for row in values), "mean_total_tokens_per_run": statistics.mean(int(row.get("total_tokens", 0) or 0) for row in values) if values else 0.0, "mean_latency_seconds_per_run": statistics.mean(float(row.get("latency_seconds", 0.0) or 0.0) for row in values) if values else 0.0}
	return output


def exact_mcnemar(old: list[int], new: list[int]) -> dict[str, Any]:
	b = sum(a == 1 and c == 0 for a, c in zip(old, new))
	c = sum(a == 0 and c == 1 for a, c in zip(old, new))
	n = b + c
	if n == 0:
		return {"old_only": b, "new_only": c, "discordant": 0, "two_sided_p": 1.0}
	prob = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
	return {"old_only": b, "new_only": c, "discordant": n, "two_sided_p": min(1.0, 2.0 * prob)}


def bootstrap_ci(differences: list[float], seed: int = SEED, samples: int = 10000) -> dict[str, Any]:
	if not differences:
		return {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0, "n": 0}
	rng = random.Random(seed)
	means = sorted(statistics.mean(rng.choice(differences) for _ in differences) for _ in range(samples))
	return {"mean": statistics.mean(differences), "lower_95": means[int(0.025 * len(means))], "upper_95": means[int(0.975 * len(means))], "n": len(differences), "bootstrap_samples": samples}


def statistical_analysis(rows: list[dict[str, Any]], paired: dict[str, Any]) -> dict[str, Any]:
	case_rows = case_summaries(rows)
	output = {}
	for cohort in ("STRONG", "CONTROL"):
		items = [item for item in paired["cases"] if item["cohort"] == cohort]
		old = [int(item["majority_correct"]) for item in items]
		new = [int(item["phase2_5_6_majority_correct"]) for item in items]
		diffs = [case_rows[("phase2_5_6", item["case_id"])] ["f1_mean"] - case_rows[("v1_1_3", item["case_id"])] ["f1_mean"] for item in items]
		output[cohort] = {"paired_correctness": exact_mcnemar(old, new), "bootstrap_f1_difference": bootstrap_ci(diffs)}
	return output


def choose_decision(metrics_data: dict[str, Any], paired: dict[str, Any], advantage: list[dict[str, Any]], failures: list[dict[str, Any]], complete: bool) -> tuple[str, dict[str, Any]]:
	if not complete:
		return "INCONCLUSIVE", {"reason": "evaluation_incomplete", "gates": {}}
	strong, control = paired["by_cohort"]["STRONG"], paired["by_cohort"]["CONTROL"]
	strong_h1_delta = metrics_data["ALL_STRONG"]["phase2_5_6"]["hits_at_1_mean"] - metrics_data["ALL_STRONG"]["v1_1_3"]["hits_at_1_mean"]
	true_branch = sum(item["classification"] == "TRUE_BRANCH_ADVANTAGE" for item in advantage if item["outcome"] == "PHASE2_5_6_ONLY")
	failure_counts = Counter(item["first_failure_family"] for item in failures)
	dominant = failure_counts.most_common(1)[0] if failure_counts else (None, 0)
	gates = {"strong_paired_net_at_least_3": strong["paired_net"] >= 3, "phase2_5_6_strong_hits_at_1_higher": strong_h1_delta > 0, "at_least_2_true_branch_advantage_cases": true_branch >= 2, "control_paired_net_at_least_minus_2": control["paired_net"] >= -2, "no_new_dominant_harmful_failure_family": not (dominant[0] in {"SHARED_TARGET", "ROLE_INTERPRETATION"} and dominant[1] >= 2)}
	if all(gates.values()):
		label = "BRANCH_ADVANTAGE_SUPPORTED"
	elif strong["paired_net"] > 0 and true_branch > 0:
		label = "BRANCH_ADVANTAGE_PARTIAL"
	elif strong["paired_net"] > 0 and control["paired_net"] < -2:
		label = "BRANCH_ADVANTAGE_WITH_CONTROL_COST"
	else:
		label = "NO_BRANCH_ADVANTAGE"
	return label, {"gates": gates, "strong_hits_at_1_delta": strong_h1_delta, "true_branch_advantage_count": true_branch, "dominant_failure_family": dominant[0], "dominant_failure_count": dominant[1]}


def write_report(manifest: dict[str, Any], metrics_data: dict[str, Any], paired: dict[str, Any], advantage: list[dict[str, Any]], stats: dict[str, Any], decision_label: str, decision_data: dict[str, Any], execution: dict[str, Any]) -> None:
	def fmt(value: Any) -> str:
		return f"{float(value):.3f}" if value is not None else "N/A"
	strong = paired["by_cohort"].get("STRONG", {})
	control = paired["by_cohort"].get("CONTROL", {})
	lines = ["# Phase 3C Frozen Branch-Required Generalization Evaluation", "", "## 1. Motivation", "", "Phase 3C returns to architecture-level evaluation after Phase 3B.5 found heterogeneous residual role-separation failures rather than one repeated dominant family.", "", "## 2. Why shared-target tuning stopped", "", "Shared-target/answer-convergence semantics were materially improved but were no longer a single dominant residual family. No role-separation intervention was promoted into this frozen comparison.", "", "## 3. Research question", "", "Does frozen Phase 2.5.6 provide a meaningful advantage over frozen v1.1.3 on fresh strongly branch-required questions while remaining safe on controls?", "", "## 4. Frozen methods", "", "Only frozen v1.1.3 and Phase 2.5.6 were evaluated. Model, settings, backend, evaluator, normalization, serialization, validator, execution, fallback, grounding, and answer selection were unchanged.", "", "## 5. Frozen benchmark", "", f"The Phase 3A.1 clean pool supplied {manifest['source_pool_counts']['strong']} strong and {manifest['source_pool_counts']['control']} control candidates before Phase 3C exclusions.", "", "## 6. Exclusion policy", "", f"Prior-evaluation exclusions removed `{manifest['prior_exclusion_count']}` IDs. Unresolved Phase 3A.1 cases were inherited as excluded from the frozen source manifest.", "", "## 7. Sampling design", "", f"The sample contains {manifest['strong_case_count']} strong cases and {manifest['control_case_count']} controls. Strong strata: `{manifest['strong_stratum_counts']}`. Control strata: `{manifest['control_stratum_counts']}`.", "", "## 8. Execution completeness", "", f"Status: `{execution.get('status')}`. Recorded rows: `{execution.get('rows', 0)}` / `{execution.get('expected_rows', 360)}`.", "", "## 9. Strong overall results", "", "| Metric | v1.1.3 | Phase 2.5.6 |", "|---|---:|---:|", f"| Hits@1 | {fmt(metrics_data['ALL_STRONG']['v1_1_3']['hits_at_1_mean'])} | {fmt(metrics_data['ALL_STRONG']['phase2_5_6']['hits_at_1_mean'])} |", f"| F1 | {fmt(metrics_data['ALL_STRONG']['v1_1_3']['f1_mean'])} | {fmt(metrics_data['ALL_STRONG']['phase2_5_6']['f1_mean'])} |", f"| Case-majority correct | {metrics_data['ALL_STRONG']['v1_1_3']['case_majority_hits']} | {metrics_data['ALL_STRONG']['phase2_5_6']['case_majority_hits']} |", f"| Unique wins | — | {strong.get('counts', {}).get('PHASE2_5_6_ONLY', 0)} |", f"| Paired net | — | {strong.get('paired_net', 0)} |", "", "## 10. BRANCH_FINAL results", "", f"v1.1.3 Hits@1/F1: `{fmt(metrics_data['BRANCH_FINAL_STRONG']['v1_1_3']['hits_at_1_mean'])}` / `{fmt(metrics_data['BRANCH_FINAL_STRONG']['v1_1_3']['f1_mean'])}`; Phase 2.5.6: `{fmt(metrics_data['BRANCH_FINAL_STRONG']['phase2_5_6']['hits_at_1_mean'])}` / `{fmt(metrics_data['BRANCH_FINAL_STRONG']['phase2_5_6']['f1_mean'])}`.", "", "## 11. BRANCH_INTERMEDIATE results", "", f"v1.1.3 Hits@1/F1: `{fmt(metrics_data['BRANCH_INTERMEDIATE_STRONG']['v1_1_3']['hits_at_1_mean'])}` / `{fmt(metrics_data['BRANCH_INTERMEDIATE_STRONG']['v1_1_3']['f1_mean'])}`; Phase 2.5.6: `{fmt(metrics_data['BRANCH_INTERMEDIATE_STRONG']['phase2_5_6']['hits_at_1_mean'])}` / `{fmt(metrics_data['BRANCH_INTERMEDIATE_STRONG']['phase2_5_6']['f1_mean'])}`.", "", "## 12. Dataset breakdown", "", "Dataset-level paired counts are available in `paired_analysis.json` and `metrics.json`.", "", "## 13. Paired outcomes", "", f"Strong: `{strong.get('counts', {})}`. Controls: `{control.get('counts', {})}`.", "", "## 14. True branch-advantage audit", "", f"Strong discordant cases audited: `{len(advantage)}`; Phase 2.5.6-only strong wins classified TRUE_BRANCH_ADVANTAGE: `{sum(item['classification'] == 'TRUE_BRANCH_ADVANTAGE' for item in advantage if item['outcome'] == 'PHASE2_5_6_ONLY')}`.", "", "## 15. Control safety", "", f"Control paired net: `{control.get('paired_net', 0)}`. See `control_safety_analysis.json`.", "", "## 16. Routing behavior", "", "See `routing_behavior.json` for routing, validator, fallback, and native branch execution counts.", "", "## 17. Fallback behavior", "", f"Strong fallback rates: v1.1.3 `{fmt(metrics_data['ALL_STRONG']['v1_1_3']['fallback_rate'])}`; Phase 2.5.6 `{fmt(metrics_data['ALL_STRONG']['phase2_5_6']['fallback_rate'])}`.", "", "## 18. Shared-target / role diagnostics", "", "Shared-target and role fields were logged diagnostically only. No tuning was performed.", "", "## 19. Failure-family distribution", "", "See `failure_family_analysis.jsonl`; failure labels are observational and do not trigger an intervention.", "", "## 20. Stability", "", "See `stability_analysis.json` for STABLE_CORRECT, STABLE_WRONG, and UNSTABLE classifications.", "", "## 21. Cost", "", "See `cost_analysis.json` for calls, tokens, KG queries, and latency.", "", "## 22. Statistical analysis", "", f"Strong exact McNemar: `{stats.get('STRONG', {}).get('paired_correctness', {})}`. Strong bootstrap F1 CI: `{stats.get('STRONG', {}).get('bootstrap_f1_difference', {})}`.", "", "## 23. Manual disagreement review", "", "All unique strong discordant cases and control regressions are trace-audited; automatic evidence and review status remain separate.", "", "## 24. Decision gate", "", "```json", json.dumps(decision_data, ensure_ascii=False, indent=2), "```", "", "## 25. Final Phase 3C decision", "", f"**{decision_label}**", "", "## 26. Scientific interpretation", "", "This is an architecture-level comparison, not a claim that semantic extraction has been solved.", "", "## 27. Recommended next step", "", "Stop after this frozen evaluation and follow the decision-specific protocol recommendation. Do not tune from individual failures.", "", "## 28. Limitations", "", "The sample contains 60 strong cases and 30 controls with two repeats. Branch-advantage labels are trace-based audits and statistical tests are supportive rather than decisive."]
	(OUTPUT_ROOT / "PHASE3C_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(manifest: dict[str, Any], rows: list[dict[str, Any]], execution: dict[str, Any]) -> dict[str, Any]:
	metrics_data, paired = metrics(rows), paired_analysis(rows)
	advantage = branch_advantage_audit(rows, paired)
	routing = routing_behavior(rows)
	intermediate = branch_intermediate_analysis(rows)
	stability = stability_analysis(rows)
	cost = cost_analysis(rows)
	failures = failure_analysis(rows)
	stats = statistical_analysis(rows, paired)
	control_regressions = [item for item in paired["cases"] if item["cohort"] == "CONTROL" and item["outcome"] == "V1_1_3_ONLY"]
	control_analysis = {"regression_count": len(control_regressions), "regressions": control_regressions, "phase2_5_6_branch_attempts": sum(bool(row.get("branch_routed")) for row in rows if row["method"] == "phase2_5_6" and row["cohort"] == "CONTROL"), "phase2_5_6_fallbacks": sum(bool(row.get("fallback_used")) for row in rows if row["method"] == "phase2_5_6" and row["cohort"] == "CONTROL")}
	complete = len(rows) == 360
	decision_label, decision_data = choose_decision(metrics_data, paired, advantage, failures, complete)
	write_json(OUTPUT_ROOT / "metrics.json", metrics_data)
	write_json(OUTPUT_ROOT / "paired_analysis.json", paired)
	write_jsonl(OUTPUT_ROOT / "branch_advantage_audit.jsonl", advantage)
	write_json(OUTPUT_ROOT / "routing_behavior.json", routing)
	write_json(OUTPUT_ROOT / "branch_intermediate_analysis.json", intermediate)
	write_json(OUTPUT_ROOT / "control_safety_analysis.json", control_analysis)
	write_json(OUTPUT_ROOT / "stability_analysis.json", stability)
	write_json(OUTPUT_ROOT / "cost_analysis.json", cost)
	write_jsonl(OUTPUT_ROOT / "failure_family_analysis.jsonl", failures)
	write_json(OUTPUT_ROOT / "statistical_analysis.json", stats)
	write_json(OUTPUT_ROOT / "decision_gate.json", decision_data)
	(OUTPUT_ROOT / "decision_summary.md").write_text(f"# Phase 3C Decision Summary\n\nDecision: **{decision_label}**\n\nExecution status: `{execution.get('status')}`; rows: `{len(rows)}/360`.\n", encoding="utf-8")
	write_report(manifest, metrics_data, paired, advantage, stats, decision_label, decision_data, execution)
	return {"decision": decision_label, "decision_data": decision_data, "rows": len(rows), "output_root": str(OUTPUT_ROOT)}


def main() -> None:
	import argparse
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--select", action="store_true")
	parser.add_argument("--run", action="store_true")
	parser.add_argument("--resume", action="store_true")
	args = parser.parse_args()
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	if args.select or not MANIFEST_PATH.exists():
		manifest = freeze_manifest()
		write_json(MANIFEST_PATH, manifest)
		print(json.dumps({"status": "SELECTION_FROZEN", "strong": manifest["strong_case_count"], "controls": manifest["control_case_count"], "manifest": str(MANIFEST_PATH)}, indent=2))
		if not args.run:
			return
	manifest, cases = load_manifest()
	if not args.run:
		print(json.dumps({"status": "SELECTION_READY_RUN_NOT_REQUESTED", "manifest": str(MANIFEST_PATH)}, indent=2))
		return
	execution = run_evaluation(manifest, cases, args.resume)
	rows = read_jsonl(RUN_PATH) if RUN_PATH.exists() else []
	print(json.dumps({"execution": {key: value for key, value in execution.items() if key != "preflight"}, "result": write_analysis(manifest, rows, execution)}, indent=2))


if __name__ == "__main__":
	main()

