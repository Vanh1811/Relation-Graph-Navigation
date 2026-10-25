#!/usr/bin/env python3
"""Run the frozen Phase 3B strong-vs-control paired evaluation.

This harness does not change either reasoning method.  It loads the filtered
Phase 3A.1 manifest, removes all explicitly excluded diagnostic/manual-review
IDs, freezes a deterministic 20-strong/20-control sample, runs both frozen
methods for two repeats, and writes the required paired analysis artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_cor_baseline_experiment import (  # noqa: E402
	evaluate,
	integer_usage,
	load_safe_project_env,
	normalize_topic_entities,
	prediction_names,
	relation_chain_from_trace,
)
from run_phase3a_branch_necessity_benchmark import (  # noqa: E402
	known_prior_ids,
	load_candidates,
)
from chain_of_relations.kg_backend import get_backend  # noqa: E402
from chain_of_relations.methods.adaptive_v1_1_3 import AdaptiveRoutingAgent  # noqa: E402
from chain_of_relations.methods.cor.agent import CoRAgent  # noqa: E402
from chain_of_relations.methods.phase2_5_5.agent import CanonicalNormalizationAgent  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs/experiments/phase3b/frozen_strong_vs_control"
PHASE3A1_MANIFEST = ROOT / "outputs/experiments/phase3a_1/safety_cap_sensitivity/frozen_future_eval_manifest.json"
PHASE3A_AUDIT = ROOT / "outputs/experiments/phase3a/branch_necessity_benchmark/manual_quality_audit.json"
PHASE3A1_REVIEW = ROOT / "outputs/experiments/phase3a_1/safety_cap_sensitivity/manual_review_cases.json"
SEED = 20261025
STRONG_TARGET = 20
CONTROL_TARGET = 20
DEFAULT_REPEATS = 2
METHODS = ("v1_1_3", "phase2_5_6")


def read_json(path: Path) -> dict[str, Any]:
	return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	# Some diagnostic counters intentionally retain a missing/None category.
	# Sorting JSON keys would compare None with strings and fail after a live run.
	path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def canonical(value: Any) -> str:
	return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def manual_ids() -> set[str]:
	ids: set[str] = set()
	if PHASE3A_AUDIT.exists():
		data = read_json(PHASE3A_AUDIT)
		ids.update(str(row.get("case_id")) for row in data.get("entries", []) if row.get("case_id"))
	if PHASE3A1_REVIEW.exists():
		data = read_json(PHASE3A1_REVIEW)
		ids.update(str(row.get("case_id")) for row in data.get("records", []) if row.get("case_id"))
	return ids


def group_key(candidate: dict[str, Any]) -> str:
	return f"{candidate['dataset']}::{candidate['gold_structure']}"


def shuffled_groups(pool: list[dict[str, Any]], rng: random.Random) -> dict[str, list[dict[str, Any]]]:
	groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for candidate in pool:
		groups[group_key(candidate)].append(candidate)
	for values in groups.values():
		values.sort(key=lambda item: item["case_id"])
		rng.shuffle(values)
	return dict(sorted(groups.items()))


def choose_strong(pool: list[dict[str, Any]], target: int, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, int]]:
	groups = shuffled_groups(pool, rng)
	chosen: list[dict[str, Any]] = []
	allocation: dict[str, int] = {}
	# The frozen Phase 3B design asks for five from every available stratum.
	for key in sorted(groups):
		count = min(5, len(groups[key]))
		allocation[key] = count
		chosen.extend(groups[key][:count])
	if len(chosen) < target:
		remaining = [item for key in sorted(groups) for item in groups[key][allocation[key]:]]
		for item in remaining:
			if len(chosen) >= target:
				break
			chosen.append(item)
			allocation[group_key(item)] = allocation.get(group_key(item), 0) + 1
	if len(chosen) != target:
		raise RuntimeError(f"could not allocate exactly {target} strong cases; got {len(chosen)}")
	return sorted(chosen, key=lambda item: item["case_id"]), allocation


def choose_controls(pool: list[dict[str, Any]], target: int, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, int]]:
	groups = shuffled_groups(pool, rng)
	chosen: list[dict[str, Any]] = []
	allocation: Counter[str] = Counter()
	# Round-robin gives every available dataset/structure stratum a turn.
	queues = {key: list(values) for key, values in groups.items()}
	while len(chosen) < target and any(queues.values()):
		for key in sorted(queues):
			if queues[key] and len(chosen) < target:
				chosen.append(queues[key].pop(0))
				allocation[key] += 1
	if len(chosen) != target:
		raise RuntimeError(f"could not allocate exactly {target} control cases; got {len(chosen)}")
	return sorted(chosen, key=lambda item: item["case_id"]), dict(sorted(allocation.items()))


def build_sample_manifest() -> tuple[dict[str, Any], list[dict[str, Any]]]:
	if not PHASE3A1_MANIFEST.exists():
		raise SystemExit(f"missing filtered Phase 3A.1 manifest: {PHASE3A1_MANIFEST}")
	manifest = read_json(PHASE3A1_MANIFEST)
	candidates, candidate_summary = load_candidates()
	by_id = {candidate["case_id"]: candidate for candidate in candidates}
	prior = known_prior_ids()
	manual = manual_ids()
	strong_ids = [str(value) for value in manifest.get("strong_branch_benchmark_ids", [])]
	control_ids = [str(value) for value in manifest.get("answer_equivalent_branch_control_ids", [])]
	strong_pool = [by_id[value] for value in strong_ids if value in by_id and value not in prior and value not in manual]
	control_pool = [by_id[value] for value in control_ids if value in by_id and value not in prior and value not in manual]
	rng = random.Random(SEED)
	strong, strong_allocation = choose_strong(strong_pool, STRONG_TARGET, rng)
	controls, control_allocation = choose_controls(control_pool, CONTROL_TARGET, rng)
	selected = [{**candidate, "cohort": "strong"} for candidate in strong] + [{**candidate, "cohort": "control"} for candidate in controls]
	selected.sort(key=lambda item: (item["cohort"], item["case_id"]))
	selected_ids = {item["case_id"] for item in selected}
	if len(selected_ids) != STRONG_TARGET + CONTROL_TARGET:
		raise RuntimeError("sample contains duplicate case IDs")
	selected_rows = [
		{
			"case_id": item["case_id"],
			"dataset": item["dataset"],
			"split": item["split"],
			"cohort": item["cohort"],
			"gold_structure": item["gold_structure"],
			"question": item["question"],
			"topic_entities": item["topic_entities"],
			"topic_entity_names": item["topic_entity_names"],
			"gold_answers": item["gold_answers"],
			"answer_ids": item["answer_ids"],
		}
		for item in selected
	]
	output = {
		"status": "FROZEN",
		"experiment": "phase3b_frozen_strong_vs_control",
		"seed": SEED,
		"repeats_per_method": DEFAULT_REPEATS,
		"source_manifest": str(PHASE3A1_MANIFEST),
		"source_manifest_phase3a_1_decision": manifest.get("phase3a_1_decision"),
		"source_manifest_verification_status": manifest.get("verification_status"),
		"sampling_policy": "20 strong cases with five per available dataset/structure stratum; 20 controls round-robin across available strata",
		"selection_excludes_model_outcomes": True,
		"excluded_prior_diagnostic_count": len(prior),
		"excluded_manual_review_count": len(manual),
		"excluded_manual_review_ids": sorted(manual),
		"strong_source_count": len(strong_ids),
		"control_source_count": len(control_ids),
		"strong_eligible_count": len(strong_pool),
		"control_eligible_count": len(control_pool),
		"strong_allocation": strong_allocation,
		"control_allocation": control_allocation,
		"candidate_summary": candidate_summary,
		"strong_case_count": len(strong),
		"control_case_count": len(controls),
		"cases": selected_rows,
	}
	return output, selected


def api_preflight() -> dict[str, Any]:
	load_safe_project_env()
	result: dict[str, Any] = {
		"model_name": os.getenv("MODEL_NAME", ""),
		"openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
		"freebase_endpoint": os.getenv("FREEBASE_SPARQL_ENDPOINT", ""),
		"openai_available": False,
		"freebase_available": False,
	}
	if result["openai_api_key_present"]:
		base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
		request = urllib.request.Request(
			f"{base_url}/models",
			headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Accept": "application/json"},
		)
		try:
			try:
				import certifi
				context = ssl.create_default_context(cafile=certifi.where())
			except ImportError:  # pragma: no cover - project environment includes certifi
				context = ssl.create_default_context()
			with urllib.request.urlopen(request, timeout=15, context=context) as response:
				payload = json.loads(response.read().decode("utf-8"))
			result["openai_available"] = bool(payload.get("data"))
			expected = os.getenv("MODEL_NAME", "")
			result["model_list_contains_configured_model"] = any(item.get("id") == expected for item in payload.get("data", [])) if expected else False
		except Exception as exc:  # pragma: no cover - external service
			result["openai_error"] = f"{type(exc).__name__}: {exc}"
	endpoint = result["freebase_endpoint"]
	if endpoint:
		query = urllib.parse.quote("SELECT * WHERE { ?s ?p ?o } LIMIT 1")
		url = f"{endpoint}?query={query}&format=json"
		try:
			with urllib.request.urlopen(url, timeout=15) as response:
				body = response.read(2000).decode("utf-8", "replace")
			result["freebase_available"] = response.status == 200 and "bindings" in body
		except Exception as exc:  # pragma: no cover - external service
			result["freebase_error"] = f"{type(exc).__name__}: {exc}"
	return result


def make_agent(method: str) -> Any:
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
	if method == "v1_1_3":
		return AdaptiveRoutingAgent(cor, confidence_threshold=0.75, max_branches=3)
	if method == "phase2_5_6":
		return CanonicalNormalizationAgent(cor, confidence_threshold=0.75, max_branches=3)
	raise ValueError(f"unknown frozen method: {method}")


def router_event(history: list[dict[str, Any]], method: str) -> dict[str, Any]:
	wanted = "adaptive_router_v1_1_3" if method == "v1_1_3" else "phase2_5_5_canonical_normalization_router"
	for event in reversed(history):
		if event.get("type") == wanted:
			return event
	return {}


def execution_status(result: dict[str, Any], event: dict[str, Any], method: str) -> str:
	decision = result.get("router_decision", result)
	if not isinstance(decision, dict):
		decision = {}
	if method == "phase2_5_6" and str(event.get("validator_decision", "")).upper() == "REJECT":
		return "VALIDATOR_REJECT_FALLBACK"
	if result.get("fallback_used"):
		return "BRANCH_ATTEMPTED_FALLBACK" if result.get("adaptive_native_result") else "COR_FALLBACK"
	if decision.get("router_execution_mode") == "branch":
		native = result.get("adaptive_native_result") if isinstance(result.get("adaptive_native_result"), dict) else result
		if native.get("branch_failures"):
			return "BRANCH_EXECUTION_FAILURE"
		if native.get("results") or int(native.get("merged_candidate_count", 0) or 0) > 0:
			return "BRANCH_EXECUTION_SUCCESS"
		return "BRANCH_EXECUTION_EMPTY"
	return "COR_EXECUTION"


def run_one(agent: Any, method: str, repeat: int, candidate: dict[str, Any]) -> dict[str, Any]:
	question = str(candidate["question"])
	gold_answers = [str(value) for value in candidate.get("gold_answers", []) if str(value).strip()]
	topic_entities = normalize_topic_entities(candidate)
	history: list[dict[str, Any]] = []
	sparql_history: list[dict[str, Any]] = []
	step_history: list[dict[str, Any]] = []
	started = time.perf_counter()
	error: str | None = None
	try:
		if not topic_entities:
			raise ValueError("selected candidate has no valid topic entities")
		result = agent.answer(
			question=question,
			topic_entities=topic_entities,
			prompt_history=history,
			sparql_history=sparql_history,
			step_history=step_history,
		)
	except Exception as exc:  # pragma: no cover - external services and large DB
		result = {"action": "ERROR", "results": [], "reasoning_chains": ""}
		error = f"{type(exc).__name__}: {exc}"
	latency_ms = (time.perf_counter() - started) * 1000.0
	prediction = prediction_names(result)
	hit, f1 = evaluate(prediction, gold_answers)
	native = result.get("adaptive_native_result") if isinstance(result.get("adaptive_native_result"), dict) else result
	native_prediction = prediction_names(native)
	native_hit, native_f1 = evaluate(native_prediction, gold_answers)
	event = router_event(history, method)
	decision = result.get("router_decision", result)
	decision = decision if isinstance(decision, dict) else {}
	serialization = event.get("serialization", {}) if isinstance(event.get("serialization"), dict) else {}
	normalization = event.get("normalization", {}) if isinstance(event.get("normalization"), dict) else {}
	return {
		"case_id": candidate["case_id"],
		"dataset": candidate["dataset"],
		"split": candidate["split"],
		"cohort": candidate["cohort"],
		"gold_structure": candidate["gold_structure"],
		"repeat": repeat,
		"method": method,
		"question": question,
		"gold_answers": gold_answers,
		"answer_ids": candidate.get("answer_ids", []),
		"prediction": prediction,
		"native_prediction": native_prediction,
		"hits_at_1": hit,
		"f1": f1,
		"native_hits_at_1": native_hit,
		"native_f1": native_f1,
		"relation_chain": relation_chain_from_trace(result),
		"router_structure": result.get("router_structure", "BASELINE_UNCHANGED" if method == "v1_1_3" else "UNKNOWN"),
		"router_execution_mode": result.get("router_execution_mode", "cor"),
		"router_confidence": result.get("router_confidence", 0.0),
		"router_reason": result.get("router_reason", ""),
		"requested_final_answer_type": result.get("requested_final_answer_type", ""),
		"requested_final_semantic_property": result.get("requested_final_semantic_property", ""),
		"shared_answer_type": result.get("shared_answer_type", ""),
		"validation_passed": result.get("validation_passed", method == "v1_1_3"),
		"validation_failures": result.get("validation_failures", event.get("validation_failures", [])),
		"validator_decision": event.get("validator_decision", "NOT_APPLICABLE" if method == "v1_1_3" else "UNKNOWN"),
		"fallback_used": bool(result.get("fallback_used", False)),
		"fallback_reason": result.get("fallback_reason"),
		"final_execution_mode": result.get("final_execution_mode", "cor"),
		"native_execution_mode": result.get("native_execution_mode", "cor"),
		"execution_status": execution_status(result, event, method),
		"branch_attempted": result.get("router_execution_mode") == "branch",
		"branch_self_sufficient": result.get("router_execution_mode") == "branch" and not result.get("fallback_used") and bool(native.get("results")),
		"branches_created": native.get("branches_created", 0),
		"branch_decomposition": native.get("branch_decomposition", {}),
		"branch_traces": native.get("branch_traces", []),
		"branch_candidate_counts": native.get("branch_candidate_counts", []),
		"merged_candidate_count": native.get("merged_candidate_count", 0),
		"branch_failures": native.get("branch_failures", []),
		"merge_method": native.get("merge_method", "NONE"),
		"merge_outcome": native.get("merge_outcome", ""),
		"raw_constraints": event.get("constraints", []),
		"raw_parsed_result": event.get("parsed_result", {}),
		"canonical_semantic_representation": event.get("canonical_semantic_representation", normalization.get("canonical_proposal", {})),
		"normalization_status": event.get("normalization_status", "NOT_APPLICABLE" if method == "v1_1_3" else "UNKNOWN"),
		"serialization_status": serialization.get("serialization_status", "NOT_APPLICABLE" if method == "v1_1_3" else "UNKNOWN"),
		"serialized_branches": serialization.get("serialized_branches", []),
		"validator_reason": event.get("validation_failures", result.get("validation_failures", [])),
		"llm_calls": len(history),
		"kg_queries": len(sparql_history),
		"input_tokens": integer_usage(history, "input_tokens"),
		"output_tokens": integer_usage(history, "output_tokens"),
		"total_tokens": integer_usage(history, "total_tokens"),
		"latency_ms": latency_ms,
		"error": error,
		"prompt_history": history,
		"sparql_history": sparql_history,
		"step_history": step_history,
	}


def aggregate_metrics(rows: list[dict[str, Any]], cohort: str | None = None) -> dict[str, Any]:
	filtered = [row for row in rows if cohort is None or row["cohort"] == cohort]
	by_method = {method: [row for row in filtered if row["method"] == method] for method in METHODS}
	output: dict[str, Any] = {"cohort": cohort or "overall", "run_count": len(filtered)}
	for method, values in by_method.items():
		by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
		for row in values:
			by_case[row["case_id"]].append(row)
		output[method] = {
			"run_count": len(values),
			"case_count": len(by_case),
			"hits_at_1_run_mean": sum(int(row.get("hits_at_1", 0)) for row in values) / len(values) if values else 0.0,
			"f1_run_mean": sum(float(row.get("f1", 0.0)) for row in values) / len(values) if values else 0.0,
			"hits_at_1_case_majority": sum(sum(int(row.get("hits_at_1", 0)) for row in items) >= (len(items) + 1) // 2 for items in by_case.values()),
			"f1_case_mean": sum(sum(float(row.get("f1", 0.0)) for row in items) / len(items) for items in by_case.values()) / len(by_case) if by_case else 0.0,
			"fallback_runs": sum(bool(row.get("fallback_used")) for row in values),
			"branch_attempt_runs": sum(bool(row.get("branch_attempted")) for row in values),
			"branch_self_sufficient_runs": sum(bool(row.get("branch_self_sufficient")) for row in values),
			"errors": sum(bool(row.get("error")) for row in values),
		}
	return output


def paired_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
	by_case_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		by_case_method[(row["case_id"], row["method"])].append(row)
	paired: list[dict[str, Any]] = []
	counts: Counter[tuple[str, str]] = Counter()
	for case_id in sorted({row["case_id"] for row in rows}):
		methods = {method: by_case_method.get((case_id, method), []) for method in METHODS}
		if not all(methods.values()):
			continue
		v113 = methods["v1_1_3"]
		p256 = methods["phase2_5_6"]
		v_hit = sum(int(row.get("hits_at_1", 0)) for row in v113) >= (len(v113) + 1) // 2
		p_hit = sum(int(row.get("hits_at_1", 0)) for row in p256) >= (len(p256) + 1) // 2
		if v_hit and p_hit:
			outcome = "BOTH_CORRECT"
		elif v_hit:
			outcome = "V1_1_3_ONLY"
		elif p_hit:
			outcome = "PHASE2_5_6_ONLY"
		else:
			outcome = "BOTH_WRONG"
		cohort = v113[0]["cohort"]
		counts[(cohort, outcome)] += 1
		paired.append({
			"case_id": case_id,
			"cohort": cohort,
			"dataset": v113[0]["dataset"],
			"gold_structure": v113[0]["gold_structure"],
			"outcome": outcome,
			"v1_1_3_hit_majority": v_hit,
			"phase2_5_6_hit_majority": p_hit,
			"v1_1_3_f1_mean": sum(float(row.get("f1", 0.0)) for row in v113) / len(v113),
			"phase2_5_6_f1_mean": sum(float(row.get("f1", 0.0)) for row in p256) / len(p256),
		})
	overall = {outcome: sum(counts[(cohort, outcome)] for cohort in ("strong", "control")) for outcome in ("BOTH_CORRECT", "V1_1_3_ONLY", "PHASE2_5_6_ONLY", "BOTH_WRONG")}
	return {
		"paired_case_count": len(paired),
		"counts": {
			**{cohort: {outcome: counts[(cohort, outcome)] for outcome in ("BOTH_CORRECT", "V1_1_3_ONLY", "PHASE2_5_6_ONLY", "BOTH_WRONG")} for cohort in ("strong", "control")},
			"overall": overall,
		},
		"cases": paired,
	}


def routing_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
	values = [row for row in rows if row["method"] == "phase2_5_6"]
	return {
		"run_count": len(values),
		"by_cohort": {
			cohort: {
				"router_structure": dict(Counter(row.get("router_structure") for row in values if row["cohort"] == cohort)),
				"execution_status": dict(Counter(row.get("execution_status") for row in values if row["cohort"] == cohort)),
				"validator_decision": dict(Counter(row.get("validator_decision") for row in values if row["cohort"] == cohort)),
				"fallback_reason": dict(Counter(row.get("fallback_reason") for row in values if row.get("cohort") == cohort)),
				"normalization_status": dict(Counter(row.get("normalization_status") for row in values if row["cohort"] == cohort)),
				"serialization_status": dict(Counter(row.get("serialization_status") for row in values if row["cohort"] == cohort)),
				"branch_attempts": sum(bool(row.get("branch_attempted")) for row in values if row["cohort"] == cohort),
				"branch_self_sufficient": sum(bool(row.get("branch_self_sufficient")) for row in values if row["cohort"] == cohort),
			}
			for cohort in ("strong", "control")
		},
	}


def stability_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
	result: dict[str, Any] = {}
	for method in METHODS:
		method_rows = [row for row in rows if row["method"] == method]
		by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
		for row in method_rows:
			by_case[row["case_id"]].append(row)
		result[method] = {
			"case_count": len(by_case),
			"answer_stable_cases": sum(len({(row.get("hits_at_1"), round(float(row.get("f1", 0.0)), 6)) for row in items}) == 1 for items in by_case.values()),
			"router_structure_stable_cases": sum(len({row.get("router_structure") for row in items}) == 1 for items in by_case.values()),
			"execution_status_stable_cases": sum(len({row.get("execution_status") for row in items}) == 1 for items in by_case.values()),
			"error_free_cases": sum(not any(row.get("error") for row in items) for items in by_case.values()),
		}
	return result


def cost_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
	result: dict[str, Any] = {}
	for method in METHODS:
		values = [row for row in rows if row["method"] == method]
		result[method] = {
			"run_count": len(values),
			"llm_calls": sum(int(row.get("llm_calls", 0)) for row in values),
			"kg_queries": sum(int(row.get("kg_queries", 0)) for row in values),
			"input_tokens": sum(int(row.get("input_tokens", 0)) for row in values),
			"output_tokens": sum(int(row.get("output_tokens", 0)) for row in values),
			# The existing prompt history occasionally reports an incomplete
			# per-event total_tokens value.  Use the authoritative input+output
			# fields for cost aggregation and retain the reported value separately.
			"total_tokens": sum(int(row.get("input_tokens", 0)) + int(row.get("output_tokens", 0)) for row in values),
			"reported_total_tokens": sum(int(row.get("total_tokens", 0)) for row in values),
			"latency_ms_total": sum(float(row.get("latency_ms", 0.0)) for row in values),
			"latency_ms_mean": sum(float(row.get("latency_ms", 0.0)) for row in values) / len(values) if values else 0.0,
		}
	return result


def failure_cause(row: dict[str, Any]) -> str:
	if row.get("error"):
		return "RUNTIME_ERROR"
	if row.get("method") == "phase2_5_6":
		if row.get("validator_decision") == "REJECT":
			return "VALIDATOR_REJECTION"
		if row.get("normalization_status") in {"NORMALIZATION_FAILED", "UNKNOWN"}:
			return "NORMALIZATION_FAILURE"
		if row.get("serialization_status") in {"FAILED", "UNKNOWN"}:
			return "SERIALIZATION_FAILURE"
	if row.get("fallback_used"):
		return "FALLBACK"
	if row.get("branch_attempted") and row.get("execution_status") != "BRANCH_EXECUTION_SUCCESS":
		return "BRANCH_EXECUTION_FAILURE"
	if not row.get("hits_at_1"):
		return "ANSWER_WRONG"
	return "NONE"


def decision_label(rows: list[dict[str, Any]], paired: dict[str, Any]) -> str:
	if any(row.get("error") for row in rows):
		return "INCONCLUSIVE"
	counts = paired["counts"]["overall"]
	strong = paired["counts"]["strong"]
	control = paired["counts"]["control"]
	if control["PHASE2_5_6_ONLY"] > control["V1_1_3_ONLY"] + control["BOTH_CORRECT"]:
		return "CONTROL_REGRESSION_CONCERN"
	if strong["PHASE2_5_6_ONLY"] > strong["V1_1_3_ONLY"] and control["PHASE2_5_6_ONLY"] >= control["V1_1_3_ONLY"]:
		return "BRANCH_ADVANTAGE_SUPPORTED"
	if strong["PHASE2_5_6_ONLY"] > strong["V1_1_3_ONLY"]:
		return "BRANCH_ADVANTAGE_PARTIAL"
	if counts["PHASE2_5_6_ONLY"] == counts["V1_1_3_ONLY"] == 0:
		return "NO_BRANCH_ADVANTAGE"
	return "INCONCLUSIVE"


def write_reports(sample: dict[str, Any], rows: list[dict[str, Any]], preflight: dict[str, Any]) -> None:
	paired = paired_analysis(rows)
	routing = routing_behavior(rows)
	stability = stability_analysis(rows)
	cost = cost_analysis(rows)
	metrics = {
		"overall": aggregate_metrics(rows),
		"strong": aggregate_metrics(rows, "strong"),
		"control": aggregate_metrics(rows, "control"),
	}
	failures = [
		{
			"case_id": row["case_id"],
			"cohort": row["cohort"],
			"dataset": row["dataset"],
			"method": row["method"],
			"repeat": row["repeat"],
			"gold_structure": row["gold_structure"],
			"hits_at_1": row["hits_at_1"],
			"f1": row["f1"],
			"cause": failure_cause(row),
			"error": row.get("error"),
			"router_structure": row.get("router_structure"),
			"router_execution_mode": row.get("router_execution_mode"),
			"validator_decision": row.get("validator_decision"),
			"fallback_used": row.get("fallback_used"),
			"fallback_reason": row.get("fallback_reason"),
			"execution_status": row.get("execution_status"),
			"validation_failures": row.get("validation_failures"),
		}
		for row in rows
		if failure_cause(row) != "NONE"
	]
	write_json(OUTPUT_ROOT / "paired_analysis.json", paired)
	write_json(OUTPUT_ROOT / "routing_behavior.json", routing)
	write_json(OUTPUT_ROOT / "stability_analysis.json", stability)
	write_json(OUTPUT_ROOT / "cost_analysis.json", cost)
	write_json(OUTPUT_ROOT / "metrics.json", metrics)
	write_jsonl(OUTPUT_ROOT / "failure_analysis.jsonl", failures)
	decision = decision_label(rows, paired)
	strong_cases = [item for item in paired["cases"] if item["cohort"] == "strong"]
	control_cases = [item for item in paired["cases"] if item["cohort"] == "control"]
	for name, case_rows in (("strong_case_analysis.md", strong_cases), ("control_case_analysis.md", control_cases)):
		lines = [f"# {name.replace('_', ' ').replace('.md', '').title()}", "", "| Case | Dataset | Gold structure | Outcome | v1.1.3 F1 | Phase 2.5.6 F1 |", "|---|---|---|---|---:|---:|"]
		lines.extend(f"| {item['case_id']} | {item['dataset']} | {item['gold_structure']} | {item['outcome']} | {item['v1_1_3_f1_mean']:.4f} | {item['phase2_5_6_f1_mean']:.4f} |" for item in case_rows)
		(OUTPUT_ROOT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
	decision_text = f"# Phase 3B decision summary\n\nDecision: **{decision}**\n\n"
	decision_text += "The sample is an initial frozen 20-strong/20-control diagnostic only. It is not the full Phase 3 benchmark.\n\n"
	decision_text += f"Strong paired outcomes: `{paired['counts']['strong']}`\n\nControl paired outcomes: `{paired['counts']['control']}`\n\n"
	decision_text += f"Phase 3A.1 source decision: `{sample['source_manifest_phase3a_1_decision']}`.\n"
	(OUTPUT_ROOT / "decision_summary.md").write_text(decision_text, encoding="utf-8")
	report = f"""# Phase 3B frozen strong-vs-control evaluation

## 1. Scope and frozen inputs

This is the initial Phase 3B gate using only the filtered Phase 3A.1 manifest. The Phase 3A.1 decision remains `{sample['source_manifest_phase3a_1_decision']}`; the 32 unresolved IDs were removed by that manifest before sampling.

## 2. Research question

Does the frozen Phase 2.5.6 combined branch-aware stack improve branch-required cases while preserving answer-equivalent controls compared with frozen adaptive v1.1.3?

## 3. Frozen methods

- `v1_1_3`: existing `AdaptiveRoutingAgent` with the v1.1.3 router.
- `phase2_5_6`: existing `CanonicalNormalizationAgent`, reusing the Phase 2.5.5 normalization, Phase 2.5.3B serialization/validator, and unchanged branch execution/fallback.
- Shared CoR settings: depth 3, relation width 3, relation sampling threshold 1000, max token 1024, confidence threshold 0.75.

## 4. Sampling and exclusions

Seed: `{SEED}`. The frozen sample contains `{sample['strong_case_count']}` strong and `{sample['control_case_count']}` control cases, with `{sample['repeats_per_method']}` repeats per method. Prior Phase 2.4.x–2.6 diagnostics, Phase 3A manual QA, and Phase 3A.1 manual-review IDs were excluded before sampling.

## 5. Preflight

```json
{json.dumps(preflight, ensure_ascii=False, indent=2)}
```

## 6. Execution completeness

Observed run rows: `{len(rows)}`. Expected rows: `{sample['strong_case_count'] + sample['control_case_count']} cases × {len(METHODS)} methods × {sample['repeats_per_method']} repeats = {(sample['strong_case_count'] + sample['control_case_count']) * len(METHODS) * sample['repeats_per_method']}`.

## 7. Paired outcomes

See `paired_analysis.json`; outcomes are reported separately for strong and control cohorts.

## 8. QA metrics

See `metrics.json` and the run-level records for Hits@1 and F1. Both per-run and per-case repeat aggregation are retained.

Overall run means: v1.1.3 Hits@1 `{metrics['overall']['v1_1_3']['hits_at_1_run_mean']:.4f}`, F1 `{metrics['overall']['v1_1_3']['f1_run_mean']:.4f}`; Phase 2.5.6 Hits@1 `{metrics['overall']['phase2_5_6']['hits_at_1_run_mean']:.4f}`, F1 `{metrics['overall']['phase2_5_6']['f1_run_mean']:.4f}`.

Strong run means: v1.1.3 Hits@1 `{metrics['strong']['v1_1_3']['hits_at_1_run_mean']:.4f}`, F1 `{metrics['strong']['v1_1_3']['f1_run_mean']:.4f}`; Phase 2.5.6 Hits@1 `{metrics['strong']['phase2_5_6']['hits_at_1_run_mean']:.4f}`, F1 `{metrics['strong']['phase2_5_6']['f1_run_mean']:.4f}`.

Control run means: v1.1.3 Hits@1 `{metrics['control']['v1_1_3']['hits_at_1_run_mean']:.4f}`, F1 `{metrics['control']['v1_1_3']['f1_run_mean']:.4f}`; Phase 2.5.6 Hits@1 `{metrics['control']['phase2_5_6']['hits_at_1_run_mean']:.4f}`, F1 `{metrics['control']['phase2_5_6']['f1_run_mean']:.4f}`.

## 9. Phase 2.5.6 routing behavior

See `routing_behavior.json` for router structure, normalization, serialization, validator, fallback, and native branch success counts.

## 10. Branch self-sufficiency

Native branch attempts and self-sufficient branch execution are recorded per run and cohort in `routing_behavior.json`.

## 11. Intermediate-answer limitation

BRANCH_INTERMEDIATE cases are retained as a separate `gold_structure` stratum in the sample and reports; they are not pooled silently with BRANCH_FINAL cases.

## 12. Repeat stability

See `stability_analysis.json` for answer, route, execution, and error-free agreement across repeats.

## 13. Cost and latency

See `cost_analysis.json` for LLM calls, tokens, KG queries, and latency by method.

## 14. Failure analysis

`failure_analysis.jsonl` records validator, serialization, fallback, branch-execution, runtime, and answer failures without collapsing distinct causes.

## 15. Strong-case analysis

See `strong_case_analysis.md`.

## 16. Control-case analysis

See `control_case_analysis.md`.

## 17. Decision

Decision label: **{decision}**.

## 18. Limitations

This is an initial 40-case diagnostic, not a full 1,533-strong benchmark. The source Phase 3A.1 safety-cap analysis was inconclusive because 63 capped records remained unresolved at cap 5,000; those IDs were excluded from the filtered future manifest.

## 19. Reproducibility

`sample_manifest.json` freezes the selected IDs and provenance. `run_results.jsonl` retains both methods, repeats, router metadata, execution traces, token counts, and endpoint query histories.

## 20. Required next step

Do not tune or migrate methods from this result. Interpret the initial gate using the decision label above; only a subsequent explicit instruction should expand the frozen evaluation.

## 21. Output index

`sample_manifest.json`, `run_results.jsonl`, `paired_analysis.json`, `metrics.json`, `routing_behavior.json`, `stability_analysis.json`, `cost_analysis.json`, `failure_analysis.jsonl`, `strong_case_analysis.md`, `control_case_analysis.md`, and `decision_summary.md`.
"""
	(OUTPUT_ROOT / "PHASE3B_REPORT.md").write_text(report, encoding="utf-8")
	write_json(OUTPUT_ROOT / "execution_summary.json", {"decision": decision, "preflight": preflight, "row_count": len(rows), "paired": paired["counts"]})


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--prepare", action="store_true", help="freeze the sample and stop")
	parser.add_argument("--run", action="store_true", help="run both frozen methods")
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
	args = parser.parse_args()
	if args.repeats <= 0:
		raise SystemExit("--repeats must be positive")
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	sample, selected = build_sample_manifest()
	sample["repeats_per_method"] = args.repeats
	write_json(OUTPUT_ROOT / "sample_manifest.json", sample)
	if args.prepare and not args.run:
		print(json.dumps({"sample_manifest": str(OUTPUT_ROOT / "sample_manifest.json"), "strong": sample["strong_case_count"], "control": sample["control_case_count"]}, indent=2))
		return
	if not args.run:
		raise SystemExit("choose --prepare or --run")
	preflight = api_preflight()
	if not preflight.get("openai_api_key_present") or not preflight.get("model_name"):
		raise SystemExit(f"LLM preflight failed: {preflight}")
	if not preflight.get("openai_available") or not preflight.get("freebase_available"):
		write_json(OUTPUT_ROOT / "preflight.json", preflight)
		raise SystemExit(f"live endpoint/API preflight failed: {json.dumps(preflight, ensure_ascii=False)}")
	write_json(OUTPUT_ROOT / "preflight.json", preflight)
	selected_by_id = {item["case_id"]: item for item in selected}
	expected = {(method, repeat, case_id) for method in METHODS for repeat in range(1, args.repeats + 1) for case_id in selected_by_id}
	rows: list[dict[str, Any]] = []
	path = OUTPUT_ROOT / "run_results.jsonl"
	if args.resume and path.exists():
		rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
	existing = {(row.get("method"), int(row.get("repeat", 0)), row.get("case_id")) for row in rows}
	for method in METHODS:
		for repeat in range(1, args.repeats + 1):
			agent = make_agent(method)
			for index, candidate in enumerate(selected, start=1):
				key = (method, repeat, candidate["case_id"])
				if key in existing:
					continue
				print(f"[{method}] repeat={repeat} {index}/{len(selected)} {candidate['case_id']}", flush=True)
				row = run_one(agent, method, repeat, candidate)
				rows.append(row)
				existing.add(key)
				with path.open("a", encoding="utf-8") as handle:
					handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
	rows.sort(key=lambda row: (row["method"], int(row["repeat"]), row["case_id"]))
	write_jsonl(path, rows)
	missing = sorted(expected - {(row.get("method"), int(row.get("repeat", 0)), row.get("case_id")) for row in rows})
	if missing:
		print(json.dumps({"status": "INCOMPLETE", "missing_count": len(missing), "missing": missing[:20]}, indent=2))
	else:
		write_reports(sample, rows, preflight)
		print(json.dumps({"status": "COMPLETE", "rows": len(rows), "output_root": str(OUTPUT_ROOT)}, indent=2))


if __name__ == "__main__":
	main()
