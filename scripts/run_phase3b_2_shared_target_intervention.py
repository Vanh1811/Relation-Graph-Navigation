#!/usr/bin/env python3
"""Run the Phase 3B.2 shared-target extraction intervention.

The comparison is deliberately narrow:

* ``phase2_5_6`` is the frozen Phase 2.5.6 stack.
* ``phase3b_2`` changes only the shared-target extraction prompt and audit
  fields, then reuses the same normalization, serialization, validator,
  branch execution, fallback, and Freebase backend.

The script freezes a separate 24-case validation manifest before model calls:
8 BRANCH_INTERMEDIATE strong cases, 8 BRANCH_FINAL strong cases, and 8
answer-equivalent controls.  It also runs the new intervention on the frozen
12-case Phase 3B.1 development diagnostic set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import ssl
import sys
import time
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
from chain_of_relations.methods.cor.agent import CoRAgent  # noqa: E402
from chain_of_relations.methods.phase2_5_5.agent import CanonicalNormalizationAgent  # noqa: E402
from chain_of_relations.methods.phase3b_2.agent import SharedTargetInterventionAgent  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs/experiments/phase3b_2/shared_target_intervention"
PHASE3A1_MANIFEST = ROOT / "outputs/experiments/phase3a_1/safety_cap_sensitivity/frozen_future_eval_manifest.json"
PHASE3A_AUDIT = ROOT / "outputs/experiments/phase3a/branch_necessity_benchmark/manual_quality_audit.json"
PHASE3A1_REVIEW = ROOT / "outputs/experiments/phase3a_1/safety_cap_sensitivity/manual_review_cases.json"
PHASE3B1_ROOT = ROOT / "outputs/experiments/phase3b_1/strong_failure_causal_analysis"
PHASE3B_ROOT = ROOT / "outputs/experiments/phase3b/frozen_strong_vs_control"
SEED = 20261026
DEVELOPMENT_REPEATS = 2
FRESH_REPEATS = 2
METHODS = ("phase2_5_6", "phase3b_2")


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


def target_type(proposal: Any) -> str:
	if isinstance(proposal, dict):
		value = proposal.get("shared_target")
		if isinstance(value, dict):
			return upper(value.get("type") or value.get("target")) or "NONE"
		return upper(value) or "NONE"
	return "NONE"


def canonical_target_type(proposal: Any) -> str:
	return target_type(proposal)


def expected_target(gold_structure: str, cohort: str) -> str:
	if cohort == "control":
		return "ANSWER"
	return "INTERMEDIATE" if gold_structure == "BRANCH_INTERMEDIATE" else "ANSWER"


def manual_review_ids() -> set[str]:
	ids: set[str] = set()
	if PHASE3A_AUDIT.exists():
		data = read_json(PHASE3A_AUDIT)
		ids.update(text(item.get("case_id")) for item in data.get("entries", []) if item.get("case_id"))
	if PHASE3A1_REVIEW.exists():
		data = read_json(PHASE3A1_REVIEW)
		ids.update(text(item.get("case_id")) for item in data.get("records", []) if item.get("case_id"))
	return ids


def ids_in_output_roots(patterns: tuple[str, ...]) -> set[str]:
	pattern = re.compile(r"(?:WebQ(?:Trn|Test)-[A-Za-z0-9_]+|\b[0-9]{10,13}\b)")
	ids: set[str] = set()
	for glob in patterns:
		for directory in ROOT.glob(glob):
			if not directory.is_dir():
				continue
			for path in directory.rglob("*"):
				if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
					continue
				try:
					ids.update(pattern.findall(path.read_text(encoding="utf-8", errors="ignore")))
				except OSError:
					continue
	return ids


def fresh_exclusion_ids() -> set[str]:
	"""Collect only pre-existing Phase 2.x/Phase 3 diagnostic IDs."""
	ids = set(known_prior_ids())
	ids.update(ids_in_output_roots(("outputs/experiments/phase2_*", "outputs/experiments/phase3b*")))
	ids.update(manual_review_ids())
	return {value for value in ids if value}


def source_candidate_pools() -> tuple[dict[str, Any], list[dict[str, Any]]]:
	if not PHASE3A1_MANIFEST.exists():
		raise SystemExit(f"missing Phase 3A.1 manifest: {PHASE3A1_MANIFEST}")
	manifest = read_json(PHASE3A1_MANIFEST)
	candidates, summary = load_candidates()
	by_id = {candidate["case_id"]: candidate for candidate in candidates}
	strong_ids = [text(value) for value in manifest.get("strong_branch_benchmark_ids", [])]
	control_ids = [text(value) for value in manifest.get("answer_equivalent_branch_control_ids", [])]
	excluded = fresh_exclusion_ids()
	strong = [by_id[value] for value in strong_ids if value in by_id and value not in excluded]
	controls = [by_id[value] for value in control_ids if value in by_id and value not in excluded]
	return {
		"source_manifest": str(PHASE3A1_MANIFEST),
		"source_manifest_phase3a_1_decision": manifest.get("phase3a_1_decision"),
		"source_manifest_verification_status": manifest.get("verification_status"),
		"candidate_summary": summary,
		"excluded_id_count": len(excluded),
		"excluded_ids": sorted(excluded),
		"strong_source_count": len(strong),
		"control_source_count": len(controls),
	}, strong + [{**candidate, "_cohort_source": "control"} for candidate in controls] + [{**candidate, "_cohort_source": "strong"} for candidate in strong]


def balanced_sample(pool: list[dict[str, Any]], size: int, rng: random.Random) -> list[dict[str, Any]]:
	groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for candidate in pool:
		groups[candidate["dataset"]].append(candidate)
	for values in groups.values():
		values.sort(key=lambda item: item["case_id"])
		rng.shuffle(values)
	selected: list[dict[str, Any]] = []
	queues = {key: list(values) for key, values in sorted(groups.items())}
	while len(selected) < size and any(queues.values()):
		for key in sorted(queues):
			if queues[key] and len(selected) < size:
				selected.append(queues[key].pop(0))
	if len(selected) != size:
		raise RuntimeError(f"could not select {size} cases; selected {len(selected)}")
	return sorted(selected, key=lambda item: item["case_id"])


def public_case(candidate: dict[str, Any], cohort: str) -> dict[str, Any]:
	return {
		"case_id": candidate["case_id"],
		"dataset": candidate["dataset"],
		"split": candidate["split"],
		"cohort": cohort,
		"gold_structure": candidate["gold_structure"],
		"question": candidate["question"],
		"topic_entities": candidate["topic_entities"],
		"topic_entity_names": candidate["topic_entity_names"],
		"gold_answers": candidate["gold_answers"],
		"answer_ids": candidate.get("answer_ids", []),
		"answer_variable": candidate["answer_variable"],
		"shared_target_variable": candidate["shared_target_variable"],
		"patterns": candidate["patterns"],
		"branches": candidate["branches"],
		"source_path": candidate["source_path"],
		"expected_shared_target_type": expected_target(candidate["gold_structure"], cohort),
	}


def build_fresh_manifest() -> tuple[dict[str, Any], list[dict[str, Any]]]:
	metadata, pooled = source_candidate_pools()
	strong_intermediate = [item for item in pooled if item.get("_cohort_source") == "strong" and item["gold_structure"] == "BRANCH_INTERMEDIATE"]
	strong_final = [item for item in pooled if item.get("_cohort_source") == "strong" and item["gold_structure"] == "BRANCH_FINAL"]
	controls = [item for item in pooled if item.get("_cohort_source") == "control"]
	rng = random.Random(SEED)
	selected = [
		*[(item, "strong_intermediate") for item in balanced_sample(strong_intermediate, 8, rng)],
		*[(item, "strong_final") for item in balanced_sample(strong_final, 8, rng)],
		*[(item, "control") for item in balanced_sample(controls, 8, rng)],
	]
	selected_rows = [public_case(candidate, cohort) for candidate, cohort in selected]
	selected_rows.sort(key=lambda item: (item["cohort"], item["case_id"]))
	manifest = {
		"status": "FROZEN",
		"experiment": "phase3b_2_shared_target_intervention",
		"seed": SEED,
		"repeats_per_method": FRESH_REPEATS,
		"sampling_policy": "8 BRANCH_INTERMEDIATE strong, 8 BRANCH_FINAL strong, and 8 answer-equivalent controls; balanced round-robin by dataset where available",
		"selection_excludes_model_outcomes": True,
		"excluded_phase2_and_phase3_diagnostic_count": metadata["excluded_id_count"],
		"source_metadata": metadata,
		"cohort_counts": dict(Counter(item["cohort"] for item in selected_rows)),
		"dataset_counts": dict(Counter(f"{item['cohort']}|{item['dataset']}" for item in selected_rows)),
		"cases": selected_rows,
	}
	return manifest, selected_rows


def load_frozen_fresh_manifest() -> tuple[dict[str, Any], list[dict[str, Any]]]:
	path = OUTPUT_ROOT / "fresh_validation_manifest.json"
	if path.exists():
		manifest = read_json(path)
		return manifest, list(manifest.get("cases", []))
	manifest, selected = build_fresh_manifest()
	write_json(path, manifest)
	return manifest, selected


def load_development_manifest() -> tuple[dict[str, Any], list[dict[str, Any]]]:
	path = PHASE3B1_ROOT / "target_case_manifest.json"
	if not path.exists():
		raise SystemExit(f"missing Phase 3B.1 target manifest: {path}")
	manifest = read_json(path)
	return manifest, list(manifest.get("cases", []))


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
			except ImportError:
				context = ssl.create_default_context()
			with urllib.request.urlopen(request, timeout=15, context=context) as response:
				payload = json.loads(response.read().decode("utf-8"))
			result["openai_available"] = bool(payload.get("data"))
			result["model_list_contains_configured_model"] = any(item.get("id") == result["model_name"] for item in payload.get("data", []))
		except Exception as exc:
			result["openai_error"] = f"{type(exc).__name__}: {exc}"
	endpoint = result["freebase_endpoint"]
	if endpoint:
		query = urllib.parse.quote("SELECT * WHERE { ?s ?p ?o } LIMIT 1")
		try:
			with urllib.request.urlopen(f"{endpoint}?query={query}&format=json", timeout=15) as response:
				body = response.read(2000).decode("utf-8", "replace")
			result["freebase_available"] = response.status == 200 and "bindings" in body
		except Exception as exc:
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
	if method == "phase2_5_6":
		return CanonicalNormalizationAgent(cor, confidence_threshold=0.75, max_branches=3)
	if method == "phase3b_2":
		return SharedTargetInterventionAgent(cor, confidence_threshold=0.75, max_branches=3)
	raise ValueError(f"unknown method: {method}")


def router_event(history: list[dict[str, Any]], method: str) -> dict[str, Any]:
	wanted = "phase2_5_5_canonical_normalization_router" if method == "phase2_5_6" else "phase3b_2_shared_target_intervention_router"
	for event in reversed(history):
		if event.get("type") == wanted:
			return event
	return {}


def execution_status(result: dict[str, Any], event: dict[str, Any]) -> str:
	if str(event.get("validator_decision", "")).upper() == "REJECT":
		return "VALIDATOR_REJECT_FALLBACK"
	if result.get("fallback_used"):
		return "BRANCH_ATTEMPTED_FALLBACK" if result.get("adaptive_native_result") else "COR_FALLBACK"
	decision = result.get("router_decision", result)
	if isinstance(decision, dict) and decision.get("router_execution_mode") == "branch":
		native = result.get("adaptive_native_result") if isinstance(result.get("adaptive_native_result"), dict) else result
		if native.get("branch_failures"):
			return "BRANCH_EXECUTION_FAILURE"
		if native.get("results") or int(native.get("merged_candidate_count", 0) or 0) > 0:
			return "BRANCH_EXECUTION_SUCCESS"
		return "BRANCH_EXECUTION_EMPTY"
	return "COR_EXECUTION"


def run_one(agent: Any, method: str, repeat: int, candidate: dict[str, Any]) -> dict[str, Any]:
	question = text(candidate["question"])
	gold_answers = [text(value) for value in candidate.get("gold_answers", []) if text(value)]
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
	except Exception as exc:  # pragma: no cover - external API/Freebase path
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
	parsed = event.get("parsed_result", {}) if isinstance(event.get("parsed_result"), dict) else {}
	canonical = event.get("canonical_semantic_representation", normalization.get("canonical_proposal", {}))
	shared_extraction = event.get("shared_target_extraction", {}) if isinstance(event.get("shared_target_extraction"), dict) else {}
	cohort = candidate.get("cohort", "development")
	gold_structure = candidate.get("gold_structure", "")
	return {
		"case_id": candidate["case_id"],
		"dataset": candidate.get("dataset", ""),
		"split": candidate.get("split", ""),
		"cohort": cohort,
		"gold_structure": gold_structure,
		"expected_shared_target_type": expected_target(gold_structure, cohort),
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
		"router_structure": result.get("router_structure", "UNKNOWN"),
		"router_execution_mode": result.get("router_execution_mode", "cor"),
		"router_confidence": result.get("router_confidence", 0.0),
		"router_reason": result.get("router_reason", ""),
		"shared_answer_type": result.get("shared_answer_type", ""),
		"validation_passed": result.get("validation_passed", False),
		"validation_failures": result.get("validation_failures", event.get("validation_failures", [])),
		"validator_decision": event.get("validator_decision", "UNKNOWN"),
		"fallback_used": bool(result.get("fallback_used", False)),
		"fallback_reason": result.get("fallback_reason"),
		"final_execution_mode": result.get("final_execution_mode", "cor"),
		"native_execution_mode": result.get("native_execution_mode", "cor"),
		"execution_status": execution_status(result, event),
		"branch_attempted": result.get("router_execution_mode") == "branch",
		"branch_self_sufficient": result.get("router_execution_mode") == "branch" and not result.get("fallback_used") and bool(native.get("results")),
		"branches_created": native.get("branches_created", 0),
		"branch_candidate_counts": native.get("branch_candidate_counts", []),
		"merged_candidate_count": native.get("merged_candidate_count", 0),
		"branch_failures": native.get("branch_failures", []),
		"merge_method": native.get("merge_method", "NONE"),
		"merge_outcome": native.get("merge_outcome", ""),
		"raw_constraints": event.get("constraints", []),
		"raw_parsed_result": parsed,
		"canonical_semantic_representation": canonical,
		"shared_target_extraction": shared_extraction,
		"raw_shared_target_type": target_type(parsed),
		"canonical_shared_target_type": canonical_target_type(canonical),
	"shared_target_description": shared_extraction.get("shared_target_description", "") or text(parsed.get("shared_target", {}).get("description") if isinstance(parsed.get("shared_target"), dict) else ""),
		"requested_answer_description": shared_extraction.get("requested_answer_description", ""),
		"requires_downstream_relation": shared_extraction.get("requires_downstream_relation"),
	"requires_downstream_relation_correct": (
		None
		if shared_extraction.get("requires_downstream_relation") is None
		else bool(shared_extraction.get("requires_downstream_relation")) == (expected_target(gold_structure, cohort) == "INTERMEDIATE")
	),
		"downstream_relation_description": shared_extraction.get("downstream_relation_description", ""),
		"normalization_status": event.get("normalization_status", "UNKNOWN"),
		"serialization_status": serialization.get("serialization_status", "UNKNOWN"),
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


def target_correct(row: dict[str, Any]) -> bool:
	return row.get("canonical_shared_target_type") == row.get("expected_shared_target_type")


def case_majority(rows: list[dict[str, Any]], predicate) -> dict[str, bool]:
	grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		grouped[row["case_id"]].append(row)
	return {case_id: sum(bool(predicate(row)) for row in values) >= (len(values) + 1) // 2 for case_id, values in grouped.items()}


def semantic_records(rows: list[dict[str, Any]], baseline_by_key: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
	records: list[dict[str, Any]] = []
	for row in rows:
		baseline = baseline_by_key.get((row["case_id"], int(row["repeat"])))
		records.append({
			"case_id": row["case_id"],
			"dataset": row.get("dataset"),
			"cohort": row.get("cohort"),
			"gold_structure": row.get("gold_structure"),
			"repeat": row.get("repeat"),
			"method": row.get("method"),
			"gold_shared_target_type": row.get("expected_shared_target_type"),
			"shared_target_type": row.get("canonical_shared_target_type"),
			"raw_shared_target_type": row.get("raw_shared_target_type"),
			"shared_target_correct": target_correct(row),
			"shared_target_description": row.get("shared_target_description", ""),
			"description_adequate_automatic": bool(text(row.get("shared_target_description"))),
			"requested_answer_description": row.get("requested_answer_description", ""),
			"requires_downstream_relation": row.get("requires_downstream_relation"),
			"requires_downstream_relation_correct": (
				None if row.get("requires_downstream_relation") is None else bool(row.get("requires_downstream_relation")) == (row.get("expected_shared_target_type") == "INTERMEDIATE")
			),
			"downstream_relation_description": row.get("downstream_relation_description", ""),
			"structure_correct": row.get("router_structure") == row.get("gold_structure"),
			"baseline_target_type_same_repeat": (
				baseline.get("canonical_shared_target_type") if baseline else None
			) == row.get("canonical_shared_target_type") if baseline else None,
			"semantic_change_vs_phase2_5_6": (
				baseline is not None and baseline.get("canonical_shared_target_type") != row.get("canonical_shared_target_type")
			),
		})
	return records


def semantic_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
	def requires_correct(row: dict[str, Any]) -> bool | None:
		if row.get("requires_downstream_relation") is None:
			return row.get("requires_downstream_relation_correct")
		return bool(row.get("requires_downstream_relation")) == (row.get("expected_shared_target_type") == "INTERMEDIATE")

	output: dict[str, Any] = {}
	for cohort in ("strong_intermediate", "strong_final", "control"):
		cohort_rows = [row for row in rows if row.get("cohort") == cohort]
		output[cohort] = {}
		for method in METHODS:
			values = [row for row in cohort_rows if row.get("method") == method]
			by_case_target = case_majority(values, target_correct)
			available_requires = [row for row in values if requires_correct(row) is not None]
			by_case_requires = case_majority(available_requires, lambda row: bool(requires_correct(row)))
			output[cohort][method] = {
				"run_count": len(values),
				"case_count": len(by_case_target),
				"shared_target_accuracy_run": sum(target_correct(row) for row in values) / len(values) if values else 0.0,
				"shared_target_accuracy_case_majority": sum(by_case_target.values()) / len(by_case_target) if by_case_target else 0.0,
				"description_adequacy_rate_run": sum(bool(text(row.get("shared_target_description"))) for row in values) / len(values) if values else 0.0,
				"requires_downstream_relation_available_run_count": sum(requires_correct(row) is not None for row in values),
				"requires_downstream_relation_accuracy_run": sum(bool(requires_correct(row)) for row in values if requires_correct(row) is not None) / sum(requires_correct(row) is not None for row in values) if any(requires_correct(row) is not None for row in values) else None,
				"requires_downstream_relation_accuracy_case_majority": sum(by_case_requires.values()) / len(by_case_requires) if by_case_requires else None,
				"false_intermediate_run_rate": sum(row.get("canonical_shared_target_type") == "INTERMEDIATE" for row in values) / len(values) if values else 0.0,
				"false_intermediate_case_count": sum(1 for _case_id, case_rows in _group_by_case(values).items() if _majority_value(case_rows, "canonical_shared_target_type") == "INTERMEDIATE"),
			}
	return output


def _group_by_case(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
	grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		grouped[row["case_id"]].append(row)
	return grouped


def _majority_value(rows: list[dict[str, Any]], field: str) -> Any:
	values = [row.get(field) for row in rows]
	return Counter(values).most_common(1)[0][0] if values else None


def qa_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
	output: dict[str, Any] = {}
	for cohort in ("strong_intermediate", "strong_final", "control"):
		output[cohort] = {}
		for method in METHODS:
			values = [row for row in rows if row.get("cohort") == cohort and row.get("method") == method]
			output[cohort][method] = {
				"run_count": len(values),
				"hits_at_1_mean": sum(int(row.get("hits_at_1", 0)) for row in values) / len(values) if values else 0.0,
				"f1_mean": sum(float(row.get("f1", 0.0)) for row in values) / len(values) if values else 0.0,
				"fallback_rate": sum(bool(row.get("fallback_used")) for row in values) / len(values) if values else 0.0,
				"validator_pass_rate": sum(str(row.get("validator_decision", "")).upper() == "PASS" for row in values) / len(values) if values else 0.0,
				"native_branch_attempt_rate": sum(bool(row.get("branch_attempted")) for row in values) / len(values) if values else 0.0,
				"native_branch_success_rate": sum(row.get("execution_status") == "BRANCH_EXECUTION_SUCCESS" for row in values) / len(values) if values else 0.0,
			}
	return output


def paired_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
	by_key = {(row["case_id"], row["method"], int(row["repeat"])): row for row in rows}
	cases = sorted({row["case_id"] for row in rows})
	paired: list[dict[str, Any]] = []
	counts: Counter[tuple[str, str]] = Counter()
	for case_id in cases:
		case_rows = [row for row in rows if row["case_id"] == case_id]
		cohort = case_rows[0].get("cohort", "")
		baseline = [by_key.get((case_id, "phase2_5_6", repeat)) for repeat in range(1, FRESH_REPEATS + 1)]
		intervention = [by_key.get((case_id, "phase3b_2", repeat)) for repeat in range(1, FRESH_REPEATS + 1)]
		if not all(baseline) or not all(intervention):
			continue
		base_hit = sum(int(row.get("hits_at_1", 0)) for row in baseline) >= (len(baseline) + 1) // 2
		int_hit = sum(int(row.get("hits_at_1", 0)) for row in intervention) >= (len(intervention) + 1) // 2
		if base_hit and int_hit:
			outcome = "BOTH_CORRECT"
		elif base_hit:
			outcome = "PHASE2_5_6_ONLY"
		elif int_hit:
			outcome = "PHASE3B_2_ONLY"
		else:
			outcome = "BOTH_WRONG"
		counts[(cohort, outcome)] += 1
		paired.append({
			"case_id": case_id,
			"cohort": cohort,
			"dataset": case_rows[0].get("dataset"),
			"gold_structure": case_rows[0].get("gold_structure"),
			"outcome": outcome,
			"phase2_5_6_hit_majority": base_hit,
			"phase3b_2_hit_majority": int_hit,
			"phase2_5_6_target_majority": _majority_value(baseline, "canonical_shared_target_type"),
			"phase3b_2_target_majority": _majority_value(intervention, "canonical_shared_target_type"),
			"phase2_5_6_f1_mean": sum(float(row.get("f1", 0.0)) for row in baseline) / len(baseline),
			"phase3b_2_f1_mean": sum(float(row.get("f1", 0.0)) for row in intervention) / len(intervention),
		})
	return {
		"paired_case_count": len(paired),
		"counts": {
			"strong_intermediate": {outcome: counts[("strong_intermediate", outcome)] for outcome in ("BOTH_CORRECT", "PHASE2_5_6_ONLY", "PHASE3B_2_ONLY", "BOTH_WRONG")},
			"strong_final": {outcome: counts[("strong_final", outcome)] for outcome in ("BOTH_CORRECT", "PHASE2_5_6_ONLY", "PHASE3B_2_ONLY", "BOTH_WRONG")},
			"control": {outcome: counts[("control", outcome)] for outcome in ("BOTH_CORRECT", "PHASE2_5_6_ONLY", "PHASE3B_2_ONLY", "BOTH_WRONG")},
		},
		"cases": paired,
	}


def routing_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
	output: dict[str, Any] = {}
	for method in METHODS:
		values = [row for row in rows if row.get("method") == method]
		output[method] = {
			"structure_counts": dict(Counter(row.get("router_structure") for row in values)),
			"target_counts": dict(Counter(row.get("canonical_shared_target_type") for row in values)),
			"execution_status_counts": dict(Counter(row.get("execution_status") for row in values)),
			"normalization_status_counts": dict(Counter(row.get("normalization_status") for row in values)),
			"serialization_status_counts": dict(Counter(row.get("serialization_status") for row in values)),
			"validator_decision_counts": dict(Counter(row.get("validator_decision") for row in values)),
			"fallback_runs": sum(bool(row.get("fallback_used")) for row in values),
			"native_branch_attempts": sum(bool(row.get("branch_attempted")) for row in values),
			"native_branch_successes": sum(row.get("execution_status") == "BRANCH_EXECUTION_SUCCESS" for row in values),
		}
	return output


def validator_execution_impact(rows: list[dict[str, Any]]) -> dict[str, Any]:
	output: dict[str, Any] = {}
	for method in METHODS:
		values = [row for row in rows if row.get("method") == method]
		output[method] = {
			"run_count": len(values),
			"validator_pass_rate": sum(row.get("validator_decision") == "PASS" for row in values) / len(values) if values else 0.0,
			"fallback_rate": sum(bool(row.get("fallback_used")) for row in values) / len(values) if values else 0.0,
			"native_branch_attempts": sum(bool(row.get("branch_attempted")) for row in values),
			"native_branch_successes": sum(row.get("execution_status") == "BRANCH_EXECUTION_SUCCESS" for row in values),
			"execution_status_counts": dict(Counter(row.get("execution_status") for row in values)),
		}
	return output


def stability_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
	output: dict[str, Any] = {}
	for method in METHODS:
		values = [row for row in rows if row.get("method") == method]
		by_case = _group_by_case(values)
		target_agree = []
		route_agree = []
		answer_agree = []
		error_free = []
		for case_rows in by_case.values():
			if len(case_rows) < 2:
				continue
			target_agree.append(len({row.get("canonical_shared_target_type") for row in case_rows}) == 1)
			route_agree.append(len({row.get("router_structure") for row in case_rows}) == 1)
			answer_agree.append(len({int(row.get("hits_at_1", 0)) for row in case_rows}) == 1)
			error_free.append(all(not row.get("error") for row in case_rows))
		output[method] = {
			"case_count": len(by_case),
			"target_repeat_agreement": sum(target_agree) / len(target_agree) if target_agree else 0.0,
			"route_repeat_agreement": sum(route_agree) / len(route_agree) if route_agree else 0.0,
			"answer_repeat_agreement": sum(answer_agree) / len(answer_agree) if answer_agree else 0.0,
			"error_free_case_rate": sum(error_free) / len(error_free) if error_free else 0.0,
		}
	return output


def failure_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
	output: list[dict[str, Any]] = []
	for row in rows:
		cause = "NONE"
		if row.get("error"):
			cause = "RUNTIME_ERROR"
		elif not target_correct(row):
			cause = "SHARED_TARGET_EXTRACTION_ERROR"
		elif row.get("normalization_status") in {"PARTIAL_NORMALIZATION", "NORMALIZATION_FAILED"}:
			cause = "NORMALIZATION_ERROR"
		elif row.get("validator_decision") not in {"PASS", "NOT_APPLICABLE"}:
			cause = "VALIDATOR_REJECTION"
		elif row.get("branch_attempted") and row.get("execution_status") != "BRANCH_EXECUTION_SUCCESS":
			cause = "BRANCH_EXECUTION_ERROR"
		elif not row.get("hits_at_1"):
			cause = "ANSWER_ERROR"
		if cause != "NONE":
			output.append({
				"case_id": row.get("case_id"),
				"dataset": row.get("dataset"),
				"cohort": row.get("cohort"),
				"gold_structure": row.get("gold_structure"),
				"method": row.get("method"),
				"repeat": row.get("repeat"),
				"cause": cause,
				"gold_target": row.get("expected_shared_target_type"),
				"predicted_target": row.get("canonical_shared_target_type"),
				"router_structure": row.get("router_structure"),
				"validator_decision": row.get("validator_decision"),
				"execution_status": row.get("execution_status"),
				"fallback_used": row.get("fallback_used"),
				"hits_at_1": row.get("hits_at_1"),
				"f1": row.get("f1"),
				"validation_failures": row.get("validation_failures"),
			})
	return output


def development_summary(
	development_rows: list[dict[str, Any]],
	baseline_rows: list[dict[str, Any]],
	development_cases: list[dict[str, Any]],
) -> dict[str, Any]:
	primary_ids = {
		text(item.get("case_id"))
		for item in read_json(PHASE3B1_ROOT / "aggregate_failure_summary.json").get("case_records", [])
		if item.get("primary_cause") == "SHARED_TARGET_ERROR"
	}
	baseline = [row for row in baseline_rows if row.get("method") == "phase2_5_6"]
	intervention = [row for row in development_rows if row.get("method") == "phase3b_2"]
	base_majority = case_majority(baseline, target_correct)
	int_majority = case_majority(intervention, target_correct)
	primary_corrected = sorted(case_id for case_id in primary_ids if int_majority.get(case_id, False) and not base_majority.get(case_id, False))
	final_cases = [item for item in development_cases if item.get("gold_structure") == "BRANCH_FINAL"]
	final_regressions = sorted(
		item["case_id"] for item in final_cases
		if base_majority.get(item["case_id"], False) and not int_majority.get(item["case_id"], False)
	)
	return {
		"development_case_count": len(development_cases),
		"primary_shared_target_error_case_ids": sorted(primary_ids),
		"primary_shared_target_error_count": len(primary_ids),
		"primary_cases_corrected": primary_corrected,
		"primary_cases_corrected_count": len(primary_corrected),
		"development_sanity_gate": "PASS" if len(primary_corrected) >= 4 and len(final_regressions) <= 1 else "FAIL",
		"final_target_regressions": final_regressions,
		"baseline_target_accuracy_by_case": base_majority,
		"intervention_target_accuracy_by_case": int_majority,
	}


def fresh_decision(semantic: dict[str, Any], paired: dict[str, Any], validator: dict[str, Any], development: dict[str, Any]) -> tuple[str, dict[str, Any]]:
	intermediate_base = semantic["strong_intermediate"]["phase2_5_6"]["shared_target_accuracy_case_majority"]
	intermediate_new = semantic["strong_intermediate"]["phase3b_2"]["shared_target_accuracy_case_majority"]
	final_base = semantic["strong_final"]["phase2_5_6"]["shared_target_accuracy_case_majority"]
	final_new = semantic["strong_final"]["phase3b_2"]["shared_target_accuracy_case_majority"]
	control_false = semantic["control"]["phase3b_2"]["false_intermediate_case_count"]
	intermediate_counts = paired["counts"]["strong_intermediate"]
	qa_net = intermediate_counts["PHASE3B_2_ONLY"] + paired["counts"]["strong_final"]["PHASE3B_2_ONLY"] + paired["counts"]["control"]["PHASE3B_2_ONLY"] - (
		intermediate_counts["PHASE2_5_6_ONLY"] + paired["counts"]["strong_final"]["PHASE2_5_6_ONLY"] + paired["counts"]["control"]["PHASE2_5_6_ONLY"]
	)
	new_failure_counts = Counter(
		row.get("execution_status") for row in []
	)
	baseline_statuses = set(validator["phase2_5_6"]["execution_status_counts"])
	intervention_statuses = set(validator["phase3b_2"]["execution_status_counts"])
	new_systematic = sorted(intervention_statuses - baseline_statuses)
	gates = {
		"development_sanity": development["development_sanity_gate"] == "PASS",
		"intermediate_accuracy_gain_at_least_25_points": intermediate_new - intermediate_base >= 0.25,
		"branch_final_regression_at_most_one_case": final_new >= final_base - (1 / 8),
		"control_false_intermediate_at_most_one_of_eight": control_false <= 1,
		"fresh_paired_qa_net_nonnegative": qa_net >= 0,
		"no_new_systematic_validator_executor_family": len(new_systematic) == 0,
	}
	if not all(gates.values()):
		if gates["intermediate_accuracy_gain_at_least_25_points"] and not gates["branch_final_regression_at_most_one_case"]:
			decision = "SHARED_TARGET_INTERVENTION_REGRESSES"
		elif gates["intermediate_accuracy_gain_at_least_25_points"]:
			decision = "SHARED_TARGET_INTERVENTION_PARTIAL"
		else:
			decision = "SHARED_TARGET_INTERVENTION_NOT_SUPPORTED"
	else:
		decision = "SHARED_TARGET_INTERVENTION_SUPPORTED"
	return decision, {
		"decision": decision,
		"gates": gates,
		"intermediate_accuracy_gain": intermediate_new - intermediate_base,
		"branch_final_accuracy_change": final_new - final_base,
		"control_false_intermediate_case_count": control_false,
		"fresh_paired_qa_net": qa_net,
		"new_validator_executor_status_families": new_systematic,
	}


def manual_review_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Prepare the exact cases required for manual inspection after the run."""
	baseline = {(row["case_id"], int(row["repeat"])): row for row in rows if row["method"] == "phase2_5_6"}
	intervention = [row for row in rows if row["method"] == "phase3b_2"]
	changed_or_intermediate = []
	for row in intervention:
		peer = baseline.get((row["case_id"], int(row["repeat"])))
		if row.get("gold_structure") == "BRANCH_INTERMEDIATE" or (peer and peer.get("canonical_shared_target_type") != row.get("canonical_shared_target_type")):
			changed_or_intermediate.append({
				"case_id": row["case_id"],
				"dataset": row.get("dataset"),
				"cohort": row.get("cohort"),
				"gold_structure": row.get("gold_structure"),
				"repeat": row.get("repeat"),
				"question": row.get("question"),
				"gold_target": row.get("expected_shared_target_type"),
				"phase2_5_6_target": peer.get("canonical_shared_target_type") if peer else None,
				"phase3b_2_target": row.get("canonical_shared_target_type"),
				"shared_target_extraction": row.get("shared_target_extraction"),
				"raw_parsed_result": row.get("raw_parsed_result"),
				"normalization_status": row.get("normalization_status"),
				"validator_decision": row.get("validator_decision"),
				"execution_status": row.get("execution_status"),
				"hits_at_1": row.get("hits_at_1"),
			})
	return changed_or_intermediate


def write_reports(
	manifest: dict[str, Any],
	development: dict[str, Any],
	rows: list[dict[str, Any]],
	development_rows: list[dict[str, Any]],
	baseline_development_rows: list[dict[str, Any]],
	preflight: dict[str, Any],
) -> None:
	baseline_by_key = {(row["case_id"], int(row["repeat"])): row for row in rows if row["method"] == "phase2_5_6"}
	semantic_rows = semantic_records(rows, baseline_by_key)
	semantic = semantic_metrics(rows)
	qa = qa_metrics(rows)
	paired = paired_analysis(rows)
	routing = routing_behavior(rows)
	validator = validator_execution_impact(rows)
	stability = stability_analysis(rows)
	failures = failure_analysis(rows)
	dev_summary = development_summary(development_rows, baseline_development_rows, development["cases"])
	decision, gate_summary = fresh_decision(semantic, paired, validator, dev_summary)
	write_jsonl(OUTPUT_ROOT / "shared_target_extraction_results.jsonl", semantic_rows)
	write_json(OUTPUT_ROOT / "semantic_metrics.json", semantic | {"development": dev_summary})
	write_json(OUTPUT_ROOT / "qa_metrics.json", qa)
	write_json(OUTPUT_ROOT / "paired_analysis.json", paired)
	write_json(OUTPUT_ROOT / "routing_behavior.json", routing)
	write_json(OUTPUT_ROOT / "validator_execution_impact.json", validator)
	write_json(OUTPUT_ROOT / "stability_analysis.json", stability)
	write_jsonl(OUTPUT_ROOT / "failure_analysis.jsonl", failures)
	write_json(OUTPUT_ROOT / "development_summary.json", dev_summary)
	write_json(OUTPUT_ROOT / "decision_gate.json", gate_summary)
	write_json(OUTPUT_ROOT / "manual_review_cases.json", {"status": "REQUIRES_MANUAL_REVIEW", "cases": manual_review_records(rows)})

	decision_text = f"""# Phase 3B.2 decision summary

Decision: **{decision}**

Development sanity gate: `{dev_summary['development_sanity_gate']}`; primary shared-target cases corrected: `{dev_summary['primary_cases_corrected_count']}/{dev_summary['primary_shared_target_error_count']}`.

Fresh validation gates:

```json
{json.dumps(gate_summary, ensure_ascii=False, indent=2)}
```

The frozen downstream stack was not modified. Phase 3C was not run.
"""
	(OUTPUT_ROOT / "decision_summary.md").write_text(decision_text, encoding="utf-8")

	intermediate = semantic["strong_intermediate"]
	final = semantic["strong_final"]
	control = semantic["control"]
	def metric_text(value: Any) -> str:
		return "NA" if value is None else f"{float(value):.3f}"
	report = f"""# Phase 3B.2 Shared-Target Semantic Extraction Intervention

## 1. Motivation

Phase 3B.1 concluded `SEMANTIC_EXTRACTION_BOTTLENECK_SUPPORTED`, with repeated BRANCH_INTERMEDIATE cases extracted as `shared_target=ANSWER`. This phase tests only a targeted semantic extraction intervention.

## 2. Phase 3B.1 causal evidence

The frozen Phase 3B.1 audit contained `{len(development['cases'])}` development cases and identified `{dev_summary['primary_shared_target_error_count']}` independent primary shared-target-error cases across CWQ and GrailQA.

## 3. Research question

Can the intervention distinguish answer-level convergence from intermediate-level convergence without harming BRANCH_FINAL or answer-equivalent control behavior?

## 4. Narrow intervention

The Phase 3B.2 router changes only the extraction prompt and records an explicit convergence-point self-check, `requires_downstream_relation`, and semantic descriptions. No question-specific fresh examples or gold logical forms are exposed.

## 5. Frozen downstream stack

Phase 2.5.5 normalization, Phase 2.5.3B serialization, the existing validator, branch execution, fallback, relation scorer, entity grounding, evaluator, and Freebase backend were reused unchanged.

## 6. Development diagnostic set

The exact frozen 12-case Phase 3B.1 target manifest was reused for regression diagnostics. Development sanity gate: **{dev_summary['development_sanity_gate']}**; corrected primary cases: `{dev_summary['primary_cases_corrected_count']}/{dev_summary['primary_shared_target_error_count']}`; final-target regressions: `{len(dev_summary['final_target_regressions'])}`.

## 7. Fresh validation sampling

The fresh manifest uses seed `{manifest['seed']}` and exactly `{len(manifest['cases'])}` cases: `{manifest['cohort_counts']}`. Sampling was frozen before live execution and excludes prior Phase 2.x/Phase 3 diagnostic and manual-review IDs. Each method has `{manifest['repeats_per_method']}` repeats per case.

## 8. Shared-target semantic metrics

| Cohort | Phase 2.5.6 | Phase 3B.2 |
|---|---:|---:|
| BRANCH_INTERMEDIATE strong | {intermediate['phase2_5_6']['shared_target_accuracy_case_majority']:.3f} | {intermediate['phase3b_2']['shared_target_accuracy_case_majority']:.3f} |
| BRANCH_FINAL strong | {final['phase2_5_6']['shared_target_accuracy_case_majority']:.3f} | {final['phase3b_2']['shared_target_accuracy_case_majority']:.3f} |
| Controls (expected ANSWER) | {control['phase2_5_6']['shared_target_accuracy_case_majority']:.3f} | {control['phase3b_2']['shared_target_accuracy_case_majority']:.3f} |

## 9. BRANCH_INTERMEDIATE results

Phase 2.5.6 target accuracy was `{intermediate['phase2_5_6']['shared_target_accuracy_case_majority']:.3f}` and Phase 3B.2 was `{intermediate['phase3b_2']['shared_target_accuracy_case_majority']:.3f}`. The run-level `requires_downstream_relation` accuracy was `{metric_text(intermediate['phase3b_2']['requires_downstream_relation_accuracy_run'])}` for Phase 3B.2.

## 10. BRANCH_FINAL preservation

Phase 3B.2 BRANCH_FINAL target accuracy was `{final['phase3b_2']['shared_target_accuracy_case_majority']:.3f}`. Target regressions are listed in `development_summary.json` and `failure_analysis.jsonl`.

## 11. Control safety

The Phase 3B.2 control false-intermediate case count was `{control['phase3b_2']['false_intermediate_case_count']}` of 8. The run-level false-intermediate rate was `{control['phase3b_2']['false_intermediate_run_rate']:.3f}`.

## 12. End-to-end QA

| Cohort | Metric | Phase 2.5.6 | Phase 3B.2 |
|---|---|---:|---:|
| Intermediate strong | Hits@1 | {qa['strong_intermediate']['phase2_5_6']['hits_at_1_mean']:.3f} | {qa['strong_intermediate']['phase3b_2']['hits_at_1_mean']:.3f} |
| Intermediate strong | F1 | {qa['strong_intermediate']['phase2_5_6']['f1_mean']:.3f} | {qa['strong_intermediate']['phase3b_2']['f1_mean']:.3f} |
| Final strong | Hits@1 | {qa['strong_final']['phase2_5_6']['hits_at_1_mean']:.3f} | {qa['strong_final']['phase3b_2']['hits_at_1_mean']:.3f} |
| Controls | Hits@1 | {qa['control']['phase2_5_6']['hits_at_1_mean']:.3f} | {qa['control']['phase3b_2']['hits_at_1_mean']:.3f} |

## 13. Validator/fallback impact

| Metric | Phase 2.5.6 | Phase 3B.2 |
|---|---:|---:|
| Validator pass rate | {validator['phase2_5_6']['validator_pass_rate']:.3f} | {validator['phase3b_2']['validator_pass_rate']:.3f} |
| Fallback rate | {validator['phase2_5_6']['fallback_rate']:.3f} | {validator['phase3b_2']['fallback_rate']:.3f} |
| Native branch attempts | {validator['phase2_5_6']['native_branch_attempts']} | {validator['phase3b_2']['native_branch_attempts']} |
| Native branch success | {validator['phase2_5_6']['native_branch_successes']} | {validator['phase3b_2']['native_branch_successes']} |

## 14. Stability

See `stability_analysis.json` for repeat agreement of target type, structure, answer, and error-free execution.

## 15. Failure analysis

`failure_analysis.jsonl` separates shared-target extraction errors from normalization, validator, execution, runtime, and answer failures. No downstream component was patched.

## 16. Manual review

`manual_review_cases.json` enumerates all fresh BRANCH_INTERMEDIATE cases and every changed-target case for manual inspection. Automatic labels and manual labels are kept separate; the generated report does not silently convert automatic adequacy checks into human judgments.

## 17. Decision gate

```json
{json.dumps(gate_summary, ensure_ascii=False, indent=2)}
```

## 18. Final Phase 3B.2 decision

**{decision}**

## 19. Recommended next step

Stop at this phase. Do not modify the validator or executor and do not run Phase 3C. If the semantic gate passes but downstream execution remains limiting, treat that as the next bottleneck rather than combining interventions.

## 20. Limitations

The fresh validation set has 24 cases and two repeats per method. Semantic description adequacy remains a separately auditable field; automatic non-empty checks are not a substitute for human semantic review. The Phase 3A.1 source decision remains `{manifest.get('source_metadata', {}).get('source_manifest_phase3a_1_decision')}`.

## 21. Reproducibility

Required artifacts: `development_case_manifest.json`, `fresh_validation_manifest.json`, `shared_target_extraction_results.jsonl`, `fresh_run_results.jsonl`, `semantic_metrics.json`, `qa_metrics.json`, `paired_analysis.json`, `routing_behavior.json`, `validator_execution_impact.json`, `stability_analysis.json`, `failure_analysis.jsonl`, and `decision_summary.md`.
"""
	(OUTPUT_ROOT / "PHASE3B_2_REPORT.md").write_text(report, encoding="utf-8")
	write_json(OUTPUT_ROOT / "execution_summary.json", {"decision": decision, "preflight": preflight, "fresh_row_count": len(rows), "development_row_count": len(development_rows), "paired_counts": paired["counts"]})


def load_previous_development_baseline() -> list[dict[str, Any]]:
	path = PHASE3B_ROOT / "run_results.jsonl"
	if not path.exists():
		raise SystemExit(f"missing frozen Phase 3B run results: {path}")
	rows = [row for row in read_jsonl(path) if row.get("method") == "phase2_5_6"]
	return rows


def run_intervention_rows(
	method: str,
	selected: list[dict[str, Any]],
	output_path: Path,
	resumed: bool,
	) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	if resumed and output_path.exists():
		rows = read_jsonl(output_path)
	existing = {(row.get("method"), int(row.get("repeat", 0)), row.get("case_id")) for row in rows}
	agent = make_agent(method)
	for repeat in range(1, DEVELOPMENT_REPEATS + 1):
		for index, candidate in enumerate(selected, start=1):
			key = (method, repeat, candidate["case_id"])
			if key in existing:
				continue
			print(f"[{method}] development repeat={repeat} {index}/{len(selected)} {candidate['case_id']}", flush=True)
			rows.append(run_one(agent, method, repeat, candidate))
			existing.add(key)
			write_jsonl(output_path, rows)
	return rows


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--prepare", action="store_true", help="freeze manifests and stop")
	parser.add_argument("--run", action="store_true", help="run development intervention and fresh comparison")
	parser.add_argument("--resume", action="store_true", help="resume existing run files")
	args = parser.parse_args()
	if not args.prepare and not args.run:
		raise SystemExit("choose --prepare or --run")
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	development, development_cases = load_development_manifest()
	write_json(OUTPUT_ROOT / "development_case_manifest.json", development)
	manifest, selected_rows = load_frozen_fresh_manifest()
	if args.prepare and not args.run:
		print(json.dumps({"development_cases": len(development_cases), "fresh_cases": len(selected_rows), "manifest": str(OUTPUT_ROOT / "fresh_validation_manifest.json")}, indent=2))
		return
	preflight = api_preflight()
	write_json(OUTPUT_ROOT / "preflight.json", preflight)
	if not preflight.get("openai_api_key_present") or not preflight.get("model_name") or not preflight.get("openai_available") or not preflight.get("freebase_available"):
		raise SystemExit(f"live preflight failed: {json.dumps(preflight, ensure_ascii=False)}")
	candidates, _ = load_candidates()
	by_id = {candidate["case_id"]: candidate for candidate in candidates}
	development_selected: list[dict[str, Any]] = []
	for item in development_cases:
		candidate = by_id.get(item["case_id"])
		if not candidate:
			raise SystemExit(f"missing candidate metadata for development case {item['case_id']}")
		development_selected.append({**candidate, "cohort": "development"})
	# Keep the gold structure/metadata from the frozen Phase 3B.1 manifest.
	for item, frozen in zip(development_selected, development_cases):
		item["gold_structure"] = frozen["gold_structure"]
	baseline_development_raw = load_previous_development_baseline()
	development_by_id = {item["case_id"]: item for item in development_selected}
	baseline_development: list[dict[str, Any]] = []
	for raw in baseline_development_raw:
		frozen = development_by_id.get(raw.get("case_id"))
		if not frozen:
			continue
		updated = dict(raw)
		updated["cohort"] = "development"
		updated["gold_structure"] = frozen["gold_structure"]
		updated["expected_shared_target_type"] = expected_target(frozen["gold_structure"], "development")
		if not updated.get("canonical_shared_target_type"):
			updated["canonical_shared_target_type"] = target_type(updated.get("canonical_semantic_representation"))
		baseline_development.append(updated)
	development_path = OUTPUT_ROOT / "development_run_results.jsonl"
	development_rows = run_intervention_rows("phase3b_2", development_selected, development_path, args.resume)
	# The fresh run file contains both methods and is resumable per method/repeat/case.
	fresh_path = OUTPUT_ROOT / "fresh_run_results.jsonl"
	fresh_rows: list[dict[str, Any]] = read_jsonl(fresh_path) if args.resume and fresh_path.exists() else []
	selected_by_id = {item["case_id"]: item for item in selected_rows}
	existing = {(row.get("method"), int(row.get("repeat", 0)), row.get("case_id")) for row in fresh_rows}
	for method in METHODS:
		agent = make_agent(method)
		for repeat in range(1, FRESH_REPEATS + 1):
			for index, candidate in enumerate(selected_rows, start=1):
				key = (method, repeat, candidate["case_id"])
				if key in existing:
					continue
				print(f"[{method}] fresh repeat={repeat} {index}/{len(selected_rows)} {candidate['case_id']}", flush=True)
				fresh_rows.append(run_one(agent, method, repeat, {**candidate, "cohort": candidate["cohort"]}))
				existing.add(key)
				write_jsonl(fresh_path, fresh_rows)
		fresh_rows.sort(key=lambda row: (row["method"], int(row["repeat"]), row["case_id"]))
	write_jsonl(fresh_path, fresh_rows)
	expected = {(method, repeat, case_id) for method in METHODS for repeat in range(1, FRESH_REPEATS + 1) for case_id in selected_by_id}
	observed = {(row.get("method"), int(row.get("repeat", 0)), row.get("case_id")) for row in fresh_rows}
	if expected - observed:
		missing = sorted(expected - observed)
		raise SystemExit(f"fresh run incomplete: {len(missing)} rows missing; first={missing[:5]}")
	write_reports(manifest, development, fresh_rows, development_rows, baseline_development, preflight)
	print(json.dumps({"status": "COMPLETE", "fresh_rows": len(fresh_rows), "development_rows": len(development_rows), "output_root": str(OUTPUT_ROOT)}, indent=2))


if __name__ == "__main__":
	main()
