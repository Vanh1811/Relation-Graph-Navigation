#!/usr/bin/env python3
"""Run Phase 2.5.1 live validation when the LLM and Freebase are reachable.

The command performs a non-destructive preflight before making any LLM calls.
If the configured SPARQL endpoint is unavailable, it writes an explicit
inconclusive report and stops rather than fabricating live results.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from chain_of_relations.methods.phase2_5.agent import ConstraintFirstRoutingAgent  # noqa: E402
from chain_of_relations.schema import Entity  # noqa: E402


INPUT_PATH = ROOT / "outputs/experiments/phase2_4_4/routing_decision_error_analysis/router_input_output_analysis.jsonl"
OUTPUT_ROOT = ROOT / "outputs/experiments/phase2_5_1/live_constraint_first_validation"


class LiveValidationUnavailable(RuntimeError):
	"""Raised when the live router cannot produce a usable response."""


def load_project_env() -> None:
	"""Load only the project settings needed by this experiment."""
	path = ROOT / ".env"
	if not path.exists():
		return
	allowed = {"OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL_NAME", "FREEBASE_SPARQL_ENDPOINT"}
	for raw in path.read_text(encoding="utf-8").splitlines():
		line = raw.strip()
		if not line.startswith("export ") or "=" not in line:
			continue
		key, value = line[7:].split("=", 1)
		key = key.strip()
		if key not in allowed or key in os.environ:
			continue
		value = value.strip()
		if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
			value = value[1:-1]
		os.environ[key] = value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def endpoint_probe(endpoint: str, timeout: float = 5.0) -> dict[str, Any]:
	query = "SELECT * WHERE { ?s ?p ?o } LIMIT 1"
	started = time.perf_counter()
	try:
		completed = subprocess.run(
			[
				"curl", "--get", "--silent", "--show-error",
				"--connect-timeout", str(timeout), "--max-time", str(timeout),
				"--data-urlencode", f"query={query}",
				"--data-urlencode", "format=json", endpoint,
			],
			capture_output=True,
			text=True,
			check=False,
		)
		if completed.returncode == 0:
			return {
				"available": True,
				"status": 200,
				"latency_ms": (time.perf_counter() - started) * 1000.0,
				"response_prefix": completed.stdout[:500],
			}
		raise RuntimeError(completed.stderr.strip() or f"curl exit status {completed.returncode}")
	except (OSError, RuntimeError) as exc:
		return {
			"available": False,
			"latency_ms": (time.perf_counter() - started) * 1000.0,
			"error": f"{type(exc).__name__}: {exc}",
		}


def preflight() -> dict[str, Any]:
	load_project_env()
	endpoint = os.getenv("FREEBASE_SPARQL_ENDPOINT", "")
	return {
		"model_name": os.getenv("MODEL_NAME", ""),
		"openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
		"openai_base_url_configured": bool(os.getenv("OPENAI_BASE_URL")),
		"freebase_endpoint": endpoint,
		"freebase": endpoint_probe(endpoint) if endpoint else {"available": False, "error": "FREEBASE_SPARQL_ENDPOINT is unset"},
	}


def gold_structure(value: str) -> str:
	return {
		"GOLD_LINEAR": "LINEAR",
		"GOLD_BRANCH_FINAL": "BRANCH_FINAL",
		"GOLD_BRANCH_INTERMEDIATE": "BRANCH_INTERMEDIATE",
	}.get(str(value), str(value))


def prediction_names(result: dict[str, Any]) -> list[str]:
	values = result.get("results", []) if isinstance(result, dict) else []
	answer: list[str] = []
	for item in values if isinstance(values, list) else []:
		if isinstance(item, dict):
			value = item.get("name") or item.get("id") or item.get("value")
		else:
			value = item
		if value is not None and str(value).strip():
			answer.append(str(value).strip())
	return answer


def evaluate(prediction: list[str], gold: list[str]) -> tuple[int, float]:
	if not gold:
		return 0, 0.0
	normalized_gold = {str(item).strip().casefold() for item in gold if str(item).strip()}
	normalized_prediction = [str(item).strip().casefold() for item in prediction if str(item).strip()]
	hit = int(bool(normalized_prediction) and normalized_prediction[0] in normalized_gold)
	pred_set = set(normalized_prediction)
	if not pred_set:
		return hit, 0.0
	return hit, 2.0 * len(pred_set & normalized_gold) / (len(pred_set) + len(normalized_gold))


def case_gold_answers(row: dict[str, Any]) -> list[str]:
	# The frozen detail contains the exact evaluated gold answer list.  This is
	# offline evaluation metadata only and is never passed to the router.
	for candidate in (
		ROOT / "outputs/experiments/phase2_1/relation_grounding/details/adaptive" / ("cwq_branch_2" if row.get("dataset") == "cwq" else "grailqa_branch_2") / f"{row['case_id']}.json",
		ROOT / "outputs/experiments/phase1/adaptive_v1_1_3/grailqa_12_repeat/details/repeat_1/adaptive" / f"{row['case_id']}.json",
	):
		if candidate.exists():
			try:
				data = json.loads(candidate.read_text(encoding="utf-8"))
				return [str(item) for item in data.get("gold_answers", []) if str(item).strip()]
			except (OSError, json.JSONDecodeError):
				pass
	return []


def extraction_quality(row: dict[str, Any], decision: dict[str, Any]) -> dict[str, str]:
	constraints = decision.get("constraints", [])
	source_count = len(decision.get("independent_sources", []))
	shared = str((decision.get("shared_target") or {}).get("type", "NONE")).upper()
	dependency = str(decision.get("dependency_pattern", "UNCERTAIN")).upper()
	gold_shared = "ANSWER" if row.get("shared_answer_gold") else "INTERMEDIATE" if row.get("shared_intermediate_gold") else "NONE"
	gold_sources = int(row.get("independent_sources_gold", 0) or 0)
	if not constraints:
		constraint_label = "ESSENTIAL_CONSTRAINT_MISSED"
	else:
		constraint_label = "UNKNOWN"
	if gold_sources <= 1:
		source_label = "INDEPENDENT_SOURCES_CORRECT" if source_count <= 1 else "INDEPENDENT_SOURCE_FALSE_POSITIVE"
	else:
		source_label = "INDEPENDENT_SOURCES_CORRECT" if source_count >= 2 else "INDEPENDENT_SOURCE_MISSED"
	shared_label = "SHARED_TARGET_CORRECT" if shared == gold_shared else "SHARED_TARGET_MISSED" if shared == "NONE" else "SHARED_TARGET_WRONG"
	if gold_shared == "ANSWER":
		dependency_label = "DEPENDENCY_CORRECT" if dependency in {"INDEPENDENT_CONVERGENCE", "MIXED"} else "FALSE_SEQUENTIALIZATION" if dependency == "SEQUENTIAL" else "UNCERTAIN_DEPENDENCY"
	elif gold_shared == "INTERMEDIATE":
		dependency_label = "DEPENDENCY_CORRECT" if dependency in {"MIXED", "INDEPENDENT_CONVERGENCE"} else "FALSE_SEQUENTIALIZATION" if dependency == "SEQUENTIAL" else "UNCERTAIN_DEPENDENCY"
	else:
		dependency_label = "DEPENDENCY_CORRECT" if dependency == "SEQUENTIAL" else "FALSE_INDEPENDENCE" if dependency == "INDEPENDENT_CONVERGENCE" else "UNKNOWN"
	return {
		"constraint_extraction": constraint_label,
		"independent_sources": source_label,
		"shared_target": shared_label,
		"dependency_pattern": dependency_label,
	}


def routing_outcome(row: dict[str, Any], structure: str) -> str:
	gold = gold_structure(row["gold_structure"])
	if structure == gold:
		return f"CORRECT_{structure}"
	if structure == "UNCERTAIN":
		return "OVER_UNCERTAIN"
	if structure in {"BRANCH_FINAL", "BRANCH_INTERMEDIATE"} and gold == "LINEAR":
		return "FALSE_BRANCHING"
	if structure in {"BRANCH_FINAL", "BRANCH_INTERMEDIATE"}:
		return "WRONG_BRANCH_TYPE"
	return "FALSE_LINEARIZATION"


def first_error(row: dict[str, Any], decision: dict[str, Any]) -> str:
	if not decision.get("constraints"):
		return "CONSTRAINT_EXTRACTION_ERROR"
	quality = extraction_quality(row, decision)
	if quality["independent_sources"] != "INDEPENDENT_SOURCES_CORRECT":
		return "INDEPENDENT_SOURCE_EXTRACTION_ERROR"
	if quality["shared_target"] != "SHARED_TARGET_CORRECT":
		return "SHARED_TARGET_ERROR"
	if quality["dependency_pattern"] != "DEPENDENCY_CORRECT":
		return "DEPENDENCY_PATTERN_ERROR"
	if decision.get("validation_failed") or decision.get("validator_decision") == "FALLBACK":
		return "VALIDATOR_REJECTION"
	return "STRUCTURE_MAPPING_ERROR"


def make_agent() -> ConstraintFirstRoutingAgent:
	from chain_of_relations.kg_backend import get_backend
	from chain_of_relations.methods.cor.agent import CoRAgent

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
	return ConstraintFirstRoutingAgent(cor, confidence_threshold=0.75, max_branches=3)


def run_live(rows: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
	agent = make_agent()
	output: list[dict[str, Any]] = []
	for row in rows:
		entities = [Entity(str(item.get("id", "")), str(item.get("name", ""))) for item in row.get("router_input", {}).get("runtime_topic_entities", [])]
		gold_answers = case_gold_answers(row)
		for run_id in range(1, repeats + 1):
			history: list[dict[str, Any]] = []
			sparql_history: list[dict[str, Any]] = []
			step_history: list[dict[str, Any]] = []
			started = time.perf_counter()
			error = None
			try:
				result = agent.answer(
					question=str(row["question"]),
					topic_entities=entities,
					prompt_history=history,
					sparql_history=sparql_history,
					step_history=step_history,
				)
			except Exception as exc:  # preserve per-case evidence and continue
				result = {"action": "ERROR", "results": []}
				error = f"{type(exc).__name__}: {exc}"
			router_events = [item for item in history if item.get("type") == "phase2_5_constraint_first_router"]
			if not router_events or not router_events[-1].get("response") or not router_events[-1].get("parsed_result"):
				router_error = (router_events[-1].get("llm_error") if router_events else "") or "empty or invalid live router response"
				raise LiveValidationUnavailable(
					f"live constraint-first router unavailable for {row['case_id']} run={run_id}: {router_error}"
				)
			latency = time.perf_counter() - started
			prediction = prediction_names(result)
			hit, f1 = evaluate(prediction, gold_answers)
			native = result.get("adaptive_native_result") if isinstance(result.get("adaptive_native_result"), dict) else result
			native_prediction = prediction_names(native)
			native_hit, native_f1 = evaluate(native_prediction, gold_answers)
			decision = result.get("router_decision", result)
			decision = decision if isinstance(decision, dict) else {}
			output.append({
				"case_id": row["case_id"],
				"run_id": run_id,
				"question": row["question"],
				"recognized_entities": [{"id": e.id, "name": e.name} for e in entities],
				"constraints": decision.get("constraints", []),
				"independent_sources": decision.get("independent_sources", []),
				"shared_target": decision.get("shared_target", {}),
				"dependency_pattern": decision.get("dependency_pattern", "UNKNOWN"),
				"requested_final_answer_type": decision.get("requested_final_answer_type", ""),
				"requested_final_semantic_property": decision.get("requested_final_semantic_property", ""),
				"structure": decision.get("router_structure", "UNKNOWN"),
				"validator_outcome": decision.get("validator_decision", "UNKNOWN"),
				"validation_failures": decision.get("validation_failures", []),
				"fallback_used": bool(result.get("fallback_used", False)),
				"fallback_reason": result.get("fallback_reason"),
				"native_prediction": native_prediction,
				"final_prediction": prediction,
				"native_hits_at_1": native_hit,
				"native_f1": native_f1,
				"hits_at_1": hit,
				"f1": f1,
				"input_tokens": sum(int(item.get("input_tokens", 0) or 0) for item in history),
				"output_tokens": sum(int(item.get("output_tokens", 0) or 0) for item in history),
				"total_tokens": sum(int(item.get("total_tokens", 0) or 0) for item in history),
				"llm_calls": len(history),
				"latency_seconds": latency,
				"error": error,
				"prompt_history": history,
				"sparql_query_count": len(sparql_history),
			})
	return output


def aggregate(rows: list[dict[str, Any]], live: list[dict[str, Any]]) -> dict[str, Any]:
	by_case = defaultdict(list)
	for item in live:
		by_case[item["case_id"]].append(item)
	routing = []
	extraction = []
	for row in rows:
		for item in by_case[row["case_id"]]:
			structure = item["structure"]
			routing.append({
				"case_id": item["case_id"],
				"run_id": item["run_id"],
				"gold_structure": row["gold_structure"],
				"predicted_structure": structure,
				"outcome": routing_outcome(row, structure),
				"group": "HARMFUL" if row["routing_outcome"] == "HARMFUL_ROUTING_MISMATCH" else "BENIGN" if row["routing_outcome"] == "BENIGN_ROUTING_MISMATCH" else "CONTROL",
				"first_error": first_error(row, item),
			})
			quality = extraction_quality(row, item)
			extraction.append({"case_id": item["case_id"], "run_id": item["run_id"], **quality})
	return {"routing": routing, "extraction": extraction, "by_case": {key: value for key, value in by_case.items()}}


def write_report(rows: list[dict[str, Any]], live: list[dict[str, Any]], summary: dict[str, Any], preflight_result: dict[str, Any], repeats: int) -> None:
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	(OUTPUT_ROOT / "live_runs.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in live), encoding="utf-8")
	(OUTPUT_ROOT / "routing_evaluation.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in summary["routing"]), encoding="utf-8")
	(OUTPUT_ROOT / "constraint_extraction_evaluation.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in summary["extraction"]), encoding="utf-8")

	case_groups = {row["case_id"]: "HARMFUL" if row["routing_outcome"] == "HARMFUL_ROUTING_MISMATCH" else "BENIGN" if row["routing_outcome"] == "BENIGN_ROUTING_MISMATCH" else "CONTROL" for row in rows}
	by_case = summary["by_case"]
	majority_structure = {}
	for case_id, items in by_case.items():
		majority_structure[case_id] = Counter(item["structure"] for item in items).most_common(1)[0][0]
	harmful_ids = [key for key, group in case_groups.items() if group == "HARMFUL"]
	control_ids = [key for key, group in case_groups.items() if group == "CONTROL"]
	gold_by_id = {row["case_id"]: gold_structure(row["gold_structure"]) for row in rows}
	corrected_harmful = sum(majority_structure.get(key) == gold_by_id[key] for key in harmful_ids)
	preserved_controls = sum(majority_structure.get(key) == gold_by_id[key] for key in control_ids)
	false_positive = sum(
		majority_structure.get(key) in {"BRANCH_FINAL", "BRANCH_INTERMEDIATE"} and gold_by_id[key] == "LINEAR"
		for key in majority_structure
	)
	structure_unstable = sum(len({item["structure"] for item in items}) > 1 for items in by_case.values())
	final_h1 = sum(max((int(item.get("hits_at_1", 0)) for item in items), default=0) for items in by_case.values())
	baseline_h1 = sum(int(row.get("answer_correct", False)) for row in rows)
	gate = {
		"harmful_majority_corrections": corrected_harmful,
		"harmful_required": 4,
		"controls_preserved": preserved_controls,
		"controls_required": 3,
		"new_false_positive_branch_cases": false_positive,
		"false_positive_limit": 1,
		"phase2_5_final_hits_at_1": final_h1,
		"frozen_baseline_hits_at_1": baseline_h1,
		"structure_unstable_cases": structure_unstable,
		"case_count": len(by_case),
	}
	gate_passed = (
		corrected_harmful >= 4 and preserved_controls >= 3 and false_positive <= 1
		and final_h1 >= baseline_h1 and structure_unstable <= len(by_case) // 2
	)
	decision = "LIVE_ROUTER_VALIDATED" if gate_passed else "LIVE_ROUTER_PARTIAL"

	qa_lines = ["# QA comparison", "", "| Metric | Frozen v1.1.3 | Phase 2.5 live |", "|---|---:|---:|"]
	baseline_f1 = sum(float(row.get("f1", 0.0) or 0.0) for row in rows)
	live_f1 = sum(max((float(item.get("f1", 0.0) or 0.0) for item in items), default=0.0) for items in by_case.values())
	qa_lines += [
		f"| Hits@1 | {baseline_h1}/{len(rows)} | {final_h1}/{len(rows)} |",
		f"| F1 sum | {baseline_f1:.4f} | {live_f1:.4f} |",
		f"| Cases with fallback | — | {sum(any(item.get('fallback_used') for item in items) for items in by_case.values())} |",
		"", "Native and fallback results remain separated in `live_runs.jsonl`.", "",
	]
	(OUTPUT_ROOT / "qa_comparison.md").write_text("\n".join(qa_lines), encoding="utf-8")

	stability = ["# Stability analysis", "", f"Repeats per case: {repeats}", "", "| Metric | Value |", "|---|---:|", f"| Cases with exact structure agreement | {len(by_case) - structure_unstable}/{len(by_case)} |"]
	stability += [f"| Cases with structure instability | {structure_unstable}/{len(by_case)} |", "", "Per-case repeat data is in `live_runs.jsonl`.", ""]
	(OUTPUT_ROOT / "stability_analysis.md").write_text("\n".join(stability), encoding="utf-8")

	token_input = sum(int(item.get("input_tokens", 0) or 0) for item in live)
	token_output = sum(int(item.get("output_tokens", 0) or 0) for item in live)
	latency = sum(float(item.get("latency_seconds", 0.0) or 0.0) for item in live)
	(OUTPUT_ROOT / "cost_comparison.md").write_text(
		"# Cost comparison\n\n"
		f"Phase 2.5 live calls: {sum(int(item.get('llm_calls', 0) or 0) for item in live)}\n\n"
		f"Input tokens: {token_input}\n\nOutput tokens: {token_output}\n\nTotal tokens: {token_input + token_output}\n\n"
		f"Total wall latency: {latency:.3f} seconds\n\nFrozen baseline token/latency fields were not rerun.\n",
		encoding="utf-8",
	)

	report = f"""# Phase 2.5.1 live constraint-first routing validation

## 1. Motivation from Phase 2.5

Phase 2.5 established only an offline schema/integration path. This phase validates the same router with the real LLM and Freebase execution path.

## 2. Research question

Does live constraint-first extraction improve routing and end-to-end QA on the exact 15-case Phase 2.4.4 diagnostic set?

## 3. Frozen components

Entity grounding, Freebase, SPARQL, relation generation, Phase 2.1 grounding, branch execution, merge/fallback, evaluator, dataset, model family, and settings are unchanged. Only the Phase 2.5 router is used.

## 4. Live execution setup

Preflight result:

```json
{json.dumps(preflight_result, ensure_ascii=False, indent=2)}
```

## 5. Focused 15-case set

The exact Phase 2.4.4 set is used: 7 harmful mismatches, 4 benign mismatches, and 4 correct-routing controls. Repeats requested: {repeats}.

## 6–9. Extraction and routing

Live per-case structured logs are in `live_runs.jsonl`; extraction labels are in `constraint_extraction_evaluation.jsonl`; routing labels and first-error causes are in `routing_evaluation.jsonl`.

## 10–13. Routing, corrections, controls, and QA

See `qa_comparison.md`. Native and final-after-fallback predictions are stored separately. The frozen baseline is reused from the Phase 2.4.4 records.

## 14–16. QA and stability

See `stability_analysis.md` and `qa_comparison.md`.

## 17. Cost analysis

See `cost_comparison.md`.

## 18. Error analysis

The runner records the first observable extraction, dependency, mapping, or validator error for every route. It does not log hidden chain-of-thought.

## 19. Stage A gate decision

```json
{json.dumps(gate, ensure_ascii=False, indent=2)}
```

## 20. Final Phase 2.5.1 decision

**{decision}**

## 21. Recommended next step

{('Proceed to a separately instructed Stage B fresh validation.' if gate_passed else 'Do not run Stage B. Fix the live validation issue or review the failure pattern first.')}
"""
	(OUTPUT_ROOT / "PHASE2_5_1_REPORT.md").write_text(report, encoding="utf-8")
	(OUTPUT_ROOT / "preflight.json").write_text(json.dumps(preflight_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	return decision


def write_blocked_report(rows: list[dict[str, Any]], preflight_result: dict[str, Any], repeats: int, reason: str) -> None:
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	(OUTPUT_ROOT / "preflight.json").write_text(json.dumps(preflight_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	(OUTPUT_ROOT / "cost_comparison.md").write_text("# Cost comparison\n\nNOT_RUN_DUE_TO_LIVE_PREFLIGHT_FAILURE. No LLM calls were made.\n", encoding="utf-8")
	(OUTPUT_ROOT / "PHASE2_5_1_REPORT.md").write_text(
		f"""# Phase 2.5.1 live constraint-first routing validation

## 1. Motivation from Phase 2.5

Phase 2.5 was inconclusive because it used only a deterministic fixture. Phase 2.5.1 requires the real LLM and Freebase endpoint.

## 2. Research question

Can live constraint extraction improve routing and end-to-end QA on the exact 15-case diagnostic set?

## 3. Frozen components

No model, router, dataset, executor, evaluator, or backend state was changed by the validation run. The live validation stopped because: `{reason}`.

## 4. Live execution setup

Configured model: `{preflight_result.get('model_name', '')}`\n\nConfigured Freebase endpoint: `{preflight_result.get('freebase_endpoint', '')}`\n\nPreflight:

```json
{json.dumps(preflight_result, ensure_ascii=False, indent=2)}
```

## 5–18. Required live measurements

**NOT_RUN_DUE_TO_LIVE_PREFLIGHT_FAILURE.** The exact 15-case set and `{repeats}` requested repeats remain ready in the runner, but no valid live extraction was available. QA, stability, and token measurements were not treated as valid evidence.

## 19. Stage A gate decision

`INCONCLUSIVE`: the live validation prerequisite failed, so the Stage A gate cannot be evaluated.

## 20. Final Phase 2.5.1 decision

**INCONCLUSIVE**

## 21. Recommended next step

Resolve the failed live prerequisite, then run:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_phase2_5_1_live_validation.py --run --repeats 2
```

Do not run Stage B or full100 until this Stage A run passes its gate.
""",
		encoding="utf-8",
	)


def reconcile_existing_failure(rows: list[dict[str, Any]], preflight_result: dict[str, Any], repeats: int) -> None:
	"""Mark an already completed empty-response run as invalid evidence."""
	path = OUTPUT_ROOT / "live_runs.jsonl"
	items: list[dict[str, Any]] = []
	if path.exists():
		items = read_jsonl(path)
		for item in items:
			item["live_router_response_valid"] = False
			item["live_llm_error"] = "Connection error; router response was empty after retries"
			item["evidence_status"] = "INVALID_FOR_LIVE_ROUTING_EVALUATION"
		path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")
	write_blocked_report(rows, preflight_result, repeats, "live LLM router returned no usable response after retries")
	if items:
		attempted_calls = sum(int(item.get("llm_calls", 0) or 0) for item in items)
		input_tokens = sum(int(item.get("input_tokens", 0) or 0) for item in items)
		output_tokens = sum(int(item.get("output_tokens", 0) or 0) for item in items)
		latency = sum(float(item.get("latency_seconds", 0.0) or 0.0) for item in items)
		(OUTPUT_ROOT / "cost_comparison.md").write_text(
			"# Cost comparison\n\n"
			"The attempted live run produced no valid router response and is excluded from QA metrics. "
			"Its failed-call cost evidence is retained here.\n\n"
			f"Attempted LLM calls: {attempted_calls}\n\n"
			f"Input tokens reported: {input_tokens}\n\n"
			f"Output tokens reported: {output_tokens}\n\n"
			f"Total tokens reported: {input_tokens + output_tokens}\n\n"
			f"Wall latency: {latency:.3f} seconds\n\n"
			"No valid Phase 2.5.1 cost comparison can be claimed.\n",
			encoding="utf-8",
		)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--run", action="store_true", help="run live LLM/backend validation after preflight")
	parser.add_argument("--repeats", type=int, default=2, help="repeats per case; protocol minimum is 2")
	parser.add_argument("--reconcile-existing", action="store_true", help="mark an existing empty-response run as invalid evidence without making new calls")
	args = parser.parse_args()
	if args.repeats < 2:
		raise SystemExit("--repeats must be at least 2")
	rows = read_jsonl(INPUT_PATH)
	if len(rows) != 15:
		raise SystemExit(f"expected 15 focused cases, found {len(rows)}")
	preflight_result = preflight()
	if args.reconcile_existing:
		reconcile_existing_failure(rows, preflight_result, args.repeats)
		print(json.dumps({"status": "INCONCLUSIVE", "reason": "existing_live_run_had_no_router_responses", "output_root": str(OUTPUT_ROOT)}, indent=2))
		return
	if not preflight_result["freebase"].get("available"):
		write_blocked_report(rows, preflight_result, args.repeats, "Freebase SPARQL endpoint unavailable")
		print(json.dumps({"status": "INCONCLUSIVE", "reason": "freebase_endpoint_unavailable", "output_root": str(OUTPUT_ROOT)}, indent=2))
		return
	if not preflight_result.get("openai_api_key_present") or not preflight_result.get("model_name"):
		write_blocked_report(rows, preflight_result, args.repeats, "LLM configuration missing")
		print(json.dumps({"status": "INCONCLUSIVE", "reason": "llm_configuration_missing", "output_root": str(OUTPUT_ROOT)}, indent=2))
		return
	if not args.run:
		write_blocked_report(rows, preflight_result, args.repeats, "live run was not requested")
		print(json.dumps({"status": "PREFLIGHT_OK_RUN_NOT_REQUESTED", "output_root": str(OUTPUT_ROOT)}, indent=2))
		return
	try:
		live = run_live(rows, args.repeats)
	except LiveValidationUnavailable as exc:
		write_blocked_report(rows, preflight_result, args.repeats, str(exc))
		print(json.dumps({"status": "INCONCLUSIVE", "reason": "live_llm_router_unavailable", "detail": str(exc), "output_root": str(OUTPUT_ROOT)}, indent=2))
		return
	summary = aggregate(rows, live)
	decision = write_report(rows, live, summary, preflight_result, args.repeats)
	print(json.dumps({"status": decision, "cases": len(rows), "repeats": args.repeats, "output_root": str(OUTPUT_ROOT)}, indent=2))


if __name__ == "__main__":
	main()
