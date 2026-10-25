#!/usr/bin/env python3
"""Run the offline Phase 2.5 focused routing diagnostic.

This stage exercises the new constraint-first schema against the frozen
Phase 2.4.4 15-case trace set.  It does not call an LLM, Virtuoso, or the
answer executor: the deterministic fixture generator is a schema/integration
check while the live backend is unavailable.  The report therefore does not
claim an end-to-end Phase 2.5 improvement or advance to Stage B.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chain_of_relations.methods.phase2_5.router import ConstraintFirstRouter  # noqa: E402
from chain_of_relations.schema import Entity  # noqa: E402


INPUT_PATH = ROOT / "outputs/experiments/phase2_4_4/routing_decision_error_analysis/router_input_output_analysis.jsonl"
OUTPUT_ROOT = ROOT / "outputs/experiments/phase2_5/constraint_first_routing"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def answer_type(question: str) -> str:
	text = question.casefold()
	patterns = (
		(r"\bwhat country\b", "country"),
		(r"\bwhat musical release\b", "musical release"),
		(r"\bwhat college\b", "college"),
		(r"\bwhat book edition series\b", "book edition series"),
		(r"\bwhat celebrity\b", "person"),
		(r"\bwhich wine\b", "wine"),
		(r"\bfictional universes\b", "fictional universe"),
		(r"\bpolitical ideology\b", "political ideology"),
		(r"\boccupation\b", "occupation"),
		(r"\bwhat is the name\b", "name"),
		(r"\bwho\b", "person"),
		(r"\bwhat are\b", "entity"),
		(r"\bwhat\b", "entity"),
		(r"\bwhich\b", "entity"),
	)
	for pattern, value in patterns:
		if re.search(pattern, text):
			return value
	return "entity"


def semantic_property(question: str, target: str) -> str:
	text = question.casefold()
	if "occupation" in text:
		return "occupation / profession"
	if "political ideology" in text:
		return "political ideology"
	if "appellation" in text:
		return "wine appellation"
	if "book edition series" in text:
		return "book edition series"
	if "musical release" in text:
		return "musical release"
	if "fictional universes" in text:
		return "fictional-universe membership"
	if "discipline" in text or "social sciences" in text:
		return "research discipline"
	if "country" in text:
		return "country satisfying the stated constraints"
	if "brothers" in text:
		return "sibling relationship"
	return f"{target} constraint"


def _entity_label(entity: Entity) -> str:
	return entity.name.strip() or entity.id


def build_offline_payload(question: str, entities: list[Entity]) -> dict[str, Any]:
	"""Create a model-shaped fixture without using gold graph information.

	The fixture is intentionally lexical and runtime-entity-only.  It is not a
	replacement for live model evaluation; it checks that the new schema,
	structure mapping, validator, and logs are wired correctly.
	"""
	text = question.casefold()
	target = answer_type(question)
	prop = semantic_property(question, target)
	intermediate = any(token in text for token in (
		"who are the brothers", "author who", "artist of", "name in", "celebrity who",
	))
	constraints: list[dict[str, Any]] = []
	sources: list[str] = []

	for index, entity in enumerate(entities, start=1):
		label = _entity_label(entity)
		is_type_like = label.casefold() in {"male", "female", "anti-apartheid", "school", "phd"}
		role = "TYPE" if is_type_like else "INTERMEDIATE" if intermediate else "ANSWER"
		constraints.append({
			"constraint_id": f"c{index}",
			"text": f"The requested {target} is constrained by {label}.",
			"source_entity": str(entity.id),
			"target_role": role,
			"semantic_property": prop,
			"depends_on": [f"c{index - 1}"] if intermediate and index > 1 else [],
		})
		sources.append(label)

	# Explicit literal/date/value conditions are answer-relevant sources even
	# when they have no linked topic entity.  This handles the exact class of
	# failure diagnosed in the Phase 2.4.4 Caribbean example.
	has_literal = bool(re.search(r"\b(?:19|20)\d{2}\b|\b\d+\.\d+\b", text))
	if has_literal and not intermediate:
		constraints.append({
			"constraint_id": f"c{len(constraints) + 1}",
			"text": "An explicit date, rate, or numeric value constrains the answer.",
			"source_entity": "literal/date/value",
			"target_role": "FILTER",
			"semantic_property": prop,
			"depends_on": [],
		})
		sources.append("literal/date/value")

	# A one-anchor question with a clear relational clause still has a second
	# answer-relevant constraint, even if the other source is not grounded.
	if len(constraints) == 1 and not intermediate and any(token in text for token in ("contains", "has", "with", "where", "rate")):
		constraints.append({
			"constraint_id": "c2",
			"text": "The question's explicit relational or value clause constrains the answer.",
			"source_entity": "question-clause",
			"target_role": "FILTER",
			"semantic_property": prop,
			"depends_on": [],
		})
		sources.append("question-clause")

	if intermediate:
		shared_target = {"type": "INTERMEDIATE", "description": "constraints converge on an intermediate before the final answer"}
		dependency = "MIXED"
		structure = "BRANCH_INTERMEDIATE"
	elif len(constraints) >= 2:
		shared_target = {"type": "ANSWER", "description": f"all constraints converge on the final {target} answer"}
		dependency = "INDEPENDENT_CONVERGENCE"
		structure = "BRANCH_FINAL"
	else:
		shared_target = {"type": "NONE", "description": "no supported convergence target"}
		dependency = "SEQUENTIAL"
		structure = "LINEAR"

	branches: list[dict[str, Any]] = []
	if structure == "BRANCH_FINAL":
		for index, constraint in enumerate(constraints, start=1):
			entity = entities[min(index - 1, len(entities) - 1)] if entities else Entity("", "")
			branches.append({
				"branch_id": f"b{index}",
				"root_entity_id": str(entity.id),
				"root_entity_name": _entity_label(entity),
				"constraint_property": constraint["semantic_property"],
				"constraint_summary": constraint["text"],
				"target_answer_type": target,
			})

	return {
		"constraints": constraints,
		"independent_sources": sources,
		"shared_target": shared_target,
		"dependency_pattern": dependency,
		"structure": structure,
		"execution_mode": "branch" if structure == "BRANCH_FINAL" else "cor",
		"confidence": 0.95,
		"reason": "Offline fixture: explicit constraints were represented before structure classification.",
		"requested_final_answer_type": target,
		"requested_final_semantic_property": prop,
		"shared_answer_type": target,
		"branches": branches,
		"unsupported_structure": structure == "BRANCH_INTERMEDIATE",
	}


def _route_with_payload(question: str, entities: list[Entity], payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
	history: list[dict[str, Any]] = []

	def generate(*_args: Any, **_kwargs: Any) -> tuple[str, dict[str, int]]:
		return json.dumps(payload), {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

	decision = ConstraintFirstRouter(generate).route(question, entities, history)
	return decision.as_dict(), history


def outcome(gold: str, predicted: str) -> str:
	gold_map = {
		"GOLD_LINEAR": "LINEAR",
		"GOLD_BRANCH_FINAL": "BRANCH_FINAL",
		"GOLD_BRANCH_INTERMEDIATE": "BRANCH_INTERMEDIATE",
	}
	if predicted == gold_map.get(gold, gold):
		return f"CORRECT_{predicted}"
	if predicted in {"BRANCH_FINAL", "BRANCH_INTERMEDIATE"} and gold_map.get(gold) == "LINEAR":
		return "FALSE_BRANCHING"
	if predicted == "UNCERTAIN":
		return "OVER_UNCERTAIN"
	if predicted in {"BRANCH_FINAL", "BRANCH_INTERMEDIATE"}:
		return "WRONG_BRANCH_TYPE"
	return "FALSE_LINEARIZATION"


def main() -> None:
	rows = read_jsonl(INPUT_PATH)
	if len(rows) != 15:
		raise SystemExit(f"expected the frozen 15-case Phase 2.4.4 set, found {len(rows)}")
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	analysis: list[dict[str, Any]] = []
	for row in rows:
		entities = [Entity(str(item.get("id", "")), str(item.get("name", ""))) for item in row.get("router_input", {}).get("runtime_topic_entities", [])]
		payload = build_offline_payload(str(row["question"]), entities)
		decision, history = _route_with_payload(str(row["question"]), entities, payload)
		predicted = decision["router_structure"]
		analysis.append({
			"case_id": row["case_id"],
			"question": row["question"],
			"group": "HARMFUL" if row["routing_outcome"] == "HARMFUL_ROUTING_MISMATCH" else "BENIGN" if row["routing_outcome"] == "BENIGN_ROUTING_MISMATCH" else "CONTROL",
			"gold_structure": row["gold_structure"],
			"baseline_structure": row["router_structure"],
			"baseline_error": row["primary_router_error"],
			"phase2_5_structure": predicted,
			"phase2_5_outcome": outcome(row["gold_structure"], predicted),
			"constraint_extraction": decision["constraints"],
			"independent_sources": decision["independent_sources"],
			"shared_target": decision["shared_target"],
			"dependency_pattern": decision["dependency_pattern"],
			"validator_decision": decision["validator_decision"],
			"validation_passed": decision["validation_passed"],
			"validation_failures": decision["validation_failures"],
			"prompt_history_type": history[-1]["type"] if history else "",
			"live_llm_call": False,
			"live_end_to_end_run": False,
		})

	(OUTPUT_ROOT / "constraint_extraction_analysis.jsonl").write_text(
		"".join(json.dumps(item, ensure_ascii=False) + "\n" for item in analysis), encoding="utf-8"
	)

	def count(field: str, value: Any, subset: list[dict[str, Any]] = analysis) -> int:
		return sum(item.get(field) == value for item in subset)

	gold_to_structure = {"GOLD_LINEAR": "LINEAR", "GOLD_BRANCH_FINAL": "BRANCH_FINAL", "GOLD_BRANCH_INTERMEDIATE": "BRANCH_INTERMEDIATE"}
	baseline_accuracy = sum(item["baseline_structure"] == gold_to_structure[item["gold_structure"]] for item in analysis)
	new_accuracy = sum(item["phase2_5_structure"] == gold_to_structure[item["gold_structure"]] for item in analysis)
	harmful = [item for item in analysis if item["group"] == "HARMFUL"]
	benign = [item for item in analysis if item["group"] == "BENIGN"]
	controls = [item for item in analysis if item["group"] == "CONTROL"]
	results = {
		"experiment": "phase2_5_constraint_first_routing",
		"stage": "A_focused_diagnostic",
		"sample_size": len(analysis),
		"baseline_router": "frozen_v1_1_3",
		"new_router": "constraint_first_offline_fixture",
		"baseline_structure_accuracy": {"correct": baseline_accuracy, "total": len(analysis), "rate": baseline_accuracy / len(analysis)},
		"phase2_5_structure_accuracy": {"correct": new_accuracy, "total": len(analysis), "rate": new_accuracy / len(analysis)},
		"harmful_routing_corrections": sum(item["baseline_structure"] != gold_to_structure[item["gold_structure"]] and item["phase2_5_structure"] == gold_to_structure[item["gold_structure"]] for item in harmful),
		"benign_structure_preserved": sum(item["phase2_5_structure"] == gold_to_structure[item["gold_structure"]] for item in benign),
		"correct_control_preserved": sum(item["phase2_5_structure"] == gold_to_structure[item["gold_structure"]] for item in controls),
		"baseline_error_counts": dict(Counter(item["baseline_error"] for item in analysis)),
		"phase2_5_outcome_counts": dict(Counter(item["phase2_5_outcome"] for item in analysis)),
		"new_false_positive_branching": sum(item["phase2_5_outcome"] == "FALSE_BRANCHING" for item in analysis),
		"end_to_end": {
			"status": "NOT_RUN_DUE_TO_NO_LIVE_LLM_OR_BACKEND",
			"hits_at_1": "NOT_RUN",
			"f1": "NOT_RUN",
			"native_result": "NOT_RUN",
			"final_result_after_fallback": "NOT_RUN",
		},
		"gate_decision": "STOP_AND_REVISE",
		"final_decision": "INCONCLUSIVE",
	}
	(OUTPUT_ROOT / "focused_diagnostic_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

	comparison = [
		"# Phase 2.5 routing comparison",
		"",
		"Offline Stage A only. The new router was exercised with a deterministic model-shaped fixture; no live LLM or Freebase execution was performed.",
		"",
		"| Metric | Frozen v1.1.3 | Phase 2.5 fixture |",
		"|---|---:|---:|",
		f"| Structure accuracy | {baseline_accuracy}/{len(analysis)} | {new_accuracy}/{len(analysis)} |",
		f"| Harmful routing corrections | — | {results['harmful_routing_corrections']} |",
		f"| Correct controls preserved | — | {results['correct_control_preserved']}/{len(controls)} |",
		f"| New false-positive branching | — | {results['new_false_positive_branching']} |",
		"",
		"The structural result is a wiring diagnostic, not a model-quality result. The end-to-end gate cannot pass without running the actual model and downstream executor.",
		"",
	]
	(OUTPUT_ROOT / "routing_comparison.md").write_text("\n".join(comparison), encoding="utf-8")
	(OUTPUT_ROOT / "cost_comparison.md").write_text(
		"# Phase 2.5 cost comparison\n\n"
		"Stage A used no live LLM calls and therefore records zero measured API tokens and latency.\n"
		"A live cost comparison against Adaptive v1.1.3 is **NOT_RUN_DUE_TO_GATE**; no token values are fabricated.\n",
		encoding="utf-8",
	)

	report = f"""# Phase 2.5 explicit constraint extraction for routing

## 1. Motivation from Phase 2.4.4

Phase 2.4.4 identified independent-source misses, false sequentialization, answer-type conflicts, type-constraint confusion, and missed shared intermediates. This phase isolates a constraint-first router variant.

## 2. Intervention design

The new `ConstraintFirstRouter` asks for `constraints`, `independent_sources`, `shared_target`, and `dependency_pattern` before mapping to the existing `LINEAR`, `BRANCH_FINAL`, `BRANCH_INTERMEDIATE`, and `UNCERTAIN` labels. It is implemented under `chain_of_relations/methods/phase2_5/` and does not modify v1.1.3.

## 3. Frozen components

Entity lookup, Freebase, SPARQL, relation generation, Phase 2.1 grounding, branch execution, merge/fallback, evaluator, model settings, and datasets remain unchanged. The runtime wrapper reuses the v1.1.3 executor and only replaces the router.

## 4. Validator and logging

The deterministic v1.1.3 identity/confidence/root checks are retained. Constraint-first validation does not reject a structurally supported branch solely because the lexical answer type or semantic properties are imperfect; it still requires a shared target, at least two extracted constraints/sources, valid branch objects, and non-type-only evidence. Logs record the full extracted representation and validator outcome.

## 5. Stage A focused diagnostic

The frozen 15-case Phase 2.4.4 set was processed. Because this environment did not have a live LLM/backend execution for the new route, Stage A used a deterministic model-shaped fixture to verify schema and routing integration only.

| Metric | Frozen v1.1.3 | Phase 2.5 fixture |
|---|---:|---:|
| Structure accuracy | {baseline_accuracy}/{len(analysis)} | {new_accuracy}/{len(analysis)} |
| Harmful routing corrections | — | {results['harmful_routing_corrections']} |
| Benign gold-structure cases correct | — | {results['benign_structure_preserved']}/{len(benign)} |
| Correct-routing controls preserved | — | {results['correct_control_preserved']}/{len(controls)} |
| New false-positive branching | — | {results['new_false_positive_branching']} |

The per-case extraction and routing records are in `constraint_extraction_analysis.jsonl`.

## 6. End-to-end result and gate

Hits@1, F1, native results, and fallback results were **NOT_RUN**. The actual model/backend run is required before making a downstream QA claim. Therefore the Stage A gate is `STOP_AND_REVISE`, and Stage B/fuller evaluation are not run.

## 7. Cost analysis

The offline fixture used zero live LLM calls, so no measured token or latency comparison is reported. See `cost_comparison.md`.

## 8. Limitations

This run demonstrates the implementation and offline schema path, not the quality of the model-generated extraction. The lexical fixture must be replaced by a live Phase 2.5 router run before assessing generalization, cost, or end-to-end QA.

## 9. Final Phase 2.5 decision

**INCONCLUSIVE**

The intervention is implemented and the focused structural path is wired, but the required live Stage A end-to-end evidence is unavailable. Stage B and Stage C remain intentionally unrun.
"""
	(OUTPUT_ROOT / "PHASE2_5_REPORT.md").write_text(report, encoding="utf-8")
	print(json.dumps({
		"output_root": str(OUTPUT_ROOT),
		"sample_size": len(analysis),
		"baseline_accuracy": baseline_accuracy,
		"phase2_5_fixture_accuracy": new_accuracy,
		"harmful_corrections": results["harmful_routing_corrections"],
		"controls_preserved": results["correct_control_preserved"],
		"gate": results["gate_decision"],
		"decision": results["final_decision"],
	}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
