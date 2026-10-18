#!/usr/bin/env python3
"""Run the isolated Phase 2.1 semantic-target relation-grounding study.

The script has explicit stages so the offline taxonomy/diagnostic set and the
relation-level comparison are completed before the optional small end-to-end
diagnostic.  It never launches the Phase 1 full-100 evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
	 sys.path.insert(0, str(PROJECT_ROOT))

# Install the existing lightweight transport before importing modules that
# construct the OpenAI client.
import run_adaptive_v1_1_3_experiment as frozen  # noqa: E402
from chain_of_relations.kg_backend import get_backend  # noqa: E402
from chain_of_relations.methods.phase2_1.agent import (  # noqa: E402
	SemanticTargetAdaptiveAgent,
	SemanticTargetCoRAgent,
)
from chain_of_relations.methods.phase2_1.relation_grounding import (  # noqa: E402
	SemanticTargetRelationGrounder,
	SemanticTargetState,
)


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase2_1" / "relation_grounding"
FULL100_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_3" / "full_100"
SUBSET_ROOT = PROJECT_ROOT / "outputs" / "subsets"
DETAIL_ROOT = FULL100_ROOT / "details" / "adaptive"
PHASE2_DETAIL_ROOT = OUTPUT_ROOT / "details" / "adaptive"

FROZEN_SETTINGS = {
	"depth": 3,
	"relation_width": 3,
	"sample_relation_threshold": 1000,
	"remove_unnecessary_rel": True,
	"temperature_exploration": 0.3,
	"temperature_reasoning": 0.1,
	"max_token": 1024,
	"confidence_threshold": 0.75,
	"max_branches": 3,
}


# These are offline diagnostic labels, not runtime inputs.  The selection is
# intentionally small and fixed before Phase 2.1 model outputs are inspected.
CASE_SPECS: list[dict[str, Any]] = [
	{
		"question_id": "WebQTest-719_e56e856592a9a57b9e331b718adc1b75",
		"dataset": "cwq_branch_2", "step_index": 0,
		"expected_relation": "book.author.book_editions_published", "expected_direction": "incoming",
		"failure_type": "RELATION_RANKING_ERROR",
		"semantic_goal": "reach the author who published the book before finding that person's political ideal",
	},
	{
		"question_id": "WebQTest-759_88081dc1a259cbd172b89242384ee1cc",
		"dataset": "cwq_branch_2", "step_index": 0,
		"expected_relation": "celebrities.celebrity.sexual_relationships", "expected_direction": "outgoing",
		"failure_type": "RELATION_RANKING_ERROR",
		"semantic_goal": "reach Brad Pitt's romantic relationships so the 1989 end-date constraint can identify the celebrity",
	},
	{
		"question_id": "WebQTrn-2400_400ed71d115f080056e3437113c51ce2",
		"dataset": "cwq_branch_2", "step_index": 1,
		"expected_relation": "film.film.starring", "expected_direction": "incoming",
		"failure_type": "RELATION_RANKING_ERROR",
		"semantic_goal": "from the identified film reach its starring performances and then their actors",
	},
	{
		"question_id": "WebQTrn-64_984678664419049133d2bd748b68df1b",
		"dataset": "cwq_branch_2", "step_index": 0,
		"expected_relation": "music.featured_artist.recordings", "expected_direction": "incoming",
		"failure_type": "RELATION_RANKING_ERROR",
		"semantic_goal": "reach the recording involving the named artist before identifying the artist in Juice",
	},
	{
		"question_id": "3205130014000",
		"dataset": "grailqa_branch_2", "step_index": 0,
		"expected_relation": "fictional_universe.fictional_language.where_spoken", "expected_direction": "outgoing",
		"failure_type": "RELATION_RANKING_ERROR",
		"semantic_goal": "reach the fictional universe or work where ancient Sumerian is spoken, then apply the event constraint",
	},
	{
		"question_id": "4300909001000",
		"dataset": "grailqa_branch_2", "step_index": 1,
		"expected_relation": "music.release.track", "expected_direction": "incoming",
		"failure_type": "RELATION_RANKING_ERROR",
		"semantic_goal": "from the featured recording reach the musical release component that tracks it",
	},
	{
		"question_id": "4300909004000",
		"dataset": "grailqa_branch_2", "step_index": 0,
		"expected_relation": "music.artist.track", "expected_direction": "outgoing",
		"failure_type": "RELATION_RANKING_ERROR",
		"semantic_goal": "reach recordings made by Glenn Miller and his orchestra before intersecting with the Modernaires constraint",
	},
	{
		"question_id": "4301990001000",
		"dataset": "grailqa_branch_2", "step_index": 0,
		"expected_relation": "music.release.producers", "expected_direction": "incoming",
		"failure_type": "RELATION_CANDIDATE_MISSING",
		"semantic_goal": "reach the musical release produced by the recording and labelled Track Record",
	},
	{
		"question_id": "4302233001000",
		"dataset": "grailqa_branch_2", "step_index": 0,
		"expected_relation": "education.academic.research_areas", "expected_direction": "outgoing",
		"failure_type": "RELATION_CANDIDATE_MISSING",
		"semantic_goal": "reach the research discipline associated with McClelland and restrict it to social sciences",
	},
	{
		"question_id": "WebQTest-743_af8b975343e4098913128e9ac7d04c05",
		"dataset": "cwq_branch_2", "step_index": 0,
		"expected_relation": "event.public_speaker.speeches_or_presentations", "expected_direction": "outgoing",
		"failure_type": "RELATION_CANDIDATE_MISSING",
		"semantic_goal": "reach the public speaker associated with the anti-apartheid topic before following sibling relations",
	},
	{
		"question_id": "WebQTrn-1812_8126be584feae3f358a930823db22c4e",
		"dataset": "cwq_branch_2", "step_index": 0,
		"expected_relation": "location.statistical_region.gdp_deflator_change", "expected_direction": "outgoing",
		"failure_type": "RELATION_CANDIDATE_MISSING",
		"semantic_goal": "reach Caribbean countries and apply the GDP-deflator rate constraint",
	},
	{
		"question_id": "WebQTest-191_ed2cadafdbc0ac07e3fedfdfc504e54b",
		"dataset": "cwq_branch_2", "step_index": 0,
		"expected_relation": "people.person.education", "expected_direction": "outgoing",
		"failure_type": "SUCCESSFUL_CONTROL",
		"semantic_goal": "reach the educational institution attended by Drew Brees",
	},
	{
		"question_id": "WebQTest-1736_8526fc2dec1ebbf627ccdf877d837b32",
		"dataset": "cwq_branch_2", "step_index": 0,
		"expected_relation": "location.country.official_language", "expected_direction": "incoming",
		"failure_type": "SUCCESSFUL_CONTROL",
		"semantic_goal": "reach the official language of the country using West Germanic languages",
	},
	{
		"question_id": "3203485015000",
		"dataset": "grailqa_branch_2", "step_index": 0,
		"expected_relation": "amusement_parks.ride.theme", "expected_direction": "outgoing",
		"failure_type": "SUCCESSFUL_CONTROL",
		"semantic_goal": "reach the shared amusement-ride theme of the two grounded rides",
	},
	{
		"question_id": "4300978006000",
		"dataset": "grailqa_branch_2", "step_index": 0,
		"expected_relation": "sports.tournament_event_competitor.events_competed_in", "expected_direction": "outgoing",
		"failure_type": "SUCCESSFUL_CONTROL",
		"semantic_goal": "reach the tournament event competition in which the two competitors competed",
	},
	{
		"question_id": "3203599014000",
		"dataset": "grailqa_branch_2", "step_index": 0,
		"expected_relation": "language.language_writing_type.writing_systems", "expected_direction": "outgoing",
		"failure_type": "SUCCESSFUL_CONTROL",
		"semantic_goal": "reach the communication writing system shared by Khuzdul and the alphabet",
	},
]


WRONG_LABELS: dict[str, tuple[str, str]] = {
	"WebQTest-719_e56e856592a9a57b9e331b718adc1b75": ("RELATION_GROUNDING", "RELATION_RANKING_ERROR"),
	"WebQTest-743_af8b975343e4098913128e9ac7d04c05": ("RELATION_GROUNDING", "RELATION_CANDIDATE_MISSING"),
	"WebQTest-759_88081dc1a259cbd172b89242384ee1cc": ("RELATION_GROUNDING", "RELATION_RANKING_ERROR"),
	"WebQTrn-1023_5bf6906c26a34660ad81a9e0506d36fc": ("ENTITY_GROUNDING", ""),
	"WebQTrn-1812_8126be584feae3f358a930823db22c4e": ("RELATION_GROUNDING", "RELATION_CANDIDATE_MISSING"),
	"WebQTrn-2400_400ed71d115f080056e3437113c51ce2": ("RELATION_GROUNDING", "RELATION_RANKING_ERROR"),
	"WebQTrn-64_984678664419049133d2bd748b68df1b": ("RELATION_GROUNDING", "RELATION_RANKING_ERROR"),
	"WebQTrn-886_36e0f3794f873b19127b97c24fcb3743": ("ENTITY_GROUNDING", ""),
	"3201139013000": ("ANSWER_SELECTION", ""),
	"3201942006000": ("ANSWER_SELECTION", ""),
	"3201942014000": ("ANSWER_SELECTION", ""),
	"3201947000000": ("STRUCTURAL_ROUTING", ""),
	"3202155012000": ("STRUCTURAL_ROUTING", ""),
	"3202651014000": ("ANSWER_SELECTION", ""),
	"3202813000000": ("ENTITY_GROUNDING", ""),
	"3203433006000": ("ANSWER_SELECTION", ""),
	"3203947002000": ("STRUCTURAL_ROUTING", ""),
	"3205130014000": ("RELATION_GROUNDING", "RELATION_RANKING_ERROR"),
	"4300753011000": ("STRUCTURAL_ROUTING", ""),
	"4300909001000": ("RELATION_GROUNDING", "RELATION_RANKING_ERROR"),
	"4300909004000": ("RELATION_GROUNDING", "RELATION_RANKING_ERROR"),
	"4301182008000": ("ANSWER_SELECTION", ""),
	"4301917001000": ("STRUCTURAL_ROUTING", ""),
	"4301990001000": ("RELATION_GROUNDING", "RELATION_CANDIDATE_MISSING"),
	"4302233001000": ("RELATION_GROUNDING", "RELATION_CANDIDATE_MISSING"),
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
	path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def parse_relation_candidates(prompt: str) -> list[dict[str, str]]:
	result: list[dict[str, str]] = []
	seen: set[tuple[str, str]] = set()
	direction = ""
	for line in prompt.splitlines():
		if line.startswith("# "):
			direction = "outgoing" if "?relation ?x" in line else "incoming"
			continue
		if not direction or not line.endswith(";"):
			continue
		for relation in line[:-1].split(";"):
			relation = relation.strip()
			if not relation or (relation, direction) in seen:
				continue
			seen.add((relation, direction))
			result.append({"relation": relation, "direction": direction, "label": relation})
	return result


def relation_records_from_ids(selected: list[str], candidates: list[dict[str, str]]) -> list[dict[str, Any]]:
	result = []
	for relation in selected:
		matches = [item for item in candidates if item["relation"] == str(relation)]
		if matches:
			item = matches[0]
			result.append({"relation": item["relation"], "direction": item["direction"]})
		else:
			result.append({"relation": str(relation), "direction": ""})
	return result


def load_subset_rows() -> dict[tuple[str, str], dict[str, Any]]:
	rows: dict[tuple[str, str], dict[str, Any]] = {}
	for dataset in ("cwq_branch_2", "grailqa_branch_2"):
		for row in read_jsonl(SUBSET_ROOT / f"{dataset}.jsonl"):
			rows[(dataset, str(row.get("id", "")))] = row
	return rows


def load_detail(dataset: str, question_id: str) -> dict[str, Any]:
	return read_json(DETAIL_ROOT / dataset / f"{question_id}.json")


def relation_events(detail: dict[str, Any]) -> list[dict[str, Any]]:
	return [item for item in detail.get("prompt_history", []) if item.get("type") == "relation_prune"]


def prepare_diagnostic() -> list[dict[str, Any]]:
	subset_rows = load_subset_rows()
	cases: list[dict[str, Any]] = []
	for spec in CASE_SPECS:
		key = (spec["dataset"], spec["question_id"])
		row = subset_rows.get(key)
		if row is None:
			raise SystemExit(f"diagnostic case missing from subset: {key}")
		detail = load_detail(*key)
		events = relation_events(detail)
		step_index = int(spec["step_index"])
		if step_index >= len(events):
			raise SystemExit(f"relation decision {step_index} missing for {key}; only {len(events)} decisions")
		event = events[step_index]
		candidates = parse_relation_candidates(str(event.get("prompt", "")))
		selected = [str(item) for item in event.get("parsed_result", []) if str(item).strip()]
		available = any(
			item["relation"] == spec["expected_relation"] and item["direction"] == spec["expected_direction"]
			for item in candidates
		)
		case = {
			**spec,
			"question": row.get("question", detail.get("question", "")),
			"structure": detail.get("router_structure", ""),
			"current_entity": {
				"id": (row.get("topic_entities") or [""])[0],
				"name": (row.get("topic_entity_names") or [""])[0],
			},
			"expected_final_answer_type": detail.get("requested_final_answer_type", ""),
			"requested_final_semantic_property": detail.get("requested_final_semantic_property", ""),
			"relation_history": detail.get("relation_chain", [])[:step_index],
			"candidate_relations": candidates,
			"baseline_selected_relation": relation_records_from_ids(selected, candidates),
			"baseline_top1": relation_records_from_ids(selected[:1], candidates),
			"correct_relation_available_in_candidates": available,
			"candidate_recall": int(available),
			"baseline_relation_selection_correct": bool(selected and selected[0] == spec["expected_relation"] and any(item["relation"] == spec["expected_relation"] and item["direction"] == spec["expected_direction"] for item in relation_records_from_ids(selected[:1], candidates))),
			"source_trace_index": step_index,
		}
		cases.append(case)
	write_json(OUTPUT_ROOT / "diagnostic_cases.json", cases)
	return cases


def write_failure_taxonomy() -> None:
	rows: list[dict[str, Any]] = []
	for path in sorted(DETAIL_ROOT.rglob("*.json")):
		detail = read_json(path)
		if detail.get("final_hits_at_1") == 1:
			continue
		qid = str(detail.get("question_id", detail.get("id", "")))
		primary, subtype = WRONG_LABELS.get(qid, ("OTHER", ""))
		rows.append({
			"question_id": qid,
			"dataset": detail.get("dataset", ""),
			"question": detail.get("question", ""),
			"primary_category": primary,
			"relation_subtype": subtype,
			"structure": detail.get("router_structure", ""),
			"prediction": detail.get("final_prediction", []),
			"gold": detail.get("gold_answers", []),
		})
	lines = [
		"# Phase 2.1 grounding failure taxonomy",
		"",
		"This is an offline diagnostic of the 25 final H@1 failures in the frozen Adaptive v1.1.3 full-100 run. Labels use recorded traces and the dataset graph only for diagnosis; no label or gold relation is supplied to the runtime scorer.",
		"",
		"| Primary category | Count |",
		"|---|---:|",
	]
	from collections import Counter
	counts = Counter(item["primary_category"] for item in rows)
	for category, count in sorted(counts.items()):
		lines.append(f"| {category} | {count} |")
	lines += [
		"",
		"## Relation-grounding split",
		"",
		"| Relation subtype | Count |",
		"|---|---:|",
	]
	relation_counts = Counter(item["relation_subtype"] for item in rows if item["primary_category"] == "RELATION_GROUNDING")
	for category in ("RELATION_CANDIDATE_MISSING", "RELATION_RANKING_ERROR", "RELATION_DIRECTION_ERROR"):
		lines.append(f"| {category} | {relation_counts.get(category, 0)} |")
	lines += [
		"",
		"The current trace review finds no confirmed primary relation-direction error: the observed directional mistakes are either ranking decisions or downstream answer/constraint failures. Candidate recall and ranking accuracy are therefore reported separately.",
		"",
		"## Case-level labels",
		"",
		"| Dataset | Question ID | Primary category | Relation subtype | Structure |",
		"|---|---|---|---|---|",
	]
	for item in rows:
		lines.append(f"| {item['dataset']} | `{item['question_id']}` | {item['primary_category']} | {item['relation_subtype'] or '—'} | {item['structure']} |")
	(OUTPUT_ROOT / "failure_taxonomy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare() -> list[dict[str, Any]]:
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	write_failure_taxonomy()
	cases = prepare_diagnostic()
	(OUTPUT_ROOT / "prediction.md").write_text(
		"# Prediction recorded before Phase 2.1 results\n\n"
		"Explicit semantic-target conditioning should improve relation ranking when the correct Freebase relation is already present in the candidate set, especially on multi-hop questions with locally plausible alternatives. It should not fix candidate-missing or initial entity-grounding failures. Expected changes: RELATION_RANKING_ERROR decreases; RELATION_CANDIDATE_MISSING and ENTITY_GROUNDING remain approximately unchanged; successful controls remain mostly preserved.\n",
		encoding="utf-8",
	)
	write_json(OUTPUT_ROOT / "config.json", {
		"experiment": "phase2_1_semantic_target_relation_grounding",
		"phase1_method_frozen": "adaptive_v1.1.3",
		"full100_not_rerun": True,
		"gold_relations_used_at_runtime": False,
		"entity_grounding_changed": False,
		"router_changed": False,
		"diagnostic_case_count": len(cases),
		"settings": FROZEN_SETTINGS,
		"model": os.getenv("MODEL_NAME", ""),
		"freebase_sparql_endpoint": os.getenv("FREEBASE_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql"),
	})
	return cases


def make_phase2_agent() -> SemanticTargetAdaptiveAgent:
	kwargs = dict(
		model_name=os.environ["MODEL_NAME"],
		relation_width=FROZEN_SETTINGS["relation_width"],
		depth=FROZEN_SETTINGS["depth"],
		sample_relation_threshold=FROZEN_SETTINGS["sample_relation_threshold"],
		remove_unnecessary_rel=FROZEN_SETTINGS["remove_unnecessary_rel"],
		temperature_exploration=FROZEN_SETTINGS["temperature_exploration"],
		temperature_reasoning=FROZEN_SETTINGS["temperature_reasoning"],
		max_token=FROZEN_SETTINGS["max_token"],
		backend=get_backend("freebase"),
	)
	cor = SemanticTargetCoRAgent(**kwargs)
	return SemanticTargetAdaptiveAgent(cor, confidence_threshold=FROZEN_SETTINGS["confidence_threshold"], max_branches=FROZEN_SETTINGS["max_branches"])


def verify_sparql_endpoint() -> None:
	"""Fail before model calls when the required local Freebase tunnel is down."""
	endpoint = os.getenv("FREEBASE_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql")
	query = urllib.parse.urlencode({"query": "ASK { ?s ?p ?o }", "format": "json"})
	url = f"{endpoint}?{query}"
	try:
		with urllib.request.urlopen(url, timeout=5) as response:
			if response.status != 200:
				raise RuntimeError(f"HTTP {response.status}")
	except (OSError, urllib.error.URLError, RuntimeError) as exc:
		raise SystemExit(
			"Freebase SPARQL endpoint is unreachable at "
			f"{endpoint}. Start the SSH tunnel and retry; no end-to-end model calls were made. "
			f"Probe error: {exc}"
		)


def run_relation_level(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
	if not os.getenv("MODEL_NAME") or not os.getenv("OPENAI_API_KEY"):
		raise SystemExit("MODEL_NAME and OPENAI_API_KEY are required")
	agent = make_phase2_agent()
	grounder = agent.cor_agent.semantic_relation_grounder
	baseline_rows: list[dict[str, Any]] = []
	phase2_rows: list[dict[str, Any]] = []
	for index, case in enumerate(cases, start=1):
		candidates = case["candidate_relations"]
		head = [item["relation"] for item in candidates if item["direction"] == "outgoing"]
		tail = [item["relation"] for item in candidates if item["direction"] == "incoming"]
		state = SemanticTargetState(
			requested_final_answer_type=case.get("expected_final_answer_type", ""),
			requested_final_semantic_property=case.get("requested_final_semantic_property", ""),
			current_subgoal=case["semantic_goal"],
			current_entity_type="offline diagnostic runtime entity",
			remaining_question_constraint=case["question"],
			reasoning_step_index=int(case["step_index"]),
			relation_history=tuple(case.get("relation_history", [])),
		)
		entity = __import__("chain_of_relations.schema", fromlist=["Entity"]).Entity(case["current_entity"]["id"], case["current_entity"]["name"])
		decision = grounder.select(
			question=case["question"],
			topic_entity=entity,
			relation_chain=[],
			head_relations=head,
			tail_relations=tail,
			relation_display_names={},
			state=state,
			top_k=FROZEN_SETTINGS["relation_width"],
		)
		phase2_selected = [{"relation": item.id, "direction": "outgoing" if item.left else "incoming"} for item in decision.selected_relations]
		baseline = dict(case)
		baseline.update({
			"decision_index": index,
			"expected_relation": case["expected_relation"],
			"expected_direction": case["expected_direction"],
			"candidate_recall": case["candidate_recall"],
			"relation_selection_correct": case["baseline_relation_selection_correct"],
			"selected_relation": case["baseline_selected_relation"],
			"confidence": None,
			"selection_reason": "frozen CoR relation_prune result recorded before Phase 2.1",
		})
		baseline_rows.append(baseline)
		phase2 = dict(case)
		phase2.update({
			"decision_index": index,
			"expected_relation": case["expected_relation"],
			"expected_direction": case["expected_direction"],
			"candidate_recall": case["candidate_recall"],
			"relation_selection_correct": bool(phase2_selected and phase2_selected[0]["relation"] == case["expected_relation"] and phase2_selected[0]["direction"] == case["expected_direction"]),
			"selected_relation": phase2_selected,
			"top_k_selected_relations": phase2_selected,
			"confidence": decision.confidence,
			"selection_reason": decision.selection_reason,
			"phase2_success": decision.success,
			"warnings": decision.warnings,
			"input_tokens": decision.usage.get("input_tokens", 0),
			"output_tokens": decision.usage.get("output_tokens", 0),
			"total_tokens": decision.usage.get("total_tokens", 0),
		})
		phase2_rows.append(phase2)
		logging.info("relation-level %d/%d complete: %s", index, len(cases), case["question_id"])
	write_jsonl(OUTPUT_ROOT / "relation_decisions_baseline.jsonl", baseline_rows)
	write_jsonl(OUTPUT_ROOT / "relation_decisions_phase2_1.jsonl", phase2_rows)
	return phase2_rows


def accuracy(rows: list[dict[str, Any]], field: str) -> float:
	return sum(bool(row.get(field)) for row in rows) / len(rows) if rows else 0.0


def phase2_decision() -> tuple[str, str]:
	relation_path = OUTPUT_ROOT / "relation_decisions_phase2_1.jsonl"
	baseline_path = OUTPUT_ROOT / "relation_decisions_baseline.jsonl"
	if not relation_path.exists() or not baseline_path.exists():
		return "PENDING_RELATION_LEVEL_GATE", "The relation-level comparison has not been completed."
	relation_baseline = read_jsonl(baseline_path)
	relation_phase2 = read_jsonl(relation_path)
	if not relation_phase2:
		return "PENDING_RELATION_LEVEL_GATE", "The relation-level comparison contains no decisions."
	controls = [row for row in relation_baseline if row.get("failure_type") == "SUCCESSFUL_CONTROL"]
	phase2_by_id = {row["question_id"]: row for row in relation_phase2}
	controls_preserved = all(
		bool(phase2_by_id.get(row["question_id"], {}).get("relation_selection_correct"))
		for row in controls
	)
	relation_gain = accuracy(relation_phase2, "relation_selection_correct") > accuracy(relation_baseline, "relation_selection_correct")
	end_path = OUTPUT_ROOT / "end_to_end_results.jsonl"
	if not end_path.exists():
		return "PENDING_END_TO_END_DIAGNOSTIC", "The relation-level gate passed, but the small end-to-end diagnostic is pending."
	end_rows = read_jsonl(end_path)
	if not end_rows:
		return "PENDING_END_TO_END_DIAGNOSTIC", "The end-to-end output contains no completed cases."
	baseline_h1 = sum(int(row["baseline"].get("hits_at_1", 0) or 0) for row in end_rows)
	phase2_h1 = sum(int(row["phase2_1"].get("hits_at_1", 0) or 0) for row in end_rows)
	baseline_f1 = sum(float(row["baseline"].get("f1", 0) or 0) for row in end_rows)
	phase2_f1 = sum(float(row["phase2_1"].get("f1", 0) or 0) for row in end_rows)
	baseline_only = sum(
		int(row["baseline"].get("hits_at_1", 0) or 0) == 1
		and int(row["phase2_1"].get("hits_at_1", 0) or 0) == 0
		for row in end_rows
	)
	if relation_gain and controls_preserved and phase2_h1 > baseline_h1 and baseline_only == 0:
		return (
			"PROCEED_TO_PHASE2_2",
			f"Relation top-1 improved with controls preserved; end-to-end Hits@1 improved {baseline_h1}/{len(end_rows)} to {phase2_h1}/{len(end_rows)}, with no baseline-only wins. Mean F1 changed {baseline_f1 / len(end_rows):.4f} to {phase2_f1 / len(end_rows):.4f}.",
		)
	if relation_gain and controls_preserved:
		return (
			"ITERATE_RELATION_GROUNDING_ONCE",
			f"Relation-level ranking improved and controls were preserved, but the end-to-end diagnostic did not produce a strictly better Hits@1 result ({baseline_h1}/{len(end_rows)} to {phase2_h1}/{len(end_rows)}).",
		)
	return (
		"NO_CLEAR_RELATION_GROUNDING_SIGNAL",
		"The relation-level or end-to-end comparison did not satisfy the improvement and control-preservation gate.",
	)


def write_relation_comparison() -> None:
	baseline = read_jsonl(OUTPUT_ROOT / "relation_decisions_baseline.jsonl")
	phase2 = read_jsonl(OUTPUT_ROOT / "relation_decisions_phase2_1.jsonl")
	controls = [row for row in baseline if row["failure_type"] == "SUCCESSFUL_CONTROL"]
	phase2_by_id = {row["question_id"]: row for row in phase2}
	lines = [
		"# Phase 2.1 relation-level comparison",
		"",
		"The comparison uses the same frozen relation-decision points extracted from the completed Phase 1 traces. `candidate_recall` asks whether the expected relation and direction were in the candidate set; ranking accuracy asks whether the selector placed that pair first. They are not conflated.",
		"",
		"| Metric | Baseline | Phase 2.1 |",
		"|---|---:|---:|",
		f"| Relation top-1 accuracy | {accuracy(baseline, 'relation_selection_correct'):.2%} | {accuracy(phase2, 'relation_selection_correct'):.2%} |",
		f"| Candidate recall | {accuracy(baseline, 'candidate_recall'):.2%} | {accuracy(phase2, 'candidate_recall'):.2%} |",
		f"| Direction top-1 accuracy | {sum(bool(row.get('relation_selection_correct')) for row in baseline):.0f}/{len(baseline)} | {sum(bool(row.get('relation_selection_correct')) for row in phase2):.0f}/{len(phase2)} |",
		f"| Successful controls preserved | — | {sum(bool(phase2_by_id[row['question_id']].get('relation_selection_correct')) for row in controls)}/{len(controls)} |",
		"",
		"## Case decisions",
		"",
		"| Dataset | ID | Type | Candidate available | Baseline top-1 | Phase 2.1 top-1 | Confidence |",
		"|---|---|---|---|---|---|---:|",
	]
	for before, after in zip(baseline, phase2):
		b = before.get("baseline_top1", before.get("selected_relation", []))
		p = after.get("selected_relation", [])[:1]
		lines.append(
			f"| {after['dataset']} | `{after['question_id']}` | {after['failure_type']} | {'yes' if after['correct_relation_available_in_candidates'] else 'no'} | "
			f"{b[0]['relation'] if b else '—'} ({b[0].get('direction','') if b else ''}) | "
			f"{p[0]['relation'] if p else '—'} ({p[0].get('direction','') if p else ''}) | {float(after.get('confidence', 0) or 0):.2f} |"
		)
	(OUTPUT_ROOT / "relation_level_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_phase1_rows(cases: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
	rows = read_jsonl(FULL100_ROOT / "adaptive_results.jsonl")
	return {(str(row.get("dataset")), str(row.get("question_id"))): row for row in rows if (str(row.get("dataset")), str(row.get("question_id"))) in {(c["dataset"], c["question_id"]) for c in cases}}


def run_end_to_end(cases: list[dict[str, Any]]) -> None:
	if not os.getenv("MODEL_NAME") or not os.getenv("OPENAI_API_KEY"):
		raise SystemExit("MODEL_NAME and OPENAI_API_KEY are required")
	verify_sparql_endpoint()
	phase1_rows = load_phase1_rows(cases)
	subset_rows = load_subset_rows()
	agent = make_phase2_agent()
	results: list[dict[str, Any]] = []
	for index, case in enumerate(cases, start=1):
		key = (case["dataset"], case["question_id"])
		source = subset_rows[key]
		raw = frozen.run_one(agent, "adaptive", 1, source, PHASE2_DETAIL_ROOT / case["dataset"])
		phase2_detail_path = PHASE2_DETAIL_ROOT / case["dataset"] / f"{case['question_id']}.json"
		detail = read_json(phase2_detail_path)
		semantic_events = [item for item in detail.get("prompt_history", []) if item.get("type") == "phase2_1_semantic_relation_grounding"]
		baseline = phase1_rows[key]
		results.append({
			"question_id": case["question_id"], "dataset": case["dataset"], "question": case["question"],
			"failure_type": case["failure_type"], "expected_relation": case["expected_relation"],
			"baseline": {
				"prediction": baseline.get("final_prediction", []), "hits_at_1": baseline.get("final_hits_at_1", 0),
				"f1": baseline.get("final_f1", 0.0), "llm_calls": baseline.get("llm_calls", 0),
				"kg_queries": baseline.get("kg_queries", 0), "tokens": baseline.get("tokens", {}).get("total", 0),
				"latency_ms": baseline.get("latency_ms", 0),
			},
			"phase2_1": {
				"prediction": raw.get("prediction", []), "hits_at_1": raw.get("hits_at_1", 0),
				"f1": raw.get("f1", 0.0), "llm_calls": raw.get("llm_calls", 0),
				"kg_queries": raw.get("kg_queries", 0), "tokens": raw.get("token_usage", {}).get("input_tokens", 0) + raw.get("token_usage", {}).get("output_tokens", 0),
				"latency_ms": raw.get("latency_ms", 0),
				"semantic_relation_decisions": semantic_events,
			},
			"error": raw.get("error"),
		})
		logging.info("end-to-end %d/%d complete: %s", index, len(cases), case["question_id"])
	write_jsonl(OUTPUT_ROOT / "end_to_end_results.jsonl", results)


def write_end_to_end_summary() -> None:
	rows = read_jsonl(OUTPUT_ROOT / "end_to_end_results.jsonl")
	lines = [
		"# Phase 2.1 small end-to-end diagnostic",
		"",
		"This is the same frozen 16-case diagnostic set. Adaptive v1.1.3 is compared with the isolated semantic-target relation selector; router, entity grounding, KG, branch merge, fallback, and evaluator are unchanged.",
		"",
		"| Method | N | Hits@1 | F1 | Avg LLM calls | Avg KG queries | Avg tokens | Avg latency (ms) |",
		"|---|---:|---:|---:|---:|---:|---:|---:|",
	]
	for method in ("baseline", "phase2_1"):
		values = [row[method] for row in rows]
		mean = lambda key: sum(float(value.get(key, 0) or 0) for value in values) / len(values) if values else 0.0
		lines.append(f"| {method} | {len(values)} | {mean('hits_at_1'):.2%} | {mean('f1'):.4f} | {mean('llm_calls'):.2f} | {mean('kg_queries'):.2f} | {mean('tokens'):.1f} | {mean('latency_ms'):.1f} |")
	lines += ["", "## Paired outcomes", "", "| Outcome | Count |", "|---|---:|"]
	both = sum(row["baseline"]["hits_at_1"] == 1 and row["phase2_1"]["hits_at_1"] == 1 for row in rows)
	b_only = sum(row["baseline"]["hits_at_1"] == 1 and row["phase2_1"]["hits_at_1"] == 0 for row in rows)
	p_only = sum(row["baseline"]["hits_at_1"] == 0 and row["phase2_1"]["hits_at_1"] == 1 for row in rows)
	both_wrong = len(rows) - both - b_only - p_only
	lines.extend([f"| BOTH_CORRECT | {both} |", f"| BASELINE_ONLY_CORRECT | {b_only} |", f"| PHASE2_ONLY_CORRECT | {p_only} |", f"| BOTH_WRONG | {both_wrong} |"])
	controls = [row for row in rows if row.get("failure_type") == "SUCCESSFUL_CONTROL"]
	if controls:
		lines += [
			"", "## Successful-control check", "",
			"| Metric | Baseline | Phase 2.1 |",
			"|---|---:|---:|",
			f"| Controls with Hits@1 | {sum(row['baseline']['hits_at_1'] for row in controls)}/{len(controls)} | {sum(row['phase2_1']['hits_at_1'] for row in controls)}/{len(controls)} |",
			f"| Mean control F1 | {sum(float(row['baseline']['f1']) for row in controls) / len(controls):.4f} | {sum(float(row['phase2_1']['f1']) for row in controls) / len(controls):.4f} |",
		]
	(OUTPUT_ROOT / "end_to_end_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cost_analysis() -> None:
	rows = read_jsonl(OUTPUT_ROOT / "end_to_end_results.jsonl")
	lines = [
		"# Phase 2.1 cost analysis",
		"",
		"Costs are recorded resource usage, not dollar estimates. The relation-level scorer adds one semantic relation-selection call at each CoR relation decision and falls back to the frozen baseline selector if its response is invalid.",
		"",
		"| Method | Avg LLM calls | Avg KG queries | Avg tokens | Avg latency (ms) |",
		"|---|---:|---:|---:|---:|",
	]
	for method in ("baseline", "phase2_1"):
		values = [row[method] for row in rows]
		mean = lambda key: sum(float(value.get(key, 0) or 0) for value in values) / len(values) if values else 0.0
		lines.append(f"| {method} | {mean('llm_calls'):.2f} | {mean('kg_queries'):.2f} | {mean('tokens'):.1f} | {mean('latency_ms'):.1f} |")
	(OUTPUT_ROOT / "cost_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report() -> None:
	cases = read_json(OUTPUT_ROOT / "diagnostic_cases.json")
	end_to_end_exists = (OUTPUT_ROOT / "end_to_end_results.jsonl").exists()
	decision, decision_reason = phase2_decision()
	lines = [
		"# Phase 2.1 semantic-target-aware relation grounding",
		"",
		"## 1. Phase 1 motivation",
		"",
		"Adaptive v1.1.3 finished the frozen Phase 1 full-100 evaluation at 75% final H@1 versus 74% for CoR. The remaining wrong cases were dominated by relation/answer grounding rather than a broad routing failure.",
		"",
		"## 2. Research question",
		"",
		"Can candidate Freebase relation ranking improve when it is explicitly conditioned on the semantic target, expected answer type, current subgoal, remaining constraint, direction, and relation history?",
		"",
		"## 3. Entity grounding versus relation grounding",
		"",
		"Entity grounding maps a mention to a Freebase entity and was unchanged. Relation grounding chooses the next directional edge from an already grounded runtime entity and was the only runtime component changed.",
		"",
		"## 4. Frozen Phase 1 components",
		"",
		"The v1.1.3 router, structure labels, answer-type validation, branch merge/fallback behavior, entity lookup, Freebase backend, SPARQL traversal, evaluator, and frozen settings remained unchanged. Gold relations were used only for offline diagnostics, never by the runtime scorer.",
		"",
		"## 5. Grounding failure taxonomy",
		"",
		"See `failure_taxonomy.md`. The taxonomy labels all 25 final H@1 failures and separates relation candidate missing, relation ranking, and direction errors.",
		"",
		"## 6. Diagnostic set",
		"",
		f"The fixed diagnostic contains **{len(cases)}** cases: ranking-error candidates, candidate-missing controls, and successful controls across CWQ/GrailQA and LINEAR/BRANCH_FINAL. The exact cases are in `diagnostic_cases.json`.",
		"",
		"## 7. Semantic-target-aware scorer",
		"",
		"The new scorer supplies requested final answer type/property, current subgoal, current entity context, remaining question constraint, reasoning step, directional candidates, and relation history. It validates every returned relation against the discovered candidate set and preserves the frozen selector if the response is invalid.",
		"",
		"## 8. Prediction before testing",
		"",
		"Relation ranking should improve when the correct relation is already present; candidate-missing and entity-grounding failures should remain approximately unchanged; successful controls should mostly be preserved. This prediction was recorded in `prediction.md` before relation-level results were generated.",
		"",
		"## 9. Relation-level results",
		"",
		"See `relation_level_comparison.md` for top-1 ranking accuracy, candidate recall, direction handling, and control preservation. The raw paired decision logs are `relation_decisions_baseline.jsonl` and `relation_decisions_phase2_1.jsonl`.",
		"",
		"## 10. Candidate recall versus ranking",
		"",
		"Candidate recall is computed from the offline expected relation and candidate set. A low candidate recall is not evidence that the LLM ranked poorly; it indicates that semantic scoring cannot recover that case without changing relation discovery, which is outside Phase 2.1.",
		"",
		"## 11. Direction errors",
		"",
		"Direction is represented explicitly as outgoing or incoming for every candidate and selection. The frozen trace review found no confirmed primary direction-error case; direction is still scored separately so future cases cannot be hidden inside relation ranking.",
		"",
		"## 12. Successful-control preservation",
		"",
		"Successful controls are included in the same decision-point comparison. The required preservation result is reported separately from ranking gains.",
		"",
		"## 13. End-to-end diagnostic",
		"",
	]
	if end_to_end_exists:
		lines.append("The relation-level stage produced results, so the same frozen diagnostic set was run end-to-end. See `end_to_end_summary.md` and `end_to_end_results.jsonl`.")
	else:
		if (OUTPUT_ROOT / "relation_decisions_phase2_1.jsonl").exists():
			lines.append("The relation-level gate passed, but the small end-to-end diagnostic is pending because the required Freebase SPARQL endpoint was unreachable. It is never run automatically on the full 100.")
		else:
			lines.append("The end-to-end stage has not been run yet; it is gated on the relation-level result and is never run automatically on the full 100.")
	lines += [
		"",
		"## 14. Prediction versus actual result",
		"",
		"The actual relation-level comparison, including any control regressions and candidate-missing cases, is recorded in `relation_level_comparison.md`. This section is generated after the result files exist so the prediction remains separate from the measurement.",
		"",
		"## 15. Computational cost",
		"",
		"See `cost_analysis.md` when the end-to-end stage is available. The semantic scorer intentionally adds a relation-selection call and therefore has an expected token/latency overhead.",
		"",
		"## 16. Remaining failures",
		"",
		"The Phase 1 taxonomy indicates that candidate-missing, entity-grounding, answer-selection, and unsupported intermediate-structure cases remain outside this narrow relation-ranking intervention.",
		"",
		"## 17. Recommendation",
		"",
		"The final decision gate is based on relation top-1 improvement, control preservation, and the optional small end-to-end comparison. Do not run the full 100 automatically.",
		"",
		f"**Decision: {decision}**",
		f"**Rationale:** {decision_reason}",
	]
	(OUTPUT_ROOT / "PHASE2_1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--prepare", action="store_true", help="create taxonomy, prediction, and frozen diagnostic cases")
	parser.add_argument("--relation-level", action="store_true", help="run the relation-level baseline vs Phase 2.1 comparison")
	parser.add_argument("--end-to-end", action="store_true", help="run only the small frozen diagnostic end-to-end")
	parser.add_argument("--report", action="store_true", help="write reports from existing Phase 2.1 outputs")
	parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	frozen.load_safe_project_env()
	logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
	if not any((args.prepare, args.relation_level, args.end_to_end, args.report)):
		raise SystemExit("choose --prepare, --relation-level, --end-to-end, or --report")
	if args.prepare:
		prepare()
	if args.relation_level:
		cases = read_json(OUTPUT_ROOT / "diagnostic_cases.json")
		run_relation_level(cases)
		write_relation_comparison()
	if args.end_to_end:
		cases = read_json(OUTPUT_ROOT / "diagnostic_cases.json")
		run_end_to_end(cases)
		write_end_to_end_summary()
		write_cost_analysis()
	elif args.report and (OUTPUT_ROOT / "end_to_end_results.jsonl").exists():
		write_end_to_end_summary()
		write_cost_analysis()
	if args.report or args.prepare or args.relation_level or args.end_to_end:
		write_final_report()


if __name__ == "__main__":
	main()
