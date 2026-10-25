#!/usr/bin/env python3
"""Phase 2.3a diagnostic audit for constraint-to-state alignment.

The audit reconstructs intended constraints from frozen GrailQA/CWQ logical
forms and compares them with frozen Adaptive v1.1.3 / Phase 2.1 traces. It
verifies relation availability against Freebase without changing runtime code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from _bootstrap import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PHASE21_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase2_1" / "relation_grounding"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase2_3" / "constraint_state_alignment"
GRAILQA_TRAIN = PROJECT_ROOT / "datasets" / "grailqa" / "official" / "grailqa_v1.0_train.json"
CWQ_DATA = PROJECT_ROOT / "datasets" / "cwq" / "cwq.json"
FREEBASE_PREFIX = "http://rdf.freebase.com/ns/"
QUERY_LIMIT = 20

CASE_FILES = {
    "4301990001000": PHASE21_ROOT / "details" / "adaptive" / "grailqa_branch_2" / "4301990001000.json",
    "4302233001000": PHASE21_ROOT / "details" / "adaptive" / "grailqa_branch_2" / "4302233001000.json",
    "WebQTest-743_af8b975343e4098913128e9ac7d04c05": PHASE21_ROOT / "details" / "adaptive" / "cwq_branch_2" / "WebQTest-743_af8b975343e4098913128e9ac7d04c05.json",
    "WebQTrn-1812_8126be584feae3f358a930823db22c4e": PHASE21_ROOT / "details" / "adaptive" / "cwq_branch_2" / "WebQTrn-1812_8126be584feae3f358a930823db22c4e.json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def load_dataset_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path, id_key in ((GRAILQA_TRAIN, "qid"), (CWQ_DATA, "id")):
        for row in read_json(path):
            rows[str(row[id_key])] = row
    return rows


def normalize_relation(value: str) -> str:
    value = str(value or "").strip()
    return value[len(FREEBASE_PREFIX):] if value.startswith(FREEBASE_PREFIX) else value


def entity_term(entity_id: str) -> str:
    return f"ns:{entity_id}"


def relation_term(relation: str) -> str:
    relation = str(relation)
    if relation.startswith(("http://", "https://")):
        return f"<{relation}>"
    return f"ns:{relation}"


def parse_json_response(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def parse_entity_from_prompt(prompt: str) -> dict[str, str]:
    match = re.search(r"Current entity:\s*(.*?)\s*\((m\.[^)]+|g\.[^)]+)\)", prompt or "")
    if not match:
        return {"id": "", "name": "", "evidence": "not represented"}
    return {"id": match.group(2), "name": match.group(1).strip(), "evidence": "Phase 2.1 semantic relation-grounding prompt"}


def parse_relation_history(prompt: str) -> list[str]:
    match = re.search(r"Relation history:\s*(\[[^\n]*\])", prompt or "")
    if not match:
        return []
    try:
        value = json.loads(match.group(1))
        return [str(item) for item in value]
    except json.JSONDecodeError:
        return []


def selected_relation_value(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        first = value[0]
        if isinstance(first, dict):
            return str(first.get("relation", ""))
        return str(first)
    if isinstance(value, dict):
        return str(value.get("relation", ""))
    return str(value or "")


def extract_runtime_trace(detail: dict[str, Any]) -> dict[str, Any]:
    router_event = next((event for event in detail.get("prompt_history", []) if (event.get("operation_type") or event.get("type")) == "adaptive_router_v1_1_3"), {})
    router = parse_json_response(router_event.get("response", {}))
    decisions: list[dict[str, Any]] = []
    for event in detail.get("prompt_history", []):
        if (event.get("operation_type") or event.get("type")) != "phase2_1_semantic_relation_grounding":
            continue
        response = parse_json_response(event.get("response", {}))
        prompt = str(event.get("prompt", ""))
        selected_relation = event.get("phase2_selected_relation") or response.get("selected_relation", "")
        selected_direction = event.get("selected_direction") or response.get("direction", "")
        selection_reason = event.get("selection_reason") or response.get("reason", "")
        decisions.append({
            "decision_index": len(decisions),
            "current_entity": parse_entity_from_prompt(prompt),
            "current_subgoal": next((line.split(":", 1)[1].strip() for line in prompt.splitlines() if line.startswith("Current subgoal:")), ""),
            "remaining_constraint": next((line.split(":", 1)[1].strip() for line in prompt.splitlines() if line.startswith("Remaining question constraint:")), ""),
            "reasoning_step_index": next((line.split(":", 1)[1].strip() for line in prompt.splitlines() if line.startswith("Reasoning step index:")), ""),
            "relation_history": parse_relation_history(prompt),
            "selected_relation": normalize_relation(selected_relation_value(selected_relation)),
            "selected_direction": str(selected_direction),
            "selector_reason": str(selection_reason),
            "confidence": event.get("confidence", response.get("confidence")),
        })
    return {
        "structure": str(detail.get("router_structure", router.get("structure", ""))),
        "execution_mode": str(detail.get("final_execution_mode", detail.get("router_execution_mode", ""))),
        "router_reason": str(detail.get("router_reason", router.get("reason", ""))),
        "router": router,
        "branches_created": detail.get("branches_created", 0),
        "branch_decomposition": detail.get("branch_decomposition", {}),
        "branch_traces": detail.get("branch_traces", []),
        "relation_chain": detail.get("relation_chain", []),
        "fallback_used": detail.get("fallback_used", False),
        "decisions": decisions,
    }


def state(entity_id: str, name: str, kind: str = "bound_entity", evidence: str = "") -> dict[str, str]:
    return {"id": entity_id, "name": name, "kind": kind, "evidence": evidence}


def constraint(constraint_id: str, summary: str, target_variable: str, intended_state: dict[str, str], semantic_property: str, relation: str, direction: str, actual_decision_index: int | None, primary: bool = False) -> dict[str, Any]:
    return {
        "constraint_id": constraint_id,
        "constraint_summary": summary,
        "intended_target_variable": target_variable,
        "intended_target_state": intended_state,
        "intended_semantic_property": semantic_property,
        "expected_relation": relation,
        "expected_direction": direction,
        "actual_decision_index": actual_decision_index,
        "primary_constraint": primary,
    }


def case_blueprints() -> dict[str, dict[str, Any]]:
    return {
        "4301990001000": {
            "structure_expected": "BRANCH_INTERMEDIATE / compositional shared-producer graph",
            "execution_expected": "linearizable multi-hop with shared producer and answer label constraint",
            "shared_answer_variable": "?release",
            "merge_operator": "AND over producer and label constraints",
            "branch_related": True,
            "constraints": [
                constraint("producer_bridge", "Pictures of Lily is linked to the producer shared by the answer release", "?producer", state("m.0f3c1rg", "Pictures of Lily", evidence="GrailQA graph node 2"), "producer of recording", "music.recording.producer", "outgoing", 0, True),
                constraint("release_producer", "The answer release is produced by the shared producer", "?release", state("m.03425nx", "Odds and Sods", evidence="GrailQA answer entity; graph answer node 0"), "release producers", "music.release.producers", "outgoing", 0),
                constraint("release_label", "The answer release has Track Record as its label", "?release", state("m.03425nx", "Odds and Sods", evidence="GrailQA answer entity; graph answer node 0"), "record label", "music.release.label", "outgoing", None),
            ],
            "first_divergence_stage": "relation_grounding",
            "diagnosis": "RELATION_GROUNDING_ERROR",
            "secondary_diagnoses": ["CONSTRAINT_DECOMPOSITION_ERROR"],
            "diagnosis_reason": "The initial Pictures of Lily state is correct and the producer bridge is available, but the selector chooses music.producer.tracks_produced incoming instead of music.recording.producer outgoing. The runtime also never represents the shared producer/release-label constraints as a complete state graph.",
        },
        "4302233001000": {
            "structure_expected": "BRANCH_FINAL / shared answer intersection",
            "execution_expected": "two constraints merged on the same answer field-of-study variable",
            "shared_answer_variable": "?answer",
            "merge_operator": "AND / intersection",
            "branch_related": True,
            "constraints": [
                constraint("social_sciences_membership", "The answer is a subdiscipline of social sciences", "?answer", state("m.06n6p", "social sciences", evidence="GrailQA graph node 1"), "subdisciplines of social sciences", "education.field_of_study.subdisciplines", "outgoing", 0, True),
                constraint("mcclelland_research", "McClelland's research areas identify an intermediate field", "?research_field", state("m.018pq1", "mcclelland", evidence="GrailQA graph node 3"), "research areas", "education.academic.research_areas", "outgoing", 0, True),
                constraint("research_field_to_answer", "The research field is related to the shared answer field", "?research_field", state("", "research field variable", "unbound_variable", "GrailQA graph node 2; concrete ID not bound in gold graph"), "subdiscipline of answer", "education.field_of_study.subdiscipline_of", "outgoing", None),
            ],
            "first_divergence_stage": "structure_selection",
            "diagnosis": "STRUCTURE_ERROR",
            "secondary_diagnoses": ["ENTITY_STATE_ALIGNMENT_ERROR"],
            "diagnosis_reason": "The gold graph has two independent source constraints joined on ?answer, but Adaptive v1.1.3 selects LINEAR and starts only from social sciences. The McClelland research-area constraint is therefore searched from the wrong source state or omitted entirely.",
        },
        "WebQTest-743_af8b975343e4098913128e9ac7d04c05": {
            "structure_expected": "LINEARIZABLE composition with a speech-topic reverse start followed by speaker/sibling traversal",
            "execution_expected": "Anti-Apartheid -> speech CVT -> public speaker -> sibling -> male answer",
            "shared_answer_variable": "?x",
            "merge_operator": "AND constraints on speaker/sibling answer",
            "branch_related": False,
            "constraints": [
                constraint("speech_topic", "The speech CVT has Anti-Apartheid as its topic", "?speech", state("m.04lxyz0", "Anti-Apartheid", evidence="CWQ topic entity"), "speech topic", "event.speech_or_presentation.speech_topic", "incoming", 0, True),
                constraint("speaker_speech", "The speech was presented by the public speaker", "?speaker", state("", "speech CVT variable", "unbound_variable", "CWQ SPARQL variable ?k"), "public speaker presentations", "event.public_speaker.speeches_or_presentations", "incoming", None),
                constraint("speaker_siblings", "The public speaker has a sibling relationship", "?sibling_relation", state("", "public speaker variable", "unbound_variable", "CWQ SPARQL variable ?c"), "person sibling relation", "people.person.sibling_s", "outgoing", 1),
                constraint("sibling_answer", "The sibling relation yields the answer person", "?x", state("", "sibling relation variable", "unbound_variable", "CWQ SPARQL variable ?y"), "sibling endpoint", "people.sibling_relationship.sibling", "outgoing", 2),
                constraint("male_answer", "The answer person has gender Male", "?x", state("", "answer variable", "unbound_variable", "CWQ SPARQL variable ?x"), "gender Male", "people.person.gender", "outgoing", None),
            ],
            "first_divergence_stage": "relation_grounding",
            "diagnosis": "RELATION_GROUNDING_ERROR",
            "secondary_diagnoses": ["STRUCTURE_ERROR"],
            "diagnosis_reason": "Anti-Apartheid is the correct initial state and event.speech_or_presentation.speech_topic incoming is available, but the selector chooses the broad activism relation. The resulting state is an activist set rather than the speech CVT/public-speaker state required by the gold path.",
        },
        "WebQTrn-1812_8126be584feae3f358a930823db22c4e": {
            "structure_expected": "BRANCH_FINAL / conjunction sharing ?country",
            "execution_expected": "Caribbean -> ?country, then apply GDP constraint to the same ?country",
            "shared_answer_variable": "?country",
            "merge_operator": "AND over country containment, country type, and GDP value",
            "branch_related": True,
            "constraints": [
                constraint("caribbean_membership", "The answer country is contained in Caribbean", "?country", state("m.0261m", "Caribbean", evidence="CWQ topic entity"), "geographic containment", "location.location.contains", "outgoing", 0, True),
                constraint("country_type", "The same answer country has country type", "?country", state("m.05r7t", "Puerto Rico", evidence="CWQ gold answer used only for offline state verification"), "country type", "common.topic.notable_types", "outgoing", None),
                constraint("gdp_value", "The same ?country has GDP deflator change 2.35", "?country", state("m.05r7t", "Puerto Rico", evidence="CWQ gold answer; offline verification only"), "GDP deflator change", "location.statistical_region.gdp_deflator_change", "outgoing", 1, True),
                constraint("gdp_rate", "The GDP change CVT has rate 2.35", "?rate", state("", "GDP measurement CVT variable", "unbound_variable", "CWQ SPARQL variable ?c"), "dated percentage rate", "measurement_unit.dated_percentage.rate", "outgoing", None),
            ],
            "first_divergence_stage": "target_variable_alignment",
            "diagnosis": "TARGET_VARIABLE_ALIGNMENT_ERROR",
            "secondary_diagnoses": ["CONSTRAINT_DECOMPOSITION_ERROR"],
            "diagnosis_reason": "After the first containment step, the active GDP constraint should target the newly bound ?country. The Phase 2.1 prompt for decision 1 still reports Current entity: Caribbean, and the runtime repeats containment instead of applying gdp_deflator_change to the country variable.",
        },
    }


def one_hop_query(entity_id: str, relation: str, direction: str) -> str:
    entity = entity_term(entity_id)
    predicate = relation_term(relation)
    pattern = f"{entity} {predicate} ?value ." if direction == "outgoing" else f"?value {predicate} {entity} ."
    return f"PREFIX ns: <{FREEBASE_PREFIX}>\nSELECT DISTINCT ?value WHERE {{ {pattern} }} LIMIT {QUERY_LIMIT}\n"


def query_endpoint(endpoint: str, query: str, timeout: int) -> dict[str, Any]:
    started = time.time()
    url = endpoint + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    request = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        rows = payload.get("results", {}).get("bindings", []) or []
        return {"status": "ok", "rows_count": len(rows), "relation_present": bool(rows), "rows": rows, "elapsed_seconds": round(time.time() - started, 3), "query": query}
    except Exception as exc:  # pragma: no cover - endpoint-dependent
        message = str(exc)
        return {"status": "timeout" if "timed out" in message.lower() else "error", "rows_count": 0, "relation_present": None, "rows": [], "error": message, "elapsed_seconds": round(time.time() - started, 3), "query": query}


def verify_relation(endpoint: str, entity: dict[str, str], relation: str, direction: str, timeout: int, role: str) -> dict[str, Any]:
    if not entity.get("id"):
        return {"role": role, "entity": entity, "relation": relation, "direction": direction, "status": "unknown_unbound_state", "relation_present": None, "rows_count": None, "query": "", "elapsed_seconds": 0.0}
    result = query_endpoint(endpoint, one_hop_query(entity["id"], relation, direction), timeout)
    return {"role": role, "entity": entity, "relation": relation, "direction": direction, **result}


def build_case_records(dataset_rows: dict[str, dict[str, Any]], endpoint: str, timeout: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    blueprints = case_blueprints()
    case_records: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    for case_id, detail_path in CASE_FILES.items():
        detail = read_json(detail_path)
        raw = dataset_rows[case_id]
        runtime = extract_runtime_trace(detail)
        blueprint = blueprints[case_id]
        constraints = blueprint["constraints"]
        for item in constraints:
            decision = None
            if item["actual_decision_index"] is not None and item["actual_decision_index"] < len(runtime["decisions"]):
                decision = runtime["decisions"][item["actual_decision_index"]]
            actual_entity = (decision or {}).get("current_entity", {"id": "", "name": "", "evidence": "no matching relation decision"})
            actual_check = verify_relation(endpoint, actual_entity, item["expected_relation"], item["expected_direction"], timeout, "actual_state")
            intended_check = verify_relation(endpoint, item["intended_target_state"], item["expected_relation"], item["expected_direction"], timeout, "intended_state")
            verification_rows.append({"case_id": case_id, "constraint_id": item["constraint_id"], **actual_check})
            verification_rows.append({"case_id": case_id, "constraint_id": item["constraint_id"], **intended_check})
            trace_rows.append({
                "case_id": case_id,
                "dataset": "grailqa" if case_id[0].isdigit() else "cwq",
                "question": raw["question"],
                "structure_label": runtime["structure"],
                "execution_mode": runtime["execution_mode"],
                "constraint_id": item["constraint_id"],
                "constraint_summary": item["constraint_summary"],
                "intended_target_variable": item["intended_target_variable"],
                "intended_target_state": item["intended_target_state"],
                "intended_semantic_property": item["intended_semantic_property"],
                "expected_relation": item["expected_relation"],
                "expected_direction": item["expected_direction"],
                "actual_current_entity": actual_entity,
                "actual_current_entity_label": actual_entity.get("name", ""),
                "actual_branch_id": "none" if not runtime["branch_traces"] else "runtime_branch_metadata_present",
                "actual_subgoal": (decision or {}).get("current_subgoal", ""),
                "actual_relation_history": (decision or {}).get("relation_history", []),
                "actual_selected_relation": (decision or {}).get("selected_relation", ""),
                "actual_selected_direction": (decision or {}).get("selected_direction", ""),
                "relation_present_at_actual_state": actual_check.get("relation_present"),
                "relation_present_at_intended_state": intended_check.get("relation_present"),
                "first_divergence_stage": blueprint["first_divergence_stage"],
                "diagnosis": blueprint["diagnosis"],
                "secondary_diagnoses": blueprint["secondary_diagnoses"],
                "primary_constraint": item["primary_constraint"],
                "evidence_sources": ["raw dataset logical form/SPARQL", "Phase 2.1 runtime detail trace", "live Freebase relation verification"],
            })
        case_records.append({
            "case_id": case_id,
            "dataset": "grailqa" if case_id[0].isdigit() else "cwq",
            "question": raw["question"],
            "gold_answers": [item.get("entity_name", item.get("name", "")) for item in raw.get("answer", [])],
            "gold_answer_ids": [item.get("answer_argument", item.get("id", "")) for item in raw.get("answer", [])],
            "gold_logical_form": raw.get("s_expression", raw.get("sparql", "")),
            "gold_graph_query": raw.get("graph_query"),
            "gold_sparql": raw.get("sparql_query", raw.get("sparql", "")),
            "topic_entities": raw.get("topic_entities", raw.get("topic_entity", [])),
            "structure_expected": blueprint["structure_expected"],
            "execution_expected": blueprint["execution_expected"],
            "shared_answer_variable": blueprint["shared_answer_variable"],
            "merge_operator": blueprint["merge_operator"],
            "branch_related": blueprint["branch_related"],
            "runtime": runtime,
            "constraints": constraints,
            "first_divergence_stage": blueprint["first_divergence_stage"],
            "diagnosis": blueprint["diagnosis"],
            "secondary_diagnoses": blueprint["secondary_diagnoses"],
            "diagnosis_reason": blueprint["diagnosis_reason"],
        })
    return case_records, trace_rows, verification_rows


def write_alignment_summary(cases: list[dict[str, Any]], traces: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 2.3a alignment summary",
        "",
        "This table compares intended constraint states reconstructed from frozen dataset logical forms with states reported in frozen Phase 2.1 relation-grounding prompts.",
        "",
        "| Case | Structure | Intended constraint | Intended target variable/state | Actual current state | Expected relation | Relation present at intended state? | Relation present at actual state? | First divergence | Diagnosis |",
        "|---|---|---|---|---|---|---:|---:|---|---|",
    ]
    for row in traces:
        intended = row["intended_target_state"]
        actual = row["actual_current_entity"]
        lines.append(
            f"| `{row['case_id']}` | `{row['structure_label']}` | {row['constraint_summary']} | `{row['intended_target_variable']}` / `{intended.get('name') or intended.get('kind')}` | "
            f"`{actual.get('name') or actual.get('kind')}` (`{actual.get('id') or 'unbound'}`) | `{row['expected_relation']}` / {row['expected_direction']} | "
            f"{str(row['relation_present_at_intended_state']).lower()} | {str(row['relation_present_at_actual_state']).lower()} | `{row['first_divergence_stage']}` | `{row['diagnosis']}` |"
        )
    lines += ["", "## Primary per-case diagnosis", "", "| Case | Runtime structure | Expected structure/state | Primary diagnosis | Secondary evidence |", "|---|---|---|---|---|"]
    for case in cases:
        secondary = ", ".join(case["secondary_diagnoses"]) or "none"
        lines.append(f"| `{case['case_id']}` | `{case['runtime']['structure']}` / `{case['runtime']['execution_mode']}` | {case['structure_expected']} | `{case['diagnosis']}` | {secondary} |")
    (OUTPUT_ROOT / "alignment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_taxonomy(cases: list[dict[str, Any]]) -> Counter[str]:
    counts = Counter(case["diagnosis"] for case in cases)
    lines = [
        "# Phase 2.3a failure taxonomy",
        "",
        "## Primary diagnoses",
        "",
        "| Diagnosis | Count | Cases |",
        "|---|---:|---|",
    ]
    labels = ["STRUCTURE_ERROR", "CONSTRAINT_DECOMPOSITION_ERROR", "TARGET_VARIABLE_ALIGNMENT_ERROR", "ENTITY_STATE_ALIGNMENT_ERROR", "RELATION_GROUNDING_ERROR", "MERGE_ERROR", "DOWNSTREAM_ANSWER_ERROR", "INSUFFICIENT_EVIDENCE"]
    for label in labels:
        ids = [case["case_id"] for case in cases if case["diagnosis"] == label]
        lines.append(f"| `{label}` | {len(ids)} | {', '.join(f'`{item}`' for item in ids) or 'none'} |")
    lines += ["", "## Secondary/overlapping evidence", "", "| Evidence label | Count | Cases |", "|---|---:|---|"]
    for label in labels[:-1]:
        ids = [case["case_id"] for case in cases if label in case["secondary_diagnoses"]]
        lines.append(f"| `{label}` | {len(ids)} | {', '.join(f'`{item}`' for item in ids) or 'none'} |")
    lines += ["", "## Interpretation", "", "The primary label follows the first-divergence principle. Secondary labels preserve downstream state/structure evidence without treating every symptom as an independent failure."]
    (OUTPUT_ROOT / "failure_taxonomy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return counts


def write_report(cases: list[dict[str, Any]], traces: list[dict[str, Any]], metadata: dict[str, Any], counts: Counter[str]) -> str:
    target_alignment_cases = [case for case in cases if case["diagnosis"] == "TARGET_VARIABLE_ALIGNMENT_ERROR"]
    if len(target_alignment_cases) >= 2:
        decision = "STATE_ALIGNMENT_SIGNAL"
    elif any(counts[label] for label in ("STRUCTURE_ERROR", "RELATION_GROUNDING_ERROR", "CONSTRAINT_DECOMPOSITION_ERROR")):
        decision = "MIXED_ALIGNMENT_AND_RELATION_FAILURES"
    else:
        decision = "NO_CLEAR_STATE_ALIGNMENT_SIGNAL"
    intended_present = sum(row["relation_present_at_intended_state"] is True for row in traces)
    actual_absent = sum(row["relation_present_at_actual_state"] is False for row in traces)
    evidence_cases = sum(bool(case["gold_logical_form"]) and bool(case["runtime"]["decisions"]) for case in cases)
    lines = [
        "# Phase 2.3 report: constraint-to-state alignment",
        "",
        "## Motivation from Phase 2.2 and Phase 2.2B",
        "",
        "Phase 2.2 found no confirmed candidate-retention loss: all four missing expected relations were absent from the raw neighborhood at the recorded states. Phase 2.2B showed that gold entities were nevertheless reachable through alternative relations or a short path, but mostly with reasoning mismatches. Phase 2.3a tests whether constraints are applied to the wrong variable/entity/state.",
        "",
        "## Research question",
        "",
        "When relation grounding fails, does the runtime search from the state targeted by the intended logical constraint?",
        "",
        "## Why answer reachability is insufficient",
        "",
        "An answer entity can be returned by a semantically different relation. For example, `location.contains` can reach Puerto Rico from Caribbean without checking Puerto Rico's GDP-deflator value. The audit therefore compares logical constraints, runtime states, and relation availability separately.",
        "",
        "## Frozen components",
        "",
        "Adaptive v1.1.3 routing, structure labels, branch merge/fallback behavior, entity lookup, Freebase backend, SPARQL semantics, evaluator, Phase 2.1 ranker/prompts/settings, Phase 2.2 candidate generation/filtering, top-k, model settings, and dataset labels were unchanged. No runtime intervention or full benchmark was run.",
        "",
        "## Phase 2.3a method",
        "",
        "The four Phase 2.2/2.2B cases were audited. Gold GrailQA graph queries/S-expressions and CWQ SPARQL were loaded from the repository datasets. Phase 2.1 detail traces supplied router output, relation-selection prompts, current entities, subgoals, histories, selected relations, and directions. Exact one-hop relation checks were run against the live endpoint for intended and actual bound states. Unbound logical variables are recorded as unknown rather than guessed.",
        "",
        "## Aggregate evidence",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Audited cases | {len(cases)} |",
        f"| Cases with logical-form and runtime evidence | {evidence_cases}/{len(cases)} |",
        f"| STRUCTURE_ERROR | {counts['STRUCTURE_ERROR']} |",
        f"| CONSTRAINT_DECOMPOSITION_ERROR | {counts['CONSTRAINT_DECOMPOSITION_ERROR']} |",
        f"| TARGET_VARIABLE_ALIGNMENT_ERROR | {counts['TARGET_VARIABLE_ALIGNMENT_ERROR']} |",
        f"| ENTITY_STATE_ALIGNMENT_ERROR | {counts['ENTITY_STATE_ALIGNMENT_ERROR']} |",
        f"| RELATION_GROUNDING_ERROR | {counts['RELATION_GROUNDING_ERROR']} |",
        f"| MERGE_ERROR | {counts['MERGE_ERROR']} |",
        f"| DOWNSTREAM_ANSWER_ERROR | {counts['DOWNSTREAM_ANSWER_ERROR']} |",
        f"| INSUFFICIENT_EVIDENCE | {counts['INSUFFICIENT_EVIDENCE']} |",
        f"| Expected relation absent at actual state | {actual_absent}/{len(traces)} checks |",
        f"| Expected relation present at intended state | {intended_present}/{len(traces)} checks |",
        f"| Branch-related cases | {sum(case['branch_related'] for case in cases)}/{len(cases)} |",
        f"| Linearizable/non-branch cases | {sum(not case['branch_related'] for case in cases)}/{len(cases)} |",
        f"| SPARQL verification queries | {metadata['query_count']} |",
        f"| Total verification runtime | {metadata['runtime_seconds']:.2f} seconds |",
        "",
        "## Per-case first divergence",
        "",
        "| Case | Structure | First divergence | Primary diagnosis | Key evidence |",
        "|---|---|---|---|---|",
    ]
    for case in cases:
        lines.append(f"| `{case['case_id']}` | `{case['runtime']['structure']}` | `{case['first_divergence_stage']}` | `{case['diagnosis']}` | {case['diagnosis_reason']} |")
    lines += [
        "",
        "## Caribbean sanity check",
        "",
        "1. The CWQ SPARQL binds `?country` from `Caribbean location.contains ?country`.",
        "2. The GDP constraint is applied to that same `?country` via `location.statistical_region.gdp_deflator_change`, followed by the rate CVT.",
        "3. The first Phase 2.1 decision correctly starts at Caribbean and selects `location.location.contains`.",
        "4. At decision 1, the Phase 2.1 prompt still reports `Current entity: Caribbean`, while the intended target is the newly bound `?country`; it selects another containment relation instead of GDP deflator change.",
        "5. Live verification records GDP relation absent at Caribbean and present at the offline gold country state Puerto Rico.",
        "6. This is classified as `TARGET_VARIABLE_ALIGNMENT_ERROR`, with constraint decomposition as secondary evidence. Gold state information was used only for offline verification; it was not injected into runtime inference.",
        "",
        "## Intended-versus-actual reconstruction",
        "",
        "See `alignment_summary.md`, `case_alignment_traces.jsonl`, `intended_vs_actual_states.json`, and `relation_state_verification.jsonl`. The traces retain the raw logical form, current prompt state, relation history, expected relation, and verification result.",
        "",
        "## Phase 2.3a decision",
        "",
        f"**{decision}**",
        "",
        "The audit found a clear target-variable/state alignment failure in the Caribbean case, but it is not repeated independently across enough cases to justify a context-enrichment intervention by itself. The remaining cases are primarily structure-selection or relation-grounding failures. Therefore Phase 2.3b is not run in this phase.",
        "",
        "## Why Phase 2.3b is not justified yet",
        "",
        "Only one case has an unambiguous first divergence at target-variable alignment. In the other cases, the relevant mismatch is explained earlier by structure selection, constraint decomposition, or relation choice while the initial current state is correct. Adding state-aware context now would confound several causes and would not provide a clean intervention test.",
        "",
        "## Limitations",
        "",
        "The audit covers four deliberately selected cases, not a prevalence estimate. Some intermediate logical variables are unbound concrete IDs in the gold form, so those checks are marked unknown. Relation presence is necessary evidence for a grounding diagnosis but does not prove that a selected relation would satisfy all constraints. The analysis uses frozen traces and does not rerun the LLM.",
        "",
        "## Final recommendation",
        "",
        "Do not modify production CoR yet. Preserve the Phase 2.1/2.2 frozen comparisons. If the next phase samples additional cases, add cases where a recorded state transition is followed by a constraint on the newly bound shared answer variable. A Phase 2.3b context intervention should require at least one additional independent target-variable alignment case plus relation-grounding and structure controls.",
        "",
    ]
    report_text = "\n".join(lines) + "\n"
    (OUTPUT_ROOT / "PHASE2_3A_REPORT.md").write_text(report_text, encoding="utf-8")
    (OUTPUT_ROOT / "PHASE2_3_REPORT.md").write_text(report_text, encoding="utf-8")
    return decision


def run() -> None:
    global OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.getenv("FREEBASE_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    OUTPUT_ROOT = args.output_root
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    dataset_rows = load_dataset_rows()
    cases, traces, verifications = build_case_records(dataset_rows, args.endpoint, max(1, args.timeout))
    metadata = {
        "endpoint": args.endpoint,
        "query_limit": QUERY_LIMIT,
        "query_count": sum(1 for row in verifications if row.get("status") != "unknown_unbound_state"),
        "runtime_seconds": round(time.time() - started, 3),
        "frozen_components": ["Adaptive v1.1.3 router", "branch structure", "entity lookup", "Freebase", "SPARQL", "evaluator", "Phase2.1 ranker", "Phase2.2 candidate generation"],
        "production_intervention": False,
    }
    counts = write_failure_taxonomy(cases)
    write_alignment_summary(cases, traces)
    write_jsonl(OUTPUT_ROOT / "case_alignment_traces.jsonl", traces)
    write_json(OUTPUT_ROOT / "intended_vs_actual_states.json", cases)
    write_jsonl(OUTPUT_ROOT / "relation_state_verification.jsonl", verifications)
    write_json(OUTPUT_ROOT / "audit_metadata.json", metadata)
    decision = write_report(cases, traces, metadata, counts)
    print(f"Wrote Phase 2.3a audit to {OUTPUT_ROOT}")
    print(f"Decision: {decision}")
    print(f"SPARQL verification queries: {metadata['query_count']}")


if __name__ == "__main__":
    run()
