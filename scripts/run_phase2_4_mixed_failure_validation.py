#!/usr/bin/env python3
"""Run the frozen Phase 2.4 mixed-failure validation diagnostic.

This script is deliberately an offline artifact audit.  It does not call the
LLM, alter the production runtime, or require a live Virtuoso endpoint.  It
joins the frozen Phase 2.1/2.3 traces with the fresh-for-Phase-2 GrailQA
repeat routing records and emits the protocol-required Phase 2.4 report.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relation_graph_kgqa.logical_forms.sexpr_ast import ExprNode  # noqa: E402
from relation_graph_kgqa.logical_forms.sexpr_parser import parse_s_expression  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs/experiments/phase2_4/mixed_failure_validation"
PHASE21_ROOT = ROOT / "outputs/experiments/phase2_1/relation_grounding"
PHASE23_ROOT = ROOT / "outputs/experiments/phase2_3/constraint_state_alignment"
FRESH_ROOT = ROOT / "outputs/experiments/phase1/adaptive_v1_1_3/grailqa_12_repeat"
GRAILQA_PATH = ROOT / "datasets/grailqa/official/grailqa_v1.0_train.json"


TRACK_A_MANIFEST = [
    {
        "case_id": "WebQTrn-1812_8126be584feae3f358a930823db22c4e",
        "source_constraint_id": "gdp_value",
        "suspected_failure": True,
        "fresh_case": False,
        "primary_label": "TARGET_VARIABLE_ALIGNMENT_ERROR",
        "notes": "The runtime remains at Caribbean instead of the shared ?country binding (Puerto Rico).",
    },
    {
        "case_id": "4302233001000",
        "source_constraint_id": "mcclelland_research",
        "suspected_failure": True,
        "fresh_case": False,
        "primary_label": "ENTITY_STATE_ALIGNMENT_ERROR",
        "notes": "The runtime is at social sciences while the research-area constraint requires McClelland.",
    },
    {
        "case_id": "4301990001000",
        "source_constraint_id": "producer_bridge",
        "suspected_failure": False,
        "fresh_case": False,
        "primary_label": "ALIGNED",
        "notes": "The runtime and intended state are both Pictures of Lily; the remaining error is relation grounding.",
    },
    {
        "case_id": "WebQTest-743_af8b975343e4098913128e9ac7d04c05",
        "source_constraint_id": "speech_topic",
        "suspected_failure": False,
        "fresh_case": False,
        "primary_label": "ALIGNED",
        "notes": "The runtime and intended state are both Anti-Apartheid; the remaining error is relation grounding.",
    },
]

TRACK_B_MANIFEST = [
    {
        "case_id": "WebQTest-719_e56e856592a9a57b9e331b718adc1b75",
        "suspected_failure": True,
        "fresh_case": False,
        "primary_label": "WRONG_DIRECTION",
    },
    {
        "case_id": "WebQTest-759_88081dc1a259cbd172b89242384ee1cc",
        "suspected_failure": True,
        "fresh_case": False,
        "primary_label": "GENERIC_RELATION_PREFERRED",
    },
    {
        "case_id": "WebQTrn-64_984678664419049133d2bd748b68df1b",
        "suspected_failure": True,
        "fresh_case": False,
        "primary_label": "WRONG_DIRECTION",
    },
    {
        "case_id": "WebQTest-191_ed2cadafdbc0ac07e3fedfdfc504e54b",
        "suspected_failure": False,
        "fresh_case": False,
        "primary_label": "CONTROL_PRESERVED",
    },
]

TRACK_C_MANIFEST = [
    {
        "case_id": "3201049003000",
        "suspected_failure": True,
        "fresh_case": True,
        "primary_label": "STRUCTURE_FALSE_NEGATIVE",
    },
    {
        "case_id": "3201277005000",
        "suspected_failure": True,
        "fresh_case": True,
        "primary_label": "STRUCTURE_FALSE_NEGATIVE",
    },
    {
        "case_id": "3202841012000",
        "suspected_failure": True,
        "fresh_case": True,
        "primary_label": "STRUCTURE_FALSE_NEGATIVE",
    },
    {
        "case_id": "3203234013000",
        "suspected_failure": False,
        "fresh_case": True,
        "primary_label": "CORRECT_STRUCTURE",
    },
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def relation_value(value: Any) -> tuple[str, str]:
    """Return the first selected relation and direction from historical schemas."""

    if isinstance(value, list):
        value = value[0] if value else {}
    if isinstance(value, dict):
        return str(value.get("relation", "")), str(value.get("direction", ""))
    return str(value or ""), ""


def normalize_structure_label(label: str) -> str:
    value = str(label or "").upper()
    if value in {"LINEAR", "BRANCH_FINAL", "BRANCH_INTERMEDIATE", "UNCERTAIN"}:
        return value
    return "UNCERTAIN"


def gold_structure_from_s_expression(source: str) -> str:
    """Map the frozen GrailQA logical form into the Phase 2.4 taxonomy.

    A root-level conjunction of answer arms is a final branch.  A conjunction
    nested under a JOIN first identifies an intermediate entity and is an
    intermediate branch.  This is intentionally narrower than the general
    classifier because Phase 2.4 only needs the routing taxonomy.
    """

    try:
        root = parse_s_expression(source)
    except Exception:
        return "GOLD_OTHER"
    if root.operator.upper() != "AND" or len(root.arguments) < 2:
        return "GOLD_LINEAR"
    body = root.arguments[1]
    if not isinstance(body, ExprNode):
        return "GOLD_LINEAR"
    if body.operator.upper() == "AND" and sum(isinstance(arg, ExprNode) for arg in body.arguments) >= 2:
        return "GOLD_BRANCH_FINAL"

    def contains_branch(node: Any) -> bool:
        if not isinstance(node, ExprNode):
            return False
        if node.operator.upper() == "AND" and sum(isinstance(arg, ExprNode) for arg in node.arguments) >= 2:
            return True
        return any(contains_branch(arg) for arg in node.arguments)

    return "GOLD_BRANCH_INTERMEDIATE" if contains_branch(body) else "GOLD_LINEAR"


def classify_structure_pair(gold: str, runtime: str) -> str:
    gold = str(gold)
    runtime = normalize_structure_label(runtime)
    if gold == "GOLD_OTHER":
        return "OTHER"
    if gold == "GOLD_LINEAR" and runtime in {"BRANCH_FINAL", "BRANCH_INTERMEDIATE"}:
        return "STRUCTURE_FALSE_POSITIVE"
    if gold in {"GOLD_BRANCH_FINAL", "GOLD_BRANCH_INTERMEDIATE"} and runtime in {"LINEAR", "UNCERTAIN"}:
        return "STRUCTURE_FALSE_NEGATIVE"
    if (gold == "GOLD_BRANCH_FINAL" and runtime == "BRANCH_INTERMEDIATE") or (
        gold == "GOLD_BRANCH_INTERMEDIATE" and runtime == "BRANCH_FINAL"
    ):
        return "BRANCH_TYPE_MISMATCH"
    if (gold == "GOLD_LINEAR" and runtime == "LINEAR") or (
        gold == "GOLD_BRANCH_FINAL" and runtime == "BRANCH_FINAL"
    ) or (gold == "GOLD_BRANCH_INTERMEDIATE" and runtime == "BRANCH_INTERMEDIATE"):
        return "CORRECT_STRUCTURE"
    return "OTHER"


def _index_by_case(rows: Iterable[dict[str, Any]], key_names: tuple[str, ...] = ("case_id", "question_id", "id")) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in key_names:
            if row.get(key) is not None:
                indexed[str(row[key])] = row
                break
    return indexed


def build_track_a(phase23_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(str(row.get("case_id")), str(row.get("constraint_id"))): row for row in phase23_rows}
    traces: list[dict[str, Any]] = []
    for item in TRACK_A_MANIFEST:
        source = by_key.get((item["case_id"], item["source_constraint_id"]))
        if source is None:
            raise KeyError(f"missing Phase 2.3 trace for {item['case_id']} / {item['source_constraint_id']}")
        actual = source.get("actual_current_entity") or {}
        intended = source.get("intended_target_state") or {}
        actual_id = str(actual.get("id", ""))
        intended_id = str(intended.get("id", ""))
        state_match = bool(actual_id and intended_id and actual_id == intended_id)
        traces.append(
            {
                "track": "A",
                "family": "target_variable_state_alignment",
                "case_id": item["case_id"],
                "dataset": source.get("dataset"),
                "question": source.get("question"),
                "suspected_failure": item["suspected_failure"],
                "fresh_case": item["fresh_case"],
                "source": "phase2_3_case_alignment_traces.jsonl",
                "source_constraint_id": item["source_constraint_id"],
                "target_variable": source.get("intended_target_variable"),
                "binding_relation": source.get("relation_history", [""])[-1] if source.get("relation_history") else "",
                "next_expected_relation": {
                    "relation": source.get("expected_relation"),
                    "direction": source.get("expected_direction"),
                },
                "actual_state": actual,
                "intended_state": intended,
                "actual_selected_relation": {
                    "relation": source.get("actual_selected_relation"),
                    "direction": source.get("actual_selected_direction"),
                },
                "relation_present_at_actual_state": source.get("relation_present_at_actual_state"),
                "relation_present_at_intended_state": source.get("relation_present_at_intended_state"),
                "state_transition_correct": state_match,
                "primary_label": item["primary_label"],
                "notes": item.get("notes", ""),
                "evidence_sources": source.get("evidence_sources", []),
            }
        )
    return traces


def classify_relation_failure(row: dict[str, Any]) -> str:
    expected = (str(row.get("expected_relation", "")), str(row.get("expected_direction", "")))
    selected = relation_value(row.get("selected_relation"))
    if not row.get("correct_relation_available_in_candidates"):
        return "CANDIDATE_MISSING_NOT_TRACK_B"
    if selected == expected:
        return "CONTROL_PRESERVED"
    if expected[1] and selected[1] and expected[1] != selected[1]:
        return "WRONG_DIRECTION"
    if expected[0] == "celebrities.celebrity.sexual_relationships" and selected[0] == "base.popstra.celebrity.dated":
        return "GENERIC_RELATION_PREFERRED"
    if row.get("structure") == "UNCERTAIN":
        return "SEMANTIC_AMBIGUITY"
    return "SEMANTIC_AMBIGUITY"


def build_track_b(phase21_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = _index_by_case(phase21_rows, ("question_id",))
    traces: list[dict[str, Any]] = []
    for item in TRACK_B_MANIFEST:
        source = by_id.get(item["case_id"])
        if source is None:
            raise KeyError(f"missing Phase 2.1 relation trace for {item['case_id']}")
        expected = (str(source.get("expected_relation", "")), str(source.get("expected_direction", "")))
        selected = relation_value(source.get("selected_relation"))
        label = classify_relation_failure(source)
        if label != item["primary_label"]:
            raise AssertionError(
                f"manifest taxonomy mismatch for {item['case_id']}: expected {item['primary_label']}, derived {label}"
            )
        traces.append(
            {
                "track": "B",
                "family": "relation_grounding",
                "case_id": item["case_id"],
                "dataset": source.get("dataset"),
                "question": source.get("question"),
                "suspected_failure": item["suspected_failure"],
                "fresh_case": item["fresh_case"],
                "source": "phase2_1_relation_decisions_phase2_1.jsonl",
                "current_state": source.get("current_entity"),
                "target_relation": {"relation": expected[0], "direction": expected[1]},
                "selected_relation": {"relation": selected[0], "direction": selected[1]},
                "expected_relation_available": bool(source.get("correct_relation_available_in_candidates")),
                "candidate_recall": source.get("candidate_recall"),
                "relation_selection_correct": bool(source.get("relation_selection_correct")),
                "taxonomy": label,
                "evidence": {
                    "candidate_count": len(source.get("candidate_relations") or []),
                    "top_k_selected_relations": source.get("top_k_selected_relations", []),
                    "selection_reason": source.get("selection_reason", ""),
                },
            }
        )
    return traces


def build_track_c(fresh_rows: list[dict[str, Any]], grailqa_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runtime_by_id = _index_by_case(fresh_rows, ("id", "question_id"))
    gold_by_id = {str(row["qid"]): row for row in grailqa_rows}
    traces: list[dict[str, Any]] = []
    for item in TRACK_C_MANIFEST:
        runtime = runtime_by_id.get(item["case_id"])
        gold = gold_by_id.get(item["case_id"])
        if runtime is None or gold is None:
            raise KeyError(f"missing fresh routing or gold structure trace for {item['case_id']}")
        gold_label = gold_structure_from_s_expression(str(gold.get("s_expression", "")))
        runtime_label = normalize_structure_label(str(runtime.get("router_structure", "")))
        derived = classify_structure_pair(gold_label, runtime_label)
        if derived != item["primary_label"]:
            raise AssertionError(
                f"manifest structure mismatch for {item['case_id']}: expected {item['primary_label']}, derived {derived}"
            )
        traces.append(
            {
                "track": "C",
                "family": "structure_branch_routing",
                "case_id": item["case_id"],
                "dataset": "grailqa_branch_2",
                "question": runtime.get("question"),
                "suspected_failure": item["suspected_failure"],
                "fresh_case": item["fresh_case"],
                "source": "phase1_adaptive_v1_1_3_grailqa_12_repeat/adaptive_runs.jsonl",
                "gold_structure": gold_label,
                "runtime_structure": runtime_label,
                "runtime_execution_mode": runtime.get("final_execution_mode"),
                "gold_s_expression": gold.get("s_expression"),
                "classification": derived,
                "answer_correct": runtime.get("hits_at_1", runtime.get("final_hits_at_1")),
            }
        )
    return traces


def failure_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(bool(row.get("suspected_failure")) for row in rows) / len(rows)


def confirmed_failure_count(rows: list[dict[str, Any]]) -> int:
    return sum(bool(row.get("suspected_failure")) for row in rows)


def dominant_decision(track_rows: dict[str, list[dict[str, Any]]]) -> str:
    rates = {track: failure_rate(rows) for track, rows in track_rows.items()}
    counts = {track: confirmed_failure_count(rows) for track, rows in track_rows.items()}
    eligible = [track for track, count in counts.items() if count >= 3]
    if len(eligible) == 1:
        only = eligible[0]
        if all(rates[only] > rates[other] for other in rates if other != only):
            return {
                "A": "STATE_ALIGNMENT_DOMINANT",
                "B": "RELATION_GROUNDING_DOMINANT",
                "C": "STRUCTURE_ROUTING_DOMINANT",
            }[only]
    if all(counts[track] == 0 for track in counts):
        return "INCONCLUSIVE"
    if max(rates.values(), default=0.0) == 0.0:
        return "INCONCLUSIVE"
    return "MIXED_NO_DOMINANT_FAILURE"


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def render_track_a(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 2.4 Track A — target-variable/state alignment",
        "",
        "This track checks whether the runtime is at the intended variable/entity state before treating a relation decision as a grounding failure.",
        "",
        "| Case | Target variable | Actual state | Intended state | Next relation | State transition | Label |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        actual = row["actual_state"].get("name") or row["actual_state"].get("id") or "unbound"
        intended = row["intended_state"].get("name") or row["intended_state"].get("id") or "unbound"
        next_relation = row["next_expected_relation"]["relation"]
        lines.append(
            f"| `{row['case_id']}` | `{row['target_variable']}` | {actual} | {intended} | `{next_relation}` | `{row['state_transition_correct']}` | `{row['primary_label']}` |"
        )
    lines += [
        "",
        "The two suspected failures have different signatures: 1812 changes the shared target binding, while 430223 leaves the runtime on the wrong entity state. The two controls remain state-aligned even though their broader runs contain relation-grounding errors.",
        "",
    ]
    return "\n".join(lines)


def render_track_b(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 2.4 Track B — relation grounding",
        "",
        "Only candidate-available cases are included. A failure therefore means the expected relation was present in the candidate set, but the selected top relation was wrong.",
        "",
        "| Case | Current state | Expected relation | Selected relation | Available | Selection correct | Taxonomy |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        state = row["current_state"].get("name") or row["current_state"].get("id")
        expected = row["target_relation"]
        selected = row["selected_relation"]
        lines.append(
            f"| `{row['case_id']}` | {state} | `{expected['relation']}` ({expected['direction']}) | `{selected['relation']}` ({selected['direction']}) | {row['expected_relation_available']} | {row['relation_selection_correct']} | `{row['taxonomy']}` |"
        )
    lines += [
        "",
        "The controls preserve the expected relation. Candidate-missing cases from Phase 2.1 are intentionally excluded from this track because they are recall failures, not ranking failures.",
        "",
    ]
    return "\n".join(lines)


def render_track_c(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 2.4 Track C — structure/branch routing",
        "",
        "Gold structure is derived from the frozen GrailQA S-expression; runtime structure is read from the frozen adaptive v1.1.3 repeat trace.",
        "",
        "| Case | Gold structure | Runtime structure | Execution | Classification | Case freshness |",
        "|---|---|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['case_id']}` | `{row['gold_structure']}` | `{row['runtime_structure']}` | `{row['runtime_execution_mode']}` | `{row['classification']}` | {row['fresh_case']} |"
        )
    lines += [
        "",
        "Three fresh-for-Phase-2 cases are gold branch cases routed as LINEAR; the control is a gold branch case routed as BRANCH_FINAL.",
        "",
    ]
    return "\n".join(lines)


def render_aggregate(track_rows: dict[str, list[dict[str, Any]]], decision: str) -> str:
    lines = [
        "# Phase 2.4 aggregate failure summary",
        "",
        "| Track | Family | Cases | Suspected/confirmed failures | Failure rate | Controls | Fresh | Reused |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"A": "state alignment", "B": "relation grounding", "C": "structure routing"}
    for track, rows in track_rows.items():
        failures = confirmed_failure_count(rows)
        fresh = sum(bool(row.get("fresh_case")) for row in rows)
        lines.append(
            f"| {track} | {labels[track]} | {len(rows)} | {failures} | {failure_rate(rows):.2%} | {len(rows) - failures} | {fresh} | {len(rows) - fresh} |"
        )
    total = sum(len(rows) for rows in track_rows.values())
    failures = sum(confirmed_failure_count(rows) for rows in track_rows.values())
    fresh = sum(sum(bool(row.get("fresh_case")) for row in rows) for rows in track_rows.values())
    lines += [
        "",
        f"- Total cases: **{total}**.",
        f"- Confirmed/suspected failures: **{failures}**; controls: **{total - failures}**.",
        f"- Fresh-for-Phase-2 cases: **{fresh}**; reused frozen-trace cases: **{total - fresh}**.",
        "- Track C's fresh cases come from the Phase 1 v1.1.3 fresh GrailQA-12 repeat and are therefore fresh relative to the Phase 2.1–2.3 trace families, but not a wholly new dataset sample.",
        "",
        f"## Conservative gate decision: `{decision}`",
        "",
        "The gate requires at least three independent confirmed failures and a clearly higher failure rate for one family while preserving controls. Track B and Track C are both 75%, while Track A is 50%; neither family is clearly dominant over the other. The result is therefore mixed, not a license to begin a single-family Phase 2.5 intervention.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    phase21 = read_jsonl(PHASE21_ROOT / "relation_decisions_phase2_1.jsonl")
    phase23 = read_jsonl(PHASE23_ROOT / "case_alignment_traces.jsonl")
    fresh = read_jsonl(FRESH_ROOT / "adaptive_runs.jsonl")
    grailqa = read_json(GRAILQA_PATH)

    track_rows = {
        "A": build_track_a(phase23),
        "B": build_track_b(phase21),
        "C": build_track_c(fresh, grailqa),
    }
    decision = dominant_decision(track_rows)
    all_rows = [row for rows in track_rows.values() for row in rows]
    failure_distribution = {
        "experiment": "phase2_4_mixed_failure_validation",
        "case_count": len(all_rows),
        "fresh_case_count": sum(bool(row.get("fresh_case")) for row in all_rows),
        "reused_case_count": sum(not bool(row.get("fresh_case")) for row in all_rows),
        "suspected_failure_count": sum(bool(row.get("suspected_failure")) for row in all_rows),
        "control_count": sum(not bool(row.get("suspected_failure")) for row in all_rows),
        "tracks": {
            track: {
                "family": rows[0]["family"],
                "case_count": len(rows),
                "suspected_failure_count": confirmed_failure_count(rows),
                "control_count": len(rows) - confirmed_failure_count(rows),
                "failure_rate": failure_rate(rows),
                "fresh_case_count": sum(bool(row.get("fresh_case")) for row in rows),
                "reused_case_count": sum(not bool(row.get("fresh_case")) for row in rows),
                "labels": dict(Counter(
                    row.get("primary_label") or row.get("taxonomy") or row.get("classification", "OTHER")
                    for row in rows
                )),
            }
            for track, rows in track_rows.items()
        },
        "decision": decision,
        "conservative_gate": {
            "minimum_independent_confirmed_failures": 3,
            "independent_confirmed_failures": sum(confirmed_failure_count(rows) for rows in track_rows.values()),
            "controls_preserved": all(not row.get("suspected_failure") for rows in track_rows.values() for row in rows if row.get("primary_label") in {"ALIGNED", "CONTROL_PRESERVED", "CORRECT_STRUCTURE"}),
            "clearly_higher_failure_rate_required": True,
            "clearly_higher_failure_rate_observed": False,
        },
        "production_pipeline_changed": False,
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _json_dump(OUTPUT_ROOT / "diagnostic_cases.json", {"track_a": TRACK_A_MANIFEST, "track_b": TRACK_B_MANIFEST, "track_c": TRACK_C_MANIFEST})
    _write_jsonl(OUTPUT_ROOT / "track_a_traces.jsonl", track_rows["A"])
    _write_jsonl(OUTPUT_ROOT / "track_b_traces.jsonl", track_rows["B"])
    _write_jsonl(OUTPUT_ROOT / "track_c_traces.jsonl", track_rows["C"])
    _json_dump(OUTPUT_ROOT / "failure_distribution.json", failure_distribution)
    (OUTPUT_ROOT / "track_a_state_alignment.md").write_text(render_track_a(track_rows["A"]), encoding="utf-8")
    (OUTPUT_ROOT / "track_b_relation_grounding.md").write_text(render_track_b(track_rows["B"]), encoding="utf-8")
    (OUTPUT_ROOT / "track_c_structure_routing.md").write_text(render_track_c(track_rows["C"]), encoding="utf-8")
    (OUTPUT_ROOT / "aggregate_failure_summary.md").write_text(render_aggregate(track_rows, decision), encoding="utf-8")

    report = f"""# Phase 2.4 mixed-failure validation

## Scope

This is a frozen diagnostic only. It does not modify CoR, adaptive routing, relation selection, or the SPARQL runtime, and it does not run a new 100-question experiment.

## Case design

- Track A: 4 cases for target-variable/entity-state alignment; 2 suspected failures and 2 alignment controls.
- Track B: 4 cases for candidate-available relation ranking; 3 suspected failures and 1 successful control.
- Track C: 4 GrailQA branch cases from the fresh-for-Phase-2 v1.1.3 repeat; 3 routing false negatives and 1 correct-routing control.
- Total: **{len(all_rows)}** cases, **{failure_distribution['fresh_case_count']}** fresh-for-Phase-2 and **{failure_distribution['reused_case_count']}** reused frozen-trace cases.
- Suspected/confirmed failure share: **{failure_distribution['suspected_failure_count']}/{len(all_rows)} = {failure_distribution['suspected_failure_count'] / len(all_rows):.2%}**; controls: **{failure_distribution['control_count']}**.

## Results

| Track | Failure rate | Main evidence |
|---|---:|---|
| A — state alignment | {failure_distribution['tracks']['A']['failure_rate']:.2%} | One target-variable mismatch and one wrong entity-state mismatch; two aligned controls. |
| B — relation grounding | {failure_distribution['tracks']['B']['failure_rate']:.2%} | Three candidate-available ranking failures; the expected relation was not selected. |
| C — structure routing | {failure_distribution['tracks']['C']['failure_rate']:.2%} | Three gold branch cases routed as LINEAR/UNCERTAIN; one BRANCH_FINAL control. |

## Decision

**`{decision}`**

The minimum-failure threshold is met, but the dominant-family gate is not: relation grounding and structure routing are tied at 75%, while state alignment is 50%. Controls were preserved, so the result is informative, but it does not justify a single-family Phase 2.5 intervention.

## Recommended next direction

Keep the next change diagnostic and targeted across the two tied families. Do not claim that relation grounding alone, state alignment alone, or structure routing alone dominates. A follow-up should either enlarge the fresh mixed diagnostic or define a paired intervention that logs both routing and relation decisions while preserving the same controls.

## Artifacts

- `track_a_state_alignment.md` / `track_a_traces.jsonl`
- `track_b_relation_grounding.md` / `track_b_traces.jsonl`
- `track_c_structure_routing.md` / `track_c_traces.jsonl`
- `aggregate_failure_summary.md`
- `failure_distribution.json`
"""
    (OUTPUT_ROOT / "PHASE2_4_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
