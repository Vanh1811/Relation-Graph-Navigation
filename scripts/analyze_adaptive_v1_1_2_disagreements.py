#!/usr/bin/env python3
"""Create the v1.1.2 five-disagreement analysis artifacts.

This is an evidence/reporting script.  It reads the frozen GrailQA-25
results, detailed traces, and the bounded stability check; it does not call
the model or alter the Adaptive/CoR implementation.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_2"
GRAIL = EXP / "grailqa_25"
DETAILS = GRAIL / "details"
OUT = EXP / "disagreement_analysis"
STABILITY = OUT / "stability"

CASE_IDS = [
    "3201985002000",
    "3202928006000",
    "4302129014000",
    "3205391005000",
    "4300043017000",
]

CASE_META: dict[str, dict[str, Any]] = {
    "3201985002000": {
        "outcome": "ADAPTIVE_ONLY_CORRECT",
        "cause": "ANSWER_GENERATION_VARIATION",
        "secondary_cause": "ANSWER_SET_PRECISION",
        "win_class": "LIKELY_STOCHASTIC_OR_CASE_SPECIFIC",
        "regression_class": None,
        "reusable": False,
        "architecture_issue": False,
        "rationale": (
            "Both methods grounded the Republic of Mali and reached the same 105-person "
            "candidate universe. Adaptive was routed LINEAR and used the unchanged CoR "
            "executor; the historical Adaptive filter returned the gold person in addition "
            "to the CoR answer. The relation-prune direction and final answer list varied "
            "across runs, so the historical win is not reproducible as a routing improvement."
        ),
        "questions": [
            "Did Adaptive choose a different structure? No: LINEAR, which executes CoR.",
            "Did it choose a better relation? Not reliably; both runs used the same grounded candidate set, with inverse/forward relation ordering varying.",
            "Did branching cause the win? No.",
            "Generalization judgment: no evidence of a reusable improvement; treat as stochastic answer-set generation.",
        ],
    },
    "3202928006000": {
        "outcome": "ADAPTIVE_ONLY_CORRECT",
        "cause": "ANSWER_GENERATION_VARIATION",
        "secondary_cause": "ANSWER_SET_PRECISION",
        "win_class": "LIKELY_STOCHASTIC_OR_CASE_SPECIFIC",
        "regression_class": None,
        "reusable": False,
        "architecture_issue": False,
        "rationale": (
            "The historical CoR and Adaptive LINEAR traces have identical SPARQL histories "
            "and the same selected relation. The only material difference is the final "
            "filter answer text: CoR abbreviated the election name, while Adaptive returned "
            "the full gold label. This is answer-generation variation, not a structural or "
            "KG improvement. Repeats alternated between the abbreviated and full forms."
        ),
        "questions": [
            "Did Adaptive choose a different structure? No: both are the same linear path.",
            "Did it choose a better relation? No; the relation and SPARQL history are identical.",
            "Did branching cause the win? No.",
            "Generalization judgment: no; exact-answer generation is unstable for this case.",
        ],
    },
    "4302129014000": {
        "outcome": "ADAPTIVE_ONLY_CORRECT",
        "cause": "ANSWER_SET_PRECISION",
        "secondary_cause": "OTHER",
        "win_class": "ANSWER_SET_PRECISION_WIN",
        "regression_class": None,
        "reusable": False,
        "architecture_issue": False,
        "rationale": (
            "The historical Adaptive BRANCH_FINAL run used two grounded roots: cows and "
            "Époisses de Bourgogne. The cow-milk branch produced Langres; the texture branch "
            "produced five cheese candidates, and ID intersection retained Langres. This is "
            "a genuine candidate-intersection mechanism. However, the repeated runs changed "
            "routing and branch/fallback behavior and did not reproduce the historical win, "
            "so the mechanism is promising but not yet a stable reusable result."
        ),
        "questions": [
            "Did Adaptive choose a different structure? Yes: BRANCH_FINAL versus linear CoR.",
            "Did it choose a better relation? It exposed the texture relation as a second constraint; the key gain was intersection, not a universally better single relation.",
            "Did branching directly cause the historical win? Yes; ID intersection narrowed the candidates to Langres.",
            "Generalization judgment: plausible reusable branching pattern, but not established because the repeated outcomes were unstable.",
        ],
    },
    "3205391005000": {
        "outcome": "COR_ONLY_CORRECT",
        "cause": "RELATION_GROUNDING",
        "secondary_cause": "MERGE_FAILURE",
        "win_class": None,
        "regression_class": "RELATION_GROUNDING_REGRESSION",
        "reusable": False,
        "architecture_issue": False,
        "rationale": (
            "Adaptive selected BRANCH_FINAL for two entities, but the second branch grounded "
            "the shared fact as military rank/Commander rather than the shared occupation "
            "Soldier. The useful Soldier and Commander branch results therefore had no valid "
            "intersection, triggering empty-strict-merge fallback. Historical CoR directly "
            "selected occupation and answered Soldier. Repeats also show stochastic branch "
            "and filter behavior, but the trace identifies relation grounding as the primary "
            "failure."
        ),
        "questions": [
            "Was branching selected correctly? The two-entity structure was plausible, but the unconstrained shared-fact target was too vague.",
            "Did one branch choose a wrong relation? Yes: the Nikodemos branch selected service/rank rather than shared occupation.",
            "Did merge/fallback produce the regression? Yes, after the incompatible branch outputs produced no intersection.",
            "Fixability judgment: likely fixable through better constraint specificity and validation; no architecture change is required by this trace.",
        ],
    },
    "4300043017000": {
        "outcome": "COR_ONLY_CORRECT",
        "cause": "STRUCTURAL_ROUTING",
        "secondary_cause": "RELATION_GROUNDING",
        "win_class": None,
        "regression_class": "ROUTING_REGRESSION",
        "reusable": False,
        "architecture_issue": False,
        "rationale": (
            "Adaptive consistently interpreted the requested parent gender as a shared genus "
            "question. It created two taxonomic branches and merged on Salvelinus, while CoR "
            "followed parent classifications to parent_sex and returned Male. This is a "
            "routing/answer-type error with downstream relation grounding consequences. The "
            "wrong routing is repeatable in the bounded check, but it is a prompt/validation "
            "problem rather than evidence that the branch architecture itself is invalid."
        ),
        "questions": [
            "Was branching selected correctly? No; the router assigned the wrong shared answer type (genus instead of parent gender).",
            "Did one branch choose a wrong relation? Yes, taxonomic higher-classification paths replaced parent_sex.",
            "Did merge/fallback produce the regression? The merge was internally successful, but it merged the wrong semantic target.",
            "Fixability judgment: likely fixable with router semantic validation/constraints; no architecture change is required by this trace.",
        ],
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in read_jsonl(path)}


def load_detail(method: str, case_id: str) -> dict[str, Any]:
    return json.loads((DETAILS / method / f"{case_id}.json").read_text(encoding="utf-8"))


def selected_relations(detail: dict[str, Any]) -> list[list[str]]:
    values: list[list[str]] = []
    for item in detail.get("prompt_history", []):
        if item.get("type") != "relation_prune":
            continue
        parsed = item.get("parsed_result")
        if isinstance(parsed, list):
            values.append([str(value) for value in parsed])
    return values


def sparql_summary(detail: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in detail.get("sparql_history", []):
        query = str(item.get("sparql", "")).replace("\n", " ")
        result.append({
            "search_type": item.get("search_type", ""),
            "result_count": item.get("result_count", -1),
            "status": item.get("status", ""),
            "timed_out": item.get("timed_out", False),
            "query": query,
        })
    return result


def step_summary(detail: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in detail.get("step_history", []):
        if item.get("step_type") == "sparql":
            result.append({
                "type": "sparql",
                "search_type": item.get("search_type", ""),
                "result_count": item.get("result_count", -1),
                "status": item.get("status", ""),
                "filter_caused_empty": item.get("filter_caused_empty", False),
            })
        elif item.get("step_type") == "llm":
            result.append({
                "type": "llm",
                "operation_type": item.get("operation_type", ""),
                "candidate_size": item.get("candidate_size", -1),
                "filtered_candidate_size": item.get("filtered_candidate_size", -1),
                "answers": item.get("answers", []),
                "parsed_result": item.get("parsed_result", ""),
            })
    return result


def histories_equal(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    return [str(x.get("sparql", "")) for x in left] == [str(x.get("sparql", "")) for x in right]


def observations(cor_rows: dict[str, dict[str, Any]], adaptive_rows: dict[str, dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for method, rows in (("cor", cor_rows), ("adaptive", adaptive_rows)):
        for case_id in CASE_IDS:
            row = rows[case_id]
            result[(method, case_id)] = [{"source": "historical", **row}]
    stability_path = STABILITY / "results.jsonl"
    if stability_path.exists():
        for row in read_jsonl(stability_path):
            result[(str(row["method"]), str(row["id"]))].append({
                "source": f"repeat_{row['repeat']}",
                **row,
            })
    return result


def prediction_text(row: dict[str, Any]) -> str:
    values = row.get("prediction", [])
    return "; ".join(str(value) for value in values) if values else "∅"


def stability_line(items: list[dict[str, Any]]) -> str:
    predictions = [tuple(str(value) for value in item.get("prediction", [])) for item in items]
    hits = [int(item.get("hits_at_1", 0)) for item in items]
    f1s = [float(item.get("f1", 0.0)) for item in items]
    exact = len(set(predictions)) == 1
    hit_stable = len(set(hits)) == 1
    return (
        f"predictions={['; '.join(p) if p else '∅' for p in predictions]}; "
        f"Hits@1={hits}; F1={[round(value, 4) for value in f1s]}; "
        f"exact_prediction_stable={'yes' if exact else 'no'}; "
        f"hit_stable={'yes' if hit_stable else 'no'}"
    )


def case_json(case_id: str, cor_rows: dict[str, dict[str, Any]], adaptive_rows: dict[str, dict[str, Any]], obs: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
    meta = CASE_META[case_id]
    cor = cor_rows[case_id]
    adaptive = adaptive_rows[case_id]
    cor_detail = load_detail("cor", case_id)
    adaptive_detail = load_detail("adaptive", case_id)
    return {
        "id": case_id,
        "outcome": meta["outcome"],
        "question": cor.get("question", ""),
        "gold_answers": cor.get("gold_answers", []),
        "primary_cause": meta["cause"],
        "secondary_cause": meta["secondary_cause"],
        "win_classification": meta["win_class"],
        "regression_classification": meta["regression_class"],
        "reusable_mechanism": meta["reusable"],
        "architecture_issue": meta["architecture_issue"],
        "rationale": meta["rationale"],
        "cor": {
            "prediction": cor.get("prediction", []),
            "hits_at_1": cor.get("hits_at_1", 0),
            "f1": cor.get("f1", 0.0),
            "relation_chain": cor.get("relation_chain", []),
            "selected_relations": selected_relations(cor_detail),
            "sparql": sparql_summary(cor_detail),
            "steps": step_summary(cor_detail),
        },
        "adaptive": {
            "prediction": adaptive.get("prediction", []),
            "native_prediction": adaptive.get("native_prediction", []),
            "hits_at_1": adaptive.get("hits_at_1", 0),
            "f1": adaptive.get("f1", 0.0),
            "native_hits_at_1": adaptive.get("native_hits_at_1", 0),
            "native_f1": adaptive.get("native_f1", 0.0),
            "router_structure": adaptive.get("router_structure", ""),
            "router_execution_mode": adaptive.get("router_execution_mode", ""),
            "router_confidence": adaptive.get("router_confidence", 0.0),
            "router_reason": adaptive.get("router_reason", ""),
            "relation_chain": adaptive.get("relation_chain", []),
            "fallback_used": adaptive.get("fallback_used", False),
            "fallback_reason": adaptive.get("fallback_reason"),
            "branch_decomposition": adaptive.get("branch_decomposition", {}),
            "branch_traces": adaptive.get("branch_traces", []),
            "branch_candidate_counts": adaptive.get("branch_candidate_counts", []),
            "merge_method": adaptive.get("merge_method", ""),
            "merge_outcome": adaptive.get("merge_outcome", ""),
            "merge_inputs": adaptive.get("merge_inputs", []),
            "selected_relations": selected_relations(adaptive_detail),
            "sparql": sparql_summary(adaptive_detail),
            "steps": step_summary(adaptive_detail),
        },
        "stability": {
            "cor": obs[("cor", case_id)],
            "adaptive": obs[("adaptive", case_id)],
        },
    }


def md_case(case: dict[str, Any]) -> str:
    meta = CASE_META[case["id"]]
    cor = case["cor"]
    adaptive = case["adaptive"]
    lines = [
        f"## {case['id']} — {case['outcome']}",
        "",
        f"Question: `{case['question']}`",
        f"Gold answer(s): {', '.join(case['gold_answers'])}",
        "",
        f"Primary cause: **{case['primary_cause']}**; secondary: **{case['secondary_cause']}**.",
        f"Classification: `{case.get('win_classification') or case.get('regression_classification')}`.",
        "",
        meta["rationale"],
        "",
        "### CoR trace",
        "",
        f"- Prediction: `{prediction_text(cor)}`; Hits@1={cor['hits_at_1']}; F1={cor['f1']:.4f}.",
        f"- Relation chain: `{cor['relation_chain']}`.",
        f"- Selected relation responses: `{cor['selected_relations']}`.",
        f"- SPARQL result counts: `{[item.get('result_count') for item in cor['sparql']]}`.",
        "",
        "### Adaptive trace",
        "",
        f"- Historical prediction: `{prediction_text(adaptive)}`; Hits@1={adaptive['hits_at_1']}; F1={adaptive['f1']:.4f}.",
        f"- Native prediction: `{'; '.join(adaptive['native_prediction']) if adaptive['native_prediction'] else '∅'}`; native Hits@1={adaptive['native_hits_at_1']}; native F1={adaptive['native_f1']:.4f}.",
        f"- Router: `{adaptive['router_structure']}` / `{adaptive['router_execution_mode']}`, confidence {adaptive['router_confidence']}; fallback={adaptive['fallback_used']} ({adaptive['fallback_reason'] or 'none'}).",
        f"- Relation chain: `{adaptive['relation_chain']}`.",
        f"- Branch candidate counts: `{adaptive['branch_candidate_counts']}`; merge method=`{adaptive['merge_method']}`; merge outcome=`{adaptive['merge_outcome']}`.",
        f"- Selected relation responses: `{adaptive['selected_relations']}`.",
        f"- SPARQL result counts: `{[item.get('result_count') for item in adaptive['sparql']]}`.",
        "",
        "### Interpretation",
        "",
    ]
    lines.extend(f"- {item}" for item in meta["questions"])
    lines += [
        "",
        "### Stability observations",
        "",
        f"- CoR: {stability_line(case['stability']['cor'])}.",
        f"- Adaptive: {stability_line(case['stability']['adaptive'])}.",
        "",
    ]
    return "\n".join(lines)


def build_equivalence(cases: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], str]:
    checks: list[dict[str, Any]] = []
    for case_id in ("3201985002000", "3202928006000"):
        case = cases[case_id]
        cor = case["cor"]
        adaptive = case["adaptive"]
        cor_sparql = [item["query"] for item in cor["sparql"]]
        adaptive_sparql = [item["query"] for item in adaptive["sparql"] if item.get("search_type") != "adaptive_router"]
        checks.append({
            "id": case_id,
            "same_cor_executor": True,
            "same_non_router_sparql_history": cor_sparql == adaptive_sparql,
            "cor_sparql_count": len(cor_sparql),
            "adaptive_non_router_sparql_count": len(adaptive_sparql),
            "cor_relation_responses": cor["selected_relations"],
            "adaptive_relation_responses": adaptive["selected_relations"],
        })
    data = {
        "LINEAR_EQUIVALENT_TO_COR": True,
        "BITWISE_OR_OBSERVATIONAL_EQUIVALENCE": False,
        "checks": checks,
        "differences": [
            "Adaptive makes an additional router LLM call before LINEAR execution.",
            "The router call is not passed as context to CoR, but it changes the stochastic API call sequence.",
            "The LLM API sends no seed; relation-prune, reasoning, and filter outputs can vary even with identical settings.",
            "Both methods use the same CoRAgent code, model, depth=3, relation_width=3, relation threshold=1000, temperatures (0.3/0.1), max tokens=1024, backend, retry behavior, and fresh downstream histories.",
        ],
    }
    md = """# LINEAR equivalence check

## Conclusion

`LINEAR_EQUIVALENT_TO_COR = true` at the implementation/configuration level.

The Adaptive LINEAR path calls the same `CoRAgent` implementation with the same
runtime settings and does not pass router context into CoR. It is therefore the
same intended reasoning method.

`BITWISE_OR_OBSERVATIONAL_EQUIVALENCE = false` under the current experiment
runtime. Adaptive adds one router API call before downstream CoR, the LLM API
does not provide a seed, and the model is stochastic. Consequently, a LINEAR
Adaptive result is not guaranteed to be the same output as a separately run
unchanged-CoR baseline.

## Compared settings

| Item | Finding |
|---|---|
| CoR implementation | Same `CoRAgent` code |
| Model/backend | Same configured model and Freebase backend |
| Depth/width/threshold | Same: 3 / 3 / 1000 |
| Temperatures/max tokens | Same CoR values: 0.3 exploration, 0.1 reasoning, 1024 max tokens |
| Prompt/history | Same downstream CoR prompt templates and fresh histories; Adaptive adds a router prompt before them |
| Router context | Not passed into downstream CoR |
| Seed | None available |
| Retry behavior | Same LLMAPI retry behavior |
| SPARQL | Case 3202928006000 has identical non-router SPARQL history; 3201985002000 has the same candidate universe but different relation ordering/direction |
| Final answer generation | Same filter mechanism, but model output varies |

## Case checks

"""
    for item in checks:
        md += (
            f"- `{item['id']}`: same CoR executor={item['same_cor_executor']}; "
            f"same non-router SPARQL history={item['same_non_router_sparql_history']}.\n"
        )
    md += """
## Experimental-validity consequence

The historical LINEAR wins must not be interpreted as evidence that the router
itself improved the reasoning path. The comparison is valid as an end-to-end
Adaptive-vs-CoR comparison, but causal attribution requires either controlled
repeated runs or deterministic/seeding support. No implementation change is
made in this analysis.
"""
    return data, md


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cor_rows = load_by_id(GRAIL / "cor_results.jsonl")
    adaptive_rows = load_by_id(GRAIL / "adaptive_results.jsonl")
    obs = observations(cor_rows, adaptive_rows)
    cases = OrderedDict((case_id, case_json(case_id, cor_rows, adaptive_rows, obs)) for case_id in CASE_IDS)

    (OUT / "disagreement_5.json").write_text(json.dumps({"cases": list(cases.values())}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_lines = [
        "# Five-disagreement summary",
        "",
        "This report uses the frozen fresh GrailQA-25 paired run and the existing detailed traces. It does not modify routing or merge logic.",
        "",
        "| ID | Outcome | Structure | Fallback | Primary cause | Reusable? | Architecture issue? |",
        "|---|---|---|---|---|---|---|",
    ]
    for case_id, case in cases.items():
        adaptive = case["adaptive"]
        summary_lines.append(
            f"| {case_id} | {case['outcome']} | {adaptive['router_structure']} | "
            f"{'yes: ' + str(adaptive['fallback_reason']) if adaptive['fallback_used'] else 'no'} | "
            f"{case['primary_cause']} | {'yes (mechanism)' if case['reusable_mechanism'] else 'no / unstable'} | "
            f"{'yes' if case['architecture_issue'] else 'no'} |"
        )
    summary_lines += [
        "",
        "## Aggregate interpretation",
        "",
        "- Adaptive-only wins: 0/3 are reproducibly reusable under the three-observation check. One case (4302129014000) exposes a plausible reusable branch-intersection mechanism, but its outcome was unstable.",
        "- CoR-only regressions: 2/2 appear fixable without changing the architecture; one is relation grounding plus empty merge, and one is a repeatable wrong answer-type routing decision.",
        "- Stochasticity: 8/10 method/case series changed exact predictions across the three observations; several also changed Hits@1. This prevents a clean full-100 causal interpretation without tighter experimental controls.",
        "",
        "## Decision",
        "",
        "**RUN_SMALL_STABILITY_CHECK_FIRST** is the gate recommendation. The bounded stability check has now been completed, but it shows that the five disagreements are not sufficiently reproducible to justify the full 100-question evaluation yet. Do not run full 100 in this task.",
        "",
    ]
    (OUT / "disagreement_5_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    (OUT / "adaptive_only_3.md").write_text(
        "# Adaptive-only wins\n\n" + "\n".join(md_case(cases[item]) for item in CASE_IDS[:3]),
        encoding="utf-8",
    )
    (OUT / "cor_only_2.md").write_text(
        "# CoR-only regressions\n\n" + "\n".join(md_case(cases[item]) for item in CASE_IDS[3:]),
        encoding="utf-8",
    )

    equivalence_data, equivalence_md = build_equivalence(cases)
    (OUT / "linear_equivalence_check.md").write_text(equivalence_md, encoding="utf-8")

    stability_lines = [
        "# Stability check",
        "",
        "The original GrailQA-25 observation plus two new repeats were used for each method/case. No output was selected or cherry-picked; all observations are reported.",
        "",
        "| ID | Method | Historical | Repeat 1 | Repeat 2 | Exact prediction stable? | Hits@1 stable? |",
        "|---|---|---|---|---|---|---|",
    ]
    for case_id in CASE_IDS:
        for method in ("cor", "adaptive"):
            items = obs[(method, case_id)]
            signatures = [tuple(str(v) for v in item.get("prediction", [])) for item in items]
            hits = [int(item.get("hits_at_1", 0)) for item in items]
            fmt = ["; ".join(sig) if sig else "∅" for sig in signatures]
            stability_lines.append(
                f"| {case_id} | {method} | {fmt[0]} | {fmt[1]} | {fmt[2]} | "
                f"{'yes' if len(set(signatures)) == 1 else 'no'} | {'yes' if len(set(hits)) == 1 else 'no'} |"
            )
    stability_lines += [
        "",
        "## Findings",
        "",
        "- CoR was exact-prediction stable only for the two consistently wrong Adaptive-only cases; CoR changed outputs on both regression cases and the cheese case.",
        "- Adaptive was exact-prediction stable only for the Salvelinus routing regression, which means that error is reproducible. The other four Adaptive series changed predictions, and three changed Hits@1.",
        "- The historical Adaptive-only outcomes for 3201985002000 and 3202928006000 did not persist across repeats. 4302129014000 also changed routing/output across repeats despite its plausible historical branch-intersection win.",
        "- The check therefore supports a stability/experimental-control gate before any full-100 claim. It does not justify selecting the best repeat or changing the algorithm in this task.",
        "",
    ]
    (OUT / "stability_check.md").write_text("\n".join(stability_lines), encoding="utf-8")

    final_lines = [
        "# Adaptive v1.1.2 disagreement analysis report",
        "",
        "## 1. Fresh GrailQA-25 recap",
        "",
        "The frozen fresh sample produced: unchanged CoR Hits@1=68%, F1=0.4894; Adaptive native Hits@1=68%, F1=0.5257; Adaptive final Hits@1=72%, F1=0.5330. Paired outcomes were 15 BOTH_CORRECT, 3 ADAPTIVE_ONLY_CORRECT, 2 COR_ONLY_CORRECT, and 5 BOTH_WRONG.",
        "",
        "The five disagreement cases were analyzed from their existing traces first. Because the traces showed likely stochastic answer variation, the bounded stability check added two repeats per method/case, for 20 new runs and three total observations per method/case.",
        "",
        "## 2. The five cases",
        "",
    ]
    final_lines.extend(md_case(cases[item]) for item in CASE_IDS)
    final_lines += [
        "## 3. LINEAR-vs-CoR equivalence",
        "",
        "`LINEAR_EQUIVALENT_TO_COR = true` at code/configuration level, but observational equivalence is false because Adaptive makes an extra stochastic router call and no seed is available. The LINEAR wins therefore cannot be causally attributed to routing alone.",
        "",
        "See `linear_equivalence_check.md` for the prompt, candidate, settings, history, retry, and SPARQL comparison.",
        "",
        "## 4. Stability findings",
        "",
        "The bounded check found exact-prediction instability in 8/10 method/case series. Adaptive's two historical LINEAR wins were not stable: the Mali case returned only Cheick Oumar Sissoko in both repeats, and the election case alternated between the abbreviated and full election label. The historical cheese branch win also did not reproduce in the two repeats. The Salvelinus routing regression was stable and therefore is a concrete router semantic failure to fix before claiming broad benefit.",
        "",
        "## 5. Experimental-validity assessment",
        "",
        "The branch mechanism is technically meaningful in 4302129014000: independent constraints and ID intersection can improve candidate precision. However, the observed Adaptive-only aggregate signal is currently confounded by model-output variation, especially because the LINEAR path is not observationally identical to an independently run CoR call. The regressions do not require an architectural rewrite, but they do require better routing/constraint validation and controlled evaluation conditions.",
        "",
        "## 6. Recommendation",
        "",
        "**RUN_SMALL_STABILITY_CHECK_FIRST**. That check has been completed here and shows that the current five-case disagreements are too unstable for a defensible full-100 causal comparison. The next research step should be to improve experimental control and address the identified routing/grounding errors, then rerun a pre-registered small check. Do not run the full 100-question evaluation yet.",
        "",
        "## 7. Output files",
        "",
        "- `disagreement_5_summary.md` — required table and aggregate decision.",
        "- `disagreement_5.json` — machine-readable evidence and classifications.",
        "- `adaptive_only_3.md` — the three Adaptive-only cases.",
        "- `cor_only_2.md` — the two CoR-only cases.",
        "- `linear_equivalence_check.md` — mandatory equivalence analysis.",
        "- `stability_check.md` — bounded three-observation stability results.",
        "",
        "The full 100-question Phase 1 evaluation should not proceed because the five-case disagreements are not sufficiently reproducible, and the current LINEAR comparison has a stochastic call-sequence confound that prevents clean causal attribution.",
        "",
    ]
    (EXP / "DISAGREEMENT_ANALYSIS_REPORT.md").write_text("\n".join(final_lines), encoding="utf-8")

    print(json.dumps({
        "cases": len(cases),
        "output_dir": str(OUT),
        "report": str(EXP / "DISAGREEMENT_ANALYSIS_REPORT.md"),
        "linear_equivalent_to_cor": equivalence_data["LINEAR_EQUIVALENT_TO_COR"],
        "stability_rows": sum(len(value) for value in obs.values()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
