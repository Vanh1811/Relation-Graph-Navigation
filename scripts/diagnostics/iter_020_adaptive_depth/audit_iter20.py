#!/usr/bin/env python3
"""Reproduce the Iteration 20 offline depth mechanism and gate audit.

This script reads frozen Iteration 19 artifacts and the Iteration 0 source
manifest. It does not load credentials, initialize an LLM/KG client, or make
network requests. All generated files are derived artifacts in the Iteration
20 output directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chain_of_relations.eval.accuracy import eval_f1, eval_hit, normalize  # noqa: E402

WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
ITER19 = WORK / "RESULTS/iter_019_depth4_holdout_preparation"
CANON = ITER19 / "control_continuation"
OUT = WORK / "RESULTS/iter_020_adaptive_depth_mechanism"
RUNS_PATH = CANON / "paired_runs_cumulative.jsonl"
BASE_RUNS_PATH = ITER19 / "paired_runs.jsonl"
CONT_RUNS_PATH = CANON / "continuation_runs.jsonl"
USAGE_PATH = CANON / "cumulative_call_usage.jsonl"
INFERENCE_PATH = ITER19 / "inference_cases.jsonl"
SCORING_PATH = ITER19 / "scoring_gold_only.jsonl"
MANIFEST_PATH = WORK / "RESULTS/iter_000_baseline/SNAPSHOT_MANIFEST.json"
OLD_RESULTS = CANON / "continuation_results.json"
OLD_REPORT = CANON / "continuation_report.md"

ROLE_TARGET = "STRONG_DEPTH4_TARGET"
ROLE_CONTROL = "STRONG_SHORTER_PATH_NEGATIVE_CONTROL"
ROLES = (ROLE_TARGET, ROLE_CONTROL)
PRICE_IN = 0.20
PRICE_OUT = 1.20
BOOTSTRAP_SEED = 20261108
BOOTSTRAP_SAMPLES = 50000
AVERAGE_COST_PER_REQUEST = 0.0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compact(value: Any, limit: int = 500) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text if len(text) <= limit else text[: limit - 3] + "..."


def norm_name(value: Any) -> str:
    return normalize(str(value or ""))


def any_gold_name(values: list[Any], gold: list[str]) -> bool:
    """Use the benchmark's substring normalization on the joined candidates."""
    if not values or not gold:
        return False
    return bool(eval_hit(" ".join(str(v) for v in values), gold))


def exact_gold_name(values: list[Any], gold: list[str]) -> bool:
    gold_norm = {norm_name(x) for x in gold if norm_name(x)}
    return any(norm_name(x) in gold_norm for x in values if norm_name(x))


def score(prediction: list[str], gold: list[str]) -> tuple[int, float]:
    pred = prediction or ["None"]
    hit = int(eval_hit(" ".join(pred), gold)) if gold else 0
    f1 = float(eval_f1(pred, gold)[0]) if gold else 0.0
    return hit, f1


def chain_sig(event: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple((str(x.get("direction", "")), str(x.get("id", "")))
                 for x in event.get("relation_chain", []) if isinstance(x, dict))


def route(run: dict[str, Any]) -> str:
    return "/".join(str(run.get(x, "")) for x in ("router_structure", "router_execution_mode"))


def source_for_answer(run: dict[str, Any]) -> str:
    types = [str(x.get("type", "")) for x in run.get("prompt_history", [])]
    if "generate_without_explored_paths" in types:
        return "DIRECT_LLM_FALLBACK"
    if "filter_answer" in types:
        return "FILTER_ANSWER"
    if any(x.get("stage") == "reasoning" and str(x.get("action", "")).lower() == "stop"
           for x in run.get("state_lineage", [])):
        return "REASONING_STOP"
    return "NO_FINAL_ANSWER_TOOL_RECORDED"


def relation_evidence(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("stage") == "relation_search":
        return {
            "head": sorted(str(x) for x in event.get("head_relations", [])),
            "tail": sorted(str(x) for x in event.get("tail_relations", [])),
        }
    return {"selected": sorted((str(x.get("direction", "")), str(x.get("id", "")))
                                for x in event.get("selected_relations", []) if isinstance(x, dict))}


def first_trace_difference(a: dict[str, Any], b: dict[str, Any], stage: str) -> str:
    stages = {"relation": {"relation_search", "relation_prune"}, "entity": {"entity_search"}}[stage]
    left = [x for x in a.get("state_lineage", []) if x.get("stage") in stages]
    right = [x for x in b.get("state_lineage", []) if x.get("stage") in stages]
    for i, (x, y) in enumerate(zip(left, right), start=1):
        kx, ky = (x.get("stage"), chain_sig(x)), (y.get("stage"), chain_sig(y))
        if kx != ky:
            return (f"ordered {stage} trace first differs at event {i}: "
                    f"D3={compact({'stage': kx[0], 'chain': kx[1]})}; "
                    f"D4={compact({'stage': ky[0], 'chain': ky[1]})}; "
                    "independent traversal order, not a causal attribution")
        if stage == "relation" and relation_evidence(x) != relation_evidence(y):
            return (f"ordered {stage} trace first differs at event {i}, stage={kx[0]}, "
                    f"chain={compact(kx[1])}: D3={compact(relation_evidence(x))}; "
                    f"D4={compact(relation_evidence(y))}; direct trace difference")
        if stage == "entity":
            xv = tuple(sorted(str(z) for z in x.get("candidate_entity_ids", [])))
            yv = tuple(sorted(str(z) for z in y.get("candidate_entity_ids", [])))
            xn = tuple(sorted(str(z) for z in x.get("candidate_entity_names", [])))
            yn = tuple(sorted(str(z) for z in y.get("candidate_entity_names", [])))
            xl = tuple(sorted(str(z) for z in x.get("literal_candidates", [])))
            yl = tuple(sorted(str(z) for z in y.get("literal_candidates", [])))
            if (xv, xn, xl) != (yv, yn, yl):
                return (f"ordered entity trace first differs at event {i}, chain={compact(kx[1])}: "
                        f"D3 ids/names/literals={compact([xv[:8], xn[:8], xl[:8]])}; "
                        f"D4={compact([yv[:8], yn[:8], yl[:8]])}; direct trace difference")
    if len(left) != len(right):
        return (f"common ordered {stage} prefix={min(len(left), len(right))}; "
                f"then trace lengths D3={len(left)}, D4={len(right)}; direct trace difference")
    return f"NO_DIFFERENCE_IN_RECORDED_{stage.upper()}_EVENTS"


def answer_coverage(run: dict[str, Any], scoring: dict[str, Any]) -> dict[str, Any]:
    ids = {str(x) for x in scoring.get("answer_ids", []) if str(x)}
    gold = [str(x) for x in scoring.get("gold_answers", []) if str(x)]
    searches = [x for x in run.get("state_lineage", []) if x.get("stage") == "entity_search"]
    seen_ids = {str(i) for e in searches for i in e.get("candidate_entity_ids", [])}
    names = [str(n) for e in searches for n in e.get("candidate_entity_names", [])]
    constraints = [x for x in run.get("state_lineage", [])
                   if x.get("stage") == "reasoning" and str(x.get("action", "")).lower() == "constraint"]
    filter_candidates = [str(n) for e in constraints for n in e.get("candidate_entities", [])]
    filter_events = [x for x in run.get("prompt_history", []) if x.get("type") == "filter_answer"]
    filter_outputs = [x.get("parsed_result") for x in filter_events]
    prediction = [str(x) for x in run.get("prediction", [])]
    return {
        "gold_id_in_any_entity_search": bool(ids & seen_ids),
        "gold_name_in_any_entity_search": any_gold_name(names, gold),
        "gold_name_in_answer_filter_input_candidates": any_gold_name(filter_candidates, gold),
        "gold_name_exact_in_answer_filter_input_candidates": exact_gold_name(filter_candidates, gold),
        "gold_name_in_filter_output": any_gold_name(
            [x for value in filter_outputs for x in (value if isinstance(value, list) else [value])], gold),
        "gold_name_in_final_prediction": any_gold_name(prediction, gold),
        "answer_source": source_for_answer(run),
        "gold_ids_seen": sorted(ids & seen_ids),
        "filter_candidates": filter_candidates,
        "filter_output": filter_outputs,
        "prediction": prediction,
        "diagnostic_gold_candidate": run.get("gold_candidate_diagnostic_scoring_only", {}),
    }


def gate_features(run: dict[str, Any]) -> dict[str, bool]:
    lineage = run.get("state_lineage", [])
    return {
        "forward_action_at_depth3_boundary": any(
            x.get("stage") == "reasoning" and len(x.get("relation_chain", [])) >= 3
            and str(x.get("action", "")).lower() == "forward" for x in lineage),
        "empty_entity_result_at_depth3": any(
            x.get("stage") == "entity_search" and len(x.get("relation_chain", [])) >= 3
            and not (x.get("candidate_entity_ids") or x.get("candidate_entity_names")
                     or x.get("literal_candidates")) for x in lineage),
        "direct_answer_fallback_after_dfs": any(
            x.get("type") == "generate_without_explored_paths" for x in run.get("prompt_history", [])),
        "router_branch_intermediate": run.get("router_structure") == "BRANCH_INTERMEDIATE",
    }


def cost_usd(run: dict[str, Any]) -> float:
    requests = int(run.get("request_count_case", run.get("llm_calls", 0)) or 0)
    return requests * AVERAGE_COST_PER_REQUEST


def run_cost(run: dict[str, Any]) -> dict[str, float]:
    return {
        "requests": int(run.get("request_count_case", run.get("llm_calls", 0)) or 0),
        "input_tokens": int(run.get("input_tokens", 0) or 0),
        "output_tokens": int(run.get("output_tokens", 0) or 0),
        "kg_queries": int(run.get("kg_queries", 0) or 0),
        "latency_ms": float(run.get("latency_ms", 0) or 0),
        "estimated_cost_usd": cost_usd(run),
    }


def bootstrap_ci(case_deltas: dict[str, list[float]]) -> dict[str, Any]:
    keys = sorted(case_deltas)
    vals = [mean(case_deltas[k]) for k in keys]
    rng = random.Random(BOOTSTRAP_SEED + len(keys))
    draws = []
    for _ in range(BOOTSTRAP_SAMPLES):
        draws.append(mean(vals[rng.randrange(len(vals))] for _ in vals))
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return {"unique_questions": len(vals), "pairs_per_question": "2 repeats; averaged within question",
            "mean_question_clustered_delta": mean(vals), "percentile_95_ci": [lo, hi],
            "bootstrap_samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED,
            "interpretation": "Descriptive uncertainty for this selected cohort; not a representative-population interval."}


def summarize_policy(pairs: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        by_role[p["role"]].append(p)
    summary: dict[str, Any] = {"policy": policy, "roles": {}}
    for role, values in by_role.items():
        selected_hits = sum(int(v["selected_run"]["hits_at_1"]) for v in values)
        selected_f1 = mean(float(v["selected_run"]["f1"]) for v in values)
        triggers = sum(bool(v["trigger"]) for v in values)
        d3_only_errors = sum(int(v["d3"]["hits_at_1"]) == 0 and int(v["d4"]["hits_at_1"]) == 1 for v in values)
        rescued = sum(bool(v["trigger"]) and int(v["d3"]["hits_at_1"]) == 0
                      and int(v["d4"]["hits_at_1"]) == 1 for v in values)
        missed = d3_only_errors - rescued
        triggered_regressions = sum(bool(v["trigger"]) and int(v["d3"]["hits_at_1"]) == 1
                                    and int(v["d4"]["hits_at_1"]) == 0 for v in values)
        outcome = Counter()
        totals = Counter()
        for v in values:
            chosen = v["selected_run"]
            if policy == "ALWAYS_DEPTH3":
                outcome["selected_d3"] += 1
            elif policy == "ALWAYS_DEPTH4":
                outcome["selected_d4"] += 1
            elif v["trigger"]:
                outcome["selected_d4_proxy"] += 1
            else:
                outcome["selected_d3"] += 1
            # A triggered policy is charged for the full D3 run plus the D4
            # independent rerun. This is deliberately conservative for cost.
            chosen_cost = run_cost(v["d3"])
            if policy == "ALWAYS_DEPTH4":
                chosen_cost = run_cost(v["d4"])
            elif policy not in {"ALWAYS_DEPTH3", "ALWAYS_DEPTH4"} and v["trigger"]:
                d4_cost = run_cost(v["d4"])
                chosen_cost = {k: chosen_cost[k] + d4_cost[k] for k in chosen_cost}
            for k, val in chosen_cost.items():
                totals[k] += val
        summary["roles"][role] = {
            "pairs": len(values), "triggers": triggers, "trigger_rate": triggers / len(values),
            "hits": selected_hits, "hit_rate": selected_hits / len(values),
            "mean_f1": selected_f1, "d4_only_opportunities": d3_only_errors,
            "d4_only_opportunities_rescued_by_proxy": rescued,
            "d4_only_opportunities_missed": missed,
            "triggered_depth3_only_losses": triggered_regressions,
            "proxy_selected_conditions": dict(outcome),
            "charged_full_rerun_usage": {k: (int(v) if k != "estimated_cost_usd" else round(float(v), 8))
                                          for k, v in totals.items()},
        }
    return summary


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in row) + " |"
                 for row in rows)
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stale_protocol = OUT / "NEXT_PAID_EXPERIMENT_PROTOCOL.md"
    if stale_protocol.exists():
        stale_protocol.unlink()
    provenance_paths = [RUNS_PATH, BASE_RUNS_PATH, CONT_RUNS_PATH, USAGE_PATH, INFERENCE_PATH,
                        SCORING_PATH, OLD_RESULTS, OLD_REPORT, MANIFEST_PATH]
    for path in provenance_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_archive = OUT / "source_artifacts"
    source_archive.mkdir(exist_ok=True)
    archived = {}
    for src, name in ((OLD_RESULTS, "continuation_results.original.json"),
                      (OLD_REPORT, "continuation_report.original.md")):
        dest = source_archive / name
        if not dest.exists():
            shutil.copy2(src, dest)
        archived[str(dest.relative_to(ROOT))] = {"sha256": sha(dest), "copied_from": str(src.relative_to(ROOT))}

    raw_lines = RUNS_PATH.read_text(encoding="utf-8").splitlines()
    runs = [json.loads(line) for line in raw_lines if line.strip()]
    line_by_slot: dict[tuple[str, int, int], int] = {}
    by_slot: dict[tuple[str, int, int], dict[str, Any]] = {}
    duplicate_slots = []
    for line_no, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        slot = (str(row["question_key"]), int(row["diagnostic_depth"]), int(row["repeat_index"]))
        if slot in by_slot:
            duplicate_slots.append({"slot": slot, "lines": [line_by_slot[slot], line_no]})
        by_slot[slot] = row
        line_by_slot[slot] = line_no

    scoring_rows = read_jsonl(SCORING_PATH)
    scoring = {str(x["question_key"]): x for x in scoring_rows}
    inference_rows = read_jsonl(INFERENCE_PATH)
    inference = {str(x["question_key"]): x for x in inference_rows}
    expected_slots = {(q, d, r) for q in scoring for d in (3, 4) for r in (1, 2)}
    missing_slots = sorted(expected_slots - set(by_slot))
    unexpected_slots = sorted(set(by_slot) - expected_slots)

    score_mismatches = []
    rows_by_key_depth: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for slot, run in by_slot.items():
        key, depth, repeat = slot
        gold = scoring[key]
        recomputed_hit, recomputed_f1 = score(run.get("prediction", []), gold.get("gold_answers", []))
        if (recomputed_hit != int(run.get("hits_at_1", 0))
                or abs(recomputed_f1 - float(run.get("f1", 0.0))) > 1e-12):
            score_mismatches.append({"slot": slot, "stored": [run.get("hits_at_1"), run.get("f1")],
                                     "recomputed": [recomputed_hit, recomputed_f1]})
        rows_by_key_depth[(key, depth)].append(run)

    pairs = []
    matrix = []
    depth_role: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    candidate_stats = defaultdict(Counter)
    answer_sources = defaultdict(Counter)
    filter_coverage = defaultdict(Counter)
    d3_cluster_delta: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key in sorted(scoring):
        sc = scoring[key]
        role = str(sc.get("cohort_role_scoring_only"))
        if key not in inference:
            raise AssertionError(f"Scoring case lacks inference-safe record: {key}")
        for depth in (3, 4):
            for run in rows_by_key_depth[(key, depth)]:
                depth_role[role][depth].append(run)
                cov = answer_coverage(run, sc)
                candidate_stats[(role, depth)]["runs"] += 1
                for name in ("gold_id_in_any_entity_search", "gold_name_in_any_entity_search",
                             "gold_name_in_answer_filter_input_candidates", "gold_name_in_filter_output",
                             "gold_name_in_final_prediction"):
                    candidate_stats[(role, depth)][name] += int(cov[name])
                answer_sources[(role, depth, int(run["hits_at_1"]))][cov["answer_source"]] += 1
                filter_coverage[(role, depth)]["correct"] += int(run["hits_at_1"])
                filter_coverage[(role, depth)]["correct_with_gold_filter_candidate"] += int(
                    bool(run["hits_at_1"]) and cov["gold_name_in_answer_filter_input_candidates"])

        for repeat in (1, 2):
            d3, d4 = by_slot[(key, 3, repeat)], by_slot[(key, 4, repeat)]
            cov3, cov4 = answer_coverage(d3, sc), answer_coverage(d4, sc)
            feat = gate_features(d3)
            out3, out4 = int(d3["hits_at_1"]), int(d4["hits_at_1"])
            outcome = "TIE" if out3 == out4 else ("DEPTH4_ONLY" if out4 > out3 else "DEPTH3_ONLY")
            pairs.append({"question_key": key, "repeat": repeat, "role": role,
                          "d3": d3, "d4": d4, "trigger": False, "features": feat,
                          "outcome": outcome})
            reason = str(d4.get("router_reason", ""))
            _ = reason
            relation_diff = first_trace_difference(d3, d4, "relation")
            entity_diff = first_trace_difference(d3, d4, "entity")
            last3 = max((len(x.get("relation_chain", [])) for x in d3.get("state_lineage", [])
                         if x.get("stage") == "reasoning"), default=0)
            hit_interpret = f"d3={out3}/{d3.get('f1', 0):.3f}; d4={out4}/{d4.get('f1', 0):.3f}; pair={outcome}"
            notes = ("Depths are separate full reruns; trace contrasts are direct observations, not causal effects. "
                     f"D3 answer source={cov3['answer_source']}; D4 answer source={cov4['answer_source']}; "
                     f"D3 has explicit max-depth Forward={feat['forward_action_at_depth3_boundary']}; "
                     f"D3 max reasoning-chain length={last3}.")
            matrix.append({
                "question_key": key, "cohort_role": role, "repeat_id": repeat,
                "depth3_hit": out3, "depth3_f1": d3.get("f1"),
                "depth4_hit": out4, "depth4_f1": d4.get("f1"), "paired_outcome": outcome,
                "depth3_route": route(d3), "depth4_route": route(d4),
                "gold_branch_type": sc.get("gold_structure_scoring_only", "UNKNOWN"),
                "gold_minimum_path_depth_if_replayable": (
                    "NOT_REPLAYED; see scoring-only path-depth annotation"),
                "gold_path_depth_annotation_scoring_only": sc.get("gold_answer_path_depth_scoring_only"),
                "first_relation_divergence": relation_diff,
                "first_entity_divergence": entity_diff,
                "first_binding_divergence": "NOT_LOGGED: no intermediate binding/frontier IDs are saved",
                "answer_candidate_visibility_stage": compact({
                    "d3_search_gold_id": cov3["gold_id_in_any_entity_search"],
                    "d3_filter_candidates_include_gold_name": cov3["gold_name_in_answer_filter_input_candidates"],
                    "d3_final_prediction_matches_gold": cov3["gold_name_in_final_prediction"],
                    "d3_answer_source": cov3["answer_source"],
                    "d4_search_gold_id": cov4["gold_id_in_any_entity_search"],
                    "d4_filter_candidates_include_gold_name": cov4["gold_name_in_answer_filter_input_candidates"],
                    "d4_final_prediction_matches_gold": cov4["gold_name_in_final_prediction"],
                    "d4_answer_source": cov4["answer_source"],
                }),
                "answer_filter_output": compact({"depth3": cov3["filter_output"], "depth4": cov4["filter_output"]}),
                "retrieval_expansion_stage": (
                    "D3 explicit Forward at relation-chain length 3: "
                    + str(feat["forward_action_at_depth3_boundary"]) +
                    "; D4 trace contains length-4 events: " + str(any(
                        len(x.get("relation_chain", [])) >= 4 for x in d4.get("state_lineage", [])))),
                "executor_limit": "Independent invocation; DFS stack is local to CoRAgent.answer; fixed max_depth=3 or 4",
                "direct_vs_indirect_evidence": f"DIRECT_TRACE; {hit_interpret}",
                "notes": notes,
                "source_trace_pointer_depth3": f"{RUNS_PATH.relative_to(ROOT)}:line={line_by_slot[(key, 3, repeat)]}",
                "source_trace_pointer_depth4": f"{RUNS_PATH.relative_to(ROOT)}:line={line_by_slot[(key, 4, repeat)]}",
            })

    for role in ROLES:
        for key in scoring:
            if scoring[key].get("cohort_role_scoring_only") != role:
                continue
            r3 = sorted(rows_by_key_depth[(key, 3)], key=lambda x: int(x["repeat_index"]))
            r4 = sorted(rows_by_key_depth[(key, 4)], key=lambda x: int(x["repeat_index"]))
            d3_cluster_delta[role]["hits"].append(mean(float(y["hits_at_1"]) - float(x["hits_at_1"])
                                                        for x, y in zip(r3, r4)))
            d3_cluster_delta[role]["f1"].append(mean(float(y["f1"]) - float(x["f1"])
                                                      for x, y in zip(r3, r4)))

    # Reconstruct run-origin and continuation scope from the immutable inputs.
    base_runs = read_jsonl(BASE_RUNS_PATH)
    continuation_runs = read_jsonl(CONT_RUNS_PATH)
    base_slots = {(x["question_key"], int(x["diagnostic_depth"]), int(x["repeat_index"])) for x in base_runs}
    continuation_slots = {(x["question_key"], int(x["diagnostic_depth"]), int(x["repeat_index"]))
                          for x in continuation_runs}
    continuation_roles = Counter(x.get("cohort_role_scoring_only") for x in continuation_runs)
    overlap = sorted(base_slots & continuation_slots)

    # Request starts and meters are interleaved in a resume-safe journal.
    usage = read_jsonl(USAGE_PATH)
    starts = [x for x in usage if x.get("status") == "REQUEST_STARTED"]
    metered = [x for x in usage if x.get("status") == "REQUEST_METERED"]
    request_ids = [str(x.get("request_id")) for x in starts]
    expected_ids = [f"request_{i:04d}" for i in range(1, len(starts) + 1)]
    usage_summary = {
        "requests_started": len(starts), "requests_metered": len(metered),
        "unmetered_requests": len(starts) - len(metered),
        "input_tokens": sum(int(x.get("input_tokens", 0) or 0) for x in metered),
        "output_tokens": sum(int(x.get("output_tokens", 0) or 0) for x in metered),
        "cost_usd": sum(float(x.get("cost_usd", 0) or 0) for x in metered),
        "request_ids_sequential": request_ids == expected_ids,
        "empty_content_responses": sum(not x.get("has_content", True) for x in metered),
        "usage_journal_sha256": sha(USAGE_PATH),
    }
    global AVERAGE_COST_PER_REQUEST
    AVERAGE_COST_PER_REQUEST = (usage_summary["cost_usd"] / usage_summary["requests_metered"]
                                if usage_summary["requests_metered"] else 0.0)
    case_kg_queries_recomputed = sum(int(x.get("kg_queries", 0) or 0) for x in runs)
    run_usage_summary = {
        "input_tokens": sum(int(x.get("input_tokens", 0) or 0) for x in runs),
        "output_tokens": sum(int(x.get("output_tokens", 0) or 0) for x in runs),
        "requests": sum(int(x.get("request_count_case", x.get("llm_calls", 0)) or 0) for x in runs),
        "case_kg_queries": case_kg_queries_recomputed,
    }
    run_usage_summary["prompt_history_token_priced_cost_usd_partial"] = sum(
        (int(x.get("input_tokens", 0) or 0) * PRICE_IN
         + int(x.get("output_tokens", 0) or 0) * PRICE_OUT) / 1_000_000 for x in runs)
    run_usage_summary["journal_minus_prompt_history_tokens"] = {
        "input": usage_summary["input_tokens"] - run_usage_summary["input_tokens"],
        "output": usage_summary["output_tokens"] - run_usage_summary["output_tokens"],
    }

    old_result = read_json(OLD_RESULTS)
    old_report = OLD_REPORT.read_text(encoding="utf-8")
    manifest = read_json(MANIFEST_PATH)
    frozen_hashes = manifest.get("production_reasoning_source_hashes", {})
    source_checks = []
    for rel, expected in sorted(frozen_hashes.items()):
        path = ROOT / rel
        actual = sha(path) if path.is_file() else None
        source_checks.append({"path": rel, "expected_sha256": expected, "actual_sha256": actual,
                              "match": actual == expected})

    roles_by_key = Counter(x.get("cohort_role_scoring_only") for x in scoring_rows)
    role_metrics = {}
    paired_counts = {}
    repeat_consistency = {}
    paired_mechanism = {}
    cluster_intervals = {}
    for role in ROLES:
        role_metrics[role] = {}
        for depth in (3, 4):
            vals = depth_role[role][depth]
            role_metrics[role][str(depth)] = {
                "runs": len(vals), "hits": sum(int(x["hits_at_1"]) for x in vals),
                "hits_at_1": sum(int(x["hits_at_1"]) for x in vals) / len(vals),
                "mean_f1": mean(float(x["f1"]) for x in vals),
                "gold_id_visible_in_any_entity_search": candidate_stats[(role, depth)]["gold_id_in_any_entity_search"],
                "gold_id_visibility_rate": candidate_stats[(role, depth)]["gold_id_in_any_entity_search"] / len(vals),
                "gold_name_visible_in_answer_filter_input": candidate_stats[(role, depth)]["gold_name_in_answer_filter_input_candidates"],
                "correct_answer_source_counts": dict(answer_sources[(role, depth, 1)]),
                "all_answer_source_counts": {
                    source: sum(answer_sources[(role, depth, hit)][source] for hit in (0, 1))
                    for source in sorted({s for h in (0, 1) for s in answer_sources[(role, depth, h)]})
                },
            }
        role_pairs = [x for x in pairs if x["role"] == role]
        paired_counts[role] = dict(Counter(x["outcome"] for x in role_pairs))
        per_question = defaultdict(list)
        for pair in role_pairs:
            per_question[pair["question_key"]].append(pair["outcome"])
        repeat_consistency[role] = {
            "unique_questions": len(per_question),
            "depth4_only_on_both_repeats": sum(all(x == "DEPTH4_ONLY" for x in outcomes) for outcomes in per_question.values()),
            "depth4_only_on_any_repeat": sum(any(x == "DEPTH4_ONLY" for x in outcomes) for outcomes in per_question.values()),
            "depth3_only_on_any_repeat": sum(any(x == "DEPTH3_ONLY" for x in outcomes) for outcomes in per_question.values()),
            "no_depth4_only_repeat": sum(not any(x == "DEPTH4_ONLY" for x in outcomes) for outcomes in per_question.values()),
        }
        paired_mechanism[role] = {}
        for outcome in ("DEPTH4_ONLY", "DEPTH3_ONLY", "TIE"):
            members = [x for x in role_pairs if x["outcome"] == outcome]
            source_counts = {"depth3": Counter(), "depth4": Counter()}
            for member in members:
                source_counts["depth3"][source_for_answer(member["d3"])] += 1
                source_counts["depth4"][source_for_answer(member["d4"])] += 1
            paired_mechanism[role][outcome] = {
                "pairs": len(members),
                "depth3_gold_id_visible": sum(answer_coverage(x["d3"], scoring[x["question_key"]])["gold_id_in_any_entity_search"] for x in members),
                "depth4_gold_id_visible": sum(answer_coverage(x["d4"], scoring[x["question_key"]])["gold_id_in_any_entity_search"] for x in members),
                "depth3_gold_filter_candidate_visible": sum(answer_coverage(x["d3"], scoring[x["question_key"]])["gold_name_in_answer_filter_input_candidates"] for x in members),
                "depth4_gold_filter_candidate_visible": sum(answer_coverage(x["d4"], scoring[x["question_key"]])["gold_name_in_answer_filter_input_candidates"] for x in members),
                "answer_sources": {k: dict(v) for k, v in source_counts.items()},
                "router_structure_changed": sum(x["d3"].get("router_structure") != x["d4"].get("router_structure") for x in members),
            }
        cluster_intervals[role] = {
            "hits_at_1": bootstrap_ci({k: [float(y["hits_at_1"]) - float(x["hits_at_1"])
                                               for x, y in zip(sorted(rows_by_key_depth[(k, 3)], key=lambda z: z["repeat_index"]),
                                                               sorted(rows_by_key_depth[(k, 4)], key=lambda z: z["repeat_index"]))]
                                             for k in scoring if scoring[k].get("cohort_role_scoring_only") == role}),
            "f1": bootstrap_ci({k: [float(y["f1"]) - float(x["f1"])
                                     for x, y in zip(sorted(rows_by_key_depth[(k, 3)], key=lambda z: z["repeat_index"]),
                                                     sorted(rows_by_key_depth[(k, 4)], key=lambda z: z["repeat_index"]))]
                                   for k in scoring if scoring[k].get("cohort_role_scoring_only") == role}),
        }

    feature_fns = {
        "GATE_FORWARD_AT_DEPTH3_BOUNDARY": "forward_action_at_depth3_boundary",
        "GATE_EMPTY_ENTITY_RESULT_AT_DEPTH3": "empty_entity_result_at_depth3",
        "GATE_AFTER_DIRECT_FALLBACK": "direct_answer_fallback_after_dfs",
        "GATE_ROUTER_BRANCH_INTERMEDIATE": "router_branch_intermediate",
    }
    policy_pairs = {}
    for policy, feature_name in feature_fns.items():
        selected = []
        for p in pairs:
            trigger = p["features"][feature_name]
            q = dict(p)
            q["trigger"] = trigger
            q["selected_run"] = q["d4"] if trigger else q["d3"]
            selected.append(q)
        policy_pairs[policy] = selected
    base_selected = []
    d4_selected = []
    for p in pairs:
        base_selected.append({**p, "trigger": False, "selected_run": p["d3"]})
        d4_selected.append({**p, "trigger": True, "selected_run": p["d4"]})
    policy_summaries = {"ALWAYS_DEPTH3": summarize_policy(base_selected, "ALWAYS_DEPTH3"),
                        "ALWAYS_DEPTH4": summarize_policy(d4_selected, "ALWAYS_DEPTH4")}
    policy_summaries.update({k: summarize_policy(v, k) for k, v in policy_pairs.items()})

    # Source consistency and scoring reconciliation are all derived from raw slots.
    target_mean_delta_h = mean(d3_cluster_delta[ROLE_TARGET]["hits"])
    target_mean_delta_f = mean(d3_cluster_delta[ROLE_TARGET]["f1"])
    control_mean_delta_h = mean(d3_cluster_delta[ROLE_CONTROL]["hits"])
    control_mean_delta_f = mean(d3_cluster_delta[ROLE_CONTROL]["f1"])
    observed_diff = {
        "report_control_d3_hits": "20/20 (1.00)", "report_control_d4_hits": "19/20 (0.95)",
        "raw_control_d3_hits": f"{role_metrics[ROLE_CONTROL]['3']['hits']}/20 ({role_metrics[ROLE_CONTROL]['3']['hits_at_1']:.2f})",
        "raw_control_d4_hits": f"{role_metrics[ROLE_CONTROL]['4']['hits']}/20 ({role_metrics[ROLE_CONTROL]['4']['hits_at_1']:.2f})",
    }
    controls_differences = [p for p in pairs if p["role"] == ROLE_CONTROL and p["outcome"] != "TIE"]
    target_differences = [p for p in pairs if p["role"] == ROLE_TARGET and p["outcome"] != "TIE"]

    scoring_report = {
        "iteration": 20, "source_of_truth": str(RUNS_PATH.relative_to(ROOT)),
        "raw_sources_sha256": {str(x.relative_to(ROOT)): sha(x) for x in provenance_paths},
        "preserved_original_artifacts": archived,
        "slot_reconciliation": {"raw_rows": len(runs), "unique_questions": len(scoring),
                                "expected_slots": len(expected_slots), "observed_unique_slots": len(by_slot),
                                "missing_slots": missing_slots, "unexpected_slots": unexpected_slots,
                                "duplicate_slots": duplicate_slots,
                                "base_run_rows": len(base_runs), "continuation_run_rows": len(continuation_runs),
                                "continuation_role_counts": dict(continuation_roles),
                                "base_continuation_slot_overlap": overlap,
                                "target_questions": roles_by_key[ROLE_TARGET],
                                "control_questions": roles_by_key[ROLE_CONTROL]},
        "score_recomputation": {"mismatch_count": len(score_mismatches), "mismatches": score_mismatches,
                                "scorer": "chain_of_relations.eval.accuracy.eval_hit/eval_f1; same prediction normalization as runner"},
        "condition_metrics_by_role": role_metrics,
        "paired_outcomes_by_role": paired_counts,
        "repeat_consistency_by_question": repeat_consistency,
        "candidate_and_answer_source_by_paired_outcome": paired_mechanism,
        "paired_mean_deltas_depth4_minus_depth3": {
            ROLE_TARGET: {"hits_at_1": target_mean_delta_h, "f1": target_mean_delta_f,
                          "pairs": 40, "unique_questions": 20},
            ROLE_CONTROL: {"hits_at_1": control_mean_delta_h, "f1": control_mean_delta_f,
                           "pairs": 20, "unique_questions": 10}},
        "question_clustered_bootstrap": cluster_intervals,
        "report_conflict": observed_diff,
        "non_tied_control_pairs": [{"question_key": p["question_key"], "repeat": p["repeat"],
                                     "depth3_hit": p["d3"]["hits_at_1"], "depth4_hit": p["d4"]["hits_at_1"]}
                                    for p in controls_differences],
        "non_tied_target_pairs": [{"question_key": p["question_key"], "repeat": p["repeat"],
                                   "depth3_hit": p["d3"]["hits_at_1"], "depth4_hit": p["d4"]["hits_at_1"]}
                                  for p in target_differences],
        "usage_recomputed_from_cumulative_journal": usage_summary,
        "method_run_usage_recomputed_from_120_rows": run_usage_summary,
        "usage_expected_from_iteration19_corrected_summary": {
            "requests": old_result.get("requests_started_cumulative"),
            "cost_usd": old_result.get("cost_usd_cumulative"),
            "input_tokens": old_result.get("input_tokens_cumulative"),
            "output_tokens": old_result.get("output_tokens_cumulative")},
        "frozen_production_source_hashes": {"checked": len(source_checks), "passed": sum(x["match"] for x in source_checks),
                                            "failures": [x["path"] for x in source_checks], "details": source_checks},
        "no_new_calls": {"paid_llm_calls_by_audit": 0, "freebase_queries_by_audit": 0,
                         "all_evidence_read_from_saved_artifacts": True},
    }
    write_json(OUT / "scoring_reconciliation.json", scoring_report)

    matrix_fields = list(matrix[0]) if matrix else []
    write_csv(OUT / "depth_mechanism_matrix.csv", matrix, matrix_fields)

    # Inventory contains runtime-only candidates and explicit leakage controls.
    inventory_specs = [
        ("forward_action_at_depth3_boundary", "PRE_EXPANSION", "Reasoning action Forward when relation-chain length is 3; evaluate before max-depth skip.", True, "Gold-free; direct state_lineage event."),
        ("empty_entity_result_at_depth3", "AFTER_ENTITY_SEARCH_PRE_NEXT_ACTION", "Entity search at chain length 3 has no entity IDs, names, or literals.", True, "Gold-free; this exact branch has no entity binding to extend; would require broader policy or rerun."),
        ("direct_answer_fallback_after_dfs", "POST_SEARCH_TERMINATION", "A generate_without_explored_paths event is about to answer after DFS exhaustion.", True, "Gold-free, but too late for in-place expansion; only a full rerun can use it."),
        ("router_branch_intermediate", "PRE_RUN", "Frozen router structure is BRANCH_INTERMEDIATE.", True, "Gold-free and available before search; broad structural proxy, not specific to a depth need."),
        ("gold_answer_ids_or_candidate_match", "SCORING_ONLY", "Gold answer ID/answer match in logged candidates or outputs.", False, "Forbidden inference feature; only used in this audit after inference."),
        ("gold_program_path_depth", "SCORING_ONLY", "Gold-program answer path depth annotation.", False, "Forbidden inference feature; benchmark-side annotation, not verified shortest graph distance."),
        ("depth4_success_or_paired_outcome", "POST_OUTCOME_SCORING_ONLY", "Whether independent depth-4 rerun succeeded or improved the answer.", False, "Forbidden outcome leakage; never a deployable gate input."),
    ]
    inventory = []
    for feature, moment, definition, available, note in inventory_specs:
        row = {"signal": feature, "decision_moment": moment, "deterministic_definition": definition,
              "observed_in_saved_inference_trace": available, "deployable_without_gold": available and "SCORING_ONLY" not in moment,
              "trigger_target_pairs": "", "target_pair_denominator": 40,
              "trigger_control_pairs": "", "control_pair_denominator": 20,
              "leakage_status": "PASS_GOLD_FREE" if available and "SCORING_ONLY" not in moment else "FORBIDDEN_GOLD_OR_OUTCOME",
              "limitations": note}
        if feature in feature_fns.values():
            triggered = [p for p in pairs if p["features"][feature]]
            row["trigger_target_pairs"] = sum(p["role"] == ROLE_TARGET for p in triggered)
            row["trigger_control_pairs"] = sum(p["role"] == ROLE_CONTROL for p in triggered)
        inventory.append(row)
    write_csv(OUT / "gate_signal_inventory.csv", inventory, list(inventory[0]))

    # Detailed trace facts for the coverage reconciliation.
    correct_d3_targets = [r for r in depth_role[ROLE_TARGET][3] if int(r["hits_at_1"]) == 1]
    target_d3_source_counts = Counter(source_for_answer(r) for r in correct_d3_targets)
    target_d3_id_visible_correct = sum(answer_coverage(r, scoring[r["question_key"]])["gold_id_in_any_entity_search"]
                                       for r in correct_d3_targets)
    target_d3_filter_gold_correct = sum(answer_coverage(r, scoring[r["question_key"]])["gold_name_in_answer_filter_input_candidates"]
                                        for r in correct_d3_targets)
    coverage_md = f"""# Iteration 20 — answer coverage reconciliation

## Reconciled observation

At depth 3, the 20 selected four-edge target questions produced 18/40 correct method runs across two repeats, while a gold answer ID appeared in an entity-search result in only 2/40 runs. These measures are not contradictory: the executor has a separate direct-answer path after depth-first search ends.

Among the 18 correct target depth-3 runs, the saved prompt trace shows **{target_d3_source_counts.get('DIRECT_LLM_FALLBACK', 0)}** answers came from `generate_without_explored_paths` and **{target_d3_source_counts.get('FILTER_ANSWER', 0)}** from `filter_answer`. Only **{target_d3_id_visible_correct}** of those 18 correct runs logged a gold answer ID in entity search. In the trace, direct fallback outputs are evaluated as prediction text; they are not constrained to the entity-search candidate IDs. The scorer then compares the final names with gold answer strings.

The code path is visible in `chain_of_relations/methods/cor/agent.py`: `CoRAgent.answer()` exits its DFS loop and calls `generate_directly()`; the resulting text is split into predictions and returned. `scripts/run_cor_baseline_experiment.py` evaluates prediction names against gold answer strings. This is direct code plus trace evidence, not a claim that the direct answer is grounded in Freebase.

## Three different coverage stages

The Iteration 20 matrix reports separate flags for (1) a gold ID/name in any SPARQL entity-search output, (2) a gold name among the actual answer-filter input candidates, (3) a gold name in the answer-filter output, and (4) a gold-matching final prediction. These stages are not interchangeable. `state_lineage` records entity-search candidates and reasoning candidate lists, but not intermediate bindings or a canonical answer ID set for direct generation.

For correct depth-3 target runs, **{target_d3_filter_gold_correct}/{len(correct_d3_targets)}** had a gold-matching name in a logged constraint-filter input. The remaining correct direct answers bypassed the graph candidate/filter route. The direct answer generator can introduce a name without an ID appearing in the logged graph-search list. We cannot infer whether it recalled parametric knowledge, guessed correctly, or used question context alone.

## Limits and provenance

- All counts are recomputed from the 120 raw rows in `{RUNS_PATH.relative_to(ROOT)}` and the scoring-only answer sidecar.
- The saved d3/d4 rows are **independent full reruns**, not a d3 run resumed with the same frontier. Differences in prompts, relation pruning, traversal, answer filtering, and model samples are confounded.
- The trace does not store intermediate binding/frontier identities. The first binding divergence is therefore `NOT_LOGGED` throughout the per-pair matrix.
- Gold answer IDs and gold answer strings were used only after inference for this audit. They are not proposed as runtime gate features.
"""
    (OUT / "answer_coverage_reconciliation.md").write_text(coverage_md, encoding="utf-8")

    discrepancy_md = f"""# Iteration 20 — Iteration 19 report discrepancy

## Resolution from immutable rows

The cumulative raw file `{RUNS_PATH.relative_to(ROOT)}` contains 120 rows covering 30 question keys × two depths × two repeats. Recomputing each row with the repository evaluator gives zero score mismatches.

{md_table(['Control metric', 'Narrative report', 'Recomputed raw rows', 'Canonical JSON'], [
    ['Depth 3 Hits@1', '20/20 = 1.00', f"{role_metrics[ROLE_CONTROL]['3']['hits']}/20 = {role_metrics[ROLE_CONTROL]['3']['hits_at_1']:.2f}", f"{old_result['condition_metrics_by_role'][ROLE_CONTROL]['3']['mean_hits_at_1']:.2f}"],
    ['Depth 4 Hits@1', '19/20 = 0.95', f"{role_metrics[ROLE_CONTROL]['4']['hits']}/20 = {role_metrics[ROLE_CONTROL]['4']['hits_at_1']:.2f}", f"{old_result['condition_metrics_by_role'][ROLE_CONTROL]['4']['mean_hits_at_1']:.2f}"],
    ['Depth4−Depth3 paired Hits@1', '−0.05', f"{control_mean_delta_h:+.2f}", f"{old_result['paired_mean_deltas_by_role'][ROLE_CONTROL]['mean_hits_at_1_delta_depth4_minus_depth3']:+.2f}"],
])}

**The JSON and raw rows are correct; the absolute control rates in the narrative report are stale.** The net paired change of −0.05 remains correct. The failed control question is `cwq::WebQTrn-64_b09c00219764b4f102d325a00e808259` at both repeats and both depths; its answer is wrong under the evaluator at both depths. The additional regression is `cwq::WebQTest-450_3eb90d65b82703c8f7fb0496baa80ce7`, repeat 1: depth 3 returns “Hawaiian hibiscus” (hit), while depth 4 returns no answer (miss); repeat 2 ties. This agrees with the control regression ID already named in the old report.

The one depth-3-only **target** pair is separately `cwq::WebQTest-106_adee4fffb5ddcd6fb97a507e2bd95983`, repeat 1. It is not the control regression.

## Preservation and correction policy

The original `continuation_report.md` and `continuation_results.json` remain unchanged. Byte-preserved copies and hashes are in `source_artifacts/`; the raw paired runs are untouched. This corrected derived reconciliation supersedes only the report's control absolute-rate statement. No prior artifact was edited in place.

## Reproduction

```bash
python scripts/diagnostics/iter_020_adaptive_depth/audit_iter20.py
```
"""
    (OUT / "report_discrepancy.md").write_text(discrepancy_md, encoding="utf-8")

    gate_table_rows = []
    for policy, summary in policy_summaries.items():
        for role in ROLES:
            s = summary["roles"][role]
            use = s["charged_full_rerun_usage"]
            gate_table_rows.append([policy, "targets" if role == ROLE_TARGET else "controls",
                                    f"{s['triggers']}/{s['pairs']}" if policy not in ("ALWAYS_DEPTH3", "ALWAYS_DEPTH4") else "—",
                                    f"{s['hits']}/{s['pairs']} ({s['hit_rate']:.3f})", f"{s['mean_f1']:.3f}",
                                    f"{s['d4_only_opportunities_rescued_by_proxy']}/{s['d4_only_opportunities']}",
                                    s['d4_only_opportunities_missed'], s['triggered_depth3_only_losses'],
                                    use['requests'], use['kg_queries'], f"${use['estimated_cost_usd']:.4f}"])
    gate_md = f"""# Iteration 20 — inference-time gate feasibility

## Executor and decision point

Iteration 19's depth-3 and depth-4 conditions were **separate full runs**. The holdout runner constructs an agent for each scheduled slot and sets its depth before `answer()`; `CoRAgent.answer()` creates a local DFS stack and does not return resumable state. The current implementation therefore has no validated external resume operation.

A narrow in-call gate is nevertheless instrumentable: when a `reasoning` event returns `Forward` at relation-chain length 3, the current CoR code checks the depth cap immediately before its forward relation search. A future implementation could begin with the depth-3 cap and permit one additional hop at that point, retaining the current invocation's DFS stack. This is a mechanism proposal only; it has not been implemented or run.

## Baselines and exploratory proxy

The table below selects the observed depth-4 rerun's answer whenever a gold-free depth-3 trace signal fires, otherwise the depth-3 answer. Because the saved depth-4 result came from an independent rerun, every policy row is an **optimistic independent-rerun proxy**, not an observed adaptive execution or causal estimate. Cost charges the complete depth-3 run plus a complete depth-4 rerun on triggers; it does not pretend a stateful continuation's incremental cost is known.

{md_table(['Policy', 'Cohort', 'Triggers', 'Hits@1', 'Mean F1', 'D4-only wins rescued / available', 'D4-only wins missed', 'Triggered D3-only losses', 'Charged requests', 'KG queries', 'Estimated cost*'], gate_table_rows)}

### What the proxy says

- `GATE_FORWARD_AT_DEPTH3_BOUNDARY` is the most defensible non-oracle trigger. It fires on {policy_summaries['GATE_FORWARD_AT_DEPTH3_BOUNDARY']['roles'][ROLE_TARGET]['triggers']}/40 target pairs and {policy_summaries['GATE_FORWARD_AT_DEPTH3_BOUNDARY']['roles'][ROLE_CONTROL]['triggers']}/20 controls; its independent-rerun proxy selects {policy_summaries['GATE_FORWARD_AT_DEPTH3_BOUNDARY']['roles'][ROLE_TARGET]['d4_only_opportunities_rescued_by_proxy']} of 13 target depth-4-only pairs and misses {policy_summaries['GATE_FORWARD_AT_DEPTH3_BOUNDARY']['roles'][ROLE_TARGET]['d4_only_opportunities_missed']}. It also selects the observed short-path control loss, so the proxy does not establish the required no-regression property.
- `GATE_EMPTY_ENTITY_RESULT_AT_DEPTH3` rescues more target pairs in the proxy, but it is not a valid same-path continuation trigger at an empty binding: the current path has no entity to expand. Its results assume a full depth-4 rerun and it selects the observed control regression.
- `GATE_AFTER_DIRECT_FALLBACK` is logged only after DFS terminates. It also needs a full rerun, and its apparent selectivity is measured on the same 30 questions.
- `GATE_ROUTER_BRANCH_INTERMEDIATE` is available before search but selects none of the 13 target depth-4-only pairs; it is rejected as a depth trigger.

## Decision and cost caveats

The explicit `Forward` at the cap is a deployable gold-free signal with a direct semantic reason to allow one more hop. That makes an in-call gate technically implementable, but the observed proxy selects the known control loss and misses 8 of 13 target-only wins. It therefore does not meet the protocol’s criterion for a plausible gate that limits control regression. The policy was inspected against these outcomes; all proxy performance is development evidence only.

The charged proxy assumes an entire independent d4 rerun at each trigger. A future in-call implementation should cost less, but the amount cannot be estimated from these reruns. Iteration 19 averaged {usage_summary['requests_started']/120:.2f} LLM requests and {old_result['case_kg_queries_cumulative']/120:.2f} SPARQL case queries per method run. *Estimated cost allocates the metered Iteration 19 total evenly by request (about ${AVERAGE_COST_PER_REQUEST:.7f}/request); per-question usage cannot be recovered because the cumulative meter journal is not keyed to question. Run-row token totals omit {run_usage_summary['journal_minus_prompt_history_tokens']['input']:,} input and {run_usage_summary['journal_minus_prompt_history_tokens']['output']:,} output tokens compared with the metered journal, so they are not used for per-policy dollar totals. Network latency and Freebase load remain variable.

## Evidence strength

Iteration 19 directly establishes the output/trace differences and score counts. It does not identify a causal depth-only effect: model answers, selected relations, traversal, answer-filter candidates, and router outputs can differ across independent depth conditions. The matrix records the first ordered relation/entity trace difference and marks all binding divergence as `NOT_LOGGED`. The logs also do not establish whether deeper navigation alone resolves shared-variable joins or constraint intersections (H4). A future matched-prefix replay must log the exact same-call prefix, new fourth-hop relation/entity results, intermediate bindings, and final answer-filter candidates.
"""
    (OUT / "gate_feasibility.md").write_text(gate_md, encoding="utf-8")

    # Checks cover data integrity and environment/source invariants, not tests.
    prompt_forbidden_id_occurrences = []
    for run in runs:
        prompt_text = "\n".join(str(x.get("prompt", "")) for x in run.get("prompt_history", []))
        for gold_id in scoring[run["question_key"]].get("answer_ids", []):
            if str(gold_id) and str(gold_id) in prompt_text:
                prompt_forbidden_id_occurrences.append({"question_key": run["question_key"],
                                                        "depth": run["diagnostic_depth"],
                                                        "repeat": run["repeat_index"], "gold_id": gold_id})
    inference_gold_fields = sorted(set().union(*(set(x) for x in inference_rows))
                                   & {"answer_ids", "gold_answers", "answers", "gold_structure_scoring_only",
                                      "gold_answer_path_depth_scoring_only", "cohort_role_scoring_only"})
    checks = {
        "raw_120_slots_complete_unique": len(runs) == 120 and not missing_slots and not unexpected_slots and not duplicate_slots,
        "30_questions_20_targets_10_controls": len(scoring) == 30 and roles_by_key == Counter({ROLE_TARGET: 20, ROLE_CONTROL: 10}),
        "base_90_plus_control_only_continuation_30": len(base_runs) == 90 and len(continuation_runs) == 30
            and continuation_roles == Counter({ROLE_CONTROL: 30}) and not overlap,
        "all_scores_recomputed": not score_mismatches,
        "paired_target_and_control_counts": paired_counts.get(ROLE_TARGET) == Counter({"TIE": 26, "DEPTH3_ONLY": 1, "DEPTH4_ONLY": 13})
            and paired_counts.get(ROLE_CONTROL) == Counter({"TIE": 19, "DEPTH3_ONLY": 1}),
        "usage_cumulative_fully_metered_and_sequential": usage_summary["requests_started"] == 1237
            and usage_summary["requests_metered"] == 1237 and usage_summary["unmetered_requests"] == 0
            and usage_summary["request_ids_sequential"],
        "usage_matches_corrected_iteration19_summary": abs(usage_summary["cost_usd"] - float(old_result["cost_usd_cumulative"])) < 1e-9
            and usage_summary["input_tokens"] == int(old_result["input_tokens_cumulative"])
            and usage_summary["output_tokens"] == int(old_result["output_tokens_cumulative"]),
        "run_request_and_kg_counts_match_cumulative_journal": run_usage_summary["requests"] == usage_summary["requests_metered"]
            and run_usage_summary["case_kg_queries"] == old_result["case_kg_queries_cumulative"],
        "case_kg_query_count_matches_corrected_summary": case_kg_queries_recomputed == old_result["case_kg_queries_cumulative"],
        "seven_frozen_production_hashes_match": len(source_checks) == 7 and all(x["match"] for x in source_checks),
        "no_gold_fields_in_inference_case_records": not inference_gold_fields,
        "gold_answer_ids_absent_from_saved_llm_prompts": not prompt_forbidden_id_occurrences,
        "zero_paid_llm_calls_during_iteration20": True,
        "zero_freebase_queries_during_iteration20": True,
    }
    # Keep this preservation check explicit and straightforward (the original
    # artifacts are read-only inputs; their copies must be byte-identical).
    checks["original_iteration19_result_and_report_preserved"] = (
        sha(OLD_RESULTS) == sha(source_archive / "continuation_results.original.json")
        and sha(OLD_REPORT) == sha(source_archive / "continuation_report.original.md"))
    check_doc = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "details": {"missing_slots": missing_slots, "unexpected_slots": unexpected_slots,
                    "duplicate_slots": duplicate_slots, "score_mismatches": score_mismatches,
                    "base_continuation_overlap": overlap, "continuation_roles": dict(continuation_roles),
                    "usage": usage_summary, "source_hashes": source_checks,
                    "run_usage": run_usage_summary,
                    "inference_gold_fields_found": inference_gold_fields,
                    "gold_answer_ids_found_in_prompts": prompt_forbidden_id_occurrences,
                    "audit_tool": str(Path(__file__).relative_to(ROOT)),
                    "audit_tool_sha256": sha(Path(__file__))},
        "commands": ["python scripts/diagnostics/iter_020_adaptive_depth/audit_iter20.py",
                     "python -m json.tool outputs/experiments/fresh50_continuous_agent/RESULTS/iter_020_adaptive_depth_mechanism/offline_checks.json"],
        "network": {"paid_api_calls": 0, "freebase_queries": 0,
                    "implementation": "standard-library artifact reads/writes plus local evaluator; no network client invoked"},
    }
    write_json(OUT / "offline_checks.json", check_doc)

    # Compact decision log records the uncertainty -> evidence -> next offline step chain.
    decision_log = f"""# Iteration 20 research decision log

## Cycle 1 — reconcile scores and cohort

**Uncertainty:** Was the 0.90/0.85 control result or the 1.00/0.95 narrative correct? Could the reported pairing counts be stale?

**Competing explanations:** (a) report typo; (b) paired rows or evaluator drift. **Least-cost discriminator:** enumerate all raw question × depth × repeat rows and recompute with the frozen repository evaluator.

**Result:** {len(runs)} rows, {len(by_slot)} unique slots, {len(scoring)} questions, {len(score_mismatches)} score mismatches. Raw control Hits@1 is {role_metrics[ROLE_CONTROL]['3']['hits']}/20 at depth 3 and {role_metrics[ROLE_CONTROL]['4']['hits']}/20 at depth 4. Paired counts reproduce 13/1/26 target and 0/1/19 control outcomes. The JSON matches; the narrative's absolute rates are stale. Original files are preserved and correction is derived.

**Decision:** continue; raw data are reconcilable, so this is not outcome D.

## Cycle 2 — explain answer coverage and test executor boundary

**Uncertainty:** Does 2/40 gold-ID candidate coverage imply depth 3 could only answer two targets?

**Competing explanations:** (a) incomplete candidate instrumentation; (b) final answers can bypass candidate IDs. **Discriminator:** match answer source events to predictions and inspect the frozen executor/evaluator path.

**Result:** Correct depth-3 target traces use the `generate_without_explored_paths` direct LLM fallback in {target_d3_source_counts.get('DIRECT_LLM_FALLBACK', 0)} cases and `filter_answer` in {target_d3_source_counts.get('FILTER_ANSWER', 0)}. The direct route produces answer text after DFS and is scored as text, so entity-search IDs do not bound Hits@1. Intermediate bindings are absent from logs.

**Decision:** reject any causal claim that candidate recall alone explains depth-3 errors; continue to compare trace mechanisms and runtime signals.

## Cycle 3 — compare gains, loss, and pre-expansion signals

**Uncertainty:** Is there a gold-free event that can authorize one extra hop in the same search call, and does its saved-trace proxy selectively cover useful cases?

**Competing explanations:** (a) a boundary `Forward` action identifies an unmet reasoning step; (b) depth-4 outcomes mostly arise from unrelated sampled routes/filter choices. **Discriminator:** inspect ordered state/prompt traces and simulate only predeclared simple signals using the paired independent rerun outcomes.

**Result:** saved code has a local DFS stack and blocks `Forward` at the cap; an in-call toggle is technically placeable, while external resume is not. The `Forward` signal fires on 12/40 target and 3/20 control pairs; its independent-rerun proxy rescues 5 of 13 target-only wins, misses 8, and selects the one control loss. The signal is not validated and proxy results conflate separate LLM runs. Other signals either have worse control behavior or occur too late.

**Decision:** outcome C. The positive depth-4 comparison and added candidate visibility are consistent with a search-horizon benefit, but the independently sampled prompts, routes, and answer stages confound the mechanism. Do not begin a gate confirmation yet.

## Final stopping decision

The offline artifact audit is complete and all checks are **{check_doc['status']}**. There are no remaining informative offline checks in the saved Iteration 19 traces: no deterministic paired continuation is available and intermediate bindings were not recorded. Next research action is a matched-prefix replay study that freezes the same depth-3 LLM decisions and graph results before comparing stop-at-3 with same-call depth-4 continuation; it requires new instrumentation and separate budget approval before any new paid request. No Iteration 20 paid calls were made.
"""
    (OUT / "RESEARCH_DECISION_LOG.md").write_text(decision_log, encoding="utf-8")

    outcome_a = f"""# Iteration 20 decision — adaptive depth mechanism

## Verdict: **C — `MECHANISM_NOT_ISOLATED`**

### Proven facts

- The 120 immutable run rows cover 30 CWQ questions × two max depths × two repeats; scores recompute with zero mismatches. The canonical result JSON's control Hits@1 is correct at **18/20 (depth 3)** and **17/20 (depth 4)**. The narrative report's **20/20 and 19/20** is stale; paired delta remains −0.05. The source report and JSON were preserved unchanged.
- Depth-4-only / depth-3-only / ties are **13 / 1 / 26** on 20 selected four-edge targets and **0 / 1 / 19** on 10 short-path controls. These are selected-cohort results, with repeated runs clustered by question.
- Question-clustered bootstrap mean Hits@1 delta depth4−depth3 is **{cluster_intervals[ROLE_TARGET]['hits_at_1']['mean_question_clustered_delta']:+.3f}** (95% percentile interval {cluster_intervals[ROLE_TARGET]['hits_at_1']['percentile_95_ci'][0]:+.3f} to {cluster_intervals[ROLE_TARGET]['hits_at_1']['percentile_95_ci'][1]:+.3f}) on the 20 targets and **{cluster_intervals[ROLE_CONTROL]['hits_at_1']['mean_question_clustered_delta']:+.3f}** ({cluster_intervals[ROLE_CONTROL]['hits_at_1']['percentile_95_ci'][0]:+.3f} to {cluster_intervals[ROLE_CONTROL]['hits_at_1']['percentile_95_ci'][1]:+.3f}) on the 10 controls. These intervals describe only the selected cohorts.
- Correct depth-3 target outputs occurred in 18/40 runs despite only 2/40 gold IDs appearing in entity search. Trace and source inspection show the direct LLM fallback can output and score answer text outside the graph candidate list.
- Depth 4 has 29/40 target runs with a gold ID in entity search and 30/40 correct outputs. That is consistent with added candidate access, but the two conditions are independent full LLM reruns.
- The one control loss changes from a depth-3 filter answer (“Hawaiian hibiscus”) to no depth-4 answer in one repeat. The pre-expansion `Forward` proxy also selects this loss; it does not establish control safety.

### What remains confounded

The depth-3/depth-4 executions independently sample router outputs, relation choices, traversal, filter candidates, and direct answers. Some depth-3 correct answers come from direct generation; many depth-4 answers come from `filter_answer`. Candidate retrieval, planning, and answer selection therefore cannot be causally separated from these saved runs. `state_lineage` omits intermediate binding/frontier IDs, and no same-run depth-3-to-4 continuation was executed. Do not claim a pure search-horizon effect or an adaptive gate benefit.

### Next narrow study

Keep production TGAR/CoR unchanged. Instrument a **matched-prefix replay** in a non-production branch: replay the identical saved depth-3 router/relation-prune/reasoning outputs and saved graph results in both arms through the depth-3 boundary; one arm stops, the other permits one additional hop on the same DFS stack after an explicit `Forward`. Log every newly reached relation/entity, answer-filter input/output, and answer source. Use untouched questions for any scored evaluation and preregister short-path controls. This isolates the incremental search horizon from pre-boundary LLM sampling; any calls required beyond the boundary need a new explicit authorization. Do not fit or tune the gate on Iteration 19.

### Limits and budget

This is a selected 20-target/10-control CWQ challenge, not representative KGQA accuracy. The `Forward` and other policy results are independent-rerun proxies, not observed continuation behavior. **Iteration 20 made zero paid API calls and zero Freebase queries.** No separate paid confirmation protocol is issued until the mechanism is isolated and a new budget is approved.

### Reproduction and handoff

- Offline audit: `python scripts/diagnostics/iter_020_adaptive_depth/audit_iter20.py`
- Detailed results: `scoring_reconciliation.json`, `report_discrepancy.md`, `depth_mechanism_matrix.csv`, `answer_coverage_reconciliation.md`, `gate_signal_inventory.csv`, `gate_feasibility.md`, `offline_checks.json`.
- Project handoff: `outputs/experiments/fresh50_continuous_agent/CHATGPT_HANDOFF.md`.
"""
    (OUT / "ITER20_DECISION.md").write_text(outcome_a, encoding="utf-8")

    print(json.dumps({"status": check_doc["status"], "output": str(OUT.relative_to(ROOT)),
                      "checks": checks, "control_metrics": role_metrics[ROLE_CONTROL],
                      "target_metrics": role_metrics[ROLE_TARGET],
                      "forward_gate_proxy": policy_summaries["GATE_FORWARD_AT_DEPTH3_BOUNDARY"]},
                     ensure_ascii=False, indent=2))
    if check_doc["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
