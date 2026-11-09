#!/usr/bin/env python3
"""Prepare, but do not run, an independent depth-4 validation cohort.

This is an offline, outcome-blind cohort freeze. Gold structure and answers are
written only to scoring_gold_only.jsonl and never to inference_cases.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]

import run_fresh50_strong_branch_three_way as fresh  # noqa: E402
import run_phase3c_frozen_generalization as phase3c  # noqa: E402
from relation_graph_kgqa.sparql.parser import parse_query  # noqa: E402

OUT = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_019_depth4_holdout_preparation"
FRESH50_MANIFEST = ROOT / "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256/manifest_frozen.json"
SENSITIVITY_RESULTS = ROOT / "outputs/experiments/phase3a_1/safety_cap_sensitivity/sensitivity_results.jsonl"
SEED = 20261107
TARGET_COUNT = 20
CONTROL_COUNT = 10


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def normalize_node(value: str) -> str:
    value = value.strip("<>")
    prefix = "http://rdf.freebase.com/ns/"
    if value.startswith(prefix):
        return "ns:" + value[len(prefix):]
    if value.startswith(("m.", "g.")):
        return "ns:" + value
    return value


def answer_path_depth(case: dict[str, Any]) -> tuple[int | None, str]:
    """Maximum shortest undirected triple-graph distance over linked topics."""
    features = parse_query(case.get("gold_sparql", ""))
    if not features.parse_success:
        return None, "sparql_parse_failed"
    variables = features.select_variables
    answer_var = "?x" if "?x" in variables else (variables[0] if len(variables) == 1 else None)
    if not answer_var:
        return None, "answer_variable_ambiguous"
    anchors = {normalize_node(item) for item in case.get("topic_entities", []) if item}
    if not anchors:
        return None, "no_topic_entity"
    adjacency: dict[str, set[str]] = {}
    for triple in features.triple_patterns:
        subject, object_ = normalize_node(triple.subject), normalize_node(triple.object)
        if not ((subject.startswith("?") or subject.startswith("ns:")) and (object_.startswith("?") or object_.startswith("ns:"))):
            continue
        adjacency.setdefault(subject, set()).add(object_)
        adjacency.setdefault(object_, set()).add(subject)
    distances: list[int] = []
    for anchor in anchors:
        seen = {anchor: 0}
        queue = deque([anchor])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency.get(node, ()):
                if neighbor not in seen:
                    seen[neighbor] = seen[node] + 1
                    queue.append(neighbor)
        if answer_var in seen:
            distances.append(seen[answer_var])
    return (max(distances), "ok") if distances else (None, "answer_disconnected_from_topics")


def main() -> None:
    strong_pool, source_manifest, prior_ids, fresh_meta = fresh.load_fresh_pool()
    prior_ids = set(prior_ids)
    fresh50 = json.loads(FRESH50_MANIFEST.read_text(encoding="utf-8"))
    prior_ids.update(str(case["question_id"]) for case in fresh50["cases"])
    sensitivity_ids = {
        str(row["case_id"])
        for line in SENSITIVITY_RESULTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row.get("case_id")
    }
    prior_ids.update(sensitivity_ids)

    # Reuse the already validated strong pool; prior IDs from the two newer
    # sources above are removed explicitly.
    eligible_strong = [case for case in strong_pool if case["case_id"] not in prior_ids]

    depth_failures: Counter[str] = Counter()
    for case in eligible_strong:
        depth, status = answer_path_depth(case)
        case["_answer_path_depth"] = depth
        if status != "ok" and case["dataset"] == "cwq":
            depth_failures[f"strong:{status}"] += 1

    targets = [case for case in eligible_strong if case["dataset"] == "cwq" and case["_answer_path_depth"] == 4]
    controls = [
        case for case in eligible_strong
        if case["dataset"] == "cwq"
        and case["_answer_path_depth"] is not None
        and case["_answer_path_depth"] <= 3
    ]
    targets.sort(key=lambda case: case["case_id"])
    controls.sort(key=lambda case: case["case_id"])
    if len(targets) != TARGET_COUNT:
        depth_counts = Counter((case["dataset"], case["_answer_path_depth"]) for case in eligible_strong)
        raise SystemExit(f"Expected a census of {TARGET_COUNT} eligible CWQ depth-4 strong cases; found {len(targets)}. Depth counts={depth_counts}; parse issues={depth_failures}")
    if len(controls) < CONTROL_COUNT:
        raise SystemExit(f"Need {CONTROL_COUNT} eligible CWQ shorter-path strong negative controls; found {len(controls)}")

    selected_controls = random.Random(SEED).sample(controls, CONTROL_COUNT)
    selected_controls.sort(key=lambda case: case["case_id"])
    selected = [(case, "STRONG_DEPTH4_TARGET") for case in targets] + [(case, "STRONG_SHORTER_PATH_NEGATIVE_CONTROL") for case in selected_controls]
    selected_keys = [f"cwq::{case['case_id']}" for case, _ in selected]
    schedule_rng = random.Random(SEED + 1)
    run_schedule = []
    for key in selected_keys:
        for repeat_index in (1, 2):
            condition_order = [3, 4]
            schedule_rng.shuffle(condition_order)
            for depth in condition_order:
                run_schedule.append({"question_key": key, "repeat_index": repeat_index, "depth": depth})
    if len(set(selected_keys)) != len(selected_keys):
        raise SystemExit("Duplicate case keys in proposed independent cohort")

    inference_cases = []
    scoring = []
    for case, role in selected:
        key = f"{case['dataset']}::{case['case_id']}"
        inference_cases.append({
            "question_key": key,
            "case_id": case["case_id"],
            "dataset": case["dataset"],
            "split": case.get("split"),
            "question": case["question"],
            "topic_entities": case["topic_entities"],
            "topic_entity_names": case.get("topic_entity_names", []),
        })
        scoring.append({
            "question_key": key,
            "cohort_role_scoring_only": role,
            "gold_answers": case["gold_answers"],
            "answer_ids": case.get("answer_ids", []),
            "gold_structure_scoring_only": case["gold_structure"],
            "gold_answer_path_depth_scoring_only": case["_answer_path_depth"],
            "necessity_strength_scoring_only": case.get("necessity_strength"),
        })

    input_paths = [
        phase3c.SOURCE_MANIFEST, phase3c.STRONG_BENCHMARK,
        FRESH50_MANIFEST, SENSITIVITY_RESULTS,
        ROOT / "data/processed/cwq_train_analysis.jsonl", ROOT / "data/processed/cwq_test_analysis.jsonl",
        ROOT / "scripts/run_phase3c_frozen_generalization.py",
        ROOT / "scripts/run_fresh50_strong_branch_three_way.py",
        ROOT / "scripts/diagnostics/prepare_fresh50_depth4_holdout.py",
        ROOT / "scripts/diagnostics/run_fresh50_depth4_holdout.py",
    ]
    manifest = {
        "status": "FROZEN_PREPARED_NOT_RUN",
        "experiment": "independent_depth3_vs_depth4_cwq_depth_challenge",
        "selection_seed": SEED,
        "selection_policy": (
            "Census all eligible, previously unused CWQ strongly-branch-required cases whose maximum shortest undirected path "
            "distance from a linked topic entity to the gold SPARQL answer variable is four; sample 10 previously unused CWQ strongly-branch-required "
            "negative controls with gold answer distance <=3 using a fixed seed. Selection uses gold graph structure only for "
            "offline cohort definition/scoring; no gold fields are passed to inference. This is a targeted depth challenge, not a representative KGQA sample."
        ),
        "cases_total": len(selected),
        "strong_depth4_case_count": len(targets),
        "control_case_count": len(selected_controls),
        "strong_cases_are_census_of_eligible_depth4_cwq_pool": True,
        "negative_control_strata": dict(Counter(f"cwq|{case['gold_structure']}|gold_distance_{case['_answer_path_depth']}" for case in selected_controls)),
        "cohort_split_counts": {
            "targets": dict(Counter(str(case.get("split") or "unknown") for case in targets)),
            "negative_controls": dict(Counter(str(case.get("split") or "unknown") for case in selected_controls)),
        },
        "eligible_pool_counts": {
            "strong_all": len(eligible_strong),
            "strong_by_dataset_structure": dict(Counter(f"{c['dataset']}|{c['gold_structure']}" for c in eligible_strong)),
            "strong_by_cwq_gold_distance": dict(Counter(str(c["_answer_path_depth"]) for c in eligible_strong if c["dataset"] == "cwq")),
            "eligible_cwq_strong_negative_controls": len(controls),
            "eligible_cwq_strong_by_structure_and_distance": dict(Counter(f"{c['dataset']}|{c['gold_structure']}|{c['_answer_path_depth']}" for c in controls)),
            "graph_depth_parse_failures": dict(depth_failures),
        },
        "exclusions": {
            "prior_ids_from_phase2_to_phase3c_and_selected_manifests": len(prior_ids),
            "fresh50_manifest_ids_added": len(fresh50["cases"]),
            "phase3a1_sensitivity_case_ids_added": len(sensitivity_ids),
            "fresh_pool_metadata": fresh_meta,
            "fresh_pool_source_manifest_sha256": fresh.sha256(phase3c.SOURCE_MANIFEST),
        },
        "input_hashes": {str(path.relative_to(ROOT)): sha(path) for path in input_paths},
        "prior_excluded_ids_sha256": hashlib.sha256("\n".join(sorted(prior_ids)).encode()).hexdigest(),
        "inference_file": "inference_cases.jsonl",
        "scoring_file": "scoring_gold_only.jsonl",
        "treatment_protocol": {
            "method": "frozen TGAR v1.1.3 with CoR search depth only changed",
            "paired_conditions": [3, 4],
            "repeats_per_condition": 2,
            "total_method_runs": len(selected) * 2 * 2,
            "all_prompts_routing_relation_width_executor_filter_and_scoring_frozen": True,
            "gold_answers_gold_sparql_gold_path_and_cohort_role_hidden_from_inference": True,
            "condition_order_seed": SEED + 1,
            "randomized_run_schedule": run_schedule,
            "frozen_runtime_parameters": {
                "model": "gpt-5.6-luna", "relation_width": 3, "sample_relation_threshold": 1000,
                "remove_unnecessary_relations": True, "temperature_exploration": 0.3,
                "temperature_reasoning": 0.1, "max_completion_tokens": 1024,
                "router_confidence_threshold": 0.75, "router_max_branches": 3,
                "reasoning_effort": "low", "timeout_seconds": 120, "retries": 0,
            },
        },
        "primary_metrics": ["paired Hits@1", "paired F1", "per-case repeat consistency", "control regressions", "API requests", "tokens", "estimated cost"],
        "development_gate": {
            "strong_cases": "Report paired changes across two repeats; a consistent case win means both paired repeats are depth4-only. Do not convert this targeted 20-case census into a population-wide superiority claim.",
            "negative_controls": "Report paired changes across two repeats; flag any case that is depth3-only in both repeats. One-off flips remain visible as stochastic variation.",
            "interpretation": "Independent exploratory evidence for the depth-challenge stratum only; not broad population generalization or a deployment gate.",
        },
        "required_new_authorization": {
            "expected_method_runs": len(selected) * 2 * 2,
            "request_cap": 1000,
            "cost_cap_usd": 0.70,
            "model": "gpt-5.6-luna",
            "max_completion_tokens": 1024,
            "timeout_seconds": 120,
            "retries": 0,
            "status": "NOT_AUTHORIZED",
            "cost_estimate_basis": "Iteration 18 used 59 requests and $0.0292766 across 8 runs (7.375 requests and about $0.00366 per run); 120 runs imply roughly 885 requests and $0.44 before headroom. Caps are hard stops, not spending targets.",
        },
        "case_keys_scoring_only": selected_keys,
    }
    write_json(OUT / "holdout_manifest.json", manifest)
    write_jsonl(OUT / "inference_cases.jsonl", inference_cases)
    write_jsonl(OUT / "scoring_gold_only.jsonl", scoring)
    lines = [
        "# Iteration 19 — independent depth-4 holdout preparation", "",
        "Status: **prepared and frozen; not run**. This selection is independent of the prior Phase 2–3C and Fresh50 diagnostic IDs checked by the exclusion scan.", "",
        f"The eligible pool contains **{len(targets)} CWQ strong cases with maximum shortest gold-query distance four**; all are included. Ten previously unused CWQ strong cases with maximum shortest answer distance at most three were sampled as depth negative controls with seed `{SEED}`. The cohort spans train/test splits: targets `{manifest['cohort_split_counts']['targets']}`, controls `{manifest['cohort_split_counts']['negative_controls']}`.", "",
        "This is deliberately a **targeted depth challenge**, not a representative sample of all KGQA questions. Gold graph depth determines eligibility and is held out of inference. The exact inference and scoring files are separated; `inference_cases.jsonl` contains no gold answers or gold SPARQL.", "",
        "## Proposed test", "",
        "Run frozen TGAR/CoR at depths 3 and 4, two repeats per case and condition (120 method runs). Keep prompts, model, routing, relation width, executor, filtering, and scoring fixed. Compare paired Hits@1/F1 and repeat consistency; report all individual flips and costs. The preregistered development criterion is in `holdout_manifest.json`.", "",
        "## Budget gate", "",
        "No API requests were made. The Iteration 18 pilot averaged 7.375 requests and about $0.00366 per run; 120 runs project to about 885 requests and $0.44. A new hard cap of 1,000 requests or $0.70 is proposed, with 1,024 completion tokens, 120-second timeout, and zero retries. This cap is **not authorized** and does not include any later research beyond this holdout.", "",
        "## Reproduction and artifacts", "",
        "Prepare the frozen cohort offline with `PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/prepare_fresh50_depth4_holdout.py`. The manifest and input hashes are in this directory. The paired runner is implemented and passes a no-network dry run; paid execution remains separately gated by a new explicit authorization.", "",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    hashes = {path.name: sha(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "artifact_hashes.json"}
    hashes["preparation_script_sha256"] = sha(Path(__file__))
    write_json(OUT / "artifact_hashes.json", hashes)
    print(json.dumps({
        "status": manifest["status"], "strong_depth4": len(targets), "cwq_controls_eligible": len(controls),
        "controls_selected": len(selected_controls), "method_runs_proposed": manifest["treatment_protocol"]["total_method_runs"],
        "excluded_prior_ids": len(prior_ids), "api_calls_made": 0, "artifact_hashes": len(hashes),
    }, indent=2))


if __name__ == "__main__":
    main()
