#!/usr/bin/env python3
"""Reproducible zero-call diagnostics for the Fresh50 continuous research agent.

This tool only reads frozen experiment artifacts and archived traces. It never
calls an LLM or writes to Freebase. Outputs are stored in a new experiment
directory and are not used by the production inference path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from scripts.audit_fresh50_tgar_vs_cfp import METHODS, SOURCE_ROOT, gold_relation_tokens, integrity as baseline_integrity, load_bundle
from scripts.diagnostics.run_fresh50_autonomous_diagnosis import graph_summary


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/fresh50_continuous_agent"
FROZEN = ROOT / "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256"
PRIOR_AUDIT = ROOT / "outputs/experiments/fresh50_strong_branch_root_cause_audit"
PRIOR_DIAG = ROOT / "outputs/experiments/fresh50_autonomous_causal_diagnosis"
SOURCE_CODE = [
    "chain_of_relations/methods/cor/agent.py",
    "chain_of_relations/methods/cor/tools/filter.py",
    "chain_of_relations/methods/branch_v1/agent.py",
    "chain_of_relations/methods/adaptive_v1_1_3/agent.py",
    "chain_of_relations/methods/adaptive_v1_1_3/router.py",
    "chain_of_relations/methods/phase2_5_5/agent.py",
    "chain_of_relations/methods/phase2_5_5/router.py",
    "chain_of_relations/tools/entity_prune.py",
    "chain_of_relations/tools/relation_prune.py",
    "chain_of_relations/tools/entity_search.py",
    "chain_of_relations/kg_backend/freebase_backend.py",
    "chain_of_relations/kg_backend/freebase/query_template.yml",
    "scripts/audit_fresh50_tgar_vs_cfp.py",
    "scripts/diagnostics/run_fresh50_autonomous_diagnosis.py",
    "scripts/diagnostics/run_fresh50_continuous_agent.py",
    "scripts/diagnostics/test_fresh50_continuous_agent.py",
    "Codex_Continuous_Research_Agent_Until_Decision.md",
]
METHOD_DIRS = [
    "chain_of_relations/methods/cor",
    "chain_of_relations/methods/branch_v1",
    "chain_of_relations/methods/adaptive_v1_1_3",
    "chain_of_relations/methods/phase2_5_5",
]
LEDGER_HEADER = [
    "iteration", "question_key", "hypothesis", "supporting_trace", "confidence",
    "outcome", "source_artifact", "notes",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def normalized_label(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"\s+", " ", text)


def resolve_labels_to_candidate_ids(
    answers: Iterable[Any], candidate_pairs: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    """Resolve legacy labels only when they uniquely name an allowed candidate."""
    label_to_ids: dict[str, list[str]] = defaultdict(list)
    allowed_ids: set[str] = set()
    for entity_id, label in candidate_pairs:
        entity_id = str(entity_id).strip()
        if not entity_id:
            continue
        allowed_ids.add(entity_id)
        key = normalized_label(label)
        if key and entity_id not in label_to_ids[key]:
            label_to_ids[key].append(entity_id)

    resolved: list[str] = []
    unresolved: list[str] = []
    ambiguous: list[str] = []
    for answer in answers:
        if not isinstance(answer, str) or not normalized_label(answer):
            unresolved.append(str(answer))
            continue
        matches = label_to_ids.get(normalized_label(answer), [])
        if len(matches) == 1:
            if matches[0] not in resolved:
                resolved.append(matches[0])
        elif len(matches) > 1:
            ambiguous.append(answer)
        else:
            unresolved.append(answer)
    if ambiguous:
        status = "AMBIGUOUS_LABEL_REJECTED"
    elif unresolved:
        status = "UNMAPPED_LABEL_REJECTED"
    elif not resolved:
        status = "EMPTY_OUTPUT"
    else:
        status = "RESOLVED_UNIQUE_CANDIDATE_IDS"
    return {
        "status": status,
        "answer_ids": resolved,
        "unresolved_labels": unresolved,
        "ambiguous_labels": ambiguous,
        "allowed_candidate_ids": sorted(allowed_ids),
        "membership_valid": all(entity_id in allowed_ids for entity_id in resolved),
        "deduplicated": True,
    }


def validate_id_payload(payload: Any, allowed_ids: Iterable[str]) -> dict[str, Any]:
    """Validate the proposed strict shape {answer_ids: [candidate ID, ...]}.

    This validates schema and membership only; it does not select answers.
    """
    allowed = set(map(str, allowed_ids))
    try:
        obj = json.loads(payload) if isinstance(payload, str) else payload
    except (TypeError, json.JSONDecodeError):
        return {"valid": False, "status": "INVALID_JSON", "answer_ids": []}
    if not isinstance(obj, dict) or set(obj) != {"answer_ids"} or not isinstance(obj["answer_ids"], list):
        return {"valid": False, "status": "INVALID_SCHEMA", "answer_ids": []}
    values = obj["answer_ids"]
    if any(not isinstance(value, str) or not value.strip() for value in values):
        return {"valid": False, "status": "INVALID_ID_VALUE", "answer_ids": []}
    if any(value not in allowed for value in values):
        return {"valid": False, "status": "ID_OUTSIDE_CANDIDATE_SET", "answer_ids": []}
    unique = list(dict.fromkeys(values))
    return {"valid": True, "status": "VALID", "answer_ids": unique, "duplicates_removed": len(values) - len(unique)}


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_layout() -> None:
    for relative in ("RESULTS", "TESTS", "RESULTS/iter_000_baseline"):
        (OUT / relative).mkdir(parents=True, exist_ok=True)


def source_paths() -> list[Path]:
    paths: set[Path] = set()
    for directory in (FROZEN, PRIOR_AUDIT, PRIOR_DIAG):
        paths.update(path for path in directory.rglob("*") if path.is_file())
    for relative in SOURCE_CODE:
        path = ROOT / relative
        if path.is_file():
            paths.add(path)
    for directory in METHOD_DIRS:
        path = ROOT / directory
        paths.update(item for item in path.rglob("*") if item.is_file())
    return sorted(paths)


def bootstrap() -> dict[str, Any]:
    ensure_layout()
    status_path = OUT / "STATUS.json"
    if status_path.exists():
        status = load_json(status_path)
        return {"resumed": True, "status": status}

    bundle = load_bundle()
    integrity_result = baseline_integrity(bundle)
    if len(bundle["manifest"]) != 50 or len(bundle["run_lines"]) != 150:
        raise RuntimeError("Frozen cohort integrity mismatch; refusing to initialize the study.")
    snapshot_root = OUT / "RESULTS/iter_000_baseline/source_artifacts"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    if any(snapshot_root.iterdir()):
        raise RuntimeError("Partial baseline snapshot exists without STATUS.json; inspect manually before resuming.")
    manifest_rows: list[dict[str, Any]] = []
    for source in source_paths():
        relative = source.relative_to(ROOT)
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        before = sha256(source)
        after = sha256(destination)
        if before != after:
            raise RuntimeError(f"Baseline copy hash mismatch: {relative}")
        manifest_rows.append({"path": str(relative), "size_bytes": source.stat().st_size, "sha256": before})

    git_head = os.popen("git rev-parse HEAD").read().strip()
    git_branch = os.popen("git branch --show-current").read().strip()
    git_status = os.popen("git status --short").read().splitlines()
    production_hashes = {
        relative: sha256(ROOT / relative)
        for relative in SOURCE_CODE
        if (ROOT / relative).is_file() and any(relative.startswith(prefix + "/") for prefix in METHOD_DIRS)
    }
    baseline = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(ROOT),
        "git_head": git_head,
        "git_branch": git_branch,
        "git_status_short_before_new_diagnostics": git_status,
        "frozen_questions": len(bundle["manifest"]),
        "frozen_run_rows": len(bundle["run_lines"]),
        "integrity": integrity_result,
        "files": manifest_rows,
        "production_reasoning_source_hashes": production_hashes,
        "llm_calls": 0,
        "new_kg_queries": 0,
        "snapshot_is_immutable": True,
    }
    dump_json(OUT / "RESULTS/iter_000_baseline/SNAPSHOT_MANIFEST.json", baseline)
    dump_json(OUT / "RESULTS/iter_000_baseline/frozen_integrity.json", integrity_result)

    status = {
        "status": "RUNNING",
        "iteration": 0,
        "current_hypothesis": "A strict deterministic ID adapter may repair formatting or output-mapping errors, independently of semantic selection quality.",
        "last_completed_experiment": "Frozen baseline snapshot and 50-question/150-run integrity verification.",
        "best_supported_conclusion": "Prior evidence confirms 7 conditional answer-filter contributions; this does not validate a deployable selector.",
        "confirmed_rescues": 7,
        "control_regressions": 0,
        "holdout_evaluated": False,
        "remaining_questions": [
            "Does strict label-to-ID conversion improve any of the seven fixed-candidate failures?",
            "For the other 14 failures, where is the first observable loss of the gold ID?",
            "Can archived relation-prune prompts distinguish candidate recall from wrong active state?",
            "Can any executor counterfactual be run without changing the production contract or making paid calls?",
        ],
        "next_action": "Run deterministic strict candidate-label-to-ID adapter on seven failures and five controls; do not call an LLM.",
        "budget_remaining": {
            "RESEARCH_MAX_LLM_CALLS": os.environ.get("RESEARCH_MAX_LLM_CALLS", "UNSET"),
            "RESEARCH_MAX_LLM_COST_USD": os.environ.get("RESEARCH_MAX_LLM_COST_USD", "UNSET"),
            "paid_calls_used": 0,
            "paid_cost_usd": 0.0,
        },
        "artifact_paths": [
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_000_baseline/SNAPSHOT_MANIFEST.json",
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_000_baseline/frozen_integrity.json",
        ],
    }
    dump_json(OUT / "STATUS.json", status)
    dump_json(OUT / "COST_LEDGER.json", {
        "llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
        "cost_usd": 0.0,
        "configured_call_budget": os.environ.get("RESEARCH_MAX_LLM_CALLS", "UNSET"),
        "configured_cost_budget_usd": os.environ.get("RESEARCH_MAX_LLM_COST_USD", "UNSET"),
        "paid_experiment_authorized": False,
        "note": "No positive project call/cost budget is configured; no paid call was made.",
    })
    write_csv(OUT / "EVIDENCE_LEDGER.csv", [], fields=LEDGER_HEADER)
    (OUT / "ITERATION_LOG.md").write_text(
        "# Iteration log (append-only)\n\n"
        f"## Iteration 0 — baseline snapshot — {datetime.now(timezone.utc).isoformat()}\n\n"
        f"- Command: `PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/run_fresh50_continuous_agent.py --init`\n"
        f"- Frozen integrity: {integrity_result['status']}; {len(bundle['manifest'])} questions / {len(bundle['run_lines'])} runs.\n"
        f"- HEAD: `{git_head}`; branch: `{git_branch}`. Existing dirty working tree was preserved.\n"
        f"- Snapshot files: {len(manifest_rows)}; hashes recorded in `RESULTS/iter_000_baseline/SNAPSHOT_MANIFEST.json`.\n"
        "- Paid API calls and new KG queries: 0.\n",
        encoding="utf-8",
    )
    (OUT / "README.md").write_text(
        "# Fresh50 continuous research agent\n\n"
        "This workspace resumes the Fresh50 causal investigation without altering frozen evaluation outputs or production inference. "
        "Read `STATUS.json` and `HYPOTHESIS_QUEUE.md` before resuming. Each iteration is saved under `RESULTS/iter_NNN_*`; "
        "the original frozen source artifacts are hash-snapshotted under `RESULTS/iter_000_baseline/source_artifacts/`.\n\n"
        "No repository clone or branch switch was performed. Existing user worktree changes were left in place. No paid calls are made unless positive project call and cost limits are explicitly configured.\n",
        encoding="utf-8",
    )
    (OUT / "HYPOTHESIS_QUEUE.md").write_text(
        "# Hypothesis queue\n\n"
        "## 1. Output mapping / schema contributes to the seven fixed-candidate misses\n"
        "- Experiment: strict unique candidate-label-to-ID adapter; same frozen candidates and archived selector text; 7 misses + 5 both-correct controls.\n"
        "- Distinguishing result: any corrected miss with no candidate or ranking change would support an output-mapping cause.\n"
        "- Falsifier/no-go: exact unique mapping preserves wrong IDs or rejects an unmapped response; schema-only intervention cannot rescue the errors.\n"
        "- Cost: zero API calls, zero KG calls.\n\n"
        "## 2. Relation candidate exposure versus wrong active state\n"
        "- Experiment: compare each archived relation-prune candidate list and selected IDs with gold predicate inventory across the 21 misses and five controls.\n"
        "- Distinguishing result: gold predicates exposed and omitted is only candidate-ranking evidence if the active binding aligns; global exposure alone is descriptive.\n"
        "- Cost: offline, zero calls.\n\n"
        "## 3. Reconstruct first gold-ID loss for 14 non-selector misses\n"
        "- Use stored entity-search candidate IDs and bounded read-only KG probes if the local endpoint is reachable. Split absent-at-all-searches (12) from earlier-only (2).\n"
        "- No inference calls; no gold values passed into a selector.\n\n"
        "## 4. Non-oracle constrained semantic selector pilot\n"
        "- Up to 12 calls on 7 misses + 5 controls; requires positive `RESEARCH_MAX_LLM_CALLS` and `RESEARCH_MAX_LLM_COST_USD`, explicit model/token/timeout settings.\n"
        "- Hold until that budget is configured; do not treat API key presence or account top-up as a budget.\n",
        encoding="utf-8",
    )
    (OUT / "WORKING_DECISION.md").write_text(
        "# Working decision\n\n"
        "Current evidence supports a conditional final answer-filter contribution on 7/21 TGAR misses, not a deployable repair. "
        "The next zero-cost test is whether strict ID conversion changes those outcomes; after that, use archived relation/entity traces to investigate the remaining 14. "
        "Do not change production methods, use oracle IDs at inference, or claim generalization from Fresh50.\n",
        encoding="utf-8",
    )
    return {"resumed": False, "status": status}


def case_ids(candidate_ids: list[Any], candidate_names: list[Any]) -> list[tuple[str, str]]:
    return [
        (str(candidate_ids[index]), str(candidate_names[index]))
        for index in range(min(len(candidate_ids), len(candidate_names)))
        if str(candidate_ids[index]).strip()
    ]


def set_f1(predicted: set[str], gold: set[str]) -> float:
    if not predicted or not gold:
        return 0.0
    overlap = len(predicted & gold)
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def strict_selector_probe() -> dict[str, Any]:
    result_dir = OUT / "RESULTS/iter_001_id_schema_probe"
    result_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    previous = list(csv.DictReader((PRIOR_DIAG / "answer_selection_counterfactuals.csv").open(newline="", encoding="utf-8")))
    targeted = [
        row for row in previous
        if row["counterfactual_status"] == "RESCUED_BY_GOLD_ID_SELECTION_FROM_FIXED_CANDIDATES"
        or row["cohort_role"] == "BOTH_CORRECT_CONTROL"
    ]
    if len(targeted) != 12:
        raise RuntimeError(f"Expected 7 fixed-candidate misses and 5 controls, got {len(targeted)}")

    non_gold_inputs: list[dict[str, Any]] = []
    score_only: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    for row in targeted:
        dataset, qid = row["question_key"].split("::", 1)
        key = (dataset, qid)
        run = bundle["runs"][(dataset, qid, "v1_1_3")][1]
        gold = bundle["gold"][key]
        searches = [event for event in (run.get("sparql_history") or []) if event.get("search_type") == "entity_search"]
        final_search = searches[-1] if searches else {}
        pairs = case_ids(final_search.get("candidate_entity_ids", []), final_search.get("candidate_entity_names", []))
        answers = list(row["selector_answers"] and json.loads(row["selector_answers"]) or [])
        resolution = resolve_labels_to_candidate_ids(answers, pairs)
        mapped = resolution["status"] == "RESOLVED_UNIQUE_CANDIDATE_IDS"
        proposed_ids = resolution["answer_ids"] if mapped else []
        validation = validate_id_payload({"answer_ids": proposed_ids}, resolution["allowed_candidate_ids"])
        gold_ids = set(map(str, gold.get("answer_ids", []) or []))
        predicted_ids = set(validation.get("answer_ids", [])) if validation.get("valid") and mapped else set()
        official_hit = int(run.get("hits_at_1", 0) or 0)
        id_hit = int(bool(predicted_ids & gold_ids))
        role = row["cohort_role"]
        non_gold_inputs.append({
            "question_key": row["question_key"],
            "question": bundle["manifest"][key].get("question", run.get("question", "")),
            "candidate_entity_ids": [entity_id for entity_id, _ in pairs],
            "candidate_entity_names": [name for _, name in pairs],
            "archived_filter_answers": answers,
            "role": role,
        })
        score_only.append({"question_key": row["question_key"], "gold_answer_ids": sorted(gold_ids)})
        output_rows.append({
            "question_key": row["question_key"],
            "cohort_role": role,
            "candidate_count": len(pairs),
            "gold_candidate_count": len(set(json.loads(row["gold_ids_in_final_candidate_set"]))),
            "archived_filter_answers": json.dumps(answers, ensure_ascii=False),
            "adapter_status": resolution["status"],
            "resolved_answer_ids": json.dumps(validation.get("answer_ids", []), ensure_ascii=False),
            "unresolved_labels": json.dumps(resolution["unresolved_labels"], ensure_ascii=False),
            "ambiguous_labels": json.dumps(resolution["ambiguous_labels"], ensure_ascii=False),
            "candidate_membership_valid": validation.get("valid", False) and resolution["membership_valid"],
            "unique_label_map_succeeded": mapped,
            "strict_id_schema_valid": validation.get("valid", False),
            "official_hits_at_1": official_hit,
            "strict_id_adapter_hits_at_1": id_hit,
            "hits_delta": id_hit - official_hit,
            "official_f1": float(run.get("f1", 0.0) or 0.0),
            "strict_id_adapter_f1": set_f1(predicted_ids, gold_ids),
            "gold_ids_visible_in_final_candidates": bool(gold_ids & set(entity_id for entity_id, _ in pairs)),
            "used_gold_by_adapter": False,
            "llm_calls": 0,
            "adapter_added_tokens": 0,
        })

    # Inputs and labels are intentionally split; only scoring code reads gold IDs.
    (result_dir / "selector_inputs_non_gold.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in non_gold_inputs), encoding="utf-8"
    )
    (result_dir / "scoring_labels_gold_only.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in score_only), encoding="utf-8"
    )
    write_csv(result_dir / "strict_id_adapter_cases.csv", output_rows)

    fail_rows = [row for row in output_rows if row["cohort_role"] == "TGAR_FAILURE"]
    control_rows = [row for row in output_rows if row["cohort_role"] == "BOTH_CORRECT_CONTROL"]
    rescued = sum(row["official_hits_at_1"] == 0 and row["strict_id_adapter_hits_at_1"] == 1 for row in fail_rows)
    regressions = sum(row["official_hits_at_1"] == 1 and row["strict_id_adapter_hits_at_1"] == 0 for row in control_rows)
    invalid = sum(not row["unique_label_map_succeeded"] for row in output_rows)
    ambiguity = sum(bool(row["ambiguous_labels"] != "[]") for row in output_rows)
    metrics = {
        "experiment": "strict legacy-label-to-candidate-ID adapter probe",
        "hypothesis": "Some confirmed answer-filter failures are caused by output label parsing or candidate-ID mapping rather than semantic selection.",
        "competing_explanation": "The selector chose the wrong entity (or abstained) even though candidate IDs and labels map correctly.",
        "cohort": {"target_failures": len(fail_rows), "both_correct_controls": len(control_rows)},
        "gold_visible_to_adapter": False,
        "adapter_input_sha256": sha256(result_dir / "selector_inputs_non_gold.jsonl"),
        "scoring_labels_sha256": sha256(result_dir / "scoring_labels_gold_only.jsonl"),
        "candidate_id_map_validated": True,
        "strict_id_schema": "JSON object with exactly one key `answer_ids`, list of unique allowed candidate ID strings; unknown IDs rejected",
        "rescued_failures": rescued,
        "failure_gate": "PASS" if rescued >= 5 else "FAIL",
        "control_regressions": regressions,
        "controls_preserved": len(control_rows) - regressions,
        "invalid_or_unmapped_outputs": invalid,
        "ambiguous_label_outputs": ambiguity,
        "cases_with_valid_id_schema": sum(row["strict_id_schema_valid"] for row in output_rows),
        "cases_with_successful_unique_label_map": len(output_rows) - invalid,
        "candidate_coverage_failures": sum(not row["gold_ids_visible_in_final_candidates"] for row in fail_rows),
        "mean_f1_all_12": sum(row["strict_id_adapter_f1"] for row in output_rows) / len(output_rows),
        "mean_f1_target_7": sum(row["strict_id_adapter_f1"] for row in fail_rows) / len(fail_rows),
        "added_llm_calls": 0,
        "added_tokens": 0,
        "added_selector_latency_ms": 0,
        "claim_established": "A deterministic output adapter can enforce candidate-ID membership, but cannot repair semantic choices in this cohort.",
        "claim_not_established": "This does not test an LLM prompted to choose IDs directly, and it is not independent validation.",
        "command": "FRESH50_DIAG_MAX_KG_QUERIES=0 PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/run_fresh50_continuous_agent.py --selector-probe",
        "python_version": platform.python_version(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(result_dir / "results.json", metrics)
    dump_json(result_dir / "run_metadata.json", {
        "command": metrics["command"],
        "python_version": platform.python_version(),
        "repo_head": os.popen("git rev-parse HEAD").read().strip(),
        "diagnostic_script_sha256": sha256(ROOT / "scripts/diagnostics/run_fresh50_continuous_agent.py"),
        "unit_test_sha256": sha256(ROOT / "scripts/diagnostics/test_fresh50_continuous_agent.py"),
        "llm_calls": 0,
        "kg_queries": 0,
        "source": "archived v1.1.3 prompt_history and sparql_history plus frozen gold IDs used only in scoring file",
    })
    spec = [
        "# Preregistered experiment: strict candidate-ID adapter probe", "",
        "**Hypothesis:** output parsing or label-to-ID mapping explains some of the seven known fixed-candidate misses.",
        "**Competing explanation:** the legacy selector returns a wrong candidate label or no candidate; representation conversion cannot fix semantic selection.",
        "**Cohort:** the seven confirmed final-candidate failures plus five frozen both-correct controls.",
        "**Method input:** question, stable candidate IDs/names, and archived answer strings. Gold IDs are stored separately for scoring and never passed to the adapter.",
        "**Intervention:** normalize candidate labels with Unicode NFKC, casefold and whitespace collapse; map only a unique exact label to its stable candidate ID; reject missing/ambiguous labels; deduplicate IDs; require strict JSON ID schema and candidate membership.",
        "**Metrics:** case-level Hits@1 and ID-set F1; rescued failures; control losses; invalid/missing/ambiguous outputs; candidate coverage; added calls/tokens/latency.",
        "**Development gate:** at least 5/7 rescued and 0/5 controls lost. This is a development gate only, not a generalization claim.",
        "**Falsifier:** outputs map to wrong candidate IDs or no ID and the strict adapter changes no answer; this rejects parser/schema conversion as the repair.",
        "**Budget:** zero LLM calls, zero KG calls, no additional retrieval.",
        "**Leakage protection:** no gold ID, gold label or gold-derived order is exposed to resolver; gold is scored in a separate pass.",
    ]
    (result_dir / "EXPERIMENT_SPEC.md").write_text("\n".join(spec) + "\n", encoding="utf-8")
    dump_json(result_dir / "artifact_hashes.json", {
        "selector_inputs_non_gold.jsonl": sha256(result_dir / "selector_inputs_non_gold.jsonl"),
        "scoring_labels_gold_only.jsonl": sha256(result_dir / "scoring_labels_gold_only.jsonl"),
        "strict_id_adapter_cases.csv": sha256(result_dir / "strict_id_adapter_cases.csv"),
        "results.json": sha256(result_dir / "results.json"),
    })
    return metrics


def relation_exposure_audit() -> dict[str, Any]:
    result_dir = OUT / "RESULTS/iter_002_relation_exposure_audit"
    result_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    failures = [
        key for key in sorted(bundle["manifest"])
        if not int(bundle["runs"][(key[0], key[1], "v1_1_3")][1].get("hits_at_1", 0))
    ]
    controls = [
        key for key in sorted(bundle["manifest"])
        if all(int(bundle["runs"][(key[0], key[1], method)][1].get("hits_at_1", 0)) for method in METHODS)
    ][:5]
    rows: list[dict[str, Any]] = []
    for key in failures + controls:
        run = bundle["runs"][(key[0], key[1], "v1_1_3")][1]
        gold_sparql = str(bundle["gold"][key].get("gold_sparql", ""))
        gold_relations = sorted(gold_relation_tokens(gold_sparql))
        events = [event for event in (run.get("prompt_history") or []) if event.get("type") == "relation_prune"]
        all_prompts = [str(event.get("prompt", "") or "") for event in events]
        selected_sets = [set(map(str, event.get("parsed_result", []) or [])) for event in events]
        exposed = {
            relation for relation in gold_relations
            if any(re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(relation)}(?![A-Za-z0-9_.-])", prompt) for prompt in all_prompts)
        }
        selected = {
            relation for relation in gold_relations
            if any(relation in selected_set for selected_set in selected_sets)
        }
        key_text = f"{key[0]}::{key[1]}"
        rows.append({
            "question_key": key_text,
            "cohort_role": "TGAR_FAILURE" if key in failures else "BOTH_CORRECT_CONTROL",
            "question": run.get("question", ""),
            "gold_relation_count": len(gold_relations),
            "gold_relations": json.dumps(gold_relations, ensure_ascii=False),
            "relation_prune_event_count": len(events),
            "gold_predicates_exposed_in_any_archived_relation_prompt": json.dumps(sorted(exposed), ensure_ascii=False),
            "gold_predicates_selected_in_any_archived_relation_prune": json.dumps(sorted(selected), ensure_ascii=False),
            "gold_predicates_exposed_but_never_selected": json.dumps(sorted(exposed - selected), ensure_ascii=False),
            "gold_predicates_not_seen_in_any_archived_relation_prompt": json.dumps(sorted(set(gold_relations) - exposed), ensure_ascii=False),
            "causal_status": "DESCRIPTIVE_ONLY_ACTIVE_BINDING_ALIGNMENT_UNAVAILABLE",
            "limitation": "A predicate appearing anywhere in a question's relation-prune prompts does not prove it was offered at the gold-aligned active entity or that the gold plan required that direction at that step.",
        })
    write_csv(result_dir / "gold_relation_exposure.csv", rows)
    failure_rows = [row for row in rows if row["cohort_role"] == "TGAR_FAILURE"]
    control_rows = [row for row in rows if row["cohort_role"] == "BOTH_CORRECT_CONTROL"]
    summary = {
        "experiment": "archived relation-predicate exposure audit",
        "cohort": {"TGAR_failures": len(failure_rows), "both_correct_controls": len(control_rows)},
        "tgar_failures_with_any_gold_predicate_exposed": sum(bool(json.loads(row["gold_predicates_exposed_in_any_archived_relation_prompt"])) for row in failure_rows),
        "tgar_failures_with_any_gold_predicate_selected": sum(bool(json.loads(row["gold_predicates_selected_in_any_archived_relation_prune"])) for row in failure_rows),
        "controls_with_any_gold_predicate_exposed": sum(bool(json.loads(row["gold_predicates_exposed_in_any_archived_relation_prompt"])) for row in control_rows),
        "causal_conclusion": "None: prompts are not aligned to gold active-variable bindings; global predicate exposure cannot establish relation ranking failure.",
        "next_discriminator": "If endpoint is reachable, replay selected archived relation-search SPARQL queries and attach candidate outputs to the exact prune prompt; then align sources/directions only for gold triples sharing a verified active entity ID.",
        "llm_calls": 0,
        "kg_queries": 0,
        "command": "PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/run_fresh50_continuous_agent.py --relation-exposure-audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(result_dir / "results.json", summary)
    dump_json(result_dir / "artifact_hashes.json", {
        "gold_relation_exposure.csv": sha256(result_dir / "gold_relation_exposure.csv"),
        "results.json": sha256(result_dir / "results.json"),
    })
    dump_json(result_dir / "run_metadata.json", {
        "command": summary["command"],
        "python_version": platform.python_version(),
        "repo_head": os.popen("git rev-parse HEAD").read().strip(),
        "diagnostic_script_sha256": sha256(ROOT / "scripts/diagnostics/run_fresh50_continuous_agent.py"),
        "llm_calls": 0,
        "kg_queries": 0,
        "method": "offline exact-token comparison of gold predicates against archived prompts/selections",
    })
    return summary


def root_anchor(query: str, search_type: str) -> tuple[str, str]:
    entity = r"(?:m|g)\.[A-Za-z0-9_-]+"
    outgoing = re.search(rf"ns:({entity})\s+\?relation\s+\?x\b", query)
    incoming = re.search(rf"\?x\s+\?relation\s+ns:({entity})\b", query)
    if outgoing:
        return outgoing.group(1), "OUTGOING"
    if incoming:
        return incoming.group(1), "INCOMING"
    if search_type == "relation_search_head":
        match = re.search(rf"ns:({entity})", query)
        return (match.group(1), "OUTGOING") if match else ("", "UNKNOWN")
    match = re.search(rf"ns:({entity})", query)
    return (match.group(1), "INCOMING") if match else ("", "UNKNOWN")


def freebase_predicate(value: str) -> str:
    value = str(value or "").strip()
    for prefix in ("http://rdf.freebase.com/ns/", "https://rdf.freebase.com/ns/", "ns:"):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def replay_relation_query(query: str, endpoint: str, timeout_seconds: int, max_bytes: int) -> dict[str, Any]:
    payload = urlencode({"query": query, "format": "json"}).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/sparql-results+json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            http_status = response.status
            raw = response.read(max_bytes + 1)
        elapsed = time.perf_counter() - started
        if len(raw) > max_bytes:
            return {"status": "RESPONSE_LIMIT_EXCEEDED", "http_status": http_status, "elapsed_seconds": elapsed, "relation_ids": [], "result_count": None, "error": "Response byte cap exceeded"}
        result = json.loads(raw.decode("utf-8"))
        bindings = result.get("results", {}).get("bindings", [])
        vars_ = result.get("head", {}).get("vars", [])
        relation_var = "relation" if "relation" in vars_ else next((name for name in vars_ if "relation" in name.lower()), "")
        if not relation_var:
            return {
                "status": "INVALID_RESULT_SHAPE", "http_status": http_status,
                "elapsed_seconds": elapsed, "result_count": len(bindings),
                "head_vars": vars_, "relation_variable": "", "relation_ids": [],
                "error": "SPARQL JSON response has no relation variable",
            }
        relation_ids = sorted({
            freebase_predicate(binding.get(relation_var, {}).get("value", ""))
            for binding in bindings
            if relation_var and binding.get(relation_var, {}).get("type") == "uri"
        })
        return {
            "status": "HTTP_OK", "http_status": http_status, "elapsed_seconds": elapsed,
            "result_count": len(bindings), "head_vars": vars_, "relation_variable": relation_var,
            "relation_ids": relation_ids, "error": None,
        }
    except HTTPError as exc:
        elapsed = time.perf_counter() - started
        error_body = exc.read(2000).decode("utf-8", errors="replace")
        return {"status": "HTTP_ERROR", "http_status": exc.code, "elapsed_seconds": elapsed, "relation_ids": [], "result_count": None, "error": error_body}
    except (TimeoutError, URLError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        elapsed = time.perf_counter() - started
        return {"status": "QUERY_ERROR", "http_status": None, "elapsed_seconds": elapsed, "relation_ids": [], "result_count": None, "error": f"{type(exc).__name__}: {exc}"}


def root_relation_replay() -> dict[str, Any]:
    """Replay the first archived CoR relation candidate searches, read-only."""
    result_dir = OUT / "RESULTS/iter_003_root_relation_replay"
    cache_dir = result_dir / "query_cache"
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    bundle = load_bundle()
    endpoint = os.environ.get("FRESH50_DIAG_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql")
    max_queries = max(0, int(os.environ.get("RESEARCH_MAX_KG_QUERIES", "40")))
    timeout_seconds = max(1, int(os.environ.get("RESEARCH_KG_TIMEOUT_SECONDS", "8")))
    max_bytes = max(1024, int(os.environ.get("RESEARCH_KG_MAX_RESPONSE_BYTES", str(8 * 1024 * 1024))))

    counterfactuals = list(csv.DictReader((PRIOR_DIAG / "answer_selection_counterfactuals.csv").open(newline="", encoding="utf-8")))
    fixed_selection_keys = {
        tuple(row["question_key"].split("::", 1)) for row in counterfactuals
        if row["counterfactual_status"] == "RESCUED_BY_GOLD_ID_SELECTION_FROM_FIXED_CANDIDATES"
    }
    failures = [
        key for key in sorted(bundle["manifest"])
        if not int(bundle["runs"][(key[0], key[1], "v1_1_3")][1].get("hits_at_1", 0)) and key not in fixed_selection_keys
    ]
    controls = [
        key for key in sorted(bundle["manifest"])
        if all(int(bundle["runs"][(key[0], key[1], method)][1].get("hits_at_1", 0)) for method in METHODS)
    ][:5]

    cases: list[dict[str, Any]] = []
    query_records: dict[str, dict[str, Any]] = {}
    score_records: list[dict[str, Any]] = []
    for key in failures + controls:
        run = bundle["runs"][(key[0], key[1], "v1_1_3")][1]
        gold = bundle["gold"][key]
        entity_event = next((e for e in run.get("sparql_history", []) if e.get("search_type") == "entity_search"), None)
        first_entity_trace = int(entity_event.get("trace_id", 10**9)) if entity_event else 10**9
        first_prune = next((e for e in run.get("prompt_history", []) if e.get("type") == "relation_prune"), None)
        prune_trace = int(first_prune.get("trace_id", 10**9)) if first_prune else 10**9
        initial_searches = [
            event for event in (run.get("sparql_history") or [])
            if event.get("search_type") in {"relation_search_head", "relation_search_tail"}
            and int(event.get("trace_id", 10**9)) < min(first_entity_trace, prune_trace)
        ]
        selected = list(map(str, (first_prune or {}).get("parsed_result", []) or []))
        selected_set = set(selected)
        graph = graph_summary(str(gold.get("gold_sparql", "")))
        triples = anchor_gold_bgp_to_root(
            graph.get("triple_patterns", []), graph.get("values_bindings", {}), topic_root)
        search_specs: list[dict[str, Any]] = []
        for event in initial_searches:
            query = str(event.get("sparql", ""))
            anchor, direction = root_anchor(query, str(event.get("search_type", "")))
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            search_specs.append({
                "search_type": event.get("search_type", ""),
                "trace_id": event.get("trace_id"),
                "query_sha256": query_hash,
                "query": query,
                "root_entity_id": anchor,
                "direction": direction,
            })
            query_records.setdefault(query_hash, {
                "query": query,
                "query_sha256": query_hash,
                "search_type": event.get("search_type", ""),
                "root_entity_id": anchor,
                "direction": direction,
                "question_key": f"{key[0]}::{key[1]}",
                "trace_id": event.get("trace_id"),
            })

        expected_by_search: dict[str, list[str]] = {}
        for spec in search_specs:
            anchor_term = f"ns:{spec['root_entity_id']}" if spec["root_entity_id"] else ""
            direction = spec["direction"]
            expected: set[str] = set()
            if anchor_term:
                for triple in triples:
                    if direction == "OUTGOING" and triple["subject"] == anchor_term:
                        expected.add(freebase_predicate(triple["predicate"]))
                    elif direction == "INCOMING" and triple["object"] == anchor_term:
                        expected.add(freebase_predicate(triple["predicate"]))
            expected_by_search[spec["query_sha256"]] = sorted(expected)

        key_text = f"{key[0]}::{key[1]}"
        cases.append({
            "question_key": key_text,
            "cohort_role": "TGAR_FAILURE" if key in failures else "BOTH_CORRECT_CONTROL",
            "question": run.get("question", ""),
            "initial_relation_search_count": len(search_specs),
            "first_prune_trace_id": prune_trace if prune_trace < 10**9 else None,
            "first_prune_selected_relations": selected,
            "first_prune_prompt": str((first_prune or {}).get("prompt", "")),
            "gold_relations": sorted(gold_relation_tokens(str(gold.get("gold_sparql", "")))),
            "search_specs": search_specs,
            "expected_by_search": expected_by_search,
            "gold_answer_ids_scoring_only": list(map(str, gold.get("answer_ids", []) or [])),
        })
        score_records.append({
            "question_key": key_text,
            "gold_answer_ids": list(map(str, gold.get("answer_ids", []) or [])),
            "gold_sparql_sha256": hashlib.sha256(str(gold.get("gold_sparql", "")).encode()).hexdigest(),
        })

    query_input_rows = [
        {key: value for key, value in record.items() if key not in {"query"}}
        | {"query": record["query"]}
        for _, record in sorted(query_records.items())
    ]
    query_inputs_path = result_dir / "relation_query_inputs_non_gold.jsonl"
    query_inputs_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in query_input_rows), encoding="utf-8")
    (result_dir / "scoring_gold_only.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in score_records), encoding="utf-8")

    cache: dict[str, dict[str, Any]] = {}
    new_attempts = 0
    for query_hash, record in sorted(query_records.items()):
        cache_path = cache_dir / f"{query_hash}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("query_sha256") == query_hash:
                cache[query_hash] = cached
                continue
        if new_attempts >= max_queries:
            cache[query_hash] = {
                "query_sha256": query_hash, "status": "QUERY_BUDGET_EXHAUSTED", "http_status": None,
                "elapsed_seconds": 0.0, "relation_ids": [], "result_count": None, "error": "RESEARCH_MAX_KG_QUERIES reached",
            }
            continue
        result = replay_relation_query(record["query"], endpoint, timeout_seconds, max_bytes)
        result.update({"query_sha256": query_hash, "query_status_time_utc": datetime.now(timezone.utc).isoformat()})
        dump_json(cache_path, result)
        cache[query_hash] = result
        new_attempts += 1

    case_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for query_hash, record in sorted(query_records.items()):
        response = cache.get(query_hash, {})
        relation_ids = set(map(str, response.get("relation_ids", []) or []))
        query_rows.append({
            "question_key": record["question_key"],
            "search_type": record["search_type"],
            "trace_id": record["trace_id"],
            "root_entity_id": record["root_entity_id"],
            "direction": record["direction"],
            "query_sha256": query_hash,
            "query_status": response.get("status", "NOT_RUN"),
            "http_status": response.get("http_status"),
            "elapsed_seconds": response.get("elapsed_seconds"),
            "result_count": response.get("result_count"),
            "candidate_relation_count": len(relation_ids),
            "candidate_relation_ids": json.dumps(sorted(relation_ids), ensure_ascii=False),
            "error": response.get("error"),
        })

    for case in cases:
        root_results = []
        for spec in case["search_specs"]:
            response = cache.get(spec["query_sha256"], {})
            relation_ids = set(map(str, response.get("relation_ids", []) or []))
            expected = set(case["expected_by_search"].get(spec["query_sha256"], []))
            prompt = case["first_prune_prompt"]
            prompt_exposed = {
                relation for relation in relation_ids
                if relation and re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(relation)}(?![A-Za-z0-9_.-])", prompt)
            }
            root_results.append({
                "search_type": spec["search_type"], "root_entity_id": spec["root_entity_id"],
                "direction": spec["direction"], "query_status": response.get("status", "NOT_RUN"),
                "candidate_relation_count": len(relation_ids),
                "expected_gold_root_relations": sorted(expected),
                "expected_gold_relations_in_candidate_pool": sorted(expected & relation_ids),
                "expected_gold_relations_missing_from_candidate_pool": sorted(expected - relation_ids),
                "prompt_contains_replayed_candidates": len(prompt_exposed),
                "replay_candidates_not_visible_in_prompt": len(relation_ids - prompt_exposed),
                "gold_relations_selected_by_first_prune": sorted(expected & relation_ids & set(case["first_prune_selected_relations"])),
            })
        all_expected = set(x for entry in root_results for x in entry["expected_gold_root_relations"])
        all_present = set(x for entry in root_results for x in entry["expected_gold_relations_in_candidate_pool"])
        all_missing = set(x for entry in root_results for x in entry["expected_gold_relations_missing_from_candidate_pool"])
        all_selected = set(x for entry in root_results for x in entry["gold_relations_selected_by_first_prune"])
        all_status = [entry["query_status"] for entry in root_results]
        if not root_results or any(status != "HTTP_OK" for status in all_status):
            classification = "QUERY_FAILED_OR_NO_INITIAL_SEARCH"
        elif not all_expected:
            classification = "NO_GOLD_EDGE_AT_REPLAYED_TOPIC_ROOT"
        elif all_missing:
            classification = "GOLD_ROOT_EDGE_MISSING_FROM_REPLAYED_CANDIDATE_POOL"
        elif all_selected:
            classification = "AT_LEAST_ONE_GOLD_ROOT_EDGE_SELECTED"
        else:
            classification = "GOLD_ROOT_EDGE_AVAILABLE_BUT_OMITTED_BY_FIRST_PRUNE"
        case_rows.append({
            "question_key": case["question_key"],
            "cohort_role": case["cohort_role"],
            "question": case["question"],
            "root_edge_classification": classification,
            "gold_root_relations": json.dumps(sorted(all_expected), ensure_ascii=False),
            "gold_root_relations_in_replayed_candidates": json.dumps(sorted(all_present), ensure_ascii=False),
            "gold_root_relations_missing_from_replayed_candidates": json.dumps(sorted(all_missing), ensure_ascii=False),
            "gold_root_relations_selected_by_first_prune": json.dumps(sorted(all_selected), ensure_ascii=False),
            "first_prune_selected_relations": json.dumps(case["first_prune_selected_relations"], ensure_ascii=False),
            "search_results": json.dumps(root_results, ensure_ascii=False, sort_keys=True),
            "causal_interpretation": (
                "Source and direction matched a concrete gold edge. Candidate availability plus omission is direct first-hop relation-selection evidence, but no answer-changing plan-only counterfactual was run."
                if classification == "GOLD_ROOT_EDGE_AVAILABLE_BUT_OMITTED_BY_FIRST_PRUNE"
                else "No direct root-edge relation-selection attribution from this test."
            ),
        })

    write_csv(result_dir / "root_relation_case_audit.csv", case_rows)
    write_csv(result_dir / "relation_query_replay.csv", query_rows)
    failure_rows = [row for row in case_rows if row["cohort_role"] == "TGAR_FAILURE"]
    control_rows = [row for row in case_rows if row["cohort_role"] == "BOTH_CORRECT_CONTROL"]
    omitted_failures = [row for row in failure_rows if row["root_edge_classification"] == "GOLD_ROOT_EDGE_AVAILABLE_BUT_OMITTED_BY_FIRST_PRUNE"]
    omitted_controls = [row for row in control_rows if row["root_edge_classification"] == "GOLD_ROOT_EDGE_AVAILABLE_BUT_OMITTED_BY_FIRST_PRUNE"]
    summary = {
        "experiment": "initial root relation candidate replay with exact topic/source and direction alignment",
        "cohort": {"remaining_TGAR_failures": len(failure_rows), "both_correct_controls": len(control_rows)},
        "unique_queries_planned": len(query_records),
        "new_queries_attempted": new_attempts,
        "query_cache_hits": len(query_records) - new_attempts,
        "query_status_counts": {status: sum(1 for row in query_rows if row["query_status"] == status) for status in sorted({row["query_status"] for row in query_rows})},
        "gold_root_edge_available_but_omitted_failures": len(omitted_failures),
        "gold_root_edge_available_but_omitted_controls": len(omitted_controls),
        "gold_root_edge_available_and_selected_failures": sum(row["root_edge_classification"] == "AT_LEAST_ONE_GOLD_ROOT_EDGE_SELECTED" for row in failure_rows),
        "gold_root_edge_absent_failures": sum(row["root_edge_classification"] == "GOLD_ROOT_EDGE_MISSING_FROM_REPLAYED_CANDIDATE_POOL" for row in failure_rows),
        "claim_scope": "Only root relations incident to a gold topic entity are aligned. Candidate availability and omission at this exact first state supports relation-selection contribution, but answer-changing causality still needs a plan-only intervention through the same executor.",
        "endpoint": endpoint,
        "query_timeout_seconds": timeout_seconds,
        "response_byte_cap": max_bytes,
        "new_llm_calls": 0,
        "diagnostic_script_sha256": sha256(ROOT / "scripts/diagnostics/run_fresh50_continuous_agent.py"),
        "command": "PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/run_fresh50_continuous_agent.py --root-relation-replay",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(result_dir / "results.json", summary)
    (result_dir / "EXPERIMENT_SPEC.md").write_text(
        "# Preregistered experiment: root relation candidate replay\n\n"
        "**Hypothesis:** for some unresolved TGAR misses, a gold-required relation at the actual topic entity was available in the exact initial relation-search result but omitted by the first relation-prune output.\n"
        "**Competing explanations:** the relation is absent from the candidate pool; the correct relation is selected and the first error is later; or the gold query has no first-hop edge at the active topic.\n"
        "**Cohort:** the 14 misses outside the seven fixed-candidate answer-selection cases plus five both-correct controls.\n"
        "**Controlled state:** replay only archived `relation_search_head` / `relation_search_tail` SPARQL preceding the first relation-prune, preserving the exact topic entity and direction. Compare against the first prune's archived selected relation IDs.\n"
        "**Gold use:** gold relation structures are diagnostic-only; neither gold triples nor answer IDs are sent to the SPARQL endpoint. Gold IDs are only scored/annotated locally.\n"
        "**Metric:** exact gold root-edge candidate presence, first-prune selection, control omissions, candidate-query status.\n"
        "**Budget/timeout:** at most 40 unique read-only queries; 8 second timeout each; 8 MiB response cap; no retries; cache by query SHA256.\n"
        "**Falsifier:** root-required relations are consistently selected or absent from the aligned pool; global route differences remain unresolved.\n"
        "**Interpretation boundary:** direct aligned candidate omission supports a first-hop selection contribution; no answer-changing executor counterfactual is implied.\n",
        encoding="utf-8",
    )
    dump_json(result_dir / "artifact_hashes.json", {
        "relation_query_inputs_non_gold.jsonl": sha256(query_inputs_path),
        "scoring_gold_only.jsonl": sha256(result_dir / "scoring_gold_only.jsonl"),
        "relation_query_replay.csv": sha256(result_dir / "relation_query_replay.csv"),
        "root_relation_case_audit.csv": sha256(result_dir / "root_relation_case_audit.csv"),
        "results.json": sha256(result_dir / "results.json"),
    })
    dump_json(result_dir / "run_metadata.json", {
        "command": summary["command"],
        "python_version": platform.python_version(),
        "repo_head": os.popen("git rev-parse HEAD").read().strip(),
        "diagnostic_script_sha256": sha256(ROOT / "scripts/diagnostics/run_fresh50_continuous_agent.py"),
        "endpoint": endpoint,
        "timeout_seconds": timeout_seconds,
        "max_unique_query_budget": max_queries,
        "new_query_attempts": new_attempts,
        "llm_calls": 0,
        "response_byte_cap": max_bytes,
    })
    return summary


def bgp_variables(triple: dict[str, str]) -> set[str]:
    return {term[1:] for term in (triple["subject"], triple["predicate"], triple["object"]) if term.startswith("?")}


def sparql_term_from_binding(binding: dict[str, Any]) -> str:
    value = str(binding.get("value", ""))
    if binding.get("type") == "uri":
        prefix = "http://rdf.freebase.com/ns/"
        if value.startswith(prefix):
            return "ns:" + value[len(prefix):]
        return "<" + value.replace(">", "%3E") + ">"
    if binding.get("type") == "literal" or binding.get("type") == "typed-literal":
        term = json.dumps(value, ensure_ascii=False)
        language = str(binding.get("xml:lang", binding.get("lang", "")) or "")
        datatype = str(binding.get("datatype", "") or "")
        if language:
            return term + "@" + language
        if datatype:
            return term + "^^<" + datatype.replace(">", "%3E") + ">"
        return term
    raise ValueError(f"Unsupported SPARQL binding type: {binding.get('type')!r}")


def binding_value_for_score(binding: dict[str, Any]) -> str:
    value = str(binding.get("value", ""))
    if binding.get("type") == "uri":
        return freebase_predicate(value)
    return value


def replay_binding_query(query: str, endpoint: str, timeout_seconds: int, max_bytes: int) -> dict[str, Any]:
    payload = urlencode({"query": query, "format": "json"}).encode("utf-8")
    request = Request(endpoint, data=payload, headers={
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/sparql-results+json",
    }, method="POST")
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status_code = response.status
            raw = response.read(max_bytes + 1)
        elapsed = time.perf_counter() - started
        if len(raw) > max_bytes:
            return {"status": "RESPONSE_LIMIT_EXCEEDED", "http_status": status_code,
                    "elapsed_seconds": elapsed, "rows": [], "error": "Response byte cap exceeded"}
        result = json.loads(raw.decode("utf-8"))
        variables = result.get("head", {}).get("vars", [])
        rows = result.get("results", {}).get("bindings", [])
        if not variables or not isinstance(rows, list):
            return {"status": "INVALID_RESULT_SHAPE", "http_status": status_code,
                    "elapsed_seconds": elapsed, "rows": [], "error": "Missing SPARQL variables or bindings"}
        if len(rows) >= 1001:
            return {"status": "ROW_LIMIT_EXCEEDED", "http_status": status_code,
                    "elapsed_seconds": elapsed, "rows": [], "row_count": len(rows),
                    "error": "Diagnostic row cap (1000) reached"}
        normalized_rows = []
        for row in rows:
            normalized_rows.append({
                variable: sparql_term_from_binding(row[variable])
                for variable in variables if variable in row
            })
        return {"status": "HTTP_OK", "http_status": status_code,
                "elapsed_seconds": elapsed, "variables": variables,
                "rows": normalized_rows, "row_count": len(rows), "error": None}
    except HTTPError as exc:
        elapsed = time.perf_counter() - started
        return {"status": "HTTP_ERROR", "http_status": exc.code, "elapsed_seconds": elapsed,
                "rows": [], "error": exc.read(2000).decode("utf-8", errors="replace")}
    except (TimeoutError, URLError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        elapsed = time.perf_counter() - started
        return {"status": "QUERY_ERROR", "http_status": None, "elapsed_seconds": elapsed,
                "rows": [], "error": f"{type(exc).__name__}: {exc}"}


def build_bgp_step_query(triple: dict[str, str], bindings: list[dict[str, str]], limit: int = 1001) -> str:
    variables = sorted(bgp_variables(triple))
    bound_variables = sorted(set.intersection(*(set(row) for row in bindings))) if bindings else []
    projected = sorted(set(variables) | set(bound_variables))
    projection = " ".join("?" + variable for variable in projected) if projected else "*"
    lines = ["PREFIX ns: <http://rdf.freebase.com/ns/>", f"SELECT DISTINCT {projection} WHERE {{"]
    if bound_variables:
        values_header = " ".join("?" + variable for variable in bound_variables)
        lines.append(f"  VALUES ({values_header}) {{")
        for row in bindings:
            values = " ".join(row[variable] for variable in bound_variables)
            lines.append(f"    ({values})")
        lines.append("  }")
    lines.append("  " + " ".join((triple["subject"], triple["predicate"], triple["object"])) + " .")
    lines.append(f"}} LIMIT {limit}")
    return "\n".join(lines)


def ordered_bgp(root_id: str, triples: list[dict[str, str]]) -> list[dict[str, str]]:
    remaining = list(triples)
    ordered: list[dict[str, str]] = []
    bound: set[str] = set()
    root_term = f"ns:{root_id}"
    while remaining:
        def score(triple: dict[str, str]) -> tuple[int, int, int]:
            variables = bgp_variables(triple)
            overlap = len(variables & bound)
            contains_root = int(root_term in triple.values())
            constants = 3 - len(variables)
            return overlap, contains_root, constants
        selected = max(remaining, key=score)
        if ordered and not (bgp_variables(selected) & bound):
            raise ValueError("Gold BGP contains a disconnected pattern; refusing to infer a join order")
        ordered.append(selected)
        bound.update(bgp_variables(selected))
        remaining.remove(selected)
    return ordered


def execute_bgp_with_bindings(
    triples: list[dict[str, str]], root_id: str, answer_variable: str,
    endpoint: str, timeout_seconds: int, max_bytes: int, max_queries: int,
) -> dict[str, Any]:
    ordered = ordered_bgp(root_id, triples)
    bindings: list[dict[str, str]] = [{}]
    step_rows: list[dict[str, Any]] = []
    query_records: list[dict[str, Any]] = []
    for index, triple in enumerate(ordered, start=1):
        input_binding_count = len(bindings)
        if index > 1:
            triple_variables = bgp_variables(triple)
            if not any(triple_variables & set(row) for row in bindings):
                return {"status": "DISCONNECTED_OR_EMPTY_BINDINGS", "steps": step_rows,
                        "query_records": query_records, "answer_ids": [],
                        "error": "No current binding joins to the next triple"}
        bound_variables = sorted(set.intersection(*(set(row) for row in bindings))) if bindings else []
        chunks = [bindings[i:i + 100] for i in range(0, len(bindings), 100)] if bound_variables else [[]]
        next_bindings: list[dict[str, str]] = []
        query_statuses: list[str] = []
        for chunk in chunks:
            query = build_bgp_step_query(triple, chunk)
            if len(query_records) >= max_queries:
                return {"status": "QUERY_BUDGET_EXHAUSTED", "steps": step_rows,
                        "query_records": query_records, "answer_ids": [],
                        "error": "RESEARCH_BGP_MAX_KG_QUERIES reached"}
            response = replay_binding_query(query, endpoint, timeout_seconds, max_bytes)
            query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
            query_records.append({
                "step": index, "query_sha256": query_hash,
                "query": query, "status": response["status"],
                "elapsed_seconds": response.get("elapsed_seconds"),
                "row_count": response.get("row_count"),
                "error": response.get("error"),
            })
            query_statuses.append(response["status"])
            if response["status"] != "HTTP_OK":
                return {"status": "QUERY_FAILED", "steps": step_rows,
                        "query_records": query_records, "answer_ids": [],
                        "error": response.get("error")}
            next_bindings.extend(response["rows"])
        unique_bindings = {canonical_json_hash(row): row for row in next_bindings}
        bindings = list(unique_bindings.values())
        if len(bindings) > 1000:
            return {"status": "BINDING_ROW_LIMIT_EXCEEDED", "steps": step_rows,
                    "query_records": query_records, "answer_ids": [],
                    "error": "Diagnostic binding cap (1000) reached"}
        step_rows.append({
            "step": index, "triple": triple,
            "input_binding_count": input_binding_count,
            "output_binding_count": len(bindings),
            "query_statuses": query_statuses,
            "bound_variables": sorted(set().union(*(set(row) for row in bindings))) if bindings else [],
        })
        if not bindings:
            break
    answer_key = answer_variable.lstrip("?")
    answers = sorted({
        freebase_predicate(row[answer_key][3:-1]) if row[answer_key].startswith("<")
        else row[answer_key][3:] if row[answer_key].startswith("ns:")
        else row[answer_key]
        for row in bindings if answer_key in row
    })
    return {"status": "BGP_EXECUTED", "steps": step_rows,
            "query_records": query_records, "answer_ids": answers, "error": None}


def finalize_bgp_binding_artifacts(result_dir: Path, endpoint: str, timeout_seconds: int,
                                   max_bytes: int, max_queries: int) -> dict[str, Any]:
    audit_path = result_dir / "bgp_binding_case_audit.csv"
    trace_path = result_dir / "read_only_query_trace.jsonl"
    audits = list(csv.DictReader(audit_path.open(newline="", encoding="utf-8")))
    query_records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    failures = sum(row["cohort_role"] == "TGAR_FAILURE" for row in audits)
    controls = sum(row["cohort_role"] == "BOTH_CORRECT_CONTROL" for row in audits)
    summary = {
        "experiment": "diagnostic-only staged BGP execution with intermediate binding preservation",
        "cohort": {"aligned_root_omission_failures": failures, "aligned_root_omission_controls": controls},
        "cases_with_bgp_gold_answer_covered": sum(int(row["diagnostic_only_adapter_hits_at_1"]) for row in audits),
        "case_count": len(audits), "read_only_kg_queries": len(query_records),
        "query_status_counts": {status: sum(1 for row in query_records if row["status"] == status)
                                 for status in sorted({row["status"] for row in query_records})},
        "gold_passed_to_inference": False, "gold_bgp_used_for_diagnostic_adapter": True,
        "gold_answer_ids_used_only_for_scoring": True, "production_code_changed": False, "llm_calls": 0,
        "claim_scope": "Tests whether a read-only staged adapter can preserve shared bindings and cover the gold answer in these two selected graphs. It does not establish end-to-end model accuracy, official answer filtering, or that the omitted first relation alone caused the failure.",
        "endpoint": endpoint, "query_timeout_seconds": timeout_seconds,
        "response_byte_cap": max_bytes, "max_query_budget": max_queries,
        "command": "RESEARCH_BGP_MAX_KG_QUERIES=40 RESEARCH_KG_TIMEOUT_SECONDS=8 RESEARCH_KG_MAX_RESPONSE_BYTES=8388608 PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/run_fresh50_continuous_agent.py --gold-bgp-binding-replay",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(result_dir / "results.json", summary)
    (result_dir / "EXPERIMENT_SPEC.md").write_text(
        "# Preregistered diagnostic: staged BGP binding preservation\n\n"
        "**Hypothesis:** exact first-hop relation omission may coexist with a path-only executor limitation on the failed question; a diagnostic graph executor that carries shared IDs can traverse the gold BGP.\n"
        "**Cohort:** every prior root-replay case where a gold root-edge predicate was present in the exact replayed candidate pool but absent from first prune: one TGAR miss and one both-correct control.\n"
        "**Intervention:** non-production adapter, fed diagnostic gold triple patterns, executes one triple pattern at a time over read-only SPARQL and carries all shared variable bindings forward. No production API or source is modified.\n"
        "**Scoring separation:** gold answer IDs are held in a separate file and only used to score answer coverage. Gold BGP is oracle diagnostic input and is explicitly not ordinary inference.\n"
        "**Limits:** max 40 queries, 8-second timeout, 8 MiB response cap, 1000 binding rows per step, no retries. SPARQL filters/modifiers are not applied; answer coverage is not official Hits@1.\n"
        "**Interpretation:** successful binding-preserving traversal establishes feasibility of a diagnostic joined execution path only. It cannot prove that this interface change alone fixes TGAR or that the first-prune omission caused the original miss.\n",
        encoding="utf-8")
    dump_json(result_dir / "artifact_hashes.json", {
        name: sha256(result_dir / name) for name in (
            "oracle_bgp_inputs_diagnostic_only.jsonl", "scoring_gold_only.jsonl",
            "read_only_query_trace.jsonl", "bgp_binding_case_audit.csv", "results.json")
    })
    dump_json(result_dir / "run_metadata.json", {
        "command": summary["command"], "python_version": platform.python_version(),
        "repo_head": os.popen("git rev-parse HEAD").read().strip(),
        "diagnostic_script_sha256": sha256(ROOT / "scripts/diagnostics/run_fresh50_continuous_agent.py"),
        "endpoint": endpoint, "timeout_seconds": timeout_seconds,
        "max_query_budget": max_queries, "queries_attempted": len(query_records),
        "response_byte_cap": max_bytes, "llm_calls": 0, "production_code_changed": False,
    })
    return summary


def gold_bgp_binding_replay() -> dict[str, Any]:
    """Diagnostic-only, staged execution of the two aligned root-omission BGPs."""
    result_dir = OUT / "RESULTS/iter_004_gold_bgp_binding_replay"
    endpoint = os.environ.get("FRESH50_DIAG_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql")
    max_queries = max(0, int(os.environ.get("RESEARCH_BGP_MAX_KG_QUERIES", "40")))
    timeout_seconds = max(1, int(os.environ.get("RESEARCH_KG_TIMEOUT_SECONDS", "8")))
    max_bytes = max(1024, int(os.environ.get("RESEARCH_KG_MAX_RESPONSE_BYTES", str(8 * 1024 * 1024))))
    if result_dir.exists() and any(result_dir.iterdir()):
        required = ["oracle_bgp_inputs_diagnostic_only.jsonl", "scoring_gold_only.jsonl",
                    "read_only_query_trace.jsonl", "bgp_binding_case_audit.csv"]
        if all((result_dir / name).is_file() for name in required) and not (result_dir / "results.json").exists():
            return finalize_bgp_binding_artifacts(result_dir, endpoint, timeout_seconds, max_bytes, max_queries)
        raise RuntimeError(f"Refusing to overwrite existing immutable iteration: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    prior = OUT / "RESULTS/iter_003_root_relation_replay/root_relation_case_audit.csv"
    selected = [row for row in csv.DictReader(prior.open(newline="", encoding="utf-8"))
                if row["root_edge_classification"] == "GOLD_ROOT_EDGE_AVAILABLE_BUT_OMITTED_BY_FIRST_PRUNE"]
    target_failures = [row for row in selected if row["cohort_role"] == "TGAR_FAILURE"]
    controls = [row for row in selected if row["cohort_role"] == "BOTH_CORRECT_CONTROL"]
    if len(target_failures) != 1 or len(controls) != 1:
        raise RuntimeError(f"Expected one failure and one control with aligned omission; got {len(target_failures)} and {len(controls)}")
    case_inputs = []
    score_only = []
    audits = []
    all_query_records = []
    used_queries = 0
    for root_row in target_failures + controls:
        dataset, qid = root_row["question_key"].split("::", 1)
        key = (dataset, qid)
        gold = bundle["gold"][key]
        graph = graph_summary(str(gold.get("gold_sparql", "")))
        searches = json.loads(root_row["search_results"])
        root_ids = [item["root_entity_id"] for item in searches if item.get("root_entity_id")]
        topic_root = root_ids[0] if root_ids else ""
        answer_variable = str(graph.get("answer_variable", ""))
        triples = anchor_gold_bgp_to_root(
            graph.get("triple_patterns", []), graph.get("values_bindings", {}), root_id)
        non_gold_input = {
            "question_key": root_row["question_key"],
            "cohort_role": root_row["cohort_role"],
            "question": bundle["runs"][(dataset, qid, "v1_1_3")][1].get("question", ""),
            "root_entity_id": topic_root,
            "gold_bgp_triple_patterns_diagnostic_only": triples,
            "answer_variable": answer_variable,
            "adapter_contract": "sequential read-only triple expansion; carry all bound variables forward and join on shared variable IDs",
        }
        case_inputs.append(non_gold_input)
        score_only.append({"question_key": root_row["question_key"],
                           "gold_answer_ids": sorted(map(str, gold.get("answer_ids", []) or []))})
        try:
            result = execute_bgp_with_bindings(
                triples=triples, root_id=topic_root, answer_variable=answer_variable,
                endpoint=endpoint, timeout_seconds=timeout_seconds, max_bytes=max_bytes,
                max_queries=max_queries - used_queries,
            )
        except (ValueError, KeyError) as exc:
            result = {"status": "INPUT_OR_GRAPH_ERROR", "steps": [], "query_records": [],
                      "answer_ids": [], "error": f"{type(exc).__name__}: {exc}"}
        used_queries += len(result.get("query_records", []))
        all_query_records.extend([{**record, "question_key": root_row["question_key"]}
                                  for record in result.get("query_records", [])])
        gold_ids = set(map(str, gold.get("answer_ids", []) or []))
        replayed = set(map(str, result.get("answer_ids", [])))
        audits.append({
            "question_key": root_row["question_key"], "cohort_role": root_row["cohort_role"],
            "root_relation_omission_trace": root_row["gold_root_relations_selected_by_first_prune"],
            "root_entity_id": topic_root, "graph_topology": graph.get("topology"),
            "gold_triple_pattern_count": len(triples), "answer_variable": answer_variable,
            "adapter_status": result["status"], "diagnostic_only_adapter_hits_at_1": int(bool(gold_ids & replayed)),
            "diagnostic_only_answer_ids": json.dumps(sorted(replayed), ensure_ascii=False),
            "gold_answer_ids_scoring_only": json.dumps(sorted(gold_ids), ensure_ascii=False),
            "gold_answer_covered_by_bgp": json.dumps(sorted(gold_ids & replayed), ensure_ascii=False),
            "adapter_step_trace": json.dumps(result.get("steps", []), ensure_ascii=False, sort_keys=True),
            "query_sha256s": json.dumps([q["query_sha256"] for q in result.get("query_records", [])]),
            "query_count": len(result.get("query_records", [])), "adapter_error": result.get("error"),
            "gold_filters_applied": False,
            "interpretation_limit": "Oracle diagnostic only; gold BGP supplied to read-only adapter. SPARQL FILTER/modifier semantics are not applied, so BGP answer coverage is not official Hits@1 and cannot prove deployable CoR performance.",
        })
    (result_dir / "oracle_bgp_inputs_diagnostic_only.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in case_inputs), encoding="utf-8")
    (result_dir / "scoring_gold_only.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in score_only), encoding="utf-8")
    (result_dir / "read_only_query_trace.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in all_query_records), encoding="utf-8")
    write_csv(result_dir / "bgp_binding_case_audit.csv", audits)
    summary = {
        "experiment": "diagnostic-only staged BGP execution with intermediate binding preservation",
        "cohort": {"aligned_root_omission_failures": len(target_failures), "aligned_root_omission_controls": len(controls)},
        "cases_with_bgp_gold_answer_covered": sum(row["diagnostic_only_adapter_hits_at_1"] for row in audits),
        "case_count": len(audits), "read_only_kg_queries": used_queries,
        "query_status_counts": {status: sum(1 for row in all_query_records if row["status"] == status)
                                 for status in sorted({row["status"] for row in all_query_records})},
        "gold_passed_to_inference": False,
        "gold_bgp_used_for_diagnostic_adapter": True,
        "gold_answer_ids_used_only_for_scoring": True,
        "production_code_changed": False, "llm_calls": 0,
        "claim_scope": "Tests whether a read-only staged adapter can preserve shared bindings and cover the gold answer in these two selected graphs. It does not establish end-to-end model accuracy, official answer filtering, or that the omitted first relation alone caused the failure.",
        "endpoint": endpoint, "query_timeout_seconds": timeout_seconds,
        "response_byte_cap": max_bytes, "max_query_budget": max_queries,
        "command": "RESEARCH_BGP_MAX_KG_QUERIES=40 RESEARCH_KG_TIMEOUT_SECONDS=8 RESEARCH_KG_MAX_RESPONSE_BYTES=8388608 PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/run_fresh50_continuous_agent.py --gold-bgp-binding-replay",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(result_dir / "results.json", summary)
    (result_dir / "EXPERIMENT_SPEC.md").write_text(
        "# Preregistered diagnostic: staged BGP binding preservation\n\n"
        "**Hypothesis:** exact first-hop relation omission may coexist with a path-only executor limitation on the failed question; a diagnostic graph executor that carries shared IDs can traverse the gold BGP.\n"
        "**Cohort:** every prior root-replay case where a gold root-edge predicate was present in the exact replayed candidate pool but absent from first prune: one TGAR miss and one both-correct control.\n"
        "**Intervention:** non-production adapter, fed diagnostic gold triple patterns, executes one triple pattern at a time over read-only SPARQL and carries all shared variable bindings forward. No production API or source is modified.\n"
        "**Scoring separation:** gold answer IDs are held in a separate file and only used to score answer coverage. Gold BGP is oracle diagnostic input and is explicitly not ordinary inference.\n"
        "**Limits:** max 40 queries, 8-second timeout, 8 MiB response cap, 1000 binding rows per step, no retries. SPARQL filters/modifiers are not applied; answer coverage is not official Hits@1.\n"
        "**Interpretation:** successful binding-preserving traversal establishes feasibility of a diagnostic joined execution path only. It cannot prove that this interface change alone fixes TGAR or that the first-prune omission caused the original miss.\n",
        encoding="utf-8")
    dump_json(result_dir / "artifact_hashes.json", {
        name: sha256(result_dir / name) for name in (
            "oracle_bgp_inputs_diagnostic_only.jsonl", "scoring_gold_only.jsonl",
            "read_only_query_trace.jsonl", "bgp_binding_case_audit.csv", "results.json")
    })
    dump_json(result_dir / "run_metadata.json", {
        "command": summary["command"], "python_version": platform.python_version(),
        "repo_head": os.popen("git rev-parse HEAD").read().strip(),
        "diagnostic_script_sha256": sha256(ROOT / "scripts/diagnostics/run_fresh50_continuous_agent.py"),
        "endpoint": endpoint, "timeout_seconds": timeout_seconds,
        "max_query_budget": max_queries, "queries_attempted": used_queries,
        "response_byte_cap": max_bytes, "llm_calls": 0, "production_code_changed": False,
    })
    return summary


def find_gold_entity_path(root_id: str, answer_variable: str,
                          triples: list[dict[str, str]]) -> list[tuple[dict[str, str], bool]]:
    from collections import deque
    root = f"ns:{root_id}"
    answer = answer_variable if answer_variable.startswith("?") else f"?{answer_variable}"
    queue = deque([(root, [])])
    visited = {root}
    while queue:
        node, path = queue.popleft()
        if node == answer:
            return path
        for triple in triples:
            if triple["subject"] == node:
                neighbor, left = triple["object"], True
            elif triple["object"] == node:
                neighbor, left = triple["subject"], False
            else:
                continue
            if neighbor.startswith("?") and neighbor in visited:
                continue
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [(triple, left)]))
    return []


def anchor_gold_bgp_to_root(triples: list[dict[str, str]], values: dict[str, list[str]],
                            root_id: str) -> list[dict[str, str]]:
    root_term = f"ns:{root_id}"
    substitutions: dict[str, str] = {}
    for variable, terms in values.items():
        if root_term in terms:
            substitutions[f"?{variable}"] = root_term
        elif len(terms) == 1:
            substitutions[f"?{variable}"] = terms[0]
    return [
        {key: substitutions.get(term, term) for key, term in triple.items()}
        for triple in triples
    ]


def root_path_candidate_replay(all_remaining: bool = False) -> dict[str, Any]:
    """Run gold-derived path prefixes through the unchanged CoR entity query builder."""
    iteration = 6 if all_remaining else 5
    result_dir = OUT / f"RESULTS/iter_{iteration:03d}_root_path_candidate_replay"
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing immutable iteration: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    from chain_of_relations.kg_backend.freebase_backend import FreebaseBackend
    from chain_of_relations.schema import Entity, Relation
    from chain_of_relations.tools.entity_search import _build_entities_extract

    bundle = load_bundle()
    prior = OUT / "RESULTS/iter_003_root_relation_replay/root_relation_case_audit.csv"
    prior_rows = list(csv.DictReader(prior.open(newline="", encoding="utf-8")))
    selected = (
        [row for row in prior_rows if row["cohort_role"] in {"TGAR_FAILURE", "BOTH_CORRECT_CONTROL"}]
        if all_remaining else
        [row for row in prior_rows
         if row["root_edge_classification"] == "GOLD_ROOT_EDGE_AVAILABLE_BUT_OMITTED_BY_FIRST_PRUNE"]
    )
    endpoint = os.environ.get("FRESH50_DIAG_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql")
    timeout_seconds = max(1, int(os.environ.get("RESEARCH_KG_TIMEOUT_SECONDS", "8")))
    max_bytes = max(1024, int(os.environ.get("RESEARCH_KG_MAX_RESPONSE_BYTES", str(8 * 1024 * 1024))))
    default_query_cap = "100" if all_remaining else "16"
    max_queries = max(0, int(os.environ.get("RESEARCH_PATH_MAX_KG_QUERIES", default_query_cap)))
    max_binding_rows = 1000
    backend = FreebaseBackend()
    input_rows: list[dict[str, Any]] = []
    scoring_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    attempted = 0
    for root_row in selected:
        dataset, qid = root_row["question_key"].split("::", 1)
        gold = bundle["gold"][(dataset, qid)]
        graph = graph_summary(str(gold.get("gold_sparql", "")))
        searches = json.loads(root_row["search_results"])
        root_ids = [item["root_entity_id"] for item in searches if item.get("root_entity_id")]
        root_id = root_ids[0] if root_ids else ""
        triples = anchor_gold_bgp_to_root(
            graph.get("triple_patterns", []), graph.get("values_bindings", {}), root_id)
        path = find_gold_entity_path(root_id, str(graph.get("answer_variable", "")), triples)
        relations = [Relation(id=triple["predicate"].removeprefix("ns:").removeprefix(":"), left=left)
                     for triple, left in path]
        key = (dataset, qid)
        run = bundle["runs"][(dataset, qid, "v1_1_3")][1]
        input_rows.append({
            "question_key": root_row["question_key"], "cohort_role": root_row["cohort_role"],
            "question": run.get("question", ""), "root_entity_id": root_id,
            "gold_relation_path_diagnostic_only": [
                {"predicate": relation.id, "direction": "OUTGOING" if relation.left else "INCOMING"}
                for relation in relations
            ],
            "adapter": "unchanged chain_of_relations.tools.entity_search._build_entities_extract + FreebaseBackend.build_sparql_entities",
            "oracle_relation_path_used": True,
        })
        scoring_rows.append({"question_key": root_row["question_key"],
                             "gold_answer_ids": sorted(map(str, gold.get("answer_ids", []) or []))})
        prefix_rows = []
        status = "PATH_NOT_FOUND"
        if path:
            status = "PATH_EXECUTING"
            for length in range(1, len(relations) + 1):
                relation_chain = relations[:length]
                extract, error = _build_entities_extract(
                    Entity(id=root_id, name="diagnostic root"), relation_chain, backend)
                if error:
                    status = "QUERY_BUILDER_ERROR"
                    break
                query = backend.build_sparql_entities(extract)
                query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
                if attempted >= max_queries:
                    status = "QUERY_BUDGET_EXHAUSTED"
                    break
                attempted += 1
                response = replay_binding_query(query, endpoint, timeout_seconds, max_bytes)
                if response["status"] != "HTTP_OK":
                    status = response["status"]
                    query_rows.append({
                        "question_key": root_row["question_key"], "prefix_length": length,
                        "query_sha256": query_hash, "query": query,
                        "status": response["status"], "elapsed_seconds": response.get("elapsed_seconds"),
                        "candidate_count": None, "gold_answer_candidate": None,
                        "error": response.get("error"),
                    })
                    break
                candidate_ids = sorted({
                    row["targetEntity"][3:] for row in response["rows"]
                    if row.get("targetEntity", "").startswith("ns:")
                })
                gold_ids = set(map(str, gold.get("answer_ids", []) or []))
                prefix_rows.append({
                    "prefix_length": length,
                    "relation_chain": [{"id": relation.id, "direction": "OUTGOING" if relation.left else "INCOMING"}
                                        for relation in relation_chain],
                    "candidate_count": len(candidate_ids),
                    "gold_answer_ids_present": sorted(gold_ids & set(candidate_ids)),
                    "candidate_ids": candidate_ids,
                    "query_sha256": query_hash,
                    "possibly_truncated_at_1000": len(candidate_ids) >= max_binding_rows,
                })
                query_rows.append({
                    "question_key": root_row["question_key"], "prefix_length": length,
                    "query_sha256": query_hash, "query": query,
                    "status": "HTTP_OK", "elapsed_seconds": response.get("elapsed_seconds"),
                    "candidate_count": len(candidate_ids),
                    "gold_answer_candidate": bool(gold_ids & set(candidate_ids)),
                    "possibly_truncated_at_1000": len(candidate_ids) >= max_binding_rows,
                    "error": None,
                })
            else:
                status = "ALL_PATH_PREFIXES_EXECUTED"
        gold_ids = set(map(str, gold.get("answer_ids", []) or []))
        final_candidates = set(prefix_rows[-1]["candidate_ids"]) if prefix_rows else set()
        hit = bool(gold_ids & final_candidates)
        archived_searches = [event for event in run.get("sparql_history", []) if event.get("search_type") == "entity_search"]
        archived_ids = set()
        for event in archived_searches:
            archived_ids.update(map(str, event.get("candidate_entity_ids", []) or []))
        audit_rows.append({
            "question_key": root_row["question_key"], "cohort_role": root_row["cohort_role"],
            "root_entity_id": root_id, "gold_answer_ids_scoring_only": json.dumps(sorted(gold_ids)),
            "gold_path_length": len(relations), "path_status": status,
            "path_prefixes": json.dumps(prefix_rows, ensure_ascii=False, sort_keys=True),
            "gold_answer_in_final_gold_path_candidates": hit,
            "gold_answer_in_archived_entity_search_candidates": bool(gold_ids & archived_ids),
            "final_path_candidate_count": len(final_candidates),
            "gold_filters_or_non_path_branches_applied": False,
            "interpretation": "Uses the gold-derived relation sequence as an oracle diagnostic through the unchanged CoR entity-search query builder. Candidate coverage means a path could retrieve the answer without side constraints; no LLM ranking, pruning, reasoning, or final filtering is tested.",
        })
    (result_dir / "gold_path_inputs_diagnostic_only.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in input_rows), encoding="utf-8")
    (result_dir / "scoring_gold_only.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scoring_rows), encoding="utf-8")
    write_csv(result_dir / "path_candidate_case_audit.csv", audit_rows)
    write_csv(result_dir / "path_prefix_queries.csv", query_rows)
    summary = {
        "experiment": "oracle gold-path candidate replay through unchanged CoR entity-search query builder",
        "iteration": iteration,
        "cohort": {"TGAR_failures": sum(row["cohort_role"] == "TGAR_FAILURE" for row in audit_rows),
                    "both_correct_controls": sum(row["cohort_role"] == "BOTH_CORRECT_CONTROL" for row in audit_rows)},
        "cases_with_gold_answer_in_path_candidates": sum(bool(row["gold_answer_in_final_gold_path_candidates"]) for row in audit_rows),
        "cases_with_gold_answer_absent_from_archived_candidates": sum(not bool(row["gold_answer_in_archived_entity_search_candidates"]) for row in audit_rows),
        "read_only_kg_queries": attempted, "query_status_counts": {
            value: sum(row["status"] == value for row in query_rows)
            for value in sorted({row["status"] for row in query_rows})
        },
        "llm_calls": 0, "production_code_changed": False,
        "claim_scope": "Separates path candidate coverage under oracle relations from archived TGAR candidate coverage. Even if the gold answer is present, this does not validate automatic relation selection or answer filtering; absent results may reflect missing side constraints or route expressivity.",
        "endpoint": endpoint, "query_timeout_seconds": timeout_seconds,
        "response_byte_cap": max_bytes, "max_queries": max_queries,
        "command": (
            "RESEARCH_PATH_MAX_KG_QUERIES=100 RESEARCH_KG_TIMEOUT_SECONDS=8 RESEARCH_KG_MAX_RESPONSE_BYTES=8388608 PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/run_fresh50_continuous_agent.py --remaining-gold-path-replay"
            if all_remaining else
            "RESEARCH_PATH_MAX_KG_QUERIES=16 RESEARCH_KG_TIMEOUT_SECONDS=8 RESEARCH_KG_MAX_RESPONSE_BYTES=8388608 PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/run_fresh50_continuous_agent.py --root-path-candidate-replay"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(result_dir / "results.json", summary)
    spec_lines = [
        "# Preregistered diagnostic: remaining Fresh50 oracle root-path candidate replay" if all_remaining
        else "# Preregistered diagnostic: oracle root-path candidate replay",
        "",
        "**Question:** can the unchanged CoR path query builder retrieve the gold answer when given the exact gold relation sequence, without non-path constraints?",
        ("**Cohort:** the 14 TGAR failures outside the seven fixed-candidate selector cases plus five both-correct controls."
         if all_remaining else
         "**Cohort:** the one TGAR failure and one both-correct control whose first relation prune omitted a gold-required root relation that was present in the exact initial candidate pool."),
        "**Intervention:** use each gold query's shortest root-to-answer relation path to build every prefix with the existing `_build_entities_extract` and `FreebaseBackend.build_sparql_entities`; issue read-only queries only.",
        "**Gold use:** relation path is oracle diagnostic input; gold IDs remain scoring-only. No LLM, production method, or answer filter is used.",
        ("**Limits:** at most 100 queries, 8-second timeout, 8 MiB response cap, existing entity query LIMIT 1000, no retries. GrailQA VALUES anchors are substituted from the stored gold query. Side constraints and non-path branches are not applied."
         if all_remaining else
         "**Limits:** at most 16 queries, 8-second timeout, 8 MiB response cap, existing entity query LIMIT 1000, no retries. Side constraints and non-path branches are not applied."),
        "**Interpretation:** final candidate presence means the path-query builder can retrieve the answer under oracle relations. It does not show the model would select those relations or that route alone suffices for correctness.",
    ]
    (result_dir / "EXPERIMENT_SPEC.md").write_text("\n\n".join(spec_lines) + "\n", encoding="utf-8")
    dump_json(result_dir / "artifact_hashes.json", {
        name: sha256(result_dir / name) for name in (
            "gold_path_inputs_diagnostic_only.jsonl", "scoring_gold_only.jsonl",
            "path_candidate_case_audit.csv", "path_prefix_queries.csv", "results.json")
    })
    dump_json(result_dir / "run_metadata.json", {
        "command": summary["command"], "python_version": platform.python_version(),
        "repo_head": os.popen("git rev-parse HEAD").read().strip(),
        "diagnostic_script_sha256": sha256(ROOT / "scripts/diagnostics/run_fresh50_continuous_agent.py"),
        "endpoint": endpoint, "timeout_seconds": timeout_seconds,
        "max_query_budget": max_queries, "queries_attempted": attempted,
        "response_byte_cap": max_bytes, "llm_calls": 0, "production_code_changed": False,
    })
    return summary


def archived_path_divergence_audit() -> dict[str, Any]:
    """Offline exact-prefix comparison of archived TGAR paths against oracle gold paths."""
    iteration = 7
    result_dir = OUT / "RESULTS/iter_007_archived_path_divergence_audit"
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing immutable iteration: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    prior = OUT / "RESULTS/iter_006_root_path_candidate_replay/path_candidate_case_audit.csv"
    oracle_rows = list(csv.DictReader(prior.open(newline="", encoding="utf-8")))
    by_key = {row["question_key"]: row for row in oracle_rows}
    attribution_path = PRIOR_DIAG / "entity_candidate_reachability.csv"
    reachability = {row["question_key"]: row for row in csv.DictReader(attribution_path.open(newline="", encoding="utf-8"))}
    audit_rows: list[dict[str, Any]] = []
    for key_text, oracle_row in sorted(by_key.items()):
        dataset, qid = key_text.split("::", 1)
        key = (dataset, qid)
        run = bundle["runs"][(dataset, qid, "v1_1_3")][1]
        root_id = oracle_row["root_entity_id"]
        oracle_prefixes = json.loads(oracle_row["path_prefixes"])
        gold_sequence = oracle_prefixes[-1]["relation_chain"] if oracle_prefixes else []
        gold_pairs = [(entry["id"], entry["direction"]) for entry in gold_sequence]
        archived_candidates = []
        for event in run.get("sparql_history", []) or []:
            if event.get("search_type") != "entity_search":
                continue
            query = str(event.get("sparql", "") or "")
            archived_graph = graph_summary(query)
            event_path = find_gold_entity_path(
                root_id, archived_graph.get("answer_variable", "?targetEntity"),
                archived_graph.get("triple_patterns", []))
            sequence = [
                (triple["predicate"].removeprefix("ns:").removeprefix(":"),
                 "OUTGOING" if left else "INCOMING")
                for triple, left in event_path
            ]
            common = 0
            for expected, observed in zip(gold_pairs, sequence):
                if expected != observed:
                    break
                common += 1
            candidate_ids = set(map(str, event.get("candidate_entity_ids", []) or []))
            archived_candidates.append({
                "trace_id": event.get("trace_id"), "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "path_sequence": sequence, "gold_prefix_length": common,
                "executed_path_length": len(sequence),
                "archived_gold_answer_present": bool(candidate_ids & set(json.loads(oracle_row["gold_answer_ids_scoring_only"]))),
                "candidate_count": len(candidate_ids), "query_status": event.get("status", "unknown"),
            })
        best = max(archived_candidates, key=lambda row: (row["gold_prefix_length"], row["executed_path_length"]), default={})
        prefix_length = int(best.get("gold_prefix_length", 0))
        next_edge = gold_sequence[prefix_length] if prefix_length < len(gold_sequence) else None
        first_prune = next((event for event in (run.get("prompt_history") or []) if event.get("type") == "relation_prune"), {})
        prompt = str(first_prune.get("prompt", "") or "")
        selected_relations = set(map(str, first_prune.get("parsed_result", []) or []))
        first_gold_edge = gold_sequence[0] if gold_sequence else None
        if first_gold_edge:
            relation_id = first_gold_edge["id"]
            relation_pattern = rf"(?<![A-Za-z0-9_.-]){re.escape(relation_id)}(?![A-Za-z0-9_.-])"
            prompt_exposed = bool(re.search(relation_pattern, prompt))
            selected_by_predicate = relation_id in selected_relations
        else:
            prompt_exposed = False
            selected_by_predicate = False
        reach = reachability.get(key_text, {})
        gold_ids = set(json.loads(oracle_row["gold_answer_ids_scoring_only"]))
        archived_answer_present = any(row["archived_gold_answer_present"] for row in archived_candidates)
        if prefix_length == len(gold_sequence):
            divergence = "FULL_GOLD_PATH_PREFIX_EXECUTED"
        elif prefix_length == 0:
            divergence = "DIVERGED_AT_FIRST_GOLD_RELATION"
        else:
            divergence = "DIVERGED_AFTER_GOLD_PATH_PREFIX"
        audit_rows.append({
            "question_key": key_text, "cohort_role": oracle_row["cohort_role"],
            "gold_path_length": len(gold_sequence), "max_archived_gold_path_prefix": prefix_length,
            "divergence_status": divergence,
            "first_unreached_gold_edge": json.dumps(next_edge, ensure_ascii=False) if next_edge else "",
            "best_archived_event": json.dumps(best, ensure_ascii=False, sort_keys=True),
            "archived_entity_search_event_count": len(archived_candidates),
            "archived_gold_answer_seen_anywhere": archived_answer_present,
            "frozen_candidate_reachability": reach.get("reachability_status", "UNKNOWN"),
            "gold_answer_in_oracle_path_candidates": oracle_row["gold_answer_in_final_gold_path_candidates"],
            "gold_first_relation_in_first_prune_prompt": prompt_exposed,
            "gold_first_relation_predicate_selected": selected_by_predicate,
            "gold_first_relation_direction": first_gold_edge["direction"] if first_gold_edge else "",
            "first_prune_selected_relations": json.dumps(sorted(selected_relations)),
            "gold_path_relations": json.dumps(gold_sequence, ensure_ascii=False),
            "archived_path_events": json.dumps(archived_candidates, ensure_ascii=False, sort_keys=True),
            "interpretation": "Offline exact relation-ID and direction path comparison. Oracle-path candidate replay used a gold-derived relation sequence; this audit identifies where archived search paths diverged but cannot prove a single intervention would change the final answer.",
        })
    write_csv(result_dir / "archived_gold_path_alignment.csv", audit_rows)
    failure_rows = [row for row in audit_rows if row["cohort_role"] == "TGAR_FAILURE"]
    control_rows = [row for row in audit_rows if row["cohort_role"] == "BOTH_CORRECT_CONTROL"]
    summary = {
        "experiment": "offline archived TGAR path alignment against gold root-to-answer relation paths",
        "cohort": {"remaining_TGAR_failures": len(failure_rows), "both_correct_controls": len(control_rows)},
        "oracle_path_answer_coverage": sum(row["gold_answer_in_oracle_path_candidates"] == "True" for row in audit_rows),
        "failure_max_prefix_distribution": {
            str(length): sum(int(row["max_archived_gold_path_prefix"]) == length for row in failure_rows)
            for length in sorted({int(row["max_archived_gold_path_prefix"]) for row in failure_rows})
        },
        "failures_with_no_gold_path_prefix_executed": sum(int(row["max_archived_gold_path_prefix"]) == 0 for row in failure_rows),
        "failures_with_full_gold_path_executed": sum(row["divergence_status"] == "FULL_GOLD_PATH_PREFIX_EXECUTED" for row in failure_rows),
        "failures_with_gold_answer_seen_in_archived_entity_candidates": sum(row["archived_gold_answer_seen_anywhere"] for row in failure_rows),
        "failures_first_gold_relation_exposed_in_prompt": sum(row["gold_first_relation_in_first_prune_prompt"] == "True" for row in failure_rows),
        "failures_first_gold_relation_predicate_selected": sum(row["gold_first_relation_predicate_selected"] == "True" for row in failure_rows),
        "controls_with_no_gold_path_prefix_executed": sum(int(row["max_archived_gold_path_prefix"]) == 0 for row in control_rows),
        "llm_calls": 0, "kg_queries": 0,
        "claim_scope": "If a gold answer is retrieved by an oracle path using the unchanged query builder but archived TGAR never executes the full gold path, route selection/planning is supported as a bottleneck in those cases. This is not an answer-changing intervention and does not generalize beyond this frozen cohort.",
        "command": "PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/run_fresh50_continuous_agent.py --archived-path-divergence-audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(result_dir / "results.json", summary)
    (result_dir / "EXPERIMENT_SPEC.md").write_text(
        "# Offline diagnostic: archived route versus oracle gold-path prefixes\n\n"
        "**Cohort:** the 14 unresolved TGAR misses plus five frozen both-correct controls.\n"
        "**Method:** derive shortest gold root-to-answer relation sequences from frozen gold query structures; parse archived `entity_search` query paths and compute exact common relation-ID/direction prefix. Compare with read-only oracle path candidate results from iteration 6.\n"
        "**Gold use:** gold paths and answer IDs are diagnostic/scoring only; no gold is sent to any LLM. No KG request is made.\n"
        "**Metric:** maximum archived gold-path prefix, full-path execution, archived gold ID presence, first relation exposure/selection, and control comparison.\n"
        "**Limit:** association and trace alignment only. This does not inject one relation and hold later decisions fixed, nor prove a final-answer counterfactual.\n",
        encoding="utf-8")
    dump_json(result_dir / "artifact_hashes.json", {
        name: sha256(result_dir / name) for name in ("archived_gold_path_alignment.csv", "results.json")
    })
    dump_json(result_dir / "run_metadata.json", {
        "command": summary["command"], "python_version": platform.python_version(),
        "repo_head": os.popen("git rev-parse HEAD").read().strip(),
        "diagnostic_script_sha256": sha256(ROOT / "scripts/diagnostics/run_fresh50_continuous_agent.py"),
        "graph_parser_script_sha256": sha256(ROOT / "scripts/diagnostics/run_fresh50_autonomous_diagnosis.py"),
        "llm_calls": 0, "kg_queries": 0,
    })
    return summary


def parse_directional_relation_prompt(prompt: str) -> tuple[set[str], set[str]]:
    head: set[str] = set()
    tail: set[str] = set()
    for section in re.split(r"(?m)^# ", prompt):
        header, separator, body = section.partition("\n")
        if not separator:
            continue
        target = head if re.search(r"\?relation\s+\?x\b", header) else (
            tail if re.search(r"\?x\s+\?relation\b", header) else None
        )
        if target is None:
            continue
        for token in body.replace("\n", " ").split(";"):
            relation = token.strip()
            if relation and not relation.startswith("Answer:"):
                target.add(relation)
    return head, tail


def relation_edge_match_kind(expected: tuple[str, str], observed: tuple[str, str],
                            inverse_map: dict[str, list[str]]) -> str:
    if expected == observed:
        return "EXACT"
    expected_id, expected_direction = expected
    observed_id, observed_direction = observed
    if (observed_id in inverse_map.get(expected_id, [])
            and expected_direction != observed_direction):
        return "EXPLICIT_INVERSE_EQUIVALENT"
    return "NO_MATCH"


def archived_path_direction_audit() -> dict[str, Any]:
    """Correct archived root-prune analysis for the predicate's selected direction."""
    result_dir = OUT / "RESULTS/iter_008_archived_path_direction_audit"
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing immutable iteration: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    previous = list(csv.DictReader((OUT / "RESULTS/iter_007_archived_path_divergence_audit/archived_gold_path_alignment.csv").open(newline="", encoding="utf-8")))
    output_rows = []
    for row in previous:
        dataset, qid = row["question_key"].split("::", 1)
        run = bundle["runs"][(dataset, qid, "v1_1_3")][1]
        first_prune = next((event for event in (run.get("prompt_history") or []) if event.get("type") == "relation_prune"), {})
        prompt = str(first_prune.get("prompt", "") or "")
        head, tail = parse_directional_relation_prompt(prompt)
        selected = list(map(str, first_prune.get("parsed_result", []) or []))
        selected_direction = {}
        for relation in selected:
            if relation in head:
                selected_direction[relation] = "OUTGOING"
            elif relation in tail:
                selected_direction[relation] = "INCOMING"
            else:
                selected_direction[relation] = "UNKNOWN_NOT_IN_DIRECTIONAL_PROMPT"
        gold_path = json.loads(row["gold_path_relations"])
        first_gold = gold_path[0] if gold_path else {}
        gold_relation = str(first_gold.get("id", ""))
        gold_direction = str(first_gold.get("direction", ""))
        matching_candidates = head if gold_direction == "OUTGOING" else tail
        offered = gold_relation in matching_candidates
        output_rows.append({
            **row,
            "first_prune_head_candidate_count": len(head),
            "first_prune_tail_candidate_count": len(tail),
            "gold_first_relation_in_matching_direction_candidates": offered,
            "gold_first_relation_selected_direction": selected_direction.get(gold_relation, "NOT_SELECTED"),
            "gold_first_relation_selected_in_matching_direction": selected_direction.get(gold_relation) == gold_direction,
            "selected_relation_direction_map": json.dumps(selected_direction, ensure_ascii=False, sort_keys=True),
            "direction_inference_method": "reconstruct _clean_relations behavior: selected predicate maps to outgoing if it appears in head; else incoming if in tail; current CoR parser gives head precedence for duplicate IDs",
        })
    write_csv(result_dir / "archived_gold_path_direction_alignment.csv", output_rows)
    failures = [row for row in output_rows if row["cohort_role"] == "TGAR_FAILURE"]
    controls = [row for row in output_rows if row["cohort_role"] == "BOTH_CORRECT_CONTROL"]
    summary = {
        "experiment": "direction-aware archived root relation-prune alignment",
        "cohort": {"remaining_TGAR_failures": len(failures), "both_correct_controls": len(controls)},
        "failures_with_gold_first_relation_offered_in_correct_direction": sum(bool(row["gold_first_relation_in_matching_direction_candidates"]) for row in failures),
        "failures_with_gold_first_relation_selected_in_correct_direction": sum(bool(row["gold_first_relation_selected_in_matching_direction"]) for row in failures),
        "failures_with_gold_first_relation_predicate_selected_wrong_or_unknown_direction": sum(
            row["gold_first_relation_selected_direction"] not in {"NOT_SELECTED", row["gold_first_relation_direction"]}
            for row in failures),
        "controls_with_gold_first_relation_offered_in_correct_direction": sum(bool(row["gold_first_relation_in_matching_direction_candidates"]) for row in controls),
        "controls_with_gold_first_relation_selected_in_correct_direction": sum(bool(row["gold_first_relation_selected_in_matching_direction"]) for row in controls),
        "llm_calls": 0, "kg_queries": 0,
        "claim_scope": "Correct direction-aware comparison of the first gold relation against archived root-prune candidates and selected relations. Supports relation-prune decision error only when the exact direction was offered but not selected; does not by itself prove the answer would change.",
        "command": "PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/run_fresh50_continuous_agent.py --archived-path-direction-audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(result_dir / "results.json", summary)
    (result_dir / "EXPERIMENT_SPEC.md").write_text(
        "# Direction-aware archived root relation-prune audit\n\n"
        "Reuses iteration 7's frozen exact path-prefix comparisons and reconstructs relation direction from the archived directional prompt and `_clean_relations` parser behavior. Reports whether the first gold path edge was offered and selected in its required direction. This is offline trace analysis; no LLM or KG calls. Selection evidence is not an answer-changing counterfactual.\n",
        encoding="utf-8")
    dump_json(result_dir / "artifact_hashes.json", {
        name: sha256(result_dir / name) for name in ("archived_gold_path_direction_alignment.csv", "results.json")
    })
    dump_json(result_dir / "run_metadata.json", {
        "command": summary["command"], "python_version": platform.python_version(),
        "repo_head": os.popen("git rev-parse HEAD").read().strip(),
        "diagnostic_script_sha256": sha256(ROOT / "scripts/diagnostics/run_fresh50_continuous_agent.py"),
        "relation_prune_source_sha256": sha256(ROOT / "chain_of_relations/tools/relation_prune.py"),
        "llm_calls": 0, "kg_queries": 0,
    })
    return summary


def lookup_inverse_relation_catalog() -> dict[str, Any]:
    """Read Freebase's explicit inverse-property metadata for audited route predicates."""
    result_dir = OUT / "RESULTS/iter_009_inverse_relation_catalog"
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing immutable iteration: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    prior = OUT / "RESULTS/iter_007_archived_path_divergence_audit/archived_gold_path_alignment.csv"
    relation_ids: set[str] = set()
    for row in csv.DictReader(prior.open(newline="", encoding="utf-8")):
        for edge in json.loads(row["gold_path_relations"]):
            relation_ids.add(str(edge["id"]))
        for event in json.loads(row["archived_path_events"]):
            for relation, _direction in event.get("path_sequence", []):
                relation_ids.add(str(relation))
    endpoint = os.environ.get("FRESH50_DIAG_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql")
    timeout_seconds = max(1, int(os.environ.get("RESEARCH_KG_TIMEOUT_SECONDS", "8")))
    max_bytes = max(1024, int(os.environ.get("RESEARCH_KG_MAX_RESPONSE_BYTES", str(8 * 1024 * 1024))))
    values = " ".join(f"ns:{relation}" for relation in sorted(relation_ids))
    query = (
        "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
        "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
        "SELECT DISTINCT ?relation ?inverse WHERE {\n"
        f"  VALUES ?relation {{ {values} }}\n"
        "  { ?relation ns:type.property.reverse_property ?inverse . } UNION\n"
        "  { ?relation owl:inverseOf ?inverse . } UNION\n"
        "  { ?inverse ns:type.property.reverse_property ?relation . } UNION\n"
        "  { ?inverse owl:inverseOf ?relation . }\n"
        "} LIMIT 1001"
    )
    timeout_seconds = max(timeout_seconds, 12)
    response = replay_binding_query(query, endpoint, timeout_seconds, max_bytes)
    raw_pairs = []
    if response["status"] == "HTTP_OK":
        for row in response["rows"]:
            relation = str(row.get("relation", ""))
            inverse = str(row.get("inverse", ""))
            if relation.startswith("ns:") and inverse.startswith("ns:"):
                raw_pairs.append((relation[3:], inverse[3:]))
    symmetric_pairs = sorted({tuple(sorted((left, right))) for left, right in raw_pairs if left != right})
    inverse_map: dict[str, list[str]] = defaultdict(list)
    for left, right in symmetric_pairs:
        inverse_map[left].append(right)
        inverse_map[right].append(left)
    metadata = {
        "experiment": "lookup explicitly declared inverse properties for gold and archived route predicates",
        "unique_relation_ids": len(relation_ids), "query_status": response["status"],
        "result_rows": response.get("row_count", 0), "explicit_inverse_pairs": len(symmetric_pairs),
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "query_timeout_seconds": timeout_seconds, "response_byte_cap": max_bytes,
        "llm_calls": 0, "read_only_kg_queries": 1,
        "claim_scope": "Only explicitly recorded owl:inverseOf or type.property.reverse_property pairs are treated as semantic inverse equivalents; unrecorded inverses remain unmatched.",
        "endpoint": endpoint, "command": "PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/run_fresh50_continuous_agent.py --inverse-relation-catalog",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(result_dir / "relation_inverse_map.json", dict(sorted(inverse_map.items())))
    dump_json(result_dir / "query_metadata.json", metadata)
    (result_dir / "query.sparql").write_text(query + "\n", encoding="utf-8")
    (result_dir / "query_result.json").write_text(json.dumps(response, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    dump_json(result_dir / "artifact_hashes.json", {
        name: sha256(result_dir / name) for name in ("relation_inverse_map.json", "query_metadata.json", "query.sparql", "query_result.json")
    })
    dump_json(result_dir / "run_metadata.json", {
        "command": metadata["command"], "python_version": platform.python_version(),
        "repo_head": os.popen("git rev-parse HEAD").read().strip(),
        "diagnostic_script_sha256": sha256(ROOT / "scripts/diagnostics/run_fresh50_continuous_agent.py"),
        "query_sha256": metadata["query_sha256"], "llm_calls": 0, "kg_queries": 1,
    })
    return metadata


def semantic_archived_path_audit() -> dict[str, Any]:
    """Recompute route-prefix alignment allowing only KG-declared inverse properties."""
    result_dir = OUT / "RESULTS/iter_010_semantic_archived_path_audit"
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing immutable iteration: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    inverse_map = load_json(OUT / "RESULTS/iter_009_inverse_relation_catalog/relation_inverse_map.json")
    previous = list(csv.DictReader((OUT / "RESULTS/iter_007_archived_path_divergence_audit/archived_gold_path_alignment.csv").open(newline="", encoding="utf-8")))
    directional = {row["question_key"]: row for row in csv.DictReader(
        (OUT / "RESULTS/iter_008_archived_path_direction_audit/archived_gold_path_direction_alignment.csv").open(newline="", encoding="utf-8"))}
    bundle = load_bundle()
    output_rows = []
    for row in previous:
        dataset, qid = row["question_key"].split("::", 1)
        run = bundle["runs"][(dataset, qid, "v1_1_3")][1]
        gold_sequence_dict = json.loads(row["gold_path_relations"])
        gold_sequence = [(edge["id"], edge["direction"]) for edge in gold_sequence_dict]
        paths = []
        for event in json.loads(row["archived_path_events"]):
            observed = [tuple(edge) for edge in event.get("path_sequence", [])]
            matches = [relation_edge_match_kind(expected, actual, inverse_map)
                       for expected, actual in zip(gold_sequence, observed)]
            common = 0
            for match in matches:
                if match == "NO_MATCH":
                    break
                common += 1
            paths.append({
                **event, "semantic_gold_prefix_length": common,
                "prefix_match_kinds": matches[:common],
                "exact_relation_prefix_length": sum(kind == "EXACT" for kind in matches[:common]),
                "explicit_inverse_prefix_edges": sum(kind == "EXPLICIT_INVERSE_EQUIVALENT" for kind in matches[:common]),
            })
        best = max(paths, key=lambda item: (item["semantic_gold_prefix_length"], item["executed_path_length"]), default={})
        max_prefix = int(best.get("semantic_gold_prefix_length", 0))
        next_edge = gold_sequence_dict[max_prefix] if max_prefix < len(gold_sequence_dict) else None

        first_gold = gold_sequence_dict[0] if gold_sequence_dict else {}
        gold_id, gold_direction = str(first_gold.get("id", "")), str(first_gold.get("direction", ""))
        prompt_event = next((event for event in (run.get("prompt_history") or []) if event.get("type") == "relation_prune"), {})
        head, tail = parse_directional_relation_prompt(str(prompt_event.get("prompt", "") or ""))
        selected = list(map(str, prompt_event.get("parsed_result", []) or []))
        selected_dirs = {}
        for relation in selected:
            if relation in head:
                selected_dirs[relation] = "OUTGOING"
            elif relation in tail:
                selected_dirs[relation] = "INCOMING"
        expected_candidates = head if gold_direction == "OUTGOING" else tail
        inverse_candidates = tail if gold_direction == "OUTGOING" else head
        explicit_inverses = set(inverse_map.get(gold_id, []))
        offered_exact = gold_id in expected_candidates
        offered_inverse = bool(explicit_inverses & inverse_candidates)
        selected_exact = selected_dirs.get(gold_id) == gold_direction
        selected_bidirectionally = gold_id in selected and gold_id in head and gold_id in tail
        opposite_direction = "INCOMING" if gold_direction == "OUTGOING" else "OUTGOING"
        selected_inverse_ids = sorted(relation for relation in explicit_inverses
                                      if selected_dirs.get(relation) == opposite_direction)
        reachability = directional[row["question_key"]]
        output_rows.append({
            "question_key": row["question_key"], "cohort_role": row["cohort_role"],
            "gold_path_length": len(gold_sequence), "max_archived_semantic_gold_prefix": max_prefix,
            "semantic_divergence_status": "FULL_SEMANTIC_GOLD_PATH_EXECUTED" if max_prefix == len(gold_sequence)
                else "DIVERGED_AT_FIRST_EDGE" if max_prefix == 0 else "DIVERGED_AFTER_SEMANTIC_PREFIX",
            "next_unreached_gold_edge": json.dumps(next_edge, ensure_ascii=False) if next_edge else "",
            "best_archived_path_event": json.dumps(best, ensure_ascii=False, sort_keys=True),
            "archived_gold_answer_seen_anywhere": row["archived_gold_answer_seen_anywhere"],
            "gold_answer_in_oracle_path_candidates": row["gold_answer_in_oracle_path_candidates"],
            "first_gold_relation_id": gold_id, "first_gold_relation_direction": gold_direction,
            "first_relation_exact_offered_in_direction": offered_exact,
            "explicit_inverse_offered_in_opposite_direction": offered_inverse,
            "first_relation_semantically_offered": offered_exact or offered_inverse,
            "first_relation_exact_selected_in_direction": selected_exact,
            "first_relation_selected_and_expanded_both_directions": selected_bidirectionally,
            "explicit_inverse_selected_in_opposite_direction": json.dumps(selected_inverse_ids),
            "first_relation_semantically_selected": selected_exact or selected_bidirectionally or bool(selected_inverse_ids),
            "prior_max_exact_id_prefix": row["max_archived_gold_path_prefix"],
            "prefix_match_kinds": json.dumps(best.get("prefix_match_kinds", [])),
            "interpretation": "Path edges match by exact predicate+direction or an explicitly recorded Freebase inverse predicate with opposite direction. Gold route remains oracle input; no answer-changing single-step intervention was run.",
        })
    write_csv(result_dir / "semantic_archived_path_alignment.csv", output_rows)
    failures = [row for row in output_rows if row["cohort_role"] == "TGAR_FAILURE"]
    controls = [row for row in output_rows if row["cohort_role"] == "BOTH_CORRECT_CONTROL"]
    summary = {
        "experiment": "archived gold-path alignment with explicit inverse-property equivalence",
        "cohort": {"remaining_TGAR_failures": len(failures), "both_correct_controls": len(controls)},
        "oracle_path_gold_answer_coverage": sum(row["gold_answer_in_oracle_path_candidates"] == "True" for row in output_rows),
        "failures_with_full_semantic_gold_path_executed": sum(row["semantic_divergence_status"] == "FULL_SEMANTIC_GOLD_PATH_EXECUTED" for row in failures),
        "failures_with_semantic_prefix_zero": sum(int(row["max_archived_semantic_gold_prefix"]) == 0 for row in failures),
        "failures_with_gold_answer_in_archived_candidates": sum(row["archived_gold_answer_seen_anywhere"] == "True" for row in failures),
        "failures_with_gold_first_relation_semantically_offered": sum(bool(row["first_relation_semantically_offered"]) for row in failures),
        "failures_with_gold_first_relation_semantically_selected": sum(bool(row["first_relation_semantically_selected"]) for row in failures),
        "controls_with_full_semantic_gold_path_executed": sum(row["semantic_divergence_status"] == "FULL_SEMANTIC_GOLD_PATH_EXECUTED" for row in controls),
        "explicit_inverse_pairs_available": len(inverse_map), "llm_calls": 0, "kg_queries": 0,
        "claim_scope": "Oracle gold-path candidate availability plus semantic path-prefix trace matching can support a navigation/selection bottleneck where archived routes diverge. This does not prove one-step answer-changing causality or validate a deployable selector.",
        "command": "PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/run_fresh50_continuous_agent.py --semantic-archived-path-audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(result_dir / "results.json", summary)
    (result_dir / "EXPERIMENT_SPEC.md").write_text(
        "# Semantic archived route alignment with explicit inverse properties\n\n"
        "Compares each archived TGAR entity-search path to the gold root-to-answer path. An edge matches only if predicate and direction are exact, or if Freebase explicitly declares the two predicates inverse and route direction is opposite. Inverse mappings come from a bounded read-only endpoint query. Gold routes are oracle diagnostic inputs; candidate coverage is not official answer accuracy.\n",
        encoding="utf-8")
    dump_json(result_dir / "artifact_hashes.json", {
        name: sha256(result_dir / name) for name in ("semantic_archived_path_alignment.csv", "results.json")
    })
    dump_json(result_dir / "run_metadata.json", {
        "command": summary["command"], "python_version": platform.python_version(),
        "repo_head": os.popen("git rev-parse HEAD").read().strip(),
        "diagnostic_script_sha256": sha256(ROOT / "scripts/diagnostics/run_fresh50_continuous_agent.py"),
        "inverse_map_sha256": sha256(OUT / "RESULTS/iter_009_inverse_relation_catalog/relation_inverse_map.json"),
        "llm_calls": 0, "kg_queries": 0,
    })
    return summary


def add_ledger_rows(rows: list[dict[str, Any]]) -> None:
    path = OUT / "EVIDENCE_LEDGER.csv"
    existing = list(csv.DictReader(path.open(newline="", encoding="utf-8"))) if path.exists() else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_HEADER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing + rows)


def append_iteration(text: str) -> None:
    path = OUT / "ITERATION_LOG.md"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + text.rstrip() + "\n")


def update_status(iteration: int, experiment: str, conclusion: str, next_action: str, artifacts: list[str]) -> None:
    path = OUT / "STATUS.json"
    status = load_json(path)
    status.update({
        "status": "RUNNING",
        "iteration": iteration,
        "current_hypothesis": conclusion,
        "last_completed_experiment": experiment,
        "best_supported_conclusion": "7/21 conditional final-selection contributions remain; parser-only mapping has now been tested separately.",
        "confirmed_rescues": 7,
        "control_regressions": 0,
        "holdout_evaluated": False,
        "next_action": next_action,
        "artifact_paths": sorted(set(status.get("artifact_paths", []) + artifacts)),
    })
    dump_json(path, status)


def record_kg_usage(iteration: int, count: int, experiment: str) -> None:
    path = OUT / "COST_LEDGER.json"
    ledger = load_json(path) if path.exists() else {}
    ledger["read_only_kg_queries"] = int(ledger.get("read_only_kg_queries", 0)) + int(count)
    ledger.setdefault("llm_calls", 0)
    ledger.setdefault("input_tokens", 0)
    ledger.setdefault("output_tokens", 0)
    ledger.setdefault("cost_usd", 0.0)
    ledger.setdefault("kg_query_log", []).append({
        "iteration": iteration, "experiment": experiment,
        "queries": int(count), "recorded_utc": datetime.now(timezone.utc).isoformat(),
    })
    dump_json(path, ledger)


def log_evidence() -> None:
    bundle = load_bundle()
    attribution = list(csv.DictReader((PRIOR_DIAG / "question_attribution.csv").open(newline="", encoding="utf-8")))
    rows: list[dict[str, Any]] = []
    for row in attribution:
        if int(row.get("tgar_hit", "0") or 0):
            continue
        rows.append({
            "iteration": 0,
            "question_key": row["question_key"],
            "hypothesis": row.get("primary_causal_class", "UNKNOWN"),
            "supporting_trace": row.get("first_divergence_stage", "UNKNOWN"),
            "confidence": row.get("confidence", "LOW"),
            "outcome": row.get("counterfactual_outcome", "NOT_TESTED"),
            "source_artifact": "outputs/experiments/fresh50_autonomous_causal_diagnosis/question_attribution.csv",
            "notes": row.get("unresolved_alternatives", ""),
        })
    add_ledger_rows(rows)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"--init", "--selector-probe", "--relation-exposure-audit", "--root-relation-replay", "--gold-bgp-binding-replay", "--root-path-candidate-replay", "--remaining-gold-path-replay", "--archived-path-divergence-audit", "--archived-path-direction-audit", "--inverse-relation-catalog", "--semantic-archived-path-audit"}:
        raise SystemExit("Usage: run_fresh50_continuous_agent.py --init|--selector-probe|--relation-exposure-audit|--root-relation-replay|--gold-bgp-binding-replay|--root-path-candidate-replay|--remaining-gold-path-replay|--archived-path-divergence-audit|--archived-path-direction-audit|--inverse-relation-catalog|--semantic-archived-path-audit")
    action = sys.argv[1]
    if action == "--init":
        outcome = bootstrap()
        if not outcome["resumed"]:
            log_evidence()
        print(json.dumps(outcome, indent=2, ensure_ascii=False))
        return

    if not (OUT / "STATUS.json").is_file():
        raise SystemExit("Initialize the continuous workspace first with --init")
    if action == "--selector-probe":
        result = strict_selector_probe()
        ledger_rows = []
        case_rows = list(csv.DictReader((OUT / "RESULTS/iter_001_id_schema_probe/strict_id_adapter_cases.csv").open(newline="", encoding="utf-8")))
        for row in case_rows:
            ledger_rows.append({
                "iteration": 1,
                "question_key": row["question_key"],
                "hypothesis": "H6_OUTPUT_SCHEMA_OR_LABEL_MAPPING",
                "supporting_trace": row["adapter_status"],
                "confidence": "DIRECT_DETERMINISTIC_REPLAY",
                "outcome": f"strict Hits@1={row['strict_id_adapter_hits_at_1']}; candidate membership={row['candidate_membership_valid']}",
                "source_artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_001_id_schema_probe/strict_id_adapter_cases.csv",
                "notes": "Gold IDs were scoring-only; resolver saw frozen question, candidate ID/name pairs, and archived answer strings.",
            })
        add_ledger_rows(ledger_rows)
        next_action = "Run archived relation-predicate exposure audit across 21 misses and five controls; keep all results descriptive unless active bindings align."
        append_iteration(
            f"## Iteration 1 — strict candidate-ID adapter — {datetime.now(timezone.utc).isoformat()}\n\n"
            f"- Command: `{result['command']}`\n- Results: {result['rescued_failures']}/7 failures rescued; {result['control_regressions']}/5 controls regressed; "
            f"{result['invalid_or_unmapped_outputs']} unmapped/invalid outputs; {result['ambiguous_label_outputs']} ambiguous labels.\n"
            "- Decision: NO-GO for schema/label-to-ID conversion as an accuracy intervention; this does not reject a new semantic ID-constrained selector.\n"
            "- Leakage/cost: gold scoring file was separate; adapter used no gold; 0 LLM/KG calls; 0 added tokens.\n"
            f"- Input SHA256: `{result['adapter_input_sha256']}`; scoring-only SHA256: `{result['scoring_labels_sha256']}`.\n"
        )
        update_status(1, "strict candidate-ID adapter on 7 failures + 5 controls", "Schema conversion may fix invalid formatting only; does it alter wrong semantic selections?", next_action, [
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_001_id_schema_probe/results.json",
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_001_id_schema_probe/strict_id_adapter_cases.csv",
        ])
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if action == "--root-relation-replay":
        result = root_relation_replay()
        case_rows = list(csv.DictReader((OUT / "RESULTS/iter_003_root_relation_replay/root_relation_case_audit.csv").open(newline="", encoding="utf-8")))
        add_ledger_rows([{
            "iteration": 3,
            "question_key": row["question_key"],
            "hypothesis": "H2_ALIGNED_FIRST_HOP_RELATION_SELECTION",
            "supporting_trace": row["root_edge_classification"],
            "confidence": "READ_ONLY_EXACT_SOURCE_AND_DIRECTION_REPLAY",
            "outcome": f"gold_root={row['gold_root_relations']}; candidates={row['gold_root_relations_in_replayed_candidates']}; selected={row['gold_root_relations_selected_by_first_prune']}",
            "source_artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_003_root_relation_replay/root_relation_case_audit.csv",
            "notes": row["causal_interpretation"],
        } for row in case_rows])
        omitted = result["gold_root_edge_available_but_omitted_failures"]
        next_action = (
            "Inspect exact query/prompt source-binding maps for the aligned first-hop omissions, then attempt one minimal plan-only counterfactual using archived candidate pools through the existing executor only if the public runtime accepts fixed relation steps without paid LLM calls. Otherwise prepare the budget/architecture handoff."
            if omitted else
            "Continue with the earlier-only entity cases and compare exact saved entity-search queries against gold binding edges; preserve no-cause conclusion for root relation selection unless an aligned omission appears."
        )
        append_iteration(
            f"## Iteration 3 — read-only aligned root relation replay — {datetime.now(timezone.utc).isoformat()}\n\n"
            f"- Command: `{result['command']}`\n- Exact candidate queries: {result['unique_queries_planned']}; attempted {result['new_queries_attempted']}; cached {result['query_cache_hits']}.\n"
            f"- Aligned first-hop root-edge candidates available but omitted: {omitted} failures / {result['gold_root_edge_available_but_omitted_controls']} controls.\n"
            "- Interpretation: first-hop relation-selection evidence is counted only when the replayed query source and direction match a concrete gold triple. This is not yet an answer-changing plan-only counterfactual.\n"
            "- Cost: 0 LLM calls; all KG operations were read-only, bounded by a 40-query cap, 8s timeout and 8 MiB response cap.\n"
        )
        update_status(3, "read-only exact initial relation-candidate replay", "Did the first CoR relation-prune omit a gold-required relation at the exact topic source and direction?", next_action, [
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_003_root_relation_replay/results.json",
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_003_root_relation_replay/root_relation_case_audit.csv",
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_003_root_relation_replay/relation_query_replay.csv",
        ])
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if action == "--gold-bgp-binding-replay":
        result = gold_bgp_binding_replay()
        record_kg_usage(4, result["read_only_kg_queries"], "diagnostic-only staged BGP binding replay")
        case_rows = list(csv.DictReader((OUT / "RESULTS/iter_004_gold_bgp_binding_replay/bgp_binding_case_audit.csv").open(newline="", encoding="utf-8")))
        add_ledger_rows([{
            "iteration": 4,
            "question_key": row["question_key"],
            "hypothesis": "H3_SHARED_BINDING_EXECUTOR_EXPRESSIVITY",
            "supporting_trace": row["adapter_status"],
            "confidence": "ORACLE_DIAGNOSTIC_READ_ONLY_BGP_EXECUTION",
            "outcome": f"BGP answer covered={row['gold_answer_covered_by_bgp']}; gold filters applied={row['gold_filters_applied']}",
            "source_artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_004_gold_bgp_binding_replay/bgp_binding_case_audit.csv",
            "notes": row["interpretation_limit"],
        } for row in case_rows])
        append_iteration(
            f"## Iteration 4 — diagnostic staged BGP binding replay — {datetime.now(timezone.utc).isoformat()}\n\n"
            f"- Command: `{result['command']}`\n- Cohort: {result['cohort']['aligned_root_omission_failures']} aligned-omission failure + {result['cohort']['aligned_root_omission_controls']} control.\n"
            f"- Gold answers covered by diagnostic BGP adapter: {result['cases_with_bgp_gold_answer_covered']}/{result['case_count']}; read-only KG queries: {result['read_only_kg_queries']}.\n"
            "- Interpretation: the adapter receives gold BGPs and is oracle diagnostic only; coverage is not official accuracy and does not attribute the original miss to one step. Gold SPARQL filters were not applied.\n"
            "- Cost: 0 LLM calls; read-only SPARQL only; production source unchanged.\n"
        )
        next_action = (
            "Inspect adapter traces for join-binding survival and compare the failure/control. Then finalize the no-go boundary: parser-only selection rejected, relation omission not sufficient, exact BGP diagnostic capability bounded; any deployable selector conclusion requires the authorized 12-call non-oracle pilot and independent cohort."
        )
        update_status(4, "diagnostic-only staged BGP execution with intermediate binding preservation", "Does a binding-preserving read-only diagnostic adapter cover the answer in the matched failed and control graphs?", next_action, [
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_004_gold_bgp_binding_replay/results.json",
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_004_gold_bgp_binding_replay/bgp_binding_case_audit.csv",
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_004_gold_bgp_binding_replay/read_only_query_trace.jsonl",
        ])
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if action == "--archived-path-divergence-audit":
        result = archived_path_divergence_audit()
        case_rows = list(csv.DictReader((OUT / "RESULTS/iter_007_archived_path_divergence_audit/archived_gold_path_alignment.csv").open(newline="", encoding="utf-8")))
        add_ledger_rows([{
            "iteration": 7,
            "question_key": row["question_key"],
            "hypothesis": "H5_ARCHIVED_ROUTE_GOLD_PATH_DIVERGENCE",
            "supporting_trace": f"max exact gold path prefix={row['max_archived_gold_path_prefix']}/{row['gold_path_length']}",
            "confidence": "OFFLINE_TRACE_ALIGNMENT_PLUS_ORACLE_PATH_REPLAY",
            "outcome": row["divergence_status"],
            "source_artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_007_archived_path_divergence_audit/archived_gold_path_alignment.csv",
            "notes": row["interpretation"],
        } for row in case_rows])
        append_iteration(
            f"## Iteration 7 — archived TGAR path alignment — {datetime.now(timezone.utc).isoformat()}\n\n"
            f"- Command: `{result['command']}`\n- Scope: {result['cohort']['remaining_TGAR_failures']} unresolved TGAR misses + {result['cohort']['both_correct_controls']} controls.\n"
            f"- Oracle gold-path candidate coverage: {result['oracle_path_answer_coverage']}/{len(case_rows)}; failures with no exact gold-path prefix: {result['failures_with_no_gold_path_prefix_executed']}; full gold path executed: {result['failures_with_full_gold_path_executed']}.\n"
            f"- Gold first relation appeared in first prune for {result['failures_first_gold_relation_exposed_in_prompt']}/{result['cohort']['remaining_TGAR_failures']} failures; predicate was selected for {result['failures_first_gold_relation_predicate_selected']}.\n"
            "- Interpretation remains trace-level; one relation-only intervention and paid selector validation have not been run. No LLM/KG calls.\n"
        )
        update_status(7, "offline archived TGAR path alignment against gold root-to-answer paths", "Do archived failures diverge from gold paths even though the unchanged CoR query builder retrieves gold answers on oracle paths?", "Complete per-case interpretation, source/hash verification, and the final budget handoff for the unrun 12-call non-oracle selector pilot.", [
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_007_archived_path_divergence_audit/results.json",
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_007_archived_path_divergence_audit/archived_gold_path_alignment.csv",
        ])
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if action == "--archived-path-direction-audit":
        result = archived_path_direction_audit()
        case_rows = list(csv.DictReader((OUT / "RESULTS/iter_008_archived_path_direction_audit/archived_gold_path_direction_alignment.csv").open(newline="", encoding="utf-8")))
        add_ledger_rows([{
            "iteration": 8,
            "question_key": row["question_key"],
            "hypothesis": "H5_DIRECTION_ALIGNED_RELATION_SELECTION",
            "supporting_trace": f"offered={row['gold_first_relation_in_matching_direction_candidates']}; selected={row['gold_first_relation_selected_in_matching_direction']}",
            "confidence": "DIRECTION_RECONSTRUCTED_FROM_FROZEN_PROMPT_AND_PARSER",
            "outcome": f"max gold path prefix={row['max_archived_gold_path_prefix']}/{row['gold_path_length']}",
            "source_artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_008_archived_path_direction_audit/archived_gold_path_direction_alignment.csv",
            "notes": "Trace-level relation-selection evidence; not an answer-changing intervention.",
        } for row in case_rows])
        append_iteration(
            f"## Iteration 8 — direction-aware root-prune audit — {datetime.now(timezone.utc).isoformat()}\n\n"
            f"- Command: `{result['command']}`\n- Correct-direction gold root relation offered: {result['failures_with_gold_first_relation_offered_in_correct_direction']}/{result['cohort']['remaining_TGAR_failures']} failures.\n"
            f"- Correct-direction relation selected: {result['failures_with_gold_first_relation_selected_in_correct_direction']}/{result['cohort']['remaining_TGAR_failures']} failures; control offers {result['controls_with_gold_first_relation_offered_in_correct_direction']}/{result['cohort']['both_correct_controls']} and selects {result['controls_with_gold_first_relation_selected_in_correct_direction']}.\n"
            "- Iteration 7's prompt-presence summary was not direction-aware and is superseded by this audit. No new LLM/KG calls.\n"
        )
        update_status(8, "direction-aware archived root relation-prune alignment", "How often did root pruning choose the correct relation in the correct direction on unresolved misses versus controls?", "Use the corrected counts and prefix traces to set the final research boundary; the paid non-oracle selector validation remains blocked by unset project budget.", [
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_008_archived_path_direction_audit/results.json",
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_008_archived_path_direction_audit/archived_gold_path_direction_alignment.csv",
        ])
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if action == "--inverse-relation-catalog":
        result = lookup_inverse_relation_catalog()
        record_kg_usage(9, result["read_only_kg_queries"], "lookup Freebase explicit inverse-property metadata")
        update_status(9, "lookup of explicitly declared inverse properties for audited relations", "Do exact predicate mismatches in archived routes reflect semantically equivalent inverse properties?", "Recompute route alignment with explicit inverse-property metadata, then write the decision handoff for the unrun paid selector pilot.", [
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_009_inverse_relation_catalog/relation_inverse_map.json",
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_009_inverse_relation_catalog/query_metadata.json",
        ])
        append_iteration(
            f"## Iteration 9 — explicit Freebase inverse-property lookup — {datetime.now(timezone.utc).isoformat()}\n\n"
            f"- Command: `{result['command']}`\n- Unique route predicates: {result['unique_relation_ids']}; explicitly declared inverse pairs: {result['explicit_inverse_pairs']}; query status: {result['query_status']}.\n"
            "- This lookup fixes a limitation in the exact-ID path audit: different Freebase predicates may encode the same edge in reverse direction.\n"
            "- Cost: one bounded read-only KG query; 0 LLM calls.\n"
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if action == "--semantic-archived-path-audit":
        result = semantic_archived_path_audit()
        case_rows = list(csv.DictReader((OUT / "RESULTS/iter_010_semantic_archived_path_audit/semantic_archived_path_alignment.csv").open(newline="", encoding="utf-8")))
        add_ledger_rows([{
            "iteration": 10,
            "question_key": row["question_key"],
            "hypothesis": "H5_SEMANTIC_GOLD_PATH_DIVERGENCE",
            "supporting_trace": f"semantic prefix={row['max_archived_semantic_gold_prefix']}/{row['gold_path_length']}",
            "confidence": "ARCHIVED_TRACE_PLUS_EXPLICIT_FREEBASE_INVERSE_MAP",
            "outcome": row["semantic_divergence_status"],
            "source_artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_010_semantic_archived_path_audit/semantic_archived_path_alignment.csv",
            "notes": row["interpretation"],
        } for row in case_rows])
        append_iteration(
            f"## Iteration 10 — semantic archived path alignment — {datetime.now(timezone.utc).isoformat()}\n\n"
            f"- Command: `{result['command']}`\n- Oracle path answer coverage: {result['oracle_path_gold_answer_coverage']}/{len(case_rows)}; full semantic gold paths executed by TGAR: {result['failures_with_full_semantic_gold_path_executed']}/{result['cohort']['remaining_TGAR_failures']} failures.\n"
            f"- Failures with no semantic gold prefix: {result['failures_with_semantic_prefix_zero']}; first gold relation semantically offered: {result['failures_with_gold_first_relation_semantically_offered']}; selected: {result['failures_with_gold_first_relation_semantically_selected']}.\n"
            "- Inverse relation equivalence is based only on explicit Freebase metadata; no LLM/KG calls in this alignment pass.\n"
        )
        update_status(10, "semantic archived TGAR path alignment with explicit inverse-property equivalence", "After accounting for declared inverse properties, how often do archived TGAR routes execute the gold path while the same CoR query builder can retrieve the answer under oracle routes?", "Complete the final case-level interpretation and hand off the still-unvalidated paid selector pilot with a precise 12-call budget request.", [
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_010_semantic_archived_path_audit/results.json",
            "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_010_semantic_archived_path_audit/semantic_archived_path_alignment.csv",
        ])
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if action in {"--root-path-candidate-replay", "--remaining-gold-path-replay"}:
        all_remaining = action == "--remaining-gold-path-replay"
        result = root_path_candidate_replay(all_remaining=all_remaining)
        iteration = result["iteration"]
        relative = f"outputs/experiments/fresh50_continuous_agent/RESULTS/iter_{iteration:03d}_root_path_candidate_replay"
        record_kg_usage(iteration, result["read_only_kg_queries"], "oracle root-path candidate replay through existing CoR query builder")
        case_rows = list(csv.DictReader((ROOT / relative / "path_candidate_case_audit.csv").open(newline="", encoding="utf-8")))
        add_ledger_rows([{
            "iteration": iteration,
            "question_key": row["question_key"],
            "hypothesis": "H4_GOLD_RELATION_PATH_CANDIDATE_RECALL",
            "supporting_trace": row["path_status"],
            "confidence": "ORACLE_PATH_READ_ONLY_BACKEND_REPLAY",
            "outcome": f"gold answer in path candidates={row['gold_answer_in_final_gold_path_candidates']}; archived candidate observed={row['gold_answer_in_archived_entity_search_candidates']}",
            "source_artifact": relative + "/path_candidate_case_audit.csv",
            "notes": row["interpretation"],
        } for row in case_rows])
        append_iteration(
            f"## Iteration 5 — oracle path candidate replay — {datetime.now(timezone.utc).isoformat()}\n\n"
            f"- Command: `{result['command']}`\n- Gold answer present in path-prefix final candidates: {result['cases_with_gold_answer_in_path_candidates']}/{len(case_rows)}; read-only queries: {result['read_only_kg_queries']}.\n"
            "- The unchanged CoR entity-search query builder received the gold relation sequence; side constraints, relation selection, LLM reasoning and final filtering were not tested.\n"
            "- Decision value: candidate coverage under oracle routing distinguishes path-query availability from the archived route; it does not establish end-to-end causality.\n"
            "- Cost: 0 LLM calls; read-only SPARQL only; production source unchanged.\n"
        )
        update_status(iteration, "oracle gold-path candidate replay through the unchanged CoR entity-search builder", "Across unresolved failures, can the existing path query builder retrieve gold answers under oracle relation paths?", "Assess case-level path coverage. Then finalize a constrained-budget handoff because the non-oracle selector pilot has no explicitly configured paid call/cost ceiling.", [
            relative + "/results.json",
            relative + "/path_candidate_case_audit.csv",
            relative + "/path_prefix_queries.csv",
        ])
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    result = relation_exposure_audit()
    evidence_rows = []
    case_rows = list(csv.DictReader((OUT / "RESULTS/iter_002_relation_exposure_audit/gold_relation_exposure.csv").open(newline="", encoding="utf-8")))
    for row in case_rows:
        evidence_rows.append({
            "iteration": 2,
            "question_key": row["question_key"],
            "hypothesis": "H2_RELATION_CANDIDATE_EXPOSURE",
            "supporting_trace": f"{row['relation_prune_event_count']} relation_prune prompts; exposed={row['gold_predicates_exposed_in_any_archived_relation_prompt']}",
            "confidence": "DESCRIPTIVE_NO_ACTIVE_BINDING_ALIGNMENT",
            "outcome": f"selected_gold_predicates={row['gold_predicates_selected_in_any_archived_relation_prune']}",
            "source_artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_002_relation_exposure_audit/gold_relation_exposure.csv",
            "notes": row["limitation"],
        })
    add_ledger_rows(evidence_rows)
    next_action = "Check local read-only SPARQL endpoint availability; if reachable, replay only bounded archived relation-search queries for the highest-information failures and controls, then align candidate IDs with stored prune prompts."
    append_iteration(
        f"## Iteration 2 — archived relation predicate exposure — {datetime.now(timezone.utc).isoformat()}\n\n"
        f"- Command: `{result['command']}`\n- Scope: {result['cohort']['TGAR_failures']} TGAR misses + {result['cohort']['both_correct_controls']} controls.\n"
        f"- Outcome: some gold predicates are visible somewhere in traces, but active state alignment is unavailable; causal claim count remains 0.\n"
        f"- Decision: exposure alone cannot distinguish wrong relation ranking from a wrong active entity. Next test is bounded replay of saved relation-search SPARQL only if endpoint is up.\n"
        "- API calls / KG queries: 0 / 0.\n"
    )
    update_status(2, "archived relation-predicate exposure audit", "Gold predicate exposure in unaligned prompts is descriptive, not a relation-ranking cause.", next_action, [
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_002_relation_exposure_audit/results.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_002_relation_exposure_audit/gold_relation_exposure.csv",
    ])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

