"""Freeze a fresh Iteration 29 cohort and parser-fixed manifest, offline only."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
from pathlib import Path
from typing import Any

from scripts.diagnostics.iter_029_terminal_selector.runner_attempt2 import canonical_json, sha256_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_028_terminal_boundary_repair"
SOURCE = ROOT / "data/interim/grailqa_dev.jsonl"
OLD_COHORT = RESULTS / "NEXT_INFERENCE_COHORT_NON_GOLD.jsonl"
OLD_RUN = RESULTS / "RUNS/iter29_terminal_candidate_selector"
OUT_COHORT = RESULTS / "NEXT_INFERENCE_COHORT_ATTEMPT2_NON_GOLD.jsonl"
OUT_MANIFEST = RESULTS / "NEXT_MANIFEST_DRAFT_ATTEMPT2.json"
PROMPT_PATH = RESULTS / "TERMINAL_SELECTOR_PROMPT_V1.txt"
N = 800
SEED = 2902


def _load_preparation_helpers() -> Any:
    path = ROOT / "scripts/diagnostics/iter_028_terminal_boundary_repair/prepare_next_protocol.py"
    spec = importlib.util.spec_from_file_location("iter28_prepare_helpers", path)
    if not spec or not spec.loader:
        raise RuntimeError("could not load frozen cohort normalization helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare() -> dict[str, Any]:
    helpers = _load_preparation_helpers()
    prior_ids, prior_texts, audited = helpers.gather_prior_exposure()

    # The preceding run's output directory is intentionally excluded by the
    # general audit because it lives under this protocol's output root. Add
    # its actual 100 exposed cases explicitly before sampling this replacement.
    paired_path = OLD_RUN / "PAIRED_CASES_NON_GOLD.jsonl"
    paired_rows = [json.loads(line) for line in paired_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(paired_rows) != 100:
        raise RuntimeError(f"expected exactly 100 exposed prior cases, found {len(paired_rows)}")
    exposed_ids: set[str] = set()
    exposed_texts: set[str] = set()
    for row in paired_rows:
        key = str(row.get("question_key", ""))
        if key.startswith("grailqa::"):
            exposed_ids.add(key.split("::", 1)[1])
        question = row.get("question")
        if isinstance(question, str):
            normalized = helpers.normalize_question(question)
            if normalized:
                exposed_texts.add(normalized)
    prior_ids.update(exposed_ids)
    prior_texts.update(exposed_texts)
    audited.append({"path": str(paired_path.relative_to(ROOT)), "sha256": sha256_file(paired_path)})

    prior_norms = sorted(prior_texts)
    prior_token_sets = [set(text.split()) for text in prior_norms]
    source_bytes = SOURCE.read_bytes()
    source_sha = sha256_bytes(source_bytes)
    pool: list[dict[str, Any]] = []
    exact_excluded = 0
    near_excluded = 0
    for line in source_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        # Read only fields permitted in the live, gold-free inference cohort.
        row = {
            "id": str(raw.get("id", "")).strip(),
            "question": raw.get("question", ""),
            "topic_entities": raw.get("topic_entities", []),
            "topic_entity_names": raw.get("topic_entity_names", []),
            "split": raw.get("split", ""),
        }
        if row["split"] != "dev" or not row["id"] or not isinstance(row["question"], str) or not row["question"].strip():
            continue
        if not isinstance(row["topic_entities"], list) or not row["topic_entities"]:
            continue
        if not isinstance(row["topic_entity_names"], list) or len(row["topic_entities"]) != len(row["topic_entity_names"]):
            continue
        if row["id"] in prior_ids:
            continue
        row["norm"] = helpers.normalize_question(row["question"])
        row["tokens"] = set(row["norm"].split())
        if row["norm"] in prior_texts:
            exact_excluded += 1
            continue
        if any(
            helpers._is_near_duplicate(row["tokens"], row["norm"], tokens, text)
            for text, tokens in zip(prior_norms, prior_token_sets)
        ):
            near_excluded += 1
            continue
        pool.append(row)

    pool.sort(key=lambda row: row["id"])
    random.Random(SEED).shuffle(pool)
    selected: list[dict[str, Any]] = []
    for row in pool:
        if any(
            helpers._is_near_duplicate(row["tokens"], row["norm"], old["tokens"], old["norm"])
            for old in selected
        ):
            continue
        selected.append(row)
        if len(selected) == N:
            break
    if len(selected) != N:
        raise RuntimeError(f"only {len(selected)} eligible untouched rows remain; expected {N}")

    cohort: list[dict[str, Any]] = []
    for index, row in enumerate(selected, 1):
        safe = {
            "case_id": f"iter29_attempt2_case_{index:04d}",
            "question_key": f"grailqa::{row['id']}",
            "dataset": "grailqa",
            "split": "dev",
            "question": row["question"],
            "topic_entities": row["topic_entities"],
            "topic_entity_names": row["topic_entity_names"],
        }
        safe["question_sha256"] = sha256_bytes(safe["question"].encode("utf-8"))
        safe["safe_row_sha256"] = sha256_bytes(canonical_json(safe))
        cohort.append(safe)

    cohort_bytes = b"".join(canonical_json(row) + b"\n" for row in cohort)
    cohort_hash = sha256_bytes(cohort_bytes)
    keys = [row["question_key"] for row in cohort]
    helper_hashes = sorted(audited, key=lambda item: item["path"])
    manifest_path = RESULTS / "NEXT_MANIFEST_DRAFT.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = copy.deepcopy(manifest)
    manifest.update({
        "status": "FROZEN_AWAITING_EXPLICIT_AUTHORIZATION",
        "experiment": "terminal_candidate_selector_v1_paired_with_frozen_upstream_candidates_parser_fix_attempt2",
        "manifest_scope": "one fresh fixed 800-question one-shot run; one baseline pass per question; at most one terminal selector request only on the independently recomputed frozen gold-free trigger; no resume",
        "sampling_seed": SEED,
        "sampling_policy": "uniform pseudorandom order from valid GrailQA dev rows after prior-ID, exact-text and near-duplicate exclusions, including all 100 questions exposed in Iteration 29 attempt 1; select first 800",
        "question_keys": keys,
        "question_keys_sha256": sha256_bytes("\n".join(keys).encode("utf-8")),
        "inference_cohort_path": OUT_COHORT.relative_to(ROOT).as_posix(),
        "inference_cohort_sha256": cohort_hash,
        "per_question_safe_row_sha256": [row["safe_row_sha256"] for row in cohort],
        "source_dataset_sha256": source_sha,
        "run_directory": "RUNS/iter29_terminal_candidate_selector_attempt2",
        "consumption_marker_path": "ITER29_ATTEMPT2_AUTHORIZATION_CONSUMPTION.json",
    })
    manifest["prior_exposure_audit"] = {
        "audited_artifact_count": len(helper_hashes),
        "audited_artifacts_sha256": sha256_bytes(canonical_json(helper_hashes)),
        "discovered_prior_grailqa_question_id_count": len(prior_ids),
        "discovered_prior_grailqa_question_ids_sha256": sha256_bytes("\n".join(sorted(prior_ids)).encode("utf-8")),
        "prior_question_text_count": len(prior_texts),
        "exact_text_exclusion_count": exact_excluded,
        "near_duplicate_exclusion_count": near_excluded,
        "question_ids_excluded_by_prior_exposure": sorted(prior_ids),
        "attempt1_exposed_question_count": len(exposed_ids),
        "attempt1_exposed_question_keys_sha256": sha256_bytes("\n".join(sorted(f"grailqa::{value}" for value in exposed_ids)).encode("utf-8")),
        "attempt1_paired_artifact_path": str(paired_path.relative_to(ROOT)),
        "attempt1_paired_artifact_sha256": sha256_file(paired_path),
        "limitations": "Audit covers local experiment artifacts and explicitly includes all 100 Iteration 29 attempt-1 paired cases; it cannot prove that no question was exposed outside these artifacts.",
    }

    implementation = manifest["implementation"]
    old_map = dict(implementation.get("source_files_sha256", {}))
    old_map.pop("scripts/diagnostics/iter_029_terminal_selector/runner.py", None)
    old_map.pop("scripts/diagnostics/iter_028_terminal_boundary_repair/scoring_gate.py", None)
    source_map: dict[str, str] = {}
    for relative in sorted(old_map):
        source_path = ROOT / relative
        if source_path.is_file():
            source_map[relative] = sha256_file(source_path)
    additional_sources = [
        "chain_of_relations/methods/cor/agent.py",
        "scripts/diagnostics/iter_029_terminal_selector/path_parser_attempt2.py",
        "scripts/diagnostics/iter_029_terminal_selector/runner_attempt2.py",
        "scripts/diagnostics/iter_029_terminal_selector/scoring_gate_attempt2.py",
        "scripts/diagnostics/iter_029_terminal_selector/prepare_attempt2.py",
        "scripts/diagnostics/iter_029_terminal_selector/test_runner_attempt2.py",
    ]
    for relative in additional_sources:
        path = ROOT / relative
        if path.is_file():
            source_map[relative] = sha256_file(path)
    implementation.update({
        "runner_path": "scripts/diagnostics/iter_029_terminal_selector/runner_attempt2.py",
        "runner_sha256": source_map["scripts/diagnostics/iter_029_terminal_selector/runner_attempt2.py"],
        "external_gate_path": "scripts/diagnostics/iter_029_terminal_selector/scoring_gate_attempt2.py",
        "external_gate_sha256": source_map["scripts/diagnostics/iter_029_terminal_selector/scoring_gate_attempt2.py"],
        "offline_test_command": "PYTHONPATH=src:. .venv/bin/python -m unittest scripts.diagnostics.iter_029_terminal_selector.test_runner_attempt2 scripts.diagnostics.iter_028_terminal_boundary_repair.test_iter28_terminal_boundary -v",
        "offline_tests_passed": None,
        "offline_tests_total": None,
        "python_compile_passed": None,
        "source_files_sha256": source_map,
        "source_hash_scope": "Frozen production/runtime dependencies plus attempt-2 runner, independent parser, independent gate, cohort preparer, and tests. No answer labels, scorer outputs or posthoc quarantine contents are included.",
        "parser_contract": "Independently parse predicate IDs from CoR triple lines; exclude candidate.target; preserve path order and deduplicate; runner and gate use separate implementations and gate recomputes IDs from raw serialized path.",
    })
    manifest["gold_scoring"].update({
        "validator_path": "scripts/diagnostics/iter_029_terminal_selector/scoring_gate_attempt2.py",
        "posthoc_iteration27_quarantine_used": False,
    })
    manifest["api_caps"].update({
        "price_note": "Official OpenAI GPT-5.6 Luna model page checked 2026-11-09: USD 0.20 per million uncached input tokens and USD 1.20 per million output tokens. Cached input pricing is excluded from the conservative full-cap reservation. Source: https://developers.openai.com/api/docs/models/gpt-5.6-luna",
    })
    manifest["authorization"] = {
        "state": "AUTHORIZATION_REQUIRED",
        "authorized": False,
        "required_confirmation": "A fresh explicit user authorization must name this attempt-2 manifest SHA-256 and accept the exact API, cost, SPARQL, timeout, wall-time, zero-retry and no-resume caps. Attempt-1 authorization is consumed and does not roll over.",
        "prior_authorizations_roll_over": False,
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json({key: value for key, value in manifest.items() if key != "manifest_sha256"}))

    for path, data in ((OUT_COHORT, cohort_bytes), (OUT_MANIFEST, (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))):
        if path.exists():
            if path.read_bytes() != data:
                raise RuntimeError(f"refusing to overwrite different frozen attempt-2 artifact: {path}")
        else:
            path.write_bytes(data)
    return {
        "status": manifest["status"],
        "cases": len(cohort),
        "attempt1_exposed_excluded": len(exposed_ids),
        "eligible_pool_after_exclusions": len(pool),
        "exact_text_excluded": exact_excluded,
        "near_duplicate_excluded": near_excluded,
        "cohort_sha256": cohort_hash,
        "manifest_sha256": manifest["manifest_sha256"],
        "api_caps": manifest["api_caps"],
        "sparql_caps": manifest["sparql_caps"],
        "time_caps": manifest["time_caps"],
    }


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, sort_keys=True, indent=2))
