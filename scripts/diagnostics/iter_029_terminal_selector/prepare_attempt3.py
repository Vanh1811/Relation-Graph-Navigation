"""Freeze a fresh Iteration 29 attempt-3 cohort and manifest, offline only."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.diagnostics.iter_029_terminal_selector.runner_attempt3 import canonical_json, sha256_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_028_terminal_boundary_repair"
SOURCE = ROOT / "data/interim/grailqa_dev.jsonl"
ATTEMPT1_RUN = RESULTS / "RUNS/iter29_terminal_candidate_selector"
ATTEMPT2_RUN = RESULTS / "RUNS/iter29_terminal_candidate_selector_attempt2"
ATTEMPT1_COHORT = RESULTS / "NEXT_INFERENCE_COHORT_NON_GOLD.jsonl"
ATTEMPT2_COHORT = RESULTS / "NEXT_INFERENCE_COHORT_ATTEMPT2_NON_GOLD.jsonl"
OUT_COHORT = RESULTS / "NEXT_INFERENCE_COHORT_ATTEMPT3_NON_GOLD.jsonl"
OUT_MANIFEST = RESULTS / "NEXT_MANIFEST_DRAFT_ATTEMPT3.json"
AUTHORIZATION_PATH = RESULTS / "ITER29_ATTEMPT3_AUTHORIZATION_RECORD.json"
PROMPT_PATH = RESULTS / "TERMINAL_SELECTOR_PROMPT_V1.txt"
N = 800
SEED = 2903
API_REQUEST_CAP = 3300
SPARQL_ATTEMPT_CAP = 6500
GLOBAL_SPEND_CEILING_USD = 10.0


def _load_preparation_helpers() -> Any:
    path = ROOT / "scripts/diagnostics/iter_028_terminal_boundary_repair/prepare_next_protocol.py"
    spec = importlib.util.spec_from_file_location("iter28_prepare_helpers", path)
    if not spec or not spec.loader:
        raise RuntimeError("could not load frozen cohort normalization helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_immutable(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite a different frozen attempt-3 artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _offline_checks() -> dict[str, Any]:
    tests = [
        "scripts.diagnostics.iter_029_terminal_selector.test_runner_attempt3",
        "scripts.diagnostics.iter_028_terminal_boundary_repair.test_iter28_terminal_boundary",
    ]
    command = [str(ROOT / ".venv/bin/python"), "-m", "unittest", *tests, "-v"]
    result = subprocess.run(
        command, cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": "src:."},
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode:
        raise RuntimeError(f"attempt-3 offline tests failed:\n{result.stdout[-5000:]}\n{result.stderr[-5000:]}")
    compile_paths = [
        ROOT / "scripts/diagnostics/iter_029_terminal_selector/runner_attempt3.py",
        ROOT / "scripts/diagnostics/iter_029_terminal_selector/scoring_gate_attempt3.py",
        ROOT / "scripts/diagnostics/iter_029_terminal_selector/prepare_attempt3.py",
        ROOT / "scripts/diagnostics/iter_029_terminal_selector/test_runner_attempt3.py",
    ]
    for path in compile_paths:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return {
        "passed": True,
        "command": "PYTHONPATH=src:. .venv/bin/python -m unittest " + " ".join(tests) + " -v",
        "test_output_tail": (result.stdout + result.stderr)[-3000:],
    }


def prepare() -> dict[str, Any]:
    offline_checks = _offline_checks()
    helpers = _load_preparation_helpers()
    prior_ids, prior_texts, audited = helpers.gather_prior_exposure()

    # This experiment's earlier run artifacts are deliberately skipped by the
    # general exposure scanner. Add every attempted question, including the
    # unpaired request-failure case 633, from both attempts explicitly.
    exposure_sets: list[dict[str, Any]] = []
    exposed_ids: set[str] = set()
    exposed_texts: set[str] = set()
    attempt_exposure_counts: dict[str, int] = {}
    for label, run_dir, cohort_path, expected_paired, expected_attempted in (
        ("attempt1", ATTEMPT1_RUN, ATTEMPT1_COHORT, 100, 100),
        ("attempt2", ATTEMPT2_RUN, ATTEMPT2_COHORT, 632, 633),
    ):
        paired_path = run_dir / "PAIRED_CASES_NON_GOLD.jsonl"
        events_path = run_dir / "CASE_STATE_EVENTS.jsonl"
        paired_rows = [json.loads(line) for line in paired_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        state_rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(paired_rows) != expected_paired:
            raise RuntimeError(f"{label}: expected {expected_paired} durable pairs, found {len(paired_rows)}")
        attempted_keys = {
            str(event.get("question_key", "")) for event in state_rows
            if event.get("event") == "CASE_ATTEMPTED" and str(event.get("question_key", "")).startswith("grailqa::")
        }
        attempted_keys.update(
            str(row.get("question_key", "")) for row in paired_rows
            if str(row.get("question_key", "")).startswith("grailqa::")
        )
        if len(attempted_keys) != expected_attempted:
            raise RuntimeError(f"{label}: expected {expected_attempted} distinct exposed questions, found {len(attempted_keys)}")
        cohort_rows = [json.loads(line) for line in cohort_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        cohort_by_key = {str(row.get("question_key", "")): row for row in cohort_rows}
        missing = sorted(attempted_keys - set(cohort_by_key))
        if missing:
            raise RuntimeError(f"{label}: exposed identities missing from frozen source cohort: {missing[:3]}")
        for key in attempted_keys:
            exposed_ids.add(key.split("::", 1)[1])
            question = cohort_by_key[key].get("question")
            if isinstance(question, str):
                normalized = helpers.normalize_question(question)
                if normalized:
                    exposed_texts.add(normalized)
        attempt_exposure_counts[label] = len(attempted_keys)
        exposure_sets.append({
            "attempt": label,
            "attempted_question_count": len(attempted_keys),
            "attempted_question_keys_sha256": sha256_bytes("\n".join(sorted(attempted_keys)).encode("utf-8")),
            "paired_question_count": len(paired_rows),
            "paired_artifact": str(paired_path.relative_to(ROOT)),
            "paired_artifact_sha256": sha256_file(paired_path),
            "state_events_artifact": str(events_path.relative_to(ROOT)),
            "state_events_artifact_sha256": sha256_file(events_path),
            "source_cohort_artifact": str(cohort_path.relative_to(ROOT)),
            "source_cohort_artifact_sha256": sha256_file(cohort_path),
        })
        for extra in (run_dir / "RUN_INTEGRITY.json", run_dir / "RUN_METADATA.json", run_dir / "REQUEST_EVENTS.jsonl"):
            if extra.is_file():
                audited.append({"path": str(extra.relative_to(ROOT)), "sha256": sha256_file(extra)})
        audited.extend({"path": item, "sha256": digest} for item, digest in (
            (str(paired_path.relative_to(ROOT)), sha256_file(paired_path)),
            (str(events_path.relative_to(ROOT)), sha256_file(events_path)),
            (str(cohort_path.relative_to(ROOT)), sha256_file(cohort_path)),
        ))
    prior_ids.update(exposed_ids)
    prior_texts.update(exposed_texts)

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
            "case_id": f"iter29_attempt3_case_{index:04d}",
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
        "status": "FROZEN_WITH_STANDING_USER_AUTHORIZATION",
        "experiment": "terminal_candidate_selector_v1_paired_with_frozen_upstream_candidates_parser_fix_attempt3_request_failure_policy",
        "manifest_scope": "one fresh fixed 800-question one-shot run; one baseline pass per question; at most one terminal selector request only on the independently recomputed frozen gold-free trigger; no retries and no resume",
        "sampling_seed": SEED,
        "sampling_policy": "uniform pseudorandom order from valid GrailQA dev rows after prior-ID, exact-text and near-duplicate exclusions, including all 100 questions exposed in attempt 1 and all 633 attempted questions from attempt 2, including its unpaired HTTP-error case; select first 800",
        "question_keys": keys,
        "question_keys_sha256": sha256_bytes("\n".join(keys).encode("utf-8")),
        "inference_cohort_path": OUT_COHORT.relative_to(ROOT).as_posix(),
        "inference_cohort_sha256": cohort_hash,
        "per_question_safe_row_sha256": [row["safe_row_sha256"] for row in cohort],
        "source_dataset_sha256": source_sha,
        "run_directory": "RUNS/iter29_terminal_candidate_selector_attempt3",
        "consumption_marker_path": "ITER29_ATTEMPT3_AUTHORIZATION_CONSUMPTION.json",
        "authorization_record_path": AUTHORIZATION_PATH.relative_to(ROOT).as_posix(),
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
        "attempt_exposure_counts": attempt_exposure_counts,
        "attempt_exposure_lineage": exposure_sets,
        "attempt1_and_attempt2_exposed_question_count": len(exposed_ids),
        "attempt1_and_attempt2_exposed_question_keys_sha256": sha256_bytes("\n".join(sorted(f"grailqa::{value}" for value in exposed_ids)).encode("utf-8")),
        "limitations": "Audit covers available local experiment artifacts and explicitly includes every CASE_ATTEMPTED identity from Iteration 29 attempts 1 and 2, including attempt 2 case 633, which has no paired prediction. It cannot prove that no question was exposed outside these artifacts.",
    }

    implementation = manifest["implementation"]
    old_map = dict(implementation.get("source_files_sha256", {}))
    old_map.pop("scripts/diagnostics/iter_029_terminal_selector/runner.py", None)
    old_map.pop("scripts/diagnostics/iter_028_terminal_boundary_repair/scoring_gate.py", None)
    for relative in list(old_map):
        if any(token in relative for token in ("runner_attempt2.py", "scoring_gate_attempt2.py", "prepare_attempt2.py", "test_runner_attempt2.py")):
            old_map.pop(relative, None)
    source_map: dict[str, str] = {}
    for relative in sorted(old_map):
        source_path = ROOT / relative
        if source_path.is_file():
            source_map[relative] = sha256_file(source_path)
    additional_sources = [
        "chain_of_relations/methods/cor/agent.py",
        "scripts/diagnostics/iter_029_terminal_selector/path_parser_attempt2.py",
        "scripts/diagnostics/iter_029_terminal_selector/runner_attempt3.py",
        "scripts/diagnostics/iter_029_terminal_selector/scoring_gate_attempt3.py",
        "scripts/diagnostics/iter_029_terminal_selector/prepare_attempt3.py",
        "scripts/diagnostics/iter_029_terminal_selector/test_runner_attempt3.py",
    ]
    for relative in additional_sources:
        path = ROOT / relative
        if path.is_file():
            source_map[relative] = sha256_file(path)
    implementation.update({
        "runner_path": "scripts/diagnostics/iter_029_terminal_selector/runner_attempt3.py",
        "runner_sha256": source_map["scripts/diagnostics/iter_029_terminal_selector/runner_attempt3.py"],
        "external_gate_path": "scripts/diagnostics/iter_029_terminal_selector/scoring_gate_attempt3.py",
        "external_gate_sha256": source_map["scripts/diagnostics/iter_029_terminal_selector/scoring_gate_attempt3.py"],
        "offline_test_command": offline_checks["command"],
        "offline_tests_passed": offline_checks["passed"],
        "offline_test_output_tail": offline_checks["test_output_tail"],
        "python_compile_passed": True,
        "source_files_sha256": source_map,
        "source_hash_scope": "Frozen production/runtime dependencies plus attempt-3 runner, independent parser, independent gate, cohort preparer, and offline tests. No answer labels, scorer outputs or posthoc quarantine contents are included.",
        "parser_contract": "Independently parse predicate IDs from CoR triple lines; exclude candidate.target; preserve path order and deduplicate; runner and gate use separate implementations and gate recomputes IDs from raw serialized path.",
    })
    manifest["gold_scoring"].update({
        "validator_path": "scripts/diagnostics/iter_029_terminal_selector/scoring_gate_attempt3.py",
        "posthoc_iteration27_quarantine_used": False,
    })
    manifest["api_caps"].update({
        "price_note": "Official OpenAI GPT-5.6 Luna model page checked 2026-11-09: USD 0.20 per million uncached input tokens and USD 1.20 per million output tokens. Cached input pricing is excluded from the conservative full-cap reservation. Source: https://developers.openai.com/api/docs/models/gpt-5.6-luna",
    })
    input_price = float(manifest["api_caps"]["rate_assumption_input_uncached_usd_per_million"])
    output_price = float(manifest["api_caps"]["rate_assumption_output_usd_per_million"])
    reservation = (
        manifest["api_caps"]["per_request_input_token_cap"] * input_price
        + manifest["api_caps"]["per_request_completion_token_cap"] * output_price
    ) / 1_000_000
    attempt1_integrity = json.loads((ATTEMPT1_RUN / "RUN_INTEGRITY.json").read_text(encoding="utf-8"))
    attempt2_integrity = json.loads((ATTEMPT2_RUN / "RUN_INTEGRITY.json").read_text(encoding="utf-8"))
    previous_known_and_reserved = (
        float(attempt1_integrity["measured_rate_cost_usd"])
        + float(attempt2_integrity["api"]["known_rate_cost_usd"])
        + float(attempt2_integrity["api"]["uncertain_request_reserved_cost_usd"])
    )
    max_cost = API_REQUEST_CAP * reservation
    if previous_known_and_reserved + max_cost > GLOBAL_SPEND_CEILING_USD:
        raise RuntimeError("attempt-3 worst-case reservation would exceed the cumulative Iteration 29 US$10 ceiling")
    manifest["api_caps"].update({
        "api_request_cap": API_REQUEST_CAP,
        "cost_cap_usd": round(max_cost, 10),
        "expected_cost_estimate_usd": 2.0,
        "full_request_cap_worst_case_usd_at_assumed_rates": round(max_cost, 10),
    })
    manifest["sparql_caps"].update({"physical_read_only_select_cap": SPARQL_ATTEMPT_CAP})
    manifest["cumulative_resource_budget"] = {
        "global_hard_spend_ceiling_usd": GLOBAL_SPEND_CEILING_USD,
        "prior_attempt1_known_rate_cost_usd": round(float(attempt1_integrity["measured_rate_cost_usd"]), 10),
        "prior_attempt2_known_rate_cost_usd": round(float(attempt2_integrity["api"]["known_rate_cost_usd"]), 10),
        "prior_attempt2_uncertain_request_reservation_usd": round(float(attempt2_integrity["api"]["uncertain_request_reserved_cost_usd"]), 10),
        "prior_known_and_reserved_exposure_usd": round(previous_known_and_reserved, 10),
        "attempt3_max_request_reservation_usd": round(max_cost, 10),
        "cumulative_worst_case_usd": round(previous_known_and_reserved + max_cost, 10),
        "attempt3_max_physical_sparql_selects": SPARQL_ATTEMPT_CAP,
        "previous_attempt1_and_attempt2_physical_selects": int(attempt1_integrity["physical_read_only_select_attempts"]) + int(attempt2_integrity["sparql"]["physical_read_only_select_attempts"]),
    }
    manifest["failure_treatment"] = "ITT_EMPTY_BASELINE_REQUEST_FAILURE_AND_SELECTOR_BASELINE_FALLBACK_V2"
    manifest["response_failure_policy"] = {
        "baseline_length_empty_or_local_input_cap": "record as a preregistered baseline failure with empty predictions in both paired conditions; continue without retry",
        "baseline_http_non200_or_transport_timeout": "record exactly one HTTP/transport failure with status/error metadata and full request reservation; empty paired predictions; continue without retry",
        "selector_length_empty_malformed_or_unknown_id": "record one invalid selector and preserve the exact baseline prediction",
        "selector_http_non200_or_transport_timeout": "record one HTTP/transport failure with status/error metadata and full request reservation; preserve the exact baseline prediction; continue without retry",
        "selector_local_input_cap": "no API request; record local cap and preserve exact baseline prediction",
        "duplicate_prompt_or_request_hash": "block duplicate; halt run; do not retry",
        "SPARQL_http_error_or_timeout": "record attempt and stop incomplete without score or resume",
        "API_budget_or_hard_overall_deadline": "stop incomplete without score or resume",
        "scoring": "offline only after all 800 paired predictions are durable and the independent gate returns SCOREABLE",
    }
    manifest["authorization"] = {
        "state": "AUTHORIZED_BY_STANDING_USER_DIRECTIVE",
        "authorized": True,
        "authorization_record_path": AUTHORIZATION_PATH.relative_to(ROOT).as_posix(),
        "authorization_basis": "User explicitly directed the agent to continue all work through the final result without requesting further permission. The execution record binds that standing authorization to this exact frozen manifest hash and the stated tighter caps.",
        "authorization_text": "I authorize every work until you reach the final result no need to ask my permission anymore.",
        "prior_authorizations_roll_over": False,
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json({key: value for key, value in manifest.items() if key != "manifest_sha256"}))

    authorization = {
        "authorized": True,
        "manifest_sha256": manifest["manifest_sha256"],
        "authorization_text": "I authorize every work until you reach the final result no need to ask my permission anymore. This exact-hash execution record applies that standing instruction to Iteration 29 attempt 3 under the manifest's listed caps.",
        "authorization_basis": "Standing user authorization in conversation; no new approval was requested.",
        "accepted_caps": {
            "model": manifest["api_caps"]["model"],
            "api_request_cap": manifest["api_caps"]["api_request_cap"],
            "per_request_input_token_cap": manifest["api_caps"]["per_request_input_token_cap"],
            "per_request_completion_token_cap": manifest["api_caps"]["per_request_completion_token_cap"],
            "cost_cap_usd": manifest["api_caps"]["cost_cap_usd"],
            "physical_read_only_select_cap": manifest["sparql_caps"]["physical_read_only_select_cap"],
            "endpoint": manifest["sparql_caps"]["endpoint"],
            "per_request_timeout_seconds": manifest["api_caps"]["per_request_timeout_seconds"],
            "wall_clock_cap_hours": manifest["time_caps"]["wall_clock_cap_hours"],
            "max_retries": 0,
            "resume_allowed": False,
        },
    }

    _write_immutable(OUT_COHORT, cohort_bytes)
    _write_immutable(OUT_MANIFEST, (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    _write_immutable(AUTHORIZATION_PATH, (json.dumps(authorization, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return {
        "status": manifest["status"],
        "cases": len(cohort),
        "attempt1_exposed_excluded": attempt_exposure_counts["attempt1"],
        "attempt2_exposed_excluded": attempt_exposure_counts["attempt2"],
        "all_prior_exposed_excluded": len(exposed_ids),
        "eligible_pool_after_exclusions": len(pool),
        "exact_text_excluded": exact_excluded,
        "near_duplicate_excluded": near_excluded,
        "cohort_sha256": cohort_hash,
        "manifest_sha256": manifest["manifest_sha256"],
        "api_caps": manifest["api_caps"],
        "sparql_caps": manifest["sparql_caps"],
        "time_caps": manifest["time_caps"],
        "cumulative_resource_budget": manifest["cumulative_resource_budget"],
    }


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, sort_keys=True, indent=2))
