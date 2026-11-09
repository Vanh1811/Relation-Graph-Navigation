"""Freeze one untouched, gold-free GrailQA development cohort for Iteration 29."""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_028_terminal_boundary_repair"
SOURCE = ROOT / "data/interim/grailqa_dev.jsonl"
SAMPLE_N = 800
SEED = 2801
TOKEN_JACCARD_THRESHOLD = 0.90
TOKEN_LENGTH_RATIO_MIN = 0.82

PROMPT = """You are a conservative KGQA answer selector.
Choose answer entities only from the supplied candidate list. Use only the question, the executed relation path, the baseline answer IDs, and the candidate labels. Do not search, call tools, invent IDs, or use information outside this payload. Select every candidate needed to give the complete answer and exclude candidates that do not answer the question. If the supplied evidence does not let you decide safely, return UNSURE.

Return exactly one JSON object with exactly these fields:
{"status":"SELECT"|"UNSURE","selected_ids":[string,...]}

For SELECT, selected_ids must be non-empty, unique, and copied exactly from candidate IDs. For UNSURE, selected_ids must be []. Do not return explanations or markdown."""
PROMPT_BYTES = (PROMPT + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def normalize_question(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def token_set(text: str) -> set[str]:
    return set(normalize_question(text).split())


def question_id(value: Any) -> str:
    value = str(value or "").strip()
    if value.startswith("grailqa::"):
        value = value.split("::", 1)[1]
    return value


def add_identity(
    *,
    qid: Any = None,
    dataset: str = "",
    question: Any = None,
    ids: set[str],
    texts: set[str],
) -> None:
    if isinstance(question, str):
        normalized = normalize_question(question)
        if normalized:
            texts.add(normalized)
    dataset = dataset.casefold().replace("-", "")
    raw_id = question_id(qid)
    if raw_id.isdigit() and (dataset in {"grailqa", "grailqa1.0"} or str(qid).startswith("grailqa::")):
        ids.add(raw_id)


def _visit_manifest(value: Any, *, ids: set[str], texts: set[str], source_name: str) -> None:
    """Visit only case identity fields; gold-valued fields are never accessed."""
    if isinstance(value, list):
        for item in value:
            _visit_manifest(item, ids=ids, texts=texts, source_name=source_name)
        return
    if not isinstance(value, dict):
        return
    dataset = str(value.get("dataset", ""))
    qid = value.get("question_key", value.get("question_id", value.get("qid")))
    if qid is None and dataset.casefold().replace("-", "") == "grailqa" and "question" in value:
        qid = value.get("id")
    add_identity(qid=qid, dataset=dataset, question=value.get("question"), ids=ids, texts=texts)
    for key in ("question_keys", "question_ids"):
        records = value.get(key)
        if isinstance(records, list):
            for record in records:
                if isinstance(record, str) and record.startswith("grailqa::"):
                    add_identity(qid=record, dataset="grailqa", ids=ids, texts=texts)
                elif dataset.casefold().replace("-", "") == "grailqa":
                    add_identity(qid=record, dataset=dataset, ids=ids, texts=texts)
    for key in ("cases", "strong_cases", "control_cases", "case_rows"):
        nested = value.get(key)
        if isinstance(nested, list):
            _visit_manifest(nested, ids=ids, texts=texts, source_name=source_name)


def gather_prior_exposure() -> tuple[set[str], set[str], list[dict[str, str]]]:
    ids: set[str] = set()
    texts: set[str] = set()
    audited_files: list[dict[str, str]] = []
    exp_root = ROOT / "outputs/experiments"
    eligible_jsonl_names = {
        "CASE_TRACES.jsonl",
        "INFERENCE_COHORT_NON_GOLD.jsonl",
        "PAIRED_RESULTS_NON_GOLD.jsonl",
    }
    for path in sorted(exp_root.rglob("*")):
        if not path.is_file():
            continue
        # Never count this draft's own frozen cohort/manifest as prior exposure.
        if path.is_relative_to(OUTPUT):
            continue
        relative = str(path.relative_to(ROOT))
        if "POSTHOC_SCORING_QUARANTINE" in relative or "GOLD_ONLY" in path.name or "scoring" in path.name.casefold():
            continue
        if path.name in eligible_jsonl_names:
            raw = path.read_bytes()
            audited_files.append({"path": relative, "sha256": sha256_bytes(raw)})
            for line in raw.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                qkey = row.get("question_key")
                dataset = str(row.get("dataset", ""))
                qid = qkey or row.get("question_id") or row.get("qid")
                if qkey and str(qkey).startswith("grailqa::"):
                    dataset = "grailqa"
                add_identity(qid=qid, dataset=dataset, question=row.get("question"), ids=ids, texts=texts)
            continue
        if "manifest" in path.name.casefold() and path.suffix == ".json":
            try:
                raw = path.read_bytes()
                parsed = json.loads(raw)
            except (OSError, json.JSONDecodeError):
                continue
            audited_files.append({"path": relative, "sha256": sha256_bytes(raw)})
            _visit_manifest(parsed, ids=ids, texts=texts, source_name=relative)
    return ids, texts, audited_files


def _is_near_duplicate(
    candidate_tokens: set[str], candidate_norm: str,
    previous_tokens: set[str], previous_norm: str,
) -> bool:
    if candidate_norm == previous_norm:
        return True
    if not candidate_tokens or not previous_tokens:
        return False
    length_ratio = min(len(candidate_tokens), len(previous_tokens)) / max(len(candidate_tokens), len(previous_tokens))
    if length_ratio < TOKEN_LENGTH_RATIO_MIN:
        return False
    jaccard = len(candidate_tokens & previous_tokens) / len(candidate_tokens | previous_tokens)
    return jaccard >= TOKEN_JACCARD_THRESHOLD


def _write_immutable(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f"refusing to overwrite a different frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def prepare() -> dict[str, Any]:
    prior_ids, prior_texts, audited_files = gather_prior_exposure()
    source_bytes = SOURCE.read_bytes()
    source_sha = sha256_bytes(source_bytes)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source_bytes.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        # The protocol uses only identity, question, split, and topic-linking fields.
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
        row["norm"] = normalize_question(row["question"])
        row["tokens"] = set(row["norm"].split())
        records.append(row)

    prior_norms = sorted(prior_texts)
    prior_token_sets = [set(text.split()) for text in prior_norms]
    exact_text_excluded: list[str] = []
    near_text_excluded: list[str] = []
    pool: list[dict[str, Any]] = []
    for row in records:
        if row["norm"] in prior_texts:
            exact_text_excluded.append(row["id"])
            continue
        if any(
            _is_near_duplicate(row["tokens"], row["norm"], tokens, text)
            for text, tokens in zip(prior_norms, prior_token_sets)
        ):
            near_text_excluded.append(row["id"])
            continue
        pool.append(row)

    rng = random.Random(SEED)
    pool.sort(key=lambda row: row["id"])
    rng.shuffle(pool)
    selected: list[dict[str, Any]] = []
    for row in pool:
        if any(
            _is_near_duplicate(row["tokens"], row["norm"], prior["tokens"], prior["norm"])
            for prior in selected
        ):
            continue
        selected.append(row)
        if len(selected) == SAMPLE_N:
            break
    if len(selected) != SAMPLE_N:
        raise SystemExit(f"only {len(selected)} untouched rows remain; cannot freeze {SAMPLE_N}")

    cohort_rows: list[dict[str, Any]] = []
    question_keys: list[str] = []
    row_hashes: list[str] = []
    for index, row in enumerate(selected, 1):
        safe = {
            "case_id": f"iter29_case_{index:04d}",
            "question_key": f"grailqa::{row['id']}",
            "dataset": "grailqa",
            "split": "dev",
            "question": row["question"],
            "topic_entities": row["topic_entities"],
            "topic_entity_names": row["topic_entity_names"],
        }
        safe["question_sha256"] = sha256_bytes(safe["question"].encode("utf-8"))
        row_digest = sha256_bytes(canonical(safe))
        safe["safe_row_sha256"] = row_digest
        cohort_rows.append(safe)
        question_keys.append(safe["question_key"])
        row_hashes.append(row_digest)
    cohort_bytes = b"".join(canonical(row) + b"\n" for row in cohort_rows)
    cohort_sha = sha256_bytes(cohort_bytes)
    ids_sha = sha256_bytes("\n".join(question_keys).encode("utf-8"))
    exposure_bytes = canonical(sorted(audited_files, key=lambda item: item["path"]))
    exposure_sha = sha256_bytes(exposure_bytes)
    prompt_sha = sha256_bytes(PROMPT_BYTES)

    manifest: dict[str, Any] = {
        "iteration": 29,
        "status": "DRAFT_AUTHORIZATION_REQUIRED",
        "evidence_class": "FRESH_UNTOUCHED_GRAILQA_DEV_SLICE_NOT_OFFICIAL_TEST_HOLDOUT",
        "experiment": "terminal_candidate_selector_v1_paired_with_frozen_upstream_candidates",
        "manifest_scope": "one fixed 800-question run; one baseline pass per question; at most one terminal selector request only on the frozen gold-free trigger",
        "failure_treatment": "ITT_EMPTY_METHOD_FAILURE_AND_SELECTOR_BASELINE_FALLBACK_V1",
        "dataset": "GrailQA",
        "split": "dev",
        "sample_size": SAMPLE_N,
        "sampling_seed": SEED,
        "sampling_policy": "uniform pseudorandom order from valid GrailQA dev rows after prior-ID, exact-text, and near-duplicate exclusions; select first 800",
        "near_duplicate_policy": {
            "normalization": "case-fold, tokenize ASCII letters/digits, collapse whitespace",
            "token_jaccard_threshold": TOKEN_JACCARD_THRESHOLD,
            "minimum_token_count_ratio": TOKEN_LENGTH_RATIO_MIN,
            "applied_against": "all discovered prior-exposure question texts and earlier selected questions",
        },
        "question_keys": question_keys,
        "question_keys_sha256": ids_sha,
        "inference_cohort_path": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_028_terminal_boundary_repair/NEXT_INFERENCE_COHORT_NON_GOLD.jsonl",
        "inference_cohort_sha256": cohort_sha,
        "per_question_safe_row_sha256": row_hashes,
        "source_dataset_path": "data/interim/grailqa_dev.jsonl",
        "source_dataset_sha256": source_sha,
        "prior_exposure_audit": {
            "audited_artifact_count": len(audited_files),
            "audited_artifacts_sha256": exposure_sha,
            "discovered_prior_grailqa_question_id_count": len(prior_ids),
            "discovered_prior_grailqa_question_ids_sha256": sha256_bytes("\n".join(sorted(prior_ids)).encode("utf-8")),
            "prior_question_text_count": len(prior_texts),
            "exact_text_exclusion_count": len(exact_text_excluded),
            "near_duplicate_exclusion_count": len(near_text_excluded),
            "question_ids_excluded_by_prior_exposure": sorted(prior_ids),
            "exact_text_excluded_question_keys": [f"grailqa::{value}" for value in sorted(exact_text_excluded)],
            "near_duplicate_excluded_question_keys": [f"grailqa::{value}" for value in sorted(near_text_excluded)],
            "limitations": "This audit covers question identities and texts found in local experiment traces, non-gold cohorts, and manifests; it cannot prove that no question was exposed outside these artifacts.",
        },
        "baseline": {
            "method": "unchanged TGAR v1.1.3 / CoR runtime",
            "one_baseline_pass_per_question": True,
            "production_sources_modified": False,
        },
        "intervention": {
            "method": "TerminalCandidateSelector-v1",
            "model": "gpt-5.6-luna",
            "prompt_path": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_028_terminal_boundary_repair/TERMINAL_SELECTOR_PROMPT_V1.txt",
            "prompt_sha256": prompt_sha,
            "reasoning_effort": "low",
            "max_completion_tokens": 1024,
            "max_input_tokens": 7000,
            "max_candidate_count": 100,
            "maximum_selector_requests_per_case": 1,
            "frozen_trigger": "route=CoR; terminal source is direct reasoning STOP or one graph filter; at least 2 unique entity candidate IDs and unique aligned labels from the final successful entity_search; baseline output contains nonempty candidate IDs only; no literal candidates; exact answer-name-to-ID mapping; aggregate/quantity question excluded; no failed/ambiguous candidate provenance; local prompt estimate within 7000 tokens",
            "candidate_context": "question, source relation IDs, baseline output IDs, candidate IDs and labels; no new KG lookup or relation traversal",
            "fallback": "If selector says UNSURE, returns malformed/empty/non-stop/unknown/repeated IDs, or fails local validation, preserve baseline predictions exactly and record one attempt; no retry.",
        },
        "primary_endpoint": "paired all-cohort Hits@1 difference; intervention prediction falls back to baseline on every ineligible or invalid-selector case",
        "secondary_endpoints": [
            "paired F1 and per-question wins/losses/ties",
            "valid response and selector fallback rates",
            "eligible-case rate and candidate membership/provenance accuracy",
            "baseline-correct regression count",
            "latency, model calls, SPARQL attempts, estimated cost",
        ],
        "analysis_plan": {
            "primary_denominator": "all 800 fixed questions, intention-to-treat",
            "eligible_only_analysis": "exploratory; never substitutes for all-cohort analysis",
            "minimum_eligible_cases_for_scoring": 1,
            "minimum_eligible_cases_for_promotion": 150,
            "uncertainty": "paired 95% confidence interval for Hits@1 risk difference and exact two-sided McNemar test",
            "sample_size_rationale": {
                "observed_strict_eligibility_rate": "13/38 = 34.2% from non-gold Iteration 27 trace audit",
                "expected_eligible_cases": 274,
                "discordance_planning_rate": "3/12 = 25% development-only disagreement rate from Iteration 26 attempt 4",
                "minimum_detectable_all_cohort_effect": "approximately 5 percentage points at 80% power, two-sided alpha 0.05 using n ~= (1.96+0.84)^2 * discordance_rate / effect^2; planning approximation only",
                "no_confirmatory_claim_from_prior_development_cohorts": True,
            },
            "promotion_criteria": [
                "all-cohort paired Hits@1 improvement with two-sided 95% CI lower bound > 0 and exact McNemar p < 0.05",
                "at least 150 selector-eligible independent questions",
                "at least 90% valid non-fallback selector responses among selector calls",
                "100% candidate ID membership and provenance validity for applied selections",
                "zero baseline-correct regressions",
                "incremental spend and latency remain within frozen caps",
            ],
            "mandatory_stop": "If the first 100 fixed cases produce zero eligible cases, stop before further inference; label MECHANISM_FAILURE_NO_ELIGIBILITY and do not score. If the full run has fewer than 150 eligible cases, no promotion claim; report underpowered/low-coverage result.",
        },
        "response_failure_policy": {
            "baseline_length_empty_or_local_input_cap": "record as a preregistered method failure with empty predictions in both paired conditions; continue without retry",
            "selector_length_empty_malformed_or_unknown_id": "one request only; record invalid selector and use exact baseline prediction as the paired intervention output",
            "transport_timeout_http_error_budget_or_deadline": "abort; no complete score and no resume",
            "KG_select_error": "record; candidate path ineligible; preserve baseline output",
            "duplicate_prompt_or_request_hash": "no retry; record case failure or abort according to whether any remote request began; external validator rejects repeated request hashes",
        },
        "api_caps": {
            "model": "gpt-5.6-luna",
            "api_request_cap": 3600,
            "per_request_input_token_cap": 7000,
            "per_request_completion_token_cap": 1024,
            "per_request_timeout_seconds": 120,
            "max_retries": 0,
            "resume_allowed": False,
            "cost_cap_usd": 10.00,
            "rate_assumption_input_uncached_usd_per_million": 0.20,
            "rate_assumption_output_usd_per_million": 1.20,
            "full_request_cap_worst_case_usd_at_assumed_rates": 9.46368,
            "expected_cost_estimate_usd": 2.00,
            "price_note": "Rates match the prior explicit gpt-5.6-luna authorization, not a currently verified public listing. Reconfirm exact model billing before any new authorization; do not substitute the separately listed gpt-6-luna rate.",
        },
        "sparql_caps": {
            "endpoint": "http://127.0.0.1:8890/sparql",
            "physical_read_only_select_cap": 12801,
            "per_case_select_cap": 16,
            "preflight_selects_included": 1,
            "mutations_allowed": False,
        },
        "time_caps": {"wall_clock_cap_hours": 24, "per_request_timeout_seconds": 120},
        "authorization": {
            "state": "AUTHORIZATION_REQUIRED",
            "authorized": False,
            "required_confirmation": "new explicit user authorization naming the final manifest SHA-256 and accepting exact caps after current model price and endpoint preflight are checked",
            "prior_authorizations_roll_over": False,
        },
        "gold_scoring": {
            "offline_only_after_external_validator_returns_SCOREABLE": True,
            "validator_path": "scripts/diagnostics/iter_028_terminal_boundary_repair/scoring_gate.py",
            "gold_labels_loaded_by_live_runner": False,
            "posthoc_iteration27_quarantine_used": False,
        },
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical(manifest))
    cohort_path = OUTPUT / "NEXT_INFERENCE_COHORT_NON_GOLD.jsonl"
    prompt_path = OUTPUT / "TERMINAL_SELECTOR_PROMPT_V1.txt"
    manifest_path = OUTPUT / "NEXT_MANIFEST_DRAFT.json"
    _write_immutable(cohort_path, cohort_bytes)
    _write_immutable(prompt_path, PROMPT_BYTES)
    _write_immutable(manifest_path, (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return {
        "sample_size": SAMPLE_N,
        "prior_exposed_ids": len(prior_ids),
        "prior_question_texts": len(prior_texts),
        "exact_text_excluded": len(exact_text_excluded),
        "near_duplicate_excluded": len(near_text_excluded),
        "remaining_pool": len(pool),
        "cohort_sha256": cohort_sha,
        "question_keys_sha256": ids_sha,
        "manifest_sha256": manifest["manifest_sha256"],
        "audited_artifact_count": len(audited_files),
    }


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, sort_keys=True, indent=2))
