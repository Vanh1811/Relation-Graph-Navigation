"""Freeze attempt-4 manifest and exact approval text without transport."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.iter_026_answer_filter import cohort

ROOT = cohort.ROOT
RESULTS = cohort.OUT
PREP = RESULTS / "ATTEMPT4_PREPARATION"
MANIFEST_PATH = RESULTS / "LIVE_MANIFEST_ATTEMPT4.json"
HASH_PATH = RESULTS / "LIVE_MANIFEST_ATTEMPT4.sha256"
AUTH_PATH = RESULTS / "ATTEMPT4_AUTHORIZATION_REQUIRED.md"
MAX_API_REQUESTS = 140
MAX_SELECT_ATTEMPTS = 300
INPUT_CAP = 7000
COMPLETION_CAP = 1024
INPUT_PRICE = 0.20
OUTPUT_PRICE = 1.20
PER_REQUEST_RESERVATION = (INPUT_CAP * INPUT_PRICE + COMPLETION_CAP * OUTPUT_PRICE) / 1_000_000
API_COST_CAP = round(MAX_API_REQUESTS * PER_REQUEST_RESERVATION, 7)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cohort_path = PREP / "COHORT_MANIFEST.json"
    inference_path = PREP / "INFERENCE_COHORT_NON_GOLD.jsonl"
    gold_path = PREP / "SCORING_GOLD_ONLY.jsonl"
    cohort_manifest = json.loads(cohort_path.read_text(encoding="utf-8"))
    parent = json.loads((RESULTS / "LIVE_MANIFEST_ATTEMPT3.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in inference_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 12 or [row["question_key"] for row in rows] != cohort_manifest["case_keys"]:
        raise RuntimeError("attempt-4 inference cohort and manifest must contain the same ordered twelve cases")
    if any(any("gold" in key.casefold() or "scoring" in key.casefold() for key in row) for row in rows):
        raise RuntimeError("gold/scoring data appears in the inference cohort")

    production_sources = {
        relative: file_sha(ROOT / relative)
        for relative in parent.get("source_files_sha256", {})
    }
    extra_docs = {
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/ATTEMPT4_FROZEN_PROTOCOL.md",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/ATTEMPT4_RUNNER_SAFETY_REVIEW.md",
    }
    production_sources.update({relative: file_sha(ROOT / relative) for relative in extra_docs})
    diagnostic_sources = {
        str(path.relative_to(ROOT)): file_sha(path)
        for path in sorted((ROOT / "scripts/diagnostics/iter_026_answer_filter").glob("*.py"))
    }
    if "scripts/diagnostics/iter_026_answer_filter/run_live_attempt4.py" not in diagnostic_sources:
        raise RuntimeError("attempt-4 one-shot runner is missing from diagnostic source hashes")
    manifest: dict[str, Any] = {
        "iteration": 26,
        "attempt_number": 4,
        "status": "FROZEN_AWAITING_EXPLICIT_AUTHORIZATION",
        "decision_gate": "A — CAP-AWARE DEVELOPMENT RUN READY; STOP BEFORE TRANSPORT",
        "evidence_class": "DEVELOPMENT_EVALUATION; NOT_CONFIRMATORY_HOLDOUT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_scope": "One fresh one-shot run of twelve fixed, disjoint GrailQA dev questions. Each question receives one TGAR v1.1.3 upstream trajectory and, only at the final eligible answer-filter boundary, a candidate-ID selector fork. This cohort is development data because earlier Iteration 26 outcomes informed the runtime changes.",
        "model": "gpt-5.6-luna",
        "api": "OpenAI Chat Completions API at https://api.openai.com/v1/chat/completions",
        "api_model_documentation_url": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
        "api_price_source_url": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
        "api_price_verified_utc": "2026-11-08",
        "api_input_price_usd_per_million_uncached": INPUT_PRICE,
        "api_cached_input_price_usd_per_million_not_assumed": 0.02,
        "api_output_price_usd_per_million": OUTPUT_PRICE,
        "cost_reservation_basis": "Uncached input and the full completion cap per request; no prompt-cache discount assumed.",
        "api_request_cap": MAX_API_REQUESTS,
        "api_cost_cap_usd": API_COST_CAP,
        "full_request_cap_worst_case_reservation_usd": API_COST_CAP,
        "per_request_worst_case_reservation_usd": PER_REQUEST_RESERVATION,
        "input_tokens_per_request": INPUT_CAP,
        "completion_tokens_per_request": COMPLETION_CAP,
        "reasoning_effort": "low",
        "sampling_parameters": "temperature and unsupported sampling controls omitted; model defaults apply",
        "timeout_seconds_per_request": 120,
        "wall_clock_cap_seconds": 28800,
        "max_retries": 0,
        "resume_allowed": False,
        "sparql_attempt_cap": MAX_SELECT_ATTEMPTS,
        "sparql_endpoint": "http://127.0.0.1:8890/sparql",
        "sparql_operations": "Physical read-only SELECT only, including endpoint preflight, failed/uncertain calls, navigation, and hidden label lookups.",
        "sample_size": 12,
        "question_keys": cohort_manifest["case_keys"],
        "dataset_allocation": {"grailqa": 12},
        "answer_cardinality_allocation": {"SINGLE": 6, "MULTI": 6},
        "branch_family_allocation_scoring_only": {"BRANCH_FINAL": 11, "BRANCH_INTERMEDIATE": 1},
        "allocation_by_cardinality_and_branch_family": cohort_manifest["selected_cell_counts"],
        "selection_seed": cohort_manifest["selection_seed"],
        "selection_policy": cohort_manifest["selection_policy"],
        "cohort_manifest_sha256": file_sha(cohort_path),
        "inference_cohort_sha256": file_sha(inference_path),
        "scoring_gold_sha256_offline_only": file_sha(gold_path),
        "cohort_directory": "ATTEMPT4_PREPARATION",
        "attempt3_question_ids_excluded": cohort_manifest["attempt3_question_ids_excluded"],
        "candidate_snapshots_available_before_run": 0,
        "new_upstream_generation_required": True,
        "baseline": parent["baseline"],
        "intervention": parent["intervention"],
        "input_cap_policy": {
            "baseline_oversized_prompt": "Record empty paired predictions as BASELINE_INPUT_CAP_EXCEEDED_NO_OUTPUT; continue next fixed case; no truncation, reservation, transport, or retry.",
            "selector_oversized_prompt": "Record SKIPPED_INPUT_CAP_EXCEEDED and retain baseline prediction; continue without transport.",
            "scope_of_local_catch": "Only typed InputTokenLimitExceeded; global cap exhaustion, duplicate prompt, timeout, and uncertain transport stop the one-shot run.",
        },
        "upstream_empty_or_truncated_policy": "One completed HTTP response with finish_reason other than stop or empty content is recorded as a case-level upstream failure with empty paired predictions and no retry; continue to the next fixed case.",
        "analysis_status": "Development-screen metrics only; passing thresholds does not establish confirmatory or independent efficacy.",
        "prior_attempt_accounting": {
            "attempt1_manifest_sha256": "d2a0ae41bf11e5a3a9d32629b219e778539775ad7d6a2977838e575b482435aa",
            "attempt2_manifest_sha256": "e7ff984fad4062987347a72a9e97cd0c6cd1f580c37c4672cd15bffcc5fafb1c",
            "attempt3_manifest_sha256": "ee5e74706db2410122170af37e56ba6d7429615f2a1bb128c79c7c641420da8f",
            "attempts_1_to_3_consumed": True,
            "attempt3_api_requests_started": 33,
            "attempt3_physical_select_attempts": 51,
            "attempt3_completed_cases": 4,
            "attempt3_authorization_and_unused_caps_roll_over": False,
        },
        "run_directory": "RUNS/iter26_live_attempt4",
        "one_shot_authorization_consumption_marker": "AUTHORIZATION_CONSUMPTION_ATTEMPT4.json",
        "authorization_consumed": False,
        "attempt4_preparation_transport_activity": {"api_requests": 0, "physical_sparql_attempts": 0, "spend_usd": 0.0},
        "source_files_sha256": production_sources,
        "diagnostic_files_sha256": diagnostic_sources,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = file_sha(MANIFEST_PATH)
    HASH_PATH.write_text(f"{digest}  {MANIFEST_PATH.name}\n", encoding="utf-8")
    (RESULTS / "ATTEMPT4_AUTHORIZATION_REQUIRED.md").write_text(f"""# Iteration 26 attempt 4 — explicit authorization required

No live API or SPARQL request has been made for attempt 4. Attempts 1–3 are consumed and their unused caps do not roll over.

Frozen manifest: `LIVE_MANIFEST_ATTEMPT4.json`  
SHA-256: `{digest}`

Exact proposed authorization:

> I authorize Iteration 26 attempt 4 manifest `{digest}`: one fresh one-shot run of the twelve fixed GrailQA dev questions listed in the manifest; gpt-5.6-luna only; up to 140 OpenAI Chat Completions requests, 7,000 input tokens and 1,024 completion tokens per request, with a US$0.368032 worst-case rate-based spend ceiling; up to 300 physical read-only SPARQL SELECT attempts including the endpoint preflight, failed attempts, navigation, and hidden label lookups; endpoint `http://127.0.0.1:8890/sparql`; maximum 120 seconds per request and 8 hours overall; zero retries and no resume. Oversized baseline prompts are recorded as empty paired case failures and the run continues; oversized selector prompts fall back to the baseline without a request. This is a development evaluation, not a confirmatory holdout. This is a new authorization; prior manifests are consumed and their unused caps do not roll over.

The runner verifies the exact manifest, cohort, source/code hashes, output path, and a new consumption marker before reading credentials or making the endpoint preflight. It performs no API-key test or SPARQL call before this exact approval.
""", encoding="utf-8")
    print(json.dumps({
        "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "manifest_sha256": digest,
        "api_request_cap": MAX_API_REQUESTS,
        "api_cost_cap_usd": API_COST_CAP,
        "sparql_attempt_cap": MAX_SELECT_ATTEMPTS,
        "sample_size": len(rows),
        "authorization_file": str((RESULTS / "ATTEMPT4_AUTHORIZATION_REQUIRED.md").relative_to(ROOT)),
        "transport_counts": {"api_requests": 0, "physical_sparql_attempts": 0, "spend_usd": 0.0},
    }, indent=2))


if __name__ == "__main__":
    main()
