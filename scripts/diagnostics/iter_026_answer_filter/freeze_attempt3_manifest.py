"""Freeze attempt-3 manifest and exact approval wording without transport."""
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
PREP = RESULTS / "ATTEMPT3_PREPARATION"
MANIFEST_PATH = RESULTS / "LIVE_MANIFEST_ATTEMPT3.json"
HASH_PATH = RESULTS / "LIVE_MANIFEST_ATTEMPT3.sha256"
AUTH_PATH = RESULTS / "ATTEMPT3_AUTHORIZATION_REQUIRED.md"
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
    attempt2 = json.loads((RESULTS / "LIVE_MANIFEST_ATTEMPT2.json").read_text(encoding="utf-8"))
    inference_rows = [
        json.loads(line)
        for line in inference_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(inference_rows) != 12 or len(cohort_manifest.get("case_keys", [])) != 12:
        raise RuntimeError("attempt-3 frozen input must have twelve fixed questions")
    if [row["question_key"] for row in inference_rows] != cohort_manifest["case_keys"]:
        raise RuntimeError("cohort file and cohort manifest question order differ")
    if any(any("gold" in key.casefold() or "scoring" in key.casefold() for key in row) for row in inference_rows):
        raise RuntimeError("gold/scoring data appears in the inference cohort")

    production_sources = dict(attempt2.get("source_files_sha256", {}))
    production_sources["chain_of_relations/tools/entity_prune.py"] = file_sha(ROOT / "chain_of_relations/tools/entity_prune.py")
    production_sources["outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/ATTEMPT3_FROZEN_PROTOCOL.md"] = file_sha(RESULTS / "ATTEMPT3_FROZEN_PROTOCOL.md")
    production_sources["outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/ATTEMPT3_RUNNER_SAFETY_REVIEW.md"] = file_sha(RESULTS / "ATTEMPT3_RUNNER_SAFETY_REVIEW.md")
    # Recompute rather than inherit old digests: the exact input bytes to this
    # fresh attempt are verified at authorization preflight.
    production_sources = {
        relative: file_sha(ROOT / relative)
        for relative in sorted(production_sources)
    }
    diagnostic_sources = {
        str(path.relative_to(ROOT)): file_sha(path)
        for path in sorted((ROOT / "scripts/diagnostics/iter_026_answer_filter").glob("*.py"))
    }
    if "scripts/diagnostics/iter_026_answer_filter/run_live_attempt3.py" not in diagnostic_sources:
        raise RuntimeError("attempt-3 one-shot runner is missing from frozen diagnostic sources")

    manifest: dict[str, Any] = {
        "iteration": 26,
        "attempt_number": 3,
        "status": "FROZEN_AWAITING_EXPLICIT_AUTHORIZATION",
        "decision_gate": "A — INDEPENDENT_SELECTOR_EVALUATION_READY; STOP BEFORE TRANSPORT",
        "evidence_class": "OFFLINE_DIAGNOSTIC_AND_FIXTURE_TESTS_ONLY; NO_ATTEMPT3_LIVE_EVIDENCE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_scope": "One fresh one-shot execution of twelve previously unused fixed GrailQA dev questions; one new TGAR v1.1.3 upstream trajectory per question and a candidate-ID selector fork only at the final eligible CoR answer-filter boundary. This is a new authorization; attempts 1 and 2 are consumed and no unused caps roll over.",
        "model": "gpt-5.6-luna",
        "api": "OpenAI Chat Completions API at https://api.openai.com/v1/chat/completions",
        "api_model_documentation_url": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
        "api_price_source_url": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
        "api_price_verified_utc": "2026-11-08",
        "api_input_price_usd_per_million_uncached": INPUT_PRICE,
        "api_cached_input_price_usd_per_million_not_assumed": 0.02,
        "api_output_price_usd_per_million": OUTPUT_PRICE,
        "cost_reservation_basis": "Uncached input and full completion cap per request; no prompt-cache discount assumed. Rate estimate/reservation only, not an invoice.",
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
        "sparql_operations": "Physical read-only SELECT only, including the endpoint preflight, failed/uncertain calls, navigation, and hidden label lookups; no SERVICE or update operations.",
        "sample_size": 12,
        "question_keys": cohort_manifest["case_keys"],
        "dataset_allocation": {"grailqa": 12},
        "answer_cardinality_allocation": {"SINGLE": 6, "MULTI": 6},
        "branch_family_allocation_scoring_only": {"BRANCH_FINAL": 6, "BRANCH_INTERMEDIATE": 6},
        "allocation_by_cardinality_and_branch_family": cohort_manifest["selected_cell_counts"],
        "selection_seed": cohort_manifest["selection_seed"],
        "selection_policy": cohort_manifest["selection_policy"],
        "cohort_manifest_sha256": file_sha(cohort_path),
        "inference_cohort_sha256": file_sha(inference_path),
        "scoring_gold_sha256_offline_only": file_sha(gold_path),
        "cohort_directory": "ATTEMPT3_PREPARATION",
        "candidate_snapshots_available_before_run": 0,
        "new_upstream_generation_required": True,
        "baseline": attempt2["baseline"],
        "intervention": attempt2["intervention"],
        "upstream_empty_or_truncated_policy": "One completed HTTP response with finish_reason other than stop or empty content is a case-level upstream failure. Preserve finish_reason, raw-response hash, usage and prompt hash; set both paired predictions to empty, mark selector ineligible, score as a tie in the all-question intention-to-treat denominator, and continue to the next fixed question. No local or remote retry. Transport uncertainty, timeout, gateway halt, or budget exhaustion aborts the one-shot run and forbids resume.",
        "primary_analysis": {
            "population": "All twelve fixed questions; baseline-upstream failures remain in the denominator as paired no-output ties.",
            "metric": "Paired Hits@1 wins, losses, ties and net wins; intervention minus baseline.",
            "eligibility_floor": 8,
            "minimum_valid_nonempty_selector_rate": 0.9,
            "minimum_candidate_membership_rate_among_accepted_outputs": 1.0,
            "minimum_net_hits_at_1_wins": 3,
            "maximum_baseline_correct_regressions": 0,
            "minimum_mean_f1_delta": 0.0,
            "promotion_result": "PASS qualifies only to propose a separate fresh larger experiment requiring separate explicit authorization; otherwise fail/inconclusive and no tuning on this cohort.",
        },
        "prior_attempt_accounting": {
            "attempt1_manifest_sha256": "d2a0ae41bf11e5a3a9d32629b219e778539775ad7d6a2977838e575b482435aa",
            "attempt1_api_requests_started": 1,
            "attempt1_physical_sparql_attempts": 1,
            "attempt1_uncertain_reserved_cost_usd": 0.0026288,
            "attempt2_manifest_sha256": "e7ff984fad4062987347a72a9e97cd0c6cd1f580c37c4672cd15bffcc5fafb1c",
            "attempt2_api_requests_started": 72,
            "attempt2_physical_sparql_attempts": 146,
            "attempt2_measured_rate_estimated_cost_usd": 0.0373552,
            "attempt2_completed_cases": 7,
            "attempt2_authorization_consumed": True,
            "attempt2_no_resume": True,
            "attempt3_caps_are_new_and_do_not_roll_over": True,
        },
        "run_directory": "RUNS/iter26_live_attempt3",
        "one_shot_authorization_consumption_marker": "AUTHORIZATION_CONSUMPTION_ATTEMPT3.json",
        "authorization_consumed": False,
        "attempt3_preparation_transport_activity": {
            "api_requests": 0,
            "physical_sparql_attempts": 0,
            "spend_usd": 0.0,
        },
        "source_files_sha256": production_sources,
        "diagnostic_files_sha256": diagnostic_sources,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = file_sha(MANIFEST_PATH)
    HASH_PATH.write_text(f"{digest}  {MANIFEST_PATH.name}\n", encoding="utf-8")
    authorization = f"""# Iteration 26 attempt 3 — explicit authorization required

No live API or SPARQL request has been made for attempt 3. Attempt 1 and attempt 2 authorizations are consumed; their unused caps expired.

Frozen manifest: `LIVE_MANIFEST_ATTEMPT3.json`  
SHA-256: `{digest}`

Exact proposed authorization:

> I authorize Iteration 26 attempt 3 manifest `{digest}`: one fresh one-shot run of the twelve fixed GrailQA dev questions listed in the manifest; gpt-5.6-luna only; up to 140 OpenAI Chat Completions requests, 7,000 input tokens and 1,024 completion tokens per request, with a US$0.368032 worst-case rate-based spend ceiling; up to 300 physical read-only SPARQL SELECT attempts including the endpoint preflight, failed attempts, navigation, and hidden label lookups; endpoint `http://127.0.0.1:8890/sparql`; maximum 120 seconds per request and 8 hours overall; zero retries and no resume. Empty/truncated model responses are recorded as one-shot case failures and do not trigger retries. This is a new authorization; prior manifests are consumed and their unused caps do not roll over.

The runner verifies this exact manifest, the cohort, all listed source hashes, the output path, and a new consumption marker before reading credentials. It performs no API-key test or SPARQL preflight before this exact approval.
"""
    AUTH_PATH.write_text(authorization, encoding="utf-8")
    print(json.dumps({
        "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "manifest_sha256": digest,
        "api_request_cap": MAX_API_REQUESTS,
        "api_cost_cap_usd": API_COST_CAP,
        "sparql_attempt_cap": MAX_SELECT_ATTEMPTS,
        "sample_size": len(inference_rows),
        "authorization_file": str(AUTH_PATH.relative_to(ROOT)),
        "transport_counts": {"api_requests": 0, "physical_sparql_attempts": 0, "spend_usd": 0.0},
    }, indent=2))


if __name__ == "__main__":
    main()
