"""Build the immutable Iteration 27 manifest and hash sidecar offline."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_027_results_first_answer_repair"

SOURCE_PATHS = [
    "scripts/diagnostics/iter_027_results_first_answer_repair/__init__.py",
    "scripts/diagnostics/iter_027_results_first_answer_repair/prepare_iter27.py",
    "scripts/diagnostics/iter_027_results_first_answer_repair/scv.py",
    "scripts/diagnostics/iter_027_results_first_answer_repair/provenance.py",
    "scripts/diagnostics/iter_027_results_first_answer_repair/run_live.py",
    "scripts/diagnostics/iter_027_results_first_answer_repair/score_completed_run.py",
    "scripts/diagnostics/iter_027_results_first_answer_repair/freeze_manifest.py",
    "scripts/diagnostics/iter_027_results_first_answer_repair/test_scv.py",
    "scripts/diagnostics/iter_027_results_first_answer_repair/test_run_live.py",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_027_results_first_answer_repair/ITER27_CAMPAIGN_PLAN.md",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_027_results_first_answer_repair/METHOD_SPEC.md",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_027_results_first_answer_repair/THESIS_READY_FINDINGS.md",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_027_results_first_answer_repair/OFFLINE_TEST_RESULTS.json",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_027_results_first_answer_repair/COHORT_MANIFEST.json",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_027_results_first_answer_repair/INFERENCE_COHORT_NON_GOLD.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_027_results_first_answer_repair/UPSTREAM_CANDIDATE_COVERAGE.csv",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_027_results_first_answer_repair/DEV_DIAGNOSTIC_CASE_MATRIX.csv",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_027_results_first_answer_repair/DEV_EMPTY_ONLY_COUNTERFACTUAL.json",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_027_results_first_answer_repair/ITER26_SOURCE_HASHES.json",
    "scripts/diagnostics/iter_026_answer_filter/run_live_attempt4.py",
    "scripts/diagnostics/iter_026_answer_filter/strict_runtime_attempt4.py",
    "scripts/diagnostics/iter_026_answer_filter/gateway_attempt4.py",
    "scripts/diagnostics/iter_026_answer_filter/gateway.py",
    "scripts/diagnostics/iter_026_answer_filter/runtime.py",
    "scripts/diagnostics/iter_026_answer_filter/strict_runtime.py",
    "scripts/diagnostics/iter_026_answer_filter/selector.py",
    "scripts/diagnostics/iter_026_answer_filter/transport.py",
    "scripts/diagnostics/transport_safety_repair/hard_deadline.py",
    "chain_of_relations/schema.py",
    "chain_of_relations/llm_api.py",
    "chain_of_relations/kg_backend/__init__.py",
    "chain_of_relations/kg_backend/freebase_backend.py",
    "chain_of_relations/kg_backend/freebase/db_func.py",
    "chain_of_relations/methods/adaptive_v1_1_3/__init__.py",
    "chain_of_relations/methods/adaptive_v1_1_3/agent.py",
    "chain_of_relations/methods/adaptive_v1_1_3/router.py",
    "chain_of_relations/methods/cor/__init__.py",
    "chain_of_relations/methods/cor/agent.py",
    "chain_of_relations/methods/cor/dfs_stack.py",
    "chain_of_relations/methods/cor/prompt.yml",
    "chain_of_relations/methods/cor/tools/__init__.py",
    "chain_of_relations/methods/cor/tools/filter.py",
    "scripts/run_cor_baseline_experiment.py",
    "pyproject.toml",
    "data/processed/grailqa_dev_analysis.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/ATTEMPT4_LIVE_REPORT.md",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/ITER26_DECISION.md",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/RUNS/iter26_live_attempt4/CASE_TRACES.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/RUNS/iter26_live_attempt4/PAIRED_RESULTS_NON_GOLD.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/RUNS/iter26_live_attempt4/REQUEST_JOURNAL.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/RUNS/iter26_live_attempt4/SPARQL_QUERY_JOURNAL.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/RUNS/iter26_live_attempt4/RUN_METADATA.json",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/ATTEMPT4_PREPARATION/COHORT_MANIFEST.json",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/ATTEMPT4_PREPARATION/INFERENCE_COHORT_NON_GOLD.jsonl",
    "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/ATTEMPT4_PREPARATION/SCORING_GOLD_ONLY.jsonl",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> str:
    missing = [relative for relative in SOURCE_PATHS if not (ROOT / relative).is_file()]
    if missing:
        raise SystemExit(f"missing frozen source files: {missing}")
    cohort_manifest_path = OUT / "COHORT_MANIFEST.json"
    inference_path = OUT / "INFERENCE_COHORT_NON_GOLD.jsonl"
    cohort = json.loads(cohort_manifest_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in inference_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 40 or cohort.get("sample_size") != 40:
        raise SystemExit("frozen cohort must contain exactly 40 rows")
    if [row["question_key"] for row in rows] != cohort.get("selected_question_keys"):
        raise SystemExit("cohort question order differs from cohort manifest")
    try:
        from scripts.diagnostics.iter_027_results_first_answer_repair.scv import INSTRUCTIONS
    except ImportError as exc:
        raise SystemExit("could not load the frozen SCV instruction text") from exc

    digest = lambda rel: _sha(ROOT / rel)
    max_per_request = (7000 * 0.20 + 1024 * 1.20) / 1_000_000
    full_reservation = 400 * max_per_request
    if round(max_per_request, 10) != 0.0026288 or round(full_reservation, 8) != 1.05152:
        raise AssertionError("worst-case rate reservation calculation changed")
    manifest = {
        "iteration": 27,
        "status": "FROZEN_AWAITING_EXPLICIT_AUTHORIZATION",
        "method": "SCV-EmptyOnly",
        "evidence_class": "FRESH_GRAILQA_DEV_DEVELOPMENT_EVALUATION_NOT_CONFIRMATORY_HOLDOUT",
        "authorization_consumed": False,
        "authorization_note": "This manifest and prior user intent do not authorize live API or SPARQL calls. Separate exact-hash approval is required.",
        "cohort_dataset": "GrailQA dev",
        "cohort_size": 40,
        "question_keys": [row["question_key"] for row in rows],
        "cohort_manifest_sha256": _sha(cohort_manifest_path),
        "inference_cohort_sha256": _sha(inference_path),
        "source_dataset_sha256": digest("data/processed/grailqa_dev_analysis.jsonl"),
        "selection_policy": "Exact historical GrailQA ID exclusion plus normalized-exact/high token-Jaccard near-duplicate screen at >=0.80; stable SHA-256 rank; unstratified; no gold answers, gold SPARQL/logical form, structure label, route, candidate coverage or correctness used in selection. Semantic paraphrase identity is not proven.",
        "method_trigger": "One completed CoR route; exactly one final entity filter; existing conservative entity-only gate passes; baseline entity IDs empty; nonempty valid IDs/labels with shared executed query+trace provenance. All other cases preserve unchanged TGAR output.",
        "baseline": "Frozen production TGAR v1.1.3 config as called by scripts/diagnostics/iter_026_answer_filter/run_live_attempt4.py; exactly one run per case.",
        "selector_prompt_sha256": hashlib.sha256(INSTRUCTIONS.encode("utf-8")).hexdigest(),
        "response_schema": {
            "schema_version": "iter27.scv_empty_only.v1",
            "required_keys": ["schema_version", "decision", "candidate_ids"],
            "decisions": ["select", "abstain"],
            "candidate_membership_required": True,
            "duplicate_or_ambiguous_labels_rejected": True,
        },
        "comparator": "Same-run unchanged baseline answer vs SCV output from that exact captured candidate snapshot; baseline and snapshot fsynced before selector call.",
        "model": "gpt-5.6-luna",
        "endpoint": "OpenAI Chat Completions /v1/chat/completions",
        "reasoning_effort": "low",
        "sampling": "Model defaults; temperature omitted; streaming disabled.",
        "api_request_cap": 400,
        "api_cost_cap_usd": 1.05152,
        "api_input_price_usd_per_million_uncached": 0.20,
        "api_output_price_usd_per_million": 1.20,
        "pricing_source": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
        "pricing_checked_date": "2026-11-08",
        "per_request_worst_case_reservation_usd": 0.0026288,
        "full_request_cap_worst_case_reservation_usd": 1.05152,
        "input_tokens_per_request": 7000,
        "completion_tokens_per_request": 1024,
        "sparql_attempt_cap": 650,
        "sparql_endpoint": "http://127.0.0.1:8890/sparql",
        "sparql_allowlist": "SELECT only; physical preflight, failed attempts, navigation and hidden label lookups all count.",
        "timeout_seconds_per_request": 120,
        "wall_clock_cap_seconds": 7200,
        "max_retries": 0,
        "resume_allowed": False,
        "one_shot_run": True,
        "run_directory": "RUNS/iter27_scv_empty_only",
        "consumption_marker": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_027_results_first_answer_repair/AUTHORIZATION_CONSUMPTION.json",
        "expected_non_gold_outputs": [
            "RUN_METADATA.json", "PREFLIGHT.json", "CASE_TRACES.jsonl",
            "PAIRED_RESULTS_NON_GOLD.jsonl", "FROZEN_CASE_SNAPSHOTS_NON_GOLD.jsonl",
            "REQUEST_JOURNAL.jsonl", "SPARQL_QUERY_JOURNAL.jsonl", "ATTEMPT_JOURNAL.jsonl",
            "CASE_PROGRESS.jsonl",
        ],
        "postrun_gold_scoring": "Only after RUN_METADATA.status=COMPLETE and 40/40 paired rows are durable; scorer reads the GrailQA dev answer_ids and writes gold-only analysis. No gold fields enter live inference.",
        "primary_metric": "Question-level paired Hits@1 delta across all 40 intended cases; wins/losses/ties, net wins, paired bootstrap 95% interval.",
        "acceptance_gates": {
            "minimum_eligible_cases": 8,
            "positive_paired_hits_delta": True,
            "minimum_net_hits_wins": 3,
            "paired_bootstrap_ci_lower_gt_zero": True,
            "valid_nonempty_selector_rate_among_eligible_minimum": 0.95,
            "accepted_candidate_membership_minimum": 1.0,
            "baseline_correct_regressions_maximum": 0,
            "all_40_intended_cases_in_primary_denominator": True,
            "no_cost_query_timeout_retry_resume_or_integrity_breach": True,
        },
        "cost_basis": "400 requests * ((7000*0.20)+(1024*1.20))/1,000,000 = USD 1.05152 worst-case uncached rate reservation.",
        "cap_rationale": "Iteration 26 attempt 4 used 74 API requests and 109 physical SELECTs for 12 cases. Linear 40-case projection is approximately 247 API requests and 364 SELECTs before selector calls. Caps 400/650 include margin; two hours is the frozen maximum, not an expected-runtime guarantee.",
        "source_files_sha256": {relative: digest(relative) for relative in SOURCE_PATHS},
        "live_transport_counts_at_freeze": {"api_requests": 0, "physical_sparql_select_attempts": 0, "spend_usd": 0.0},
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path = OUT / "FROZEN_EVAL_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest_sha = _sha(manifest_path)
    (OUT / "FROZEN_EVAL_MANIFEST.sha256").write_text(f"{manifest_sha}  FROZEN_EVAL_MANIFEST.json\n", encoding="utf-8")
    return manifest_sha


if __name__ == "__main__":
    print(build())
