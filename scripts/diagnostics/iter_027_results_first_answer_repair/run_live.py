"""One-shot Iteration 27 paired evaluation for SCV-EmptyOnly.

The live path is inert unless an exact frozen manifest and explicit matching
authorization are supplied. It captures unchanged TGAR once per case, fsyncs
the baseline and candidate snapshot, and only then may send one SCV request for
an empty entity-only baseline. No gold/scoring data is opened by this process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.diagnostics.iter_027_results_first_answer_repair.scv import (  # noqa: E402
    build_scv_payload,
    eligible_for_scv_empty_only,
    validate_scv_response,
)
from scripts.diagnostics.iter_027_results_first_answer_repair.provenance import (  # noqa: E402
    observed_relation_provenance,
)

OUT = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_027_results_first_answer_repair"
COHORT = OUT / "INFERENCE_COHORT_NON_GOLD.jsonl"
COHORT_MANIFEST = OUT / "COHORT_MANIFEST.json"
MANIFEST = OUT / "FROZEN_EVAL_MANIFEST.json"
MANIFEST_SHA = OUT / "FROZEN_EVAL_MANIFEST.sha256"
CONSUMPTION_MARKER = OUT / "AUTHORIZATION_CONSUMPTION.json"
RUN_RELATIVE = "RUNS/iter27_scv_empty_only"
INPUT_PRICE = 0.20
OUTPUT_PRICE = 1.20
MAX_INPUT_TOKENS = 7000
MAX_COMPLETION_TOKENS = 1024


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_fsync(path: Path, row: dict[str, Any]) -> None:
    data = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    with path.open("ab", buffering=0) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _approved(manifest_sha: str, approval_sha: str) -> dict[str, Any]:
    frozen = MANIFEST_SHA.read_text(encoding="utf-8").split()[0]
    if manifest_sha != frozen or sha256_file(MANIFEST) != frozen:
        raise SystemExit("frozen manifest hash mismatch; no transport permitted")
    if approval_sha != frozen:
        raise SystemExit("explicit authorization must name the exact manifest SHA-256")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    _verify_frozen_inputs(manifest)
    return manifest


def _verify_frozen_inputs(manifest: dict[str, Any]) -> None:
    frozen = MANIFEST_SHA.read_text(encoding="utf-8").split()[0]
    if sha256_file(MANIFEST) != frozen:
        raise SystemExit("manifest sidecar mismatch")
    required = {
        "status": "FROZEN_AWAITING_EXPLICIT_AUTHORIZATION",
        "iteration": 27,
        "method": "SCV-EmptyOnly",
        "model": "gpt-5.6-luna",
        "api_request_cap": 400,
        "api_cost_cap_usd": 1.05152,
        "api_input_price_usd_per_million_uncached": INPUT_PRICE,
        "api_output_price_usd_per_million": OUTPUT_PRICE,
        "per_request_worst_case_reservation_usd": 0.0026288,
        "full_request_cap_worst_case_reservation_usd": 1.05152,
        "input_tokens_per_request": MAX_INPUT_TOKENS,
        "completion_tokens_per_request": MAX_COMPLETION_TOKENS,
        "sparql_attempt_cap": 650,
        "sparql_endpoint": "http://127.0.0.1:8890/sparql",
        "timeout_seconds_per_request": 120,
        "wall_clock_cap_seconds": 7200,
        "max_retries": 0,
        "resume_allowed": False,
        "run_directory": RUN_RELATIVE,
        "authorization_consumed": False,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise SystemExit("manifest differs from the frozen Iteration 27 protocol")
    if sha256_file(COHORT_MANIFEST) != manifest.get("cohort_manifest_sha256"):
        raise SystemExit("frozen cohort-manifest hash mismatch")
    if sha256_file(COHORT) != manifest.get("inference_cohort_sha256"):
        raise SystemExit("frozen inference cohort hash mismatch")
    for relative, expected in manifest.get("source_files_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"frozen source hash mismatch: {relative}")


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(manifest_sha: str, approval_sha: str) -> dict[str, Any]:
    manifest = _approved(manifest_sha, approval_sha)
    output = (OUT / manifest["run_directory"]).resolve()
    if output != (OUT / RUN_RELATIVE).resolve():
        raise SystemExit("output directory differs from the frozen manifest")
    if output.exists() or CONSUMPTION_MARKER.exists():
        raise SystemExit("one-shot authorization already consumed or output exists; no resume")
    rows = read_jsonl(COHORT)
    if len(rows) != 40 or [row["question_key"] for row in rows] != manifest["question_keys"]:
        raise SystemExit("cohort identities/order differ from the frozen manifest")
    forbidden = ("gold", "scoring", "answer_ids", "logical_form", "sparql_gold")
    if any(any(term in key.casefold() for term in forbidden) for row in rows for key in row):
        raise SystemExit("gold/scoring/answer field found in the inference cohort")
    if any(
        not isinstance(row.get("topic_entities"), list)
        or not isinstance(row.get("topic_entity_names"), list)
        or not row["topic_entities"]
        or len(row["topic_entities"]) != len(row["topic_entity_names"])
        for row in rows
    ):
        raise SystemExit("topic entity IDs/names missing or misaligned")

    # Heavy KGQA/model modules are imported only after frozen hashes and cohort
    # checks pass, and only on the explicitly authorized live path.
    from chain_of_relations.schema import Entity
    from scripts.diagnostics.iter_026_answer_filter.gateway_attempt4 import (
        Attempt4Gateway,
        InputTokenLimitExceeded,
    )
    from scripts.diagnostics.iter_026_answer_filter.run_live_attempt4 import (
        _baseline_ids,
        _selector_eligibility,
        make_attempt4_tgar_agent,
    )
    from scripts.diagnostics.iter_026_answer_filter.strict_runtime_attempt4 import (
        answer_case_once,
        selector_response_metadata,
    )
    from scripts.diagnostics.iter_026_answer_filter.transport import (
        FreebaseSelectTransport,
        OpenAIChatCompletionsTransport,
    )

    # Read secrets only after the manifest and every frozen hash have passed.
    from scripts.run_cor_baseline_experiment import load_safe_project_env
    load_safe_project_env()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY unavailable; no live call started")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")
    if base_url and base_url != "https://api.openai.com/v1":
        raise SystemExit("frozen transport requires the official OpenAI API base URL")

    OUT.mkdir(parents=True, exist_ok=True)
    marker = {
        "iteration": 27,
        "manifest_sha256": manifest_sha,
        "status": "CONSUMED_BEFORE_PREFLIGHT; ONE_SHOT_NO_RESUME",
        "run_directory": RUN_RELATIVE,
        "recorded_time_epoch": time.time(),
    }
    descriptor = os.open(CONSUMPTION_MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(marker, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(OUT, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    output.mkdir(parents=True, exist_ok=False)

    gateway = Attempt4Gateway(
        output,
        manifest_sha256=manifest_sha,
        model_transport=OpenAIChatCompletionsTransport(api_key, timeout_seconds=120),
        sparql_transport=FreebaseSelectTransport(timeout_seconds=120),
        api_request_cap=manifest["api_request_cap"],
        api_cost_cap_usd=manifest["api_cost_cap_usd"],
        input_tokens_per_request=MAX_INPUT_TOKENS,
        output_tokens_per_request=MAX_COMPLETION_TOKENS,
        input_usd_per_million=INPUT_PRICE,
        output_usd_per_million=OUTPUT_PRICE,
        sparql_attempt_cap=manifest["sparql_attempt_cap"],
        timeout_seconds=120,
        wall_seconds=7200,
    )
    metadata: dict[str, Any] = {
        "iteration": 27,
        "manifest_sha256": manifest_sha,
        "cohort_sha256": sha256_file(COHORT),
        "method": "SCV-EmptyOnly",
        "model": manifest["model"],
        "status": "RUNNING",
        "evidence_class": "FRESH_GRAILQA_DEV_DEVELOPMENT_EVALUATION",
        "gold_loaded_by_live_process": False,
        "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(output / "RUN_METADATA.json", metadata)
    traces = output / "CASE_TRACES.jsonl"
    paired = output / "PAIRED_RESULTS_NON_GOLD.jsonl"
    snapshots = output / "FROZEN_CASE_SNAPSHOTS_NON_GOLD.jsonl"
    progress = output / "CASE_PROGRESS.jsonl"
    completed: list[dict[str, Any]] = []
    active_case: str | None = None
    try:
        agent = make_attempt4_tgar_agent(gateway)
        preflight = gateway.select_once("SELECT * WHERE { ?s ?p ?o } LIMIT 1")
        _write_json(output / "PREFLIGHT.json", {
            "status": "HTTP_OK", "binding_count": len(preflight["bindings"]),
            "raw_response_sha256": preflight.get("raw_response_sha256"),
        })
        for item in rows:
            if gateway.halted:
                raise RuntimeError("gateway halted; later cases will not be attempted")
            active_case = item["case_id"]
            gateway.set_case_context(active_case)
            append_fsync(progress, {"event": "CASE_STARTED", "case_id": active_case})
            entities = [
                Entity(str(entity_id), str(label))
                for entity_id, label in zip(item["topic_entities"], item["topic_entity_names"])
                if str(entity_id).strip() and str(label).strip()
            ]
            prompt_history: list[dict[str, Any]] = []
            sparql_history: list[dict[str, Any]] = []
            step_history: list[dict[str, Any]] = []
            api_start, select_start = gateway.api_starts, gateway.sparql_starts
            token_in_start, token_out_start = gateway.api_input_tokens, gateway.api_output_tokens
            cost_start, reserve_start = gateway.api_actual_cost_usd, gateway.api_reserved_usd
            case_start = time.monotonic()
            baseline, failure = answer_case_once(
                agent, question=item["question"], topic_entities=entities,
                prompt_history=prompt_history, sparql_history=sparql_history,
                step_history=step_history,
            )
            if failure is None:
                baseline_status = "BASELINE_COMPLETED"
            elif failure.get("failure_kind") == "INPUT_CAP_EXCEEDED":
                baseline_status = "BASELINE_INPUT_CAP_EXCEEDED_NO_OUTPUT"
                baseline = {"results": [], "router_execution_mode": "cor", "error": baseline_status}
            else:
                baseline_status = "UPSTREAM_EMPTY_OR_TRUNCATED_FAILURE"
                baseline = {"results": [], "router_execution_mode": "cor", "error": baseline_status}
            baseline_ids = _baseline_ids(baseline)
            filters = [event for event in prompt_history if event.get("type") == "filter_answer"]
            route = str(baseline.get("router_execution_mode", "cor"))
            legacy_eligible, legacy_reason, filter_event, candidates = _selector_eligibility(
                baseline, item["question"], prompt_history, sparql_history
            ) if failure is None else (False, baseline_status, None, [])
            eligible, reason = eligible_for_scv_empty_only(
                baseline_status=baseline_status,
                baseline_entity_only=legacy_eligible,
                baseline_prediction_ids=baseline_ids,
                router_execution_mode=route,
                final_filter_event_count=len(filters),
                candidates=candidates,
            )
            # Durable paired baseline and exact candidate state before treatment.
            snapshot_row = {
                "case_id": active_case,
                "question_key": item["question_key"],
                "baseline_status": baseline_status,
                "baseline_prediction_ids": baseline_ids,
                "router_execution_mode": route,
                "legacy_entity_filter_gate": legacy_reason,
                "scv_eligible": eligible,
                "scv_eligibility_reason": reason,
                "filter_trace_id": filter_event.get("trace_id") if filter_event else None,
                "filter_context": filter_event.get("prompt") if filter_event else None,
                "candidate_snapshot": candidates,
                "candidate_snapshot_sha256": manifest_hash_candidates(candidates),
                "observed_relation_provenance": observed_relation_provenance(
                    sparql_history, int(filter_event["trace_id"])
                ) if filter_event and str(filter_event.get("trace_id", "")).isdigit() else [],
            }
            append_fsync(snapshots, snapshot_row)

            intervention_ids = list(baseline_ids)
            scv_row: dict[str, Any] | None = None
            if eligible and filter_event is not None:
                relation_evidence = [
                    relation
                    for source in snapshot_row["observed_relation_provenance"]
                    for relation in source["observed_relation_ids_in_executed_query"]
                ]
                payload = build_scv_payload(
                    question=item["question"],
                    frozen_filter_context=str(filter_event["prompt"]),
                    router_execution_mode=route,
                    filter_trace_id=int(filter_event["trace_id"]),
                    selected_relation_evidence=relation_evidence,
                    candidates=candidates,
                )
                messages = [{
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                }]
                try:
                    response = gateway.call_model_once(
                        messages,
                        max_completion_tokens=MAX_COMPLETION_TOKENS,
                        reasoning_effort="low",
                    )
                    checked = validate_scv_response(
                        response.get("content"), candidates=candidates,
                        expected_candidate_snapshot_sha256=snapshot_row["candidate_snapshot_sha256"],
                        finish_reason=response.get("finish_reason"),
                    )
                    if checked["status"] == "VALID_SELECTION":
                        intervention_ids = list(checked["selected_ids"])
                    scv_row = {
                        **checked,
                        "api_request_sent": True,
                        "response_metadata": selector_response_metadata(response),
                        "candidate_membership_valid": checked["status"] == "VALID_SELECTION"
                        and set(checked["selected_ids"]).issubset({candidate["id"] for candidate in candidates}),
                    }
                except InputTokenLimitExceeded as exc:
                    scv_row = {
                        "status": "SKIPPED_INPUT_CAP_EXCEEDED",
                        "valid": False,
                        "selected_ids": [],
                        "prediction_source": "UNCHANGED_EMPTY_BASELINE",
                        "error": exc.metadata(),
                        "api_request_sent": False,
                        "candidate_membership_valid": False,
                    }
            case = {
                "case_id": active_case,
                "question_key": item["question_key"],
                "dataset": item["dataset"],
                "baseline_prediction_ids": baseline_ids,
                "intervention_prediction_ids": intervention_ids,
                "baseline_status": baseline_status,
                "router_execution_mode": route,
                "filter_event_count": len(filters),
                "scv_eligible": eligible,
                "scv_eligibility_reason": reason,
                "candidate_ids_snapshot": [candidate["id"] for candidate in candidates],
                "candidate_snapshot_sha256": snapshot_row["candidate_snapshot_sha256"],
                "scv_invoked": scv_row is not None,
                "scv_api_request_sent": bool(scv_row and scv_row.get("api_request_sent")),
                "scv": scv_row,
                "prediction_source": scv_row["prediction_source"] if scv_row else "UNCHANGED_BASELINE",
                "api_requests_started": gateway.api_starts - api_start,
                "api_input_tokens": gateway.api_input_tokens - token_in_start,
                "api_output_tokens": gateway.api_output_tokens - token_out_start,
                "api_actual_cost_usd": gateway.api_actual_cost_usd - cost_start,
                "api_reserved_cost_usd": gateway.api_reserved_usd - reserve_start,
                "sparql_attempts_started": gateway.sparql_starts - select_start,
                "latency_seconds": time.monotonic() - case_start,
                "baseline_error": baseline.get("error"),
                "upstream_failure_metadata": failure,
            }
            append_fsync(traces, {
                **case,
                "prompt_history": prompt_history,
                "sparql_history": sparql_history,
                "step_history": step_history,
            })
            append_fsync(paired, case)
            completed.append(case)
            append_fsync(progress, {"event": "CASE_FINISHED", "case_id": active_case})
            active_case = None
            gateway.set_case_context(None)
            if gateway.halted:
                raise RuntimeError("gateway halted; one-shot run stops without retries")
    except BaseException as exc:
        metadata.update({
            "status": "ABORTED_NO_RESUME",
            "evidence_class": "INCOMPLETE_LIVE_ATTEMPT",
            "cases_completed": len(completed),
            "current_case_id": active_case,
            "abort_error_type": type(exc).__name__,
            "api_requests_started": gateway.api_starts,
            "api_input_tokens": gateway.api_input_tokens,
            "api_output_tokens": gateway.api_output_tokens,
            "api_actual_cost_usd": gateway.api_actual_cost_usd,
            "api_reserved_cost_usd": gateway.api_reserved_usd,
            "physical_sparql_attempts": gateway.sparql_starts,
        })
        _write_json(output / "RUN_METADATA.json", metadata)
        raise
    metadata.update({
        "status": "COMPLETE" if len(completed) == len(rows) else "INCOMPLETE_NO_RESUME",
        "evidence_class": "FRESH_GRAILQA_DEV_DEVELOPMENT_EVALUATION",
        "cases_started": len(rows),
        "cases_completed": len(completed),
        "eligible_cases": sum(case["scv_eligible"] for case in completed),
        "selector_requests": sum(case["scv_api_request_sent"] for case in completed),
        "api_requests_started": gateway.api_starts,
        "api_input_tokens": gateway.api_input_tokens,
        "api_output_tokens": gateway.api_output_tokens,
        "api_actual_cost_usd": gateway.api_actual_cost_usd,
        "api_reserved_cost_usd": gateway.api_reserved_usd,
        "physical_sparql_attempts": gateway.sparql_starts,
        "retries": 0,
        "resumed": False,
        "gold_loaded_by_live_process": False,
        "completed_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    _write_json(output / "RUN_METADATA.json", metadata)
    return metadata


def manifest_hash_candidates(candidates: list[dict[str, Any]]) -> str | None:
    if not candidates:
        return None
    from scripts.diagnostics.iter_026_answer_filter.selector import candidate_snapshot_sha256
    return candidate_snapshot_sha256(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify frozen hashes and cohort; perform no transport")
    parser.add_argument("--run", action="store_true", help="run only after the same manifest hash is explicitly authorized")
    parser.add_argument("--manifest-sha256", default="")
    parser.add_argument("--authorize-manifest-sha256", default="")
    args = parser.parse_args()
    if args.check == args.run:
        parser.error("choose exactly one of --check or --run")
    if args.check:
        frozen = MANIFEST_SHA.read_text(encoding="utf-8").split()[0]
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        _verify_frozen_inputs(manifest)
        rows = read_jsonl(COHORT)
        if len(rows) != 40 or [row["question_key"] for row in rows] != manifest.get("question_keys"):
            raise SystemExit("cohort identities/order differ from frozen manifest")
        print(json.dumps({
            "status": "FROZEN_INPUTS_AND_SOURCES_OK_NO_TRANSPORT",
            "manifest_sha256": frozen,
            "question_count": len(read_jsonl(COHORT)),
            "api_requests": 0,
            "physical_sparql_select_attempts": 0,
            "spend_usd": 0.0,
        }, indent=2))
        return
    if not args.manifest_sha256 or not args.authorize_manifest_sha256:
        parser.error("--run requires --manifest-sha256 and --authorize-manifest-sha256")
    print(json.dumps(run(args.manifest_sha256, args.authorize_manifest_sha256), indent=2))


if __name__ == "__main__":
    main()
