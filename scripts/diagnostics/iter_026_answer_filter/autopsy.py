"""Reconstruct the Iteration 14 selector pilot from immutable local artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
PREP = WORK / "RESULTS/iter_013_selector_pilot_preparation"
PILOT = WORK / "RESULTS/iter_014_selector_pilot"
OUT = WORK / "RESULTS/iter_026_answer_selector_readiness"
FRESH50_CONFIG = ROOT / "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256/run_config.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _candidate_ids(request: dict[str, Any]) -> tuple[list[str], list[str]]:
    user_text = request["messages"][-1]["content"]
    marker = "Candidates (ID and label):\n"
    encoded = user_text.split(marker, 1)[1].split("\n\nReturn", 1)[0]
    candidates = json.loads(encoded)
    return [row["id"] for row in candidates], [row["label"] for row in candidates]


def build_autopsy() -> dict[str, Any]:
    cases = {row["case_id"]: row for row in csv.DictReader((PILOT / "case_results.csv").open(newline="", encoding="utf-8"))}
    requests = {row["case_id"]: row for row in jsonl(PREP / "requests_non_gold.jsonl")}
    references = {row["case_id"]: row for row in jsonl(PREP / "archived_filter_reference_non_gold.jsonl")}
    outputs = {row["case_id"]: row for row in jsonl(PILOT / "selector_outputs_non_gold.jsonl")}
    scored = {row["case_id"]: row for row in jsonl(PILOT / "scored_results_gold_only.jsonl")}
    trace_rows = jsonl(PILOT / "call_trace.jsonl")
    trace_lines: dict[str, list[int]] = {}
    for line_number, record in enumerate(trace_rows, 1):
        trace_lines.setdefault(record["case_id"], []).append(line_number)
    if any(len(trace_lines.get(f"case_{i:02d}", [])) != 2 for i in range(1, 13)):
        raise RuntimeError("Iteration 14 call trace does not contain two archived journal rows per case")
    if not all(len(mapping) == 12 for mapping in (cases, requests, references, outputs, scored)):
        raise RuntimeError("Iteration 14 artifacts do not align to the expected 12 cases")

    rows = []
    for index in range(1, 13):
        case_id = f"case_{index:02d}"
        result, request, reference, output, gold = cases[case_id], requests[case_id], references[case_id], outputs[case_id], scored[case_id]
        candidate_ids, candidate_labels = _candidate_ids(request)
        gold_ids = set(gold["gold_answer_ids_scoring_only"])
        gold_present = bool(gold_ids & set(candidate_ids))
        accepted = output.get("answer_ids") or []
        validation = str(output.get("validation", "UNAVAILABLE"))
        api_error = output.get("api_error")
        is_empty = api_error == "EMPTY_RESPONSE"
        if is_empty:
            raw_class = "EMPTY_CONTENT_REPORTED; RAW_RESPONSE_NOT_PERSISTED"
            response_chars = "0_REPORTED_EMPTY_CONTENT"
            suspected = "EMPTY_CONTENT_AT_128_OUTPUT_TOKENS; TRUNCATION_NOT_CONFIRMED"
        elif validation == "VALID":
            raw_class = "NONEMPTY_CONTENT_PASSED_SAVED_SCHEMA_AND_MEMBERSHIP_CHECK; RAW_TEXT_NOT_PERSISTED"
            response_chars = "UNAVAILABLE_RAW_RESPONSE_NOT_PERSISTED"
            if accepted:
                suspected = "VALID_ID_OUTPUT; SEMANTIC_SELECTION_OUTCOME_OBSERVED; RAW_RATIONALE_UNAVAILABLE"
            else:
                suspected = "VALID_EMPTY_SELECTION_WITH_GOLD_PRESENT; SEMANTIC_ABSTENTION_OUTCOME_OBSERVED"
        else:
            raw_class = f"{validation}; RAW_RESPONSE_NOT_PERSISTED"
            response_chars = "UNAVAILABLE_RAW_RESPONSE_NOT_PERSISTED"
            suspected = "VALIDATION_OR_RESPONSE_FAILURE_CLASS_NOT_RECONSTRUCTABLE_FROM_RAW_TEXT"

        if result["cohort_role"] == "TGAR_FAILURE":
            baseline_status = "BASELINE_INCORRECT"
        else:
            baseline_status = "BASELINE_CORRECT_CONTROL"
        wrong_selected = bool(accepted) and not (set(accepted) & gold_ids)
        selected_labels = [candidate_labels[candidate_ids.index(entity_id)] for entity_id in accepted if entity_id in candidate_ids]
        raw_prompt = request["messages"][-1]["content"]
        filter_prompt = reference.get("archived_filter_prompt") or ""
        rows.append({
            "case_id": case_id,
            "question_key": result["question_key"],
            "cohort_role_scoring_only": result["cohort_role"],
            "baseline_status_scoring_only": baseline_status,
            "original_filter_parsed_output_non_gold": reference.get("archived_filter_parsed_result"),
            "candidate_count": len(candidate_ids),
            "gold_present_in_candidate_pool_diagnostic_only": str(gold_present).lower(),
            "gold_answer_count_diagnostic_only": len(gold_ids),
            "selector_raw_output_class": raw_class,
            "accepted_id_s": json.dumps(accepted, ensure_ascii=False),
            "selected_display_label_s_non_gold": json.dumps(selected_labels, ensure_ascii=False),
            "unselected_candidate_count": len(candidate_ids) - len(accepted),
            "invalid_or_rejected_id_s": "UNAVAILABLE_RAW_RESPONSE_NOT_PERSISTED",
            "parser_outcome": validation,
            "selector_hits_at_1_scoring_only": result["hits_at_1"],
            "selector_f1_scoring_only": result["f1"],
            "selector_output_token_count": result["output_tokens"],
            "response_length_chars": response_chars,
            "finish_reason": "NOT_RECORDED",
            "finish_metadata_available": "false",
            "selector_prompt_sha256": request["prompt_sha256"],
            "selector_prompt_included_original_reasoning_context": "false",
            "original_filter_prompt_sha256": reference.get("archived_filter_prompt_sha256"),
            "original_filter_prompt_included_reasoning_context": str("Reasoning Chain (SPARQL-style):" in filter_prompt).lower(),
            "selected_id_semantically_missed_gold_diagnostic_only": str(wrong_selected).lower(),
            "suspected_failure_class": suspected,
            "evidence_strength": "OBSERVED_OUTCOME; RAW_COMPLETION_UNAVAILABLE" if not is_empty else "OBSERVED_EMPTY_STATUS_AND_TOKEN_CAP; CAUSE_UNIDENTIFIED",
            "trace_link": (
                f"iter_014_selector_pilot/call_trace.jsonl:L{trace_lines[case_id][0]}-L{trace_lines[case_id][1]}; "
                f"iter_014_selector_pilot/selector_outputs_non_gold.jsonl:L{index}; "
                f"iter_014_selector_pilot/case_results.csv:case={case_id}; "
                f"iter_013_selector_pilot_preparation/requests_non_gold.jsonl:L{index}; "
                f"iter_013_selector_pilot_preparation/archived_filter_reference_non_gold.jsonl:L{index}; "
                f"iter_014_selector_pilot/scored_results_gold_only.jsonl:L{index}"
            ),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "OLD_SELECTOR_AUTOPSY.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    source_paths = [
        PREP / "requests_non_gold.jsonl", PREP / "archived_filter_reference_non_gold.jsonl",
        PREP / "scoring_gold_only.jsonl", PREP / "results.json",
        PILOT / "case_results.csv", PILOT / "call_trace.jsonl", PILOT / "selector_outputs_non_gold.jsonl",
        PILOT / "scored_results_gold_only.jsonl", PILOT / "run_metadata.json", PILOT / "results.json", PILOT / "report.md",
        ROOT / "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256/manifest_frozen.json",
        ROOT / "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256/run_config.json",
        ROOT / "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256/runs.jsonl",
        ROOT / "chain_of_relations/methods/adaptive_v1_1_3/agent.py",
        ROOT / "chain_of_relations/methods/adaptive_v1_1_3/router.py",
        ROOT / "chain_of_relations/methods/cor/agent.py",
        ROOT / "chain_of_relations/methods/cor/prompt.yml",
        ROOT / "chain_of_relations/methods/cor/tools/filter.py",
        ROOT / "scripts/diagnostics/prepare_fresh50_selector_pilot.py",
        ROOT / "scripts/diagnostics/run_fresh50_selector_pilot.py",
    ]
    try:
        import subprocess
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain=v1"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    except Exception:
        head, branch, status = "UNAVAILABLE", "UNAVAILABLE", []
    fresh50_config = json.loads(FRESH50_CONFIG.read_text(encoding="utf-8"))
    fresh50_fingerprint = fresh50_config.get("code_fingerprints", {})
    provenance = {
        "iteration": 26,
        "evidence_class": "RETROSPECTIVE_DEVELOPMENT",
        "source_files": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in source_paths if path.is_file()
        ],
        "repository_head": head,
        "branch": branch,
        "repository_status_at_autopsy_time": status,
        "fresh50_source_run_git_head": fresh50_fingerprint.get("git_head"),
        "fresh50_source_run_working_tree_dirty": fresh50_fingerprint.get("working_tree_dirty"),
        "fresh50_source_run_working_tree_status_count": fresh50_fingerprint.get("working_tree_status_count"),
        "pilot_model": "gpt-5.6-luna",
        "pilot_max_completion_tokens": 128,
        "pilot_timeout_seconds": 120,
        "pilot_retries": 0,
        "raw_model_completions_persisted": False,
        "finish_reason_persisted": False,
        "selector_request_contains_question_and_candidate_id_label_pairs": True,
        "selector_request_contains_prior_reasoning_context": False,
        "original_filter_request_contains_reasoning_chain": True,
        "case_count": len(rows),
        "empty_response_count": sum(row["parser_outcome"] == "API_ERROR" for row in rows),
        "valid_output_count": sum(row["parser_outcome"] == "VALID" for row in rows),
        "valid_failure_cases_with_wrong_or_empty_selection": sum(
            row["cohort_role_scoring_only"] == "TGAR_FAILURE" and row["parser_outcome"] == "VALID"
            and (row["selected_id_semantically_missed_gold_diagnostic_only"] == "true" or row["accepted_id_s"] == "[]")
            for row in rows
        ),
    }
    (OUT / "AUTOPSY_PROVENANCE.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return provenance


if __name__ == "__main__":
    result = build_autopsy()
    print(json.dumps({k: result[k] for k in ("case_count", "empty_response_count", "valid_output_count", "valid_failure_cases_with_wrong_or_empty_selection", "repository_head", "branch")}, indent=2))
