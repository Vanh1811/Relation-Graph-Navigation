"""Prepare Iteration 27 offline artifacts; no API, SPARQL, or gold use for sampling.

The cohort builder reads only GrailQA IDs, question text, split and topic-entity
fields for selection. Gold labels are opened only in the separate historical
Iteration 26 case-matrix function, after that cohort had already completed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_027_results_first_answer_repair"
I26 = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness"
RUN = I26 / "RUNS/iter26_live_attempt4"
DATA = ROOT / "data/processed/grailqa_dev_analysis.jsonl"
SAMPLE_N = 40
SELECTION_SEED = "iter27-scv-empty-only-grailqa-dev-v1"
NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.80


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def historical_grailqa_ids() -> set[str]:
    """Use dataset-qualified IDs in prior output records, not arbitrary numbers."""
    result = subprocess.run(
        ["rg", "-l", r"grailqa::[0-9]{8,15}", "outputs/"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("could not inventory prior GrailQA identity references")
    ids: set[str] = set()
    for relative in result.stdout.splitlines():
        path = ROOT / relative
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        ids.update(re.findall(r"grailqa::([0-9]{8,15})", text))
    return ids


def historical_question_texts(known_ids: set[str], source_rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Collect prior question text for identity/lexical checks; never read answers."""
    texts: list[tuple[str, str]] = []
    by_id = {str(row.get("id", "")): row for row in source_rows}
    for question_id in sorted(known_ids):
        row = by_id.get(question_id)
        if row and isinstance(row.get("question"), str):
            texts.append((row["question"], f"data/processed/grailqa_dev_analysis.jsonl#{question_id}"))

    # Include questions already materialized in bounded experiment outputs (and
    # CWQ wording where available); ignore copied source_artifacts to avoid
    # treating an untouched source population as previously studied examples.
    for path in (ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS").rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
            continue
        if "source_artifacts" in path.parts or path.stat().st_size > 5_000_000:
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if path.suffix.lower() == ".csv":
            try:
                for row in csv.DictReader(raw.splitlines()):
                    for key in ("question", "question_text", "utterance"):
                        value = row.get(key)
                        if value and len(value.split()) >= 3:
                            texts.append((value, str(path.relative_to(ROOT))))
            except csv.Error:
                pass
            continue
        lines = raw.splitlines() if path.suffix.lower() == ".jsonl" else [raw]
        for line in lines:
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            stack = [obj]
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key.casefold() in {"question", "question_text", "utterance"} and isinstance(child, str) and len(child.split()) >= 3:
                            texts.append((child, str(path.relative_to(ROOT))))
                        elif isinstance(child, (dict, list)):
                            stack.append(child)
                elif isinstance(value, list):
                    stack.extend(value)
    dedup: dict[str, tuple[str, str]] = {}
    for text, source in texts:
        key = normalized(text)
        if key:
            dedup.setdefault(key, (text, source))
    return list(dedup.values())


def build_cohort() -> dict[str, Any]:
    source_rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    previous_ids = historical_grailqa_ids()
    prior_text = historical_question_texts(previous_ids, source_rows)
    prior_norm = {normalized(text) for text, _ in prior_text}
    prior_tokens = [(words(text), text, source) for text, source in prior_text]
    ranked: list[tuple[str, dict[str, Any]]] = []
    missing_topic_entity_metadata = 0
    for row in source_rows:
        question_id = str(row.get("id", ""))
        question = row.get("question")
        if not question_id or not isinstance(question, str) or not question.strip():
            continue
        if str(row.get("split", "")).casefold() != "dev" or question_id in previous_ids:
            continue
        topic_entities = list(row.get("topic_entities") or [])
        topic_entity_names = list(row.get("topic_entity_names") or [])
        if not topic_entities or len(topic_entities) != len(topic_entity_names):
            missing_topic_entity_metadata += 1
            continue
        rank = hashlib.sha256(f"{SELECTION_SEED}:{question_id}".encode()).hexdigest()
        ranked.append((rank, {
            "question_id": question_id,
            "question": question,
            "split": str(row.get("split", "dev")),
            "topic_entities": topic_entities,
            "topic_entity_names": topic_entity_names,
        }))
    ranked.sort(key=lambda item: (item[0], item[1]["question_id"]))
    selected: list[dict[str, Any]] = []
    screened = 0
    overlap_records: list[dict[str, Any]] = []
    for rank, candidate in ranked:
        screened += 1
        text = candidate["question"]
        norm = normalized(text)
        tokens = words(text)
        if norm in prior_norm:
            overlap_records.append({"question_id": candidate["question_id"], "reason": "NORMALIZED_EXACT_PRIOR_TEXT"})
            continue
        duplicate_of = None
        for prior_words, prior_question, prior_source in prior_tokens:
            union = tokens | prior_words
            jaccard = len(tokens & prior_words) / len(union) if union else 1.0
            if jaccard >= NEAR_DUPLICATE_JACCARD_THRESHOLD:
                duplicate_of = {"prior_source": prior_source, "jaccard": round(jaccard, 6)}
                break
        if duplicate_of:
            overlap_records.append({"question_id": candidate["question_id"], "reason": "HIGH_LEXICAL_OVERLAP", **duplicate_of})
            continue
        candidate["selection_rank_sha256"] = rank
        candidate["max_prior_question_token_jaccard"] = round(max(
            (len(tokens & old) / len(tokens | old) for old, _, _ in prior_tokens if tokens | old), default=0.0
        ), 6)
        selected.append(candidate)
        if len(selected) == SAMPLE_N:
            break
    if len(selected) != SAMPLE_N:
        raise RuntimeError(f"only {len(selected)} clean questions found; do not silently reduce cohort")

    inference_rows = []
    for index, row in enumerate(selected, 1):
        if not row["topic_entities"] or len(row["topic_entities"]) != len(row["topic_entity_names"]):
            raise RuntimeError(f"topic entity IDs/names missing or misaligned for {row['question_id']}")
        inference_rows.append({
            "case_id": f"iter27_case_{index:02d}",
            "question_key": f"grailqa::{row['question_id']}",
            "dataset": "grailqa",
            "source_split": "dev",
            "question": row["question"],
            "topic_entities": row["topic_entities"],
            "topic_entity_names": row["topic_entity_names"],
        })
    inference_bytes = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode() for row in inference_rows)
    (OUT / "INFERENCE_COHORT_NON_GOLD.jsonl").write_bytes(inference_bytes)
    manifest = {
        "iteration": 27,
        "evidence_class": "FROZEN_PREAUTHORIZATION_DEVELOPMENT_COHORT; NO_TRANSPORT",
        "status": "FROZEN_COHORT_NO_BASELINE_CANDIDATE_SNAPSHOTS; LIVE_COLLECTION_REQUIRES_NEW_AUTHORIZATION",
        "sample_size": SAMPLE_N,
        "source_dataset": "GrailQA dev",
        "source_dataset_path": "data/processed/grailqa_dev_analysis.jsonl",
        "source_dataset_sha256": sha256_file(DATA),
        "selection_seed": SELECTION_SEED,
        "selection_policy": "Stable SHA-256 rank over all GrailQA dev IDs after exact historical-ID exclusion and normalized-exact/high-token-Jaccard duplicate screens. Selection did not inspect answer IDs/text, gold query/logical form, annotated structure, route predictions, candidate coverage, or correctness. The sample is unstratified; route/cardinality strata are descriptive after inference/scoring.",
        "historical_grailqa_question_id_exclusion_count": len(previous_ids),
        "remaining_source_rows_excluded_for_missing_topic_metadata": missing_topic_entity_metadata,
        "historical_question_texts_available_for_lexical_screen": len(prior_text),
        "identity_or_lexical_exclusions_before_selection": overlap_records,
        "candidates_ranked_before_selection": len(ranked),
        "ranked_candidates_screened_until_sample_filled": screened,
        "high_threshold_token_jaccard_cutoff": NEAR_DUPLICATE_JACCARD_THRESHOLD,
        "selected_question_keys": [row["question_key"] for row in inference_rows],
        "selected_questions": [
            {"case_id": row["case_id"], "question_key": row["question_key"],
             "selection_rank_sha256": selected[index]["selection_rank_sha256"],
             "max_prior_question_token_jaccard": selected[index]["max_prior_question_token_jaccard"]}
            for index, row in enumerate(inference_rows)
        ],
        "inference_cohort_path": "INFERENCE_COHORT_NON_GOLD.jsonl",
        "inference_cohort_sha256": sha256_bytes(inference_bytes),
        "gold_fields_in_inference_file": False,
        "semantic_paraphrase_identity_proven": False,
        "candidate_snapshots_available_before_live_collection": 0,
        "live_collection_required": True,
        "transport_counts": {"api_requests": 0, "physical_sparql_select_attempts": 0, "spend_usd": 0.0},
    }
    (OUT / "COHORT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest


def request_payloads(request_rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    pending: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    labels_by_case: dict[str, list[dict[str, Any]]] = {}
    selector_output: dict[str, dict[str, Any]] = {}
    for row in request_rows:
        case_id = row.get("case_id")
        if row.get("event") == "REQUEST_STARTED":
            body = row.get("request_body") or {}
            messages = body.get("messages") or []
            payload = None
            for message in messages:
                content = message.get("content", "") if isinstance(message, dict) else ""
                try:
                    value = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(value, dict) and isinstance(value.get("candidate_entities"), list) and value.get("instructions"):
                    payload = value
                    break
            if payload is not None:
                pending[row.get("request_sha256", "")] = (case_id, payload["candidate_entities"])
                labels_by_case[case_id] = payload["candidate_entities"]
        elif row.get("event") == "REQUEST_FINISHED":
            key = row.get("request_sha256", "")
            if key in pending:
                case_id, _ = pending.pop(key)
                selector_output[case_id] = {
                    "content": row.get("content"),
                    "finish_reason": row.get("finish_reason"),
                    "status": row.get("status"),
                    "usage": row.get("usage"),
                    "request_sha256": key,
                    "response_sha256": row.get("raw_response_sha256"),
                }
    return labels_by_case, selector_output


def build_i26_matrix() -> None:
    paired = {row["case_id"]: row for row in read_jsonl(RUN / "PAIRED_RESULTS_NON_GOLD.jsonl")}
    traces = {row["case_id"]: row for row in read_jsonl(RUN / "CASE_TRACES.jsonl")}
    gold = {row["case_id"]: row for row in read_jsonl(I26 / "ATTEMPT4_PREPARATION/SCORING_GOLD_ONLY.jsonl")}
    requests = read_jsonl(RUN / "REQUEST_JOURNAL.jsonl")
    labels, selector_outputs = request_payloads(requests)
    score = json.loads((RUN / "PAIRED_ANALYSIS_GOLD_ONLY.json").read_text())
    scored = {row["case_id"]: row for row in score["scored_cases"]}
    raw_paths = {
        "paired_results": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/RUNS/iter26_live_attempt4/PAIRED_RESULTS_NON_GOLD.jsonl",
        "case_traces": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/RUNS/iter26_live_attempt4/CASE_TRACES.jsonl",
        "request_journal": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/RUNS/iter26_live_attempt4/REQUEST_JOURNAL.jsonl",
        "sparql_journal": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/RUNS/iter26_live_attempt4/SPARQL_QUERY_JOURNAL.jsonl",
        "gold_scoring_sidecar": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/ATTEMPT4_PREPARATION/SCORING_GOLD_ONLY.jsonl",
        "inference_cohort": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_026_answer_selector_readiness/ATTEMPT4_PREPARATION/INFERENCE_COHORT_NON_GOLD.jsonl",
    }
    source_hashes = {key: sha256_file(ROOT / path) for key, path in raw_paths.items()}
    (OUT / "ITER26_SOURCE_HASHES.json").write_text(json.dumps({"paths": raw_paths, "sha256": source_hashes}, indent=2, sort_keys=True) + "\n")
    fieldnames = [
        "evidence_class", "case_id", "question_key", "question", "answer_cardinality_scoring_only",
        "branch_family_scoring_only", "router_execution_mode", "baseline_status", "baseline_prediction_ids",
        "selector_eligible", "eligibility_reason", "candidate_pool_kind", "candidate_ids_and_labels",
        "gold_answer_ids_scoring_only", "gold_in_selector_candidate_set_scoring_only",
        "baseline_correct_scoring_only", "selector_correct_scoring_only", "paired_hits", "delta_f1",
        "selector_status", "selector_finish_reason", "selector_response_exact", "schema_membership_validation",
        "first_supported_divergence", "constraint_and_provenance_evidence", "same_upstream_snapshot",
        "baseline_input_cap_failure", "baseline_input_tokens", "api_requests", "input_tokens", "output_tokens",
        "api_cost_usd", "physical_sparql_attempts", "case_latency_seconds", "source_paths_and_sha256",
    ]
    with (OUT / "DEV_DIAGNOSTIC_CASE_MATRIX.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for case_id, row in paired.items():
            trace = traces[case_id]
            gold_row = gold[case_id]
            score_row = scored[case_id]
            candidate_ids = list(row.get("candidate_ids_snapshot") or [])
            candidate_labels = labels.get(case_id, [])
            candidate_pool_kind = "FROZEN_SELECTOR_POOL" if candidate_ids else "NO_SELECTOR_POOL_CAPTURED"
            if not candidate_ids:
                graph_results = [e for e in trace.get("sparql_history", []) if e.get("candidate_entity_ids")]
                if graph_results:
                    terminal = graph_results[-1]
                    candidate_ids = list(terminal.get("candidate_entity_ids") or [])
                    candidate_labels = [
                        {"id": entity_id, "label": label}
                        for entity_id, label in zip(candidate_ids, terminal.get("candidate_entity_names") or [])
                    ]
                    candidate_pool_kind = "LAST_GRAPH_SEARCH_ONLY_NOT_SELECTOR_ELIGIBLE"
            if labels.get(case_id) and row.get("candidate_ids_snapshot"):
                candidate_labels = labels[case_id]
            pool_ids = [str(item.get("id")) for item in candidate_labels if isinstance(item, dict)]
            gold_ids = list(gold_row.get("gold_answer_ids_scoring_only") or [])
            selector = row.get("selector") or {}
            selector_response = selector_outputs.get(case_id, {})
            if selector.get("api_request_sent") and row.get("baseline_prediction_ids") != row.get("intervention_prediction_ids"):
                divergence = "SAME_FROZEN_UPSTREAM; FIRST_OBSERVED_DIVERGENCE_IS_SELECTOR_CHOICE"
            elif selector.get("status") == "AMBIGUOUS_DISPLAY_LABEL":
                divergence = "SELECTOR_RESPONSE_REJECTED_FOR_AMBIGUOUS_DUPLICATE_LABEL; BASELINE_FALLBACK"
            elif selector.get("api_request_sent"):
                divergence = "SAME_FROZEN_UPSTREAM; SELECTOR_CHOICE_DID_NOT_CHANGE_FINAL_IDS"
            elif row.get("execution_status") == "BASELINE_INPUT_CAP_EXCEEDED_NO_OUTPUT":
                divergence = "BASELINE_PROMPT_REJECTED_LOCALLY_BEFORE_TRANSPORT; NO_PAIRED_INFERENCE"
            else:
                divergence = f"NO_SELECTOR_CALL:{row.get('selector_eligibility_reason')}"
            history = trace.get("prompt_history", [])
            relation_outputs = [
                e.get("parsed_result") for e in history
                if "relation" in str(e.get("type", "")).casefold() and e.get("parsed_result") is not None
            ]
            provenance = {
                "filter_context_sha256": sha256_bytes(str(next((e.get("prompt", "") for e in history if e.get("type") == "filter_answer"), "")).encode()),
                "entity_search_trace_ids": [e.get("trace_id") for e in trace.get("sparql_history", []) if e.get("search_type") == "entity_search"],
                "terminal_search_type": next((e.get("search_type") for e in reversed(trace.get("sparql_history", [])) if e.get("candidate_entity_ids")), None),
                "relation_model_outputs": relation_outputs,
                "limitations": "Per-candidate support for every natural-language constraint is not independently established by the captured filter snapshot.",
            }
            source_json = {"paths": raw_paths, "sha256": source_hashes}
            writer.writerow({
                "evidence_class": "ITER26_ATTEMPT4_DEVELOPMENT_ONLY; GOLD FIELDS SCORING-ONLY",
                "case_id": case_id,
                "question_key": row["question_key"],
                "question": next((x.get("question") for x in read_jsonl(I26 / "ATTEMPT4_PREPARATION/INFERENCE_COHORT_NON_GOLD.jsonl") if x["case_id"] == case_id), ""),
                "answer_cardinality_scoring_only": gold_row.get("answer_cardinality_stratum_scoring_only"),
                "branch_family_scoring_only": gold_row.get("branch_family_scoring_only"),
                "router_execution_mode": row.get("router_execution_mode"),
                "baseline_status": row.get("execution_status"),
                "baseline_prediction_ids": json.dumps(row.get("baseline_prediction_ids") or [], ensure_ascii=False),
                "selector_eligible": row.get("selector_eligible"),
                "eligibility_reason": row.get("selector_eligibility_reason"),
                "candidate_pool_kind": candidate_pool_kind,
                "candidate_ids_and_labels": json.dumps(candidate_labels, ensure_ascii=False),
                "gold_answer_ids_scoring_only": json.dumps(gold_ids, ensure_ascii=False),
                "gold_in_selector_candidate_set_scoring_only": score_row.get("gold_present_in_candidates_diagnostic_only") if row.get("selector_eligible") else "NOT_APPLICABLE_NO_FROZEN_SELECTOR_POOL",
                "baseline_correct_scoring_only": bool(score_row.get("baseline", {}).get("hits_at_1")),
                "selector_correct_scoring_only": bool(score_row.get("intervention", {}).get("hits_at_1")),
                "paired_hits": score_row.get("paired_hits"),
                "delta_f1": score_row.get("f1_delta"),
                "selector_status": selector.get("status"),
                "selector_finish_reason": selector_response.get("finish_reason"),
                "selector_response_exact": selector_response.get("content"),
                "schema_membership_validation": json.dumps({"valid": selector.get("valid"), "candidate_membership_valid": selector.get("candidate_membership_valid"), "error": selector.get("error")}, ensure_ascii=False),
                "first_supported_divergence": divergence,
                "constraint_and_provenance_evidence": json.dumps(provenance, ensure_ascii=False),
                "same_upstream_snapshot": bool(row.get("selector_invoked")),
                "baseline_input_cap_failure": json.dumps(row.get("upstream_failure_metadata")) if row.get("upstream_failure_metadata") else "",
                "baseline_input_tokens": (row.get("upstream_failure_metadata") or {}).get("estimated_input_tokens", ""),
                "api_requests": row.get("api_requests_started"),
                "input_tokens": row.get("api_input_tokens"),
                "output_tokens": row.get("api_output_tokens"),
                "api_cost_usd": row.get("api_actual_cost_usd"),
                "physical_sparql_attempts": row.get("sparql_attempts_started"),
                "case_latency_seconds": row.get("latency_seconds"),
                "source_paths_and_sha256": json.dumps(source_json, ensure_ascii=False),
            })

    # Historical counterfactual: the frozen empty-only trigger is evaluated on
    # attempt-4 traces, without model calls and without tuning on the three
    # gold-labeled win/loss outcomes.
    eligible_ids = []
    for case_id, row in paired.items():
        if not row.get("selector_eligible") or row.get("execution_status") != "BASELINE_COMPLETED":
            continue
        if row.get("baseline_prediction_ids"):
            continue
        eligible_ids.append(case_id)
    counterfactual = {
        "status": "OFFLINE_COUNTERFACTUAL_ONLY_NO_LIVE_SELECTOR_CALLS",
        "evidence_class": "ITER26_DEVELOPMENT_REPLAY; NOT AN EVALUATION OF MODEL OUTPUTS",
        "trigger": "completed CoR baseline; exactly one final entity filter; frozen nonempty candidate pool with provenance; baseline entity prediction is empty",
        "attempt4_cases": len(paired),
        "eligible_cases_under_empty_only_trigger": eligible_ids,
        "eligible_case_count": len(eligible_ids),
        "selector_requests_would_have_been_sent": 0,
        "paired_changes_observed": 0,
        "paired_hits": {"wins": 0, "losses": 0, "ties": len(paired)},
        "interpretation": "Attempt 4 contains no completed case satisfying the empty-baseline trigger with the required frozen selector boundary. The rule would make no changes on this cohort; this is a coverage result, not evidence that the selector succeeds or fails when invoked.",
    }
    (OUT / "DEV_EMPTY_ONLY_COUNTERFACTUAL.json").write_text(json.dumps(counterfactual, indent=2, sort_keys=True) + "\n")


def build_candidate_coverage(cohort: dict[str, Any]) -> None:
    fields = ["evidence_class", "case_id", "question_key", "source_split", "candidate_snapshot_available_before_live", "baseline_answer_available_before_live", "historical_identity_match", "gold_diagnostic_status", "required_next_capture"]
    with (OUT / "UPSTREAM_CANDIDATE_COVERAGE.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in read_jsonl(OUT / "INFERENCE_COHORT_NON_GOLD.jsonl"):
            writer.writerow({
                "evidence_class": "GOLD_FREE_PREAUTHORIZATION_INVENTORY",
                "case_id": row["case_id"], "question_key": row["question_key"], "source_split": row["source_split"],
                "candidate_snapshot_available_before_live": False,
                "baseline_answer_available_before_live": False,
                "historical_identity_match": False,
                "gold_diagnostic_status": "NOT_ACCESSED_BEFORE_FROZEN_CANDIDATE_AND_BASELINE",
                "required_next_capture": "One unchanged TGAR trajectory; persist exact final candidate IDs/labels/provenance and baseline output before SCV; gold scoring only after all live predictions are frozen.",
            })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = build_cohort()
    build_candidate_coverage(cohort)
    build_i26_matrix()
    print(json.dumps({
        "cohort_size": cohort["sample_size"],
        "historical_ids_excluded": cohort["historical_grailqa_question_id_exclusion_count"],
        "candidate_snapshots_available_for_fresh_cohort": cohort["candidate_snapshots_available_before_live_collection"],
        "transport": cohort["transport_counts"],
        "outputs": ["COHORT_MANIFEST.json", "INFERENCE_COHORT_NON_GOLD.jsonl", "UPSTREAM_CANDIDATE_COVERAGE.csv", "DEV_DIAGNOSTIC_CASE_MATRIX.csv", "DEV_EMPTY_ONLY_COUNTERFACTUAL.json", "ITER26_SOURCE_HASHES.json"],
    }, indent=2))


if __name__ == "__main__":
    main()
