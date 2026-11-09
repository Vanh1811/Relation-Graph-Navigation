#!/usr/bin/env python3
"""Prepare (but never submit) the fixed-candidate 7+5 selector pilot inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
INPUT = WORK / "RESULTS/iter_001_id_schema_probe/selector_inputs_non_gold.jsonl"
SCORING = WORK / "RESULTS/iter_001_id_schema_probe/scoring_labels_gold_only.jsonl"
RUNS = ROOT / "outputs/experiments/fresh50_strong_branch_cor_vs_v113_vs_p256/runs.jsonl"
DEFAULT_OUT = WORK / "RESULTS/iter_013_selector_pilot_preparation"

SYSTEM = (
    "Select the entity IDs that directly answer the question from the supplied candidates. "
    "The candidate list is the complete allowed answer set. Select all and only candidates "
    "that answer the question; preserve multiple valid answers when the question asks for them. "
    "Do not invent or normalize entities. If no candidate answers the question, return an empty list. "
    'Return only JSON in this schema: {"answer_ids": ["candidate-id", ...]}.'
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)

    source_inputs = read_jsonl(INPUT)
    scoring = {row["question_key"]: row for row in read_jsonl(SCORING)}
    runs = {
        row["question_key"]: row
        for row in read_jsonl(RUNS)
        if row.get("method") == "v1_1_3"
    }
    if len(source_inputs) != 12:
        raise RuntimeError(f"Expected 12 fixed-candidate cases, got {len(source_inputs)}")

    request_rows = []
    scoring_rows = []
    reference_rows = []
    counts: dict[str, int] = {}
    for index, item in enumerate(source_inputs, start=1):
        key = item["question_key"]
        ids = list(item.get("candidate_entity_ids") or [])
        names = list(item.get("candidate_entity_names") or [])
        if not ids or len(ids) != len(names):
            raise RuntimeError(f"Invalid candidate ID/name mapping for {key}: {len(ids)} IDs, {len(names)} names")
        candidates = [{"id": entity_id, "label": label} for entity_id, label in zip(ids, names)]
        candidate_names = [candidate["label"] for candidate in candidates]

        run = runs[key]
        filters = [
            (event_index, event)
            for event_index, event in enumerate(run.get("prompt_history", []))
            if event.get("type") == "filter_answer"
        ]
        matching = [
            (event_index, event)
            for event_index, event in filters
            if all(label in (event.get("prompt") or "") for label in candidate_names)
        ]
        if not matching:
            raise RuntimeError(f"Could not align archived selector prompt to exact frozen candidate pool: {key}")
        original_index, original = matching[-1]
        archived_answers = item.get("archived_filter_answers", [])
        if str(original.get("parsed_result")) not in {str(value) for value in archived_answers}:
            raise RuntimeError(f"Archived filter output mismatch for aligned candidate pool: {key}")

        case_id = f"case_{index:02d}"
        user_content = (
            f"Question: {item['question']}\n\n"
            "Candidates (ID and label):\n"
            f"{json.dumps(candidates, ensure_ascii=False, separators=(',', ':'))}\n\n"
            "Return the selected candidate IDs using the required JSON schema."
        )
        request_rows.append(
            {
                "case_id": case_id,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                "max_completion_tokens": 128,
                "temperature": "OMIT_USE_MODEL_DEFAULT",
                "timeout_seconds": 120,
                "retry_limit": 0,
                "allowed_candidate_ids": ids,
                "prompt_sha256": hashlib.sha256(user_content.encode()).hexdigest(),
            }
        )
        role = item["role"]
        counts[role] = counts.get(role, 0) + 1
        scoring_rows.append(
            {
                "case_id": case_id,
                "question_key": key,
                "cohort_role_scoring_only": role,
                "gold_answer_ids_scoring_only": scoring[key]["gold_answer_ids"],
            }
        )
        reference_rows.append(
            {
                "case_id": case_id,
                "question_key": key,
                "source_candidate_count": len(candidates),
                "candidate_ids_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()
                ).hexdigest(),
                "archived_filter_event_index": original_index,
                "archived_filter_prompt_sha256": hashlib.sha256(
                    (original.get("prompt") or "").encode()
                ).hexdigest(),
                "archived_filter_prompt": original.get("prompt"),
                "archived_filter_response": original.get("response"),
                "archived_filter_parsed_result": original.get("parsed_result"),
            }
        )

    with (out / "requests_non_gold.jsonl").open("w", encoding="utf-8") as stream:
        for row in request_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "scoring_gold_only.jsonl").open("w", encoding="utf-8") as stream:
        for row in scoring_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "archived_filter_reference_non_gold.jsonl").open("w", encoding="utf-8") as stream:
        for row in reference_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    result = {
        "experiment": "offline preparation of fixed-candidate non-oracle ID selector pilot",
        "claim_scope": "Inputs are prepared and candidate pools aligned to archived final filter prompts; no request was submitted and no accuracy result is claimed.",
        "command": "PYTHONPATH=src:. .venv/bin/python scripts/diagnostics/prepare_fresh50_selector_pilot.py",
        "case_count": len(request_rows),
        "cohort_counts_scoring_only": counts,
        "all_candidate_id_name_pairs_valid": True,
        "all_archived_candidate_pools_prompt_aligned": True,
        "input_tokens": "not billed; tokenization not run",
        "llm_calls": 0,
        "read_only_kg_queries": 0,
        "paid_cost_usd": 0.0,
        "required_live_limits": {
            "RESEARCH_MAX_LLM_CALLS": 12,
            "RESEARCH_MAX_LLM_COST_USD": "must be explicitly set to a positive user-approved value",
            "MODEL_NAME": "use existing project configuration",
            "max_completion_tokens_per_call": 128,
            "timeout_seconds_per_call": 120,
            "retries": 0,
        },
        "development_gate": "rescue at least 5/7 failures and lose 0/5 both-correct controls; this does not establish generalization",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {
            "selector_inputs_non_gold.jsonl": sha256(INPUT),
            "scoring_labels_gold_only.jsonl": sha256(SCORING),
            "official_runs.jsonl": sha256(RUNS),
        },
    }
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    hashes = {
        path.name: sha256(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
