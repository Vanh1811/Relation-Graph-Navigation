"""Rebuild historical candidate/cache coverage without network or model calls."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
WORK = REPO / "outputs/experiments/fresh50_continuous_agent"
SOURCE = WORK / "RESULTS/iter_019_depth4_holdout_preparation/control_continuation/paired_runs_cumulative.jsonl"
CACHE = WORK / "RESULTS/iter_022_gated_matched_prefix/LOCAL_LABEL_CACHE.json"
PAIRS = WORK / "RESULTS/iter_019_depth4_holdout_preparation/control_continuation/paired_case_results_complete.csv"
ITER22_CASES = WORK / "RESULTS/iter_022_gated_matched_prefix/PILOT_CASES.csv"
ITER23_CASES = WORK / "RESULTS/iter_023_label_coverage_and_prefix_readiness/CASE_USAGE_AUDIT.csv"
OUT = WORK / "RESULTS/iter_025_cache_compatible_readiness/CANDIDATE_COVERAGE.csv"
SUMMARY_OUT = WORK / "RESULTS/iter_025_cache_compatible_readiness/COVERAGE_SUMMARY.json"
SELECTION_SEED = "iteration25-cache-prefix-readiness-2026-11-08"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _role_by_question(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = row["question_key"]
            role = row.get("cohort_role_scoring_only", "")
            previous = result.setdefault(key, role)
            if previous != role:
                raise ValueError(f"inconsistent scoring-only cohort role for {key}")
    return result


def _consumed_slots() -> dict[tuple[str, int], str]:
    consumed: dict[tuple[str, int], str] = {}
    for path, iteration in ((ITER22_CASES, 22), (ITER23_CASES, 23)):
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = f"cwq::{row['question_id']}"
                repeat = int(row["repeat"])
                slot = (key, repeat)
                if slot in consumed:
                    raise ValueError(f"duplicate consumed slot {slot}")
                consumed[slot] = f"ITERATION_{iteration}"
    return consumed


def build_inventory(source: Path = SOURCE, cache_path: Path = CACHE,
                    pair_path: Path = PAIRS) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return one deterministic row for each historical question identity.

    The profile unions only inference-time entity-search IDs from archived
    depth-3 rows. Gold/scoring fields are never read from the run archive.
    Cohort role and paired outcomes are read separately for offline screening.
    """
    rows = load_jsonl(source)
    cache_doc = json.loads(cache_path.read_text(encoding="utf-8"))
    cache_list = cache_doc.get("labels")
    if not isinstance(cache_list, list):
        raise ValueError("expected label cache to preserve its provenance-bearing list format")
    cache: dict[str, str] = {}
    for entry in cache_list:
        if not isinstance(entry, dict) or not isinstance(entry.get("entity_id"), str) or not isinstance(entry.get("label"), str):
            raise ValueError("malformed trusted cache entry")
        if entry["entity_id"] in cache and cache[entry["entity_id"]] != entry["label"]:
            raise ValueError(f"conflicting cache labels for {entry['entity_id']}")
        cache[entry["entity_id"]] = entry["label"]

    roles = _role_by_question(pair_path)
    consumed = _consumed_slots()
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for line_number, row in enumerate(rows, 1):
        if row.get("diagnostic_depth") == 3:
            groups[str(row["question_key"])].append((line_number, row))

    pair_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    with pair_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            d3 = int(row["depth3_hits_at_1"])
            d4 = int(row["depth4_hits_at_1"])
            if d4 > d3:
                label = "D4_ONLY"
            elif d3 > d4:
                label = "D3_ONLY"
            else:
                label = "TIE"
            pair_outcomes[row["question_key"]][label] += 1

    output: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "archive_rows_total": len(rows),
        "depth3_rows": sum(len(v) for v in groups.values()),
        "question_count": len(groups),
        "label_cache_unique_ids": len(cache),
        "label_cache_provenance": cache_doc.get("provenance"),
        "label_cache_manifest_sha256": cache_doc.get("manifest_sha256"),
        "label_cache_source_sha256_recorded": cache_doc.get("source_archive_sha256"),
        "archive_sha256": sha256_file(source),
        "cache_sha256": sha256_file(cache_path),
        "paired_results_sha256": sha256_file(pair_path),
        "questions": [],
    }
    for question_key in sorted(groups):
        records = sorted(groups[question_key], key=lambda x: (int(x[1].get("repeat_index", 0)), x[0]))
        first = records[0][1]
        unique_ids: set[str] = set()
        appearances = 0
        candidate_name_pairs: dict[str, set[str]] = defaultdict(set)
        pointers: list[dict[str, Any]] = []
        entity_searches = 0
        malformed_pair_count = 0
        empty_name_count = 0
        path_queries: list[str] = []
        stage_counts: Counter[str] = Counter()
        action_values: list[str] = []
        forward_boundary_count = 0
        forward_action_count = 0
        repeats: set[int] = set()
        prior_slots = sorted(
            f"{iteration}:repeat{repeat}" for (key, repeat), iteration in consumed.items() if key == question_key
        )
        exact_state_fields_observed: set[str] = set()
        for line_number, record in records:
            repeat = int(record.get("repeat_index", 0))
            repeats.add(repeat)
            for event in record.get("sparql_history", []) or []:
                if event.get("search_type") != "entity_search":
                    continue
                entity_searches += 1
                ids = event.get("candidate_entity_ids") or []
                names = event.get("candidate_entity_names") or []
                if len(ids) != len(names):
                    malformed_pair_count += abs(len(ids) - len(names)) or 1
                for index, entity_id in enumerate(ids):
                    if not isinstance(entity_id, str) or not entity_id:
                        malformed_pair_count += 1
                        continue
                    unique_ids.add(entity_id)
                    appearances += 1
                    name = names[index] if index < len(names) else None
                    if isinstance(name, str) and name.strip():
                        candidate_name_pairs[entity_id].add(name)
                    else:
                        empty_name_count += 1
                query = event.get("sparql", "")
                query_hash = sha256_bytes(query.encode("utf-8")) if isinstance(query, str) else ""
                if query_hash:
                    path_queries.append(query_hash)
                pointers.append({
                    "source": str(source.relative_to(REPO)),
                    "line": line_number,
                    "repeat": repeat,
                    "depth": record.get("diagnostic_depth"),
                    "trace_id": event.get("trace_id"),
                    "query_sha256": query_hash,
                    "result_count": event.get("result_count"),
                    "status": event.get("status"),
                })
            for event in record.get("state_lineage", []) or []:
                stage_counts[str(event.get("stage", "UNKNOWN"))] += 1
                for field in ("dfs_stack", "active_bindings", "binding_entity_ids", "frontier_entity_ids"):
                    if field in event:
                        exact_state_fields_observed.add(field)
                action = event.get("action")
                if action is not None:
                    action_values.append(str(action))
                    if action == "Forward":
                        forward_action_count += 1
                        if len(event.get("relation_chain") or []) == 3:
                            forward_boundary_count += 1
            for step in record.get("step_history", []) or []:
                for field in ("action", "parsed_action", "decision"):
                    if field in step:
                        action_values.append(str(step[field]))

        covered = sorted(unique_ids & cache.keys())
        uncovered = sorted(unique_ids - cache.keys())
        name_conflicts = sorted(entity_id for entity_id, names in candidate_name_pairs.items() if len(names) > 1)
        cache_label_mismatches = sorted(
            entity_id for entity_id, label in cache.items()
            if entity_id in candidate_name_pairs and label not in candidate_name_pairs[entity_id]
        )
        outcomes = pair_outcomes.get(question_key, Counter())
        role = roles.get(question_key, str(first.get("cohort_role_scoring_only", "UNKNOWN")))
        d4_only = outcomes.get("D4_ONLY", 0)
        d3_only = outcomes.get("D3_ONLY", 0)
        ties = outcomes.get("TIE", 0)
        question = str(first.get("question", ""))
        info = {
            "question_key": question_key,
            "dataset": first.get("dataset"),
            "question_id": first.get("question_id"),
            "role": role,
            "d4_only_pairs": d4_only,
            "d3_only_pairs": d3_only,
            "tie_pairs": ties,
            "d3_run_count": len(records),
            "repeat_indices": sorted(repeats),
            "prospective_repeat_index": 3,
            "prospective_repeat3_absent_from_archive": 3 not in repeats,
            "previous_authorization_slots_consumed": prior_slots,
            "prospective_repeat3_slot_unconsumed": (question_key, 3) not in consumed and 3 not in repeats,
            "entity_search_events": entity_searches,
            "candidate_id_appearances": appearances,
            "candidate_unique_ids": len(unique_ids),
            "cache_covered_ids": len(covered),
            "cache_coverage_fraction": (len(covered) / len(unique_ids)) if unique_ids else None,
            "uncovered_ids": uncovered,
            "uncovered_ids_count": len(uncovered),
            "candidate_name_conflicts": name_conflicts,
            "cache_label_content_mismatches": cache_label_mismatches,
            "malformed_id_name_pair_count": malformed_pair_count,
            "empty_or_missing_candidate_name_count": empty_name_count,
            "forward_action_events_recorded": sum(v == "Forward" for v in action_values),
            "forward_action_events_total": forward_action_count,
            "forward_depth3_boundary_events": forward_boundary_count,
            "action_fields_recorded": bool(action_values),
            "lineage_stage_counts": dict(sorted(stage_counts.items())),
            "exact_prefix_fields_observed": sorted(exact_state_fields_observed),
            "exact_prefix_state_complete": all(field in exact_state_fields_observed for field in ("dfs_stack", "active_bindings", "frontier_entity_ids")),
            "path_query_sha256s": sorted(set(path_queries)),
            "trace_pointers": pointers,
            "question_text_sha256": sha256_bytes(question.encode("utf-8")),
            "screening_status": "REJECT_FOR_EXACT_PREFIX_PILOT" if not all(field in exact_state_fields_observed for field in ("dfs_stack", "active_bindings", "frontier_entity_ids")) else "REVIEW",
            "screening_reason": "Archived state lacks ordered DFS stack and active binding/frontier ID tuples required to freeze a matched prefix; cache coverage is historical only.",
        }
        info["technical_eligible"] = bool(
            info["exact_prefix_state_complete"]
            and info["candidate_unique_ids"] > 0
            and info["cache_covered_ids"] == info["candidate_unique_ids"]
            and not name_conflicts
            and not cache_label_mismatches
            and malformed_pair_count == 0
            and empty_name_count == 0
            and forward_boundary_count > 0
        )
        info["deterministic_tie_break_sha256"] = sha256_bytes(f"{SELECTION_SEED}\0{question_key}".encode("utf-8"))
        output.append(info)
        diagnostics["questions"].append({k: info[k] for k in (
            "question_key", "role", "d4_only_pairs", "d3_only_pairs", "tie_pairs",
            "candidate_unique_ids", "cache_covered_ids", "uncovered_ids_count", "exact_prefix_state_complete",
        )})

    if len(output) != 30:
        raise ValueError(f"expected 30 unique historical questions, found {len(output)}")
    if any(len(item["repeat_indices"]) != 2 for item in output):
        raise ValueError("expected two archived D3 repeat records per question")
    ranked = sorted(output, key=lambda item: (
        not item["technical_eligible"],
        not item["exact_prefix_state_complete"],
        not (item["candidate_unique_ids"] > 0 and item["cache_covered_ids"] == item["candidate_unique_ids"]),
        bool(item["candidate_name_conflicts"] or item["cache_label_content_mismatches"]),
        -item["forward_depth3_boundary_events"],
        item["deterministic_tie_break_sha256"],
    ))
    for rank, item in enumerate(ranked, 1):
        item["technical_rank"] = rank
    diagnostics["role_counts"] = dict(Counter(item["role"] for item in output))
    diagnostics["exact_prefix_complete_cases"] = sum(item["exact_prefix_state_complete"] for item in output)
    diagnostics["coverage_distribution"] = {
        "all_d3_cache_complete": sum(item["candidate_unique_ids"] > 0 and item["cache_covered_ids"] == item["candidate_unique_ids"] for item in output),
        "zero_coverage": sum(item["candidate_unique_ids"] > 0 and item["cache_covered_ids"] == 0 for item in output),
        "partial_coverage": sum(0 < item["cache_covered_ids"] < item["candidate_unique_ids"] for item in output),
    }
    diagnostics["previous_authorization_consumed_slots"] = len(consumed)
    diagnostics["prospective_repeat3_slots_absent_and_unconsumed"] = sum(item["prospective_repeat3_slot_unconsumed"] for item in output)
    diagnostics["technical_eligible_candidates"] = sum(item["technical_eligible"] for item in output)
    diagnostics["selection_seed"] = SELECTION_SEED
    return output, diagnostics


def write_inventory(path: Path = OUT) -> dict[str, Any]:
    inventory, summary = build_inventory()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "question_key", "dataset", "question_id", "cohort_role_scoring_only", "question_text_sha256",
        "depth4_only_pairs", "depth3_only_pairs", "tie_pairs", "depth3_archived_runs", "depth3_repeat_indices",
        "prospective_repeat_index", "prospective_repeat3_absent_from_archive",
        "previous_authorization_slots_consumed_json", "prospective_repeat3_slot_unconsumed",
        "technical_eligible", "technical_rank", "deterministic_tie_break_sha256",
        "depth3_entity_search_event_count", "candidate_id_appearances", "candidate_unique_id_count",
        "cache_covered_unique_id_count", "cache_coverage_fraction", "uncovered_ids_json", "uncovered_id_count",
        "candidate_name_conflicts_json", "cache_label_content_mismatches_json", "malformed_id_name_pair_count",
        "empty_or_missing_candidate_name_count", "forward_action_events_total", "forward_depth3_boundary_events",
        "action_fields_recorded",
        "lineage_stage_counts_json", "exact_prefix_fields_observed_json", "exact_prefix_state_complete",
        "path_query_sha256s_json", "trace_pointers_json", "screening_status", "screening_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in inventory:
            writer.writerow({
                "question_key": item["question_key"], "dataset": item["dataset"], "question_id": item["question_id"],
                "cohort_role_scoring_only": item["role"], "question_text_sha256": item["question_text_sha256"],
                "depth4_only_pairs": item["d4_only_pairs"], "depth3_only_pairs": item["d3_only_pairs"], "tie_pairs": item["tie_pairs"],
                "depth3_archived_runs": item["d3_run_count"], "depth3_repeat_indices": json.dumps(item["repeat_indices"]),
                "prospective_repeat_index": item["prospective_repeat_index"],
                "prospective_repeat3_absent_from_archive": str(item["prospective_repeat3_absent_from_archive"]).lower(),
                "previous_authorization_slots_consumed_json": json.dumps(item["previous_authorization_slots_consumed"], separators=(",", ":")),
                "prospective_repeat3_slot_unconsumed": str(item["prospective_repeat3_slot_unconsumed"]).lower(),
                "technical_eligible": str(item["technical_eligible"]).lower(),
                "technical_rank": item["technical_rank"],
                "deterministic_tie_break_sha256": item["deterministic_tie_break_sha256"],
                "depth3_entity_search_event_count": item["entity_search_events"], "candidate_id_appearances": item["candidate_id_appearances"],
                "candidate_unique_id_count": item["candidate_unique_ids"], "cache_covered_unique_id_count": item["cache_covered_ids"],
                "cache_coverage_fraction": "" if item["cache_coverage_fraction"] is None else f"{item['cache_coverage_fraction']:.6f}",
                "uncovered_ids_json": json.dumps(item["uncovered_ids"], separators=(",", ":")), "uncovered_id_count": item["uncovered_ids_count"],
                "candidate_name_conflicts_json": json.dumps(item["candidate_name_conflicts"], separators=(",", ":")),
                "cache_label_content_mismatches_json": json.dumps(item["cache_label_content_mismatches"], separators=(",", ":")),
                "malformed_id_name_pair_count": item["malformed_id_name_pair_count"],
                "empty_or_missing_candidate_name_count": item["empty_or_missing_candidate_name_count"],
                "forward_action_events_total": item["forward_action_events_total"],
                "forward_depth3_boundary_events": item["forward_depth3_boundary_events"],
                "action_fields_recorded": str(item["action_fields_recorded"]).lower(),
                "lineage_stage_counts_json": json.dumps(item["lineage_stage_counts"], sort_keys=True, separators=(",", ":")),
                "exact_prefix_fields_observed_json": json.dumps(item["exact_prefix_fields_observed"], separators=(",", ":")),
                "exact_prefix_state_complete": str(item["exact_prefix_state_complete"]).lower(),
                "path_query_sha256s_json": json.dumps(item["path_query_sha256s"], separators=(",", ":")),
                "trace_pointers_json": json.dumps(item["trace_pointers"], sort_keys=True, separators=(",", ":")),
                "screening_status": item["screening_status"], "screening_reason": item["screening_reason"],
            })
    summary["candidate_inventory_sha256"] = sha256_file(path)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = write_inventory()
    print(json.dumps({k: v for k, v in result.items() if k != "questions"}, indent=2, sort_keys=True))
