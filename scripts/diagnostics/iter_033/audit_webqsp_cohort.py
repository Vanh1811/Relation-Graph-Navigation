#!/usr/bin/env python3
"""Offline source, exposure, and frozen-trigger audit for WebQSP test.

This script never calls OpenAI or a SPARQL endpoint. The official WebQSP ZIP
is read-only input (default: /tmp/webqsp-official.zip); no cohort manifest is
created because this source has too few questions matching Iteration 32's
frozen count-intent gate after conservative historical exclusions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS"
LOCAL_WEBQSP = ROOT / "datasets/webqsp/webqsp.json"
OUTDIR = RESULTS / "iter_033_scalar_generalization_and_upstream_decision"
OUT = OUTDIR / "WEBQSP_COHORT_AUDIT.json"
sys.path.insert(0, str(ROOT))
from chain_of_relations.methods.iter_030_operator_recovery.relation_aware_count import classify_count_intent  # noqa: E402

TEXT_EXTENSIONS = {".csv", ".json", ".jsonl", ".md", ".txt", ".log", ".tsv"}
QID_PATTERN = re.compile(r"WebQTest-\d+", re.IGNORECASE)
QUESTION_KEYS = {"question", "rawquestion", "questiontext", "question_text", "raw_question", "utterance"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def add_question_texts(value: Any, out: set[str], key: str = "") -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            add_question_texts(v, out, str(k).casefold())
    elif isinstance(value, list):
        for item in value:
            add_question_texts(item, out, key)
    elif isinstance(value, str) and (key in QUESTION_KEYS or key.endswith("_question")):
        text = value.strip()
        if len(text) >= 12:
            out.add(text)


def archived_questions_and_ids(root: Path) -> tuple[set[str], set[str], int, int]:
    questions: set[str] = set()
    ids: set[str] = set()
    file_count = byte_count = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in TEXT_EXTENSIONS:
            continue
        if "iter_033_scalar_generalization_and_upstream_decision" in path.parts:
            continue
        try:
            raw = path.read_bytes()
            content = raw.decode("utf-8", errors="ignore")
        except OSError:
            continue
        file_count += 1
        byte_count += len(raw)
        ids.update(match.group(0) for match in QID_PATTERN.finditer(content))
        suffix = path.suffix.casefold()
        if suffix == ".csv" or suffix == ".tsv":
            try:
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    delimiter = "\t" if suffix == ".tsv" else ","
                    for row in csv.DictReader(stream, delimiter=delimiter):
                        for key, value in row.items():
                            if value and (key or "").casefold() in QUESTION_KEYS | {"question_text", "raw_question"}:
                                text = value.strip()
                                if len(text) >= 12:
                                    questions.add(text)
            except (OSError, csv.Error):
                pass
        elif suffix in {".json", ".jsonl"}:
            try:
                if suffix == ".json":
                    add_question_texts(json.loads(content), questions)
                else:
                    for line in content.splitlines():
                        if not line.strip():
                            continue
                        try:
                            add_question_texts(json.loads(line), questions)
                        except json.JSONDecodeError:
                            continue
            except (json.JSONDecodeError, UnicodeError):
                pass
        elif suffix in {".md", ".txt", ".log"}:
            for match in re.finditer(r"(?im)^\s*(?:[-*]\s*)?(?:raw\s+)?question\s*[:=]\s*(.{12,})$", content):
                questions.add(match.group(1).strip())
    return questions, ids, file_count, byte_count


def parse_official_zip(zip_path: Path) -> tuple[str, bytes, str, bytes, list[dict[str, Any]], bytes]:
    zip_hash = sha256_file(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        data_name = next(name for name in names if name.endswith("WebQSP.test.json"))
        evaluator_name = next(name for name in names if name.endswith("/eval.py"))
        raw = archive.read(data_name)
        evaluator = archive.read(evaluator_name)
        data = json.loads(raw)
    return zip_hash, raw, data_name, evaluator, data["Questions"], evaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-zip", type=Path, default=Path("/tmp/webqsp-official.zip"))
    args = parser.parse_args()
    if not args.official_zip.exists():
        raise SystemExit(f"official WebQSP source ZIP not found: {args.official_zip}")

    zip_hash, official_raw, official_name, evaluator, official_questions, _ = parse_official_zip(args.official_zip)
    local = json.loads(LOCAL_WEBQSP.read_text(encoding="utf-8"))
    official_by_id = {row["QuestionId"]: row for row in official_questions}
    local_by_id = {row["id"]: row for row in local}
    if len(local_by_id) != len(local) or len(official_by_id) != len(official_questions):
        raise SystemExit("duplicate WebQSP IDs in local or official source")
    if set(local_by_id) != set(official_by_id):
        raise SystemExit("local/official WebQSP test ID sets differ")

    valid_parses_by_id: dict[str, list[dict[str, Any]]] = {}
    empty_gold_ids: list[str] = []
    multi_parse_ids: list[str] = []
    answer_union_mismatch_ids: list[str] = []
    local_answer_exactly_matches_any_parse_ids: list[str] = []
    topic_name_mismatch: list[dict[str, str]] = []
    topic_id_not_found: list[str] = []
    question_mismatch_ids: list[str] = []
    local_empty_answer_ids: list[str] = []
    for qid, local_row in local_by_id.items():
        official_row = official_by_id[qid]
        if local_row.get("question") != official_row.get("RawQuestion"):
            question_mismatch_ids.append(qid)
        parses = [
            parse for parse in official_row.get("Parses", [])
            if (parse.get("AnnotatorComment") or {}).get("ParseQuality") == "Complete"
            and (parse.get("AnnotatorComment") or {}).get("QuestionQuality") == "Good"
        ]
        valid_parses_by_id[qid] = parses
        if not parses:
            raise SystemExit(f"no official Good+Complete parse for {qid}")
        if len(parses) > 1:
            multi_parse_ids.append(qid)
        all_answers = [answer for parse in parses for answer in parse.get("Answers", [])]
        if not all_answers:
            empty_gold_ids.append(qid)
        local_answers = {str(answer.get("id", "")) for answer in local_row.get("answer", []) if answer.get("id") is not None}
        official_union = {str(answer.get("AnswerArgument", "")) for answer in all_answers if answer.get("AnswerArgument") is not None}
        if local_answers != official_union:
            answer_union_mismatch_ids.append(qid)
        if local_answers in [
            {str(answer.get("AnswerArgument", "")) for answer in parse.get("Answers", []) if answer.get("AnswerArgument") is not None}
            for parse in parses
        ]:
            local_answer_exactly_matches_any_parse_ids.append(qid)
        if not local_answers:
            local_empty_answer_ids.append(qid)
        topic = (local_row.get("topic_entity") or [{}])[0]
        candidate_ids = {parse.get("TopicEntityMid") for parse in parses}
        if topic.get("id") not in candidate_ids:
            topic_id_not_found.append(qid)
        else:
            official_names = {parse.get("TopicEntityName") for parse in parses if parse.get("TopicEntityMid") == topic.get("id")}
            if topic.get("name") not in official_names:
                topic_name_mismatch.append({"question_id": qid, "topic_id": str(topic.get("id")), "local_name": str(topic.get("name")), "official_names": sorted(str(x) for x in official_names)})

    archived_questions, archived_ids, archive_files, archive_bytes = archived_questions_and_ids(RESULTS)
    local_ids = set(local_by_id)
    exposed_ids = sorted(local_ids & archived_ids, key=lambda value: int(value.split("-")[-1]))
    # Full historical archive scan is intentionally conservative: any ID
    # mentioned in an earlier artifact is excluded regardless of artifact role.
    candidate_ids = sorted(local_ids - set(exposed_ids), key=lambda value: int(value.split("-")[-1]))
    normalized_history = {normalize(q): q for q in archived_questions}
    exact_question_ids = []
    near_duplicate_matches = []
    for qid in candidate_ids:
        question = local_by_id[qid]["question"]
        norm = normalize(question)
        if norm in normalized_history:
            exact_question_ids.append(qid)
            continue
        best_score = 0
        best_question = ""
        for prior in archived_questions:
            score = fuzz.ratio(norm, normalize(prior))
            if score > best_score:
                best_score, best_question = score, prior
                if best_score == 100:
                    break
        if best_score >= 90:
            near_duplicate_matches.append({"question_id": qid, "score": best_score, "prior_question": best_question})
    clean_ids = set(candidate_ids) - set(exact_question_ids) - {row["question_id"] for row in near_duplicate_matches}
    count_intent_ids = []
    for qid in sorted(clean_ids, key=lambda value: int(value.split("-")[-1])):
        triggered, _reason = classify_count_intent(local_by_id[qid]["question"])
        if triggered:
            count_intent_ids.append(qid)

    evaluator_text = evaluator.decode("utf-8", errors="replace")
    evaluator_policy = {
        "best_parse_f1": "best per-question F1 over the official valid gold parses",
        "empty_gold_empty_prediction": "F1 1.0 per official CalculatePRF1; precision/recall convention inspect evaluator source",
        "empty_gold_nonempty_prediction": "F1 0.0 per official CalculatePRF1",
        "empty_gold_ids": len(empty_gold_ids),
        "evaluator_contains_empty_gold_branch": "no labeled answer" in evaluator_text,
    }
    out = {
        "status": "NO_CLEAN_INDEPENDENT_COHORT",
        "iteration": 33,
        "network_calls": 0,
        "api_requests": 0,
        "sparql_attempts": 0,
        "official_source": {
            "zip_path_at_audit": str(args.official_zip),
            "zip_sha256": zip_hash,
            "data_member": official_name,
            "data_member_sha256": sha256_bytes(official_raw),
            "evaluator_member_sha256": sha256_bytes(evaluator),
            "local_webqsp_sha256": sha256_file(LOCAL_WEBQSP),
            "official_version": "1.0",
            "freebase_version": "2015-08-09",
        },
        "source_crosswalk": {
            "official_test_rows": len(official_questions),
            "local_rows": len(local),
            "ids_exactly_equal": True,
            "question_text_exact_matches": len(local) - len(question_mismatch_ids),
            "all_rows_have_good_complete_parse": True,
            "multi_good_complete_parse_rows": len(multi_parse_ids),
            "empty_official_answer_rows": len(empty_gold_ids),
            "local_answer_empty_rows": len(local_empty_answer_ids),
            "local_flat_answer_projection_differs_from_official_parse_union": len(answer_union_mismatch_ids),
            "local_answer_projection_exactly_matches_at_least_one_official_parse": len(local_answer_exactly_matches_any_parse_ids),
            "topic_entity_ids_found_in_official_parse": len(local) - len(topic_id_not_found),
            "topic_label_mismatches": topic_name_mismatch,
            "evaluator_semantics": evaluator_policy,
        },
        "exposure_scan": {
            "scope": str(RESULTS.relative_to(ROOT)),
            "iteration_033_excluded": True,
            "historical_text_files_scanned": archive_files,
            "historical_bytes_scanned": archive_bytes,
            "unique_prior_question_strings": len(archived_questions),
            "unique_prior_question_strings_sha256": sha256_bytes("\n".join(sorted(normalize(q) for q in archived_questions)).encode("utf-8")),
            "question_id_overlaps_excluded": exposed_ids,
            "question_id_overlap_count": len(exposed_ids),
            "exact_normalized_question_overlaps_in_remainder": exact_question_ids,
            "near_duplicate_threshold_rapidfuzz_ratio_gte_90": near_duplicate_matches,
            "clean_by_exposure_remainder": len(clean_ids),
        },
        "frozen_trigger_feasibility": {
            "rule_source": "Iteration 30 classify_count_intent; unchanged, gold-free question-text classifier",
            "trigger_positive_questions": count_intent_ids,
            "trigger_positive_count": len(count_intent_ids),
            "serializer_acceptance_prevalence_reference": "39/800 = 4.875% development-only planning reference; not assumed for WebQSP",
            "max_available_frozen_trigger_positive_questions": len(count_intent_ids),
            "if_all_four_were_correct_exact_two_sided_95pct_precision_lower_bound": 0.025 ** 0.25,
            "adequate_positive_cohort_available": False,
            "reason": "Only four trigger-positive questions remain by exposure. Even an ideal 4/4 result would have a very wide exact binomial interval; no posthoc threshold or sample-size rule is being imposed.",
            "gold_status_after_audit": "DEVELOPMENT_EXPOSED; official labels were opened for source crosswalk and projection diagnostics, so no WebQSP row is now an untouched holdout for this project.",
            "no_prospective_manifest_created": True,
        },
        "question_id_mismatch_ids": question_mismatch_ids,
        "local_answer_projection_mismatch_ids": answer_union_mismatch_ids,
        "multi_parse_ids": multi_parse_ids,
    }
    if len(local) != 1639 or len(exposed_ids) != 11 or len(clean_ids) != 1628 or len(count_intent_ids) != 4 or len(local_answer_exactly_matches_any_parse_ids) != 1591:
        raise SystemExit("WebQSP/exposure/trigger audit count differs from the reviewed historical crosswalk; inspect output before drawing a conclusion")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "source_crosswalk": out["source_crosswalk"], "exposure_scan": {k: v for k, v in out["exposure_scan"].items() if k not in {"question_id_overlaps_excluded", "exact_normalized_question_overlaps_in_remainder", "near_duplicate_threshold_rapidfuzz_ratio_gte_90"}}, "frozen_trigger_feasibility": out["frozen_trigger_feasibility"], "output": str(OUT.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
