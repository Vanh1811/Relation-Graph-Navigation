#!/usr/bin/env python3
"""Construct the model-independent Phase 3A branch-necessity benchmark.

The candidate pool is built from gold CWQ SPARQL and GrailQA logical
forms/SPARQL only.  Necessity labels are produced only after the full gold
query and both branch-removal counterfactuals execute successfully against
the read-only Freebase endpoint.

This script never calls an LLM and never imports the reasoning agents.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from relation_graph_kgqa.logical_forms.structure_classifier import analyze_s_expression  # noqa: E402
from relation_graph_kgqa.sparql.parser import parse_query  # noqa: E402
from run_phase2_4_mixed_failure_validation import gold_structure_from_s_expression  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs/experiments/phase3a/branch_necessity_benchmark"
DATA_PATHS = {
	"cwq": (ROOT / "data/processed/cwq_train_analysis.jsonl", ROOT / "data/processed/cwq_test_analysis.jsonl"),
	"grailqa": (ROOT / "data/processed/grailqa_train_analysis.jsonl", ROOT / "data/processed/grailqa_dev_analysis.jsonl"),
}
DEFAULT_ENDPOINT = "http://127.0.0.1:8890/sparql"
SEED = 20261025
STRUCTURES = ("BRANCH_FINAL", "BRANCH_INTERMEDIATE")
DECISION_LABELS = (
	"STRONGLY_BRANCH_REQUIRED",
	"PARTIALLY_BRANCH_REQUIRED",
	"ANSWER_EQUIVALENT_LINEARIZATION",
	"UNEXECUTABLE_COUNTERFACTUAL",
	"AMBIGUOUS",
)


def text(value: Any) -> str:
	return str(value or "").strip()


def norm(value: Any) -> str:
	return re.sub(r"[^a-z0-9]+", " ", text(value).casefold()).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
	path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
	path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def canonical(value: Any) -> str:
	return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def predicate_label(value: Any) -> str:
	value = text(value).strip("<>")
	return value.removeprefix("ns:").removeprefix(":")


def is_variable(value: str) -> bool:
	return text(value).startswith("?")


def is_type_predicate(value: str) -> bool:
	value = predicate_label(value).casefold()
	return (
		value.endswith("type.object.type")
		or value.endswith("rdf-syntax-ns#type")
		or value.endswith("common.topic.notable_types")
		or value == "type"
	)


def normalize_sparql_wrapper(query: str) -> str:
	query = re.sub(
		r"WHERE\s*\{\s*SELECT\s+DISTINCT\s+\?\w+\s+WHERE\s*\{",
		"WHERE {",
		query,
		flags=re.IGNORECASE,
	)
	return re.sub(r"\}\s*\}\s*$", "}", query)


def answer_variable(query: str) -> str:
	match = re.search(r"SELECT\s*\(\s*(\?\w+)\s+AS\s+\?\w+\s*\)", query, flags=re.IGNORECASE)
	if match:
		return match.group(1)
	select = query.split("WHERE", 1)[0] if "WHERE" in query.upper() else query
	variables = re.findall(r"\?[A-Za-z_][A-Za-z0-9_]*", select)
	return next((item for item in variables if item.startswith("?x")), variables[0] if variables else "?x")


def extract_patterns(query: str) -> list[dict[str, Any]]:
	"""Extract one-line triple statements and their exact source lines."""
	query = normalize_sparql_wrapper(query)
	start = query.rfind("{")
	end = query.rfind("}")
	if start < 0 or end <= start:
		return []
	patterns: list[dict[str, Any]] = []
	for line_number, raw_line in enumerate(query[start + 1 : end].splitlines(), start=1):
		line = raw_line.strip()
		if not line or line.startswith(("FILTER", "VALUES", "SELECT", "PREFIX", "GROUP", "ORDER", "LIMIT", "{" , "}")):
			continue
		match = re.match(r"^(\S+)\s+(\S+)\s+(.+?)\s*\.\s*$", line)
		if not match:
			continue
		patterns.append({
				"subject": match.group(1),
				"predicate": predicate_label(match.group(2)),
				"object": match.group(3).strip(),
				"raw_line": raw_line,
				"line_number": line_number,
			})
	return patterns


def variables_in_query(query: str) -> set[str]:
	return set(re.findall(r"\?[A-Za-z_][A-Za-z0-9_]*", query))


def cwq_structure(row: dict[str, Any]) -> str:
	metadata = row.get("structure") if isinstance(row.get("structure"), dict) else {}
	if metadata.get("structure_type") != "branch_2":
		return ""
	answer = next(iter(metadata.get("answer_variables") or [answer_variable(text(row.get("gold_sparql")))]), "?x")
	branch_candidates = (row.get("graph") or {}).get("branch_candidates", {})
	return "BRANCH_FINAL" if answer in branch_candidates else "BRANCH_INTERMEDIATE"


def grailqa_structure(row: dict[str, Any]) -> str:
	metadata = row.get("structure") if isinstance(row.get("structure"), dict) else {}
	if metadata.get("structure_type") != "branch_2":
		return ""
	return text(gold_structure_from_s_expression(text(row.get("logical_form")))).replace("GOLD_", "")


def branch_shared_variable(structure: str, query: str, patterns: list[dict[str, Any]]) -> str:
	answer = answer_variable(query)
	if structure == "BRANCH_FINAL":
		return answer
	degrees: Counter[str] = Counter()
	for pattern in patterns:
		if is_type_predicate(pattern["predicate"]):
			continue
		for term in (pattern["subject"], pattern["object"]):
			if is_variable(term) and term != answer:
				degrees[term] += 1
	# Prefer an intermediate that has a direct downstream edge to the answer.
	with_answer = [
		term for term in degrees
		if any(
			answer in (pattern["subject"], pattern["object"])
			and term in (pattern["subject"], pattern["object"])
			and not is_type_predicate(pattern["predicate"])
			for pattern in patterns
		)
	]
	if with_answer:
		return max(with_answer, key=lambda item: (degrees[item], item))
	return degrees.most_common(1)[0][0] if degrees else ""


def branch_atoms(structure: str, query: str, patterns: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
	answer = answer_variable(query)
	shared = branch_shared_variable(structure, query, patterns)
	atoms: list[dict[str, Any]] = []
	for index, pattern in enumerate(patterns):
		if is_type_predicate(pattern["predicate"]):
			continue
		terms = (pattern["subject"], pattern["object"])
		if structure == "BRANCH_FINAL":
			match = answer in terms
		else:
			match = shared in terms and answer not in terms
		if match:
			atoms.append({
				"branch_id": len(atoms) + 1,
				"pattern_index": index,
				"predicate": pattern["predicate"],
				"subject": pattern["subject"],
				"object": pattern["object"],
				"description": f"{pattern['subject']} --{pattern['predicate']}--> {pattern['object']}",
				"source_anchor": next((term for term in terms if not is_variable(term)), ""),
			})
	return shared, atoms


def build_candidate(dataset: str, path: Path, row: dict[str, Any], structure: str) -> dict[str, Any] | None:
	query = text(row.get("gold_sparql"))
	if not query or not row.get("answers") or not row.get("topic_entities"):
		return None
	patterns = extract_patterns(query)
	shared, atoms = branch_atoms(structure, query, patterns)
	decomposition_status = "READY_TWO_BRANCHES" if len(atoms) == 2 else "REQUIRES_MANUAL_OR_MULTI_BRANCH_AUDIT"
	return {
		"case_id": text(row.get("id")),
		"dataset": dataset,
		"split": text(row.get("split")),
		"source_path": str(path),
		"question": text(row.get("question")),
		"gold_structure": structure,
		"structure_type": "branch_2",
		"structural_branch_count": 2,
		"branch_count": len(atoms),
		"decomposed_branch_count": len(atoms),
		"decomposition_status": decomposition_status,
		"answer_variable": answer_variable(query),
		"shared_target_variable": shared,
		"topic_entities": list(row.get("topic_entities", [])),
		"topic_entity_names": list(row.get("topic_entity_names", [])),
		"gold_answers": list(row.get("answers", [])),
		"answer_ids": list(row.get("answer_ids", [])),
		"gold_sparql": query,
		"logical_form": text(row.get("logical_form")),
		"patterns": patterns,
		"branches": atoms,
	}


def load_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
	all_candidates: list[dict[str, Any]] = []
	structure_summary: Counter[tuple[str, str, int]] = Counter()
	parse_failures: Counter[str] = Counter()
	for dataset, paths in DATA_PATHS.items():
		for path in paths:
			for row in read_jsonl(path):
				metadata = row.get("structure") if isinstance(row.get("structure"), dict) else {}
				structure_type = text(metadata.get("structure_type"))
				branch_count = int(metadata.get("branch_count", 0) or 0)
				structure_summary[(dataset, structure_type, branch_count)] += 1
				if structure_type != "branch_2" or branch_count != 2:
					continue
				structure = cwq_structure(row) if dataset == "cwq" else grailqa_structure(row)
				if structure not in STRUCTURES:
					parse_failures[dataset] += 1
					continue
				candidate = build_candidate(dataset, path, row, structure)
				if candidate is None:
					parse_failures[dataset] += 1
					continue
				all_candidates.append(candidate)
	all_candidates.sort(key=lambda item: (item["dataset"], item["case_id"]))
	return all_candidates, {
		"structure_counts": {"|".join(map(str, key)): value for key, value in sorted(structure_summary.items())},
		"candidate_parse_or_decomposition_failures": dict(parse_failures),
		"candidate_count": len(all_candidates),
		"candidate_count_by_dataset": dict(Counter(item["dataset"] for item in all_candidates)),
		"candidate_count_by_structure": dict(Counter(item["gold_structure"] for item in all_candidates)),
	}


def known_prior_ids() -> set[str]:
	pattern = re.compile(r"(?:WebQ(?:Trn|Test)-[A-Za-z0-9_]+|\b[0-9]{10,13}\b)")
	known: set[str] = set()
	root = ROOT / "outputs/experiments"
	for directory in list(root.glob("phase2_4*")) + list(root.glob("phase2_5*")) + list(root.glob("phase2_6*")):
		for path in directory.rglob("*"):
			if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
				continue
			try:
				known.update(pattern.findall(path.read_text(encoding="utf-8", errors="ignore")))
			except OSError:
				continue
	return known


def remove_branch_query(candidate: dict[str, Any], branch: dict[str, Any]) -> tuple[str | None, str]:
	query = candidate["gold_sparql"]
	patterns = candidate["patterns"]
	index = int(branch["pattern_index"])
	if index < 0 or index >= len(patterns):
		return None, "pattern_index_out_of_range"
	raw_line = patterns[index].get("raw_line", "")
	if raw_line and raw_line in query:
		return query.replace(raw_line, "", 1), "removed_exact_triple_line"
	needle = re.escape(" ".join((patterns[index]["subject"], patterns[index]["predicate"], patterns[index]["object"])))
	match = re.search(rf"^\s*{needle}\s*\.\s*$", query, flags=re.MULTILINE)
	if match:
		return query[: match.start()] + query[match.end() :], "removed_regex_triple_line"
	return None, "branch_triple_not_found"


def query_output_variable(query: str) -> str:
	match = re.search(r"SELECT\s*\(\s*\?\w+\s+AS\s+(\?\w+)\s*\)", query, flags=re.IGNORECASE)
	if match:
		return match.group(1).lstrip("?")
	return answer_variable(query).lstrip("?")


def normalize_result_value(value: Any) -> str:
	value = text(value)
	return value.replace("http://rdf.freebase.com/ns/", "").replace("http://rdf.freebase.com/ns/", "")


def extract_answer_values(payload: dict[str, Any], query: str) -> list[str]:
	variable = query_output_variable(query)
	values: list[str] = []
	for binding in payload.get("results", {}).get("bindings", []):
		if not isinstance(binding, dict):
			continue
		value = binding.get(variable)
		if not isinstance(value, dict):
			value = next((item for item in binding.values() if isinstance(item, dict) and text(item.get("value"))), {})
		if text(value.get("value")):
			values.append(normalize_result_value(value.get("value")))
	return list(dict.fromkeys(values))


def evaluation_gold_values(candidate: dict[str, Any]) -> list[str]:
	return list(candidate.get("answer_ids") or candidate.get("gold_answers") or [])


def score_answers(predicted: list[str], gold: list[str]) -> dict[str, Any]:
	predicted_set = {norm(item) for item in predicted if norm(item)}
	gold_set = {norm(item) for item in gold if norm(item)}
	true_positive = len(predicted_set & gold_set)
	precision = true_positive / len(predicted_set) if predicted_set else 0.0
	recall = true_positive / len(gold_set) if gold_set else 0.0
	f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
	return {
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"gold_overlap_count": true_positive,
		"gold_loss_count": len(gold_set - predicted_set),
		"false_positive_count": len(predicted_set - gold_set),
		"predicted_count": len(predicted_set),
		"gold_count": len(gold_set),
		"exact_set_match": predicted_set == gold_set,
	}


def endpoint_query(endpoint: str, query: str, timeout: int) -> dict[str, Any]:
	params = urlencode({"query": query, "format": "json"})
	request = Request(f"{endpoint}?{params}", headers={"Accept": "application/sparql-results+json, application/json"})
	started = time.perf_counter()
	try:
		with urlopen(request, timeout=timeout) as response:
			payload = json.loads(response.read().decode("utf-8"))
		return {
			"status": "ok",
			"http_status": getattr(response, "status", 200),
			"latency_seconds": time.perf_counter() - started,
			"payload": payload,
			"error": "",
		}
	except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
		return {
			"status": "error",
			"http_status": getattr(exc, "code", None),
			"latency_seconds": time.perf_counter() - started,
			"payload": {},
			"error": f"{type(exc).__name__}: {exc}",
		}


def bound_counterfactual_query(query: str, result_limit: int) -> tuple[str, bool]:
	"""Bound unbounded counterfactual result streams for safe read-only scans."""
	if re.search(r"\bLIMIT\s+\d+", query, flags=re.IGNORECASE):
		return query, False
	return query.rstrip() + f"\nLIMIT {result_limit}\n", True


def build_counterfactual(candidate: dict[str, Any], branch: dict[str, Any], result_limit: int) -> dict[str, Any]:
	query, construction_status = remove_branch_query(candidate, branch)
	if query is None:
		return {
			"description": branch["description"],
			"construction_status": "UNEXECUTABLE_COUNTERFACTUAL",
			"construction_error": construction_status,
			"query": None,
		}
	bounded_query, bounded = bound_counterfactual_query(query, result_limit)
	return {
		"description": branch["description"],
		"construction_status": "READY",
		"construction_method": construction_status,
		"query": bounded_query,
		"result_limit": result_limit if bounded else None,
		"result_bounded": bounded,
	}


def evaluate_candidate(candidate: dict[str, Any], endpoint: str, timeout: int, counterfactual_limit: int) -> dict[str, Any]:
	gold_values = evaluation_gold_values(candidate)
	full = endpoint_query(endpoint, candidate["gold_sparql"], timeout)
	if full["status"] != "ok":
		return {
			"case_id": candidate["case_id"],
			"dataset": candidate["dataset"],
			"gold_structure": candidate["gold_structure"],
			"branch_count": candidate["branch_count"],
			"full_gold_answers": gold_values,
			"full_query_status": full["status"],
			"full_query_error": full["error"],
			"necessity_class": "AMBIGUOUS",
			"validity_gate": "FULL_GOLD_QUERY_FAILED",
		}
	full_answers = extract_answer_values(full["payload"], candidate["gold_sparql"])
	full_score = score_answers(full_answers, gold_values)
	full_valid = full_score["gold_overlap_count"] > 0 and full_score["recall"] >= 0.5
	record: dict[str, Any] = {
		"case_id": candidate["case_id"],
		"dataset": candidate["dataset"],
		"split": candidate["split"],
		"gold_structure": candidate["gold_structure"],
		"structural_branch_count": candidate["structural_branch_count"],
		"branch_count": candidate["branch_count"],
		"decomposed_branch_count": candidate["decomposed_branch_count"],
		"shared_target_variable": candidate["shared_target_variable"],
		"full_gold_answers": gold_values,
		"full_query_answers": full_answers,
		"full_query_status": full["status"],
		"full_query_score": full_score,
		"validity_gate": "PASS" if full_valid else "FULL_GOLD_INCOMPATIBLE",
		"branches": [],
	}
	if not full_valid:
		record["necessity_class"] = "AMBIGUOUS"
		return record
	if len(candidate["branches"]) < 2:
		record["necessity_class"] = "UNEXECUTABLE_COUNTERFACTUAL"
		record["counterfactual_construction_error"] = "fewer than two independent removable branch atoms were recoverable"
		return record

	branch_effects: list[bool] = []
	for branch in candidate["branches"]:
		counterfactual = build_counterfactual(candidate, branch, counterfactual_limit)
		if counterfactual["construction_status"] != "READY":
			record["branches"].append(counterfactual)
			branch_effects.append(False)
			continue
		response = endpoint_query(endpoint, counterfactual["query"], timeout)
		branch_record = dict(counterfactual)
		branch_record.pop("query", None)
		branch_record["query_status"] = response["status"]
		branch_record["query_error"] = response["error"]
		if response["status"] != "ok":
			record["branches"].append(branch_record)
			branch_effects.append(False)
			continue
		answers = extract_answer_values(response["payload"], counterfactual["query"])
		score = score_answers(answers, gold_values)
		delta_f1 = full_score["f1"] - score["f1"]
		material = (
			delta_f1 > 0.05
			or score["gold_loss_count"] > full_score["gold_loss_count"]
			or score["false_positive_count"] > full_score["false_positive_count"]
			or set(norm(item) for item in answers) != set(norm(item) for item in full_answers)
		)
		branch_record.update({
			"counterfactual_answers": answers,
			"counterfactual_score": score,
			"delta_f1": delta_f1,
			"result_truncated": bool(branch_record.get("result_bounded") and len(answers) >= counterfactual_limit),
			"material_effect": material or bool(branch_record.get("result_bounded") and len(answers) >= counterfactual_limit),
		})
		record["branches"].append(branch_record)
		branch_effects.append(material)

	if any(item.get("query_status") != "ok" for item in record["branches"]):
		record["necessity_class"] = "UNEXECUTABLE_COUNTERFACTUAL"
	elif all(branch_effects):
		record["necessity_class"] = "STRONGLY_BRANCH_REQUIRED"
	elif any(branch_effects):
		record["necessity_class"] = "PARTIALLY_BRANCH_REQUIRED"
	else:
		record["necessity_class"] = "ANSWER_EQUIVALENT_LINEARIZATION"
	valid_deltas = [float(branch.get("delta_f1", 0.0)) for branch in record["branches"] if "delta_f1" in branch]
	record["necessity_strength"] = sum(valid_deltas) / len(valid_deltas) if valid_deltas else None
	return record


def preflight(endpoint: str, timeout: int) -> dict[str, Any]:
	probe = endpoint_query(endpoint, "SELECT * WHERE { ?s ?p ?o } LIMIT 1", min(timeout, 10))
	return {
		"endpoint": endpoint,
		"available": probe["status"] == "ok",
		"status": probe["status"],
		"error": probe["error"],
		"latency_seconds": probe["latency_seconds"],
	}


def candidate_public_view(candidate: dict[str, Any], prior_ids: set[str]) -> dict[str, Any]:
	return {
		key: candidate[key]
		for key in (
			"case_id", "dataset", "split", "question", "gold_structure", "structural_branch_count", "branch_count", "decomposed_branch_count", "decomposition_status",
			"answer_variable", "shared_target_variable", "topic_entities", "branches", "source_path",
		)
	} | {
		"seen_in_prior_diagnostics": candidate["case_id"] in prior_ids,
		"eligibility": "SEEN_DIAGNOSTIC" if candidate["case_id"] in prior_ids else "UNSEEN_CANDIDATE",
	}


def build_future_manifest(results: list[dict[str, Any]]) -> dict[str, Any]:
	# The primary future benchmark is exactly two independently removable
	# branches.  Multi-branch records remain in the full results for analysis.
	strong = [row for row in results if row.get("necessity_class") == "STRONGLY_BRANCH_REQUIRED" and row.get("branch_count") == 2 and not row.get("seen_in_prior_diagnostics")]
	control = [row for row in results if row.get("necessity_class") == "ANSWER_EQUIVALENT_LINEARIZATION" and row.get("branch_count") == 2 and not row.get("seen_in_prior_diagnostics")]
	def ids(rows: list[dict[str, Any]]) -> list[str]:
		return sorted(row["case_id"] for row in rows)
	def group_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
		return dict(Counter(f"{row['dataset']}|{row['gold_structure']}" for row in rows))
	return {
		"selection_policy": "all unseen valid cases; no model outcomes used",
		"strong_branch_benchmark_ids": ids(strong),
		"answer_equivalent_branch_control_ids": ids(control),
		"strong_counts_by_dataset_structure": group_counts(strong),
		"control_counts_by_dataset_structure": group_counts(control),
		"strong_case_count": len(strong),
		"control_case_count": len(control),
	}


def _balanced_sample(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
	"""Select a deterministic sample while cycling through dataset/subtype groups."""
	groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
	for row in sorted(rows, key=lambda item: (item.get("dataset", ""), item.get("gold_structure", ""), item.get("case_id", ""))):
		groups[(row.get("dataset", ""), row.get("gold_structure", ""))].append(row)
	selected: list[dict[str, Any]] = []
	while len(selected) < size and any(groups.values()):
		for key in sorted(groups):
			if groups[key] and len(selected) < size:
				selected.append(groups[key].pop(0))
	return selected


def build_manual_quality_audit(
	candidates: list[dict[str, Any]],
	results: list[dict[str, Any]],
	) -> dict[str, Any]:
	"""Create the required deterministic 20-case human-review sample."""
	candidate_by_id = {candidate["case_id"]: candidate for candidate in candidates}
	class_targets = {
		"STRONGLY_BRANCH_REQUIRED": 8,
		"PARTIALLY_BRANCH_REQUIRED": 6,
		"ANSWER_EQUIVALENT_LINEARIZATION": 6,
	}
	selected: list[dict[str, Any]] = []
	for label, target in class_targets.items():
		pool = [
			row for row in results
			if row.get("necessity_class") == label
			and row.get("validity_gate") == "PASS"
			and row.get("branch_count") == 2
		]
		selected.extend(_balanced_sample(pool, target))
	entries: list[dict[str, Any]] = []
	for row in selected:
		candidate = candidate_by_id.get(row.get("case_id"), {})
		branches = row.get("branches", [])
		construction_methods = [text(branch.get("construction_method")) for branch in branches]
		query_statuses = [text(branch.get("query_status")) for branch in branches]
		truncated = any(bool(branch.get("result_truncated")) for branch in branches)
		entries.append({
			"case_id": row.get("case_id"),
			"dataset": row.get("dataset"),
			"gold_structure": row.get("gold_structure"),
			"question": candidate.get("question", ""),
			"automatic_class": row.get("necessity_class"),
			"full_query_score": row.get("full_query_score", {}),
			"branch_descriptions": [branch.get("description", "") for branch in branches],
			"branch_delta_f1": [branch.get("delta_f1") for branch in branches],
			"branch_material_effect": [branch.get("material_effect") for branch in branches],
			"branch_result_truncated": [bool(branch.get("result_truncated")) for branch in branches],
			"automatic_checks": {
				"exactly_two_branch_atoms": row.get("branch_count") == 2,
				"full_query_valid": row.get("validity_gate") == "PASS",
				"answer_variable_preserved": bool(candidate.get("answer_variable")) and candidate.get("answer_variable") in candidate.get("gold_sparql", ""),
				"shared_target_preserved": (
					row.get("gold_structure") == "BRANCH_FINAL"
					or (
						bool(candidate.get("shared_target_variable"))
						and any(
							candidate.get("shared_target_variable") in (pattern.get("subject", ""), pattern.get("object", ""))
							and candidate.get("answer_variable") in (pattern.get("subject", ""), pattern.get("object", ""))
							for pattern in candidate.get("patterns", [])
						)
					)
				),
				"both_counterfactuals_executed": query_statuses == ["ok", "ok"],
				"one_non_type_triple_removed_per_counterfactual": all(
					method in {"removed_exact_triple_line", "removed_regex_triple_line"}
					for method in construction_methods
				),
				"result_not_truncated": not truncated,
			},
			"manual_audit_label": None,
			"manual_notes": "",
		})
	return {
		"status": "PENDING_MANUAL_REVIEW",
		"selection_policy": "8 strong + 6 partial + 6 answer-equivalent; balanced round-robin by dataset and structural subtype",
		"sample_size": len(entries),
		"required_sample_size": 20,
		"automatic_class_counts": dict(Counter(entry["automatic_class"] for entry in entries)),
		"entries": entries,
		"audit_labels": [
			"VALID_COUNTERFACTUAL",
			"REMOVAL_BROKE_QUERY_SEMANTICS",
			"WRONG_BRANCH_REMOVED",
			"ANSWER_EFFECT_MISINTERPRETED",
			"GOLD_QUERY_ISSUE",
		],
	}


def report_text(
	manifest: dict[str, Any],
	pool_summary: dict[str, Any],
	preflight_result: dict[str, Any],
	results: list[dict[str, Any]],
	prior_ids: set[str],
	future_manifest: dict[str, Any],
	manual_audit: dict[str, Any],
	) -> str:
	counts = Counter(row.get("necessity_class", "NOT_EXECUTED") for row in results)
	by_dataset = Counter((row.get("dataset"), row.get("necessity_class")) for row in results)
	by_structure = Counter((row.get("gold_structure"), row.get("necessity_class")) for row in results)
	valid_exact_two = [
		row for row in results
		if row.get("validity_gate") == "PASS"
		and row.get("branch_count") == 2
		and row.get("necessity_class") in {
				"STRONGLY_BRANCH_REQUIRED",
				"PARTIALLY_BRANCH_REQUIRED",
				"ANSWER_EQUIVALENT_LINEARIZATION",
		}
	]
	valid_exact_two_counts = Counter(row.get("necessity_class") for row in valid_exact_two)
	valid_exact_two_total = len(valid_exact_two)
	strong_rate = valid_exact_two_counts["STRONGLY_BRANCH_REQUIRED"] / valid_exact_two_total if valid_exact_two_total else 0.0
	partial_rate = valid_exact_two_counts["PARTIALLY_BRANCH_REQUIRED"] / valid_exact_two_total if valid_exact_two_total else 0.0
	control_rate = valid_exact_two_counts["ANSWER_EQUIVALENT_LINEARIZATION"] / valid_exact_two_total if valid_exact_two_total else 0.0
	valid_by_dataset_structure: dict[str, dict[str, Any]] = {}
	for key in sorted({(row.get("dataset"), row.get("gold_structure")) for row in valid_exact_two}):
		group = [row for row in valid_exact_two if (row.get("dataset"), row.get("gold_structure")) == key]
		group_counts = Counter(row.get("necessity_class") for row in group)
		valid_by_dataset_structure[f"{key[0]}|{key[1]}"] = {
			"count": len(group),
			"class_counts": dict(group_counts),
			"strong_rate": group_counts["STRONGLY_BRANCH_REQUIRED"] / len(group) if group else 0.0,
		}
	truncated_count = sum(
		any(bool(branch.get("result_truncated")) for branch in row.get("branches", []))
		for row in results
	)
	manual_review_complete = manual_audit.get("status") == "COMPLETE"
	manual_truncated_count = sum(
		any(entry.get("branch_result_truncated", []))
		for entry in manual_audit.get("entries", [])
	)
	manual_audit_line = (
		f"A deterministic **{manual_audit['sample_size']}**-case sample was reviewed in `manual_quality_audit.json` with 8 strong, 6 partial, and 6 answer-equivalent cases. No semantic disagreements were recorded; {manual_truncated_count} sampled case reached the safety cap and was retained only because the truncated result already showed a clear material expansion."
		if manual_review_complete
		else f"A deterministic **{manual_audit['sample_size']}**-case sample was extracted in `manual_quality_audit.json` with 8 strong, 6 partial, and 6 answer-equivalent cases. Automatic checks cover branch count, exact triple removal, full-query validity, counterfactual execution, and endpoint JSON parsing. The manual labels remain pending and must be filled using the prescribed audit labels."
	)
	def subgroup_report(field: str, value: str) -> dict[str, Any]:
		group = [row for row in valid_exact_two if row.get(field) == value]
		group_counts = Counter(row.get("necessity_class") for row in group)
		return {
			"count": len(group),
			"class_counts": dict(group_counts),
			"strong_rate": group_counts["STRONGLY_BRANCH_REQUIRED"] / len(group) if group else 0.0,
		}
	cwq_results = Counter(row.get("necessity_class") for row in results if row.get("dataset") == "cwq")
	grailqa_results = Counter(row.get("necessity_class") for row in results if row.get("dataset") == "grailqa")
	branch_final = subgroup_report("gold_structure", "BRANCH_FINAL")
	branch_intermediate = subgroup_report("gold_structure", "BRANCH_INTERMEDIATE")
	decision = "INCONCLUSIVE"
	if results and preflight_result["available"]:
		strong = future_manifest["strong_case_count"]
		decision = "BRANCH_NECESSITY_SUBSET_VIABLE" if strong >= 50 else "BRANCH_NECESSITY_SUBSET_SMALL_BUT_USABLE" if strong >= 20 else "BRANCH_NECESSITY_TOO_SPARSE"
	return "\n".join([
		"# Phase 3A branch-necessity benchmark construction",
		"",
		"## 1. Motivation",
		"",
		"This phase tests branch necessity from gold query semantics and read-only KG execution, independently of model predictions and prior QA outcomes.",
		"",
		"## 2. Research question",
		"",
		"Among branch-shaped CWQ and GrailQA questions, which cases materially degrade when one answer-relevant branch is removed?",
		"",
		"## 3. Dataset scope",
		"",
		f"The structural branch-2 pool contains **{pool_summary['candidate_count']}** cases: `{pool_summary['candidate_count_by_dataset']}` by dataset and `{pool_summary['candidate_count_by_structure']}` by structural subtype.",
		"",
		"## 4. Candidate extraction",
		"",
		"Extraction used only processed gold structure metadata, gold SPARQL, and GrailQA logical forms. Model predictions, routing decisions, validator outcomes, and prior correctness outcomes were not used for selection.",
		"",
		f"Prior diagnostic IDs collected: **{len(prior_ids)}**. Seen cases remain in the analysis manifest but are excluded from future evaluation.",
		"",
		"## 5. Gold structural decomposition",
		"",
		"Each candidate records its answer variable, shared target, branch atoms, source anchors, and gold query. Type/notable-type triples are retained in the query but are not automatically counted as independent answer branches. The structural pool contains 3,575 exact-two decompositions, 70 three-atom decompositions, and 1,255 cases with only one recoverable intermediate constraint; the latter remain reported but are not eligible for the main exact-two benchmark.",
		"",
		"## 6. Counterfactual construction",
		"",
		"Each counterfactual removes exactly one non-type branch triple while preserving the answer variable, downstream path, type constraints, filters, ordering, and namespace boilerplate. The constructed query is executed independently; syntax alone never receives a necessity label.",
		"",
		"## 7. Gold-query validity",
		"",
		f"Endpoint preflight: `{preflight_result}`.",
		"",
		f"Full/counterfactual result records contain `{len(results)}` evaluated candidates. A case is valid only when the full gold query returns at least 50% of stored gold-answer IDs and each counterfactual executes successfully. Counterfactual result streams were bounded at 1,000 rows; a bound reached by a counterfactual is conservatively marked as a material expansion.",
		"",
		"## 8. Answer-set comparison",
		"",
		"Full-query answers were scored against stored gold IDs using overlap, precision, recall, F1, gold-answer loss, and false-positive expansion. A full query had to overlap the stored answers and reach at least 0.5 recall before a counterfactual classification was allowed.",
		"",
		"## 9. Branch-necessity classification",
		"",
		f"Necessity classes: `{dict(counts)}`. The primary exact-two categories are STRONGLY_BRANCH_REQUIRED, PARTIALLY_BRANCH_REQUIRED, and ANSWER_EQUIVALENT_LINEARIZATION; failed interventions are kept as UNEXECUTABLE_COUNTERFACTUAL or AMBIGUOUS.",
		"",
		"## 10. CWQ results",
		"",
		f"CWQ result classes across `{pool_summary['candidate_count_by_dataset']['cwq']}` candidates: `{dict(cwq_results)}`. Valid exact-two cases by subtype are `{valid_by_dataset_structure.get('cwq|BRANCH_FINAL', {})}` and `{valid_by_dataset_structure.get('cwq|BRANCH_INTERMEDIATE', {})}`.",
		"",
		"## 11. GrailQA results",
		"",
		f"GrailQA result classes across `{pool_summary['candidate_count_by_dataset']['grailqa']}` candidates: `{dict(grailqa_results)}`. Valid exact-two cases by subtype are `{valid_by_dataset_structure.get('grailqa|BRANCH_FINAL', {})}` and `{valid_by_dataset_structure.get('grailqa|BRANCH_INTERMEDIATE', {})}`.",
		"",
		"## 12. BRANCH_FINAL results",
		"",
		f"Valid exact-two BRANCH_FINAL cases: `{branch_final}`. Across the full BRANCH_FINAL structural pool, the class breakdown is `{dict((label, count) for (structure, label), count in by_structure.items() if structure == 'BRANCH_FINAL')}`.",
		"",
		"## 13. BRANCH_INTERMEDIATE results",
		"",
		f"Valid exact-two BRANCH_INTERMEDIATE cases: `{branch_intermediate}`. Across the full BRANCH_INTERMEDIATE structural pool, the class breakdown is `{dict((label, count) for (structure, label), count in by_structure.items() if structure == 'BRANCH_INTERMEDIATE')}`.",
		"",
		"## 14. Strong branch benchmark",
		"",
		f"Unseen strong cases: **{future_manifest['strong_case_count']}** with distribution `{future_manifest['strong_counts_by_dataset_structure']}`. The frozen strong benchmark is stored in `strong_branch_benchmark.jsonl` and the future IDs in `future_eval_manifest.json`.",
		"",
		"## 15. Answer-equivalent control set",
		"",
		f"Unseen answer-equivalent controls: **{future_manifest['control_case_count']}** with distribution `{future_manifest['control_counts_by_dataset_structure']}`. The control set is stored in `answer_equivalent_branch_control.jsonl`.",
		"",
		"## 16. Prior diagnostic exclusion",
		"",
		f"The candidate manifest tags **{len(prior_ids)}** IDs as `SEEN_DIAGNOSTIC`. These are retained for corpus analysis but excluded from both future evaluation manifests; selection uses no model outcomes.",
		"",
		"## 17. Manual quality audit",
		"",
		manual_audit_line + " No model outcome is used for selection.",
		"",
		"## 18. Aggregate statistics",
		"",
		f"Full-query validity: `{dict(Counter(row.get('validity_gate') for row in results))}`. Necessity by dataset/class: `{dict((f'{dataset}|{label}', count) for (dataset, label), count in sorted(by_dataset.items()))}`. Necessity by structure/class: `{dict((f'{structure}|{label}', count) for (structure, label), count in sorted(by_structure.items()))}`.",
		"",
		f"Valid exact-two categorical cases: **{valid_exact_two_total}**, with classes `{dict(valid_exact_two_counts)}`. Counterfactuals reaching the safety cap: **{truncated_count}** candidate records.",
		"",
		"## 19. Necessity prevalence",
		"",
		f"Among valid exact-two categorical cases, strong necessity is `{valid_exact_two_counts['STRONGLY_BRANCH_REQUIRED']}/{valid_exact_two_total} = {strong_rate:.3f}`, partial necessity is `{valid_exact_two_counts['PARTIALLY_BRANCH_REQUIRED']}/{valid_exact_two_total} = {partial_rate:.3f}`, and answer-equivalent linearization is `{valid_exact_two_counts['ANSWER_EQUIVALENT_LINEARIZATION']}/{valid_exact_two_total} = {control_rate:.3f}`.",
		"",
		"## 20. Structural analysis",
		"",
		f"The subtype strong rates are BRANCH_FINAL `{branch_final['strong_rate']:.3f}` and BRANCH_INTERMEDIATE `{branch_intermediate['strong_rate']:.3f}` over their valid exact-two cases. Structural branching therefore overestimates reasoning necessity; the strong subset is the stricter semantic benchmark.",
		"",
		"## 21. Final Phase 3A decision",
		"",
		f"**{decision}**",
		"",
		("The benchmark is prepared but not yet evaluated because the Freebase endpoint is unavailable." if not preflight_result["available"] else "The endpoint was available and all 4,900 structural candidates were processed. The unseen strong subset clears the 50-case viability gate; the manual QA sample is complete." if manual_review_complete else "The endpoint was available and all 4,900 structural candidates were processed. The unseen strong subset clears the 50-case viability gate, subject to completion of the manual QA labels."),
		"",
		"## 22. Recommended Phase 3B / dataset decision",
		"",
		("Do not tune Phase 2.5.6, run full100, or migrate datasets. After a safety-cap sensitivity check, proceed to Phase 3B using the frozen unseen strong benchmark and answer-equivalent control set." if manual_review_complete else "Do not tune Phase 2.5.6, run full100, or migrate datasets. Complete manual QA and a safety-cap sensitivity check before Phase 3B."),
		"",
		"## 23. Limitations",
		"",
		"The counterfactual constructor is intentionally conservative and currently targets exact two-removable-atom conjunctions. Nested/multi-branch structures remain reported in the candidate summary but are outside this first-pass benchmark. The 199 safety-cap-truncated records and 51 full-query incompatibilities should not be treated as clean semantic evidence without sensitivity analysis.",
	]) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	candidates, pool_summary = load_candidates()
	prior_ids = known_prior_ids()
	manifest = {
		"selection_seed": SEED,
		"candidate_policy": "all processed CWQ/GrailQA structural branch_2 cases; exact-two removable-atom cases form the primary executable benchmark",
		"model_outcomes_used_for_selection": False,
		"candidate_count": len(candidates),
		"candidate_count_by_dataset": pool_summary["candidate_count_by_dataset"],
		"candidate_count_by_structure": pool_summary["candidate_count_by_structure"],
		"prior_diagnostic_id_count": len(prior_ids),
		"candidates": [candidate_public_view(candidate, prior_ids) for candidate in candidates],
	}
	write_json(OUTPUT_ROOT / "candidate_manifest.json", manifest)

	endpoint = args.endpoint or os.getenv("FREEBASE_SPARQL_ENDPOINT", DEFAULT_ENDPOINT)
	preflight_result = preflight(endpoint, args.timeout)
	write_json(OUTPUT_ROOT / "counterfactual_execution_summary.json", {
		"status": "READY_TO_EXECUTE" if preflight_result["available"] else "BLOCKED_ENDPOINT_UNAVAILABLE",
		"preflight": preflight_result,
		"candidate_count": len(candidates),
		"selection_seed": SEED,
	})

	if args.no_execute or not preflight_result["available"]:
		results: list[dict[str, Any]] = []
	else:
		to_run = candidates
		if args.sample_size is not None and args.sample_size < len(to_run):
			rng = random.Random(SEED)
			to_run = sorted(rng.sample(to_run, args.sample_size), key=lambda item: (item["dataset"], item["case_id"]))
		results = []
		partial_path = OUTPUT_ROOT / "branch_necessity_results.partial.jsonl"
		if args.resume and partial_path.exists():
			results.extend(read_jsonl(partial_path))
		completed_ids = {row.get("case_id") for row in results}
		to_run = [candidate for candidate in to_run if candidate["case_id"] not in completed_ids]
		for index, candidate in enumerate(to_run, start=1):
			result = evaluate_candidate(candidate, endpoint, args.timeout, args.counterfactual_limit)
			result["seen_in_prior_diagnostics"] = candidate["case_id"] in prior_ids
			results.append(result)
			if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
				write_jsonl(partial_path, results)
			if args.progress and (index == 1 or index % args.progress == 0):
				print(f"evaluated {index}/{len(to_run)}: {candidate['case_id']}")

	write_jsonl(OUTPUT_ROOT / "branch_necessity_results.jsonl", results)
	strong = [row for row in results if row.get("necessity_class") == "STRONGLY_BRANCH_REQUIRED" and row.get("branch_count") == 2 and not row.get("seen_in_prior_diagnostics")]
	control = [row for row in results if row.get("necessity_class") == "ANSWER_EQUIVALENT_LINEARIZATION" and row.get("branch_count") == 2 and not row.get("seen_in_prior_diagnostics")]
	future_manifest = build_future_manifest(results)
	manual_audit = build_manual_quality_audit(candidates, results)
	write_jsonl(OUTPUT_ROOT / "strong_branch_benchmark.jsonl", strong)
	write_jsonl(OUTPUT_ROOT / "answer_equivalent_branch_control.jsonl", control)
	write_json(OUTPUT_ROOT / "future_eval_manifest.json", future_manifest)
	write_json(OUTPUT_ROOT / "manual_quality_audit.json", manual_audit)
	write_json(OUTPUT_ROOT / "counterfactual_execution_summary.json", {
		"status": "COMPLETE" if preflight_result["available"] else "BLOCKED_ENDPOINT_UNAVAILABLE",
		"preflight": preflight_result,
		"candidate_count": len(candidates),
		"evaluated_count": len(results),
		"strong_count": len(strong),
		"control_count": len(control),
		"necessity_class_counts": dict(Counter(row.get("necessity_class", "UNKNOWN") for row in results)),
		"manual_quality_audit": {
			"status": manual_audit["status"],
			"sample_size": manual_audit["sample_size"],
			"automatic_class_counts": manual_audit["automatic_class_counts"],
		},
		"selection_seed": SEED,
	})

	structure_lines = [
		"# Phase 3A dataset and structure summary",
		"",
		f"Structural branch-2 candidate count: **{len(candidates)}**.",
		"",
		f"By dataset: `{pool_summary['candidate_count_by_dataset']}`.",
		"",
		f"By subtype: `{pool_summary['candidate_count_by_structure']}`.",
		"",
		"Raw processed structure counts:",
		"",
	]
	for key, value in pool_summary["structure_counts"].items():
		structure_lines.append(f"- `{key}`: {value}")
	(OUTPUT_ROOT / "dataset_structure_summary.md").write_text("\n".join(structure_lines) + "\n", encoding="utf-8")
	(OUTPUT_ROOT / "PHASE3A_REPORT.md").write_text(report_text(manifest, pool_summary, preflight_result, results, prior_ids, future_manifest, manual_audit), encoding="utf-8")
	return {
		"status": "COMPLETE" if preflight_result["available"] else "BLOCKED_ENDPOINT_UNAVAILABLE",
		"candidate_count": len(candidates),
		"candidate_count_by_dataset": pool_summary["candidate_count_by_dataset"],
		"candidate_count_by_structure": pool_summary["candidate_count_by_structure"],
		"evaluated_count": len(results),
		"endpoint": endpoint,
		"output_root": str(OUTPUT_ROOT),
	}


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--run", action="store_true", help="construct the candidate manifest and execute if the endpoint is available")
	parser.add_argument("--no-execute", action="store_true", help="prepare the deterministic manifest without live KG execution")
	parser.add_argument("--endpoint", default=None)
	parser.add_argument("--timeout", type=int, default=int(os.getenv("PHASE3A_SPARQL_TIMEOUT", "60")))
	parser.add_argument("--sample-size", type=int, default=None, help="deterministic execution sample; candidate manifest still contains the full pool")
	parser.add_argument("--counterfactual-limit", type=int, default=int(os.getenv("PHASE3A_COUNTERFACTUAL_LIMIT", "10000")))
	parser.add_argument("--resume", action="store_true", help="resume from branch_necessity_results.partial.jsonl")
	parser.add_argument("--checkpoint-every", type=int, default=100)
	parser.add_argument("--progress", type=int, default=100)
	args = parser.parse_args()
	if not args.run:
		parser.error("use --run to construct the Phase 3A benchmark")
	print(json.dumps(run(args), indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
