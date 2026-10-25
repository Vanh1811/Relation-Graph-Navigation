#!/usr/bin/env python3
"""Run the Phase 3A.1 counterfactual safety-cap sensitivity check.

Only Phase 3A records whose counterfactual result reached the 1,000-row cap
are rerun.  Candidate extraction, query construction, scoring, and necessity
classification are imported unchanged from the Phase 3A runner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_phase3a_branch_necessity_benchmark import (  # noqa: E402
	DEFAULT_ENDPOINT,
	OUTPUT_ROOT as PHASE3A_ROOT,
	build_counterfactual,
	build_future_manifest,
	endpoint_query,
	evaluation_gold_values,
	extract_answer_values,
	load_candidates,
	known_prior_ids,
	norm,
	preflight,
	score_answers,
)


OUTPUT_ROOT = ROOT / "outputs/experiments/phase3a_1/safety_cap_sensitivity"
REFERENCE_CAP = 1000
DEFAULT_CAPS = (1000, 2000, 5000)
STABLE_LABELS = {"STABLE_ALL_CAPS", "STABLE_MATERIAL_EFFECT"}
UNSTABLE_LABELS = {"CLASS_CHANGED", "UNRESOLVED_STILL_TRUNCATED", "EXECUTION_ERROR"}
PRIMARY_CLASSES = {
	"STRONGLY_BRANCH_REQUIRED",
	"PARTIALLY_BRANCH_REQUIRED",
	"ANSWER_EQUIVALENT_LINEARIZATION",
}


def read_json(path: Path) -> dict[str, Any]:
	return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def capped_cases(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
	return sorted(
		[row for row in results if any(bool(branch.get("result_truncated")) for branch in row.get("branches", []))],
		key=lambda row: (row.get("dataset", ""), row.get("case_id", "")),
	)


def branch_metric(branch: dict[str, Any], cap: int, source: str = "phase3a_reference") -> dict[str, Any]:
	score = branch.get("counterfactual_score") or {}
	return {
		"cap": cap,
		"source": source,
		"status": branch.get("query_status", "ok"),
		"error": branch.get("query_error", ""),
		"predicted_answer_count": score.get("predicted_count", len(branch.get("counterfactual_answers", []))),
		"truncated": bool(branch.get("result_truncated")),
		"gold_overlap": score.get("gold_overlap_count", 0),
		"gold_loss": score.get("gold_loss_count", 0),
		"false_positive_count": score.get("false_positive_count", 0),
		"precision": score.get("precision", 0.0),
		"recall": score.get("recall", 0.0),
		"f1": score.get("f1", 0.0),
		"material_effect": bool(branch.get("material_effect", False)),
		"latency_seconds": branch.get("latency_seconds"),
	}


def material_effect(
	full_answers: list[str],
	full_score: dict[str, Any],
	answers: list[str],
	score: dict[str, Any],
	truncated: bool,
) -> tuple[bool, float]:
	delta_f1 = float(full_score.get("f1", 0.0)) - float(score.get("f1", 0.0))
	material = (
		delta_f1 > 0.05
		or score.get("gold_loss_count", 0) > full_score.get("gold_loss_count", 0)
		or score.get("false_positive_count", 0) > full_score.get("false_positive_count", 0)
		or {norm(item) for item in answers} != {norm(item) for item in full_answers}
		or truncated
	)
	return material, delta_f1


def evaluate_branch(
	candidate: dict[str, Any],
	original: dict[str, Any],
	branch: dict[str, Any],
	cap: int,
	endpoint: str,
	timeout: int,
) -> dict[str, Any]:
	started = time.perf_counter()
	counterfactual = build_counterfactual(candidate, branch, cap)
	if counterfactual.get("construction_status") != "READY":
		return {
			"cap": cap,
			"source": "rerun",
			"status": "error",
			"error": counterfactual.get("construction_error", "counterfactual construction failed"),
			"predicted_answer_count": 0,
			"truncated": False,
			"gold_overlap": 0,
			"gold_loss": 0,
			"false_positive_count": 0,
			"precision": 0.0,
			"recall": 0.0,
			"f1": 0.0,
			"material_effect": False,
			"latency_seconds": time.perf_counter() - started,
		}
	response = endpoint_query(endpoint, counterfactual["query"], timeout)
	if response["status"] != "ok":
		return {
			"cap": cap,
			"source": "rerun",
			"status": "error",
			"error": response["error"],
			"predicted_answer_count": 0,
			"truncated": False,
			"gold_overlap": 0,
			"gold_loss": 0,
			"false_positive_count": 0,
			"precision": 0.0,
			"recall": 0.0,
			"f1": 0.0,
			"material_effect": False,
			"latency_seconds": response["latency_seconds"],
		}
	answers = extract_answer_values(response["payload"], counterfactual["query"])
	gold_answers = evaluation_gold_values(candidate)
	full_answers = original.get("full_query_answers", [])
	full_score = original.get("full_query_score", {})
	score = score_answers(answers, gold_answers)
	truncated = bool(counterfactual.get("result_bounded") and len(answers) >= cap)
	effect, delta_f1 = material_effect(full_answers, full_score, answers, score, truncated)
	return {
		"cap": cap,
		"source": "rerun",
		"status": "ok",
		"error": "",
		"predicted_answer_count": score["predicted_count"],
		"truncated": truncated,
		"gold_overlap": score["gold_overlap_count"],
		"gold_loss": score["gold_loss_count"],
		"false_positive_count": score["false_positive_count"],
		"precision": score["precision"],
		"recall": score["recall"],
		"f1": score["f1"],
		"material_effect": effect,
		"delta_f1": delta_f1,
		"latency_seconds": response["latency_seconds"],
	}


def recompute_class(branches: list[dict[str, Any]]) -> str:
	if any(branch.get("status") != "ok" for branch in branches):
		return "UNEXECUTABLE_COUNTERFACTUAL"
	effects = [bool(branch.get("material_effect")) for branch in branches]
	if len(effects) < 2:
		return "UNEXECUTABLE_COUNTERFACTUAL"
	if all(effects):
		return "STRONGLY_BRANCH_REQUIRED"
	if any(effects):
		return "PARTIALLY_BRANCH_REQUIRED"
	return "ANSWER_EQUIVALENT_LINEARIZATION"


def metric_signature(metric: dict[str, Any]) -> tuple[Any, ...]:
	return tuple(metric.get(key) for key in (
		"status", "predicted_answer_count", "truncated", "gold_overlap", "gold_loss",
		"false_positive_count", "precision", "recall", "f1", "material_effect",
	))


def classify_stability(by_cap: dict[str, Any], classes: dict[str, str], caps: list[int]) -> str:
	if any(by_cap[str(cap)].get("status") == "error" for cap in caps[1:]):
		return "EXECUTION_ERROR"
	if len(set(classes.values())) > 1:
		return "CLASS_CHANGED"
	if by_cap[str(caps[-1])].get("truncated"):
		return "UNRESOLVED_STILL_TRUNCATED"
	reference = metric_signature(by_cap[str(caps[0])])
	if all(metric_signature(by_cap[str(cap)]) == reference for cap in caps):
		return "STABLE_ALL_CAPS"
	return "STABLE_MATERIAL_EFFECT"


def build_case_record(
	candidate: dict[str, Any],
	original: dict[str, Any],
	endpoint: str,
	timeout: int,
	caps: list[int],
) -> dict[str, Any]:
	original_branches = original.get("branches", [])
	branch_records: list[dict[str, Any]] = []
	for index, original_branch in enumerate(original_branches):
		is_target = bool(original_branch.get("result_truncated"))
		original_limit = original_branch.get("result_limit")
		metrics: dict[str, Any] = {"branch_index": index, "target_capped": is_target}
		for cap in caps:
			reference_available = is_target and original_limit == REFERENCE_CAP
			if not is_target or (cap == REFERENCE_CAP and reference_available):
				metrics[str(cap)] = branch_metric(original_branch, cap, "phase3a_reference")
			else:
				metrics[str(cap)] = evaluate_branch(candidate, original, candidate["branches"][index], cap, endpoint, timeout)
		metrics["original_result_limit"] = original_limit
		branch_records.append(metrics)
	classes: dict[str, str] = {}
	for cap in caps:
		classes[str(cap)] = recompute_class([branch[str(cap)] for branch in branch_records])
	by_cap_for_stability = {
		str(cap): {
			"status": "error" if any(branch[str(cap)].get("status") == "error" for branch in branch_records) else "ok",
			"truncated": any(branch[str(cap)].get("truncated") for branch in branch_records),
			"metrics": [branch[str(cap)] for branch in branch_records],
		}
		for cap in caps
	}
	# Stability uses the full branch vector, not only the first branch.
	if any(by_cap_for_stability[str(cap)]["status"] == "error" for cap in caps[1:]):
		stability = "EXECUTION_ERROR"
	elif len(set(classes.values())) > 1:
		stability = "CLASS_CHANGED"
	elif any(by_cap_for_stability[str(caps[-1])]["truncated"] for _ in [0]):
		stability = "UNRESOLVED_STILL_TRUNCATED"
	else:
		vectors = [
			tuple(metric_signature(branch[str(cap)]) for branch in branch_records)
			for cap in caps
		]
		stability = "STABLE_ALL_CAPS" if len(set(vectors)) == 1 else "STABLE_MATERIAL_EFFECT"
	return {
		"case_id": original["case_id"],
		"dataset": original["dataset"],
		"gold_structure": original["gold_structure"],
		"original_class": original.get("necessity_class"),
		"branch_results": branch_records,
		"recomputed_class_by_cap": classes,
		"stability": stability,
		"future_strong": original["case_id"] in FUTURE_STRONG_IDS,
		"future_control": original["case_id"] in FUTURE_CONTROL_IDS,
		"future_manifest_action": "KEEP" if stability in STABLE_LABELS else "REMOVE",
	}


def balanced_ids(ids: list[str], metadata: dict[str, dict[str, Any]], size: int) -> list[str]:
	groups: dict[tuple[str, str], list[str]] = defaultdict(list)
	for case_id in sorted(ids):
		item = metadata[case_id]
		groups[(item["dataset"], item["gold_structure"])].append(case_id)
	selected: list[str] = []
	while len(selected) < size and any(groups.values()):
		for key in sorted(groups):
			if groups[key] and len(selected) < size:
				selected.append(groups[key].pop(0))
	return selected


def write_report(
	 summary: dict[str, Any],
	 capped: list[dict[str, Any]],
	 results: list[dict[str, Any]],
	 future_manifest: dict[str, Any],
	 audit_status: str,
) -> None:
	future_strong_ids = set(future_manifest.get("strong_branch_benchmark_ids", []))
	future_control_ids = set(future_manifest.get("answer_equivalent_branch_control_ids", []))
	counts = Counter(row.get("stability", "NOT_RUN") for row in results)
	changed = [row for row in results if row.get("stability") == "CLASS_CHANGED"]
	unresolved = [row for row in results if row.get("stability") == "UNRESOLVED_STILL_TRUNCATED"]
	errors = [row for row in results if row.get("stability") == "EXECUTION_ERROR"]
	stable = [row for row in results if row.get("stability") in STABLE_LABELS]
	evaluable = len(results) - len(unresolved) - len(errors)
	stability_rate = len(stable) / evaluable if evaluable else 0.0
	strong_capped = [
		row for row in capped
		if row.get("necessity_class") == "STRONGLY_BRANCH_REQUIRED"
	]
	strong_result_by_id = {row.get("case_id"): row for row in results}
	strong_changed = [
		strong_result_by_id[row["case_id"]]
		for row in strong_capped
		if row["case_id"] in strong_result_by_id
		and strong_result_by_id[row["case_id"]].get("stability") in {"CLASS_CHANGED", "UNRESOLVED_STILL_TRUNCATED", "EXECUTION_ERROR"}
	]
	strong_downgraded_partial = [
		strong_result_by_id[row["case_id"]] for row in strong_capped
		if row["case_id"] in strong_result_by_id
		and any(value == "PARTIALLY_BRANCH_REQUIRED" for value in strong_result_by_id[row["case_id"]].get("recomputed_class_by_cap", {}).values())
	]
	strong_downgraded_control = [
		strong_result_by_id[row["case_id"]] for row in strong_capped
		if row["case_id"] in strong_result_by_id
		and any(value == "ANSWER_EQUIVALENT_LINEARIZATION" for value in strong_result_by_id[row["case_id"]].get("recomputed_class_by_cap", {}).values())
	]
	group_rows: dict[str, dict[str, int]] = {}
	for row in capped:
		key = f"{row['dataset']}|{row['gold_structure']}"
		group_rows.setdefault(key, {"capped": 0, "stable": 0, "changed": 0, "unresolved": 0})
		group_rows[key]["capped"] += 1
	for row in results:
		key = f"{row['dataset']}|{row['gold_structure']}"
		if row.get("stability") in STABLE_LABELS:
			group_rows[key]["stable"] += 1
		elif row.get("stability") == "CLASS_CHANGED":
			group_rows[key]["changed"] += 1
		elif row.get("stability") == "UNRESOLVED_STILL_TRUNCATED":
			group_rows[key]["unresolved"] += 1
		else:
			group_rows[key]["changed"] += 1
	decision = summary.get("decision", "SAFETY_CAP_INCONCLUSIVE")
	lines = [
		"# Phase 3A.1 safety-cap sensitivity check",
		"",
		"## 1. Motivation",
		"",
		"This check tests whether the Phase 3A benchmark labels depend materially on the 1,000-row counterfactual safety cap.",
		"",
		"## 2. Research question",
		"",
		"Do capped counterfactuals retain the same answer-effect conclusion when evaluated at higher result limits?",
		"",
		"## 3. Frozen methodology",
		"",
		"Candidate extraction, branch decomposition, query construction, stored-answer scoring, prior-ID exclusion, and necessity classification are imported unchanged from Phase 3A. Only the counterfactual row cap varies.",
		"",
		"## 4. Capped-case identification",
		"",
		f"Phase 3A contains **{len(capped)}** capped candidate records, verified from `branch_necessity_results.jsonl`. No uncapped candidate was rerun.",
		"",
		"## 5. Sensitivity cap configuration",
		"",
		f"Reference cap: **1,000**. Higher caps: `{summary.get('caps', [])}`. Phase 3A recorded capped branches at `{summary.get('phase3a_recorded_cap_counts', {})}`; the {summary.get('reference_cap_repaired_case_count', 0)} inherited non-1,000 cases were rerun at all comparison caps. Endpoint: `{summary.get('endpoint', '')}`.",
		"",
		"## 6. Execution results",
		"",
		f"Run status: **{summary.get('status')}**. Results processed: `{len(results)}/{len(capped)}`. Stability counts: `{dict(counts)}`.",
		"",
		"## 7. Classification stability",
		"",
		f"Stable cases: **{len(stable)}**; class-changed: **{len(changed)}**; unresolved: **{len(unresolved)}**; execution errors: **{len(errors)}**. Classification stability among evaluable cases: **{stability_rate:.3f}**.",
		"",
		"## 8. Strong-class sensitivity",
		"",
		f"Original strong capped cases: **{len(strong_capped)}**. Strong cases with any downgrade, unresolved result, or execution error: **{len(strong_changed)}**. Downgraded to partial at any cap: **{len(strong_downgraded_partial)}**; downgraded to answer-equivalent at any cap: **{len(strong_downgraded_control)}**.",
		"",
		"## 9. Dataset/structure breakdown",
		"",
		"| Group | Capped | Stable | Changed | Unresolved |",
		"|---|---:|---:|---:|---:|",
		*[f"| {key.replace('|', ' ')} | {value['capped']} | {value['stable']} | {value['changed']} | {value['unresolved']} |" for key, value in sorted(group_rows.items())],
		"",
		"## 10. Future-manifest impact",
		"",
		f"Affected future strong IDs: **{sum(row.get('case_id') in future_strong_ids for row in capped)}**. Affected future controls: **{sum(row.get('case_id') in future_control_ids for row in capped)}**. Unstable future IDs removed: **{sum(row.get('stability') in UNSTABLE_LABELS and (row.get('case_id') in future_strong_ids or row.get('case_id') in future_control_ids) for row in results)}**.",
		"",
		"## 11. Manual review",
		"",
		(
			f"Manual-review status: **{audit_status}**. All {len(unresolved)} unresolved records and five stable-material-effect cases were reviewed; no disagreements were recorded. Unresolved cases remain excluded from future evaluation where applicable."
			if audit_status == "COMPLETE"
			else f"Manual-review status: **{audit_status}**. All CLASS_CHANGED and UNRESOLVED_STILL_TRUNCATED records must be inspected before any manifest removal; five deterministic stable-material-effect cases are also sampled in `manual_review_cases.json`."
		),
		"",
		"## 12. Robustness gate",
		"",
		f"Required stability is 0.98; observed stability is {stability_rate:.3f}. The original strong-case change rate is {len(strong_changed) / len(strong_capped) if strong_capped else 0.0:.3f}. Decision gate result: **{decision}**.",
		"",
		"## 13. Final Phase 3A.1 decision",
		"",
		f"**{decision}**",
		"",
		"## 14. Frozen benchmark status",
		"",
		f"Frozen manifest status: `{summary.get('frozen_manifest_status', 'NOT_READY')}`. The Phase 3A future manifest is retained only when no affected future ID is unstable or unresolved.",
		"",
		"## 15. Phase 3B recommendation",
		"",
		"Do not execute Phase 3B in this phase. If the robustness gate passes, use the deterministic 20-strong/20-control sampling recommendation after reviewing the safety-cap cases.",
		"",
		"## 16. Limitations",
		"",
		"This check only tests capped counterfactuals and does not validate uncapped cases again. If the highest cap remains truncated or the endpoint is unavailable, the result is inconclusive rather than evidence of stability.",
	]
	(OUTPUT_ROOT / "PHASE3A_1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	phase3a_results = read_jsonl(PHASE3A_ROOT / "branch_necessity_results.jsonl")
	phase3a_future = read_json(PHASE3A_ROOT / "future_eval_manifest.json")
	candidates, _ = load_candidates()
	candidate_by_id = {candidate["case_id"]: candidate for candidate in candidates}
	candidate_metadata = {
		candidate["case_id"]: {
			"dataset": candidate["dataset"],
			"gold_structure": candidate["gold_structure"],
		}
		for candidate in candidates
	}
	results_by_id = {row["case_id"]: row for row in phase3a_results}
	capped = capped_cases(phase3a_results)
	prior_ids = known_prior_ids()
	global FUTURE_STRONG_IDS, FUTURE_CONTROL_IDS
	FUTURE_STRONG_IDS = set(phase3a_future.get("strong_branch_benchmark_ids", []))
	FUTURE_CONTROL_IDS = set(phase3a_future.get("answer_equivalent_branch_control_ids", []))
	caps = sorted(set(args.caps))
	if REFERENCE_CAP not in caps:
		caps.insert(0, REFERENCE_CAP)
	manifest = {
		"phase3a_root": str(PHASE3A_ROOT),
		"selection_policy": "all and only Phase 3A candidate records with at least one result_truncated=true",
		"candidate_count": len(capped),
		"reference_cap": REFERENCE_CAP,
		"higher_caps": [cap for cap in caps if cap != REFERENCE_CAP],
		"model_outcomes_used": False,
		"cases": [
			{
				"case_id": row["case_id"],
				"dataset": row["dataset"],
				"gold_structure": row["gold_structure"],
				"original_class": row.get("necessity_class"),
				"capped_branch_indices": [index for index, branch in enumerate(row.get("branches", [])) if branch.get("result_truncated")],
				"recorded_result_limits": sorted({branch.get("result_limit") for branch in row.get("branches", []) if branch.get("result_truncated")}),
				"future_strong": row["case_id"] in FUTURE_STRONG_IDS,
				"future_control": row["case_id"] in FUTURE_CONTROL_IDS,
			}
			for row in capped
		],
	}
	write_json(OUTPUT_ROOT / "capped_case_manifest.json", manifest)

	endpoint = args.endpoint or os.getenv("FREEBASE_SPARQL_ENDPOINT", DEFAULT_ENDPOINT)
	probe = preflight(endpoint, args.timeout)
	results: list[dict[str, Any]] = []
	partial_path = OUTPUT_ROOT / "sensitivity_results.partial.jsonl"
	if args.resume and partial_path.exists():
		results.extend(read_jsonl(partial_path))
	result_by_id = {row.get("case_id"): row for row in results}
	completed = set(result_by_id)
	# Three inherited Phase 3A rows were produced with a 10,000-row cap during
	# the earlier resumed run.  Recompute those cases at every requested cap so
	# the 1,000-row reference is genuinely comparable.
	for original in capped:
		if any(
			branch.get("result_truncated")
			and branch.get("result_limit") != REFERENCE_CAP
			for branch in original.get("branches", [])
		):
			completed.discard(original["case_id"])
	status = "COMPLETE"
	if args.no_execute or not probe["available"]:
		status = "BLOCKED_ENDPOINT_UNAVAILABLE" if not probe["available"] else "PREPARED_NO_EXECUTE"
	else:
		for index, original in enumerate(capped, start=1):
			if original["case_id"] in completed:
				continue
			candidate = candidate_by_id[original["case_id"]]
			result_by_id[original["case_id"]] = build_case_record(candidate, original, endpoint, args.timeout, caps)
			results = list(result_by_id.values())
			if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
				write_jsonl(partial_path, results)
			if args.progress and (index == 1 or index % args.progress == 0):
				print(f"sensitivity evaluated {index}/{len(capped)}: {original['case_id']}")

	# If the endpoint becomes unavailable during execution, incomplete cases remain explicit.
	results = list(result_by_id.values())
	results.sort(key=lambda row: (row.get("dataset", ""), row.get("case_id", "")))
	write_jsonl(OUTPUT_ROOT / "sensitivity_results.jsonl", results)
	counts = Counter(row.get("stability", "NOT_RUN") for row in results)
	stable = [row for row in results if row.get("stability") in STABLE_LABELS]
	changed = [row for row in results if row.get("stability") == "CLASS_CHANGED"]
	unresolved = [row for row in results if row.get("stability") == "UNRESOLVED_STILL_TRUNCATED"]
	errors = [row for row in results if row.get("stability") == "EXECUTION_ERROR"]
	evaluable = len(results) - len(unresolved) - len(errors)
	stability_rate = len(stable) / evaluable if evaluable else 0.0
	strong_capped = [row for row in capped if row.get("necessity_class") == "STRONGLY_BRANCH_REQUIRED"]
	strong_result_by_id = {row.get("case_id"): row for row in results}
	strong_changed = [
		strong_result_by_id[row["case_id"]]
		for row in strong_capped
		if row["case_id"] in strong_result_by_id
		and strong_result_by_id[row["case_id"]].get("stability") in UNSTABLE_LABELS
	]
	strong_change_rate = len(strong_changed) / len(strong_capped) if strong_capped else 0.0
	if status != "COMPLETE":
		decision = "SAFETY_CAP_INCONCLUSIVE"
	elif unresolved or errors:
		decision = "SAFETY_CAP_INCONCLUSIVE"
	elif stability_rate >= 0.98 and strong_change_rate <= 0.02:
		decision = "SAFETY_CAP_ROBUST"
	elif stability_rate >= 0.98:
		decision = "SAFETY_CAP_MINOR_SENSITIVITY"
	else:
		decision = "SAFETY_CAP_MATERIAL_SENSITIVITY"
	unstable_ids = {row["case_id"] for row in results if row.get("stability") in UNSTABLE_LABELS}
	future_strong_keep = [case_id for case_id in FUTURE_STRONG_IDS if case_id not in unstable_ids]
	future_control_keep = [case_id for case_id in FUTURE_CONTROL_IDS if case_id not in unstable_ids]
	frozen_identical = status == "COMPLETE" and not unstable_ids.intersection(FUTURE_STRONG_IDS | FUTURE_CONTROL_IDS)
	frozen_manifest = dict(phase3a_future)
	frozen_manifest.update({
		"source_manifest": str(PHASE3A_ROOT / "future_eval_manifest.json"),
		"phase3a_1_decision": decision,
		"identical_to_phase3a_manifest": frozen_identical,
		"verification_status": "VERIFIED" if status == "COMPLETE" else "NOT_VERIFIED",
		"removed_unstable_or_unresolved_ids": sorted(unstable_ids.intersection(FUTURE_STRONG_IDS | FUTURE_CONTROL_IDS)),
		"strong_branch_benchmark_ids": sorted(future_strong_keep),
		"answer_equivalent_branch_control_ids": sorted(future_control_keep),
		"strong_case_count": len(future_strong_keep),
		"control_case_count": len(future_control_keep),
	})
	frozen_manifest_status = (
		"IDENTICAL_TO_PHASE3A"
		if frozen_identical
		else "NOT_VERIFIED_ENDPOINT_UNAVAILABLE"
		if status != "COMPLETE"
		else "FILTERED_UNSTABLE_IDS"
	)
	strong_sample = balanced_ids(future_strong_keep, candidate_metadata, min(20, len(future_strong_keep)))
	control_sample = balanced_ids(future_control_keep, candidate_metadata, min(20, len(future_control_keep)))
	write_json(OUTPUT_ROOT / "frozen_future_eval_manifest.json", frozen_manifest)
	write_json(OUTPUT_ROOT / "cap_comparison_summary.json", {
		"status": status,
		"endpoint": endpoint,
		"preflight": probe,
		"caps": caps,
		"phase3a_recorded_cap_counts": dict(Counter(
			branch.get("result_limit")
			for row in capped
			for branch in row.get("branches", [])
			if branch.get("result_truncated")
		)),
		"reference_cap_repaired_case_count": sum(
			any(branch.get("result_truncated") and branch.get("result_limit") != REFERENCE_CAP for branch in row.get("branches", []))
			for row in capped
		),
		"candidate_count": len(capped),
		"evaluated_count": len(results),
		"stability_counts": dict(counts),
		"classification_stability_rate": stability_rate,
		"original_strong_capped_count": len(strong_capped),
		"original_strong_changed_count": len(strong_changed),
		"original_strong_change_rate": strong_change_rate,
		"future_strong_affected_count": sum(row["case_id"] in FUTURE_STRONG_IDS for row in capped),
		"future_control_affected_count": sum(row["case_id"] in FUTURE_CONTROL_IDS for row in capped),
		"removed_future_ids": sorted(unstable_ids.intersection(FUTURE_STRONG_IDS | FUTURE_CONTROL_IDS)),
		"decision": decision,
		"frozen_manifest_status": frozen_manifest_status,
	})
	strong_lines = [
		"# Strong-class sensitivity",
		"",
		f"Original strongly branch-required capped cases: **{len(strong_capped)}**.",
		f"Strong cases retained: **{len(strong_capped) - len(strong_changed)}**.",
		f"Strong cases downgraded to partial at any cap: **{sum(any(value == 'PARTIALLY_BRANCH_REQUIRED' for value in strong_result_by_id.get(row['case_id'], {}).get('recomputed_class_by_cap', {}).values()) for row in strong_capped)}**.",
		f"Strong cases downgraded to answer-equivalent at any cap: **{sum(any(value == 'ANSWER_EQUIVALENT_LINEARIZATION' for value in strong_result_by_id.get(row['case_id'], {}).get('recomputed_class_by_cap', {}).values()) for row in strong_capped)}**.",
		f"Strong cases unresolved: **{sum(strong_result_by_id.get(row['case_id'], {}).get('stability') == 'UNRESOLVED_STILL_TRUNCATED' for row in strong_capped)}**.",
		f"Strong cases with execution errors: **{sum(strong_result_by_id.get(row['case_id'], {}).get('stability') == 'EXECUTION_ERROR' for row in strong_capped)}**.",
	]
	(OUTPUT_ROOT / "strong_class_sensitivity.md").write_text("\n".join(strong_lines) + "\n", encoding="utf-8")
	impact_lines = [
		"# Future-manifest impact",
		"",
		f"Affected strong IDs: **{sum(row.get('future_strong', False) for row in results)}**.",
		f"Affected control IDs: **{sum(row.get('future_control', False) for row in results)}**.",
		f"Removed future IDs: `{sorted(unstable_ids.intersection(FUTURE_STRONG_IDS | FUTURE_CONTROL_IDS))}`.",
		f"Frozen manifest status: **{frozen_manifest_status}**.",
	]
	(OUTPUT_ROOT / "future_manifest_impact.md").write_text("\n".join(impact_lines) + "\n", encoding="utf-8")
	manual_cases = changed + unresolved
	manual_cases.extend([row for row in results if row.get("stability") == "STABLE_MATERIAL_EFFECT"][:5])
	write_json(OUTPUT_ROOT / "manual_review_cases.json", {
		"status": "PENDING_EXECUTION" if status != "COMPLETE" else "PENDING_MANUAL_REVIEW" if manual_cases else "NO_CASES_REQUIRED",
		"case_ids": [row["case_id"] for row in manual_cases],
		"records": manual_cases,
	})
	recommendation = [
		"# Phase 3B sampling recommendation",
		"",
		"Do not execute Phase 3B from this artifact.",
		"",
		f"Decision prerequisite: `{decision}`.",
		f"Recommended strong sample ({len(strong_sample)} cases): `{strong_sample}`.",
		f"Recommended answer-equivalent control sample ({len(control_sample)} cases): `{control_sample}`.",
		"",
		"Sampling is deterministic and round-robin across dataset and structural subtype groups; it is not based on model outcomes.",
	]
	(OUTPUT_ROOT / "phase3b_sampling_recommendation.md").write_text("\n".join(recommendation) + "\n", encoding="utf-8")
	write_report({
		"status": status,
		"endpoint": endpoint,
		"caps": caps,
		"phase3a_recorded_cap_counts": dict(Counter(
			branch.get("result_limit")
			for row in capped
			for branch in row.get("branches", [])
			if branch.get("result_truncated")
		)),
		"reference_cap_repaired_case_count": sum(
			any(branch.get("result_truncated") and branch.get("result_limit") != REFERENCE_CAP for branch in row.get("branches", []))
			for row in capped
		),
		"decision": decision,
		"frozen_manifest_status": frozen_manifest_status,
	}, capped, results, phase3a_future, "PENDING_EXECUTION" if status != "COMPLETE" else "PENDING_MANUAL_REVIEW" if manual_cases else "NO_CASES_REQUIRED")
	return {
		"status": status,
		"candidate_count": len(capped),
		"evaluated_count": len(results),
		"caps": caps,
		"endpoint": endpoint,
		"decision": decision,
		"output_root": str(OUTPUT_ROOT),
	}


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--run", action="store_true", help="run higher-cap queries for capped Phase 3A records")
	parser.add_argument("--no-execute", action="store_true", help="prepare outputs without live endpoint execution")
	parser.add_argument("--endpoint", default=None)
	parser.add_argument("--timeout", type=int, default=int(os.getenv("PHASE3A1_SPARQL_TIMEOUT", "60")))
	parser.add_argument("--caps", nargs="+", type=int, default=list(DEFAULT_CAPS))
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--checkpoint-every", type=int, default=10)
	parser.add_argument("--progress", type=int, default=10)
	args = parser.parse_args()
	if not args.run and not args.no_execute:
		parser.error("use --run to execute or --no-execute to prepare")
	print(json.dumps(run(args), indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
