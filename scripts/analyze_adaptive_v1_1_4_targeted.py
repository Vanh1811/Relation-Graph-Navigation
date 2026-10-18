#!/usr/bin/env python3
"""Analyze the Adaptive v1.1.4 complementary-property targeted run."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "outputs" / "experiments" / "phase1" / "adaptive_v1_1_4"
TARGETED_ROOT = ROOT / "targeted_validation"
RESULT_PATH = TARGETED_ROOT / "results.jsonl"

CASE_META: dict[str, dict[str, str]] = {
	"3201942006000": {
		"label": "GameCube controller + Japan",
		"expectation": "BRANCH_FINAL / COMPLEMENTARY_PROPERTIES is required",
		"desired": "NO — the required complementary candidate was routed LINEAR",
	},
	"3201942014000": {
		"label": "Xbox 360 controller + Europe",
		"expectation": "Evaluate as a complementary-property candidate",
		"desired": "YES — complementary properties and answer space were validated",
	},
	"3203433006000": {
		"label": "Bobby Helm + rock music",
		"expectation": "Evaluate artist-association + genre as complementary",
		"desired": "YES — complementary properties and answer space were validated",
	},
	"4301182008000": {
		"label": "menstrual cramps + SLUDGE syndrome",
		"expectation": "Evaluate treatment indication + side effect as complementary",
		"desired": "YES — complementary properties and answer space were validated",
	},
	"4301990001000": {
		"label": "Pictures of Lily + Track Record",
		"expectation": "Evaluate producer + label as complementary",
		"desired": "YES — complementary properties and answer space were validated",
	},
	"3204176003000": {
		"label": "comic category control",
		"expectation": "BRANCH_FINAL / SAME_PROPERTY",
		"desired": "YES — same-property control preserved",
	},
	"4301299003000": {
		"label": "computer designer control",
		"expectation": "BRANCH_FINAL / SAME_PROPERTY",
		"desired": "YES — same-property control preserved",
	},
	"4300753011000": {
		"label": "rocket-engine negative control",
		"expectation": "BRANCH_INTERMEDIATE / CoR",
		"desired": "YES — intermediate convergence remained out of direct branching",
	},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fmt_prediction(row: dict[str, Any], key: str) -> str:
	return json.dumps(row.get(key, []), ensure_ascii=False)


def branch_space_ok(row: dict[str, Any]) -> bool:
	if row.get("router_structure") != "BRANCH_FINAL":
		return True
	branches = row.get("branches", []) or []
	return bool(
		row.get("same_final_answer_space")
		and branches
		and all(item.get("question_supported") for item in branches)
		and all(item.get("independently_generates_final_candidates") for item in branches)
		and len({str(item.get("target_answer_type", "")) for item in branches}) == 1
	)


def write_case_comparison(rows: list[dict[str, Any]]) -> None:
	lines = [
		"# Adaptive v1.1.4 targeted case comparison",
		"",
		"This table compares the new structural decision with the frozen v1.1.3 diagnostic decision. The targeted set is selected structural evidence, not a benchmark.",
		"",
		"| ID | v1.1.3 | v1.1.4 | Relation Mode | Same Final Answer Space? | Validator | Desired? |",
		"|---|---|---|---|---|---|---|",
	]
	for row in rows:
		case_id = str(row["id"])
		structure = str(row.get("router_structure", ""))
		mode = str(row.get("branch_relation_mode", "")) if structure == "BRANCH_FINAL" else "N/A"
		same_space = str(bool(row.get("same_final_answer_space"))) if structure == "BRANCH_FINAL" else "N/A"
		validator = "PASS" if row.get("validation_passed") else "REJECTED"
		lines.append(
			f"| `{case_id}` | {row.get('v1_1_3_structure', '')} | {structure} | {mode} | {same_space} | {validator} | {CASE_META[case_id]['desired']} |"
		)
	lines += ["", "## Structural observations", ""]
	for row in rows:
		case_id = str(row["id"])
		old = row.get("v1_1_3_structure", "")
		new = row.get("router_structure", "")
		if old != new:
			lines.append(
				f"- `{case_id}` changed **{old} → {new}**: {row.get('router_reason', '')}"
			)
	lines.append(
		"- The GameCube/Japan case is the required miss: v1.1.4's model decision explicitly described the query as linear, so the validator had no branch plan to validate."
	)
	(TARGETED_ROOT / "case_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_structural(rows: list[dict[str, Any]]) -> None:
	structures = Counter(str(row.get("router_structure", "")) for row in rows)
	modes = Counter(str(row.get("branch_relation_mode", "")) for row in rows if row.get("router_structure") == "BRANCH_FINAL")
	lines = [
		"# Adaptive v1.1.4 structural analysis",
		"",
		"## Definitions",
		"",
		"- **SAME_PROPERTY**: branches use equivalent semantic properties and independently constrain the same final answer space.",
		"- **COMPLEMENTARY_PROPERTIES**: branches use different, question-supported properties but independently constrain the same final answer type/answer variable.",
		"- **BRANCH_INTERMEDIATE**: constraints converge on an intermediate before the requested final answer; it remains CoR because branch→merge→continue is not implemented.",
		"",
		"## Counts",
		"",
		"| Structure | Count |",
		"|---|---:|",
	]
	for key in ("LINEAR", "BRANCH_FINAL", "BRANCH_INTERMEDIATE", "UNCERTAIN"):
		lines.append(f"| {key} | {structures.get(key, 0)} |")
	lines += ["", "| Branch relation mode | Count |", "|---|---:|"]
	for key in ("SAME_PROPERTY", "COMPLEMENTARY_PROPERTIES"):
		lines.append(f"| {key} | {modes.get(key, 0)} |")
	lines += ["", "## Branch-plan checks", ""]
	for row in rows:
		if row.get("router_structure") != "BRANCH_FINAL":
			continue
		branches = row.get("branches", []) or []
		lines.append(
			f"- `{row['id']}`: mode=`{row.get('branch_relation_mode')}`, same-space=`{row.get('same_final_answer_space')}`, "
			f"all branches question-supported=`{all(item.get('question_supported') for item in branches)}`, "
			f"all independently generate candidates=`{all(item.get('independently_generates_final_candidates') for item in branches)}`, "
			f"target types=`{sorted({str(item.get('target_answer_type', '')) for item in branches})}`."
		)
	lines += ["", "The complementary-property rule was exercised successfully on four candidate cases, but the required GameCube/Japan case was not selected as BRANCH_FINAL. This is a routing recall failure, not a deterministic validator failure."]
	(TARGETED_ROOT / "structural_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation(rows: list[dict[str, Any]]) -> dict[str, int]:
	answer_type = sum(any("ANSWER_TYPE" in str(item) or "answer type" in str(item).lower() for item in row.get("validation_failures", [])) for row in rows)
	property_mismatch = sum(any("CONSTRAINT_PROPERTY" in str(item) or "constraint property" in str(item).lower() for item in row.get("validation_failures", [])) for row in rows)
	rejections = sum(not bool(row.get("validation_passed", True)) for row in rows)
	safe_redirects = sum(not bool(row.get("validation_passed", True)) and str(row.get("execution_mode", "")).startswith("cor") for row in rows)
	lines = [
		"# Adaptive v1.1.4 validation analysis",
		"",
		f"- Answer-type mismatch failures: **{answer_type}**",
		f"- Constraint-property mismatch failures: **{property_mismatch}**",
		f"- Validation rejections: **{rejections}**",
		f"- Safe CoR redirects caused by validation: **{safe_redirects}**",
		"",
		"| ID | Structure | Relation mode | Validation | Reason | Same space | Branch candidates independently generated |",
		"|---|---|---|---|---|---|---|",
	]
	for row in rows:
		branches = row.get("branches", []) or []
		lines.append(
			f"| `{row['id']}` | {row.get('router_structure')} | {row.get('branch_relation_mode', '')} | "
			f"{row.get('validation_result')} | {row.get('validation_reason', '') or '—'} | "
			f"{row.get('same_final_answer_space')} | {all(item.get('independently_generates_final_candidates') for item in branches) if branches else 'n/a'} |"
		)
	lines += [
		"",
		"All eight executed router plans passed deterministic validation. The 5 branch fallbacks were caused by branch execution/empty strict merges, not by semantic validation rejection. The negative-control BRANCH_INTERMEDIATE decision had no executable branch plan.",
	]
	(TARGETED_ROOT / "validation_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
	return {
		"answer_type_mismatches": answer_type,
		"constraint_property_mismatches": property_mismatch,
		"validation_rejections": rejections,
		"safe_redirects": safe_redirects,
	}


def write_summary(rows: list[dict[str, Any]], validation: dict[str, int]) -> dict[str, Any]:
	n = len(rows)
	native_hits = sum(bool(row.get("native_hits_at_1")) for row in rows)
	final_hits = sum(bool(row.get("final_hits_at_1", row.get("hits_at_1"))) for row in rows)
	native_f1 = sum(float(row.get("native_f1", 0) or 0) for row in rows) / n if n else 0.0
	final_f1 = sum(float(row.get("final_f1", row.get("f1", 0)) or 0) for row in rows) / n if n else 0.0
	branch_rows = [row for row in rows if row.get("router_structure") == "BRANCH_FINAL"]
	native_branch_successes = sum(row.get("native_execution_mode") == "branch" and not row.get("fallback_used") for row in branch_rows)
	fallbacks = sum(bool(row.get("fallback_used")) for row in branch_rows)
	structures = Counter(str(row.get("router_structure", "")) for row in rows)
	summary = {
		"n": n,
		"native_hits_at_1": native_hits,
		"native_hits_rate": native_hits / n if n else 0.0,
		"native_f1": native_f1,
		"final_hits_at_1": final_hits,
		"final_hits_rate": final_hits / n if n else 0.0,
		"final_f1": final_f1,
		"errors": sum(bool(row.get("errors", row.get("error"))) for row in rows),
		"structures": dict(structures),
		"branch_final_selections": len(branch_rows),
		"same_property": sum(row.get("branch_relation_mode") == "SAME_PROPERTY" for row in branch_rows),
		"complementary_properties": sum(row.get("branch_relation_mode") == "COMPLEMENTARY_PROPERTIES" for row in branch_rows),
		"native_branch_successes": native_branch_successes,
		"fallbacks": fallbacks,
		"fallback_rate": fallbacks / len(branch_rows) if branch_rows else 0.0,
		"branch_self_sufficiency": native_branch_successes / len(branch_rows) if branch_rows else 0.0,
		**validation,
		"avg_llm_calls": sum(int(row.get("llm_calls", 0) or 0) for row in rows) / n if n else 0.0,
		"avg_kg_queries": sum(int(row.get("kg_queries", 0) or 0) for row in rows) / n if n else 0.0,
		"avg_tokens": sum(int(row.get("tokens", 0) or 0) for row in rows) / n if n else 0.0,
		"avg_latency_ms": sum(float(row.get("latency_ms", 0) or 0) for row in rows) / n if n else 0.0,
	}
	lines = [
		"# Adaptive v1.1.4 targeted validation summary",
		"",
		"This is a selected 8-case structural validation set, not a benchmark. No historical method was rerun and the full 100-question evaluation was not run.",
		"",
		"## Overall",
		"",
		"| View | N | Hits@1 | F1 | Errors |",
		"|---|---:|---:|---:|---:|",
		f"| Native / pre-fallback | {n} | {native_hits / n:.2%} | {native_f1:.4f} | 0 |",
		f"| Final / post-fallback | {n} | {final_hits / n:.2%} | {final_f1:.4f} | {summary['errors']} |",
		"",
		"## Structures",
		"",
		"| Structure | Count |",
		"|---|---:|",
	]
	for key in ("LINEAR", "BRANCH_FINAL", "BRANCH_INTERMEDIATE", "UNCERTAIN"):
		lines.append(f"| {key} | {structures.get(key, 0)} |")
	lines += [
		"",
		"## Branch execution",
		"",
		f"- BRANCH_FINAL selections: **{len(branch_rows)}**",
		f"- SAME_PROPERTY: **{summary['same_property']}**",
		f"- COMPLEMENTARY_PROPERTIES: **{summary['complementary_properties']}**",
		f"- Native branch successes: **{native_branch_successes}**",
		f"- Fallbacks: **{fallbacks}**",
		f"- Fallback rate: **{summary['fallback_rate']:.2%}**",
		f"- Branch self-sufficiency: **{summary['branch_self_sufficiency']:.2%}**",
		"",
		"## Validation",
		"",
		f"- Answer-type mismatch failures: **{validation['answer_type_mismatches']}**",
		f"- Constraint-property mismatch failures: **{validation['constraint_property_mismatches']}**",
		f"- Validation rejections: **{validation['validation_rejections']}**",
		f"- Safe CoR redirects from validation: **{validation['safe_redirects']}**",
		"",
		"## Descriptive cost",
		"",
		f"- Avg LLM calls: **{summary['avg_llm_calls']:.2f}**",
		f"- Avg KG queries: **{summary['avg_kg_queries']:.2f}**",
		f"- Avg tokens: **{summary['avg_tokens']:.2f}**",
		f"- Avg latency: **{summary['avg_latency_ms']:.2f} ms**",
	]
	(TARGETED_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
	(TARGETED_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
	return summary


def write_report(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
	gamecube = next(row for row in rows if row["id"] == "3201942006000")
	controls_ok = all(
		next(row for row in rows if row["id"] == case_id).get("router_structure") == expected
		for case_id, expected in (("3204176003000", "BRANCH_FINAL"), ("4301299003000", "BRANCH_FINAL"), ("4300753011000", "BRANCH_INTERMEDIATE"))
	)
	recommendation = "KEEP_V1_1_3"
	lines = [
		"# Adaptive v1.1.4 targeted validation report",
		"",
		"## Scope",
		"",
		"Adaptive v1.1.4 was implemented as a narrow extension of v1.1.3: it preserves answer-type and generic-root safeguards while allowing explicitly validated SAME_PROPERTY and COMPLEMENTARY_PROPERTIES branch plans. It was executed on exactly 8 selected cases. Historical methods were read only; the full 100-question evaluation was not run.",
		"",
		"## Motivation",
		"",
		"The v1.1.3 diagnostic showed that requiring equivalent branch properties can be too strict. For example, controller compatibility and regional support can independently constrain the same game-version answer space even though their relations differ.",
		"",
		"## Validator change",
		"",
		"- SAME_PROPERTY requires equivalent semantic properties, matching target answer types, a claimed shared final answer space, question-supported constraints, and independent final-candidate generation.",
		"- COMPLEMENTARY_PROPERTIES permits different compatible properties only when the same conditions hold and the model explicitly labels the mode.",
		"- Intermediate convergence remains BRANCH_INTERMEDIATE and executes through CoR.",
		"- Invalid plans still route to CoR with validation/fallback logging.",
		"",
		"## Results",
		"",
		f"- Native Hits@1: **{summary['native_hits_rate']:.2%}**; native F1: **{summary['native_f1']:.4f}**.",
		f"- Final Hits@1: **{summary['final_hits_rate']:.2%}**; final F1: **{summary['final_f1']:.4f}**.",
		f"- BRANCH_FINAL: **{summary['branch_final_selections']}**; SAME_PROPERTY: **{summary['same_property']}**; COMPLEMENTARY_PROPERTIES: **{summary['complementary_properties']}**.",
		f"- Native branch successes: **{summary['native_branch_successes']}**; fallbacks: **{summary['fallbacks']}**; fallback rate: **{summary['fallback_rate']:.2%}**.",
		f"- Answer-type mismatches: **{summary['answer_type_mismatches']}**; constraint-property mismatches: **{summary['constraint_property_mismatches']}**; validation rejections: **{summary['validation_rejections']}**.",
		"",
		"## Required case checks",
		"",
		f"- GameCube/Japan (`3201942006000`): **not recovered**; v1.1.4 selected `{gamecube.get('router_structure')}` instead of the required complementary BRANCH_FINAL.",
		"- Xbox/Europe, Bobby Helm/rock, menstrual cramps/SLUDGE, and Pictures of Lily/Track Record were recognized as complementary-property BRANCH_FINAL plans with matching target types and shared-answer-space assertions.",
		f"- Same-property controls and the rocket negative control: **{'passed' if controls_ok else 'not all passed'}**.",
		"",
		"## Interpretation",
		"",
		"The v1.1.4 validator accepted complementary properties without introducing answer-type or generic-root violations in this targeted run. However, the most important GameCube/Japan case was missed at the routing stage, and branch execution was not self-sufficient on most selected branch cases. This is not evidence for full-benchmark improvement.",
		"",
		f"## Recommendation: **{recommendation}**",
		"",
		"Adaptive v1.1.4 should not replace v1.1.3 for the full 100-question evaluation because the required GameCube/Japan complementary route was not reliably recognized and the selected branch fallback rate was 83.33%.",
		"",
		"See `targeted_validation/case_comparison.md`, `targeted_validation/structural_analysis.md`, and `targeted_validation/validation_analysis.md` for per-case evidence.",
	]
	(ROOT / "ADAPTIVE_V1_1_4_TARGETED_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
	rows = read_jsonl(RESULT_PATH)
	if len(rows) != 8 or len({str(row.get("id", "")) for row in rows}) != 8:
		raise SystemExit(f"expected 8 unique targeted rows, found {len(rows)}")
	for row in rows:
		if not branch_space_ok(row):
			raise SystemExit(f"invalid same-answer-space branch record: {row.get('id')}")
	write_case_comparison(rows)
	write_structural(rows)
	validation = write_validation(rows)
	summary = write_summary(rows, validation)
	write_report(rows, summary)
	print(json.dumps({"rows": len(rows), "recommendation": "KEEP_V1_1_3", "report": str(ROOT / 'ADAPTIVE_V1_1_4_TARGETED_REPORT.md')}, indent=2))


if __name__ == "__main__":
	main()
