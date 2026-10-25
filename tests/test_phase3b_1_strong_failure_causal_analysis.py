from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_phase3b_1_strong_failure_causal_analysis as audit  # noqa: E402


def _candidate(structure: str = "BRANCH_INTERMEDIATE") -> dict:
	return {
		"case_id": "case-1",
		"dataset": "cwq",
		"gold_structure": structure,
		"topic_entities": ["m.source_a", "m.source_b"],
		"topic_entity_names": ["Source A", "Source B"],
		"question": "test question",
		"gold_answers": ["Answer"],
		"answer_variable": "?x",
		"shared_target_variable": "?y",
		"patterns": [],
		"branches": [],
	}


def _row(**updates: object) -> dict:
	row = {
		"raw_parsed_result": {
			"shared_target": "ANSWER",
			"dependency_pattern": "parallel independent constraints",
			"independent_source_groups": [{"source_anchor": {"id": "m.source_a"}}, {"source_anchor": {"id": "m.source_b"}}],
			"constraints": [{"source_anchor": "m.source_a"}, {"source_anchor": "m.source_b"}],
		},
		"canonical_semantic_representation": {"structure": "BRANCH_FINAL"},
		"normalization_status": "NORMALIZED_SUCCESSFULLY",
		"serialization_status": "COMPLETE",
		"validator_decision": "PASS",
		"branch_attempted": True,
		"branch_self_sufficient": False,
		"execution_status": "BRANCH_ATTEMPTED_FALLBACK",
		"fallback_used": True,
		"fallback_reason": "empty_strict_merge",
		"hits_at_1": 0,
		"prediction": [],
	}
	row.update(updates)
	return row


def test_selected_target_cases_have_required_paired_counts() -> None:
	paired = audit.read_json(audit.INPUT_ROOT / "paired_analysis.json")
	candidates, _ = audit.load_candidates()
	selected = audit.selected_cases(paired, {row["case_id"]: row for row in candidates})
	assert len(selected) == 12
	assert {row["paired_outcome"] for row in selected} == {"BOTH_WRONG", "V1_1_3_ONLY", "PHASE2_5_6_ONLY"}
	assert sum(row["paired_outcome"] == "BOTH_WRONG" for row in selected) == 10


def test_first_divergence_precedence_keeps_shared_target_before_execution() -> None:
	candidate = _candidate("BRANCH_INTERMEDIATE")
	cause, _, semantic = audit.first_divergence(candidate, _row())
	assert cause == "SHARED_TARGET_ERROR"
	assert semantic["shared_target"] == "WRONG"


def test_normalization_is_first_after_correct_semantics() -> None:
	candidate = _candidate("BRANCH_INTERMEDIATE")
	row = _row(
		raw_parsed_result={
			"shared_target": "INTERMEDIATE",
			"dependency_pattern": "intermediate continuation",
			"independent_source_groups": [{"source_anchor": {"id": "m.source_a"}}, {"source_anchor": {"id": "m.source_b"}}],
			"constraints": [{"source_anchor": "m.source_a"}, {"source_anchor": "m.source_b"}],
		},
		normalization_status="PARTIAL_NORMALIZATION",
		validator_decision="FALLBACK",
	)
	cause, _, semantic = audit.first_divergence(candidate, row)
	assert semantic["semantic_adequate"] is True
	assert cause == "NORMALIZATION_ERROR"


def test_case_level_tie_is_mixed_and_gate_requires_independent_cases() -> None:
	assert audit.case_primary(["SHARED_TARGET_ERROR", "NORMALIZATION_ERROR"]) == "MIXED_FAILURES"
	summary = {
		"semantic_extraction_gate": {"independent_cases": 3, "datasets": ["cwq", "grailqa"], "supported": True},
		"intermediate_execution_gate": {"independent_cases": 0, "supported": False},
		"relation_grounding_gate": {"independent_cases": 0, "supported": False},
		"validator_or_serialization_gate": {"independent_cases": 2, "supported": False},
		"independent_case_primary_counts": {"SHARED_TARGET_ERROR": 3},
	}
	assert audit.decision(summary) == "SEMANTIC_EXTRACTION_BOTTLENECK_SUPPORTED"
