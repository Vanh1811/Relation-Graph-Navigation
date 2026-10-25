from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_phase3b_frozen_strong_vs_control as phase3b  # noqa: E402


def _row(case_id: str, cohort: str, method: str, hit: int, f1: float, repeat: int = 1) -> dict:
	return {
		"case_id": case_id,
		"cohort": cohort,
		"dataset": "cwq",
		"gold_structure": "BRANCH_FINAL",
		"method": method,
		"repeat": repeat,
		"hits_at_1": hit,
		"f1": f1,
		"error": None,
		"fallback_used": False,
		"branch_attempted": False,
		"branch_self_sufficient": False,
		"router_structure": "BRANCH_FINAL",
		"execution_status": "COR_EXECUTION",
	}


def test_sample_manifest_has_exact_frozen_counts() -> None:
	manifest, selected = phase3b.build_sample_manifest()
	assert manifest["strong_case_count"] == 20
	assert manifest["control_case_count"] == 20
	assert len(selected) == 40
	assert len({item["case_id"] for item in selected}) == 40
	assert manifest["strong_allocation"] == {
		"cwq::BRANCH_FINAL": 5,
		"cwq::BRANCH_INTERMEDIATE": 5,
		"grailqa::BRANCH_FINAL": 5,
		"grailqa::BRANCH_INTERMEDIATE": 5,
	}
	selected_ids = {item["case_id"] for item in selected}
	assert selected_ids.isdisjoint(phase3b.manual_ids())
	assert selected_ids.isdisjoint(phase3b.known_prior_ids())
	assert manifest["source_manifest_verification_status"] == "VERIFIED"


def test_paired_outcomes_are_separated_by_cohort() -> None:
	rows = [
		_row("strong-a", "strong", "v1_1_3", 1, 1.0),
		_row("strong-a", "strong", "phase2_5_6", 0, 0.0),
		_row("control-a", "control", "v1_1_3", 0, 0.0),
		_row("control-a", "control", "phase2_5_6", 1, 1.0),
	]
	paired = phase3b.paired_analysis(rows)
	assert paired["counts"]["strong"]["V1_1_3_ONLY"] == 1
	assert paired["counts"]["control"]["PHASE2_5_6_ONLY"] == 1
	assert paired["counts"]["overall"]["V1_1_3_ONLY"] == 1


def test_decision_rule_detects_control_regression() -> None:
	paired = {
		"counts": {
			"strong": {"BOTH_CORRECT": 0, "V1_1_3_ONLY": 0, "PHASE2_5_6_ONLY": 1, "BOTH_WRONG": 0},
			"control": {"BOTH_CORRECT": 0, "V1_1_3_ONLY": 1, "PHASE2_5_6_ONLY": 3, "BOTH_WRONG": 0},
			"overall": {"BOTH_CORRECT": 0, "V1_1_3_ONLY": 1, "PHASE2_5_6_ONLY": 4, "BOTH_WRONG": 0},
		}
	}
	assert phase3b.decision_label([], paired) == "CONTROL_REGRESSION_CONCERN"


def test_decision_rule_marks_runtime_errors_inconclusive() -> None:
	paired = {
		"counts": {
			"strong": {"BOTH_CORRECT": 0, "V1_1_3_ONLY": 0, "PHASE2_5_6_ONLY": 0, "BOTH_WRONG": 0},
			"control": {"BOTH_CORRECT": 0, "V1_1_3_ONLY": 0, "PHASE2_5_6_ONLY": 0, "BOTH_WRONG": 0},
			"overall": {"BOTH_CORRECT": 0, "V1_1_3_ONLY": 0, "PHASE2_5_6_ONLY": 0, "BOTH_WRONG": 0},
		}
	}
	assert phase3b.decision_label([{"error": "connection"}], paired) == "INCONCLUSIVE"


def test_repeat_stability_counts_are_reported_per_method() -> None:
	rows = [
		_row("case-a", "strong", "v1_1_3", 1, 1.0, repeat=1),
		_row("case-a", "strong", "v1_1_3", 1, 1.0, repeat=2),
		_row("case-a", "strong", "phase2_5_6", 1, 1.0, repeat=1),
		_row("case-a", "strong", "phase2_5_6", 0, 0.0, repeat=2),
	]
	stability = phase3b.stability_analysis(rows)
	assert stability["v1_1_3"]["answer_stable_cases"] == 1
	assert stability["phase2_5_6"]["answer_stable_cases"] == 0
