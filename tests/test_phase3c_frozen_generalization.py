import json
from pathlib import Path

import scripts.run_phase3c_frozen_generalization as phase3c


def test_manifest_has_frozen_strata_and_no_duplicates():
	manifest = json.loads(phase3c.MANIFEST_PATH.read_text(encoding="utf-8"))
	assert manifest["status"] == "FROZEN"
	assert manifest["strong_case_count"] == 60
	assert manifest["control_case_count"] == 30
	assert manifest["strong_stratum_counts"] == {
		"cwq|BRANCH_FINAL": 15,
		"cwq|BRANCH_INTERMEDIATE": 15,
		"grailqa|BRANCH_FINAL": 15,
		"grailqa|BRANCH_INTERMEDIATE": 15,
	}
	assert manifest["control_stratum_counts"] == {
		"cwq|BRANCH_INTERMEDIATE": 10,
		"grailqa|BRANCH_FINAL": 10,
		"grailqa|BRANCH_INTERMEDIATE": 10,
	}
	cases = manifest["strong_cases"] + manifest["control_cases"]
	assert len({case["case_id"] for case in cases}) == 90
	assert not ({case["case_id"] for case in manifest["strong_cases"]} & {case["case_id"] for case in manifest["control_cases"]})


def test_manifest_necessity_classes_are_separated():
	manifest = json.loads(phase3c.MANIFEST_PATH.read_text(encoding="utf-8"))
	assert {case["necessity"] for case in manifest["strong_cases"]} == {"STRONGLY_BRANCH_REQUIRED"}
	assert {case["necessity"] for case in manifest["control_cases"]} == {"ANSWER_EQUIVALENT_LINEARIZATION"}


def _row(method, case_id, cohort="STRONG", correct=0, f1=0.0, **extra):
	row = {
		"method": method,
		"case_id": case_id,
		"cohort": cohort,
		"dataset": "cwq",
		"gold_structure": "BRANCH_INTERMEDIATE",
		"hits_at_1": correct,
		"f1": f1,
		"final_prediction": ["answer"] if correct else [],
		"gold_answers": ["answer"],
	}
	row.update(extra)
	return row


def test_paired_analysis_reports_method_only_wins():
	rows = [
		_row("v1_1_3", "case-a", correct=0),
		_row("v1_1_3", "case-a", correct=0),
		_row("phase2_5_6", "case-a", correct=1, f1=1.0),
		_row("phase2_5_6", "case-a", correct=1, f1=1.0),
	]
	paired = phase3c.paired_analysis(rows)
	assert paired["by_cohort"]["STRONG"]["counts"]["PHASE2_5_6_ONLY"] == 1
	assert paired["by_cohort"]["STRONG"]["paired_net"] == 1


def test_branch_advantage_requires_native_branch_success():
	rows = [
		_row("v1_1_3", "case-a", correct=0, branch_self_sufficient=False),
		_row("v1_1_3", "case-a", correct=0, branch_self_sufficient=False),
		_row("phase2_5_6", "case-a", correct=1, f1=1.0, branch_routed=True, branch_self_sufficient=True),
		_row("phase2_5_6", "case-a", correct=1, f1=1.0, branch_routed=True, branch_self_sufficient=True),
	]
	paired = phase3c.paired_analysis(rows)
	audit = phase3c.branch_advantage_audit(rows, paired)
	assert audit[0]["classification"] == "TRUE_BRANCH_ADVANTAGE"


def test_failure_family_identifies_shared_target_after_pipeline_checks():
	row = _row(
		"phase2_5_6",
		"case-a",
		correct=0,
		normalization_status="NORMALIZED_SUCCESSFULLY",
		serialization_status="COMPLETE",
		validator_result="VALIDATOR_PASS",
		fallback_used=False,
		branch_routed=True,
		gold_structure="BRANCH_INTERMEDIATE",
		canonical_semantic_representation={"shared_target": {"type": "ANSWER"}},
	)
	assert phase3c.failure_family(row) == "SHARED_TARGET"


def test_decision_rule_supports_branch_advantage_when_all_gates_pass():
	metrics = {
		"ALL_STRONG": {
			"v1_1_3": {"hits_at_1_mean": 0.40},
			"phase2_5_6": {"hits_at_1_mean": 0.50},
		}
	}
	paired = {
		"by_cohort": {
			"STRONG": {"paired_net": 3},
			"CONTROL": {"paired_net": 0},
		}
	}
	advantage = [
		{"classification": "TRUE_BRANCH_ADVANTAGE", "outcome": "PHASE2_5_6_ONLY"},
		{"classification": "TRUE_BRANCH_ADVANTAGE", "outcome": "PHASE2_5_6_ONLY"},
	]
	label, details = phase3c.choose_decision(metrics, paired, advantage, [], complete=True)
	assert label == "BRANCH_ADVANTAGE_SUPPORTED"
	assert all(details["gates"].values())
