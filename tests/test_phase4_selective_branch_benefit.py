import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/experiments/phase4/selective_branch_benefit_analysis"


def test_phase4_manifest_has_required_frozen_cohorts():
	manifest = json.loads((OUTPUT / "phase4_analysis_manifest.json").read_text(encoding="utf-8"))
	assert manifest["status"] == "FROZEN_ANALYSIS"
	assert manifest["strong_case_count"] == 60
	assert manifest["cohort_counts"] == {
		"TRUE_BRANCH_WIN": 4,
		"V1_1_3_ONLY_LOSS": 7,
		"BOTH_CORRECT": 25,
		"BOTH_WRONG": 24,
	}
	assert len({case["case_id"] for case in manifest["cases"]}) == 60


def test_phase4_context_samples_are_deterministic():
	manifest = json.loads((OUTPUT / "phase4_analysis_manifest.json").read_text(encoding="utf-8"))
	ids = {outcome: [case["case_id"] for case in manifest["cases"] if case["paired_outcome"] == outcome] for outcome in ("BOTH_CORRECT", "BOTH_WRONG")}
	rng = random.Random(20261029)
	for outcome in ids:
		expected = sorted(rng.sample(ids[outcome], 5))
		assert manifest["context_review_ids"][outcome] == expected


def test_phase4_outputs_cover_all_required_artifacts():
	required = {
		"PHASE4_REPORT.md",
		"phase4_analysis_manifest.json",
		"benefit_feature_matrix.jsonl",
		"true_branch_win_audit.jsonl",
		"v1_1_3_only_loss_audit.jsonl",
		"both_correct_context.jsonl",
		"both_wrong_context.jsonl",
		"branch_final_vs_intermediate.json",
		"constraint_loss_analysis.jsonl",
		"fallback_analysis.json",
		"native_branch_advantage_analysis.json",
		"feature_prevalence_by_outcome.json",
		"candidate_selective_rules.json",
		"control_rule_trigger_analysis.json",
		"cost_aware_benefit_analysis.json",
		"decision_summary.md",
	}
	assert required <= {path.name for path in OUTPUT.iterdir()}
	assert sum(1 for line in (OUTPUT / "PHASE4_REPORT.md").read_text(encoding="utf-8").splitlines() if line.startswith("## ")) == 24


def test_phase4_rules_are_descriptive_not_deployed():
	rules = json.loads((OUTPUT / "candidate_selective_rules.json").read_text(encoding="utf-8"))
	assert len(rules["rules"]) <= 3
	assert rules["validated"] is False
	assert all(rule["true_wins_covered"] <= 2 for rule in rules["rules"])


def test_phase4_true_wins_are_native_and_losses_are_audited():
	wins = [json.loads(line) for line in (OUTPUT / "true_branch_win_audit.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
	losses = [json.loads(line) for line in (OUTPUT / "v1_1_3_only_loss_audit.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
	assert len(wins) == 4
	assert len(losses) == 7
	assert Counter(item["native_classification"] for item in wins) == Counter({"NATIVE_BRANCH_ADVANTAGE": 4})
	assert all(item["manual_review"]["status"] == "TRACE_REVIEWED" for item in wins + losses)
