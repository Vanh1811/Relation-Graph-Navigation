import unittest

from scripts.analyze_phase2_6_fresh_failure_semantics import (
	branch_intermediate_audit,
	dependency_audit,
	first_divergence,
	raw_target_type,
	runtime_decomposition,
	shared_target_audit,
)


class Phase26SemanticAnalysisTests(unittest.TestCase):
	def test_raw_target_type_accepts_string_and_object_forms(self):
		self.assertEqual(raw_target_type({"shared_target": "INTERMEDIATE"}), "INTERMEDIATE")
		self.assertEqual(raw_target_type({"shared_target": {"type": "ANSWER"}}), "ANSWER")

	def test_shared_target_audit_detects_final_intermediate_confusion(self):
		gold = {"shared_target": {"type": "INTERMEDIATE"}}
		runtime = runtime_decomposition({"raw_extraction": {"shared_target": "ANSWER"}})
		self.assertEqual(shared_target_audit(gold, runtime), "SHARED_TARGET_WRONG_FINAL_VS_INTERMEDIATE")

	def test_dependency_audit_detects_false_sequentialization(self):
		gold = {"dependency_pattern": "INDEPENDENT_CONVERGENCE"}
		runtime = runtime_decomposition({"raw_extraction": {"dependency_pattern": "SEQUENTIAL_CHAIN"}})
		self.assertEqual(dependency_audit(gold, runtime), "FALSE_SEQUENTIALIZATION")

	def test_branch_intermediate_is_separate_from_execution_support(self):
		gold = {"gold_structure": "BRANCH_INTERMEDIATE", "shared_target": {"type": "INTERMEDIATE"}}
		runtime = runtime_decomposition({
		"structure": "BRANCH_INTERMEDIATE",
		"raw_extraction": {"shared_target": "INTERMEDIATE"},
		"validator_result": "VALIDATOR_PASS",
		"execution_status": "NOT_APPLICABLE_NON_EXECUTABLE_STRUCTURE",
	})
		self.assertEqual(branch_intermediate_audit(gold, runtime), "DETECTED_BUT_EXECUTION_UNSUPPORTED")

	def test_first_divergence_prioritizes_source_loss_before_validator(self):
		gold = {
			"gold_structure": "BRANCH_FINAL",
			"shared_target": {"type": "ANSWER"},
			"independent_source_count": 2,
			"required_relation_sequence": ["p1"],
			"dependency_pattern": "INDEPENDENT_CONVERGENCE",
		}
		runtime = runtime_decomposition({
			"structure": "LINEAR",
			"raw_extraction": {
				"constraints": [{"constraint_id": "c1"}],
				"independent_source_groups": [{"source_anchor": "a"}],
				"shared_target": "ANSWER",
				"dependency_pattern": "SEQUENTIAL",
			},
			"validator_result": "VALIDATOR_REJECT",
		})
		self.assertEqual(first_divergence(gold, runtime, case_failed=True), "INDEPENDENT_SOURCE_EXTRACTION")


if __name__ == "__main__":
	unittest.main()
