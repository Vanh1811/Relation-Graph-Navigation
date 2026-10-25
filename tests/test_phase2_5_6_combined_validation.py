import json
import unittest

from scripts.run_phase2_5_6_combined_validation import (
	PHASE_STACK,
	combined_gate,
	primary_failure,
	upstream_semantic_status,
)


class Phase256CombinedValidationTests(unittest.TestCase):
	def ref(self, **overrides):
		value = {
			"case_id": "case",
			"gold_structure": "GOLD_BRANCH_FINAL",
			"shared_answer_gold": True,
			"shared_intermediate_gold": False,
			"independent_sources_gold": 2,
		}
		value.update(overrides)
		return value

	def row(self, proposal, **overrides):
		value = {
			"semantic_structure": proposal.get("structure", "BRANCH_FINAL"),
			"serialization_status": "COMPLETE",
			"validator_result": "VALIDATOR_PASS",
			"branch_routed": True,
			"execution_status": "EXECUTION_SUCCESS",
			"hits_at_1": 1,
			"sparql_query_count": 1,
			"parsed_result": proposal,
			"prompt_history": [{
				"type": "phase2_5_5_canonical_normalization_router",
				"parsed_result": proposal,
			}],
		}
		value.update(overrides)
		return value

	def proposal(self, **overrides):
		value = {
			"structure": "BRANCH_FINAL",
			"shared_target": "ANSWER",
			"dependency_pattern": "INDEPENDENT_CONVERGENCE",
			"independent_source_groups": [{"source_anchor": "m.a"}, {"source_anchor": "literal: 2"}],
			"constraints": [{"answer_relevant": True, "target_variable_role": "final_answer"}],
		}
		value.update(overrides)
		return value

	def test_stack_order_is_frozen_and_explicit(self):
		self.assertEqual(PHASE_STACK[0], "phase2_5_3a_extraction")
		self.assertEqual(PHASE_STACK[1], "phase2_5_5_canonical_normalization")
		self.assertEqual(PHASE_STACK[2], "phase2_5_3b_serialization")
		self.assertEqual(PHASE_STACK[3:], ("unchanged_validator", "unchanged_branch_execution_and_fallback"))

	def test_semantic_alignment_does_not_label_schema_fields_as_semantic_errors(self):
		proposal = self.proposal()
		status, reasons = upstream_semantic_status(self.ref(), self.row(proposal))
		self.assertEqual(status, "SEMANTICALLY_ALIGNED")
		self.assertEqual(reasons, [])

	def test_shared_target_error_is_upstream(self):
		status, _ = upstream_semantic_status(self.ref(), self.row(self.proposal(shared_target="NONE")))
		self.assertEqual(status, "UPSTREAM_SHARED_TARGET_ERROR")

	def test_failure_attribution_preserves_pipeline_order(self):
		row = self.row(self.proposal(), serialization_status="FAILED", validator_result="VALIDATOR_REJECT", hits_at_1=0)
		self.assertEqual(primary_failure(self.ref(), row, "SEMANTICALLY_ALIGNED"), "SERIALIZATION")
		row["serialization_status"] = "COMPLETE"
		self.assertEqual(primary_failure(self.ref(), row, "SEMANTICALLY_ALIGNED"), "VALIDATOR")

	def test_gate_requires_all_thresholds(self):
		ref_rows = [
			{"case_id": "control", "routing_outcome": "CORRECT_ROUTING_CONTROL", "gold_structure": "GOLD_BRANCH_FINAL"},
		]
		rows = [{
			"case_id": "control", "semantic_structure": "BRANCH_FINAL", "validator_result": "VALIDATOR_PASS",
			"fallback_used": False, "branch_self_sufficient": True, "hits_at_1": 1,
		}]
		# The real gate is defined for 15 cases; this assertion checks it does not
		# silently pass a partial evaluation.
		gate = combined_gate(ref_rows, rows)
		self.assertFalse(gate["pass"])


if __name__ == "__main__":
	unittest.main()
