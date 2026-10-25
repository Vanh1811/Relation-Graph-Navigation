import unittest

from scripts.analyze_phase2_5_4_semantic_validity import (
	branch_count_audit,
	decision_for,
	executability,
	primary_classification,
	property_audit,
	shared_target_audit,
	target_type_audit,
	type_only_audit,
)


def ref(gold="GOLD_BRANCH_FINAL", answer=True, intermediate=False, sources=2, answer_type="occupation"):
	return {
		"gold_structure": gold,
		"shared_answer_gold": answer,
		"shared_intermediate_gold": intermediate,
		"independent_sources_gold": sources,
		"requested_final_answer_type": answer_type,
		"gold_source_anchors": [{"id": "m.a"}, {"id": "m.b"}][:sources],
	}


def row(**updates):
	value = {
		"serialization_status": "COMPLETE",
		"serialization_errors": [],
		"semantic_structure": "BRANCH_FINAL",
		"validator_result": "VALIDATOR_REJECT",
		"validator_reason": ["shared_target has no description"],
		"serialized_branches": [
			{"root_entity_id": "m.a", "constraint_property": "occupation", "constraint_summary": "A", "target_answer_type": "occupation"},
			{"root_entity_id": "m.b", "constraint_property": "occupation", "constraint_summary": "B", "target_answer_type": "occupation"},
		],
	}
	value.update(updates)
	return value


def proposal(**updates):
	value = {
		"shared_target": "ANSWER",
		"requested_final_answer_type": "occupation",
		"shared_answer_type": "occupation",
		"constraints": [
			{"source_anchor": "m.a", "target_variable_role": "ANSWER", "answer_relevant": True, "semantic_property": "occupation"},
			{"source_anchor": "m.b", "target_variable_role": "ANSWER", "answer_relevant": True, "semantic_property": "occupation"},
		],
		"independent_source_groups": [
			{"source_anchor": "m.a", "constraint_ids": ["c1"]},
			{"source_anchor": "m.b", "constraint_ids": ["c2"]},
		],
	}
	value.update(updates)
	return value


class TestPhase254Audit(unittest.TestCase):
	def test_primary_classification_is_single_and_prioritizes_incomplete_serialization(self):
		label, _ = primary_classification(ref(), row(serialization_status="FAILED"), proposal())
		self.assertEqual(label, "SERIALIZATION_INCOMPLETE")

	def test_complete_gold_aligned_plan_is_schema_incompatible(self):
		label, _ = primary_classification(ref(), row(), proposal())
		self.assertEqual(label, "VALID_BUT_SCHEMA_INCOMPATIBLE")

	def test_wrong_shared_target_is_upstream_extraction_error(self):
		label, reasons = primary_classification(
			ref(gold="GOLD_BRANCH_INTERMEDIATE", answer=False, intermediate=True),
			row(semantic_structure="BRANCH_FINAL"),
			proposal(shared_target="ANSWER"),
		)
		self.assertEqual(label, "UPSTREAM_EXTRACTION_WRONG")
		self.assertTrue(reasons)

	def test_shared_target_representation_is_distinguished_from_wrong_target(self):
		self.assertEqual(shared_target_audit(ref(), proposal()), "SHARED_TARGET_REPRESENTATION_MISMATCH")
		self.assertEqual(
			shared_target_audit(ref(gold="GOLD_BRANCH_INTERMEDIATE", answer=False, intermediate=True), proposal(shared_target="ANSWER")),
			"SHARED_TARGET_GENUINELY_WRONG",
		)

	def test_type_only_reason_can_be_false_when_constraints_are_answer_relevant(self):
		value = row(validator_reason=["branch plan cannot be supported by type-only constraints"])
		self.assertEqual(type_only_audit(value, proposal()), "FALSE_TYPE_ONLY_REJECTION")

	def test_complementary_properties_are_not_called_genuine_mismatch(self):
		value = row(validator_reason=["CONSTRAINT_PROPERTY_MISMATCH: incompatible"])
		p = proposal()
		p["constraints"][1]["semantic_property"] = "rank"
		self.assertEqual(property_audit(ref(), value, p), "COMPLEMENTARY_PROPERTY_VALID")

	def test_target_type_categories(self):
		self.assertEqual(target_type_audit(ref(answer_type="political ideology"), proposal(requested_final_answer_type="ideology")), "TYPE_NORMALIZATION_MISMATCH")
		self.assertEqual(target_type_audit(ref(), proposal(requested_final_answer_type="", shared_answer_type="")), "MISSING_TYPE_METADATA")
		self.assertEqual(target_type_audit(ref(), proposal(requested_final_answer_type="gender")), "GENUINE_TARGET_TYPE_ERROR")

	def test_branch_count_audit_detects_missing_source(self):
		value = row(validator_reason=["BRANCH_FINAL requires at least two branches"])
		p = proposal(independent_source_groups=[{"source_anchor": "m.a", "constraint_ids": ["c1"]}])
		self.assertEqual(branch_count_audit(ref(), value, p), "UPSTREAM_SOURCE_EXTRACTION_ERROR")

	def test_executability_requires_final_answer_branch_shape(self):
		self.assertEqual(executability("VALID_BUT_SCHEMA_INCOMPATIBLE", ref(), row()), "SAFE_TO_EXECUTE")
		self.assertEqual(
			executability("VALID_BUT_SCHEMA_INCOMPATIBLE", ref(gold="GOLD_BRANCH_INTERMEDIATE", answer=False, intermediate=True), row(semantic_structure="BRANCH_INTERMEDIATE")),
			"NOT_SAFE_TO_EXECUTE",
		)
		self.assertEqual(executability("SERIALIZATION_INCOMPLETE", ref(), row(serialization_status="FAILED")), "UNKNOWN")

	def test_intervention_gate_requires_over_rejection_cases(self):
		counts = {"VALID_BUT_SCHEMA_INCOMPATIBLE": 5}
		case_labels = {"a": {"VALID_BUT_SCHEMA_INCOMPATIBLE"}, "b": {"VALID_BUT_SCHEMA_INCOMPATIBLE"}, "c": {"VALID_BUT_SCHEMA_INCOMPATIBLE"}}
		decision = decision_for(counts, case_labels, {"INVALID_SHARED_TARGET": 5}, [])
		self.assertFalse(decision["validator_intervention_gate"])
		self.assertTrue(decision["schema_intervention_gate"])
		self.assertEqual(decision["main_decision"], "VALIDATOR_SCHEMA_MISMATCH_IS_MAJOR")


if __name__ == "__main__":
	unittest.main()
