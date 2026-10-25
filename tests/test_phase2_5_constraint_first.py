import json
import unittest

from chain_of_relations.methods.phase2_5.router import ConstraintFirstRouter
from chain_of_relations.schema import Entity


def make_router(payload):
	def generate(*_args, **_kwargs):
		return json.dumps(payload), {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}

	return ConstraintFirstRouter(generate)


def constraint(cid, source, role="ANSWER", prop="occupation", depends=None):
	return {
		"constraint_id": cid,
		"text": f"constraint {cid}",
		"source_entity": source,
		"target_role": role,
		"semantic_property": prop,
		"depends_on": depends or [],
	}


class TestConstraintFirstSchema(unittest.TestCase):
	def test_shared_answer_is_decided_after_extraction(self):
		decision = make_router({
			"constraints": [constraint("c1", "m.a"), constraint("c2", "m.b")],
			"independent_sources": ["A", "B"],
			"shared_target": {"type": "ANSWER", "description": "same occupation"},
			"dependency_pattern": "INDEPENDENT_CONVERGENCE",
			"structure": "LINEAR",
			"confidence": 0.99,
			"reason": "constraints first",
			"requested_final_answer_type": "occupation",
			"requested_final_semantic_property": "occupation",
			"shared_answer_type": "occupation",
			"branches": [
				{"branch_id": "a", "root_entity_id": "m.a", "root_entity_name": "A", "constraint_property": "occupation", "constraint_summary": "occupation of A", "target_answer_type": "occupation"},
				{"branch_id": "b", "root_entity_id": "m.b", "root_entity_name": "B", "constraint_property": "occupation", "constraint_summary": "occupation of B", "target_answer_type": "occupation"},
			],
		}) .route("what is the occupation of A and B?", [Entity("m.a", "A"), Entity("m.b", "B")], [])
		self.assertEqual(decision.structure, "BRANCH_FINAL")
		self.assertTrue(decision.validation_passed)
		self.assertEqual(decision.shared_target["type"], "ANSWER")
		self.assertEqual(decision.dependency_pattern, "INDEPENDENT_CONVERGENCE")

	def test_shared_intermediate_maps_to_logged_cor_route(self):
		decision = make_router({
			"constraints": [constraint("c1", "m.a", "INTERMEDIATE"), constraint("c2", "m.b", "INTERMEDIATE", depends=["c1"])],
			"independent_sources": ["A", "B"],
			"shared_target": {"type": "INTERMEDIATE", "description": "shared intermediate"},
			"dependency_pattern": "MIXED",
			"structure": "UNCERTAIN",
			"confidence": 0.99,
			"requested_final_answer_type": "person",
			"requested_final_semantic_property": "person via intermediate",
			"shared_answer_type": "person",
			"branches": [],
		}) .route("who is connected through A and B?", [Entity("m.a", "A"), Entity("m.b", "B")], [])
		self.assertEqual(decision.structure, "BRANCH_INTERMEDIATE")
		self.assertEqual(decision.execution_mode, "cor")
		self.assertTrue(decision.unsupported_structure)

	def test_type_only_constraints_do_not_create_final_branch(self):
		decision = make_router({
			"constraints": [constraint("c1", "m.a", "TYPE", "gender"), constraint("c2", "m.b", "TYPE", "gender")],
			"independent_sources": ["Male", "Female"],
			"shared_target": {"type": "ANSWER", "description": "same answer"},
			"dependency_pattern": "INDEPENDENT_CONVERGENCE",
			"structure": "BRANCH_FINAL",
			"confidence": 0.99,
			"requested_final_answer_type": "person",
			"requested_final_semantic_property": "person",
			"shared_answer_type": "person",
			"branches": [],
		}) .route("which people?", [Entity("m.a", "Male"), Entity("m.b", "Female")], [])
		self.assertNotEqual(decision.execution_mode, "branch")

	def test_property_conflict_does_not_short_circuit_explicit_shared_answer(self):
		payload = {
			"constraints": [constraint("c1", "m.a", prop="produced by"), constraint("c2", "m.b", prop="record label")],
			"independent_sources": ["Pictures of Lily", "Track Record"],
			"shared_target": {"type": "ANSWER", "description": "same musical release"},
			"dependency_pattern": "INDEPENDENT_CONVERGENCE",
			"structure": "UNCERTAIN",
			"confidence": 0.99,
			"requested_final_answer_type": "musical release",
			"requested_final_semantic_property": "musical release",
			"shared_answer_type": "musical release",
			"branches": [
				{"branch_id": "a", "root_entity_id": "m.a", "root_entity_name": "Pictures of Lily", "constraint_property": "produced by", "constraint_summary": "release produced by A", "target_answer_type": "musical release"},
				{"branch_id": "b", "root_entity_id": "m.b", "root_entity_name": "Track Record", "constraint_property": "record label", "constraint_summary": "release labelled B", "target_answer_type": "musical release"},
			],
		}
		decision = make_router(payload).route("what musical release is produced by A and labelled B?", [Entity("m.a", "Pictures of Lily"), Entity("m.b", "Track Record")], [])
		self.assertEqual(decision.structure, "BRANCH_FINAL")
		self.assertTrue(decision.validation_passed)

class TestConstraintFirstLogging(unittest.TestCase):
	def test_logs_extraction_fields(self):
		payload = {
			"constraints": [constraint("c1", "m.a")],
			"independent_sources": ["A"],
			"shared_target": {"type": "NONE", "description": ""},
			"dependency_pattern": "SEQUENTIAL",
			"structure": "LINEAR",
			"confidence": 0.99,
			"requested_final_answer_type": "person",
			"requested_final_semantic_property": "person",
			"shared_answer_type": "person",
			"branches": [],
		}
		history = []
		decision = make_router(payload).route("Which person?", [Entity("m.a", "A")], history)
		self.assertTrue(decision.as_dict()["constraints"])
		self.assertEqual(history[-1]["type"], "phase2_5_constraint_first_router")
		self.assertIn("dependency_pattern", history[-1])


if __name__ == "__main__":
	unittest.main()
