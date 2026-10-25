import json
import unittest

from chain_of_relations.methods.phase2_5_3a.router import SourceTargetExtractionRouter
from chain_of_relations.schema import Entity


def make_router(payload):
	calls = []

	def generate(*_args, **_kwargs):
		calls.append(1)
		return json.dumps(payload), {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}

	router = SourceTargetExtractionRouter(generate)
	router._test_calls = calls
	return router


def make_sequence_router(responses):
	calls = []

	def generate(*_args, **_kwargs):
		index = len(calls)
		calls.append(1)
		response = responses[min(index, len(responses) - 1)]
		return response, {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}

	router = SourceTargetExtractionRouter(generate)
	router._test_calls = calls
	return router


def constraint(cid, anchor, role="ANSWER", prop="occupation", depends=None, answer_relevant=True):
	return {
		"constraint_id": cid,
		"source_anchor": anchor,
		"constraint_text": f"constraint {cid}",
		"target_variable_role": role,
		"answer_relevant": answer_relevant,
		"semantic_property": prop,
		"depends_on": depends or [],
	}


class TestPhase253AExtraction(unittest.TestCase):
	def test_multiple_independent_source_groups_create_final_branch(self):
		router = make_router({
			"constraints": [constraint("c1", "m.a"), constraint("c2", "m.b", prop="record label")],
			"independent_source_groups": [
				{"source_anchor": "m.a", "constraint_ids": ["c1"]},
				{"source_anchor": "m.b", "constraint_ids": ["c2"]},
			],
			"shared_target": {"type": "ANSWER", "description": "same answer"},
			"dependency_pattern": "INDEPENDENT_CONVERGENCE",
			"structure": "LINEAR",
			"confidence": 0.99,
			"requested_final_answer_type": "occupation",
			"requested_final_semantic_property": "occupation",
			"shared_answer_type": "occupation",
			"branches": [],
		})
		decision = router.route("what is shared by A and B?", [Entity("m.a", "A"), Entity("m.b", "B")], [])
		self.assertEqual(decision.structure, "BRANCH_FINAL")
		self.assertEqual(decision.execution_mode, "cor")
		self.assertFalse(decision.validation_passed)

	def test_shared_intermediate_is_preserved(self):
		router = make_router({
			"constraints": [
				constraint("c1", "m.a", role="INTERMEDIATE", prop="author"),
				constraint("c2", "m.b", role="INTERMEDIATE", prop="party"),
			],
			"independent_source_groups": [
				{"source_anchor": "m.a", "constraint_ids": ["c1"]},
				{"source_anchor": "m.b", "constraint_ids": ["c2"]},
			],
			"shared_target": {"type": "INTERMEDIATE", "description": "shared person"},
			"dependency_pattern": "MIXED",
			"structure": "LINEAR",
			"confidence": 0.99,
			"requested_final_answer_type": "person",
			"requested_final_semantic_property": "person",
			"shared_answer_type": "person",
			"branches": [],
		})
		decision = router.route("who is connected through A and B?", [Entity("m.a", "A"), Entity("m.b", "B")], [])
		self.assertEqual(decision.structure, "BRANCH_INTERMEDIATE")
		self.assertEqual(decision.execution_mode, "cor")

	def test_type_only_group_is_not_an_independent_answer_source(self):
		router = make_router({
			"constraints": [
				constraint("c1", "male", role="TYPE", prop="gender", answer_relevant=False),
				constraint("c2", "female", role="TYPE", prop="gender", answer_relevant=False),
			],
			"independent_source_groups": [
				{"source_anchor": "male", "constraint_ids": ["c1"]},
				{"source_anchor": "female", "constraint_ids": ["c2"]},
			],
			"shared_target": {"type": "ANSWER", "description": "same answer"},
			"dependency_pattern": "INDEPENDENT_CONVERGENCE",
			"structure": "BRANCH_FINAL",
			"confidence": 0.99,
			"requested_final_answer_type": "person",
			"requested_final_semantic_property": "person",
			"shared_answer_type": "person",
			"branches": [],
		})
		decision = router.route("which people?", [Entity("m.a", "male"), Entity("m.b", "female")], [])
		self.assertNotEqual(decision.execution_mode, "branch")

	def test_self_check_and_retry_guard_are_logged_without_second_call(self):
		router = make_router({
			"constraints": [constraint("c1", "m.a")],
			"independent_source_groups": [{"source_anchor": "m.a", "constraint_ids": ["c1"]}],
			"shared_target": {"type": "NONE", "description": ""},
			"dependency_pattern": "SEQUENTIAL",
			"structure": "LINEAR",
			"confidence": 0.99,
			"requested_final_answer_type": "person",
			"requested_final_semantic_property": "person",
			"shared_answer_type": "person",
			"extraction_self_check": {
				"all_answer_relevant_constraints_present": True,
				"all_independent_sources_present": True,
				"shared_target_verified": True,
				"correction_needed": False,
				"correction_note": "complete",
			},
			"branches": [],
		})
		history = []
		router.route("which person?", [Entity("m.a", "A")], history)
		self.assertEqual(router._test_calls, [1])
		self.assertFalse(history[-1]["retry_used"])
		self.assertFalse(history[-1]["extraction_completeness"]["suspicious"])

	def test_empty_extraction_uses_exactly_one_bounded_retry(self):
		payload = {
			"constraints": [constraint("c1", "m.a")],
			"independent_source_groups": [{"source_anchor": "m.a", "constraint_ids": ["c1"]}],
			"shared_target": {"type": "NONE", "description": ""},
			"dependency_pattern": "SEQUENTIAL",
			"structure": "LINEAR",
			"confidence": 0.99,
			"requested_final_answer_type": "person",
			"requested_final_semantic_property": "person",
			"shared_answer_type": "person",
			"branches": [],
		}
		router = make_sequence_router([None, json.dumps(payload)])
		history = []
		decision = router.route("which person?", [Entity("m.a", "A")], history)
		self.assertEqual(decision.structure, "LINEAR")
		self.assertEqual(router._test_calls, [1, 1])
		self.assertTrue(history[-1]["retry_used"])
		self.assertEqual(history[-1]["retry_reason"], "empty_model_response")

	def test_structure_mapping_and_validator_contract_are_unchanged(self):
		router = make_router({
			"constraints": [constraint("c1", "m.a"), constraint("c2", "m.b")],
			"independent_source_groups": [
				{"source_anchor": "m.a", "constraint_ids": ["c1"]},
				{"source_anchor": "m.b", "constraint_ids": ["c2"]},
			],
			"shared_target": {"type": "ANSWER", "description": "same occupation"},
			"dependency_pattern": "INDEPENDENT_CONVERGENCE",
			"structure": "LINEAR",
			"confidence": 0.99,
			"requested_final_answer_type": "occupation",
			"requested_final_semantic_property": "occupation",
			"shared_answer_type": "occupation",
			"branches": [{"branch_id": "a"}, {"branch_id": "b"}],
		})
		decision = router.route("what occupation?", [Entity("m.a", "A"), Entity("m.b", "B")], [])
		self.assertEqual(decision.structure, "BRANCH_FINAL")
		self.assertFalse(decision.validation_passed)
		self.assertTrue(any("constraint_property" in value for value in decision.validation_failures))

	def test_completeness_check_accepts_string_shared_target(self):
		router = make_router({
			"constraints": [constraint("c1", "m.a")],
			"independent_source_groups": [{"source_anchor": "m.a", "constraint_ids": ["c1"]}],
			"shared_target": "NONE",
			"dependency_pattern": "SEQUENTIAL",
			"structure": "LINEAR",
			"confidence": 0.99,
			"requested_final_answer_type": "person",
			"requested_final_semantic_property": "person",
			"shared_answer_type": "person",
			"branches": [],
		})
		history = []
		decision = router.route("which person?", [Entity("m.a", "A")], history)
		self.assertEqual(decision.structure, "LINEAR")
		self.assertFalse(history[-1]["extraction_completeness"]["shared_target_verified"])


if __name__ == "__main__":
	unittest.main()
