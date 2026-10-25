import json
import unittest

from chain_of_relations.methods.phase2_5_3b.router import BranchPlanSerializationRouter
from chain_of_relations.methods.phase2_5_3b.serializer import serialize_branch_plan
from chain_of_relations.schema import Entity


def constraint(cid, anchor, prop="occupation", text=None):
	return {
		"constraint_id": cid,
		"source_anchor": anchor,
		"constraint_text": text or f"constraint {cid}",
		"target_variable_role": "ANSWER",
		"answer_relevant": True,
		"semantic_property": prop,
		"depends_on": [],
	}


def payload():
	return {
		"constraints": [constraint("c1", "m.a"), constraint("c2", "m.b")],
		"independent_source_groups": [
			{"source_anchor": "m.a", "constraint_ids": ["c1"]},
			{"source_anchor": "m.b", "constraint_ids": ["c2"]},
		],
		"shared_target": {"type": "ANSWER", "description": "same occupation"},
		"dependency_pattern": "INDEPENDENT_CONVERGENCE",
		"structure": "BRANCH_FINAL",
		"confidence": 0.99,
		"requested_final_answer_type": "occupation",
		"requested_final_semantic_property": "occupation",
		"shared_answer_type": "occupation",
		"branches": [],
	}


class TestPhase253BSerialization(unittest.TestCase):
	def test_maps_semantic_constraints_to_exact_executable_fields(self):
		proposal = payload()
		constraints = [
			BranchPlanSerializationRouter._normalise_constraint(item, index)
			for index, item in enumerate(proposal["constraints"], start=1)
		]
		groups = BranchPlanSerializationRouter._normalise_source_groups(proposal, constraints)
		result = serialize_branch_plan(proposal, constraints, groups, [Entity("m.a", "A"), Entity("m.b", "B")], "what occupation?")
		self.assertEqual(result["serialization_status"], "COMPLETE")
		self.assertEqual(result["serialization_errors"], [])
		self.assertEqual(result["serialized_branches"], [
			{
				"branch_id": "branch_1",
				"root_entity_id": "m.a",
				"root_entity_name": "A",
				"constraint_property": "occupation",
				"constraint_summary": "constraint c1",
				"target_answer_type": "occupation",
			},
			{
				"branch_id": "branch_2",
				"root_entity_id": "m.b",
				"root_entity_name": "B",
				"constraint_property": "occupation",
				"constraint_summary": "constraint c2",
				"target_answer_type": "occupation",
			},
		])

	def test_root_resolution_uses_existing_entities_only(self):
		proposal = payload()
		constraints = [
			BranchPlanSerializationRouter._normalise_constraint(item, index)
			for index, item in enumerate(proposal["constraints"], start=1)
		]
		groups = BranchPlanSerializationRouter._normalise_source_groups(proposal, constraints)
		result = serialize_branch_plan(proposal, constraints, groups, [Entity("m.a", "A")], "what occupation?")
		self.assertIn("ROOT_ENTITY_UNRESOLVED", result["serialization_errors"])
		self.assertEqual(result["serialization_status"], "PARTIAL")

	def test_missing_property_is_not_fabricated(self):
		proposal = payload()
		proposal["constraints"][0]["semantic_property"] = ""
		constraints = [
			BranchPlanSerializationRouter._normalise_constraint(item, index)
			for index, item in enumerate(proposal["constraints"], start=1)
		]
		groups = BranchPlanSerializationRouter._normalise_source_groups(proposal, constraints)
		result = serialize_branch_plan(proposal, constraints, groups, [Entity("m.a", "A"), Entity("m.b", "B")], "what occupation?")
		self.assertIn("MISSING_CONSTRAINT_PROPERTY", result["serialization_errors"])
		self.assertEqual(len(result["serialized_branches"]), 1)

	def test_branch_ids_are_deterministic_when_group_order_changes(self):
		proposal = payload()
		constraints = [
			BranchPlanSerializationRouter._normalise_constraint(item, index)
			for index, item in enumerate(proposal["constraints"], start=1)
		]
		groups = BranchPlanSerializationRouter._normalise_source_groups(proposal, constraints)
		reversed_groups = list(reversed(groups))
		entities = [Entity("m.a", "A"), Entity("m.b", "B")]
		first = serialize_branch_plan(proposal, constraints, groups, entities, "what occupation?")
		second = serialize_branch_plan(proposal, constraints, reversed_groups, entities, "what occupation?")
		self.assertEqual(first["serialized_branches"], second["serialized_branches"])

	def test_router_calls_model_once_and_validator_receives_serialized_branches(self):
		calls = []

		def generate(*_args, **_kwargs):
			calls.append(1)
			return json.dumps(payload()), {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}

		router = BranchPlanSerializationRouter(generate)
		history = []
		decision = router.route("what occupation?", [Entity("m.a", "A"), Entity("m.b", "B")], history)
		self.assertEqual(calls, [1])
		self.assertEqual(history[-1]["serialization"]["serialization_status"], "COMPLETE")
		self.assertEqual(decision.proposed_branches[0]["branch_id"], "branch_1")
		self.assertEqual(decision.proposed_branches[0]["root_entity_id"], "m.a")

	def test_non_branch_routes_do_not_create_executable_branches(self):
		proposal = payload()
		proposal["structure"] = "LINEAR"
		constraints = [
			BranchPlanSerializationRouter._normalise_constraint(item, index)
			for index, item in enumerate(proposal["constraints"], start=1)
		]
		groups = BranchPlanSerializationRouter._normalise_source_groups(proposal, constraints)
		result = serialize_branch_plan(proposal, constraints, groups, [Entity("m.a", "A"), Entity("m.b", "B")], "what occupation?")
		self.assertEqual(result["serialization_scope"], "NON_EXECUTABLE_STRUCTURE")
		self.assertEqual(result["serialized_branches"], [])


if __name__ == "__main__":
	unittest.main()
