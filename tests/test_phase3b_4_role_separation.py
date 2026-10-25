import json
import unittest

from chain_of_relations.methods.phase3b_4.router import (
	RoleSeparationRouter,
	derive_shared_target_type,
	role_consistency,
	separate_roles,
)
from chain_of_relations.schema import Entity


def payload(same=False, requires=True, target="ANSWER"):
	return {
		"constraints": [
			{
				"constraint_id": "c1",
				"source_anchor": "m.a",
				"constraint_text": "constraint A",
				"target_variable_role": "INTERMEDIATE",
				"answer_relevant": True,
				"semantic_property": "film relation",
				"depends_on": [],
			},
			{
				"constraint_id": "c2",
				"source_anchor": "m.b",
				"constraint_text": "constraint B",
				"target_variable_role": "ANSWER",
				"answer_relevant": True,
				"semantic_property": "actor relation",
				"depends_on": [],
			},
		],
		"independent_source_groups": [
			{"source_anchor": "m.a", "constraint_ids": ["c1"]},
			{"source_anchor": "m.b", "constraint_ids": ["c2"]},
		],
		"requested_answer_role": {"description": "the actor", "entity_type": "person"},
		"convergence_role": {"description": "the performance", "entity_type": "performance"},
		"roles_same_entity": same,
		"role_comparison": "YES" if same else "NO",
		"role_comparison_reason": "role comparison",
		"requires_downstream_relation": requires,
		"downstream_relation_description": "performance to actor" if requires else "",
		"shared_target": {"type": target, "description": "model supplied target"},
		"shared_target_description": "model supplied target",
		"requested_answer_description": "the requested actor",
		"shared_target_self_check": {
			"convergence_variable_description": "the performance",
			"resulting_variable_is_requested_answer": same,
			"valid_shared_latent_target": not same,
			"downstream_step_required": requires,
			"rationale": "role comparison",
		},
		"dependency_pattern": "MIXED" if requires else "INDEPENDENT_CONVERGENCE",
		"structure": "BRANCH_INTERMEDIATE" if requires else "BRANCH_FINAL",
		"execution_mode": "branch" if not requires else "cor",
		"confidence": 0.99,
		"reason": "two constraints",
		"requested_final_answer_type": "person",
		"requested_final_semantic_property": "actor",
		"shared_answer_type": "person",
		"branches": [] if requires else [
			{"branch_id": "b1", "root_entity_id": "m.a", "root_entity_name": "A", "constraint_property": "film", "constraint_summary": "A", "target_answer_type": "person"},
			{"branch_id": "b2", "root_entity_id": "m.b", "root_entity_name": "B", "constraint_property": "film", "constraint_summary": "B", "target_answer_type": "person"},
		],
		"unsupported_structure": False,
	}


class TestPhase3B4RoleSeparation(unittest.TestCase):
	def test_requested_and_convergence_roles_are_extracted(self):
		enriched, metadata = separate_roles(payload(False, True))
		self.assertEqual(enriched["requested_answer_role"]["entity_type"], "person")
		self.assertEqual(enriched["convergence_role"]["entity_type"], "performance")
		self.assertFalse(metadata["roles_same_entity"])

	def test_deterministic_target_derivation(self):
		self.assertEqual(derive_shared_target_type(True, False), "ANSWER")
		self.assertEqual(derive_shared_target_type(False, True), "INTERMEDIATE")
		self.assertEqual(derive_shared_target_type(False, False), "NONE")
		self.assertEqual(derive_shared_target_type(None, True), "NONE")

	def test_model_target_is_not_allowed_to_override_role_derivation(self):
		enriched, metadata = separate_roles(payload(False, True, target="ANSWER"))
		self.assertEqual(metadata["input_shared_target_type"], "ANSWER")
		self.assertEqual(enriched["shared_target"]["type"], "INTERMEDIATE")

	def test_inconsistent_same_role_and_downstream_is_conservative(self):
		enriched, metadata = separate_roles(payload(True, True))
		self.assertEqual(enriched["shared_target"]["type"], "NONE")
		self.assertEqual(metadata["role_separation_consistency"], "INCONSISTENT")
		self.assertIn("same_roles_cannot_require_downstream_relation", metadata["role_separation_issues"])

	def test_role_consistency_requires_explicit_role_fields(self):
		status, issues = role_consistency({"description": "", "entity_type": ""}, {"description": "", "entity_type": ""}, None, None)
		self.assertEqual(status, "INCONSISTENT")
		self.assertGreaterEqual(len(issues), 4)

	def test_router_records_role_fields_and_derived_target(self):
		def generate(*_args, **_kwargs):
			return json.dumps(payload(False, True)), {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}

		history = []
		router = RoleSeparationRouter(generate)
		decision = router.route("which actor is connected through two constraints?", [Entity("m.a", "A"), Entity("m.b", "B")], history)
		self.assertEqual(decision.shared_target["type"], "INTERMEDIATE")
		self.assertEqual(history[-1]["type"], "phase3b_4_answer_convergence_role_separation_router")
		self.assertEqual(history[-1]["derived_shared_target_type"], "INTERMEDIATE")
		self.assertEqual(history[-1]["requested_answer_role"]["entity_type"], "person")

	def test_branch_final_preserves_answer_when_roles_match(self):
		def generate(*_args, **_kwargs):
			return json.dumps(payload(True, False)), {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}

		history = []
		decision = RoleSeparationRouter(generate).route(
			"which person is directly constrained by both sources?",
			[Entity("m.a", "A"), Entity("m.b", "B")],
			history,
		)
		self.assertEqual(decision.shared_target["type"], "ANSWER")


if __name__ == "__main__":
	unittest.main()
