import json
import unittest

from chain_of_relations.methods.phase2_5_5.normalizer import normalize_proposal
from chain_of_relations.methods.phase3b_2.router import SharedTargetInterventionRouter
from chain_of_relations.schema import Entity


def constraint(cid, anchor, role="ANSWER", prop="occupation"):
	return {
		"constraint_id": cid,
		"source_anchor": anchor,
		"constraint_text": f"constraint {cid}",
		"target_variable_role": role,
		"answer_relevant": True,
		"semantic_property": prop,
		"depends_on": [],
	}


def payload(target, requires, structure="BRANCH_FINAL"):
	return {
		"constraints": [constraint("c1", "m.a"), constraint("c2", "m.b")],
		"independent_source_groups": [
			{"source_anchor": "m.a", "constraint_ids": ["c1"]},
			{"source_anchor": "m.b", "constraint_ids": ["c2"]},
		],
		"shared_target": {"type": target, "description": f"{target.lower()} target"},
		"shared_target_description": f"{target.lower()} target",
		"requested_answer_description": "requested final answer",
		"requires_downstream_relation": requires,
		"downstream_relation_description": "downstream relation" if requires else "",
		"shared_target_self_check": {
			"convergence_variable_description": "convergence variable",
			"resulting_variable_is_requested_answer": not requires,
			"valid_shared_latent_target": requires,
			"downstream_step_required": requires,
			"rationale": "explicit self-check",
		},
		"dependency_pattern": "MIXED" if requires else "INDEPENDENT_CONVERGENCE",
		"structure": structure,
		"confidence": 0.99,
		"requested_final_answer_type": "occupation",
		"requested_final_semantic_property": "occupation",
		"shared_answer_type": "occupation",
		"branches": [
			{"branch_id": "b1", "root_entity_id": "m.a", "root_entity_name": "A", "constraint_property": "occupation", "constraint_summary": "A", "target_answer_type": "occupation"},
			{"branch_id": "b2", "root_entity_id": "m.b", "root_entity_name": "B", "constraint_property": "occupation", "constraint_summary": "B", "target_answer_type": "occupation"},
		] if structure == "BRANCH_FINAL" else [],
	}


class TestPhase3B2SharedTargetIntervention(unittest.TestCase):
	def make_router(self, model_payload):
		def generate(*_args, **_kwargs):
			return json.dumps(model_payload), {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}
		return SharedTargetInterventionRouter(generate)

	def test_prompt_contains_convergence_self_check_without_gold_labels(self):
		router = self.make_router(payload("INTERMEDIATE", True, structure="LINEAR"))
		prompt = router._build_prompt("which actor played a role in a film?", [Entity("m.a", "A"), Entity("m.b", "B")])
		self.assertIn("If the independent constraints are satisfied", prompt)
		self.assertIn("Do not label INTERMEDIATE merely because the query is multi-hop", prompt)
		self.assertIn("target_variable_role MUST be exactly one of ANSWER, INTERMEDIATE, FILTER, or TYPE", prompt)
		self.assertNotIn("gold SPARQL", prompt.split("Question:", 1)[0].replace("Never use gold answers, gold SPARQL", ""))

	def test_intermediate_target_and_downstream_requirement_are_exposed(self):
		history = []
		router = self.make_router(payload("INTERMEDIATE", True, structure="LINEAR"))
		decision = router.route("who is connected through A and B?", [Entity("m.a", "A"), Entity("m.b", "B")], history)
		self.assertEqual(decision.shared_target["type"], "INTERMEDIATE")
		self.assertEqual(decision.structure, "BRANCH_INTERMEDIATE")
		self.assertEqual(history[-1]["type"], "phase3b_2_shared_target_intervention_router")
		self.assertTrue(history[-1]["shared_target_extraction"]["requires_downstream_relation"])

	def test_answer_target_is_preserved_for_branch_final(self):
		history = []
		router = self.make_router(payload("ANSWER", False, structure="BRANCH_FINAL"))
		decision = router.route("which answer is shared by A and B?", [Entity("m.a", "A"), Entity("m.b", "B")], history)
		self.assertEqual(decision.shared_target["type"], "ANSWER")
		self.assertNotEqual(decision.structure, "BRANCH_INTERMEDIATE")
		self.assertFalse(history[-1]["shared_target_extraction"]["requires_downstream_relation"])

	def test_none_remains_conservative_and_normalization_does_not_infer_semantics(self):
		proposal = payload("NONE", False, structure="LINEAR")
		proposal["shared_target"] = "NONE"
		result = normalize_proposal(proposal, "a linear question")
		self.assertEqual(result["canonical_proposal"]["shared_target"]["type"], "NONE")
		self.assertNotEqual(result["canonical_proposal"]["shared_target"]["type"], "INTERMEDIATE")

	def test_equivalent_intermediate_enum_is_canonicalized_only(self):
		proposal = payload("INTERMEDIATE_ENTITY", True, structure="LINEAR")
		result = normalize_proposal(proposal, "a question")
		self.assertEqual(result["canonical_proposal"]["shared_target"]["type"], "INTERMEDIATE")


if __name__ == "__main__":
	unittest.main()
