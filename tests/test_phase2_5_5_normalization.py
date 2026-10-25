import json
import unittest

from chain_of_relations.methods.phase2_5_5.normalizer import normalize_proposal
from chain_of_relations.methods.phase2_5_5.router import CanonicalNormalizationRouter
from chain_of_relations.schema import Entity


def base_payload():
	return {
		"shared_target": "answer",
		"dependency_pattern": "independent_parallel_branches",
		"structure": "BRANCH_FINAL",
		"confidence": 0.99,
		"requested_final_answer_type": "political ideology",
		"shared_answer_type": "political ideology",
		"requested_final_semantic_property": "political_ideology",
		"constraints": [
			{
				"constraint_id": "c1",
				"source_anchor": "m.a",
				"target_variable_role": "final_answer",
				"answer_relevant": True,
				"semantic_property": "associated_with",
				"constraint_text": "A constrains the answer",
			},
			{
				"constraint_id": "c2",
				"source_anchor": "m.b",
				"target_variable_role": "final_answer_entity",
				"answer_relevant": True,
				"semantic_property": "associated_with",
				"constraint_text": "B constrains the answer",
			},
		],
		"independent_source_groups": [
			{"source_anchor": "m.a", "constraint_ids": ["c1"]},
			{"source_anchor": "m.b", "constraint_ids": ["c2"]},
		],
		"branches": [],
	}


class TestPhase255Normalization(unittest.TestCase):
	def test_shared_target_is_canonicalized_without_semantic_change(self):
		result = normalize_proposal(base_payload(), "which political ideology?")
		proposal = result["canonical_proposal"]
		self.assertEqual(proposal["shared_target"]["type"], "ANSWER")
		self.assertIn("shared political ideology target", proposal["shared_target"]["description"])
		self.assertFalse(result["fields"]["shared_target"]["semantic_change"])

	def test_dependency_and_roles_use_known_equivalents(self):
		result = normalize_proposal(base_payload(), "which political ideology?")
		proposal = result["canonical_proposal"]
		self.assertEqual(proposal["dependency_pattern"], "INDEPENDENT_CONVERGENCE")
		self.assertEqual([item["target_role"] for item in proposal["constraints"]], ["ANSWER", "ANSWER"])

	def test_unknown_values_are_not_forced_into_valid_semantics(self):
		payload = base_payload()
		payload["shared_target"] = "SOMETHING_ELSE"
		payload["dependency_pattern"] = "unrelated semantic graph"
		payload["constraints"][0]["target_variable_role"] = "unseen_role"
		result = normalize_proposal(payload, "question")
		self.assertEqual(result["canonical_proposal"]["shared_target"]["type"], "SOMETHING_ELSE")
		self.assertEqual(result["canonical_proposal"]["dependency_pattern"], "UNKNOWN")
		self.assertEqual(result["canonical_proposal"]["constraints"][0]["target_role"], "UNKNOWN")
		self.assertEqual(result["normalization_status"], "PARTIAL_NORMALIZATION")
		self.assertTrue(result["normalization_errors"])

	def test_intermediate_is_never_promoted_to_answer(self):
		payload = base_payload()
		payload["shared_target"] = "INTERMEDIATE"
		result = normalize_proposal(payload, "question")
		self.assertEqual(result["canonical_proposal"]["shared_target"]["type"], "INTERMEDIATE")

	def test_normalization_is_deterministic(self):
		payload = base_payload()
		self.assertEqual(normalize_proposal(payload, "question"), normalize_proposal(payload, "question"))

	def test_router_calls_model_once_and_keeps_validator_path(self):
		calls = []

		def generate(*_args, **_kwargs):
			calls.append(1)
			return json.dumps(base_payload()), {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}

		router = CanonicalNormalizationRouter(generate)
		history = []
		decision = router.route("which political ideology?", [Entity("m.a", "A"), Entity("m.b", "B")], history)
		self.assertEqual(calls, [1])
		self.assertEqual(history[-1]["type"], "phase2_5_5_canonical_normalization_router")
		self.assertEqual(history[-1]["normalization"]["canonical_proposal"]["shared_target"]["type"], "ANSWER")
		self.assertTrue(decision.proposed_branches)


if __name__ == "__main__":
	unittest.main()
