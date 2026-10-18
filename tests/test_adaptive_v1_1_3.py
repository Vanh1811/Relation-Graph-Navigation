import json
import unittest

from chain_of_relations.methods.adaptive_v1_1_3 import (
	AdaptiveRouter,
	extract_requested_answer_target,
)
from chain_of_relations.methods.adaptive_v1_1_3.agent import AdaptiveRoutingAgent
from chain_of_relations.methods.branch_v1.agent import BranchV1Agent
from chain_of_relations.schema import Entity


def make_router(payload):
	def generate(*_args, **_kwargs):
		return json.dumps(payload), {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

	return AdaptiveRouter(generate)


class TestAnswerTypeExtraction(unittest.TestCase):
	def test_parent_gender_is_gender_and_parent_sex(self):
		result = extract_requested_answer_target(
			"what is the name of the hybrid parent gender that splake and brook trout both belong to?"
		)
		self.assertEqual(result["requested_final_answer_type"], "gender")
		self.assertIn("parent sex", result["requested_final_semantic_property"])

	def test_genus_is_not_inferred_for_parent_gender(self):
		result = extract_requested_answer_target(
			"what is the name of the hybrid parent gender that splake and brook trout both belong to?"
		)
		self.assertNotEqual(result["requested_final_answer_type"], "genus")


class TestAnswerTypeAwareValidation(unittest.TestCase):
	def test_parent_gender_genus_proposal_is_rejected(self):
		decision = make_router({
			"structure": "BRANCH_FINAL",
			"execution_mode": "branch",
			"confidence": 0.99,
			"reason": "Two taxonomic roots.",
			"requested_final_answer_type": "genus",
			"requested_final_semantic_property": "higher classification",
			"shared_answer_type": "genus",
			"branches": [
				{"branch_id": "a", "root_entity_id": "m.a", "root_entity_name": "Splake", "constraint_property": "higher classification", "constraint_summary": "genus of Splake", "target_answer_type": "genus"},
				{"branch_id": "b", "root_entity_id": "m.b", "root_entity_name": "Brook trout", "constraint_property": "higher classification", "constraint_summary": "genus of brook trout", "target_answer_type": "genus"},
			],
		}).route(
			"what is the name of the hybrid parent gender that splake and brook trout both belong to?",
			[Entity("m.a", "Splake"), Entity("m.b", "Brook trout")],
			[],
		)
		self.assertFalse(decision.validation_passed)
		self.assertEqual(decision.execution_mode, "cor")
		self.assertTrue(any("ANSWER_TYPE_MISMATCH" in item for item in decision.validation_failures))

	def test_compatible_branch_properties_are_accepted(self):
		decision = make_router({
			"structure": "BRANCH_FINAL",
			"execution_mode": "branch",
			"confidence": 0.99,
			"reason": "Both entities constrain the same occupation.",
			"requested_final_answer_type": "occupation",
			"requested_final_semantic_property": "shared occupation",
			"shared_answer_type": "occupation",
			"branches": [
				{"branch_id": "a", "root_entity_id": "m.a", "root_entity_name": "Colonel James Braddock", "constraint_property": "occupation", "constraint_summary": "occupation of Braddock", "target_answer_type": "occupation"},
				{"branch_id": "b", "root_entity_id": "m.b", "root_entity_name": "Nikodemos", "constraint_property": "occupation", "constraint_summary": "occupation of Nikodemos", "target_answer_type": "occupation"},
			],
		}).route(
			"what do colonel james braddock and nikodemos have in common? occupation",
			[Entity("m.a", "Colonel James Braddock"), Entity("m.b", "Nikodemos")],
			[],
		)
		self.assertTrue(decision.validation_passed)
		self.assertTrue(decision.branch_property_compatibility)

	def test_incompatible_branch_properties_are_rejected(self):
		decision = make_router({
			"structure": "BRANCH_FINAL",
			"execution_mode": "branch",
			"confidence": 0.99,
			"reason": "Two facts.",
			"requested_final_answer_type": "shared property or entity",
			"requested_final_semantic_property": "shared property",
			"shared_answer_type": "shared property or entity",
			"branches": [
				{"branch_id": "a", "root_entity_id": "m.a", "root_entity_name": "Colonel James Braddock", "constraint_property": "occupation", "constraint_summary": "occupation", "target_answer_type": "shared property or entity"},
				{"branch_id": "b", "root_entity_id": "m.b", "root_entity_name": "Nikodemos", "constraint_property": "rank", "constraint_summary": "military rank", "target_answer_type": "shared property or entity"},
			],
		}).route(
			"what do colonel james braddock and nikodemos have in common?",
			[Entity("m.a", "Colonel James Braddock"), Entity("m.b", "Nikodemos")],
			[],
		)
		self.assertFalse(decision.validation_passed)
		self.assertTrue(any("CONSTRAINT_PROPERTY_MISMATCH" in item for item in decision.validation_failures))

	def test_linear_behavior_remains_cor(self):
		decision = make_router({
			"structure": "LINEAR",
			"execution_mode": "cor",
			"confidence": 0.95,
			"reason": "Sequential path.",
			"requested_final_answer_type": "person",
			"requested_final_semantic_property": "sequential relation path",
			"shared_answer_type": "person",
			"branches": [],
		}).route("Which person?", [Entity("m.a", "A")], [])
		self.assertTrue(decision.validation_passed)
		self.assertEqual(decision.structure, "LINEAR")
		self.assertEqual(decision.execution_mode, "cor")

	def test_branch_intermediate_remains_cor(self):
		decision = make_router({
			"structure": "BRANCH_INTERMEDIATE",
			"execution_mode": "cor",
			"confidence": 0.95,
			"reason": "Constraints meet at an intermediate.",
			"requested_final_answer_type": "designer",
			"requested_final_semantic_property": "designer via intermediate",
			"shared_answer_type": "designer",
			"branches": [],
		}).route("Who designed it?", [Entity("m.a", "Methane")], [])
		self.assertTrue(decision.validation_passed)
		self.assertEqual(decision.execution_mode, "cor")
		self.assertTrue(decision.unsupported_structure)


class TestBranchConstraintGuidance(unittest.TestCase):
	def test_constraint_property_reaches_relation_selection_prompt(self):
		questions = []

		class FakeCoR:
			def answer(self, question, topic_entities, prompt_history, sparql_history, step_history):
				questions.append(question)
				return {"action": "Stop", "results": [{"id": "m.answer", "name": "Soldier"}], "reasoning_chains": ""}

		result = BranchV1Agent(FakeCoR()).answer(
			"What do A and B have in common?",
			[Entity("m.a", "A"), Entity("m.b", "B")],
			branch_plan=[
				{"branch_id": "a", "root_entity_id": "m.a", "root_entity_name": "A", "constraint_property": "occupation", "constraint_summary": "occupation of A", "target_answer_type": "occupation"},
				{"branch_id": "b", "root_entity_id": "m.b", "root_entity_name": "B", "constraint_property": "occupation", "constraint_summary": "occupation of B", "target_answer_type": "occupation"},
			],
			shared_answer_type="occupation",
			requested_final_answer_type="occupation",
			requested_final_semantic_property="shared occupation",
		)
		self.assertEqual(len(questions), 2)
		self.assertTrue(all("Required semantic property for this branch: occupation" in item for item in questions))
		self.assertEqual(result["branch_constraint_properties"], ["occupation", "occupation"])
		self.assertTrue(result["branch_property_compatibility"])


class TestAdaptiveFallbackLogging(unittest.TestCase):
	def test_invalid_plan_keeps_validation_and_fallback_separate(self):
		class FakeCoR:
			def _llm_generate(self, *_args, **_kwargs):
				return json.dumps({
					"structure": "BRANCH_FINAL", "execution_mode": "branch", "confidence": 0.99,
					"reason": "bad properties", "requested_final_answer_type": "genus",
					"requested_final_semantic_property": "higher classification", "shared_answer_type": "genus",
					"branches": [
						{"branch_id": "a", "root_entity_id": "m.a", "root_entity_name": "Splake", "constraint_property": "higher classification", "constraint_summary": "genus", "target_answer_type": "genus"},
						{"branch_id": "b", "root_entity_id": "m.b", "root_entity_name": "Brook trout", "constraint_property": "higher classification", "constraint_summary": "genus", "target_answer_type": "genus"},
					],
				}), {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

			def answer(self, question, topic_entities, prompt_history, sparql_history, step_history):
				return {"action": "Stop", "results": [{"id": "m.answer", "name": "Male"}], "reasoning_chains": ""}

		result = AdaptiveRoutingAgent(FakeCoR()).answer(
			"what is the name of the hybrid parent gender that splake and brook trout both belong to?",
			[Entity("m.a", "Splake"), Entity("m.b", "Brook trout")],
			prompt_history=[], sparql_history=[], step_history=[],
		)
		self.assertTrue(result["validation_failed"])
		self.assertTrue(result["fallback_used"])
		self.assertEqual(result["fallback_reason"], "invalid_branch_plan")
		self.assertIn("requested_final_answer_type", result)
		self.assertEqual(result["results"][0]["name"], "Male")


if __name__ == "__main__":
	unittest.main()
