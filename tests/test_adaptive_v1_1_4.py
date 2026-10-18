import json
import unittest

from chain_of_relations.methods.adaptive_v1_1_4 import AdaptiveRouter
from chain_of_relations.methods.adaptive_v1_1_4.agent import AdaptiveRoutingAgent
from chain_of_relations.schema import Entity


def make_router(payload):
	def generate(*_args, **_kwargs):
		return json.dumps(payload), {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

	return AdaptiveRouter(generate)


def branch(branch_id, root_id, root_name, prop, summary, target, *, supported=True, independent=True):
	return {
		"branch_id": branch_id,
		"root_entity_id": root_id,
		"root_entity_name": root_name,
		"constraint_property": prop,
		"constraint_summary": summary,
		"target_answer_type": target,
		"question_supported": supported,
		"independently_generates_final_candidates": independent,
	}


class TestAdaptiveV114Validation(unittest.TestCase):
	def test_same_property_branch_remains_valid(self):
		decision = make_router({
			"structure": "BRANCH_FINAL",
			"execution_mode": "branch",
			"branch_relation_mode": "SAME_PROPERTY",
			"same_final_answer_space": True,
			"confidence": 0.99,
			"requested_final_answer_type": "comic book category",
			"requested_final_semantic_property": "comic book category",
			"shared_answer_type": "comic book category",
			"branches": [
				branch("a", "m.a", "Mystery of the Human Thunderbolt", "comic book category", "category of A", "comic book category"),
				branch("b", "m.b", "The Avengers, Volume 4", "comic book category", "category of B", "comic book category"),
			],
		}).route(
			"what comic book category do A and B belong to?",
			[Entity("m.a", "Mystery of the Human Thunderbolt"), Entity("m.b", "The Avengers, Volume 4")],
			[],
		)
		self.assertTrue(decision.validation_passed)
		self.assertEqual(decision.branch_relation_mode, "SAME_PROPERTY")
		self.assertEqual(decision.execution_mode, "branch")

	def test_complementary_gamecube_japan_branch_is_valid(self):
		decision = make_router({
			"structure": "BRANCH_FINAL",
			"execution_mode": "branch",
			"branch_relation_mode": "COMPLEMENTARY_PROPERTIES",
			"same_final_answer_space": True,
			"confidence": 0.99,
			"requested_final_answer_type": "game version",
			"requested_final_semantic_property": "game version satisfying controller and regional compatibility constraints",
			"shared_answer_type": "game version",
			"branches": [
				branch("a", "m.c", "Nintendo GameCube Controller", "controller compatibility", "games compatible with the controller", "game version"),
				branch("b", "m.j", "Japan", "regional support", "games supported in Japan", "game version"),
			],
		}).route(
			"which game version for the nintendo gamecube controller supports the japan region?",
			[Entity("m.c", "Nintendo GameCube Controller"), Entity("m.j", "Japan")],
			[],
		)
		self.assertTrue(decision.validation_passed)
		self.assertEqual(decision.branch_relation_mode, "COMPLEMENTARY_PROPERTIES")
		self.assertTrue(decision.same_final_answer_space)

	def test_complementary_branch_rejects_different_answer_types(self):
		decision = make_router({
			"structure": "BRANCH_FINAL",
			"execution_mode": "branch",
			"branch_relation_mode": "COMPLEMENTARY_PROPERTIES",
			"same_final_answer_space": True,
			"confidence": 0.99,
			"requested_final_answer_type": "game version",
			"requested_final_semantic_property": "compatibility",
			"shared_answer_type": "game version",
			"branches": [
				branch("a", "m.c", "Controller", "compatibility", "games", "game version"),
				branch("b", "m.j", "Japan", "region", "regions", "region"),
			],
		}).route("which game version is supported?", [Entity("m.c", "Controller"), Entity("m.j", "Japan")], [])
		self.assertFalse(decision.validation_passed)
		self.assertTrue(any("target" in item for item in decision.validation_failures))

	def test_complementary_branch_rejects_intermediate_or_unrelated_branch(self):
		intermediate = make_router({
			"structure": "BRANCH_FINAL",
			"execution_mode": "branch",
			"branch_relation_mode": "COMPLEMENTARY_PROPERTIES",
			"same_final_answer_space": True,
			"confidence": 0.99,
			"requested_final_answer_type": "person",
			"requested_final_semantic_property": "designer via engine",
			"shared_answer_type": "person",
			"branches": [
				branch("a", "m.a", "Methane", "engine fuel", "engine using methane", "person", independent=False),
				branch("b", "m.b", "Staged combustion cycle", "engine cycle", "engine using staged combustion", "person"),
			],
		}).route("who designed the rocket engine using methane and staged combustion?", [Entity("m.a", "Methane"), Entity("m.b", "Staged combustion cycle")], [])
		self.assertFalse(intermediate.validation_passed)
		self.assertTrue(any("independently" in item for item in intermediate.validation_failures))

		unrelated = make_router({
			"structure": "BRANCH_FINAL",
			"execution_mode": "branch",
			"branch_relation_mode": "COMPLEMENTARY_PROPERTIES",
			"same_final_answer_space": True,
			"confidence": 0.99,
			"requested_final_answer_type": "broadcast",
			"requested_final_semantic_property": "broadcast associated with artist and genre",
			"shared_answer_type": "broadcast",
			"branches": [
				branch("a", "m.a", "Bobby Helm", "artist association", "broadcasts by Bobby Helm", "broadcast", supported=False),
				branch("b", "m.b", "Rock music", "genre", "rock broadcasts", "broadcast"),
			],
		}).route("which broadcasts are about weather?", [Entity("m.a", "Bobby Helm"), Entity("m.b", "Rock music")], [])
		self.assertFalse(unrelated.validation_passed)
		self.assertTrue(any("question-supported" in item for item in unrelated.validation_failures))

	def test_rocket_engine_remains_branch_intermediate(self):
		decision = make_router({
			"structure": "BRANCH_INTERMEDIATE",
			"execution_mode": "cor",
			"branch_relation_mode": "",
			"same_final_answer_space": False,
			"confidence": 0.99,
			"requested_final_answer_type": "person",
			"requested_final_semantic_property": "designer via intermediate",
			"shared_answer_type": "person",
			"branches": [],
		}).route("who designed the rocket engine?", [Entity("m.a", "Methane")], [])
		self.assertTrue(decision.validation_passed)
		self.assertEqual(decision.structure, "BRANCH_INTERMEDIATE")
		self.assertEqual(decision.execution_mode, "cor")

class TestAdaptiveV114FallbackLogging(unittest.TestCase):
	def test_invalid_complementary_plan_uses_named_fallback_reason(self):
		class FakeCoR:
			def _llm_generate(self, *_args, **_kwargs):
				return json.dumps({
					"structure": "BRANCH_FINAL",
					"execution_mode": "branch",
					"branch_relation_mode": "COMPLEMENTARY_PROPERTIES",
					"same_final_answer_space": True,
					"confidence": 0.99,
					"requested_final_answer_type": "game version",
					"requested_final_semantic_property": "compatibility",
					"shared_answer_type": "game version",
					"branches": [
						branch("a", "m.a", "Controller", "compatibility", "games", "game version"),
						branch("b", "m.b", "Japan", "region", "regions", "region"),
					],
				}), {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

			def answer(self, question, topic_entities, prompt_history, sparql_history, step_history):
				return {"action": "Stop", "results": [{"id": "m.answer", "name": "Answer"}], "reasoning_chains": ""}

		result = AdaptiveRoutingAgent(FakeCoR()).answer(
			"which game version is supported?",
			[Entity("m.a", "Controller"), Entity("m.b", "Japan")],
			prompt_history=[], sparql_history=[], step_history=[],
		)
		self.assertTrue(result["validation_failed"])
		self.assertTrue(result["fallback_used"])
		self.assertEqual(result["validation_reason"], "invalid_complementary_branch")
		self.assertEqual(result["fallback_reason"], "invalid_complementary_branch")


if __name__ == "__main__":
	unittest.main()
