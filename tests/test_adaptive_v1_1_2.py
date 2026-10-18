import json
import unittest
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from chain_of_relations.methods.adaptive_v1_1_2.agent import AdaptiveRoutingAgent
from chain_of_relations.methods.adaptive_v1_1_2.router import AdaptiveRouter
from scripts.analyze_adaptive_v1_1_2 import branch_self_sufficiency
from chain_of_relations.schema import Entity


def make_router(payload):
	def generate(*_args, **_kwargs):
		return json.dumps(payload), {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

	return AdaptiveRouter(generate)


class TestAdaptiveV112Router(unittest.TestCase):
	def test_linear_classification(self):
		decision = make_router({
			"structure": "LINEAR", "execution_mode": "cor", "confidence": 0.95,
			"reason": "Sequential book to author to school path.",
			"shared_answer_type": "school", "branches": [], "unsupported_structure": False,
		}).route("Where did the author go to school?", [Entity("m.book", "A Book")], [])
		self.assertEqual(decision.structure, "LINEAR")
		self.assertEqual(decision.execution_mode, "cor")
		self.assertTrue(decision.validation_passed)

	def test_branch_final_classification(self):
		decision = make_router({
			"structure": "BRANCH_FINAL", "execution_mode": "branch", "confidence": 0.95,
			"reason": "Two players directly constrain one tournament.",
			"shared_answer_type": "tennis tournament",
			"branches": [
				{"branch_id": "a", "root_entity_id": "m.k", "root_entity_name": "Katarina", "constraint_summary": "Events Katarina won.", "target_answer_type": "tennis tournament"},
				{"branch_id": "b", "root_entity_id": "m.a", "root_entity_name": "Andrea Hlaváčková", "constraint_summary": "Events Andrea lost.", "target_answer_type": "tennis tournament"},
			],
		}).route("Which tournament?", [Entity("m.k", "Katarina"), Entity("m.a", "Andrea Hlaváčková")], [])
		self.assertEqual(decision.structure, "BRANCH_FINAL")
		self.assertEqual(decision.execution_mode, "branch")
		self.assertTrue(decision.validation_passed)

	def test_branch_intermediate_is_logged_but_executes_as_cor(self):
		decision = make_router({
			"structure": "BRANCH_INTERMEDIATE", "execution_mode": "cor", "confidence": 0.9,
			"reason": "Two constraints identify an engine before designer.",
			"shared_answer_type": "rocket engine", "branches": [], "unsupported_structure": True,
		}).route("Who designed it?", [Entity("m.a", "Methane"), Entity("m.b", "Staged combustion cycle")], [])
		self.assertEqual(decision.structure, "BRANCH_INTERMEDIATE")
		self.assertEqual(decision.execution_mode, "cor")
		self.assertTrue(decision.unsupported_structure)
		self.assertTrue(decision.validation_passed)

	def test_uncertain_defaults_to_cor(self):
		decision = make_router({
			"structure": "UNCERTAIN", "execution_mode": "cor", "confidence": 0.9,
			"reason": "The shared answer space is unclear.", "shared_answer_type": "", "branches": [], "unsupported_structure": False,
		}).route("What is the answer?", [Entity("m.a", "A")], [])
		self.assertEqual(decision.structure, "UNCERTAIN")
		self.assertEqual(decision.execution_mode, "cor")
		self.assertTrue(decision.validation_passed)

	def test_generic_root_rejection(self):
		decision = make_router({
			"structure": "BRANCH_FINAL", "execution_mode": "branch", "confidence": 0.99,
			"reason": "Two constraints.", "shared_answer_type": "person",
			"branches": [
				{"branch_id": "a", "root_entity_id": "m.school", "root_entity_name": "School", "constraint_summary": "School relation."},
				{"branch_id": "b", "root_entity_id": "m.male", "root_entity_name": "Male", "constraint_summary": "Gender relation."},
			],
		}).route("Which person?", [Entity("m.school", "School"), Entity("m.male", "Male")], [])
		self.assertFalse(decision.validation_passed)
		self.assertTrue(any("generic/type-like" in item for item in decision.validation_failures))
		self.assertEqual(decision.execution_mode, "cor")

	def test_shared_final_answer_space_validation(self):
		decision = make_router({
			"structure": "BRANCH_FINAL", "execution_mode": "branch", "confidence": 0.99,
			"reason": "Two constraints.", "shared_answer_type": "event",
			"branches": [
				{"branch_id": "a", "root_entity_id": "m.a", "root_entity_name": "Artist A", "constraint_summary": "Events played.", "target_answer_type": "event"},
				{"branch_id": "b", "root_entity_id": "m.b", "root_entity_name": "Artist B", "constraint_summary": "Awards received.", "target_answer_type": "person"},
			],
		}).route("Which event?", [Entity("m.a", "Artist A"), Entity("m.b", "Artist B")], [])
		self.assertFalse(decision.validation_passed)
		self.assertIn("different final answer spaces", " ".join(decision.validation_failures))


class TestAdaptiveV112Fallback(unittest.TestCase):
	def test_branch_intermediate_executes_as_cor_and_logs_unsupported(self):
		class FakeCoR:
			def _llm_generate(self, *_args, **_kwargs):
				return json.dumps({
					"structure": "BRANCH_INTERMEDIATE", "execution_mode": "cor", "confidence": 0.9,
					"reason": "Constraints meet at an intermediate.", "shared_answer_type": "engine", "branches": [], "unsupported_structure": True,
				}), {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

			def answer(self, question, topic_entities, prompt_history, sparql_history, step_history):
				return {"action": "Stop", "results": [{"id": "m.answer", "name": "Answer"}], "reasoning_chains": ""}

		result = AdaptiveRoutingAgent(FakeCoR()).answer(
			"Who designed it?", [Entity("m.a", "Methane"), Entity("m.b", "Staged combustion cycle")],
			prompt_history=[], sparql_history=[], step_history=[],
		)
		self.assertEqual(result["router_structure"], "BRANCH_INTERMEDIATE")
		self.assertEqual(result["final_execution_mode"], "cor")
		self.assertTrue(result["unsupported_structure"])
		self.assertFalse(result["fallback_used"])

	def test_branch_self_sufficiency_calculation(self):
		rows = [
			{"router_structure": "BRANCH_FINAL", "router_execution_mode": "branch", "validation_passed": True, "fallback_used": False, "final_execution_mode": "branch"},
			{"router_structure": "BRANCH_FINAL", "router_execution_mode": "branch", "validation_passed": True, "fallback_used": True, "final_execution_mode": "cor_fallback"},
			{"router_structure": "LINEAR", "router_execution_mode": "cor", "validation_passed": True, "fallback_used": False, "final_execution_mode": "cor"},
		]
		result = branch_self_sufficiency(rows)
		self.assertEqual(result["branch_final_selections"], 2)
		self.assertEqual(result["native_branch_successes"], 1)
		self.assertEqual(result["branch_fallbacks"], 1)
		self.assertEqual(result["branch_self_sufficiency_rate"], 0.5)

	def test_fallback_keeps_native_and_final_results_separate(self):
		class FakeCoR:
			def __init__(self):
				self.calls = 0

			def _llm_generate(self, *_args, **_kwargs):
				return json.dumps({
					"structure": "BRANCH_FINAL", "execution_mode": "branch", "confidence": 0.99,
					"reason": "Two direct constraints.", "shared_answer_type": "answer", "unsupported_structure": False,
					"branches": [
						{"branch_id": "a", "root_entity_id": "m.a", "root_entity_name": "A", "constraint_summary": "A answer."},
						{"branch_id": "b", "root_entity_id": "m.b", "root_entity_name": "B", "constraint_summary": "B answer."},
					],
				}), {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

			def answer(self, question, topic_entities, prompt_history, sparql_history, step_history):
				self.calls += 1
				if len(topic_entities) == 1:
					name = "Native A" if self.calls == 1 else "Native B"
				else:
					name = "Fallback Answer"
				return {"action": "Stop", "results": [{"id": f"m.{self.calls}", "name": name}], "reasoning_chains": ""}

		fake = FakeCoR()
		result = AdaptiveRoutingAgent(fake).answer(
			"Which answer?", [Entity("m.a", "A"), Entity("m.b", "B")],
			prompt_history=[], sparql_history=[], step_history=[],
		)
		self.assertTrue(result["fallback_used"])
		self.assertEqual(result["final_execution_mode"], "cor_fallback")
		self.assertEqual(result["results"][0]["name"], "Fallback Answer")
		self.assertEqual(result["adaptive_native_result"]["results"], [])
		self.assertEqual(result["adaptive_native_result"]["merge_outcome"], "NO_INTERSECTION_VALID_DISAGREEMENT")
		self.assertEqual(result["native_execution_mode"], "branch")


if __name__ == "__main__":
	unittest.main()
