import json
import unittest

from chain_of_relations.methods.adaptive_v1_1.router import AdaptiveRouter
from chain_of_relations.methods.adaptive_v1_1.agent import AdaptiveRoutingAgent
from chain_of_relations.methods.branch_v1.agent import BranchV1Agent
from chain_of_relations.schema import Entity


def make_router(payload):
	def generate(*_args, **_kwargs):
		return json.dumps(payload), {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

	return AdaptiveRouter(generate)


class AdaptiveBranchRoutingTests(unittest.TestCase):
	def test_exact_id_merge_has_priority(self):
		merged, methods = BranchV1Agent._merge_candidates(
			[
				[{"id": "m.answer", "name": "Conservatism"}],
				[{"id": "m.answer", "name": "Conservatism"}, {"id": "m.other", "name": "Liberalism"}],
			]
		)
		self.assertEqual(merged, [{"id": "m.answer", "name": "Conservatism"}])
		self.assertEqual(methods, ["ID"])

	def test_normalized_label_matches_different_ids(self):
		merged, methods = BranchV1Agent._merge_candidates(
			[
				[{"id": "m.generated", "name": '“Steve  Jobs”'}],
				[{"id": "m.freebase", "name": "Steve Jobs"}],
			]
		)
		self.assertEqual(merged[0]["name"], '“Steve  Jobs”')
		self.assertEqual(methods, ["NORMALIZED_LABEL"])

	def test_genuine_non_overlapping_valid_branches_do_not_merge(self):
		merged, methods = BranchV1Agent._merge_candidates(
			[
				[{"id": "m.one", "name": "Qatar Ladies Open"}],
				[{"id": "m.two", "name": "French Open"}],
			]
		)
		self.assertEqual(merged, [])
		self.assertEqual(methods, [])

	def test_empty_branch_is_unusable_and_falls_back(self):
		unusable, reason = AdaptiveRoutingAgent._branch_result_is_unusable(
			{
				"action": "MERGE_AND",
				"branch_candidate_counts": [1, 0],
				"merged_candidate_count": 0,
			}
		)
		self.assertTrue(unusable)
		self.assertEqual(reason, "branch_empty")

	def test_candidate_preservation_trace_covers_retrieval_to_merge(self):
		class FakeCoR:
			def answer(self, question, topic_entities, prompt_history, sparql_history, step_history):
				sparql_history.append(
					{
						"search_type": "entity_search",
						"result_count": 1,
						"candidate_entity_ids": ["m.answer"],
						"candidate_entity_names": ["Conservatism"],
						"literal_candidates": [],
					}
				)
				step_history.append({"step_type": "llm", "operation_type": "reasoning", "answers": ["Conservatism"]})
				return {
					"action": "Stop",
					"results": [{"id": "m.answer", "name": "Conservatism"}],
					"reasoning_chains": "",
				}

		result = BranchV1Agent(FakeCoR()).answer(
			question="Which ideology?",
			topic_entities=[Entity("m.a", "Party A"), Entity("m.b", "Party B")],
			branch_plan=[
				{"root_entity_id": "m.a", "root_entity_name": "Party A", "constraint_summary": "A ideology"},
				{"root_entity_id": "m.b", "root_entity_name": "Party B", "constraint_summary": "B ideology"},
			],
		)
		trace = result["branch_traces"][0]
		self.assertEqual(trace["retrieved_candidates"][0]["candidates"][0]["id"], "m.answer")
		self.assertEqual(trace["post_filter_candidates"], [["Conservatism"]])
		self.assertEqual(trace["final_branch_answers"][0]["name"], "Conservatism")
		self.assertEqual(result["merge_method"], "ID")

	def test_empty_shared_answer_type_is_rejected(self):
		payload = {
			"mode": "branch",
			"branch_count": 2,
			"shared_answer_type": "",
			"confidence": 0.99,
			"reason": "Two constraints.",
			"branches": [
				{"branch_id": "a", "root_entity_id": "m.a", "root_entity_name": "Artist A", "constraint_summary": "Events played."},
				{"branch_id": "b", "root_entity_id": "m.b", "root_entity_name": "Artist B", "constraint_summary": "Events played."},
			],
		}
		decision = make_router(payload).route(
			"Which event?",
			[Entity("m.a", "Artist A"), Entity("m.b", "Artist B")],
			[],
		)
		self.assertFalse(decision.validation_passed)
		self.assertIn("shared_answer_type is empty", decision.validation_failures)

	def test_sequential_multi_hop_selects_cor(self):
		router = make_router(
			{
				"mode": "cor",
				"branch_count": 0,
				"shared_answer_type": "school",
				"confidence": 0.91,
				"reason": "Sequential author-to-education path.",
				"branches": [],
			}
		)
		decision = router.route(
			"Where did the author go to high school?",
			[Entity("m.book", "To Kill a Mockingbird")],
			[],
		)
		self.assertEqual(decision.router_mode, "cor")
		self.assertTrue(decision.validation_passed)
		self.assertEqual(decision.branch_count, 0)

	def test_independent_shared_answer_selects_two_branches(self):
		router = make_router(
			{
				"mode": "branch",
				"branch_count": 2,
				"shared_answer_type": "tennis tournament",
				"confidence": 0.91,
				"reason": "Two player-specific constraints identify the same event.",
				"branches": [
					{
						"branch_id": "branch_a",
						"root_entity_id": "m.katarina",
						"root_entity_name": "Katarina",
						"constraint_summary": "Events Katarina won.",
					},
					{
						"branch_id": "branch_b",
						"root_entity_id": "m.andrea",
						"root_entity_name": "Andrea Hlaváčková",
						"constraint_summary": "Events Andrea lost.",
					},
				],
			}
		)
		decision = router.route(
			"Which tournament did Katarina beat Andrea Hlaváčková in?",
			[Entity("m.katarina", "Katarina"), Entity("m.andrea", "Andrea Hlaváčková")],
			[],
		)
		self.assertEqual(decision.router_mode, "branch")
		self.assertTrue(decision.validation_passed)
		self.assertEqual(decision.branch_count, 2)

	def test_generic_roots_are_rejected(self):
		router = make_router(
			{
				"mode": "branch",
				"branch_count": 2,
				"shared_answer_type": "person",
				"confidence": 0.99,
				"reason": "Two constraints.",
				"branches": [
					{"branch_id": "a", "root_entity_id": "m.school", "root_entity_name": "School", "constraint_summary": "School relation."},
					{"branch_id": "b", "root_entity_id": "m.male", "root_entity_name": "Male", "constraint_summary": "Gender relation."},
				],
			}
		)
		decision = router.route(
			"Which person?",
			[Entity("m.school", "School"), Entity("m.male", "Male")],
			[],
		)
		self.assertFalse(decision.validation_passed)
		self.assertTrue(any("generic/type-like" in item for item in decision.validation_failures))

	def test_more_than_three_branches_is_rejected(self):
		entities = [Entity(f"m.{i}", f"Person {i}") for i in range(4)]
		payload = {
			"mode": "branch",
			"branch_count": 4,
			"shared_answer_type": "event",
			"confidence": 0.99,
			"reason": "Four constraints.",
			"branches": [
				{
					"branch_id": str(i),
					"root_entity_id": entity.id,
					"root_entity_name": entity.name,
					"constraint_summary": f"Constraint {i}",
				}
				for i, entity in enumerate(entities)
			],
		}
		decision = make_router(payload).route("Which event?", entities, [])
		self.assertFalse(decision.validation_passed)
		self.assertTrue(any("MAX_BRANCHES" in item for item in decision.validation_failures))

	def test_duplicate_branches_are_rejected(self):
		payload = {
			"mode": "branch",
			"branch_count": 2,
			"shared_answer_type": "event",
			"confidence": 0.99,
			"reason": "Duplicate proposal.",
			"branches": [
				{"branch_id": "a", "root_entity_id": "m.a", "root_entity_name": "Artist A", "constraint_summary": "Events played."},
				{"branch_id": "b", "root_entity_id": "m.a", "root_entity_name": "Artist A", "constraint_summary": "Events played."},
			],
		}
		decision = make_router(payload).route(
			"Which event?",
			[Entity("m.a", "Artist A"), Entity("m.b", "Artist B")],
			[],
		)
		self.assertFalse(decision.validation_passed)
		self.assertIn("branch roots are duplicated", decision.validation_failures)
		self.assertIn("duplicate branch root and constraint", decision.validation_failures)


if __name__ == "__main__":
	unittest.main()
