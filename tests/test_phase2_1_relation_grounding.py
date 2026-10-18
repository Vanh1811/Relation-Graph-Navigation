import json
import unittest

from chain_of_relations.methods.phase2_1.relation_grounding import (
	SemanticTargetRelationGrounder,
	SemanticTargetState,
	build_semantic_target_prompt,
)
from chain_of_relations.schema import Entity, Relation


class TestPhase21RelationGrounding(unittest.TestCase):
	def state(self):
		return SemanticTargetState(
			requested_final_answer_type="person",
			requested_final_semantic_property="starring actor",
			current_subgoal="reach performances and then their actors",
			current_entity_type="film",
			remaining_question_constraint="the film released in 1989",
			reasoning_step_index=1,
			relation_history=("->film.film.release_date",),
		)

	def test_prompt_contains_semantic_target_and_directional_candidates(self):
		prompt = build_semantic_target_prompt(
			"Who starred in the film?",
			Entity("m.film", "Example film"),
			self.state(),
			SemanticTargetRelationGrounder.candidates(
				["film.film.release_date"], ["film.film.starring"]
			),
		)
		self.assertIn("Requested final answer type: person", prompt)
		self.assertIn("direction=incoming", prompt)
		self.assertIn("film.film.starring", prompt)
		self.assertIn("Relation history", prompt)

	def test_valid_response_selects_only_a_candidate_and_preserves_direction(self):
		calls = []

		def generate(prompt, **kwargs):
			calls.append((prompt, kwargs))
			return json.dumps({
				"ranked_relations": [
					{"relation": "film.film.starring", "direction": "incoming"},
					{"relation": "invented.relation", "direction": "outgoing"},
				],
				"selected_relation": "film.film.starring",
				"direction": "incoming",
				"reason": "The target is a starring actor.",
				"confidence": 0.91,
			}), {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}

		decision = SemanticTargetRelationGrounder(generate).select(
			question="Who starred in the film?",
			topic_entity=Entity("m.film", "Example film"),
			relation_chain=[],
			head_relations=["film.film.release_date"],
			tail_relations=["film.film.starring"],
			relation_display_names={},
			state=self.state(),
			top_k=3,
		)
		self.assertTrue(decision.success)
		self.assertEqual(decision.selected_direction, "incoming")
		self.assertEqual([(item.id, item.left) for item in decision.selected_relations], [
			("film.film.starring", False),
		])
		self.assertEqual(decision.confidence, 0.91)
		self.assertEqual(len(calls), 1)

	def test_invalid_response_fails_closed(self):
		def generate(*_args, **_kwargs):
			return json.dumps({
				"ranked_relations": [{"relation": "invented.relation", "direction": "outgoing"}],
				"selected_relation": "invented.relation",
				"direction": "outgoing",
			}), {"total_tokens": 3}

		decision = SemanticTargetRelationGrounder(generate).select(
			question="Which person?",
			topic_entity=Entity("m.x", "X"),
			relation_chain=[],
			head_relations=["people.person.education"],
			tail_relations=[],
			relation_display_names={},
			state=self.state(),
		)
		self.assertFalse(decision.success)
		self.assertEqual(decision.selected_relations, [])
		self.assertIn("no valid candidate", decision.warnings[0])

	def test_immediate_reverse_relation_is_filtered(self):
		def generate(*_args, **_kwargs):
			return json.dumps({
				"ranked_relations": [
					{"relation": "people.person.education", "direction": "incoming"},
					{"relation": "people.person.occupation", "direction": "outgoing"},
				],
				"confidence": 0.8,
			}), {"total_tokens": 2}

		decision = SemanticTargetRelationGrounder(generate).select(
			question="Which occupation?",
			topic_entity=Entity("m.person", "Person"),
			relation_chain=[Relation("people.person.education", left=True)],
			head_relations=["people.person.occupation"],
			tail_relations=["people.person.education"],
			relation_display_names={},
			state=self.state(),
			top_k=3,
		)
		self.assertTrue(decision.success)
		self.assertEqual([item.id for item in decision.selected_relations], ["people.person.occupation"])


if __name__ == "__main__":
	unittest.main()
