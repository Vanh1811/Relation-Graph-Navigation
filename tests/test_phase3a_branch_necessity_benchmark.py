import unittest

from scripts.run_phase3a_branch_necessity_benchmark import (
	answer_variable,
	branch_atoms,
	build_future_manifest,
	build_manual_quality_audit,
	extract_answer_values,
	normalize_sparql_wrapper,
	remove_branch_query,
	score_answers,
)


class Phase3ABenchmarkTests(unittest.TestCase):
	def test_nested_grailqa_wrapper_is_normalized(self):
		query = "SELECT (?x0 AS ?value) WHERE { SELECT DISTINCT ?x0 WHERE { ?x0 :p :o . } }"
		normalized = normalize_sparql_wrapper(query)
		self.assertIn("WHERE { ?x0 :p :o .", normalized)
		self.assertFalse(normalized.rstrip().endswith("} }"))

	def test_answer_variable_preserves_grailqa_source_variable(self):
		self.assertEqual(answer_variable("SELECT (?x0 AS ?value) WHERE { ?x0 :p :o . }"), "?x0")
		self.assertEqual(answer_variable("SELECT DISTINCT ?x WHERE { ?x :p :o . }"), "?x")

	def test_branch_final_uses_answer_incident_non_type_atoms(self):
		patterns = [
			{"subject": "?x", "predicate": "type.object.type", "object": ":Thing"},
			{"subject": ":a", "predicate": "p1", "object": "?x"},
			{"subject": "?x", "predicate": "p2", "object": ":b"},
		]
		shared, atoms = branch_atoms("BRANCH_FINAL", "SELECT DISTINCT ?x WHERE { }", patterns)
		self.assertEqual(shared, "?x")
		self.assertEqual([item["predicate"] for item in atoms], ["p1", "p2"])

	def test_branch_intermediate_preserves_downstream_edge(self):
		patterns = [
			{"subject": ":a", "predicate": "p1", "object": "?m"},
			{"subject": ":b", "predicate": "p2", "object": "?m"},
			{"subject": "?m", "predicate": "p3", "object": "?x"},
		]
		shared, atoms = branch_atoms("BRANCH_INTERMEDIATE", "SELECT DISTINCT ?x WHERE { }", patterns)
		self.assertEqual(shared, "?m")
		self.assertEqual([item["predicate"] for item in atoms], ["p1", "p2"])

	def test_answer_set_score_counts_false_positive_expansion(self):
		score = score_answers(["gold", "extra"], ["gold"])
		self.assertEqual(score["gold_overlap_count"], 1)
		self.assertEqual(score["false_positive_count"], 1)
		self.assertLess(score["f1"], 1.0)

	def test_result_binding_uses_alias_variable(self):
		payload = {"results": {"bindings": [{"value": {"type": "uri", "value": "http://rdf.freebase.com/ns/m.abc"}}]}}
		self.assertEqual(extract_answer_values(payload, "SELECT (?x0 AS ?value) WHERE { }"), ["m.abc"])

	def test_single_branch_removal_preserves_the_other_triple(self):
		candidate = {
			"gold_sparql": "SELECT DISTINCT ?x WHERE {\n  :a :p ?x .\n  :b :q ?x .\n}",
			"patterns": [
				{"subject": ":a", "predicate": ":p", "object": "?x", "raw_line": "  :a :p ?x ."},
				{"subject": ":b", "predicate": ":q", "object": "?x", "raw_line": "  :b :q ?x ."},
			],
		}
		query, method = remove_branch_query(candidate, {"pattern_index": 0})
		self.assertEqual(method, "removed_exact_triple_line")
		self.assertNotIn(":a :p ?x", query)
		self.assertIn(":b :q ?x", query)

	def test_future_manifest_excludes_seen_ids(self):
		rows = [
			{"case_id": "seen", "dataset": "cwq", "gold_structure": "BRANCH_FINAL", "branch_count": 2, "necessity_class": "STRONGLY_BRANCH_REQUIRED", "seen_in_prior_diagnostics": True},
			{"case_id": "fresh", "dataset": "grailqa", "gold_structure": "BRANCH_FINAL", "branch_count": 2, "necessity_class": "STRONGLY_BRANCH_REQUIRED", "seen_in_prior_diagnostics": False},
			{"case_id": "control", "dataset": "grailqa", "gold_structure": "BRANCH_FINAL", "branch_count": 2, "necessity_class": "ANSWER_EQUIVALENT_LINEARIZATION", "seen_in_prior_diagnostics": False},
		]
		manifest = build_future_manifest(rows)
		self.assertEqual(manifest["strong_branch_benchmark_ids"], ["fresh"])
		self.assertEqual(manifest["answer_equivalent_branch_control_ids"], ["control"])

	def test_manual_sample_is_deterministic_and_balanced(self):
		candidate = {
			"case_id": "a",
			"question": "q",
			"answer_variable": "?x",
			"shared_target_variable": "",
			"gold_structure": "BRANCH_FINAL",
			"gold_sparql": "SELECT ?x WHERE { :a :p ?x . :b :q ?x . }",
			"patterns": [],
		}
		branch = {"description": "branch", "construction_method": "removed_exact_triple_line", "query_status": "ok", "delta_f1": 1.0, "material_effect": True}
		result = {"case_id": "a", "dataset": "cwq", "gold_structure": "BRANCH_FINAL", "branch_count": 2, "validity_gate": "PASS", "necessity_class": "STRONGLY_BRANCH_REQUIRED", "full_query_score": {}, "branches": [branch, branch]}
		first = build_manual_quality_audit([candidate], [result])
		second = build_manual_quality_audit([candidate], [result])
		self.assertEqual(first, second)
		self.assertEqual(first["sample_size"], 1)


if __name__ == "__main__":
	unittest.main()
