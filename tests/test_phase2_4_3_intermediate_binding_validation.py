import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_phase2_4_3_intermediate_binding_validation import (  # noqa: E402
    _nested_join_dependencies,
    binding_audit,
    choose_decision,
    dependent_constraint_statuses,
    group_summary,
    semantic_success,
)
from relation_graph_kgqa.logical_forms.sexpr_parser import parse_s_expression  # noqa: E402


class Phase243IntermediateBindingTests(unittest.TestCase):
    def test_nested_join_extracts_one_intermediate_without_duplicate_visits(self):
        root = parse_s_expression(
            "(AND answer_type (JOIN outer.relation (JOIN inner.relation m.anchor)))"
        )
        dependencies = []
        _nested_join_dependencies(root, dependencies, [0])
        self.assertEqual(len(dependencies), 1)
        self.assertEqual(dependencies[0]["intermediate_variable"], "?intermediate_1")
        self.assertEqual(dependencies[0]["binding_relation"], "(JOIN inner.relation m.anchor)")

    def test_binding_preserved_requires_runtime_continuation(self):
        detail = {
            "router_structure": "LINEAR",
            "step_history": [
                {
                    "step_type": "sparql",
                    "search_type": "relation_search_head",
                    "sparql": "SELECT ?relation WHERE { ns:m.a ns:r.one ?e_1 . ?e_1 ?relation ?x }",
                },
                {
                    "step_type": "sparql",
                    "search_type": "entity_search",
                    "sparql": "SELECT ?targetEntity WHERE { ns:m.a ns:r.one ?e_1 . ?e_1 ns:r.two ?targetEntity }",
                },
            ],
        }
        result = binding_audit(detail, True)
        self.assertEqual(result["binding_status"], "BINDING_PRESERVED")
        self.assertEqual(result["execution_binding_status"], "LINEAR_BINDING_PRESERVED")

        not_created = binding_audit({"router_structure": "LINEAR", "step_history": []}, True)
        self.assertEqual(not_created["binding_status"], "BINDING_NOT_CREATED")

    def test_dependent_constraint_statuses_reflect_missing_binding(self):
        gold = {
            "intermediates": [
                {
                    "intermediate_variable": "?x",
                    "dependent_constraints": [{"constraint": "r2"}],
                }
            ]
        }
        binding = {"binding_status": "BINDING_NOT_CREATED"}
        state = {"explicit_wrong_state": False}
        statuses = dependent_constraint_statuses(gold, binding, state)
        self.assertEqual(statuses[0]["status"], "OMITTED")

    def test_answer_only_success_is_not_semantic_validation(self):
        self.assertEqual(
            semantic_success(True, [], {"lost_relations": ["r2"]}, {"branch_traces": []}),
            "ANSWER_ONLY_SUCCESS",
        )
        self.assertEqual(
            semantic_success(
                True,
                [{"status": "APPLIED_TO_CORRECT_BINDING"}],
                {"lost_relations": ["alternate_gold_relation"]},
                {"branch_traces": [{"status": "SUCCESS"}]},
            ),
            "SEMANTICALLY_VALID_SUCCESS",
        )

    def test_group_aggregation_and_partial_decision(self):
        rows = []
        for index in range(4):
            rows.append(
                {
                    "group": "INTERMEDIATE_DEPENDENT",
                    "intermediate_variable": "?x",
                    "binding_status": "BINDING_NOT_CREATED" if index == 0 else "BINDING_PRESERVED",
                    "dependent_constraints": [{"status": "OMITTED"}] if index == 0 else [{"status": "APPLIED_TO_CORRECT_BINDING"}],
                    "answer_correct": index > 0,
                    "primary_failure": "INTERMEDIATE_BINDING_LOSS" if index == 0 else "NO_BINDING_FAILURE",
                    "semantic_success": "ANSWER_ONLY_SUCCESS" if index > 0 else "NOT_APPLICABLE",
                }
            )
        for group in ("SHARED_ANSWER_BRANCH", "LINEAR_CONTROL"):
            rows.append(
                {
                    "group": group,
                    "intermediate_variable": None,
                    "binding_status": "UNKNOWN",
                    "dependent_constraints": [],
                    "answer_correct": True,
                    "primary_failure": "NO_BINDING_FAILURE",
                    "semantic_success": "ANSWER_ONLY_SUCCESS",
                }
            )
        summary = group_summary(rows)
        self.assertEqual(summary["INTERMEDIATE_DEPENDENT"]["binding_loss_count"], 1)
        self.assertEqual(choose_decision(rows, summary), "PARTIAL_INTERMEDIATE_BINDING_SIGNAL")


if __name__ == "__main__":
    unittest.main()
