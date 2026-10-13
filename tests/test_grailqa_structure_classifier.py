import unittest

from relation_graph_kgqa.logical_forms.structure_classifier import analyze_s_expression


class GrailQAStructureClassifierTests(unittest.TestCase):
    def test_linear(self) -> None:
        structure, _ = analyze_s_expression("(AND type (JOIN relation (JOIN relation2 m.a)))")
        self.assertEqual(structure.structure_type, "linear")
        self.assertEqual(structure.branch_count, 0)

    def test_two_and_three_branch(self) -> None:
        two, _ = analyze_s_expression("(AND type (AND (JOIN r1 m.a) (JOIN r2 m.b)))")
        three, _ = analyze_s_expression(
            "(AND type (AND (JOIN r1 m.a) (JOIN r2 m.b) (JOIN r3 m.c)))"
        )
        self.assertEqual(two.structure_type, "branch_2")
        self.assertEqual(two.branch_count, 2)
        self.assertEqual(three.structure_type, "branch_3_plus")
        self.assertEqual(three.branch_count, 3)

    def test_nested_comparative_superlative_and_count(self) -> None:
        nested, _ = analyze_s_expression(
            "(AND type (AND (JOIN r1 m.a) (AND (JOIN r2 m.b) (JOIN r3 m.c))))"
        )
        comparative, _ = analyze_s_expression("(AND type (ge relation 3))")
        superlative, _ = analyze_s_expression("(ARGMAX (JOIN relation m.a) relation)")
        count, _ = analyze_s_expression("(COUNT (AND type (JOIN relation m.a)))")
        self.assertEqual(nested.structure_type, "nested_branch")
        self.assertEqual(nested.nested_branch_count, 1)
        self.assertEqual(comparative.structure_type, "comparative")
        self.assertEqual(superlative.structure_type, "superlative")
        self.assertEqual(count.structure_type, "other_or_unknown")
        self.assertTrue(count.has_count)

    def test_unknown_operator_is_conservative(self) -> None:
        structure, _ = analyze_s_expression("(UNKNOWN_OP (JOIN relation m.a))")
        self.assertEqual(structure.structure_type, "other_or_unknown")
        self.assertIn("UNKNOWN_OP", structure.unsupported_constructs)
