import unittest

from relation_graph_kgqa.logical_forms.sexpr_ast import Atom, ExprNode
from relation_graph_kgqa.logical_forms.sexpr_parser import SExprParseError, parse_s_expression


class SExprParserTests(unittest.TestCase):
    def test_parses_arbitrary_nesting(self) -> None:
        root = parse_s_expression("(AND type (JOIN relation (JOIN relation2 m.abc)))")
        self.assertIsInstance(root, ExprNode)
        self.assertEqual(root.operator, "AND")
        self.assertEqual(root.arguments[0], Atom("type", "unknown"))
        nested = root.arguments[1]
        self.assertIsInstance(nested, ExprNode)
        self.assertIsInstance(nested.arguments[1], ExprNode)

    def test_preserves_unknown_operator(self) -> None:
        root = parse_s_expression("(FUTURE_OP (JOIN relation m.abc))")
        self.assertEqual(root.operator, "FUTURE_OP")

    def test_rejects_trailing_or_unbalanced_tokens(self) -> None:
        with self.assertRaises(SExprParseError):
            parse_s_expression("(JOIN relation m.abc))")
        with self.assertRaises(SExprParseError):
            parse_s_expression("(JOIN relation m.abc")
