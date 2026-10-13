import unittest

from relation_graph_kgqa.logical_forms.sexpr_tokenizer import (
    SExprTokenizationError,
    tokenize,
)


class SExprTokenizerTests(unittest.TestCase):
    def test_preserves_nested_token_order(self) -> None:
        tokens = tokenize("(AND type (JOIN relation m.abc))")
        self.assertEqual(
            [token.value for token in tokens],
            ["(", "AND", "type", "(", "JOIN", "relation", "m.abc", ")", ")"],
        )

    def test_reports_unterminated_literal(self) -> None:
        with self.assertRaises(SExprTokenizationError):
            tokenize('(JOIN relation "unterminated)')
