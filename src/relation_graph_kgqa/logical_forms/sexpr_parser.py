"""Recursive parser for GrailQA S-expression logical forms."""

from __future__ import annotations

from relation_graph_kgqa.logical_forms.sexpr_ast import Atom, Expr, ExprNode
from relation_graph_kgqa.logical_forms.sexpr_tokenizer import (
    SExprToken,
    SExprTokenizationError,
    classify_atom,
    tokenize,
)


class SExprParseError(ValueError):
    """Raised for malformed or incomplete S-expressions."""


def _parse_node(tokens: list[SExprToken], index: int) -> tuple[Expr, int]:
    if index >= len(tokens):
        raise SExprParseError("unexpected end of S-expression")
    token = tokens[index]
    if token.kind == "atom":
        return Atom(token.value, classify_atom(token.value)), index + 1
    if token.kind != "lparen":
        raise SExprParseError(f"unexpected ')' at position {token.position}")

    operator_index = index + 1
    if operator_index >= len(tokens) or tokens[operator_index].kind != "atom":
        raise SExprParseError(f"expression at position {token.position} has no operator")
    operator = tokens[operator_index].value
    arguments: list[Expr] = []
    index = operator_index + 1
    while index < len(tokens) and tokens[index].kind != "rparen":
        argument, index = _parse_node(tokens, index)
        arguments.append(argument)
    if index >= len(tokens):
        raise SExprParseError(f"missing ')' for expression starting at position {token.position}")
    return ExprNode(operator, arguments), index + 1


def parse_s_expression(source: str) -> ExprNode:
    try:
        tokens = tokenize(source)
    except SExprTokenizationError as exc:
        raise SExprParseError(str(exc)) from exc
    root, index = _parse_node(tokens, 0)
    if index != len(tokens):
        token = tokens[index]
        raise SExprParseError(f"trailing token {token.value!r} at position {token.position}")
    if not isinstance(root, ExprNode):
        raise SExprParseError("root of an S-expression must be parenthesized")
    return root
