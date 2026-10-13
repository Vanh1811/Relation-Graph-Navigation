"""Structural classification for GrailQA S-expression ASTs."""

from __future__ import annotations

from dataclasses import dataclass, field

from .graph_builder import LogicalFormGraph, build_logical_form_graph
from .sexpr_ast import Expr, ExprNode, walk_expr
from .sexpr_parser import SExprParseError, parse_s_expression


SUPPORTED_OPERATORS = {
    "AND",
    "JOIN",
    "R",
    "COUNT",
    "ARGMAX",
    "ARGMIN",
    "GT",
    "GE",
    "LT",
    "LE",
}
COMPARISON_OPERATORS = {"GT", "GE", "LT", "LE", "EQ", "NE", "=", "!="}


@dataclass
class LogicalStructure:
    structure_type: str
    operators: list[str] = field(default_factory=list)
    branch_count: int = 0
    max_depth: int = 0
    nested_branch_count: int = 0
    max_branch_depth: int = 0
    has_and: bool = False
    has_or: bool = False
    has_argmax: bool = False
    has_argmin: bool = False
    has_count: bool = False
    has_comparison: bool = False
    parse_success: bool = False
    classifier_confidence: float = 0.0
    classifier_reasons: list[str] = field(default_factory=list)
    parser_errors: list[str] = field(default_factory=list)
    unsupported_constructs: list[str] = field(default_factory=list)
    ascii_structure: str = ""


def _branch_arity(node: ExprNode) -> int:
    if node.operator.upper() != "AND":
        return 0
    return sum(isinstance(argument, ExprNode) for argument in node.arguments)


def _collect_properties(root: ExprNode) -> tuple[dict[str, int], int, int, int, int]:
    operator_counts: dict[str, int] = {}
    branch_count = 0
    nested_branch_count = 0
    max_branch_depth = 0
    max_depth = 0

    def visit(node: Expr, depth: int, branch_depth: int) -> None:
        nonlocal branch_count, nested_branch_count, max_branch_depth, max_depth
        if not isinstance(node, ExprNode):
            return
        max_depth = max(max_depth, depth)
        operator = node.operator
        operator_counts[operator] = operator_counts.get(operator, 0) + 1
        arity = _branch_arity(node)
        next_branch_depth = branch_depth
        if arity >= 2:
            branch_count = max(branch_count, arity)
            next_branch_depth = branch_depth + 1
            max_branch_depth = max(max_branch_depth, next_branch_depth)
            if branch_depth > 0:
                nested_branch_count += 1
        for argument in node.arguments:
            visit(argument, depth + 1, next_branch_depth)

    visit(root, 1, 0)
    return operator_counts, branch_count, max_depth, nested_branch_count, max_branch_depth


def _classify(
    operators: list[str],
    branch_count: int,
    nested_branch_count: int,
    has_or: bool,
    has_argmax: bool,
    has_argmin: bool,
    has_count: bool,
    has_comparison: bool,
    unsupported: list[str],
) -> tuple[str, float, list[str]]:
    if unsupported or has_or:
        reasons = [f"unsupported operator: {item}" for item in unsupported]
        if has_or:
            reasons.append("OR is not part of the first-pass GrailQA taxonomy")
        return "other_or_unknown", 0.55, reasons
    if has_argmax or has_argmin:
        return "superlative", 0.92, ["ARGMAX or ARGMIN is present"]
    if has_comparison:
        return "comparative", 0.9, ["a comparison operator is present"]
    if has_count:
        return "other_or_unknown", 0.65, ["COUNT is preserved as aggregation and is not assigned a structural label"]
    if nested_branch_count:
        return "nested_branch", 0.86, ["a branch contains another branch point"]
    if branch_count >= 3:
        return "branch_3_plus", 0.86, [f"a conjunction has {branch_count} logical arms"]
    if branch_count == 2:
        return "branch_2", 0.88, ["a conjunction has 2 logical arms"]
    if branch_count == 0:
        return "linear", 0.84, ["no multi-arm conjunction was detected"]
    return "other_or_unknown", 0.5, ["logical form did not match a supported structural shape"]


def analyze_root(root: ExprNode) -> tuple[LogicalStructure, LogicalFormGraph]:
    counts, branch_count, max_depth, nested_count, max_branch_depth = _collect_properties(root)
    operators = list(counts)
    normalized = {operator.upper() for operator in operators}
    has_or = "OR" in normalized
    has_argmax = "ARGMAX" in normalized
    has_argmin = "ARGMIN" in normalized
    has_count = "COUNT" in normalized
    has_comparison = any(operator.upper() in COMPARISON_OPERATORS for operator in operators)
    unsupported = [operator for operator in operators if operator.upper() not in SUPPORTED_OPERATORS]
    label, confidence, reasons = _classify(
        operators,
        branch_count,
        nested_count,
        has_or,
        has_argmax,
        has_argmin,
        has_count,
        has_comparison,
        unsupported,
    )
    graph = build_logical_form_graph(root)
    structure = LogicalStructure(
        structure_type=label,
        operators=operators,
        branch_count=branch_count,
        max_depth=max_depth,
        nested_branch_count=nested_count,
        max_branch_depth=max_branch_depth,
        has_and="AND" in normalized,
        has_or=has_or,
        has_argmax=has_argmax,
        has_argmin=has_argmin,
        has_count=has_count,
        has_comparison=has_comparison,
        parse_success=True,
        classifier_confidence=confidence,
        classifier_reasons=reasons,
        unsupported_constructs=unsupported,
        ascii_structure=graph.ascii,
    )
    return structure, graph


def analyze_s_expression(source: str | None) -> tuple[LogicalStructure, LogicalFormGraph | None]:
    if not source:
        return (
            LogicalStructure(
                structure_type="other_or_unknown",
                classifier_reasons=["gold S-expression is missing"],
                parser_errors=["gold S-expression is missing"],
            ),
            None,
        )
    try:
        root = parse_s_expression(source)
    except SExprParseError as exc:
        return (
            LogicalStructure(
                structure_type="other_or_unknown",
                classifier_reasons=["S-expression parsing failed"],
                parser_errors=[str(exc)],
            ),
            None,
        )
    return analyze_root(root)
