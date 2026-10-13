"""Reasoning-structure graph and rendering for S-expression ASTs."""

from __future__ import annotations

from dataclasses import dataclass

from .sexpr_ast import Atom, Expr, ExprNode


@dataclass(frozen=True)
class LogicalGraphNode:
    node_id: int
    label: str
    kind: str


@dataclass(frozen=True)
class LogicalGraphEdge:
    source: int
    target: int


@dataclass
class LogicalFormGraph:
    nodes: list[LogicalGraphNode]
    edges: list[LogicalGraphEdge]
    ascii: str


def _render(node: Expr, prefix: str = "", is_last: bool = True) -> list[str]:
    connector = "└── " if is_last else "├── "
    if isinstance(node, Atom):
        lines = [f"{prefix}{connector}{node.value}"]
        return lines
    lines = [f"{prefix}{connector}{node.operator}"]
    child_prefix = prefix + ("    " if is_last else "│   ")
    for index, argument in enumerate(node.arguments):
        lines.extend(_render(argument, child_prefix, index == len(node.arguments) - 1))
    return lines


def build_logical_form_graph(root: ExprNode) -> LogicalFormGraph:
    nodes: list[LogicalGraphNode] = []
    edges: list[LogicalGraphEdge] = []

    def visit(node: Expr, parent: int | None = None) -> int:
        node_id = len(nodes)
        if isinstance(node, ExprNode):
            graph_node = LogicalGraphNode(node_id, node.operator, "operator")
        else:
            graph_node = LogicalGraphNode(node_id, node.value, node.atom_type)
        nodes.append(graph_node)
        if parent is not None:
            edges.append(LogicalGraphEdge(parent, node_id))
        if isinstance(node, ExprNode):
            for argument in node.arguments:
                visit(argument, node_id)
        return node_id

    visit(root)
    return LogicalFormGraph(nodes, edges, "\n".join(_render(root)))
