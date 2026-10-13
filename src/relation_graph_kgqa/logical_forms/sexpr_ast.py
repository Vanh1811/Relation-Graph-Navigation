"""AST types for GrailQA S-expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Union


@dataclass(frozen=True)
class Atom:
    value: str
    atom_type: str = "unknown"

    def render(self) -> str:
        return self.value


@dataclass
class ExprNode:
    operator: str
    arguments: list["Expr"]

    def render(self) -> str:
        contents = " ".join(argument.render() for argument in self.arguments)
        return f"({self.operator}{(' ' + contents) if contents else ''})"


Expr = Union[ExprNode, Atom]


def walk_expr(node: Expr) -> Iterator[ExprNode]:
    """Yield every expression node in preorder."""

    if isinstance(node, ExprNode):
        yield node
        for argument in node.arguments:
            yield from walk_expr(argument)
