"""Small deterministic tokenizer for GrailQA S-expressions."""

from __future__ import annotations

from dataclasses import dataclass
import re


class SExprTokenizationError(ValueError):
    """Raised when an S-expression cannot be tokenized."""


@dataclass(frozen=True)
class SExprToken:
    kind: str
    value: str
    position: int


_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:\^\^.*)?$")


def classify_atom(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return "literal"
    if _NUMBER_RE.match(value):
        return "number"
    if value.startswith(("m.", "g.")):
        return "entity"
    return "unknown"


def tokenize(source: str) -> list[SExprToken]:
    if not isinstance(source, str) or not source.strip():
        raise SExprTokenizationError("S-expression source is empty")

    tokens: list[SExprToken] = []
    index = 0
    length = len(source)
    while index < length:
        char = source[index]
        if char.isspace():
            index += 1
            continue
        if char in "()":
            tokens.append(SExprToken("lparen" if char == "(" else "rparen", char, index))
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
            start = index
            index += 1
            escaped = False
            while index < length:
                current = source[index]
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == quote:
                    index += 1
                    break
                index += 1
            else:
                raise SExprTokenizationError(f"unterminated quoted literal at position {start}")
            tokens.append(SExprToken("atom", source[start:index], start))
            continue

        start = index
        while index < length and not source[index].isspace() and source[index] not in "()":
            index += 1
        if start == index:
            raise SExprTokenizationError(f"unexpected character at position {index}")
        value = source[start:index]
        tokens.append(SExprToken("atom", value, start))

    return tokens
