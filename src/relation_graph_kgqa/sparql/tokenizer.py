"""A deliberately small tokenizer for the SPARQL constructs used by CWQ."""

from __future__ import annotations

from typing import List


_PUNCTUATION = set("{}();,[].")


def tokenize_sparql(text: str) -> List[str]:
    """Tokenize SPARQL without trying to implement the full SPARQL grammar.

    IRIs, quoted literals, variables, prefixed names, operators, and grouping
    punctuation are kept as individual tokens. Dots in Freebase predicates and
    decimal literals are retained unless the dot is a statement terminator.
    """

    tokens: List[str] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char == "#":
            newline = text.find("\n", index)
            index = length if newline == -1 else newline + 1
            continue

        if text.startswith('"""', index) or text.startswith("'''", index):
            quote = text[index : index + 3]
            end = text.find(quote, index + 3)
            if end == -1:
                tokens.append(text[index:])
                break
            tokens.append(text[index : end + 3])
            index = end + 3
            continue

        if char in {'"', "'"}:
            quote = char
            cursor = index + 1
            while cursor < length:
                if text[cursor] == "\\":
                    cursor += 2
                    continue
                if text[cursor] == quote:
                    cursor += 1
                    break
                cursor += 1
            tokens.append(text[index:cursor])
            index = cursor
            continue

        if char == "<":
            closing = text.find(">", index + 1)
            inside = text[index + 1 : closing] if closing != -1 else ""
            if closing != -1 and inside and not any(item.isspace() for item in inside):
                tokens.append(text[index : closing + 1])
                index = closing + 1
                continue

        if char in "!<>=":
            if index + 1 < length and text[index : index + 2] in {"!=", "<=", ">=", "="}:
                tokens.append(text[index : index + 2])
                index += 2
            else:
                tokens.append(char)
                index += 1
            continue

        if char in _PUNCTUATION and char != ".":
            tokens.append(char)
            index += 1
            continue

        if char == ".":
            next_char = text[index + 1] if index + 1 < length else ""
            if not next_char or next_char.isspace() or next_char in "{}();,":
                tokens.append(".")
                index += 1
                continue

        cursor = index
        while cursor < length:
            current = text[cursor]
            if current.isspace() or current in "{}();,":
                break
            if current == "#":
                break
            if current == ".":
                next_char = text[cursor + 1] if cursor + 1 < length else ""
                if not next_char or next_char.isspace() or next_char in "{}();,?\"'":
                    break
            cursor += 1
        if cursor == index:
            tokens.append(char)
            index += 1
        else:
            tokens.append(text[index:cursor])
            index = cursor

    return tokens


def matching_token(tokens: List[str], start: int, opening: str = "{", closing: str = "}") -> int:
    """Return the matching closing token index, or ``-1``."""

    depth = 0
    for index in range(start, len(tokens)):
        token = tokens[index]
        if token == opening:
            depth += 1
        elif token == closing:
            depth -= 1
            if depth == 0:
                return index
    return -1
