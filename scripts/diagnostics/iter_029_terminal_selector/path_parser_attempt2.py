"""Parse relation IDs from CoR's serialized triple-line reasoning paths."""
from __future__ import annotations

import re


_NODE = re.compile(r"^(?:e\d+|TopicEntity\(.*\))$", re.DOTALL)
_TRIPLE = re.compile(
    r"^\s*((?:e\d+|TopicEntity\(.*\)))\s+(\S+)\s+((?:e\d+|TopicEntity\(.*\)))\s+\.\s*$"
)


def parse_executed_relation_ids(path: object) -> list[str]:
    """Return distinct relation predicates in path order.

    CoR serializes one traversed edge per line as ``node relation node .``.
    ``candidate.target`` is a display-only terminal annotation and is excluded.
    Both outgoing and incoming edges use the same predicate position, so their
    traversal direction does not alter the Freebase relation ID.
    """
    if not isinstance(path, str) or not path.strip():
        return []
    result: list[str] = []
    seen: set[str] = set()
    for line in path.splitlines():
        match = _TRIPLE.fullmatch(line)
        if not match:
            continue
        subject, predicate, object_ = (part.strip() for part in match.groups())
        if predicate == "candidate.target":
            continue
        if not _NODE.fullmatch(subject) or not _NODE.fullmatch(object_):
            continue
        if not predicate or any(char.isspace() for char in predicate):
            continue
        if predicate not in seen:
            seen.add(predicate)
            result.append(predicate)
    return result
