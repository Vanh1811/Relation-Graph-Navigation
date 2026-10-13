"""Lightweight extraction of graph-relevant SPARQL features."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence

from .tokenizer import matching_token, tokenize_sparql


@dataclass(frozen=True)
class TriplePattern:
    subject: str
    predicate: str
    object: str


@dataclass
class QueryFeatures:
    select_variables: List[str] = field(default_factory=list)
    triple_patterns: List[TriplePattern] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    order_expressions: List[str] = field(default_factory=list)
    limit: int | None = None
    has_union: bool = False
    has_optional: bool = False
    has_minus: bool = False
    has_values: bool = False
    has_filter: bool = False
    has_aggregation: bool = False
    has_order_by: bool = False
    has_limit: bool = False
    has_comparison: bool = False
    has_group_by: bool = False
    has_having: bool = False
    parse_success: bool = False
    parser_errors: List[str] = field(default_factory=list)
    unsupported_constructs: List[str] = field(default_factory=list)


_KEYWORDS = {
    "FILTER",
    "OPTIONAL",
    "MINUS",
    "VALUES",
    "BIND",
    "GRAPH",
    "SERVICE",
    "UNION",
}


def _upper(token: str) -> str:
    return token.upper()


def _skip_parenthesized(tokens: Sequence[str], index: int) -> int:
    if index >= len(tokens) or tokens[index] != "(":
        return index
    match = matching_token(list(tokens), index, "(", ")")
    return len(tokens) if match == -1 else match + 1


def _skip_until_statement_end(tokens: Sequence[str], index: int) -> int:
    paren_depth = 0
    brace_depth = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "(":
            paren_depth += 1
        elif token == ")":
            paren_depth = max(0, paren_depth - 1)
        elif token == "{":
            brace_depth += 1
        elif token == "}":
            if brace_depth == 0 and paren_depth == 0:
                return index
            brace_depth = max(0, brace_depth - 1)
        elif token == "." and paren_depth == 0 and brace_depth == 0:
            return index + 1
        index += 1
    return index


def _consume_object(tokens: Sequence[str], index: int, end: int) -> tuple[str, int]:
    """Consume an object and an adjacent datatype/language suffix."""

    if index >= end:
        return "", index
    object_ = tokens[index]
    index += 1
    while index < end and (
        tokens[index].startswith("^^") or tokens[index].startswith("@")
    ):
        object_ += tokens[index]
        index += 1
    return object_, index


def _parse_group(tokens: Sequence[str], start: int, end: int, features: QueryFeatures) -> None:
    index = start
    while index < end:
        token = tokens[index]
        upper = _upper(token)

        if token == "." or upper == "UNION":
            if upper == "UNION":
                features.has_union = True
            index += 1
            continue
        if token == "{":
            match = matching_token(list(tokens), index, "{", "}")
            if match == -1 or match > end:
                features.parser_errors.append("unmatched group brace")
                return
            _parse_group(tokens, index + 1, match, features)
            index = match + 1
            continue
        if token == "}":
            return

        if upper == "FILTER":
            features.has_filter = True
            if index + 1 < end and tokens[index + 1] == "(":
                index = _skip_parenthesized(tokens, index + 1)
            elif index + 1 < end and tokens[index + 1] == "NOT":
                index = _skip_until_statement_end(tokens, index + 1)
            else:
                index = _skip_until_statement_end(tokens, index + 1)
            continue

        if upper in {"OPTIONAL", "MINUS", "GRAPH", "SERVICE"}:
            if upper == "OPTIONAL":
                features.has_optional = True
            elif upper == "MINUS":
                features.has_minus = True
            else:
                features.unsupported_constructs.append(upper)
            index += 1
            if index < end and tokens[index] == "{":
                match = matching_token(list(tokens), index, "{", "}")
                if match == -1:
                    features.parser_errors.append(f"unmatched {upper} group")
                    return
                _parse_group(tokens, index + 1, match, features)
                index = match + 1
            else:
                index = _skip_until_statement_end(tokens, index)
            continue

        if upper == "VALUES":
            features.has_values = True
            features.unsupported_constructs.append("VALUES")
            index = _skip_until_statement_end(tokens, index + 1)
            continue

        if upper in {"BIND", "EXISTS", "NOT"}:
            features.unsupported_constructs.append(upper)
            index = _skip_until_statement_end(tokens, index + 1)
            continue

        if index + 2 >= end:
            features.parser_errors.append(f"incomplete triple near token {token!r}")
            return

        subject, predicate = tokens[index : index + 2]
        object_, next_index = _consume_object(tokens, index + 2, end)
        if not object_:
            features.parser_errors.append(f"incomplete triple near token {token!r}")
            return
        if subject in {";", ",", "(", ")"} or predicate in {";", ",", "(", ")"}:
            features.parser_errors.append(f"invalid triple near token {token!r}")
            index = _skip_until_statement_end(tokens, index + 1)
            continue
        features.triple_patterns.append(TriplePattern(subject, predicate, object_))
        index = next_index

        # Expand compact predicate/object lists such as ?x p a ; q b .
        while index < end:
            separator = tokens[index]
            if separator == ".":
                index += 1
                break
            if separator == ";":
                index += 1
                if index < end and tokens[index] == ".":
                    index += 1
                    break
                if index + 1 >= end:
                    features.parser_errors.append("incomplete semicolon triple")
                    break
                predicate = tokens[index]
                object_, next_index = _consume_object(tokens, index + 1, end)
                if not object_:
                    features.parser_errors.append("incomplete semicolon triple")
                    break
                features.triple_patterns.append(TriplePattern(subject, predicate, object_))
                index = next_index
                continue
            if separator == ",":
                index += 1
                if index >= end:
                    features.parser_errors.append("incomplete comma triple")
                    break
                object_, next_index = _consume_object(tokens, index, end)
                if not object_:
                    features.parser_errors.append("incomplete comma triple")
                    break
                features.triple_patterns.append(TriplePattern(subject, predicate, object_))
                index = next_index
                continue
            if separator == "}":
                break
            # CWQ normally terminates triples with '.', but avoid silently
            # consuming an unexpected construct as a predicate/object.
            features.parser_errors.append(f"expected triple separator, found {separator!r}")
            index = _skip_until_statement_end(tokens, index)
            break


def _extract_filter_texts(tokens: Sequence[str]) -> List[str]:
    filters: List[str] = []
    index = 0
    while index < len(tokens):
        if _upper(tokens[index]) != "FILTER":
            index += 1
            continue
        start = index + 1
        if start < len(tokens) and tokens[start] == "(":
            end = matching_token(list(tokens), start, "(", ")")
            if end == -1:
                filters.append(" ".join(tokens[start:]))
                break
            filters.append(" ".join(tokens[start : end + 1]))
            index = end + 1
        else:
            end = _skip_until_statement_end(tokens, start)
            filters.append(" ".join(tokens[start:end]))
            index = end
    return filters


def _where_bounds(tokens: Sequence[str]) -> tuple[int, int] | None:
    where_index = next((i for i, token in enumerate(tokens) if _upper(token) == "WHERE"), -1)
    start = next((i for i in range(where_index + 1, len(tokens)) if tokens[i] == "{"), -1)
    if start == -1:
        start = next((i for i, token in enumerate(tokens) if token == "{"), -1)
    if start == -1:
        return None
    end = matching_token(list(tokens), start, "{", "}")
    return None if end == -1 else (start, end)


def parse_query(sparql: str) -> QueryFeatures:
    features = QueryFeatures()
    tokens = tokenize_sparql(sparql or "")
    where = _where_bounds(tokens)

    if where is None:
        features.parser_errors.append("WHERE group not found or unmatched")
        return features

    where_start, where_end = where
    before_where = " ".join(tokens[:where_start])
    features.select_variables = sorted(set(re.findall(r"\?[A-Za-z_][A-Za-z0-9_]*", before_where)))
    features.filters = _extract_filter_texts(tokens[where_start : where_end + 1])
    features.has_filter = bool(features.filters)

    features.has_union = any(_upper(token) == "UNION" for token in tokens[where_start : where_end + 1])
    features.has_order_by = bool(re.search(r"\bORDER\s+BY\b", sparql, re.IGNORECASE))
    features.has_limit = bool(re.search(r"\bLIMIT\s+\d+", sparql, re.IGNORECASE))
    features.has_aggregation = bool(
        re.search(r"\b(COUNT|SUM|AVG|MIN|MAX|SAMPLE|GROUP_CONCAT)\s*\(", sparql, re.IGNORECASE)
    )
    features.has_group_by = bool(re.search(r"\bGROUP\s+BY\b", sparql, re.IGNORECASE))
    features.has_having = bool(re.search(r"\bHAVING\b", sparql, re.IGNORECASE))
    features.has_optional = bool(re.search(r"\bOPTIONAL\b", sparql, re.IGNORECASE))
    features.has_minus = bool(re.search(r"\bMINUS\b", sparql, re.IGNORECASE))
    features.has_values = bool(re.search(r"\bVALUES\b", sparql, re.IGNORECASE))

    limit_match = re.search(r"\bLIMIT\s+(\d+)", sparql, re.IGNORECASE)
    if limit_match:
        features.limit = int(limit_match.group(1))
    order_match = re.search(
        r"\bORDER\s+BY\s+(.*?)(?:\bLIMIT\s+\d+|$)", sparql, re.IGNORECASE | re.DOTALL
    )
    if order_match:
        features.order_expressions = [order_match.group(1).strip()]

    comparison_text = " ".join(features.filters)
    # ``?x != ?c`` and ``?x != ns:m...`` are generic anti-cycle filters in
    # CWQ. Language guards also contain equality. Comparative questions use
    # ordered operators, so only those operators count here. The comparison
    # is detected inside FILTER text, where angle brackets are not IRI syntax.
    features.has_comparison = bool(
        re.search(r"(?<![<>])(?:>=|<=|>|<)(?![>])", comparison_text)
    )

    _parse_group(tokens, where_start + 1, where_end, features)
    features.parse_success = bool(features.triple_patterns) and not features.parser_errors

    if features.has_optional:
        features.unsupported_constructs.append("OPTIONAL")
    if features.has_minus:
        features.unsupported_constructs.append("MINUS")
    if features.has_values:
        features.unsupported_constructs.append("VALUES")
    if features.has_group_by:
        features.unsupported_constructs.append("GROUP BY")
    if features.has_having:
        features.unsupported_constructs.append("HAVING")
    features.unsupported_constructs = sorted(set(features.unsupported_constructs))
    return features
