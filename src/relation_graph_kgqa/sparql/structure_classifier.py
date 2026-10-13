"""Initial deterministic structural classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from relation_graph_kgqa.datasets.schemas import QAExample, StructuralMetadata

from .graph_builder import QueryGraph, build_query_graph
from .parser import QueryFeatures


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    reasons: List[str]


def _is_superlative(features: QueryFeatures) -> bool:
    if features.has_order_by and features.limit == 1:
        return True
    if features.has_aggregation and any(
        marker in " ".join(features.order_expressions).upper() for marker in ("MIN", "MAX")
    ):
        return True
    return False


def classify_structure(features: QueryFeatures, graph: QueryGraph) -> ClassificationResult:
    reasons: List[str] = []
    if not features.parse_success:
        reasons.append("SPARQL graph parsing did not produce a reliable triple graph")
        reasons.extend(features.parser_errors)
        return ClassificationResult("other_or_unknown", 0.0, reasons)

    if features.has_union:
        return ClassificationResult(
            "other_or_unknown",
            0.65,
            ["UNION is present and is outside the first-pass conjunction classifier"],
        )
    if features.unsupported_constructs:
        return ClassificationResult(
            "other_or_unknown",
            0.6,
            [f"unsupported construct: {item}" for item in features.unsupported_constructs],
        )
    if _is_superlative(features):
        return ClassificationResult(
            "superlative",
            0.9,
            ["ORDER BY with LIMIT 1 indicates a top/bottom-ranked answer"],
        )
    if features.has_comparison:
        return ClassificationResult(
            "comparative",
            0.88,
            ["a comparison operator occurs inside a FILTER expression"],
        )

    candidates = graph.branch_candidates()
    branch_count = max(candidates.values(), default=0)
    if graph.has_nested_branch():
        reasons.append(f"{len(candidates)} branch points are separated by a deeper graph path")
        return ClassificationResult("nested_branch", 0.78, reasons)
    if branch_count >= 3:
        reasons.append(f"a shared variable has {branch_count} independent graph arms")
        return ClassificationResult("branch_3_plus", 0.82, reasons)
    if branch_count == 2:
        reasons.append("a shared variable has 2 independent graph arms")
        return ClassificationResult("branch_2", 0.84, reasons)

    reasons.append("relevant graph is a single path without a detected conjunction branch")
    return ClassificationResult("linear", 0.8, reasons)


def build_structural_metadata(
    features: QueryFeatures, graph: QueryGraph, result: ClassificationResult
) -> StructuralMetadata:
    branch_candidates = graph.branch_candidates()
    return StructuralMetadata(
        structure_type=result.label,
        branch_count=max(branch_candidates.values(), default=0),
        max_depth=graph.max_reasoning_depth(),
        variable_count=len(graph.variables),
        shared_variable_count=sum(
            1 for variable in graph.variables if len(graph.adjacency().get(variable, [])) >= 2
        ),
        has_comparison=features.has_comparison,
        has_superlative=_is_superlative(features),
        has_union=features.has_union,
        parse_success=features.parse_success,
        answer_variables=list(features.select_variables),
        triple_count=len(features.triple_patterns),
        connected_components=len(graph.connected_components()),
        has_filter=features.has_filter,
        has_aggregation=features.has_aggregation,
        has_order_by=features.has_order_by,
        has_limit=features.has_limit,
        nested_branch=graph.has_nested_branch(),
        classifier_confidence=result.confidence,
        classifier_reasons=list(result.reasons),
        parser_errors=list(features.parser_errors),
        unsupported_constructs=list(features.unsupported_constructs),
    )


def classify_example(example: QAExample) -> tuple[StructuralMetadata, QueryFeatures, QueryGraph]:
    if not example.gold_sparql:
        features = QueryFeatures(parser_errors=["gold SPARQL is missing"])
        graph = QueryGraph(answer_variables=[])
        result = ClassificationResult("other_or_unknown", 0.0, ["gold SPARQL is missing"])
        return build_structural_metadata(features, graph, result), features, graph

    from .parser import parse_query

    features = parse_query(example.gold_sparql)
    graph = build_query_graph(features, example)
    result = classify_structure(features, graph)
    return build_structural_metadata(features, graph, result), features, graph
