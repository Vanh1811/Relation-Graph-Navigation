"""Graph construction and first-pass structural measurements."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Set, Tuple

from relation_graph_kgqa.datasets.schemas import QAExample

from .parser import QueryFeatures, TriplePattern


def normalize_term(term: str) -> str:
    value = str(term).strip()
    if value.startswith("<") and value.endswith(">"): 
        value = value[1:-1]
    if value.startswith("ns:"):
        value = value[3:]
    return value


def is_variable(term: str) -> bool:
    return str(term).startswith("?")


@dataclass(frozen=True)
class GraphEdge:
    subject: str
    predicate: str
    object: str


@dataclass
class QueryGraph:
    nodes: Set[str] = field(default_factory=set)
    variables: Set[str] = field(default_factory=set)
    constants: Set[str] = field(default_factory=set)
    edges: List[GraphEdge] = field(default_factory=list)
    answer_variables: List[str] = field(default_factory=list)
    topic_terms: Set[str] = field(default_factory=set)

    def adjacency(self) -> Dict[str, List[GraphEdge]]:
        result: Dict[str, List[GraphEdge]] = defaultdict(list)
        for edge in self.edges:
            result[edge.subject].append(edge)
            result[edge.object].append(edge)
        return result

    def outgoing(self) -> Dict[str, List[GraphEdge]]:
        result: Dict[str, List[GraphEdge]] = defaultdict(list)
        for edge in self.edges:
            result[edge.subject].append(edge)
        return result

    def incoming(self) -> Dict[str, List[GraphEdge]]:
        result: Dict[str, List[GraphEdge]] = defaultdict(list)
        for edge in self.edges:
            result[edge.object].append(edge)
        return result

    def connected_components(self) -> List[Set[str]]:
        adjacency = self.adjacency()
        unseen = set(self.nodes)
        components: List[Set[str]] = []
        while unseen:
            root = min(unseen)
            queue = deque([root])
            component: Set[str] = set()
            while queue:
                node = queue.popleft()
                if node in component:
                    continue
                component.add(node)
                unseen.discard(node)
                for edge in adjacency.get(node, []):
                    other = edge.object if edge.subject == node else edge.subject
                    if other not in component:
                        queue.append(other)
            components.append(component)
        return components

    def distances_from(self, sources: Iterable[str]) -> Dict[str, int]:
        adjacency = self.adjacency()
        distances: Dict[str, int] = {}
        queue = deque()
        for source in sources:
            if source in self.nodes and source not in distances:
                distances[source] = 0
                queue.append(source)
        while queue:
            node = queue.popleft()
            for edge in adjacency.get(node, []):
                other = edge.object if edge.subject == node else edge.subject
                if other not in distances:
                    distances[other] = distances[node] + 1
                    queue.append(other)
        return distances

    def branch_candidates(self) -> Dict[str, int]:
        """Return variables whose local graph has at least two logical arms.

        The metric is intentionally conservative: an ordinary path has one arm
        after its parent edge is accounted for, while a Y-shaped graph has two.
        Topic entities remain in the topology because a second mentioned entity
        can itself be a required branch constraint.
        """

        adjacency = self.adjacency()
        outgoing = self.outgoing()
        incoming = self.incoming()
        candidates: Dict[str, int] = {}

        for variable in sorted(self.variables):
            neighbors = []
            for edge in adjacency.get(variable, []):
                other = edge.object if edge.subject == variable else edge.subject
                neighbors.append(other)
            if not neighbors:
                continue

            outbound = sum(
                1
                for edge in outgoing.get(variable, [])
                if edge.object != variable
            )
            inbound = sum(
                1
                for edge in incoming.get(variable, [])
                if edge.subject != variable
            )
            arms = max(len(neighbors) - 1, outbound, inbound)
            if arms >= 2:
                candidates[variable] = arms
        return candidates

    def has_nested_branch(self) -> bool:
        candidates = set(self.branch_candidates())
        if len(candidates) < 2:
            return False
        adjacency = self.adjacency()
        for start in candidates:
            queue = deque([(start, 0)])
            seen = {start}
            while queue:
                node, distance = queue.popleft()
                for edge in adjacency.get(node, []):
                    other = edge.object if edge.subject == node else edge.subject
                    if other in self.topic_terms or other in seen:
                        continue
                    if other in candidates and distance + 1 >= 2:
                        return True
                    seen.add(other)
                    queue.append((other, distance + 1))
        return False

    def max_reasoning_depth(self) -> int:
        if not self.nodes:
            return 0
        sources = self.topic_terms or self.constants
        distances = self.distances_from(sources)
        answer_distances = [distances[var] for var in self.answer_variables if var in distances]
        if answer_distances:
            return max(answer_distances)
        return max(distances.values(), default=0)

    def ascii_view(self, max_nodes: int = 80) -> str:
        if not self.nodes:
            return "(empty graph)"
        adjacency = self.adjacency()
        roots = list(self.topic_terms) + list(self.answer_variables)
        roots.extend(sorted(self.nodes))
        visited: Set[str] = set()
        lines: List[str] = []

        def walk(node: str, prefix: str, is_last: bool) -> None:
            if len(visited) >= max_nodes:
                return
            connector = "└── " if is_last else "├── "
            if node not in visited:
                lines.append(f"{prefix}{connector}{node}")
                visited.add(node)
            else:
                return
            neighbors: List[Tuple[str, str]] = []
            for edge in adjacency.get(node, []):
                other = edge.object if edge.subject == node else edge.subject
                relation = normalize_term(edge.predicate)
                neighbors.append((other, relation))
            neighbors.sort()
            for index, (other, relation) in enumerate(neighbors):
                if other in visited:
                    continue
                child_prefix = prefix + ("    " if is_last else "│   ")
                lines.append(f"{child_prefix}{'└── ' if index == len(neighbors) - 1 else '├── '}{relation} → {other}")
                walk(other, child_prefix, index == len(neighbors) - 1)

        root = next((item for item in roots if item in self.nodes), min(self.nodes))
        lines.append(root)
        visited.add(root)
        for edge in sorted(adjacency.get(root, []), key=lambda item: (item.object, item.predicate)):
            other = edge.object if edge.subject == root else edge.subject
            if other in visited:
                continue
            relation = normalize_term(edge.predicate)
            lines.append(f"├── {relation} → {other}")
            walk(other, "│   ", False)
        return "\n".join(lines)


def _topic_terms(example: QAExample) -> Set[str]:
    terms: Set[str] = set()
    for entity_id in example.topic_entities:
        normalized = normalize_term(entity_id)
        if normalized:
            terms.add(normalized)
            terms.add(f"m.{normalized}" if not normalized.startswith(("m.", "g.")) else normalized)
    return terms


def build_query_graph(features: QueryFeatures, example: QAExample) -> QueryGraph:
    graph = QueryGraph(answer_variables=list(features.select_variables))
    graph.topic_terms = _topic_terms(example)
    for pattern in features.triple_patterns:
        subject = normalize_term(pattern.subject)
        predicate = normalize_term(pattern.predicate)
        object_ = normalize_term(pattern.object)
        graph.nodes.update((subject, object_))
        graph.edges.append(GraphEdge(subject, predicate, object_))
        if is_variable(subject):
            graph.variables.add(subject)
        else:
            graph.constants.add(subject)
        if is_variable(object_):
            graph.variables.add(object_)
        else:
            graph.constants.add(object_)
    return graph
