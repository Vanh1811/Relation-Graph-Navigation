"""Deterministic local backend for the KQA Pro knowledge base.

The backend intentionally exposes only dataset access and exact KoPL data
semantics.  It does not choose a reasoning path, call an LLM, or emulate an
unsupported operator.  Phase 6 uses this adapter to keep data compatibility
separate from the frozen v1.1.3 and Phase 2.5.6 reasoning policies.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote, unquote


class UnsupportedKQAOperator(ValueError):
    """Raised when a program asks the local adapter for unsupported semantics."""


class KQAProBackend:
    """Execute the supported deterministic subset of KQA Pro's local KB."""

    name = "kqa_pro"
    default_entity_prefix: Optional[str] = "Q"

    # This is the backend subset used by Phase 6.  The complete KoPL language
    # is deliberately not silently approximated here.
    SUPPORTED_OPERATORS = {
        "Find",
        "Relate",
        "FilterConcept",
        "FilterStr",
        "What",
        "QueryName",
        "QueryRelation",
        "QueryAttr",
        "And",
    }

    def __init__(self, kb_path: str | Path):
        self.kb_path = Path(kb_path)
        with self.kb_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or not isinstance(payload.get("entities"), dict):
            raise ValueError("KQA Pro KB must contain an entities object")
        self.concepts: Dict[str, Mapping[str, Any]] = dict(payload.get("concepts", {}))
        self.entities: Dict[str, Mapping[str, Any]] = dict(payload["entities"])
        self._name_index: Dict[str, List[str]] = defaultdict(list)
        self._concept_name_index: Dict[str, List[str]] = defaultdict(list)
        self._relation_index: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self._relation_label_index: Dict[str, set[str]] = defaultdict(set)
        self._rdf_triples: Optional[List[Tuple[str, str, str]]] = None
        self._rdf_by_subject: Dict[str, List[Tuple[str, str, str]]] = {}
        self._rdf_by_object: Dict[str, List[Tuple[str, str, str]]] = {}
        self._rdf_by_predicate: Dict[str, List[Tuple[str, str, str]]] = {}
        self._build_indices()

    @staticmethod
    def _norm(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().casefold())

    @staticmethod
    def _dedupe(values: Iterable[str]) -> List[str]:
        return list(dict.fromkeys(str(value) for value in values if str(value).strip()))

    def _build_indices(self) -> None:
        for entity_id, entity in self.entities.items():
            name = str(entity.get("name", "")).strip()
            if name:
                self._name_index[self._norm(name)].append(entity_id)
            for relation in entity.get("relations", []) or []:
                predicate = str(relation.get("predicate", "")).strip()
                object_id = str(relation.get("object", "")).strip()
                direction = str(relation.get("direction", "forward")).strip().casefold()
                if predicate and object_id and direction in {"forward", "backward"}:
                    self._relation_index[(entity_id, predicate, direction)].append(object_id)
                    self._relation_label_index[self._norm(predicate)].add(predicate)
            for concept_id in entity.get("instanceOf", []) or []:
                self._relation_index[(entity_id, "instance_of", "forward")].append(str(concept_id))

        for concept_id, concept in self.concepts.items():
            name = str(concept.get("name", "")).strip()
            if name:
                self._name_index[self._norm(name)].append(str(concept_id))
                self._concept_name_index[self._norm(name)].append(str(concept_id))

        for index in (self._name_index, self._concept_name_index, self._relation_index):
            for key, values in index.items():
                index[key] = self._dedupe(values)

    # ------------------------------------------------------------------
    # Backend protocol and identity helpers

    def build_sparql_relations(self, query: str) -> str:
        return (
            "PREFIX kqa: <urn:kqa:>\n"
            "SELECT DISTINCT ?relation\n"
            "WHERE {\n"
            f"  {query}\n"
            "} LIMIT 1000"
        )

    def build_sparql_entities(self, query: str) -> str:
        return (
            "PREFIX kqa: <urn:kqa:>\n"
            "SELECT DISTINCT ?targetEntity\n"
            "WHERE {\n"
            f"  {query}\n"
            "} LIMIT 1000"
        )

    def build_sparql_constraints(self, query: str) -> str:
        return (
            "PREFIX kqa: <urn:kqa:>\n"
            "SELECT DISTINCT ?constraintTarget\n"
            "WHERE {\n"
            f"  {query}\n"
            "} LIMIT 1000"
        )

    def build_sparql_constraint_pairs(self, query: str) -> str:
        return (
            "PREFIX kqa: <urn:kqa:>\n"
            "SELECT DISTINCT ?constraintTarget ?constraintRelation\n"
            "WHERE {\n"
            f"  {query}\n"
            "} LIMIT 1000"
        )

    def normalize_uri(self, value: str) -> str:
        text = str(value or "").strip()
        for prefix in ("kqa:", "wd:", "http://www.wikidata.org/entity/"):
            if text.startswith(prefix):
                return text[len(prefix):]
        if text.startswith("<") and text.endswith(">"):
            return self.normalize_uri(text[1:-1])
        return text

    def is_entity_id(self, value: str) -> bool:
        return self.normalize_uri(value).startswith("Q")

    def format_entity_node(self, entity_id: str) -> str:
        return f"kqa:{self.normalize_uri(entity_id)}"

    def format_relation_node(self, relation_id: str) -> str:
        relation = str(relation_id or "").strip()
        return f"<urn:kqa:rel:{quote(relation, safe='._-')}>"

    def relation_id2label(self, relation_id: str) -> str:
        return str(relation_id or "").strip()

    def relation_ids2labels(self, relation_ids: List[str]) -> Dict[str, str]:
        return {str(relation_id): self.relation_id2label(relation_id) for relation_id in relation_ids}

    def is_unnecessary_relation(self, relation: str) -> bool:
        return str(relation or "").strip() in {"instance_of", "pred:name"}

    def id2entity_name_or_type(self, entity_id: str) -> str:
        entity_id = self.normalize_uri(entity_id)
        if entity_id in self.entities:
            return str(self.entities[entity_id].get("name", "UnName_Entity")) or "UnName_Entity"
        if entity_id in self.concepts:
            return str(self.concepts[entity_id].get("name", "UnName_Entity")) or "UnName_Entity"
        return "UnName_Entity"

    def id2entity_names(self, entity_ids: List[str]) -> Dict[str, str]:
        return {str(entity_id): self.id2entity_name_or_type(entity_id) for entity_id in entity_ids}

    def lookup_entities(self, name: str) -> List[str]:
        """Return all exact case-insensitive name matches in stable order."""
        return sorted(self._dedupe(self._name_index.get(self._norm(name), [])))

    def lookup_concepts(self, name: str) -> List[str]:
        return sorted(self._dedupe(self._concept_name_index.get(self._norm(name), [])))

    # ------------------------------------------------------------------
    # Exact local graph semantics

    def relation_candidates(self, entity_id: str, direction: str = "forward") -> List[str]:
        entity_id = self.normalize_uri(entity_id)
        direction = str(direction or "forward").casefold()
        graph = self._all_rdf_triples()
        if direction == "forward":
            return sorted({predicate for subject, predicate, _object_id in graph if subject == entity_id})
        if direction == "backward":
            return sorted({predicate for _subject, predicate, object_id in graph if object_id == entity_id})
        raise ValueError(f"Unsupported KQA relation direction: {direction}")

    def traverse(self, entity_ids: Iterable[str], predicate: str, direction: str = "forward") -> List[str]:
        direction = str(direction or "forward").casefold()
        if direction not in {"forward", "backward"}:
            raise ValueError(f"Unsupported KQA relation direction: {direction}")
        targets = {self.normalize_uri(entity_id) for entity_id in entity_ids}
        graph = self._all_rdf_triples()
        if direction == "forward":
            values = [object_id for subject, relation, object_id in graph if subject in targets and relation == str(predicate).strip()]
        else:
            values = [subject for subject, relation, object_id in graph if object_id in targets and relation == str(predicate).strip()]
        return sorted(set(values))

    def relation_between(self, source_ids: Iterable[str], target_ids: Iterable[str]) -> List[str]:
        sources = {self.normalize_uri(value) for value in source_ids}
        targets = {self.normalize_uri(value) for value in target_ids}
        return sorted({
            predicate
            for subject, predicate, object_id in self._all_rdf_triples()
            if subject in sources and object_id in targets
        })

    def _all_rdf_triples(self) -> List[Tuple[str, str, str]]:
        if self._rdf_triples is not None:
            return self._rdf_triples
        triples: List[Tuple[str, str, str]] = []
        for entity_id, entity in self.entities.items():
            for relation in entity.get("relations", []) or []:
                predicate = str(relation.get("predicate", "")).strip()
                object_id = str(relation.get("object", "")).strip()
                direction = str(relation.get("direction", "forward")).strip().casefold()
                if not predicate or not object_id:
                    continue
                if direction == "forward":
                    triples.append((entity_id, predicate, object_id))
                elif direction == "backward":
                    triples.append((object_id, predicate, entity_id))
            for concept_id in entity.get("instanceOf", []) or []:
                triples.append((entity_id, "instance_of", str(concept_id)))
        self._rdf_triples = sorted(set(triples))
        by_subject: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        by_object: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        by_predicate: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        for triple in self._rdf_triples:
            subject, predicate, object_id = triple
            by_subject[subject].append(triple)
            by_object[object_id].append(triple)
            by_predicate[predicate].append(triple)
        self._rdf_by_subject = dict(by_subject)
        self._rdf_by_object = dict(by_object)
        self._rdf_by_predicate = dict(by_predicate)
        return self._rdf_triples

    def is_instance_of(self, entity_id: str, concept_name: str) -> bool:
        target_ids = set(self.lookup_concepts(concept_name))
        if not target_ids:
            return False
        pending = list(self.entities.get(self.normalize_uri(entity_id), {}).get("instanceOf", []) or [])
        seen: set[str] = set()
        while pending:
            concept_id = str(pending.pop())
            if concept_id in seen:
                continue
            seen.add(concept_id)
            if concept_id in target_ids:
                return True
            pending.extend(self.concepts.get(concept_id, {}).get("instanceOf", []) or [])
        return False

    @staticmethod
    def _value_text(value: Any) -> str:
        if isinstance(value, dict):
            raw = value.get("value", "")
            text = str(raw).strip()
            if value.get("type") == "date":
                match = re.fullmatch(r"(\d{4})/(\d{1,2})/(\d{1,2})", text)
                if match:
                    return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
            return text
        return str(value or "").strip()

    def attributes(self, entity_ids: Iterable[str], key: str) -> List[Dict[str, Any]]:
        wanted = self._norm(key)
        result: List[Dict[str, Any]] = []
        for entity_id in entity_ids:
            entity = self.entities.get(self.normalize_uri(entity_id), {})
            for attribute in entity.get("attributes", []) or []:
                if self._norm(attribute.get("key")) == wanted:
                    result.append(dict(attribute))
        return result

    def filter_string(self, entity_ids: Iterable[str], key: str, expected: str) -> List[str]:
        expected_norm = self._norm(expected)
        return sorted({
            self.normalize_uri(entity_id)
            for entity_id in entity_ids
            if any(self._norm(self._value_text(item.get("value"))) == expected_norm for item in self.attributes([entity_id], key))
        })

    # ------------------------------------------------------------------
    # Deterministic supported KoPL execution

    def execute_program(self, program: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        functions = [str(step.get("function", "")) for step in program]
        unsupported = sorted(set(functions) - self.SUPPORTED_OPERATORS)
        if unsupported:
            raise UnsupportedKQAOperator(
                f"KQA Pro backend intentionally does not approximate: {', '.join(unsupported)}"
            )
        states: Dict[int, List[Any]] = {}
        for index, step in enumerate(program):
            function = str(step.get("function", ""))
            deps = [int(dep) for dep in step.get("dependencies", [])]
            inputs = list(step.get("inputs", []) or [])
            if function == "Find":
                states[index] = self.lookup_entities(str(inputs[0]) if inputs else "")
            elif function == "Relate":
                if len(deps) != 1 or len(inputs) < 2:
                    raise ValueError("Relate requires one dependency and predicate/direction inputs")
                states[index] = self.traverse(states[deps[0]], str(inputs[0]), str(inputs[1]))
            elif function == "FilterConcept":
                if len(deps) != 1 or not inputs:
                    raise ValueError("FilterConcept requires one dependency and a concept")
                states[index] = [
                    entity_id for entity_id in states[deps[0]] if self.is_instance_of(str(entity_id), str(inputs[0]))
                ]
            elif function == "FilterStr":
                if len(deps) != 1 or len(inputs) < 2:
                    raise ValueError("FilterStr requires one dependency and key/value inputs")
                states[index] = self.filter_string(states[deps[0]], str(inputs[0]), str(inputs[1]))
            elif function == "And":
                if len(deps) < 2:
                    raise ValueError("And requires at least two dependencies")
                common = set(states[deps[0]])
                for dep in deps[1:]:
                    common.intersection_update(states[dep])
                states[index] = sorted(common)
            elif function in {"What", "QueryName"}:
                states[index] = [self.id2entity_name_or_type(value) for value in states[deps[0]]]
            elif function == "QueryAttr":
                if len(deps) != 1 or not inputs:
                    raise ValueError("QueryAttr requires one dependency and an attribute key")
                values = [self._value_text(item.get("value")) for item in self.attributes(states[deps[0]], str(inputs[0]))]
                states[index] = self._dedupe(values)
            elif function == "QueryRelation":
                if len(deps) != 2:
                    raise ValueError("QueryRelation requires two dependencies")
                states[index] = self.relation_between(states[deps[0]], states[deps[1]])
        final = states.get(len(program) - 1, [])
        return {"values": final, "states": states, "final_function": functions[-1] if functions else ""}

    # ------------------------------------------------------------------
    # Compatibility methods used by the existing generic search helpers.

    def parse_relations(self, relations: List[Dict[str, Any]]) -> List[str]:
        return [self.normalize_uri(row.get("relation", {}).get("value", "")) for row in relations if row.get("relation")]

    def parse_entities(self, entities: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        literals: List[str] = []
        entity_ids: List[str] = []
        for row in entities:
            binding = row.get("targetEntity", {})
            value = str(binding.get("value", "")).strip()
            if not value:
                continue
            if binding.get("type") in {"literal", "typed-literal"}:
                literals.append(value)
            elif binding.get("type") == "uri":
                entity_ids.append(self.normalize_uri(value))
        return literals, entity_ids

    def execute_sparql(self, sparql_txt: str) -> List[Dict[str, Any]]:
        return self.execute_sparql_with_meta(sparql_txt).get("rows", [])

    def execute_sparql_with_meta(self, sparql_txt: str) -> Dict[str, Any]:
        """Execute the restricted query shapes emitted by generic search tools.

        This compatibility path is deliberately small; Phase 6's authoritative
        execution uses ``execute_program``.  Complex SPARQL is rejected rather
        than approximated.
        """
        body_match = re.search(r"WHERE\s*\{(.*)\}\s*LIMIT", sparql_txt, re.I | re.S)
        if not body_match:
            return {"rows": [], "status": "unsupported_query", "timed_out": False, "error": "query shape"}
        body = body_match.group(1)
        triples = []
        for raw in body.splitlines():
            line = raw.strip()
            if not line or not line.endswith(".") or line.startswith("FILTER"):
                continue
            bits = line[:-1].split()
            if len(bits) == 3:
                triples.append(tuple(bits))
        if not triples:
            return {"rows": [], "status": "unsupported_query", "timed_out": False, "error": "no triples"}

        def token_value(token: str, binding: Mapping[str, str]) -> str:
            if token.startswith("?"):
                return binding.get(token, token)
            token = token.strip("<>")
            if token.startswith("urn:kqa:rel:"):
                return unquote(token[len("urn:kqa:rel:"):])
            if token.startswith("kqa:"):
                return token[4:]
            return token

        def unify(pattern: str, value: str, binding: Dict[str, str]) -> Optional[Dict[str, str]]:
            if pattern.startswith("?"):
                previous = binding.get(pattern)
                if previous is not None and previous != value:
                    return None
                result = dict(binding)
                result[pattern] = value
                return result
            return dict(binding) if token_value(pattern, binding) == value else None

        bindings: List[Dict[str, str]] = [{}]
        graph = self._all_rdf_triples()
        for subject, predicate, object_value in triples:
            next_bindings: List[Dict[str, str]] = []
            for binding in bindings:
                bound_subject = token_value(subject, binding) if not subject.startswith("?") or subject in binding else None
                bound_predicate = token_value(predicate, binding) if not predicate.startswith("?") or predicate in binding else None
                bound_object = token_value(object_value, binding) if not object_value.startswith("?") or object_value in binding else None
                if bound_subject is not None:
                    candidates = self._rdf_by_subject.get(bound_subject, [])
                elif bound_object is not None:
                    candidates = self._rdf_by_object.get(bound_object, [])
                elif bound_predicate is not None:
                    candidates = self._rdf_by_predicate.get(bound_predicate, [])
                else:
                    candidates = graph
                for graph_subject, graph_predicate, graph_object in candidates:
                    current = unify(subject, graph_subject, binding)
                    if current is None:
                        continue
                    current = unify(predicate, graph_predicate, current)
                    if current is None:
                        continue
                    current = unify(object_value, graph_object, current)
                    if current is not None:
                        next_bindings.append(current)
            bindings = next_bindings
            if not bindings:
                break

        if "?relation" in body:
            values = sorted({binding.get("?relation", "") for binding in bindings if binding.get("?relation")})
            rows = [{"relation": {"type": "uri", "value": value}} for value in values]
            return {"rows": rows, "status": "ok", "timed_out": False, "error": ""}
        if "?targetEntity" in body:
            values = sorted({binding.get("?targetEntity", "") for binding in bindings if binding.get("?targetEntity")})
            rows = []
            for value in values:
                if self.is_entity_id(value):
                    rows.append({"targetEntity": {"type": "uri", "value": value}})
                else:
                    rows.append({"targetEntity": {"type": "literal", "value": value}})
            return {"rows": rows, "status": "ok", "timed_out": False, "error": ""}
        return {"rows": [], "status": "unsupported_query", "timed_out": False, "error": "unsupported local query"}


__all__ = ["KQAProBackend", "UnsupportedKQAOperator"]
