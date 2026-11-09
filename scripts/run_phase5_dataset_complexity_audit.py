#!/usr/bin/env python3
"""Audit the gold reasoning complexity of Phase 3C against a KQA Pro sample.

This is an offline audit. It does not call an LLM, Freebase, or any inference
runtime. The Phase 3C source is the frozen strong-case manifest plus the
original processed CWQ/GrailQA records. KQA Pro is read from a frozen local
validation file downloaded from the documented fallback mirror.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
PHASE3C_ROOT = ROOT / "outputs/experiments/phase3c/frozen_branch_required_generalization"
OUTPUT_ROOT = ROOT / "outputs/experiments/phase5/dataset_complexity_audit"
KQA_SOURCE = OUTPUT_ROOT / "source/kqa_pro_val.json"
PHASE3C_MANIFEST = PHASE3C_ROOT / "phase3c_eval_manifest.json"
SELECTION_SEED = 20261030
MANUAL_SEED = 20261031

KQA_SOURCE_URL = "https://huggingface.co/datasets/drt/kqa_pro/resolve/main/val.json?download=true"
KQA_OFFICIAL_REPO = "https://github.com/shijx12/KQAPro_Baselines"
KQA_PAPER = "https://aclanthology.org/2022.acl-long.422/"
KQA_OFFICIAL_DOWNLOAD = "https://cloud.tsinghua.edu.cn/f/04ce81541e704a648b03/?dl=1"

NORMALIZED_FLAGS = [
    "multi_hop",
    "conjunction",
    "disjunction",
    "nested_composition",
    "comparison",
    "superlative",
    "counting",
    "aggregation",
    "attribute_query",
    "qualifier_reasoning",
    "verification",
    "temporal_filter",
    "numeric_filter",
    "set_operation",
    "shared_intermediate",
    "requires_downstream_relation",
    "three_plus_constraints",
    "three_plus_branches",
]

KQA_FAMILIES = [
    "Count",
    "QueryAttr",
    "QueryAttrQualifier",
    "QueryName",
    "QueryRelation",
    "QueryRelationQualifier",
    "SelectAmong",
    "SelectBetween",
    "Verify",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path_from_manifest(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    return ROOT / path.relative_to(ROOT) if str(path).startswith(str(ROOT)) else ROOT / path.name


def load_source_records(manifest_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    wanted: Dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        wanted[str(source_path_from_manifest(str(row["source_path"])))].add(str(row["case_id"]))
    found: Dict[str, Mapping[str, Any]] = {}
    for path_text, ids in wanted.items():
        path = Path(path_text)
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record_id = str(record.get("id", ""))
                if record_id in ids:
                    found[record_id] = record
    missing = set().union(*wanted.values()) - set(found)
    if missing:
        raise RuntimeError(f"Could not resolve {len(missing)} frozen source records: {sorted(missing)[:5]}")
    return found


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def numeric_summary(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    clean = [float(value) for value in values if value is not None]
    return {
        "n": len(clean),
        "mean": mean(clean) if clean else None,
        "median": median(clean) if clean else None,
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
        "p25": percentile(clean, 0.25),
        "p75": percentile(clean, 0.75),
        "p90": percentile(clean, 0.90),
    }


def text(value: Any) -> str:
    return "" if value is None else str(value)


def unique_vars(form: str) -> List[str]:
    return sorted(set(re.findall(r"\?[A-Za-z_][A-Za-z0-9_]*", form)))


def sparql_triples(sparql: str) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    for raw in sparql.splitlines():
        line = raw.strip()
        if not line or line.upper().startswith(("PREFIX", "FILTER", "VALUES", "SELECT", "ORDER", "LIMIT", "GROUP", "HAVING")):
            continue
        if line in {"{", "}", "}", "} UNION {"} or line.startswith(("{", "}", "UNION")):
            continue
        if not line.endswith("."):
            continue
        body = line[:-1].strip()
        pieces = body.split(None, 2)
        if len(pieces) == 3 and pieces[1] not in {"AS"}:
            triples.append((pieces[0], pieces[1], pieces[2]))
    return triples


def predicate_is_type(predicate: str) -> bool:
    lowered = predicate.lower()
    return (
        "type.object.type" in lowered
        or "common.topic.notable_types" in lowered
        or predicate.endswith("rdf-syntax-ns#type>")
        or predicate.endswith("rdf:type")
    )


def filter_count(sparql: str) -> int:
    return len(re.findall(r"\bFILTER\s*\(", sparql, re.IGNORECASE))


def comparison_count(sparql: str) -> int:
    # Exclude the ubiquitous inequality boilerplate used to keep answer,
    # entity, and literal variables distinct. Treat a comparison as semantic
    # only when it uses an ordering operator or a numeric/date literal.
    count = 0
    for raw in re.findall(r"FILTER\s*\((.*?)\)", sparql, re.I | re.S):
        # URI references contain angle brackets and identifiers may contain
        # digits, so remove URI tokens before looking for ordering syntax or
        # numeric/date operands.
        cleaned = re.sub(r"<[^>]*>", "URI", raw)
        has_ordering = bool(re.search(r">=|<=|(?<![<>=])>(?!=)|(?<![<>=])<(?!=)", cleaned))
        has_numeric_or_date_operand = bool(re.search(r"(?:xsd:|\"?[-+]?\d+(?:\.\d+)?\"?|date|datetime|year)", cleaned, re.I))
        if has_ordering and has_numeric_or_date_operand:
            count += 1
    return count


def sparql_operator_counts(sparql: str, triples: Sequence[Tuple[str, str, str]], branch_count: int) -> Dict[str, int]:
    counts = Counter()
    counts["RELATION"] = sum(not predicate_is_type(predicate) for _, predicate, _ in triples)
    counts["TYPE"] = sum(predicate_is_type(predicate) for _, predicate, _ in triples)
    counts["FILTER"] = filter_count(sparql)
    counts["COMPARISON"] = comparison_count(sparql)
    counts["SUPERLATIVE"] = int(bool(re.search(r"ORDER\s+BY|LIMIT\s+1|ARGMAX|ARGMIN", sparql, re.I)))
    counts["AGGREGATION"] = len(re.findall(r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(", sparql, re.I))
    counts["QUALIFIER"] = int("[" in sparql and "]" in sparql) + len(re.findall(r"qualif", sparql, re.I))
    # Branch conjunction is reported separately. This family is reserved for
    # explicit union/disjunction/set-like SPARQL structure.
    counts["SET"] = len(re.findall(r"\bUNION\b|\bOR\b", sparql, re.I))
    counts["CONVERGENCE"] = int(branch_count > 1)
    return {key: int(value) for key, value in counts.items() if value}


def feature_flags_from_common(
    *,
    relation_count: int,
    logical_depth: int,
    branch_count: int,
    constraint_count: int,
    operator_counts: Mapping[str, int],
    qualifier: bool,
    comparison: bool,
    superlative: bool,
    counting: bool,
    aggregation: bool,
    attribute_query: bool,
    verification: bool,
    temporal: bool,
    numeric: bool,
    nested: bool,
    disjunction: bool,
    shared_intermediate: bool,
    downstream: bool,
) -> Dict[str, bool]:
    return {
        "multi_hop": bool(relation_count >= 2 or logical_depth >= 3),
        "conjunction": bool(branch_count >= 2 or operator_counts.get("CONVERGENCE", 0) > 0),
        "disjunction": bool(disjunction),
        "nested_composition": bool(nested),
        "comparison": bool(comparison),
        "superlative": bool(superlative),
        "counting": bool(counting),
        "aggregation": bool(aggregation),
        "attribute_query": bool(attribute_query),
        "qualifier_reasoning": bool(qualifier),
        "verification": bool(verification),
        "temporal_filter": bool(temporal),
        "numeric_filter": bool(numeric),
        "set_operation": bool(operator_counts.get("SET", 0) > 0 or disjunction),
        "shared_intermediate": bool(shared_intermediate),
        "requires_downstream_relation": bool(downstream),
        "three_plus_constraints": bool(constraint_count >= 3),
        "three_plus_branches": bool(branch_count >= 3),
    }


def phase3c_categories(flags: Mapping[str, bool], structure: str, relation_count: int) -> List[str]:
    labels: List[str] = []
    if relation_count >= 2:
        labels.append("LINEAR_MULTI_HOP")
    if flags["conjunction"]:
        labels.append("TWO_SOURCE_CONJUNCTION")
    if structure == "BRANCH_FINAL":
        labels.append("BRANCH_FINAL")
    if structure == "BRANCH_INTERMEDIATE":
        labels.append("BRANCH_INTERMEDIATE")
    if flags["nested_composition"]:
        labels.append("NESTED_COMPOSITION")
    if flags["comparison"]:
        labels.append("COMPARATIVE")
    if flags["superlative"]:
        labels.append("SUPERLATIVE")
    if flags["counting"]:
        labels.append("COUNT")
    if flags["aggregation"]:
        labels.append("AGGREGATION")
    if flags["qualifier_reasoning"]:
        labels.append("QUALIFIER")
    if flags["temporal_filter"]:
        labels.append("TEMPORAL")
    if flags["numeric_filter"]:
        labels.append("NUMERIC_FILTER")
    if flags["set_operation"]:
        labels.append("SET_OPERATION")
    if flags["three_plus_constraints"]:
        labels.append("MIXED_COMPLEX")
    return sorted(set(labels))


def phase3c_feature(manifest_row: Mapping[str, Any], source: Mapping[str, Any]) -> Dict[str, Any]:
    sparql = text(source.get("gold_sparql") or manifest_row.get("gold_sparql"))
    triples = sparql_triples(sparql)
    structure = text(manifest_row.get("gold_structure") or source.get("structure", {}).get("structure_type"))
    source_structure = source.get("structure") or {}
    branch_count = int(manifest_row.get("branch_count") or source_structure.get("branch_count") or 0)
    relation_count = sum(not predicate_is_type(predicate) for _, predicate, _ in triples)
    type_count = sum(predicate_is_type(predicate) for _, predicate, _ in triples)
    vars_ = unique_vars(sparql)
    logical_depth = int(source_structure.get("max_depth") or 0)
    if not logical_depth:
        logical_depth = max(1, sparql.count("{") - sparql.count("UNION"))
    operator_counts = sparql_operator_counts(sparql, triples, branch_count)
    comparison = bool(operator_counts.get("COMPARISON") or re.search(r"ORDER\s+BY", sparql, re.I))
    superlative = bool(operator_counts.get("SUPERLATIVE"))
    counting = bool(operator_counts.get("AGGREGATION") and re.search(r"COUNT\s*\(", sparql, re.I))
    aggregation = bool(operator_counts.get("AGGREGATION"))
    qualifier = bool(operator_counts.get("QUALIFIER"))
    temporal = bool(re.search(r"date|year|datetime|time", sparql, re.I))
    # Do not classify entity-inequality boilerplate (`?x != <entity>`) as a
    # numeric filter. Count typed numeric literals and ordering comparisons
    # whose operand contains a numeric literal/date-like value.
    numeric = bool(
        re.search(r"xsd:(?:int|integer|double|float|decimal)", sparql, re.I)
        or re.search(
            r"(?:>=|<=|>|<)\s*(?:\"?[-+]?\d+(?:\.\d+)?|\"[^\"]*(?:date|year|time)[^\"]*\")",
            sparql,
            re.I,
        )
        or re.search(
            r"(?:\"?[-+]?\d+(?:\.\d+)?|\"[^\"]*(?:date|year|time)[^\"]*\")\s*(?:>=|<=|>|<)",
            sparql,
            re.I,
        )
    )
    literal_object = any('"' in obj or "'" in obj for _, _, obj in triples)
    attribute_predicate = any(re.search(r"(?:\.name$|\.id$|\.date$|\.number$|\.year$|\.value$|\.key$)", predicate, re.I) for _, predicate, _ in triples)
    attribute_query = bool(literal_object or attribute_predicate)
    verification = bool(re.search(r"ASK|FILTER\s*\(", sparql, re.I) and len(triples) <= 2)
    nested = bool(source_structure.get("nested_branch") or logical_depth >= 4 or structure == "BRANCH_INTERMEDIATE")
    shared_intermediate = structure == "BRANCH_INTERMEDIATE"
    downstream = shared_intermediate or bool(source_structure.get("max_depth", 0) >= 3 and len(triples) >= 3)
    constraint_count = relation_count + type_count + filter_count(sparql) + int(comparison) + int(qualifier)
    flags = feature_flags_from_common(
        relation_count=relation_count,
        logical_depth=logical_depth,
        branch_count=branch_count,
        constraint_count=constraint_count,
        operator_counts=operator_counts,
        qualifier=qualifier,
        comparison=comparison,
        superlative=superlative,
        counting=counting,
        aggregation=aggregation,
        attribute_query=attribute_query,
        verification=verification,
        temporal=temporal,
        numeric=numeric,
        nested=nested,
        disjunction=bool(re.search(r"\bUNION\b|\bOR\b", sparql, re.I)),
        shared_intermediate=shared_intermediate,
        downstream=downstream,
    )
    operator_count = sum(operator_counts.values())
    return {
        "case_id": manifest_row["case_id"],
        "dataset": manifest_row["dataset"],
        "split": manifest_row.get("split", source.get("split")),
        "question": source.get("question", manifest_row.get("question", "NOT_AVAILABLE")),
        "gold_structure": structure,
        "branch_necessity_class": manifest_row.get("necessity", "NOT_AVAILABLE"),
        "gold_answer": source.get("answers", manifest_row.get("gold_answers", "NOT_AVAILABLE")),
        "gold_logical_form": source.get("logical_form") or manifest_row.get("logical_form") or "NOT_AVAILABLE",
        "gold_sparql": sparql or "NOT_AVAILABLE",
        "raw_source_path": manifest_row.get("source_path", "NOT_AVAILABLE"),
        "raw_feature_basis": "processed gold CWQ SPARQL or GrailQA gold S-expression/SPARQL; no model traces",
        "relation_count": relation_count,
        "entity_anchor_count": len(source.get("topic_entities") or manifest_row.get("topic_entities") or []),
        "variable_count": len(vars_),
        "logical_form_depth": logical_depth,
        "max_path_length": int(source_structure.get("max_depth") or logical_depth),
        "branch_count": branch_count,
        "branching_factor": branch_count,
        "post_merge_depth": 1 if shared_intermediate else 0,
        "constraint_count": constraint_count,
        "operator_counts": operator_counts,
        "operator_count": operator_count,
        "operator_families": sorted(operator_counts),
        "complexity_categories": phase3c_categories(flags, structure, relation_count),
        "complexity_flags": flags,
        "dependency_properties": {
            "direct_final_convergence": structure == "BRANCH_FINAL",
            "shared_intermediate": shared_intermediate,
            "requires_downstream_relation": downstream,
            "nested_dependency": nested,
            "multiple_independent_sources": branch_count >= 2,
        },
    }


def kqa_family(function: str) -> str:
    if function == "Count":
        return "Count"
    if function == "What":
        return "QueryName"
    if function == "QueryAttrUnderCondition":
        return "QueryAttr"
    if function == "QueryAttr":
        return "QueryAttr"
    if function == "QueryAttrQualifier":
        return "QueryAttrQualifier"
    if function == "QueryRelation":
        return "QueryRelation"
    if function == "QueryRelationQualifier":
        return "QueryRelationQualifier"
    if function == "SelectAmong":
        return "SelectAmong"
    if function == "SelectBetween":
        return "SelectBetween"
    if function.startswith("Verify"):
        return "Verify"
    raise ValueError(f"Unsupported KQA terminal function: {function}")


def kqa_depth(program: Sequence[Mapping[str, Any]]) -> int:
    depths: Dict[int, int] = {}
    for index, step in enumerate(program):
        deps = [int(dep) for dep in step.get("dependencies", [])]
        depths[index] = 1 + max((depths[dep] for dep in deps), default=0)
    return max(depths.values(), default=0)


def kqa_operator_family(function: str) -> str:
    if function in {"Find", "FindAll"}:
        return "ENTITY_SEARCH"
    if function == "Relate" or function.startswith("QueryRelation"):
        return "RELATION"
    if function.startswith("Filter") or function.startswith("QFilter"):
        return "FILTER"
    if function in {"And", "Or"}:
        return "SET"
    if function == "Count":
        return "AGGREGATION"
    if function.startswith("Select"):
        return "COMPARISON_SELECTION"
    if function.startswith("Verify"):
        return "VERIFICATION"
    if function.startswith("QueryAttr") or function == "What":
        return "ATTRIBUTE_QUERY"
    return function.upper()


def kqa_features(sample: Mapping[str, Any]) -> Dict[str, Any]:
    program = sample["program"]
    functions = [text(step.get("function")) for step in program]
    function_counts = Counter(functions)
    terminal = functions[-1]
    family = kqa_family(terminal)
    relation_ops = sum(function_counts[name] for name in {"Relate", "QueryRelation", "QueryRelationQualifier"})
    qualifier = any("Qualifier" in name or name.startswith("QFilter") for name in functions)
    comparison = any(name in {"FilterNum", "FilterYear", "FilterDate", "QFilterNum", "QFilterYear", "QFilterDate", "SelectAmong", "SelectBetween", "VerifyNum", "VerifyYear", "VerifyDate"} for name in functions)
    superlative = any(name in {"SelectAmong", "SelectBetween"} for name in functions)
    counting = "Count" in functions
    aggregation = counting
    attribute = any(name.startswith("QueryAttr") or name == "What" for name in functions)
    verification = any(name.startswith("Verify") for name in functions)
    temporal = any(name in {"FilterYear", "FilterDate", "QFilterYear", "QFilterDate", "VerifyYear", "VerifyDate"} for name in functions)
    numeric = any(name in {"FilterNum", "QFilterNum", "VerifyNum"} for name in functions)
    disjunction = "Or" in functions
    set_operation = any(name in {"And", "Or"} for name in functions)
    independent_sources = sum(not step.get("dependencies") and step.get("function") in {"Find", "FindAll"} for step in program)
    branch_count = max(independent_sources, 1)
    and_indices = {index for index, step in enumerate(program) if step.get("function") in {"And", "Or"}}
    downstream = any(
        step.get("function") in {"Relate", "QueryRelation", "QueryRelationQualifier"}
        and any(int(dep) in and_indices for dep in step.get("dependencies", []))
        for step in program
    )
    nested = kqa_depth(program) >= 4 or any(len(step.get("dependencies", [])) >= 2 for step in program)
    constraint_count = (
        function_counts["Find"]
        + function_counts["FilterConcept"]
        + function_counts["Relate"]
        + sum(function_counts[name] for name in {"QueryRelation", "QueryRelationQualifier"})
        + sum(function_counts[name] for name in {"FilterStr", "FilterNum", "FilterYear", "FilterDate", "QFilterStr", "QFilterNum", "QFilterYear", "QFilterDate"})
        + sum(function_counts[name] for name in {"QueryAttr", "QueryAttrQualifier", "QueryAttrUnderCondition"})
        + int(qualifier)
        + int(comparison)
        + int(aggregation)
        + int(verification)
    )
    operator_families = sorted({kqa_operator_family(name) for name in functions})
    flags = feature_flags_from_common(
        relation_count=relation_ops,
        logical_depth=kqa_depth(program),
        branch_count=branch_count,
        constraint_count=constraint_count,
        operator_counts={family: 1 for family in operator_families},
        qualifier=qualifier,
        comparison=comparison,
        superlative=superlative,
        counting=counting,
        aggregation=aggregation,
        attribute_query=attribute,
        verification=verification,
        temporal=temporal,
        numeric=numeric,
        nested=nested,
        disjunction=disjunction,
        shared_intermediate=False,
        downstream=downstream,
    )
    flags["conjunction"] = "And" in functions
    flags["set_operation"] = set_operation
    return {
        "sample_index": sample["sample_index"],
        "question": sample["question"],
        "answer": sample.get("answer", "NOT_AVAILABLE"),
        "family": family,
        "actual_terminal_function": terminal,
        "program": program,
        "sparql": sample.get("sparql", "NOT_AVAILABLE"),
        "program_length": len(program),
        "program_depth": kqa_depth(program),
        "entity_anchor_count": sum(step.get("function") == "Find" for step in program),
        "variable_count_if_derivable": None,
        "relation_operation_count": relation_ops,
        "branching_factor_if_derivable": max((len(step.get("dependencies", [])) for step in program), default=0),
        "independent_source_count_if_possible": independent_sources,
        "constraint_count": constraint_count,
        "operator_counts": dict(sorted(function_counts.items())),
        "operator_count": len(program),
        "operator_families": operator_families,
        "normalized_operator_families": operator_families,
        "normalized_depth": kqa_depth(program),
        "complexity_categories": kqa_categories(flags, family),
        "complexity_flags": flags,
        "dependency_properties": {
            "multiple_independent_sources": independent_sources >= 2,
            "nested_dependency": nested,
            "requires_downstream_relation": downstream,
        },
    }


def kqa_categories(flags: Mapping[str, bool], family: str) -> List[str]:
    labels = [family]
    if flags["multi_hop"]:
        labels.append("LINEAR")
    if flags["conjunction"]:
        labels.append("CONJUNCTION")
    if flags["nested_composition"]:
        labels.append("NESTED")
    if flags["aggregation"]:
        labels.append("AGGREGATION")
    if flags["comparison"]:
        labels.append("COMPARATIVE")
    if flags["qualifier_reasoning"]:
        labels.append("QUALIFIER")
    if flags["verification"]:
        labels.append("VERIFY")
    if flags["set_operation"]:
        labels.append("SET_OPERATION")
    return sorted(set(labels))


def sample_kqa(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[str, List[Tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[kqa_family(text(row["program"][-1]["function"]))].append((index, row))
    missing = {family: len(grouped[family]) for family in KQA_FAMILIES if len(grouped[family]) < 20}
    if missing:
        raise RuntimeError(f"KQA family has fewer than 20 rows: {missing}")
    rng = random.Random(SELECTION_SEED)
    selected: List[Dict[str, Any]] = []
    selections: Dict[str, List[int]] = {}
    for family in KQA_FAMILIES:
        choices = sorted(grouped[family], key=lambda pair: pair[0])
        picked = sorted(rng.sample(choices, 20), key=lambda pair: pair[0])
        selections[family] = [index for index, _ in picked]
        for index, row in picked:
            selected.append({"sample_index": index, "family": family, **row})
    selected.sort(key=lambda row: (row["family"], row["sample_index"]))
    manifest = {
        "status": "FROZEN_BEFORE_COMPLEXITY_ANALYSIS",
        "dataset": "KQA Pro",
        "source_url": KQA_SOURCE_URL,
        "official_repository": KQA_OFFICIAL_REPO,
        "paper_url": KQA_PAPER,
        "official_download_url": KQA_OFFICIAL_DOWNLOAD,
        "source_split": "val",
        "source_file": str(KQA_SOURCE),
        "source_sha256": sha256(KQA_SOURCE),
        "source_version": "Hugging Face dataset drt/kqa_pro commit unavailable from file API; local source hash recorded",
        "download_date": str(date.today()),
        "official_source_status": "official repository/paper identified; linked Tsinghua download returned Link does not exist",
        "fallback_reason": "Required official download endpoint was unavailable, so the exact validation split was retrieved from the public Hugging Face mirror.",
        "selection_seed": SELECTION_SEED,
        "sample_size": len(selected),
        "per_family_target": 20,
        "family_mapping": {
            "What": "QueryName",
            "QueryAttrUnderCondition": "QueryAttr",
            "VerifyStr/VerifyNum/VerifyYear/VerifyDate": "Verify",
        },
        "families": KQA_FAMILIES,
        "family_counts": dict(Counter(row["family"] for row in selected)),
        "selected_source_indices": selections,
        "sample_indices_are_original_validation_order": True,
    }
    return selected, manifest


def entropy(counts: Mapping[str, int]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return float(-sum((value / total) * math.log2(value / total) for value in counts.values() if value))


def dataset_flag_prevalence(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result = {}
    for flag in NORMALIZED_FLAGS:
        count = sum(bool(row["complexity_flags"].get(flag)) for row in rows)
        result[flag] = {"count": count, "n": len(rows), "prevalence": count / len(rows) if rows else None}
    return result


def normalized_stats(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> Dict[str, Dict[str, Optional[float]]]:
    return {field: numeric_summary(row.get(field) for row in rows) for field in fields}


def build_comparisons(phase: Sequence[Mapping[str, Any]], kqa: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    common_fields = ["logical_form_depth", "variable_count", "relation_count", "constraint_count", "operator_count", "max_path_length", "post_merge_depth"]
    phase_stats = normalized_stats(phase, common_fields)
    kqa_stats = normalized_stats(
        [
            {
                "logical_form_depth": row["normalized_depth"],
                "variable_count": row["variable_count_if_derivable"],
                "relation_count": row["relation_operation_count"],
                "constraint_count": row["constraint_count"],
                "operator_count": row["operator_count"],
                "max_path_length": row["normalized_depth"],
                "post_merge_depth": int(row["complexity_flags"].get("requires_downstream_relation", False)),
            }
            for row in kqa
        ],
        common_fields,
    )
    flag_compare = {}
    phase_flags = dataset_flag_prevalence(phase)
    kqa_flags = dataset_flag_prevalence(kqa)
    for flag in NORMALIZED_FLAGS:
        flag_compare[flag] = {"phase3c_strong": phase_flags[flag], "kqa_pro_sample": kqa_flags[flag]}
    comparison = {
        "sample_sizes": {"phase3c_strong": len(phase), "kqa_pro": len(kqa)},
        "cross_dataset_warning": "Raw SPARQL, S-expression, and KoPL token lengths are not treated as equivalent. Comparisons use normalized features and report raw forms separately.",
        "normalized_numeric_statistics": {"phase3c_strong": phase_stats, "kqa_pro_sample": kqa_stats},
        "normalized_feature_prevalence": flag_compare,
        "table_rows": [
            {"dimension": flag, "phase3c_count": phase_flags[flag]["count"], "phase3c_percent": phase_flags[flag]["prevalence"], "kqa_count": kqa_flags[flag]["count"], "kqa_percent": kqa_flags[flag]["prevalence"]}
            for flag in NORMALIZED_FLAGS
        ],
    }
    phase_struct = Counter(label for row in phase for label in row["complexity_categories"])
    kqa_struct = Counter()
    for row in kqa:
        labels = row["complexity_categories"]
        if "AGGREGATION" in labels:
            primary = "AGGREGATION"
        elif "QUALIFIER" in labels:
            primary = "QUALIFIER"
        elif "COMPARATIVE" in labels:
            primary = "COMPARATIVE"
        elif "NESTED" in labels:
            primary = "NESTED"
        elif "CONJUNCTION" in labels:
            primary = "CONJUNCTION"
        elif "VERIFY" in labels:
            primary = "VERIFY"
        else:
            primary = "LINEAR"
        kqa_struct[primary] += 1
    structure = {
        "phase3c_multi_label_distribution": dict(sorted(phase_struct.items())),
        "kqa_pro_multi_label_distribution": dict(sorted(Counter(label for row in kqa for label in row["complexity_categories"]).items())),
        "kqa_pro_primary_comparison_distribution": dict(sorted(kqa_struct.items())),
        "mapping_note": "KQA Pro primary labels are a descriptive comparison view; its multi-label categories are retained and are not forced into a one-to-one equivalence with Phase 3C structures.",
    }
    phase_families = Counter(family for row in phase for family in row["operator_families"])
    kqa_families = Counter(family for row in kqa for family in row["normalized_operator_families"])
    diversity = {
        "phase3c_strong": {
            "unique_normalized_operator_families": len(phase_families),
            "operator_family_counts": dict(sorted(phase_families.items())),
            "distribution_entropy_bits": entropy(phase_families),
            "mean_operator_families_per_question": mean(len(row["operator_families"]) for row in phase),
        },
        "kqa_pro_sample": {
            "unique_normalized_operator_families": len(kqa_families),
            "operator_family_counts": dict(sorted(kqa_families.items())),
            "distribution_entropy_bits": entropy(kqa_families),
            "mean_operator_families_per_question": mean(len(row["normalized_operator_families"]) for row in kqa),
        },
        "interpretation_note": "Entropy is descriptive and is not used as a benchmark label or a tuned separation score.",
    }
    depth = {
        "phase3c_strong": depth_distribution([int(row["logical_form_depth"]) for row in phase]),
        "kqa_pro_sample": depth_distribution([int(row["normalized_depth"]) for row in kqa]),
        "metric_note": "Phase 3C uses the processed gold logical-form/SPARQL depth field; KQA Pro uses dependency depth of the KoPL program.",
    }
    constraints = {
        "counting_rule": "relation conditions + type conditions + filter conditions + comparison conditions + qualifier conditions; KQA additionally counts explicit attribute/query, verification, and aggregation requirements from KoPL operators.",
        "phase3c_strong": numeric_summary(row["constraint_count"] for row in phase),
        "kqa_pro_sample": numeric_summary(row["constraint_count"] for row in kqa),
        "histograms": {
            "phase3c_strong": dict(sorted(Counter(row["constraint_count"] for row in phase).items())),
            "kqa_pro_sample": dict(sorted(Counter(row["constraint_count"] for row in kqa).items())),
        },
    }
    return comparison, structure, diversity, depth, constraints


def depth_distribution(values: Sequence[int]) -> Dict[str, Any]:
    buckets = Counter("5+" if value >= 5 else str(value) for value in values)
    return {"counts": dict(sorted(buckets.items(), key=lambda pair: (pair[0] == "5+", pair[0]))), "statistics": numeric_summary(values), "n": len(values)}


def reasoning_sketch(row: Mapping[str, Any], dataset: str) -> str:
    if dataset == "phase3c":
        predicates = [key for key, value in row.get("operator_counts", {}).items() if value]
        return f"{row.get('gold_structure')} with {row.get('branch_count')} branch source(s); {row.get('relation_count')} relation condition(s), operator families {', '.join(predicates)}; answer is recovered after {row.get('post_merge_depth')} post-merge step(s)."
    functions = [step.get("function", "?") for step in row.get("program", [])]
    return f"KoPL executes {len(functions)} steps: {' -> '.join(functions)}; dependency depth {row.get('normalized_depth')}, with normalized families {', '.join(row.get('normalized_operator_families', []))}."


def make_manual_audit(phase: Sequence[Mapping[str, Any]], kqa: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(MANUAL_SEED)
    phase_sorted = sorted(phase, key=lambda row: (row["logical_form_depth"], row["constraint_count"], row["case_id"]))
    kqa_sorted = sorted(kqa, key=lambda row: (row["normalized_depth"], row["constraint_count"], row["sample_index"]))
    phase_indices = sorted(set([0, len(phase_sorted) // 2, len(phase_sorted) - 1] + rng.sample(range(len(phase_sorted)), 7)))
    kqa_indices = sorted(set([0, len(kqa_sorted) // 2, len(kqa_sorted) - 1] + rng.sample(range(len(kqa_sorted)), 7)))
    result = []
    for index in phase_indices[:10]:
        row = phase_sorted[index]
        result.append({"audit_id": f"phase3c-{index}", "dataset": "Phase3C", "sample_seed": MANUAL_SEED, "question": row["question"], "gold_reasoning_form": row.get("gold_logical_form", row.get("gold_sparql")), "normalized_features": row, "reasoning_sketch": reasoning_sketch(row, "phase3c"), "review_status": "AUTOMATIC_FEATURES_MANUALLY_INSPECTED"})
    for index in kqa_indices[:10]:
        row = kqa_sorted[index]
        result.append({"audit_id": f"kqa-{index}", "dataset": "KQA Pro", "sample_seed": MANUAL_SEED, "question": row["question"], "gold_reasoning_form": row["program"], "normalized_features": row, "reasoning_sketch": reasoning_sketch(row, "kqa"), "review_status": "AUTOMATIC_FEATURES_MANUALLY_INSPECTED"})
    return result


def capability_gap_matrix(comparison: Mapping[str, Any]) -> List[Dict[str, Any]]:
    feature_map = comparison["normalized_feature_prevalence"]
    rows = [
        ("QUALIFIER", "qualifier_reasoning", "NONE", "NONE", "explicit qualifier-aware relation state"),
        ("COUNT_AGGREGATION", "aggregation", "NONE", "PARTIAL (COUNT-like SPARQL forms are not a validated runtime capability)", "aggregation-aware execution and answer-state semantics"),
        ("COMPARISON_SELECTION", "comparison", "PARTIAL (gold filters/order are represented offline)", "PARTIAL", "explicit comparison and selection state"),
        ("TEMPORAL_FILTER", "temporal_filter", "PARTIAL (dataset-specific gold forms)", "PARTIAL", "temporal-aware relation and value constraints"),
        ("SET_OPERATION", "set_operation", "PARTIAL (branch intersection only)", "PARTIAL (And/Or not equivalent to branch execution)", "general set-operation state and executor support"),
        ("ATTRIBUTE_QUERY", "attribute_query", "PARTIAL (relation traversal)", "PARTIAL (QueryAttr family)", "attribute-aware answer-state handling"),
        ("VERIFICATION", "verification", "NONE established in Phase 3C", "NONE established in Phase 3C", "verification-specific execution and evaluator contract"),
        ("NESTED_COMPOSITION", "nested_composition", "PARTIAL; BRANCH_INTERMEDIATE remains limited", "PARTIAL", "validated nested dependency and branch-to-merge continuation"),
    ]
    result = []
    for family, flag, v1_support, p25_support, extension in rows:
        result.append({"operator_family": family, "phase3c_prevalence": feature_map[flag]["phase3c_strong"], "kqa_pro_prevalence": feature_map[flag]["kqa_pro_sample"], "v1_1_3_support": v1_support, "phase2_5_6_support": p25_support, "required_extension": extension})
    return result


def markdown_questions(phase: Sequence[Mapping[str, Any]]) -> str:
    groups = [("GrailQA", "BRANCH_FINAL"), ("GrailQA", "BRANCH_INTERMEDIATE"), ("CWQ", "BRANCH_FINAL"), ("CWQ", "BRANCH_INTERMEDIATE")]
    lines = ["# Phase 3C Strong Questions", "", "These are the exact 60 frozen Phase 3C strong cases recovered from the original processed dataset records. Gold forms are retained without truncation.", ""]
    for dataset, structure in groups:
        lines += [f"## {dataset} {structure}", ""]
        rows = [row for row in phase if row["dataset"].lower() == dataset.lower() and row["gold_structure"] == structure]
        for index, row in enumerate(sorted(rows, key=lambda item: item["case_id"]), 1):
            lines += [f"{index}. `{row['case_id']}`", f"   Question: {row['question']}", f"   Gold logical form: {row['gold_logical_form']}", f"   Complexity summary: relation_count={row['relation_count']}; constraints={row['constraint_count']}; depth={row['logical_form_depth']}; path_length={row['max_path_length']}; branches={row['branch_count']}; categories={', '.join(row['complexity_categories'])}", ""]
    return "\n".join(lines)


def markdown_kqa_examples(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# KQA Pro Complexity Examples", "", "Three frozen examples per normalized KQA Pro family from the 180-question validation sample. The family mapping is recorded in `kqa_pro_sample_manifest.json`.", ""]
    for family in KQA_FAMILIES:
        lines += [f"## {family}", ""]
        examples = [row for row in rows if row["family"] == family][:3]
        for index, row in enumerate(examples, 1):
            lines += [f"{index}. Sample index `{row['sample_index']}`", f"   Question: {row['question']}", f"   KoPL program: `{json.dumps(row['program'], ensure_ascii=False)}`", f"   Normalized labels: {', '.join(row['complexity_categories'])}", f"   Reasoning depth: {row['normalized_depth']}", f"   Operator families: {', '.join(row['normalized_operator_families'])}", ""]
    return "\n".join(lines)


def decision(comparison: Mapping[str, Any], phase: Sequence[Mapping[str, Any]], kqa: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    flags = comparison["normalized_feature_prevalence"]
    phase_branch = Counter(row["gold_structure"] for row in phase)
    kqa_richer_dims = []
    for flag in ["nested_composition", "comparison", "aggregation", "qualifier_reasoning", "verification", "set_operation", "three_plus_constraints"]:
        if flags[flag]["kqa_pro_sample"]["prevalence"] > flags[flag]["phase3c_strong"]["prevalence"] + 0.10:
            kqa_richer_dims.append(flag)
    phase_shallow = all(not row["complexity_flags"].get("nested_composition") for row in phase) or sum(row["complexity_flags"].get("nested_composition", False) for row in phase) < len(phase) * 0.25
    kqa_has_more = len(kqa_richer_dims) >= 3
    if not kqa_has_more:
        adequacy = "CURRENT_BENCHMARK_ADEQUATE"
        hypothesis = "NOT_SUPPORTED"
    elif phase_shallow and kqa_has_more:
        adequacy = "CURRENT_BENCHMARK_NARROW"
        hypothesis = "SUPPORTED"
    else:
        adequacy = "COMPLEXITY_GAP_MIXED"
        hypothesis = "PARTIALLY_SUPPORTED"
    return {
        "dataset_adequacy_decision": adequacy,
        "dataset_hypothesis_conclusion": hypothesis,
        "evidence": {"kqa_materially_richer_dimensions": kqa_richer_dims, "phase3c_structure_counts": dict(phase_branch), "phase3c_shallow_nested_composition": phase_shallow, "threshold": "KQA prevalence exceeds Phase 3C by more than 10 percentage points in at least three independent normalized dimensions"},
        "interpretation": "This decision is based on gold reasoning structure, normalized operators, dependency depth, and semantic constraints; it does not use model accuracy.",
    }


def report(
    phase: Sequence[Mapping[str, Any]],
    kqa: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    comparison: Mapping[str, Any],
    structure: Mapping[str, Any],
    diversity: Mapping[str, Any],
    depth: Mapping[str, Any],
    constraints: Mapping[str, Any],
    capability: Sequence[Mapping[str, Any]],
    decision_result: Mapping[str, Any],
) -> str:
    flags = comparison["normalized_feature_prevalence"]
    def pct(name: str, dataset: str) -> str:
        return f"{flags[name][dataset]['count']}/{flags[name][dataset]['n']} ({flags[name][dataset]['prevalence']:.1%})"
    return f"""# Phase 5 Dataset Complexity Adequacy Audit

## 1. Motivation

Phase 3C established `NO_BRANCH_ADVANTAGE` on a frozen 60-case strong branch-required cohort. Phase 5 tests whether that benchmark was compositionally narrow enough to miss advantages that might appear on a broader reasoning space.

## 2. Dataset hypothesis

`H0`: Phase 3C is compositionally comparable to a balanced KQA Pro sample.

`H1`: Phase 3C is materially narrower, especially in operator diversity, nesting, aggregation, qualification, comparison, and multi-stage composition.

The audit uses gold reasoning structures and does not use model accuracy to define complexity.

## 3. Frozen Phase 3C source

The source was `phase3c_eval_manifest.json` plus the original processed CWQ/GrailQA records named by each frozen row's `source_path`. Exactly **{len(phase)}** strong cases were recovered with no duplicate IDs and no unresolved source records. The verified strata are `{dict(Counter((row['dataset'], row['gold_structure']) for row in phase))}`.

## 4. Exact Phase 3C question set

The full question catalog is in `PHASE3C_60_QUESTIONS.md`; `phase3c_question_catalog.csv` contains the required machine-readable fields.

## 5. Phase 3C complexity methodology

CWQ complexity was derived from gold SPARQL and processed gold structure metadata. GrailQA complexity was derived from gold S-expressions, gold SPARQL, and processed logical-form metadata. Model traces and inference outcomes were not used. Raw form lengths were not treated as cross-dataset-equivalent complexity measures.

## 6. Phase 3C complexity distribution

The machine-readable features are in `phase3c_complexity_features.jsonl`; aggregate statistics are in `phase3c_complexity_summary.json`.

Key normalized prevalence:

| Dimension | Phase 3C strong | KQA Pro sample |
|---|---:|---:|
| Multi-hop | {pct('multi_hop','phase3c_strong')} | {pct('multi_hop','kqa_pro_sample')} |
| Conjunction | {pct('conjunction','phase3c_strong')} | {pct('conjunction','kqa_pro_sample')} |
| Shared intermediate | {pct('shared_intermediate','phase3c_strong')} | {pct('shared_intermediate','kqa_pro_sample')} |
| Nested composition | {pct('nested_composition','phase3c_strong')} | {pct('nested_composition','kqa_pro_sample')} |
| 3+ constraints | {pct('three_plus_constraints','phase3c_strong')} | {pct('three_plus_constraints','kqa_pro_sample')} |
| Comparison | {pct('comparison','phase3c_strong')} | {pct('comparison','kqa_pro_sample')} |
| Superlative | {pct('superlative','phase3c_strong')} | {pct('superlative','kqa_pro_sample')} |
| Counting | {pct('counting','phase3c_strong')} | {pct('counting','kqa_pro_sample')} |
| Aggregation | {pct('aggregation','phase3c_strong')} | {pct('aggregation','kqa_pro_sample')} |
| Qualifier reasoning | {pct('qualifier_reasoning','phase3c_strong')} | {pct('qualifier_reasoning','kqa_pro_sample')} |
| Verification | {pct('verification','phase3c_strong')} | {pct('verification','kqa_pro_sample')} |
| Temporal filtering | {pct('temporal_filter','phase3c_strong')} | {pct('temporal_filter','kqa_pro_sample')} |
| Numeric filtering | {pct('numeric_filter','phase3c_strong')} | {pct('numeric_filter','kqa_pro_sample')} |
| Set operation | {pct('set_operation','phase3c_strong')} | {pct('set_operation','kqa_pro_sample')} |

## 7. KQA Pro source and sample design

The official paper and baseline repository identify KQA Pro and its KoPL programs. The repository's linked Tsinghua download was unavailable at audit time, so the exact `val.json` split was retrieved from the public `drt/kqa_pro` mirror and frozen locally. Source metadata, hash, URL, split, date, seed `20261030`, and family mapping are in `kqa_pro_sample_manifest.json`.

The sample contains **{len(kqa)}** questions: 20 per each of nine normalized terminal-function families. `QueryAttrUnderCondition` is mapped to `QueryAttr`; `What` is mapped to `QueryName`; all `Verify*` functions are mapped to `Verify`. The original terminal function remains in every feature row.

## 8. KQA Pro complexity methodology

KQA Pro uses explicit KoPL programs. Program length, dependency depth, source anchors, relation operations, filters, qualifiers, comparison/selection, counting, verification, and normalized operator families were extracted directly from the frozen programs.

## 9. Cross-dataset normalization

SPARQL token count, S-expression token count, and KoPL program length are reported separately and are not treated as identical. Cross-dataset comparisons use normalized flags, dependency depth, semantic constraint count, operator-family diversity, and dependency properties.

## 10. Structural comparison

Phase 3C is intentionally balanced across `BRANCH_FINAL` and `BRANCH_INTERMEDIATE`, while KQA Pro is sampled by terminal-function family. The full multi-label distributions are in `structure_distribution_comparison.json`.

## 11. Operator diversity comparison

Phase 3C normalized operator-family entropy is {diversity['phase3c_strong']['distribution_entropy_bits']:.3f} bits across {diversity['phase3c_strong']['unique_normalized_operator_families']} families. KQA Pro is {diversity['kqa_pro_sample']['distribution_entropy_bits']:.3f} bits across {diversity['kqa_pro_sample']['unique_normalized_operator_families']} families. These are descriptive comparisons only.

## 12. Depth comparison

The depth distributions and mean, median, 75th, and 90th percentiles are in `depth_analysis.json`. Phase 3C depth is based on processed gold logical-form/SPARQL structure; KQA Pro depth is KoPL dependency depth.

## 13. Constraint-count comparison

Constraint counting rules and histograms are in `constraint_count_analysis.json`. The count is a semantic condition count, not a token count.

## 14. Nested reasoning comparison

The audit preserves `BRANCH_INTERMEDIATE` and `requires_downstream_relation` separately. This avoids claiming that every Phase 3C case is shallow merely because it has two branches. At the same time, the current Phase 3C design does not sample general KoPL nesting, verification, or aggregation families in a balanced way.

## 15. Aggregation/comparison comparison

The KQA Pro sample explicitly covers Count, SelectAmong, SelectBetween, and Verify families. Phase 3C contains any such semantics only when present in its CWQ/GrailQA gold forms; it was not selected to balance these operator families.

## 16. Qualifier reasoning comparison

Qualifier prevalence is reported as a normalized flag. KQA Pro has explicit qualifier/query-qualifier operators, while Phase 3C qualifier detection is based on gold SPARQL qualifier patterns. The capability matrix marks current runtime support conservatively rather than equating gold prevalence with method support.

## 17. Manual audit

`manual_audit.jsonl` contains 10 deterministic Phase 3C cases and 10 deterministic KQA Pro cases selected with seed `20261031`, including low, median, and high complexity positions. Each record retains its gold form, normalized features, and a short reasoning sketch.

## 18. Capability gap matrix

`capability_gap_matrix.json` records prevalence, conservative v1.1.3 and Phase 2.5.6 support status, and the extension needed for each broader operator family. It does not claim that KQA Pro execution is already supported.

## 19. Dataset-adequacy decision

**{decision_result['dataset_adequacy_decision']}**

The rule and evidence are stored in `decision_summary.md`. A benchmark-adequacy label is distinct from a method-performance result.

## 20. Dataset-hypothesis conclusion

**{decision_result['dataset_hypothesis_conclusion']}**

The negative Phase 3C result should therefore be interpreted as a valid result for the frozen two-source branch-necessity benchmark. If the decision is narrow or mixed, it does not invalidate Phase 3C; it limits the scope of the conclusion and motivates a separate KQA Pro baseline-transfer phase.

## 21. Implication for Phase 3C interpretation

Phase 3C established that the frozen branch-aware method did not outperform v1.1.3 on a semantically strong branch-required sample. This audit does not overturn that result. It tests a different hypothesis: whether the benchmark covers the broader compositional space of a general relation-graph architecture.

## 22. Recommendation for future evaluation

Do not implement a new reasoning method in this phase. If the complexity gap is supported or mixed, the next stage should be **Phase 6 — KQA Pro Backend + Frozen Baseline Transfer**, beginning with baseline execution and data compatibility rather than method improvement.

## 23. Limitations

The KQA Pro comparison uses a balanced 180-question validation sample rather than the full dataset. Its family labels are derived from terminal KoPL functions because the local release does not provide a separate family field. Cross-dataset normalization is necessarily approximate. Phase 3C uses Freebase-oriented gold forms while KQA Pro uses its own KoPL/SPARQL conventions. Capability support labels are conservative audit judgments, not runtime benchmark results.

## 24. Final decision

Phase 5 is complete as an offline dataset-complexity audit. No router, prompt, validator, normalizer, branch executor, fallback, relation scorer, evaluator, or inference configuration was modified.
"""


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = read_json(PHASE3C_MANIFEST)
    strong_manifest = manifest["strong_cases"]
    if len(strong_manifest) != 60 or len({row["case_id"] for row in strong_manifest}) != 60:
        raise RuntimeError("Phase 3C strong manifest is not exactly 60 unique cases")
    sources = load_source_records(strong_manifest)
    phase_features = [phase3c_feature(row, sources[row["case_id"]]) for row in strong_manifest]
    if len({row["case_id"] for row in phase_features}) != 60 or any(not row["question"] for row in phase_features):
        raise RuntimeError("Phase 3C catalog recovery failed")

    catalog_path = OUTPUT_ROOT / "phase3c_question_catalog.csv"
    with catalog_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "dataset", "question", "gold_structure", "gold_logical_form", "gold_answer", "branch_necessity_class"])
        writer.writeheader()
        for row in phase_features:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], (list, dict)) else row[key] for key in writer.fieldnames})
    write_jsonl(OUTPUT_ROOT / "phase3c_complexity_features.jsonl", phase_features)
    summary = {
        "sample_size": len(phase_features),
        "strata": dict(Counter(f"{row['dataset']}|{row['gold_structure']}" for row in phase_features)),
        "numeric_statistics": normalized_stats(phase_features, ["relation_count", "variable_count", "logical_form_depth", "max_path_length", "operator_count", "constraint_count", "post_merge_depth"]),
        "flag_prevalence": dataset_flag_prevalence(phase_features),
        "category_counts": dict(sorted(Counter(category for row in phase_features for category in row["complexity_categories"]).items())),
        "methodology": "Gold SPARQL/S-expression and processed gold structure only; no model traces or accuracy labels.",
    }
    write_json(OUTPUT_ROOT / "phase3c_complexity_summary.json", summary)

    kqa_rows = read_json(KQA_SOURCE)
    sampled, kqa_manifest = sample_kqa(kqa_rows)
    write_json(OUTPUT_ROOT / "kqa_pro_sample_manifest.json", kqa_manifest)
    kqa_features_rows = [kqa_features(row) for row in sampled]
    write_jsonl(OUTPUT_ROOT / "kqa_pro_complexity_features.jsonl", kqa_features_rows)
    comparison, structure, diversity, depth, constraints = build_comparisons(phase_features, kqa_features_rows)
    write_json(OUTPUT_ROOT / "complexity_comparison.json", comparison)
    write_json(OUTPUT_ROOT / "structure_distribution_comparison.json", structure)
    write_json(OUTPUT_ROOT / "operator_diversity_analysis.json", diversity)
    write_json(OUTPUT_ROOT / "depth_analysis.json", depth)
    write_json(OUTPUT_ROOT / "constraint_count_analysis.json", constraints)
    write_json(OUTPUT_ROOT / "capability_gap_matrix.json", capability_gap_matrix(comparison))
    write_jsonl(OUTPUT_ROOT / "manual_audit.jsonl", make_manual_audit(phase_features, kqa_features_rows))
    (OUTPUT_ROOT / "PHASE3C_60_QUESTIONS.md").write_text(markdown_questions(phase_features))
    (OUTPUT_ROOT / "KQA_PRO_COMPLEXITY_EXAMPLES.md").write_text(markdown_kqa_examples(kqa_features_rows))
    decision_result = decision(comparison, phase_features, kqa_features_rows)
    write_json(OUTPUT_ROOT / "decision_summary.json", decision_result)
    (OUTPUT_ROOT / "decision_summary.md").write_text(
        f"# Phase 5 Decision Summary\n\n"
        f"Dataset adequacy decision: **{decision_result['dataset_adequacy_decision']}**\n\n"
        f"Dataset hypothesis conclusion: **{decision_result['dataset_hypothesis_conclusion']}**\n\n"
        f"KQA-richer normalized dimensions: `{decision_result['evidence']['kqa_materially_richer_dimensions']}`\n\n"
        "This is an offline gold-complexity decision. It does not use model accuracy and does not authorize a method change.\n"
    )
    (OUTPUT_ROOT / "PHASE5_REPORT.md").write_text(report(phase_features, kqa_features_rows, kqa_manifest, comparison, structure, diversity, depth, constraints, capability_gap_matrix(comparison), decision_result))
    print(json.dumps({"status": "COMPLETE", "output_root": str(OUTPUT_ROOT), "phase3c_cases": len(phase_features), "kqa_cases": len(kqa_features_rows), **decision_result}, indent=2))


if __name__ == "__main__":
    main()
