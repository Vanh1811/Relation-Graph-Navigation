#!/usr/bin/env python3
"""Create the fixed, offline Iteration 24 label-source audit manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_024_label_source_audit"
UNRESOLVED_IDS = [
    "m.04stk8g", "m.0ndvpvf", "m.04st848", "m.0hzdjww",
    "m.0hz9k44", "m.04stpzt", "m.04strb6",
]
PREFLIGHT_QUERY = "SELECT * WHERE { ?s ?p ?o } LIMIT 1"
VALUES = " ".join(f"ns:{entity_id}" for entity_id in UNRESOLVED_IDS)
SOURCE_QUERY = (
    "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "SELECT DISTINCT ?entity ?predicate ?value\n"
    "WHERE {\n"
    f"  VALUES ?entity {{ {VALUES} }}\n"
    "  VALUES ?predicate { ns:type.object.name rdfs:label ns:common.topic.alias ns:type.object.key }\n"
    "  ?entity ?predicate ?value .\n"
    "}\n"
    "LIMIT 2000"
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    runner_rel = "scripts/diagnostics/iter_024_label_source_audit/run_label_source_audit.py"
    safety_rel = "scripts/diagnostics/transport_safety_repair/hard_deadline.py"
    document = {
        "schema_version": 1,
        "iteration": 24,
        "name": "fixed-seven-entity-label-source-audit",
        "objective": "Inspect whether any non-English/unfiltered names, labels, aliases, or keys are present for Iteration 23's seven unresolved entity IDs. Do not adopt any value as a CoR prompt label automatically.",
        "parent_iteration_manifest_sha256": "ff57d84266295e7778b00e733fdba9e8a9232f4eed87971f7b0e029cefc331f2",
        "endpoint": "http://127.0.0.1:8890/sparql",
        "fixed_entity_ids": UNRESOLVED_IDS,
        "api": {"enabled": False, "requests": 0, "spend_cap_usd": 0},
        "sparql": {
            "operation": "read_only_SELECT_only",
            "maximum_physical_attempts": 2,
            "preflight_attempts_included": 1,
            "retries": 0,
            "maximum_seconds_per_attempt": 120,
            "maximum_elapsed_seconds": 1800,
            "queries": [
                {"category": "endpoint_preflight", "query": PREFLIGHT_QUERY,
                 "sha256": digest(PREFLIGHT_QUERY)},
                {"category": "label_source_evidence", "query": SOURCE_QUERY,
                 "sha256": digest(SOURCE_QUERY)},
            ],
        },
        "runtime_source_sha256": {
            runner_rel: sha_file(ROOT / runner_rel),
            safety_rel: sha_file(ROOT / safety_rel),
        },
        "decision_constraints": [
            "No API or LLM calls.",
            "No query beyond the two frozen SELECT strings.",
            "No retries; a timeout consumes the attempt and stops the audit.",
            "Candidate predicate values are evidence only; aliases and keys are not promoted to names.",
            "Any later label-cache update requires a separate semantic review and must preserve provenance.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "PROPOSED_MANIFEST.json"
    payload = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    manifest_path.write_text(payload, encoding="utf-8")
    manifest_hash = digest(payload)
    (OUT / "PROPOSED_MANIFEST.sha256").write_text(f"{manifest_hash}  PROPOSED_MANIFEST.json\n", encoding="utf-8")
    authorization_text = (
        f"I authorize Iteration 24 manifest {manifest_hash}: one read-only Freebase label-source audit "
        "for the seven fixed Iteration 23 unresolved IDs using at most 2 SPARQL SELECT attempts total "
        "including 1 endpoint preflight, no OpenAI/API requests, no retries, max 120 seconds per request, "
        "max 30 minutes total, endpoint http://127.0.0.1:8890/sparql."
    )
    template = {
        "authorization_status": "PENDING",
        "manifest_sha256": manifest_hash,
        "approval_verbatim": authorization_text,
        "approved_at": None,
        "transport_opened": False,
        "actual_usage": {"api_requests": 0, "read_only_sparql_attempts": 0, "retries": 0},
    }
    (OUT / "AUTHORIZATION_TEMPLATE.json").write_text(
        json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest_sha256": manifest_hash, "authorization": authorization_text}, indent=2))


if __name__ == "__main__":
    main()
