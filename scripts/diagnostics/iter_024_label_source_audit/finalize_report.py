#!/usr/bin/env python3
"""Reconcile Iteration 24's saved evidence and write its report offline."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_024_label_source_audit"


def main() -> None:
    manifest_bytes = (OUT / "PROPOSED_MANIFEST.json").read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    expected_hash = (OUT / "PROPOSED_MANIFEST.sha256").read_text(encoding="utf-8").split()[0]
    if manifest_hash != expected_hash:
        raise SystemExit("manifest hash mismatch")
    manifest = json.loads(manifest_bytes)
    result = json.loads((OUT / "RESULT.json").read_text(encoding="utf-8"))
    state = json.loads((OUT / "RUN_STATE.json").read_text(encoding="utf-8"))
    approval = json.loads((OUT / "AUTHORIZATION.json").read_text(encoding="utf-8"))
    journal = [json.loads(line) for line in (OUT / "QUERY_JOURNAL.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    started = [event for event in journal if event.get("status") == "ATTEMPT_STARTED"]
    completed = [event for event in journal if event.get("status") == "OK"]
    if (result.get("status") != "COMPLETE" or result.get("manifest_sha256") != manifest_hash
            or result.get("api_requests") != 0 or result.get("sparql_attempts") != 2
            or result.get("retries") != 0 or state.get("status") != "COMPLETE"):
        raise SystemExit("result/state does not match completed fixed audit caps")
    if (approval.get("authorization_status") != "APPROVED"
            or approval.get("manifest_sha256") != manifest_hash
            or approval.get("actual_usage") != {"api_requests": 0, "read_only_sparql_attempts": 2, "retries": 0}
            or len(started) != 2 or len(completed) != 2):
        raise SystemExit("authorization usage or query journal does not reconcile")
    expected_queries = manifest["sparql"]["queries"]
    for index, spec in enumerate(expected_queries):
        begin = started[index]
        end = completed[index]
        if (begin.get("query_index") != index + 1 or end.get("query_index") != index + 1
                or begin.get("query_sha256") != spec["sha256"]
                or end.get("query_sha256") != spec["sha256"]
                or begin.get("category") != spec["category"]
                or end.get("category") != spec["category"]
                or begin.get("retries") != 0 or end.get("retries") != 0):
            raise SystemExit(f"query {index + 1} differs from frozen manifest")
    evidence = next(item for item in result["results"] if item["category"] == "label_source_evidence")
    if evidence.get("row_count") != 0 or evidence.get("rows"):
        raise SystemExit("expected source-evidence query to have returned zero rows")

    csv_path = OUT / "LABEL_SOURCE_COVERAGE.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "entity_id", "type.object.name_observed", "rdfs:label_observed",
            "common.topic.alias_observed", "type.object.key_observed", "cache_updated",
        ])
        writer.writeheader()
        for entity_id in manifest["fixed_entity_ids"]:
            writer.writerow({
                "entity_id": entity_id,
                "type.object.name_observed": False,
                "rdfs:label_observed": False,
                "common.topic.alias_observed": False,
                "type.object.key_observed": False,
                "cache_updated": False,
            })

    report = f"""# Iteration 24 report: fixed-ID label-source audit

## Outcome

**Complete; no alternate label evidence found.** The batched source query returned zero rows for all seven fixed IDs across `type.object.name`, `rdfs:label`, `common.topic.alias`, and `type.object.key`. No values were added to the frozen cache and none were substituted into prompts.

This establishes only that the active endpoint returned no triples for those seven IDs and those four predicates under the frozen query. It does not establish that the IDs have no outgoing triples under every predicate, nor does it establish what labels may exist in another Freebase export or graph context.

## Fixed scope and accounting

- Manifest: `{manifest_hash}`
- IDs: {len(manifest['fixed_entity_ids'])} fixed Iteration 23 misses
- SPARQL: {result['sparql_attempts']} physical read-only SELECT attempts (one endpoint preflight, one batched evidence query)
- API: {result['api_requests']} requests; estimated API spend $0
- Retries: {result['retries']}
- Elapsed: {result['elapsed_seconds']} seconds
- Endpoint preflight: one row returned; endpoint was reachable
- Label-source evidence query: zero rows returned
- Frozen cache changes: none

The per-entity summary is in `LABEL_SOURCE_COVERAGE.csv`; raw response bindings and query hashes are in `QUERY_JOURNAL.jsonl`.

## Interpretation

Iteration 23's label-resolution stop remains valid. No canonical names, labels, aliases, or keys were found among the four explicitly queried predicates, so these seven entities cannot be safely labeled for the frozen CoR prompt path using the audited endpoint evidence. This does not show a CoR retrieval or reasoning failure: all three Iteration 23 slots had stopped before the measured depth boundary.

## Recommended next step

Keep the frozen Iteration 22 label cache unchanged. Continue offline by selecting new fixed questions whose required displayed entities are already covered by trusted cached labels, then audit the candidate manifest before any new model pilot. A broader graph-structure probe for these seven IDs would be a separate, newly manifested and authorized SPARQL scope.

## Reproducibility and safety

The manifest and authorization reconcile with the saved query journal: two attempts, no retries, zero API requests. The audit used a killable worker with a 120-second hard deadline per request and a 30-minute overall cap. It did not run or resume Iteration 23.
"""
    (OUT / "ITER24_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": "PASS_ACCOUNTING_RECONCILED",
        "manifest_sha256": manifest_hash,
        "sparql_attempts": result["sparql_attempts"],
        "api_requests": result["api_requests"],
        "retries": result["retries"],
        "label_source_rows": evidence["row_count"],
        "entities_without_audited_label_source": len(manifest["fixed_entity_ids"]),
    }, indent=2))


if __name__ == "__main__":
    main()
