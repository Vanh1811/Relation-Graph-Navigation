"""Run Iteration 23 replay and deterministic synthetic readiness fixture only."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
for relative in (
    "scripts/diagnostics/iter_021_matched_prefix",
    "scripts/diagnostics/iter_022_gated_matched_prefix",
    "scripts/diagnostics/iter_023_label_coverage_and_prefix_readiness",
):
    path = str(ROOT / relative)
    if path not in sys.path:
        sys.path.insert(0, path)

from chain_of_relations.kg_backend.freebase_backend import FreebaseBackend  # noqa: E402
from chain_of_relations.schema import Entity, Relation  # noqa: E402
from chain_of_relations.tools.entity_search import EntitySearchInput, entity_search  # noqa: E402
from label_resolver import BoundedLabelResolver, PhysicalQueryMeter, install_metered_freebase_backend  # noqa: E402
from pilot import dry_run_case  # noqa: E402
from test_pilot import prefix_fixture, projection_fixture  # noqa: E402


def main() -> dict:
    out = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness"
    prior = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_022_gated_matched_prefix"
    traces = [json.loads(line) for line in (prior / "RAW_CASE_TRACES.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    observed_misses = sorted({entity_id for trace in traces for entity_id in trace.get("missing_ids", [])})
    required_misses = sorted({
        "m.04st848", "m.04stk8g", "m.04stpzt", "m.04strb6",
        "m.0hz9k44", "m.0hzdjww", "m.0ndvpvf",
    })
    if observed_misses != required_misses:
        raise SystemExit("archived Iteration 22 label misses did not reproduce exactly")

    cache_doc = json.loads((prior / "LOCAL_LABEL_CACHE.json").read_text(encoding="utf-8"))
    cache = {str(row["entity_id"]): str(row["label"]) for row in cache_doc["labels"]}
    fake_calls: list[str] = []

    def synthetic_transport(query: str) -> dict:
        fake_calls.append(query)
        if "VALUES ?entity" in query:
            return {"status": "ok", "rows": [{
                "entity": {"type": "uri", "value": "http://rdf.freebase.com/ns/m.iter23synthetic"},
                "targetEntity": {"type": "literal", "xml:lang": "en", "value": "Synthetic resolved Ω label"},
            }]}
        return {"status": "ok", "rows": [{
            "targetEntity": {"type": "uri", "value": "http://rdf.freebase.com/ns/m.iter23synthetic"},
        }]}

    meter = PhysicalQueryMeter(hard_cap=140, transport=synthetic_transport, transport_kind="synthetic_fixture")
    resolver = BoundedLabelResolver(label_cache=cache, meter=meter)
    backend = FreebaseBackend()
    with install_metered_freebase_backend(backend, meter=meter, resolver=resolver):
        candidate_output = entity_search(
            EntitySearchInput(
                topic_entity=Entity("m.syntheticroot", "Synthetic root"),
                relation_chain=[Relation("synthetic.relation")],
            ),
            backend=backend,
        )

    if candidate_output.entity_ids != ["m.iter23synthetic"]:
        raise SystemExit(f"synthetic entity-search IDs differ: {candidate_output.entity_ids}")
    if candidate_output.entity_names != ["Synthetic resolved Ω label"]:
        raise SystemExit(f"synthetic resolved label missing from normal candidate path: {candidate_output.entity_names}")
    if meter.category_counts != {"navigation": 1, "label_resolution": 1}:
        raise SystemExit(f"synthetic query routing changed: {meter.category_counts}")

    pair_result = dry_run_case(
        snapshot=prefix_fixture(),
        projection=projection_fixture(),
        projection_variable_order=["?a", "?b", "?endpoint"],
        llm_response="r.next",
        parsed_selection=[{"id": "r.next", "direction": "OUTGOING"}],
        kg_query="SELECT ?e3 ?e4 WHERE { ?e3 <http://rdf.freebase.com/ns/r.next> ?e4 }",
        kg_rows=[{"?e3": "m.c", "?e4": "m.iter23synthetic"}],
    )
    non_forward = dry_run_case(
        snapshot=prefix_fixture(boundary_action="Finish"),
        projection=projection_fixture(),
        projection_variable_order=["?a", "?b", "?endpoint"],
        llm_response="r.next",
        parsed_selection=[{"id": "r.next", "direction": "OUTGOING"}],
        kg_query="SELECT ?e3 ?e4 WHERE {}", kg_rows=[],
    )
    if pair_result.get("status") != "SYNTHETIC_PAIR_VALID":
        raise SystemExit(f"synthetic STOP/ONE_HOP pair failed: {pair_result}")
    if pair_result.get("stop_parent_hash") != pair_result.get("one_hop_parent_hash"):
        raise SystemExit("STOP and ONE_HOP do not share an identical parent hash")
    if pair_result.get("new_edge_count") != 1 or pair_result.get("new_entity_ids") != ["m.iter23synthetic"]:
        raise SystemExit("synthetic continuation did not add exactly one expected binding")
    if non_forward.get("reason") != "NON_FORWARD_BOUNDARY":
        raise SystemExit("synthetic non-Forward fixture did not remain ineligible")

    result = {
        "iteration": 23,
        "evidence_class": "OFFLINE_REPLAY_AND_SYNTHETIC_FIXTURE_ONLY",
        "status": "PASS",
        "archived_iter22_miss_reproduction": {
            "trace_count": len(traces), "unique_miss_count": len(observed_misses),
            "ids": observed_misses, "exact_match": True,
        },
        "dynamic_label_synthetic": {
            "unknown_id": "m.iter23synthetic", "label": candidate_output.entity_names[0],
            "same_entity_search_path": True, "query_attempts": meter.physical_attempts,
            "query_categories": meter.category_counts, "transport_kind": meter.transport_kind,
            "network_calls": 0, "api_requests": 0,
        },
        "matched_prefix_fixture": {
            "status": pair_result["status"], "eligible_pairs": 1,
            "same_parent_hash": pair_result["stop_parent_hash"] == pair_result["one_hop_parent_hash"],
            "prefix_hash": pair_result["prefix_hash"], "new_edge_count": pair_result["new_edge_count"],
            "new_entity_ids": pair_result["new_entity_ids"],
            "simulated_api_attempts": pair_result["api_attempts_simulated"],
            "simulated_sparql_attempts": pair_result["sparql_attempts_simulated"],
        },
        "non_forward_fixture": {"status": "INELIGIBLE", "reason": non_forward["reason"]},
        "live_api_requests": 0,
        "live_sparql_attempts": 0,
        "gold_data_loaded": False,
    }
    (out / "SYNTHETIC_PILOT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
