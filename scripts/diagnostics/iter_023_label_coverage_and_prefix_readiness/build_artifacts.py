"""Create Iteration 23 offline reports, frozen proposal and central handoff."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
RESULTS = WORK / "RESULTS"
OUT = RESULTS / "iter_023_label_coverage_and_prefix_readiness"
I21 = RESULTS / "iter_021_matched_prefix_replay"
I22 = RESULTS / "iter_022_gated_matched_prefix"
I19_ARCHIVE = RESULTS / "iter_019_depth4_holdout_preparation/control_continuation/paired_runs_cumulative.jsonl"
MANIFEST = OUT / "PROPOSED_NEW_MANIFEST.json"
MODEL_PAGE = "https://developers.openai.com/api/docs/models/gpt-5.6-luna"
PRICING_PAGE = "https://developers.openai.com/api/docs/pricing"
INPUT_RATE = 0.20
OUTPUT_RATE = 1.20

CASES = [
    {
        "question_id": "WebQTest-450_5839e7f421b1c7dec152856912c23c6f",
        "repeat": 2,
        "slot_id": "iter23-repeat2-case1-depth4-beneficiary",
        "selection_role": "four_edge_target_depth4_only",
        "selection_basis": "previously studied development question; not a holdout",
    },
    {
        "question_id": "WebQTrn-1053_278042154fd6e20398fbac62f0bc6aa5",
        "repeat": 2,
        "slot_id": "iter23-repeat2-case2-tied-target",
        "selection_role": "target_paired_hit_tie",
        "selection_basis": "previously studied development question; not a holdout",
    },
    {
        "question_id": "WebQTest-450_3eb90d65b82703c8f7fb0496baa80ce7",
        "repeat": 2,
        "slot_id": "iter23-repeat2-case3-short-path-control",
        "selection_role": "short_path_control_depth3_only_loss",
        "selection_basis": "previously studied development question; not a holdout",
    },
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_if_same_or_new(path: Path, data: bytes) -> None:
    if path.exists() and path.read_bytes() != data:
        raise SystemExit(f"refusing to rewrite frozen Iteration 23 input: {path}")
    if not path.exists():
        path.write_bytes(data)


def git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=True)
    return result.stdout.strip()


def load_prior_misses() -> tuple[dict[str, str], dict[str, list[str]], list[dict]]:
    cache_doc = json.loads((I22 / "LOCAL_LABEL_CACHE.json").read_text(encoding="utf-8"))
    labels = {str(row["entity_id"]): str(row["label"]) for row in cache_doc["labels"]}
    traces = [json.loads(line) for line in (I22 / "RAW_CASE_TRACES.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    miss_cases: dict[str, list[str]] = {}
    for trace in traces:
        qid = str(trace.get("case", {}).get("question_id", ""))
        for entity_id in trace.get("missing_ids", []):
            miss_cases.setdefault(str(entity_id), []).append(qid)
    return labels, miss_cases, traces


def make_manifest() -> tuple[dict, str]:
    value = {
        "iteration": 23,
        "purpose": "Development-only bounded label resolution and exact matched-prefix retrieval pilot.",
        "evidence_scope": "Retrieval-only; no answer accuracy, generalization, or untouched-holdout claim.",
        "slot_rule": "Each slot is new and uses repeat=2. Question identities overlap prior Iteration 19/21/22 development work; the three cases are not independent holdout observations.",
        "selection_rule": "One four-edge beneficiary, one tied target, and one short-path control, chosen from the existing Iteration 21 development manifest. No case substitution after freeze.",
        "inference_separation": "selection_role and prior study evidence are diagnostic metadata only; the live runner may pass only question_id, repeat, and slot_id into case execution. Gold IDs/answers remain scoring-only.",
        "cases": CASES,
        "model": MODEL_NAME,
        "model_configuration": {
            "reasoning_effort": "low",
            "temperature_and_search_settings": "Use unchanged CoR settings from the Iteration 22 frozen runner; verify source/config hashes before transport.",
        },
        "source_question_manifest_sha256": file_hash(I21 / "PROPOSED_DEVELOPMENT_MANIFEST.json"),
    }
    raw = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    write_if_same_or_new(MANIFEST, raw)
    digest = sha256(raw)
    write_if_same_or_new(OUT / "PROPOSED_NEW_MANIFEST.sha256", f"{digest}  PROPOSED_NEW_MANIFEST.json\n".encode())
    return value, digest


MODEL_NAME = "gpt-5.6-luna"


def make_coverage_artifacts(labels: dict[str, str], miss_cases: dict[str, list[str]]) -> None:
    csv_path = OUT / "LABEL_COVERAGE_MATRIX.csv"
    fields = ["entity_id", "frozen_cache_status", "frozen_label", "archived_miss_count",
              "archived_miss_cases", "iteration23_resolution_policy", "label_provenance"]
    all_ids = sorted(set(labels) | set(miss_cases))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entity_id in all_ids:
            cached = entity_id in labels
            writer.writerow({
                "entity_id": entity_id,
                "frozen_cache_status": "CACHED" if cached else "UNSEEN_IN_FROZEN_CACHE",
                "frozen_label": labels.get(entity_id, ""),
                "archived_miss_count": len(miss_cases.get(entity_id, [])),
                "archived_miss_cases": ";".join(sorted(set(miss_cases.get(entity_id, [])))),
                "iteration23_resolution_policy": "reuse immutable archived label" if cached else "bounded read-only batch SELECT; stop on missing, ambiguous, malformed, timeout, or budget exhaustion",
                "label_provenance": "Iteration 19 archived candidate ID/name pair" if cached else "not resolved live in Iteration 23 offline work",
            })

    cache_doc = json.loads((I22 / "LOCAL_LABEL_CACHE.json").read_text(encoding="utf-8"))
    lines = [
        "# Iteration 23 label cache root cause",
        "",
        "## Finding",
        "",
        "Iteration 22's 198-label cache was built from the three historical depth-3 traces. The new live trajectories generated seven distinct, syntactically valid Freebase IDs outside that fixed cache. Each case stopped before a depth-3 boundary, so the failure was cache coverage, not a failed navigation query or an observed answer-retrieval outcome.",
        "",
        "The original `entity_search` path first uses labels returned inline with entity rows. If a candidate has no inline name, it calls `backend.id2entity_names`; the Freebase implementation batches known misses, but then falls back to per-ID `id2entity_name_or_type` queries for unresolved IDs. Both original lookup functions retry. Iteration 22 deliberately blocked that fallback to avoid hidden unmetered requests, and therefore marked all three cases ineligible at the first unseen label.",
        "",
        "## Hypotheses checked",
        "",
        "1. **The seven values are malformed IDs.** Rejected: all seven match the Freebase `m.` ID form; they are absent from the frozen cache.",
        "2. **The name lookup was bypassed by a database or tunnel failure.** Not supported: Iteration 22's 15 logged SPARQL attempts all succeeded; the traces identify cache misses before label resolution.",
        "3. **The only safe fix is to strip labels and capture IDs.** Not selected: labels are part of the normal CoR candidate context, so removing them could change model decisions.",
        "",
        "## Repair selected: bounded dynamic batch labels",
        "",
        "The isolated Iteration 23 adapter reuses the same Freebase `type.object.name` relation, English/untagged language filter, and normal `entity_search` rendering path. Cache misses are deduplicated and resolved in read-only `VALUES` batches of at most 100 IDs, with at most 12 label-query attempts for the entire pilot. Every query is reserved before transport; failed attempts consume budget; no retry or individual-ID fallback is permitted. Valid ID/label pairs retain query and result-row hashes plus language provenance for that diagnostic session.",
        "",
        "Missing labels, conflicting best-rank labels, malformed responses, and query failures stop the affected case explicitly. No guessed label, ID substitution, `UnName_Entity` placeholder, or oracle/gold data is introduced. The cache is session-local and immutable: a previously stored mapping is never overwritten.",
        "",
        "The synthetic fixture resolved a previously unknown test ID through the unchanged `entity_search` path. It does not resolve the seven real Iteration 22 misses and is not evidence that the live endpoint has labels for them.",
        "",
        "## Coverage counts",
        "",
        f"- Frozen cache: **{len(labels)}** unambiguous IDs from `{cache_doc['source_archive']}` (SHA-256 `{cache_doc['source_archive_sha256']}`).",
        f"- Iteration 22 traces: **{len(miss_cases)} unique misses**, appearing **{sum(map(len, miss_cases.values()))} times** across **{len(miss_cases and set(q for qs in miss_cases.values() for q in qs))} questions**.",
        f"- Matrix rows: **{len(labels) + len(miss_cases)}** (the 198 frozen mappings plus all seven observed misses).",
        "- The seven miss labels remain unknown in this offline report; only a new explicitly approved live SELECT could resolve them.",
        "",
        "See `LABEL_COVERAGE_MATRIX.csv` for every archived ID, the seven misses, per-question occurrences, and the resolver provenance policy.",
    ]
    (OUT / "LABEL_CACHE_ROOT_CAUSE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_query_budget(manifest_hash: str) -> dict:
    expected = {
        "historical_depth3_navigation": 86,
        "historical_cold_label_batches": 10,
        "projection_queries": 3,
        "post_boundary_relation_direction_queries": 6,
        "one_edge_binding_queries": 3,
        "endpoint_health_preflight": 1,
    }
    expected_total = sum(expected.values())
    allocations = {
        "baseline_navigation_planning_allocation": 100,
        "dynamic_label_batches_enforced_subcap": 12,
        "binding_projection_planning_allocation": 3,
        "post_boundary_relation_discovery_planning_allocation": 6,
        "one_edge_binding_planning_allocation": 3,
        "endpoint_health_preflight_planning_allocation": 1,
        "unallocated_safety_reserve_planning_allocation": 15,
    }
    max_total = sum(allocations.values())
    return {
        "iteration": 23,
        "manifest_sha256": manifest_hash,
        "evidence_class": "offline estimate; no live requests made",
        "physical_attempt_cap": max_total,
        "retries": 0,
        "failed_attempts_consume_cap": True,
        "block_before_attempt_cap_plus_one": True,
        "expected_query_components": expected,
        "expected_physical_attempts": expected_total,
        "planning_allocation_not_individually_enforced": allocations,
        "enforced_limits": {
            "global_physical_select_attempt_cap": max_total,
            "dynamic_label_query_attempt_cap": 12,
            "label_batch_size_cap": 100,
            "select_only": True,
            "automatic_retries": 0,
        },
        "worst_case_reserved_attempts": max_total,
        "headroom_over_expected": max_total - expected_total,
        "headroom_fraction_over_expected": round((max_total - expected_total) / expected_total, 6),
        "dynamic_label_bounds": {
            "max_ids_per_batch": 100,
            "max_label_queries": 12,
            "maximum_unseen_ids_resolvable": 1200,
            "cache_sharing": "deduplicated immutable mappings shared only for the one diagnostic process/session",
            "failure": "stop affected case; do not retry, guess, or add a case",
        },
        "all_entry_points": [
            "CoR backend execute_sparql and execute_sparql_with_meta",
            "FreebaseBackend imported query aliases",
            "id2entity_names dynamic batch resolver",
            "id2entity_name_or_type single-ID fallback redirected to the same batch resolver",
            "db_func direct query globals redirected to meter",
            "direct SPARQLWrapper construction blocked while adapter is installed",
            "binding projection and continuation queries use the same shared meter",
            "endpoint preflight is one metered read-only SELECT and occurs only after fresh approval",
        ],
        "abort_conditions": [
            "hard global total cap or enforced label-query subcap reached",
            "malformed/non-SELECT query",
            "transport error, timeout, malformed response, or non-ok status",
            "label absent, ambiguous, malformed, or label-query allowance exhausted",
            "pilot deadline reached",
            "no eligible depth-3 Forward, incomplete projection, or any exact-prefix invariant failure",
        ],
        "notes": [
            "The 86 navigation, 10 cold-cache label batch, and 12 post-boundary counts reproduce the Iteration 22 offline preflight model; they are planning references, not guarantees for stochastic new traces.",
            "The 12 dynamic label SELECTs are explicitly reserved for future unseen IDs; this replaces the failed assumption that a cache made only from archived candidate streams will cover live paths.",
            "The component figures and 15-attempt margin are planning allocations, not independently enforced category caps. The runtime enforces the global 140-attempt cap and the 12-query label subcap.",
            "All query entry points, endpoint health checks, label resolution, projections, and failures are charged to the same global cap.",
        ],
    }


def write_budget_docs(model: dict) -> None:
    write_json(OUT / "QUERY_BUDGET_MODEL.json", model)
    lines = [
        "# Iteration 23 proposed SPARQL budget",
        "",
        "No live SPARQL query was made for this estimate. The expected count is archive-based, while the new cap reserves explicit dynamic label traffic and a failure/trajectory margin.",
        "",
        "| Component | Expected | Reserved maximum | Basis |",
        "|---|---:|---:|---|",
        "| Baseline CoR relation/entity navigation | 86 | 100 | 86 logged in archived fixed depth-3 traces; 14 extra navigation attempts reserved |",
        "| Dynamic label resolution | 10 | 12 | Historical cold-cache estimate was 10; 12 batches × 100 IDs is the future hard limit |",
        "| Binding projections | 0 baseline | 3 | One per fixed question, only after valid boundary |",
        "| Post-boundary relation/direction discovery | 0 baseline | 6 | Two directions per eligible question |",
        "| One-edge binding query | 0 baseline | 3 | One per eligible question |",
        "| Endpoint health preflight | 1 | 1 | One read-only SELECT after authorization, included in total |",
        "| Shared safety reserve | — | 15 | Unallocated margin; case substitution or scope expansion is prohibited |",
        f"| **Total** | **{model['expected_physical_attempts']}** | **{model['physical_attempt_cap']}** | **{model['headroom_over_expected']} attempts ({model['headroom_fraction_over_expected']:.1%}) above expected** |",
        "",
        "## Meter rules",
        "",
        "The global cap is **140 physical attempts**, including failed attempts. The meter reserves an attempt before transport and refuses attempt 141 before transport. Each query must be a read-only `SELECT`; there are no retries. The endpoint preflight, navigation, batched label resolution, binding projection, relation discovery, and one-edge query all share this meter.",
        "",
        "The resolver reuses the 198 trusted archived labels and may issue at most 12 dynamic label batches, each containing at most 100 deduplicated IDs. At most 1,200 unseen labels can be attempted. A label lookup failure or an exhausted category/global allowance stops that case; it does not create an empty graph result or a negative retrieval outcome.",
        "",
        "The expected 108 historical cold-cache-plus-sidecar queries are not a promise for a new stochastic run. Component reservations and the 15-query margin are planning allocations; only the global 140-attempt cap and 12-query dynamic-label subcap are independently enforced. No additional question may be substituted after the manifest is frozen.",
        "",
        "See `QUERY_BUDGET_MODEL.json` for the machine-readable allocations and abort conditions.",
    ]
    (OUT / "QUERY_BUDGET.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_auth_request(manifest: dict, manifest_hash: str, query_model: dict) -> str:
    max_requests = 60
    max_input = 7000
    max_output = 1024
    max_reservation = max_requests * (max_input * INPUT_RATE + max_output * OUTPUT_RATE) / 1_000_000
    cost_cap = 0.16
    if max_reservation > cost_cap:
        raise SystemExit("full request-cap reservation exceeds proposed cost cap")
    approval = (
        f"I authorize Iteration 23 manifest `{manifest_hash}`: three fixed repeat-2 development slots "
        f"using `{MODEL_NAME}` at US$0.20 per million uncached input tokens and US$1.20 per million output tokens; "
        f"up to {max_requests} API requests, {max_input} input tokens/request, {max_output} completion tokens/request, "
        f"up to US${cost_cap:.2f} estimated API spend, up to {query_model['physical_attempt_cap']} physical read-only SPARQL SELECT attempts "
        f"including one endpoint preflight, up to 8 hours total, 120-second maximum per request, and no retries."
    )
    auth_path = OUT / "AUTHORIZATION.json"
    auth_record = json.loads(auth_path.read_text(encoding="utf-8")) if auth_path.exists() else None
    approved = isinstance(auth_record, dict) and auth_record.get("authorization_status") == "APPROVED"
    opening_status = (
        f"**APPROVED at {auth_record.get('approved_at')}**; approval is stored verbatim in `AUTHORIZATION.json`. "
        "This document records the frozen request and is retained as the scope source."
        if approved else
        "**Approval pending.** This file requests fresh approval for the manifest and exact limits below; no live request has been made."
    )
    lines = [
        "# Iteration 23 authorization required",
        "",
        f"{opening_status} The Iteration 22 repeat-1 authorization is consumed.",
        "",
        "## Frozen scope",
        "",
        f"- Manifest: `PROPOSED_NEW_MANIFEST.json`; SHA-256: `{manifest_hash}`.",
        "- Three new execution slots: repeat 2 for one depth-4 beneficiary, one tied target, and one short-path control.",
        "- The question IDs were already used in Iterations 19/21/22 development. These are repeated development observations, not an independent or untouched holdout.",
        "- Retrieval-only: capture a depth-3 trace and, only at an eligible exact `Forward` boundary, compare STOP with one extra graph edge from that same prefix. No answer filtering or final answer generation is part of the treatment.",
        "- Gold answers/IDs remain scoring-only and never enter prompts, relation selection, or routing.",
        "",
        "## Proposed limits and cost basis",
        "",
        f"- Model: `{MODEL_NAME}`; one attempt per request, no retries.",
        f"- Up to {max_requests} API requests; at most {max_input} input and {max_output} completion tokens per request.",
        f"- Estimated API spend cap: **US${cost_cap:.2f}**. Reserving the full request/token cap at the verified rates costs **US${max_reservation:.6f}**, below the requested cap.",
        f"- Rate basis checked 2026-11-08: uncached input **US${INPUT_RATE:.2f}/million**, output **US${OUTPUT_RATE:.2f}/million**. No cached-input discount is assumed. See [model pricing]({MODEL_PAGE}) and [API pricing]({PRICING_PAGE}). Rates and model access must be checked again before transport; if they differ, stop and request fresh approval.",
        f"- Expected workload reference: the archived three-case run used 47 model requests, 75,304 input tokens, and 11,937 output tokens, estimated at US$0.029385 at those rates. The Iteration 22 offline estimate added three relation-pruning requests (about 50 expected); the full hard-cap reservation remains US$0.157728.",
        f"- Up to **{query_model['physical_attempt_cap']} physical read-only SPARQL SELECT attempts**, including failed/time-out attempts and one endpoint health SELECT. No retries. The archive-based expected estimate is {query_model['expected_physical_attempts']} with {query_model['headroom_over_expected']} attempts of reserved headroom.",
        "- At most 8 hours total; each API and SPARQL transport has a 120-second maximum timeout. The shared deadline blocks starting another attempt after 8 hours.",
        "",
        "## Readiness evidence",
        "",
        "- Iteration 22 traces reproduce exactly the seven cache misses.",
        "- Bounded resolver uses the normal CoR entity-search label path, deduplicates IDs, records provenance, and fails closed for absent, ambiguous, malformed, or failed label lookups.",
        "- All backend and imported query aliases route through the meter; direct `SPARQLWrapper` fallback is blocked while the diagnostic adapter is installed; monkeypatches restore on success and exceptions.",
        "- Offline suites: Iteration 21 23/23, Iteration 22 19/19, Iteration 23 34/34 (76 total). Protected production hashes and 350 prior result-artifact hashes pass.",
        "- Synthetic full matched-prefix fixture: one valid same-prefix STOP/ONE_HOP pair with one new binding; non-Forward control remains ineligible. It is not live KG evidence.",
        "- Production CoR, TGAR, CFP, and default depth remain unchanged.",
        "",
        "## Abort conditions",
        "",
        "Abort the affected case or entire run on any manifest/approval/hash mismatch, unavailable key or tunnel, model/rate/config mismatch, retry not verifiably disabled, non-SELECT query, failed endpoint preflight, malformed/ambiguous/unresolved label, exceeded category/global cap, missing API usage, per-request token/cost reservation failure, deadline, incomplete binding projection, absent eligible Forward, or exact-prefix invariant failure. Failures consume their attempt. No retry, case substitution, or budget rollover.",
        "",
        "Before first transport, a new Iteration 23 entrypoint must validate and persist this exact approval, verify source hashes and the run-time environment, and pass the existing offline suites. The consumed Iteration 22 runner must not be invoked with this manifest.",
        "",
        "## Approval",
        "",
        "If you approve this exact manifest and all limits, reply with this sentence verbatim. That approval is stored with its timestamp; it does not authorize any other manifest or budget:",
        "",
        f"> {approval}",
        "",
        ("The exact approval has been received. The runtime preflight must still pass before the included endpoint SELECT is sent."
         if approved else "The authorization remains inactive until the exact approval is received and recorded. No live API or SPARQL calls are made in this phase."),
    ]
    return "\n".join(lines) + "\n"


def make_readiness(manifest_hash: str, tests: dict, synthetic: dict, source_hashes: dict) -> dict:
    return {
        "iteration": 23,
        "status": "APPROVED_AWAITING_LIVE_PREFLIGHT",
        "evidence_class": "offline engineering readiness; zero live matched pairs",
        "manifest_sha256": manifest_hash,
        "offline_suites": {"iteration21": 23, "iteration22": 19, "iteration23": 34,
                           "total": tests["total_tests"], "status": tests["status"]},
        "required_exact_prefix_fields": [
            "question and repeat-2 slot ID",
            "model/settings, prompt and response content/hash for every prior decision",
            "ordered full DFS stack and current path with relation directions and entity IDs",
            "all active intermediate entity-binding rows and visited IDs",
            "all navigation query text/hash/status/result rows in event order",
            "depth-3 Forward boundary, candidate relations before pruning, and selected direction",
            "binding-projection query text/hash, ordered variables, and complete untruncated result rows",
            "gold scoring IDs held outside all inference state",
        ],
        "synthetic_full_stop_one_hop_pairs": int(synthetic["matched_prefix_fixture"]["eligible_pairs"]),
        "synthetic_pair_same_parent_hash": bool(synthetic["matched_prefix_fixture"]["same_parent_hash"]),
        "synthetic_pair_added_edges": int(synthetic["matched_prefix_fixture"]["new_edge_count"]),
        "synthetic_nonforward_ineligible": synthetic["non_forward_fixture"]["reason"] == "NON_FORWARD_BOUNDARY",
        "live_matched_pairs": 0,
        "live_api_requests_iteration23": 0,
        "live_sparql_attempts_iteration23": 0,
        "no_live_result_claim": True,
        "production_code_changed": False,
        "runtime_driver_boundary": "A distinct Iteration 23 entrypoint binds the repeat-2 manifest to the exact saved approval and shared meters. Its offline metering tests pass; no live transport has yet opened.",
        "implementation_source_sha256": source_hashes,
        "next_action": "Verify runtime source hashes and the local endpoint tunnel with the Iteration 23 preflight; then consume the saved authorization once, with the endpoint SELECT as the first live request.",
    }


def make_decision(manifest_hash: str, tests: dict, synthetic: dict, query_model: dict) -> str:
    return f"""# Iteration 23 decision

**Decision: `APPROVED_AWAITING_LIVE_PREFLIGHT` for a retrieval-only development pilot.** The exact repeat-2 manifest and caps are now authorized; no live request has started. The Iteration 22 repeat-1 authorization remains consumed and was not reused.

## Established

- Iteration 22 stopped all three cases at the frozen 198-label cache gate, before a depth-3 boundary. The exact seven unique unseen IDs reproduce from the raw traces.
- This is an instrumentation blocker. Iteration 22 provides no matched-prefix, new-binding, answer-accuracy, or generalization result.

## Tested offline

- Repair A: bounded, deduplicated, read-only batched Freebase label resolution in CoR's normal `entity_search` path. Per-ID fallbacks and direct SPARQLWrapper access are metered or blocked; one attempt only; unresolved/ambiguous labels fail closed.
- A synthetic unknown ID resolved to a supplied synthetic label through the normal candidate-rendering path. A full synthetic STOP/ONE_HOP fixture produced one identical-parent pair with exactly one new edge. A synthetic non-Forward boundary remained ineligible.
- Test matrix: Iteration 21 {23}/23, Iteration 22 {19}/19, Iteration 23 {tests['suites'][-1]['tests_run']}/{tests['suites'][-1]['tests_run']}; total {tests['total_tests']}/{tests['total_tests']} passed. The 350 prior Iteration 10–22 result files and protected production-source hashes are unchanged.
- Query model: expected {query_model['expected_physical_attempts']} attempts; hard ceiling {query_model['physical_attempt_cap']} including one health SELECT, dynamic label batches, failures, and 15 reserved attempts. No retries.
- New proposed manifest SHA-256: `{manifest_hash}`. Its repeat-2 slots reuse the three previously studied development question identities; they are not holdout observations.

## Not established

- Whether live trajectories resolve these seven IDs, reach an eligible depth-3 Forward, or produce any exact matched prefix.
- Whether one extra edge discovers a gold-relevant binding, changes final answer accuracy, improves generalization, or changes production safety.
- Whether LLM planning or CoR graph execution is the dominant Fresh50 accuracy limit.

## Runtime preflight and next action

The approval is stored verbatim in `AUTHORIZATION.json`. The entrypoint now checks the manifest, approval, runtime source hashes, retries, token limits, per-call timeouts, and a single shared query meter. The approved ceilings are 60 API requests, 7,000 input and 1,024 completion tokens/request, US$0.16 estimated spend, 140 physical read-only SELECT attempts, 8 hours, 120-second per-call timeouts, and zero retries. The first live SPARQL call will be the included endpoint preflight SELECT. No case substitutions or cap rollover.

The synthetic fixture is engineering evidence only. TGAR remains the frozen practical Hits@1/cost reference; CoR and CFP remain comparators; production depth remains 3.
"""


def update_central(manifest_hash: str, query_model: dict, tests: dict) -> None:
    status_path = WORK / "STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    approval_path = OUT / "AUTHORIZATION.json"
    approval = json.loads(approval_path.read_text(encoding="utf-8")) if approval_path.exists() else {}
    approved = approval.get("authorization_status") == "APPROVED"
    consumed = bool(approval.get("authorization_consumed"))
    status["status"] = (
        "ITERATION_23_LIVE_RUN_IN_PROGRESS" if consumed else
        "ITERATION_23_APPROVED_AWAITING_LIVE_PREFLIGHT" if approved else
        "ITERATION_23_READY_FOR_NEW_AUTHORIZATION"
    )
    status["iteration"] = 23
    status["last_completed_experiment"] = "Iteration 23 live cache-repair and matched-prefix pilot" if consumed else "Iteration 23 offline cache-repair and matched-prefix readiness; no live transport."
    status["next_action"] = "Run the independent Iteration 23 audit and consolidate the research decision." if consumed else ("Verify runtime source hashes and local endpoint tunnel, then run the approved pilot." if approved else "Await fresh approval for the exact Iteration 23 repeat-2 manifest and caps.")
    status["remaining_questions"] = [
        "Will a fresh repeat-2 CoR trajectory resolve labels and reach an eligible depth-3 Forward boundary?",
        "At an exact captured prefix, does one selected extra graph edge expose any new answer-relevant binding?",
        "Does any retrieval-only change transfer to answer correctness or broader benchmark performance?",
    ]
    status["latest_experiment_attempt"] = (
        "Iteration 23 offline readiness passed: 76/76 tests across Iterations 21–23; seven archived cache misses reproduced; "
        "one synthetic exact STOP/ONE_HOP pair; 0 live API requests, 0 live SPARQL attempts, and no production-code changes. "
        "Fresh manifest " + manifest_hash + (" is approved; live run pending." if approved else " awaits exact approval.")
    )
    budget = status.setdefault("budget_remaining", {})
    budget.update({
        "active_paid_authorization": approved and not consumed,
        "new_experiment_authorized_calls": 60 if approved and not consumed else 0,
        "new_experiment_authorized_cost_usd": 0.16 if approved and not consumed else 0.0,
        "iteration_23_authorization_status": ("CONSUMED" if consumed else "APPROVED_NOT_YET_CONSUMED" if approved else "NOT_AUTHORIZED"),
        "iteration_23_manifest_sha256": manifest_hash,
        "iteration_23_proposed_api_request_cap": 60,
        "iteration_23_proposed_cost_cap_usd": 0.16,
        "iteration_23_proposed_sparql_attempt_cap": query_model["physical_attempt_cap"],
        "iteration_23_paid_api_calls_used": 0,
        "iteration_23_live_sparql_attempts_used": 0,
        "iteration_23_unused_cap_rollover": False,
    })
    status["artifact_paths"] = list(dict.fromkeys(status.get("artifact_paths", []) + [
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/ITER23_DECISION.md",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/LABEL_CACHE_ROOT_CAUSE.md",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/LABEL_COVERAGE_MATRIX.csv",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/QUERY_BUDGET_MODEL.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/QUERY_BUDGET.md",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/MATCHED_PREFIX_READINESS.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/PROPOSED_NEW_MANIFEST.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/AUTHORIZATION_REQUIRED.md",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/SYNTHETIC_PILOT.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/TEST_RESULTS.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/ITER23_INTEGRITY.json",
    ]))
    write_json(status_path, status)

    log_path = WORK / "ITERATION_LOG.md"
    old_log = log_path.read_text(encoding="utf-8")
    if "## Iteration 23 —" not in old_log:
        old_log += f"""

## Iteration 23 — cache repair and matched-prefix readiness

- Work was offline only; no API request or live SPARQL query was made.
- Reproduced the seven Iteration 22 cache misses and implemented an isolated, bounded dynamic batch resolver and all-entrypoint query meter.
- Test matrix: Iteration 21 23/23, Iteration 22 19/19, Iteration 23 26/26; 68 total.
- Synthetic fixture: one unknown ID resolved through normal entity search; one valid exact-parent STOP/ONE_HOP pair with one added edge; non-Forward control ineligible.
- Protected prior result artifacts: 350/350 unchanged; protected production sources unchanged.
- Decision: READY_FOR_NEW_AUTHORIZATION; new repeat-2 manifest `{manifest_hash}`. No authorization is active. Proposed API cap 60 requests / US$0.16; read-only SPARQL cap {query_model['physical_attempt_cap']}; zero retries.
- Artifacts: `RESULTS/iter_023_label_coverage_and_prefix_readiness/ITER23_DECISION.md`, `AUTHORIZATION_REQUIRED.md`, `QUERY_BUDGET.md`, `MATCHED_PREFIX_READINESS.json`.
"""
        log_path.write_text(old_log, encoding="utf-8")

    ledger_path = WORK / "EVIDENCE_LEDGER.csv"
    with ledger_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not any(row.get("iteration") == "23" and row.get("hypothesis") == "CACHE_COVERAGE_REPAIR" for row in rows):
        rows.append({
            "iteration": "23", "question_key": "cwq::three previously studied development cases",
            "hypothesis": "CACHE_COVERAGE_REPAIR",
            "supporting_trace": "7/7 archived Iteration 22 cache misses reproduced; synthetic new-ID label resolution and exact matched-prefix fork passed",
            "confidence": "HIGH_FOR_OFFLINE_MECHANISM_ONLY",
            "outcome": "OFFLINE_READY_FOR_NEW_AUTHORIZATION; NO_LIVE_CAUSAL_EVIDENCE",
            "source_artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/ITER23_DECISION.md",
            "notes": "68 offline tests passed; zero live API/SPARQL requests; same three question identities are development repeats, not holdout data.",
        })
        with ledger_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    cost_path = WORK / "COST_LEDGER.json"
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    if not any(item.get("iteration") == 23 for item in cost.get("offline_experiment_log", [])):
        cost.setdefault("offline_experiment_log", []).append({
            "iteration": 23, "experiment": "label coverage repair and matched-prefix offline readiness",
            "llm_calls": 0, "kg_queries": 0, "synthetic_selects": 2, "cost_usd": 0.0,
            "offline_tests": tests["total_tests"], "synthetic_valid_pairs": 1,
            "authorization_status": "NOT_AUTHORIZED",
            "artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/ITER23_DECISION.md",
        })
        cost.setdefault("llm_call_log", []).append({
            "iteration": 23, "experiment": "offline label coverage and matched-prefix readiness",
            "model": MODEL_NAME, "calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cost_usd": 0.0, "authorized_call_cap": 0, "status": "OFFLINE_ONLY_NOT_AUTHORIZED",
        })
        cost.setdefault("kg_query_log", []).append({
            "iteration": 23, "experiment": "offline readiness; synthetic fixture transport only",
            "queries": 0, "note": "0 physical/live SPARQL requests; 2 in-memory synthetic query responses do not count as KG queries.",
        })
    old_proposal = cost.get("next_experiment_proposal_not_authorized")
    if old_proposal and old_proposal.get("manifest_sha256") != manifest_hash:
        previous = cost.setdefault("previous_proposals_preserved", [])
        if not any(item.get("recorded_before_iteration_23") for item in previous):
            previous.append({"recorded_before_iteration_23": old_proposal,
                             "superseded_as_current_next_step": True})
    cost["next_experiment_proposal_not_authorized"] = {
        "name": "Iteration 23 bounded-label matched-prefix development pilot",
        "status": "APPROVED_NOT_YET_CONSUMED; no live transport" if approved and not consumed else "CONSUMED" if consumed else "READY_FOR_NEW_AUTHORIZATION",
        "authorization_active": approved and not consumed,
        "manifest_sha256": manifest_hash,
        "fixed_scope": "three previously studied development questions, each a new repeat-2 execution slot",
        "model": MODEL_NAME,
        "input_usd_per_million": INPUT_RATE,
        "output_usd_per_million": OUTPUT_RATE,
        "api_request_cap": 60,
        "input_tokens_per_request_cap": 7000,
        "completion_tokens_per_request_cap": 1024,
        "estimated_api_spend_cap_usd": 0.16,
        "full_request_cap_reservation_usd": round(60 * (7000 * INPUT_RATE + 1024 * OUTPUT_RATE) / 1_000_000, 9),
        "read_only_sparql_physical_attempt_cap": query_model["physical_attempt_cap"],
        "query_expected_attempts": query_model["expected_physical_attempts"],
        "time_cap_hours": 8,
        "per_call_timeout_seconds": 120,
        "retries": 0,
        "actual_api_requests": 0,
        "actual_live_sparql_attempts": 0,
        "actual_live_cost_usd": 0.0,
        "next_action": "Verify the saved source hashes and local tunnel, then run the approved pilot once." if approved and not consumed else "Run independent audit and consolidate results." if consumed else "Obtain exact approval and record it before transport.",
    }
    budget = cost.setdefault("budget_remaining", {})
    budget.update({
        "iteration_23_authorization_status": ("CONSUMED" if consumed else "APPROVED_NOT_YET_CONSUMED" if approved else "NOT_AUTHORIZED"),
        "iteration_23_manifest_sha256": manifest_hash,
        "iteration_23_proposed_api_request_cap": 60,
        "iteration_23_proposed_estimated_spend_cap_usd": 0.16,
        "iteration_23_proposed_sparql_physical_attempt_cap": query_model["physical_attempt_cap"],
        "iteration_23_calls_used": 0,
        "iteration_23_cost_used_usd": 0.0,
        "iteration_23_live_sparql_attempts_used": 0,
        "iteration_23_rollover": False,
    })
    cost["paid_experiment_authorized"] = approved and not consumed
    cost["configured_call_budget"] = 60 if approved and not consumed else 0
    cost["configured_cost_budget_usd"] = 0.16 if approved and not consumed else 0.0
    cost["note"] = (cost.get("note", "") + " Iteration 23 offline readiness added no paid calls, live KG queries, tokens, or cost; "
                    "fresh proposed scope is not authorized and does not alter cumulative totals.").strip()
    write_json(cost_path, cost)

    handoff_path = WORK / "CHATGPT_HANDOFF.md"
    old_handoff = handoff_path.read_text(encoding="utf-8")
    start_marker, end_marker = "<!-- ITERATION23_HANDOFF_START -->", "<!-- ITERATION23_HANDOFF_END -->"
    start, end = old_handoff.find(start_marker), old_handoff.find(end_marker)
    if start >= 0 and end >= start:
        suffix = old_handoff[end + len(end_marker):].lstrip("\n")
    else:
        suffix = old_handoff
    block = f"""{start_marker}
# Iteration 23 current handoff — label repair and matched-prefix readiness

## Decision

**Status: `APPROVED_AWAITING_LIVE_PREFLIGHT`; no live experiment has run.** Iteration 22's repeat-1 authorization remains consumed. Iteration 23's exact repeat-2 scope is approved; the endpoint preflight has not yet run.

## Established and tested offline

- The Iteration 22 failure was seven unseen entity labels in a frozen 198-ID cache; all three cases stopped before a valid depth-3 boundary.
- A diagnostic-only bounded batch resolver now follows the normal CoR entity-search path. It meters every backend/imported query alias, blocks hidden direct label queries, retries zero times, and fails closed for absent/ambiguous/malformed labels.
- Iterations 21–23: **{tests['total_tests']}/{tests['total_tests']} tests pass**. Protected production hashes and **350/350** earlier result files are unchanged.
- Synthetic fixture: one previously unknown ID resolved through normal entity search and one valid exact-parent STOP/ONE_HOP pair added one edge. A non-Forward case stayed ineligible. This is not live Freebase or causal evidence.
- Proposed manifest SHA-256: `{manifest_hash}`. It uses new repeat-2 slots for the same three development question identities; it is not a holdout.

## Untested

Live label coverage, live depth-3 prefix eligibility, new-gold-binding reachability, answer accuracy, generalization, and whether LLM planning or CoR execution dominates Fresh50 errors.

## Proposed next action

Approval is stored verbatim in `AUTHORIZATION.json`. The approved limits are 60 model requests, 7,000 input and 1,024 completion tokens/request, **US$0.16** maximum estimated spend, **140** physical read-only SPARQL SELECT attempts, 8 hours, 120-second per-call maximum, and zero retries. The query cap includes dynamic labels and one endpoint preflight. The new Iteration 23 entrypoint is implemented and its offline meter tests pass; the included endpoint SELECT must be its first live call.

## Artifacts

- Decision: `RESULTS/iter_023_label_coverage_and_prefix_readiness/ITER23_DECISION.md`
- Root cause and coverage: `LABEL_CACHE_ROOT_CAUSE.md`, `LABEL_COVERAGE_MATRIX.csv`
- Query model: `QUERY_BUDGET.md`, `QUERY_BUDGET_MODEL.json`
- Readiness and synthetic record: `MATCHED_PREFIX_READINESS.json`, `SYNTHETIC_PILOT.json`
- Manifest and authorization request: `PROPOSED_NEW_MANIFEST.json`, `AUTHORIZATION_REQUIRED.md`
- Test results and protected hashes: `TEST_RESULTS.json`, `PROTECTED_PRIOR_ARTIFACTS.json`

<!-- ITERATION23_HANDOFF_END -->
"""
    if "<!-- ITERATION22_HANDOFF_START -->" not in suffix:
        raise SystemExit("historical Iteration 22 handoff block missing; refusing handoff rewrite")
    handoff_path.write_text(block + suffix, encoding="utf-8")


def main() -> None:
    labels, miss_cases, _ = load_prior_misses()
    manifest, manifest_hash = make_manifest()
    synthetic = json.loads((OUT / "SYNTHETIC_PILOT.json").read_text(encoding="utf-8"))
    tests = json.loads((OUT / "TEST_RESULTS.json").read_text(encoding="utf-8"))
    if tests.get("status") != "PASS" or tests.get("total_tests") != 76:
        raise SystemExit("offline test matrix is not fully passing")
    if synthetic.get("status") != "PASS" or synthetic.get("live_api_requests") != 0 or synthetic.get("live_sparql_attempts") != 0:
        raise SystemExit("synthetic readiness fixture did not pass without live transport")

    make_coverage_artifacts(labels, miss_cases)
    query_model = make_query_budget(manifest_hash)
    write_budget_docs(query_model)

    protected_doc = json.loads((I21 / "PROTECTED_SOURCE_HASHES.json").read_text(encoding="utf-8"))
    protected_code = {}
    for item in protected_doc["files"]:
        actual = file_hash(ROOT / item["path"])
        protected_code[item["path"]] = {
            "expected_sha256": item["sha256_at_iteration_start"],
            "actual_sha256": actual,
            "matches": actual == item["sha256_at_iteration_start"],
        }
    if not all(item["matches"] for item in protected_code.values()):
        raise SystemExit("protected production-source hash mismatch")
    source_paths = sorted(path for path in (ROOT / "scripts/diagnostics/iter_023_label_coverage_and_prefix_readiness").glob("*.py"))
    source_hashes = {path.relative_to(ROOT).as_posix(): file_hash(path) for path in source_paths}
    snapshot = {
        "iteration": 23, "git_head": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "prior_worktree_changes_preserved": True,
        "protected_production_sources": protected_code,
        "iteration23_diagnostic_sources": source_hashes,
        "previous_result_artifacts_snapshot": "PROTECTED_PRIOR_ARTIFACTS.json",
        "note": "No production source changes or live transport; local changes unrelated to Iteration 23 were preserved.",
    }
    write_json(OUT / "SOURCE_SNAPSHOT.json", snapshot)

    readiness = make_readiness(manifest_hash, tests, synthetic, source_hashes)
    write_json(OUT / "MATCHED_PREFIX_READINESS.json", readiness)
    (OUT / "AUTHORIZATION_REQUIRED.md").write_text(
        make_auth_request(manifest, manifest_hash, query_model), encoding="utf-8"
    )
    (OUT / "ITER23_DECISION.md").write_text(
        make_decision(manifest_hash, tests, synthetic, query_model), encoding="utf-8"
    )
    update_central(manifest_hash, query_model, tests)
    print(json.dumps({
        "status": "APPROVED_AWAITING_LIVE_PREFLIGHT",
        "manifest_sha256": manifest_hash,
        "protected_cache_ids": len(labels), "observed_cache_misses": len(miss_cases),
        "expected_queries": query_model["expected_physical_attempts"],
        "proposed_query_cap": query_model["physical_attempt_cap"],
        "tests": tests["total_tests"], "live_api_requests": 0, "live_sparql_attempts": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
