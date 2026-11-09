"""Regenerate Iteration 23's human handoff/ledgers from the audited live record.

Offline only. Does not contact OpenAI or Virtuoso and never rewrites raw journals.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "outputs/experiments/fresh50_continuous_agent"
OUT = WORK / "RESULTS/iter_023_label_coverage_and_prefix_readiness"
MANIFEST_SHA = "ff57d84266295e7778b00e733fdba9e8a9232f4eed87971f7b0e029cefc331f2"
UNRESOLVED = ["m.04st848", "m.04stk8g", "m.04stpzt", "m.04strb6", "m.0hz9k44", "m.0hzdjww", "m.0ndvpvf"]
RESULT = json.loads((OUT / "PILOT_RESULTS.json").read_text())
AUDIT = json.loads((OUT / "PILOT_INTEGRITY.json").read_text())
AUTH = json.loads((OUT / "AUTHORIZATION.json").read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def update_readiness() -> None:
    path = OUT / "MATCHED_PREFIX_READINESS.json"
    old = json.loads(path.read_text())
    old.update({
        "status": "LIVE_PILOT_COMPLETE_REPAIR_NOT_READY",
        "evidence_class": "three-slot repeat-2 live retrieval-only pilot; no eligible prefixes; no causal retrieval result",
        "offline_suites": {"iteration21": 23, "iteration22": 19, "iteration23": 36, "total": 78, "status": "PASS"},
        "live_matched_pairs": 0,
        "live_api_requests_iteration23": 7,
        "live_api_metered_requests": 6,
        "live_api_uncertain_requests": 1,
        "live_sparql_attempts_iteration23": 19,
        "live_label_query_attempts": 3,
        "live_eligible_cases": 0,
        "live_unresolved_label_ids": UNRESOLVED,
        "live_label_rows_with_bound_names": 0,
        "all_slots_ineligible_before_depth3_boundary": True,
        "binding_projection_queries": 0,
        "stop_one_hop_forks": 0,
        "new_entity_or_gold_reachability_observed": False,
        "actual_rate_based_api_cost_usd": 0.003738,
        "uncertain_reserved_api_cost_usd": 0.0026288,
        "maximum_rate_based_api_exposure_usd": 0.0063668,
        "protocol_deviations": {
            "same_prompt_retries_after_interruption": 1,
            "wrapper_or_sdk_retries": 0,
            "maximum_observed_api_call_seconds": 163,
            "authorized_maximum_api_call_seconds": 120,
            "no_retry_and_timeout_caps_compliant": False
        },
        "production_code_changed": False,
        "runtime_driver_boundary": "The approved live runtime reused its one successful preflight and completed the three slots. Post-run audit found the label blocker plus elapsed-time and repeated-prompt protocol deviations; the original pre-live transport statement is superseded.",
        "post_live_audit_sources_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scripts/diagnostics/iter_023_label_coverage_and_prefix_readiness/run_iter23_pilot.py",
                "scripts/diagnostics/iter_023_label_coverage_and_prefix_readiness/audit_live_pilot.py",
                "scripts/diagnostics/iter_023_label_coverage_and_prefix_readiness/finalize_live_artifacts.py",
            )
        },
        "next_action": "Repair and validate label provenance plus hard wall-clock/no-repeat safeguards offline. Any further transport needs a new manifest and fresh authorization; do not rerun these consumed slots."
    })
    old["no_live_result_claim"] = True
    write_json(path, old)


def update_label_matrix() -> None:
    path = OUT / "LABEL_COVERAGE_MATRIX.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    slot_by_id: dict[str, list[str]] = {entity_id: [] for entity_id in UNRESOLVED}
    slot_by_id["m.04stk8g"] = [
        "iter23-repeat2-case1-depth4-beneficiary",
        "iter23-repeat2-case3-short-path-control",
    ]
    slot_by_id["m.0ndvpvf"] = ["iter23-repeat2-case2-tied-target"]
    slot_by_id["m.04st848"] = ["iter23-repeat2-case3-short-path-control"]
    slot_by_id["m.04stpzt"] = ["iter23-repeat2-case3-short-path-control"]
    slot_by_id["m.04strb6"] = ["iter23-repeat2-case3-short-path-control"]
    slot_by_id["m.0hz9k44"] = ["iter23-repeat2-case3-short-path-control"]
    slot_by_id["m.0hzdjww"] = ["iter23-repeat2-case3-short-path-control"]
    extra = ["iteration23_live_slots", "iteration23_live_lookup_status", "iteration23_live_batch_count", "iteration23_bound_name_rows"]
    for row in rows:
        entity_id = row["entity_id"]
        slots = slot_by_id.get(entity_id, [])
        row["iteration23_live_slots"] = "|".join(slots)
        row["iteration23_live_lookup_status"] = "UNRESOLVED_NO_BOUND_TYPE_OBJECT_NAME" if slots else "NOT_A_LIVE_CACHE_MISS"
        row["iteration23_live_batch_count"] = str(len(slots))
        row["iteration23_bound_name_rows"] = "0" if slots else ""
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def update_ledgers() -> None:
    status_path = WORK / "STATUS.json"
    status = json.loads(status_path.read_text())
    status["status"] = "ITERATION_23_COMPLETE_REPAIR_NOT_READY"
    status["last_completed_experiment"] = "Iteration 23 live repeat-2 label-resolution pilot; 0/3 eligible and 0 matched-prefix pairs"
    status["remaining_questions"] = [
        "Are the seven IDs missing English/untagged type.object.name values in this endpoint, or is the label data/filter inconsistent with the intended CoR renderer?",
        "Can the label path be repaired without substituting aliases or changing CoR model inputs?",
        "How can a future runner enforce an elapsed 120-second hard deadline and prevent same-prompt reattempt after interruption?",
        "At a valid exact depth-3 prefix, does one additional edge expose any new answer-relevant binding?"
    ]
    status["next_action"] = "Offline: diagnose missing type.object.name bindings and hard timeout/no-repeat behavior. Iteration 23 authorization is consumed; prepare a fresh manifest and exact authorization only after repair is ready."
    status["latest_experiment_attempt"] = (
        "Iteration 23 pilot completed for three fixed repeat-2 development slots: 0/3 eligible, 0 matched prefixes; "
        "7 API attempts (6 metered, 1 uncertain), 11,136 input/1,259 completion tokens, $0.003738 metered plus "
        "$0.0026288 reserved uncertainty, and 19 successful SELECT attempts. Seven distinct labels remained unbound. "
        "Audit flags one same-prompt reattempt after operator interruption and a 163-second API call against the 120-second cap."
    )
    status["budget_remaining"].update({
        "active_paid_authorization": False,
        "new_experiment_authorized_calls": 0,
        "new_experiment_authorized_cost_usd": 0.0,
        "cumulative_llm_calls": 1464,
        "cumulative_input_tokens": 1891832,
        "cumulative_output_tokens": 361682,
        "cumulative_llm_cost_usd": 0.8123848,
        "read_only_kg_queries": 2271,
        "iteration_23_authorization_status": "CONSUMED_WITH_PROTOCOL_DEVIATIONS_REPAIR_NOT_READY",
        "iteration_23_api_requests_used": 7,
        "iteration_23_metered_input_tokens": 11136,
        "iteration_23_metered_completion_tokens": 1259,
        "iteration_23_rate_based_cost_usd": 0.003738,
        "iteration_23_uncertain_reserved_cost_usd": 0.0026288,
        "iteration_23_maximum_rate_based_exposure_usd": 0.0063668,
        "iteration_23_sparql_attempts_used": 19,
        "iteration_23_same_prompt_retries": 1,
        "iteration_23_max_observed_api_call_seconds": 163,
        "iteration_23_unused_authorization_rollover": False,
        "note": "Iteration 23 authorization is consumed. The three repeat-2 slots produced no eligible prefix. Actual rate-based API cost was $0.003738; one interrupted attempt has $0.0026288 uncertain reserved cost. A same-prompt reattempt and 163-second call exceeded the no-retry/120-second conditions. No headroom rolls forward."
    })
    status["artifact_paths"] = sorted(set(status["artifact_paths"] + [
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/PILOT_INTEGRITY.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/CASE_USAGE_AUDIT.csv",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/POST_RUN_ACCOUNTING_CORRECTIONS.json",
        "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/ITER23_INTEGRITY_PRELIVE.json",
        "scripts/diagnostics/iter_023_label_coverage_and_prefix_readiness/audit_live_pilot.py",
        "scripts/diagnostics/iter_023_label_coverage_and_prefix_readiness/finalize_live_artifacts.py"
    ]))
    write_json(status_path, status)

    cost_path = WORK / "COST_LEDGER.json"
    cost = json.loads(cost_path.read_text())
    cost.update({
        "llm_calls": 1464,
        "input_tokens": 1891832,
        "output_tokens": 361682,
        "cost_usd": 0.8123848,
        "read_only_kg_queries": 2271,
        "paid_experiment_authorized": False,
        "uncertain_reserved_api_cost_usd_iteration23": 0.0026288,
        "iteration23_maximum_rate_based_exposure_usd": 0.0063668,
        "note": "Cumulative cost_usd includes rate-based metered usage. Iteration 23 adds $0.003738 metered cost and $0.0026288 uncertain reserved exposure from one interrupted API attempt. Its approved authorization is consumed with protocol deviations; no unused cap rolls forward."
    })
    if not any(item.get("iteration") == 23 and item.get("experiment") == "authorized repeat-2 live label-resolution/matched-prefix pilot"
               for item in cost["kg_query_log"]):
        cost["kg_query_log"].append({
        "iteration": 23, "experiment": "authorized repeat-2 live label-resolution/matched-prefix pilot",
        "queries": 19, "recorded_utc": "2026-11-08T04:35:24Z",
        "note": "1 successful endpoint preflight + 15 navigation + 3 batched label SELECTs; all 19 SELECT attempts returned ok. The 3 label batches returned entity rows but zero bound type.object.name values for seven distinct IDs; 0/3 eligible and no prefix projection/fork."
        })
    if not any(item.get("iteration") == 23 and item.get("experiment") == "authorized repeat-2 live label-resolution/matched-prefix pilot"
               for item in cost["llm_call_log"]):
        cost["llm_call_log"].append({
        "iteration": 23, "experiment": "authorized repeat-2 live label-resolution/matched-prefix pilot",
        "model": "gpt-5.6-luna", "calls": 7, "input_tokens": 11136,
        "output_tokens": 1259, "cost_usd": 0.003738,
        "uncertain_reserved_cost_usd": 0.0026288,
        "maximum_rate_based_exposure_usd": 0.0063668,
        "authorized_call_cap": 60, "authorized_cost_cap_usd": 0.16,
        "authorized_sparql_query_cap": 140, "sparql_queries": 19,
        "wrapper_sdk_retries": 0, "same_prompt_retries": 1,
        "max_observed_call_seconds": 163, "authorized_call_timeout_seconds": 120,
        "status": "COMPLETED_0_OF_3_ELIGIBLE_WITH_PROTOCOL_DEVIATIONS",
        "source": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/PILOT_INTEGRITY.json"
        })
    budget = cost["budget_remaining"]
    budget.update({
        "new_experiment_authorized_calls": 0,
        "new_experiment_authorized_cost_usd": 0.0,
        "iteration_23_authorization_status": "CONSUMED_WITH_PROTOCOL_DEVIATIONS_REPAIR_NOT_READY",
        "iteration_23_manifest_sha256": MANIFEST_SHA,
        "iteration_23_calls_used": 7,
        "iteration_23_metered_cost_used_usd": 0.003738,
        "iteration_23_uncertain_reserved_cost_usd": 0.0026288,
        "iteration_23_sparql_attempts_used": 19,
        "iteration_23_same_prompt_retries": 1,
        "iteration_23_max_observed_call_seconds": 163,
        "iteration_23_unused_request_headroom_expired": 53,
        "iteration_23_unused_estimated_cost_headroom_expired_usd": 0.1536332,
        "iteration_23_unused_sparql_headroom_expired": 121,
        "iteration_23_rollover": False,
        "note": "Iteration 23's 60-request/$0.16/140-SELECT authorization is consumed. 7 API attempts and 19 SELECT attempts were used; remaining request/query/dollar headroom expired and cannot roll over. The two recorded protocol deviations are documented in PILOT_INTEGRITY.json."
    })
    proposal = cost["next_experiment_proposal_not_authorized"]
    proposal.update({
        "status": "CONSUMED_WITH_PROTOCOL_DEVIATIONS; REPAIR_NOT_READY",
        "authorization_active": False,
        "actual_api_requests": 7,
        "actual_metered_api_cost_usd": 0.003738,
        "uncertain_reserved_api_cost_usd": 0.0026288,
        "actual_sparql_attempts": 19,
        "eligible_cases": 0,
        "live_matched_prefix_pairs": 0,
        "same_prompt_retries": 1,
        "max_observed_api_call_seconds": 163,
        "next_action": "Repair label data and hard wall-clock/no-repeat safeguards offline; any further transport requires a fresh manifest and approval."
    })
    write_json(cost_path, cost)


def update_evidence_and_log() -> None:
    evidence = WORK / "EVIDENCE_LEDGER.csv"
    with evidence.open(newline="", encoding="utf-8") as f:
        existing = list(csv.DictReader(f))
        fields = list(existing[0])
    if not any(row.get("supporting_trace") == "iter23-live-case-usage-audit" for row in existing):
        with evidence.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow({
                "iteration": 23,
                "question_key": "cwq::three fixed repeat-2 development slots",
                "hypothesis": "LIVE_LABEL_COVERAGE_AND_MATCHED_PREFIX_READINESS",
                "supporting_trace": "iter23-live-case-usage-audit",
                "confidence": "HIGH_FOR_OBSERVED_LABEL_LOOKUP_OUTCOME; NO_CAUSAL_PREFIX_EVIDENCE",
                "outcome": "0_OF_3_ELIGIBLE; 7_UNRESOLVED_IDS; 0_MATCHED_PAIRS",
                "source_artifact": "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_023_label_coverage_and_prefix_readiness/PILOT_INTEGRITY.json",
                "notes": "19 successful SELECTs including one reused preflight; three OPTIONAL name batches returned 1,1,6 entity rows with 0 bound type.object.name values. 7 API attempts, 6 metered and one uncertain; one same-prompt repeat and a 163-second call violate the approved no-retry/120-second conditions."
            })
    log = WORK / "ITERATION_LOG.md"
    text = log.read_text(encoding="utf-8")
    marker = "## Iteration 23 — authorized live repeat-2 pilot"
    if marker not in text:
        text += """


## Iteration 23 — authorized live repeat-2 pilot

- Manifest: `ff57d84266295e7778b00e733fdba9e8a9232f4eed87971f7b0e029cefc331f2`; three fixed repeat-2 development slots, same previously studied question identities. Authorization is consumed; no request, cost, or query headroom rolls over.
- Outcome: all 3 cases were processed, but **0/3 became eligible**. The seven distinct live IDs were unresolved by the normal `type.object.name` lookup. Three batched label queries returned 1, 1, and 6 entity rows, with zero bound label values. No depth-3 boundary, binding projection, STOP/ONE_HOP pair, new binding, or gold-reachability result exists.
- Usage: 7 Luna API attempts; 6 metered (11,136 input / 1,259 completion tokens, $0.003738) and one uncertain reserved attempt ($0.0026288); maximum rate-based exposure $0.0063668. SPARQL: 19 successful read-only SELECT attempts (1 preflight, 15 navigation, 3 labels). No SDK/wrapper retry metadata was recorded.
- Protocol deviations: request 1 lasted 163 seconds against the 120-second maximum. A local Ctrl-C interrupted request 4; request 5 immediately reused its prompt hash. Conservatively counted as one repeated-prompt retry, so strict no-retry compliance failed. No further transport was made.
- Integrity: 78/78 offline tests passed; 350/350 prior artifacts and 4/4 protected production sources are unchanged. `LIVE_RUN_STATE.json` was corrected to remove the first attempt's stale stop reason; its original state remains in `LIVE_RUN_STATE_FIRST_STOP.json`.
- Decision: `REPAIR_NOT_READY`. Do not infer a negative retrieval or accuracy result. Repair label availability/provenance and enforce elapsed hard timeout plus call-once semantics offline; any further live work requires a fresh manifest and authorization.
- Audit: `RESULTS/iter_023_label_coverage_and_prefix_readiness/PILOT_INTEGRITY.json`; per-case metering: `CASE_USAGE_AUDIT.csv`; raw evidence remains in the request/query journals and case traces.
"""
        log.write_text(text, encoding="utf-8")


def update_handoff() -> None:
    path = WORK / "CHATGPT_HANDOFF.md"
    text = path.read_text(encoding="utf-8")
    start = text.index("<!-- ITERATION23_HANDOFF_START -->")
    end = text.index("<!-- ITERATION23_HANDOFF_END -->") + len("<!-- ITERATION23_HANDOFF_END -->")
    block = """<!-- ITERATION23_HANDOFF_START -->
# Iteration 23 current handoff — live label-resolution pilot

## Decision

**`REPAIR_NOT_READY`.** The exact authorized repeat-2 pilot completed all three fixed development slots, but **0/3 were eligible** and no live matched prefix or STOP/ONE_HOP pair exists. The authorization is consumed. No headroom rolls forward and no further transport should occur under this approval.

## Established and tested offline

- Iteration 22's blocker was seven IDs missing from its frozen 198-ID cache. The bounded, metered resolver, synthetic unknown-label flow, exact-prefix serializer, and synthetic STOP/ONE_HOP fork passed offline.
- Iterations 21–23 offline suites: **78/78 passed**. **350/350** prior result artifacts and **4/4** protected production sources remain unchanged.
- The synthetic matched-prefix fixture is engineering evidence only; it is not live causal evidence.

## Live Iteration 23 result

- Frozen manifest `ff57d84266295e7778b00e733fdba9e8a9232f4eed87971f7b0e029cefc331f2`; three repeat-2 slots reuse previously studied question identities and are development data, not a holdout.
- All cases ended `INELIGIBLE / LABEL_RESOLUTION_STOP`. The seven distinct unresolved IDs were `m.04stk8g`, `m.0ndvpvf`, `m.04st848`, `m.0hzdjww`, `m.0hz9k44`, `m.04stpzt`, and `m.04strb6`.
- Three batched `type.object.name` SELECTs returned entity rows (1, 1, and 6), but **zero rows bound `?targetEntity`**. Therefore the resolver correctly failed closed. No valid depth-3 boundary, binding projection, continuation edge, new binding, or gold reachability was observed. This is not evidence that another edge helps or does not help retrieval.
- Usage: **7 API attempts**, 6 with measured usage (11,136 input / 1,259 completion tokens; **$0.003738** rate estimate), one uncertain reserved attempt (**$0.0026288**), and **19 successful SELECT attempts** including the original preflight. Maximum rate-based cost exposure was **$0.0063668**, below the approved $0.16 ceiling.

## Protocol deviations

- API request 1 took **163 seconds** by journal timestamps, exceeding the approved 120-second per-call maximum; the configured SDK timeout did not enforce a hard elapsed deadline.
- A local Ctrl-C interrupted API request 4. Request 5 immediately reused the same prompt hash. For conservative accounting this is one repeated-prompt retry, so the strict zero-retry condition was not met, although SDK/wrapper retry metadata remained zero. The uncertain attempt is charged at its full reservation. No more live requests were sent.

## Next action

Offline, determine why these seven entities have no English or untagged `type.object.name` binding in the active store and whether a faithful name source exists. Add a hard elapsed API deadline and a persistent call-once guard; test both before any new pilot. A new manifest and fresh approval are required before further live requests. Keep TGAR as the frozen Hits@1/cost reference, CoR and CFP as comparators, and production depth at 3.

## Key artifacts

- Decision/root cause: `RESULTS/iter_023_label_coverage_and_prefix_readiness/ITER23_DECISION.md`, `LABEL_CACHE_ROOT_CAUSE.md`
- Complete audit and slot accounting: `PILOT_INTEGRITY.json`, `CASE_USAGE_AUDIT.csv`
- Raw traces and meters: `RAW_CASE_TRACES.jsonl`, `SPARQL_QUERY_JOURNAL.jsonl`, `REQUEST_JOURNAL.jsonl`
- Authorization and preserved first attempt: `AUTHORIZATION.json`, `PILOT_RESULTS_FIRST_STOP.json`, `LIVE_RUN_STATE_FIRST_STOP.json`, `RUNTIME_ENTRYPOINT_FIRST_ATTEMPT.py`
- Resume validation and source correction: `RESUME_PREFLIGHT.json`, `CONTINUATION_CODE_CHANGE.json`, `POST_RUN_ACCOUNTING_CORRECTIONS.json`

<!-- ITERATION23_HANDOFF_END -->"""
    path.write_text(text[:start] + block + text[end:], encoding="utf-8")


def write_reports() -> None:
    decision = """# Iteration 23 decision — live repeat-2 label-resolution pilot

**Decision: `REPAIR_NOT_READY`.** The approved three-slot repeat-2 pilot completed, but no slot reached a valid depth-3 Forward boundary. The live result is ineligible for matched-prefix or retrieval-effect analysis, and the consumed approval cannot be reused.

## What happened

- The first endpoint preflight SELECT succeeded at 2026-11-08 04:25:15Z. A local runner-state bug stopped execution before a question or API request. The bug was corrected; the exact successful preflight was audited and reused without replay.
- Three fixed repeat-2 slots then ran against the approved endpoint and model. All three terminated with `LABEL_RESOLUTION_STOP` before a depth-3 boundary.
- The union of unresolved IDs was `m.04stk8g`, `m.0ndvpvf`, `m.04st848`, `m.0hzdjww`, `m.0hz9k44`, `m.04stpzt`, `m.04strb6`. The repeated `m.04stk8g` appeared in two slots.
- Three batched label SELECTs returned 1, 1, and 6 rows due to their `VALUES` plus `OPTIONAL` shape. In all rows `?entity` was bound but `?targetEntity` was unbound. The evidence therefore supports missing English/untagged `type.object.name` values for these lookups; it does not prove that no labels exist in other languages or predicates.

## Metered usage and integrity

- SPARQL: **19/140** physical SELECT attempts, all successful: 1 preflight, 15 navigation, 3 label batches. There were no projections or continuation queries.
- API: **7/60** attempts; 6 returned measured usage (11,136 input and 1,259 completion tokens) for **$0.003738** at the approved rates. One locally interrupted attempt is conservatively reserved at **$0.0026288**; maximum rate-based exposure is **$0.0063668**, under $0.16.
- Total elapsed time from initial authorization run start to completion: 609.543 seconds. 78/78 offline tests passed. 350/350 earlier result artifacts and 4/4 protected production sources are unchanged.

## Protocol deviations

The first API request lasted **163 seconds**, exceeding the approved 120-second maximum. The SDK's configured timeout was not a hard elapsed-time cap. I then sent a local Ctrl-C while another call was active; request 4 was recorded as interrupted, and request 5 reused its identical prompt hash. Conservatively this is one repeated-prompt retry, so the zero-retry condition was not met. SDK/wrapper retry counters remained zero. The raw journal is preserved, the uncertain request is charged at its full reservation, and no further live calls were made.

## Interpretation and next action

This pilot confirms that bounded dynamic lookup alone does not make these live trajectories eligible when the active Freebase name lookup returns no usable canonical name. It provides **no** negative or positive result about extra-hop retrieval, answer accuracy, or generalization. Offline, inspect the active database's label availability and implement a faithful name path only if it preserves the CoR prompt inputs. Also implement and test a hard elapsed request deadline and persistent no-repeat behavior. Then prepare a new manifest and seek fresh authorization before any transport.

The Iteration 22 cache-miss failure remains an instrumentation/data-coverage finding. TGAR remains the frozen practical Hits@1/cost reference; production depth stays 3.

## Audit artifacts

- `PILOT_INTEGRITY.json` and `CASE_USAGE_AUDIT.csv`
- `AUTHORIZATION.json`, `PILOT_RESULTS.json`, `LIVE_RUN_STATE.json`
- `RAW_CASE_TRACES.jsonl`, `SPARQL_QUERY_JOURNAL.jsonl`, `REQUEST_JOURNAL.jsonl`
- `LIVE_RUN_STATE_FIRST_STOP.json`, `PILOT_RESULTS_FIRST_STOP.json`, `RUNTIME_ENTRYPOINT_FIRST_ATTEMPT.py`
"""
    (OUT / "ITER23_DECISION.md").write_text(decision, encoding="utf-8")

    root_cause = """# Iteration 23 label-cache root cause — live confirmation

## Finding

The frozen Iteration 22 cache contains 198 labels. Its seven archived misses were reproduced offline. In the authorized Iteration 23 repeat-2 pilot, the normal CoR path encountered those same seven distinct IDs across three cases. The bounded resolver made three batched `type.object.name` lookups. They returned respectively 1, 1, and 6 rows, but **zero rows bound the label variable `?targetEntity`**. The query uses `VALUES ?entity` and `OPTIONAL` name lookup, so a row count reflects the input entity binding and does not mean that a usable label was returned.

The evidence establishes that these queries found no English or untagged `type.object.name` for those IDs in the active dataset/query context. It does not establish that all languages, aliases, or other predicates lack labels. No alternate label predicate was substituted into model prompts.

## Per-case results

| Slot | Outcome | IDs without a bound name |
|---|---|---|
| depth-4 beneficiary | INELIGIBLE before depth 3 | `m.04stk8g` |
| tied target | INELIGIBLE before depth 3 | `m.0ndvpvf` |
| short-path control | INELIGIBLE before depth 3 | `m.04stk8g`, `m.04st848`, `m.0hzdjww`, `m.0hz9k44`, `m.04stpzt`, `m.04strb6` |

The shared `m.04stk8g` was queried once in each of two different case-local trajectories. Those are separate case lookups, not a transport retry. The seven-ID union matches the archived Iteration 22 miss set.

## Mechanism and repair boundary

The original entity-search path may render each candidate name as part of CoR's prompt. Replacing a missing label with an ID, a guessed label, or an alias would change prompt semantics. The diagnostic resolver therefore fails closed when no eligible `type.object.name` is bound. This behavior is correct for the frozen method contract, but the live pilot demonstrates that the current data source/lookup is insufficient for these trajectories. The synthetic unknown-ID fixture proves only the adapter mechanics when a label exists.

Do not retry these consumed slots or infer a retrieval failure. Offline, inspect the Freebase store and original renderer's label assumptions. Any alternative name source must be checked for identity, language, ambiguity, and semantic equivalence before a fresh protocol is prepared.

## Resource/accounting note

Three label SELECT attempts were included in the 19 total physical attempts. All three returned `ok`; no query was retried. The raw row binding evidence is in `SPARQL_QUERY_JOURNAL.jsonl`, and slot-level usage/misses are in `CASE_USAGE_AUDIT.csv`.
"""
    (OUT / "LABEL_CACHE_ROOT_CAUSE.md").write_text(root_cause, encoding="utf-8")

    query = (OUT / "QUERY_BUDGET.md").read_text(encoding="utf-8")
    if "## Actual Iteration 23 consumption" not in query:
        query += """

## Actual Iteration 23 consumption

The authorized run used **19/140** physical read-only SELECT attempts: 1 endpoint preflight, 15 navigation, and 3 label-resolution batches. All returned `ok`; no projection or post-boundary query ran. The label batches returned 1, 1, and 6 entity rows with zero bound name values. The 12-query label subcap and 8-hour global deadline were not reached.

This authorization is consumed. Unused query headroom is **not** available for another run or investigation under this approval; any future live work needs a new manifest and authorization. See `PILOT_INTEGRITY.json` for the full accounting and protocol deviations.
"""
        (OUT / "QUERY_BUDGET.md").write_text(query, encoding="utf-8")
    model_path = OUT / "QUERY_BUDGET_MODEL.json"
    model = json.loads(model_path.read_text())
    model["evidence_class"] = "archive-based planning estimate plus audited actual Iteration 23 usage"
    model["actual_live_usage"] = {
        "status": "AUTHORIZATION_CONSUMED",
        "physical_select_attempts": 19,
        "endpoint_preflight": 1,
        "navigation": 15,
        "dynamic_label_resolution": 3,
        "binding_projection": 0,
        "one_edge_queries": 0,
        "all_queries_successful": True,
        "remaining_cap_rollover": False,
        "eligible_slots": 0
    }
    write_json(model_path, model)

    auth_request = OUT / "AUTHORIZATION_REQUIRED.md"
    current = auth_request.read_text(encoding="utf-8")
    banner = "> **Historical request: fulfilled and consumed.** The manifest below is no longer available for execution. Its approval and actual use are recorded in `AUTHORIZATION.json`; do not reuse or treat remaining caps as active.\n\n"
    if not current.startswith("> **Historical request:"):
        auth_request.write_text(banner + current, encoding="utf-8")


def write_postrun_correction_record() -> None:
    path = OUT / "POST_RUN_ACCOUNTING_CORRECTIONS.json"
    record = {
        "iteration": 23,
        "live_run_finished_at_utc": "2026-11-08T04:35:24Z",
        "network_activity_after_live_run": False,
        "first_attempt_runner_sha256": "d81ef2a890554e1365dd22ed0b58853f2291a69495d01a499380ed10091fd601",
        "continuation_runner_sha256_at_execution": "17e5eb8c156bb7681d07b6eb7ac33e9b93090d3351ae65e60729b9ad06090c5e",
        "post_run_runner_sha256": hashlib.sha256((ROOT / "scripts/diagnostics/iter_023_label_coverage_and_prefix_readiness/run_iter23_pilot.py").read_bytes()).hexdigest(),
        "post_run_changes": [
            "Clear the initial stop_reason when creating resumed LIVE_RUN_STATE so a completed continuation does not retain a stale terminal reason.",
            "Correct final state/accounting fields to distinguish zero SDK/wrapper retries from one same-prompt reattempt after a local interruption.",
            "Add independent offline audit and final reporting only; raw request/query journals and first-stop artifacts remain unmodified."
        ],
        "preserved_first_stop_artifacts": ["PILOT_RESULTS_FIRST_STOP.json", "LIVE_RUN_STATE_FIRST_STOP.json", "RUNTIME_ENTRYPOINT_FIRST_ATTEMPT.py"],
        "recorded_at": "2026-11-08"
    }
    write_json(path, record)


def main() -> None:
    assert AUDIT["audit_status"] == "ACCOUNTING_RECONCILED_WITH_PROTOCOL_DEVIATIONS"
    assert AUTH["authorization_consumed"] is True
    auth_path = OUT / "AUTHORIZATION.json"
    auth_doc = json.loads(auth_path.read_text())
    auth_doc["execution_status"] = "CONSUMED_WITH_PROTOCOL_DEVIATIONS_REPAIR_NOT_READY"
    write_json(auth_path, auth_doc)
    update_readiness()
    update_label_matrix()
    update_ledgers()
    update_evidence_and_log()
    update_handoff()
    write_reports()
    write_postrun_correction_record()
    print("Iteration 23 live handoff and ledgers updated offline; raw journals were not rewritten.")


if __name__ == "__main__":
    main()
