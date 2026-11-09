from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path

from coverage import CACHE, PAIRS, SOURCE, build_inventory
from provenance import build_label_audit
from safe_runner import (
    CacheMissNoTransport,
    DuplicatePrompt,
    ResumeForbidden,
    SafeDiagnosticRunner,
    SafetyError,
    empty_frontier_allowed,
    prefix_state_sha256,
    validate_matched_pair,
    validate_prefix_capture,
)


MANIFEST = hashlib.sha256(b"fixture manifest").hexdigest()
SEVEN = {"m.04st848", "m.04stk8g", "m.04stpzt", "m.04strb6", "m.0hz9k44", "m.0hzdjww", "m.0ndvpvf"}


def single_attempt(fn):
    fn.max_retries = 0
    return fn


def runner(path: Path, **changes) -> SafeDiagnosticRunner:
    settings = dict(
        journal_path=path,
        manifest_sha256=MANIFEST,
        api_request_cap=3,
        api_cost_cap_usd=1.0,
        input_tokens_per_request=7000,
        output_tokens_per_request=1024,
        input_usd_per_million=0.20,
        output_usd_per_million=1.20,
        sparql_attempt_cap=2,
        timeout_seconds=1.0,
        wall_seconds=10.0,
    )
    settings.update(changes)
    return SafeDiagnosticRunner(**settings)


def valid_prefix(**changes):
    state = {
        "prompt_sha256": hashlib.sha256(b"prompt").hexdigest(),
        "prior_response_sha256": hashlib.sha256(b"response").hexdigest(),
        "ordered_dfs_stack": ["rel:one", "rel:two"],
        "active_binding_entity_ids": ["m.abc"],
        "frontier_entity_ids": ["m.xyz"],
        "kg_query_result_sha256s": [hashlib.sha256(b"kg-result").hexdigest()],
        "replay_status": "COMPLETE",
    }
    state.update(changes)
    state["prefix_sha256"] = prefix_state_sha256({**state, "prefix_sha256": ""})
    return state


class Iter25CoverageTests(unittest.TestCase):
    def test_full_inventory_has_30_unique_questions_and_balanced_roles(self):
        inventory, summary = build_inventory()
        self.assertEqual(len(inventory), 30)
        self.assertEqual(summary["role_counts"], {"STRONG_DEPTH4_TARGET": 20, "STRONG_SHORTER_PATH_NEGATIVE_CONTROL": 10})
        self.assertEqual(summary["depth3_rows"], 60)
        self.assertEqual(summary["exact_prefix_complete_cases"], 0)

    def test_cache_is_198_provenance_bearing_ids_and_d3_only(self):
        inventory, summary = build_inventory()
        self.assertEqual(summary["label_cache_unique_ids"], 198)
        self.assertEqual(summary["label_cache_source_sha256_recorded"], summary["archive_sha256"])
        self.assertTrue(all(row["d3_run_count"] == 2 for row in inventory))
        self.assertTrue(all(row["d3_run_count"] == len(row["repeat_indices"]) for row in inventory))

    def test_missing_live_candidates_are_outside_frozen_cache(self):
        inventory, _ = build_inventory()
        cached = {entry["entity_id"] for entry in json.loads(CACHE.read_text())["labels"]}
        self.assertTrue(SEVEN.isdisjoint(cached))
        self.assertEqual(len(inventory), 30)

    def test_candidate_names_and_ids_are_audited_without_fuzzy_join(self):
        inventory, _ = build_inventory()
        self.assertTrue(all(row["candidate_name_conflicts"] == [] for row in inventory))
        self.assertTrue(all(row["cache_label_content_mismatches"] == [] for row in inventory))
        self.assertTrue(all(row["malformed_id_name_pair_count"] == 0 for row in inventory))

    def test_historical_question_is_screened_out_if_exact_state_is_absent(self):
        inventory, _ = build_inventory()
        self.assertTrue(all(row["screening_status"] == "REJECT_FOR_EXACT_PREFIX_PILOT" for row in inventory))
        self.assertTrue(all(not row["exact_prefix_state_complete"] for row in inventory))

    def test_new_repeat_slots_are_deduplicated_from_consumed_live_slots(self):
        inventory, summary = build_inventory()
        self.assertEqual(summary["previous_authorization_consumed_slots"], 6)
        self.assertEqual(summary["prospective_repeat3_slots_absent_and_unconsumed"], 30)
        self.assertTrue(all(row["repeat_indices"] == [1, 2] for row in inventory))
        self.assertTrue(all(row["prospective_repeat3_slot_unconsumed"] for row in inventory))

    def test_seven_ids_are_candidate_bindings_not_gold_or_archived_labels(self):
        audit = build_label_audit()
        self.assertEqual(set(audit["ids"]), SEVEN)
        self.assertEqual(audit["frozen_cache_contains_ids"], [])
        self.assertEqual(audit["gold_scoring_id_overlap"], [])
        self.assertEqual(audit["archived_inference_candidate_name_observations"], [])
        self.assertTrue(all(not mentions for mentions in audit["prompt_mentions"].values()))
        for entity_id, rows in audit["candidate_source_observations"].items():
            self.assertTrue(any(row["iteration"] == 22 and row["binding_name"] == "targetEntity" for row in rows), entity_id)
            self.assertTrue(any(row["iteration"] == 23 and row["binding_name"] == "targetEntity" for row in rows), entity_id)
        for entity_id, lookups in audit["iteration23_label_lookup_observations"].items():
            self.assertTrue(lookups, entity_id)
            self.assertTrue(all(row["status"] == "ok" and not row["targetEntity_value_bound"] for row in lookups), entity_id)
        self.assertTrue(audit["iter24_all_four_predicates_zero_for_all_seven"])


class Iter25RunnerSafetyTests(unittest.TestCase):
    def test_model_start_is_fsynced_before_transport_and_prompt_is_allowlisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            guard = runner(path)
            def fake_transport(prompt):
                records = [json.loads(x) for x in path.read_text().splitlines()]
                return {"ok": True, "last_event": records[-1]["event"], "prompt": prompt}
            fake_transport.max_retries = 0

            result = guard.call_model_once({"question": "Who?", "candidate_entities": [{"id": "m.abc", "label": "A"}]}, fake_transport)
            self.assertEqual(result["ok"], True)
            self.assertEqual(result["last_event"], "ATTEMPT_STARTED")
            self.assertNotIn("gold", result["prompt"].lower())

    def test_gold_metadata_and_non_allowlisted_payload_fail_before_journaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            guard = runner(path)
            with self.assertRaises(SafetyError):
                guard.call_model_once({"question": "Who?", "gold_answer_ids": ["m.abc"]}, single_attempt(lambda _: None))
            with self.assertRaises(SafetyError):
                guard.call_model_once({"question": "Who?", "cohort_role_scoring_only": "TARGET"}, single_attempt(lambda _: None))
            self.assertEqual([json.loads(x)["event"] for x in path.read_text().splitlines()], ["RUN_STARTED"])

    def test_same_prompt_digest_is_never_submitted_twice_in_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = runner(Path(tmp) / "attempts.jsonl")
            payload = {"question": "Who?"}
            guard.call_model_once(payload, single_attempt(lambda _: "ok"))
            with self.assertRaises(DuplicatePrompt):
                guard.call_model_once(payload, single_attempt(lambda _: self.fail("duplicate transport ran")))

    def test_restart_after_success_refuses_to_resume_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            runner(path).call_model_once({"question": "Who?"}, single_attempt(lambda _: "ok"))
            with self.assertRaises(ResumeForbidden):
                runner(path)

    def test_timeout_is_uncertain_full_reservation_and_restart_stays_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            guard = runner(path, timeout_seconds=0.05)
            with self.assertRaises(TimeoutError):
                guard.call_model_once({"question": "wait"}, single_attempt(lambda _: time.sleep(0.5)))
            records = [json.loads(x) for x in path.read_text().splitlines()]
            started = next(record for record in records if record["event"] == "ATTEMPT_STARTED")
            self.assertEqual(started["reserved_tokens"], {"completion": 1024, "input": 7000})
            self.assertEqual(records[-1]["outcome"], "UNCERTAIN_TIMEOUT_FULL_RESERVATION")
            with self.assertRaises(DuplicatePrompt):
                guard.call_model_once({"question": "wait"}, single_attempt(lambda _: self.fail("timed-out prompt was repeated")))
            with self.assertRaises(ResumeForbidden):
                runner(path)

    def test_crash_after_durable_start_marker_blocks_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            path.write_text(json.dumps({"event": "RUN_STARTED", "manifest_sha256": MANIFEST,
                                        "recorded_at_epoch": time.time()}) + "\n")
            with path.open("a") as stream:
                stream.write(json.dumps({"event": "ATTEMPT_STARTED", "kind": "api", "manifest_sha256": MANIFEST,
                                         "payload_sha256": hashlib.sha256(b"prompt").hexdigest()}) + "\n")
            with self.assertRaises(ResumeForbidden):
                runner(path)

    def test_api_worst_case_reservation_exceeding_budget_blocks_before_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = runner(Path(tmp) / "attempts.jsonl", api_cost_cap_usd=0.001)
            with self.assertRaises(SafetyError):
                guard.call_model_once({"question": "Who?"}, single_attempt(lambda _: self.fail("transport ran")))
            self.assertEqual(guard.api_starts, 0)

    def test_overall_wall_cap_blocks_before_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = [100.0]
            guard = runner(Path(tmp) / "attempts.jsonl", wall_seconds=1.0, clock=lambda: now[0])
            now[0] += 2.0
            with self.assertRaises(SafetyError):
                guard.call_model_once({"question": "late"}, single_attempt(lambda _: self.fail("expired transport ran")))

    def test_select_only_meter_and_attempt_101_rejected_before_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = runner(Path(tmp) / "attempts.jsonl", sparql_attempt_cap=100)
            with self.assertRaises(SafetyError):
                guard.select_once("INSERT DATA { <a> <b> <c> }", single_attempt(lambda: self.fail("transport ran")))
            with self.assertRaises(SafetyError):
                guard.select_once("SELECT * WHERE { SERVICE <https://invalid> {?s ?p ?o} }", single_attempt(lambda: self.fail("transport ran")))
            for _ in range(100):
                guard.select_once("SELECT * WHERE { ?s ?p ?o } LIMIT 1", single_attempt(lambda: []))
            with self.assertRaises(SafetyError):
                guard.select_once("SELECT * WHERE { ?s ?p ?o } LIMIT 1", single_attempt(lambda: self.fail("cap overrun transport ran")))
            self.assertEqual(guard.sparql_starts, 100)

    def test_cache_miss_never_invokes_hidden_resolver(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = runner(Path(tmp) / "attempts.jsonl")
            cache_path = Path(tmp) / "labels.json"
            cache_doc = {"manifest_sha256": MANIFEST, "source_archive_sha256": hashlib.sha256(b"archive").hexdigest(),
                         "labels": [{"entity_id": "m.abc", "label": "Known"}]}
            raw = json.dumps(cache_doc).encode()
            cache_path.write_bytes(raw)
            expected = {"expected_cache_sha256": hashlib.sha256(raw).hexdigest(),
                        "expected_source_archive_sha256": cache_doc["source_archive_sha256"],
                        "expected_provenance_manifest_sha256": MANIFEST}
            self.assertEqual(guard.resolve_label_cache_only("m.abc", cache_path, **expected), "Known")
            with self.assertRaises(CacheMissNoTransport):
                guard.resolve_label_cache_only("m.04st848", cache_path, **expected)

    def test_unknown_transport_retry_policy_blocks_before_journaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            guard = runner(path)
            with self.assertRaises(SafetyError):
                guard.call_model_once({"question": "Who?"}, lambda _: "no retry metadata")
            self.assertEqual([json.loads(x)["event"] for x in path.read_text().splitlines()], ["RUN_STARTED"])

    def test_bad_freebase_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = runner(Path(tmp) / "attempts.jsonl")
            cache_path = Path(tmp) / "labels.json"
            cache_doc = {"manifest_sha256": MANIFEST, "source_archive_sha256": hashlib.sha256(b"archive").hexdigest(), "labels": []}
            cache_path.write_text(json.dumps(cache_doc))
            with self.assertRaises(SafetyError):
                guard.resolve_label_cache_only("not-an-id", cache_path,
                    expected_cache_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                    expected_source_archive_sha256=cache_doc["source_archive_sha256"],
                    expected_provenance_manifest_sha256=MANIFEST)

    def test_label_cache_provenance_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = runner(Path(tmp) / "attempts.jsonl")
            cache_path = Path(tmp) / "labels.json"
            cache_doc = {"manifest_sha256": MANIFEST, "source_archive_sha256": hashlib.sha256(b"archive").hexdigest(),
                         "labels": [{"entity_id": "m.abc", "label": "Known"}]}
            cache_path.write_text(json.dumps(cache_doc))
            with self.assertRaises(SafetyError):
                guard.resolve_label_cache_only("m.abc", cache_path,
                    expected_cache_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                    expected_source_archive_sha256=hashlib.sha256(b"wrong archive").hexdigest(),
                    expected_provenance_manifest_sha256=MANIFEST)

    def test_prefix_capture_requires_complete_ordered_bindings_and_hashes(self):
        good = valid_prefix()
        validate_prefix_capture(good)
        bad = dict(good)
        bad.pop("ordered_dfs_stack")
        with self.assertRaises(SafetyError):
            validate_prefix_capture(bad)
        bad = valid_prefix(active_binding_entity_ids=["invalid"])
        with self.assertRaises(SafetyError):
            validate_prefix_capture(bad)
        bad = valid_prefix(replay_status="PARTIAL")
        with self.assertRaises(SafetyError):
            validate_prefix_capture(bad)

    def test_matched_pair_rejects_prompt_stack_or_binding_divergence(self):
        stop = valid_prefix(arm="STOP")
        one_hop = valid_prefix(arm="ONE_HOP", added_relation_traversals=1)
        validate_matched_pair(stop, one_hop)
        divergent = dict(one_hop)
        divergent["prompt_sha256"] = hashlib.sha256(b"different prompt").hexdigest()
        divergent["prefix_sha256"] = prefix_state_sha256(divergent)
        with self.assertRaises(SafetyError):
            validate_matched_pair(stop, divergent)
        divergent = dict(one_hop)
        divergent["ordered_dfs_stack"] = ["rel:two", "rel:one"]
        divergent["prefix_sha256"] = prefix_state_sha256(divergent)
        with self.assertRaises(SafetyError):
            validate_matched_pair(stop, divergent)
        divergent = dict(one_hop)
        divergent["active_binding_entity_ids"] = ["m.other"]
        divergent["prefix_sha256"] = prefix_state_sha256(divergent)
        with self.assertRaises(SafetyError):
            validate_matched_pair(stop, divergent)

    def test_failed_kg_call_cannot_be_mislabeled_as_empty_frontier(self):
        self.assertTrue(empty_frontier_allowed(kg_status="ok", result_ids=[]))
        with self.assertRaises(SafetyError):
            empty_frontier_allowed(kg_status="timeout", result_ids=[])


if __name__ == "__main__":
    unittest.main(verbosity=2)
