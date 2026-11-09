from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SAFE_RUNNER_DIR = REPO / "scripts/diagnostics/iter_025_cache_compatible_readiness"
if str(SAFE_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(SAFE_RUNNER_DIR))

from safe_runner import (  # noqa: E402
    HardDeadlineExceeded,
    ResumeForbidden,
    SafeDiagnosticRunner,
    SafetyError,
)


class DeclaredNoRetryTransport:
    max_retries = 0

    def __init__(self, journal_path: Path):
        self.journal_path = journal_path

    def __call__(self, prompt: str) -> str:
        records = [json.loads(line) for line in self.journal_path.read_text().splitlines()]
        starts = [row for row in records if row.get("event") == "ATTEMPT_STARTED"]
        return json.dumps({"schema_version": "iter26.answer_selector.v1", "journal_starts_before_transport": len(starts)})


class HangingDeclaredNoRetryTransport:
    max_retries = 0

    def __call__(self, prompt: str) -> str:
        time.sleep(0.5)
        return "late"


def make_runner(journal: Path, *, calls=2, cost=1.0, sparql=2, timeout=2.0):
    return SafeDiagnosticRunner(
        journal,
        manifest_sha256="a" * 64,
        api_request_cap=calls,
        api_cost_cap_usd=cost,
        input_tokens_per_request=7000,
        output_tokens_per_request=1024,
        input_usd_per_million=0.20,
        output_usd_per_million=1.20,
        sparql_attempt_cap=sparql,
        timeout_seconds=timeout,
        wall_seconds=30,
    )


class SafetyIntegrationTests(unittest.TestCase):
    def payload(self, question="one?"):
        return {
            "question": question,
            "constraints": {"frozen_tgar_filter_context": "Q: one?\nReasoning Chain (SPARQL-style): x"},
            "candidate_entities": [{"id": "m.one", "label": "One"}],
            "instructions": "Return versioned candidate IDs only.",
        }

    def test_fsync_journal_attempt_marker_precedes_fixture_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "attempts.jsonl"
            runner = make_runner(journal)
            response = runner.call_model_once(self.payload(), DeclaredNoRetryTransport(journal))
            self.assertEqual(json.loads(response)["journal_starts_before_transport"], 1)
            rows = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual([row["event"] for row in rows], ["RUN_STARTED", "ATTEMPT_STARTED", "ATTEMPT_FINISHED"])

    def test_restart_after_attempt_is_forbidden_even_when_call_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "attempts.jsonl"
            runner = make_runner(journal)
            runner.call_model_once(self.payload(), DeclaredNoRetryTransport(journal))
            with self.assertRaises(ResumeForbidden):
                make_runner(journal)

    def test_timeout_is_killed_and_restart_is_blocked_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "attempts.jsonl"
            runner = make_runner(journal, timeout=0.12)
            with self.assertRaises(HardDeadlineExceeded):
                runner.call_model_once(self.payload(), HangingDeclaredNoRetryTransport())
            with self.assertRaises(ResumeForbidden):
                make_runner(journal, timeout=0.12)
            rows = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertIn("UNCERTAIN_TIMEOUT_FULL_RESERVATION", [row.get("outcome") for row in rows])

    def test_gold_metadata_is_rejected_before_fixture_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "attempts.jsonl"
            runner = make_runner(journal)
            payload = self.payload()
            payload["gold_answer_ids"] = ["m.one"]
            with self.assertRaises(SafetyError):
                runner.call_model_once(payload, DeclaredNoRetryTransport(journal))
            rows = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual([row["event"] for row in rows], ["RUN_STARTED"])

    def test_select_only_query_cap_is_physical_and_has_no_retry_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "attempts.jsonl"
            runner = make_runner(journal, sparql=1)

            def select_transport():
                return {"rows": []}

            select_transport.max_retries = 0
            runner.select_once("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1", select_transport)
            with self.assertRaises(SafetyError):
                runner.select_once("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1", select_transport)
            rows = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual(sum(row.get("kind") == "sparql" and row.get("event") == "ATTEMPT_STARTED" for row in rows), 1)


if __name__ == "__main__":
    unittest.main()
