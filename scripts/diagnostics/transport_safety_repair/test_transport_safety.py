"""Offline adversarial checks; fake callables only, no network/API access."""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hard_deadline import (  # noqa: E402
    HardDeadlineExceeded,
    call_with_hard_deadline,
)
from model_call_guard import (  # noqa: E402
    DuplicatePromptAttempt,
    OneShotModelCaller,
)


class TransportSafetyTests(unittest.TestCase):
    def test_fast_transport_returns_value(self):
        self.assertEqual({"answer": "fixture"}, call_with_hard_deadline(
            lambda: {"answer": "fixture"}, 2.0,
        ))

    def test_hanging_transport_is_killed_at_hard_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "ran_after_timeout"

            def hang_then_mark():
                time.sleep(0.65)
                marker.write_text("unexpected", encoding="utf-8")

            started = time.monotonic()
            with self.assertRaises(HardDeadlineExceeded):
                call_with_hard_deadline(hang_then_mark, 0.15)
            elapsed = time.monotonic() - started
            time.sleep(0.7)
            self.assertLess(elapsed, 0.6)
            self.assertFalse(marker.exists(), "timed-out child kept running after return")

    def test_same_prompt_is_blocked_after_timeout_without_second_transport(self):
        caller = OneShotModelCaller()
        digest = "a" * 64

        with tempfile.TemporaryDirectory() as directory:
            started_marker = Path(directory) / "started"
            finished_marker = Path(directory) / "finished"

            def timeout_once():
                started_marker.write_text("started", encoding="utf-8")
                time.sleep(0.65)
                finished_marker.write_text("finished", encoding="utf-8")

            with self.assertRaises(HardDeadlineExceeded):
                caller.call_once(digest, timeout_once, 0.15)
            self.assertTrue(started_marker.exists(), "first transport worker never started")
            with self.assertRaises(DuplicatePromptAttempt):
                caller.call_once(digest, lambda: finished_marker.write_text("duplicate"), 1.0)
            time.sleep(0.7)
            self.assertFalse(finished_marker.exists(), "timed-out worker continued after return")
            self.assertEqual("UNCERTAIN_OR_FAILED", caller.outcomes[digest])

    def test_distinct_prompt_hashes_are_independent(self):
        caller = OneShotModelCaller()
        self.assertEqual("one", caller.call_once("b" * 64, lambda: "one", 2.0))
        self.assertEqual("two", caller.call_once("c" * 64, lambda: "two", 2.0))

    def test_invalid_hash_never_calls_transport(self):
        caller = OneShotModelCaller()
        calls = []
        with self.assertRaises(ValueError):
            caller.call_once("not-a-hash", lambda: calls.append("sent"), 1.0)
        self.assertEqual([], calls)


if __name__ == "__main__":
    unittest.main()
