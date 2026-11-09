"""One-shot prompt guard that composes with the hard-deadline worker."""
from __future__ import annotations

import re
from typing import Any, Callable

from hard_deadline import call_with_hard_deadline


class DuplicatePromptAttempt(RuntimeError):
    """A prompt hash has already been claimed in this pilot process."""


class OneShotModelCaller:
    """Reject duplicate prompt hashes and execute each call in a killable worker.

    Claiming happens before transport. A timeout, transport error, malformed
    response, or later caller-side usage validation failure must never cause
    the same prompt to be submitted again. The caller should also use a fresh
    output directory per authorization; process-local state is not a restart
    journal.
    """

    def __init__(self) -> None:
        self.claimed_prompt_hashes: set[str] = set()
        self.outcomes: dict[str, str] = {}

    @staticmethod
    def validate_hash(prompt_sha256: str) -> str:
        if not isinstance(prompt_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", prompt_sha256):
            raise ValueError("prompt_sha256 must be a lowercase SHA-256 digest")
        return prompt_sha256

    def claim(self, prompt_sha256: str) -> str:
        digest = self.validate_hash(prompt_sha256)
        if digest in self.claimed_prompt_hashes:
            raise DuplicatePromptAttempt("prompt hash was already submitted; duplicate transport blocked")
        self.claimed_prompt_hashes.add(digest)
        self.outcomes[digest] = "CLAIMED"
        return digest

    def call_once(self, prompt_sha256: str, transport: Callable[[], Any],
                  timeout_seconds: float) -> Any:
        digest = self.claim(prompt_sha256)
        try:
            value = call_with_hard_deadline(transport, timeout_seconds)
        except BaseException as exc:
            self.outcomes[digest] = "UNCERTAIN_OR_FAILED"
            raise
        self.outcomes[digest] = "RETURNED"
        return value

