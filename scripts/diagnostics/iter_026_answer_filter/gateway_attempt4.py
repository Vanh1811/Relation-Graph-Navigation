"""Attempt-4 gateway with a typed, non-consuming per-request input-cap error."""
from __future__ import annotations

from typing import Any

from scripts.diagnostics.iter_026_answer_filter.gateway import (
    BudgetExhausted,
    ResearchCallOnceGateway,
    canonical_json,
    sha256_bytes,
)


class InputTokenLimitExceeded(BudgetExhausted):
    """A local prompt-size rejection; no request or budget reservation occurred."""

    def __init__(self, estimate: int, limit: int, prompt_sha256: str, case_id: str | None) -> None:
        self.estimate = int(estimate)
        self.limit = int(limit)
        self.prompt_sha256 = prompt_sha256
        self.case_id = case_id
        super().__init__(f"estimated input {estimate} exceeds per-request cap {limit}; no transport")

    def metadata(self) -> dict[str, Any]:
        return {
            "status": "INPUT_CAP_EXCEEDED_BEFORE_TRANSPORT",
            "estimated_input_tokens": self.estimate,
            "input_token_limit": self.limit,
            "prompt_sha256": self.prompt_sha256,
            "case_id": self.case_id,
            "api_request_sent": False,
            "budget_reservation_created": False,
        }


class Attempt4Gateway(ResearchCallOnceGateway):
    """Keep global failures fatal while making only input oversize case-local."""

    def call_model_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_completion_tokens: int,
        reasoning_effort: str = "low",
    ) -> dict[str, Any]:
        # Let the frozen parent validate malformed messages and completion caps.
        if not isinstance(messages, list) or not messages or any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message["role"], str)
            or not isinstance(message["content"], str)
            for message in messages
        ):
            return super().call_model_once(
                messages, max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
            )
        self._check_live()
        prompt_hash = sha256_bytes(canonical_json(messages))
        if prompt_hash in self.prompt_hashes:
            # Preserve the parent gateway's fatal duplicate-prompt handling.
            return super().call_model_once(
                messages, max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
            )
        if not 1 <= int(max_completion_tokens) <= self.output_tokens_per_request:
            return super().call_model_once(
                messages, max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
            )
        if self.api_starts >= self.api_request_cap:
            raise BudgetExhausted("API request cap reached")
        reservation = (
            self.input_tokens_per_request * self.input_price
            + self.output_tokens_per_request * self.output_price
        ) / 1_000_000
        if self.api_reserved_usd + reservation > self.api_cost_cap_usd + 1e-12:
            raise BudgetExhausted("worst-case API reservation would exceed the total cost cap")
        estimate = int(self.token_counter(messages))
        if estimate > self.input_tokens_per_request:
            raise InputTokenLimitExceeded(
                estimate=estimate,
                limit=self.input_tokens_per_request,
                prompt_sha256=sha256_bytes(canonical_json(messages)),
                case_id=self.active_case_id,
            )
        return super().call_model_once(
            messages,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )
