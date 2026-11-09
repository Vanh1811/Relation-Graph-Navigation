"""No-retry model-call accounting primitives for a separately approved pilot."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Callable, Mapping

from deadline import DeadlineExceeded, DeadlineGuard


MODEL_NAME = "gpt-5.6-luna"
INPUT_USD_PER_MILLION = 0.20
OUTPUT_USD_PER_MILLION = 1.20


class ModelBudgetStop(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelCaps:
    manifest_sha256: str
    request_cap: int
    input_tokens_per_request: int
    completion_tokens_per_request: int
    estimated_spend_cap_usd: float
    model: str = MODEL_NAME
    input_usd_per_million: float = INPUT_USD_PER_MILLION
    output_usd_per_million: float = OUTPUT_USD_PER_MILLION
    retries: int = 0

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 hex digest")
        if self.request_cap < 1 or self.input_tokens_per_request < 1 or self.completion_tokens_per_request < 1:
            raise ValueError("request and token caps must be positive")
        if self.estimated_spend_cap_usd <= 0 or self.retries != 0:
            raise ValueError("spend cap must be positive and retries must be zero")

    def max_full_reservation(self) -> float:
        return estimate_cost(
            self.input_tokens_per_request,
            self.completion_tokens_per_request,
            self.input_usd_per_million,
            self.output_usd_per_million,
        )


def estimate_cost(input_tokens: int, output_tokens: int,
                  input_rate: float = INPUT_USD_PER_MILLION,
                  output_rate: float = OUTPUT_USD_PER_MILLION) -> float:
    return input_tokens * input_rate / 1_000_000 + output_tokens * output_rate / 1_000_000


def authorization_matches(record: Mapping[str, Any] | None, caps: ModelCaps) -> bool:
    if not isinstance(record, Mapping):
        return False
    approval_text = str(record.get("approval_verbatim", ""))
    expected = {
        "authorization_status": "APPROVED",
        "manifest_sha256": caps.manifest_sha256,
        "model": caps.model,
        "api_request_cap": caps.request_cap,
        "input_tokens_per_request_cap": caps.input_tokens_per_request,
        "completion_tokens_per_request_cap": caps.completion_tokens_per_request,
        "estimated_api_spend_cap_usd": caps.estimated_spend_cap_usd,
        "input_usd_per_million": caps.input_usd_per_million,
        "output_usd_per_million": caps.output_usd_per_million,
        "retries": 0,
    }
    return (
        bool(approval_text.strip())
        and bool(record.get("approved_at"))
        and all(record.get(key) == value for key, value in expected.items())
        and caps.manifest_sha256 in approval_text
        and "no retries" in approval_text.lower()
    )


class NoRetryModelMeter:
    """Reserve each request before transport and charge failed calls safely."""

    def __init__(self, caps: ModelCaps, *, deadline: DeadlineGuard | None = None) -> None:
        self.caps = caps
        self.deadline = deadline
        self.attempts = 0
        self.estimated_cost_usd = 0.0
        self.uncertain_reserved_usd = 0.0
        self.events: list[dict[str, Any]] = []

    def call_once(self, *, authorization: Mapping[str, Any] | None,
                  input_token_reservation: int, completion_token_cap: int,
                  prompt: str, transport: Callable[[], tuple[str | None, Mapping[str, Any]]]) -> tuple[str, Mapping[str, Any]]:
        if not authorization_matches(authorization, self.caps):
            raise ModelBudgetStop("no exact, approved authorization for this manifest and budget")
        if self.deadline is not None:
            try:
                self.deadline.assert_open()
            except DeadlineExceeded as exc:
                raise ModelBudgetStop("pilot wall-clock deadline reached before model request") from exc
        if not isinstance(input_token_reservation, int) or not 0 <= input_token_reservation <= self.caps.input_tokens_per_request:
            raise ModelBudgetStop("input reservation exceeds the per-request cap")
        if not isinstance(completion_token_cap, int) or not 1 <= completion_token_cap <= self.caps.completion_tokens_per_request:
            raise ModelBudgetStop("completion cap exceeds the per-request cap")
        if self.attempts >= self.caps.request_cap:
            raise ModelBudgetStop("model request cap reached")
        reserve = estimate_cost(input_token_reservation, completion_token_cap,
                                self.caps.input_usd_per_million, self.caps.output_usd_per_million)
        if self.estimated_cost_usd + self.uncertain_reserved_usd + reserve > self.caps.estimated_spend_cap_usd + 1e-12:
            raise ModelBudgetStop("request worst-case reservation exceeds remaining spend cap")

        self.attempts += 1  # Consume before the one injected call, including errors.
        event = {
            "event": "ATTEMPT_STARTED", "attempt": self.attempts,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "input_token_reservation": input_token_reservation,
            "completion_token_cap": completion_token_cap,
            "worst_case_reservation_usd": reserve, "retries": 0,
        }
        self.events.append(event)
        try:
            content, usage = transport()
        except BaseException as exc:
            self.uncertain_reserved_usd += reserve
            event.update({"status": "TRANSPORT_ERROR", "error": str(exc)})
            raise ModelBudgetStop("single model attempt failed; no retry") from exc
        if not isinstance(usage, Mapping) or usage.get("error"):
            self.uncertain_reserved_usd += reserve
            event.update({"status": "MISSING_OR_ERROR_USAGE"})
            raise ModelBudgetStop("missing/error usage; full reservation held and no retry")
        used_input, used_output = usage.get("input_tokens"), usage.get("output_tokens")
        if not isinstance(used_input, int) or not isinstance(used_output, int):
            self.uncertain_reserved_usd += reserve
            event.update({"status": "INVALID_USAGE"})
            raise ModelBudgetStop("invalid usage; full reservation held and no retry")
        if used_input > input_token_reservation or used_output > completion_token_cap:
            self.uncertain_reserved_usd += reserve
            event.update({"status": "USAGE_EXCEEDED_RESERVATION", "input_tokens": used_input,
                          "output_tokens": used_output})
            raise ModelBudgetStop("observed usage exceeded this request's reservation")
        actual = estimate_cost(used_input, used_output,
                               self.caps.input_usd_per_million, self.caps.output_usd_per_million)
        if self.estimated_cost_usd + self.uncertain_reserved_usd + actual > self.caps.estimated_spend_cap_usd + 1e-12:
            self.uncertain_reserved_usd += reserve
            event.update({"status": "SPEND_CAP_EXCEEDED", "actual_cost_usd": actual})
            raise ModelBudgetStop("observed cost exceeds total spend cap")
        self.estimated_cost_usd += actual
        event.update({"status": "METERED", "input_tokens": used_input,
                      "output_tokens": used_output, "estimated_cost_usd": actual})
        if not isinstance(content, str) or not content.strip():
            event["status"] = "EMPTY_RESPONSE"
            raise ModelBudgetStop("empty response metered once; no retry")
        return content, usage
