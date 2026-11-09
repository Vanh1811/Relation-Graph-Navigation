"""One-shot TGAR adapter with typed handling for over-cap prompts."""
from __future__ import annotations

import importlib
import hashlib
import json
from contextlib import contextmanager
from typing import Any, Iterator

from scripts.diagnostics.iter_026_answer_filter.runtime import MODEL_NAME
from scripts.diagnostics.iter_026_answer_filter.strict_runtime import (
    EmptyOrTruncatedModelResponse,
    install_strict_helper_boundaries,
    selector_response_metadata,
)
from scripts.diagnostics.iter_026_answer_filter.gateway_attempt4 import (
    Attempt4Gateway,
    InputTokenLimitExceeded,
)


class InputCapCaseFailure(RuntimeError):
    """Case-local input-size failure; no request was sent and no cap was spent."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = {"failure_kind": "INPUT_CAP_EXCEEDED", **metadata}
        super().__init__("model prompt exceeded the authorized input-token cap before transport")


class Attempt4LLMAPI:
    max_retries = 0

    def __init__(self, gateway: Attempt4Gateway, model_name: str | None = None) -> None:
        if model_name not in (None, MODEL_NAME):
            raise ValueError(f"Iteration 26 is frozen to {MODEL_NAME}")
        self.gateway = gateway
        self.model_name = MODEL_NAME

    def generate(
        self,
        user_prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ) -> tuple[str, dict[str, int]]:
        del temperature  # model defaults are frozen; unsupported controls are omitted
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        try:
            result = self.gateway.call_model_once(
                messages,
                max_completion_tokens=min(max(1, int(max_tokens)), 1024),
                reasoning_effort="low",
            )
        except InputTokenLimitExceeded as exc:
            raise InputCapCaseFailure(exc.metadata()) from exc
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        normalized_usage = {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
        content = result.get("content")
        prompt_hash = hashlib.sha256(json.dumps(
            messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        metadata = {
            "finish_reason": result.get("finish_reason"),
            "raw_response_sha256": result.get("raw_response_sha256"),
            "usage": normalized_usage,
            "prompt_sha256": prompt_hash,
            "content_empty": not isinstance(content, str) or not content.strip(),
            "one_shot": True,
        }
        if metadata["finish_reason"] != "stop" or metadata["content_empty"]:
            raise EmptyOrTruncatedModelResponse(metadata)
        return str(content), normalized_usage


@contextmanager
def install_attempt4_llm_constructor(gateway: Attempt4Gateway) -> Iterator[None]:
    module = importlib.import_module("chain_of_relations.methods.cor.agent")
    original = module.LLMAPI
    module.LLMAPI = lambda model_name=None, **_: Attempt4LLMAPI(gateway, model_name)
    try:
        yield
    finally:
        module.LLMAPI = original


def make_attempt4_tgar_agent(gateway: Attempt4Gateway) -> Any:
    from chain_of_relations.kg_backend.freebase_backend import FreebaseBackend
    from chain_of_relations.methods.adaptive_v1_1_3.agent import AdaptiveRoutingAgent
    from chain_of_relations.methods.cor.agent import CoRAgent
    from scripts.diagnostics.iter_026_answer_filter.runtime import GatewayFreebaseBackend

    backend: FreebaseBackend = GatewayFreebaseBackend(gateway)
    with install_attempt4_llm_constructor(gateway):
        cor = CoRAgent(
            model_name=MODEL_NAME,
            relation_width=3,
            depth=3,
            sample_relation_threshold=1000,
            remove_unnecessary_rel=True,
            temperature_exploration=0.3,
            temperature_reasoning=0.1,
            max_token=1024,
            backend=backend,
        )
    return AdaptiveRoutingAgent(cor, confidence_threshold=0.75, max_branches=3)


def answer_case_once(agent: Any, **kwargs: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Treat only input-cap and empty/truncated model responses as case-local."""
    try:
        with install_strict_helper_boundaries():
            return agent.answer(**kwargs), None
    except EmptyOrTruncatedModelResponse as exc:
        return None, {"failure_kind": "EMPTY_OR_TRUNCATED_RESPONSE", **exc.metadata}
    except InputCapCaseFailure as exc:
        return None, {"failure_kind": "INPUT_CAP_EXCEEDED", **exc.metadata}


def selector_call_or_cap_fallback(
    gateway: Attempt4Gateway,
    messages: list[dict[str, str]],
    *,
    max_completion_tokens: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Catch only a local prompt-size rejection; global budget failures propagate."""
    try:
        return gateway.call_model_once(
            messages,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort="low",
        ), None
    except InputTokenLimitExceeded as exc:
        return None, exc.metadata()


