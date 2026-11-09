"""One-call, finish-aware diagnostic runtime for a future I26 attempt.

All changes to frozen TGAR helper bindings are temporary and process-local.
Production source and prompts remain untouched. This module never performs a
network request itself; it delegates exactly once to the supplied gateway.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from contextlib import contextmanager
from typing import Any, Iterator

from scripts.diagnostics.iter_026_answer_filter.runtime import MODEL_NAME


HELPER_MODULES = (
    "chain_of_relations.methods.cor.tools.filter",
    "chain_of_relations.tools.reasoning",
    "chain_of_relations.tools.relation_prune",
    "chain_of_relations.tools.generate_directly",
    "chain_of_relations.tools.entity_prune",
)


class EmptyOrTruncatedModelResponse(RuntimeError):
    """A completed HTTP response that cannot safely continue the TGAR step."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = dict(metadata)
        super().__init__(
            "model response is empty or unfinished "
            f"(finish_reason={self.metadata.get('finish_reason')!r})"
        )


class StrictGatewayLLMAPI:
    """LLMAPI facade that preserves finish metadata and rejects truncation."""

    max_retries = 0

    def __init__(self, gateway: Any, model_name: str | None = None) -> None:
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
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        result = self.gateway.call_model_once(
            messages,
            max_completion_tokens=min(max(1, int(max_tokens)), 1024),
            reasoning_effort="low",
        )
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        normalized_usage = {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
        content = result.get("content")
        finish_reason = result.get("finish_reason")
        prompt_sha = hashlib.sha256(json.dumps(
            messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        metadata = {
            "finish_reason": finish_reason,
            "raw_response_sha256": result.get("raw_response_sha256"),
            "usage": normalized_usage,
            "prompt_sha256": prompt_sha,
            "content_empty": not isinstance(content, str) or not content.strip(),
            "one_shot": True,
        }
        if finish_reason != "stop" or metadata["content_empty"]:
            raise EmptyOrTruncatedModelResponse(metadata)
        return str(content), normalized_usage


def strict_run_llm(
    llm_generate: Any,
    prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    system_prompt: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Call the supplied generator exactly once; never swallow or retry errors."""
    del max_retries  # frozen call sites pass retry counts; this adapter does not honor them
    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    text, usage = llm_generate(prompt, **kwargs)
    if not isinstance(text, str) or not text.strip():
        raise EmptyOrTruncatedModelResponse({
            "finish_reason": "MISSING_OR_EMPTY_AT_CALLER_BOUNDARY",
            "raw_response_sha256": None,
            "usage": dict(usage) if isinstance(usage, dict) else {},
            "content_empty": True,
            "one_shot": True,
        })
    return text, usage


@contextmanager
def install_strict_helper_boundaries() -> Iterator[None]:
    """Temporarily replace every CoR-used retrying helper with one strict call."""
    modules = [importlib.import_module(name) for name in HELPER_MODULES]
    originals = [(module, module._run_llm) for module in modules]
    try:
        for module, _ in originals:
            module._run_llm = strict_run_llm
        yield
    finally:
        for module, original in originals:
            module._run_llm = original


@contextmanager
def install_strict_llm_constructor(gateway: Any) -> Iterator[None]:
    """Temporarily bind CoRAgent's imported LLMAPI to the strict facade."""
    module = importlib.import_module("chain_of_relations.methods.cor.agent")
    original = module.LLMAPI
    module.LLMAPI = lambda model_name=None, **_: StrictGatewayLLMAPI(gateway, model_name)
    try:
        yield
    finally:
        module.LLMAPI = original


def make_strict_tgar_agent(gateway: Any) -> Any:
    """Construct the frozen TGAR v1.1.3 agent with the strict call-once LLM."""
    from chain_of_relations.kg_backend.freebase_backend import FreebaseBackend
    from chain_of_relations.methods.adaptive_v1_1_3.agent import AdaptiveRoutingAgent
    from chain_of_relations.methods.cor.agent import CoRAgent
    from scripts.diagnostics.iter_026_answer_filter.runtime import GatewayFreebaseBackend

    backend: FreebaseBackend = GatewayFreebaseBackend(gateway)
    with install_strict_llm_constructor(gateway):
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
    """Return a completed answer or a case-local empty/truncated failure."""
    try:
        with install_strict_helper_boundaries():
            result = agent.answer(**kwargs)
        return result, None
    except EmptyOrTruncatedModelResponse as exc:
        return None, exc.metadata


def selector_response_metadata(api_result: dict[str, Any]) -> dict[str, Any]:
    """Preserve actual finish, raw response hash, and usage for selector calls."""
    usage = api_result.get("usage") if isinstance(api_result.get("usage"), dict) else {}
    return {
        "finish_reason": api_result.get("finish_reason"),
        "raw_response_sha256": api_result.get("raw_response_sha256"),
        "usage": {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        },
    }
