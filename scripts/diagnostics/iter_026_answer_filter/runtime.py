"""Diagnostic-only adapters that reuse frozen TGAR code under one-shot I/O."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from chain_of_relations.kg_backend.freebase_backend import FreebaseBackend
from chain_of_relations.kg_backend.freebase import db_func
from chain_of_relations.methods.adaptive_v1_1_3.agent import AdaptiveRoutingAgent
from chain_of_relations.methods.cor.agent import CoRAgent

MODEL_NAME = "gpt-5.6-luna"


class GatewayLLMAPI:
    """Replacement LLMAPI facade; tool-level retries hit the one-shot guard."""

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
    ) -> tuple[str | None, dict[str, int]]:
        # GPT-5.6 Luna only accepts its default sampling behavior. The frozen
        # TGAR call sites may pass temperatures; this adapter deliberately
        # omits temperature and frequency/presence penalties from the request.
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        result = self.gateway.call_model_once(
            messages,
            max_completion_tokens=min(max(1, int(max_tokens)), 1024),
            reasoning_effort="low",
        )
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        return result.get("content"), {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }


class GatewayFreebaseBackend(FreebaseBackend):
    """Freebase backend using the I26 metered, no-retry SELECT gateway."""

    def __init__(self, gateway: Any) -> None:
        self.gateway = gateway
        self._label_cache: dict[str, str] = {}

    def execute_sparql_with_meta(self, sparql_txt: str) -> dict[str, Any]:
        try:
            result = self.gateway.select_once(sparql_txt)
            return {
                "rows": result["bindings"], "status": "ok", "timed_out": False,
                "attempts": 1, "error": "",
                "response_sha256": result.get("raw_response_sha256"),
            }
        except Exception as exc:
            return {
                "rows": [], "status": "timeout" if "deadline" in type(exc).__name__.lower() else "error",
                "timed_out": "deadline" in type(exc).__name__.lower(),
                "attempts": 1, "error": type(exc).__name__,
            }

    def execute_sparql(self, sparql_txt: str) -> list[dict[str, Any]]:
        return self.execute_sparql_with_meta(sparql_txt)["rows"]

    @staticmethod
    def _best_label(rows: list[dict[str, Any]]) -> str | None:
        best: tuple[int, str] | None = None
        for row in rows:
            binding = row.get("targetEntity", {})
            label = str(binding.get("value", "")).strip()
            if not label:
                continue
            rank = db_func._label_rank(str(binding.get("xml:lang", "")))
            if best is None or rank < best[0]:
                best = (rank, label)
        return best[1] if best else None

    def id2entity_name_or_type(self, entity_id: str) -> str:
        entity_id = str(entity_id or "").strip()
        if not entity_id:
            return "UnName_Entity"
        if entity_id in self._label_cache:
            return self._label_cache[entity_id]
        query = db_func._render_id2name_query(entity_id)
        rows = self.execute_sparql_with_meta(query)["rows"]
        label = self._best_label(rows) or "UnName_Entity"
        self._label_cache[entity_id] = label
        return label

    def id2entity_names(self, entity_ids: list[str]) -> dict[str, str]:
        pending = list(dict.fromkeys(str(value).strip() for value in entity_ids if str(value).strip()))
        output = {entity_id: self._label_cache[entity_id] for entity_id in pending if entity_id in self._label_cache}
        pending = [entity_id for entity_id in pending if entity_id not in output]
        batch_size = max(1, int(db_func.SPARQL_LABEL_BATCH_SIZE))
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            query = db_func._render_ids2names_query(batch)
            rows = self.execute_sparql_with_meta(query)["rows"]
            best: dict[str, tuple[int, str]] = {}
            for row in rows:
                uri = str(row.get("entity", {}).get("value", "")).strip()
                entity_id = db_func._normalize_entity_uri(uri)
                binding = row.get("targetEntity", {})
                label = str(binding.get("value", "")).strip()
                if not entity_id or not label:
                    continue
                rank = db_func._label_rank(str(binding.get("xml:lang", "")))
                if entity_id not in best or rank < best[entity_id][0]:
                    best[entity_id] = (rank, label)
            for entity_id in batch:
                label = best.get(entity_id, (2, ""))[1]
                if not label:
                    label = self.id2entity_name_or_type(entity_id)
                self._label_cache[entity_id] = label
                output[entity_id] = label
        return output


@contextmanager
def install_gateway_llm(gateway: Any) -> Iterator[None]:
    """Patch only the constructor binding used by CoRAgent, then restore it."""
    import chain_of_relations.methods.cor.agent as cor_agent_module

    original = cor_agent_module.LLMAPI
    cor_agent_module.LLMAPI = lambda model_name=None, **_: GatewayLLMAPI(gateway, model_name)
    try:
        yield
    finally:
        cor_agent_module.LLMAPI = original


def make_tgar_agent(gateway: Any) -> AdaptiveRoutingAgent:
    """Build frozen TGAR v1.1.3 using only the diagnostic one-shot adapters."""
    backend = GatewayFreebaseBackend(gateway)
    with install_gateway_llm(gateway):
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
