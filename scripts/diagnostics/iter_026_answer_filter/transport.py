"""Single-attempt HTTP transports for a separately authorized Iteration 26 run.

These adapters are inert until called. They do not read credentials, create
connections, or contact OpenAI/Freebase at import time. Network calls are kept
to one direct HTTP request; retries and redirects are not followed.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import ssl
from typing import Any, Callable
from urllib.parse import urlencode, urlsplit


OPENAI_HOST = "api.openai.com"
OPENAI_PATH = "/v1/chat/completions"
MODEL_NAME = "gpt-5.6-luna"
SPARQL_ENDPOINT = "http://127.0.0.1:8890/sparql"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _verified_certifi_context() -> ssl.SSLContext:
    """Build a hostname-verifying TLS context from the installed CA bundle."""
    try:
        import certifi
    except ImportError as exc:
        raise RuntimeError("certifi is required for verified OpenAI TLS; refusing transport") from exc
    context = ssl.create_default_context(cafile=certifi.where())
    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        raise RuntimeError("verified TLS settings are unavailable; refusing transport")
    return context


def count_input_tokens(messages: list[dict[str, str]]) -> int:
    """Conservatively count message text plus fixed per-message framing."""
    try:
        import tiktoken
    except ImportError as exc:  # fail before any network attempt
        raise RuntimeError("tiktoken is required for the 7,000-token preflight") from exc
    try:
        encoding = tiktoken.encoding_for_model(MODEL_NAME)
    except KeyError as exc:
        raise RuntimeError(f"No local tokenizer mapping for {MODEL_NAME}; refusing transport") from exc
    return sum(len(encoding.encode(message["role"])) + len(encoding.encode(message["content"])) + 8 for message in messages) + 8


class OpenAIChatCompletionsTransport:
    """One HTTPS POST to the official Chat Completions endpoint; no SDK retries."""

    max_retries = 0

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 120.0,
        connection_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("an API key must be supplied explicitly")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.__api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.connection_factory = connection_factory

    def __call__(self, request_body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request_body, dict) or request_body.get("model") != MODEL_NAME:
            raise ValueError(f"request model must be frozen to {MODEL_NAME}")
        if set(request_body) != {"model", "messages", "max_completion_tokens", "reasoning_effort", "stream"}:
            raise ValueError("request body contains unsupported or unfrozen fields")
        if not 1 <= int(request_body["max_completion_tokens"]) <= 1024:
            raise ValueError("completion cap is outside the frozen 1..1024 range")
        if request_body["reasoning_effort"] != "low" or request_body["stream"] is not False:
            raise ValueError("reasoning effort and streaming mode differ from the frozen request contract")
        body = _json_bytes(request_body)
        if self.connection_factory is None:
            connection = http.client.HTTPSConnection(
                OPENAI_HOST,
                timeout=self.timeout_seconds,
                context=_verified_certifi_context(),
            )
        else:
            connection = self.connection_factory(OPENAI_HOST, timeout=self.timeout_seconds)
        try:
            connection.request(
                "POST",
                OPENAI_PATH,
                body=body,
                headers={
                    "Authorization": f"Bearer {self.__api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            response = connection.getresponse()
            response_body = response.read()
            try:
                parsed = json.loads(response_body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                parsed = None
            choice = ((parsed or {}).get("choices") or [{}])[0] if isinstance(parsed, dict) else {}
            message = choice.get("message") or {}
            usage = (parsed or {}).get("usage") or {} if isinstance(parsed, dict) else {}
            return {
                "http_status": int(response.status),
                "raw_response_sha256": _sha256(response_body),
                "raw_response_body": response_body.decode("utf-8", errors="replace"),
                "content": message.get("content"),
                "finish_reason": choice.get("finish_reason"),
                "usage": {
                    "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "output_tokens": int(usage.get("completion_tokens", 0) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                },
                "error_code": ((parsed or {}).get("error") or {}).get("code") if isinstance(parsed, dict) else None,
            }
        finally:
            connection.close()


class FreebaseSelectTransport:
    """One form-encoded HTTP POST to the already-forwarded local endpoint."""

    max_retries = 0

    def __init__(
        self,
        endpoint: str = SPARQL_ENDPOINT,
        *,
        timeout_seconds: float = 120.0,
        connection_factory: Callable[..., Any] = http.client.HTTPConnection,
    ) -> None:
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
            or parsed.port != 8890 or parsed.path != "/sparql"
            or parsed.username or parsed.password or parsed.query or parsed.fragment
        ):
            raise ValueError("Iteration 26 permits only http://127.0.0.1:8890/sparql")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self.connection_factory = connection_factory

    def __call__(self, query: str) -> dict[str, Any]:
        body = urlencode({"query": query, "format": "json"}).encode("utf-8")
        connection = self.connection_factory("127.0.0.1", 8890, timeout=self.timeout_seconds)
        try:
            connection.request(
                "POST",
                "/sparql",
                body=body,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/sparql-results+json, application/json",
                },
            )
            response = connection.getresponse()
            response_body = response.read()
            try:
                parsed = json.loads(response_body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                parsed = None
            bindings = None
            if isinstance(parsed, dict):
                bindings = (parsed.get("results") or {}).get("bindings")
            return {
                "http_status": int(response.status),
                "raw_response_sha256": _sha256(response_body),
                "raw_response_body": response_body.decode("utf-8", errors="replace"),
                "bindings": bindings,
            }
        finally:
            connection.close()
