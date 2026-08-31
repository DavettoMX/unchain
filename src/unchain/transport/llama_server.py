"""Typed wrapper over llama-server's native HTTP API.

Endpoints used: ``/health``, ``/props``, ``/completion`` (with optional
``json_schema`` grammar-constrained decoding), ``/tokenize`` and
``/detokenize``. The OpenAI-compatible layer is deliberately untouched.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from unchain.transport.http import HttpClient, TransportError, parse_url
from unchain.transport.sse import iter_sse_data


class SamplingParams(BaseModel):
    """Pass-through sampling knobs mapped onto llama.cpp /completion fields."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int = Field(default=512, ge=1)
    stop: list[str] = Field(default_factory=list)
    seed: int | None = None
    cache_prompt: bool = True

    def to_request(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "n_predict": self.max_tokens,
            "cache_prompt": self.cache_prompt,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.stop:
            payload["stop"] = self.stop
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload


class CompletionChunk(BaseModel):
    content: str = ""
    stop: bool = False
    stop_type: str | None = None


class LlamaServer:
    """Native llama.cpp server client (one in-flight request at a time)."""

    def __init__(self, base_url: str, *, connect_timeout: float = 10.0) -> None:
        host, port = parse_url(base_url)
        self.base_url = f"http://{host}" + (f":{port}" if port != 80 else "")
        self._client = HttpClient(host, port, connect_timeout=connect_timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> LlamaServer:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def health(self) -> bool:
        response = await self._client.request("GET", "/health", timeout=10.0)
        return response.status == 200

    async def props(self) -> dict[str, Any]:
        response = await self._client.request("GET", "/props", timeout=10.0)
        props: dict[str, Any] = response.json()
        return props

    async def completion(
        self,
        prompt: str,
        *,
        json_schema: str | dict[str, Any] | None = None,
        sampling: SamplingParams | None = None,
        timeout: float = 300.0,
    ) -> str:
        """Blocking completion; returns the generated text."""
        payload = self._completion_payload(prompt, json_schema, sampling)
        response = await self._client.request(
            "POST", "/completion", json_body=payload, timeout=timeout
        )
        return str(response.json()["content"])

    async def completion_stream(
        self,
        prompt: str,
        *,
        json_schema: str | dict[str, Any] | None = None,
        sampling: SamplingParams | None = None,
        idle_timeout: float = 300.0,
    ) -> AsyncIterator[CompletionChunk]:
        """Streamed completion; yields chunks until the server signals stop."""
        payload = self._completion_payload(prompt, json_schema, sampling)
        payload["stream"] = True
        chunks = self._client.stream(
            "POST", "/completion", json_body=payload, idle_timeout=idle_timeout
        )
        async for data in iter_sse_data(chunks):
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                raise TransportError(f"malformed SSE payload: {data[:256]!r}") from exc
            if not isinstance(event, dict):
                continue
            yield CompletionChunk(
                content=str(event.get("content", "")),
                stop=bool(event.get("stop", False)),
                stop_type=event.get("stop_type"),
            )

    async def tokenize(self, content: str, *, add_special: bool = False) -> list[int]:
        response = await self._client.request(
            "POST", "/tokenize", json_body={"content": content, "add_special": add_special}
        )
        return [int(token) for token in response.json()["tokens"]]

    async def detokenize(self, tokens: list[int]) -> str:
        response = await self._client.request("POST", "/detokenize", json_body={"tokens": tokens})
        return str(response.json()["content"])

    def _completion_payload(
        self,
        prompt: str,
        json_schema: str | dict[str, Any] | None,
        sampling: SamplingParams | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"prompt": prompt}
        if sampling is not None:
            payload.update(sampling.to_request())
        if json_schema is not None:
            if isinstance(json_schema, dict):
                json_schema = json.dumps(json_schema)
            payload["json_schema"] = json_schema
        return payload
