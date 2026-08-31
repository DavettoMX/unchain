from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from unchain.agent.loop import CompletionBackend
from unchain.models import Message
from unchain.tools.registry import ToolRegistry
from unchain.transport.llama_server import CompletionChunk, SamplingParams


class FakeBackend:
    """In-memory CompletionBackend that replays scripted outputs."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []
        self.schemas: list[Any] = []
        self.samplings: list[Any] = []

    async def completion(
        self,
        prompt: str,
        *,
        json_schema: str | dict[str, Any] | None = None,
        sampling: SamplingParams | None = None,
        timeout: float = 300.0,
    ) -> str:
        self.prompts.append(prompt)
        self.schemas.append(json_schema)
        self.samplings.append(sampling)
        return self.outputs.pop(0)

    async def completion_stream(
        self,
        prompt: str,
        *,
        json_schema: str | dict[str, Any] | None = None,
        sampling: SamplingParams | None = None,
        idle_timeout: float = 300.0,
    ) -> AsyncIterator[CompletionChunk]:
        self.prompts.append(prompt)
        self.schemas.append(json_schema)
        self.samplings.append(sampling)
        text = self.outputs.pop(0)
        for index in range(0, len(text), 8):
            yield CompletionChunk(content=text[index : index + 8])


def demo_registry() -> ToolRegistry:
    from pydantic import BaseModel

    class EchoArgs(BaseModel):
        text: str

    class FailArgs(BaseModel):
        pass

    registry = ToolRegistry()

    async def echo(text: str) -> str:
        return f"echo: {text}"

    def boom() -> str:
        raise RuntimeError("boom")

    registry.add("echo", description="echo it", args_model=EchoArgs, handler=echo)
    registry.add("fail", description="always fails", args_model=FailArgs, handler=boom)
    return registry


@pytest.fixture
def registry() -> Iterator[ToolRegistry]:
    yield demo_registry()


@pytest.fixture
def backend() -> Iterator[FakeBackend]:
    yield FakeBackend([])


def transcript_roles(messages: list[Message]) -> list[str]:
    return [message.role for message in messages]


def _unused(_backend: CompletionBackend) -> None:  # keep the Protocol import honest
    return None
