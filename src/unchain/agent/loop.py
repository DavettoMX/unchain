"""The agent loop: generate (grammar-constrained) -> parse -> execute -> feed back."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from unchain.agent.envelope import EnvelopeError, parse_envelope
from unchain.agent.history import PromptTemplate, build_system_prompt
from unchain.models import Message, RunStatus, StepRecord
from unchain.tools.executor import ToolResult, execute_tool
from unchain.tools.registry import ToolRegistry
from unchain.transport.llama_server import CompletionChunk, LlamaServer, SamplingParams


class CompletionBackend(Protocol):
    """What the agent needs from a llama.cpp backend (structural typing)."""

    async def completion(
        self,
        prompt: str,
        *,
        json_schema: str | dict[str, Any] | None = None,
        sampling: SamplingParams | None = None,
        timeout: float = 300.0,
    ) -> str: ...

    def completion_stream(
        self,
        prompt: str,
        *,
        json_schema: str | dict[str, Any] | None = None,
        sampling: SamplingParams | None = None,
        idle_timeout: float = 300.0,
    ) -> AsyncIterator[CompletionChunk]: ...


# -- events ---------------------------------------------------------------


@dataclass(frozen=True)
class IterationStarted:
    iteration: int


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallStarted:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolCallFinished:
    name: str
    result: ToolResult


@dataclass(frozen=True)
class FinalAnswer:
    text: str


@dataclass(frozen=True)
class AgentFailed:
    message: str


AgentEvent = (
    IterationStarted | TextDelta | ToolCallStarted | ToolCallFinished | FinalAnswer | AgentFailed
)
EventCallback = Callable[[AgentEvent], Awaitable[None] | None]


@dataclass
class AgentRunResult:
    status: RunStatus
    final: str | None = None
    steps: list[StepRecord] = field(default_factory=list)
    iterations: int = 0
    history: list[Message] = field(default_factory=list)


class RunRecorder(Protocol):
    """Persistence hook implemented by the state layer (asyncpg)."""

    async def on_run_start(self, session_id: UUID | None) -> UUID | None: ...

    async def on_step(self, run_id: UUID, step: StepRecord) -> None: ...

    async def on_run_end(self, run_id: UUID, status: RunStatus, final: str | None) -> None: ...


@dataclass(frozen=True)
class AgentConfig:
    max_iterations: int = 16
    max_invalid_retries: int = 3
    tool_timeout: float = 60.0
    sampling: SamplingParams = field(default_factory=SamplingParams)
    template: PromptTemplate = field(default_factory=PromptTemplate)
    persona: str | None = None
    stream: bool = True


class Agent:
    """Tool-using agent driving a llama.cpp server through the native API."""

    def __init__(
        self,
        server: LlamaServer | CompletionBackend,
        registry: ToolRegistry,
        *,
        config: AgentConfig | None = None,
    ) -> None:
        self._server = server
        self._registry = registry
        self._config = config or AgentConfig()

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    async def aclose(self) -> None:
        close = getattr(self._server, "aclose", None)
        if close is not None:
            await close()

    async def run(
        self,
        prompt: str,
        *,
        history: Sequence[Message] | None = None,
        session_id: UUID | None = None,
        recorder: RunRecorder | None = None,
        on_event: EventCallback | None = None,
    ) -> AgentRunResult:
        """Run the agent loop for one user prompt.

        ``history`` is prior conversation state (usually loaded from a
        session); the returned result carries the extended history so the
        caller can persist it.
        """
        run_id = await recorder.on_run_start(session_id) if recorder else None
        messages: list[Message] = list(history or [])
        messages.append(Message(role="user", content=prompt))
        system_prompt = build_system_prompt(self._registry, persona=self._config.persona)
        schema = self._registry.envelope_schema()
        steps: list[StepRecord] = []
        result = AgentRunResult(status="failed", steps=steps, history=messages)
        invalid = 0

        async def emit(event: AgentEvent) -> None:
            if on_event is not None:
                maybe = on_event(event)
                if maybe is not None:
                    await maybe

        async def record(step: StepRecord) -> None:
            steps.append(step)
            if recorder is not None and run_id is not None:
                await recorder.on_step(run_id, step)

        try:
            for iteration in range(1, self._config.max_iterations + 1):
                await emit(IterationStarted(iteration))
                result.iterations = iteration
                transcript = self._config.template.build(system_prompt, messages)
                started = time.perf_counter()
                text = await self._generate(transcript, schema, emit)
                await record(
                    StepRecord(
                        "generate",
                        {"prompt_chars": len(transcript), "output": text},
                        (time.perf_counter() - started) * 1000.0,
                    )
                )
                messages.append(Message(role="assistant", content=text))
                try:
                    envelope = parse_envelope(text)
                except EnvelopeError as exc:
                    invalid += 1
                    await record(StepRecord("error", {"message": str(exc)}))
                    if invalid > self._config.max_invalid_retries:
                        result.status = "failed"
                        await emit(AgentFailed(f"model output kept failing validation: {exc}"))
                        return await self._finish(result, recorder, run_id)
                    messages.append(
                        Message(
                            role="user",
                            content=(
                                f"Your previous output was invalid: {exc}. "
                                "Respond again with a single JSON object containing "
                                'either "tool" or "final".'
                            ),
                        )
                    )
                    continue
                if envelope.kind == "final":
                    final = envelope.final
                    assert final is not None
                    result.status = "completed"
                    result.final = final
                    await emit(FinalAnswer(final))
                    return await self._finish(result, recorder, run_id)
                # tool branch (grammar makes unknown names near-impossible; checked anyway)
                call = envelope.tool
                assert call is not None
                if call.name not in self._registry:
                    messages.append(
                        Message(
                            role="user",
                            content=(
                                f"Unknown tool {call.name!r}. Registered tools: "
                                f"{', '.join(self._registry.names())}. Try again."
                            ),
                        )
                    )
                    continue
                await emit(ToolCallStarted(call.name, call.arguments))
                tool_result = await execute_tool(
                    self._registry,
                    call.name,
                    call.arguments,
                    timeout=self._config.tool_timeout,
                )
                await emit(ToolCallFinished(call.name, tool_result))
                await record(
                    StepRecord("tool_call", {"name": call.name, "arguments": call.arguments})
                )
                await record(
                    StepRecord("tool_result", tool_result.payload(), tool_result.duration_ms)
                )
                messages.append(
                    Message(
                        role="tool",
                        content=json.dumps(tool_result.payload()),
                        tool_name=call.name,
                    )
                )
            result.status = "max_iterations"
            await emit(
                AgentFailed(f"no final answer within {self._config.max_iterations} iterations")
            )
        except Exception as exc:
            result.status = "failed"
            await emit(AgentFailed(f"{type(exc).__name__}: {exc}"))
        return await self._finish(result, recorder, run_id)

    async def _generate(
        self,
        transcript: str,
        schema: dict[str, Any],
        emit: Callable[[AgentEvent], Awaitable[None]],
    ) -> str:
        if not self._config.stream:
            return await self._server.completion(
                transcript, json_schema=schema, sampling=self._config.sampling
            )
        pieces: list[str] = []
        async for chunk in self._server.completion_stream(
            transcript, json_schema=schema, sampling=self._config.sampling
        ):
            if chunk.content:
                pieces.append(chunk.content)
                await emit(TextDelta(chunk.content))
        return "".join(pieces)

    async def _finish(
        self,
        result: AgentRunResult,
        recorder: RunRecorder | None,
        run_id: UUID | None,
    ) -> AgentRunResult:
        if recorder is not None and run_id is not None:
            await recorder.on_run_end(run_id, result.status, result.final)
        return result
