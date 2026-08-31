"""Deterministic tool execution: validate, dispatch, never crash the loop."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from unchain.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    value: Any = None
    error: str | None = None
    duration_ms: float = 0.0

    def payload(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "value": self.value if self.ok else None,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
        }


async def execute_tool(
    registry: ToolRegistry,
    name: str,
    arguments: dict[str, Any],
    *,
    timeout: float = 60.0,
) -> ToolResult:
    """Validate arguments against the tool's model, then run the handler.

    Sync handlers are offloaded to a worker thread; async handlers are
    awaited directly. All failures (bad args, unknown tool, timeout,
    exception) come back as ``ToolResult(ok=False)`` — never as raises.
    """
    started = time.perf_counter()
    try:
        tool = registry.get(name)
    except KeyError:
        return _fail(f"unknown tool: {name}", started)
    try:
        args = tool.args_model.model_validate(arguments)
        kwargs = args.model_dump()
    except ValidationError as exc:
        return _fail(f"invalid arguments for {name}: {exc.errors(include_url=False)}", started)
    try:
        if asyncio.iscoroutinefunction(tool.handler):
            value = await asyncio.wait_for(tool.handler(**kwargs), timeout)
        else:
            value = await asyncio.wait_for(asyncio.to_thread(tool.handler, **kwargs), timeout)
    except TimeoutError:
        return _fail(f"tool {name} timed out after {timeout}s", started)
    except Exception as exc:
        return _fail(f"tool {name} raised {type(exc).__name__}: {exc}", started)
    return ToolResult(True, value=_serialize(value), duration_ms=_elapsed(started))


def _fail(message: str, started: float) -> ToolResult:
    return ToolResult(False, error=message, duration_ms=_elapsed(started))


def _elapsed(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _serialize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if value is None or isinstance(value, bool | int | float | str):
        return value
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return str(value)
