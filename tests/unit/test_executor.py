from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel

from unchain.tools.executor import ToolResult, execute_tool
from unchain.tools.registry import ToolRegistry


class EchoArgs(BaseModel):
    text: str


class ModelArgs(BaseModel):
    pass


class Payload(BaseModel):
    value: int


def build_registry() -> tuple[ToolRegistry, dict[str, Any]]:
    registry = ToolRegistry()
    seen: dict[str, Any] = {}

    async def slow(text: str) -> str:
        await asyncio.sleep(10)
        return text

    async def quick(text: str) -> str:
        seen["quick"] = text
        return f"echo: {text}"

    def sync_fail() -> str:
        raise RuntimeError("boom")

    def returns_model() -> Payload:
        return Payload(value=7)

    def returns_dict() -> dict[str, Any]:
        return {"nested": [1, 2]}

    registry.add("slow", args_model=EchoArgs, handler=slow)
    registry.add("quick", args_model=EchoArgs, handler=quick)
    registry.add("fails", args_model=ModelArgs, handler=sync_fail)
    registry.add("model", args_model=ModelArgs, handler=returns_model)
    registry.add("dct", args_model=ModelArgs, handler=returns_dict)
    return registry, seen


async def test_async_handler_success() -> None:
    registry, seen = build_registry()
    result = await execute_tool(registry, "quick", {"text": "hi"})
    assert result.ok
    assert result.value == "echo: hi"
    assert seen == {"quick": "hi"}
    assert result.duration_ms >= 0


async def test_sync_handler_offloaded() -> None:
    registry, _ = build_registry()
    result = await execute_tool(registry, "model", {})
    assert result.ok
    assert result.value == {"value": 7}


async def test_dict_serialization() -> None:
    registry, _ = build_registry()
    result = await execute_tool(registry, "dct", {})
    assert result.value == {"nested": [1, 2]}


async def test_invalid_arguments_never_reaches_handler() -> None:
    registry, seen = build_registry()
    result = await execute_tool(registry, "quick", {"wrong": "field"})
    assert not result.ok
    assert result.error is not None
    assert "invalid arguments" in result.error
    assert seen == {}  # handler was never invoked


async def test_unknown_tool() -> None:
    registry, _ = build_registry()
    result = await execute_tool(registry, "ghost", {})
    assert not result.ok
    assert "unknown tool" in (result.error or "")


async def test_timeout() -> None:
    registry, _ = build_registry()
    result = await execute_tool(registry, "slow", {"text": "x"}, timeout=0.05)
    assert not result.ok
    assert "timed out" in (result.error or "")


async def test_handler_exception_captured() -> None:
    registry, _ = build_registry()
    result = await execute_tool(registry, "fails", {})
    assert not result.ok
    assert "RuntimeError" in (result.error or "")
    assert "boom" in (result.error or "")


def test_payload_shape() -> None:
    failure = ToolResult(False, error="bad", duration_ms=1.5)
    assert failure.payload() == {
        "ok": False,
        "value": None,
        "error": "bad",
        "duration_ms": 1.5,
    }
    success = ToolResult(True, value=[1], duration_ms=2.0)
    assert success.payload()["value"] == [1]
