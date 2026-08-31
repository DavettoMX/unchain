"""Tool registry: Pydantic models to JSON Schema, envelope grammar building."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

Handler = Callable[..., Any]
AsyncHandler = Callable[..., Awaitable[Any]]


class UnknownToolError(KeyError):
    """Raised when a tool name is not present in the registry."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Handler


class ToolRegistry:
    """Registry of callable tools exposed to the agent.

    The registry derives, from the registered Pydantic arg models, a single
    JSON Schema (the "envelope") that constrains llama.cpp decoding so the
    model can only emit a valid tool call or a final answer.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._envelope: dict[str, Any] | None = None
        self._fingerprint: str | None = None

    def register(
        self,
        name: str,
        *,
        description: str = "",
        args_model: type[BaseModel] | None = None,
    ) -> Callable[[Handler], Handler]:
        def decorator(handler: Handler) -> Handler:
            self.add(name, description=description, args_model=args_model, handler=handler)
            return handler

        return decorator

    def add(
        self,
        name: str,
        *,
        description: str = "",
        args_model: type[BaseModel] | None = None,
        handler: Handler,
    ) -> None:
        if args_model is None:
            args_model = _empty_model()
        if not description:
            description = (handler.__doc__ or "").strip()
        if name in self._tools:
            raise ValueError(f"tool already registered: {name!r}")
        self._tools[name] = ToolDefinition(name, description, args_model, handler)
        self._envelope = None

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise UnknownToolError(name) from exc

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def definitions(self) -> list[ToolDefinition]:
        return [self._tools[name] for name in self.names()]

    def validate_args(self, name: str, arguments: dict[str, Any]) -> BaseModel:
        return self.get(name).args_model.model_validate(arguments)

    def envelope_schema(self) -> dict[str, Any]:
        """JSON Schema of the constrained output envelope (cached per state)."""
        if self._envelope is None:
            branches = [_tool_branch(tool) for tool in self._tools.values()]
            branches.append(_FINAL_BRANCH)
            self._envelope = {"anyOf": branches}
        return self._envelope

    def fingerprint(self) -> str:
        """Stable hash of the current envelope schema, for cache keys."""
        if self._fingerprint is None or self._envelope is None:
            canonical = json.dumps(self.envelope_schema(), sort_keys=True)
            self._fingerprint = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        return self._fingerprint

    def tool_descriptions(self) -> str:
        """Human-readable tool list for the system prompt."""
        lines = []
        for tool in self.definitions():
            args = json.dumps(_strip_titles(tool.args_model.model_json_schema()))
            lines.append(f"- {tool.name}: {tool.description}\n  arguments schema: {args}")
        return "\n".join(lines)


def _empty_model() -> type[BaseModel]:
    class _NoArgs(BaseModel):
        pass

    return _NoArgs


_FINAL_BRANCH: dict[str, Any] = {
    "type": "object",
    "properties": {"final": {"type": "string"}},
    "required": ["final"],
    "additionalProperties": False,
}


def _tool_branch(tool: ToolDefinition) -> dict[str, Any]:
    args_schema = _strip_titles(tool.args_model.model_json_schema())
    return {
        "type": "object",
        "properties": {
            "tool": {
                "type": "object",
                "properties": {
                    "name": {"enum": [tool.name]},
                    "arguments": args_schema,
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        },
        "required": ["tool"],
        "additionalProperties": False,
    }


def _strip_titles(schema: Any) -> Any:
    """Recursively drop pydantic's cosmetic ``title`` annotations."""
    if isinstance(schema, dict):
        return {key: _strip_titles(value) for key, value in schema.items() if key != "title"}
    if isinstance(schema, list):
        return [_strip_titles(item) for item in schema]
    return schema
