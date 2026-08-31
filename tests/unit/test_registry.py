from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from unchain.tools.registry import UnknownToolError


class WeatherArgs(BaseModel):
    city: str
    units: str = "celsius"


class _Handler:
    def __call__(self, city: str, units: str = "celsius") -> str:
        return f"{city}:{units}"


def make_registry() -> Any:
    from unchain.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.add(
        "get_weather",
        description="weather for a city",
        args_model=WeatherArgs,
        handler=_Handler(),
    )
    return registry


def test_register_and_get() -> None:
    registry = make_registry()
    tool = registry.get("get_weather")
    assert tool.description == "weather for a city"
    assert tool.args_model is WeatherArgs
    assert "get_weather" in registry
    assert registry.names() == ["get_weather"]


def test_duplicate_registration_rejected() -> None:
    registry = make_registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.add("get_weather", handler=_Handler())


def test_unknown_tool() -> None:
    registry = make_registry()
    with pytest.raises(UnknownToolError):
        registry.get("nope")


def test_validate_args_parses_defaults() -> None:
    registry = make_registry()
    args = registry.validate_args("get_weather", {"city": "CDMX"})
    assert isinstance(args, WeatherArgs)
    assert args.units == "celsius"


def test_envelope_schema_shape() -> None:
    registry = make_registry()
    schema = registry.envelope_schema()
    assert set(schema) == {"anyOf"}
    branches: list[dict[str, Any]] = schema["anyOf"]
    assert len(branches) == 2  # one tool + final
    tool_branch, final_branch = branches
    tool_obj = tool_branch["properties"]["tool"]
    assert tool_obj["properties"]["name"] == {"enum": ["get_weather"]}
    assert tool_branch["required"] == ["tool"]
    assert tool_branch["additionalProperties"] is False
    assert final_branch["properties"] == {"final": {"type": "string"}}
    assert final_branch["required"] == ["final"]


def test_envelope_schema_has_no_titles() -> None:
    registry = make_registry()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            assert "title" not in node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(registry.envelope_schema())


def test_envelope_schema_cached_and_invalidated() -> None:
    registry = make_registry()
    first = registry.envelope_schema()
    assert registry.envelope_schema() is first
    registry.add("other", handler=_Handler())
    assert registry.envelope_schema() is not first
    assert len(registry.envelope_schema()["anyOf"]) == 3


def test_fingerprint_stable_then_changes() -> None:
    registry = make_registry()
    fp = registry.fingerprint()
    assert registry.fingerprint() == fp
    registry.add("other", handler=_Handler())
    assert registry.fingerprint() != fp


def test_fingerprint_deterministic_across_instances() -> None:
    assert make_registry().fingerprint() == make_registry().fingerprint()


def test_tool_descriptions_mention_tools_and_schemas() -> None:
    registry = make_registry()
    description = registry.tool_descriptions()
    assert "get_weather" in description
    assert "weather for a city" in description
    assert "city" in description


def test_decorator_registration() -> None:
    from unchain.tools.registry import ToolRegistry

    registry = ToolRegistry()

    class Args(BaseModel):
        x: int

    @registry.register("double", description="doubles x", args_model=Args)
    def double(x: int) -> int:
        """docstring fallback"""
        return x * 2

    assert registry.get("double").handler is double
    args = registry.validate_args("double", {"x": 21})
    assert isinstance(args, Args)
    assert args.x == 21


def test_no_args_model_defaults_to_empty() -> None:
    from unchain.tools.registry import ToolRegistry

    registry = ToolRegistry()

    class Args(BaseModel):
        pass

    registry.add("ping", handler=lambda: "pong")
    schema = registry.envelope_schema()
    tool_branch = schema["anyOf"][0]
    arguments = json.dumps(tool_branch["properties"]["tool"]["properties"]["arguments"])
    assert json.loads(arguments) == {"properties": {}, "type": "object"}
