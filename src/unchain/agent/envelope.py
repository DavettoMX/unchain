"""Parsing of grammar-constrained agent output (the "envelope")."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError


class EnvelopeError(ValueError):
    """Model output failed the envelope contract."""


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Envelope:
    kind: Literal["tool", "final"]
    tool: ToolCall | None = None
    final: str | None = None


def parse_envelope(raw: str) -> Envelope:
    """Parse model output into a tool call or a final answer.

    The llama.cpp grammar already constrains decoding into this shape; this
    parser is the second validation boundary before any tool executes.
    """
    text = raw.strip()
    try:
        data = _loads_object(text)
    except EnvelopeError:
        data = _loads_object(_extract_json_block(text))  # lenient retry for fences/noise
    if "final" in data:
        final = data["final"]
        if not isinstance(final, str):
            raise EnvelopeError('"final" must be a string')
        return Envelope("final", final=final)
    if "tool" in data:
        node = data["tool"]
        if not isinstance(node, dict):
            raise EnvelopeError('"tool" must be an object with "name" and "arguments"')
        try:
            call = ToolCall.model_validate(node)
        except ValidationError as exc:
            raise EnvelopeError(f"invalid tool call: {exc.errors(include_url=False)}") from exc
        return Envelope("tool", tool=call)
    raise EnvelopeError('output must contain either "tool" or "final"')


def _loads_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EnvelopeError(f"output is not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise EnvelopeError("output must be a JSON object")
    return data


def _extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise EnvelopeError("output contains no JSON object")
    return text[start : end + 1]
