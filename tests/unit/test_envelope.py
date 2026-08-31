from __future__ import annotations

import pytest

from unchain.agent.envelope import EnvelopeError, parse_envelope


def test_tool_envelope() -> None:
    envelope = parse_envelope('{"tool": {"name": "calc", "arguments": {"a": 1}}}')
    assert envelope.kind == "tool"
    assert envelope.tool is not None
    assert envelope.tool.name == "calc"
    assert envelope.tool.arguments == {"a": 1}


def test_final_envelope() -> None:
    envelope = parse_envelope('{"final": "the answer is 42"}')
    assert envelope.kind == "final"
    assert envelope.final == "the answer is 42"


def test_surrounding_whitespace_tolerated() -> None:
    envelope = parse_envelope('  \n{"final": "ok"}  ')
    assert envelope.kind == "final"


def test_markdown_fence_extracted() -> None:
    raw = '```json\n{"tool": {"name": "echo", "arguments": {"text": "hi"}}}\n```'
    envelope = parse_envelope(raw)
    assert envelope.kind == "tool"
    assert envelope.tool is not None
    assert envelope.tool.name == "echo"


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "[1, 2, 3]",
        '"just a string"',
        "{}",
        '{"answer": "no key"}',
        '{"tool": "not an object"}',
        '{"tool": {"arguments": {}}}',
        '{"tool": {"name": "x"}}',
        '{"final": 42}',
    ],
)
def test_invalid_outputs_rejected(raw: str) -> None:
    with pytest.raises(EnvelopeError):
        parse_envelope(raw)


def test_error_message_is_actionable() -> None:
    with pytest.raises(EnvelopeError, match=r"either .tool. or .final"):
        parse_envelope("{}")
