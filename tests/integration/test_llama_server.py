"""Integration tests against a live llama-server.

Opt-in: set UNCHAIN_SERVER (e.g. http://127.0.0.1:8080) to run.
Requires a llama.cpp server with a GGUF model loaded.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import pytest

from unchain.agent import Agent, AgentConfig
from unchain.builtins import default_registry
from unchain.transport import LlamaServer

SERVER = os.environ.get("UNCHAIN_SERVER", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not SERVER, reason="UNCHAIN_SERVER not set"),
]


@pytest.fixture
async def server() -> AsyncIterator[LlamaServer]:
    llama = LlamaServer(SERVER)
    yield llama
    await llama.aclose()


async def test_health(server: LlamaServer) -> None:
    assert await server.health() is True


async def test_props_reports_model(server: LlamaServer) -> None:
    props = await server.props()
    assert "model_path" in props or "default_generation_settings" in props


async def test_tokenize_detokenize_roundtrip(server: LlamaServer) -> None:
    text = "unchain agentic runtime"
    tokens = await server.tokenize(text)
    assert tokens
    assert await server.detokenize(tokens) == text


async def test_grammar_constrained_completion(server: LlamaServer) -> None:
    schema = {
        "type": "object",
        "properties": {"choice": {"enum": ["yes", "no"]}},
        "required": ["choice"],
        "additionalProperties": False,
    }
    prompt = 'Answer strictly with JSON. Question: is 2+2=4? Reply with {"choice": "yes"}.'
    output = await server.completion(prompt, json_schema=schema, timeout=120.0)
    data = json.loads(output)
    assert data == {"choice": "yes"} or data == {"choice": "no"}


async def test_agent_smoke_with_echo_tool(server: LlamaServer) -> None:
    agent = Agent(server, default_registry(), config=AgentConfig(max_iterations=8))
    result = await agent.run("Use the echo tool to echo the word unchain, then finish.")
    assert result.status == "completed"
    assert result.final is not None
