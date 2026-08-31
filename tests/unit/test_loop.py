from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from conftest import FakeBackend, demo_registry, transcript_roles
from unchain.agent import Agent, AgentConfig
from unchain.agent.history import PromptTemplate, build_system_prompt, fit_history
from unchain.models import Message


def make_agent(outputs: list[str], **config: Any) -> tuple[Agent, FakeBackend]:
    backend = FakeBackend(outputs)
    agent = Agent(backend, demo_registry(), config=AgentConfig(**config))
    return agent, backend


def tool_call(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps({"tool": {"name": name, "arguments": arguments}})


async def test_tool_call_then_final() -> None:
    agent, _ = make_agent(
        [
            tool_call("echo", {"text": "hi"}),
            '{"final": "all done"}',
        ]
    )
    result = await agent.run("say hi")
    assert result.status == "completed"
    assert result.final == "all done"
    assert transcript_roles(result.history) == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    kinds = [step.kind for step in result.steps]
    assert kinds == ["generate", "tool_call", "tool_result", "generate"]
    tool_step = result.steps[1]
    assert tool_step.payload["name"] == "echo"
    assert tool_step.payload["arguments"] == {"text": "hi"}


async def test_grammar_schema_sent_on_every_generation() -> None:
    agent, backend = make_agent(['{"final": "ok"}'])
    await agent.run("ping")
    assert backend.schemas == [agent.registry.envelope_schema()]
    assert backend.prompts, "backend saw no prompts"


async def test_prompt_contains_markers_and_contract() -> None:
    agent, backend = make_agent(['{"final": "ok"}'])
    await agent.run("what is 2+2")
    prompt = backend.prompts[0]
    assert "<|im_start|>user\nwhat is 2+2<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "echo" in prompt
    assert '"final"' in prompt


async def test_tool_result_fed_back_into_transcript() -> None:
    agent, backend = make_agent(
        [
            tool_call("echo", {"text": "hi"}),
            '{"final": "done"}',
        ]
    )
    await agent.run("x")
    second_prompt = backend.prompts[1]
    assert "[tool result: echo]" in second_prompt
    assert "echo: hi" in second_prompt


async def test_invalid_output_retried_with_correction() -> None:
    agent, _ = make_agent(["utter nonsense", '{"final": "recovered"}'])
    result = await agent.run("x")
    assert result.status == "completed"
    assert result.final == "recovered"
    correction = result.history[2]
    assert correction.role == "user"
    assert "invalid" in correction.content


async def test_repeated_invalid_output_fails() -> None:
    agent, _ = make_agent(["bad", "worse", "nope", "still bad", '{"final": "late"}'])
    result = await agent.run("x")
    assert result.status == "failed"
    assert result.final is None


async def test_unknown_tool_gets_corrective_feedback() -> None:
    agent, _backend = make_agent(
        [
            tool_call("ghost", {}),
            '{"final": "ok"}',
        ]
    )
    result = await agent.run("x")
    assert result.status == "completed"
    assert "Unknown tool" in result.history[2].content
    assert "echo" in result.history[2].content


async def test_failing_tool_does_not_crash_loop() -> None:
    agent, _ = make_agent(
        [
            tool_call("fail", {}),
            '{"final": "survived"}',
        ]
    )
    result = await agent.run("x")
    assert result.status == "completed"
    tool_message = result.history[2]
    assert tool_message.role == "tool"
    payload = json.loads(tool_message.content)
    assert payload["ok"] is False
    assert "RuntimeError" in payload["error"]


async def test_max_iterations_exhausted() -> None:
    agent, _ = make_agent([tool_call("echo", {"text": "loop"})] * 5, max_iterations=5)
    result = await agent.run("x")
    assert result.status == "max_iterations"
    assert result.iterations == 5
    assert result.final is None


async def test_events_streamed_in_order() -> None:
    agent, _ = make_agent(
        [
            tool_call("echo", {"text": "hi"}),
            '{"final": "done"}',
        ]
    )
    names: list[str] = []

    async def on_event(event: Any) -> None:
        names.append(type(event).__name__)

    await agent.run("x", on_event=on_event)
    # collapse consecutive TextDelta events (streaming emits several per turn)
    collapsed = [n for i, n in enumerate(names) if i == 0 or names[i - 1] != n]
    assert collapsed == [
        "IterationStarted",
        "TextDelta",
        "ToolCallStarted",
        "ToolCallFinished",
        "IterationStarted",
        "TextDelta",
        "FinalAnswer",
    ]


async def test_non_streaming_path() -> None:
    agent, _ = make_agent(['{"final": "ok"}'], stream=False)
    result = await agent.run("x")
    assert result.final == "ok"


async def test_prior_history_is_extended_and_returned() -> None:
    agent, _ = make_agent(['{"final": "ok"}'])
    prior = [Message(role="user", content="earlier"), Message(role="assistant", content="...")]
    result = await agent.run("next", history=prior)
    assert result.history[:2] == prior
    assert result.history[2].content == "next"


# -- prompt history / template ---------------------------------------------


def test_template_custom_markers() -> None:
    template = PromptTemplate(
        system_start="SYS: ",
        system_end="\n",
        user_start="USER: ",
        user_end="\n",
        assistant_start="ASSISTANT: ",
        assistant_end="\n",
    )
    text = template.build("be good", [Message(role="user", content="hi")])
    assert text == "SYS: be good\nUSER: hi\nASSISTANT: "


def test_system_prompt_lists_tools() -> None:
    prompt = build_system_prompt(demo_registry())
    assert "echo" in prompt
    assert "calculator" not in prompt  # demo registry only has echo/fail
    assert '"final"' in prompt


async def test_fit_history_drops_oldest_until_budget() -> None:
    messages = [Message(role="user", content=f"m{i}") for i in range(10)]

    async def count(msgs: Sequence[Message]) -> int:
        return len(msgs) * 10  # 10 fake "tokens" per message

    kept = await fit_history(messages, count_tokens=count, budget=60, keep_last=2)
    assert len(kept) == 6  # 6 * 10 <= 60, and never below keep_last
    assert kept[0].content == "m4"


async def test_fit_history_never_drops_below_keep_last() -> None:
    messages = [Message(role="user", content=f"m{i}") for i in range(10)]

    async def count(msgs: Sequence[Message]) -> int:
        return 10_000

    kept = await fit_history(messages, count_tokens=count, budget=10, keep_last=3)
    assert kept == messages[-3:]
