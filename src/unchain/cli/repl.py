"""Interactive multi-turn REPL on top of the agent loop."""

from __future__ import annotations

import asyncio
import sys
from typing import TextIO
from uuid import UUID

from unchain.agent import (
    Agent,
    AgentEvent,
    EventCallback,
    FinalAnswer,
    ToolCallFinished,
    ToolCallStarted,
)
from unchain.models import Message
from unchain.state.repo import SessionRepo


async def repl(
    agent: Agent,
    *,
    session_repo: SessionRepo | None = None,
    session_id: UUID | None = None,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    history: list[Message] = []
    if session_repo is not None and session_id is not None:
        history = list(await session_repo.list_messages(session_id))
        print(f"resuming session {session_id} ({len(history)} messages)", file=err)

    print("unchain repl — /exit to quit, /history to show transcript", file=err)
    while True:
        try:
            line = await asyncio.to_thread(input, "unchain> ")
        except (EOFError, KeyboardInterrupt):
            print(file=err)
            return 0
        command = line.strip()
        if not command:
            continue
        if command in ("/exit", "/quit"):
            return 0
        if command == "/history":
            _print_history(history, out)
            continue
        prior_len = len(history)
        result = await agent.run(
            command,
            history=history,
            session_id=session_id,
            on_event=_make_printer(err),
        )
        history = result.history
        if result.final is not None:
            print(result.final, file=out)
        else:
            print(f"[agent stopped: {result.status}]", file=err)
        if session_repo is not None and session_id is not None:
            await session_repo.append_all(session_id, history[prior_len:])


def _make_printer(err: TextIO) -> EventCallback:
    async def on_event(event: AgentEvent) -> None:
        match event:
            case ToolCallStarted(name, arguments):
                print(f"  -> tool {name}({arguments})", file=err, flush=True)
            case ToolCallFinished(name, result):
                outcome = result.value if result.ok else f"ERROR: {result.error}"
                print(f"  <- {name} = {outcome!s:.300}", file=err, flush=True)
            case FinalAnswer():
                print(file=err)
            case _:
                pass

    return on_event


def _print_history(history: list[Message], out: TextIO) -> None:
    for message in history:
        label = message.role if message.role != "tool" else f"tool:{message.tool_name}"
        print(f"[{label}] {message.content[:400]}", file=out)
