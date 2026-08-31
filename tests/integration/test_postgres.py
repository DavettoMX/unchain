"""Integration tests against Postgres (docker compose up unchain-db).

Opt-in: set UNCHAIN_DSN to run; migrations are applied on a fresh DB.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import pytest

from unchain.builtins import default_registry
from unchain.models import Message, StepRecord
from unchain.state.migrate import applied_versions, migrate
from unchain.state.pool import connect
from unchain.state.repo import PgRunRecorder, RunRepo, SessionRepo, ToolRepo

DSN = os.environ.get("UNCHAIN_DSN", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="UNCHAIN_DSN not set"),
]


@pytest.fixture
async def db() -> AsyncIterator[Any]:
    database = await connect(DSN)
    await migrate(database.pool)
    yield database
    await database.close()


async def test_migrate_is_idempotent(db: Any) -> None:
    versions = await migrate(db.pool)
    assert versions == []
    assert await applied_versions(db.pool) == [1, 2]


async def test_session_message_roundtrip(db: Any) -> None:
    sessions = SessionRepo(db)
    session = await sessions.create(meta={"origin": "test"})
    assert isinstance(session.id, UUID)

    await sessions.append_all(
        session.id,
        [
            Message(role="user", content="hi"),
            Message(role="assistant", content='{"final": "hello"}'),
            Message(role="tool", content='{"ok": true}', tool_name="echo"),
        ],
    )
    loaded = await sessions.list_messages(session.id)
    assert [message.content for message in loaded] == [
        "hi",
        '{"final": "hello"}',
        '{"ok": true}',
    ]
    assert loaded[2].tool_name == "echo"


async def test_run_recording(db: Any) -> None:
    sessions = SessionRepo(db)
    session = await sessions.create()

    recorder = PgRunRecorder(db)
    run_id = await recorder.on_run_start(session.id)
    assert run_id is not None
    await recorder.on_step(run_id, StepRecord("tool_call", {"name": "echo"}))
    await recorder.on_step(run_id, StepRecord("tool_result", {"ok": True}, 3.2))
    await recorder.on_run_end(run_id, "completed", "done")

    steps = await RunRepo(db).list_steps(run_id)
    assert steps == [
        ("tool_call", {"name": "echo"}),
        ("tool_result", {"ok": True}),
    ]
    row = await db.fetchrow("SELECT status, final FROM runs WHERE id = $1", run_id)
    assert row is not None
    assert row["status"] == "completed"
    assert row["final"] == "done"


async def test_tool_registry_sync(db: Any) -> None:
    repo = ToolRepo(db)
    registry = default_registry()
    await repo.sync_registry(registry)

    stored = {tool.name: tool for tool in await repo.list_all()}
    assert set(stored) >= {"calculator", "current_time", "echo"}
    assert stored["echo"].enabled is True
    assert stored["echo"].args_schema["properties"]["text"]["type"] == "string"

    await repo.set_enabled("echo", False)
    assert "echo" not in await repo.enabled_names()
    await repo.set_enabled("echo", True)
    assert "echo" in await repo.enabled_names()
