"""Repositories: sessions/messages, runs/steps, dynamic tool definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from unchain.models import Message, RunStatus, StepRecord
from unchain.state.pool import Database
from unchain.tools.registry import ToolDefinition, ToolRegistry


@dataclass(frozen=True)
class Session:
    id: UUID
    created_at: datetime
    meta: dict[str, Any]


@dataclass(frozen=True)
class StoredTool:
    name: str
    version: int
    description: str
    args_schema: dict[str, Any]
    enabled: bool


class SessionRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def create(self, meta: dict[str, Any] | None = None) -> Session:
        row = await self._db.fetchrow(
            "INSERT INTO sessions (meta) VALUES ($1::jsonb) RETURNING *",
            json.dumps(meta or {}),
        )
        assert row is not None
        return Session(row["id"], row["created_at"], _loads(row["meta"]))

    async def get(self, session_id: UUID) -> Session | None:
        row = await self._db.fetchrow("SELECT * FROM sessions WHERE id = $1", session_id)
        if row is None:
            return None
        return Session(row["id"], row["created_at"], _loads(row["meta"]))

    async def append_message(self, session_id: UUID, message: Message) -> None:
        await self._db.execute(
            "INSERT INTO messages (session_id, role, content, tool_name) VALUES ($1, $2, $3, $4)",
            session_id,
            message.role,
            message.content,
            message.tool_name,
        )

    async def append_all(self, session_id: UUID, messages: list[Message]) -> None:
        for message in messages:
            await self.append_message(session_id, message)

    async def list_messages(self, session_id: UUID) -> list[Message]:
        rows = await self._db.fetch(
            "SELECT role, content, tool_name FROM messages WHERE session_id = $1 ORDER BY id",
            session_id,
        )
        return [
            Message(role=row["role"], content=row["content"], tool_name=row["tool_name"])
            for row in rows
        ]


class RunRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def start_run(self, session_id: UUID) -> UUID:
        row = await self._db.fetchrow(
            "INSERT INTO runs (session_id) VALUES ($1) RETURNING id", session_id
        )
        assert row is not None
        return row["id"]  # type: ignore[no-any-return]

    async def finish_run(self, run_id: UUID, status: RunStatus, final: str | None) -> None:
        await self._db.execute(
            "UPDATE runs SET status = $2, final = $3, finished_at = now() WHERE id = $1",
            run_id,
            status,
            final,
        )

    async def record_step(self, run_id: UUID, step: StepRecord) -> None:
        await self._db.execute(
            """
            INSERT INTO run_steps (run_id, kind, payload, duration_ms)
            VALUES ($1, $2, $3::jsonb, $4)
            """,
            run_id,
            step.kind,
            json.dumps(step.payload),
            step.duration_ms,
        )

    async def list_steps(self, run_id: UUID) -> list[tuple[str, dict[str, Any]]]:
        rows = await self._db.fetch(
            "SELECT kind, payload FROM run_steps WHERE run_id = $1 ORDER BY id", run_id
        )
        return [(row["kind"], _loads(row["payload"])) for row in rows]


class PgRunRecorder:
    """Bridges the agent loop's RunRecorder protocol onto the runs tables."""

    def __init__(self, db: Database) -> None:
        self._runs = RunRepo(db)

    async def on_run_start(self, session_id: UUID | None) -> UUID | None:
        if session_id is None:
            return None
        return await self._runs.start_run(session_id)

    async def on_step(self, run_id: UUID, step: StepRecord) -> None:
        await self._runs.record_step(run_id, step)

    async def on_run_end(self, run_id: UUID, status: RunStatus, final: str | None) -> None:
        await self._runs.finish_run(run_id, status, final)


class ToolRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def upsert(self, tool: ToolDefinition, *, enabled: bool = True) -> None:
        args_schema = tool.args_model.model_json_schema()
        await self._db.execute(
            """
            INSERT INTO tools (name, description, args_schema, enabled)
            VALUES ($1, $2, $3::jsonb, $4)
            ON CONFLICT (name) DO UPDATE SET
                description = EXCLUDED.description,
                args_schema = EXCLUDED.args_schema,
                enabled = EXCLUDED.enabled,
                version = tools.version + 1,
                updated_at = now()
            """,
            tool.name,
            tool.description,
            json.dumps(args_schema),
            enabled,
        )

    async def sync_registry(self, registry: ToolRegistry) -> None:
        for tool in registry.definitions():
            await self.upsert(tool)

    async def set_enabled(self, name: str, enabled: bool) -> None:
        await self._db.execute(
            "UPDATE tools SET enabled = $2, updated_at = now() WHERE name = $1", name, enabled
        )

    async def list_all(self) -> list[StoredTool]:
        rows = await self._db.fetch("SELECT * FROM tools ORDER BY name")
        return [
            StoredTool(
                name=row["name"],
                version=row["version"],
                description=row["description"],
                args_schema=_loads(row["args_schema"]),
                enabled=row["enabled"],
            )
            for row in rows
        ]

    async def enabled_names(self) -> list[str]:
        rows = await self._db.fetch("SELECT name FROM tools WHERE enabled ORDER BY name")
        return [row["name"] for row in rows]


def _loads(value: str | bytes | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    result: dict[str, Any] = json.loads(value)
    return result


__all__ = [
    "PgRunRecorder",
    "RunRepo",
    "Session",
    "SessionRepo",
    "StoredTool",
    "ToolRepo",
]
