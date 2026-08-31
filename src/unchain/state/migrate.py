"""Minimal forward-only SQL migration runner (no external tooling)."""

from __future__ import annotations

import asyncpg

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            meta        JSONB NOT NULL DEFAULT '{}'::jsonb
        );

        CREATE TABLE IF NOT EXISTS messages (
            id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            session_id  UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role        TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
            content     TEXT NOT NULL,
            tool_name   TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS messages_session_idx ON messages (session_id, id);

        CREATE TABLE IF NOT EXISTS runs (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id  UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at TIMESTAMPTZ,
            status      TEXT NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running', 'completed', 'failed', 'max_iterations')),
            final       TEXT
        );
        CREATE INDEX IF NOT EXISTS runs_session_idx ON runs (session_id, started_at);

        CREATE TABLE IF NOT EXISTS run_steps (
            id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            run_id       UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            kind         TEXT NOT NULL
                         CHECK (kind IN ('generate', 'tool_call', 'tool_result', 'error')),
            payload      JSONB NOT NULL,
            duration_ms  DOUBLE PRECISION,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS run_steps_run_idx ON run_steps (run_id, id);
        """,
    ),
    (
        2,
        """
        CREATE TABLE IF NOT EXISTS tools (
            name         TEXT PRIMARY KEY,
            version      INT NOT NULL DEFAULT 1,
            description  TEXT NOT NULL DEFAULT '',
            args_schema  JSONB NOT NULL,
            enabled      BOOLEAN NOT NULL DEFAULT TRUE,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
    ),
]


async def migrate(pool: asyncpg.Pool) -> list[int]:
    """Apply pending migrations; returns the list of applied versions."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unchain_migrations (
                version    INT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        applied: set[int] = {
            record["version"]
            for record in await conn.fetch("SELECT version FROM unchain_migrations")
        }
    versions: list[int] = []
    for version, sql in MIGRATIONS:
        if version in applied:
            continue
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute("INSERT INTO unchain_migrations (version) VALUES ($1)", version)
        versions.append(version)
    return versions


async def applied_versions(pool: asyncpg.Pool) -> list[int]:
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT version FROM unchain_migrations ORDER BY version")
    return [row["version"] for row in rows]
