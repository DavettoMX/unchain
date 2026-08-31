"""asyncpg pool creation with startup retry (compose containers may be slow)."""

from __future__ import annotations

import asyncio
from typing import Any

import asyncpg

DEFAULT_DSN = "postgresql://unchain:unchain@127.0.0.1:5432/unchain"


class Database:
    """Thin convenience wrapper over an asyncpg connection pool."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @property
    def pool(self) -> asyncpg.Pool:
        return self._pool

    async def execute(self, sql: str, *args: Any) -> str:
        async with self._pool.acquire() as conn:
            return str(await conn.execute(sql, *args))

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        async with self._pool.acquire() as conn:
            return list(await conn.fetch(sql, *args))

    async def fetchrow(self, sql: str, *args: Any) -> Any | None:
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(sql, *args)

    async def close(self) -> None:
        await self._pool.close()


async def connect(
    dsn: str = DEFAULT_DSN,
    *,
    min_size: int = 1,
    max_size: int = 5,
    attempts: int = 8,
    delay: float = 0.5,
) -> Database:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            pool = await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size)
            if pool is None:  # pragma: no cover - asyncpg returns a pool
                raise RuntimeError("asyncpg.create_pool returned None")
            return Database(pool)
        except (OSError, asyncpg.PostgresError, RuntimeError) as exc:
            last_error = exc
            await asyncio.sleep(delay)
    raise ConnectionError(f"could not reach postgres at {dsn}: {last_error}") from last_error
