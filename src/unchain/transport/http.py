"""Minimal asynchronous HTTP/1.1 client built on raw asyncio streams.

Zero third-party dependencies: plain sockets, Content-Length and chunked
transfer decoding, connection reuse for sequential requests. TLS is out of
scope (local-first llama.cpp deployments speak plain HTTP on localhost).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

_MAX_LINE = 64 * 1024
_READ_TIMEOUT = 60.0


class TransportError(Exception):
    """Base class for transport-level failures."""


class ProtocolError(TransportError):
    """The server produced a malformed HTTP response."""


class HTTPStatusError(TransportError):
    def __init__(self, status: int, reason: str, path: str, body: bytes) -> None:
        self.status = status
        self.reason = reason
        self.path = path
        self.body = body
        super().__init__(
            f"HTTP {status} {reason} for {path}: {body.decode(errors='replace')[:512]}"
        )


@dataclass(frozen=True)
class HttpResponse:
    status: int
    reason: str
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


def parse_url(base_url: str) -> tuple[str, int]:
    url = base_url if "://" in base_url else f"http://{base_url}"
    parts = urlsplit(url)
    if parts.scheme != "http":
        raise TransportError(f"only plain http:// URLs are supported, got {url!r}")
    host = parts.hostname
    if not host:
        raise TransportError(f"URL has no host: {url!r}")
    return host, parts.port or 80


class _StaleConnection(TransportError):
    """A pooled connection died before yielding a response head."""


class HttpClient:
    """Single-connection async HTTP/1.1 client with keep-alive reuse.

    One in-flight request at a time (the agent loop is strictly sequential).
    Streaming responses always close the connection afterwards.
    """

    def __init__(self, host: str, port: int, *, connect_timeout: float = 10.0) -> None:
        self._host = host
        self._port = port
        self._connect_timeout = connect_timeout
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._alive = False

    async def aclose(self) -> None:
        self._alive = False
        writer, self._writer = self._writer, None
        self._reader = None
        if writer is not None:
            writer.close()

    async def __aenter__(self) -> HttpClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 120.0,
    ) -> HttpResponse:
        """Send one request and read the full response body."""
        body = b"" if json_body is None else json.dumps(json_body).encode()
        for attempt in range(2):
            try:
                status, reason, resp_headers = await self._send_head(method, path, body, headers)
            except _StaleConnection:
                if attempt == 1:
                    raise
                continue  # pooled connection was dead; retry once on a fresh one
            raw = await asyncio.wait_for(self._read_body(resp_headers), timeout)
            self._release(resp_headers)
            if status >= 400:
                raise HTTPStatusError(status, reason, path, raw)
            return HttpResponse(status, reason, resp_headers, raw)
        raise TransportError("unreachable")  # pragma: no cover

    async def stream(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        headers: dict[str, str] | None = None,
        idle_timeout: float = 300.0,
    ) -> AsyncIterator[bytes]:
        """Send one request and yield response body chunks as they arrive."""
        body = b"" if json_body is None else json.dumps(json_body).encode()
        status, reason, resp_headers = await self._send_head(method, path, body, headers)
        reader = self._reader
        assert reader is not None
        if status >= 400:
            raw = await self._read_body(resp_headers)
            self._release(resp_headers)
            raise HTTPStatusError(status, reason, path, raw)
        try:
            if _is_chunked(resp_headers):
                while True:
                    size = await self._read_chunk_size(reader, idle_timeout)
                    if size == 0:
                        await self._read_trailers(reader, idle_timeout)
                        break
                    chunk = await asyncio.wait_for(reader.readexactly(size + 2), idle_timeout)
                    if not chunk.endswith(b"\r\n"):
                        raise ProtocolError("chunk not terminated by CRLF")
                    yield chunk[:-2]
            elif "content-length" in resp_headers:
                remaining = _content_length(resp_headers)
                while remaining > 0:
                    chunk = await asyncio.wait_for(
                        reader.readexactly(min(remaining, 64 * 1024)), idle_timeout
                    )
                    remaining -= len(chunk)
                    yield chunk
            else:
                while True:
                    chunk = await asyncio.wait_for(reader.read(64 * 1024), idle_timeout)
                    if not chunk:
                        break
                    yield chunk
        finally:
            await self.aclose()

    # -- internals ---------------------------------------------------------

    async def _connect(self) -> None:
        if self._alive and self._reader is not None and self._writer is not None:
            return
        await self.aclose()
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port, limit=_MAX_LINE),
                self._connect_timeout,
            )
        except TimeoutError as exc:
            raise TransportError(
                f"connect timeout to {self._host}:{self._port} after {self._connect_timeout}s"
            ) from exc
        except OSError as exc:
            raise TransportError(f"connect to {self._host}:{self._port} failed: {exc}") from exc
        self._alive = True

    async def _send_head(
        self, method: str, path: str, body: bytes, extra: dict[str, str] | None
    ) -> tuple[int, str, dict[str, str]]:
        lines = [
            f"{method} {path} HTTP/1.1",
            f"Host: {self._host}:{self._port}",
            "Accept: */*",
        ]
        if body:
            lines.append("Content-Type: application/json")
        lines.append(f"Content-Length: {len(body)}")
        if extra:
            for key, value in extra.items():
                lines.append(f"{key}: {value}")
        head = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")
        try:
            await self._connect()
            writer = self._writer
            reader = self._reader
            assert writer is not None and reader is not None
            writer.write(head + body)
            await writer.drain()
            return await self._read_head(reader)
        except (BrokenPipeError, ConnectionResetError, asyncio.IncompleteReadError) as exc:
            self._alive = False
            raise _StaleConnection(str(exc)) from exc

    async def _read_head(self, reader: asyncio.StreamReader) -> tuple[int, str, dict[str, str]]:
        try:
            status_line = await reader.readuntil(b"\r\n")
        except asyncio.LimitOverrunError as exc:
            raise ProtocolError("status line too long") from exc
        parts = status_line.decode("latin-1").strip().split(" ", 2)
        if len(parts) < 2 or not parts[1].isdigit():
            raise ProtocolError(f"malformed status line: {status_line!r}")
        status = int(parts[1])
        reason = parts[2] if len(parts) == 3 else ""
        headers: dict[str, list[str]] = {}
        while True:
            line = await reader.readuntil(b"\r\n")
            if line == b"\r\n":
                break
            text = line.decode("latin-1").strip()
            name, _, value = text.partition(":")
            headers.setdefault(name.strip().lower(), []).append(value.strip())
        merged = {key: ", ".join(values) for key, values in headers.items()}
        return status, reason, merged

    async def _read_body(self, resp_headers: dict[str, str]) -> bytes:
        reader = self._reader
        assert reader is not None
        if _is_chunked(resp_headers):
            chunks: list[bytes] = []
            while True:
                size = await self._read_chunk_size(reader, _READ_TIMEOUT)
                if size == 0:
                    await self._read_trailers(reader, _READ_TIMEOUT)
                    return b"".join(chunks)
                chunk = await reader.readexactly(size + 2)
                if not chunk.endswith(b"\r\n"):
                    raise ProtocolError("chunk not terminated by CRLF")
                chunks.append(chunk[:-2])
        if "content-length" in resp_headers:
            length = _content_length(resp_headers)
            return b"" if length == 0 else await reader.readexactly(length)
        return await reader.read(-1)  # read until EOF (connection closes)

    async def _read_chunk_size(self, reader: asyncio.StreamReader, timeout: float) -> int:
        line = await asyncio.wait_for(reader.readuntil(b"\r\n"), timeout)
        text = line.decode("latin-1").split(";", 1)[0].strip()
        try:
            return int(text, 16)
        except ValueError as exc:
            raise ProtocolError(f"malformed chunk size: {text!r}") from exc

    async def _read_trailers(self, reader: asyncio.StreamReader, timeout: float) -> None:
        while True:
            line = await asyncio.wait_for(reader.readuntil(b"\r\n"), timeout)
            if line == b"\r\n":
                return

    def _release(self, resp_headers: dict[str, str]) -> None:
        if resp_headers.get("connection", "").lower() == "close":
            self._alive = False


def _is_chunked(headers: dict[str, str]) -> bool:
    return "chunked" in headers.get("transfer-encoding", "").lower()


def _content_length(headers: dict[str, str]) -> int:
    value = headers.get("content-length", "")
    try:
        return int(value)
    except ValueError as exc:
        raise ProtocolError(f"malformed content-length: {value!r}") from exc
