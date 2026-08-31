from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from unchain.transport.http import HttpClient, HTTPStatusError, TransportError, parse_url


class MiniHttpServer:
    """Tiny asyncio HTTP/1.1 server speaking just enough for the tests."""

    def __init__(self) -> None:
        self.server: asyncio.Server | None = None
        self.connections = 0
        self.requests: list[tuple[str, dict[str, str], bytes]] = []

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)

    @property
    def port(self) -> int:
        assert self.server is not None and self.server.sockets
        return int(self.server.sockets[0].getsockname()[1])

    async def stop(self) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        try:
            while True:
                try:
                    status_line = await reader.readuntil(b"\r\n")
                except (asyncio.IncompleteReadError, ConnectionResetError):
                    break
                headers: dict[str, str] = {}
                while True:
                    line = await reader.readuntil(b"\r\n")
                    if line == b"\r\n":
                        break
                    name, _, value = line.decode("latin-1").strip().partition(":")
                    headers[name.strip().lower()] = value.strip()
                length = int(headers.get("content-length", "0"))
                body = await reader.readexactly(length) if length else b""
                path = status_line.decode("latin-1").split(" ")[1]
                self.requests.append((path, headers, body))
                response, keep_alive = self._route(path, body)
                writer.write(response)
                await writer.drain()
                if not keep_alive:
                    break
        finally:
            writer.close()

    def _route(self, path: str, body: bytes) -> tuple[bytes, bool]:
        if path == "/ok":
            return _plain(200, b'{"status": "ok"}'), True
        if path == "/echo":
            return _plain(200, body), True
        if path == "/error":
            return _plain(404, b"nope"), True
        if path == "/close":
            return (
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",
                False,
            )
        if path == "/chunked":
            return (
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                + _chunk(b"hello ")
                + _chunk(b"world")
                + b"0\r\n\r\n"
            ), True
        if path == "/sse":
            payload = (
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                + _chunk(b'data: {"content": "he"}\n\n')
                + _chunk(b'data: {"content": "llo"}\n\n')
                + _chunk(b"data: [DONE]\n\n")
                + b"0\r\n\r\n"
            )
            return payload, True
        return _plain(404, b"not found"), True


def _plain(status: int, body: bytes) -> bytes:
    reason = {200: "OK", 404: "Not Found"}[status]
    return f"HTTP/1.1 {status} {reason}\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body


def _chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


@pytest.fixture
async def mini() -> AsyncIterator[MiniHttpServer]:
    server = MiniHttpServer()
    await server.start()
    yield server
    await server.stop()


async def test_get_json(mini: MiniHttpServer) -> None:
    async with HttpClient("127.0.0.1", mini.port) as client:
        response = await client.request("GET", "/ok")
    assert response.status == 200
    assert response.json() == {"status": "ok"}


async def test_post_echoes_body(mini: MiniHttpServer) -> None:
    async with HttpClient("127.0.0.1", mini.port) as client:
        response = await client.request("POST", "/echo", json_body={"a": [1, 2]})
    assert response.json() == {"a": [1, 2]}
    path, headers, body = mini.requests[-1]
    assert path == "/echo"
    assert headers["content-type"] == "application/json"
    assert json.loads(body) == {"a": [1, 2]}


async def test_status_error_raises(mini: MiniHttpServer) -> None:
    async with HttpClient("127.0.0.1", mini.port) as client:
        with pytest.raises(HTTPStatusError) as info:
            await client.request("GET", "/error")
    assert info.value.status == 404
    assert info.value.body == b"nope"


async def test_chunked_body_reassembled(mini: MiniHttpServer) -> None:
    async with HttpClient("127.0.0.1", mini.port) as client:
        response = await client.request("GET", "/chunked")
    assert response.body == b"hello world"


async def test_stream_yields_chunks(mini: MiniHttpServer) -> None:
    async with HttpClient("127.0.0.1", mini.port) as client:
        chunks = [chunk async for chunk in client.stream("GET", "/chunked")]
    assert b"".join(chunks) == b"hello world"


async def test_stream_sse(mini: MiniHttpServer) -> None:
    from unchain.transport.sse import iter_sse_data

    async with HttpClient("127.0.0.1", mini.port) as client:
        raw = client.stream("POST", "/sse", json_body={})
        events = [event async for event in iter_sse_data(raw)]
    assert json.loads(events[0])["content"] == "he"
    assert json.loads(events[1])["content"] == "llo"
    assert events[2] == "[DONE]"


async def test_keep_alive_reuses_connection(mini: MiniHttpServer) -> None:
    async with HttpClient("127.0.0.1", mini.port) as client:
        await client.request("GET", "/ok")
        await client.request("GET", "/ok")
    assert mini.connections == 1
    assert len(mini.requests) == 2


async def test_connection_close_header_drops_pool(mini: MiniHttpServer) -> None:
    async with HttpClient("127.0.0.1", mini.port) as client:
        await client.request("GET", "/close")
        await client.request("GET", "/ok")
    assert mini.connections == 2


async def test_unreachable_server_raises_transport_error() -> None:
    client = HttpClient("127.0.0.1", 1, connect_timeout=0.5)
    with pytest.raises(TransportError):
        await client.request("GET", "/ok")
    await client.aclose()


def test_parse_url() -> None:
    assert parse_url("http://127.0.0.1:8080") == ("127.0.0.1", 8080)
    assert parse_url("127.0.0.1:9000") == ("127.0.0.1", 9000)
    assert parse_url("http://localhost") == ("localhost", 80)
    with pytest.raises(TransportError):
        parse_url("https://example.com")
