"""Incremental Server-Sent Events parsing (WHATWG subset: data fields only)."""

from __future__ import annotations

from collections.abc import AsyncIterator


class SSEParser:
    """Feed raw bytes, get complete SSE ``data`` payloads back.

    Handles CRLF and LF line endings, ``comment`` lines, ignored fields
    (``event:``, ``id:``, ``retry:``) and chunk boundaries that split
    lines at arbitrary positions.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._data: list[str] = []

    def feed(self, chunk: bytes) -> list[str]:
        self._buffer += chunk
        events: list[str] = []
        while True:
            idx = self._buffer.find(b"\n")
            if idx < 0:
                break
            line = bytes(self._buffer[:idx])
            del self._buffer[: idx + 1]
            if line.endswith(b"\r"):
                line = line[:-1]
            self._handle_line(line, events)
        return events

    def flush(self) -> list[str]:
        """Drain the parser at end-of-stream, dispatching pending data."""
        events: list[str] = []
        if self._buffer:
            line = bytes(self._buffer)
            self._buffer.clear()
            if line.endswith(b"\r"):
                line = line[:-1]
            self._handle_line(line, events)
        if self._data:
            events.append("\n".join(self._data))
            self._data = []
        return events

    def _handle_line(self, line: bytes, events: list[str]) -> None:
        if line == b"":
            if self._data:
                events.append("\n".join(self._data))
                self._data = []
            return
        if line.startswith(b":"):
            return
        name, sep, value = line.partition(b":")
        if name != b"data":
            return
        if sep and value[:1] == b" ":
            value = value[1:]
        self._data.append(value.decode("utf-8", errors="replace"))


async def iter_sse_data(chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """Convert a raw byte stream into SSE ``data`` payload strings."""
    parser = SSEParser()
    async for chunk in chunks:
        for event in parser.feed(chunk):
            yield event
    for event in parser.flush():
        yield event
