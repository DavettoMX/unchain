from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from unchain.transport.sse import SSEParser, iter_sse_data


def collect(parser: SSEParser, chunks: list[bytes]) -> list[str]:
    events: list[str] = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.flush())
    return events


def test_single_event() -> None:
    parser = SSEParser()
    assert parser.feed(b"data: hello\n\n") == ["hello"]
    assert parser.flush() == []


def test_event_split_across_chunks() -> None:
    parser = SSEParser()
    events: list[str] = []
    for chunk in [b"da", b"ta: he", b"llo\n", b"\n"]:
        events.extend(parser.feed(chunk))
    assert events == ["hello"]


def test_crlf_line_endings() -> None:
    parser = SSEParser()
    assert parser.feed(b"data: hi\r\n\r\n") == ["hi"]


def test_no_leading_space_and_extra_space_preserved() -> None:
    parser = SSEParser()
    assert parser.feed(b"data:nospace\n\n") == ["nospace"]
    assert parser.feed(b"data:  twospace\n\n") == [" twospace"]


def test_multiline_data_joined() -> None:
    parser = SSEParser()
    assert parser.feed(b"data: line1\ndata: line2\n\n") == ["line1\nline2"]


def test_comments_and_other_fields_ignored() -> None:
    parser = SSEParser()
    assert parser.feed(b": ping\nevent: x\nid: 7\ndata: payload\n\n") == ["payload"]


def test_done_sentinel_is_plain_data() -> None:
    parser = SSEParser()
    assert parser.feed(b"data: [DONE]\n\n") == ["[DONE]"]


def test_flush_dispatches_pending_without_newline() -> None:
    parser = SSEParser()
    assert parser.feed(b"data: tail") == []
    assert parser.flush() == ["tail"]


def test_empty_data_line_yields_empty_string() -> None:
    parser = SSEParser()
    assert parser.feed(b"data:\n\n") == [""]


def test_blank_line_without_data_is_ignored() -> None:
    parser = SSEParser()
    assert parser.feed(b"\n\ndata: x\n\n") == ["x"]


async def test_iter_sse_data_reassembles_stream() -> None:
    async def chunks() -> AsyncIterator[bytes]:
        for chunk in [b'data: {"a": 1}\n\n', b'data: {"b": 2}\n\n', b"data: [DONE]\n\n"]:
            yield chunk

    events = [event async for event in iter_sse_data(chunks())]
    assert events == ['{"a": 1}', '{"b": 2}', "[DONE]"]


@pytest.mark.parametrize(
    "split_at",
    [1, 3, 7, 11],
)
def test_splits_at_arbitrary_boundaries(split_at: int) -> None:
    raw = b"data: alpha\ndata: beta\n\ndata: gamma\n\n"
    parser = SSEParser()
    events: list[str] = []
    for start in range(0, len(raw), split_at):
        events.extend(parser.feed(raw[start : start + split_at]))
    events.extend(parser.flush())
    assert events == ["alpha\nbeta", "gamma"]
