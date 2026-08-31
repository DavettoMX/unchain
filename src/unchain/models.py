"""Shared domain models used across the agent loop and the state layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
StepKind = Literal["generate", "tool_call", "tool_result", "error"]
RunStatus = Literal["running", "completed", "failed", "max_iterations"]


@dataclass
class Message:
    """One entry of a multi-turn conversation transcript."""

    role: Role
    content: str
    tool_name: str | None = None


@dataclass
class StepRecord:
    """One recorded step of an agent run (generation, tool call, error)."""

    kind: StepKind
    payload: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
