"""Multi-turn prompt assembly: templates, system prompt, token budgeting."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass

from unchain.models import Message
from unchain.tools.registry import ToolRegistry

OUTPUT_CONTRACT = """\
Respond with EXACTLY ONE JSON object, no other text:
{"tool": {"name": "<tool>", "arguments": {...}}}
to call a tool, or
{"final": "<answer>"}
when you have enough information to answer. Call tools instead of guessing \
facts you can compute or look up."""


@dataclass(frozen=True)
class PromptTemplate:
    """Role markers for the raw-prompt transcript (ChatML-style default).

    llama-server's native /completion endpoint applies no chat template, so
    Unchain assembles the transcript itself. The defaults match ChatML
    (Qwen family); override for other models.
    """

    system_start: str = "<|im_start|>system\n"
    system_end: str = "<|im_end|>\n"
    user_start: str = "<|im_start|>user\n"
    user_end: str = "<|im_end|>\n"
    assistant_start: str = "<|im_start|>assistant\n"
    assistant_end: str = "<|im_end|>\n"

    def build(self, system: str, messages: Iterable[Message]) -> str:
        parts: list[str] = []
        if system:
            parts += [self.system_start, system, self.system_end]
        for message in messages:
            parts.append(self._marker(message.role, opening=True))
            if message.role == "tool" and message.tool_name:
                parts.append(f"[tool result: {message.tool_name}]\n")
            parts.append(message.content)
            parts.append(self._marker(message.role, opening=False))
        parts.append(self.assistant_start)
        return "".join(parts)

    def _marker(self, role: str, *, opening: bool) -> str:
        pair = {
            "system": (self.system_start, self.system_end),
            "user": (self.user_start, self.user_end),
            "assistant": (self.assistant_start, self.assistant_end),
            "tool": (self.user_start, self.user_end),
        }[role]
        return pair[0] if opening else pair[1]


def build_system_prompt(registry: ToolRegistry, *, persona: str | None = None) -> str:
    header = persona or "You are Unchain, a tool-using agent running on a local llama.cpp server."
    tools = registry.tool_descriptions() or "(no tools registered)"
    return f"{header}\n\nAvailable tools:\n{tools}\n\nOutput contract:\n{OUTPUT_CONTRACT}"


async def fit_history(
    messages: Sequence[Message],
    *,
    count_tokens: Callable[[Sequence[Message]], Awaitable[int]],
    budget: int,
    keep_last: int = 4,
) -> list[Message]:
    """Drop oldest messages until the transcript fits the token budget.

    ``count_tokens`` receives the assembled transcript text and returns an
    int (typically ``LlamaServer.tokenize`` wrapped to count tokens).
    At least the last ``keep_last`` messages are always retained.
    """
    kept = list(messages)
    while len(kept) > keep_last:
        if await count_tokens(kept) <= budget:
            break
        kept = kept[1:]
    return kept
