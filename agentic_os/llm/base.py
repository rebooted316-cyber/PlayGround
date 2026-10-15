"""Provider-neutral message types and the LLM provider interface.

Roadmap stage 4 (Large Language Models): every agent in the OS talks to a
model through :class:`LLMProvider`, so the rest of the framework never
depends on a specific vendor SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from agentic_os.tools.registry import Tool


@dataclass
class ToolCall:
    """A request from the model to run a tool."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """One entry in a conversation.

    ``role`` is ``"user"``, ``"assistant"`` or ``"tool"``. Tool messages carry
    the result of a single :class:`ToolCall` (``tool_call_id``). ``raw`` keeps
    the provider-native content of an assistant turn so it can be replayed to
    the same provider unchanged (required for thinking blocks).
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    is_error: bool = False
    raw: Any = None

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: list[ToolCall] | None = None, raw: Any = None) -> Message:
        return cls(role="assistant", content=content, tool_calls=list(tool_calls or []), raw=raw)

    @classmethod
    def tool(cls, tool_call_id: str, content: str, is_error: bool = False) -> Message:
        return cls(role="tool", content=content, tool_call_id=tool_call_id, is_error=is_error)


@dataclass
class LLMResponse:
    """A normalized model response."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    raw_content: Any = None

    def to_message(self) -> Message:
        return Message.assistant(self.text, self.tool_calls, raw=self.raw_content)


class LLMError(RuntimeError):
    """Raised when a provider cannot produce a usable response."""


class LLMProvider(ABC):
    """Interface every model backend implements."""

    name: str = "base"

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        """Run one model turn over ``messages`` and return the response."""
