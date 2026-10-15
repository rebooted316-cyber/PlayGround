"""Short-term (working) memory: the recent conversation, within a token budget."""

from __future__ import annotations

from agentic_os.llm.base import Message
from agentic_os.llm.tokens import estimate_tokens


def _message_tokens(msg: Message) -> int:
    total = estimate_tokens(msg.content) + 4
    for call in msg.tool_calls:
        total += estimate_tokens(call.name) + estimate_tokens(str(call.arguments))
    return total


class ConversationBuffer:
    """Keeps whole conversation turns, evicting the oldest past ``max_tokens``.

    A turn starts at a user message and includes the assistant replies and
    tool results that follow it, so tool calls and their results are never
    separated by eviction.
    """

    def __init__(self, max_tokens: int = 8000) -> None:
        self.max_tokens = max_tokens
        self._turns: list[list[Message]] = []

    def add_turn(self, messages: list[Message]) -> None:
        if not messages:
            return
        if messages[0].role != "user":
            raise ValueError("A turn must start with a user message")
        self._turns.append(list(messages))
        self._evict()

    def _evict(self) -> None:
        while len(self._turns) > 1 and self.tokens() > self.max_tokens:
            self._turns.pop(0)

    def tokens(self) -> int:
        return sum(_message_tokens(m) for turn in self._turns for m in turn)

    def messages(self) -> list[Message]:
        return [m for turn in self._turns for m in turn]

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
