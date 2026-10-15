"""Claude backend built on the official Anthropic Python SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentic_os.llm.base import LLMError, LLMProvider, LLMResponse, Message, ToolCall

if TYPE_CHECKING:  # pragma: no cover
    from agentic_os.tools.registry import Tool

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider(LLMProvider):
    """Calls Claude through the Messages API.

    * Thinking is left at the model default (adaptive on current models).
    * ``effort`` controls depth/token spend via ``output_config``.
    * ``fallbacks=True`` enables server-side refusal fallbacks
      (``fallbacks="default"``), so a policy decline is retried on a
      fallback model inside the same call.
    * Assistant turns are replayed with their original content blocks, which
      keeps thinking blocks valid across turns.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        effort: str | None = "high",
        fallbacks: bool = True,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on env
                raise LLMError(
                    "The 'anthropic' package is required: pip install 'agentic-os[anthropic]'"
                ) from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks

    # ------------------------------------------------------------------ API
    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": to_anthropic_messages(messages),
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [t.to_anthropic_schema() for t in tools]
        if self.effort:
            params["output_config"] = {"effort": self.effort}

        if self.fallbacks:
            response = self.client.beta.messages.create(
                **params, betas=[FALLBACK_BETA], fallbacks="default"
            )
        else:
            response = self.client.messages.create(**params)
        return from_anthropic_response(response)


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert neutral messages to Messages API params.

    Consecutive tool results are grouped into a single user message, and
    adjacent same-role messages are merged so roles always alternate.
    """
    out: list[dict[str, Any]] = []

    def push(role: str, blocks: list[Any]) -> None:
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for msg in messages:
        if msg.role == "user":
            push("user", [{"type": "text", "text": msg.content}])
        elif msg.role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": msg.content,
            }
            if msg.is_error:
                block["is_error"] = True
            push("user", [block])
        elif msg.role == "assistant":
            if msg.raw is not None:
                push("assistant", list(msg.raw))
                continue
            blocks: list[Any] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for call in msg.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                )
            push("assistant", blocks)
        else:
            raise ValueError(f"Unknown message role: {msg.role!r}")
    return out


def from_anthropic_response(response: Any) -> LLMResponse:
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {})))
    usage = {}
    if getattr(response, "usage", None) is not None:
        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(response.usage, "output_tokens", 0) or 0,
        }
    return LLMResponse(
        text="".join(texts),
        tool_calls=calls,
        stop_reason=response.stop_reason or "end_turn",
        usage=usage,
        model=getattr(response, "model", ""),
        raw_content=list(response.content),
    )
