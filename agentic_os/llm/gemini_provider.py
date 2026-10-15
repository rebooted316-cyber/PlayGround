"""Gemini backend built on the official Google Gen AI SDK (``google-genai``)."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

from agentic_os.llm.base import LLMError, LLMProvider, LLMResponse, Message, ToolCall

if TYPE_CHECKING:  # pragma: no cover
    from agentic_os.tools.registry import Tool

DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
_GENERATED_PREFIX = "gemcall_"
_ids = itertools.count(1)
_REFUSAL_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}


def _types():
    try:
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - depends on env
        raise LLMError("The 'google-genai' package is required: pip install 'agentic-os[gemini]'") from exc
    return types


class GeminiProvider(LLMProvider):
    """Calls Gemini (and other models served by the Gemini API, such as Gemma).

    * Tools are sent as function declarations with their JSON Schema;
      the SDK's automatic function calling is disabled because the agent
      loop executes tools itself.
    * Model turns are replayed with their original content (including
      thought signatures), which Gemini requires for multi-step tool use.
    """

    name = "gemini"

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL, *, api_key: str | None = None, client: Any = None) -> None:
        if client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - depends on env
                raise LLMError("The 'google-genai' package is required: pip install 'agentic-os[gemini]'") from exc
            client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.client = client
        self.model = model

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        types = _types()
        config: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if system:
            config["system_instruction"] = system
        if tools:
            config["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t.name, description=t.description, parameters_json_schema=t.parameters
                        )
                        for t in tools
                    ]
                )
            ]
        response = self.client.models.generate_content(
            model=self.model,
            contents=to_gemini_contents(messages),
            config=types.GenerateContentConfig(**config),
        )
        return from_gemini_response(response, self.model)


def to_gemini_contents(messages: list[Message]) -> list[Any]:
    """Convert neutral messages to Gemini ``Content`` objects.

    Tool results become ``function_response`` parts in a user turn; Gemini
    matches them by function name, so names are looked up from the calls.
    """
    types = _types()
    out: list[Any] = []
    call_names: dict[str, str] = {}

    def push(role: str, parts: list[Any]) -> None:
        if out and out[-1].role == role:
            out[-1].parts.extend(parts)
        else:
            out.append(types.Content(role=role, parts=list(parts)))

    for msg in messages:
        if msg.role == "user":
            push("user", [types.Part(text=msg.content)])
        elif msg.role == "assistant":
            for call in msg.tool_calls:
                call_names[call.id] = call.name
            if msg.raw is not None:
                push("model", list(msg.raw.parts or []))
                continue
            parts: list[Any] = []
            if msg.content:
                parts.append(types.Part(text=msg.content))
            for call in msg.tool_calls:
                parts.append(types.Part(function_call=types.FunctionCall(name=call.name, args=call.arguments)))
            push("model", parts)
        elif msg.role == "tool":
            name = call_names.get(msg.tool_call_id or "")
            if name is None:
                raise ValueError(f"Tool result {msg.tool_call_id!r} has no matching tool call")
            payload = {"error": msg.content} if msg.is_error else {"result": msg.content}
            response_id = None if (msg.tool_call_id or "").startswith(_GENERATED_PREFIX) else msg.tool_call_id
            push(
                "user",
                [types.Part(function_response=types.FunctionResponse(id=response_id, name=name, response=payload))],
            )
        else:
            raise ValueError(f"Unknown message role: {msg.role!r}")
    return out


def from_gemini_response(response: Any, model: str = "") -> LLMResponse:
    usage_meta = getattr(response, "usage_metadata", None)
    usage = {
        "input_tokens": (getattr(usage_meta, "prompt_token_count", 0) or 0) if usage_meta else 0,
        "output_tokens": (getattr(usage_meta, "candidates_token_count", 0) or 0) if usage_meta else 0,
    }
    served = getattr(response, "model_version", None) or model
    candidates = getattr(response, "candidates", None) or []
    if not candidates or candidates[0].content is None:
        return LLMResponse(stop_reason="refusal", usage=usage, model=served)

    candidate = candidates[0]
    texts: list[str] = []
    calls: list[ToolCall] = []
    for part in candidate.content.parts or []:
        if part.function_call is not None:
            fc = part.function_call
            calls.append(
                ToolCall(id=fc.id or f"{_GENERATED_PREFIX}{next(_ids)}", name=fc.name, arguments=dict(fc.args or {}))
            )
        elif part.text and not part.thought:
            texts.append(part.text)

    reason = getattr(candidate.finish_reason, "name", str(candidate.finish_reason or "STOP"))
    if reason == "MAX_TOKENS":
        stop = "max_tokens"
    elif reason in _REFUSAL_REASONS:
        stop = "refusal"
    elif calls:
        stop = "tool_use"
    else:
        stop = "end_turn"
    return LLMResponse(
        text="".join(texts),
        tool_calls=calls,
        stop_reason=stop,
        usage=usage,
        model=served,
        raw_content=candidate.content,
    )
