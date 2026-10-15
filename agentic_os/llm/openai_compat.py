"""Any model behind an OpenAI-compatible Chat Completions API (standard library only).

Covers OpenAI, xAI (Grok), OpenRouter, Groq, DeepSeek, Mistral, Together,
Hugging Face Inference Providers, local Ollama / LM Studio / vLLM, and any
other server that speaks ``POST {base_url}/chat/completions``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from agentic_os.llm.base import LLMError, LLMProvider, LLMResponse, Message, ToolCall

if TYPE_CHECKING:  # pragma: no cover
    from agentic_os.tools.registry import Tool

_FINISH = {"stop": "end_turn", "tool_calls": "tool_use", "function_call": "tool_use", "length": "max_tokens",
           "content_filter": "refusal"}


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    """POST JSON and decode the JSON reply, turning HTTP errors into :class:`LLMError`."""
    data = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise LLMError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Cannot reach {url}: {exc.reason}") from exc
    try:
        return json.loads(body) if body.strip() else {}
    except json.JSONDecodeError as exc:
        raise LLMError(f"Non-JSON reply from {url}: {body[:200]}") from exc


class OpenAICompatProvider(LLMProvider):
    name = "openai_compat"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 300.0,
    ) -> None:
        if not model:
            raise LLMError("A model id is required for an OpenAI-compatible provider")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.extra_headers = dict(extra_headers or {})
        self.timeout = timeout

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages, system),
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
        headers = dict(self.extra_headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return from_openai_response(post_json(f"{self.base_url}/chat/completions", payload, headers, self.timeout), self.model)


def to_openai_messages(messages: list[Message], system: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    for msg in messages:
        if msg.role == "user":
            out.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": msg.content or None}
            if msg.tool_calls:
                item["tool_calls"] = [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                    for c in msg.tool_calls
                ]
            elif not msg.content:
                item["content"] = ""
            out.append(item)
        elif msg.role == "tool":
            content = f"ERROR: {msg.content}" if msg.is_error else msg.content
            out.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": content})
        else:
            raise ValueError(f"Unknown message role: {msg.role!r}")
    return out


def from_openai_response(data: dict[str, Any], model: str = "") -> LLMResponse:
    if data.get("error"):
        err = data["error"]
        raise LLMError(f"Provider error: {err.get('message', err) if isinstance(err, dict) else err}")
    choices = data.get("choices") or []
    usage = data.get("usage") or {}
    usage_out = {"input_tokens": usage.get("prompt_tokens", 0) or 0, "output_tokens": usage.get("completion_tokens", 0) or 0}
    served = data.get("model") or model
    if not choices:
        return LLMResponse(stop_reason="refusal", usage=usage_out, model=served)
    choice = choices[0]
    message = choice.get("message") or {}
    calls = []
    for i, call in enumerate(message.get("tool_calls") or []):
        fn = call.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except json.JSONDecodeError:
            args = {"_unparsed_arguments": raw_args}
        if not isinstance(args, dict):
            args = {"_unparsed_arguments": raw_args}
        calls.append(ToolCall(id=call.get("id") or f"call_{i}", name=fn.get("name", ""), arguments=args))
    content = message.get("content")
    if isinstance(content, list):  # some servers return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    stop = _FINISH.get(choice.get("finish_reason") or "", "tool_use" if calls else "end_turn")
    if calls and stop == "end_turn":
        stop = "tool_use"
    return LLMResponse(text=content or "", tool_calls=calls, stop_reason=stop, usage=usage_out, model=served)
