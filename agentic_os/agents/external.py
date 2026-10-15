"""External agents: plug agents that run elsewhere into the OS.

An :class:`ExternalAgent` behaves like a native agent (``run``, ``as_tool``),
so it can join crews, be delegated to by a manager, and appear in the process
table. Three protocols cover most agent products:

* ``openai``: an OpenAI-compatible chat endpoint (e.g. Hermes via OpenRouter,
  Grok via xAI, or an agent gateway such as OpenClaw that exposes one).
* ``a2a``: the Agent2Agent protocol (JSON-RPC ``message/send``), the open
  standard for agent-to-agent calls.
* ``webhook``: a plain HTTP endpoint receiving ``{"task", "context"}`` and
  returning ``{"output": ...}`` (or plain text), the simplest bridge for a
  custom agent or a self-hosted CLI agent behind a tiny server.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from agentic_os.agents.base import AgentResult
from agentic_os.llm.base import LLMError
from agentic_os.llm.openai_compat import post_json
from agentic_os.providers import mask
from agentic_os.tools.registry import Tool, ToolRegistry

PROTOCOLS = ("openai", "a2a", "webhook")

AGENT_PRESETS: list[dict[str, str]] = [
    {"name": "hermes", "label": "Hermes (Nous)", "protocol": "openai", "url": "https://openrouter.ai/api/v1",
     "note": "Hermes przez OpenRouter: podaj klucz OpenRouter i model Hermes z openrouter.ai/models."},
    {"name": "grok", "label": "Grok (xAI)", "protocol": "openai", "url": "https://api.x.ai/v1",
     "note": "Klucz z console.x.ai i nazwa modelu Grok."},
    {"name": "openclaw", "label": "OpenClaw", "protocol": "openai", "url": "",
     "note": "Adres Twojej bramki OpenClaw z włączonym endpointem zgodnym z OpenAI."},
    {"name": "opencode", "label": "OpenCode", "protocol": "webhook", "url": "",
     "note": "OpenCode działa na Twoim komputerze: wystaw go pod publicznym adresem (A2A lub webhook)."},
    {"name": "a2a-agent", "label": "Dowolny agent A2A", "protocol": "a2a", "url": "",
     "note": "Każdy agent zgodny z Agent2Agent (np. zbudowany w Google ADK)."},
    {"name": "webhook-agent", "label": "Dowolny agent (webhook)", "protocol": "webhook", "url": "",
     "note": "POST {task, context} → {output}."},
]


@dataclass
class ExternalAgentConfig:
    name: str
    protocol: str
    url: str
    api_key: str = ""
    model: str = ""
    description: str = ""

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["api_key"] = mask(self.api_key)
        data["has_key"] = bool(self.api_key)
        return data


class ExternalAgent:
    """Adapter that makes a remote agent look like a native :class:`Agent`."""

    kind = "external"

    def __init__(self, config: ExternalAgentConfig, timeout: float = 300.0) -> None:
        if config.protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}")
        if not re.match(r"https?://", config.url or ""):
            raise ValueError("url must start with http:// or https://")
        if config.protocol == "openai" and not config.model:
            raise ValueError("The openai protocol needs a model id")
        self.config = config
        self.name = config.name
        self.instructions = config.description or f"External {config.protocol} agent at {config.url}"
        self.tools = ToolRegistry()
        self.memory = None
        self.llm = None
        self.timeout = timeout

    # ----------------------------------------------------------- protocols
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}

    def _call_openai(self, prompt: str) -> tuple[str, dict[str, int]]:
        from agentic_os.llm.openai_compat import from_openai_response

        payload = {"model": self.config.model, "messages": [{"role": "user", "content": prompt}]}
        data = post_json(f"{self.config.url.rstrip('/')}/chat/completions", payload, self._headers(), self.timeout)
        resp = from_openai_response(data, self.config.model)
        return resp.text, resp.usage

    def _call_a2a(self, prompt: str) -> str:
        payload = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "message/send",
            "params": {"message": {"role": "user", "kind": "message", "messageId": uuid.uuid4().hex,
                                   "parts": [{"kind": "text", "text": prompt}]}},
        }
        data = post_json(self.config.url, payload, self._headers(), self.timeout)
        if data.get("error"):
            raise LLMError(f"A2A error: {data['error']}")
        return extract_a2a_text(data.get("result") or {})

    def _call_webhook(self, task: str, context: str | None) -> str:
        body = json.dumps({"task": task, "context": context}).encode()
        req = urllib.request.Request(self.config.url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", **self._headers()})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                text = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            raise LLMError(f"HTTP {exc.code} from webhook: {exc.read().decode('utf-8', 'replace')[:300]}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Cannot reach webhook: {exc.reason}") from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(data, dict):
            for key in ("output", "result", "response", "text", "message"):
                if isinstance(data.get(key), str):
                    return data[key]
        return text

    # ---------------------------------------------------------------- API
    def run(self, task: str, context: str | None = None) -> AgentResult:
        prompt = f"{context}\n\n{task}" if context else task
        usage: dict[str, int] = {}
        try:
            if self.config.protocol == "openai":
                output, usage = self._call_openai(prompt)
            elif self.config.protocol == "a2a":
                output = self._call_a2a(prompt)
            else:
                output = self._call_webhook(task, context)
        except LLMError as exc:
            return AgentResult(output=f"External agent error: {exc}", agent=self.name, stopped_reason="error", steps=1)
        return AgentResult(output=output, agent=self.name, steps=1, usage=usage)

    def as_tool(self, name: str | None = None, description: str | None = None) -> Tool:
        def delegate(task: str) -> str:
            return self.run(task).output

        return Tool(
            name=name or f"ask_{self.name}",
            description=description or f"Delegate a self-contained task to the external '{self.name}' agent. {self.instructions}",
            parameters={"type": "object", "properties": {"task": {"type": "string", "description": "A complete instruction."}},
                        "required": ["task"]},
            func=delegate,
            tags={"agent", "external"},
        )


def extract_a2a_text(result: dict[str, Any]) -> str:
    """Pull text out of an A2A ``Message`` or ``Task`` result."""

    def texts(parts: Any) -> list[str]:
        return [p.get("text", "") for p in parts or [] if isinstance(p, dict) and p.get("text")]

    found = texts(result.get("parts"))
    for artifact in result.get("artifacts") or []:
        found += texts(artifact.get("parts"))
    if not found:
        found = texts(((result.get("status") or {}).get("message") or {}).get("parts"))
    return "\n".join(found) or json.dumps(result)[:2000]
