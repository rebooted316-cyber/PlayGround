"""Model providers configured at runtime (from the War Room UI or code).

Each provider has a kind (``anthropic``, ``gemini`` or ``openai_compat``), an
API key, an optional base URL and a default model. Keys are stored server-side
in a JSON file readable only by the owner, and are never returned in full.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass
from typing import Any

from agentic_os.llm.base import LLMError, LLMProvider

KINDS = ("anthropic", "gemini", "openai_compat")

PRESETS: list[dict[str, str]] = [
    {"id": "anthropic", "label": "Anthropic (Claude)", "kind": "anthropic", "base_url": "", "default_model": "claude-opus-5-5"},
    {"id": "gemini", "label": "Google Gemini", "kind": "gemini", "base_url": "", "default_model": "gemini-flash-latest"},
    {"id": "openai", "label": "OpenAI", "kind": "openai_compat", "base_url": "https://api.openai.com/v1", "default_model": ""},
    {"id": "openrouter", "label": "OpenRouter", "kind": "openai_compat", "base_url": "https://openrouter.ai/api/v1", "default_model": "openrouter/free"},
    {"id": "xai", "label": "xAI (Grok)", "kind": "openai_compat", "base_url": "https://api.x.ai/v1", "default_model": ""},
    {"id": "groq", "label": "Groq", "kind": "openai_compat", "base_url": "https://api.groq.com/openai/v1", "default_model": ""},
    {"id": "deepseek", "label": "DeepSeek", "kind": "openai_compat", "base_url": "https://api.deepseek.com/v1", "default_model": ""},
    {"id": "mistral", "label": "Mistral", "kind": "openai_compat", "base_url": "https://api.mistral.ai/v1", "default_model": ""},
    {"id": "huggingface", "label": "Hugging Face", "kind": "openai_compat", "base_url": "https://router.huggingface.co/v1", "default_model": ""},
    {"id": "ollama", "label": "Ollama (lokalnie)", "kind": "openai_compat", "base_url": "http://localhost:11434/v1", "default_model": ""},
    {"id": "custom", "label": "Własny (zgodny z OpenAI)", "kind": "openai_compat", "base_url": "", "default_model": ""},
]

_ID = re.compile(r"[a-zA-Z0-9_-]{1,40}")


@dataclass
class ProviderConfig:
    id: str
    kind: str
    api_key: str = ""
    base_url: str = ""
    default_model: str = ""
    label: str = ""

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["api_key"] = mask(self.api_key)
        data["has_key"] = bool(self.api_key)
        return data


def mask(key: str) -> str:
    """Hide a secret; long keys keep their last 4 characters so they can be told apart."""
    if not key:
        return ""
    return ("•" * 8) + key[-4:] if len(key) >= 16 else "•" * 8


class ProviderRegistry:
    """Thread-safe store of provider configs, optionally persisted to ``path``."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._items: dict[str, ProviderConfig] = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for item in json.load(fh):
                    self._items[item["id"]] = ProviderConfig(**item)

    def _save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump([asdict(p) for p in self._items.values()], fh)
        os.chmod(self.path, 0o600)

    def upsert(self, config: ProviderConfig, keep_existing_key: bool = True) -> ProviderConfig:
        if not _ID.fullmatch(config.id):
            raise ValueError("Provider id must match [a-zA-Z0-9_-]{1,40}")
        if config.kind not in KINDS:
            raise ValueError(f"Provider kind must be one of {KINDS}")
        if config.kind == "openai_compat" and not config.base_url:
            raise ValueError("An OpenAI-compatible provider needs a base_url")
        if config.base_url and not re.match(r"https?://", config.base_url):
            raise ValueError("base_url must start with http:// or https://")
        with self._lock:
            existing = self._items.get(config.id)
            if keep_existing_key and existing and not config.api_key:
                config.api_key = existing.api_key  # editing other fields keeps the stored key
            self._items[config.id] = config
            self._save()
        return config

    def remove(self, provider_id: str) -> bool:
        with self._lock:
            found = self._items.pop(provider_id, None) is not None
            self._save()
        return found

    def get(self, provider_id: str) -> ProviderConfig:
        try:
            return self._items[provider_id]
        except KeyError:
            raise LLMError(f"No provider named {provider_id!r}") from None

    def ids(self) -> list[str]:
        return list(self._items)

    def public(self) -> list[dict[str, Any]]:
        return [p.public() for p in self._items.values()]

    def build(self, provider_id: str, model: str | None = None) -> LLMProvider:
        """Instantiate an :class:`LLMProvider` for a stored config."""
        cfg = self.get(provider_id)
        model = model or cfg.default_model
        if cfg.kind == "anthropic":
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on env
                raise LLMError("The 'anthropic' package is required: pip install 'agentic-os[anthropic]'") from exc
            from agentic_os.llm.anthropic_provider import AnthropicProvider

            client = anthropic.Anthropic(api_key=cfg.api_key or None, base_url=cfg.base_url or None)
            # Server-side refusal fallbacks exist only on the Anthropic API itself.
            return AnthropicProvider(model or "claude-opus-5-5", client=client, fallbacks=not cfg.base_url)
        if cfg.kind == "gemini":
            from agentic_os.llm.gemini_provider import DEFAULT_GEMINI_MODEL, GeminiProvider

            return GeminiProvider(model or DEFAULT_GEMINI_MODEL, api_key=cfg.api_key or None)
        from agentic_os.llm.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(model or "", base_url=cfg.base_url, api_key=cfg.api_key or None)
