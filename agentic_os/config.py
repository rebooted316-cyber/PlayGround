"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _installed(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


@dataclass
class Settings:
    """Configuration for an :class:`~agentic_os.kernel.AgenticOS` instance.

    Environment variables (all optional):

    ``AGENTIC_OS_PROVIDER``  ``auto`` (default), ``anthropic``, ``gemini`` or ``mock``
    ``AGENTIC_OS_MODEL``     model id (default ``claude-opus-5-5``; Gemini uses
                             ``gemini-flash-latest`` unless a non-Claude id is set)
    ``AGENTIC_OS_EFFORT``    ``low`` | ``medium`` | ``high`` | ``xhigh`` | ``max``
    ``AGENTIC_OS_FALLBACKS`` enable server-side refusal fallbacks (default on)
    ``AGENTIC_OS_MAX_STEPS`` max reasoning/tool steps per agent run
    ``AGENTIC_OS_DATA_DIR``  where memory/feedback are persisted (optional)
    ``AGENTIC_OS_API_KEY``   bearer token required by the HTTP server (optional)
    """

    provider: str = field(default_factory=lambda: os.environ.get("AGENTIC_OS_PROVIDER", "auto"))
    model: str = field(default_factory=lambda: os.environ.get("AGENTIC_OS_MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: os.environ.get("AGENTIC_OS_EFFORT", "high"))
    fallbacks: bool = field(default_factory=lambda: _env_bool("AGENTIC_OS_FALLBACKS", True))
    max_steps: int = field(default_factory=lambda: int(os.environ.get("AGENTIC_OS_MAX_STEPS", "10")))
    data_dir: str | None = field(default_factory=lambda: os.environ.get("AGENTIC_OS_DATA_DIR"))
    api_key: str | None = field(default_factory=lambda: os.environ.get("AGENTIC_OS_API_KEY"))

    def resolved_provider(self) -> str:
        """Pick a concrete provider for ``auto``: Claude, then Gemini, when credentials exist."""
        if self.provider != "auto":
            return self.provider
        if (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")) and _installed("anthropic"):
            return "anthropic"
        if (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")) and _installed("google.genai"):
            return "gemini"
        return "mock"
