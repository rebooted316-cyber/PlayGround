"""LLM layer (roadmap stage 4)."""

from agentic_os.llm.base import LLMError, LLMProvider, LLMResponse, Message, ToolCall
from agentic_os.llm.mock import HeuristicLLM, ScriptedLLM
from agentic_os.llm.tokens import estimate_tokens, truncate_to_tokens


def create_llm(settings=None) -> LLMProvider:
    """Build the provider described by ``settings`` (see :class:`Settings`)."""
    from agentic_os.config import Settings

    settings = settings or Settings()
    provider = settings.resolved_provider()
    if provider == "anthropic":
        from agentic_os.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings.model, effort=settings.effort, fallbacks=settings.fallbacks)
    if provider == "gemini":
        from agentic_os.llm.gemini_provider import DEFAULT_GEMINI_MODEL, GeminiProvider

        model = settings.model if not settings.model.startswith("claude") else DEFAULT_GEMINI_MODEL
        return GeminiProvider(model)
    if provider == "mock":
        return HeuristicLLM()
    raise LLMError(f"Unknown provider: {provider!r}")


__all__ = [
    "HeuristicLLM",
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "ScriptedLLM",
    "ToolCall",
    "create_llm",
    "estimate_tokens",
    "truncate_to_tokens",
]
