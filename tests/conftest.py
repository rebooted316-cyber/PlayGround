import pytest

from agentic_os.llm.base import LLMResponse, ToolCall


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Tests never call a real model."""
    monkeypatch.setenv("AGENTIC_OS_PROVIDER", "mock")
    monkeypatch.delenv("AGENTIC_OS_API_KEY", raising=False)
    monkeypatch.delenv("AGENTIC_OS_DATA_DIR", raising=False)


def tool_call(name, call_id="c1", **arguments):
    return LLMResponse(tool_calls=[ToolCall(call_id, name, arguments)], stop_reason="tool_use")
