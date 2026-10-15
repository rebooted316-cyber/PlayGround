from types import SimpleNamespace

from agentic_os.config import Settings
from agentic_os.llm import HeuristicLLM, Message, ScriptedLLM, ToolCall, create_llm, estimate_tokens, truncate_to_tokens
from agentic_os.llm.anthropic_provider import FALLBACK_BETA, AnthropicProvider, to_anthropic_messages
from agentic_os.tools import calculator


class FakeClient:
    def __init__(self, response):
        self.calls = []
        self.response = response
        self.messages = SimpleNamespace(create=self._create("messages"))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create("beta")))

    def _create(self, path):
        def create(**kwargs):
            self.calls.append((path, kwargs))
            return self.response

        return create


def _response(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )


def test_tool_results_are_grouped_into_one_user_message():
    msgs = [
        Message.user("hi"),
        Message.assistant("", [ToolCall("a", "x", {}), ToolCall("b", "y", {})]),
        Message.tool("a", "1"),
        Message.tool("b", "boom", is_error=True),
        Message.user("next"),
    ]
    out = to_anthropic_messages(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    blocks = out[2]["content"]
    assert [b["type"] for b in blocks] == ["tool_result", "tool_result", "text"]
    assert blocks[1]["is_error"] is True
    assert out[1]["content"][0] == {"type": "tool_use", "id": "a", "name": "x", "input": {}}


def test_assistant_raw_content_is_replayed_unchanged():
    raw = [SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="ok")]
    out = to_anthropic_messages([Message.user("q"), Message.assistant("ok", raw=raw)])
    assert out[1]["content"] == raw


def test_provider_parses_response_and_sends_fallbacks():
    client = FakeClient(
        _response(
            SimpleNamespace(type="text", text="Let me compute."),
            SimpleNamespace(type="tool_use", id="t1", name="calculator", input={"expression": "1+1"}),
            stop_reason="tool_use",
        )
    )
    provider = AnthropicProvider(client=client, effort="high")
    resp = provider.complete([Message.user("1+1?")], system="sys", tools=[calculator])
    path, kwargs = client.calls[0]
    assert path == "beta"
    assert kwargs["fallbacks"] == "default" and kwargs["betas"] == [FALLBACK_BETA]
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["output_config"] == {"effort": "high"}
    assert kwargs["tools"][0]["name"] == "calculator" and "input_schema" in kwargs["tools"][0]
    assert "thinking" not in kwargs
    assert resp.text == "Let me compute."
    assert resp.tool_calls[0].arguments == {"expression": "1+1"}
    assert resp.usage == {"input_tokens": 11, "output_tokens": 7}
    assert resp.to_message().raw == resp.raw_content


def test_provider_without_fallbacks_uses_standard_endpoint():
    client = FakeClient(_response(SimpleNamespace(type="text", text="hi")))
    AnthropicProvider(client=client, fallbacks=False, effort=None).complete([Message.user("hi")])
    path, kwargs = client.calls[0]
    assert path == "messages" and "fallbacks" not in kwargs and "output_config" not in kwargs


def test_scripted_llm_records_calls_and_supports_callables():
    llm = ScriptedLLM(["one", lambda m, s, t: f"saw {len(m)}"])
    assert llm.complete([Message.user("a")]).text == "one"
    assert llm.complete([Message.user("a"), Message.user("b")]).text == "saw 2"
    assert len(llm.calls) == 2


def test_heuristic_llm_uses_calculator_for_math():
    resp = HeuristicLLM().complete([Message.user("what is 2 * (3 + 4)?")], tools=[calculator])
    assert resp.tool_calls[0].name == "calculator"
    assert resp.tool_calls[0].arguments["expression"] == "2 * (3 + 4)"


def test_create_llm_respects_provider_setting():
    assert isinstance(create_llm(Settings(provider="mock")), HeuristicLLM)


def test_token_estimates():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world, this is a test") > 3
    text = "word " * 500
    short = truncate_to_tokens(text, 50)
    assert estimate_tokens(short) <= 51 and short.endswith("…")
