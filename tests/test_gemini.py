from types import SimpleNamespace

import pytest

types = pytest.importorskip("google.genai.types")

from agentic_os import Agent, Settings
from agentic_os.llm import Message, ToolCall, create_llm
from agentic_os.llm.gemini_provider import GeminiProvider, from_gemini_response, to_gemini_contents
from agentic_os.tools import calculator


def _response(*parts, finish="STOP", model="gemini-flash-latest-001"):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=list(parts)), finish_reason=finish)
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=12, candidates_token_count=5),
        model_version=model,
    )


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_tool_loop_through_gemini_provider():
    client = FakeClient(
        _response(
            types.Part(text="Checking."),
            types.Part(function_call=types.FunctionCall(id="fc1", name="calculator", args={"expression": "6*7"})),
        ),
        _response(types.Part(text="6*7 = 42")),
    )
    result = Agent("m", GeminiProvider(client=client), tools=[calculator]).run("What is 6*7?")
    assert result.output == "6*7 = 42" and result.tool_calls[0]["result"] == "6*7 = 42"
    assert result.usage == {"input_tokens": 24, "output_tokens": 10}

    first = client.calls[0]
    config = first["config"]
    assert first["model"] == "gemini-flash-latest"
    assert config.automatic_function_calling.disable is True
    decl = config.tools[0].function_declarations[0]
    assert decl.name == "calculator" and decl.parameters_json_schema["required"] == ["expression"]
    assert config.system_instruction

    replay = client.calls[1]["contents"]
    assert [c.role for c in replay] == ["user", "model", "user"]
    assert replay[1].parts[1].function_call.id == "fc1"  # original model turn replayed
    fr = replay[2].parts[0].function_response
    assert (fr.id, fr.name, fr.response) == ("fc1", "calculator", {"result": "6*7 = 42"})


def test_calls_without_ids_get_generated_ids_not_sent_back():
    resp = from_gemini_response(_response(types.Part(function_call=types.FunctionCall(name="calculator", args={}))))
    call = resp.tool_calls[0]
    assert call.id.startswith("gemcall_") and resp.stop_reason == "tool_use"
    contents = to_gemini_contents([Message.user("x"), resp.to_message(), Message.tool(call.id, "boom", is_error=True)])
    fr = contents[2].parts[0].function_response
    assert fr.id is None and fr.name == "calculator" and fr.response == {"error": "boom"}


def test_thoughts_are_hidden_and_stop_reasons_mapped():
    resp = from_gemini_response(_response(types.Part(text="secret plan", thought=True), types.Part(text="Answer")))
    assert resp.text == "Answer" and resp.stop_reason == "end_turn" and resp.model == "gemini-flash-latest-001"
    assert from_gemini_response(_response(types.Part(text="cut"), finish="MAX_TOKENS")).stop_reason == "max_tokens"
    assert from_gemini_response(_response(types.Part(text=""), finish="SAFETY")).stop_reason == "refusal"
    assert from_gemini_response(types.GenerateContentResponse(candidates=[])).stop_reason == "refusal"


def test_manual_messages_convert_without_raw():
    contents = to_gemini_contents(
        [Message.user("q"), Message.assistant("", [ToolCall("a", "calculator", {"expression": "1+1"})]), Message.tool("a", "2")]
    )
    assert contents[1].parts[0].function_call.args == {"expression": "1+1"}


def test_settings_select_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert Settings(provider="auto").resolved_provider() == "gemini"
    llm = create_llm(Settings(provider="gemini"))
    assert isinstance(llm, GeminiProvider) and llm.model == "gemini-flash-latest"
    assert create_llm(Settings(provider="gemini", model="gemma-4-31b-it")).model == "gemma-4-31b-it"
