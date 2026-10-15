import json
import os
import stat
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentic_os import Agent, AgenticOS, Settings
from agentic_os.agents.external import ExternalAgent, ExternalAgentConfig, extract_a2a_text
from agentic_os.llm import LLMError, Message, ScriptedLLM
from agentic_os.llm.openai_compat import OpenAICompatProvider, from_openai_response, to_openai_messages
from agentic_os.providers import ProviderConfig, ProviderRegistry, mask
from agentic_os.server import make_server, serve
from agentic_os.tools import calculator


class Stub:
    """Tiny local HTTP server returning scripted JSON replies and recording requests."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
                status, payload = stub.replies.pop(0)
                data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def chat_reply(content=None, tool_calls=None, finish="stop"):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"model": "stub-model", "choices": [{"message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3}}


# ------------------------------------------------------------ OpenAI-compat
def test_openai_compat_provider_runs_tool_loop():
    stub = Stub([
        (200, chat_reply(None, [{"id": "c1", "type": "function", "function": {"name": "calculator", "arguments": '{"expression": "6*7"}'}}], "tool_calls")),
        (200, chat_reply("42")),
    ])
    try:
        llm = OpenAICompatProvider("stub-model", base_url=stub.url + "/v1", api_key="sk-test")
        result = Agent("m", llm, tools=[calculator]).run("6*7?")
    finally:
        stub.close()
    assert result.output == "42" and result.tool_calls[0]["result"] == "6*7 = 42"
    assert result.usage == {"input_tokens": 20, "output_tokens": 6}
    first, second = stub.requests
    assert first["path"] == "/v1/chat/completions" and first["auth"] == "Bearer sk-test"
    assert first["body"]["messages"][0]["role"] == "system"
    assert first["body"]["tools"][0]["function"]["name"] == "calculator"
    roles = [m["role"] for m in second["body"]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert second["body"]["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "6*7 = 42"}


def test_openai_compat_errors_and_edge_cases():
    stub = Stub([(401, {"error": {"message": "bad key"}})])
    try:
        with pytest.raises(LLMError, match="HTTP 401"):
            OpenAICompatProvider("m", base_url=stub.url).complete([Message.user("hi")])
    finally:
        stub.close()
    bad_args = from_openai_response(chat_reply(None, [{"id": "x", "function": {"name": "t", "arguments": "{oops"}}]))
    assert bad_args.tool_calls[0].arguments == {"_unparsed_arguments": "{oops"} and bad_args.stop_reason == "tool_use"
    assert from_openai_response({"choices": []}).stop_reason == "refusal"
    assert from_openai_response(chat_reply("cut", finish="length")).stop_reason == "max_tokens"
    with pytest.raises(LLMError):
        from_openai_response({"error": {"message": "nope"}})
    msgs = to_openai_messages([Message.user("q"), Message.tool("c", "boom", is_error=True)])
    assert msgs[1]["content"] == "ERROR: boom"
    with pytest.raises(LLMError):
        OpenAICompatProvider("")


# ---------------------------------------------------------------- providers
def test_provider_registry_masks_persists_and_validates(tmp_path):
    path = str(tmp_path / "providers.json")
    reg = ProviderRegistry(path)
    reg.upsert(ProviderConfig("groq", "openai_compat", api_key="gsk_secret_long_1234", base_url="https://api.groq.com/openai/v1", default_model="m1"))
    pub = reg.public()[0]
    assert pub["api_key"] == mask("gsk_secret_long_1234") and pub["api_key"].endswith("1234") and "secret" not in pub["api_key"]
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    reg.upsert(ProviderConfig("groq", "openai_compat", base_url="https://api.groq.com/openai/v1", default_model="m2"))
    assert ProviderRegistry(path).get("groq").api_key == "gsk_secret_long_1234"  # edit without key keeps key
    llm = ProviderRegistry(path).build("groq")
    assert isinstance(llm, OpenAICompatProvider) and llm.model == "m2"
    for bad in (ProviderConfig("bad id", "openai_compat", base_url="https://x"), ProviderConfig("x", "nope"),
                ProviderConfig("x", "openai_compat"), ProviderConfig("x", "openai_compat", base_url="ftp://x")):
        with pytest.raises(ValueError):
            reg.upsert(bad)
    assert reg.remove("groq") and not reg.remove("groq")
    with pytest.raises(LLMError):
        reg.build("groq")


# ----------------------------------------------------------- external agents
def test_external_agents_all_protocols():
    stub = Stub([
        (200, chat_reply("hermes says hi")),
        (200, {"jsonrpc": "2.0", "id": "1", "result": {"kind": "task", "artifacts": [{"parts": [{"kind": "text", "text": "a2a done"}]}]}}),
        (200, {"output": "webhook done"}),
        (200, "plain text reply"),
        (500, "boom"),
    ])
    try:
        oa = ExternalAgent(ExternalAgentConfig("hermes", "openai", stub.url + "/v1", api_key="k", model="hermes-x"))
        assert oa.run("hi").output == "hermes says hi"
        a2a = ExternalAgent(ExternalAgentConfig("ext", "a2a", stub.url + "/a2a"))
        assert a2a.run("task", context="ctx").output == "a2a done"
        hook = ExternalAgent(ExternalAgentConfig("hook", "webhook", stub.url + "/hook"))
        assert hook.run("t", context="c").output == "webhook done"
        assert hook.run("t").output == "plain text reply"
        failed = hook.run("t")
        assert failed.stopped_reason == "error" and "500" in failed.output
    finally:
        stub.close()
    assert stub.requests[0]["auth"] == "Bearer k"
    a2a_req = stub.requests[1]["body"]
    assert a2a_req["method"] == "message/send" and a2a_req["params"]["message"]["parts"][0]["text"] == "ctx\n\ntask"
    assert stub.requests[2]["body"] == {"task": "t", "context": "c"}
    assert extract_a2a_text({"parts": [{"kind": "text", "text": "direct"}]}) == "direct"
    assert extract_a2a_text({"status": {"message": {"parts": [{"text": "in status"}]}}}) == "in status"
    with pytest.raises(ValueError):
        ExternalAgent(ExternalAgentConfig("x", "openai", "https://x"))  # openai needs a model
    with pytest.raises(ValueError):
        ExternalAgent(ExternalAgentConfig("x", "a2a", "file:///etc/passwd"))


# ------------------------------------------------------------------- kernel
def test_kernel_persists_agents_providers_knowledge_and_chat_sessions(tmp_path):
    settings = Settings(provider="mock", data_dir=str(tmp_path))
    system = AgenticOS(settings)
    system.providers.upsert(ProviderConfig("local", "openai_compat", base_url="http://127.0.0.1:9/v1", default_model="m"))
    system.spawn("analyst", "You analyse data.", tool_tags=["math"], provider="local", model="m2")
    system.register_external(ExternalAgentConfig("hook", "webhook", "http://127.0.0.1:9/hook", api_key="secret"))
    system.ingest("Agentic OS keeps knowledge across restarts.", source="persist.md")
    assert system.describe("analyst")["provider"] == "local" and system.describe("analyst")["model"] == "m2"
    assert stat.S_IMODE(os.stat(tmp_path / "agents.json").st_mode) == 0o600
    system.shutdown()

    restored = AgenticOS(settings)
    assert {"assistant", "analyst", "hook"} <= set(restored.agents)
    assert isinstance(restored.agents["hook"], ExternalAgent) and restored.agents["hook"].config.api_key == "secret"
    assert restored.agents["analyst"].tools.names() == ["calculator"]
    assert len(restored.knowledge) == 1
    restored.kill("analyst")
    assert "analyst" not in AgenticOS(settings).agents
    restored.shutdown()


def test_chat_sessions_are_isolated_and_crews_run_as_processes():
    llm = ScriptedLLM([lambda m, s, t: f"history={len(m)}"] * 6)
    system = AgenticOS(Settings(provider="mock"), llm=llm)
    assert system.chat("hi", session="a").result["output"] == "history=1"
    assert system.chat("again", session="a").result["output"] == "history=3"
    assert system.chat("other", session="b").result["output"] == "history=1"
    assert system.agents["assistant"].memory is None  # sessions never leak into the shared agent
    system.reset_chat(session="a")
    assert system.chat("fresh", session="a").result["output"] == "history=1"
    proc = system.submit_crew("do it", ["assistant"], process="sequential")
    done = system.wait(proc.pid, timeout=10)
    assert done.status == "completed" and done.agent == "crew:sequential" and done.result["process"] == "sequential"
    system.shutdown()


def test_graph_links_similar_knowledge_and_memories():
    system = AgenticOS(Settings(provider="mock"))
    system.ingest("Solar panels convert sunlight into electricity.", source="solar.md")
    system.ingest("Wind turbines convert wind into electricity.", source="wind.md")
    system.memory.remember("The user cares about renewable electricity from sunlight", kind="fact")
    g = system.graph(threshold=0.1)
    kinds = {n["kind"] for n in g["nodes"]}
    assert {"document", "chunk", "fact"} <= kinds
    assert any(e["kind"] == "part" for e in g["edges"]) and any(e["kind"] == "similar" for e in g["edges"])
    system.shutdown()


# ------------------------------------------------------------------- server
def _req(base, method, path, body=None, token="pw"):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            return resp.status, json.loads(raw) if "json" in ctype else raw.decode()
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


@pytest.fixture
def war_room(tmp_path):
    system = AgenticOS(Settings(provider="mock", api_key="pw", data_dir=str(tmp_path)))
    httpd = make_server(system, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", system
    httpd.shutdown()
    httpd.server_close()
    system.shutdown()


def test_war_room_static_app_is_public_but_api_is_protected(war_room):
    base, _ = war_room
    status, html = _req(base, "GET", "/", token=None)
    assert status == 200 and "Agentic OS" in html and "app.js" in html
    for path in ("/app.js", "/app.css", "/manifest.webmanifest", "/sw.js", "/icon.svg", "/icon-192.png"):
        req = urllib.request.Request(base + path)
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200 and resp.read()
    assert _req(base, "GET", "/providers", token=None)[0] == 401
    assert _req(base, "GET", "/auth/check")[1] == {"ok": True}


def test_war_room_api_flow(war_room):
    base, system = war_room
    status, prov = _req(base, "POST", "/providers", {"id": "groq", "kind": "openai_compat", "api_key": "gsk_live_abcd1234",
                                                    "base_url": "https://api.groq.com/openai/v1", "default_model": "m"})
    assert status == 201 and "gsk_live" not in json.dumps(prov) and prov["has_key"]
    listing = _req(base, "GET", "/providers")[1]
    assert "gsk_live" not in json.dumps(listing) and any(p["id"] == "xai" for p in listing["presets"])
    assert _req(base, "POST", "/providers", {"id": "bad", "kind": "openai_compat"})[0] == 400

    status, agent = _req(base, "POST", "/agents", {"name": "coder", "provider": "groq", "model": "m2", "tools": ["calculator"]})
    assert status == 201 and agent["provider"] == "groq" and agent["tools"] == ["calculator"]
    status, ext = _req(base, "POST", "/agents/external", {"name": "grok", "protocol": "openai", "url": "https://api.x.ai/v1",
                                                         "api_key": "xai-secret", "model": "grok-x"})
    assert status == 201 and ext["kind"] == "external"
    agents = _req(base, "GET", "/agents")[1]
    assert {"coder", "grok"} <= {a["name"] for a in agents["agents"]} and agents["external_presets"]
    assert "xai-secret" not in json.dumps(agents)
    assert _req(base, "DELETE", "/agents/grok")[0] == 200 and "grok" not in system.agents
    assert _req(base, "DELETE", "/agents/ghost")[0] == 404

    status, proc = _req(base, "POST", "/chat", {"message": "What is 2+2?", "session": "s1"})
    assert status == 202
    done = system.wait(proc["pid"], timeout=10)
    assert done.status == "completed" and "4" in done.result["output"]
    assert _req(base, "POST", "/chat/reset", {"session": "s1"})[1] == {"ok": True}

    assert _req(base, "POST", "/knowledge", {"text": "The war room has a brain view.", "metadata": {"source": "ui.md"}})[0] == 201
    docs = _req(base, "GET", "/knowledge")[1]["documents"]
    assert docs[0]["source"] == "ui.md" and docs[0]["chunks"] == 1
    graph = _req(base, "GET", "/graph")[1]
    assert any(n["kind"] == "document" for n in graph["nodes"])

    status, crew = _req(base, "POST", "/crew", {"task": "2+2?", "agents": ["assistant"], "process": "sequential"})
    assert status == 202 and system.wait(crew["pid"], timeout=10).status == "completed"
    assert _req(base, "POST", "/crew", {"task": "x", "agents": []})[0] == 400
    assert _req(base, "DELETE", "/providers/groq")[0] == 200 and _req(base, "DELETE", "/providers/groq")[0] == 404


def test_serve_refuses_public_interface_without_password():
    system = AgenticOS(Settings(provider="mock"))
    with pytest.raises(SystemExit, match="AGENTIC_OS_API_KEY"):
        serve(system, host="0.0.0.0", port=0)
    system.shutdown()
