import io
import json
import os
import stat
import sys
import textwrap
import threading
import urllib.error
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import tool_call

from agentic_os import AgenticOS, Settings
from agentic_os.llm import LLMResponse, ScriptedLLM, ToolCall
from agentic_os.mcp import MCPClient, MCPError, MCPServerConfig
from agentic_os.memory.importer import parse_export, parse_text_memories
from agentic_os.mods import Mod, ModParam, ModRegistry, mod_from_dict
from agentic_os.server import make_server
from agentic_os.skills import SkillRegistry, parse_skill_md

SKILL_MD = textwrap.dedent("""\
    ---
    name: invoice-checker
    description: Check invoices for arithmetic errors.
    tools: calculator
    ---
    1. Recompute every line total with the calculator.
    2. Report mismatches.
    """)


# ------------------------------------------------------------------ skills
def test_parse_skill_md_and_registry(tmp_path):
    skill = parse_skill_md(SKILL_MD)
    assert (skill.name, skill.tools) == ("invoice-checker", ["calculator"])
    assert skill.instructions.startswith("1. Recompute")
    bare = parse_skill_md("# Summarize meeting notes\nUse bullet points.", fallback_name="Meeting Notes!")
    assert bare.name == "Meeting-Notes" and bare.description == "Summarize meeting notes"
    with pytest.raises(ValueError):
        parse_skill_md("---\nname: x\n---\n")
    reg = SkillRegistry(str(tmp_path / "skills.json"))
    reg.add(skill)
    assert SkillRegistry(str(tmp_path / "skills.json")).names() == ["invoice-checker"]
    assert "invoice-checker: Check invoices" in reg.catalog(["invoice-checker", "ghost"])
    loader = reg.loader_tool(allowed=["invoice-checker"])
    assert "Recompute" in loader(name="invoice-checker") and "Unknown skill" in loader(name="other")


def test_agent_with_skill_sees_catalog_and_loads_it():
    load = LLMResponse(tool_calls=[ToolCall("c1", "load_skill", {"name": "invoice-checker"})], stop_reason="tool_use")
    llm = ScriptedLLM([load, "checked"])
    system = AgenticOS(Settings(provider="mock"), llm=llm)
    system.add_skill(parse_skill_md(SKILL_MD))
    agent = system.spawn("auditor", "You audit invoices.", tools=[], skills=["invoice-checker", "missing"])
    assert set(agent.tools.names()) == {"load_skill", "calculator"}
    assert system.describe("auditor")["skills"] == ["invoice-checker"]
    result = agent.run("check invoice 7")
    assert "invoice-checker: Check invoices" in llm.calls[0]["system"]
    assert "Recompute every line total" in result.tool_calls[0]["result"]
    system.shutdown()


# --------------------------------------------------------------------- mods
class Echo:
    def __init__(self):
        self.seen = []
        seen = self.seen

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, body=None):
                seen.append({"method": self.command, "path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                data = json.dumps({"ok": True, "path": self.path}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._reply()

            def do_POST(self):
                self._reply(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_mods_call_apis_and_keep_secret_headers(tmp_path):
    echo = Echo()
    try:
        get_mod = Mod("weather", "Weather for a city", echo.url + "/w/{city}", "GET", {"Authorization": "Bearer k1"},
                      [ModParam("city"), ModParam("days", "integer", required=False)])
        tool = get_mod.to_tool()
        assert tool.name == "mod_weather" and tool.parameters["required"] == ["city"]
        assert json.loads(tool(city="Kraków", days=3))["path"] == "/w/Krak%C3%B3w?days=3"
        post = mod_from_dict({"name": "notify", "description": "Send", "url": echo.url + "/n", "method": "post",
                              "params": [{"name": "text"}]})
        post.to_tool()(text="hi")
        assert echo.seen[0]["auth"] == "Bearer k1" and echo.seen[1]["body"] == {"text": "hi"}
    finally:
        echo.close()
    path = str(tmp_path / "mods.json")
    reg = ModRegistry(path)
    reg.add(get_mod)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert get_mod.public()["headers"]["Authorization"] == "•" * 8  # short secrets are fully hidden
    edited = mod_from_dict({**get_mod.public(), "description": "Edited"})  # masked header comes back from the UI
    reg.add(edited)
    assert ModRegistry(path).all()[0].headers["Authorization"] == "Bearer k1"
    for bad in ({"name": "x y", "description": "d", "url": "https://x"}, {"name": "x", "description": "d", "url": "ftp://x"},
                {"name": "x", "description": "d", "url": "https://x", "params": [{"name": "a b"}]}):
        with pytest.raises(ValueError):
            mod_from_dict(bad)


# ---------------------------------------------------------------------- MCP
class FakeMCP:
    """Streamable-HTTP MCP server: JSON for initialize/call, SSE for tools/list, 202 for notifications."""

    def __init__(self):
        self.requests = []
        reqs = self.requests

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                reqs.append({"msg": msg, "session": self.headers.get("Mcp-Session-Id"),
                             "version": self.headers.get("MCP-Protocol-Version"), "auth": self.headers.get("Authorization")})
                if "id" not in msg:
                    self.send_response(202)
                    self.end_headers()
                    return
                method = msg["method"]
                if method == "initialize":
                    result, sse = {"protocolVersion": "2025-06-18", "serverInfo": {"name": "fake"}, "capabilities": {}}, False
                elif method == "tools/list":
                    result, sse = {"tools": [{"name": "add", "description": "Add numbers", "inputSchema": {
                        "type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}]}, True
                elif method == "tools/call":
                    args = msg["params"]["arguments"]
                    result, sse = {"content": [{"type": "text", "text": str(args["a"] + args["b"])}]}, False
                else:
                    result, sse = None, False
                reply = {"jsonrpc": "2.0", "id": msg["id"], "result": result} if result is not None else \
                    {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "nope"}}
                if sse:
                    data = (": ping\n\nevent: message\ndata: " + json.dumps(reply) + "\n\n").encode()
                    ctype = "text/event-stream"
                else:
                    data, ctype = json.dumps(reply).encode(), "application/json"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Mcp-Session-Id", "sess-42")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/mcp"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_mcp_http_server_tools_reach_agents(tmp_path):
    fake = FakeMCP()
    llm = ScriptedLLM([tool_call("mcp_fake_add", a=2, b=40), "42"])
    system = AgenticOS(Settings(provider="mock", data_dir=str(tmp_path)), llm=llm)
    try:
        client = system.add_mcp(MCPServerConfig("fake", "http", fake.url, headers={"Authorization": "Bearer t"}))
        assert client.status()["connected"] and client.status()["tools"] == ["add"]
        assert "mcp_fake_add" in system.agents["assistant"].tools  # shared registry sees MCP tools
        result = system.run("2+40 via mcp")
        assert result.result["tool_calls"][0]["result"] == "42"
        methods = [r["msg"]["method"] for r in fake.requests]
        assert methods[:3] == ["initialize", "notifications/initialized", "tools/list"]
        assert fake.requests[0]["session"] is None and fake.requests[2]["session"] == "sess-42"
        assert fake.requests[2]["version"] == "2025-06-18" and fake.requests[0]["auth"] == "Bearer t"
        assert json.loads((tmp_path / "mcp.json").read_text())[0]["name"] == "fake"
        assert system.remove_mcp("fake") and "mcp_fake_add" not in system.tools
    finally:
        system.shutdown()
        fake.close()


STDIO_SERVER = textwrap.dedent("""\
    import json, sys
    for line in sys.stdin:
        msg = json.loads(line)
        if "id" not in msg:
            continue
        if msg["method"] == "initialize":
            result = {"protocolVersion": "2025-06-18", "serverInfo": {"name": "stdio-fake"}}
        elif msg["method"] == "tools/list":
            result = {"tools": [{"name": "shout", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}]}
        else:
            result = {"content": [{"type": "text", "text": msg["params"]["arguments"]["text"].upper()}]}
        print("log line that is not json", flush=True)
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
    """)


def test_mcp_stdio_server(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(STDIO_SERVER)
    client = MCPClient(MCPServerConfig("local", "stdio", command=f"{sys.executable} {script}"), timeout=10).connect()
    try:
        assert client.server_info["name"] == "stdio-fake"
        (tool,) = client.as_tools()
        assert tool.name == "mcp_local_shout" and tool(text="hej") == "HEJ"
    finally:
        client.close()
    with pytest.raises(MCPError):
        MCPClient(MCPServerConfig("bad", "stdio", command="definitely-not-a-command-xyz")).connect()
    with pytest.raises(ValueError):
        MCPServerConfig("bad", "http", url="file:///x").validate()


# ----------------------------------------------------------- memory import
CHATGPT = [{"title": "Godot game", "mapping": {
    "a": {"message": {"author": {"role": "system"}, "content": {"parts": ["sys"]}, "create_time": 1}},
    "b": {"message": {"author": {"role": "user"}, "content": {"parts": ["I am building a Godot game in C#"]}, "create_time": 2}},
    "c": {"message": {"author": {"role": "assistant"}, "content": {"parts": ["Cool!"]}, "create_time": 3}},
    "d": {"message": None}}}]
CLAUDE = [{"name": "Diet", "chat_messages": [{"sender": "human", "text": "I am vegetarian"}, {"sender": "assistant", "text": "Noted"}]}]
TAKEOUT = [{"header": "Gemini Apps", "title": "Prompted plan a trip to Kraków", "time": "2026-01-01T00:00:00Z"}]
GENERIC = {"data": [{"title": "Grok chat", "messages": [{"sender": "USER", "content": "I love sci-fi"}, {"sender": "ASSISTANT", "content": "Nice"}]}]}


def test_memory_parsers_detect_formats():
    gpt = parse_export(json.dumps(CHATGPT).encode(), "conversations.json")
    assert gpt.source == "ChatGPT" and gpt.conversations[0].messages == [("user", "I am building a Godot game in C#"), ("assistant", "Cool!")]
    claude = parse_export(json.dumps(CLAUDE).encode())
    assert claude.source == "Claude" and claude.conversations[0].user_text() == "I am vegetarian"
    takeout = parse_export(json.dumps(TAKEOUT).encode(), "MyActivity.json")
    assert takeout.source.startswith("Gemini") and takeout.conversations[0].user_text() == "plan a trip to Kraków"
    generic = parse_export(json.dumps(GENERIC).encode(), "grok-export.json")
    assert generic.source == "Grok" and generic.conversations[0].title == "Grok chat"
    assert parse_export(json.dumps(["Likes tea", "Lives in Poland"]).encode()).facts == ["Likes tea", "Lives in Poland"]
    assert parse_export(json.dumps([{"memory": "Prefers short answers"}]).encode()).facts == ["Prefers short answers"]
    assert parse_text_memories("# Memories\n- Name is Daniel\n2. Uses Godot\n\n• Likes Polish UI\nok") == \
        ["Name is Daniel", "Uses Godot", "Likes Polish UI"]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("conversations.json", json.dumps(CHATGPT))
        zf.writestr("user.json", json.dumps({"email": "x@example.com"}))
        zf.writestr("memories.txt", "- Works on Agentic OS")
    merged = parse_export(buf.getvalue(), "export.zip")
    assert len(merged.conversations) == 1 and merged.facts == ["Works on Agentic OS"]
    with pytest.raises(ValueError):
        parse_export(b"PK\x03\x04garbage", "broken.zip")


def test_kernel_import_with_distillation(tmp_path):
    llm = ScriptedLLM(['{"facts": ["User builds a Godot game in C#"]}'])
    system = AgenticOS(Settings(provider="mock", data_dir=str(tmp_path)), llm=llm)
    report = system.import_memory(json.dumps(CHATGPT).encode(), "conversations.json", distill=True)
    assert report == {"source": "ChatGPT", "conversations_found": 1, "facts_added": 0, "episodes_added": 1,
                      "distilled_facts": 1, "memories_total": 2}
    facts = system.memory.recall("Godot game C#", kind="fact")
    assert facts and facts[0].metadata["imported"] is True
    system.shutdown()
    assert len(AgenticOS(Settings(provider="mock", data_dir=str(tmp_path))).memory) == 2  # persisted


# ------------------------------------------------------------------- server
def _call(base, method, path, body=None, raw=None, ctype="application/json"):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + path, method=method, data=data, headers={"Content-Type": ctype, "Authorization": "Bearer pw"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_extension_routes(tmp_path):
    system = AgenticOS(Settings(provider="mock", api_key="pw", data_dir=str(tmp_path)))
    httpd = make_server(system, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    fake = FakeMCP()
    try:
        assert _call(base, "POST", "/skills", {"markdown": SKILL_MD})[0] == 201
        assert _call(base, "POST", "/skills", {"name": "tone", "description": "Polite tone", "instructions": "Be polite.", "tools": "calculator"})[1]["tools"] == ["calculator"]
        assert {s["name"] for s in _call(base, "GET", "/skills")[1]["skills"]} == {"invoice-checker", "tone"}
        assert _call(base, "POST", "/agents", {"name": "acct", "tools": [], "skills": ["tone"]})[1]["skills"] == ["tone"]

        status, mod = _call(base, "POST", "/mods", {"name": "ping", "description": "Ping", "url": "https://example.com/ping",
                                                    "headers": {"X-Key": "secret-value-1"}})
        assert status == 201 and mod["tool"] == "mod_ping" and "secret" not in json.dumps(mod) and "mod_ping" in system.tools
        assert "secret" not in json.dumps(_call(base, "GET", "/mods")[1])

        status, server = _call(base, "POST", "/mcp", {"name": "fake", "transport": "http", "url": fake.url, "token": "tok"})
        assert status == 201 and server["connected"] and "tok" not in json.dumps(server)
        assert _call(base, "POST", "/mcp/fake/reconnect", {})[1]["tools"] == ["add"]
        assert _call(base, "POST", "/mcp", {"name": "dead", "transport": "http", "url": "http://127.0.0.1:9/mcp"})[0] == 400

        status, rep = _call(base, "POST", "/memory/import?filename=conversations.json", raw=json.dumps(CLAUDE).encode(),
                            ctype="application/octet-stream")
        assert status == 200 and rep["source"] == "Claude" and rep["episodes_added"] == 1
        status, rep = _call(base, "POST", "/memory/import", {"text": "- Lubi krótkie odpowiedzi\n- Mieszka w Polsce"})
        assert status == 200 and rep["facts_added"] == 2
        assert _call(base, "POST", "/memory/import", raw=b"  ", ctype="application/octet-stream")[0] == 400

        for path in ("/skills/tone", "/mods/ping", "/mcp/fake"):
            assert _call(base, "DELETE", path)[0] == 200
        assert _call(base, "DELETE", "/skills/tone")[0] == 404
    finally:
        fake.close()
        httpd.shutdown()
        httpd.server_close()
        system.shutdown()
