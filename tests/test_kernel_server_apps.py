import json
import threading
import urllib.error
import urllib.request

import pytest

from agentic_os import AgenticOS, Settings
from agentic_os.apps import data_report, research_assistant, support_triage
from agentic_os.kernel import AgenticOSError
from agentic_os.llm import HeuristicLLM
from agentic_os.server import make_server


@pytest.fixture
def system():
    s = AgenticOS(Settings(provider="mock"))
    yield s
    s.shutdown()


def test_kernel_run_submit_and_status(system):
    proc = system.run("What is 21 * 2?")
    assert proc.status == "completed" and "42" in proc.result["output"]
    queued = system.submit("What is 5 + 5?")
    assert system.wait(queued.pid, timeout=10).status == "completed"
    status = system.status()
    assert status["processes"]["completed"] == 2 and "calculator" in status["tools"]
    assert [p["pid"] for p in system.ps()] == [proc.pid, queued.pid]


def test_kernel_spawn_tools_ingest_and_feedback(system):
    @system.tool
    def shout(text: str) -> str:
        """Uppercase text."""
        return text.upper()

    assert "shout" in system.agent("assistant").tools  # shared registry sees new tools
    analyst = system.spawn("analyst", tool_tags=["math"])
    assert analyst.tools.names() == ["calculator"]
    assert system.ingest("Agentic OS ships with hybrid retrieval.", source="notes") == 1
    proc = system.run("hybrid retrieval?")
    system.give_feedback(proc.pid, 1.0, "great")
    assert len(system.feedback) == 1 and system.memory.recall("hybrid retrieval", kind="lesson")
    with pytest.raises(AgenticOSError):
        system.agent("ghost")


def test_failed_process_is_recorded():
    class Broken(HeuristicLLM):
        def complete(self, *a, **k):
            raise RuntimeError("model down")

    s = AgenticOS(Settings(provider="mock"), llm=Broken())
    proc = s.run("hi")
    assert proc.status == "failed" and "model down" in proc.error
    s.shutdown()


def _request(base, method, path, body=None, token=None):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


@pytest.fixture
def api():
    s = AgenticOS(Settings(provider="mock", api_key="secret"))
    httpd = make_server(s, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    s.shutdown()


def test_http_api_end_to_end(api):
    assert _request(api, "GET", "/health") == (200, {"status": "ok"})
    assert _request(api, "GET", "/status")[0] == 401
    tok = "secret"
    status, proc = _request(api, "POST", "/run", {"task": "What is 3 * 3?"}, tok)
    assert status == 200 and "9" in proc["result"]["output"]
    status, _ = _request(api, "POST", "/run", {"task": "2+2?", "async": True}, tok)
    assert status == 202
    assert _request(api, "GET", f"/processes/{proc['pid']}", token=tok)[1]["status"] == "completed"
    assert _request(api, "POST", f"/processes/{proc['pid']}/feedback", {"reward": 1}, tok) == (200, {"ok": True})
    assert _request(api, "POST", f"/processes/{proc['pid']}/feedback", {"reward": 9}, tok)[0] == 400
    assert _request(api, "POST", "/knowledge", {"text": "The sky is blue.", "metadata": {"source": "sky"}}, tok)[0] == 201
    assert _request(api, "POST", "/agents", {"name": "calc", "tool_tags": ["math"]}, tok)[1]["tools"] == ["calculator"]
    assert _request(api, "POST", "/agents", {"name": "bad name"}, tok)[0] == 400
    assert _request(api, "POST", "/run", {}, tok)[0] == 400
    assert _request(api, "POST", "/run", {"task": "x", "agent": "ghost"}, tok)[0] == 404
    trace = _request(api, "GET", f"/traces/{proc['result']['trace_id']}", token=tok)[1]["spans"]
    assert {s["name"] for s in trace} >= {"agent.run", "llm.complete", "tool.call"}
    goal = _request(api, "POST", "/goal", {"goal": "add 2+2 and then report it"}, tok)[1]
    assert len(goal["plan"]["tasks"]) == 2
    metrics = _request(api, "GET", "/metrics", token=tok)[1]
    assert metrics["counters"]["agent.runs"] >= 2
    assert _request(api, "GET", "/nope", token=tok)[0] == 404


def test_apps_run_offline(system):
    system.ingest("Hybrid search combines BM25 and vectors.", source="rag.md")
    res = research_assistant(system, "What does hybrid search combine?")
    assert [t["agent"] for t in res.transcript] == ["researcher", "writer"]
    assert "rag.md" in res.transcript[0]["output"]
    assert data_report(system, "a,b\n1,2\n3,4", "What is 3 + 4?")
    triage = support_triage(system)
    assert "Settings > Security" in triage.run("I forgot my password, how do I reset it?")
    assert triage.run("My parcel arrived damaged")  # falls through to the LLM agent
