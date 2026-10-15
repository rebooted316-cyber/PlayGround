"""War Room access to the core concepts: thinking modes, delegation and tool approval."""

import json
import threading
import time
import urllib.request

import pytest

from agentic_os import AgenticOS, Settings
from agentic_os.llm import LLMResponse, ScriptedLLM
from agentic_os.llm.base import ToolCall
from agentic_os.server import make_server


def call(name, **arguments):
    return LLMResponse(text="", tool_calls=[ToolCall(id=f"c-{name}", name=name, arguments=arguments)], stop_reason="tool_use")


def make_os(tmp_path, script):
    return AgenticOS(Settings(provider="mock", data_dir=str(tmp_path)), llm=ScriptedLLM(script))


def test_reflective_agent_retries_and_keeps_best_attempt(tmp_path):
    system = make_os(tmp_path, [
        "Paris is in Germany.",                               # attempt 1
        '{"score": 2, "reason": "wrong country"}',            # judge
        "- Check the country.",                               # critique
        "Paris is the capital of France.",                    # attempt 2
        '{"score": 9, "reason": "correct"}',                  # judge
    ])
    system.spawn("critic", mode="reflective", conversational=True)
    proc = system.chat("Where is Paris?", agent="critic")
    assert proc.status == "completed", proc.error
    assert proc.result["output"] == "Paris is the capital of France."
    assert proc.result["reflection"]["best_score"] == 0.9
    assert len(proc.result["reflection"]["attempts"]) == 2
    assert any(r.metadata.get("kind") == "lesson" for r in system.memory.store.all())
    buffer = next(iter(system._sessions.values()))
    assert len([m for m in buffer.messages() if m.role == "user"]) == 1  # only the best attempt is kept


def test_adaptive_agent_learns_from_feedback_and_persists(tmp_path):
    system = make_os(tmp_path, ["one", "two"])
    system.spawn("learner", mode="adaptive")
    proc = system.run("Say something", agent="learner")
    assert proc.strategy in system.strategies and proc.result["strategy"] == proc.strategy
    system.give_feedback(proc.pid, 1.0)
    system.give_feedback(proc.pid, -1.0)  # changing the rating replaces it instead of counting twice
    stats = system._bandits["learner"].stats[proc.strategy]
    assert (stats.pulls, stats.total_reward) == (1, -1.0)
    learning = system.describe("learner")["learning"]
    assert learning["strategies"][proc.strategy]["pulls"] == 1

    reloaded = make_os(tmp_path, [])
    assert reloaded.describe("learner")["mode"] == "adaptive"
    assert reloaded._bandits["learner"].stats[proc.strategy].pulls == 1


def test_agent_delegates_to_another_agent(tmp_path):
    system = make_os(tmp_path, [
        call("ask_researcher", task="Find the answer"),       # boss delegates
        "42",                                                 # researcher answers
        "The researcher says 42.",                            # boss answers
    ])
    system.spawn("researcher", tools=[])
    system.spawn("boss", delegates=["researcher", "boss"])
    assert "ask_researcher" in system.agent("boss").tools
    assert "ask_boss" not in system.agent("boss").tools  # an agent cannot delegate to itself
    proc = system.run("What is the answer?", agent="boss")
    assert proc.result["output"] == "The researcher says 42."
    assert proc.result["tool_calls"][0]["result"] == "42"


def test_delegation_depth_is_limited(tmp_path):
    system = make_os(tmp_path, [call("ask_b", task="x"), call("ask_a", task="x"), call("ask_b", task="x"),
                                call("ask_a", task="x"), "b3", "a2", "b1", "a0"])
    system.spawn("a", delegates=["b"])
    system.spawn("b", delegates=["a"])
    proc = system.run("loop", agent="a")
    assert proc.status == "completed"
    assert proc.result["output"] == "a0"
    blocked = system.llm.calls[4]["messages"][-1]  # b at depth 3 is told to stop delegating
    assert "Delegation limit reached" in blocked.content


def _approval_os(tmp_path):
    system = make_os(tmp_path, [call("calculator", expression="2+2"), "The result is 4."])
    system.spawn("careful", permission="ask")
    return system


def _wait_for_approval(system):
    for _ in range(200):
        if system.pending_approvals():
            return system.pending_approvals()[0]
        time.sleep(0.01)
    raise AssertionError("no approval request appeared")


@pytest.mark.parametrize("approve", [True, False])
def test_tool_calls_wait_for_operator(tmp_path, approve):
    system = _approval_os(tmp_path)
    proc = system.submit("What is 2+2?", agent="careful")
    pending = _wait_for_approval(system)
    assert pending["tool"] == "calculator" and pending["pid"] == proc.pid
    assert system.process(proc.pid).status == "waiting"
    system.decide_approval(pending["id"], approve)
    system.wait(proc.pid, timeout=5)
    result = system.process(proc.pid).result["tool_calls"][0]
    assert result["is_error"] is (not approve)
    assert ("4" in result["result"]) is approve
    assert system.pending_approvals() == []


def test_http_api_for_modes_and_approvals(tmp_path):
    system = _approval_os(tmp_path)
    server = make_server(system, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def req(method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    try:
        status, agent = req("POST", "/agents", {"name": "smart", "mode": "reflective", "permission": "smart", "delegates": ["careful"]})
        assert status == 201 and agent["mode"] == "reflective" and agent["permission"] == "smart"
        assert agent["delegates"] == ["careful"] and "ask_user" in agent["tools"]
        assert req("POST", "/agents", {"name": "legacy", "approval": True})[1]["permission"] == "ask"
        assert req("POST", "/agents", {"name": "bad2", "permission": "root"})[0] == 400
        status, changed = req("POST", "/agents/smart/settings", {"permission": "full", "mode": "standard"})
        assert status == 200 and (changed["permission"], changed["mode"]) == ("full", "standard")
        assert req("POST", "/agents/smart/settings", {"permission": "root"})[0] == 400
        assert req("POST", "/agents/ghost/settings", {"permission": "ask"})[0] == 404
        assert req("POST", "/agents", {"name": "bad", "mode": "psychic"})[0] == 400
        status, proc = req("POST", "/chat", {"message": "2+2?", "agent": "careful"})
        pending = _wait_for_approval(system)
        assert req("GET", "/approvals")[1]["approvals"][0]["id"] == pending["id"]
        assert req("POST", f"/approvals/{pending['id']}", {"approve": "yes"})[0] == 400
        assert req("POST", f"/approvals/{pending['id']}", {"approve": True})[0] == 200
        system.wait(proc["pid"], timeout=5)
        assert req("POST", "/approvals/nope", {"approve": True})[0] == 404
    finally:
        server.shutdown()
        system.shutdown()


def test_smart_permission_runs_read_only_tools_and_asks_for_changes(tmp_path):
    system = make_os(tmp_path, [call("calculator", expression="2+2"), call("remember_fact", fact="likes tea"), "Done."])
    system.spawn("smart", permission="smart")
    proc = system.submit("Compute and remember", agent="smart")
    pending = _wait_for_approval(system)
    assert pending["tool"] == "remember_fact"  # the calculator ran without asking
    system.decide_approval(pending["id"], True)
    system.wait(proc.pid, timeout=5)
    calls = system.process(proc.pid).result["tool_calls"]
    assert [c["is_error"] for c in calls] == [False, False]


@pytest.mark.parametrize("answer, expected", [("Blue", "The user answered: Blue"), ("", "did not answer")])
def test_agent_can_ask_the_user_a_question(tmp_path, answer, expected):
    system = make_os(tmp_path, [call("ask_user", question="Which colour?"), "OK"])
    system.spawn("curious", permission="smart")
    proc = system.submit("Paint it", agent="curious")
    pending = _wait_for_approval(system)
    assert pending["kind"] == "question" and pending["arguments"]["question"] == "Which colour?"
    system.decide_approval(pending["id"], answer=answer)
    system.wait(proc.pid, timeout=5)
    assert expected in system.process(proc.pid).result["tool_calls"][0]["result"]


def test_autonomous_agent_never_waits_and_has_no_ask_user(tmp_path):
    system = make_os(tmp_path, [call("remember_fact", fact="x"), "Done."])
    agent = system.spawn("auto")
    assert agent.permission == "auto" and "ask_user" not in agent.tools
    assert system.run("go", agent="auto").result["tool_calls"][0]["is_error"] is False


def test_full_access_sees_every_tool_and_agent(tmp_path):
    system = make_os(tmp_path, [])
    system.spawn("helper", tools=[])
    root = system.spawn("root", tools=["calculator"], permission="full")
    system.spawn("late", tools=[])  # created afterwards: still reachable
    names = root.tools.names()
    assert {"calculator", "remember_fact", "search_knowledge", "ask_helper", "ask_late", "ask_assistant"} <= set(names)
    assert "ask_root" not in names and "ask_user" not in names
    assert root.max_steps >= 25


def test_always_allow_is_remembered_and_can_be_reset(tmp_path):
    system = make_os(tmp_path, [call("calculator", expression="1+1"), "two", call("calculator", expression="1+1"), "two"])
    system.spawn("careful", permission="ask")
    proc = system.submit("1+1", agent="careful")
    system.decide_approval(_wait_for_approval(system)["id"], True, remember=True)
    system.wait(proc.pid, timeout=5)
    assert system.run("again", agent="careful").result["tool_calls"][0]["is_error"] is False  # no prompt
    assert make_os(tmp_path, []).describe("careful")["always_allow"] == ["calculator"]
    system.reset_always_allow("careful")
    assert make_os(tmp_path, []).describe("careful")["always_allow"] == []


def test_old_approval_flag_in_saved_specs_maps_to_ask(tmp_path):
    (tmp_path / "agents.json").write_text(json.dumps([{"type": "native", "name": "old", "approval": True}]))
    assert make_os(tmp_path, []).describe("old")["permission"] == "ask"


def test_read_only_classification():
    from agentic_os.mods import mod_from_dict
    assert AgenticOS.is_read_only(mod_from_dict({"name": "w", "url": "https://x.test", "method": "GET"}).to_tool())
    assert not AgenticOS.is_read_only(mod_from_dict({"name": "w", "url": "https://x.test", "method": "POST"}).to_tool())


def test_permission_and_mode_can_change_at_any_time(tmp_path):
    system = make_os(tmp_path, [call("remember_fact", fact="x"), "Done.", call("remember_fact", fact="y"), "Done."])
    system.spawn("flex", conversational=True)
    first = system.chat("hello", agent="flex", wait=False)  # creates a session that must survive the change
    system.wait(first.pid, timeout=5)
    flex = system.update_agent("flex", permission="ask", mode="adaptive")
    assert (flex.permission, flex.mode) == ("ask", "adaptive") and "ask_user" in flex.tools
    assert ("flex", next(iter(system._sessions))[1]) in system._sessions  # chat history kept
    reloaded = make_os(tmp_path, [])
    assert reloaded.describe("flex")["permission"] == "ask"  # persisted

    flex = system.update_agent("flex", permission="full")
    assert "ask_user" not in flex.tools and "ask_assistant" in flex.tools and flex.mode == "adaptive"
    with pytest.raises(ValueError):
        system.update_agent("flex", permission="root")


def test_switching_to_full_releases_waiting_calls_and_running_tasks_follow_live(tmp_path):
    system = make_os(tmp_path, [call("calculator", expression="1+1"), call("calculator", expression="2+2"), "Done."])
    system.spawn("careful", permission="ask")
    proc = system.submit("add twice", agent="careful")
    _wait_for_approval(system)
    system.update_agent("careful", permission="full")  # the waiting call is let through ...
    system.wait(proc.pid, timeout=5)
    calls = system.process(proc.pid).result["tool_calls"]
    assert [c["is_error"] for c in calls] == [False, False]  # ... and the next one no longer asks
    assert system.pending_approvals() == []
