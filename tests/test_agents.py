import json

from conftest import tool_call

from agentic_os.agents import Agent, Crew, GoalAgent, ReflexAgent, Rule
from agentic_os.learning import IntentRouter
from agentic_os.llm import LLMResponse, ScriptedLLM, ToolCall
from agentic_os.memory import ConversationBuffer, LongTermMemory
from agentic_os.rag import RAGPipeline
from agentic_os.tools import calculator


def test_react_loop_calls_tool_then_answers():
    llm = ScriptedLLM([tool_call("calculator", expression="6*7"), "The answer is 42."])
    agent = Agent("math", llm, tools=[calculator])
    result = agent.run("What is 6*7?")
    assert result.output == "The answer is 42."
    assert result.stopped_reason == "completed" and result.steps == 2
    assert result.tool_calls[0]["result"] == "6*7 = 42"
    second_call_messages = llm.calls[1]["messages"]
    assert second_call_messages[-1].role == "tool" and second_call_messages[-1].content == "6*7 = 42"


def test_parallel_tool_calls_are_all_executed():
    both = LLMResponse(
        tool_calls=[ToolCall("a", "calculator", {"expression": "1+1"}), ToolCall("b", "calculator", {"expression": "2+2"})],
        stop_reason="tool_use",
    )
    agent = Agent("m", ScriptedLLM([both, "done"]), tools=[calculator])
    result = agent.run("two sums")
    assert [c["result"] for c in result.tool_calls] == ["1+1 = 2", "2+2 = 4"]


def test_unknown_tool_and_bad_args_are_reported_to_model():
    llm = ScriptedLLM([tool_call("nope"), tool_call("calculator", "c2"), "ok"])
    result = Agent("m", llm, tools=[calculator]).run("x")
    assert result.tool_calls[0]["is_error"] and "Unknown tool" in result.tool_calls[0]["result"]
    assert result.tool_calls[1]["is_error"] and "Missing required" in result.tool_calls[1]["result"]


def test_max_steps_refusal_and_truncation():
    looping = Agent("m", ScriptedLLM([tool_call("calculator", expression="1+1")] * 3), tools=[calculator], max_steps=3)
    assert looping.run("loop").stopped_reason == "max_steps"

    refusal = Agent("m", ScriptedLLM([LLMResponse(stop_reason="refusal")])).run("bad")
    assert refusal.stopped_reason == "refusal" and refusal.output

    truncated = LLMResponse(text="partial", tool_calls=[ToolCall("t", "calculator", {})], stop_reason="max_tokens")
    res = Agent("m", ScriptedLLM([truncated]), tools=[calculator]).run("x")
    assert res.stopped_reason == "max_tokens" and res.tool_calls == []
    assert res.messages[-1].role == "tool" and res.messages[-1].is_error  # history stays valid


def test_approval_hook_blocks_tools():
    llm = ScriptedLLM([tool_call("calculator", expression="1+1"), "blocked"])
    result = Agent("m", llm, tools=[calculator], approve_tool=lambda call: False).run("x")
    assert result.tool_calls[0]["is_error"] and "not approved" in result.tool_calls[0]["result"]


def test_short_term_memory_carries_conversation():
    llm = ScriptedLLM(["Nice to meet you, Ada.", lambda msgs, s, t: f"history={len(msgs)}"])
    agent = Agent("chat", llm, memory=ConversationBuffer())
    agent.run("My name is Ada.")
    assert agent.run("What is my name?").output == "history=3"


def test_knowledge_and_memories_are_injected_with_citations():
    rag = RAGPipeline()
    rag.add_document("Agentic OS was created to teach agentic AI.", source="about.md")
    mem = LongTermMemory()
    mem.remember("The user is learning agentic AI", kind="fact")
    llm = ScriptedLLM(["It teaches agentic AI [1]."])
    result = Agent("a", llm, knowledge=rag, long_term=mem).run("Why was Agentic OS created to teach agentic AI?")
    prompt = llm.calls[0]["messages"][-1].content
    assert "<knowledge>" in prompt and "about.md" in prompt and "<memories>" in prompt
    assert result.citations[0]["source"] == "about.md"
    assert any(m.kind == "episode" for m in mem.recall("Why was Agentic OS created", k=5))


def test_agent_as_tool_enables_delegation():
    worker = Agent("worker", ScriptedLLM(["worker result"]))
    boss = Agent("boss", ScriptedLLM([tool_call("ask_worker", task="do it"), "boss used: worker result"]), tools=[worker.as_tool()])
    assert boss.run("delegate").tool_calls[0]["result"] == "worker result"


def test_goal_agent_executes_plan_in_order_with_context():
    plan = {"goal": "g", "tasks": [{"id": "t1", "description": "find"}, {"id": "t2", "description": "write", "depends_on": ["t1"]}]}
    planner_llm = ScriptedLLM([json.dumps(plan), "FINAL"])
    worker_llm = ScriptedLLM(["facts", lambda msgs, s, t: msgs[-1].content])
    result = GoalAgent(planner_llm, Agent("w", worker_llm)).run("g")
    assert result.output == "FINAL" and result.succeeded
    assert "[t1] find\nfacts" in result.plan.get("t2").result


def test_goal_agent_skips_tasks_after_failure():
    plan = {"tasks": [{"id": "t1", "description": "a"}, {"id": "t2", "description": "b", "depends_on": ["t1"]}]}
    worker = Agent("w", ScriptedLLM([LLMResponse(stop_reason="refusal")]))
    result = GoalAgent(ScriptedLLM([json.dumps(plan), "summary"]), worker).run("g")
    assert [t.status for t in result.plan.tasks] == ["failed", "skipped"]


def test_crew_sequential_and_manager():
    a = Agent("researcher", ScriptedLLM(["notes"]), instructions="Research things.")
    b = Agent("writer", ScriptedLLM([lambda msgs, s, t: "article from " + msgs[-1].content.split("\n")[2]]), instructions="Write.")
    seq = Crew([a, b], process="sequential").run("topic")
    assert seq.output == "article from notes" and len(seq.transcript) == 2

    w = Agent("writer", ScriptedLLM(["draft"]), instructions="Write.")
    manager_llm = ScriptedLLM([tool_call("ask_writer", task="write it"), "final: draft"])
    res = Crew([w], llm=manager_llm, process="manager").run("make an article")
    assert res.output == "final: draft"
    assert res.transcript[0] == {"agent": "writer", "input": "write it", "output": "draft"}


def test_crew_router_with_trained_classifier():
    router = IntentRouter({
        "math": ["calculate 5 plus 7", "what is 12 times 9", "compute the square root of 81", "divide 100 by 4"],
        "writer": ["write a poem about the sea", "draft an email to my boss", "compose a short story", "write a blog post"],
    })
    agents = {"math": Agent("math", ScriptedLLM(["MATH"]), instructions="m"), "writer": Agent("writer", ScriptedLLM(["WRITER"]), instructions="w")}
    assert Crew(agents, process="router", router=router).run("compute 17 times 3").output == "MATH"


def test_crew_router_falls_back_to_llm():
    agents = {"math": Agent("math", ScriptedLLM(["MATH"]), instructions="m"), "writer": Agent("writer", ScriptedLLM(["WRITER"]), instructions="w")}
    assert Crew(agents, llm=ScriptedLLM(["writer"]), process="router").route("poem please") == "writer"


def test_reflex_agent():
    agent = ReflexAgent([Rule.regex("greet", r"\bhello\b", lambda t: "hi!")], default=lambda t: "?")
    assert agent.run("Hello there") == "hi!" and agent.last_rule == "greet"
    assert agent.run("bye") == "?" and agent.last_rule is None
