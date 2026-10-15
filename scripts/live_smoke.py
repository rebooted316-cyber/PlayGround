"""Live smoke test against a real model.

Exercises each major layer once with the real model and reports PASS/FAIL per
check. Provider is chosen with AGENTIC_OS_PROVIDER (anthropic | gemini):

* anthropic: ANTHROPIC_API_KEY, or an Anthropic-compatible gateway such as
  OpenRouter (ANTHROPIC_BASE_URL=https://openrouter.ai/api, ANTHROPIC_AUTH_TOKEN,
  AGENTIC_OS_MODEL=anthropic/claude-opus-5.5 or openrouter/free,
  AGENTIC_OS_FALLBACKS=false, AGENTIC_OS_EFFORT= to omit effort).
* gemini: GEMINI_API_KEY, AGENTIC_OS_MODEL=gemini-flash-latest (default) or
  another model served by the Gemini API.

Run: python scripts/live_smoke.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback

from agentic_os import Agent, AgenticOS, ConversationBuffer, LLMProvider, RAGPipeline, Settings
from agentic_os.agents import Crew, GoalAgent
from agentic_os.learning import LLMJudge
from agentic_os.llm import create_llm
from agentic_os.tools import calculator

PROVIDER = os.environ.get("AGENTIC_OS_PROVIDER", "anthropic")
MODEL = os.environ.get("AGENTIC_OS_MODEL", "claude-opus-5-5" if PROVIDER == "anthropic" else "gemini-flash-latest")
EFFORT = os.environ.get("AGENTIC_OS_EFFORT", "low")
SETTINGS = Settings(provider=PROVIDER, model=MODEL, effort=EFFORT)

SERVED: set[str] = set()


class RecordingProvider(LLMProvider):
    """Wraps a provider and remembers which models actually served requests (routers vary)."""

    def __init__(self, inner: LLMProvider) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = getattr(inner, "model", None)

    def complete(self, *args, **kwargs):
        response = self.inner.complete(*args, **kwargs)
        if response.model:
            SERVED.add(response.model)
        return response


llm = RecordingProvider(create_llm(SETTINGS))
results: list[tuple[str, bool, str]] = []
usage = {"input_tokens": 0, "output_tokens": 0}


def track(result) -> None:
    for key in usage:
        usage[key] += result.usage.get(key, 0)


def check(name: str):
    def wrap(fn):
        start = time.perf_counter()
        try:
            detail = fn() or ""
            results.append((name, True, detail))
            status = "PASS"
        except Exception as exc:  # noqa: BLE001 - report every failure
            detail = f"{type(exc).__name__}: {exc}"
            results.append((name, False, detail))
            traceback.print_exc()
            status = "FAIL"
        print(f"[{status}] {name} ({time.perf_counter() - start:.1f}s) {detail[:300]}", flush=True)
        return fn

    return wrap


@check("tool use: calculator loop")
def _():
    res = Agent("math", llm, tools=[calculator]).run("What is 1234 * 5678? Use the calculator.")
    track(res)
    assert res.stopped_reason == "completed", res.stopped_reason
    assert any(c["tool"] == "calculator" and not c["is_error"] for c in res.tool_calls), res.tool_calls
    assert "7006652" in res.output.replace(",", "").replace(" ", ""), res.output
    return f"steps={res.steps} output={res.output!r}"


@check("RAG: grounded answer with citation")
def _():
    rag = RAGPipeline()
    rag.add_document("The Zephyr-9 drone has a flight time of 47 minutes and weighs 1.2 kg.", source="zephyr.md")
    rag.add_document("The Corvid-3 drone has a flight time of 22 minutes.", source="corvid.md")
    res = Agent("rag", llm, knowledge=rag).run("What is the flight time of the Zephyr-9 drone?")
    track(res)
    assert "47" in res.output, res.output
    assert re.search(r"\[\d\]", res.output), f"no citation in: {res.output}"
    return res.output


@check("short-term memory: multi-turn")
def _():
    agent = Agent("chat", llm, memory=ConversationBuffer())
    track(agent.run("Hi! My favourite colour is teal. Just acknowledge briefly."))
    res = agent.run("What is my favourite colour? Answer with one word.")
    track(res)
    assert "teal" in res.output.lower(), res.output
    return res.output


@check("planning: goal decomposition + execution")
def _():
    worker = Agent("worker", llm, tools=[calculator], max_steps=6)
    result = GoalAgent(llm, worker).run(
        "Compute 15% of 2,480, then compute that result multiplied by 3, then state both numbers in one sentence."
    )
    for r in result.task_results.values():
        track(r)
    assert len(result.plan.tasks) >= 2, result.plan.to_dict()
    assert result.succeeded, result.plan.to_dict()
    assert "1116" in result.output.replace(",", ""), result.output
    return f"tasks={len(result.plan.tasks)} output={result.output[:200]!r}"


@check("multi-agent: manager delegates to specialist")
def _():
    writer = Agent("writer", llm, instructions="You write very short, punchy product taglines.", max_steps=3)
    crew = Crew([writer], llm=llm, process="manager")
    res = crew.run("Get a tagline for a solar-powered backpack from the writer, then give it to me.")
    assert any(t["agent"] == "writer" for t in res.transcript), res.transcript
    return res.output[:200]


@check("self-improvement: LLM judge scores")
def _():
    judge = LLMJudge(llm, rubric="Factually correct and directly answers the question.")
    good = judge("What is the capital of France?", "Paris.")
    bad = judge("What is the capital of France?", "Berlin.")
    assert 0.0 <= bad < good <= 1.0, (good, bad, judge.last_reason)
    return f"good={good} bad={bad}"


@check("kernel: AgenticOS process end-to-end")
def _():
    with AgenticOS(SETTINGS, llm=llm) as system:
        system.ingest("Agentic OS assigns every agent run a process id (PID).", source="os.md")
        proc = system.run("According to the knowledge base, what does Agentic OS assign every agent run?")
        assert proc.status == "completed", proc.error
        assert "pid" in proc.result["output"].lower() or "process" in proc.result["output"].lower(), proc.result
        return proc.result["output"][:200]


passed = sum(ok for _, ok, _ in results)
print("\n" + json.dumps({"provider": PROVIDER, "model": MODEL, "effort": EFFORT, "fallbacks": SETTINGS.fallbacks, "served_by": sorted(SERVED), "passed": passed, "total": len(results), "usage_tracked": usage}))
sys.exit(0 if passed == len(results) else 1)
