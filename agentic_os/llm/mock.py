"""Offline providers: deterministic test doubles and a heuristic demo model.

They let the whole OS run (and be tested) with no network or API key.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Callable

from agentic_os.llm.base import LLMError, LLMProvider, LLMResponse, Message, ToolCall

ScriptItem = str | LLMResponse | Callable[..., str | LLMResponse]


class ScriptedLLM(LLMProvider):
    """Returns pre-defined responses in order. Records every call.

    Each script item is a string (final text), an :class:`LLMResponse`, or a
    callable ``(messages, system, tools) -> str | LLMResponse``.
    """

    name = "scripted"

    def __init__(self, script: list[ScriptItem]) -> None:
        self.script = list(script)
        self.calls: list[dict] = []

    def complete(self, messages, *, system=None, tools=None, max_tokens=16000) -> LLMResponse:
        self.calls.append({"messages": list(messages), "system": system, "tools": list(tools or [])})
        if not self.script:
            raise LLMError("ScriptedLLM ran out of responses")
        item = self.script.pop(0)
        if callable(item) and not isinstance(item, LLMResponse):
            item = item(messages, system, tools)
        if isinstance(item, str):
            item = LLMResponse(text=item)
        return item


_MATH = re.compile(r"[-+*/().\d\s^%]*\d[-+*/().\d\s^%]*[-+*/^%][-+*/().\d\s^%]*\d[-+*/().\d\s^%]*")
_ids = itertools.count(1)


class HeuristicLLM(LLMProvider):
    """A tiny rule-based stand-in for a real model, for offline demos.

    It picks a tool by simple heuristics (math -> calculator, otherwise
    knowledge search), summarizes tool results, and emits JSON plans when
    asked by the planner. It is not intelligent; it shows the plumbing.
    """

    name = "heuristic"

    def complete(self, messages, *, system=None, tools=None, max_tokens=16000) -> LLMResponse:
        tool_names = {t.name for t in (tools or [])}
        last = messages[-1] if messages else Message.user("")

        if system and "JSON plan" in system:
            return LLMResponse(text=json.dumps(self._plan(last.content)))

        if last.role == "tool":
            results = []
            for msg in reversed(messages):
                if msg.role != "tool":
                    break
                results.append(msg.content)
            body = "\n".join(reversed(results))
            return LLMResponse(text=f"Based on the tool results:\n{body}")

        text = last.content
        task = re.search(r"<task>\s*(.*?)\s*</task>", text, re.DOTALL)
        knowledge = re.search(r"<knowledge>\s*(.*?)\s*</knowledge>", text, re.DOTALL)
        if task:
            text = task.group(1)
        match = _MATH.search(text)
        if knowledge and not match:
            return LLMResponse(text=f"From the knowledge base:\n{knowledge.group(1)}")
        if "calculator" in tool_names and match:
            return self._call("calculator", {"expression": match.group(0).strip()})
        if "search_knowledge" in tool_names:
            return self._call("search_knowledge", {"query": text})
        return LLMResponse(text=f"[offline heuristic model] {text.strip()}")

    @staticmethod
    def _call(name: str, args: dict) -> LLMResponse:
        call = ToolCall(id=f"call_{next(_ids)}", name=name, arguments=args)
        return LLMResponse(tool_calls=[call], stop_reason="tool_use")

    @staticmethod
    def _plan(goal: str) -> dict:
        goal_line = goal.splitlines()[0] if goal else ""
        goal_line = goal_line.removeprefix("Goal:").strip()
        parts = [p.strip(" .") for p in re.split(r"\bthen\b|\band\b|;|\.\s", goal_line) if p.strip(" .")]
        parts = parts[:5] or [goal_line]
        tasks = []
        for i, part in enumerate(parts, 1):
            tasks.append(
                {"id": f"t{i}", "description": part, "depends_on": [f"t{i-1}"] if i > 1 else []}
            )
        return {"goal": goal_line, "tasks": tasks}
