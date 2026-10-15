"""Decision-making & planning (roadmap stage 7).

A :class:`Planner` decomposes a goal into a :class:`Plan`: a DAG of tasks
with dependencies. :meth:`Plan.execution_order` gives a valid order
(topological sort), and :class:`~agentic_os.agents.goal.GoalAgent` executes it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from agentic_os.llm.base import LLMProvider, Message


class PlanError(ValueError):
    pass


@dataclass
class Task:
    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    agent: str | None = None
    status: str = "pending"  # pending | done | failed | skipped
    result: str | None = None


@dataclass
class Plan:
    goal: str
    tasks: list[Task]

    def get(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.id == task_id:
                return t
        raise KeyError(task_id)

    def validate(self) -> None:
        ids = [t.id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise PlanError("Duplicate task ids in plan")
        for t in self.tasks:
            for dep in t.depends_on:
                if dep not in ids:
                    raise PlanError(f"Task {t.id!r} depends on unknown task {dep!r}")
        self.execution_order()

    def execution_order(self) -> list[Task]:
        """Topological order (Kahn's algorithm), stable w.r.t. the listed order."""
        remaining = {t.id: set(t.depends_on) for t in self.tasks}
        order: list[Task] = []
        while remaining:
            ready = [t for t in self.tasks if t.id in remaining and not remaining[t.id]]
            if not ready:
                raise PlanError(f"Plan has a dependency cycle among: {sorted(remaining)}")
            for t in ready:
                order.append(t)
                del remaining[t.id]
                for deps in remaining.values():
                    deps.discard(t.id)
        return order

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "tasks": [
                {"id": t.id, "description": t.description, "depends_on": t.depends_on, "agent": t.agent, "status": t.status}
                for t in self.tasks
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], goal: str = "") -> Plan:
        tasks = []
        for i, item in enumerate(data.get("tasks", []), 1):
            if isinstance(item, str):
                item = {"description": item}
            tasks.append(
                Task(
                    id=str(item.get("id") or f"t{i}"),
                    description=str(item.get("description") or item.get("task") or "").strip(),
                    depends_on=[str(d) for d in item.get("depends_on", [])],
                    agent=item.get("agent"),
                )
            )
        tasks = [t for t in tasks if t.description]
        if not tasks:
            raise PlanError("Plan contains no tasks")
        plan = cls(goal=str(data.get("goal") or goal), tasks=tasks)
        plan.validate()
        return plan


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of model output (tolerates code fences/prose)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise PlanError("No JSON object found in model output")
        candidate = text[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise PlanError(f"Invalid JSON plan: {exc}") from exc
    if not isinstance(data, dict):
        raise PlanError("Plan JSON must be an object")
    return data


PLANNER_SYSTEM = """You are the planning module of an agent system. Decompose the user's goal into a small set \
of concrete sub-tasks (usually 2-6) that, done in order, achieve it. Each task must be specific enough for one \
agent to complete with tools. Use depends_on to list the ids of tasks whose results a task needs.{agents}

Respond with a JSON plan and nothing else, in exactly this shape:
{{"goal": "...", "tasks": [{{"id": "t1", "description": "...", "depends_on": []{agent_field}}}]}}"""


class Planner:
    """LLM-driven hierarchical task decomposition, with a safe fallback."""

    def __init__(self, llm: LLMProvider, agents: dict[str, str] | None = None, max_tokens: int = 4000) -> None:
        self.llm = llm
        self.agents = agents or {}
        self.max_tokens = max_tokens

    def system_prompt(self) -> str:
        if self.agents:
            roster = "\n".join(f"- {name}: {desc}" for name, desc in self.agents.items())
            return PLANNER_SYSTEM.format(
                agents=f"\n\nAssign each task to the best agent from this roster:\n{roster}",
                agent_field=', "agent": "<name>"',
            )
        return PLANNER_SYSTEM.format(agents="", agent_field="")

    def plan(self, goal: str, context: str | None = None) -> Plan:
        prompt = f"Goal: {goal}" + (f"\n\nContext:\n{context}" if context else "")
        response = self.llm.complete([Message.user(prompt)], system=self.system_prompt(), max_tokens=self.max_tokens)
        try:
            plan = Plan.from_dict(extract_json(response.text), goal=goal)
        except PlanError:
            plan = Plan(goal=goal, tasks=[Task(id="t1", description=goal)])
        for t in plan.tasks:
            if t.agent and self.agents and t.agent not in self.agents:
                t.agent = None
        return plan


__all__ = ["Plan", "PlanError", "Planner", "Task", "extract_json"]
