"""Multi-agent collaboration.

Roadmap stages 5 and 7: several specialized agents work together under one
of four processes:

* ``sequential``: a pipeline; each agent builds on the previous output.
* ``manager``: a manager agent delegates to specialists through tools.
* ``router``: one specialist is chosen per request (a trained classifier or the LLM).
* ``plan``: a planner decomposes the goal and assigns tasks to agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_os.agents.base import Agent
from agentic_os.agents.goal import GoalAgent
from agentic_os.llm.base import LLMProvider, Message
from agentic_os.tools.registry import Tool

PROCESSES = ("sequential", "manager", "router", "plan")


@dataclass
class CrewResult:
    output: str
    process: str
    transcript: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"output": self.output, "process": self.process, "transcript": self.transcript}


class Crew:
    def __init__(
        self,
        agents: list[Agent] | dict[str, Agent],
        llm: LLMProvider | None = None,
        process: str = "manager",
        router: Any = None,
        manager_instructions: str | None = None,
    ) -> None:
        if process not in PROCESSES:
            raise ValueError(f"process must be one of {PROCESSES}")
        self.agents = agents if isinstance(agents, dict) else {a.name: a for a in agents}
        if not self.agents:
            raise ValueError("A crew needs at least one agent")
        self.llm = llm or next(iter(self.agents.values())).llm
        self.process = process
        self.router = router
        self.manager_instructions = manager_instructions or (
            "You are the manager of a team of specialist agents. Break the request into parts, delegate each part "
            "to the most suitable specialist using your tools (give them complete, self-contained instructions), "
            "check their work, and then write the final answer yourself."
        )

    def run(self, task: str) -> CrewResult:
        return getattr(self, f"_run_{self.process}")(task)

    # ------------------------------------------------------------ processes
    def _run_sequential(self, task: str) -> CrewResult:
        transcript, previous = [], None
        for name, agent in self.agents.items():
            context = f"Output from the previous agent:\n{previous}" if previous else None
            res = agent.run(task, context=context)
            transcript.append({"agent": name, "input": task, "output": res.output})
            previous = res.output
        return CrewResult(previous or "", "sequential", transcript)

    def _run_manager(self, task: str) -> CrewResult:
        transcript: list[dict[str, Any]] = []

        def recorded(agent: Agent) -> Tool:
            base = agent.as_tool()

            def delegate(task: str) -> str:
                out = agent.run(task).output
                transcript.append({"agent": agent.name, "input": task, "output": out})
                return out

            base.func = delegate
            return base

        manager = Agent(
            "manager",
            self.llm,
            instructions=self.manager_instructions
            + "\n\nYour team:\n"
            + "\n".join(f"- ask_{n}: {a.instructions.splitlines()[0][:160]}" for n, a in self.agents.items()),
            tools=[recorded(a) for a in self.agents.values()],
            max_steps=12,
        )
        res = manager.run(task)
        transcript.append({"agent": "manager", "input": task, "output": res.output})
        return CrewResult(res.output, "manager", transcript)

    def route(self, task: str) -> str:
        """Choose the agent for ``task``: trained router first, then the LLM."""
        if self.router is not None:
            choice = self.router.predict(task)
            if choice in self.agents:
                return choice
        roster = "\n".join(f"- {n}: {a.instructions.splitlines()[0][:160]}" for n, a in self.agents.items())
        response = self.llm.complete(
            [Message.user(f"Request: {task}\n\nAgents:\n{roster}\n\nReply with only the name of the best agent.")],
            system="You route requests to the single most suitable agent.",
            max_tokens=256,
        )
        answer = response.text.strip().strip("`'\". ").lower()
        for name in self.agents:
            if name.lower() == answer or name.lower() in answer:
                return name
        return next(iter(self.agents))

    def _run_router(self, task: str) -> CrewResult:
        name = self.route(task)
        res = self.agents[name].run(task)
        return CrewResult(res.output, "router", [{"agent": name, "input": task, "output": res.output}])

    def _run_plan(self, task: str) -> CrewResult:
        result = GoalAgent(self.llm, self.agents).run(task)
        transcript = [
            {"agent": t.agent or "default", "task_id": t.id, "input": t.description, "output": t.result, "status": t.status}
            for t in result.plan.tasks
        ]
        return CrewResult(result.output, "plan", transcript)
