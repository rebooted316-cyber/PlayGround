"""Goal-oriented agent: plan, execute sub-tasks in dependency order, synthesize."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_os.agents.base import Agent, AgentResult
from agentic_os.llm.base import LLMProvider, Message
from agentic_os.planning import Plan, Planner

SYNTH_SYSTEM = (
    "You are the final step of a multi-step agent workflow. Using the results of the completed sub-tasks, "
    "write the final answer to the overall goal. Be direct, keep any citations, and mention sub-tasks that failed."
)


@dataclass
class GoalResult:
    goal: str
    output: str
    plan: Plan
    task_results: dict[str, AgentResult] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return all(t.status == "done" for t in self.plan.tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "output": self.output,
            "plan": self.plan.to_dict(),
            "tasks": {tid: r.to_dict() for tid, r in self.task_results.items()},
        }


class GoalAgent:
    """Hierarchical planner-executor.

    Args:
        llm: Model used for planning and synthesis.
        workers: One default worker agent, or a ``{name: agent}`` roster the
            planner assigns tasks to.
        planner: Custom planner (defaults to an LLM :class:`Planner`).
    """

    def __init__(
        self,
        llm: LLMProvider,
        workers: Agent | dict[str, Agent],
        planner: Planner | None = None,
        name: str = "goal_agent",
    ) -> None:
        self.name = name
        self.llm = llm
        self.workers = workers if isinstance(workers, dict) else {workers.name: workers}
        self.default_worker = next(iter(self.workers.values()))
        roster = {n: a.instructions.split("\n")[0][:160] for n, a in self.workers.items()} if len(self.workers) > 1 else None
        self.planner = planner or Planner(llm, agents=roster)

    def run(self, goal: str, plan: Plan | None = None) -> GoalResult:
        plan = plan or self.planner.plan(goal)
        results: dict[str, AgentResult] = {}
        for task in plan.execution_order():
            deps = [plan.get(d) for d in task.depends_on]
            if any(d.status != "done" for d in deps):
                task.status = "skipped"
                continue
            context = f"Overall goal: {goal}"
            if deps:
                context += "\n\nResults of prerequisite tasks:\n" + "\n\n".join(
                    f"[{d.id}] {d.description}\n{d.result}" for d in deps
                )
            worker = self.workers.get(task.agent or "", self.default_worker)
            res = worker.run(task.description, context=context)
            results[task.id] = res
            task.result = res.output
            task.status = "done" if res.stopped_reason == "completed" else "failed"

        output = self._synthesize(goal, plan)
        return GoalResult(goal=goal, output=output, plan=plan, task_results=results)

    def _synthesize(self, goal: str, plan: Plan) -> str:
        if len(plan.tasks) == 1:
            return plan.tasks[0].result or "The task could not be completed."
        report = "\n\n".join(
            f"[{t.id}] ({t.status}) {t.description}\n{t.result or ''}".strip() for t in plan.tasks
        )
        response = self.llm.complete(
            [Message.user(f"Goal: {goal}\n\nSub-task results:\n{report}")], system=SYNTH_SYSTEM, max_tokens=8000
        )
        return response.text or report
