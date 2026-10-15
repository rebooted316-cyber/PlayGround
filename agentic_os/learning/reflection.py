"""Self-improvement loops (roadmap stage 9).

* :class:`LLMJudge`: model-graded evaluation against a rubric.
* :class:`ReflectiveAgent`: Reflexion-style retry. Evaluate, critique,
  retry with the lessons learned, and store lessons in long-term memory.
* :class:`AdaptiveAgent`: a bandit learns which prompt strategy earns the
  highest reward, from automatic or human feedback.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentic_os.agents.base import Agent, AgentResult
from agentic_os.learning.bandit import Bandit, EpsilonGreedy
from agentic_os.learning.feedback import FeedbackStore
from agentic_os.llm.base import LLMProvider, Message
from agentic_os.memory.long_term import LongTermMemory
from agentic_os.planning import PlanError, extract_json
from agentic_os.prompts import PromptStrategy

Evaluator = Callable[[str, str], float]  # (task, output) -> score in [0, 1]


class LLMJudge:
    """Grades an output 0-10 against a rubric; returns a score in [0, 1]."""

    def __init__(self, llm: LLMProvider, rubric: str = "Correct, complete, well-grounded and concise.") -> None:
        self.llm = llm
        self.rubric = rubric
        self.last_reason = ""

    def __call__(self, task: str, output: str) -> float:
        response = self.llm.complete(
            [Message.user(f"<task>\n{task}\n</task>\n<answer>\n{output}\n</answer>")],
            system=(
                "You are a strict evaluator. Grade the answer to the task from 0 to 10 using this rubric: "
                f"{self.rubric}\nReply with JSON only: {{\"score\": <0-10>, \"reason\": \"<one sentence>\"}}"
            ),
            max_tokens=1000,
        )
        try:
            data = extract_json(response.text)
            score = float(data.get("score", 0))
            self.last_reason = str(data.get("reason", ""))
        except (PlanError, TypeError, ValueError):
            self.last_reason = "Could not parse judge output."
            return 0.0
        return max(0.0, min(1.0, score / 10))


@dataclass
class Attempt:
    output: str
    score: float
    critique: str = ""


@dataclass
class ReflectionResult:
    output: str
    best_score: float
    attempts: list[Attempt] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "best_score": self.best_score,
            "attempts": [a.__dict__ for a in self.attempts],
        }


class ReflectiveAgent:
    def __init__(
        self,
        agent: Agent,
        evaluator: Evaluator,
        max_attempts: int = 3,
        threshold: float = 0.8,
        lessons: LongTermMemory | None = None,
    ) -> None:
        self.agent = agent
        self.evaluator = evaluator
        self.max_attempts = max_attempts
        self.threshold = threshold
        self.lessons = lessons

    def _critique(self, task: str, output: str, score: float) -> str:
        response = self.agent.llm.complete(
            [
                Message.user(
                    f"Task: {task}\n\nYour previous answer (score {score:.2f}/1.00):\n{output}\n\n"
                    "In 2-4 bullet points, say what was wrong or missing and exactly what to do differently next time."
                )
            ],
            system="You are reflecting on your own work to improve it. Be specific and actionable.",
            max_tokens=1500,
        )
        return response.text.strip()

    def run(self, task: str) -> ReflectionResult:
        reflections: list[str] = []
        if self.lessons is not None:
            reflections += [m.text for m in self.lessons.recall(task, k=3, kind="lesson")]
        attempts: list[Attempt] = []
        for _ in range(self.max_attempts):
            context = None
            if reflections:
                context = "Lessons from earlier attempts (apply them):\n" + "\n".join(reflections)
            res = self.agent.run(task, context=context)
            score = self.evaluator(task, res.output)
            attempt = Attempt(res.output, score)
            attempts.append(attempt)
            if score >= self.threshold:
                break
            attempt.critique = self._critique(task, res.output, score)
            reflections.append(attempt.critique)
            if self.lessons is not None:
                self.lessons.remember(attempt.critique, kind="lesson", task=task[:200])
        best = max(attempts, key=lambda a: a.score)
        return ReflectionResult(best.output, best.score, attempts)


class AdaptiveAgent:
    """Learns which prompt strategy works best via a bandit over rewards.

    Rewards come from ``evaluator`` (automatic) and/or :meth:`reward`
    (human feedback for a ``run_id``, in [-1, 1]).
    """

    def __init__(
        self,
        agent: Agent,
        strategies: dict[str, PromptStrategy],
        bandit: Bandit | None = None,
        evaluator: Evaluator | None = None,
        feedback: FeedbackStore | None = None,
    ) -> None:
        self.agent = agent
        self.strategies = strategies
        self.bandit = bandit or EpsilonGreedy(list(strategies))
        self.evaluator = evaluator
        self.feedback = feedback or FeedbackStore()
        self._pending: dict[str, str] = {}

    def run(self, task: str) -> tuple[str, AgentResult]:
        arm = self.bandit.select()
        self.agent.strategy = self.strategies[arm]
        result = self.agent.run(task)
        run_id = uuid.uuid4().hex[:12]
        self._pending[run_id] = arm
        if self.evaluator is not None:
            self.reward(run_id, self.evaluator(task, result.output), source="evaluator")
        return run_id, result

    def reward(self, run_id: str, reward: float, comment: str = "", source: str = "human") -> None:
        arm = self._pending.get(run_id)
        if arm is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        self.feedback.add(run_id, max(-1.0, min(1.0, reward)), arm=arm, comment=comment, source=source)
        self.bandit.update(arm, reward)
