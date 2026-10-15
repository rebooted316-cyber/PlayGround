"""Multi-armed bandits: reinforcement learning for choosing among options.

Each *arm* is a choice the agent can make (a prompt strategy, a model, a
tool configuration). Rewards from feedback update the estimates, and the
bandit balances exploring new arms against exploiting the best one.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


@dataclass
class ArmStats:
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean(self) -> float:
        return self.total_reward / self.pulls if self.pulls else 0.0


class Bandit:
    def __init__(self, arms: list[str], seed: int | None = 0) -> None:
        if not arms:
            raise ValueError("A bandit needs at least one arm")
        self.arms = list(arms)
        self.stats = {a: ArmStats() for a in self.arms}
        self.rng = random.Random(seed)

    def select(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def update(self, arm: str, reward: float) -> None:
        if arm not in self.stats:
            raise KeyError(f"Unknown arm: {arm!r}")
        s = self.stats[arm]
        s.pulls += 1
        s.total_reward += float(reward)

    def best(self) -> str:
        return max(self.arms, key=lambda a: (self.stats[a].mean, self.stats[a].pulls))

    def summary(self) -> dict[str, dict[str, float]]:
        return {a: {"pulls": s.pulls, "mean_reward": round(s.mean, 4)} for a, s in self.stats.items()}

    def to_dict(self) -> dict[str, Any]:
        return {a: [s.pulls, s.total_reward] for a, s in self.stats.items()}

    def load_dict(self, data: dict[str, Any]) -> None:
        for arm, (pulls, total) in data.items():
            if arm in self.stats:
                self.stats[arm] = ArmStats(int(pulls), float(total))


class EpsilonGreedy(Bandit):
    """Explore a random arm with probability epsilon, else exploit the best."""

    def __init__(self, arms: list[str], epsilon: float = 0.1, seed: int | None = 0) -> None:
        super().__init__(arms, seed)
        self.epsilon = epsilon

    def select(self) -> str:
        untried = [a for a in self.arms if self.stats[a].pulls == 0]
        if untried:
            return untried[0]
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.arms)
        return self.best()


class UCB1(Bandit):
    """Upper Confidence Bound: optimism in the face of uncertainty."""

    def __init__(self, arms: list[str], c: float = math.sqrt(2), seed: int | None = 0) -> None:
        super().__init__(arms, seed)
        self.c = c

    def select(self) -> str:
        for a in self.arms:
            if self.stats[a].pulls == 0:
                return a
        total = sum(s.pulls for s in self.stats.values())
        return max(
            self.arms,
            key=lambda a: self.stats[a].mean + self.c * math.sqrt(math.log(total) / self.stats[a].pulls),
        )
