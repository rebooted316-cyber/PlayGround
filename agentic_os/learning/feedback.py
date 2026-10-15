"""Feedback collection: the reward signal for self-improvement (RLHF-style)."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Feedback:
    run_id: str
    reward: float
    arm: str | None = None
    comment: str = ""
    source: str = "human"
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class FeedbackStore:
    """Append-only feedback log, optionally persisted as JSON Lines."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.items: list[Feedback] = []
        self._lock = threading.Lock()
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                self.items = [Feedback(**json.loads(line)) for line in fh if line.strip()]

    def add(self, run_id: str, reward: float, arm: str | None = None, comment: str = "", source: str = "human", **metadata: Any) -> Feedback:
        if not -1.0 <= reward <= 1.0:
            raise ValueError("reward must be in [-1, 1]")
        fb = Feedback(run_id, reward, arm, comment, source, metadata=metadata)
        with self._lock:
            self.items.append(fb)
            if self.path:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(fb)) + "\n")
        return fb

    def for_run(self, run_id: str) -> list[Feedback]:
        return [f for f in self.items if f.run_id == run_id]

    def average(self, arm: str | None = None) -> float:
        items = [f.reward for f in self.items if arm is None or f.arm == arm]
        return sum(items) / len(items) if items else 0.0

    def __len__(self) -> int:
        return len(self.items)
