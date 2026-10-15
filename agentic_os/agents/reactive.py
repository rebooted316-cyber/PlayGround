"""Reactive (reflex) agents: condition -> action rules, no model, no planning.

Roadmap stage 5 contrasts *task-oriented reactive* agents with *goal-oriented*
ones. Reflex agents are fast, cheap and predictable; use them as guards or
routers in front of LLM agents.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Rule:
    name: str
    condition: Callable[[str], bool]
    action: Callable[[str], str]

    @classmethod
    def regex(cls, name: str, pattern: str, action: Callable[[str], str]) -> Rule:
        compiled = re.compile(pattern, re.IGNORECASE)
        return cls(name, lambda text: bool(compiled.search(text)), action)


class ReflexAgent:
    """Fires the first rule whose condition matches the percept."""

    def __init__(self, rules: list[Rule], default: Callable[[str], str] | None = None, name: str = "reflex") -> None:
        self.name = name
        self.rules = list(rules)
        self.default = default or (lambda text: "No rule matched.")
        self.last_rule: str | None = None

    def run(self, percept: str) -> str:
        for rule in self.rules:
            if rule.condition(percept):
                self.last_rule = rule.name
                return rule.action(percept)
        self.last_rule = None
        return self.default(percept)
