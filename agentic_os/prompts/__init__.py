"""Prompt engineering & adaptation (roadmap stage 8).

* :class:`PromptTemplate`: validated ``{variable}`` templates.
* Prompt strategies: zero-shot, few-shot, chain-of-thought (step-by-step
  reasoning), and role prompting, all composable over a base system prompt.
* :class:`DynamicFewShot`: picks the examples most similar to the current
  task from an example bank (task adaptation at prompt time).
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field

from agentic_os.memory.embeddings import Embedder, HashingEmbedder, cosine


class PromptError(ValueError):
    pass


class PromptTemplate:
    """A ``str.format``-style template that reports missing variables clearly."""

    def __init__(self, template: str) -> None:
        self.template = template
        self.variables = sorted(
            {name for _, name, _, _ in string.Formatter().parse(template) if name}
        )

    def render(self, **values: object) -> str:
        missing = [v for v in self.variables if v not in values]
        if missing:
            raise PromptError(f"Missing prompt variable(s): {', '.join(missing)}")
        return self.template.format(**values)

    def partial(self, **values: object) -> PromptTemplate:
        """Fill some variables now, leave the rest for later."""
        text = self.template
        for key, val in values.items():
            text = text.replace("{" + key + "}", str(val).replace("{", "{{").replace("}", "}}"))
        return PromptTemplate(text)


@dataclass
class Example:
    input: str
    output: str


@dataclass
class PromptStrategy:
    """Transforms a base system prompt for a given task."""

    name: str = "zero_shot"

    def apply(self, system: str, task: str = "") -> str:
        return system


@dataclass
class ZeroShot(PromptStrategy):
    name: str = "zero_shot"


@dataclass
class FewShot(PromptStrategy):
    examples: list[Example] = field(default_factory=list)
    name: str = "few_shot"

    def apply(self, system: str, task: str = "") -> str:
        return system + _render_examples(self.examples)


@dataclass
class ChainOfThought(PromptStrategy):
    """Asks the model to reason step by step before answering.

    Current Claude models already think adaptively, so this mostly shapes
    the *visible* answer: worked steps, then a clearly marked result.
    """

    name: str = "chain_of_thought"

    def apply(self, system: str, task: str = "") -> str:
        return (
            system
            + "\n\nWork through the problem step by step: break it into parts, use tools to check facts and arithmetic, "
            "then give the final answer on its own line starting with 'Answer:'."
        )


@dataclass
class RolePrompt(PromptStrategy):
    role: str = "a meticulous expert"
    name: str = "role"

    def apply(self, system: str, task: str = "") -> str:
        return f"You are {self.role}.\n\n{system}"


class DynamicFewShot(PromptStrategy):
    """Few-shot prompting with examples retrieved by similarity to the task."""

    def __init__(self, examples: list[Example], k: int = 3, embedder: Embedder | None = None) -> None:
        super().__init__(name="dynamic_few_shot")
        self.embedder = embedder or HashingEmbedder()
        self.k = k
        self._bank = [(ex, self.embedder.embed(ex.input)) for ex in examples]

    def select(self, task: str) -> list[Example]:
        qv = self.embedder.embed(task)
        ranked = sorted(self._bank, key=lambda item: cosine(qv, item[1]), reverse=True)
        return [ex for ex, _ in ranked[: self.k]]

    def apply(self, system: str, task: str = "") -> str:
        return system + _render_examples(self.select(task))


def _render_examples(examples: list[Example]) -> str:
    if not examples:
        return ""
    body = "\n\n".join(
        f"<example>\n<input>{ex.input}</input>\n<output>{ex.output}</output>\n</example>" for ex in examples
    )
    return f"\n\nHere are examples of the expected input and output:\n{body}"


def default_strategies() -> dict[str, PromptStrategy]:
    return {s.name: s for s in (ZeroShot(), ChainOfThought(), RolePrompt())}


__all__ = [
    "ChainOfThought",
    "DynamicFewShot",
    "Example",
    "FewShot",
    "PromptError",
    "PromptStrategy",
    "PromptTemplate",
    "RolePrompt",
    "ZeroShot",
    "default_strategies",
]
