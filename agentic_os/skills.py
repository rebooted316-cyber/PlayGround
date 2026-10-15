"""Skills: reusable instruction packs that agents load on demand.

A skill is a name, a one-line description and full instructions (optionally
the tools it relies on), the same shape as a ``SKILL.md`` file:

    ---
    name: invoice-checker
    description: Check invoices for errors and missing fields.
    tools: calculator, csv_summary
    ---
    Step-by-step instructions...

Agents see only the names and descriptions of their skills in the system
prompt and call ``load_skill`` to read the full instructions when a task
needs them (progressive disclosure keeps prompts small).
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

from agentic_os.tools.registry import Tool

_NAME = re.compile(r"[a-zA-Z0-9_-]{1,64}")


@dataclass
class Skill:
    name: str
    description: str
    instructions: str
    tools: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError("Skill name must match [a-zA-Z0-9_-]{1,64}")
        if not self.description.strip() or not self.instructions.strip():
            raise ValueError("A skill needs a description and instructions")


def parse_skill_md(text: str, fallback_name: str = "") -> Skill:
    """Parse a SKILL.md document (YAML-style frontmatter + markdown body)."""
    meta: dict[str, str] = {}
    body = text.strip()
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", body, re.DOTALL)
    if match:
        for line in match.group(1).splitlines():
            if ":" in line and not line.startswith((" ", "\t", "#")):
                key, _, value = line.partition(":")
                meta[key.strip().lower()] = value.strip().strip("\"'")
        body = match.group(2).strip()
    name = meta.get("name") or fallback_name
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-")[:64]
    description = meta.get("description") or (body.splitlines()[0].lstrip("# ").strip() if body else "")
    tools = [t.strip() for t in re.split(r"[,\s]+", meta.get("tools", "").strip("[]")) if t.strip()]
    skill = Skill(name=name, description=description, instructions=body, tools=tools)
    skill.validate()
    return skill


class SkillRegistry:
    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._skills: dict[str, Skill] = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for item in json.load(fh):
                    self._skills[item["name"]] = Skill(**item)

    def _save(self) -> None:
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump([asdict(s) for s in self._skills.values()], fh)

    def add(self, skill: Skill) -> Skill:
        skill.validate()
        with self._lock:
            self._skills[skill.name] = skill
            self._save()
        return skill

    def remove(self, name: str) -> bool:
        with self._lock:
            found = self._skills.pop(name, None) is not None
            self._save()
        return found

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def names(self) -> list[str]:
        return sorted(self._skills)

    def catalog(self, names: list[str]) -> str:
        """System-prompt section listing the given skills (name + description only)."""
        skills = [self._skills[n] for n in names if n in self._skills]
        if not skills:
            return ""
        lines = "\n".join(f"- {s.name}: {s.description}" for s in skills)
        return (
            "\n\nYou have these skills. When a task matches one, call load_skill with its name first and "
            f"follow the instructions it returns:\n{lines}"
        )

    def loader_tool(self, allowed: list[str] | None = None) -> Tool:
        def load_skill(name: str) -> str:
            skill = self._skills.get(name)
            if skill is None or (allowed is not None and name not in allowed):
                options = ", ".join(allowed if allowed is not None else self.names()) or "none"
                return f"Unknown skill {name!r}. Available: {options}"
            extra = f"\n\nTools this skill uses: {', '.join(skill.tools)}" if skill.tools else ""
            return f"# Skill: {skill.name}\n\n{skill.instructions}{extra}"

        return Tool(
            name="load_skill",
            description="Load the full instructions of one of your skills by name.",
            parameters={"type": "object", "properties": {"name": {"type": "string", "description": "Skill name."}},
                        "required": ["name"]},
            func=load_skill,
            tags={"skills", "readonly"},
        )

    def public(self) -> list[dict[str, Any]]:
        return [asdict(s) for s in self._skills.values()]
