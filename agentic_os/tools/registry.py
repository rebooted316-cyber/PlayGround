"""Function calling: turn plain Python functions into model-callable tools.

Roadmap stage 3 (Programming & AI Frameworks): the ``@tool`` decorator
derives a JSON Schema from type hints and the docstring, and
:class:`ToolRegistry` validates arguments and executes calls safely.
"""

from __future__ import annotations

import inspect
import json
import re
import types
import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    tuple: "array",
    dict: "object",
}


class ToolError(Exception):
    """Raised for invalid tool definitions, unknown tools or bad arguments."""


@dataclass
class ToolResult:
    """Outcome of executing one tool call."""

    content: str
    is_error: bool = False


@dataclass
class Tool:
    """A callable exposed to models, with its JSON Schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    tags: set[str] = field(default_factory=set)

    def to_anthropic_schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)


def _json_type(annotation: Any) -> dict[str, Any]:
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _json_type(non_none[0])
        return {}
    if origin is typing.Literal:
        return {"type": _JSON_TYPES.get(type(args[0]), "string"), "enum": list(args)}
    if origin in (list, tuple, set):
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = _json_type(args[0])
        return schema
    if origin is dict:
        return {"type": "object"}
    if annotation in _JSON_TYPES:
        return {"type": _JSON_TYPES[annotation]}
    return {}


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into a summary and per-arg descriptions."""
    if not doc:
        return "", {}
    doc = inspect.cleandoc(doc)
    parts = re.split(r"\n\s*(?:Args|Arguments|Parameters):\s*\n", doc, maxsplit=1)
    summary = parts[0].strip()
    arg_docs: dict[str, str] = {}
    if len(parts) == 2:
        current = None
        for line in parts[1].splitlines():
            m = re.match(r"\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)", line)
            if m and not line.startswith("        "):
                current = m.group(1)
                arg_docs[current] = m.group(2).strip()
            elif current and line.strip():
                arg_docs[current] += " " + line.strip()
            elif re.match(r"\s*(Returns|Raises):", line):
                break
    return summary, arg_docs


def make_tool(
    func: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
    tags: Iterable[str] = (),
) -> Tool:
    """Build a :class:`Tool` from a function's signature and docstring."""
    summary, arg_docs = _parse_docstring(func.__doc__ or "")
    hints = typing.get_type_hints(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in inspect.signature(func).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        schema = _json_type(hints.get(pname, str))
        if pname in arg_docs:
            schema["description"] = arg_docs[pname]
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        elif isinstance(param.default, (str, int, float, bool)):
            schema["default"] = param.default
        properties[pname] = schema
    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    desc = description or summary or (func.__name__.replace("_", " "))
    return Tool(name or func.__name__, desc, parameters, func, set(tags))


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    tags: Iterable[str] = (),
) -> Any:
    """Decorator: ``@tool`` or ``@tool(name=..., tags=[...])``."""

    def wrap(f: Callable[..., Any]) -> Tool:
        return make_tool(f, name=name, description=description, tags=tags)

    return wrap(func) if func is not None else wrap


def _coerce(value: Any, schema: dict[str, Any], arg: str) -> Any:
    expected = schema.get("type")
    if expected is None:
        return value
    if "enum" in schema and value not in schema["enum"]:
        raise ToolError(f"Argument {arg!r} must be one of {schema['enum']}, got {value!r}")
    try:
        if expected == "integer":
            if isinstance(value, bool):
                raise ValueError
            if isinstance(value, float) and not value.is_integer():
                raise ValueError
            return int(value)
        if expected == "number":
            if isinstance(value, bool):
                raise ValueError
            return float(value) if not isinstance(value, (int, float)) else value
        if expected == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in {"true", "false"}:
                return value.lower() == "true"
            raise ValueError
        if expected == "string":
            if isinstance(value, (dict, list)):
                raise ValueError
            return str(value)
        if expected == "array":
            if isinstance(value, str):
                value = json.loads(value)
            if not isinstance(value, list):
                raise ValueError
            return value
        if expected == "object":
            if isinstance(value, str):
                value = json.loads(value)
            if not isinstance(value, dict):
                raise ValueError
            return value
    except (TypeError, ValueError) as exc:
        raise ToolError(f"Argument {arg!r} should be {expected}, got {value!r}") from exc
    return value


class ToolRegistry:
    """Holds tools and executes model tool calls with validation."""

    def __init__(self, tools: Iterable[Tool | Callable[..., Any]] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, t: Tool | Callable[..., Any]) -> Tool:
        if not isinstance(t, Tool):
            t = make_tool(t)
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", t.name):
            raise ToolError(f"Invalid tool name: {t.name!r}")
        self._tools[t.name] = t
        return t

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"Unknown tool: {name!r}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def names(self) -> list[str]:
        return sorted(self._tools)

    def subset(self, names: Iterable[str] | None = None, tags: Iterable[str] | None = None) -> ToolRegistry:
        """A new registry with only the named and/or tagged tools."""
        names = set(names or [])
        tags = set(tags or [])
        chosen = [
            t for t in self._tools.values() if (t.name in names) or (tags and t.tags & tags)
        ]
        return ToolRegistry(chosen)

    def validate(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        t = self.get(name)
        if not isinstance(arguments, dict):
            raise ToolError("Tool arguments must be a JSON object")
        props = t.parameters.get("properties", {})
        missing = [r for r in t.parameters.get("required", []) if r not in arguments]
        if missing:
            raise ToolError(f"Missing required argument(s) for {name}: {', '.join(missing)}")
        unknown = [a for a in arguments if a not in props]
        if unknown:
            raise ToolError(f"Unknown argument(s) for {name}: {', '.join(unknown)}")
        return {k: _coerce(v, props[k], k) for k, v in arguments.items()}

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run a tool. Errors are returned as ``is_error`` results, never raised,
        so the model can see what went wrong and recover."""
        try:
            args = self.validate(name, arguments)
            output = self._tools[name].func(**args)
        except ToolError as exc:
            return ToolResult(str(exc), is_error=True)
        except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model
            return ToolResult(f"{type(exc).__name__}: {exc}", is_error=True)
        if isinstance(output, str):
            return ToolResult(output)
        try:
            return ToolResult(json.dumps(output, default=str))
        except (TypeError, ValueError):
            return ToolResult(str(output))
