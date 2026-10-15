"""Mods: custom tools defined at runtime, without code.

A mod turns any HTTP API into an agent tool: a name, a description, a URL,
a method, optional headers (for API keys) and the parameters the model fills
in. GET sends parameters as the query string, other methods as a JSON body.
``{param}`` placeholders in the URL are filled from the parameters too.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

from agentic_os.providers import mask
from agentic_os.tools.registry import Tool

_NAME = re.compile(r"[a-zA-Z0-9_-]{1,48}")
_TYPES = {"string", "integer", "number", "boolean"}
MAX_RESPONSE = 8000


@dataclass
class ModParam:
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True


@dataclass
class Mod:
    name: str
    description: str
    url: str
    method: str = "GET"
    headers: dict[str, str] = field(default_factory=dict)
    params: list[ModParam] = field(default_factory=list)
    enabled: bool = True

    def validate(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError("Mod name must match [a-zA-Z0-9_-]{1,48}")
        if not re.match(r"https?://", self.url):
            raise ValueError("Mod url must start with http:// or https://")
        if self.method.upper() not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError("Unsupported HTTP method")
        for p in self.params:
            if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]{0,40}", p.name) or p.type not in _TYPES:
                raise ValueError(f"Invalid parameter {p.name!r} (type must be one of {sorted(_TYPES)})")

    @property
    def tool_name(self) -> str:
        return f"mod_{self.name}".replace("-", "_")[:64]

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["headers"] = {k: mask(v) for k, v in self.headers.items()}
        data["tool"] = self.tool_name
        return data

    def to_tool(self, timeout: float = 60.0) -> Tool:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {p.name: {"type": p.type, "description": p.description} for p in self.params},
        }
        required = [p.name for p in self.params if p.required]
        if required:
            schema["required"] = required

        def call(**kwargs: Any) -> str:
            url = self.url
            for key in list(kwargs):
                if "{" + key + "}" in url:
                    url = url.replace("{" + key + "}", urllib.parse.quote(str(kwargs.pop(key)), safe=""))
            method = self.method.upper()
            data = None
            headers = {"User-Agent": "agentic-os-mod", **self.headers}
            if method == "GET" or method == "DELETE":
                if kwargs:
                    url += ("&" if "?" in url else "?") + urllib.parse.urlencode(kwargs)
            else:
                data = json.dumps(kwargs).encode()
                headers["Content-Type"] = "application/json"
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    text = resp.read(MAX_RESPONSE + 1).decode("utf-8", "replace")
            except urllib.error.HTTPError as exc:
                return f"HTTP {exc.code}: {exc.read(2000).decode('utf-8', 'replace')}"
            except urllib.error.URLError as exc:
                return f"Request failed: {exc.reason}"
            return text[:MAX_RESPONSE] + (" …[truncated]" if len(text) > MAX_RESPONSE else "")

        tags = {"mod", "readonly"} if self.method.upper() == "GET" else {"mod"}
        return Tool(name=self.tool_name, description=self.description, parameters=schema, func=call, tags=tags)


def mod_from_dict(data: dict[str, Any]) -> Mod:
    params = [p if isinstance(p, ModParam) else ModParam(**p) for p in data.get("params") or []]
    mod = Mod(
        name=str(data.get("name", "")), description=str(data.get("description", "")), url=str(data.get("url", "")),
        method=str(data.get("method", "GET")).upper(), headers={str(k): str(v) for k, v in (data.get("headers") or {}).items()},
        params=params, enabled=bool(data.get("enabled", True)),
    )
    mod.validate()
    return mod


class ModRegistry:
    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._mods: dict[str, Mod] = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for item in json.load(fh):
                    self._mods[item["name"]] = mod_from_dict(item)

    def _save(self) -> None:
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # headers may hold API keys
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump([asdict(m) for m in self._mods.values()], fh)

    def add(self, mod: Mod, keep_existing_headers: bool = True) -> Mod:
        mod.validate()
        with self._lock:
            existing = self._mods.get(mod.name)
            if keep_existing_headers and existing:
                for key, value in existing.headers.items():
                    # Editing keeps stored secrets unless replaced: absent, empty or masked values mean "unchanged".
                    if not mod.headers.get(key) or mod.headers[key].startswith("•"):
                        mod.headers[key] = value
            self._mods[mod.name] = mod
            self._save()
        return mod

    def remove(self, name: str) -> Mod | None:
        with self._lock:
            mod = self._mods.pop(name, None)
            self._save()
        return mod

    def all(self) -> list[Mod]:
        return list(self._mods.values())
