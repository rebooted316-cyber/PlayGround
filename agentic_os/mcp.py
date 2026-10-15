"""Model Context Protocol (MCP) client: plug any MCP server's tools into the OS.

Two transports:

* ``http``: Streamable HTTP (remote servers). JSON-RPC over POST; replies may
  be plain JSON or a Server-Sent Events stream. The ``Mcp-Session-Id`` header
  is carried across requests.
* ``stdio``: a local command speaking newline-delimited JSON-RPC on
  stdin/stdout (for example ``npx -y @modelcontextprotocol/server-filesystem /data``).
  The command must be installed where Agentic OS runs.

Each server tool becomes an agent tool named ``mcp_<server>_<tool>``.
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import re
import shlex
import subprocess
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

from agentic_os.providers import mask
from agentic_os.tools.registry import Tool

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "agentic-os", "version": "0.1.0"}
_NAME = re.compile(r"[a-zA-Z0-9_-]{1,32}")


class MCPError(RuntimeError):
    pass


@dataclass
class MCPServerConfig:
    name: str
    transport: str = "http"  # http | stdio
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    command: str = ""
    env: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError("MCP server name must match [a-zA-Z0-9_-]{1,32}")
        if self.transport == "http":
            if not re.match(r"https?://", self.url):
                raise ValueError("MCP url must start with http:// or https://")
        elif self.transport == "stdio":
            if not self.command.strip():
                raise ValueError("A stdio MCP server needs a command")
        else:
            raise ValueError("transport must be 'http' or 'stdio'")

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["headers"] = {k: mask(v) for k, v in self.headers.items()}
        data["env"] = {k: mask(v) for k, v in self.env.items()}
        return data


def _parse_sse(body: str, want_id: Any) -> dict[str, Any] | None:
    """Return the JSON-RPC message with ``want_id`` from an SSE stream body."""
    data_lines: list[str] = []
    for line in body.splitlines() + [""]:
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif line == "" and data_lines:
            try:
                msg = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                msg = None
            data_lines = []
            if isinstance(msg, dict) and msg.get("id") == want_id:
                return msg
    return None


class _HttpTransport:
    def __init__(self, cfg: MCPServerConfig, timeout: float) -> None:
        self.cfg = cfg
        self.timeout = timeout
        self.session_id: str | None = None
        self.negotiated = False

    def send(self, message: dict[str, Any]) -> dict[str, Any] | None:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", **self.cfg.headers}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.negotiated:
            headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
        req = urllib.request.Request(self.cfg.url, data=json.dumps(message).encode(), method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self.session_id = sid
                ctype = resp.headers.get("Content-Type", "")
                body = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            raise MCPError(f"HTTP {exc.code} from MCP server: {exc.read(500).decode('utf-8', 'replace')}") from exc
        except urllib.error.URLError as exc:
            raise MCPError(f"Cannot reach MCP server: {exc.reason}") from exc
        if "id" not in message:
            return None  # notification: 202 Accepted, no body expected
        if "text/event-stream" in ctype:
            reply = _parse_sse(body, message["id"])
        else:
            reply = json.loads(body) if body.strip() else None
        if reply is None:
            raise MCPError("MCP server sent no response")
        return reply

    def close(self) -> None:
        pass


class _StdioTransport:
    def __init__(self, cfg: MCPServerConfig, timeout: float) -> None:
        self.timeout = timeout
        env = {**os.environ, **cfg.env}
        try:
            self.proc = subprocess.Popen(
                shlex.split(cfg.command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1, env=env,
            )
        except OSError as exc:
            raise MCPError(f"Cannot start MCP command: {exc}") from exc
        self.lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.lock = threading.Lock()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put("")  # EOF marker

    def send(self, message: dict[str, Any]) -> dict[str, Any] | None:
        assert self.proc.stdin is not None
        with self.lock:
            try:
                self.proc.stdin.write(json.dumps(message) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise MCPError("MCP process is not running") from exc
            if "id" not in message:
                return None
            while True:
                try:
                    line = self.lines.get(timeout=self.timeout)
                except queue.Empty:
                    raise MCPError("MCP server did not answer in time") from None
                if line == "":
                    raise MCPError("MCP process exited")
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue  # ignore log lines on stdout
                if isinstance(msg, dict) and msg.get("id") == message["id"]:
                    return msg

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()


class MCPClient:
    """A connected MCP server and its tools."""

    def __init__(self, config: MCPServerConfig, timeout: float = 60.0) -> None:
        config.validate()
        self.config = config
        self.timeout = timeout
        self._ids = itertools.count(1)
        self.transport: _HttpTransport | _StdioTransport | None = None
        self.tools: list[dict[str, Any]] = []
        self.server_info: dict[str, Any] = {}
        self.error: str | None = None

    def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        assert self.transport is not None
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": next(self._ids), "method": method}
        if params is not None:
            message["params"] = params
        reply = self.transport.send(message) or {}
        if reply.get("error"):
            err = reply["error"]
            raise MCPError(f"MCP error {err.get('code')}: {err.get('message')}")
        return reply.get("result") or {}

    def connect(self) -> MCPClient:
        try:
            if self.config.transport == "http":
                self.transport = _HttpTransport(self.config, self.timeout)
            else:
                self.transport = _StdioTransport(self.config, self.timeout)
            init = self._request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                                "clientInfo": CLIENT_INFO})
            self.server_info = init.get("serverInfo") or {}
            if isinstance(self.transport, _HttpTransport):
                self.transport.negotiated = True
            self.transport.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            tools: list[dict[str, Any]] = []
            cursor = None
            for _ in range(20):  # pagination guard
                page = self._request("tools/list", {"cursor": cursor} if cursor else {})
                tools.extend(page.get("tools") or [])
                cursor = page.get("nextCursor")
                if not cursor:
                    break
            self.tools = tools
            self.error = None
        except (MCPError, ValueError, json.JSONDecodeError) as exc:
            self.error = str(exc)
            self.close()
            raise MCPError(self.error) from exc
        return self

    def call(self, tool: str, arguments: dict[str, Any]) -> str:
        if self.transport is None:
            raise MCPError("MCP server is not connected")
        result = self._request("tools/call", {"name": tool, "arguments": arguments})
        parts = []
        for item in result.get("content") or []:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "resource":
                parts.append(json.dumps(item.get("resource", {}))[:4000])
            else:
                parts.append(f"[{item.get('type')} content]")
        if not parts and result.get("structuredContent") is not None:
            parts.append(json.dumps(result["structuredContent"]))
        text = "\n".join(parts) or "(no content)"
        if result.get("isError"):
            raise MCPError(text)
        return text

    def as_tools(self) -> list[Tool]:
        out = []
        for spec in self.tools:
            remote = spec.get("name", "")
            local = re.sub(r"[^a-zA-Z0-9_-]", "_", f"mcp_{self.config.name}_{remote}")[:64]
            schema = spec.get("inputSchema") or {"type": "object", "properties": {}}
            schema.setdefault("type", "object")
            read_only = bool((spec.get("annotations") or {}).get("readOnlyHint"))  # the server's own hint
            out.append(Tool(
                name=local,
                description=(spec.get("description") or f"{remote} (MCP server {self.config.name})")[:1024],
                parameters=schema, func=_McpToolFunc(self, remote),
                tags={"mcp", f"mcp:{self.config.name}"} | ({"readonly"} if read_only else set()),
            ))
        return out

    def status(self) -> dict[str, Any]:
        return {**self.config.public(), "connected": self.transport is not None and self.error is None,
                "error": self.error, "server": self.server_info.get("name"),
                "tools": [t.get("name") for t in self.tools]}

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()
        self.transport = None


class _McpToolFunc:
    """Callable bound to one remote tool (keeps argument names free of collisions)."""

    def __init__(self, client: MCPClient, remote: str) -> None:
        self.client = client
        self.remote = remote

    def __call__(self, **kwargs: Any) -> str:
        return self.client.call(self.remote, kwargs)


def config_from_dict(data: dict[str, Any]) -> MCPServerConfig:
    cfg = MCPServerConfig(
        name=str(data.get("name", "")), transport=str(data.get("transport", "http")), url=str(data.get("url", "")),
        headers={str(k): str(v) for k, v in (data.get("headers") or {}).items()},
        command=str(data.get("command", "")), env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
    )
    cfg.validate()
    return cfg
