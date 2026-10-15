"""HTTP API for deploying Agentic OS (roadmap stage 11).

Standard library only (``http.server``), so it deploys anywhere Python
runs. Put it behind a reverse proxy (TLS, rate limits) in production.

Endpoints (JSON in, JSON out):

=======  ================================  =========================================
GET      /health                           liveness probe
GET      /status                           OS summary
GET      /metrics                          counters and latency percentiles
GET      /agents                           registered agents
POST     /agents                           spawn ``{name, instructions?, tools?, tool_tags?, conversational?,
                                           mode?, permission?, delegates?}``
POST     /run                              ``{task, agent?, context?, async?}`` -> process
GET      /processes                        process table
GET      /processes/{pid}                  one process, with its result
POST     /processes/{pid}/feedback         ``{reward, comment?}``, reward in [-1, 1]
POST     /knowledge                        ``{text, metadata?}`` or ``{documents: [...]}``
POST     /goal                             ``{goal, agents?}`` -> plan + result
GET      /traces/{trace_id}                spans for one agent run
GET      /                                 War Room web app (PWA)
GET      /providers                        configured model providers (keys masked) + presets
POST     /providers                        add/update ``{id, kind, api_key?, base_url?, default_model?}``
POST     /providers/{id}/test              send a tiny prompt to check the key works
DELETE   /providers/{id}                   remove a provider
POST     /settings/default                 ``{provider, model?}`` OS default model
POST     /agents/external                  ``{name, protocol, url, api_key?, model?, description?}``
DELETE   /agents/{name}                    remove an agent
POST     /chat                             ``{message, agent?, session?, async?}`` multi-turn chat
POST     /chat/reset                       ``{agent?, session?}``
POST     /crew                             ``{task, agents, process?}`` run a crew (async)
GET      /knowledge                        ingested documents
GET      /graph                            knowledge/memory graph for the Brain view
GET/POST /skills, DELETE /skills/{name}    skills (``{name, description, instructions, tools}`` or ``{markdown}``)
GET/POST /mods, DELETE /mods/{name}        custom HTTP tools
GET/POST /mcp, DELETE /mcp/{name}          MCP servers; POST /mcp/{name}/reconnect
POST     /memory/import?filename=&distill= raw export file (zip/json/txt, up to 60 MB) or JSON ``{text}``
GET      /approvals                        tool calls and agent questions waiting for the operator
POST     /approvals/{id}                   ``{approve: bool, remember?: bool}`` or ``{answer: str}`` for a question
DELETE   /agents/{name}/allowed            forget the tools approved with "always allow"
POST     /agents/{name}/settings           ``{permission?, mode?}`` change an agent at any time
=======  ================================  =========================================

Set ``AGENTIC_OS_API_KEY`` to require ``Authorization: Bearer <key>``. The
server refuses to listen on a public interface without it.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from agentic_os.agents.external import AGENT_PRESETS, PROTOCOLS, ExternalAgentConfig
from agentic_os.kernel import AgenticOS, AgenticOSError
from agentic_os.llm.base import LLMError, Message
from agentic_os.mcp import MCPError, config_from_dict
from agentic_os.mods import mod_from_dict
from agentic_os.providers import PRESETS, ProviderConfig
from agentic_os.rag.pipeline import Document
from agentic_os.skills import Skill, parse_skill_md

MAX_BODY = 2 * 1024 * 1024
MAX_UPLOAD = 60 * 1024 * 1024
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json"),
    "/sw.js": ("sw.js", "application/javascript; charset=utf-8"),
    "/icon.svg": ("icon.svg", "image/svg+xml"),
    "/icon-192.png": ("icon-192.png", "image/png"),
    "/icon-512.png": ("icon-512.png", "image/png"),
}
_NAME = re.compile(r"[a-zA-Z0-9_-]{1,40}")


class HTTPError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _require(body: dict[str, Any], key: str, kind: type = str) -> Any:
    value = body.get(key)
    if not isinstance(value, kind) or (kind is str and not value.strip()):
        raise HTTPError(400, f"'{key}' is required and must be a {kind.__name__}")
    return value


def make_handler(system: AgenticOS) -> type[BaseHTTPRequestHandler]:
    api_key = system.settings.api_key

    class Handler(BaseHTTPRequestHandler):
        server_version = "AgenticOS/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter default logging
            system.metrics.incr("http.requests")

        # ----------------------------------------------------------- helpers
        def _send(self, status: int, payload: Any) -> None:
            data = json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise HTTPError(413, "Request body too large")
            if not length:
                return {}
            try:
                body = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as exc:
                raise HTTPError(400, f"Invalid JSON: {exc}") from exc
            if not isinstance(body, dict):
                raise HTTPError(400, "JSON body must be an object")
            return body

        def _authorize(self) -> None:
            path = self.path.split("?", 1)[0]
            if not api_key or path == "/health" or path in STATIC_FILES:
                return
            header = self.headers.get("Authorization", "")
            token = header.removeprefix("Bearer ").strip()
            if not hmac.compare_digest(token.encode(), api_key.encode()):
                raise HTTPError(401, "Missing or invalid API key")

        def _static(self, path: str) -> bool:
            entry = STATIC_FILES.get(path)
            if entry is None:
                return False
            filename, ctype = entry
            with open(os.path.join(STATIC_DIR, filename), "rb") as fh:
                data = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)
            return True

        def _dispatch(self, method: str) -> None:
            try:
                path = self.path.split("?", 1)[0]
                if method == "GET" and self._static(path):
                    return
                self._authorize()
                status, payload = self._route(method, path.rstrip("/") or "/")
                self._send(status, payload)
            except HTTPError as exc:
                self._send(exc.status, {"error": exc.message})
            except AgenticOSError as exc:
                self._send(404, {"error": str(exc)})
            except (LLMError, MCPError, ValueError) as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                system.metrics.incr("http.errors")
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_DELETE(self) -> None:
            self._dispatch("DELETE")

        # ------------------------------------------------------------ routes
        def _route(self, method: str, path: str) -> tuple[int, Any]:
            if method == "GET":
                if path == "/health":
                    return 200, {"status": "ok"}
                if path == "/status":
                    return 200, system.status()
                if path == "/metrics":
                    return 200, system.metrics.snapshot()
                if path == "/agents":
                    return 200, {"agents": [system.describe(n) for n in system.agents],
                                 "external_presets": AGENT_PRESETS, "tools": system.tools.names()}
                if path == "/providers":
                    return 200, {"providers": system.providers.public(), "presets": PRESETS,
                                 "default": {"provider": system.prefs.get("default_provider") or "env",
                                             "model": getattr(system.llm, "model", None),
                                             "name": getattr(system.llm, "name", None)}}
                if path == "/knowledge":
                    docs: dict[str, dict[str, Any]] = {}
                    for rec in system.knowledge.vectors.all():
                        doc_id = str(rec.metadata.get("doc_id"))
                        entry = docs.setdefault(doc_id, {"id": doc_id, "source": rec.metadata.get("source")
                                                         or rec.metadata.get("title") or "bez nazwy",
                                                         "chunks": 0, "preview": rec.text[:160]})
                        entry["chunks"] += 1
                    return 200, {"documents": list(docs.values())}
                if path == "/graph":
                    return 200, system.graph()
                if path == "/auth/check":
                    return 200, {"ok": True}
                if path == "/skills":
                    return 200, {"skills": system.skills.public()}
                if path == "/mods":
                    return 200, {"mods": [m.public() for m in system.mods.all()]}
                if path == "/mcp":
                    return 200, {"servers": [c.status() for c in system.mcp.values()]}
                if path == "/processes":
                    return 200, {"processes": system.ps()}
                if path == "/approvals":
                    return 200, {"approvals": system.pending_approvals()}
                if m := re.fullmatch(r"/processes/(\w+)", path):
                    return 200, system.process(m.group(1)).to_dict()
                if m := re.fullmatch(r"/traces/(\w+)", path):
                    return 200, {"spans": system.tracer.trace(m.group(1))}
                raise HTTPError(404, f"No route for GET {path}")

            if method == "DELETE":
                if m := re.fullmatch(r"/providers/([\w-]+)", path):
                    if not system.providers.remove(m.group(1)):
                        raise HTTPError(404, f"No provider {m.group(1)!r}")
                    return 200, {"ok": True}
                if m := re.fullmatch(r"/agents/([\w-]+)/allowed", path):
                    system.reset_always_allow(m.group(1))
                    return 200, {"ok": True}
                if m := re.fullmatch(r"/agents/([\w-]+)", path):
                    system.agent(m.group(1))
                    system.kill(m.group(1))
                    return 200, {"ok": True}
                removers = {"skills": system.remove_skill, "mods": system.remove_mod, "mcp": system.remove_mcp}
                if (m := re.fullmatch(r"/(skills|mods|mcp)/([\w-]+)", path)) and removers[m.group(1)](m.group(2)):
                    return 200, {"ok": True}
                raise HTTPError(404, f"No route or item for DELETE {path}")

            if path == "/memory/import":
                query = urllib.parse.parse_qs(self.path.partition("?")[2])
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_UPLOAD:
                    raise HTTPError(413, "Plik jest za duży (max 60 MB)")
                raw = self.rfile.read(length) if length else b""
                filename = query.get("filename", [""])[0]
                distill = query.get("distill", ["0"])[0] in ("1", "true", "yes")
                if self.headers.get("Content-Type", "").startswith("application/json"):
                    try:
                        payload = json.loads(raw or b"{}")
                    except json.JSONDecodeError as exc:
                        raise HTTPError(400, f"Invalid JSON: {exc}") from exc
                    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
                        raw, filename = payload["text"].encode(), payload.get("filename") or "wklejone.txt"
                        distill = bool(payload.get("distill", distill))
                if not raw.strip():
                    raise HTTPError(400, "Pusty plik")
                return 200, system.import_memory(raw, filename, distill=distill)

            body = self._body()
            if path == "/skills":
                if isinstance(body.get("markdown"), str):
                    skill = parse_skill_md(body["markdown"], str(body.get("name") or ""))
                else:
                    tools = body.get("tools") or []
                    if isinstance(tools, str):
                        tools = [t.strip() for t in tools.split(",") if t.strip()]
                    skill = Skill(name=str(body.get("name", "")), description=str(body.get("description", "")),
                                  instructions=str(body.get("instructions", "")), tools=list(tools))
                return 201, system.add_skill(skill).__dict__
            if path == "/mods":
                return 201, system.add_mod(mod_from_dict(body)).public()
            if path == "/mcp":
                data = dict(body)
                if data.get("token"):
                    data.setdefault("headers", {})["Authorization"] = f"Bearer {data.pop('token')}"
                client = system.add_mcp(config_from_dict(data))
                return 201, client.status()
            if m := re.fullmatch(r"/mcp/([\w-]+)/reconnect", path):
                return 200, system.reconnect_mcp(m.group(1)).status()
            if path == "/providers":
                pid = _require(body, "id")
                cfg = ProviderConfig(
                    id=pid, kind=str(body.get("kind", "openai_compat")), api_key=str(body.get("api_key") or ""),
                    base_url=str(body.get("base_url") or ""), default_model=str(body.get("default_model") or ""),
                    label=str(body.get("label") or pid),
                )
                system.providers.upsert(cfg)
                if body.get("make_default"):
                    system.set_default_model(pid, cfg.default_model or None)
                return 201, cfg.public()
            if m := re.fullmatch(r"/providers/([\w-]+)/test", path):
                llm = system.providers.build(m.group(1), body.get("model") or None)
                reply = llm.complete([Message.user("Reply with exactly: OK")], max_tokens=200)
                return 200, {"ok": True, "reply": reply.text[:200], "model": reply.model or getattr(llm, "model", None)}
            if path == "/settings/default":
                system.set_default_model(_require(body, "provider"), body.get("model") or None)
                return 200, {"ok": True}
            if path == "/agents/external":
                name = _require(body, "name")
                if not _NAME.fullmatch(name):
                    raise HTTPError(400, "Agent name must match [a-zA-Z0-9_-]{1,40}")
                protocol = str(body.get("protocol", "openai"))
                if protocol not in PROTOCOLS:
                    raise HTTPError(400, f"protocol must be one of {PROTOCOLS}")
                agent = system.register_external(ExternalAgentConfig(
                    name=name, protocol=protocol, url=_require(body, "url"), api_key=str(body.get("api_key") or ""),
                    model=str(body.get("model") or ""), description=str(body.get("description") or ""),
                ))
                return 201, system.describe(agent.name)
            if path == "/chat":
                message = _require(body, "message")
                proc = system.chat(message, agent=str(body.get("agent") or "assistant"),
                                   session=str(body.get("session") or "default"), wait=not body.get("async", True))
                return (202 if body.get("async", True) else 200), proc.to_dict()
            if path == "/chat/reset":
                system.reset_chat(str(body.get("agent") or "assistant"), str(body.get("session") or "default"))
                return 200, {"ok": True}
            if path == "/crew":
                agents = body.get("agents")
                if not isinstance(agents, list) or not agents:
                    raise HTTPError(400, "'agents' must be a non-empty list")
                proc = system.submit_crew(_require(body, "task"), agents, str(body.get("process") or "manager"))
                return 202, proc.to_dict()
            if path == "/run":
                task = _require(body, "task")
                agent = body.get("agent", "assistant")
                context = body.get("context")
                if body.get("async"):
                    return 202, system.submit(task, agent=agent, context=context).to_dict()
                proc = system.run(task, agent=agent, context=context)
                return (200 if proc.status == "completed" else 500), proc.to_dict()
            if path == "/agents":
                name = _require(body, "name")
                if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", name):
                    raise HTTPError(400, "Agent name must match [a-zA-Z0-9_-]{1,40}")
                kwargs: dict[str, Any] = {
                    "tools": body.get("tools"),
                    "tool_tags": body.get("tool_tags"),
                    "conversational": bool(body.get("conversational", False)),
                }
                if body.get("instructions"):
                    kwargs["instructions"] = str(body["instructions"])
                if isinstance(body.get("skills"), list):
                    kwargs["skills"] = [str(s) for s in body["skills"]]
                if body.get("provider") and body["provider"] != "default":
                    kwargs["provider"] = str(body["provider"])
                    kwargs["model"] = str(body.get("model") or "") or None
                kwargs["mode"] = str(body.get("mode") or "standard")
                kwargs["permission"] = str(body.get("permission") or ("ask" if body.get("approval") else "auto"))
                if isinstance(body.get("delegates"), list):
                    kwargs["delegates"] = [str(d) for d in body["delegates"]]
                agent = system.spawn(name, **kwargs)
                return 201, system.describe(agent.name)
            if m := re.fullmatch(r"/agents/([\w-]+)/settings", path):
                agent = system.update_agent(m.group(1), permission=body.get("permission") or None,
                                            mode=body.get("mode") or None)
                return 200, system.describe(agent.name)
            if m := re.fullmatch(r"/approvals/(\w+)", path):
                answer, approve = body.get("answer"), body.get("approve")
                if answer is not None:
                    if not isinstance(answer, str):
                        raise HTTPError(400, "'answer' must be text")
                    system.decide_approval(m.group(1), answer=answer)
                elif isinstance(approve, bool):
                    system.decide_approval(m.group(1), approve, remember=bool(body.get("remember", False)))
                else:
                    raise HTTPError(400, "Send 'approve' (true/false) or 'answer'")
                return 200, {"ok": True}
            if m := re.fullmatch(r"/processes/(\w+)/feedback", path):
                reward = body.get("reward")
                if not isinstance(reward, (int, float)) or not -1 <= reward <= 1:
                    raise HTTPError(400, "'reward' must be a number in [-1, 1]")
                system.give_feedback(m.group(1), float(reward), str(body.get("comment", "")))
                return 200, {"ok": True}
            if path == "/knowledge":
                docs = body.get("documents") or ([body] if body.get("text") else [])
                if not docs:
                    raise HTTPError(400, "Provide 'text' or 'documents'")
                chunks = 0
                for d in docs:
                    if not isinstance(d, dict) or not isinstance(d.get("text"), str):
                        raise HTTPError(400, "Each document needs a 'text' string")
                    chunks += system.ingest(Document(d["text"], dict(d.get("metadata") or {})))
                return 201, {"documents": len(docs), "chunks": chunks}
            if path == "/goal":
                result = system.goal(_require(body, "goal"), agents=body.get("agents"))
                return 200, result.to_dict()
            raise HTTPError(404, f"No route for POST {path}")

    return Handler


def make_server(system: AgenticOS, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(system))


def serve(system: AgenticOS, host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover - blocking
    if host not in ("127.0.0.1", "localhost", "::1") and not system.settings.api_key:
        raise SystemExit("Refusing to serve on a public interface without AGENTIC_OS_API_KEY (the War Room password).")
    httpd = make_server(system, host, port)
    print(f"Agentic OS War Room on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        system.shutdown(wait=False)
