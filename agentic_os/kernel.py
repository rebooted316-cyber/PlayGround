"""The Agentic OS kernel.

Ties every roadmap layer into one runtime, like an operating system for
agents:

* **Agents** are the programs: spawned from a template and registered by name.
* **Processes**: every run gets a PID, a status and a result, and runs
  synchronously or on a worker pool.
* **Shared services**: a tool registry (system calls), long-term memory, a
  knowledge base (RAG), feedback, tracing and metrics.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from agentic_os.agents.base import DEFAULT_INSTRUCTIONS, Agent
from agentic_os.agents.external import ExternalAgent, ExternalAgentConfig
from agentic_os.agents.goal import GoalAgent, GoalResult
from agentic_os.agents.multi import Crew
from agentic_os.config import Settings
from agentic_os.learning.bandit import EpsilonGreedy
from agentic_os.learning.feedback import FeedbackStore
from agentic_os.learning.reflection import LLMJudge, ReflectiveAgent
from agentic_os.llm import LLMError, LLMProvider, Message, create_llm
from agentic_os.llm.base import ToolCall
from agentic_os.mcp import MCPClient, MCPError, MCPServerConfig, config_from_dict
from agentic_os.memory.embeddings import cosine
from agentic_os.memory.importer import parse_export
from agentic_os.memory.long_term import LongTermMemory
from agentic_os.memory.short_term import ConversationBuffer
from agentic_os.mods import Mod, ModRegistry
from agentic_os.observability import Metrics, Tracer
from agentic_os.planning import PlanError, extract_json
from agentic_os.prompts import PromptStrategy, default_strategies
from agentic_os.providers import ProviderRegistry
from agentic_os.rag.pipeline import Document, RAGPipeline
from agentic_os.skills import Skill, SkillRegistry
from agentic_os.tools.builtin import builtin_tools
from agentic_os.tools.registry import Tool, ToolRegistry, make_tool

AGENT_MODES = ("standard", "reflective", "adaptive")
MAX_DELEGATION_DEPTH = 3  # stops agents that delegate to each other from looping forever
APPROVAL_TIMEOUT = 300.0  # seconds a tool call or question waits for the operator
FULL_ACCESS_MAX_STEPS = 25
# How much an agent may do without the operator:
#   ask   - every tool call waits for approval
#   smart - read-only tools run freely, anything that changes something needs approval,
#           and the agent can ask the user a question when it judges it necessary
#   auto  - works autonomously within its selected tools, never waits for the user
#   full  - every tool, MCP server, mod and agent in the system, no confirmations
PERMISSIONS = ("ask", "smart", "auto", "full")
PERMISSION_NOTES = {
    "ask": "\n\nThe operator approves each of your tool calls. If a call is denied, do not repeat it: "
           "try another way or explain what you would need.",
    "smart": "\n\nAct on your own for routine, reversible steps. When a choice is ambiguous, irreversible, costly "
             "or depends on the user's preferences, call ask_user first. Tools that change things outside this "
             "system wait for the operator's approval; if one is denied, do not repeat it.",
    "auto": "\n\nWork autonomously: the user will not answer questions during the task, so make reasonable "
            "decisions yourself and state your assumptions in the answer.",
    "full": "\n\nYou have full access to every tool and agent in this system and no confirmations are required. "
            "Work autonomously, state your assumptions, and avoid irreversible actions the task does not need.",
}


class AgenticOSError(RuntimeError):
    pass


@dataclass
class Process:
    pid: str
    agent: str
    task: str
    status: str = "queued"  # queued | running | waiting (for the operator) | completed | failed
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    strategy: str | None = None  # prompt strategy an adaptive agent picked for this run
    _future: Future | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "agent": self.agent,
            "task": self.task,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "strategy": self.strategy,
        }


@dataclass
class Approval:
    """Something waiting for the operator (human in the loop).

    ``kind="tool"``: a tool call to allow or deny. ``kind="question"``: the
    agent asked the user something (``ask_user``) and waits for an answer.
    """

    id: str
    pid: str | None
    agent: str
    tool: str
    arguments: dict[str, Any]
    kind: str = "tool"
    created_at: float = field(default_factory=time.time)
    decision: bool | None = None
    answer: str | None = None
    _event: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "pid": self.pid, "agent": self.agent, "tool": self.tool,
                "arguments": self.arguments, "created_at": self.created_at}


class _LayeredRegistry(ToolRegistry):
    """A live view of a shared registry plus agent-specific tools (e.g. delegation)."""

    def __init__(self, base: ToolRegistry, extra: Iterable[Tool]) -> None:
        self._base = base
        self._extra = {t.name: t for t in extra}

    @property
    def _tools(self) -> dict[str, Tool]:  # type: ignore[override]
        return {**self._base._tools, **self._extra}

    def register(self, t: Tool | Callable[..., Any]) -> Tool:
        if not isinstance(t, Tool):
            t = make_tool(t)
        self._extra[t.name] = t
        return t

    def unregister(self, name: str) -> None:
        self._extra.pop(name, None)


class _FullAccessRegistry(ToolRegistry):
    """Every tool in the system plus delegation to every other agent, always current."""

    def __init__(self, system: AgenticOS, owner: str) -> None:
        self._system = system
        self._owner = owner

    @property
    def _tools(self) -> dict[str, Tool]:  # type: ignore[override]
        tools = dict(self._system.tools._tools)
        for name in list(self._system.agents):
            if name != self._owner:
                tools[f"ask_{name}"] = self._system._delegate_tool(name)
        return tools

    def register(self, t: Tool | Callable[..., Any]) -> Tool:
        return self._system.tools.register(t)


class _ReadOnlyMemory:
    """Lets reflective attempts see the conversation without each one being saved to it."""

    def __init__(self, buffer: ConversationBuffer) -> None:
        self._buffer = buffer

    def messages(self) -> list[Message]:
        return self._buffer.messages()

    def add_turn(self, messages: list[Message]) -> None:
        pass


class _RunResult:
    """An agent result plus extra fields (reflection attempts, chosen strategy)."""

    def __init__(self, result: Any, **extra: Any) -> None:
        self.result = result
        self.extra = extra

    def to_dict(self) -> dict[str, Any]:
        return {**self.result.to_dict(), **self.extra}


class _ReflectiveRunner:
    """Runs an agent Reflexion-style: answer, grade, critique, retry; keeps the best attempt."""

    def __init__(self, agent: Agent, lessons: LongTermMemory | None, max_attempts: int = 3,
                 threshold: float = 0.8) -> None:
        self.agent = agent
        self.lessons = lessons
        self.max_attempts = max_attempts
        self.threshold = threshold

    def run(self, task: str, context: str | None = None) -> _RunResult:
        buffer = self.agent.memory
        worker = copy.copy(self.agent)
        worker.memory = _ReadOnlyMemory(buffer) if buffer is not None else None
        results: list[Any] = []

        class _Recorder:
            llm = worker.llm

            @staticmethod
            def run(task: str, context: str | None = None) -> Any:
                merged = "\n\n".join(c for c in (outer, context) if c) or None
                res = worker.run(task, context=merged)
                results.append(res)
                return res

        outer = context
        reflection = ReflectiveAgent(_Recorder(), LLMJudge(worker.llm), max_attempts=self.max_attempts,  # type: ignore[arg-type]
                                     threshold=self.threshold, lessons=self.lessons).run(task)
        best_i = max(range(len(reflection.attempts)), key=lambda i: reflection.attempts[i].score)
        best = results[best_i]
        if buffer is not None:
            buffer.add_turn(best.messages)
        usage = {k: sum(r.usage.get(k, 0) for r in results) for k in ("input_tokens", "output_tokens")}
        return _RunResult(best, usage=usage, reflection={
            "best_score": round(reflection.best_score, 3),
            "attempts": [{"score": round(a.score, 3), "critique": a.critique} for a in reflection.attempts],
        })


class AgenticOS:
    """Create one instance per application; it owns all agents and services."""

    def __init__(self, settings: Settings | None = None, llm: LLMProvider | None = None, max_workers: int = 4) -> None:
        self.settings = settings or Settings()
        self.llm = llm or create_llm(self.settings)
        self.tracer = Tracer()
        self.metrics = Metrics()
        data_dir = self.settings.data_dir
        self.memory = LongTermMemory(path=os.path.join(data_dir, "memory.json") if data_dir else None)
        self.feedback = FeedbackStore(os.path.join(data_dir, "feedback.jsonl") if data_dir else None)
        self.knowledge = RAGPipeline()
        self.tools = ToolRegistry(builtin_tools())
        self.tools.register(self.knowledge.as_tool())
        for t in self.memory.as_tools():
            self.tools.register(t)
        self.providers = ProviderRegistry(self._path("providers.json"))
        self.skills = SkillRegistry(self._path("skills.json"))
        self.tools.register(self.skills.loader_tool())
        self.mods = ModRegistry(self._path("mods.json"))
        for mod in self.mods.all():
            if mod.enabled:
                self.tools.register(mod.to_tool())
        self.mcp: dict[str, MCPClient] = {}
        self.agents: dict[str, Agent | ExternalAgent] = {}
        self.processes: dict[str, Process] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agentic-os")
        self._specs: dict[str, dict[str, Any]] = {}
        self._sessions: dict[tuple[str, str], ConversationBuffer] = {}
        self._restoring = False
        self._local = threading.local()  # current pid and delegation depth of the running thread
        self.approvals: dict[str, Approval] = {}
        self.strategies = default_strategies()
        self._bandits: dict[str, EpsilonGreedy] = {}
        self._bandit_state: dict[str, Any] = self._load_json("bandits.json", {})
        self._rewards: dict[str, float] = {}  # pid -> reward already credited to a bandit
        self.prefs: dict[str, Any] = self._load_json("settings.json", {})
        if llm is None and self.prefs.get("default_provider"):
            try:
                self.llm = self.providers.build(self.prefs["default_provider"], self.prefs.get("default_model"))
            except LLMError:
                pass  # keep the environment-configured model if the stored one is unusable
        saved_specs = self._load_json("agents.json", [])  # read before spawning overwrites it
        self._restoring = True
        self.spawn("assistant")
        self._restore(saved_specs)
        saved_mcp = self._load_json("mcp.json", [])
        if saved_mcp:  # reconnect in the background so a slow server never blocks startup
            threading.Thread(target=self._reconnect_mcp, args=(saved_mcp,), daemon=True).start()

    # --------------------------------------------------------- persistence
    def _path(self, name: str) -> str | None:
        return os.path.join(self.settings.data_dir, name) if self.settings.data_dir else None

    def _load_json(self, name: str, default: Any) -> Any:
        path = self._path(name)
        if not path or not os.path.exists(path):
            return default
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def _save_json(self, name: str, data: Any) -> None:
        path = self._path(name)
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # write a private temp file and swap it in, so a concurrent reader never sees a half-written file
        tmp = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # may hold API keys
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def _restore(self, saved_specs: list[dict[str, Any]]) -> None:
        """Re-create saved agents and re-ingest saved knowledge after a restart."""
        self._restoring = True
        try:
            for spec in saved_specs:
                spec = dict(spec)
                try:
                    if spec.pop("type", "native") == "external":
                        self.register_external(ExternalAgentConfig(**spec))
                    else:
                        if spec.pop("approval", False):  # older specs stored a yes/no approval flag
                            spec.setdefault("permission", "ask")
                        self.spawn(spec.pop("name"), **spec)
                except (LLMError, ValueError, TypeError):
                    continue  # skip a spec that no longer works rather than failing startup
            path = self._path("knowledge.jsonl")
            if path and os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            item = json.loads(line)
                            self.knowledge.add_document(Document(item["text"], item.get("metadata", {}), id=item["id"]))
        finally:
            self._restoring = False

    def _reconnect_mcp(self, saved: list[dict[str, Any]]) -> None:
        for item in saved:
            try:
                cfg = config_from_dict(item)
            except ValueError:
                continue
            client = MCPClient(cfg)
            self.mcp[cfg.name] = client
            try:
                client.connect()
                for t in client.as_tools():
                    self.tools.register(t)
            except MCPError:
                pass  # status shows the error; the user can retry from the UI

    def _save_mcp(self) -> None:
        self._save_json("mcp.json", [c.config.__dict__ for c in self.mcp.values()])

    def _save_specs(self) -> None:
        if not self._restoring:
            self._save_json("agents.json", list(self._specs.values()))

    # --------------------------------------------------------------- agents
    def spawn(
        self,
        name: str,
        instructions: str = DEFAULT_INSTRUCTIONS,
        *,
        tools: Iterable[str] | None = None,
        tool_tags: Iterable[str] | None = None,
        strategy: PromptStrategy | None = None,
        use_knowledge: bool = True,
        use_memory: bool = True,
        conversational: bool = False,
        max_steps: int | None = None,
        llm: LLMProvider | None = None,
        provider: str | None = None,
        model: str | None = None,
        skills: Iterable[str] | None = None,
        mode: str = "standard",
        permission: str = "auto",
        delegates: Iterable[str] | None = None,
        always_allow: Iterable[str] | None = None,
    ) -> Agent:
        """Create and register an agent.

        ``tools=None`` and ``tool_tags=None`` give the agent the shared
        registry (it sees tools registered later too); otherwise it gets a
        fixed subset. ``provider``/``model`` pick a configured provider (see
        :attr:`providers`); otherwise the OS default model is used.

        ``mode`` is ``standard``, ``reflective`` (grade, critique and retry,
        storing lessons) or ``adaptive`` (a bandit learns which prompt
        strategy earns the best feedback). ``permission`` is one of
        :data:`PERMISSIONS` (see :meth:`decide_approval` for answering).
        ``delegates`` names agents this one may hand sub-tasks to;
        ``always_allow`` lists tools the operator approved permanently.
        """
        if mode not in AGENT_MODES:
            raise ValueError(f"mode must be one of {AGENT_MODES}")
        if permission not in PERMISSIONS:
            raise ValueError(f"permission must be one of {PERMISSIONS}")
        delegate_names = [d for d in dict.fromkeys(delegates or []) if d != name and _valid_name(d)]
        if provider:
            llm = self.providers.build(provider, model)
        skill_names = [s for s in (skills or []) if self.skills.get(s)]
        if tools is None and tool_tags is None:
            registry = self.tools
        else:
            needed = set(tools or []) | {"load_skill"} if skill_names else set(tools or [])
            for s in skill_names:
                needed |= set(self.skills.get(s).tools)
            registry = self.tools.subset(needed, tool_tags)
        extra = [self._delegate_tool(d) for d in delegate_names]
        if permission in ("ask", "smart"):
            extra.append(self._ask_user_tool(name))
        if permission == "full":
            registry = _FullAccessRegistry(self, name)
        elif extra:
            registry = _LayeredRegistry(registry, extra)
        steps = max_steps or self.settings.max_steps
        if permission == "full":
            steps = max(steps, FULL_ACCESS_MAX_STEPS)
        agent = Agent(
            name,
            llm or self.llm,
            instructions=instructions + self.skills.catalog(skill_names) + PERMISSION_NOTES[permission],
            tools=registry,
            memory=ConversationBuffer() if conversational else None,
            long_term=self.memory if use_memory else None,
            knowledge=self.knowledge if use_knowledge else None,
            strategy=strategy,
            max_steps=steps,
            tracer=self.tracer,
            metrics=self.metrics,
            approve_tool=self._approval_hook(name),  # reads the agent's current permission on every call
        )
        agent.provider_id = provider
        agent.model_name = model or getattr(agent.llm, "model", None)
        agent.skill_names = skill_names
        agent.mode, agent.permission, agent.delegates = mode, permission, delegate_names
        agent.always_allow = {t for t in (always_allow or []) if isinstance(t, str)}
        if mode == "adaptive":
            bandit = EpsilonGreedy(list(self.strategies), epsilon=0.15, seed=None)
            bandit.load_dict(self._bandit_state.get(name, {}))
            self._bandits[name] = bandit
        else:
            self._bandits.pop(name, None)
        self.agents[name] = agent
        if strategy is None:  # prompt strategies are code objects and are not persisted
            self._specs[name] = {
                "type": "native", "name": name, "instructions": instructions,
                "tools": list(tools) if tools is not None else None,
                "tool_tags": list(tool_tags) if tool_tags is not None else None,
                "use_knowledge": use_knowledge, "use_memory": use_memory, "conversational": conversational,
                "max_steps": max_steps, "provider": provider, "model": model, "skills": skill_names,
                "mode": mode, "permission": permission, "delegates": delegate_names,
                "always_allow": sorted(agent.always_allow),
            }
            self._save_specs()
        return agent

    def register_external(self, config: ExternalAgentConfig) -> ExternalAgent:
        """Plug in an agent that runs elsewhere (OpenAI-compatible, A2A or webhook)."""
        if not config.name or not all(c.isalnum() or c in "_-" for c in config.name):
            raise ValueError("Agent name must match [a-zA-Z0-9_-]")
        agent = ExternalAgent(config)
        self.agents[config.name] = agent
        self._specs[config.name] = {"type": "external", **config.__dict__}
        self._save_specs()
        return agent

    def set_default_model(self, provider: str, model: str | None = None) -> None:
        """Make a configured provider the OS default (used by agents without their own provider)."""
        llm = self.providers.build(provider, model)
        self.llm = llm
        for agent in self.agents.values():
            if isinstance(agent, Agent) and not getattr(agent, "provider_id", None):
                agent.llm = llm
                agent.model_name = getattr(llm, "model", None)
        self.prefs.update(default_provider=provider, default_model=model)
        self._save_json("settings.json", self.prefs)

    def describe(self, name: str) -> dict[str, Any]:
        agent = self.agent(name)
        if isinstance(agent, ExternalAgent):
            return {"name": name, "kind": "external", "protocol": agent.config.protocol, "url": agent.config.url,
                    "model": agent.config.model, "description": agent.instructions, "tools": []}
        return {
            "name": name, "kind": "native", "provider": getattr(agent, "provider_id", None) or "default",
            "model": getattr(agent, "model_name", None) or getattr(agent.llm, "model", None),
            "tools": agent.tools.names(), "conversational": agent.memory is not None,
            "skills": getattr(agent, "skill_names", []),
            "mode": getattr(agent, "mode", "standard"), "permission": getattr(agent, "permission", "auto"),
            "always_allow": sorted(getattr(agent, "always_allow", set())),
            "delegates": getattr(agent, "delegates", []), "learning": self._learning(name),
            "description": self._specs.get(name, {}).get("instructions", agent.instructions)[:300],
        }

    def _learning(self, name: str) -> dict[str, Any] | None:
        bandit = self._bandits.get(name)
        if bandit is None:
            return None
        rated = any(st.pulls for st in bandit.stats.values())
        return {"strategies": bandit.summary(), "best": bandit.best() if rated else None}

    def _save_bandits(self) -> None:
        self._bandit_state.update({n: b.to_dict() for n, b in self._bandits.items()})
        self._save_json("bandits.json", self._bandit_state)

    # ----------------------------------------------------- human in the loop
    def _wait_for_operator(self, agent_name: str, tool: str, arguments: dict[str, Any], kind: str) -> Approval:
        """Park the running process until the operator answers (or the timeout passes)."""
        pid = getattr(self._local, "pid", None)
        item = Approval(uuid.uuid4().hex[:10], pid, agent_name, tool, arguments, kind=kind)
        proc = self.processes.get(pid) if pid else None
        with self._lock:
            self.approvals[item.id] = item
        if proc is not None:
            proc.status = "waiting"
        try:
            item._event.wait(APPROVAL_TIMEOUT)
        finally:
            with self._lock:
                self.approvals.pop(item.id, None)
            if proc is not None and proc.status == "waiting":
                proc.status = "running"
        return item

    @staticmethod
    def is_read_only(tool: Tool | None) -> bool:
        """Tools that only read (or delegate to an agent with its own permissions) are safe to run unasked."""
        if tool is None:
            return True  # an unknown tool fails on its own without side effects
        return "readonly" in tool.tags or ("agent" in tool.tags and "external" not in tool.tags)

    def _approval_hook(self, agent_name: str) -> Callable[[ToolCall], bool]:
        def hook(call: ToolCall) -> bool:
            agent = self.agents.get(agent_name)
            # the permission is looked up live, so a change applies even to a task that is already running
            permission = getattr(agent, "permission", "auto")
            if permission in ("auto", "full"):
                return True
            if call.name == "ask_user" or call.name in getattr(agent, "always_allow", ()):
                return True
            if permission == "smart" and agent is not None:
                tool = agent.tools._tools.get(call.name)
                if self.is_read_only(tool):
                    return True
            item = self._wait_for_operator(agent_name, call.name, dict(call.arguments or {}), "tool")
            return bool(item.decision)

        return hook

    def _ask_user_tool(self, agent_name: str) -> Tool:
        def ask_user(question: str) -> str:
            item = self._wait_for_operator(agent_name, "ask_user", {"question": question}, "question")
            if not item.answer:
                return ("The user did not answer. Make the most reasonable decision yourself and state the "
                        "assumption in your answer.")
            return f"The user answered: {item.answer}"

        return Tool(
            name="ask_user",
            description=("Ask the user a short question and wait for the answer. Use it only when a decision is "
                         "ambiguous, irreversible, costly or depends on the user's preferences."),
            parameters={
                "type": "object",
                "properties": {"question": {"type": "string", "description": "One clear question, with options if useful."}},
                "required": ["question"],
            },
            func=ask_user,
            tags={"human"},
        )

    def pending_approvals(self) -> list[dict[str, Any]]:
        with self._lock:
            return [a.public() for a in sorted(self.approvals.values(), key=lambda a: a.created_at)]

    def decide_approval(self, approval_id: str, approve: bool = True, *, remember: bool = False,
                        answer: str | None = None) -> None:
        """Answer the operator queue: allow/deny a tool call or answer an agent's question.

        ``remember=True`` with ``approve=True`` adds the tool to the agent's
        ``always_allow`` list so it is not asked again.
        """
        with self._lock:
            item = self.approvals.get(approval_id)
        if item is None:
            raise AgenticOSError(f"No pending approval {approval_id!r} (it may have timed out)")
        if item.kind == "question":
            item.answer = (answer or "").strip()[:4000]
            item.decision = True
        else:
            item.decision = bool(approve)
            agent = self.agents.get(item.agent)
            if approve and remember and isinstance(agent, Agent):
                agent.always_allow.add(item.tool)
                if item.agent in self._specs:
                    self._specs[item.agent]["always_allow"] = sorted(agent.always_allow)
                    self._save_specs()
        item._event.set()

    def update_agent(self, name: str, *, permission: str | None = None, mode: str | None = None) -> Agent:
        """Change an agent's permission level and/or thinking mode at any time.

        The agent is rebuilt from its saved spec, so tools, delegation and
        instructions follow the new setting; chat sessions, learned strategy
        stats and "always allow" approvals are kept. Tool calls already
        waiting for approval are allowed when switching to ``auto``/``full``.
        """
        agent = self.agent(name)
        if not isinstance(agent, Agent):
            raise ValueError("External agents have no permission or mode settings")  # noqa: TRY004 - maps to HTTP 400
        spec = self._specs.get(name)
        if spec is None:
            raise ValueError(f"Agent {name!r} was created in code with a custom strategy and cannot be changed here")
        spec = {k: v for k, v in spec.items() if k not in ("type", "name")}
        if permission is not None:
            spec["permission"] = permission
        if mode is not None:
            spec["mode"] = mode
        spec["always_allow"] = sorted(getattr(agent, "always_allow", set()))
        if name in self._bandits:
            self._save_bandits()
        updated = self.spawn(name, **spec)
        if updated.permission in ("auto", "full"):
            for item in list(self.approvals.values()):
                if item.agent == name and item.kind == "tool":
                    item.decision = True
                    item._event.set()
        return updated

    def reset_always_allow(self, name: str) -> None:
        """Forget the tools the operator approved permanently for an agent."""
        agent = self.agent(name)
        if isinstance(agent, Agent):
            agent.always_allow = set()
            if name in self._specs:
                self._specs[name]["always_allow"] = []
                self._save_specs()

    # ------------------------------------------------------------ delegation
    def _delegate_tool(self, target: str) -> Tool:
        """A tool that hands a sub-task to another agent, resolved when called."""

        def delegate(task: str) -> str:
            depth = getattr(self._local, "depth", 0)
            if depth >= MAX_DELEGATION_DEPTH:
                return "Delegation limit reached; finish this part yourself."
            other = self.agents.get(target)
            if other is None:
                return f"There is no agent named {target!r} any more."
            self._local.depth = depth + 1
            try:
                return other.run(task).output
            finally:
                self._local.depth = depth

        return Tool(
            name=f"ask_{target}",
            description=f"Delegate a self-contained sub-task to the '{target}' agent and get its answer back.",
            parameters={
                "type": "object",
                "properties": {"task": {"type": "string", "description": "A complete, self-contained instruction."}},
                "required": ["task"],
            },
            func=delegate,
            tags={"agent"},
        )

    # ------------------------------------------------------- skills & mods
    def add_skill(self, skill: Skill) -> Skill:
        return self.skills.add(skill)

    def remove_skill(self, name: str) -> bool:
        return self.skills.remove(name)

    def add_mod(self, mod: Mod) -> Mod:
        mod = self.mods.add(mod)
        self.tools.unregister(mod.tool_name)
        if mod.enabled:
            self.tools.register(mod.to_tool())
        return mod

    def remove_mod(self, name: str) -> bool:
        mod = self.mods.remove(name)
        if mod is not None:
            self.tools.unregister(mod.tool_name)
        return mod is not None

    # ------------------------------------------------------------------ MCP
    def add_mcp(self, config: MCPServerConfig) -> MCPClient:
        """Connect an MCP server and expose its tools to agents (raises MCPError if it cannot connect)."""
        if config.name in self.mcp:
            self.remove_mcp(config.name)
        client = MCPClient(config).connect()
        self.mcp[config.name] = client
        for t in client.as_tools():
            self.tools.register(t)
        self._save_mcp()
        return client

    def remove_mcp(self, name: str) -> bool:
        client = self.mcp.pop(name, None)
        if client is None:
            return False
        for t in list(self.tools):
            if f"mcp:{name}" in t.tags:
                self.tools.unregister(t.name)
        client.close()
        self._save_mcp()
        return True

    def reconnect_mcp(self, name: str) -> MCPClient:
        client = self.mcp.get(name)
        if client is None:
            raise AgenticOSError(f"No MCP server {name!r}")
        return self.add_mcp(client.config)

    # ------------------------------------------------------- memory import
    def import_memory(self, data: bytes, filename: str = "", distill: bool = False,
                      max_conversations: int = 500, max_distill: int = 30) -> dict[str, Any]:
        """Import memories exported from another assistant (ChatGPT, Claude, Gemini, Grok, ...)."""
        parsed = parse_export(data, filename)
        facts_added = episodes_added = distilled = 0
        for fact in parsed.facts[:5000]:
            self.memory.remember(fact, kind="fact", persist=False, source=parsed.source, imported=True)
            facts_added += 1
        convs = [c for c in parsed.conversations if c.user_text()][:max_conversations]
        for conv in convs:
            self.memory.remember(f"[{parsed.source}] {conv.title}: {conv.user_text()}", kind="episode",
                                 persist=False, source=parsed.source, imported=True)
            episodes_added += 1
        if distill and convs:
            for conv in convs[:max_distill]:
                transcript = "\n".join(f"{r}: {t[:800]}" for r, t in conv.messages[:30])
                try:
                    reply = self.llm.complete(
                        [Message.user(f"Conversation '{conv.title}':\n{transcript}")],
                        system=("Extract durable facts about the USER from this conversation: preferences, "
                                "goals, projects, background, constraints. Skip anything temporary or about the "
                                'assistant. Reply with JSON only: {"facts": ["...", "..."]} (empty list if none).'),
                        max_tokens=1500,
                    )
                    facts = extract_json(reply.text).get("facts") or []
                except (LLMError, PlanError, ValueError):
                    continue
                for fact in facts[:10]:
                    if isinstance(fact, str) and fact.strip():
                        self.memory.remember(fact.strip()[:500], kind="fact", persist=False,
                                             source=f"{parsed.source} (wyciągnięte)", imported=True)
                        distilled += 1
        self.memory.save()
        return {"source": parsed.source, "conversations_found": len(parsed.conversations),
                "facts_added": facts_added, "episodes_added": episodes_added, "distilled_facts": distilled,
                "memories_total": len(self.memory)}

    def agent(self, name: str) -> Agent:
        try:
            return self.agents[name]
        except KeyError:
            raise AgenticOSError(f"No agent named {name!r}. Running: {sorted(self.agents)}") from None

    def kill(self, name: str) -> None:
        if name == "assistant":
            raise AgenticOSError("The default 'assistant' agent cannot be removed")
        self.agents.pop(name, None)
        self._bandits.pop(name, None)
        if self._specs.pop(name, None) is not None:
            self._save_specs()
        for key in [k for k in self._sessions if k[0] == name]:
            del self._sessions[key]

    # ---------------------------------------------------------------- tools
    def tool(self, func: Callable[..., Any] | None = None, **kwargs: Any):
        """Register a function as a system-wide tool (usable as a decorator)."""

        def wrap(f: Callable[..., Any]) -> Tool:
            return self.tools.register(f if isinstance(f, Tool) else make_tool(f, **kwargs))

        return wrap(func) if func is not None else wrap

    # ------------------------------------------------------------ knowledge
    def ingest(self, content: str | Document, **metadata: Any) -> int:
        """Add a document (text, ``Document`` or file path) to the knowledge base. Returns chunk count."""
        if isinstance(content, Document):
            doc = content
        elif len(content) < 4096 and "\n" not in content and os.path.isfile(content):
            with open(content, encoding="utf-8") as fh:
                doc = Document(fh.read(), {"source": os.path.basename(content), **metadata})
        else:
            doc = Document(content, metadata)
        chunks = len(self.knowledge.add_document(doc))
        path = self._path("knowledge.jsonl")
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"id": doc.id, "text": doc.text, "metadata": doc.metadata}) + "\n")
        return chunks

    # ------------------------------------------------------------ processes
    def _new_process(self, agent: str, task: str) -> Process:
        self.agent(agent)
        proc = Process(pid=uuid.uuid4().hex[:10], agent=agent, task=task)
        with self._lock:
            self.processes[proc.pid] = proc
        return proc

    def _with_mode(self, proc: Process, runner: Any) -> Any:
        """Apply the agent's thinking mode (reflective / adaptive) for this run."""
        if not isinstance(runner, Agent):
            return runner
        mode = getattr(runner, "mode", "standard")
        bandit = self._bandits.get(proc.agent)
        if mode == "adaptive" and bandit is not None:
            proc.strategy = bandit.select()
            runner = copy.copy(runner)
            runner.strategy = self.strategies[proc.strategy]
        elif mode == "reflective":
            return _ReflectiveRunner(runner, self.memory if runner.long_term is not None else None)
        return runner

    def _execute(self, proc: Process, context: str | None, runner: Any = None) -> Process:
        proc.status, proc.started_at = "running", time.time()
        self._local.pid = proc.pid
        try:
            result = self._with_mode(proc, runner or self.agents[proc.agent]).run(proc.task, context=context)
            proc.result = result.to_dict()
            if proc.strategy:
                proc.result["strategy"] = proc.strategy
            proc.status = "completed"
            self.metrics.incr("os.processes.completed")
        except Exception as exc:  # noqa: BLE001 - recorded on the process
            proc.status, proc.error = "failed", f"{type(exc).__name__}: {exc}"
            self.metrics.incr("os.processes.failed")
        finally:
            proc.finished_at = time.time()
            self._local.pid = None
        return proc

    def run(self, task: str, agent: str = "assistant", context: str | None = None) -> Process:
        """Run a task to completion in the calling thread."""
        return self._execute(self._new_process(agent, task), context)

    def submit(self, task: str, agent: str = "assistant", context: str | None = None) -> Process:
        """Queue a task on the worker pool; poll :meth:`ps` or call :meth:`wait`."""
        proc = self._new_process(agent, task)
        proc._future = self._executor.submit(self._execute, proc, context)
        return proc

    def chat(self, message: str, agent: str = "assistant", session: str = "default", wait: bool = True) -> Process:
        """Multi-turn chat: each (agent, session) pair keeps its own conversation memory."""
        target = self.agent(agent)
        proc = self._new_process(agent, message)
        runner = target
        if isinstance(target, Agent):
            buffer = self._sessions.setdefault((agent, session), ConversationBuffer())
            runner = copy.copy(target)  # same tools/memory/knowledge, per-session conversation
            runner.memory = buffer
        if wait:
            return self._execute(proc, None, runner)
        proc._future = self._executor.submit(self._execute, proc, None, runner)
        return proc

    def reset_chat(self, agent: str = "assistant", session: str = "default") -> None:
        self._sessions.pop((agent, session), None)

    def wait(self, pid: str, timeout: float | None = None) -> Process:
        proc = self.process(pid)
        if proc._future is not None:
            proc._future.result(timeout=timeout)
        return proc

    def process(self, pid: str) -> Process:
        try:
            return self.processes[pid]
        except KeyError:
            raise AgenticOSError(f"No process {pid!r}") from None

    def ps(self) -> list[dict[str, Any]]:
        return [
            {k: v for k, v in p.to_dict().items() if k != "result"}
            for p in sorted(self.processes.values(), key=lambda p: p.created_at)
        ]

    def give_feedback(self, pid: str, reward: float, comment: str = "") -> None:
        """Record a reward in [-1, 1] for a finished process (human feedback)."""
        proc = self.process(pid)
        self.feedback.add(pid, reward, arm=proc.agent, comment=comment, strategy=proc.strategy)
        bandit = self._bandits.get(proc.agent)
        if bandit is not None and proc.strategy in bandit.stats:
            if pid in self._rewards:  # changed their mind: replace the earlier reward
                bandit.stats[proc.strategy].total_reward += reward - self._rewards[pid]
            else:
                bandit.update(proc.strategy, reward)
            self._rewards[pid] = reward
            self._save_bandits()
        if reward >= 0.5 and proc.result:
            self.memory.remember(
                f"A well-rated answer to '{proc.task[:200]}': {proc.result['output'][:400]}",
                kind="lesson",
                agent=proc.agent,
            )

    # ------------------------------------------------- higher-level runtimes
    def goal(self, goal: str, agents: Iterable[str] | None = None) -> GoalResult:
        names = list(agents) if agents else ["assistant"]
        return GoalAgent(self.llm, {n: self.agent(n) for n in names}).run(goal)

    def crew(self, agents: Iterable[str], process: str = "manager", router: Any = None) -> Crew:
        return Crew({n: self.agent(n) for n in agents}, self.llm, process=process, router=router)

    def submit_crew(self, task: str, agents: Iterable[str], process: str = "manager") -> Process:
        """Run a crew on the worker pool as a process (shows up in :meth:`ps`)."""
        crew = self.crew(agents, process=process)

        class _CrewRunner:
            def run(self, task: str, context: str | None = None):
                return crew.run(f"{context}\n\n{task}" if context else task)

        proc = Process(pid=uuid.uuid4().hex[:10], agent=f"crew:{process}", task=task)
        with self._lock:
            self.processes[proc.pid] = proc
        proc._future = self._executor.submit(self._execute, proc, None, _CrewRunner())
        return proc

    # ---------------------------------------------------------------- brain
    def graph(self, limit: int = 150, threshold: float = 0.3, per_node: int = 3) -> dict[str, Any]:
        """Knowledge graph of documents, chunks and memories linked by semantic similarity."""
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        vectors: list[tuple[str, list[float]]] = []
        docs: dict[str, str] = {}
        for rec in self.knowledge.vectors.all()[:limit]:
            doc_id = str(rec.metadata.get("doc_id", ""))
            if doc_id and doc_id not in docs:
                docs[doc_id] = str(rec.metadata.get("source") or rec.metadata.get("title") or f"dokument {doc_id[:6]}")
                nodes.append({"id": f"doc:{doc_id}", "label": docs[doc_id], "kind": "document", "text": ""})
            nodes.append({"id": rec.id, "label": rec.text[:40], "kind": "chunk", "text": rec.text[:500]})
            if doc_id:
                edges.append({"source": f"doc:{doc_id}", "target": rec.id, "weight": 1.0, "kind": "part"})
            vectors.append((rec.id, rec.vector))
        for rec in self.memory.store.all()[: max(0, limit - len(vectors))]:
            kind = rec.metadata.get("kind", "fact")
            nodes.append({"id": rec.id, "label": rec.text[:40], "kind": kind, "text": rec.text[:500]})
            vectors.append((rec.id, rec.vector))
        for i, (a_id, a_vec) in enumerate(vectors):
            scored = sorted(
                ((cosine(a_vec, b_vec), b_id) for j, (b_id, b_vec) in enumerate(vectors) if j > i),
                reverse=True,
            )
            for score, b_id in scored[:per_node]:
                if score >= threshold:
                    edges.append({"source": a_id, "target": b_id, "weight": round(score, 3), "kind": "similar"})
        return {"nodes": nodes, "edges": edges}

    # --------------------------------------------------------------- status
    def status(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for p in self.processes.values():
            counts[p.status] = counts.get(p.status, 0) + 1
        return {
            "provider": getattr(self.llm, "name", type(self.llm).__name__),
            "model": getattr(self.llm, "model", None),
            "agents": sorted(self.agents),
            "default_provider": self.prefs.get("default_provider") or "env",
            "tools": self.tools.names(),
            "knowledge_chunks": len(self.knowledge),
            "memories": len(self.memory),
            "processes": counts,
            "feedback": len(self.feedback),
            "approvals": len(self.approvals),
        }

    def shutdown(self, wait: bool = True) -> None:
        for client in self.mcp.values():
            client.close()
        self._executor.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


def _valid_name(name: str) -> bool:
    return 0 < len(name) <= 40 and all(c.isalnum() or c in "_-" for c in name)
