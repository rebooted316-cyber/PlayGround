"""The core tool-using agent.

Roadmap stages 1 and 5: an agent is a loop of *perceive -> decide -> act*.
Here the model decides (text or tool calls), tools act, results are fed
back as observations, and the loop ends when the model answers without
calling a tool (the ReAct pattern).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from agentic_os.llm.base import LLMProvider, Message, ToolCall
from agentic_os.memory.long_term import LongTermMemory
from agentic_os.memory.short_term import ConversationBuffer
from agentic_os.observability import Metrics, Tracer
from agentic_os.prompts import PromptStrategy, ZeroShot
from agentic_os.rag.pipeline import RAGPipeline, RetrievedChunk
from agentic_os.tools.registry import Tool, ToolRegistry, ToolResult

DEFAULT_INSTRUCTIONS = (
    "You are a capable, honest assistant running inside Agentic OS. Use the available tools when they help "
    "you get a correct answer, and do not guess facts a tool could check. When knowledge passages are provided, "
    "ground your answer in them and cite them as [n]. If you cannot complete the task, say what is missing."
)

ApprovalHook = Callable[[ToolCall], bool]


@dataclass
class AgentResult:
    """Everything that happened during one :meth:`Agent.run`."""

    output: str
    agent: str
    stopped_reason: str = "completed"
    steps: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    trace_id: str | None = None
    messages: list[Message] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "agent": self.agent,
            "stopped_reason": self.stopped_reason,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "usage": self.usage,
            "citations": self.citations,
            "trace_id": self.trace_id,
        }


class Agent:
    """A goal-directed, tool-using LLM agent.

    Args:
        name: Identifier used in traces and multi-agent setups.
        llm: Model provider.
        instructions: Base system prompt.
        tools: Tools the agent may call.
        memory: Short-term conversation memory (enables multi-turn chat).
        long_term: Long-term memory; relevant memories are recalled per task
            and each run is stored as an episode.
        knowledge: RAG pipeline; relevant passages are retrieved per task.
        strategy: Prompt strategy applied to ``instructions``.
        max_steps: Maximum model turns per run.
        approve_tool: Optional human-in-the-loop hook; return False to block a call.
    """

    def __init__(
        self,
        name: str = "assistant",
        llm: LLMProvider | None = None,
        *,
        instructions: str = DEFAULT_INSTRUCTIONS,
        tools: ToolRegistry | Iterable[Tool] | None = None,
        memory: ConversationBuffer | None = None,
        long_term: LongTermMemory | None = None,
        knowledge: RAGPipeline | None = None,
        retrieve_k: int = 4,
        strategy: PromptStrategy | None = None,
        max_steps: int = 10,
        max_tokens: int = 16000,
        remember_episodes: bool = True,
        approve_tool: ApprovalHook | None = None,
        tracer: Tracer | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        if llm is None:
            from agentic_os.llm import create_llm

            llm = create_llm()
        self.name = name
        self.llm = llm
        self.instructions = instructions
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools or [])
        self.memory = memory
        self.long_term = long_term
        self.knowledge = knowledge
        self.retrieve_k = retrieve_k
        self.strategy = strategy or ZeroShot()
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.remember_episodes = remember_episodes
        self.approve_tool = approve_tool
        self.tracer = tracer or Tracer()
        self.metrics = metrics or Metrics()

    # ------------------------------------------------------------ prompting
    def system_prompt(self, task: str) -> str:
        return self.strategy.apply(self.instructions, task)

    def build_user_message(self, task: str, context: str | None = None) -> tuple[Message, list[RetrievedChunk]]:
        """Augment the task with recalled memories, retrieved knowledge and extra context.

        Retrieved material goes into the user turn (not the system prompt) so
        the conversation stays append-only and cache-friendly.
        """
        sections: list[str] = []
        chunks: list[RetrievedChunk] = []
        if self.long_term is not None and len(self.long_term):
            memories = self.long_term.recall(task, k=3)
            if memories:
                body = "\n".join(f"- ({m.kind}) {m.text}" for m in memories)
                sections.append(f"<memories>\nPossibly relevant memories from earlier work:\n{body}\n</memories>")
        if self.knowledge is not None and len(self.knowledge):
            text, chunks = self.knowledge.build_context(task, k=self.retrieve_k)
            if text:
                sections.append(f"<knowledge>\n{text}\n</knowledge>")
        if context:
            sections.append(f"<context>\n{context}\n</context>")
        if not sections:
            return Message.user(task), chunks
        return Message.user("\n\n".join(sections) + f"\n\n<task>\n{task}\n</task>"), chunks

    # ------------------------------------------------------------- execution
    def _execute(self, call: ToolCall) -> ToolResult:
        if self.approve_tool is not None and not self.approve_tool(call):
            return ToolResult("This tool call was not approved by the operator.", is_error=True)
        if call.name not in self.tools:
            return ToolResult(f"Unknown tool: {call.name}. Available: {', '.join(self.tools.names())}", is_error=True)
        with self.tracer.span("tool.call", tool=call.name), self.metrics.timer(f"tool.{call.name}"):
            result = self.tools.execute(call.name, call.arguments)
        self.metrics.incr("tool.calls")
        if result.is_error:
            self.metrics.incr("tool.errors")
        return result

    def run(self, task: str, context: str | None = None) -> AgentResult:
        """Work on ``task`` until the model answers or ``max_steps`` is reached."""
        with self.tracer.span("agent.run", agent=self.name) as root, self.metrics.timer("agent.run"):
            self.metrics.incr("agent.runs")
            user_msg, chunks = self.build_user_message(task, context)
            history = self.memory.messages() if self.memory is not None else []
            turn: list[Message] = [user_msg]
            system = self.system_prompt(task)
            tool_list = list(self.tools) or None
            result = AgentResult(output="", agent=self.name, trace_id=root.trace_id)
            result.citations = [
                {"n": i + 1, "source": c.source, "chunk_id": c.id} for i, c in enumerate(chunks)
            ]
            usage = {"input_tokens": 0, "output_tokens": 0}
            last_text = ""

            for step in range(1, self.max_steps + 1):
                result.steps = step
                with self.tracer.span("llm.complete", step=step), self.metrics.timer("llm.complete"):
                    response = self.llm.complete(
                        history + turn, system=system, tools=tool_list, max_tokens=self.max_tokens
                    )
                for key in usage:
                    usage[key] += response.usage.get(key, 0)
                turn.append(response.to_message())
                last_text = response.text or last_text

                if response.stop_reason == "refusal":
                    result.stopped_reason = "refusal"
                    result.output = response.text or "The model declined this request."
                    break

                if response.tool_calls and response.stop_reason != "max_tokens":
                    for call in response.tool_calls:
                        outcome = self._execute(call)
                        result.tool_calls.append(
                            {"tool": call.name, "arguments": call.arguments, "result": outcome.content[:2000], "is_error": outcome.is_error}
                        )
                        turn.append(Message.tool(call.id, outcome.content, outcome.is_error))
                    continue

                if response.tool_calls:  # truncated mid tool call: close the calls so history stays valid
                    for call in response.tool_calls:
                        turn.append(Message.tool(call.id, "Not executed: the response was truncated.", True))
                if response.stop_reason == "max_tokens":
                    result.stopped_reason = "max_tokens"
                result.output = response.text
                break
            else:
                result.stopped_reason = "max_steps"
                result.output = last_text or "Stopped: reached the step limit before finishing."

            result.usage = usage
            result.messages = turn
            root.attributes.update(steps=result.steps, stopped=result.stopped_reason)
            self.metrics.incr("llm.input_tokens", usage["input_tokens"])
            self.metrics.incr("llm.output_tokens", usage["output_tokens"])

            if self.memory is not None:
                self.memory.add_turn(turn)
            if self.long_term is not None and self.remember_episodes and result.stopped_reason == "completed":
                self.long_term.remember(
                    f"Task: {task[:300]}\nOutcome: {result.output[:500]}", kind="episode", agent=self.name
                )
            return result

    def chat(self, message: str) -> str:
        """Multi-turn convenience: enables short-term memory on first use."""
        if self.memory is None:
            self.memory = ConversationBuffer()
        return self.run(message).output

    def as_tool(self, name: str | None = None, description: str | None = None) -> Tool:
        """Expose this agent as a tool, so other agents can delegate to it."""

        def delegate(task: str) -> str:
            return self.run(task).output

        return Tool(
            name=name or f"ask_{self.name}",
            description=description
            or f"Delegate a self-contained sub-task to the '{self.name}' agent and get its answer back.",
            parameters={
                "type": "object",
                "properties": {"task": {"type": "string", "description": "A complete, self-contained instruction."}},
                "required": ["task"],
            },
            func=delegate,
            tags={"agent"},
        )

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, tools={self.tools.names()})"
