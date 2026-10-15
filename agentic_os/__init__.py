"""Agentic OS: an operating system for AI agents, built stage by stage
from the "Roadmap to Learn Agentic AI"."""

from agentic_os.agents import Agent, AgentResult, Crew, GoalAgent, ReflexAgent, Rule
from agentic_os.config import Settings
from agentic_os.kernel import AgenticOS, Process
from agentic_os.llm import HeuristicLLM, LLMProvider, LLMResponse, Message, ScriptedLLM, ToolCall, create_llm
from agentic_os.memory import ConversationBuffer, LongTermMemory, VectorStore
from agentic_os.rag import Document, RAGPipeline
from agentic_os.tools import Tool, ToolRegistry, tool

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentResult",
    "AgenticOS",
    "ConversationBuffer",
    "Crew",
    "Document",
    "GoalAgent",
    "HeuristicLLM",
    "LLMProvider",
    "LLMResponse",
    "LongTermMemory",
    "Message",
    "Process",
    "RAGPipeline",
    "ReflexAgent",
    "Rule",
    "ScriptedLLM",
    "Settings",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "VectorStore",
    "__version__",
    "create_llm",
    "tool",
]
