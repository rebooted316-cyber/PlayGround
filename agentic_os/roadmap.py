"""Where each stage of the "Roadmap to Learn Agentic AI" lives in this codebase."""

from __future__ import annotations

ROADMAP: list[dict] = [
    {"stage": 1, "title": "Introduction to Agentic AI", "modules": ["agentic_os.agents.base", "agentic_os.kernel"],
     "covers": "autonomous decision-making loop, agent vs. plain AI, core capabilities, real-world runtime"},
    {"stage": 2, "title": "Fundamentals of AI & ML", "modules": ["agentic_os.learning.ml", "agentic_os.memory.embeddings"],
     "covers": "supervised learning, softmax regression, gradient descent, regularization, feature hashing"},
    {"stage": 3, "title": "Programming & AI Frameworks", "modules": ["agentic_os.tools.registry", "agentic_os.tools.builtin"],
     "covers": "Python tools, function calling via JSON Schema, validation, orchestration"},
    {"stage": 4, "title": "Large Language Models", "modules": ["agentic_os.llm"],
     "covers": "provider abstraction, Claude via the Anthropic SDK, tokens, context windows"},
    {"stage": 5, "title": "Understanding AI Agents", "modules": ["agentic_os.agents"],
     "covers": "reactive vs. goal-oriented agents, ReAct loop, multi-agent workflows"},
    {"stage": 6, "title": "AI Memory & Knowledge Retrieval", "modules": ["agentic_os.memory"],
     "covers": "short-term buffer, long-term vector memory, vector store, chunking, semantic search"},
    {"stage": 7, "title": "Decision-Making & Planning", "modules": ["agentic_os.planning", "agentic_os.agents.goal", "agentic_os.agents.multi"],
     "covers": "goal decomposition, hierarchical plans, dependency ordering, multi-agent strategies"},
    {"stage": 8, "title": "Prompt Engineering & Adaptation", "modules": ["agentic_os.prompts"],
     "covers": "templates, zero/few-shot, chain-of-thought, role prompts, dynamic few-shot selection"},
    {"stage": 9, "title": "Reinforcement Learning & Self-Improvement", "modules": ["agentic_os.learning"],
     "covers": "human feedback, bandits (epsilon-greedy, UCB1), LLM judges, Reflexion-style retries"},
    {"stage": 10, "title": "Retrieval-Augmented Generation", "modules": ["agentic_os.rag"],
     "covers": "BM25 + vector hybrid search, reciprocal rank fusion, cited context, retrieval as a tool"},
    {"stage": 11, "title": "Deploying AI Agents", "modules": ["agentic_os.server", "agentic_os.observability", "Dockerfile"],
     "covers": "HTTP API, async processes, worker pool, auth, tracing, latency metrics, containers"},
    {"stage": 12, "title": "Real-World AI Agent Applications", "modules": ["agentic_os.apps", "examples/"],
     "covers": "research assistant, data processing, support automation, decision support"},
]


def render() -> str:
    lines = ["Agentic OS: Roadmap to Learn Agentic AI", ""]
    for item in ROADMAP:
        lines.append(f"{item['stage']:>2}. {item['title']}")
        lines.append(f"    modules: {', '.join(item['modules'])}")
        lines.append(f"    covers:  {item['covers']}")
    return "\n".join(lines)
