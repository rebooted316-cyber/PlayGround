# Agentic OS

An operating system for AI agents, built stage by stage from the
**"Roadmap to Learn Agentic AI"**. Each of the roadmap's 12 stages is a
working, tested module. Read the code to learn a stage, or use the whole
thing as a small, dependency-free agent framework running on Claude.

```text
                         ┌────────────────────── AgenticOS kernel ───────────────────────┐
  CLI / HTTP API  ──►    │  agents (programs)   processes (PIDs, status, results, pool)  │
  (stage 11)             │                                                               │
                         │  Agent loop: perceive ─► decide (LLM) ─► act (tools) ─► observe│
                         │     │            │               │                           │
                         │  prompts (8)   llm (4)        tools (3)                       │
                         │  memory (6) ◄──┴── rag (10) ──┴── planning (7) / crews (5)     │
                         │  learning: ML (2), bandits + feedback + reflection (9)        │
                         │  observability: traces + metrics (11)                         │
                         └───────────────────────────────────────────────────────────────┘
```

## Quick start

```bash
pip install -e ".[anthropic,gemini]"   # core has zero dependencies; extras add Claude / Gemini
export ANTHROPIC_API_KEY=...           # or GEMINI_API_KEY=...; without a key everything runs offline

agentic-os roadmap                     # how the 12 stages map to modules
agentic-os run "What is 1234 * 5678?"
agentic-os run "How does hybrid search work?" --docs notes.md
agentic-os run --goal "Research X and then write a summary"
agentic-os chat
agentic-os serve --port 8000           # HTTP API
```

```python
from agentic_os import AgenticOS

with AgenticOS() as system:
    system.ingest("Agentic OS gives every agent run a PID.", source="overview.md")

    @system.tool
    def lookup_order(order_id: str) -> dict:
        """Look up an order by id.

        Args:
            order_id: The order identifier, e.g. "A-1001".
        """
        return {"order_id": order_id, "status": "shipped"}

    proc = system.run("Where is order A-1001, and what does Agentic OS give each run?")
    print(proc.result["output"], proc.result["citations"])
```

**Provider selection.** `AGENTIC_OS_PROVIDER=auto` (the default) uses Claude
when the `anthropic` package and an `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`
are present, then Gemini when `google-genai` and a `GEMINI_API_KEY` are present,
and otherwise an offline heuristic model, so the examples and tests run anywhere.
To force a provider, set `AGENTIC_OS_PROVIDER=anthropic` (for example with an
`ant auth login` profile), `=gemini` (default model `gemini-flash-latest`) or
`=mock`. Anthropic-compatible gateways such as OpenRouter work through the
`anthropic` provider with `ANTHROPIC_BASE_URL`. The default model is
`claude-opus-5-5`, with `effort=high` and server-side refusal fallbacks on.
See [`agentic_os/config.py`](agentic_os/config.py) for every setting.

## The roadmap, implemented

| # | Roadmap stage | Where | What you get |
|---|---|---|---|
| 1 | Introduction to Agentic AI | `agents/base.py`, `kernel.py` | The autonomous perceive→decide→act loop; agents as OS processes |
| 2 | Fundamentals of AI & ML | `learning/ml.py`, `memory/embeddings.py` | Softmax regression trained by gradient descent (powers `IntentRouter`); feature-hashing embeddings |
| 3 | Programming & AI Frameworks | `tools/` | `@tool` turns Python functions into JSON-Schema tools; validated, sandboxed execution |
| 4 | Large Language Models | `llm/` | Provider interface, Claude (Anthropic SDK) and Gemini (Google Gen AI SDK), token estimates and context budgets |
| 5 | Understanding AI Agents | `agents/` | Reactive (`ReflexAgent`) vs. goal-oriented (`GoalAgent`) agents, ReAct tool loop, multi-agent `Crew` |
| 6 | AI Memory & Knowledge Retrieval | `memory/` | Short-term `ConversationBuffer`, long-term `LongTermMemory`, `VectorStore`, chunking |
| 7 | Decision-Making & Planning | `planning/`, `agents/goal.py` | LLM goal decomposition into a task DAG, topological execution, failure propagation |
| 8 | Prompt Engineering & Adaptation | `prompts/` | Templates, zero/few-shot, chain-of-thought, role prompts, dynamic few-shot by similarity |
| 9 | RL & Self-Improvement | `learning/` | Human feedback store, ε-greedy/UCB1 bandits over strategies, LLM judge, Reflexion retries + lessons |
| 10 | Retrieval-Augmented Generation | `rag/` | BM25 + vector hybrid search, Reciprocal Rank Fusion, cited context, retrieval as a tool |
| 11 | Deploying AI Agents | `server/`, `observability/`, `Dockerfile` | Stdlib HTTP API, async process pool, bearer auth, traces, latency percentiles |
| 12 | Real-World Applications | `apps/`, `examples/` | Research assistant, data reports, support triage automation, decision briefs |

[`docs/ROADMAP.md`](docs/ROADMAP.md) walks through each stage with the
concepts, the code to read, and exercises.

## War Room (web app)

`agentic-os serve` also serves the **War Room**, a mobile-first web app (Polish UI) that you can install on
Android as an app (PWA). It needs `AGENTIC_OS_API_KEY` as its password.

| Tab | What you can do |
|---|---|
| 💬 Czat | Multi-turn chat with any agent; see the tools it used and its sources; 👍/👎 feedback |
| 🤖 Agenci | Create agents (instructions, model, tools, skills, thinking mode, delegation, tool approval), connect external agents (OpenAI-compatible such as Hermes/Grok/OpenClaw, A2A, webhook), run crews |
| 📋 Zadania | Every run as a process with status and result |
| 🧠 Mózg | Knowledge graph of documents, facts, episodes and lessons; **import memory** from ChatGPT, Claude, Gemini (Takeout), Grok, DeepSeek, Mistral exports, or a pasted memory list |
| 📚 Wiedza | Add documents to the knowledge base (RAG) |
| 🧩 Rozszerzenia | **Skills** (SKILL.md instruction packs), **mods** (turn any HTTP API into a tool), **MCP servers** (remote Streamable HTTP or local stdio) |
| 🔑 Modele | API keys for any LLM: Anthropic, Gemini, OpenAI, xAI, OpenRouter, Groq, DeepSeek, Mistral, Hugging Face, Ollama or any OpenAI-compatible server |
| 📊 Monitoring | Token usage, tool calls, latency percentiles |

Keys and secrets are stored only on the server (owner-only files under `AGENTIC_OS_DATA_DIR`) and are returned masked.

## Core concepts

**Agent** (`Agent`): a model, instructions, tools, and optional memory and
knowledge. `run(task)` loops: the model either calls tools (results are fed
back as observations) or answers. Tool errors are returned to the model so it
can recover. `max_steps`, refusals and truncation are handled explicitly, and
an `approve_tool` hook enables human-in-the-loop control.

**Kernel** (`AgenticOS`): owns shared services (tool registry, long-term
memory, knowledge base, feedback, tracer, metrics). It spawns agents from
templates and runs tasks as processes, synchronously (`run`) or on a worker
pool (`submit` / `wait` / `ps`).

**Multi-agent**: `Crew` supports `sequential` pipelines, a `manager` that
delegates via tools, a `router` (trained classifier or LLM), and `plan`
(planner assigns tasks to agents). Any agent becomes a tool with `agent.as_tool()`.

**Self-improvement**: `AdaptiveAgent` treats prompt strategies as bandit
arms and learns from rewards (automatic or human). `ReflectiveAgent` grades,
critiques and retries, and stores lessons in long-term memory for future runs.

All of this is available from the kernel and the War Room, not only in code:
`spawn(name, mode="reflective" | "adaptive", permission="ask" | "smart" | "auto" | "full", delegates=[...])`.

* `mode="reflective"`: every chat or task answer is graded, critiqued and retried. Only the best attempt is
  kept in the conversation, and the critiques are stored as lessons.
* `mode="adaptive"`: each run picks a prompt strategy (concise, step by step, expert) with an epsilon-greedy
  bandit. 👍/👎 in the chat updates it, and the stats persist in `bandits.json`.
* `permission` sets how much the agent may do on its own:
  * `ask`: every tool call waits until the operator taps *Zezwól*, *Zawsze* (always allow this tool for
    this agent) or *Odmów* in the War Room (`GET /approvals`, `POST /approvals/{id}`).
  * `smart`: read-only tools run freely. Tools tagged `readonly` include the built-ins, knowledge search,
    memory recall, `GET` mods and MCP tools whose server sets `readOnlyHint`. Anything that changes
    something waits for approval, and the agent gets an `ask_user` tool to ask the user a question when it
    judges a decision ambiguous, irreversible or a matter of preference.
  * `auto` (default): autonomous within its selected tools, never waits for the user.
  * `full`: every tool, MCP server, mod and agent in the system (including ones added later), with no
    confirmations and a higher step limit.

  Approvals and questions that nobody answers time out after 5 minutes: the tool call is denied and the
  question is left for the agent to decide.

  Permission and thinking mode can be changed at any time with `update_agent(name, permission=..., mode=...)`
  (`POST /agents/{name}/settings`). In the War Room this is done from the agent card or the selector next to
  the chat. The change applies immediately, even to a task that is already running. Switching to `auto` or
  `full` lets any tool call that is waiting for approval through. Chat history, learned strategy stats and
  "always allow" approvals are kept.
* `delegates=[...]`: gives the agent `ask_<agent>` tools to hand off sub-tasks. Delegation is limited to 3
  levels so agents that call each other cannot loop forever.

## HTTP API

```bash
AGENTIC_OS_API_KEY=change-me agentic-os serve --host 0.0.0.0 --port 8000

curl -s localhost:8000/run -H 'Authorization: Bearer change-me' \
     -d '{"task": "What is 2^16?"}'
curl -s localhost:8000/run -H 'Authorization: Bearer change-me' \
     -d '{"task": "Summarize our docs", "async": true}'        # -> 202 + pid
curl -s localhost:8000/processes/<pid> -H 'Authorization: Bearer change-me'
```

Endpoints: `GET /health /status /metrics /agents /processes /processes/{pid} /traces/{trace_id}`,
`POST /run /agents /knowledge /goal /processes/{pid}/feedback`. Details are in
[`agentic_os/server/__init__.py`](agentic_os/server/__init__.py).

```bash
docker build -t agentic-os .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY -e AGENTIC_OS_API_KEY=change-me agentic-os
```

## Development

```bash
pip install -e ".[dev,anthropic]"
pytest -q            # 60 tests, fully offline (no API calls)
ruff check .
for f in examples/*.py; do python "$f"; done
```

### Design choices

* **Zero runtime dependencies** in the core. The Anthropic SDK is an optional extra.
* **Provider-neutral messages** (`llm/base.py`); Claude assistant turns are
  replayed with their original content blocks so thinking stays valid.
* **Append-only conversations**: retrieved knowledge and memories go into the
  user turn, not the system prompt, which keeps history stable and cache-friendly.
* **Swappable baselines**: the hashing embedder, brute-force vector store and
  in-process server are deliberately simple. Each sits behind a small interface,
  so you can drop in real embeddings, a vector DB or another web framework.

## License

MIT
