# Learning Agentic AI with Agentic OS

A stage-by-stage guide to the "Roadmap to Learn Agentic AI". Each stage
lists the concepts, the code that implements them, a way to try it, and
exercises. Everything runs offline; set `ANTHROPIC_API_KEY` to see real
model behaviour.

---

## 1. Introduction to Agentic AI

**Concepts.** An *agent* is AI with autonomous decision-making: given a goal,
it chooses actions, observes the results and continues until done. Plain AI
answers once; an agent loops. Core capabilities: perception (inputs and tool
results), reasoning (the model), action (tools), and memory.

**Code.** `agentic_os/agents/base.py` (`Agent.run`, the loop) and
`agentic_os/kernel.py` (agents running as OS processes).

**Try it.** `agentic-os run "What is 1234 * 5678?" --json`, then look at
`steps` and `tool_calls`.

**Exercises.** Add a `max_cost` stop condition to `Agent.run` using `usage`.

## 2. Fundamentals of AI & ML

**Concepts.** Supervised learning, loss functions, gradient descent,
regularization, and feature representations. The roadmap also lists RL for
smart agents; that is covered in stage 9.

**Code.** `learning/ml.py`: `SoftmaxClassifier` computes the cross-entropy
gradient `p_c - 1[c = y]` by hand, with learning-rate decay and L2.
`memory/embeddings.py`: feature hashing turns text into vectors.

**Try it.** Train an `IntentRouter` (see `examples/05_self_improvement.py`)
and plot `classifier.loss_history`.

**Exercises.** Add mini-batches; add a held-out accuracy metric; compare
against a nearest-centroid classifier.

## 3. Programming & AI Frameworks

**Concepts.** Function calling: the model sees each tool's name, description
and JSON Schema, and emits structured calls that your code executes.

**Code.** `tools/registry.py`: schema generation from type hints and
docstrings, argument validation and coercion, and errors returned to the
model instead of raised. `tools/builtin.py`: an AST-whitelisted calculator
(no `eval`).

**Try it.** `examples/02_custom_tools.py`.

**Exercises.** Add `Enum` support to the schema generator; add a per-tool
timeout.

## 4. Large Language Models

**Concepts.** Transformers, tokens, context windows, sampling. Agents need
a clean provider boundary and must budget context.

**Code.** `llm/base.py` (neutral `Message`/`LLMResponse`),
`llm/anthropic_provider.py` (Claude Messages API: tool use, effort, refusal
fallbacks, and verbatim replay of assistant content blocks), `llm/tokens.py`
(estimates and truncation), and `llm/mock.py` (offline models for tests and demos).

**Exercises.** Add streaming to `AnthropicProvider` with
`client.messages.stream(...).get_final_message()`; add a second provider.

## 5. Understanding AI Agents

**Concepts.** Agent architectures: reactive (condition→action), goal-oriented
(plan→act), utility-based, and multi-agent. Task-oriented agents execute;
goal-oriented agents decide *what* to execute.

**Code.** `agents/reactive.py` (`ReflexAgent`), `agents/goal.py`
(`GoalAgent`), `agents/multi.py` (`Crew` with four collaboration processes).
`apps/support_triage` combines a reflex front-line with an LLM fallback.

**Exercises.** Add a `debate` process to `Crew` in which two agents critique
each other before a judge decides.

## 6. AI Memory & Knowledge Retrieval

**Concepts.** Short-term memory is the context window; long-term memory is
external storage queried by similarity. Vector databases, semantic search and
chunking.

**Code.** `memory/short_term.py` (turn-aware eviction never splits a tool call
from its result), `memory/long_term.py` (fact / episode / lesson memories, and
memory tools agents can call), `memory/vector_store.py`, and `memory/chunking.py`.

**Exercises.** Add time-decayed recall scores; summarize evicted turns
instead of dropping them.

## 7. Decision-Making & Planning

**Concepts.** Autonomous goal setting, hierarchical planning, multi-agent
problem solving, and planning that improves through feedback.

**Code.** `planning/__init__.py` (LLM planner, task DAG, Kahn topological sort,
cycle detection, robust JSON extraction) and `agents/goal.py` (executes tasks
in order, passes prerequisite results as context, skips tasks whose
dependencies failed, and synthesizes a final answer).

**Exercises.** Add re-planning: when a task fails, ask the planner for a
revised plan for the remaining work.

## 8. Prompt Engineering & Adaptation

**Concepts.** Zero-, one- and few-shot prompting, chain-of-thought,
dynamic prompt tuning, and instruction adaptation per task.

**Code.** `prompts/__init__.py`: `PromptTemplate`, `FewShot`, `ChainOfThought`,
`RolePrompt`, and `DynamicFewShot` (retrieves the most similar examples for
each task).

**Exercises.** Build a prompt A/B test with `AdaptiveAgent` and an `LLMJudge`.

## 9. Reinforcement Learning & Self-Improvement

**Concepts.** RL from human feedback, reward signals, explore vs. exploit,
and learning from mistakes.

**Code.** `learning/feedback.py` (reward log), `learning/bandit.py`
(ε-greedy, UCB1), and `learning/reflection.py` (`LLMJudge`, `ReflectiveAgent`
Reflexion loop with lessons persisted to long-term memory, and `AdaptiveAgent`
bandit over prompt strategies). In the kernel, `give_feedback()` turns
well-rated answers into lessons.

**Exercises.** Implement Thompson sampling; use `/processes/{pid}/feedback`
rewards to drive an `AdaptiveAgent` behind the API.

## 10. Retrieval-Augmented Generation

**Concepts.** Combine search with generation: retrieve, then generate with
citations. Hybrid search, context expansion, embeddings.

**Code.** `rag/bm25.py` and `rag/pipeline.py`: dual indexing, Reciprocal Rank
Fusion, a token-budgeted numbered context, and `as_tool()` for agentic
retrieval (the agent decides when to search).

**Exercises.** Add neighbour-chunk context expansion; add a re-ranking step
with an LLM.

## 11. Deploying AI Agents

**Concepts.** Serving agents through APIs, scaling, latency, monitoring.

**Code.** `server/` (stdlib HTTP API, async processes, bearer auth, body
limits), `observability/` (nested trace spans, counters, p50/p95 latencies),
`kernel.py` (thread-pool process execution), `Dockerfile`, and the CI workflow.

**Exercises.** Add an SSE endpoint that streams process status; export spans
to OpenTelemetry.

## 12. Real-World AI Agent Applications

**Code.** `apps/`: `research_assistant` (autonomous research and writing),
`data_report` (data processing), `support_triage` (business workflow
automation), and `decision_brief` (decision support). See `examples/06_real_world_apps.py`.

**Exercises.** Build your own app from the primitives: spawn specialists,
pick a `Crew` process, add domain tools, and expose it through the API.
