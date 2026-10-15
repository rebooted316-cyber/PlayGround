"""Real-world agent applications (roadmap stage 12), built only from OS primitives.

* :func:`research_assistant`: autonomous research over the knowledge base
  (researcher -> writer pipeline with citations).
* :func:`data_report`: autonomous data processing (CSV -> statistics -> report).
* :func:`support_triage`: business automation; a reflex agent handles the
  routine cases instantly and an LLM agent handles the rest.
* :func:`decision_brief`: decision support; a planned, multi-step analysis
  of options.
"""

from __future__ import annotations

from typing import Any

from agentic_os.agents.multi import Crew, CrewResult
from agentic_os.agents.reactive import ReflexAgent, Rule
from agentic_os.kernel import AgenticOS


def research_assistant(system: AgenticOS, question: str) -> CrewResult:
    if "researcher" not in system.agents:
        system.spawn(
            "researcher",
            "You are a research specialist. Search the knowledge base thoroughly (several focused queries if "
            "needed) and report the relevant facts as bullet points, each with its [n] citation. Do not speculate.",
            tool_tags=["knowledge", "math"],
        )
    if "writer" not in system.agents:
        system.spawn(
            "writer",
            "You are a technical writer. Turn the research notes you are given into a clear, well-structured "
            "answer to the question. Keep every citation from the notes and add no unsupported claims.",
            tools=[],
            use_knowledge=False,
        )
    return Crew({"researcher": system.agent("researcher"), "writer": system.agent("writer")}, system.llm, "sequential").run(question)


def data_report(system: AgenticOS, csv_text: str, question: str = "Summarize this dataset.") -> str:
    if "analyst" not in system.agents:
        system.spawn(
            "analyst",
            "You are a data analyst. Always compute statistics with the csv_summary and calculator tools rather "
            "than estimating. Report key figures, notable patterns and caveats concisely.",
            tool_tags=["data", "math"],
            use_knowledge=False,
        )
    proc = system.run(question, agent="analyst", context=f"CSV data:\n{csv_text}")
    if proc.status != "completed" or not proc.result:
        raise RuntimeError(proc.error or "analysis failed")
    return proc.result["output"]


def support_triage(system: AgenticOS, faq: dict[str, str] | None = None) -> ReflexAgent:
    """Return a triage agent: instant answers for known intents, LLM for the rest."""
    faq = faq or {
        r"\b(reset|forgot)\b.*\bpassword\b": "You can reset your password at Settings > Security > Reset password.",
        r"\b(refund|money back)\b": "Refunds are available within 30 days of purchase from Billing > Orders.",
        r"\b(hours|open|opening)\b": "Support is available 24/7 via chat; phone support runs 9:00-17:00 on weekdays.",
    }
    if "support" not in system.agents:
        system.spawn(
            "support",
            "You are a friendly customer-support agent. Answer from the knowledge base when possible; if you "
            "cannot resolve the issue, say you are escalating it to a human and summarize it in one line.",
            tool_tags=["knowledge"],
        )
    rules = [Rule.regex(f"faq_{i}", pattern, lambda _t, a=answer: a) for i, (pattern, answer) in enumerate(faq.items())]
    return ReflexAgent(rules, default=lambda text: system.agent("support").run(text).output, name="triage")


def decision_brief(system: AgenticOS, question: str, options: list[str]) -> dict[str, Any]:
    goal = (
        f"Help decide: {question}\nOptions: {', '.join(options)}. Then evaluate each option's benefits, costs "
        "and risks using the knowledge base where relevant, then recommend one option with a short justification."
    )
    return system.goal(goal).to_dict()


__all__ = ["data_report", "decision_brief", "research_assistant", "support_triage"]
