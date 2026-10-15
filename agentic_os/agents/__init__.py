"""Agent architectures (roadmap stages 1 and 5)."""

from agentic_os.agents.base import DEFAULT_INSTRUCTIONS, Agent, AgentResult
from agentic_os.agents.goal import GoalAgent, GoalResult
from agentic_os.agents.multi import Crew, CrewResult
from agentic_os.agents.reactive import ReflexAgent, Rule

__all__ = [
    "DEFAULT_INSTRUCTIONS",
    "Agent",
    "AgentResult",
    "Crew",
    "CrewResult",
    "GoalAgent",
    "GoalResult",
    "ReflexAgent",
    "Rule",
]
