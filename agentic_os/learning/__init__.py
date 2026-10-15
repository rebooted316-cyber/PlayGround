"""Learning: ML fundamentals (stage 2) and self-improvement (stage 9)."""

from agentic_os.learning.bandit import UCB1, Bandit, EpsilonGreedy
from agentic_os.learning.feedback import Feedback, FeedbackStore
from agentic_os.learning.ml import IntentRouter, SoftmaxClassifier
from agentic_os.learning.reflection import AdaptiveAgent, Attempt, LLMJudge, ReflectionResult, ReflectiveAgent

__all__ = [
    "UCB1",
    "AdaptiveAgent",
    "Attempt",
    "Bandit",
    "EpsilonGreedy",
    "Feedback",
    "FeedbackStore",
    "IntentRouter",
    "LLMJudge",
    "ReflectionResult",
    "ReflectiveAgent",
    "SoftmaxClassifier",
]
