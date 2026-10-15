"""Stages 2 and 9: learned routing, bandits over prompt strategies, Reflexion."""

from agentic_os import Agent
from agentic_os.learning import AdaptiveAgent, IntentRouter, LLMJudge, ReflectiveAgent
from agentic_os.llm import create_llm
from agentic_os.prompts import ChainOfThought, RolePrompt, ZeroShot
from agentic_os.tools import calculator

llm = create_llm()

# Stage 2: a softmax classifier trained by gradient descent routes requests.
router = IntentRouter({
    "math": ["what is 3 plus 4", "multiply 12 by 9", "square root of 144", "compute 15 percent of 80"],
    "writing": ["write a haiku", "draft an apology email", "summarize this paragraph", "rewrite this more formally"],
})
print("route:", router.predict("what is 7 times 8"), router.classifier.loss_history[-1])

# Stage 9a: a bandit learns which prompt strategy earns the most reward.
agent = Agent("solver", llm, tools=[calculator])
adaptive = AdaptiveAgent(
    agent,
    {"zero_shot": ZeroShot(), "chain_of_thought": ChainOfThought(), "expert": RolePrompt(role="a careful mathematician")},
    evaluator=lambda task, output: 1.0 if "42" in output else 0.0,
)
for _ in range(10):
    adaptive.run("What is 6 * 7?")
print(adaptive.bandit.summary())

# Stage 9b: evaluate -> critique -> retry with lessons (Reflexion).
reflective = ReflectiveAgent(agent, evaluator=LLMJudge(llm), max_attempts=2, threshold=0.7)
print(reflective.run("What is 17 * 23? Show the result.").to_dict())
