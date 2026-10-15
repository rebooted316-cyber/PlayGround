import pytest

from agentic_os.agents import Agent
from agentic_os.learning import (
    UCB1,
    AdaptiveAgent,
    EpsilonGreedy,
    FeedbackStore,
    LLMJudge,
    ReflectiveAgent,
    SoftmaxClassifier,
)
from agentic_os.llm import ScriptedLLM
from agentic_os.memory import LongTermMemory
from agentic_os.prompts import ChainOfThought, ZeroShot


def test_softmax_classifier_learns_with_decreasing_loss():
    texts = ["refund my order", "money back please", "I want a refund", "reset my password", "forgot password", "cannot log in password"]
    labels = ["billing", "billing", "billing", "account", "account", "account"]
    clf = SoftmaxClassifier(epochs=40).fit(texts, labels)
    assert clf.loss_history[-1] < clf.loss_history[0]
    assert clf.predict("please refund me") == "billing"
    assert clf.predict("password reset link") == "account"
    assert sum(clf.predict_proba("x").values()) == pytest.approx(1.0)


@pytest.mark.parametrize("bandit_cls", [EpsilonGreedy, UCB1])
def test_bandits_converge_to_best_arm(bandit_cls):
    bandit = bandit_cls(["a", "b", "c"], seed=1)
    true_means = {"a": 0.2, "b": 0.8, "c": 0.5}
    import random

    rng = random.Random(7)
    for _ in range(400):
        arm = bandit.select()
        bandit.update(arm, 1.0 if rng.random() < true_means[arm] else 0.0)
    assert bandit.best() == "b"
    assert bandit.stats["b"].pulls > bandit.stats["a"].pulls


def test_bandit_persistence_roundtrip():
    b = EpsilonGreedy(["x", "y"])
    b.update("x", 1)
    c = EpsilonGreedy(["x", "y"])
    c.load_dict(b.to_dict())
    assert c.stats["x"].pulls == 1


def test_feedback_store_persists(tmp_path):
    path = str(tmp_path / "fb.jsonl")
    store = FeedbackStore(path)
    store.add("r1", 1.0, arm="cot")
    store.add("r2", -1.0, arm="zero")
    assert FeedbackStore(path).average("cot") == 1.0
    with pytest.raises(ValueError):
        store.add("r3", 5)


def test_llm_judge_parses_score():
    judge = LLMJudge(ScriptedLLM(['{"score": 7, "reason": "mostly right"}', "garbage"]))
    assert judge("t", "o") == pytest.approx(0.7) and judge.last_reason == "mostly right"
    assert judge("t", "o") == 0.0


def test_reflective_agent_retries_with_lessons():
    llm = ScriptedLLM(["bad answer", "- be precise", lambda msgs, s, t: "good answer" if "be precise" in msgs[-1].content else "bad"])
    lessons = LongTermMemory()
    scores = iter([0.2, 0.9])
    result = ReflectiveAgent(Agent("a", llm), evaluator=lambda task, out: next(scores), lessons=lessons).run("task")
    assert result.output == "good answer" and result.best_score == 0.9
    assert len(result.attempts) == 2 and result.attempts[0].critique == "- be precise"
    assert lessons.recall("task be precise", kind="lesson")


def test_adaptive_agent_learns_preferred_strategy():
    llm = ScriptedLLM([lambda msgs, s, t: "stepwise" if "step by step" in s else "plain"] * 41)
    agent = AdaptiveAgent(
        Agent("a", llm),
        {"zero_shot": ZeroShot(), "chain_of_thought": ChainOfThought()},
        evaluator=lambda task, out: 1.0 if out == "stepwise" else 0.0,
    )
    for _ in range(40):
        agent.run("solve")
    assert agent.bandit.best() == "chain_of_thought"
    run_id, _ = agent.run("again")
    agent.reward(run_id, 1.0, comment="great")
    assert agent.feedback.items[-1].comment == "great"
