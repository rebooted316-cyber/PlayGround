import pytest

from agentic_os.prompts import ChainOfThought, DynamicFewShot, Example, FewShot, PromptError, PromptTemplate, RolePrompt


def test_template_render_and_missing_variables():
    t = PromptTemplate("Summarize {doc} for {audience}.")
    assert t.variables == ["audience", "doc"]
    assert t.render(doc="X", audience="kids") == "Summarize X for kids."
    with pytest.raises(PromptError, match="audience"):
        t.render(doc="X")
    assert t.partial(audience="experts").render(doc="Y") == "Summarize Y for experts."


def test_strategies_compose():
    base = "Be helpful."
    assert "step by step" in ChainOfThought().apply(base)
    assert RolePrompt(role="a lawyer").apply(base).startswith("You are a lawyer.")
    few = FewShot(examples=[Example("2+2", "4")]).apply(base)
    assert "<input>2+2</input>" in few and few.startswith(base)


def test_dynamic_few_shot_picks_similar_examples():
    bank = [
        Example("translate hello to French", "bonjour"),
        Example("translate cat to Spanish", "gato"),
        Example("compute the area of a circle of radius 2", "12.57"),
    ]
    picked = DynamicFewShot(bank, k=1).select("what is the area of a circle with radius 5")
    assert picked[0].output == "12.57"
