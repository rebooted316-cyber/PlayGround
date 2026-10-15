import pytest

from agentic_os.tools import ToolError, ToolRegistry, calculator, csv_summary, safe_eval, text_stats, tool


@tool(tags=["demo"])
def greet(name: str, times: int = 1, loud: bool = False, tags: list[str] | None = None) -> str:
    """Greet someone.

    Args:
        name: Who to greet.
        times: How many times.
    """
    text = " ".join([f"hello {name}"] * times)
    return text.upper() if loud else text


def test_schema_from_type_hints_and_docstring():
    schema = greet.parameters
    assert greet.description == "Greet someone."
    assert schema["required"] == ["name"]
    assert schema["properties"]["name"] == {"type": "string", "description": "Who to greet."}
    assert schema["properties"]["times"]["type"] == "integer"
    assert schema["properties"]["times"]["default"] == 1
    assert schema["properties"]["loud"]["type"] == "boolean"
    assert schema["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}


def test_execute_coerces_and_validates():
    reg = ToolRegistry([greet])
    assert reg.execute("greet", {"name": "ada", "times": "2", "loud": "true"}).content == "HELLO ADA HELLO ADA"
    missing = reg.execute("greet", {})
    assert missing.is_error and "Missing required" in missing.content
    unknown = reg.execute("greet", {"name": "a", "bogus": 1})
    assert unknown.is_error and "Unknown argument" in unknown.content
    bad = reg.execute("greet", {"name": "a", "times": 1.5})
    assert bad.is_error
    assert reg.execute("nope", {}).is_error


def test_tool_exceptions_become_error_results():
    @tool
    def explode() -> str:
        """Always fails."""
        raise RuntimeError("kaboom")

    result = ToolRegistry([explode]).execute("explode", {})
    assert result.is_error and "kaboom" in result.content


def test_non_string_results_are_json():
    assert ToolRegistry([text_stats]).execute("text_stats", {"text": "a b. c"}).content.startswith("{")


def test_registry_subset_and_names():
    reg = ToolRegistry([greet, calculator, text_stats])
    assert reg.subset(tags=["math"]).names() == ["calculator"]
    assert reg.subset(names=["greet"]).names() == ["greet"]
    with pytest.raises(ToolError):
        reg.register(tool(lambda: 1, name="bad name!"))


def test_calculator_is_safe():
    assert calculator(expression="2^10") == "2^10 = 1024"
    assert safe_eval("sqrt(16) + pi") == pytest.approx(4 + 3.14159, rel=1e-4)
    for evil in ["__import__('os').system('ls')", "open('x')", "(1).__class__", "9**9**9"]:
        with pytest.raises((ValueError, SyntaxError)):
            safe_eval(evil)


def test_csv_summary():
    out = csv_summary(csv_text="city,sales\nParis,10\nRome,30\nParis,20\n")
    assert out["rows"] == 3
    assert out["columns"]["sales"]["mean"] == 20.0
    assert out["columns"]["city"]["top"][0] == ("Paris", 2)
