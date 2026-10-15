"""Stage 3: turn Python functions into tools with @tool."""

from agentic_os import Agent, tool
from agentic_os.llm import create_llm


@tool
def convert_temperature(value: float, to: str = "fahrenheit") -> str:
    """Convert a temperature between Celsius and Fahrenheit.

    Args:
        value: The temperature to convert.
        to: Target unit, "fahrenheit" or "celsius".
    """
    if to == "fahrenheit":
        return f"{value}°C = {value * 9 / 5 + 32:.1f}°F"
    return f"{value}°F = {(value - 32) * 5 / 9:.1f}°C"


print(convert_temperature.parameters)  # JSON Schema generated from hints + docstring

agent = Agent("converter", create_llm(), tools=[convert_temperature])
print(agent.run("Convert 21.5 °C to Fahrenheit.").output)
