"""Roles: preset agents with curated tools + skills, e.g. quant-coder + quant-tester crew."""

from agentic_os import AgenticOS

with AgenticOS() as system:
    print("roles:", [r["id"] for r in system.role_presets()])
    system.spawn_role("quant", "quant-coder")
    system.spawn_role("tester", "quant-tester")
    print("quant tools:", system.describe("quant")["tools"])
    print("tester skills:", system.describe("tester")["skills"])

    crew = system.crew(["quant", "tester"], process="sequential")
    result = crew.run("Design a test plan for an SMA(10,30) strategy on a 60-point price series.")
    print("crew ->", result.output[:300])
