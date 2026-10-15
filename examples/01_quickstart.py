"""Quickstart: boot the OS, give it knowledge, run a task.

Runs offline with the heuristic model. Set ANTHROPIC_API_KEY (and
`pip install agentic-os[anthropic]`) to run on Claude instead.
"""

from agentic_os import AgenticOS

with AgenticOS() as system:
    system.ingest(
        "Agentic OS treats agents like programs: each run is a process with a PID, "
        "and agents share tools, long-term memory and a knowledge base.",
        source="overview.md",
    )
    proc = system.run("How does Agentic OS treat agents?")
    print(proc.result["output"])
    print("citations:", proc.result["citations"])

    proc = system.run("What is 1234 * 5678?")
    print(proc.result["output"])
    print(system.status())
