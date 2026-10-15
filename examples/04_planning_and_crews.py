"""Stages 5 and 7: goal decomposition and multi-agent collaboration."""

from agentic_os import AgenticOS

with AgenticOS() as system:
    system.spawn("researcher", "You research facts carefully and cite sources.", tool_tags=["knowledge"])
    system.spawn("writer", "You write clear, concise summaries.", tools=[], use_knowledge=False)
    system.ingest("Jolt is a physics engine. Godot 4.6 made Jolt the default for 3D.", source="godot.md")

    result = system.goal("Find out what changed about physics in Godot 4.6 and then write a one-line summary")
    for task in result.plan.tasks:
        print(f"[{task.status}] {task.id}: {task.description}")
    print(result.output)

    for process in ("sequential", "manager", "plan"):
        crew = system.crew(["researcher", "writer"], process=process)
        print(process, "->", crew.run("Summarize the Godot 4.6 physics change.").output[:120])
