import json

import pytest

from agentic_os.llm import ScriptedLLM
from agentic_os.planning import Plan, PlanError, Planner, Task, extract_json


def test_execution_order_and_cycle_detection():
    plan = Plan("g", [Task("c", "C", ["a", "b"]), Task("a", "A"), Task("b", "B", ["a"])])
    assert [t.id for t in plan.execution_order()] == ["a", "b", "c"]
    with pytest.raises(PlanError, match="cycle"):
        Plan("g", [Task("a", "A", ["b"]), Task("b", "B", ["a"])]).validate()
    with pytest.raises(PlanError, match="unknown"):
        Plan("g", [Task("a", "A", ["zzz"])]).validate()


def test_extract_json_tolerates_prose_and_fences():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Plan: {"tasks": []} done') == {"tasks": []}
    with pytest.raises(PlanError):
        extract_json("no json here")


def test_planner_parses_llm_plan_and_drops_unknown_agents():
    plan_json = {
        "goal": "ship",
        "tasks": [
            {"id": "t1", "description": "research", "agent": "researcher"},
            {"id": "t2", "description": "write", "depends_on": ["t1"], "agent": "ghost"},
        ],
    }
    llm = ScriptedLLM([json.dumps(plan_json)])
    plan = Planner(llm, agents={"researcher": "finds facts", "writer": "writes"}).plan("ship")
    assert [t.id for t in plan.tasks] == ["t1", "t2"]
    assert plan.tasks[0].agent == "researcher" and plan.tasks[1].agent is None
    assert "researcher" in llm.calls[0]["system"]


def test_planner_falls_back_to_single_task():
    plan = Planner(ScriptedLLM(["I can't make a plan"])).plan("do the thing")
    assert len(plan.tasks) == 1 and plan.tasks[0].description == "do the thing"
