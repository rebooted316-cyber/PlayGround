"""Role presets with curated tools and skills (incl. quant-coder / quant-tester)."""

from agentic_os import AgenticOS, Settings
from agentic_os.roles import ROLE_PRESETS, ROLE_SKILLS, get_preset


def test_presets_cover_quant_roles():
    ids = {r["id"] for r in ROLE_PRESETS}
    assert {"quant-coder", "quant-tester", "researcher", "writer"} <= ids
    quant = get_preset("quant-coder")
    assert "portfolio_metrics" in quant["tools"] and "quant-coding" in quant["skills"]
    tester = get_preset("quant-tester")
    assert "python_check" in tester["tools"] and "quant-testing" in tester["skills"]
    assert set(ROLE_SKILLS) >= {"quant-coding", "quant-testing"}


def test_spawn_role_seeds_skills_and_tools(tmp_path):
    system = AgenticOS()
    try:
        quant = system.spawn_role("quant1", "quant-coder")
        tools = set(quant.tools._tools)
        assert {"portfolio_metrics", "backtest_sma", "python_check", "calculator", "load_skill"} <= tools
        assert system.skills.get("quant-coding") is not None
        # skill instructions are loadable through the agent's loader tool
        out = quant.tools.get("load_skill").func(name="quant-coding")
        assert "look-ahead" in out

        tester = system.spawn_role("test1", "quant-tester")
        assert "python_check" in tester.tools._tools
        assert "quant-testing" in tester.skill_names
    finally:
        system.shutdown()


def test_quant_tools_compute():
    system = AgenticOS()
    try:
        metrics = system.tools.get("portfolio_metrics").func([0.01, -0.005, 0.02])
        assert metrics["periods"] == 3 and metrics["win_rate"] == 0.6667
        prices = [100 + i * 0.5 + (5 if i > 30 else 0) for i in range(60)]
        bt = system.tools.get("backtest_sma").func(prices)
        assert bt["trades"] >= 1 and "strategy_return" in bt
        ok = system.tools.get("python_check").func("def f(x):\n    return x*2\n")
        assert ok["ok"] and ok["functions"] == 1
        bad = system.tools.get("python_check").func("def broken(:\n")
        assert bad["ok"] is False and bad["line"] == 1
    finally:
        system.shutdown()


def test_roles_endpoint(tmp_path):
    import threading

    from agentic_os.server import make_server
    from tests.test_war_room import _req

    system = AgenticOS(Settings(provider="mock", api_key="pw", data_dir=str(tmp_path)))
    httpd = make_server(system, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        status, data = _req(base, "GET", "/roles")
        assert status == 200 and any(r["id"] == "quant-coder" for r in data["roles"])
        status, agent = _req(base, "POST", "/agents",
                             {"name": "qc", "preset": "quant-coder"})
        assert status == 201 and "portfolio_metrics" in agent["tools"]
        assert "quant-coding" in agent["skills"]
        status, _ = _req(base, "POST", "/agents", {"name": "bad", "preset": "nope"})
        assert status == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
        system.shutdown()
