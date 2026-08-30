from __future__ import annotations

from pathlib import Path

from hvac_pid.advanced_tuning import ReplayPIDProposer
from hvac_pid.embedded_demo import (
    ALGORITHM_ORDER,
    demo_scenario,
    run_all_demos,
    write_interactive_html,
)


ROOT = Path(__file__).resolve().parents[1]


def test_seven_algorithms_expose_temperature_and_safe_deployment() -> None:
    traces = run_all_demos(project_root=ROOT, scenario=demo_scenario(), provider="replay", seed=29)
    assert tuple(traces) == ALGORITHM_ORDER
    for key, trace in traces.items():
        assert trace.summary["stable"] == 1, key
        assert len(trace.rows) == 301
        assert trace.rows[0]["temperature_c"] > trace.rows[-1]["temperature_c"]
        assert any(row["in_comfort_band"] for row in trace.rows)
        assert any(row["door_open"] for row in trace.rows)
        assert all(row["comfort_low_c"] < row["setpoint_c"] < row["comfort_high_c"] for row in trace.rows)
    assert traces["fnn"].forced_fallback and not traces["fnn"].deployment_accepted
    assert traces["zn"].forced_fallback and not traces["zn"].deployment_accepted
    assert traces["rl"].deployment_accepted and not traces["rl"].forced_fallback
    assert [row["tool"] for row in traces["llm"].agent_trace] == [
        "inspect_history", "evaluate_candidate", "evaluate_candidate",
        "evaluate_candidate", "finish",
    ]


def test_html_contains_live_temperature_diagram_and_llm_audit(tmp_path: Path) -> None:
    traces = run_all_demos(project_root=ROOT, scenario=demo_scenario(), provider="replay", seed=31)
    output = tmp_path / "demo.html"
    write_interactive_html(output, traces)
    document = output.read_text(encoding="utf-8")
    for required in (
        "当前被控温度", "目标±0.5°C稳定带", "开门扰动", "node-pi",
        "压缩机限制器", "LLM Agent 工具调用与安全门审计", "无网络录制的 Agent 工具调用 replay",
        "inspect_history", "evaluate_candidate", "finish",
        "开始", "暂停", "复位",
    ):
        assert required in document
    # A literal newline inside a single-quoted JavaScript string makes the
    # entire page fail before draw() can attach button handlers.
    assert "function drawAgentAudit" in document
    assert "String.fromCharCode(10,10)" in document
    assert "row.tool+'\n" not in document
    # The low-frequency scheduler must not sit on top of the feedback path.
    assert 'M1040 300V420H315V300' in document
    # SVG presentation attributes keep the diagram readable even when a local
    # file browser fails to apply the outer page stylesheet.
    assert 'fill="#e7f6ee" stroke="#17864b"' in document
    assert 'fill="#fff4df" stroke="#e37a12"' in document
    assert 'fill="#11243a"' in document
    assert 'text-anchor: middle' in document
    assert '经典：Z-N · IMC · BO · Safe BO' in document
    assert '>温度传感器反馈 T</text>' in document
    assert '温度传感器反馈：测温 → 算误差' not in document


def test_replay_proposer_is_deterministic_and_changes_no_more_than_ten_percent() -> None:
    context = {"current_gains": {"kp": 0.4, "ki": 0.004}, "current_metrics": {}}
    first = ReplayPIDProposer().propose(context)
    second = ReplayPIDProposer().propose(context)
    assert first == second
    assert abs(first.kp / 0.4 - 1.0) <= 0.10
    assert abs(first.ki / 0.004 - 1.0) <= 0.10
