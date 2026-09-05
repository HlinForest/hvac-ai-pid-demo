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
    # Acceptance outcomes are batch-dependent by design (read from the default
    # artifact dir, canonical dedup-P1 batch outputs_review_v3 where both FNN
    # and RL passed the sealed deployment gate; see RESULTS.md). Z-N still
    # trips the visible commissioning gate on this demo scenario.
    assert traces["fnn"].deployment_accepted and not traces["fnn"].forced_fallback
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
    assert 'text-anchor: middle' in document
    assert '经典：Z-N · IMC · BO · Safe BO' in document
    assert '>温度传感器反馈 T</text>' in document
    # The standalone artifact must explain the closed loop on its own, not
    # only inside the Streamlit wrapper or the invisible aria description.
    assert '闭环路径：测温 → 与目标相减得误差' in document
    # One source of truth for diagram text styling: the SVG's own <style>.
    # A second rule set in the page <head> would silently win or lose on
    # document order alone (16px vs 17px drift).
    assert document.count("#system-diagram text") == 1
    # Presentation attributes that CSS always overrides are dead markup; the
    # marker arrow keeps its fill because nothing overrides it there.
    assert 'class="node" fill=' not in document
    assert 'class="feedback" fill=' not in document
    assert 'fill="#11243a"/></marker>' in document


def test_system_diagram_labels_fit_inside_their_nodes() -> None:
    import re

    from hvac_pid.embedded_demo import _system_diagram_svg

    svg = _system_diagram_svg()
    font_by_class = {"small": 13.0, "tiny": 11.0, "feedback": 12.0, "": 16.0}

    def text_width(content: str, font_size: float) -> float:
        return sum(font_size if ord(char) > 0x2E80 else 0.6 * font_size for char in content)

    checked = 0
    for match in re.finditer(r'<g id="(node-[^"]+)"[^>]*>(.*?)</g>', svg, re.DOTALL):
        group_id, body = match.group(1), match.group(2)
        bounds: list[tuple[float, float, float, float]] = []
        for rect in re.finditer(
            r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', body
        ):
            x, y, width, height = map(float, rect.groups())
            bounds.append((x, y, x + width, y + height))
        for circle in re.finditer(r'<circle cx="([\d.]+)" cy="([\d.]+)" r="([\d.]+)"', body):
            cx, cy, radius = map(float, circle.groups())
            bounds.append((cx - radius, cy - radius, cx + radius, cy + radius))
        assert bounds, group_id
        for text in re.finditer(r'<text x="([\d.]+)" y="([\d.]+)"(?: class="(\w*)")?>([^<]+)</text>', body):
            x, y = float(text.group(1)), float(text.group(2))
            css_class, content = text.group(3) or "", text.group(4)
            font_size = font_by_class[css_class]
            width = text_width(content, font_size)
            left, top, right, bottom = bounds[0]
            assert left + 2 <= x - width / 2, (group_id, content)
            assert x + width / 2 <= right - 2, (group_id, content)
            assert top + 2 <= y <= bottom - 2, (group_id, content)
            checked += 1
    # Every diagram label must be covered; zero matches means the regex
    # drifted away from the emitted markup, not that the diagram is clean.
    assert checked >= 15, checked


def test_replay_proposer_is_deterministic_and_changes_no_more_than_ten_percent() -> None:
    context = {"current_gains": {"kp": 0.4, "ki": 0.004}, "current_metrics": {}}
    first = ReplayPIDProposer().propose(context)
    second = ReplayPIDProposer().propose(context)
    assert first == second
    assert abs(first.kp / 0.4 - 1.0) <= 0.10
    assert abs(first.ki / 0.004 - 1.0) <= 0.10
