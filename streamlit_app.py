"""Interactive seven-algorithm temperature-control demonstration.

Run with ``streamlit run streamlit_app.py``.  The application never trains a
controller in the browser: it replays the accepted deployment traces used by
the offline HTML report and labels shadow candidates and safety fallback.
"""

from __future__ import annotations

import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from hvac_pid.embedded_demo import (
    ALGORITHM_ORDER,
    DISPLAY_NAMES,
    _system_diagram_svg,
    demo_scenario,
    run_all_demos,
)


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="ESP32 七算法温度闭环 Demo", page_icon="🌡️", layout="wide")


@st.cache_resource(show_spinner="正在装载冻结策略并运行统一虚拟场景……")
def load_traces(setpoint_c: float, door_load_w: float):
    return run_all_demos(
        project_root=ROOT,
        scenario=demo_scenario(setpoint_c=setpoint_c, door_load_w=door_load_w),
        provider="replay",
        seed=29,
    )


def trace_frame(trace) -> pd.DataFrame:
    return pd.DataFrame(trace.rows)


def temperature_chart(frame: pd.DataFrame, show_candidate: bool) -> alt.Chart:
    domain = [float(frame["simulated_minute"].min()), float(frame["simulated_minute"].max())]
    band = alt.Chart(frame).mark_area(color="#60a5fa", opacity=0.14).encode(
        x=alt.X("simulated_minute:Q", title="模拟物理时间（min）", scale=alt.Scale(domain=domain)),
        y=alt.Y("comfort_low_c:Q", title="室内温度（°C）", scale=alt.Scale(zero=False)),
        y2="comfort_high_c:Q",
    )
    actual = alt.Chart(frame).mark_line(color="#ef4444", strokeWidth=3).encode(
        x="simulated_minute:Q", y="temperature_c:Q",
        tooltip=["simulated_minute", "temperature_c", "status"],
    )
    target = alt.Chart(frame).mark_line(color="#0f172a", strokeDash=[8, 5], strokeWidth=2).encode(
        x="simulated_minute:Q", y="setpoint_c:Q"
    )
    first_door = frame[frame["door_open"] == 1].head(1)
    door = alt.Chart(first_door).mark_rule(color="#f59e0b", strokeWidth=3).encode(x="simulated_minute:Q")
    chart = band + target + actual + door
    if show_candidate:
        candidate = alt.Chart(frame).mark_line(color="#a855f7", strokeDash=[5, 4], strokeWidth=2).encode(
            x="simulated_minute:Q", y="candidate_temperature_c:Q"
        )
        chart += candidate
    return chart.properties(height=420).interactive()


def all_temperature_chart(traces) -> alt.Chart:
    frames = []
    for key in ALGORITHM_ORDER:
        columns = ["simulated_minute", "temperature_c", "setpoint_c", "comfort_low_c", "comfort_high_c"]
        part = trace_frame(traces[key])[columns]
        part["algorithm"] = DISPLAY_NAMES[key]
        frames.append(part)
    data = pd.concat(frames, ignore_index=True)
    reference = frames[0]
    band = alt.Chart(reference).mark_area(color="#60a5fa", opacity=0.10).encode(
        x=alt.X("simulated_minute:Q", title="模拟物理时间（min）"),
        y=alt.Y("comfort_low_c:Q", title="室内温度（°C）", scale=alt.Scale(zero=False)),
        y2="comfort_high_c:Q",
    )
    target = alt.Chart(reference).mark_line(color="#0f172a", strokeDash=[8, 5], strokeWidth=2).encode(
        x="simulated_minute:Q", y="setpoint_c:Q"
    )
    lines = alt.Chart(data).mark_line(strokeWidth=2.2).encode(
        x="simulated_minute:Q", y="temperature_c:Q",
        color=alt.Color("algorithm:N", title="实际部署控制器"),
        tooltip=["algorithm", "simulated_minute", "temperature_c"],
    )
    return (band + target + lines).properties(height=450).interactive()


def read_serial_snapshot(port: str, baud: int) -> str:
    """Read one ESP32 telemetry line; importing pyserial remains optional."""
    try:
        import serial  # type: ignore
    except ImportError:
        return "未安装 pyserial；运行 pip install pyserial 后即可连接目标板。"
    try:
        with serial.Serial(port, baudrate=baud, timeout=1.2) as device:
            line = device.readline().decode("utf-8", errors="replace").strip()
            return line or "串口已打开，但暂未收到完整遥测行。"
    except Exception as exc:  # hardware/permission errors belong in the UI
        return f"串口读取失败：{exc}"


st.title("ESP32 七算法温度闭环可视化 Demo")
st.caption("90 秒演示对应 5 小时虚拟物理时间；它用于展示与软件验收，不代表真实空调能在 90 秒内把房间降温。")

with st.sidebar:
    st.header("统一场景")
    setpoint = st.slider("目标温度 r（°C）", 20.0, 27.0, 24.0, 0.1)
    door_load = st.slider("开门额外热负荷（W）", 500, 5000, 3200, 100)
    mode = st.radio("展示模式", ["单算法逐步演示", "七算法同场比较"])
    selected = st.selectbox("算法", ALGORITHM_ORDER, format_func=lambda key: DISPLAY_NAMES[key])
    st.divider()
    st.header("ESP32 串口（可选）")
    port = st.text_input("串口", "COM3")
    baud = st.selectbox("波特率", [115200, 57600, 9600], index=0)
    if st.button("读取一帧目标板遥测", use_container_width=True):
        st.session_state["serial_snapshot"] = read_serial_snapshot(port, baud)
    if "serial_snapshot" in st.session_state:
        st.code(st.session_state["serial_snapshot"], language="json")

traces = load_traces(setpoint, float(door_load))

with st.expander("先看完整闭环框图：温度为什么会变化？", expanded=True):
    components.html(
        "<div style='font-family:Arial,sans-serif;background:#f8fafc;padding:12px;border-radius:14px'>"
        + _system_diagram_svg()
        + "<p><b>循环：</b>测温 → 与目标相减 → 安全 PI 计算容量请求 → 限制器处理最低频率、斜率、量化和启停 → 空调改变温度 → 再次测温。FNN/RL 只低频改增益；LLM 不进入 100 ms 实时环。</p></div>",
        height=510,
        scrolling=False,
    )

if mode == "七算法同场比较":
    st.subheader("七条实际部署温度曲线")
    st.altair_chart(all_temperature_chart(traces), use_container_width=True)
    st.info("Z‑N 与 FNN 的候选方案没有通过本场景验收，所以实际部署曲线使用 IMC 安全回退；这不是宣称候选算法已经成功。")
    summary = pd.DataFrame([traces[key].summary for key in ALGORITHM_ORDER])
    st.dataframe(
        summary[["display_name", "stable_first_minute", "door_recovery_minutes", "max_undershoot_c", "fallback_used", "deployment_accepted"]]
        .rename(columns={
            "display_name": "算法", "stable_first_minute": "稳定时间/min",
            "door_recovery_minutes": "扰动恢复/min", "max_undershoot_c": "最大过冷/°C",
            "fallback_used": "发生回退", "deployment_accepted": "候选验收通过",
        }), use_container_width=True, hide_index=True,
    )
    combined = pd.concat([trace_frame(traces[key]) for key in ALGORITHM_ORDER], ignore_index=True)
    st.download_button(
        "下载七算法逐时间步 CSV", combined.to_csv(index=False).encode("utf-8-sig"),
        "seven_algorithm_temperature.csv", "text/csv",
    )
else:
    trace = traces[selected]
    frame = trace_frame(trace)
    st.session_state.setdefault("play_index", 0)
    st.session_state.setdefault("playing", False)
    st.session_state.play_index = min(st.session_state.play_index, len(frame) - 1)

    controls = st.columns([1, 1, 1, 1, 4])
    if controls[0].button("▶ 开始", use_container_width=True):
        st.session_state.playing = True
    if controls[1].button("⏸ 暂停", use_container_width=True):
        st.session_state.playing = False
    if controls[2].button("↺ 复位", use_container_width=True):
        st.session_state.playing = False
        st.session_state.play_index = 0
    if controls[3].button("🚪 跳到开门", use_container_width=True):
        st.session_state.play_index = int(frame.index[frame["door_open"] == 1][0])
    play_index = controls[4].slider(
        "演示墙钟进度", 0, len(frame) - 1, st.session_state.play_index,
        label_visibility="collapsed",
    )
    st.session_state.play_index = play_index
    visible = frame.iloc[: play_index + 1]
    current = frame.iloc[play_index]

    st.subheader(DISPLAY_NAMES[selected])
    if trace.forced_fallback:
        st.warning(f"{DISPLAY_NAMES[selected]} 候选被拒绝；红色实际温度由 IMC 安全参数控制，紫色虚线是候选影子温度。")

    meter, chart_col = st.columns([1, 4])
    with meter:
        st.metric("当前温度", f"{current.temperature_c:.2f} °C", f"{current.temperature_c-current.setpoint_c:+.2f} °C")
        st.metric("目标温度", f"{current.setpoint_c:.2f} °C")
        st.metric("压缩机容量", f"{current.command_pct:.1f} %")
        st.write(f"**状态：{current.status}**")
        st.progress(min(1.0, max(0.0, float(current.command_pct) / 100.0)), text="执行器限制器后的实际容量")
        st.caption(f"墙钟 {current.wall_clock_second:.1f}/90 s · 模拟 {current.simulated_minute:.0f}/300 min")
    with chart_col:
        st.altair_chart(temperature_chart(visible, trace.forced_fallback), use_container_width=True)

    auxiliary = visible[["simulated_minute", "command_pct", "kp", "ki"]].set_index("simulated_minute")
    left, right = st.columns(2)
    left.subheader("压缩机实际容量")
    left.line_chart(auxiliary[["command_pct"]], height=220)
    right.subheader("低频调度后的 Kp / Ki")
    right.line_chart(auxiliary[["kp", "ki"]], height=220)
    st.download_button(
        "下载当前算法 CSV", frame.to_csv(index=False).encode("utf-8-sig"),
        f"{selected}_temperature_demo.csv", "text/csv",
    )

    if st.session_state.playing:
        if st.session_state.play_index < len(frame) - 1:
            st.session_state.play_index += 2
            time.sleep(0.12)
            st.rerun()
        else:
            st.session_state.playing = False
