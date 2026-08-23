"""Interactive HVAC AI-PID comparison demo.

Run: streamlit run streamlit_app.py
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from hvac_pid.ai_controllers import FNNGainController, IncrementalRLController
from hvac_pid.config import dynamic_demo_scenario
from hvac_pid.controllers import PIController, identify_fopdt, imc_pi, ziegler_nichols_pi
from hvac_pid.metrics import calculate_metrics
from hvac_pid.simulator import simulate
from hvac_pid.tuning import BayesianGainTuner


DISPLAY_NAMES = {
    "Z-N": "Z-N 反应曲线法",
    "IMC": "IMC 内模控制",
    "Bayesian Auto-tune": "贝叶斯自动整定",
    "FNN Self-tuning": "FNN 在线自整定",
    "RL Self-tuning": "RL 在线自整定",
}


st.set_page_config(page_title="HVAC AI-PID Lab", layout="wide")
st.title("变频精密 / 机柜空调 AI-PID 仿真平台")
st.caption("把空调理解为“慢慢降温的冰箱”，PID 是调节压缩机力度的两个旋钮。此页面对比：固定旋钮、离线找出的固定旋钮、以及运行中会微调的旋钮。")

with st.sidebar:
    st.header("工况与扰动")
    setpoint = st.slider("目标温度 (°C)", 20.0, 28.0, 23.8, 0.1)
    outdoor = st.slider("室外基准温度 (°C)", 26.0, 45.0, 34.0, 0.5)
    equipment = st.slider("持续设备热负荷 (W)", 100, 3000, 550, 50)
    door = st.slider("开门热负荷跃变 (W)", 0, 5000, 2600, 100)
    duration = st.slider("开门持续时间 (min)", 0, 60, 12, 1)
    selected = st.multiselect(
        "并联算法", ["Z-N", "IMC", "Bayesian Auto-tune", "FNN Self-tuning", "RL Self-tuning"],
        default=["Z-N", "IMC", "Bayesian Auto-tune", "FNN Self-tuning", "RL Self-tuning"],
    )

scenario = replace(
    dynamic_demo_scenario(), setpoint_after_c=setpoint, outdoor_c=outdoor,
    internal_load_w=float(equipment), door_open_load_w=float(door), door_open_duration_minutes=float(duration),
)
@st.cache_resource
def load_trained_self_tuning_models() -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load models that were trained by `python main.py`, never train in the UI."""
    artifact_dir = Path(__file__).parent / "outputs"
    fnn_path, rl_path = artifact_dir / "fnn_rule_table.npy", artifact_dir / "rl_q_table.npy"
    return (
        np.load(fnn_path) if fnn_path.exists() else None,
        np.load(rl_path) if rl_path.exists() else None,
    )

model = identify_fopdt(scenario)
fallback = imc_pi(model)
fnn_rule_table, rl_q_table = load_trained_self_tuning_models()
controllers = {}
if "Z-N" in selected:
    controllers["Z-N"] = PIController(*ziegler_nichols_pi(model))
if "IMC" in selected:
    controllers["IMC"] = PIController(*fallback)
if "Bayesian Auto-tune" in selected:
    with st.spinner("正在为当前工况执行离线搜索；完成后，这组 Kp/Ki 在本次运行中固定不变…"):
        gains = BayesianGainTuner(iterations=3, candidates=256).tune(scenario, seed=11)
    controllers["Bayesian Auto-tune"] = PIController(gains.kp, gains.ki)
if "FNN Self-tuning" in selected:
    if fnn_rule_table is None:
        st.warning("FNN 尚未离线训练。请先运行 `python main.py --quick`，生成 fnn_rule_table.npy 后再使用在线自整定。")
    else:
        controllers["FNN Self-tuning"] = FNNGainController(fallback, rule_table=fnn_rule_table)
if "RL Self-tuning" in selected:
    if rl_q_table is None:
        st.warning("RL 尚未离线训练。请先运行 `python main.py --quick`，生成 rl_q_table.npy 后再使用在线自整定。")
    else:
        controllers["RL Self-tuning"] = IncrementalRLController(fallback, q_table=rl_q_table)

if not controllers:
    st.info("请至少选择一种算法。")
    st.stop()

frames, metric_rows = [], []
for name, controller in controllers.items():
    result = simulate(scenario, controller, seed=17)
    metrics = calculate_metrics(result)
    metric_rows.append({"algorithm": name, **metrics})
    frames.append(pd.DataFrame({
        "hour": result.minute / 60.0, "algorithm": name, "algorithm_zh": DISPLAY_NAMES[name], "zone_c": result.zone_c,
        "setpoint_c": result.setpoint_c, "command_pct": result.command * 100.0,
        "kp": result.kp, "ki": result.ki, "load_w": result.internal_load_w,
    }))
data = pd.concat(frames, ignore_index=True)
metrics_df = pd.DataFrame(metric_rows).set_index("algorithm")

col1, col2 = st.columns(2)
with col1:
    st.subheader("温度跟踪")
    st.line_chart(data.pivot(index="hour", columns="algorithm_zh", values="zone_c"), height=280)
    st.caption(f"设定值（最终）：{setpoint:.1f} °C；FOPDT: K={model.process_gain_c_per_u:.2f}, τ={model.time_constant_minutes:.1f} min, L={model.delay_minutes:.1f} min")
with col2:
    st.subheader("压缩机输出功率 / PWM")
    st.line_chart(data.pivot(index="hour", columns="algorithm_zh", values="command_pct"), height=280)

st.subheader("AI 实时输出 Kp、Ki")
st.caption("贝叶斯自动整定：本次仿真开始前搜索一次，之后固定。FNN/RL：使用此前离线训练的模型，每隔一段时间根据温度偏差和变化趋势微调。")
gain_view = data.melt(id_vars=["hour", "algorithm", "algorithm_zh"], value_vars=["kp", "ki"], var_name="gain", value_name="value")
# Streamlit's built-in chart does not reliably support pandas MultiIndex
# columns (created by pivoting algorithm and gain together), so flatten each
# series name before passing it to the chart.
gain_view["series"] = gain_view["algorithm_zh"] + " · " + gain_view["gain"].str.upper()
gain_chart = gain_view.pivot_table(index="hour", columns="series", values="value", aggfunc="first")
st.line_chart(gain_chart, height=260)

st.subheader("统一指标")
st.dataframe(metrics_df[["itae_c_hour2", "settling_time_hour", "max_undershoot_c", "compressor_output_variance", "fallback_events", "objective"]].style.format("{:.4g}"), use_container_width=True)
st.download_button("下载逐时间步 CSV", data.to_csv(index=False).encode("utf-8-sig"), "hvac_dynamic_timeseries.csv", "text/csv")
