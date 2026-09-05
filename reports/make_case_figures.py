# -*- coding: utf-8 -*-
"""生成《AI-PI 整定实验报告》图 12-27：三标准工况 x 五算法逐算法闭环响应曲线（15 张单图）
+ 七算法统一演示曲线。

数据来源（v3 封存批次，不重新训练、不改写封存产物）：
- archive/outputs_review_v3/policy_manifest_v3.json    （IMC 部署增益）
- archive/outputs_review_v3/global_bayesian_tuning.csv （BO 全局固定增益）
- archive/outputs_review_v3/fnn_rule_table.npy / fnn_context_coefficients.npy / rl_q_table.npy
- archive/outputs_review_v3/rl_training_transitions.csv（RL 覆盖掩码重建，逻辑同 pipeline）
- archive/outputs_review_v3/case_metrics.csv           （复现自检基准）
- archive/outputs_embedded_demo/*_temperature_demo.csv （七算法统一演示）

自检：按封存种子（7 + 400_003 + 工况序号）复现三工况仿真，逐算法与
case_metrics.csv 的 RMSE/IAE/ITAE/最大过冷/综合目标对比，任一偏差超限即报错退出，
保证图中曲线与封存验收指标严格同源。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hvac_pid.config import Scenario, typical_case_scenarios            # noqa: E402
from hvac_pid.controllers import identify_fopdt_audit, ziegler_nichols_pi  # noqa: E402
from hvac_pid.metrics import calculate_metrics                           # noqa: E402
from hvac_pid.pipeline import _controllers_for_scenario                  # noqa: E402
from hvac_pid.simulator import simulate                                  # noqa: E402

from make_figures import (                                               # noqa: E402
    C_AI_AUTO, C_AI_SELF, C_CLASSICAL, FIG_DIR, GRID, INK, INK2, MUTED,
)

V3 = PROJECT_ROOT / "archive/outputs_review_v3"
DEMO = PROJECT_ROOT / "archive/outputs_embedded_demo"
PIPELINE_SEED = 7
CASE_SEED_OFFSET = 400_003

# 算法展示顺序与样式（颜色区分大类：经典灰 / 自动整定橙 / 自整定蓝）
ALGO_STYLE = {
    "Ziegler-Nichols":    {"color": "#898781", "label": "Z-N 反应曲线法", "file": "ZN"},
    "IMC PI":             {"color": "#52514e", "label": "IMC-λ",          "file": "IMC"},
    "Bayesian Auto-tune": {"color": "#eb6834", "label": "贝叶斯 BO",      "file": "BO"},
    "FNN Self-tuning PI": {"color": "#8fbded", "label": "FNN 自整定",     "file": "FNN"},
    "RL Self-tuning PI":  {"color": "#2a78d6", "label": "RL 自整定",      "file": "RL"},
}
ALGO_ORDER = list(ALGO_STYLE.keys())

# (case_key, 工况短名, 工况参数说明, 起始图号) —— 每工况 5 张，按 ALGO_ORDER 递增
CASE_INFO = [
    ("初次快速降温", "工况一", "31.5→24 °C，外温 35 °C，延迟 6 min", 12),
    ("设定温度突变", "工况二", "1 h 时 25→23 °C，外温 34 °C，延迟 7 min", 17),
    ("持续外界热扰动", "工况三", "外温 37±4.5 °C，3 h 开门 12 min +2600 W，延迟 8 min", 22),
]

DEMO_FILES = [
    ("zn_temperature_demo.csv",      {"color": "#b3b2a8", "ls": "--", "label": "Z-N 反应曲线法"}),
    ("imc_temperature_demo.csv",     {"color": "#52514e", "ls": "-",  "label": "IMC-λ"}),
    ("bo_temperature_demo.csv",      {"color": "#eb6834", "ls": "-",  "label": "贝叶斯 BO"}),
    ("safe_bo_temperature_demo.csv", {"color": "#b34a1f", "ls": "--", "label": "安全 BO"}),
    ("llm_temperature_demo.csv",     {"color": "#7a5ea8", "ls": "-",  "label": "LLM Agent"}),
    ("fnn_temperature_demo.csv",     {"color": "#8fbded", "ls": "-",  "label": "FNN 自整定"}),
    ("rl_temperature_demo.csv",      {"color": "#2a78d6", "ls": "-",  "label": "RL 自整定", "lw": 2.0}),
]


def load_v3_policies():
    """从 v3 封存产物重建五算法控制器所需的全部参数。"""
    import json

    manifest = json.loads((V3 / "policy_manifest_v3.json").read_text(encoding="utf-8"))
    imc_gains = (manifest["validated_fallback_gains"]["kp"], manifest["validated_fallback_gains"]["ki"])

    commissioning_fopdt, _, _ = identify_fopdt_audit(Scenario())
    zn_gains = ziegler_nichols_pi(commissioning_fopdt)

    bo = pd.read_csv(V3 / "global_bayesian_tuning.csv")
    fixed_gains = (float(bo["kp"].iloc[-1]), float(bo["ki"].iloc[-1]))

    fnn_rule_table = np.load(V3 / "fnn_rule_table.npy")
    fnn_context = np.load(V3 / "fnn_context_coefficients.npy")
    rl_q_table = np.load(V3 / "rl_q_table.npy")

    transitions = pd.read_csv(V3 / "rl_training_transitions.csv")
    rl_covered = np.zeros((5, 5, 3), dtype=bool)
    for _, row in transitions.iterrows():
        rl_covered[int(row["state_error_bin"]), int(row["state_delta_bin"]), int(row["state_command_bin"])] = True

    return fixed_gains, zn_gains, imc_gains, fnn_rule_table, rl_q_table, rl_covered, fnn_context


def reproduce_cases():
    """按封存种子复现三工况 x 五算法仿真，自检与 case_metrics.csv 一致；返回结果与指标。"""
    fixed_gains, zn_gains, imc_gains, fnn_rule_table, rl_q_table, rl_covered, fnn_context = load_v3_policies()
    reference = pd.read_csv(V3 / "case_metrics.csv")
    results: dict[str, dict[str, object]] = {}
    metrics_all: dict[str, dict[str, dict[str, float]]] = {}
    worst = 0.0
    for case_index, (case_name, scenario) in enumerate(typical_case_scenarios().items()):
        controllers = _controllers_for_scenario(
            scenario, fixed_gains, zn_gains, imc_gains, fnn_rule_table, rl_q_table, rl_covered, fnn_context
        )
        results[case_name] = {}
        metrics_all[case_name] = {}
        for name, controller in controllers.items():
            result = simulate(scenario, controller, seed=PIPELINE_SEED + CASE_SEED_OFFSET + case_index)
            results[case_name][name] = result
            metrics = calculate_metrics(result)
            metrics_all[case_name][name] = metrics
            ref = reference[(reference["case"] == case_name) & (reference["controller"] == name)].iloc[0]
            for key in ("rmse_c", "iae_c_hour", "itae_c_hour2", "max_undershoot_c", "objective"):
                diff = abs(float(metrics[key]) - float(ref[key]))
                scale = max(abs(float(ref[key])), 1e-9)
                worst = max(worst, diff / scale)
                if diff / scale > 1e-6:
                    raise SystemExit(
                        "复现自检失败：" + case_name + "/" + name + "/" + key
                        + " 复现值 %.8f vs 封存值 %.8f" % (metrics[key], float(ref[key]))
                    )
    print("复现自检通过：3 工况 x 5 算法与 case_metrics.csv 最大相对偏差 %.2e" % worst)
    return results, metrics_all


def _style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def plot_single(case_label, case_desc, scenario, result, metrics, algo_name, fignum):
    """单算法 × 单工况图：上温度（设定值 + 舒适带 + 事件），下容量指令，角落标注关键指标。"""
    style = ALGO_STYLE[algo_name]
    hour = result.minute / 60.0

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08})
    ax = axes[0]
    ax.plot(hour, result.setpoint_c, color=INK, linestyle="--", linewidth=1.4, label="设定值")
    ax.fill_between(hour, result.setpoint_c - 0.5, result.setpoint_c + 0.5,
                    color=MUTED, alpha=0.14, label="±0.5 °C 舒适带")
    if scenario.door_open_load_w > 0:
        door_start = scenario.door_open_hour
        door_end = door_start + scenario.door_open_duration_minutes / 60.0
        ax.axvspan(door_start, door_end, color=C_AI_AUTO, alpha=0.10, label="开门扰动")
    if scenario.setpoint_change_hour is not None:
        ax.axvline(scenario.setpoint_change_hour, color=MUTED, linestyle=":", linewidth=1.2)
    ax.plot(hour, result.zone_c, color=style["color"], linewidth=1.9, label=style["label"])
    ax.set_ylabel("区域温度（°C）")
    temps = np.concatenate([result.zone_c, result.setpoint_c])
    lo, hi = float(temps.min()), float(temps.max())
    ax.set_ylim(lo - 0.3, hi + 0.28 * (hi - lo))
    _style_axis(ax)
    ax.legend(fontsize=8.5, frameon=True, edgecolor=GRID, loc="upper right", ncol=2)

    ax = axes[1]
    ax.plot(hour, result.command * 100.0, color=style["color"], linewidth=1.4)
    ax.set_ylabel("压缩机容量指令（%）")
    ax.set_xlabel("时间（h）")
    ax.set_ylim(-3, 105)
    _style_axis(ax)

    fig.suptitle("图 %d  %s · %s —— %s" % (fignum, case_label, case_desc, style["label"]),
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.925,
             "ITAE %.3f °C·h² · 调节时间 %.2f h · 最大过冷 %.3f °C"
             % (metrics["itae_c_hour2"], metrics["settling_time_hour"], metrics["max_undershoot_c"]),
             fontsize=10, color=INK2, ha="left", va="top")
    filename = "fig%02d_%s_%s.png" % (fignum, case_label, style["file"])
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("生成 " + filename)


def plot_embedded_demo():
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08})
    first = True
    for filename, style in DEMO_FILES:
        df = pd.read_csv(DEMO / filename)
        minute = df["simulated_minute"].to_numpy()
        if first:
            axes[0].plot(minute, df["setpoint_c"], color=INK, linestyle="--", linewidth=1.4, label="设定值")
            axes[0].fill_between(minute, df["comfort_low_c"], df["comfort_high_c"],
                                 color=MUTED, alpha=0.14, label="±0.5 °C 舒适带")
            door = df["door_open"].to_numpy() > 0
            if door.any():
                axes[0].axvspan(minute[door].min(), minute[door].max(), color=C_AI_AUTO, alpha=0.10,
                                label="开门扰动")
            first = False
        axes[0].plot(minute, df["temperature_c"], color=style["color"], linestyle=style["ls"],
                     linewidth=style.get("lw", 1.4), label=style["label"])
        axes[1].plot(minute, df["command_pct"], color=style["color"], linestyle=style["ls"],
                     linewidth=style.get("lw", 1.2), label=style["label"])
    axes[0].set_ylabel("机柜温度（°C）")
    lo, hi = float(axes[0].get_ylim()[0]), float(axes[0].get_ylim()[1])
    axes[0].set_ylim(lo, hi + 0.38 * (hi - lo))  # 顶部留白给图例
    _style_axis(axes[0])
    axes[0].legend(ncol=4, fontsize=8, frameon=True, edgecolor=GRID, loc="upper right",
                   columnspacing=1.0, handlelength=2.0)
    axes[1].set_ylabel("压缩机容量指令（%）")
    axes[1].set_xlabel("时间（min）")
    axes[1].set_ylim(-3, 105)
    _style_axis(axes[1])

    fig.suptitle("图 27  七算法统一演示（ESP32 温度闭环：30→24 °C，min 189 开门扰动）",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.savefig(FIG_DIR / "fig27_七算法统一演示.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("生成 fig27_七算法统一演示.png")


if __name__ == "__main__":
    case_results, case_metrics = reproduce_cases()
    scenarios = typical_case_scenarios()
    for case_name, case_label, case_desc, base_fignum in CASE_INFO:
        for offset, algo_name in enumerate(ALGO_ORDER):
            plot_single(case_label, case_desc, scenarios[case_name],
                        case_results[case_name][algo_name], case_metrics[case_name][algo_name],
                        algo_name, base_fignum + offset)
    plot_embedded_demo()
    print("完成。")
