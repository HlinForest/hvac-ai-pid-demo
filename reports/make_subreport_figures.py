# -*- coding: utf-8 -*-
"""拆分报告配图生成脚本。

输出目录：reports/分报告/figures/（自包含：新图 + 从 reports/figures/ 复制复用图）
新增：
  - fig_fopdt_fit_3cases.png       三工况 168 h 阶跃响应 vs FOPDT 最小二乘拟合（重跑辨识并断言封存值）
  - fig_fopdt_crossval.png         FOPDT 代理 vs 3R2C 物理模型 24 h 交叉验证（封存 CSV）
  - fig_fopdt_fit_convergence.png  最小二乘优化收敛轨迹（封存 fopdt_fit_history.csv）
  - fig_flow_{zn,imc,bo,safebo,llm,fnn,rl}.png   七张算法运行框图
  - fig_case{1,2,3}_{safebo,llm}.png  安全 BO / LLM 三工况补跑曲线（冻结增益 + 封存种子）
  - fig_mixed_{zn,imc,bo,fnn,rl}.png  混合工况五算法曲线（封存 dynamic_timeseries.csv）
  - fig_mixed_{safebo,llm}.png     混合工况补跑曲线
  - 复制：fig12–fig26（三工况 x 五算法）、fig1（架构框图）
指标数据：reports/分报告/data/supplement_metrics.json（补跑口径标注）。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hvac_pid.config import dynamic_demo_scenario, typical_case_scenarios  # noqa: E402
from hvac_pid.controllers import PIController, identify_fopdt_audit        # noqa: E402
from hvac_pid.metrics import calculate_metrics                             # noqa: E402
from hvac_pid.simulator import simulate                                    # noqa: E402
from make_figures import (                                                 # noqa: E402
    BASE, C_AI_AUTO, C_AI_SELF, C_CLASSICAL, GRID, INK, INK2, MUTED,
    SURFACE, arrow, box, wash,
)

SUB = Path(__file__).resolve().parent / "分报告"
FIG = SUB / "figures"
DATA = SUB / "data"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
SRC_FIG = Path(__file__).resolve().parent / "figures"

PIPELINE_SEED = 7
CASE_SEED_OFFSET = 400_003
DYNAMIC_SEED_OFFSET = 300_003

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Segoe UI"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": INK2,
})

C_BAD = "#c0392b"   # 失效方法（教材两点法读数）
C_OK = "#1d9e75"    # 有效方法（有界最小二乘）

CASE_INFO = [
    ("初次快速降温", "工况一", "31.5→24 °C，外温 35 °C，延迟 6 min"),
    ("设定温度突变", "工况二", "1 h 时 25→23 °C，外温 34 °C，延迟 7 min"),
    ("持续外界热扰动", "工况三", "外温 37±4.5 °C，3 h 开门 12 min +2600 W，延迟 8 min"),
]
SEALED_FOPDT = {  # outputs_review_v3/engineering_report.md §1.1 / 报告附录 D.3
    "初次快速降温": (59.89, 911.1, 6.0),
    "设定温度突变": (59.89, 912.1, 7.0),
    "持续外界热扰动": (64.63, 913.2, 8.0),
}
SUPPLEMENT_GAINS = {
    "safebo": {"kp": 0.3838, "ki": 0.004441, "label": "风险感知安全 BO"},
    "llm": {"kp": 0.4057, "ki": 0.002769, "label": "LLM Agent"},
}
DYN_NAME = {
    "Ziegler-Nichols": "Z-N 反应曲线法",
    "IMC PI": "IMC-λ",
    "Bayesian Auto-tune": "贝叶斯 BO",
    "FNN Self-tuning PI": "FNN 自整定",
    "RL Self-tuning PI": "RL 自整定",
}
DYN_COLOR = {
    "Ziegler-Nichols": C_CLASSICAL,
    "IMC PI": "#52514e",
    "Bayesian Auto-tune": C_AI_AUTO,
    "FNN Self-tuning PI": "#8fbded",
    "RL Self-tuning PI": C_AI_SELF,
}
METRIC_KEYS = ("itae_c_hour2", "settling_time_hour", "disturbance_recovery_time_hour",
               "max_undershoot_c", "compressor_output_variance", "objective",
               "max_overheat_c", "rmse_c")


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


def fig_two_point_failure():
    """教材两点法（读 t28/t63）在本对象上失效的实证图。

    左：实测交叉点 vs 同参数一阶理论交叉点——28.3% 点提前 66%、63.2% 点仅提前 14%，
        这个不对称使两点法解出负的纯滞后（L = -237 min）。
    右：把试验末点当成稳态幅值 A 时，两点法得到的 τ 随试验长度从 54 min 漂到 1194 min，
        没有收敛性，说明“多测一会儿”也救不回来。
    """
    from hvac_pid.config import Scenario

    model, history, _ = identify_fopdt_audit(Scenario())
    minute = np.asarray([h["minute"] for h in history], dtype=float)
    y0 = float(history[0]["initial_zone_c"])
    measured = np.asarray([h["measured_zone_c"] for h in history], dtype=float)
    drop = y0 - measured
    amp = y0 - float(history[0]["steady_zone_c"])
    tau = float(model.time_constant_minutes)
    delay = float(model.delay_minutes)
    hours = minute / 60.0

    def cross(target):
        idx = np.nonzero(drop / amp >= target)[0]
        return float(minute[idx[0]]) if idx.size else float("nan")

    t28_m, t63_m = cross(0.283), cross(0.632)
    t28_t, t63_t = delay + 0.3312 * tau, delay + tau
    assert abs(tau - 911.08) < 1.0, tau
    assert abs(delay - 5.0) < 0.5, delay
    assert abs(t28_m - 104.0) < 1.0 and abs(t63_m - 786.0) < 1.0, (t28_m, t63_m)

    fit = y0 - amp * (1.0 - np.exp(-np.maximum(minute - delay, 0.0) / tau))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.9))

    ax1.plot(hours, measured, color=MUTED, lw=1.3, label="3R2C 实测响应")
    ax1.plot(hours, fit, color=C_AI_AUTO, lw=1.8, ls="--", label="FOPDT 最小二乘拟合")
    for frac in (0.283, 0.632):
        ax1.axhline(y0 - frac * amp, color=BASE, ls=":", lw=1.0)

    pairs = [
        (0.283, t28_m, t28_t, "实测 t28 = 104", "一阶理论 306.7", "提前 66%"),
        (0.632, t63_m, t63_t, "实测 t63 = 786", "一阶理论 916.1", "提前 14%"),
    ]
    ymin = y0 - amp - 0.55
    for frac, tm, tt, lab_m, lab_t, gap in pairs:
        level = y0 - frac * amp
        ax1.plot([tm / 60.0, tm / 60.0], [ymin, level], color=BASE, ls=":", lw=0.8)
        ax1.plot([tt / 60.0, tt / 60.0], [ymin, level], color=BASE, ls=":", lw=0.8)
        ax1.plot(tm / 60.0, level, "o", color=C_BAD, ms=8, zorder=6)
        ax1.plot(tt / 60.0, level, "o", color=C_OK, ms=8, zorder=6)
        ax1.annotate("", xy=(tt / 60.0, level + 0.10), xytext=(tm / 60.0, level + 0.10),
                     arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.0))
        ax1.text((tm + tt) / 120.0, level + 0.20, gap, ha="center", va="bottom",
                 fontsize=9, color=INK2)
        ax1.text(tm / 60.0 - 0.15, level - 0.16, lab_m, ha="right", va="top",
                 fontsize=9, color=C_BAD)
        ax1.text(tt / 60.0 + 0.15, level - 0.16, lab_t, ha="left", va="top",
                 fontsize=9, color=C_OK)

    ax1.set_xlim(0, 17.5)
    ax1.set_ylim(ymin, y0 + 0.35)
    ax1.set_xlabel("时间（h）")
    ax1.set_ylabel("区域温度（°C）")
    ax1.set_title("交叉点对比：实测 vs 同参数一阶理论", fontsize=11.5,
                  color=INK, fontweight="bold")
    ax1.grid(True, color=GRID, lw=0.6)
    ax1.legend(fontsize=9, loc="upper right", framealpha=0.92)
    ax1.text(0.025, 0.045,
             ("两点法：\nτ = 1.5 × (786 − 104) = 1023 min\n"
              "L = 786 − 1023 = −237 min\n（负延迟，公式失效）"),
             transform=ax1.transAxes, fontsize=9.5, color=C_BAD, va="bottom", ha="left",
             bbox=dict(facecolor="white", edgecolor=C_BAD, boxstyle="round,pad=0.4"))

    lengths = [2, 4, 8, 12, 24, 48, 72, 168]
    taus, shares = [], []
    for hlen in lengths:
        n = int(hlen * 60)
        tt, dd = minute[:n], drop[:n]
        a_used = float(dd[-1])
        shares.append(100.0 * a_used / amp)
        i28 = np.nonzero(dd / a_used >= 0.283)[0]
        i63 = np.nonzero(dd / a_used >= 0.632)[0]
        taus.append(1.5 * (tt[i63[0]] - tt[i28[0]]) if i28.size and i63.size else np.nan)

    ax2.plot(lengths, taus, "o-", color=C_BAD, lw=1.8, ms=6,
             label="两点法 τ（末点当稳态 A）")
    ax2.axhline(tau, color=C_OK, ls="--", lw=1.6, label=f"最小二乘 τ = {tau:.1f} min")
    ax2.set_xscale("log")
    ax2.set_xticks(lengths)
    ax2.set_xticklabels([str(v) for v in lengths])
    ax2.set_xlabel("试验时长（h）")
    ax2.set_ylabel("两点法得到的 τ（min）")
    ax2.set_title("试验越短，τ 越小；且始终不收敛到真值", fontsize=11.5,
                  color=INK, fontweight="bold")
    ax2.grid(True, color=GRID, lw=0.6, which="both")
    ax2.legend(fontsize=9, loc="upper left", framealpha=0.92)
    ax2.text(0.975, 0.045,
             (f"A_used / 真值：{shares[0]:.0f}%（2 h）→ {shares[-1]:.0f}%（168 h）\n"
              "两点法 τ 从 54 min 漂到 1194 min，无收敛性"),
             transform=ax2.transAxes, fontsize=9.5, color=INK2, va="bottom", ha="right",
             bbox=dict(facecolor="white", edgecolor=BASE, boxstyle="round,pad=0.4"))

    fig.suptitle("为什么不用教材 t28/t63 两点法定 τ、L：本对象上它会解出负的纯滞后",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig_fopdt_two_point_failure.png")


# ---------------------------------------------------------------- FOPDT 图
def fig_fopdt_fit():
    scenarios = typical_case_scenarios()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, (case, scenario) in zip(axes, scenarios.items()):
        model, history, _ = identify_fopdt_audit(scenario)
        k_seal, tau_seal, l_seal = SEALED_FOPDT[case]
        assert abs(model.process_gain_c_per_u - k_seal) / k_seal < 0.01, (case, model)
        assert abs(model.time_constant_minutes - tau_seal) / tau_seal < 0.01, (case, model)
        assert abs(model.delay_minutes - l_seal) <= 0.51, (case, model)
        minute = np.asarray([h["minute"] for h in history])
        measured = np.asarray([h["measured_zone_c"] for h in history])
        y0 = float(history[0]["initial_zone_c"])
        amp = float(history[0]["steady_zone_c"])  # y0 - A
        amp = y0 - amp
        delay = float(model.delay_minutes)
        tau = float(model.time_constant_minutes)
        elapsed = np.maximum(minute - delay, 0.0)
        fit_zone = y0 - amp * (1.0 - np.exp(-elapsed / tau))
        nrmse = float(history[0]["fit_normalized_rmse_pct"])
        hours = minute / 60.0
        ax.plot(hours, measured, color=MUTED, lw=1.0, label="3R2C 物理模型实测")
        ax.plot(hours, fit_zone, color=C_AI_AUTO, lw=1.8, ls="--", label="FOPDT 最小二乘拟合")
        ax.axvline(delay / 60.0, color=BASE, ls=":", lw=1.0)
        ax.set_xlabel("时间（h）")
        ax.set_title(case, fontsize=11, color=INK, fontweight="bold")
        ax.grid(True, color=GRID, lw=0.6)
        ax.text(0.03, 0.05,
                (f"K = {model.process_gain_c_per_u:.2f} °C/指令\n"
                 f"τ = {tau:.1f} min\n"
                 f"L = {delay:.0f} min\n"
                 f"归一化 RMSE ≈ {nrmse:.1f}%"),
                transform=ax.transAxes, fontsize=9, color=INK2,
                va="bottom", ha="left",
                bbox=dict(facecolor="white", edgecolor=BASE, boxstyle="round,pad=0.35"))
    axes[0].set_ylabel("区域温度（°C）")
    axes[0].legend(fontsize=9, loc="upper right", framealpha=0.9)
    fig.suptitle("三标准工况 168 h 阶跃试验：3R2C 实测响应 vs FOPDT 最小二乘拟合",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig_fopdt_fit_3cases.png")


def fig_fopdt_crossval():
    df = pd.read_csv(PROJECT_ROOT / "outputs" / "fopdt_cross_validation_timeseries.csv")
    cases = list(dict.fromkeys(df["case"]))
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.6), sharex=True)
    for ax, case in zip(axes, cases):
        sub = df[df["case"] == case]
        t = sub["minute"] / 60.0
        ax.plot(t, sub["physical_zone_c"], color=INK2, lw=1.4, label="3R2C 物理模型")
        ax.plot(t, sub["fopdt_zone_c"], color=C_AI_AUTO, lw=1.4, ls="--", label="FOPDT 代理")
        ax.fill_between(t, sub["physical_zone_c"], sub["fopdt_zone_c"], color=C_AI_AUTO, alpha=0.15)
        maxerr = float(sub["error_c"].abs().max())
        ax.set_ylabel("°C")
        ax.set_title(f"{case}（24 h 交叉验证，最大绝对误差 {maxerr:.2f} °C）",
                     fontsize=10.5, color=INK, loc="left")
        ax.grid(True, color=GRID, lw=0.6)
        ax.legend(fontsize=8.5, loc="lower right", ncol=2, framealpha=0.9)
    axes[-1].set_xlabel("时间（h）")
    fig.suptitle("FOPDT 代理模型 vs 3R2C 物理模型阶跃响应（降阶层交叉验证，归一化 RMSE ≈ 12%）",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, "fig_fopdt_crossval.png")


def fig_fopdt_convergence():
    df = pd.read_csv(PROJECT_ROOT / "outputs_review_v3" / "fopdt_fit_history.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.plot(df["evaluation"], df["rmse_c"], "o", color=MUTED, ms=4, alpha=0.6,
            label="候选 (A, τ, L) 的 RMSE")
    ax.plot(df["evaluation"], df["best_rmse_c"], color=C_AI_AUTO, lw=1.8, label="当前最优 RMSE")
    last = df.iloc[-1]
    ax.annotate((f"收敛：K = {last['best_process_gain_c_per_u']:.2f} °C/指令，"
                 f"τ = {last['best_tau_minutes']:.1f} min，L = {last['best_delay_minutes']:.0f} min\n"
                 f"最终 RMSE = {last['best_rmse_c']:.4f} °C（{int(last['evaluation'])} 次评估）"),
                xy=(last["evaluation"], last["best_rmse_c"]),
                xytext=(0.30, 0.55), textcoords="axes fraction", fontsize=9.5, color=INK2,
                arrowprops=dict(arrowstyle="-|>", color=MUTED),
                bbox=dict(facecolor="white", edgecolor=BASE, boxstyle="round,pad=0.4"))
    ax.set_xlabel("最小二乘优化评估次数")
    ax.set_ylabel("RMSE（°C）")
    ax.grid(True, color=GRID, lw=0.6)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_title("FOPDT 有界最小二乘拟合的收敛过程（168 h 虚拟阶跃，scipy least_squares/TRF）",
                 fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "fig_fopdt_fit_convergence.png")


# ---------------------------------------------------------------- 框图原语
def _canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def _vflow(ax, ys, items, x=16, w=68):
    centers = []
    for y, (text, sub, face, edge) in zip(ys, items):
        box(ax, x, y - 6.5, w, 13, text, sub=sub, face=face, edge=edge, tsize=10.5, subsize=8.5)
        centers.append(y)
    cx = x + w / 2
    for a, b in zip(centers, centers[1:]):
        arrow(ax, cx, a - 6.5, cx, b + 6.5)
    return centers


def flow_zn():
    fig, ax = _canvas(8.6, 5.6)
    box(ax, 2, 88, 44, 10, "输入", sub="稳定工况机柜 + 温度传感器历史",
        face=wash(C_CLASSICAL, 0.12), edge=C_CLASSICAL, tsize=10.5)
    _vflow(ax, [74, 56, 38, 20], [
        ("① 168 h 阶跃试验", "稳态工作点 u0 → u0+12%，记录温度响应至稳态", "white", C_CLASSICAL),
        ("② 有界最小二乘拟合", "同时拟合 A/τ/L，K = A/Δu（详见报告 01）", "white", C_CLASSICAL),
        ("③ 代入 Z-N 公式", "Kp = 0.9τ/(K·L)，Ti = 3.33L，Ki = Kp/Ti", "white", C_CLASSICAL),
        ("④ 限幅冻结", "np.clip 工程限幅（不属 Z-N 原公式）后写入控制器", "white", C_CLASSICAL),
    ])
    arrow(ax, 22, 88, 26, 81)
    box(ax, 58, 0, 40, 11, "输出：固定 Kp/Ti", sub="v3：原值 2.97/0.178 被限幅为 Kp=1.5, Ki=0.08",
        face=wash(C_CLASSICAL, 0.18), edge=C_CLASSICAL, tsize=10.5)
    arrow(ax, 50, 13.5, 64, 11.2, color=C_CLASSICAL)
    ax.set_title("Z-N 反应曲线法：运行框图", fontsize=13, fontweight="bold", color=INK)
    save(fig, "fig_flow_zn.png")


def flow_imc():
    fig, ax = _canvas(8.6, 5.6)
    box(ax, 2, 88, 44, 10, "输入", sub="FOPDT (K, τ, L) + 训练工况集（与 Z-N 共享辨识数据）",
        face=wash(C_CLASSICAL, 0.12), edge=C_CLASSICAL, tsize=10.5)
    _vflow(ax, [74, 56, 38, 20], [
        ("① λ 扫描（仅训练工况）", "21 个候选 λ 逐一全闭环仿真评估目标 J", "white", C_CLASSICAL),
        ("② 选 λ* 并冻结", "λ* = 26.9 min（保守公式值 303.7 min 差近 6 倍）", "white", C_CLASSICAL),
        ("③ 代入 IMC 公式", "Kp = τ/[K(λ+L)]，Ti = min[τ, 4(λ+L)]，Ki = Kp/Ti", "white", C_CLASSICAL),
        ("④ 冻结部署", "验证/测试集不再回调参数", "white", C_CLASSICAL),
    ], x=12, w=52)
    arrow(ax, 22, 88, 26, 81)
    box(ax, 58, 0, 40, 11, "输出：固定 Kp/Ti", sub="同时担任全部 AI 方法的回退基准",
        face=wash(C_CLASSICAL, 0.18), edge=C_CLASSICAL, tsize=10.5)
    arrow(ax, 38, 13.5, 64, 11.2, color=C_CLASSICAL)
    box(ax, 68, 48, 30, 13, "保守公式 λ = max(τ/3, 3L, 12 min)",
        sub="→ fallback_gains 故障回退", face=wash(MUTED, 0.15), edge=MUTED, dashed=True,
        tsize=9, subsize=8)
    arrow(ax, 83, 48, 62, 27, dashed=True, color=MUTED)
    ax.set_title("IMC-λ 内模控制：运行框图", fontsize=13, fontweight="bold", color=INK)
    save(fig, "fig_flow_imc.png")


def flow_bo():
    fig, ax = _canvas(8.6, 6.0)
    box(ax, 2, 90, 44, 9, "输入", sub="训练工况集 + 综合目标 J（多指标加权，不稳定罚 1e6）",
        face=wash(C_AI_AUTO, 0.12), edge=C_AI_AUTO, tsize=10.5)
    _vflow(ax, [76, 58, 40, 22], [
        ("① 初始化样本", "IMC / Z-N / 随机点各若干个", "white", C_AI_AUTO),
        ("② 候选全闭环仿真评估", "候选为 (log Kp, log Ki)；每个候选跑完整仿真算 J", "white", C_AI_AUTO),
        ("③ GP 近似“增益→J”", "Matérn-5/2 高斯过程回归", "white", C_AI_AUTO),
        ("④ EI 期望改进选下一点", "迭代至预算（本次 10 次评估）", "white", C_AI_AUTO),
    ])
    arrow(ax, 22, 90, 26, 83)
    ax.plot([84, 88], [22, 22], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.plot([88, 88], [22, 58], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    arrow(ax, 88, 58, 84.2, 58, dashed=True, color=MUTED)
    ax.text(90.5, 40, "迭代", fontsize=9, color=MUTED, rotation=90, va="center")
    box(ax, 52, 2, 46, 11, "输出：全局固定 Kp/Ki", sub="v3 部署 Kp=0.5355, Ki=0.0063；运行时零负担",
        face=wash(C_AI_AUTO, 0.15), edge=C_AI_AUTO, tsize=10.5)
    arrow(ax, 50, 15.5, 62, 13.2, color=C_AI_AUTO)
    ax.set_title("贝叶斯 BO 自动整定：运行框图", fontsize=13, fontweight="bold", color=INK)
    save(fig, "fig_flow_bo.png")


def flow_safebo():
    fig, ax = _canvas(8.6, 6.0)
    box(ax, 2, 90, 44, 9, "输入", sub="训练+留出工况 + 风险目标（均值 + 0.75×标准差）",
        face=wash(C_AI_AUTO, 0.12), edge=C_AI_AUTO, tsize=10.5)
    _vflow(ax, [76, 58, 40, 22], [
        ("① 每候选重复噪声仿真 ×2", "同候选跨工况得均值与标准差 σ_J", "white", C_AI_AUTO),
        ("② 风险目标 = 均值 J + 0.75σ_J", "主动避开“平均好但波动大”的候选", "white", C_AI_AUTO),
        ("③ β=2 安全 GP + 安全条件门", "稳定 / 斜率 / 最低频率 / 冷偏差 ≤4 °C / 风险分 ≤1.25×IMC", "white", C_AI_AUTO),
        ("④ EI 迭代 7 次评估", "每次评估含跨场景重复仿真", "white", C_AI_AUTO),
    ])
    arrow(ax, 22, 90, 26, 83)
    box(ax, 52, 2, 46, 11, "输出：固定 Kp/Ki", sub="Kp=0.3838, Ki=0.004441；风险目标 33.58（该基准最优）",
        face=wash(C_AI_AUTO, 0.15), edge=C_AI_AUTO, tsize=10.5)
    arrow(ax, 50, 15.5, 62, 13.2, color=C_AI_AUTO)
    ax.set_title("风险感知安全 BO（RaGoOSE 式）：运行框图", fontsize=13, fontweight="bold", color=INK)
    save(fig, "fig_flow_safebo.png")


def flow_llm():
    fig, ax = _canvas(8.8, 5.8)
    box(ax, 2, 88, 44, 10, "输入", sub="历史整定记录 + 指标工具 + 工况信息",
        face=wash(C_AI_AUTO, 0.12), edge=C_AI_AUTO, tsize=10.5)
    box(ax, 16, 62, 44, 16, "LLM Agent（低频离线）",
        sub="自主诊断 → 提出候选 Kp/Ki\n（真实 qwen-max 调用：6 步、2 次候选试验）",
        face="white", edge=C_AI_AUTO, tsize=10.5)
    box(ax, 4, 34, 40, 16, "工具调用", sub="查看历史 → 候选仿真 → 读取指标",
        face="white", edge=C_AI_AUTO, tsize=10.5)
    box(ax, 52, 34, 44, 16, "确定性安全门",
        sub="限幅 → 风险调整改进判定\n改进不足即拒绝（最终决定权在仿真器）",
        face="white", edge=C_AI_AUTO, tsize=10.5)
    arrow(ax, 24, 88, 30, 79, color=C_AI_AUTO)
    arrow(ax, 38, 62, 26, 52, color=C_AI_AUTO)
    arrow(ax, 44, 42, 52, 42, color=C_AI_AUTO)
    arrow(ax, 66, 50, 58, 61, color=MUTED, dashed=True)
    ax.text(68, 55, "反馈指标", fontsize=9, color=MUTED)
    box(ax, 16, 4, 36, 12, "接受 → 冻结部署", sub="真实调用部署：Kp=0.4057, Ki=0.002769",
        face=wash(C_AI_AUTO, 0.15), edge=C_AI_AUTO, tsize=10)
    box(ax, 60, 4, 36, 12, "拒绝 → 保留已接受候选", sub="真实会话：候选①（风险 33.31）被拒",
        face="white", edge=MUTED, dashed=True, tsize=10)
    arrow(ax, 64, 34, 38, 16.8, color=C_AI_AUTO)
    arrow(ax, 82, 34, 80, 16.8, color=MUTED, dashed=True)
    ax.set_title("LLM Agent 工具调用整定：运行框图", fontsize=13, fontweight="bold", color=INK)
    save(fig, "fig_flow_llm.png")


def _two_lane(title, offline, online, out_sub, fname):
    """offline/online: [(x, text, sub)] 各 4/3 框；固定两排布局。"""
    fig, ax = _canvas(9.2, 6.2)
    ax.plot([2, 98], [50, 50], color=BASE, ls="--", lw=1.0)
    ax.text(3, 52, "离线（PC）", fontsize=10, fontweight="bold", color=C_AI_SELF)
    ax.text(3, 44, "在线（每 2 s）", fontsize=10, fontweight='bold', color=C_AI_SELF)
    pos_top = [(2, 84, 44, 12), (54, 84, 44, 12), (54, 62, 44, 12), (2, 62, 44, 12)]
    for (x, y, w, h), (text, sub) in zip(pos_top, offline):
        box(ax, x, y, w, h, text, sub=sub, face="white", edge=C_AI_SELF, tsize=10.5, subsize=8.5)
    arrow(ax, 46, 90, 54, 90, color=C_AI_SELF)
    arrow(ax, 76, 84, 76, 74, color=C_AI_SELF)
    arrow(ax, 54, 68, 46, 68, color=C_AI_SELF)
    pos_bot = [(2, 26, 28, 14), (36, 26, 28, 14), (70, 26, 28, 14)]
    for (x, y, w, h), (text, sub) in zip(pos_bot, online):
        box(ax, x, y, w, h, text, sub=sub, face="white", edge=C_AI_SELF, tsize=10, subsize=8.5)
    arrow(ax, 30, 33, 36, 33, color=C_AI_SELF)
    arrow(ax, 64, 33, 70, 33, color=C_AI_SELF)
    arrow(ax, 24, 62, 16, 42, color=MUTED, dashed=True)
    ax.text(8, 51, "部署", fontsize=9, color=MUTED)
    box(ax, 36, 4, 28, 10, "输出：在线 Kp/Ki", sub=out_sub,
        face=wash(C_AI_SELF, 0.15), edge=C_AI_SELF, tsize=10)
    arrow(ax, 84, 26, 56, 14.5, color=C_AI_SELF)
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK)
    save(fig, fname)


def flow_fnn():
    _two_lane(
        "FNN 在线自整定（5×5 TSK 规则面）：运行框图",
        [("① 造标签", "48 训练场景 × 各做一轮完整 BO\n→ 4464 样本（等效 4181 对象小时）"),
         ("② 学规则面", "(e, ė) 模糊格 → 5×5 = 25 条\n零阶 TSK 规则后件"),
         ("③ 验收门", "规则覆盖 ≥80% 且不劣于 IMC；\n不过则自动写 IMC 回退表（候选另存审计）"),
         ("产物", "fnn_rule_table.npy（25 规则 + 上下文系数）")],
        [("④ 读输入", "e = Tz−Tsp，ė = Δe/Δt\n（变化率口径与训练一致）"),
         ("⑤ 查表加权", "相邻 2×2 = 4 条规则\n加权平均（78 µs/次）"),
         ("⑥ 安全限幅", "增益界 + 变化率 ≤25%\n异常回退 IMC")],
        "交给 100 ms PI 内环执行",
        "fig_flow_fnn.png")


def flow_rl():
    _two_lane(
        "RL 在线自整定（离线 Q 学习）：运行框图",
        [("① 定义 MDP", "状态 = bin(e,5)×bin(ė,5)×bin(容量,3) = 75\n动作 = 相对 IMC 的增益目标比例 ×9"),
         ("② 离线 Q 学习 750 回合", "同一 3R2C 虚拟环境；ε 0.25→0.03\n验证集早停选表（31.95 < 32.91）"),
         ("③ 部署门", "均值差于 IMC 2% 或热状态覆盖 <80%\n→ 整表拒绝，回退 IMC"),
         ("产物", "rl_q_table.npy（Q 表 5×5×3×9 = 675 项）")],
        [("④ 状态分箱", "e、ė、上次实际容量\n→ 75 个状态之一"),
         ("⑤ argmax(Q)", "屏蔽不安全动作后查表\n（88 µs/次）"),
         ("⑥ ±10% 安全变化层", "向目标增益受限步进\n异常回退 IMC")],
        "交给 100 ms PI 内环执行",
        "fig_flow_rl.png")


# ---------------------------------------------------------------- 运行曲线
def _metrics_line(m):
    return (f"ITAE {m['itae_c_hour2']:.2f} °C·h²｜调节 {m['settling_time_hour']:.2f} h｜"
            f"扰动恢复 {m['disturbance_recovery_time_hour']:.2f} h｜过冷 {m['max_undershoot_c']:.2f} °C｜"
            f"指令方差 {m['compressor_output_variance']:.3f}")


def _plot_run(ax1, ax2, minute, zone, setpoint, command, title, color, metrics, events):
    ax1.plot(minute, zone, color=color, lw=1.3, label="区域温度")
    ax1.plot(minute, setpoint, color=INK2, lw=1.0, ls="--", label="设定值")
    ax1.fill_between(minute, setpoint - 0.5, setpoint + 0.5, color="#7fb069", alpha=0.12,
                     label="±0.5 °C 舒适带")
    sp = np.asarray(setpoint, dtype=float)
    jumps = np.flatnonzero(np.abs(np.diff(sp)) > 0.01)
    if jumps.size:
        t = float(minute[jumps[0] + 1])
        ax1.axvline(t, color=MUTED, ls="--", lw=1.0)
        ax2.axvline(t, color=MUTED, ls="--", lw=1.0)
        ax1.annotate("设定值阶跃", xy=(t, 0.97), xycoords=("data", "axes fraction"),
                     fontsize=8, color=MUTED, ha="left", va="top")
    for (t0, t1, lbl) in events:
        ax1.axvspan(t0, t1, color=C_AI_AUTO, alpha=0.10)
        ax1.annotate(lbl, xy=((t0 + t1) / 2.0, 0.02), xycoords=("data", "axes fraction"),
                     fontsize=8, color=C_AI_AUTO, ha="center", va="bottom")
    ax1.set_ylabel("温度（°C）")
    ax1.set_title(title, fontsize=11, fontweight="bold", color=INK, loc="left")
    ax1.grid(True, color=GRID, lw=0.6)
    ax1.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax1.text(0.01, 0.03, metrics, transform=ax1.transAxes, fontsize=8.5, color=INK2,
             va="bottom", ha="left",
             bbox=dict(facecolor="white", edgecolor=BASE, boxstyle="round,pad=0.3"))
    ax2.plot(minute, command * 100.0, color=color, lw=1.1)
    ax2.axhline(25, color=BASE, ls=":", lw=1.0)
    ax2.set_ylabel("容量指令（%）")
    ax2.set_xlabel("时间（min）")
    ax2.set_ylim(-3, 103)
    ax2.grid(True, color=GRID, lw=0.6)


def _events_for(scenario):
    events = []
    if scenario.door_open_hour is not None:
        t0 = scenario.door_open_hour * 60.0
        t1 = t0 + scenario.door_open_duration_minutes
        events.append((t0, t1, f"开门 {scenario.door_open_duration_minutes:.0f} min"))
    if scenario.occupied_load_add_w and scenario.occupied_start_hour is not None:
        events.append((scenario.occupied_start_hour * 60.0,
                       scenario.occupied_end_hour * 60.0, "人员负荷窗口"))
    return events


def _run_figure(scenario, gains, seed, title, color, fname, events):
    controller = PIController(*gains)
    result = simulate(scenario, controller, seed=seed)
    m = calculate_metrics(result)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.8, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2.1, 1]})
    _plot_run(ax1, ax2, result.minute, result.zone_c, result.setpoint_c, result.command,
              title, color, _metrics_line(m), events)
    fig.tight_layout()
    save(fig, fname)
    return {k: float(m[k]) for k in METRIC_KEYS}


def supplement_runs():
    out = {"gains": SUPPLEMENT_GAINS,
           "label": "补跑：冻结增益（不做任何训练/搜索），与 v3 同对象同约束同种子",
           "cases": {}, "dynamic": {}}
    scenarios = typical_case_scenarios()
    short_of = {c[0]: c[1] for c in CASE_INFO}
    desc_of = {c[0]: c[2] for c in CASE_INFO}
    for key, info in SUPPLEMENT_GAINS.items():
        gains = (info["kp"], info["ki"])
        out["cases"][key] = {}
        for case_index, (case, scenario) in enumerate(scenarios.items()):
            title = f"{short_of[case]}（{desc_of[case]}）— {info['label']}（补跑：冻结增益）"
            out["cases"][key][case] = _run_figure(
                scenario, gains, PIPELINE_SEED + CASE_SEED_OFFSET + case_index,
                title, C_AI_AUTO, f"fig_case{case_index + 1}_{key}.png", _events_for(scenario))
        scenario = dynamic_demo_scenario()
        title = f"混合工况 12 h 复合测试 — {info['label']}（补跑：冻结增益）"
        m = _run_figure(scenario, gains, PIPELINE_SEED + DYNAMIC_SEED_OFFSET,
                        title, C_AI_AUTO, f"fig_mixed_{key}.png", _events_for(scenario))
        m["fallback_events"] = 0.0
        out["dynamic"][key] = m
    (DATA / "supplement_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  saved data/supplement_metrics.json")


def mixed_figures_main5():
    ts = pd.read_csv(PROJECT_ROOT / "outputs_review_v3" / "dynamic_timeseries.csv")
    met = pd.read_csv(PROJECT_ROOT / "outputs_review_v3" / "dynamic_metrics.csv").set_index("controller")
    file_key = {"Ziegler-Nichols": "zn", "IMC PI": "imc", "Bayesian Auto-tune": "bo",
                "FNN Self-tuning PI": "fnn", "RL Self-tuning PI": "rl"}
    scenario = dynamic_demo_scenario()
    for name, sub in ts.groupby("controller"):
        m = met.loc[name]
        mm = {k: float(m[k]) for k in ("itae_c_hour2", "settling_time_hour",
                                       "disturbance_recovery_time_hour", "max_undershoot_c",
                                       "compressor_output_variance")}
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.8, 6.4), sharex=True,
                                       gridspec_kw={"height_ratios": [2.1, 1]})
        title = f"混合工况 12 h 复合测试 — {DYN_NAME[name]}（v3 封存数据）"
        _plot_run(ax1, ax2, sub["minute"].to_numpy(), sub["zone_c"].to_numpy(),
                  sub["setpoint_c"].to_numpy(), sub["command_pct"].to_numpy() / 100.0,
                  title, DYN_COLOR[name], _metrics_line(mm), _events_for(scenario))
        fig.tight_layout()
        save(fig, f"fig_mixed_{file_key[name]}.png")


def copy_figures():
    names = ["fig1_系统结构框图.png"]
    names += [
        "fig04_imc_lambda扫描.png",
        "fig05_bo搜索轨迹.png",
        "fig06_安全bo风险搜索.png",
        "fig07_llm工具调用轨迹.png",
        "fig08_fnn训练收敛.png",
        "fig09_rl训练收敛.png",
    ]
    names += [
        "fig12_工况一_ZN.png", "fig13_工况一_IMC.png", "fig14_工况一_BO.png",
        "fig15_工况一_FNN.png", "fig16_工况一_RL.png",
        "fig17_工况二_ZN.png", "fig18_工况二_IMC.png", "fig19_工况二_BO.png",
        "fig20_工况二_FNN.png", "fig21_工况二_RL.png",
        "fig22_工况三_ZN.png", "fig23_工况三_IMC.png", "fig24_工况三_BO.png",
        "fig25_工况三_FNN.png", "fig26_工况三_RL.png",
    ]
    for n in names:
        shutil.copy2(SRC_FIG / n, FIG / n)
    print(f"  copied {len(names)} figures")


def main():
    print("[1/6] FOPDT 拟合 / 交叉验证 / 收敛图（重跑辨识并断言封存值）")
    fig_two_point_failure()
    fig_fopdt_fit()
    fig_fopdt_crossval()
    fig_fopdt_convergence()
    print("[2/6] 七张算法运行框图")
    flow_zn(); flow_imc(); flow_bo(); flow_safebo(); flow_llm(); flow_fnn(); flow_rl()
    print("[3/6] 安全 BO / LLM 三工况与混合工况补跑")
    supplement_runs()
    print("[4/6] 混合工况五算法曲线（封存 CSV）")
    mixed_figures_main5()
    print("[5/6] 复制复用图")
    copy_figures()
    print("[6/6] done")


if __name__ == "__main__":
    main()
