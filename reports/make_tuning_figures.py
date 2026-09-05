# -*- coding: utf-8 -*-
"""生成《AI-PI 整定实验报告》图 3-9：七算法的「整定过程图」。

只读封存 CSV 画图，不做任何仿真、不改写封存产物：
- 图 3  Z-N   FOPDT 阶跃辨识拟合     archive/outputs_review_v3/classical_tuning_history.csv
- 图 4  IMC   lambda 扫描            archive/outputs_review_v3/imc_lambda_tuning.csv（v3 封存口径）
- 图 5  BO    贝叶斯搜索轨迹          archive/outputs_review_v3/bayesian_search_history.csv
- 图 6  安全BO 风险目标搜索           archive/outputs_advanced_quick/safe_bo_history.csv
- 图 7  LLM   工具调用轨迹            archive/outputs_llm_matrix/qwen-max_std/llm_agent_trace.csv
- 图 8  FNN   训练收敛               archive/outputs_review_v3/fnn_training_history.csv
- 图 9  RL    训练收敛               archive/outputs_review_v3/rl_training_history.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from make_figures import (                                               # noqa: E402
    C_AI_AUTO, C_AI_SELF, C_CLASSICAL, FIG_DIR, GRID, INK, INK2, MUTED,
)

V3 = PROJECT_ROOT / "archive/outputs_review_v3"
ADV = PROJECT_ROOT / "archive/outputs_advanced_quick"
BENCH = PROJECT_ROOT / "archive/outputs_tuning_benchmark"
DEMO = PROJECT_ROOT / "archive/outputs_embedded_demo"
MATRIX_STD = PROJECT_ROOT / "archive/outputs_llm_matrix" / "qwen-max_std"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("生成 " + name)


# ---------------------------------------------------------------- 图 3 Z-N
def fig03_zn_fopdt():
    df = pd.read_csv(V3 / "classical_tuning_history.csv")
    hours = df["minute"] / 60.0
    initial = float(df["initial_zone_c"].iloc[0])
    steady = float(df["steady_zone_c"].iloc[0])
    fitted = initial + (steady - initial) * df["fopdt_fraction"]
    k = float(df["process_gain_c_per_u"].iloc[0])
    tau = float(df["time_constant_minutes"].iloc[0])
    delay = float(df["delay_minutes"].iloc[0])
    t63 = float(df["t63_minutes"].iloc[0])
    rmse_pct = float(df["fit_normalized_rmse_pct"].iloc[0])

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.plot(hours, df["measured_zone_c"], color=C_CLASSICAL, lw=1.4, label="3R2C 全模型阶跃响应（实测）")
    ax.plot(hours, fitted, color=C_AI_AUTO, lw=1.8, ls="--", label="FOPDT 最小二乘拟合")
    ax.axvline(t63 / 60.0, color=MUTED, ls=":", lw=1.2)
    ax.text(t63 / 60.0 + 2, initial - 0.15, "t63 = %.0f min" % t63, fontsize=8.5, color=INK2)
    ax.text(0.985, 0.04,
            "阶跃 +12%% 容量 → 温降 %.2f °C\nK = %.1f °C/指令,  τ = %.0f min,  L = %.0f min\n归一化 RMSE = %.1f%%"
            % (initial - steady, k, tau, delay, rmse_pct),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=GRID))
    ax.set_xlabel("时间（h）")
    ax.set_ylabel("区域温度（°C）")
    _style(ax)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    fig.suptitle("图 3  Z-N / IMC 的数据基础：168 h 虚拟阶跃试验与 FOPDT 辨识", fontsize=13,
                 fontweight="bold", x=0.02, ha="left")
    _save(fig, "fig03_zn_fopdt辨识.png")


# ---------------------------------------------------------------- 图 4 IMC
def fig04_imc_lambda():
    df = pd.read_csv(V3 / "imc_lambda_tuning.csv")
    selected = df[df["selected"] == True]  # noqa: E712
    formula = df[df["formula_default"] == True]  # noqa: E712

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.plot(df["lambda_minutes"], df["mean_training_objective"], "o-", color=C_CLASSICAL,
            lw=1.6, ms=5, label="训练集平均目标")
    ax.plot(df["lambda_minutes"], df["worst_training_objective"], "s--", color=MUTED,
            lw=1.2, ms=4, label="训练集最差目标")
    if not selected.empty:
        row = selected.iloc[0]
        ax.scatter([row["lambda_minutes"]], [row["mean_training_objective"]], s=130,
                   facecolor="none", edgecolor=C_AI_SELF, linewidth=2.2, zorder=5,
                   label="选中 λ* = %.1f min" % row["lambda_minutes"])
    if not formula.empty:
        row = formula.iloc[0]
        ax.scatter([row["lambda_minutes"]], [row["mean_training_objective"]], marker="x",
                   s=90, color=C_AI_AUTO, linewidth=2.2, zorder=5,
                   label="保守公式回退值 λ = %.1f min" % row["lambda_minutes"])
    ax.set_xlabel("闭环时间常数 λ（min）")
    ax.set_ylabel("训练工况目标值（越低越好）")
    _style(ax)
    ax.legend(fontsize=9, frameon=False)
    fig.suptitle("图 4  IMC-λ 整定：只在训练工况扫描 λ，选中后冻结", fontsize=13,
                 fontweight="bold", x=0.02, ha="left")
    _save(fig, "fig04_imc_lambda扫描.png")


# ---------------------------------------------------------------- 图 5 BO
def fig05_bo_search():
    df = pd.read_csv(V3 / "bayesian_search_history.csv")
    x = df["evaluation"]
    phases = df["phase"].astype(str)
    is_init = phases.str.contains("初始")

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.plot(x, df["best_objective"], color=C_AI_AUTO, lw=1.8, label="当前最优目标值")
    ax.scatter(x[is_init], df.loc[is_init, "objective"], color=MUTED, s=42, zorder=5,
               label="初始化点（IMC / 随机）")
    ax.scatter(x[~is_init], df.loc[~is_init, "objective"], facecolor="none",
               edgecolor=C_AI_AUTO, s=52, linewidth=1.8, zorder=5, label="EI 选点评估")
    best_row = df.loc[df["best_objective"].idxmin()]
    ax.annotate("最优 Kp=%.4f, Ki=%.5f" % (best_row["best_kp"], best_row["best_ki"]),
                xy=(len(df), df["best_objective"].iloc[-1]), xytext=(0.62, 0.32),
                textcoords="axes fraction", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
    ax.set_xlabel("评估次数")
    ax.set_ylabel("目标值（越低越好）")
    _style(ax)
    ax.legend(fontsize=9, frameon=False)
    fig.suptitle("图 5  贝叶斯 BO 整定：GP + EI 搜索轨迹（log 增益空间）", fontsize=13,
                 fontweight="bold", x=0.02, ha="left")
    _save(fig, "fig05_bo搜索轨迹.png")


# ---------------------------------------------------------------- 图 6 安全 BO
def fig06_safe_bo():
    df = pd.read_csv(ADV / "safe_bo_history.csv")
    x = df["evaluation"]
    safe = df["safe"].astype(bool)

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.errorbar(x, df["mean_objective"], yerr=df["objective_std"], fmt="none",
                ecolor=GRID, elinewidth=1.6, capsize=3, zorder=2)
    ax.scatter(x[safe], df.loc[safe, "risk_objective"], color=C_AI_AUTO, s=56, zorder=5,
               label="安全候选（风险目标）")
    ax.scatter(x[~safe], df.loc[~safe, "risk_objective"], marker="x", s=64,
               color=MUTED, linewidth=2.0, zorder=5, label="不安全候选（被抑制）")
    ax.plot(x, df["risk_objective"], color=C_AI_AUTO, lw=1.0, alpha=0.45, zorder=3)
    best = df.loc[df["risk_objective"].idxmin()]
    ax.annotate("最优风险目标 %.2f（Kp=%.4f, Ki=%.5f）"
                % (best["risk_objective"], best["candidate_kp"], best["candidate_ki"]),
                xy=(best["evaluation"], best["risk_objective"]), xytext=(0.30, 0.75),
                textcoords="axes fraction", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
    ax.set_xlabel("评估次数")
    ax.set_ylabel("目标值（误差棒 = ±1σ 跨工况标准差）")
    ax.set_xticks(x)
    _style(ax)
    ax.legend(fontsize=9, frameon=False)
    fig.suptitle("图 6  风险感知安全 BO：风险目标 = 均值 + 0.75×标准差", fontsize=13,
                 fontweight="bold", x=0.02, ha="left")
    _save(fig, "fig06_安全bo风险搜索.png")


# ---------------------------------------------------------------- 图 7 LLM
def fig07_llm_trace():
    df = pd.read_csv(MATRIX_STD / "llm_agent_trace.csv")
    x = df["step"]
    tool_label = {"inspect_history": "查看历史", "evaluate_candidate": "评估候选",
                  "finish": "终止决策"}
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.plot(x, df["risk_objective"], "-", color=MUTED, lw=1.2, zorder=2)
    for _, row in df.iterrows():
        accepted = row["accepted"]
        if pd.isna(accepted):
            color, marker, size = INK2, "o", 52
        elif accepted >= 1:
            color, marker, size = C_AI_SELF, "o", 110
        else:
            color, marker, size = C_AI_AUTO, "x", 70
        ax.scatter([row["step"]], [row["risk_objective"]], color=color, marker=marker,
                   s=size, linewidth=2.0, zorder=5)
        ax.annotate(tool_label.get(row["tool"], row["tool"]),
                    (row["step"], row["risk_objective"]),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=8.5, color=INK2)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_AI_SELF, markersize=10,
               label="接受（确定性宿主安全门）"),
        Line2D([0], [0], marker="x", color=C_AI_AUTO, markersize=9, linewidth=2,
               label="拒绝（风险调整改进不足）"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=INK2, markersize=8,
               label="信息步骤（查看历史 / 终止）"),
    ]
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="upper right")
    ax.set_xlabel("Agent 步骤")
    ax.set_ylabel("候选风险目标值（越低越好）")
    ax.set_xticks(x)
    _style(ax)
    fig.suptitle("图 7  LLM Agent 工具调用轨迹（真实 qwen-max 调用）：6 步、2 次候选试验、1 次接受",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    _save(fig, "fig07_llm工具调用轨迹.png")


# ---------------------------------------------------------------- 图 8 FNN
def fig08_fnn_training():
    df = pd.read_csv(V3 / "fnn_training_history.csv")
    x = df["training_scenario"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))

    axes[0].plot(x, df["training_log_rmse"], color=C_AI_SELF, lw=1.8)
    i_min = int(df["training_log_rmse"].idxmin())
    axes[0].scatter([df["training_scenario"].iloc[i_min]], [df["training_log_rmse"].iloc[i_min]],
                    facecolor="none", edgecolor=C_AI_AUTO, s=90, linewidth=2.0, zorder=5)
    axes[0].annotate("最低点后回升：多工况标签冲突\n（见 8 章局限性）",
                     xy=(df["training_scenario"].iloc[i_min], df["training_log_rmse"].iloc[i_min]),
                     xytext=(0.30, 0.30), textcoords="axes fraction", fontsize=8.5,
                     color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
    axes[0].set_xlabel("参与训练的工况数")
    axes[0].set_ylabel("log-RMSE（标签拟合）")
    axes[0].set_title("（a）规则后件拟合误差", fontsize=10.5, color=INK)
    _style(axes[0])

    axes[1].plot(x, df["rule_coverage_pct"], color=C_AI_SELF, lw=1.8)
    axes[1].axhline(80, color=MUTED, ls=":", lw=1.2)
    axes[1].text(x.iloc[-1], 81.5, "验收门下限 80%", fontsize=8.5, color=INK2, ha="right")
    axes[1].set_xlabel("参与训练的工况数")
    axes[1].set_ylabel("规则覆盖率（%）")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("（b）5×5 规则面覆盖率", fontsize=10.5, color=INK)
    _style(axes[1])

    fig.suptitle("图 8  FNN 自整定训练：4464 个 BO 最优增益标签 → 25 条 TSK 规则",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "fig08_fnn训练收敛.png")


# ---------------------------------------------------------------- 图 9 RL
def fig09_rl_training():
    df = pd.read_csv(V3 / "rl_training_history.csv")
    x = df["episode"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))

    axes[0].plot(x, df["moving_average_reward_20"], color=C_AI_SELF, lw=1.8,
                 label="回合奖励（20 回合滑窗）")
    ax2 = axes[0].twinx()
    ax2.plot(x, df["epsilon"], color=MUTED, lw=1.2, ls="--", label="探索率 ε")
    ax2.set_ylabel("探索率 ε", color=MUTED)
    ax2.tick_params(axis="y", colors=MUTED)
    ax2.spines[["top"]].set_visible(False)
    axes[0].set_xlabel("训练回合")
    axes[0].set_ylabel("回合奖励")
    axes[0].set_title("（a）训练奖励收敛与 ε 衰减", fontsize=10.5, color=INK)
    _style(axes[0])
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[0].legend(h1 + h2, l1 + l2, fontsize=8.5, frameon=False, loc="lower right")

    ckpt = df.dropna(subset=["checkpoint_validation_objective"])
    baseline = float(df["validation_baseline_objective"].dropna().iloc[-1])
    learned = float(df["validation_learned_objective"].dropna().iloc[-1])
    axes[1].axhline(baseline, color=C_CLASSICAL, lw=1.4, ls="--",
                    label="IMC 基线 %.2f" % baseline)
    axes[1].plot(ckpt["episode"], ckpt["checkpoint_validation_objective"], "o",
                 color=MUTED, ms=4.5, label="检查点验证目标（每 25 回合）")
    axes[1].plot(x, df["best_checkpoint_validation_objective"], color=C_AI_SELF, lw=1.8,
                 label="运行最优检查点")
    axes[1].annotate("早停选表 %.2f" % learned,
                     xy=(x.iloc[-1], df["best_checkpoint_validation_objective"].iloc[-1]),
                     xytext=(0.55, 0.55), textcoords="axes fraction", fontsize=9, color=INK,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
    axes[1].set_xlabel("训练回合")
    axes[1].set_ylabel("验证集目标值（越低越好）")
    axes[1].set_title("（b）验证集早停选表", fontsize=10.5, color=INK)
    _style(axes[1])
    axes[1].legend(fontsize=8.5, frameon=False, loc="upper right")

    fig.suptitle("图 9  RL 自整定训练：750 回合表格式 Q 学习（36,000 转移）",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "fig09_rl训练收敛.png")


if __name__ == "__main__":
    fig03_zn_fopdt()
    fig04_imc_lambda()
    fig05_bo_search()
    fig06_safe_bo()
    fig07_llm_trace()
    fig08_fnn_training()
    fig09_rl_training()
    print("完成。")
