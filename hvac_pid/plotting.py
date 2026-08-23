from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Windows development environment: ensure case-study figures embedded in the
# Chinese report keep their Chinese labels instead of rendering empty squares.
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from .simulator import SimulationResult


SERIES_COLORS = {
    "Bayesian Auto-tune": "#6B7280",
    "Ziegler-Nichols": "#D97706",
    "IMC PI": "#2563EB",
    "FNN Self-tuning PI": "#7C3AED",
    "RL Self-tuning PI": "#DC2626",
}

DISPLAY_NAMES = {
    "Bayesian Auto-tune": "贝叶斯自动整定",
    "Ziegler-Nichols": "Z-N 反应曲线法",
    "IMC PI": "IMC 内模控制",
    "FNN Self-tuning PI": "FNN 在线自整定",
    "RL Self-tuning PI": "RL 在线自整定",
}


def plot_dynamic_comparison(
    results: dict[str, SimulationResult],
    metrics: dict[str, dict[str, float]],
    path: Path,
) -> None:
    fig = plt.figure(figsize=(11.5, 10.4))
    grid = fig.add_gridspec(4, 1, height_ratios=[2.2, 1.4, 1.3, 1.2])
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[3, 0]),
    ]
    axes[1].sharex(axes[0])
    reference = next(iter(results.values()))
    hour = reference.minute / 60.0

    axes[0].plot(hour, reference.setpoint_c, color="#111827", linestyle="--", linewidth=2.0, label="设定温度")
    axes[0].fill_between(
        hour,
        reference.setpoint_c - 0.5,
        reference.setpoint_c + 0.5,
        color="#9CA3AF",
        alpha=0.18,
        label="舒适区间（±0.5 °C）",
    )
    for name, result in results.items():
        axes[0].plot(hour, result.zone_c, label=DISPLAY_NAMES[name], linewidth=1.8, color=SERIES_COLORS[name])
    axes[0].set_ylabel("室内温度（°C）")
    axes[0].set_title("混合动态工况下的温度跟踪对比")
    axes[0].legend(ncol=3, fontsize=9)
    axes[0].grid(alpha=0.22)

    for name, result in results.items():
        axes[1].plot(hour, result.command * 100.0, label=DISPLAY_NAMES[name], linewidth=1.5, color=SERIES_COLORS[name])
    axes[1].set_ylabel("压缩机 PWM 指令（%）")
    axes[1].set_ylim(-2, 102)
    axes[1].set_xlabel("仿真时间（小时）")
    axes[1].grid(alpha=0.22)

    for name, result in results.items():
        axes[2].plot(hour, result.kp, label=f"{DISPLAY_NAMES[name]} Kp", linewidth=1.2, color=SERIES_COLORS[name])
        axes[2].plot(hour, result.ki * 25.0, linestyle="--", linewidth=1.0, color=SERIES_COLORS[name])
    axes[2].set_ylabel("参数值（Ki 为便于同图显示放大 25 倍）")
    axes[2].set_xlabel("仿真时间（小时）")
    axes[2].legend(ncol=2, fontsize=8)
    axes[2].grid(alpha=0.22)

    names = list(results)
    objectives = [metrics[name]["objective"] for name in names]
    colors = [SERIES_COLORS[name] for name in names]
    bars = axes[3].bar(np.arange(len(names)), objectives, color=colors, alpha=0.9)
    axes[3].set_ylabel("综合评价分数（越低越好）")
    axes[3].set_xticks(np.arange(len(names)), [DISPLAY_NAMES[name] for name in names], rotation=12, ha="right")
    axes[3].grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, objectives, strict=True):
        axes[3].text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    axes[3].set_xlim(-0.6, len(names) - 0.4)

    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_training_labels(rows: list[dict[str, float]], path: Path) -> None:
    kp = np.asarray([row["label_kp"] for row in rows])
    ki = np.asarray([row["label_ki"] for row in rows])
    delay = np.asarray([row["actuator_delay_minutes"] for row in rows])
    improvement = np.asarray(
        [(row["zn_objective"] - row["label_objective"]) / max(row["zn_objective"], 1e-9) * 100.0 for row in rows]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    scatter = axes[0].scatter(kp, ki, c=delay, cmap="viridis", s=42, edgecolor="none")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("离线优化得到的 Kp")
    axes[0].set_ylabel("离线优化得到的 Ki（1/分钟）")
    axes[0].set_title("不同训练工况下的贝叶斯优化标签")
    axes[0].grid(alpha=0.2, which="both")
    fig.colorbar(scatter, ax=axes[0], label="执行器延迟（分钟）")

    axes[1].hist(improvement, bins=min(14, max(5, len(rows) // 4)), color="#059669", alpha=0.85)
    axes[1].axvline(float(np.median(improvement)), color="#111827", linestyle="--", label=f"median {np.median(improvement):.1f}%")
    axes[1].set_xlabel("相对 Z-N 的目标函数改善（%）")
    axes[1].set_ylabel("训练工况数量")
    axes[1].set_title("贝叶斯优化标签相对 Z-N 的质量")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_classical_tuning_process(
    identification_history: list[dict[str, float]],
    fit_history: list[dict[str, float]],
    calculation_steps: list[dict[str, object]],
    zn_gains: tuple[float, float],
    imc_gains: tuple[float, float],
    path: Path,
) -> None:
    minute = np.asarray([row["minute"] for row in identification_history])
    measured = np.asarray([row["measured_zone_c"] for row in identification_history])
    y0 = float(identification_history[0]["initial_zone_c"])
    y_inf = float(identification_history[0]["steady_zone_c"])
    fitted_fraction = np.asarray([row["fopdt_fraction"] for row in identification_history])
    fitted = y0 - (y0 - y_inf) * fitted_fraction
    delay = float(identification_history[0]["delay_minutes"])
    t28 = float(identification_history[0]["t28_minutes"])
    t63 = float(identification_history[0]["t63_minutes"])

    fig = plt.figure(figsize=(12, 9.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.34, wspace=0.25)
    ax_response = fig.add_subplot(grid[0, :])
    ax_fit = fig.add_subplot(grid[1, 0])
    ax_formula = fig.add_subplot(grid[1, 1])
    hour = minute / 60.0
    ax_response.plot(hour, measured, color="#2563EB", linewidth=2.0, label="3R2C 虚拟阶跃实测")
    ax_response.plot(hour, fitted, color="#DC2626", linestyle="--", linewidth=1.8, label="有界最小二乘 FOPDT")
    for value, label, color in ((delay, "延迟 L", "#7C3AED"), (t28, "28.3% 时刻", "#059669"), (t63, "63.2% 时刻", "#D97706")):
        if np.isfinite(value):
            ax_response.axvline(value / 60.0, color=color, linestyle=":", linewidth=1.4, label=f"{label}={value:.1f} min")
    ax_response.set_title("步骤 1–10：长时虚拟阶跃 → FOPDT 有界最小二乘辨识")
    ax_response.set_xlabel("阶跃试验时间（小时）")
    ax_response.set_ylabel("室内温度（°C）")
    ax_response.grid(alpha=0.22)
    ax_response.legend(fontsize=8, ncol=2)

    evaluation = np.asarray([row["evaluation"] for row in fit_history], dtype=float)
    rmse = np.asarray([row["rmse_c"] for row in fit_history], dtype=float)
    best_rmse = np.asarray([row["best_rmse_c"] for row in fit_history], dtype=float)
    candidate_k = np.asarray([row["candidate_process_gain_c_per_u"] for row in fit_history], dtype=float)
    ax_fit.plot(evaluation, rmse, color="#94A3B8", marker="o", markersize=2.8, linewidth=0.9, label="本次候选 RMSE")
    ax_fit.plot(evaluation, best_rmse, color="#059669", linewidth=2.1, label="截至当前最小 RMSE")
    ax_fit.set_xlabel("最小二乘残差函数评价次数")
    ax_fit.set_ylabel("温度拟合 RMSE（°C）")
    ax_fit.set_title("FOPDT 拟合不是一步：每次候选均保留")
    ax_fit.grid(alpha=0.22)
    ax_fit.legend(fontsize=8, loc="upper right")
    ax_k = ax_fit.twinx()
    ax_k.plot(evaluation, candidate_k, color="#7C3AED", alpha=0.42, linewidth=1.0, label="候选 K")
    ax_k.set_ylabel("候选过程增益 K（°C/指令）", color="#7C3AED")
    ax_k.tick_params(axis="y", colors="#7C3AED")

    def result_for(method: str, quantity: str) -> str:
        row = next(item for item in calculation_steps if item["method"] == method and item["quantity"] == quantity)
        return str(row["result"])

    formula_text = "\n".join(
        [
            "步骤 11–15｜Z-N 公式与安全限幅",
            result_for("Ziegler-Nichols", "未限幅Kp"),
            result_for("Ziegler-Nichols", "未限幅Ki"),
            f"最终：Kp={zn_gains[0]:.8g}，Ki={zn_gains[1]:.8g}",
            "",
            "步骤 16–20｜IMC 速度选择与参数换算",
            result_for("IMC PI", "闭环速度lambda"),
            result_for("IMC PI", "未限幅Kp"),
            result_for("IMC PI", "未限幅Ki"),
            f"最终：Kp={imc_gains[0]:.8g}，Ki={imc_gains[1]:.8g}",
        ]
    )
    ax_formula.axis("off")
    ax_formula.set_title("公式阶段可以直接计算，但前提是辨识已经完成")
    ax_formula.text(
        0.02,
        0.96,
        formula_text,
        va="top",
        ha="left",
        fontsize=10.5,
        linespacing=1.45,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    fig.suptitle("Z-N / IMC：从名义工作点、阶跃响应、FOPDT 拟合到部署 Kp/Ki", fontsize=15, y=0.99)
    fig.subplots_adjust(top=0.94, bottom=0.07, left=0.08, right=0.93)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bayesian_search_trace(rows: list[dict[str, object]], path: Path) -> None:
    evaluation = np.asarray([float(row["evaluation"]) for row in rows])
    kp = np.asarray([float(row["candidate_kp"]) for row in rows])
    ki = np.asarray([float(row["candidate_ki"]) for row in rows])
    objective = np.asarray([float(row["objective"]) for row in rows])
    best_objective = np.asarray([float(row["best_objective"]) for row in rows])
    best_kp = np.asarray([float(row["best_kp"]) for row in rows])
    best_ki = np.asarray([float(row["best_ki"]) for row in rows])

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10.2))
    scatter = axes[0].scatter(kp, ki, c=evaluation, cmap="viridis", s=70, edgecolor="white", linewidth=0.6)
    axes[0].plot(kp, ki, color="#94A3B8", linewidth=0.8, alpha=0.7)
    axes[0].scatter([best_kp[-1]], [best_ki[-1]], marker="*", s=230, color="#DC2626", label="最终最优参数")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("本轮候选 Kp（对数轴）")
    axes[0].set_ylabel("本轮候选 Ki（1/min，对数轴）")
    axes[0].set_title("搜索轨迹：每个点都代表一组实际仿真候选参数")
    axes[0].grid(alpha=0.2, which="both")
    axes[0].legend()
    fig.colorbar(scatter, ax=axes[0], label="评价序号")

    axes[1].plot(evaluation, objective, "o-", color="#94A3B8", label="本轮候选目标函数")
    axes[1].plot(evaluation, best_objective, color="#059669", linewidth=2.4, label="截至当前的最好结果")
    axes[1].set_xlabel("实际仿真评价次数")
    axes[1].set_ylabel("综合目标函数（越低越好）")
    axes[1].set_title("收敛过程：最优值只能保持或下降")
    axes[1].grid(alpha=0.22)
    axes[1].legend()

    ax_ki = axes[2].twinx()
    kp_line = axes[2].step(evaluation, best_kp, where="post", color="#7C3AED", linewidth=2.0, label="当前最优 Kp")
    ki_line = ax_ki.step(evaluation, best_ki, where="post", color="#D97706", linewidth=2.0, label="当前最优 Ki")
    axes[2].set_xlabel("实际仿真评价次数")
    axes[2].set_ylabel("当前最优 Kp")
    ax_ki.set_ylabel("当前最优 Ki（1/min）")
    axes[2].set_title("整定过程中最优 Kp、Ki 如何逐步被替换")
    axes[2].grid(alpha=0.22)
    axes[2].legend(kp_line + ki_line, ["当前最优 Kp", "当前最优 Ki"], loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fnn_training_trace(rows: list[dict[str, float]], path: Path) -> None:
    sample = np.asarray([row["training_scenario"] for row in rows])
    rmse = np.asarray([row["training_log_rmse"] for row in rows])
    coverage = np.asarray([row["rule_coverage_pct"] for row in rows])
    mean_kp = np.asarray([row["mean_rule_kp"] for row in rows])
    center_kp = np.asarray([row["center_rule_kp"] for row in rows])
    high_kp = np.asarray([row["high_error_rule_kp"] for row in rows])
    mean_ki = np.asarray([row["mean_rule_ki"] for row in rows])
    center_ki = np.asarray([row["center_rule_ki"] for row in rows])
    high_ki = np.asarray([row["high_error_rule_ki"] for row in rows])

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.8))
    ax_cov = axes[0].twinx()
    loss_line = axes[0].plot(sample, rmse, color="#DC2626", linewidth=2.0, label="规则拟合 log-RMSE")
    cov_line = ax_cov.plot(sample, coverage, color="#059669", linewidth=2.0, label="25条规则覆盖率")
    axes[0].set_xlabel("已加入的离线训练工况数")
    axes[0].set_ylabel("拟合误差（越低越好）")
    ax_cov.set_ylabel("规则覆盖率（%）")
    ax_cov.set_ylim(0, 105)
    axes[0].set_title("FNN不是凭空产生规则：标签越多，覆盖和拟合逐渐稳定")
    axes[0].grid(alpha=0.22)
    axes[0].legend(loss_line + cov_line, ["规则拟合 log-RMSE", "规则覆盖率"], loc="best")

    axes[1].plot(sample, mean_kp, label="25条规则平均 Kp", color="#2563EB", linewidth=2.0)
    axes[1].plot(sample, center_kp, label="中心规则 Kp", color="#7C3AED")
    axes[1].plot(sample, high_kp, label="大正误差规则 Kp", color="#D97706")
    axes[1].set_xlabel("已加入的离线训练工况数")
    axes[1].set_ylabel("规则后件 Kp")
    axes[1].set_title("拟合过程中代表性 Kp 规则怎样收敛")
    axes[1].grid(alpha=0.22)
    axes[1].legend()

    axes[2].plot(sample, mean_ki, label="25条规则平均 Ki", color="#2563EB", linewidth=2.0)
    axes[2].plot(sample, center_ki, label="中心规则 Ki", color="#7C3AED")
    axes[2].plot(sample, high_ki, label="大正误差规则 Ki", color="#D97706")
    axes[2].set_xlabel("已加入的离线训练工况数")
    axes[2].set_ylabel("规则后件 Ki（1/min）")
    axes[2].set_title("拟合过程中代表性 Ki 规则怎样收敛")
    axes[2].grid(alpha=0.22)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rl_training_trace(rows: list[dict[str, float]], path: Path) -> None:
    episode = np.asarray([row["episode"] for row in rows])
    reward = np.asarray([row["episode_reward"] for row in rows])
    moving = np.asarray([row["moving_average_reward_20"] for row in rows])
    td = np.asarray([row["mean_abs_td_error"] for row in rows])
    epsilon = np.asarray([row["epsilon"] for row in rows])
    coverage = np.asarray([row["state_coverage_pct"] for row in rows])
    policy_changes = np.asarray([row["greedy_policy_changes"] for row in rows])
    mean_kp = np.asarray([row["mean_kp"] for row in rows])
    mean_ki = np.asarray([row["mean_ki"] for row in rows])

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 12.0))
    axes[0].plot(episode, reward, color="#CBD5E1", linewidth=0.8, label="单回合累计奖励")
    axes[0].plot(episode, moving, color="#2563EB", linewidth=2.1, label="20回合移动平均")
    axes[0].set_ylabel("累计奖励（越高越好）")
    axes[0].set_title("RL训练曲线：观察长期趋势，而不是挑选单个幸运回合")
    axes[0].grid(alpha=0.22)
    axes[0].legend()

    ax_eps = axes[1].twinx()
    td_line = axes[1].plot(episode, td, color="#DC2626", linewidth=1.4, label="平均绝对 TD 误差")
    eps_line = ax_eps.plot(episode, epsilon, color="#7C3AED", linewidth=1.6, label="探索率 ε")
    axes[1].set_ylabel("TD误差")
    ax_eps.set_ylabel("探索率 ε")
    axes[1].set_title("预测误差和探索率随训练推进而变化")
    axes[1].grid(alpha=0.22)
    axes[1].legend(td_line + eps_line, ["平均绝对 TD 误差", "探索率 ε"], loc="best")

    ax_changes = axes[2].twinx()
    cov_line = axes[2].plot(episode, coverage, color="#059669", linewidth=2.0, label="状态覆盖率")
    change_line = ax_changes.plot(episode, policy_changes, color="#D97706", linewidth=1.0, alpha=0.8, label="本回合策略改变格数")
    axes[2].set_ylabel("25状态覆盖率（%）")
    axes[2].set_ylim(0, 105)
    ax_changes.set_ylabel("策略改变格数")
    axes[2].set_title("训练是否可信：既要覆盖状态，也要看策略是否停止频繁翻转")
    axes[2].grid(alpha=0.22)
    axes[2].legend(cov_line + change_line, ["状态覆盖率", "本回合策略改变格数"], loc="best")

    ax_ki = axes[3].twinx()
    kp_line = axes[3].plot(episode, mean_kp, color="#2563EB", linewidth=1.6, label="回合平均 Kp")
    ki_line = ax_ki.plot(episode, mean_ki, color="#D97706", linewidth=1.6, label="回合平均 Ki")
    axes[3].set_xlabel("离线训练回合")
    axes[3].set_ylabel("回合平均 Kp")
    ax_ki.set_ylabel("回合平均 Ki（1/min）")
    axes[3].set_title("RL整定过程中实际使用的 Kp、Ki 也会随动作变化")
    axes[3].grid(alpha=0.22)
    axes[3].legend(kp_line + ki_line, ["回合平均 Kp", "回合平均 Ki"], loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_convergence_overview(
    bayesian_rows: list[dict[str, object]],
    fnn_rows: list[dict[str, float]],
    rl_rows: list[dict[str, float]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    axes[0].plot([float(row["evaluation"]) for row in bayesian_rows], [float(row["best_objective"]) for row in bayesian_rows], color="#059669", linewidth=2.2)
    axes[0].set_title("贝叶斯：最优目标函数")
    axes[0].set_xlabel("仿真评价次数")
    axes[0].set_ylabel("越低越好")
    axes[1].plot([row["training_scenario"] for row in fnn_rows], [row["training_log_rmse"] for row in fnn_rows], color="#7C3AED", linewidth=2.2)
    axes[1].set_title("FNN：规则拟合误差")
    axes[1].set_xlabel("训练工况数")
    axes[1].set_ylabel("越低越好")
    axes[2].plot([row["episode"] for row in rl_rows], [row["moving_average_reward_20"] for row in rl_rows], color="#DC2626", linewidth=2.2)
    axes[2].set_title("RL：20回合平均奖励")
    axes[2].set_xlabel("训练回合")
    axes[2].set_ylabel("越高越好")
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.suptitle("三种 AI 整定/训练的收敛证据（量纲不同，不直接比数值）", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_case_comparisons(cases: dict[str, dict[str, SimulationResult]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    slugs = {"初次快速降温": "initial_cooling", "设定温度突变": "setpoint_step", "持续外界热扰动": "sustained_heat"}
    for case_name, results in cases.items():
        reference = next(iter(results.values()))
        hour = reference.minute / 60.0
        fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.7), sharex=True)
        axes[0].plot(hour, reference.setpoint_c, color="#111827", linestyle="--", linewidth=1.8, label="设定温度")
        axes[0].fill_between(hour, reference.setpoint_c - 0.5, reference.setpoint_c + 0.5, color="#9CA3AF", alpha=0.14, label="±0.5°C 目标带")
        for name, result in results.items():
            axes[0].plot(hour, result.zone_c, label=DISPLAY_NAMES[name], linewidth=1.5, color=SERIES_COLORS[name])
            axes[1].plot(hour, result.command * 100.0, label=DISPLAY_NAMES[name], linewidth=1.3, color=SERIES_COLORS[name])
        axes[0].set_title(f"{case_name}：温度跟踪")
        axes[0].set_ylabel("温度（°C）")
        axes[0].grid(alpha=0.22)
        axes[0].legend(ncol=3, fontsize=8)
        axes[1].set_title("同一工况下的压缩机容量指令")
        axes[1].set_ylabel("PWM 指令（%）")
        axes[1].set_xlabel("仿真时间（小时）")
        axes[1].set_ylim(-2, 102)
        axes[1].grid(alpha=0.22)
        path = output_dir / f"case_{slugs[case_name]}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        generated.append(path)
    return generated


def plot_case_studies(
    cases: dict[str, dict[str, SimulationResult]],
    path: Path,
) -> None:
    """One temperature-response panel per separately defined engineering case."""
    fig, axes = plt.subplots(len(cases), 1, figsize=(11.5, 3.25 * len(cases)), sharex=False)
    axes = np.atleast_1d(axes)
    for axis, (case_name, results) in zip(axes, cases.items(), strict=True):
        reference = next(iter(results.values()))
        hour = reference.minute / 60.0
        axis.plot(hour, reference.setpoint_c, color="#111827", linestyle="--", linewidth=1.8, label="设定值")
        axis.fill_between(hour, reference.setpoint_c - 0.5, reference.setpoint_c + 0.5, color="#9CA3AF", alpha=0.14)
        for name, result in results.items():
            axis.plot(hour, result.zone_c, label=DISPLAY_NAMES[name], linewidth=1.4, color=SERIES_COLORS[name])
        axis.set_title(case_name)
        axis.set_ylabel("室温 [°C]")
        axis.set_xlabel("时间 [h]")
        axis.grid(alpha=0.22)
        axis.legend(ncol=3, fontsize=8)
    fig.suptitle("三类独立极限工况：温度跟踪对比", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
