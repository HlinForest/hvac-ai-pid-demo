"""Generate one self-contained Chinese experiment document per controller."""
from __future__ import annotations

import base64
import csv
import html
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Scenario
from .controllers import FOPDT
from .metrics import calculate_metrics
from .report import _ensure_katex_assets, _katex_head
from .simulator import SimulationResult
from .tuning import TuneResult


matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

DISPLAY_NAMES = {
    "Bayesian Auto-tune": "贝叶斯自动整定（固定参数）",
    "Ziegler-Nichols": "Z-N 反应曲线法",
    "IMC PI": "IMC 内模控制",
    "FNN Self-tuning PI": "FNN 在线自整定",
    "RL Self-tuning PI": "RL 在线自整定",
}

REPORT_FILES = {
    "Ziegler-Nichols": "01_zn_reaction_curve.html",
    "IMC PI": "02_imc.html",
    "Bayesian Auto-tune": "03_bayesian_auto_tune.html",
    "FNN Self-tuning PI": "04_fnn_self_tuning.html",
    "RL Self-tuning PI": "05_rl_self_tuning.html",
}

SCENARIO_SLUGS = {
    "初次快速降温": "initial_cooling",
    "设定温度突变": "setpoint_step",
    "持续外界热扰动": "sustained_heat",
}

METHOD_PROFILES = {
    "Ziegler-Nichols": {
        "直觉": "先让虚拟空调做一次小幅阶跃测试，观察它要等多久才开始降温、降温有多快，再按经验公式给 PI 一组较积极的固定参数。",
        "优点": "计算简单、无需训练数据、实现体积小；对启动和扰动的反应通常较快，适合作为经典基准。",
        "缺点": "经验公式偏激进，容易出现过冷和压缩机指令波动；模型或设备工况变化后需要重新辨识和整定。",
        "适合": "设备特性较稳定、可以做安全阶跃试验、优先考虑实现简单和响应速度的场景。",
    },
    "IMC PI": {
        "直觉": "根据房间反应速度和延迟，主动选择一个希望的闭环速度；宁可慢一点，也要让输出更平滑、稳定裕量更大。",
        "优点": "参数有明确物理意义，抗模型误差能力较好；输出通常平稳，对压缩机友好。",
        "缺点": "保守设置可能导致降温和抗扰恢复过慢；模型辨识不准确时仍需重新调整闭环时间。",
        "适合": "重视平稳、安全和设备寿命，允许温度恢复稍慢的精密空调基础控制。",
    },
    "Bayesian Auto-tune": {
        "直觉": "在代表性虚拟工况里反复试不同 Kp、Ki，用较少的试验次数找到一套综合得分较好的固定参数。",
        "优点": "不依赖手工经验，可同时考虑误差、过冷、能耗和波动；上线后仍是普通固定 PI，部署最简单。",
        "缺点": "优化质量受训练工况覆盖范围和目标函数权重影响；运行中不会主动适应新工况。",
        "适合": "工况变化有限，希望离线自动找参数、上线保持简单可解释的场景。",
    },
    "FNN Self-tuning PI": {
        "直觉": "先把离线优化得到的好参数压缩成 5×5 模糊规则表；运行时根据误差和误差变化，只插值当前附近的 4 条规则。",
        "优点": "能够在线改变 Kp、Ki；规则表小、每次只算 4 条，推理轻量且变化连续。",
        "缺点": "需要离线标签或专家规则；未覆盖区域的效果取决于边界设计，规则表仍需重新训练和验证。",
        "适合": "工况经常变化，又希望在低成本 MCU 上实现在线自适应的场景。",
    },
    "RL Self-tuning PI": {
        "直觉": "在虚拟空调中反复试小幅增减 Kp、Ki，学习不同误差状态下哪一种动作更有利；上线后只查 Q 表，不再随机探索。",
        "优点": "能直接围绕综合控制目标学习增益动作；策略表很小，可加入动作屏蔽、限幅和回退。",
        "缺点": "训练量和场景覆盖要求最高；小样本策略容易在未见工况表现差，解释性也弱于固定 PI 和 FNN。",
        "适合": "有可靠仿真器、能够进行大量离线训练，并愿意投入更多验证工作的研究型方案。",
    },
}

ALGORITHM_EXCERPTS = {
    "Ziegler-Nichols": [("hvac_pid/controllers.py", "class PIController"), ("hvac_pid/controllers.py", "def identify_fopdt_audit"), ("hvac_pid/controllers.py", "def ziegler_nichols_pi")],
    "IMC PI": [("hvac_pid/controllers.py", "class PIController"), ("hvac_pid/controllers.py", "def identify_fopdt_audit"), ("hvac_pid/controllers.py", "def imc_pi")],
    "Bayesian Auto-tune": [("hvac_pid/tuning.py", "class BayesianGainTuner"), ("hvac_pid/tuning.py", "def tune_global_fixed")],
    "FNN Self-tuning PI": [("hvac_pid/ai_controllers.py", "class FNNGainController"), ("hvac_pid/ai_controllers.py", "def train_fnn_rule_table")],
    "RL Self-tuning PI": [("hvac_pid/ai_controllers.py", "class IncrementalRLController"), ("hvac_pid/ai_controllers.py", "def train_offline_q_policy")],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _source_block(relative_path: str, marker: str) -> str:
    """Return the actual top-level function/class source block named by marker."""
    lines = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(marker))
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if re.match(r"^(class |def )", lines[index]):
            end = index
            break
    return "\n".join(lines[start:end]).rstrip()


def _image_data(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _number(value: float) -> str:
    return f"{value:.4g}"


def _html_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _scenario_rows(scenario: Scenario) -> list[dict[str, str]]:
    return [
        {"项目": "仿真时长", "数值": f"{scenario.duration_hours:g} 小时"},
        {"项目": "热模型步长", "数值": f"{scenario.dt_minutes:g} 分钟"},
        {"项目": "初始室温 / 目标温度", "数值": f"{scenario.initial_zone_c:g} °C / {scenario.setpoint_c:g} °C"},
        {"项目": "室外温度", "数值": f"基准 {scenario.outdoor_c:g} °C，昼夜波动振幅 {scenario.outdoor_amplitude_c:g} °C"},
        {"项目": "持续设备热负荷", "数值": f"{scenario.internal_load_w:g} W"},
        {"项目": "开门热扰动", "数值": "未启用" if scenario.door_open_hour is None else f"第 {scenario.door_open_hour:g} 小时开始，{scenario.door_open_duration_minutes:g} 分钟，{scenario.door_open_load_w:g} W"},
        {"项目": "压缩机", "数值": f"容量 {scenario.cooling_capacity_w:g} W；延迟 {scenario.actuator_delay_minutes:g} 分钟；惯性 {scenario.actuator_tau_minutes:g} 分钟"},
    ]


def _gain_explanation(name: str, result: SimulationResult) -> str:
    kp_min, kp_max = float(np.min(result.kp)), float(np.max(result.kp))
    ki_min, ki_max = float(np.min(result.ki)), float(np.max(result.ki))
    if name == "Ziegler-Nichols":
        return f"Kp={kp_min:.4g}、Ki={ki_min:.4g}。两者经过长时虚拟阶跃、FOPDT多次候选拟合、Z-N公式换算和安全限幅后固定；正式运行中不变化。"
    if name == "IMC PI":
        return f"Kp={kp_min:.4g}、Ki={ki_min:.4g}。先辨识FOPDT，再只用训练工况选择闭环速度λ并按IMC公式换算；留出测试中不变化。"
    if name == "Bayesian Auto-tune":
        return f"Kp={kp_min:.4g}、Ki={ki_min:.4g}。离线搜索结束后固定；本次场景中不再根据温度实时改动。"
    update_count = int(np.count_nonzero(np.diff(result.kp, prepend=result.kp[0]) != 0))
    return (
        f"Kp 在 {kp_min:.4g}–{kp_max:.4g} 之间，Ki 在 {ki_min:.4g}–{ki_max:.4g} 之间变化；"
        f"本次轨迹记录到约 {update_count} 次 Kp 改变。参数变化受上下限与单次变化率保护。"
    )


def _metric_rows(result: SimulationResult) -> list[dict[str, str]]:
    metrics = calculate_metrics(result)
    return [
        {"指标": "ITAE（越小越好）", "结果": f"{_number(metrics['itae_c_hour2'])} °C·h²", "说明": "越早消除偏差，分数越低"},
        {"指标": "调节时间", "结果": f"{_number(metrics['settling_time_hour'])} 小时", "说明": "进入并持续处于 ±0.5 °C 区间所需时间"},
        {"指标": "最大过冷", "结果": f"{_number(metrics['max_undershoot_c'])} °C", "说明": "低于设定温度的最大幅度"},
        {"指标": "压缩机容量指令方差", "结果": _number(metrics["compressor_output_variance"]), "说明": "越小表示容量指令越平稳；不是寿命量"},
        {"指标": "制冷能耗代理", "结果": f"{_number(metrics['cooling_energy_kwh'])} kWh", "说明": "按模拟制冷量累积的比较指标"},
        {"指标": "安全回退次数", "结果": str(int(metrics["fallback_events"])), "说明": "AI 参数异常时回到 IMC 的次数"},
    ]


def _plot_one_algorithm(name: str, case_name: str, result: SimulationResult, path: Path) -> None:
    hour = result.minute / 60.0
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10), sharex=True)
    fig.suptitle(f"{DISPLAY_NAMES[name]}｜{case_name}", fontsize=16)
    axes[0].plot(hour, result.zone_c, color="#2563EB", label="室内温度")
    axes[0].plot(hour, result.setpoint_c, color="#111827", linestyle="--", label="设定温度")
    axes[0].fill_between(hour, result.setpoint_c - 0.5, result.setpoint_c + 0.5, color="#9CA3AF", alpha=0.16, label="舒适区间（±0.5 °C）")
    axes[0].set_ylabel("温度（°C）")
    axes[0].legend()
    axes[0].grid(alpha=0.22)
    axes[1].plot(hour, result.command * 100, color="#D97706", label="压缩机容量指令", drawstyle="steps-post")
    axes[1].set_ylabel("PWM 指令（%）")
    axes[1].set_ylim(-2, 102)
    axes[1].legend()
    axes[1].grid(alpha=0.22)
    ax_ki = axes[2].twinx()
    kp_line = axes[2].plot(hour, result.kp, color="#7C3AED", label="Kp")
    ki_line = ax_ki.plot(hour, result.ki, color="#059669", label="Ki")
    axes[2].set_ylabel("Kp")
    ax_ki.set_ylabel("Ki")
    axes[2].set_xlabel("仿真时间（小时）")
    axes[2].legend(kp_line + ki_line, ["Kp", "Ki"], loc="upper right")
    axes[2].grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _gain_summary_rows(case_results: dict[str, dict[str, SimulationResult]], name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case_name, results in case_results.items():
        result = results[name]
        kp_min, kp_max = float(np.min(result.kp)), float(np.max(result.kp))
        ki_min, ki_max = float(np.min(result.ki)), float(np.max(result.ki))
        kp_changes = int(np.count_nonzero(np.abs(np.diff(result.kp)) > 1e-12))
        ki_changes = int(np.count_nonzero(np.abs(np.diff(result.ki)) > 1e-12))
        rows.append(
            {
                "工况": case_name,
                "Kp 范围": f"{kp_min:.5g}" if np.isclose(kp_min, kp_max) else f"{kp_min:.5g}–{kp_max:.5g}",
                "Ki 范围": f"{ki_min:.5g}" if np.isclose(ki_min, ki_max) else f"{ki_min:.5g}–{ki_max:.5g}",
                "运行中更新": "不更新（水平线）" if kp_changes + ki_changes == 0 else f"Kp {kp_changes} 次 / Ki {ki_changes} 次",
            }
        )
    return rows


def _method_profile_table(name: str) -> str:
    profile = METHOD_PROFILES[name]
    rows = [{"项目": key, "说明": value} for key, value in profile.items()]
    return _html_table(rows, ["项目", "说明"])


def _scenario_result_rows(
    case_results: dict[str, dict[str, SimulationResult]], name: str
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case_name, results in case_results.items():
        metrics = calculate_metrics(results[name])
        rows.append(
            {
                "独立场景": case_name,
                "ITAE": _number(metrics["itae_c_hour2"]),
                "调节时间（小时）": _number(metrics["settling_time_hour"]),
                "最大过冷（°C）": _number(metrics["max_undershoot_c"]),
                "容量指令方差": _number(metrics["compressor_output_variance"]),
                "能耗代理（kWh）": _number(metrics["cooling_energy_kwh"]),
            }
        )
    return rows


def _case_rows_for_algorithm(case_rows: list[dict[str, object]], name: str) -> list[dict[str, str]]:
    selected = [row for row in case_rows if row["controller"] == name]
    return [
        {
            "独立场景": str(row["case"]),
            "ITAE": _number(float(row["itae_c_hour2"])),
            "调节时间（小时）": _number(float(row["settling_time_hour"])),
            "最大过冷（°C）": _number(float(row["max_undershoot_c"])),
            "容量指令方差": _number(float(row["compressor_output_variance"])),
        }
        for row in selected
    ]


def _training_section(
    name: str,
    output_dir: Path,
    commissioning_fopdt: FOPDT,
    classical_gains: dict[str, tuple[float, float]],
    global_tune: TuneResult,
    fnn_history: list[dict[str, float]],
    rl_history: list[dict[str, float]],
) -> str:
    if name in ("Ziegler-Nichols", "IMC PI"):
        kp, ki = classical_gains[name]
        with (output_dir / "classical_tuning_steps.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            all_steps = list(csv.DictReader(handle))
        selected_methods = ("共同FOPDT辨识", name, "IMC λ训练集调参") if name == "IMC PI" else ("共同FOPDT辨识", name)
        selected_steps = [row for row in all_steps if row["method"] in selected_methods]
        rows = [
            {
                "步骤": row["step"],
                "模块": row["method"],
                "计算量": row["quantity"],
                "公式或操作": row["formula_or_action"],
                "代入数值": row["substitution"],
                "本步结果": row["result"],
                "为什么做": row["meaning"],
            }
            for row in selected_steps
        ]
        with (output_dir / "fopdt_fit_history.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            fit_history = list(csv.DictReader(handle))
        fit_rows = [
            {
                "评价": row["evaluation"],
                "候选K": f"{float(row['candidate_process_gain_c_per_u']):.7g}",
                "候选τ(min)": f"{float(row['candidate_tau_minutes']):.7g}",
                "候选L(min)": f"{float(row['candidate_delay_minutes']):.7g}",
                "本次RMSE(°C)": f"{float(row['rmse_c']):.7g}",
                "当前最小RMSE(°C)": f"{float(row['best_rmse_c']):.7g}",
            }
            for row in fit_history
        ]
        return f"""
<h2>2. 从初始工况到 Kp、Ki：每一步完整计算</h2>
<div class="answer"><strong>比较协议：</strong>Z-N/IMC 先用 168 h 加速虚拟阶跃辨识 FOPDT。IMC 另在训练工况中只搜索一个鲁棒性参数 λ，Kp/Ti 始终受 IMC 公式约束；进入留出集后冻结。这样才不会把未经任务调节的保守回退参数与已优化控制器直接比较。</div>
<h3>2.1 FOPDT 到底是什么？</h3>
<p>FOPDT 是“一阶惯性 + 纯延迟”的英文缩写。直觉上，它把复杂机房的阶跃曲线压缩成三个数：<strong>K</strong> 表示制冷指令增加后最终能降多少温，<strong>L</strong> 表示命令发出后要空等多久，<strong>τ</strong> 表示开始响应后还要多慢才接近新温度。它只是给控制器算参数的低阶地图，不是 3R2C 物理房间本身。</p>
<img src="data:image/png;base64,{_image_data(output_dir / 'classical_tuning_process.png')}" alt="经典PI阶跃辨识与参数计算过程">
<h3>2.2 FOPDT 最小二乘拟合的全部候选</h3>
<p>下表每一行都是拟合器实际评价过的一组 K、τ、L。旧版 12 h 试验末点只达到最终温降约 59.3%，却被当作 63.2% 交叉点；该问题已经修正。现在交叉点只用于诊断，最终 K、τ、L 来自完整曲线的有界最小二乘拟合。</p>
{_html_table(fit_rows, ["评价", "候选K", "候选τ(min)", "候选L(min)", "本次RMSE(°C)", "当前最小RMSE(°C)"])}
<p class="card">完整机器可读记录：<a href="../fopdt_fit_history.csv">fopdt_fit_history.csv</a>；逐分钟阶跃温度：<a href="../classical_tuning_history.csv">classical_tuning_history.csv</a>。</p>
<h3>2.3 从名义参数到最终增益的逐步代入</h3>
{_html_table(rows, ["步骤", "模块", "计算量", "公式或操作", "代入数值", "本步结果", "为什么做"])}
<p class="card">最终冻结 Kp={kp:.8g}、Ki={ki:.8g}。完整计算表可下载：<a href="../classical_tuning_steps.csv">classical_tuning_steps.csv</a>。进入三个测试工况后不再重新辨识或改变参数。</p>
"""
    if name == "Bayesian Auto-tune":
        first = global_tune.history[0]
        bayes_steps = [
            {"步骤": "1", "操作": "规定安全搜索域", "输入": "工程增益边界", "输出": "Kp∈[0.002,1.5]，Ki∈[1e-5,0.08]"},
            {"步骤": "2", "操作": "对数归一化", "输入": "Kp、Ki 跨多个数量级", "输出": "把 log(Kp)、log(Ki) 映射到 [0,1]²"},
            {"步骤": "3", "操作": "加入 IMC 初始候选", "输入": "第一个训练工况的 FOPDT/IMC", "输出": f"Kp={float(first['candidate_kp']):.7g}，Ki={float(first['candidate_ki']):.7g}"},
            {"步骤": "4", "操作": "生成随机初始探索", "输入": "固定随机种子", "输出": "6 组 log-uniform Kp/Ki 候选"},
            {"步骤": "5", "操作": "真实评价每个候选", "输入": f"每组参数 × {len(fnn_history)} 个离线训练工况", "输出": "逐工况运行 3R2C+PI 并计算综合目标 J"},
            {"步骤": "6", "操作": "汇总全局目标", "输入": "全部训练工况 J", "输出": "Jglobal=mean(Js)，越低越好"},
            {"步骤": "7", "操作": "拟合高斯过程", "输入": "已评价的参数与目标", "输出": "每个未试候选的预测均值 μ 与不确定度 σ"},
            {"步骤": "8", "操作": "计算期望改进 EI", "输入": "每轮 512 个随机候选", "输出": "选择 EI 最大的一组 Kp/Ki 做下一次真实仿真"},
            {"步骤": "9", "操作": "重复代理更新", "输入": f"{global_tune.evaluations - 7} 次 EI 迭代", "输出": "每轮追加真实目标并重新拟合高斯过程"},
            {"步骤": "10", "操作": "冻结当前最好参数", "输入": f"共 {global_tune.evaluations} 次评价", "输出": f"Kp={global_tune.kp:.8g}，Ki={global_tune.ki:.8g}，目标={global_tune.score:.7g}"},
        ]
        rows = [
            {
                "评价次数": str(int(row["evaluation"])),
                "阶段": str(row["phase"]),
                "候选 Kp/Ki": f"{float(row['candidate_kp']):.6g} / {float(row['candidate_ki']):.6g}",
                "本轮目标": f"{float(row['objective']):.5g}",
                "当前最优 Kp/Ki": f"{float(row['best_kp']):.6g} / {float(row['best_ki']):.6g}",
                "当前最优目标": f"{float(row['best_objective']):.5g}",
            }
            for row in global_tune.history
        ]
        return f"""
<h2>2. 离线自动整定：完整贝叶斯搜索轨迹</h2>
<p>下图的每一个点都真正重跑了训练工况集，不是插值出来的装饰曲线。搜索先评价 IMC/随机初始点，再用期望改进 EI 选下一组 Kp/Ki；只有搜索结束后才冻结最优参数。</p>
<h3>2.1 从搜索范围到最终固定增益</h3>
{_html_table(bayes_steps, ["步骤", "操作", "输入", "输出"])}
<img src="data:image/png;base64,{_image_data(output_dir / 'bayesian_search_trace.png')}" alt="贝叶斯优化搜索、收敛和KpKi变化轨迹">
<h3>2.2 每一次真实候选及截至当前最好值</h3>
{_html_table(rows, ["评价次数", "阶段", "候选 Kp/Ki", "本轮目标", "当前最优 Kp/Ki", "当前最优目标"])}
<p class="card">最终冻结 Kp={global_tune.kp:.6g}、Ki={global_tune.ki:.6g}；共进行 {global_tune.evaluations} 次真实批量仿真评价。完整机器可读记录位于 <a href="../bayesian_search_history.csv">bayesian_search_history.csv</a>。</p>
"""
    if name == "FNN Self-tuning PI":
        fnn_steps = [
            {"步骤": "1", "操作": "生成监督标签", "Kp/Ki怎样得到": f"对 {len(fnn_history)} 个训练工况分别运行贝叶斯优化，得到各自较优固定 Kp/Ki"},
            {"步骤": "2", "操作": "重放标签工况", "Kp/Ki怎样得到": "用标签 Kp/Ki 运行3R2C，每5 min记录 e 与误差变化率 ė=Δe/Δt"},
            {"步骤": "3", "操作": "软分配模糊单元", "Kp/Ki怎样得到": "e、ė各分配给相邻两个中心，以双线性权重更新最多4条规则"},
            {"步骤": "4", "操作": "加权累计 log 增益", "Kp/Ki怎样得到": "按隶属权重累计 log(Kp_label)、log(Ki_label) 和有效样本量"},
            {"步骤": "5", "操作": "正则化规则后件", "Kp/Ki怎样得到": "在内部验证集从多种IMC先验权重中选择，抑制稀疏格子的极端外推"},
            {"步骤": "6", "操作": "处理未覆盖规则", "Kp/Ki怎样得到": "没有有效训练权重的单元保留 IMC Kp/Ki，不凭空外推"},
            {"步骤": "7", "操作": "逐批检查拟合", "Kp/Ki怎样得到": "记录规则覆盖率、log-RMSE、最大规则变化量和代表性规则增益"},
            {"步骤": "8", "操作": "冻结 5×5×2 规则表", "Kp/Ki怎样得到": f"本轮最终覆盖 {int(fnn_history[-1]['occupied_rules'])}/25 条规则"},
            {"步骤": "9", "操作": "部署时在线插值", "Kp/Ki怎样得到": "当前 e、ė每轴各激活2个相邻中心，双线性插值得到动态 Kp/Ki，仅计算4条规则"},
            {"步骤": "10", "操作": "安全处理", "Kp/Ki怎样得到": "覆盖至少80%、验证目标不劣于IMC才部署；之后仍经过上下限、单次±10%变化率和异常回退"},
        ]
        rows = [
            {
                "训练工况": str(int(row["training_scenario"])),
                "规则覆盖": f"{int(row['occupied_rules'])}/25 ({row['rule_coverage_pct']:.1f}%)",
                "log-RMSE": f"{row['training_log_rmse']:.5g}",
                "平均 Kp/Ki": f"{row['mean_rule_kp']:.6g} / {row['mean_rule_ki']:.6g}",
                "大误差规则 Kp/Ki": f"{row['high_error_rule_kp']:.6g} / {row['high_error_rule_ki']:.6g}",
            }
            for row in fnn_history
        ]
        return f"""
<h2>2. 离线拟合：25条 FNN 规则如何逐步稳定</h2>
<p>当前 FNN 是可解释的零阶 TSK 规则表，不是用反向传播跑 epoch 的深层网络。因此真实的拟合过程是：逐个加入 BO 标签工况，重放 e/ė 轨迹，更新落入模糊单元的 Kp/Ki 后件。</p>
<h3>2.1 从贝叶斯标签到运行时动态增益</h3>
{_html_table(fnn_steps, ["步骤", "操作", "Kp/Ki怎样得到"])}
<img src="data:image/png;base64,{_image_data(output_dir / 'fnn_training_trace.png')}" alt="FNN规则覆盖、拟合误差与参数收敛过程">
<h3>2.2 每加入一个训练工况后的规则状态</h3>
{_html_table(rows, ["训练工况", "规则覆盖", "log-RMSE", "平均 Kp/Ki", "大误差规则 Kp/Ki"])}
<p class="card">图和表使用了全部 {len(fnn_history)} 个训练工况；完整记录位于 <a href="../fnn_training_history.csv">fnn_training_history.csv</a>。训练后回放目标={fnn_history[-1].get('validation_learned_objective', float('nan')):.6g}，IMC基线={fnn_history[-1].get('validation_baseline_objective', float('nan')):.6g}，部署验收={'通过' if fnn_history[-1].get('deployment_accepted', 0.0) > 0.5 else '拒绝并回退IMC'}。</p>
"""

    rl_steps = [
        {"步骤": "1", "操作": "定义状态", "Kp/Ki怎样得到": "e和误差变化率ė各5档，再加入上一次实际容量的停机/部分/高负荷3档；Q表为5×5×3×9"},
        {"步骤": "2", "操作": "定义动作", "Kp/Ki怎样得到": "9个动作是相对IMC的绝对目标比例：Kp、Ki各取0.75、1.0、1.3"},
        {"步骤": "3", "操作": "安全初始化", "Kp/Ki怎样得到": "未访问状态默认选择IMC不改增益；每回合从IMC和真实ė=0开始"},
        {"步骤": "4", "操作": "选择训练工况", "Kp/Ki怎样得到": "随机抽取一个带噪3R2C工况并随机化初温；不人工伪造状态覆盖"},
        {"步骤": "5", "操作": "ε-greedy选动作", "Kp/Ki怎样得到": "以 ε 随机探索，否则选择当前 Q 最大动作；ε 从0.25降到0.03"},
        {"步骤": "6", "操作": "产生本步增益", "Kp/Ki怎样得到": "先屏蔽冷房间提高增益等无益动作，再把绝对目标比例乘IMC；实际增益每次最多变化±10%"},
        {"步骤": "7", "操作": "推进物理环境", "Kp/Ki怎样得到": "保持动作5 min，通过最低频率、量化、斜率和启停约束后推进带噪3R2C"},
        {"步骤": "8", "操作": "计算奖励", "Kp/Ki怎样得到": "按5 min区间平均惩罚温差、舒适超限、指令变化、能量代理与增益动作"},
        {"步骤": "9", "操作": "更新 Q 值", "Kp/Ki怎样得到": "Q←Q+0.12[r+0.94 max Q′−Q]"},
        {"步骤": "10", "操作": "重复训练", "Kp/Ki怎样得到": f"{len(rl_history)} 回合、约 {int(rl_history[-1]['environment_steps']):,} 个环境步"},
        {"步骤": "11", "操作": "部署验收门", "Kp/Ki怎样得到": "内部验证早停选表；目标若比IMC差2%以上或25热状态覆盖低于80%则拒绝，上线ε=0"},
    ]
    checkpoints = [row for index, row in enumerate(rl_history) if index == 0 or (index + 1) % 25 == 0 or index == len(rl_history) - 1]
    rows = [
        {
            "回合": str(int(row["episode"])),
            "20回合平均奖励": f"{row['moving_average_reward_20']:.5g}",
            "平均 TD 误差": f"{row['mean_abs_td_error']:.5g}",
            "状态覆盖": f"{int(row['visited_states'])}/25 ({row['state_coverage_pct']:.1f}%)",
            "探索率": f"{row['epsilon']:.4g}",
            "回合平均 Kp/Ki": f"{row['mean_kp']:.6g} / {row['mean_ki']:.6g}",
        }
        for row in checkpoints
    ]
    total_steps = int(rl_history[-1]["environment_steps"]) if rl_history else 0
    return f"""
<h2>2. 离线 RL 训练：奖励、TD误差、状态覆盖与参数变化</h2>
<p>RL 不是直接给出一张 Q 表。它在虚拟热环境中训练 {len(rl_history)} 回合，约 {total_steps:,} 个一分钟对象步、36,000次五分钟策略决策；每次选择相对IMC的绝对Kp/Ki目标并更新Q值。图中完整绘制所有回合，表格每25回合抽取一个可读检查点。</p>
<h3>2.1 从 Q 表初始化到部署 Kp/Ki</h3>
{_html_table(rl_steps, ["步骤", "操作", "Kp/Ki怎样得到"])}
<img src="data:image/png;base64,{_image_data(output_dir / 'rl_training_trace.png')}" alt="RL回合奖励、TD误差、覆盖率、策略变化与KpKi训练轨迹">
<h3>2.2 每25回合训练检查点</h3>
{_html_table(rows, ["回合", "20回合平均奖励", "平均 TD 误差", "状态覆盖", "探索率", "回合平均 Kp/Ki"])}
<p class="card">完整逐回合数据位于 <a href="../rl_training_history.csv">rl_training_history.csv</a>。训练后回放目标={rl_history[-1].get('validation_learned_objective', float('nan')):.6g}，IMC基线={rl_history[-1].get('validation_baseline_objective', float('nan')):.6g}，部署验收={'通过' if rl_history[-1].get('deployment_accepted', 0.0) > 0.5 else '拒绝并回退IMC'}。</p>
"""


def write_algorithm_reports(
    output_dir: str | Path,
    case_scenarios: dict[str, Scenario],
    case_results: dict[str, dict[str, SimulationResult]],
    case_rows: list[dict[str, object]],
    *,
    commissioning_fopdt: FOPDT,
    classical_gains: dict[str, tuple[float, float]],
    global_tune: TuneResult,
    fnn_history: list[dict[str, float]],
    rl_history: list[dict[str, float]],
) -> dict[str, str]:
    """Write five audit-friendly reports, each covering all three engineering cases."""
    output_dir = Path(output_dir)
    report_dir = output_dir / "algorithm_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    _ensure_katex_assets(output_dir / "engineering_report.html")
    simulation_code = _source_block("hvac_pid/simulator.py", "def simulate")
    plant_code = _source_block("hvac_pid/plant.py", "class ThermalPlant3R2C")
    generated: dict[str, str] = {}
    algorithm_names = list(next(iter(case_results.values())).keys())
    for name in algorithm_names:
        image_paths: dict[str, Path] = {}
        for case_name, results in case_results.items():
            image_path = report_dir / f"{Path(REPORT_FILES[name]).stem}_{SCENARIO_SLUGS[case_name]}.png"
            _plot_one_algorithm(name, case_name, results[name], image_path)
            image_paths[case_name] = image_path
        algorithm_code = "\n\n".join(
            f"# {relative_path} — {marker}\n{_source_block(relative_path, marker)}"
            for relative_path, marker in ALGORITHM_EXCERPTS[name]
        )
        case_table = _scenario_result_rows(case_results, name)
        gain_table = _gain_summary_rows(case_results, name)
        training_section = _training_section(name, output_dir, commissioning_fopdt, classical_gains, global_tune, fnn_history, rl_history)
        scenario_sections: list[str] = []
        for index, (case_name, scenario) in enumerate(case_scenarios.items(), start=1):
            result = case_results[case_name][name]
            scenario_sections.append(
                f"""
<section class="experiment">
<h2>{index + 4}. 工况 {index}：{html.escape(case_name)}</h2>
<h3>仿真设置</h3>{_html_table(_scenario_rows(scenario), ["项目", "数值"])}
<h3>Kp、Ki 在这一工况怎样变化？</h3><div class="card">{html.escape(_gain_explanation(name, result))}</div>
<img src="data:image/png;base64,{_image_data(image_paths[case_name])}" alt="{html.escape(DISPLAY_NAMES[name])} {html.escape(case_name)}波形">
<h3>这一工况的效果评估</h3>{_html_table(_metric_rows(result), ["指标", "结果", "说明"])}
</section>"""
            )
        output_file = report_dir / REPORT_FILES[name]
        document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(DISPLAY_NAMES[name])} 实验文档</title>
{_katex_head('../')}
<style>
@page {{ size:A4; margin:15mm; }} body {{ font-family:"Microsoft YaHei",Arial,sans-serif; color:#172033; max-width:1080px; margin:36px auto; line-height:1.6; }}
h1 {{ color:#0f3d67; border-bottom:3px solid #36a3d9; padding-bottom:10px; }} h2 {{ color:#0f3d67; margin-top:30px; }}
.card {{ background:#f3f8fc; border-left:4px solid #36a3d9; padding:14px 18px; border-radius:5px; }}
.answer {{ background:#fff7df; border:1px solid #efd797; padding:14px 18px; border-radius:5px; }}
.experiment {{ border-top:2px solid #d7e0e9; margin-top:36px; padding-top:4px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }} th {{ background:#0f3d67; color:white; }} th,td {{ padding:8px; text-align:left; border:1px solid #d7e0e9; }} tr:nth-child(even) {{ background:#f6f9fc; }}
img {{ width:100%; border:1px solid #d7e0e9; margin:12px 0; }} pre {{ white-space:pre-wrap; word-break:break-word; background:#101827; color:#E5E7EB; padding:14px; border-radius:6px; font-size:12px; line-height:1.45; }}
</style></head><body>
<h1>{html.escape(DISPLAY_NAMES[name])}：三工况独立仿真实验文档</h1>
<p>本文件只讨论这一种算法，并分别给出“初次快速降温”“设定温度突变”“持续外界热扰动”三组独立实验。每组都有自己的仿真设置、温度/PWM/Kp/Ki 波形与效果指标。</p>
<h2>1. 这套方法的直觉、优点和缺点</h2>{_method_profile_table(name)}
{training_section}
<h2>3. 整定/训练阶段与部署运行阶段必须分开</h2>
<div class="answer">Z-N、IMC 和贝叶斯 PI 在独立安装调试阶段得到参数，进入三个测试工况后全部冻结，所以运行图中 Kp/Ki 是水平线。FNN 和 RL 也先离线训练并冻结规则表/策略，但它们部署时会根据 e、Δe 输出受限的动态 Kp/Ki。测试工况不参与整定或训练。</div>
{_html_table(gain_table, ["工况", "Kp 范围", "Ki 范围", "运行中更新"])}
<h2>4. 三工况结果总览</h2>{_html_table(case_table, ["独立场景", "ITAE", "调节时间（小时）", "最大过冷（°C）", "容量指令方差", "能耗代理（kWh）"])}
{''.join(scenario_sections)}
<h2>8. 仿真函数：房间与压缩机如何被推进？</h2>
<p>每一步读取环境和负荷，PI 计算 PWM，压缩机经过延迟/惯性后提供制冷量，最后更新空气与墙体温度。以下是实际使用的热模型代码与仿真循环代码。</p>
<h3>热模型函数</h3><pre>{html.escape(plant_code)}</pre>
<h3>仿真循环函数</h3><pre>{html.escape(simulation_code)}</pre>
<h2>9. 本算法的实际代码</h2><pre>{html.escape(algorithm_code)}</pre>
<p>解释原则：ITAE 和调节时间越小，说明越早回到设定温度；最大过冷越小，表示不会降得太低；容量指令方差越小，命令越平稳，但它不是寿命量。不能只按一项排名。</p>
</body></html>"""
        output_file.write_text(document, encoding="utf-8")
        generated[name] = str(output_file.relative_to(output_dir)).replace("\\", "/")
    return generated
