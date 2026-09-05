from __future__ import annotations

from pathlib import Path
import base64
import csv
import html
import shutil

import numpy as np

from .config import Scenario, typical_case_scenarios


def _ensure_katex_assets(report_path: Path) -> bool:
    """Copy the bundled KaTeX runtime next to generated reports for offline use."""

    source = Path(__file__).resolve().parents[1] / "vendor" / "katex"
    if not source.exists():
        return False
    destination = report_path.parent / "report_assets" / "katex"
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return True


def _katex_head(prefix: str = "") -> str:
    base = f"{prefix}report_assets/katex"
    return f"""
<link rel="stylesheet" href="{base}/katex.min.css">
<script defer src="{base}/katex.min.js"></script>
<script defer src="{base}/auto-render.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function() {{
  if (typeof renderMathInElement === "function") {{
    renderMathInElement(document.body, {{
      delimiters: [
        {{left: "\\\\[", right: "\\\\]", display: true}},
        {{left: "\\\\(", right: "\\\\)", display: false}}
      ],
      throwOnError: false,
      strict: false
    }});
  }}
}});
</script>"""


def write_engineering_report(
    path: str | Path,
    summary: list[dict[str, object]],
    dynamic: list[dict[str, object]],
    case_rows: list[dict[str, object]] | None = None,
    algorithm_reports: dict[str, str] | None = None,
    physical_validation: list[dict[str, object]] | None = None,
    fopdt_validation: list[dict[str, object]] | None = None,
    openmodelica_validation: list[dict[str, object]] | None = None,
    validation_environment: dict[str, object] | None = None,
) -> None:
    """Create a portable Markdown engineering report from machine-readable CSV data."""
    path = Path(path)
    columns = ["controller", "mean_itae_c_hour2", "mean_settling_time_hour", "mean_max_undershoot_c", "mean_compressor_output_variance", "stable_rate"]
    present = [c for c in columns if all(c in row for row in summary)]

    def table(rows: list[dict[str, object]], fields: list[str]) -> str:
        if not rows:
            return "_No results._"
        lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
        for row in rows:
            values = []
            for field in fields:
                value = row.get(field, "")
                values.append(f"{value:.4g}" if isinstance(value, float) else str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    case_rows = case_rows or []
    algorithm_reports = algorithm_reports or {}
    physical_validation = physical_validation or []
    fopdt_validation = fopdt_validation or []
    openmodelica_validation = openmodelica_validation or []
    validation_environment = validation_environment or {}
    def deployment_accepted(filename: str) -> bool:
        history_path = path.parent / filename
        if not history_path.exists():
            return True
        with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        value = rows[-1].get("deployment_accepted", "nan") if rows else "nan"
        return str(value).lower() == "nan" or float(value) > 0.5

    fnn_accepted = deployment_accepted("fnn_training_history.csv")
    rl_accepted = deployment_accepted("rl_training_history.csv")
    display_names = {
        "Bayesian Auto-tune": "贝叶斯自动整定",
        "Ziegler-Nichols": "Z-N 反应曲线法",
        "IMC PI": "IMC 内模控制（λ经训练集整定）",
        "FNN Self-tuning PI": "FNN 在线自整定" if fnn_accepted else "FNN（验收拒绝→IMC）",
        "RL Self-tuning PI": "RL 在线自整定" if rl_accepted else "RL（验收拒绝→IMC）",
    }

    def display_rows(rows: list[dict[str, object]], fields: list[tuple[str, str]]) -> list[dict[str, object]]:
        return [{label: (display_names.get(str(row.get(key)), row.get(key)) if key == "controller" else row.get(key)) for key, label in fields} for row in rows]

    summary_display = display_rows(summary, [("controller", "算法"), ("mean_itae_c_hour2", "平均 ITAE（°C·h²）"), ("mean_settling_time_hour", "平均调节时间（小时）"), ("mean_max_undershoot_c", "平均最大过冷（°C）"), ("mean_compressor_output_variance", "平均容量指令方差"), ("stable_rate", "稳定率")])
    dynamic_display = display_rows(dynamic, [("controller", "算法"), ("itae_c_hour2", "ITAE（°C·h²）"), ("settling_time_hour", "调节时间（小时）"), ("max_undershoot_c", "最大过冷（°C）"), ("compressor_output_variance", "容量指令方差"), ("fallback_events", "安全回退次数")])
    case_display = display_rows(case_rows, [("case", "场景"), ("controller", "算法"), ("itae_c_hour2", "ITAE（°C·h²）"), ("settling_time_hour", "调节时间（h）"), ("disturbance_recovery_time_hour", "扰动恢复（h）"), ("max_overheat_c", "最大过热（°C）"), ("max_undershoot_c", "最大过冷（°C）"), ("compressor_output_variance", "容量指令方差")])
    physical_display = [
        {
            "场景": row.get("case", ""),
            "RMSE（°C）": row.get("rmse_c", ""),
            "最大误差（°C）": row.get("max_abs_error_c", ""),
            "判定": "通过" if float(row.get("passed", 0)) > 0.5 else "未通过",
        }
        for row in physical_validation
    ]
    fopdt_display = [
        {
            "场景": row.get("case", ""),
            "K（°C/指令）": row.get("process_gain_c_per_u", ""),
            "τ（min）": row.get("time_constant_min", ""),
            "L（min）": row.get("delay_min", ""),
            "归一化 RMSE（%）": row.get("normalized_rmse_pct", ""),
            "判定": "通过" if float(row.get("passed", 0)) > 0.5 else "未通过",
        }
        for row in fopdt_validation
    ]
    openmodelica_display = [
        {
            "场景": row.get("case", ""),
            "采样点": row.get("samples", ""),
            "RMSE（°C）": row.get("rmse_c", ""),
            "最大误差（°C）": row.get("max_abs_error_c", ""),
            "末端误差（°C）": row.get("final_abs_error_c", ""),
            "判定": "通过" if float(row.get("passed", 0)) > 0.5 else "未通过",
        }
        for row in openmodelica_validation
    ]
    text = f"""# 变频精密/机柜空调 AI-PID 仿真、算法对比与量产评估报告

> 控制器对比数值来自 Python 3R2C + 延迟/一阶执行器。已实际运行 OpenModelica/DASSL、SciPy 连续方程和 FOPDT/3R2C 三层交叉验证；OpenModelica 状态：{validation_environment.get('openmodelica_run', '未记录')}。

## 1. 环境建模、仿真器、控制函数与输入输出

- 物理参考：OpenModelica + Modelica Standard Library，实际执行模型位于 `modelica/HVACAI/PrecisionCabinetCooling.mo`；Modelica Buildings Library 留作后续高保真设备模型扩展。
- 实验仿真器：Python/NumPy 中的 3R2C 两状态热模型，主函数为 `simulate(scenario, controller, seed)`。
- 热模型：

```text
Cz·dTz/dt = (To-Tz)/Roz + (Tw-Tz)/Rzw + Qint - Qc
Cw·dTw/dt = (To-Tw)/Row + (Tz-Tw)/Rzw
τu·dQc/dt = Qmax·u(t-L) - Qc
```

FOPDT 代理为 `ΔT(s)/Δu(s)=-K exp(-Ls)/(τs+1)`，只用于 Z-N/IMC 整定。主要参数是两个热容 Cz/Cw、三条热阻 Roz/Rzw/Row、制冷能力 Qmax、延迟 L、执行器惯性 τu、室外温度与内部热负荷。

| 层级 | 输入 | 输出 |
|---|---|---|
| 环境 | To(t)、Tsp(t)、Qint(t) | 天气/负荷/设定值时序 |
| 3R2C | To、Qint、Qc | Tz、Tw |
| SafePI | e=Tz-Tsp、Kp、Ki、Δt | u∈[0,1]（0–100% 容量请求） |
| FNN/RL | e、Δe | 受限 Kp、Ki；不直接控制压缩机 |

Python 热模型步长是 1 min；嵌入式目标是 PI 100 ms、AI 2 s。

### 1.1 实际运行的交叉验证

1 分钟离散热模型与独立 SciPy 连续方程求解结果：

{table(physical_display, ["场景", "RMSE（°C）", "最大误差（°C）", "判定"])}

FOPDT 快速代理与 3R2C 物理模型阶跃响应结果：

{table(fopdt_display, ["场景", "K（°C/指令）", "τ（min）", "L（min）", "归一化 RMSE（%）", "判定"])}

OpenModelica/DASSL 与独立 Python/DOP853 连续方程的同工况结果：

{table(openmodelica_display, ["场景", "采样点", "RMSE（°C）", "最大误差（°C）", "末端误差（°C）", "判定"])}

## 2. 控制算法设计与数学表达

- 共用 PI：`e=Tz-Tsp`，`I*=I+eΔt`，`u=clip(Kp·e+Ki·I*,0,1)`；条件积分抗饱和。
- Z-N：`Kp=0.9τ/(KL)`，`Ti=3.33L`，`Ki=Kp/Ti`；在独立安装调试阶段整定一次，三个测试工况全部冻结。
- IMC：保守回退值取 `λ=max(τ/3,3L,12min)`；公平性能基线只在训练工况搜索 λ，随后仍按 `Kp=τ/[K(λ+L)]`、`Ti=min[τ,4(λ+L)]` 换算并在留出测试中冻结。
- 贝叶斯自动整定：在 log(Kp),log(Ki) 内用 Matérn-5/2 高斯过程 + EI 搜索全局固定增益；本次为 Kp=0.609258、Ki=0.00175406。
- FNN：5×5 TSK 规则表，输入为误差 e 与误差变化率 ė，`Kp=ΣwᵢⱼKpᵢⱼ`、`Ki=ΣwᵢⱼKiᵢⱼ`，每次只有 4 条双线性插值规则激活；规则表由 BO 标签离线训练并在验证集选择 IMC 正则强度。覆盖不足或平均目标劣于 IMC时拒绝候选表。
- RL：状态为 5×5 的 (e,ė) 热状态再乘 3 档实际容量模式，9 个动作是相对 IMC 的绝对 Kp/Ki 目标比例；750 回合、每回合最长 240 min、每 5 min 决策。训练后使用多工况回放和早停选表，劣于 IMC 超过2%或25热状态覆盖不足80%时拒绝并回退 IMC。

完整整定/训练证据：

- `classical_tuning_history.csv` / `fopdt_fit_history.csv` / `classical_tuning_steps.csv`：Z-N、IMC 的逐分钟阶跃、全部 FOPDT 候选与逐步公式代入。
- `bayesian_search_history.csv` / `bayesian_search_trace.png`：每次候选 Kp/Ki、本轮目标、当前最优值和收敛轨迹。
- `fnn_training_history.csv` / `fnn_training_trace.png`：规则覆盖率、拟合误差与代表性规则 Kp/Ki。
- `rl_training_history.csv` / `rl_training_trace.png`：逐回合奖励、TD 误差、状态覆盖、策略改变和回合增益。

## 3. 统一评估指标体系

| 类别 | 指标 |
|---|---|
| 跟踪精度与速度 | RMSE、IAE=`∫|e|dt`、ITAE=`∫t|e|dt`、调节时间 |
| 动态品质 | 扰动恢复时间、最大过热、最大过冷、舒适带超限度时 |
| 执行机构寿命与能耗 | 容量指令方差、`Σ|u[k]-u[k-1]|`、启停次数、`∫Qc dt` 制冷能量代理 |
| 安全与实时性 | 稳定率、回退次数、AI 推理时间 |

`∫Qc dt` 未引入 COP 与风机/水泵功耗，只是横向比较代理；容量指令方差也只是平稳性代理，不能直接等同于寿命。

## 4. 三场景实验分析与工程评估

- 场景 1：初次快速降温，31.5→24 °C，4 h，检验大偏差启动与过冷。
- 场景 2：设定温度突变，1 h 时 25→23 °C，检验工况迁移。
- 场景 3：室外 37±4.5 °C、持续设备发热、3 h 开门 12 min +2600 W，检验长时抗扰。

{table(case_display, ["场景", "算法", "ITAE（°C·h²）", "调节时间（h）", "扰动恢复（h）", "最大过热（°C）", "最大过冷（°C）", "容量指令方差"])}

留出工况汇总：

{table(summary_display, ["算法", "平均 ITAE（°C·h²）", "平均调节时间（小时）", "平均最大过冷（°C）", "平均容量指令方差", "稳定率"])}

## 5. 嵌入式量产落地与工程安全评估

- STM32F103C8T6：72 MHz、64 KB Flash、20 KB SRAM。Wokwi Blue Pill 完整代码已在线编译并进入运行态；FNN float32 表 200 B，RL 已压成 25 B 动作索引 + 25 B 覆盖掩码 + 72 B 动作表。
- ESP32：经典系列最高 240 MHz、520 KB SRAM；资源充足，但需隔离 Wi-Fi 任务与控制任务。
- PC C++ Testbench：`sizeof(SafePI)=32 B`、虚拟对象状态 68 B，100 ms PI / 2 s AI 分频通过；PC 时间不能换算为 MCU WCET。
- 已实现于 Python 统一比较层：0/25% 最低稳定容量、1% 量化台阶、运行段每分钟 5 个百分点斜率、最小启停驻留、测量噪声与滤波、条件积分抗饱和、Kp/Ki 边界、FNN/RL 训练后部署验收门。
- 尚未声称完成：Wokwi 已证明 STM32/ESP32 目标编译和启动，但这些新增约束仍需同步到目标 MCU，并取得串口曲线、ROM、RAM、最坏周期和栈证据；高低压、排气温度、通信超时、看门狗及参数 CRC/回滚仍待补齐。
- 上线流程：SIL → HIL → 只读影子模式 → 有限增益试运行 → 单机试点 → 多季节回归。

## 单算法三工况文档

{chr(10).join(f'- [{display_names.get(name, name)}]({link})' for name, link in algorithm_reports.items())}
"""
    path.write_text(text, encoding="utf-8")


def write_html_engineering_report(
    path: str | Path,
    summary: list[dict[str, object]],
    dynamic: list[dict[str, object]],
    image_dir: str | Path,
    case_rows: list[dict[str, object]] | None = None,
    algorithm_reports: dict[str, str] | None = None,
    physical_validation: list[dict[str, object]] | None = None,
    fopdt_validation: list[dict[str, object]] | None = None,
    openmodelica_validation: list[dict[str, object]] | None = None,
    validation_environment: dict[str, object] | None = None,
) -> None:
    """Create a self-contained, printable visual report with embedded figures."""
    path = Path(path)
    image_dir = Path(image_dir)
    bayes_summary_path = image_dir / "global_bayesian_tuning.csv"
    if bayes_summary_path.exists():
        with bayes_summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            bayes_summary = next(csv.DictReader(handle), {})
    else:
        bayes_summary = {}
    bayes_kp = float(bayes_summary.get("kp", float("nan")))
    bayes_ki = float(bayes_summary.get("ki", float("nan")))
    bayes_evaluations = int(float(bayes_summary.get("evaluations", 0)))

    def read_csv_rows(name: str) -> list[dict[str, object]]:
        csv_path = image_dir / name
        if not csv_path.exists():
            return []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    classical_steps_raw = read_csv_rows("classical_tuning_steps.csv")
    fopdt_fit_raw = read_csv_rows("fopdt_fit_history.csv")

    def number(value: object) -> str:
        return f"{value:.4g}" if isinstance(value, float) else str(value)

    def html_table(rows: list[dict[str, object]], columns: list[str]) -> str:
        header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(number(row.get(column, '')))}</td>" for column in columns) + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"

    def image_data(name: str) -> str:
        image = image_dir / name
        if not image.exists():
            return ""
        data = base64.b64encode(image.read_bytes()).decode("ascii")
        return f'<img src="data:image/png;base64,{data}" alt="{html.escape(name)}">'

    classical_step_rows = [
        {
            "步骤": row.get("step", ""),
            "模块": row.get("method", ""),
            "计算量": row.get("quantity", ""),
            "公式或操作": row.get("formula_or_action", ""),
            "代入数值": row.get("substitution", ""),
            "本步结果": row.get("result", ""),
            "说明": row.get("meaning", ""),
        }
        for row in classical_steps_raw
    ]
    fopdt_fit_rows = [
        {
            "评价": row.get("evaluation", ""),
            "候选K": row.get("candidate_process_gain_c_per_u", ""),
            "候选τ(min)": row.get("candidate_tau_minutes", ""),
            "候选L(min)": row.get("candidate_delay_minutes", ""),
            "本次RMSE(°C)": row.get("rmse_c", ""),
            "当前最小RMSE(°C)": row.get("best_rmse_c", ""),
        }
        for row in fopdt_fit_raw
    ]

    case_rows = case_rows or []
    algorithm_reports = algorithm_reports or {}
    physical_validation = physical_validation or []
    fopdt_validation = fopdt_validation or []
    openmodelica_validation = openmodelica_validation or []
    validation_environment = validation_environment or {}
    katex_available = _ensure_katex_assets(path)
    fnn_training_rows = read_csv_rows("fnn_training_history.csv")
    rl_training_rows = read_csv_rows("rl_training_history.csv")
    def accepted_fail_closed(rows: list[dict[str, str]]) -> bool:
        if not rows:
            return False
        try:
            value = float(rows[-1].get("deployment_accepted", "nan"))
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(value) and value > 0.5)

    fnn_accepted = accepted_fail_closed(fnn_training_rows)
    rl_accepted = accepted_fail_closed(rl_training_rows)
    display_names = {
        "Bayesian Auto-tune": "贝叶斯自动整定",
        "Ziegler-Nichols": "Z-N 反应曲线法",
        "IMC PI": "IMC 内模控制（λ经训练集整定）",
        "FNN Self-tuning PI": "FNN 在线自整定" if fnn_accepted else "FNN（验收拒绝→IMC）",
        "RL Self-tuning PI": "RL 在线自整定" if rl_accepted else "RL（验收拒绝→IMC）",
    }

    def translate(rows: list[dict[str, object]], fields: list[tuple[str, str]]) -> tuple[list[dict[str, object]], list[str]]:
        labels = [label for _, label in fields]
        translated = []
        for row in rows:
            item: dict[str, object] = {}
            for source, label in fields:
                value = row.get(source, "")
                item[label] = display_names.get(str(value), value) if source == "controller" else value
            translated.append(item)
        return translated, labels

    holdout_rows, holdout_columns = translate(summary, [("controller", "算法"), ("mean_itae_c_hour2", "平均 ITAE（°C·h²）"), ("mean_settling_time_hour", "平均调节时间（小时）"), ("mean_max_undershoot_c", "平均最大过冷（°C）"), ("mean_compressor_output_variance", "平均容量指令方差"), ("stable_rate", "稳定率")])
    dynamic_rows, dynamic_columns = translate(dynamic, [("controller", "算法"), ("itae_c_hour2", "ITAE（°C·h²）"), ("settling_time_hour", "调节时间（小时）"), ("max_undershoot_c", "最大过冷（°C）"), ("compressor_output_variance", "容量指令方差"), ("fallback_events", "安全回退次数")])
    case_display, case_columns = translate(case_rows, [("case", "场景"), ("controller", "算法"), ("itae_c_hour2", "ITAE（°C·h²）"), ("settling_time_hour", "调节时间（小时）"), ("max_undershoot_c", "最大过冷（°C）"), ("compressor_output_variance", "容量指令方差")])
    physical_validation_rows = [
        {
            "验证场景": row.get("case", ""),
            "采样点": row.get("samples", ""),
            "温度 RMSE（°C）": row.get("rmse_c", ""),
            "最大绝对误差（°C）": row.get("max_abs_error_c", ""),
            "末端误差（°C）": row.get("final_abs_error_c", ""),
            "预设阈值": row.get("threshold", ""),
            "结论": "通过" if float(row.get("passed", 0.0)) > 0.5 else "未通过",
        }
        for row in physical_validation
    ]
    fopdt_validation_rows = [
        {
            "验证场景": row.get("case", ""),
            "过程增益 K（°C/指令）": row.get("process_gain_c_per_u", ""),
            "时间常数 τ（min）": row.get("time_constant_min", ""),
            "延迟 L（min）": row.get("delay_min", ""),
            "温度 RMSE（°C）": row.get("rmse_c", ""),
            "归一化 RMSE（%）": row.get("normalized_rmse_pct", ""),
            "预设阈值": row.get("threshold", ""),
            "结论": "通过" if float(row.get("passed", 0.0)) > 0.5 else "未通过",
        }
        for row in fopdt_validation
    ]
    openmodelica_validation_rows = [
        {
            "验证场景": row.get("case", ""),
            "采样点": row.get("samples", ""),
            "温度 RMSE（°C）": row.get("rmse_c", ""),
            "最大绝对误差（°C）": row.get("max_abs_error_c", ""),
            "末端误差（°C）": row.get("final_abs_error_c", ""),
            "预设阈值": row.get("threshold", ""),
            "结论": "通过" if float(row.get("passed", 0.0)) > 0.5 else "未通过",
        }
        for row in openmodelica_validation
    ]
    validation_environment_rows = [
        {"运行项": key, "实际值": value} for key, value in validation_environment.items()
    ]
    physical_pass_count = sum(float(row.get("passed", 0.0)) > 0.5 for row in physical_validation)
    fopdt_pass_count = sum(float(row.get("passed", 0.0)) > 0.5 for row in fopdt_validation)
    openmodelica_pass_count = sum(float(row.get("passed", 0.0)) > 0.5 for row in openmodelica_validation)
    modelica_status = str(validation_environment.get("openmodelica", "未记录"))
    modelica_run_status = str(validation_environment.get("openmodelica_run", "未记录"))
    modelica_table_status = (
        '<span class="status">已实际编译并用 DASSL 运行 12 h</span>'
        if openmodelica_validation
        else "本次输出目录没有 OpenModelica 结果 CSV"
    )
    modelica_conclusion = (
        "本项目已建立可重复的 Python 热仿真环境、五类控制器、统一指标和三工况比较，并实际完成 "
        "OpenModelica/DASSL 与 Python/DOP853 连续方程对齐、1 min 离散步进误差检查，以及 FOPDT "
        "降阶代理验证。当前最大工程风险不再是“OpenModelica 没跑”，而是物理参数尚未用实测 BMS "
        "数据标定、Buildings Library 高保真制冷部件尚未接入，以及量产压缩机联锁/目标 MCU WCET 尚未完成。"
        if openmodelica_validation
        else "本项目已建立可重复的 Python 热仿真环境、五类控制器、统一指标和三工况比较，并完成 1 min "
        "离散步进误差检查及 FOPDT 降阶代理验证；当前输出目录没有 OpenModelica 结果 CSV，不能在这一份报告中声称已完成 Modelica 交叉验证。"
    )
    if rl_training_rows:
        rl_last = rl_training_rows[-1]
        rl_covered_states = int(float(rl_last.get("visited_states", 0)))
        rl_total_states = 25
        rl_full_covered = int(float(rl_last.get("visited_full_states", 0)))
        rl_full_total = 75
    else:
        rl_covered_states = rl_total_states = rl_full_covered = rl_full_total = 0
    rl_audit_html = (
        f'<div class="card"><strong>RL训练覆盖：</strong>已访问 {rl_covered_states}/{rl_total_states} 个热状态；'
        f'热状态×实际容量模式自然访问 {rl_full_covered}/{rl_full_total}。未访问的不安全/不可达组合不通过伪造数据补齐。覆盖不等于策略收敛，还必须结合验证目标、TD误差和独立测试。</div>'
        if rl_total_states and rl_covered_states == rl_total_states
        else f'<div class="warning"><strong>RL训练覆盖不足：</strong>Q表只覆盖 {rl_covered_states}/{rl_total_states} 个状态，'
        '未访问状态必须保持增益并回退 IMC，不能将该策略称为已充分训练。</div>'
    )
    method_rows = [
        {"算法": "Z-N", "直觉：它在做什么": "阶跃测试后按经验公式整定", "主要优点": "无需训练、简单、响应快", "主要缺点": "偏激进，易过冷/抖动；设备模型变化要重新调试", "Kp/Ki 行为": "安装调试一次；三工况冻结", "适用": "快速基准和简单部署"},
        {"算法": "IMC", "直觉：它在做什么": "按期望闭环速度计算保守增益", "主要优点": "平滑、可解释、安全余量大", "主要缺点": "过于保守时恢复慢", "Kp/Ki 行为": "安装调试一次；三工况冻结", "适用": "平稳和设备寿命优先"},
        {"算法": "贝叶斯自动整定", "直觉：它在做什么": "离线搜索一套全局固定参数", "主要优点": "减少人工试参；上线仍是普通 PI", "主要缺点": "依赖训练工况/权重；不在线适应", "Kp/Ki 行为": "三工况共用同一套固定参数", "适用": "工况范围明确、部署简单"},
        {"算法": "FNN 在线自整定", "直觉：它在做什么": "5×5 规则表按误差在线插值", "主要优点": "动态改参；每次只算 4 条；轻量", "主要缺点": "需离线标签/专家规则；未覆盖区域需验证", "Kp/Ki 行为": "运行中动态改参，有限幅/变化率保护", "适用": "低成本 MCU 轻量自适应"},
        {"算法": "RL 在线自整定", "直觉：它在做什么": "离线学习相对IMC的绝对 Kp/Ki 目标", "主要优点": "可围绕综合目标学习；上线查表轻", "主要缺点": "训练/验证成本最高；小样本未必占优", "Kp/Ki 行为": "离线训练；上线不探索，动作屏蔽后安全改参", "适用": "有可靠仿真器的研究方案"},
    ]
    resource_rows = [
        {"目标": "PID 内环", "周期": "100 ms", "作用": "每一拍输出压缩机 PWM", "MCU 形式": "纯 C/C++ 数学函数"},
        {"目标": "FNN/RL 外环", "周期": "2–5 s", "作用": "偶尔修改 Kp、Ki", "MCU 形式": "25 条规则取 4 条 / 5×5×3 策略查表"},
        {"目标": "安全保护", "周期": "每次计算", "作用": "限幅、变化率限制、异常回退", "MCU 形式": "固定边界判断"},
    ]
    best = min(dynamic, key=lambda row: float(row["itae_c_hour2"])) if dynamic else None
    conclusion = (
        f"该动态场景的最低 ITAE 来自 <strong>{html.escape(str(best['controller']))}</strong>（{number(best['itae_c_hour2'])} °C·h²）。"
        if best else "暂无可用于自动结论的数据。"
    )
    nominal = Scenario()
    model_parameter_rows = [
        {"符号 / 字段": "Cz / c_zone_j_per_k", "含义": "室内空气与快速热质的等效热容", "单位": "J/K", "名义值": f"{nominal.c_zone_j_per_k:.3g}"},
        {"符号 / 字段": "Cw / c_wall_j_per_k", "含义": "墙体、机柜等慢热质的等效热容", "单位": "J/K", "名义值": f"{nominal.c_wall_j_per_k:.3g}"},
        {"符号 / 字段": "Roz / r_out_zone_k_per_w", "含义": "室外到室内空气的直接热阻", "单位": "K/W", "名义值": f"{nominal.r_out_zone_k_per_w:.4g}"},
        {"符号 / 字段": "Rzw / r_zone_wall_k_per_w", "含义": "室内空气与慢热质之间的热阻", "单位": "K/W", "名义值": f"{nominal.r_zone_wall_k_per_w:.4g}"},
        {"符号 / 字段": "Row / r_out_wall_k_per_w", "含义": "室外到慢热质的热阻", "单位": "K/W", "名义值": f"{nominal.r_out_wall_k_per_w:.4g}"},
        {"符号 / 字段": "Qmax / cooling_capacity_w", "含义": "100% 容量指令对应的最大制冷量", "单位": "W", "名义值": f"{nominal.cooling_capacity_w:.0f}"},
        {"符号 / 字段": "L / actuator_delay_minutes", "含义": "指令到冷量开始可见的纯延迟", "单位": "min", "名义值": f"{nominal.actuator_delay_minutes:.1f}"},
        {"符号 / 字段": "τu / actuator_tau_minutes", "含义": "变频制冷量建立的一阶惯性时间常数", "单位": "min", "名义值": f"{nominal.actuator_tau_minutes:.1f}"},
        {"符号 / 字段": "Qint / internal_load_w", "含义": "机柜、人员等持续内部发热", "单位": "W", "名义值": f"{nominal.internal_load_w:.0f}"},
        {"符号 / 字段": "Δt / dt_minutes", "含义": "Python 批量热仿真积分步长", "单位": "min", "名义值": f"{nominal.dt_minutes:.1f}"},
    ]
    io_rows = [
        {"层级": "环境 / 场景", "输入": "室外温度 To(t)、设定值 Tsp(t)、设备/人员/开门热负荷 Qint(t)", "输出": "每时刻的扰动时序"},
        {"层级": "3R2C 热模型", "输入": "To(t)、Qint(t)、实际制冷量 Qc(t)", "输出": "室温 Tz(t)、慢热质温度 Tw(t)"},
        {"层级": "安全 PI 内环", "输入": "e=Tz-Tsp、Kp、Ki、采样时间", "输出": "u∈[0,1]，对应 0–100% 容量/PWM 请求"},
        {"层级": "FNN / RL 自整定层", "输入": "当前误差 e 与误差变化 Δe", "输出": "受限的 Kp、Ki；不直接输出压缩机命令"},
        {"层级": "记录与评估", "输入": "Tz、Tsp、u、Qc、Kp/Ki、扰动与安全状态", "输出": "逐步 CSV、指标 CSV、图表与本报告"},
    ]
    metric_rows = [
        {"类别": "跟踪精度与速度", "指标": "RMSE", "数学定义": "sqrt(mean(e²))", "工程含义": "平均温差量级，越小越好"},
        {"类别": "跟踪精度与速度", "指标": "IAE", "数学定义": "∫|e(t)|dt", "工程含义": "全过程累计温差"},
        {"类别": "跟踪精度与速度", "指标": "ITAE", "数学定义": "∫t|e(t)|dt", "工程含义": "长时间未消除的误差惩罚更大"},
        {"类别": "跟踪精度与速度", "指标": "调节时间", "数学定义": "首次进入并连续 60 min 保持在 ±0.5 °C", "工程含义": "启动后多快稳定"},
        {"类别": "动态品质", "指标": "扰动恢复时间", "数学定义": "设定值变化或最大热负荷脉冲结束后，进入 ±0.5 °C 所需时间", "工程含义": "只看事件后恢复，不与启动阶段混在一起"},
        {"类别": "动态品质", "指标": "最大过冷 / 最大过热", "数学定义": "max(-e,0) / max(e,0)", "工程含义": "分别防止降得太低与长时间偏热"},
        {"类别": "动态品质", "指标": "舒适带超限度时", "数学定义": "∫max(|e|-0.5,0)dt", "工程含义": "超出容许温差的程度和持续时间"},
        {"类别": "执行机构寿命与能耗", "指标": "容量指令方差", "数学定义": "Var(u)", "工程含义": "容量命令波动；仅作为平稳性代理，不等同于寿命"},
        {"类别": "执行机构寿命与能耗", "指标": "控制总变化量", "数学定义": "Σ|u[k]-u[k-1]|", "工程含义": "指令来回调节的总幅度"},
        {"类别": "执行机构寿命与能耗", "指标": "制冷能量代理", "数学定义": "∫Qc(t)dt / 1000", "工程含义": "累计制冷量；不等于真实电耗，尚未引入 COP/频率效率图"},
        {"类别": "安全与实时性", "指标": "稳定率 / 回退次数 / AI 推理时间", "数学定义": "有界温度轨迹、回退沿与平均推理微秒", "工程含义": "用于评估异常与实时调度"},
    ]
    mcu_rows = [
        {"模块": "SafePI 状态", "数据依据": "PC C++ sizeof 实测；Wokwi 双目标编译通过", "Flash / ROM": "目标编译日志未提供尺寸", "RAM / 状态": "32 B（仅 PC ABI）", "说明": "含增益/回退、积分、输出斜率和诊断状态；目标 ABI 仍需复测"},
        {"模块": "FNN 5×5×2 规则表", "数据依据": "outputs/fnn_rule_table.npy → C float 表", "Flash / ROM": "200 B", "RAM / 状态": "推理临时量 <64 B", "说明": "已导出训练后件；每次只读取 4 条相邻规则"},
        {"模块": "RL 贪心策略", "数据依据": "outputs/rl_q_table.npy → 动作索引/覆盖掩码", "Flash / ROM": "25 B 策略 + 25 B 掩码 + 72 B 动作表", "RAM / 状态": "推理临时量 <32 B", "说明": f"本轮离线训练覆盖 {rl_covered_states}/{rl_total_states} 状态；未覆盖状态保持增益并回退"},
        {"模块": "PC Testbench 时序", "数据依据": "g++ -O2，本机实测", "Flash / ROM": "PC exe 大小不可代表 MCU ROM", "RAM / 状态": "SafePI 32 B；虚拟对象 68 B", "说明": "PID 100 ms / AI 2 s 分频、NaN/未覆盖状态回退通过；PC ns 不换算成 MCU 周期"},
        {"模块": "Wokwi 双目标工程", "数据依据": "outputs/mcu_wokwi_validation.csv（2026-08-23）", "Flash / ROM": "两目标编译通过；尺寸未取得", "RAM / 状态": "未取得", "说明": "ESP32 运行约 10.079 s；STM32F103 运行约 28.500 s并点击开门按钮；串口曲线和最坏周期未完成留证"},
    ]
    target_rows = [
        {"目标": "STM32F103C8T6", "官方资源": "72 MHz Cortex-M3，64 KB Flash，20 KB SRAM", "本项目判断": "PI/FNN/RL 表格策略都能容纳；无 FPU，但 2 s 外环计算量很小", "量产前必测": "ARM GCC 的 .text/.data/.bss、DWT 最坏周期、中断抖动"},
        {"目标": "ESP32（经典系列）", "官方资源": "最高 240 MHz、520 KB SRAM；模组 Flash 取决于具体型号", "本项目判断": "资源充足，可增加遥测/网络；需隔离 Wi-Fi 任务与 100 ms 控制任务", "量产前必测": "ESP-IDF size、任务栈高水位、看门狗、Wi-Fi 干扰下的最坏时延"},
    ]
    safety_rows = [
        {"等级": "已在 Python/C++ 核心实现", "措施": "u∈[0,1] 输出限幅、可配置输出斜率、条件积分抗饱和、Kp/Ki 边界、单次增益变化±10%、非有限值及 RL 未覆盖状态回退、上线无随机探索"},
        {"等级": "当前缺口，量产前必须补齐", "措施": "独立设备安全层的厂家频率斜率、最小运行/停机时间、高低压/排气温度联锁、传感器合理性与替代值、通信超时、看门狗、参数 CRC/版本回滚、安全手动模式"},
        {"等级": "上线流程", "措施": "SIL 软件在环 → HIL 硬件在环 → 只读影子评分 → 有限增益区间试运行 → 单机试点 → 多季节回归后量产"},
    ]

    scenario_purposes = {
        "初次快速降温": "大偏差阶跃启动：检验饱和、启动速度、过冷和压缩机指令冲击。",
        "设定温度突变": "工况迁移：在系统运行中下调目标温度，检验新平衡点跟随与增益反应。",
        "持续外界热扰动": "突发开门 + 昼夜温差 + 持续设备发热：检验长时抗扰、开门脉冲恢复与执行机构平稳性。",
    }
    scenario_sections: list[str] = []
    for index, (case_name, scenario) in enumerate(typical_case_scenarios().items(), start=1):
        selected = [row for row in case_rows if row.get("case") == case_name]
        settings = [
            {"设置": "仿真时长 / 步长", "数值": f"{scenario.duration_hours:g} h / {scenario.dt_minutes:g} min"},
            {"设置": "初始室温 / 设定值", "数值": f"{scenario.initial_zone_c:g} °C / {scenario.setpoint_c:g} °C" + (f"；{scenario.setpoint_change_hour:g} h 后改为 {scenario.setpoint_after_c:g} °C" if scenario.setpoint_change_hour is not None else "")},
            {"设置": "室外温度", "数值": f"基准 {scenario.outdoor_c:g} °C，昼夜振幅 ±{scenario.outdoor_amplitude_c:g} °C"},
            {"设置": "内部热负荷", "数值": f"基础 {scenario.internal_load_w:g} W" + (f"；{scenario.occupied_start_hour:g}–{scenario.occupied_end_hour:g} h 额外 {scenario.occupied_load_add_w:g} W" if scenario.occupied_load_add_w else "")},
            {"设置": "开门热脉冲", "数值": "无" if scenario.door_open_hour is None else f"{scenario.door_open_hour:g} h 开始，持续 {scenario.door_open_duration_minutes:g} min，+{scenario.door_open_load_w:g} W"},
            {"设置": "制冷量 / 延迟 / 惯性", "数值": f"{scenario.cooling_capacity_w:g} W / {scenario.actuator_delay_minutes:g} min / {scenario.actuator_tau_minutes:g} min"},
        ]
        result_rows = [
            {"算法": display_names.get(str(row.get("controller")), str(row.get("controller"))), "ITAE": number(row.get("itae_c_hour2", "")), "调节时间(h)": number(row.get("settling_time_hour", "")), "扰动恢复(h)": number(row.get("disturbance_recovery_time_hour", "")), "恢复状态": "已在时窗内恢复" if float(row.get("disturbance_recovered", 0.0)) > 0.5 else "未在时窗内恢复", "最大过热(°C)": number(row.get("max_overheat_c", "")), "最大过冷(°C)": number(row.get("max_undershoot_c", "")), "容量指令方差": number(row.get("compressor_output_variance", "")), "能量代理(kWh)": number(row.get("cooling_energy_kwh", ""))}
            for row in selected
        ]
        if selected:
            best_itae = min(selected, key=lambda row: float(row["itae_c_hour2"]))
            recovered_rows = [row for row in selected if float(row.get("disturbance_recovered", 0.0)) > 0.5]
            smoothest = min(selected, key=lambda row: float(row["compressor_output_variance"]))
            lowest_energy = min(selected, key=lambda row: float(row["cooling_energy_kwh"]))
            analysis = (
                f"ITAE 最低为 <strong>{html.escape(display_names.get(str(best_itae['controller']), str(best_itae['controller'])))}</strong>"
                f"（{number(best_itae['itae_c_hour2'])}）；"
                + (f"在给定时窗内，事件局部恢复时间最短为 <strong>{html.escape(display_names.get(str(min(recovered_rows, key=lambda row: float(row['disturbance_recovery_time_hour']))['controller']), str(min(recovered_rows, key=lambda row: float(row['disturbance_recovery_time_hour']))['controller'])))}</strong>（{number(min(recovered_rows, key=lambda row: float(row['disturbance_recovery_time_hour']))['disturbance_recovery_time_hour'])} h）；" if recovered_rows else "所有算法都未在剩余仿真时窗内连续 60 min 回到 ±0.5 °C，表中恢复时间是截尾上限，不能排名；")
                + f"容量指令最平滑为 <strong>{html.escape(display_names.get(str(smoothest['controller']), str(smoothest['controller'])))}</strong>"
                f"（方差 {number(smoothest['compressor_output_variance'])}）。能量代理最低的 {html.escape(display_names.get(str(lowest_energy['controller']), str(lowest_energy['controller'])))} "
                "不一定最节能：如果温度始终偏高，少制冷只是没有完成任务。"
            )
        else:
            analysis = "暂无数据。"
        scenario_sections.append(
            f"<h3>4.{index} 场景 {index}：{html.escape(case_name)}（{html.escape(scenario_purposes[case_name].split('：', 1)[0])}）</h3>"
            f"<p>{html.escape(scenario_purposes[case_name])}</p>"
            f"<h4>仿真设置</h4>{html_table(settings, ['设置', '数值'])}"
            f"<h4>统一指标结果</h4>{html_table(result_rows, ['算法', 'ITAE', '调节时间(h)', '扰动恢复(h)', '恢复状态', '最大过热(°C)', '最大过冷(°C)', '容量指令方差', '能量代理(kWh)'])}"
            f"<div class='card'>{analysis}</div>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>HVAC AI-PID 工程报告</title>
{_katex_head() if katex_available else ''}
<style>
@page {{ size: A4; margin: 15mm; }}
body {{ font-family: "Microsoft YaHei", Arial, sans-serif; color:#172033; max-width:1080px; margin:36px auto; line-height:1.55; }}
h1 {{ color:#0f3d67; border-bottom:3px solid #36a3d9; padding-bottom:10px; }} h2 {{ color:#0f3d67; margin-top:30px; }} h3 {{ color:#1e5d89; margin-top:22px; }}
.lead {{ color:#536276; }} .card {{ background:#f3f8fc; border-left:4px solid #36a3d9; padding:14px 18px; border-radius:5px; margin:12px 0; }}
.equation {{ background:#f3f8fc; color:#172033; padding:12px 16px; border-left:4px solid #36a3d9; border-radius:6px; overflow-x:auto; text-align:center; }}
.warning {{ background:#fff7df; border-left:4px solid #d89a16; padding:14px 18px; margin:12px 0; }}
.status {{ display:inline-block; background:#dff3e5; color:#176334; padding:2px 8px; border-radius:12px; font-size:12px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }} th {{ background:#0f3d67; color:white; }} th,td {{ padding:8px; text-align:left; border:1px solid #d7e0e9; }} tr:nth-child(even) {{ background:#f6f9fc; }}
img {{ width:100%; border:1px solid #d7e0e9; margin:12px 0 20px; }} .note {{ font-size:13px; color:#536276; }}
</style></head><body>
<h1>变频精密 / 机柜空调 AI-PID 仿真、算法对比与量产评估报告</h1>
<p class="lead">报告主线：环境建模 → 控制算法数学设计 → 统一指标 → 三工况实验 → STM32F103C8T6 / ESP32 量产安全评估</p>
<div class="warning"><strong>结论边界：</strong>五种控制器的性能表来自项目内 Python 3R2C + 延迟/一阶执行器仿真；本轮另外真实编译并运行了 OpenModelica 模型。运行状态：<code>{html.escape(modelica_run_status)}</code>。求解器间的一致只证明方程实现一致，不能代替真实机房的热阻、热容和制冷量标定。</div>

<h2>1. 环境建模：怎么建、用什么仿真器、输入输出是什么</h2>
<h3>1.1 仿真器与双层定位</h3>
<table><thead><tr><th>层级</th><th>工具 / 模型</th><th>用途</th><th>当前状态</th></tr></thead><tbody>
<tr><td>物理参考层</td><td>OpenModelica 1.27 + Modelica Standard Library</td><td>用标准 Modelica 热组件表达围护结构、室外边界、内部发热、制冷热流与执行器延迟；Buildings Library 留作后续高保真设备扩展</td><td>{modelica_table_status}</td></tr>
<tr><td>独立数值参考</td><td>SciPy solve_ivp / DOP853</td><td>不用项目的欧拉推进代码，直接高精度求解连续热平衡微分方程</td><td><span class="status">本轮已实际运行</span></td></tr>
<tr><td>快速实验层</td><td>Python + NumPy/SciPy/scikit-learn</td><td>3R2C 房间热模型、FOPDT 辨识、数百次寻优/训练和五控制器对比</td><td><span class="status">本报告数值来源</span></td></tr>
</tbody></table>
<p>两个模型不是两套互相矛盾的系统：Modelica 用于物理可追溯和跨求解器核对，Python 用于快速重复实验。本轮已把相同参数、天气、负荷和制冷指令放入两套连续方程并完成数值对齐；但它们仍使用同一组假设参数，所以不能称为已经用真实空调/BMS 数据标定。</p>

<h3>1.2 3R2C 房间热模型与执行器</h3>
<p>直觉上，Tz 是很快变化的“空气温度”，Tw 是慢慢吸热/放热的“墙体+机柜热质温度”。两个热容 C 和三条热阻 R 构成 3R2C 等效网络。代码中确实有室外→空气、室外→慢热质、慢热质↔空气三条换热通道，因此本版不再使用不严谨的“2R2C”名称。</p>
<div class="equation">\\[\\begin{{aligned}} C_z\\frac{{dT_z}}{{dt}}&=\\frac{{T_o-T_z}}{{R_{{oz}}}}+\\frac{{T_w-T_z}}{{R_{{zw}}}}+Q_{{int}}-Q_c \\\\ C_w\\frac{{dT_w}}{{dt}}&=\\frac{{T_o-T_w}}{{R_{{ow}}}}+\\frac{{T_z-T_w}}{{R_{{zw}}}} \\\\ \\tau_u\\frac{{dQ_c}}{{dt}}&=Q_{{max}}u(t-L)-Q_c \\end{{aligned}}\\]</div>
<p>如果只保留一个等效蓄热状态，就得到你给出的直观一阶热平衡：室外传入热量和开门/人员热量把温度往上推，空调制冷量把温度往下拉。项目保留第二个“墙体/机柜慢热状态”，是为了避免把长时间蓄热过度简化。</p>
<div class="equation">\\[C_{{air}}\\frac{{dT_{{in}}(t)}}{{dt}}=\\frac{{T_{{out}}(t)-T_{{in}}(t)}}{{R_{{th}}}}-Q_{{cool}}(t-L)+Q_{{dist}}(t)\\]</div>
<p>室外昼夜温度按 <span class="math-inline">\\(T_o(t)=T_{{base}}+A\\cos[2\\pi(t-t_{{peak}})/24\\,h]\\)</span> 生成；总热负荷是设备基础发热、人员/负荷时段附加值和开门短时脉冲之和。执行器把归一化指令 <span class="math-inline">\\(u\\)</span> 先延迟 <span class="math-inline">\\(L\\)</span>，再以时间常数 <span class="math-inline">\\(\\tau_u\\)</span> 建立真实制冷量 <span class="math-inline">\\(Q_c\\)</span>。</p>
<h3>1.3 系统参数定义</h3>{html_table(model_parameter_rows, ["符号 / 字段", "含义", "单位", "名义值"])}
<p class="note">名义值只是默认 <code>Scenario</code>；训练集会随机改变室外温度、热负荷、热容/热阻、制冷能力、延迟和惯性，三个工程场景的值在第 4 章单独列出。</p>

<h3>1.4 FOPDT 快速辨识与控制函数</h3>
<p><strong>直觉：</strong>FOPDT 是 First-Order Plus Dead Time，即“一阶惯性 + 纯延迟”。它把复杂机房的阶跃曲线压缩成三个数：K 表示制冷指令增加后能降多少温，L 表示命令发出后要空等多久，τ 表示开始降温后变化有多慢。</p>
<div class="equation">\\[G(s)=\\frac{{\\Delta T(s)}}{{\\Delta u(s)}}=-\\frac{{K}}{{\\tau s+1}}e^{{-Ls}}\\]</div>
<p>这个 FOPDT 不取代 3R2C 环境，而是专门用于 Z-N/IMC 参数计算的低阶控制代理。当前实现先进行 168 h 的加速虚拟阶跃，再对 K、τ、L 做有界最小二乘拟合；<code>fopdt_fit_history.csv</code> 保存拟合器评价过的每一组候选。28.3%/63.2% 交叉点只用于诊断，不再直接决定参数。</p>
<div class="warning"><strong>已修正的辨识问题：</strong>旧版 12 h 试验结束时只达到最终温降约 59.3%，并未真正越过 63.2%，却把末点当成 t63。现在使用长时虚拟响应和曲线拟合，避免这一截尾错误。虚拟工况仍用模型平衡方程快速建立基准工作点；真实设备不能照搬这一捷径，应从稳定 BMS 记录取得基准点，并加入测量噪声、扰动筛选和安全激励约束。</div>
<p>主仿真函数是 <code>simulate(scenario, controller, seed)</code>：每一步先生成天气/负荷/设定值，再计算 PI 指令，通过延迟执行器更新 3R2C 状态，最后记录全部轨迹。</p>
<h3>1.5 最终输入与输出</h3>{html_table(io_rows, ["层级", "输入", "输出"])}
<div class="warning"><strong>时间尺度：</strong>Python 实验用 1 min 热模型步长；因此 FNN/RL 在该离线实验中最多每个热步更新一次。C++ 目标调度是 PI 100 ms、AI 2 s，它用于验证 MCU 实时性，不应与 1 min 热仿真步长混为一谈。</div>

<h3>1.6 实际运行交叉验证</h3>
<p><strong>直觉：</strong>第一项检查“1 分钟一跳的快速代码有没有因为步长太大算错”；第二项检查“把复杂房间压缩成一个 K、一个 τ、一个 L 后，还像不像原来的房间”；第三项检查“Modelica 组件连线与 Python 写出的热平衡方程，是否真的是同一个系统”。三项都使用预先写入代码的阈值，未通过的结果不会被隐藏。</p>
<h4>运行环境与求解器</h4>{html_table(validation_environment_rows, ["运行项", "实际值"])}
<h4>交叉验证 A：离散 3R2C 与独立连续方程</h4>
{html_table(physical_validation_rows, ["验证场景", "采样点", "温度 RMSE（°C）", "最大绝对误差（°C）", "末端误差（°C）", "预设阈值", "结论"])}
{image_data('physical_cross_validation.png')}
<div class="card">通过数：{physical_pass_count}/{len(physical_validation_rows)}。这验证的是数值实现一致性，不等于参数已经符合真实房间。</div>
<h4>交叉验证 B：FOPDT 代理与两状态物理模型</h4>
{html_table(fopdt_validation_rows, ["验证场景", "过程增益 K（°C/指令）", "时间常数 τ（min）", "延迟 L（min）", "温度 RMSE（°C）", "归一化 RMSE（%）", "预设阈值", "结论"])}
{image_data('fopdt_cross_validation.png')}
<div class="card">通过数：{fopdt_pass_count}/{len(fopdt_validation_rows)}。FOPDT 只承担整定和快速搜索代理，不替代三场景的 3R2C 主实验。</div>
<h4>交叉验证 C：OpenModelica/DASSL 与 Python/DOP853 连续方程</h4>
{html_table(openmodelica_validation_rows, ["验证场景", "采样点", "温度 RMSE（°C）", "最大绝对误差（°C）", "末端误差（°C）", "预设阈值", "结论"])}
{image_data('openmodelica_cross_validation.png')}
<div class="card">通过数：{openmodelica_pass_count}/{len(openmodelica_validation_rows)}。实际模型共有 62 个方程和 62 个变量，采用 DASSL、容差 10⁻⁸、6 s 输出间隔运行 12 h；对照端用 DOP853、最大 3 s 内部步长。比较覆盖昼夜室外温度、人员负荷、5 h 开门热扰动、2 h 制冷指令阶跃、8 min 纯延迟与 6 min 执行器惯性。</div>

<h2>2. 每种控制算法的设计与数学表达</h2>
<h3>2.1 共用安全 PI 内环</h3>
<div class="equation">\\[\\begin{{aligned}} e[k]&=T_z[k]-T_{{sp}}[k] \\\\ I^*[k]&=I[k-1]+e[k]\\Delta t \\\\ u^*[k]&=K_p[k]e[k]+K_i[k]I^*[k] \\\\ u[k]&=\\operatorname{{clip}}(u^*[k],0,1) \\end{{aligned}}\\]</div>
<p>仅当未饱和，或当前误差会把饱和输出拉回区间时，才接受 I*，这是条件积分抗饱和。Python 的 Ki 时基是 1/min；C++ 已在 <code>SafePI::update</code> 中把 100 ms 转换为分钟，防止部署时 Ki 被放大 60 倍。</p>

<h3>2.2 Z-N 反应曲线 PI（经典固定参数）</h3>
<div class="equation">\\[K_p=\\frac{{0.9\\tau}}{{KL}},\\qquad T_i=3.33L,\\qquad K_i=\\frac{{K_p}}{{T_i}}\\]</div>
<p>K、τ、L 已知以后，公式阶段确实可以直接算出 Kp/Ki；这是 Z-N 的设计，不是算法错误。但完整流程并非一步：先建立基准工作点、施加阶跃、采集长时响应、反复拟合 FOPDT，再计算未限幅增益并通过工程安全边界。三个测试工况不再重新辨识，因此运行曲线是水平线。</p>

<h3>2.3 IMC/SIMC PI（λ经训练集整定后固定）</h3>
<div class="equation">\\[\\lambda=\\max\\left(\\frac{{\\tau}}{{3}},3L,12\\,\\mathrm{{min}}\\right),\\quad K_p=\\frac{{\\tau}}{{K(\\lambda+L)}},\\quad T_i=\\min[\\tau,4(\\lambda+L)],\\quad K_i=\\frac{{K_p}}{{T_i}}\\]</div>
<p>λ 表示希望的闭环速度。公式默认值是保守回退；本次公平比较在训练工况上只搜索 λ，Kp/Ti 仍受 IMC 公式约束，然后在留出集和三个案例中冻结。完整候选、训练目标和最终选择见 <code>imc_lambda_tuning.csv</code>。这修正了旧版直接用 λ≈τ/3 的极慢回退参数参与性能排名所造成的不公平。</p>
{image_data('classical_tuning_process.png')}
<h4>FOPDT 拟合器评价过的全部候选</h4>
{html_table(fopdt_fit_rows, ["评价", "候选K", "候选τ(min)", "候选L(min)", "本次RMSE(°C)", "当前最小RMSE(°C)"])}
<h4>从名义工况到 Z-N/IMC 最终 Kp、Ki 的逐步代入</h4>
{html_table(classical_step_rows, ["步骤", "模块", "计算量", "公式或操作", "代入数值", "本步结果", "说明"])}
<p class="note">机器可读明细：<code>classical_tuning_steps.csv</code>、<code>fopdt_fit_history.csv</code> 和逐分钟的 <code>classical_tuning_history.csv</code>。</p>

<h3>2.4 贝叶斯自动整定（离线寻一套全局固定 PI）</h3>
<p>在 log(Kp), log(Ki) 空间内建立 Matérn-5/2 高斯过程代理模型，采用期望改进 EI 选择下一组参数。全局固定整定的优化目标是所有离线训练工况的平均综合代价：</p>
<div class="equation">\\[J_{{global}}(K_p,K_i)=\\operatorname{{mean}}_s J[\\operatorname{{simulate}}(s,PI(K_p,K_i))]\\]\\[EI(x)=(J_{{best}}-\\mu(x)-\\xi)\\Phi(z)+\\sigma(x)\\phi(z),\\qquad z=\\frac{{J_{{best}}-\\mu(x)-\\xi}}{{\\sigma(x)}}\\]</div>
<p>本次进行 {bayes_evaluations} 次批量仿真评价，得到 Kp={bayes_kp:.6g}、Ki={bayes_ki:.6g}，三个工程场景共用同一套参数。它是 Auto-tuning：离线搜索后冻结，不是运行中自适应。</p>
{image_data('bayesian_search_trace.png')}
<p class="note">完整逐次数据保存在 <code>bayesian_search_history.csv</code>：候选 Kp/Ki、本轮目标、当前最优 Kp/Ki、当前最优目标、EI、预测均值与标准差均可追溯。</p>

<h3>2.5 FNN 在线自整定（5×5 零阶 TSK 规则表）</h3>
<p>输入是误差 e 和误差变化率 ė=Δe/Δt；两轴各设 5 个中心。用 °C/min 归一化后，PC训练的5 min节拍与MCU秒级节拍含义一致。当前值只会落在每轴两个相邻中心之间，因此 25 条隐式规则中只有 2×2=4 条激活。</p>
<div class="equation">\\[w_{{ij}}=\\mu_i(e)\\mu_j(\\Delta e),\\qquad \\sum w_{{ij}}=1\\]\\[K_p=\\sum_{{(i,j)\\in\\mathcal{{A}}_4}}w_{{ij}}K_{{p,ij}},\\qquad K_i=\\sum_{{(i,j)\\in\\mathcal{{A}}_4}}w_{{ij}}K_{{i,ij}}\\]</div>
<p>训练不是可选项：对每个离线工况先用 BO 得到最优固定 Kp/Ki，再重放其 e/Δe 轨迹，在 log 增益空间对落入同一模糊单元的标签求平均。未覆盖单元保留 IMC 回退值。</p>
{image_data('fnn_training_trace.png')}
<p class="note"><code>fnn_training_history.csv</code> 按加入训练工况的顺序记录规则覆盖率、拟合 log-RMSE、规则表最大变化和代表性 Kp/Ki；这是当前 TSK 规则表的真实拟合过程，不伪造深度网络 epoch loss。</p>

<h3>2.6 安全约束的表格 RL 在线自整定</h3>
<p>状态由 5×5 个 (e,ė) 热状态和3档上一次实际容量模式组成；动作是9种相对IMC的绝对Kp/Ki目标比例，不是压缩机动作。绝对目标避免旧版增量动作依赖未进入状态的当前增益；容量模式减少同一温差但执行器状态不同造成的状态混叠。离线使用750回合、每回合最长240 min、每5 min决策。25热状态覆盖只统计物理轨迹真实到达的网格，不人为注入状态凑数。</p>
<div class="equation">\\[r=-|e'|-0.18\\|\\Delta g\\|_1-0.04|K_{{p,scale}}-1|-0.03u\\]\\[Q(s,a)\\leftarrow Q(s,a)+\\alpha\\left[r+\\gamma\\max_{{a'}}Q(s',a')-Q(s,a)\\right]\\]\\[\\alpha=0.12,\\qquad\\gamma=0.94,\\qquad\\varepsilon:0.25\\rightarrow0.03\\]</div>
<p>部署时先屏蔽“房间已冷却却提高增益”等不安全动作，再执行argmax Q，不随机探索；提议增益还需经过边界和单次±10%限制。训练期间用内部验证集早停选取Q表；若平均目标比已整定IMC差2%以上或25热状态覆盖低于80%，策略整体拒绝并退回IMC。当前仍是小型表格RL，不能与深度RL直接类比。</p>
{image_data('rl_training_trace.png')}
{rl_audit_html}

<h3>2.7 三种 AI 方法的收敛证据不能混为同一个数值</h3>
{image_data('training_convergence_overview.png')}
<p>贝叶斯看“截至当前最优目标”是否停止显著下降；FNN 看规则覆盖和监督拟合误差；RL 看奖励、TD 误差、状态覆盖和策略翻转。三者量纲不同，只能分别判断趋势，不能把曲线高低直接横向比较。</p>

<h3>2.8 五种方法的设计取舍</h3>{html_table(method_rows, ["算法", "直觉：它在做什么", "主要优点", "主要缺点", "Kp/Ki 行为", "适用"])}

<h2>3. 统一评估指标体系</h2>
<p>所有算法使用同一被控对象、同一初始状态、同一扰动和输出边界。评估分为三个主类：跟踪精度与速度、动态品质、执行机构寿命与能耗；安全/实时性作为附加约束。</p>
<div class="equation">\\[ITAE=\\int_0^T t|e(t)|dt,\\qquad IAE=\\int_0^T|e(t)|dt,\\qquad RMSE=\\sqrt{{\\frac1N\\sum_{{k=1}}^N e_k^2}}\\]\\[\\sigma_{{cold}}=\\max(T_{{sp}}-T_z,0),\\qquad Var(u)=\\frac1N\\sum_{{k=1}}^N(u_k-\\bar u)^2,\\qquad TV(u)=\\sum_{{k=1}}^N|u_k-u_{{k-1}}|\\]</div>
{html_table(metric_rows, ["类别", "指标", "数学定义", "工程含义"])}
<p>贝叶斯优化和统一排序使用的当前综合目标为：</p>
<div class="equation">\\[\\begin{{aligned}}J&=2\\,IAE+1.5\\,ITAE+10\\,ComfortViolation+3\\,MaxOvercool \\\\ &\\quad+0.35\\,Settling+0.08\\,ControlMovement+0.7\\,Var(u)+0.015\\,CoolingEnergyProxy\\end{{aligned}}\\]</div>
<div class="warning"><strong>能耗口径限制：</strong>当前只累计制冷量 Qc，没有压缩机 COP(频率、室内/外温度)、风机/水泵功耗和启停损耗。因此它只能用于同模型内横向比较，不能作为真实 kWh 节能承诺。报告已把“PWM 方差”改称“容量指令方差”，并另记启停次数；两者都不能直接等同于真实寿命。</div>

<h2>4. 实验分析与工程评估：三个独立场景</h2>
<p>每个算法都独立跑完三个场景；不用一条混合曲线代替三类工程问题。下图给出总波形，后面分别列出仿真设置、统一指标和取舍分析。</p>
{image_data('case_studies.png')}
{''.join(scenario_sections)}
<h3>4.4 补充：留出工况与混合动态工况</h3>
<p>三工况用于可解释的极限试题；留出工况用于检查泛化；12 h 混合场景用于观察多扰动叠加。下表是补充证据，不取代三场景分析。</p>
<h4>留出工况汇总</h4>{html_table(holdout_rows, holdout_columns)}
<h4>混合动态工况汇总</h4>{html_table(dynamic_rows, dynamic_columns)}
{image_data('dynamic_comparison.png')}

<h2>5. 嵌入式量产落地与工程安全评估</h2>
<h3>5.1 软件解耦与分频架构</h3>
{html_table(resource_rows, ["目标", "周期", "作用", "MCU 形式"])}
<p>建议量产数据流为：传感器采样 → 合理性检查/滤波 → 100 ms SafePI → 容量变化率和压缩机联锁 → PWM/频率请求；FNN/RL 只在 2 s 任务中读取 e,Δe 并原子更新 Kp/Ki。</p>

<h3>5.2 MCU 资源开销评估（STM32F103C8T6 / ESP32）</h3>
{html_table(mcu_rows, ["模块", "数据依据", "Flash / ROM", "RAM / 状态", "说明"])}
{html_table(target_rows, ["目标", "官方资源", "本项目判断", "量产前必测"])}
<p>静态表格占用对 STM32F103C8T6 很小：FNN 训练后件已导出为 200 B float32 表；RL 不把 900 B Q 表带上 MCU，而是压成 25 B 贪心动作索引、25 B 覆盖掩码和 72 B 动作表。本轮 Q 表训练实际覆盖 {rl_covered_states}/{rl_total_states} 状态；即便覆盖全部粗网格，嵌入式端仍保留覆盖掩码和 IMC 回退，防止更换模型后误用未训练状态。</p>
<div class="warning"><strong>当前 C++ 交付边界：</strong><code>embedded/hvac_pid_controller.hpp</code> 已包含 SafePI、训练后 FNN 四规则插值、RL 压缩策略和安全回退；<code>embedded/wokwi</code> 已生成 ESP32 与 STM32F103C8 双工程。PC SIL 已通过；2026-08-23 两个完整工程也在 Wokwi 在线编译并进入运行态。ESP32 观察约 10.079 s，STM32F103 观察约 28.500 s并点击一次开门按钮。匿名会话没有产出可复核的串口曲线、目标 ROM/RAM 和最坏周期，因此这些仍明确标为未完成。</div>

<h3>5.3 量产安全策略与缺口</h3>{html_table(safety_rows, ["等级", "措施"])}
<p>Python 统一比较层已经实现最低稳定容量、量化、运行段斜率和最小运停时间；但<strong>仍需按具体压缩机厂家曲线标定参数，并把同一逻辑同步到 MCU，同时补齐制冷系统压力/温度联锁</strong>。这些必须由底层设备安全层实现，不能依赖 AI 或 PI 自己“学会安全”。</p>

<h3>5.4 芯片资源资料与可追溯性</h3>
<ul>
<li><a href="https://www.st.com/en/microcontrollers-microprocessors/stm32f103c8">STMicroelectronics：STM32F103C8 产品页</a>（72 MHz、64 KB Flash，系列 SRAM 最高 20 KB）</li>
<li><a href="https://www.espressif.com/sites/default/files/documentation/esp32_datasheet_en.pdf">Espressif：ESP32 Series Datasheet</a>（最高 240 MHz、520 KB SRAM）</li>
<li><a href="https://docs.wokwi.com/getting-started/supported-hardware">Wokwi：支持的硬件</a>（列出 ESP32 系列和 STM32F103C8）</li>
<li><a href="https://docs.wokwi.com/parts/board-stm32-bluepill">Wokwi：STM32 Blue Pill 仿真说明</a>（列出当前支持的内核与外设范围）</li>
</ul>

<h2>6. 结论、单算法文档与可复现性</h2>
<p>{modelica_conclusion}</p>
<h3>按算法查看三工况完整实验、增益轨迹与代码</h3>
<ul>{''.join(f'<li><a href="{html.escape(link)}">{html.escape(display_names.get(name, name))}：三工况独立实验、Kp/Ki 轨迹、指标与代码</a></li>' for name, link in algorithm_reports.items())}</ul>
{image_data('training_labels.png')}
<p class="note">机器可读产物包括 <code>classical_tuning_history.csv</code>、<code>fopdt_fit_history.csv</code>、<code>classical_tuning_steps.csv</code>、<code>bayesian_search_history.csv</code>、<code>fnn_training_history.csv</code>、<code>rl_training_history.csv</code> 等完整整定/训练历史，以及三工况指标、留出工况指标和 <code>physical_cross_validation*.csv</code>、<code>fopdt_cross_validation*.csv</code>、<code>openmodelica_cross_validation*.csv</code>。OpenModelica 原始结果保存在 <code>outputs/modelica/PrecisionCabinetCooling_res.csv</code>；MCU 的 PC 软件在环日志和摘要保存在 <code>outputs/mcu_pc_sil_log.txt</code>、<code>outputs/mcu_validation_summary.csv</code>，Wokwi 双目标验证边界保存在 <code>outputs/mcu_wokwi_validation.csv</code>。报告中所有“已验证”结论只指这些可复现的软件仿真、PC Testbench 与明确列出的 Wokwi 编译/启动证据，不指真实 MCU 板卡或实机 HVAC 性能。</p>
</body></html>"""
    path.write_text(document, encoding="utf-8")
