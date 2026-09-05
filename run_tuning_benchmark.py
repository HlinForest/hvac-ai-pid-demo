from __future__ import annotations

import argparse
import csv
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hvac_pid.ai_controllers import (
    FNNGainController,
    IncrementalRLController,
    train_fnn_rule_table,
    train_offline_q_policy,
)
from hvac_pid.config import Scenario, sample_scenarios, typical_case_scenarios
from hvac_pid.controllers import PIController, identify_fopdt_audit, ziegler_nichols_pi
from hvac_pid.metrics import calculate_metrics
from hvac_pid.simulator import SimulationResult, simulate
from hvac_pid.tuning import generate_label_rows, tune_global_fixed, tune_global_imc_lambda


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _first_band_minute(result: SimulationResult, band_c: float = 0.5) -> float:
    indices = np.flatnonzero(np.abs(result.zone_c - result.setpoint_c) <= band_c)
    return float(result.minute[indices[0]]) if indices.size else float("nan")


def _time_call(function, *args, **kwargs):
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - started


def run_benchmark(
    output_dir: str | Path,
    *,
    train_samples: int = 8,
    bo_iterations: int = 2,
    rl_episodes: int = 750,
    seed: int = 23,
) -> dict[str, object]:
    """Run one reproducible, engineering-scale PI tuning timing benchmark.

    The plant is still a calibrated-model surrogate, not measured BMS data.
    Wall-clock values are measured on the current PC. Simulated process hours
    show how much plant time would be consumed if unsafe virtual exploration
    were naively transferred to a real unit.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    problem = typical_case_scenarios()["初次快速降温"]
    training = sample_scenarios(train_samples, seed=seed, duration_hours=5.0)
    stages: list[dict[str, object]] = []

    (model, _, _), identify_seconds = _time_call(identify_fopdt_audit, problem)
    stages.append({"stage": "FOPDT identification", "wall_seconds": identify_seconds, "notes": "168 h accelerated virtual step; shared by Z-N/IMC"})

    zn_started = time.perf_counter()
    zn_gains = ziegler_nichols_pi(model)
    zn_formula_seconds = time.perf_counter() - zn_started
    stages.append({"stage": "Z-N formula", "wall_seconds": zn_formula_seconds, "notes": "one formula evaluation after identification"})

    imc_tune, imc_seconds = _time_call(
        tune_global_imc_lambda, training, model, seed=seed + 404
    )
    imc_gains = (imc_tune.kp, imc_tune.ki)
    stages.append({"stage": "IMC lambda search", "wall_seconds": imc_seconds, "notes": f"{imc_tune.evaluations} lambda candidates x {len(training)} training scenarios"})

    global_bo, global_bo_seconds = _time_call(
        tune_global_fixed, training, iterations=max(1, bo_iterations), seed=seed + 808
    )
    bo_gains = (global_bo.kp, global_bo.ki)
    stages.append({"stage": "Bayesian global PI search", "wall_seconds": global_bo_seconds, "notes": f"{global_bo.evaluations} gain candidates x {len(training)} scenarios"})

    labels, label_seconds = _time_call(
        generate_label_rows,
        training,
        bo_iterations=bo_iterations,
        seed=seed,
        progress=False,
    )
    stages.append({"stage": "FNN BO label generation", "wall_seconds": label_seconds, "notes": f"{len(labels)} per-context gain labels"})
    (fnn_table, fnn_history), fnn_fit_seconds = _time_call(
        train_fnn_rule_table, training, labels, imc_gains, return_history=True
    )
    fnn_accepted = bool(fnn_history[-1]["deployment_accepted"] > 0.5)
    stages.append({"stage": "FNN rule fitting and gate", "wall_seconds": fnn_fit_seconds, "notes": f"deployment accepted={fnn_accepted}"})

    (rl_table, rl_history), rl_seconds = _time_call(
        train_offline_q_policy,
        training,
        fallback_gains=imc_gains,
        seed=seed + 1_616,
        episodes=rl_episodes,
        return_history=True,
    )
    rl_accepted = bool(rl_history[-1]["deployment_accepted"] > 0.5)
    stages.append({"stage": "RL training and gate", "wall_seconds": rl_seconds, "notes": f"{rl_episodes} episodes; deployment accepted={rl_accepted}"})

    controllers = {
        "Z-N": PIController(*zn_gains),
        "IMC lambda-tuned": PIController(*imc_gains),
        "Bayesian fixed PI": PIController(*bo_gains),
        "FNN self-tuning": FNNGainController(imc_gains, rule_table=fnn_table),
        "RL self-tuning": IncrementalRLController(imc_gains, q_table=rl_table),
    }
    method_timing = {
        "Z-N": identify_seconds + zn_formula_seconds,
        "IMC lambda-tuned": identify_seconds + imc_seconds,
        "Bayesian fixed PI": global_bo_seconds,
        "FNN self-tuning": identify_seconds + imc_seconds + label_seconds + fnn_fit_seconds,
        "RL self-tuning": identify_seconds + imc_seconds + rl_seconds,
    }
    label_evaluations = sum(int(row["bo_evaluations"]) + 2 for row in labels)
    validation_count = max(1, len(training) // 5) if len(training) > 1 else 0
    fit_count = len(training) - validation_count
    imc_virtual_hours = 168.0 + imc_tune.evaluations * len(training) * 5.0
    # Each per-context BO label currently identifies FOPDT twice: once for BO
    # seeding and once for its Z-N/IMC audit metrics. These are cheap in
    # accelerated software but would dominate any naive real-plant campaign.
    label_virtual_hours = len(training) * 2.0 * 168.0 + label_evaluations * 5.0
    simulated_hours = {
        "Z-N": 168.0,
        "IMC lambda-tuned": imc_virtual_hours,
        "Bayesian fixed PI": 168.0 + global_bo.evaluations * len(training) * 5.0,
        "FNN self-tuning": imc_virtual_hours + label_virtual_hours + fit_count * 5.0 + validation_count * 2.0 * 5.0,
        "RL self-tuning": imc_virtual_hours + rl_episodes * 4.0 + validation_count * 2.0 * 5.0,
    }
    method_rows: list[dict[str, object]] = []
    for name, controller in controllers.items():
        # Paired comparison: every controller sees the identical measurement-noise trace.
        result, evaluation_seconds = _time_call(simulate, problem, controller, seed=seed + 50_000)
        metrics = calculate_metrics(result)
        method_rows.append(
            {
                "method": name,
                "offline_wall_seconds": method_timing[name],
                "equivalent_simulated_process_hours": simulated_hours[name],
                "evaluation_wall_seconds": evaluation_seconds,
                "kp_min": float(np.min(result.kp)),
                "kp_max": float(np.max(result.kp)),
                "ki_min": float(np.min(result.ki)),
                "ki_max": float(np.max(result.ki)),
                "first_in_band_minutes": _first_band_minute(result),
                "settling_time_hours": metrics["settling_time_hour"],
                "objective": metrics["objective"],
                "mean_ai_inference_us_pc": metrics["mean_ai_inference_us"],
                "deployment_accepted": "not_applicable" if name in {"Z-N", "IMC lambda-tuned", "Bayesian fixed PI"} else fnn_accepted if name.startswith("FNN") else rl_accepted,
            }
        )

    _write_csv(output / "tuning_stage_times.csv", stages)
    _write_csv(output / "tuning_method_summary.csv", method_rows)
    _write_csv(output / "imc_lambda_candidates.csv", list(imc_tune.history))

    names = [str(row["method"]) for row in method_rows]
    pc_seconds = [float(row["offline_wall_seconds"]) for row in method_rows]
    process_hours = [float(row["equivalent_simulated_process_hours"]) for row in method_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(names, pc_seconds, color="#2563EB")
    axes[0].set_xlabel("Current PC wall time (s)")
    axes[0].set_title("Offline tuning/training compute time")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(names, process_hours, color="#D97706")
    axes[1].set_xlabel("Equivalent simulated plant time (h)")
    axes[1].set_title("Virtual experience consumed (not real commissioning)")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "tuning_runtime.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# 工程化 PI 调参耗时基准",
        "",
        "> 这是带真实执行器约束的工程化模型基准，不是真实 BMS 或台架实测。",
        "",
        f"- 运行平台：{platform.platform()} / Python {platform.python_version()}",
        f"- 调参问题：机柜空调 {problem.initial_zone_c:g}→{problem.setpoint_c:g} °C，制冷量 {problem.cooling_capacity_w:g} W，延迟 {problem.actuator_delay_minutes:g} min",
        f"- 离线工况：{train_samples}；BO 迭代：{bo_iterations}；RL 回合：{rl_episodes}",
        "",
        "| 方法 | PC 离线调参/训练 (s) | 等效虚拟对象时间 (h) | 首次进入±0.5°C (min) | 目标值 | PC AI 推理 (us) | 部署验收 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in method_rows:
        lines.append(
            f"| {row['method']} | {float(row['offline_wall_seconds']):.4f} | {float(row['equivalent_simulated_process_hours']):.1f} | "
            f"{float(row['first_in_band_minutes']):.1f} | {float(row['objective']):.4f} | {float(row['mean_ai_inference_us_pc']):.3f} | {row['deployment_accepted']} |"
        )
    lines.extend(
        [
            "",
            "## 如何解释这些时间",
            "",
            "- Z-N/IMC 的公式计算几乎不耗时；真正昂贵的是安全对象辨识和采数。",
            "- BO/FNN/RL 是 PC 对虚拟对象的加速计算。若把候选试验串行搬到真实压缩机，不但不安全，还会消耗数百至数千对象小时。",
            "- `mean_ai_inference_us_pc` 是本机测量，不是 STM32/ESP32 WCET。目标板必须在量产编译配置下用 DWT/周期计数器或 GPIO 脉冲实测。",
            "- 真正的设备结果需要带时间戳的 BMS 数据：设定值、测量温度、室外温度/负荷代理、请求/实际压缩机频率、告警和运行模式。",
        ]
    )
    (output / "tuning_runtime_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"stages": stages, "methods": method_rows, "output": str(output.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Time five PI tuning approaches on one constrained cabinet-HVAC problem")
    parser.add_argument("--output", type=Path, default=Path("outputs_tuning_benchmark"))
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument("--bo-iterations", type=int, default=2)
    parser.add_argument("--rl-episodes", type=int, default=750)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    result = run_benchmark(
        args.output,
        train_samples=args.train_samples,
        bo_iterations=args.bo_iterations,
        rl_episodes=args.rl_episodes,
        seed=args.seed,
    )
    print(f"Benchmark written to {result['output']}")


if __name__ == "__main__":
    import warnings
    warnings.warn(
        "Direct entry run_tuning_benchmark.py is a thin wrapper; prefer 'python run.py benchmark ...'",
        DeprecationWarning, stacklevel=2)
    main()
