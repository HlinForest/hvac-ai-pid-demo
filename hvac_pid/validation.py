from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime
import os
from pathlib import Path
import platform
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import solve_ivp

from .config import Scenario, dynamic_demo_scenario, typical_case_scenarios
from .controllers import identify_fopdt
from .plant import ThermalPlant3R2C, effective_outdoor_resistance, thermal_equilibrium


matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _command_profile(case_name: str, minute: float) -> float:
    """Open-loop commands used only for model cross-validation."""

    if case_name == "初次快速降温":
        return 0.65
    if case_name == "设定温度突变":
        return 0.25 if minute < 60.0 else 0.55
    return 0.40


def _initial_command(case_name: str) -> float:
    return {"初次快速降温": 0.0, "设定温度突变": 0.25, "持续外界热扰动": 0.40}[case_name]


def _run_discrete_open_loop(scenario: Scenario, case_name: str) -> dict[str, np.ndarray]:
    plant = ThermalPlant3R2C(scenario)
    initial_u = _initial_command(case_name)
    plant.reset(initial_u=initial_u)
    minute = np.arange(scenario.steps, dtype=float) * scenario.dt_minutes
    zone = np.empty_like(minute)
    wall = np.empty_like(minute)
    cooling = np.empty_like(minute)
    command = np.empty_like(minute)
    zone[0], wall[0], cooling[0] = plant.zone_c, plant.wall_c, plant.cooling_w
    command[0] = _command_profile(case_name, 0.0)
    for index in range(1, scenario.steps):
        source_minute = float(minute[index - 1])
        command[index] = _command_profile(case_name, float(minute[index]))
        zone[index], wall[index] = plant.step(
            _command_profile(case_name, source_minute),
            scenario.outdoor_at(source_minute),
            scenario.load_at(source_minute),
        )
        cooling[index] = plant.cooling_w
    return {"minute": minute, "zone_c": zone, "wall_c": wall, "cooling_w": cooling, "command": command}


def _run_continuous_reference(scenario: Scenario, case_name: str, minute: np.ndarray) -> dict[str, np.ndarray]:
    """Independently solve the continuous equations with SciPy DOP853."""

    initial_u = _initial_command(case_name)
    initial_zone = scenario.initial_zone_c
    initial_wall = 0.7 * initial_zone + 0.3 * scenario.outdoor_c
    initial_cooling = initial_u * scenario.cooling_capacity_w
    delay = scenario.actuator_delay_minutes
    tau_seconds = max(scenario.actuator_tau_minutes * 60.0, 1e-9)

    def rhs(second: float, state: np.ndarray) -> np.ndarray:
        current_minute = second / 60.0
        delayed_minute = current_minute - delay
        delayed_u = initial_u if delayed_minute < 0.0 else _command_profile(case_name, delayed_minute)
        zone_c, wall_c, cooling_w = state
        outdoor_c = scenario.outdoor_at(current_minute)
        internal_load_w = scenario.load_at(current_minute)
        zone_heat_w = (
            (outdoor_c - zone_c) / scenario.r_out_zone_k_per_w
            + (wall_c - zone_c) / scenario.r_zone_wall_k_per_w
            + internal_load_w
            - cooling_w
        )
        wall_heat_w = (
            (outdoor_c - wall_c) / scenario.r_out_wall_k_per_w
            + (zone_c - wall_c) / scenario.r_zone_wall_k_per_w
        )
        cooling_rate = (delayed_u * scenario.cooling_capacity_w - cooling_w) / tau_seconds
        return np.asarray(
            [
                zone_heat_w / scenario.c_zone_j_per_k,
                wall_heat_w / scenario.c_wall_j_per_k,
                cooling_rate,
            ],
            dtype=float,
        )

    seconds = minute * 60.0
    solved = solve_ivp(
        rhs,
        (float(seconds[0]), float(seconds[-1])),
        np.asarray([initial_zone, initial_wall, initial_cooling], dtype=float),
        t_eval=seconds,
        method="DOP853",
        rtol=1e-9,
        atol=1e-11,
        max_step=10.0,
    )
    if not solved.success:
        raise RuntimeError(f"continuous reference solve failed: {solved.message}")
    return {"minute": minute, "zone_c": solved.y[0], "wall_c": solved.y[1], "cooling_w": solved.y[2]}


def _fopdt_step_response(scenario: Scenario) -> dict[str, np.ndarray | float]:
    s = replace(scenario.constant_copy(duration_hours=24.0), sensor_noise_std_c=0.0)
    model = identify_fopdt(s)
    resistance = effective_outdoor_resistance(s)
    heat_at_setpoint = (s.outdoor_c - s.setpoint_c) / resistance + s.internal_load_w
    initial_u = float(np.clip(heat_at_setpoint / s.cooling_capacity_w, 0.15, 0.70))
    step = float(min(0.12, 0.90 - initial_u))
    if step < 0.03:
        step = 0.03
        initial_u = 0.80
    initial_zone, initial_wall = thermal_equilibrium(s, initial_u, s.outdoor_c, s.internal_load_w)
    plant = ThermalPlant3R2C(s)
    plant.reset(zone_c=initial_zone, wall_c=initial_wall, initial_u=initial_u)
    minute = np.arange(s.steps, dtype=float) * s.dt_minutes
    physical = np.empty_like(minute)
    for index in range(s.steps):
        physical[index], _ = plant.step(initial_u + step, s.outdoor_c, s.internal_load_w)
    elapsed = np.maximum(minute - model.delay_minutes, 0.0)
    proxy = initial_zone - model.process_gain_c_per_u * step * (
        1.0 - np.exp(-elapsed / model.time_constant_minutes)
    )
    return {
        "minute": minute,
        "physical_c": physical,
        "proxy_c": proxy,
        "initial_c": initial_zone,
        "step": step,
        "process_gain": model.process_gain_c_per_u,
        "tau": model.time_constant_minutes,
        "delay": model.delay_minutes,
    }


def _plot_physical_validation(series: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    fig, axes = plt.subplots(len(series), 2, figsize=(11.5, 3.1 * len(series)), sharex=False)
    for row, (case_name, values) in enumerate(series.items()):
        hour = values["minute"] / 60.0
        axes[row, 0].plot(hour, values["continuous_c"], color="#111827", linewidth=2.0, label="连续方程参考（SciPy）")
        axes[row, 0].plot(hour, values["discrete_c"], color="#2563EB", linestyle="--", linewidth=1.4, label="项目 1 分钟离散模型")
        axes[row, 0].set_title(case_name)
        axes[row, 0].set_ylabel("室内温度（°C）")
        axes[row, 0].grid(alpha=0.22)
        axes[row, 0].legend(fontsize=8)
        error = values["discrete_c"] - values["continuous_c"]
        axes[row, 1].plot(hour, error, color="#DC2626", linewidth=1.3)
        axes[row, 1].axhline(0.0, color="#111827", linewidth=0.8)
        axes[row, 1].set_title("离散模型 − 连续参考")
        axes[row, 1].set_ylabel("温度差（°C）")
        axes[row, 1].grid(alpha=0.22)
        axes[row, 0].set_xlabel("仿真时间（小时）")
        axes[row, 1].set_xlabel("仿真时间（小时）")
    fig.suptitle("热模型数值交叉验证：独立连续求解器 vs 项目离散推进", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_fopdt_validation(series: dict[str, dict[str, np.ndarray | float]], path: Path) -> None:
    fig, axes = plt.subplots(len(series), 1, figsize=(11.5, 3.0 * len(series)), sharex=False)
    axes = np.atleast_1d(axes)
    for axis, (case_name, values) in zip(axes, series.items(), strict=True):
        hour = np.asarray(values["minute"]) / 60.0
        axis.plot(hour, np.asarray(values["physical_c"]), color="#0F766E", linewidth=1.8, label="两状态物理热模型")
        axis.plot(hour, np.asarray(values["proxy_c"]), color="#D97706", linestyle="--", linewidth=1.5, label="辨识后的 FOPDT 代理")
        axis.set_title(case_name)
        axis.set_ylabel("室内温度（°C）")
        axis.set_xlabel("阶跃试验时间（小时）")
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
    fig.suptitle("控制代理交叉验证：FOPDT 与两状态物理模型的阶跃响应", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _find_openmodelica() -> Path | None:
    """Locate omc without requiring the installer to modify PATH."""

    path_candidate = shutil.which("omc")
    env_home = os.environ.get("OPENMODELICAHOME", "")
    env_bin = os.environ.get("OPENMODELICA_BIN", "")
    candidates = [
        Path(path_candidate) if path_candidate else None,
        Path(env_bin) if env_bin else None,
        Path(env_home) / "bin" / ("omc.exe" if os.name == "nt" else "omc") if env_home else None,
        Path(os.environ.get("MODELICA_OMC", "")) if os.environ.get("MODELICA_OMC") else None,
    ]
    return next((candidate for candidate in candidates if candidate is not None and candidate.exists()), None)


def _modelica_table_value(second: float, points: tuple[tuple[float, float], ...]) -> float:
    """Match Modelica TimeTable's piecewise-linear interpolation."""

    if second <= points[0][0]:
        return points[0][1]
    for (left_t, left_y), (right_t, right_y) in zip(points, points[1:], strict=True):
        if second <= right_t:
            fraction = (second - left_t) / max(right_t - left_t, 1e-12)
            return left_y + fraction * (right_y - left_y)
    return points[-1][1]


def _read_openmodelica_result(path: Path) -> dict[str, np.ndarray]:
    """Read only the evidence columns and collapse pre/post-event duplicate rows."""

    rows_by_time: dict[float, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"time", "zone.T", "wall.T", "actuator.y", "commandInput.y", "outdoorK.y", "totalLoad.y"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"OpenModelica CSV missing columns: {sorted(missing)}")
        for row in reader:
            rows_by_time[float(row["time"])] = row
    ordered = [rows_by_time[second] for second in sorted(rows_by_time)]
    return {
        "second": np.asarray([float(row["time"]) for row in ordered]),
        "zone_c": np.asarray([float(row["zone.T"]) - 273.15 for row in ordered]),
        "wall_c": np.asarray([float(row["wall.T"]) - 273.15 for row in ordered]),
        "actuator": np.asarray([float(row["actuator.y"]) for row in ordered]),
        "command": np.asarray([float(row["commandInput.y"]) for row in ordered]),
        "outdoor_c": np.asarray([float(row["outdoorK.y"]) - 273.15 for row in ordered]),
        "load_w": np.asarray([float(row["totalLoad.y"]) for row in ordered]),
    }


def _run_modelica_matching_reference(second: np.ndarray) -> dict[str, np.ndarray]:
    """Solve the exact Modelica validation profile independently with SciPy."""

    scenario = replace(dynamic_demo_scenario(), sensor_noise_std_c=0.0)
    occupied_points = ((0.0, 0.0), (5399.0, 0.0), (5400.0, 1100.0), (34199.0, 1100.0), (34200.0, 0.0), (43200.0, 0.0))
    door_points = ((0.0, 0.0), (17999.0, 0.0), (18000.0, 2600.0), (18720.0, 2600.0), (18721.0, 0.0), (43200.0, 0.0))
    initial_command = 0.45
    command_after = 0.57
    command_step_second = 7200.0
    delayed_step_second = command_step_second + 60.0 * scenario.actuator_delay_minutes
    actuator_tau_second = 60.0 * scenario.actuator_tau_minutes

    def rhs(current_second: float, state: np.ndarray) -> np.ndarray:
        zone_c, wall_c, actuator = state
        outdoor_c = 34.0 + 4.0 * np.sin(2.0 * np.pi * current_second / 86400.0 - np.pi / 12.0)
        load_w = 550.0 + _modelica_table_value(current_second, occupied_points) + _modelica_table_value(current_second, door_points)
        delayed_command = initial_command if current_second < delayed_step_second else command_after
        cooling_w = scenario.cooling_capacity_w * actuator
        return np.asarray(
            [
                (
                    (outdoor_c - zone_c) / scenario.r_out_zone_k_per_w
                    + (wall_c - zone_c) / scenario.r_zone_wall_k_per_w
                    + load_w
                    - cooling_w
                )
                / scenario.c_zone_j_per_k,
                (
                    (outdoor_c - wall_c) / scenario.r_out_wall_k_per_w
                    + (zone_c - wall_c) / scenario.r_zone_wall_k_per_w
                )
                / scenario.c_wall_j_per_k,
                (delayed_command - actuator) / actuator_tau_second,
            ]
        )

    solved = solve_ivp(
        rhs,
        (float(second[0]), float(second[-1])),
        np.asarray([28.5, 30.15, initial_command]),
        t_eval=second,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        max_step=3.0,
    )
    if not solved.success:
        raise RuntimeError(f"OpenModelica matching reference failed: {solved.message}")
    return {"zone_c": solved.y[0], "wall_c": solved.y[1], "actuator": solved.y[2]}


def _plot_openmodelica_validation(
    modelica: dict[str, np.ndarray], reference: dict[str, np.ndarray], path: Path
) -> None:
    hour = modelica["second"] / 3600.0
    error = modelica["zone_c"] - reference["zone_c"]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.6), sharex=True)
    axes[0].plot(hour, modelica["zone_c"], color="#111827", linewidth=2.0, label="OpenModelica / DASSL")
    axes[0].plot(hour, reference["zone_c"], color="#2563EB", linestyle="--", linewidth=1.4, label="Python 独立连续方程 / DOP853")
    axes[0].set_ylabel("室内温度（°C）")
    axes[0].set_title("同参数、同天气、同热负荷、同制冷指令的温度响应")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.22)
    axes[1].plot(hour, error, color="#DC2626", linewidth=1.2)
    axes[1].axhline(0.0, color="#111827", linewidth=0.8)
    axes[1].set_ylabel("温度差（°C）")
    axes[1].set_title("OpenModelica − Python 连续参考")
    axes[1].grid(alpha=0.22)
    load_axis = axes[2].twinx()
    axes[2].plot(hour, modelica["command"] * 100.0, color="#0F766E", linewidth=1.5, label="制冷容量指令")
    load_axis.plot(hour, modelica["load_w"], color="#D97706", linewidth=1.2, label="总内部热负荷")
    axes[2].set_ylabel("制冷容量指令（%）")
    load_axis.set_ylabel("总内部热负荷（W）")
    axes[2].set_xlabel("仿真时间（小时）")
    axes[2].set_title("交叉验证输入：2 h 指令阶跃、人员负荷与 5 h 开门扰动")
    axes[2].grid(alpha=0.22)
    lines = axes[2].lines + load_axis.lines
    axes[2].legend(lines, [line.get_label() for line in lines], loc="upper right", fontsize=9)
    fig.suptitle("OpenModelica 实际运行交叉验证", y=1.005, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _openmodelica_cross_validation(output: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    result_path = output / "modelica" / "PrecisionCabinetCooling_res.csv"
    if not result_path.exists():
        return [], []
    modelica = _read_openmodelica_result(result_path)
    reference = _run_modelica_matching_reference(modelica["second"])
    difference = modelica["zone_c"] - reference["zone_c"]
    rmse = float(np.sqrt(np.mean(difference**2)))
    maximum = float(np.max(np.abs(difference)))
    final = float(abs(difference[-1]))
    passed = rmse <= 0.01 and maximum <= 0.03
    rows = [
        {
            "validation": "OpenModelica DASSL vs Python DOP853连续方程",
            "case": "12小时混合动态开环验证",
            "samples": int(modelica["second"].size),
            "rmse_c": rmse,
            "max_abs_error_c": maximum,
            "final_abs_error_c": final,
            "threshold": "RMSE≤0.01°C且最大绝对误差≤0.03°C",
            "passed": int(passed),
            "modelica_result": str(result_path.resolve()),
        }
    ]
    timeseries = [
        {
            "second": float(modelica["second"][index]),
            "hour": float(modelica["second"][index] / 3600.0),
            "modelica_zone_c": float(modelica["zone_c"][index]),
            "python_reference_zone_c": float(reference["zone_c"][index]),
            "error_c": float(difference[index]),
            "outdoor_c": float(modelica["outdoor_c"][index]),
            "internal_load_w": float(modelica["load_w"][index]),
            "command": float(modelica["command"][index]),
            "actuator": float(modelica["actuator"][index]),
        }
        for index in range(modelica["second"].size)
    ]
    _plot_openmodelica_validation(modelica, reference, output / "openmodelica_cross_validation.png")
    return rows, timeseries


def run_cross_validation(output_dir: str | Path) -> dict[str, object]:
    """Run reproducible numerical and model-order cross-validation experiments."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    physical_rows: list[dict[str, object]] = []
    physical_timeseries: list[dict[str, object]] = []
    physical_series: dict[str, dict[str, np.ndarray]] = {}
    fopdt_rows: list[dict[str, object]] = []
    fopdt_timeseries: list[dict[str, object]] = []
    fopdt_series: dict[str, dict[str, np.ndarray | float]] = {}

    for case_name, raw_scenario in typical_case_scenarios().items():
        scenario = replace(raw_scenario, sensor_noise_std_c=0.0)
        discrete = _run_discrete_open_loop(scenario, case_name)
        continuous = _run_continuous_reference(scenario, case_name, discrete["minute"])
        difference = discrete["zone_c"] - continuous["zone_c"]
        rmse = float(np.sqrt(np.mean(difference**2)))
        maximum = float(np.max(np.abs(difference)))
        final = float(abs(difference[-1]))
        passed = maximum <= 0.15
        physical_rows.append(
            {
                "validation": "1分钟离散3R2C vs SciPy连续方程",
                "case": case_name,
                "samples": int(scenario.steps),
                "rmse_c": rmse,
                "max_abs_error_c": maximum,
                "final_abs_error_c": final,
                "threshold": "最大绝对误差≤0.15°C",
                "passed": int(passed),
            }
        )
        physical_series[case_name] = {
            "minute": discrete["minute"],
            "discrete_c": discrete["zone_c"],
            "continuous_c": continuous["zone_c"],
        }
        for index, minute in enumerate(discrete["minute"]):
            physical_timeseries.append(
                {
                    "case": case_name,
                    "minute": float(minute),
                    "command": float(discrete["command"][index]),
                    "discrete_zone_c": float(discrete["zone_c"][index]),
                    "continuous_zone_c": float(continuous["zone_c"][index]),
                    "error_c": float(difference[index]),
                }
            )

        fopdt = _fopdt_step_response(scenario)
        proxy_error = np.asarray(fopdt["proxy_c"]) - np.asarray(fopdt["physical_c"])
        response_size = max(float(fopdt["process_gain"]) * float(fopdt["step"]), 1e-9)
        proxy_rmse = float(np.sqrt(np.mean(proxy_error**2)))
        normalized_rmse = 100.0 * proxy_rmse / response_size
        proxy_maximum = float(np.max(np.abs(proxy_error)))
        proxy_passed = normalized_rmse <= 15.0
        fopdt_rows.append(
            {
                "validation": "FOPDT代理 vs 3R2C阶跃响应",
                "case": case_name,
                "process_gain_c_per_u": float(fopdt["process_gain"]),
                "time_constant_min": float(fopdt["tau"]),
                "delay_min": float(fopdt["delay"]),
                "rmse_c": proxy_rmse,
                "normalized_rmse_pct": normalized_rmse,
                "max_abs_error_c": proxy_maximum,
                "threshold": "归一化RMSE≤15%",
                "passed": int(proxy_passed),
            }
        )
        fopdt_series[case_name] = fopdt
        for index, minute in enumerate(np.asarray(fopdt["minute"])):
            fopdt_timeseries.append(
                {
                    "case": case_name,
                    "minute": float(minute),
                    "physical_zone_c": float(np.asarray(fopdt["physical_c"])[index]),
                    "fopdt_zone_c": float(np.asarray(fopdt["proxy_c"])[index]),
                    "error_c": float(proxy_error[index]),
                }
            )

    _plot_physical_validation(physical_series, output / "physical_cross_validation.png")
    _plot_fopdt_validation(fopdt_series, output / "fopdt_cross_validation.png")
    openmodelica_rows, openmodelica_timeseries = _openmodelica_cross_validation(output)
    omc = _find_openmodelica()
    if omc is not None:
        try:
            omc_version = subprocess.run(
                [str(omc), "--version"], capture_output=True, text=True, timeout=15, check=False
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            omc_version = "版本读取失败"
        openmodelica_status = f"{omc_version}；可执行文件 {omc}"
    else:
        openmodelica_status = "未检测到 omc 可执行文件"
    modelica_run_status = (
        "成功：实际使用 DASSL 运行 12 h / 62 方程模型，结果已与 Python 连续方程交叉比较"
        if openmodelica_rows
        else f"未运行：{output / 'modelica' / 'PrecisionCabinetCooling_res.csv'} 不存在（历史证据见 outputs/modelica/PrecisionCabinetCooling_res.csv，当前未复现）"
    )
    environment = {
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "continuous_solver": "SciPy solve_ivp / DOP853, rtol=1e-9, atol=1e-11, max_step=10s",
        "openmodelica": openmodelica_status,
        "openmodelica_run": modelica_run_status,
        "openmodelica_solver": "DASSL, tolerance=1e-8, output interval=6s, duration=12h",
    }
    return {
        "physical_rows": physical_rows,
        "physical_timeseries": physical_timeseries,
        "fopdt_rows": fopdt_rows,
        "fopdt_timeseries": fopdt_timeseries,
        "openmodelica_rows": openmodelica_rows,
        "openmodelica_timeseries": openmodelica_timeseries,
        "environment": environment,
    }
