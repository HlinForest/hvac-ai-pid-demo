from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import least_squares

from .config import Scenario
from .plant import ThermalPlant3R2C, effective_outdoor_resistance, thermal_equilibrium


class GainPredictor(Protocol):
    def predict_gains(self, context: np.ndarray) -> tuple[float, float, bool]: ...


class PIController:
    """Cooling PI: positive (zone - setpoint) error requests more cooling."""

    def __init__(self, kp: float, ki: float):
        self.kp = float(kp)
        self.ki = float(ki)
        self.integral = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def update(self, error_c: float, dt_minutes: float, **_: object) -> float:
        candidate_integral = self.integral + error_c * dt_minutes
        raw_candidate = self.kp * error_c + self.ki * candidate_integral
        saturated = float(np.clip(raw_candidate, 0.0, 1.0))
        # Conditional integration: integrate only when unsaturated or when the
        # present error would move an already-saturated output back inward.
        integrate = (
            0.0 <= raw_candidate <= 1.0
            or (raw_candidate > 1.0 and error_c < 0.0)
            or (raw_candidate < 0.0 and error_c > 0.0)
        )
        if integrate:
            self.integral = candidate_integral
        raw = self.kp * error_c + self.ki * self.integral
        return float(np.clip(raw, 0.0, 1.0))


class ScheduledPIController(PIController):
    """Slow gain scheduling with output bounds, slew limits, and fallback.

    Legacy/experimental path only (see docs/ALGORITHM_GUIDE.md): formal v3/v4
    reports and Streamlit comparisons do NOT use this class.  Its
    ``max_fractional_change=0.35`` is intentionally distinct from the
    deployed v4 default ``MAX_FRACTIONAL_GAIN_CHANGE=0.10`` in
    :mod:`hvac_pid.safety` and must not be quoted as the safety bound.
    """

    def __init__(
        self,
        predictor: GainPredictor,
        fallback_gains: tuple[float, float],
        update_interval_minutes: float = 30.0,
        max_fractional_change: float = 0.35,
    ):
        super().__init__(*fallback_gains)
        self.predictor = predictor
        self.fallback_gains = fallback_gains
        self.update_interval_minutes = update_interval_minutes
        self.max_fractional_change = max_fractional_change
        self._last_update_minute = -np.inf
        self.gain_history: list[tuple[float, float, float, bool]] = []

    def reset(self) -> None:
        super().reset()
        self.kp, self.ki = self.fallback_gains
        self._last_update_minute = -np.inf
        self.gain_history = []

    @staticmethod
    def _slew(old: float, target: float, fraction: float) -> float:
        lower = old * (1.0 - fraction)
        upper = old * (1.0 + fraction)
        return float(np.clip(target, lower, upper))

    def update(self, error_c: float, dt_minutes: float, **kwargs: object) -> float:
        minute = float(kwargs["minute"])
        scenario = kwargs["scenario"]
        if not isinstance(scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        if minute - self._last_update_minute >= self.update_interval_minutes:
            context = scenario.context_vector(
                zone_c=float(kwargs["measurement_c"]),
                outdoor_c=float(kwargs["outdoor_c"]),
                setpoint_c=float(kwargs["setpoint_c"]),
                internal_load_w=float(kwargs["internal_load_w"]),
            )
            target_kp, target_ki, out_of_domain = self.predictor.predict_gains(context)
            if out_of_domain:
                target_kp, target_ki = self.fallback_gains
            if np.isneginf(self._last_update_minute):
                # The commissioning prediction may be applied at startup. Later
                # changes are deliberately slow so the supervisory layer cannot
                # create an abrupt inner-loop gain jump.
                self.kp, self.ki = float(target_kp), float(target_ki)
            else:
                self.kp = self._slew(self.kp, target_kp, self.max_fractional_change)
                self.ki = self._slew(self.ki, target_ki, self.max_fractional_change)
            self._last_update_minute = minute
            self.gain_history.append((minute, self.kp, self.ki, out_of_domain))
        return super().update(error_c, dt_minutes)


@dataclass(frozen=True)
class FOPDT:
    process_gain_c_per_u: float
    time_constant_minutes: float
    delay_minutes: float


def _fopdt_drop(minute: np.ndarray, amplitude_c: float, tau_minutes: float, delay_minutes: float) -> np.ndarray:
    elapsed = np.maximum(np.asarray(minute, dtype=float) - delay_minutes, 0.0)
    return amplitude_c * (1.0 - np.exp(-elapsed / max(tau_minutes, 1e-9)))


def identify_fopdt_audit(
    scenario: Scenario,
) -> tuple[FOPDT, list[dict[str, float]], list[dict[str, float]]]:
    """Identify an FOPDT proxy and retain both response and optimizer history.

    The physical plant has two thermal states, so its response is not exactly
    first order.  A long virtual step test is therefore fitted by bounded least
    squares instead of pretending that a short, censored 63.2% crossing is a
    valid time constant.  The fitted FOPDT is only a commissioning proxy for
    Z-N/IMC; all final experiments still run on the 3R2C plant.
    """

    # The slow wall/cabinet state needs several simulated days to expose its
    # tail.  This costs seconds in software, but would not be a practical real-
    # plant commissioning duration.  A real installation would fit BMS history
    # or use a shorter, safety-approved excitation experiment.
    s = scenario.constant_copy(duration_hours=168.0)
    r_eff = effective_outdoor_resistance(s)
    heat_at_setpoint = (s.outdoor_c - s.setpoint_c) / r_eff + s.internal_load_w
    u0 = float(np.clip(heat_at_setpoint / s.cooling_capacity_w, 0.15, 0.70))
    step = float(min(0.12, 0.90 - u0))
    if step < 0.03:
        step = 0.03
        u0 = 0.80

    y0, w0 = thermal_equilibrium(s, u0, s.outdoor_c, s.internal_load_w)
    physical_equilibrium_zone_c, _ = thermal_equilibrium(s, u0 + step, s.outdoor_c, s.internal_load_w)

    plant = ThermalPlant3R2C(s)
    plant.reset(zone_c=y0, wall_c=w0, initial_u=u0)
    times: list[float] = []
    measured_zone: list[float] = []
    for index in range(s.steps):
        minute = index * s.dt_minutes
        zone_c, _ = plant.step(u0 + step, s.outdoor_c, s.internal_load_w)
        times.append(minute)
        measured_zone.append(zone_c)

    minute_array = np.asarray(times, dtype=float)
    measured_drop = y0 - np.asarray(measured_zone, dtype=float)
    sample_stride = max(1, int(round(5.0 / s.dt_minutes)))
    fit_minute = minute_array[::sample_stride]
    fit_drop = measured_drop[::sample_stride]
    amplitude_guess = max(float(np.mean(measured_drop[-max(2, int(round(60.0 / s.dt_minutes))):])), 0.01)
    tau_guess = max(s.duration_hours * 60.0 / 12.0, 4.0 * s.actuator_tau_minutes, 2.0 * s.dt_minutes)
    delay_guess = max(s.actuator_delay_minutes, s.dt_minutes)
    upper_amplitude = max(2.0 * float(np.max(measured_drop)), 2.0 * amplitude_guess, 1.0)
    upper_tau = 2.0 * s.duration_hours * 60.0
    upper_delay = max(delay_guess + 60.0, 2.0 * delay_guess)
    fit_history: list[dict[str, float]] = []
    best_rmse = float("inf")
    best_parameters = np.asarray([amplitude_guess, tau_guess, delay_guess], dtype=float)

    def residual(parameters: np.ndarray) -> np.ndarray:
        nonlocal best_rmse, best_parameters
        amplitude_c, tau_minutes, delay_minutes = map(float, parameters)
        prediction = _fopdt_drop(fit_minute, amplitude_c, tau_minutes, delay_minutes)
        error = prediction - fit_drop
        rmse_c = float(np.sqrt(np.mean(np.square(error))))
        if rmse_c < best_rmse:
            best_rmse = rmse_c
            best_parameters = np.asarray(parameters, dtype=float).copy()
        fit_history.append(
            {
                "evaluation": float(len(fit_history) + 1),
                "candidate_amplitude_c": amplitude_c,
                "candidate_process_gain_c_per_u": amplitude_c / step,
                "candidate_tau_minutes": tau_minutes,
                "candidate_delay_minutes": delay_minutes,
                "rmse_c": rmse_c,
                "best_rmse_c": best_rmse,
                "best_process_gain_c_per_u": float(best_parameters[0] / step),
                "best_tau_minutes": float(best_parameters[1]),
                "best_delay_minutes": float(best_parameters[2]),
            }
        )
        return error

    fitted = least_squares(
        residual,
        x0=np.asarray([amplitude_guess, tau_guess, delay_guess], dtype=float),
        bounds=(
            np.asarray([0.01, 2.0 * s.dt_minutes, delay_guess], dtype=float),
            np.asarray([upper_amplitude, upper_tau, upper_delay], dtype=float),
        ),
        max_nfev=200,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    fitted_amplitude, tau, delay = map(float, fitted.x)
    process_gain = max(fitted_amplitude / step, 1e-3)
    predicted_drop = _fopdt_drop(minute_array, fitted_amplitude, tau, delay)
    fit_rmse = float(np.sqrt(np.mean(np.square(predicted_drop - measured_drop))))
    measured_fraction = measured_drop / max(fitted_amplitude, 1e-9)

    def crossing(target: float) -> float:
        for idx, value in enumerate(measured_fraction):
            if value >= target:
                return times[idx]
        return float("nan")

    t28 = crossing(0.283)
    t63 = crossing(0.632)
    model = FOPDT(process_gain, float(tau), float(delay))
    history: list[dict[str, float]] = []
    final_slope_c_per_hour = float(
        (measured_zone[-1] - measured_zone[-1 - max(1, int(round(60.0 / s.dt_minutes)))])
    )
    for minute, zone_c, fraction, fitted_drop in zip(
        times, measured_zone, measured_fraction, predicted_drop, strict=True
    ):
        history.append(
            {
                "minute": float(minute),
                "baseline_command": float(u0),
                "step_command": float(u0 + step),
                "command_step": float(step),
                "measured_zone_c": float(zone_c),
                "measured_fraction": float(fraction),
                "fopdt_fraction": float(fitted_drop / max(fitted_amplitude, 1e-9)),
                "initial_zone_c": float(y0),
                "steady_zone_c": float(y0 - fitted_amplitude),
                "physical_equilibrium_zone_c": float(physical_equilibrium_zone_c),
                "process_gain_c_per_u": float(process_gain),
                "time_constant_minutes": float(tau),
                "delay_minutes": float(delay),
                "t28_minutes": float(t28),
                "t63_minutes": float(t63),
                "fit_rmse_c": fit_rmse,
                "fit_normalized_rmse_pct": float(100.0 * fit_rmse / max(fitted_amplitude, 1e-9)),
                "identification_duration_hours": float(s.duration_hours),
                "final_measured_fraction": float(measured_fraction[-1]),
                "final_slope_c_per_hour": final_slope_c_per_hour,
            }
        )
    return model, history, fit_history


def identify_fopdt_with_history(scenario: Scenario) -> tuple[FOPDT, list[dict[str, float]]]:
    """Return the fitted FOPDT model and full virtual step-response history."""
    model, history, _ = identify_fopdt_audit(scenario)
    return model, history


def identify_fopdt(scenario: Scenario) -> FOPDT:
    """Return the FOPDT model; use ``identify_fopdt_with_history`` for audit plots."""
    model, _ = identify_fopdt_with_history(scenario)
    return model


def classical_tuning_calculation_steps(
    scenario: Scenario,
    model: FOPDT,
    identification_history: list[dict[str, float]],
    fit_history: list[dict[str, float]],
) -> list[dict[str, object]]:
    """Build an audit table from nominal conditions to deployed Z-N/IMC gains."""
    s = scenario.constant_copy(duration_hours=168.0)
    first = identification_history[0]
    r_eff = effective_outdoor_resistance(s)
    heat_at_setpoint = (s.outdoor_c - s.setpoint_c) / r_eff + s.internal_load_w
    u0 = float(first["baseline_command"])
    u1 = float(first["step_command"])
    delta_u = u1 - u0
    y0 = float(first["initial_zone_c"])
    fitted_y_inf = float(first["steady_zone_c"])
    t28 = float(first["t28_minutes"])
    t63 = float(first["t63_minutes"])
    fit_rmse = float(first["fit_rmse_c"])
    rows: list[dict[str, object]] = []

    def add(method: str, quantity: str, formula: str, substitution: str, result: str, meaning: str) -> None:
        rows.append(
            {
                "step": len(rows) + 1,
                "method": method,
                "quantity": quantity,
                "formula_or_action": formula,
                "substitution": substitution,
                "result": result,
                "meaning": meaning,
            }
        )

    add("共同FOPDT辨识", "名义工况", "固定室外温度、目标、热负荷和容量", f"To={s.outdoor_c:g}°C, Tsp={s.setpoint_c:g}°C, Qint={s.internal_load_w:g}W, Qmax={s.cooling_capacity_w:g}W", "生成恒定辨识工况", "先去掉昼夜温差、开门和设定值变化，避免把扰动误认为对象动态")
    add("共同FOPDT辨识", "等效室外热阻", "Reff=1/[1/Roz+1/(Rzw+Row)]", f"Roz={s.r_out_zone_k_per_w:g}, Rzw={s.r_zone_wall_k_per_w:g}, Row={s.r_out_wall_k_per_w:g} K/W", f"Reff={r_eff:.8g} K/W", "只用于在虚拟环境中建立一个名义基准工作点")
    add("共同FOPDT辨识", "基准热负荷", "Qbal=(To-Tsp)/Reff+Qint", f"({s.outdoor_c:g}-{s.setpoint_c:g})/{r_eff:.8g}+{s.internal_load_w:g}", f"Qbal={heat_at_setpoint:.6g} W", "目标温度附近需要抵消的总热量")
    add("共同FOPDT辨识", "基准制冷指令", "u0=clip(Qbal/Qmax,0.15,0.70)", f"{heat_at_setpoint:.6g}/{s.cooling_capacity_w:g}", f"u0={u0:.8f} ({100*u0:.4f}%)", "真实系统中应由稳定运行记录获得，而不是依赖未知模型参数")
    add("共同FOPDT辨识", "阶跃输入", "u1=u0+Δu", f"Δu={delta_u:.6g}", f"u1={u1:.8f} ({100*u1:.4f}%)", "在虚拟机房中把制冷能力提高12个百分点")
    add("共同FOPDT辨识", "采集阶跃响应", "运行3R2C+执行器并记录 Tz(t)", f"时长={first['identification_duration_hours']:.0f}h, 步长={s.dt_minutes:g}min", f"{len(identification_history)}个温度样本", "长时虚拟试验用于暴露慢墙体/机柜蓄热尾部")
    add("共同FOPDT辨识", "FOPDT曲线", "T_hat(t)=T0-A[1-exp(-(t-L)/tau)], t>L", f"T0={y0:.6g}°C；初值见fopdt_fit_history.csv", "同时拟合A、tau、L", "FOPDT把复杂阶跃响应压缩成三个控制参数，不是原始物理模型")
    add("共同FOPDT辨识", "最小二乘拟合", "min Σ[T_hat(t)-T_meas(t)]²", f"共{len(fit_history)}次残差函数评价", f"RMSE={fit_rmse:.6g}°C", "每次候选K、tau、L及当前最优值全部保存")
    add("共同FOPDT辨识", "过程增益K", "K=A/Δu", f"A={y0-fitted_y_inf:.6g}°C, Δu={delta_u:.6g}", f"K={model.process_gain_c_per_u:.8g} °C/指令", "制冷指令增加1.0时，稳态温度下降幅度的拟合值")
    add("共同FOPDT辨识", "时间常数与延迟", "由有界最小二乘联合估计", f"实测诊断交叉点 t28={t28:.6g}min, t63={t63:.6g}min", f"tau={model.time_constant_minutes:.8g}min, L={model.delay_minutes:.8g}min", "交叉点只作诊断，不再把未到达的试验末点冒充t63")
    add("共同FOPDT辨识", "FOPDT传递函数", "G(s)=-K exp(-Ls)/(tau s+1)", f"K={model.process_gain_c_per_u:.8g}, tau={model.time_constant_minutes:.8g}, L={model.delay_minutes:.8g}", "完成低阶代理", "负号表示制冷指令增大使室温下降")

    zn_kp_raw = 0.9 * model.time_constant_minutes / (model.process_gain_c_per_u * model.delay_minutes)
    zn_ti = 3.33 * model.delay_minutes
    zn_ki_raw = zn_kp_raw / zn_ti
    zn_kp = float(np.clip(zn_kp_raw, 0.002, 1.5))
    zn_ki = float(np.clip(zn_ki_raw, 1e-5, 0.08))
    add("Ziegler-Nichols", "未限幅Kp", "Kp_raw=0.9tau/(KL)", f"0.9×{model.time_constant_minutes:.8g}/({model.process_gain_c_per_u:.8g}×{model.delay_minutes:.8g})", f"Kp_raw={zn_kp_raw:.8g}", "Z-N反应曲线经验公式")
    add("Ziegler-Nichols", "积分时间Ti", "Ti=3.33L", f"3.33×{model.delay_minutes:.8g}", f"Ti={zn_ti:.8g}min", "先算积分时间，再换算代码使用的Ki")
    add("Ziegler-Nichols", "未限幅Ki", "Ki_raw=Kp_raw/Ti", f"{zn_kp_raw:.8g}/{zn_ti:.8g}", f"Ki_raw={zn_ki_raw:.8g} min^-1", "Python PI使用并联形式 u=Kp e+Ki∫e dt")
    add("Ziegler-Nichols", "安全限幅", "Kp=clip(Kp_raw,0.002,1.5); Ki=clip(Ki_raw,1e-5,0.08)", f"Kp_raw={zn_kp_raw:.8g}, Ki_raw={zn_ki_raw:.8g}", f"部署Kp={zn_kp:.8g}, Ki={zn_ki:.8g}", "本例Z-N原值过于激进，因此两个增益都被安全边界截断")

    closed_loop_time = max(model.time_constant_minutes / 3.0, 3.0 * model.delay_minutes, 12.0)
    imc_kp_raw = model.time_constant_minutes / (model.process_gain_c_per_u * (closed_loop_time + model.delay_minutes))
    imc_ti = min(model.time_constant_minutes, 4.0 * (closed_loop_time + model.delay_minutes))
    imc_ki_raw = imc_kp_raw / imc_ti
    imc_kp = float(np.clip(imc_kp_raw, 0.002, 1.5))
    imc_ki = float(np.clip(imc_ki_raw, 1e-5, 0.08))
    add("IMC PI", "闭环速度lambda", "lambda=max(tau/3,3L,12min)", f"max({model.time_constant_minutes:.8g}/3,3×{model.delay_minutes:.8g},12)", f"lambda={closed_loop_time:.8g}min", "lambda越大越保守；这是工程选择，不是神经网络训练")
    add("IMC PI", "未限幅Kp", "Kp_raw=tau/[K(lambda+L)]", f"{model.time_constant_minutes:.8g}/[{model.process_gain_c_per_u:.8g}×({closed_loop_time:.8g}+{model.delay_minutes:.8g})]", f"Kp_raw={imc_kp_raw:.8g}", "由期望闭环速度和FOPDT模型计算")
    add("IMC PI", "积分时间Ti", "Ti=min[tau,4(lambda+L)]", f"min({model.time_constant_minutes:.8g},4×({closed_loop_time:.8g}+{model.delay_minutes:.8g}))", f"Ti={imc_ti:.8g}min", "限制积分环节不要比对象慢得不合理")
    add("IMC PI", "未限幅Ki", "Ki_raw=Kp_raw/Ti", f"{imc_kp_raw:.8g}/{imc_ti:.8g}", f"Ki_raw={imc_ki_raw:.8g} min^-1", "换算为并联PI的积分增益")
    add("IMC PI", "安全限幅", "Kp=clip(Kp_raw,0.002,1.5); Ki=clip(Ki_raw,1e-5,0.08)", f"Kp_raw={imc_kp_raw:.8g}, Ki_raw={imc_ki_raw:.8g}", f"部署Kp={imc_kp:.8g}, Ki={imc_ki:.8g}", "本例IMC原值位于安全边界内，因此限幅前后相同")
    return rows


def ziegler_nichols_pi(model: FOPDT) -> tuple[float, float]:
    kp = 0.9 * model.time_constant_minutes / (model.process_gain_c_per_u * model.delay_minutes)
    ti = 3.33 * model.delay_minutes
    ki = kp / max(ti, 1e-6)
    return float(np.clip(kp, 0.002, 1.5)), float(np.clip(ki, 1e-5, 0.08))


def imc_pi(model: FOPDT, closed_loop_time_minutes: float | None = None) -> tuple[float, float]:
    """Return SIMC/IMC PI gains for a selected closed-loop time ``lambda``.

    With no explicit lambda this keeps the conservative engineering fallback.
    A comparison experiment may tune only lambda on commissioning scenarios;
    the controller remains constrained to the IMC formula instead of becoming
    an unconstrained two-gain optimiser.
    """
    closed_loop_time = (
        max(model.time_constant_minutes / 3.0, 3.0 * model.delay_minutes, 12.0)
        if closed_loop_time_minutes is None
        else max(float(closed_loop_time_minutes), model.delay_minutes)
    )
    kp = model.time_constant_minutes / (
        model.process_gain_c_per_u * (closed_loop_time + model.delay_minutes)
    )
    ti = min(model.time_constant_minutes, 4.0 * (closed_loop_time + model.delay_minutes))
    ki = kp / max(ti, 1e-6)
    return float(np.clip(kp, 0.002, 1.5)), float(np.clip(ki, 1e-5, 0.08))
