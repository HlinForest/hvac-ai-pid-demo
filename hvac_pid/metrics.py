from __future__ import annotations

import numpy as np

from .simulator import SimulationResult


def calculate_metrics(result: SimulationResult, comfort_band_c: float = 0.5) -> dict[str, float]:
    error = result.zone_c - result.setpoint_c
    dt_hours = float(np.median(np.diff(result.minute))) / 60.0
    abs_error = np.abs(error)
    comfort_excess = np.maximum(abs_error - comfort_band_c, 0.0)
    undershoot = np.maximum(-error, 0.0)
    overheat = np.maximum(error, 0.0)
    movement = np.abs(np.diff(result.command, prepend=result.command[0]))
    itae = float(np.sum(result.minute / 60.0 * abs_error) * dt_hours)
    command_variance = float(np.var(result.command))
    stable = bool(np.all(np.isfinite(result.zone_c)) and np.all((result.zone_c > 5.0) & (result.zone_c < 45.0)))

    within = abs_error <= comfort_band_c
    settling_hours = result.minute[-1] / 60.0
    # First point after which the signal remains in-band for at least 60 min.
    window = max(1, int(round(60.0 / max(np.median(np.diff(result.minute)), 1e-6))))
    for idx in range(max(0, len(within) - window)):
        if np.all(within[idx : idx + window]):
            settling_hours = result.minute[idx] / 60.0
            break

    # Event-local recovery: for a setpoint step, count from the setpoint change;
    # for a heat pulse, count from the end of the largest positive load jump.
    # If the case has neither, this reduces to the ordinary startup settling time.
    recovery_start = 0
    setpoint_changes = np.flatnonzero(np.abs(np.diff(result.setpoint_c)) > 1e-9) + 1
    load_delta = np.diff(result.internal_load_w)
    positive_load_changes = np.flatnonzero(load_delta > 1e-9) + 1
    if positive_load_changes.size:
        event_start = int(positive_load_changes[np.argmax(load_delta[positive_load_changes - 1])])
        later_negative = np.flatnonzero(load_delta[event_start:] < -1e-9) + event_start + 1
        recovery_start = int(later_negative[0]) if later_negative.size else event_start
    elif setpoint_changes.size:
        recovery_start = int(setpoint_changes[-1])

    recovery_hours = (result.minute[-1] - result.minute[recovery_start]) / 60.0
    disturbance_recovered = False
    for idx in range(recovery_start, max(recovery_start, len(within) - window)):
        if np.all(within[idx : idx + window]):
            recovery_hours = (result.minute[idx] - result.minute[recovery_start]) / 60.0
            disturbance_recovered = True
            break

    metrics = {
        "rmse_c": float(np.sqrt(np.mean(error**2))),
        "iae_c_hour": float(np.sum(abs_error) * dt_hours),
        "itae_c_hour2": itae,
        "comfort_violation_c_hour": float(np.sum(comfort_excess) * dt_hours),
        "max_undershoot_c": float(np.max(undershoot)),
        "max_overheat_c": float(np.max(overheat)),
        "settling_time_hour": float(settling_hours),
        "disturbance_recovery_time_hour": float(recovery_hours),
        "disturbance_recovered": float(disturbance_recovered),
        "cooling_energy_kwh": float(np.sum(result.cooling_w) * dt_hours / 1000.0),
        "control_movement": float(np.sum(movement)),
        "compressor_output_variance": command_variance,
        "mean_ai_inference_us": float(np.mean(result.inference_us[result.inference_us > 0])) if np.any(result.inference_us > 0) else 0.0,
        "fallback_events": float(np.count_nonzero(np.diff(result.fallback_active.astype(int), prepend=0) > 0)),
        "stable": float(stable),
    }
    metrics["objective"] = objective_from_metrics(metrics)
    return metrics


def objective_from_metrics(metrics: dict[str, float]) -> float:
    if metrics["stable"] < 0.5:
        return 1e6
    return float(
        2.0 * metrics["iae_c_hour"]
        + 1.5 * metrics["itae_c_hour2"]
        + 10.0 * metrics["comfort_violation_c_hour"]
        + 3.0 * metrics["max_undershoot_c"]
        + 0.35 * metrics["settling_time_hour"]
        + 0.08 * metrics["control_movement"]
        + 0.7 * metrics["compressor_output_variance"]
        + 0.015 * metrics["cooling_energy_kwh"]
    )
