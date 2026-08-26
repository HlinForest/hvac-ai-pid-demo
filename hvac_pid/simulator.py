from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Scenario
from .controllers import PIController
from .plant import ThermalPlant3R2C
from .actuator import CompressorCommandLimiter


@dataclass
class SimulationResult:
    minute: np.ndarray
    zone_c: np.ndarray
    wall_c: np.ndarray
    setpoint_c: np.ndarray
    outdoor_c: np.ndarray
    internal_load_w: np.ndarray
    measurement_c: np.ndarray
    requested_command: np.ndarray
    command: np.ndarray
    cooling_w: np.ndarray
    kp: np.ndarray
    ki: np.ndarray
    inference_us: np.ndarray
    fallback_active: np.ndarray
    minimum_running_command: float
    command_slew_limit_per_minute: float


def simulate(scenario: Scenario, controller: PIController, seed: int = 0) -> SimulationResult:
    plant = ThermalPlant3R2C(scenario)
    limiter = CompressorCommandLimiter(scenario)
    controller.reset()
    rng = np.random.default_rng(seed)

    minute = np.arange(scenario.steps, dtype=float) * scenario.dt_minutes
    zone = np.empty(scenario.steps)
    wall = np.empty(scenario.steps)
    setpoint = np.empty(scenario.steps)
    outdoor = np.empty(scenario.steps)
    load = np.empty(scenario.steps)
    measurement = np.empty(scenario.steps)
    requested_command = np.empty(scenario.steps)
    command = np.empty(scenario.steps)
    cooling = np.empty(scenario.steps)
    kp = np.empty(scenario.steps)
    ki = np.empty(scenario.steps)
    inference_us = np.zeros(scenario.steps)
    fallback_active = np.zeros(scenario.steps, dtype=bool)

    filtered_measurement = float(plant.zone_c)
    for index, current_minute in enumerate(minute):
        outdoor[index] = scenario.outdoor_at(float(current_minute))
        load[index] = scenario.load_at(float(current_minute))
        setpoint[index] = scenario.setpoint_at(float(current_minute))
        raw_measurement = plant.zone_c + rng.normal(0.0, scenario.sensor_noise_std_c)
        filter_alpha = scenario.dt_minutes / max(
            scenario.sensor_filter_tau_minutes + scenario.dt_minutes,
            scenario.dt_minutes,
        )
        filtered_measurement += filter_alpha * (raw_measurement - filtered_measurement)
        measurement[index] = filtered_measurement
        error = filtered_measurement - setpoint[index]
        requested_command[index] = controller.update(
            float(error),
            scenario.dt_minutes,
            minute=float(current_minute),
            measurement_c=float(filtered_measurement),
            setpoint_c=float(setpoint[index]),
            outdoor_c=float(outdoor[index]),
            internal_load_w=float(load[index]),
            applied_command=float(limiter.command),
            scenario=scenario,
        )
        command[index] = limiter.update(requested_command[index], scenario.dt_minutes)
        diagnostics = getattr(controller, "diagnostics", lambda: {})()
        kp[index] = float(diagnostics.get("kp", controller.kp))
        ki[index] = float(diagnostics.get("ki", controller.ki))
        inference_us[index] = float(diagnostics.get("inference_us", 0.0))
        fallback_active[index] = bool(diagnostics.get("fallback_active", False))
        zone[index], wall[index] = plant.step(command[index], outdoor[index], load[index])
        cooling[index] = plant.cooling_w

    return SimulationResult(
        minute, zone, wall, setpoint, outdoor, load, measurement,
        requested_command, command, cooling, kp, ki, inference_us, fallback_active,
        scenario.minimum_running_command, scenario.command_slew_rate_per_minute,
    )
