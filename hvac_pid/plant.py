from __future__ import annotations

from collections import deque

import numpy as np

from .config import Scenario


class ThermalPlant3R2C:
    """Two-state building model with delayed, first-order cooling actuation."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        delay_steps = max(0, int(round(scenario.actuator_delay_minutes / scenario.dt_minutes)))
        self._delay: deque[float] = deque(maxlen=delay_steps + 1)
        self.zone_c = scenario.initial_zone_c
        self.wall_c = scenario.initial_zone_c
        self.cooling_w = 0.0
        self.reset()

    def reset(
        self,
        zone_c: float | None = None,
        wall_c: float | None = None,
        initial_u: float = 0.0,
    ) -> None:
        self.zone_c = self.scenario.initial_zone_c if zone_c is None else float(zone_c)
        if wall_c is None:
            self.wall_c = 0.7 * self.zone_c + 0.3 * self.scenario.outdoor_c
        else:
            self.wall_c = float(wall_c)
        initial_u = float(np.clip(initial_u, 0.0, 1.0))
        self.cooling_w = initial_u * self.scenario.cooling_capacity_w
        self._delay.clear()
        self._delay.extend([initial_u] * self._delay.maxlen)

    def step(self, u: float, outdoor_c: float, internal_load_w: float) -> tuple[float, float]:
        s = self.scenario
        u = float(np.clip(u, 0.0, 1.0))
        self._delay.append(u)
        delayed_u = self._delay[0]
        target_cooling_w = delayed_u * s.cooling_capacity_w

        substeps = max(1, int(getattr(s, "integration_substeps", 1)))
        dt_seconds = s.dt_minutes * 60.0 / substeps
        tau_seconds = max(s.actuator_tau_minutes * 60.0, dt_seconds)
        for _ in range(substeps):
            self.cooling_w += (target_cooling_w - self.cooling_w) * dt_seconds / tau_seconds

            zone_heat_w = (
                (outdoor_c - self.zone_c) / s.r_out_zone_k_per_w
                + (self.wall_c - self.zone_c) / s.r_zone_wall_k_per_w
                + internal_load_w
                - self.cooling_w
            )
            wall_heat_w = (
                (outdoor_c - self.wall_c) / s.r_out_wall_k_per_w
                + (self.zone_c - self.wall_c) / s.r_zone_wall_k_per_w
            )

            self.zone_c += zone_heat_w * dt_seconds / s.c_zone_j_per_k
            self.wall_c += wall_heat_w * dt_seconds / s.c_wall_j_per_k
        return float(self.zone_c), float(self.wall_c)


def thermal_equilibrium(s: Scenario, u: float, outdoor_c: float, internal_load_w: float) -> tuple[float, float]:
    """Solve the static 3R2C temperatures for a fixed command and disturbance."""

    a = np.asarray(
        [
            [1.0 / s.r_out_zone_k_per_w + 1.0 / s.r_zone_wall_k_per_w, -1.0 / s.r_zone_wall_k_per_w],
            [-1.0 / s.r_zone_wall_k_per_w, 1.0 / s.r_out_wall_k_per_w + 1.0 / s.r_zone_wall_k_per_w],
        ]
    )
    b = np.asarray(
        [
            outdoor_c / s.r_out_zone_k_per_w + internal_load_w - s.cooling_capacity_w * u,
            outdoor_c / s.r_out_wall_k_per_w,
        ]
    )
    zone_c, wall_c = np.linalg.solve(a, b)
    return float(zone_c), float(wall_c)


def effective_outdoor_resistance(s: Scenario) -> float:
    direct_conductance = 1.0 / s.r_out_zone_k_per_w
    wall_conductance = 1.0 / (s.r_zone_wall_k_per_w + s.r_out_wall_k_per_w)
    return 1.0 / (direct_conductance + wall_conductance)


# Backward-compatible name for notebooks or scripts created before the model
# was correctly identified as three thermal resistances and two capacitances.
# P2: new code must import ThermalPlant3R2C; this alias emits a warning.
def __getattr__(name: str):  # PEP 562 lazy alias with warning
    if name == "ThermalPlant2R2C":
        import warnings as _warnings

        _warnings.warn(
            "ThermalPlant2R2C is a legacy alias; use ThermalPlant3R2C.",
            DeprecationWarning,
            stacklevel=2,
        )
        return ThermalPlant3R2C
    raise AttributeError(name)
