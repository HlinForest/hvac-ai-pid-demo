from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class Scenario:
    """Parameters for a small 3R2C single-zone cooling simulation.

    Thermal resistances use K/W, capacitances J/K, power W, and time minutes.
    The context fields intentionally match quantities that can be estimated from
    BMS history or a commissioning/identification experiment.
    """

    dt_minutes: float = 1.0
    duration_hours: float = 5.0
    setpoint_c: float = 24.0
    initial_zone_c: float = 28.0
    outdoor_c: float = 34.0
    outdoor_amplitude_c: float = 0.0
    outdoor_peak_hour: float = 7.0
    internal_load_w: float = 900.0
    occupied_load_add_w: float = 0.0
    occupied_start_hour: float = 1.0
    occupied_end_hour: float = 4.0
    door_open_hour: float | None = None
    door_open_duration_minutes: float = 0.0
    door_open_load_w: float = 0.0
    setpoint_change_hour: float | None = None
    setpoint_after_c: float | None = None
    c_zone_j_per_k: float = 1.8e6
    c_wall_j_per_k: float = 12.0e6
    r_out_zone_k_per_w: float = 0.012
    r_zone_wall_k_per_w: float = 0.006
    r_out_wall_k_per_w: float = 0.020
    cooling_capacity_w: float = 7000.0
    actuator_delay_minutes: float = 5.0
    actuator_tau_minutes: float = 4.0
    sensor_noise_std_c: float = 0.0

    FEATURE_NAMES: ClassVar[tuple[str, ...]] = (
        "outdoor_c",
        "setpoint_c",
        "absolute_error_c",
        "internal_load_w",
        "c_zone_j_per_k",
        "c_wall_j_per_k",
        "r_out_zone_k_per_w",
        "r_zone_wall_k_per_w",
        "r_out_wall_k_per_w",
        "cooling_capacity_w",
        "actuator_delay_minutes",
        "actuator_tau_minutes",
    )

    @property
    def steps(self) -> int:
        return int(round(self.duration_hours * 60.0 / self.dt_minutes)) + 1

    def outdoor_at(self, minute: float) -> float:
        if self.outdoor_amplitude_c == 0.0:
            return self.outdoor_c
        phase = 2.0 * np.pi * (minute / 60.0 - self.outdoor_peak_hour) / 24.0
        return float(self.outdoor_c + self.outdoor_amplitude_c * np.cos(phase))

    def load_at(self, minute: float) -> float:
        hour = minute / 60.0
        occupied = self.occupied_start_hour <= hour < self.occupied_end_hour
        door_open = (
            self.door_open_hour is not None
            and self.door_open_hour * 60.0 <= minute < self.door_open_hour * 60.0 + self.door_open_duration_minutes
        )
        return float(
            self.internal_load_w
            + (self.occupied_load_add_w if occupied else 0.0)
            + (self.door_open_load_w if door_open else 0.0)
        )

    def setpoint_at(self, minute: float) -> float:
        if (
            self.setpoint_change_hour is not None
            and self.setpoint_after_c is not None
            and minute >= 60.0 * self.setpoint_change_hour
        ):
            return float(self.setpoint_after_c)
        return float(self.setpoint_c)

    def context_vector(
        self,
        *,
        zone_c: float | None = None,
        outdoor_c: float | None = None,
        setpoint_c: float | None = None,
        internal_load_w: float | None = None,
    ) -> np.ndarray:
        zone = self.initial_zone_c if zone_c is None else zone_c
        outdoor = self.outdoor_c if outdoor_c is None else outdoor_c
        setpoint = self.setpoint_c if setpoint_c is None else setpoint_c
        load = self.internal_load_w if internal_load_w is None else internal_load_w
        return np.asarray(
            [
                outdoor,
                setpoint,
                abs(zone - setpoint),
                load,
                self.c_zone_j_per_k,
                self.c_wall_j_per_k,
                self.r_out_zone_k_per_w,
                self.r_zone_wall_k_per_w,
                self.r_out_wall_k_per_w,
                self.cooling_capacity_w,
                self.actuator_delay_minutes,
                self.actuator_tau_minutes,
            ],
            dtype=float,
        )

    def constant_copy(self, *, duration_hours: float | None = None) -> "Scenario":
        return replace(
            self,
            duration_hours=self.duration_hours if duration_hours is None else duration_hours,
            outdoor_amplitude_c=0.0,
            occupied_load_add_w=0.0,
            door_open_hour=None,
            door_open_duration_minutes=0.0,
            door_open_load_w=0.0,
            setpoint_change_hour=None,
            setpoint_after_c=None,
            sensor_noise_std_c=0.0,
        )


def sample_scenarios(count: int, seed: int, duration_hours: float = 5.0) -> list[Scenario]:
    """Sample commissioning contexts used for offline label generation."""

    rng = np.random.default_rng(seed)
    scenarios: list[Scenario] = []
    for _ in range(count):
        setpoint = rng.uniform(22.5, 25.5)
        scenarios.append(
            Scenario(
                duration_hours=duration_hours,
                setpoint_c=float(setpoint),
                initial_zone_c=float(setpoint + rng.uniform(0.0, 5.5)),
                outdoor_c=float(rng.uniform(28.0, 39.0)),
                internal_load_w=float(rng.uniform(300.0, 1900.0)),
                c_zone_j_per_k=float(rng.uniform(1.1e6, 3.2e6)),
                c_wall_j_per_k=float(rng.uniform(7.0e6, 22.0e6)),
                r_out_zone_k_per_w=float(rng.uniform(0.008, 0.020)),
                r_zone_wall_k_per_w=float(rng.uniform(0.004, 0.011)),
                r_out_wall_k_per_w=float(rng.uniform(0.012, 0.032)),
                cooling_capacity_w=float(rng.uniform(4500.0, 10500.0)),
                actuator_delay_minutes=float(rng.uniform(2.0, 12.0)),
                actuator_tau_minutes=float(rng.uniform(2.0, 10.0)),
            )
        )
    return scenarios


def nominal_scenario() -> Scenario:
    return Scenario()


def dynamic_demo_scenario() -> Scenario:
    """A held-out profile with weather, occupancy, and a setpoint change."""

    return Scenario(
        duration_hours=12.0,
        setpoint_c=24.5,
        setpoint_change_hour=2.0,
        setpoint_after_c=23.8,
        initial_zone_c=28.5,
        outdoor_c=34.0,
        outdoor_amplitude_c=4.0,
        outdoor_peak_hour=7.0,
        internal_load_w=550.0,
        occupied_load_add_w=1100.0,
        occupied_start_hour=1.5,
        occupied_end_hour=9.5,
        door_open_hour=5.0,
        door_open_duration_minutes=12.0,
        door_open_load_w=2600.0,
        c_zone_j_per_k=2.45e6,
        c_wall_j_per_k=16.0e6,
        r_out_zone_k_per_w=0.014,
        r_zone_wall_k_per_w=0.008,
        r_out_wall_k_per_w=0.024,
        cooling_capacity_w=7600.0,
        actuator_delay_minutes=8.0,
        actuator_tau_minutes=6.0,
        sensor_noise_std_c=0.025,
    )


def typical_case_scenarios() -> dict[str, Scenario]:
    """Three separately reported engineering cases, not one mixed profile."""
    return {
        "初次快速降温": Scenario(
            duration_hours=4.0,
            setpoint_c=24.0,
            initial_zone_c=31.5,
            outdoor_c=35.0,
            outdoor_amplitude_c=2.0,
            internal_load_w=650.0,
            cooling_capacity_w=7600.0,
            actuator_delay_minutes=6.0,
            actuator_tau_minutes=4.0,
        ),
        "设定温度突变": Scenario(
            duration_hours=5.0,
            setpoint_c=25.0,
            setpoint_change_hour=1.0,
            setpoint_after_c=23.0,
            initial_zone_c=25.5,
            outdoor_c=34.0,
            outdoor_amplitude_c=3.0,
            internal_load_w=800.0,
            cooling_capacity_w=7600.0,
            actuator_delay_minutes=7.0,
            actuator_tau_minutes=5.0,
        ),
        "持续外界热扰动": Scenario(
            duration_hours=6.0,
            setpoint_c=24.0,
            initial_zone_c=25.0,
            outdoor_c=37.0,
            outdoor_amplitude_c=4.5,
            internal_load_w=600.0,
            occupied_load_add_w=1800.0,
            occupied_start_hour=1.0,
            occupied_end_hour=5.0,
            door_open_hour=3.0,
            door_open_duration_minutes=12.0,
            door_open_load_w=2600.0,
            cooling_capacity_w=8200.0,
            actuator_delay_minutes=8.0,
            actuator_tau_minutes=6.0,
        ),
    }
