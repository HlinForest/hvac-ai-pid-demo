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
    sensor_noise_std_c: float = 0.04
    sensor_filter_tau_minutes: float = 2.0
    command_slew_rate_per_minute: float = 0.05
    command_quantization: float = 0.01
    minimum_running_command: float = 0.25
    minimum_on_minutes: float = 5.0
    minimum_off_minutes: float = 3.0
    # v4 physics: internal integration substeps per outer dt step.
    # Outer interface stays 1 min (controller/metrics unchanged); each plant
    # step is subdivided into N=6 explicit-Euler substeps (10 s each) holding
    # inputs constant. Convergence study: 初次快速降温 max error
    # 0.150350 (N=1) -> 0.074927 (N=2) -> 0.037400 (N=4) -> 0.024919 (N=6),
    # confirming first-order discretization error, fixed by substepping
    # rather than by loosening the 0.15 C threshold.
    integration_substeps: int = 6

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
                sensor_noise_std_c=float(rng.uniform(0.025, 0.075)),
            )
        )
    return scenarios


def sample_adaptive_scenarios(count: int, seed: int, duration_hours: float = 5.0) -> list[Scenario]:
    """Create a stratified operating-domain data set for adaptive controllers.

    ``sample_scenarios`` is intentionally a simple hot-start commissioning set.
    That is adequate for fixed-gain tuning, but it does not exercise the signs
    of error and error-rate needed by a two-dimensional FNN/RL state grid.  This
    curriculum adds physically reachable cold starts, setpoint changes, heat
    pulses and weather/load ramps.  The list is shuffled deterministically so
    the final validation slice contains a mixture of scenario families.
    """

    if count <= 0:
        return []
    rng = np.random.default_rng(seed)
    base = sample_scenarios(count, seed=seed + 17, duration_hours=duration_hours)
    scenarios: list[Scenario] = []
    for index, scenario in enumerate(base):
        family = index % 7
        common = {
            "outdoor_amplitude_c": float(rng.uniform(0.0, 3.5)),
            "outdoor_peak_hour": float(rng.uniform(2.0, 8.0)),
        }
        if family == 0:
            # Ordinary hot start: positive error that falls toward zero.
            updated = replace(scenario, **common)
        elif family == 1:
            # Cold start: the compressor must remain off while the room warms.
            updated = replace(
                scenario,
                initial_zone_c=float(scenario.setpoint_c - rng.uniform(0.6, 3.5)),
                **common,
            )
        elif family == 2:
            # Lowering the setpoint creates a positive error-rate event.
            change_hour = float(rng.uniform(0.8, max(1.0, duration_hours - 1.2)))
            updated = replace(
                scenario,
                initial_zone_c=float(scenario.setpoint_c + rng.uniform(-0.4, 1.0)),
                setpoint_change_hour=change_hour,
                setpoint_after_c=float(scenario.setpoint_c - rng.uniform(0.8, 2.2)),
                **common,
            )
        elif family == 3:
            # Raising the setpoint creates a negative error-rate event and an
            # intentional compressor unload/stop transition.
            change_hour = float(rng.uniform(0.8, max(1.0, duration_hours - 1.2)))
            updated = replace(
                scenario,
                initial_zone_c=float(scenario.setpoint_c + rng.uniform(1.0, 4.0)),
                setpoint_change_hour=change_hour,
                setpoint_after_c=float(scenario.setpoint_c + rng.uniform(0.8, 2.2)),
                **common,
            )
        elif family == 4:
            # A finite door/occupancy heat pulse drives error upward and then
            # back down after the pulse, exercising both derivative signs.
            pulse_start = float(rng.uniform(0.8, max(1.0, duration_hours - 1.5)))
            pulse_duration = float(rng.uniform(15.0, 50.0))
            updated = replace(
                scenario,
                initial_zone_c=float(scenario.setpoint_c + rng.uniform(-0.5, 1.5)),
                door_open_hour=pulse_start,
                door_open_duration_minutes=pulse_duration,
                door_open_load_w=float(rng.uniform(1800.0, 4200.0)),
                **common,
            )
        elif family == 5:
            # Sustained load transition covers slow positive error-rate states.
            start = float(rng.uniform(0.6, max(0.8, duration_hours / 2.0)))
            end = float(rng.uniform(max(start + 0.6, duration_hours * 0.65), duration_hours - 0.1))
            updated = replace(
                scenario,
                initial_zone_c=float(scenario.setpoint_c + rng.uniform(-1.0, 2.0)),
                occupied_start_hour=start,
                occupied_end_hour=end,
                occupied_load_add_w=float(rng.uniform(900.0, 2600.0)),
                **common,
            )
        else:
            # Rare but valid corner: a room is already well below the old
            # setpoint and an operator lowers the setpoint shortly afterwards.
            # The room remains cold while delta-error becomes strongly positive;
            # this is the only physically meaningful route to the upper-right
            # derivative corner of the negative-error state grid.
            updated = replace(
                scenario,
                initial_zone_c=float(scenario.setpoint_c - rng.uniform(3.5, 5.0)),
                setpoint_change_hour=float(rng.uniform(0.20, 0.45)),
                setpoint_after_c=float(scenario.setpoint_c - rng.uniform(0.8, 1.3)),
                **common,
            )
        scenarios.append(updated)

    order = rng.permutation(len(scenarios))
    return [scenarios[int(index)] for index in order]


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
        sensor_noise_std_c=0.05,
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
            sensor_noise_std_c=0.05,
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
            sensor_noise_std_c=0.05,
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
            sensor_noise_std_c=0.05,
        ),
    }
