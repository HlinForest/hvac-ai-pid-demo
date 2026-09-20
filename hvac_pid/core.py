"""Time is measured in minutes; cooling commands are fractions in [0, 1]."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Gains:
    kp: float
    ki: float

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in (self.kp, self.ki)):
            raise ValueError("PI gains must be finite and non-negative")


@dataclass(frozen=True)
class Scenario:
    gain: float = 8.0
    tau: float = 20.0
    delay: float = 2.0
    ambient: float = 30.0
    initial: float = 28.0
    setpoint: float = 24.0
    dt: float = 0.1
    duration: float = 120.0
    disturbance_at: float = 60.0
    disturbance: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in self.__dict__.values()):
            raise ValueError("Scenario values must be finite")
        if min(self.gain, self.tau, self.dt, self.duration) <= 0 or self.delay < 0:
            raise ValueError("gain, tau, dt, duration must be positive; delay >= 0")
        if not math.isclose(self.duration / self.dt, round(self.duration / self.dt)):
            raise ValueError("duration must be a multiple of dt")
        if not math.isclose(self.delay / self.dt, round(self.delay / self.dt)):
            raise ValueError("delay must be a multiple of dt")

    @property
    def steps(self):
        return round(self.duration / self.dt)


class Plant:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.temperature = scenario.initial
        self.queue = deque([0.0] * round(scenario.delay / scenario.dt))
        self.decay = math.exp(-scenario.dt / scenario.tau)

    def step(self, command: float, disturbance: float) -> float:
        delayed = command
        if self.queue:
            delayed = self.queue.popleft()
            self.queue.append(command)
        equilibrium = self.scenario.ambient + disturbance - self.scenario.gain * delayed
        self.temperature = equilibrium + (self.temperature - equilibrium) * self.decay
        return self.temperature


# region pi
class PI:
    def __init__(self, gains: Gains):
        self.gains = gains
        self.integral = 0.0  # Output contribution, not an accumulated error.

    def update(self, error: float, dt: float) -> float:
        candidate = self.integral + self.gains.ki * error * dt
        raw = self.gains.kp * error + candidate
        if 0 <= raw <= 1 or (raw > 1 and error < 0) or (raw < 0 and error > 0):
            self.integral = candidate
        return min(1.0, max(0.0, self.gains.kp * error + self.integral))
# endregion pi


@dataclass
class Trace:
    # Each row describes [time, time + dt); temperature is the interval START.
    time: np.ndarray
    temperature: np.ndarray
    next_temperature: np.ndarray
    setpoint: np.ndarray
    disturbance: np.ndarray
    command: np.ndarray
    kp: np.ndarray
    ki: np.ndarray
    dt: float


@dataclass
class Evaluation:
    iae: float
    undershoot: float
    movement: float
    trace: Trace

    def metrics(self):
        return {"iae": self.iae, "undershoot": self.undershoot, "movement": self.movement}


def score(trace: Trace) -> Evaluation:
    error = trace.temperature - trace.setpoint
    return Evaluation(
        iae=float(np.abs(error).sum() * trace.dt),
        undershoot=float(max(0.0, -error.min(), trace.setpoint[-1] - trace.next_temperature[-1])),
        movement=float(np.abs(np.diff(trace.command, prepend=0.0)).sum()),
        trace=trace,
    )


class GainPolicy(Protocol):
    def choose(self, observation: np.ndarray) -> int: ...


ACTION_SCALES = np.asarray([(p, i) for p in (0.5, 1.0, 2.0) for i in (0.5, 1.0, 2.0)])


class TuningEnv:
    """One step chooses gains and runs two minutes of the SAME simulator.

    Observations are partial: error, error slope, previous command, and gains
    relative to the commissioning SIMC anchor. Hidden delay state is not exposed.
    """
    def __init__(self, scenario: Scenario, anchor: Gains, interval: float = 2.0):
        ratio = interval / scenario.dt
        if ratio < 1 or not math.isclose(ratio, round(ratio)):
            raise ValueError("decision interval must be a positive multiple of dt")
        self.scenario, self.anchor = scenario, anchor
        self.interval_steps = round(ratio)
        self.reset()

    def reset(self) -> np.ndarray:
        self.plant = Plant(self.scenario)
        self.pi = PI(self.anchor)
        self.index = 0
        self.previous_error = self.scenario.initial - self.scenario.setpoint
        self.slope = 0.0
        self.command = 0.0
        self.rows: list[tuple] = []
        return self.observation()

    def observation(self):
        return np.asarray([
            self.plant.temperature - self.scenario.setpoint, self.slope, self.command,
            self.pi.gains.kp / self.anchor.kp if self.anchor.kp else 1.0,
            self.pi.gains.ki / self.anchor.ki if self.anchor.ki else 1.0,
        ], dtype=float)

    def advance(self, gains: Gains, count: int) -> float:
        s = self.scenario
        self.pi.gains = gains
        iae = 0.0
        for _ in range(min(count, s.steps - self.index)):
            minute = self.index * s.dt
            temperature = self.plant.temperature
            error = temperature - s.setpoint
            disturbance = s.disturbance if minute >= s.disturbance_at else 0.0
            self.command = self.pi.update(error, s.dt)
            next_temperature = self.plant.step(self.command, disturbance)
            self.rows.append((minute, temperature, next_temperature, s.setpoint,
                              disturbance, self.command, gains.kp, gains.ki))
            iae += abs(error) * s.dt
            self.index += 1
        return iae

    # region environment_step
    def step(self, action: int):
        if self.index >= self.scenario.steps:
            raise RuntimeError("Episode finished; call reset before stepping again")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 9:
            raise ValueError("action must be an integer from 0 to 8")
        scales = ACTION_SCALES[action]
        gains = Gains(self.anchor.kp * scales[0], self.anchor.ki * scales[1])
        start = self.index
        reward = -self.advance(gains, self.interval_steps)
        error = self.plant.temperature - self.scenario.setpoint
        elapsed = (self.index - start) * self.scenario.dt
        self.slope = (error - self.previous_error) / elapsed
        self.previous_error = error
        done = self.index == self.scenario.steps
        return self.observation(), reward, done
    # endregion environment_step

    def trace(self):
        columns = np.asarray(self.rows, dtype=float).T
        return Trace(*columns, dt=self.scenario.dt)


def simulate(scenario: Scenario, gains: Gains, policy: GainPolicy | None = None) -> Trace:
    env = TuningEnv(scenario, gains)
    if policy is None:
        env.advance(gains, scenario.steps)
    else:
        observation = env.observation()
        done = False
        while not done:
            observation, _, done = env.step(policy.choose(observation))
    return env.trace()


def evaluate(scenario: Scenario, gains: Gains, policy: GainPolicy | None = None) -> Evaluation:
    return score(simulate(scenario, gains, policy))
