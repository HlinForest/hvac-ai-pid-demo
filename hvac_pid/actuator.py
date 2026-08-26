from __future__ import annotations

import numpy as np

from .config import Scenario


class CompressorCommandLimiter:
    """Apply the command constraints of a small variable-speed compressor.

    Zero means stopped. A running command is at least ``u_min`` and is
    quantised. Start/stop transitions obey dwell-time interlocks; slew limiting
    applies between non-zero running points. The unavoidable 0 -> u_min start
    and u_min -> 0 stop transitions are tracked separately from running slew.
    """

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.command = 0.0
        self.is_on = False
        self.minutes_in_state = float("inf")
        self.start_events = 0
        self.stop_events = 0
        self.interlock_events = 0

    def reset(self, initial_command: float = 0.0) -> None:
        initial = float(np.clip(initial_command, 0.0, 1.0))
        self.is_on = initial > 0.0
        self.command = max(initial, self.scenario.minimum_running_command) if self.is_on else 0.0
        self.minutes_in_state = float("inf")
        self.start_events = 0
        self.stop_events = 0
        self.interlock_events = 0

    def _quantize(self, value: float) -> float:
        quantum = self.scenario.command_quantization
        if quantum <= 0.0:
            return float(value)
        return float(np.round(value / quantum) * quantum)

    def update(self, requested: float, dt_minutes: float) -> float:
        s = self.scenario
        requested = float(np.clip(requested, 0.0, 1.0))
        wants_on = requested >= 0.5 * max(s.minimum_running_command, s.command_quantization)

        if self.is_on and not wants_on:
            if self.minutes_in_state < s.minimum_on_minutes:
                self.interlock_events += 1
                wants_on = True
            else:
                self.is_on = False
                self.command = 0.0
                self.minutes_in_state = 0.0
                self.stop_events += 1
        elif not self.is_on and wants_on:
            if self.minutes_in_state < s.minimum_off_minutes:
                self.interlock_events += 1
                wants_on = False
            else:
                self.is_on = True
                self.command = float(s.minimum_running_command)
                self.minutes_in_state = 0.0
                self.start_events += 1

        if self.is_on and wants_on:
            target = max(requested, s.minimum_running_command)
            max_step = max(s.command_slew_rate_per_minute, 0.0) * dt_minutes
            target = float(np.clip(target, self.command - max_step, self.command + max_step))
            self.command = float(np.clip(self._quantize(target), s.minimum_running_command, 1.0))

        self.minutes_in_state += dt_minutes
        return self.command
