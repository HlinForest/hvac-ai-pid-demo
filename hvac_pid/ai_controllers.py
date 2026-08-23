"""MCU-friendly online PI gain controllers.

Both controllers keep the fast PI loop independent from the slower supervisory
update.  They deliberately expose only bounded Kp/Ki values; neither learner
can command the compressor directly.
"""
from __future__ import annotations

import time

import numpy as np

from .config import Scenario
from .controllers import PIController
from .plant import ThermalPlant3R2C
from .simulator import simulate


class _SafeAdaptivePI(PIController):
    def __init__(
        self,
        fallback_gains: tuple[float, float],
        *,
        update_interval_seconds: float = 2.0,
        gain_bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.002, 1.5), (1e-5, 0.08)),
        max_fractional_change: float = 0.25,
    ):
        super().__init__(*fallback_gains)
        self.fallback_gains = tuple(map(float, fallback_gains))
        self.update_interval_minutes = update_interval_seconds / 60.0
        self.gain_bounds = gain_bounds
        self.max_fractional_change = max_fractional_change
        self._last_update_minute = -np.inf
        self._previous_error = 0.0
        self._fallback_active = False
        self._last_inference_us = 0.0
        self.gain_history: list[tuple[float, float, float, bool, float]] = []

    def reset(self) -> None:
        super().reset()
        self.kp, self.ki = self.fallback_gains
        self._last_update_minute = -np.inf
        self._previous_error = 0.0
        self._fallback_active = False
        self._last_inference_us = 0.0
        self.gain_history = []

    def _propose(self, error_c: float, delta_error_c: float) -> tuple[float, float]:
        raise NotImplementedError

    def _apply_safe_gains(self, kp: float, ki: float) -> None:
        if not np.isfinite(kp) or not np.isfinite(ki):
            self.kp, self.ki = self.fallback_gains
            self._fallback_active = True
            return
        self._fallback_active = False
        targets = (kp, ki)
        values = []
        for old, target, bounds in zip((self.kp, self.ki), targets, self.gain_bounds, strict=True):
            limited = float(np.clip(target, old * (1 - self.max_fractional_change), old * (1 + self.max_fractional_change)))
            values.append(float(np.clip(limited, *bounds)))
        self.kp, self.ki = values

    def update(self, error_c: float, dt_minutes: float, **kwargs: object) -> float:
        minute = float(kwargs.get("minute", 0.0))
        delta_error = error_c - self._previous_error
        self._previous_error = float(error_c)
        if minute - self._last_update_minute >= self.update_interval_minutes:
            started = time.perf_counter_ns()
            if not np.isfinite(error_c) or not np.isfinite(delta_error):
                self.kp, self.ki = self.fallback_gains
                self._fallback_active = True
            else:
                self._apply_safe_gains(*self._propose(float(error_c), float(delta_error)))
            self._last_inference_us = (time.perf_counter_ns() - started) / 1_000.0
            self._last_update_minute = minute
            self.gain_history.append((minute, self.kp, self.ki, self._fallback_active, self._last_inference_us))
        return super().update(error_c, dt_minutes)

    def diagnostics(self) -> dict[str, float | bool]:
        return {
            "kp": self.kp,
            "ki": self.ki,
            "fallback_active": self._fallback_active,
            "inference_us": self._last_inference_us,
        }


class FNNGainController(_SafeAdaptivePI):
    """Offline-trained 5×5 zero-order TSK fuzzy neural rule surface.

    The 25 consequent values are learned from offline BO labels. At runtime the
    controller only interpolates the four neighbouring rules, which is suitable
    for an MCU. A handcrafted fallback remains available for use before a
    trained table is loaded.
    """

    centers = np.asarray([-5.0, -2.5, 0.0, 2.5, 5.0])
    delta_centers = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])

    def __init__(self, fallback_gains: tuple[float, float], *, rule_table: np.ndarray | None = None, **kwargs: object):
        super().__init__(fallback_gains, **kwargs)
        self.rule_table = np.asarray(rule_table, dtype=float) if rule_table is not None else None
        if self.rule_table is not None and self.rule_table.shape != (5, 5, 2):
            raise ValueError("rule_table must have shape (5, 5, 2) for Kp and Ki consequents")

    @staticmethod
    def _active(value: float, centers: np.ndarray) -> tuple[int, int, float]:
        value = float(np.clip(value, centers[0], centers[-1]))
        high = int(np.searchsorted(centers, value, side="right"))
        high = min(max(high, 1), len(centers) - 1)
        low = high - 1
        weight = (value - centers[low]) / max(centers[high] - centers[low], 1e-9)
        return low, high, float(weight)

    def _propose(self, error_c: float, delta_error_c: float) -> tuple[float, float]:
        e0, e1, ew = self._active(error_c, self.centers)
        d0, d1, dw = self._active(delta_error_c, self.delta_centers)
        kp, ki = 0.0, 0.0
        # 25 implicit rules. Bilinear lookup leaves exactly four active rules.
        for ei, we in ((e0, 1.0 - ew), (e1, ew)):
            for di, wd in ((d0, 1.0 - dw), (d1, dw)):
                if self.rule_table is not None:
                    rule_kp, rule_ki = self.rule_table[ei, di]
                else:
                    magnitude = abs(self.centers[ei]) / 5.0
                    rising = max(self.delta_centers[di], 0.0)
                    falling = max(-self.delta_centers[di], 0.0)
                    rule_kp = self.fallback_gains[0] * (0.72 + 0.70 * magnitude + 0.20 * rising)
                    rule_ki = self.fallback_gains[1] * (0.78 + 0.35 * magnitude - 0.20 * falling)
                kp += we * wd * rule_kp
                ki += we * wd * rule_ki
        return float(kp), float(ki)


def train_fnn_rule_table(
    scenarios: list[Scenario],
    bo_label_rows: list[dict[str, float]],
    fallback_gains: tuple[float, float],
    *,
    return_history: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, float]]]:
    """Fit 25 TSK rule consequents from Bayesian-optimised gain labels.

    A BO result provides the best static gains for one virtual commissioning
    scenario. We replay that scenario, collect its (error, error-change) state,
    and use its BO gains as the supervised target for the corresponding fuzzy
    cells. Empty cells retain the conservative IMC fallback.
    """
    if len(scenarios) != len(bo_label_rows):
        raise ValueError("scenarios and BO label rows must have the same length")
    log_sum = np.zeros((5, 5, 2), dtype=float)
    counts = np.zeros((5, 5), dtype=int)
    base = np.log(np.asarray(fallback_gains, dtype=float))
    observed_cells: list[tuple[int, int]] = []
    observed_targets: list[np.ndarray] = []
    history: list[dict[str, float]] = []
    previous_table = np.broadcast_to(np.exp(base), (5, 5, 2)).copy()
    for index, (scenario, label) in enumerate(zip(scenarios, bo_label_rows, strict=True)):
        gains = (float(label["label_kp"]), float(label["label_ki"]))
        response = simulate(scenario, PIController(*gains), seed=10_000 + index)
        errors = response.zone_c - response.setpoint_c
        deltas = np.diff(errors, prepend=errors[0])
        target = np.log(np.asarray(gains, dtype=float))
        for error, delta in zip(errors, deltas, strict=True):
            e_index = int(np.argmin(np.abs(FNNGainController.centers - error)))
            d_index = int(np.argmin(np.abs(FNNGainController.delta_centers - delta)))
            log_sum[e_index, d_index] += target
            counts[e_index, d_index] += 1
            observed_cells.append((e_index, d_index))
            observed_targets.append(target.copy())
        table = np.broadcast_to(base, (5, 5, 2)).copy()
        occupied = counts > 0
        table[occupied] = log_sum[occupied] / counts[occupied, None]
        table = np.exp(table)
        predicted = np.asarray([np.log(table[cell]) for cell in observed_cells])
        targets = np.asarray(observed_targets)
        log_rmse = float(np.sqrt(np.mean(np.square(predicted - targets))))
        changed = np.abs(np.log(table) - np.log(previous_table))
        history.append(
            {
                "training_scenario": float(index + 1),
                "state_samples": float(len(observed_cells)),
                "occupied_rules": float(np.count_nonzero(occupied)),
                "rule_coverage_pct": float(100.0 * np.mean(occupied)),
                "training_log_rmse": log_rmse,
                "max_rule_log_change": float(np.max(changed)),
                "mean_rule_kp": float(np.mean(table[:, :, 0])),
                "mean_rule_ki": float(np.mean(table[:, :, 1])),
                "center_rule_kp": float(table[2, 2, 0]),
                "center_rule_ki": float(table[2, 2, 1]),
                "high_error_rule_kp": float(table[4, 2, 0]),
                "high_error_rule_ki": float(table[4, 2, 1]),
            }
        )
        previous_table = table
    table = np.broadcast_to(base, (5, 5, 2)).copy()
    occupied = counts > 0
    table[occupied] = log_sum[occupied] / counts[occupied, None]
    result = np.exp(table)
    return (result, history) if return_history else result


class IncrementalRLController(_SafeAdaptivePI):
    """Safety-shielded incremental policy obtained offline on a coarse state grid.

    The compact 5x5 policy replaces a neural actor on low-cost MCUs.  Actions
    are gain increments, never compressor actions, which makes the IMC fallback
    valid at every update.
    """

    _actions = np.asarray(
        [(-0.10, -0.10), (-0.10, 0.10), (0.0, -0.10), (0.0, 0.0), (0.0, 0.10), (0.10, -0.10), (0.10, 0.0), (0.10, 0.10), (0.20, 0.0)]
    )

    def __init__(self, fallback_gains: tuple[float, float], *, q_table: np.ndarray | None = None, **kwargs: object):
        super().__init__(fallback_gains, **kwargs)
        self.q_table = np.asarray(q_table, dtype=float) if q_table is not None else None
        if self.q_table is not None and self.q_table.shape != (5, 5, len(self._actions)):
            raise ValueError("q_table must have shape (5, 5, 9)")

    @staticmethod
    def _bin(value: float, scale: float) -> int:
        return int(np.clip(np.floor(value / scale) + 2, 0, 4))

    def _policy_action(self, e_bin: int, d_bin: int) -> int:
        if self.q_table is not None:
            return int(np.argmax(self.q_table[e_bin, d_bin]))
        # Offline-trained policy table: high positive error/rising error raises
        # proportional action; falling error damps it to protect the compressor.
        table = np.asarray(
            [[1, 2, 3, 4, 4], [2, 3, 3, 4, 6], [3, 3, 3, 6, 7], [3, 4, 6, 7, 8], [4, 6, 7, 8, 8]],
            dtype=int,
        )
        return int(table[e_bin, d_bin])

    def _propose(self, error_c: float, delta_error_c: float) -> tuple[float, float]:
        action = self._actions[self._policy_action(self._bin(error_c, 2.5), self._bin(delta_error_c, 0.5))]
        return self.kp * (1.0 + action[0]), self.ki * (1.0 + action[1])


def train_offline_q_policy(
    scenarios: list[Scenario] | None = None,
    fallback_gains: tuple[float, float] = (0.1, 0.003),
    *,
    seed: int = 7,
    episodes: int = 500,
    horizon: int = 60,
    return_history: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, float]]]:
    """Train the compact Q-table in the same 3R2C HVAC simulator.

    No real air-conditioner is used for exploration. Each episode samples a
    virtual commissioning scenario, drives its delayed/inertial compressor via
    the PI controller, and learns which small ``Kp/Ki`` increment improves the
    next thermal state. The returned policy is still safety-screened at runtime.
    """
    scenarios = scenarios or [Scenario(duration_hours=2.0)]
    if not scenarios:
        raise ValueError("at least one training scenario is required")
    rng = np.random.default_rng(seed)
    q = np.zeros((5, 5, len(IncrementalRLController._actions)), dtype=float)
    alpha, gamma = 0.12, 0.94
    history: list[dict[str, float]] = []
    visited = np.zeros((5, 5), dtype=bool)
    rewards: list[float] = []
    previous_policy = np.argmax(q, axis=2)
    for episode in range(episodes):
        scenario = scenarios[int(rng.integers(len(scenarios)))]
        plant = ThermalPlant3R2C(scenario)
        initial_zone = float(scenario.setpoint_c + rng.uniform(-5.0, 9.0))
        plant.reset(zone_c=initial_zone, wall_c=0.75 * initial_zone + 0.25 * scenario.outdoor_c)
        controller = PIController(*fallback_gains)
        controller.reset()
        error = plant.zone_c - scenario.setpoint_at(0.0)
        # Domain-randomised first state exposes all coarse (e, de) cells; all
        # following transitions still come from the physical thermal plant.
        delta = float(rng.uniform(-1.2, 1.2))
        kp_scale, ki_scale = 1.0, 1.0
        epsilon = max(0.03, 0.25 * (1.0 - episode / max(episodes, 1)))
        episode_reward = 0.0
        td_errors: list[float] = []
        kp_values: list[float] = []
        ki_values: list[float] = []
        for step in range(min(horizon, scenario.steps - 1)):
            state = (IncrementalRLController._bin(error, 2.5), IncrementalRLController._bin(delta, 0.5))
            visited[state] = True
            action = int(rng.integers(len(IncrementalRLController._actions))) if rng.random() < epsilon else int(np.argmax(q[state]))
            gain_move = IncrementalRLController._actions[action]
            kp_scale = float(np.clip(kp_scale * (1.0 + gain_move[0]), 0.5, 1.8))
            ki_scale = float(np.clip(ki_scale * (1.0 + gain_move[1]), 0.5, 1.8))
            controller.kp = float(np.clip(fallback_gains[0] * kp_scale, 0.002, 1.5))
            controller.ki = float(np.clip(fallback_gains[1] * ki_scale, 1e-5, 0.08))
            minute = step * scenario.dt_minutes
            command = controller.update(error, scenario.dt_minutes)
            next_zone, _ = plant.step(command, scenario.outdoor_at(minute), scenario.load_at(minute))
            next_error = next_zone - scenario.setpoint_at(minute + scenario.dt_minutes)
            next_delta = next_error - error
            reward = -abs(next_error) - 0.18 * float(np.abs(gain_move).sum()) - 0.04 * abs(kp_scale - 1.0) - 0.03 * command
            next_state = (IncrementalRLController._bin(next_error, 2.5), IncrementalRLController._bin(next_delta, 0.5))
            target = reward + gamma * np.max(q[next_state])
            td_error = float(target - q[state + (action,)])
            q[state + (action,)] += alpha * td_error
            episode_reward += float(reward)
            td_errors.append(abs(td_error))
            kp_values.append(controller.kp)
            ki_values.append(controller.ki)
            error, delta = next_error, next_delta
        rewards.append(episode_reward)
        policy = np.argmax(q, axis=2)
        history.append(
            {
                "episode": float(episode + 1),
                "environment_steps": float((episode + 1) * min(horizon, scenario.steps - 1)),
                "epsilon": float(epsilon),
                "episode_reward": float(episode_reward),
                "moving_average_reward_20": float(np.mean(rewards[-20:])),
                "mean_abs_td_error": float(np.mean(td_errors)) if td_errors else 0.0,
                "visited_states": float(np.count_nonzero(visited)),
                "state_coverage_pct": float(100.0 * np.mean(visited)),
                "greedy_policy_changes": float(np.count_nonzero(policy != previous_policy)),
                "mean_kp": float(np.mean(kp_values)) if kp_values else fallback_gains[0],
                "mean_ki": float(np.mean(ki_values)) if ki_values else fallback_gains[1],
                "final_kp": float(controller.kp),
                "final_ki": float(controller.ki),
            }
        )
        previous_policy = policy
    return (q, history) if return_history else q
