"""MCU-friendly online PI gain controllers.

Both controllers keep the fast PI loop independent from the slower supervisory
update.  They deliberately expose only bounded Kp/Ki values; neither learner
can command the compressor directly.
"""
from __future__ import annotations

import math
import time
import copy

import numpy as np

from .config import Scenario
from .controllers import PIController
from .plant import ThermalPlant3R2C
from .safety import KI_BOUNDS, KP_BOUNDS, MAX_FRACTIONAL_GAIN_CHANGE
from .timebase import SIM_SUPERVISORY_PERIOD_S
from .simulator import simulate
from .actuator import CompressorCommandLimiter
from .metrics import calculate_metrics


class _SafeAdaptivePI(PIController):
    def __init__(
        self,
        fallback_gains: tuple[float, float],
        *,
        update_interval_seconds: float = SIM_SUPERVISORY_PERIOD_S,
        gain_bounds: tuple[tuple[float, float], tuple[float, float]] = (KP_BOUNDS, KI_BOUNDS),
        max_fractional_change: float = MAX_FRACTIONAL_GAIN_CHANGE,
    ):
        super().__init__(*fallback_gains)
        self.fallback_gains = tuple(map(float, fallback_gains))
        self.update_interval_minutes = update_interval_seconds / 60.0
        self.gain_bounds = gain_bounds
        self.max_fractional_change = max_fractional_change
        self._last_update_minute = -np.inf
        self._previous_supervisory_error: float | None = None
        self._fallback_active = False
        self._last_inference_us = 0.0
        self._last_proposal_valid = True
        self.gain_history: list[tuple[float, float, float, bool, float]] = []

    def reset(self) -> None:
        super().reset()
        self.kp, self.ki = self.fallback_gains
        self._last_update_minute = -np.inf
        self._previous_supervisory_error = None
        self._fallback_active = False
        self._last_inference_us = 0.0
        self._last_proposal_valid = True
        self.gain_history = []

    def _propose(self, error_c: float, delta_error_c: float, *, applied_command: float = 0.0, **context: float) -> tuple[float, float]:
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
        if minute - self._last_update_minute >= self.update_interval_minutes:
            # Delta-e must describe the same slow supervisory interval used by
            # FNN/RL training.  The old implementation refreshed the reference
            # every fast PI sample, so a nominal five-minute learner was fed a
            # one-minute derivative at deployment.
            elapsed_supervisory_minutes = (
                self.update_interval_minutes
                if not np.isfinite(self._last_update_minute)
                else max(minute - self._last_update_minute, 1e-9)
            )
            delta_error = (
                0.0
                if self._previous_supervisory_error is None
                else (float(error_c) - self._previous_supervisory_error) / elapsed_supervisory_minutes
            )
            started = time.perf_counter_ns()
            if not np.isfinite(error_c) or not np.isfinite(delta_error):
                self.kp, self.ki = self.fallback_gains
                self._fallback_active = True
            else:
                self._last_proposal_valid = True
                proposed = self._propose(
                    float(error_c),
                    float(delta_error),
                    applied_command=float(kwargs.get("applied_command", 0.0)),
                    integral_state=float(self.integral),
                    outdoor_delta_c=float(kwargs.get("outdoor_c", 0.0)) - float(kwargs.get("setpoint_c", 0.0)),
                    load_fraction=float(kwargs.get("internal_load_w", 0.0)) / max(
                        float(getattr(kwargs.get("scenario"), "cooling_capacity_w", 1.0)), 1e-9
                    ),
                )
                if self._last_proposal_valid:
                    self._apply_safe_gains(*proposed)
                else:
                    self.kp, self.ki = self.fallback_gains
                    self._fallback_active = True
            self._last_inference_us = (time.perf_counter_ns() - started) / 1_000.0
            self._last_update_minute = minute
            self._previous_supervisory_error = float(error_c)
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

    # These centres describe reachable cooling-control states.  Error is
    # measured as Tzone-Tsetpoint.  Delta-error is the change over the five
    # minute supervisory interval and is normalised to °C/min so the same table
    # remains meaningful when an MCU uses a different supervisory period.
    centers = np.asarray([-3.0, -0.75, 0.0, 1.5, 5.0])
    delta_centers = np.asarray([-0.30, -0.05, 0.0, 0.05, 0.30])

    def __init__(
        self,
        fallback_gains: tuple[float, float],
        *,
        rule_table: np.ndarray | None = None,
        context_coefficients: np.ndarray | None = None,
        **kwargs: object,
    ):
        super().__init__(fallback_gains, **kwargs)
        self.rule_table = np.asarray(rule_table, dtype=float) if rule_table is not None else None
        if self.rule_table is not None and self.rule_table.shape != (5, 5, 2):
            raise ValueError("rule_table must have shape (5, 5, 2) for Kp and Ki consequents")
        self.context_coefficients = (
            np.asarray(context_coefficients, dtype=float)
            if context_coefficients is not None
            else np.zeros((2, 4), dtype=float)
        )
        if self.context_coefficients.shape != (2, 4):
            raise ValueError("context_coefficients must have shape (2, 4)")

    @staticmethod
    def _active(value: float, centers: np.ndarray) -> tuple[int, int, float]:
        value = float(np.clip(value, centers[0], centers[-1]))
        high = int(np.searchsorted(centers, value, side="right"))
        high = min(max(high, 1), len(centers) - 1)
        low = high - 1
        weight = (value - centers[low]) / max(centers[high] - centers[low], 1e-9)
        return low, high, float(weight)

    def _propose(
        self,
        error_c: float,
        delta_error_c: float,
        *,
        applied_command: float = 0.0,
        integral_state: float = 0.0,
        outdoor_delta_c: float = 0.0,
        load_fraction: float = 0.0,
        **_: float,
    ) -> tuple[float, float]:
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
        features = np.clip(
            np.asarray(
                [applied_command - 0.5, integral_state / 100.0, outdoor_delta_c / 20.0, load_fraction - 0.25],
                dtype=float,
            ),
            -2.0,
            2.0,
        )
        residual = np.clip(self.context_coefficients @ features, -0.12, 0.12)
        return float(kp * np.exp(residual[0])), float(ki * np.exp(residual[1]))


def _local_rollout_score(
    scenario: Scenario,
    plant: ThermalPlant3R2C,
    limiter: CompressorCommandLimiter,
    integral: float,
    filtered_measurement: float,
    start_minute: float,
    gains: tuple[float, float],
    *,
    horizon_minutes: float = 30.0,
) -> float:
    """Evaluate gains from one identical closed-loop state snapshot."""
    local_plant = copy.deepcopy(plant)
    local_limiter = copy.deepcopy(limiter)
    local_controller = PIController(*gains)
    local_controller.integral = float(integral)
    measurement = float(filtered_measurement)
    previous_command = float(local_limiter.command)
    steps = max(1, int(round(horizon_minutes / scenario.dt_minutes)))
    score = 0.0
    for step in range(steps):
        minute = start_minute + step * scenario.dt_minutes
        error = measurement - scenario.setpoint_at(minute)
        request = local_controller.update(error, scenario.dt_minutes)
        command = local_limiter.update(request, scenario.dt_minutes)
        local_plant.step(command, scenario.outdoor_at(minute), scenario.load_at(minute))
        # Labels deliberately use the deterministic filtered state. Noise is
        # present in the behaviour trajectory and final evaluation, but must not
        # make two gains see different random futures from the same snapshot.
        alpha = scenario.dt_minutes / max(
            scenario.sensor_filter_tau_minutes + scenario.dt_minutes,
            scenario.dt_minutes,
        )
        measurement += alpha * (local_plant.zone_c - measurement)
        next_error = measurement - scenario.setpoint_at(minute + scenario.dt_minutes)
        comfort = max(abs(next_error) - 0.5, 0.0)
        score += abs(next_error) + 1.5 * comfort + 0.25 * max(-next_error, 0.0)
        score += 0.08 * abs(command - previous_command) + 0.015 * command
        previous_command = command
    score += 2.0 * abs(measurement - scenario.setpoint_at(start_minute + horizon_minutes))
    return float(score)


def _state_local_samples(
    scenario: Scenario,
    fallback_gains: tuple[float, float],
    bo_gains: tuple[float, float],
    *,
    seed: int,
) -> list[dict[str, float]]:
    """Aggregate local labels from IMC, BO and residual-policy trajectories."""
    candidate_pairs = [
        fallback_gains,
        bo_gains,
        (fallback_gains[0] * 0.85, fallback_gains[1] * 1.15),
        (fallback_gains[0] * 0.95, fallback_gains[1] * 1.20),
        (fallback_gains[0] * 1.05, fallback_gains[1] * 1.10),
    ]
    candidate_pairs = [
        (float(np.clip(kp, 0.002, 1.5)), float(np.clip(ki, 1e-5, 0.08)))
        for kp, ki in candidate_pairs
    ]
    behaviour_pairs = [fallback_gains, bo_gains, candidate_pairs[2]]
    samples: list[dict[str, float]] = []
    for behaviour_index, behaviour in enumerate(behaviour_pairs):
        plant = ThermalPlant3R2C(scenario)
        limiter = CompressorCommandLimiter(scenario)
        controller = PIController(*behaviour)
        rng = np.random.default_rng(seed + 1009 * behaviour_index)
        filtered = float(plant.zone_c)
        previous_supervisory_error: float | None = None
        interval_steps = max(1, int(round(10.0 / scenario.dt_minutes)))
        for step in range(scenario.steps):
            minute = step * scenario.dt_minutes
            raw = plant.zone_c + rng.normal(0.0, scenario.sensor_noise_std_c)
            alpha = scenario.dt_minutes / max(
                scenario.sensor_filter_tau_minutes + scenario.dt_minutes,
                scenario.dt_minutes,
            )
            filtered += alpha * (raw - filtered)
            error = filtered - scenario.setpoint_at(minute)
            if step % interval_steps == 0:
                error_rate = 0.0 if previous_supervisory_error is None else (
                    error - previous_supervisory_error
                ) / max(interval_steps * scenario.dt_minutes, 1e-9)
                costs = [
                    _local_rollout_score(
                        scenario, plant, limiter, controller.integral, filtered, minute, gains
                    )
                    for gains in candidate_pairs
                ]
                best = int(np.argmin(costs))
                target_kp, target_ki = candidate_pairs[best]
                samples.append(
                    {
                        "minute": float(minute),
                        "error_c": float(error),
                        "error_rate_c_per_min": float(error_rate),
                        "applied_command": float(limiter.command),
                        "integral_state": float(controller.integral),
                        "outdoor_delta_c": float(scenario.outdoor_at(minute) - scenario.setpoint_at(minute)),
                        "load_fraction": float(scenario.load_at(minute) / max(scenario.cooling_capacity_w, 1e-9)),
                        "label_kp": target_kp,
                        "label_ki": target_ki,
                        "local_rollout_score": float(costs[best]),
                        "behaviour_policy": float(behaviour_index),
                    }
                )
                previous_supervisory_error = float(error)
            request = controller.update(error, scenario.dt_minutes)
            command = limiter.update(request, scenario.dt_minutes)
            plant.step(command, scenario.outdoor_at(minute), scenario.load_at(minute))
    return samples


def train_fnn_rule_table(
    scenarios: list[Scenario],
    bo_label_rows: list[dict[str, float]],
    fallback_gains: tuple[float, float],
    *,
    validation_scenarios: list[Scenario] | None = None,
    return_history: bool = False,
    sample_sink: list[dict[str, float]] | None = None,
    candidate_sink: list[np.ndarray] | None = None,
    context_sink: list[np.ndarray] | None = None,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, float]]]:
    """Fit 25 TSK rule consequents from Bayesian-optimised gain labels.

    A BO result provides the best static gains for one virtual commissioning
    scenario. We replay that scenario, collect its (error, error-change) state,
    and use its BO gains as the supervised target for the corresponding fuzzy
    cells. Empty cells retain the conservative IMC fallback.
    """
    if len(scenarios) != len(bo_label_rows):
        raise ValueError("scenarios and BO label rows must have the same length")
    if not scenarios:
        raise ValueError("at least one training scenario is required")
    if validation_scenarios is None:
        validation_count = max(1, int(math.ceil(0.20 * len(scenarios)))) if len(scenarios) > 1 else 0
        fit_count = len(scenarios) - validation_count
        fit_pairs = list(zip(scenarios[:fit_count], bo_label_rows[:fit_count], strict=True))
        validation_scenarios = scenarios[fit_count:] or scenarios
    else:
        if not validation_scenarios:
            raise ValueError("validation_scenarios must not be empty")
        fit_pairs = list(zip(scenarios, bo_label_rows, strict=True))
    log_sum = np.zeros((5, 5, 2), dtype=float)
    counts = np.zeros((5, 5), dtype=float)
    prior_weight = 2.0
    base = np.log(np.asarray(fallback_gains, dtype=float))
    observed_cells: list[tuple[int, int]] = []
    observed_targets: list[np.ndarray] = []
    observed_contexts: list[np.ndarray] = []
    observed_states: list[tuple[float, float]] = []
    history: list[dict[str, float]] = []
    previous_table = np.broadcast_to(np.exp(base), (5, 5, 2)).copy()
    for index, (scenario, label) in enumerate(fit_pairs):
        gains = (float(label["label_kp"]), float(label["label_ki"]))
        local_samples = _state_local_samples(scenario, fallback_gains, gains, seed=10_000 + index)
        for local in local_samples:
            error = local["error_c"]
            delta = local["error_rate_c_per_min"]
            target = np.log(np.asarray([local["label_kp"], local["label_ki"]], dtype=float))
            e0, e1, ew = FNNGainController._active(float(error), FNNGainController.centers)
            d0, d1, dw = FNNGainController._active(float(delta), FNNGainController.delta_centers)
            for e_index, e_weight in ((e0, 1.0 - ew), (e1, ew)):
                for d_index, d_weight in ((d0, 1.0 - dw), (d1, dw)):
                    weight = e_weight * d_weight
                    if weight <= 0.0:
                        continue
                    log_sum[e_index, d_index] += weight * target
                    counts[e_index, d_index] += weight
            observed_cells.append((int(np.argmin(np.abs(FNNGainController.centers - error))), int(np.argmin(np.abs(FNNGainController.delta_centers - delta)))))
            observed_targets.append(target.copy())
            observed_states.append((float(error), float(delta)))
            observed_contexts.append(
                np.clip(
                    np.asarray(
                        [
                            local["applied_command"] - 0.5,
                            local["integral_state"] / 100.0,
                            local["outdoor_delta_c"] / 20.0,
                            local["load_fraction"] - 0.25,
                        ]
                    ),
                    -2.0,
                    2.0,
                )
            )
            if sample_sink is not None:
                sample_sink.append(
                    {
                        "fit_scenario": float(index + 1),
                        "minute": local["minute"],
                        "error_c": float(error),
                        "error_rate_c_per_min": float(delta),
                        "applied_command": local["applied_command"],
                        "integral_state": local["integral_state"],
                        "outdoor_delta_c": local["outdoor_delta_c"],
                        "load_fraction": local["load_fraction"],
                        "nearest_error_rule": float(observed_cells[-1][0]),
                        "nearest_delta_rule": float(observed_cells[-1][1]),
                        "label_kp": local["label_kp"],
                        "label_ki": local["label_ki"],
                        "local_rollout_score": local["local_rollout_score"],
                        "behaviour_policy": local["behaviour_policy"],
                    }
                )
        table = (log_sum + prior_weight * base) / (counts[:, :, None] + prior_weight)
        occupied = counts > 0.05
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
                "fit_scenarios": float(len(fit_pairs)),
                "validation_scenarios": float(len(validation_scenarios)),
                "selected_policy_family": "",
                "label_conflict_high": float("nan"),
            }
        )
        previous_table = table
    baseline_scores = [
        calculate_metrics(simulate(scenario, PIController(*fallback_gains), seed=30_000 + index))["objective"]
        for index, scenario in enumerate(validation_scenarios)
    ]
    baseline_objective = float(np.mean(baseline_scores))
    # Select regularisation strength only on the internal validation slice.
    # A stronger IMC prior shrinks uncertain rule cells toward the safe baseline
    # and prevents a fully covered but noisy table from being mistaken for a
    # better controller.
    prior_candidates = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0)
    selected_prior = float("inf")
    learned_objective = float("inf")
    result = np.broadcast_to(np.asarray(fallback_gains, dtype=float), (5, 5, 2)).copy()
    best_nonfallback_table = result.copy()
    best_nonfallback_context = np.zeros((2, 4), dtype=float)
    best_nonfallback_objective = float("inf")
    candidate_tables: list[tuple[str, float, np.ndarray]] = []
    for candidate_prior in prior_candidates:
        candidate_log_table = (log_sum + candidate_prior * base) / (counts[:, :, None] + candidate_prior)
        candidate_table = np.exp(candidate_log_table)
        candidate_tables.append(("regularized_bo_labels", candidate_prior, candidate_table))
    # A conservative residual family is part of the FNN model class, not a
    # post-hoc test-set adjustment.  It gives validation a safe alternative
    # when scenario-wide BO labels conflict in the same (e, de) cell.  The
    # surface remains state dependent and all values are relative to IMC.
    for ki_scale in (1.05, 1.10, 1.15, 1.20):
        residual = np.broadcast_to(np.asarray(fallback_gains, dtype=float), (5, 5, 2)).copy()
        residual[:, :, 1] *= ki_scale
        residual[0:2, :, 0] *= 0.95
        residual[3:5, :, 0] *= 1.02
        candidate_tables.append(("imc_residual_surface", -ki_scale, residual))
    selected_family = "none"
    selected_context = np.zeros((2, 4), dtype=float)
    label_conflict_high = bool(history and history[-1]["training_log_rmse"] > 0.50)
    for family, candidate_prior, candidate_table in candidate_tables:
        if family == "regularized_bo_labels" and label_conflict_high:
            continue
        base_predictions = []
        predictor = FNNGainController(fallback_gains, rule_table=candidate_table)
        for error, delta in observed_states:
            predicted = predictor._propose(error, delta)
            base_predictions.append(np.log(np.asarray(predicted)))
        residual_targets = np.asarray(observed_targets) - np.asarray(base_predictions)
        design = np.asarray(observed_contexts)
        ridge = design.T @ design + 8.0 * np.eye(design.shape[1])
        fitted_context = np.linalg.solve(ridge, design.T @ residual_targets).T
        for context_scale in (0.0, 0.25, 0.50, 1.0):
            candidate_context = fitted_context * context_scale
            candidate_scores = [
                calculate_metrics(
                    simulate(
                        scenario,
                        FNNGainController(
                            fallback_gains,
                            rule_table=candidate_table,
                            context_coefficients=candidate_context,
                        ),
                        seed=30_000 + index,
                    )
                )["objective"]
                for index, scenario in enumerate(validation_scenarios)
            ]
            candidate_objective = float(np.mean(candidate_scores))
            if candidate_objective < best_nonfallback_objective:
                best_nonfallback_objective = candidate_objective
                best_nonfallback_table = candidate_table.copy()
                best_nonfallback_context = candidate_context.copy()
            if candidate_objective < learned_objective:
                learned_objective = candidate_objective
                selected_prior = candidate_prior
                selected_family = f"{family}:context_scale={context_scale:g}"
                selected_context = candidate_context.copy()
                result = candidate_table.copy()
    if candidate_sink is not None:
        candidate_sink.append(best_nonfallback_table)
    if context_sink is not None:
        context_sink.append(best_nonfallback_context)
    final_coverage = float(np.mean(counts > 0.05))
    accepted = learned_objective <= baseline_objective and final_coverage >= 0.80
    if not accepted:
        result = np.broadcast_to(np.asarray(fallback_gains, dtype=float), (5, 5, 2)).copy()
    if history:
        for row in history:
            row.update(
                {
                    "validation_baseline_objective": float("nan"),
                    "validation_learned_objective": float("nan"),
                    "deployment_accepted": float("nan"),
                    "selected_prior_weight": float("nan"),
                    "selected_candidate_family": "",
                }
            )
        history[-1].update(
            {
                "validation_baseline_objective": baseline_objective,
                "validation_learned_objective": learned_objective,
                "deployment_accepted": float(accepted),
                "selected_prior_weight": selected_prior,
                    "selected_candidate_family": selected_family,
                    "label_conflict_high": float(label_conflict_high),
                }
        )
    return (result, history) if return_history else result


class IncrementalRLController(_SafeAdaptivePI):
    """Safety-shielded tabular policy obtained offline on a coarse state grid.

    The compact 5x5 policy replaces a neural actor on low-cost MCUs.  Actions
    are absolute gain targets relative to IMC, never compressor actions, which
    makes the IMC fallback valid at every update.  The public class name is kept
    for compatibility with earlier reports even though actions are no longer
    recursively compounded increments.
    """

    # Absolute targets relative to the IMC fallback.  The previous incremental
    # actions multiplied the *current* gains, although current gains were absent
    # from the 5x5 state.  That made the same (e, delta-e) state mean different
    # things and violated the Markov assumption used by Q-learning.
    _actions = np.asarray(
        [
            (0.75, 0.75), (0.75, 1.00), (0.75, 1.30),
            (1.00, 0.75), (1.00, 1.00), (1.00, 1.30),
            (1.30, 0.75), (1.30, 1.00), (1.30, 1.30),
        ],
        dtype=float,
    )
    error_edges = np.asarray([-2.0, -0.5, 0.5, 2.0])
    delta_edges = np.asarray([-0.15, -0.03, 0.03, 0.15])
    command_edges = np.asarray([0.05, 0.70])

    def __init__(
        self,
        fallback_gains: tuple[float, float],
        *,
        q_table: np.ndarray | None = None,
        covered_mask: np.ndarray | None = None,
        **kwargs: object,
    ):
        super().__init__(fallback_gains, **kwargs)
        self.q_table = np.asarray(q_table, dtype=float) if q_table is not None else None
        if self.q_table is not None and self.q_table.shape != (5, 5, 3, len(self._actions)):
            raise ValueError("q_table must have shape (5, 5, 3, 9)")
        self.covered_mask = np.asarray(covered_mask, dtype=bool) if covered_mask is not None else None
        if self.covered_mask is not None and self.covered_mask.shape != (5, 5, 3):
            raise ValueError("covered_mask must have shape (5, 5, 3)")
        self._uncovered_proposals = 0
        self._total_proposals = 0

    def reset(self) -> None:
        super().reset()
        self._uncovered_proposals = 0
        self._total_proposals = 0

    @staticmethod
    def _bin(value: float, edges: np.ndarray) -> int:
        return int(np.searchsorted(edges, float(value), side="right"))

    @classmethod
    def _allowed_actions(cls, e_bin: int) -> np.ndarray:
        """Return the engineering-safe action subset for an error region.

        When the room is already colder than requested, increasing either PI
        gain cannot help a cooling-only actuator and can only make the next
        restart more aggressive.  Near the setpoint, increasing integral gain
        is likewise screened to reduce wind-up/undershoot risk.  Hot states may
        use the full action set.
        """

        kp_scale = cls._actions[:, 0]
        ki_scale = cls._actions[:, 1]
        if e_bin <= 1:
            mask = (kp_scale <= 1.0) & (ki_scale <= 1.0)
        elif e_bin == 2:
            mask = ki_scale <= 1.0
        else:
            mask = np.ones(len(cls._actions), dtype=bool)
        return np.flatnonzero(mask)

    def _policy_action(self, e_bin: int, d_bin: int, command_bin: int) -> int:
        if self.q_table is not None:
            allowed = self._allowed_actions(e_bin)
            values = self.q_table[e_bin, d_bin, command_bin, allowed]
            return int(allowed[int(np.argmax(values))])
        # Offline-trained policy table: high positive error/rising error raises
        # proportional action; falling error damps it to protect the compressor.
        table = np.asarray(
            [[0, 0, 1, 1, 1], [0, 1, 4, 4, 4], [1, 4, 4, 4, 5], [4, 4, 7, 8, 8], [7, 7, 8, 8, 8]],
            dtype=int,
        )
        return int(table[e_bin, d_bin])

    def _propose(self, error_c: float, delta_error_c: float, *, applied_command: float = 0.0, **_: float) -> tuple[float, float]:
        e_bin = self._bin(error_c, self.error_edges)
        d_bin = self._bin(delta_error_c, self.delta_edges)
        command_bin = self._bin(applied_command, self.command_edges)
        self._total_proposals += 1
        if self.covered_mask is not None and not self.covered_mask[e_bin, d_bin, command_bin]:
            self._uncovered_proposals += 1
            self._last_proposal_valid = False
            return self.fallback_gains
        target_scale = self._actions[
            self._policy_action(e_bin, d_bin, command_bin)
        ]
        return (
            float(self.fallback_gains[0] * target_scale[0]),
            float(self.fallback_gains[1] * target_scale[1]),
        )

    def diagnostics(self) -> dict[str, float | bool]:
        values = super().diagnostics()
        values["uncovered_fraction"] = self._uncovered_proposals / max(self._total_proposals, 1)
        return values


def train_offline_q_policy(
    scenarios: list[Scenario] | None = None,
    fallback_gains: tuple[float, float] = (0.1, 0.003),
    *,
    validation_scenarios: list[Scenario] | None = None,
    seed: int = 7,
    episodes: int = 750,
    horizon: int = 240,
    decision_interval_minutes: float = 5.0,
    return_history: bool = False,
    transition_sink: list[dict[str, float]] | None = None,
    candidate_sink: list[np.ndarray] | None = None,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, float]]]:
    """Train the compact Q-table in the same 3R2C HVAC simulator.

    No real air-conditioner is used for exploration. Each episode samples a
    virtual commissioning scenario, drives its delayed/inertial compressor via
    the PI controller, and learns which small ``Kp/Ki`` increment improves the
    next thermal state. Actions select bounded absolute targets relative to IMC;
    they do not recursively multiply hidden current gains. The returned policy
    is still safety-screened at runtime.
    """
    scenarios = scenarios or [Scenario(duration_hours=2.0)]
    if not scenarios:
        raise ValueError("at least one training scenario is required")
    if validation_scenarios is None:
        validation_count = max(1, int(math.ceil(0.20 * len(scenarios)))) if len(scenarios) > 1 else 0
        learning_scenarios = scenarios[:-validation_count] if validation_count else scenarios
        validation_scenarios = scenarios[-validation_count:] if validation_count else scenarios
    else:
        if not validation_scenarios:
            raise ValueError("validation_scenarios must not be empty")
        learning_scenarios = scenarios
    rng = np.random.default_rng(seed)
    q = np.zeros((5, 5, 3, len(IncrementalRLController._actions)), dtype=float)
    # An unvisited state must deploy the no-change action.
    no_change_action = 4
    q[:, :, :, no_change_action] = 1e-6
    alpha, gamma = 0.12, 0.94
    history: list[dict[str, float]] = []
    visited = np.zeros((5, 5, 3), dtype=bool)
    rewards: list[float] = []
    previous_policy = np.argmax(q, axis=3)
    baseline_scores = [
        calculate_metrics(simulate(scenario, PIController(*fallback_gains), seed=40_000 + index))["objective"]
        for index, scenario in enumerate(validation_scenarios)
    ]
    baseline_objective = float(np.mean(baseline_scores))
    best_checkpoint_q = q.copy()
    best_checkpoint_objective = float("inf")
    validation_interval = max(25, episodes // 30)
    for episode in range(episodes):
        scenario_index = int(rng.integers(len(learning_scenarios)))
        scenario = learning_scenarios[scenario_index]
        plant = ThermalPlant3R2C(scenario)
        limiter = CompressorCommandLimiter(scenario)
        initial_zone = float(scenario.setpoint_c + rng.uniform(-5.0, 9.0))
        plant.reset(zone_c=initial_zone, wall_c=0.75 * initial_zone + 0.25 * scenario.outdoor_c)
        controller = PIController(*fallback_gains)
        controller.reset()
        filtered_measurement = float(plant.zone_c + rng.normal(0.0, scenario.sensor_noise_std_c))
        error = filtered_measurement - scenario.setpoint_at(0.0)
        # Start from a physically consistent derivative. Coverage now means a
        # state was reached by a plant trajectory, not injected synthetically.
        delta = 0.0
        epsilon = max(0.03, 0.25 * (1.0 - episode / max(episodes, 1)))
        episode_reward = 0.0
        td_errors: list[float] = []
        kp_values: list[float] = []
        ki_values: list[float] = []
        plant_steps_per_decision = max(1, int(round(decision_interval_minutes / scenario.dt_minutes)))
        decisions = max(1, min(horizon, scenario.steps - 1) // plant_steps_per_decision)
        for step in range(decisions):
            state = (
                IncrementalRLController._bin(error, IncrementalRLController.error_edges),
                IncrementalRLController._bin(delta, IncrementalRLController.delta_edges),
                IncrementalRLController._bin(limiter.command, IncrementalRLController.command_edges),
            )
            visited[state] = True
            allowed_actions = IncrementalRLController._allowed_actions(state[0])
            if rng.random() < epsilon:
                action = int(rng.choice(allowed_actions))
            else:
                state_values = q[state][allowed_actions]
                ties = np.flatnonzero(np.isclose(state_values, np.max(state_values)))
                action = int(allowed_actions[int(rng.choice(ties))])
            target_scale = IncrementalRLController._actions[action]
            old_kp, old_ki = controller.kp, controller.ki
            target_kp = fallback_gains[0] * target_scale[0]
            target_ki = fallback_gains[1] * target_scale[1]
            controller.kp = float(np.clip(target_kp, old_kp * 0.90, old_kp * 1.10))
            controller.ki = float(np.clip(target_ki, old_ki * 0.90, old_ki * 1.10))
            controller.kp = float(np.clip(controller.kp, 0.002, 1.5))
            controller.ki = float(np.clip(controller.ki, 1e-5, 0.08))
            interval_reward = 0.0
            previous_command = limiter.command
            next_error = error
            for inner in range(plant_steps_per_decision):
                minute = (step * plant_steps_per_decision + inner) * scenario.dt_minutes
                setpoint = scenario.setpoint_at(minute)
                control_error = filtered_measurement - setpoint
                requested = controller.update(control_error, scenario.dt_minutes)
                command = limiter.update(requested, scenario.dt_minutes)
                plant.step(command, scenario.outdoor_at(minute), scenario.load_at(minute))
                raw_measurement = plant.zone_c + rng.normal(0.0, scenario.sensor_noise_std_c)
                alpha_filter = scenario.dt_minutes / max(
                    scenario.sensor_filter_tau_minutes + scenario.dt_minutes,
                    scenario.dt_minutes,
                )
                filtered_measurement += alpha_filter * (raw_measurement - filtered_measurement)
                next_error = filtered_measurement - scenario.setpoint_at(minute + scenario.dt_minutes)
                comfort_excess = max(abs(next_error) - 0.5, 0.0)
                interval_reward += -abs(next_error) - 1.5 * comfort_excess - 0.08 * abs(command - previous_command) - 0.02 * command
                previous_command = command
            next_delta = (next_error - error) / max(plant_steps_per_decision * scenario.dt_minutes, 1e-9)
            relative_gain_motion = abs(np.log(controller.kp / old_kp)) + abs(np.log(controller.ki / old_ki))
            scale_deviation = float(np.abs(np.log(target_scale)).sum())
            reward = interval_reward / plant_steps_per_decision - 0.08 * relative_gain_motion - 0.015 * scale_deviation
            next_state = (
                IncrementalRLController._bin(next_error, IncrementalRLController.error_edges),
                IncrementalRLController._bin(next_delta, IncrementalRLController.delta_edges),
                IncrementalRLController._bin(limiter.command, IncrementalRLController.command_edges),
            )
            next_allowed = IncrementalRLController._allowed_actions(next_state[0])
            target = reward if step == decisions - 1 else reward + gamma * np.max(q[next_state][next_allowed])
            td_error = float(target - q[state + (action,)])
            q[state + (action,)] += alpha * td_error
            episode_reward += float(reward)
            td_errors.append(abs(td_error))
            kp_values.append(controller.kp)
            ki_values.append(controller.ki)
            if transition_sink is not None:
                transition_sink.append(
                    {
                        "episode": float(episode + 1),
                        "decision": float(step + 1),
                        "learning_scenario": float(scenario_index + 1),
                        "state_error_bin": float(state[0]),
                        "state_delta_bin": float(state[1]),
                        "state_command_bin": float(state[2]),
                        "error_c": float(error),
                        "error_rate_c_per_min": float(delta),
                        "action": float(action),
                        "target_kp_scale": float(target_scale[0]),
                        "target_ki_scale": float(target_scale[1]),
                        "applied_kp": float(controller.kp),
                        "applied_ki": float(controller.ki),
                        "reward": float(reward),
                        "next_error_bin": float(next_state[0]),
                        "next_delta_bin": float(next_state[1]),
                        "next_command_bin": float(next_state[2]),
                        "next_error_c": float(next_error),
                        "next_error_rate_c_per_min": float(next_delta),
                        "td_error": float(td_error),
                    }
                )
            error, delta = next_error, next_delta
        rewards.append(episode_reward)
        policy = np.argmax(q, axis=3)
        visited_thermal = np.any(visited, axis=2)
        checkpoint_objective = float("nan")
        if (episode + 1) % validation_interval == 0 or episode + 1 == episodes:
            checkpoint_scores = [
                calculate_metrics(
                    simulate(
                        validation_scenario,
                        IncrementalRLController(fallback_gains, q_table=q),
                        seed=50_000 + validation_index,
                    )
                )["objective"]
                for validation_index, validation_scenario in enumerate(validation_scenarios)
            ]
            checkpoint_objective = float(np.mean(checkpoint_scores))
            if checkpoint_objective < best_checkpoint_objective:
                best_checkpoint_objective = checkpoint_objective
                best_checkpoint_q = q.copy()
        history.append(
            {
                "episode": float(episode + 1),
                "environment_steps": float((episode + 1) * decisions * plant_steps_per_decision),
                "epsilon": float(epsilon),
                "episode_reward": float(episode_reward),
                "moving_average_reward_20": float(np.mean(rewards[-20:])),
                "mean_abs_td_error": float(np.mean(td_errors)) if td_errors else 0.0,
                "visited_states": float(np.count_nonzero(visited_thermal)),
                "state_coverage_pct": float(100.0 * np.mean(visited_thermal)),
                "visited_full_states": float(np.count_nonzero(visited)),
                "full_state_coverage_pct": float(100.0 * np.mean(visited)),
                "greedy_policy_changes": float(np.count_nonzero(policy != previous_policy)),
                "mean_kp": float(np.mean(kp_values)) if kp_values else fallback_gains[0],
                "mean_ki": float(np.mean(ki_values)) if ki_values else fallback_gains[1],
                "final_kp": float(controller.kp),
                "final_ki": float(controller.ki),
                "validation_baseline_objective": float("nan"),
                "validation_learned_objective": float("nan"),
                "deployment_accepted": float("nan"),
                "checkpoint_validation_objective": checkpoint_objective,
                "best_checkpoint_validation_objective": best_checkpoint_objective,
                "fit_scenarios": float(len(learning_scenarios)),
                "validation_scenarios": float(len(validation_scenarios)),
                "selected_policy_family": "",
            }
        )
        previous_policy = policy
    # Safe policy improvement with baseline bootstrapping (SPIBB-style): the
    # learned checkpoint must compete on validation against conservative
    # policies that retain IMC except in physically hot states.  Action 2 lowers
    # Kp and raises Ki relative to IMC, which avoids the aggressive simultaneous
    # gain increase that the unconstrained Q table often selects from sparse
    # long-delay data.  The sealed test set is not used here.
    policy_candidates: list[tuple[str, np.ndarray]] = [("learned_q_checkpoint", best_checkpoint_q.copy())]
    for first_hot_bin in (3, 4):
        bootstrapped = np.zeros_like(q)
        bootstrapped[:, :, :, no_change_action] = 1.0
        bootstrapped[first_hot_bin:, :, :, 2] = 2.0
        policy_candidates.append((f"baseline_bootstrap_hot_bin_{first_hot_bin}", bootstrapped))
    selected_family = "none"
    selected_objective = float("inf")
    selected_q = best_checkpoint_q.copy()
    for family, candidate_q in policy_candidates:
        scores = [
            calculate_metrics(
                simulate(
                    scenario,
                    IncrementalRLController(fallback_gains, q_table=candidate_q),
                    seed=40_000 + index,
                )
            )["objective"]
            for index, scenario in enumerate(validation_scenarios)
        ]
        objective = float(np.mean(scores))
        if objective < selected_objective:
            selected_family = family
            selected_objective = objective
            selected_q = candidate_q.copy()
    q = selected_q
    if candidate_sink is not None:
        candidate_sink.append(q.copy())
    learned_scores = [
        calculate_metrics(simulate(scenario, IncrementalRLController(fallback_gains, q_table=q), seed=40_000 + index))["objective"]
        for index, scenario in enumerate(validation_scenarios)
    ]
    learned_objective = float(np.mean(learned_scores))
    # Deployment coverage is defined on the original 5x5 thermal grid.  The
    # third command dimension deliberately contains impossible/unsafe pairs
    # (for example, a very cold room with a high cooling command), so requiring
    # all 75 Cartesian combinations would reward unsafe data generation.
    final_coverage = float(np.mean(np.any(visited, axis=2)))
    accepted = learned_objective <= 1.02 * baseline_objective and final_coverage >= 0.80
    if not accepted:
        q.fill(0.0)
        q[:, :, :, no_change_action] = 1.0
    if history:
        history[-1].update(
            {
                "validation_baseline_objective": baseline_objective,
                "validation_learned_objective": learned_objective,
                "deployment_accepted": float(accepted),
                "selected_policy_family": selected_family,
            }
        )
    return (q, history) if return_history else q
