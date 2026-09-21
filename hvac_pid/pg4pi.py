"""Policy-gradient PI tuning for the small HVAC plant.

The sensitivity extension uses :mod:`hvac_pid.core`'s ``PI`` directly.
``PI`` has a useful and slightly unusual convention: its integral is an
output contribution, and anti-windup holds that contribution while the
unclipped PI output is saturated in the direction of the error.  PG4PI uses
the same controller and propagates the corresponding two sensitivities.

Only the policy is differentiated.  The plant is run forward to obtain a
trajectory, but no derivative of its state transition is used.  This is the
score-function estimator used by the upstream PG4PI example, adapted to a
bounded, scalar HVAC command.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Iterable

import numpy as np

from .core import Gains, PI, Plant, Scenario
from .tuning import HIGH, LOW, TuningResult, identify, in_bounds, simc


# These are the same logarithmic gain limits used by the BO tuner.  Keep the
# public constants here too: it makes the policy parameterization inspectable
# without requiring callers to know about the BO implementation.
GAIN_LOW = np.exp(LOW)
GAIN_HIGH = np.exp(HIGH)
DEFAULT_NOISE_STD = 0.1
DEFAULT_LEARNING_RATE = 1e-5


def _sigmoid(value):
    """Numerically stable logistic map for scalar or array input."""
    value = np.asarray(value, dtype=float)
    positive = value >= 0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


# region pg_parameters
def gains_from_logits(logits) -> Gains:
    """Map unconstrained policy coordinates to the BO gain domain.

    The map is logistic in log-gain coordinates, so every finite coordinate
    produces positive gains strictly inside the BO bounds.
    """
    point = np.asarray(logits, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("PG4PI logits must be a finite length-two vector")
    # Keep the floating-point image away from the exact endpoints even for a
    # very large finite logit.  This preserves a finite inverse coordinate and
    # still honors the closed BO bounds in ordinary precision.
    fraction = np.clip(_sigmoid(point), np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    values = np.exp(LOW + fraction * (HIGH - LOW))
    return Gains(float(values[0]), float(values[1]))


def logits_from_gains(gains: Gains) -> np.ndarray:
    """Return the inverse of :func:`gains_from_logits`.

    Finite logits represent the open interval.  A gain at either BO boundary
    is rejected instead of being silently clipped or mapped to an arbitrary
    large number.
    """
    values = np.asarray([gains.kp, gains.ki], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= GAIN_LOW) or np.any(values >= GAIN_HIGH):
        raise ValueError(
            "PG4PI anchor gains must lie strictly inside the BO domain "
            f"({GAIN_LOW.tolist()}, {GAIN_HIGH.tolist()})"
        )
    fraction = (np.log(values) - LOW) / (HIGH - LOW)
    return np.log(fraction / (1.0 - fraction))


def gain_logit_jacobian(gains: Gains, logits) -> np.ndarray:
    """Derivative of ``(kp, ki)`` with respect to the two logits."""
    point = np.asarray(logits, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("PG4PI logits must be a finite length-two vector")
    fraction = _sigmoid(point)
    values = np.asarray([gains.kp, gains.ki], dtype=float)
    return np.diag(values * (HIGH - LOW) * fraction * (1.0 - fraction))
# endregion pg_parameters


@dataclass
class PIUpdate:
    """One PI update and its conditioned parameter sensitivity."""

    mean: float
    raw: float
    integral: float
    integral_sensitivity: np.ndarray
    mean_sensitivity: np.ndarray
    integrated: bool
    saturated: bool

    def __iter__(self):
        """Allow the common ``mean, jacobian = update_with_sensitivity`` form."""
        yield self.mean
        yield self.mean_sensitivity


# region pg_sensitivity
class SensitivityPI(PI):
    """The shared core PI controller plus conditioned gain sensitivities.

    ``super().update`` remains the source of the controller output.  The
    candidate branch is inspected before that call so the sensitivity follows
    exactly the same anti-windup decision, including the case where a rejected
    candidate leaves an otherwise unsaturated accepted output.
    """

    def __init__(self, gains: Gains):
        super().__init__(gains)
        self.integral_sensitivity = np.zeros(2, dtype=float)

    def update_with_sensitivity(self, error: float, dt: float) -> PIUpdate:
        previous_sensitivity = self.integral_sensitivity.copy()
        candidate = self.integral + self.gains.ki * error * dt
        candidate_sensitivity = np.asarray(
            [previous_sensitivity[0], previous_sensitivity[1] + error * dt],
            dtype=float,
        )
        candidate_raw = self.gains.kp * error + candidate
        integrated = (
            0.0 <= candidate_raw <= 1.0
            or (candidate_raw > 1.0 and error < 0.0)
            or (candidate_raw < 0.0 and error > 0.0)
        )

        # Keep the shared PI implementation authoritative for both the
        # accepted output-contribution and the final clipping operation.
        mean = super().update(error, dt)
        if integrated:
            self.integral_sensitivity = candidate_sensitivity

        raw = self.gains.kp * error + self.integral
        saturated = raw < 0.0 or raw > 1.0
        if saturated:
            mean_sensitivity = np.zeros(2, dtype=float)
        else:
            mean_sensitivity = np.asarray([error, 0.0], dtype=float) + self.integral_sensitivity
        return PIUpdate(
            mean=float(mean),
            raw=float(raw),
            integral=float(self.integral),
            integral_sensitivity=self.integral_sensitivity.copy(),
            mean_sensitivity=mean_sensitivity,
            integrated=bool(integrated),
            saturated=bool(saturated),
        )


def pi_mean_and_sensitivity(errors: Iterable[float], gains: Gains, dt: float):
    """Evaluate PI means and ``d mean / d(kp, ki)`` for a fixed error history.

    This helper is useful for checking the policy derivative independently of
    the plant.  The returned Jacobian has shape ``(len(errors), 2)``.
    """
    errors = np.asarray(list(errors), dtype=float)
    if errors.ndim != 1:
        raise ValueError("errors must be one-dimensional")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    means = np.empty(errors.size, dtype=float)
    jacobian = np.empty((errors.size, 2), dtype=float)
    controller = SensitivityPI(gains)
    for index, error in enumerate(errors):
        update = controller.update_with_sensitivity(error, dt)
        means[index] = update.mean
        jacobian[index] = update.mean_sensitivity
    return means, jacobian


# endregion pg_sensitivity


@dataclass
class PG4PITrace:
    """A stochastic fixed-gain HVAC trajectory.

    ``latent`` is the Gaussian action before command clipping.  It is kept in
    the trace because it is the action whose log likelihood supplies the
    policy score.  ``command`` is the value actually sent to :class:`Plant`.
    """

    time: np.ndarray
    temperature: np.ndarray
    next_temperature: np.ndarray
    setpoint: np.ndarray
    disturbance: np.ndarray
    error: np.ndarray
    mean: np.ndarray
    latent: np.ndarray
    command: np.ndarray
    noise: np.ndarray
    mean_sensitivity: np.ndarray
    integral: np.ndarray
    integral_sensitivity: np.ndarray
    kp: np.ndarray
    ki: np.ndarray
    dt: float

    @property
    def preclip(self):
        """Alias for the pre-clipping Gaussian action."""
        return self.latent

    @property
    def iae(self) -> float:
        return float(np.abs(self.error).sum() * self.dt)

    @property
    def return_value(self) -> float:
        return -self.iae


def _noise_array(steps: int, noise_std: float, rng, noise) -> np.ndarray:
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and non-negative")
    if noise is not None:
        values = np.asarray(noise, dtype=float)
        if values.ndim == 0:
            values = np.full(steps, float(values), dtype=float)
        if values.shape != (steps,) or not np.all(np.isfinite(values)):
            raise ValueError(f"noise must contain exactly {steps} finite values")
        return values.copy()
    if rng is None:
        rng = np.random.default_rng()
    return np.asarray(rng.normal(0.0, noise_std, size=steps), dtype=float)


def rollout(
    scenario: Scenario,
    gains: Gains,
    *,
    rng=None,
    noise=None,
    noise_std: float = DEFAULT_NOISE_STD,
) -> PG4PITrace:
    """Run one fixed-gain, noisy HVAC trajectory.

    ``noise`` can be supplied to replay a trajectory exactly.  If omitted,
    one Gaussian draw is made at every physical plant step.  The latent action
    is clipped only for the plant command; the latent value remains in the
    returned trace and in :func:`trajectory_gradient`.
    """
    steps = scenario.steps
    noise_values = _noise_array(steps, noise_std, rng, noise)
    plant = Plant(scenario)
    time = np.arange(steps, dtype=float) * scenario.dt
    temperature = np.empty(steps, dtype=float)
    next_temperature = np.empty(steps, dtype=float)
    setpoint = np.full(steps, scenario.setpoint, dtype=float)
    disturbance = np.empty(steps, dtype=float)
    error_values = np.empty(steps, dtype=float)
    means = np.empty(steps, dtype=float)
    latent = np.empty(steps, dtype=float)
    command = np.empty(steps, dtype=float)
    mean_sensitivity = np.empty((steps, 2), dtype=float)
    integral_values = np.empty(steps, dtype=float)
    integral_sensitivity = np.empty((steps, 2), dtype=float)

    controller = SensitivityPI(gains)
    for index, minute in enumerate(time):
        current_temperature = plant.temperature
        error = current_temperature - scenario.setpoint
        current_disturbance = scenario.disturbance if minute >= scenario.disturbance_at else 0.0
        update = controller.update_with_sensitivity(error, scenario.dt)
        action_latent = update.mean + noise_values[index]
        action = min(1.0, max(0.0, action_latent))

        temperature[index] = current_temperature
        next_temperature[index] = plant.step(action, current_disturbance)
        disturbance[index] = current_disturbance
        error_values[index] = error
        means[index] = update.mean
        latent[index] = action_latent
        command[index] = action
        mean_sensitivity[index] = update.mean_sensitivity
        integral_values[index] = update.integral
        integral_sensitivity[index] = update.integral_sensitivity

    return PG4PITrace(
        time=time,
        temperature=temperature,
        next_temperature=next_temperature,
        setpoint=setpoint,
        disturbance=disturbance,
        error=error_values,
        mean=means,
        latent=latent,
        command=command,
        noise=noise_values,
        mean_sensitivity=mean_sensitivity,
        integral=integral_values,
        integral_sensitivity=integral_sensitivity,
        kp=np.full(steps, gains.kp, dtype=float),
        ki=np.full(steps, gains.ki, dtype=float),
        dt=scenario.dt,
    )


# region pg_score
def trajectory_gradient(
    trace: PG4PITrace,
    noise_std: float = DEFAULT_NOISE_STD,
    *,
    return_to_go: bool = True,
) -> np.ndarray:
    """Estimate the gradient of trajectory return with respect to ``kp, ki``.

    The Gaussian score is evaluated on ``trace.latent``.  In particular,
    replacing it with ``trace.command`` would silently make saturated actions
    contribute a different and biased score.  The Jacobian in the trace is
    conditioned on the observed error history; no plant derivative appears.
    """
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and non-negative")
    if trace.mean_sensitivity.shape != (trace.mean.size, 2):
        raise ValueError("trace mean_sensitivity has an invalid shape")
    if noise_std == 0:
        return np.zeros(2, dtype=float)
    score = (trace.latent - trace.mean) / (noise_std * noise_std)
    rewards = -np.abs(trace.error) * trace.dt
    weights = np.cumsum(rewards[::-1])[::-1] if return_to_go else np.full(trace.mean.size, rewards.sum())
    gradient = np.sum((weights * score)[:, None] * trace.mean_sensitivity, axis=0)
    if not np.all(np.isfinite(gradient)):
        raise FloatingPointError("non-finite PG4PI trajectory gradient")
    return np.asarray(gradient, dtype=float)
# endregion pg_score


def _noise_summary(values: np.ndarray, noise_std: float) -> dict:
    return {
        "std": float(noise_std),
        "mean": float(np.mean(values)) if values.size else 0.0,
        "sample_std": float(np.std(values)) if values.size else 0.0,
        "l2": float(np.linalg.norm(values)),
        "first": float(values[0]) if values.size else 0.0,
        "last": float(values[-1]) if values.size else 0.0,
        "samples": int(values.size),
    }


def _first_update_report(
    episode: int,
    gains: Gains,
    logits: np.ndarray,
    gradient_gains: np.ndarray,
    gradient_logits: np.ndarray,
    updated_logits: np.ndarray,
    trace: PG4PITrace,
    learning_rate: float,
    noise_std: float,
    return_to_go: bool,
) -> dict:
    updated_gains = gains_from_logits(updated_logits)
    nonzero = np.flatnonzero(np.linalg.norm(trace.mean_sensitivity, axis=1) > 0.0)
    if nonzero.size:
        index = int(nonzero[0])
        if noise_std:
            score_value = float((trace.latent[index] - trace.mean[index]) / (noise_std * noise_std))
            score_sum = np.sum(
                ((trace.latent - trace.mean) / (noise_std * noise_std))[:, None]
                * trace.mean_sensitivity,
                axis=0,
            )
        else:
            score_value = 0.0
            score_sum = np.zeros(2, dtype=float)
        rewards = -np.abs(trace.error) * trace.dt
        reward_to_go = float(np.sum(rewards[index:]))
        score_weight = reward_to_go if return_to_go else trace.return_value
        weighted_score = score_weight * score_value * trace.mean_sensitivity[index]
        first_nonzero = {
            "index": index,
            "time": float(trace.time[index]),
            "temperature": float(trace.temperature[index]),
            "error": float(trace.error[index]),
            "mean": float(trace.mean[index]),
            "latent": float(trace.latent[index]),
            "command": float(trace.command[index]),
            "mean_sensitivity": trace.mean_sensitivity[index].tolist(),
            "score": score_value,
            "reward_to_go": reward_to_go,
            "score_weight": float(score_weight),
            "weighted_score_contribution": weighted_score.tolist(),
            "score_sum": score_sum.tolist(),
        }
    else:
        first_nonzero = None
    return {
        "episode": int(episode),
        "return": float(trace.return_value),
        "iae": float(trace.iae),
        "gains_before": {"kp": gains.kp, "ki": gains.ki},
        "gains_after": {"kp": updated_gains.kp, "ki": updated_gains.ki},
        "logits_before": logits.tolist(),
        "logits_after": updated_logits.tolist(),
        "gradient_kp_ki": gradient_gains.tolist(),
        "gradient_logits": gradient_logits.tolist(),
        "learning_rate": float(learning_rate),
        "noise": _noise_summary(trace.noise, noise_std),
        "first_nonzero_sensitivity_step": first_nonzero,
        "first_step": {
            "time": float(trace.time[0]) if trace.time.size else 0.0,
            "temperature": float(trace.temperature[0]) if trace.temperature.size else None,
            "error": float(trace.error[0]) if trace.error.size else None,
            "mean": float(trace.mean[0]) if trace.mean.size else None,
            "latent": float(trace.latent[0]) if trace.latent.size else None,
            "command": float(trace.command[0]) if trace.command.size else None,
            "mean_sensitivity": trace.mean_sensitivity[0].tolist() if trace.mean.size else [0.0, 0.0],
            "integral": float(trace.integral[0]) if trace.integral.size else None,
            "integral_sensitivity": trace.integral_sensitivity[0].tolist() if trace.mean.size else [0.0, 0.0],
        },
    }


def _simc_anchor(scenario: Scenario) -> Gains:
    model = identify(scenario)[0]
    anchor = simc(model)
    # The BO domain is a policy parameterization contract.  A classical
    # anchor outside it is a commissioning error, not a reason to alter the
    # controller silently.
    if (
        not in_bounds(anchor)
        or anchor.kp <= GAIN_LOW[0]
        or anchor.kp >= GAIN_HIGH[0]
        or anchor.ki <= GAIN_LOW[1]
        or anchor.ki >= GAIN_HIGH[1]
    ):
        raise ValueError(
            "SIMC gains are outside the PG4PI/BO domain: "
            f"kp={anchor.kp:.12g}, ki={anchor.ki:.12g}; "
            f"allowed kp=[{GAIN_LOW[0]:.12g}, {GAIN_HIGH[0]:.12g}], "
            f"ki=[{GAIN_LOW[1]:.12g}, {GAIN_HIGH[1]:.12g}]"
        )
    return anchor


# region pg_update
def tune(
    scenario: Scenario,
    seed: int = 0,
    episodes: int = 500,
    *,
    noise_std: float = DEFAULT_NOISE_STD,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    return_to_go: bool = True,
) -> TuningResult:
    """Tune one HVAC object with fixed-gain stochastic PI trajectories.

    The SIMC gains initialize a pair of bounded log-gain coordinates.  Each
    episode uses one fixed gain pair for all physical steps, then updates the
    coordinates once using the negative-IAE trajectory return.  ``best`` in
    the returned :class:`~hvac_pid.tuning.TuningResult` is the final parameter
    value, including when an earlier episode happened to score better.
    """
    if not isinstance(episodes, (int, np.integer)) or episodes < 0:
        raise ValueError("episodes must be a non-negative integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and non-negative")

    start = perf_counter()
    anchor = _simc_anchor(scenario)
    logits = logits_from_gains(anchor)
    rng = np.random.default_rng(int(seed))
    trials: list[dict] = []
    first_update = None

    for episode in range(int(episodes)):
        gains = gains_from_logits(logits)
        trace = rollout(scenario, gains, rng=rng, noise_std=noise_std)
        gradient_gains = trajectory_gradient(trace, noise_std, return_to_go=return_to_go)
        gain_jacobian = gain_logit_jacobian(gains, logits)
        gradient_logits = gain_jacobian.T @ gradient_gains
        if not np.all(np.isfinite(gradient_logits)):
            raise FloatingPointError("non-finite PG4PI log-gain gradient")
        updated_logits = logits + learning_rate * gradient_logits
        if not np.all(np.isfinite(updated_logits)):
            raise FloatingPointError("non-finite PG4PI parameter update")

        summary = _noise_summary(trace.noise, noise_std)
        trial = {
            "episode": int(episode),
            "return": float(trace.return_value),
            "iae": float(trace.iae),
            "kp": float(gains.kp),
            "ki": float(gains.ki),
            "gradient": gradient_gains.tolist(),
            "gradient_kp": float(gradient_gains[0]),
            "gradient_ki": float(gradient_gains[1]),
            "gradient_logits": gradient_logits.tolist(),
            "logits": logits.tolist(),
            "noise": summary,
            "noise_std": float(noise_std),
            "noise_mean": summary["mean"],
            "noise_l2": summary["l2"],
            "steps": int(scenario.steps),
            "status": "updated",
        }
        trials.append(trial)
        if first_update is None:
            first_update = _first_update_report(
                episode, gains, logits, gradient_gains, gradient_logits,
                updated_logits, trace, learning_rate, noise_std, return_to_go
            )
        logits = updated_logits

    final_gains = gains_from_logits(logits)
    cost = {
        "algorithm": "PG4PI-HVAC",
        "seed": int(seed),
        "episodes": int(episodes),
        "simulations": int(episodes),
        "identification_steps": int(round(240 / scenario.dt)),
        "plant_steps": int(episodes) * scenario.steps,
        "proposal_rounds": 0,
        "noise_std": float(noise_std),
        "noise": {
            "distribution": "normal",
            "std": float(noise_std),
            "samples_per_episode": int(scenario.steps),
        },
        "learning_rate": float(learning_rate),
        "return_to_go": bool(return_to_go),
        "initial_gains": {"kp": anchor.kp, "ki": anchor.ki},
        "final_gains": {"kp": final_gains.kp, "ki": final_gains.ki},
        "first_update": first_update,
        "seconds": perf_counter() - start,
    }
    return TuningResult(
        best=final_gains,
        trials=trials,
        status="complete",
        stop_reason="budget exhausted",
        cost=cost,
    )
# endregion pg_update


def first_update_report(result: TuningResult):
    """Return the numeric first-update report stored in a tuning result."""
    return result.cost.get("first_update")


__all__ = [
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_NOISE_STD",
    "GAIN_LOW",
    "GAIN_HIGH",
    "LOW",
    "HIGH",
    "PG4PITrace",
    "PIUpdate",
    "SensitivityPI",
    "first_update_report",
    "gain_logit_jacobian",
    "gains_from_logits",
    "logits_from_gains",
    "pi_mean_and_sensitivity",
    "rollout",
    "trajectory_gradient",
    "tune",
]
