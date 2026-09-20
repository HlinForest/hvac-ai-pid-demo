"""Identification, classical rules and Bayesian experimental design."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from .core import Gains, Plant, Scenario, evaluate


@dataclass(frozen=True)
class Identified:
    gain: float
    tau: float
    delay: float


def step_experiment(scenario: Scenario):
    """Start at ambient equilibrium; apply u=0.5 for 240 minutes."""
    probe = replace(scenario, initial=scenario.ambient, duration=240.0,
                    disturbance=0.0)
    plant = Plant(probe)
    t = np.arange(probe.steps + 1) * probe.dt
    temperature = [plant.temperature]
    for _ in range(probe.steps):
        temperature.append(plant.step(0.5, 0.0))
    return t, np.asarray(temperature)


# region identification
def identify(scenario: Scenario):
    time, temperature = step_experiment(scenario)
    drop = temperature[0] - temperature

    def residual(parameters):
        gain, tau, delay = parameters
        prediction = 0.5 * gain * (1 - np.exp(-np.maximum(time - delay, 0) / tau))
        return prediction - drop

    fit = least_squares(residual, x0=[8.0, 20.0, 2.0],
                        bounds=([0.1, 0.1, 0.0], [100, 200, 30]),
                        xtol=1e-10, ftol=1e-10, gtol=1e-10)
    if not fit.success:
        raise RuntimeError(f"Step identification failed: {fit.message}")
    return Identified(*map(float, fit.x)), time, temperature
# endregion identification


# region classical
def zn(model: Identified) -> Gains:
    if model.delay <= 1e-8:
        raise ValueError("Z-N reaction-curve formula needs a positive delay")
    kp = 0.9 * model.tau / (model.gain * model.delay)
    return Gains(kp, kp / (3.33 * model.delay))


def simc(model: Identified, lam: float | None = None) -> Gains:
    lam = model.tau / 3 if lam is None else lam
    if lam <= 0:
        raise ValueError("lambda must be positive")
    kp = model.tau / (model.gain * (lam + model.delay))
    ti = min(model.tau, 4 * (lam + model.delay))
    return Gains(kp, kp / ti)
# endregion classical


@dataclass
class TuningResult:
    best: Gains | None
    trials: list[dict] = field(default_factory=list)
    status: str = "complete"
    stop_reason: str = "budget exhausted"
    cost: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


LOW = np.log([0.01, 0.0001])
HIGH = np.log([3.0, 0.5])


def encode(gains: Gains):
    return (np.log([gains.kp, gains.ki]) - LOW) / (HIGH - LOW)


def decode(point) -> Gains:
    return Gains(*np.exp(LOW + np.asarray(point) * (HIGH - LOW)))


def in_bounds(gains: Gains) -> bool:
    return 0.01 <= gains.kp <= 3 and 0.0001 <= gains.ki <= 0.5


def initial_samples(scenario: Scenario, seed: int):
    model, _, _ = identify(scenario)
    rng = np.random.default_rng(seed)
    gains = [zn(model), simc(model), *(decode(p) for p in rng.random((3, 2)))]
    # Classical observations stay exact even when a formula falls outside the
    # AI proposal domain. Both BO and LLM receive these same observations.
    return model, gains


def run_trial(scenario: Scenario, gains: Gains, index: int, source: str, **extra):
    start = perf_counter()
    result = evaluate(scenario, gains)
    return {"trial": index, "source": source, **asdict(gains), **result.metrics(),
            "seconds": perf_counter() - start, "status": "evaluated", **extra}


def best_gains(trials: list[dict]) -> Gains | None:
    valid = [t for t in trials if t["status"] == "evaluated"]
    if not valid:
        return None
    best = min(valid, key=lambda t: t["iae"])
    return Gains(best["kp"], best["ki"])


def fit_gp(x, y):
    # Fixed kernel hyperparameters keep the teaching loop small and predictable.
    return GaussianProcessRegressor(kernel=Matern(length_scale=[0.3, 0.3], nu=2.5),
                                    alpha=1e-6, normalize_y=True,
                                    optimizer=None).fit(x, y)


# region expected_improvement
def expected_improvement(mean, std, best):
    improvement = best - mean
    safe_std = np.maximum(std, 1e-12)
    z = improvement / safe_std
    ei = improvement * ndtr(z) + safe_std * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return np.where(std > 1e-12, ei, np.maximum(improvement, 0.0))
# endregion expected_improvement


def tune(scenario: Scenario, seed: int = 0, rounds: int = 15) -> TuningResult:
    if rounds < 0:
        raise ValueError("rounds cannot be negative")
    start = perf_counter()
    _, initial = initial_samples(scenario, seed)
    trials = [run_trial(scenario, gains, i, source)
              for i, (gains, source) in enumerate(zip(initial, ["ZN", "SIMC", "random", "random", "random"]))]
    rng = np.random.default_rng(seed + 1000)
    # region bo_loop
    for _ in range(rounds):
        x = np.asarray([encode(Gains(t["kp"], t["ki"])) for t in trials])
        y = np.asarray([t["iae"] for t in trials])
        gp = fit_gp(x, y)
        candidates = rng.random((1024, 2))
        mean, std = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mean, std, y.min())
        selected = int(np.argmax(ei))
        gains = decode(candidates[selected])
        trials.append(run_trial(scenario, gains, len(trials), "BO",
                                predicted_mean=float(mean[selected]),
                                predicted_std=float(std[selected]), ei=float(ei[selected])))
    # endregion bo_loop
    return TuningResult(best_gains(trials), trials, cost={
        "simulations": len(trials), "identification_steps": round(240 / scenario.dt),
        "plant_steps": len(trials) * scenario.steps, "proposal_rounds": rounds,
        "seconds": perf_counter() - start,
    })
