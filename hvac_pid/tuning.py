from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .config import Scenario
from .controllers import PIController, identify_fopdt, imc_pi, ziegler_nichols_pi
from .metrics import calculate_metrics
from .simulator import simulate


@dataclass(frozen=True)
class GainBounds:
    kp: tuple[float, float] = (0.002, 1.5)
    ki: tuple[float, float] = (1e-5, 0.08)


@dataclass(frozen=True)
class TuneResult:
    kp: float
    ki: float
    score: float
    evaluations: int
    history: tuple[dict[str, object], ...] = ()


class BayesianGainTuner:
    """Bounded offline Bayesian optimization for two positive PI gains."""

    def __init__(self, bounds: GainBounds | None = None, iterations: int = 8, candidates: int = 768):
        self.bounds = bounds or GainBounds()
        self.iterations = int(iterations)
        self.candidates = int(candidates)
        self._log_low = np.log([self.bounds.kp[0], self.bounds.ki[0]])
        self._log_high = np.log([self.bounds.kp[1], self.bounds.ki[1]])

    def _decode(self, unit_point: np.ndarray) -> tuple[float, float]:
        log_gain = self._log_low + np.asarray(unit_point) * (self._log_high - self._log_low)
        kp, ki = np.exp(log_gain)
        return float(kp), float(ki)

    def _encode(self, gains: tuple[float, float]) -> np.ndarray:
        clipped = np.clip(
            np.log(np.asarray(gains, dtype=float)),
            self._log_low,
            self._log_high,
        )
        return (clipped - self._log_low) / (self._log_high - self._log_low)

    def tune(self, scenario: Scenario, seed: int = 0) -> TuneResult:
        rng = np.random.default_rng(seed)
        model = identify_fopdt(scenario)
        seeds = [self._encode(imc_pi(model)), self._encode(ziegler_nichols_pi(model))]
        # Log-uniform exploration is important because Ki spans several decades.
        seeds.extend(rng.random((5, 2)))
        x = np.asarray(seeds, dtype=float)

        def evaluate(point: np.ndarray) -> float:
            kp, ki = self._decode(point)
            result = simulate(scenario, PIController(kp, ki), seed=seed)
            return calculate_metrics(result)["objective"]

        y = np.asarray([evaluate(point) for point in x])
        history: list[dict[str, object]] = []

        def record(point: np.ndarray, score: float, phase: str, *, expected_improvement: float = float("nan"), predicted_mean: float = float("nan"), predicted_std: float = float("nan")) -> None:
            seen_scores = np.asarray([float(row["objective"]) for row in history] + [float(score)])
            seen_points = [np.asarray(row["_unit_point"], dtype=float) for row in history] + [np.asarray(point, dtype=float)]
            best_index = int(np.argmin(seen_scores))
            kp, ki = self._decode(point)
            best_kp, best_ki = self._decode(seen_points[best_index])
            history.append(
                {
                    "evaluation": len(history) + 1,
                    "phase": phase,
                    "candidate_kp": kp,
                    "candidate_ki": ki,
                    "objective": float(score),
                    "best_kp": best_kp,
                    "best_ki": best_ki,
                    "best_objective": float(seen_scores[best_index]),
                    "expected_improvement": expected_improvement,
                    "predicted_mean": predicted_mean,
                    "predicted_std": predicted_std,
                    "_unit_point": np.asarray(point, dtype=float).tolist(),
                }
            )

        for index, (point, score) in enumerate(zip(x, y, strict=True)):
            phase = "IMC初始点" if index == 0 else "Z-N初始点" if index == 1 else "随机初始探索"
            record(point, float(score), phase)
        kernel = ConstantKernel(1.0, (0.05, 20.0)) * Matern(
            length_scale=np.ones(2) * 0.35,
            length_scale_bounds=(0.03, 3.0),
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-2))

        for iteration in range(self.iterations):
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=0,
                random_state=seed + iteration,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(x, y)
            candidates = rng.random((self.candidates, 2))
            mean, std = gp.predict(candidates, return_std=True)
            std = np.maximum(std, 1e-12)
            improvement = np.min(y) - mean - 0.01
            z = improvement / std
            expected_improvement = improvement * norm.cdf(z) + std * norm.pdf(z)
            candidate_index = int(np.argmax(expected_improvement))
            next_point = candidates[candidate_index]
            next_score = evaluate(next_point)
            x = np.vstack([x, next_point])
            y = np.append(y, next_score)
            record(
                next_point,
                float(next_score),
                "贝叶斯 EI 迭代",
                expected_improvement=float(expected_improvement[candidate_index]),
                predicted_mean=float(mean[candidate_index]),
                predicted_std=float(std[candidate_index]),
            )

        best = int(np.argmin(y))
        kp, ki = self._decode(x[best])
        public_history = tuple({key: value for key, value in row.items() if key != "_unit_point"} for row in history)
        return TuneResult(kp=kp, ki=ki, score=float(y[best]), evaluations=len(y), history=public_history)


def tune_global_fixed(
    scenarios: list[Scenario], *, iterations: int = 5, seed: int = 0
) -> TuneResult:
    """Bayesian search for one robust pair of gains across all commissioning contexts."""
    if not scenarios:
        raise ValueError("at least one scenario is required")
    tuner = BayesianGainTuner(iterations=0, candidates=512)
    rng = np.random.default_rng(seed)
    x = np.vstack([tuner._encode(imc_pi(identify_fopdt(scenarios[0]))), rng.random((6, 2))])

    def evaluate(point: np.ndarray) -> float:
        gains = tuner._decode(point)
        scores = [calculate_metrics(simulate(scenario, PIController(*gains), seed=seed + index))["objective"] for index, scenario in enumerate(scenarios)]
        return float(np.mean(scores))

    y = np.asarray([evaluate(point) for point in x])
    history: list[dict[str, object]] = []

    def record(point: np.ndarray, score: float, phase: str, *, expected_improvement: float = float("nan"), predicted_mean: float = float("nan"), predicted_std: float = float("nan")) -> None:
        seen_scores = np.asarray([float(row["objective"]) for row in history] + [float(score)])
        seen_points = [np.asarray(row["_unit_point"], dtype=float) for row in history] + [np.asarray(point, dtype=float)]
        best_index = int(np.argmin(seen_scores))
        kp, ki = tuner._decode(point)
        best_kp, best_ki = tuner._decode(seen_points[best_index])
        history.append(
            {
                "evaluation": len(history) + 1,
                "phase": phase,
                "candidate_kp": kp,
                "candidate_ki": ki,
                "objective": float(score),
                "best_kp": best_kp,
                "best_ki": best_ki,
                "best_objective": float(seen_scores[best_index]),
                "expected_improvement": expected_improvement,
                "predicted_mean": predicted_mean,
                "predicted_std": predicted_std,
                "_unit_point": np.asarray(point, dtype=float).tolist(),
            }
        )

    for index, (point, score) in enumerate(zip(x, y, strict=True)):
        record(point, float(score), "IMC初始点" if index == 0 else "随机初始探索")
    kernel = ConstantKernel(1.0, (0.05, 20.0)) * Matern(length_scale=np.ones(2) * 0.35, length_scale_bounds=(0.03, 3.0), nu=2.5) + WhiteKernel(noise_level=1e-5)
    for iteration in range(iterations):
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0, random_state=seed + iteration)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(x, y)
        candidates = rng.random((tuner.candidates, 2))
        mean, std = gp.predict(candidates, return_std=True)
        std = np.maximum(std, 1e-12)
        improvement = np.min(y) - mean - 0.01
        z = improvement / std
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        candidate_index = int(np.argmax(ei))
        point = candidates[candidate_index]
        score = evaluate(point)
        x, y = np.vstack([x, point]), np.append(y, score)
        record(
            point,
            float(score),
            "贝叶斯 EI 迭代",
            expected_improvement=float(ei[candidate_index]),
            predicted_mean=float(mean[candidate_index]),
            predicted_std=float(std[candidate_index]),
        )
    best = int(np.argmin(y))
    kp, ki = tuner._decode(x[best])
    public_history = tuple({key: value for key, value in row.items() if key != "_unit_point"} for row in history)
    return TuneResult(kp, ki, float(y[best]), len(y), public_history)


def generate_label_rows(
    scenarios: list[Scenario],
    *,
    bo_iterations: int,
    seed: int,
    progress: bool = True,
) -> list[dict[str, float]]:
    tuner = BayesianGainTuner(iterations=bo_iterations)
    rows: list[dict[str, float]] = []
    for index, scenario in enumerate(scenarios):
        tuned = tuner.tune(scenario, seed=seed + index * 31)
        model = identify_fopdt(scenario)
        zn_metrics = calculate_metrics(simulate(scenario, PIController(*ziegler_nichols_pi(model)), seed=seed))
        imc_metrics = calculate_metrics(simulate(scenario, PIController(*imc_pi(model)), seed=seed))
        row = dict(zip(Scenario.FEATURE_NAMES, scenario.context_vector(), strict=True))
        row.update(
            {
                "label_kp": tuned.kp,
                "label_ki": tuned.ki,
                "label_objective": tuned.score,
                "zn_objective": zn_metrics["objective"],
                "imc_objective": imc_metrics["objective"],
                "bo_evaluations": float(tuned.evaluations),
            }
        )
        rows.append({key: float(value) for key, value in row.items()})
        if progress and ((index + 1) % max(1, len(scenarios) // 8) == 0 or index + 1 == len(scenarios)):
            print(f"  label generation: {index + 1:>3}/{len(scenarios)}")
    return rows
