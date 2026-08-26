from __future__ import annotations

"""Risk-aware safe BO and simulator-gated LLM supervision for PI tuning.

The LLM is deliberately outside the real-time loop.  It may propose Kp/Ki,
but deterministic bounds, trust-region limits, repeated simulation and safety
constraints decide whether a proposal is accepted.
"""

from dataclasses import dataclass, replace
import json
import os
from typing import Protocol
from urllib import request
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .config import Scenario
from .controllers import PIController
from .metrics import calculate_metrics
from .simulator import simulate
from .tuning import GainBounds, TuneResult


@dataclass(frozen=True)
class RiskSafetyConfig:
    """Acceptance rules shared by safe BO and the LLM supervisor."""

    repeats: int = 2
    risk_weight: float = 0.75
    max_undershoot_c: float = 2.0
    max_slew_violations: float = 0.0
    max_subminimum_fraction: float = 0.0
    safe_probability_beta: float = 2.0
    validation_tolerance: float = 1.02
    max_risk_ratio_to_baseline: float = 1.25


@dataclass(frozen=True)
class CandidateEvaluation:
    kp: float
    ki: float
    mean_objective: float
    objective_std: float
    risk_objective: float
    worst_objective: float
    safety_margin: float
    safe: bool
    metrics_mean: dict[str, float]


@dataclass(frozen=True)
class SupervisedTuneResult:
    kp: float
    ki: float
    score: float
    accepted: bool
    fallback_used: bool
    evaluations: int
    history: tuple[dict[str, object], ...]


def _safety_margin(metrics: dict[str, float], config: RiskSafetyConfig) -> float:
    """q <= 0 means safe; positive values quantify the largest violation."""

    slew_margin = 0.5 if metrics["running_slew_violation_count"] > config.max_slew_violations else -0.5
    subminimum_margin = 0.5 if metrics["sub_minimum_running_fraction"] > config.max_subminimum_fraction + 1e-12 else -0.5
    return float(
        max(
            # Strictly negative inside the safe set, positive after violation.
            # The half-unit offset is appropriate for integer/binary metrics
            # and prevents every safe observation from lying exactly on q=0.
            0.5 - metrics["stable"],
            metrics["max_undershoot_c"] - config.max_undershoot_c,
            slew_margin,
            subminimum_margin,
        )
    )


def _relative_safety(result: CandidateEvaluation, baseline_risk: float, config: RiskSafetyConfig) -> CandidateEvaluation:
    """Add a safe-optimization budget that forbids catastrophic degradation."""

    relative_margin = result.risk_objective / max(baseline_risk, 1e-9) - config.max_risk_ratio_to_baseline
    margin = float(max(result.safety_margin, relative_margin))
    return replace(result, safety_margin=margin, safe=bool(margin <= 0.0))


def evaluate_candidate(
    scenarios: list[Scenario],
    gains: tuple[float, float],
    *,
    seed: int,
    config: RiskSafetyConfig,
) -> CandidateEvaluation:
    """Evaluate one gain pair over scenarios and independent noise repeats."""

    if not scenarios:
        raise ValueError("at least one scenario is required")
    kp, ki = map(float, gains)
    rows: list[dict[str, float]] = []
    for repeat in range(max(1, config.repeats)):
        for index, scenario in enumerate(scenarios):
            noise_seed = seed + repeat * 100_003 + index * 977
            rows.append(calculate_metrics(simulate(scenario, PIController(kp, ki), seed=noise_seed)))
    objectives = np.asarray([row["objective"] for row in rows], dtype=float)
    margins = np.asarray([_safety_margin(row, config) for row in rows], dtype=float)
    keys = rows[0].keys()
    metrics_mean = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    std = float(np.std(objectives, ddof=1)) if len(objectives) > 1 else 0.0
    risk = float(np.mean(objectives) + config.risk_weight * std)
    margin = float(np.max(margins))
    return CandidateEvaluation(
        kp=kp,
        ki=ki,
        mean_objective=float(np.mean(objectives)),
        objective_std=std,
        risk_objective=risk,
        worst_objective=float(np.max(objectives)),
        safety_margin=margin,
        safe=bool(margin <= 1e-12),
        metrics_mean=metrics_mean,
    )


class RiskAwareSafeBOTuner:
    """RaGoOSE-style constrained BO for a two-gain PI controller.

    This is an engineering adaptation, not a bit-for-bit reproduction: one GP
    models the mean-plus-standard-deviation objective and a second GP models the
    scalar safety margin.  Only conservatively predicted-safe candidates are
    selected after the known baseline and local seed evaluations.
    """

    def __init__(
        self,
        *,
        bounds: GainBounds | None = None,
        iterations: int = 8,
        candidates: int = 768,
        config: RiskSafetyConfig | None = None,
    ) -> None:
        self.bounds = bounds or GainBounds()
        self.iterations = int(iterations)
        self.candidates = int(candidates)
        self.config = config or RiskSafetyConfig()
        self._log_low = np.log([self.bounds.kp[0], self.bounds.ki[0]])
        self._log_high = np.log([self.bounds.kp[1], self.bounds.ki[1]])

    def _encode(self, gains: tuple[float, float]) -> np.ndarray:
        value = np.clip(np.log(np.asarray(gains, dtype=float)), self._log_low, self._log_high)
        return (value - self._log_low) / (self._log_high - self._log_low)

    def _decode(self, point: np.ndarray) -> tuple[float, float]:
        value = self._log_low + np.asarray(point, dtype=float) * (self._log_high - self._log_low)
        kp, ki = np.exp(value)
        return float(kp), float(ki)

    @staticmethod
    def _gp(seed: int) -> GaussianProcessRegressor:
        kernel = ConstantKernel(1.0, (0.05, 20.0)) * Matern(
            length_scale=np.ones(2) * 0.30,
            length_scale_bounds=(0.03, 3.0),
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 0.2))
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0, random_state=seed)

    def tune(
        self,
        scenarios: list[Scenario],
        baseline_gains: tuple[float, float],
        *,
        seed: int = 0,
    ) -> TuneResult:
        if not scenarios:
            raise ValueError("at least one scenario is required")
        rng = np.random.default_rng(seed)
        base = self._encode(baseline_gains)
        # Local multiplicative seeds preserve the known controller's scale.
        multipliers = ((1.0, 1.0), (0.80, 0.80), (1.20, 0.90), (0.90, 1.20))
        x = np.asarray([self._encode((baseline_gains[0] * a, baseline_gains[1] * b)) for a, b in multipliers])
        evaluations: list[CandidateEvaluation] = []
        history: list[dict[str, object]] = []
        baseline_risk: float | None = None

        def run(point: np.ndarray, phase: str, predicted_margin: float = float("nan"), predicted_margin_std: float = float("nan")) -> None:
            nonlocal baseline_risk
            raw_result = evaluate_candidate(scenarios, self._decode(point), seed=seed + len(evaluations) * 10_007, config=self.config)
            if baseline_risk is None:
                baseline_risk = raw_result.risk_objective
            result = _relative_safety(raw_result, baseline_risk, self.config)
            evaluations.append(result)
            history.append(
                {
                    "evaluation": len(evaluations),
                    "phase": phase,
                    "candidate_kp": result.kp,
                    "candidate_ki": result.ki,
                    "mean_objective": result.mean_objective,
                    "objective_std": result.objective_std,
                    "risk_objective": result.risk_objective,
                    "worst_objective": result.worst_objective,
                    "safety_margin": result.safety_margin,
                    "safe": int(result.safe),
                    "predicted_safety_margin": predicted_margin,
                    "predicted_safety_std": predicted_margin_std,
                }
            )

        for index, point in enumerate(x):
            run(point, "known baseline" if index == 0 else "local qualification seed")

        for iteration in range(self.iterations):
            risk_y = np.asarray([item.risk_objective for item in evaluations])
            safety_y = np.asarray([item.safety_margin for item in evaluations])
            objective_gp, safety_gp = self._gp(seed + iteration), self._gp(seed + 1000 + iteration)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                objective_gp.fit(x, risk_y)
                safety_gp.fit(x, safety_y)
            pool = rng.random((self.candidates, 2))
            mean, std = objective_gp.predict(pool, return_std=True)
            q_mean, q_std = safety_gp.predict(pool, return_std=True)
            conservative_q = q_mean + self.config.safe_probability_beta * q_std
            predicted_safe = conservative_q <= 0.0
            # If the safety GP is initially too conservative, remain in a small
            # trust region around the best observed safe point.
            safe_indices = [i for i, item in enumerate(evaluations) if item.safe]
            if not np.any(predicted_safe) and safe_indices:
                best_safe = min(safe_indices, key=lambda i: risk_y[i])
                distance = np.linalg.norm(pool - x[best_safe], axis=1)
                predicted_safe = distance <= 0.10
            if not np.any(predicted_safe):
                break
            # Lower confidence bound: minimize performance while still learning.
            acquisition = mean - 1.25 * std
            acquisition[~predicted_safe] = np.inf
            selected = int(np.argmin(acquisition))
            point = pool[selected]
            run(point, "safe risk-aware BO", float(q_mean[selected]), float(q_std[selected]))
            x = np.vstack([x, point])

        safe_indices = [i for i, item in enumerate(evaluations) if item.safe]
        # The baseline can itself fail a newly tightened requirement; retain it
        # instead of silently deploying an unevaluated or unsafe candidate.
        best = min(safe_indices, key=lambda i: evaluations[i].risk_objective) if safe_indices else 0
        for index, row in enumerate(history):
            row["selected"] = int(index == best)
        chosen = evaluations[best]
        return TuneResult(chosen.kp, chosen.ki, chosen.risk_objective, len(evaluations), tuple(history))


@dataclass(frozen=True)
class LLMGainProposal:
    kp: float
    ki: float
    diagnosis: str
    rationale: str
    confidence: float = 0.5


class GainProposalProvider(Protocol):
    name: str

    def propose(self, context: dict[str, object]) -> LLMGainProposal: ...


_PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "kp": {"type": "number"},
        "ki": {"type": "number"},
        "diagnosis": {"type": "string"},
        "rationale": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["kp", "ki", "diagnosis", "rationale", "confidence"],
    "additionalProperties": False,
}


def _parse_proposal(text: str) -> LLMGainProposal:
    data = json.loads(text)
    return LLMGainProposal(
        kp=float(data["kp"]), ki=float(data["ki"]), diagnosis=str(data["diagnosis"]),
        rationale=str(data["rationale"]), confidence=float(data["confidence"]),
    )


class OpenAIResponsesPIDProposer:
    """OpenAI Responses API adapter; credentials are read only at call time."""

    def __init__(self, model: str, *, base_url: str = "https://api.openai.com/v1", timeout_s: float = 60.0) -> None:
        if not model:
            raise ValueError("an explicit OpenAI model name is required")
        self.model, self.base_url, self.timeout_s = model, base_url.rstrip("/"), timeout_s
        self.name = f"OpenAI Responses/{model}"

    def propose(self, context: dict[str, object]) -> LLMGainProposal:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        payload = {
            "model": self.model,
            "store": False,
            "instructions": (
                "You are an offline HVAC PI tuning supervisor. Diagnose metrics and propose one conservative Kp/Ki pair. "
                "Never propose actuator commands. Stay within supplied bounds and prefer changes below 25%. "
                "A deterministic simulator and safety gate, not you, has final authority."
            ),
            "input": json.dumps(context, ensure_ascii=False),
            "text": {"format": {"type": "json_schema", "name": "pi_gain_proposal", "strict": True, "schema": _PROPOSAL_SCHEMA}},
            "max_output_tokens": 500,
        }
        req = request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_s) as response:  # noqa: S310 - configured API endpoint
            body = json.loads(response.read().decode("utf-8"))
        text = body.get("output_text", "")
        if not text:
            for item in body.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text += content.get("text", "")
        if not text:
            raise RuntimeError("Responses API returned no output_text")
        return _parse_proposal(text)


class OllamaPIDProposer:
    def __init__(self, model: str, *, base_url: str = "http://localhost:11434", timeout_s: float = 120.0) -> None:
        if not model:
            raise ValueError("an explicit Ollama model name is required")
        self.model, self.base_url, self.timeout_s = model, base_url.rstrip("/"), timeout_s
        self.name = f"Ollama/{model}"

    def propose(self, context: dict[str, object]) -> LLMGainProposal:
        payload = {
            "model": self.model,
            "stream": False,
            "format": _PROPOSAL_SCHEMA,
            "messages": [
                {"role": "system", "content": "Act as an offline PI gain proposer. Output only schema-valid JSON; never output actuator commands."},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
        }
        req = request.Request(f"{self.base_url}/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=self.timeout_s) as response:  # noqa: S310 - user-selected local endpoint
            body = json.loads(response.read().decode("utf-8"))
        return _parse_proposal(body["message"]["content"])


class PhysicsInformedHeuristicProposer:
    """Deterministic dry-run provider used when no real LLM is configured."""

    name = "physics heuristic (not an LLM)"

    def propose(self, context: dict[str, object]) -> LLMGainProposal:
        current = context["current_gains"]
        metrics = context["current_metrics"]
        kp, ki = float(current["kp"]), float(current["ki"])
        if float(metrics["max_undershoot_c"]) > 0.45:
            scale_p, scale_i, diagnosis = 0.88, 0.80, "冷过冲偏大，降低比例和积分作用"
        elif float(metrics["settling_time_hour"]) > 2.0:
            scale_p, scale_i, diagnosis = 1.15, 1.10, "响应偏慢，在小步长内增强比例和积分作用"
        else:
            scale_p, scale_i, diagnosis = 1.03, 0.95, "稳定后微调，减轻积分并略增强比例"
        return LLMGainProposal(kp * scale_p, ki * scale_i, diagnosis, "离线流程连通性演示；该结果不能称为大模型实验", 1.0)


class LLMSupervisoryTuner:
    def __init__(
        self,
        provider: GainProposalProvider,
        *,
        bounds: GainBounds | None = None,
        rounds: int = 6,
        max_change_fraction: float = 0.25,
        min_improvement_fraction: float = 0.005,
        config: RiskSafetyConfig | None = None,
    ) -> None:
        self.provider, self.bounds, self.rounds = provider, bounds or GainBounds(), int(rounds)
        self.max_change_fraction, self.min_improvement_fraction = max_change_fraction, min_improvement_fraction
        self.config = config or RiskSafetyConfig()

    def tune(self, scenarios: list[Scenario], baseline_gains: tuple[float, float], *, seed: int = 0) -> SupervisedTuneResult:
        if not scenarios:
            raise ValueError("at least one scenario is required")
        current = evaluate_candidate(scenarios, baseline_gains, seed=seed, config=self.config)
        baseline = current
        history: list[dict[str, object]] = []
        for round_index in range(self.rounds):
            context: dict[str, object] = {
                "round": round_index + 1,
                "architecture": "offline proposal -> bounds/trust region -> repeated simulator -> safety gate",
                "current_gains": {"kp": current.kp, "ki": current.ki},
                "current_metrics": current.metrics_mean,
                "current_risk_objective": current.risk_objective,
                "bounds": {"kp": self.bounds.kp, "ki": self.bounds.ki},
                "maximum_fractional_change": self.max_change_fraction,
                "safety_limits": self.config.__dict__,
                "prior_decisions": history[-3:],
            }
            proposal = self.provider.propose(context)
            low = np.asarray([current.kp, current.ki]) * (1.0 - self.max_change_fraction)
            high = np.asarray([current.kp, current.ki]) * (1.0 + self.max_change_fraction)
            hard_low = np.asarray([self.bounds.kp[0], self.bounds.ki[0]])
            hard_high = np.asarray([self.bounds.kp[1], self.bounds.ki[1]])
            clamped = np.clip(np.asarray([proposal.kp, proposal.ki]), np.maximum(low, hard_low), np.minimum(high, hard_high))
            candidate = evaluate_candidate(scenarios, (float(clamped[0]), float(clamped[1])), seed=seed + (round_index + 1) * 20_011, config=self.config)
            candidate = _relative_safety(candidate, baseline.risk_objective, self.config)
            improvement = (current.risk_objective - candidate.risk_objective) / max(abs(current.risk_objective), 1e-9)
            accepted = candidate.safe and improvement >= self.min_improvement_fraction
            reason = "accepted: safe and risk objective improved" if accepted else ("rejected: safety constraint" if not candidate.safe else "rejected: insufficient risk-adjusted improvement")
            history.append(
                {
                    "round": round_index + 1,
                    "provider": self.provider.name,
                    "raw_kp": proposal.kp,
                    "raw_ki": proposal.ki,
                    "candidate_kp": candidate.kp,
                    "candidate_ki": candidate.ki,
                    "diagnosis": proposal.diagnosis,
                    "rationale": proposal.rationale,
                    "confidence": proposal.confidence,
                    "risk_objective": candidate.risk_objective,
                    "safety_margin": candidate.safety_margin,
                    "safe": int(candidate.safe),
                    "improvement_fraction": improvement,
                    "accepted": int(accepted),
                    "decision": reason,
                }
            )
            if accepted:
                current = candidate
        # Final fallback rule: the supervisor may not degrade the qualified IMC baseline.
        qualified = current.safe and current.risk_objective <= baseline.risk_objective * self.config.validation_tolerance
        deployed = current if qualified else baseline
        return SupervisedTuneResult(
            deployed.kp, deployed.ki, deployed.risk_objective,
            accepted=qualified and (deployed.kp, deployed.ki) != (baseline.kp, baseline.ki),
            fallback_used=not qualified,
            evaluations=1 + len(history), history=tuple(history),
        )
