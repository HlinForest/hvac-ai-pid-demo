from __future__ import annotations

"""Risk-aware safe BO and simulator-gated LLM/Agent PI tuning.

The LLM Agent is deliberately outside the real-time loop.  It may inspect the
experiment history, request a Kp/Ki simulation, or stop.  Deterministic bounds,
trust-region limits, repeated simulation and safety constraints decide whether
a request is accepted and which parameters are deployed.
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


@dataclass(frozen=True)
class AgentAction:
    """One tool request selected by an LLM tuning agent."""

    tool: str
    arguments: dict[str, object]
    reason: str = ""


@dataclass(frozen=True)
class AgentTuneResult:
    kp: float
    ki: float
    score: float
    accepted: bool
    fallback_used: bool
    evaluations: int
    stop_reason: str
    trace: tuple[dict[str, object], ...]


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
                "Never propose actuator commands. Stay within supplied bounds and keep each change at or below 10%. "
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


class ReplayPIDProposer:
    """Deterministic recorded-response provider for an offline LLM demo.

    This provider is intentionally labelled as replay data.  It makes the full
    schema/bounds/simulation/safety-gate path runnable without network access or
    credentials, but its result must never be reported as a live model call.
    """

    name = "recorded LLM replay (not a live model call)"

    def __init__(self) -> None:
        self._round = 0

    def propose(self, context: dict[str, object]) -> LLMGainProposal:
        current = context["current_gains"]
        kp, ki = float(current["kp"]), float(current["ki"])
        scripts = (
            (0.94, 0.90, "启动阶段冷过冲风险偏高", "记录中的模型建议先减小积分，再小幅降低比例作用"),
            (0.98, 0.94, "温度已接近目标但容量变化仍偏大", "记录中的模型建议继续保持保守积分，避免穿越设定值"),
            (1.02, 0.98, "跟踪稳定后可轻微恢复比例作用", "记录中的模型建议只做小于安全信赖域的微调"),
        )
        scale_p, scale_i, diagnosis, rationale = scripts[min(self._round, len(scripts) - 1)]
        self._round += 1
        return LLMGainProposal(kp * scale_p, ki * scale_i, diagnosis, rationale, 0.76)


class LLMSupervisoryTuner:
    def __init__(
        self,
        provider: GainProposalProvider,
        *,
        bounds: GainBounds | None = None,
        rounds: int = 6,
        max_change_fraction: float = 0.10,
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


class AgentPolicy(Protocol):
    """Policy that chooses the next safe host-side tuning tool."""

    name: str

    def decide(self, state: dict[str, object]) -> AgentAction: ...


_AGENT_TOOLS = (
    {
        "type": "function",
        "name": "inspect_history",
        "description": "Inspect current gains, metrics, trial budget and prior simulator decisions without changing control parameters.",
        "strict": True,
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "evaluate_candidate",
        "description": "Request one Kp/Ki experiment. The host clamps the pair, runs repeated simulations, and independently accepts or rejects it.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "kp": {"type": "number"},
                "ki": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["kp", "ki", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "finish",
        "description": "Stop experimenting and deploy the best host-accepted gains, or IMC when no candidate was accepted.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
)


def _agent_action_from_function_call(item: dict[str, object]) -> AgentAction:
    name = str(item.get("name", ""))
    raw_arguments = item.get("arguments", {})
    if isinstance(raw_arguments, str):
        arguments = json.loads(raw_arguments)
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        raise ValueError("agent tool arguments must be a JSON object")
    if name not in {"inspect_history", "evaluate_candidate", "finish"}:
        raise ValueError(f"unsupported agent tool: {name}")
    reason = str(arguments.get("reason", ""))
    return AgentAction(name, dict(arguments), reason)


class ReplayPIDAgentPolicy:
    """Deterministic recording of a tool-using agent for offline demos."""

    name = "recorded tool-using LLM agent replay (not a live model call)"

    def __init__(self) -> None:
        self._step = 0

    def decide(self, state: dict[str, object]) -> AgentAction:
        current = state["current_gains"]
        kp, ki = float(current["kp"]), float(current["ki"])
        scripts = (
            AgentAction("inspect_history", {}, "先读取基线指标、剩余试验预算和安全限制"),
            AgentAction(
                "evaluate_candidate",
                {"kp": kp * 0.94, "ki": ki * 0.90, "reason": "先减弱积分以降低启动阶段过冷风险"},
                "先减弱积分以降低启动阶段过冷风险",
            ),
            AgentAction(
                "evaluate_candidate",
                {"kp": kp * 0.98, "ki": ki * 0.94, "reason": "根据上一轮结果继续小步减少控制动作"},
                "根据上一轮结果继续小步减少控制动作",
            ),
            AgentAction(
                "evaluate_candidate",
                {"kp": kp * 1.02, "ki": ki * 0.98, "reason": "在安全候选附近轻微恢复比例作用"},
                "在安全候选附近轻微恢复比例作用",
            ),
            AgentAction("finish", {"reason": "试验预算已足够，停止并部署安全门接受的最优参数"}, "试验完成"),
        )
        action = scripts[min(self._step, len(scripts) - 1)]
        self._step += 1
        return action


class OpenAIResponsesPIDAgentPolicy:
    """Responses API function-calling policy; host code executes every tool."""

    def __init__(self, model: str, *, base_url: str = "https://api.openai.com/v1", timeout_s: float = 60.0) -> None:
        if not model:
            raise ValueError("an explicit OpenAI model name is required")
        self.model, self.base_url, self.timeout_s = model, base_url.rstrip("/"), timeout_s
        self.name = f"OpenAI Responses tool agent/{model}"

    def decide(self, state: dict[str, object]) -> AgentAction:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        payload = {
            "model": self.model,
            "store": False,
            "instructions": (
                "You are an offline HVAC PI auto-tuning agent. Choose exactly one provided function tool. "
                "Use inspect_history when evidence is insufficient, evaluate_candidate for one conservative experiment, "
                "and finish when the remaining budget is not worth the risk. Never output actuator commands. "
                "The deterministic host clamps gains and has sole authority to accept, reject, deploy, or fall back."
            ),
            "input": json.dumps(state, ensure_ascii=False),
            "tools": list(_AGENT_TOOLS),
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_output_tokens": 500,
        }
        req = request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_s) as response:  # noqa: S310 - configured official endpoint
            body = json.loads(response.read().decode("utf-8"))
        for item in body.get("output", []):
            if item.get("type") == "function_call":
                return _agent_action_from_function_call(item)
        raise RuntimeError("Responses API agent returned no function_call")


class OllamaPIDAgentPolicy:
    """Local Ollama function-calling policy, intended for Qwen3-class models."""

    def __init__(self, model: str, *, base_url: str = "http://localhost:11434", timeout_s: float = 120.0) -> None:
        if not model:
            raise ValueError("an explicit Ollama model name is required")
        self.model, self.base_url, self.timeout_s = model, base_url.rstrip("/"), timeout_s
        self.name = f"Ollama tool agent/{model}"

    def decide(self, state: dict[str, object]) -> AgentAction:
        ollama_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in _AGENT_TOOLS
        ]
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Act as an offline HVAC PI tuning agent. Call one tool only. Never generate actuator commands. "
                        "The host safety gate has final authority."
                    ),
                },
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
            ],
            "tools": ollama_tools,
        }
        req = request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_s) as response:  # noqa: S310 - user-selected local endpoint
            body = json.loads(response.read().decode("utf-8"))
        calls = body.get("message", {}).get("tool_calls", [])
        if not calls:
            raise RuntimeError("Ollama agent returned no tool call")
        return _agent_action_from_function_call(calls[0].get("function", {}))


class LLMAgentAutoTuner:
    """Bounded tool-using agent loop for offline PI auto-tuning.

    The policy controls experiment selection and stopping only. It cannot
    bypass hard gain bounds, the ±trust region, repeated simulation, the
    deterministic acceptance rule, or final IMC fallback.
    """

    def __init__(
        self,
        policy: AgentPolicy,
        *,
        bounds: GainBounds | None = None,
        max_steps: int = 8,
        max_trials: int = 5,
        max_change_fraction: float = 0.10,
        min_improvement_fraction: float = 0.005,
        config: RiskSafetyConfig | None = None,
    ) -> None:
        self.policy = policy
        self.bounds = bounds or GainBounds()
        self.max_steps = max(1, int(max_steps))
        self.max_trials = max(1, int(max_trials))
        self.max_change_fraction = float(max_change_fraction)
        self.min_improvement_fraction = float(min_improvement_fraction)
        self.config = config or RiskSafetyConfig()

    def tune(self, scenarios: list[Scenario], baseline_gains: tuple[float, float], *, seed: int = 0) -> AgentTuneResult:
        if not scenarios:
            raise ValueError("at least one scenario is required")
        baseline = evaluate_candidate(scenarios, baseline_gains, seed=seed, config=self.config)
        current = baseline
        trace: list[dict[str, object]] = []
        trials = 0
        stop_reason = "maximum agent steps reached"
        provider_error = False

        for step in range(1, self.max_steps + 1):
            state: dict[str, object] = {
                "step": step,
                "role": "offline experiment-selection agent; no real-time actuator authority",
                "available_tools": [tool["name"] for tool in _AGENT_TOOLS],
                "current_gains": {"kp": current.kp, "ki": current.ki},
                "current_metrics": current.metrics_mean,
                "current_risk_objective": current.risk_objective,
                "baseline_risk_objective": baseline.risk_objective,
                "remaining_trials": self.max_trials - trials,
                "hard_bounds": {"kp": self.bounds.kp, "ki": self.bounds.ki},
                "maximum_fractional_change": self.max_change_fraction,
                "safety_limits": self.config.__dict__,
                "recent_tool_results": trace[-4:],
            }
            try:
                action = self.policy.decide(state)
            except Exception as exc:
                provider_error = True
                stop_reason = f"agent provider error: {type(exc).__name__}: {exc}"
                trace.append({"step": step, "provider": self.policy.name, "tool": "provider_error", "decision": stop_reason})
                break

            if action.tool == "inspect_history":
                trace.append(
                    {
                        "step": step,
                        "provider": self.policy.name,
                        "tool": action.tool,
                        "reason": action.reason,
                        "current_kp": current.kp,
                        "current_ki": current.ki,
                        "risk_objective": current.risk_objective,
                        "safe": int(current.safe),
                        "remaining_trials": self.max_trials - trials,
                        "decision": "inspection returned; no parameters changed",
                    }
                )
                continue

            if action.tool == "finish":
                stop_reason = action.reason or "agent requested finish"
                trace.append(
                    {
                        "step": step,
                        "provider": self.policy.name,
                        "tool": action.tool,
                        "reason": action.reason,
                        "current_kp": current.kp,
                        "current_ki": current.ki,
                        "risk_objective": current.risk_objective,
                        "safe": int(current.safe),
                        "remaining_trials": self.max_trials - trials,
                        "decision": "agent stopped; host will run final deployment qualification",
                    }
                )
                break

            if action.tool != "evaluate_candidate":
                provider_error = True
                stop_reason = f"invalid agent tool: {action.tool}"
                trace.append({"step": step, "provider": self.policy.name, "tool": action.tool, "decision": stop_reason})
                break
            if trials >= self.max_trials:
                stop_reason = "hard experiment budget exhausted"
                trace.append({"step": step, "provider": self.policy.name, "tool": action.tool, "decision": stop_reason})
                break

            try:
                raw = np.asarray([float(action.arguments["kp"]), float(action.arguments["ki"])], dtype=float)
                if not np.all(np.isfinite(raw)):
                    raise ValueError("non-finite gains")
            except (KeyError, TypeError, ValueError) as exc:
                provider_error = True
                stop_reason = f"invalid agent candidate: {exc}"
                trace.append({"step": step, "provider": self.policy.name, "tool": action.tool, "decision": stop_reason})
                break

            low = np.asarray([current.kp, current.ki]) * (1.0 - self.max_change_fraction)
            high = np.asarray([current.kp, current.ki]) * (1.0 + self.max_change_fraction)
            hard_low = np.asarray([self.bounds.kp[0], self.bounds.ki[0]])
            hard_high = np.asarray([self.bounds.kp[1], self.bounds.ki[1]])
            limited = np.clip(raw, np.maximum(low, hard_low), np.minimum(high, hard_high))
            trials += 1
            candidate = evaluate_candidate(
                scenarios,
                (float(limited[0]), float(limited[1])),
                seed=seed + trials * 20_011,
                config=self.config,
            )
            candidate = _relative_safety(candidate, baseline.risk_objective, self.config)
            improvement = (current.risk_objective - candidate.risk_objective) / max(abs(current.risk_objective), 1e-9)
            accepted = bool(candidate.safe and improvement >= self.min_improvement_fraction)
            if accepted:
                current = candidate
            decision = (
                "accepted by deterministic host gate"
                if accepted
                else ("rejected: safety constraint" if not candidate.safe else "rejected: insufficient risk-adjusted improvement")
            )
            trace.append(
                {
                    "step": step,
                    "provider": self.policy.name,
                    "tool": action.tool,
                    "reason": action.reason or str(action.arguments.get("reason", "")),
                    "raw_kp": float(raw[0]),
                    "raw_ki": float(raw[1]),
                    "limited_kp": candidate.kp,
                    "limited_ki": candidate.ki,
                    "risk_objective": candidate.risk_objective,
                    "safety_margin": candidate.safety_margin,
                    "safe": int(candidate.safe),
                    "improvement_fraction": improvement,
                    "accepted": int(accepted),
                    "remaining_trials": self.max_trials - trials,
                    "decision": decision,
                }
            )
        qualified = current.safe and current.risk_objective <= baseline.risk_objective * self.config.validation_tolerance
        deployed = current if qualified else baseline
        changed = (deployed.kp, deployed.ki) != (baseline.kp, baseline.ki)
        return AgentTuneResult(
            deployed.kp,
            deployed.ki,
            deployed.risk_objective,
            accepted=bool(qualified and changed),
            fallback_used=bool(provider_error or not qualified or not changed),
            evaluations=1 + trials,
            stop_reason=stop_reason,
            trace=tuple(trace),
        )
