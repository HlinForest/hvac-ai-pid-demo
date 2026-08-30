from __future__ import annotations

from dataclasses import replace

from hvac_pid.advanced_tuning import (
    AgentAction,
    LLMAgentAutoTuner,
    LLMGainProposal,
    LLMSupervisoryTuner,
    ReplayPIDAgentPolicy,
    RiskAwareSafeBOTuner,
    RiskSafetyConfig,
    evaluate_candidate,
)
from hvac_pid.config import sample_adaptive_scenarios
from hvac_pid.controllers import identify_fopdt, imc_pi
from hvac_pid.tuning import GainBounds


class ExtremeProvider:
    name = "test extreme provider"

    def propose(self, context: dict[str, object]) -> LLMGainProposal:
        return LLMGainProposal(999.0, 999.0, "故意越界", "验证硬边界和信赖域", 0.2)


class BrokenAgentPolicy:
    name = "broken test agent"

    def decide(self, state: dict[str, object]) -> AgentAction:
        raise RuntimeError("simulated provider outage")


def _small_scenarios():
    return [replace(item, duration_hours=1.5) for item in sample_adaptive_scenarios(3, seed=123, duration_hours=1.5)]


def test_candidate_evaluation_records_noise_risk_and_safety() -> None:
    scenarios = _small_scenarios()
    gains = imc_pi(identify_fopdt(scenarios[0]))
    result = evaluate_candidate(
        scenarios, gains, seed=4,
        config=RiskSafetyConfig(repeats=2, max_undershoot_c=10.0),
    )
    assert result.mean_objective > 0.0
    assert result.objective_std >= 0.0
    assert result.risk_objective >= result.mean_objective
    assert "running_slew_violation_count" in result.metrics_mean


def test_risk_aware_safe_bo_returns_bounded_audited_gains() -> None:
    scenarios = _small_scenarios()
    baseline = imc_pi(identify_fopdt(scenarios[0]))
    bounds = GainBounds()
    result = RiskAwareSafeBOTuner(
        bounds=bounds, iterations=1, candidates=48,
        config=RiskSafetyConfig(repeats=1, max_undershoot_c=10.0),
    ).tune(scenarios, baseline, seed=9)
    assert bounds.kp[0] <= result.kp <= bounds.kp[1]
    assert bounds.ki[0] <= result.ki <= bounds.ki[1]
    assert result.evaluations >= 4
    assert any("safety_margin" in row for row in result.history)
    assert sum(int(row["selected"]) for row in result.history) == 1


def test_llm_supervisor_clamps_extreme_proposal_and_keeps_audit() -> None:
    scenarios = _small_scenarios()
    baseline = imc_pi(identify_fopdt(scenarios[0]))
    result = LLMSupervisoryTuner(
        ExtremeProvider(), rounds=1, max_change_fraction=0.25,
        config=RiskSafetyConfig(repeats=1, max_undershoot_c=10.0),
    ).tune(scenarios, baseline, seed=13)
    row = result.history[0]
    assert float(row["raw_kp"]) == 999.0
    assert float(row["raw_ki"]) == 999.0
    assert float(row["candidate_kp"]) <= baseline[0] * 1.25 + 1e-12
    assert float(row["candidate_ki"]) <= baseline[1] * 1.25 + 1e-12
    assert "decision" in row
    assert result.evaluations == 2


def test_llm_agent_uses_bounded_tools_and_persists_host_decisions() -> None:
    scenarios = _small_scenarios()
    baseline = imc_pi(identify_fopdt(scenarios[0]))
    result = LLMAgentAutoTuner(
        ReplayPIDAgentPolicy(), max_steps=6, max_trials=3,
        max_change_fraction=0.10,
        config=RiskSafetyConfig(repeats=1, max_undershoot_c=10.0),
    ).tune(scenarios, baseline, seed=17)
    assert [row["tool"] for row in result.trace] == [
        "inspect_history", "evaluate_candidate", "evaluate_candidate",
        "evaluate_candidate", "finish",
    ]
    assert result.evaluations == 4
    assert sum(row["tool"] == "evaluate_candidate" for row in result.trace) == 3
    for row in result.trace:
        assert row["tool"] in {"inspect_history", "evaluate_candidate", "finish"}
        assert "decision" in row
        if row["tool"] == "evaluate_candidate":
            assert GainBounds().kp[0] <= float(row["limited_kp"]) <= GainBounds().kp[1]
            assert GainBounds().ki[0] <= float(row["limited_ki"]) <= GainBounds().ki[1]


def test_llm_agent_provider_failure_returns_unchanged_imc_fallback() -> None:
    scenarios = _small_scenarios()
    baseline = imc_pi(identify_fopdt(scenarios[0]))
    result = LLMAgentAutoTuner(
        BrokenAgentPolicy(), max_steps=3, max_trials=2,
        config=RiskSafetyConfig(repeats=1, max_undershoot_c=10.0),
    ).tune(scenarios, baseline, seed=23)
    assert result.fallback_used
    assert not result.accepted
    assert (result.kp, result.ki) == baseline
    assert result.trace[0]["tool"] == "provider_error"
