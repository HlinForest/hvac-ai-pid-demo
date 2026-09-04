from __future__ import annotations

from dataclasses import replace
import json
from urllib import request

from hvac_pid.advanced_tuning import (
    AgentAction,
    LLMAgentAutoTuner,
    LLMGainProposal,
    LLMSupervisoryTuner,
    OllamaPIDAgentPolicy,
    OllamaPIDProposer,
    OpenAICompatibleChatAgentPolicy,
    OpenAICompatibleChatProposer,
    OpenAIResponsesPIDAgentPolicy,
    OpenAIResponsesPIDProposer,
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


class _FakeResponse:
    def __init__(self, body: dict) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _patch_urlopen(monkeypatch, body: dict) -> list[request.Request]:
    captured: list[request.Request] = []

    def fake_urlopen(req: request.Request, timeout: float) -> _FakeResponse:  # noqa: ARG001
        captured.append(req)
        return _FakeResponse(body)

    monkeypatch.setattr(request, "urlopen", fake_urlopen)
    return captured


def test_openai_compatible_chat_proposer_parses_json_content(monkeypatch) -> None:
    content = json.dumps(
        {"kp": 2.4, "ki": 0.18, "diagnosis": "冷过冲偏大", "rationale": "降低积分作用", "confidence": 0.7}
    )
    captured = _patch_urlopen(
        monkeypatch, {"choices": [{"message": {"content": content}}]}
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    proposer = OpenAICompatibleChatProposer("qwen-plus")
    proposal = proposer.propose({"current_gains": {"kp": 2.5, "ki": 0.2}})
    assert (proposal.kp, proposal.ki) == (2.4, 0.18)
    assert proposal.diagnosis == "冷过冲偏大"
    assert proposer.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert captured[0].full_url.endswith("/chat/completions")
    assert captured[0].get_header("Authorization") == "Bearer sk-test"


def test_openai_compatible_chat_agent_policy_parses_tool_call(monkeypatch) -> None:
    body = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "evaluate_candidate",
                                "arguments": json.dumps({"kp": 2.35, "ki": 0.18, "reason": "降低过冷风险"}),
                            }
                        }
                    ]
                }
            }
        ]
    }
    captured = _patch_urlopen(monkeypatch, body)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    policy = OpenAICompatibleChatAgentPolicy("qwen-plus")
    action = policy.decide({"current_gains": {"kp": 2.5, "ki": 0.2}})
    assert action.tool == "evaluate_candidate"
    assert action.arguments == {"kp": 2.35, "ki": 0.18, "reason": "降低过冷风险"}
    assert action.reason == "降低过冷风险"
    payload = json.loads(captured[0].data.decode("utf-8"))
    assert [tool["function"]["name"] for tool in payload["tools"]] == [
        "inspect_history", "evaluate_candidate", "finish"
    ]


def test_openai_compatible_providers_require_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    proposer = OpenAICompatibleChatProposer("qwen-plus")
    policy = OpenAICompatibleChatAgentPolicy("qwen-plus")
    for provider_call in (lambda: proposer.propose({}), lambda: policy.decide({})):
        try:
            provider_call()
        except RuntimeError as exc:
            assert "DASHSCOPE_API_KEY" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for missing API key")


def test_openai_compatible_overrides_apply(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-other")
    captured = _patch_urlopen(
        monkeypatch, {"choices": [{"message": {"content": json.dumps(
            {"kp": 1.0, "ki": 0.1, "diagnosis": "d", "rationale": "r", "confidence": 0.5}
        )}}]}
    )
    proposer = OpenAICompatibleChatProposer(
        "deepseek-chat", base_url="https://api.deepseek.com/v1", api_key_env="DEEPSEEK_API_KEY"
    )
    proposer.propose({})
    assert proposer.base_url == "https://api.deepseek.com/v1"
    assert captured[0].full_url == "https://api.deepseek.com/v1/chat/completions"
    assert captured[0].get_header("Authorization") == "Bearer sk-other"

def test_openai_responses_agent_policy_parses_function_call(monkeypatch) -> None:
    body = {
        "output": [
            {"type": "function_call", "name": "finish", "arguments": json.dumps({"reason": "预算已尽"})}
        ]
    }
    captured = _patch_urlopen(monkeypatch, body)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    policy = OpenAIResponsesPIDAgentPolicy("gpt-5-mini")
    action = policy.decide({"current_gains": {"kp": 2.5, "ki": 0.2}})
    assert action.tool == "finish"
    assert action.reason == "预算已尽"
    assert captured[0].full_url.endswith("/responses")
    assert captured[0].get_header("Authorization") == "Bearer sk-test"
    payload = json.loads(captured[0].data.decode("utf-8"))
    assert payload["tool_choice"] == "required"
    assert [tool["name"] for tool in payload["tools"]] == ["inspect_history", "evaluate_candidate", "finish"]


def test_openai_responses_proposer_parses_output_text(monkeypatch) -> None:
    content = json.dumps({"kp": 2.4, "ki": 0.18, "diagnosis": "d", "rationale": "r", "confidence": 0.6})
    captured = _patch_urlopen(monkeypatch, {"output_text": content})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    proposer = OpenAIResponsesPIDProposer("gpt-5-mini")
    proposal = proposer.propose({"current_gains": {"kp": 2.5, "ki": 0.2}})
    assert (proposal.kp, proposal.ki) == (2.4, 0.18)
    assert captured[0].full_url.endswith("/responses")


def test_ollama_agent_policy_parses_tool_call(monkeypatch) -> None:
    body = {"message": {"tool_calls": [{"function": {"name": "inspect_history", "arguments": {}}}]}}
    captured = _patch_urlopen(monkeypatch, body)
    policy = OllamaPIDAgentPolicy("qwen3:0.6b")
    action = policy.decide({"current_gains": {"kp": 2.5, "ki": 0.2}})
    assert action.tool == "inspect_history"
    assert captured[0].full_url.endswith("/api/chat")
    payload = json.loads(captured[0].data.decode("utf-8"))
    assert payload["model"] == "qwen3:0.6b"
    assert payload["stream"] is False


def test_ollama_proposer_parses_message_content(monkeypatch) -> None:
    content = json.dumps({"kp": 2.4, "ki": 0.18, "diagnosis": "d", "rationale": "r", "confidence": 0.6})
    captured = _patch_urlopen(monkeypatch, {"message": {"content": content}})
    proposer = OllamaPIDProposer("qwen3:0.6b")
    proposal = proposer.propose({"current_gains": {"kp": 2.5, "ki": 0.2}})
    assert (proposal.kp, proposal.ki) == (2.4, 0.18)
    assert captured[0].full_url.endswith("/api/chat")


def test_openai_responses_providers_require_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    proposer = OpenAIResponsesPIDProposer("gpt-5-mini")
    policy = OpenAIResponsesPIDAgentPolicy("gpt-5-mini")
    for provider_call in (lambda: proposer.propose({}), lambda: policy.decide({})):
        try:
            provider_call()
        except RuntimeError as exc:
            assert "OPENAI_API_KEY" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for missing API key")

