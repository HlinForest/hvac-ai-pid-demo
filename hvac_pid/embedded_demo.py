from __future__ import annotations

"""Seven-algorithm temperature-control demo and self-contained HTML report."""

from dataclasses import dataclass, replace
import csv
import html
import json
import os
import warnings
from pathlib import Path
from typing import Callable
import zlib

import numpy as np

from .advanced_tuning import (
    AgentPolicy,
    LLMAgentAutoTuner,
    OllamaPIDAgentPolicy,
    OpenAICompatibleChatAgentPolicy,
    OpenAIResponsesPIDAgentPolicy,
    ReplayPIDAgentPolicy,
    RiskSafetyConfig,
)
from .ai_controllers import FNNGainController, IncrementalRLController
from .config import Scenario, sample_adaptive_scenarios
from .controllers import PIController, identify_fopdt, ziegler_nichols_pi
from .metrics import calculate_metrics
from .policy_bundle import PolicyBundle
from .safety import MAX_FRACTIONAL_GAIN_CHANGE
from .simulator import SimulationResult, simulate
from .timebase import DEMO_SUPERVISORY_PERIOD_S, MCU_AI_PERIOD_MS, MCU_FAST_PI_PERIOD_MS


ALGORITHM_ORDER = ("zn", "imc", "bo", "safe-bo", "fnn", "rl", "llm")
DISPLAY_NAMES = {
    "zn": "Z-N 反应曲线法",
    "imc": "IMC 鲁棒 PI",
    "bo": "普通贝叶斯优化固定 PI",
    "safe-bo": "风险感知安全 BO",
    "fnn": "FNN 规则自整定",
    "rl": "RL 冻结策略自整定",
    "llm": "LLM Agent 自动调参",
}
DESCRIPTIONS = {
    "zn": "上位机辨识后按经验公式计算固定增益，ESP32 只执行安全 PI。",
    "imc": "上位机依据 FOPDT 模型选择鲁棒闭环速度，作为所有智能方法的安全回退。",
    "bo": "上位机离线搜索固定 Kp/Ki，部署后不再在线探索。",
    "safe-bo": "在性能均值之外惩罚噪声方差，并用安全代理限制候选。",
    "fnn": "演示监督周期120 s用误差和误差变化率插值四条规则；MCU墙钟AI周期2 s/快环100 ms；未通过验收时只做影子计算。",
    "rl": "训练在虚拟对象上完成，演示监督周期120 s查冻结策略，不进行随机探索；MCU墙钟AI周期2 s。",
    "llm": "大模型 Agent 自主选择查看历史、运行候选仿真或停止；安全门拥有最终决定权。",
}


@dataclass(frozen=True)
class DemoTrace:
    algorithm: str
    display_name: str
    description: str
    source: str
    result: SimulationResult
    rows: tuple[dict[str, object], ...]
    summary: dict[str, object]
    deployment_accepted: bool
    forced_fallback: bool
    agent_trace: tuple[dict[str, object], ...] = ()


def demo_scenario(*, setpoint_c: float = 24.0, door_load_w: float = 3200.0) -> Scenario:
    """A labelled five-hour virtual cabinet-cooling demonstration.

    The HTML maps the five simulated hours to 90 wall-clock seconds.  That time
    acceleration is presentation-only and is always shown beside the plot.
    """

    return Scenario(
        duration_hours=5.0,
        setpoint_c=float(setpoint_c),
        initial_zone_c=30.0,
        outdoor_c=34.0,
        outdoor_amplitude_c=1.0,
        internal_load_w=650.0,
        c_zone_j_per_k=1.45e6,
        c_wall_j_per_k=8.0e6,
        r_out_zone_k_per_w=0.012,
        r_zone_wall_k_per_w=0.006,
        r_out_wall_k_per_w=0.020,
        cooling_capacity_w=8200.0,
        actuator_delay_minutes=4.0,
        actuator_tau_minutes=3.0,
        sensor_noise_std_c=0.04,
        door_open_hour=3.15,
        door_open_duration_minutes=15.0,
        door_open_load_w=float(door_load_w),
    )


def _commissioning_training_scenarios(scenario: Scenario, *, count: int = 2, seed: int = 0) -> list[Scenario]:
    """Short variants of the visible commissioning scenario.

    The LLM agent used to tune exclusively on the fixed
    ``sample_adaptive_scenarios`` curriculum, i.e. it optimised a different
    distribution from the one it was later deployed on ("blind tuning").
    Mixing in shortened, jittered variants of the commissioning scenario lets
    the safety gate judge candidates on the deployment distribution too.
    """

    rng = np.random.default_rng(seed)
    variants: list[Scenario] = []
    for _ in range(max(0, count)):
        door_hour = scenario.door_open_hour
        if door_hour is not None:
            door_hour = float(rng.uniform(1.6, 2.3))
        variants.append(
            replace(
                scenario,
                duration_hours=3.0,
                initial_zone_c=float(scenario.initial_zone_c + rng.uniform(-1.5, 1.5)),
                outdoor_c=float(scenario.outdoor_c + rng.uniform(-1.0, 1.0)),
                door_open_hour=door_hour,
                door_open_load_w=float(scenario.door_open_load_w * rng.uniform(0.85, 1.15)),
            )
        )
    return variants


def _last_csv(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def _accepted(artifact_dir: Path, name: str) -> bool:
    row = _last_csv(artifact_dir / name)
    return bool(row and float(row.get("deployment_accepted", 0.0)) > 0.5)


def _resolve_bundle(
    artifact_dir: Path, project_root: Path
) -> PolicyBundle:
    """Single entry for all policy loads; no caller may bypass it.

    BO / Safe-BO / FNN / RL all resolve inside ``artifact_dir`` first.
    Legacy archive reads (if any) are flagged in ``bundle.provenance``.
    """
    return PolicyBundle.load(artifact_dir, project_root=project_root)


def _status_rows(
    scenario: Scenario,
    result: SimulationResult,
    *,
    algorithm: str,
    forced_fallback: bool,
    candidate_result: SimulationResult | None = None,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    band = 0.5
    stable_window_minutes = 30.0
    required = max(1, int(round(stable_window_minutes / scenario.dt_minutes)))
    streak = 0
    stable_first = float("nan")
    event_start = float(scenario.door_open_hour or 1e9) * 60.0
    event_end = event_start + float(scenario.door_open_duration_minutes)
    recovered_first = float("nan")
    rows: list[dict[str, object]] = []
    for index, minute in enumerate(result.minute):
        error = float(result.zone_c[index] - result.setpoint_c[index])
        finite = bool(np.isfinite(result.zone_c[index]) and np.isfinite(result.command[index]))
        in_band = finite and abs(error) <= band
        streak = streak + 1 if in_band else 0
        base_status = "降温中"
        if not finite:
            base_status = "故障"
        elif minute >= event_start and (minute < event_end or streak < required):
            base_status = "扰动恢复中"
        elif in_band and streak < required:
            base_status = "进入目标带"
        elif streak >= required:
            base_status = "已稳定"
            candidate_time = float(minute - (required - 1) * scenario.dt_minutes)
            if not np.isfinite(stable_first) and candidate_time < event_start:
                stable_first = candidate_time
            if minute >= event_end and not np.isfinite(recovered_first):
                recovered_first = candidate_time
        fallback = bool(forced_fallback or result.fallback_active[index])
        status = f"安全回退 · {base_status}" if fallback and base_status != "故障" else base_status
        door_open = bool(event_start <= minute < event_end)
        rows.append(
            {
                "algorithm": algorithm,
                "display_name": DISPLAY_NAMES[algorithm],
                "wall_clock_second": float(minute / result.minute[-1] * 90.0),
                "simulated_minute": float(minute),
                "temperature_c": float(result.zone_c[index]),
                "measurement_c": float(result.measurement_c[index]),
                "setpoint_c": float(result.setpoint_c[index]),
                "comfort_low_c": float(result.setpoint_c[index] - band),
                "comfort_high_c": float(result.setpoint_c[index] + band),
                "requested_command_pct": float(result.requested_command[index] * 100.0),
                "command_pct": float(result.command[index] * 100.0),
                "kp": float(result.kp[index]),
                "ki": float(result.ki[index]),
                "candidate_temperature_c": (
                    float(candidate_result.zone_c[index]) if candidate_result is not None else float(result.zone_c[index])
                ),
                "candidate_kp": (
                    float(candidate_result.kp[index]) if candidate_result is not None else float(result.kp[index])
                ),
                "candidate_ki": (
                    float(candidate_result.ki[index]) if candidate_result is not None else float(result.ki[index])
                ),
                "door_open": int(door_open),
                "in_comfort_band": int(in_band),
                "fallback_active": int(fallback),
                "status": status,
            }
        )
    metrics = calculate_metrics(result)
    candidate_metrics = calculate_metrics(candidate_result) if candidate_result is not None else None
    first_band = next((float(row["simulated_minute"]) for row in rows if row["in_comfort_band"]), float("nan"))
    recovery = recovered_first - event_end if np.isfinite(recovered_first) else float("nan")
    # “回退成功”不等于“温控验收通过”：回退后仍未通过 validation 即记失败。
    fallback_failed = bool(
        forced_fallback and not (metrics["validation_passed"] > 0.5 and np.isfinite(stable_first))
    )
    summary: dict[str, object] = {
        "algorithm": algorithm,
        "display_name": DISPLAY_NAMES[algorithm],
        "kp_initial": float(result.kp[0]),
        "ki_initial": float(result.ki[0]),
        "kp_final": float(result.kp[-1]),
        "ki_final": float(result.ki[-1]),
        "first_in_band_minute": first_band,
        "stable_first_minute": stable_first,
        "door_event_minute": event_start,
        "door_recovery_minutes": recovery,
        "final_temperature_c": float(result.zone_c[-1]),
        "objective": float(metrics["objective"]),
        "candidate_objective": float(candidate_metrics["objective"]) if candidate_metrics is not None else float(metrics["objective"]),
        "max_overheat_c": float(metrics["max_overheat_c"]),
        "max_undershoot_c": float(metrics["max_undershoot_c"]),
        "control_movement": float(metrics["control_movement"]),
        "stable": int(metrics["stable"] > 0.5 and np.isfinite(stable_first)),
        # Split acceptance: demo comfort-band hold vs core numeric boundedness.
        "bounded": int(metrics["bounded"] > 0.5),
        "comfort_held": int(np.isfinite(stable_first) and metrics["bounded"] > 0.5),
        "recovery_ok": int(metrics["disturbance_recovered"] > 0.5),
        "actuator_compliant": int(metrics["actuator_compliant"] > 0.5),
        "validation_passed": int(metrics["validation_passed"] > 0.5 and np.isfinite(stable_first)),
        "fallback_used": int(forced_fallback or np.any(result.fallback_active)),
        "fallback_failed": int(fallback_failed),
    }
    return tuple(rows), summary


def _settles_before_disturbance(result: SimulationResult, scenario: Scenario, *, band_c: float = 0.5, hold_minutes: float = 30.0) -> bool:
    event_minute = float(scenario.door_open_hour or scenario.duration_hours) * 60.0
    before_event = result.minute < event_minute
    within = np.abs(result.zone_c - result.setpoint_c) <= band_c
    required = max(1, int(round(hold_minutes / scenario.dt_minutes)))
    streak = 0
    for valid, in_band in zip(before_event, within, strict=True):
        if not valid:
            break
        streak = streak + 1 if in_band else 0
        if streak >= required:
            return True
    return False


@dataclass(frozen=True)
class _DemoContext:
    """Shared inputs every algorithm preparation may need."""

    root: Path
    artifact_dir: Path
    scenario: Scenario
    model_fopdt: object
    fallback: tuple[float, float]
    bundle: PolicyBundle
    seed: int
    provider: str
    model: str
    base_url: str
    api_key_env: str


@dataclass(frozen=True)
class _PreparedDemo:
    """One algorithm's controller and provenance, before the commissioning gate."""

    controller: PIController
    source: str
    deployment_accepted: bool = True
    forced_fallback: bool = False
    candidate_result: SimulationResult | None = None
    agent_trace: tuple[dict[str, object], ...] = ()
    agent_baseline_gains: tuple[float, float] | None = None
    agent_deployed_gains: tuple[float, float] | None = None


def _prepare_zn(context: _DemoContext) -> _PreparedDemo:
    return _PreparedDemo(
        controller=PIController(*ziegler_nichols_pi(context.model_fopdt)),
        source="FOPDT identification + Z-N formula",
    )


def _prepare_imc(context: _DemoContext) -> _PreparedDemo:
    return _PreparedDemo(
        controller=PIController(*context.fallback),
        source="selected IMC lambda artifact",
    )


def _prepare_bo(context: _DemoContext) -> _PreparedDemo:
    # P0: BO gains come from the same PolicyBundle (artifact_dir-controlled).
    gains = context.bundle.bo_gains
    source = f"offline bounded Bayesian optimization artifact ({context.bundle.provenance.get('bo', '')})"
    return _PreparedDemo(
        controller=PIController(*gains),
        source=source,
    )


def _prepare_safe_bo(context: _DemoContext) -> _PreparedDemo:
    gains = context.bundle.safe_bo_gains
    source = f"RaGoOSE-style risk-aware safe BO artifact ({context.bundle.provenance.get('safe_bo', '')})"
    return _PreparedDemo(
        controller=PIController(*gains),
        source=source,
    )


def _prepare_fnn(context: _DemoContext) -> _PreparedDemo:
    # P0: load BOTH the rule table and the (2,4) context coefficients via
    # the bundle; the shadow candidate also carries its context.
    accepted = bool(context.bundle.fnn_accepted)
    candidate_path = context.artifact_dir / "fnn_rule_table_candidate.npy"
    candidate_context_path = context.artifact_dir / "fnn_context_coefficients_candidate.npy"
    if candidate_path.exists():
        candidate_table = np.load(candidate_path)
    else:
        candidate_table = np.asarray(context.bundle.fnn_rule_table)
    if candidate_context_path.exists():
        candidate_context = np.load(candidate_context_path)
    else:
        candidate_context = np.asarray(context.bundle.fnn_context)
    shadow: SimulationResult | None = None
    if accepted:
        controller: PIController = FNNGainController(
            context.fallback,
            rule_table=np.asarray(context.bundle.fnn_rule_table),
            context_coefficients=np.asarray(context.bundle.fnn_context),
            update_interval_seconds=DEMO_SUPERVISORY_PERIOD_S,
        )
    else:
        controller = PIController(*context.fallback)
        shadow = simulate(
            context.scenario,
            FNNGainController(
                context.fallback,
                rule_table=candidate_table,
                context_coefficients=candidate_context,
                update_interval_seconds=DEMO_SUPERVISORY_PERIOD_S,
            ),
            seed=context.seed,
        )
    return _PreparedDemo(
        controller=controller,
        source="validated FNN table+context" if accepted else "rejected FNN candidate in shadow mode + IMC fallback",
        deployment_accepted=accepted,
        forced_fallback=not accepted,
        candidate_result=shadow,
    )


def _prepare_rl(context: _DemoContext) -> _PreparedDemo:
    # P0: load BOTH the Q table and the (5,5,3) coverage mask used by the
    # sealed acceptance; uncovered states fail closed to IMC.
    accepted = bool(context.bundle.rl_accepted)
    if accepted:
        controller: PIController = IncrementalRLController(
            context.fallback,
            q_table=np.asarray(context.bundle.rl_q_table),
            covered_mask=np.asarray(context.bundle.rl_covered_mask),
            update_interval_seconds=DEMO_SUPERVISORY_PERIOD_S,
        )
        forced = False
    else:
        controller = PIController(*context.fallback)
        forced = True
    return _PreparedDemo(
        controller=controller,
        source="validated frozen RL policy+mask" if accepted else "RL candidate rejected + IMC fallback",
        deployment_accepted=accepted,
        forced_fallback=forced,
    )


def _prepare_llm(context: _DemoContext) -> _PreparedDemo:
    if context.provider == "replay":
        policy: AgentPolicy = ReplayPIDAgentPolicy()
    elif context.provider == "ollama":
        policy = OllamaPIDAgentPolicy(context.model or "qwen3:0.6b")
    elif context.provider == "openai":
        if not context.model:
            raise ValueError("--model is required for the OpenAI provider")
        policy = OpenAIResponsesPIDAgentPolicy(context.model)
    elif context.provider == "openai-compatible":
        if not context.model:
            raise ValueError("--model is required for the openai-compatible provider")
        policy = OpenAICompatibleChatAgentPolicy(context.model, base_url=context.base_url, api_key_env=context.api_key_env)
    else:
        raise ValueError("provider must be replay, ollama, openai, or openai-compatible")
    training = sample_adaptive_scenarios(4, seed=context.seed + 30, duration_hours=3.0)
    training = training + _commissioning_training_scenarios(context.scenario, count=2, seed=context.seed + 32)
    tuned = LLMAgentAutoTuner(
        policy,
        max_steps=6,
        max_trials=3,
        max_change_fraction=MAX_FRACTIONAL_GAIN_CHANGE,
        config=RiskSafetyConfig(repeats=2, max_undershoot_c=4.0),
    ).tune(training, context.fallback, seed=context.seed + 31)
    return _PreparedDemo(
        controller=PIController(tuned.kp, tuned.ki),
        source=policy.name,
        deployment_accepted=bool(tuned.accepted or not tuned.fallback_used),
        forced_fallback=bool(tuned.fallback_used),
        agent_trace=tuned.trace,
        agent_baseline_gains=(float(context.fallback[0]), float(context.fallback[1])),
        agent_deployed_gains=(float(tuned.kp), float(tuned.ki)),
    )


_DEMO_PREPARATIONS: dict[str, Callable[[_DemoContext], _PreparedDemo]] = {
    "zn": _prepare_zn,
    "imc": _prepare_imc,
    "bo": _prepare_bo,
    "safe-bo": _prepare_safe_bo,
    "fnn": _prepare_fnn,
    "rl": _prepare_rl,
    "llm": _prepare_llm,
}


def run_algorithm_demo(
    algorithm: str,
    *,
    project_root: str | Path,
    scenario: Scenario | None = None,
    provider: str = "replay",
    model: str = "",
    base_url: str = "",
    api_key_env: str = "",
    seed: int = 71,
    artifact_dir: str | Path | None = None,
) -> DemoTrace:
    if algorithm not in ALGORITHM_ORDER:
        raise ValueError(f"unknown algorithm: {algorithm}")
    root = Path(project_root)
    scenario = scenario or demo_scenario()
    if artifact_dir is not None:
        resolved_artifact_dir = Path(artifact_dir)
    else:
        env_dir = os.environ.get("HVAC_ARTIFACT_DIR", "")
        if env_dir:
            resolved_artifact_dir = Path(env_dir)
        else:
            # Transitional fallback only; v4 callers must pass --artifact-dir
            # or HVAC_ARTIFACT_DIR pointing at artifacts/runs/<run_id>.
            # Canonical sealed batch is archive/outputs_review_v3 (dedup P1).
            warnings.warn(
                "artifact_dir not given; falling back to legacy "
                "archive/outputs_review_v3. Pass --artifact-dir explicitly.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved_artifact_dir = root / "archive/outputs_review_v3"
    artifact_dir = resolved_artifact_dir
    model_fopdt = identify_fopdt(scenario)
    # P0: one bundle for every algorithm; IMC fallback is the bundle's
    # selected lambda row, never a per-caller re-identification.
    bundle = _resolve_bundle(artifact_dir, root)
    context = _DemoContext(
        root=root,
        artifact_dir=artifact_dir,
        scenario=scenario,
        model_fopdt=model_fopdt,
        fallback=tuple(bundle.imc_gains),
        bundle=bundle,
        seed=seed,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    prepared = _DEMO_PREPARATIONS[algorithm](context)
    result = simulate(scenario, prepared.controller, seed=seed)
    # One commissioning gate for every algorithm: the controller must settle
    # the visible scenario before the door disturbance, or the deterministic
    # IMC fallback deploys and the candidate survives only as a shadow curve.
    if not _settles_before_disturbance(result, scenario):
        prepared = replace(
            prepared,
            controller=PIController(*context.fallback),
            candidate_result=result,
            deployment_accepted=False,
            forced_fallback=True,
            source=prepared.source + "; candidate rejected by visible commissioning gate, IMC fallback deployed",
        )
        result = simulate(scenario, prepared.controller, seed=seed)
    rows, summary = _status_rows(
        scenario,
        result,
        algorithm=algorithm,
        forced_fallback=prepared.forced_fallback,
        candidate_result=prepared.candidate_result,
    )
    summary.update(
        {
            "description": DESCRIPTIONS[algorithm],
            "source": prepared.source,
            "deployment_accepted": int(prepared.deployment_accepted),
            "provider": provider if algorithm == "llm" else "not_applicable",
            # Provenance: every row traces to the same bundle + code version.
            "policy_source": context.bundle.provenance.get(algorithm, context.bundle.provenance.get("artifact_dir", "")),
            "artifact_dir": str(artifact_dir.resolve()) if hasattr(artifact_dir, "resolve") else str(artifact_dir),
            "bundle_bo_legacy": context.bundle.provenance.get("bo_legacy_fallback", "0"),
            "bundle_safe_bo_legacy": context.bundle.provenance.get("safe_bo_legacy_fallback", "0"),
        }
    )
    evaluated_agent_rows = [row for row in prepared.agent_trace if row.get("tool") == "evaluate_candidate"]
    if evaluated_agent_rows:
        accepted_rows = [row for row in evaluated_agent_rows if int(row.get("accepted", 0)) == 1]
        described = accepted_rows[-1] if accepted_rows else evaluated_agent_rows[-1]
        baseline = prepared.agent_baseline_gains or (float("nan"), float("nan"))
        deployed = prepared.agent_deployed_gains or (float("nan"), float("nan"))
        summary.update(
            {
                "llm_baseline_kp": float(baseline[0]),
                "llm_baseline_ki": float(baseline[1]),
                "llm_deployed_kp": float(deployed[0]),
                "llm_deployed_ki": float(deployed[1]),
                "llm_final_kp_change_pct": 100.0 * (float(deployed[0]) - float(baseline[0])) / float(baseline[0]),
                "llm_final_ki_change_pct": 100.0 * (float(deployed[1]) - float(baseline[1])) / float(baseline[1]),
                "llm_raw_kp": float(described["raw_kp"]),
                "llm_raw_ki": float(described["raw_ki"]),
                "llm_limited_kp": float(described["limited_kp"]),
                "llm_limited_ki": float(described["limited_ki"]),
                "llm_described_trial": "accepted" if accepted_rows else "last_rejected",
                "llm_accepted_trials": len(accepted_rows),
                "llm_diagnosis": "Agent自主选择工具与候选试验",
                "llm_rationale": str(described.get("reason", "")),
                "llm_decision": str(described["decision"]),
                "llm_replay_disclosure": int(provider == "replay"),
                "llm_agent_steps": len(prepared.agent_trace),
                "llm_agent_trials": len(evaluated_agent_rows),
                "llm_trials_log": [
                    {
                        "step": row["step"],
                        "raw_kp": row["raw_kp"],
                        "raw_ki": row["raw_ki"],
                        "limited_kp": row["limited_kp"],
                        "limited_ki": row["limited_ki"],
                        "improvement_fraction": row.get("improvement_fraction"),
                        "accepted": row["accepted"],
                        "decision": row["decision"],
                    }
                    for row in evaluated_agent_rows
                ],
            }
        )
    return DemoTrace(
        algorithm,
        DISPLAY_NAMES[algorithm],
        DESCRIPTIONS[algorithm],
        prepared.source,
        result,
        rows,
        summary,
        prepared.deployment_accepted,
        prepared.forced_fallback,
        prepared.agent_trace,
    )


def run_all_demos(
    *,
    project_root: str | Path,
    scenario: Scenario | None = None,
    provider: str = "replay",
    model: str = "",
    base_url: str = "",
    api_key_env: str = "",
    seed: int = 71,
    artifact_dir: str | Path | None = None,
) -> dict[str, DemoTrace]:
    scenario = scenario or demo_scenario()
    return {
        algorithm: run_algorithm_demo(
            algorithm,
            project_root=project_root,
            scenario=scenario,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            seed=seed,
            artifact_dir=artifact_dir,
        )
        for algorithm in ALGORITHM_ORDER
    }


def write_trace_csv(path: Path, trace: DemoTrace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace.rows[0]))
        writer.writeheader()
        writer.writerows(trace.rows)


def write_agent_trace_csv(path: Path, trace: DemoTrace) -> None:
    """Persist every Agent tool request and deterministic host decision."""

    if not trace.agent_trace:
        return
    fields: list[str] = []
    for row in trace.agent_trace:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trace.agent_trace)


def write_summary_csv(path: Path, traces: dict[str, DemoTrace]) -> None:
    rows = [trace.summary for trace in traces.values()]
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _serializable_trace(trace: DemoTrace) -> dict[str, object]:
    keys = (
        "wall_clock_second", "simulated_minute", "temperature_c", "setpoint_c",
        "comfort_low_c", "comfort_high_c", "command_pct", "kp", "ki",
        "candidate_temperature_c", "candidate_kp", "candidate_ki", "door_open",
        "fallback_active", "status",
    )
    return {
        "display_name": trace.display_name,
        "description": trace.description,
        "source": trace.source,
        "accepted": trace.deployment_accepted,
        "forced_fallback": trace.forced_fallback,
        "agent_trace": trace.agent_trace,
        "summary": trace.summary,
        "rows": [
            {key: (round(float(row[key]), 6) if isinstance(row[key], (float, np.floating)) else row[key]) for key in keys}
            for row in trace.rows
        ],
    }


def _system_diagram_svg() -> str:
    return """
<svg id="system-diagram" viewBox="0 0 1180 465" role="img" aria-labelledby="diagram-title diagram-desc">
  <title id="diagram-title">温度反馈闭环系统框图</title>
  <desc id="diagram-desc">上位机生成经过验收的增益，ESP32读取设定温度和测量温度，经安全PI与压缩机限制器控制空调，温度传感器形成反馈。</desc>
  <defs>
    <style>
      #system-diagram text { font-family: "Microsoft YaHei", Arial, sans-serif; text-anchor: middle; fill: #11243a; font-size: 16px; }
      #system-diagram .small { fill: #607086; font-size: 13px; }
      #system-diagram .tiny { fill: #607086; font-size: 11px; }
      #system-diagram .feedback { fill: #607086; font-size: 12px; }
    </style>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#11243a"/></marker>
  </defs>
  <g class="edges" fill="none" stroke="#11243a" stroke-width="2" marker-end="url(#arrow)">
    <path d="M156 250H265"/><path d="M365 250H470"/><path d="M600 250H700"/><path d="M850 250H960"/>
    <path d="M1040 300V420H315V300"/><path d="M555 135V205"/><path d="M555 330V295"/>
  </g>
  <g id="node-host" class="node"><rect x="350" y="15" width="410" height="120" fill="#e8f2ff" stroke="#1677ff" stroke-width="2"/><text x="555" y="43">离线调参 / 训练 / Agent 上位机</text><text x="555" y="70" class="small">经典：Z-N · IMC · BO · Safe BO</text><text x="555" y="94" class="small">智能：FNN · RL · LLM Agent</text><text x="555" y="117" class="tiny">仅下发已验收参数或冻结策略</text></g>
  <g id="node-setpoint" class="node"><rect x="30" y="215" width="126" height="70" fill="#ffffff" stroke="#91a4b8" stroke-width="2"/><text x="93" y="246">目标温度 r</text><text x="93" y="270" class="small">例如 24°C</text></g>
  <g id="node-error" class="node"><circle cx="315" cy="250" r="50" fill="#ffffff" stroke="#1677ff" stroke-width="2"/><text x="315" y="245">误差 Σ</text><text x="315" y="270" class="small">e = T-r</text></g>
  <g id="node-pi" class="node"><rect x="470" y="205" width="130" height="90" fill="#e7f6ee" stroke="#17864b" stroke-width="2"/><text x="535" y="242">安全 PI</text><text x="535" y="270" class="small">100 ms</text></g>
  <g id="node-limiter" class="node"><rect x="700" y="200" width="150" height="100" fill="#fff4df" stroke="#e37a12" stroke-width="2"/><text x="775" y="235">压缩机限制器</text><text x="775" y="263" class="small">斜率 · 量化</text><text x="775" y="285" class="small">最低频率 · 启停</text></g>
  <g id="node-plant" class="node"><rect x="960" y="195" width="170" height="110" fill="#e8f2ff" stroke="#1677ff" stroke-width="2"/><text x="1045" y="232">空调 / 虚拟对象</text><text x="1045" y="262" class="small">容量指令 u</text><text x="1045" y="286" class="small">产生室内温度 T</text></g>
  <g id="node-scheduler" class="node"><rect x="440" y="330" width="230" height="65" fill="#f5f8fb" stroke="#7446b8" stroke-width="2"/><text x="555" y="357">FNN / RL 低频调度</text><text x="555" y="381" class="small">只建议 Kp、Ki · 2 s</text></g>
  <text x="820" y="412" class="feedback">温度传感器反馈 T</text>
</svg>"""


def write_interactive_html(path: Path, traces: dict[str, DemoTrace], *, title: str = "ESP32 七算法温度闭环 Demo") -> None:
    payload = json.dumps({key: _serializable_trace(value) for key, value in traces.items()}, ensure_ascii=False, separators=(",", ":"))
    options = "".join(f'<option value="{key}">{html.escape(value.display_name)}</option>' for key, value in traces.items())
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--ink:#11243a;--muted:#607086;--line:#d9e2ec;--blue:#1677ff;--cyan:#00a6a6;--orange:#e37a12;--green:#17864b;--red:#c43d4b;--panel:#f5f8fb;--band:#dff4e8}}
*{{box-sizing:border-box}}body{{margin:0;font-family:"Microsoft YaHei",Arial,sans-serif;color:var(--ink);background:#fff}}main{{max-width:1240px;margin:auto;padding:24px}}
h1{{margin:0 0 6px;font-size:30px}}h2{{margin:30px 0 12px;font-size:22px}}p{{line-height:1.65}}.lead{{color:var(--muted);margin-top:0}}
.controls{{display:flex;gap:12px;align-items:end;flex-wrap:wrap;padding:12px 0}}label{{font-size:13px;color:var(--muted)}}select,button,input{{font:inherit;padding:8px 12px;border:1px solid var(--line);background:#fff;border-radius:6px}}button{{cursor:pointer}}button.primary{{background:var(--blue);color:#fff;border-color:var(--blue)}}
.live{{display:grid;grid-template-columns:220px 1fr;gap:22px;align-items:stretch}}.thermo{{background:var(--panel);padding:18px;border-radius:10px}}.temperature{{font-size:42px;font-weight:600;margin:12px 0}}.status{{font-size:18px;color:var(--green)}}.fallback{{color:var(--orange)}}
.chart-wrap{{position:relative}}svg.chart{{width:100%;height:auto;background:#fff;border:1px solid var(--line)}}.axis{{stroke:var(--muted);stroke-width:1}}.grid{{stroke:var(--line);stroke-width:1}}.tick{{font-size:12px;fill:var(--muted)}}
#system-diagram{{display:block;width:100%;height:auto;background:var(--panel);border-radius:10px}}#system-diagram .node rect,#system-diagram .node circle{{fill:#fff;stroke:#91a4b8;stroke-width:2}}#system-diagram .active rect,#system-diagram .active circle{{stroke:var(--blue);stroke-width:5;fill:#e8f2ff}}.feedback-steps{{color:var(--muted);margin-top:10px}}
.summary{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}.summary>div{{border-top:3px solid var(--blue);padding:12px;background:var(--panel)}}.summary strong{{display:block;font-size:22px;margin-top:6px}}.notice{{padding:12px 16px;background:#fff4df;border-left:4px solid var(--orange)}}
.legend{{display:flex;gap:15px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin:8px 0}}.swatch{{display:inline-block;width:18px;height:3px;margin-right:5px;vertical-align:middle}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left}}th{{background:var(--panel)}}
@media(max-width:760px){{.live{{grid-template-columns:1fr}}.summary{{grid-template-columns:1fr 1fr}}main{{padding:14px}}}}
</style></head><body><main>
<h1>{html.escape(title)}</h1><p class="lead">直接观察被控温度从 30°C 接近目标、进入 ±0.5°C 稳定带，并在开门扰动后恢复。90 秒墙钟时间对应 5 小时模拟物理时间；这不是实际设备降温速度。</p>
<h2>完整温度反馈闭环</h2>{_system_diagram_svg()}<p class="feedback-steps">闭环路径：测温 → 与目标相减得误差 → 安全 PI 计算容量请求 → 压缩机限制器处理斜率、量化、最低频率与启停 → 空调改变室内温度 → 再次测温。FNN / RL 每 2 秒只建议 Kp / Ki；LLM Agent 只在上位机调用仿真工具，不进入 100 ms 实时环。</p>
<div class="controls"><label>算法<br><select id="algorithm">{options}<option value="all">七算法同场比较</option></select></label><button class="primary" id="play">开始</button><button id="pause">暂停</button><button id="reset">复位</button><button id="event">跳到开门扰动</button><label>播放速度<br><select id="speed"><option value="1">1×（约90秒）</option><option value="3" selected>3×</option><option value="10">10×</option></select></label><label style="flex:1;min-width:240px">时间位置<br><input id="scrub" type="range" min="0" max="300" value="0" style="width:100%"></label></div>
<div class="live"><section class="thermo"><div>当前被控温度</div><div class="temperature" id="temp">-- °C</div><div>目标：<strong id="sp">-- °C</strong></div><div>温差：<strong id="err">-- °C</strong></div><div>容量：<strong id="cmd">-- %</strong></div><div>Kp / Ki：<strong id="gains">--</strong></div><p class="status" id="status">准备开始</p><p id="clock">墙钟 0 s · 模拟 0 min</p></section>
<section class="chart-wrap"><svg class="chart" id="temperature-chart" viewBox="0 0 960 430" role="img" aria-label="温度、设定值与稳定带随时间变化"></svg><div class="legend" id="legend"></div></section></div>
<h2>压缩机容量</h2><svg class="chart" id="command-chart" viewBox="0 0 960 230" role="img" aria-label="压缩机容量百分比随时间变化"></svg>
<div class="summary"><div>首次进入稳定带<strong id="first-band">--</strong></div><div>连续稳定起点<strong id="stable-at">--</strong></div><div>开门后恢复<strong id="recovery">--</strong></div><div>部署状态<strong id="accepted">--</strong></div></div>
<p class="notice" id="description"></p>
<section id="llm-section" style="display:none"><h2>LLM Agent 工具调用与安全门审计</h2><p>Agent 可以选择 <code>inspect_history</code>、<code>evaluate_candidate</code> 或 <code>finish</code>；它不能直接接受参数，更不能输出压缩机容量。</p><pre id="llm-audit" style="white-space:pre-wrap;background:#f5f8fb;padding:14px;border-radius:8px"></pre></section>
<h2>嵌入式验收证据与边界</h2><table><thead><tr><th>层级</th><th>结果</th><th>已有证据</th><th>仍不能证明</th></tr></thead><tbody>
<tr><td>Python统一场景</td><td>通过</td><td>七条实际部署温度均进入稳定带；开门后恢复；失败候选明确回退</td><td>不代表真实空调响应时间</td></tr>
<tr><td>PC C++ 软件在环</td><td>通过</td><td>七档增益边界、90秒加速稳定、NaN回退、100ms/2s调度逻辑</td><td>PC纳秒耗时不能写成ESP32 WCET</td></tr>
<tr><td>ESP32 目标编译</td><td>通过</td><td>PlatformIO esp32dev：RAM 22,024 B（6.7%），Flash 294,425 B（22.5%）</td><td>尚未证明板端连续运行和串口时序</td></tr>
<tr><td>Wokwi</td><td>早期双目标工程仅编译/启动</td><td>旧FNN/RL工程已有编译启动记录</td><td>不能冒充新版七算法固件或10分钟板端验收</td></tr>
<tr><td>实体ESP32</td><td>待接板</td><td><code>embedded/validate_esp32_serial.py</code> 已提供不可伪造的600秒验收入口</td><td>无真实串口CSV前不写“通过”</td></tr>
<tr><td>Modbus真实空调</td><td>默认只读</td><td>从站、寄存器、缩放、字节序、CRC、超时和告警回退均可配置</td><td>没有厂商寄存器表时禁止写容量</td></tr>
</tbody></table>
<h2>七种算法在系统中的位置</h2><table><thead><tr><th>算法</th><th>运行位置</th><th>ESP32实际执行</th><th>安全边界</th></tr></thead><tbody>
<tr><td>Z-N</td><td>上位机辨识与公式</td><td>固定安全PI</td><td>执行器限制与增益边界</td></tr><tr><td>IMC</td><td>上位机模型整定</td><td>固定安全PI / 全局回退</td><td>鲁棒λ与独立联锁</td></tr><tr><td>普通BO</td><td>上位机离线搜索</td><td>验收后的固定PI</td><td>不允许真实设备在线探索</td></tr><tr><td>风险安全BO</td><td>上位机风险与安全代理</td><td>验收后的固定PI</td><td>预测安全集与重复噪声仿真</td></tr><tr><td>FNN</td><td>PC训练，ESP32插值</td><td>每2秒候选Kp/Ki；当前拒绝后回退</td><td>规则验收、±10%与IMC回退</td></tr><tr><td>RL</td><td>PC训练，ESP32查冻结策略</td><td>每2秒候选Kp/Ki</td><td>无在线探索、状态掩码与回退</td></tr><tr><td>LLM Agent</td><td>本地/云端上位机</td><td>自主选仿真工具；只下发已验收固定参数</td><td>工具白名单、试验预算、±10%、仿真门和IMC回退</td></tr>
</tbody></table>
</main><script>
const traces={payload}; const colors={{"zn":"#c43d4b","imc":"#17864b","bo":"#1677ff","safe-bo":"#00a6a6","fnn":"#e37a12","rl":"#7446b8","llm":"#39475b"}};
const select=document.getElementById('algorithm'), scrub=document.getElementById('scrub'); let timer=null,index=0;
const nodes=['node-error','node-pi','node-limiter','node-plant','node-error','node-scheduler'];
function selectedKeys(){{return select.value==='all'?Object.keys(traces):[select.value]}}
function linePath(rows,key,x,y,limit){{let d=''; for(let i=0;i<=limit&&i<rows.length;i++){{const px=x(rows[i].simulated_minute),py=y(rows[i][key]);d+=(i?'L':'M')+px.toFixed(1)+','+py.toFixed(1)}}return d}}
function axisSvg(width,height,margin,yMin,yMax,yLabel){{let s=`<rect x="${{margin.l}}" y="${{margin.t}}" width="${{width-margin.l-margin.r}}" height="${{height-margin.t-margin.b}}" fill="white" stroke="#d9e2ec"/>`;for(let i=0;i<=5;i++){{const yy=margin.t+(height-margin.t-margin.b)*i/5;const v=yMax-(yMax-yMin)*i/5;s+=`<line class="grid" x1="${{margin.l}}" x2="${{width-margin.r}}" y1="${{yy}}" y2="${{yy}}"/><text class="tick" x="${{margin.l-10}}" y="${{yy+4}}" text-anchor="end">${{v.toFixed(1)}}</text>`}}for(let i=0;i<=5;i++){{const xx=margin.l+(width-margin.l-margin.r)*i/5;s+=`<text class="tick" x="${{xx}}" y="${{height-15}}" text-anchor="middle">${{i}} h</text>`}}s+=`<text class="tick" x="18" y="${{height/2}}" transform="rotate(-90 18 ${{height/2}})" text-anchor="middle">${{yLabel}}</text>`;return s}}
function drawTemperature(){{const svg=document.getElementById('temperature-chart'),W=960,H=430,m={{l:68,r:22,t:22,b:48}},rows=traces[Object.keys(traces)[0]].rows;const yMin=22,yMax=31;const x=v=>m.l+(W-m.l-m.r)*v/300,y=v=>m.t+(H-m.t-m.b)*(yMax-v)/(yMax-yMin);let s=axisSvg(W,H,m,yMin,yMax,'温度 °C');const low=rows[0].comfort_low_c,high=rows[0].comfort_high_c;s+=`<rect x="${{m.l}}" y="${{y(high)}}" width="${{W-m.l-m.r}}" height="${{y(low)-y(high)}}" fill="#dff4e8" opacity=".8"/><text x="${{W-m.r-8}}" y="${{y(high)+16}}" text-anchor="end" fill="#17864b" font-size="12">目标±0.5°C稳定带</text>`;const eventX=x(rows.find(r=>r.door_open)?.simulated_minute||189);s+=`<line x1="${{eventX}}" x2="${{eventX}}" y1="${{m.t}}" y2="${{H-m.b}}" stroke="#e37a12" stroke-width="2" stroke-dasharray="7 5"/><text x="${{eventX+6}}" y="${{m.t+18}}" fill="#e37a12" font-size="12">开门扰动</text>`;for(const key of selectedKeys()){{const t=traces[key];s+=`<path d="${{linePath(t.rows,'temperature_c',x,y,index)}}" fill="none" stroke="${{colors[key]}}" stroke-width="3"/>`;if(t.forced_fallback) s+=`<path d="${{linePath(t.rows,'candidate_temperature_c',x,y,index)}}" fill="none" stroke="${{colors[key]}}" stroke-width="2" stroke-dasharray="4 5" opacity=".65"/>`}}const active=traces[select.value==='all'?'imc':select.value];s+=`<path d="${{linePath(active.rows,'setpoint_c',x,y,index)}}" fill="none" stroke="#11243a" stroke-width="2" stroke-dasharray="8 5"/>`;const cx=x(active.rows[index].simulated_minute),cy=y(active.rows[index].temperature_c);s+=`<circle cx="${{cx}}" cy="${{cy}}" r="6" fill="${{colors[select.value]||'#11243a'}}"/>`;svg.innerHTML=s;document.getElementById('legend').innerHTML=selectedKeys().map(k=>`<span><i class="swatch" style="background:${{colors[k]}}"></i>${{traces[k].display_name}}</span>`).join('')+'<span><i class="swatch" style="background:#11243a"></i>目标温度</span>'}}
function drawCommand(){{const svg=document.getElementById('command-chart'),W=960,H=230,m={{l:68,r:22,t:18,b:42}},x=v=>m.l+(W-m.l-m.r)*v/300,y=v=>m.t+(H-m.t-m.b)*(100-v)/100;let s=axisSvg(W,H,m,0,100,'容量 %');for(const key of selectedKeys())s+=`<path d="${{linePath(traces[key].rows,'command_pct',x,y,index)}}" fill="none" stroke="${{colors[key]}}" stroke-width="2.5"/>`;svg.innerHTML=s}}
function fmt(v,suffix=' min'){{return Number.isFinite(Number(v))?Number(v).toFixed(1)+suffix:'未达到'}}
function drawAgentAudit(key,t,sum){{
  const section=document.getElementById('llm-section');
  section.style.display=key==='llm'?'block':'none';
  if(key!=='llm')return;
  const lines=(t.agent_trace||[]).map(row=>{{
    const parts=[
      '步骤 '+row.step+' · '+row.tool,
      '原因：'+(row.reason||'—'),
      '主机决定：'+row.decision
    ];
    if(row.limited_kp!==undefined){{
      parts.push('原始 Kp/Ki：'+Number(row.raw_kp).toFixed(6)+' / '+Number(row.raw_ki).toFixed(7));
      parts.push('限幅 Kp/Ki：'+Number(row.limited_kp).toFixed(6)+' / '+Number(row.limited_ki).toFixed(7));
      parts.push('风险目标：'+Number(row.risk_objective).toFixed(3)+' · 安全='+row.safe+' · 接受='+row.accepted);
    }}
    return parts.join(String.fromCharCode(10)+'  ');
  }});
  const disclosure=sum.llm_replay_disclosure
    ? '声明：这是无网络录制的 Agent 工具调用 replay，不是本次实时模型调用。'+String.fromCharCode(10,10)
    : '';
  document.getElementById('llm-audit').textContent=disclosure+lines.join(String.fromCharCode(10,10));
}}
function draw(){{const key=select.value==='all'?'imc':select.value,t=traces[key],r=t.rows[index],sum=t.summary;document.getElementById('temp').textContent=r.temperature_c.toFixed(2)+' °C';document.getElementById('sp').textContent=r.setpoint_c.toFixed(2)+' °C';document.getElementById('err').textContent=(r.temperature_c-r.setpoint_c>=0?'+':'')+(r.temperature_c-r.setpoint_c).toFixed(2)+' °C';document.getElementById('cmd').textContent=r.command_pct.toFixed(1)+' %';document.getElementById('gains').textContent=r.kp.toFixed(4)+' / '+r.ki.toFixed(5);const st=document.getElementById('status');st.textContent=r.status;st.className='status '+(r.fallback_active?'fallback':'');document.getElementById('clock').textContent='墙钟 '+r.wall_clock_second.toFixed(1)+' s · 模拟 '+r.simulated_minute.toFixed(0)+' min';document.getElementById('first-band').textContent=fmt(sum.first_in_band_minute);document.getElementById('stable-at').textContent=fmt(sum.stable_first_minute);document.getElementById('recovery').textContent=fmt(sum.door_recovery_minutes);document.getElementById('accepted').textContent=t.accepted?'已通过部署门':'已拒绝/回退';document.getElementById('description').textContent=t.description+' 产物来源：'+t.source;drawAgentAudit(key,t,sum);document.querySelectorAll('#system-diagram .node').forEach(n=>n.classList.remove('active'));document.getElementById(nodes[index%nodes.length]).classList.add('active');scrub.value=index;drawTemperature();drawCommand()}}
function stop(){{if(timer)clearInterval(timer);timer=null}}function play(){{stop();timer=setInterval(()=>{{if(index>=300){{stop();return}}index++;draw()}},300/Number(document.getElementById('speed').value))}}
document.getElementById('play').onclick=play;document.getElementById('pause').onclick=stop;document.getElementById('reset').onclick=()=>{{stop();index=0;draw()}};document.getElementById('event').onclick=()=>{{index=Math.max(0,traces[Object.keys(traces)[0]].rows.findIndex(r=>r.door_open));draw()}};select.onchange=()=>{{index=0;draw()}};scrub.oninput=()=>{{index=Number(scrub.value);draw()}};draw();
</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def write_esp32_profile_header(path: Path, traces: dict[str, DemoTrace]) -> None:
    enum_rows = ",\n  ".join(f"{key.replace('-', '_').upper()}" for key in ALGORITHM_ORDER)
    profile_rows = []
    for key in ALGORITHM_ORDER:
        row = traces[key].summary
        profile_rows.append(
            "  {\"%s\", %.9gf, %.9gf, %s, %s}"
            % (
                key,
                float(row["kp_final"]),
                float(row["ki_final"]),
                "true" if traces[key].deployment_accepted else "false",
                "true" if traces[key].forced_fallback else "false",
            )
        )
    joined_profiles = ",\n".join(profile_rows)
    manifest = ";".join(
        "%s:%.9g:%.9g:%d:%d"
        % (
            key,
            float(traces[key].summary["kp_final"]),
            float(traces[key].summary["ki_final"]),
            int(traces[key].deployment_accepted),
            int(traces[key].forced_fallback),
        )
        for key in ALGORITHM_ORDER
    )
    profile_crc32 = zlib.crc32(manifest.encode("ascii")) & 0xFFFFFFFF
    content = f"""#pragma once

#include <cstdint>

namespace hvac_mcu {{ namespace demo {{

enum class AlgorithmId : uint8_t {{
  {enum_rows}
}};

struct ControllerProfile {{
  const char *name;
  float kp;
  float ki;
  bool accepted;
  bool fallback_required;
}};

static constexpr ControllerProfile kProfiles[7] = {{
{joined_profiles}
}};

static constexpr uint32_t kProfileVersion = 1u;
static constexpr const char kProfileManifest[] = "{manifest}";
static constexpr uint32_t kProfileCrc32 = 0x{profile_crc32:08X}u;
static constexpr uint32_t kPidPeriodMs = 100u;
static constexpr uint32_t kAdaptivePeriodMs = 2000u;
static constexpr uint32_t kTelemetryPeriodMs = 500u;

}} }}  // namespace hvac_mcu::demo
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def write_demo_bundle(output_dir: str | Path, traces: dict[str, DemoTrace], *, project_root: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for key, trace in traces.items():
        write_trace_csv(output / f"{key.replace('-', '_')}_temperature_demo.csv", trace)
        if trace.agent_trace:
            write_agent_trace_csv(output / f"{key.replace('-', '_')}_agent_trace.csv", trace)
        write_interactive_html(output / f"{key.replace('-', '_')}_temperature_demo.html", {key: trace}, title=f"{trace.display_name}温度闭环 Demo")
    write_summary_csv(output / "seven_algorithm_summary.csv", traces)
    write_interactive_html(output / "temperature_control_demo.html", traces)
    header = Path(project_root) / "embedded" / "generated_demo_profiles.hpp"
    write_esp32_profile_header(header, traces)
    return {
        "interactive_html": str((output / "temperature_control_demo.html").resolve()),
        "summary_csv": str((output / "seven_algorithm_summary.csv").resolve()),
        "llm_agent_trace_csv": str((output / "llm_agent_trace.csv").resolve()),
        "esp32_profile_header": str(header.resolve()),
    }
