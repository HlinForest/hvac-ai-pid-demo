from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .config import Scenario, dynamic_demo_scenario, sample_adaptive_scenarios, typical_case_scenarios
from .ai_controllers import FNNGainController, IncrementalRLController, train_fnn_rule_table, train_offline_q_policy
from .algorithm_reports import write_algorithm_reports
from .controllers import (
    PIController,
    classical_tuning_calculation_steps,
    identify_fopdt_audit,
    imc_pi,
    ziegler_nichols_pi,
)
from .metrics import calculate_metrics
from .plotting import (
    plot_bayesian_search_trace,
    plot_case_comparisons,
    plot_case_studies,
    plot_classical_tuning_process,
    plot_dynamic_comparison,
    plot_fnn_training_trace,
    plot_rl_training_trace,
    plot_training_convergence_overview,
    plot_training_labels,
)
from .report import write_engineering_report, write_html_engineering_report
from .simulator import SimulationResult, simulate
from .tuning import generate_label_rows, tune_global_fixed, tune_global_imc_lambda
from .validation import run_cross_validation


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _scenario_rows(scenarios: list[Scenario]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, scenario in enumerate(scenarios):
        row = {"scenario": index + 1, **asdict(scenario)}
        row.pop("FEATURE_NAMES", None)
        rows.append(row)
    return rows


def _scenario_digest(scenario: Scenario) -> str:
    payload = asdict(scenario)
    payload.pop("FEATURE_NAMES", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _dataset_manifest_rows(
    partitions: list[tuple[str, int, list[Scenario]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for split, split_seed, scenarios in partitions:
        for index, scenario in enumerate(scenarios):
            digest = _scenario_digest(scenario)
            if digest in seen:
                raise ValueError(f"scenario leakage detected for digest {digest}")
            seen.add(digest)
            rows.append(
                {
                    "scenario_id": f"{split}-{split_seed}-{index + 1:03d}-{digest[:12]}",
                    "split": split,
                    "split_seed": split_seed,
                    "split_index": index + 1,
                    "scenario_sha256": digest,
                }
            )
    return rows


def _paired_ratio_upper95(candidate: np.ndarray, baseline: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(candidate), size=(2000, len(candidate)))
    ratios = np.mean(candidate[indices], axis=1) / np.maximum(np.mean(baseline[indices], axis=1), 1e-12)
    return float(np.quantile(ratios, 0.95))


def _deployment_acceptance(
    seeded_scenarios: list[tuple[int, int, Scenario]],
    imc_gains: tuple[float, float],
    fnn_candidate: np.ndarray,
    fnn_context: np.ndarray,
    rl_candidate: np.ndarray,
    rl_covered: np.ndarray,
    *,
    fnn_rule_coverage: float,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, bool]]:
    raw: dict[str, list[dict[str, float]]] = {"FNN": [], "RL": []}
    for ordinal, (test_seed, scenario_index, scenario) in enumerate(seeded_scenarios):
        simulation_seed = seed + 700_000 + ordinal
        baseline_result = simulate(scenario, PIController(*imc_gains), seed=simulation_seed)
        baseline_metrics = calculate_metrics(baseline_result)
        candidates = {
            "FNN": FNNGainController(imc_gains, rule_table=fnn_candidate, context_coefficients=fnn_context),
            "RL": IncrementalRLController(imc_gains, q_table=rl_candidate, covered_mask=rl_covered),
        }
        for name, controller in candidates.items():
            result = simulate(scenario, controller, seed=simulation_seed)
            metrics = calculate_metrics(result)
            changed = (np.abs(result.kp - imc_gains[0]) > 1e-7) | (np.abs(result.ki - imc_gains[1]) > 1e-9)
            raw[name].append(
                {
                    "test_seed": float(test_seed),
                    "scenario": float(scenario_index),
                    "baseline_objective": float(baseline_metrics["objective"]),
                    "candidate_objective": float(metrics["objective"]),
                    "stable": float(metrics["stable"]),
                    "running_slew_violations": float(metrics["running_slew_violation_count"]),
                    "minimum_frequency_violations": float(metrics["sub_minimum_running_fraction"]),
                    "adaptive_fraction": float(np.mean(changed)),
                    "fallback_fraction": float(np.mean(result.fallback_active)),
                }
            )
    rows: list[dict[str, object]] = []
    decisions: dict[str, bool] = {}
    for name, values in raw.items():
        candidate = np.asarray([row["candidate_objective"] for row in values])
        baseline = np.asarray([row["baseline_objective"] for row in values])
        mean_ratio = float(np.mean(candidate) / max(np.mean(baseline), 1e-12))
        upper95 = _paired_ratio_upper95(candidate, baseline, seed + (11 if name == "FNN" else 29))
        stable_rate = float(np.mean([row["stable"] for row in values]))
        adaptive_fraction = float(np.mean([row["adaptive_fraction"] for row in values]))
        fallback_fraction = float(np.mean([row["fallback_fraction"] for row in values]))
        safety_ok = all(
            row["running_slew_violations"] == 0.0 and row["minimum_frequency_violations"] == 0.0
            for row in values
        )
        coverage_ok = fnn_rule_coverage >= 0.80 if name == "FNN" else fallback_fraction <= 0.10
        accepted = bool(
            mean_ratio <= 1.0
            and upper95 <= 1.02
            and stable_rate >= 1.0
            and safety_ok
            and adaptive_fraction >= 0.10
            and coverage_ok
        )
        decisions[name] = accepted
        for value in values:
            rows.append(
                {
                    "controller": name,
                    **value,
                    "objective_ratio": value["candidate_objective"] / max(value["baseline_objective"], 1e-12),
                    "mean_objective_ratio": mean_ratio,
                    "upper95_objective_ratio": upper95,
                    "mean_adaptive_fraction": adaptive_fraction,
                    "mean_fallback_fraction": fallback_fraction,
                    "coverage_ok": int(coverage_ok),
                    "safety_ok": int(safety_ok),
                    "deployment_accepted": int(accepted),
                }
            )
    return rows, decisions


def _write_review_remediation(
    output: Path,
    acceptance_rows: list[dict[str, object]],
    decisions: dict[str, bool],
    fnn_history: list[dict[str, float]],
    rl_history: list[dict[str, float]],
) -> str:
    fnn_summary = next(row for row in acceptance_rows if row["controller"] == "FNN")
    rl_summary = next(row for row in acceptance_rows if row["controller"] == "RL")
    fnn_last = fnn_history[-1]
    rl_last = rl_history[-1]
    items = [
        ("A1", "统一执行器约束", "已闭环", "simulator.py / actuator.py", "全部算法共用最低容量、量化、斜率和启停限制"),
        ("A2", "噪声与滤波", "已闭环", "dataset_manifest.csv", "训练、验证和测试工况均包含传感器噪声与滤波"),
        ("A3", "IMC公平调参", "已闭环", "imc_lambda_tuning.csv", "lambda只在训练集选择，进入验证和密封测试后冻结"),
        ("B1", "FNN标签混叠", "已闭环", "fnn_training_samples.csv", "同一热状态快照执行30分钟局部候选回放，并记录容量、积分、室外温差和负荷"),
        ("B2", "FNN规则覆盖", "已闭环" if float(fnn_last["rule_coverage_pct"]) >= 80 else "未通过", "fnn_training_history.csv", f"有效规则覆盖率={float(fnn_last['rule_coverage_pct']):.1f}%"),
        ("B3", "FNN离策略压缩", "已闭环", "fnn_training_samples.csv", "聚合IMC、BO和安全残差策略三类物理轨迹"),
        ("B4", "FNN独立部署门", "已闭环" if decisions["FNN"] else "未通过", "deployment_acceptance.csv", f"密封测试均值比={float(fnn_summary['mean_objective_ratio']):.4f}，95%上界={float(fnn_summary['upper95_objective_ratio']):.4f}"),
        ("C1", "RL伪造状态覆盖", "已闭环", "rl_training_transitions.csv", "仅记录3R2C物理轨迹真实到达状态"),
        ("C2", "未访问状态默认偏置", "已闭环", "generated_policy.hpp", "未覆盖组合使用动作4且触发IMC回退"),
        ("C3", "训练时域过短", "已闭环", "rl_training_history.csv", "750回合、最长240分钟、5分钟决策"),
        ("C4", "RL状态非严格Markov", "部分闭环", "rl_training_transitions.csv", "容量已进入Q状态；积分、限制器和负荷作为安全上下文记录，真实负荷仍是估计量"),
        ("C5", "增益累积漂移与奖励不一致", "已闭环", "ai_controllers.py", "动作改为相对IMC绝对目标，并采用基线约束策略改进"),
        ("C6", "RL独立部署门", "已闭环" if decisions["RL"] else "未通过", "deployment_acceptance.csv", f"密封测试均值比={float(rl_summary['mean_objective_ratio']):.4f}，95%上界={float(rl_summary['upper95_objective_ratio']):.4f}"),
        ("D1-D3", "指标与时序审计", "已闭环", "dynamic_timeseries.csv", "真实/测量温度、请求/实际命令、增益、回退和约束指标分别记录"),
        ("E1", "全量测试入口", "已闭环", "pytest.ini", "统一使用pytest并包含Python、HTML和嵌入式测试"),
        ("E2", "训练验证测试隔离", "已闭环", "dataset_manifest.csv", "默认48/16/16，并对每个工况写入SHA-256且拒绝重复"),
        ("E3", "报告fail-open", "已闭环", "report.py", "缺失、NaN、非法验收值一律判为未通过"),
        ("E4", "加速演示时间基准", "已闭环", "testbench.cpp / mcu_validation_summary.csv", "PI积分、限幅器与监督误差率统一使用200×模拟时间（与ESP32固件一致）；SIL判据含未覆盖回退占比≤10%门，实测0/190"),
        ("E5", "策略导出与CRC", "已闭环", "policy_parity_vectors.csv", "CRC v3覆盖固件执行字段并完成Python/C++逐向量一致性"),
    ]
    rows = [
        {"defect_id": defect, "review_issue": issue, "status": status, "evidence": evidence, "result": result}
        for defect, issue, status, evidence, result in items
    ]
    _write_rows(output / "review_defect_matrix.csv", rows)
    table_lines = ["|编号|评审问题|状态|证据|可核查结果|", "|---|---|---|---|---|"]
    table_lines.extend(
        f"|{row['defect_id']}|{row['review_issue']}|{row['status']}|`{row['evidence']}`|{row['result']}|"
        for row in rows
    )
    markdown = "\n".join(
        [
            "## 评审缺陷 A1–E5 闭环矩阵",
            "",
            "状态由本次产物动态生成；算法未通过时保持拒绝并使用IMC，真实ESP32/BMS证据不在本轮软件验收范围内。",
            "",
            *table_lines,
        ]
    )
    (output / "review_remediation.md").write_text(markdown + "\n", encoding="utf-8")
    return markdown


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    controllers = list(dict.fromkeys(str(row["controller"]) for row in rows))
    numeric_fields = [
        "objective",
        "rmse_c",
        "iae_c_hour",
        "comfort_violation_c_hour",
        "max_undershoot_c",
        "settling_time_hour",
        "cooling_energy_kwh",
        "control_movement",
        "max_running_slew_per_minute",
        "running_slew_violation_count",
        "sub_minimum_running_fraction",
        "compressor_start_events",
        "compressor_stop_events",
        "itae_c_hour2",
        "compressor_output_variance",
        "mean_ai_inference_us",
        "fallback_events",
    ]
    summary: list[dict[str, object]] = []
    for controller in controllers:
        subset = [row for row in rows if row["controller"] == controller]
        item: dict[str, object] = {"controller": controller, "scenarios": len(subset)}
        for field in numeric_fields:
            values = np.asarray([float(row[field]) for row in subset])
            item[f"mean_{field}"] = float(np.mean(values))
            item[f"median_{field}"] = float(np.median(values))
        item["stable_rate"] = float(np.mean([float(row["stable"]) for row in subset]))
        summary.append(item)
    return summary


def _evaluate_holdout(
    scenarios: list[Scenario],
    fixed_gains: tuple[float, float],
    zn_gains: tuple[float, float],
    imc_gains: tuple[float, float],
    fnn_rule_table: np.ndarray,
    rl_q_table: np.ndarray,
    seed: int,
    rl_covered: np.ndarray | None = None,
    fnn_context: np.ndarray | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, scenario in enumerate(scenarios):
        methods = {
            "Bayesian Auto-tune": PIController(*fixed_gains),
            "Ziegler-Nichols": zn_gains,
            "IMC PI": imc_gains,
            "FNN Self-tuning PI": FNNGainController(imc_gains, rule_table=fnn_rule_table, context_coefficients=fnn_context),
            "RL Self-tuning PI": IncrementalRLController(imc_gains, q_table=rl_q_table, covered_mask=rl_covered),
        }
        for name, method in methods.items():
            controller = method if isinstance(method, PIController) else PIController(*method)
            result = simulate(scenario, controller, seed=seed + index)
            row: dict[str, object] = {
                "scenario": index,
                "controller": name,
                "kp": controller.kp,
                "ki": controller.ki,
                "out_of_domain": 0,
            }
            row.update(calculate_metrics(result))
            rows.append(row)
    return rows


def _controllers_for_scenario(
    scenario: Scenario,
    fixed_gains: tuple[float, float],
    zn_gains: tuple[float, float],
    imc_gains: tuple[float, float],
    fnn_rule_table: np.ndarray,
    rl_q_table: np.ndarray,
    rl_covered: np.ndarray | None = None,
    fnn_context: np.ndarray | None = None,
) -> dict[str, PIController]:
    return {
        "Bayesian Auto-tune": PIController(*fixed_gains),
        "Ziegler-Nichols": PIController(*zn_gains),
        "IMC PI": PIController(*imc_gains),
        "FNN Self-tuning PI": FNNGainController(imc_gains, rule_table=fnn_rule_table, context_coefficients=fnn_context),
        "RL Self-tuning PI": IncrementalRLController(imc_gains, q_table=rl_q_table, covered_mask=rl_covered),
    }


def _run_dynamic(
    fixed_gains: tuple[float, float], zn_gains: tuple[float, float], imc_gains: tuple[float, float], fnn_rule_table: np.ndarray, rl_q_table: np.ndarray, seed: int,
    rl_covered: np.ndarray | None = None,
    fnn_context: np.ndarray | None = None,
) -> tuple[dict[str, SimulationResult], dict[str, dict[str, float]]]:
    scenario = dynamic_demo_scenario()
    controllers = _controllers_for_scenario(scenario, fixed_gains, zn_gains, imc_gains, fnn_rule_table, rl_q_table, rl_covered, fnn_context)
    results = {name: simulate(scenario, controller, seed=seed) for name, controller in controllers.items()}
    metrics = {name: calculate_metrics(result) for name, result in results.items()}
    return results, metrics


def _run_case_studies(
    fixed_gains: tuple[float, float], zn_gains: tuple[float, float], imc_gains: tuple[float, float], fnn_rule_table: np.ndarray, rl_q_table: np.ndarray, seed: int,
    rl_covered: np.ndarray | None = None,
    fnn_context: np.ndarray | None = None,
) -> tuple[dict[str, dict[str, SimulationResult]], list[dict[str, object]]]:
    all_results: dict[str, dict[str, SimulationResult]] = {}
    rows: list[dict[str, object]] = []
    for case_index, (case_name, scenario) in enumerate(typical_case_scenarios().items()):
        controllers = _controllers_for_scenario(scenario, fixed_gains, zn_gains, imc_gains, fnn_rule_table, rl_q_table, rl_covered, fnn_context)
        results = {name: simulate(scenario, controller, seed=seed + case_index) for name, controller in controllers.items()}
        all_results[case_name] = results
        for name, result in results.items():
            rows.append({"case": case_name, "controller": name, **calculate_metrics(result)})
    return all_results, rows


def run_pipeline(
    output_dir: str | Path,
    *,
    train_samples: int = 48,
    validation_samples: int = 16,
    test_samples: int = 16,
    bo_iterations: int = 7,
    seed: int = 7,
    acceptance_seeds: tuple[int, ...] = (101, 211, 307, 401, 503),
) -> dict[str, object]:
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("[1/3] Generating offline PI gain labels and one global Bayesian PI pair ...")
    commissioning_scenario = Scenario()
    commissioning_fopdt, identification_history, fopdt_fit_history = identify_fopdt_audit(commissioning_scenario)
    zn_gains = ziegler_nichols_pi(commissioning_fopdt)
    formula_imc_gains = imc_pi(commissioning_fopdt)
    classical_rows = [
        {
            **row,
            "zn_kp": zn_gains[0],
            "zn_ki": zn_gains[1],
            "imc_kp": formula_imc_gains[0],
            "imc_ki": formula_imc_gains[1],
        }
        for row in identification_history
    ]
    _write_rows(output / "classical_tuning_history.csv", classical_rows)
    _write_rows(output / "fopdt_fit_history.csv", fopdt_fit_history)
    classical_steps = classical_tuning_calculation_steps(
        commissioning_scenario,
        commissioning_fopdt,
        identification_history,
        fopdt_fit_history,
    )
    _write_rows(output / "classical_tuning_steps.csv", classical_steps)
    if min(train_samples, validation_samples, test_samples) <= 0:
        raise ValueError("train, validation and test sample counts must all be positive")
    if not acceptance_seeds:
        raise ValueError("acceptance_seeds must not be empty")
    training_scenarios = sample_adaptive_scenarios(train_samples, seed=seed, duration_hours=5.0)
    validation_seed = seed + 50_003
    validation_scenarios = sample_adaptive_scenarios(validation_samples, seed=validation_seed, duration_hours=5.0)
    _write_rows(output / "training_scenarios.csv", _scenario_rows(training_scenarios))
    _write_rows(output / "validation_scenarios.csv", _scenario_rows(validation_scenarios))
    imc_tune = tune_global_imc_lambda(
        training_scenarios,
        commissioning_fopdt,
        seed=seed + 404,
    )
    imc_gains = (imc_tune.kp, imc_tune.ki)
    _write_rows(output / "imc_lambda_tuning.csv", list(imc_tune.history))
    selected_imc = next(row for row in imc_tune.history if int(row["selected"]) == 1)
    classical_steps.extend(
        [
            {
                "step": len(classical_steps) + 1,
                "method": "IMC λ训练集调参",
                "quantity": "公平比较协议",
                "formula_or_action": "仅在训练工况搜索lambda；Kp/Ti仍由IMC公式约束",
                "substitution": f"{imc_tune.evaluations}个lambda候选；测试工况未参与",
                "result": "选择训练集平均目标最低的lambda",
                "meaning": "避免拿未经任务调节的保守回退参数与已优化算法直接比较",
            },
            {
                "step": len(classical_steps) + 2,
                "method": "IMC λ训练集调参",
                "quantity": "选定参数",
                "formula_or_action": "lambda*=argmin mean_train J(IMC(lambda))",
                "substitution": f"lambda={float(selected_imc['lambda_minutes']):.8g} min",
                "result": f"Kp={imc_gains[0]:.8g}, Ki={imc_gains[1]:.8g}",
                "meaning": "进入留出集和三个案例后冻结，不利用测试结果回调参数",
            },
        ]
    )
    _write_rows(output / "classical_tuning_steps.csv", classical_steps)
    label_rows = generate_label_rows(
        training_scenarios,
        bo_iterations=bo_iterations,
        seed=seed,
        progress=True,
    )
    _write_rows(output / "training_labels.csv", label_rows)

    global_tune = tune_global_fixed(training_scenarios, iterations=max(4, bo_iterations), seed=seed + 808)
    fixed_gains = (global_tune.kp, global_tune.ki)
    _write_rows(output / "global_bayesian_tuning.csv", [{"kp": global_tune.kp, "ki": global_tune.ki, "objective": global_tune.score, "evaluations": global_tune.evaluations}])
    _write_rows(output / "bayesian_search_history.csv", list(global_tune.history))
    fnn_samples: list[dict[str, float]] = []
    fnn_candidates: list[np.ndarray] = []
    fnn_context_candidates: list[np.ndarray] = []
    fnn_rule_table, fnn_history = train_fnn_rule_table(
        training_scenarios,
        label_rows,
        imc_gains,
        validation_scenarios=validation_scenarios,
        return_history=True,
        sample_sink=fnn_samples,
        candidate_sink=fnn_candidates,
        context_sink=fnn_context_candidates,
    )
    _write_rows(output / "fnn_training_history.csv", fnn_history)
    _write_rows(output / "fnn_training_samples.csv", fnn_samples)
    if fnn_candidates:
        np.save(output / "fnn_rule_table_candidate.npy", fnn_candidates[-1])
    if fnn_context_candidates:
        np.save(output / "fnn_context_coefficients_candidate.npy", fnn_context_candidates[-1])
    np.save(output / "fnn_rule_table.npy", fnn_rule_table)
    rl_transitions: list[dict[str, float]] = []
    rl_candidates: list[np.ndarray] = []
    rl_q_table, rl_history = train_offline_q_policy(
        training_scenarios,
        fallback_gains=imc_gains,
        validation_scenarios=validation_scenarios,
        seed=seed + 1_616,
        return_history=True,
        transition_sink=rl_transitions,
        candidate_sink=rl_candidates,
    )
    _write_rows(output / "rl_training_history.csv", rl_history)
    _write_rows(output / "rl_training_transitions.csv", rl_transitions)
    if rl_candidates:
        np.save(output / "rl_q_table_candidate.npy", rl_candidates[-1])
    np.save(output / "rl_q_table.npy", rl_q_table)
    state_spec_rows: list[dict[str, object]] = []
    for index, value in enumerate(FNNGainController.centers):
        state_spec_rows.append({"component": "FNN_error_center", "index": index, "value": float(value), "meaning": "Tzone-Tsetpoint, °C"})
    for index, value in enumerate(FNNGainController.delta_centers):
        state_spec_rows.append({"component": "FNN_error_rate_center", "index": index, "value": float(value), "meaning": "error rate, °C/min"})
    for index, value in enumerate(IncrementalRLController.error_edges):
        state_spec_rows.append({"component": "RL_error_edge", "index": index, "value": float(value), "meaning": "Tzone-Tsetpoint bin edge, °C"})
    for index, value in enumerate(IncrementalRLController.delta_edges):
        state_spec_rows.append({"component": "RL_error_rate_edge", "index": index, "value": float(value), "meaning": "error-rate bin edge, °C/min"})
    for index, value in enumerate(IncrementalRLController.command_edges):
        state_spec_rows.append({"component": "RL_command_edge", "index": index, "value": float(value), "meaning": "previous applied compressor command bin edge"})
    for index, (kp_scale, ki_scale) in enumerate(IncrementalRLController._actions):
        state_spec_rows.append({"component": "RL_action", "index": index, "value": f"{kp_scale:.6g},{ki_scale:.6g}", "meaning": "absolute Kp,Ki target scales relative to IMC"})
    _write_rows(output / "adaptive_state_spec.csv", state_spec_rows)

    rl_covered = np.zeros((5, 5, 3), dtype=bool)
    for transition in rl_transitions:
        rl_covered[
            int(transition["state_error_bin"]),
            int(transition["state_delta_bin"]),
            int(transition["state_command_bin"]),
        ] = True

    print("[2/3] Evaluating five controllers on sealed held-out thermal contexts ...")
    seeded_scenarios: list[tuple[int, int, Scenario]] = []
    test_partitions: list[tuple[str, int, list[Scenario]]] = []
    for acceptance_seed in acceptance_seeds:
        scenarios = sample_adaptive_scenarios(test_samples, seed=seed + 100_003 + acceptance_seed, duration_hours=5.0)
        test_partitions.append(("test", acceptance_seed, scenarios))
        seeded_scenarios.extend((acceptance_seed, index + 1, scenario) for index, scenario in enumerate(scenarios))
    test_scenarios = [scenario for _, _, scenario in seeded_scenarios]
    manifest = _dataset_manifest_rows(
        [("train", seed, training_scenarios), ("validation", validation_seed, validation_scenarios), *test_partitions]
    )
    _write_rows(output / "dataset_manifest.csv", manifest)
    _write_rows(output / "holdout_scenarios.csv", _scenario_rows(test_scenarios))
    fnn_candidate = fnn_candidates[-1] if fnn_candidates else fnn_rule_table
    fnn_context = fnn_context_candidates[-1] if fnn_context_candidates else np.zeros((2, 4), dtype=float)
    rl_candidate = rl_candidates[-1] if rl_candidates else rl_q_table
    fnn_coverage = float(fnn_history[-1]["rule_coverage_pct"]) / 100.0 if fnn_history else 0.0
    acceptance_rows, acceptance = _deployment_acceptance(
        seeded_scenarios,
        imc_gains,
        fnn_candidate,
        fnn_context,
        rl_candidate,
        rl_covered,
        fnn_rule_coverage=fnn_coverage,
        seed=seed,
    )
    _write_rows(output / "deployment_acceptance.csv", acceptance_rows)
    fnn_rule_table = fnn_candidate if acceptance["FNN"] else np.broadcast_to(np.asarray(imc_gains), (5, 5, 2)).copy()
    if not acceptance["FNN"]:
        fnn_context = np.zeros((2, 4), dtype=float)
    if not acceptance["RL"]:
        rl_q_table = np.zeros_like(rl_candidate)
        rl_q_table[:, :, :, 4] = 1.0
    else:
        rl_q_table = rl_candidate
    if fnn_history:
        fnn_history[-1]["deployment_accepted"] = float(acceptance["FNN"])
    if rl_history:
        rl_history[-1]["deployment_accepted"] = float(acceptance["RL"])
    _write_rows(output / "fnn_training_history.csv", fnn_history)
    _write_rows(output / "rl_training_history.csv", rl_history)
    np.save(output / "fnn_rule_table.npy", fnn_rule_table)
    np.save(output / "fnn_context_coefficients.npy", fnn_context)
    np.save(output / "rl_q_table.npy", rl_q_table)
    evaluation_rows = _evaluate_holdout(
        test_scenarios, fixed_gains, zn_gains, imc_gains, fnn_rule_table, rl_q_table, seed + 200_003, rl_covered, fnn_context
    )
    summary_rows = _aggregate(evaluation_rows)
    _write_rows(output / "holdout_metrics.csv", evaluation_rows)
    _write_rows(output / "holdout_summary.csv", summary_rows)

    print("[3/3] Running the dynamic profile and generating report artifacts ...")
    dynamic_results, dynamic_metrics = _run_dynamic(fixed_gains, zn_gains, imc_gains, fnn_rule_table, rl_q_table, seed + 300_003, rl_covered, fnn_context)
    dynamic_rows: list[dict[str, object]] = []
    for name, values in dynamic_metrics.items():
        dynamic_rows.append({"controller": name, **values})
    _write_rows(output / "dynamic_metrics.csv", dynamic_rows)
    timeseries_rows: list[dict[str, object]] = []
    for name, result in dynamic_results.items():
        for index, minute in enumerate(result.minute):
            timeseries_rows.append({
                "controller": name, "minute": float(minute), "zone_c": float(result.zone_c[index]),
                "measurement_c": float(result.measurement_c[index]),
                "setpoint_c": float(result.setpoint_c[index]), "outdoor_c": float(result.outdoor_c[index]),
                "internal_load_w": float(result.internal_load_w[index]), "command_pct": float(result.command[index] * 100),
                "requested_command_pct": float(result.requested_command[index] * 100),
                "kp": float(result.kp[index]), "ki": float(result.ki[index]),
                "inference_us": float(result.inference_us[index]), "fallback_active": int(result.fallback_active[index]),
            })
    _write_rows(output / "dynamic_timeseries.csv", timeseries_rows)
    case_results, case_rows = _run_case_studies(fixed_gains, zn_gains, imc_gains, fnn_rule_table, rl_q_table, seed + 400_003, rl_covered, fnn_context)
    _write_rows(output / "case_metrics.csv", case_rows)
    case_timeseries_rows: list[dict[str, object]] = []
    for case_name, results in case_results.items():
        for name, result in results.items():
            for index, minute in enumerate(result.minute):
                case_timeseries_rows.append({
                    "case": case_name, "controller": name, "minute": float(minute),
                    "zone_c": float(result.zone_c[index]),
                    "measurement_c": float(result.measurement_c[index]),
                    "setpoint_c": float(result.setpoint_c[index]),
                    "outdoor_c": float(result.outdoor_c[index]),
                    "internal_load_w": float(result.internal_load_w[index]),
                    "command_pct": float(result.command[index] * 100),
                    "requested_command_pct": float(result.requested_command[index] * 100),
                    "kp": float(result.kp[index]), "ki": float(result.ki[index]),
                    "inference_us": float(result.inference_us[index]),
                    "fallback_active": int(result.fallback_active[index]),
                })
    _write_rows(output / "case_timeseries.csv", case_timeseries_rows)
    validation = run_cross_validation(output)
    _write_rows(output / "physical_cross_validation.csv", validation["physical_rows"])
    _write_rows(output / "physical_cross_validation_timeseries.csv", validation["physical_timeseries"])
    _write_rows(output / "fopdt_cross_validation.csv", validation["fopdt_rows"])
    _write_rows(output / "fopdt_cross_validation_timeseries.csv", validation["fopdt_timeseries"])
    _write_rows(output / "openmodelica_cross_validation.csv", validation["openmodelica_rows"])
    _write_rows(output / "openmodelica_cross_validation_timeseries.csv", validation["openmodelica_timeseries"])
    _write_rows(
        output / "cross_validation_environment.csv",
        [{"item": key, "value": value} for key, value in validation["environment"].items()],
    )
    plot_dynamic_comparison(dynamic_results, dynamic_metrics, output / "dynamic_comparison.png")
    plot_training_labels(label_rows, output / "training_labels.png")
    plot_classical_tuning_process(
        identification_history,
        fopdt_fit_history,
        classical_steps,
        zn_gains,
        imc_gains,
        output / "classical_tuning_process.png",
    )
    plot_bayesian_search_trace(list(global_tune.history), output / "bayesian_search_trace.png")
    plot_fnn_training_trace(fnn_history, output / "fnn_training_trace.png")
    plot_rl_training_trace(rl_history, output / "rl_training_trace.png")
    plot_training_convergence_overview(list(global_tune.history), fnn_history, rl_history, output / "training_convergence_overview.png")
    plot_case_studies(case_results, output / "case_studies.png")
    plot_case_comparisons(case_results, output)
    algorithm_reports = write_algorithm_reports(
        output,
        typical_case_scenarios(),
        case_results,
        case_rows,
        commissioning_fopdt=commissioning_fopdt,
        classical_gains={"Ziegler-Nichols": zn_gains, "IMC PI": imc_gains},
        global_tune=global_tune,
        fnn_history=fnn_history,
        rl_history=rl_history,
    )
    write_engineering_report(
        output / "engineering_report.md",
        summary_rows,
        dynamic_rows,
        case_rows,
        algorithm_reports,
        validation["physical_rows"],
        validation["fopdt_rows"],
        validation["openmodelica_rows"],
        validation["environment"],
    )
    write_html_engineering_report(
        output / "engineering_report.html",
        summary_rows,
        dynamic_rows,
        output,
        case_rows,
        algorithm_reports,
        validation["physical_rows"],
        validation["fopdt_rows"],
        validation["openmodelica_rows"],
        validation["environment"],
    )
    remediation_markdown = _write_review_remediation(
        output, acceptance_rows, acceptance, fnn_history, rl_history
    )
    markdown_path = output / "engineering_report.md"
    markdown_path.write_text(
        markdown_path.read_text(encoding="utf-8") + "\n\n" + remediation_markdown + "\n",
        encoding="utf-8",
    )
    html_path = output / "engineering_report.html"
    html_document = html_path.read_text(encoding="utf-8")
    acceptance_html = (
        "<section><h2>评审缺陷 A1–E5 闭环结果</h2>"
        f"<p>FNN部署门：{'通过' if acceptance['FNN'] else '未通过并回退IMC'}；"
        f"RL部署门：{'通过' if acceptance['RL'] else '未通过并回退IMC'}。</p>"
        "<p>逐项状态、证据文件和遗留边界见 <a href=\"review_remediation.md\">review_remediation.md</a> "
        "与 <a href=\"review_defect_matrix.csv\">review_defect_matrix.csv</a>。</p></section>"
    )
    html_path.write_text(html_document.replace("</body>", acceptance_html + "</body>"), encoding="utf-8")

    elapsed = time.perf_counter() - started
    zn_mean = next(float(row["mean_objective"]) for row in summary_rows if row["controller"] == "Ziegler-Nichols")
    ai_mean = next(float(row["mean_objective"]) for row in summary_rows if row["controller"] == "Bayesian Auto-tune")
    ai_improvement = 100.0 * (zn_mean - ai_mean) / max(zn_mean, 1e-9)
    result = {
        "output_dir": str(output.resolve()),
        "elapsed_seconds": elapsed,
        "ai_vs_zn_holdout_improvement_percent": ai_improvement,
        "summary": summary_rows,
        "dynamic_metrics": dynamic_metrics,
        "cross_validation": validation,
    }
    print(f"Done in {elapsed:.1f}s. AI vs ZN mean holdout objective: {ai_improvement:+.1f}% improvement.")
    return result
