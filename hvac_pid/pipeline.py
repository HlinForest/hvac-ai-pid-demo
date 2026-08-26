from __future__ import annotations

import csv
from dataclasses import asdict
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
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, scenario in enumerate(scenarios):
        methods = {
            "Bayesian Auto-tune": PIController(*fixed_gains),
            "Ziegler-Nichols": zn_gains,
            "IMC PI": imc_gains,
            "FNN Self-tuning PI": FNNGainController(imc_gains, rule_table=fnn_rule_table),
            "RL Self-tuning PI": IncrementalRLController(imc_gains, q_table=rl_q_table),
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
) -> dict[str, PIController]:
    return {
        "Bayesian Auto-tune": PIController(*fixed_gains),
        "Ziegler-Nichols": PIController(*zn_gains),
        "IMC PI": PIController(*imc_gains),
        "FNN Self-tuning PI": FNNGainController(imc_gains, rule_table=fnn_rule_table),
        "RL Self-tuning PI": IncrementalRLController(imc_gains, q_table=rl_q_table),
    }


def _run_dynamic(
    fixed_gains: tuple[float, float], zn_gains: tuple[float, float], imc_gains: tuple[float, float], fnn_rule_table: np.ndarray, rl_q_table: np.ndarray, seed: int
) -> tuple[dict[str, SimulationResult], dict[str, dict[str, float]]]:
    scenario = dynamic_demo_scenario()
    controllers = _controllers_for_scenario(scenario, fixed_gains, zn_gains, imc_gains, fnn_rule_table, rl_q_table)
    results = {name: simulate(scenario, controller, seed=seed) for name, controller in controllers.items()}
    metrics = {name: calculate_metrics(result) for name, result in results.items()}
    return results, metrics


def _run_case_studies(
    fixed_gains: tuple[float, float], zn_gains: tuple[float, float], imc_gains: tuple[float, float], fnn_rule_table: np.ndarray, rl_q_table: np.ndarray, seed: int
) -> tuple[dict[str, dict[str, SimulationResult]], list[dict[str, object]]]:
    all_results: dict[str, dict[str, SimulationResult]] = {}
    rows: list[dict[str, object]] = []
    for case_index, (case_name, scenario) in enumerate(typical_case_scenarios().items()):
        controllers = _controllers_for_scenario(scenario, fixed_gains, zn_gains, imc_gains, fnn_rule_table, rl_q_table)
        results = {name: simulate(scenario, controller, seed=seed + case_index) for name, controller in controllers.items()}
        all_results[case_name] = results
        for name, result in results.items():
            rows.append({"case": case_name, "controller": name, **calculate_metrics(result)})
    return all_results, rows


def run_pipeline(
    output_dir: str | Path,
    *,
    train_samples: int = 48,
    test_samples: int = 16,
    bo_iterations: int = 7,
    seed: int = 7,
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
    training_scenarios = sample_adaptive_scenarios(train_samples, seed=seed, duration_hours=5.0)
    _write_rows(output / "training_scenarios.csv", _scenario_rows(training_scenarios))
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
    fnn_rule_table, fnn_history = train_fnn_rule_table(
        training_scenarios,
        label_rows,
        imc_gains,
        return_history=True,
        sample_sink=fnn_samples,
        candidate_sink=fnn_candidates,
    )
    _write_rows(output / "fnn_training_history.csv", fnn_history)
    _write_rows(output / "fnn_training_samples.csv", fnn_samples)
    if fnn_candidates:
        np.save(output / "fnn_rule_table_candidate.npy", fnn_candidates[-1])
    np.save(output / "fnn_rule_table.npy", fnn_rule_table)
    rl_transitions: list[dict[str, float]] = []
    rl_candidates: list[np.ndarray] = []
    rl_q_table, rl_history = train_offline_q_policy(
        training_scenarios,
        fallback_gains=imc_gains,
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

    print("[2/3] Evaluating five controllers on held-out thermal contexts ...")
    test_scenarios = sample_adaptive_scenarios(test_samples, seed=seed + 100_003, duration_hours=5.0)
    _write_rows(output / "holdout_scenarios.csv", _scenario_rows(test_scenarios))
    evaluation_rows = _evaluate_holdout(test_scenarios, fixed_gains, zn_gains, imc_gains, fnn_rule_table, rl_q_table, seed + 200_003)
    summary_rows = _aggregate(evaluation_rows)
    _write_rows(output / "holdout_metrics.csv", evaluation_rows)
    _write_rows(output / "holdout_summary.csv", summary_rows)

    print("[3/3] Running the dynamic profile and generating report artifacts ...")
    dynamic_results, dynamic_metrics = _run_dynamic(fixed_gains, zn_gains, imc_gains, fnn_rule_table, rl_q_table, seed + 300_003)
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
    case_results, case_rows = _run_case_studies(fixed_gains, zn_gains, imc_gains, fnn_rule_table, rl_q_table, seed + 400_003)
    _write_rows(output / "case_metrics.csv", case_rows)
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
