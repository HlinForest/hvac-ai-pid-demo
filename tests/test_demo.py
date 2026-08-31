from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hvac_pid.config import Scenario, sample_adaptive_scenarios, sample_scenarios
from hvac_pid.controllers import PIController, identify_fopdt, identify_fopdt_audit, imc_pi, ziegler_nichols_pi
from hvac_pid.ai_controllers import FNNGainController, IncrementalRLController, train_fnn_rule_table, train_offline_q_policy
from hvac_pid.actuator import CompressorCommandLimiter
from hvac_pid.pipeline import run_pipeline
from hvac_pid.plant import ThermalPlant2R2C
from hvac_pid.scheduler import GainScheduler
from hvac_pid.tuning import GainBounds, generate_label_rows


class PlantAndControllerTests(unittest.TestCase):
    def test_more_cooling_reduces_zone_temperature(self) -> None:
        scenario = Scenario(
            duration_hours=1.0,
            initial_zone_c=30.0,
            outdoor_c=30.0,
            internal_load_w=0.0,
            actuator_delay_minutes=0.0,
            actuator_tau_minutes=1.0,
        )
        plant_off = ThermalPlant2R2C(scenario)
        plant_on = ThermalPlant2R2C(scenario)
        for _ in range(60):
            plant_off.step(0.0, 30.0, 0.0)
            plant_on.step(0.6, 30.0, 0.0)
        self.assertLess(plant_on.zone_c, plant_off.zone_c - 2.0)

    def test_pi_output_is_bounded_and_integrator_does_not_run_away(self) -> None:
        controller = PIController(kp=2.0, ki=1.0)
        outputs = [controller.update(10.0, 1.0) for _ in range(500)]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in outputs))
        self.assertAlmostEqual(controller.integral, 0.0)

    def test_classical_tunings_are_positive_and_bounded(self) -> None:
        model = identify_fopdt(Scenario())
        bounds = GainBounds()
        for kp, ki in (ziegler_nichols_pi(model), imc_pi(model)):
            self.assertGreaterEqual(kp, bounds.kp[0])
            self.assertLessEqual(kp, bounds.kp[1])
            self.assertGreaterEqual(ki, bounds.ki[0])
            self.assertLessEqual(ki, bounds.ki[1])

    def test_fopdt_identification_is_not_censored_at_the_test_horizon(self) -> None:
        _, response, fit_history = identify_fopdt_audit(Scenario())
        first = response[0]
        self.assertGreater(len(fit_history), 1)
        self.assertGreater(first["final_measured_fraction"], 0.95)
        self.assertLess(first["t63_minutes"], response[-1]["minute"])
        self.assertLess(first["fit_normalized_rmse_pct"], 15.0)

    def test_online_controllers_are_bounded_and_diagnostic(self) -> None:
        fallback = imc_pi(identify_fopdt(Scenario()))
        scenarios = sample_scenarios(3, seed=5, duration_hours=1.0)
        labels = generate_label_rows(scenarios, bo_iterations=1, seed=5, progress=False)
        fnn_table = train_fnn_rule_table(scenarios, labels, fallback)
        rl_table = train_offline_q_policy(scenarios, fallback, seed=5, episodes=8, horizon=5)
        self.assertEqual(fnn_table.shape, (5, 5, 2))
        self.assertEqual(rl_table.shape, (5, 5, 3, 9))
        for controller in (FNNGainController(fallback, rule_table=fnn_table), IncrementalRLController(fallback, q_table=rl_table)):
            controller.reset()
            for minute in range(5):
                output = controller.update(3.0, 1.0, minute=float(minute))
                self.assertTrue(0.0 <= output <= 1.0)
            status = controller.diagnostics()
            self.assertTrue(0.002 <= float(status["kp"]) <= 1.5)
            self.assertTrue(1e-5 <= float(status["ki"]) <= 0.08)

    def test_adaptive_curriculum_contains_both_error_signs_and_events(self) -> None:
        scenarios = sample_adaptive_scenarios(18, seed=17, duration_hours=5.0)
        self.assertTrue(any(item.initial_zone_c < item.setpoint_c for item in scenarios))
        self.assertTrue(any(item.initial_zone_c > item.setpoint_c for item in scenarios))
        self.assertTrue(any(item.setpoint_change_hour is not None for item in scenarios))
        self.assertTrue(any(item.door_open_hour is not None for item in scenarios))

    def test_rl_no_change_policy_preserves_fallback_gains(self) -> None:
        fallback = (0.4, 0.004)
        q = np.zeros((5, 5, 3, 9), dtype=float)
        q[:, :, :, 4] = 1.0
        controller = IncrementalRLController(fallback, q_table=q)
        controller.reset()
        for minute in range(11):
            controller.update(2.0, 1.0, minute=float(minute))
        self.assertAlmostEqual(controller.kp, fallback[0])
        self.assertAlmostEqual(controller.ki, fallback[1])

    def test_compressor_constraints_apply_quantisation_floor_slew_and_dwell(self) -> None:
        scenario = Scenario(
            command_slew_rate_per_minute=0.05,
            command_quantization=0.01,
            minimum_running_command=0.25,
            minimum_on_minutes=3.0,
            minimum_off_minutes=2.0,
        )
        limiter = CompressorCommandLimiter(scenario)
        commands = [limiter.update(1.0, 1.0) for _ in range(5)]
        self.assertTrue(np.allclose(commands[:5], [0.30, 0.35, 0.40, 0.45, 0.50]))
        self.assertTrue(all(value == 0.0 or value >= 0.25 for value in commands))
        self.assertTrue(all(np.isclose(value / 0.01, round(value / 0.01)) for value in commands))
        limiter.reset()
        self.assertEqual(limiter.update(0.30, 1.0), 0.30)
        self.assertEqual(limiter.update(0.0, 1.0), 0.25)
        self.assertEqual(limiter.update(0.0, 1.0), 0.25)
        self.assertEqual(limiter.update(0.0, 1.0), 0.0)


class LearningTests(unittest.TestCase):
    def test_scheduler_predicts_finite_in_range_gains(self) -> None:
        scenarios = sample_scenarios(8, seed=11, duration_hours=2.0)
        rows = generate_label_rows(scenarios, bo_iterations=1, seed=11, progress=False)
        scheduler = GainScheduler(seed=11).fit(rows)
        kp, ki, _ = scheduler.predict_gains(scenarios[0].context_vector())
        self.assertTrue(np.isfinite(kp) and np.isfinite(ki))
        self.assertTrue(0.002 <= kp <= 1.5)
        self.assertTrue(1e-5 <= ki <= 0.08)

    def test_end_to_end_pipeline_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            run_pipeline(
                output, train_samples=8, validation_samples=3, test_samples=3,
                bo_iterations=1, seed=19, acceptance_seeds=(101,),
            )
            expected = {
                "training_labels.csv",
                "training_scenarios.csv",
                "validation_scenarios.csv",
                "dataset_manifest.csv",
                "deployment_acceptance.csv",
                "classical_tuning_history.csv",
                "classical_tuning_steps.csv",
                "fopdt_fit_history.csv",
                "bayesian_search_history.csv",
                "fnn_training_history.csv",
                "fnn_training_samples.csv",
                "rl_training_history.csv",
                "rl_training_transitions.csv",
                "adaptive_state_spec.csv",
                "global_bayesian_tuning.csv",
                "imc_lambda_tuning.csv",
                "fnn_rule_table.npy",
                "fnn_rule_table_candidate.npy",
                "fnn_context_coefficients.npy",
                "fnn_context_coefficients_candidate.npy",
                "rl_q_table.npy",
                "rl_q_table_candidate.npy",
                "holdout_scenarios.csv",
                "holdout_metrics.csv",
                "holdout_summary.csv",
                "dynamic_metrics.csv",
                "dynamic_timeseries.csv",
                "case_metrics.csv",
                "engineering_report.md",
                "engineering_report.html",
                "algorithm_reports",
                "report_assets",
                "dynamic_comparison.png",
                "case_studies.png",
                "training_labels.png",
                "classical_tuning_process.png",
                "bayesian_search_trace.png",
                "fnn_training_trace.png",
                "rl_training_trace.png",
                "training_convergence_overview.png",
                "case_initial_cooling.png",
                "case_setpoint_step.png",
                "case_sustained_heat.png",
                "physical_cross_validation.csv",
                "physical_cross_validation_timeseries.csv",
                "fopdt_cross_validation.csv",
                "fopdt_cross_validation_timeseries.csv",
                "cross_validation_environment.csv",
                "physical_cross_validation.png",
                "fopdt_cross_validation.png",
                "review_defect_matrix.csv",
                "review_remediation.md",
            }
            self.assertEqual(expected, {item.name for item in output.iterdir()})
            reports = output / "algorithm_reports"
            scenario_suffixes = {
                "initial_cooling", "setpoint_step", "sustained_heat"
            }
            expected_reports = {
                "01_zn_reaction_curve.html",
                "02_imc.html",
                "03_bayesian_auto_tune.html",
                "04_fnn_self_tuning.html",
                "05_rl_self_tuning.html",
            }
            for stem in (
                "01_zn_reaction_curve",
                "02_imc",
                "03_bayesian_auto_tune",
                "04_fnn_self_tuning",
                "05_rl_self_tuning",
            ):
                expected_reports.update(
                    f"{stem}_{suffix}.png" for suffix in scenario_suffixes
                )
            self.assertEqual(
                expected_reports,
                {item.name for item in reports.iterdir()},
            )


if __name__ == "__main__":
    unittest.main()
