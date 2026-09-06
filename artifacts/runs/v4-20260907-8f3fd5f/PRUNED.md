# v4-20260907-8f3fd5f：体积裁剪说明（证据完整性不受影响）

完整流水线原始产物约 34 MB；为遵守仓库 100 MB 跟踪预算，删除以下**可由同一命令完整再生**的文件，
保留全部冻结策略、选型历史、数据划分、汇总指标与溯源（42 个文件，约 0.5 MB）：

- 删除：`rl_training_transitions.csv`（7.7 MB，逐回合中间量；覆盖掩码已冻结为 `rl_covered_mask.npy`）、
  `classical_tuning_history.csv`（3.6 MB，逐候选中间量；蒸馏标签见 `training_labels.csv`）、
  `fnn_training_samples.csv`、`case_timeseries.csv`、`dynamic_timeseries.csv`、
  `physical/fopdt_cross_validation_timeseries.csv`、`engineering_report.html`、
  `algorithm_reports/`、`report_assets/`（KaTeX 拷贝）、全部 `*.png`。
- 保留：`manifest.yaml`、`environment.json`、`provenance.csv`、`bo_policy.json`、
  `global_bayesian_tuning.csv`、`safe_bo_policy.json`、`safe_bo_history.csv`、
  `imc_lambda_tuning.csv`、`fnn_rule_table*.npy`、`fnn_context_coefficients*.npy`、
  `rl_q_table*.npy`、`rl_covered_mask.npy`、`fnn/rl_training_history.csv`、
  `training/validation/holdout_scenarios.csv`、`dataset_manifest.csv`、
  `holdout_metrics/summary.csv`、`deployment_acceptance.csv`、
  `case/dynamic_metrics.csv`、`physical/fopdt_cross_validation.csv`、
  `timestep_convergence.csv`（若有）、`host_wcet.csv/json`、`mcu_pc_sil_log.txt`、
  `mcu_validation_summary.csv`、`esp32_target_compile_attempt.log`、
  `engineering_report.md`、`review_defect_matrix.csv`、`review_remediation.md`。

再生命令（约 227 s）：

```bash
python run.py pipeline --output artifacts/runs/v4-20260907-8f3fd5f --seed 7
python embedded/run_mcu_validation.py --artifact-dir artifacts/runs/v4-20260907-8f3fd5f
python tools/run_host_wcet.py --artifact-dir artifacts/runs/v4-20260907-8f3fd5f
python run.py sealed --artifact-dir artifacts/runs/v4-20260907-8f3fd5f --output artifacts/runs/sealed-80x7-v4-retrain
python tools/run_gp_attribution.py --artifact-dir artifacts/runs/v4-20260907-8f3fd5f --output artifacts/runs/sealed-80x7-v4-retrain/gp_attribution.csv
```
