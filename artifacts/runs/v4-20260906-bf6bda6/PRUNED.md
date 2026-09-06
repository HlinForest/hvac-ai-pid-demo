# v4-20260906-bf6bda6：体积裁剪说明（证据完整性不受影响）

完整流水线原始产物约 35 MB（含 147 个文件）；为遵守仓库 100 MB 跟踪预算，删除以下**可由同一命令完整再生**的文件，
保留全部冻结策略、选型历史、数据划分（含新增资格集）、汇总指标与溯源（约 0.5 MB）：

- 删除：`rl_training_transitions.csv`（逐回合中间量；覆盖掩码已冻结为 `rl_covered_mask.npy`）、
  `classical_tuning_history.csv`（逐候选中间量；蒸馏标签见 `training_labels.csv`）、
  `fnn_training_samples.csv`、`case_timeseries.csv`、`dynamic_timeseries.csv`、
  `physical/fopdt_cross_validation_timeseries.csv`、`engineering_report.html`、
  `algorithm_reports/`、`report_assets/`（KaTeX 拷贝）、全部 `*.png`。
- 保留：`manifest.yaml`、`environment.json`、`provenance.csv`、`bo_policy.json`、
  `global_bayesian_tuning.csv`、`safe_bo_policy.json`、`safe_bo_history.csv`、
  `imc_lambda_tuning.csv`、`fnn_rule_table*.npy`、`fnn_context_coefficients*.npy`、
  `rl_q_table*.npy`、`rl_covered_mask.npy`、`fnn/rl_training_history.csv`、
  `training/validation/qualification_scenarios.csv`、`dataset_manifest.csv`（跨划分泄漏检查）、
  `holdout_metrics/summary.csv`、`deployment_acceptance.csv`（资格集 32 行）、
  `case/dynamic_metrics.csv`、`physical/fopdt_cross_validation.csv`、
  `host_wcet.csv/json`、`mcu_pc_sil_log.txt`、`mcu_validation_summary.csv`、
  `engineering_report.md`、`review_defect_matrix.csv`、`review_remediation.md`。
- `SHA256SUMS` + `evidence_index.csv` 为**裁剪前完整 147 文件**的哈希索引（含已删文件）；
  按下述命令重跑可恢复全部文件并逐项验哈希。

再生命令（约 200 s + 后续评价）：

```bash
python run.py pipeline --output artifacts/runs/v4-20260906-bf6bda6 --seed 7
python embedded/run_mcu_validation.py --artifact-dir artifacts/runs/v4-20260906-bf6bda6
python tools/run_host_wcet.py --artifact-dir artifacts/runs/v4-20260906-bf6bda6
python run.py sealed --artifact-dir artifacts/runs/v4-20260906-bf6bda6 --output artifacts/runs/sealed-80x7-b3 --protocol-label E2-B3 --design-note "..."
python tools/run_gp_attribution.py --artifact-dir artifacts/runs/v4-20260906-bf6bda6 --output artifacts/runs/sealed-80x7-b3/gp_attribution.csv
python tools/make_budget_table.py --run artifacts/runs/v4-20260906-bf6bda6 --sealed artifacts/runs/sealed-80x7-b3
python tools/analyze_failures.py --sealed artifacts/runs/sealed-80x7-b3
python tools/make_sealed_figures.py --artifact-dir artifacts/runs/v4-20260906-bf6bda6 --sealed artifacts/runs/sealed-80x7-b3
```
