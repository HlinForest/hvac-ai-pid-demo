# v4-20260906-bf6bda6：体积裁剪说明（部分原始证据未随仓库保存）

完整流水线原始产物约 35 MB（含 147 个文件）；为遵守仓库 100 MB 跟踪预算，删除以下**可由同一命令再生语义等价文件**的文件，
仓库内保留全部冻结策略、选型历史、数据划分（含新增资格集）、汇总指标与溯源（约 0.5 MB）。
注意：`SHA256SUMS` 只是被删文件在删除前的指纹，不能恢复文件本身；计时、绝对路径、运行环境等不保证逐字节重现。
重跑得语义等价文件（冻结策略数值、汇总均值/CI 在容差内一致），已删文件的逐字节哈希不保证重现；如需逐字节验收，请索取完整产物压缩包（以其 SHA-256 为准），勿用重跑文件冒充原始文件。
缺失部分：已删文件本身未随仓库保存（见下述删除清单），仅保留其删除前哈希；`engineering_report.html`、`*.png` 等渲染物亦未保留。

完整流水线原始产物约 35 MB（含 147 个文件）；为遵守仓库 100 MB 跟踪预算，删除以下**可由同一命令再生语义等价文件**的文件，
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
- `SHA256SUMS` + `evidence_index.csv` 为**裁剪前完整 147 文件删除前的哈希索引**（含已删文件）；
  仅对仓库内保留文件可逐项验哈希（`sha256sum -c SHA256SUMS` 会对已删文件报缺失，属预期行为）；重跑不保证恢复已删文件的逐字节哈希。

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
