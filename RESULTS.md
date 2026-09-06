# RESULTS (v4-20260906-2870262)

每个数字可定位到 `run_id + 文件 + 行/列 + git SHA(2870262)`。完整溯源见
`artifacts/runs/v4-20260906-2870262/provenance.csv`。

## 数值交叉验证（v4 已全绿）

| case | samples | rmse (°C) | max (°C) | threshold | passed | 来源 |
|---|---|---:|---:|---|---|---|
| 初次快速降温 | 241 | 0.016672 | 0.025154 | ≤0.15 | 1 | `physical_cross_validation.csv:2` |
| 设定温度突变 | 301 | 0.008150 | 0.012704 | ≤0.15 | 1 | `physical_cross_validation.csv:3` |
| 持续外界热扰动 | 361 | 0.001820 | 0.003174 | ≤0.15 | 1 | `physical_cross_validation.csv:4` |

v3 同表第 2 行为 `max=0.150350, passed=0`；v4 物理推进改为 10 s×6 子步进后通过，
阈值未动。收敛证据见 `timestep_convergence.csv`。

## FOPDT 代理验证（v4 已全绿）

| case | normalized RMSE | threshold | passed | 来源 |
|---|---|---|---|---|
| 初次快速降温 | 11.95% | ≤15% | 1 | `fopdt_cross_validation.csv:2` |
| 设定温度突变 | 11.92% | ≤15% | 1 | `fopdt_cross_validation.csv:3` |
| 持续外界热扰动 | 11.88% | ≤15% | 1 | `fopdt_cross_validation.csv:4` |

## 七算法可比性声明

- v3 80 场景密封排名仅适用于 Z-N/IMC/BO/FNN/RL；Safe BO 与 LLM 不可加入同一排名。
- v4 `embedded_smoke/seven_algorithm_summary.csv`（同一 demo 场景 + 同一 commissioning 门，7 行）仅验证 `--artifact-dir` 管道与统一评价 plumbing，不作为 80 密封排名。
- v4 统一 `80×7` 密封已完成首轮：`artifacts/runs/sealed-80x7-v4/`（560 条主测试记录，同 80 场景、同噪声种子、同六子步积分器、同指标；`sealed_80_details.csv:2-561`，`sealed_provenance.json` 记录 code/manifest/policy 哈希）。
  - 统计口径：`mean_ratio_to_imc` 为均值之比，`paired_ratio_*` 为逐场景配对目标比均值 + bootstrap 95% CI；两者不同，不得混用。
  - 本轮（冻结策略来自 `archive/outputs_review_v3` + `archive/outputs_advanced_quick`，v4 物理）：
    - Z-N 1.4995 [1.4105,1.5954] 显著差于 IMC；
    - BO 1.0168 [0.9975,1.0375] 不能称优于 IMC；
    - Safe-BO 0.8911 [0.8597,0.9214] 显著优于 IMC，但选中为 `evaluation=2/local qualification seed (0.3838/0.00444)`，证明的是该保守参数集，尚不能归因于 GP 搜索机制（见 `sealed_provenance.json:safe_bo_attribution` + `sealed_ablation.csv` 随机/保守对照 3–8× 更差）；
    - FNN 1.0046 [0.9964,1.0133] 非劣于 IMC（上界 1.0133 ≤1.02），不足以称显著优于 IMC；
    - RL 0.9664 [0.9434,0.9917] 在本协议内改善（含覆盖掩码一致加载，fallback 5% 场景）；
    - LLM(replay-frozen) 1.0000 与 IMC 持平（训练集无接受候选，回退 IMC；`llm_policy.json:accepted=false`），非实时模型调用；18 次真实矩阵仍为独立证据（`archive/outputs_llm_matrix_v3`，15/18 stable 按演示判据，3×hard_r73 回退后仍 stable=0 已拆为 `fallback_failed`，不得直接解释为数学发散或全部验收通过）。
  - 完整 v4 48/16/16 重训 FNN/RL + SafeBO/LLM 同清单冻结仍待算力；新 `run.py sealed` + `PolicyBundle` 已消除隐式归档依赖，新流水线（`bo_policy.json`/`safe_bo_policy.json`/`rl_covered_mask.npy`/`provenance.csv`）不再触发 legacy 回退。
- LLM 矩阵：18 运行 / 18 真实 / 5 部署（`archive/outputs_llm_matrix_v3/matrix_summary.csv:2–19`，`AGGREGATE_REPORT.md:3–6`）。

## 工程与依赖

- pytest：39 passed（`tests/` 8 文件，`def test_` 39 个；文档曾误写 25/36，已以实测为准）。
- CI：`.github/workflows/ci.yml`（pytest + 路径检查 + 快速实验 + 报告一致性；numpy 1.26.4/2.0.0 矩阵）。
- 依赖：`requirements.txt` 已放宽 `numpy>=1.24`（无 `<2` 上限），v4 基线在 1.26.4 验证；`pyproject.toml` 新增（requires-python ≥3.10）。
- 时间口径：仿真 300 s / 演示 120 s / MCU 2 s+100 ms（`hvac_pid/timebase.py`）；安全界 ±10%（`hvac_pid/safety.py`）。
- 旧结论→新结论对照：
  - 数值验证“1 行失败”→“3/3 通过（物理改变，非阈值放宽）”；
  - “仅 2 次 LLM 真实会话”→“18 次真实、5/18 部署”；
  - “七算法统一排名”→“仅 5 算法可比，SafeBO/LLM 标记不可比较”；
  - “Modelica 已实际运行/未运行矛盾”→“历史证据，当前未复现”；
  - “更新周期统一 2 s / 增益 25%”→“300 s/120 s/2 s 三口径 + 10% 信任域”。
