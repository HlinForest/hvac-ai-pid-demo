# RESULTS (E2 当前轮：sealed-80x7-v4)

每个数字可定位到 `run_id + 文件 + 行/列 + git SHA`。E2 当前轮执行 SHA 见
`artifacts/runs/sealed-80x7-v4/sealed_provenance.json:code_sha`（含 `code_dirty`/`worktree_status_sha16` 脏位标记）；E1 数值证据见
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

## E2 主实验：冻结策略统一重评（非统一训练比较）

- v3 80 场景密封排名仅适用于 Z-N/IMC/BO/FNN/RL；Safe BO 与 LLM 不可加入同一排名。
- v4 `embedded_smoke/seven_algorithm_summary.csv`（同一 demo 场景 + 同一 commissioning 门，7 行）仅验证 `--artifact-dir` 管道与统一评价 plumbing，不作为 80 密封排名。
- E2 已完成当前轮：`artifacts/runs/sealed-80x7-v4/`（560 条主测试记录，同 80 场景、同噪声种子、同六子步积分器、同指标；`sealed_80_details.csv:2-561`，`sealed_provenance.json` 记录 experiment_id/code_sha/dirty/manifest/policy 哈希，`sealed_scenarios.csv` 为全场景清单，`manifest.yaml` 为冻结拷贝）。
  - 统计主口径：`mean_ratio_to_imc` = `paired_ratio_mean` = 逐场景配对比均值 `mean(obj/imc)` + bootstrap 95% CI（2000 重采样）；`ratio_of_means_to_imc` 为次统计量（平均目标之比），仅对照。两者不同，不得混用。
  - 验收主口径：`validation_passed = bounded & comfort_held & recovery_ok & actuator_compliant`；`stable` 仅为数值有界（5–45°C），100% 不代表验收通过；`fallback_failed = 有回退且 validation 未通过`。
  - 本轮冻结来源：FNN/RL/IMC/BO 为 v3 冻结产物，Safe-BO 为历史独立批次保守参数（legacy 回退=1），LLM 为 replay 冻结（训练集无接受候选，等于 IMC，非实时调用）。结论限定为“已冻结策略在同一 v4 仿真条件下表现如何”，不得写成统一训练预算比较。

| 算法 | 平均目标 J↓ | 主统计量 配对比均值 [95% CI] | 次统计量 均值之比 | 温控验收通过 | 数值有界 |
|---|---|---|---|---|---|
| Z-N | 61.72 | 1.6520 [1.5427,1.7666] 显著差 | 1.4964 | 6/80 (7.5%) | 80/80 |
| IMC | 41.24 | 1.0000（基准） | 1.0000 | 39/80 (48.8%) | 80/80 |
| BO | 41.95 | 0.9996 [0.9808,1.0193] 与 IMC 相当 | 1.0171 | 38/80 (47.5%) | 80/80 |
| Safe-BO（保守参数） | 36.72 | 0.9185 [0.8863,0.9516] 本轮最优 | 0.8903 | 51/80 (63.8%) | 80/80 |
| FNN | 41.44 | 0.9980 [0.9886,1.0084] 非劣（上界≤1.02），未证显著优 | 1.0048 | 40/80 (50.0%) | 80/80 |
| RL | 39.83 | 0.9817 [0.9620,1.0070] 非劣，CI 含 1，未证显著优 | 0.9658 | 39/80 (48.8%) | 80/80 |
| LLM replay | 41.24 | 1.0000 与 IMC 持平 | 1.0000 | 39/80 (48.8%) | 80/80 |

  - 来源：`sealed_80_summary.csv:2-8`（本轮已修正旧版统计名实颠倒：旧 `paired 0.9664` 实为均值之比，新主统计量 RL 为 0.9817，结论由“显著改善”更正为“未证显著优”）。
  - 失败分析：有界 80/80、执行器合规 80/80 全过；瓶颈在舒适带保持（IMC 44/80、Safe-BO 60/80）与扰动恢复（IMC 44→39、Safe-BO 60→51）。即目标值尚可但未持续在带——回退基准本身在困难场景也不合格，不得称“回退基准本身合格”。RL 4 个 fallback 场景回退后仍未通过（`fallback_failed=4`），FNN/其他固定增益无回退。
  - E3 消融：4 组随机固定增益配对比 3.07–8.00× 更差，保守 IMC 公式 5.89× 更差（`sealed_ablation.csv`）；Safe-BO 当前证据支持该保守参数集，不证明 GP 搜索机制本身（`safe_bo_attribution: evaluation=2/local qualification seed`）。
  - LLM：replay 冻结等于 IMC（`llm_policy.json:accepted=false`）；18 次真实矩阵为独立 E4 证据，不参与 E2 排名。
  - B1 的 legacy 状态为历史：FNN/RL 掩码与 Safe-BO 回退问题已由下轮 B2 统一重训解决（见下节）。
- LLM 矩阵（E4 独立证据）：18 运行 / 18 真实调用 / 18 数值有界稳定（演示 `stable` 口径）/ 5 部署、13 回退（`archive/outputs_llm_matrix_v3/matrix_summary.csv:2–19`）。回退≠验收通过；跨协议不得与 E2 同排名。

## E2-B2：统一重训后重评（统一训练预算比较，单一种子单预算）

- 训练冻结：`artifacts/runs/v4-20260907-8f3fd5f/`（`run.py pipeline` 48/16/16，约 227 s；`bo/safe_bo_policy.json` + `rl_covered_mask.npy` 冻结，
  `bo_legacy_fallback=0/safe_bo_legacy_fallback=0`，FNN/RL 部署验收均通过；大体积渲染物已按 `PRUNED.md` 裁剪，再生命令见该文件）。
- 重评：`artifacts/runs/sealed-80x7-v4-retrain/`（同 80 场景/种子/积分器/指标；一致性校验通过）。

| 算法 | 平均目标 J↓ | 主统计量 配对比均值 [95% CI] | 温控验收通过 | 读法 |
|---|---|---|---|---|
| Z-N | 61.72 | 1.6509 [1.5407,1.7630] | 6/80 (7.5%) | 显著差 |
| IMC | 41.24 | 1.0000（基准） | 40/80 (50.0%) | 基准 |
| BO | 45.02 | 1.0785 [1.0424,1.1146] | 33/80 (41.2%) | 显著差于 IMC（14 评估过拟合，同预算不可靠） |
| Safe-BO | 41.24 | 1.0000 等于 IMC | 40/80 (50.0%) | 本轮选中基线本身（`evaluation=1/known baseline`），无增益无损失 |
| FNN | 41.17 | 0.9920 [0.9762,1.0102] | 41/80 (51.2%) | 与 IMC 持平 |
| RL | 39.80 | 0.9801 [0.9611,1.0041] | 39/80 (48.8%) | 与 IMC 持平，CI 含 1 |
| LLM replay | 41.24 | 1.0000 与 IMC 持平 | 40/80 (50.0%) | 等于 IMC，非实时调用 |

  - 来源：`sealed-80x7-v4-retrain/sealed_80_summary.csv:2-8`。B2 阴性结果如实保留：统一预算下无任何方法显著优于 IMC。
  - E3+ 同预算归因（`gp_attribution.csv:2-6`，同 48 训练谱选型、同 80 密封评价）：
    BO-full 密封 45.02（1.0785）/ 初值最优 46.30（1.1567）/ 纯随机14 49.52（1.3266）/ 保守公式 250.23（7.98×）。
    GP 相对贡献为正（训练均值好 6.0%，密封配对比好约 6.8%，比纯随机好约 18.7%），但绝对值仍不如 IMC——“GP 机制有效，但本预算下不足以超越调好的 IMC”。

## 硬件门与交付物

- 目标三门（实体遥测/目标 WCET/HIL）：3/3 PENDING（`python tools/check_hardware_gates.py --run artifacts/runs/v4-20260907-8f3fd5f`），不得写成已完成。
- 软件侧：PC-SIL PASS（`mcu_pc_sil_log.txt`，75 组对拍，CRC 一致）；主机 WCET 参考（`host_wcet.csv`，HOST-ONLY）；
  ESP32 本机刷新失败（用户路径中文致 xtensa 工具链失效，`esp32_target_compile_attempt.log` 存证）。
- 主报告 DOCX：`docs/主报告.docx`（`reports/md2docx.py` 生成，`python tools/check_docx.py` 逐节 ALL OK）。

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
