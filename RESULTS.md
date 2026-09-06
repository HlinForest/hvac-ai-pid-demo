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

## E2-B2：统一重训后重评（历史；部署门用了密封集，独立性有缺陷）

- 训练冻结：`artifacts/runs/v4-20260907-8f3fd5f/`（`run.py pipeline` 48/16/16，约 227 s；`bo/safe_bo_policy.json` + `rl_covered_mask.npy` 冻结，
  `bo_legacy_fallback=0/safe_bo_legacy_fallback=0`；大体积渲染物已按 `PRUNED.md` 裁剪，再生命令见该文件）。
- 重评：`artifacts/runs/sealed-80x7-v4-retrain/`（同 80 场景/种子/积分器/指标；一致性校验通过）。
- 保留为历史对照；当前主结论以 E2-B3 为准。B2 的 E3+ 旧归因（噪声种子与 BO 训练不一致）已作废，由 B3 严格对照取代。

| 算法 | 平均目标 J↓ | 主统计量 配对比均值 [95% CI] | 温控验收通过 | 读法 |
|---|---|---|---|---|
| Z-N | 61.72 | 1.6509 [1.5407,1.7630] | 6/80 (7.5%) | 显著差 |
| IMC | 41.24 | 1.0000（基准） | 40/80 (50.0%) | 基准 |
| BO | 45.02 | 1.0785 [1.0424,1.1146] | 33/80 (41.2%) | 显著差于 IMC；原因可能涉及搜索预算、初始化及训练评分条件，尚未完成归因（禁止径称“过拟合”） |
| Safe-BO | 41.24 | 1.0000 等于 IMC | 40/80 (50.0%) | 本轮选中基线本身（`evaluation=1/known baseline`），无增益无损失 |
| FNN | 41.17 | 0.9920 [0.9762,1.0102] | 41/80 (51.2%) | 与 IMC 持平 |
| RL | 39.80 | 0.9801 [0.9611,1.0041] | 39/80 (48.8%) | 未发现显著优势（CI 含 1）；非劣成立（上界 1.0041 ≤ 预设 1.02 界） |
| LLM replay | 41.24 | 1.0000 与 IMC 持平 | 40/80 (50.0%) | 等于 IMC，非实时调用 |

  - 来源：`sealed-80x7-v4-retrain/sealed_80_summary.csv:2-8`。B2 阴性结果如实保留。
  - E3+ 旧归因已作废（训练噪声与 BO 不一致，追加预算/初始化/随机性混杂），结论以 B3 严格对照为准。

## E2-B3：独立密封重评（历史；80 场景与 B2 完全复用，新结论以 B4 为准）

- 训练冻结：`artifacts/runs/v4-20260906-bf6bda6/`（48 训练/16 验证/16 资格/80 密封四向隔离，部署门仅用资格集：FNN 未通过→回退 IMC，RL 通过；约 198 s；
  `SHA256SUMS` 为裁剪前完整 147 文件索引，再生命令见 `PRUNED.md`）。
- 重评：`artifacts/runs/sealed-80x7-b3/`（`experiment_id=E2-B3`；LLM replay 在完整 48 训练集冻结；一致性校验通过）。
- 预算台账：`sealed-80x7-b3/budget_table.csv`（各方法训练场景数/候选评估数/闭环仿真数/等效对象小时；只有 BO/随机臂可用“同预算”措辞）。

| 算法 | 平均目标 J↓ | 主统计量 配对比均值 [95% CI] | 温控验收通过 | 读法 |
|---|---|---|---|---|
| Z-N | 61.72 | 1.6509 [1.5407,1.7630] | 6/80 (7.5%) | 显著差 |
| IMC | 41.24 | 1.0000（基准） | 40/80 (50.0%) | 基准 |
| BO | 45.02 | 1.0785 [1.0424,1.1146] | 33/80 (41.2%) | 显著差于 IMC；原因尚未完成归因（预算/初始化/评分条件均可能） |
| Safe-BO | 41.24 | 1.0000 等于 IMC | 40/80 (50.0%) | 本轮选中基线本身，无增益无损失 |
| FNN | 41.24 | 1.0000 等于 IMC | 40/80 (50.0%) | 资格集部署门未通过→回退 IMC（独立性修复后门开始起作用的实例） |
| RL | 39.80 | 0.9801 [0.9611,1.0041] | 39/80 (48.8%) | 未发现显著优势（CI 含 1）；非劣成立（上界 ≤1.02） |
| LLM replay | 41.24 | 1.0000 与 IMC 持平 | 40/80 (50.0%) | 完整 48 训练集冻结仍无接受候选，等于 IMC；非实时调用 |

  - 来源：`sealed-80x7-b3/sealed_80_summary.csv:2-8`。
  - E3+ 严格对照（`gp_attribution.csv` 12 行 + `gp_attribution_summary.json`；同初值同噪声，4 搜索种子）：
    朴素混合池 GP−随机 −0.0057 [−0.0373,+0.0284]；**双向聚类（PRIMARY）GP−随机 −0.0057 [−0.1040,+0.1335]（含 0）**，
    GP−初值聚类 −0.1129 [−0.2083,−0.0301]。读法：**未发现 GP 相对同预算随机的优势，机制贡献未经证实**（仅 4 次搜索重复）。
  - 失败初筛（`failure_cause_summary.csv`）：7 个场景全算法失败（1 疑似容量 + 6 疑似窗口）；
    IMC 系 33 个疑似调参残余。初筛不是因果，见 B4 控制实验验证。

## E2-B4：全新密封集复评（当前主实验；一次评价，不重训）

- 新场景：`acceptance_seeds=601,611,621,631,641`，与训练/验证/资格/旧密封零摘要重叠（已验）；
  冻结 B3 策略评价一次。`artifacts/runs/sealed-80x7-b4/`（`experiment_id=E2-B4`，0.81 MB 全量入库，`SHA256SUMS` 索引）。

| 算法 | 平均目标 J↓ | 主统计量 配对比均值 [95% CI] | 温控验收通过 | 读法 |
|---|---|---|---|---|
| Z-N | 59.86 | 1.7803 [1.6208,1.9623] | 2/80 (2.5%) | 显著差 |
| IMC | 38.24 | 1.0000（基准） | 35/80 (43.8%) | 基准（新集稍难，验收低于 B3 的 40/80） |
| BO | 41.28 | 1.0630 [1.0306,1.0964] | 29/80 (36.2%) | 显著差于 IMC；原因尚未完成归因 |
| Safe-BO | 38.24 | 1.0000 等于 IMC | 35/80 (43.8%) | 选中基线本身，无增益无损失 |
| FNN（实际执行） | 38.24 | 1.0000 等于 IMC | 35/80 (43.8%) | 资格门拒绝→回退 IMC；**候选本身** 0.9792 [0.9680,0.9895]（事后对照，不得作为部署结论） |
| RL | 36.41 | 0.9614 [0.9429,0.9795] | 37/80 (46.2%) | **目标 J 显著优于 IMC**（区间整体低于 1）；幅度小（~4%），验收 37/80 vs 35/80 仅为观察值+2（未做显著性检验），需多种子重复确认 |
| LLM replay | 38.24 | 1.0000 与 IMC 持平 | 35/80 (43.8%) | 完整 48 训练集仍无接受候选；非实时调用 |

  - 来源：`sealed-80x7-b4/sealed_80_summary.csv:2-8`。B3→B4 变化唯一显著的是 RL 由“未发现优势”转为“小幅显著优”，
    其余结论稳定（BO 差、Safe-BO/FNN/LLM 等于 IMC、Z-N 垫底）——单次新集即可改变边缘结论，多种子重复仍是缺口。
  - E3+ 同协议归因（新密封集，`gp_attribution_summary.json`）：聚类 GP−随机 −0.0059 [−0.1040,+0.1335]（含 0），
    机制贡献未经证实（与 B3 一致）。
  - 失败控制实验（`failure_control_summary.csv`，IMC 失败 45 例 × 5 控制；`gains-x0.5`/`gains-x2.0` 为 Kp,Ki 同比例缩放，非单调比例项）：
    疑似调参 41 例中增益减半（Kp,Ki 同×0.5）翻转 28 例（确证为增益敏感型调参失败），延长观察翻转 9 例；
    疑似窗口 4 例中延长翻转 2 例；B3 的疑似容量 1 例在任何控制下均未翻转（初筛误报，已改判未解决）。
    RL 在新集挽救 #17/#51 且零新增（B3 为救 1 坏 2）；BO 新增 6 失败、挽救 0。
    J 方向一致：失败均值 ~48 vs 通过均值 ~26，目标与验收无系统性冲突（组均值仅作一致性旁证，不作因果证明）。

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
