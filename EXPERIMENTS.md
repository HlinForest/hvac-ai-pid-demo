# EXPERIMENTS (v4)

清单：`experiments/manifests/v4.yaml`（唯一可执行事实源，缺字段立即报错；`hvac_pid/manifest.py:load_manifest`）；基线：`baseline-v3-pre-v4`（SHA `2870262`，仅历史对照，执行以实际 `git rev-parse HEAD` + 脏位 + `provenance.csv` 为准）。
历史 15 个顶层 `outputs*` 为只读归档；新产物只写 `artifacts/runs/<run_id>/`。

## 实验登记（正文/图表/CSV 统一编号）

| 编号 | 实验 | 本轮定位 | 证据 |
|---|---|---|---|
| E1 | 数值积分与 FOPDT 校验 | 证明模型实现与代理精度（已通过） | `artifacts/runs/v4-20260906-2870262/` |
| E2-B1 | 混合冻结策略统一重评（80×7=560） | 历史主实验；混合冻结来源，非统一训练比较 | `artifacts/runs/sealed-80x7-v4/` |
| E2-B2 | 统一重训后重评（80×7=560） | 当前主实验；同一 48/16/16 流水线冻结，legacy 双清零 | `artifacts/runs/sealed-80x7-v4-retrain/` + 训练冻结 `artifacts/runs/v4-20260907-8f3fd5f/` |
| E3 | 同初始化同预算整定对照（随机×4 + 保守公式） | Safe-BO 机制归因：搜索 vs 保守初值 | `sealed_ablation.csv` + `safe_bo_attribution` |
| E3+ | GP 同预算归因（BO-full/初值最优/纯随机14/保守公式） | GP 机制贡献：同训练谱选型、同密封评价 | `sealed-80x7-v4-retrain/gp_attribution.csv`（`run.py attribution`） |
| E4 | LLM 真实调用矩阵（18 次） | 独立实验：采用率/回退后验收率/成本，不参与 E2 排名 | `archive/outputs_llm_matrix_v3/` |
| E5-sw | 策略对拍与 PC-SIL（含主机 WCET 参考） | 证明 Python/C++ 软件一致性，不代替实体温控实验 | `v4-20260907-8f3fd5f/mcu_pc_sil_log.txt` + `host_wcet.csv` |
| E5-target | 实体遥测/目标 WCET/HIL | 待外部验证（三门 PENDING，审计为准） | `embedded/hardware_gates.md` + `python tools/check_hardware_gates.py` |

E2-B1 命名统一为“冻结策略重评”，禁止写成“七算法统一训练冻结比较”；E2-B2 为单一种子单预算统一重训，可写成统一训练预算比较（多种子重复仍缺）。

## v4 运行

- `artifacts/runs/v4-20260906-2870262/`：数值/FOPDT 交叉验证 + 步长收敛 + 七算法冒烟 + 溯源（E1）。
- `artifacts/runs/v4-20260907-8f3fd5f/`：B2 统一重训冻结（48/16/16，约 227 s，`bo/safe_bo_policy.json` + `rl_covered_mask.npy` 冻结，FNN/RL 部署验收均通过；34 MB 原始产物已按 `PRUNED.md` 裁剪至 0.5 MB 可再生子集）。
- `artifacts/runs/sealed-80x7-v4/`：E2-B1 七算法同一 80 场景密封（560 行，`run.py sealed --artifact-dir <bundle> --output <run>`；候选 vs 执行、主统计 CI、消融、回退失败拆分；附 `sealed_scenarios.csv` 全场景清单 + `manifest.yaml` 冻结拷贝 + `sealed_provenance.json:E2/code_sha/dirty/policy 哈希`）。
- `artifacts/runs/sealed-80x7-v4-retrain/`：E2-B2 同一密封清单在 B2 冻结策略上的重评（`policy_provenance` legacy 双零）+ `gp_attribution.csv/json`（E3+ 同预算归因）。
- E2 统计主口径：主统计量为逐场景配对比均值 `mean(obj/imc)`（`mean_ratio_to_imc` = `paired_ratio_mean` 点估计）+ bootstrap 95% CI（2000 重采样，`seed+77`）；次统计量 `ratio_of_means_to_imc`（`mean(obj)/mean(imc)`）仅作对照，不得混用。验收主口径：`validation_passed = bounded & comfort_held & recovery_ok & actuator_compliant`；`fallback_failed = 有回退且 validation 未通过`；`stable` 仅表示数值有界（5–45°C 有限），不得等同验收通过。
- 复现：`python run.py crossval artifacts/runs/<run_id>`；
  `python run.py embedded --algorithm all --provider replay --artifact-dir <artifact_dir> --output <run>/embedded_smoke`；
  `python run.py sealed --artifact-dir <artifact_dir> --output <run>/sealed-80x7`。
- 策略契约：`hvac_pid/policy_bundle.py:PolicyBundle`（IMC/BO/SafeBO/FNN 表+上下文/RL 表+掩码同目录冻结；缺文件报错，legacy 回退显式标记）。

## 场景与划分

- 正式划分：48 训练 / 16 验证 / 16 测试，三向隔离，`scenario_id` + SHA-256；密封 `5×16=80`（`acceptance_seeds=101,211,307,401,503`，`seed+100003+acceptance_seed`，噪声种子 `seed+700000+ordinal`，六子步积分器固定）。
- 三类典型场景：初次快速降温（4 h）、设定温度突变（5 h）、持续外界热扰动（6 h）。
- B2 轮 `sealed-80x7-v4-retrain` 为同一清单在统一重训冻结策略上的重评（`bo_legacy_fallback=0/safe_bo_legacy_fallback=0`，RL 掩码冻结非重建，FNN/RL 部署验收通过）；B1 的 legacy 说明保留为历史，不再是现状。

## 数值验证（已修复）

- 现象：v3 `physical_cross_validation.csv` 初次快速降温 `max=0.150350°C > 0.15°C`，`passed=0`。
- 收敛实验证据（`timestep_convergence.csv`）：N=1→0.150350，N=2→0.074927，N=4→0.037400，N=6→0.025154，确认为显式欧拉一阶离散误差。
- 修复：`Scenario.integration_substeps=6`（外层 60 s 接口不变，内层 10 s×6 子步进）；未放宽阈值。
- v4 结果：3/3 物理通过（0.025154 / 0.012704 / 0.003174），3/3 FOPDT 通过（归一化 RMSE 11.88–11.95% ≤ 15%）。

## Modelica

- `archive/outputs_review_v3/modelica/PrecisionCabinetCooling_res.csv`（DASSL 12 h）为历史证据，SHA-256 记录于 `provenance.csv`（`ad68bd1e…`）。
- v3 部分报告曾同时写“未运行”与“已实际运行”，属快照矛盾；v4 统一标记为“历史证据，当前未复现”，新 run 目录未发现 `modelica/*.csv` 即判未运行（路径为 run 相对路径，不再写死 `outputs/modelica`）。
- 本机无 `omc` 时 `run.py crossval` 只刷新数值验证并明确打印未复跑。

## LLM

- `archive/outputs_llm_matrix_v3`：18 次运行、18 真实调用、18 数值有界稳定（演示口径）、5/18 部署。以 18 次为准，禁止再写“仅 2 次真实会话”或“15/18 stable”。
- LLM 同时报告多次运行分布与回退率；不同协议数据不参与统一排名；`replay` 为离线工具调用轨迹，非实时模型调用。

## 硬件门（目标门待外部验证，不得仿真冒充）

- 实体板 10 分钟遥测、目标 WCET、HIL、Modbus 写入：`待外部验证`（`embedded/hardware_gates.md`，审计 `python tools/check_hardware_gates.py --run <run>` 当前 3/3 PENDING）。
- 已有（软件侧，非硬件证据）：PC-SIL（100 ms/2 s 分频、CRC、75 组 Python/C++ 对拍 ~2.98e-8，B2 run 内 `mcu_pc_sil_log.txt:PASS`）、
  主机 WCET 参考（`host_wcet.csv`，HOST-ONLY，不得代入目标门）、ESP32 历史目标编译（RAM 22,036 B / Flash 296,169 B；本机刷新失败，`esp32_target_compile_attempt.log` 存证，用户路径中文致工具链失效）、Wokwi 双目标启动。
- 主报告 DOCX：`docs/主报告.docx` 由 `docs/主报告.md` 经 `reports/md2docx.py` 生成，`python tools/check_docx.py` 逐节核对（标题 1:1/表格/关键数字/旧值扫描/非空节）。
