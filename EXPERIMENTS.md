# EXPERIMENTS (v4)

清单：`experiments/manifests/v4.yaml`（唯一可执行事实源，缺字段立即报错；`hvac_pid/manifest.py:load_manifest`）；基线：`baseline-v3-pre-v4`（SHA `2870262`，仅历史对照，执行以实际 `git rev-parse HEAD` + `provenance.csv` 为准）。
历史 15 个顶层 `outputs*` 为只读归档；新产物只写 `artifacts/runs/<run_id>/`。

## v4 运行

- `artifacts/runs/v4-20260906-2870262/`：数值/FOPDT 交叉验证 + 步长收敛 + 七算法冒烟 + 溯源。
- `artifacts/runs/sealed-80x7-v4/`：七算法同一 80 场景密封（560 行，`run.py sealed --artifact-dir <bundle> --output <run>`；候选 vs 执行、配对 CI、消融、回退失败拆分）。
- 复现：`python run.py crossval artifacts/runs/<run_id>`；
  `python run.py embedded --algorithm all --provider replay --artifact-dir <artifact_dir> --output <run>/embedded_smoke`；
  `python run.py sealed --artifact-dir <artifact_dir> --output <run>/sealed-80x7`。
- 策略契约：`hvac_pid/policy_bundle.py:PolicyBundle`（IMC/BO/SafeBO/FNN 表+上下文/RL 表+掩码同目录冻结；缺文件报错，legacy 回退显式标记）。

## 场景与划分

- 正式划分：48 训练 / 16 验证 / 16 测试，三向隔离，`scenario_id` + SHA-256；密封 `5×16=80`（`acceptance_seeds=101,211,307,401,503`，`seed+100003+acceptance_seed`，噪声种子 `seed+700000+ordinal`，六子步积分器固定）。
- 三类典型场景：初次快速降温（4 h）、设定温度突变（5 h）、持续外界热扰动（6 h）。
- v3 的 80 场景密封只覆盖 Z-N/IMC/BO/FNN/RL；v4 首轮 `sealed-80x7-v4` 已将 SafeBO/LLM(replay-frozen) 接入同一清单（见 `sealed_provenance.json`），但 SafeBO 仍为 legacy 保守参数、RL 掩码为重建值；新流水线重训后 legacy 标记清零。

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

- `archive/outputs_llm_matrix_v3`：18 次运行、18 真实调用、18 stable、5/18 部署。以 18 次为准，禁止再写“仅 2 次真实会话”。
- LLM 同时报告多次运行分布与回退率；不同协议数据不参与统一排名；`replay` 为离线工具调用轨迹，非实时模型调用。

## 硬件门（待外部验证，不得仿真冒充）

- 实体板 10 分钟遥测、真实 WCET、HIL、Modbus 写入：`待外部验证`。
- 已有：PC-SIL（100 ms/2 s 分频、CRC、75 组 Python/C++ 对拍 ~2.98e-8）、ESP32 目标编译（RAM 22,036 B / Flash 296,169 B）、Wokwi 双目标启动。
