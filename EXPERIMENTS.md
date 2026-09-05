# EXPERIMENTS (v4)

清单：`experiments/manifests/v4.yaml`；基线：`baseline-v3-pre-v4`（SHA `2870262`）。
历史 15 个顶层 `outputs*` 为只读归档；新产物只写 `artifacts/runs/<run_id>/`。

## v4 运行

- `artifacts/runs/v4-20260906-2870262/`：数值/FOPDT 交叉验证 + 步长收敛 + 七算法冒烟 + 溯源。
- 复现：`python run_cross_validation.py artifacts/runs/<run_id>`；
  `python run_embedded_demo.py --algorithm all --provider replay --artifact-dir <artifact_dir> --output <run>/embedded_smoke`。

## 场景与划分

- 正式划分：48 训练 / 16 验证 / 16 测试，三向隔离，`scenario_id` + SHA-256；密封 `5×16=80`。
- 三类典型场景：初次快速降温（4 h）、设定温度突变（5 h）、持续外界热扰动（6 h）。
- v3 的 80 场景密封只覆盖 Z-N/IMC/BO/FNN/RL；Safe BO 与 LLM 来自其他测试集。
  在接入同一场景清单前，禁止七算法统一排名（见 `sealed_80_summary.csv`）。

## 数值验证（已修复）

- 现象：v3 `physical_cross_validation.csv` 初次快速降温 `max=0.150350°C > 0.15°C`，`passed=0`。
- 收敛实验证据（`timestep_convergence.csv`）：N=1→0.150350，N=2→0.074927，N=4→0.037400，N=6→0.025154，确认为显式欧拉一阶离散误差。
- 修复：`Scenario.integration_substeps=6`（外层 60 s 接口不变，内层 10 s×6 子步进）；未放宽阈值。
- v4 结果：3/3 物理通过（0.025154 / 0.012704 / 0.003174），3/3 FOPDT 通过（归一化 RMSE 11.88–11.95% ≤ 15%）。

## Modelica

- `outputs/modelica/PrecisionCabinetCooling_res.csv`（DASSL 12 h）为历史证据，SHA-256 记录于 `provenance.csv`（`ad68bd1e…`）。
- v3 部分报告曾同时写“未运行”与“已实际运行”，属快照矛盾；v4 统一标记为“历史证据，当前未复现”，新 run 目录未发现 `modelica/*.csv` 即判未运行（路径为 run 相对路径，不再写死 `outputs/modelica`）。
- 本机无 `omc` 时 `run_cross_validation.py` 只刷新数值验证并明确打印未复跑。

## LLM

- `outputs_llm_matrix_v3`：18 次运行、18 真实调用、18 stable、5/18 部署。以 18 次为准，禁止再写“仅 2 次真实会话”。
- LLM 同时报告多次运行分布与回退率；不同协议数据不参与统一排名；`replay` 为离线工具调用轨迹，非实时模型调用。

## 硬件门（待外部验证，不得仿真冒充）

- 实体板 10 分钟遥测、真实 WCET、HIL、Modbus 写入：`待外部验证`。
- 已有：PC-SIL（100 ms/2 s 分频、CRC、75 组 Python/C++ 对拍 ~2.98e-8）、ESP32 目标编译（RAM 22,036 B / Flash 296,169 B）、Wokwi 双目标启动。
