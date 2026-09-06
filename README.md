# HVAC AI–PI 自动整定演示

> v4 对账批次（进行中）：唯一事实源见 \experiments/manifests/v4.yaml\，新产物只写 \rtifacts/runs/<run_id>/\；架构/实验/结果分别见 [ARCHITECTURE.md](ARCHITECTURE.md)、[EXPERIMENTS.md](EXPERIMENTS.md)、[RESULTS.md](RESULTS.md)。v4 首个 run 为 \rtifacts/runs/v4-20260906-2870262/\（数值3/3+FOPDT3/3全绿，步长收敛+七算法冒烟+溯源）。
> 统一入口：`python run.py <pipeline|benchmark|crossval|embedded|llm-agent|advanced|matrix> ...`（`run_*.py` 保留为薄封装）；文档构建：`python build_docs.py <detailed|manual>`；指南与设计文档见 [docs/](docs/)。


> 新增风险感知安全 BO，以及真正的工具调用式 LLM Agent 自动整定：Agent 自主选择查看历史、运行候选仿真或停止，宿主安全门负责限幅、试验、验收和回退。详见 [docs/SOTA_LLM_PI_TUNING.md](docs/SOTA_LLM_PI_TUNING.md)。Q-learning 保留为教学/对照方法，不称为 SOTA。

## 第一次打开？看这里

三条路，对号入座：

| 你想干嘛 | 看哪里 / 跑什么 |
|---|---|
| 跑实验 | `python run.py --help`（7 个子命令；冒烟先跑 `python main.py --quick`） |
| 看报告 | `reports/分报告/`（00 总览 → 01 对象 → 02–08 各算法，每篇 md + docx 成对） |
| 玩演示 | 双击 `run_quick_demo.bat`，或 `streamlit run streamlit_app.py` |
| 怎么复现历史数据 | `experiments/manifests/v4.yaml` + `experiments/manifests/archive_map.csv` |

顶层目录地图（其余都是历史或生成物，不用挨个点开）：

| 目录 | 是什么 | 什么时候进 |
|---|---|---|
| `hvac_pid/` | 核心代码（对象、算法、仿真、评估） | 改算法、查实现 |
| `tools/` | 各实验命令行入口（`run_*`、`render_report`、`aggregate`） | 直接跑某个实验（一般用 `run.py` 代替） |
| `main.py` + `run.py` | 主流水线 + 统一入口 | 跑完整实验只认它俩 |
| `reports/分报告/` | 正式报告 00–12（md 为源，docx 为生成物；00 总览、01–03 入门主线、04–10 七算法、11 统一比较、12 部署验证） | 读结论、查数据溯源 |
| `reports/` 根 | 旧版总报告 + 配图脚本（`make_*`） | 重画图、看历史总报告 |
| `archive/` | 封存历史实验批次（v3 等 8 批，只读） | 核对报告里的旧数字 |
| `artifacts/runs/` | v4 新实验产物（按 run_id 分目录） | 看最新结果、写新报告 |
| `experiments/manifests/` | 实验清单与搬迁台账（v4.yaml、archive_map.csv） | 复现、查"东西搬哪去了" |
| `embedded/` | MCU 固件、SIL、Wokwi 工程 | 玩板子、看 PC-SIL |
| `modelica/` | OpenModelica 物理参考模型 | 做物理交叉验证 |
| `docs/` | 指南与设计文档（DEMO_GUIDE、SOTA、评审记录） | 上手、先看 [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md) |
| `tests/` | pytest 测试（39 个） | `python -m pytest tests -q` |
| `outputs/` | 可重跑的空 scratch（默认输出目录，gitignore） | 跑完实验看新鲜结果 |
| `vendor/` | KaTeX 前端资源（报告渲染用） | 不用管 |

## 评审整改 v3（正式密封测试）

`archive/outputs_review_v3/` 是本轮不覆盖历史结果的正式产物。训练、验证、测试按 48/16/16 个工况三向隔离；每个工况有稳定 `scenario_id` 和 SHA-256，IMC 的 `lambda` 只在训练集搜索，验证后冻结。候选发布使用 5 个新测试种子、每种子 16 个独立工况，共 80 个密封测试场景。

| 策略 | 相对调优 IMC 的平均目标比 | 单侧 95% 上界 | 实际采用非 IMC 比例 | 测试回退率 | 结论 |
|---|---:|---:|---:|---:|---|
| FNN | 0.9979 | 1.0067 | 100.0% | 0.0% | 通过非劣与安全门 |
| RL | 0.9595 | 0.9743 | 39.88% | 0.19% | 通过性能、安全与覆盖门 |

两者稳定率均为 100%，新增运行斜率、最低频率和启停违规均为 0。FNN 的原始状态局部标签仍有较高冲突（训练 log-RMSE 1.059），因此部署门没有采用高冲突拟合表，而选择验证集通过的保守 IMC 残差 TSK 曲面；这是真正产生非 IMC 增益的候选，不是把回退曲线冒充 FNN。RL 部署的是验证集选择的基线约束冻结策略，未覆盖状态强制回退 IMC。

嵌入式策略清单升级为 CRC v3，覆盖全部执行字段。PC-SIL 已验证 100 ms PI、2 s AI 调度、CRC 启动重算、故障回退及 75 组 Python/C++ 一致性向量（最大绝对误差约 `2.98e-8`）。ESP32 官方工具链编译成功，静态 RAM 22,036 B（6.7%）、Flash 296,169 B（22.6%）；这些不是实体板 10 分钟可靠性或 WCET 证据。逐项状态见 [review_remediation.md](archive/outputs_review_v3/review_remediation.md)。

## ESP32 七算法温度闭环 Demo（新增）

这一版把“温度是否真的稳定”放在算法分数之前。统一场景从 30°C 开始，目标为 24°C；曲线必须进入 24±0.5°C 并保持，随后注入开门扰动，再观察容量升高和温度恢复。90 秒墙钟演示对应 5 小时虚拟物理时间，不能解释为真实房间 90 秒降温。

七个独立入口：

```powershell
python run.py embedded --algorithm zn
python run.py embedded --algorithm imc
python run.py embedded --algorithm bo
python run.py embedded --algorithm safe-bo
python run.py embedded --algorithm fnn
python run.py embedded --algorithm rl
python run.py embedded --algorithm llm --provider replay
```

统一生成与交互展示：

```powershell
python run.py embedded --algorithm all --provider replay --output archive/outputs_embedded_demo
streamlit run streamlit_app.py
```

- `archive/outputs_embedded_demo/temperature_control_demo.html`：可离线打开的交互式温度回放、完整系统框图、七算法同场比较和 LLM Agent 工具审计；
- `archive/outputs_embedded_demo/llm_agent_trace.csv`：Agent 每一步工具、原始/限幅增益、仿真风险、安全门决定和剩余预算；
- `archive/outputs_embedded_demo/ESP32_七算法温度闭环_零基础教学.pptx`：20 页、原生可编辑框图和图表的零基础教学材料；
- `embedded/esp32_seven_algorithm_demo/`：ESP32 PlatformIO 双模式固件；
- `embedded/modbus_config.hpp`：真实设备寄存器配置，默认 `kWritesEnabled=false`；
- `embedded/validate_esp32_serial.py`：真实 ESP32 连续 10 分钟、丢周期与 WCET 验收脚本。

七算法演示目录保留生成当时的候选/回退状态；正式 v3 密封测试则以 `archive/outputs_review_v3/deployment_acceptance.csv` 为准，FNN 与 RL 均已通过。页面必须继续分别显示候选与实际执行曲线，任何新候选未通过时仍明确回退 IMC。LLM Agent 的 `replay` 是无网络录制的工具调用轨迹，不是实时模型调用。`ollama` 和 `openai` 模式会让模型在上位机从 `inspect_history`、`evaluate_candidate`、`finish` 中自主选择下一步；宿主程序仍独立执行硬边界、单轮 ±10%、重复仿真、试验预算和安全门。Agent 不进入 ESP32 的 100 ms 控制循环，也没有写压缩机容量的工具。

LLM Agent 自动整定数据流：

```text
历史指标 + 当前 Kp/Ki + 剩余预算
                 ↓
          [LLM Agent 决策]
       ┌─────────┼──────────┐
       ↓         ↓          ↓
  查看历史   请求候选仿真   结束整定
                  ↓
     [宿主限幅 → 多场景仿真 → 安全门]
                  ↓
       指标/接受或拒绝原因反馈给 Agent
                  ↺
     最终只导出已验收 Kp/Ki；否则 IMC 回退
```

## 当前版本：变频精密/机柜空调仿真平台

本项目实现双层模型：`modelica/HVACAI` 是 OpenModelica + Modelica Standard Library 的可执行物理参考模型（Modelica Buildings 留作后续高保真扩展）；`hvac_pid` 是可批量运行的 3R2C/FOPDT 快速控制代理。3R2C 的直觉是“室内空气”和“墙体/机柜”两个蓄热体，通过室外到空气、室外到慢热质、慢热质到空气三条换热通道交换热量。前者用于可追溯的热物理校验，后者用于控制算法寻优与可视化。

传统五种控制算法的逐段中文代码解析见 [docs/ALGORITHM_GUIDE.md](docs/ALGORITHM_GUIDE.md)；七算法嵌入式展示入口和落地边界以本节为准。

已实现五类控制器：Z-N、只在训练集整定 λ 的 IMC、全局贝叶斯优化固定 PI、25 规则（每次激活 4 条）的 FNN 自整定 PI、以及安全屏蔽的表格 RL 自整定 PI。RL 使用 5×5 热状态、3 档实际容量模式和 9 个相对 IMC 的绝对增益目标；不再递归累乘隐藏的当前增益。所有方法共用 0/最低稳定频率、量化、运行斜率、最小启停驻留、传感器噪声与滤波约束。

“自动整定”和“在线自整定”严格分开：贝叶斯优化在运行前搜索一组固定 `Kp/Ki`；FNN 从闭环状态快照出发做短时域安全候选回放并训练 IMC 残差 TSK 规则，RL 在相同虚拟房间中离线训练并经基线约束筛选，二者只在运行中低频微调参数，绝不在真实设备上探索。

快速运行：

```powershell
python -m pip install -r requirements.txt
python main.py --quick
python run.py benchmark
streamlit run streamlit_app.py
# 只重渲染已有实验结果为可打印 HTML 报告
python run.py render outputs
```

`streamlit run` 启动的是持续运行的本地 Web 服务：终端显示 URL 后，请在浏览器打开 `http://localhost:8501`；需要停止时按 `Ctrl+C`。

`run.py benchmark` 使用同一个受约束机柜空调问题实际计时 Z-N、IMC λ 搜索、贝叶斯固定 PI、FNN 标签/规则训练和 RL 训练，输出 PC 墙钟时间、等效虚拟对象时间、最终增益、控制指标与部署验收结果。PC 时间不是 MCU WCET；真实设备辨识时间和目标板周期需单独实测。

第二阶段 MCU 软件在环与 Wokwi 工程：

```powershell
python embedded/run_mcu_validation.py
python embedded/wokwi/prepare_projects.py
```

STM32F103C8T6 与 ESP32 的验证入口、接线图、验收证据和当前完成边界见 [embedded/README.md](embedded/README.md)。当前已完成 PC 软件在环；两种 MCU 的完整代码也已在 Wokwi 在线编译并进入运行态，但串口曲线、ROM/RAM 与最坏周期尚未取得可信留证。不能用 PC 结果代替这些目标数据。

结果目录包含 IMC λ 候选、贝叶斯每次试验、FNN 每批拟合、RL 每回合训练与训练后部署验收的完整历史 CSV，以及三类独立场景结果。FNN/RL 若在离线回放中明显劣于已调 IMC，产物会自动退回安全基线；因此“生成了表”不等于“训练已收敛”。

这是一个可直接运行的 Python 仿真项目。它用单区域 **3R2C 建筑热模型**模拟空调制冷，自动生成不同工况下的最优 `Kp`、`Ki` 标签，训练“误差状态 → PI 参数”的 FNN/RL 增益调度器，并与固定 PI、Ziegler–Nichols（ZN）和 IMC PI 比较。

> 这个项目是算法验证与开发模板，不是可直接写入真实 PLC 的控制器。真实设备上线前必须做模型校准、影子运行、上下限/变化率约束、故障回退和逐级闭环验证。

## 一键运行

环境要求：Python 3.10+。

```powershell
python -m pip install -r requirements.txt
python main.py --quick
```

完整实验（默认 48/16/16 训练/验证/测试，并在 5×16 个密封场景验收）：

```powershell
python main.py --output archive/outputs_review_v3 --validation-samples 16 --acceptance-seeds 101,211,307,401,503
```

运行测试：

```powershell
python -m pytest tests -q
```

自定义规模：

```powershell
python main.py --train-samples 80 --validation-samples 24 --test-samples 24 --bo-iterations 10 --seed 42 --acceptance-seeds 101,211,307,401,503 --output outputs_large
```

## 演示做了什么

```text
随机热工况 ──> 有边界的贝叶斯优化 ──> 固定全局 Kp/Ki（自动整定）
                      │
                      ├──> 状态快照局部安全回放 ──> 训练 25 条残差 TSK 规则
                      └──> 3R2C 虚拟房间 ──> 训练并基线约束 RL 策略

FNN/RL 训练产物 ──> 低频微调 Kp/Ki（在线自整定） ──> 安全 PI ──> HVAC 3R2C 模型
                                                         │
                                                         └──> 异常时回退 IMC PI
```

1. 随机化室外温度、设定温度、初始温差、内部负荷、建筑热容/热阻、制冷能力、执行器延迟和惯性。
2. 在每个工况的仿真环境中，对 `log(Kp)`、`log(Ki)` 做有边界的贝叶斯优化；稳定性越界会得到高惩罚。
3. 从真实闭环状态快照对候选增益做局部安全回放，训练 FNN 的 25 条残差 TSK 规则；在同一 3R2C 环境中离线训练并约束 RL 策略。
4. 在完全冻结的密封测试工况上比较固定 PI、ZN、IMC、FNN 和 RL；查看测试结果后再改模型必须更换测试种子。
5. 分别运行初次降温、设定点突变、持续外界热扰动三类场景，并生成可视化报告。

## 输入、输出与指标

PI 控制器输入：

- 误差 `e = T_zone - T_setpoint`；误差为正表示需要增加制冷。
- 当前 `Kp`、`Ki` 和采样周期。

PI 控制器输出：

- PI 先生成 `u_request ∈ [0,1]`；执行器层再施加默认 25% 最低运行容量、1% 量化、5%/min 运行斜率及最小启停驻留，得到实际 `u`。

AI 调度器上下文（输入特征）：

- 室外温度、设定温度、当前绝对温差、内部负荷估计；
- 区域/围护结构热容与三条热阻；
- 最大制冷量、执行器纯延迟、执行器时间常数。

AI 输出：

- `Kp`、`Ki`。二者是动作/参数，不是 context。

评价指标：ITAE、RMSE、IAE、舒适区违规度时、最大过冷、进入舒适带时间、制冷能耗代理量、控制动作总变化、容量指令方差、启停次数、运行斜率违规数和稳定率。综合目标越低越好，权重集中写在 `hvac_pid/metrics.py`。

## 生成文件

默认写入 `outputs/`；本轮正式评审证据写入 `archive/outputs_review_v3/`：

- `training_labels.csv`：训练数据、`Kp/Ki` 标签、ZN/IMC 对照分数；
- `global_bayesian_tuning.csv`：跨训练工况搜索得到的一套固定全局 `Kp/Ki`；
- `imc_lambda_tuning.csv`：IMC 在训练工况上评价的全部 λ 候选及最终选择；
- `bayesian_search_history.csv`：贝叶斯优化每一次候选、目标值、当前最优值与代理模型信息；
- `fnn_rule_table.npy`：由 BO 标签学习的 5×5×2 FNN 规则后件表；
- `fnn_training_history.csv`：FNN 每批样本、规则覆盖率、拟合误差和规则参数变化；
- `rl_q_table.npy`：验收后可部署的 5×5×3×9 RL Q 表；若验收失败则为IMC不改增益回退表；
- `rl_q_table_candidate.npy`：验收前候选Q表，用于审计而不直接部署；
- `fnn_rule_table_candidate.npy`：验收前候选FNN规则表；
- `dataset_manifest.csv`：训练/验证/测试工况 ID、种子、分区及 SHA-256，用于拒绝数据泄漏；
- `deployment_acceptance.csv`：5×16 个密封场景中 FNN/RL 相对 IMC 的配对结果、95% 上界、安全事件、实际采用率和回退率；
- `policy_parity_vectors.csv`：Python/C++ 的 FNN、RL 和故障回退一致性测试向量；
- `policy_manifest_v3.json`：版本、验收标志、增益边界、FNN 参数、RL 冻结策略、覆盖掩码及 CRC 的可审计清单；
- `training_scenarios.csv`、`validation_scenarios.csv`、`fnn_training_samples.csv`、`rl_training_transitions.csv`、`holdout_scenarios.csv`：三向数据、逐状态监督样本、逐步 RL 转移和密封测试工况；
- `rl_training_history.csv`：RL 每回合奖励、TD 误差、探索率、状态覆盖率、策略变化和 Kp/Ki；
- `classical_tuning_history.csv`：Z-N/IMC 共用的 168 h 虚拟阶跃逐分钟响应；
- `fopdt_fit_history.csv`：有界最小二乘实际评价过的每一组 K、τ、L 候选与拟合误差；
- `classical_tuning_steps.csv`：从名义工况、阶跃输入、FOPDT 参数到 Z-N/IMC 未限幅值和部署值的逐步代入；
- `holdout_metrics.csv`：逐测试工况、逐控制器指标；
- `holdout_summary.csv`：留出集汇总；
- `dynamic_metrics.csv`：动态 12 小时场景指标；
- `dynamic_timeseries.csv`：五种算法的逐时间步真实/测量温度、请求/实际容量指令、增益、扰动与安全事件；
- `case_metrics.csv`：初次快速降温、设定温度突变、持续外界热扰动三类场景的独立指标；
- `dynamic_comparison.png`：温度、控制量和综合目标对比；
- `training_labels.png`：最优标签分布和相对 ZN 改善分布；
- `bayesian_search_trace.png`、`fnn_training_trace.png`、`rl_training_trace.png`：三种 AI 方法完整整定/训练轨迹；
- `engineering_report.md`：自动生成的工程分析报告。
- `engineering_report.html`：带图表、排版和可打印样式的自包含可视化报告；浏览器打开后可直接“打印为 PDF”。

## 为什么强化学习不直接控制压缩机

当前问题只有两个连续参数，离线贝叶斯优化的样本效率、可解释性和安全约束都更合适。强化学习更适合多执行器、长时域和强耦合优化，但真实 HVAC 上不能在线随机探索。若以后引入 RL，建议只让它做上位机设定点或残差修正，并保留 PI、动作屏蔽和安全回退。

## 迁移到真实 HVAC 的替换点

- 用历史 BMS 数据或安全阶跃试验辨识热容、热阻、延迟和负荷估计器；
- 用真实天气/占用时序替换随机工况生成器；
- 用项目 KPI 重设目标权重，并在多季节留出数据上验证；
- 先只读推理和影子评分，再允许有限区间内下发 `Kp/Ki`；
- 保留现有 PLC/PI 作为底层闭环，通信中断、域外输入或异常振荡时立即回退。

## 项目结构

```text
main.py                    主流水线（48/16/16 训练/验证/测试 + 80 密封验收）
run.py                     统一入口（pipeline/benchmark/crossval/embedded/llm-agent/advanced/matrix/render）
tools/                     各实验的命令行入口（run_*.py、render_report.py、aggregate_llm_matrix.py）
hvac_pid/config.py         工况、动态天气/负荷与训练分布
hvac_pid/plant.py          3R2C 热模型和执行器延迟/惯性
hvac_pid/controllers.py    PI、抗饱和、ZN、IMC、FOPDT 辨识
hvac_pid/ai_controllers.py FNN 与安全屏蔽的表格 RL 参数自整定
hvac_pid/tuning.py         贝叶斯优化及标签生成
modelica/HVACAI/           Modelica Buildings 物理参考模型
embedded/                  100 ms PID / 2 s AI、PC SIL 与 STM32/ESP32 Wokwi 工程
streamlit_app.py           三通道交互式 Demo
hvac_pid/metrics.py        指标与综合目标
hvac_pid/pipeline.py       训练、评估、保存流程
hvac_pid/plotting.py       结果图
tests/test_demo.py         单元与端到端测试
```
