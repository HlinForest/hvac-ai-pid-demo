# HVAC AI–PI 自动整定演示

## 当前版本：变频精密/机柜空调仿真平台

本项目实现双层模型：`modelica/HVACAI` 是 OpenModelica + Modelica Standard Library 的可执行物理参考模型（Modelica Buildings 留作后续高保真扩展）；`hvac_pid` 是可批量运行的 3R2C/FOPDT 快速控制代理。3R2C 的直觉是“室内空气”和“墙体/机柜”两个蓄热体，通过室外到空气、室外到慢热质、慢热质到空气三条换热通道交换热量。前者用于可追溯的热物理校验，后者用于控制算法寻优与可视化。

五种控制算法的逐段中文代码解析见 [ALGORITHM_GUIDE.md](ALGORITHM_GUIDE.md)。

已实现五类控制器：Z-N、IMC、全局贝叶斯优化固定 PI、25 规则（每次激活 4 条）的 FNN 自整定 PI、以及安全屏蔽的增量式 RL 自整定 PI。

“自动整定”和“在线自整定”严格分开：贝叶斯优化在运行前搜索一组固定 `Kp/Ki`；FNN 先由 BO 标签离线训练规则表、RL 先在相同的虚拟房间中离线训练 Q 表，二者只在运行中低频微调参数，绝不在真实设备上探索。

快速运行：

```powershell
cd "C:\Users\厉飞雨\Documents\New project\hvac_ai_pid_demo"
python -m pip install -r requirements.txt
python main.py --quick
streamlit run streamlit_app.py
# 只重渲染已有实验结果为可打印 HTML 报告
python render_report.py outputs
```

`streamlit run` 启动的是持续运行的本地 Web 服务：终端显示 URL 后，请在浏览器打开 `http://localhost:8501`；需要停止时按 `Ctrl+C`。

第二阶段 MCU 软件在环与 Wokwi 工程：

```powershell
python embedded/run_mcu_validation.py
python embedded/wokwi/prepare_projects.py
```

STM32F103C8T6 与 ESP32 的验证入口、接线图、验收证据和当前完成边界见 [embedded/README.md](embedded/README.md)。当前已完成 PC 软件在环；两种 MCU 的完整代码也已在 Wokwi 在线编译并进入运行态，但串口曲线、ROM/RAM 与最坏周期尚未取得可信留证。不能用 PC 结果代替这些目标数据。

结果目录包含贝叶斯每次试验、FNN 每批拟合、RL 每回合训练的完整历史 CSV，以及三类独立场景的 `case_metrics.csv` 与完整 `engineering_report.html`。报告先解释直觉，再展示整定/训练轨迹，并列出 ITAE、调节时间、最大过冷、压缩机指令方差、AI 推理时间和回退事件。快速模式当前只能证明流程跑通：贝叶斯出现明确更优候选；FNN 只激活过 5/25 条规则，RL 虽访问 25/25 个粗网格状态但奖励仍有明显波动，因此两者都不能宣称充分收敛。

这是一个可直接运行的 Python 仿真项目。它用单区域 **3R2C 建筑热模型**模拟空调制冷，自动生成不同工况下的最优 `Kp`、`Ki` 标签，训练“误差状态 → PI 参数”的 FNN/RL 增益调度器，并与固定 PI、Ziegler–Nichols（ZN）和 IMC PI 比较。

> 这个项目是算法验证与开发模板，不是可直接写入真实 PLC 的控制器。真实设备上线前必须做模型校准、影子运行、上下限/变化率约束、故障回退和逐级闭环验证。

## 一键运行

环境要求：Python 3.10+。

```powershell
cd "C:\Users\厉飞雨\Documents\New project\hvac_ai_pid_demo"
python -m pip install -r requirements.txt
python main.py --quick
```

完整实验（默认 48 个训练工况、16 个留出测试工况）：

```powershell
python main.py
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

自定义规模：

```powershell
python main.py --train-samples 80 --test-samples 24 --bo-iterations 10 --seed 42 --output outputs_large
```

## 演示做了什么

```text
随机热工况 ──> 有边界的贝叶斯优化 ──> 固定全局 Kp/Ki（自动整定）
                      │
                      ├──> BO 标签 ──> 训练 25 条 FNN 规则表
                      └──> 3R2C 虚拟房间 ──> 训练 RL Q 表

FNN/RL 训练产物 ──> 低频微调 Kp/Ki（在线自整定） ──> 安全 PI ──> HVAC 3R2C 模型
                                                         │
                                                         └──> 异常时回退 IMC PI
```

1. 随机化室外温度、设定温度、初始温差、内部负荷、建筑热容/热阻、制冷能力、执行器延迟和惯性。
2. 在每个工况的仿真环境中，对 `log(Kp)`、`log(Ki)` 做有边界的贝叶斯优化；稳定性越界会得到高惩罚。
3. 用优化得到的 `Kp/Ki` 标签训练 FNN 的 25 条规则表，并在同一 3R2C 环境中离线训练 RL Q 表。
4. 在未参加训练的工况上比较固定 PI、ZN、IMC、FNN 和 RL。
5. 分别运行初次降温、设定点突变、持续外界热扰动三类场景，并生成可视化报告。

## 输入、输出与指标

PI 控制器输入：

- 误差 `e = T_zone - T_setpoint`；误差为正表示需要增加制冷。
- 当前 `Kp`、`Ki` 和采样周期。

PI 控制器输出：

- `u ∈ [0, 1]`，代表 0%–100% 制冷指令；包含条件积分抗饱和。

AI 调度器上下文（输入特征）：

- 室外温度、设定温度、当前绝对温差、内部负荷估计；
- 区域/围护结构热容与三条热阻；
- 最大制冷量、执行器纯延迟、执行器时间常数。

AI 输出：

- `Kp`、`Ki`。二者是动作/参数，不是 context。

评价指标：ITAE、RMSE、IAE、舒适区违规度时、最大过冷、进入舒适带时间、制冷能耗代理量、控制动作总变化、压缩机输出方差和稳定率。综合目标越低越好，权重集中写在 `hvac_pid/metrics.py`，便于按项目要求调整。

## 生成文件

默认写入 `outputs/`：

- `training_labels.csv`：训练数据、`Kp/Ki` 标签、ZN/IMC 对照分数；
- `global_bayesian_tuning.csv`：跨训练工况搜索得到的一套固定全局 `Kp/Ki`；
- `bayesian_search_history.csv`：贝叶斯优化每一次候选、目标值、当前最优值与代理模型信息；
- `fnn_rule_table.npy`：由 BO 标签学习的 5×5×2 FNN 规则后件表；
- `fnn_training_history.csv`：FNN 每批样本、规则覆盖率、拟合误差和规则参数变化；
- `rl_q_table.npy`：仅由仿真训练的 5×5×9 RL 增益增量策略表；
- `rl_training_history.csv`：RL 每回合奖励、TD 误差、探索率、状态覆盖率、策略变化和 Kp/Ki；
- `classical_tuning_history.csv`：Z-N/IMC 共用的 168 h 虚拟阶跃逐分钟响应；
- `fopdt_fit_history.csv`：有界最小二乘实际评价过的每一组 K、τ、L 候选与拟合误差；
- `classical_tuning_steps.csv`：从名义工况、阶跃输入、FOPDT 参数到 Z-N/IMC 未限幅值和部署值的逐步代入；
- `holdout_metrics.csv`：逐测试工况、逐控制器指标；
- `holdout_summary.csv`：留出集汇总；
- `dynamic_metrics.csv`：动态 12 小时场景指标；
- `dynamic_timeseries.csv`：五种算法的逐时间步温度、PWM、增益、扰动与安全事件；
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
main.py                    命令行入口
hvac_pid/config.py         工况、动态天气/负荷与训练分布
hvac_pid/plant.py          3R2C 热模型和执行器延迟/惯性
hvac_pid/controllers.py    PI、抗饱和、ZN、IMC、FOPDT 辨识
hvac_pid/ai_controllers.py FNN 与安全屏蔽的增量 RL 参数自整定
hvac_pid/tuning.py         贝叶斯优化及标签生成
modelica/HVACAI/           Modelica Buildings 物理参考模型
embedded/                  100 ms PID / 2 s AI、PC SIL 与 STM32/ESP32 Wokwi 工程
streamlit_app.py           三通道交互式 Demo
hvac_pid/metrics.py        指标与综合目标
hvac_pid/pipeline.py       训练、评估、保存流程
hvac_pid/plotting.py       结果图
tests/test_demo.py         单元与端到端测试
```
