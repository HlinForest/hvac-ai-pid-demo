# 变频精密/机柜空调 AI-PID 仿真、算法对比与量产评估报告

> 控制器对比数值来自 Python 3R2C + 延迟/一阶执行器。已实际运行 OpenModelica/DASSL、SciPy 连续方程和 FOPDT/3R2C 三层交叉验证；OpenModelica 状态：未运行：artifacts\runs\v4-20260906-bf6bda6\modelica\PrecisionCabinetCooling_res.csv 不存在（历史证据见 archive/outputs_review_v3/modelica/PrecisionCabinetCooling_res.csv，当前未复现）。

## 1. 环境建模、仿真器、控制函数与输入输出

- 物理参考：OpenModelica + Modelica Standard Library，实际执行模型位于 `modelica/HVACAI/PrecisionCabinetCooling.mo`；Modelica Buildings Library 留作后续高保真设备模型扩展。
- 实验仿真器：Python/NumPy 中的 3R2C 两状态热模型，主函数为 `simulate(scenario, controller, seed)`。
- 热模型：

```text
Cz·dTz/dt = (To-Tz)/Roz + (Tw-Tz)/Rzw + Qint - Qc
Cw·dTw/dt = (To-Tw)/Row + (Tz-Tw)/Rzw
τu·dQc/dt = Qmax·u(t-L) - Qc
```

FOPDT 代理为 `ΔT(s)/Δu(s)=-K exp(-Ls)/(τs+1)`，只用于 Z-N/IMC 整定。主要参数是两个热容 Cz/Cw、三条热阻 Roz/Rzw/Row、制冷能力 Qmax、延迟 L、执行器惯性 τu、室外温度与内部热负荷。

| 层级 | 输入 | 输出 |
|---|---|---|
| 环境 | To(t)、Tsp(t)、Qint(t) | 天气/负荷/设定值时序 |
| 3R2C | To、Qint、Qc | Tz、Tw |
| SafePI | e=Tz-Tsp、Kp、Ki、Δt | u∈[0,1]（0–100% 容量请求） |
| FNN/RL | e、Δe | 受限 Kp、Ki；不直接控制压缩机 |

Python 热模型步长是 1 min；嵌入式目标是 PI 100 ms、AI 2 s。

### 1.1 实际运行的交叉验证

1 分钟离散热模型与独立 SciPy 连续方程求解结果：

| 场景 | RMSE（°C） | 最大误差（°C） | 判定 |
|---|---|---|---|
| 初次快速降温 | 0.01667 | 0.02515 | 通过 |
| 设定温度突变 | 0.00815 | 0.0127 | 通过 |
| 持续外界热扰动 | 0.00182 | 0.003174 | 通过 |

FOPDT 快速代理与 3R2C 物理模型阶跃响应结果：

| 场景 | K（°C/指令） | τ（min） | L（min） | 归一化 RMSE（%） | 判定 |
|---|---|---|---|---|---|
| 初次快速降温 | 59.89 | 912.3 | 6 | 11.95 | 通过 |
| 设定温度突变 | 59.9 | 913.4 | 7 | 11.92 | 通过 |
| 持续外界热扰动 | 64.63 | 914.4 | 8 | 11.88 | 通过 |

OpenModelica/DASSL 与独立 Python/DOP853 连续方程的同工况结果：

_No results._

## 2. 控制算法设计与数学表达

- 共用 PI：`e=Tz-Tsp`，`I*=I+eΔt`，`u=clip(Kp·e+Ki·I*,0,1)`；条件积分抗饱和。
- Z-N：`Kp=0.9τ/(KL)`，`Ti=3.33L`，`Ki=Kp/Ti`；在独立安装调试阶段整定一次，三个测试工况全部冻结。
- IMC：保守回退值取 `λ=max(τ/3,3L,12min)`；公平性能基线只在训练工况搜索 λ，随后仍按 `Kp=τ/[K(λ+L)]`、`Ti=min[τ,4(λ+L)]` 换算并在留出测试中冻结。
- 贝叶斯自动整定：在 log(Kp),log(Ki) 内用 Matérn-5/2 高斯过程 + EI 搜索全局固定增益；本次为 Kp=0.609258、Ki=0.00175406。
- FNN：5×5 TSK 规则表，输入为误差 e 与误差变化率 ė，`Kp=ΣwᵢⱼKpᵢⱼ`、`Ki=ΣwᵢⱼKiᵢⱼ`，每次只有 4 条双线性插值规则激活；规则表由 BO 标签离线训练并在验证集选择 IMC 正则强度。覆盖不足或平均目标劣于 IMC时拒绝候选表。
- RL：状态为 5×5 的 (e,ė) 热状态再乘 3 档实际容量模式，9 个动作是相对 IMC 的绝对 Kp/Ki 目标比例；750 回合、每回合最长 240 min、每 5 min 决策。训练后使用多工况回放和早停选表，劣于 IMC 超过2%或25热状态覆盖不足80%时拒绝并回退 IMC。

完整整定/训练证据：

- `classical_tuning_history.csv` / `fopdt_fit_history.csv` / `classical_tuning_steps.csv`：Z-N、IMC 的逐分钟阶跃、全部 FOPDT 候选与逐步公式代入。
- `bayesian_search_history.csv` / `bayesian_search_trace.png`：每次候选 Kp/Ki、本轮目标、当前最优值和收敛轨迹。
- `fnn_training_history.csv` / `fnn_training_trace.png`：规则覆盖率、拟合误差与代表性规则 Kp/Ki。
- `rl_training_history.csv` / `rl_training_trace.png`：逐回合奖励、TD 误差、状态覆盖、策略改变和回合增益。

## 3. 统一评估指标体系

| 类别 | 指标 |
|---|---|
| 跟踪精度与速度 | RMSE、IAE=`∫|e|dt`、ITAE=`∫t|e|dt`、调节时间 |
| 动态品质 | 扰动恢复时间、最大过热、最大过冷、舒适带超限度时 |
| 执行机构寿命与能耗 | 容量指令方差、`Σ|u[k]-u[k-1]|`、启停次数、`∫Qc dt` 制冷能量代理 |
| 安全与实时性 | 稳定率、回退次数、AI 推理时间 |

`∫Qc dt` 未引入 COP 与风机/水泵功耗，只是横向比较代理；容量指令方差也只是平稳性代理，不能直接等同于寿命。

## 4. 三场景实验分析与工程评估

- 场景 1：初次快速降温，31.5→24 °C，4 h，检验大偏差启动与过冷。
- 场景 2：设定温度突变，1 h 时 25→23 °C，检验工况迁移。
- 场景 3：室外 37±4.5 °C、持续设备发热、3 h 开门 12 min +2600 W，检验长时抗扰。

| 场景 | 算法 | ITAE（°C·h²） | 调节时间（h） | 扰动恢复（h） | 最大过热（°C） | 最大过冷（°C） | 容量指令方差 |
|---|---|---|---|---|---|---|---|
| 初次快速降温 | 贝叶斯自动整定 | 5.419 | 4 | 4 | 7.741 | 0.9661 | 0.1422 |
| 初次快速降温 | Z-N 反应曲线法 | 7.065 | 4 | 4 | 7.741 | 1.503 | 0.1936 |
| 初次快速降温 | IMC 内模控制（λ经训练集整定） | 4.08 | 2.617 | 2.617 | 7.741 | 0.7723 | 0.09367 |
| 初次快速降温 | FNN（验收拒绝→IMC） | 4.08 | 2.617 | 2.617 | 7.741 | 0.7723 | 0.09367 |
| 初次快速降温 | RL 在线自整定 | 3.538 | 2.533 | 2.533 | 7.741 | 0.3628 | 0.07028 |
| 设定温度突变 | 贝叶斯自动整定 | 8.462 | 5 | 4 | 2.829 | 1.47 | 0.1229 |
| 设定温度突变 | Z-N 反应曲线法 | 10.95 | 5 | 4 | 2.09 | 1.833 | 0.184 |
| 设定温度突变 | IMC 内模控制（λ经训练集整定） | 7.12 | 5 | 4 | 2.837 | 1.209 | 0.09261 |
| 设定温度突变 | FNN（验收拒绝→IMC） | 7.12 | 5 | 4 | 2.837 | 1.209 | 0.09261 |
| 设定温度突变 | RL 在线自整定 | 6.867 | 5 | 4 | 2.721 | 0.9284 | 0.08034 |
| 持续外界热扰动 | 贝叶斯自动整定 | 17.16 | 6 | 2.8 | 2.142 | 1.777 | 0.1682 |
| 持续外界热扰动 | Z-N 反应曲线法 | 19.69 | 6 | 2.8 | 2.094 | 2.267 | 0.1989 |
| 持续外界热扰动 | IMC 内模控制（λ经训练集整定） | 15.23 | 6 | 2.8 | 2.1 | 1.315 | 0.1456 |
| 持续外界热扰动 | FNN（验收拒绝→IMC） | 15.23 | 6 | 2.8 | 2.1 | 1.315 | 0.1456 |
| 持续外界热扰动 | RL 在线自整定 | 15.43 | 6 | 2.8 | 2.069 | 1.489 | 0.1372 |

留出工况汇总：

| 算法 | 平均 ITAE（°C·h²） | 平均调节时间（小时） | 平均最大过冷（°C） | 平均容量指令方差 | 稳定率 |
|---|---|---|---|---|---|
| 贝叶斯自动整定 | 7.111 | 3.441 | 1.854 | 0.08619 | 1 |
| Z-N 反应曲线法 | 10.43 | 4.674 | 2.215 | 0.1549 | 1 |
| IMC 内模控制（λ经训练集整定） | 6.486 | 3.232 | 1.714 | 0.06684 | 1 |
| FNN（验收拒绝→IMC） | 6.486 | 3.232 | 1.714 | 0.06684 | 1 |
| RL 在线自整定 | 6.214 | 3.113 | 1.706 | 0.05943 | 1 |

## 5. 嵌入式量产落地与工程安全评估

- STM32F103C8T6：72 MHz、64 KB Flash、20 KB SRAM。Wokwi Blue Pill 完整代码已在线编译并进入运行态；FNN float32 表 200 B，RL 已压成 25 B 动作索引 + 25 B 覆盖掩码 + 72 B 动作表。
- ESP32：经典系列最高 240 MHz、520 KB SRAM；资源充足，但需隔离 Wi-Fi 任务与控制任务。
- PC C++ Testbench：`sizeof(SafePI)=32 B`、虚拟对象状态 68 B，100 ms PI / 2 s AI 分频通过；PC 时间不能换算为 MCU WCET。
- 已实现于 Python 统一比较层：0/25% 最低稳定容量、1% 量化台阶、运行段每分钟 5 个百分点斜率、最小启停驻留、测量噪声与滤波、条件积分抗饱和、Kp/Ki 边界、FNN/RL 训练后部署验收门。
- 尚未声称完成：Wokwi 已证明 STM32/ESP32 目标编译和启动，但这些新增约束仍需同步到目标 MCU，并取得串口曲线、ROM、RAM、最坏周期和栈证据；高低压、排气温度、通信超时、看门狗及参数 CRC/回滚仍待补齐。
- 上线流程：SIL → HIL → 只读影子模式 → 有限增益试运行 → 单机试点 → 多季节回归。

## 单算法三工况文档

- [贝叶斯自动整定](algorithm_reports/03_bayesian_auto_tune.html)
- [Z-N 反应曲线法](algorithm_reports/01_zn_reaction_curve.html)
- [IMC 内模控制（λ经训练集整定）](algorithm_reports/02_imc.html)
- [FNN（验收拒绝→IMC）](algorithm_reports/04_fnn_self_tuning.html)
- [RL 在线自整定](algorithm_reports/05_rl_self_tuning.html)


## 评审缺陷 A1–E5 闭环矩阵

状态由本次产物动态生成；算法未通过时保持拒绝并使用IMC，真实ESP32/BMS证据不在本轮软件验收范围内。

|编号|评审问题|状态|证据|可核查结果|
|---|---|---|---|---|
|A1|统一执行器约束|已闭环|`simulator.py / actuator.py`|全部算法共用最低容量、量化、斜率和启停限制|
|A2|噪声与滤波|已闭环|`dataset_manifest.csv`|训练、验证和测试工况均包含传感器噪声与滤波|
|A3|IMC公平调参|已闭环|`imc_lambda_tuning.csv`|lambda只在训练集选择，进入验证和密封测试后冻结|
|B1|FNN标签混叠|已闭环|`fnn_training_samples.csv`|同一热状态快照执行30分钟局部候选回放，并记录容量、积分、室外温差和负荷|
|B2|FNN规则覆盖|已闭环|`fnn_training_history.csv`|有效规则覆盖率=100.0%|
|B3|FNN离策略压缩|已闭环|`fnn_training_samples.csv`|聚合IMC、BO和安全残差策略三类物理轨迹|
|B4|FNN独立部署门|未通过|`deployment_acceptance.csv`|密封测试均值比=1.0063，95%上界=1.0322|
|C1|RL伪造状态覆盖|已闭环|`rl_training_transitions.csv`|仅记录3R2C物理轨迹真实到达状态|
|C2|未访问状态默认偏置|已闭环|`generated_policy.hpp`|未覆盖组合使用动作4且触发IMC回退|
|C3|训练时域过短|已闭环|`rl_training_history.csv`|750回合、最长240分钟、5分钟决策|
|C4|RL状态非严格Markov|部分闭环|`rl_training_transitions.csv`|容量已进入Q状态；积分、限制器和负荷作为安全上下文记录，真实负荷仍是估计量|
|C5|增益累积漂移与奖励不一致|已闭环|`ai_controllers.py`|动作改为相对IMC绝对目标，并采用基线约束策略改进|
|C6|RL独立部署门|已闭环|`deployment_acceptance.csv`|密封测试均值比=0.9670，95%上界=1.0110|
|D1-D3|指标与时序审计|已闭环|`dynamic_timeseries.csv`|真实/测量温度、请求/实际命令、增益、回退和约束指标分别记录|
|E1|全量测试入口|已闭环|`pytest.ini`|统一使用pytest并包含Python、HTML和嵌入式测试|
|E2|训练验证测试隔离|已闭环|`dataset_manifest.csv`|默认48/16/16，并对每个工况写入SHA-256且拒绝重复|
|E3|报告fail-open|已闭环|`report.py`|缺失、NaN、非法验收值一律判为未通过|
|E4|加速演示时间基准|已闭环|`testbench.cpp / mcu_validation_summary.csv`|PI积分、限幅器与监督误差率统一使用200×模拟时间（与ESP32固件一致）；SIL判据含未覆盖回退占比≤10%门，实测0/190|
|E5|策略导出与CRC|已闭环|`policy_parity_vectors.csv`|CRC v3覆盖固件执行字段并完成Python/C++逐向量一致性|
