# 五类 HVAC AI-PID 算法代码解析

本项目把“算法决定 `Kp/Ki`”和“PI 计算压缩机 PWM”分开。这样 Z-N、IMC、贝叶斯优化、FNN、RL 都能使用同一热模型、同一 PI 内环和同一安全限制，比较才公平。

## 0. 所有算法共用的对象与安全 PI

### 热对象：3R2C + 执行器滞后/惯性

实现：`hvac_pid/plant.py` 的 `ThermalPlant3R2C.step()`。

- `zone_c`：机柜/房间空气温度；`wall_c`：围护结构等效温度。
- 室外温度通过两条热阻向室内传热；设备负荷、人员热负荷和开门热负荷直接进入室内热平衡。
- `u ∈ [0, 1]` 是压缩机归一化制冷能力。它先经过纯滞后队列，再经一阶惯性成为实际 `cooling_w`。
- 热平衡的核心是：

  ```text
  C_zone · dT_zone/dt = (T_out-T_zone)/R_out_zone
                       + (T_wall-T_zone)/R_zone_wall
                       + Q_load - Q_cooling
  ```

因此 `e = T_zone - T_setpoint > 0` 表示房间偏热，需要增加制冷。

### PI 内环：所有算法的共同执行器

实现：`hvac_pid/controllers.py` 的 `PIController.update()`。

```text
I[k] = I[k-1] + e[k] · Δt
u_raw = Kp · e[k] + Ki · I[k]
u = clamp(u_raw, 0, 1)
```

代码不是无条件积分：当输出已经饱和且当前误差会继续把输出推向饱和方向时，不更新积分项。这是**条件积分抗饱和**，避免压缩机长时间 100% 后积分累积，随后导致严重过冷。

`hvac_pid/ai_controllers.py` 中的 `_SafeAdaptivePI` 再给 FNN/RL 增加四层安全保护：

1. AI 只允许每 2 s 更新一次，PI 预留给 100 ms 执行；
2. `Kp ∈ [0.002, 1.5]`、`Ki ∈ [1e-5, 0.08]`；
3. 单次增益变化不超过当前值的 25%；
4. 输入/模型异常时立即恢复 IMC 的 `fallback_gains`。

## 1. Ziegler–Nichols（Z-N）反应曲线 PI

实现：`identify_fopdt()`、`ziegler_nichols_pi()`，位于 `hvac_pid/controllers.py`。

### 第一步：从阶跃响应辨识 FOPDT

`identify_fopdt()` 在稳定工况给压缩机一个小阶跃，得到：

```text
G(s) = K · exp(-L·s) / (τ·s + 1)
```

- `K`：压缩机指令改变一个单位造成的稳态温度变化量（项目中取正的制冷增益绝对值）；
- `L`：压缩机延迟；
- `τ`：热对象时间常数。

当前代码先运行 168 h 的加速虚拟阶跃，再用有界最小二乘同时拟合温降幅度 `A`、时间常数 `τ` 和延迟 `L`，最后计算 `K=A/Δu`。`t28/t63` 仍被记录，但只用于检查曲线是否确实走过关键响应区间，不再直接决定参数。这样修正了旧版 12 h 试验尚未到达 63.2% 却把末点当作 `t63` 的截尾问题。全部候选保存在 `outputs/fopdt_fit_history.csv`，逐项计算保存在 `outputs/classical_tuning_steps.csv`。真实 OpenModelica/BMS 导出 CSV 后也应采用曲线拟合，并加入噪声和扰动筛选。

### 第二步：代入 Z-N 公式

```text
Kp = 0.9τ / (K·L)
Ti = 3.33L
Ki = Kp / Ti
```

特点：响应快、易过冲。机柜空调热惯性大且延迟明显时，Z-N 往往导致较高 PWM 波动和过冷；它在本项目中是“激进经典基线”。最后的 `np.clip` 是工程安全限制，不属于 Z-N 原公式。

## 2. IMC（内模控制）PI

实现：`imc_pi()`，位于 `hvac_pid/controllers.py`。

IMC 同样用上面的 FOPDT 模型，但主动选择闭环时间常数 `λ`：

```text
λ = max(τ/3, 3L, 12 min)
Kp = τ / [K(λ+L)]
Ti = min(τ, 4(λ+L))
Ki = Kp / Ti
```

`λ` 越大，控制越稳但恢复越慢。上述公式值保留为**保守故障回退**；用于性能比较的 IMC 通过 `tune_global_imc_lambda()` 只在训练工况搜索 λ，Kp/Ti 仍受 IMC 公式约束，测试集不再回调参数。这样能公平回答“IMC 是结构不合适，还是 λ 没选对”。

## 3. 贝叶斯优化（Bayesian Auto-tune）固定 PI

实现：`BayesianGainTuner`、`tune_global_fixed()`，位于 `hvac_pid/tuning.py`。

### 搜索变量和评价函数

直接搜索 `log(Kp), log(Ki)`，不是在线搜索压缩机 PWM。对数空间可处理 `Ki` 跨多个数量级的问题。每个候选参数均运行完整闭环仿真，并由 `calculate_metrics()` 计算目标函数：

```text
J = 2·IAE + 1.5·ITAE + 10·舒适区违规
  + 3·最大过冷 + 0.35·调节时间
  + 0.08·控制变化量 + 0.7·容量指令方差 + 能耗项
```

不稳定或温度越界的轨迹会收到 `1e6` 大惩罚。

### 为什么是“贝叶斯”

1. 先用 IMC、Z-N 与随机点初始化样本；
2. 用高斯过程 `GaussianProcessRegressor` 近似 “增益 → J”；
3. 对大量候选点计算 EI（Expected Improvement，期望改进）；
4. 选择 EI 最大的点做下一次真实仿真；
5. `tune_global_fixed()` 把多个训练工况上的平均 `J` 作为目标，最终输出**一套全局固定** `Kp/Ki`。

因此该算法离线运行，结果写入 `outputs/global_bayesian_tuning.csv`；运行时没有学习负担，最适合“只允许固定参数”的低成本 MCU。

## 4. 稀疏 FNN 在线自整定

实现：`FNNGainController`，位于 `hvac_pid/ai_controllers.py`。

这里的 FNN 是面向 MCU 的**零阶 TSK 模糊神经/规则面增益调度器**，而不是大规模深度网络。它需要训练：先对一批虚拟工况执行贝叶斯优化，得到每个工况的优质 `Kp/Ki` 标签；再把这些标签按 `(e, Δe)` 落入的模糊格子学习为 25 个规则后件，保存为 `fnn_rule_table.npy`。

### 输入和模糊化

- 输入 1：`e=Tzone-Tsetpoint`，中心点为 `[-3, -0.75, 0, 1.5, 5] °C`；
- 输入 2：误差变化率 `ė=Δe/Δt`，中心点为 `[-0.30, -0.05, 0, 0.05, 0.30] °C/min`；使用变化率而不是“每个采样的变化量”，使 PC 的 5 min 训练节拍和 MCU 的秒级节拍具有相同物理含义；
- 每个输入由 `_active()` 找到相邻的两个三角隶属函数和插值权重。

完整规则面为 `5 × 5 = 25` 条规则；但每次只有 `2 × 2 = 4` 条相邻规则非零。因此 `_propose()` 的双层循环只执行四次加权计算。

### 规则输出

运行时，每条规则从离线训练得到的表中读取候选参数：

```text
[Kp_rule, Ki_rule] = fnn_rule_table[误差格子, 误差变化格子]
Kp(t), Ki(t) = 4 条相邻规则的加权平均
```

直观上：训练把“这种偏热程度、这种升温/降温趋势下，哪组旋钮在离线仿真中更好”记进规则表；运行时只需查四个邻格并加权。最后仍必须经过共同的增益限幅和变化率限制。

训练集采用冷热启动、设定值上调/下调、开门热脉冲和持续负荷变化组成的分层课程。规则表在内部验证集上选择 IMC 先验权重；只有覆盖率不少于 80% 且平均目标不劣于 IMC 才允许部署，否则 `fnn_rule_table.npy` 自动写成 IMC 回退表，同时把未通过候选保存在 `fnn_rule_table_candidate.npy` 供审计。

## 5. 离线 Q-learning RL 安全自整定

实现：`train_offline_q_policy()` 与 `IncrementalRLController`，位于 `hvac_pid/ai_controllers.py`。

### 离线训练

状态空间非常小：

```text
thermal_state = bin(e, 5 档) × bin(ė, 5 档) = 25 个热状态
command_mode = bin(上一次实际容量指令, 3 档)
action = 相对 IMC 的绝对 Kp 目标比例 × Ki 目标比例 = 9 个动作
Q-table shape = 5 × 5 × 3 × 9
```

训练函数在同一个 **3R2C 房间 + 受约束压缩机 + 带噪传感器** 虚拟环境中执行 750 个 episode，每回合最长 240 min、每 5 min 决策。25 热状态覆盖率只计算物理轨迹实际到达的状态，不再随机伪造首个误差变化率来凑满网格；同时另行记录 75 个“热状态×容量模式”组合的自然覆盖率。冷房间高制冷等不安全组合不会为了凑数而注入。

```text
r = mean_interval[-|误差|-1.5·舒适带超限-0.08·|Δu|-0.02u]
    - 0.08·|实际增益变化| - 0.015·|目标比例偏离1|
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',·)) - Q(s,a)]
```

训练产物保存为 `outputs/rl_q_table.npy`。它是“先在仿真中学习，后在控制器中查询”，不是让真实机柜空调试错。

### 在线执行

`IncrementalRLController`（为兼容旧接口保留类名）只需：状态分箱 → 屏蔽不安全动作 → `argmax(Q[state])` → 取得相对 IMC 的绝对增益目标 → 走同一 ±10% 安全变化层。绝对目标修复了旧版“当前 Kp/Ki 未进入状态却递归相乘”的非马尔可夫问题；上一次实际容量模式用于区分停机、部分负荷和高负荷。训练期间每隔若干回合在内部验证集早停选表，最终若平均目标比 IMC 差 2% 以上或25热状态覆盖低于80%，整张策略拒绝部署并退回 IMC。

## 6. 如何读结果

- `ITAE` 小：误差消除得早，长时间偏差惩罚更大；
- `settling_time` 小：更快进入并保持在 ±0.5 °C 舒适带；
- `max_undershoot` 小：过冷更轻；
- `compressor_output_variance` 小：容量指令更平稳；它只是代理量，不能直接换算压缩机寿命；
- `fallback_events` 非零：说明自适应控制触发了保护，应调查异常输入或策略边界。

不能只按单一目标函数排名。精密/机柜空调更应同时看稳定率、ITAE、扰动恢复、容量指令方差和启停次数。

## 7. 旧代码说明

`hvac_pid/scheduler.py` 的 `GainScheduler` / `ScheduledPIController` 是项目原有的“监督学习工况→增益”实验代码。它保留用于兼容与对照，但正式报告和 Streamlit 的五组对比不再使用它，避免把算法数目变成六组。
