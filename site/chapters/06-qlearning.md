# 06 Q-Learning：每两分钟选择一次增益

Q-Learning 在本项目中承担的是慢速增益调度，而不是替代 PI 控制器。Q(s,a)
表示从当前观测执行动作 a 后未来折扣 reward 的估计；由于 reward 是负 IAE，
Q 越大表示预计误差代价越小。PI 仍然每
0.1 分钟（6 秒）读取误差并产生一次制冷输出；Q-Learning 只在每个 2 分钟区间
开始时读取一次观测，选择下一段要使用的 Kp、Ki。这个时间尺度的分离决定了
部署边界：Q 表是低频决策器，PI 的积分器、延迟执行状态和输出限幅仍然属于
实时控制路径。

动作不是把上一次增益继续乘上一个倍数，而是相对于本对象 commissioning 时
得到的 SIMC 锚点选择一个绝对组合。这样做把增益范围约束在九个可审查的点，
同时让不同对象的绝对 Kp、Ki 由各自的锚点承担。模型文件保存的是策略表；
锚点、动作表和控制器的运行时状态仍需要由部署程序保存。

## 先看两个时间尺度

默认 Scenario 的物理步长是 0.1 分钟，时长是 120 分钟，因此一个回合有
1200 个物理步。TuningEnv 的默认 decision interval 是 2.0 分钟，正好对应
20 个物理步；一个完整回合因此有 60 次增益决策。

| 层次 | 周期 | 每次做什么 | 需要保留的状态 |
|---|---:|---|---|
| PI 与仿真对象 | 0.1 分钟 | 用当前误差更新积分器，输出 [0,1] 制冷指令；对象推进一步 | 仿真的 PI integral、Plant 温度、延迟队列、上一条指令 |
| Q-Learning 策略 | 2 分钟 | 把观测映射成离散状态，查九个 Q 值并选动作 | Q 表、SIMC 锚点、上次增益和观测派生量 |

core.py 中的 advance 会在一个策略动作下循环 20 次 PI.update 和 Plant.step。
它对每个物理步累加 |e|dt，而不是每两分钟只看一次温度。于是模型的一个
reward 是一个区间的绝对误差积分，PI 的高频积分行为仍然完整地参与了评分。

运行时的状态不会随着策略调用自动重置。reset 只用于开始一个新回合；真实
设备中相当于进程启动、控制回路重新接管或明确的运行模式切换。若每次 2 分钟
调用都重新初始化积分器或延迟队列，得到的响应就不再是本实现的响应。

## 五个可见量，以及看不见的状态

每次策略决策拿到的原始观测是

$$
o_t=(e_t,\dot e_t,u_{t-1},s_{p,t},s_{i,t}).
$$

| 分量 | 单位或范围 | core.py 中的来源 | 含义 |
|---|---|---|---|
| e | ℃ | plant.temperature - setpoint | 当前温度相对设定值的误差 |
| \dot e | ℃/min | 两次策略决策间误差差 / 实际 elapsed | 上一个 2 分钟区间的误差变化率 |
| u | [0,1] | PI.update 的最后输出 | 上一个物理步的制冷指令 |
| s_p | 无量纲 | 当前 Kp / anchor.kp | 当前比例增益相对锚点的倍数 |
| s_i | 无量纲 | 当前 Ki / anchor.ki | 当前积分增益相对锚点的倍数 |

误差和斜率来自区间两端，u 是进入下一次决策时已经存在的输出。最后一个
不满 decision interval 的区间仍按实际 elapsed 计算斜率；默认 120 分钟可以
被 20 个物理步整除，所以参考运行没有这个尾段差异。

这五个量不是完整对象模型。Plant 内部的延迟队列没有放进观测，PI 的积分
贡献也没有单独暴露；对象的 gain、tau、delay、扰动相位同样不是输入字段。
两个时刻可能有相同的五个数，却有不同的延迟队列或积分器状态，下一段温度
响应便可能不同。因此这里的离散状态只是工程上选择的观测表示，存在状态
别名和非 Markov 风险，不能把它当作已经包含全部物理状态的收敛证明。

神经网络章节使用同一组五个量，但先按

$$
\text{scale}=[4,\;0.2,\;1,\;2,\;2]
$$

逐分量相除，得到 float32 的五维输入。Q-Learning 不使用这个连续归一化；
它直接对原始误差、斜率和输出分箱。这个区别很重要：改变神经网络的 scale
不会改变本章 Q 表的索引，但改变阈值会直接改变表中状态。

## 分箱、状态索引与表的尺寸

rl.py 的 state_index 对前三个连续量调用 NumPy digitize：

| 量 | 分割点 | 档位数 |
|---|---|---:|
| e | -1.0、-0.2、0.2、1.0 ℃ | 5 |
| \dot e | -0.1、-0.02、0.02、0.1 ℃/min | 5 |
| u | 0.33、0.67 | 3 |

例如误差落在 [-0.2, 0.2) 时是中间档；边界遵循 digitize 的默认规则。
当前增益不再用阈值分箱，而是计算它与九个 ACTION_SCALES 行的平方距离，
取距离最小的行。由于环境动作总是九个组合之一，正常运行时这个索引正好
是 0 到 8。

因此状态索引本身是四元组 (e_bin, d_bin, u_bin, gain_index)，不是五元组：
第五个原始观测分量 s_i 和第四个 s_p 一起被压缩成一个九组合索引。逻辑上
可以把四元组展平为

$$
\mathrm{id}=(((e_{\rm bin}\times5+d_{\rm bin})\times3+u_{\rm bin})
\times9+\mathrm{gain\_index}),
$$

但代码没有另建一列整数 ID，而是直接用元组索引 NumPy 数组。这样共有

$$
5\times5\times3\times9=675
$$

个离散状态；每个状态保存九个动作价值，所以 q 的实际形状是
 (5,5,3,9,9)，总数为 6075 个浮点值。最后一个维度是动作维度，不能误读
为温度预测或九个连续制冷指令。

QPolicy.choose 对 q[state_index(observation)] 调用 np.argmax。NumPy 在并列
最大值时返回第一个位置，所以完全未访问、九个值都仍为 0 的状态也确定地
选择动作 0。这个 tie rule 是代码行为，不是针对过冷或输出变化的安全回退。
训练中的探索动作则由 epsilon 分布随机抽取 0 到 8；探索和贪心选择是两条
不同路径。

## 动作是锚点的绝对倍数

每个对象在训练开始前通过一次 240 分钟阶跃辨识得到 SIMC 锚点
anchor=(Kp0,Ki0)。动作表是：

| 动作 | Kp/Kp0 | Ki/Ki0 |
|---:|---:|---:|
| 0 | 0.5 | 0.5 |
| 1 | 0.5 | 1 |
| 2 | 0.5 | 2 |
| 3 | 1 | 0.5 |
| 4 | 1 | 1 |
| 5 | 1 | 2 |
| 6 | 2 | 0.5 |
| 7 | 2 | 1 |
| 8 | 2 | 2 |

TuningEnv.step 先取 ACTION_SCALES[action]，再构造
Gains(anchor.kp * scale_p, anchor.ki * scale_i)。因此连续两次选 8
仍然是 (2Kp0,2Ki0)，不会在第二次变成四倍。动作 4 是锚点本身；代码没有
另设一个“失败后回退”的隐式动作。

这也说明换对象时为什么不能只拷贝 model.npz。Q 表学到的是“观测下选哪个
倍数组合”，不是绝对的 Kp、Ki。若把表接到错误的 commissioning 锚点上，
动作编号合法，物理增益却会整体偏移；实际装置应把锚点版本和模型版本一起
记录。

## Reward、终止和一次环境转移

在一个动作区间内，core.py 对每个物理步累加

$$
\mathrm{IAE}_{t}=\sum_{k\in\mathrm{interval}}|e_k|\Delta t,
\qquad r_t=-\mathrm{IAE}_{t}.
$$

若这 20 个物理步的平均绝对误差为 0.3 ℃，该区间 reward 就是
 -20×0.3×0.1=-0.6。它没有能耗项、动作变化惩罚或终端奖金。默认回合
 60 个区间的 reward 直接求和等于完整回合不折扣 IAE 的负值。

<<< @/../hvac_pid/core.py#environment_step

TuningEnv._step_gains 让 advance 实际执行 min(count, remaining_steps)，再把
最后误差计算成下一条观测；done 只有 index == scenario.steps 时为真。默认
对象每次正好运行 20 步，但实现也能处理末尾不足一段的回合。TD 学习在 done
时不再 bootstrap；评估中的 IAE 始终把所有物理步按 dt 积分。

训练的 gamma=0.99 只出现在 TD target，不改变 history.return 的直接求和：
训练目标的折扣回报和报告的 IAE 是两个量。前者用于更新局部价值，后者用于
比较完整温控响应，不能用其中一个替换另一个。

## 实际发生的 TD 更新

Q 值是一个状态动作对的长期回报估计。一次转移先读取当前格子的旧值，再
根据是否终止生成目标：

$$
y_t=
\begin{cases}
r_t,&\mathrm{done},\\
r_t+\gamma\max_{a'}Q(s_{t+1},a'),&\mathrm{otherwise}.
\end{cases}
$$

然后计算 TD 误差并只改当前状态、当前动作的那个格子：

$$
\delta_t=y_t-Q(s_t,a_t),\qquad
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+0.15\delta_t.
$$

<<< @/../hvac_pid/rl.py#td_update

参考记录的第一条逐转移更新来自实际训练运行，并不是手写示例：

| 量 | training.json 中的值 |
|---|---:|
| state | [4, 2, 0, 4] |
| action | 5 |
| reward | -8.1255066466 |
| next_state | [4, 3, 2, 5] |
| done | false |
| old_q | 0 |
| td_error | -8.1255066466 |
| new_q | -1.2188259970 |

表从零开始，下一状态的九个动作仍全为零，所以 max Q(s',a')=0，target 就是
 -8.1255066466。代入 α=0.15 得到
 0+0.15×(-8.1255066466)=-1.2188259970，与文件记录一致。

<!--@include: ../generated/td-example.md-->

这个更新是逐转移立即执行的；Q-Learning 没有 replay buffer、target network
或反向传播。它的可审计性来自索引和数值都很小，代价是每个离散格子只能
独立积累样本，状态别名会把不同物理阶段的 reward 合并。

## 训练预算、冻结评估与参考结果

训练开始时，48 个训练对象各做一次 commissioning，之后每个回合随机选择一个
已缓存的锚点。epsilon 按 episode index 从 1.0 下降，在总预算的 80% 处到
0.05 并保持；探索动作仍然完整地推进环境并写入 Q 更新。500 回合因此产生
30,000 个两分钟转移、600,000 个物理步和 115,200 个 commissioning 物理步。

参考运行访问到 288 个离散状态，675 个状态中仍有 387 个没有访问。增加回合
预算会增加已见状态和同一格子的样本，但不会恢复被观测丢掉的延迟队列，也
不会把九个动作变成连续动作。这个预算效应是理解曲线的依据，不应只看最后
一个 return 数字。

![Q-Learning 的训练回报与探索概率](/results/06-qlearning/learning.svg)

history 中第 1、250、500 回合 return 分别为 -80.0258、-74.0704、-55.7562；
它们混合了不同训练对象和当时的 epsilon。训练耗时记录为 5.8869 s，不能
把这个 host 上的秒数当成目标设备的保证。

![训练三个阶段的一张 Q 表策略切片](/results/06-qlearning/qtable.svg)

策略切片固定输出档位和当前增益，只改变误差、斜率档位。图中的数字是
argmax 动作编号，不是温度预测。切片变化说明部分格子被更新，不代表所有
675 个状态都已经覆盖。

训练完成后，model.npz 被加载为冻结 QPolicy。冻结评估关闭 epsilon，不再写
Q 表；simulate 只重复“取观测、查表、运行 20 个物理步”。默认对象的实际
指标如下，均来自同一份 reference metrics：

![冻结 Q 表后的温度、输出和在线增益](/results/06-qlearning/response.svg)

<!--@include: ../generated/06-qlearning.md-->

Q-Learning 的默认对象 IAE 是 72.0293 ℃·min，SIMC 是 77.9950 ℃·min；最大
过冷为 0.0297 ℃，movement 为 5.1668，而 SIMC movement 为 1.3750。这个
参考运行减少了误差积分，却增加了输出变化，说明“IAE 更小”不是唯一的
嵌入式验收指标。验证分区平均 IAE 是 81.4143 ℃·min，不能用默认对象曲线
替代训练分布上的评价。

## 冻结运行的状态与计算量

冻结策略的调用路径可以写成：

$$
\text{传感器误差、斜率、输出、增益比}
\rightarrow\text{四个索引}
\rightarrow\text{Q 表九值}
\rightarrow\text{argmax}
\rightarrow\text{锚点倍数}
\rightarrow\text{PI 每 0.1 分钟运行}.
$$

先区分仿真器和真实控制器：TuningEnv/Plant 的温度变量与 delay queue 是
仿真环境状态，真实设备不需要为了运行 Q 表而复制一个 Plant 模型。控制器
本身需要持续保存 PI 的 integral、实际传感器/执行器状态、上一决策误差、
当前 command 和当前 Kp/Ki；若执行器有硬件缓冲，那是设备自己的运行态，
不是策略表要维护的延迟队列。对应的 SIMC anchor 也必须作为配置持久化。
设备若能直接提供温度和输出，观测的前两项仍要按本实现的时间间隔计算；
策略没有从 Q 表中推回这些量的办法。

在线不需要 history.csv、visits、snapshots、epsilon、TD error 或训练用的
commissioning 轨迹。训练可以离线完成；正式闭环只做分箱、元组索引和长度
9 的 argmax。模型文件本身由 QPolicy.save 写入 q 数组，没有把 anchor 和
阈值写进同一个元数据结构，因此固件配置需要显式绑定这两类约定。

benchmark 的实际 CPU 参考行如下：

| 组件 | 决策周期 | 参数或表项 | 工件 bytes | 中位 / P95 μs |
|---|---:|---:|---:|---:|
| PI control update | 0.1 min | 2 | 0 | 1.2 / 1.9 |
| Q-Learning frozen policy | 2.0 min | 6075 | 48856 | 26.7 / 56.05 |
| DQN frozen policy | 2.0 min | 1545 | 8829 | 68.5 / 130.355 |
| PPO frozen actor | 2.0 min | 4804 | 22397 | 153.05 / 302.61 |

这些数字来自 100 次 warm-up、1000 个 batch=1 调用的 CPU wall-time，包含
观测转换，不含梯度和探索。bytes 是当前模型工件文件大小，不是固件链接
后的 RAM/Flash 占用；μs 是参考主机的观测，不能外推为任何 MCU 的时序承诺。
Q 表在本实现中由 NumPy 默认浮点类型保存，所以表文件大小还包含 npz 封装。

## 结果复现入口

下面的命令生成默认的 500 回合参考风格产物；输出目录中的 model.npz、
training.json、history.csv、图和响应记录对应本章说明。

~~~bash
python -m hvac_pid qlearning
python -m hvac_pid explain --method qlearning --output outputs/tutorial
~~~
