# 07 DQN：用一个小网络估计九个动作

DQN（Deep Q-Network）在本项目中仍然是九动作的离散增益选择器。它没有改变
对象、奖励或策略周期，只把上一章按格子保存的 Q 表换成一个函数：
输入五个可见量，输出九个动作价值。这里的 replay 是保存历史
(s,a,r,s',done) 过渡、供随机 batch 重用的回放池；目标网络是较慢更新的
Qθ- 副本，用来生成相对稳定的下一状态 target。网络参数因此可以在相邻
观测之间共享，但训练需要这两个结构和梯度更新。

对嵌入式控制器而言，DQN 只位于 2 分钟的监督决策点。PI 仍然每 0.1 分钟
执行一次，延迟对象和 PI 积分器仍然连续运行。训练期的回放池、目标网络、
优化器和 epsilon 不属于冻结推理路径；它们是离线整定时才存在的工具。

## 先把两个时间尺度和调用边界分开

默认对象时长 120 分钟、物理步长 0.1 分钟，所以一回合推进 1200 次 PI 与
Plant 更新。TuningEnv 每 2 分钟调用一次 DQN，动作随后固定 20 个物理步。
一回合是 60 次策略转移，而不是 1200 次网络前向。

| 层次 | 周期 | 数据流 | 在线职责 |
|---|---:|---|---|
| PI/对象闭环 | 0.1 分钟 | 误差 → anti-windup PI → [0,1] 输出 → 延迟对象 | 保持实时控制和积分状态 |
| DQN 策略 | 2 分钟 | 五维观测 → 归一化 → 5→32→32→9 → argmax | 选择下一段的九个增益组合之一 |

core.py 的 advance 在每次动作后执行 20 次 PI.update 和 Plant.step，并在
物理步上计算 IAE。网络不会直接预测接下来 20 个温度，也不会取代
PI.update；它只决定这 20 步使用的 Kp、Ki。

控制器需跨决策保留 PI 积分输出、上一条指令、上一决策误差及斜率、当前增益和 SIMC 锚点；一次 DQN.choose 调用不清空这些状态。仿真器另外保留 Plant 温度、输入延迟队列和物理步索引，真实控制器无需为网络推理复制这套对象状态。reset 在这里表示新仿真回合。

## 五维观测与神经网络的尺度

原始观测为

$$
o=(e,\dot e,u,s_p,s_i),
$$

五个分量分别是当前温度减设定值、两分钟误差斜率、上一物理步的制冷输出、
当前 Kp 相对 SIMC 锚点的倍数、当前 Ki 相对锚点的倍数。

| 分量 | 原始单位/范围 | 固定除数 | 网络输入 |
|---|---|---:|---|
| e | ℃ | 4 | e/4 |
| \dot e | ℃/min | 0.2 | \dot e/0.2 |
| u | [0,1] | 1 | u |
| s_p | 无量纲 | 2 | s_p/2 |
| s_i | 无量纲 | 2 | s_i/2 |

rl.py 的 normalized_observation 用 [4, 0.2, 1, 2, 2] 做逐元素除法，并返回
float32。DQNPolicy.choose 对一条观测直接得到形状 (9,) 的向量；批量训练
时 learn_batch 把 state 列成 (B,5)，默认 B=64。这个 scale 是部署协议的一
部分，不能在固件中只换成“看起来更合适”的范围而不重新训练。

五个量仍然不是完整的物理状态。延迟队列、PI integral、对象的 gain/tau/delay
和扰动相位没有进入输入。相同的归一化观测可能来自不同的延迟队列，下一次
温度转移也可能不同，因此 DQN 学的是部分观测下的近似价值函数。回放的
随机化可以减弱相邻样本相关性，但不能补回缺失的状态信息。

## 网络输出表示什么

network() 构造的网络为

~~~text
(5,) -> Linear(5,32) -> ReLU -> Linear(32,32) -> ReLU -> Linear(32,9)
~~~

对一条观测，9 个输出分别是 Qθ(s,0) 到 Qθ(s,8)。这里的 Q 是“从当前
观测执行该动作后，未来折扣负 IAE 的估计”；由于 reward 是负 IAE，价值
越大表示预计误差代价越小。它们不是九个温度预测，也不是九个 [0,1] 制冷
占空比。
参数总数为

$$
(5\times32+32)+(32\times32+32)+(32\times9+9)=1545.
$$

DQNPolicy.choose 用 torch.no_grad 执行前向，再对九个输出调用 argmax。
PyTorch 的一维 argmax 在并列最大值时返回第一个索引，因此九个输出完全
相同时动作确定为 0；随机初始化网络通常并不全相同，此时 argmax 返回当时
最大值的最小索引。代码没有额外的安全回退规则。

TuningEnv.step 将这个整数动作查到九组 ACTION_SCALES，构造
Gains(anchor.kp*scale_p, anchor.ki*scale_i)。增益是锚点的绝对倍数，同一个
动作连续选两次仍是同一组 Kp、Ki，不会累乘。模型 checkpoint 只保存网络
state_dict；当前 DQN 文件没有把锚点、scale 或 interval 写进权重文件，部署
配置必须显式绑定这些约定。

## 从动作到 replay 记录

每个两分钟区间的 reward 是

$$
r_t=-\sum_{k\in\mathrm{interval}}|e_k|\Delta t,
\qquad \Delta t=0.1\ \mathrm{min}.
$$

默认 20 个物理步，因此一个 reward 是这个区间的负 IAE；完整回合的
history.return 是 60 个 reward 的直接相加，不折扣。环境的 done 只在
index == scenario.steps 时为真，终止转移不再 bootstrap。没有额外终端奖惩，
也没有把 movement 自动加入 reward。

训练中的一条 replay 记录是

$$
(s_t,a_t,r_t,s_{t+1},done_t),
$$

其中 s 和 s' 已是归一化后的形状 (5,) float32 数组，a 是整数，r 是标量，
done 是布尔值。代码用 deque(maxlen=10000) 保存这些 tuple；它不是一个会在
设备上持续增长的日志。

训练时先为 48 个对象各做一次 240 分钟 commissioning，得到并缓存 SIMC 锚点。
每个回合随机选一个对象，每个动作按 epsilon 选择随机整数或网络 argmax，
推进 20 个物理步，再把这条 transition 放入 deque。累计到 256 条后，代码
每次无放回随机抽 64 条，并把抽出的列表转换为 states (64,5)、next_states
(64,5)、actions (64,)、rewards (64,) 和 done (64,) 的 torch 张量。

默认 500 回合共有 30,000 条交互。前 255 条还不足 warm-up，因此总梯度更新
是 29,745 次；deque 在后期会丢弃最早的记录，只保留最近 10,000 条。它能
打散相邻动作的时间相关性，却不会把来自不同对象、不同延迟阶段的记录变成
完整 Markov 状态。

## 当前网络、目标网络和梯度停止

当前网络 Qθ 负责给实际执行的动作打分，目标网络 Qθ- 负责提供下一状态的
参考值。目标网络的职责是让一批更新期间的 target 相对稳定；它不是第二个
在线决策器。训练开始时两者复制同一组随机初始化参数，之后每 200 次梯度
更新才把当前网络硬复制到目标网络。

对 batch 中第 i 行，learn_batch 先用 gather 取实际动作列：

$$
q_i=Q_\theta(s_i,a_i).
$$

再在 no_grad 区域计算目标网络对下一状态的九个输出并取最大值：

$$
y_i=r_i+\gamma(1-done_i)\max_{a'}Q_{\theta^-}(s'_i,a'),
\qquad \gamma=0.99.
$$

done 为真时乘子为 0，所以终止行只使用 reward；非终止行才使用下一状态
估计。expected 张量在 no_grad 区域生成，不会把梯度传入目标网络。当前
网络用 Smooth L1（Huber）损失拟合它：

$$
\mathcal L=\frac1{64}\sum_{i=1}^{64}\operatorname{Huber}(q_i-y_i).
$$

随后 optimizer.zero_grad、loss.backward 和 optimizer.step 只更新 Qθ。
target.requires_grad_(False) 与 no_grad 两层约束共同表达了“目标只提供数值”
这一数据流。训练 loss 是预测当前 target 的误差，不是闭环 IAE。

<<< @/../hvac_pid/dqn.py#dqn_update

## 首个实际 batch 怎样核对

training.json 的 first_update 保存了真实抽样 batch，而不是为了讲解手写的
常数。它的整体形状是 states (64,5)、actions (64,)、rewards (64,)、
next_states (64,5)、done (64,)，predicted、future、targets 都是 (64,)，
loss 是一个标量。首行实际数值为：

| 首行字段 | 值 |
|---|---:|
| state | [0.0454453, 0.0497818, 0.6416025, 1.0, 0.25] |
| action | 3 |
| reward | -0.3424653113 |
| done | false |
| predicted | 0.0730380937 |
| future | 0.3048094809 |
| target | -0.0407039225 |
| batch Huber loss | 0.9902936220 |

对于这行，-0.3424653113+0.99×0.3048094809
=-0.0407039225；target 网络的计算不参与梯度。表中的 loss 是 64 行 Huber
值的均值，不能用首行 residual 单独代替。这个记录可以同时检查归一化、
gather 的动作列、done 掩码、target 网络和损失 reduction 是否与源码一致。

## 训练曲线与冻结评估

参考训练每回合选择一个训练对象，epsilon 从 1.0 按 episode 预算下降到
0.05。前四个完整回合只累积 replay，直到第五回合的第 16 个物理决策附近
才达到 256 条经验；因此 history 中早期 loss 为空是 warm-up 的结果，不是
网络计算失败。第 500 回合平均 loss 为 0.3378，500 回合 CPU 训练耗时为
58.5514 s；这些数混合了随机动作、对象和 replay 分布。

![DQN 训练回报和预测损失](/results/07-dqn/learning.svg)

<!--@include: ../generated/rl-training.md-->

训练完成后，model.pt 被加载为 eval 模式的 DQNPolicy，epsilon、replay、
target 网络和 optimizer 都不再参与选择。冻结策略每 2 分钟做一次
normalize→前向→argmax，然后把动作交给同一个 TuningEnv/PI 流程。

![DQN 冻结评估时的温控与在线增益](/results/07-dqn/response.svg)

<!--@include: ../generated/07-dqn.md-->

默认对象上 DQN 的 IAE 为 69.6565 ℃·min，SIMC 为 77.9950 ℃·min；两者最大
过冷都为 0，DQN movement 为 2.9924，SIMC 为 1.3750。验证分区平均 IAE
为 61.7822 ℃·min。这个结果说明参考配置在这些对象上降低了误差积分，但
它仍然只能选择九个离散增益组合，且 movement 是另一项约束；loss 变小本身
不能推出冻结响应一定更好。

## 嵌入式推理路径和持久状态

冻结 DQN 的在线路径只有：

$$
\text{五个原始观测}
\rightarrow\text{固定 scale 除法}
\rightarrow(5\to32\to32\to9)\text{前向}
\rightarrow\text{第一个最大值}
\rightarrow\text{锚点增益}
\rightarrow\text{PI 高频闭环}.
$$

在线控制器保留 PI 积分输出、上一决策误差及斜率、当前指令、当前增益与 SIMC 锚点。限幅直接作用于本拍计算值，没有独立的历史状态。Plant 温度、输入延迟队列与步索引仅属于仿真器；DQN 权重不保存或恢复这些闭环状态。

正式闭环不需要保存 replay deque、done 历史、first_update、Huber loss、
epsilon、target 网络或 Adam 状态。网络只做推理，且 DQNPolicy.choose 已用
no_grad 和 eval 模式；模型不会在线训练。由于 DQN 的 torch.save 文件只含
state_dict，部署封装还应保存网络版本、五个 scale、九个动作表、2 分钟
决策周期和 commissioning 锚点。

benchmark 的已测 CPU 数据把控制更新和冻结策略分开：

| 组件 | 调用周期 | 参数或表项 | 工件 bytes | 中位 / P95 μs |
|---|---:|---:|---:|---:|
| PI control update | 0.1 min | 2 | 0 | 1.2 / 1.9 |
| Q-Learning frozen policy | 2.0 min | 6075 | 48856 | 26.7 / 56.05 |
| DQN frozen policy | 2.0 min | 1545 | 8829 | 68.5 / 130.355 |
| PPO frozen actor | 2.0 min | 4804 | 22397 | 153.05 / 302.61 |

benchmark 协议是 CPU、batch=1、先 warm-up 100 次再测 1000 次；时间包含
观测转换，不含梯度和探索。artifact bytes 是当前参考文件的大小，不等于
链接器计算出的 MCU Flash/RAM；中位数和 P95 也只是该参考主机上的数据，
不能替设备验收时序。

## 结果复现入口

下面的命令使用 CPU 版 PyTorch 运行 DQN，并把训练报告、冻结模型、图和
响应记录写入 outputs/tutorial 对应目录。

~~~bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m hvac_pid dqn
python -m hvac_pid explain --method dqn --output outputs/tutorial
~~~
