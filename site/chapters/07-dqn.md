# 07 DQN：用一个小网络估计九个动作

Q-Learning 为每个离散状态单独保存 9 个数。DQN 保留相同的对象、观测、动作、2 分钟决策间隔和奖励，但用一个神经网络从连续观测直接预测 9 个动作价值。网络可以在相邻观测之间共享参数，代价是训练更复杂、价值估计变成近似值。

## 安装 CPU 版并运行

DQN 只需要 CPU 版 PyTorch：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m hvac_pid dqn
python -m hvac_pid explain --method dqn --output outputs/tutorial
```

训练产物在 `outputs/tutorial/07-dqn/`，单步/单批解释写入 `outputs/tutorial/explain/dqn.json`。解释记录包含真实 helper 运行的输入 batch、预测值、目标值、Huber loss 和参数更新前后的中间量；不要用手写的假 batch 替代它。

| 文件 | 内容 |
|---|---|
| `model.pt` | CPU 网络权重 |
| `history.csv` | 每回合 return、epsilon、平均 Huber loss |
| `training.json` | 训练成本、验证 IAE 和首个 batch 更新记录 |
| `learning.svg` | 探索回报和算法相关的 loss/epsilon |
| `response.svg` | 冻结 DQN 的温度、输出和在线增益 |
| `metrics.json` | DQN 与 SIMC 的统一闭环指标 |

基础章节不导入 DQN 模块，因此不装 torch 也能运行前面的 BO、FNN 和 Q-Learning。

## 输入和网络形状

DQN 使用上一章的长度 5 观测

$$
o=(e,\dot e,u,s_p,s_i),
$$

先除以固定尺度 `[4,0.2,1,2,2]`，得到一个形状为 `(5,)` 的 `float32` 向量。批量训练时状态和下一状态都是 `(64,5)`。64 是回放池一次抽取的 batch size，不是温度通道数。

网络结构是

```text
(5,) -> Linear(5,32) -> ReLU -> Linear(32,32) -> ReLU -> Linear(32,9)
```

因此单条前向输出形状为 `(9,)`，批量输出形状为 `(64,9)`；第 `a` 列是离散动作 `a=0..8` 的价值，不是 9 个温度预测，也不是 9 个制冷指令。参数总数是

$$
(5\times32+32)+(32\times32+32)+(32\times9+9)=1545.
$$

选动作时取 9 个值的 `argmax`，并使用与 Q 表相同的“第一个最大值”规则。PI 仍在每个 0.1 分钟物理步中产生制冷输出，DQN 只有在每 2 分钟区间开始时才调用一次。

## 回放池把不同阶段混在一个 batch

每次环境步保存一条

$$
(s,a,r,s',done)
$$

其中 `s,s'` 存的是归一化后形状 `(5,)` 的数组，`a` 是标量整数，`r` 是单位为 ℃·min 的负区间 IAE，`done` 是布尔值。回放池最多 10,000 条；累积到 256 条后，每次随机抽 64 条训练。500 回合、每回合 60 个两分钟转移，最终有约 30,000 条交互，因此 deque 会丢掉最早的记录，只保留最近 10,000 条；随机 batch 仍跨越多个对象和温度阶段。

随机抽样的作用是减弱相邻环境步的相关性。它不会增加信息量，也不会把部分观测变成完整状态；延迟队列仍然没有输入网络。

## 当前网络和目标网络

当前网络参数为 $\theta$，目标网络参数为 $\theta^-$。对 batch 中第 $i$ 条经验，当前网络只取实际动作那一列：

$$
q_i=Q_\theta(s_i,a_i).
$$

目标网络对下一状态输出 9 个值并取最大：

$$
y_i=r_i+\gamma(1-done_i)\max_{a'}Q_{\theta^-}(s'_i,a'),
\qquad \gamma=0.99.
$$

训练使用 Huber（Smooth L1）损失

$$
\mathcal L=\frac1{64}\sum_{i=1}^{64}\operatorname{Huber}(q_i-y_i).
$$

反向传播只更新当前网络；目标网络不接收梯度，每完成 200 次梯度更新才硬复制一次当前权重。

<<< @/../hvac_pid/dqn.py#dqn_update

解释命令的 `first_update` 可以逐项复算：读取 `states` `(64,5)`、`actions` `(64,)`、`rewards` `(64,)`、`next_states` `(64,5)` 和 `done` `(64,)`，运行 `net(states).gather(1, actions[:,None]).squeeze(1)` 得到 `(64,)` 的预测，运行冻结 `target(next_states).max(dim=1).values` 得到 `(64,)` 的 future，再按上式生成 `(64,)` 的 targets，最后计算一个标量 loss。JSON 保存的是实际 batch 中的数字和 loss，因而它能发现网络维度、done 掩码或 target 同步改错。

这里同样要区分目标和评估：DQN 的 TD target 使用 $\gamma=.99$，参考表中的统一 IAE 仍然是不折扣的完整 120 分钟误差积分。训练 loss 下降说明当前网络更接近此刻的 target，不等于默认对象的 IAE 必然下降。

## 参考训练量和曲线

![DQN 训练回报和预测损失](/results/07-dqn/learning.svg)

参考运行 500 回合产生 30,000 次两分钟转移、600,000 个物理步和 29,745 次梯度更新。第 1 回合还没有 256 条经验，所以 loss 为空；第 256 回合已经开始更新，记录的平均 loss 为 `0.5104`；第 500 回合 loss 为 `0.3378`。探索回报依然混合了 epsilon 随机动作和不同训练对象，不能把最后一条 loss 曲线当成控制性能曲线。

<!--@include: ../generated/rl-training.md-->

## 冻结评估：动作空间仍然只有九格

训练完成后，`model.pt` 被加载为冻结策略，评估关闭 epsilon。参考附带运行在默认对象上得到：

![DQN 冻结评估时的温控与在线增益](/results/07-dqn/response.svg)

<!--@include: ../generated/07-dqn.md-->

DQN 的 IAE 是 `69.6564946135 ℃·min`，SIMC 是 `77.9949859105 ℃·min`；DQN 最大过冷为 0，movement 为 `2.9924091046`，高于 SIMC 的 `1.3750302331`。验证分区平均 IAE 为 `61.7822 ℃·min`，与默认对象的顺序不同，说明一次标称曲线不足以概括训练分布上的行为。

DQN 没有超过附带 BO/FNN 的默认 IAE，不能据此推出“网络不适合整定”：它只能在 9 个相对锚点动作中选，BO 可以在连续 log 参数域提出任意候选；同时 DQN 还面对部分观测、有限回合和目标网络滞后。这些是本实现的具体边界。

## 失败方式：loss 变好，曲线不变好

目标网络让训练较稳定，但 target 本身随着复制而改变。若 replay batch 的分布、epsilon 或训练对象发生变化，loss 的绝对值也会变化。即使 loss 下降，策略仍可能频繁选择高 movement 的动作，或在延迟状态下过早加大 Ki。要判断控制效果，必须冻结网络后重新跑完整评估，并同时看 IAE、过冷和 movement。

动作仍然是相对对象的 SIMC 锚点。模型只输出动作编号，不输出绝对 Kp、Ki；把网络接到不同 commissioning 锚点上会改变实际增益。网络也没有访问真实延迟队列，较长延迟工况下可能把相同可见观测当作相同状态。

## 练习：只增大回合预算

```bash
python -m hvac_pid dqn --episodes 1000 --seed 0 --output outputs/dqn-1000
python -m hvac_pid explain --method dqn --output outputs/dqn-1000
```

验证以下量：

1. `cost.transitions` 是否从 30,000 变为 60,000，`updates` 是否随 replay 可用步数增加；
2. 网络权重形状和 `first_update` 的 batch 形状是否仍为 `(64,5)`、`(64,)`、`(64,9)` 对应的 gather 结果；
3. 最后 loss 的变化是否同时伴随冻结评估 IAE 的变化；
4. movement 是否因动作更频繁而上升。

如果只看 `history.csv` 最后一列，不跑冻结响应，就无法知道训练改进是否传到温控目标。

## 在线位置和 CPU 成本

DQN 的训练发生在离线环境。参考 500 回合 CPU 训练耗时约 `61.9990 s`，包含训练对象的 commissioning，不含验证和绘图；设备不同会改变这个数字。在线冻结策略每 2 分钟做一次 `5→32→32→9` 前向，默认 60 次/回合，单线程 CPU 的计算量很小；模型不需要在线调用服务，PI 仍负责高频控制。

因此 DQN 的在线成本低，主要负担在训练和验证数据；部署前还必须保存与模型匹配的锚点约定、动作表和版本。下一章把动作提议从离散增益改为模型服务的工具调用：LLM 每次试一组固定参数，再由同一个仿真器评分。
