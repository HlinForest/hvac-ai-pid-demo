# 08 PPO：连续动作先试，再把更新限制住

Q-Learning 和 DQN 只能在 9 个增益组合中选择。PPO 把动作改成两个连续的相对倍数：每 2 分钟直接给出 Kp、Ki 相对 SIMC 锚点的 log2 倍数。它在一条轨迹上采样动作，用 GAE 估计优势，再用 clipped ratio 做多轮小更新。

## 跑通命令和产物

需要 CPU 版 torch：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m hvac_pid ppo --episodes 500 --seed 0 --output outputs/tutorial
python -m hvac_pid explain --method ppo --output outputs/tutorial
```

产物在 `outputs/tutorial/10-ppo/`，解释记录在 `outputs/tutorial/explain/ppo.json`：

| 文件 | 内容 |
|---|---|
| `model.pt` | 带算法和动作约定元数据的 actor checkpoint |
| `training.json` | 回合、转移、更新、首个 PPO batch 和验证 IAE |
| `history.csv` | 每回合 reward 的直接累加（不折扣）和更新 loss |
| `learning.svg` | 回报与 loss |
| `response.svg` | 冻结 actor 与 SIMC 的闭环响应 |
| `PPO.csv` | 温度、输出、Kp、Ki 的逐物理步记录 |

`explain` 只读取已经训练的 `training.json`，不会重新训练。`all` 会在生成 PPO 训练产物后自动写这份解释。

## 观测、动作和增益单位

PPO 读与 Q-Learning 相同的五维观测：

$$
o=(e,\dot e,u,s_p,s_i)\in\mathbb R^5.
$$

原始单位依次是 ℃、℃/min、无量纲、无量纲、无量纲。连续策略先除以 `[4,0.2,1,2,2]`，所以 actor 的单条输入形状是 `(5,)`，训练 batch 是 `(B,5)`；`first_update.states` 保存的就是归一化后的长度 5 向量。

actor 是 `5→64→64→4`：最后 4 个数分成两项 mean 和两项 log standard deviation，形状均为 `(B,2)`。对每个决策点采样一个未压缩动作

$$
z=\mu+\exp(\log\sigma)\epsilon,\qquad \epsilon\sim\mathcal N(0,I_2),
$$

再做 `tanh` 得到 `action∈[-1,1]^2`。核心环境把它变为

$$
(K_p,K_i)=(K_{p,0}2^{a_0},K_{i,0}2^{a_1}),
$$

所以实际增益分别落在 SIMC 锚点的 0.5 到 2 倍。Kp 单位是 ℃^-1，Ki 是 (℃·min)^-1；动作本身没有温度单位。

actor 共有

$$
(5\times64+64)+(64\times64+64)+(64\times4+4)=4804
$$

个参数。value network 也是 `5→64→64→1`，只用于训练，不参与冻结后的在线动作。

实际的 tanh 变换和 log probability 修正在共享实现中：

<<< @/../hvac_pid/continuous.py#gaussian_sample

## 一回合怎样变成 PPO batch

训练开始时先为 48 个训练对象各做一次 240 分钟阶跃辨识，得到并缓存 48 个 SIMC 锚点；500 个回合只随机抽取这些已有锚点，不会每个回合重新做阶跃。之后 `TuningEnv` 每次推进 20 个物理步：

1. actor 读取一个 `(5,)` 观测并采样 `(2,)` 的 pre-tanh action；
2. 环境映射增益，PI 跑 20 个 0.1 分钟区间；
3. reward 是这两分钟 IAE 的负值，单位 ℃·min；
4. 保存 state、pre-tanh action、old log probability、value、reward、next value 和 terminal flag。

120 分钟的一回合有 60 个转移，因此一个回合的数组形状分别是 `states(60,5)`、`actions(60,2)`、`old_log_probs(60,)`、`rewards(60,)` 和 `values(60,)`。最后一条真实终止转移不 bootstrap；固定 horizon 的 truncated 记录会阻止 GAE 继续跨界，但仍可以使用 boundary 的 next value。

## GAE 和 clipped objective

先算一步 TD 残差

$$
\delta_t=r_t+\gamma(1-terminated_t)V(s_{t+1})-V(s_t),
\qquad \gamma=0.99.
$$

从后往前累加广义优势估计：

$$
A_t=\delta_t+\gamma\lambda(1-terminated_t)(1-truncated_t)A_{t+1},
\qquad \lambda=0.95.
$$

<<< @/../hvac_pid/ppo.py#ppo_gae

回报 target 是 $R_t=A_t+V(s_t)$。每次 PPO 更新先把整条轨迹优势标准化，再随机打乱，默认 4 个 epoch、batch size 64。因为单回合只有 60 条样本，每个 epoch 实际是一个 batch。

旧策略与新策略的概率比为

$$
r_t(\theta)=\exp\left(\log\pi_\theta(a_t|s_t)-\log\pi_{old}(a_t|s_t)\right).
$$

策略损失取 clipped surrogate：

$$
L_\pi=-\frac1B\sum_t\min\left(r_tA_t,\operatorname{clip}(r_t,0.8,1.2)A_t\right).
$$

总 loss 是

$$
L=L_\pi+0.5\,\mathrm{MSE}(V,R)-0.01\,\mathrm{mean}(H).
$$

<<< @/../hvac_pid/ppo.py#ppo_loss

这里的 `gamma=.99` 是训练 target 的折扣；最后对整条响应计算的 IAE 仍是不折扣的绝对误差积分。

## 实际首个更新的数字

下面的数来自附带参考运行 `experiments/reference/10-ppo/training.json`（seed 0、500 回合）的首个实际 update；它们只核对一次更新，不代表 500 回合最终性能：

| 量 | 值 |
|---|---:|
| `states[0]` | `[0.0173864,-0.0264151,0.6576164,0.6941699,0.9268699]` |
| `pre_tanh_action[0]` | `[0.8445497,0.4992776]` |
| `old_log_prob` | `-1.3667552` |
| `new_log_prob` | `-1.3667552` |
| `ratio` | `1.0` |
| `value` | `-0.0648556` |
| `raw_advantage` | `-0.5486007` |
| `normalized_advantage` | `0.6646457` |
| `return` | `-0.6134563` |
| `value_loss` | `160.6029358` |
| `entropy` | `1.3137754` |
| `loss` | `80.2883301` |

因为新旧 log probability 在第一批仍相同，$r=\exp(0)=1$。这个样本的未取平均策略项是 $-1\times0.6646457=-0.6646457$；表里的 `policy_loss≈3.18e-8` 是整个 minibatch 对 60 个标准化优势取平均后的结果，接近零来自批量平均，不是该样本的损失。总 loss 主要来自 `0.5×160.6029358`，再减去 `0.01×1.3137754`。`ppo_loss` 中的 clip 是对 surrogate 项的比较，不是把网络输出强行裁成 `[0.8,1.2]`。表里的 `loss` 是首个 minibatch 的标量，整回合 `history.loss` 是本回合多个 minibatch/epoch 的平均，不要把两者混为一个样本损失。

<!--@include: ../generated/first-update-ppo.md-->

<!--@include: ../generated/10-ppo.md-->

真正把轨迹数组转成 torch batch、做 4 个 epoch 更新的是下面这段；它也保存了表中首个 minibatch 的字段：

<<< @/../hvac_pid/ppo.py#ppo_update

## 训练曲线和失败方式

![PPO 训练回报与损失](/results/10-ppo/learning.svg)

PPO 每个回合更新 actor 和 value network，默认 500 回合共有 30,000 个两分钟转移、2,000 个 PPO minibatch 更新（每回合 4 个 epoch），另有训练对象的 commissioning。`history.return` 是这一回合 reward 的直接累加，也就是不折扣的负 IAE；gamma 只参与 GAE 和 value target。回报包含高斯探索动作；冻结评估时 actor 使用 `tanh(mean)`，所以训练曲线不能直接当作最终响应。

一个实际失败点是 tanh 饱和：pre-tanh 数值很大时，执行动作接近 ±1，log probability 必须保留 tanh Jacobian 修正；把已饱和 action 直接代回普通高斯密度会得到不同的概率。另一个失败点是 advantage 标准化：不标准化就把不同回合的 IAE 尺度直接送进 policy update，学习率的有效大小会明显改变。

若 value loss 下降而 IAE 不降，先检查动作是否长期靠近 ±1，再看固定策略的 `PPO.csv`。首个记录中的 ratio 是 1；clipped objective 限制一次更新的策略项，不保证对象延迟下的温度响应一定改善，也不代表 ratio 本身会被硬截断。

![PPO 冻结 actor 的温度和在线增益](/results/10-ppo/response.svg)

## 练习：只改回合数

```bash
python -m hvac_pid ppo --episodes 1000 --seed 0 --output outputs/ppo-1000
python -m hvac_pid explain --method ppo --output outputs/ppo-1000
```

验证 `training.json` 的 `transitions=60000`、`updates=4000`，而 `first_update` 仍是第一批的实际值；再比较冻结 `PPO.csv` 的 IAE、过冷和 movement。保持首个 update 的输入和预算协议不变，检查最终指标是否变化；判断 clipping 是否生效应阅读 `ppo_loss` 源码与实际 loss，而不是查找没有导出的后续 ratio 字段。若只看最终 loss，无法判断更多回合是否真的改善控制。

## 在线位置和 CPU 成本

算法依据是 [PPO 原论文](https://arxiv.org/abs/1707.06347) 的 clipped surrogate。这里采用一条完整温控轨迹作为 rollout、两层各 64 单元、4 个 epoch 和 tanh 连续增益动作；这些是本项目的教学配置，不是在复现论文中的机器人或 Atari 成绩。

PPO 的 rollout、GAE 和优化器都在整定阶段运行。正式闭环只加载 actor，每 2 分钟做一次 `(5,)→(2,)` 前向和 `tanh`，再由 PI 每 0.1 分钟执行；value network 和 optimizer 不在线使用。参考 CPU 时间、checkpoint 字节数和中位/P95 推理时间由 `benchmark` 的实际输出给出，不把 smoke 运行的秒数当成 500 回合基准，也不把 checkpoint 文件大小当成 Python/PyTorch 运行时内存。

本章下载：[冻结模型](/results/10-ppo/model.pt)、[完整训练记录](/results/10-ppo/training.json)、[响应 CSV](/results/10-ppo/PPO.csv)。
