# 11 CrossQ：把当前和下一状态放进同一次 critic 前向

CrossQ 保留 SAC 的连续高斯 actor、熵项、双 critic 和 replay，但取消 target critic。每个 critic 把 current `(state,action)` 和 next `(next_state,next_action)` 拼成一个 batch，用 Batch Renormalization（BRN）共享这两种分布的统计量；然后用这次前向得到的 next Q 构造 target。实现还让 actor/alpha 每两次 critic 更新一次。

## 运行和查看首个更新

```bash
python -m hvac_pid crossq --episodes 500 --seed 0 --output outputs/tutorial
python -m hvac_pid explain --method crossq --output outputs/tutorial
```

产物在 `outputs/tutorial/13-crossq/`：`model.pt`、`training.json`、`history.csv`、`learning.svg`、`response.svg`、`CrossQ.csv` 和 `metrics.json`。在线 checkpoint 只保存 actor 和动作元数据；BRN critic 的 running statistics 属于训练状态，不能从 actor 文件推断。

## 五维观测和连续动作

输入是归一化的 `(5,)` 观测，原始单位为 `[℃,℃/min,1,1,1]`；每个动作是 `(2,)` 的 `[-1,1]` 向量，通过

$$
(K_p,K_i)=(K_{p,0}2^{a_0},K_{i,0}2^{a_1})
$$

变成相对 SIMC 的增益，Kp 单位是 ℃^-1，Ki 单位是 (℃·min)^-1。Gaussian actor 结构是 `5→64→64→4`，输出 mean/log_std 各 `(B,2)`。

每个 CrossQ critic 的输入是 `(B,7)`，第一层后接 64 维 BRN、第二层后接 64 维 BRN，最后输出 `(B,)` 的 Q。两种状态动作不会分别 forward：对 batch size B，`forward_current_next` 把观测和动作拼成 `(2B,7)`，一次通过 critic，再切回 current `(B,)` 和 next `(B,)`。每个 critic 的可学习参数是 4993（BRN 的 running_mean/var 是 buffer，不计参数），两个 critic 共 9986；在线只运行 4804 参数的 actor。

BRN 对 `(2B,64)` 训练 batch 的均值、方差和 running buffer 更新实现如下：

<<< @/../hvac_pid/continuous.py#batch_renorm

## 没有 target critic 的 target

先在 `no_grad` 下从 actor 的 next state 采样 `(next_action,next_log_prob)`。两个 CrossQ critic 分别对 `(current,current_next)` 做联合前向，得到 `q1_current,q1_next,q2_current,q2_next`。然后：

$$
\tilde Q(s',a')=\min(q_{1,next},q_{2,next})-\alpha\log\pi(a'|s'),
$$

$$
y=r+\gamma(1-terminated)\tilde Q(s',a'),
\qquad \gamma=0.99.
$$

current 两个 Q 的 MSE 与同一个 target 相加得到 critic loss。没有 target 网络复制，也没有 Polyak 参数；稳定性来自联合 batch 的 BRN 和双 Q，而不是一个慢速 target critic。

## 用一次真实首更新复算

以下数来自附带完整参考运行 `experiments/reference/13-crossq/training.json` 的 seed 0、500 回合记录；`explain` 会从同一 `training.json` 读取这条首个更新：

| 量 | 值 |
|---|---:|
| `states[0]` | `[0.1183130,-0.3641794,0.4704475,0.4883600,0.3220652]` |
| `actions[0]` | `[0.9432625,0.7953969]` |
| `reward` | `-0.9613781` |
| `alpha` | `1.0` |
| `next_action[0]` | `[0.9314243,0.9462093]` |
| `next_log_prob` | `-0.0405073` |
| `next_q1`, `next_q2` | `-0.1451200`, `0.1911631` |
| `target` | `-1.0649446` |
| `q1`, `q2` | `-0.1491923`, `0.1162545` |
| `critic_loss` | `12.7075739` |
| `actor_updated` | `false` |

取较小的 next Q 后，

$$
\tilde Q=-0.1451200-1\times(-0.0405073)=-0.1046127,
$$

$$
y=-0.9613781+0.99\times(-0.1046127)=-1.0649446.
$$

第一次 update 的编号是 1，`policy_delay=2`，所以首批只更新两个 critic，actor 和 alpha 的 loss 为空。第二次 update 才进入 actor/alpha 分支，并临时把 critic 切到 eval 模式，避免策略更新仅因为 forward 就改变 BRN running statistics。

一次联合 current/next critic 更新的真实代码在这里：

<<< @/../hvac_pid/crossq.py#crossq_update

<!--@include: ../generated/first-update-crossq.md-->

## BRN 为什么要联合 current/next

BRN 在训练模式下从 `(2B,64)` 联合 batch 计算均值和方差，同时维护 running statistics；在评估模式下使用 running statistics。若 current 和 next 样本分别通过 BRN，critic 会看到两套统计，target 的分布关系会改变；CrossQ 的实现明确把它们拼接后一起过每一层。

这也带来一个需要观察的边界：batch size 只有 64，且 replay 混合了训练对象和时刻，BRN 的 batch 统计会随 replay 分布变化。`CrossQPolicy.choose` 加载 actor 后不调用 critic，因此在线不需要 BRN；训练和评估的模式切换仍是实现的一部分。

## 训练曲线和失败方式

![CrossQ 训练回报与 loss](/results/13-crossq/learning.svg)

500 回合的交互量与其他连续 replay 方法相同：30,000 个两分钟转移、600,000 个物理步，warmup 后约 29,745 个 critic update，actor/alpha 约每两个 update 执行一次。`history.return` 是每回合 reward 的直接累加，即不折扣的负 IAE；gamma 只出现在 critic target 和 GAE/TD 类训练目标中。

![CrossQ 冻结 actor 的温度和在线增益](/results/13-crossq/response.svg)

<!--@include: ../generated/13-crossq.md-->

若把 CrossQ 当成“无 target 的 SAC”而忘记联合 BRN，得到的 critic 更新已经不是当前实现。若在 actor 更新阶段让 BRN 继续 train，采样 actor 动作会修改 running statistics，下一次 critic target 变化不仅来自参数更新。若误以为没有 target critic 就不需要 `terminated` mask，最后一个转移仍会错误地 bootstrap。

CrossQ 仍只看到五维部分观测，连续动作仍由输出边界限制，movement 可能增加。目标是讲清这种结构在小对象上的行为，不从一次损失下降推断对真实设备的稳定性。

## 练习：只改变回合预算

```bash
python -m hvac_pid crossq --episodes 1000 --seed 0 --output outputs/crossq-1000
python -m hvac_pid explain --method crossq --output outputs/crossq-1000
```

检查 transitions/updates、actor 延迟次数和 `first_update` 的 `actor_updated=false`。比较最终冻结 actor 的 IAE、过冷和 movement；若首个 update 的 target 变化，先检查 seed、warmup 或联合 batch 顺序，而不是把它解释成训练更久的效果。

## 在线位置和 CPU 成本

### 对照论文时先看配置差异

[CrossQ 的 ICLR 2024 论文及附录配置](https://arxiv.org/html/1902.05605v4#A1.SS2) 使用更宽的 Critic 和更大的回放与 batch。本章保留联合前向、BRN、无目标 Critic 和 UTD=1，但把计算预算缩小：

| 设置 | 论文主实验 | 本章 |
|---|---:|---:|
| Critic 隐层宽度 | 2048 | 64 |
| batch | 256 | 64 |
| 回放容量 | 1,000,000 | 10,000 |
| 学习率 | 0.001 | 0.0003 |
| actor 更新间隔 | 3 | 2 |
| Adam β₁ | 0.5 | 0.9 |
| BRN 专用 warm-up | 100,000 步 | 无，首次 Critic 更新即启用修正 |

本章只在 Critic 中使用 BRN，actor 为普通 MLP；BRN 的 `rmax=3`、`dmax=5` 固定，统计量更新率为 0.01。它与回放先积累 256 条经验的 warmup 是两回事。这些差异意味着我们在研究 CrossQ 的关键机制如何作用于 PI 调参，而不是复制论文的完整训练配置。

名义工况 seed 0 的 IAE 为 `59.9434`，但出现 `0.1560℃` 过冷；五种子的 IAE 平均为 `67.1125`，标准差 `20.6896`，其中 seed 1 为 `104.0129`。归一化没有消除训练随机性，在本项目上也没有稳定地胜过所有基线。

CrossQ 的 replay、两个 BRN critic、alpha 和 actor optimizer 都在离线训练。在线每 2 分钟只执行 `5→64→64→4` actor 的确定性 mean，再由 PI 高频执行；没有 target critic 前向和没有外部服务调用。`benchmark` 测量的是 actor batch=1 的中位数/P95、参数量和 checkpoint 文件大小；文件大小不能代替运行时内存估计，也不承诺某种硬件可以部署。

本章下载：[冻结模型](/results/13-crossq/model.pt)、[完整训练记录](/results/13-crossq/training.json)、[响应 CSV](/results/13-crossq/CrossQ.csv)。
