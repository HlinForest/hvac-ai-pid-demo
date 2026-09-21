# 10 SAC：让连续增益在价值和熵之间取平衡

SAC 使用与 TD3 相同的五维观测、二维连续动作和 twin critic，但 actor 是高斯策略，target 中显式加入熵项。训练阶段保留动作随机性，`alpha` 会根据目标熵自动调整；冻结评估时只使用 actor 的确定性均值。

## 运行和解释

```bash
python -m hvac_pid sac --episodes 500 --seed 0 --output outputs/tutorial
python -m hvac_pid explain --method sac --output outputs/tutorial
```

结果在 `outputs/tutorial/12-sac/`，包括 `model.pt`、`training.json`、`history.csv`、`learning.svg`、`response.svg`、`SAC.csv` 和 `metrics.json`。`explain/sac.json` 的首个 update 读取实际 replay batch；它不会重新采样或伪造一组“容易算”的 Q 值。

## 输入、动作和策略输出

输入仍是归一化的长度 5 向量

$$
o=(e/4,\dot e/0.2,u,s_p/2,s_i/2)\in\mathbb R^5.
$$

actor 是 `5→64→64→4`，最后拆成 `(mean, log_std)` 两个 `(B,2)` 张量。训练采样

$$
z=\mu+\sigma\epsilon,
\qquad a=\tanh(z)\in[-1,1]^2,
$$

再按 $2^a$ 映射为 SIMC 锚点的 Kp/Ki 倍数。动作单位无量纲；实际 Kp 是 ℃^-1，Ki 是 (℃·min)^-1。`squashed_gaussian` 用 pre-tanh 值计算稳定的 tanh Jacobian 修正，避免把接近 ±1 的动作当成普通高斯概率。

两个 critic 各接收 `(state, action)` 形状 `(B,7)`，输出 `(B,)` 的 Q 值；actor 只有在在线推理时被加载。replay 容量 10,000、warmup 256、batch 64、学习率 `3e-4`、折扣 `0.99`、target soft update `0.005`。

## SAC target 的组成

对下一状态从当前高斯 actor 采样 `next_action` 和 `next_log_prob`，两个 target critic 取较小值，并减去熵代价：

$$
\tilde Q(s',a')=\min(Q_1^-,Q_2^-)-\alpha\log\pi(a'|s').
$$

$$
y=r+\gamma(1-terminated)\tilde Q(s',a').
$$

<<< @/../hvac_pid/sac.py#sac_target

critic 用两个 MSE 拟合 target；actor 损失是

$$
L_\pi=\operatorname{mean}\left[\alpha\log\pi(a|s)-\min(Q_1(s,a),Q_2(s,a))\right].
$$

在 log-alpha 坐标中，自动温度更新为

$$
L_\alpha=-\log\alpha\;\operatorname{mean}(\log\pi(a|s)+H_{target}),
\qquad H_{target}=-2.
$$

alpha 初始为 1。它不是奖励权重，也不是温度对象的摄氏值；它调节价值和策略熵的相对尺度。

## 实际首个 batch 的一项复算

下面来自附带参考运行 `experiments/reference/12-sac/training.json`（seed 0、500 回合）的实际首个 update：

| 量 | 值 |
|---|---:|
| `states[0]` | `[0.1183130,-0.3641794,0.4704475,0.4883600,0.3220652]` |
| `actions[0]` | `[0.9432625,0.7953969]` |
| `reward` | `-0.9613781` |
| `alpha` | `1.0` |
| `next_action[0]` | `[0.3734372,0.8109848]` |
| `next_log_prob` | `-1.2206299` |
| `next_q1`, `next_q2` | `-0.1512899`, `0.0712416` |
| `target` | `0.0972686` |
| `q1`, `q2` | `-0.1726386`, `0.0839126` |
| `critic_loss` | `12.3715734` |
| `actor_loss` | `-1.1636822` |
| `alpha_loss` | `-0.0` |

因为较小的下一 Q 是 `-0.1512899`，熵修正为

$$
\tilde Q=-0.1512899-1\times(-1.2206299)=1.0693400,
$$

所以

$$
y=-0.9613781+0.99\times1.0693400=0.0972686.
$$

若忽略 `-alpha*log_prob`，会得到约 `-1.1117` 的完全不同 target。三个 loss 都来自同一个 batch，但 `critic_loss` 是两个 critic 的 MSE 之和，`actor_loss` 和 `alpha_loss` 是随后独立反向传播的标量；不能把它们当成一个单独样本的 IAE。

<!--@include: ../generated/first-update-sac.md-->

## 训练动作和冻结动作不同

500 回合训练共有 30,000 个两分钟转移；warmup 后每个转移都可触发一次 batch update，参考 `training.json` 的 updates 和 seconds 是实际运行结果。训练时 actor 采样高斯噪声，target 同样采样；评估加载 `SACPolicy` 后调用 `sample(..., deterministic=True)`，只取 tanh(mean)。因此 `history.csv` 的 return、critic loss 和冻结 `SAC.csv` 的温控 IAE 应分开阅读。

![SAC 训练回报与 loss](/results/12-sac/learning.svg)

![SAC 冻结 actor 的温度和在线增益](/results/12-sac/response.svg)

<!--@include: ../generated/12-sac.md-->

## 失败方式：熵项或边界的符号错了

若把 target 写成 `min_q + alpha*log_prob`，由于 log probability 通常为负，会把熵鼓励反向；若在 alpha loss 中忘记 detach log probability，温度更新会意外改变 actor 反向路径。若 eval 仍采样而不是 deterministic，单次曲线会混入探索噪声，无法和固定方法稳定比较。

critic、actor 和 alpha 的真实更新顺序在这里：

<<< @/../hvac_pid/sac.py#sac_update

SAC 也没有解决部分观测问题。延迟队列的隐藏状态不在五维输入中，两个相同可见状态可能需要不同动作。熵只能增加动作覆盖，不能补回未观测信息。连续动作能细调 Kp、Ki，却可能增加 movement；统一比较仍要同时读三项指标。

## 练习：改变 replay 看到的轨迹数

```bash
python -m hvac_pid sac --episodes 1000 --seed 0 --output outputs/sac-1000
python -m hvac_pid explain --method sac --output outputs/sac-1000
```

核对 transitions 是否变为 60,000，updates 是否超过 29,745；`first_update` 应保持首个 batch 的结构和 seed=0 的数值，最终 `alpha`、actor 参数、冻结 IAE 和 movement 才可能变化。若 alpha 远离 1，结合 `history.csv` 的回报和动作分布解释它，而不是把 alpha 当作控制增益。

## 在线位置和 CPU 成本

本实现参考 [SAC Algorithms and Applications](https://arxiv.org/abs/1812.05905) 的自动熵系数版本，使用两个 Q 网络及其目标副本，不另设独立 V 网络。本文的 64×64 小网络、500 回合预算、PI 增益动作和 IAE 奖励是教学设定，不能把本章数值当成论文基准的复现。

训练阶段才使用 replay、两个 target critic、alpha optimizer 和随机策略。上线后只保留一个 `5→64→64→4` actor，确定性输出每 2 分钟更新一次增益，PI 仍每 0.1 分钟运行。`benchmark` 报告 actor 推理的 CPU 中位数/P95、参数量和 checkpoint 字节数；后者不等于带 Python/PyTorch 运行时的内存。SAC 不需要在线模型服务，但需要把 deterministic 评估约定和动作边界一同保存。

本章下载：[冻结模型](/results/12-sac/model.pt)、[完整训练记录](/results/12-sac/training.json)、[响应 CSV](/results/12-sac/SAC.csv)。
