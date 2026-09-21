# 11 CrossQ：把当前和下一状态放进同一次 critic 前向

CrossQ 沿用 SAC 的随机连续 actor、熵项和双 Q 估计，但故意取消 target critic。它把 replay batch 中的 current 行和 next 行拼成同一个 2B 行输入，在每个 critic 的 Batch Renormalization（BRN）层共享一次 batch 统计。当前 Q 负责反向传播，next Q 只在 detached target 中使用。

这里的 Actor 是从五维观测产生二维动作分布的策略网络，Critic 是给定观测和动作后估计未来折扣回报的网络；回报是负 IAE，Q 越大表示预期累计误差越小。CrossQ 的工程重点不是把这些定义隐藏在“无 target 的 SAC”一句话中，而是说明 joint forward 如何改变 critic 看到的分布、哪些统计会持久化、哪些状态在线根本不存在。

## 观测、动作和网络计算对象

原始观测为

$$
o=(e,\dot e,u,s_p,s_i),
$$

送入 actor 前除以 [4, 0.2, 1, 2, 2]，形成 (B,5) 的 float32 tensor。actor 的 MLP 是 5→64→64→4，输出 mean 和 log_std 两个 (B,2) 张量；训练用 reparameterized Gaussian 和 tanh，确定性在线动作取 tanh(mean)。

动作 a∈[-1,1]^2 在 TuningEnv.step_continuous 中按

$$
K_p=K_{p,0}2^{a_0},\qquad K_i=K_{i,0}2^{a_1}
$$

映射到相对 SIMC 锚点的实际增益。actor 有 4804 个可学习参数。它没有读取 Plant 延迟队列，所以 CrossQ 仍然只处理部分观测。

每个 CrossQ critic 的前向是：

$$
(o,a)\in\mathbb R^7
\xrightarrow{\text{Linear }7\to64}
\text{BRN}_{64}
\xrightarrow{\text{ReLU}}
\text{Linear }64\to64
\xrightarrow{\text{BRN}_{64}}
\text{ReLU}
\xrightarrow{\text{Linear }64\to1}Q.
$$

一个 critic 有 4993 个可学习参数；BRN 的 running_mean、running_var 和 num_batches_tracked 是 buffer，不计入可学习参数。两个 critic 共 9986 个参数。CrossQCritic.forward_current_next 负责把 current 和 next 拼接后只做一次 forward，再按原 batch 大小切回两部分。

## BRN 的实际公式和 detached 统计

对任一 BRN 层，联合输入记为 $x\in\mathbb R^{2B\times64}$。训练模式下代码先计算联合 batch 的均值、无偏差修正关闭的方差和标准差：

$$
m=\operatorname{mean}_{0}(x),\qquad
v=\operatorname{mean}_{0}\bigl((x-m)^2\bigr),\qquad
s=\sqrt{v+\varepsilon}.
$$

它从 running buffer 复制 $m_r,v_r$，并计算

$$
s_r=\sqrt{v_r+\varepsilon},
$$

$$
r=\operatorname{clip}\left(\frac{\operatorname{stopgrad}(s)}{s_r},
\frac1{r_{\max}},r_{\max}\right),
$$

$$
d=\operatorname{clip}\left(
\frac{\operatorname{stopgrad}(m)-m_r}{s_r},
-d_{\max},d_{\max}\right).
$$

本仓库固定 rmax=3、dmax=5、momentum=0.01。归一化激活为

$$
\hat x=\left(\frac{x-m}{s}\right)r+d,
\qquad y=\gamma_{\mathrm{BRN}}\hat x+\beta_{\mathrm{BRN}}.
$$

这里的 r、d 和 running buffer 都不进入反向图；输入 x 和可学习的 scale、bias 仍然可以接收梯度。一次训练 forward 结束后，代码在 no_grad 中用 momentum=0.01 更新 running_mean、running_var，并增加计数。

eval 模式不再计算当前 batch 的 m、v、r、d，而是直接使用 running statistics。因而 BRN 的 train/eval 切换是算法行为的一部分。训练时 actor/alpha 分支若让 critic 继续处于 train 模式，即使没有 critic optimizer step，单纯评估动作也会改写这些 buffer。

<<< @/../hvac_pid/continuous.py#batch_renorm

## 为什么必须联合 current 和 next

对 batch size B=64，当前行和下一行拼接后输入形状为 (128,7)。第一层线性变换后形成 (128,64)，BRN 在这 128 行上只计算一套 m、v、r、d；第二层再次对同一组 current/next 配对共享统计。forward_current_next 最后将前 64 行返回为 current Q，后 64 行返回为 next Q。

如果 current 和 next 分别 forward，每一组都会拥有自己的均值与方差。critic 看到的 Q 标度就不再只是参数变化，还会因为“这是 current 还是 next 的另一套归一化坐标”发生变化。CrossQ 的 target 又直接使用当前 critic 的 next 输出，因此这两个分布必须在同一次 forward 中保持同一统计参考。joint batch 并不让 current 与 next 变成同一个样本；它们仍占据不同的行，只共享该 batch 的归一化坐标。

BRN 的 running statistics 也解释了为什么 next Q 不应在策略更新分支再次 train forward。训练 batch 的联合统计会被 replay 分布、对象切换和时间位置影响；actor 更新阶段只是求策略梯度，不能因为重新评估动作而额外推进 running buffer。

## 没有 target critic 的 target 路径

update_crossq 先在 no_grad 中从当前 actor 的 next_states 采样 next_actions 和 next_log_prob。随后 q1、q2 各自调用 forward_current_next，得到

$$
(q_{1,c},q_{1,n}),\qquad(q_{2,c},q_{2,n}).
$$

这一步处于 critic 的 train 模式，所以 joint BRN 的 batch 统计和 running buffer 会按设计更新。然后代码进入 no_grad，读取当前 log_alpha 的指数作为 alpha，计算

$$
\tilde Q_n=
\min(q_{1,n},q_{2,n})-\alpha\log\pi(a'|s'),
$$

$$
y=r+\gamma(1-\mathit{terminated})\tilde Q_n,
\qquad \gamma=0.99.
$$

no_grad 包住 target 的算术，因此 y 对 q1_next、q2_next、alpha、next_log_prob 和 actor 参数都停止梯度。q1_current、q2_current 仍保留第一次 joint forward 的计算图，两个 MSE 只从 current 行反向传播：

$$
L_Q=\operatorname{MSE}(q_{1,c},y)
+\operatorname{MSE}(q_{2,c},y).
$$

没有 target actor、target critic 或 Polyak 参数。CrossQ 不是把 target 网络藏起来，而是让同一个 critic 的 next 行产生 target，再用 BRN 的联合分布控制 current/next 的尺度关系。

<<< @/../hvac_pid/crossq.py#crossq_update

## 首个真实 batch 的数值路径

下面的数来自 experiments/reference/13-crossq/training.json 的 seed 0、500 回合参考运行；首个 update 由同一训练记录写出。

| 量 | 值 |
|---|---:|
| states[0] | [0.1183130,-0.3641794,0.4704475,0.4883600,0.3220652] |
| actions[0] | [0.9432625,0.7953969] |
| reward | -0.9613781 |
| alpha | 1.0 |
| next_action[0] | [0.9314243,0.9462093] |
| next_log_prob | -0.0405073 |
| next_q1, next_q2 | -0.1451200, 0.1911631 |
| target | -1.0649446 |
| q1, q2 | -0.1491923, 0.1162545 |
| critic_loss | 12.7075739 |
| actor_updated | false |

取较小的 next Q 并加入熵项：

$$
\tilde Q_n=-0.1451200-1\times(-0.0405073)
=-0.1046127,
$$

所以

$$
y=-0.9613781+0.99\times(-0.1046127)
=-1.0649446.
$$

第一个 update 的编号是 1，policy_delay=2；本次只更新两个 critic 和其 BRN buffers，actor 与 alpha loss 为空。actor_updated=false 不是“没有 actor”，而是延迟分支的可观测记录。

## 延迟策略分支与 buffer 生命周期

当 update index 为 2 的倍数时，代码先记录两个 critic 原来的 train/eval 状态，再调用 q1.eval()、q2.eval()。同时暂时关闭两个 critic 参数的 requires_grad。这样策略动作经过 Q 的输入仍能求导，但 actor batch 不会改变 BRN 的 running mean、running var 或计数。

策略损失沿用 SAC 形式：

$$
L_\pi=\operatorname{mean}\left[
\alpha\log\pi_\theta(a|s)
-\min(q_1(s,a),q_2(s,a))
\right].
$$

alpha 在 actor loss 中 detach，log probability 在 alpha loss 中 detach：

$$
L_\alpha=-\operatorname{mean}\left[
\log\alpha\left(\operatorname{stopgrad}(\log\pi_\theta(a|s))
-2\right)\right].
$$

actor 和 alpha optimizer step 完成后，代码恢复 critic 的 requires_grad 和原来的 train/eval 状态。CrossQ 每两个 critic update 执行一次 actor/alpha 分支，但不会在任何时刻创建或软更新 target critic。

## 训练量、论文差异和读法

500 回合产生 30,000 条两分钟 transition、600,000 个物理步；warmup 256 后约 29,745 个 UTD=1 critic update，actor/alpha 约每两个 update 执行一次。history.return 是探索策略下不折扣的负 IAE 累加，gamma 只出现在 critic target。

CrossQ 论文及附录的设置与此仓库不同。保留原链接和关键差异：

| 设置 | 论文主实验 | 本章 |
|---|---:|---:|
| Critic 隐层宽度 | 2048 | 64 |
| batch | 256 | 64 |
| 回放容量 | 1,000,000 | 10,000 |
| 学习率 | 0.001 | 0.0003 |
| actor 更新间隔 | 3 | 2 |
| Adam β₁ | 0.5 | 0.9 |
| BRN 专用 warm-up | 100,000 步 | 无，首次 Critic 更新即启用修正 |

本章只在 critic 使用 BRN，actor 是普通 MLP；回放先积累 256 条经验的 warmup 和 BRN 的专用 warm-up 不是同一个机制。论文配置链接为 [CrossQ 的 ICLR 2024 论文及附录配置](https://arxiv.org/html/1902.05605v4#A1.SS2)。本实现保留联合前向、BRN、无 target critic 和 UTD=1，但没有复制论文的宽网络或数据预算。

名义工况 seed 0 的 IAE 为 59.9434，并出现 0.1560℃ 过冷；五种子 IAE 平均为 67.1125，标准差为 20.6896，其中 seed 1 为 104.0129。数字用于描述本仓库配置下的波动，不能推出 CrossQ 在所有对象上胜出。

![CrossQ 训练回报与 loss](/results/13-crossq/learning.svg)

![CrossQ 冻结 actor 的温度和在线增益](/results/13-crossq/response.svg)

<!--@include: ../generated/13-crossq.md-->

## 在线边界、持久状态和 CPU 口径

CrossQPolicy.choose 每次把一个长度 5 的观测转成 batch=1，执行 actor.eval()、mean/log_std 前向和 torch.no_grad()，返回 tanh(mean) 的两个 float32 动作。在线 checkpoint 只包含 Gaussian actor 的 4804 个参数与动作元数据；两个 BRN critic、running statistics、alpha 和 replay 都不加载。

在线每两分钟调用一次 actor，环境用动作换算 Kp、Ki；随后普通 PI 每 0.1 分钟运行一次。控制侧跨调用保存 PI 积分输出、上一条误差/斜率和上一条指令。Plant 当前温度、输入延迟队列和物理步索引属于本地仿真器，真实控制器不要求保存或复制 Plant 模型状态，而是由传感器、执行器和联锁系统提供相应的过程信息。策略不在线执行 BRN。

参考 benchmark 使用 CPU 单线程、batch=1、100 次 warm-up、无梯度、无探索，并计入观测转换。CrossQ actor 的 1000 次样本中位数为 153.1 μs，P95 为 283.965 μs，4804 个参数，checkpoint 文件为 22,397 bytes。文件大小是磁盘 artifact，不能当成 Python/PyTorch RSS，也不含解释器、线程库、驱动和安全联锁；它不构成特定硬件可部署的保证。

## 失败模式和适用边界

若把 CrossQ 简化为“删掉 target 的 SAC”，却把 current 和 next 分别过 BRN，critic 的输入分布和 target 分布都已经改变。若 actor/alpha 更新期间忘记切到 eval，单次策略 forward 会改写 running statistics，下一次 target 的变化就不再只由参数更新解释。

若在 no_grad 外构造 target，next Q、alpha 或 log probability 会把梯度路径接回 critic、actor 或温度；若省略 terminated mask，最后一条真实终止 transition 仍会错误 bootstrap。若把 truncated 与 terminated 合并，也会失去更新器保留的时间限制语义。

joint BRN 和双 Q 不能解决五维观测没有延迟队列的问题；连续动作也不自动减少 movement。首个 target 或一条下降的 loss 只能用于核对实现路径，不能作为真实设备稳定性或普遍性能结论。

## 结果复现入口

以下命令在本地生成训练记录，再从同一记录导出首个 update 解释：

~~~bash
python -m hvac_pid crossq --episodes 500 --seed 0 --output outputs/tutorial
python -m hvac_pid explain --method crossq --output outputs/tutorial
~~~

输出目录为 outputs/tutorial/13-crossq/。本章下载：[冻结模型](/results/13-crossq/model.pt)、[完整训练记录](/results/13-crossq/training.json)、[响应 CSV](/results/13-crossq/CrossQ.csv)。
