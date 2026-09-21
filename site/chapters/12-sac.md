# 10 SAC：让连续增益在价值和熵之间取平衡

SAC 和上一章使用同一个五维观测、二维动作以及双 critic 结构，但 actor 不输出一个固定动作。它输出对每个动作维度的高斯均值和标准差，再经过 tanh 得到合法的增益尺度。训练目标同时考虑 Q 值和策略熵；冻结评估时，SACPolicy 明确切换到 tanh(mean) 的确定性动作。

本实现是一个离线训练、在线执行 actor 加普通 PI 的架构。replay、target critic、alpha 优化器和随机采样只在训练程序中存在。理解 SAC 的关键，是沿着一个 batch 区分三条梯度路径：critic 拟合 detached target，actor 通过重参数化样本穿过 Q，alpha 只根据 detached 的 log probability 调整。

这里的 Actor 是从观测产生动作分布的策略网络；Critic 是给定观测和动作后估计未来折扣回报的网络。回报是负 IAE，Q 越大表示预期累计误差越小。alpha 是熵系数，不是第四类控制器状态。

## 输入、动作和网络对象

环境给出的原始观测是

$$
o=(e,\dot e,u,s_p,s_i),
$$

连续实现把它除以 [4, 0.2, 1, 2, 2]，得到形状 (5,) 的 float32 观测。策略看到的是温度误差、误差斜率、上一次执行指令和相对 SIMC 增益，而不是对象的延迟队列。

GaussianActor 的 body 是 5→64→64→4。最后四个数沿最后一维拆成两个 (B,2) 张量：均值 μ 和 log standard deviation。log_std 在 parameters_for_distribution 中裁剪到 [-20,2]；这是防止初始更新产生零方差或过大的指数，不是对动作再做一次裁剪。actor 参数量为 4804。

对训练 batch，代码使用重参数化：

$$
\sigma=\exp(\operatorname{clip}(\log\sigma,-20,2)),\qquad
\epsilon\sim\mathcal N(0,I),
$$

$$
z=\mu+\sigma\odot\epsilon,\qquad
a=\tanh(z).
$$

由于 tanh 把动作限制在 [-1,1]，环境再按

$$
K_p=K_{p,0}2^{a_0},\qquad K_i=K_{i,0}2^{a_1}
$$

映射为相对 SIMC 锚点的增益。动作本身没有 Kp/Ki 的物理单位；物理单位留在核心 PI 的两个参数上。

两个 critic 各接收 (state, action) 的 (B,7) 拼接，使用 7→64→64→1，输出 (B,)。critic 参数不与 actor 共享。target_critics 从当前 critics 初始化，并在训练中通过软更新跟随；实现没有独立 V 网络。

## tanh 概率和重参数化为什么要保留 pre-tanh

actor loss 和 entropy target 需要的是动作在 squashed policy 下的 log probability。若只把 a 当作普通高斯样本，tanh 的体积变化会被漏掉。对每个动作维度，代码使用 pre-tanh 的 z 计算稳定的 Jacobian 修正：

$$
\log\pi(a|s)
=\sum_j\left[
\log\mathcal N(z_j;\mu_j,\sigma_j)
-2\left(\log 2-z_j-\operatorname{softplus}(-2z_j)\right)
\right].
$$

这里的第二项等于减去 log(1-tanh(z)^2)，但直接用 z 的 softplus 形式在动作接近 ±1 时更稳定。若从已经饱和的 a 反推 z，数值精度会丢失，entropy 和 alpha 的更新就会被污染。

训练采样返回三项：action、log_prob、mean。action 通过 z=mean+std*noise 与 actor 参数相连，这条路径让 actor loss 可以做低方差的 reparameterization gradient。target 中的采样仍是随机的，但处在 no_grad 区域。

冻结推理调用 sample(..., deterministic=True)，此时 pre_tanh 直接取 mean；仍可计算 log_prob，但 SACPolicy.choose 只返回 tanh(mean)。所以训练曲线中的探索回报和冻结响应 CSV 属于两个不同的执行模式。

## replay 和 target 的实际组成

训练先创建容量 10,000 的 ReplayBuffer。warmup 的前 256 条 transition 从二维动作空间均匀采样；之后从当前 GaussianActor 采样动作。每次环境调用选择一组增益，PI 在 20 个 0.1 分钟步内推进 Plant，返回两分钟区间负 IAE。

tensor_batch 把无放回抽取的 64 条 transition 变成：

| 张量 | 形状 | 进入哪里 |
|---|---:|---|
| states, next_states | (64,5) | actor 和 current/target critic |
| actions | (64,2) | current critic 的数据动作 |
| rewards | (64,) | Bellman target |
| terminated, truncated | (64,) | target bootstrap mask |

sac_target_q 对 next_states 采样 next_action 和 next_log_prob，然后在 torch.no_grad() 中评估两个 target critic。先取较小的 next Q，再减去熵项：

$$
\tilde Q(s',a')
=\min(Q_1^-(s',a'),Q_2^-(s',a'))
-\alpha\log\pi(a'|s').
$$

最终 target 为

$$
y=r+\gamma(1-\mathit{terminated})\tilde Q(s',a'),
\qquad \gamma=0.99.
$$

这里的 alpha 是 exp(log_alpha)，初值为 1。target 使用 alpha.detach()，并且整个 target 函数没有梯度；next actor 的均值、标准差、随机噪声和 target critic 参数都不会被 critic loss 更新。

<<< @/../hvac_pid/sac.py#sac_target

当前两个 critic 对同一个 y 分别做 MSE：

$$
L_Q=\operatorname{MSE}(Q_1(s,a),y)
+\operatorname{MSE}(Q_2(s,a),y).
$$

target 不含当前 critic 的计算图，避免把 critic 更新变成追逐自身输出的闭环。

## 首个真实 batch 的复算

以下数值来自 experiments/reference/12-sac/training.json 的 seed 0、500 回合参考运行。它们由 update_sac 写入训练记录，explain 不会重新生成一组方便计算的样本。

| 量 | 值 |
|---|---:|
| states[0] | [0.1183130,-0.3641794,0.4704475,0.4883600,0.3220652] |
| actions[0] | [0.9432625,0.7953969] |
| reward | -0.9613781 |
| alpha | 1.0 |
| next_action[0] | [0.3734372,0.8109848] |
| next_log_prob | -1.2206299 |
| next_q1, next_q2 | -0.1512899, 0.0712416 |
| target | 0.0972686 |
| q1, q2 | -0.1726386, 0.0839126 |
| critic_loss | 12.3715734 |
| actor_loss | -1.1636822 |
| alpha_loss | -0.0 |

最小 next Q 是 -0.1512899。熵修正先给出

$$
\tilde Q=-0.1512899-1\times(-1.2206299)
=1.0693400,
$$

于是

$$
y=-0.9613781+0.99\times1.0693400
=0.0972686.
$$

如果把符号写成 min_q + alpha*log_prob，这个 log probability 为负时会把熵鼓励反向；如果完全去掉熵项，target 约为 -1.1117，已经不是这次 batch 的 target。

critic_loss 是 64 个样本上两个 MSE 的和；actor_loss 和 alpha_loss 是随后在同一 batch 上重新采样或重用 log_prob 路径得到的批量标量，不能解释成某一条 transition 的 IAE。

一次真实 update 的实现入口如下：

<<< @/../hvac_pid/sac.py#sac_update

<!--@include: ../generated/first-update-sac.md-->

## 三条梯度路径和 alpha 更新

第一段先用 detached target 更新两个 critic。随后代码保存 critics 的 requires_grad 状态并暂时设为 false，重新从当前 states 采样 action 和 log_prob。actor loss 是

$$
L_\pi=\operatorname{mean}\left[
\alpha\log\pi_\theta(a|s)
-\min(Q_1(s,a),Q_2(s,a))
\right].
$$

这里的 action 通过 z 的重参数化与 actor 相连，所以 Q 对 action 的导数可以回到 μ 和 log_std；critic 权重不接收 actor 的梯度。alpha.detach() 保证 actor 更新不把 alpha 当成 actor 的可学习路径。

然后才更新温度参数。目标熵固定为动作维度的负数：

$$
H_{\mathrm{target}}=-2,\qquad
L_\alpha=-\operatorname{mean}\left[
\log\alpha\left(\operatorname{stopgrad}(\log\pi_\theta(a|s))
+H_{\mathrm{target}}\right)\right].
$$

实现写作 -(log_alpha * (log_prob.detach() + target_entropy)).mean()。log_prob 的 detach 很关键：alpha optimizer 只能改变 log_alpha，不能借此再次改变 actor。alpha 不是奖励权重、摄氏温度或 PI 增益，它只调整熵项与 Q 的相对数值尺度。

完成 actor 和 alpha step 后，代码用 tau=0.005 对当前两个 critic 做 Polyak 更新：

$$
\phi^- \leftarrow (1-0.005)\phi^-+0.005\phi.
$$

SAC 的 target actor 没有独立副本，next action 由当前 actor 采样但处于 no_grad；慢速对象只有两个 target critic。每个 batch 都软更新一次，不像 TD3 那样等到延迟 actor step 才更新 target。

因此一次 update 的参数写入顺序是 critic optimizer、actor optimizer、alpha optimizer、target critic soft update。四个 optimizer/state 的生命周期都属于训练进程；冻结 checkpoint 只抽出 actor_state_dict 和验证动作约定的 config。

## 训练记录和在线切分

500 回合对应 30,000 条两分钟 transition、600,000 个物理步；warmup 后每条 transition 都触发一次更新，参考记录约有 29,745 个 update。每个 episode 的 history.return 是探索策略下 reward 的直接累加，不折扣；gamma 只进入 target。

training.json 记录学习率 3e-4、gamma 0.99、batch 64、warmup 256、replay capacity 10,000、target tau 0.005、target entropy -2 和训练结束的 alpha。冻结评估重新 reset 环境，加载 actor，关闭随机采样，再将最终的 tanh(mean) 每两分钟交给环境。

![SAC 训练回报与 loss](/results/12-sac/learning.svg)

![SAC 冻结 actor 的温度和在线增益](/results/12-sac/response.svg)

<!--@include: ../generated/12-sac.md-->

SACPolicy.choose 的持久对象只有 actor 参数和 config。每次调用都执行 batch=1 的 observation conversion、eval 模式和 no_grad；policy 不持有 replay、critic、target critic、alpha optimizer 或噪声状态。控制侧持久化的是 PI 积分输出、上一条误差/斜率和上一条指令。仿真 TuningEnv 另保存 Plant 当前温度、延迟队列和物理步索引；这些只属于仿真器，真实控制器不要求保存或复制 Plant 状态。

因此在线顺序是：两分钟边界调用一次 SAC actor；将两个动作映射为 Kp、Ki；在接下来的二十个 0.1 分钟 tick 里运行普通 PI 和 Plant；下一个边界再提供新的五维观测。SAC 不在线学习，也不需要外部模型服务。

训练端与在线端的边界可以从 checkpoint 字段核对：保存的是 actor_state_dict、observation scale、action bounds、2**action gain transform、interval=2.0 和最终 alpha 等 config；target critic 的权重、BRN 类似的训练 buffer、replay 行和 optimizer state 都不在冻结文件中。alpha 只用于复现训练配置，choose 不读取它来随机采样。

benchmark 的参考协议是 CPU 单线程、batch=1、100 次 warm-up、无梯度、无探索，并包含观测转换。参考机器上的 SAC actor 为 4804 个参数，1000 次调用中位数 361.4 μs，P95 645.62 μs，checkpoint 文件 22,333 bytes。文件字节数只表示磁盘 artifact，不是 Python/PyTorch 运行时 RSS，也不包含解释器、线程库、驱动或安全联锁；它不能承诺任何硬件部署能力。

## 失败模式和论文配置边界

若把 log probability 当作未经过 tanh 修正的普通高斯概率，alpha 会依据错误的熵变化；若从 clipped action 反推 pre-tanh 值，接近边界时的数值误差会直接进入 alpha 和 actor 梯度。训练时不能把 latent z、执行动作 a 和确定性 mean 混为一个变量。

若忘记 target 的 no_grad 或 alpha 的 detach，critic、actor 和温度会产生意外的跨优化器梯度。若 eval 仍使用随机样本，单次响应会含探索噪声，无法和 SIMC 或其他冻结方法按同一轨迹口径比较。

双 critic 与 entropy 并没有补回延迟队列这一未观测状态。更高的动作覆盖也可能增加 movement；IAE、过冷和 movement 仍需一起读取。reference 曲线只说明当前对象、预算和种子的行为，不能作为真实空调对象的稳定性结论。

本实现参考 [SAC Algorithms and Applications](https://arxiv.org/abs/1812.05905) 的自动熵系数版本，使用两个 Q 网络及其目标副本，不另设独立 V 网络。这里的 64×64 网络、500 回合预算、PI 增益动作和 IAE 奖励与论文基准不同，因此不能把本章数值当成论文复现或普遍性能结论。

## 结果复现入口

命令只使用已有 CPU 仿真；第二条读取第一条写出的首个 update 记录：

~~~bash
python -m hvac_pid sac --episodes 500 --seed 0 --output outputs/tutorial
python -m hvac_pid explain --method sac --output outputs/tutorial
~~~

输出目录为 outputs/tutorial/12-sac/，包含 model.pt、training.json、history.csv、learning.svg、response.svg、SAC.csv 和 metrics.json。本章下载：[冻结模型](/results/12-sac/model.pt)、[完整训练记录](/results/12-sac/training.json)、[响应 CSV](/results/12-sac/SAC.csv)。
