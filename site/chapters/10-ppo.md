# 08 PPO：连续动作先试，再把更新限制住

PPO（Proximal Policy Optimization）把前两章的九个离散增益组合换成两个连续动作：每 2 分钟分别给出 Kp、Ki 相对 SIMC 锚点的 log2 倍数。它仍然只负责慢速增益决策；PI 每 0.1 分钟运行一次，闭环中的积分器、输出限幅和对象状态继续由同一个控制路径维护。

本实现的训练数据流是“采一整条温控轨迹，再做四轮小批更新”。策略网络（actor）是把观测映射为动作分布的函数；价值网络（value network，也就是 critic）把观测映射为一个标量 V(s)，表示当前位置的长期回报估计。GAE 用 reward、V(s) 和下一状态价值构造优势 A_t，即该动作结果相对基准价值的改进估计；clipped objective 再限制一次更新对旧策略概率的改变。冻结运行时只保留 actor 的确定性均值路径，不保留 value network 或优化器。

## 两个时间尺度决定了 PPO 的边界

默认回合是 120 分钟、物理步长 0.1 分钟，即 1200 次 PI/Plant 物理步。TuningEnv 的策略间隔是 2.0 分钟，对应 20 个物理步；一条完整轨迹有 60 个连续动作和 60 个 reward。

| 层次 | 周期 | 计算 | 冻结运行时是否需要 |
|---|---:|---|---|
| PI 与实际对象 | 0.1 分钟 | 误差 → PI.update → [0,1] 输出 → 对象响应 | 是，属于控制器/设备运行态 |
| PPO actor | 2 分钟 | 五维观测 → 5→64→64→4 → mean → tanh | 是，只做前向和增益换算 |
| PPO 训练器 | 离线 | rollout、GAE、ratio、反向传播和 Adam | 否 |

在仿真中，Plant 的温度和延迟队列由 core.py 保存；在真实控制器中这些是物理过程本身，不需要为了运行 actor 而复制一个 Plant 仿真器。PI 的积分器、当前输出和传感器采样历史仍是控制器状态，不能因为策略每两分钟被调用就重置。

## 五个观测量与固定尺度

actor 接收的原始观测是

$$
o=(e,\dot e,u,s_p,s_i)\in\mathbb R^5.
$$

它们依次表示当前温度误差（℃）、过去 2 分钟的误差斜率（℃/min）、上一物理步的制冷输出、当前 Kp/anchor.kp 和 Ki/anchor.ki。后两项是无量纲增益比，动作也只在这个相对锚点的坐标中定义。

continuous.py 与 PPO 更新都使用同一固定除数：

| 原始分量 | 除数 | 网络输入 |
|---|---:|---|
| e | 4 | e/4 |
| \dot e | 0.2 | \dot e/0.2 |
| u | 1 | u |
| s_p | 2 | s_p/2 |
| s_i | 2 | s_i/2 |

policy_observation 将结果变为形状 (1,5) 的 float32；一条 rollout 保存的 states 在更新时变成 (60,5)。scale 写入连续策略 checkpoint 的 metadata，read_checkpoint 会检查它，因此改变 scale 会被识别为协议不匹配，而不是静默地产生另一种控制器。

这五个量仍是部分观测。延迟队列、PI integral、对象 gain/tau/delay 和扰动相位没有直接输入；同一观测可能对应不同的隐藏动态。PPO 的随机动作和 GAE 只改变估计与优化方式，不能把这个五维观测自动变成完整 Markov 状态。

## 连续动作从高斯 latent 到物理增益

actor 的 body 是 5→64→64→4。最后四个输出分成两组，每组形状为 (B,2)：mean 是高斯 latent 的均值，log_std 是对数标准差。log_std 在 [-20,2] 内裁剪，防止分布在训练早期变成零宽或指数爆炸。

训练时对每个决策点采样

$$
z=\mu+\exp(\log\sigma)\epsilon,\qquad
\epsilon\sim\mathcal N(0,I_2),
$$

再把 latent 通过 tanh 得到执行动作

$$
a=\tanh(z)\in[-1,1]^2.
$$

core.py 的 step_continuous 接收这两个动作，做 scales=exp2(action)，再构造

$$
K_p=K_{p,0}2^{a_0},\qquad K_i=K_{i,0}2^{a_1}.
$$

因此 a=-1、0、1 分别对应锚点的 0.5、1、2 倍；它是 log2 倍数而不是温度或制冷输出。动作每次覆盖当前两项增益，不会把上次倍数再次相乘。actor 参数总数是

$$
(5\times64+64)+(64\times64+64)+(64\times4+4)=4804.
$$

普通高斯的密度定义在 z 上，但环境执行的是 tanh(z)。变量变换的 log probability 必须扣除 tanh Jacobian：

$$
\log\pi(a|s)=
\sum_j\left[
\log\mathcal N(z_j;\mu_j,\sigma_j)
-\log(1-\tanh^2(z_j))
\right].
$$

共享实现从 pre-tanh 的 z 计算稳定的 correction，而不是从已经饱和的 a 反推；|z| 很大时仍能保持数值稳定。这个 log probability 同时用于记录旧策略概率和后续的新旧 ratio，少了 Jacobian 就不是当前执行动作的概率。

<<< @/../hvac_pid/continuous.py#gaussian_sample

冻结的 PPOPolicy.choose 不再采样 epsilon，而是取 actor 的 mean 后执行 tanh(mean)。所以训练时的高斯探索和冻结曲线的确定性策略是两条不同路径；训练 reward 不能直接视为最终响应。

## 一条轨迹如何形成 PPO batch

训练先为 48 个对象各做一次 240 分钟 commissioning，缓存对应的 SIMC anchor。每个 episode 随机抽一个对象，直到 done 为止重复以下数据流：

1. 用当前 actor 对归一化观测得到 mean、log_std，采样 pre-tanh z 和 a。
2. 把 a 传给 TuningEnv，运行 20 个 0.1 分钟物理步，得到区间 reward。
3. 保存 raw state、z、old_log_prob、当前 value、next value 和终止标志。
4. 用这一整条轨迹计算 advantages 与 returns，然后更新 actor 和 value。

一个完整回合的数组形状是 states (60,5)、pre_tanh_actions (60,2)、old_log_probs (60,)、rewards (60,)、values (60,) 和 next_values (60,)。动作本身在环境中是 tanh(z)，但更新必须保留 z，因为 gaussian_log_prob 需要 pre-tanh 数值来重算 Jacobian 修正。

core.py 的默认时长正好结束于 scenario.steps，因此 continuous.environment_step 返回 is_terminal=True、is_truncated=False 的最后一条转移。这里没有另加终端惩罚；reward 仍只是这段 2 分钟的负 IAE，一回合 history.return 是 60 个 reward 的直接相加，不折扣。

compute_gae 同时支持真实终止和 rollout 截断。令 I_term 表示 terminated，
I_trunc 表示 truncated：

$$
\delta_t=r_t+\gamma(1-I_{\rm term})V(s_{t+1})-V(s_t),
$$

$$
A_t=\delta_t+\gamma\lambda(1-I_{\rm term})(1-I_{\rm trunc})A_{t+1}.
$$

真实终止时，TD 残差不使用下一价值，GAE 也不跨回合递归。若固定采样窗口在一个仍可继续的对象上被截断，边界转移仍可以使用 next value 做 delta，但 continuation mask 为 0，优势不会跨越这条 rollout 边界。参考运行只有真实回合终止，所以最后一条使用 terminal mask；这两个标志不能合并成一个“done 后全部清零”的实现细节。

## GAE、归一化和 clipped objective

本实现 gamma=0.99、GAE lambda=0.95。compute_gae 从最后一条向前保存 carry，得到 advantages，再用

$$
R_t=A_t+V(s_t)
$$

作为 value network 的回归 target。每条完整轨迹的 advantage 在更新前按整条轨迹标准化：

$$
\hat A_t=\frac{A_t-\operatorname{mean}(A)}
{\operatorname{std}(A;\mathrm{unbiased}=False)+10^{-8}}.
$$

这一步改变 policy gradient 的数值尺度，但不改变环境 reward。60 个样本少于默认 batch size 64，所以每个 epoch 实际是一个 60 行 minibatch；代码随机打乱这 60 行，连续做 4 个 epoch，故每回合有 4 次 optimizer update。

PPO 的概率比使用同一个已保存的 pre-tanh z。下面的 log π(a|s) 指已经包含 tanh Jacobian 修正的 squashed-action 概率；新旧两项使用同一 z 和同一变换，因此 Jacobian 在 ratio 的代数差中相互抵消，但代码仍从 z 计算两项修正后的 log probability：

$$
r_t(\theta)=\exp\left(\log\pi_\theta(a_t|s_t)
-\log\pi_{\mathrm{old}}(a_t|s_t)\right).
$$

代码构造两个 surrogate：

$$
u_t=r_t\hat A_t,\qquad
c_t=\operatorname{clip}(r_t,0.8,1.2)\hat A_t,
$$

并使用

$$
L_\pi=-\operatorname{mean}\left[\min(u_t,c_t)\right].
$$

这个 min 不是把实际 ratio 永久硬截断。对正优势，ratio 超过 1.2 时 min 选择 1.2 倍的项，限制继续放大好动作；ratio 低于 0.8 时仍可使用未裁剪项。对负优势，乘法会反转大小关系，ratio 低于 0.8 时 min 选择 0.8 倍的项，限制继续压低坏动作的概率；ratio 高于 1.2 时仍可使用未裁剪项。ratio 本身仍由 exp(log_prob-old_log_prob) 产生，可能落在区间外；被 clamp 的只是比较用的第二个 surrogate。

总损失为

$$
L=L_\pi+0.5\,\operatorname{MSE}(V,R)-0.01\,\operatorname{mean}(H),
$$

其中代码中的 entropy 是从 actor 新采样一个 bounded action 后取 -fresh_log_prob 得到的估计，Jacobian 修正也在其中。value network 只服务于训练，不参与冻结动作；actor optimizer 和 value optimizer 共享一次 backward 的标量 total，但两者参数集合独立。

<<< @/../hvac_pid/ppo.py#ppo_gae

<<< @/../hvac_pid/ppo.py#ppo_loss

## 首个实际 minibatch 的数值

training.json 的 first_update 是首个 epoch、首个随机 minibatch 中被记录的一行。它不是一个单独样本的完整 loss：states 批量形状为 (60,5)，pre-tanh actions 为 (60,2)，value、return、advantage 都按 60 行计算。记录样本的实际字段为：

| 字段 | 值 |
|---|---:|
| normalized state | [0.0173864, -0.0264151, 0.6576164, 0.6941699, 0.9268699] |
| pre-tanh action z | [0.8445497, 0.4992776] |
| actor mean μ | [0.1047429, 0.1014479] |
| log_std | [0.0408457, 0.0448138] |
| value V(s) | -0.0648556 |
| raw advantage | -0.5486007 |
| normalized advantage | 0.6646457 |
| return R | -0.6134563 |
| old/new log probability | -1.3667552 / -1.3667552 |
| ratio | 1.0000000 |

这是更新前重新读取同一批 z 的第一行，所以 old 和 new log probability 相同，ratio 为 1。raw advantage 经过整批标准化后变为正数，是因为它相对该批平均值的位置发生了变化；不能只凭 raw 数字猜 policy 项符号。

整批实际记录的 policy_loss 约 3.18e-8、value_loss 为 160.6029358、entropy 为 1.3137754、total loss 为 80.2883301。policy_loss 接近零来自 ratio=1 时对标准化优势的批量平均；该样本单独的策略项仍是 -1×0.6646457。value_loss 是 60 个 value 与 return 的均方误差，不是 (-0.0648556+0.6134563)^2 的单样本值。

<!--@include: ../generated/first-update-ppo.md-->

<!--@include: ../generated/10-ppo.md-->

真正把轨迹数组变成 torch batch、做四个 epoch 的代码如下；它也保留了首个 minibatch 的 mean、log_std、value、ratio 和三个损失分量。

<<< @/../hvac_pid/ppo.py#ppo_update

## 训练结果、失败模式与冻结响应

500 回合参考运行产生 30,000 个两分钟转移、600,000 个物理步、2,000 个 minibatch updates（每回合 4 个 epoch），commissioning 另计。训练 CPU 耗时为 33.5178 s。history.return 是不折扣的负 IAE，gamma 只参与 value target 和 GAE；训练过程中 actor 使用高斯探索，冻结曲线使用 tanh(mean)。

![PPO 训练回报与损失](/results/10-ppo/learning.svg)

一个容易漏掉的失败点是 tanh 饱和：pre-tanh z 变大后动作靠近 ±1，从动作本身估计 Jacobian 会丢精度，必须按 continuous.py 从 z 计算稳定修正。另一个失败点是 rollout 边界：把 truncated 和 terminated 混为一类，会错误地丢掉可 bootstrap 的边界价值，或让 GAE 穿过本不应连接的下一条轨迹。

若 value loss 下降而闭环 IAE 没有下降，仍需检查动作是否经常贴近 ±1、旧 log probability 是否与同一 z 对应，以及冻结 actor 的确定性响应。clipping 控制的是每次 policy update 的 surrogate，不保证延迟对象上的温度响应一定改善。

![PPO 冻结 actor 的温度和在线增益](/results/10-ppo/response.svg)

参考默认对象的 PPO IAE 为 78.5673 ℃·min，SIMC 为 77.9950 ℃·min；两者最大过冷都为 0，PPO movement 为 1.4584，SIMC 为 1.3750。验证分区平均 IAE 为 76.5256 ℃·min。这个结果应与训练 loss 和默认响应一起阅读，不能只用其中一个指标宣称策略改善。

## 冻结运行需要什么，排除什么

冻结 PPO 的在线数据流是：

$$
\text{五维观测}
\rightarrow\text{固定 scale}
\rightarrow\text{actor mean}
\rightarrow\tanh
\rightarrow\exp_2
\rightarrow\text{SIMC 锚点的 Kp,Ki}
\rightarrow\text{PI 每 0.1 分钟执行}.
$$

在线需要 actor 权重、checkpoint 中的动作边界/scale/interval 约定，以及与对象匹配的 SIMC anchor。PI integral、当前输出和传感器历史由控制器保留；仿真评估时的 Plant 温度和 delay queue 属于模拟器状态，真实设备由物理过程提供，不应为了 actor 推理再额外运行一份 Plant。

在线不需要 value network、replay、GAE 数组、old_log_probs、advantages、returns、entropy、optimizer、随机探索或四个 update epoch。PPOPolicy.choose 使用 no_grad、eval 和 tanh(mean)，因此不会改变 checkpoint 或更新任何训练统计。checkpoint 的 model.pt 保存 actor state_dict 及连续策略 metadata，但 anchor 仍是外部 commissioning 配置。

benchmark 的实际 CPU 参考值为：

| 组件 | 调用周期 | 参数或表项 | 工件 bytes | 中位 / P95 μs |
|---|---:|---:|---:|---:|
| PI control update | 0.1 min | 2 | 0 | 1.2 / 1.9 |
| Q-Learning frozen policy | 2.0 min | 6075 | 48856 | 26.7 / 56.05 |
| DQN frozen policy | 2.0 min | 1545 | 8829 | 68.5 / 130.355 |
| PPO frozen actor | 2.0 min | 4804 | 22397 | 153.05 / 302.61 |

这是 100 次 warm-up、1000 个 batch=1 CPU 调用的 wall-time，包含观测转换，不含梯度和探索。文件 bytes 是 reference artifact 大小，不是目标 MCU 的内存占用；μs 是参考主机上的测量，不能直接当作硬件时序保证。

本章依据 [PPO 原论文](https://arxiv.org/abs/1707.06347) 的 clipped
surrogate，但网络宽度、完整回合 rollout、4 个 epoch 和 HVAC 动作映射
都是本项目配置，不是论文实验结果。

本章下载：[冻结模型](/results/10-ppo/model.pt)、[完整训练记录](/results/10-ppo/training.json)、[响应 CSV](/results/10-ppo/PPO.csv)。

## 结果复现入口

以下命令生成 PPO 的训练记录、冻结 actor、响应 CSV 和图；explain 读取已有 training.json，不会再跑一遍训练。

~~~bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m hvac_pid ppo --episodes 500 --seed 0 --output outputs/tutorial
python -m hvac_pid explain --method ppo --output outputs/tutorial
~~~
