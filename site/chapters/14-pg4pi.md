# 12 PG4PI：把 PI 的两个参数直接作为策略参数

前面几章训练一个神经网络，让它根据观测选择 PI 增益。这里改变策略的结构：待学习的量就是两个数，$K_p$ 和 $K_i$。一条训练轨迹内固定这两个数；跑完整条轨迹，计算梯度，再更新参数。训练结束后，控制循环只执行普通 PI。

这一章来自 [RL-optimal-pid](https://github.com/sharma1256/RL-optimal-pid) 对应的 [NeurIPS 2025 论文](https://papers.nips.cc/paper_files/paper/2025/hash/e40e65df08fc9b925c1da59fc25a2bd6-Abstract-Conference.html)。仓库提供 model-based PG4PID 和 model-free PG4PI。本项目采用后一条路线，并将温控适配命名为 **PG4PI-HVAC**。原方法的线性系统设定、目标函数和理论条件，与这里的延迟、输出限幅、条件积分和有限时间 IAE 有区别；温控曲线不能用来证明原论文的全局最优性结论。

## 产物和评价口径

整定只在本地仿真运行，不调用模型接口，也不需要 PyTorch。输出在 outputs/tutorial/14-pg4pi/：

| 文件 | 先看什么 |
|---|---|
| `result.json` | 初始参数、最终参数、噪声设置和第一次更新 |
| `trials.csv` | 每条轨迹使用的参数、回报和估计梯度 |
| `PG4PI-HVAC.csv` | 关闭探索后，用最终参数运行的完整响应 |
| `learning.svg` | 有噪声的训练回报与参数变化 |
| `response.svg` | 最终固定 PI 与初始 SIMC 的响应 |

![PG4PI-HVAC 的训练回报与参数](/results/14-pg4pi/learning.svg)

左图来自有探索噪声的训练，不能直接作为最终控制性能。右图中，每个点代表一条轨迹使用的一组固定参数；相邻点的变化发生在两条轨迹之间，不是在一次控制过程中每两分钟重新选增益。

## 第一步：把 PI 写成带参数的策略

沿用第一章的误差 $e_t=T_t-T_{\mathrm{set}}$。积分状态 $b_t$ 保存的是控制输出贡献：

$$
b_{t+1}^{\mathrm{candidate}}=b_t+K_i e_t\Delta t.
$$

先用候选积分检查抗饱和条件。允许积分时接收候选值，否则保留 $b_t$；随后用**实际保留下来的积分**计算：

$$
\mu_t=\operatorname{clip}(K_pe_t+b_{t+1},0,1).
$$

$\mu_t$ 就是共用 PI 返回的控制量。与前几章一样，$K_p$ 的单位是 ℃⁻¹，$K_i$ 的单位是 $({}^{\circ}\mathrm{C}\cdot\mathrm{min})^{-1}$。

训练时在这个均值周围采样：

$$
z_t=\mu_t+\sigma\epsilon_t,\qquad
\epsilon_t\sim\mathcal N(0,1),\qquad
u_t=\operatorname{clip}(z_t,0,1).
$$

区分两个变量：`latent` 保存 $z_t$，`command` 保存对象实际收到的 $u_t$。例如 $z_t=1.04$ 时，对象收到 1；计算高斯对数概率时仍然使用 1.04。

这也是训练噪声与执行限幅的边界：参数学习需要知道采样出了什么，执行对象只能接受合法的制冷指令。

## 第二步：固定误差历史，求 PI 均值的导数

先不连接对象，给 PI 一串已知误差。这样可以单独核对控制器导数：

```python
from hvac_pid.core import Gains
from hvac_pid.pg4pi import pi_mean_and_sensitivity

means, jacobian = pi_mean_and_sensitivity(
    [1.0, 0.8, 0.6], Gains(0.2, 0.04), dt=0.1
)
print(means)
print(jacobian)
```

第一步候选积分为 $0+0.04\times1\times0.1=0.004$，输出 $\mu_0=0.204$，没有触发限幅。对应导数为：

$$
\left[\frac{\partial\mu_0}{\partial K_p},
\frac{\partial\mu_0}{\partial K_i}\right]=[1,0.1].
$$

第二步累计积分为 $0.0072$，输出 $0.1672$；导数为 $[0.8,0.18]$。第三步输出为 $0.1296$，导数为 $[0.6,0.24]$。`means` 的形状是 `(3,)`，`jacobian` 是 `(3,2)`，两列分别对应 Kp 和 Ki。

这里必须传播积分状态的敏感度。若只对当前的 $K_i e_t\Delta t$ 求导，就会漏掉此前已经接受的积分。

令 $s_t=\partial b_t/\partial(K_p,K_i)$，允许积分时：

$$
s_{t+1}=s_t+[0,e_t\Delta t].
$$

拒绝积分时 $s_{t+1}=s_t$。最终输出未饱和时，均值导数是 $[e_t,0]+s_{t+1}$；输出落在限幅外侧时，当前光滑分支的导数为零。限幅和条件积分的切换点不可微，测试要区分分支内部的导数与边界约定。

<<< @/../hvac_pid/pg4pi.py#pg_sensitivity

一个特别容易写错的例子：设 $K_p=0.2$、$K_i=1$、$e=4$、$\Delta t=0.1$、原积分为零。候选输出为 1.2，因而拒绝积分；最终输出应为 $0.2\times4+0=0.8$，不是 1。敏感度计算必须跟随共用 PI 的实际分支。

以上导数都**以观察到的误差历史为条件**。没有计算温度对参数的导数，也没有对对象的 K、τ、L 求导；这是后续 score-function 估计与直接对仿真模型反向传播的区别。

## 第三步：从控制输出样本得到 score

固定 $\sigma$，高斯策略对均值的对数概率导数为：

$$
\nabla_K\log p(z_t\mid h_t,K)
=\frac{z_t-\mu_t}{\sigma^2}\nabla_K\mu_t,
\qquad K=(K_p,K_i).
$$

$h_t$ 表示到当前时刻观察到的历史。虽然只优化两个参数，PI 的均值仍然依赖此前的积分状态。

继续使用第一步的 $\mu_0=0.204$、导数 $[1,0.1]$。取一个用于核算的采样值 $z_0=0.214$、$\sigma=0.1$，则：

$$
\frac{0.214-0.204}{0.1^2}[1,0.1]=[1,0.1].
$$

这只是一个样本的 score，还不是整条轨迹的性能梯度。它告诉我们：在当前参数下，增大参数会怎样改变这个随机控制输出的对数概率。

如果采样值超过执行边界，也仍然使用原始 $z_t$。把所有大于 1 的样本替换成 1 后再代入高斯公式，会改变估计量；执行端的限幅不意味着可以丢掉原始样本。

## 第四步：用轨迹回报给 score 加权

每个物理步的奖励沿用负绝对误差：

$$
r_t=-|e_t|\Delta t,\qquad R=\sum_t r_t=-\mathrm{IAE}.
$$

完整轨迹估计可以写成：

$$
\widehat{\nabla_K J}=R\sum_t\nabla_K\log p(z_t\mid h_t,K).
$$

默认实现采用 return-to-go 形式，将每个 score 的权重改成从该时刻到结束的奖励和，即 $G_t=\sum_{k=t}^{N-1}r_k$，计算 $\sum_tG_t\nabla_K\log p(z_t\mid h_t,K)$。这样不会用采样之前已经发生的误差给当前动作加权。两种形式都只使用已经采集的轨迹；`result.json` 保存 `return_to_go=true`。

<<< @/../hvac_pid/pg4pi.py#pg_score

注意梯度的符号：$J$ 是希望增大的负 IAE，因此使用梯度上升。一次负回报不表示两个参数都应该减小，更新方向还取决于 score 的符号。

噪声太小会使 $1/\sigma^2$ 项变大，少量样本的估计可能很不稳定；噪声太大则会使训练控制响应明显偏离最终的确定性 PI。训练曲线变差时，先查看采样值、score 和参数更新量，不能只看最终 IAE 判断代码是否正确。

## 第五步：把更新留在合法的参数空间

直接更新 Kp、Ki 可能得到负值。这里用两个无约束坐标 $\theta$ 表示增益，在与 BO 相同的对数参数范围内映射：

$$
K_j=\exp\bigl(\ell_j+(h_j-\ell_j)\operatorname{sigmoid}(\theta_j)\bigr).
$$

对应范围为 $K_p\in[0.01,3]$、$K_i\in[0.0001,0.5]$。初始值来自当前对象的 SIMC；若初始值不满足参数化条件，则明确报告错误。

<<< @/../hvac_pid/pg4pi.py#pg_parameters

用链式法则将增益梯度换成 $\theta$ 的梯度：

$$
\nabla_\theta J=
\left(\frac{\partial K}{\partial\theta}\right)^\top\nabla_KJ.
$$

随后只在轨迹结束后更新一次。下一条轨迹重新初始化对象和积分状态，整条轨迹使用更新后的固定参数。

<<< @/../hvac_pid/pg4pi.py#pg_update

打开 [第一次更新记录](/results/explain/pg4pi.json)，依次核对初始参数、回报、增益梯度、坐标梯度和更新后的参数。最终参数取训练结束时的值，不因为某条曲线更差就自动换回 SIMC。

参考运行的第 0 条轨迹从 `(0.28846154, 0.01442308)` 开始，带噪声回报为 `-79.08364733`。前 47 个物理步的均值处于饱和分支，均值导数为零；第 47 步，也就是 4.7 分钟，首次出现非零敏感度：

| 中间量 | 实际值 |
|---|---:|
| 当前误差 | 3.44671793 ℃ |
| 均值 μ | 0.99921678 |
| 限幅前样本 z | 1.19524262 |
| 执行指令 | 1 |
| 均值对 Kp、Ki 的导数 | `[3.44671793, 0.34467179]` |
| `(z-μ)/σ²` | 19.60258316 |
| 剩余回报 G₄₇ | -60.57591920 |

这一项给增益梯度贡献 `[-4092.78622495, -409.27862249]`。它清楚地展示了为什么需要保存限幅前样本：执行指令虽然是 1，梯度计算使用的仍是 1.19524262。

把整条轨迹的贡献相加，再乘参数映射的 Jacobian，得到坐标梯度 `[-3017.24990304, 6957.49468408]`。学习率为 `0.00001`，所以

$$
\theta^+=[0.36160675,0.33794665]
+0.00001[-3017.24990304,6957.49468408]
=[0.33143425,0.40752160].
$$

重新映射后得到 `(Kp,Ki)=(0.27666414,0.01664207)`。第一次更新让 Kp 减小、Ki 增大，不是给两个参数施加同一个缩放倍数。以上中间量均来自实际 `tune` 循环。

## 第六步：关闭噪声，回到共用评价入口

```python
from hvac_pid.core import Scenario, evaluate
from hvac_pid.pg4pi import tune

result = tune(Scenario(), seed=0, episodes=500)
evaluation = evaluate(Scenario(), result.best)
print(result.best, evaluation.metrics())
```

这里的 `evaluate` 来自共用内核。最终评估没有 PG4PI 专用的温控模型、探索噪声或在线更新。

![最终固定 PI 的确定性响应](/results/14-pg4pi/response.svg)

<!--@include: ../generated/14-pg4pi.md-->

训练回报与上表的差别有两个来源：前者包含输出探索，且每条轨迹可能使用不同增益；上表使用最终一组参数并关闭噪声。如果没有改善，保留结果，再检查更新是否过大、梯度方差是否明显，以及参数是否接近映射边界。

本次 500 条轨迹之后，最终参数为 `(0.63425671, 0.03746760)`，确定性 IAE 为 `62.0793`，小于 SIMC 的 `77.9950`；输出总变化量从 `1.3750` 增至 `1.4065`。这只说明当前对象和当前种子的表现。与 RL 的预训练模型不同，得到这两个数已经为当前对象支付了 500 条轨迹的成本；新对象需要重新运行。

### 五个种子让失败显现出来

同样的 500 条轨迹预算，种子 0 到 4 的名义工况 IAE 分别为 `62.08、52.82、107.58、164.50、82.30`，平均 `93.86`，样本标准差 `44.74`。因此不能用上面 seed 0 的曲线宣称 PG4PI 优于 SIMC。

在未见对象 6 上，seed 2 的最终增益降到了 `(0.01320837, 0.00016400)`，接近搜索范围下界，最终 IAE 为 `626.9151`。可以直接检查这个失败：

![PG4PI 在未见对象上的低增益失败](/results/repeats/pg4pi-failure/response.svg)

低增益使比例输出偏小，积分贡献累积也慢，120 分钟内留下了很大的正误差。最后三条带噪声轨迹的回报仍约为 `-619、-620、-621`，参数没有恢复。边界附近 sigmoid 映射的导数变小，加上单轨迹 score 估计的方差，会让后续更新难以把参数移回来；这是与记录一致的机制分析，不是已证明的唯一原因。

下载 [全部 500 次更新](/results/repeats/pg4pi-failure/result.json) 和 [最终响应](/results/repeats/pg4pi-failure/PG4PI-HVAC.csv)，从参数开始接近下界的那一轮向前检查梯度。该对象没有用于修改学习率或重新挑选种子，结果也没有替换成 SIMC。

## 原方法核对与温控适配分别保存

仓库提供 `pg4pi-reference` 独立入口，固定上游提交，在隔离的依赖环境运行原设定。其结果与 `14-pg4pi` 分开保存，记录实际成功或失败状态。基础安装与 CI 不自动拉取和运行上游项目。

首次运行需要联网获取固定提交和独立依赖；该入口需要前面神经网络章节安装的 CPU PyTorch。适配包安装在输出目录的 `deps`，NumPy、SciPy、Matplotlib 和 PyTorch 继承当前环境。之后查看该目录的 `config.json`、`one-update.json` 和 `results.json`。命令返回失败时先读保存的错误信息，不能将“已下载源码”当作“已完成实验”。

阅读 [上游源码](https://github.com/sharma1256/RL-optimal-pid/blob/37c8882d3828a2811120f9467a9e88b4bd95449f/PG4PI.py) 时，先找到四件事：策略输出、轨迹采集、回报累计、PI 参数更新；再对照本章的限幅、积分状态和 IAE 修改。固定提交信息见仓库的 `experiments/pg4pi-upstream.json`，不要把会变化的 `main` 当作实验版本。

原设定核对回答“是否执行并理解了上游方法”；温控适配回答“带这些工程条件时，它表现怎样”。两者的环境、奖励和结果不能合并成同一张成绩表。

<!--@include: ../generated/pg4pi-reference.md-->

核对数据：[原设定配置与状态](/results/pg4pi-reference/config.json)、[一次更新](/results/pg4pi-reference/one-update.json)、[更新前中间量](/results/pg4pi-reference/update-trace.json)、[原设定运行结果](/results/pg4pi-reference/results.json)。

本次固定提交的 REA 实验实际跑完 10 轮。第 0 轮记录了 $a=-0.0136638582$、$\mu=-0.0033419730$、$G=-72.5305951$、`score_p=3.04256535` 和 $\alpha=0.01$。上游脚本执行的是下面这个更新：

$$
K_p^+=K_p-\alpha(a-\mu)G\,score_p
=0.001-0.01(-0.0103218852)(-72.5305951)(3.04256535)
\approx-0.02177824.
$$

此时 `score_i=0`，所以 Ki 仍为 0.001。这里的 `score_p` 是上游函数返回的导数项，不能把它直接当成前面已经包含高斯噪声系数的完整 score。原程序的控制符号、状态扩展与更新式也不同；负 Kp 是这次原设定的记录，不是本项目制冷 PI 可使用的参数。温控适配明确改用历史条件下的完整高斯 score、积分敏感度、正参数化和负 IAE。

## 在线放在哪里，要留下什么

PG4PI 的轨迹采样、梯度、随机数和参数更新都在整定阶段运行。正式控制时只留下 Kp、Ki 和普通 PI 的积分状态，每 0.1 分钟计算一次控制量。无需 Critic、回放池，也无需神经网络前向计算。

它的主要成本是为当前对象采集训练轨迹。默认 500 条、每条 1200 个物理步，共 600,000 步；这些都是仿真交互，不能把仿真完成时间解释为真实温控对象的整定时间。换一个对象重新整定时，成本重新计算，不能记成一次免费的参数预测。

## 结果复现入口

默认预算为 500 条训练轨迹，冻结评估关闭探索：

```bash
python -m hvac_pid pg4pi --episodes 500 --seed 0 --output outputs/tutorial
python -m hvac_pid explain --method pg4pi --output outputs/tutorial
```

`trials.csv` 保存每条轨迹的参数、回报和梯度，`result.json` 保存最终增益及交互成本。预算增加意味着更多当前对象试验，不保证最终 IAE 单调下降；本章五种子结果已展示参数更新的不稳定性。

原设定核对使用独立入口：

```bash
python -m hvac_pid pg4pi-reference --install-deps --seed 0 --output outputs/pg4pi-reference
```

本章数据：[逐轨迹记录](/results/14-pg4pi/trials.csv)、[配置与梯度](/results/14-pg4pi/result.json)、[最终响应 CSV](/results/14-pg4pi/PG4PI-HVAC.csv)。
