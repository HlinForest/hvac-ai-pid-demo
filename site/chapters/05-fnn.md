# 05 FNN：把多对象整定压缩成一次增益推荐

这一章的 FNN（fuzzy neural network）不是在闭环中学习的深层网络。
它是一个固定高斯前件、可学习后件的零阶 TSK 模型。
训练目标来自许多对象上已经完成的 BO 试验，推理时只根据一个新对象的三项辨识值给出一组 Kp、Ki。

FNN 的输出随后交给同一个条件积分 PI。
它改变的是闭环开始前的两个增益，不读取运行中的温度序列，不维护隐藏状态，也不在 0.1 min 采样周期内更新参数。
因此它把 BO 的多次试验成本换成一次性模型推理，但仍然依赖阶跃辨识和训练标签的覆盖范围。

## 先看它改变了 PI 的哪一部分

对象先由阶跃响应辨识为：

$$
x=(K,\tau,L),
$$

其中 $K$ 是制冷增益（℃），$\tau$ 是时间常数（min），$L$ 是输入延迟（min）。
FNN 先预测两个对数增益，再通过指数映射得到两个正数：

$$
\widehat y=(\widehat{\log K_p},\widehat{\log K_i}),
\qquad
(\widehat K_p,\widehat K_i)=\exp(\widehat y).
$$

Kp 的单位是 ℃⁻¹，Ki 的单位是 (℃·min)⁻¹。
指数映射只保证预测增益为正，不保证它们一定在 BO 的提案边界内，也不保证闭环一定稳定。

拿到增益以后，在线路径仍是 core.py 的 PI.update。
其中 I 表示条件积分逻辑提交后的积分项：

$$
I^\prime=I+K_i e\Delta t,\qquad
u=\operatorname{clip}_{[0,1]}(K_p e+I).
$$

当输出饱和且误差仍把输出推向饱和方向时，代码不提交这一步的积分项。
FNN 没有替换这个逻辑，也没有学习一个新的温度状态方程。

## 输入、输出和范围

FNN.predict() 接收一个 Identified 对象，并取它的 gain、tau、delay 三个标量。
单对象公开输入形状是 (3,)，批量训练输入是 (N,3)。
当前训练范围由 fnn.py 固定为：

$$
K\in[8,12]\ {}^\circ\mathrm C,\qquad
\tau\in[12,32]\ \mathrm{min},\qquad
L\in[1,5]\ \mathrm{min}.
$$

| 张量或字段 | 形状 | 单位或含义 |
| --- | --- | --- |
| 对象输入 x | (3,) 或 (N,3) | K/℃、tau/min、delay/min |
| 归一化输入 | (N,3) | 三个无量纲坐标 |
| 前件隶属度 | (N,3,3) | 三个输入各有低/中/高 |
| 规则强度 | (N,3,3,3) | 三个输入隶属度的乘积 |
| 归一化规则权重 W | (N,27) | 每行权重和为 1 |
| 后件矩阵 C | (27,2) | 54 个可学习系数 |
| log 增益预测 | (N,2) | log Kp、log Ki |
| Gains 输出 | (2,) | Kp、Ki 两个正数 |

它不接收温度、误差、上一拍指令、扰动时刻或真实对象内部状态；这些变量属于闭环仿真，不是 FNN 的特征。

## 固定前件如何得到 27 条规则

先按训练范围做逐维归一化：

$$
\widetilde x_i=\frac{x_i-x_{i,\min}}{x_{i,\max}-x_{i,\min}}.
$$

三个高斯中心固定在 0、0.5、1，宽度参数固定为 0.35：

$$
\mu_c(\widetilde x_i)
=\exp\left[
-\frac12\left(\frac{\widetilde x_i-c}{0.35}\right)^2
\right],
\qquad c\in\{0,0.5,1\}.
$$

每个输入选一个低、中或高隶属函数。
三维组合总数为 $3^3=27$。
规则的未归一化强度是三个隶属度的乘积：

$$
w_{abc}=
\mu_a(\widetilde K)\,
\mu_b(\widetilde\tau)\,
\mu_c(\widetilde L).
$$

最后除以所有 27 条规则强度之和：

$$
\bar w_{abc}=
\frac{w_{abc}}{\sum_{a,b,c}w_{abc}}.
$$

代码使用 einsum 形成 (N,3,3,3)，再按固定顺序 reshape 成 (N,27)。
它没有学习中心，也没有学习宽度。
归一化使输出成为后件的连续加权组合，而不是先选中一个硬网格单元。

<<< @/../hvac_pid/fnn.py#memberships

## 一个输入的权重计算

取参考范围中心的对象 $(K,\tau,L)=(10,22,3)$。
归一化后是 $(0.5,0.5,0.5)$。
每个输入的中间隶属度是 1，两侧隶属度由代码得到：

$$
a=\exp\left[-\frac12\left(\frac{0.5}{0.35}\right)^2\right]
=0.3604477886.
$$

因为三个维度相同，未归一化权重可按侧边维度的个数分组。
全中规则的强度为 1，只有一个侧边的规则强度为 $a$，有两个侧边的规则强度为 $a^2$，三个侧边的规则强度为 $a^3$。
总和也可以写为 $(1+2a)^3$。

展开索引 13 的中、中、中规则权重是 0.19621692。
只有一个输入落在侧边的规则权重是 0.07072595。
三个输入都在侧边的规则权重是 0.00918890。
27 个权重加总为 1。

这个数值例子说明了 FNN 的平滑性来源：对象参数在邻近范围内变化时，多条规则会同时改变权重。
平滑并不等同于外推可靠：超出输入范围时，代码仍会计算高斯值，但没有一个训练对象为该区域提供验证。

explain/fnn.json 保存单对象 input 的形状 (3,) 和 weights 的形状 (27,)。
批量矩阵计算才会临时使用 (1,3) 和 (1,27)。
这两个表示层级不能互换，否则会把一条对象记录误读成批量记录。

## 后件回归和 54 个参数

设训练对象的规则权重矩阵为 $W\in\mathbb R^{N\times27}$。
后件矩阵为 $C\in\mathbb R^{27\times2}$，目标矩阵为：

$$
Y=
\begin{bmatrix}
\log K_{p,1} & \log K_{i,1}\\
\vdots & \vdots\\
\log K_{p,N} & \log K_{i,N}
\end{bmatrix}
\in\mathbb R^{N\times2}.
$$

网络预测是：

$$
\widehat Y=WC.
$$

固定前件之后，训练是一个带 ridge 的线性最小二乘问题：

$$
C=(W^\mathsf{T}W+\lambda I_{27})^{-1}W^\mathsf{T}Y.
$$

实现用 np.linalg.solve 求解线性方程，没有显式构造逆矩阵。
ridge 只收缩 27×2 的后件系数，前件中心和宽度保持固定。
因此模型的可学习参数数量是 $27\times2=54$，而不是把每个训练对象保存成一套参数。

<<< @/../hvac_pid/fnn.py#fit

训练时先从 48 个对象构造 W 和 Y；每个对象的 Y 标签是该对象 5 个初始样本加 15 次 BO 提案中实际测到的最佳增益。
标签是有限预算的观测结果，不是连续增益平面上的理论最优解。

## 数据分区和 ridge 选择

固定分区记录在 splits.json。
48 个对象只用于拟合后件，12 个 validation 对象用于比较 ridge，另有 12 个 test 对象留给后续比较。
validation 的 IAE 是把每个候选网络输出送回同一个闭环仿真器后得到的，不是只比较 log 增益回归误差。
测试 IAE 没有反馈到 ridge 选择。

参考训练在三个候选值上的结果是：

| ridge λ | 验证集平均 IAE / ℃·min |
| ---: | ---: |
| 0.0001 | 47.3449 |
| 0.001 | 45.9596 |
| 0.01 | 45.6304 |

最终模型使用 ridge=0.01。
它是以 validation 闭环 IAE 选择的，不是因为某个系数看起来更小。
ridge 太小会让后件跟随 48 个有限标签的偶然性，太大则会把不同对象的增益预测收缩到过于相近。

<!--@include: ../generated/fnn-selection.md-->

第一个训练对象的辨识值约为 (8.715739,24.798263,2.9)。
它的 BO 标签是 (Kp=0.8830017346, Ki=0.0914364435)。
这组标签恰好等于该对象的 ZN 初始样本，但在其他对象上标签可能来自 BO 提案。
训练代码只把最终 tuned.best 写入 gains，保留完整 trials 供追溯。

## 从辨识到一次推理

默认对象先调用 identify()，得到 (8.0,20.0,2.0)。
FNN.predict() 将其包装为一条 (1,3) 输入，调用 features() 得到 (1,27) 权重。
随后执行：

$$
(1,27)\times(27,2)=(1,2).
$$

这两个 log 增益取指数后成为 Gains(kp,ki)。
参考运行的默认输出是 Kp=0.7855128312、Ki=0.1504231524。
该推荐随后固定到 1200 步 PI 仿真，不再更新后件。

默认对象的闭环指标为：

| 方法 | IAE / ℃·min | 最大过冷 / ℃ | movement |
| --- | ---: | ---: | ---: |
| FNN | 48.4676 | 0.0336 | 1.8116 |

<!--@include: ../generated/05-fnn.md-->

这个 IAE 略低于附带 BO 记录的 48.7413，但不能解释为网络超过了全局最优。BO 标签只覆盖有限的候选，FNN 的平滑组合可能落到教师未测试的参数附近。
同样的平滑也可能在另一个对象上产生更差的折中，所以 validation 和 holdout 仍然需要通过闭环指标判断。

![教师参数、预测参数与学到的规则后件](/results/05-fnn/learning.svg)

## 真实训练代码路径

experiments/run.py 的 fnn() 先读取 split，再对 train 和 validation 的每个场景调用 identify()。
每个场景调用 tuning.tune()，默认得到 20 个仿真 trial。
labels.json 在生成每一条记录后写入辨识值、BO 标签、成本和完整试验列表。

所有标签完成后，代码从 train 行提取 identified 的 gain、tau、delay，形成 (48,3) 输入。
它从 gains 提取 (48,2) 正增益目标，并在 fit() 内转换成 log 目标。
三个 ridge 各自 fit 一个临时模型，再对 12 个 validation 场景调用 predict() 和 evaluate()。

mean_validation_iae 最小的模型被保存到 model.npz。
随后用它计算训练对象的预测，生成 learning.svg，再对默认场景做一次预测和闭环评价。
training.json 保存选中的 ridge、nominal_gains、分区数量和成本字段。

这个过程没有在线更新的分支。如果 model.npz 被加载，load() 只恢复 27×2 的 consequents；前件范围、中心和宽度仍由源代码常量决定。
模型文件不是一个包含对象状态或温度历史的完整控制器。

## 外推、标签和目标的限制

输入范围是训练数据的设计范围，而不是 features() 的运行时校验范围；把 K=20 或 L=8 传入时，代码仍会归一化并计算高斯权重。
这种行为是数值上的外推，不是已验证的泛化能力。

教师标签来自每个对象自己的有限预算。
若某次 BO 预算没有找到好点，FNN 会把这个有限试验结果当作监督目标。
标签的噪声和对象辨识误差会通过 W、Y 进入后件系数。

当前目标只最小化每个闭环的 IAE。
过冷和 movement 被记录，但没有用于 ridge 选择或后件回归损失。
若真实约束需要限制过冷、阀门变化或运行时间，必须将它们纳入明确的评价协议。

FNN 只有三个静态对象特征；它没有观察扰动后误差是否恢复，也没有根据在线状态重算增益。
固定推荐在对象参数变化后仍然沿用旧输出，除非外部流程重新辨识并重新调用模型。

## 离线成本、在线调用和产物

参考训练使用 48 个 train 和 12 个 validation 对象。
每个对象的教师标签含 5 个初始样本和 15 个 BO 提案，因此 label_simulations 为 1200 次。
对应的 label_plant_steps 为 1,440,000。

training.json 还记录 label 的辨识步数 288,000、ridge 选择仿真 36 次和 selection_plant_steps 43,200。
这些字段是本次编排的成本账，不代表一台设备必须按相同 CPU 速度完成真实试验。
参考机器的总墙钟约 11.8524 s，标签阶段约 10.6966 s，fit 和 selection 阶段约 1.1556 s。

FNN 的在线路径是一次辨识、一次 (1,27)×(27,2) 矩阵乘法，然后进入固定 PI。
PI 仍每 0.1 min 更新一次温度控制量；FNN 不在每周期调用。
model.npz 只保存 27×2 个后件数组，参考文件大小约 708 bytes；磁盘文件大小不等于运行时内存占用，也不构成 MCU 资源保证。

| 产物 | 作用 |
| --- | --- |
| labels.json | 48/12 对象的辨识值、BO 标签、试验和成本 |
| splits.json | train、validation、test 对象分区 |
| model.npz | 27×2 的 consequents |
| training.json | ridge 选择、默认输出和成本 |
| learning.svg | 教师增益、预测增益和规则后件 |
| FNN.csv | 默认推荐的温度、输出和固定增益 |
| response.svg | 默认闭环响应 |

本章数据下载：[模型](/results/05-fnn/model.npz)、[训练记录](/results/05-fnn/training.json)、[标签 JSON](/results/05-fnn/labels.json)、[响应 CSV](/results/05-fnn/FNN.csv)。

## 结果复现入口

默认 FNN 编排入口是：

```bash
python -m hvac_pid fnn --output outputs/tutorial
```

单条解释记录来自实际 features() 和 predict() helper：

```bash
python -m hvac_pid explain --method fnn --output outputs/tutorial
```

生成的 explain/fnn.json 记录单个输入 (3,)、27 个归一化权重和 (27,2) 后件形状；输出目录中的 labels、training、response 和 model 应作为同一 seed 与 rounds 的一组产物解释。

下一章开始讨论另一种边界：控制过程中每隔一段时间重新选择增益的在线策略。
