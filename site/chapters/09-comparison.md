# 14 统一比较：把收益、失败和成本放在同一张表

前面的章节让不同方法各自完成了一个实验。本章规定比较协议，再把 12 种方法放在同一对象、同一 120 分钟窗口、同一 IAE 计算下：

1. Z-N；
2. SIMC；
3. BO；
4. FNN；
5. Q-Learning；
6. DQN；
7. PPO；
8. TD3；
9. SAC；
10. CrossQ；
11. PG4PI-HVAC；
12. LLM。

这里的“12 种”是整定方法的数量，不是声称它们都在真实设备上部署过。现代 RL 方法使用 64×64 量级的小型 CPU 网络，目标是让读者看清动作、更新和失败方式，不是论文级 SOTA 排名。

## 先生成同一批产物

完整本地实验（包含新方法）使用：

```bash
python -m hvac_pid all --output outputs/tutorial
python -m hvac_pid site --output outputs/tutorial --destination site
```

这会训练 Q-Learning、DQN、PPO、TD3、SAC、CrossQ，各 500 回合；运行 PG4PI-HVAC 的 500 条轨迹；生成 BO/FNN/LLM 记录并执行比较。LLM 没有合法 live 记录时，`all` 只保留 `not_run` 状态，不会发起隐式请求。缺少 CPU torch 时可明确跳过所有神经 RL：

```bash
python -m hvac_pid all --without-torch --output outputs/tutorial-numpy
```

这个别名等价于旧的 `--without-dqn`，会跳过 DQN、PPO、TD3、SAC、CrossQ；Q-Learning、BO、FNN、PG4PI 等不依赖 torch 的方法仍可运行。每种方法也可以单独运行，例如：

```bash
python -m hvac_pid ppo --episodes 500 --output outputs/tutorial
python -m hvac_pid td3 --episodes 500 --output outputs/tutorial
python -m hvac_pid sac --episodes 500 --output outputs/tutorial
python -m hvac_pid crossq --episodes 500 --output outputs/tutorial
python -m hvac_pid pg4pi --episodes 500 --output outputs/tutorial
```

每个命令的章节目录、`training.json`/`result.json`、曲线和解释记录由对应章节说明。统一的单步核对入口是：

```bash
python -m hvac_pid explain --method ppo --output outputs/tutorial
python -m hvac_pid explain --method td3 --output outputs/tutorial
python -m hvac_pid explain --method sac --output outputs/tutorial
python -m hvac_pid explain --method crossq --output outputs/tutorial
python -m hvac_pid explain --method pg4pi --output outputs/tutorial
```

`explain/<method>.json` 保存实际 helper 的中间量和一个单步/单批计算；它不是重新训练，也不会替换已保存的模型结果。

## 两个问题，两个协议

### 协议 A：冻结参数，只改变对象

先在标称对象 $(K=8,\tau=20,L=2)$ 上 commissioning，得到固定方法的增益和在线策略的权重/锚点。然后改变一个对象因素：

| case | 改变 |
|---|---|
| `nominal` | 默认对象 |
| `slower` | $\tau=30$ min |
| `long_delay` | $L=5$ min |
| `larger_load` | 第 60 分钟扰动从 1℃ 增到 1.5℃ |

Z-N、SIMC、BO、FNN、LLM、PG4PI-HVAC 使用标称 commissioning 得到的固定参数；Q-Learning、DQN、PPO、TD3、SAC、CrossQ 使用冻结权重，并按各自约定从标称 SIMC 锚点选择动作。测试时不重新辨识、不继续训练。

这一协议回答“原来的结果遇到对象变化还能不能用”，不是回答“遇到新对象重新调一次谁最好”。`protocol.json` 记录了每个 case 的对象和冻结规则。

每个 case 有三种图：

![所有方法的总览响应](/results/09-compare/nominal/response.svg)

`response.svg` 保留 12 条全量曲线，适合确认文件中确实有全部方法；为了读清楚，主要看以下分组：

![六种固定方法的标称响应](/results/09-compare/nominal/fixed.svg)

固定组包含 Z-N、SIMC、BO、FNN、LLM、PG4PI-HVAC。`fixed.svg` 的纵轴、扰动时刻和温度单位与全量图相同。

![六种在线策略的标称响应](/results/09-compare/nominal/online.svg)

在线组包含 SIMC 锚点以及 Q-Learning、DQN、PPO、TD3、SAC、CrossQ；下半图显示每 2 分钟选出的增益。长延迟和大负荷工况也使用相同的 fixed/online 分组。

<!--@include: ../generated/compare-nominal.md-->

<!--@include: ../generated/compare-long_delay.md-->

<!--@include: ../generated/compare-slower.md-->

<!--@include: ../generated/compare-larger_load.md-->

看图时先找控制输出离开 1 的时刻，再看温度是否在扰动后回到目标附近，最后对照 IAE、过冷和 movement。一个在线策略在长延迟下 IAE 更低，可能同时带来更大输出变化；固定策略在标称对象上更好，也不表示它对负荷变化更稳。

## 附带旧七方法的实际数字

仓库已有参考运行（旧代码提交）对前七种方法给出这些标称指标；新五种方法的完整参考记录要等对应训练产物生成后由表格导入，教程不预先写训练数字：

| 方法 | 模式 | IAE / ℃·min | 最大过冷 / ℃ | movement |
|---|---|---:|---:|---:|
| Z-N | 固定 | 49.3479 | 0.0178 | 1.7473 |
| SIMC | 固定 | 77.9950 | 0 | 1.3750 |
| BO | 固定 | 48.7413 | 0.0577 | 2.3376 |
| FNN | 固定 | 48.4676 | 0.0336 | 1.8116 |
| LLM | 固定 | 48.4865 | 0.0337 | 1.8772 |
| Q-Learning | 在线 | 72.0293 | 0.0297 | 5.1668 |
| DQN | 在线 | 69.6565 | 0 | 2.9924 |

这些数来自 `experiments/reference/` 的实际 CSV/JSON，运行环境为 Python 3.11.7、NumPy 2.4.6、SciPy 1.17.1、scikit-learn 1.9.1、CPU torch 2.14.0。它们只是一个种子和一个简单对象上的记录；新的 PPO、TD3、SAC、CrossQ、PG4PI-HVAC 需要阅读各自 `training.json`/`result.json` 后再解释，不把预期结果写成事实。

## 协议 B：十二个未见对象

第二个实验使用 `splits.json` 中从未参与训练或 ridge 选择的 12 个测试对象。它和协议 A 不同：

| 方法 | 新对象上的动作/参数来源 |
|---|---|
| Z-N、SIMC | 对新对象重新阶跃辨识并计算公式 |
| BO | 新对象重新做 5 个初始 + 15 个提案 |
| FNN | 新对象辨识后，用冻结网络一次预测 |
| Q/DQN/PPO/TD3/SAC/CrossQ | 新对象做 SIMC 锚点，权重冻结，在线选择动作 |
| PG4PI-HVAC | 在新对象上按固定轨迹预算重新训练，取最后一次更新后的增益 |
| LLM | `compare` 不隐式发起 12 次服务调用，因此没有 LLM holdout 结果 |

固定方法的“重新 commissioning”和在线策略的“冻结权重”必须分开记录。BO/PG4PI 的新对象试验成本算在 holdout 的 `tuning_simulations` 中；FNN 和 RL 不把测试 IAE 回传训练。生成表格：

<!--@include: ../generated/holdout.md-->

`holdout.svg` 用点和分布展示 12 个对象的 IAE；逐对象的数值仍在 `09-compare/holdout.json`，可检查最大、最小和异常对象。这里不把 12 个对象的平均数改写成普遍理论结论，先看 spread 和每种方法的模式。

## 统一指标和张量边界

所有方法最终都通过同一 `evaluate` 评分：输入是一个 `Scenario` 和固定/在线策略，输出 `Evaluation` 三个标量：IAE（℃·min）、undershoot（℃）、movement（无量纲输出变化量）。闭环内部每个物理步的温度、指令、Kp、Ki 是长度 1200 的序列；策略只在每个 2 分钟区间动作一次，默认是 60 次决策。

| 方法组 | commissioning 输入 | 在线策略输出 | 主要形状 |
|---|---|---|---|
| Z-N/SIMC | `(K,tau,delay)` 三标量 | 固定 `(Kp,Ki)` | 标量 → 2 标量 |
| BO | `(Kp,Ki)` 候选和标量 IAE | 固定最佳 `(Kp,Ki)` | `(N,2)` → `(N,)` |
| FNN | `(N,3)` 对象参数 | 固定 `(2,)` 增益 | `(N,27)@(27,2)` |
| Q/DQN | 每步 `(5,)` 观测 | 9 格动作 | Q 表或 `(5,)→(9,)` |
| PPO/TD3/SAC/CrossQ | 每步 `(5,)` 观测 | 二维连续增益动作 `[-1,1]` | 在线输入 `(5,)`，动作输出 `(2,)` |
| PG4PI-HVAC | 一条轨迹的对象与奖励 | 轨迹后冻结最后一次更新的 `(Kp,Ki)` | trajectory → scalar return |
| LLM | 消息列表和 5 个初始 trial | 工具参数 `(kp,ki,reason)` | JSON 对象 → 标量指标 |

RL 训练中的 reward 通常是两分钟区间 IAE 的负值，并使用 $\gamma=.99$ 的折扣目标；统一表格的完整回合 IAE 不折扣。PG4PI-HVAC 直接以整条轨迹的负 IAE 做训练，最终参数取最后一次更新后的增益，不能把它和所有 RL 的局部 TD target 混称为同一个优化目标。

## 成本：训练、整定和在线推理分开

<!--@include: ../generated/costs.md-->

表中 PG4PI 的 500 条轨迹是当前对象的一次整定预算，不能当作一个跨对象预训练模型。Z-N 和 SIMC 共享同一次 2400 步阶跃辨识，成本表应标明共享关系；不要给它们各算一遍物理试验后再做方法排名。

在线策略的调用量来自 `09-compare/frozen.csv`：固定方法是 0 次 AI 调用，FNN 只在闭环前推荐一次，在线 RL 默认每回合 60 次策略选择。CPU 推理基准单独运行：

```bash
python -m hvac_pid benchmark --output outputs/tutorial --samples 1000
```

`benchmark` 在当前机器上实际计时固定策略和已保存的在线模型，写入 `outputs/tutorial/benchmark/result.json`、`timings.csv`；随后 `site` 命令读取这些数据生成表；它不把训练秒数、网络服务延迟或物理传感器时间混在一起。没有 torch 时使用 `--without-torch` 只测可用的 NumPy 方法。参考 CPU 数字会随线程、Python 和模型版本变化，因此应报告自己的 benchmark 文件，而不是复制别人的毫秒数。

<!--@include: ../generated/benchmark.md-->

这些是当前 CPU、batch=1、单 PyTorch 线程、100 次预热后 1000 次调用的实测值。基准在五种子训练结束后单独运行；训练耗时则保留实际运行期间的墙钟时间，部分种子曾并行运行，会受机器负载影响。模型文件大小不包含解释器、PyTorch 和临时张量。

## 练习：改变一个预算，再做重复运行

先只改变 RL/PG4PI 轨迹预算，保持种子和对象协议：

```bash
python -m hvac_pid all --episodes 100 --rounds 15 --output outputs/tutorial-budget100
python -m hvac_pid explain --method all --output outputs/tutorial-budget100
```

这里只改变 `episodes`，BO/LLM 的 proposal rounds 保持 15。验证 `training.json`/`result.json` 中的 episodes、proposal rounds、仿真次数与命令一致，再检查 frozen/holdout 的图是否重新生成。不要用新预算得到的测试结果去改训练超参数描述。

若要看随机性，运行五个种子：

```bash
python -m hvac_pid repeat --seeds 0 1 2 3 4 --output outputs/repeated
```

<!--@include: ../generated/repeats.md-->

![五种子名义工况，每个点代表一个独立种子](/results/repeats/nominal.svg)

本次实际完成 seed 0–4：六种在线方法每个种子各训练 500 回合，PG4PI 在名义对象和每个新对象上各使用 500 条轨迹。上表包含 220 条冻结工况评估；另外保存了 660 条未见对象评估。LLM 的既有真实调用只出现在单次比较，不进入五种子均值。

先读两个容易被单次成绩掩盖的事实。PG4PI 在 seed 0 的名义工况 IAE 为 62.08，但五种子均值为 93.86，劣于 SIMC 的 77.99；CrossQ 的均值为 67.11，但标准差达到 20.69。BO 和 FNN 在这个简单对象上反而有更低且波动较小的 IAE。这里没有把网络方法的失败换成经典控制结果。

未见对象必须另读：每个种子先对 12 个测试对象求平均，再跨五种子比较，PG4PI 的平均 IAE 为 259.77，种子间标准差 255.95；DQN 为 62.97±4.00，SIMC 为 75.21。PG4PI 的低增益失败轨迹在[PI 策略梯度章节](./14-pg4pi#五个种子让失败显现出来)逐项分析。测试结果没有用于改学习率、选训练轮数或重新挑种子。

复算数据：[逐种子冻结结果](/results/repeats/observations.json)、[逐对象测试结果](/results/repeats/holdout.json)、[各种子成本与配置](/results/repeats/runs.json)、[未见对象汇总](/results/repeats/holdout-summary.json)。经典方法没有随机训练，因此五次相同值的标准差为 0，不代表做过五次有噪声的物理辨识。

这个命令会为每个种子重建训练和 BO 标签，报告均值、样本标准差和范围；它不是把同一个模型评估五遍。生成的 `downloads.md` 列出每章 JSON、CSV 和 SVG 的可下载路径：

<details>
<summary>展开全部实验数据、模型与图表下载</summary>

<!--@include: ../generated/downloads.md-->

</details>

## 解释边界

这套比较能验证代码路径、预算、单位和对象变化下的行为。它不能把一维温度模型的结果直接换算成电费，也不能证明某个方法在真实建筑、其他传感器采样率或更长延迟下稳定。图中出现的优势和失败应回到对应 `trace`、动作记录和实际产物解释；没有生成的训练结果就留在“待运行”，不补一个好看的数字。
