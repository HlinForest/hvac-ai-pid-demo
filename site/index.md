# 动手做 AI 自动整定

这是一套可以反复运行的小型温控实验。对象只有一个温度状态，却保留了三件会让整定变难的事：20 分钟惯性、2 分钟输入延迟，以及第 60 分钟开始的等效热负荷。所有方法最终都把 Kp、Ki 交给同一个条件积分 PI，输出都在 0 到 1 之间；因此你可以把一条曲线追溯到参数、观测、动作和实际仿真结果。

![同一对象上的方法响应总览](/results/09-compare/nominal/response.svg)

图中的总览适合看差异，章节中的 fixed/online 分组更适合读温度和输出细节。IAE 的单位是 ℃·min，movement 是输出变化量，不是电耗。模型里的温度、负荷和指令也不对应某个真实建筑的标定值。

## 你会做什么

从左到右，实验把“调两个数”拆成几个不同的问题：

| 问题 | 章节 |
|---|---|
| PI 的第一步输出和饱和怎样算？ | [01 跑通温控](./chapters/01-temperature) |
| 对象的增益、惯性、延迟怎样测？ | [02 看懂对象](./chapters/02-plant) |
| Z-N 和 SIMC 怎样从辨识值给出基线？ | [03 经典整定](./chapters/03-classical) |
| 有限仿真预算怎样选择下一点？ | [04 BO](./chapters/04-bo) |
| 许多整定案例能否一次预测增益？ | [05 FNN](./chapters/05-fnn) |
| 每两分钟改变一次离散增益会怎样？ | [06 Q-Learning](./chapters/06-qlearning) |
| 用网络估计九个离散动作会怎样？ | [07 DQN](./chapters/07-dqn) |
| 连续动作的 clipped policy gradient 如何处理延迟？ | [10 PPO](./chapters/10-ppo) |
| 双 critic 和延迟 actor 更新如何工作？ | [11 TD3](./chapters/11-td3) |
| 熵正则怎样让连续增益保持探索？ | [12 SAC](./chapters/12-sac) |
| 没有 target critic 时如何用联合 Batch Renorm？ | [13 CrossQ](./chapters/13-crossq) |
| PI 本身作为两个策略参数如何更新？ | [14 PG4PI-HVAC](./chapters/14-pg4pi) |
| 模型读完反馈会提出什么下一次试验？ | [08 LLM](./chapters/08-llm) |
| 工况变化、新对象和 CPU 成本怎样比较？ | [09 统一比较](./chapters/09-comparison) |

网站导航会把新连续 RL 章节放在 LLM 与统一比较之前；文件名保留已有 01–09 URL，所以旧链接仍可打开。

## 先跑一行 PI

在仓库根目录准备环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m hvac_pid temperature
```

打开 `outputs/tutorial/01-temperature/response.svg`。不安装 torch、不配置模型服务，也能完成前七章中的 BO、FNN、Q-Learning 和经典方法。第一章会从 `Manual.csv` 的真实第一行核对：$e_0=4\,^{\circ}\mathrm{C}$、$K_p=0.15$、$K_i=0.005$、$\Delta t=0.1$ min 时，输出是 `0.602`。

## 方法怎样接在同一个控制循环上

内层物理循环永远以 0.1 分钟为步长：

```text
温度 T(k) → e(k)=T(k)-24 → PI(Kp,Ki) → clip 到 [0,1] → 延迟对象 → T(k+1)
```

固定参数方法在循环开始前只推荐一次 `(Kp,Ki)`。Q-Learning、DQN、PPO、TD3、SAC、CrossQ 每 2 分钟从观测

$$
(e,\dot e,u,s_p,s_i)\in\mathbb R^5
$$

选择下一组相对 SIMC 锚点的增益；PI 仍每 0.1 分钟执行。PG4PI 在整条轨迹内固定两个参数，结束后才更新。LLM 在仿真循环外通过工具提案，每次提案也保持一组固定增益。

## 每章都可以核对一件真实中间量

```bash
python -m hvac_pid explain --method classical --output outputs/tutorial
python -m hvac_pid explain --method bo --output outputs/tutorial
python -m hvac_pid explain --method qlearning --output outputs/tutorial
python -m hvac_pid explain --method dqn --output outputs/tutorial
```

对已经完成训练的连续方法：

```bash
python -m hvac_pid explain --method ppo --output outputs/tutorial
python -m hvac_pid explain --method td3 --output outputs/tutorial
python -m hvac_pid explain --method sac --output outputs/tutorial
python -m hvac_pid explain --method crossq --output outputs/tutorial
python -m hvac_pid explain --method pg4pi --output outputs/tutorial
```

这些 JSON 来自和训练/整定共用的函数：BO 的第一个 EI 候选、Q 的一次 TD 更新、DQN 的首个 batch、连续 RL 的首个 critic/actor 或 PPO loss，以及 PG4PI 的首条轨迹。它们是查公式和张量形状的入口，不是另一个评分体系。

## 需要 torch 时再安装

连续 RL 和 DQN 使用 CPU 版 PyTorch：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m hvac_pid all --episodes 500 --rounds 15 --output outputs/tutorial
python -m hvac_pid site --output outputs/tutorial --destination site
```

如果只想跑 NumPy 方法：

```bash
python -m hvac_pid all --without-torch --output outputs/tutorial-numpy
```

这会明确跳过所有神经 RL，不会把缺失的结果写成 0 分或用 SIMC 冒充。

## 先看哪些图

完成整套实验后，打开：

- `/results/09-compare/nominal/fixed.svg`：Z-N、SIMC、BO、FNN、LLM、PG4PI-HVAC；
- `/results/09-compare/nominal/online.svg`：SIMC 锚点和六种在线策略；
- `/results/09-compare/nominal/response.svg`：12 种方法总览；
- `/results/09-compare/holdout.svg`：12 个未见对象的 IAE 分布。

不要只看一条下降的训练曲线。固定方法要查看实际响应 CSV；在线方法要把动作记录和温度响应对齐；RL 的训练 target 带折扣；history.return 是探索下的不折扣负 IAE，critic loss、entropy 和最终冻结 IAE 也应分别阅读。比较章节把这些量分开。

## 运行 CPU 推理基准

```bash
python -m hvac_pid benchmark --output outputs/tutorial --samples 1000
```

基准使用 batch=1、100 次 warm-up，报告控制更新、FNN 一次推荐和冻结策略的 CPU 中位数/P95、参数量和 checkpoint 文件大小。文件大小不等于 Python/PyTorch 运行时内存，也不包含线程库、解释器或安全联锁；教程不据此承诺某个 MCU 可以部署。

最后阅读 [统一比较](./chapters/09-comparison)，再决定要给某个方法更多试验、更多回合，还是换一个延迟和负荷。每次只改一个预算或对象参数，保留原产物作对照。
