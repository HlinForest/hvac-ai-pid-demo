# 动手做 AI 自动整定

你已经知道 Kp、Ki 会改变控制效果。接下来，把“凭经验调参数”拆成几个可以运行的小实验：让程序搜索，让模型从案例中学习，让策略在运行中调整，再让大模型读反馈提出下一次建议。

整个实验课只用一个温度状态。对象有惯性、有延迟，也会突然增加热负荷。每种方法都调用同一套 PI 和仿真程序，因此图上的差异能追溯到参数选择。

![同一对象上七种已实际运行方法的温度与输出曲线](/results/09-compare/nominal/response.svg)

这张图来自仓库附带的默认实验，包含 DeepSeek Flash 十五轮真实整定后得到的参数。先观察不同方法何时退出满功率制冷，再看 60 分钟后的负荷变化。LLM 在标称对象上只带来了小幅改善；换工况后是否保留这点收益，要看后面的独立测试。

## 从哪里开始

如果你想先看到结果，从[跑通温控](./chapters/01-temperature)开始，几秒钟后就能打开第一张曲线。随后用[阶跃实验](./chapters/02-plant)认识对象，用[经典整定](./chapters/03-classical)建立对照。

| 想探索的问题 | 对应实验 |
|---|---|
| 少做几次试验，能否找到好参数？ | [BO](./chapters/04-bo) |
| 整定案例能否变成可复用的经验？ | [FNN](./chapters/05-fnn) |
| 工况变化时，能否边控制边调参数？ | [Q-Learning](./chapters/06-qlearning) |
| 用网络代替 Q 表后会怎样？ | [DQN](./chapters/07-dqn) |
| 大模型读懂反馈后，下一次会怎么试？ | [LLM](./chapters/08-llm) |
| 这些方法的收益与成本分别是什么？ | [统一比较](./chapters/09-comparison) |

## 运行一次，再改一个量

本课面向会使用 Python、已经了解 PID 的读者。每章先运行实验，然后把公式中的量与代码、记录和图对起来。你可以只做某一章；涉及训练的章节会说明前置步骤。

基础实验使用 NumPy、SciPy、scikit-learn 和 Matplotlib。DQN 额外使用 CPU 版 PyTorch，LLM 额外需要一个支持工具调用的接口。安装方法见[第一章](./chapters/01-temperature#安装与第一次运行)。

教程的组织方式参考 [Hands-on Modern RL 的价值迭代与 Q-Learning 实验](https://walkinglabs.github.io/hands-on-modern-rl/chapter03_mdp/value-experiment)：先把一个小实验跑通，再用具体计算解释曲线是怎样产生的。
