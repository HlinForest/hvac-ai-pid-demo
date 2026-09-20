# 03 经典整定：给 AI 一组清楚的对照

先用阶跃辨识得到 $K,\tau,L$，再分别计算 Z-N 与 SIMC 参数。这一步不用搜索，也不用训练，计算过程可以直接手算。

```bash
python -m hvac_pid classical
```

结果保存在 `03-classical/`。先看 `gains.json`，再看温度曲线。

<!--@include: ../generated/gains.md-->

## Z-N：代入一次反应曲线公式

本课使用 Ziegler–Nichols 反应曲线 PI 规则：

$$
K_p=\frac{0.9\tau}{KL},\qquad T_i=3.33L,\qquad K_i=\frac{K_p}{T_i}.
$$

用 $K=8$、$\tau=20$、$L=2$ 代入，得到 $K_p=1.125$、$T_i=6.66$ 分钟、$K_i\approx0.1689$。这些是计算结果，程序不会因为参数看起来大就把它们改小。

Z-N 使用的过程增益是制冷增益的正幅值。控制方向已经通过 $e=T-T_{\mathrm{set}}$ 处理，不要再给 Kp 加负号。

## SIMC：用 λ 表达期望速度

这里使用 Skogestad 简化 IMC（SIMC）PI 规则：

$$
K_p=\frac{\tau}{K(\lambda+L)},\quad
T_i=\min[\tau,4(\lambda+L)],\quad K_i=K_p/T_i.
$$

默认选择 $\lambda=\tau/3$，得到 $K_p\approx0.2885$、$K_i\approx0.01442$。λ 是你选择的响应尺度。它变大时，参数通常更温和；它不是模型训练出来的隐藏常数。

<<< @/../hvac_pid/tuning.py#classical

公式出处见 [Skogestad 的整定规则论文](https://skoge.folk.ntnu.no/publications/2003/tuningPID/README.html)。本课保留 SIMC 名称，避免把不同 IMC 变体混写成同一个公式。

## 对照温度与输出

![Z-N 与 SIMC 的温控响应](/results/03-classical/response.svg)

<!--@include: ../generated/03-classical.md-->

默认工况里，Z-N 的 IAE 更低。观察它何时退出满功率制冷，以及目标附近的输出变化。SIMC 的响应更缓和，降低输出变化量的同时，也留下较长时间的温度误差。哪种取舍更合适，需要结合目标来看；后面统一用 IAE 优化，同时列出输出变化。

## 动手：λ 扫描

运行脚本还会自动比较 λ 为 2、7、20 分钟的结果：

![三个 λ 对应的温控响应](/results/03-classical/lambda.svg)

先找最小 λ 的曲线。它是否更早接近目标？增加延迟后，这组参数是否仍然合适？运行：

```bash
python -m hvac_pid classical --delay 5 --output outputs/classical-delay-five
```

这次会对新对象重新辨识并整定。第九章还有另一种实验：保持旧参数不变，只改变对象，用来观察原整定结果能否继续适用。两者回答不同问题。

下一章把公式得到的两组参数交给 BO，作为它已经看过的初始实验。
