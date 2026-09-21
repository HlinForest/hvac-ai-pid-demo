# 03 经典整定：先做一个能手算的基线

现在已经有一阶加纯延迟的辨识结果。本章把 $(K,\tau,L)$ 直接代入两条经典 PI 规则：Ziegler–Nichols 反应曲线规则和 SIMC 规则。它们只在闭环开始前给出一组固定的 $(K_p,K_i)$，不搜索，也不训练。后面的 AI 方法要回答的是“在相同对象和预算下，是否值得承担额外成本”。

## 跑一次并查看输入输出

```bash
python -m hvac_pid classical
```

默认产物位于 `outputs/tutorial/03-classical/`：

| 文件 | 内容 |
|---|---|
| `gains.json` | 辨识值和 Z-N、SIMC 的 Kp/Ki |
| `ZN.csv`, `SIMC.csv` | 各自 120 分钟的温度、输出和固定增益 |
| `metrics.json` | 两组参数的 IAE、过冷、movement |
| `lambda.json` | λ=2、7、20 min 的扫描结果 |
| `response.svg`, `lambda.svg` | 对照曲线 |

也可以生成解释用的单次计算记录：

```bash
python -m hvac_pid explain --method classical --output outputs/tutorial
```

它写入 `outputs/tutorial/explain/classical.json`，包含辨识输入、公式中间量和最终增益。命令的 JSON 是核对手算的入口；本章的图表仍来自 `03-classical/`。

## Z-N：从三个对象参数得到两个增益

项目使用反应曲线 PI 公式

$$
K_p=\frac{0.9\tau}{KL},
\qquad
T_i=3.33L,
\qquad
K_i=\frac{K_p}{T_i}.
$$

它接收：

| 输入 | 形状 | 单位 |
|---|---|---|
| `identified.gain` | 标量 | ℃ |
| `identified.tau` | 标量 | min |
| `identified.delay` | 标量 | min |

输出是 `Gains(kp, ki)` 两个标量，其中 Kp 的单位是 ℃^-1，Ki 的单位是 (℃·min)^-1。这里 $K$ 取制冷增益的正幅值，控制方向已经由 $e=T-T_{set}$ 决定，所以不需要再给 Kp 加负号。

附带参考记录的辨识值可以读作 $K=8,\tau=20,L=2$。代入得到

$$
K_p=\frac{0.9\times20}{8\times2}=1.125,
\qquad
T_i=3.33\times2=6.66\ \mathrm{min},
$$

$$
K_i=\frac{1.125}{6.66}=0.1689189189\ \bigl({}^{\circ}\mathrm{C}\!\cdot\!\mathrm{min}\bigr)^{-1}.
$$

`gains.json` 中的浮点表示为 `Kp=1.1250000000000353`、`Ki=0.16891891891892927`，差异来自拟合值不是手写的整数。初始误差 4℃ 会让比例项单独达到 4.5，因此 PI 的输出第一步立刻被裁剪到 1；这不是公式出错，而是该规则相对于 0 到 1 执行器范围很激进。

## SIMC：用 λ 明确选择响应尺度

项目中的 SIMC 简化规则是

$$
K_p=\frac{\tau}{K(\lambda+L)},
\qquad
T_i=\min\left[\tau,4(\lambda+L)\right],
\qquad
K_i=\frac{K_p}{T_i}.
$$

默认 `lambda=None` 时，代码取 $\lambda=\tau/3$。对默认对象，$\lambda=6.6666667$ min，于是

$$
K_p=\frac{20}{8(6.6667+2)}=0.2884615385,
$$

$$
T_i=\min(20,4(6.6667+2))=20\ \mathrm{min},
\qquad
K_i=0.01442307692.
$$

SIMC 的 Kp 和 Ki 都明显小于 Z-N，因此第一步同样会受输出上限影响，但积分积累慢得多。默认参考运行的 SIMC IAE 是 `77.9950 ℃·min`，movement 是 `1.3750`；Z-N 的 IAE 是 `49.3479 ℃·min`，movement 是 `1.7473`，最大过冷为 `0.0178℃`。这个对象和 120 分钟窗口下，Z-N 更快，SIMC 更平缓；这不是对两条规则在所有对象上的排序证明。

<<< @/../hvac_pid/tuning.py#classical

片段中 `zn` 和 `simc` 都只返回 `Gains`，没有调用学习器。经典方法的“推理”就是几次标量乘除法；闭环中的 1200 次 PI 更新仍然由 `core.py` 的同一个 `PI.update` 完成。

## λ 扫描为什么要看三张曲线

`classical` 命令额外运行 λ=2、7、20 min：

| λ | IAE / ℃·min | 最大过冷 / ℃ | movement |
|---:|---:|---:|---:|
| 2 | 60.8467 | 0 | 1.4000 |
| 7 | 78.5282 | 0 | 1.3750 |
| 20 | 148.6361 | 0 | 0.8682 |

这些数来自附带的 `lambda.json`。λ 变大时，SIMC 让闭环更保守，输出变化减少，但在当前 120 分钟窗口留下更长的温度误差。λ 变小并不意味着可以无限追求速度：延迟是固定的，过小的 λ 可能把控制器推到饱和并放大模型误差。

![Z-N 与 SIMC 的温控响应](/results/03-classical/response.svg)

<!--@include: ../generated/gains.md-->

![三个 λ 对应的温控响应](/results/03-classical/lambda.svg)

<!--@include: ../generated/03-classical.md-->

读曲线时先看 0 到 2 分钟的延迟，再看输出离开 1 的时间，最后看 60 分钟扰动后的恢复。只看终点温度会漏掉大部分 IAE；只看 IAE 又会漏掉输出变化和过冷。

## 一个可重复的失败方式：对象变慢

把延迟改成 5 分钟并重新运行：

```bash
python -m hvac_pid classical --delay 5 --output outputs/classical-delay5
```

这条命令先重新做阶跃辨识，再用新 $L$ 算 Z-N/SIMC。若你拿默认 `gains.json` 的参数直接放到 `temperature --delay 5`，回答的是“旧参数遇到变长延迟会怎样”；若运行 `classical --delay 5`，回答的是“重新测量后经典规则会怎样”。两条实验都合理，但不能把它们的数字混成同一结论。

还可以直接观察 Z-N 公式的边界：当 $L$ 接近 0 时，$K_p$ 和 $K_i$ 会迅速变大；实现会拒绝 `delay<=1e-8` 的 Z-N 计算。SIMC 仍需要正 λ，但它不使用 $1/L$ 的同样奇异项。代码的异常和边界是方法行为的一部分，不能在教程里悄悄加一个裁剪值。

## 动手：只改 λ，并验证中间量

经典命令的 λ 扫描已固定为 2、7、20。练习是用 `explain` 的 JSON 验证 λ=2 的一个手算中间量：

```bash
python -m hvac_pid explain --method classical --output outputs/classical-explain
```

读取 `outputs/classical-explain/explain/classical.json`，找到默认辨识值和 SIMC 的 `lambda`。先计算 $K_p=20/[8(2+2)]=0.625$，再与你用项目 API 或临时脚本计算的 λ=2 参数比较；不要把命令默认产生的 λ=6.6667 结果误当成 λ=2。曲线和 `lambda.json` 则验证它确实使用了 λ=2、7、20 三个输入。

如果你要改变扫描集合，需要改实验代码；本章的最小练习只要求用现有输出核对公式和单位。下一章让每次完整仿真反过来影响下一次参数提案。

## 放到在线控制前后

经典规则的在线部分只有两项：上线前做一次 240 分钟阶跃辨识，再做一次几次乘除法得到固定 $(K_p,K_i)$。闭环在线阶段每 0.1 分钟只运行 PI，输入形状是一个标量温度，输出形状是一个标量指令；没有每周期 BO 或模型推理。参考机的整定计算不到秒级，真正的成本是受控阶跃占用的物理时间和扰动风险。

经典参数适合作为后面方法的锚点和 sanity check。它没有从历史案例泛化，也没有观察过程中改变参数；下一章用相同评分函数、相同 0 到 1 边界和相同对象，试试有限预算的贝叶斯优化。
