# 01 跑通温控：先改一组参数

这一章只做一件事：让温度从 28℃ 降到 24℃，观察 PI 在什么时刻加大制冷、什么时刻收回输出。先不用 AI，后面的算法都要控制同一个对象。

## 安装与第一次运行

在仓库根目录执行。Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m hvac_pid temperature
```

Linux 或 macOS 使用 `source .venv/bin/activate` 激活环境，其余命令相同。

打开 `outputs/tutorial/01-temperature/response.svg`。同时生成的 `Manual.csv` 保存每一个采样时刻的温度、控制输出与参数，`metrics.json` 保存本次指标。

![手工参数与激进参数的温控响应](/results/01-temperature/response.svg)

先读上半图。虚线是 24℃ 目标；温度越高，控制器越想增加制冷。再读下半图，输出 1 表示满功率指令。即使指令已经拉满，温度也不会立即下降：对象需要时间响应，而且输入有两分钟延迟。

第 60 分钟出现持续热负荷。留意它先改变温度变化趋势，随后 PI 才通过误差调整输出。这里没有把已知扰动直接前馈给控制器。

## 只保留一套 PI

制冷系统使用误差 $e=T-T_{\mathrm{set}}$。温度高于目标时误差为正，控制器应增加输出。离散更新为：

$$
I_{k+1}=I_k+K_i e_k\Delta t,\qquad
u_k=\operatorname{clip}(K_p e_k+I_{k+1},0,1).
$$

初始误差是 4℃。手工参数为 $K_p=0.15$、$K_i=0.005$，采样间隔是 0.1 分钟。第一次积分增加 $0.005\times4\times0.1=0.002$，因此第一次输出是 $0.15\times4+0.002=0.602$。你可以在 CSV 第一行找到它。

实际代码还处理输出饱和：当输出已经达到上限，而且误差还在推动它继续增加时，暂停积分。误差反向、能够把输出拉回范围时，再允许积分。

<<< @/../hvac_pid/core.py#pi

积分量保存的是输出贡献。后面 RL 修改 Ki 时，已积累的贡献保持不变，新的 Ki 只影响接下来的积分增量。

## 用一个容易核对的指标

本课用绝对误差积分 IAE 作为主要目标：

$$
\mathrm{IAE}=\sum_{k=0}^{N-1}|T_k-T_{\mathrm{set}}|\Delta t.
$$

这里用每个时间区间开始时的温度求和，单位为 ℃·min。一个持续 10 分钟的 1℃ 误差贡献 10；一个持续 5 分钟的 2℃ 误差也贡献 10。为了看出这两个过程的区别，我们同时查看曲线、最大过冷和输出总变化量。

<!--@include: ../generated/01-temperature.md-->

输出总变化量从初始指令 0 开始累计 $\sum|u_k-u_{k-1}|$。它描述指令变化的频繁程度，不代表电耗。后面如果某方法 IAE 降低、输出却频繁跳动，这两个事实会一起保留。

## 动手：只增加 Kp

保持 Ki 不变，把 Kp 从 0.15 改为 0.6：

```bash
python -m hvac_pid temperature --kp 0.6 --ki 0.005 --output outputs/larger-kp
```

先预测第一步输出，再运行命令。检查满功率制冷持续了多久，温度是否更早接近目标。然后把 Ki 改为 0.03，使用另一个输出目录。比较这两次变化分别改善了哪一段响应。

你已经能运行闭环并修改参数。下一章暂时断开 PI，用一次固定制冷指令的阶跃，测量对象本身的响应。
