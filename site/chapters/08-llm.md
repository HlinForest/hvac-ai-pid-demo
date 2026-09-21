# 13 LLM：读反馈，再试一组固定参数

本章把整定器换成一个兼容 Chat Completions 工具调用的语言模型。模型每一轮读取已完成试验，调用一次 `evaluate_gains` 提出下一组固定 Kp、Ki；工具用项目自己的仿真器评分，再把结果放回下一轮消息。LLM 不在 0.1 分钟循环里控制制冷，也不直接写温度或执行器。

## 先用无调用模式检查流程

下面的命令不会发起模型请求，只创建明确的 `not_run` 记录：

```bash
python -m hvac_pid llm --output outputs/tutorial
```

要进行真实调用，必须显式加 `--live`，并把密钥放在环境变量中：

```powershell
$env:LLM_BASE_URL = "https://api.deepseek.com"
$env:LLM_MODEL = "deepseek-flash"
$env:DEEPSEEK_API_KEY = "你的密钥"
python -m hvac_pid llm --live --key-env DEEPSEEK_API_KEY --reasoning-effort none
```

这里的占位符只说明变量位置；不要把真实密钥写进命令历史、Markdown、`result.json` 或截图。程序只保存请求消息、响应、仿真结果和接口返回的 usage，不保存 Authorization 请求头。也可以用 `--base-url`、`--model`、`--key-env` 接入兼容接口；项目不会自动从 `.env` 读取密钥。

真实运行和解释命令：

```bash
python -m hvac_pid explain --method llm --output outputs/tutorial
```

`--live` 最多默认 15 轮模型提案；不加 `--live` 不会用规则或 SIMC 假装完成。

## 一轮请求的输入和输出

模型看到的是一个消息列表，不是张量。第一轮消息包含：

| 字段 | 形状/类型 | 单位或含义 |
|---|---|---|
| `identified_model` | 对象 3 个标量 | gain/℃、tau/min、delay/min |
| `experiment` | 对象 7 个标量 | 环境、目标、采样和扰动配置 |
| `initial_trials` | 长度 5 的数组 | Z-N、SIMC 和 3 个随机样本的完整指标 |
| `proposal_budget` | 标量整数 | 剩余模型提案数 |

工具参数是一个对象 `{kp: number, ki: number, reason: string}`，Kp 单位是 ℃^-1，Ki 是 (℃·min)^-1。参数边界为 Kp `[0.01,3]`、Ki `[0.0001,0.5]`。工具返回一个 trial 对象，包含标量 IAE（℃·min）、过冷、movement、状态和试验编号。

消息历史会逐轮增长：模型的 tool call 和工具反馈被追加到列表中。如果接口没有返回带 id 的标准 tool call，代码会退回把反馈作为 user 消息继续；它不会把失败提案改写成成功提案。每轮只允许一个 `evaluate_gains` 调用。

## 实际参考运行的前三轮

附带 `experiments/reference/08-llm/result.json` 是一次 15 轮真实记录。前五次初始仿真与 BO 完全相同。模型的前三个建议和工具返回是：

| 轮次 | Kp | Ki | IAE / ℃·min | 发生了什么 |
|---:|---:|---:|---:|---|
| 1 | 1.08 | 0.20 | 48.6172 | 在当前最好点 1.034/0.238 和 Z-N 附近取中间值 |
| 2 | 1.15 | 0.22 | 48.7495 | 更大增益反而变差 |
| 3 | 1.05 | 0.19 | 48.5851 | 读到变差后向较小增益移动 |

这些数字来自工具实际运行的 120 分钟仿真，不是模型自己写进理由的分数。第二轮的 IAE 比第一轮高，程序没有替换成 SIMC，也没有隐藏这条记录；它把 `48.7495373970` 反馈给下一轮。

## 历史最佳和曲线

![DeepSeek Flash 十五轮真实提案及历史最佳值](/results/08-llm/search.svg)

初始五次的最佳 IAE 是 `48.7412801021`。第 15 轮建议 `Kp=1.02, Ki=0.192`，实际 IAE 是 `48.4864992382`，相对初始最佳约降低 0.52%。附带最佳响应的最大过冷是 `0.0336830℃`，movement 是 `1.8771968`。

![DeepSeek 最佳参数对应的温度与输出](/results/08-llm/response.svg)

<!--@include: ../generated/llm-metrics.md-->

<!--@include: ../generated/llm-status.md-->

这次 15 次调用的记录用量合计为输入 47,059 tokens、输出 1,730 tokens，其中缓存命中字段合计 42,880；这些是接口返回的统计，不在教程里据此推算费用。附带运行耗时约 `89.1399 s`，机器、网络和模型服务变化后不能复用这个时间。

## 实验循环的代码边界

LLM 只负责提案，三个角色各有清晰边界：

| 角色 | 责任 |
|---|---|
| LLM | 从历史和约束中提出 Kp、Ki，并给出理由 |
| `evaluate_gains` | 用固定参数跑完整仿真，返回统一指标 |
| 实验循环 | 校验参数、保存 exchanges、计数预算、选实际已测最佳 |

代码先通过 `Gains` 检查有限且非负，再通过 proposal bounds 检查范围。它不会静默裁剪越界值。模型返回两个 tool call、错误函数名、缺少 reason、非数字参数或越界参数时，本轮标记为 `invalid`，消耗提案轮数，但不增加仿真次数。

若网络请求抛出异常，状态变为 `incomplete`，已经完成的初始和前几轮 trial 保留，当前请求及错误类型写进 `exchanges`。没有自动重试、供应商切换或用经典规则填补缺失轮次。完成 15 轮后，程序从所有 `status=evaluated` 的 trial 中选择实际 IAE 最低的一项；最佳项可以是初始样本。

<!--@include: ../generated/llm-trials.md-->

## 失败方式：理由听起来对，实验结果却更差

语言模型的理由不是评分函数。参考第二轮给出“再提高一点”的理由，但真实 IAE 变差；工具返回的数字才改变下一轮上下文。模型也可能反复试很接近的参数，或在有限预算内没有超过初始样本。曲线中历史最佳折线只会下降，不能把它当作每一轮试验都改善。

另一个边界是上下文和成本：每轮消息包含更长的历史，因此输入 token 会增长；工具仿真仍要占 CPU。模型没有看到真实 $K,\tau,L$，只看到阶跃辨识结果；若辨识错误，模型的搜索会在错误的对象描述上进行。它也只在固定标称对象上试验，不会自动学习延迟变化或跨设备的稳定性。

## 练习：缩短提案预算并核对一次反馈

先跑一个不需要服务的结构检查：

```bash
python -m hvac_pid llm --rounds 3 --output outputs/llm-not-live
```

它应生成 `status=not_run`，`cost.calls=0`，而不是生成模型结果。若已经有合法的服务配置，再显式运行：

```bash
python -m hvac_pid llm --live --rounds 3 --key-env DEEPSEEK_API_KEY --reasoning-effort none --output outputs/llm-three
python -m hvac_pid explain --method llm --output outputs/llm-three
```

验证五个初始 trial 加三轮模型 trial 是否共 8 次仿真；读 `result.json` 的任一轮：

1. request 中是否有上一轮完整结果；
2. response 是否恰好有一个 `evaluate_gains` 调用；
3. trial 的 IAE 是否来自仿真返回，而非 reason；
4. 下一轮消息是否包含该 trial；
5. 任何越界或服务错误是否只增加记录而不伪造仿真。

练习可能产生真实接口费用，预算和服务配置由操作者决定；教程不会在 `compare` 或 `all` 中隐式发起调用。

## 在线位置和 CPU 成本

LLM 适合离线 commissioning 或低频人工监督的整定建议。每轮模型请求和完整仿真都发生在控制闭环外，默认 15 次提案加 5 次初始仿真，共 20 次 1200 步闭环模拟；参考 `result.json` 的本地仿真部分约为 24,000 个物理步。真正的在线控制仍是 PI 每 0.1 分钟执行一次，固定 Kp、Ki 不需要调用模型。

因此 LLM 的主要成本是外部服务延迟、token 和网络可用性，而不是部署一个神经网络到控制器上。密钥管理、网络隔离和审批属于服务接入工程；本项目只保留兼容接口记录，不把它描述成设备端部署。下一章把所有方法放在统一的冻结工况和新对象测试中，分开观察收益、失败和成本。
