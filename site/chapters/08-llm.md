# 13 LLM：用工具调用提出下一组固定 PI 增益

本章的语言模型只负责提出候选。
它读取已经完成的整定试验，通过兼容 Chat Completions 的工具调用提交 Kp、Ki 和一段理由。
项目自己的 evaluate_gains 工具运行完整冷却仿真并返回指标，实验循环再把真实结果放回下一轮消息。

这条链路和 PI 的控制时钟分开。
LLM 不在 0.1 min 温度循环里运行，不直接写温度或执行器，也没有权限跳过项目的参数校验。
一次 live 运行结束后，只把实际已测的最佳增益交给固定 PI。

仓库中的参考结果是一次已经保存的 15 轮记录。
本地默认入口只生成明确的 not_run 状态，不需要外部服务。
本文不发起任何 API 调用；外部服务的命令只作为接口边界和结果复现入口保留。

## LLM 改变的仍是两个 PI 增益

模型提出的工具参数是 Kp、Ki 两个正数。
Kp 的单位是 ℃⁻¹，Ki 的单位是 (℃·min)⁻¹。
固定到控制器后，每个物理步仍执行：

$$
I^\prime=I+K_i e\Delta t,\qquad
u=\operatorname{clip}_{[0,1]}(K_p e+I).
$$

PI 的条件积分规则、对象的一个温度状态、2 min 输入延迟和 1200 步评价窗口都不由模型修改。
模型只看完成一轮试验后的标量反馈，不能在试验内部干预温度轨迹。

每轮模型提案的边界为：

$$
0.01\le K_p\le3,\qquad
0.0001\le K_i\le0.5.
$$

边界在 system 消息中说明一次，并在 Python 侧用 in_bounds() 再检查一次。
越界候选不会被静默裁剪到最近边界。

## 第一条请求的消息结构

ChatClient.__call__ 构造一个 JSON 请求。
它包含 model、messages、一个工具定义、强制的 tool_choice，以及 temperature=0。
若调用参数带 reasoning_effort，代码也把它写入请求。
URL 是 base_url 去掉末尾斜杠后拼接 /chat/completions。

第一轮 messages 有两条：

| 消息 | 内容 | 形状或单位 |
| --- | --- | --- |
| system | 控制方向、目标、单位、边界、每轮一个工具调用的约束 | 字符串 |
| user | identified_model、experiment、initial_trials、proposal_budget | JSON 对象 |

identified_model 有 gain、tau、delay 三个标量。
默认参考值是 K=8 ℃、tau=20 min、delay=2 min。
experiment 从 Scenario 中排除 gain、tau、delay，留下 ambient、initial、setpoint、dt、duration、disturbance_at、disturbance 七个标量。

initial_trials 是长度 5 的数组。
每一项包含 trial、source、kp、ki、iae、undershoot、movement 和 status 等字段。
proposal_budget 是剩余模型提案轮数，默认是 15。
模型看见的是辨识值和已完成的整段试验摘要，不是实时温度张量。

工具定义名为 evaluate_gains：

$$
\{\mathrm{kp}:\mathrm{number},\
\mathrm{ki}:\mathrm{number},\
\mathrm{reason}:\mathrm{string}\}.
$$

三个字段都是 required，additionalProperties=false。
tool_choice 要求接口调用这个函数，代码随后仍会验证响应中是否真的只有一个同名工具调用。
reason 只是追踪信息，不是评价函数，也不会覆盖仿真返回的 IAE。

## 一轮请求怎样形成闭环反馈

live tune() 先做一次 identify()，再运行五个初始样本：
ZN、SIMC 和三个 seed 控制的 log 空间随机点。
这五次完整仿真先写入 result 状态和 messages。

每轮开始时，代码复制当前 messages 作为 sent_messages。
调用计数 calls 先加一，随后把 sent_messages 交给 ChatClient 或注入的测试 client。
响应原文和本轮耗时都会进入 exchanges，usage 也从响应体保存到 cost.usage。

如果响应带有带 id 的标准 tool call，下一轮消息追加：

1. assistant 消息，其中保留模型的 tool_calls；
2. tool 消息，其中 tool_call_id 对应该 id，content 是 JSON 编码的 trial 和剩余轮数。

如果工具调用没有 id，代码走一个兼容分支，把同一反馈作为 user 消息追加。
这不是把模型的响应改写成成功结果，只是改变下一轮上下文的消息角色。
每轮最多允许一个 evaluate_gains 调用。

## 参数解析和试验生命周期

解析阶段先要求 calls 的长度为 1，函数名必须是 evaluate_gains。
arguments 通过 JSON 解析，键集合必须恰好是 kp、ki、reason。
reason 必须是非空字符串，kp 和 ki 的类型必须严格是 int 或 float。
布尔值不会因为在 Python 中属于 int 子类而被接受，因为代码检查的是精确类型。

随后构造 Gains。
Gains 先拒绝非有限或负数，in_bounds() 再检查 BO/LLM 共用的上下界。
实现没有自动取绝对值、补缺省字段或截断数值。

参数全部通过后，run_trial() 调用 evaluate(scenario, candidate)。
这次仿真返回 IAE、undershoot、movement、seconds 和 status=evaluated。
trial 的 source 为 LLM，并附上 round 和模型给出的 reason。
cost.simulations 加一，cost.plant_steps 增加一个 Scenario.steps，即 1200。

如果 JSON 解析、字段、类型、Gains 或边界检查失败，内层异常捕获会生成：

| 字段 | 值 |
| --- | --- |
| trial | 当前列表长度 |
| round | 当前模型轮次 |
| source | LLM |
| status | invalid |
| error | 校验错误文本 |

invalid trial 会进入 trials，也会带着 remaining_rounds 反馈给下一轮。
它消耗一次模型调用和一轮提案预算，但不增加仿真次数。
best_gains() 只从 status=evaluated 的行选择，因此 invalid 不可能伪装成一个增益结果。

## 网络失败和解析失败是两条不同路径

如果 url 请求抛出异常，或者响应没有 choices、message 或合法的 tool_calls 列表，外层 catch 会将运行标为 incomplete。
exchanges 保存本轮 request、错误类型和耗时。
当前请求不会生成 trial，已经完成的初始仿真和前面有效 trial 保留在 result.json。

这条路径没有重试，没有供应商切换，也没有用 SIMC 或 ZN 填补缺失轮次。
调用次数已经增加，所以 cost.calls 能显示“尝试过一次请求”，而 cost.simulations 不会假装增加。

如果响应结构通过外层检查，但 arguments 不满足字段、类型或边界条件，则走 invalid trial 路径。
这种差别让保存的 result 可以区分网络不可用和模型给出不合法候选。
两种失败都不会把一个未运行的仿真写成 evaluated。

当所有提案轮次完成时，代码从 evaluated 行取实际 IAE 最小者。
初始五个 trial 总是先于模型调用完成，所以即使所有模型调用都 invalid，仍有可追踪的初始结果。
停止原因按最终最佳是否低于初始五次最佳记录为 budget exhausted; improved initial samples 或 no improvement over initial samples。

## 参考运行的真实记录

附带记录的默认对象仍是 K=8、tau=20、delay=2，采样 0.1 min，时长 120 min。
初始五次与 BO 章节完全相同：

| 来源 | Kp | Ki | IAE / ℃·min |
| --- | ---: | ---: | ---: |
| ZN | 1.125000 | 0.168919 | 49.3479 |
| SIMC | 0.288462 | 0.014423 | 77.9950 |
| random | 0.378296 | 0.000995 | 195.8128 |
| random | 0.012633 | 0.000115 | 645.4688 |
| random | 1.034115 | 0.237824 | 48.7413 |

前三轮模型提案分别是：

| 轮次 | Kp | Ki | IAE / ℃·min | 上下文变化 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1.08 | 0.20 | 48.6172 | 在初始最好点和 ZN 附近取中间值 |
| 2 | 1.15 | 0.22 | 48.7495 | 更大的两个增益反而变差 |
| 3 | 1.05 | 0.19 | 48.5851 | 读到变差后向较小增益移动 |

第二轮的 48.7495373970 来自仿真工具。
它没有被模型理由中的“趋势”替换，仍然作为下一轮 user/tool feedback 的结果出现。

第 15 轮提出 Kp=1.02、Ki=0.192，实际 IAE 为 48.4864992382。
相对初始最佳 48.7412801021，下降约 0.52%。
该最佳响应的最大过冷是 0.0336830 ℃，movement 是 1.8771968。

![DeepSeek Flash 十五轮真实提案及历史最佳值](/results/08-llm/search.svg)

<!--@include: ../generated/llm-trials.md-->

![DeepSeek 最佳参数对应的温度与输出](/results/08-llm/response.svg)

<!--@include: ../generated/llm-metrics.md-->

<!--@include: ../generated/llm-status.md-->

参考记录的 15 次调用共保存输入 47,059 tokens、输出 1,730 tokens，其中缓存命中合计 42,880。
usage 是接口返回的统计，不从 token 数推算费用。
该记录的 base_url 是 api.deepseek.com，模型为 deepseek-flash，reasoning_effort 为 none。
结果中还保存 calls=15、simulations=20、plant_steps=24,000、identification_steps=2,400。

参考机器的总墙钟约 89.1399 s。
它包含网络服务耗时和本地仿真，不能作为其他网络、模型或设备的延迟保证。
Authorization 请求头不写入 result.json；保存的是 messages、响应、工具结果和 usage。

## 为什么模型理由不等于控制指标

reason 字段允许记录模型怎样解释一次提案，但评价函数只读温度轨迹。
参考第二轮的理由认为增益继续增大可能有益，真实 IAE 却比第一轮更高。
实验循环不接受模型自报的分数，也不因为理由听起来合理而跳过 evaluate_gains。

历史最佳曲线只记录目前已经测到的最小 IAE。
后续 trial 变差不会让历史最佳上升，也不会从 trials 中删除。
因此一条下降曲线只能说明“曾经测到的最好值”，不能说明每一轮都改善。

LLM 也没有看到真实的 K、tau、delay，只看到阶跃辨识结果。
辨识偏差、固定场景、扰动配置和单对象预算都会限制它的搜索。
它没有从多对象历史中学习一个可加载的模型，result.json 也不是可部署的网络权重。

## 服务和仿真边界

模型请求发生在完整仿真之外。
一次有效工具调用对应一个固定候选和一个完整 120 min 仿真；模型不会在 1200 个物理步之间插入控制动作。
默认 15 次提案加 5 次初始试验，所以最多 20 次仿真和 24,000 个闭环步。

在线控制阶段只保留最终 Kp、Ki。
PI 每 0.1 min 计算一个标量指令，LLM 调用次数为 0。
若对象参数、负荷或延迟发生变化，当前流程不会自动重新整定，除非外部 commissioning 流程再次提供场景并启动实验。

服务成本和可用性来自外部接口：
上下文消息随着历史增长，输入 token 逐轮增加，网络或服务失败会直接终止当前 run。
代码没有声称某个 provider 的稳定性、费用、实时延迟或输出格式永远不变。
比较命令也不会为 12 个 holdout 对象隐式发起 LLM 请求。

## 产物和下载

| 产物 | 作用 |
| --- | --- |
| result.json | 状态、最佳增益、trial、exchanges、usage 和成本 |
| LLM.csv | 最佳候选对应的温度、输出和固定增益 |
| metrics.json | IAE、过冷、movement |
| search.svg | 参数提案与历史最佳 |
| response.svg | 最佳闭环响应 |
| explain/llm.json | 前两条已保存 exchange 的结构摘要 |

本章数据下载：[完整结果](/results/08-llm/result.json)、[响应 CSV](/results/08-llm/LLM.csv)、[指标](/results/08-llm/metrics.json)。

## 结果复现入口

不访问服务的本地状态入口是：

```bash
python -m hvac_pid llm --output outputs/tutorial
```

它写入 status=not_run、calls=0 的 result.json。
解释命令只读取已有记录，代码明确写入 note 表示不会调用 API：

```bash
python -m hvac_pid explain --method llm --output outputs/tutorial
```

参考记录的接口配置形式如下，真实密钥只应通过外部环境变量提供：

```powershell
$env:LLM_BASE_URL = "https://api.deepseek.com"
$env:LLM_MODEL = "deepseek-flash"
$env:DEEPSEEK_API_KEY = "接口密钥"
python -m hvac_pid llm --live --key-env DEEPSEEK_API_KEY --reasoning-effort none --output outputs/live
```

live 是显式外部服务入口，会产生 provider 的网络请求和可能的服务费用；本章附带数字来自已保存的 reference result.json。
更改 rounds 会改变模型调用预算和仿真次数，更改 base-url、model 或 key-env 会改变服务配置，不能与 reference 记录混合解释。

下一章把固定参数方法和低频在线策略放到统一工况与未见对象协议下比较。
