# AI-PID 调参：SOTA、LLM 方法与本项目实现

## 1. 结论

Q-learning 不是当前 PI/PID 自动调参的普遍 SOTA。它把状态和动作离散成表格，适合教学和很小的离散问题，但样本效率低、连续参数表达粗糙，也很难严格保证稳定和安全。

本项目只有 `Kp`、`Ki` 两个连续参数，一次真实实验昂贵，压缩机又不能随意试错。更合适的先进路线是：以可工作的 IMC-PI 为安全起点，用高斯过程代理昂贵实验，同时建模性能和安全裕量，把噪声方差计入风险，只尝试保守预测安全的候选，最后用独立场景验收并在失败时回退 IMC。

本项目将其实现为 `RiskAwareSafeBOTuner`。思想来自 RaGoOSE 的风险规避安全贝叶斯优化，但这是针对本项目的工程改编，不声称逐行复现论文源码。

大模型适合承担“读指标—诊断—提出下一组候选增益”的低频监督任务，不适合直接决定每分钟的压缩机命令。本项目实现 OpenAI Responses API 和本地 Ollama 接口，但建议必须经过确定性限幅、重复仿真、安全门和回退逻辑。

## 2. 被优化的问题

PI 控制器为

\[
u_{raw}(t)=K_p e(t)+K_i\int_0^t e(\xi)d\xi
\]

冷却系统采用 `e = 测量温度 - 设定温度`：`Kp` 大会更快响应当前温差，但可能抖动或过冷；`Ki` 大会更快消除长期残差，但容易积分累积、过冲和频繁启停。

`u_raw` 不是最终压缩机命令。项目还施加 0～1 限幅、每分钟 5% 斜率、1% 台阶量化、25% 最低运行命令和最小启停时间。因此调参是带约束的多场景优化：

\[
\min_{K_p,K_i}J(K_p,K_i),\qquad q_j(K_p,K_i)\le 0
\]

`J` 综合 IAE、ITAE、舒适带超限、过冷、调节时间、控制动作、命令方差和能耗；`q_j` 表示稳定性、斜率、最低运行频率等安全约束。本实现还把“候选风险分不得超过基线的 1.25 倍”作为性能安全预算，防止设备虽未数值发散却出现灾难性退化。

## 3. 重复实验与风险目标

同一增益在传感器噪声和扰动下不会得到完全相同的分数。候选 `x=(Kp,Ki)` 重复 `n` 次：

\[
\bar J(x)=\frac1n\sum_{r=1}^{n}J_r(x),\qquad
s_J(x)=\sqrt{\frac1{n-1}\sum_{r=1}^{n}(J_r(x)-\bar J(x))^2}
\]

本实现优化

\[
J_{risk}(x)=\bar J(x)+\lambda_r s_J(x)
\]

默认 `λr=0.75`。平均分相同但波动更小的增益风险分更低。RaGoOSE 把安全学习与异方差噪声下的风险规避结合，并在精密运动系统实验中验证，见 [Safe Risk-averse Bayesian Optimization for Controller Tuning](https://arxiv.org/abs/2306.13479)。

## 4. 风险感知安全 BO

```text
IMC 基线 ──┐
           ├─> 局部候选 -> 多场景×重复带噪仿真
增益边界 ──┘                    │
                                ├─> 均值/方差 -> 风险 GP
                                └─> 最大安全裕量 -> 安全 GP
                                                  │
候选池 -> μq + βσq <= 0 ? ──否──> 禁止选择
                  │是
                  v
       风险 GP 下置信界选下一点
                  │
                  v
    最优实测安全点 -> 独立留出集 -> 部署/回退
```

高斯过程只根据少量已测点估计未测点的分数与不确定性，不直接控制空调。安全 GP 给出安全裕量均值 `μq` 和标准差 `σq`，保守判据为

\[
\mu_q(x)+\beta\sigma_q(x)\le0
\]

默认 `β=2`。不确定性越大越难通过。本实现还在已知安全点周围保留小信赖域，避免初期数据太少时完全停止。SafeCtrlBO 使用加性 GP 处理多级/多控制器参数并在电机平台验证，更适合高维多环控制；本项目只有两个增益，暂不引入其复杂度。见 [SafeCtrlBO](https://arxiv.org/abs/2408.16307)。

## 5. 大模型的边界

大模型可以把“过冷、调节慢、命令抖动”等指标转成工程诊断，结合当前增益和历史建议提出下一组 `Kp/Ki`，并解释理由。它不能直接输出压缩机命令，不能绕过边界、安全仿真或回退，也不能用语言解释代替独立测试。

2026 年一项物理信息 LLM-Agent PID 研究采用“响应特征—诊断—增益建议—验收”的迭代框架，并探索 SFT 和物理信息 GRPO。这说明它是活跃研究方向，但化工仿真的成功率不能直接当成本空调项目的证据。见 [A Physics-Informed Framework for PID Tuning Using LLM Agents](https://arxiv.org/abs/2607.26594)。较早开源实现采用 OpenAI/Ollama、指标反馈和迭代建议，见 [LLM-Based PID Controller Optimization](https://github.com/ilijakamenko/LLM_PID_Tuner)。

本项目的门控流程为：

```text
当前 Kp/Ki -> 重复仿真指标 -> LLM 候选+诊断
                                  │
                                  v
                         硬边界+单轮最多±25%
                                  │
                                  v
                         多场景重复噪声仿真
                                  │
                    安全且风险分至少改善0.5%？
                         │是                 │否
                         v                   v
                    更新安全点          拒绝并保持原点
                                  │
                                  v
                            不合格回退IMC
```

OpenAI 接口使用 Responses API 严格 JSON Schema，只允许返回 `kp`、`ki`、诊断、理由和置信度，依据 [OpenAI Responses API 官方文档](https://developers.openai.com/api/reference/resources/responses/methods/create)。密钥仅在调用时从 `OPENAI_API_KEY` 读取，不写入报告或 CSV。

## 6. 运行方法

免费验证整条安全流水线（启发式明确不是 LLM）：

```powershell
python run_advanced_tuning_benchmark.py --quick --llm-provider heuristic --output outputs_advanced_quick
```

OpenAI Responses API：

```powershell
$env:OPENAI_API_KEY="你的密钥"
python run_advanced_tuning_benchmark.py --quick --llm-provider openai --llm-model "明确指定的模型名" --output outputs_advanced_openai
```

程序不会默认猜模型名。调用会产生费用。使用本地 Ollama：

```powershell
python run_advanced_tuning_benchmark.py --quick --llm-provider ollama --llm-model "已安装的模型名" --output outputs_advanced_ollama
```

仅比较数值优化：

```powershell
python run_advanced_tuning_benchmark.py --llm-provider none
```

## 7. 输出文件

- `advanced_holdout_summary.csv`：独立留出场景的均值、标准差、风险、最坏分数和安全性，是最重要的比较；
- `safe_bo_history.csv`：每个候选的预测/实测安全裕量；
- `llm_tuning_history.csv`：原始建议、限幅后候选、诊断和接受/拒绝原因；
- `tuning_wall_time.csv`：本机真实墙钟耗时，不把仿真时长冒充计算耗时；
- `advanced_tuning_report.md`：本次规模与结论。

## 8. 不能夸大的地方

1. 当前对象仍是 3R2C 虚拟模型，不等于真实空调，下一步需用 BMS、OpenModelica 或 HIL 复核。
2. “RaGoOSE 风格”表示采用风险与安全代理思想，不表示满足论文全部实验条件。
3. LLM 的论文新颖性不等于闭环性能 SOTA；在本架构中它只是候选生成器。
4. 没有真实调用模型时，启发式 dry-run 绝不能报告成 LLM 效果。
5. 上真实设备前还需人工批准、厂家边界、急停、看门狗、影子模式和灰度实验。
