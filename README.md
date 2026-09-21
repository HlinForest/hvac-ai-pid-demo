# 动手做 AI 自动整定

这个仓库用同一个温控对象比较 12 种 PI 整定方式：Z-N、SIMC、BO、FNN、Q-Learning、DQN、PPO、TD3、SAC、CrossQ、PG4PI-HVAC 和 LLM。每种方法都把结果接到同一个条件积分 PI 和同一个一阶加延迟对象上，因此可以从 CSV、JSON 和源码逐项核对曲线。

网站入口是 [动手做 AI 自动整定](https://hlinforest.github.io/hvac-ai-pid-demo/)。旧章节 URL `/chapters/01-temperature` 到 `/chapters/09-comparison` 继续保留；新增连续 RL 与 PG4PI 章节使用 `10-ppo`、`11-td3`、`12-sac`、`13-crossq`、`14-pg4pi`。阅读顺序由 VitePress 配置维护，正文在 `site/chapters/`。

对象只追踪一个温度状态。默认参数是环境 30℃、初始温度 28℃、目标 24℃、对象增益 8℃、时间常数 20 min、输入延迟 2 min、采样间隔 0.1 min、仿真时长 120 min；第 60 分钟开始加入 1℃ 等效扰动。时间单位统一为分钟，控制指令是 `[0,1]` 的无量纲值。IAE 单位为 ℃·min，movement 只表示输出变化量，不能直接解释为真实电耗。

## 先跑一条曲线

需要 Python 3.10 或更高版本。在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m hvac_pid temperature
```

Linux/macOS 激活命令为 `source .venv/bin/activate`。完成后打开 `outputs/tutorial/01-temperature/response.svg`，第一行控制量可以从 `Manual.csv` 核对：默认 `Kp=0.15`、`Ki=0.005` 时，$e_0=4\,^{\circ}\mathrm{C}$，所以

$$
u_0=0.15\times4+0.005\times4\times0.1=0.602.
$$

不需要 torch，也不需要模型服务就能完成温度、阶跃、经典、BO、FNN 和 Q-Learning 章节。

只改一个参数并保存到新目录：

```bash
python -m hvac_pid temperature --kp 0.6 --ki 0.005 --output outputs/my-first-kp
python -m hvac_pid temperature --kp 0.15 --ki 0.03 --output outputs/my-first-ki
```

运行命令不会覆盖参考实验目录；同一输出目录再次运行会更新该实验的结果。

## 按章节运行

| 命令 | 章节 | 额外依赖 |
|---|---|---|
| `python -m hvac_pid temperature` | PI 与温度曲线 | 无 |
| `python -m hvac_pid plant` | 阶跃响应和 $K,\tau,L$ 辨识 | 无 |
| `python -m hvac_pid classical` | Z-N、SIMC、λ 扫描 | 无 |
| `python -m hvac_pid bo` | 5 初始 + 15 次 BO | 无 |
| `python -m hvac_pid fnn` | 48/12 对象的 BO 教师与 FNN | 无 |
| `python -m hvac_pid qlearning` | 离散 9 动作、500 回合 | 无 |
| `python -m hvac_pid dqn` | `5→32→32→9` DQN | CPU torch |
| `python -m hvac_pid ppo` | 连续动作 clipped PPO | CPU torch |
| `python -m hvac_pid td3` | TD3 双 critic | CPU torch |
| `python -m hvac_pid sac` | SAC 熵正则双 critic | CPU torch |
| `python -m hvac_pid crossq` | CrossQ 联合 Batch Renorm critic | CPU torch |
| `python -m hvac_pid pg4pi` | PG4PI-HVAC 两参数策略梯度 | 无 |
| `python -m hvac_pid llm` | 不调用服务的状态记录 | 无 |
| `python -m hvac_pid compare` | 冻结工况和 12 个 holdout 对象 | 已有产物 |

现代连续 RL 的动作是长度 2 的 `[-1,1]` 向量，核心环境按 `2**action` 映射为相对 SIMC 锚点的 Kp、Ki 倍数，范围是 `[0.5,2]`。观测是长度 5 的 `(error, error slope, previous command, kp scale, ki scale)`，每 2 分钟选择一次，PI 仍每 0.1 分钟运行。各方法的首个 critic/actor 或 PPO 更新会写入 `training.json`，`explain` 会读取并生成对应的解释记录，章节中的公式可用这些实际数字复算。

安装 CPU torch：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

然后运行完整本地实验：

```bash
python -m hvac_pid all --episodes 500 --rounds 15 --output outputs/tutorial
python -m hvac_pid site --output outputs/tutorial --destination site
```

`all` 不会自动调用 LLM。没有 torch 时明确跳过所有神经 RL：

```bash
python -m hvac_pid all --without-torch --output outputs/tutorial-numpy
```

旧别名 `--without-dqn` 仍接受，但语义是跳过所有 torch 方法。

## 解释实际中间量

`explain` 调用已有实现中的相同 helper，并把一项可核对的中间记录写到 `explain/<method>.json`：

```bash
python -m hvac_pid explain --method classical --output outputs/tutorial
python -m hvac_pid explain --method bo --output outputs/tutorial
python -m hvac_pid explain --method fnn --output outputs/tutorial
python -m hvac_pid explain --method qlearning --output outputs/tutorial
python -m hvac_pid explain --method dqn --output outputs/tutorial
python -m hvac_pid explain --method ppo --output outputs/tutorial
python -m hvac_pid explain --method td3 --output outputs/tutorial
python -m hvac_pid explain --method sac --output outputs/tutorial
python -m hvac_pid explain --method crossq --output outputs/tutorial
python -m hvac_pid explain --method pg4pi --output outputs/tutorial
python -m hvac_pid explain --method llm --output outputs/tutorial
```

`explain --method all` 会依次生成这些文件，但在线方法必须已有对应 `training.json` 和首个 update；它不会偷偷训练模型。BO 解释记录包含 1024 个候选中的真实第一提案、GP 均值/标准差和 EI；DQN 记录首个 replay batch 的 target 与 Huber loss；PPO、TD3、SAC、CrossQ 记录首个 critic/actor 相关量；PG4PI 记录首条轨迹的回报和梯度。JSON 中的数组形状与章节正文一致。

## LLM 只在显式 live 模式调用

先生成不调用接口的记录：

```bash
python -m hvac_pid llm --output outputs/tutorial
```

配置一个兼容 Chat Completions 工具调用的服务后才使用 `--live`：

```powershell
$env:LLM_BASE_URL = "https://api.deepseek.com"
$env:LLM_MODEL = "deepseek-flash"
$env:DEEPSEEK_API_KEY = "你的密钥"
python -m hvac_pid llm --live --key-env DEEPSEEK_API_KEY --reasoning-effort none --output outputs/tutorial
```

程序不读取 `.env`，不重试，也不在失败时换供应商或用 SIMC 填结果。`result.json` 保存请求、响应、工具返回、指标和接口 usage，但不保存 Authorization 请求头。参考记录中 DeepSeek Flash 的 15 轮提案，最佳实际 IAE 为 `48.4865 ℃·min`；这是一个种子和一个对象的记录，不能当成服务或方法的普遍保证。

## 统一比较与 benchmark

完整比较需要已有 BO、FNN、Q-Learning、DQN/连续 RL、PG4PI 和（可选的）LLM 产物：

```bash
python -m hvac_pid compare --output outputs/tutorial
```

协议 A 在标称对象上 commissioning 后冻结参数/策略，再改变时间常数、延迟和扰动；协议 B 使用 12 个未见对象，FNN 和在线策略保持冻结，BO 和 PG4PI 按新对象协议重新整定，LLM 不在 compare 中隐式产生 12 次服务调用。图表按固定方法和在线方法分开，完整 12 方法总览仍保留。

只改变预算时保留新目录：

```bash
python -m hvac_pid all --episodes 100 --rounds 15 --output outputs/tutorial-budget100
python -m hvac_pid repeat --seeds 0 1 2 3 4 --output outputs/repeated
```

重复命令会重建每个种子的训练和 BO 标签，生成均值、样本标准差和范围；它不是把同一个 checkpoint 重复评估。

测量当前机器的 CPU 推理成本：

```bash
python -m hvac_pid benchmark --output outputs/tutorial --samples 1000
```

基准使用 batch=1 和 100 次 warm-up，报告 PI 更新、FNN 一次预测和冻结策略的参数量、checkpoint 文件大小、CPU 中位数和 P95。checkpoint 文件大小不等于 Python/PyTorch 运行时内存，也不包含解释器、线程库、驱动或安全联锁；结果不能直接承诺某个 MCU 可部署。在线位置只按调用频率说明：固定方法上线前推荐一次，在线方法每 2 分钟一次，PI 每 0.1 分钟一次。

## 网站构建

站点依赖已锁定在 `site/package-lock.json`：

```bash
cd site
npm ci
npm run docs:prepare
npm run docs:dev
```

`docs:prepare` 只复制 `outputs/<chapter>/` 中的 SVG/CSV/JSON 允许产物并生成数值表格，不改写手工章节正文。要把自己的实验导入站点，回到根目录执行：

```bash
python -m hvac_pid site --output outputs/tutorial --destination site
```

完整检查：

```bash
python -m pytest -q
cd site
npm run docs:prepare
npm run docs:build
npm run docs:check
```

首次做浏览器检查时运行 `npx playwright install chromium`。GitHub Actions 会执行 Python 测试、完整实验 smoke run、网站构建和浏览器检查。

## 代码和数据边界

`hvac_pid/` 是控制器、对象和学习算法；`experiments/` 编排运行、写出记录和生成站点数值表；`site/chapters/` 是手工维护的教程。`experiments/reference/` 保存一次附带参考运行，包括配置、CSV、曲线、模型和训练记录；它不继承旧报告中的数字。

参考实验使用单一温度状态和等效扰动，没有把指令换算成真实空调功率，也没有设备驱动、网络服务部署或硬件安全逻辑。教程中的“在线”只表示在每个仿真决策区间内调用冻结策略；它不承诺已通过真实设备验收。
