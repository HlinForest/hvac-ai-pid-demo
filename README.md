# 动手做 AI 自动整定

同一个温控对象，七种选择 PI 参数的方法：Z-N、SIMC、BO、FNN、Q-Learning、DQN 和 LLM。

这个项目想回答：程序能不能替我们少试几组参数？能不能从过去的整定案例中学习？工况变了以后，在线调整参数有什么用？大模型读完实验反馈，下一次建议是否更好？

从 [第一章](https://hlinforest.github.io/hvac-ai-pid-demo/chapters/01-temperature) 开始。网站采用章节正文、实际代码和实验曲线连在一起的写法。新版网站将在重构分支合并并触发 Pages 后更新。

## 先跑一条温度曲线

需要 Python 3.10+。在仓库根目录建立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m hvac_pid temperature
```

Linux/macOS 的激活命令是 `source .venv/bin/activate`。打开 `outputs/tutorial/01-temperature/response.svg`，再修改参数：

```bash
python -m hvac_pid temperature --kp 0.6 --ki 0.03 --output outputs/my-first-change
```

第一次使用不需要 PyTorch，也不需要模型 API。

## 逐章运行

| 命令 | 对应实验 |
|---|---|
| `python -m hvac_pid plant` | 阶跃响应和辨识 |
| `python -m hvac_pid classical` | Z-N、SIMC 与 λ 扫描 |
| `python -m hvac_pid bo` | 五个初始样本加十五次 BO 提案 |
| `python -m hvac_pid fnn` | 生成教师标签、训练 27 条模糊规则 |
| `python -m hvac_pid qlearning` | Q 表在线调参，默认 500 回合 |
| `python -m hvac_pid dqn` | 小型 DQN 在线调参，默认 500 回合 |
| `python -m hvac_pid llm --live` | 显式调用真实模型 |
| `python -m hvac_pid compare` | 使用已生成的模型与结果比较 |

DQN 需要额外安装 CPU 版 PyTorch：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m hvac_pid all
```

`all` 重跑全部本地实验，不调用模型 API；已有的真实 LLM 记录会保留。`--without-dqn` 可省略 DQN。独立运行 `compare` 前，先运行 BO、FNN、Q-Learning、DQN 和 `llm`（不加 `--live` 可生成明确的未运行记录）。

训练规模可以修改，例如 `python -m hvac_pid dqn --episodes 1000 --output outputs/longer`。五种子重复比较：

```bash
python -m hvac_pid repeat --seeds 0 1 2 3 4 --output outputs/repeated
```

重复命令会重建每个种子的 FNN 教师数据与 RL 模型，汇总均值、样本标准差和范围，不会隐式发起 LLM 请求。不要把这些本地重复结果称作七方法完整统计比较。

## 连接 LLM

仅维护兼容 Chat Completions 的工具调用接口。PowerShell 示例：

```powershell
$env:LLM_BASE_URL = "https://api.deepseek.com"
$env:LLM_MODEL = "deepseek-flash"
$env:DEEPSEEK_API_KEY = "你的密钥"
python -m hvac_pid llm --live --key-env DEEPSEEK_API_KEY --reasoning-effort none
python -m hvac_pid compare
```

DeepSeek 的指定工具调用要求关闭思考模式，这里显式设置 `--reasoning-effort none`，参见[DeepSeek 接口文档](https://api-docs.deepseek.com/api/create-chat-completion/)。也可通过 `--base-url`、`--model`、`--key-env` 接入其他兼容服务；不需要推理设置的接口省略 `--reasoning-effort`。`.env.example` 只提供字段说明，程序不会自动读取 `.env`。

每次 `--live` 最多调用 15 次模型。请求、响应、仿真结果和服务返回的用量保存在 `08-llm/result.json`；不保存 Authorization 请求头。缺少配置或接口失败时命令返回非零状态并保存原因，不用 IMC 或手写规则代替模型结果。

## 阅读与构建网站

仓库只维护一份正文：`site/chapters/`。参考实验位于 `experiments/reference/`，包括配置、CSV、图表、模型和训练记录。它是新版的一次实际运行，不继承旧报告数字。

在已激活 Python 环境的终端中执行：

```bash
cd site
npm ci
npm run docs:prepare
npm run docs:dev
```

准备步骤只复制图片和生成数值表格，不改写章节正文。要显示自己重跑的实验，回到仓库根目录执行：

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

浏览器检查首次使用前运行 `npx playwright install chromium`。GitHub Actions 执行 Python 测试、网站构建和浏览器检查；Pages 在 `main` 更新后发布。

## 新版结构与旧版

`hvac_pid/` 实现算法；`experiments/` 编排实验、保存结果；`site/` 解释实验。输出目录使用 `outputs/`，重复运行同一目录会更新该实验的文件，比较不同设置时使用不同目录。

时间统一为分钟，IAE 单位是 ℃·min，控制输出是 0 到 1 的制冷指令。模型只表达一个温度状态、惯性和延迟，不将其换算为真实空调节电量。

本次重构的旧版依据是提交 `00056dfe4c19ce7d8b4f317b15832cb85f7d67b8`。需要查看原有热模型、硬件工程或报告时，从 Git 历史检出该提交。新版不保留旧命令兼容层。
