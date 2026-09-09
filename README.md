# HVAC AI-PI 自动整定演示

这是一个可复现的 HVAC 仿真与控制算法项目。项目用 3R2C 热模型模拟室内空气、墙体或机柜热容和空调执行器，并比较 Ziegler-Nichols、IMC、贝叶斯优化、FNN、强化学习和 LLM Agent 等 PI 自动整定方法。

项目用于算法验证、教学和软件在环测试。它不是可以直接写入真实 PLC 或压缩机的控制器；接入真实设备前仍需要模型校准、影子运行、限幅、故障回退和逐级闭环验收。

## 先看逐步教学网站

报告已经整理成 00-12 共 13 章。每章都按“问题 → 准备 → 运行 → 走查 → 观察 → 核对 → 边界”的顺序展开，适合有基础 Python 经验的读者。

- 在线地址：<https://hlinforest.github.io/hvac-ai-pid-demo/>（GitHub Pages 开启并部署后可访问）
- 报告源文件：[reports/分报告/](reports/分报告/)
- 主报告：[docs/主报告.md](docs/主报告.md)
- 生成网站源码：[site/](site/)

本地预览：

```powershell
python tools/enrich_reports.py
python tools/build_site.py
Set-Location site
npm install
npm run docs:dev
```

命令结束后，打开终端打印的本地地址，通常是 `http://localhost:5173/hvac-ai-pid-demo/`。修改报告后重新运行前两个 Python 命令即可刷新网站内容。

## GitHub Pages 的会员限制

你看到的“需要开会员”是 GitHub 的方案限制，不是项目配置错误。GitHub Free 只能从公开仓库使用 Pages；私有仓库需要 Pro、Team 或 Enterprise。当前仓库的 Pages 工作流已经能成功构建，部署阶段会因为私有仓库没有 Pages 权限而失败。

不付费时有两种查看方式。

### 方式一：本地查看，代码继续保持私有

```powershell
python tools/enrich_reports.py
python tools/build_site.py
Set-Location site
npm install
npm run docs:dev
```

打开终端打印的地址，通常是 `http://localhost:5173/hvac-ai-pid-demo/`。这是最完整的预览方式，不需要发布仓库。

### 方式二：代码私有，网站单独放在公开仓库

1. 在 GitHub 新建一个只放网站的公开仓库，例如 `hvac-ai-pid-demo-site`。
2. 在本项目生成网站：

   ```powershell
   python tools/enrich_reports.py
   python tools/build_site.py
   Set-Location site
   npm install
   npm run docs:build
   ```

3. 将 `site/.vitepress/dist/` 内的文件复制到公开仓库根目录并推送。
4. 在公开仓库的 **Settings → Pages** 中选择 **Deploy from a branch**，分支选 `main`，目录选 `/ (root)`。
5. 页面地址会是：`https://hlinforest.github.io/hvac-ai-pid-demo-site/`。

这样公开的是生成后的报告和图片，原始算法代码仍留在本私有仓库。若以后购买 Pro，也可以继续使用当前仓库的 [Pages 工作流](https://github.com/HlinForest/hvac-ai-pid-demo/actions/workflows/pages.yml)，无需改动网站源码。

## 第一次运行项目

环境要求：Python 3.10 或更高版本。

```powershell
python -m pip install -r requirements.txt
python main.py --quick
python run.py benchmark
streamlit run streamlit_app.py
```

`streamlit run` 启动本地交互页面。终端显示地址后，在浏览器打开 `http://localhost:8501`；停止服务时按 `Ctrl+C`。

常用入口：

```powershell
python run.py --help
python run.py embedded --algorithm all --provider replay --output archive/outputs_embedded_demo
python run.py render outputs
python embedded/run_mcu_validation.py
python embedded/wokwi/prepare_projects.py
```

## 推荐阅读顺序

1. [00 总览](reports/分报告/00_我们要控制的究竟是什么.md)：先明确控制对象、输入、输出和验收标准。
2. [01 控制对象](reports/分报告/01_先把空调房间讲明白.md)：理解 3R2C 房间模型和执行器约束。
3. [02-03 基线算法](reports/分报告/02_先用传统方法得到一个基线.md)：先得到可以解释的 Z-N 和 IMC 基线。
4. [04-10 AI 方法](reports/分报告/04_贝叶斯优化到底在优化什么.md)：逐章比较 BO、FNN、RL 和 LLM Agent。
5. [11 统一比较](reports/分报告/11_把所有方法放到同一张成绩单上.md)：查看统一工况和指标。
6. [12 部署验证](reports/分报告/12_从Python到控制板.md)：了解 PC-SIL、MCU、Wokwi 和真实设备之间的证据边界。

每章同时提供 Markdown 和 DOCX；Markdown 是源文件，DOCX 由 `reports/md2docx.py` 生成。

## 实验结果与证据

实验清单和数据分区以 [experiments/manifests/v4.yaml](experiments/manifests/v4.yaml) 为准，新实验产物写入 `artifacts/runs/<run_id>/`。历史批次放在 `archive/`，用于核对旧结果，不应覆盖新的实验记录。

正式密封测试结果见 [RESULTS.md](RESULTS.md) 和 `archive/outputs_review_v3/`。报告会分别展示候选控制器和实际执行控制器；候选未通过安全门时，系统回退到 IMC 基线。生成了候选表不代表训练已经收敛。

嵌入式演示使用 90 秒墙钟回放约 5 小时虚拟物理时间，不能把它解释为真实房间在 90 秒内完成降温。PC 软件在环、ESP32/STM32 编译和 Wokwi 运行证据，也不能替代实体设备上的 10 分钟可靠性、最坏执行周期和现场安全验收。

## 主要目录

| 路径 | 用途 |
| --- | --- |
| `hvac_pid/` | 3R2C 热模型、PI 控制器、调参算法、指标和流水线 |
| `main.py`、`run.py` | 主流水线和统一命令行入口 |
| `reports/分报告/` | 13 章逐步教学报告及 DOCX |
| `reports/md2docx.py` | 将 Markdown 报告转换为 DOCX |
| `tools/` | 报告构建、实验辅助和校验脚本 |
| `site/` | VitePress 网站源文件和配置 |
| `experiments/manifests/` | 实验清单、种子和历史产物映射 |
| `artifacts/runs/` | 按 run_id 保存的新实验产物 |
| `archive/` | 只读的历史实验批次 |
| `embedded/` | PC-SIL、STM32、ESP32 和 Wokwi 工程 |
| `modelica/` | OpenModelica 物理参考模型 |
| `tests/` | 单元测试和端到端测试 |

## 生成报告和验证

重新生成所有教学章节的网站源文件：

```powershell
python tools/enrich_reports.py
python tools/build_site.py
```

重新生成 DOCX：

```powershell
python reports/md2docx.py docs/主报告.md docs/主报告.docx
```

运行项目测试和 DOCX 校验：

```powershell
python -m pytest tests -q
python tools/check_docx.py
```

## 从仿真走向真实 HVAC

上线前至少需要：用 BMS 数据或安全阶跃试验校准热模型；使用真实天气、占用和负荷数据；冻结并审查 KPI 权重；先只读推理和影子评分；保留现有 PLC/PI 作为底层闭环；对通信中断、域外输入、振荡和传感器故障执行立即回退。

详细设计和操作说明见 [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md)、[docs/ALGORITHM_GUIDE.md](docs/ALGORITHM_GUIDE.md)、[ARCHITECTURE.md](ARCHITECTURE.md) 和 [EXPERIMENTS.md](EXPERIMENTS.md)。
