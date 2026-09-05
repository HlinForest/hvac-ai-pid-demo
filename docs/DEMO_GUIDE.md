# 项目运行与演示指南

本文档面向第一次接触本项目的使用者，按顺序执行即可完成"跑通 → 生成演示 → 向他人展示"的完整流程。所有命令均在项目根目录（本文件所在目录）下的 PowerShell / 终端中执行。

## 第 0 步：环境准备（一次性）

- 需要 **Python 3.10+**。
- 安装依赖：

  ```powershell
  python -m pip install -r requirements.txt
  ```

- 可选：验证环境正常（跑测试套件）：

  ```powershell
  python -m pytest tests -q
  ```

- 可选：`build_detailed_docs.py`、`build_learning_manual.py`、`reports/md2docx.py` 等文档生成脚本额外需要 `python-docx` 和 `pillow`（不在 requirements.txt 中）：

  ```powershell
  python -m pip install python-docx pillow
  ```

## 第 1 步：跑核心流水线（生成实验数据）

**快速冒烟版**（规模小、速度快，用于确认一切正常）：

```powershell
python main.py --quick
```

或直接双击 `run_quick_demo.bat`。

**完整正式版**（48 训练 / 16 验证 / 16 测试场景 + 5×16 密封验收场景）：

```powershell
python main.py
```

流水线分三阶段执行（FOPDT 辨识与经典整定 → 贝叶斯/FNN/RL → 评估与导出）。结束后所有结果写入 `outputs/`，核心交付物：

- **`archive/outputs_review_v3/engineering_report.html`** — 主报告，浏览器直接打开，自带图表、可打印成 PDF
- 各算法指标 CSV、训练历史、对比图

注意：`python run.py crossval outputs` 只会根据**现有** CSV 重渲染报告，不会重新训练。要先跑过第 1 步。

## 第 2 步：生成离线交互演示（演示主角）

```powershell
python run.py embedded --algorithm all --provider replay
```

在 `archive/outputs_embedded_demo/` 生成：

- **`temperature_control_demo.html`** — 自包含离线交互页面（温度回放、系统图、七算法对比、LLM Agent 工具调用审计），双击即开，无需联网
- 每个算法单独的 `<算法名>_temperature_demo.html`（如 `zn_temperature_demo.html`）
- `ESP32_七算法温度闭环_零基础教学.pptx`（教学 PPT）

`--provider replay` 为离线回放模式，不需要 API key、不联网、不花钱。

只想跑单个算法时：`python run.py embedded --algorithm zn`（可选 zn/imc/bo/safe-bo/fnn/rl/llm）。

## 第 3 步：启动网页版实时演示（Streamlit）

```powershell
streamlit run streamlit_app.py
```

浏览器自动打开 `http://localhost:8501`（"ESP32 七算法温度闭环 Demo"）。网页内回放冻结的部署轨迹，展示各算法接受/回退曲线。终端按 **Ctrl+C** 停止。

## 第 4 步：演示时展示什么（建议顺序）

1. **PPT 开场**：`archive/outputs_review_v3/submission_package/HVAC_AI_PID_关键展示.pptx`（另有"含MCU验证""整定训练完整版"变体）
2. **交互演示**：Streamlit 网页，或双击 `archive/outputs_embedded_demo/temperature_control_demo.html`
3. **技术细节**：`archive/outputs_review_v3/engineering_report.html`（含三层交叉验证证据链）
4. **数据表格**：`archive/outputs_review_v3/submission_package/HVAC_AI_PID_仿真实验结果.xlsx`

PPT 和 Excel 是预构建产物，直接打开即可，没有脚本重新生成它们。

## 可选环节（按需追加）

| 演示内容 | 命令 | 前提条件 |
|---|---|---|
| OpenModelica 物理交叉验证 | 双击 `run_openmodelica_validation.bat` | 已安装 OpenModelica |
| 在 OMEdit 中查看模型 | 双击 `open_modelica_gui.bat` | 已安装 OpenModelica（通过 `OPENMODELICAHOME` 或 PATH 定位，否则改 bat 内 `OMEDIT_EXE` / `MODELICA_OMEDIT`） |
| 各算法整定耗时对比 | `python run.py benchmark` | 无 |
| Safe BO + LLM 监督整定对比 | `python run.py advanced --quick --llm-provider heuristic` | 无（启发式模式，离线） |
| LLM 自动整定（离线回放） | `python run.py llm-agent` | 无 |
| LLM 自动整定（阿里云百炼） | 把 key 写入项目根目录 `.env`（模板见 `.env.example`，或 `$env:DASHSCOPE_API_KEY="..."`），然后 `python run.py llm-agent --provider openai-compatible --model qwen-plus` | 百炼 API key，产生费用 |
| LLM 自动整定（其他 OpenAI 兼容 API） | 加 `--base-url` 和 `--api-key-env` 覆盖端点与密钥变量名（如 DeepSeek：`--base-url https://api.deepseek.com/v1 --api-key-env DEEPSEEK_API_KEY`） | 对应服务商 API key，产生费用 |
| LLM 自动整定（OpenAI 官方） | `$env:OPENAI_API_KEY="..."` 然后 `python run.py llm-agent --provider openai --model "<模型名>"` | OpenAI API key，产生费用 |
| LLM 自动整定（本地模型） | `python run.py llm-agent --provider ollama --model "<支持工具调用的模型>"` | 本地安装 Ollama |
| MCU 硬件在环（PC 端 SIL） | 见下方三连命令 | PATH 中有 g++ |

MCU SIL 三连（顺序执行）：

```powershell
python embedded\export_policy.py archive/outputs_review_v3
python embedded\wokwi\prepare_projects.py
python embedded\run_mcu_validation.py --artifact-dir archive/outputs_review_v3
```

## 最简演示路线（时间紧张时）

```powershell
python main.py --quick
python run.py embedded --algorithm all --provider replay
```

然后打开两个文件即可：

1. `archive/outputs_embedded_demo\temperature_control_demo.html`（现场交互）
2. `outputs\engineering_report.html`（技术支撑）
