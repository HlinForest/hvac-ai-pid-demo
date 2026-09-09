# 12 从 Python 到控制板：验证我们真的运行了同一套策略

> **本篇目标**：完成策略导出、对拍和 SIL，理解实体实验还差哪些步骤。
> **你需要**：第 03 篇的策略版本概念、第 11 篇的冻结策略。
> **你会得到**：MCU 头文件、CRC、对拍报告、SIL 结论与硬件门清单。
> **运行方式**：快速体验（PC 对拍 + SIL，全软件）/ 实体验证（待硬件，步骤已就绪）。
> **版本依据**：`embedded/export_policy.py` + 封存 `archive/outputs_review_v3/policy_manifest_v3.json`。

---

## 阅读路线

先导出同一份策略，再做 Python/C++ 对拍和 PC-SIL，最后列出仍待实体验证的门。

按下面的顺序阅读：

1. **问题**：先说清这一步想回答什么，避免把指标当成结论。
2. **准备**：列出前置章节、配置、数据来源和版本；每一步都先看输入，再看中间产物，最后用表格或断言核对输出。
3. **运行**：复制命令或读取封存产物，记录输出目录和随机种子。
4. **走查**：用一个具体数值代入公式，再对照实现中的关键几行。
5. **观察**：先读坐标、曲线和表格，再说明“发生了什么”和“这说明什么”。
6. **核对**：用容差、断言、哈希或独立验证判断复现是否成功。
7. **边界**：把已经证明、只作观察和仍待验证的内容分开写。

| 阅读检查 | 完成标准 |
|---|---|
| 输入明确 | 能说出本步使用的状态、参数、数据集或基线 |
| 中间过程可见 | 至少有一段数值演算、过程记录或关键代码 |
| 输出可核对 | 能定位到 CSV、图、日志、断言或封存目录 |
| 结论有边界 | 没有把仿真、replay、候选或稳定性误写成实体验收 |

---

## 这次要解决什么问题

仿真里再好的参数，烧到板子上跑的必须是同一套——差一个字节都不行。本篇回答：冻结的 Python 策略如何变成控制板上的代码，我们用什么证明"同一套"，以及哪些步骤还缺硬件做不了。

## 开始前准备什么

- 前置：冻结策略目录（`--artifact-dir`，含规则表、Q 表、覆盖掩码、IMC 基线）。
- 工具链：PC 对拍只需 Python + C++ 编译器；ESP32 目标编译需 PlatformIO；实体验证需开发板与串口。
- 预计资源：导出与对拍约数秒；SIL 约数分钟；实体 10 分钟遥测待硬件。

## 先跑一个最小实验

在哪执行 → 项目根目录。执行什么 → 策略导出与对拍：

```bash
python embedded/export_policy.py <策略目录> --header embedded/generated_policy.hpp
python embedded/run_mcu_validation.py --artifact-dir <策略目录>
```

读哪些输入 → 冻结的 FNN 表（含上下文系数）、RL 表（含覆盖掩码）、IMC 基线与部署门记录。生成哪些文件 → `generated_policy.hpp`（MCU 头文件）、`policy_manifest_v3.json`（清单 + CRC32）、`policy_parity_vectors.csv`（75 组对拍向量）、`mcu_validation_summary.csv`（SIL 结论）。怎样判断成功 → 对拍最大误差约 3e-8 量级通过，SIL 显示通过，CRC 与清单一致。

## 刚才发生了什么

导出把 Python 的浮点策略量化成 MCU 的 float32 头文件，清单记录版本号、接受标记与 CRC32；对拍用同一组输入分别跑 Python 与 C++，逐向量比对；SIL 在 PC 上按 100 ms/2 s 分频跑调度逻辑，检查增益边界、稳定、NaN 回退与 RL 未覆盖回退占比。

## 跟着代码走一遍

- 导出：`embedded/export_policy.py`——量化、清单、CRC、75 组对拍向量生成。
- 验证：`embedded/run_mcu_validation.py`——PC-SIL 与对拍执行。
- 固件：`embedded/hvac_pid_controller.hpp`——100 ms PI + 2 s 调度；`embedded/validate_esp32_serial.py`——实体 600 秒验收入口（待接板）。

## 结果应该怎么看

> **看哪里**：`policy_parity_vectors.csv` 的最大误差、`mcu_validation_summary.csv` 的通过项、清单 CRC。
> **发生了什么**：75 组对拍误差约 2.98e-8，通过；SIL 七档增益边界、90 秒加速稳定、NaN 回退、调度逻辑通过；ESP32 目标编译 RAM 约 22,036 B、Flash 约 296,169 B，通过；Wokwi 双目标仅编译/启动。
> **这说明什么**：软件层面的"同一套"已验证——Python 算出的策略就是板子上跑的策略。但 PC 纳秒耗时不能写成 ESP32 WCET，Wokwi 启动不能冒充板端验收。

FNN/RL 有各自的诊断：FNN 看规则激活与增益时变曲线（第 08 篇图 8-6），RL 看覆盖与回退计数（第 09 篇）；存在回退时，明确区分**候选曲线、实际执行曲线、IMC 基线**三条线——只画一条"执行曲线"会掩盖回退的存在。

## 改一个参数试试看

假设"CRC 是防呆的"：手动改头文件里一个增益末位小数，重跑对拍，观察 CRC 与误差如何报警。如果报警了，说明溯源链是活的；恢复文件归档。

## 如何确认复现成功

文件检查：头文件、清单、75 组向量、SIL 结论四件套齐全且 CRC 一致。数值容差：对拍误差 1e-6 量级以下通过。行为检查：RL 未覆盖回退占比 ≤10% 门。常见异常：编译器版本差异导致末位浮动——看量级，不看末位。

## 这次能得出什么结论

- 已证明：策略导出、对拍、SIL、目标编译——软件层"同一套"闭环。
- 待外部验证（不得仿真冒充）：实体板 10 分钟遥测、真实 WCET、HIL/Modbus 写入；Modbus 默认只读，无厂商寄存器表禁止写容量。
- 全套教程到此结束：我们从一个闭环出发，搭了机房、定了规则、看了七种做法、统一考了一次、验证了上板链路。结论的边界每一篇都已写明——带着边界用，才是这套教程想教的东西。

## 附录：本篇数据溯源

| 数据产物（仓库内路径） | 内容 | 支撑本篇何处 |
|---|---|---|
| `archive/outputs_review_v3/policy_manifest_v3.json` | 封存部署清单与 CRC | 导出核对 |
| `archive/outputs_review_v3/policy_parity_vectors.csv` | 75 组对拍向量 | 对拍 |
| `archive/outputs_review_v3/mcu_validation_summary.csv` | SIL 结论 | 验证 |
| `embedded/generated_policy.hpp` | MCU 头文件（生成物） | 导出 |

| 代码位置 | 作用 |
|---|---|
| `embedded/export_policy.py` | 导出、清单、CRC、对拍向量 |
| `embedded/run_mcu_validation.py` | PC-SIL 执行 |
| `embedded/validate_esp32_serial.py` | 实体验收入口（待接板） |
