# 第二阶段：STM32F103 / ESP32 轻量化验证

## ESP32 七算法主工程

新主工程位于 `embedded/esp32_seven_algorithm_demo/`，ESP32 DevKit 是当前嵌入式主目标。它与早期双目标 Wokwi 教学工程并存，不把早期十几秒运行记录冒充新版七算法固件的完整验收。

```powershell
cd embedded/esp32_seven_algorithm_demo
pio run
pio run --target upload
pio device monitor --baud 115200
```

固件提供两种模式：

- `MODE demo`：100 ms 墙钟运行一次控制任务，并用 200 倍时间加速的虚拟热对象让 5 小时温度过程在 90 秒内可见；FNN/RL 每 2 秒才运行一次候选增益调度；
- `MODE modbus`：从真实空调控制器/BMS 读取温度、实际频率和告警，并在取得真实寄存器表且显式许可后写容量请求。默认只读，ESP32 不直接驱动压缩机功率器件。

七算法演示配置由 `run_embedded_demo.py --algorithm all` 生成到 `generated_demo_profiles.hpp`，带版本、清单字符串和 CRC32。正式评审策略则由 `outputs_review_v3` 唯一自动导出，FNN/RL 均通过密封部署门；RL 是冻结策略查表，未覆盖状态回退工厂 IMC；LLM 只部署已通过上位机仿真门的固定增益。串口命令与接线说明见 `esp32_seven_algorithm_demo/README.md`。

真实目标板验收：

```powershell
python embedded/validate_esp32_serial.py --port COM3 --seconds 600
```

脚本要求连续收到板端 JSON，检查温度有限、运行时长、`missed_periods=0`、PI 最坏执行时间小于 10 ms、AI 调度小于 100 ms，并把原始遥测和结论分别保存为 CSV/JSON。未接目标板时不会生成“通过”结论。

这一阶段要回答的不是“房间热模型是否足够逼真”，而是“同一份控制算法放到低成本 MCU 上，能否按时运行、占多少资源、异常时是否安全回退”。

## 已交付的三层验证

| 层级 | 用途 | 当前状态 | 能证明什么 | 不能证明什么 |
|---|---|---|---|---|
| PC 软件在环（SIL） | 快速检查纯 C++ 数学逻辑 | 已通过 | 限幅、抗饱和、100 ms/2 s 分频、FNN/RL 查表、异常回退 | 不能代表 MCU ROM、RAM 和最坏周期 |
| ESP32 官方目标工具链 | 目标编译与静态资源报告 | 已通过：RAM 22,036 B（6.7%），Flash 296,169 B（22.6%） | 证明当前工程可编译并满足静态容量 | 尚未证明实体板周期、10 分钟可靠性、通信和真实传感器 |
| Wokwi 虚拟 MCU | STM32F103C8、ESP32 外设交互 | 两目标编译/启动已通过；遥测留证未完成 | 已证明完整代码可进入运行态 | 不能替代真实压缩机和传感器 |
| 实体开发板 HIL | 最终部署前测试 | 未开始 | 真实时钟、中断抖动、栈、通信和故障注入 | 仍不等于整机认证 |

## 一键执行 PC 软件在环

```powershell
cd "C:\Users\厉飞雨\Documents\New project\hvac_ai_pid_demo"
python embedded\export_policy.py outputs_review_v3
python embedded\wokwi\prepare_projects.py
python embedded\run_mcu_validation.py --artifact-dir outputs_review_v3
```

第一条命令只导出通过验收的部署表，并在 `generated_policy.hpp` 写入版本号、CRC32、FNN/RL 验收状态、IMC 回退参数、状态边界、FNN 上下文参数、RL 策略和覆盖掩码。CRC v3 覆盖全部执行字段，固件启动时实际重算；任一字段损坏都使用独立编译的工厂 IMC。FNN/RL 候选训练表不能手工复制进 C++。结果写入 `outputs_review_v3/mcu_pc_sil_log.txt` 和 `mcu_validation_summary.csv`。表内明确标注 PC ABI，不能把 PC 的纳秒耗时或 exe 大小写成 ESP32 实测值。

## Wokwi 双目标工程

```powershell
python embedded\wokwi\prepare_projects.py
```

- `embedded/wokwi/stm32f103`：STM32F103C8T6（Blue Pill）验证文件。
- `embedded/wokwi/esp32`：ESP32 DevKit 验证文件。
- `embedded/wokwi/common/sketch.ino`：两种 MCU 共用的 100 ms PI / 2 s AI 调度入口。
- `embedded/hvac_pid_controller.hpp`：纯 C++ 控制核心，不访问底层寄存器。
- `embedded/generated_policy.hpp`：由最终训练目录自动生成的只读部署产物；禁止手工修改。
- `outputs_review_v3/policy_manifest_v3.json`：与头文件同源生成的可审计策略清单；固件执行内容以自动生成头文件及其 CRC 校验为准。

Wokwi 右侧电路包含设定值旋钮、开门按钮、PWM 指示灯和安全回退指示灯。串口 Plotter 输出：室温、设定值、PWM 百分比、`Kp`、`Ki`、PI/AI 最坏微秒数和回退状态。

2026-08-23 在线验证记录保存在 `outputs/mcu_wokwi_validation.csv`：ESP32 完整代码编译并运行约 10.079 s；STM32F103 完整代码编译并运行约 28.500 s，并点击一次开门扰动。匿名会话中的 ESP32 串口面板未显示数据（最小串口程序也相同），STM32 串口 Plotter 面板可见但数据未能可靠读取，因此二者都只能判为“目标编译/启动通过，遥测待补”，不能判成完整验收。

## 控制核心的资源结构

- FNN：5×5 个一阶 TSK 残差规则后件，每个状态只对相邻 4 条规则做双线性插值，并使用实际容量、积分状态、室外温差和慢速负荷估计作为上下文；域外或非法输入回退 IMC。
- RL：5×5热状态再乘3档上一次实际容量模式。运行时只带75 B动作索引、75 B覆盖掩码和72 B动作目标表，不把完整5×5×3×9浮点Q表部署到MCU。
- FNN/RL共同使用误差变化率 `ė=Δe/Δt`（°C/min），避免PC的5 min训练节拍与MCU的2 s节拍使用不同含义的`Δe`。
- 安全 PI：输出限幅、条件积分抗饱和、增益边界和单次±10%增益变化限制；输入异常、算法验收失败或RL未覆盖状态时回退IMC参数。
- 压缩机限制器：0/25%最低运行容量、1%量化、5%/min运行斜率、5 min最小开机和3 min最小停机。连续内部斜坡与量化输出分离，避免100 ms周期下每步小于1%而永远无法爬升。

## Wokwi 验收记录要求

两块目标板分别保存以下证据后，才可以把第二层状态改成“已完成”：

1. 编译成功截图和工具链/核心版本。
2. Flash/ROM、静态 RAM 的编译报告。
3. 连续运行不少于 10 分钟，PI 周期 100 ms、AI 周期 2 s 无丢拍。
4. 旋钮改变设定值、按钮注入开门扰动时，三组串口曲线同步变化。
5. NaN/越界/未覆盖 RL 状态触发回退，红灯和串口标志一致。
6. STM32 使用 DWT 或 GPIO 翻转测周期；ESP32 使用芯片周期计数器或逻辑分析仪复核最坏时延。

当前纯 C++ PC-SIL 已验证自动导出表、CRC v3 启动重算、75 组 Python/C++ 一致性向量、100 ms/2 s 调度、异常回退和压缩机基本限制器；ESP32 官方工具链已给出 RAM/Flash 静态结果。实体板 10 分钟遥测、板端 WCET、真实传感器和真实空调 Modbus 仍未完成。量产前还必须加入并实测高低压与排气温度联锁、传感器断线检查、看门狗、通信超时和版本回滚。
