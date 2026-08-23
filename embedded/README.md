# 第二阶段：STM32F103 / ESP32 轻量化验证

这一阶段要回答的不是“房间热模型是否足够逼真”，而是“同一份控制算法放到低成本 MCU 上，能否按时运行、占多少资源、异常时是否安全回退”。

## 已交付的三层验证

| 层级 | 用途 | 当前状态 | 能证明什么 | 不能证明什么 |
|---|---|---|---|---|
| PC 软件在环（SIL） | 快速检查纯 C++ 数学逻辑 | 已通过 | 限幅、抗饱和、100 ms/2 s 分频、FNN/RL 查表、异常回退 | 不能代表 MCU ROM、RAM 和最坏周期 |
| Wokwi 虚拟 MCU | STM32F103C8、ESP32 目标编译与外设交互 | 两目标编译/启动已通过；遥测与资源留证未完成 | 已证明完整代码可目标编译并进入运行态 | 尚未证明串口曲线、ROM/RAM 和最坏周期，也不能替代真实压缩机和传感器 |
| 实体开发板 HIL | 最终部署前测试 | 未开始 | 真实时钟、中断抖动、栈、通信和故障注入 | 仍不等于整机认证 |

## 一键执行 PC 软件在环

```powershell
cd "C:\Users\厉飞雨\Documents\New project\hvac_ai_pid_demo"
python embedded\run_mcu_validation.py
```

结果写入 `outputs/mcu_pc_sil_log.txt` 和 `outputs/mcu_validation_summary.csv`。表内明确标注 PC ABI，不能把 PC 的纳秒耗时或 exe 大小写成 STM32/ESP32 实测值。

## Wokwi 双目标工程

```powershell
python embedded\wokwi\prepare_projects.py
```

- `embedded/wokwi/stm32f103`：STM32F103C8T6（Blue Pill）验证文件。
- `embedded/wokwi/esp32`：ESP32 DevKit 验证文件。
- `embedded/wokwi/common/sketch.ino`：两种 MCU 共用的 100 ms PI / 2 s AI 调度入口。
- `embedded/hvac_pid_controller.hpp`：纯 C++ 控制核心，不访问底层寄存器。

Wokwi 右侧电路包含设定值旋钮、开门按钮、PWM 指示灯和安全回退指示灯。串口 Plotter 输出：室温、设定值、PWM 百分比、`Kp`、`Ki`、PI/AI 最坏微秒数和回退状态。

2026-08-23 在线验证记录保存在 `outputs/mcu_wokwi_validation.csv`：ESP32 完整代码编译并运行约 10.079 s；STM32F103 完整代码编译并运行约 28.500 s，并点击一次开门扰动。匿名会话中的 ESP32 串口面板未显示数据（最小串口程序也相同），STM32 串口 Plotter 面板可见但数据未能可靠读取，因此二者都只能判为“目标编译/启动通过，遥测待补”，不能判成完整验收。

## 控制核心的资源结构

- FNN：5×5 个规则后件，每个状态只对相邻 4 条规则做双线性插值；训练后参数表为 200 B float32。
- RL：运行时只带 25 B 动作索引、25 B 训练覆盖掩码和 72 B 动作表，不把完整 Q 表部署到 MCU。
- 安全 PI：输出限制、条件积分抗饱和、增益边界、增益变化率限制、输出变化率限制；输入异常或 RL 未覆盖状态时回退 IMC 参数。

## Wokwi 验收记录要求

两块目标板分别保存以下证据后，才可以把第二层状态改成“已完成”：

1. 编译成功截图和工具链/核心版本。
2. Flash/ROM、静态 RAM 的编译报告。
3. 连续运行不少于 10 分钟，PI 周期 100 ms、AI 周期 2 s 无丢拍。
4. 旋钮改变设定值、按钮注入开门扰动时，三组串口曲线同步变化。
5. NaN/越界/未覆盖 RL 状态触发回退，红灯和串口标志一致。
6. STM32 使用 DWT 或 GPIO 翻转测周期；ESP32 使用芯片周期计数器或逻辑分析仪复核最坏时延。

量产前还必须在真实硬件中加入独立于 AI 的最小启停时间、高低压与排气温度联锁、传感器断线检查、看门狗、通信超时、参数 CRC 和版本回滚。
