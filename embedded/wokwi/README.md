# Wokwi 双 MCU 验证工程

运行 `python embedded/wokwi/prepare_projects.py`，把同一控制核心和 Arduino 入口复制到 `esp32`、`stm32f103` 两个项目目录。

在 Wokwi 新建相应 ESP32 DevKit 或 STM32 Blue Pill 工程，把目标目录中的 `sketch.ino`、`hvac_pid_controller.hpp` 和 `diagram.json` 放入工程。左侧可编辑算法，右侧有设定值旋钮、开门按钮、PWM 蓝灯、安全回退红灯和串口 Plotter。

- PI 每 100 ms 执行一次；FNN/RL 每 2 s 执行一次。
- 修改 `USE_RL_POLICY` 可切换 FNN 与 RL。
- 串口输出温度、设定值、PWM、Kp/Ki、最坏 PI/AI 微秒数和回退状态。
- 软件热对象按 60 倍加速，用于在几十秒内观察延迟、惯性、设定值与开门扰动；它不替代第一阶段 OpenModelica 物理基准。
- 当前快速训练的 RL Q 表访问了 25/25 个粗网格状态；这只证明状态覆盖，不证明奖励或策略已经收敛。嵌入式代码仍保留覆盖掩码接口，未来若导出未覆盖状态会保持当前增益并点亮回退灯。

在线实跑状态（2026-08-23）：ESP32 与 STM32F103 的完整算法均已通过 Wokwi 构建并进入运行态；详细证据边界见 `outputs/mcu_wokwi_validation.csv`。目前尚未取得可复核的串口曲线、目标 ROM/RAM 与最坏周期，因此仍需按 `embedded/README.md` 的清单补齐最终验收。
