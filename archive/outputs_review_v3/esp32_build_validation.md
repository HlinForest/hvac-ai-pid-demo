# ESP32 目标工具链编译验证

- 验证对象：`embedded/esp32_seven_algorithm_demo`
- 目标：ESP32 DevKit
- 结果：`SUCCESS`
- 静态 RAM：22,036 B / 327,680 B（6.7%）
- Flash：296,169 B / 1,310,720 B（22.6%）
- 编译耗时：19.48 s（本机墙钟，仅作构建记录）
- 策略清单：v3，编译时 CRC `0xFC5BB753`

本记录只证明当前源码可由 ESP32 官方 PlatformIO 工具链完成目标编译，且静态 RAM/Flash 未超出分区容量。它不能证明实体开发板连续运行 10 分钟、100 ms 控制周期无丢拍、板端 PI/AI 最坏执行时间、传感器可靠性、Modbus 通信或真实空调安全性。这些仍需按 `embedded/validate_esp32_serial.py` 在现场留证。
