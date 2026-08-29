# ESP32 七算法固件

这是目标板工程，不是直接驱动压缩机功率器件的程序。ESP32 输出的是经过最低频率、斜率、量化和最小启停约束后的容量请求。

```powershell
cd embedded/esp32_seven_algorithm_demo
pio run
pio run --target upload
pio device monitor --baud 115200
```

如果 Windows 用户目录或项目目录含中文，而 Xtensa 工具链报告 `Invalid argument`，请先映射纯 ASCII 盘符再编译：

```powershell
subst P: "C:\Users\你的用户名\Documents\New project\hvac_ai_pid_demo"
P:
cd \embedded\esp32_seven_algorithm_demo
pio run
```

串口命令：

```text
ALGO zn|imc|bo|safe-bo|fnn|rl|llm
MODE demo|modbus
SETPOINT 24.0
KNOB 0|1
DISTURB 0|1
RESET
STATUS
```

`MODE demo` 使用加速虚拟热对象；90 秒墙钟对应 5 小时模拟时间。`MODE modbus` 使用 `../modbus_config.hpp`。在取得真实设备寄存器表并逐项核对前，必须保持 `kWritesEnabled=false`。RS-485 收发器需要将 RO/DI/RE/DE 接到配置的 UART 与方向引脚，ESP32 不能直接接到 A/B 差分总线。

故障回退包括：参数清单 CRC 不匹配、非法温度、Modbus 超时、设备告警、策略未验收或 RL 状态未覆盖。遥测是单行 JSON，可由 `streamlit_app.py` 读取。
