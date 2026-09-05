# ESP32 七算法固件目标编译记录

- 日期：2026-08-30（加入 LLM Agent 自动整定后回归）
- 命令：`python -m platformio run`
- PlatformIO Core：6.1.19
- 平台：Espressif 32 7.0.1
- 开发板：`esp32dev`（ESP32 Dev Module，240 MHz、320 KB RAM、4 MB Flash）
- Arduino 框架：`framework-arduinoespressif32 3.20017.241212+sha.dcc1105b`
- Xtensa 工具链：`8.4.0+2021r2-patch5`
- 结果：SUCCESS
- 静态 RAM：22,024 / 327,680 B（6.7%）
- Flash：294,425 / 1,310,720 B（22.5%）

Windows 的旧 Xtensa 工具链不能可靠处理中文源码路径，因此验证时把仓库临时映射为 ASCII 盘符 `P:` 后编译。源代码和构建输入没有改变；这项路径要求已写入固件 README。

该记录证明新版七算法固件可以由真实 ESP32 工具链完成编译和链接，但不证明目标板连续运行、串口通信、实际 WCET 或真实空调响应。后四项必须连接开发板后运行：

```powershell
python embedded/validate_esp32_serial.py --port COM3 --seconds 600
```
