# 硬件门（E5-target）：状态与复现命令

> 铁律：仿真/SIL/主机计时永远不能把硬件门翻成“通过”。门状态只认目标硅片证据文件，
> 审计脚本 `python tools/check_hardware_gates.py --run <run>` 为准。

| 门 | 状态 | 目标证据文件 | 复现命令 |
|---|---|---|---|
| 实体板 10 分钟遥测 | 待外部验证 | `esp32_target_acceptance.csv`（板端 JSON 遥测，≥598 s、零 `missed_periods`、有限温度） | `python embedded/validate_esp32_serial.py --port COM3 --seconds 600`（需接板；无板时脚本拒绝生成通过结论） |
| 真实（目标）WCET | 待外部验证 | `target_wcet.csv`（硅片实测 PI<10 ms、AI<100 ms，方法与编译选项记录） | 板端 `pi_wcet_us`/`ai_wcet_us` 实测后归档；`artifacts/.../host_wcet.csv` 仅为主机参考，不得代入 |
| HIL 闭环 + Modbus 写入 | 待外部验证 | `hil_acceptance.csv`（HIL 回路 + 写入确认） | 外部 HIL 台架执行后归档 |

## 软件侧已完成（非硬件证据，仅证明软件一致性）

- PC-SIL：`artifacts/runs/v4-20260907-8f3fd5f/mcu_pc_sil_log.txt`（PASS，75 组对拍，CRC 一致，100 ms/2 s 分频，RL 覆盖门）。
- 主机 WCET 参考：`artifacts/runs/v4-20260907-8f3fd5f/host_wcet.csv`（PC Python PI/FNN/RL 单步 ~10–230 µs；PC testbench PI 300 ns / AI 42.2 µs；均为 HOST-ONLY）。
- ESP32 目标编译：历史记录（RAM/Flash 占用），本次重训后待刷新（`embedded/esp32_seven_algorithm_demo/`，需 Arduino/PlatformIO 工具链）。

## HIL 干跑（无硬件时只验证配置，不产生通过结论）

```bash
python tools/check_hardware_gates.py --run artifacts/runs/v4-20260907-8f3fd5f
# 预期：3/3 PENDING；若有人把上表写成“已完成”，先修文档再谈发布。
```
