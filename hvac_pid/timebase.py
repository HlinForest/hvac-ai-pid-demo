"""Central time-base definitions (v4 single source of truth).

Three distinct clocks exist and MUST NOT be conflated in reports:

- SIMULATION_TIME: Python closed-loop supervisory period (default 300 s = 5 min).
- DEMO_TIME:      embedded demo supervisory period (120 s).
- MCU_TIME:       MCU wall-clock AI period (2 s) with 100 ms fast PI loop.

The 200x demo virtual-time acceleration (5 sim-hours -> 90 wall-seconds)
is presentation-only.
"""
from __future__ import annotations

# Python supervisory AI period (seconds). 150x the MCU 2 s period.
SIM_SUPERVISORY_PERIOD_S: float = 300.0
SIM_SUPERVISORY_PERIOD_MIN: float = SIM_SUPERVISORY_PERIOD_S / 60.0

# Embedded-demo supervisory period (seconds). 60x the MCU 2 s period.
DEMO_SUPERVISORY_PERIOD_S: float = 120.0

# MCU wall-clock periods.
MCU_FAST_PI_PERIOD_MS: int = 100
MCU_AI_PERIOD_MS: int = 2000

# Presentation-only virtual-time acceleration for the HTML demo.
DEMO_VIRTUAL_ACCELERATION: float = 200.0

TIMEBASE_NOTE: str = (
    "Python默认AI更新周期为300 s(5 min)，嵌入式演示为120 s，"
    "MCU墙钟为2 s/100 ms分频；三者不得在报告中统一写成2 s。"
)
