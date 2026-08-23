"""Build and run the host-side MCU software-in-the-loop testbench.

This command validates the portable controller logic and scheduler on the PC.
It intentionally does not label host timings or binary size as STM32/ESP32 data.
"""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EMBEDDED = ROOT / "embedded"
OUTPUTS = ROOT / "outputs"
EXE = EMBEDDED / "testbench.exe"


def extract(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else default


def main() -> int:
    compiler = shutil.which("g++")
    if compiler is None:
        raise SystemExit("未找到 g++，无法执行 PC 软件在环测试。")

    compile_result = subprocess.run(
        [compiler, "-std=c++17", "-O2", str(EMBEDDED / "testbench.cpp"), "-o", str(EXE)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if compile_result.returncode != 0:
        raise SystemExit(compile_result.stderr or "C++ 编译失败。")

    run_result = subprocess.run(
        [str(EXE)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log = run_result.stdout + run_result.stderr
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "mcu_pc_sil_log.txt").write_text(log, encoding="utf-8")

    summary = {
        "验证层级": "PC 软件在环（不是 STM32/ESP32 目标板实测）",
        "结果": extract(r"MCU_SIL (PASS|FAIL)", log, "FAIL"),
        "PID周期_ms": extract(r"PID period=(\d+)ms", log),
        "AI周期_s": extract(r"AI period=(\d+)s", log),
        "AI调用次数": extract(r"AI calls=(\d+)", log),
        "SafePI状态字节_PC_ABI": extract(r"sizeof\(SafePI\)=(\d+) bytes", log),
        "虚拟对象状态字节_PC_ABI": extract(r"sizeof\(VirtualHVACPlant\)=(\d+) bytes", log),
        "RL未覆盖回退次数": extract(r"RL uncovered-state fallbacks=(\d+)", log),
        "NaN回退次数": extract(r"NaN fallback=(\d+)", log),
        "目标板ROM_RAM周期": "待 Wokwi/目标工具链编译实测",
    }
    with (OUTPUTS / "mcu_validation_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    print(log, end="")
    print(f"验证摘要: {OUTPUTS / 'mcu_validation_summary.csv'}")
    return 0 if run_result.returncode == 0 and summary["结果"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
