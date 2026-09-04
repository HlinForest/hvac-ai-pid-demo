"""Build and run the host-side MCU software-in-the-loop testbench.

This command validates the portable controller logic and scheduler on the PC.
It intentionally does not label host timings or binary size as STM32/ESP32 data.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EMBEDDED = ROOT / "embedded"
EXE = EMBEDDED / "testbench.exe"


def extract(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "outputs_review_v3")
    args = parser.parse_args()
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
    evidence_dir = args.artifact_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "mcu_pc_sil_log.txt").write_text(log, encoding="utf-8")

    expected_path = args.artifact_dir / "policy_parity_vectors.csv"
    parity_rows = [line.split(",") for line in log.splitlines() if line.startswith("PARITY,")]
    parity_max_error = float("inf")
    parity_pass = False
    if expected_path.exists() and parity_rows:
        with expected_path.open("r", encoding="utf-8-sig", newline="") as handle:
            expected = list(csv.DictReader(handle))
        if len(expected) == len(parity_rows):
            errors: list[float] = []
            flags_ok = True
            for wanted, actual in zip(expected, parity_rows, strict=True):
                numeric_actual = list(map(float, actual[1:]))
                wanted_values = [
                    float(wanted["error_c"]), float(wanted["error_rate_c_per_min"]),
                    float(wanted["applied_command"]), float(wanted["integral_state"]),
                    float(wanted["outdoor_delta_c"]), float(wanted["load_fraction"]),
                    float(wanted["fnn_kp"]), float(wanted["fnn_ki"]), float(wanted["rl_covered"]),
                    float(wanted["rl_kp"]), float(wanted["rl_ki"]),
                ]
                errors.extend(abs(left-right) for left, right in zip(wanted_values, numeric_actual, strict=True))
                flags_ok = flags_ok and int(wanted_values[8]) == int(numeric_actual[8])
            parity_max_error = max(errors, default=0.0)
            parity_pass = flags_ok and parity_max_error <= 1e-5
    # Gate: the testbench fails the run when the RL uncovered-state fallback
    # fraction exceeds 10% (systematic state-grid misses, e.g. a time-base
    # bug).  Re-derive the fraction here so the summary and the exit code both
    # fail closed even if the testbench binary were stale or hand-edited.
    fallback_events = extract(r"RL uncovered-state fallbacks=(\d+)/", log)
    rl_calls = extract(r"RL uncovered-state fallbacks=\d+/(\d+)", log)
    fallback_fraction_pct = None
    if fallback_events and rl_calls:
        fallback_fraction_pct = 100.0 * int(fallback_events) / int(rl_calls)
    rl_coverage_gate_pct = 10.0
    rl_coverage_pass = fallback_fraction_pct is not None and fallback_fraction_pct <= rl_coverage_gate_pct

    summary = {
        "验证层级": "PC 软件在环（不是 STM32/ESP32 目标板实测）",
        "结果": extract(r"MCU_SIL (PASS|FAIL)", log, "FAIL"),
        "PID周期_ms": extract(r"PID period=(\d+)ms", log),
        "AI周期_s": extract(r"AI period=(\d+)s", log),
        "AI调用次数": extract(r"AI calls=(\d+)", log),
        "SafePI状态字节_PC_ABI": extract(r"sizeof\(SafePI\)=(\d+) bytes", log),
        "虚拟对象状态字节_PC_ABI": extract(r"sizeof\(VirtualHVACPlant\)=(\d+) bytes", log),
        "RL未覆盖回退次数": extract(r"RL uncovered-state fallbacks=(\d+)", log),
        "RL未覆盖回退占比百分比": (
            round(fallback_fraction_pct, 4) if fallback_fraction_pct is not None else "nan"
        ),
        "RL覆盖率门_百分比": rl_coverage_gate_pct,
        "RL覆盖率门结果": "PASS" if rl_coverage_pass else "FAIL",
        "NaN回退次数": extract(r"NaN fallback=(\d+)", log),
        "策略清单版本": "3" if extract(r"manifest v3=(\d+)", log) == "1" else "INVALID",
        "Python_CPP一致性向量": len(parity_rows),
        "Python_CPP最大绝对误差": parity_max_error,
        "Python_CPP一致性结果": "PASS" if parity_pass else "FAIL",
        "目标板ROM_RAM周期": (
            "ESP32编译通过；RAM/Flash见esp32_build_validation.md；实体板周期待实测"
            if (evidence_dir / "esp32_build_validation.md").exists()
            else "待目标工具链编译与实体板周期实测"
        ),
    }
    with (evidence_dir / "mcu_validation_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    print(log, end="")
    print(f"验证摘要: {evidence_dir / 'mcu_validation_summary.csv'}")
    return 0 if (
        run_result.returncode == 0
        and summary["结果"] == "PASS"
        and parity_pass
        and rl_coverage_pass
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
