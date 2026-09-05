"""Collect and grade real ESP32 telemetry for the ten-minute acceptance run.

This script intentionally refuses to manufacture target evidence.  Connect a
flashed board, then run for example::

    python embedded/validate_esp32_serial.py --port COM3 --seconds 600
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=Path("archive/outputs_embedded_demo/esp32_target_acceptance.csv"))
    args = parser.parse_args()
    try:
        import serial  # type: ignore
    except ImportError as exc:
        raise SystemExit("缺少 pyserial，请先运行 pip install pyserial") from exc

    rows: list[dict[str, object]] = []
    started = time.monotonic()
    with serial.Serial(args.port, args.baud, timeout=1.5) as device:
        device.reset_input_buffer()
        while time.monotonic() - started < args.seconds:
            line = device.readline().decode("utf-8", errors="replace").strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["host_elapsed_s"] = time.monotonic() - started
            rows.append(row)

    if not rows:
        raise SystemExit("没有收到有效 JSON 遥测，未生成任何通过结论。")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    finite = all(math.isfinite(float(row["temperature_c"])) for row in rows)
    duration_ok = float(rows[-1]["host_elapsed_s"]) >= args.seconds - 2.0
    period_ok = max(int(row["missed_periods"]) for row in rows) == 0
    pi_ok = max(int(row["pi_wcet_us"]) for row in rows) < 10_000
    ai_ok = max(int(row["ai_wcet_us"]) for row in rows) < 100_000
    passed = finite and duration_ok and period_ok and pi_ok and ai_ok
    summary = {
        "evidence_level": "physical ESP32 serial telemetry",
        "passed": passed,
        "samples": len(rows),
        "duration_s": float(rows[-1]["host_elapsed_s"]),
        "max_missed_periods": max(int(row["missed_periods"]) for row in rows),
        "max_pi_wcet_us": max(int(row["pi_wcet_us"]) for row in rows),
        "max_ai_wcet_us": max(int(row["ai_wcet_us"]) for row in rows),
        "finite_temperature": finite,
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
