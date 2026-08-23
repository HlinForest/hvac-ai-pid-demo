"""Identify FOPDT from an OpenModelica CSV step-response export.

Example:
    python extract_fopdt.py PrecisionCabinetCooling_res.csv --temperature zone.T
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def identify(time_s: np.ndarray, temperature: np.ndarray, command: np.ndarray) -> dict[str, float]:
    if len(time_s) < 8 or np.ptp(command) <= 1e-8:
        raise ValueError("CSV must include a non-zero compressor-command step")
    step_index = int(np.argmax(np.abs(np.diff(command)))) + 1
    before = slice(max(0, step_index - max(3, len(time_s) // 20)), step_index)
    after = slice(max(step_index, len(time_s) - max(3, len(time_s) // 10)), len(time_s))
    u0, u1 = float(np.mean(command[before])), float(np.mean(command[after]))
    y0, y1_tail = float(np.mean(temperature[before])), float(np.mean(temperature[after]))
    delta_y_tail = y1_tail - y0
    if abs(delta_y_tail) <= 1e-8:
        raise ValueError("temperature response is too small to identify")
    elapsed_minutes = (time_s[step_index:] - time_s[step_index]) / 60.0
    direction = float(np.sign(delta_y_tail))
    measured_change = direction * (temperature[step_index:] - y0)
    amplitude_guess = max(float(np.mean(measured_change[-max(3, len(measured_change) // 10):])), 1e-6)
    tau_guess = max(float(elapsed_minutes[-1]) / 8.0, 1e-3)

    def first_order_change(parameters: np.ndarray) -> np.ndarray:
        amplitude, tau, delay = map(float, parameters)
        elapsed = np.maximum(elapsed_minutes - delay, 0.0)
        return amplitude * (1.0 - np.exp(-elapsed / max(tau, 1e-12)))

    fitted = least_squares(
        lambda parameters: first_order_change(parameters) - measured_change,
        x0=np.asarray([amplitude_guess, tau_guess, 0.0]),
        bounds=(
            np.asarray([1e-6, 1e-3, 0.0]),
            np.asarray([max(2.0 * float(np.max(measured_change)), 2.0 * amplitude_guess), 2.0 * elapsed_minutes[-1], max(elapsed_minutes[-1] / 3.0, 1e-3)]),
        ),
        max_nfev=300,
    )
    amplitude, tau, delay = map(float, fitted.x)
    prediction = first_order_change(fitted.x)
    rmse = float(np.sqrt(np.mean(np.square(prediction - measured_change))))
    fraction = measured_change / max(amplitude, 1e-12)

    def crossing(level: float) -> float:
        indices = np.flatnonzero(fraction >= level)
        return float(elapsed_minutes[int(indices[0])]) if len(indices) else float("nan")

    t28, t63 = crossing(0.283), crossing(0.632)
    signed_gain = direction * amplitude / (u1 - u0)
    return {
        # Controller tuning uses a positive cooling-gain magnitude; preserve
        # the physical negative sign separately for traceability.
        "process_gain_c_per_u": abs(signed_gain),
        "signed_process_gain_c_per_u": signed_gain,
        "time_constant_minutes": tau,
        "delay_minutes": delay,
        "step_time_seconds": float(time_s[step_index]),
        "fit_amplitude_c": amplitude,
        "fit_rmse_c": rmse,
        "fit_evaluations": float(fitted.nfev),
        "diagnostic_t28_minutes": t28,
        "diagnostic_t63_minutes": t63,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--time", default="time")
    parser.add_argument("--temperature", default="zone.T")
    parser.add_argument("--command", default="commandInput.y")
    parser.add_argument("--kelvin", action="store_true", help="convert Modelica Kelvin output to Celsius")
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {args.time, args.temperature, args.command}.issubset(rows[0]):
        raise SystemExit(f"CSV must contain {args.time!r}, {args.temperature!r}, and {args.command!r}")
    time_s = np.asarray([float(row[args.time]) for row in rows])
    temperature = np.asarray([float(row[args.temperature]) for row in rows])
    if args.kelvin:
        temperature -= 273.15
    command = np.asarray([float(row[args.command]) for row in rows])
    print(json.dumps(identify(time_s, temperature, command), indent=2))


if __name__ == "__main__":
    main()
