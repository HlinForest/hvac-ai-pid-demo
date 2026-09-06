"""Controlled-variable tests for sealed acceptance failures (E2 failure chapter).

Heuristic screening (analyze_failures.py) is NOT causation.  For every IMC
validation failure in --sealed, re-simulate with ONE factor changed:

  * extend: duration 5 h -> 8 h (tests the window hypothesis: failures caused
    only by a short observation window should flip to pass)
  * capacity-x1.5: cooling_capacity_w x 1.5 (tests the capacity hypothesis)
  * kp-x0.5 / kp-x2.0: IMC gains scaled (clipped to safety bounds; tests the
    tuning hypothesis: a genuine tuning shortfall may respond to gain moves)

Saturation and start/stop cycling are measured from the replayed baseline
timeseries (command at limits, compressor events).  A flip to
validation_passed=1 under exactly one control supports that cause; no flip
anywhere leaves the case as "unresolved (tuning suspected, not proven)".

Usage: python tools/failure_control_tests.py --artifact-dir <bundle> --sealed <dir>
Writes failure_control_tests.csv + failure_control_summary.csv into --sealed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hvac_pid.controllers import PIController
from hvac_pid.metrics import calculate_metrics
from hvac_pid.policy_bundle import PolicyBundle
from hvac_pid.simulator import simulate
from tools.make_sealed_figures import _scenario

CONTROLS = ("baseline-replay", "extend-8h", "capacity-x1.5", "kp-x0.5", "kp-x2.0")
KP_BOUNDS = (0.002, 1.5)
KI_BOUNDS = (1e-5, 0.08)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Failure controlled-variable tests")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _clip(gains: tuple[float, float]) -> tuple[float, float]:
    return (float(min(max(gains[0], KP_BOUNDS[0]), KP_BOUNDS[1])),
            float(min(max(gains[1], KI_BOUNDS[0]), KI_BOUNDS[1])))


def main() -> None:
    args = parse_args()
    bundle = PolicyBundle.load(args.artifact_dir, project_root=ROOT)
    imc = tuple(bundle.imc_gains)
    details = {(r["algorithm"], r["ordinal"]): r
               for r in _rows(args.sealed / "sealed_80_details.csv")}
    scenarios = {r["ordinal"]: r for r in _rows(args.sealed / "sealed_scenarios.csv")}
    failures = [r for r in _rows(args.sealed / "failure_cases.csv") if r["algorithm"] == "imc"]
    assert failures, "no IMC failures to test"
    rows: list[dict[str, object]] = []
    for case in failures:
        scen = _scenario(scenarios[case["ordinal"]])
        noise = int(scenarios[case["ordinal"]]["noise_seed"])
        frozen = details[("imc", case["ordinal"])]
        variants: dict[str, tuple[object, tuple[float, float]]] = {
            "baseline-replay": (scen, imc),
            "extend-8h": (replace(scen, duration_hours=8.0), imc),
            "capacity-x1.5": (replace(scen, cooling_capacity_w=scen.cooling_capacity_w * 1.5), imc),
            "kp-x0.5": (scen, _clip((imc[0] * 0.5, imc[1] * 0.5))),
            "kp-x2.0": (scen, _clip((imc[0] * 2.0, imc[1] * 2.0))),
        }
        for control, (scene, gains) in variants.items():
            result = simulate(scene, PIController(*gains), seed=noise)
            metrics = calculate_metrics(result)
            cmd = np.asarray(result.command)
            saturated = float(np.mean((cmd >= 0.999) | (cmd <= 0.001)))
            if control == "baseline-replay":
                assert abs(float(metrics["objective"]) - float(frozen["objective"])) < 1e-6, \
                    f"ordinal {case['ordinal']}: replay drift"
            rows.append({
                "ordinal": case["ordinal"], "screened_cause": case["cause"],
                "control": control, "kp": gains[0], "ki": gains[1],
                "validation_passed": int(metrics["validation_passed"] > 0.5),
                "comfort_held": int(metrics["comfort_held"] > 0.5),
                "recovery_ok": int(metrics["recovery_ok"] > 0.5),
                "objective": round(float(metrics["objective"]), 4),
                "rmse_c": round(float(metrics["rmse_c"]), 4),
                "max_undershoot_c": round(float(metrics["max_undershoot_c"]), 4),
                "saturated_fraction": round(saturated, 4),
                "compressor_starts": int(metrics["compressor_start_events"]),
                "compressor_stops": int(metrics["compressor_stop_events"]),
            })
    with (args.sealed / "failure_control_tests.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary: list[dict[str, object]] = []
    for control in CONTROLS:
        sub = [r for r in rows if r["control"] == control]
        for cause in ("capacity", "window", "tuning"):
            screened = [r for r in sub if r["screened_cause"] == cause]
            summary.append({
                "control": control, "screened_cause": cause,
                "cases": len(screened),
                "flipped_to_pass": sum(int(r["validation_passed"]) for r in screened),
            })
    with (args.sealed / "failure_control_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"control tests: {len(rows)} runs -> {(args.sealed / 'failure_control_tests.csv').resolve()}")
    for row in summary:
        print(f"  {row['control']:15s} {row['screened_cause']:8s} flipped {row['flipped_to_pass']}/{row['cases']}")
    print(json.dumps({"note": "flip under exactly one control supports that cause; "
                              "no flip anywhere = unresolved (tuning suspected, not proven)"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
