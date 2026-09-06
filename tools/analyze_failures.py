"""Acceptance-failure autopsy for a sealed batch (E2-B3 §5).

For every (algorithm, scenario) with ``validation_passed < 0.5``, assigns ONE
primary cause from scenario physics + recorded metrics (no new simulation):

  * capacity: peak heat load vs cooling capacity
    (internal + door pulse + occupancy add) / capacity > 0.85
    -> physically marginal regardless of tuning
  * window: last disturbance ends < 60 min before the run ends
    -> the 60-min in-band observation cannot complete
  * tuning: capacity and window both OK but comfort/recovery still fail
    -> residual controller shortfall (with failing sub-checks noted)

Also reports the objective-vs-acceptance gap: mean J of failed vs passed
runs per algorithm (a small gap means "J尚可但未通过", i.e. the objective
and the four-gate acceptance disagree).

Outputs (inside --sealed): failure_cases.csv, failure_cause_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

WINDOW_H = 1.0
LOAD_RATIO_GATE = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sealed acceptance-failure autopsy")
    parser.add_argument("--sealed", type=Path, required=True)
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def main() -> None:
    args = parse_args()
    details = _rows(args.sealed / "sealed_80_details.csv")
    scenarios = {(r["repeat_seed"], r["scenario_index"]): r
                 for r in _rows(args.sealed / "sealed_scenarios.csv")}
    cases: list[dict[str, object]] = []
    for row in details:
        if _f(row.get("validation_passed")) > 0.5:
            continue
        scen = scenarios.get((row["repeat_seed"], row["scenario_index"]), {})
        cap = _f(scen.get("cooling_capacity_w"), 7000.0)
        peak = (_f(scen.get("internal_load_w")) + _f(scen.get("door_open_load_w"))
                + _f(scen.get("occupied_load_add_w")))
        load_ratio = peak / max(cap, 1e-9)
        duration = _f(scen.get("duration_hours"), 5.0)
        events = [_f(scen.get("door_open_hour"), -1.0) + _f(scen.get("door_open_duration_minutes")) / 60.0,
                  _f(scen.get("setpoint_change_hour"), -1.0),
                  _f(scen.get("occupied_end_hour"), -1.0)]
        last_event = max([e for e in events if e >= 0.0], default=0.0)
        window_left = duration - last_event
        failed = [k for k in ("comfort_held", "recovery_ok", "bounded", "actuator_compliant")
                  if _f(row.get(k), 1.0) < 0.5]
        if load_ratio > LOAD_RATIO_GATE:
            cause = "capacity"
        elif window_left < WINDOW_H:
            cause = "window"
        else:
            cause = "tuning"
        cases.append({
            "algorithm": row["algorithm"], "ordinal": row["ordinal"],
            "repeat_seed": row["repeat_seed"], "scenario_index": row["scenario_index"],
            "objective": row["objective"], "rmse_c": row.get("rmse_c", ""),
            "max_undershoot_c": row.get("max_undershoot_c", ""),
            "failed_checks": "+".join(failed),
            "load_ratio": round(load_ratio, 3), "window_left_h": round(window_left, 2),
            "cause": cause,
        })
    with (args.sealed / "failure_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(cases)
    algos = sorted({r["algorithm"] for r in details})
    summary: list[dict[str, object]] = []
    for algo in algos:
        sub = [r for r in details if r["algorithm"] == algo]
        failed = [c for c in cases if c["algorithm"] == algo]
        passed_j = [float(r["objective"]) for r in sub if _f(r.get("validation_passed")) > 0.5]
        failed_j = [float(r["objective"]) for r in sub if _f(r.get("validation_passed")) < 0.5]
        summary.append({
            "algorithm": algo, "scenarios": len(sub),
            "failed": len(failed),
            "cause_capacity": sum(1 for c in failed if c["cause"] == "capacity"),
            "cause_window": sum(1 for c in failed if c["cause"] == "window"),
            "cause_tuning": sum(1 for c in failed if c["cause"] == "tuning"),
            "mean_J_passed": round(sum(passed_j) / max(len(passed_j), 1), 2),
            "mean_J_failed": round(sum(failed_j) / max(len(failed_j), 1), 2),
        })
    with (args.sealed / "failure_cause_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(f"failures: {len(cases)} cases -> {(args.sealed / 'failure_cases.csv').resolve()}")
    for row in summary:
        print(f"  {row['algorithm']:7s} failed={row['failed']:>2} "
              f"(capacity={row['cause_capacity']} window={row['cause_window']} tuning={row['cause_tuning']}) "
              f"J passed={row['mean_J_passed']} failed={row['mean_J_failed']}")


if __name__ == "__main__":
    main()
