"""Reproducible sealed-batch figures for the main report (Ch5).

Re-simulates chosen sealed scenarios EXACTLY (full scenario fields from
sealed_scenarios.csv + recorded noise_seed + frozen gains) and writes:

  * fig_median_coaxial.png: median-IMC-objective scenario, 7 algorithms,
    zone temperature (top) + compressor command (bottom) on shared time axis
  * fig_failure_case.png: one validation-failed scenario (tuning cause,
    highest IMC objective among tuning failures), same layout
  * exemplar_timeseries.csv: minute + per-algorithm zone/command columns
  * exemplar_manifest.csv: ordinal/noise/gains/reason per figure

No new randomness: everything is a deterministic replay of the sealed batch.
Usage: python tools/make_sealed_figures.py --artifact-dir <bundle> --sealed <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hvac_pid.ai_controllers import FNNGainController, IncrementalRLController
from hvac_pid.config import Scenario
from hvac_pid.controllers import PIController
from hvac_pid.metrics import calculate_metrics
from hvac_pid.policy_bundle import PolicyBundle
from hvac_pid.simulator import simulate

ALGORITHMS = ("zn", "imc", "bo", "safe-bo", "fnn", "rl", "llm")
COLORS = {"zn": "#9ca3af", "imc": "#111827", "bo": "#2563eb", "safe-bo": "#17864b",
          "fnn": "#9333ea", "rl": "#e37a12", "llm": "#0e7490"}
NAMES = {"zn": "Z-N", "imc": "IMC", "bo": "BO", "safe-bo": "Safe-BO",
         "fnn": "FNN", "rl": "RL", "llm": "LLM replay"}

_NONE_FIELDS = {"door_open_hour", "setpoint_change_hour", "setpoint_after_c"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible sealed figures")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _scenario(row: dict[str, str]) -> Scenario:
    import dataclasses
    fields = {f.name for f in dataclasses.fields(Scenario)}
    kwargs: dict[str, object] = {}
    for key, value in row.items():
        if key not in fields or key in ("FEATURE_NAMES",):
            continue
        if value is None or value == "":
            kwargs[key] = None if key in _NONE_FIELDS else 0.0
            continue
        field_type = next(f.type for f in dataclasses.fields(Scenario) if f.name == key)
        kwargs[key] = int(float(value)) if "int" in str(field_type) else float(value)
    return Scenario(**kwargs)  # type: ignore[arg-type]


def _controllers(bundle: PolicyBundle, prov_policies: dict) -> dict[str, object]:
    return {
        "zn": PIController(*prov_policies["zn"]),
        "imc": PIController(*prov_policies["imc"]),
        "bo": PIController(*prov_policies["bo"]),
        "safe-bo": PIController(*prov_policies["safe-bo"]),
        "fnn": FNNGainController(tuple(bundle.imc_gains), rule_table=np.asarray(bundle.fnn_rule_table),
                                 context_coefficients=np.asarray(bundle.fnn_context)),
        "rl": IncrementalRLController(tuple(bundle.imc_gains), q_table=np.asarray(bundle.rl_q_table),
                                      covered_mask=np.asarray(bundle.rl_covered_mask)),
        "llm": PIController(*prov_policies["llm"]),
    }


def _plot(results: dict[str, object], scenario: Scenario, title: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True)
    minutes = None
    for algo in ALGORITHMS:
        result = results[algo]
        minutes = np.asarray(result.minute)
        axes[0].plot(minutes / 60.0, result.zone_c, color=COLORS[algo], lw=1.6, label=NAMES[algo])
        axes[1].plot(minutes / 60.0, np.asarray(result.command) * 100.0, color=COLORS[algo], lw=1.2)
    assert minutes is not None
    setpoint = np.asarray(results["imc"].setpoint_c)
    axes[0].plot(minutes / 60.0, setpoint, color="#111827", ls="--", lw=1.2, label="setpoint")
    axes[0].fill_between(minutes / 60.0, setpoint - 0.5, setpoint + 0.5, color="#9ca3af", alpha=0.18)
    axes[0].set_ylabel("zone / setpoint (C)")
    axes[1].set_ylabel("command (%)")
    axes[1].set_xlabel("time (h)")
    axes[0].set_title(title)
    axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = args.output or (args.sealed / "figures")
    out.mkdir(parents=True, exist_ok=True)
    bundle = PolicyBundle.load(args.artifact_dir, project_root=ROOT)
    prov = json.loads((args.sealed / "sealed_provenance.json").read_text(encoding="utf-8"))
    details = _rows(args.sealed / "sealed_80_details.csv")
    scenarios = {r["ordinal"]: r for r in _rows(args.sealed / "sealed_scenarios.csv")}
    controllers = _controllers(bundle, prov["policies"])

    imc_rows = sorted([r for r in details if r["algorithm"] == "imc"],
                      key=lambda r: float(r["objective"]))
    median_row = imc_rows[len(imc_rows) // 2]
    failed = [r for r in details
              if r["algorithm"] == "imc" and float(r["validation_passed"]) < 0.5]
    fail_row = max(failed, key=lambda r: float(r["objective"])) if failed else median_row

    manifest: list[dict[str, object]] = []
    for tag, row in (("median", median_row), ("failure", fail_row)):
        scen = _scenario(scenarios[row["ordinal"]])
        noise = int(scenarios[row["ordinal"]]["noise_seed"])
        results = {algo: simulate(scen, controllers[algo], seed=noise)  # type: ignore[arg-type]
                   for algo in ALGORITHMS}
        # Replay check: IMC objective must reproduce the sealed value.
        replayed = float(calculate_metrics(results["imc"])["objective"])
        assert abs(replayed - float(row["objective"])) < 1e-6, f"replay drift: {replayed} vs {row['objective']}"
        _plot(results, scen, f"E2 sealed ordinal={row['ordinal']} ({tag})", out / f"fig_{tag}_coaxial.png")
        minutes = np.asarray(results["imc"].minute)
        ts_path = out / f"exemplar_{tag}_timeseries.csv"
        with ts_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["minute"] + [f"{a}_zone_c" for a in ALGORITHMS]
                            + [f"{a}_command" for a in ALGORITHMS] + ["setpoint_c"])
            for i, minute in enumerate(minutes):
                writer.writerow([float(minute)]
                                + [float(results[a].zone_c[i]) for a in ALGORITHMS]
                                + [float(results[a].command[i]) for a in ALGORITHMS]
                                + [float(results["imc"].setpoint_c[i])])
        manifest.append({"figure": f"fig_{tag}_coaxial.png", "timeseries": ts_path.name,
                         "ordinal": row["ordinal"], "noise_seed": noise,
                         "imc_objective": row["objective"],
                         "validation_passed": row["validation_passed"],
                         "reason": ("median IMC objective of the 80 sealed scenarios"
                                    if tag == "median" else
                                    "highest-objective IMC validation failure (tuning-analysis case)")})
        print(f"figure {tag}: ordinal={row['ordinal']} J_imc={float(row['objective']):.2f} "
              f"replay OK -> {(out / f'fig_{tag}_coaxial.png').resolve()}")
    with (out / "exemplar_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)
    print(f"exemplars: {len(manifest)} figures + timeseries -> {out.resolve()}")


if __name__ == "__main__":
    main()
