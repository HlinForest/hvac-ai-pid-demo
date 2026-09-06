"""Backfill full-field sealed_scenarios.csv for frozen pre-B3 batches.

B1/B2 sealed batches were generated before the scenario dump carried every
Scenario field.  Because scenario sampling is fully deterministic given
(acceptance seeds, pipeline seed), the 80 scenarios can be re-derived
exactly and the dump completed WITHOUT touching details/summary/provenance.

Safety: for every ordinal the IMC objective is re-simulated with the
recorded noise_seed and must equal the frozen details value (1e-6);
scenario keys (repeat_seed/scenario_index) must match the existing file.
Any mismatch aborts without writing.

Usage: python tools/backfill_sealed_scenarios.py --sealed <dir> [--artifact-dir <bundle>]
If --artifact-dir is omitted, IMC gains come from sealed_provenance.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict as _asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hvac_pid.controllers import PIController
from hvac_pid.metrics import calculate_metrics
from hvac_pid.simulator import simulate
from tools.run_sealed_evaluation import _sealed_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill full-field sealed scenarios")
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    sealed = args.sealed
    prov = json.loads((sealed / "sealed_provenance.json").read_text(encoding="utf-8"))
    seed = int(prov["sealed"]["pipeline_seed"])
    acc = tuple(int(v) for v in prov["sealed"]["acceptance_seeds"])
    per = int(prov["sealed"]["total"]) // len(acc)
    regen = _sealed_scenarios(seed, acc, per)
    assert len(regen) == 80, len(regen)
    old = _rows(sealed / "sealed_scenarios.csv")
    assert len(old) == 80, len(old)
    for (acc_seed, idx, _), row in zip(regen, old):
        assert int(row["repeat_seed"]) == acc_seed and int(row["scenario_index"]) == idx, row
    details = [r for r in _rows(sealed / "sealed_80_details.csv") if r["algorithm"] == "imc"]
    assert len(details) == 80
    if args.artifact_dir is not None:
        from hvac_pid.policy_bundle import PolicyBundle
        imc_gains = tuple(PolicyBundle.load(args.artifact_dir, project_root=ROOT).imc_gains)
    else:
        imc_gains = tuple(float(v) for v in prov["policies"]["imc"])
    for (acc_seed, idx, scenario), detail, row in zip(regen, details, old):
        noise = int(row["noise_seed"])
        replayed = float(calculate_metrics(simulate(scenario, PIController(*imc_gains), seed=noise))["objective"])
        assert abs(replayed - float(detail["objective"])) < 1e-6, \
            f"ordinal {row['ordinal']}: replay {replayed} vs frozen {detail['objective']}"
    out_rows: list[dict[str, object]] = []
    for ordinal, (acc_seed, idx, scenario) in enumerate(regen):
        full = _asdict(scenario)
        full.pop("FEATURE_NAMES", None)
        out_rows.append({"ordinal": ordinal + 1, "repeat_seed": acc_seed, "scenario_index": idx,
                         "noise_seed": seed + 700_000 + ordinal, **full})
    with (sealed / "sealed_scenarios.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"backfilled {sealed / 'sealed_scenarios.csv'} (80/80 IMC replays match frozen details)")


if __name__ == "__main__":
    main()
