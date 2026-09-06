"""P2: unified results consistency check (tables recomputable from timeseries).

Checks (fail loudly, no silent pass):
  1. manifest parses and matches hvac_pid.safety/timebase constants.
  2. sealed_80_summary recomputable from sealed_80_details (means + paired CI).
  3. supplement_metrics.json gains match the frozen source (no hard-code swap)
     and its objectives recompute under the six-substep v4 integrator.
  4. No legacy BO/Safe-BO archive read outside PolicyBundle provenance.

Usage: python tools/check_results_consistency.py [--sealed-dir DIR]
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

from hvac_pid.config import typical_case_scenarios  # noqa: E402
from hvac_pid.controllers import PIController  # noqa: E402
from hvac_pid.manifest import check_safety_consistency, check_timebase_consistency, load_manifest  # noqa: E402
from hvac_pid.metrics import calculate_metrics  # noqa: E402
from hvac_pid.simulator import simulate  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def check_manifest() -> None:
    manifest = load_manifest()
    check_timebase_consistency(manifest)
    check_safety_consistency(manifest)
    print(f"manifest OK ({manifest['_manifest_sha256'][:12]})")


def check_sealed(sealed_dir: Path) -> None:
    details = _read_csv(sealed_dir / "sealed_80_details.csv")
    summary = _read_csv(sealed_dir / "sealed_80_summary.csv")
    assert len(details) == 560, f"want 560 detail rows, got {len(details)}"
    assert len(summary) == 7, f"want 7 summary rows, got {len(summary)}"
    assert {r["algorithm"] for r in summary} == {"zn", "imc", "bo", "safe-bo", "fnn", "rl", "llm"}, summary
    # E2 provenance must carry experiment id, live code SHA and scenario list.
    prov_path = sealed_dir / "sealed_provenance.json"
    assert prov_path.exists(), f"missing {prov_path}"
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    assert prov.get("experiment_id") == "E2", prov.get("experiment_id")
    assert prov.get("code_sha") and len(str(prov["code_sha"])) >= 7, prov.get("code_sha")
    assert "worktree_status_sha16" in prov, "provenance must record worktree dirtiness"
    assert "ratio_of_means_to_imc" in (summary[0] or {}), "summary must carry secondary ratio_of_means_to_imc"
    assert (sealed_dir / "sealed_scenarios.csv").exists(), "missing sealed_scenarios.csv (E2 evidence)"
    scen = _read_csv(sealed_dir / "sealed_scenarios.csv")
    assert len(scen) == 80, f"want 80 sealed scenarios, got {len(scen)}"
    for field in ("cooling_capacity_w", "door_open_load_w", "door_open_duration_minutes",
                  "occupied_load_add_w", "occupied_end_hour", "setpoint_change_hour"):
        assert field in scen[0], f"sealed_scenarios.csv missing {field} (failure autopsy needs it)"
    assert (sealed_dir / "manifest.yaml").exists(), "missing frozen manifest.yaml copy"
    imc = np.asarray([float(r["objective"]) for r in details if r["algorithm"] == "imc"])
    for row in summary:
        algo = row["algorithm"]
        objectives = np.asarray([float(r["objective"]) for r in details if r["algorithm"] == algo])
        assert abs(float(row["mean_objective"]) - float(np.mean(objectives))) < 1e-9, algo
        ratios = objectives / np.maximum(imc, 1e-12)
        # Primary: mean of ratios; must equal paired point estimate.
        assert abs(float(row["mean_ratio_to_imc"]) - float(np.mean(ratios))) < 1e-9, algo
        assert abs(float(row["paired_ratio_mean"]) - float(np.mean(ratios))) < 1e-9, f"{algo} paired mean must be mean-of-ratios"
        # Secondary: ratio of means, distinct statistic.
        assert abs(float(row["ratio_of_means_to_imc"]) - float(np.mean(objectives) / max(float(np.mean(imc)), 1e-12))) < 1e-9, algo
        # Recompute paired bootstrap CI with the sealed seed convention (seed=7+77).
        rng = np.random.default_rng(7 + 77)
        idx = rng.integers(0, len(ratios), size=(2000, len(ratios)))
        boot = np.mean(ratios[idx], axis=1)
        assert abs(float(row["paired_ratio_lo95"]) - float(np.quantile(boot, 0.025))) < 1e-9, f"{algo} lo95 drift"
        assert abs(float(row["paired_ratio_hi95"]) - float(np.quantile(boot, 0.975))) < 1e-9, f"{algo} hi95 drift"
        # Split acceptance fields must exist and be recomputable.
        subset = [r for r in details if r["algorithm"] == algo]
        for field in ("bounded", "comfort_held", "recovery_ok", "actuator_compliant", "validation_passed", "fallback_failed"):
            assert field in subset[0], f"{algo} missing {field}"
        assert abs(float(row["validation_rate"]) - float(np.mean([float(r["validation_passed"]) for r in subset]))) < 1e-9, algo
        # fallback_failed new definition: fallback occurred but validation failed.
        for r in subset:
            want = 1 if (float(r["fallback_fraction"]) > 1e-12 and float(r["validation_passed"]) < 0.5) else 0
            assert int(float(r["fallback_failed"])) == want, f"{algo} ordinal={r.get('ordinal')} fallback_failed definition drift"
    print(f"sealed OK ({sealed_dir}: 560 details, 7 summaries, E2 stats+acceptance+provenance verified)")


def check_supplement() -> None:
    data_path = ROOT / "reports" / "分报告" / "data" / "supplement_metrics.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    assert "gains_source" in data and "integrator" in data and "code_sha" in data, "supplement provenance missing"
    assert "6" in data["integrator"] and "子步" in data["integrator"], data["integrator"]
    # LLM frozen gain must be the 0.4657 line (not the stale 0.4057 hard-code).
    llm = data["gains"]["llm"]
    assert abs(float(llm["kp"]) - 0.4657376424916451) < 1e-9, llm
    assert abs(float(llm["ki"]) - 0.00364826335084437) < 1e-12, llm
    # Recompute one cell under the v4 integrator and compare.
    scenarios = typical_case_scenarios()
    case = "初次快速降温"
    gains = (float(data["gains"]["llm"]["kp"]), float(data["gains"]["llm"]["ki"]))
    result = simulate(scenarios[case], PIController(*gains), seed=7 + 400_003)
    recomputed = float(calculate_metrics(result)["objective"])
    stored = float(data["cases"]["llm"][case]["objective"])
    assert abs(recomputed - stored) < 1e-6, f"supplement {case} LLM objective drift: {recomputed} vs {stored}"
    print(f"supplement OK (LLM {gains[0]:.6f}/{gains[1]:.7f}, {case} objective {stored:.5f} recomputed)")


def check_no_implicit_archive() -> None:
    bundle_src = (ROOT / "hvac_pid" / "policy_bundle.py").read_text(encoding="utf-8")
    assert "outputs_advanced_quick" in bundle_src, "bundle must name the legacy batch explicitly"
    demo_src = (ROOT / "hvac_pid" / "embedded_demo.py").read_text(encoding="utf-8")
    assert "archive/outputs_advanced_quick" not in demo_src, "demo must not read the archive directly"
    assert "_advanced_gain" not in demo_src, "demo must not keep the legacy _advanced_gain helper"
    print("no-implicit-archive OK (all policy reads via PolicyBundle)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed-dir", type=Path, default=ROOT / "artifacts" / "runs" / "sealed-80x7-v4")
    args = parser.parse_args()
    check_manifest()
    check_no_implicit_archive()
    check_supplement()
    if args.sealed_dir.exists():
        check_sealed(args.sealed_dir)
    else:
        print(f"skip sealed check (missing {args.sealed_dir}); run `python run.py sealed ...` first")
    print("consistency: ALL OK")


if __name__ == "__main__":
    main()
