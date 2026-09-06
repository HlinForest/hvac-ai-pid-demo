"""P2: unified results consistency check (tables recomputable from timeseries).

Checks (fail loudly, no silent pass):
  1. manifest parses and matches hvac_pid.safety/timebase constants.
  2. sealed_80_summary recomputable from sealed_80_details (means + paired CI).
  3. supplement_metrics.json gains match the frozen source (no hard-code swap)
     and its objectives recompute under the six-substep v4 integrator.
  4. No legacy BO/Safe-BO archive read outside PolicyBundle provenance.
  5. (E2-B4 only) SHA256SUMS + evidence_index integrity, sealed-vs-history
     zero-overlap, GP attribution structure, failure-control recomputation,
     provenance comparability label.

Usage: python tools/check_results_consistency.py [--sealed-dir DIR] [--require-sealed]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
    exp_id = str(prov.get("experiment_id", ""))
    assert exp_id == "E2" or exp_id.startswith("E2-"), f"unexpected experiment_id {exp_id!r}"
    assert prov.get("code_sha") and len(str(prov["code_sha"])) >= 7, prov.get("code_sha")
    assert "worktree_status_sha16" in prov, "provenance must record worktree dirtiness"
    # comparability label must name the actual batch (B4 provenance once
    # carried a hardcoded "E2-B3" string; improve.md §4 requires uniformity).
    comparability = str(prov.get("comparability", ""))
    assert comparability.startswith(exp_id), (
        f"comparability {comparability[:20]!r} does not start with experiment_id {exp_id!r}"
    )
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
    if str(prov.get("experiment_id", "")) == "E2-B4":
        check_b4_extras(sealed_dir, prov)


def _scenario_digest_from_row(row: dict[str, str]) -> str:
    """Canonical scenario digest identical to hvac_pid.pipeline._scenario_digest.

    Rebuilds a Scenario from a sealed_scenarios.csv row (same coercion as
    tools/make_sealed_figures._scenario) and hashes its canonical JSON.
    """
    import dataclasses

    from hvac_pid.config import Scenario

    none_fields = {"door_open_hour", "setpoint_change_hour", "setpoint_after_c"}
    fields = {f.name for f in dataclasses.fields(Scenario)}
    kwargs: dict[str, object] = {}
    for key, value in row.items():
        if key not in fields or key in ("FEATURE_NAMES",):
            continue
        if value is None or value == "":
            kwargs[key] = None if key in none_fields else 0.0
            continue
        ftype = next(f.type for f in dataclasses.fields(Scenario) if f.name == key)
        kwargs[key] = int(float(value)) if "int" in str(ftype) else float(value)
    scen = Scenario(**kwargs)  # type: ignore[arg-type]
    payload = dataclasses.asdict(scen)
    payload.pop("FEATURE_NAMES", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def check_b4_extras(sealed_dir: Path, prov: dict) -> None:
    """Strict E2-B4 gates (improve.md §2): integrity + isolation + attribution."""
    # 5a. Acceptance seeds must be the fresh B4 set.
    seeds = list(prov.get("sealed", {}).get("acceptance_seeds", []))
    assert seeds == [601, 611, 621, 631, 641], f"B4 acceptance_seeds drift: {seeds}"
    # 5b. SHA256SUMS integrity: every listed file exists with matching hash,
    # and evidence_index.csv agrees (path/bytes/sha256).
    sums_path = sealed_dir / "SHA256SUMS"
    assert sums_path.exists(), "B4 missing SHA256SUMS (integrity evidence required)"
    listed: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(None, 1)
        listed[name.strip().replace("\\", "/")] = digest.strip()
    assert listed, "SHA256SUMS is empty"
    for name, want in sorted(listed.items()):
        target = sealed_dir / name
        assert target.exists(), f"SHA256SUMS lists missing file: {name}"
        got = hashlib.sha256(target.read_bytes()).hexdigest()
        assert got == want, f"hash drift for {name}: {got[:12]} != {want[:12]}"
    index_path = sealed_dir / "evidence_index.csv"
    assert index_path.exists(), "B4 missing evidence_index.csv"
    indexed = {r["path"].replace("\\", "/"): r for r in _read_csv(index_path)}
    assert set(indexed) == set(listed), (
        f"evidence_index vs SHA256SUMS mismatch: "
        f"only-index={sorted(set(indexed) - set(listed))[:3]} "
        f"only-sums={sorted(set(listed) - set(indexed))[:3]}"
    )
    for name, row in indexed.items():
        assert row["sha256"] == listed[name], f"evidence_index sha drift for {name}"
        assert int(row["bytes"]) == (sealed_dir / name).stat().st_size, f"evidence_index size drift for {name}"
    print(f"b4-integrity OK ({len(listed)} files, SHA256SUMS + evidence_index agree)")
    # 5c. Zero overlap: B4 sealed digests must not intersect train/val/qual
    # splits or any historical sealed list (B1/B2/B3 + old test partitions).
    b4_rows = _read_csv(sealed_dir / "sealed_scenarios.csv")
    b4_digests = {_scenario_digest_from_row(r) for r in b4_rows}
    assert len(b4_digests) == 80, f"B4 sealed digests not unique: {len(b4_digests)}"
    train_root = ROOT / "artifacts" / "runs" / "v4-20260906-bf6bda6"
    manifest_rows = _read_csv(train_root / "dataset_manifest.csv")
    known = {r["scenario_sha256"] for r in manifest_rows}
    overlap = b4_digests.intersection(known)
    assert not overlap, f"B4 overlaps train/val/qual/test partitions: {sorted(overlap)[:2]}"
    historical = [
        ROOT / "artifacts" / "runs" / "sealed-80x7-v4" / "sealed_scenarios.csv",
        ROOT / "artifacts" / "runs" / "sealed-80x7-v4-retrain" / "sealed_scenarios.csv",
        ROOT / "artifacts" / "runs" / "sealed-80x7-b3" / "sealed_scenarios.csv",
    ]
    for hist in historical:
        if not hist.exists():
            continue
        hist_digests = {_scenario_digest_from_row(r) for r in _read_csv(hist)}
        clash = b4_digests.intersection(hist_digests)
        assert not clash, f"B4 overlaps historical {hist.parent.name}: {sorted(clash)[:2]}"
    print(f"b4-isolation OK (80 fresh digests, zero overlap with 48/16/16 + B1/B2/B3)")
    # 5d. GP attribution structure: 4 search seeds x 3 methods + cluster PRIMARY.
    gp_path = sealed_dir / "gp_attribution.csv"
    gp_sum_path = sealed_dir / "gp_attribution_summary.json"
    assert gp_path.exists() and gp_sum_path.exists(), "B4 missing GP attribution evidence"
    gp_rows = _read_csv(gp_path)
    assert len(gp_rows) == 12, f"want 12 gp attribution rows (4 seeds x 3), got {len(gp_rows)}"
    gp_sum = json.loads(gp_sum_path.read_text(encoding="utf-8"))
    assert list(gp_sum.get("search_seeds", [])) == [815, 5001, 9002, 12077], gp_sum.get("search_seeds")
    assert int(gp_sum.get("n_pooled_pairs", 0)) == 320, gp_sum.get("n_pooled_pairs")
    for key in ("cluster_gp_minus_random", "cluster_gp_minus_init"):
        block = gp_sum.get(key, {})
        assert "PRIMARY" in str(block.get("method", "")), f"{key} must be the PRIMARY cluster interval"
        assert "lo95" in block and "hi95" in block and "mean" in block, key
    print("b4-attribution OK (12 rows, 4 search seeds, cluster PRIMARY intervals present)")
    # 5e. Failure controls: recompute summary from tests; enforce gains- naming.
    ctrl_path = sealed_dir / "failure_control_tests.csv"
    ctrl_sum_path = sealed_dir / "failure_control_summary.csv"
    assert ctrl_path.exists() and ctrl_sum_path.exists(), "B4 missing failure control evidence"
    ctrl_rows = _read_csv(ctrl_path)
    got_controls = sorted({r["control"] for r in ctrl_rows})
    assert "kp-x0.5" not in got_controls and "kp-x2.0" not in got_controls, (
        f"legacy kp-x* labels must be renamed to gains-x* (improve.md §4): {got_controls}"
    )
    assert got_controls == ["baseline-replay", "capacity-x1.5", "extend-8h", "gains-x0.5", "gains-x2.0"], got_controls
    summary_rows = _read_csv(ctrl_sum_path)
    for srow in summary_rows:
        sub = [r for r in ctrl_rows if r["control"] == srow["control"] and r["screened_cause"] == srow["screened_cause"]]
        assert len(sub) == int(srow["cases"]), f"{srow['control']}/{srow['screened_cause']} case count drift"
        assert sum(int(r["validation_passed"]) for r in sub) == int(srow["flipped_to_pass"]), (
            f"{srow['control']}/{srow['screened_cause']} flip count drift"
        )
    print(f"b4-controls OK ({len(ctrl_rows)} control runs, summary recomputed, gains-x* naming)")


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
    parser.add_argument("--require-sealed", action="store_true",
                        help="fail (instead of skip) when --sealed-dir is missing; CI uses this for the B4 main batch")
    args = parser.parse_args()
    check_manifest()
    check_no_implicit_archive()
    check_supplement()
    if args.sealed_dir.exists():
        check_sealed(args.sealed_dir)
    elif args.require_sealed:
        raise SystemExit(f"required sealed batch missing: {args.sealed_dir}")
    else:
        print(f"skip sealed check (missing {args.sealed_dir}); run `python run.py sealed ...` first")
    print("consistency: ALL OK")


if __name__ == "__main__":
    main()
