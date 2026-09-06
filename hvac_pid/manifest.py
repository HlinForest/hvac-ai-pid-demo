"""v4 manifest: the single executable source of truth.

Every run directory must contain a frozen copy (``manifest.yaml``) plus
``environment.json`` and ``provenance.csv``.  This module is the only place
that parses ``experiments/manifests/v4.yaml``; missing fields fail loudly
instead of falling back to hidden defaults.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "experiments" / "manifests" / "v4.yaml"

REQUIRED_TOP = (
    "schema_version", "code", "env", "timebase", "safety",
    "scenarios", "algorithms", "thresholds", "artifacts",
    "modelica", "llm", "hardware_gates", "acceptance",
)
REQUIRED_SAFETY = (
    "max_fractional_gain_change", "kp_bounds", "ki_bounds",
    "risk_weight", "max_undershoot_c", "validation_tolerance",
    "max_risk_ratio_to_baseline", "repeats",
)
REQUIRED_TIMEBASE = (
    "sim_supervisory_period_s", "demo_supervisory_period_s",
    "mcu_ai_period_ms", "mcu_fast_pi_period_ms",
)
REQUIRED_THRESHOLDS = (
    "physical_cross_validation_max_abs_c",
    "fopdt_normalized_rmse_pct",
    "openmodelica_rmse_c",
    "openmodelica_max_abs_c",
)


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(f"manifest missing required field '{where}.{key}'")
    return mapping[key]


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load and strictly validate the v4 manifest."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest root must be a mapping: {path}")
    for key in REQUIRED_TOP:
        _require(data, key, "manifest")
    if int(data["schema_version"]) != 4:
        raise ValueError(f"unsupported schema_version: {data.get('schema_version')!r} (want 4)")
    # safety / timebase / thresholds must be complete; no silent defaults.
    safety = data["safety"]
    for key in REQUIRED_SAFETY:
        _require(safety, key, "safety")
    timebase = data["timebase"]
    for key in REQUIRED_TIMEBASE:
        _require(timebase, key, "timebase")
    thresholds = data["thresholds"]
    for key in REQUIRED_THRESHOLDS:
        _require(thresholds, key, "thresholds")
    # algorithms: canonical order + per-family specs.
    algos = data["algorithms"]
    order = _require(algos, "order", "algorithms")
    if not isinstance(order, list) or sorted(order) != sorted(
        ["zn", "imc", "bo", "safe-bo", "fnn", "rl", "llm"]
    ):
        raise ValueError(f"algorithms.order must list the 7 canonical ids, got {order!r}")
    specs = _require(algos, "specs", "algorithms")
    for key in ("fnn", "rl"):
        _require(specs, key, "algorithms.specs")
    # scenarios: train/validation/test + sealed definition.
    scenarios = data["scenarios"]
    for key in ("train", "validation", "test", "sealed_5x16"):
        _require(scenarios, key, "scenarios")
    for key in ("train", "validation", "test"):
        for sub in ("count", "seed", "duration_hours"):
            _require(scenarios[key], sub, f"scenarios.{key}")
    sealed = scenarios["sealed_5x16"]
    for sub in ("repeats", "per_repeat", "total"):
        _require(sealed, sub, "scenarios.sealed_5x16")
    if int(sealed["repeats"]) * int(sealed["per_repeat"]) != int(sealed["total"]):
        raise ValueError("scenarios.sealed_5x16 repeats*per_repeat must equal total")
    # hardware gates: explicit items + rule.
    gates = data["hardware_gates"]
    _require(gates, "items", "hardware_gates")
    _require(gates, "rule", "hardware_gates")
    if not isinstance(gates["items"], list) or not gates["items"]:
        raise ValueError("hardware_gates.items must be a non-empty list")
    artifacts = data["artifacts"]
    _require(artifacts, "root", "artifacts")
    _require(artifacts, "layout", "artifacts")
    _require(artifacts, "rule", "artifacts")
    data["_manifest_path"] = str(path.resolve())
    data["_manifest_sha256"] = manifest_sha256(path)
    return data


def manifest_sha256(path: str | Path = DEFAULT_MANIFEST) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head_sha() -> str:
    """Actual executing commit; never the frozen baseline SHA."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=10,
        )
        sha = out.stdout.strip()
        return sha if sha else "unknown-no-git"
    except Exception:
        return "unknown-no-git"


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_timebase_consistency(manifest: dict[str, Any]) -> None:
    """Cross-check manifest timebase against hvac_pid.timebase constants."""
    from . import timebase as tb

    tb_map = {
        "sim_supervisory_period_s": tb.SIM_SUPERVISORY_PERIOD_S,
        "demo_supervisory_period_s": tb.DEMO_SUPERVISORY_PERIOD_S,
        "mcu_ai_period_ms": tb.MCU_AI_PERIOD_MS,
        "mcu_fast_pi_period_ms": tb.MCU_FAST_PI_PERIOD_MS,
    }
    mismatches = []
    for key, code_value in tb_map.items():
        manifest_value = float(manifest["timebase"][key])
        if abs(float(code_value) - manifest_value) > 1e-9:
            mismatches.append(f"{key}: code={code_value} manifest={manifest_value}")
    if mismatches:
        raise ValueError("timebase drift between manifest and hvac_pid.timebase: " + "; ".join(mismatches))


def check_safety_consistency(manifest: dict[str, Any]) -> None:
    """Cross-check manifest safety against hvac_pid.safety constants."""
    from . import safety as s

    pairs = [
        ("max_fractional_gain_change", s.MAX_FRACTIONAL_GAIN_CHANGE),
        ("risk_weight", s.RISK_WEIGHT),
        ("max_undershoot_c", s.MAX_UNDERSHOOT_C),
        ("validation_tolerance", s.VALIDATION_TOLERANCE),
        ("max_risk_ratio_to_baseline", s.MAX_RISK_RATIO_TO_BASELINE),
        ("repeats", s.SAFETY_REPEATS),
    ]
    mismatches = []
    for key, code_value in pairs:
        if abs(float(manifest["safety"][key]) - float(code_value)) > 1e-12:
            mismatches.append(f"safety.{key}: code={code_value} manifest={manifest['safety'][key]}")
    if tuple(manifest["safety"]["kp_bounds"]) != tuple(s.KP_BOUNDS):
        mismatches.append(f"safety.kp_bounds: code={s.KP_BOUNDS} manifest={manifest['safety']['kp_bounds']}")
    if tuple(manifest["safety"]["ki_bounds"]) != tuple(s.KI_BOUNDS):
        mismatches.append(f"safety.ki_bounds: code={s.KI_BOUNDS} manifest={manifest['safety']['ki_bounds']}")
    if mismatches:
        raise ValueError("safety drift between manifest and hvac_pid.safety: " + "; ".join(mismatches))
