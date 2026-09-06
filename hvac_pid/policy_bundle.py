"""Unified PolicyBundle: one loader for pipeline, demo and export.

P0 fix for “七算法演示没有统一加载同一套策略”:

* BO / Safe-BO no longer read ``archive/outputs_advanced_quick`` behind the
  caller's back; they resolve inside the given ``artifact_dir`` first and
  only fall back to the legacy batch with an explicit ``legacy_fallback``
  flag recorded in provenance.
* FNN always loads **both** the 5x5 rule table and the (2,4) context
  coefficients; missing context is an error, not a silent zero table.
* RL always loads **both** the Q table and the (5,5,3) coverage mask; the
  mask gates deployment exactly as the sealed acceptance does.
* IMC fallback gains come from the selected ``imc_lambda_tuning.csv`` row
  inside the same ``artifact_dir``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LEGACY_ADVANCED_DIR = Path("archive/outputs_advanced_quick")
LEGACY_REVIEW_DIR = Path("archive/outputs_review_v3")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_selected_imc(artifact_dir: Path) -> tuple[tuple[float, float], str, str]:
    path = artifact_dir / "imc_lambda_tuning.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"IMC fallback missing: {path} not found. "
            "Run the v4 pipeline first; refusing to guess gains."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(float(row.get("selected", 0))) == 1:
                gains = (float(row["candidate_kp"]), float(row["candidate_ki"]))
                return gains, str(path), _sha256_file(path)[:16]
    raise ValueError(f"no selected==1 row in {path}")


def _read_bo_gains(
    artifact_dir: Path, project_root: Path
) -> tuple[tuple[float, float], str, str, bool]:
    """Return (gains, source_path, sha16, legacy_fallback)."""
    canonical = artifact_dir / "global_bayesian_tuning.csv"
    if canonical.exists():
        with canonical.open("r", encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle))
            return (
                (float(row["kp"]), float(row["ki"])),
                str(canonical),
                _sha256_file(canonical)[:16],
                False,
            )
    # Transitional: explicit per-run BO policy JSON (written by new pipeline).
    policy_json = artifact_dir / "bo_policy.json"
    if policy_json.exists():
        data = json.loads(policy_json.read_text(encoding="utf-8"))
        return (
            (float(data["kp"]), float(data["ki"])),
            str(policy_json),
            _sha256_file(policy_json)[:16],
            False,
        )
    legacy = project_root / LEGACY_ADVANCED_DIR / "advanced_holdout_summary.csv"
    if legacy.exists():
        warnings.warn(
            f"BO gains fall back to legacy {legacy}; re-run pipeline to freeze "
            "them inside artifact_dir.",
            UserWarning,
            stacklevel=3,
        )
        with legacy.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("method") == "ordinary BO":
                    return (
                        (float(row["kp"]), float(row["ki"])),
                        str(legacy),
                        _sha256_file(legacy)[:16],
                        True,
                    )
    raise FileNotFoundError(
        f"BO gains missing in {artifact_dir} (want global_bayesian_tuning.csv "
        f"or bo_policy.json); legacy {legacy} also absent."
    )


def _read_safe_bo_gains(
    artifact_dir: Path, project_root: Path
) -> tuple[tuple[float, float], str, str, bool]:
    canonical_json = artifact_dir / "safe_bo_policy.json"
    if canonical_json.exists():
        data = json.loads(canonical_json.read_text(encoding="utf-8"))
        return (
            (float(data["kp"]), float(data["ki"])),
            str(canonical_json),
            _sha256_file(canonical_json)[:16],
            False,
        )
    canonical_hist = artifact_dir / "safe_bo_history.csv"
    if canonical_hist.exists():
        with canonical_hist.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        selected = [r for r in rows if int(float(r.get("selected", 0))) == 1]
        row = selected[-1] if selected else rows[-1]
        return (
            (float(row["candidate_kp"]), float(row["candidate_ki"])),
            str(canonical_hist),
            _sha256_file(canonical_hist)[:16],
            False,
        )
    legacy = project_root / LEGACY_ADVANCED_DIR / "advanced_holdout_summary.csv"
    if legacy.exists():
        warnings.warn(
            f"Safe-BO gains fall back to legacy {legacy}; re-run pipeline to freeze "
            "them inside artifact_dir.",
            UserWarning,
            stacklevel=3,
        )
        with legacy.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("method") == "risk-aware safe BO":
                    return (
                        (float(row["kp"]), float(row["ki"])),
                        str(legacy),
                        _sha256_file(legacy)[:16],
                        True,
                    )
    raise FileNotFoundError(
        f"Safe-BO gains missing in {artifact_dir} (want safe_bo_policy.json "
        f"or safe_bo_history.csv); legacy {legacy} also absent."
    )


def _load_npy(path: Path, *, shape: tuple[int, ...], what: str) -> tuple[np.ndarray, str]:
    if not path.exists():
        raise FileNotFoundError(f"{what} missing: {path} not found")
    array = np.load(path)
    if array.shape != shape:
        raise ValueError(f"{what} shape {array.shape} != {shape} ({path})")
    return array, _sha256_file(path)[:16]


def _load_fnn(artifact_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    table_path = artifact_dir / "fnn_rule_table.npy"
    context_path = artifact_dir / "fnn_context_coefficients.npy"
    table, table_sha = _load_npy(table_path, shape=(5, 5, 2), what="FNN rule table")
    if not context_path.exists():
        raise FileNotFoundError(
            f"FNN context coefficients missing: {context_path} not found. "
            "The demo must load the (2,4) residual coefficients together with "
            "the rule table; refusing to silently use zeros."
        )
    context, context_sha = _load_npy(
        context_path, shape=(2, 4), what="FNN context coefficients"
    )
    return table, context, {
        "fnn_rule_table": f"{table_path}#{table_sha}",
        "fnn_context": f"{context_path}#{context_sha}",
    }


def _load_rl(artifact_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    table_path = artifact_dir / "rl_q_table.npy"
    table, table_sha = _load_npy(table_path, shape=(5, 5, 3, 9), what="RL Q table")
    mask_path = artifact_dir / "rl_covered_mask.npy"
    if mask_path.exists():
        mask = np.load(mask_path)
        if mask.shape != (5, 5, 3):
            raise ValueError(f"RL mask shape {mask.shape} != (5,5,3) ({mask_path})")
        mask = np.asarray(mask, dtype=bool)
        mask_sha = _sha256_file(mask_path)[:16]
        return table, mask, {
            "rl_q_table": f"{table_path}#{table_sha}",
            "rl_mask": f"{mask_path}#{mask_sha}",
        }
    # Transitional reconstruction from transitions (same logic as pipeline).
    transitions = artifact_dir / "rl_training_transitions.csv"
    if transitions.exists():
        warnings.warn(
            f"RL mask {mask_path} absent; reconstructing from {transitions}. "
            "Re-run pipeline to freeze rl_covered_mask.npy.",
            UserWarning,
            stacklevel=3,
        )
        mask = np.zeros((5, 5, 3), dtype=bool)
        with transitions.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                mask[
                    int(float(row["state_error_bin"])),
                    int(float(row["state_delta_bin"])),
                    int(float(row["state_command_bin"])),
                ] = True
        return table, mask, {
            "rl_q_table": f"{table_path}#{table_sha}",
            "rl_mask": f"{transitions}#reconstructed",
        }
    raise FileNotFoundError(
        f"RL coverage mask missing: {mask_path} and {transitions} both absent."
    )


def _deployment_accepted(artifact_dir: Path, name: str) -> bool:
    path = artifact_dir / name
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return False
    try:
        return float(rows[-1].get("deployment_accepted", 0.0)) > 0.5
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class PolicyBundle:
    """All seven deployable policies resolved from one artifact dir."""

    artifact_dir: Path
    imc_gains: tuple[float, float]
    bo_gains: tuple[float, float]
    safe_bo_gains: tuple[float, float]
    fnn_rule_table: np.ndarray = field(repr=False)
    fnn_context: np.ndarray = field(repr=False)
    rl_q_table: np.ndarray = field(repr=False)
    rl_covered_mask: np.ndarray = field(repr=False)
    fnn_accepted: bool = False
    rl_accepted: bool = False
    provenance: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(
        cls, artifact_dir: str | Path, *, project_root: str | Path | None = None
    ) -> "PolicyBundle":
        artifact_dir = Path(artifact_dir)
        if project_root is None:
            project_root = Path(__file__).resolve().parents[1]
        project_root = Path(project_root)
        if not artifact_dir.exists():
            raise FileNotFoundError(f"artifact_dir not found: {artifact_dir}")
        imc_gains, imc_src, imc_sha = _read_selected_imc(artifact_dir)
        bo_gains, bo_src, bo_sha, bo_legacy = _read_bo_gains(artifact_dir, project_root)
        safe_gains, safe_src, safe_sha, safe_legacy = _read_safe_bo_gains(
            artifact_dir, project_root
        )
        fnn_table, fnn_context, fnn_prov = _load_fnn(artifact_dir)
        rl_table, rl_mask, rl_prov = _load_rl(artifact_dir)
        provenance = {
            "artifact_dir": str(artifact_dir.resolve()),
            "imc": f"{imc_src}#{imc_sha}",
            "bo": f"{bo_src}#{bo_sha}",
            "bo_legacy_fallback": str(int(bo_legacy)),
            "safe_bo": f"{safe_src}#{safe_sha}",
            "safe_bo_legacy_fallback": str(int(safe_legacy)),
            **fnn_prov,
            **rl_prov,
        }
        return cls(
            artifact_dir=artifact_dir,
            imc_gains=(float(imc_gains[0]), float(imc_gains[1])),
            bo_gains=(float(bo_gains[0]), float(bo_gains[1])),
            safe_bo_gains=(float(safe_gains[0]), float(safe_gains[1])),
            fnn_rule_table=fnn_table,
            fnn_context=fnn_context,
            rl_q_table=rl_table,
            rl_covered_mask=rl_mask,
            fnn_accepted=_deployment_accepted(artifact_dir, "fnn_training_history.csv"),
            rl_accepted=_deployment_accepted(artifact_dir, "rl_training_history.csv"),
            provenance=provenance,
        )

    def summary(self) -> dict[str, object]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "imc_gains": list(self.imc_gains),
            "bo_gains": list(self.bo_gains),
            "safe_bo_gains": list(self.safe_bo_gains),
            "fnn_accepted": int(self.fnn_accepted),
            "rl_accepted": int(self.rl_accepted),
            "fnn_context_norm": float(np.linalg.norm(self.fnn_context)),
            "rl_coverage_pct": float(100.0 * np.mean(self.rl_covered_mask)),
            **{f"src_{k}": v for k, v in self.provenance.items()},
        }
