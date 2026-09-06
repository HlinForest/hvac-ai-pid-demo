"""B4 delivery gates (improve.md): LF-stable integrity + repro comparison.

Fast, no simulation: exercises tools/check_results_consistency helpers on
tmp copies of the committed B4 evidence.
"""
from __future__ import annotations

import csv
import hashlib
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_results_consistency import (  # noqa: E402
    _read_csv,
    check_repro_against_reference,
)

B4 = ROOT / "artifacts" / "runs" / "sealed-80x7-b4"
SEALED_FILES = ("sealed_80_details.csv", "sealed_80_summary.csv",
                "sealed_provenance.json", "sealed_scenarios.csv", "manifest.yaml")


def _stage_copy(tmp_path: Path) -> Path:
    assert B4.exists(), "B4 evidence batch missing"
    for name in SEALED_FILES:
        shutil.copy(B4 / name, tmp_path / name)
    return tmp_path


def test_b4_hash_index_matches_file_bytes():
    """SHA256SUMS + evidence_index agree with committed B4 bytes (LF-stable)."""
    listed: dict[str, str] = {}
    for line in (B4 / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            listed[name.strip()] = digest.strip()
    assert listed, "SHA256SUMS is empty"
    for name, want in sorted(listed.items()):
        assert "\r" not in name
        raw = (B4 / name).read_bytes()
        if Path(name).suffix.lower() not in (".png", ".npy"):
            assert b"\r" not in raw, f"{name} carries CRLF (evidence text must be LF)"
        assert hashlib.sha256(raw).hexdigest() == want, f"hash drift for {name}"
    indexed = {r["path"]: r for r in _read_csv(B4 / "evidence_index.csv")}
    assert set(indexed) == set(listed)
    for name, row in indexed.items():
        assert row["sha256"] == listed[name]
        assert int(row["bytes"]) == (B4 / name).stat().st_size


def test_repro_matches_reference(tmp_path):
    staged = _stage_copy(tmp_path)
    check_repro_against_reference(staged, B4)  # must not raise


def test_repro_drift_is_caught(tmp_path):
    staged = _stage_copy(tmp_path)
    rows = list(csv.DictReader(open(staged / "sealed_80_details.csv", encoding="utf-8-sig")))
    rows[0]["objective"] = str(float(rows[0]["objective"]) + 1e-6)
    with open(staged / "sealed_80_details.csv", "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(AssertionError):
        check_repro_against_reference(staged, B4)
