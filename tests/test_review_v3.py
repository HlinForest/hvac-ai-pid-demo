from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from embedded.export_policy import _accepted, _manifest_crc
from hvac_pid.ai_controllers import FNNGainController, IncrementalRLController, _state_local_samples
from hvac_pid.config import Scenario, sample_adaptive_scenarios
from hvac_pid.pipeline import _dataset_manifest_rows


ROOT = Path(__file__).resolve().parents[1]


def test_three_way_manifest_has_stable_unique_hashes() -> None:
    train = sample_adaptive_scenarios(6, seed=1)
    validation = sample_adaptive_scenarios(2, seed=2)
    test = sample_adaptive_scenarios(2, seed=3)
    first = _dataset_manifest_rows(
        [("train", 1, train), ("validation", 2, validation), ("test", 3, test)]
    )
    second = _dataset_manifest_rows(
        [("train", 1, train), ("validation", 2, validation), ("test", 3, test)]
    )
    assert first == second
    assert {row["split"] for row in first} == {"train", "validation", "test"}
    assert len({row["scenario_sha256"] for row in first}) == len(first)


def test_fnn_local_labels_include_deployable_context() -> None:
    scenario = Scenario(duration_hours=0.5, sensor_noise_std_c=0.04)
    rows = _state_local_samples(scenario, (0.4, 0.003), (0.45, 0.0035), seed=5)
    assert rows
    required = {
        "applied_command", "integral_state", "outdoor_delta_c", "load_fraction",
        "label_kp", "label_ki", "local_rollout_score", "behaviour_policy",
    }
    assert required <= rows[0].keys()
    assert len({row["behaviour_policy"] for row in rows}) == 3
    assert all(np.isfinite(row["local_rollout_score"]) for row in rows)


def test_uncovered_rl_state_is_fail_closed_without_argmax_bias() -> None:
    q = np.zeros((5, 5, 3, 9), dtype=float)
    q[:, :, :, 0] = 99.0
    covered = np.zeros((5, 5, 3), dtype=bool)
    controller = IncrementalRLController((0.4, 0.004), q_table=q, covered_mask=covered)
    controller.update(3.0, 1.0, minute=0.0, applied_command=0.4)
    assert controller.kp == 0.4
    assert controller.ki == 0.004
    assert controller.diagnostics()["fallback_active"]


def test_acceptance_nan_missing_and_invalid_values_fail_closed() -> None:
    assert not _accepted({})
    assert not _accepted({"deployment_accepted": "nan"})
    assert not _accepted({"deployment_accepted": "broken"})
    assert not _accepted({"deployment_accepted": "0"})
    assert _accepted({"deployment_accepted": "1"})


def test_manifest_crc_v3_covers_every_deployed_category() -> None:
    fnn = np.ones((5, 5, 2), dtype=float) * np.asarray([0.4, 0.004])
    context = np.zeros((2, 4), dtype=float)
    policy = np.full((5, 5, 3), 4, dtype=np.uint8)
    covered = np.ones((5, 5, 3), dtype=np.uint8)

    def crc(**updates: object) -> int:
        values = {
            "fnn_accepted": True,
            "rl_accepted": True,
            "fallback": np.asarray([0.4, 0.004]),
            "fnn": fnn,
            "fnn_context": context,
            "policy": policy,
            "covered": covered,
        }
        values.update(updates)
        return _manifest_crc(**values)  # type: ignore[arg-type]

    baseline = crc()
    changed_fnn = fnn.copy(); changed_fnn[0, 0, 0] += 1e-3
    changed_context = context.copy(); changed_context[0, 0] = 0.01
    changed_policy = policy.copy(); changed_policy[0, 0, 0] = 2
    changed_covered = covered.copy(); changed_covered[0, 0, 0] = 0
    assert len({
        baseline,
        crc(fnn_accepted=False),
        crc(rl_accepted=False),
        crc(fallback=np.asarray([0.41, 0.004])),
        crc(fnn=changed_fnn),
        crc(fnn_context=changed_context),
        crc(policy=changed_policy),
        crc(covered=changed_covered),
    }) == 8


def test_pc_sil_enforces_rl_coverage_gate(tmp_path: Path) -> None:
    # End-to-end: rebuild the PC SIL testbench, run it, and require the
    # RL uncovered-fallback gate (<=10%) in the summary and the exit code.
    # A supervisory-interval time-base regression (wall vs simulated time)
    # pushes the fallback fraction to ~83% and must fail this test.
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable on this runner")
    parity_source = ROOT / "archive/outputs_review_v3" / "policy_parity_vectors.csv"
    if not parity_source.exists():
        pytest.skip("policy parity vectors not generated")
    shutil.copy2(parity_source, tmp_path / "policy_parity_vectors.csv")
    result = subprocess.run(
        [sys.executable, str(ROOT / "embedded" / "run_mcu_validation.py"),
         "--artifact-dir", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    summary_path = tmp_path / "mcu_validation_summary.csv"
    summary_text = (
        summary_path.read_text(encoding="utf-8-sig", errors="replace")
        if summary_path.exists()
        else "<no mcu_validation_summary.csv was written>"
    )
    assert result.returncode == 0, result.stdout[-2000:] + "\n--- summary ---\n" + summary_text
    with (tmp_path / "mcu_validation_summary.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["结果"] == "PASS"
    assert row["RL覆盖率门结果"] == "PASS"
    assert float(row["RL未覆盖回退占比百分比"]) <= float(row["RL覆盖率门_百分比"])
    assert row["Python_CPP一致性结果"] == "PASS"
