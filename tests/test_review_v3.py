from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from embedded.export_policy import _accepted, _manifest_crc
from hvac_pid.ai_controllers import FNNGainController, IncrementalRLController, _state_local_samples
from hvac_pid.config import Scenario, sample_adaptive_scenarios
from hvac_pid.pipeline import _dataset_manifest_rows


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
