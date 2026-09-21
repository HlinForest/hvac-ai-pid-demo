import json
import subprocess

import numpy as np
import pytest

from experiments import pg4pi_reference
from hvac_pid.core import Gains, PI, Scenario, simulate
from hvac_pid.pg4pi import (
    GAIN_HIGH,
    GAIN_LOW,
    SensitivityPI,
    first_update_report,
    gains_from_logits,
    logits_from_gains,
    pi_mean_and_sensitivity,
    rollout,
    trajectory_gradient,
    tune,
)


def test_sensitivity_matches_conditioned_finite_difference_and_shared_pi_branch():
    # The first candidate (kp * e + ki * e * dt) is 1.2 and is rejected;
    # the retained integral then leaves an unsaturated mean of 0.8.
    gains = Gains(0.2, 1.0)
    shared = PI(gains)
    extended = SensitivityPI(gains)
    assert shared.update(4.0, 0.1) == pytest.approx(extended.update_with_sensitivity(4.0, 0.1).mean)
    assert extended.integral == 0.0
    assert extended.update_with_sensitivity(4.0, 0.1).mean == pytest.approx(0.8)

    errors = np.asarray([4.0, 0.1, 0.2, -0.2, 0.1])
    means, jacobian = pi_mean_and_sensitivity(errors, Gains(0.2, 0.1), 0.1)
    for coordinate in range(2):
        plus = np.asarray([0.2, 0.1])
        minus = plus.copy()
        plus[coordinate] += 1e-7
        minus[coordinate] -= 1e-7
        plus_means, _ = pi_mean_and_sensitivity(errors, Gains(*plus), 0.1)
        minus_means, _ = pi_mean_and_sensitivity(errors, Gains(*minus), 0.1)
        assert (plus_means - minus_means) / 2e-7 == pytest.approx(jacobian[:, coordinate], abs=1e-8)


def test_zero_noise_rollout_is_the_shared_pi_trajectory():
    scenario = Scenario(duration=5.0)
    gains = Gains(0.2, 0.01)
    pg_trace = rollout(scenario, gains, noise=np.zeros(scenario.steps))
    core_trace = simulate(scenario, gains)
    assert pg_trace.command == pytest.approx(core_trace.command)
    assert pg_trace.mean == pytest.approx(core_trace.command)
    assert pg_trace.next_temperature == pytest.approx(core_trace.next_temperature)


def test_gradient_uses_unclipped_latent_action():
    scenario = Scenario(duration=0.1)
    gains = Gains(0.2, 1.0)
    trace = rollout(scenario, gains, noise=[0.4], noise_std=0.1)
    assert trace.mean[0] == pytest.approx(0.8)
    assert trace.latent[0] == pytest.approx(1.2)
    assert trace.command[0] == pytest.approx(1.0)
    expected_score = (1.2 - 0.8) / 0.1**2
    expected = trace.return_value * expected_score * np.asarray([4.0, 0.0])
    assert trajectory_gradient(trace, 0.1) == pytest.approx(expected)


def test_log_gain_mapping_is_positive_bounded_and_invertible():
    logits = np.asarray([-0.4, 0.7])
    gains = gains_from_logits(logits)
    assert GAIN_LOW[0] < gains.kp < GAIN_HIGH[0]
    assert GAIN_LOW[1] < gains.ki < GAIN_HIGH[1]
    assert logits_from_gains(gains) == pytest.approx(logits)


def test_tune_is_reproducible_with_fixed_budget_and_keeps_each_trajectory_fixed():
    scenario = Scenario(duration=1.0)
    first = tune(scenario, seed=19, episodes=4)
    second = tune(scenario, seed=19, episodes=4)
    for left, right in zip(first.trials, second.trials):
        for key in ("episode", "return", "kp", "ki", "gradient", "gradient_logits", "noise"):
            assert left[key] == right[key]
    assert first.best == second.best
    assert first.cost["episodes"] == 4
    assert first.cost["simulations"] == 4
    assert first.cost["plant_steps"] == 4 * scenario.steps
    assert first.cost["first_update"] == first_update_report(first)
    for trial in first.trials:
        trace = rollout(scenario, Gains(trial["kp"], trial["ki"]), noise=np.zeros(scenario.steps))
        assert np.all(trace.kp == trial["kp"])
        assert np.all(trace.ki == trial["ki"])


def test_out_of_range_simc_anchor_is_explicit(monkeypatch):
    monkeypatch.setattr("hvac_pid.pg4pi.simc", lambda model: Gains(4.0, 0.1))
    with pytest.raises(ValueError, match="SIMC gains are outside"):
        tune(Scenario(duration=0.1), episodes=0)


def test_reference_driver_can_be_checked_offline(tmp_path, monkeypatch):
    # The normal entrypoint downloads a pinned checkout.  This test exercises
    # its recording path without making network access part of the test suite.
    config = json.loads(pg4pi_reference.CONFIG_PATH.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        pg4pi_reference,
        "_ensure_checkout",
        lambda source, cfg, timeout: {"status": "downloaded", "commit": cfg["commit"], "source": str(source)},
    )
    result = pg4pi_reference.run(tmp_path / "reference", execute=False)
    assert result["status"] == "prepared"
    saved = json.loads((tmp_path / "reference" / "config.json").read_text(encoding="utf-8"))
    assert saved["commit"] == config["commit"]


def test_reference_rejects_tracked_source_edits_even_when_head_is_unchanged(tmp_path):
    source = tmp_path / "upstream"
    source.mkdir()
    (source / "PG4PI.py").write_text("print('reference')\n", encoding="utf-8")
    bytecode = source / "__pycache__"
    bytecode.mkdir()
    (bytecode / "reference.pyc").write_bytes(b"before")
    git = lambda *args: subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    git("init", "--quiet")
    git("config", "user.email", "pg4pi-test@example.invalid")
    git("config", "user.name", "PG4PI test")
    git("add", "PG4PI.py", "__pycache__/reference.pyc")
    git("commit", "--quiet", "-m", "fixture")
    commit = git("rev-parse", "HEAD").stdout.strip()
    config = {"repository": "offline-fixture", "commit": commit}

    reused = pg4pi_reference._ensure_checkout(source, config, timeout=30)
    assert reused["status"] == "reused"

    (source / "PG4PI.py").write_text("print('edited reference')\n", encoding="utf-8")
    blocked = pg4pi_reference._ensure_checkout(source, config, timeout=30)
    assert blocked["status"] == "blocked"
    assert any(change["path"] == "PG4PI.py" for change in blocked["tracked_changes"])

    (source / "PG4PI.py").write_text("print('reference')\n", encoding="utf-8")
    (bytecode / "reference.pyc").write_bytes(b"runtime rewrite")
    assert pg4pi_reference._ensure_checkout(source, config, timeout=30)["status"] == "reused"
