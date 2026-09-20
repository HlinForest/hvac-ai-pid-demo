from dataclasses import replace
import math

import numpy as np
import pytest

from hvac_pid.core import Gains, PI, Plant, Scenario, TuningEnv, evaluate, score, simulate
from hvac_pid.tuning import Identified, identify, simc, zn


@pytest.mark.parametrize("delay", [0, 0.1, 2.0, 5.0])
def test_step_response_matches_analytic_solution(delay):
    s = Scenario(initial=30, delay=delay)
    plant = Plant(s)
    for i in range(500):
        actual = plant.step(0.5, 0)
        elapsed = max(0.0, (i + 1) * s.dt - delay)
        expected = 30 - 4 * (1 - math.exp(-elapsed / 20))
        assert actual == pytest.approx(expected, abs=1e-11)


def test_pi_saturation_and_gain_changes_preserve_integral_contribution():
    pi = PI(Gains(0.5, 0.1))
    assert pi.update(10, 0.1) == 1
    assert pi.integral == 0
    assert pi.update(0.2, 1) == pytest.approx(0.12)
    contribution = pi.integral
    pi.gains = Gains(0.25, 0.5)
    assert pi.update(0, 1) == pytest.approx(contribution)
    assert pi.update(-100, 1) == 0
    assert pi.integral == contribution


def test_zero_gains_and_metric_alignment():
    s = Scenario(duration=1)
    trace = simulate(s, Gains(0, 0))
    assert trace.temperature[0] == 28
    assert trace.time[-1] == pytest.approx(0.9)
    assert trace.next_temperature[:-1] == pytest.approx(trace.temperature[1:])
    independent = sum(abs(v - 24) * 0.1 for v in trace.temperature)
    assert score(trace).iae == pytest.approx(independent)
    assert np.all(trace.command == 0)


@pytest.mark.parametrize("gain,tau,delay", [(8, 20, 2), (10, 30, 5), (9.2, 12.5, 1.1)])
def test_identification_recovers_parameters(gain, tau, delay):
    fitted, _, _ = identify(Scenario(gain=gain, tau=tau, delay=delay))
    assert [fitted.gain, fitted.tau, fitted.delay] == pytest.approx([gain, tau, delay], abs=1e-5)


def test_classical_formula_values_and_no_hidden_gain_clipping():
    m = Identified(8, 20, 2)
    assert zn(m).kp == pytest.approx(1.125)
    assert zn(m).ki == pytest.approx(1.125 / 6.66)
    assert simc(m).kp == pytest.approx(20 / (8 * (20 / 3 + 2)))
    assert zn(Identified(8, 20, 0.1)).kp > 3
    with pytest.raises(ValueError):
        zn(Identified(8, 20, 0))


def test_env_rewards_equal_simulator_iae_and_absolute_gain_actions():
    s = Scenario(duration=5)
    anchor = Gains(0.3, 0.015)
    env = TuningEnv(s, anchor)
    observation = env.reset()
    total, done = 0, False
    while not done:
        observation, reward, done = env.step(8)
        total += reward
        assert observation[-2:] == pytest.approx([2, 2])
    assert env.index == 50  # Last decision interval is only one minute.
    assert -total == pytest.approx(score(env.trace()).iae)
    assert env.trace().command == pytest.approx(simulate(s, Gains(0.6, 0.03)).command)
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset()
    assert env.pi.integral == 0
    assert env.observation()[-2:] == pytest.approx([1, 1])


def test_holdout_changes_do_not_require_changing_frozen_controller():
    gains = simc(Identified(8, 20, 2))
    for s in [Scenario(), replace(Scenario(), delay=5), replace(Scenario(), disturbance=1.5)]:
        result = evaluate(s, gains)
        assert np.isfinite(result.iae)
        assert np.all(result.trace.kp == gains.kp)
