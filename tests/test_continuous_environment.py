import numpy as np
import pytest

from hvac_pid.core import ACTION_SCALES, Gains, Scenario, TuningEnv, evaluate, score


class ConstantPolicy:
    action_mode = "continuous"

    def __init__(self, action):
        self.action = np.asarray(action)

    def choose(self, observation):
        return self.action.copy()


def test_zero_action_is_same_fixed_pi_including_partial_interval():
    s, g = Scenario(duration=5.1), Gains(0.3, 0.015)
    fixed = evaluate(s, g)
    online = evaluate(s, g, ConstantPolicy([0, 0]))
    np.testing.assert_array_equal(fixed.trace.temperature, online.trace.temperature)
    np.testing.assert_array_equal(fixed.trace.command, online.trace.command)
    env = TuningEnv(s, g)
    total, done = 0.0, False
    while not done:
        _, reward, done = env.step_continuous([0, 0])
        total += reward
    assert total == pytest.approx(-score(env.trace()).iae)
    assert env.index == 51
    with pytest.raises(RuntimeError):
        env.step_continuous([0, 0])


@pytest.mark.parametrize("action", range(9))
def test_continuous_and_discrete_matching_actions(action):
    a = TuningEnv(Scenario(duration=2), Gains(0.3, 0.015))
    b = TuningEnv(a.scenario, a.anchor)
    obs, reward, done = a.step(action)
    other, other_reward, other_done = b.step_continuous(np.log2(ACTION_SCALES[action]))
    np.testing.assert_array_equal(obs, other)
    assert reward == other_reward and done == other_done


@pytest.mark.parametrize("action", [[2, 0], [np.nan, 0], [np.inf, 0], [0], [[0, 0]]])
def test_invalid_continuous_action_never_runs_a_controller(action):
    env = TuningEnv(Scenario(), Gains(0.3, 0.015))
    with pytest.raises(ValueError):
        env.step_continuous(action)
    assert env.index == 0 and env.rows == []
