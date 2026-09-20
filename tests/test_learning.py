import numpy as np
import pytest

from experiments.data import make_splits
from hvac_pid.core import Gains, Scenario, evaluate
from hvac_pid.fnn import FNN, features
from hvac_pid.rl import QPolicy, normalized_observation, state_index, train, update_q
from hvac_pid.tuning import Identified, expected_improvement, initial_samples, tune


def test_bo_budget_initial_points_and_reproducibility():
    s = Scenario(duration=4)
    a, b = tune(s, seed=4, rounds=2), tune(s, seed=4, rounds=2)
    assert len(a.trials) == 7
    assert a.cost["simulations"] == 7
    assert a.best == b.best
    assert [(t["kp"], t["ki"], t["iae"]) for t in a.trials] == [(t["kp"], t["ki"], t["iae"]) for t in b.trials]
    assert a.trials[0]["source"] == "ZN"
    assert a.trials[1]["source"] == "SIMC"
    assert all(t["ei"] >= 0 for t in a.trials[5:])


def test_ei_zero_variance_limit():
    assert expected_improvement(np.asarray([1., 3.]), np.zeros(2), 2) == pytest.approx([1, 0])


def test_classical_seed_outside_proposal_bounds_is_observed_not_clipped():
    result = tune(Scenario(gain=8, tau=32, delay=1, duration=4), rounds=1)
    assert result.trials[0]["kp"] == pytest.approx(3.6)
    assert result.trials[0]["ki"] > 0.5
    assert 0.01 <= result.trials[-1]["kp"] <= 3
    assert 0.0001 <= result.trials[-1]["ki"] <= 0.5


def test_splits_are_disjoint_and_reproducible():
    data = make_splits()
    assert data == make_splits()
    sets = [{(s["gain"], s["tau"], s["delay"]) for s in data["splits"][name]}
            for name in ["train", "validation", "test"]]
    assert list(map(len, sets)) == [48, 12, 12]
    assert len(set.union(*sets)) == 72


def test_fnn_learns_positive_gains_and_model_roundtrip(tmp_path):
    x = np.asarray([[8, 12, 1], [10, 20, 3], [12, 32, 5]])
    y = np.asarray([[0.2, 0.01], [0.4, 0.02], [0.8, 0.03]])
    f = features(x)
    assert f.shape == (3, 27)
    assert f.sum(axis=1) == pytest.approx(np.ones(3))
    model = FNN.fit(x, y, ridge=1e-6)
    predictions = np.asarray([[g.kp, g.ki] for g in (model.predict(Identified(*r)) for r in x)])
    assert predictions == pytest.approx(y, rel=0.005)
    model.save(tmp_path / "model.npz")
    assert FNN.load(tmp_path / "model.npz").predict(Identified(*x[1])) == model.predict(Identified(*x[1]))


def test_terminal_q_update_does_not_bootstrap():
    q = np.zeros((5, 5, 3, 9, 9))
    state, next_state = (1, 1, 1, 4), (2, 2, 1, 4)
    q[next_state] = 100
    update_q(q, state, 0, -3, next_state, True, alpha=0.5)
    assert q[state + (0,)] == -1.5
    update_q(q, state, 1, -3, next_state, False, alpha=0.5, gamma=0.9)
    assert q[state + (1,)] == 43.5


def test_q_evaluation_does_not_learn_and_training_repeats(tmp_path):
    scenarios = [Scenario(duration=4)]
    a, report = train(scenarios, episodes=3, seed=1)
    b, _ = train(scenarios, episodes=3, seed=1)
    assert np.array_equal(a.q, b.q)
    assert report["cost"]["transitions"] == 6
    before = a.q.copy()
    evaluate(scenarios[0], Gains(0.3, 0.01), a)
    assert np.array_equal(a.q, before)
    a.save(tmp_path / "q.npz")
    assert np.array_equal(QPolicy.load(tmp_path / "q.npz").q, a.q)


def test_observations_include_current_gains_without_truth():
    obs = np.asarray([0.3, -0.04, 0.5, 2., 0.5])
    assert normalized_observation(obs).shape == (5,)
    assert state_index(obs)[-1] == 6
