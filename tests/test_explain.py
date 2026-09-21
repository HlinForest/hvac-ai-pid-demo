import numpy as np
import pytest
from dataclasses import asdict

from experiments.explain import explain
from hvac_pid.artifacts import read_json, write_json
from hvac_pid.core import Scenario
from hvac_pid.tuning import tune


def test_bo_explanation_reproduces_real_first_proposal(tmp_path):
    scenario = Scenario(gain=9, tau=25, delay=3)
    result = tune(scenario, seed=3, rounds=1)
    write_json(tmp_path / "04-bo/result.json", {**result.to_dict(), "scenario": asdict(scenario), "seed": 3})
    explain(tmp_path, "bo")
    record = read_json(tmp_path / "explain/bo.json")
    trial = result.trials[5]
    assert record["seed"] == 3 and record["scenario"] == asdict(scenario)
    assert record["gains"]["kp"] == pytest.approx(trial["kp"])
    assert record["gains"]["ki"] == pytest.approx(trial["ki"])
    assert record["ei"] == pytest.approx(trial["ei"])
    assert len(record["y"]) == 5 and record["candidate_count"] == 1024


def test_dqn_update_record_can_be_recomputed():
    torch = pytest.importorskip("torch")
    from hvac_pid.dqn import learn_batch, network
    torch.manual_seed(5)
    net, target = network(), network()
    target.requires_grad_(False)
    batch = [(np.zeros(5), 2, -3.0, np.ones(5), True),
             (np.ones(5), 1, -0.5, np.zeros(5), False)]
    details = {}
    learn_batch(net, target, torch.optim.Adam(net.parameters()), batch, details=details)
    targets = np.asarray(details["rewards"]) + 0.99 * (~np.asarray(details["done"])) * details["future"]
    assert details["targets"] == pytest.approx(targets)
    difference = np.abs(np.asarray(details["predicted"]) - targets)
    loss = np.where(difference < 1, 0.5 * difference**2, difference - 0.5).mean()
    assert details["loss"] == pytest.approx(loss)
    assert all(p.grad is None for p in target.parameters())
