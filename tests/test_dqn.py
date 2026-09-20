import numpy as np
import pytest

torch = pytest.importorskip("torch")
from hvac_pid.dqn import DQNPolicy, learn_batch, network, train
from hvac_pid.core import Gains, Scenario, evaluate


def test_target_network_is_frozen_and_terminal_targets_ignore_future():
    torch.manual_seed(0)
    net, target = network(), network()
    target.requires_grad_(False)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    before = [p.detach().clone() for p in target.parameters()]
    batch = [(np.zeros(5), 0, -2., np.ones(5), True)] * 64
    loss = learn_batch(net, target, optimizer, batch)
    assert np.isfinite(loss)
    assert all(p.grad is None for p in target.parameters())
    assert all(torch.equal(old, new) for old, new in zip(before, target.parameters()))
    assert any(p.grad is not None for p in net.parameters())


def test_dqn_training_and_frozen_roundtrip(tmp_path):
    policy, result = train([Scenario(duration=120)], episodes=5, seed=3)
    assert result["cost"]["transitions"] == 300
    assert result["cost"]["updates"] == 45
    before = {k: v.clone() for k, v in policy.net.state_dict().items()}
    reference = evaluate(Scenario(duration=4), Gains(0.3, 0.01), policy)
    assert all(torch.equal(v, policy.net.state_dict()[k]) for k, v in before.items())
    policy.save(tmp_path / "model.pt")
    restored = DQNPolicy.load(tmp_path / "model.pt")
    assert evaluate(Scenario(duration=4), Gains(0.3, 0.01), restored).iae == reference.iae
