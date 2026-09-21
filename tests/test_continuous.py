import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hvac_pid.continuous import (  # noqa: E402
    BatchRenorm1d,
    CrossQCritic,
    ReplayBuffer,
    as_action,
    normalized_observation,
    squashed_gaussian,
    TransitionBatch,
)
from hvac_pid.core import Gains, Scenario, evaluate  # noqa: E402
from hvac_pid.crossq import update_crossq  # noqa: E402
from hvac_pid.ppo import compute_gae, ppo_loss, train as ppo_train  # noqa: E402
from hvac_pid.sac import train as sac_train, SACPolicy, update_sac  # noqa: E402
from hvac_pid.td3 import (train as td3_train, TD3Policy, update_td3,
                          network as td3_network)  # noqa: E402
from hvac_pid.crossq import (train as crossq_train, CrossQPolicy,
                             update_crossq)  # noqa: E402


def test_replay_normalizes_raw_observations_once_and_actions_are_bounded():
    replay = ReplayBuffer(2)
    raw = np.asarray([4.0, 0.2, 1.0, 2.0, 2.0])
    replay.push(raw, [0.0, 0.5], -1.0, raw)
    sample = replay.sample(1, np.random.default_rng(2))
    np.testing.assert_array_equal(sample.states[0], normalized_observation(raw))
    with pytest.raises(ValueError):
        as_action([1.1, 0.0])


def test_gae_bootstraps_truncation_but_not_terminal():
    rewards = [1.0, 2.0]
    values = [0.5, 0.5]
    next_values = [0.5, 3.0]
    terminal_advantage, _ = compute_gae(
        rewards, values, next_values, [False, True], [False, False]
    )
    truncated_advantage, _ = compute_gae(
        rewards, values, next_values, [False, False], [False, True]
    )
    assert terminal_advantage[1] == pytest.approx(1.5)
    assert truncated_advantage[1] == pytest.approx(4.47)
    assert truncated_advantage[0] != terminal_advantage[0]


def test_tanh_gaussian_log_probability_is_finite_at_saturation():
    mean = torch.zeros(2)
    log_std = torch.zeros(2)
    action, log_prob = squashed_gaussian(torch.tensor([20.0, -20.0]), mean, log_std)
    assert torch.all(torch.isfinite(action))
    assert torch.isfinite(log_prob)


def test_ppo_clip_uses_the_clipped_ratio_for_positive_and_negative_advantages():
    total, details = ppo_loss(
        torch.tensor([np.log(2.0), np.log(0.5)]), torch.zeros(2),
        torch.tensor([1.0, -1.0]), torch.zeros(2), torch.zeros(2), torch.zeros(2),
    )
    # Positive advantages are clipped at 1.2; a negative advantage selects
    # the lower (unclipped) objective when the ratio is too large.
    assert details["ratio_min"].item() == pytest.approx(0.5)
    assert details["ratio_max"].item() == pytest.approx(2.0)
    assert details["policy_loss"].item() == pytest.approx(-0.2)


def _batch(size=64):
    rng = np.random.default_rng(7)
    return TransitionBatch(
        rng.normal(size=(size, 5)).astype(np.float32),
        rng.uniform(-1, 1, size=(size, 2)).astype(np.float32),
        rng.normal(size=size).astype(np.float32),
        rng.normal(size=(size, 5)).astype(np.float32),
        np.zeros(size, dtype=np.bool_),
        np.zeros(size, dtype=np.bool_),
    )


def test_td3_actor_is_delayed_and_target_parameters_are_frozen():
    torch.manual_seed(5)
    actor, target_actor = td3_network(), td3_network()
    critics = __import__("hvac_pid.continuous", fromlist=["TwinCritic"]).TwinCritic()
    target_critics = __import__("hvac_pid.continuous", fromlist=["TwinCritic"]).TwinCritic()
    target_actor.load_state_dict(actor.state_dict())
    target_critics.load_state_dict(critics.state_dict())
    target_actor.requires_grad_(False)
    target_critics.requires_grad_(False)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critics.parameters(), lr=3e-4)
    actor_before = [p.detach().clone() for p in actor.parameters()]
    update_td3(actor, critics, target_actor, target_critics, actor_optimizer,
               critic_optimizer, _batch(), update_index=1)
    assert all(torch.equal(a, b) for a, b in zip(actor_before, actor.parameters()))
    update_td3(actor, critics, target_actor, target_critics, actor_optimizer,
               critic_optimizer, _batch(), update_index=2)
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, actor.parameters()))
    assert all(p.grad is None for p in target_actor.parameters())
    assert all(p.grad is None for p in target_critics.parameters())


def test_sac_update_changes_entropy_temperature_and_terminal_target_does_not_bootstrap():
    from hvac_pid.continuous import GaussianActor, TwinCritic
    from hvac_pid.sac import sac_target_q

    actor, critics, targets = GaussianActor(), TwinCritic(), TwinCritic()
    targets.load_state_dict(critics.state_dict())
    targets.requires_grad_(False)
    alpha = torch.tensor(1.0)
    batch = _batch()
    states = torch.as_tensor(batch.states)
    rewards = torch.full((len(batch.rewards),), -2.0)
    terminated = torch.ones(len(batch.rewards))
    target, *_ = sac_target_q(targets, actor, states, rewards, terminated, alpha)
    torch.testing.assert_close(target, rewards)
    log_alpha = torch.tensor(0.0, requires_grad=True)
    ao = torch.optim.Adam(actor.parameters(), lr=3e-4)
    co = torch.optim.Adam(critics.parameters(), lr=3e-4)
    lo = torch.optim.Adam([log_alpha], lr=3e-4)
    before = log_alpha.detach().clone()
    update_sac(actor, critics, targets, log_alpha, ao, co, lo, batch)
    assert torch.isfinite(log_alpha)
    assert not torch.equal(before, log_alpha.detach())


def test_batch_renorm_joint_forward_updates_once_and_eval_freezes():
    critic = CrossQCritic()
    states, actions = torch.randn(4, 5), torch.randn(4, 2).clamp(-1, 1)
    next_states, next_actions = torch.randn(4, 5), torch.randn(4, 2).clamp(-1, 1)
    critic.forward_current_next(states, actions, next_states, next_actions)
    assert critic.renorm1.num_batches_tracked.item() == 1
    before = critic.renorm1.running_mean.clone()
    critic.eval()
    critic(states, actions)
    assert critic.renorm1.num_batches_tracked.item() == 1
    torch.testing.assert_close(critic.renorm1.running_mean, before)


def test_crossq_update_has_no_target_critic_and_actor_update_freezes_brn_stats():
    from hvac_pid.continuous import GaussianActor

    actor, q1, q2 = GaussianActor(), CrossQCritic(), CrossQCritic()
    log_alpha = torch.tensor(0.0, requires_grad=True)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)
    update_crossq(actor, q1, q2, log_alpha, actor_optimizer, critic_optimizer,
                  alpha_optimizer, _batch(), update_index=2)
    assert not hasattr(q1, "target") and not hasattr(q2, "target")
    assert q1.renorm1.num_batches_tracked.item() == 1
    assert q2.renorm1.num_batches_tracked.item() == 1


@pytest.mark.parametrize(
    "trainer, policy_type, expected_updates",
    [(ppo_train, None, 20), (td3_train, TD3Policy, 45),
     (sac_train, SACPolicy, 45), (crossq_train, CrossQPolicy, 45)],
)
def test_short_training_reproducible_and_checkpoint_roundtrip(
    tmp_path, trainer, policy_type, expected_updates
):
    scenario = Scenario(duration=120)
    first, report_a = trainer([scenario], episodes=5, seed=4)
    second, report_b = trainer([scenario], episodes=5, seed=4)
    assert first.action_mode == "continuous"
    observation = [4, 0, 0, 1, 1]
    action = first.choose(observation)
    np.testing.assert_array_equal(action, second.choose(observation))
    assert action.shape == (2,)
    assert np.all(np.isfinite(action)) and np.all((-1 <= action) & (action <= 1))
    assert report_a["history"] == report_b["history"]
    assert report_a["first_update"]
    assert report_a["cost"]["episodes"] == 5
    assert report_a["cost"]["transitions"] == 300
    assert report_a["cost"]["plant_steps"] == 6000
    assert report_a["cost"]["commissioning_steps"] == 2400
    assert report_a["cost"]["updates"] == expected_updates
    assert all(np.isfinite(row["return"]) and row["return"] <= 0
               for row in report_a["history"])
    before_evaluation = {key: value.detach().clone()
                         for key, value in first.actor.state_dict().items()}
    reference = evaluate(scenario, Gains(0.3, 0.015), first)
    after_evaluation = first.actor.state_dict()
    assert all(torch.equal(before_evaluation[key], after_evaluation[key])
               for key in before_evaluation)
    repeated = evaluate(scenario, Gains(0.3, 0.015), first)
    np.testing.assert_array_equal(reference.trace.temperature, repeated.trace.temperature)
    np.testing.assert_array_equal(reference.trace.command, repeated.trace.command)
    path = tmp_path / f"{first.__class__.__name__}.pt"
    first.save(path)
    restored = first.__class__.load(path)
    np.testing.assert_array_equal(action, restored.choose(observation))
    roundtrip = evaluate(scenario, Gains(0.3, 0.015), restored)
    np.testing.assert_array_equal(reference.trace.temperature, roundtrip.trace.temperature)
    np.testing.assert_array_equal(reference.trace.command, roundtrip.trace.command)
