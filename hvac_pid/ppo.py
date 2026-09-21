"""A compact, readable CPU implementation of clipped PPO for gain tuning."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .continuous import (
    ACTION_DIM,
    DEFAULT_GAMMA,
    DEFAULT_LR,
    GaussianActor,
    OBSERVATION_DIM,
    ValueNetwork,
    as_observation,
    checkpoint,
    commissioning,
    environment_step,
    json_float,
    policy_observation,
    read_checkpoint,
    report_cost,
    seed_cpu,
    squashed_gaussian,
    normalized_observation,
)
from .core import TuningEnv


CLIP_EPSILON = 0.2
GAE_LAMBDA = 0.95
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
UPDATE_EPOCHS = 4


def network() -> GaussianActor:
    """Return the standard 5 -> 64 -> 64 -> 2 Gaussian actor."""

    return GaussianActor()


def gaussian_log_prob(mean: torch.Tensor, log_std: torch.Tensor,
                      pre_tanh: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Stable log probability for a tanh squashed diagonal Gaussian."""

    _action, log_prob = squashed_gaussian(pre_tanh, mean, log_std, eps)
    return log_prob


# region ppo_gae
def compute_gae(rewards, values, next_values=None, terminated=None, truncated=None,
                gamma: float = DEFAULT_GAMMA, gae_lambda: float = GAE_LAMBDA):
    """Compute GAE while treating terminals and time limits differently.

    ``terminated`` controls whether the one step value target bootstraps.
    ``truncated`` stops advantage recursion at a rollout boundary but keeps
    the next value in the boundary transition's TD residual.  This is the
    distinction needed when a rollout ends because a fixed horizon was met.
    """

    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    count = rewards.size
    if next_values is None:
        if values.size == count + 1:
            next_values = values[1:]
            values = values[:-1]
        elif values.size == count:
            next_values = np.empty(count, dtype=np.float32)
            if count:
                next_values[:-1] = values[1:]
                next_values[-1] = 0.0
        else:
            raise ValueError("values must have one value per reward or one extra bootstrap value")
    elif values.size != count:
        raise ValueError("values and rewards must have equal length")
    next_values = np.asarray(next_values, dtype=np.float32).reshape(-1)
    if next_values.size != count:
        raise ValueError("next_values and rewards must have equal length")
    terminated = np.zeros(count, dtype=bool) if terminated is None else np.asarray(terminated, dtype=bool).reshape(-1)
    truncated = np.zeros(count, dtype=bool) if truncated is None else np.asarray(truncated, dtype=bool).reshape(-1)
    if terminated.size != count or truncated.size != count:
        raise ValueError("terminal flags and rewards must have equal length")

    advantages = np.zeros(count, dtype=np.float32)
    carry = 0.0
    for index in range(count - 1, -1, -1):
        bootstrap = 0.0 if terminated[index] else 1.0
        delta = rewards[index] + gamma * bootstrap * next_values[index] - values[index]
        # Do not pass GAE across either a true episode end or a time-limit
        # boundary, while still using a value for the truncated transition.
        continuation = 0.0 if (terminated[index] or truncated[index]) else 1.0
        carry = delta + gamma * gae_lambda * continuation * carry
        advantages[index] = carry
    returns = advantages + values
    return advantages, returns
# endregion ppo_gae


# region ppo_loss
def ppo_loss(log_prob: torch.Tensor, old_log_prob: torch.Tensor,
             advantages: torch.Tensor, value: torch.Tensor,
             returns: torch.Tensor, entropy: torch.Tensor,
             clip_epsilon: float = CLIP_EPSILON,
             value_coef: float = VALUE_COEF,
             entropy_coef: float = ENTROPY_COEF):
    """Return PPO's clipped objective and useful scalar components."""

    ratio = torch.exp(log_prob - old_log_prob)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = (value - returns).square().mean()
    entropy_mean = entropy.mean()
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean
    return total, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_mean,
        "ratio_mean": ratio.mean(),
        "ratio_min": ratio.min(),
        "ratio_max": ratio.max(),
    }
# endregion ppo_loss


class PPOPolicy:
    action_mode = "continuous"

    def __init__(self, actor: GaussianActor | None = None, config: dict | None = None):
        self.actor = network() if actor is None else actor
        self.config = dict(config or {})
        self.actor.eval()

    def choose(self, observation) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            observations = policy_observation(observation)
            mean, _log_std = self.actor(observations)
            action = torch.tanh(mean)[0]
        return action.cpu().numpy().astype(np.float32)

    def save(self, path: Path):
        checkpoint(path, algorithm="ppo", actor=self.actor, config=self.config)

    @classmethod
    def load(cls, path: Path):
        actor = network()
        payload = read_checkpoint(path, algorithm="ppo", actor=actor)
        return cls(actor, payload["config"])


# region ppo_update
def _ppo_update(actor, value_net, actor_optimizer, value_optimizer,
                states, actions_pre_tanh, old_log_probs, advantages, returns,
                gae_deltas=None, batch_size=64, epochs=UPDATE_EPOCHS):
    states = torch.as_tensor(
        np.asarray([normalized_observation(as_observation(state)) for state in states]),
        dtype=torch.float32,
    )
    actions_pre_tanh = torch.as_tensor(np.asarray(actions_pre_tanh), dtype=torch.float32)
    old_log_probs = torch.as_tensor(np.asarray(old_log_probs), dtype=torch.float32)
    raw_advantages = torch.as_tensor(np.asarray(advantages), dtype=torch.float32)
    advantages = raw_advantages.clone()
    returns = torch.as_tensor(np.asarray(returns), dtype=torch.float32)
    gae_deltas = None if gae_deltas is None else torch.as_tensor(np.asarray(gae_deltas), dtype=torch.float32)
    count = states.shape[0]
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch_size = max(1, min(int(batch_size), count))
    losses = []
    first = None
    # A local deterministic permutation is produced by torch's seeded CPU RNG.
    for _epoch in range(int(epochs)):
        order = torch.randperm(count)
        for start in range(0, count, batch_size):
            index = order[start:start + batch_size]
            batch_states = states[index]
            mean, log_std = actor(batch_states)
            log_prob = gaussian_log_prob(mean, log_std, actions_pre_tanh[index])
            # Entropy is estimated from a fresh reparameterized sample from
            # the bounded policy, so the tanh Jacobian is included.
            _entropy_action, fresh_log_prob, _entropy_mean = actor.sample(batch_states)
            entropy = -fresh_log_prob
            value = value_net(batch_states)
            total, details = ppo_loss(log_prob, old_log_probs[index], advantages[index],
                                      value, returns[index], entropy)
            actor_optimizer.zero_grad(set_to_none=True)
            value_optimizer.zero_grad(set_to_none=True)
            total.backward()
            actor_optimizer.step()
            value_optimizer.step()
            losses.append(float(total.detach().item()))
            if first is None:
                first = {
                    "states": json_float(batch_states[0]),
                    "pre_tanh_action": json_float(actions_pre_tanh[index[0]]),
                    "old_log_prob": json_float(old_log_probs[index[0]]),
                    "new_log_prob": json_float(log_prob[0]),
                    "mean": json_float(mean[0]),
                    "log_std": json_float(log_std[0]),
                    "value": json_float(value[0]),
                    "raw_advantage": json_float(raw_advantages[index[0]]),
                    "normalized_advantage": json_float(advantages[index[0]]),
                    "gae_delta": None if gae_deltas is None else json_float(gae_deltas[index[0]]),
                    "advantage": json_float(advantages[index[0]]),
                    "return": json_float(returns[index[0]]),
                    "ratio": json_float(torch.exp(log_prob[0] - old_log_probs[index[0]])),
                    "policy_loss": json_float(details["policy_loss"]),
                    "value_loss": json_float(details["value_loss"]),
                    "entropy": json_float(details["entropy"]),
                    "loss": float(total.detach().item()),
                }
    return float(np.mean(losses)), first, len(losses)
# endregion ppo_update


ppo_gae = compute_gae
ppo_update = _ppo_update


def train(scenarios, episodes=500, seed=0, progress=None):
    if int(episodes) < 1 or not scenarios:
        raise ValueError("Training needs at least one episode and one scenario")
    episodes = int(episodes)
    started = perf_counter()
    rng = seed_cpu(seed)
    anchors = commissioning(scenarios)
    actor = network()
    value_net = ValueNetwork()
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=DEFAULT_LR)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=DEFAULT_LR)
    history = []
    transitions = plant_steps = updates = 0
    first_update = None

    for episode in range(episodes):
        scenario_index = int(rng.integers(len(scenarios)))
        env = TuningEnv(scenarios[scenario_index], anchors[scenario_index])
        observation = as_observation(env.reset())
        done = False
        total_return = 0.0
        states, pre_tanh_actions, old_log_probs = [], [], []
        rewards, values, next_values = [], [], []
        terminated, truncated = [], []
        while not done:
            state_tensor = policy_observation(observation)
            with torch.no_grad():
                mean, log_std = actor(state_tensor)
                # Recover the pre-squash sample from the sampled action only
                # for storage would lose precision near the bounds, so sample
                # once directly from the distribution here instead.
                pre_tanh = mean + log_std.exp() * torch.randn_like(mean)
                action, log_prob = squashed_gaussian(pre_tanh, mean, log_std)
                value = value_net(state_tensor)[0]
            action_np = action[0].cpu().numpy().astype(np.float32)
            next_observation, reward, is_terminal, is_truncated = environment_step(env, action_np)
            states.append(observation.copy())
            pre_tanh_actions.append(pre_tanh[0].cpu().numpy())
            old_log_probs.append(float(log_prob[0].item()))
            rewards.append(float(reward))
            values.append(float(value.item()))
            # Use the next value for a truncated boundary, and zero for a
            # true terminal; compute_gae applies the flag-specific mask.
            with torch.no_grad():
                next_value = value_net(policy_observation(next_observation))[0]
            next_values.append(float(next_value.item()))
            terminated.append(is_terminal)
            truncated.append(is_truncated)
            observation = next_observation
            total_return += float(reward)
            transitions += 1
            done = bool(is_terminal or is_truncated)

        advantages, returns = compute_gae(rewards, values, next_values,
                                          terminated, truncated)
        gae_deltas = np.asarray(rewards, dtype=np.float32) + DEFAULT_GAMMA * (
            1.0 - np.asarray(terminated, dtype=np.float32)
        ) * np.asarray(next_values, dtype=np.float32) - np.asarray(values, dtype=np.float32)
        mean_loss, detail, count = _ppo_update(
            actor, value_net, actor_optimizer, value_optimizer,
            states, pre_tanh_actions, old_log_probs, advantages, returns,
            gae_deltas=gae_deltas,
        )
        updates += count
        if first_update is None:
            first_update = detail
        plant_steps += env.index
        history.append({"episode": episode + 1, "return": total_return, "loss": mean_loss})
        if progress and (episode + 1) % 50 == 0:
            progress(f"PPO {episode + 1}/{episodes}")

    policy = PPOPolicy(actor, {
        "gamma": DEFAULT_GAMMA, "gae_lambda": GAE_LAMBDA,
        "clip_epsilon": CLIP_EPSILON, "learning_rate": DEFAULT_LR,
        "hidden_dim": 64, "batch_size": 64, "update_epochs": UPDATE_EPOCHS,
        "value_coef": VALUE_COEF, "entropy_coef": ENTROPY_COEF,
        "rollout": "one complete episode", "seed": int(seed),
        "episodes": episodes,
    })
    report = {
        "history": history,
        "first_update": first_update,
        "cost": report_cost(episodes, transitions, plant_steps,
                             sum(round(240 / s.dt) for s in scenarios),
                             updates, perf_counter() - started),
    }
    return policy, report


__all__ = ["PPOPolicy", "compute_gae", "gaussian_log_prob", "network", "ppo_gae",
           "ppo_loss", "ppo_update", "train"]
