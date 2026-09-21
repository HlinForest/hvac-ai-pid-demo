"""A compact CPU Soft Actor-Critic implementation for continuous PI gains."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .continuous import (
    ACTION_DIM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GAMMA,
    DEFAULT_LR,
    DEFAULT_REPLAY_CAPACITY,
    DEFAULT_WARMUP,
    GaussianActor,
    ReplayBuffer,
    TwinCritic,
    as_observation,
    checkpoint,
    commissioning,
    environment_step,
    json_float,
    policy_observation,
    read_checkpoint,
    report_cost,
    seed_cpu,
    tensor_batch,
    transition_mask,
)
from .core import TuningEnv
from .td3 import soft_update


TARGET_ENTROPY = -float(ACTION_DIM)
TARGET_TAU = 0.005


def network() -> GaussianActor:
    return GaussianActor()


def sample_action(actor: GaussianActor, states: torch.Tensor, deterministic: bool = False):
    """Sample an action and stable tanh corrected log probability."""

    return actor.sample(states, deterministic=deterministic)


class SACPolicy:
    action_mode = "continuous"

    def __init__(self, actor: GaussianActor | None = None, config: dict | None = None):
        self.actor = network() if actor is None else actor
        self.config = dict(config or {})
        self.actor.eval()

    def choose(self, observation) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            action, _log_prob, _mean = sample_action(self.actor, policy_observation(observation), True)
        return action[0].cpu().numpy().astype(np.float32)

    def save(self, path: Path):
        checkpoint(path, algorithm="sac", actor=self.actor, config=self.config)

    @classmethod
    def load(cls, path: Path):
        actor = network()
        payload = read_checkpoint(path, algorithm="sac", actor=actor)
        return cls(actor, payload["config"])


# region sac_target
def sac_target_q(target_critics: TwinCritic, actor: GaussianActor,
                 next_states: torch.Tensor, rewards: torch.Tensor,
                 terminated: torch.Tensor, alpha: torch.Tensor,
                 gamma: float = DEFAULT_GAMMA):
    """Entropy augmented target with no gradient through actor or targets."""

    with torch.no_grad():
        next_action, next_log_prob, _mean = sample_action(actor, next_states)
        next_q1, next_q2 = target_critics(next_states, next_action)
        next_q = torch.minimum(next_q1, next_q2) - alpha.detach() * next_log_prob
        target = rewards + gamma * transition_mask(terminated) * next_q
    return target, next_action, next_log_prob, next_q1, next_q2
# endregion sac_target


# region sac_update
def update_sac(actor, critics, target_critics, log_alpha,
               actor_optimizer, critic_optimizer, alpha_optimizer, batch,
               gamma: float = DEFAULT_GAMMA, target_tau: float = TARGET_TAU,
               target_entropy: float = TARGET_ENTROPY):
    states, actions, rewards, next_states, terminated, _truncated = tensor_batch(batch)
    alpha = log_alpha.exp()
    target, next_action, next_log_prob, next_q1, next_q2 = sac_target_q(
        target_critics, actor, next_states, rewards, terminated, alpha, gamma,
    )
    q1, q2 = critics(states, actions)
    q1_loss = nn.functional.mse_loss(q1, target)
    q2_loss = nn.functional.mse_loss(q2, target)
    critic_loss = q1_loss + q2_loss
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_optimizer.step()

    # Actor gradients pass through Q into the sampled actions, while Q
    # parameters stay frozen for this part of the update.
    previous = [parameter.requires_grad for parameter in critics.parameters()]
    for parameter in critics.parameters():
        parameter.requires_grad_(False)
    action, log_prob, _mean = sample_action(actor, states)
    actor_q1, actor_q2 = critics(states, action)
    actor_loss_tensor = (alpha.detach() * log_prob - torch.minimum(actor_q1, actor_q2)).mean()
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss_tensor.backward()
    actor_optimizer.step()
    for parameter, requires_grad in zip(critics.parameters(), previous):
        parameter.requires_grad_(requires_grad)

    alpha_loss_tensor = -(log_alpha * (log_prob.detach() + target_entropy)).mean()
    alpha_optimizer.zero_grad(set_to_none=True)
    alpha_loss_tensor.backward()
    alpha_optimizer.step()
    soft_update(critics, target_critics, target_tau)

    detail = {
        "states": json_float(states[0]),
        "actions": json_float(actions[0]),
        "reward": json_float(rewards[0]),
        "terminated": json_float(terminated[0]),
        "alpha": json_float(alpha),
        "next_action": json_float(next_action[0]),
        "next_log_prob": json_float(next_log_prob[0]),
        "next_q1": json_float(next_q1[0]),
        "next_q2": json_float(next_q2[0]),
        "target": json_float(target[0]),
        "q1": json_float(q1[0]),
        "q2": json_float(q2[0]),
        "critic_loss": float(critic_loss.detach().item()),
        "actor_loss": float(actor_loss_tensor.detach().item()),
        "alpha_loss": float(alpha_loss_tensor.detach().item()),
    }
    return float((critic_loss + actor_loss_tensor + alpha_loss_tensor).detach().item()), detail
# endregion sac_update


sac_target = sac_target_q
sac_update = update_sac


def train(scenarios, episodes=500, seed=0, progress=None):
    if int(episodes) < 1 or not scenarios:
        raise ValueError("Training needs at least one episode and one scenario")
    episodes = int(episodes)
    started = perf_counter()
    rng = seed_cpu(seed)
    anchors = commissioning(scenarios)
    actor, critics, target_critics = network(), TwinCritic(), TwinCritic()
    target_critics.load_state_dict(critics.state_dict())
    target_critics.requires_grad_(False)
    target_critics.eval()
    log_alpha = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=DEFAULT_LR)
    critic_optimizer = torch.optim.Adam(critics.parameters(), lr=DEFAULT_LR)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=DEFAULT_LR)
    replay = ReplayBuffer(DEFAULT_REPLAY_CAPACITY)
    history = []
    transitions = plant_steps = updates = 0
    first_update = None

    for episode in range(episodes):
        scenario_index = int(rng.integers(len(scenarios)))
        env = TuningEnv(scenarios[scenario_index], anchors[scenario_index])
        observation = as_observation(env.reset())
        done = False
        total_return = 0.0
        losses = []
        while not done:
            if len(replay) < DEFAULT_WARMUP:
                action = rng.uniform(-1.0, 1.0, ACTION_DIM).astype(np.float32)
            else:
                with torch.no_grad():
                    action, _log_prob, _mean = sample_action(actor, policy_observation(observation))
                action = action[0].cpu().numpy().astype(np.float32)
            next_observation, reward, is_terminal, is_truncated = environment_step(env, action)
            replay.push(observation, action, reward, next_observation,
                        terminated=is_terminal, truncated=is_truncated)
            observation = next_observation
            total_return += reward
            transitions += 1
            done = bool(is_terminal or is_truncated)
            if len(replay) >= max(DEFAULT_WARMUP, DEFAULT_BATCH_SIZE):
                batch = replay.sample(DEFAULT_BATCH_SIZE, rng)
                updates += 1
                loss, detail = update_sac(
                    actor, critics, target_critics, log_alpha,
                    actor_optimizer, critic_optimizer, alpha_optimizer, batch,
                )
                losses.append(loss)
                if first_update is None:
                    first_update = detail
        plant_steps += env.index
        history.append({"episode": episode + 1, "return": total_return,
                        "loss": float(np.mean(losses)) if losses else None})
        if progress and (episode + 1) % 50 == 0:
            progress(f"SAC {episode + 1}/{episodes}")

    policy = SACPolicy(actor, {
        "gamma": DEFAULT_GAMMA, "learning_rate": DEFAULT_LR,
        "hidden_dim": 64, "seed": int(seed), "episodes": episodes,
        "batch_size": DEFAULT_BATCH_SIZE, "warmup": DEFAULT_WARMUP,
        "replay_capacity": DEFAULT_REPLAY_CAPACITY, "target_tau": TARGET_TAU,
        "target_entropy": TARGET_ENTROPY, "alpha": float(log_alpha.exp().detach().item()),
    })
    report = {
        "history": history,
        "first_update": first_update,
        "cost": report_cost(episodes, transitions, plant_steps,
                             sum(round(240 / s.dt) for s in scenarios),
                             updates, perf_counter() - started),
    }
    return policy, report


__all__ = ["SACPolicy", "network", "sample_action", "sac_target", "sac_target_q",
           "sac_update", "train", "update_sac"]
