"""CrossQ: SAC style updates with joint Batch Renormalized critics.

CrossQ deliberately has no target critic.  Each critic sees current and next
state/action rows in one forward pass, allowing its Batch Renormalization
statistics to couple the two distributions as described by the algorithm.
"""
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
    CrossQCritic,
    GaussianActor,
    ReplayBuffer,
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


TARGET_ENTROPY = -float(ACTION_DIM)
POLICY_DELAY = 2


def network() -> GaussianActor:
    return GaussianActor()


class CrossQPolicy:
    action_mode = "continuous"

    def __init__(self, actor: GaussianActor | None = None, config: dict | None = None):
        self.actor = network() if actor is None else actor
        self.config = dict(config or {})
        self.actor.eval()

    def choose(self, observation) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            mean, _log_std = self.actor(policy_observation(observation))
            action = torch.tanh(mean)[0]
        return action.cpu().numpy().astype(np.float32)

    def save(self, path: Path):
        checkpoint(path, algorithm="crossq", actor=self.actor, config=self.config)

    @classmethod
    def load(cls, path: Path):
        actor = network()
        payload = read_checkpoint(path, algorithm="crossq", actor=actor)
        return cls(actor, payload["config"])


def _sample(actor: GaussianActor, states: torch.Tensor):
    return actor.sample(states)


# region crossq_update
def update_crossq(actor, q1, q2, log_alpha, actor_optimizer,
                  critic_optimizer, alpha_optimizer, batch,
                  update_index: int, gamma: float = DEFAULT_GAMMA,
                  policy_delay: int = POLICY_DELAY,
                  target_entropy: float = TARGET_ENTROPY):
    """One UTD1 CrossQ update, with an actor update every ``policy_delay``."""

    states, actions, rewards, next_states, terminated, _truncated = tensor_batch(batch)
    # Next policy sampling is held fixed while the joint critic forward updates
    # BRN statistics and supplies both current and next Q estimates.
    with torch.no_grad():
        next_actions, next_log_prob, _mean = _sample(actor, next_states)
    q1_current, q1_next = q1.forward_current_next(states, actions, next_states, next_actions)
    q2_current, q2_next = q2.forward_current_next(states, actions, next_states, next_actions)
    with torch.no_grad():
        alpha = log_alpha.exp()
        next_q = torch.minimum(q1_next, q2_next) - alpha * next_log_prob
        target = rewards + gamma * transition_mask(terminated) * next_q
    q1_loss = nn.functional.mse_loss(q1_current, target)
    q2_loss = nn.functional.mse_loss(q2_current, target)
    critic_loss = q1_loss + q2_loss
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_optimizer.step()

    actor_loss_tensor = None
    alpha_loss_tensor = None
    actor_updated = False
    if int(update_index) % int(policy_delay) == 0:
        # BRN must stay in evaluation mode during policy/entropy updates.  In
        # particular, this prevents an actor batch from changing its running
        # means and variances merely because an action was evaluated.
        q1_was_training, q2_was_training = q1.training, q2.training
        q1.eval()
        q2.eval()
        previous_q1 = [parameter.requires_grad for parameter in q1.parameters()]
        previous_q2 = [parameter.requires_grad for parameter in q2.parameters()]
        for parameter in q1.parameters():
            parameter.requires_grad_(False)
        for parameter in q2.parameters():
            parameter.requires_grad_(False)
        actor_actions, log_prob, _mean = _sample(actor, states)
        actor_q1 = q1(states, actor_actions)
        actor_q2 = q2(states, actor_actions)
        actor_loss_tensor = (alpha.detach() * log_prob - torch.minimum(actor_q1, actor_q2)).mean()
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss_tensor.backward()
        actor_optimizer.step()
        alpha_loss_tensor = -(log_alpha * (log_prob.detach() + target_entropy)).mean()
        alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss_tensor.backward()
        alpha_optimizer.step()
        for parameter, requires_grad in zip(q1.parameters(), previous_q1):
            parameter.requires_grad_(requires_grad)
        for parameter, requires_grad in zip(q2.parameters(), previous_q2):
            parameter.requires_grad_(requires_grad)
        q1.train(q1_was_training)
        q2.train(q2_was_training)
        actor_updated = True

    detail = {
        "states": json_float(states[0]),
        "actions": json_float(actions[0]),
        "reward": json_float(rewards[0]),
        "terminated": json_float(terminated[0]),
        "alpha": json_float(alpha),
        "next_action": json_float(next_actions[0]),
        "next_log_prob": json_float(next_log_prob[0]),
        "next_q1": json_float(q1_next[0]),
        "next_q2": json_float(q2_next[0]),
        "target": json_float(target[0]),
        "q1": json_float(q1_current[0]),
        "q2": json_float(q2_current[0]),
        "critic_loss": float(critic_loss.detach().item()),
        "actor_updated": actor_updated,
        "actor_loss": None if actor_loss_tensor is None else float(actor_loss_tensor.detach().item()),
        "alpha_loss": None if alpha_loss_tensor is None else float(alpha_loss_tensor.detach().item()),
    }
    total_loss = critic_loss if actor_loss_tensor is None else critic_loss + actor_loss_tensor + alpha_loss_tensor
    return float(total_loss.detach().item()), detail
# endregion crossq_update


def train(scenarios, episodes=500, seed=0, progress=None):
    if int(episodes) < 1 or not scenarios:
        raise ValueError("Training needs at least one episode and one scenario")
    episodes = int(episodes)
    started = perf_counter()
    rng = seed_cpu(seed)
    anchors = commissioning(scenarios)
    actor, q1, q2 = network(), CrossQCritic(), CrossQCritic()
    log_alpha = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=DEFAULT_LR)
    critic_optimizer = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=DEFAULT_LR)
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
                    action, _log_prob, _mean = _sample(actor, policy_observation(observation))
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
                loss, detail = update_crossq(
                    actor, q1, q2, log_alpha, actor_optimizer,
                    critic_optimizer, alpha_optimizer, batch, updates,
                )
                losses.append(loss)
                if first_update is None:
                    first_update = detail
        plant_steps += env.index
        history.append({"episode": episode + 1, "return": total_return,
                        "loss": float(np.mean(losses)) if losses else None})
        if progress and (episode + 1) % 50 == 0:
            progress(f"CrossQ {episode + 1}/{episodes}")

    policy = CrossQPolicy(actor, {
        "gamma": DEFAULT_GAMMA, "learning_rate": DEFAULT_LR,
        "hidden_dim": 64, "seed": int(seed), "episodes": episodes,
        "batch_size": DEFAULT_BATCH_SIZE, "warmup": DEFAULT_WARMUP,
        "replay_capacity": DEFAULT_REPLAY_CAPACITY, "policy_delay": POLICY_DELAY,
        "target_entropy": TARGET_ENTROPY, "alpha": float(log_alpha.exp().detach().item()),
        "utd_ratio": 1, "batch_renorm": True,
        "target_critics": False,
    })
    report = {
        "history": history,
        "first_update": first_update,
        "cost": report_cost(episodes, transitions, plant_steps,
                             sum(round(240 / s.dt) for s in scenarios),
                             updates, perf_counter() - started),
    }
    return policy, report


__all__ = ["CrossQPolicy", "CrossQCritic", "network", "train", "update_crossq"]
