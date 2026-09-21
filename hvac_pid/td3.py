"""A small CPU implementation of Twin Delayed DDPG (TD3)."""
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
    DeterministicActor,
    ReplayBuffer,
    TwinCritic,
    as_action,
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


POLICY_DELAY = 2
TARGET_TAU = 0.005
TARGET_POLICY_NOISE = 0.2
TARGET_NOISE_CLIP = 0.5
EXPLORATION_NOISE = 0.1


def network() -> DeterministicActor:
    return DeterministicActor()


class TD3Policy:
    action_mode = "continuous"

    def __init__(self, actor: DeterministicActor | None = None, config: dict | None = None):
        self.actor = network() if actor is None else actor
        self.config = dict(config or {})
        self.actor.eval()

    def choose(self, observation) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(policy_observation(observation))[0]
        return action.cpu().numpy().astype(np.float32)

    def save(self, path: Path):
        checkpoint(path, algorithm="td3", actor=self.actor, config=self.config)

    @classmethod
    def load(cls, path: Path):
        actor = network()
        payload = read_checkpoint(path, algorithm="td3", actor=actor)
        return cls(actor, payload["config"])


def soft_update(source: nn.Module, target: nn.Module, tau: float = TARGET_TAU):
    """Polyak update with a target that never receives gradients."""

    with torch.no_grad():
        for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
            target_parameter.mul_(1.0 - tau).add_(tau * source_parameter)


# region td3_target
def td3_target_q(target_critics: TwinCritic, target_actor: DeterministicActor,
                 next_states: torch.Tensor, rewards: torch.Tensor,
                 terminated: torch.Tensor, gamma: float = DEFAULT_GAMMA,
                 noise_std: float = TARGET_POLICY_NOISE,
                 noise_clip: float = TARGET_NOISE_CLIP):
    """Compute the clipped double-Q target with target policy smoothing."""

    with torch.no_grad():
        noise = torch.randn_like(next_states.new_empty((next_states.shape[0], ACTION_DIM))) * noise_std
        noise = noise.clamp(-noise_clip, noise_clip)
        next_action = (target_actor(next_states) + noise).clamp(-1.0, 1.0)
        next_q1, next_q2 = target_critics(next_states, next_action)
        future = torch.minimum(next_q1, next_q2)
        return rewards + gamma * transition_mask(terminated) * future, next_action, next_q1, next_q2
# endregion td3_target


# region td3_update
def update_td3(actor, critics, target_actor, target_critics,
               actor_optimizer, critic_optimizer, batch,
               update_index: int, gamma: float = DEFAULT_GAMMA,
               policy_delay: int = POLICY_DELAY, target_tau: float = TARGET_TAU,
               target_noise: float = TARGET_POLICY_NOISE,
               target_noise_clip: float = TARGET_NOISE_CLIP):
    """Perform one critic update and, on schedule, a delayed actor update."""

    states, actions, rewards, next_states, terminated, _truncated = tensor_batch(batch)
    target, next_action, next_q1, next_q2 = td3_target_q(
        target_critics, target_actor, next_states, rewards, terminated,
        gamma, target_noise, target_noise_clip,
    )
    q1, q2 = critics(states, actions)
    q1_loss = nn.functional.mse_loss(q1, target)
    q2_loss = nn.functional.mse_loss(q2, target)
    critic_loss = q1_loss + q2_loss
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_optimizer.step()

    actor_loss = None
    actor_updated = False
    if int(update_index) % int(policy_delay) == 0:
        # Freeze critic weights while retaining the derivative through Q to
        # the actor actions.  This keeps actor updates cheap and unambiguous.
        previous = [parameter.requires_grad for parameter in critics.parameters()]
        for parameter in critics.parameters():
            parameter.requires_grad_(False)
        actor_loss_tensor = -critics.q1(states, actor(states)).mean()
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss_tensor.backward()
        actor_optimizer.step()
        for parameter, requires_grad in zip(critics.parameters(), previous):
            parameter.requires_grad_(requires_grad)
        soft_update(actor, target_actor, target_tau)
        soft_update(critics, target_critics, target_tau)
        actor_loss = float(actor_loss_tensor.detach().item())
        actor_updated = True

    detail = {
        "states": json_float(states[0]),
        "actions": json_float(actions[0]),
        "reward": json_float(rewards[0]),
        "terminated": json_float(terminated[0]),
        "target_action": json_float(next_action[0]),
        "target_q1": json_float(next_q1[0]),
        "target_q2": json_float(next_q2[0]),
        "target": json_float(target[0]),
        "q1": json_float(q1[0]),
        "q2": json_float(q2[0]),
        "critic_loss": float(critic_loss.detach().item()),
        "actor_updated": actor_updated,
        "actor_loss": actor_loss,
    }
    return float(critic_loss.detach().item()), actor_loss, detail
# endregion td3_update


td3_target = td3_target_q
td3_update = update_td3


def train(scenarios, episodes=500, seed=0, progress=None):
    if int(episodes) < 1 or not scenarios:
        raise ValueError("Training needs at least one episode and one scenario")
    episodes = int(episodes)
    started = perf_counter()
    rng = seed_cpu(seed)
    anchors = commissioning(scenarios)
    actor, target_actor = network(), network()
    critics, target_critics = TwinCritic(), TwinCritic()
    target_actor.load_state_dict(actor.state_dict())
    target_critics.load_state_dict(critics.state_dict())
    target_actor.requires_grad_(False)
    target_critics.requires_grad_(False)
    target_actor.eval()
    target_critics.eval()
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=DEFAULT_LR)
    critic_optimizer = torch.optim.Adam(critics.parameters(), lr=DEFAULT_LR)
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
                    action = actor(policy_observation(observation))[0].cpu().numpy()
                action = action + rng.normal(0.0, EXPLORATION_NOISE, ACTION_DIM)
                action = np.clip(action, -1.0, 1.0).astype(np.float32)
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
                critic_loss, actor_loss, detail = update_td3(
                    actor, critics, target_actor, target_critics,
                    actor_optimizer, critic_optimizer, batch, updates,
                )
                losses.append(critic_loss if actor_loss is None else critic_loss + actor_loss)
                if first_update is None:
                    first_update = detail
        plant_steps += env.index
        history.append({"episode": episode + 1, "return": total_return,
                        "loss": float(np.mean(losses)) if losses else None})
        if progress and (episode + 1) % 50 == 0:
            progress(f"TD3 {episode + 1}/{episodes}")

    policy = TD3Policy(actor, {
        "gamma": DEFAULT_GAMMA, "learning_rate": DEFAULT_LR,
        "hidden_dim": 64, "seed": int(seed), "episodes": episodes,
        "batch_size": DEFAULT_BATCH_SIZE, "warmup": DEFAULT_WARMUP,
        "replay_capacity": DEFAULT_REPLAY_CAPACITY, "policy_delay": POLICY_DELAY,
        "target_tau": TARGET_TAU, "target_policy_noise": TARGET_POLICY_NOISE,
        "target_noise_clip": TARGET_NOISE_CLIP,
        "exploration_noise": EXPLORATION_NOISE,
    })
    report = {
        "history": history,
        "first_update": first_update,
        "cost": report_cost(episodes, transitions, plant_steps,
                             sum(round(240 / s.dt) for s in scenarios),
                             updates, perf_counter() - started),
    }
    return policy, report


__all__ = ["TD3Policy", "network", "soft_update", "td3_target", "td3_target_q",
           "td3_update", "train", "update_td3"]
