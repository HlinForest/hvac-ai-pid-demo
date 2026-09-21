"""Shared small CPU building blocks for continuous-action RL.

The environment exposes five normalized observations and two actions.  The
actions are kept in ``[-1, 1]`` here; :class:`~hvac_pid.core.TuningEnv` maps
them to positive PI-gain scales.  This module intentionally contains only
plain PyTorch modules and NumPy replay helpers so the algorithms in the
neighbouring modules remain readable and independently testable.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from .core import TuningEnv
from .rl import commissioning, normalized_observation


OBSERVATION_DIM = 5
ACTION_DIM = 2
HIDDEN_DIM = 64
OBSERVATION_SCALE = np.asarray([4.0, 0.2, 1.0, 2.0, 2.0], dtype=np.float32)
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
DEFAULT_REPLAY_CAPACITY = 10_000
DEFAULT_BATCH_SIZE = 64
DEFAULT_WARMUP = 256
DEFAULT_LR = 3e-4
DEFAULT_GAMMA = 0.99


def seed_cpu(seed: int) -> np.random.Generator:
    """Seed all random sources used by the CPU learners.

    The algorithms use a NumPy generator for environment sampling and replay
    indices, while PyTorch owns network initialisation and Gaussian noise.
    Keeping both under the same seed makes short tutorial runs repeatable.
    """

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    # All operations used by these CPU networks have deterministic kernels.
    torch.use_deterministic_algorithms(True)
    return np.random.default_rng(seed)


def as_observation(observation: Iterable[float] | np.ndarray) -> np.ndarray:
    """Validate one raw five element environment observation."""

    value = np.asarray(observation, dtype=np.float32)
    if value.shape != (OBSERVATION_DIM,):
        raise ValueError(f"observation must have shape ({OBSERVATION_DIM},)")
    if not np.all(np.isfinite(value)):
        raise ValueError("observation must contain finite values")
    return value.astype(np.float32, copy=False)


def as_action(action: Iterable[float] | np.ndarray) -> np.ndarray:
    """Validate and normalize a two dimensional continuous action."""

    value = np.asarray(action, dtype=np.float32)
    if value.shape != (ACTION_DIM,):
        raise ValueError(f"action must have shape ({ACTION_DIM},)")
    if not np.all(np.isfinite(value)):
        raise ValueError("action must contain finite values")
    if np.any(value < ACTION_LOW) or np.any(value > ACTION_HIGH):
        raise ValueError("action values must be in [-1, 1]")
    return value.astype(np.float32, copy=False)


def make_mlp(input_dim: int, output_dim: int, *, hidden_dim: int = HIDDEN_DIM) -> nn.Sequential:
    """Construct the deliberately small two-hidden-layer network."""

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class DeterministicActor(nn.Module):
    """Tanh bounded actor used by TD3 and exposed by all policies."""

    def __init__(self):
        super().__init__()
        self.net = make_mlp(OBSERVATION_DIM, ACTION_DIM)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(observations))


class GaussianActor(nn.Module):
    """Diagonal Gaussian actor whose samples are squashed by ``tanh``."""

    def __init__(self):
        super().__init__()
        self.body = make_mlp(OBSERVATION_DIM, ACTION_DIM * 2)

    def parameters_for_distribution(self, observations: torch.Tensor):
        mean, log_std = self.body(observations).chunk(2, dim=-1)
        # The lower bound avoids a zero-width distribution and the upper
        # bound prevents explosive exponentials during the first updates.
        return mean, log_std.clamp(-20.0, 2.0)

    def forward(self, observations: torch.Tensor):
        return self.parameters_for_distribution(observations)

    def sample(self, observations: torch.Tensor, *, deterministic: bool = False):
        mean, log_std = self.parameters_for_distribution(observations)
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = mean + log_std.exp() * torch.randn_like(mean)
        action, log_prob = squashed_gaussian(pre_tanh, mean, log_std)
        return action, log_prob, mean


class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = make_mlp(OBSERVATION_DIM, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations).squeeze(-1)


class Critic(nn.Module):
    """A scalar Q network over normalized observations and actions."""

    def __init__(self):
        super().__init__()
        self.net = make_mlp(OBSERVATION_DIM + ACTION_DIM, 1)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([observations, actions], dim=-1)).squeeze(-1)


class TwinCritic(nn.Module):
    """Two independent critics used for clipped double Q estimates."""

    def __init__(self):
        super().__init__()
        self.q1 = Critic()
        self.q2 = Critic()

    def forward(self, observations: torch.Tensor, actions: torch.Tensor):
        return self.q1(observations, actions), self.q2(observations, actions)


# region gaussian_sample
def squashed_gaussian(pre_tanh: torch.Tensor, mean: torch.Tensor,
                      log_std: torch.Tensor, eps: float = 1e-6):
    """Return ``tanh(pre_tanh)`` and its corrected summed log probability.

    The correction is evaluated from the pre-squash value and clamped near
    the action bounds, which keeps both sampling and entropy updates finite.
    """

    normal = torch.distributions.Normal(mean, log_std.exp())
    action = torch.tanh(pre_tanh)
    # log(1 - tanh(x)^2) in a stable form.  Evaluating the Jacobian from the
    # already saturated action loses precision at large |x|.
    correction = 2.0 * (np.log(2.0) - pre_tanh - nn.functional.softplus(-2.0 * pre_tanh))
    log_prob = (normal.log_prob(pre_tanh) - correction).sum(dim=-1)
    return action, log_prob
# endregion gaussian_sample


@dataclass
class TransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    @property
    def done(self) -> np.ndarray:
        return np.logical_or(self.terminated, self.truncated)


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool


class ReplayBuffer:
    """Bounded replay with explicit terminal/truncation flags.

    A time-limit truncation should still bootstrap a value target.  Keeping
    the flags separate makes that distinction visible in every algorithm.
    """

    def __init__(self, capacity: int = DEFAULT_REPLAY_CAPACITY):
        if int(capacity) < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self._items: deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def push(self, state, action, reward, next_state, terminated=False,
             truncated=False):
        self._items.append(Transition(
            normalized_observation(as_observation(state)).copy(), as_action(action).copy(), float(reward),
            normalized_observation(as_observation(next_state)).copy(), bool(terminated), bool(truncated),
        ))

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> TransitionBatch:
        if len(self) < int(batch_size):
            raise ValueError("not enough transitions in replay")
        rng = np.random.default_rng() if rng is None else rng
        indices = rng.choice(len(self), int(batch_size), replace=False)
        rows = [self[int(index)] for index in indices]
        return TransitionBatch(
            np.asarray([row.state for row in rows], dtype=np.float32),
            np.asarray([row.action for row in rows], dtype=np.float32),
            np.asarray([row.reward for row in rows], dtype=np.float32),
            np.asarray([row.next_state for row in rows], dtype=np.float32),
            np.asarray([row.terminated for row in rows], dtype=np.bool_),
            np.asarray([row.truncated for row in rows], dtype=np.bool_),
        )


def tensor_batch(batch: TransitionBatch, device: str = "cpu"):
    return (
        torch.as_tensor(batch.states, dtype=torch.float32, device=device),
        torch.as_tensor(batch.actions, dtype=torch.float32, device=device),
        torch.as_tensor(batch.rewards, dtype=torch.float32, device=device),
        torch.as_tensor(batch.next_states, dtype=torch.float32, device=device),
        torch.as_tensor(batch.terminated, dtype=torch.float32, device=device),
        torch.as_tensor(batch.truncated, dtype=torch.float32, device=device),
    )


def transition_mask(terminated: torch.Tensor) -> torch.Tensor:
    """Mask value bootstrapping only at true terminals."""

    return 1.0 - terminated.float()


def environment_step(env: TuningEnv, action: np.ndarray):
    """Call the core continuous environment and return its raw observation."""

    action = as_action(action)
    next_observation, reward, done = env.step_continuous(action)
    return as_observation(next_observation), float(reward), bool(done), False


def checkpoint(path, *, algorithm: str, actor: nn.Module, config: dict[str, Any] | None = None):
    """Save the frozen actor and enough metadata to validate a load."""

    payload = {
        "format": "hvac-pid-continuous-v1",
        "algorithm": str(algorithm),
        "action_mode": "continuous",
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "observation_scale": OBSERVATION_SCALE.tolist(),
        "action_bounds": [ACTION_LOW, ACTION_HIGH],
        "gain_transform": "2**action",
        "interval": 2.0,
        "actor_state_dict": {key: value.detach().cpu().clone()
                              for key, value in actor.state_dict().items()},
        "config": dict(config or {}),
    }
    torch.save(payload, path)


def read_checkpoint(path, *, algorithm: str, actor: nn.Module):
    """Load and validate a checkpoint into an actor."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format", "algorithm", "action_mode", "observation_dim",
                "action_dim", "observation_scale", "action_bounds",
                "gain_transform", "interval", "actor_state_dict", "config"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("invalid continuous policy checkpoint")
    if not isinstance(payload["config"], dict):
        raise ValueError("continuous policy checkpoint config must be a mapping")
    if payload["format"] != "hvac-pid-continuous-v1":
        raise ValueError("unsupported continuous policy checkpoint format")
    if payload["algorithm"] != algorithm:
        raise ValueError(f"checkpoint is for {payload['algorithm']}, expected {algorithm}")
    if payload["action_mode"] != "continuous" or payload["observation_dim"] != OBSERVATION_DIM:
        raise ValueError("checkpoint observation/action metadata does not match")
    if payload["action_dim"] != ACTION_DIM:
        raise ValueError("checkpoint action dimension does not match")
    if not np.allclose(np.asarray(payload["observation_scale"], dtype=float), OBSERVATION_SCALE):
        raise ValueError("checkpoint observation normalization does not match")
    if list(payload["action_bounds"]) != [ACTION_LOW, ACTION_HIGH]:
        raise ValueError("checkpoint action bounds do not match")
    if payload["gain_transform"] != "2**action" or float(payload["interval"]) != 2.0:
        raise ValueError("checkpoint gain transform does not match")
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    return payload


def policy_observation(observation) -> torch.Tensor:
    return torch.as_tensor(normalized_observation(as_observation(observation)),
                           dtype=torch.float32).unsqueeze(0)


def json_float(value):
    """Convert torch/NumPy scalar values for JSON-compatible reports."""

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def report_cost(episodes: int, transitions: int, plant_steps: int,
                commissioning_steps: int, updates: int, seconds: float) -> dict:
    return {
        "episodes": int(episodes),
        "transitions": int(transitions),
        "plant_steps": int(plant_steps),
        "commissioning_steps": int(commissioning_steps),
        "updates": int(updates),
        "seconds": float(seconds),
    }


__all__ = [
    "ACTION_DIM", "ACTION_HIGH", "ACTION_LOW", "OBSERVATION_DIM",
    "OBSERVATION_SCALE", "DEFAULT_BATCH_SIZE", "DEFAULT_GAMMA",
    "DEFAULT_LR", "DEFAULT_REPLAY_CAPACITY", "DEFAULT_WARMUP",
    "ReplayBuffer", "Transition", "TransitionBatch", "BatchRenorm1d",
    "Critic", "CrossQCritic", "DeterministicActor", "GaussianActor", "TwinCritic",
    "ValueNetwork", "as_action", "as_observation", "checkpoint",
    "commissioning", "environment_step", "json_float", "make_mlp",
    "normalized_observation", "policy_observation", "read_checkpoint",
    "report_cost", "seed_cpu", "squashed_gaussian", "tensor_batch",
    "gaussian_sample", "batch_renorm", "transition_mask",
]


# region batch_renorm
class BatchRenorm1d(nn.Module):
    """Batch Renormalization for CrossQ's joint current/next batches.

    The running statistics are buffers, so ``eval()`` freezes them exactly as
    PyTorch BatchNorm does.  In training mode each call updates the buffers
    once; CrossQ calls this module on concatenated current and next samples.
    """

    def __init__(self, features: int, eps: float = 1e-5, momentum: float = 0.01,
                 rmax: float = 3.0, dmax: float = 5.0):
        super().__init__()
        self.features = int(features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.rmax = float(rmax)
        self.dmax = float(dmax)
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.register_buffer("running_mean", torch.zeros(features))
        self.register_buffer("running_var", torch.ones(features))
        self.register_buffer("num_batches_tracked", torch.zeros((), dtype=torch.long))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != self.features:
            raise ValueError("BatchRenorm1d expects a [batch, features] tensor")
        if self.training:
            batch_mean = inputs.mean(dim=0)
            batch_var = inputs.var(dim=0, unbiased=False)
            batch_std = torch.sqrt(batch_var + self.eps)
            # These are constants for the current normalized batch.  Detached
            # clones also keep the running-stat buffer version out of the
            # backward graph while it is updated below.
            running_mean = self.running_mean.detach().clone()
            running_var = self.running_var.detach().clone()
            running_std = torch.sqrt(running_var + self.eps)
            ratio = (batch_std.detach() / running_std).clamp(1.0 / self.rmax, self.rmax).detach()
            delta = ((batch_mean.detach() - running_mean) / running_std).clamp(-self.dmax, self.dmax).detach()
            normalized = (inputs - batch_mean) / batch_std
            normalized = normalized * ratio + delta
            with torch.no_grad():
                self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * batch_mean)
                self.running_var.mul_(1.0 - self.momentum).add_(self.momentum * batch_var)
                self.num_batches_tracked.add_(1)
        else:
            normalized = (inputs - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return normalized * self.weight + self.bias
# endregion batch_renorm


# Short names used by the tutorial's equation references.
gaussian_sample = squashed_gaussian
batch_renorm = BatchRenorm1d


class CrossQCritic(nn.Module):
    """A critic with BRN layers and a joint current/next forward."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(OBSERVATION_DIM + ACTION_DIM, HIDDEN_DIM)
        self.renorm1 = BatchRenorm1d(HIDDEN_DIM)
        self.linear2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.renorm2 = BatchRenorm1d(HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor):
        hidden = torch.cat([observations, actions], dim=-1)
        hidden = torch.relu(self.renorm1(self.linear1(hidden)))
        hidden = torch.relu(self.renorm2(self.linear2(hidden)))
        return self.output(hidden).squeeze(-1)

    def forward_current_next(self, observations: torch.Tensor, actions: torch.Tensor,
                             next_observations: torch.Tensor, next_actions: torch.Tensor):
        count = observations.shape[0]
        joined_observations = torch.cat([observations, next_observations], dim=0)
        joined_actions = torch.cat([actions, next_actions], dim=0)
        values = self.forward(joined_observations, joined_actions)
        return values[:count], values[count:]
