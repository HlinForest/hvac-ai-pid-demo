"""Small CPU DQN. Imported only by DQN commands; torch is optional."""
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .core import TuningEnv
from .rl import commissioning, epsilon_at, normalized_observation


def network():
    return nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 9))


class DQNPolicy:
    def __init__(self, net):
        self.net = net.eval()

    def choose(self, observation):
        with torch.no_grad():
            values = self.net(torch.from_numpy(normalized_observation(observation)))
        return int(values.argmax().item())

    def save(self, path: Path):
        torch.save(self.net.state_dict(), path)

    @classmethod
    def load(cls, path: Path):
        net = network()
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return cls(net)


# region dqn_update
def learn_batch(net, target, optimizer, batch, gamma=0.99):
    states, actions, rewards, next_states, done = zip(*batch)
    states = torch.tensor(np.asarray(states), dtype=torch.float32)
    next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    done = torch.tensor(done, dtype=torch.bool)
    predicted = net(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        future = target(next_states).max(dim=1).values
        expected = rewards + gamma * (~done).float() * future
    loss = nn.functional.smooth_l1_loss(predicted, expected)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())
# endregion dqn_update


def train(scenarios, episodes=500, seed=0, progress=None):
    if episodes < 1 or not scenarios:
        raise ValueError("Training needs at least one episode and one scenario")
    start = perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    net, target = network(), network()
    target.load_state_dict(net.state_dict())
    target.requires_grad_(False)
    target.eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    policy = DQNPolicy(net)
    memory = deque(maxlen=10_000)
    anchors = commissioning(scenarios)
    history, transitions, updates, plant_steps = [], 0, 0, 0
    for episode in range(episodes):
        index = int(rng.integers(len(scenarios)))
        env = TuningEnv(scenarios[index], anchors[index])
        observation = env.reset()
        epsilon = epsilon_at(episode, episodes)
        total, done, losses = 0.0, False, []
        while not done:
            action = int(rng.integers(9)) if rng.random() < epsilon else policy.choose(observation)
            next_observation, reward, done = env.step(action)
            memory.append((normalized_observation(observation), action, reward,
                           normalized_observation(next_observation), done))
            if len(memory) >= 256:
                indices = rng.choice(len(memory), 64, replace=False)
                batch = [memory[int(i)] for i in indices]
                losses.append(learn_batch(net, target, optimizer, batch))
                updates += 1
                if updates % 200 == 0:
                    target.load_state_dict(net.state_dict())
            observation = next_observation
            total += reward
            transitions += 1
        plant_steps += env.index
        history.append({"episode": episode + 1, "return": total, "epsilon": epsilon,
                        "loss": float(np.mean(losses)) if losses else None})
        if progress and (episode + 1) % 50 == 0:
            progress(f"DQN {episode + 1}/{episodes}")
    return policy, {"history": history, "cost": {
        "episodes": episodes, "transitions": transitions, "plant_steps": plant_steps,
        "commissioning_steps": sum(round(240 / s.dt) for s in scenarios),
        "updates": updates, "seconds": perf_counter() - start,
    }}
