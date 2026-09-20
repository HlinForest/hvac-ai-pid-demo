"""Tabular Q-learning on the shared online gain-selection environment."""
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .core import ACTION_SCALES, TuningEnv, evaluate
from .tuning import identify, simc


def state_index(observation):
    error, slope, command, kp_scale, ki_scale = observation
    e = np.digitize(error, [-1.0, -0.2, 0.2, 1.0])
    d = np.digitize(slope, [-0.1, -0.02, 0.02, 0.1])
    u = np.digitize(command, [0.33, 0.67])
    gain_index = int(np.argmin(np.sum((ACTION_SCALES - [kp_scale, ki_scale]) ** 2, axis=1)))
    return int(e), int(d), int(u), gain_index


def normalized_observation(observation):
    # Same observable quantities as Q-learning, with continuous scaling.
    return np.asarray(observation, dtype=np.float32) / np.asarray([4, 0.2, 1, 2, 2], dtype=np.float32)


def epsilon_at(episode, episodes):
    return 1.0 - 0.95 * min(1.0, episode / max(1, int(episodes * 0.8)))


@dataclass
class QPolicy:
    q: np.ndarray

    def choose(self, observation):
        # First maximum is the tie rule, including entirely unseen states.
        return int(np.argmax(self.q[state_index(observation)]))

    def save(self, path: Path):
        np.savez(path, q=self.q)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["q"].copy())


# region td_update
def update_q(q, state, action, reward, next_state, done, alpha=0.15, gamma=0.99):
    target = reward if done else reward + gamma * np.max(q[next_state])
    delta = target - q[state + (action,)]
    q[state + (action,)] += alpha * delta
    return float(delta)
# endregion td_update


def commissioning(scenarios):
    return [simc(identify(s)[0]) for s in scenarios]


def validate(policy, scenarios):
    return [evaluate(s, simc(identify(s)[0]), policy).iae for s in scenarios]


def train(scenarios, episodes=500, seed=0, progress=None):
    if episodes < 1 or not scenarios:
        raise ValueError("Training needs at least one episode and one scenario")
    start = perf_counter()
    anchors = commissioning(scenarios)
    rng = np.random.default_rng(seed)
    policy = QPolicy(np.zeros((5, 5, 3, 9, 9)))
    visits = np.zeros(policy.q.shape[:-1], dtype=int)
    history, transitions, plant_steps = [], 0, 0
    snapshots = []
    example_update = None
    for episode in range(episodes):
        index = int(rng.integers(len(scenarios)))
        env = TuningEnv(scenarios[index], anchors[index])
        observation = env.reset()
        epsilon = epsilon_at(episode, episodes)
        total, done = 0.0, False
        while not done:
            state = state_index(observation)
            action = int(rng.integers(9)) if rng.random() < epsilon else policy.choose(observation)
            old = float(policy.q[state + (action,)])
            next_observation, reward, done = env.step(action)
            delta = update_q(policy.q, state, action, reward, state_index(next_observation), done)
            if example_update is None:
                example_update = {"state": list(state), "action": action, "reward": reward,
                                  "next_state": list(state_index(next_observation)), "done": done,
                                  "old_q": old, "td_error": delta,
                                  "new_q": float(policy.q[state + (action,)])}
            visits[state] += 1
            observation = next_observation
            total += reward
            transitions += 1
        plant_steps += env.index
        history.append({"episode": episode + 1, "return": total, "epsilon": epsilon})
        if episode + 1 in {1, max(1, episodes // 2), episodes}:
            snapshots.append(policy.q.copy())
        if progress and (episode + 1) % 50 == 0:
            progress(f"Q-Learning {episode + 1}/{episodes}")
    return policy, {"history": history, "example_update": example_update,
                    "visited_states": int(np.count_nonzero(visits)), "total_states": int(visits.size),
                    "snapshots": snapshots, "cost": {
                        "episodes": episodes, "transitions": transitions, "plant_steps": plant_steps,
                        "commissioning_steps": sum(round(240 / s.dt) for s in scenarios),
                        "seconds": perf_counter() - start,
                    }}
