"""Explicit chapter and implementation names shared by experiments and export."""
from importlib import import_module

ONLINE = {
    "qlearning": ("Q-Learning", "06-qlearning", "rl", "QPolicy", "model.npz"),
    "dqn": ("DQN", "07-dqn", "dqn", "DQNPolicy", "model.pt"),
    "ppo": ("PPO", "10-ppo", "ppo", "PPOPolicy", "model.pt"),
    "td3": ("TD3", "11-td3", "td3", "TD3Policy", "model.pt"),
    "sac": ("SAC", "12-sac", "sac", "SACPolicy", "model.pt"),
    "crossq": ("CrossQ", "13-crossq", "crossq", "CrossQPolicy", "model.pt"),
}


def implementation(method):
    return import_module("hvac_pid." + ONLINE[method][2])


def load_policies(output, include_neural=True):
    policies = {}
    for method, (name, chapter, _, class_name, file) in ONLINE.items():
        if include_neural or method == "qlearning":
            policies[name] = getattr(implementation(method), class_name).load(output / chapter / file)
    return policies
