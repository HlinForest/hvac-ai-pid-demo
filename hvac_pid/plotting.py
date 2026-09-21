"""Plots use actual traces and trial files, never hard-coded performance numbers."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import Gains
from .tuning import encode, expected_improvement, fit_gp


COLORS = ["#206583", "#d16c38", "#369581", "#75569a", "#b0445d", "#858329", "#5e86b7"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.prop_cycle": plt.cycler(color=COLORS),
                     "svg.fonttype": "none", "svg.hashsalt": "hvac-v5"})


def finish(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def responses(results, path, gains=False):
    fig, axes = plt.subplots(4 if gains else 2, 1, figsize=(9, 8 if gains else 5.7), sharex=True)
    for name, result in results.items():
        t = result.trace
        axes[0].plot(np.r_[t.time, t.time[-1] + t.dt],
                     np.r_[t.temperature, t.next_temperature[-1]], label=name)
        axes[1].plot(t.time, t.command, label=name)
        if gains:
            axes[2].plot(t.time, t.kp, label=name)
            axes[3].plot(t.time, t.ki, label=name)
    trace = next(iter(results.values())).trace
    axes[0].axhline(trace.setpoint[0], color="#62727e", linestyle="--", linewidth=1, label="Setpoint")
    axes[0].legend(ncol=3, frameon=False, fontsize=9)
    axes[0].set_ylabel("Temperature (°C)")
    axes[1].set_ylabel("Cooling command")
    axes[1].set_ylim(-0.03, 1.05)
    if gains:
        axes[2].set_ylabel("Kp (1/°C)")
        axes[3].set_ylabel("Ki (1/(°C min))")
    axes[-1].set_xlabel("Time (min)")
    for ax in axes:
        ax.grid(alpha=0.15)
    finish(fig, path)


def step_curves(curves, path):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for label, time, temperature in curves:
        ax.plot(time, temperature, label=label)
    ax.set(xlabel="Time (min)", ylabel="Temperature (°C)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    finish(fig, path)


def search(trials, path):
    valid = [row for row in trials if row["status"] == "evaluated"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    values = np.asarray([t["iae"] for t in valid])
    points = axes[0].scatter([t["kp"] for t in valid], [t["ki"] for t in valid], c=values, cmap="viridis_r")
    for i, t in enumerate(valid):
        axes[0].annotate(str(i + 1), (t["kp"], t["ki"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[0].set(xscale="log", yscale="log", xlabel="Kp (1/°C)", ylabel="Ki (1/(°C min))")
    fig.colorbar(points, ax=axes[0], label="IAE (°C min)")
    axes[1].plot(np.arange(1, len(valid) + 1), values, "o", alpha=0.5, label="Trial")
    axes[1].plot(np.arange(1, len(valid) + 1), np.minimum.accumulate(values), label="Best so far")
    axes[1].set(xlabel="Simulation number", ylabel="IAE (°C min)")
    axes[1].legend(frameon=False)
    finish(fig, path)


def surrogate(trials, path):
    initial = trials[:5]
    x = np.asarray([encode(Gains(t["kp"], t["ki"])) for t in initial])
    y = np.asarray([t["iae"] for t in initial])
    gp = fit_gp(x, y)
    axis = np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(axis, axis)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    mean, std = gp.predict(grid, return_std=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, values, title in zip(axes, [mean, std, expected_improvement(mean, std, y.min())],
                                 ["Predicted IAE", "Uncertainty (std)", "Expected improvement"]):
        mesh = ax.pcolormesh(xx, yy, values.reshape(xx.shape), shading="auto", cmap="viridis")
        ax.scatter(x[:, 0], x[:, 1], facecolor="white", edgecolor="#111", s=24)
        ax.set(title=title, xlabel="Normalized log Kp", ylabel="Normalized log Ki")
        fig.colorbar(mesh, ax=ax, shrink=0.8)
    finish(fig, path)


def learning(history, path, label):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    episode = [h["episode"] for h in history]
    returns = np.asarray([h["return"] for h in history])
    axes[0].plot(episode, returns, alpha=0.25, linewidth=0.8, label="Exploratory return")
    window = min(20, len(history))
    axes[0].plot(episode[window - 1:], np.convolve(returns, np.ones(window) / window, mode="valid"), label=f"{window}-episode mean")
    axes[0].set(xlabel="Episode", ylabel="Return = -IAE", title=label)
    axes[0].legend(frameon=False, fontsize=8)
    if any(h.get("loss") is not None for h in history):
        axes[1].plot(episode, [h.get("loss", np.nan) for h in history])
        axes[1].set(ylabel="Training loss (algorithm-specific)")
    else:
        axes[1].plot(episode, [h.get("epsilon", np.nan) for h in history])
        axes[1].set(ylabel="Exploration probability (where applicable)")
    axes[1].set_xlabel("Episode")
    finish(fig, path)


def pg_learning(trials, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    episodes = np.arange(1, len(trials) + 1)
    axes[0].plot(episodes, [r["return"] for r in trials], alpha=0.7)
    axes[0].set(xlabel="Training trajectory", ylabel="Exploratory return = -IAE")
    for key in ("kp", "ki"):
        axes[1].plot(episodes, [r[key] for r in trials], label=key)
    axes[1].set(xlabel="Training trajectory", ylabel="Fixed gains within a trajectory", yscale="log")
    axes[1].legend(frameon=False)
    finish(fig, path)


def distributions(rows, path, key="method", value="iae", title="Held-out objects"):
    names = list(dict.fromkeys(r[key] for r in rows))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    data = [[r[value] for r in rows if r[key] == name] for name in names]
    ax.boxplot(data, tick_labels=names, showfliers=False)
    for i, values in enumerate(data, 1):
        ax.scatter(np.full(len(values), i), values, s=12, alpha=0.5)
    ax.set(ylabel="IAE (°C min)", title=title)
    ax.tick_params(axis="x", labelrotation=35)
    finish(fig, path)


def q_snapshots(snapshots, path):
    fig, axes = plt.subplots(1, len(snapshots), figsize=(10, 3.4), squeeze=False)
    for i, (ax, q) in enumerate(zip(axes[0], snapshots)):
        # Fix u-bin=1 and current scale=(1,1); show greedy actions across e/slope.
        actions = np.argmax(q[:, :, 1, 4, :], axis=-1)
        ax.imshow(actions, origin="lower", vmin=0, vmax=8, cmap="viridis", aspect="auto")
        for row in range(5):
            for col in range(5):
                ax.text(col, row, str(actions[row, col]), ha="center", va="center", color="white")
        ax.set(xlabel="Error-slope bin", ylabel="Error bin", title=f"Snapshot {i + 1}")
    finish(fig, path)


def fnn_predictions(labels, predictions, weights, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for i, name in enumerate(["Kp", "Ki"]):
        axes[i].scatter(labels[:, i], predictions[:, i], s=22)
        lo, hi = min(labels[:, i].min(), predictions[:, i].min()), max(labels[:, i].max(), predictions[:, i].max())
        axes[i].plot([lo, hi], [lo, hi], color="#999", linestyle="--")
        axes[i].set(xlabel=f"BO teacher {name}", ylabel=f"FNN predicted {name}", xscale="log", yscale="log")
    mesh = axes[2].imshow(weights, aspect="auto", cmap="coolwarm")
    axes[2].set(xticks=[0, 1], xticklabels=["log Kp", "log Ki"], ylabel="Rule index", title="Learned consequents")
    fig.colorbar(mesh, ax=axes[2])
    finish(fig, path)
