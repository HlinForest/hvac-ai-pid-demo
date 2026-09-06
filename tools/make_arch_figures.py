"""Style-free architecture diagrams for the main report (Ch2).

Writes docs/figures/run_arch.png (runtime architecture) and
docs/figures/evidence_flow.png (evidence pipeline).  Plain matplotlib
boxes+arrows; no external assets; deterministic output.

Usage: python tools/make_arch_figures.py [--output docs/figures]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
INK, ACCENT, FILL, OFFLINE = "#111827", "#1f4e79", "#e8eef5", "#7a5a00"


def _boxes(ax, items: list[tuple[str, str]], vertical: bool = False):
    n = len(items)
    patches = []
    for i, (title, sub) in enumerate(items):
        if vertical:
            xy = (0.05, 0.92 - (i + 1) * (0.88 / n))
            w, h = 0.90, 0.88 / n - 0.03
        else:
            xy = (0.02 + i * (0.96 / n), 0.30)
            w, h = 0.96 / n - 0.03, 0.45
        box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.01",
                             facecolor=FILL, edgecolor=ACCENT, lw=1.4,
                             transform=ax.transAxes)
        ax.add_patch(box)
        cx, cy = xy[0] + w / 2, xy[1] + h / 2
        dy_t, dy_s = (0.06, 0.09) if not vertical else (0.022, 0.026)
        ax.text(cx, cy + dy_t, title, ha="center", va="center",
                fontsize=9, fontweight="bold", color=INK, transform=ax.transAxes)
        ax.text(cx, cy - dy_s, sub, ha="center", va="center",
                fontsize=7, color=INK, transform=ax.transAxes)
        patches.append((cx, cy, w, h))
    for (x0, y0, w0, h0), (x1, y1, w1, h1) in zip(patches, patches[1:]):
        if vertical:
            start, end = (x0, y0 - h0 / 2), (x1, y1 + h1 / 2)
        else:
            start, end = (x0 + w0 / 2, y0), (x1 - w1 / 2, y1)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", color=ACCENT,
                                     lw=1.4, mutation_scale=12, transform=ax.transAxes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.axis("off")
    ax.set_title("Runtime: offline tuning -> frozen policy -> online scheduling -> safe PI -> plant",
                 fontsize=10, fontweight="bold", color=ACCENT)
    _boxes(ax, [
        ("offline\ntuning/training", "48 train / 16 val"),
        ("frozen\nPolicyBundle", "hash-locked gains"),
        ("online gain\nscheduling", "FNN/RL; LLM offline only"),
        ("safe PI\n+/-10%", "fallback to IMC"),
        ("actuator +\n3R2C plant", "delay/slew/limits"),
        ("sensor\n+ filter", "noise -> closed loop"),
    ])
    fig.tight_layout()
    fig.savefig(args.output / "run_arch.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.axis("off")
    ax.set_title("Evidence: config -> splits -> freeze -> test -> stats -> report",
                 fontsize=10, fontweight="bold", color=ACCENT)
    _boxes(ax, [
        ("v4.yaml\nsingle source", "manifest + code SHA"),
        ("splits\n48/16/16/80", "train/val/qual/sealed"),
        ("train + select\n+ gate(qual)", "freeze tables"),
        ("sealed test\n80x7", "record-only, new seeds"),
        ("stats\npaired ratio + CI", "acceptance 4-gate"),
        ("report\nmain + annex", "audited numbers"),
    ], vertical=True)
    fig.tight_layout()
    fig.savefig(args.output / "evidence_flow.png", dpi=150)
    plt.close(fig)
    print(f"arch figures -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
