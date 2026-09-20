"""Fixed, disjoint object splits. No training routine receives the test split."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from hvac_pid.core import Scenario


def make_splits(seed: int = 2026):
    rng = np.random.default_rng(seed)
    splits = {}
    for name, count in [("train", 48), ("validation", 12), ("test", 12)]:
        splits[name] = [asdict(Scenario(gain=round(rng.uniform(8, 12), 6),
                                       tau=round(rng.uniform(12, 32), 6),
                                       delay=round(rng.uniform(1, 5), 1)))
                        for _ in range(count)]
    return {"seed": seed, "splits": splits}


def load_splits(path: Path):
    return {name: [Scenario(**s) for s in scenarios]
            for name, scenarios in json.loads(path.read_text(encoding="utf-8"))["splits"].items()}
