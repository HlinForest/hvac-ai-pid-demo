from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .config import Scenario
from .tuning import GainBounds


class GainScheduler:
    """Supervised context -> log(Kp, Ki) model with OOD detection."""

    def __init__(self, seed: int = 0, bounds: GainBounds | None = None):
        self.seed = seed
        self.bounds = bounds or GainBounds()
        self.model = ExtraTreesRegressor(
            n_estimators=240,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=seed,
            n_jobs=-1,
        )
        self.context_min: np.ndarray | None = None
        self.context_max: np.ndarray | None = None

    def fit(self, rows: list[dict[str, float]]) -> "GainScheduler":
        x = np.asarray([[row[name] for name in Scenario.FEATURE_NAMES] for row in rows], dtype=float)
        y = np.log(np.asarray([[row["label_kp"], row["label_ki"]] for row in rows], dtype=float))
        self.model.fit(x, y)
        self.context_min = x.min(axis=0)
        self.context_max = x.max(axis=0)
        return self

    def predict_gains(self, context: np.ndarray) -> tuple[float, float, bool]:
        if self.context_min is None or self.context_max is None:
            raise RuntimeError("GainScheduler must be fitted before prediction")
        context = np.asarray(context, dtype=float)
        span = np.maximum(self.context_max - self.context_min, 1e-12)
        margin = 0.08 * span
        out_of_domain = bool(
            np.any(context < self.context_min - margin) or np.any(context > self.context_max + margin)
        )
        clipped_context = np.clip(context, self.context_min, self.context_max)
        kp, ki = np.exp(self.model.predict(clipped_context.reshape(1, -1))[0])
        kp = float(np.clip(kp, *self.bounds.kp))
        ki = float(np.clip(ki, *self.bounds.ki))
        return kp, ki, out_of_domain

    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))

    @staticmethod
    def load(path: str | Path) -> "GainScheduler":
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, GainScheduler):
            raise TypeError("The file does not contain a GainScheduler")
        return loaded
