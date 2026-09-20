"""A 27-rule zero-order TSK network with fixed Gaussian memberships.

Only rule consequents are learned. This is supervised imitation of BO labels,
not a deep network or an online adaptive controller.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core import Gains
from .tuning import Identified


INPUT_LOW = np.asarray([8.0, 12.0, 1.0])
INPUT_HIGH = np.asarray([12.0, 32.0, 5.0])


# region memberships
def features(inputs):
    x = (np.atleast_2d(inputs) - INPUT_LOW) / (INPUT_HIGH - INPUT_LOW)
    centers = np.asarray([0.0, 0.5, 1.0])
    membership = np.exp(-0.5 * ((x[:, :, None] - centers) / 0.35) ** 2)
    rules = np.einsum("ni,nj,nk->nijk", membership[:, 0], membership[:, 1], membership[:, 2])
    rules = rules.reshape(-1, 27)
    return rules / rules.sum(axis=1, keepdims=True)
# endregion memberships


@dataclass
class FNN:
    consequents: np.ndarray

    # region fit
    @classmethod
    def fit(cls, inputs, gains, ridge: float = 0.001):
        activation = features(inputs)
        target = np.log(np.asarray(gains))
        weights = np.linalg.solve(activation.T @ activation + ridge * np.eye(27),
                                  activation.T @ target)
        return cls(weights)

    def predict(self, model: Identified) -> Gains:
        activation = features([[model.gain, model.tau, model.delay]])
        kp, ki = np.exp(activation @ self.consequents)[0]
        return Gains(float(kp), float(ki))
    # endregion fit

    def save(self, path: Path):
        np.savez(path, consequents=self.consequents)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["consequents"].copy())
