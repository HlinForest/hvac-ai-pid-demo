"""Central safety/trust-region constants (v4 single source of truth).

All controllers, tuners and demos MUST import from here instead of
hard-coding 0.10 / 0.25 / 0.35 in place.  The deployed default is ±10%.
"""
from __future__ import annotations

from dataclasses import dataclass

# Deployed default: single supervisory step may move Kp/Ki by at most ±10%.
MAX_FRACTIONAL_GAIN_CHANGE: float = 0.10

# Hard gain bounds shared by Python + embedded policy.
KP_BOUNDS: tuple[float, float] = (0.002, 1.5)
KI_BOUNDS: tuple[float, float] = (1e-5, 0.08)

# Risk gate defaults (shared by Safe BO / LLM supervisor).
RISK_WEIGHT: float = 0.75
MAX_UNDERSHOOT_C: float = 2.0
VALIDATION_TOLERANCE: float = 1.02
MAX_RISK_RATIO_TO_BASELINE: float = 1.25
SAFETY_REPEATS: int = 2


@dataclass(frozen=True)
class SafetyConfig:
    max_fractional_change: float = MAX_FRACTIONAL_GAIN_CHANGE
    kp_bounds: tuple[float, float] = KP_BOUNDS
    ki_bounds: tuple[float, float] = KI_BOUNDS
    risk_weight: float = RISK_WEIGHT
    max_undershoot_c: float = MAX_UNDERSHOOT_C
    validation_tolerance: float = VALIDATION_TOLERANCE
    max_risk_ratio_to_baseline: float = MAX_RISK_RATIO_TO_BASELINE
    repeats: int = SAFETY_REPEATS
