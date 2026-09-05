# -*- coding: utf-8 -*-
"""Shared report figure style — single source of truth (dedup P4).

Palette and rcParams previously triplicated in ``make_figures.py``,
``make_s4_figures.py`` and ``make_subreport_figures.py``. Import from here::

    from figstyle import apply_style
    apply_style(plt)

``make_figures.py`` re-exports every name below, so existing
``from make_figures import ...`` statements keep working.
"""

from __future__ import annotations

SURFACE = "#fcfcfb"     # 图表面
INK = "#0b0b0b"         # 主文字
INK2 = "#52514e"        # 次文字
MUTED = "#898781"       # 弱文字 / 坐标轴
GRID = "#e1e0d9"        # 网格线（发丝线）
BASE = "#c3c2b7"        # 基线

C_AI_SELF = "#2a78d6"   # 类别 1：AI 自整定（FNN / RL）
C_AI_AUTO = "#eb6834"   # 类别 2：AI 自动整定（BO / 安全 BO / LLM Agent）
C_CLASSICAL = "#898781"  # 基线：经典整定（Z-N / IMC，中性灰）

RC_PARAMS: dict[str, object] = {
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Segoe UI"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": INK2,
}


def apply_style(target=None):
    """Apply the shared style to ``pyplot``, ``rcParams`` or any mapping.

    Returns the object that was updated.
    """

    if target is None:
        import matplotlib

        target = matplotlib.rcParams
    params = dict(RC_PARAMS)
    if hasattr(target, "rcParams"):
        target.rcParams.update(params)
    elif hasattr(target, "update"):
        target.update(params)
    else:
        raise TypeError(f"cannot apply figure style to {type(target)!r}")
    return target


__all__ = [
    "SURFACE", "INK", "INK2", "MUTED", "GRID", "BASE",
    "C_AI_SELF", "C_AI_AUTO", "C_CLASSICAL",
    "RC_PARAMS", "apply_style",
]
