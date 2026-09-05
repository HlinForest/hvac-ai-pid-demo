# -*- coding: utf-8 -*-
"""生成《HVAC AI-PI 自动整定实验报告》所需的框图与对比图。

数据来源（均为项目已有实验结果，直接内嵌并在报告中注明出处）：
- outputs_tuning_benchmark/tuning_runtime_report.md   （调参耗时基准，Windows 10 / Python 3.11.7）
- outputs_review_v3/holdout_summary.csv               （80 场景封存验收 holdout 汇总）
- hvac_pid/plant.py、hvac_pid/ai_controllers.py       （对象模型与算法结构）

输出：reports/figures/fig1_系统结构框图.png
      reports/figures/fig2_实验流程框图.png
      reports/figures/fig10_整定耗时对比.png
      reports/figures/fig11_控制性能对比.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Single source of truth for palette + rcParams (dedup P4); re-exported here
# so existing `from make_figures import ...` statements keep working.
try:
    from reports.figstyle import (  # noqa: F401
        BASE, C_AI_AUTO, C_AI_SELF, C_CLASSICAL, GRID, INK, INK2, MUTED,
        RC_PARAMS, SURFACE, apply_style,
    )
except ImportError:
    from figstyle import (  # noqa: F401
        BASE, C_AI_AUTO, C_AI_SELF, C_CLASSICAL, GRID, INK, INK2, MUTED,
        RC_PARAMS, SURFACE, apply_style,
    )

apply_style(plt)

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# 框图绘图原语
# ----------------------------------------------------------------------------
def box(ax, x, y, w, h, text, *, face="white", edge=BASE, lw=1.2,
        tsize=10, tcolor=INK, sub=None, subsize=8, subcolor=INK2, dashed=False):
    """圆角框：主标题 + 可选副标题（副标题为约束/参数说明）。"""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=face, edgecolor=edge,
        linewidth=lw, linestyle="--" if dashed else "-",
        mutation_aspect=1.0, zorder=2,
    )
    ax.add_patch(patch)
    if sub:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center",
                fontsize=tsize, color=tcolor, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=subsize, color=subcolor, zorder=3, linespacing=1.5)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=tsize, color=tcolor, fontweight="bold", zorder=3)
    return patch


def arrow(ax, x1, y1, x2, y2, *, color=INK2, lw=1.4, style="-|>",
          dashed=False, shrink=2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
        color=color, linewidth=lw,
        linestyle=(0, (4, 3)) if dashed else "-",
        shrinkA=shrink, shrinkB=shrink, zorder=1,
    ))


def wash(hex_color, alpha=0.10):
    return (*plt.matplotlib.colors.to_rgb(hex_color), alpha)


# ----------------------------------------------------------------------------
# 图 1：系统结构框图（PI 内环 + AI 增益调度外层）
# ----------------------------------------------------------------------------
def fig1_system_block():
    fig, ax = plt.subplots(figsize=(12.4, 6.6))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    # ---- AI 增益调度层背景区（上层，2 s 周期） ----
    ai_bg = FancyBboxPatch((0.55, 4.35), 11.3, 1.95,
                           boxstyle="round,pad=0.05,rounding_size=0.12",
                           facecolor=SURFACE, edgecolor=GRID, linewidth=1.0, zorder=0)
    ax.add_patch(ai_bg)
    ax.text(0.85, 6.05, "AI 增益调度层（低频，每 2 s 输出 Kp/Ki，±10~25% 变化限幅 + IMC 回退）",
            fontsize=10.5, color=INK2, ha="left", va="center", zorder=3)

    # 自整定（在线）：FNN / RL
    box(ax, 1.0, 4.55, 3.4, 1.15, "自整定（在线）",
        face=wash(C_AI_SELF), edge=C_AI_SELF, tcolor=INK,
        sub="FNN 5×5 TSK 规则面\nRL 表格式 Q 策略（5×5×3×9）", subsize=8.5)
    # 自动整定（离线）：BO / 安全 BO / LLM Agent
    box(ax, 5.0, 4.55, 3.6, 1.15, "自动整定（离线）",
        face=wash(C_AI_AUTO), edge=C_AI_AUTO, tcolor=INK,
        sub="贝叶斯 BO / 风险感知安全 BO\nLLM Agent（监督式）", subsize=8.5)
    # 经典整定（对照）
    box(ax, 9.2, 4.55, 2.25, 1.15, "经典整定（对照）",
        face="white", edge=C_CLASSICAL, tcolor=INK,
        sub="Z-N / IMC-λ\n投运前一次固定", subsize=8.5)

    # 调度层输出 Kp/Ki → PI
    arrow(ax, 2.7, 4.55, 2.7, 3.42, color=C_AI_SELF, lw=1.8)
    ax.text(2.95, 3.98, "Kp / Ki", fontsize=9, color=C_AI_SELF, ha="left", va="center")
    # 调度层输入：来自反馈通道（Tz 输出线上引出，供 AI 调度层使用）
    arrow(ax, 10.65, 2.28, 10.65, 4.35, color=C_AI_SELF, lw=1.2, dashed=True)
    ax.text(10.78, 3.55, "Tz、e、工况\n（调度特征）", fontsize=8, color=INK2,
            ha="left", va="center", linespacing=1.4)

    # ---- 内环控制回路（下层，100 ms） ----
    loop_bg = FancyBboxPatch((0.55, 1.42), 11.3, 1.48,
                             boxstyle="round,pad=0.05,rounding_size=0.12",
                             facecolor=SURFACE, edgecolor=GRID, linewidth=1.0, zorder=0)
    ax.add_patch(loop_bg)
    ax.text(6.3, 1.61, "PI 内环控制回路（100 ms，反积分饱和）", fontsize=10,
            color=INK2, ha="center", va="center", zorder=3)

    # 比较点
    (cx, cy), r = (1.35, 2.2), 0.2
    ax.add_patch(plt.Circle((cx, cy), r, facecolor="white", edgecolor=INK2, lw=1.2, zorder=2))
    ax.text(cx, cy + 0.02, "Σ", ha="center", va="center", fontsize=11, color=INK, zorder=3)
    ax.text(cx, cy + 0.62, "设定值 Tsp", fontsize=9, color=INK, ha="center")
    arrow(ax, cx, cy + 0.55, cx, cy + r, lw=1.6)

    box(ax, 2.0, 1.75, 2.3, 0.95, "PI 控制器",
        sub="Kp·e + Ki∫e", subsize=8.5)
    box(ax, 4.75, 1.75, 2.9, 0.95, "执行器约束",
        sub="25% 最小容量 · 1% 量化\n5%/min 变频斜率 · 启停驻留", subsize=7.5)
    box(ax, 8.1, 1.75, 2.2, 0.95, "被控对象",
        sub="纯延迟 + 一阶滞后\n3R2C 热模型", subsize=8)

    # 回路连线
    arrow(ax, cx + r, cy, 2.0, cy, lw=1.6)
    ax.text(1.75, cy + 0.14, "e", fontsize=9, color=INK2, ha="center")
    arrow(ax, 4.3, cy, 4.75, cy, lw=1.6)
    arrow(ax, 7.65, cy, 8.1, cy, lw=1.6)
    ax.text(7.87, cy + 0.14, "Q", fontsize=9, color=INK2, ha="center")

    # 输出温度 + 反馈
    arrow(ax, 10.3, cy, 11.35, cy, lw=1.6)
    ax.text(11.42, cy, "Tz\n机柜温度", fontsize=9, color=INK, ha="left", va="center",
            linespacing=1.4)
    ax.plot([11.35, 11.35, 1.35, 1.35], [cy, 0.9, 0.9, cy - r], color=INK2, lw=1.4, zorder=1)
    arrow(ax, 1.35, 0.9, 1.35, cy - r - 0.02, lw=1.4)
    ax.text(6.3, 1.12, "温度反馈（1 min 仿真步长 / 传感器采样）", fontsize=8.5,
            color=MUTED, ha="center", va="center")
    ax.text(1.62, 1.72, "−", fontsize=11, color=INK, ha="center", va="center")
    ax.text(1.05, 1.72, "+", fontsize=10, color=INK, ha="center", va="center")

    ax.set_title("图 1  HVAC AI-PI 控制系统结构：100 ms PI 内环 + 2 s AI 增益调度层",
                 fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_系统结构框图.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 图 2：实验流程框图
# ----------------------------------------------------------------------------
def fig2_pipeline_block():
    fig, ax = plt.subplots(figsize=(11.6, 8.4))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 8.4)
    ax.axis("off")

    # 阶段 1：场景与数据
    box(ax, 3.3, 7.35, 5.0, 0.85, "场景数据集生成",
        sub="48 训练 / 16 验证 / 16 测试 · 逐场景 SHA-256 清单", subsize=8.5)
    ax.text(0.35, 7.78, "① 数据", fontsize=10, color=MUTED, ha="left", va="center")

    # 阶段 2：三条整定路线（并列）
    box(ax, 0.45, 5.6, 3.3, 1.15, "经典整定（基线）",
        face="white", edge=C_CLASSICAL,
        sub="Z-N 反应曲线法\nIMC-λ（训练集整定）", subsize=8.5)
    box(ax, 4.15, 5.6, 3.3, 1.15, "AI 自动整定（离线）",
        face=wash(C_AI_AUTO), edge=C_AI_AUTO,
        sub="贝叶斯 BO · 风险感知安全 BO\nLLM Agent（真实调用 qwen-max）", subsize=8.5)
    box(ax, 7.85, 5.6, 3.3, 1.15, "AI 自整定（训练）",
        face=wash(C_AI_SELF), edge=C_AI_SELF,
        sub="FNN 规则面（4,464 样本）\nRL Q 学习（750 回合）", subsize=8.5)
    ax.text(0.35, 6.55, "② 整定/训练", fontsize=10, color=MUTED, ha="left", va="center")

    for x in (2.1, 5.8, 9.5):
        arrow(ax, x, 7.35, x, 6.78)

    # 阶段 3：交叉验证与封存验收
    box(ax, 3.3, 4.35, 5.0, 0.85, "封存验收（v3 证据链）",
        sub="5 种子 × 16 场景 = 80 · 数据泄漏拒绝 · 防回退报告", subsize=8.2)
    ax.text(0.35, 4.78, "③ 验收", fontsize=10, color=MUTED, ha="left", va="center")
    # 三路汇入验收框
    for xs in (2.1, 5.8, 9.5):
        ax.plot([xs, xs], [5.6, 5.42], color=INK2, lw=1.4, zorder=1)
    ax.plot([2.1, 9.5], [5.42, 5.42], color=INK2, lw=1.4, zorder=1)
    arrow(ax, 5.8, 5.42, 5.8, 5.22, lw=1.4)

    # 阶段 4：模型交叉验证（旁注）
    box(ax, 8.9, 4.3, 2.25, 0.95, "三层交叉验证",
        sub="1 min 离散 vs DOP853\nFOPDT vs 3R2C\nOpenModelica 验证", subsize=7,
        face="white", edge=BASE)

    # 阶段 5：PC-SIL
    arrow(ax, 5.8, 4.35, 5.8, 3.9)
    box(ax, 3.3, 3.05, 5.0, 0.85, "PC 软件在环（SIL）",
        sub="200× 加速 · 100 ms PI / 2 s AI · 75 条 Py/C++ 一致性向量", subsize=8.2)
    ax.text(0.35, 3.48, "④ 在环", fontsize=10, color=MUTED, ha="left", va="center")

    # 阶段 6：策略导出与部署
    arrow(ax, 5.8, 3.05, 5.8, 2.6)
    box(ax, 3.3, 1.75, 5.0, 0.85, "策略导出（冻结）",
        sub="generated_policy.hpp + CRC v3 校验", subsize=8.5)
    ax.text(0.35, 2.18, "⑤ 部署", fontsize=10, color=MUTED, ha="left", va="center")

    arrow(ax, 5.8, 1.75, 5.8, 1.3)
    box(ax, 3.3, 0.45, 5.0, 0.85, "ESP32 固件部署",
        sub="RAM 6.7% · Flash 22.6% · Modbus-RTU", subsize=8.5)
    ax.text(0.35, 0.88, "⑥ 板级", fontsize=10, color=MUTED, ha="left", va="center")

    # 阶段 7：真机验收（未开始，虚线）
    box(ax, 8.9, 0.45, 2.25, 0.85, "真机 10 min 验收",
        sub="串口遥测 + WCET 实测\n（未开始）", subsize=7.5,
        face="white", edge=MUTED, dashed=True)
    arrow(ax, 8.3, 0.88, 8.9, 0.88, dashed=True, color=MUTED)

    ax.set_title("图 2  实验与验证流程：从场景数据集到 ESP32 部署",
                 fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_实验流程框图.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 图 10：整定耗时对比（两块小图：PC 离线耗时 / 等效对象时间）
# ----------------------------------------------------------------------------
METHODS = ["Z-N", "IMC-λ", "贝叶斯 BO", "FNN 自整定", "RL 自整定"]
CAT_COLOR = [C_CLASSICAL, C_CLASSICAL, C_AI_AUTO, C_AI_SELF, C_AI_SELF]

PC_TIME = [0.163, 3.030, 1.418, 7.652, 16.223]       # 秒
PLANT_EQ_TIME = [168, 1008, 528, 4181, 4018]          # 对象小时


def fig3_time_bars():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9))
    y = range(len(METHODS))

    for ax, values, unit, title in (
        (axes[0], PC_TIME, "s", "（a）PC 离线整定/训练耗时（秒）"),
        (axes[1], PLANT_EQ_TIME, "h", "（b）等效虚拟对象时间（对象小时）"),
    ):
        ax.barh(y, values, height=0.58, color=CAT_COLOR, edgecolor="none")
        ax.set_yticks(list(y), METHODS)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11.5, color=INK, pad=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(f"耗时（{unit}）", fontsize=9.5)
        # 柱端数值直标（每个值都标注，兼作浅色系列的对比度 relief）
        for yi, v in zip(y, values):
            label = f"{v:.1f}" if v >= 100 else f"{v:.2f}"
            ax.text(v + max(values) * 0.012, yi, label, va="center", ha="left",
                    fontsize=9, color=INK)
        ax.set_xlim(0, max(values) * 1.14)

    # 图例（≥2 个类别 → 必须有图例）
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_CLASSICAL, label="经典整定（非 AI）"),
        plt.Rectangle((0, 0), 1, 1, color=C_AI_AUTO, label="AI 自动整定（离线搜索）"),
        plt.Rectangle((0, 0), 1, 1, color=C_AI_SELF, label="AI 自整定（在线训练）"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("图 10  各整定方法的耗时对比：PC 计算 vs 等效真实对象试验",
                 fontsize=13, fontweight="bold", y=1.10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_整定耗时对比.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 图 11：控制性能对比（80 场景封存验收 holdout 目标值，越低越好）
# ----------------------------------------------------------------------------
HOLDOUT = [58.19, 39.53, 39.72, 39.26, 37.61]


def fig4_holdout_bars():
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    y = range(len(METHODS))
    ax.barh(y, HOLDOUT, height=0.58, color=CAT_COLOR, edgecolor="none")
    ax.set_yticks(list(y), METHODS)
    ax.invert_yaxis()
    ax.set_xlabel("综合目标值（越低越好，含跟踪误差与能耗惩罚）", fontsize=9.5)
    ax.set_title("图 11  80 场景封存验收 holdout 综合目标值", fontsize=13,
                 fontweight="bold", pad=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for yi, v in zip(y, HOLDOUT):
        ax.text(v + 0.7, yi, f"{v:.2f}", va="center", ha="left", fontsize=9, color=INK)
    ax.set_xlim(0, max(HOLDOUT) * 1.14)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_CLASSICAL, label="经典整定（非 AI）"),
        plt.Rectangle((0, 0), 1, 1, color=C_AI_AUTO, label="AI 自动整定（离线搜索）"),
        plt.Rectangle((0, 0), 1, 1, color=C_AI_SELF, label="AI 自整定（在线训练）"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig11_控制性能对比.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_system_block()
    fig2_pipeline_block()
    fig3_time_bars()
    fig4_holdout_bars()
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"生成 {p.name}")
    print("完成。")
