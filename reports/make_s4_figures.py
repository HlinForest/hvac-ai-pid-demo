# -*- coding: utf-8 -*-
"""生成 07/08 的新配图：FNN 规则表热力图 与 RL Q 值演化图。

- 只读封存 CSV/npy，不回写任何数据文件
- 复用 make_figures.py 的配色常量
运行：
    python reports/make_s4_figures.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports"))

import matplotlib.pyplot as plt
import numpy as np

from make_figures import (BASE, C_AI_AUTO, C_AI_SELF, C_CLASSICAL, GRID, INK,
                          INK2, MUTED, SURFACE)
from figstyle import apply_style

apply_style(plt)

import csv  # noqa: E402
import io  # noqa: E402

V3 = ROOT / "outputs_review_v3"
FIG = ROOT / "reports" / "分报告" / "figures"


def read_csv(path):
    return list(csv.DictReader(io.open(path, encoding="utf-8-sig", newline="")))


# =====================================================================
# 图 7-3：FNN 5×5 规则表
# =====================================================================
def fig_fnn_rule_table():
    EC = np.asarray([-3.0, -0.75, 0.0, 1.5, 5.0])
    DC = np.asarray([-0.30, -0.05, 0.0, 0.05, 0.30])
    IMC_KP, IMC_KI = 0.5174862694351612, 0.004053625945382633

    dep = np.load(V3 / "fnn_rule_table.npy")
    assert dep.shape == (5, 5, 2)
    # 自检：终表是解析构造的 IMC 残差曲面
    assert abs(dep[0, 0, 0] - IMC_KP * 0.95) < 1e-12
    assert abs(dep[4, 4, 0] - IMC_KP * 1.02) < 1e-12
    assert abs(dep[2, 2, 1] - IMC_KI * 1.2) < 1e-12
    assert len(np.unique(np.round(dep[:, :, 0], 9))) == 3
    assert len(np.unique(np.round(dep[:, :, 1], 9))) == 1

    # 重建 prior_weight=2.0 的拟合表
    def weights(value, centers):
        v = float(np.clip(value, centers[0], centers[-1]))
        high = int(np.searchsorted(centers, v, side="right"))
        high = min(max(high, 1), len(centers) - 1)
        low = high - 1
        return low, high, (v - centers[low]) / max(centers[high] - centers[low], 1e-9)

    samples = read_csv(V3 / "fnn_training_samples.csv")
    assert len(samples) == 4464
    log_sum = np.zeros((5, 5, 2))
    counts = np.zeros((5, 5))
    for s in samples:
        ei, eh, we = weights(float(s["error_c"]), EC)
        di, dh, wd = weights(float(s["error_rate_c_per_min"]), DC)
        tgt = np.log([float(s["label_kp"]), float(s["label_ki"])])
        for a, b, w in ((ei, di, (1 - we) * (1 - wd)), (ei, dh, (1 - we) * wd),
                        (eh, di, we * (1 - wd)), (eh, dh, we * wd)):
            log_sum[a, b] += w * tgt
            counts[a, b] += w
    fit = np.exp((log_sum + 2.0 * np.log([IMC_KP, IMC_KI])) / (counts[:, :, None] + 2.0))
    # 自检：与 fnn_training_history.csv 末行代表格对齐
    last = read_csv(V3 / "fnn_training_history.csv")[-1]
    assert abs(fit[2, 2, 0] - float(last["center_rule_kp"])) < 1e-9
    assert abs(fit[4, 2, 0] - float(last["high_error_rule_kp"])) < 1e-9

    fig = plt.figure(figsize=(13.5, 13.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.05], hspace=0.34, wspace=0.22)
    panels = [
        (gs[0, 0], dep[:, :, 0], "Kp", "(a) 部署终表 Kp：沿 ė 维完全常数、只有 3 级", 0.44, 0.56),
        (gs[0, 1], fit[:, :, 0], "Kp", "(b) prior_weight=2.0 拟合表 Kp：25 格互不相同", 0.44, 0.56),
        (gs[1, 0], dep[:, :, 1], "Ki", "(c) 部署终表 Ki：25 格全表常数", 0.0008, 0.0052),
        (gs[1, 1], fit[:, :, 1], "Ki", "(d) 拟合表 Ki：跨 4 倍", 0.0008, 0.0052),
    ]
    for spec, data, kind, title, vmin, vmax in panels:
        ax = fig.add_subplot(spec)
        im = ax.imshow(data, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"{v:g}" for v in DC], fontsize=9)
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"{v:g}" for v in EC], fontsize=9)
        ax.set_xlabel("ė 中心 (°C/min)", fontsize=9.5)
        ax.set_ylabel("e 中心 (°C)", fontsize=9.5)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
        for i in range(5):
            for j in range(5):
                v = data[i, j]
                txt = f"{v:.6f}" if kind == "Kp" else f"{v * 1e3:.4f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.6,
                        color="#3a1a00" if v > (vmin + vmax) / 2 else "#3a1a00")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (e) 权重矩阵
    ax5 = fig.add_subplot(gs[2, :])
    im5 = ax5.imshow(counts, cmap="YlGnBu", aspect="auto")
    ax5.set_xticks(range(5))
    ax5.set_xticklabels([f"{v:g}" for v in DC], fontsize=9)
    ax5.set_yticks(range(5))
    ax5.set_yticklabels([f"{v:g}" for v in EC], fontsize=9)
    ax5.set_xlabel("ė 中心 (°C/min)", fontsize=9.5)
    ax5.set_ylabel("e 中心 (°C)", fontsize=9.5)
    ax5.set_title("(e) 每格累计标签权重：中心格独占 37.8%，四角不足 0.5%", fontsize=11,
                  color=INK, loc="left", pad=8)
    for i in range(5):
        for j in range(5):
            ax5.text(j, i, f"{counts[i, j]:.0f}", ha="center", va="center", fontsize=8,
                     color="#08306b" if counts[i, j] < 900 else "#f7fbff")
    ax5.add_patch(plt.Rectangle((1.5, 1.5), 1, 1, fill=False, edgecolor="#c0392b", lw=2.2))
    ax5.text(2, 2.42, "1685 / 4464 = 37.8%", ha="center", fontsize=8.5, color="#c0392b")
    plt.colorbar(im5, ax=ax5, fraction=0.023, pad=0.012)

    fig.suptitle("图 7-3  FNN 5×5 规则表：部署终表（左列）与 prior_weight=2.0 拟合表（右列）逐格对比（v3 封存数据）",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.5, 0.035,
             "终表沿 ė 维完全常数、Kp 仅 3 级、Ki 全表常数 —— 这是「解析构造」而非「拟合产出」的直接证据"
             "；三条构造规则：e 行 0–1 = IMC×0.95，e 行 2 = IMC 原值，e 行 3–4 = IMC×1.02，Ki = IMC Ki×1.20",
             ha="center", fontsize=9.2, color=INK2)

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "fig_fnn_rule_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# =====================================================================
# 图 8-3：RL Q 值演化
# =====================================================================
def fig_rl_q_evolution():
    rows = read_csv(V3 / "rl_training_transitions.csv")
    assert len(rows) == 36000
    alpha = 0.12
    ACTS = [(0.75, 0.75), (0.75, 1.00), (0.75, 1.30), (1.00, 0.75), (1.00, 1.00),
            (1.00, 1.30), (1.30, 0.75), (1.30, 1.00), (1.30, 1.30)]
    ts = (3, 2, 1)

    q = np.zeros((5, 5, 3, 9))
    q[:, :, :, 4] = 1e-6
    snap_eps = {25, 50, 100, 200, 300, 450, 600, 750}
    q_curve = {a: [] for a in range(9)}
    eps_axis, cover_curve, td_curve = [], [], []
    seen = set()
    cur_ep, cur_td = None, []

    for r in rows:
        ep = int(float(r["episode"]))
        s = (int(float(r["state_error_bin"])), int(float(r["state_delta_bin"])),
             int(float(r["state_command_bin"])))
        a = int(float(r["action"]))
        q[s + (a,)] += alpha * float(r["td_error"])
        seen.add(s + (a,))
        if ep != cur_ep:
            if cur_ep is not None:
                eps_axis.append(cur_ep)
                cover_curve.append(len(seen))
                td_curve.append(float(np.mean(np.abs(cur_td))) if cur_td else 0.0)
            cur_ep, cur_td = ep, []
        cur_td.append(float(r["td_error"]))
        if ep in snap_eps:
            for k in range(9):
                q_curve[k].append((ep, q[ts][k]))
    # 收尾
    eps_axis.append(cur_ep)
    cover_curve.append(len(seen))
    td_curve.append(float(np.mean(np.abs(cur_td))) if cur_td else 0.0)

    dep_q = np.load(V3 / "rl_q_table.npy")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 10.6),
                                   gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.42})

    colors = plt.cm.viridis(np.linspace(0.05, 0.92, 9))
    for k in range(9):
        pts = q_curve[k]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax1.plot(xs, ys, color=colors[k], lw=1.5,
                 label=f"a{k} ({ACTS[k][0]:g}, {ACTS[k][1]:g})")
    for x, lab in ((42, "ep42\n热状态覆盖 100%"), (450, "ep450\n最佳检查点 32.77"),
                   (661, "ep661\nε 触底 0.03"), (750, "ep750\n训练结束")):
        ax1.axvline(x, color=MUTED, ls="--", lw=1.0)
        ax1.text(x, ax1.get_ylim()[1], lab.replace("\n", " "), fontsize=8.6, color=INK2,
                 ha="center", va="top", rotation=0)
    ax1.set_xlabel("训练回合", fontsize=10)
    ax1.set_ylabel("Q 值", fontsize=10)
    ax1.set_title("(a) 高频状态 s=(e_bin=3, ė_bin=2, cmd_bin=1) 的 9 个动作 Q 值演化", fontsize=12,
                  color=INK, loc="left", pad=8)
    ax1.legend(fontsize=8.4, frameon=False, ncol=3, loc="upper left")
    ax1.grid(True, color=GRID, lw=0.6)
    ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    fin = q[ts]
    best = int(np.argmax(fin))
    ax1.annotate(f"学到的 argmax = a{best} ({ACTS[best][0]:g}, {ACTS[best][1]:g})\nQ = {fin[best]:+.3f}",
                 xy=(750, fin[best]), xytext=(560, fin[best] + 1.1), fontsize=9.5,
                 color=C_AI_SELF, arrowprops=dict(arrowstyle="->", color=C_AI_SELF, lw=1.2))
    dep_a = int(np.argmax(dep_q[ts]))
    ax1.axhline(fin[dep_a], color=C_AI_AUTO, ls=":", lw=1.6)
    ax1.annotate(f"部署表强制 a{dep_a} ({ACTS[dep_a][0]:g}, {ACTS[dep_a][1]:g})\nQ = {fin[dep_a]:+.3f}",
                 xy=(750, fin[dep_a]), xytext=(560, fin[dep_a] - 1.4), fontsize=9.5,
                 color=C_AI_AUTO, arrowprops=dict(arrowstyle="->", color=C_AI_AUTO, lw=1.2))
    ax1.text(0.5, -0.16, "α = 0.12、γ = 0.94 全程常数不衰减；Q(s,a) = 1e-6·[a=4] + 0.12·Σδ（由 36000 条转移精确重建）",
             transform=ax1.transAxes, ha="center", fontsize=9, color=INK2)

    # (b) 覆盖 + TD
    ax2.plot(eps_axis, cover_curve, color=C_AI_SELF, lw=1.9, marker="o", ms=3.5,
             markevery=[eps_axis.index(e) for e in (1, 50, 200, 450, 750) if e in eps_axis])
    ax2.set_xlabel("训练回合", fontsize=10)
    ax2.set_ylabel("已被更新的 (s, a) 条目数（共 675）", fontsize=10, color=C_AI_SELF)
    ax2.tick_params(axis="y", labelcolor=C_AI_SELF)
    ax2.set_ylim(0, 700)
    for e in (1, 50, 200, 450, 750):
        if e in eps_axis:
            i = eps_axis.index(e)
            ax2.annotate(f"ep{e}: {cover_curve[i]}", (e, cover_curve[i]), fontsize=8.6,
                         color=C_AI_SELF, xytext=(6, -11), textcoords="offset points")
    ax2b = ax2.twinx()
    ax2b.plot(eps_axis, td_curve, color=MUTED, lw=0.9, alpha=0.65, label="每回合平均 |TD|")
    win = 20
    smooth = [float(np.mean(td_curve[max(0, i - win + 1):i + 1])) for i in range(len(td_curve))]
    ax2b.plot(eps_axis, smooth, color=C_AI_AUTO, lw=2.0, label=f"{win} 回合滑动平均")
    ax2b.set_ylabel("平均 |TD error|", fontsize=10, color=C_AI_AUTO)
    ax2b.tick_params(axis="y", labelcolor=C_AI_AUTO)
    ax2b.legend(fontsize=8.8, frameon=False, loc="center right")
    ax2.set_title("(b) Q 表内部参数的演化：更新覆盖面与 TD 误差", fontsize=12, color=INK,
                  loc="left", pad=8)
    ax2.grid(True, color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    for s in ("top",):
        ax2.spines[s].set_visible(False)
        ax2b.spines[s].set_visible(False)
    ax2.text(0.99, 0.28, "675 个条目中 372 个从未被更新\n其中 120 个因动作屏蔽结构性不可达\n其余是「可达但从未走到」",
             transform=ax2.transAxes, ha="right", fontsize=9, color=INK2,
             bbox=dict(fc="white", ec=BASE, lw=0.6, boxstyle="round,pad=0.5"))

    fig.suptitle("图 8-3  RL Q 值演化：高频状态的 9 个动作 Q 值与全表更新覆盖（由 36000 条转移重建）",
                 fontsize=13, fontweight="bold", color=INK)
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "fig_rl_q_evolution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)
    print(f"  s={ts} 终值: " + " ".join(f"a{i}={fin[i]:+.4f}" for i in range(9)))
    print(f"  学到的 argmax=a{best}, 部署表 argmax=a{dep_a}, 覆盖 {cover_curve[-1]}/675")


def main():
    fig_fnn_rule_table()
    fig_rl_q_evolution()


if __name__ == "__main__":
    main()
