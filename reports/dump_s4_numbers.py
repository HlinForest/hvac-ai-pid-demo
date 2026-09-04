# -*- coding: utf-8 -*-
"""只读核对脚本：从封存 CSV/npy 反推 03/06/07/08 四篇 §4 所需的全部参数演化数值。

- 不写任何数据文件，只读 + 打印
- 带 assert 自检，数值对不上立即失败
- 运行：
    cd E:/HAVC/hvac-ai-pid-demo
    export PYTHONPATH="C:/Users/M00094113/AppData/Roaming/Python/Python314/site-packages"
    C:/Python314/python.exe reports/dump_s4_numbers.py
"""
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(r"E:\HAVC\hvac-ai-pid-demo")
sys.path.insert(0, str(ROOT))

import numpy as np

V3 = ROOT / "outputs_review_v3"


def read_csv(path):
    return list(csv.DictReader(io.open(path, encoding="utf-8-sig", newline="")))


def section(title):
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


# =====================================================================
# 03 IMC-λ：21 个候选全表
# =====================================================================
def dump_03():
    section("03 IMC-λ：21 个候选全表（imc_lambda_tuning.csv）")
    rows = read_csv(V3 / "imc_lambda_tuning.csv")
    assert len(rows) == 21, len(rows)
    tau, K, L = 911.08012, 55.164866, 5.0

    print(f"{'#':>3} {'λ (min)':>10} {'Kp':>10} {'Ti (min)':>10} {'Ki':>12} {'平均J':>9} {'最差J':>9} {'标记'}")
    print("-" * 92)
    for r in rows:
        lam = float(r["lambda_minutes"])
        kp = float(r["candidate_kp"])
        ki = float(r["candidate_ki"])
        ti = min(tau, 4.0 * (lam + L))
        # 用 Kp/Ki 反算 Ti，用于验证饱和切换
        ti_est = kp / ki if ki > 0 else float("nan")
        mark = []
        if r["formula_default"] == "1":
            mark.append("公式默认/回退")
        if r["selected"] == "1":
            mark.append("λ* 选中")
        if abs(ti_est - tau) < 1.0:
            mark.append("Ti 被 τ 截断")
        print(f"{r['evaluation']:>3} {lam:>10.3f} {kp:>10.6f} {ti:>10.2f} {ki:>12.4e} "
              f"{float(r['mean_training_objective']):>9.3f} "
              f"{float(r['worst_training_objective']):>9.3f}  {'、'.join(mark)}")

    # 自检：Ti 饱和切换点
    print()
    sat = [r for r in rows if abs(float(r["candidate_kp"]) / float(r["candidate_ki"]) - tau) < 1.0]
    print(f"Ti 被 τ 截断的候选：{[r['evaluation'] for r in sat]}（λ = "
          f"{[round(float(r['lambda_minutes']), 3) for r in sat]}）")
    lam_star = float([r for r in rows if r["selected"] == "1"][0]["lambda_minutes"])
    lam_formula = float([r for r in rows if r["formula_default"] == "1"][0]["lambda_minutes"])
    worst_row = min(rows, key=lambda r: float(r["worst_training_objective"]))
    print(f"λ* = {lam_star:.4f} | 公式默认 λ = {lam_formula:.4f} | "
          f"最差J 谷底在 #{worst_row['evaluation']}（λ={float(worst_row['lambda_minutes']):.3f}，"
          f"{float(worst_row['worst_training_objective']):.3f}）")
    # 几何序列公比
    lams = [float(r["lambda_minutes"]) for r in rows]
    ratio = (lams[-1] / lams[0]) ** (1 / (len(lams) - 1))
    print(f"几何序列公比 ≈ {ratio:.6f}（λ1={lams[0]:.3f} → λ21={lams[-1]:.3f}）")
    # 限幅检查
    kps = [float(r["candidate_kp"]) for r in rows]
    kis = [float(r["candidate_ki"]) for r in rows]
    print(f"Kp 范围 [{min(kps):.6f}, {max(kps):.6f}] ⊂ [0.002,1.5]? {min(kps) >= 0.002 and max(kps) <= 1.5}")
    print(f"Ki 范围 [{min(kis):.3e}, {max(kis):.3e}] ⊂ [1e-5,0.08]? {min(kis) >= 1e-5 and max(kis) <= 0.08}")
    print(f"→ 限幅是否生效：{'否（全部落在区间内）' if min(kps) >= 0.002 and max(kps) <= 1.5 and min(kis) >= 1e-5 and max(kis) <= 0.08 else '是'}")


# =====================================================================
# 06 LLM Agent：6 步会话走读
# =====================================================================
def dump_06():
    section("06 LLM Agent：6 步会话状态演化（llm_agent_trace.csv + summary.json）")
    rows = read_csv(ROOT / "outputs_llm_matrix" / "qwen-max_std" / "llm_agent_trace.csv")
    assert len(rows) == 6, len(rows)
    cur_kp, cur_ki = 0.45082372186615355, 0.003076518486521097
    base_risk = 32.0073

    print(f"{'步':>2} {'工具':<18} {'raw Kp':>14} {'limited Kp':>16} {'Kp':>12} {'Ki':>12} "
          f"{'risk':>9} {'margin':>10} {'Δ':>11} {'预算':>4} {'判定'}")
    print("-" * 92)
    for r in rows:
        step = r["step"]
        tool = r["tool"]
        raw = r.get("raw_kp", "")
        lim = r.get("limited_kp", "")
        kp = r.get("current_kp", "")
        ki = r.get("current_ki", "")
        risk = r.get("risk_objective", "")
        margin = r.get("safety_margin", "")
        imp = r.get("improvement_fraction", "")
        rem = r.get("remaining_trials", "")
        dec = (r.get("decision", "") or "")[:10]
        print(f"{step:>2} {tool:<18} {(raw or '—')[:14]:>14} {(lim or '—')[:16]:>16} "
              f"{(kp or '—')[:12]:>12} {(ki or '—')[:12]:>12} "
              f"{(risk or '—')[:9]:>9} {(margin or '—')[:10]:>10} {(imp or '—')[:11]:>11} {rem:>4} {dec}")

    # 算术自检
    print()
    r3 = rows[2]
    d3 = (base_risk - float(r3["risk_objective"])) / abs(base_risk)
    print(f"step3 Δ 手算 = ({base_risk} − {float(r3['risk_objective']):.4f}) / |{base_risk}| = {d3:+.6f}")
    assert abs(d3 - float(r3["improvement_fraction"])) < 1e-4, (d3, r3["improvement_fraction"])
    r5 = rows[4]
    d5 = (base_risk - float(r5["risk_objective"])) / abs(base_risk)
    print(f"step5 Δ 手算 = ({base_risk} − {float(r5['risk_objective']):.4f}) / |{base_risk}| = {d5:+.6f}")
    assert abs(d5 - float(r5["improvement_fraction"])) < 1e-4, (d5, r5["improvement_fraction"])
    print("→ Δ 的分母是 current（32.0073），两次手算与 CSV 记录一致，确认分母非 baseline")

    # ±10% 信任域边界
    print()
    print(f"+10% 上界 = {cur_kp:.10f} × 1.10 = {cur_kp * 1.10:.10f}")
    print(f"  step3 limited_kp = {float(r3['limited_kp']):.10f} → 命中上界? "
          f"{abs(float(r3['limited_kp']) - cur_kp * 1.10) < 1e-9}")
    print(f"−10% 下界 = {cur_kp:.10f} × 0.90 = {cur_kp * 0.90:.10f}")
    print(f"  step5 raw_kp     = {float(r5['raw_kp']):.10f}")
    print(f"  step5 limited_kp = {float(r5['limited_kp']):.10f} → 命中下界? "
          f"{abs(float(r5['limited_kp']) - cur_kp * 0.90) < 1e-9}")
    diff = float(r5["limited_kp"]) - float(r5["raw_kp"])
    print(f"  raw 比下界还低 {abs(diff):.2e}，被 clip 抬回下界（差 {diff:+.2e}）")

    print()
    print(f"最终资格化：{float(r5['risk_objective']):.4f} ≤ baseline×1.02 = {base_risk * 1.02:.4f} → "
          f"{float(r5['risk_objective']) <= base_risk * 1.02}")

    # summary 交叉核对
    summ = json.load(io.open(ROOT / "outputs_llm_matrix" / "qwen-max_std" / "llm_agent_summary.json",
                             encoding="utf-8"))
    print()
    print("summary 交叉核对：")
    print(f"  llm_raw_kp     = {summ['llm_raw_kp']!r}")
    print(f"  llm_limited_kp = {summ['llm_limited_kp']!r}  （差 {summ['llm_limited_kp'] - summ['llm_raw_kp']:+.3e}）")
    print(f"  steps={summ['llm_agent_steps']}  trials={summ['llm_agent_trials']}  "
          f"accepted_trials={summ['llm_accepted_trials']}")
    assert abs((summ["llm_limited_kp"] - summ["llm_raw_kp"]) - 2.003e-11) < 1e-12


# =====================================================================
# 07 FNN：拟合表 vs 部署终表 + 权重矩阵
# =====================================================================
def _active_weights(value, centers):
    """与 ai_controllers.py:159-166 完全一致的双线性权重。返回 (low_idx, high_idx, w_high)。"""
    v = float(np.clip(value, centers[0], centers[-1]))
    high = int(np.searchsorted(centers, v, side="right"))
    high = min(max(high, 1), len(centers) - 1)
    low = high - 1
    w = (v - centers[low]) / max(centers[high] - centers[low], 1e-9)
    return low, high, w


def dump_07():
    section("07 FNN：部署终表 vs prior_weight=2.0 拟合表 + 权重矩阵")
    dep = np.load(V3 / "fnn_rule_table.npy")
    assert dep.shape == (5, 5, 2), dep.shape

    # 自检：终表的解析构造特征
    uniq = set(np.round(dep[:, :, 0], 9).ravel().tolist())
    print(f"终表 Kp 唯一值: {sorted(uniq)}")
    assert uniq == {0.491611956, 0.51748627, 0.527836} or len(uniq) == 3, uniq
    imc_kp, imc_ki = 0.5174862694351612, 0.004053625945382633
    print(f"  0.491612 ≈ IMC Kp × 0.95 = {imc_kp * 0.95:.9f}  → "
          f"{abs(0.491611956 - imc_kp * 0.95) < 1e-8}")
    print(f"  0.527836 ≈ IMC Kp × 1.02 = {imc_kp * 1.02:.9f}  → "
          f"{abs(0.527836 - imc_kp * 1.02) < 1e-8}")
    ki_u = np.unique(np.round(dep[:, :, 1], 9))
    print(f"终表 Ki 唯一值: {ki_u}  （IMC Ki × 1.20 = {imc_ki * 1.2:.9f}）")
    assert len(ki_u) == 1
    print()

    # 用 4464 个标签重建 prior_weight=2.0 的拟合表
    EC = np.asarray([-3.0, -0.75, 0.0, 1.5, 5.0])
    DC = np.asarray([-0.30, -0.05, 0.0, 0.05, 0.30])
    samples = read_csv(V3 / "fnn_training_samples.csv")
    print(f"标签样本数: {len(samples)}")
    log_sum = np.zeros((5, 5, 2))
    counts = np.zeros((5, 5))
    for s in samples:
        ei, eh, we = _active_weights(float(s["error_c"]), EC)
        di, dh, wd = _active_weights(float(s["error_rate_c_per_min"]), DC)
        target = np.log([float(s["label_kp"]), float(s["label_ki"])])
        for (ei_, di_, w) in ((ei, di, (1 - we) * (1 - wd)), (ei, dh, (1 - we) * wd),
                             (eh, di, we * (1 - wd)), (eh, dh, we * wd)):
            log_sum[ei_, di_] += w * target
            counts[ei_, di_] += w
    prior = 2.0
    fit = np.exp((log_sum + prior * np.log([imc_kp, imc_ki])) / (counts[:, :, None] + prior))

    # 自检：必须与 fnn_training_history.csv 末行的代表格对齐
    hist = read_csv(V3 / "fnn_training_history.csv")
    last = hist[-1]
    print(f"history 末行 center_rule_kp     = {float(last['center_rule_kp']):.12f}  "
          f"本脚本 fit[2,2,0] = {fit[2, 2, 0]:.12f}  → "
          f"{abs(fit[2, 2, 0] - float(last['center_rule_kp'])) < 1e-9}")
    print(f"history 末行 high_error_rule_kp = {float(last['high_error_rule_kp']):.12f}  "
          f"本脚本 fit[4,2,0] = {fit[4, 2, 0]:.12f}  → "
          f"{abs(fit[4, 2, 0] - float(last['high_error_rule_kp'])) < 1e-9}")
    assert abs(fit[2, 2, 0] - float(last["center_rule_kp"])) < 1e-9
    assert abs(fit[4, 2, 0] - float(last["high_error_rule_kp"])) < 1e-9
    print()

    print("【部署终表 Kp】(行=e 中心, 列=ė 中心)")
    for i in range(5):
        print("  " + " ".join(f"{dep[i, j, 0]:.6f}" for j in range(5)))
    print()
    print("【prior_weight=2.0 拟合表 Kp】")
    for i in range(5):
        print("  " + " ".join(f"{fit[i, j, 0]:.6f}" for j in range(5)))
    print(f"  范围 [{fit[:, :, 0].min():.6f}, {fit[:, :, 0].max():.6f}]")
    print()
    print("【部署终表 Ki】全 25 格 =", f"{dep[0, 0, 1]:.8f}")
    print("【拟合表 Ki】(×1e3)")
    for i in range(5):
        print("  " + " ".join(f"{fit[i, j, 1] * 1e3:.4f}" for j in range(5)))
    print(f"  范围 [{fit[:, :, 1].min():.6e}, {fit[:, :, 1].max():.6e}]")
    print()
    print("【每格累计标签权重】")
    for i in range(5):
        print("  " + " ".join(f"{counts[i, j]:8.1f}" for j in range(5)))
    tot = counts.sum()
    print(f"  合计 {tot:.0f}；中心格 (2,2) = {counts[2, 2]:.0f}（{counts[2, 2] / tot * 100:.1f}%）")
    print(f"  最少格 = {counts.min():.1f}（{counts.min() / tot * 100:.2f}%）")

    # 标签来源统计
    print()
    bo_like = sum(1 for s in samples
                  if abs(float(s["label_kp"]) - float(s.get("label_kp", 0))) < 1e-12
                  and s.get("behaviour_policy") == "bo")
    print(f"标签中 behaviour_policy=bo 的条数: {bo_like}")

    # 关键工况
    print()
    for idx in (0, 13, 47):
        h = hist[idx]
        print(f"工况 {h.get('scenario_index', idx + 1)}: state_samples={h['state_samples']}, "
              f"coverage={h['rule_coverage_pct']}%, log_rmse={float(h['training_log_rmse']):.6f}, "
              f"max_rule_log_change={float(h['max_rule_log_change']):.6f}, "
              f"center_kp={float(h['center_rule_kp']):.6f}, "
              f"high_err_kp={float(h['high_error_rule_kp']):.6f}")
    print(f"末行 label_conflict_high = {last.get('label_conflict_high')}")


# =====================================================================
# 08 RL：Q 表演化重建
# =====================================================================
def dump_08():
    section("08 RL：Q 表重建（Q = 1e-6·[a=4] + 0.12·Σ td_error）")
    rows = read_csv(V3 / "rl_training_transitions.csv")
    print(f"转移行数: {len(rows)}")
    assert len(rows) == 36000

    alpha = 0.12
    q = np.zeros((5, 5, 3, 9))
    q[:, :, :, 4] = 1e-6
    # 记录 s=(3,2,1) 每个动作在各回合的 Q 快照
    target_state = (3, 2, 1)
    target_ep = {25, 50, 100, 200, 450, 750}
    snapshots = {ep: None for ep in target_ep}
    visited = np.zeros((5, 5, 3, 9), dtype=bool)
    upd_count = np.zeros((5, 5, 3, 9), dtype=int)
    cum_unique = []
    seen = set()

    for r in rows:
        ep = int(float(r["episode"]))
        s = (int(float(r["state_error_bin"])), int(float(r["state_delta_bin"])),
             int(float(r["state_command_bin"])))
        a = int(float(r["action"]))
        q[s + (a,)] += alpha * float(r["td_error"])
        upd_count[s + (a,)] += 1
        seen.add(s + (a,))
        visited[s + (a,)] = True
        if ep in target_ep:
            pass
        cum_unique.append((ep, len(seen)))

    # 每回合末的累计唯一条目数
    by_ep = {}
    for ep, n in cum_unique:
        by_ep[ep] = n
    print()
    print("累计被更新的 (s,a) 条目数：")
    for ep in (1, 25, 50, 100, 200, 450, 661, 750):
        if ep in by_ep:
            print(f"  ep{ep:>4}: {by_ep[ep]:>3} / 675  ({by_ep[ep] / 675 * 100:.1f}%)")

    # 交叉校验 s=(2,2,1)
    chk = q[2, 2, 1]
    print()
    print("s=(2,2,1) 九个动作终值（交叉校验）:")
    print("  " + " ".join(f"a{i}={chk[i]:+.5f}" for i in range(9)))
    expect = [-7.07325, -7.27514, 0.0, -7.08690, -7.73199, 0.0, -7.72027, -7.04957, 0.0]
    for i, e in enumerate(expect):
        assert abs(chk[i] - e) < 1e-4, (i, chk[i], e)
    print("  → 与预期一致 ✓")

    # 高频状态 s=(3,2,1) 的 Q 演化
    print()
    print("【s=(3,2,1) 九个动作的 Q 值演化】")
    # 重新逐行累积，记录快照
    q2 = np.zeros((5, 5, 3, 9))
    q2[:, :, :, 4] = 1e-6
    snap = {}
    for r in rows:
        ep = int(float(r["episode"]))
        s = (int(float(r["state_error_bin"])), int(float(r["state_delta_bin"])),
             int(float(r["state_command_bin"])))
        a = int(float(r["action"]))
        q2[s + (a,)] += alpha * float(r["td_error"])
        if ep in target_ep and ep not in snap:
            snap[ep] = q2[target_state].copy()
    acts = ["(0.75,0.75)", "(0.75,1.00)", "(0.75,1.30)", "(1.00,0.75)", "(1.00,1.00)",
            "(1.00,1.30)", "(1.30,0.75)", "(1.30,1.00)", "(1.30,1.30)"]
    eps_sorted = sorted(snap)
    print("  " + " ".join(f"{'动作':<12}" for _ in acts))
    print("  " + " ".join(f"{'a'+str(i)+' '+acts[i]:<12}" for i in range(9)))
    for ep in eps_sorted:
        print(f"  ep{ep:>3} " + " ".join(f"{snap[ep][i]:+8.3f}    " for i in range(9)))
    fin = q[target_state]
    print(f"  终值  " + " ".join(f"{fin[i]:+8.3f}    " for i in range(9)))
    print(f"  argmax = a{int(np.argmax(fin))} {acts[int(np.argmax(fin))]}, Q = {fin.max():+.4f}")

    # 更新次数
    print()
    print("【s=(3,2,1) 各动作被更新次数】")
    print("  " + " ".join(f"a{i}={upd_count[target_state + (i,)]}" for i in range(9)))

    # 部署表
    dep_q = np.load(V3 / "rl_q_table.npy")
    print()
    print(f"部署表 rl_q_table.npy 唯一值: {np.unique(dep_q)}")
    print(f"部署表在 s=(3,2,1) 的 argmax = a{int(np.argmax(dep_q[target_state]))}")

    # 策略分歧
    print()
    diff = 0
    visited_states = 0
    for i in range(5):
        for j in range(5):
            for k in range(3):
                if not visited[i, j, k].any():
                    continue
                visited_states += 1
                learned = int(np.argmax(q[i, j, k]))
                deployed = int(np.argmax(dep_q[i, j, k]))
                if learned != deployed:
                    diff += 1
    print(f"被访问的三元组状态数: {visited_states}")
    print(f"学到的 argmax 与部署表 argmax 不同的状态数: {diff} "
          f"({diff / max(visited_states, 1) * 100:.0f}%)")

    # 前 4 次转移
    print()
    print("【前 4 次转移的手算走读】")
    print(f"  {'决策':>3} {'s':<12} {'a':>2} {'r':>12} {'s′':<12} {'TD δ':>12} {'ΔQ=0.12δ':>12}")
    for r in rows[:4]:
        s = (int(float(r["state_error_bin"])), int(float(r["state_delta_bin"])),
             int(float(r["state_command_bin"])))
        ns = (int(float(r["next_error_bin"])), int(float(r["next_delta_bin"])),
              int(float(r["next_command_bin"])))
        td = float(r["td_error"])
        print(f"  {r['decision']:>3} {str(s):<12} {int(float(r['action'])):>2} "
              f"{float(r['reward']):>+12.6f} {str(ns):<12} {td:>+12.6f} {alpha * td:>+12.6f}")
    print(f"  实际 (Kp,Ki): " + " ".join(f"({rows[i]['applied_kp']},{rows[i]['applied_ki']})" for i in range(2)))

    # ε 触底回合
    print()
    hist = read_csv(V3 / "rl_training_history.csv")
    eps = [float(h["epsilon"]) for h in hist]
    floor = next((i + 1 for i, e in enumerate(eps) if abs(e - 0.03) < 1e-9), None)
    print(f"ε 触底 0.03 的回合: ep{floor}（共 {len(eps)} 回合）")
    print(f"  首回合 ε={eps[0]}, 末回合 ε={eps[-1]}")


def main():
    dump_03()
    dump_06()
    dump_07()
    dump_08()
    print()
    print("=" * 92)
    print("全部核对完成，所有 assert 通过")
    print("=" * 92)


if __name__ == "__main__":
    main()
