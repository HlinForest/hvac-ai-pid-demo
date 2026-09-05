from __future__ import annotations

"""Aggregate the LLM-agent model matrix runs into one CSV + Markdown report.

Scans ``archive/outputs_llm_matrix_v3/<model>_<difficulty>_r<seed>/`` directories,
reads each ``llm_agent_summary.json`` and ``llm_agent_trace.csv``, and writes:

- ``archive/outputs_llm_matrix_v3/matrix_summary.csv`` — one row per run;
- ``archive/outputs_llm_matrix_v3/AGGREGATE_REPORT.md`` — per-model verdicts.

The report never upgrades replay/heuristic runs to "live LLM" evidence:
a ``live_llm`` column is derived from the summary's own disclosure fields.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_run(run_dir: Path) -> dict[str, object] | None:
    summary_path = run_dir / "llm_agent_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    trace_rows: list[dict[str, str]] = []
    trace_path = run_dir / "llm_agent_trace.csv"
    if trace_path.exists():
        with trace_path.open("r", encoding="utf-8-sig", newline="") as handle:
            trace_rows = list(csv.DictReader(handle))
    tools = [row.get("tool", "") for row in trace_rows]
    parts = run_dir.name.rsplit("_r", 1)
    model, _, difficulty = parts[0].partition("_")
    live = bool(summary.get("source", "")) and not summary.get("llm_replay_disclosure", 0)
    baseline_kp = float(summary.get("llm_baseline_kp", "nan"))
    deployed_kp = float(summary.get("llm_deployed_kp", "nan"))
    baseline_ki = float(summary.get("llm_baseline_ki", "nan"))
    deployed_ki = float(summary.get("llm_deployed_ki", "nan"))
    return {
        "run": run_dir.name,
        "model": model,
        "difficulty": difficulty,
        "seed": parts[1] if len(parts) > 1 else "",
        "live_llm": int(live),
        "provider": summary.get("provider", ""),
        "agent_steps": summary.get("llm_agent_steps", 0),
        "agent_trials": summary.get("llm_agent_trials", 0),
        "accepted_trials": summary.get("llm_accepted_trials", 0),
        "inspect_calls": tools.count("inspect_history"),
        "deployment_accepted": summary.get("deployment_accepted", 0),
        "fallback_used": summary.get("fallback_used", 0),
        "objective": summary.get("objective", ""),
        "max_undershoot_c": summary.get("max_undershoot_c", ""),
        "stable": summary.get("stable", 0),
        "first_in_band_minute": summary.get("first_in_band_minute", ""),
        "door_recovery_minutes": summary.get("door_recovery_minutes", ""),
        "baseline_kp": baseline_kp,
        "baseline_ki": baseline_ki,
        "deployed_kp": deployed_kp,
        "deployed_ki": deployed_ki,
        "kp_change_pct": (deployed_kp / baseline_kp - 1.0) * 100.0 if baseline_kp else "",
        "ki_change_pct": (deployed_ki / baseline_ki - 1.0) * 100.0 if baseline_ki else "",
        "described_trial": summary.get("llm_described_trial", ""),
    }


def _fmt(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    matrix_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "archive/outputs_llm_matrix_v3"
    rows = [row for d in sorted(matrix_dir.iterdir()) if d.is_dir() for row in [_load_run(d)] if row]
    if not rows:
        raise SystemExit(f"no runs found under {matrix_dir}")

    csv_path = matrix_dir / "matrix_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines: list[str] = []
    lines.append("# LLM Agent 模型矩阵汇总报告")
    lines.append("")
    lines.append(f"- 运行目录：`{matrix_dir.name}/`，共 **{len(rows)}** 次运行")
    lines.append(f"- 真实模型调用：{sum(r['live_llm'] for r in rows)} 次；replay/启发式：{sum(1 - r['live_llm'] for r in rows)} 次")
    lines.append(f"- 全部运行稳定（stable=1）：{sum(int(r['stable']) for r in rows)}/{len(rows)}")
    lines.append(f"- Agent 候选被部署（通过全部安全门）：{sum(int(r['deployment_accepted']) for r in rows)}/{len(rows)}；其余回退 IMC 基线")
    lines.append("")
    lines.append("## 按模型 × 难度汇总")
    lines.append("")
    lines.append("| 模型 | 难度 | 运行数 | 平均试验数 | 候选接受率 | 部署率 | 平均目标 J | 最大过冷均值 °C |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["model"]), str(row["difficulty"])), []).append(row)
    for (model, difficulty), group in sorted(groups.items()):
        n = len(group)
        trials = sum(int(g["agent_trials"]) for g in group) / n
        total_trials = sum(int(g["agent_trials"]) for g in group)
        acc = sum(int(g["accepted_trials"]) for g in group)
        acc_rate = acc / total_trials if total_trials else 0.0
        dep = sum(int(g["deployment_accepted"]) for g in group) / n
        obj = sum(float(g["objective"]) for g in group) / n
        und = sum(float(g["max_undershoot_c"]) for g in group) / n
        lines.append(f"| {model} | {difficulty} | {n} | {trials:.1f} | {acc_rate:.0%} | {dep:.0%} | {obj:.2f} | {und:.2f} |")
    lines.append("")
    lines.append("## 逐次运行明细")
    lines.append("")
    lines.append("| 运行 | 步数 | 试验 | 接受 | 部署 | 目标 J | 过冷 °C | ΔKp | ΔKi | 部署判定 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        verdict = "部署候选" if int(row["deployment_accepted"]) else "回退 IMC"
        lines.append(
            f"| {row['run']} | {row['agent_steps']} | {row['agent_trials']} | {row['accepted_trials']} "
            f"| {row['deployment_accepted']} | {_fmt(row['objective'])} | {_fmt(row['max_undershoot_c'])} "
            f"| {_fmt(row['kp_change_pct'], 1)}% | {_fmt(row['ki_change_pct'], 1)}% | {verdict} |"
        )
    lines.append("")
    lines.append("## 读法与边界")
    lines.append("")
    lines.append("- `部署=1` 表示 Agent 提出的候选通过了硬边界、±10% 信任域、多场景重复仿真、安全门与部署复核并优于基线；`部署=0` 表示宿主回退到调优 IMC，不是仿真发散。")
    lines.append("- 不同难度（std/hard）的目标 J 不可跨行直接比较；hard 场景扰动更强，基线 J 本身更高。")
    lines.append("- 本矩阵回答的是「LLM 能否在安全门内找到优于调优 IMC 的增益」，而不是「LLM 能否直接控制空调」——后者在本架构中被刻意禁止。")
    report_path = matrix_dir / "AGGREGATE_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "report": str(report_path), "runs": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
