"""Export figures and numeric tables only. Chapter prose is maintained by hand."""
from collections import defaultdict
from pathlib import Path
import shutil

import numpy as np

from hvac_pid.artifacts import read_json, write_csv, write_json


def cell(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ")


def table(path, headings, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] * len(headings)) + " |"]
    lines += ["| " + " | ".join(map(cell, row)) + " |" for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export(output: Path, destination: Path):
    output, destination = Path(output), Path(destination)
    generated = destination / "generated"
    public = destination / "public/results"
    generated.mkdir(parents=True, exist_ok=True)
    # Only the current run is referenced in the asset manifest. Existing unrelated
    # assets are never read, and chapter content is never modified.
    assets = []
    for file in sorted(output.rglob("*.svg")):
        relative = file.relative_to(output)
        target = public / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
        assets.append(relative.as_posix())
    write_json(public / "manifest.json", {"source": output.name, "assets": assets})
    for prefix in ["01-temperature", "03-classical", "04-bo", "05-fnn", "06-qlearning", "07-dqn"]:
        metrics = read_json(output / prefix / "metrics.json")
        table(generated / f"{prefix}.md", ["方法", "IAE / ℃·min", "最大过冷 / ℃", "输出总变化量"],
              [[name, m["iae"], m["undershoot"], m["movement"]] for name, m in metrics.items()])
    identified = read_json(output / "03-classical/gains.json")
    table(generated / "gains.md", ["方法", "Kp", "Ki / min⁻¹"],
          [[name, g["kp"], g["ki"]] for name, g in identified["gains"].items()])
    bo = read_json(output / "04-bo/result.json")
    table(generated / "bo-trials.md", ["试验", "来源", "Kp", "Ki", "IAE", "预测均值", "预测标准差"],
          [[t["trial"] + 1, t["source"], t["kp"], t["ki"], t["iae"],
            t.get("predicted_mean", "—"), t.get("predicted_std", "—")] for t in bo["trials"]])
    fnn = read_json(output / "05-fnn/training.json")
    table(generated / "fnn-selection.md", ["正则系数", "验证集平均 IAE"],
          [[r["ridge"], r["mean_validation_iae"]] for r in fnn["selection"]])
    q = read_json(output / "06-qlearning/training.json")
    dqn = read_json(output / "07-dqn/training.json")
    table(generated / "rl-training.md", ["方法", "回合", "交互步数", "训练耗时 / s", "验证集平均 IAE"],
          [[name, r["cost"]["episodes"], r["cost"]["transitions"], r["cost"]["seconds"],
            float(np.mean(r["validation_iae"]))] for name, r in [("Q-Learning", q), ("DQN", dqn)]])
    example = q["example_update"]
    table(generated / "td-example.md", ["量", "值"], [[k, v] for k, v in example.items()])
    llm = read_json(output / "08-llm/result.json")
    llm_rows = [["运行状态", llm["status"]], ["模型调用次数", llm["cost"]["calls"]],
                ["模型", llm["cost"].get("model") or "未配置"],
                ["推理设置", llm["cost"].get("reasoning_effort") or "接口默认"],
                ["停止原因", llm["stop_reason"]]]
    usage = llm["cost"].get("usage", [])
    for key, label in [("prompt_tokens", "输入 token 总数"), ("completion_tokens", "输出 token 总数"),
                       ("prompt_cache_hit_tokens", "缓存命中 token 总数")]:
        if usage and all(key in row for row in usage):
            llm_rows.append([label, sum(row[key] for row in usage)])
    table(generated / "llm-status.md", ["项目", "本次记录"], llm_rows)
    table(generated / "llm-trials.md", ["轮次", "状态", "Kp", "Ki", "IAE", "调整说明"],
          [[t.get("round", "初始"), t["status"], t.get("kp", "—"), t.get("ki", "—"),
            t.get("iae", "—"), t.get("reason", t.get("error", t.get("source", "")))]
           for t in llm["trials"]])
    if llm["best"]:
        metrics = read_json(output / "08-llm/metrics.json")
        table(generated / "llm-metrics.md", ["方法", "Kp", "Ki", "IAE", "最大过冷 / ℃", "输出变化"],
              [[name, llm["best"]["kp"], llm["best"]["ki"], values["iae"], values["undershoot"], values["movement"]]
               for name, values in metrics.items()])
    frozen = read_json(output / "09-compare/frozen.json")
    for case in ["nominal", "slower", "long_delay", "larger_load"]:
        table(generated / f"compare-{case}.md", ["方法", "参数方式", "IAE", "过冷 / ℃", "输出变化"],
              [[r["method"], "在线" if r["mode"] == "online" else "固定", r["iae"], r["undershoot"], r["movement"]]
               for r in frozen if r["case"] == case])
    holdout = read_json(output / "09-compare/holdout.json")
    grouped = defaultdict(list)
    for row in holdout:
        grouped[row["method"]].append(row["iae"])
    table(generated / "holdout.md", ["方法", "对象数", "平均 IAE", "最小 IAE", "最大 IAE"],
          [[name, len(values), float(np.mean(values)), min(values), max(values)] for name, values in grouped.items()])
    costs = read_json(output / "09-compare/costs.json")
    cost_rows = []
    for name, cost in costs.items():
        cost_rows.append([name, cost.get("label_simulations", cost.get("episodes", "—")),
                          cost.get("label_plant_steps", cost.get("plant_steps", 0)),
                          cost.get("simulations", "—"), cost.get("seconds", "—")])
    table(generated / "costs.md", ["方法", "教师仿真 / 训练回合", "物理步数", "当前整定仿真次数", "耗时 / s"], cost_rows)
    protocol = read_json(output / "09-compare/protocol.json")
    environment = protocol["environment"]
    table(generated / "environment.md", ["环境", "值"],
          [["种子", protocol["seed"]], ["Python", environment["python"]],
           ["平台", environment["platform"]], *environment["packages"].items()])


def summarize_repeats(root: Path, seeds):
    grouped = defaultdict(list)
    for seed in seeds:
        for row in read_json(root / f"seed-{seed}/09-compare/frozen.json"):
            grouped[(row["case"], row["method"])].append(row["iae"])
    rows = [{"case": case, "method": method, "runs": len(values), "mean_iae": float(np.mean(values)),
             "std_iae": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
             "min_iae": min(values), "max_iae": max(values)} for (case, method), values in grouped.items()]
    write_json(root / "summary.json", {"seeds": seeds, "rows": rows})
    write_csv(root / "summary.csv", rows)
