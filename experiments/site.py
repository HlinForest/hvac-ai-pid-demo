"""Export figures and numeric tables only. Chapter prose is maintained by hand."""
from collections import defaultdict
from pathlib import Path
import shutil
import re

import numpy as np

from hvac_pid.artifacts import read_json, write_csv, write_json
from .methods import ONLINE


PUBLIC_FILES = {
    "config.json", "metrics.json", "identification.json", "step.csv", "gains.json", "lambda.json",
    "result.json", "trials.csv", "labels.json", "splits.json", "training.json", "history.csv",
    "model.npz", "model.pt", "snapshots.npz", "frozen.csv", "frozen.json", "holdout.json",
    "costs.json", "protocol.json", "holdout-example-bo.json", "timings.csv", "summary.json",
    "summary.csv", "observations.json", "holdout-summary.json", "reference.json", "environment.json", "failure.json",
    "results.json", "one-update.json", "update-trace.json", "runs.json",
}
PUBLIC_DIRS = {"01-temperature", "02-plant", "03-classical", "04-bo", "05-fnn", "06-qlearning",
               "07-dqn", "08-llm", "09-compare", "10-ppo", "11-td3", "12-sac", "13-crossq",
               "14-pg4pi", "benchmark", "explain", "repeats", "pg4pi-reference"}


def publishable(relative):
    if relative.parts[0] not in PUBLIC_DIRS:
        return False
    if relative.name in PUBLIC_FILES or relative.suffix == ".svg":
        return True
    if relative.parent.as_posix() == "explain" and relative.suffix == ".json":
        return relative.stem in {"classical", "bo", "fnn", *ONLINE, "pg4pi", "llm"}
    return relative.suffix == ".csv" and relative.stem in {
        "Manual", "Aggressive", "ZN", "SIMC", "BO", "FNN", "Q-Learning", "DQN", "LLM",
        "PPO", "TD3", "SAC", "CrossQ", "PG4PI-HVAC"}


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
    # The published directory must match the manifest, including on local rebuilds.
    assets = []
    for file in sorted(output.rglob("*")):
        if not file.is_file():
            continue
        relative = file.relative_to(output)
        if not publishable(relative):
            continue
        if file.suffix in {".json", ".csv", ".svg"}:
            content = file.read_text(encoding="utf-8")
            if re.search(r"\bsk-[A-Za-z0-9_-]{20,}\b", content):
                raise ValueError(f"Credential-like value in public artifact: {relative}")
        target = public / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
        assets.append(relative.as_posix())
    public_root = public.resolve()
    for file in public.rglob("*"):
        if file.is_file() and file.relative_to(public).as_posix() not in assets:
            if not file.resolve().is_relative_to(public_root):
                raise ValueError(f"Public asset escapes destination: {file}")
            file.unlink()
    write_json(public / "manifest.json", {"source": output.name, "assets": assets})
    for prefix in ["01-temperature", "03-classical", "04-bo", "05-fnn",
                   *(entry[1] for entry in ONLINE.values()), "14-pg4pi"]:
        if not (output / prefix / "metrics.json").exists():
            continue
        metrics = read_json(output / prefix / "metrics.json")
        table(generated / f"{prefix}.md", ["方法", "IAE / ℃·min", "最大过冷 / ℃", "输出总变化量"],
              [[name, m["iae"], m["undershoot"], m["movement"]] for name, m in metrics.items()])
    identified = read_json(output / "03-classical/gains.json")
    table(generated / "gains.md", ["方法", "Kp / ℃⁻¹", "Ki / (℃·min)⁻¹"],
          [[name, g["kp"], g["ki"]] for name, g in identified["gains"].items()])
    bo = read_json(output / "04-bo/result.json")
    table(generated / "bo-trials.md", ["试验", "来源", "Kp", "Ki", "IAE", "预测均值", "预测标准差"],
          [[t["trial"] + 1, t["source"], t["kp"], t["ki"], t["iae"],
            t.get("predicted_mean", "—"), t.get("predicted_std", "—")] for t in bo["trials"]])
    fnn = read_json(output / "05-fnn/training.json")
    table(generated / "fnn-selection.md", ["正则系数", "验证集平均 IAE"],
          [[r["ridge"], r["mean_validation_iae"]] for r in fnn["selection"]])
    q = read_json(output / "06-qlearning/training.json")
    training = [(name, read_json(output / chapter / "training.json"))
                for name, chapter, *_ in ONLINE.values() if (output / chapter / "training.json").exists()]
    for name, report in training:
        method = next(key for key, entry in ONLINE.items() if entry[0] == name)
        update = report.get("first_update")
        if update:
            table(generated / f"first-update-{method}.md", ["第一次更新的量", "实际记录"],
                  [[key, value] for key, value in update.items()
                   if isinstance(value, (str, int, float, bool)) or value is None
                   or isinstance(value, list) and len(value) <= 5])
    table(generated / "rl-training.md", ["方法", "回合", "交互步数", "训练耗时 / s", "验证集平均 IAE"],
          [[name, r["cost"]["episodes"], r["cost"]["transitions"], r["cost"]["seconds"],
            float(np.mean(r["validation_iae"]))] for name, r in training])
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
        offline = "—" if name == "PG4PI-HVAC" else cost.get("label_simulations", cost.get("episodes", "—"))
        cost_rows.append([name, offline,
                          cost.get("label_plant_steps", cost.get("plant_steps", 0)),
                          cost.get("simulations", "—"), cost.get("seconds", "—")])
    table(generated / "costs.md", ["方法", "预训练教师仿真 / 回合", "累计仿真物理步数", "当前整定仿真次数", "耗时 / s"], cost_rows)
    protocol = read_json(output / "09-compare/protocol.json")
    environment = protocol["environment"]
    table(generated / "environment.md", ["环境", "值"],
          [["种子", protocol["seed"]], ["Python", environment["python"]],
           ["平台", environment["platform"]], *environment["packages"].items()])
    if (output / "benchmark/result.json").exists():
        measured = read_json(output / "benchmark/result.json")
        table(generated / "benchmark.md", ["方法 / 组件", "参数量", "文件 / bytes", "中位数 / μs", "P95 / μs"],
              [[r["method"], r["learned_parameters"], r["artifact_bytes"], r["median_us"], r["p95_us"]]
               for r in measured["rows"]])
    if (output / "repeats/summary.json").exists():
        repeated = read_json(output / "repeats/summary.json")
        table(generated / "repeats.md", ["工况", "方法", "种子数", "平均 IAE", "标准差", "最小", "最大"],
              [[r["case"], r["method"], r["runs"], r["mean_iae"], r["std_iae"], r["min_iae"], r["max_iae"]]
               for r in repeated["rows"]])
    reference_path = output / "pg4pi-reference/config.json"
    if reference_path.exists():
        reference = read_json(reference_path)
        table(generated / "pg4pi-reference.md", ["核对项目", "记录"],
              [[key, reference.get(key, "—")] for key in ["repository", "commit", "entrypoint", "seed", "status"]])
    table(generated / "downloads.md", ["实验产物", "下载"],
          [[asset, f"[下载](/results/{asset})"] for asset in assets if not asset.endswith(".svg")])


def summarize_repeats(root: Path, seeds):
    grouped = defaultdict(list)
    observations, holdout, runs = [], [], []
    for seed in seeds:
        directory = root / f"seed-{seed}"
        if (directory / "09-compare/protocol.json").exists():
            runs.append({"seed": seed, "protocol": read_json(directory / "09-compare/protocol.json"),
                         "costs": {k: v for k, v in read_json(directory / "09-compare/costs.json").items() if k != "LLM"},
                         "training_configs": {name: read_json(directory / chapter / "training.json").get("config", {})
                                              for name, chapter, *_ in ONLINE.values()
                                              if (directory / chapter / "training.json").exists()}})
        for row in read_json(root / f"seed-{seed}/09-compare/frozen.json"):
            if row["method"] == "LLM":
                continue
            grouped[(row["case"], row["method"])].append(row["iae"])
            observations.append({"seed": seed, **row})
        file = root / f"seed-{seed}/09-compare/holdout.json"
        if file.exists():
            holdout.extend({"seed": seed, **row} for row in read_json(file))
    rows = [{"case": case, "method": method, "runs": len(values), "mean_iae": float(np.mean(values)),
             "std_iae": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
             "min_iae": min(values), "max_iae": max(values)} for (case, method), values in grouped.items()]
    write_json(root / "summary.json", {"seeds": seeds, "rows": rows})
    write_csv(root / "summary.csv", rows)
    write_json(root / "observations.json", observations)
    write_json(root / "runs.json", runs)
    write_json(root / "holdout.json", holdout)
    by_method = defaultdict(list)
    for seed in seeds:
        for name in dict.fromkeys(r["method"] for r in holdout):
            values = [r["iae"] for r in holdout if r["seed"] == seed and r["method"] == name]
            if values:
                by_method[name].append(float(np.mean(values)))
    write_json(root / "holdout-summary.json", [
        {"method": name, "seed_means": values, "mean_iae": float(np.mean(values)),
         "std_between_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
        for name, values in by_method.items()])
    from hvac_pid.plotting import distributions
    if observations:
        distributions([r for r in observations if r["case"] == "nominal"], root / "nominal.svg",
                      title="Five-seed nominal evaluation (each dot is a seed)")
    failures = [r for r in holdout if r["method"] == "PG4PI-HVAC"]
    if failures:
        # Inspect the worst recorded outcome, without feeding it back into training.
        from hvac_pid.core import Gains, Scenario, evaluate
        from hvac_pid.tuning import identify, simc
        from hvac_pid.plotting import responses
        from .run import save_evaluations
        worst = max(failures, key=lambda row: row["iae"])
        record = read_json(root / f"seed-{worst['seed']}/09-compare/pg4pi-holdout/object-{worst['object']}.json")
        destination = root / "pg4pi-failure"
        write_json(destination / "result.json", record)
        scenario = Scenario(**record["scenario"])
        results = {"PG4PI-HVAC": evaluate(scenario, Gains(**record["best"])),
                   "SIMC": evaluate(scenario, simc(identify(scenario)[0]))}
        save_evaluations(destination, results)
        responses(results, destination / "response.svg")
