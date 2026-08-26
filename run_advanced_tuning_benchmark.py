from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np

from hvac_pid.advanced_tuning import (
    LLMSupervisoryTuner,
    OllamaPIDProposer,
    OpenAIResponsesPIDProposer,
    PhysicsInformedHeuristicProposer,
    RiskAwareSafeBOTuner,
    RiskSafetyConfig,
    evaluate_candidate,
)
from hvac_pid.config import sample_adaptive_scenarios
from hvac_pid.controllers import identify_fopdt, imc_pi
from hvac_pid.tuning import tune_global_fixed, tune_global_imc_lambda


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _provider(args: argparse.Namespace):
    if args.llm_provider == "heuristic":
        return PhysicsInformedHeuristicProposer()
    if args.llm_provider == "openai":
        return OpenAIResponsesPIDProposer(args.llm_model, base_url=args.llm_base_url or "https://api.openai.com/v1")
    if args.llm_provider == "ollama":
        return OllamaPIDProposer(args.llm_model, base_url=args.llm_base_url or "http://localhost:11434")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Risk-aware safe BO and simulator-gated LLM PI tuning benchmark")
    parser.add_argument("--output", type=Path, default=Path("outputs_advanced_tuning"))
    parser.add_argument("--train-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=8)
    parser.add_argument("--safe-bo-iterations", type=int, default=8)
    parser.add_argument("--llm-rounds", type=int, default=6)
    parser.add_argument("--llm-provider", choices=("none", "heuristic", "openai", "ollama"), default="heuristic")
    parser.add_argument("--llm-model", default="", help="required for openai/ollama; never silently chooses a model")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.llm_provider in {"openai", "ollama"} and not args.llm_model:
        parser.error("--llm-model is required for a real LLM provider")
    if args.quick:
        args.train_samples, args.test_samples = 7, 4
        args.safe_bo_iterations, args.llm_rounds = 3, 3
    return args


def main() -> None:
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = RiskSafetyConfig(repeats=2, max_undershoot_c=4.0)
    training = sample_adaptive_scenarios(args.train_samples, seed=args.seed, duration_hours=5.0)
    holdout = sample_adaptive_scenarios(args.test_samples, seed=args.seed + 50_000, duration_hours=5.0)
    model = identify_fopdt(training[0])

    timings: list[dict[str, object]] = []
    started = time.perf_counter()
    imc_tune = tune_global_imc_lambda(training, model, seed=args.seed, candidates=13)
    imc = (imc_tune.kp, imc_tune.ki)
    timings.append({"method": "IMC lambda tuning", "wall_seconds": time.perf_counter() - started, "evaluations": imc_tune.evaluations})

    started = time.perf_counter()
    ordinary_bo = tune_global_fixed(training, iterations=args.safe_bo_iterations, seed=args.seed + 1)
    timings.append({"method": "ordinary Bayesian optimization", "wall_seconds": time.perf_counter() - started, "evaluations": ordinary_bo.evaluations})

    started = time.perf_counter()
    safe_bo = RiskAwareSafeBOTuner(iterations=args.safe_bo_iterations, candidates=512, config=config).tune(training, imc, seed=args.seed + 2)
    timings.append({"method": "RaGoOSE-style risk-aware safe BO", "wall_seconds": time.perf_counter() - started, "evaluations": safe_bo.evaluations})

    methods: dict[str, tuple[float, float]] = {
        "IMC baseline": imc,
        "ordinary BO": (ordinary_bo.kp, ordinary_bo.ki),
        "risk-aware safe BO": (safe_bo.kp, safe_bo.ki),
    }
    provider = _provider(args)
    llm_result = None
    if provider is not None:
        started = time.perf_counter()
        llm_result = LLMSupervisoryTuner(provider, rounds=args.llm_rounds, config=config).tune(training, imc, seed=args.seed + 3)
        timings.append({"method": f"LLM supervisor: {provider.name}", "wall_seconds": time.perf_counter() - started, "evaluations": llm_result.evaluations})
        label = "heuristic dry-run (NOT LLM)" if args.llm_provider == "heuristic" else f"LLM-supervised ({provider.name})"
        methods[label] = (llm_result.kp, llm_result.ki)

    holdout_rows: list[dict[str, object]] = []
    for method_index, (name, gains) in enumerate(methods.items()):
        evaluation = evaluate_candidate(holdout, gains, seed=args.seed + 90_000 + method_index * 1000, config=config)
        holdout_rows.append(
            {
                "method": name,
                "kp": gains[0], "ki": gains[1],
                "mean_objective": evaluation.mean_objective,
                "objective_std": evaluation.objective_std,
                "risk_objective": evaluation.risk_objective,
                "worst_objective": evaluation.worst_objective,
                "safety_margin": evaluation.safety_margin,
                "safe": int(evaluation.safe),
                **{f"mean_{key}": value for key, value in evaluation.metrics_mean.items()},
            }
        )

    _write_rows(output / "safe_bo_history.csv", list(safe_bo.history))
    _write_rows(output / "ordinary_bo_history.csv", list(ordinary_bo.history))
    _write_rows(output / "imc_tuning_history.csv", list(imc_tune.history))
    _write_rows(output / "llm_tuning_history.csv", list(llm_result.history) if llm_result else [])
    _write_rows(output / "advanced_holdout_summary.csv", holdout_rows)
    _write_rows(output / "tuning_wall_time.csv", timings)

    best_safe = min((row for row in holdout_rows if row["safe"]), key=lambda row: float(row["risk_objective"]), default=None)
    provider_note = (
        "未运行大模型。heuristic 只验证完整安全流水线，不能作为 LLM 效果证据。"
        if args.llm_provider == "heuristic"
        else ("未配置 LLM。" if provider is None else f"已调用 {provider.name}；每个建议均经过边界、25% 信赖域和重复仿真安全门。")
    )
    report = f"""# 高级 AI-PI 调参可复现实验

## 结论

- 工程 SOTA 候选：RaGoOSE 风格的风险感知安全贝叶斯优化；这里是面向本项目的复现/改编，不声称与论文源码逐行相同。
- LLM 架构：低频离线诊断和候选增益生成；仿真器、安全约束和回退 PI 拥有最终决定权。
- {provider_note}
- 本次留出集最优安全方法：{best_safe['method'] if best_safe else '无方法满足当前安全阈值'}。

## 实验规模

- 训练场景：{len(training)}；独立留出场景：{len(holdout)}；每个候选每场景重复噪声仿真：{config.repeats} 次。
- 安全 BO 搜索迭代：{args.safe_bo_iterations}；LLM/代理建议轮数：{args.llm_rounds}。
- 风险目标：样本均值 + {config.risk_weight} × 样本标准差。
- 安全条件：稳定、执行器运行斜率无违规、无最低运行频率违规、最大冷偏差不超过 {config.max_undershoot_c} °C，且调参风险分不超过 IMC 基线的 {config.max_risk_ratio_to_baseline} 倍。

## 文件

- `advanced_holdout_summary.csv`：独立留出集比较，不能用训练分数代替。
- `safe_bo_history.csv`：每次候选的均值、标准差、风险值、预测/实测安全裕量。
- `llm_tuning_history.csv`：原始建议、限幅后建议、诊断、接受/拒绝原因。
- `tuning_wall_time.csv`：在本机本次规模下的实际墙钟耗时。

详细原理、论文边界、公式和运行方式见项目根目录 `SOTA_LLM_PI_TUNING.md`。
"""
    (output / "advanced_tuning_report.md").write_text(report, encoding="utf-8")
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
