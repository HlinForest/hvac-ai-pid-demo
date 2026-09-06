"""Seven-algorithm sealed evaluation on one 80-scenario list (E2).

E2 = frozen-policy reevaluation: policies frozen elsewhere are re-evaluated
on the *same* 5x16 sealed scenarios with the *same* noise seeds and the
*same* six-substep v4 integrator.  It is NOT a unified-training comparison.
Policy origins differ (v3 frozen / legacy conservative / replay-frozen);
see ``sealed_provenance.json:policy_provenance`` + ``safe_bo_attribution``.

Primary statistic: mean of paired per-scenario ratios
``mean(objective/imc)`` with bootstrap 95% CI (``paired_ratio_*`` ==
``mean_ratio_to_imc`` point estimate).  Secondary: ``ratio_of_means_to_imc``
(``mean(obj)/mean(imc)``), reported for reference only.

Outputs (inside ``--output``):
  * ``sealed_80_details.csv`` — 560 rows (80 scenarios x 7 algorithms) with
    per-scenario metrics, candidate vs deployed objectives for FNN/RL, and
    split acceptance fields (bounded / comfort_held / recovery_ok /
    actuator_compliant / validation_passed / fallback_failed).
  * ``sealed_80_summary.csv`` — 7 rows with mean objective, paired-ratio CI,
    adoption / failure / fallback rates and wall cost placeholder.
  * ``sealed_ablation.csv`` — random-search + fixed-conservative controls on
    the same sealed list, so Safe-BO gains can be attributed to search vs
    conservative initialization.
  * ``sealed_provenance.json`` — code SHA, manifest SHA, policy hashes,
    integrator label and the exact sealed seeds.

LLM here is the deterministic replay agent (``ReplayPIDAgentPolicy`` frozen
once on the training split); it is labelled as replay, never as a live
model call.  Live LLM matrices remain separate evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hvac_pid.advanced_tuning import LLMAgentAutoTuner, ReplayPIDAgentPolicy, RiskSafetyConfig
from hvac_pid.ai_controllers import FNNGainController, IncrementalRLController
from hvac_pid.config import Scenario, sample_adaptive_scenarios
from hvac_pid.controllers import PIController, identify_fopdt, imc_pi, ziegler_nichols_pi
from hvac_pid.manifest import git_head_sha, load_manifest, manifest_sha256
from hvac_pid.metrics import calculate_metrics
from hvac_pid.policy_bundle import PolicyBundle
from hvac_pid.simulator import simulate
import subprocess

ALGORITHMS = ("zn", "imc", "bo", "safe-bo", "fnn", "rl", "llm")
DEFAULT_ACCEPTANCE_SEEDS = (101, 211, 307, 401, 503)
INTEGRATOR_LABEL = "六子步v4积分器(外层60s/内层10s×6, integration_substeps=6)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seven-algorithm 80-scenario sealed evaluation")
    parser.add_argument("--artifact-dir", type=Path, required=True,
                        help="frozen policies, e.g. archive/outputs_review_v3 or artifacts/runs/<run_id>")
    parser.add_argument("--output", type=Path, default=Path("artifacts/runs/sealed-80x7"),
                        help="output directory for sealed CSVs")
    parser.add_argument("--seed", type=int, default=7, help="pipeline seed the policies were trained with")
    parser.add_argument("--per-repeat", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--acceptance-seeds", type=str, default="101,211,307,401,503",
                        help="comma-separated sealed repeat ids")
    return parser.parse_args()


def _sealed_scenarios(seed: int, acceptance_seeds: tuple[int, ...], per_repeat: int) -> list[tuple[int, int, Scenario]]:
    out: list[tuple[int, int, Scenario]] = []
    for acceptance_seed in acceptance_seeds:
        scenarios = sample_adaptive_scenarios(per_repeat, seed=seed + 100_003 + acceptance_seed, duration_hours=5.0)
        out.extend((acceptance_seed, index + 1, scenario) for index, scenario in enumerate(scenarios))
    return out


def _paired_ratio_ci(candidate: np.ndarray, baseline: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Primary statistic (E2): mean of paired per-scenario ratios + 95% bootstrap CI.

    Each bootstrap resample draws scenarios with replacement and recomputes
    ``mean(candidate[i] / baseline[i])``.  This treats every sealed scenario
    equally.  It differs from the secondary ``ratio_of_means``
    (``mean(candidate) / mean(baseline)``), which is dominated by
    large-objective scenarios.  Do not conflate the two.
    """
    rng = np.random.default_rng(seed)
    ratios = candidate / np.maximum(baseline, 1e-12)
    indices = rng.integers(0, len(candidate), size=(2000, len(candidate)))
    boot = np.mean(ratios[indices], axis=1)
    return float(np.mean(ratios)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _ratio_of_means(candidate: np.ndarray, baseline: np.ndarray) -> float:
    """Secondary point estimate only: mean(candidate)/mean(baseline)."""
    return float(np.mean(candidate) / max(float(np.mean(baseline)), 1e-12))


def _git_is_dirty() -> tuple[bool, str]:
    """Worktree dirtiness + short status hash for provenance (E2 evidence)."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=str(ROOT), timeout=10)
        porcelain = out.stdout.strip()
        digest = hashlib.sha256(porcelain.encode("utf-8")).hexdigest()[:16] if porcelain else "clean"
        return (bool(porcelain), digest)
    except Exception:
        return (False, "unknown-no-git")


def _frozen_llm_gains(artifact_dir: Path, output_dir: Path, training: list[Scenario], fallback: tuple[float, float], seed: int) -> tuple[tuple[float, float], str]:
    frozen = artifact_dir / "llm_policy.json"
    if frozen.exists():
        data = json.loads(frozen.read_text(encoding="utf-8"))
        return ((float(data["kp"]), float(data["ki"])), f"{frozen}#frozen")
    # Deterministic replay freeze on the training split (never sealed).
    # Written to the sealed output dir, never back into the read-only archive.
    tuned = LLMAgentAutoTuner(ReplayPIDAgentPolicy(), max_steps=6, max_trials=3,
                              config=RiskSafetyConfig()).tune(training, fallback, seed=seed)
    payload = {"kp": float(tuned.kp), "ki": float(tuned.ki), "provider": "replay-frozen",
               "trace_steps": len(tuned.trace), "accepted": bool(tuned.accepted)}
    (output_dir / "llm_policy.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ((float(tuned.kp), float(tuned.ki)), "LLMAgentAutoTuner(Replay, training-split freeze)")


def main() -> None:
    args = parse_args()
    artifact_dir: Path = args.artifact_dir
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    acceptance_seeds = tuple(int(v.strip()) for v in args.acceptance_seeds.split(",") if v.strip())
    assert len(acceptance_seeds) == args.repeats, "repeats must match --acceptance-seeds length"
    assert args.per_repeat * args.repeats == 80, "sealed protocol is 5x16=80"

    manifest = load_manifest()
    bundle = PolicyBundle.load(artifact_dir, project_root=ROOT)
    sealed = _sealed_scenarios(args.seed, acceptance_seeds, args.per_repeat)
    assert len(sealed) == 80, len(sealed)

    # Frozen policies (all from the same bundle, except Z-N which is formula).
    commissioning = Scenario()
    zn_gains = ziegler_nichols_pi(identify_fopdt(commissioning))
    imc_gains = tuple(bundle.imc_gains)
    bo_gains = tuple(bundle.bo_gains)
    safe_gains = tuple(bundle.safe_bo_gains)
    # Training split for the one-time LLM replay freeze (never sealed).
    training = sample_adaptive_scenarios(8, seed=args.seed, duration_hours=5.0)
    llm_gains, llm_source = _frozen_llm_gains(artifact_dir, output, training, imc_gains, seed=args.seed + 31337)
    # FNN/RL candidate vs deployed (candidate files may equal deployed when accepted).
    fnn_candidate_path = artifact_dir / "fnn_rule_table_candidate.npy"
    fnn_candidate = np.load(fnn_candidate_path) if fnn_candidate_path.exists() else np.asarray(bundle.fnn_rule_table)
    fnn_ctx_candidate_path = artifact_dir / "fnn_context_coefficients_candidate.npy"
    fnn_ctx_candidate = np.load(fnn_ctx_candidate_path) if fnn_ctx_candidate_path.exists() else np.asarray(bundle.fnn_context)
    rl_candidate_path = artifact_dir / "rl_q_table_candidate.npy"
    rl_candidate = np.load(rl_candidate_path) if rl_candidate_path.exists() else np.asarray(bundle.rl_q_table)

    def controllers_for(fnn_table: np.ndarray, fnn_ctx: np.ndarray, rl_table: np.ndarray) -> dict[str, PIController]:
        return {
            "zn": PIController(*zn_gains),
            "imc": PIController(*imc_gains),
            "bo": PIController(*bo_gains),
            "safe-bo": PIController(*safe_gains),
            "fnn": FNNGainController(imc_gains, rule_table=fnn_table, context_coefficients=fnn_ctx),
            "rl": IncrementalRLController(imc_gains, q_table=rl_table, covered_mask=np.asarray(bundle.rl_covered_mask)),
            "llm": PIController(*llm_gains),
        }

    deployed_controllers = controllers_for(np.asarray(bundle.fnn_rule_table), np.asarray(bundle.fnn_context), np.asarray(bundle.rl_q_table))
    candidate_controllers = controllers_for(fnn_candidate, fnn_ctx_candidate, rl_candidate)

    details: list[dict[str, object]] = []
    started = time.perf_counter()
    for ordinal, (repeat_seed, scenario_index, scenario) in enumerate(sealed):
        noise_seed = args.seed + 700_000 + ordinal
        baseline_obj: float | None = None
        per_algo: dict[str, dict[str, float]] = {}
        per_candidate: dict[str, float] = {}
        for algo in ALGORITHMS:
            deployed = deployed_controllers[algo]
            result = simulate(scenario, deployed, seed=noise_seed)
            metrics = calculate_metrics(result)
            per_algo[algo] = metrics
            if algo == "imc":
                baseline_obj = float(metrics["objective"])
            if algo in ("fnn", "rl"):
                cand_result = simulate(scenario, candidate_controllers[algo], seed=noise_seed)
                per_candidate[algo] = float(calculate_metrics(cand_result)["objective"])
        assert baseline_obj is not None
        for algo in ALGORITHMS:
            metrics = per_algo[algo]
            details.append({
                "repeat_seed": repeat_seed,
                "scenario_index": scenario_index,
                "ordinal": ordinal + 1,
                "algorithm": algo,
                "noise_seed": noise_seed,
                "kp_deployed": float(deployed_controllers[algo].kp),
                "ki_deployed": float(deployed_controllers[algo].ki),
                "objective": float(metrics["objective"]),
                "candidate_objective": float(per_candidate.get(algo, metrics["objective"])),
                "baseline_imc_objective": float(baseline_obj),
                "objective_ratio_to_imc": float(metrics["objective"] / max(baseline_obj, 1e-12)),
                "rmse_c": float(metrics["rmse_c"]),
                "max_undershoot_c": float(metrics["max_undershoot_c"]),
                "bounded": float(metrics["bounded"]),
                "comfort_held": float(metrics.get("comfort_held", metrics["stable"])),
                "recovery_ok": float(metrics.get("disturbance_recovered", 0.0)),
                "actuator_compliant": float(metrics.get("actuator_compliant", 1.0)),
                "validation_passed": float(metrics.get("validation_passed", metrics["stable"])),
                "stable": float(metrics["stable"]),
                "fallback_fraction": float(metrics.get("fallback_fraction", 0.0)),
                # 回退失败 = 发生过回退但温控验收仍不通过（《报告改进》§二.2）。
                # 旧口径 (fallback>10% 且数值不稳定) 过窄，无法表示"回退后仍不达标"。
                "fallback_failed": int(float(metrics.get("fallback_fraction", 0.0)) > 1e-12 and float(metrics.get("validation_passed", metrics["stable"])) < 0.5),
                "integrator": INTEGRATOR_LABEL,
            })
    # Summary with paired CIs vs IMC (same sealed list, same seeds).
    imc_objectives = np.asarray([r["objective"] for r in details if r["algorithm"] == "imc"], dtype=float)
    summary: list[dict[str, object]] = []
    for algo in ALGORITHMS:
        objectives = np.asarray([r["objective"] for r in details if r["algorithm"] == algo], dtype=float)
        paired_mean, lo95, hi95 = _paired_ratio_ci(objectives, imc_objectives, seed=args.seed + 77)
        ratios = objectives / np.maximum(imc_objectives, 1e-12)
        subset = [r for r in details if r["algorithm"] == algo]
        summary.append({
            "algorithm": algo,
            "scenarios": len(subset),
            "mean_objective": float(np.mean(objectives)),
            "std_objective": float(np.std(objectives, ddof=1)) if len(objectives) > 1 else 0.0,
            # 主统计量：逐场景配对比均值 mean(objective/imc) 及其 bootstrap CI。
            "mean_ratio_to_imc": float(np.mean(ratios)),
            "paired_ratio_mean": paired_mean,
            "paired_ratio_lo95": lo95,
            "paired_ratio_hi95": hi95,
            # 次统计量：平均目标之比 mean(obj)/mean(imc)，仅作对照，不得与主统计量混用。
            "ratio_of_means_to_imc": _ratio_of_means(objectives, imc_objectives),
            "noninferior_to_imc": int(hi95 <= 1.02),
            "better_than_imc": int(hi95 < 1.0),
            "stable_rate": float(np.mean([r["stable"] for r in subset])),
            "validation_rate": float(np.mean([r["validation_passed"] for r in subset])),
            "fallback_rate": float(np.mean([1.0 if float(r["fallback_fraction"]) > 0.0 else 0.0 for r in subset])),
            "mean_fallback_fraction": float(np.mean([r["fallback_fraction"] for r in subset])),
            "deployment_note": "replay-frozen, not live" if algo == "llm" else "",
        })
    # Ablation on the same sealed list: random fixed gains + conservative IMC formula.
    rng = np.random.default_rng(args.seed + 999)
    log_kp = rng.uniform(np.log(0.002), np.log(1.5), size=4)
    log_ki = rng.uniform(np.log(1e-5), np.log(0.08), size=4)
    ablation_gains = [(float(np.exp(a)), float(np.exp(b))) for a, b in zip(log_kp, log_ki)]
    conservative = imc_pi(identify_fopdt(commissioning))  # formula default, no lambda search
    ablation_rows: list[dict[str, object]] = []
    for idx, gains in enumerate(ablation_gains + [conservative]):
        name = f"random-{idx+1}" if idx < 4 else "conservative-imc-formula"
        objectives = []
        for ordinal, (_, _, scenario) in enumerate(sealed):
            metrics = calculate_metrics(simulate(scenario, PIController(*gains), seed=args.seed + 700_000 + ordinal))
            objectives.append(float(metrics["objective"]))
        objectives_arr = np.asarray(objectives)
        paired_mean, lo95, hi95 = _paired_ratio_ci(objectives_arr, imc_objectives, seed=args.seed + 101 + idx)
        ablation_rows.append({
            "method": name, "kp": gains[0], "ki": gains[1],
            "mean_objective": float(np.mean(objectives_arr)),
            "paired_ratio_mean": paired_mean, "paired_ratio_lo95": lo95, "paired_ratio_hi95": hi95,
            "ratio_of_means_to_imc": _ratio_of_means(objectives_arr, imc_objectives),
            "note": "same 80 sealed scenarios/seeds; isolates search vs conservative init" if "random" in name else "IMC formula default without lambda search",
        })

    def _write(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    _write(output / "sealed_80_details.csv", details)
    _write(output / "sealed_80_summary.csv", summary)
    _write(output / "sealed_ablation.csv", ablation_rows)
    # E2 evidence: full sealed scenario list (params, not just seeds).
    scen_rows: list[dict[str, object]] = []
    for ordinal, (repeat_seed, scenario_index, scenario) in enumerate(sealed):
        scen_rows.append({
            "ordinal": ordinal + 1, "repeat_seed": repeat_seed, "scenario_index": scenario_index,
            "noise_seed": args.seed + 700_000 + ordinal,
            "duration_hours": float(scenario.duration_hours), "dt_minutes": float(scenario.dt_minutes),
            "initial_zone_c": float(scenario.initial_zone_c), "setpoint_c": float(scenario.setpoint_c),
            "outdoor_c": float(scenario.outdoor_c), "internal_load_w": float(scenario.internal_load_w),
            "door_open_hour": float(scenario.door_open_hour) if scenario.door_open_hour is not None else -1.0,
            "setpoint_change_hour": float(scenario.setpoint_change_hour) if scenario.setpoint_change_hour is not None else -1.0,
            "setpoint_after_c": float(scenario.setpoint_after_c) if scenario.setpoint_after_c is not None else -1.0,
            "integration_substeps": int(getattr(scenario, "integration_substeps", 6)),
        })
    _write(output / "sealed_scenarios.csv", scen_rows)
    # Freeze the executing manifest copy alongside results.
    try:
        manifest_src = Path(str(manifest.get("_manifest_path", ROOT / "experiments/manifests/v4.yaml")))
        if manifest_src.exists():
            (output / "manifest.yaml").write_bytes(manifest_src.read_bytes())
    except Exception:
        pass
    # P1 attribution: is the Safe-BO gain from search or from the init seed?
    safe_bo_attribution = "unknown (no safe_bo_history.csv in artifact_dir)"
    safe_hist = artifact_dir / "safe_bo_history.csv"
    if safe_hist.exists():
        try:
            with safe_hist.open("r", encoding="utf-8-sig", newline="") as handle:
                hist = list(csv.DictReader(handle))
            selected = [r for r in hist if int(float(r.get("selected", 0))) == 1]
            if selected:
                phase = str(selected[-1].get("phase", ""))
                safe_bo_attribution = (
                    f"selected={selected[-1].get('evaluation')}/{phase} "
                    f"kp={selected[-1].get('candidate_kp')} ki={selected[-1].get('candidate_ki')}; "
                    "if phase is 'local qualification seed', the gain is the conservative init, "
                    "not a GP search improvement"
                )
        except Exception as exc:
            safe_bo_attribution = f"unreadable safe_bo_history.csv: {exc}"
    elif (ROOT / "archive/outputs_advanced_quick/safe_bo_history.csv").exists():
        safe_bo_attribution = (
            "legacy archive/outputs_advanced_quick/safe_bo_history.csv: "
            "selected evaluation=2/local qualification seed (0.3838/0.00444); "
            "current evidence supports THIS conservative parameter set, not the BO search mechanism itself"
        )
    provenance = {
        "experiment_id": "E2",
        "experiment_name": "frozen-policy reevaluation on unified v4 sealed list (NOT unified training)",
        "code_sha": git_head_sha(),
        "code_dirty": _git_is_dirty()[0],
        "worktree_status_sha16": _git_is_dirty()[1],
        "manifest_sha256": manifest.get("_manifest_sha256", manifest_sha256()),
        "manifest_path": str(manifest.get("_manifest_path", "experiments/manifests/v4.yaml")),
        "artifact_dir": str(artifact_dir.resolve()),
        "policy_provenance": bundle.provenance,
        "policies": {"zn": list(zn_gains), "imc": list(imc_gains), "bo": list(bo_gains),
                     "safe-bo": list(safe_gains), "llm": list(llm_gains), "llm_source": llm_source},
        "safe_bo_attribution": safe_bo_attribution,
        "stat_primary": "paired_ratio_* = mean of per-scenario ratios mean(obj/imc) + bootstrap 95% CI (2000 resamples, seed+77); PRIMARY for E2 conclusions",
        "stat_secondary": "ratio_of_means_to_imc = mean(obj)/mean(imc); reference only, dominated by large-objective scenarios; do not conflate with primary",
        "stat_note": "mean_ratio_to_imc equals paired_ratio_mean point estimate (both are mean of ratios); ratio_of_means_to_imc is the distinct secondary statistic",
        "sealed": {"repeats": args.repeats, "per_repeat": args.per_repeat, "total": len(sealed),
                   "acceptance_seeds": list(acceptance_seeds), "pipeline_seed": args.seed},
        "integrator": INTEGRATOR_LABEL,
        "elapsed_seconds": time.perf_counter() - started,
        "comparability": "E2: all 7 on same 80 scenarios/seeds/metrics/integrator; frozen policies from mixed origins (v3 frozen + legacy conservative + replay-frozen); LLM replay is NOT live; do not claim unified training or cross-protocol ranking",
    }
    (output / "sealed_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"sealed 80x7: {len(details)} detail rows, {len(summary)} summary rows -> {output.resolve()}")
    for row in summary:
        print(f"  {row['algorithm']:7s} mean={float(row['mean_objective']):.3f} "
              f"paired_ratio={float(row['paired_ratio_mean']):.4f} [{float(row['paired_ratio_lo95']):.4f},{float(row['paired_ratio_hi95']):.4f}] "
              f"stable={float(row['stable_rate']):.2f} validation={float(row['validation_rate']):.2f}")


if __name__ == "__main__":
    main()
