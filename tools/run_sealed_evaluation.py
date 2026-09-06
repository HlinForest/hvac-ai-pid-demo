"""Seven-algorithm sealed evaluation on one 80-scenario list (P1).

All seven policies come from a single :class:`PolicyBundle` (``--artifact-dir``)
and are evaluated on the *same* 5x16 sealed scenarios with the *same* noise
seeds and the *same* six-substep v4 integrator.  No test-after-tune: the
sealed list is never used for training or model selection.

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
    """Mean paired ratio + 95% bootstrap CI (percentile, 2000 resamples)."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(candidate), size=(2000, len(candidate)))
    ratios = np.mean(candidate[indices], axis=1) / np.maximum(np.mean(baseline[indices], axis=1), 1e-12)
    return float(np.mean(ratios)), float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


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
                "fallback_failed": int(float(metrics.get("fallback_fraction", 0.0)) > 0.10 and metrics["stable"] < 0.5),
                "integrator": INTEGRATOR_LABEL,
            })
    # Summary with paired CIs vs IMC (same sealed list, same seeds).
    imc_objectives = np.asarray([r["objective"] for r in details if r["algorithm"] == "imc"], dtype=float)
    summary: list[dict[str, object]] = []
    for algo in ALGORITHMS:
        objectives = np.asarray([r["objective"] for r in details if r["algorithm"] == algo], dtype=float)
        mean_ratio, lo95, hi95 = _paired_ratio_ci(objectives, imc_objectives, seed=args.seed + 77)
        ratios = objectives / np.maximum(imc_objectives, 1e-12)
        subset = [r for r in details if r["algorithm"] == algo]
        summary.append({
            "algorithm": algo,
            "scenarios": len(subset),
            "mean_objective": float(np.mean(objectives)),
            "std_objective": float(np.std(objectives, ddof=1)) if len(objectives) > 1 else 0.0,
            "mean_ratio_to_imc": float(np.mean(ratios)),
            "paired_ratio_mean": mean_ratio,
            "paired_ratio_lo95": lo95,
            "paired_ratio_hi95": hi95,
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
        mean_ratio, lo95, hi95 = _paired_ratio_ci(objectives_arr, imc_objectives, seed=args.seed + 101 + idx)
        ablation_rows.append({
            "method": name, "kp": gains[0], "ki": gains[1],
            "mean_objective": float(np.mean(objectives_arr)),
            "paired_ratio_mean": mean_ratio, "paired_ratio_lo95": lo95, "paired_ratio_hi95": hi95,
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
        "code_sha": git_head_sha(),
        "manifest_sha256": manifest.get("_manifest_sha256", manifest_sha256()),
        "manifest_path": str(manifest.get("_manifest_path", "experiments/manifests/v4.yaml")),
        "artifact_dir": str(artifact_dir.resolve()),
        "policy_provenance": bundle.provenance,
        "policies": {"zn": list(zn_gains), "imc": list(imc_gains), "bo": list(bo_gains),
                     "safe-bo": list(safe_gains), "llm": list(llm_gains), "llm_source": llm_source},
        "safe_bo_attribution": safe_bo_attribution,
        "stat_note": "mean_ratio_to_imc is ratio-of-means; paired_ratio_* is mean-of-paired-ratios with bootstrap CI; report both, do not conflate",
        "sealed": {"repeats": args.repeats, "per_repeat": args.per_repeat, "total": len(sealed),
                   "acceptance_seeds": list(acceptance_seeds), "pipeline_seed": args.seed},
        "integrator": INTEGRATOR_LABEL,
        "elapsed_seconds": time.perf_counter() - started,
        "comparability": "all 7 on same 80 scenarios/seeds/metrics; LLM is replay-frozen, not live; do not rank across protocols",
    }
    (output / "sealed_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"sealed 80x7: {len(details)} detail rows, {len(summary)} summary rows -> {output.resolve()}")
    for row in summary:
        print(f"  {row['algorithm']:7s} mean={float(row['mean_objective']):.3f} "
              f"paired_ratio={float(row['paired_ratio_mean']):.4f} [{float(row['paired_ratio_lo95']):.4f},{float(row['paired_ratio_hi95']):.4f}] "
              f"stable={float(row['stable_rate']):.2f}")


if __name__ == "__main__":
    main()
