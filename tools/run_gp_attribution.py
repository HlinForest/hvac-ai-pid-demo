"""Strict GP-vs-random attribution (E3+, post-1904e09 protocol).

For each search seed, all arms share the SAME 7 init points
(1 IMC + 6 random from ``rng(search_seed)``), the SAME training split and
the SAME scoring noise (``search_seed + index``, which for the canonical
seed 815 reproduces the pipeline BO exactly):

  * gp-full: 7 init + 7 GP-EI steps via ``tune_global_fixed`` (14 evals)
  * rand-14: same 7 init + 7 further i.i.d. draws continuing the same RNG
    stream (14 evals) -> the ONLY difference vs gp-full is how the last 7
    points were chosen
  * init-only-7: best of the shared 7 init points, zero further evals
  * imc: frozen tuned baseline (reference)

Each arm's winner (by training mean) is scored on the SAME sealed 80 with
the E2 primary statistic.  Reported per search seed plus pooled paired
differences (gp-minus-random, gp-minus-init) with bootstrap 95% CIs.

Causal wording licensed by this output: arms differ ONLY in the 7 appended
points, so a pooled gp-minus-random CI below 0 supports a GP contribution
*under this budget*; anything else stays an observation, not a mechanism.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hvac_pid.config import sample_adaptive_scenarios
from hvac_pid.controllers import PIController, identify_fopdt, imc_pi
from hvac_pid.manifest import git_head_sha, load_manifest, manifest_sha256
from hvac_pid.metrics import calculate_metrics
from hvac_pid.policy_bundle import PolicyBundle
from hvac_pid.simulator import simulate
from hvac_pid.tuning import BayesianGainTuner, tune_global_fixed

CANONICAL_SEED = 815  # pipeline: seed(7) + 808


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict GP-vs-random attribution (E3+)")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/runs/sealed-80x7-v4-retrain/gp_attribution.csv"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--train-samples", type=int, default=48)
    parser.add_argument("--bo-iterations", type=int, default=7)
    parser.add_argument("--search-seeds", type=str, default="815,5001,9002,12077")
    parser.add_argument("--acceptance-seeds", type=str, default="101,211,307,401,503")
    return parser.parse_args()


def _paired_ci(candidate: np.ndarray, baseline: np.ndarray, seed: int) -> tuple[float, float, float]:
    ratios = candidate / np.maximum(baseline, 1e-12)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ratios), size=(2000, len(ratios)))
    boot = np.mean(ratios[idx], axis=1)
    return float(np.mean(ratios)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _diff_ci(diffs: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(2000, len(diffs)))
    boot = np.mean(diffs[idx], axis=1)
    return float(np.mean(diffs)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    args = parse_args()
    search_seeds = [int(v.strip()) for v in args.search_seeds.split(",") if v.strip()]
    acceptance_seeds = tuple(int(v.strip()) for v in args.acceptance_seeds.split(",") if v.strip())
    manifest = load_manifest()
    bundle = PolicyBundle.load(args.artifact_dir, project_root=ROOT)
    tuner = BayesianGainTuner()
    started = time.perf_counter()

    training = sample_adaptive_scenarios(args.train_samples, seed=args.train_seed, duration_hours=5.0)
    imc_init = imc_pi(identify_fopdt(training[0]))
    sealed = []
    for acc in acceptance_seeds:
        sealed.extend(sample_adaptive_scenarios(16, seed=args.seed + 100_003 + acc, duration_hours=5.0))
    assert len(sealed) == 80

    def train_score(gains: tuple[float, float], noise_base: int) -> float:
        return float(np.mean([
            calculate_metrics(simulate(s, PIController(*gains), seed=noise_base + i))["objective"]
            for i, s in enumerate(training)]))

    def sealed_obj(gains: tuple[float, float]) -> np.ndarray:
        return np.asarray([
            float(calculate_metrics(simulate(s, PIController(*gains),
                                             seed=args.seed + 700_000 + k))["objective"])
            for k, s in enumerate(sealed)])

    imc_sealed = sealed_obj(tuple(bundle.imc_gains))
    rows: list[dict[str, object]] = []
    pooled_gr: list[float] = []
    pooled_gi: list[float] = []
    for search_seed in search_seeds:
        rng = np.random.default_rng(search_seed)
        init_points = [tuner._encode(imc_init)] + [p for p in rng.random((6, 2))]
        extra_points = [p for p in rng.random((args.bo_iterations, 2))]
        init_gains = [tuner._decode(p) for p in init_points]
        init_scores = [train_score(g, search_seed) for g in init_gains]
        init_best = init_gains[int(np.argmin(init_scores))]
        # GP arm: exact pipeline procedure on the same split.
        gp_tune = tune_global_fixed(training, iterations=args.bo_iterations, seed=search_seed)
        gp_best = (float(gp_tune.kp), float(gp_tune.ki))
        if search_seed == CANONICAL_SEED:
            frozen = tuple(bundle.bo_gains)
            assert abs(gp_best[0] - frozen[0]) < 1e-9 and abs(gp_best[1] - frozen[1]) < 1e-9, \
                f"canonical reproduction drift: {gp_best} vs frozen {frozen}"
            print(f"canonical seed {search_seed}: reproduced frozen BO {gp_best}")
        # Random arm: SHARED init + 7 continuing draws, identical scoring.
        extra_gains = [tuner._decode(p) for p in extra_points]
        extra_scores = [train_score(g, search_seed) for g in extra_gains]
        pool_gains = init_gains + extra_gains
        pool_scores = init_scores + extra_scores
        rand_best = pool_gains[int(np.argmin(pool_scores))]
        arms = {"init-only-7": init_best, "gp-full": gp_best, "random-14": rand_best}
        sealed_arms = {name: sealed_obj(g) for name, g in arms.items()}
        for name, gains in arms.items():
            mean, lo, hi = _paired_ci(sealed_arms[name], imc_sealed, seed=args.seed + 77)
            rows.append({
                "search_seed": search_seed, "method": name,
                "kp": gains[0], "ki": gains[1],
                "train_evals": 7 if name == "init-only-7" else 14,
                "train_mean_objective": float(np.min(init_scores)) if name == "init-only-7" else (
                    float(gp_tune.score) if name == "gp-full" else float(np.min(pool_scores))),
                "sealed_mean_objective": float(np.mean(sealed_arms[name])),
                "paired_ratio_mean": mean, "paired_ratio_lo95": lo, "paired_ratio_hi95": hi,
            })
        pooled_gr.extend((sealed_arms["gp-full"] / np.maximum(imc_sealed, 1e-12)
                          - sealed_arms["random-14"] / np.maximum(imc_sealed, 1e-12)).tolist())
        pooled_gi.extend((sealed_arms["gp-full"] / np.maximum(imc_sealed, 1e-12)
                          - sealed_arms["init-only-7"] / np.maximum(imc_sealed, 1e-12)).tolist())

    diff_gr = _diff_ci(np.asarray(pooled_gr), args.seed + 913)
    diff_gi = _diff_ci(np.asarray(pooled_gi), args.seed + 914)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment_id": "E3+", "code_sha": git_head_sha(),
        "manifest_sha256": manifest.get("_manifest_sha256", manifest_sha256()),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "search_seeds": search_seeds, "n_pooled_pairs": len(pooled_gr),
        "pooled_gp_minus_random": {"mean": diff_gr[0], "lo95": diff_gr[1], "hi95": diff_gr[2]},
        "pooled_gp_minus_init": {"mean": diff_gi[0], "lo95": diff_gi[1], "hi95": diff_gi[2]},
        "reading": ("GP contribution supported under this budget (pooled gp-random CI below 0)"
                    if diff_gr[2] < 0 else
                    "no pooled GP advantage over same-budget random; mechanism contribution unverified"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.with_name("gp_attribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"gp strict attribution: {len(rows)} rows -> {args.output.resolve()}")
    for r in rows:
        print(f"  seed={r['search_seed']} {str(r['method']):12s} train={float(r['train_mean_objective']):.2f} "
              f"sealed={float(r['sealed_mean_objective']):.2f} paired={float(r['paired_ratio_mean']):.4f}")
    print(f"pooled gp-random diff: {diff_gr[0]:+.4f} [{diff_gr[1]:+.4f},{diff_gr[2]:+.4f}] (n={len(pooled_gr)})")
    print(f"pooled gp-init   diff: {diff_gi[0]:+.4f} [{diff_gi[1]:+.4f},{diff_gi[2]:+.4f}] (n={len(pooled_gi)})")


if __name__ == "__main__":
    main()
