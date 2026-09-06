"""GP mechanism attribution under the same training budget (E3+).

Compares, on the SAME training split and the SAME sealed list:
  * bo-full: frozen BO gains from --artifact-dir (7 init + 7 GP = 14 evals)
  * init-only-7: best of the identical 7 init points (1 IMC + 6 random,
    seed+808, no GP steps) -> isolates the GP search contribution
  * random-14: best of 14 i.i.d. log-uniform random gains (same 14-eval
    budget, no model) -> isolates search-vs-luck
  * conservative-imc-formula: IMC formula default, no search at all
  * imc: frozen tuned baseline (reference, ratio = 1)

Training selection uses mean objective over --train-samples scenarios
(seed=train-seed, noise train-seed+index, identical to pipeline
tune_global_fixed).  Sealed evaluation reuses the E2 convention
(acceptance seeds, noise seed+700000+ordinal, six-substep integrator).

Output: gp_attribution.csv with sealed mean objective, paired
mean-of-ratios + bootstrap 95% CI vs IMC, and training-selection scores.
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
from hvac_pid.tuning import BayesianGainTuner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Same-budget GP attribution (E3+)")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/runs/sealed-80x7-v4-retrain/gp_attribution.csv"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-seed", type=int, default=7,
                        help="training split seed (pipeline uses --seed)")
    parser.add_argument("--train-samples", type=int, default=48)
    parser.add_argument("--bo-eval-seed", type=int, default=7 + 808,
                        help="pipeline tune_global_fixed seed (seed+808)")
    parser.add_argument("--random-seed", type=int, default=7 + 5500)
    parser.add_argument("--acceptance-seeds", type=str, default="101,211,307,401,503")
    return parser.parse_args()


def _paired_ci(candidate: np.ndarray, baseline: np.ndarray, seed: int) -> tuple[float, float, float]:
    ratios = candidate / np.maximum(baseline, 1e-12)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ratios), size=(2000, len(ratios)))
    boot = np.mean(ratios[idx], axis=1)
    return float(np.mean(ratios)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    args = parse_args()
    acceptance_seeds = tuple(int(v.strip()) for v in args.acceptance_seeds.split(",") if v.strip())
    manifest = load_manifest()
    bundle = PolicyBundle.load(args.artifact_dir, project_root=ROOT)
    tuner = BayesianGainTuner()
    started = time.perf_counter()

    training = sample_adaptive_scenarios(args.train_samples, seed=args.train_seed, duration_hours=5.0)

    def train_score(gains: tuple[float, float]) -> float:
        scores = [calculate_metrics(simulate(s, PIController(*gains), seed=args.train_seed + i))["objective"]
                  for i, s in enumerate(training)]
        return float(np.mean(scores))

    # Exact pipeline init (1 IMC + 6 random from bo-eval seed).
    imc_init = imc_pi(identify_fopdt(training[0]))
    rng_init = np.random.default_rng(args.bo_eval_seed)
    init_points = [tuner._encode(imc_init)] + [p for p in rng_init.random((6, 2))]
    init_gains = [tuner._decode(p) for p in init_points]
    init_scores = [train_score(g) for g in init_gains]
    init_best = init_gains[int(np.argmin(init_scores))]

    # Same-budget pure random search: 14 i.i.d. draws, best on training.
    rng = np.random.default_rng(args.random_seed)
    random_gains = [tuner._decode(p) for p in rng.random((14, 2))]
    random_scores = [train_score(g) for g in random_gains]
    random_best = random_gains[int(np.argmin(random_scores))]

    conservative = imc_pi(identify_fopdt(training[0])) if False else None
    # Conservative formula default: replicate controllers.imc_pi default path
    # (closed_loop_time=None -> max(tau/3, 3L, 12)).  Call without scan.
    from hvac_pid.controllers import imc_pi as _imc
    import inspect as _inspect
    if "closed_loop_time" in _inspect.signature(_imc).parameters:
        conservative_gains = _imc(identify_fopdt(training[0]), closed_loop_time=None)
    else:  # pragma: no cover - signature guard
        conservative_gains = _imc(identify_fopdt(training[0]))
    conservative_gains = (float(conservative_gains[0]), float(conservative_gains[1]))

    contenders: dict[str, tuple[float, float]] = {
        "imc": tuple(bundle.imc_gains),
        "bo-full": tuple(bundle.bo_gains),
        "init-only-7": (float(init_best[0]), float(init_best[1])),
        "random-14": (float(random_best[0]), float(random_best[1])),
        "conservative-imc-formula": conservative_gains,
    }
    train_lookup = {
        "imc": train_score(tuple(bundle.imc_gains)),
        "bo-full": train_score(tuple(bundle.bo_gains)),
        "init-only-7": float(np.min(init_scores)),
        "random-14": float(np.min(random_scores)),
        "conservative-imc-formula": train_score(conservative_gains),
    }

    sealed: list = []
    for acc in acceptance_seeds:
        sealed.extend(sample_adaptive_scenarios(16, seed=args.seed + 100_003 + acc, duration_hours=5.0))
    assert len(sealed) == 80
    sealed_obj: dict[str, np.ndarray] = {}
    for name, gains in contenders.items():
        vals = [float(calculate_metrics(simulate(s, PIController(*gains),
                        seed=args.seed + 700_000 + k))["objective"]) for k, s in enumerate(sealed)]
        sealed_obj[name] = np.asarray(vals)
    imc_sealed = sealed_obj["imc"]

    rows: list[dict[str, object]] = []
    for name, gains in contenders.items():
        mean, lo, hi = _paired_ci(sealed_obj[name], imc_sealed, seed=args.seed + 77)
        rows.append({
            "method": name,
            "kp": gains[0], "ki": gains[1],
            "train_evals": 14 if name in ("bo-full", "random-14") else (7 if name == "init-only-7" else 0),
            "train_mean_objective": train_lookup[name],
            "sealed_mean_objective": float(np.mean(sealed_obj[name])),
            "paired_ratio_mean": mean, "paired_ratio_lo95": lo, "paired_ratio_hi95": hi,
            "note": {
                "imc": "frozen tuned baseline",
                "bo-full": "frozen BO (7 init + 7 GP-EI); same 14-eval budget as random-14",
                "init-only-7": "best of identical 7 init points, zero GP steps; BO-full minus init-only = GP contribution",
                "random-14": "best of 14 i.i.d. log-uniform draws on same training split",
                "conservative-imc-formula": "formula default, no search",
            }[name],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "experiment_id": "E3+", "code_sha": git_head_sha(),
        "manifest_sha256": manifest.get("_manifest_sha256", manifest_sha256()),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "bo_gains": list(bundle.bo_gains), "imc_gains": list(bundle.imc_gains),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    print(f"gp attribution: {len(rows)} rows -> {args.output.resolve()}")
    for r in rows:
        print(f"  {str(r['method']):24s} train={float(r['train_mean_objective']):.2f} "
              f"sealed={float(r['sealed_mean_objective']):.2f} "
              f"paired={float(r['paired_ratio_mean']):.4f} "
              f"[{float(r['paired_ratio_lo95']):.4f},{float(r['paired_ratio_hi95']):.4f}]")


if __name__ == "__main__":
    main()
