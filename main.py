from __future__ import annotations

import argparse
from pathlib import Path

from hvac_pid.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HVAC 3R2C simulation: fixed/ZN/IMC PI versus AI gain scheduling"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="result directory")
    parser.add_argument("--train-samples", type=int, default=48, help="offline labeled contexts")
    parser.add_argument("--validation-samples", type=int, default=16, help="model-selection contexts")
    parser.add_argument("--test-samples", type=int, default=16, help="held-out contexts")
    parser.add_argument(
        "--acceptance-seeds",
        type=str,
        default="101,211,307,401,503",
        help="comma-separated sealed test seeds",
    )
    parser.add_argument("--bo-iterations", type=int, default=7, help="Bayesian optimization steps per label")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small smoke demo (20 train, 6 test, 3 BO iterations)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.train_samples = 20
        args.validation_samples = 6
        args.test_samples = 6
        args.bo_iterations = 3
        args.acceptance_seeds = "101"
    acceptance_seeds = tuple(int(value.strip()) for value in args.acceptance_seeds.split(",") if value.strip())
    if not acceptance_seeds:
        raise ValueError("--acceptance-seeds must contain at least one integer")
    result = run_pipeline(
        args.output,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        bo_iterations=args.bo_iterations,
        seed=args.seed,
        acceptance_seeds=acceptance_seeds,
    )
    print(f"Results: {result['output_dir']}")


if __name__ == "__main__":
    main()
