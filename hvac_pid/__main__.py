"""python -m hvac_pid --help"""
import argparse
from dataclasses import replace
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hands-on AI PI tuning experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["temperature", "plant", "classical", "bo", "fnn", "qlearning", "dqn", "llm", "compare", "all", "repeat", "site"]:
        sub = commands.add_parser(name)
        sub.add_argument("--output", type=Path, default=Path("outputs/tutorial"))
        if name in {"bo", "fnn", "qlearning", "dqn", "llm", "compare", "all", "repeat"}:
            sub.add_argument("--seed", type=int, default=0)
        if name in {"bo", "fnn", "llm", "compare", "all", "repeat"}:
            sub.add_argument("--rounds", type=int, default=15)
        if name in {"qlearning", "dqn", "all", "repeat"}:
            sub.add_argument("--episodes", type=int, default=500)
        if name in {"temperature", "plant", "classical", "bo"}:
            sub.add_argument("--gain", type=float, default=8.0)
            sub.add_argument("--tau", type=float, default=20.0)
            sub.add_argument("--delay", type=float, default=2.0)
        if name == "temperature":
            sub.add_argument("--kp", type=float, default=0.15)
            sub.add_argument("--ki", type=float, default=0.005)
        if name == "llm":
            sub.add_argument("--live", action="store_true", help="Explicitly enable billable model calls")
            sub.add_argument("--model")
            sub.add_argument("--base-url")
            sub.add_argument("--key-env", default="DASHSCOPE_API_KEY")
            sub.add_argument("--reasoning-effort", help="Optional compatible API setting, e.g. none for DeepSeek named tool calls")
        if name in {"all", "compare", "repeat"}:
            sub.add_argument("--without-dqn", action="store_true")
        if name == "repeat":
            sub.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        if name == "site":
            sub.add_argument("--destination", type=Path, default=Path("site"))
    args = parser.parse_args(argv)
    from threadpoolctl import threadpool_limits
    from experiments import run
    from .core import Scenario
    with threadpool_limits(limits=1):
        options = vars(args).copy()
        command = options.pop("command")
        if command in {"temperature", "plant", "classical", "bo"}:
            options["scenario"] = replace(Scenario(), **{k: options.pop(k) for k in ("gain", "tau", "delay")})
        if command in {"all", "compare", "repeat"}:
            options["include_dqn"] = not options.pop("without_dqn")
        if command in {"qlearning", "dqn"}:
            run.rl_experiment(method=command, **options)
        elif command == "all":
            run.all_experiments(**options)
        elif command == "repeat":
            from experiments.site import summarize_repeats
            seeds = options.pop("seeds")
            root = options.pop("output")
            options.pop("seed")
            for seed in seeds:
                run.all_experiments(output=root / f"seed-{seed}", seed=seed, **options)
            summarize_repeats(root, seeds)
        elif command == "site":
            from experiments.site import export
            export(**options)
        else:
            result = getattr(run, command)(**options)
            if command == "llm":
                print(result["status"] + ": " + result["stop_reason"])
                if args.live and result["status"] != "complete":
                    return 2
    print(f"Saved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
