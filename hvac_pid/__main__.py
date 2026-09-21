"""python -m hvac_pid --help"""
import argparse
from dataclasses import replace
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hands-on AI PI tuning experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    from experiments.methods import ONLINE
    for name in ["temperature", "plant", "classical", "bo", "fnn", *ONLINE, "pg4pi", "pg4pi-reference", "llm", "compare", "all", "repeat", "site", "explain", "benchmark"]:
        sub = commands.add_parser(name)
        sub.add_argument("--output", type=Path, default=Path("outputs/tutorial"))
        if name in {"bo", "fnn", *ONLINE, "pg4pi", "pg4pi-reference", "llm", "compare", "all", "repeat", "explain"}:
            sub.add_argument("--seed", type=int, default=0)
        if name in {"bo", "fnn", "llm", "compare", "all", "repeat"}:
            sub.add_argument("--rounds", type=int, default=15)
        if name in {*ONLINE, "pg4pi", "compare", "all", "repeat"}:
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
            sub.add_argument("--without-torch", "--without-dqn", dest="without_dqn", action="store_true",
                             help="Run NumPy methods only; omit all neural RL methods")
        if name == "explain":
            sub.add_argument("--method", choices=["classical", "bo", "fnn", *ONLINE, "pg4pi", "llm", "all"], default="all")
        if name == "benchmark":
            sub.add_argument("--samples", type=int, default=1000)
            sub.add_argument("--without-torch", dest="without_dqn", action="store_true")
        if name == "pg4pi-reference":
            sub.set_defaults(output=Path("outputs/pg4pi-reference"))
            sub.add_argument("--install-deps", action="store_true")
            sub.add_argument("--timeout", type=int, default=300)
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
        if command in {"all", "compare", "repeat", "benchmark"}:
            options["include_dqn"] = not options.pop("without_dqn")
        if command in ONLINE:
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
        elif command == "explain":
            from experiments.explain import explain
            explain(**options)
        elif command == "benchmark":
            from experiments.benchmark import benchmark
            benchmark(**options)
        elif command == "pg4pi-reference":
            from experiments.pg4pi_reference import run as run_reference
            report = run_reference(**options)
            print("Upstream reference: " + report["status"])
            if report["status"] != "complete":
                return 2
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
