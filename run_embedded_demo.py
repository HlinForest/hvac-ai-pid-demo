from __future__ import annotations

import argparse
import json
from pathlib import Path

from hvac_pid.embedded_demo import (
    ALGORITHM_ORDER,
    demo_scenario,
    run_algorithm_demo,
    run_all_demos,
    write_demo_bundle,
    write_interactive_html,
    write_trace_csv,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ESP32 seven-algorithm closed-loop temperature demonstration"
    )
    parser.add_argument("--algorithm", choices=(*ALGORITHM_ORDER, "all"), default="all")
    parser.add_argument("--provider", choices=("replay", "ollama", "openai"), default="replay")
    parser.add_argument("--model", default="", help="required for live OpenAI; optional Ollama override")
    parser.add_argument("--setpoint", type=float, default=24.0)
    parser.add_argument("--door-load", type=float, default=3200.0)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", type=Path, default=Path("outputs_embedded_demo"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = demo_scenario(setpoint_c=args.setpoint, door_load_w=args.door_load)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.algorithm == "all":
        traces = run_all_demos(
            project_root=ROOT,
            scenario=scenario,
            provider=args.provider,
            model=args.model,
            seed=args.seed,
        )
        files = write_demo_bundle(output, traces, project_root=ROOT)
        print(json.dumps(files, ensure_ascii=False, indent=2))
        return

    trace = run_algorithm_demo(
        args.algorithm,
        project_root=ROOT,
        scenario=scenario,
        provider=args.provider,
        model=args.model,
        seed=args.seed,
    )
    stem = args.algorithm.replace("-", "_") + "_temperature_demo"
    csv_path = output / f"{stem}.csv"
    html_path = output / f"{stem}.html"
    summary_path = output / f"{stem}_summary.json"
    write_trace_csv(csv_path, trace)
    write_interactive_html(html_path, {args.algorithm: trace}, title=f"{trace.display_name}温度闭环 Demo")
    summary_path.write_text(json.dumps(trace.summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"html": str(html_path), "csv": str(csv_path), "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
