from __future__ import annotations

"""Run the tool-using LLM Agent PI auto-tuning temperature demonstration."""

import argparse
import json
from pathlib import Path

from hvac_pid.embedded_demo import (
    demo_scenario,
    run_algorithm_demo,
    write_agent_trace_csv,
    write_interactive_html,
    write_trace_csv,
)
from hvac_pid.env import load_project_env


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tool-using LLM Agent HVAC PI auto-tuning demo")
    parser.add_argument(
        "--provider",
        choices=("replay", "ollama", "openai", "openai-compatible"),
        default="replay",
    )
    parser.add_argument("--model", default="", help="required for OpenAI/openai-compatible; optional Ollama override")
    parser.add_argument(
        "--base-url",
        default="",
        help="openai-compatible endpoint override; default is Alibaba Bailian (DashScope) compatible mode",
    )
    parser.add_argument(
        "--api-key-env",
        default="",
        help="environment variable holding the API key for openai-compatible; default DASHSCOPE_API_KEY",
    )
    parser.add_argument("--setpoint", type=float, default=24.0)
    parser.add_argument("--door-load", type=float, default=3200.0)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", type=Path, default=Path("outputs_llm_agent"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_project_env(ROOT)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace = run_algorithm_demo(
        "llm",
        project_root=ROOT,
        scenario=demo_scenario(setpoint_c=args.setpoint, door_load_w=args.door_load),
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        seed=args.seed,
    )
    html_path = output / "llm_agent_temperature_demo.html"
    temperature_path = output / "llm_agent_temperature.csv"
    audit_path = output / "llm_agent_trace.csv"
    summary_path = output / "llm_agent_summary.json"
    write_interactive_html(html_path, {"llm": trace}, title="LLM Agent PI 自动整定温度闭环 Demo")
    write_trace_csv(temperature_path, trace)
    write_agent_trace_csv(audit_path, trace)
    summary_path.write_text(json.dumps(trace.summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "interactive_html": str(html_path),
        "temperature_csv": str(temperature_path),
        "agent_trace_csv": str(audit_path),
        "summary_json": str(summary_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import warnings
    warnings.warn(
        "Direct entry run_llm_agent_demo.py is a thin wrapper; prefer 'python run.py llm-agent ...'",
        DeprecationWarning, stacklevel=2)
    main()
