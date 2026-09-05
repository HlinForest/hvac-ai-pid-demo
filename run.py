"""Unified experiment entry point (dedup P4).

Thin dispatcher over the existing per-task modules; no logic moved here.
Prefer this over calling the individual ``run_*.py`` scripts directly::

    python run.py pipeline --quick
    python run.py benchmark --help
    python run.py crossval artifacts/runs/<run_id>
    python run.py embedded --algorithm all --provider replay
    python run.py llm-agent --provider replay
    python run.py advanced --quick --llm-provider heuristic
    python run.py matrix outputs_llm_matrix_v3

Each subcommand forwards its remaining arguments verbatim to the
corresponding module's ``main()``.
"""

from __future__ import annotations

import argparse
import importlib
import sys

_SUBCOMMANDS: dict[str, str] = {
    "pipeline": "main",
    "benchmark": "run_tuning_benchmark",
    "crossval": "run_cross_validation",
    "embedded": "run_embedded_demo",
    "llm-agent": "run_llm_agent_demo",
    "advanced": "run_advanced_tuning_benchmark",
    "matrix": "aggregate_llm_matrix",
}


def _delegate(module_name: str, argv: list[str]) -> None:
    sys.argv = [f"{module_name}.py", *argv]
    module = importlib.import_module(module_name)
    module.main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified HVAC AI-PI experiment CLI (delegates to run_* modules)",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    helps = {
        "pipeline": "full 48/16/16 pipeline (main.py)",
        "benchmark": "PI tuning timing benchmark",
        "crossval": "numerical/OpenModelica cross-validation",
        "embedded": "ESP32 seven-algorithm temperature demo",
        "llm-agent": "tool-using LLM agent demo",
        "advanced": "risk-aware safe BO + LLM supervisory benchmark",
        "matrix": "aggregate an LLM matrix directory",
    }
    for command in _SUBCOMMANDS:
        sub.add_parser(command, help=helps[command]).add_argument(
            "args", nargs=argparse.REMAINDER, help="arguments forwarded verbatim",
        )
    return parser


def main() -> None:
    # Bypass argparse once a valid subcommand is present so that flags such
    # as --help are forwarded verbatim to the target module instead of being
    # consumed by this dispatcher.
    if len(sys.argv) >= 2 and sys.argv[1] in _SUBCOMMANDS:
        rest = sys.argv[2:]
        if sys.argv[1] == "matrix" and (not rest or rest == ["--help"] or rest == ["-h"]):
            print("usage: run.py matrix <matrix_dir>  (default: outputs_llm_matrix_v3)")
            return
        _delegate(_SUBCOMMANDS[sys.argv[1]], rest)
        return
    build_parser().parse_args()


if __name__ == "__main__":
    main()
