"""Unified experiment entry point (dedup P4).

Thin dispatcher over the existing per-task modules; no logic moved here.
Prefer this over calling the individual ``run_*.py`` scripts directly::

    python run.py pipeline --quick
    python run.py benchmark --help
    python run.py crossval artifacts/runs/<run_id>
    python run.py embedded --algorithm all --provider replay
    python run.py sealed --artifact-dir archive/outputs_review_v3 --output artifacts/runs/sealed-80x7
    python run.py llm-agent --provider replay
    python run.py advanced --quick --llm-provider heuristic
    python run.py matrix archive/outputs_llm_matrix_v3
    python run.py render outputs

Each subcommand forwards its remaining arguments verbatim to the
corresponding module's ``main()``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

_SUBCOMMANDS: dict[str, str] = {
    "pipeline": "main",
    "benchmark": "tools.run_tuning_benchmark",
    "crossval": "tools.run_cross_validation",
    "embedded": "tools.run_embedded_demo",
    "sealed": "tools.run_sealed_evaluation",
    "llm-agent": "tools.run_llm_agent_demo",
    "advanced": "tools.run_advanced_tuning_benchmark",
    "matrix": "tools.aggregate_llm_matrix",
    "render": "tools.render_report",
    "attribution": "tools.run_gp_attribution",
    "host-wcet": "tools.run_host_wcet",
}


def _delegate(module_name: str, argv: list[str]) -> None:
    sys.argv = [f"{module_name}.py", *argv]
    if "." not in module_name:
        # Top-level entry (main.py): import by bare name.
        module = importlib.import_module(module_name)
    else:
        # tools/ entries: ensure the project root (parent of tools/) is
        # importable, then load as a submodule (namespace package, no
        # tools/__init__.py required).
        root = str(Path(__file__).resolve().parent)
        if root not in sys.path:
            sys.path.insert(0, root)
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
        "sealed": "seven-algorithm 80-scenario sealed evaluation (P1)",
        "llm-agent": "tool-using LLM agent demo",
        "advanced": "risk-aware safe BO + LLM supervisory benchmark",
        "matrix": "aggregate an LLM matrix directory",
        "render": "render an outputs dir into the HTML engineering report",
        "attribution": "same-budget GP attribution on sealed list (E3+)",
        "host-wcet": "host-only WCET reference (NOT target evidence)",
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
            print("usage: run.py matrix <matrix_dir>  (default: archive/outputs_llm_matrix_v3)")
            return
        _delegate(_SUBCOMMANDS[sys.argv[1]], rest)
        return
    build_parser().parse_args()


if __name__ == "__main__":
    main()
