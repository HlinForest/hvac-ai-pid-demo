"""Unified document builder (dedup P4).

Thin dispatcher over the two existing builders; no logic moved here::

    python build_docs.py detailed --formal        # 01 全链路技术报告
    python build_docs.py detailed --beginner      # 02 零基础解释版
    python build_docs.py manual                   # 零基础学习手册

``build_detailed_docs.py`` / ``build_learning_manual.py`` remain in place.
Requires optional deps: ``python -m pip install python-docx pillow``.
"""

from __future__ import annotations

import argparse
import sys


def _cmd_detailed(args: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Build 01/02 detailed docs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--formal", action="store_true", help="01 全链路提交报告")
    group.add_argument("--beginner", action="store_true", help="02 零基础解释版")
    parsed = parser.parse_args(args)
    import build_detailed_docs

    print(build_detailed_docs.build(bool(parsed.formal)))


def _cmd_manual(args: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Build the beginner learning manual")
    parser.parse_args(args)
    import build_learning_manual

    print(build_learning_manual.build())


_COMMANDS = {"detailed": _cmd_detailed, "manual": _cmd_manual}


def main() -> None:
    # Manual dispatch so subcommand flags (e.g. --help) reach the subcommand
    # parser instead of being consumed by the top-level parser.
    if len(sys.argv) >= 2 and sys.argv[1] in _COMMANDS:
        _COMMANDS[sys.argv[1]](sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(_COMMANDS))
    parser.parse_args()


if __name__ == "__main__":
    main()
