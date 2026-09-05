from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.aggregate_llm_matrix import _load_run  # noqa: E402


def _write_run(run_dir: Path, *, accepted: int, live: bool) -> None:
    run_dir.mkdir(parents=True)
    summary = {
        "provider": "openai-compatible",
        "source": "OpenAI-compatible chat tool agent/qwen-test" if live else "recorded tool-using LLM agent replay (not a live model call)",
        "llm_replay_disclosure": 0 if live else 1,
        "llm_agent_steps": 5,
        "llm_agent_trials": 2,
        "llm_accepted_trials": accepted,
        "deployment_accepted": accepted,
        "fallback_used": 1 - accepted,
        "objective": 45.0,
        "max_undershoot_c": 1.0,
        "stable": 1,
        "first_in_band_minute": 34.0,
        "door_recovery_minutes": 41.0,
        "llm_baseline_kp": 0.45,
        "llm_baseline_ki": 0.003,
        "llm_deployed_kp": 0.44 if accepted else 0.45,
        "llm_deployed_ki": 0.0028 if accepted else 0.003,
        "llm_described_trial": "accepted" if accepted else "last_rejected",
    }
    (run_dir / "llm_agent_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (run_dir / "llm_agent_trace.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "tool", "decision"])
        writer.writeheader()
        writer.writerow({"step": 1, "tool": "inspect_history", "decision": "ok"})
        writer.writerow({"step": 2, "tool": "evaluate_candidate", "decision": "accepted" if accepted else "rejected"})


def test_load_run_extracts_model_difficulty_and_live_flag(tmp_path: Path) -> None:
    run_dir = tmp_path / "qwen-test_std_r71"
    _write_run(run_dir, accepted=1, live=True)
    row = _load_run(run_dir)
    assert row is not None
    assert row["model"] == "qwen-test"
    assert row["difficulty"] == "std"
    assert row["seed"] == "71"
    assert row["live_llm"] == 1
    assert row["inspect_calls"] == 1
    assert row["deployment_accepted"] == 1
    assert row["kp_change_pct"] < 0.0


def test_load_run_marks_replay_as_not_live(tmp_path: Path) -> None:
    run_dir = tmp_path / "qwen-test_hard_r72"
    _write_run(run_dir, accepted=0, live=False)
    row = _load_run(run_dir)
    assert row is not None
    assert row["live_llm"] == 0
    assert row["fallback_used"] == 1


def test_load_run_returns_none_without_summary(tmp_path: Path) -> None:
    empty = tmp_path / "qwen-test_std_r73"
    empty.mkdir()
    assert _load_run(empty) is None
