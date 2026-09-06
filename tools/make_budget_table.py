"""Counted compute budget ledger (NOT a complete measured cost).

Every closed-loop simulation costs one 5 h object run.  Counts are derived
from the frozen histories inside --run (no re-simulation).  Deliberately
NOT covered (see omissions below): short-horizon replays, repeated
evaluations inside training loops beyond the counted rounds, integration
substeps, plotting/report rendering, and wall-clock time except an optional
pipeline total passed via --wall-seconds.  Do NOT use this table for exact
cost ranking between methods with different training flows.

Counted rows:
  * imc-lambda: rows(imc_lambda_tuning.csv) x train scenarios
  * bo: rows(bayesian_search_history.csv) x train scenarios (= evals)
  * safe-bo: rows(safe_bo_history.csv) x train scenarios
  * fnn-labels: sum(training_labels.csv:bo_evaluations) + 2 x train (ZN/IMC
    reference sims per scenario, see tuning.generate_label_rows)
  * fnn-select: one (baseline, learned) validation round (finite
    validation_learned_objective rows)
  * rl-train: environment steps = sum(rl_training_history.csv:environment_steps);
    falls back to rows(rl_training_transitions.csv) only if present.
    Steps are NOT full 5 h sims and stay in their own row.
  * rl-select: one validation round per non-NaN checkpoint + final
  * deploy-gate: 3 sims (IMC + 2 candidates) per qualification scenario
  * sealed-final: 7 algorithms x 80 scenarios (record-only)

Only the BO/random arms share an identical budget; every other pair of
methods must NOT be called a "same-budget comparison".

Usage: python tools/make_budget_table.py --run <run> --sealed <sealed> [--wall-seconds N]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOURS_PER_SIM = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training-budget ledger")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=float("nan"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    run, sealed = args.run, args.sealed
    train_n = len(_rows(run / "training_scenarios.csv"))
    val_n = len(_rows(run / "validation_scenarios.csv"))
    try:
        qual_n = len(_rows(run / "qualification_scenarios.csv"))
    except FileNotFoundError:
        qual_n = 0
    imc_evals = len(_rows(run / "imc_lambda_tuning.csv"))
    bo_evals = len(_rows(run / "global_bayesian_tuning.csv")) and len(_rows(run / "bayesian_search_history.csv"))
    safe_evals = len(_rows(run / "safe_bo_history.csv"))
    labels = _rows(run / "training_labels.csv")
    label_sims = int(sum(float(r.get("bo_evaluations", 0.0)) for r in labels)) + 2 * train_n
    fnn_hist = _rows(run / "fnn_training_history.csv")
    rl_hist = _rows(run / "rl_training_history.csv")

    def _finite(rows: list[dict[str, str]], key: str) -> int:
        n = 0
        for r in rows:
            try:
                v = float(r.get(key, "nan"))
            except (TypeError, ValueError):
                continue
            if v == v:  # not NaN
                n += 1
        return n

    def _fsum(rows: list[dict[str, str]], key: str) -> float:
        total = 0.0
        for r in rows:
            try:
                v = float(r.get(key, "nan"))
            except (TypeError, ValueError):
                continue
            if v == v:
                total += v
        return total

    # Selection cost = validation full sims actually run: FNN one
    # (baseline, learned) round; RL one round per non-NaN checkpoint + final.
    fnn_select_sims = 2 * val_n if _finite(fnn_hist, "validation_learned_objective") else 0
    rl_select_sims = (_finite(rl_hist, "checkpoint_validation_objective") + 1) * val_n
    # RL environment steps survive pruning via the history file (transitions
    # CSV is pruned from the repo; never emit -1).
    rl_steps_hist = _fsum(rl_hist, "environment_steps")
    try:
        rl_steps = len(_rows(run / "rl_training_transitions.csv"))
        rl_steps_note = "rows(rl_training_transitions.csv)"
    except FileNotFoundError:
        rl_steps = int(rl_steps_hist)
        rl_steps_note = "sum(rl_training_history.csv:environment_steps); transitions CSV pruned"
    gate_rows = _rows(run / "deployment_acceptance.csv")
    # deployment gate: per qualification scenario, IMC baseline + FNN + RL sims.
    gate_sims = qual_n * 3 if qual_n else 0
    legend = [
        {"method": "zn-formula", "train_scenarios": 1, "candidate_evals": 0,
         "closed_loop_sims": 0, "note": "commissioning FOPDT only; no search"},
        {"method": "imc-lambda", "train_scenarios": train_n, "candidate_evals": imc_evals,
         "closed_loop_sims": imc_evals * train_n, "note": "lambda scan on training split"},
        {"method": "bo", "train_scenarios": train_n, "candidate_evals": bo_evals,
         "closed_loop_sims": bo_evals * train_n, "note": "7 init + 7 GP-EI; same 14-eval budget as E3+ random-14 only"},
        {"method": "safe-bo", "train_scenarios": train_n, "candidate_evals": safe_evals,
         "closed_loop_sims": safe_evals * train_n, "note": "risk-constrained local search"},
        {"method": "fnn-labels", "train_scenarios": train_n, "candidate_evals": int(sum(float(r.get('bo_evaluations', 0.0)) for r in labels)),
         "closed_loop_sims": label_sims, "note": "per-scenario BO labels + ZN/IMC references"},
        {"method": "fnn-select", "train_scenarios": val_n, "candidate_evals": 2,
         "closed_loop_sims": fnn_select_sims, "note": "one (baseline, learned) validation round"},
        {"method": "rl-train", "train_scenarios": train_n, "candidate_evals": rl_steps,
         "closed_loop_sims": 0, "note": f"offline environment STEPS, not full sims ({rl_steps_note})"},
        {"method": "rl-select", "train_scenarios": val_n, "candidate_evals": rl_select_sims // max(val_n, 1),
         "closed_loop_sims": rl_select_sims, "note": "one validation round per checkpoint + final"},
        {"method": "deploy-gate", "train_scenarios": qual_n, "candidate_evals": 3,
         "closed_loop_sims": qual_n * 3 if qual_n else 0,
         "note": "qualification split only (IMC+FNN+RL per scenario); sealed test never decides"},
        {"method": "sealed-final", "train_scenarios": 80, "candidate_evals": 7,
         "closed_loop_sims": 560, "note": "record-only final evaluation, 7 algorithms x 80 scenarios"},
    ]
    for row in legend:
        row["object_hours"] = round(row["closed_loop_sims"] * HOURS_PER_SIM, 1)
    out = args.output or (sealed / "budget_table.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(legend[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(legend)
    meta = {"title": "counted compute budget (NOT a complete measured cost)",
            "run": str(run.resolve()), "sealed": str(sealed.resolve()),
            "hours_per_sim": HOURS_PER_SIM, "pipeline_wall_seconds": args.wall_seconds,
            "omissions": ["short-horizon replays", "repeated in-loop evaluations beyond counted rounds",
                          "integration substeps", "plotting/report rendering",
                          "per-method wall time (only pipeline total, if given)"],
            "warning": "Only BO/random arms share an identical budget."}
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"budget table: {len(legend)} rows -> {out.resolve()}")
    for row in legend:
        print(f"  {row['method']:16s} sims={row['closed_loop_sims']:>6} obj-h={row['object_hours']:>9} :: {row['note']}")


if __name__ == "__main__":
    main()
