"""Host-only WCET reference (NOT target evidence).

Measures Python control-step wall time on THIS host (PC) for PI / FNN /
RL dispatch, plus parses the PC testbench worst-case timers from
mcu_pc_sil_log.txt (also host timings).  Target MCU WCET (ESP32/STM32
on-target measurement) remains pending external validation and MUST NOT
be filled in from these numbers.

Output: <run>/host_wcet.csv + <run>/host_wcet.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Host-only WCET reference")
    parser.add_argument("--artifact-dir", type=Path, required=True,
                        help="frozen policies, e.g. artifacts/runs/v4-20260907-8f3fd5f")
    parser.add_argument("--output", type=Path, default=None,
                        help="output run dir (default: <artifact-dir>)")
    parser.add_argument("--repeats", type=int, default=2000)
    return parser.parse_args()


def _bench(fn, repeats: int) -> dict[str, float]:
    fn()  # warmup
    samples = []
    for _ in range(repeats):
        tick = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - tick)
    arr = np.asarray(samples, dtype=float) / 1000.0  # us
    return {"mean_us": float(np.mean(arr)), "p50_us": float(np.median(arr)),
            "p99_us": float(np.quantile(arr, 0.99)), "max_us": float(np.max(arr)),
            "n": int(repeats)}


def main() -> None:
    args = parse_args()
    from hvac_pid.config import Scenario
    from hvac_pid.controllers import PIController
    from hvac_pid.policy_bundle import PolicyBundle
    from hvac_pid.simulator import simulate

    bundle = PolicyBundle.load(args.artifact_dir, project_root=ROOT)
    out = args.output or args.artifact_dir
    out.mkdir(parents=True, exist_ok=True)
    scenario = Scenario()
    controllers = {
        "pi": PIController(*bundle.imc_gains),
    }
    try:
        from hvac_pid.ai_controllers import FNNGainController, IncrementalRLController
        controllers["fnn"] = FNNGainController(tuple(bundle.imc_gains),
                                               rule_table=np.asarray(bundle.fnn_rule_table),
                                               context_coefficients=np.asarray(bundle.fnn_context))
        controllers["rl"] = IncrementalRLController(
            tuple(bundle.imc_gains), q_table=np.asarray(bundle.rl_q_table),
            covered_mask=np.asarray(bundle.rl_covered_mask))
    except Exception as exc:  # pragma: no cover - bundle must be loadable
        raise RuntimeError(f"cannot build AI controllers: {exc}") from exc

    rows: list[dict[str, object]] = []
    for name, controller in controllers.items():
        controller.reset()
        stats = _bench(lambda c=controller: c.update(0.5, 1.0, outdoor_c=34.0,
                                                     internal_load_w=900.0,
                                                     setpoint_c=24.0), args.repeats)
        rows.append({"scope": "HOST-ONLY (PC Python, NOT target MCU)", "controller": name, **stats,
                     "verdict": "reference only"})
    # Parse PC testbench worst timers (host binary, not target silicon).
    log_path = Path(args.artifact_dir) / "mcu_pc_sil_log.txt"
    pc_pi = pc_ai = ""
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        pc_pi = re.search(r"worst PI=(\S+)", log)
        pc_ai = re.search(r"worst AI=(\S+)", log)
        rows.append({"scope": "HOST-ONLY (PC testbench.exe, NOT target MCU)",
                     "controller": "c++_safe_pi",
                     "mean_us": "", "p50_us": "", "p99_us": "",
                     "max_us": pc_pi.group(1) if pc_pi else "see log",
                     "n": "", "verdict": "reference only"})
        rows.append({"scope": "HOST-ONLY (PC testbench.exe, NOT target MCU)",
                     "controller": "c++_ai_dispatch",
                     "mean_us": "", "p50_us": "", "p99_us": "",
                     "max_us": pc_ai.group(1) if pc_ai else "see log",
                     "n": "", "verdict": "reference only"})
    with (out / "host_wcet.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    meta = {"scope": "HOST-ONLY reference; target MCU WCET pending external validation",
            "artifact_dir": str(args.artifact_dir.resolve()),
            "note": "Do not copy these numbers into target WCET gates."}
    (out / "host_wcet.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    print(f"host WCET reference: {len(rows)} rows -> {(out / 'host_wcet.csv').resolve()}")
    for r in rows:
        print(f"  {r['controller']:16s} max={r['max_us']} ({r['scope'][:24]}...)")


if __name__ == "__main__":
    main()
