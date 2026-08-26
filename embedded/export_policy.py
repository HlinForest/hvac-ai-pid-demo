"""Export validated Python FNN/RL artifacts to a deterministic MCU header."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hvac_pid.ai_controllers import FNNGainController, IncrementalRLController


def _last_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows[-1]


def _fmt(value: float) -> str:
    text = f"{float(value):.9g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text + "f"


def _array_1d(values: np.ndarray) -> str:
    return "{" + ",".join(_fmt(value) for value in values) + "}"


def _array_fnn(table: np.ndarray) -> str:
    rows = []
    for e_index in range(5):
        cells = ["{" + _fmt(table[e_index, d_index, 0]) + "," + _fmt(table[e_index, d_index, 1]) + "}" for d_index in range(5)]
        rows.append("{" + ",".join(cells) + "}")
    return "{" + ",\n    ".join(rows) + "}"


def _covered_states(path: Path) -> np.ndarray:
    covered = np.zeros((5, 5, 3), dtype=np.uint8)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            covered[
                int(float(row["state_error_bin"])),
                int(float(row["state_delta_bin"])),
                int(float(row["state_command_bin"])),
            ] = 1
    return covered


def _policy(q_table: np.ndarray) -> np.ndarray:
    policy = np.full((5, 5, 3), 4, dtype=np.uint8)
    for e_index in range(5):
        allowed = IncrementalRLController._allowed_actions(e_index)
        for d_index in range(5):
            for command_index in range(3):
                values = q_table[e_index, d_index, command_index, allowed]
                policy[e_index, d_index, command_index] = int(allowed[int(np.argmax(values))])
    return policy


def _array_u8_3d(values: np.ndarray) -> str:
    planes = []
    for e_index in range(values.shape[0]):
        rows = ["{" + ",".join(str(int(value)) for value in values[e_index, d_index]) + "}" for d_index in range(values.shape[1])]
        planes.append("{" + ",".join(rows) + "}")
    return "{" + ",\n    ".join(planes) + "}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--header", type=Path, default=Path(__file__).with_name("generated_policy.hpp"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    header = args.header.resolve()

    fnn_history = _last_csv(output / "fnn_training_history.csv")
    rl_history = _last_csv(output / "rl_training_history.csv")
    imc_rows: list[dict[str, str]]
    with (output / "imc_lambda_tuning.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        imc_rows = list(csv.DictReader(handle))
    selected_imc = next(row for row in imc_rows if int(float(row["selected"])) == 1)
    fallback = np.asarray([float(selected_imc["candidate_kp"]), float(selected_imc["candidate_ki"])])

    fnn = np.load(output / "fnn_rule_table.npy")
    q_table = np.load(output / "rl_q_table.npy")
    if fnn.shape != (5, 5, 2):
        raise ValueError(f"unexpected FNN shape: {fnn.shape}")
    if q_table.shape != (5, 5, 3, 9):
        raise ValueError(f"unexpected RL shape: {q_table.shape}")

    fnn_accepted = float(fnn_history["deployment_accepted"]) > 0.5
    rl_accepted = float(rl_history["deployment_accepted"]) > 0.5
    policy = _policy(q_table)
    covered = _covered_states(output / "rl_training_transitions.csv")
    crc = zlib.crc32(fnn.astype("<f4").tobytes())
    crc = zlib.crc32(q_table.astype("<f4").tobytes(), crc) & 0xFFFFFFFF

    action_rows = "{" + ",".join("{" + _fmt(kp) + "," + _fmt(ki) + "}" for kp, ki in IncrementalRLController._actions) + "}"
    content = f"""#pragma once

#include <cstdint>

namespace hvac_mcu {{ namespace generated {{

static constexpr uint32_t kArtifactVersion = 2u;
static constexpr uint32_t kArtifactCrc32 = 0x{crc:08X}u;
static constexpr bool kFnnAccepted = {'true' if fnn_accepted else 'false'};
static constexpr bool kRlAccepted = {'true' if rl_accepted else 'false'};
static constexpr float kFallbackKp = {_fmt(fallback[0])};
static constexpr float kFallbackKi = {_fmt(fallback[1])};
static const float kFnnErrorCenters[5] = {_array_1d(FNNGainController.centers)};
static const float kFnnErrorRateCenters[5] = {_array_1d(FNNGainController.delta_centers)};
static const float kFnnRuleTable[5][5][2] = {_array_fnn(fnn)};
static const float kRlErrorEdges[4] = {_array_1d(IncrementalRLController.error_edges)};
static const float kRlErrorRateEdges[4] = {_array_1d(IncrementalRLController.delta_edges)};
static const float kRlCommandEdges[2] = {_array_1d(IncrementalRLController.command_edges)};
static const float kRlTargetScales[9][2] = {action_rows};
static const uint8_t kRlPolicy[5][5][3] = {_array_u8_3d(policy)};
static const uint8_t kRlCovered[5][5][3] = {_array_u8_3d(covered)};

}} }}  // namespace hvac_mcu::generated
"""
    header.write_text(content, encoding="utf-8", newline="\n")
    print(f"exported {header}")
    print(f"FNN accepted={fnn_accepted}; RL accepted={rl_accepted}; CRC32=0x{crc:08X}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
