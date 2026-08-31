"""Export validated Python FNN/RL artifacts to a deterministic MCU header."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import struct
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
    # The manifest is executed as float32 on the MCU. Quantise first, then emit
    # enough decimal digits to round-trip that exact bit pattern; otherwise a
    # float64 value near a rounding boundary can produce a header one ULP away
    # from the bytes covered by Python's CRC.
    text = f"{float(np.float32(value)):.9g}"
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


def _array_2d(values: np.ndarray) -> str:
    return "{" + ",".join("{" + ",".join(_fmt(item) for item in row) + "}" for row in values) + "}"


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


def _accepted(row: dict[str, str]) -> bool:
    try:
        value = float(row.get("deployment_accepted", "nan"))
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and value > 0.5)


def _manifest_crc(
    *,
    fnn_accepted: bool,
    rl_accepted: bool,
    fallback: np.ndarray,
    fnn: np.ndarray,
    fnn_context: np.ndarray,
    policy: np.ndarray,
    covered: np.ndarray,
) -> int:
    payload = bytearray(b"HVACPID3")
    payload.extend(struct.pack("<I", 3))
    payload.extend(bytes((int(fnn_accepted), int(rl_accepted))))
    arrays = (
        np.asarray(fallback, dtype="<f4"),
        np.asarray([0.002, 1.5, 1e-5, 0.08, 0.10], dtype="<f4"),
        np.asarray(FNNGainController.centers, dtype="<f4"),
        np.asarray(FNNGainController.delta_centers, dtype="<f4"),
        np.asarray(fnn, dtype="<f4"),
        np.asarray(fnn_context, dtype="<f4"),
        np.asarray(IncrementalRLController.error_edges, dtype="<f4"),
        np.asarray(IncrementalRLController.delta_edges, dtype="<f4"),
        np.asarray(IncrementalRLController.command_edges, dtype="<f4"),
        np.asarray(IncrementalRLController._actions, dtype="<f4"),
    )
    for values in arrays:
        payload.extend(values.tobytes(order="C"))
    payload.extend(np.asarray(policy, dtype=np.uint8).tobytes(order="C"))
    payload.extend(np.asarray(covered, dtype=np.uint8).tobytes(order="C"))
    return zlib.crc32(payload) & 0xFFFFFFFF


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
    context_path = output / "fnn_context_coefficients.npy"
    fnn_context = np.load(context_path) if context_path.exists() else np.zeros((2, 4), dtype=float)
    q_table = np.load(output / "rl_q_table.npy")
    if fnn.shape != (5, 5, 2):
        raise ValueError(f"unexpected FNN shape: {fnn.shape}")
    if fnn_context.shape != (2, 4):
        raise ValueError(f"unexpected FNN context shape: {fnn_context.shape}")
    if q_table.shape != (5, 5, 3, 9):
        raise ValueError(f"unexpected RL shape: {q_table.shape}")

    fnn_accepted = _accepted(fnn_history)
    rl_accepted = _accepted(rl_history)
    policy = _policy(q_table)
    covered = _covered_states(output / "rl_training_transitions.csv")
    crc = _manifest_crc(
        fnn_accepted=fnn_accepted,
        rl_accepted=rl_accepted,
        fallback=fallback,
        fnn=fnn,
        fnn_context=fnn_context,
        policy=policy,
        covered=covered,
    )

    action_rows = "{" + ",".join("{" + _fmt(kp) + "," + _fmt(ki) + "}" for kp, ki in IncrementalRLController._actions) + "}"
    content = f"""#pragma once

#include <cstdint>

namespace hvac_mcu {{ namespace generated {{

static constexpr uint32_t kArtifactVersion = 3u;
static constexpr uint32_t kArtifactCrc32 = 0x{crc:08X}u;
static constexpr bool kFnnAccepted = {'true' if fnn_accepted else 'false'};
static constexpr bool kRlAccepted = {'true' if rl_accepted else 'false'};
static constexpr float kFallbackKp = {_fmt(fallback[0])};
static constexpr float kFallbackKi = {_fmt(fallback[1])};
static constexpr float kMinimumKp = 0.002f;
static constexpr float kMaximumKp = 1.5f;
static constexpr float kMinimumKi = 0.00001f;
static constexpr float kMaximumKi = 0.08f;
static constexpr float kMaximumGainChangeFraction = 0.10f;
static const float kFnnErrorCenters[5] = {_array_1d(FNNGainController.centers)};
static const float kFnnErrorRateCenters[5] = {_array_1d(FNNGainController.delta_centers)};
static const float kFnnRuleTable[5][5][2] = {_array_fnn(fnn)};
static const float kFnnContextCoefficients[2][4] = {_array_2d(fnn_context)};
static const float kRlErrorEdges[4] = {_array_1d(IncrementalRLController.error_edges)};
static const float kRlErrorRateEdges[4] = {_array_1d(IncrementalRLController.delta_edges)};
static const float kRlCommandEdges[2] = {_array_1d(IncrementalRLController.command_edges)};
static const float kRlTargetScales[9][2] = {action_rows};
static const uint8_t kRlPolicy[5][5][3] = {_array_u8_3d(policy)};
static const uint8_t kRlCovered[5][5][3] = {_array_u8_3d(covered)};

}} }}  // namespace hvac_mcu::generated
"""
    header.write_text(content, encoding="utf-8", newline="\n")
    manifest_path = output / "policy_manifest_v3.json"
    manifest = {
        "magic": "HVACPID3",
        "artifact_version": 3,
        "crc32": f"0x{crc:08X}",
        "fnn_accepted": fnn_accepted,
        "rl_accepted": rl_accepted,
        "factory_fallback_note": "CRC/version failure uses separately compiled factory IMC, not these serialized values",
        "validated_fallback_gains": {"kp": float(np.float32(fallback[0])), "ki": float(np.float32(fallback[1]))},
        "gain_bounds": {"kp": [0.002, 1.5], "ki": [1e-5, 0.08], "maximum_change_fraction": 0.10},
        "fnn": {
            "error_centers": np.asarray(FNNGainController.centers, dtype=np.float32).tolist(),
            "error_rate_centers": np.asarray(FNNGainController.delta_centers, dtype=np.float32).tolist(),
            "rule_table": np.asarray(fnn, dtype=np.float32).tolist(),
            "context_coefficients": np.asarray(fnn_context, dtype=np.float32).tolist(),
        },
        "rl": {
            "error_edges": np.asarray(IncrementalRLController.error_edges, dtype=np.float32).tolist(),
            "error_rate_edges": np.asarray(IncrementalRLController.delta_edges, dtype=np.float32).tolist(),
            "command_edges": np.asarray(IncrementalRLController.command_edges, dtype=np.float32).tolist(),
            "target_scales": np.asarray(IncrementalRLController._actions, dtype=np.float32).tolist(),
            "policy": policy.tolist(),
            "covered_mask": covered.tolist(),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    parity_path = output / "policy_parity_vectors.csv"
    with parity_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "error_c", "error_rate_c_per_min", "applied_command", "integral_state",
            "outdoor_delta_c", "load_fraction", "fnn_kp", "fnn_ki",
            "rl_error_bin", "rl_error_rate_bin", "rl_command_bin", "rl_action",
            "rl_covered", "rl_kp", "rl_ki",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        fnn_controller = FNNGainController(tuple(fallback), rule_table=fnn, context_coefficients=fnn_context)
        for error in FNNGainController.centers:
            for error_rate in FNNGainController.delta_centers:
                for command in (0.0, 0.35, 0.90):
                    integral_state = float(error * 5.0)
                    outdoor_delta_c = 10.0
                    load_fraction = 0.20
                    fnn_gains = fnn_controller._propose(
                        float(error), float(error_rate), applied_command=float(command),
                        integral_state=integral_state, outdoor_delta_c=outdoor_delta_c,
                        load_fraction=load_fraction,
                    )
                    e_bin = IncrementalRLController._bin(float(error), IncrementalRLController.error_edges)
                    d_bin = IncrementalRLController._bin(float(error_rate), IncrementalRLController.delta_edges)
                    c_bin = IncrementalRLController._bin(float(command), IncrementalRLController.command_edges)
                    is_covered = bool(rl_accepted and covered[e_bin, d_bin, c_bin])
                    action = int(policy[e_bin, d_bin, c_bin]) if is_covered else 4
                    writer.writerow(
                        {
                            "error_c": float(error),
                            "error_rate_c_per_min": float(error_rate),
                            "applied_command": float(command),
                            "integral_state": integral_state,
                            "outdoor_delta_c": outdoor_delta_c,
                            "load_fraction": load_fraction,
                            "fnn_kp": fnn_gains[0],
                            "fnn_ki": fnn_gains[1],
                            "rl_error_bin": e_bin,
                            "rl_error_rate_bin": d_bin,
                            "rl_command_bin": c_bin,
                            "rl_action": action,
                            "rl_covered": int(is_covered),
                            "rl_kp": float(fallback[0] * IncrementalRLController._actions[action, 0]),
                            "rl_ki": float(fallback[1] * IncrementalRLController._actions[action, 1]),
                        }
                    )
    print(f"exported {header}")
    print(f"manifest={manifest_path}")
    print(f"parity vectors={parity_path}")
    print(f"FNN accepted={fnn_accepted}; RL accepted={rl_accepted}; CRC32=0x{crc:08X}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
