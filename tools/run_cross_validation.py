"""Refresh numerical/OpenModelica cross-validation artifacts without retraining AI controllers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from hvac_pid.validation import run_cross_validation


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and export model cross-validation")
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("outputs"))
    args = parser.parse_args()
    output = args.output_dir
    validation = run_cross_validation(output)
    mappings = {
        "physical_cross_validation.csv": "physical_rows",
        "physical_cross_validation_timeseries.csv": "physical_timeseries",
        "fopdt_cross_validation.csv": "fopdt_rows",
        "fopdt_cross_validation_timeseries.csv": "fopdt_timeseries",
        "openmodelica_cross_validation.csv": "openmodelica_rows",
        "openmodelica_cross_validation_timeseries.csv": "openmodelica_timeseries",
    }
    for filename, key in mappings.items():
        write_rows(output / filename, validation[key])
    write_rows(
        output / "cross_validation_environment.csv",
        [{"item": key, "value": value} for key, value in validation["environment"].items()],
    )
    if validation["openmodelica_rows"]:
        row = validation["openmodelica_rows"][0]
        print(
            "OpenModelica cross-validation passed="
            f"{bool(row['passed'])}, RMSE={float(row['rmse_c']):.8f} °C, "
            f"max={float(row['max_abs_error_c']):.8f} °C"
        )
    else:
        print("OpenModelica result CSV not found; numerical validations were refreshed only.")


if __name__ == "__main__":
    import warnings
    warnings.warn(
        "Direct entry run_cross_validation.py is a thin wrapper; prefer 'python run.py crossval ...'",
        DeprecationWarning, stacklevel=2)
    main()
