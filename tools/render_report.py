"""Render an existing outputs directory into the visual HTML engineering report.

Usage: python run.py render outputs  (or: python tools/render_report.py outputs)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from hvac_pid.report import write_engineering_report, write_html_engineering_report
from hvac_pid.algorithm_reports import REPORT_FILES


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    converted: list[dict[str, object]] = []
    for row in rows:
        parsed: dict[str, object] = {}
        for key, value in row.items():
            if value in ("", None):
                parsed[key] = value
                continue
            try:
                parsed[key] = float(value)
            except ValueError:
                parsed[key] = value
        converted.append(parsed)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Create engineering_report.html from existing CSV and PNG artifacts")
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("outputs"))
    args = parser.parse_args()
    output = args.output_dir
    case_metrics = output / "case_metrics.csv"
    report_dir = output / "algorithm_reports"
    algorithm_reports = {
        name: f"algorithm_reports/{filename}"
        for name, filename in REPORT_FILES.items()
        if (report_dir / filename).exists()
    }
    environment_path = output / "cross_validation_environment.csv"
    validation_environment = {
        str(row["item"]): row["value"] for row in read_rows(environment_path)
    } if environment_path.exists() else {}
    physical_validation_path = output / "physical_cross_validation.csv"
    fopdt_validation_path = output / "fopdt_cross_validation.csv"
    openmodelica_validation_path = output / "openmodelica_cross_validation.csv"
    report_arguments = (
        read_rows(output / "holdout_summary.csv"),
        read_rows(output / "dynamic_metrics.csv"),
        read_rows(case_metrics) if case_metrics.exists() else [],
        algorithm_reports,
        read_rows(physical_validation_path) if physical_validation_path.exists() else [],
        read_rows(fopdt_validation_path) if fopdt_validation_path.exists() else [],
        read_rows(openmodelica_validation_path) if openmodelica_validation_path.exists() else [],
        validation_environment,
    )
    write_engineering_report(output / "engineering_report.md", *report_arguments)
    write_html_engineering_report(
        output / "engineering_report.html",
        report_arguments[0],
        report_arguments[1],
        output,
        *report_arguments[2:],
    )
    print((output / "engineering_report.html").resolve())


if __name__ == "__main__":
    main()
