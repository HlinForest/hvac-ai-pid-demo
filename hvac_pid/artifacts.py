"""Small, explicit result files. No report prose is generated here."""
from dataclasses import asdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_trace(path: Path, trace):
    columns = {key: value for key, value in asdict(trace).items() if isinstance(value, np.ndarray)}
    write_csv(path, [dict(zip(columns, values)) for values in zip(*columns.values())])


def metadata():
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "matplotlib", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    root = Path(__file__).resolve().parents[1]
    source = hashlib.sha256()
    for file in sorted([*root.glob("hvac_pid/*.py"), *root.glob("experiments/*.py"), root / "experiments/splits.json"]):
        source.update(file.relative_to(root).as_posix().encode())
        source.update(file.read_bytes())
    return {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
            "packages": packages, "commit": commit.stdout.strip(), "dirty": bool(dirty.stdout),
            "source_sha256": source.hexdigest()}
